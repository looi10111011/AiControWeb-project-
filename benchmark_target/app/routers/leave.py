import calendar as pycalendar
from datetime import date, datetime, timedelta, timezone

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from benchmark_target.app.calendar_util import count_working_days
from benchmark_target.app.config import TEMPLATES_DIR
from benchmark_target.app.db import get_conn, log_audit
from benchmark_target.app.deps import can, get_current_user, get_scope
from benchmark_target.app.faults import check_and_consume_fault

router = APIRouter(prefix="/leave")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

_CAL_DAY_PRIORITY = {"pending": 0, "approved": 1, "rejected": 2, "cancelled": 3}


def _trend(now: int, past: int) -> dict:
    """Same shape/reasoning as pim.py's _trend(): a real now-vs-30-days-ago delta, never a
    fabricated one. `past` here is "created 30+ days ago and currently status X" — the
    closest honest analogue to pim's hire_date cutoff, since leave_requests has no column
    timestamping when a status itself last changed (only updated_at, which any edit bumps)."""
    if past == 0:
        return {"pct": 100, "direction": "up"} if now > 0 else {"pct": 0, "direction": "flat"}
    pct = round((now - past) / past * 100)
    direction = "up" if pct > 0 else "down" if pct < 0 else "flat"
    return {"pct": abs(pct), "direction": direction}


def _forbidden(request: Request, user: dict):
    return templates.TemplateResponse(request, "403.html", {"user": user}, status_code=403)


def _leave_request_in_scope(user: dict, row, scope: str) -> bool:
    """row: a leave_requests row, optionally joined with employees.supervisor_id."""
    if scope == "all":
        return True
    if scope == "own":
        return row["employee_id"] == user.get("employee_id")
    if scope == "direct_reports":
        return row["supervisor_id"] == user.get("employee_id")
    return False


def _available_balance(conn, employee_id: int, leave_type_id: int) -> int:
    balance = conn.execute(
        "SELECT balance_days FROM leave_balances WHERE employee_id = ? AND leave_type_id = ?",
        (employee_id, leave_type_id),
    ).fetchone()["balance_days"]
    reserved = conn.execute(
        """SELECT COALESCE(SUM(days), 0) d FROM leave_requests
           WHERE employee_id = ? AND leave_type_id = ? AND status IN ('pending', 'approved')""",
        (employee_id, leave_type_id),
    ).fetchone()["d"]
    return balance - reserved


@router.get("/requests")
def list_requests(
    request: Request,
    q: str = "",
    status: str = "",
    leave_type_id: str = "",
    date_from: str = "",
    date_to: str = "",
    cal: str = "",
):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)

    scope = get_scope(user["role"], "leave", "read")
    if scope is None:
        return _forbidden(request, user)

    scope_where = ["1=1"]
    scope_params: list = []
    if scope == "own":
        scope_where.append("lr.employee_id = ?")
        scope_params.append(user.get("employee_id"))
    elif scope == "direct_reports":
        scope_where.append("e.supervisor_id = ?")
        scope_params.append(user.get("employee_id"))
    # scope == "all": no extra restriction
    scope_where_sql = " AND ".join(scope_where)

    where = list(scope_where)
    params = list(scope_params)
    if q:
        where.append("(e.first_name LIKE ? OR e.last_name LIKE ?)")
        like = f"%{q}%"
        params.extend([like, like])
    if status:
        where.append("lr.status = ?")
        params.append(status)
    if leave_type_id:
        where.append("lr.leave_type_id = ?")
        params.append(leave_type_id)
    if date_from:
        where.append("lr.end_date >= ?")
        params.append(date_from)
    if date_to:
        where.append("lr.start_date <= ?")
        params.append(date_to)
    where_sql = " AND ".join(where)

    # Calendar month shown in the sidebar widget — independent of the filters above (it's
    # a browse-by-date view, not tied to the current search), defaults to this month, and
    # is navigable via ?cal=YYYY-MM so a real seed fixture in a past month is reachable.
    try:
        cal_year, cal_month = (int(part) for part in cal.split("-", 1))
        cal_first = date(cal_year, cal_month, 1)
    except (ValueError, TypeError):
        cal_first = date.today().replace(day=1)
        cal_year, cal_month = cal_first.year, cal_first.month
    cal_last = date(cal_year, cal_month, pycalendar.monthrange(cal_year, cal_month)[1])
    prev_month_end = cal_first - timedelta(days=1)
    next_month_start = cal_last + timedelta(days=1)

    with get_conn() as conn:
        rows = [
            dict(row, initial=row["first_name"][:1].upper())
            for row in conn.execute(
                f"""SELECT lr.*, e.first_name, e.last_name, e.department, lt.name AS leave_type_name
                    FROM leave_requests lr
                    JOIN employees e ON e.id = lr.employee_id
                    JOIN leave_types lt ON lt.id = lr.leave_type_id
                    WHERE {where_sql}
                    ORDER BY lr.created_at DESC""",
                params,
            ).fetchall()
        ]
        leave_types = conn.execute("SELECT * FROM leave_types ORDER BY id").fetchall()
        balances = []
        annual_entitlement = None
        if user.get("employee_id"):
            balances = conn.execute(
                """SELECT lt.name, lb.balance_days FROM leave_balances lb
                   JOIN leave_types lt ON lt.id = lb.leave_type_id
                   WHERE lb.employee_id = ? ORDER BY lt.id""",
                (user["employee_id"],),
            ).fetchall()
            annual_entitlement = conn.execute(
                "SELECT leave_balance_annual FROM employees WHERE id = ?", (user["employee_id"],)
            ).fetchone()["leave_balance_annual"]

        # Stat row: scope-restricted (a supervisor sees counts for their own team, not the
        # whole org) but independent of the status/type/date filters above — a summary of
        # what this view covers, not of the current filtered result.
        def _count(extra_sql: str = "", extra_params: tuple = ()) -> int:
            return conn.execute(
                f"""SELECT COUNT(*) c FROM leave_requests lr
                    JOIN employees e ON e.id = lr.employee_id
                    WHERE {scope_where_sql} {extra_sql}""",
                [*scope_params, *extra_params],
            ).fetchone()["c"]

        def _sparkline(extra_sql: str = "", extra_params: tuple = (), days: int = 7) -> list[int]:
            start = date.today() - timedelta(days=days - 1)
            found = {
                r["d"]: r["c"]
                for r in conn.execute(
                    f"""SELECT substr(lr.created_at, 1, 10) d, COUNT(*) c FROM leave_requests lr
                        JOIN employees e ON e.id = lr.employee_id
                        WHERE {scope_where_sql} {extra_sql} AND lr.created_at >= ?
                        GROUP BY d""",
                    [*scope_params, *extra_params, start.isoformat()],
                ).fetchall()
            }
            return [found.get((start + timedelta(days=i)).isoformat(), 0) for i in range(days)]

        thirty_days_ago = (date.today() - timedelta(days=30)).isoformat()
        stat_total_now = _count()
        stat_total_past = _count("AND lr.created_at <= ?", (thirty_days_ago,))
        stat_approved_now = _count("AND lr.status = 'approved'")
        stat_approved_past = _count("AND lr.status = 'approved' AND lr.created_at <= ?", (thirty_days_ago,))
        stat_pending_now = _count("AND lr.status = 'pending'")
        stat_pending_past = _count("AND lr.status = 'pending' AND lr.created_at <= ?", (thirty_days_ago,))
        stat_rejected_now = _count("AND lr.status = 'rejected'")
        stat_rejected_past = _count("AND lr.status = 'rejected' AND lr.created_at <= ?", (thirty_days_ago,))

        stat_total = {"value": stat_total_now, "spark": _sparkline(), **_trend(stat_total_now, stat_total_past)}
        stat_approved = {
            "value": stat_approved_now, "spark": _sparkline("AND lr.status = 'approved'"),
            **_trend(stat_approved_now, stat_approved_past),
        }
        stat_pending = {
            "value": stat_pending_now, "spark": _sparkline("AND lr.status = 'pending'"),
            **_trend(stat_pending_now, stat_pending_past),
        }
        stat_rejected = {
            "value": stat_rejected_now, "spark": _sparkline("AND lr.status = 'rejected'"),
            **_trend(stat_rejected_now, stat_rejected_past),
        }

        # Calendar dot per day: the highest-priority status among any scope-visible request
        # overlapping that day (pending needs attention first, so it wins the tie).
        cal_rows = conn.execute(
            f"""SELECT lr.start_date, lr.end_date, lr.status FROM leave_requests lr
                JOIN employees e ON e.id = lr.employee_id
                WHERE {scope_where_sql} AND NOT (lr.end_date < ? OR lr.start_date > ?)""",
            [*scope_params, cal_first.isoformat(), cal_last.isoformat()],
        ).fetchall()
        day_status: dict[int, str] = {}
        for cr in cal_rows:
            cursor = max(date.fromisoformat(cr["start_date"]), cal_first)
            stop = min(date.fromisoformat(cr["end_date"]), cal_last)
            while cursor <= stop:
                current = day_status.get(cursor.day)
                if current is None or _CAL_DAY_PRIORITY[cr["status"]] < _CAL_DAY_PRIORITY[current]:
                    day_status[cursor.day] = cr["status"]
                cursor += timedelta(days=1)

    cal_weeks = pycalendar.Calendar(firstweekday=6).monthdayscalendar(cal_year, cal_month)
    cal_grid = [
        [{"day": d, "status": day_status.get(d)} if d else None for d in week]
        for week in cal_weeks
    ]

    can_approve = can(user["role"], "leave", "approve")
    can_create = can(user["role"], "leave", "create")

    return templates.TemplateResponse(
        request,
        "leave_list.html",
        {
            "user": user,
            "requests": rows,
            "q": q,
            "status": status,
            "leave_type_id": leave_type_id,
            "date_from": date_from,
            "date_to": date_to,
            "leave_types": leave_types,
            "balances": balances,
            "annual_entitlement": annual_entitlement,
            "can_approve": can_approve,
            "can_create": can_create,
            "stat_total": stat_total,
            "stat_approved": stat_approved,
            "stat_pending": stat_pending,
            "stat_rejected": stat_rejected,
            "cal_grid": cal_grid,
            "cal_label": cal_first.strftime("%B %Y"),
            "cal_prev": f"{prev_month_end.year:04d}-{prev_month_end.month:02d}",
            "cal_next": f"{next_month_start.year:04d}-{next_month_start.month:02d}",
        },
    )


@router.get("/requests/new")
def new_request_form(request: Request):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    if not can(user["role"], "leave", "create") or not user.get("employee_id"):
        return _forbidden(request, user)

    with get_conn() as conn:
        leave_types = conn.execute("SELECT * FROM leave_types ORDER BY id").fetchall()

    return templates.TemplateResponse(
        request, "leave_form.html", {"user": user, "leave_types": leave_types, "error": None}
    )


@router.post("/requests/new")
def create_request(
    request: Request,
    leave_type_id: int = Form(...),
    start_date: str = Form(...),
    end_date: str = Form(...),
    comment: str = Form(""),
):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    if not can(user["role"], "leave", "create") or not user.get("employee_id"):
        return _forbidden(request, user)

    employee_id = user["employee_id"]
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)

    def _render_error(msg: str):
        with get_conn() as conn:
            leave_types = conn.execute("SELECT * FROM leave_types ORDER BY id").fetchall()
        return templates.TemplateResponse(
            request, "leave_form.html", {"user": user, "leave_types": leave_types, "error": msg},
            status_code=422,
        )

    if end < start:
        return _render_error("End date cannot be before start date.")

    days = count_working_days(start, end)
    if days == 0:
        return _render_error("This date range contains no working days.")

    with get_conn() as conn:
        overlap = conn.execute(
            """SELECT 1 FROM leave_requests
               WHERE employee_id = ? AND status IN ('pending', 'approved')
               AND NOT (end_date < ? OR start_date > ?)""",
            (employee_id, start_date, end_date),
        ).fetchone()
        if overlap:
            return _render_error("You already have a pending or approved leave request that overlaps these dates.")

        available = _available_balance(conn, employee_id, leave_type_id)
        if days > available:
            return _render_error(f"Insufficient balance: {available} day(s) available, {days} requested.")

        now = datetime.now(timezone.utc).isoformat()
        cur = conn.execute(
            """INSERT INTO leave_requests
               (employee_id, leave_type_id, start_date, end_date, days, status, comment, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, 'pending', ?, ?, ?)""",
            (employee_id, leave_type_id, start_date, end_date, days, comment, now, now),
        )
        log_audit(conn, user["username"], "create", "leave_request", str(cur.lastrowid), None)

    return RedirectResponse("/leave/requests", status_code=303)


@router.post("/requests/{request_id}/approve")
def approve_request(request: Request, request_id: int, decision_comment: str = Form("")):
    return _decide(request, request_id, "approved", decision_comment)


@router.post("/requests/{request_id}/reject")
def reject_request(request: Request, request_id: int, decision_comment: str = Form("")):
    return _decide(request, request_id, "rejected", decision_comment)


def _decide(request: Request, request_id: int, new_status: str, decision_comment: str):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)

    scope = get_scope(user["role"], "leave", "approve")
    if scope is None:
        return _forbidden(request, user)

    with get_conn() as conn:
        row = conn.execute(
            """SELECT lr.*, e.supervisor_id FROM leave_requests lr
               JOIN employees e ON e.id = lr.employee_id WHERE lr.id = ?""",
            (request_id,),
        ).fetchone()
        if row is None:
            return templates.TemplateResponse(
                request, "404.html", {"user": user, "what": "Leave request"}, status_code=404
            )
        if not _leave_request_in_scope(user, row, scope):
            return _forbidden(request, user)
        if row["status"] != "pending":
            # already decided (e.g. double-submit) — no-op, not an error
            return RedirectResponse("/leave/requests", status_code=303)

        if new_status == "approved":
            fault = check_and_consume_fault(conn, "leave.approval.submit")
            if fault == "conflict":
                raise HTTPException(409, "This leave request was modified by someone else. Reload and try again.")
            if fault == "error":
                raise HTTPException(500, "Unexpected error while approving this leave request.")

        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            """UPDATE leave_requests SET status = ?, approver_id = ?, decision_comment = ?, updated_at = ?
               WHERE id = ?""",
            (new_status, user.get("employee_id"), decision_comment, now, request_id),
        )
        log_audit(conn, user["username"], new_status, "leave_request", str(request_id), decision_comment)

    return RedirectResponse("/leave/requests", status_code=303)


@router.post("/requests/bulk-approve")
def bulk_approve(request: Request, request_ids: list[str] = Form(...)):
    return _bulk_decide(request, request_ids, "approved")


@router.post("/requests/bulk-reject")
def bulk_reject(request: Request, request_ids: list[str] = Form(...)):
    return _bulk_decide(request, request_ids, "rejected")


def _bulk_decide(request: Request, request_ids: list[str], new_status: str):
    """Same rules as _decide() (per-row scope + pending-only), applied over a checkbox
    selection from the list page. Unlike the single-row endpoint, an unauthorized or
    already-decided row is silently skipped rather than failing the whole batch — the
    checkbox selection is just "attempt these," not an all-or-nothing transaction."""
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)

    scope = get_scope(user["role"], "leave", "approve")
    if scope is None:
        return _forbidden(request, user)

    now = datetime.now(timezone.utc).isoformat()
    with get_conn() as conn:
        for rid_str in request_ids:
            try:
                rid = int(rid_str)
            except ValueError:
                continue
            row = conn.execute(
                """SELECT lr.*, e.supervisor_id FROM leave_requests lr
                   JOIN employees e ON e.id = lr.employee_id WHERE lr.id = ?""",
                (rid,),
            ).fetchone()
            if row is None or not _leave_request_in_scope(user, row, scope) or row["status"] != "pending":
                continue
            if new_status == "approved":
                fault = check_and_consume_fault(conn, "leave.approval.submit")
                if fault in ("conflict", "error"):
                    continue
            conn.execute(
                """UPDATE leave_requests SET status = ?, approver_id = ?, decision_comment = ?, updated_at = ?
                   WHERE id = ?""",
                (new_status, user.get("employee_id"), "Bulk action", now, rid),
            )
            log_audit(conn, user["username"], new_status, "leave_request", str(rid), "bulk action")

    return RedirectResponse("/leave/requests", status_code=303)


@router.post("/requests/{request_id}/cancel")
def cancel_request(request: Request, request_id: int):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)

    scope = get_scope(user["role"], "leave", "cancel")
    if scope is None:
        return _forbidden(request, user)

    with get_conn() as conn:
        row = conn.execute("SELECT * FROM leave_requests WHERE id = ?", (request_id,)).fetchone()
        if row is None:
            return templates.TemplateResponse(
                request, "404.html", {"user": user, "what": "Leave request"}, status_code=404
            )
        if not _leave_request_in_scope(user, row, scope):
            return _forbidden(request, user)
        if row["status"] != "pending":
            return RedirectResponse("/leave/requests", status_code=303)

        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "UPDATE leave_requests SET status = 'cancelled', updated_at = ? WHERE id = ?",
            (now, request_id),
        )
        log_audit(conn, user["username"], "cancel", "leave_request", str(request_id), None)

    return RedirectResponse("/leave/requests", status_code=303)

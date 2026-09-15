from datetime import date, datetime, timezone

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
def list_requests(request: Request, status: str = ""):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)

    scope = get_scope(user["role"], "leave", "read")
    if scope is None:
        return _forbidden(request, user)

    where = ["1=1"]
    params: list = []
    if scope == "own":
        where.append("lr.employee_id = ?")
        params.append(user.get("employee_id"))
    elif scope == "direct_reports":
        where.append("e.supervisor_id = ?")
        params.append(user.get("employee_id"))
    if status:
        where.append("lr.status = ?")
        params.append(status)
    where_sql = " AND ".join(where)

    with get_conn() as conn:
        rows = conn.execute(
            f"""SELECT lr.*, e.first_name, e.last_name, lt.name AS leave_type_name
                FROM leave_requests lr
                JOIN employees e ON e.id = lr.employee_id
                JOIN leave_types lt ON lt.id = lr.leave_type_id
                WHERE {where_sql}
                ORDER BY lr.created_at DESC""",
            params,
        ).fetchall()
        leave_types = conn.execute("SELECT * FROM leave_types ORDER BY id").fetchall()
        balances = []
        if user.get("employee_id"):
            balances = conn.execute(
                """SELECT lt.name, lb.balance_days FROM leave_balances lb
                   JOIN leave_types lt ON lt.id = lb.leave_type_id
                   WHERE lb.employee_id = ? ORDER BY lt.id""",
                (user["employee_id"],),
            ).fetchall()

    can_approve = can(user["role"], "leave", "approve")
    can_create = can(user["role"], "leave", "create")

    return templates.TemplateResponse(
        request,
        "leave_list.html",
        {
            "user": user,
            "requests": rows,
            "status": status,
            "leave_types": leave_types,
            "balances": balances,
            "can_approve": can_approve,
            "can_create": can_create,
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

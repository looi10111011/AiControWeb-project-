from datetime import date, datetime, timedelta, timezone

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from benchmark_target.app.config import TEMPLATES_DIR
from benchmark_target.app.db import get_conn, log_audit
from benchmark_target.app.deps import can, get_current_user, get_scope
from benchmark_target.app.faults import check_and_consume_fault

router = APIRouter(prefix="/time")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

WEEKDAY_LABELS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


def _forbidden(request: Request, user: dict):
    return templates.TemplateResponse(request, "403.html", {"user": user}, status_code=403)


def _timesheet_in_scope(user: dict, row, scope: str) -> bool:
    if scope == "all":
        return True
    if scope == "own":
        return row["employee_id"] == user.get("employee_id")
    if scope == "direct_reports":
        return row["supervisor_id"] == user.get("employee_id")
    return False


def _week_dates(week_start: str) -> list[str]:
    start = date.fromisoformat(week_start)
    return [(start + timedelta(days=i)).isoformat() for i in range(5)]


@router.get("/timesheets")
def list_timesheets(request: Request, view: str = "mine"):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)

    approve_scope = get_scope(user["role"], "timesheet", "approve")
    show_team = view == "team" and approve_scope is not None

    with get_conn() as conn:
        if show_team:
            if approve_scope == "all":
                rows = conn.execute(
                    """SELECT t.*, e.first_name, e.last_name FROM timesheets t
                       JOIN employees e ON e.id = t.employee_id ORDER BY t.week_start_date DESC"""
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT t.*, e.first_name, e.last_name FROM timesheets t
                       JOIN employees e ON e.id = t.employee_id
                       WHERE e.supervisor_id = ? ORDER BY t.week_start_date DESC""",
                    (user["employee_id"],),
                ).fetchall()
        elif user.get("employee_id"):
            rows = conn.execute(
                """SELECT t.*, e.first_name, e.last_name FROM timesheets t
                   JOIN employees e ON e.id = t.employee_id
                   WHERE t.employee_id = ? ORDER BY t.week_start_date DESC""",
                (user["employee_id"],),
            ).fetchall()
        else:
            rows = []

    return templates.TemplateResponse(
        request,
        "time_list.html",
        {
            "user": user,
            "timesheets": rows,
            "view": "team" if show_team else "mine",
            "can_view_team": approve_scope is not None,
            "can_create": can(user["role"], "timesheet", "create") and bool(user.get("employee_id")),
        },
    )


@router.get("/timesheets/new")
def new_timesheet_form(request: Request):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    if not can(user["role"], "timesheet", "create") or not user.get("employee_id"):
        return _forbidden(request, user)

    return templates.TemplateResponse(
        request,
        "time_form.html",
        {
            "user": user,
            "week_start_date": "",
            "day_entries": list(zip(WEEKDAY_LABELS, [{"project": "", "hours": ""}] * 5)),
            "error": None,
            "editing_id": None,
        },
    )


@router.post("/timesheets/new")
async def create_timesheet(request: Request, week_start_date: str = Form(...)):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    if not can(user["role"], "timesheet", "create") or not user.get("employee_id"):
        return _forbidden(request, user)

    form = await request.form()
    week_dates = _week_dates(week_start_date)
    now = datetime.now(timezone.utc).isoformat()

    with get_conn() as conn:
        existing = conn.execute(
            "SELECT 1 FROM timesheets WHERE employee_id = ? AND week_start_date = ?",
            (user["employee_id"], week_start_date),
        ).fetchone()
        if existing:
            return templates.TemplateResponse(
                request,
                "time_form.html",
                {
                    "user": user,
                    "week_start_date": week_start_date,
                    "day_entries": list(zip(WEEKDAY_LABELS, [{"project": "", "hours": ""}] * 5)),
                    "error": "A timesheet for this week already exists.",
                    "editing_id": None,
                },
                status_code=409,
            )

        cur = conn.execute(
            "INSERT INTO timesheets (employee_id, week_start_date, status, created_at, updated_at) VALUES (?, ?, 'draft', ?, ?)",
            (user["employee_id"], week_start_date, now, now),
        )
        ts_id = cur.lastrowid
        for i, work_date in enumerate(week_dates):
            project = (form.get(f"project_{i}") or "").strip()
            hours = form.get(f"hours_{i}") or "0"
            try:
                hours_f = float(hours)
            except ValueError:
                hours_f = 0
            if project and hours_f > 0:
                conn.execute(
                    "INSERT INTO timesheet_entries (timesheet_id, work_date, project, hours) VALUES (?, ?, ?, ?)",
                    (ts_id, work_date, project, hours_f),
                )
        log_audit(conn, user["username"], "create", "timesheet", str(ts_id), None)

    return RedirectResponse(f"/time/timesheets/{ts_id}", status_code=303)


@router.get("/timesheets/{timesheet_id}")
def view_timesheet(request: Request, timesheet_id: int):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)

    with get_conn() as conn:
        row = conn.execute(
            """SELECT t.*, e.first_name, e.last_name, e.supervisor_id FROM timesheets t
               JOIN employees e ON e.id = t.employee_id WHERE t.id = ?""",
            (timesheet_id,),
        ).fetchone()
        if row is None:
            return templates.TemplateResponse(
                request, "404.html", {"user": user, "what": "Timesheet"}, status_code=404
            )

        read_scope = get_scope(user["role"], "timesheet", "read")
        approve_scope = get_scope(user["role"], "timesheet", "approve")
        is_owner = row["employee_id"] == user.get("employee_id")
        can_view = is_owner or (read_scope and _timesheet_in_scope(user, row, read_scope)) or \
            (approve_scope and _timesheet_in_scope(user, row, approve_scope))
        if not can_view:
            return _forbidden(request, user)

        entries = conn.execute(
            "SELECT * FROM timesheet_entries WHERE timesheet_id = ? ORDER BY work_date", (timesheet_id,)
        ).fetchall()
        total_hours = sum(e["hours"] for e in entries)

    can_submit = is_owner and row["status"] in ("draft", "rejected")
    can_edit = is_owner and row["status"] in ("draft", "rejected")
    can_approve = approve_scope and _timesheet_in_scope(user, row, approve_scope) and row["status"] == "submitted"

    return templates.TemplateResponse(
        request,
        "time_view.html",
        {
            "user": user,
            "timesheet": row,
            "entries": entries,
            "total_hours": total_hours,
            "can_submit": can_submit,
            "can_edit": can_edit,
            "can_approve": can_approve,
        },
    )


@router.post("/timesheets/{timesheet_id}/submit")
def submit_timesheet(request: Request, timesheet_id: int):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)

    with get_conn() as conn:
        row = conn.execute("SELECT * FROM timesheets WHERE id = ?", (timesheet_id,)).fetchone()
        if row is None:
            return templates.TemplateResponse(
                request, "404.html", {"user": user, "what": "Timesheet"}, status_code=404
            )
        if row["employee_id"] != user.get("employee_id"):
            return _forbidden(request, user)
        if row["status"] not in ("draft", "rejected"):
            return RedirectResponse(f"/time/timesheets/{timesheet_id}", status_code=303)

        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "UPDATE timesheets SET status = 'submitted', updated_at = ? WHERE id = ?", (now, timesheet_id)
        )
        log_audit(conn, user["username"], "submit", "timesheet", str(timesheet_id), None)

    return RedirectResponse(f"/time/timesheets/{timesheet_id}", status_code=303)


def _decide(request: Request, timesheet_id: int, new_status: str):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)

    scope = get_scope(user["role"], "timesheet", "approve")
    if scope is None:
        return _forbidden(request, user)

    with get_conn() as conn:
        row = conn.execute(
            """SELECT t.*, e.supervisor_id FROM timesheets t
               JOIN employees e ON e.id = t.employee_id WHERE t.id = ?""",
            (timesheet_id,),
        ).fetchone()
        if row is None:
            return templates.TemplateResponse(
                request, "404.html", {"user": user, "what": "Timesheet"}, status_code=404
            )
        if not _timesheet_in_scope(user, row, scope):
            return _forbidden(request, user)
        if row["status"] != "submitted":
            return RedirectResponse(f"/time/timesheets/{timesheet_id}", status_code=303)

        if new_status == "approved":
            fault = check_and_consume_fault(conn, "timesheet.approval.submit")
            if fault == "conflict":
                raise HTTPException(409, "This timesheet was modified by someone else. Reload and try again.")
            if fault == "error":
                raise HTTPException(500, "Unexpected error while approving this timesheet.")

        now = datetime.now(timezone.utc).isoformat()
        conn.execute("UPDATE timesheets SET status = ?, updated_at = ? WHERE id = ?", (new_status, now, timesheet_id))
        log_audit(conn, user["username"], new_status, "timesheet", str(timesheet_id), None)

    return RedirectResponse(f"/time/timesheets/{timesheet_id}", status_code=303)


@router.get("/timesheets/{timesheet_id}/edit")
def edit_timesheet_form(request: Request, timesheet_id: int):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)

    with get_conn() as conn:
        row = conn.execute("SELECT * FROM timesheets WHERE id = ?", (timesheet_id,)).fetchone()
        if row is None:
            return templates.TemplateResponse(
                request, "404.html", {"user": user, "what": "Timesheet"}, status_code=404
            )
        if row["employee_id"] != user.get("employee_id") or row["status"] not in ("draft", "rejected"):
            return _forbidden(request, user)

        existing = conn.execute(
            "SELECT work_date, project, hours FROM timesheet_entries WHERE timesheet_id = ?", (timesheet_id,)
        ).fetchall()
        by_date = {e["work_date"]: {"project": e["project"], "hours": e["hours"]} for e in existing}

    day_entries = [
        (label, by_date.get(work_date, {"project": "", "hours": ""}))
        for label, work_date in zip(WEEKDAY_LABELS, _week_dates(row["week_start_date"]))
    ]

    return templates.TemplateResponse(
        request,
        "time_form.html",
        {
            "user": user,
            "week_start_date": row["week_start_date"],
            "day_entries": day_entries,
            "error": None,
            "editing_id": timesheet_id,
        },
    )


@router.post("/timesheets/{timesheet_id}/edit")
async def update_timesheet(request: Request, timesheet_id: int):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)

    form = await request.form()
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM timesheets WHERE id = ?", (timesheet_id,)).fetchone()
        if row is None:
            return templates.TemplateResponse(
                request, "404.html", {"user": user, "what": "Timesheet"}, status_code=404
            )
        if row["employee_id"] != user.get("employee_id") or row["status"] not in ("draft", "rejected"):
            return _forbidden(request, user)

        conn.execute("DELETE FROM timesheet_entries WHERE timesheet_id = ?", (timesheet_id,))
        for i, work_date in enumerate(_week_dates(row["week_start_date"])):
            project = (form.get(f"project_{i}") or "").strip()
            hours = form.get(f"hours_{i}") or "0"
            try:
                hours_f = float(hours)
            except ValueError:
                hours_f = 0
            if project and hours_f > 0:
                conn.execute(
                    "INSERT INTO timesheet_entries (timesheet_id, work_date, project, hours) VALUES (?, ?, ?, ?)",
                    (timesheet_id, work_date, project, hours_f),
                )

        now = datetime.now(timezone.utc).isoformat()
        conn.execute("UPDATE timesheets SET status = 'draft', updated_at = ? WHERE id = ?", (now, timesheet_id))
        log_audit(conn, user["username"], "update", "timesheet", str(timesheet_id), None)

    return RedirectResponse(f"/time/timesheets/{timesheet_id}", status_code=303)


@router.post("/timesheets/{timesheet_id}/approve")
def approve_timesheet(request: Request, timesheet_id: int):
    return _decide(request, timesheet_id, "approved")


@router.post("/timesheets/{timesheet_id}/reject")
def reject_timesheet(request: Request, timesheet_id: int):
    return _decide(request, timesheet_id, "rejected")

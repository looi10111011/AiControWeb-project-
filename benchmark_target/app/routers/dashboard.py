from datetime import date, datetime, timezone

from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from benchmark_target.app.config import TEMPLATES_DIR
from benchmark_target.app.db import get_conn
from benchmark_target.app.deps import get_current_user

router = APIRouter()
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Human-friendly labels for the Recent Activity feed, keyed by the exact (action, resource)
# pairs every router's log_audit() call already writes (see db.py::log_audit call sites).
# Anything not listed falls back to a generic "<action> <resource>" string rather than
# silently dropping the row.
_ACTIVITY_LABELS = {
    ("create", "employee"): "New employee added",
    ("update", "employee"): "Employee profile updated",
    ("deactivate", "employee"): "Employee deactivated",
    ("activate", "employee"): "Employee reactivated",
    ("create", "leave_request"): "New leave request submitted",
    ("approved", "leave_request"): "Leave request approved",
    ("rejected", "leave_request"): "Leave request rejected",
    ("cancel", "leave_request"): "Leave request cancelled",
    ("create", "user"): "New user account created",
    ("change_role", "user"): "User role changed",
    ("login_success", "user"): "User signed in",
    ("hire", "candidate"): "Candidate hired",
    ("status_change", "candidate"): "Candidate status updated",
    ("schedule_interview", "candidate"): "Interview scheduled",
    ("toggle_status", "vacancy"): "Vacancy status changed",
    ("create", "timesheet"): "Timesheet created",
    ("submit", "timesheet"): "Timesheet submitted",
    ("upload", "document"): "Document uploaded",
    ("self_review", "review"): "Self review submitted",
    ("manager_review", "review"): "Manager review submitted",
}

_ACTIVITY_ICON = {
    "employee": ("users", "chip-indigo"),
    "leave_request": ("calendar", "chip-amber"),
    "user": ("user-circle", "chip-indigo"),
    "candidate": ("briefcase", "chip-green"),
    "vacancy": ("briefcase", "chip-green"),
    "timesheet": ("clock", "chip-amber"),
    "document": ("file", "chip-pink"),
    "review": ("star", "chip-pink"),
}


def _relative_time(iso_ts: str) -> str:
    try:
        ts = datetime.fromisoformat(iso_ts)
    except ValueError:
        return iso_ts
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    seconds = (datetime.now(timezone.utc) - ts).total_seconds()
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        return f"{int(seconds // 60)}m ago"
    if seconds < 86400:
        return f"{int(seconds // 3600)}h ago"
    return f"{int(seconds // 86400)}d ago"


@router.get("/")
def root(request: Request):
    return RedirectResponse("/dashboard", status_code=303)


@router.get("/dashboard")
def dashboard(request: Request):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)

    today = date.today().isoformat()
    department_breakdown = []
    on_leave_today = 0
    open_positions = 0
    hiring_pipeline = []
    leave_summary = {"approved": 0, "pending": 0, "rejected": 0, "total": 0}
    recent_activity = []

    with get_conn() as conn:
        total_employees = conn.execute(
            "SELECT COUNT(*) c FROM employees WHERE status = 'active'"
        ).fetchone()["c"]
        total_users = conn.execute("SELECT COUNT(*) c FROM users WHERE enabled = 1").fetchone()["c"]
        my_employee = None
        if user.get("employee_id"):
            my_employee = conn.execute(
                "SELECT * FROM employees WHERE id = ?", (user["employee_id"],)
            ).fetchone()
        direct_reports_count = 0
        if user["role"] == "supervisor" and user.get("employee_id"):
            direct_reports_count = conn.execute(
                "SELECT COUNT(*) c FROM employees WHERE supervisor_id = ?", (user["employee_id"],)
            ).fetchone()["c"]

        # Org-wide widgets: real queries only, no fabricated deltas — this app keeps no
        # historical snapshot to compute an honest "vs last month" from, so the richer
        # dashboard shows current counts and real breakdowns, not invented trend %.
        if user["role"] != "ess":
            department_breakdown = [
                dict(row) for row in conn.execute(
                    "SELECT department, COUNT(*) c FROM employees WHERE status = 'active' "
                    "GROUP BY department ORDER BY c DESC"
                ).fetchall()
            ]
            on_leave_today = conn.execute(
                "SELECT COUNT(*) c FROM leave_requests WHERE status = 'approved' "
                "AND start_date <= ? AND end_date >= ?", (today, today),
            ).fetchone()["c"]
            open_positions = conn.execute(
                "SELECT COUNT(*) c FROM vacancies WHERE status = 'open'"
            ).fetchone()["c"]

            pipeline_counts = {
                row["status"]: row["c"]
                for row in conn.execute("SELECT status, COUNT(*) c FROM candidates GROUP BY status").fetchall()
            }
            hiring_pipeline = [
                {"label": "Applied", "count": pipeline_counts.get("applied", 0)},
                {"label": "Shortlisted", "count": pipeline_counts.get("shortlisted", 0)},
                {"label": "Interview", "count": pipeline_counts.get("interview", 0)},
                {"label": "Hired", "count": pipeline_counts.get("hired", 0)},
            ]

            # All-time totals, not "this month": the seed fixtures use fixed exemplar
            # dates (2026-01/02) rather than dates relative to today, so a month filter
            # here would show an empty widget on this data regardless of when it's viewed.
            leave_counts = {
                row["status"]: row["c"]
                for row in conn.execute(
                    "SELECT status, COUNT(*) c FROM leave_requests GROUP BY status"
                ).fetchall()
            }
            leave_summary = {
                "approved": leave_counts.get("approved", 0),
                "pending": leave_counts.get("pending", 0),
                "rejected": leave_counts.get("rejected", 0),
            }
            leave_summary["total"] = sum(leave_summary.values())

        if user["role"] == "admin":
            for row in conn.execute(
                "SELECT timestamp, actor_username, action, resource, resource_id, details "
                "FROM audit_log ORDER BY timestamp DESC LIMIT 6"
            ).fetchall():
                icon_name, chip_class = _ACTIVITY_ICON.get(row["resource"], ("file", "chip-indigo"))
                title = _ACTIVITY_LABELS.get(
                    (row["action"], row["resource"]),
                    f"{row['action'].replace('_', ' ').capitalize()} {row['resource'].replace('_', ' ')}",
                )
                recent_activity.append({
                    "title": title,
                    "detail": f"by {row['actor_username']}" + (f" · {row['details']}" if row["details"] else ""),
                    "when": _relative_time(row["timestamp"]),
                    "icon": icon_name,
                    "chip_class": chip_class,
                })

    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {
            "user": user,
            "total_employees": total_employees,
            "total_users": total_users,
            "my_employee": my_employee,
            "direct_reports_count": direct_reports_count,
            "department_breakdown": department_breakdown,
            "on_leave_today": on_leave_today,
            "open_positions": open_positions,
            "hiring_pipeline": hiring_pipeline,
            "leave_summary": leave_summary,
            "recent_activity": recent_activity,
            "today_label": date.today().strftime("%A, %b %d, %Y"),
        },
    )

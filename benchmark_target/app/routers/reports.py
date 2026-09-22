import csv
import io
from datetime import date, timedelta

from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from benchmark_target.app.config import TEMPLATES_DIR
from benchmark_target.app.db import get_conn, log_audit
from benchmark_target.app.deps import can, get_current_user
from benchmark_target.app.seed import DEPARTMENTS

router = APIRouter(prefix="/reports")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Keyed by the audit_log.resource_id every report export writes (see employees_report()/
# leave_report() below) — lets reports_home() derive an icon/chip and a working "re-export"
# link structurally instead of string-matching the human-readable `details` label.
_REPORT_KIND = {
    "employees": {"icon": "users", "chip": "chip-indigo"},
    "leave": {"icon": "calendar", "chip": "chip-green"},
}


def _forbidden(request: Request, user: dict):
    return templates.TemplateResponse(request, "403.html", {"user": user}, status_code=403)


def _trend(now: int, past: int) -> dict:
    """Same shape/reasoning as leave.py's and pim.py's _trend(): a real now-vs-30-days-ago
    delta, never a fabricated one."""
    if past == 0:
        return {"pct": 100, "direction": "up"} if now > 0 else {"pct": 0, "direction": "flat"}
    pct = round((now - past) / past * 100)
    direction = "up" if pct > 0 else "down" if pct < 0 else "flat"
    return {"pct": abs(pct), "direction": direction}


def _csv_response(fieldnames: list[str], rows: list[dict], filename: str) -> StreamingResponse:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("")
def reports_home(request: Request):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    if not can(user["role"], "report", "read"):
        return _forbidden(request, user)

    thirty_days_ago = (date.today() - timedelta(days=30)).isoformat()
    week_start = (date.today() - timedelta(days=6)).isoformat()

    with get_conn() as conn:

        def _daily_series(sql: str, params: tuple, days: int = 7) -> list[int]:
            """Zero-filled, real day-by-day series for a stat card's sparkline — never a
            fabricated trend line (same reasoning as leave.py's _sparkline()). `sql` must
            select (d, v) pairs via `substr(<date col>, 1, 10) AS d, <agg> AS v ... GROUP BY
            d`, filtered by `params` to the last `days` days."""
            start = date.today() - timedelta(days=days - 1)
            found = {r["d"]: (r["v"] or 0) for r in conn.execute(sql, params).fetchall()}
            return [found.get((start + timedelta(days=i)).isoformat(), 0) for i in range(days)]

        # ---- Total Employees: active headcount, real COUNT(*) ----
        employees_now = conn.execute(
            "SELECT COUNT(*) c FROM employees WHERE status = 'active'"
        ).fetchone()["c"]
        employees_past = conn.execute(
            "SELECT COUNT(*) c FROM employees WHERE status = 'active' AND hire_date <= ?",
            (thirty_days_ago,),
        ).fetchone()["c"]
        # Sparkline: employees hired on each of the last 7 days — the only per-day signal
        # employees has (status changes aren't timestamped). Seed hires are all from 2019,
        # so this is honestly flat until a new hire is actually added through the app.
        stat_employees = {
            "value": employees_now,
            "spark": _daily_series(
                "SELECT substr(hire_date, 1, 10) d, COUNT(*) v FROM employees "
                "WHERE hire_date >= ? GROUP BY d",
                (week_start,),
            ),
            **_trend(employees_now, employees_past),
        }

        # ---- Total Leave Days: approved only, i.e. days actually granted rather than
        # merely requested (pending/rejected days were never taken) ----
        leave_days_now = conn.execute(
            "SELECT COALESCE(SUM(days), 0) v FROM leave_requests WHERE status = 'approved'"
        ).fetchone()["v"]
        leave_days_past = conn.execute(
            "SELECT COALESCE(SUM(days), 0) v FROM leave_requests "
            "WHERE status = 'approved' AND created_at <= ?",
            (thirty_days_ago,),
        ).fetchone()["v"]
        stat_leave_days = {
            "value": leave_days_now,
            "spark": _daily_series(
                "SELECT substr(created_at, 1, 10) d, COALESCE(SUM(days), 0) v FROM leave_requests "
                "WHERE status = 'approved' AND created_at >= ? GROUP BY d",
                (week_start,),
            ),
            **_trend(leave_days_now, leave_days_past),
        }

        # ---- Total Working Hours: every logged timesheet entry, any timesheet status —
        # draft/submitted/rejected hours are still real time that was worked, just not yet
        # administratively approved ----
        hours_now = conn.execute(
            "SELECT COALESCE(SUM(hours), 0) v FROM timesheet_entries"
        ).fetchone()["v"]
        hours_past = conn.execute(
            """SELECT COALESCE(SUM(te.hours), 0) v FROM timesheet_entries te
               JOIN timesheets t ON t.id = te.timesheet_id WHERE t.created_at <= ?""",
            (thirty_days_ago,),
        ).fetchone()["v"]
        stat_hours = {
            "value": hours_now,
            "spark": _daily_series(
                "SELECT substr(work_date, 1, 10) d, COALESCE(SUM(hours), 0) v "
                "FROM timesheet_entries WHERE work_date >= ? GROUP BY d",
                (week_start,),
            ),
            **_trend(hours_now, hours_past),
        }

        # ---- Reports Generated: a real audit trail of CSV exports. employees_report() and
        # leave_report() below call log_audit(..., "export", "report", ...) right before
        # returning a CSV — that's the only source of "report history" in this app, so
        # nothing here is fabricated; before any export has ever happened this is honestly 0.
        reports_now = conn.execute(
            "SELECT COUNT(*) c FROM audit_log WHERE resource = 'report' AND action = 'export'"
        ).fetchone()["c"]
        reports_past = conn.execute(
            "SELECT COUNT(*) c FROM audit_log WHERE resource = 'report' AND action = 'export' "
            "AND timestamp <= ?",
            (thirty_days_ago,),
        ).fetchone()["c"]
        stat_reports = {
            "value": reports_now,
            "spark": _daily_series(
                "SELECT substr(timestamp, 1, 10) d, COUNT(*) v FROM audit_log "
                "WHERE resource = 'report' AND action = 'export' AND timestamp >= ? GROUP BY d",
                (week_start,),
            ),
            **_trend(reports_now, reports_past),
        }

        # ---- Recent Reports: the last 10 export events behind the stat above.
        # resource_id carries the report route ("employees"/"leave"), so the icon/chip and a
        # working "re-export" link are derived structurally rather than string-matching the
        # human-readable `details` label.
        recent_reports = []
        for row in conn.execute(
            """SELECT timestamp, actor_username, resource_id, details FROM audit_log
               WHERE resource = 'report' AND action = 'export'
               ORDER BY timestamp DESC LIMIT 10"""
        ).fetchall():
            kind = _REPORT_KIND.get(row["resource_id"])
            recent_reports.append({
                "name": row["details"] or "Report",
                "generated_on": row["timestamp"][:19].replace("T", " "),
                "generated_by": row["actor_username"],
                "icon": kind["icon"] if kind else "file",
                "chip": kind["chip"] if kind else "chip-indigo",
                # The exported file itself is never persisted (no storage for report
                # exports in this app), and the filters used at export time weren't
                # recorded either — so this re-runs a fresh, unfiltered CSV of the same
                # report type rather than linking to a dead/stored file. The template
                # labels it "Re-export" (not "Download") to be honest about that.
                "download_url": f"/reports/{row['resource_id']}?format=csv" if kind else None,
            })

    return templates.TemplateResponse(
        request,
        "reports_home.html",
        {
            "user": user,
            "stat_employees": stat_employees,
            "stat_leave_days": stat_leave_days,
            "stat_hours": stat_hours,
            "stat_reports": stat_reports,
            "recent_reports": recent_reports,
        },
    )


@router.get("/employees")
def employees_report(request: Request, department: str = "", status: str = "", format: str = "html"):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    if not can(user["role"], "report", "read"):
        return _forbidden(request, user)

    where = ["1=1"]
    params: list = []
    if department:
        where.append("department = ?")
        params.append(department)
    if status:
        where.append("status = ?")
        params.append(status)
    where_sql = " AND ".join(where)

    with get_conn() as conn:
        rows = conn.execute(
            f"""SELECT employee_code, first_name, last_name, department, job_title, status, hire_date,
                       leave_balance_annual
                FROM employees WHERE {where_sql} ORDER BY department, last_name""",
            params,
        ).fetchall()
        if format == "csv":
            # Real, self-populating "Reports Generated" history for reports_home() — logged
            # here, inside the still-open connection, rather than after it (see reports_home()
            # comment on _REPORT_KIND for how resource_id/details are consumed).
            log_audit(conn, user["username"], "export", "report", "employees", "Employees Report")

    fieldnames = ["employee_code", "first_name", "last_name", "department", "job_title", "status",
                  "hire_date", "leave_balance_annual"]
    if format == "csv":
        return _csv_response(fieldnames, [dict(r) for r in rows], "employees_report.csv")

    return templates.TemplateResponse(
        request,
        "reports_employees.html",
        {"user": user, "rows": rows, "departments": DEPARTMENTS, "department": department, "status": status},
    )


@router.get("/leave")
def leave_report(request: Request, start_date: str = "", end_date: str = "", status: str = "", format: str = "html"):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    if not can(user["role"], "report", "read"):
        return _forbidden(request, user)

    where = ["1=1"]
    params: list = []
    if start_date:
        where.append("lr.end_date >= ?")
        params.append(start_date)
    if end_date:
        where.append("lr.start_date <= ?")
        params.append(end_date)
    if status:
        where.append("lr.status = ?")
        params.append(status)
    where_sql = " AND ".join(where)

    with get_conn() as conn:
        rows = conn.execute(
            f"""SELECT e.employee_code, e.first_name, e.last_name, lt.name AS leave_type,
                       lr.start_date, lr.end_date, lr.days, lr.status
                FROM leave_requests lr
                JOIN employees e ON e.id = lr.employee_id
                JOIN leave_types lt ON lt.id = lr.leave_type_id
                WHERE {where_sql} ORDER BY lr.start_date""",
            params,
        ).fetchall()
        if format == "csv":
            # Same real "Reports Generated" audit trail as employees_report() above.
            log_audit(conn, user["username"], "export", "report", "leave", "Leave Report")

    fieldnames = ["employee_code", "first_name", "last_name", "leave_type", "start_date", "end_date", "days", "status"]
    if format == "csv":
        return _csv_response(fieldnames, [dict(r) for r in rows], "leave_report.csv")

    return templates.TemplateResponse(
        request,
        "reports_leave.html",
        {"user": user, "rows": rows, "start_date": start_date, "end_date": end_date, "status": status},
    )

import csv
import io

from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from benchmark_target.app.config import TEMPLATES_DIR
from benchmark_target.app.db import get_conn
from benchmark_target.app.deps import can, get_current_user
from benchmark_target.app.seed import DEPARTMENTS

router = APIRouter(prefix="/reports")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def _forbidden(request: Request, user: dict):
    return templates.TemplateResponse(request, "403.html", {"user": user}, status_code=403)


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
    return templates.TemplateResponse(request, "reports_home.html", {"user": user})


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

    fieldnames = ["employee_code", "first_name", "last_name", "leave_type", "start_date", "end_date", "days", "status"]
    if format == "csv":
        return _csv_response(fieldnames, [dict(r) for r in rows], "leave_report.csv")

    return templates.TemplateResponse(
        request,
        "reports_leave.html",
        {"user": user, "rows": rows, "start_date": start_date, "end_date": end_date, "status": status},
    )

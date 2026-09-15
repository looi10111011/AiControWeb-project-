from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from benchmark_target.app.config import TEMPLATES_DIR
from benchmark_target.app.db import get_conn
from benchmark_target.app.deps import get_current_user

router = APIRouter()
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@router.get("/")
def root(request: Request):
    return RedirectResponse("/dashboard", status_code=303)


@router.get("/dashboard")
def dashboard(request: Request):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)

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

    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {
            "user": user,
            "total_employees": total_employees,
            "total_users": total_users,
            "my_employee": my_employee,
            "direct_reports_count": direct_reports_count,
        },
    )

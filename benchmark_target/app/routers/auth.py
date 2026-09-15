from fastapi import APIRouter, Form, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from benchmark_target.app.config import TEMPLATES_DIR
from benchmark_target.app.db import get_conn, log_audit
from benchmark_target.app.security import verify_password

router = APIRouter()
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@router.get("/login")
def login_form(request: Request):
    if request.session.get("user"):
        return RedirectResponse("/dashboard", status_code=303)
    return templates.TemplateResponse(request, "login.html", {"error": None})


@router.post("/login")
def login_submit(request: Request, username: str = Form(...), password: str = Form(...)):
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()

        if row is None or not verify_password(password, row["password_hash"]):
            log_audit(conn, username, "login_failed", "user", None, "invalid credentials")
            return templates.TemplateResponse(
                request, "login.html", {"error": "Invalid username or password."}, status_code=401
            )

        if not row["enabled"]:
            log_audit(conn, username, "login_failed", "user", str(row["id"]), "account disabled")
            return templates.TemplateResponse(
                request,
                "login.html",
                {"error": "This account has been disabled. Contact your administrator."},
                status_code=403,
            )

        log_audit(conn, username, "login_success", "user", str(row["id"]), None)

    request.session["user"] = {
        "id": row["id"],
        "username": row["username"],
        "role": row["role"],
        "employee_id": row["employee_id"],
    }
    return RedirectResponse("/dashboard", status_code=303)


@router.get("/logout")
def logout(request: Request):
    user = request.session.get("user")
    if user:
        with get_conn() as conn:
            log_audit(conn, user["username"], "logout", "user", str(user["id"]), None)
    request.session.clear()
    return RedirectResponse("/login", status_code=303)

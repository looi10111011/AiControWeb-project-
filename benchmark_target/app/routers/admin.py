from datetime import datetime, timezone

from fastapi import APIRouter, Form, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from benchmark_target.app.config import TEMPLATES_DIR
from benchmark_target.app.db import get_conn, log_audit
from benchmark_target.app.deps import can, get_current_user
from benchmark_target.app.security import hash_password

router = APIRouter(prefix="/admin")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

ROLES = ["admin", "supervisor", "ess"]


def _forbidden(request: Request, user: dict):
    return templates.TemplateResponse(request, "403.html", {"user": user}, status_code=403)


@router.get("/users")
def list_users(request: Request, q: str = ""):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    if not can(user["role"], "user", "read"):
        return _forbidden(request, user)

    with get_conn() as conn:
        where = "1=1"
        params: list = []
        if q:
            where += " AND username LIKE ?"
            params.append(f"%{q}%")
        rows = conn.execute(
            f"""SELECT u.*, e.first_name, e.last_name
                FROM users u LEFT JOIN employees e ON e.id = u.employee_id
                WHERE {where} ORDER BY u.username""",
            params,
        ).fetchall()

    return templates.TemplateResponse(
        request, "admin_users.html", {"user": user, "users": rows, "q": q}
    )


@router.get("/users/new")
def new_user_form(request: Request):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    if not can(user["role"], "user", "create"):
        return _forbidden(request, user)

    with get_conn() as conn:
        employees = conn.execute(
            """SELECT e.id, e.first_name, e.last_name, e.employee_code FROM employees e
               WHERE e.status = 'active' AND e.id NOT IN (SELECT employee_id FROM users WHERE employee_id IS NOT NULL)
               ORDER BY e.last_name"""
        ).fetchall()

    return templates.TemplateResponse(
        request,
        "admin_user_form.html",
        {"user": user, "target": None, "employees": employees, "roles": ROLES, "error": None},
    )


@router.post("/users/new")
def create_user(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    role: str = Form(...),
    employee_id: str = Form(""),
):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    if not can(user["role"], "user", "create"):
        return _forbidden(request, user)

    now = datetime.now(timezone.utc).isoformat()
    with get_conn() as conn:
        existing = conn.execute("SELECT 1 FROM users WHERE username = ?", (username,)).fetchone()
        if existing:
            employees = conn.execute(
                """SELECT e.id, e.first_name, e.last_name, e.employee_code FROM employees e
                   WHERE e.status = 'active' AND e.id NOT IN (SELECT employee_id FROM users WHERE employee_id IS NOT NULL)
                   ORDER BY e.last_name"""
            ).fetchall()
            return templates.TemplateResponse(
                request,
                "admin_user_form.html",
                {
                    "user": user,
                    "target": None,
                    "employees": employees,
                    "roles": ROLES,
                    "error": f'Username "{username}" already exists.',
                },
                status_code=409,
            )

        cur = conn.execute(
            """INSERT INTO users (username, password_hash, role, employee_id, enabled, created_at, updated_at)
               VALUES (?, ?, ?, ?, 1, ?, ?)""",
            (username.strip(), hash_password(password), role, int(employee_id) if employee_id else None, now, now),
        )
        log_audit(conn, user["username"], "create", "user", str(cur.lastrowid), username)

    return RedirectResponse("/admin/users", status_code=303)


@router.post("/users/{user_id}/toggle-enabled")
def toggle_enabled(request: Request, user_id: int):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    if not can(user["role"], "user", "update"):
        return _forbidden(request, user)
    if user_id == user["id"]:
        # never allow disabling your own currently-logged-in account through this route
        return RedirectResponse("/admin/users", status_code=303)

    with get_conn() as conn:
        row = conn.execute("SELECT enabled, username FROM users WHERE id = ?", (user_id,)).fetchone()
        if row is None:
            return templates.TemplateResponse(
                request, "404.html", {"user": user, "what": "User"}, status_code=404
            )
        new_state = 0 if row["enabled"] else 1
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "UPDATE users SET enabled = ?, updated_at = ? WHERE id = ?", (new_state, now, user_id)
        )
        log_audit(
            conn,
            user["username"],
            "enable" if new_state else "disable",
            "user",
            str(user_id),
            row["username"],
        )

    return RedirectResponse("/admin/users", status_code=303)


@router.post("/users/{user_id}/role")
def change_role(request: Request, user_id: int, role: str = Form(...)):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    if not can(user["role"], "user", "update"):
        return _forbidden(request, user)
    if role not in ROLES:
        return RedirectResponse("/admin/users", status_code=303)

    now = datetime.now(timezone.utc).isoformat()
    with get_conn() as conn:
        conn.execute("UPDATE users SET role = ?, updated_at = ? WHERE id = ?", (role, now, user_id))
        log_audit(conn, user["username"], "change_role", "user", str(user_id), role)

    return RedirectResponse("/admin/users", status_code=303)

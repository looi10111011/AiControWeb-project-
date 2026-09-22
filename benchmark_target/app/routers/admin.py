from datetime import date, datetime, timedelta, timezone

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


def _trend(now: int, past: int) -> dict:
    """Same shape/reasoning as leave.py's _trend(): a real now-vs-30-days-ago delta, never a
    fabricated one. `past` here is "created 30+ days ago and currently matching this stat's
    condition" (enabled/disabled/role='admin') — the closest honest analogue available, since
    users has no column timestamping when enabled/role last changed (only updated_at, which
    any edit bumps), only created_at."""
    if past == 0:
        return {"pct": 100, "direction": "up"} if now > 0 else {"pct": 0, "direction": "flat"}
    pct = round((now - past) / past * 100)
    direction = "up" if pct > 0 else "down" if pct < 0 else "flat"
    return {"pct": abs(pct), "direction": direction}


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
        rows = [
            dict(row, initial=row["username"][:1].upper())
            for row in conn.execute(
                f"""SELECT u.*, e.first_name, e.last_name
                    FROM users u LEFT JOIN employees e ON e.id = u.employee_id
                    WHERE {where} ORDER BY u.username""",
                params,
            ).fetchall()
        ]

        # Stat row: this page is admin-only already (no scope restriction like PIM's
        # supervisor/own scopes), so the counts are simply whole-table — but still
        # filter-independent, same as pim.py's stat row: a summary of the whole user
        # base, not of the current username search.
        def _count(extra_sql: str = "", extra_params: tuple = ()) -> int:
            return conn.execute(
                f"SELECT COUNT(*) c FROM users WHERE 1=1 {extra_sql}", [*extra_params]
            ).fetchone()["c"]

        def _sparkline(extra_sql: str = "", extra_params: tuple = (), days: int = 7) -> list[int]:
            start = date.today() - timedelta(days=days - 1)
            found = {
                r["d"]: r["c"]
                for r in conn.execute(
                    f"""SELECT substr(created_at, 1, 10) d, COUNT(*) c FROM users
                        WHERE 1=1 {extra_sql} AND created_at >= ?
                        GROUP BY d""",
                    [*extra_params, start.isoformat()],
                ).fetchall()
            }
            return [found.get((start + timedelta(days=i)).isoformat(), 0) for i in range(days)]

        thirty_days_ago = (date.today() - timedelta(days=30)).isoformat()
        stat_total_now = _count()
        stat_total_past = _count("AND created_at <= ?", (thirty_days_ago,))
        stat_enabled_now = _count("AND enabled = 1")
        stat_enabled_past = _count("AND enabled = 1 AND created_at <= ?", (thirty_days_ago,))
        stat_disabled_now = _count("AND enabled = 0")
        stat_disabled_past = _count("AND enabled = 0 AND created_at <= ?", (thirty_days_ago,))
        stat_admins_now = _count("AND role = 'admin'")
        stat_admins_past = _count("AND role = 'admin' AND created_at <= ?", (thirty_days_ago,))

        stat_total = {"value": stat_total_now, "spark": _sparkline(), **_trend(stat_total_now, stat_total_past)}
        stat_enabled = {
            "value": stat_enabled_now, "spark": _sparkline("AND enabled = 1"),
            **_trend(stat_enabled_now, stat_enabled_past),
        }
        stat_disabled = {
            "value": stat_disabled_now, "spark": _sparkline("AND enabled = 0"),
            **_trend(stat_disabled_now, stat_disabled_past),
        }
        stat_admins = {
            "value": stat_admins_now, "spark": _sparkline("AND role = 'admin'"),
            **_trend(stat_admins_now, stat_admins_past),
        }

    return templates.TemplateResponse(
        request,
        "admin_users.html",
        {
            "user": user,
            "users": rows,
            "q": q,
            "total": len(rows),
            "stat_total": stat_total,
            "stat_enabled": stat_enabled,
            "stat_disabled": stat_disabled,
            "stat_admins": stat_admins,
            "can_create": can(user["role"], "user", "create"),
            "can_update": can(user["role"], "user", "update"),
        },
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


@router.post("/users/bulk-enable")
def bulk_enable(request: Request, user_ids: list[str] = Form(...)):
    return _bulk_set_enabled(request, user_ids, 1)


@router.post("/users/bulk-disable")
def bulk_disable(request: Request, user_ids: list[str] = Form(...)):
    return _bulk_set_enabled(request, user_ids, 0)


def _bulk_set_enabled(request: Request, user_ids: list[str], new_state: int):
    """Same rules as toggle_enabled() (permission check + never let an admin enable/disable
    their own account through this route), applied over a checkbox selection from the list
    page. Unlike the single-row endpoint, an invalid, self, or not-found row is silently
    skipped rather than failing the whole batch — the checkbox selection is just "attempt
    these," not an all-or-nothing transaction (same shape as leave.py's _bulk_decide)."""
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    if not can(user["role"], "user", "update"):
        return _forbidden(request, user)

    now = datetime.now(timezone.utc).isoformat()
    with get_conn() as conn:
        for uid_str in user_ids:
            try:
                uid = int(uid_str)
            except ValueError:
                continue
            if uid == user["id"]:
                # never allow disabling (or re-enabling) your own currently-logged-in
                # account through this route — same guard as the single-row endpoint
                continue
            row = conn.execute("SELECT enabled, username FROM users WHERE id = ?", (uid,)).fetchone()
            if row is None:
                continue
            conn.execute(
                "UPDATE users SET enabled = ?, updated_at = ? WHERE id = ?", (new_state, now, uid)
            )
            log_audit(
                conn,
                user["username"],
                "enable" if new_state else "disable",
                "user",
                str(uid),
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

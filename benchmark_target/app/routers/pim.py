from datetime import date, datetime, timedelta, timezone

from fastapi import APIRouter, Form, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from benchmark_target.app.config import EMPLOYEES_PER_PAGE, TEMPLATES_DIR
from benchmark_target.app.db import get_conn, log_audit
from benchmark_target.app.deps import can, employee_in_scope, get_current_user, get_scope
from benchmark_target.app.seed import DEPARTMENTS, JOB_TITLES

router = APIRouter(prefix="/pim")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

SORT_COLUMNS = {"first_name", "last_name", "department", "hire_date", "employee_code"}
PAGE_SIZE_CHOICES = (10, 25, 50)

HIRE_RANGES = {
    "30d": ("Last 30 days", 30, None),
    "90d": ("Last 90 days", 90, None),
    "365d": ("Last 12 months", 365, None),
    "older": ("Older than 12 months", None, 365),
}


def _forbidden(request: Request, user: dict):
    return templates.TemplateResponse(request, "403.html", {"user": user}, status_code=403)


def _trend(now: int, past: int) -> dict:
    """A real, computed delta (never fabricated) between the current count and the count
    as of 30 days ago (via hire_date, the only historical signal this table has — status
    changes aren't timestamped). `past == 0` reads as +100% growth from nothing, not an
    undefined/divide-by-zero case."""
    if past == 0:
        return {"pct": 100, "direction": "up"} if now > 0 else {"pct": 0, "direction": "flat"}
    pct = round((now - past) / past * 100)
    direction = "up" if pct > 0 else "down" if pct < 0 else "flat"
    return {"pct": abs(pct), "direction": direction}


@router.get("/employees")
def list_employees(
    request: Request,
    q: str = "",
    department: str = "",
    status: str = "",
    hire_range: str = "",
    sort: str = "last_name",
    direction: str = "asc",
    page: int = 1,
    page_size: int = EMPLOYEES_PER_PAGE,
):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)

    scope = get_scope(user["role"], "employee", "read")
    if scope is None:
        return _forbidden(request, user)

    sort = sort if sort in SORT_COLUMNS else "last_name"
    direction = "DESC" if direction.lower() == "desc" else "ASC"
    page = max(1, page)
    page_size = page_size if page_size in PAGE_SIZE_CHOICES else EMPLOYEES_PER_PAGE

    scope_where = ["1=1"]
    scope_params: list = []
    if scope == "own":
        scope_where.append("id = ?")
        scope_params.append(user.get("employee_id"))
    elif scope == "direct_reports":
        scope_where.append("supervisor_id = ?")
        scope_params.append(user.get("employee_id"))
    # scope == "all": no extra restriction
    scope_where_sql = " AND ".join(scope_where)

    where = list(scope_where)
    params = list(scope_params)

    if q:
        where.append("(first_name LIKE ? OR last_name LIKE ? OR employee_code LIKE ?)")
        like = f"%{q}%"
        params.extend([like, like, like])
    if department:
        where.append("department = ?")
        params.append(department)
    if status:
        where.append("status = ?")
        params.append(status)
    if hire_range in HIRE_RANGES:
        _, newer_than_days, older_than_days = HIRE_RANGES[hire_range]
        if newer_than_days is not None:
            where.append("hire_date >= ?")
            params.append((date.today() - timedelta(days=newer_than_days)).isoformat())
        if older_than_days is not None:
            where.append("hire_date < ?")
            params.append((date.today() - timedelta(days=older_than_days)).isoformat())

    where_sql = " AND ".join(where)

    with get_conn() as conn:
        total = conn.execute(
            f"SELECT COUNT(*) c FROM employees WHERE {where_sql}", params
        ).fetchone()["c"]
        total_pages = max(1, (total + page_size - 1) // page_size)
        page = min(page, total_pages)
        offset = (page - 1) * page_size

        rows = [
            dict(row, initial=row["first_name"][:1].upper())
            for row in conn.execute(
                f"""SELECT * FROM employees WHERE {where_sql}
                    ORDER BY {sort} {direction}
                    LIMIT ? OFFSET ?""",
                [*params, page_size, offset],
            ).fetchall()
        ]

        # Stat row: scope-restricted (a supervisor sees stats about their own team, not the
        # whole org) but otherwise unfiltered by the search/department/status/date fields
        # above — it's a summary of what this view covers, not of the current search result.
        thirty_days_ago = (date.today() - timedelta(days=30)).isoformat()

        def _count(extra_sql: str = "", extra_params: tuple = ()) -> int:
            return conn.execute(
                f"SELECT COUNT(*) c FROM employees WHERE {scope_where_sql} {extra_sql}",
                [*scope_params, *extra_params],
            ).fetchone()["c"]

        total_now = _count()
        total_past = _count("AND hire_date <= ?", (thirty_days_ago,))
        active_now = _count("AND status = 'active'")
        active_past = _count("AND status = 'active' AND hire_date <= ?", (thirty_days_ago,))
        inactive_now = _count("AND status = 'terminated'")
        inactive_past = _count("AND status = 'terminated' AND hire_date <= ?", (thirty_days_ago,))
        departments_count = conn.execute(
            f"SELECT COUNT(DISTINCT department) c FROM employees WHERE {scope_where_sql} AND status = 'active'",
            scope_params,
        ).fetchone()["c"]

    return templates.TemplateResponse(
        request,
        "pim_list.html",
        {
            "user": user,
            "employees": rows,
            "departments": DEPARTMENTS,
            "hire_ranges": HIRE_RANGES,
            "page_size_choices": PAGE_SIZE_CHOICES,
            "q": q,
            "department": department,
            "status": status,
            "hire_range": hire_range,
            "sort": sort,
            "direction": direction.lower(),
            "page": page,
            "page_size": page_size,
            "total_pages": total_pages,
            "total": total,
            "showing_from": offset + 1 if rows else 0,
            "showing_to": offset + len(rows),
            "can_create": can(user["role"], "employee", "create"),
            "can_edit": can(user["role"], "employee", "update"),
            "can_deactivate": can(user["role"], "employee", "deactivate"),
            "stat_total": {"value": total_now, **_trend(total_now, total_past)},
            "stat_active": {"value": active_now, **_trend(active_now, active_past)},
            "stat_inactive": {"value": inactive_now, **_trend(inactive_now, inactive_past)},
            "stat_departments": departments_count,
        },
    )


@router.get("/employees/new")
def new_employee_form(request: Request):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    if not can(user["role"], "employee", "create"):
        return _forbidden(request, user)

    with get_conn() as conn:
        supervisors = conn.execute(
            "SELECT id, first_name, last_name, department FROM employees WHERE status = 'active' ORDER BY last_name"
        ).fetchall()

    return templates.TemplateResponse(
        request,
        "pim_form.html",
        {
            "user": user,
            "employee": None,
            "departments": DEPARTMENTS,
            "job_titles": JOB_TITLES,
            "supervisors": supervisors,
            "error": None,
        },
    )


@router.post("/employees/new")
def create_employee(
    request: Request,
    first_name: str = Form(...),
    last_name: str = Form(...),
    department: str = Form(...),
    job_title: str = Form(...),
    supervisor_id: str = Form(""),
    hire_date: str = Form(...),
    leave_balance_annual: int = Form(18),
):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    if not can(user["role"], "employee", "create"):
        return _forbidden(request, user)

    now = datetime.now(timezone.utc).isoformat()
    with get_conn() as conn:
        next_num = conn.execute("SELECT COUNT(*) c FROM employees").fetchone()["c"] + 1
        employee_code = f"EMP-{next_num:04d}"
        while conn.execute(
            "SELECT 1 FROM employees WHERE employee_code = ?", (employee_code,)
        ).fetchone():
            next_num += 1
            employee_code = f"EMP-{next_num:04d}"

        cur = conn.execute(
            """INSERT INTO employees
               (employee_code, first_name, last_name, department, job_title, supervisor_id,
                status, hire_date, leave_balance_annual, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, ?)""",
            (
                employee_code,
                first_name.strip(),
                last_name.strip(),
                department,
                job_title,
                int(supervisor_id) if supervisor_id else None,
                hire_date,
                leave_balance_annual,
                now,
                now,
            ),
        )
        log_audit(conn, user["username"], "create", "employee", str(cur.lastrowid), employee_code)

    return RedirectResponse(f"/pim/employees/{cur.lastrowid}", status_code=303)


@router.get("/employees/{employee_id}")
def view_employee(request: Request, employee_id: int):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)

    scope = get_scope(user["role"], "employee", "read")
    if scope is None:
        return _forbidden(request, user)

    with get_conn() as conn:
        employee = conn.execute("SELECT * FROM employees WHERE id = ?", (employee_id,)).fetchone()
        if employee is None:
            return templates.TemplateResponse(
                request, "404.html", {"user": user, "what": "Employee"}, status_code=404
            )
        if not employee_in_scope(user, employee, scope):
            return _forbidden(request, user)

        supervisor = None
        if employee["supervisor_id"]:
            supervisor = conn.execute(
                "SELECT id, first_name, last_name FROM employees WHERE id = ?",
                (employee["supervisor_id"],),
            ).fetchone()

    return templates.TemplateResponse(
        request,
        "pim_view.html",
        {
            "user": user,
            "employee": employee,
            "supervisor": supervisor,
            "can_edit": can(user["role"], "employee", "update") and employee_in_scope(
                user, employee, get_scope(user["role"], "employee", "update") or ""
            ),
            "can_deactivate": can(user["role"], "employee", "deactivate"),
        },
    )


@router.get("/employees/{employee_id}/edit")
def edit_employee_form(request: Request, employee_id: int):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    if not can(user["role"], "employee", "update"):
        return _forbidden(request, user)

    with get_conn() as conn:
        employee = conn.execute("SELECT * FROM employees WHERE id = ?", (employee_id,)).fetchone()
        if employee is None:
            return templates.TemplateResponse(
                request, "404.html", {"user": user, "what": "Employee"}, status_code=404
            )
        scope = get_scope(user["role"], "employee", "update")
        if not employee_in_scope(user, employee, scope):
            return _forbidden(request, user)

        supervisors = conn.execute(
            "SELECT id, first_name, last_name, department FROM employees WHERE status = 'active' AND id != ? ORDER BY last_name",
            (employee_id,),
        ).fetchall()

    return templates.TemplateResponse(
        request,
        "pim_form.html",
        {
            "user": user,
            "employee": employee,
            "departments": DEPARTMENTS,
            "job_titles": JOB_TITLES,
            "supervisors": supervisors,
            "error": None,
        },
    )


@router.post("/employees/{employee_id}/edit")
def update_employee(
    request: Request,
    employee_id: int,
    first_name: str = Form(...),
    last_name: str = Form(...),
    department: str = Form(...),
    job_title: str = Form(...),
    supervisor_id: str = Form(""),
    hire_date: str = Form(...),
    leave_balance_annual: int = Form(18),
    version: str = Form(""),
):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    if not can(user["role"], "employee", "update"):
        return _forbidden(request, user)

    now = datetime.now(timezone.utc).isoformat()
    with get_conn() as conn:
        employee = conn.execute("SELECT * FROM employees WHERE id = ?", (employee_id,)).fetchone()
        if employee is None:
            return templates.TemplateResponse(
                request, "404.html", {"user": user, "what": "Employee"}, status_code=404
            )
        scope = get_scope(user["role"], "employee", "update")
        if not employee_in_scope(user, employee, scope):
            return _forbidden(request, user)

        # optimistic concurrency (spec 20/18): reject a stale form silently overwriting a
        # newer update — the edit form carries the updated_at it was loaded with as `version`.
        if version and version != employee["updated_at"]:
            supervisors = conn.execute(
                "SELECT id, first_name, last_name, department FROM employees WHERE status = 'active' AND id != ? ORDER BY last_name",
                (employee_id,),
            ).fetchall()
            fresh = conn.execute("SELECT * FROM employees WHERE id = ?", (employee_id,)).fetchone()
            return templates.TemplateResponse(
                request,
                "pim_form.html",
                {
                    "user": user,
                    "employee": fresh,
                    "departments": DEPARTMENTS,
                    "job_titles": JOB_TITLES,
                    "supervisors": supervisors,
                    "error": "This record was updated by someone else since you opened it. "
                             "Review the current values below and save again.",
                },
                status_code=409,
            )

        conn.execute(
            """UPDATE employees SET first_name=?, last_name=?, department=?, job_title=?,
               supervisor_id=?, hire_date=?, leave_balance_annual=?, updated_at=?
               WHERE id=?""",
            (
                first_name.strip(),
                last_name.strip(),
                department,
                job_title,
                int(supervisor_id) if supervisor_id else None,
                hire_date,
                leave_balance_annual,
                now,
                employee_id,
            ),
        )
        log_audit(conn, user["username"], "update", "employee", str(employee_id), None)

    return RedirectResponse(f"/pim/employees/{employee_id}", status_code=303)


@router.post("/employees/{employee_id}/deactivate")
def deactivate_employee(request: Request, employee_id: int, confirm: str = Form("")):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    if not can(user["role"], "employee", "deactivate"):
        return _forbidden(request, user)
    if confirm != "yes":
        return RedirectResponse(f"/pim/employees/{employee_id}", status_code=303)

    now = datetime.now(timezone.utc).isoformat()
    with get_conn() as conn:
        conn.execute(
            "UPDATE employees SET status = 'terminated', updated_at = ? WHERE id = ?",
            (now, employee_id),
        )
        log_audit(conn, user["username"], "deactivate", "employee", str(employee_id), None)

    return RedirectResponse(f"/pim/employees/{employee_id}", status_code=303)


@router.post("/employees/{employee_id}/activate")
def activate_employee(request: Request, employee_id: int):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    if not can(user["role"], "employee", "deactivate"):
        return _forbidden(request, user)

    now = datetime.now(timezone.utc).isoformat()
    with get_conn() as conn:
        conn.execute(
            "UPDATE employees SET status = 'active', updated_at = ? WHERE id = ?",
            (now, employee_id),
        )
        log_audit(conn, user["username"], "activate", "employee", str(employee_id), None)

    return RedirectResponse(f"/pim/employees/{employee_id}", status_code=303)

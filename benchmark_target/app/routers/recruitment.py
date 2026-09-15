from datetime import date, datetime, timezone

from fastapi import APIRouter, Form, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from benchmark_target.app.config import TEMPLATES_DIR
from benchmark_target.app.db import get_conn, log_audit
from benchmark_target.app.deps import can, get_current_user

router = APIRouter(prefix="/recruitment")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def _forbidden(request: Request, user: dict):
    return templates.TemplateResponse(request, "403.html", {"user": user}, status_code=403)


def _guard(request: Request, action: str = "read"):
    user = get_current_user(request)
    if not user:
        return None, RedirectResponse("/login", status_code=303)
    if not can(user["role"], "recruitment", action):
        return None, _forbidden(request, user)
    return user, None


@router.get("/vacancies")
def list_vacancies(request: Request):
    user, err = _guard(request)
    if err:
        return err

    with get_conn() as conn:
        vacancies = conn.execute(
            """SELECT v.*, (SELECT COUNT(*) FROM candidates c WHERE c.vacancy_id = v.id) AS candidate_count
               FROM vacancies v ORDER BY v.status, v.title"""
        ).fetchall()

    return templates.TemplateResponse(
        request, "recruitment_vacancies.html",
        {"user": user, "vacancies": vacancies, "can_manage": can(user["role"], "recruitment", "manage")},
    )


@router.post("/vacancies/{vacancy_id}/toggle-status")
def toggle_vacancy(request: Request, vacancy_id: int):
    user, err = _guard(request, "manage")
    if err:
        return err

    with get_conn() as conn:
        row = conn.execute("SELECT status FROM vacancies WHERE id = ?", (vacancy_id,)).fetchone()
        if row is None:
            return templates.TemplateResponse(request, "404.html", {"user": user, "what": "Vacancy"}, status_code=404)
        new_status = "closed" if row["status"] == "open" else "open"
        conn.execute("UPDATE vacancies SET status = ? WHERE id = ?", (new_status, vacancy_id))
        log_audit(conn, user["username"], "toggle_status", "vacancy", str(vacancy_id), new_status)

    return RedirectResponse("/recruitment/vacancies", status_code=303)


@router.get("/vacancies/{vacancy_id}")
def view_vacancy(request: Request, vacancy_id: int):
    user, err = _guard(request)
    if err:
        return err

    with get_conn() as conn:
        vacancy = conn.execute("SELECT * FROM vacancies WHERE id = ?", (vacancy_id,)).fetchone()
        if vacancy is None:
            return templates.TemplateResponse(request, "404.html", {"user": user, "what": "Vacancy"}, status_code=404)
        candidates = conn.execute(
            "SELECT * FROM candidates WHERE vacancy_id = ? ORDER BY status, last_name", (vacancy_id,)
        ).fetchall()

    return templates.TemplateResponse(
        request, "recruitment_vacancy_detail.html",
        {"user": user, "vacancy": vacancy, "candidates": candidates, "can_manage": can(user["role"], "recruitment", "manage")},
    )


@router.get("/candidates/{candidate_id}")
def view_candidate(request: Request, candidate_id: int):
    user, err = _guard(request)
    if err:
        return err

    with get_conn() as conn:
        candidate = conn.execute(
            """SELECT c.*, v.title AS vacancy_title, v.department AS vacancy_department
               FROM candidates c JOIN vacancies v ON v.id = c.vacancy_id WHERE c.id = ?""",
            (candidate_id,),
        ).fetchone()
        if candidate is None:
            return templates.TemplateResponse(request, "404.html", {"user": user, "what": "Candidate"}, status_code=404)
        interviews = conn.execute(
            """SELECT i.*, e.first_name, e.last_name FROM interviews i
               LEFT JOIN employees e ON e.id = i.interviewer_employee_id
               WHERE i.candidate_id = ? ORDER BY i.scheduled_at""",
            (candidate_id,),
        ).fetchall()
        hired_employee = None
        if candidate["hired_employee_id"]:
            hired_employee = conn.execute(
                "SELECT * FROM employees WHERE id = ?", (candidate["hired_employee_id"],)
            ).fetchone()
        interviewers = conn.execute(
            "SELECT id, first_name, last_name FROM employees WHERE status = 'active' ORDER BY last_name"
        ).fetchall()

    return templates.TemplateResponse(
        request, "recruitment_candidate_detail.html",
        {
            "user": user,
            "candidate": candidate,
            "interviews": interviews,
            "hired_employee": hired_employee,
            "interviewers": interviewers,
            "can_manage": can(user["role"], "recruitment", "manage"),
        },
    )


@router.post("/candidates/{candidate_id}/status")
def change_candidate_status(request: Request, candidate_id: int, status: str = Form(...)):
    user, err = _guard(request, "manage")
    if err:
        return err
    if status not in ("applied", "shortlisted", "interview", "rejected"):
        return RedirectResponse(f"/recruitment/candidates/{candidate_id}", status_code=303)

    now = datetime.now(timezone.utc).isoformat()
    with get_conn() as conn:
        row = conn.execute("SELECT status FROM candidates WHERE id = ?", (candidate_id,)).fetchone()
        if row is None:
            return templates.TemplateResponse(request, "404.html", {"user": user, "what": "Candidate"}, status_code=404)
        if row["status"] == "hired":
            # hired is terminal — no further status changes through this route
            return RedirectResponse(f"/recruitment/candidates/{candidate_id}", status_code=303)

        conn.execute("UPDATE candidates SET status = ?, updated_at = ? WHERE id = ?", (status, now, candidate_id))
        log_audit(conn, user["username"], "status_change", "candidate", str(candidate_id), status)

    return RedirectResponse(f"/recruitment/candidates/{candidate_id}", status_code=303)


@router.post("/candidates/{candidate_id}/interview")
def schedule_interview(
    request: Request, candidate_id: int, scheduled_at: str = Form(...), interviewer_employee_id: str = Form("")
):
    user, err = _guard(request, "manage")
    if err:
        return err

    now = datetime.now(timezone.utc).isoformat()
    with get_conn() as conn:
        candidate = conn.execute("SELECT status FROM candidates WHERE id = ?", (candidate_id,)).fetchone()
        if candidate is None:
            return templates.TemplateResponse(request, "404.html", {"user": user, "what": "Candidate"}, status_code=404)

        conn.execute(
            "INSERT INTO interviews (candidate_id, scheduled_at, interviewer_employee_id, created_at) VALUES (?, ?, ?, ?)",
            (candidate_id, scheduled_at, int(interviewer_employee_id) if interviewer_employee_id else None, now),
        )
        conn.execute(
            "UPDATE candidates SET status = 'interview', updated_at = ? WHERE id = ?", (now, candidate_id)
        )
        log_audit(conn, user["username"], "schedule_interview", "candidate", str(candidate_id), scheduled_at)

    return RedirectResponse(f"/recruitment/candidates/{candidate_id}", status_code=303)


@router.post("/candidates/{candidate_id}/hire")
def hire_candidate(request: Request, candidate_id: int):
    user, err = _guard(request, "manage")
    if err:
        return err

    now = datetime.now(timezone.utc).isoformat()
    with get_conn() as conn:
        candidate = conn.execute(
            """SELECT c.*, v.title AS vacancy_title, v.department AS vacancy_department
               FROM candidates c JOIN vacancies v ON v.id = c.vacancy_id WHERE c.id = ?""",
            (candidate_id,),
        ).fetchone()
        if candidate is None:
            return templates.TemplateResponse(request, "404.html", {"user": user, "what": "Candidate"}, status_code=404)
        if candidate["status"] not in ("shortlisted", "interview"):
            return RedirectResponse(f"/recruitment/candidates/{candidate_id}", status_code=303)

        # Cross-module transition (spec 3.7): Candidate -> Hire -> Employee.
        next_num = conn.execute("SELECT COUNT(*) c FROM employees").fetchone()["c"] + 1
        employee_code = f"EMP-{next_num:04d}"
        while conn.execute("SELECT 1 FROM employees WHERE employee_code = ?", (employee_code,)).fetchone():
            next_num += 1
            employee_code = f"EMP-{next_num:04d}"

        cur = conn.execute(
            """INSERT INTO employees
               (employee_code, first_name, last_name, department, job_title, supervisor_id,
                status, hire_date, leave_balance_annual, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, NULL, 'active', ?, 18, ?, ?)""",
            (
                employee_code,
                candidate["first_name"],
                candidate["last_name"],
                candidate["vacancy_department"],
                candidate["vacancy_title"],
                date.today().isoformat(),
                now,
                now,
            ),
        )
        new_employee_id = cur.lastrowid

        default_balances = {"Annual": 18, "Sick": 10, "Unpaid": 999}
        for lt_row in conn.execute("SELECT id, name FROM leave_types").fetchall():
            conn.execute(
                "INSERT INTO leave_balances (employee_id, leave_type_id, balance_days) VALUES (?, ?, ?)",
                (new_employee_id, lt_row["id"], default_balances.get(lt_row["name"], 0)),
            )

        conn.execute(
            "UPDATE candidates SET status = 'hired', hired_employee_id = ?, updated_at = ? WHERE id = ?",
            (new_employee_id, now, candidate_id),
        )
        log_audit(conn, user["username"], "hire", "candidate", str(candidate_id), employee_code)

    return RedirectResponse(f"/pim/employees/{new_employee_id}", status_code=303)

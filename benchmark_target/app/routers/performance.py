from datetime import datetime, timezone

from fastapi import APIRouter, Form, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from benchmark_target.app.config import TEMPLATES_DIR
from benchmark_target.app.db import get_conn, log_audit
from benchmark_target.app.deps import can, get_current_user, get_scope

router = APIRouter(prefix="/performance")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def _forbidden(request: Request, user: dict):
    return templates.TemplateResponse(request, "403.html", {"user": user}, status_code=403)


def _review_in_scope(user: dict, row, scope: str) -> bool:
    if scope == "all":
        return True
    if scope == "own":
        return row["employee_id"] == user.get("employee_id")
    if scope == "direct_reports":
        return row["supervisor_id"] == user.get("employee_id")
    return False


def _is_manager_of(user: dict, manage_scope: str | None, row) -> bool:
    """Manager-review authority: admin's 'all' scope covers every employee, a
    supervisor's 'direct_reports' scope only their own reports."""
    if manage_scope == "all":
        return True
    if manage_scope == "direct_reports":
        return row["supervisor_id"] == user.get("employee_id")
    return False


@router.get("/reviews")
def list_reviews(request: Request, view: str = "mine"):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)

    scope = get_scope(user["role"], "performance", "read")
    if scope is None:
        return _forbidden(request, user)

    show_team = view == "team" and scope in ("direct_reports", "all")

    with get_conn() as conn:
        if show_team:
            if scope == "all":
                rows = conn.execute(
                    """SELECT r.*, e.first_name, e.last_name, rc.name AS cycle_name FROM reviews r
                       JOIN employees e ON e.id = r.employee_id
                       JOIN review_cycles rc ON rc.id = r.cycle_id
                       ORDER BY rc.id DESC, e.last_name"""
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT r.*, e.first_name, e.last_name, rc.name AS cycle_name FROM reviews r
                       JOIN employees e ON e.id = r.employee_id
                       JOIN review_cycles rc ON rc.id = r.cycle_id
                       WHERE e.supervisor_id = ? ORDER BY rc.id DESC, e.last_name""",
                    (user["employee_id"],),
                ).fetchall()
        elif user.get("employee_id"):
            rows = conn.execute(
                """SELECT r.*, e.first_name, e.last_name, rc.name AS cycle_name FROM reviews r
                   JOIN employees e ON e.id = r.employee_id
                   JOIN review_cycles rc ON rc.id = r.cycle_id
                   WHERE r.employee_id = ? ORDER BY rc.id DESC""",
                (user["employee_id"],),
            ).fetchall()
        else:
            rows = []

    return templates.TemplateResponse(
        request,
        "performance_list.html",
        {
            "user": user,
            "reviews": rows,
            "view": "team" if show_team else "mine",
            "can_view_team": scope in ("direct_reports", "all"),
        },
    )


@router.get("/reviews/{review_id}")
def view_review(request: Request, review_id: int):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)

    with get_conn() as conn:
        row = conn.execute(
            """SELECT r.*, e.first_name, e.last_name, e.supervisor_id, rc.name AS cycle_name, rc.status AS cycle_status
               FROM reviews r JOIN employees e ON e.id = r.employee_id
               JOIN review_cycles rc ON rc.id = r.cycle_id WHERE r.id = ?""",
            (review_id,),
        ).fetchone()
        if row is None:
            return templates.TemplateResponse(request, "404.html", {"user": user, "what": "Review"}, status_code=404)

        read_scope = get_scope(user["role"], "performance", "read")
        manage_scope = get_scope(user["role"], "performance", "manage")
        is_self = row["employee_id"] == user.get("employee_id")
        is_manager = _is_manager_of(user, manage_scope, row)
        can_view = is_self or is_manager or (read_scope and _review_in_scope(user, row, read_scope))
        if not can_view:
            return _forbidden(request, user)

    can_self_review = is_self and row["status"] == "self_pending" and row["cycle_status"] == "open"
    can_manager_review = is_manager and row["status"] == "manager_pending" and row["cycle_status"] == "open"

    return templates.TemplateResponse(
        request,
        "performance_view.html",
        {"user": user, "review": row, "can_self_review": can_self_review, "can_manager_review": can_manager_review},
    )


@router.post("/reviews/{review_id}/self")
def submit_self_review(request: Request, review_id: int, self_rating: int = Form(...), self_comment: str = Form("")):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)

    with get_conn() as conn:
        row = conn.execute(
            """SELECT r.*, rc.status AS cycle_status FROM reviews r
               JOIN review_cycles rc ON rc.id = r.cycle_id WHERE r.id = ?""",
            (review_id,),
        ).fetchone()
        if row is None:
            return templates.TemplateResponse(request, "404.html", {"user": user, "what": "Review"}, status_code=404)
        if row["employee_id"] != user.get("employee_id") or row["status"] != "self_pending" or row["cycle_status"] != "open":
            return _forbidden(request, user)
        if not (1 <= self_rating <= 5):
            return _forbidden(request, user)

        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            """UPDATE reviews SET self_rating = ?, self_comment = ?, status = 'manager_pending', updated_at = ?
               WHERE id = ?""",
            (self_rating, self_comment, now, review_id),
        )
        log_audit(conn, user["username"], "self_review", "review", str(review_id), None)

    return RedirectResponse(f"/performance/reviews/{review_id}", status_code=303)


@router.post("/reviews/{review_id}/manager")
def submit_manager_review(request: Request, review_id: int, manager_rating: int = Form(...), manager_comment: str = Form("")):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)

    manage_scope = get_scope(user["role"], "performance", "manage")
    if manage_scope is None:
        return _forbidden(request, user)

    with get_conn() as conn:
        row = conn.execute(
            """SELECT r.*, e.supervisor_id, rc.status AS cycle_status FROM reviews r
               JOIN employees e ON e.id = r.employee_id
               JOIN review_cycles rc ON rc.id = r.cycle_id WHERE r.id = ?""",
            (review_id,),
        ).fetchone()
        if row is None:
            return templates.TemplateResponse(request, "404.html", {"user": user, "what": "Review"}, status_code=404)
        is_manager = _is_manager_of(user, manage_scope, row)
        if not is_manager or row["status"] != "manager_pending" or row["cycle_status"] != "open":
            return _forbidden(request, user)
        if not (1 <= manager_rating <= 5):
            return _forbidden(request, user)

        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            """UPDATE reviews SET manager_rating = ?, manager_comment = ?, status = 'finalized', updated_at = ?
               WHERE id = ?""",
            (manager_rating, manager_comment, now, review_id),
        )
        log_audit(conn, user["username"], "manager_review", "review", str(review_id), None)

    return RedirectResponse(f"/performance/reviews/{review_id}", status_code=303)


@router.get("/cycles")
def list_cycles(request: Request):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    if not can(user["role"], "performance", "finalize"):
        return _forbidden(request, user)

    with get_conn() as conn:
        cycles = conn.execute(
            """SELECT rc.*,
                      (SELECT COUNT(*) FROM reviews r WHERE r.cycle_id = rc.id) AS total,
                      (SELECT COUNT(*) FROM reviews r WHERE r.cycle_id = rc.id AND r.status = 'finalized') AS finalized
               FROM review_cycles rc ORDER BY rc.id DESC"""
        ).fetchall()

    return templates.TemplateResponse(request, "performance_cycles.html", {"user": user, "cycles": cycles})


@router.post("/cycles/{cycle_id}/close")
def close_cycle(request: Request, cycle_id: int):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    if not can(user["role"], "performance", "finalize"):
        return _forbidden(request, user)

    with get_conn() as conn:
        conn.execute("UPDATE review_cycles SET status = 'closed' WHERE id = ?", (cycle_id,))
        log_audit(conn, user["username"], "close_cycle", "review_cycle", str(cycle_id), None)

    return RedirectResponse("/performance/cycles", status_code=303)

from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Request, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from benchmark_target.app.config import DATA_DIR, TEMPLATES_DIR
from benchmark_target.app.db import get_conn, log_audit
from benchmark_target.app.deps import employee_in_scope, get_current_user, get_scope

router = APIRouter(prefix="/documents")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

UPLOAD_DIR = DATA_DIR / "uploads"
ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".txt", ".docx"}
MAX_SIZE_BYTES = 2 * 1024 * 1024  # 2 MB


def _forbidden(request: Request, user: dict):
    return templates.TemplateResponse(request, "403.html", {"user": user}, status_code=403)


def _employee_scope_check(conn, user: dict, employee_id: int, action: str) -> bool:
    scope = get_scope(user["role"], "document", action)
    if scope is None:
        return False
    employee = conn.execute("SELECT * FROM employees WHERE id = ?", (employee_id,)).fetchone()
    if employee is None:
        return False
    return employee_in_scope(user, employee, scope)


@router.get("/employees/{employee_id}")
def list_documents(request: Request, employee_id: int):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)

    with get_conn() as conn:
        if not _employee_scope_check(conn, user, employee_id, "read"):
            return _forbidden(request, user)
        employee = conn.execute("SELECT * FROM employees WHERE id = ?", (employee_id,)).fetchone()
        docs = conn.execute(
            "SELECT * FROM documents WHERE employee_id = ? ORDER BY uploaded_at DESC", (employee_id,)
        ).fetchall()
        can_manage = _employee_scope_check(conn, user, employee_id, "manage")

    return templates.TemplateResponse(
        request,
        "documents_list.html",
        {
            "user": user,
            "employee": employee,
            "documents": docs,
            "can_manage": can_manage,
            "max_size_mb": MAX_SIZE_BYTES // (1024 * 1024),
            "allowed_extensions": ", ".join(sorted(ALLOWED_EXTENSIONS)),
        },
    )


def _reject(request, user, employee, docs, can_manage, error):
    return templates.TemplateResponse(
        request,
        "documents_list.html",
        {
            "user": user,
            "employee": employee,
            "documents": docs,
            "can_manage": can_manage,
            "max_size_mb": MAX_SIZE_BYTES // (1024 * 1024),
            "allowed_extensions": ", ".join(sorted(ALLOWED_EXTENSIONS)),
            "error": error,
        },
        status_code=422,
    )


@router.post("/employees/{employee_id}/upload")
async def upload_document(request: Request, employee_id: int, file: UploadFile):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)

    with get_conn() as conn:
        if not _employee_scope_check(conn, user, employee_id, "manage"):
            return _forbidden(request, user)

        employee = conn.execute("SELECT * FROM employees WHERE id = ?", (employee_id,)).fetchone()
        docs = conn.execute(
            "SELECT * FROM documents WHERE employee_id = ? ORDER BY uploaded_at DESC", (employee_id,)
        ).fetchall()

        filename = Path(file.filename or "").name
        ext = Path(filename).suffix.lower()
        if not filename:
            return _reject(request, user, employee, docs, True, "No file selected.")
        if ext not in ALLOWED_EXTENSIONS:
            return _reject(
                request, user, employee, docs, True,
                f'File type "{ext}" is not allowed. Allowed: {", ".join(sorted(ALLOWED_EXTENSIONS))}',
            )

        existing = conn.execute(
            "SELECT 1 FROM documents WHERE employee_id = ? AND filename = ?", (employee_id, filename)
        ).fetchone()
        if existing:
            return _reject(
                request, user, employee, docs, True,
                f'A document named "{filename}" already exists for this employee. Delete it first, or rename the file before uploading.',
            )

        content = await file.read()
        if len(content) > MAX_SIZE_BYTES:
            return _reject(
                request, user, employee, docs, True,
                f"File too large ({len(content) // 1024} KB). Maximum is {MAX_SIZE_BYTES // 1024} KB.",
            )

        emp_dir = UPLOAD_DIR / str(employee_id)
        emp_dir.mkdir(parents=True, exist_ok=True)
        stored_path = emp_dir / filename
        stored_path.write_bytes(content)

        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            """INSERT INTO documents (employee_id, filename, stored_path, mime_type, size_bytes, uploaded_by, uploaded_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (employee_id, filename, str(stored_path), file.content_type or "application/octet-stream",
             len(content), user["username"], now),
        )
        log_audit(conn, user["username"], "upload", "document", filename, f"employee_id={employee_id}")

    return RedirectResponse(f"/documents/employees/{employee_id}", status_code=303)


@router.get("/{document_id}/download")
def download_document(request: Request, document_id: int):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)

    with get_conn() as conn:
        doc = conn.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
        if doc is None:
            return templates.TemplateResponse(request, "404.html", {"user": user, "what": "Document"}, status_code=404)
        if not _employee_scope_check(conn, user, doc["employee_id"], "read"):
            return _forbidden(request, user)

    return FileResponse(doc["stored_path"], filename=doc["filename"], media_type=doc["mime_type"])


@router.post("/{document_id}/delete")
def delete_document(request: Request, document_id: int):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)

    with get_conn() as conn:
        doc = conn.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
        if doc is None:
            return templates.TemplateResponse(request, "404.html", {"user": user, "what": "Document"}, status_code=404)
        if not _employee_scope_check(conn, user, doc["employee_id"], "manage"):
            return _forbidden(request, user)

        Path(doc["stored_path"]).unlink(missing_ok=True)
        conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        log_audit(conn, user["username"], "delete", "document", doc["filename"], f"employee_id={doc['employee_id']}")
        employee_id = doc["employee_id"]

    return RedirectResponse(f"/documents/employees/{employee_id}", status_code=303)

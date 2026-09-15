"""Task-specific post-condition verifiers (spec 16, Layer 1). Registry of named checks —
"did the requested goal actually happen" — keyed the same way the spec's own task YAML
names them (`verifier: leave.approval.completed`). Identified by employee_code / email /
cycle name, never raw row ids, so a verifier stays valid across a reset (ids are not
reproducible across reseeds; codes are).
"""

import sqlite3

# Columns safe to check via employee.field_equals — an explicit allowlist, not raw column
# interpolation, so this can never become a SQL-injection vector via task catalog content.
_EMPLOYEE_FIELDS = {
    "department", "job_title", "status", "supervisor_id", "hire_date", "leave_balance_annual",
}


def _employee_id_by_code(conn: sqlite3.Connection, employee_code: str) -> int | None:
    row = conn.execute("SELECT id FROM employees WHERE employee_code = ?", (employee_code,)).fetchone()
    return row["id"] if row else None


def verify_employee_field_equals(conn: sqlite3.Connection, args: dict) -> tuple[bool, str]:
    field = args["field"]
    if field not in _EMPLOYEE_FIELDS:
        return False, f"field '{field}' is not verifiable (allowed: {sorted(_EMPLOYEE_FIELDS)})"
    row = conn.execute(
        f"SELECT {field} AS v FROM employees WHERE employee_code = ?", (args["employee_code"],)
    ).fetchone()
    if row is None:
        return False, f"employee {args['employee_code']} does not exist"
    actual = row["v"]
    expected = args["value"]
    return actual == expected, f"{field}={actual!r} (expected {expected!r})"


def verify_leave_request_status(conn: sqlite3.Connection, args: dict) -> tuple[bool, str]:
    emp_id = _employee_id_by_code(conn, args["employee_code"])
    if emp_id is None:
        return False, f"employee {args['employee_code']} does not exist"
    row = conn.execute(
        """SELECT lr.status FROM leave_requests lr
           JOIN leave_types lt ON lt.id = lr.leave_type_id
           WHERE lr.employee_id = ? AND lt.name = ? AND lr.start_date = ?
           ORDER BY lr.id DESC LIMIT 1""",
        (emp_id, args["leave_type"], args["start_date"]),
    ).fetchone()
    if row is None:
        return False, "no matching leave request found"
    return row["status"] == args["expected_status"], f"status={row['status']} (expected {args['expected_status']})"


def verify_leave_approval_completed(conn: sqlite3.Connection, args: dict) -> tuple[bool, str]:
    return verify_leave_request_status(conn, {**args, "expected_status": "approved"})


def verify_timesheet_status(conn: sqlite3.Connection, args: dict) -> tuple[bool, str]:
    emp_id = _employee_id_by_code(conn, args["employee_code"])
    if emp_id is None:
        return False, f"employee {args['employee_code']} does not exist"
    row = conn.execute(
        "SELECT status FROM timesheets WHERE employee_id = ? AND week_start_date = ?",
        (emp_id, args["week_start_date"]),
    ).fetchone()
    if row is None:
        return False, "no matching timesheet found"
    return row["status"] == args["expected_status"], f"status={row['status']} (expected {args['expected_status']})"


def verify_review_finalized(conn: sqlite3.Connection, args: dict) -> tuple[bool, str]:
    emp_id = _employee_id_by_code(conn, args["employee_code"])
    if emp_id is None:
        return False, f"employee {args['employee_code']} does not exist"
    row = conn.execute(
        """SELECT r.status FROM reviews r JOIN review_cycles rc ON rc.id = r.cycle_id
           WHERE r.employee_id = ? AND rc.name = ?""",
        (emp_id, args["cycle_name"]),
    ).fetchone()
    if row is None:
        return False, "no matching review found"
    return row["status"] == "finalized", f"status={row['status']} (expected finalized)"


def verify_candidate_hired(conn: sqlite3.Connection, args: dict) -> tuple[bool, str]:
    row = conn.execute(
        "SELECT status, hired_employee_id FROM candidates WHERE email = ?", (args["candidate_email"],)
    ).fetchone()
    if row is None:
        return False, f"candidate {args['candidate_email']} does not exist"
    passed = row["status"] == "hired" and row["hired_employee_id"] is not None
    return passed, f"status={row['status']}, hired_employee_id={row['hired_employee_id']}"


def verify_document_exists(conn: sqlite3.Connection, args: dict) -> tuple[bool, str]:
    emp_id = _employee_id_by_code(conn, args["employee_code"])
    if emp_id is None:
        return False, f"employee {args['employee_code']} does not exist"
    row = conn.execute(
        "SELECT 1 FROM documents WHERE employee_id = ? AND filename = ?", (emp_id, args["filename"])
    ).fetchone()
    return row is not None, f"document {args['filename']} {'found' if row else 'not found'}"


VERIFIERS = {
    "employee.field_equals": verify_employee_field_equals,
    "leave.request.status": verify_leave_request_status,
    "leave.approval.completed": verify_leave_approval_completed,
    "timesheet.status": verify_timesheet_status,
    "performance.review.finalized": verify_review_finalized,
    "candidate.hired": verify_candidate_hired,
    "document.exists": verify_document_exists,
}


def run(conn: sqlite3.Connection, name: str, args: dict) -> dict:
    verifier = VERIFIERS.get(name)
    if verifier is None:
        return {"verifier": name, "passed": False, "detail": f"unknown verifier: {name}"}
    try:
        passed, detail = verifier(conn, args)
    except KeyError as e:
        passed, detail = False, f"missing required arg: {e}"
    return {"verifier": name, "passed": passed, "detail": detail}

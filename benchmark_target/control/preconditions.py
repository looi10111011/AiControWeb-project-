"""Precondition verification (spec 15). A small, explicit set of predicate types — not a
general expression language — matching the shapes used in the spec's own task YAML example
(employee.exists, leave_balance.available, supervisor.assigned).

Each condition is {"type": "<dotted.name>", ...args}. A failing precondition means the task
fixture/catalog is broken, not that the Agent did anything wrong — the runner must classify
this as SCENARIO_SETUP_ERROR, never as a functional failure (spec 15/27).
"""

import sqlite3
from typing import Any


def _employee_id_by_code(conn: sqlite3.Connection, employee_code: str) -> int | None:
    row = conn.execute("SELECT id FROM employees WHERE employee_code = ?", (employee_code,)).fetchone()
    return row["id"] if row else None


def _check_employee_exists(conn: sqlite3.Connection, cond: dict) -> tuple[bool, str]:
    emp_id = _employee_id_by_code(conn, cond["employee_code"])
    return (emp_id is not None, f"employee_code={cond['employee_code']}")


def _check_employee_status(conn: sqlite3.Connection, cond: dict) -> tuple[bool, str]:
    row = conn.execute(
        "SELECT status FROM employees WHERE employee_code = ?", (cond["employee_code"],)
    ).fetchone()
    if row is None:
        return False, f"employee_code={cond['employee_code']} does not exist"
    passed = row["status"] == cond["status"]
    return passed, f"status={row['status']} (expected {cond['status']})"


def _check_leave_balance_available(conn: sqlite3.Connection, cond: dict) -> tuple[bool, str]:
    emp_id = _employee_id_by_code(conn, cond["employee_code"])
    if emp_id is None:
        return False, f"employee_code={cond['employee_code']} does not exist"
    row = conn.execute(
        """SELECT lb.balance_days FROM leave_balances lb
           JOIN leave_types lt ON lt.id = lb.leave_type_id
           WHERE lb.employee_id = ? AND lt.name = ?""",
        (emp_id, cond["leave_type"]),
    ).fetchone()
    if row is None:
        return False, f"no {cond['leave_type']} leave balance for {cond['employee_code']}"
    reserved = conn.execute(
        """SELECT COALESCE(SUM(days), 0) d FROM leave_requests lr
           JOIN leave_types lt ON lt.id = lr.leave_type_id
           WHERE lr.employee_id = ? AND lt.name = ? AND lr.status IN ('pending', 'approved')""",
        (emp_id, cond["leave_type"]),
    ).fetchone()["d"]
    available = row["balance_days"] - reserved
    min_days = cond["min_days"]
    return available >= min_days, f"available={available} (need >= {min_days})"


def _check_supervisor_assigned(conn: sqlite3.Connection, cond: dict) -> tuple[bool, str]:
    emp_id = _employee_id_by_code(conn, cond["employee_code"])
    if emp_id is None:
        return False, f"employee_code={cond['employee_code']} does not exist"
    row = conn.execute(
        """SELECT s.employee_code AS supervisor_code FROM employees e
           LEFT JOIN employees s ON s.id = e.supervisor_id WHERE e.id = ?""",
        (emp_id,),
    ).fetchone()
    actual = row["supervisor_code"] if row else None
    return actual == cond["supervisor_code"], f"supervisor={actual} (expected {cond['supervisor_code']})"


CHECKERS = {
    "employee.exists": _check_employee_exists,
    "employee.status": _check_employee_status,
    "leave_balance.available": _check_leave_balance_available,
    "supervisor.assigned": _check_supervisor_assigned,
}


def evaluate(conn: sqlite3.Connection, conditions: list[dict[str, Any]]) -> list[dict]:
    results = []
    for cond in conditions:
        cond_type = cond.get("type")
        checker = CHECKERS.get(cond_type)
        if checker is None:
            results.append({"condition": cond, "passed": False, "detail": f"unknown condition type: {cond_type}"})
            continue
        try:
            passed, detail = checker(conn, cond)
        except KeyError as e:
            passed, detail = False, f"missing required field: {e}"
        results.append({"condition": cond, "passed": passed, "detail": detail})
    return results

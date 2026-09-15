"""Post-task integrity checks (spec 16, Layer 4): orphan records, broken relationships,
invalid aggregates. Runs generic invariants across the whole DB rather than task-specific
logic — task-specific correctness is verifiers.py's job.
"""

import sqlite3

CHECKS = [
    (
        "duplicate_employee_code",
        """SELECT employee_code, COUNT(*) c FROM employees GROUP BY employee_code HAVING c > 1""",
        lambda row: f"employee_code {row['employee_code']} appears {row['c']} times",
    ),
    (
        "orphan_user_employee_ref",
        """SELECT u.id, u.username FROM users u
           WHERE u.employee_id IS NOT NULL
             AND NOT EXISTS (SELECT 1 FROM employees e WHERE e.id = u.employee_id)""",
        lambda row: f"user {row['username']} (id={row['id']}) references a missing employee",
    ),
    (
        "orphan_leave_request_employee_ref",
        """SELECT lr.id FROM leave_requests lr
           WHERE NOT EXISTS (SELECT 1 FROM employees e WHERE e.id = lr.employee_id)""",
        lambda row: f"leave_request {row['id']} references a missing employee",
    ),
    (
        "orphan_document_employee_ref",
        """SELECT d.id, d.filename FROM documents d
           WHERE NOT EXISTS (SELECT 1 FROM employees e WHERE e.id = d.employee_id)""",
        lambda row: f"document {row['filename']} (id={row['id']}) references a missing employee",
    ),
    (
        "orphan_candidate_hired_employee_ref",
        """SELECT c.id, c.email FROM candidates c
           WHERE c.hired_employee_id IS NOT NULL
             AND NOT EXISTS (SELECT 1 FROM employees e WHERE e.id = c.hired_employee_id)""",
        lambda row: f"candidate {row['email']} (id={row['id']}) hired_employee_id references a missing employee",
    ),
    (
        "orphan_timesheet_entry_ref",
        """SELECT te.id FROM timesheet_entries te
           WHERE NOT EXISTS (SELECT 1 FROM timesheets t WHERE t.id = te.timesheet_id)""",
        lambda row: f"timesheet_entry {row['id']} references a missing timesheet",
    ),
    (
        "negative_leave_balance",
        """SELECT employee_id, leave_type_id, balance_days FROM leave_balances WHERE balance_days < 0""",
        lambda row: f"employee_id {row['employee_id']} has negative balance for leave_type_id {row['leave_type_id']}: {row['balance_days']}",
    ),
    (
        "leave_request_over_reserved",
        # spec 17/20: pending+approved days must never exceed the balance they reserve against.
        """SELECT lb.employee_id, lb.leave_type_id, lb.balance_days,
                  COALESCE(SUM(lr.days), 0) AS reserved
           FROM leave_balances lb
           LEFT JOIN leave_requests lr ON lr.employee_id = lb.employee_id
                AND lr.leave_type_id = lb.leave_type_id AND lr.status IN ('pending', 'approved')
           GROUP BY lb.employee_id, lb.leave_type_id
           HAVING reserved > lb.balance_days""",
        lambda row: (
            f"employee_id {row['employee_id']} leave_type_id {row['leave_type_id']}: "
            f"reserved {row['reserved']} exceeds balance {row['balance_days']}"
        ),
    ),
    (
        "timesheet_entry_date_outside_week",
        """SELECT te.id, te.work_date, t.week_start_date FROM timesheet_entries te
           JOIN timesheets t ON t.id = te.timesheet_id
           WHERE te.work_date < t.week_start_date OR te.work_date > date(t.week_start_date, '+6 days')""",
        lambda row: f"timesheet_entry {row['id']} date {row['work_date']} outside week {row['week_start_date']}",
    ),
]


def check(conn: sqlite3.Connection) -> list[dict]:
    violations = []
    for name, query, describe in CHECKS:
        for row in conn.execute(query).fetchall():
            violations.append({"check": name, "detail": describe(row)})
    return violations

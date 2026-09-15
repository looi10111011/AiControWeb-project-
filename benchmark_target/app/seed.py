"""Deterministic seed data.

Spec 5/47: same seed -> same DB state, every time. Every value here is a pure function of a
fixed index — no randomness, no wall-clock. `seed()` is idempotent: it always resets first.

Password convention (fixed on purpose, not a security feature — see security.py docstring):
    "{FirstName}@123"
so e.g. Olivia Brown's password is "Olivia@123". The admin account is the one exception:
username "admin", password "Admin@123".
"""

from datetime import date, timedelta

from benchmark_target.app.db import get_conn, reset_db
from benchmark_target.app.security import hash_password

DEPARTMENTS = [
    "Engineering",
    "Sales",
    "HR",
    "Finance",
    "Marketing",
    "Support",
    "Operations",
]

JOB_TITLES = {
    "Engineering": ["Software Engineer", "Senior Software Engineer", "QA Engineer", "Engineering Manager"],
    "Sales": ["Sales Executive", "Account Manager", "Sales Manager"],
    "HR": ["HR Officer", "Recruiter", "HR Manager"],
    "Finance": ["Accountant", "Financial Analyst", "Finance Manager"],
    "Marketing": ["Marketing Executive", "Content Specialist", "Marketing Manager"],
    "Support": ["Support Agent", "Support Lead"],
    "Operations": ["Operations Coordinator", "Operations Manager"],
}

# 50 (first, last) pairs. Deliberately includes near-duplicate names (Sarah Chen / Sarah
# Chan, Michael Smith / Michael Smyth, Jon Park / John Park, Kristen Lee / Kirsten Lee) so
# search/filter tasks can't be solved by "first substring match wins" — spec 5.
NAMES: list[tuple[str, str]] = [
    ("Alice", "Nguyen"), ("Bob", "Martinez"), ("Sarah", "Chen"), ("Michael", "Smith"),
    ("Emma", "Johnson"), ("David", "Kim"), ("Olivia", "Brown"), ("James", "Wilson"),
    ("Sophia", "Davis"), ("Daniel", "Garcia"), ("Sarah", "Chan"), ("Michael", "Smyth"),
    ("Isabella", "Rodriguez"), ("Matthew", "Lopez"), ("Mia", "Gonzalez"), ("Andrew", "Anderson"),
    ("Charlotte", "Thomas"), ("Joseph", "Taylor"), ("Amelia", "Moore"), ("Jon", "Park"),
    ("John", "Park"), ("Ava", "Martin"), ("Ryan", "Jackson"), ("Grace", "Thompson"),
    ("Ethan", "White"), ("Chloe", "Harris"), ("Kristen", "Lee"), ("Kirsten", "Lee"),
    ("Noah", "Clark"), ("Lily", "Lewis"), ("William", "Robinson"), ("Zoe", "Walker"),
    ("Benjamin", "Hall"), ("Ella", "Allen"), ("Lucas", "Young"), ("Hannah", "King"),
    ("Henry", "Wright"), ("Layla", "Scott"), ("Alexander", "Green"), ("Nora", "Baker"),
    ("Jack", "Adams"), ("Victoria", "Nelson"), ("Owen", "Carter"), ("Aria", "Mitchell"),
    ("Samuel", "Perez"), ("Scarlett", "Roberts"), ("Leo", "Turner"), ("Ellie", "Phillips"),
    ("Julian", "Campbell"), ("Aubrey", "Parker"),
]

BASE_HIRE_DATE = date(2019, 1, 7)  # a Monday
BASE_TERMINATED_INDEXES = {12, 25, 38}  # deterministic: 3 terminated employees for status filter


def _hire_date(i: int) -> str:
    return (BASE_HIRE_DATE + timedelta(days=i * 17)).isoformat()


def seed() -> None:
    reset_db()
    now = "2026-01-01T00:00:00+00:00"  # fixed seed timestamp — never wall-clock (spec 47)

    with get_conn() as conn:
        employee_ids: list[int] = []
        dept_supervisor_id: dict[str, int] = {}

        for i, (first, last) in enumerate(NAMES):
            dept = DEPARTMENTS[i % len(DEPARTMENTS)]
            titles = JOB_TITLES[dept]
            is_dept_first = dept not in dept_supervisor_id
            title = titles[-1] if is_dept_first else titles[i % (len(titles) - 1)]
            status = "terminated" if i in BASE_TERMINATED_INDEXES else "active"

            cur = conn.execute(
                """INSERT INTO employees
                   (employee_code, first_name, last_name, department, job_title,
                    supervisor_id, status, hire_date, leave_balance_annual, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    f"EMP-{i + 1:04d}",
                    first,
                    last,
                    dept,
                    title,
                    None,  # supervisor set in a second pass once dept's first employee has an id
                    status,
                    _hire_date(i),
                    8 + (i % 15),
                    now,
                    now,
                ),
            )
            emp_id = cur.lastrowid
            employee_ids.append(emp_id)
            if is_dept_first:
                dept_supervisor_id[dept] = emp_id

        # second pass: assign each non-first employee in a department to that department's
        # first employee as supervisor_id
        for i, (first, last) in enumerate(NAMES):
            dept = DEPARTMENTS[i % len(DEPARTMENTS)]
            supervisor_id = dept_supervisor_id[dept]
            emp_id = employee_ids[i]
            if emp_id == supervisor_id:
                continue
            conn.execute(
                "UPDATE employees SET supervisor_id = ? WHERE id = ?",
                (supervisor_id, emp_id),
            )

        # users
        conn.execute(
            """INSERT INTO users (username, password_hash, role, employee_id, enabled, created_at, updated_at)
               VALUES (?, ?, 'admin', NULL, 1, ?, ?)""",
            ("admin", hash_password("Admin@123"), now, now),
        )

        for dept, sup_id in dept_supervisor_id.items():
            row_idx = employee_ids.index(sup_id)
            first, last = NAMES[row_idx]
            username = f"{first.lower()}.{last.lower()}"
            conn.execute(
                """INSERT INTO users (username, password_hash, role, employee_id, enabled, created_at, updated_at)
                   VALUES (?, ?, 'supervisor', ?, 1, ?, ?)""",
                (username, hash_password(f"{first}@123"), sup_id, now, now),
            )

        # a handful of ESS users on non-supervisor employees, deterministic indexes.
        # Indexes 0-6 are each department's first employee, i.e. that department's
        # supervisor (see the loop above) — deliberately excluded here to avoid a
        # username collision with the supervisor account already created for them.
        ess_indexes = [7, 10, 13, 17, 20, 24, 29, 34, 40, 45]
        for i in ess_indexes:
            first, last = NAMES[i]
            emp_id = employee_ids[i]
            username = f"{first.lower()}.{last.lower()}"
            conn.execute(
                """INSERT INTO users (username, password_hash, role, employee_id, enabled, created_at, updated_at)
                   VALUES (?, ?, 'ess', ?, 1, ?, ?)""",
                (username, hash_password(f"{first}@123"), emp_id, now, now),
            )

        # Daniel Garcia (index 9, not a department-first/supervisor employee) doubles as
        # the disabled-account login test fixture.
        disabled_idx = 9
        disabled_first, disabled_last = NAMES[disabled_idx]
        disabled_id = employee_ids[disabled_idx]
        conn.execute(
            """INSERT INTO users (username, password_hash, role, employee_id, enabled, created_at, updated_at)
               VALUES (?, ?, 'ess', ?, 0, ?, ?)""",
            (
                f"{disabled_first.lower()}.{disabled_last.lower()}",
                hash_password(f"{disabled_first}@123"),
                disabled_id,
                now,
                now,
            ),
        )

        # ---- Leave (spec 3.5) ----
        leave_type_ids: dict[str, int] = {}
        for name in ("Annual", "Sick", "Unpaid"):
            cur = conn.execute("INSERT INTO leave_types (name) VALUES (?)", (name,))
            leave_type_ids[name] = cur.lastrowid

        for i, emp_id in enumerate(employee_ids):
            annual_days = 8 + (i % 15)  # matches employees.leave_balance_annual formula above
            conn.execute(
                "INSERT INTO leave_balances (employee_id, leave_type_id, balance_days) VALUES (?, ?, ?)",
                (emp_id, leave_type_ids["Annual"], annual_days),
            )
            conn.execute(
                "INSERT INTO leave_balances (employee_id, leave_type_id, balance_days) VALUES (?, ?, ?)",
                (emp_id, leave_type_ids["Sick"], 10),
            )
            conn.execute(
                "INSERT INTO leave_balances (employee_id, leave_type_id, balance_days) VALUES (?, ?, ?)",
                (emp_id, leave_type_ids["Unpaid"], 999),
            )

        # James Wilson (idx 7, ESS user) — a pending Annual leave request awaiting his
        # supervisor's (Alice Nguyen, idx 0) approval. The canonical LEAVE-APPROVE fixture.
        james_id = employee_ids[7]
        conn.execute(
            """INSERT INTO leave_requests
               (employee_id, leave_type_id, start_date, end_date, days, status, comment, created_at, updated_at)
               VALUES (?, ?, '2026-02-02', '2026-02-04', 3, 'pending', 'Family trip', ?, ?)""",
            (james_id, leave_type_ids["Annual"], now, now),
        )

        # Ethan White (idx 24) — an already-approved historical leave, for
        # read/verification tasks that don't need a live approval workflow.
        ethan_id = employee_ids[24]
        conn.execute(
            """INSERT INTO leave_requests
               (employee_id, leave_type_id, start_date, end_date, days, status, comment,
                approver_id, decision_comment, created_at, updated_at)
               VALUES (?, ?, '2026-01-12', '2026-01-13', 2, 'approved', 'Personal', ?, 'Approved', ?, ?)""",
            (ethan_id, leave_type_ids["Sick"], dept_supervisor_id["HR"], now, now),
        )

        # Matthew Lopez (idx 13) — a rejected request, for recovery-scenario tasks.
        matthew_id = employee_ids[13]
        conn.execute(
            """INSERT INTO leave_requests
               (employee_id, leave_type_id, start_date, end_date, days, status, comment,
                approver_id, decision_comment, created_at, updated_at)
               VALUES (?, ?, '2026-01-05', '2026-01-09', 5, 'rejected', 'Vacation', ?, 'Insufficient coverage that week', ?, ?)""",
            (matthew_id, leave_type_ids["Annual"], dept_supervisor_id["Operations"], now, now),
        )

        # ---- Time / Timesheets (spec 3.6) ----
        # James Wilson — a submitted timesheet awaiting approval (pairs with the leave
        # fixture above so one employee exercises both pending-approval flows).
        cur = conn.execute(
            """INSERT INTO timesheets (employee_id, week_start_date, status, created_at, updated_at)
               VALUES (?, '2026-01-05', 'submitted', ?, ?)""",
            (james_id, now, now),
        )
        ts_id = cur.lastrowid
        for work_date, project, hours in (
            ("2026-01-05", "Platform Migration", 8),
            ("2026-01-06", "Platform Migration", 7.5),
            ("2026-01-07", "Code Review", 4),
        ):
            conn.execute(
                "INSERT INTO timesheet_entries (timesheet_id, work_date, project, hours) VALUES (?, ?, ?, ?)",
                (ts_id, work_date, project, hours),
            )

        # John Park (idx 20, Operations) — an already-approved, locked timesheet, for the
        # "locked approved records" recovery scenario (spec 3.6).
        john_park_id = employee_ids[20]
        cur = conn.execute(
            """INSERT INTO timesheets (employee_id, week_start_date, status, created_at, updated_at)
               VALUES (?, '2025-12-29', 'approved', ?, ?)""",
            (john_park_id, now, now),
        )
        ts_id2 = cur.lastrowid
        for work_date, project, hours in (
            ("2025-12-29", "Warehouse Rollout", 8),
            ("2025-12-30", "Warehouse Rollout", 8),
        ):
            conn.execute(
                "INSERT INTO timesheet_entries (timesheet_id, work_date, project, hours) VALUES (?, ?, ?, ?)",
                (ts_id2, work_date, project, hours),
            )

        # ---- Recruitment (spec 3.7) ----
        cur = conn.execute(
            "INSERT INTO vacancies (title, department, status, created_at) VALUES (?, ?, 'open', ?)",
            ("Software Engineer", "Engineering", now),
        )
        vac_eng = cur.lastrowid
        cur = conn.execute(
            "INSERT INTO vacancies (title, department, status, created_at) VALUES (?, ?, 'closed', ?)",
            ("Account Manager", "Sales", now),
        )
        vac_sales = cur.lastrowid

        candidates_seed = [
            (vac_eng, "Priya", "Natarajan", "priya.n@example.test", "applied"),
            (vac_eng, "Marcus", "Webb", "marcus.webb@example.test", "shortlisted"),
            (vac_eng, "Yuki", "Tanaka", "yuki.tanaka@example.test", "interview"),
            (vac_eng, "Diego", "Fernandez", "diego.f@example.test", "rejected"),
            (vac_sales, "Helen", "Osei", "helen.osei@example.test", "applied"),
        ]
        candidate_ids: dict[str, int] = {}
        for vacancy_id, first, last, email, status in candidates_seed:
            cur = conn.execute(
                """INSERT INTO candidates
                   (vacancy_id, first_name, last_name, email, status, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (vacancy_id, first, last, email, status, now, now),
            )
            candidate_ids[email] = cur.lastrowid

        # Yuki Tanaka's scheduled interview, interviewer = Engineering's supervisor
        conn.execute(
            """INSERT INTO interviews (candidate_id, scheduled_at, interviewer_employee_id, notes, created_at)
               VALUES (?, '2026-01-20T10:00:00+00:00', ?, NULL, ?)""",
            (candidate_ids["yuki.tanaka@example.test"], dept_supervisor_id["Engineering"], now),
        )

        # ---- Performance (spec 3.8) ----
        cur = conn.execute(
            "INSERT INTO review_cycles (name, status) VALUES ('2026 H1 Review', 'open')"
        )
        cycle_id = cur.lastrowid
        for i, emp_id in enumerate(employee_ids):
            if i == 7:
                # James Wilson already completed his self-review — ready for his
                # supervisor's manager-review step, the other half of the workflow.
                conn.execute(
                    """INSERT INTO reviews
                       (cycle_id, employee_id, self_rating, self_comment, status, updated_at)
                       VALUES (?, ?, 4, 'Shipped the platform migration ahead of schedule.', 'manager_pending', ?)""",
                    (cycle_id, emp_id, now),
                )
            else:
                conn.execute(
                    "INSERT INTO reviews (cycle_id, employee_id, status, updated_at) VALUES (?, ?, 'self_pending', ?)",
                    (cycle_id, emp_id, now),
                )


if __name__ == "__main__":
    seed()
    print("Seeded benchmark_target/data/target.db")

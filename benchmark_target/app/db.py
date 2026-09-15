"""SQLite persistence for the Target Surface.

Plain stdlib sqlite3, no ORM (spec 44: simple over microservices; SQLite fine for V1).
One connection per request via a context manager — this app is single-process, low
concurrency by design (it's a benchmark fixture, not something to load-test).
"""

import sqlite3
from contextlib import contextmanager

from benchmark_target.app.config import DATA_DIR, DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS employees (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    employee_code   TEXT UNIQUE NOT NULL,
    first_name      TEXT NOT NULL,
    last_name       TEXT NOT NULL,
    department      TEXT NOT NULL,
    job_title       TEXT NOT NULL,
    supervisor_id   INTEGER REFERENCES employees(id),
    status          TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'terminated')),
    hire_date       TEXT NOT NULL,
    leave_balance_annual INTEGER NOT NULL DEFAULT 18,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS users (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    username        TEXT UNIQUE NOT NULL,
    password_hash   TEXT NOT NULL,
    role            TEXT NOT NULL CHECK (role IN ('admin', 'supervisor', 'ess')),
    employee_id     INTEGER REFERENCES employees(id),
    enabled         INTEGER NOT NULL DEFAULT 1,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS audit_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT NOT NULL,
    actor_username  TEXT NOT NULL,
    action          TEXT NOT NULL,
    resource        TEXT NOT NULL,
    resource_id     TEXT,
    details         TEXT
);

CREATE TABLE IF NOT EXISTS leave_types (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    name    TEXT UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS leave_balances (
    employee_id     INTEGER NOT NULL REFERENCES employees(id),
    leave_type_id   INTEGER NOT NULL REFERENCES leave_types(id),
    balance_days    INTEGER NOT NULL,
    PRIMARY KEY (employee_id, leave_type_id)
);

CREATE TABLE IF NOT EXISTS leave_requests (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    employee_id     INTEGER NOT NULL REFERENCES employees(id),
    leave_type_id   INTEGER NOT NULL REFERENCES leave_types(id),
    start_date      TEXT NOT NULL,
    end_date        TEXT NOT NULL,
    days            INTEGER NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending', 'approved', 'rejected', 'cancelled')),
    comment         TEXT,
    approver_id     INTEGER REFERENCES employees(id),
    decision_comment TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS timesheets (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    employee_id     INTEGER NOT NULL REFERENCES employees(id),
    week_start_date TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'draft'
                    CHECK (status IN ('draft', 'submitted', 'approved', 'rejected')),
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    UNIQUE (employee_id, week_start_date)
);

CREATE TABLE IF NOT EXISTS timesheet_entries (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timesheet_id    INTEGER NOT NULL REFERENCES timesheets(id),
    work_date       TEXT NOT NULL,
    project         TEXT NOT NULL,
    hours           REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS vacancies (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    title       TEXT NOT NULL,
    department  TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'closed')),
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS candidates (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    vacancy_id      INTEGER NOT NULL REFERENCES vacancies(id),
    first_name      TEXT NOT NULL,
    last_name       TEXT NOT NULL,
    email           TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'applied'
                    CHECK (status IN ('applied', 'shortlisted', 'interview', 'rejected', 'hired')),
    hired_employee_id INTEGER REFERENCES employees(id),
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS interviews (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id            INTEGER NOT NULL REFERENCES candidates(id),
    scheduled_at            TEXT NOT NULL,
    interviewer_employee_id INTEGER REFERENCES employees(id),
    notes                   TEXT,
    created_at              TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS review_cycles (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    name    TEXT NOT NULL,
    status  TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'closed'))
);

CREATE TABLE IF NOT EXISTS reviews (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_id        INTEGER NOT NULL REFERENCES review_cycles(id),
    employee_id     INTEGER NOT NULL REFERENCES employees(id),
    self_rating     INTEGER,
    self_comment    TEXT,
    manager_rating  INTEGER,
    manager_comment TEXT,
    status          TEXT NOT NULL DEFAULT 'self_pending'
                    CHECK (status IN ('self_pending', 'manager_pending', 'finalized')),
    updated_at      TEXT NOT NULL,
    UNIQUE (cycle_id, employee_id)
);

CREATE TABLE IF NOT EXISTS fault_injections (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    trigger     TEXT NOT NULL,
    fault_type  TEXT NOT NULL CHECK (fault_type IN ('conflict', 'error')),
    expires_at  TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    consumed    INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS documents (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    employee_id     INTEGER NOT NULL REFERENCES employees(id),
    filename        TEXT NOT NULL,
    stored_path     TEXT NOT NULL,
    mime_type       TEXT NOT NULL,
    size_bytes      INTEGER NOT NULL,
    uploaded_by     TEXT NOT NULL,
    uploaded_at     TEXT NOT NULL,
    UNIQUE (employee_id, filename)
);

CREATE INDEX IF NOT EXISTS idx_employees_department ON employees(department);
CREATE INDEX IF NOT EXISTS idx_employees_status ON employees(status);
CREATE INDEX IF NOT EXISTS idx_employees_supervisor ON employees(supervisor_id);
CREATE INDEX IF NOT EXISTS idx_users_employee ON users(employee_id);
CREATE INDEX IF NOT EXISTS idx_leave_requests_employee ON leave_requests(employee_id);
CREATE INDEX IF NOT EXISTS idx_leave_requests_status ON leave_requests(status);
CREATE INDEX IF NOT EXISTS idx_timesheets_employee ON timesheets(employee_id);
CREATE INDEX IF NOT EXISTS idx_timesheet_entries_timesheet ON timesheet_entries(timesheet_id);
CREATE INDEX IF NOT EXISTS idx_candidates_vacancy ON candidates(vacancy_id);
CREATE INDEX IF NOT EXISTS idx_reviews_cycle ON reviews(cycle_id);
CREATE INDEX IF NOT EXISTS idx_documents_employee ON documents(employee_id);
CREATE INDEX IF NOT EXISTS idx_fault_injections_trigger ON fault_injections(trigger);
"""


# Every table, in FK-safe creation order — the Control Plane's snapshot/diff/integrity
# code (benchmark_target/control/) uses this as the default scope so it never has to be
# kept in sync by hand when a table is added here.
TABLE_NAMES = [
    "employees", "users", "leave_types", "leave_balances", "leave_requests",
    "timesheets", "timesheet_entries", "vacancies", "candidates", "interviews",
    "review_cycles", "reviews", "documents", "fault_injections", "audit_log",
]


def init_db() -> None:
    """Create the schema if it doesn't exist yet. Idempotent."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with get_conn() as conn:
        conn.executescript(SCHEMA)


def reset_db() -> None:
    """Drop and recreate every table. Used by the Control Plane's /reset — never reachable
    from the Target Surface itself."""
    with get_conn() as conn:
        conn.executescript(
            """
            DROP TABLE IF EXISTS audit_log;
            DROP TABLE IF EXISTS fault_injections;
            DROP TABLE IF EXISTS documents;
            DROP TABLE IF EXISTS reviews;
            DROP TABLE IF EXISTS review_cycles;
            DROP TABLE IF EXISTS interviews;
            DROP TABLE IF EXISTS candidates;
            DROP TABLE IF EXISTS vacancies;
            DROP TABLE IF EXISTS timesheet_entries;
            DROP TABLE IF EXISTS timesheets;
            DROP TABLE IF EXISTS leave_requests;
            DROP TABLE IF EXISTS leave_balances;
            DROP TABLE IF EXISTS leave_types;
            DROP TABLE IF EXISTS users;
            DROP TABLE IF EXISTS employees;
            """
        )
        conn.executescript(SCHEMA)


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def log_audit(conn: sqlite3.Connection, actor_username: str, action: str, resource: str,
              resource_id: str | None = None, details: str | None = None) -> None:
    from datetime import datetime, timezone

    conn.execute(
        "INSERT INTO audit_log (timestamp, actor_username, action, resource, resource_id, details) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (datetime.now(timezone.utc).isoformat(), actor_username, action, resource, resource_id, details),
    )

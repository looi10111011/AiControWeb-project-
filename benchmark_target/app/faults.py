"""Fault injection (spec 19): a benchmark-only mechanism the Control Plane arms and the
Target Surface silently obeys. Lives under app/ (not control/) because Target Surface
route handlers import `check_and_consume_fault` directly — the Agent never sees this
module or the table it reads, only the 409/500 it occasionally produces.
"""

import sqlite3
from datetime import datetime, timedelta, timezone

from benchmark_target.app.db import get_conn


def arm_fault(conn: sqlite3.Connection, trigger: str, fault_type: str, ttl_seconds: int) -> int:
    now = datetime.now(timezone.utc)
    expires_at = (now + timedelta(seconds=ttl_seconds)).isoformat()
    cur = conn.execute(
        "INSERT INTO fault_injections (trigger, fault_type, expires_at, created_at, consumed) VALUES (?, ?, ?, ?, 0)",
        (trigger, fault_type, expires_at, now.isoformat()),
    )
    return cur.lastrowid


def sweep_expired(conn: sqlite3.Connection) -> int:
    """Delete faults past their TTL or already consumed. Called on Control Plane startup
    (spec 19: 'if Runner crashes -> startup recovery -> disable stale faults') and cheap
    enough to also run before every read."""
    now = datetime.now(timezone.utc).isoformat()
    cur = conn.execute(
        "DELETE FROM fault_injections WHERE expires_at < ? OR consumed = 1", (now,)
    )
    return cur.rowcount


def check_and_consume_fault(conn: sqlite3.Connection, trigger: str) -> str | None:
    """Called by a Target Surface route right before it would otherwise succeed. Returns
    the fault_type to simulate ('conflict' -> 409, 'error' -> 500) or None. One-shot: a
    matched fault is marked consumed so the *next* attempt at the same action succeeds —
    matching the concurrency-test pattern in spec 20 (agent retries after the injected
    failure and the retry must go through).

    Deliberately uses its OWN connection/transaction, not the caller's `conn` — the
    caller is about to raise an HTTPException specifically so the route's own transaction
    rolls back, and if the "mark consumed" write shared that transaction it would be
    undone by the very rollback it triggered, making every fault re-fire forever instead
    of once. `conn` is still accepted (and used for the read) so callers can query inside
    their own already-open transaction without an extra connection when nothing fires.
    """
    now = datetime.now(timezone.utc).isoformat()
    row = conn.execute(
        """SELECT id, fault_type FROM fault_injections
           WHERE trigger = ? AND consumed = 0 AND expires_at >= ?
           ORDER BY id LIMIT 1""",
        (trigger, now),
    ).fetchone()
    if row is None:
        return None
    with get_conn() as consume_conn:
        consume_conn.execute("UPDATE fault_injections SET consumed = 1 WHERE id = ?", (row["id"],))
    return row["fault_type"]

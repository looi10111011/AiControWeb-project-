"""Scoped authoritative state snapshot + diff (spec 17). The DB is small (tens to low
hundreds of rows total across all tables), so dumping whole tables is cheap and simpler
than a real change-log — "necessary" here, not the wasteful case spec 17 warns against.
"""

import sqlite3
from typing import Any

from benchmark_target.app.db import TABLE_NAMES

# Composite-key tables: everything else uses its `id` column.
_COMPOSITE_KEYS: dict[str, tuple[str, ...]] = {
    "leave_balances": ("employee_id", "leave_type_id"),
}


def _row_key(table: str, row: dict) -> Any:
    if table in _COMPOSITE_KEYS:
        return tuple(row[k] for k in _COMPOSITE_KEYS[table])
    return row["id"]


def capture(conn: sqlite3.Connection, tables: list[str] | None = None) -> dict[str, list[dict]]:
    tables = tables or TABLE_NAMES
    unknown = [t for t in tables if t not in TABLE_NAMES]
    if unknown:
        raise ValueError(f"Unknown table(s): {unknown}")
    return {t: [dict(r) for r in conn.execute(f"SELECT * FROM {t}").fetchall()] for t in tables}


def diff(before: dict[str, list[dict]], after: dict[str, list[dict]]) -> dict[str, dict]:
    """Per-table created/updated/deleted, keyed by each table's primary key."""
    result: dict[str, dict] = {}
    for table in sorted(set(before) | set(after)):
        before_rows = {_row_key(table, r): r for r in before.get(table, [])}
        after_rows = {_row_key(table, r): r for r in after.get(table, [])}

        created = [after_rows[k] for k in after_rows.keys() - before_rows.keys()]
        deleted = [before_rows[k] for k in before_rows.keys() - after_rows.keys()]
        updated = []
        for k in before_rows.keys() & after_rows.keys():
            b, a = before_rows[k], after_rows[k]
            changes = {f: {"before": b[f], "after": a[f]} for f in a if a.get(f) != b.get(f)}
            if changes:
                updated.append({"key": k, "changes": changes})

        if created or deleted or updated:
            result[table] = {
                "created": created,
                "deleted": deleted,
                "updated": updated,
            }
    return result

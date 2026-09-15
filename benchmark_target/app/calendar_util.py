"""Deterministic benchmark work calendar (spec 6). Monday-Friday work week, a fixed
holiday list — no external calendar API, no current-time dependence.
"""

from datetime import date, timedelta

HOLIDAYS: set[str] = {
    "2026-01-01",  # New Year's Day
    "2026-04-06",  # deterministic fixture holiday
    "2026-05-01",  # Labour Day
    "2026-12-25",  # Christmas Day
}


def is_working_day(d: date) -> bool:
    return d.weekday() < 5 and d.isoformat() not in HOLIDAYS


def count_working_days(start: date, end: date) -> int:
    """Inclusive working-day count between start and end (start <= end)."""
    if end < start:
        return 0
    days = 0
    cursor = start
    while cursor <= end:
        if is_working_day(cursor):
            days += 1
        cursor += timedelta(days=1)
    return days

"""Minimal RBAC: resource + action + scope, default deny, server-side only.

Spec 3.4 explicitly says not to build a general IAM engine — this is the smallest thing that
still lets tasks exercise "unauthorized UI", "direct URL access", "scope violations" (spec 21).

Scopes, narrowest to widest: own < direct_reports < all. Frontend never decides access; every
route re-checks via `can()` / `employee_in_scope()` itself, so hiding a nav link is not
mistaken anywhere in this codebase for actual authorization.
"""

from typing import Literal

from fastapi import Request

Role = Literal["admin", "supervisor", "ess"]

# resource -> action -> scope granted to that role. Missing entry = denied (default deny).
CAPABILITIES: dict[Role, dict[str, dict[str, str]]] = {
    "admin": {
        "employee": {"read": "all", "create": "all", "update": "all", "deactivate": "all"},
        "user": {"read": "all", "create": "all", "update": "all"},
        "leave": {"read": "all", "approve": "all", "create": "all", "cancel": "all"},
        "timesheet": {"read": "all", "approve": "all", "create": "all"},
        "recruitment": {"read": "all", "manage": "all"},
        "performance": {"read": "all", "manage": "all", "finalize": "all"},
        "document": {"read": "all", "manage": "all"},
        "report": {"read": "all"},
    },
    "supervisor": {
        "employee": {"read": "direct_reports"},
        "leave": {"read": "direct_reports", "approve": "direct_reports", "create": "own", "cancel": "own"},
        "timesheet": {"read": "direct_reports", "approve": "direct_reports", "create": "own"},
        "recruitment": {"read": "all", "manage": "all"},
        "performance": {"read": "direct_reports", "manage": "direct_reports"},
        "document": {"read": "direct_reports", "manage": "own"},
    },
    "ess": {
        "employee": {"read": "own", "update": "own"},
        "leave": {"read": "own", "create": "own", "cancel": "own"},
        "timesheet": {"read": "own", "create": "own"},
        "performance": {"read": "own", "manage": "own"},
        "document": {"read": "own", "manage": "own"},
    },
}


def get_current_user(request: Request) -> dict | None:
    return request.session.get("user")


def get_scope(role: str, resource: str, action: str) -> str | None:
    return CAPABILITIES.get(role, {}).get(resource, {}).get(action)


def can(role: str, resource: str, action: str) -> bool:
    return get_scope(role, resource, action) is not None


def employee_in_scope(user: dict, employee_row, scope: str) -> bool:
    """employee_row: sqlite3.Row with at least id, supervisor_id."""
    if scope == "all":
        return True
    if scope == "own":
        return user.get("employee_id") == employee_row["id"]
    if scope == "direct_reports":
        return employee_row["supervisor_id"] == user.get("employee_id")
    return False

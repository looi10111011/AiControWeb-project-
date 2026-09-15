"""Task schema (spec 11-12): the shape every catalog task must satisfy, and identity
(task_id + revision) rules. Deliberately a hand-rolled checklist, not a pydantic model —
this validates already-loaded plain dicts (from YAML) for the catalog validator and CI,
not live API request bodies, so there is no framework already doing this for free.
"""

LEVELS = {"L1", "L2", "L3", "L4", "L5"}
VERIFICATION_TYPES = {"custom", "state_diff", "download_artifact", "http_redirect"}

REQUIRED_FIELDS = ["task_id", "revision", "level", "actors", "goal", "url", "fixture", "verification"]


def validate_task(task: dict, verifier_names: set[str], precondition_types: set[str]) -> list[str]:
    """Returns a list of error strings; empty means the task is valid."""
    errors = []

    for field in REQUIRED_FIELDS:
        if field not in task:
            errors.append(f"missing required field: {field}")
    if errors:
        return errors  # nothing else is safe to check without the basics

    if task["level"] not in LEVELS:
        errors.append(f"invalid level: {task['level']}")
    if not isinstance(task["actors"], list) or not task["actors"]:
        errors.append("actors must be a non-empty list")
    if not isinstance(task.get("goal", {}).get("description"), str) or not task["goal"]["description"].strip():
        errors.append("goal.description must be a non-empty string")
    if task["fixture"] != "base":
        errors.append(f"unknown fixture: {task['fixture']} (only 'base' exists in v1)")

    verification = task["verification"]
    v_type = verification.get("type")
    if v_type not in VERIFICATION_TYPES:
        errors.append(f"invalid verification.type: {v_type}")
    elif v_type == "custom":
        verifier = verification.get("verifier")
        if verifier not in verifier_names:
            errors.append(f"unknown verifier: {verifier}")
        if not isinstance(verification.get("args"), dict):
            errors.append("verification.args must be a dict for type=custom")
    elif v_type == "state_diff":
        if "expected" not in verification:
            errors.append("verification.expected required for type=state_diff")

    for cond in task.get("preconditions", []):
        if cond.get("type") not in precondition_types:
            errors.append(f"unknown precondition type: {cond.get('type')}")

    return errors


def task_identity(task: dict) -> str:
    return f"{task['task_id']}@r{task['revision']}"

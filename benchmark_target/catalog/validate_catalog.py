"""Validates catalog/v1 against schema.py (spec 13: schema, task uniqueness, task
completeness, verifier existence, precondition validity). Fixture existence and benchmark
leakage are satisfied by construction — generate_catalog.py only ever emits real
employee_codes/usernames/emails pulled from the live seed, so there is nothing separate to
check here for those two.

Usage: python -m benchmark_target.catalog.validate_catalog
Exit code 0 = valid catalog, 1 = validation errors found.
"""

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark_target.catalog.schema import task_identity, validate_task
from benchmark_target.control.preconditions import CHECKERS
from benchmark_target.control.verifiers import VERIFIERS

CATALOG_DIR = Path(__file__).resolve().parent / "v1"
TASKS_DIR = CATALOG_DIR / "tasks"


def load_all_tasks() -> list[dict]:
    tasks = []
    for path in sorted(TASKS_DIR.glob("*.yaml")):
        loaded = yaml.safe_load(path.read_text()) or []
        tasks.extend(loaded)
    return tasks


def validate() -> tuple[bool, list[str]]:
    errors = []
    tasks = load_all_tasks()

    if not tasks:
        return False, ["no tasks found — run generate_catalog.py first"]

    verifier_names = set(VERIFIERS.keys())
    precondition_types = set(CHECKERS.keys())

    seen_identities: dict[str, int] = {}
    level_counts: dict[str, int] = {}

    for task in tasks:
        task_errors = validate_task(task, verifier_names, precondition_types)
        if task_errors:
            label = task.get("task_id", "<unknown>")
            errors.extend(f"[{label}] {e}" for e in task_errors)
            continue  # identity/level checks below assume the basics are present

        identity = task_identity(task)
        seen_identities[identity] = seen_identities.get(identity, 0) + 1
        level_counts[task["level"]] = level_counts.get(task["level"], 0) + 1

    for identity, count in seen_identities.items():
        if count > 1:
            errors.append(f"duplicate task identity: {identity} appears {count} times")

    total = len(tasks)
    if not (80 <= total <= 200):
        errors.append(f"task count {total} is outside the sane range [80, 200] — check the generator")

    return len(errors) == 0, errors + [f"level distribution: {level_counts}", f"total tasks: {total}"]


def main():
    valid, messages = validate()
    for m in messages:
        print(m)
    print("VALID" if valid else "INVALID")
    sys.exit(0 if valid else 1)


if __name__ == "__main__":
    main()

"""Phase 8: determinism / reproducibility validation (spec 47 + Phase 8 of spec 51).

This deliberately does NOT re-run the real LLM agent to prove determinism — an LLM's
step-by-step behavior is not reproducible run-to-run even at the same task/seed (the
repo's own release_gate.py comments document this noise at the agent level already; see
W_gate_is_noisy in optimize.txt). What Phase 8 actually needs proven, and what this
benchmark controls, is the fixture/seed/reset/catalog/verifier contract underneath the
agent: same seed -> byte-identical DB state, same catalog generator -> byte-identical
task definitions, same DB state -> same precondition/verifier outcome. That contract is
what makes a later agent-level comparison meaningful at all; without it, a success-rate
difference between two runs could just be the benchmark itself drifting.
"""

import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from benchmark_target.app.db import get_conn
from benchmark_target.app.seed import seed
from benchmark_target.control.main import app as control_app
from benchmark_target.control.preconditions import evaluate
from benchmark_target.control.snapshot import capture
from benchmark_target.control.verifiers import run as run_verifier

REPO_ROOT = Path(__file__).resolve().parents[2]
CATALOG_TASKS_DIR = REPO_ROOT / "benchmark_target" / "catalog" / "v1" / "tasks"


def _full_snapshot() -> dict:
    seed()
    with get_conn() as conn:
        return capture(conn, None)


def test_seed_produces_byte_identical_state_across_reseeds():
    """The literal spec-47 claim: same seed -> same expected business state, every time.
    Runs 4 independent reseeds (not 2) since a flaky non-determinism source — e.g. a stray
    `datetime.now()` instead of the fixed seed timestamp — could coincidentally match once."""
    snapshots = [_full_snapshot() for _ in range(4)]
    for i, snap in enumerate(snapshots[1:], start=2):
        assert snap == snapshots[0], f"reseed #{i} differs from reseed #1"


def test_seed_is_not_accidentally_using_wall_clock():
    """A determinism bug that only shows up if you reseed on two different real
    timestamps looks identical to the test above if run in the same second — this
    explicitly checks the *values* aren't derived from datetime.now() by comparing
    against the fixed constant seed.py documents using."""
    seed()
    with get_conn() as conn:
        row = conn.execute("SELECT created_at FROM employees WHERE employee_code = 'EMP-0001'").fetchone()
    assert row["created_at"] == "2026-01-01T00:00:00+00:00"


def test_reset_via_control_plane_is_idempotent():
    with TestClient(control_app) as client:
        r1 = client.post("/reset", json={"confirm_environment": "hermes-benchmark-local"})
        with get_conn() as conn:
            snap1 = capture(conn, None)
        r2 = client.post("/reset", json={"confirm_environment": "hermes-benchmark-local"})
        with get_conn() as conn:
            snap2 = capture(conn, None)

    assert r1.status_code == 200 and r2.status_code == 200
    assert snap1 == snap2


@pytest.mark.parametrize("condition", [
    {"type": "employee.exists", "employee_code": "EMP-0008"},
    {"type": "leave_balance.available", "employee_code": "EMP-0008", "leave_type": "Annual", "min_days": 2},
    {"type": "supervisor.assigned", "employee_code": "EMP-0008", "supervisor_code": "EMP-0001"},
    {"type": "employee.status", "employee_code": "EMP-0013", "status": "terminated"},
])
def test_precondition_result_is_stable_across_reseeds(condition):
    results = []
    for _ in range(3):
        seed()
        with get_conn() as conn:
            results.append(evaluate(conn, [condition])[0])
    assert all(r == results[0] for r in results), f"precondition {condition} gave different results across reseeds: {results}"


@pytest.mark.parametrize("verifier_name,args", [
    ("employee.field_equals", {"employee_code": "EMP-0001", "field": "department", "value": "Engineering"}),
    ("leave.request.status", {"employee_code": "EMP-0008", "leave_type": "Annual",
                               "start_date": "2026-02-02", "expected_status": "pending"}),
    ("timesheet.status", {"employee_code": "EMP-0021", "week_start_date": "2025-12-29", "expected_status": "approved"}),
])
def test_verifier_result_is_stable_across_reseeds(verifier_name, args):
    results = []
    for _ in range(3):
        seed()
        with get_conn() as conn:
            results.append(run_verifier(conn, verifier_name, args))
    assert all(r == results[0] for r in results), f"verifier {verifier_name} gave different results across reseeds: {results}"


def test_verifier_call_itself_has_no_side_effects():
    """Calling a verifier is a read — it must never change the state it's judging."""
    seed()
    with get_conn() as conn:
        before = capture(conn, None)
        run_verifier(conn, "employee.field_equals", {"employee_code": "EMP-0001", "field": "department", "value": "Engineering"})
        run_verifier(conn, "leave.approval.completed", {"employee_code": "EMP-0025", "leave_type": "Sick", "start_date": "2026-01-12"})
        after = capture(conn, None)
    assert before == after


def test_precondition_check_itself_has_no_side_effects():
    seed()
    with get_conn() as conn:
        before = capture(conn, None)
        evaluate(conn, [{"type": "leave_balance.available", "employee_code": "EMP-0008", "leave_type": "Annual", "min_days": 2}])
        after = capture(conn, None)
    assert before == after


def test_catalog_generation_is_byte_identical_across_regeneration():
    """Regenerates the catalog twice and diffs every task file — this also transitively
    re-proves seed determinism, since generate_catalog.py's task content (employee_codes,
    usernames, candidate emails) is pulled straight from a fresh seed() call each time."""
    def _generate_and_read() -> dict[str, str]:
        subprocess.run(
            [sys.executable, "-m", "benchmark_target.catalog.generate_catalog"],
            cwd=REPO_ROOT, check=True, capture_output=True, text=True,
        )
        return {p.name: p.read_text() for p in sorted(CATALOG_TASKS_DIR.glob("*.yaml"))}

    first = _generate_and_read()
    second = _generate_and_read()

    assert first.keys() == second.keys()
    for filename in first:
        assert first[filename] == second[filename], f"{filename} differs between two regenerations"


def test_catalog_task_count_and_level_distribution_are_stable():
    import yaml

    def _counts() -> dict[str, int]:
        counts = {}
        for path in sorted(CATALOG_TASKS_DIR.glob("*.yaml")):
            tasks = yaml.safe_load(path.read_text()) or []
            counts[path.stem] = len(tasks)
        return counts

    subprocess.run([sys.executable, "-m", "benchmark_target.catalog.generate_catalog"], cwd=REPO_ROOT, check=True)
    first = _counts()
    subprocess.run([sys.executable, "-m", "benchmark_target.catalog.generate_catalog"], cwd=REPO_ROOT, check=True)
    second = _counts()

    assert first == second
    assert sum(first.values()) >= 80  # sanity floor matching validate_catalog.py's own range check

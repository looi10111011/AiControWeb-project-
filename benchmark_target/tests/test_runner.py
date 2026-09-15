"""Phase 6 Run Engine tests. Uses a real Control Plane app via FastAPI's TestClient
(in-process, no subprocess/network) so reset/fixture/precondition/verify/state-diff/
integrity all run for real against the real SQLite fixture — only the agent itself is
stubbed (StubAgentAdapter / a small mutating adapter below), per this phase's own scope:
queue, state machines, retry, stability, artifacts — not a live browser or LLM.
"""

import json

import pytest
from fastapi.testclient import TestClient

from benchmark_target.app.db import get_conn
from benchmark_target.app.seed import seed
from benchmark_target.control.main import app as control_app
from benchmark_target.runner.agent_adapters import AgentAdapterError, AgentRunResult, StubAgentAdapter
from benchmark_target.runner.artifacts import write_run_artifacts
from benchmark_target.runner.attempt_runner import run_attempt
from benchmark_target.runner.models import (
    ExecutionHealth,
    FailureClass,
    FunctionalResult,
    RunState,
)
from benchmark_target.runner.queue import InvalidTransitionError, RunManager
from benchmark_target.runner.stability_runner import run_stability
from benchmark_target.runner.stats import wilson_interval


@pytest.fixture
def control_client():
    seed()
    with TestClient(control_app) as client:
        yield client


def _passthrough_task(**overrides) -> dict:
    task = {
        "task_id": "TEST-EMPLOYEE-STATUS",
        "revision": 1,
        "level": "L1",
        "fixture": "base",
        "actors": ["admin"],
        "goal": {"description": "test task"},
        "preconditions": [],
        "allowed_changes": [],
        "forbidden_changes": [],
        "url": "http://localhost:8100/pim/employees/1",
        "verification": {
            "type": "custom",
            "verifier": "employee.field_equals",
            "args": {"employee_code": "EMP-0001", "field": "status", "value": "active"},
        },
    }
    task.update(overrides)
    return task


# ---------------------------------------------------------------------------
# attempt_runner
# ---------------------------------------------------------------------------


def test_attempt_passes_when_verification_matches(control_client):
    adapter = StubAgentAdapter([AgentRunResult(success=True, steps=4, message="done", tokens={"input": 100, "output": 50})])
    result = run_attempt(_passthrough_task(), adapter, control_client)

    assert result.passed is True
    assert result.failure_class is None
    assert result.steps == 4
    assert result.integrity_violations == []


def test_attempt_functional_fail_when_verification_does_not_match(control_client):
    task = _passthrough_task(verification={
        "type": "custom", "verifier": "employee.field_equals",
        "args": {"employee_code": "EMP-0001", "field": "status", "value": "terminated"},
    })
    adapter = StubAgentAdapter([AgentRunResult(success=True, steps=1, message="", tokens={})])
    result = run_attempt(task, adapter, control_client)

    assert result.passed is False
    assert result.failure_class == FailureClass.FUNCTIONAL_FAIL


def test_attempt_setup_error_on_failing_precondition_and_agent_never_runs(control_client):
    task = _passthrough_task(preconditions=[
        {"type": "leave_balance.available", "employee_code": "EMP-0008", "leave_type": "Annual", "min_days": 9999}
    ])
    adapter = StubAgentAdapter([AgentRunResult(success=True, steps=1, message="", tokens={})])
    result = run_attempt(task, adapter, control_client)

    assert result.passed is False
    assert result.failure_class == FailureClass.SCENARIO_SETUP_ERROR
    assert adapter.call_count == 0  # never got as far as starting the agent


def test_attempt_classifies_agent_adapter_error_as_infra(control_client):
    adapter = StubAgentAdapter([("browser crashed", "BROWSER_ERROR")])
    result = run_attempt(_passthrough_task(), adapter, control_client)

    assert result.passed is False
    assert result.failure_class == FailureClass.BROWSER_ERROR
    assert result.is_infra_failure() is True


def test_attempt_unknown_adapter_failure_class_fails_closed_to_infra(control_client):
    adapter = StubAgentAdapter([("weird", "SOMETHING_NOT_IN_THE_TAXONOMY")])
    result = run_attempt(_passthrough_task(), adapter, control_client)
    assert result.failure_class == FailureClass.INFRA_ERROR


def test_attempt_state_diff_verification(control_client):
    """Simulates a task whose action creates an employee, verified via /state-diff."""

    class CreatesEmployeeAdapter:
        def run(self, task, actor):
            import datetime

            with get_conn() as conn:
                now = datetime.datetime.now(datetime.timezone.utc).isoformat()
                conn.execute(
                    """INSERT INTO employees
                       (employee_code, first_name, last_name, department, job_title, status,
                        hire_date, leave_balance_annual, created_at, updated_at)
                       VALUES ('EMP-9999', 'Test', 'Person', 'Engineering', 'Software Engineer',
                               'active', '2026-01-01', 18, ?, ?)""",
                    (now, now),
                )
            return AgentRunResult(success=True, steps=3, message="created", tokens={})

    task = _passthrough_task(
        verification={"type": "state_diff", "expected": {"employees": {"created": [{}]}}}
    )
    result = run_attempt(task, CreatesEmployeeAdapter(), control_client)
    assert result.passed is True


def test_attempt_state_diff_verification_fails_when_nothing_created(control_client):
    task = _passthrough_task(
        verification={"type": "state_diff", "expected": {"employees": {"created": [{}]}}}
    )
    adapter = StubAgentAdapter([AgentRunResult(success=True, steps=1, message="", tokens={})])
    result = run_attempt(task, adapter, control_client)
    assert result.passed is False
    assert result.failure_class == FailureClass.FUNCTIONAL_FAIL


def test_attempt_download_artifact_verification_is_honestly_inconclusive(control_client):
    task = _passthrough_task(verification={"type": "download_artifact", "expected": {"filename_contains": "x", "content_type": "text/csv"}})
    adapter = StubAgentAdapter([AgentRunResult(success=True, steps=1, message="", tokens={})])
    result = run_attempt(task, adapter, control_client)
    assert result.passed is False
    assert result.failure_class == FailureClass.INCONCLUSIVE


# ---------------------------------------------------------------------------
# stability_runner
# ---------------------------------------------------------------------------


def test_stability_all_pass_is_stable_pass_and_clean(control_client):
    adapter = StubAgentAdapter([AgentRunResult(success=True, steps=2, message="", tokens={})])
    result = run_stability(_passthrough_task(), adapter, control_client, n=5)

    assert result.functional_result == FunctionalResult.PASS
    assert result.execution_health == ExecutionHealth.CLEAN
    assert result.valid_sample_count == 5
    assert result.total_attempt_count == 5
    assert result.success_rate == 1.0


def test_stability_all_fail_is_stable_fail(control_client):
    task = _passthrough_task(verification={
        "type": "custom", "verifier": "employee.field_equals",
        "args": {"employee_code": "EMP-0001", "field": "status", "value": "terminated"},
    })
    adapter = StubAgentAdapter([AgentRunResult(success=True, steps=1, message="", tokens={})])
    result = run_stability(task, adapter, control_client, n=5)

    assert result.functional_result == FunctionalResult.FAIL
    assert result.success_rate == 0.0


def test_stability_flaky_when_mixed(control_client):
    """A stub that actually mutates the DB on 'success' calls and not on 'failure' calls —
    each attempt gets a fresh reset first, so this genuinely exercises pass vs. fail per
    sample rather than faking it via a constant verifier."""

    class SometimesMutatingAdapter:
        def __init__(self, pattern: list[bool]):
            self._pattern = pattern
            self._i = 0

        def run(self, task, actor):
            should_succeed = self._pattern[self._i % len(self._pattern)]
            self._i += 1
            if should_succeed:
                with get_conn() as conn:
                    conn.execute("UPDATE employees SET job_title = 'Changed Title' WHERE employee_code = 'EMP-0001'")
            return AgentRunResult(success=should_succeed, steps=1, message="", tokens={})

    task = _passthrough_task(verification={
        "type": "custom", "verifier": "employee.field_equals",
        "args": {"employee_code": "EMP-0001", "field": "job_title", "value": "Changed Title"},
    })
    adapter = SometimesMutatingAdapter([True, False, True, False, True])
    result = run_stability(task, adapter, control_client, n=5)

    assert result.functional_result == FunctionalResult.FLAKY
    assert 0.0 < result.success_rate < 1.0


def test_stability_inconclusive_when_infra_exhausts_budget(control_client):
    adapter = StubAgentAdapter([("infra down", "INFRA_ERROR")])
    result = run_stability(_passthrough_task(), adapter, control_client, n=5)

    assert result.functional_result == FunctionalResult.INCONCLUSIVE
    assert result.reason == "INSUFFICIENT_VALID_ATTEMPTS"
    assert result.execution_health == ExecutionHealth.FAILED_INFRA
    assert result.total_attempt_count == 10  # 2 * n, the hard cap
    assert result.valid_sample_count == 0


def test_stability_recovers_through_transient_infra_failures(control_client):
    """First two attempts are infra failures (don't count), then it settles into passing —
    total attempts exceeds n but valid samples reach n, so this is RECOVERED, not CLEAN."""
    outcomes = [
        ("network blip", "NETWORK_ERROR"),
        ("network blip again", "NETWORK_ERROR"),
        AgentRunResult(success=True, steps=1, message="", tokens={}),
        AgentRunResult(success=True, steps=1, message="", tokens={}),
        AgentRunResult(success=True, steps=1, message="", tokens={}),
        AgentRunResult(success=True, steps=1, message="", tokens={}),
        AgentRunResult(success=True, steps=1, message="", tokens={}),
    ]

    class ScriptedAdapter:
        def __init__(self, outcomes):
            self._outcomes = outcomes
            self._i = 0

        def run(self, task, actor):
            outcome = self._outcomes[self._i]
            self._i += 1
            if isinstance(outcome, tuple):
                raise AgentAdapterError(*outcome)
            return outcome

    result = run_stability(_passthrough_task(), ScriptedAdapter(outcomes), control_client, n=5)
    assert result.functional_result == FunctionalResult.PASS
    assert result.execution_health == ExecutionHealth.RECOVERED
    assert result.total_attempt_count == 7
    assert result.valid_sample_count == 5


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------


def test_wilson_interval_all_success_is_high_and_narrow_at_top():
    low, high = wilson_interval(5, 5)
    assert high == pytest.approx(1.0, abs=0.001)
    assert low > 0.5


def test_wilson_interval_all_failure_is_low_and_narrow_at_bottom():
    low, high = wilson_interval(0, 5)
    assert low == pytest.approx(0.0, abs=0.001)
    assert high < 0.5


def test_wilson_interval_zero_n_is_degenerate():
    assert wilson_interval(0, 0) == (0.0, 0.0)


# ---------------------------------------------------------------------------
# queue / run state machine
# ---------------------------------------------------------------------------


def test_queue_dequeues_high_priority_first():
    manager = RunManager()
    manager.submit("low-1", "LOW")
    manager.submit("normal-1", "NORMAL")
    manager.submit("high-1", "HIGH")

    started = manager.start_next()
    assert started.run_id == "high-1"


def test_queue_fifo_within_same_priority():
    manager = RunManager()
    manager.submit("normal-1", "NORMAL")
    manager.submit("normal-2", "NORMAL")

    first = manager.start_next()
    assert first.run_id == "normal-1"


def test_only_one_active_run_at_a_time():
    manager = RunManager()
    manager.submit("run-a", "NORMAL")
    manager.submit("run-b", "NORMAL")

    manager.start_next()  # run-a becomes PREPARING (active)
    second = manager.start_next()
    assert second is None  # run-b stays queued while run-a occupies the environment


def test_run_completing_frees_the_environment_for_the_next_run():
    manager = RunManager()
    manager.submit("run-a", "NORMAL")
    manager.submit("run-b", "NORMAL")
    manager.start_next()

    manager.transition("run-a", RunState.RUNNING)
    manager.transition("run-a", RunState.COMPLETING)
    manager.transition("run-a", RunState.COMPLETED)

    next_run = manager.start_next()
    assert next_run.run_id == "run-b"


def test_invalid_transition_is_rejected():
    manager = RunManager()
    manager.submit("run-a", "NORMAL")
    with pytest.raises(InvalidTransitionError):
        manager.transition("run-a", RunState.COMPLETED)  # QUEUED can't jump straight to COMPLETED


def test_terminal_state_never_transitions_backward():
    manager = RunManager()
    manager.submit("run-a", "NORMAL")
    manager.start_next()
    manager.transition("run-a", RunState.CANCELLING)
    manager.transition("run-a", RunState.CANCELLED)

    with pytest.raises(InvalidTransitionError):
        manager.transition("run-a", RunState.RUNNING)


# ---------------------------------------------------------------------------
# artifacts
# ---------------------------------------------------------------------------


def test_artifacts_are_written_and_well_formed(control_client, tmp_path):
    adapter = StubAgentAdapter([AgentRunResult(success=True, steps=2, message="", tokens={"input": 10, "output": 5})])
    result = run_stability(_passthrough_task(), adapter, control_client, n=3)

    run_dir = write_run_artifacts(
        "test-run-001", [result],
        manifest={"catalog_version": "v1", "agent_profile": "stub", "provider": "none"},
        out_dir=tmp_path,
    )

    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    assert manifest["run_id"] == "test-run-001"
    assert manifest["catalog_version"] == "v1"

    task_results = json.loads((run_dir / "task_results.json").read_text())
    assert len(task_results) == 1
    assert task_results[0]["functional_result"] == "PASS"

    attempt_results = json.loads((run_dir / "attempt_results.json").read_text())
    assert len(attempt_results) == 3
    assert all(a["passed"] for a in attempt_results)

    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["total_attempts"] == 3
    assert metrics["avg_steps_on_pass"] == 2

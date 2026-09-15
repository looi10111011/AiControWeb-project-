"""Single-attempt lifecycle (spec 14/26): RESETTING -> FIXTURING -> PRECONDITION_CHECK ->
STARTING_AGENT -> RUNNING -> VERIFYING -> INTEGRITY_CHECK -> COMPLETED, or a failure state
at whichever step didn't make it through. `control_client` is anything exposing
`.post(path, json=...)` / `.get(path)` returning an object with `.status_code` / `.json()`
— either `httpx.Client(base_url=...)` against a real running Control Plane, or
`fastapi.testclient.TestClient(control_app)` for in-process tests. Both satisfy this shape,
which is what lets Phase 6 be tested without a real subprocess.
"""

import time
import uuid
from datetime import datetime, timezone

from benchmark_target.control.config import ENVIRONMENT_IDENTITY
from benchmark_target.runner.agent_adapters import AgentAdapterError
from benchmark_target.runner.models import FAILURE_CLASS_TO_ATTEMPT_STATE, AttemptResult, AttemptState, FailureClass

_KNOWN_FAILURE_CLASSES = {f.value for f in FailureClass}


def _check_state_diff_expectation(diff: dict, expected: dict) -> tuple[bool, str]:
    """Lenient shape match: an empty expected list means "none of these", a non-empty one
    means "at least this many" — the catalog's `{}` placeholder entries document intent
    (e.g. which table gets a row), not an exact-value diff."""
    for table, expectation in expected.items():
        actual = diff.get(table, {"created": [], "updated": [], "deleted": []})
        for change_kind in ("created", "updated", "deleted"):
            if change_kind not in expectation:
                continue
            expected_list = expectation[change_kind]
            actual_list = actual.get(change_kind, [])
            if len(expected_list) == 0 and len(actual_list) != 0:
                return False, f"{table}.{change_kind}: expected none, got {len(actual_list)}"
            if len(expected_list) > 0 and len(actual_list) < len(expected_list):
                return False, f"{table}.{change_kind}: expected >= {len(expected_list)}, got {len(actual_list)}"
    return True, "state diff matched expectation"


def _classify_agent_failure(failure_class_str: str) -> FailureClass:
    if failure_class_str in _KNOWN_FAILURE_CLASSES:
        return FailureClass(failure_class_str)
    return FailureClass.INFRA_ERROR  # unknown adapter failure — fail closed to infra, not silently swallowed


def run_attempt(task: dict, agent_adapter, control_client, actor: str | None = None) -> AttemptResult:
    attempt_id = uuid.uuid4().hex[:12]
    started_at = datetime.now(timezone.utc).isoformat()
    start_time = time.monotonic()
    actor = actor or task["actors"][0]

    def _finish(state: AttemptState, passed: bool, failure_class: FailureClass | None, detail: str, **kw) -> AttemptResult:
        return AttemptResult(
            attempt_id=attempt_id,
            task_id=task["task_id"],
            task_revision=task["revision"],
            state=state,
            passed=passed,
            failure_class=failure_class,
            detail=detail,
            duration_ms=(time.monotonic() - start_time) * 1000,
            started_at=started_at,
            finished_at=datetime.now(timezone.utc).isoformat(),
            **kw,
        )

    # RESETTING + FIXTURING (spec 14: every task starts from deterministic state)
    try:
        resp = control_client.post(
            "/fixtures/apply",
            json={"name": task["fixture"], "confirm_environment": ENVIRONMENT_IDENTITY},
        )
        if resp.status_code != 200:
            return _finish(AttemptState.SETUP_ERROR, False, FailureClass.SCENARIO_SETUP_ERROR,
                            f"fixture apply failed: HTTP {resp.status_code} {resp.text}")
    except Exception as e:  # noqa: BLE001 — control plane unreachable is infra, not the agent's fault
        return _finish(AttemptState.INFRA_ERROR, False, FailureClass.INFRA_ERROR, f"control plane unreachable: {e}")

    # PRECONDITION_CHECK (spec 15: a failing precondition is a setup error, never blamed on the agent)
    conditions = task.get("preconditions", [])
    if conditions:
        resp = control_client.post("/preconditions/check", json={"conditions": conditions})
        if resp.status_code != 200 or not resp.json().get("all_passed"):
            detail = resp.json() if resp.status_code == 200 else resp.text
            return _finish(AttemptState.SETUP_ERROR, False, FailureClass.SCENARIO_SETUP_ERROR,
                            f"precondition failed: {detail}")

    # Scoped before-snapshot, for the state-diff verification path and general auditability.
    snap_resp = control_client.post("/snapshot", json={"tables": None})
    if snap_resp.status_code != 200:
        return _finish(AttemptState.INFRA_ERROR, False, FailureClass.INFRA_ERROR,
                        f"snapshot failed: HTTP {snap_resp.status_code}")
    before_snapshot_id = snap_resp.json()["snapshot_id"]

    # STARTING_AGENT + RUNNING
    try:
        agent_result = agent_adapter.run(task, actor)
    except AgentAdapterError as e:
        failure_class = _classify_agent_failure(e.failure_class)
        attempt_state = FAILURE_CLASS_TO_ATTEMPT_STATE.get(failure_class, AttemptState.AGENT_ERROR)
        return _finish(attempt_state, False, failure_class, str(e))

    # VERIFYING (spec 16, Layer 1)
    verification = task["verification"]
    v_type = verification["type"]

    if v_type == "custom":
        verify_resp = control_client.post(
            "/verify", json={"verifier": verification["verifier"], "args": verification["args"]}
        )
        verify_result = verify_resp.json() if verify_resp.status_code == 200 else {"passed": False, "detail": verify_resp.text}
        verified = verify_result.get("passed", False)
    elif v_type == "state_diff":
        diff_resp = control_client.post("/state-diff", json={"before_snapshot_id": before_snapshot_id})
        if diff_resp.status_code != 200:
            return _finish(AttemptState.INFRA_ERROR, False, FailureClass.INFRA_ERROR,
                            f"state-diff failed: HTTP {diff_resp.status_code}")
        diff = diff_resp.json()["diff"]
        verified, detail = _check_state_diff_expectation(diff, verification["expected"])
        verify_result = {"passed": verified, "detail": detail, "diff": diff}
    else:
        # download_artifact / http_redirect: genuinely outside the Control Plane's DB-based
        # verification scope (spec 9's own runner does this inspection, not this engine).
        # Reported honestly as inconclusive rather than faked as a pass.
        return _finish(AttemptState.COMPLETED, False, FailureClass.INCONCLUSIVE,
                        f"verification type '{v_type}' requires runner-side artifact inspection, not implemented here",
                        steps=agent_result.steps, tokens=agent_result.tokens)

    if not verified:
        return _finish(AttemptState.VERIFYING, False, FailureClass.FUNCTIONAL_FAIL,
                        f"verification failed: {verify_result.get('detail')}",
                        steps=agent_result.steps, tokens=agent_result.tokens, verify_result=verify_result)

    # INTEGRITY_CHECK (spec 16, Layer 4)
    integrity_resp = control_client.get("/integrity")
    integrity_data = integrity_resp.json() if integrity_resp.status_code == 200 else {"healthy": False, "violations": ["integrity check unreachable"]}
    if not integrity_data.get("healthy", False):
        return _finish(AttemptState.INTEGRITY_CHECK, False, FailureClass.VERIFICATION_FAIL,
                        f"integrity violations found: {integrity_data.get('violations')}",
                        steps=agent_result.steps, tokens=agent_result.tokens,
                        verify_result=verify_result, integrity_violations=integrity_data.get("violations"))

    return _finish(AttemptState.COMPLETED, True, None, "passed",
                    steps=agent_result.steps, tokens=agent_result.tokens,
                    verify_result=verify_result, integrity_violations=[])

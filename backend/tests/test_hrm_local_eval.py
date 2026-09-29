"""W_gate_local_hrm: core/hrm_local_eval.py — suite ของ release gate บน benchmark_target

ไม่แตะ LLM/browser: agent ถูกแทนด้วย StubAgentAdapter ส่วน Control Plane + fixture + verifier รัน
จริงผ่าน TestClient บน SQLite จริง (เหมือน benchmark_target/tests/test_runner.py)
"""

from unittest.mock import patch

import pytest

from backend.app.config import settings
from backend.app.core import hrm_local_eval
from backend.app.core.hrm_local_eval import (
    HRM_LOCAL_TASK_IDS,
    attempt_to_eval_result,
    load_hrm_local_tasks,
    run_hrm_local_evaluation,
)
from backend.app.permission import rules
from backend.app.core.release_gate import _result_row, task_failed_on_infrastructure
from benchmark_target.app.seed import seed
from benchmark_target.runner.agent_adapters import AgentRunResult, StubAgentAdapter
from benchmark_target.runner.models import AttemptResult, AttemptState, FailureClass


def test_all_task_ids_exist_in_catalog():
    tasks = load_hrm_local_tasks()
    assert [t["task_id"] for t in tasks] == list(HRM_LOCAL_TASK_IDS)
    assert len(tasks) == 6


def test_missing_task_id_fails_loudly(monkeypatch):
    # ตัวหารของ success_rate เปลี่ยนเงียบๆ ไม่ได้ — id หายต้อง raise ทันที
    monkeypatch.setattr(hrm_local_eval, "HRM_LOCAL_TASK_IDS", (*HRM_LOCAL_TASK_IDS, "NO-SUCH-TASK"))
    with pytest.raises(RuntimeError, match="NO-SUCH-TASK"):
        load_hrm_local_tasks()


@pytest.mark.asyncio
async def test_no_task_passes_when_agent_does_nothing():
    """กัน task read-only (verifier จริงเสมอ) เข้ามาใน suite — agent ที่ไม่ทำอะไรต้องไม่ผ่านสักตัว
    และต้องไม่เชื่อ success=True ที่ agent รายงานเอง"""
    seed()
    stub = StubAgentAdapter([AgentRunResult(success=True, steps=3, message="stub says done", tokens={})])
    with patch.object(hrm_local_eval, "ensure_target_running"):
        report = await run_hrm_local_evaluation(adapter=stub, run_id="test-run")

    assert [r.name for r in report.results] == list(HRM_LOCAL_TASK_IDS)
    assert stub.call_count == len(HRM_LOCAL_TASK_IDS)
    passed = [r.name for r in report.results if r.success]
    assert passed == [], f"task ที่ผ่านโดย agent ไม่ได้ทำอะไร: {passed}"


@pytest.mark.asyncio
async def test_internal_navigation_is_allowed_only_inside_the_run_and_never_globally():
    """SSRF guard เปิดให้ localhost เฉพาะงานของ suite นี้ — ค่า global ต้องไม่ถูกแตะเลย และ
    crash กลาง task ต้องปิด scope กลับ"""
    global_before = settings.allow_internal_navigation
    seen = []

    def crashing_attempt(*_args, **_kwargs):
        seen.append((rules._internal_navigation_allowed(), settings.allow_internal_navigation))
        raise RuntimeError("boom")

    stub = StubAgentAdapter([AgentRunResult(success=True, steps=1, message="", tokens={})])
    with patch.object(hrm_local_eval, "ensure_target_running"),          patch("benchmark_target.runner.attempt_runner.run_attempt", side_effect=crashing_attempt):
        with pytest.raises(RuntimeError, match="boom"):
            await run_hrm_local_evaluation(adapter=stub)

    assert seen == [(True, global_before)]
    assert settings.allow_internal_navigation is global_before
    assert rules._internal_navigation_allowed() is global_before


# --- ensure_target_running: ห้ามวัดผลบนเว็บผิดตัว ---


def test_ensure_target_running_reuses_a_target_that_is_already_up():
    with patch.object(hrm_local_eval, "_target_is_up", return_value=True),          patch("uvicorn.Server") as server:
        hrm_local_eval.ensure_target_running()
    server.assert_not_called()


def test_ensure_target_running_refuses_a_port_held_by_another_program():
    with patch.object(hrm_local_eval, "_target_is_up", return_value=False),          patch.object(hrm_local_eval, "_port_is_taken", return_value=True),          patch("uvicorn.Server") as server:
        with pytest.raises(RuntimeError, match="8100"):
            hrm_local_eval.ensure_target_running()
    server.assert_not_called()


def test_ensure_target_running_starts_target_and_waits_until_it_answers():
    answers = iter([False, False, True])  # ก่อนเปิด, ยังไม่พร้อม, พร้อม
    with patch.object(hrm_local_eval, "_target_is_up", side_effect=lambda: next(answers)),          patch.object(hrm_local_eval, "_port_is_taken", return_value=False),          patch.object(hrm_local_eval.time, "sleep"),          patch("uvicorn.Server") as server:
        hrm_local_eval.ensure_target_running()
    server.assert_called_once()


def test_ensure_target_running_times_out_when_target_never_answers(monkeypatch):
    monkeypatch.setattr(hrm_local_eval, "_TARGET_READY_TIMEOUT_SECONDS", 0.0)
    with patch.object(hrm_local_eval, "_target_is_up", return_value=False),          patch.object(hrm_local_eval, "_port_is_taken", return_value=False),          patch("uvicorn.Server"):
        with pytest.raises(RuntimeError, match="ไม่พร้อม"):
            hrm_local_eval.ensure_target_running()


def _attempt(**overrides) -> AttemptResult:
    fields = dict(
        attempt_id="a1", task_id="T-1", task_revision=1, state=AttemptState.COMPLETED, passed=True,
        detail="passed", steps=5, duration_ms=2500.0,
        tokens={"input": 100, "output": 20, "cache_read": 30, "cache_creation": 4},
    )
    fields.update(overrides)
    return AttemptResult(**fields)


_TASK = {"task_id": "T-1", "goal": {"description": "do a thing"}}


def test_pass_maps_to_success_with_summed_tokens():
    result = attempt_to_eval_result(_TASK, _attempt(), AgentRunResult(True, 5, "", {}, approval_count=2))
    assert result.success is True
    assert (result.name, result.steps, result.total_tokens) == ("T-1", 5, 154)
    assert result.latency_seconds == 2.5
    assert result.approval_count == 2
    assert result.error is None


def test_functional_fail_is_a_failure_not_infra():
    attempt = _attempt(passed=False, failure_class=FailureClass.FUNCTIONAL_FAIL, detail="verification failed")
    result = attempt_to_eval_result(_TASK, attempt, None)
    assert result.success is False
    assert result.steps == 5  # ลงมือทำจริง นับเป็นความล้มเหลวของ agent
    assert result.error is None


def test_provider_outage_is_visible_to_the_gate_as_infrastructure():
    """provider ล่มกลางรัน: run_task คืนปกติด้วย steps=0 แต่ verifier ไม่เจออะไรใน DB -> attempt
    เป็น FUNCTIONAL_FAIL ข้อความของ agent ต้องไปถึงแถว ไม่งั้น gate นับเป็นความล้มเหลวจริง"""
    attempt = _attempt(
        passed=False, state=AttemptState.VERIFYING, failure_class=FailureClass.FUNCTIONAL_FAIL,
        detail="verification failed: no matching leave request found", steps=0, tokens={},
    )
    agent = AgentRunResult(success=False, steps=0, message="429 You exceeded your current quota", tokens={})
    row = _result_row(attempt_to_eval_result(_TASK, attempt, agent))
    assert row["message"].startswith("verification failed: no matching leave request found | agent: 429")
    assert task_failed_on_infrastructure(row) is True


def test_real_agent_failure_is_not_mistaken_for_infrastructure():
    attempt = _attempt(passed=False, failure_class=FailureClass.FUNCTIONAL_FAIL, detail="verification failed: x")
    agent = AgentRunResult(success=True, steps=5, message="Done, saved the record", tokens={})
    assert task_failed_on_infrastructure(_result_row(attempt_to_eval_result(_TASK, attempt, agent))) is False


def test_infra_failure_reports_zero_steps_and_error():
    """steps=0 + error ทำให้ task_failed_on_infrastructure()/run_is_invalid() ของ gate แยกออกได้"""
    attempt = _attempt(
        passed=False, state=AttemptState.INFRA_ERROR, failure_class=FailureClass.INFRA_ERROR,
        detail="control plane unreachable", steps=0, tokens={},
    )
    result = attempt_to_eval_result(_TASK, attempt, None)
    assert result.success is False
    assert result.steps == 0
    assert result.error == "control plane unreachable"

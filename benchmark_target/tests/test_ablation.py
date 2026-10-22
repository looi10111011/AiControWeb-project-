"""W_ablation: run_ablation.py / analyze_ablation.py — arm ต้อง patch ตามที่ประกาศและคืนค่าเดิมหลังออก,
schedule ต้องสมดุลทุก rep, และ dry-run (stub agent) ต้องได้ FUNCTIONAL_FAIL ทุกช่อง (ไม่มี task ที่ผ่าน
ได้โดยไม่ทำอะไร) โดยไม่แตะ LLM/browser
"""

import asyncio
import json
from collections import Counter

import pytest

from backend.app.config import settings
from backend.app.core import llm, orchestrator
from benchmark_target import analyze_ablation, run_ablation


def _snapshot_patched_values() -> dict:
    return {
        (target is settings, attribute): getattr(target, attribute)
        for arm_patches in run_ablation.ARM_PATCHES.values()
        for target, attribute, _ in arm_patches + run_ablation.COMMON_PATCHES
    }


def test_every_arm_restores_original_values():
    before = _snapshot_patched_values()
    for arm in run_ablation.ARM_NAMES:
        with run_ablation.arm_patches(arm):
            pass
    assert _snapshot_patched_values() == before


def test_full_only_disables_procedural_capture():
    compact_before = orchestrator._COMPACT_AFTER_STEPS
    with run_ablation.arm_patches("FULL"):
        assert settings.enable_procedural_memory_capture is False
        assert orchestrator._COMPACT_AFTER_STEPS == compact_before
        assert orchestrator._MAX_PREMATURE_TRUE_FINISH_RETRIES > 0


def test_baseline_disables_all_three_groups():
    with run_ablation.arm_patches("BASELINE"):
        assert orchestrator._COMPACT_AFTER_STEPS >= 10**9
        assert settings.snapshot_max_elements >= 10**9
        assert settings.read_page_data_max_rows >= 10**9
        for name in run_ablation._VERIFY_RETRY_CONSTANTS:
            assert getattr(orchestrator, name) == 0, name
        assert asyncio.run(orchestrator._scan_created_item_in_table(None, "anything")) is True


def test_patched_helpers_accept_real_call_signatures():
    """ลายเซ็นต้องตรงกับที่ run_task() เรียกจริง ไม่งั้น arm พังกลางรันจริงแทนที่จะพังในเทสต์"""
    messages = [{"role": "user", "content": "x"}]
    with run_ablation.arm_patches("BASELINE"):
        assert orchestrator._compact_stale_user_turns(messages, "goal") == (messages, 0)
        assert orchestrator._dedupe_stale_snapshots(messages, keep_last_full=1) is messages
        assert orchestrator._dedupe_stale_gated(messages, keep_last_full=0) == (messages, 0)
        sections = orchestrator._resolve_prompt_sections(
            frozenset(), goal="g", plan_text=None, elements=[], allow_fill_secret=False,
        )
        assert sections == llm.ALL_PROMPT_SECTIONS


def test_schedule_is_balanced_and_deterministic():
    tasks, arms = ["A", "B", "C"], list(run_ablation.ARM_NAMES)
    schedule = run_ablation.build_schedule(tasks, arms, reps=3, seed=7)
    assert schedule == run_ablation.build_schedule(tasks, arms, reps=3, seed=7)
    assert len(schedule) == len(tasks) * len(arms) * 3
    for rep in (1, 2, 3):
        cells = Counter((task, arm) for r, task, arm in schedule if r == rep)
        assert set(cells) == {(t, a) for t in tasks for a in arms}
        assert set(cells.values()) == {1}


def test_unsupported_verification_type_is_rejected():
    with pytest.raises(SystemExit, match="INCONCLUSIVE"):
        run_ablation.load_selected_tasks(["AUTH-LOGIN-INVALID-PASSWORD"])


def test_total_tokens_does_not_double_count_openai_cache():
    tokens = {"input": 1000, "output": 10, "cache_read": 400, "cache_creation": 0}
    assert analyze_ablation.total_tokens(tokens, "openai") == 1010
    assert analyze_ablation.uncached_input_tokens(tokens, "openai") == 600
    assert analyze_ablation.total_tokens(tokens, "anthropic") == 1410


def _record(arm, task_id, rep, passed, claimed, valid=True, tokens=None, steps=5, duration=10.0):
    return {
        "arm": arm, "task_id": task_id, "rep": rep, "try": 1, "valid": valid, "passed": passed,
        "claimed_success": claimed, "telemetry_task_id": None, "steps": steps,
        "tokens": tokens or {"input": 1000, "output": 10, "cache_read": 0, "cache_creation": 0},
        "duration_seconds": duration,
    }


def test_summary_counts_false_success_and_skips_infra():
    records = [
        _record("FULL", "T1", 1, passed=True, claimed=True),
        _record("FULL", "T1", 2, passed=False, claimed=True),
        _record("FULL", "T1", 3, passed=False, claimed=None, valid=False),
        _record("NO_VERIFY", "T1", 1, passed=False, claimed=True,
                tokens={"input": 2000, "output": 10, "cache_read": 0, "cache_creation": 0}),
    ]
    rows = analyze_ablation.enrich(records, "openai", {}, {})
    invalid = Counter(r["arm"] for r in records if not r["valid"])
    full, no_verify = analyze_ablation.summarize_arms(rows, ["FULL", "NO_VERIFY"], invalid)
    assert (full["valid_attempts"], full["infra_failures"], full["passed"], full["false_success"]) == (2, 1, 1, 1)
    assert full["success_rate"] == 0.5
    assert no_verify["false_success"] == 1
    (delta,) = analyze_ablation.paired_deltas(rows, ["FULL", "NO_VERIFY"], "FULL")
    assert delta["paired_tasks"] == 1
    assert delta["mean_success_rate_diff"] == -0.5
    assert delta["median_pct_change_total_tokens"] == pytest.approx((2010 - 1010) / 1010 * 100)


_CODEX_400 = (
    "Task stopped by an unexpected error at step 0: BadRequestError: Error code: 400 - "
    "{'detail': \"The 'gpt-5.5' model is not supported when using Codex with a ChatGPT account.\"}"
)


def test_provider_error_before_first_step_is_infra():
    from benchmark_target.runner.agent_adapters import AgentRunResult

    assert run_ablation.provider_failed_before_acting(AgentRunResult(False, 0, _CODEX_400, {}))
    assert not run_ablation.provider_failed_before_acting(AgentRunResult(False, 4, _CODEX_400, {}))
    assert not run_ablation.provider_failed_before_acting(AgentRunResult(True, 0, "stub: no action taken", {}))
    assert not run_ablation.provider_failed_before_acting(None)


def test_session_stops_when_provider_keeps_failing(tmp_path, monkeypatch):
    """smoke 2026-10-22: codex เลิกรับ gpt-5.5 — ทุกช่องต้องเป็น infra (ไม่โทษ agent) และหยุด session"""
    from benchmark_target.runner.agent_adapters import AgentRunResult, StubAgentAdapter

    monkeypatch.setattr(
        run_ablation, "_make_adapter",
        lambda *args: StubAgentAdapter([AgentRunResult(False, 0, _CODEX_400, {})]),
    )
    with pytest.raises(SystemExit, match="--resume"):
        run_ablation.main([
            "--dry-run", "--tasks", "PIM-REACTIVATE-EMPLOYEE", "--arms", "FULL,NO_VERIFY,BASELINE",
            "--reps", "1", "--max-infra-retries", "0", "--max-consecutive-infra", "2",
            "--out-dir", str(tmp_path),
        ])
    (results_path,) = tmp_path.glob("*.jsonl")
    records = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 2
    assert all(not r["valid"] and r["provider_failure"] for r in records)


def test_resume_refuses_when_model_changed(tmp_path, monkeypatch):
    results_path = tmp_path / "abl-x.jsonl"
    results_path.with_suffix(".manifest.json").write_text(json.dumps({
        "session_id": "abl-x", "dry_run": False, "provider": "openai", "model": "retired-model",
    }), encoding="utf-8")
    monkeypatch.setattr(settings, "openai_model", "new-model")
    with pytest.raises(SystemExit, match="เริ่ม session ใหม่"):
        run_ablation.main(["--resume", str(results_path)])


def test_analysis_reclassifies_records_written_before_provider_failure_field(tmp_path):
    results_path = tmp_path / "old.jsonl"
    results_path.with_suffix(".manifest.json").write_text(json.dumps({"session_id": "old"}), encoding="utf-8")
    old = {**_record("FULL", "T1", 1, passed=False, claimed=False, steps=0), "agent_message": _CODEX_400}
    results_path.write_text(json.dumps(old) + "\n", encoding="utf-8")
    _, (record,) = analyze_ablation.load_session(results_path)
    assert record["provider_failure"] is True
    assert record["valid"] is False


def test_dry_run_end_to_end(tmp_path):
    """stub บอกสำเร็จเสมอแต่ไม่ทำอะไร — ทุกช่องต้อง valid, DB ไม่ผ่าน และนับเป็น false success"""
    results_path = run_ablation.main([
        "--dry-run", "--tasks", "PIM-REACTIVATE-EMPLOYEE,TIME-APPROVE-BY-SUPERVISOR",
        "--arms", "FULL,NO_VERIFY", "--reps", "1", "--out-dir", str(tmp_path),
    ])
    records = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 4
    assert all(r["valid"] and not r["passed"] and r["claimed_success"] for r in records)
    assert {r["failure_class"] for r in records} == {"FUNCTIONAL_FAIL"}

    result = analyze_ablation.analyze(
        results_path, tmp_path / "no_usage.jsonl", tmp_path / "no_trace.jsonl",
    )
    assert result["report"].exists()
    assert [s["false_success"] for s in result["summaries"]] == [2, 2]

    # resume บน session ที่ครบแล้วต้องไม่รันเพิ่ม
    run_ablation.main(["--resume", str(results_path)])
    assert len(results_path.read_text(encoding="utf-8").splitlines()) == 4

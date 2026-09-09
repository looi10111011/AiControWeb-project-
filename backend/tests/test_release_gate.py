import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.app.core.evaluation import EvaluationReport, TaskEvalResult
from backend.app.core.release_gate import (
    build_summary,
    compare_against_baseline,
    load_latest_summary,
    run_release_gate,
    save_summary,
    _git_commit_short,
    _sanitize_for_filename,
    build_noise_baseline,
    load_recent_summaries,
    METRIC_NAMES,
    compare_tasks,
    run_release_gate_repeated,
    save_flakiness_report,
    success_rate_stats,
    task_flakiness,
    run_is_invalid,
)

# ทุกเทสต์ mock subprocess/filesystem/eval suite ตรงๆ (ไม่รัน browser/LLM จริง) เหมือน
# test_evaluation.py — tmp_path fixture ของ pytest ให้ directory ชั่วคราวจริงสำหรับเทส
# save_summary()/load_latest_summary() (ต้องเขียน/อ่านไฟล์จริงเพื่อพิสูจน์ round-trip)


def _fake_task_result(success=True, latency=1.0, llm_calls=2, approval_count=0, fastpath=False, recoveries=0):
    return TaskEvalResult(
        name="x", goal="g", success=success, steps=llm_calls, total_tokens=100, message="ok",
        latency_seconds=latency, llm_calls=llm_calls, approval_count=approval_count,
        fastpath=fastpath, recoveries=recoveries,
    )


# --- _git_commit_short() ---


def test_git_commit_short_returns_stdout_on_success():
    fake_result = MagicMock(stdout="abc1234\n")
    with patch("backend.app.core.release_gate.subprocess.run", return_value=fake_result):
        assert _git_commit_short() == "abc1234"


def test_git_commit_short_returns_unknown_when_git_unavailable():
    with patch("backend.app.core.release_gate.subprocess.run", side_effect=FileNotFoundError()):
        assert _git_commit_short() == "unknown"


def test_git_commit_short_returns_unknown_on_nonzero_exit():
    import subprocess as real_subprocess
    with patch(
        "backend.app.core.release_gate.subprocess.run",
        side_effect=real_subprocess.CalledProcessError(128, "git"),
    ):
        assert _git_commit_short() == "unknown"


# --- _sanitize_for_filename() ---


def test_sanitize_for_filename_replaces_path_separators_and_spaces():
    assert _sanitize_for_filename("claude-haiku-4-5-20251001") == "claude-haiku-4-5-20251001"
    assert _sanitize_for_filename("gemini/flash:pro") == "gemini_flash_pro"
    assert _sanitize_for_filename("model with spaces") == "model_with_spaces"


def test_sanitize_for_filename_empty_string_falls_back_to_unknown():
    assert _sanitize_for_filename("") == "unknown"


# --- build_summary() ---


def test_build_summary_includes_tags_and_aggregate():
    report = EvaluationReport(results=[_fake_task_result(success=True), _fake_task_result(success=False)])

    summary = build_summary(report, git_commit="abc123", model="claude-haiku-4-5", provider="anthropic")

    assert summary["git_commit"] == "abc123"
    assert summary["model"] == "claude-haiku-4-5"
    assert summary["provider"] == "anthropic"
    assert summary["task_count"] == 2
    assert summary["aggregate"]["success_rate"] == 0.5
    assert "timestamp" in summary


# --- save_summary() / load_latest_summary() round-trip ---


def test_save_summary_writes_readable_json(tmp_path):
    summary = build_summary(
        EvaluationReport(results=[_fake_task_result()]),
        git_commit="abc123", model="model-x", provider="anthropic",
    )

    path = save_summary(summary, results_dir=str(tmp_path))

    assert path.exists()
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["git_commit"] == "abc123"


def test_save_summary_never_overwrites_previous_runs(tmp_path):
    summary1 = build_summary(EvaluationReport(results=[]), git_commit="c1", model="m", provider="p")
    path1 = save_summary(summary1, results_dir=str(tmp_path))
    summary2 = dict(summary1)
    summary2["timestamp"] = summary1["timestamp"] + 1  # force a different filename
    path2 = save_summary(summary2, results_dir=str(tmp_path))

    assert path1 != path2
    assert path1.exists() and path2.exists()


def test_load_latest_summary_returns_none_when_directory_missing(tmp_path):
    assert load_latest_summary(str(tmp_path / "does-not-exist")) is None


def test_load_latest_summary_returns_none_when_no_json_files(tmp_path):
    assert load_latest_summary(str(tmp_path)) is None


def test_load_latest_summary_picks_highest_timestamp_field_not_mtime(tmp_path):
    old_summary = {"timestamp": 100.0, "git_commit": "old", "aggregate": {}}
    new_summary = {"timestamp": 200.0, "git_commit": "new", "aggregate": {}}
    # เขียน "new" ก่อน "old" ตั้งใจ (mtime ของ old จะใหม่กว่าจริงถ้า sort ผิดตัว) — ต้องเลือก
    # จาก field "timestamp" ข้างในไฟล์ ไม่ใช่ mtime ของระบบไฟล์
    (tmp_path / "new.json").write_text(json.dumps(new_summary), encoding="utf-8")
    (tmp_path / "old.json").write_text(json.dumps(old_summary), encoding="utf-8")

    latest = load_latest_summary(str(tmp_path))

    assert latest["git_commit"] == "new"


def test_load_latest_summary_excludes_given_path(tmp_path):
    summary = {"timestamp": 100.0, "git_commit": "only-one", "aggregate": {}}
    path = tmp_path / "only.json"
    path.write_text(json.dumps(summary), encoding="utf-8")

    assert load_latest_summary(str(tmp_path), exclude_path=path) is None


def test_load_latest_summary_skips_unreadable_json_files(tmp_path):
    (tmp_path / "corrupt.json").write_text("not valid json{{{", encoding="utf-8")
    good = {"timestamp": 100.0, "git_commit": "good", "aggregate": {}}
    (tmp_path / "good.json").write_text(json.dumps(good), encoding="utf-8")

    latest = load_latest_summary(str(tmp_path))

    assert latest["git_commit"] == "good"


# --- compare_against_baseline() ---


def _summary_with(**aggregate) -> dict:
    return {"aggregate": aggregate}


def test_compare_against_baseline_flags_success_rate_drop_as_failure():
    current = _summary_with(success_rate=0.5, p50_latency_seconds=0, p95_latency_seconds=0,
                             avg_llm_calls=0, avg_tokens=0, approval_rate=0, fastpath_hit_rate=0, recovery_rate=0)
    baseline = _summary_with(success_rate=0.9, p50_latency_seconds=0, p95_latency_seconds=0,
                              avg_llm_calls=0, avg_tokens=0, approval_rate=0, fastpath_hit_rate=0, recovery_rate=0)

    comparisons = compare_against_baseline(current, baseline, max_regression_pct=10.0)

    success = next(c for c in comparisons if c.metric == "success_rate")
    assert success.passed is False
    assert success.pct_change < 0


def test_compare_against_baseline_passes_success_rate_within_threshold():
    current = _summary_with(success_rate=0.86, p50_latency_seconds=0, p95_latency_seconds=0,
                             avg_llm_calls=0, avg_tokens=0, approval_rate=0, fastpath_hit_rate=0, recovery_rate=0)
    baseline = _summary_with(success_rate=0.9, p50_latency_seconds=0, p95_latency_seconds=0,
                              avg_llm_calls=0, avg_tokens=0, approval_rate=0, fastpath_hit_rate=0, recovery_rate=0)

    comparisons = compare_against_baseline(current, baseline, max_regression_pct=10.0)

    success = next(c for c in comparisons if c.metric == "success_rate")
    assert success.passed is True  # ~4.4% drop, within 10% threshold


def test_compare_against_baseline_flags_latency_increase_as_failure():
    current = _summary_with(success_rate=1.0, p50_latency_seconds=20.0, p95_latency_seconds=0,
                             avg_llm_calls=0, avg_tokens=0, approval_rate=0, fastpath_hit_rate=0, recovery_rate=0)
    baseline = _summary_with(success_rate=1.0, p50_latency_seconds=10.0, p95_latency_seconds=0,
                              avg_llm_calls=0, avg_tokens=0, approval_rate=0, fastpath_hit_rate=0, recovery_rate=0)

    comparisons = compare_against_baseline(current, baseline, max_regression_pct=10.0)

    latency = next(c for c in comparisons if c.metric == "p50_latency_seconds")
    assert latency.passed is False  # 100% slower
    assert latency.pct_change == 100.0


def test_compare_against_baseline_higher_llm_calls_and_tokens_fail_but_higher_success_rate_passes():
    current = _summary_with(success_rate=1.0, p50_latency_seconds=0, p95_latency_seconds=0,
                             avg_llm_calls=20, avg_tokens=5000, approval_rate=0, fastpath_hit_rate=0, recovery_rate=0)
    baseline = _summary_with(success_rate=0.8, p50_latency_seconds=0, p95_latency_seconds=0,
                              avg_llm_calls=10, avg_tokens=1000, approval_rate=0, fastpath_hit_rate=0, recovery_rate=0)

    comparisons = compare_against_baseline(current, baseline, max_regression_pct=10.0)
    by_metric = {c.metric: c for c in comparisons}

    assert by_metric["success_rate"].passed is True  # improved, not a regression
    assert by_metric["avg_llm_calls"].passed is False  # 2x more LLM calls per task
    assert by_metric["avg_tokens"].passed is False  # 5x more tokens per task


def test_compare_against_baseline_zero_baseline_and_zero_current_is_not_a_regression():
    current = _summary_with(success_rate=1.0, p50_latency_seconds=0, p95_latency_seconds=0,
                             avg_llm_calls=0, avg_tokens=0, approval_rate=0, fastpath_hit_rate=0, recovery_rate=0.0)
    baseline = _summary_with(success_rate=1.0, p50_latency_seconds=0, p95_latency_seconds=0,
                              avg_llm_calls=0, avg_tokens=0, approval_rate=0, fastpath_hit_rate=0, recovery_rate=0.0)

    comparisons = compare_against_baseline(current, baseline, max_regression_pct=10.0)

    recovery = next(c for c in comparisons if c.metric == "recovery_rate")
    assert recovery.passed is True
    assert recovery.pct_change == 0.0


def test_compare_against_baseline_zero_baseline_nonzero_current_higher_is_better_passes():
    """recovery_rate baseline=0 (ไม่เคยมี task ต้องพึ่ง repair เลย) แล้ว current เจอ recovery
    จริงและสำเร็จ (recovery_rate>0) — ควรนับเป็นดีขึ้น ไม่ใช่ regression"""
    current = _summary_with(success_rate=1.0, p50_latency_seconds=0, p95_latency_seconds=0,
                             avg_llm_calls=0, avg_tokens=0, approval_rate=0, fastpath_hit_rate=0, recovery_rate=1.0)
    baseline = _summary_with(success_rate=1.0, p50_latency_seconds=0, p95_latency_seconds=0,
                              avg_llm_calls=0, avg_tokens=0, approval_rate=0, fastpath_hit_rate=0, recovery_rate=0.0)

    comparisons = compare_against_baseline(current, baseline, max_regression_pct=10.0)

    recovery = next(c for c in comparisons if c.metric == "recovery_rate")
    assert recovery.pct_change == float("inf")
    assert recovery.passed is True  # inf change on a higher-is-better metric is an improvement


def test_compare_against_baseline_uses_settings_default_when_threshold_omitted(monkeypatch):
    from backend.app.config import settings
    monkeypatch.setattr(settings, "release_gate_max_regression_pct", 50.0)
    current = _summary_with(success_rate=0.6, p50_latency_seconds=0, p95_latency_seconds=0,
                             avg_llm_calls=0, avg_tokens=0, approval_rate=0, fastpath_hit_rate=0, recovery_rate=0)
    baseline = _summary_with(success_rate=1.0, p50_latency_seconds=0, p95_latency_seconds=0,
                              avg_llm_calls=0, avg_tokens=0, approval_rate=0, fastpath_hit_rate=0, recovery_rate=0)

    comparisons = compare_against_baseline(current, baseline)  # no explicit max_regression_pct

    success = next(c for c in comparisons if c.metric == "success_rate")
    assert success.passed is True  # 40% drop, but threshold overridden to 50%


# --- run_release_gate() ---


@pytest.mark.asyncio
async def test_run_release_gate_combines_all_three_suites_and_saves_summary(tmp_path):
    sauce_report = EvaluationReport(results=[_fake_task_result(success=True)])
    orangehrm_report = EvaluationReport(results=[_fake_task_result(success=True)])
    from backend.app.core.miniwob_eval import MiniWobReport, MiniWobResult
    miniwob_report = MiniWobReport(results=[
        MiniWobResult(task="t", utterance="u", success=True, reward=1.0, steps=2, total_tokens=50, message="ok"),
    ])

    with patch("backend.app.core.release_gate.run_evaluation", AsyncMock(return_value=sauce_report)), \
         patch("backend.app.core.release_gate.run_orangehrm_evaluation", AsyncMock(return_value=orangehrm_report)), \
         patch("backend.app.core.release_gate.run_miniwob_evaluation", AsyncMock(return_value=miniwob_report)), \
         patch("backend.app.core.release_gate._git_commit_short", return_value="deadbee"), \
         patch("backend.app.core.orchestrator.Orchestrator._llm_backend", return_value=(MagicMock(), "model-x", None, None, None)):
        outcome = await run_release_gate(provider="anthropic", results_dir=str(tmp_path))

    assert outcome["summary"]["task_count"] == 3  # 1 sauce + 1 orangehrm + 1 miniwob
    assert outcome["summary"]["git_commit"] == "deadbee"
    assert outcome["baseline"] is None  # first run ever, nothing to compare against
    assert outcome["passed"] is True
    assert outcome["comparisons"] == []
    from pathlib import Path
    assert Path(outcome["saved_path"]).exists()


@pytest.mark.asyncio
async def test_run_release_gate_skips_orangehrm_and_miniwob_when_disabled(tmp_path):
    sauce_report = EvaluationReport(results=[_fake_task_result(success=True)])

    with patch("backend.app.core.release_gate.run_evaluation", AsyncMock(return_value=sauce_report)), \
         patch("backend.app.core.release_gate.run_orangehrm_evaluation", AsyncMock()) as mock_orangehrm, \
         patch("backend.app.core.release_gate.run_miniwob_evaluation", AsyncMock()) as mock_miniwob, \
         patch("backend.app.core.orchestrator.Orchestrator._llm_backend", return_value=(MagicMock(), "model-x", None, None, None)):
        outcome = await run_release_gate(
            provider="anthropic", results_dir=str(tmp_path), include_orangehrm=False, include_miniwob=False,
        )

    mock_orangehrm.assert_not_called()
    mock_miniwob.assert_not_called()
    assert outcome["summary"]["task_count"] == 1


@pytest.mark.asyncio
async def test_run_release_gate_compares_against_prior_run_in_same_dir(tmp_path):
    prior_summary = build_summary(
        EvaluationReport(results=[_fake_task_result(success=True)]),
        git_commit="old-commit", model="model-x", provider="anthropic",
    )
    save_summary(prior_summary, results_dir=str(tmp_path))

    new_report = EvaluationReport(results=[_fake_task_result(success=False)])  # regression: was succeeding, now fails
    with patch("backend.app.core.release_gate.run_evaluation", AsyncMock(return_value=new_report)), \
         patch("backend.app.core.release_gate.run_orangehrm_evaluation", AsyncMock(return_value=EvaluationReport())), \
         patch("backend.app.core.release_gate.run_miniwob_evaluation", AsyncMock(return_value=EvaluationReport())), \
         patch("backend.app.core.orchestrator.Orchestrator._llm_backend", return_value=(MagicMock(), "model-x", None, None, None)):
        outcome = await run_release_gate(provider="anthropic", results_dir=str(tmp_path))

    assert outcome["baseline"] is not None
    assert outcome["baseline"]["git_commit"] == "old-commit"
    assert outcome["passed"] is False  # success_rate dropped from 1.0 to 0.0


@pytest.mark.asyncio
async def test_run_release_gate_uses_explicit_baseline_path_when_given(tmp_path):
    baseline_summary = build_summary(
        EvaluationReport(results=[_fake_task_result(success=True)]),
        git_commit="pinned-baseline", model="model-x", provider="anthropic",
    )
    baseline_path = tmp_path / "pinned.json"
    baseline_path.write_text(json.dumps(baseline_summary), encoding="utf-8")

    new_report = EvaluationReport(results=[_fake_task_result(success=True)])
    with patch("backend.app.core.release_gate.run_evaluation", AsyncMock(return_value=new_report)), \
         patch("backend.app.core.release_gate.run_orangehrm_evaluation", AsyncMock(return_value=EvaluationReport())), \
         patch("backend.app.core.release_gate.run_miniwob_evaluation", AsyncMock(return_value=EvaluationReport())), \
         patch("backend.app.core.orchestrator.Orchestrator._llm_backend", return_value=(MagicMock(), "model-x", None, None, None)):
        outcome = await run_release_gate(
            provider="anthropic", results_dir=str(tmp_path / "out"), baseline_path=str(baseline_path),
        )

    assert outcome["baseline"]["git_commit"] == "pinned-baseline"
    assert outcome["passed"] is True


# ---------------- W_gate_per_task: summary ต้องเก็บผลรายตัว ไม่ใช่แค่ aggregate ----------------

def test_build_summary_keeps_one_row_per_task():
    """baseline 2 รอบแรกบอกได้แค่ "14/15 ผ่าน" และตอนวัดผล P4.1 ก็อธิบายไม่ได้ว่า approval
    ที่เด้งขึ้นมาจาก task ไหน เพราะไม่เคยเก็บรายตัวไว้เลย"""
    from backend.app.core.evaluation import EvaluationReport, TaskEvalResult

    report = EvaluationReport(results=[
        TaskEvalResult("a", "goal a", True, 3, 100, "ok", latency_seconds=2.0,
                       llm_calls=3, approval_count=1),
        TaskEvalResult("b", "goal b", False, 0, 0, "", error="RuntimeError: x"),
    ])

    summary = build_summary(report, git_commit="abc", model="m", provider="p")
    rows = summary["results"]

    assert [r["name"] for r in rows] == ["a", "b"]
    assert rows[0]["approval_count"] == 1
    assert rows[1]["success"] is False
    assert rows[1]["error"] == "RuntimeError: x"
    assert summary["task_count"] == len(rows)


def test_build_summary_reads_miniwob_results_through_the_same_duck_typed_row():
    """MiniWobResult ใช้ชื่อ field "task" ไม่ใช่ "name" และไม่มี goal — ต้องไม่พังและต้องได้
    ชื่อ task กลับมา (เหตุผลเดียวกับที่ EvaluationReport reuse ทั้งสอง dataclass ได้)"""
    from backend.app.core.evaluation import EvaluationReport
    from backend.app.core.miniwob_eval import MiniWobResult

    report = EvaluationReport(results=[
        MiniWobResult(task="click-test", utterance="u", success=True, reward=1.0,
                      steps=2, total_tokens=50, message="", approval_count=2),
    ])

    row = build_summary(report, git_commit="abc", model="m", provider="p")["results"][0]

    assert row["name"] == "click-test"
    assert row["approval_count"] == 2


# --- W_gate_noise_floor (W91): baseline = median ของหลายรัน + metric ที่แกว่งเองเกินเกณฑ์
# ต้อง "รายงานได้ แต่ตัดสินไม่ได้" — เกิดจากการวัดจริงว่า gate รันซ้ำบน commit เดิมโดยไม่มี
# อะไรเปลี่ยนเลย ยัง swing เกิน 10% ด้วยตัวมันเอง (success_rate -14.3%, p95 +49.5%) ---


def _summary_with(**aggregate) -> dict:
    base = {m: 0.0 for m in METRIC_NAMES}
    base.update(aggregate)
    return {"git_commit": "abc1234", "model": "m", "provider": "p",
            "timestamp": 1.0, "aggregate": base}


def test_build_noise_baseline_uses_median_not_mean():
    """รันเดียวที่ดวงดี/ดวงร้ายต้องไม่ลาก baseline ทั้งก้อน — 124k/161k/174k/211k
    ต้องได้ median 167.5k ไม่ใช่ mean"""
    summaries = [_summary_with(avg_tokens=v) for v in (124_000.0, 161_000.0, 174_000.0, 211_000.0)]
    baseline, spread = build_noise_baseline(summaries)

    assert baseline["aggregate"]["avg_tokens"] == 167_500.0
    assert baseline["baseline_run_count"] == 4
    # spread = (max-min)/median*100 = "ตัวเลขนี้แกว่งได้เองแค่ไหนโดยไม่มีใครแก้โค้ด"
    assert round(spread["avg_tokens"], 1) == 51.9


def test_build_noise_baseline_single_run_reports_zero_spread():
    """จุดข้อมูลเดียววัด noise ไม่ได้ — ต้องได้ spread 0.0 (= เทียบแบบเดิมทุกประการ)
    ไม่ใช่ปล่อยผ่านทุก metric เพราะข้อมูลไม่พอ"""
    baseline, spread = build_noise_baseline([_summary_with(avg_tokens=100.0, success_rate=0.9)])

    assert baseline["aggregate"]["avg_tokens"] == 100.0
    assert spread["avg_tokens"] == 0.0
    assert spread["success_rate"] == 0.0


def test_compare_marks_metric_non_gating_when_its_own_spread_exceeds_threshold():
    """spread ยังถูกวัดและรายงานเหมือนเดิม — แต่ตั้งแต่ W_gate_is_noisy เป็นต้นไป metric
    ประสิทธิภาพไม่ gate อีกแล้วไม่ว่าจะแกว่งแคบแค่ไหน (เดิม avg_llm_calls ที่แกว่งเอง 8%
    ถือว่า "ตัดสินได้" — แบนด์แคบแบบนั้นคือสิ่งที่ตีธง FAIL หลอกมาแล้วสามครั้งในวันเดียว)"""
    current = _summary_with(avg_tokens=130.0, avg_llm_calls=7.3)
    baseline = _summary_with(avg_tokens=100.0, avg_llm_calls=5.8)
    noise = {m: 0.0 for m in METRIC_NAMES}
    noise["avg_tokens"] = 52.0
    noise["avg_llm_calls"] = 8.0

    by_metric = {
        c.metric: c
        for c in compare_against_baseline(current, baseline, 10.0, noise_pct=noise)
    }

    assert by_metric["avg_tokens"].spread_pct == 52.0
    assert by_metric["avg_llm_calls"].spread_pct == 8.0
    # รายงานว่าแย่ลงได้ตามปกติ แค่ไม่มีสิทธิ์ตัดสิน commit
    assert by_metric["avg_llm_calls"].passed is False
    assert by_metric["avg_tokens"].gating is False
    assert by_metric["avg_llm_calls"].gating is False


def test_only_success_rate_gates_even_without_noise_data():
    """เดิมชื่อ ..._keeps_every_metric_gating และตรึงว่า "ทุก metric gate ได้" เมื่อไม่มี
    ข้อมูล noise — W_gate_is_noisy กลับข้อนั้นโดยเจตนา: ความถูกต้อง (success_rate) ตัดสิน
    commit ส่วน step/token/latency เป็นข้อมูลประกอบ ไม่ว่าจะรู้ spread ของมันหรือไม่"""
    current = _summary_with(avg_tokens=200.0)
    baseline = _summary_with(avg_tokens=100.0)

    comparisons = compare_against_baseline(current, baseline, 10.0)
    by_metric = {c.metric: c for c in comparisons}

    assert by_metric["success_rate"].gating is True
    assert not any(c.gating for c in comparisons if c.metric != "success_rate")
    assert all(c.spread_pct is None for c in comparisons)


def test_load_recent_summaries_returns_last_n_oldest_first(tmp_path):
    """เรียงตาม field timestamp ข้างในไฟล์ (ไม่ใช่ mtime) และตัดเหลือ limit ตัวหลังสุด"""
    import json as _json

    for ts in (5.0, 1.0, 3.0, 4.0, 2.0):
        (tmp_path / f"run_{ts}.json").write_text(
            _json.dumps({"timestamp": ts, "aggregate": {m: ts for m in METRIC_NAMES}}),
            encoding="utf-8",
        )

    recent = load_recent_summaries(str(tmp_path), limit=3)

    assert [r["timestamp"] for r in recent] == [3.0, 4.0, 5.0]


def test_load_recent_summaries_excludes_the_run_just_saved(tmp_path):
    """gate เพิ่ง save run ของตัวเองไปก่อนหน้า — ห้ามเอามาเป็น baseline ของตัวเอง"""
    import json as _json
    from pathlib import Path as _Path

    own = tmp_path / "own.json"
    own.write_text(_json.dumps({"timestamp": 9.0, "aggregate": {}}), encoding="utf-8")
    (tmp_path / "prev.json").write_text(
        _json.dumps({"timestamp": 1.0, "aggregate": {}}), encoding="utf-8")

    recent = load_recent_summaries(str(tmp_path), limit=5, exclude_path=_Path(own))

    assert [r["timestamp"] for r in recent] == [1.0]


# --- W_gate_is_noisy: benchmark รอบเดียวตัดสิน commit ไม่ได้ ---
#
# หลักฐานที่ปิดเรื่องนี้ (2026-09-09): commit 3ddb18a รันสองครั้งติดโดยไม่แตะโค้ดเลย ได้
# 12/15 (ธง FAIL, ล้ม login_checkout/add_candidate/choose-list) แล้ว 15/15 (ผ่าน ไม่ล้มสักตัว)
# ก่อนหน้านั้น ec556ba ก็ให้ 1.000 แล้ว 0.800 มาแล้ว


def _summary(success_rate, rows, *, timestamp=1.0):
    # แถวที่ไม่ระบุ steps ถือว่า "ได้ลงมือทำอะไรบ้าง" — ไม่งั้น run_is_invalid() จะอ่านว่า
    # เป็นรันที่ provider ล่ม (W_gate_run_invalid) แล้วถูกตัดออกจากทุกการคำนวณ
    rows = [{"steps": 1, **row} for row in rows]
    return {
        "git_commit": "abc1234",
        "model": "m",
        "timestamp": timestamp,
        "task_count": len(rows),
        "aggregate": {"success_rate": success_rate},
        "results": rows,
    }


def test_task_flakiness_splits_stable_from_coin_flip():
    summaries = [
        _summary(0.5, [{"name": "always", "success": True}, {"name": "never", "success": False},
                       {"name": "coin", "success": True}]),
        _summary(0.5, [{"name": "always", "success": True}, {"name": "never", "success": False},
                       {"name": "coin", "success": False}]),
    ]

    verdicts = {row["name"]: row["verdict"] for row in task_flakiness(summaries)}

    assert verdicts == {"always": "stable-pass", "never": "stable-fail", "coin": "flaky"}
    # เรียงจากผ่านน้อยไปมาก — ตัวที่ต้องดูอยู่บนสุดของตาราง
    assert [row["name"] for row in task_flakiness(summaries)][0] == "never"


def test_flaky_band_includes_both_edges():
    """เกณฑ์ที่ user กำหนดคือ 20-80% — ขอบทั้งสองข้างต้องนับเป็น flaky ไม่ใช่หลุดออกไป"""
    def runs_with(passes, total):
        return [
            _summary(1.0, [{"name": "t", "success": i < passes}]) for i in range(total)
        ]

    assert task_flakiness(runs_with(1, 5))[0]["verdict"] == "flaky"        # 20%
    assert task_flakiness(runs_with(4, 5))[0]["verdict"] == "flaky"        # 80%
    assert task_flakiness(runs_with(1, 10))[0]["verdict"] == "mostly-fail"  # 10%
    assert task_flakiness(runs_with(9, 10))[0]["verdict"] == "mostly-pass"  # 90%


def test_success_rate_stats_reports_the_spread_not_one_sample():
    stats = success_rate_stats([_summary(0.8, []), _summary(1.0, []), _summary(0.933, [])])

    assert stats["min"] == 0.8
    assert stats["max"] == 1.0
    assert stats["median"] == 0.933
    assert stats["runs"] == 3
    # ไม่มีรันเลยต้องไม่ throw — gate ครั้งแรกของ repo ใหม่เจอเคสนี้
    assert success_rate_stats([])["runs"] == 0


def test_compare_tasks_names_the_task_that_changed():
    baseline = _summary(1.0, [
        {"name": "steady", "success": True, "steps": 8, "llm_calls": 8,
         "action_calls": 7, "finish_task_calls": 1},
        {"name": "flipped", "success": True, "steps": 4, "llm_calls": 4,
         "action_calls": 3, "finish_task_calls": 1},
        {"name": "ballooned", "success": True, "steps": 4, "llm_calls": 4,
         "action_calls": 3, "finish_task_calls": 1},
    ])
    current = _summary(0.667, [
        {"name": "steady", "success": True, "steps": 9, "llm_calls": 8,
         "action_calls": 7, "finish_task_calls": 1},
        {"name": "flipped", "success": False, "steps": 4, "llm_calls": 4,
         "action_calls": 3, "finish_task_calls": 1},
        {"name": "ballooned", "success": True, "steps": 15, "llm_calls": 15,
         "action_calls": 14, "finish_task_calls": 1},
        {"name": "brand new", "success": True, "steps": 1},
    ])

    diffs = {row["name"]: row for row in compare_tasks(current, baseline)}

    assert set(diffs) == {"flipped", "ballooned"}      # steady ขยับ 1 step = ไม่ใช่เรื่อง
    assert diffs["flipped"]["success_after"] is False
    assert diffs["ballooned"]["counters"]["steps"] == (4, 15)
    assert "brand new" not in diffs                    # ไม่มีใน baseline = ไม่มีอะไรให้เทียบ


def test_only_correctness_can_fail_the_gate():
    """metric ประสิทธิภาพต้องรายงานได้แต่ไม่ตัดสิน — task ที่สำเร็จใน 9 step กับ 22 step
    คือ task ที่สำเร็จเหมือนกัน (แบนด์ที่แคบเพราะ 5 รอบล่าสุดบังเอิญนิ่ง เคยตีธง FAIL ให้
    approval_rate มาแล้วสองครั้งในวันเดียว)"""
    current = {"aggregate": {"success_rate": 1.0, "avg_tokens": 200.0, "approval_rate": 2.0}}
    baseline = {"aggregate": {"success_rate": 1.0, "avg_tokens": 100.0, "approval_rate": 1.0}}

    by_metric = {c.metric: c for c in compare_against_baseline(current, baseline, 10.0)}

    assert by_metric["avg_tokens"].passed is False       # ยังรายงานว่าแย่ลง
    assert by_metric["avg_tokens"].gating is False       # แต่ไม่ถ่วง gate
    assert by_metric["approval_rate"].gating is False
    assert by_metric["success_rate"].gating is True
    assert all(c.gating is False for c in by_metric.values() if c.metric != "success_rate")


@pytest.mark.asyncio
async def test_repeated_gate_decides_on_the_median_not_the_worst_run():
    """รอบแรกตก 0.8 แต่อีกสองรอบได้ 1.0 -> median 1.0 = ผ่าน ซึ่งคือเคสจริงของ 3ddb18a"""
    outcomes = [
        {"summary": _summary(0.8, [{"name": "t", "success": False}]),
         "baseline": _summary(1.0, [{"name": "t", "success": True}]), "passed": False},
        {"summary": _summary(1.0, [{"name": "t", "success": True}]),
         "baseline": _summary(0.5, []), "passed": True},
        {"summary": _summary(1.0, [{"name": "t", "success": True}]),
         "baseline": _summary(0.5, []), "passed": True},
    ]

    with patch("backend.app.core.release_gate.run_release_gate",
               AsyncMock(side_effect=outcomes)) as mock_gate:
        result = await run_release_gate_repeated(repeats=3, max_regression_pct=10.0)

    assert mock_gate.await_count == 3
    assert result["success_rate"] == {"min": 0.8, "median": 1.0, "max": 1.0, "runs": 3}
    assert result["passed"] is True
    # baseline ต้องมาจากรอบแรกเสมอ ไม่งั้นรอบ 2-3 จะไปเทียบกับรอบ 1 ของตัวเอง
    assert result["baseline_success_rate"] == 1.0
    # 2 ใน 3 = 66.7% อยู่ในช่วง 20-80% -> flaky: median บอกว่า commit ใช้ได้ แต่ task ตัวนี้
    # ยังเชื่อผลเดี่ยวๆ ไม่ได้ ซึ่งเป็นสองคำถามคนละข้อและต้องตอบแยกกัน
    assert {row["name"]: row["verdict"] for row in result["flakiness"]} == {"t": "flaky"}


@pytest.mark.asyncio
async def test_repeated_gate_still_fails_a_real_drop():
    # แถวต้องมีจริงและมี step — summary ที่ไม่มีแถวเลยคือรันที่วัดอะไรไม่ได้
    # (W_gate_run_invalid) ซึ่งเป็นคนละเรื่องกับ commit ที่แย่ลงจริง
    half = [{"name": "a", "success": True, "steps": 3}, {"name": "b", "success": False, "steps": 5}]
    outcomes = [
        {"summary": _summary(0.5, half), "baseline": _summary(1.0, half), "passed": False},
        {"summary": _summary(0.5, half), "baseline": None, "passed": True},
        {"summary": _summary(0.6, half), "baseline": None, "passed": True},
    ]

    with patch("backend.app.core.release_gate.run_release_gate",
               AsyncMock(side_effect=outcomes)):
        result = await run_release_gate_repeated(repeats=3, max_regression_pct=10.0)

    assert result["success_rate"]["median"] == 0.5
    assert result["passed"] is False


def test_flakiness_reports_never_become_a_baseline(tmp_path):
    """รายงาน flakiness อยู่โฟลเดอร์เดียวกับ summary — ถ้าหลุดเข้าไปเป็น baseline จะกลายเป็น
    การเทียบกับศูนย์ทุก metric เงียบๆ"""
    save_summary(_summary(1.0, [{"name": "t", "success": True}]), str(tmp_path))
    save_flakiness_report({"git_commit": "abc1234", "runs": 5, "tasks": []}, str(tmp_path))

    loaded = load_recent_summaries(str(tmp_path), limit=5)

    assert len(loaded) == 1
    assert "aggregate" in loaded[0]


# --- เมนูของ run.py: key ซ้ำจะเงียบสนิท ---


def test_run_py_menu_numbers_are_unique():
    """W_gate_is_noisy (เจอตอนเพิ่มคำสั่ง flakiness 2026-09-09): ใส่ "23" ทับของ kpi ที่มีอยู่
    แล้ว — dict literal ที่มี key ซ้ำไม่ error อะไรเลย ตัวหลังชนะเงียบๆ ผลคือ `run.py
    flakiness 5` ไปเรียก KPI แล้วตีความ 5 เป็น "5 วัน" เผาการรันไปหนึ่งครั้งกว่าจะรู้ตัว

    อ่านจากซอร์สจริง ไม่ใช่จาก dict ที่โหลดแล้ว — dict ที่โหลดแล้วคือ *ผลลัพธ์หลังจากที่
    ตัวซ้ำถูกทิ้งไปแล้ว* จึงไม่มีทางเห็นปัญหา (เหตุผลเดียวกับเทสต์ทะเบียน marker ของ W108)"""
    import re
    from pathlib import Path

    source = Path(__file__).resolve().parents[2] / "run.py"
    text = source.read_text(encoding="utf-8")

    action_keys = re.findall(r'^    "(\d+)": \(', text, re.MULTILINE)
    alias_lines = re.findall(r'^    "([a-z0-9\-+_]+)": "(\d+)",', text, re.MULTILINE)

    duplicates = {k for k in action_keys if action_keys.count(k) > 1}
    assert not duplicates, f"เลขเมนูซ้ำใน run.py: {sorted(duplicates)}"

    alias_names = [name for name, _ in alias_lines]
    dup_aliases = {n for n in alias_names if alias_names.count(n) > 1}
    assert not dup_aliases, f"alias ซ้ำใน run.py: {sorted(dup_aliases)}"

    # ทุก alias ต้องชี้ไปยังเลขที่มีอยู่จริง
    unknown = {number for _, number in alias_lines} - set(action_keys)
    assert not unknown, f"alias ชี้ไปยังเลขที่ไม่มีในเมนู: {sorted(unknown)}"


# --- W_gate_run_invalid: provider ล่ม ไม่ใช่ regression ---
#
# 2026-09-09: endpoint ของ ChatGPT OAuth ปฏิเสธทุกโมเดลกลางวัน ("The 'gpt-5.4-mini' model is
# not supported when using Codex with a ChatGPT account") — flakiness 5 รอบได้ 0/5 ทุก task
# และ gate อ่านออกมาเป็น success_rate 0.000 ซึ่งจะไป fail commit ให้กับ provider ที่ล่ม


def _rows(*steps):
    return [{"name": f"t{i}", "success": False, "steps": s} for i, s in enumerate(steps)]


def test_a_run_where_nothing_moved_is_not_a_measurement():
    assert run_is_invalid(_summary(0.0, _rows(0, 0, 0))) is True
    assert run_is_invalid(_summary(0.0, [])) is True
    # task เดี่ยวที่ได้ 0 step เกิดได้ปกติ (short-circuit ของ api/routes.py) — ไม่ใช่ outage
    assert run_is_invalid(_summary(0.5, _rows(0, 4, 9))) is False


@pytest.mark.asyncio
async def test_provider_outage_does_not_fail_the_commit():
    dead = {"summary": _summary(0.0, _rows(0, 0, 0)), "baseline": _summary(1.0, _rows(5, 5, 5)),
            "passed": False}

    with patch("backend.app.core.release_gate.run_release_gate",
               AsyncMock(side_effect=[dead, dead, dead])):
        result = await run_release_gate_repeated(repeats=3, max_regression_pct=10.0)

    assert result["invalid_runs"] == 3
    assert result["measured"] is False
    assert result["passed"] is True            # ตอบไม่ได้ ไม่ใช่ตอบว่าแย่ลง
    assert result["flakiness"] == []           # ไม่มีใครถูกตราหน้าว่า stable-fail จาก outage


@pytest.mark.asyncio
async def test_one_dead_run_is_dropped_and_the_rest_still_decide():
    good = {"summary": _summary(1.0, _rows(5, 5, 5)), "baseline": _summary(1.0, _rows(5, 5, 5)),
            "passed": True}
    dead = {"summary": _summary(0.0, _rows(0, 0, 0)), "baseline": None, "passed": False}

    with patch("backend.app.core.release_gate.run_release_gate",
               AsyncMock(side_effect=[good, dead, good])):
        result = await run_release_gate_repeated(repeats=3, max_regression_pct=10.0)

    assert result["invalid_runs"] == 1
    assert result["measured"] is True
    assert result["success_rate"]["runs"] == 2      # นับเฉพาะรอบที่วัดได้จริง
    assert result["success_rate"]["median"] == 1.0
    assert result["passed"] is True

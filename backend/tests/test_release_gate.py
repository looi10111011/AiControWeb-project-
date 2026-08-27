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
    """avg_tokens ที่แกว่งเอง 52% แล้ว current ต่างจาก baseline 30% — ตัดสินไม่ได้
    ส่วน avg_llm_calls ที่แกว่งเองแค่ 8% แล้ว regress 26% ตัดสินได้และต้อง fail"""
    current = _summary_with(avg_tokens=130.0, avg_llm_calls=7.3)
    baseline = _summary_with(avg_tokens=100.0, avg_llm_calls=5.8)
    noise = {m: 0.0 for m in METRIC_NAMES}
    noise["avg_tokens"] = 52.0
    noise["avg_llm_calls"] = 8.0

    by_metric = {
        c.metric: c
        for c in compare_against_baseline(current, baseline, 10.0, noise_pct=noise)
    }

    assert by_metric["avg_tokens"].gating is False
    assert by_metric["avg_tokens"].spread_pct == 52.0
    assert by_metric["avg_llm_calls"].gating is True
    assert by_metric["avg_llm_calls"].passed is False


def test_compare_without_noise_pct_keeps_every_metric_gating():
    """ผู้เรียกเดิมที่ไม่ส่ง noise_pct (เช่น baseline_path ที่ปักหมุดเอง) ต้องได้พฤติกรรม
    เหมือนเดิมเป๊ะ — ทุก metric ยัง gate ได้ตามปกติ"""
    current = _summary_with(avg_tokens=200.0)
    baseline = _summary_with(avg_tokens=100.0)

    comparisons = compare_against_baseline(current, baseline, 10.0)

    assert all(c.gating for c in comparisons)
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

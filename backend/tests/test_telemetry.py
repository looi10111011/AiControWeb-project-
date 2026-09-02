"""W_eval_trace: telemetry writer ตัวกลาง + การต่อสายเข้าเส้นทาง eval

บั๊กจริงที่เทสต์ชุดนี้กันไม่ให้กลับมา: release gate รัน 15 task แล้วรายงานได้แค่ "14/15 ผ่าน"
โดย step_trace.jsonl มี 0 บรรทัด เพราะ evaluation.py/miniwob_eval.py เรียก run_task() ตรงๆ
ไม่ผ่าน TaskManager ซึ่งเป็นที่เดียวที่เรียก writer อยู่
"""
import json
from unittest.mock import AsyncMock, patch

import pytest

from backend.app.core.evaluation import run_evaluation
from backend.app.core.telemetry import (
    SOURCE_API,
    SOURCE_EVAL,
    new_run_id,
    write_step_trace,
    write_token_usage,
)

_FAKE_TASKS = [
    {"name": "task_a", "goal": "goal A", "max_steps": 10},
    {"name": "task_b", "goal": "goal B", "max_steps": 20},
]

_FAKE_HISTORY = [
    {"step": 0, "cmd": {"type": "goto", "url": "https://x/"}, "result": "[OK] goto", "success": True},
    {
        "step": 1, "cmd": {"type": "click", "index": 3}, "label": "Admin",
        "result": "[OK] click(3)", "success": True, "failure_class": "ok",
        "timing": {"snapshot": 0.1, "llm": 2.0, "action": 0.5},
        "tokens": {"input": 10, "output": 5, "cache_read": 0, "cache_creation": 0},
    },
]


def _fake_result(history=None, success=True, steps=2):
    return {
        "success": success, "steps": steps, "message": "done",
        "tokens": {"input": 10, "output": 5, "cache_read": 0, "cache_creation": 0},
        "history": history if history is not None else _FAKE_HISTORY,
    }


def _read_lines(path):
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


@pytest.fixture
def logs(tmp_path):
    """ชี้ทั้งสอง log path ไปที่ tmp_path — ห้ามให้เทสต์เขียนลงไฟล์จริงของโปรเจกต์"""
    trace = tmp_path / "step_trace.jsonl"
    tokens = tmp_path / "token_usage.jsonl"
    with patch("backend.app.core.telemetry.settings") as mock_settings:
        mock_settings.step_trace_log_path = str(trace)
        mock_settings.token_usage_log_path = str(tokens)
        yield trace, tokens


# ---------------- writer ตัวกลาง ----------------

def test_write_step_trace_writes_one_line_per_step_with_run_id(logs):
    trace, _ = logs
    write_step_trace(_FAKE_HISTORY, task_id="t1", provider="openai", run_id="gate-123")

    lines = _read_lines(trace)
    assert len(lines) == 2
    assert [l["step"] for l in lines] == [0, 1]
    assert all(l["run_id"] == "gate-123" for l in lines)
    assert all(l["task_id"] == "t1" for l in lines)
    assert lines[1]["action_type"] == "click"
    assert lines[1]["index"] == 3
    assert lines[1]["label"] == "Admin"
    assert lines[1]["failure_class"] == "ok"
    assert lines[1]["timing"]["llm"] == 2.0


def test_write_step_trace_omits_run_id_when_not_given(logs):
    """เส้นทาง API เดิมไม่มี run_id — ต้องไม่มี field นี้โผล่มาเปล่าๆ ให้ไฟล์รกและ parser สับสน"""
    trace, _ = logs
    write_step_trace(_FAKE_HISTORY, task_id="t1", provider="openai")

    assert all("run_id" not in l for l in _read_lines(trace))


def test_write_step_trace_writes_nothing_when_history_empty(logs):
    trace, _ = logs
    write_step_trace([], task_id="t1", provider="openai")
    write_step_trace(None, task_id="t1", provider="openai")

    assert _read_lines(trace) == []


def test_write_step_trace_truncates_long_result_text(logs):
    """ตารางเต็มๆ จาก read_page_data ยาวเป็นพันตัวอักษร ไม่มีประโยชน์ในไฟล์วิเคราะห์"""
    trace, _ = logs
    write_step_trace(
        [{"step": 1, "cmd": {"type": "read_page_data"}, "result": "x" * 5000, "success": True}],
        task_id="t1", provider="openai",
    )

    assert len(_read_lines(trace)[0]["result"]) == 300


def test_write_step_trace_never_raises_on_a_bad_history_entry(logs):
    """ห้าม throw เด็ดขาด — ปัญหาการเก็บ log ต้องไม่ทำให้ task ที่เสร็จแล้วกลายเป็น error"""
    trace, _ = logs
    write_step_trace(["not a dict", 42, _FAKE_HISTORY[0]], task_id="t1", provider="openai")

    assert len(_read_lines(trace)) == 1


def test_write_token_usage_records_source_and_run_id(logs):
    _, tokens = logs
    write_token_usage(
        task_id="t1", url="https://x/", goal="g", provider="openai",
        result=_fake_result(), status="done", error=None, duration_seconds=1.234,
        source=SOURCE_EVAL, run_id="gate-123",
    )

    row = _read_lines(tokens)[0]
    assert row["source"] == SOURCE_EVAL
    assert row["run_id"] == "gate-123"
    assert row["status"] == "done"
    assert row["steps"] == 2
    assert row["duration_seconds"] == 1.23


def test_write_token_usage_records_w1_call_breakdown(logs):
    """W_token_cut W1: ตัวนับทั้งชุดต้องลงไฟล์ตามที่ run_task() คืนมา และ result ที่ไม่มี
    field พวกนี้ (task รุ่นเก่า / เส้นทาง exception) ต้อง default เป็น 0/{} ไม่ใช่ KeyError"""
    _, tokens = logs
    result = _fake_result()
    result.update({
        "llm_calls": 9, "action_calls": 3, "finish_task_calls": 2,
        "guard_rejections": {"filter_scope": 2, "premature_true_finish": 1},
        "guard_reason_counts": {"filter_scope": 2, "premature_true_finish": 1},
        "repeated_guard_count": 1, "finish_loop_prevented": 1,
        "notool_retries": 1, "cache_hit_turns": 4, "cache_miss_turns": 5,
        "avg_input_tokens_per_call": 6100.0, "avg_output_tokens_per_call": 40.0,
    })
    write_token_usage(
        task_id="t1", url="https://x/", goal="g", provider="openai",
        result=result, status="done", error=None, duration_seconds=1.0,
    )
    row = _read_lines(tokens)[0]
    assert row["llm_calls"] == 9
    assert row["action_calls"] == 3
    assert row["finish_task_calls"] == 2
    assert row["guard_rejections"] == {"filter_scope": 2, "premature_true_finish": 1}
    assert row["notool_retries"] == 1
    assert row["cache_hit_turns"] == 4 and row["cache_miss_turns"] == 5
    assert row["avg_input_tokens_per_call"] == 6100.0
    assert row["repeated_guard_count"] == 1
    assert row["finish_loop_prevented"] == 1
    assert row["guard_reason_counts"] == {"filter_scope": 2, "premature_true_finish": 1}

    # result ที่ไม่มี field W1/W3 เลย
    write_token_usage(
        task_id="t2", url="https://x/", goal="g", provider="openai",
        result=_fake_result(), status="done", error=None, duration_seconds=1.0,
    )
    old = _read_lines(tokens)[1]
    assert old["llm_calls"] == 0
    assert old["guard_rejections"] == {}
    assert old["action_calls"] == 0
    assert old["repeated_guard_count"] == 0
    assert old["finish_loop_prevented"] == 0


def test_write_token_usage_defaults_to_api_source_and_zero_tokens_without_result(logs):
    """เส้นทาง cancelled/exception ที่ run_task() ไม่ได้คืน dict — ยังต้องมีแถวไว้ ไม่งั้น
    success rate ที่คำนวณจากไฟล์นี้เป็นเพดานบน ไม่ใช่ค่าจริง"""
    _, tokens = logs
    write_token_usage(
        task_id="t1", url="https://x/", goal="g", provider="openai",
        result=None, status="cancelled", error="หยุดโดยผู้ใช้ (Stop)", duration_seconds=5.0,
    )

    row = _read_lines(tokens)[0]
    assert row["source"] == SOURCE_API
    assert "run_id" not in row
    assert row["success"] is False
    assert row["steps"] == 0
    assert row["tokens"] == {"input": 0, "output": 0, "cache_read": 0, "cache_creation": 0}
    assert row["error"] == "หยุดโดยผู้ใช้ (Stop)"


def test_new_run_id_is_prefixed_and_unique_enough():
    a, b = new_run_id("gate"), new_run_id("eval")
    assert a.startswith("gate-") and b.startswith("eval-")
    assert a != b


# ---------------- การต่อสายเข้า run_evaluation ----------------

@pytest.mark.asyncio
async def test_run_evaluation_writes_trace_and_token_rows_for_each_task(logs):
    trace, tokens = logs
    with patch("backend.app.core.evaluation.Orchestrator") as MockOrchestrator:
        MockOrchestrator.return_value.run_task = AsyncMock(return_value=_fake_result())
        await run_evaluation(tasks=_FAKE_TASKS, provider="openai", run_id="gate-999")

    trace_lines = _read_lines(trace)
    assert len(trace_lines) == 4  # 2 task x 2 step
    assert {l["task_id"] for l in trace_lines} == {"gate-999-task_a", "gate-999-task_b"}
    assert all(l["run_id"] == "gate-999" for l in trace_lines)

    token_rows = _read_lines(tokens)
    assert len(token_rows) == 2
    assert all(r["source"] == SOURCE_EVAL for r in token_rows)
    assert all(r["run_id"] == "gate-999" for r in token_rows)
    assert {r["goal"] for r in token_rows} == {"goal A", "goal B"}


@pytest.mark.asyncio
async def test_run_evaluation_generates_its_own_run_id_when_called_standalone(logs):
    """รัน `run.py eval` เดี่ยวๆ ก็ยังต้อง group trace ได้ ไม่ใช่มีเฉพาะตอนผ่าน release gate"""
    trace, _ = logs
    with patch("backend.app.core.evaluation.Orchestrator") as MockOrchestrator:
        MockOrchestrator.return_value.run_task = AsyncMock(return_value=_fake_result())
        await run_evaluation(tasks=_FAKE_TASKS, provider="openai")

    run_ids = {l["run_id"] for l in _read_lines(trace)}
    assert len(run_ids) == 1
    assert run_ids.pop().startswith("eval-")


@pytest.mark.asyncio
async def test_run_evaluation_logs_a_token_row_but_no_trace_when_run_task_raises(logs):
    """run_task() ที่ throw จริง (browser launch พัง) ไม่มี history ให้เขียน — แต่ task ที่พัง
    ต้องไม่หายไปจาก token_usage.jsonl"""
    trace, tokens = logs
    with patch("backend.app.core.evaluation.Orchestrator") as MockOrchestrator:
        MockOrchestrator.return_value.run_task = AsyncMock(side_effect=RuntimeError("browser ตาย"))
        await run_evaluation(tasks=_FAKE_TASKS, provider="openai", run_id="gate-err")

    assert _read_lines(trace) == []
    rows = _read_lines(tokens)
    assert len(rows) == 2
    assert all(r["status"] == "error" for r in rows)
    assert all("RuntimeError: browser ตาย" in r["error"] for r in rows)


@pytest.mark.asyncio
async def test_run_evaluation_still_returns_results_when_logging_path_is_unwritable(logs, tmp_path):
    """ปัญหาการเขียน log ต้องไม่ทำให้ผล eval ที่วัดได้จริงหายไป"""
    with patch("backend.app.core.telemetry.settings") as mock_settings:
        # ชี้ไปที่ path ที่สร้างไม่ได้ (ไฟล์ถูกใช้เป็น directory)
        blocker = tmp_path / "blocker"
        blocker.write_text("not a dir", encoding="utf-8")
        mock_settings.step_trace_log_path = str(blocker / "trace.jsonl")
        mock_settings.token_usage_log_path = str(blocker / "tokens.jsonl")
        with patch("backend.app.core.evaluation.Orchestrator") as MockOrchestrator:
            MockOrchestrator.return_value.run_task = AsyncMock(return_value=_fake_result())
            report = await run_evaluation(tasks=_FAKE_TASKS, provider="openai")

    assert len(report.results) == 2
    assert report.success_rate == 1.0

from unittest.mock import AsyncMock, patch

import pytest

from backend.app.core.evaluation import (
    BENCHMARK_TASKS,
    EvaluationReport,
    TaskEvalResult,
    run_evaluation,
)

# ทุกเทสต์ mock Orchestrator ทั้งคลาส (เหมือน test_api.py) — ไม่เปิด browser จริง ไม่ยิง
# LLM API จริง ไม่ต้องรอเป็นนาทีต่อ task เหมือนรันจริงผ่าน `python run.py eval`

_FAKE_TASKS = [
    {"name": "task_a", "goal": "goal A", "max_steps": 10},
    {"name": "task_b", "goal": "goal B", "max_steps": 20},
]


def _fake_result(success=True, steps=3, input_t=10, output_t=5, cache_read=0, cache_creation=0, message="เสร็จ"):
    return {
        "success": success, "steps": steps, "message": message,
        "tokens": {"input": input_t, "output": output_t, "cache_read": cache_read, "cache_creation": cache_creation},
    }


@pytest.mark.asyncio
async def test_run_evaluation_calls_run_task_once_per_task_with_correct_args():
    with patch("backend.app.core.evaluation.Orchestrator") as MockOrchestrator:
        mock_run_task = AsyncMock(return_value=_fake_result())
        MockOrchestrator.return_value.run_task = mock_run_task

        await run_evaluation(tasks=_FAKE_TASKS, provider="anthropic", url="https://example.com")

    assert mock_run_task.await_count == 2
    first_args, first_kwargs = mock_run_task.await_args_list[0]
    assert first_args == ("https://example.com", "goal A")
    assert first_kwargs == {
        "max_steps": 10, "headless": True, "auto_approve": True,
        "confirm_plan": False, "provider": "anthropic",
    }
    second_args, _ = mock_run_task.await_args_list[1]
    assert second_args == ("https://example.com", "goal B")


@pytest.mark.asyncio
async def test_run_evaluation_uses_default_max_steps_when_task_omits_it():
    with patch("backend.app.core.evaluation.Orchestrator") as MockOrchestrator:
        mock_run_task = AsyncMock(return_value=_fake_result())
        MockOrchestrator.return_value.run_task = mock_run_task

        await run_evaluation(tasks=[{"name": "x", "goal": "g"}])

    assert mock_run_task.await_args.kwargs["max_steps"] == 20


@pytest.mark.asyncio
async def test_run_evaluation_computes_total_tokens_as_sum_of_all_four_fields():
    with patch("backend.app.core.evaluation.Orchestrator") as MockOrchestrator:
        mock_run_task = AsyncMock(
            return_value=_fake_result(input_t=100, output_t=50, cache_read=20, cache_creation=5),
        )
        MockOrchestrator.return_value.run_task = mock_run_task

        report = await run_evaluation(tasks=[{"name": "x", "goal": "g"}])

    assert report.results[0].total_tokens == 175


@pytest.mark.asyncio
async def test_run_evaluation_records_success_steps_and_message_per_task():
    with patch("backend.app.core.evaluation.Orchestrator") as MockOrchestrator:
        mock_run_task = AsyncMock(return_value=_fake_result(success=False, steps=7, message="ไม่สำเร็จ"))
        MockOrchestrator.return_value.run_task = mock_run_task

        report = await run_evaluation(tasks=[{"name": "x", "goal": "g"}])

    result = report.results[0]
    assert result.name == "x"
    assert result.goal == "g"
    assert result.success is False
    assert result.steps == 7
    assert result.message == "ไม่สำเร็จ"
    assert result.error is None


@pytest.mark.asyncio
async def test_run_evaluation_catches_exception_from_one_task_and_continues_with_the_rest():
    with patch("backend.app.core.evaluation.Orchestrator") as MockOrchestrator:
        mock_run_task = AsyncMock(side_effect=[RuntimeError("browser launch failed"), _fake_result()])
        MockOrchestrator.return_value.run_task = mock_run_task

        report = await run_evaluation(tasks=_FAKE_TASKS)

    assert mock_run_task.await_count == 2  # task_b ยังถูกรันต่อแม้ task_a throw
    failed, succeeded = report.results
    assert failed.success is False
    assert failed.steps == 0
    assert failed.total_tokens == 0
    assert failed.error == "RuntimeError: browser launch failed"
    assert succeeded.success is True
    assert succeeded.error is None


@pytest.mark.asyncio
async def test_run_evaluation_defaults_to_benchmark_tasks_when_none_given():
    with patch("backend.app.core.evaluation.Orchestrator") as MockOrchestrator:
        mock_run_task = AsyncMock(return_value=_fake_result())
        MockOrchestrator.return_value.run_task = mock_run_task

        report = await run_evaluation()

    assert len(report.results) == len(BENCHMARK_TASKS)
    assert [r.name for r in report.results] == [t["name"] for t in BENCHMARK_TASKS]


# --- EvaluationReport aggregate properties ---


def test_evaluation_report_aggregates_success_rate_avg_steps_avg_tokens():
    report = EvaluationReport(results=[
        TaskEvalResult(name="a", goal="g", success=True, steps=4, total_tokens=100, message="ok"),
        TaskEvalResult(name="b", goal="g", success=False, steps=10, total_tokens=300, message="fail"),
    ])

    assert report.success_rate == 0.5
    assert report.avg_steps == 7.0
    assert report.avg_tokens == 200.0


def test_evaluation_report_empty_results_does_not_divide_by_zero():
    report = EvaluationReport()

    assert report.success_rate == 0.0
    assert report.avg_steps == 0.0
    assert report.avg_tokens == 0.0

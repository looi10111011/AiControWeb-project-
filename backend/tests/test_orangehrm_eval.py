from unittest.mock import AsyncMock, patch

import pytest

from backend.app.core.orangehrm_eval import (
    ORANGEHRM_TASKS,
    _ORANGEHRM_URL,
    _add_candidate_task,
    run_orangehrm_evaluation,
)

# run_orangehrm_evaluation() แค่ห่อ evaluation.py::run_evaluation() ด้วย url/tasks ของ
# OrangeHRM (ไม่ import Orchestrator เองเลยในไฟล์นี้) — mock ที่
# backend.app.core.evaluation.Orchestrator (จุดที่ run_evaluation() เรียกจริง) เหมือน
# test_evaluation.py ทุกประการ


def _fake_result(success=True, steps=3, input_t=10, output_t=5, message="เสร็จ"):
    return {
        "success": success, "steps": steps, "message": message,
        "tokens": {"input": input_t, "output": output_t, "cache_read": 0, "cache_creation": 0},
    }


@pytest.mark.asyncio
async def test_run_orangehrm_evaluation_runs_all_tasks_against_orangehrm_url():
    with patch("backend.app.core.evaluation.Orchestrator") as MockOrchestrator:
        mock_run_task = AsyncMock(return_value=_fake_result())
        MockOrchestrator.return_value.run_task = mock_run_task

        report = await run_orangehrm_evaluation(provider="anthropic")

    # ORANGEHRM_TASKS (constant) + add_candidate (goal สร้างใหม่ทุกครั้ง) = +1
    assert mock_run_task.await_count == len(ORANGEHRM_TASKS) + 1
    assert len(report.results) == len(ORANGEHRM_TASKS) + 1
    for call in mock_run_task.await_args_list:
        url_arg = call.args[0]
        assert url_arg == _ORANGEHRM_URL


@pytest.mark.asyncio
async def test_run_orangehrm_evaluation_includes_expected_task_names():
    with patch("backend.app.core.evaluation.Orchestrator") as MockOrchestrator:
        MockOrchestrator.return_value.run_task = AsyncMock(return_value=_fake_result())

        report = await run_orangehrm_evaluation()

    names = {r.name for r in report.results}
    assert names == {"login_dashboard", "search_no_results", "add_candidate"}


@pytest.mark.asyncio
async def test_run_orangehrm_evaluation_passes_provider_through():
    with patch("backend.app.core.evaluation.Orchestrator") as MockOrchestrator:
        mock_run_task = AsyncMock(return_value=_fake_result())
        MockOrchestrator.return_value.run_task = mock_run_task

        await run_orangehrm_evaluation(provider="groq")

    assert mock_run_task.await_args.kwargs["provider"] == "groq"


def test_add_candidate_task_embeds_a_fresh_tag_each_call():
    """goal ต้องไม่ใช่ constant string ซ้ำทุกครั้ง (ต่างจาก task อื่นในไฟล์นี้) — กันชนกับ
    candidate ที่ tester คนอื่น/รอบก่อนหน้าสร้างไว้แล้วบน shared demo instance เดียวกัน"""
    with patch("backend.app.core.orangehrm_eval.time") as mock_time:
        mock_time.time.side_effect = [1000, 2000]
        task_a = _add_candidate_task()
        task_b = _add_candidate_task()

    assert task_a["goal"] != task_b["goal"]
    assert "1000" in task_a["goal"]
    assert "2000" in task_b["goal"]
    assert task_a["name"] == "add_candidate"


def test_add_candidate_task_is_not_in_the_constant_task_list():
    """ORANGEHRM_TASKS ต้องมีแต่ task ที่ goal คงที่ — add_candidate ต้องถูกสร้างแยกทุก
    ครั้งผ่าน run_orangehrm_evaluation() เท่านั้น ไม่ใช่ค่าคงที่ที่ประกาศไว้ล่วงหน้า"""
    assert "add_candidate" not in {t["name"] for t in ORANGEHRM_TASKS}


def test_orangehrm_url_points_to_the_login_page():
    assert _ORANGEHRM_URL == "https://opensource-demo.orangehrmlive.com/web/index.php/auth/login"

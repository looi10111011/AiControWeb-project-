from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.app.core import miniwob_eval
from backend.app.core.miniwob_eval import (
    DEFAULT_TASKS,
    MiniWobReport,
    MiniWobResult,
    _auto_approve,
    _init_script,
    run_miniwob_evaluation,
)

# เหมือน test_evaluation.py ทุกประการ: mock Orchestrator ทั้งคลาส ไม่เปิด browser จริง ไม่ยิง
# LLM API จริง — ต่างเพิ่มอีกชั้นตรงที่ _run_one_task() เปิด Playwright เองด้วย (ไม่ได้ยืม
# browser จาก Orchestrator) เลย mock async_playwright() ด้วยเสมอในทุกเทสต์ที่เรียกผ่าน
# run_miniwob_evaluation()/_run_one_task() จริง


def _fake_result(success=True, steps=3, input_t=10, output_t=5, cache_read=0, cache_creation=0, message="เสร็จ"):
    return {
        "success": success, "steps": steps, "message": message,
        "tokens": {"input": input_t, "output": output_t, "cache_read": cache_read, "cache_creation": cache_creation},
    }


def _make_fake_playwright(utterance="Click the button.", state=None):
    """คืน MagicMock ที่ปลอม async_playwright() ทั้ง context manager ให้ลึกพอสำหรับ
    _run_one_task() — page.evaluate() ถูกเรียก 2 ครั้งตามลำดับจริง (อ่าน utterance ก่อน
    เรียก run_task(), อ่าน WOB_DONE_GLOBAL/WOB_REWARD_GLOBAL หลัง run_task() จบ) ใช้
    side_effect เรียงตามลำดับนั้น"""
    state = state if state is not None else {"done": True, "reward": 1.0}
    page = MagicMock()
    page.add_init_script = AsyncMock()
    page.goto = AsyncMock()
    page.wait_for_load_state = AsyncMock()
    page.evaluate = AsyncMock(side_effect=[utterance, state])
    context = MagicMock()
    context.new_page = AsyncMock(return_value=page)
    context.close = AsyncMock()
    browser = MagicMock()
    browser.new_context = AsyncMock(return_value=context)
    browser.close = AsyncMock()
    chromium = MagicMock()
    chromium.launch = AsyncMock(return_value=browser)
    playwright_driver = MagicMock()
    playwright_driver.chromium = chromium

    fake_async_playwright_cm = MagicMock()
    fake_async_playwright_cm.__aenter__ = AsyncMock(return_value=playwright_driver)
    fake_async_playwright_cm.__aexit__ = AsyncMock(return_value=False)

    def _async_playwright_factory():
        return fake_async_playwright_cm

    return _async_playwright_factory, page


@pytest.mark.asyncio
async def test_run_one_task_reads_utterance_and_passes_it_as_goal():
    fake_factory, page = _make_fake_playwright(utterance='Click on the "OK" button.')
    with patch("backend.app.core.miniwob_eval.async_playwright", fake_factory), \
         patch("backend.app.core.miniwob_eval.Orchestrator") as MockOrchestrator, \
         patch("backend.app.core.miniwob_eval.MINIWOB_TASK_DIR") as mock_dir:
        mock_dir.__truediv__.return_value.exists.return_value = True
        mock_dir.__truediv__.return_value.resolve.return_value.as_uri.return_value = "file:///fake/click-button.html"
        mock_run_task = AsyncMock(return_value=_fake_result())
        MockOrchestrator.return_value.run_task = mock_run_task

        result = await miniwob_eval._run_one_task("click-button", max_steps=15, provider=None, headless=True)

    assert result.utterance == 'Click on the "OK" button.'
    args, kwargs = mock_run_task.await_args
    assert args[0] == "file:///fake/click-button.html"
    assert 'Click on the "OK" button.' in args[1]
    assert "finish_task" in args[1]
    assert kwargs["page"] is page
    assert kwargs["confirm_plan"] is False
    assert kwargs["max_steps"] == 15
    assert callable(kwargs["ask_user_func"])


@pytest.mark.asyncio
async def test_run_one_task_success_requires_both_done_and_positive_reward():
    fake_factory, _page = _make_fake_playwright(state={"done": True, "reward": 0.87})
    with patch("backend.app.core.miniwob_eval.async_playwright", fake_factory), \
         patch("backend.app.core.miniwob_eval.Orchestrator") as MockOrchestrator, \
         patch("backend.app.core.miniwob_eval.MINIWOB_TASK_DIR") as mock_dir:
        mock_dir.__truediv__.return_value.exists.return_value = True
        mock_dir.__truediv__.return_value.resolve.return_value.as_uri.return_value = "file:///fake/x.html"
        MockOrchestrator.return_value.run_task = AsyncMock(return_value=_fake_result())

        result = await miniwob_eval._run_one_task("x", max_steps=15, provider=None, headless=True)

    assert result.success is True
    assert result.reward == 0.87


@pytest.mark.asyncio
async def test_run_one_task_not_success_when_done_but_reward_not_positive():
    """คลิกผิดปุ่ม -> core.endEpisode(-1) -> done=True แต่ reward ติดลบ ไม่ใช่ความสำเร็จ"""
    fake_factory, _page = _make_fake_playwright(state={"done": True, "reward": -1.0})
    with patch("backend.app.core.miniwob_eval.async_playwright", fake_factory), \
         patch("backend.app.core.miniwob_eval.Orchestrator") as MockOrchestrator, \
         patch("backend.app.core.miniwob_eval.MINIWOB_TASK_DIR") as mock_dir:
        mock_dir.__truediv__.return_value.exists.return_value = True
        mock_dir.__truediv__.return_value.resolve.return_value.as_uri.return_value = "file:///fake/x.html"
        MockOrchestrator.return_value.run_task = AsyncMock(return_value=_fake_result())

        result = await miniwob_eval._run_one_task("x", max_steps=15, provider=None, headless=True)

    assert result.success is False
    assert result.reward == -1.0


@pytest.mark.asyncio
async def test_run_one_task_not_success_when_never_done():
    """max_steps หมดก่อนคลิกอะไรเลย -> WOB_DONE_GLOBAL ยังเป็น false อยู่ (ไม่ใช่ความสำเร็จ
    ไม่ว่า WOB_REWARD_GLOBAL จะเป็นค่าอะไรก็ตาม)"""
    fake_factory, _page = _make_fake_playwright(state={"done": False, "reward": 0})
    with patch("backend.app.core.miniwob_eval.async_playwright", fake_factory), \
         patch("backend.app.core.miniwob_eval.Orchestrator") as MockOrchestrator, \
         patch("backend.app.core.miniwob_eval.MINIWOB_TASK_DIR") as mock_dir:
        mock_dir.__truediv__.return_value.exists.return_value = True
        mock_dir.__truediv__.return_value.resolve.return_value.as_uri.return_value = "file:///fake/x.html"
        MockOrchestrator.return_value.run_task = AsyncMock(return_value=_fake_result())

        result = await miniwob_eval._run_one_task("x", max_steps=15, provider=None, headless=True)

    assert result.success is False


@pytest.mark.asyncio
async def test_run_one_task_raises_filenotfound_for_missing_task_file():
    with patch("backend.app.core.miniwob_eval.MINIWOB_TASK_DIR") as mock_dir:
        mock_dir.__truediv__.return_value.exists.return_value = False

        with pytest.raises(FileNotFoundError):
            await miniwob_eval._run_one_task("does-not-exist", max_steps=15, provider=None, headless=True)


@pytest.mark.asyncio
async def test_run_one_task_raises_runtime_error_when_package_not_installed():
    with patch("backend.app.core.miniwob_eval.MINIWOB_TASK_DIR", None):
        with pytest.raises(RuntimeError, match="pip install miniwob"):
            await miniwob_eval._run_one_task("click-test", max_steps=15, provider=None, headless=True)


@pytest.mark.asyncio
async def test_run_miniwob_evaluation_continues_after_one_task_errors():
    async def _fake_run_one_task(task_name, max_steps, provider, headless):
        if task_name == "broken-task":
            raise RuntimeError("browser launch failed")
        return MiniWobResult(
            task=task_name, utterance="u", success=True, reward=1.0,
            steps=3, total_tokens=100, message="ok",
        )

    with patch("backend.app.core.miniwob_eval._run_one_task", side_effect=_fake_run_one_task):
        report = await run_miniwob_evaluation(tasks=["broken-task", "click-test"])

    assert len(report.results) == 2
    failed, succeeded = report.results
    assert failed.task == "broken-task"
    assert failed.success is False
    assert failed.error == "RuntimeError: browser launch failed"
    assert succeeded.success is True
    assert succeeded.error is None


@pytest.mark.asyncio
async def test_run_miniwob_evaluation_defaults_to_default_tasks():
    async def _fake_run_one_task(task_name, max_steps, provider, headless):
        return MiniWobResult(
            task=task_name, utterance="u", success=True, reward=1.0,
            steps=1, total_tokens=1, message="ok",
        )

    with patch("backend.app.core.miniwob_eval._run_one_task", side_effect=_fake_run_one_task):
        report = await run_miniwob_evaluation()

    assert [r.task for r in report.results] == DEFAULT_TASKS


@pytest.mark.asyncio
async def test_auto_approve_returns_true():
    assert await _auto_approve({"type": "submit", "index": 1}) is True


def test_init_script_embeds_seed_and_extended_episode_time():
    script = _init_script(424242)
    assert "424242" in script
    assert str(miniwob_eval._EPISODE_MAX_TIME_MS) in script
    assert "Math.random" in script
    assert "core.startEpisode" in script


# --- MiniWobReport aggregate properties (mirror EvaluationReport tests) ---


def test_miniwob_report_aggregates_success_rate_avg_steps_avg_tokens():
    report = MiniWobReport(results=[
        MiniWobResult(task="a", utterance="u", success=True, reward=1.0, steps=4, total_tokens=100, message="ok"),
        MiniWobResult(task="b", utterance="u", success=False, reward=-1.0, steps=10, total_tokens=300, message="fail"),
    ])

    assert report.success_rate == 0.5
    assert report.avg_steps == 7.0
    assert report.avg_tokens == 200.0


def test_miniwob_report_empty_results_does_not_divide_by_zero():
    report = MiniWobReport()

    assert report.success_rate == 0.0
    assert report.avg_steps == 0.0
    assert report.avg_tokens == 0.0

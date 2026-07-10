from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.app.core import llm
from backend.app.core.actions import ActionResult
from backend.app.core.orchestrator import (
    Orchestrator,
    _MAX_PREMATURE_FALSE_FINISH_RETRIES,
    _PREMATURE_FALSE_FINISH_NUDGE,
)

# ทุกเทสต์ mock ทั้ง Playwright และ llm.next_action — ไม่เปิด browser จริง ไม่ยิง LLM API จริง
# สำคัญ: ต้องส่ง provider="anthropic" ให้ run_task() ตรงๆ เสมอ ห้ามปล่อยให้ fallback ไป
# settings.llm_provider เพราะค่านั้นอ่านจาก .env ของเครื่อง dev แต่ละคน — ถ้า .env ตั้ง
# LLM_PROVIDER=groq ไว้ (เช่นตอนทดสอบ) แล้วเทสต์ไม่ pin provider ให้ตรงกับที่ mock ไว้
# (llm.next_action) จะหลุดไปเรียก llm.next_action_groq ตัวจริงที่ไม่ได้ mock -> ยิง Groq
# API จริงระหว่างรัน pytest (เคยเกิดขึ้นมาแล้ว)

_GOTO_OK = ActionResult(True, "goto", "ไปที่ url")
_WAIT_OK = ActionResult(True, "wait_stable", "หน้านิ่งแล้ว")


def _patch_browser():
    """mock chain ให้ตรงกับของจริง:
    async_playwright() -> (sync) helper -> await .start() -> playwright
    -> await playwright.chromium.launch() -> browser -> await browser.new_page() -> page
    """
    mock_page = AsyncMock()
    mock_browser = AsyncMock()
    mock_browser.new_page = AsyncMock(return_value=mock_page)
    mock_browser.close = AsyncMock()

    mock_playwright_instance = AsyncMock()
    mock_playwright_instance.chromium.launch = AsyncMock(return_value=mock_browser)
    mock_playwright_instance.stop = AsyncMock()

    mock_p_helper = MagicMock()
    mock_p_helper.start = AsyncMock(return_value=mock_playwright_instance)

    mock_async_playwright = MagicMock(return_value=mock_p_helper)

    return mock_async_playwright, mock_browser, mock_playwright_instance


@pytest.mark.asyncio
async def test_run_task_stops_immediately_on_finish_task():
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "[0] button 'Go'"))), \
         patch("backend.app.core.orchestrator.execute") as mock_execute, \
         patch("backend.app.core.llm.build_client", return_value="fake-client"), \
         patch(
             "backend.app.core.orchestrator.llm.next_action",
             AsyncMock(return_value=(
                 "finish_task", {"success": True, "message": "เสร็จแล้ว"}, "", [],
                 llm.TokenUsage(input_tokens=100, output_tokens=20),
             )),
         ):
        result = await Orchestrator().run_task("https://example.com", "some goal", provider="anthropic")

    assert result["success"] is True
    assert result["steps"] == 0
    assert result["message"] == "เสร็จแล้ว"
    # history มี record ของ goto เริ่มต้นเสมอ แม้ finish_task ทันทีโดยไม่มี action อื่น
    assert result["history"] == [
        {"step": 0, "cmd": {"type": "goto", "url": "https://example.com"}, "result": str(_GOTO_OK)}
    ]
    # token ของรอบ next_action ที่นำไปสู่ finish_task ต้องถูกนับรวมด้วย แม้ไม่มี browser action เกิดขึ้นเลย
    assert result["tokens"] == {"input": 100, "output": 20, "cache_read": 0, "cache_creation": 0}
    mock_execute.assert_not_called()
    mock_browser.close.assert_awaited_once()
    mock_playwright_ctx.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_task_executes_action_then_finishes():
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    click_result = ActionResult(True, "click(2)", "คลิกสำเร็จ")

    next_action_calls = [
        ("browser_action", {"type": "click", "index": 2}, "tool_1", ["m1"], llm.TokenUsage(input_tokens=50, output_tokens=10)),
        ("finish_task", {"success": True, "message": "เพิ่มลงตะกร้าแล้ว"}, "", ["m2"], llm.TokenUsage(input_tokens=60, output_tokens=15)),
    ]

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "[2] button 'Add to cart'"))), \
         patch("backend.app.core.orchestrator.execute", AsyncMock(return_value=click_result)) as mock_execute, \
         patch("backend.app.core.orchestrator.llm.append_tool_result", side_effect=lambda m, tid, r: m + [r]), \
         patch("backend.app.core.orchestrator.llm.next_action", AsyncMock(side_effect=next_action_calls)):
        result = await Orchestrator().run_task("https://example.com", "add item to cart", provider="anthropic")

    assert result["success"] is True
    assert result["steps"] == 1
    assert result["message"] == "เพิ่มลงตะกร้าแล้ว"
    mock_execute.assert_awaited_once_with(
        mock_browser.new_page.return_value, {"type": "click", "index": 2}, ask_user_func=None
    )
    assert result["history"] == [
        {"step": 0, "cmd": {"type": "goto", "url": "https://example.com"}, "result": str(_GOTO_OK)},
        {
            "step": 1,
            "cmd": {"type": "click", "index": 2},
            "result": str(click_result),
            "tokens": {"input": 50, "output": 10, "cache_read": 0, "cache_creation": 0},
        },
    ]
    # ต้องรวม token ของทั้ง 2 รอบ next_action (browser_action + finish_task) ไม่ใช่แค่รอบสุดท้าย
    assert result["tokens"] == {"input": 110, "output": 25, "cache_read": 0, "cache_creation": 0}


@pytest.mark.asyncio
async def test_run_task_stops_at_max_steps_without_finish_task():
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    scroll_result = ActionResult(True, "scroll(down)", "เลื่อนแล้ว")

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.execute", AsyncMock(return_value=scroll_result)) as mock_execute, \
         patch("backend.app.core.orchestrator.llm.append_tool_result", side_effect=lambda m, tid, r: m), \
         patch(
             "backend.app.core.orchestrator.llm.next_action",
             AsyncMock(return_value=(
                 "browser_action", {"type": "scroll", "direction": "down"}, "tool_x", [],
                 llm.TokenUsage(input_tokens=30, output_tokens=5),
             )),
         ):
        result = await Orchestrator().run_task(
            "https://example.com", "goal that never finishes", max_steps=3, provider="anthropic"
        )

    assert result["success"] is False
    assert result["steps"] == 3
    assert mock_execute.await_count == 3
    # token สะสมของ next_action ต้องนับทุกรอบ (3 รอบ) ไม่ใช่แค่รอบเดียว
    assert result["tokens"] == {"input": 90, "output": 15, "cache_read": 0, "cache_creation": 0}


@pytest.mark.asyncio
async def test_run_task_closes_browser_even_if_action_raises():
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(side_effect=RuntimeError("boom"))):
        with pytest.raises(RuntimeError):
            await Orchestrator().run_task("https://example.com", "goal", provider="anthropic")

    mock_browser.close.assert_awaited_once()
    mock_playwright_ctx.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_task_overrides_premature_finish_task_false_then_succeeds():
    """เจอบ่อยกับ Llama บน Groq: เรียก finish_task(success=false) ทั้งที่ยังมี action
    ที่ทำต่อได้ชัดเจน (เช่น เห็นปุ่ม Add to cart แต่ไม่กด) — ต้องไม่ยอมรับทันที เตือนแล้ว
    บังคับให้ลองต่อ ไม่ใช่หยุด task กลางคันทั้งที่ยังทำได้"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    click_result = ActionResult(True, "click(5)", "เพิ่มลงตะกร้าสำเร็จ")

    next_action_calls = [
        ("finish_task", {"success": False, "message": "ทำต่อไม่ได้"}, "tool_f1", ["m1"], llm.TokenUsage()),
        ("browser_action", {"type": "click", "index": 5}, "tool_2", ["m2"], llm.TokenUsage()),
        ("finish_task", {"success": True, "message": "เพิ่มลงตะกร้าแล้ว"}, "", ["m3"], llm.TokenUsage()),
    ]
    append_tool_result_mock = MagicMock(side_effect=lambda m, tid, r: m + [r])

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "[5] button 'Add to cart'"))), \
         patch("backend.app.core.orchestrator.execute", AsyncMock(return_value=click_result)) as mock_execute, \
         patch("backend.app.core.orchestrator.llm.append_tool_result", append_tool_result_mock), \
         patch("backend.app.core.orchestrator.llm.next_action", AsyncMock(side_effect=next_action_calls)):
        result = await Orchestrator().run_task("https://example.com", "add item to cart", provider="anthropic")

    assert result["success"] is True
    assert result["steps"] == 1
    mock_execute.assert_awaited_once_with(
        mock_browser.new_page.return_value, {"type": "click", "index": 5}, ask_user_func=None
    )
    # ต้องเตือนกลับเข้า tool_f1 (finish_task call ที่ถูกปฏิเสธ) ก่อนลองต่อ
    append_tool_result_mock.assert_any_call(["m1"], "tool_f1", _PREMATURE_FALSE_FINISH_NUDGE)


@pytest.mark.asyncio
async def test_run_task_accepts_finish_task_false_after_max_premature_retries():
    """ถ้าโมเดลยืนยัน finish_task(success=false) ซ้ำเกิน quota การเตือนจริงๆ ต้องยอม
    รับว่าทำต่อไม่ได้จริง ไม่ใช่บังคับลองต่อไม่มีที่สิ้นสุด"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.execute") as mock_execute, \
         patch("backend.app.core.orchestrator.llm.append_tool_result", side_effect=lambda m, tid, r: m + [r]), \
         patch(
             "backend.app.core.orchestrator.llm.next_action",
             AsyncMock(return_value=(
                 "finish_task", {"success": False, "message": "ไปต่อไม่ได้จริงๆ"}, "tool_f", [],
                 llm.TokenUsage(),
             )),
         ) as mock_next_action:
        result = await Orchestrator().run_task("https://example.com", "goal", provider="anthropic")

    assert result["success"] is False
    assert result["message"] == "ไปต่อไม่ได้จริงๆ"
    # เตือนไป _MAX_PREMATURE_FALSE_FINISH_RETRIES ครั้ง + ครั้งสุดท้ายที่ยอมรับ = +1
    assert mock_next_action.await_count == _MAX_PREMATURE_FALSE_FINISH_RETRIES + 1
    mock_execute.assert_not_called()


@pytest.mark.asyncio
async def test_run_task_accepts_finish_task_false_immediately_when_no_tool_use_id():
    """finish_task(success=false) จาก fallback ตอนโมเดลไม่ยอมเรียก tool เลย (tool_use_id
    ว่าง) ไม่มี tool call จริงให้ผูก tool_result กลับ — ต้องยอมรับทันที ห้ามพยายามเตือน"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch(
             "backend.app.core.orchestrator.llm.next_action",
             AsyncMock(return_value=(
                 "finish_task", {"success": False, "message": "no tool call"}, "", [],
                 llm.TokenUsage(),
             )),
         ) as mock_next_action:
        result = await Orchestrator().run_task("https://example.com", "goal", provider="anthropic")

    assert result["success"] is False
    assert result["message"] == "no tool call"
    assert mock_next_action.await_count == 1

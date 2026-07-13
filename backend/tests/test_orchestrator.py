from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.app.core import llm
from backend.app.core.actions import ActionResult
from backend.app.core.orchestrator import (
    Orchestrator,
    _MAX_CONSECUTIVE_IDENTICAL_ACTIONS,
    _MAX_PREMATURE_FALSE_FINISH_RETRIES,
    _PREMATURE_FALSE_FINISH_NUDGE,
    _RAG_CHUNKS_PER_STEP,
)

# ทุกเทสต์ mock ทั้ง Playwright และ llm.next_action — ไม่เปิด browser จริง ไม่ยิง LLM API จริง
# สำคัญ: ต้องส่ง provider="anthropic" ให้ run_task() ตรงๆ เสมอ ห้ามปล่อยให้ fallback ไป
# settings.llm_provider เพราะค่านั้นอ่านจาก .env ของเครื่อง dev แต่ละคน — ถ้า .env ตั้ง
# LLM_PROVIDER=groq ไว้ (เช่นตอนทดสอบ) แล้วเทสต์ไม่ pin provider ให้ตรงกับที่ mock ไว้
# (llm.next_action) จะหลุดไปเรียก llm.next_action_groq ตัวจริงที่ไม่ได้ mock -> ยิง Groq
# API จริงระหว่างรัน pytest (เคยเกิดขึ้นมาแล้ว)

_GOTO_OK = ActionResult(True, "goto", "ไปที่ url")
_WAIT_OK = ActionResult(True, "wait_stable", "หน้านิ่งแล้ว")


# pacing delay ท้ายทุก step (_STEP_PACING_DELAY_SECONDS) กันไม่ให้ test suite ช้าจริง —
# เหมือน test_actions.py ที่ mock asyncio.sleep กัน _ACTION_RETRY_DELAY_SEC ค้าง
@pytest.fixture(autouse=True)
def _no_real_sleep():
    with patch("backend.app.core.orchestrator.asyncio.sleep", AsyncMock()) as mock_sleep:
        yield mock_sleep


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

    # สลับ direction ทุกครั้งกัน loop-detection guard (W5) เข้าใจผิดว่าเป็น action เดิม
    # ซ้ำติดกัน — เทสต์นี้อยากวัดพฤติกรรม max_steps ตรงๆ ไม่ใช่ loop guard
    next_action_calls = [
        (
            "browser_action", {"type": "scroll", "direction": "down" if i % 2 == 0 else "up"}, f"tool_{i}", [],
            llm.TokenUsage(input_tokens=30, output_tokens=5),
        )
        for i in range(3)
    ]

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.execute", AsyncMock(return_value=scroll_result)) as mock_execute, \
         patch("backend.app.core.orchestrator.llm.append_tool_result", side_effect=lambda m, tid, r: m), \
         patch("backend.app.core.orchestrator.llm.next_action", AsyncMock(side_effect=next_action_calls)):
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


@pytest.mark.asyncio
async def test_run_task_confirm_plan_stops_before_any_action_when_user_declines():
    """confirm_plan=True: ต้องโชว์แผนแล้วรอ user ยืนยันก่อน — ถ้า user ปฏิเสธ ห้ามลงมือ
    ทำ action ใดๆ เลย (ห้ามเรียก next_action/execute เลยแม้แต่ครั้งเดียว)"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    ask_user_func = AsyncMock(return_value=False)

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "[0] button 'Go'"))), \
         patch("backend.app.core.orchestrator.execute") as mock_execute, \
         patch("backend.app.core.orchestrator.llm.generate_plan", AsyncMock(return_value="1. ทำ A\n2. ทำ B")) as mock_generate_plan, \
         patch("backend.app.core.orchestrator.retriever.retrieve") as mock_retrieve, \
         patch("backend.app.core.orchestrator.llm.next_action") as mock_next_action:
        result = await Orchestrator().run_task(
            "https://example.com", "some goal", provider="anthropic",
            confirm_plan=True, ask_user_func=ask_user_func,
        )

    assert result["success"] is False
    assert result["steps"] == 0
    assert result["plan"] == "1. ทำ A\n2. ทำ B"
    mock_generate_plan.assert_awaited_once()
    ask_user_func.assert_awaited_once_with({"type": "confirm_plan", "plan": "1. ทำ A\n2. ทำ B"})
    mock_next_action.assert_not_called()
    mock_execute.assert_not_called()
    # W6[B]: retrieve() ต่อเข้าแค่ per-step loop เท่านั้น ไม่ใช่ generate_plan — ถ้า loop
    # ไม่เคยเริ่มเลย (user ปฏิเสธแผน) retrieve() ก็ต้องไม่ถูกเรียกเลยเช่นกัน
    mock_retrieve.assert_not_called()


@pytest.mark.asyncio
async def test_run_task_confirm_plan_proceeds_when_user_approves():
    """confirm_plan=True + user ยืนยัน -> loop ต้องทำงานตามปกติต่อ ไม่ต่างจากไม่เปิด
    confirm_plan เลย นอกจากมี plan text แนบมาด้วยตอนจบ"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    ask_user_func = AsyncMock(return_value=True)

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "[0] button 'Go'"))), \
         patch("backend.app.core.orchestrator.llm.generate_plan", AsyncMock(return_value="1. ทำ A")), \
         patch(
             "backend.app.core.orchestrator.llm.next_action",
             AsyncMock(return_value=("finish_task", {"success": True, "message": "เสร็จแล้ว"}, "", [], llm.TokenUsage())),
         ) as mock_next_action:
        result = await Orchestrator().run_task(
            "https://example.com", "some goal", provider="anthropic",
            confirm_plan=True, ask_user_func=ask_user_func,
        )

    assert result["success"] is True
    assert result["plan"] == "1. ทำ A"
    mock_next_action.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_task_without_confirm_plan_skips_plan_generation_entirely():
    """confirm_plan=False (default) -> ห้ามเรียก llm.generate_plan เลย กันเสีย token
    เปล่าๆ กับ use case ที่ไม่ต้องการ gate นี้"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.llm.generate_plan") as mock_generate_plan, \
         patch(
             "backend.app.core.orchestrator.llm.next_action",
             AsyncMock(return_value=("finish_task", {"success": True, "message": "ok"}, "", [], llm.TokenUsage())),
         ):
        result = await Orchestrator().run_task("https://example.com", "goal", provider="anthropic")

    assert result["plan"] is None
    mock_generate_plan.assert_not_called()


@pytest.mark.asyncio
async def test_run_task_stops_on_repeated_identical_action():
    """loop-detection guard: บาง provider (เจอกับ Llama บน Groq) ยังวนเรียก
    browser_action เดิมเป๊ะๆ ซ้ำๆ แม้ execute() จะสำเร็จทุกครั้ง — ต้องหยุด task เอง
    ไม่ปล่อยให้วนจนหมด max_steps เสีย token ไปเรื่อยๆ โดยไม่มีความคืบหน้า"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    click_result = ActionResult(True, "click(5)", "คลิกสำเร็จ")
    same_action = ("browser_action", {"type": "click", "index": 5}, "tool_x", [], llm.TokenUsage())

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.execute", AsyncMock(return_value=click_result)) as mock_execute, \
         patch("backend.app.core.orchestrator.llm.append_tool_result", side_effect=lambda m, tid, r: m), \
         patch("backend.app.core.orchestrator.llm.next_action", AsyncMock(return_value=same_action)) as mock_next_action:
        result = await Orchestrator().run_task(
            "https://example.com", "goal", max_steps=10, provider="anthropic"
        )

    assert result["success"] is False
    assert "ซ้ำ" in result["message"]
    # ยิงจริงแค่ (N-1) ครั้ง เพราะการเรียกซ้ำครั้งที่ N ถูกสกัดไว้ก่อน execute()
    assert result["steps"] == _MAX_CONSECUTIVE_IDENTICAL_ACTIONS - 1
    assert mock_execute.await_count == _MAX_CONSECUTIVE_IDENTICAL_ACTIONS - 1
    assert mock_next_action.await_count == _MAX_CONSECUTIVE_IDENTICAL_ACTIONS


@pytest.mark.asyncio
async def test_run_task_loop_guard_does_not_trigger_for_varied_actions():
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    click_result = ActionResult(True, "click", "สำเร็จ")

    next_action_calls = [
        ("browser_action", {"type": "click", "index": 1}, "t1", [], llm.TokenUsage()),
        ("browser_action", {"type": "click", "index": 2}, "t2", [], llm.TokenUsage()),
        ("browser_action", {"type": "click", "index": 1}, "t3", [], llm.TokenUsage()),
        ("finish_task", {"success": True, "message": "เสร็จ"}, "", [], llm.TokenUsage()),
    ]

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.execute", AsyncMock(return_value=click_result)) as mock_execute, \
         patch("backend.app.core.orchestrator.llm.append_tool_result", side_effect=lambda m, tid, r: m), \
         patch("backend.app.core.orchestrator.llm.next_action", AsyncMock(side_effect=next_action_calls)):
        result = await Orchestrator().run_task(
            "https://example.com", "goal", max_steps=10, provider="anthropic"
        )

    assert result["success"] is True
    assert mock_execute.await_count == 3


@pytest.mark.asyncio
async def test_run_task_loop_guard_resets_count_after_different_action():
    """A, A, B, A, A -> ไม่มีช่วงไหนซ้ำติดกันครบ _MAX_CONSECUTIVE_IDENTICAL_ACTIONS
    ครั้ง (สูงสุดคือ 2 ติดกัน) ต้องไม่ trigger — พิสูจน์ว่า count reset จริงตอนเจอ
    action ต่างจากเดิม ไม่ใช่แค่นับสะสมรวมทั้ง task"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    click_result = ActionResult(True, "click", "สำเร็จ")
    action_a = {"type": "click", "index": 1}
    action_b = {"type": "click", "index": 2}

    next_action_calls = [
        ("browser_action", action_a, "t1", [], llm.TokenUsage()),
        ("browser_action", action_a, "t2", [], llm.TokenUsage()),
        ("browser_action", action_b, "t3", [], llm.TokenUsage()),
        ("browser_action", action_a, "t4", [], llm.TokenUsage()),
        ("browser_action", action_a, "t5", [], llm.TokenUsage()),
        ("finish_task", {"success": True, "message": "เสร็จ"}, "", [], llm.TokenUsage()),
    ]

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.execute", AsyncMock(return_value=click_result)) as mock_execute, \
         patch("backend.app.core.orchestrator.llm.append_tool_result", side_effect=lambda m, tid, r: m), \
         patch("backend.app.core.orchestrator.llm.next_action", AsyncMock(side_effect=next_action_calls)):
        result = await Orchestrator().run_task(
            "https://example.com", "goal", max_steps=10, provider="anthropic"
        )

    assert result["success"] is True
    assert mock_execute.await_count == 5


@pytest.mark.asyncio
async def test_run_task_calls_retrieve_with_goal_page_text_and_k_then_passes_result_into_next_action():
    """W6[B]: ทุก step ต้องดึงคู่มือด้วย retrieve(query=goal, page_state=page_text ปัจจุบัน,
    k=_RAG_CHUNKS_PER_STEP) แล้วเอาผลลัพธ์ (join เป็น bullet list) ส่งต่อเข้า next_action()
    เป็น manual_context"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "[0] button 'Go'"))), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=["chunk1", "chunk2", "chunk3"]) as mock_retrieve, \
         patch(
             "backend.app.core.orchestrator.llm.next_action",
             AsyncMock(return_value=("finish_task", {"success": True, "message": "เสร็จแล้ว"}, "", [], llm.TokenUsage())),
         ) as mock_next_action:
        await Orchestrator().run_task("https://example.com", "some goal", provider="anthropic")

    mock_retrieve.assert_called_once_with(query="some goal", page_state="[0] button 'Go'", k=_RAG_CHUNKS_PER_STEP)
    manual_context = mock_next_action.await_args.args[-1]
    assert manual_context == "- chunk1\n- chunk2\n- chunk3"


@pytest.mark.asyncio
async def test_run_task_calls_retrieve_every_step_with_that_steps_page_text():
    """retrieve() ต้องถูกเรียกใหม่ทุก step ตาม page_text ของ step นั้นๆ (ไม่ใช่แค่ครั้งเดียว
    ตอนเริ่ม task) — พิสูจน์ด้วยการให้ get_snapshot คืน page_text ต่างกันทุก step"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    click_result = ActionResult(True, "click", "สำเร็จ")

    page_texts = [([], "[0] step1 page"), ([], "[0] step2 page"), ([], "[0] step3 page")]
    next_action_calls = [
        ("browser_action", {"type": "click", "index": 1}, "t1", [], llm.TokenUsage()),
        ("browser_action", {"type": "click", "index": 2}, "t2", [], llm.TokenUsage()),
        ("finish_task", {"success": True, "message": "เสร็จ"}, "", [], llm.TokenUsage()),
    ]

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(side_effect=page_texts)), \
         patch("backend.app.core.orchestrator.execute", AsyncMock(return_value=click_result)), \
         patch("backend.app.core.orchestrator.llm.append_tool_result", side_effect=lambda m, tid, r: m), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=["c"]) as mock_retrieve, \
         patch("backend.app.core.orchestrator.llm.next_action", AsyncMock(side_effect=next_action_calls)):
        await Orchestrator().run_task("https://example.com", "goal", max_steps=10, provider="anthropic")

    assert mock_retrieve.call_count == 3
    called_page_states = [c.kwargs["page_state"] for c in mock_retrieve.call_args_list]
    assert called_page_states == ["[0] step1 page", "[0] step2 page", "[0] step3 page"]


@pytest.mark.asyncio
async def test_run_task_manual_context_is_empty_string_when_retrieve_returns_no_chunks():
    """ยังไม่มีคู่มือ ingest ไว้ (หรือหาไม่เจออะไรตรงกัน) -> retrieve() คืน [] -> manual_context
    ต้องเป็น "" เฉยๆ ไม่ใช่ None หรือข้อความ placeholder"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=[]), \
         patch(
             "backend.app.core.orchestrator.llm.next_action",
             AsyncMock(return_value=("finish_task", {"success": True, "message": "เสร็จแล้ว"}, "", [], llm.TokenUsage())),
         ) as mock_next_action:
        await Orchestrator().run_task("https://example.com", "goal", provider="anthropic")

    manual_context = mock_next_action.await_args.args[-1]
    assert manual_context == ""

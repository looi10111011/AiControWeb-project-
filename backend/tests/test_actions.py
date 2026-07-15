from unittest.mock import AsyncMock, patch

import pytest
from playwright.async_api import TimeoutError as PWTimeout

from backend.app.core.actions import ActionResult, _ELEMENT_ACTION_TIMEOUT_MS, execute

# W5: retry ระดับ click/fill/select/check เมื่อ action ล้มเหลว — ไม่เสีย LLM token
# เพราะ retry อยู่ใน actions.py เอง ไม่ต้องรอ next_action() รอบใหม่ ทุกเทสต์ mock
# asyncio.sleep กันไม่ให้รอ delay จริง (_ACTION_RETRY_DELAY_SEC) ตอนรัน test suite


@pytest.fixture(autouse=True)
def _no_real_sleep():
    with patch("backend.app.core.actions.asyncio.sleep", AsyncMock()) as mock_sleep:
        yield mock_sleep


@pytest.mark.asyncio
async def test_execute_click_succeeds_first_try_without_retry():
    mock_page = AsyncMock()

    result = await execute(mock_page, {"type": "click", "index": 2})

    assert result.success is True
    assert "ลองครั้งที่" not in result.message  # สำเร็จรอบแรก ไม่ต้องพูดถึง retry เลย
    mock_page.click.assert_awaited_once()


@pytest.mark.asyncio
async def test_click_uses_short_element_timeout_by_default():
    """W5: timeout สั้นลงเหลือ 3s (จากเดิม 5s) กัน action ค้างนานเกินไปเวลารวมกับ
    retry loop — ยืนยันว่า page.click() ถูกเรียกด้วย timeout นี้จริง ไม่ใช่แค่ comment"""
    mock_page = AsyncMock()

    await execute(mock_page, {"type": "click", "index": 2})

    assert _ELEMENT_ACTION_TIMEOUT_MS == 3000
    mock_page.click.assert_awaited_once_with(
        '[data-ai-index="2"]', timeout=_ELEMENT_ACTION_TIMEOUT_MS
    )


@pytest.mark.asyncio
async def test_execute_click_retries_on_transient_failure_then_succeeds(_no_real_sleep):
    mock_page = AsyncMock()
    mock_page.click = AsyncMock(side_effect=[PWTimeout("not ready yet"), None])

    result = await execute(mock_page, {"type": "click", "index": 2})

    assert result.success is True
    assert "ลองครั้งที่ 2/3" in result.message
    assert mock_page.click.await_count == 2
    _no_real_sleep.assert_awaited_once()  # หน่วงแค่ระหว่างครั้งที่ 1->2 ครั้งเดียว


@pytest.mark.asyncio
async def test_execute_fill_gives_up_after_max_retries(_no_real_sleep):
    mock_page = AsyncMock()
    mock_page.fill = AsyncMock(side_effect=PWTimeout("still not there"))

    result = await execute(mock_page, {"type": "fill", "index": 0, "text": "hello"})

    assert result.success is False
    assert "ลองแล้ว 3 ครั้ง" in result.message
    assert mock_page.fill.await_count == 3
    assert _no_real_sleep.await_count == 2  # หน่วงระหว่างแต่ละครั้ง ไม่หน่วงหลังครั้งสุดท้าย


@pytest.mark.asyncio
async def test_execute_select_and_check_also_get_retried(_no_real_sleep):
    mock_page = AsyncMock()
    mock_page.select_option = AsyncMock(side_effect=[PWTimeout("boom"), None])
    result_select = await execute(mock_page, {"type": "select", "index": 1, "label": "A"})
    assert result_select.success is True
    assert mock_page.select_option.await_count == 2

    mock_page2 = AsyncMock()
    mock_page2.check = AsyncMock(side_effect=[PWTimeout("boom"), None])
    result_check = await execute(mock_page2, {"type": "check", "index": 3})
    assert result_check.success is True
    assert mock_page2.check.await_count == 2


@pytest.mark.asyncio
async def test_execute_does_not_retry_goto_on_failure():
    """goto/scroll/go_back/switch_tab/wait ไม่ retry เพราะ fail มักไม่ใช่เรื่อง DOM-timing
    (เช่น URL ผิดก็จะผิดซ้ำทุกครั้ง) — ต้อง dispatch แค่ครั้งเดียว"""
    mock_page = AsyncMock()
    mock_page.goto = AsyncMock(side_effect=Exception("DNS ผิด"))

    result = await execute(mock_page, {"type": "goto", "url": "https://not-a-real-domain.invalid"})

    assert result.success is False
    assert mock_page.goto.await_count == 1


@pytest.mark.asyncio
async def test_execute_retries_needs_confirmation_alias_action():
    """submit/delete/purchase/pay alias ไปเรียก click() ตัวเดิม ต้อง retry เหมือน click ปกติ"""
    mock_page = AsyncMock()
    mock_page.click = AsyncMock(side_effect=[PWTimeout("boom"), None])
    ask_user_func = AsyncMock(return_value=True)

    result = await execute(mock_page, {"type": "submit", "index": 3}, ask_user_func=ask_user_func)

    assert result.success is True
    assert result.action == "submit(3)"
    assert mock_page.click.await_count == 2
    ask_user_func.assert_awaited_once()  # permission check ถามแค่ครั้งเดียว ไม่ถามซ้ำต่อ retry


# W3[A] (ปิดจ็อบ 2026-07-15): switch_tab() implement ไว้แล้วตั้งแต่ก่อนหน้านี้ (dispatch
# ผ่าน execute()/enum ของ llm.py ครบ) แต่ไม่เคยมี unit test เลย — เพิ่มให้ครบตาม
# มาตรฐานเดียวกับ action อื่นในไฟล์นี้


@pytest.mark.asyncio
async def test_execute_switch_tab_succeeds_when_tab_exists():
    mock_target_page = AsyncMock()
    mock_other_page = AsyncMock()
    mock_page = AsyncMock()
    mock_page.context.pages = [mock_other_page, mock_target_page]

    result = await execute(mock_page, {"type": "switch_tab", "tab_index": 1})

    assert result.success is True
    assert result.action == "switch_tab(1)"
    mock_target_page.bring_to_front.assert_awaited_once()
    mock_other_page.bring_to_front.assert_not_awaited()


@pytest.mark.asyncio
async def test_execute_switch_tab_fails_when_tab_index_out_of_range():
    mock_page = AsyncMock()
    mock_page.context.pages = [AsyncMock()]  # มีแค่ 1 tab (index 0)

    result = await execute(mock_page, {"type": "switch_tab", "tab_index": 5})

    assert result.success is False
    assert "มีแค่ 1 tab" in result.message


@pytest.mark.asyncio
async def test_execute_does_not_retry_switch_tab_on_failure():
    """switch_tab ไม่ผ่าน _dispatch_with_retry (เหมือน goto/scroll/go_back/wait) —
    fail แล้วต้อง fail ทันทีไม่ retry"""
    mock_page = AsyncMock()
    mock_page.context.pages = []  # ไม่มี tab ให้สลับเลย

    result = await execute(mock_page, {"type": "switch_tab", "tab_index": 0})

    assert result.success is False
    assert "มีแค่ 0 tab" in result.message

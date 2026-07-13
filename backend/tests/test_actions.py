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

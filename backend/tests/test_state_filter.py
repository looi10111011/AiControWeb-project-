"""W19: core/state_filter.py — Deterministic State Filter (ดู W19.txt ข้อ 6)

Locator methods (input_value/is_checked/is_disabled) ต้อง mock ผ่าน page.locator ที่เป็น
MagicMock ธรรมดา (ไม่ใช่ AsyncMock ทั้งก้อน) เหมือน pattern ของ
test_actions.py::_make_select_mock_page — ไม่งั้น page.locator(selector) จะได้ coroutine
กลับมาแทน Locator object จริง"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.app.core.state_filter import (
    check_checkbox_redundant,
    check_click_redundant,
    check_fill_redundant,
    check_scroll_redundant,
)


def _make_locator_page(**locator_async_methods):
    mock_page = AsyncMock()
    locator = MagicMock()
    for name, return_value in locator_async_methods.items():
        setattr(locator, name, AsyncMock(return_value=return_value))
    mock_page.locator = MagicMock(return_value=locator)
    return mock_page, locator


# ---------------- fill ----------------


@pytest.mark.asyncio
async def test_fill_redundant_when_value_already_matches():
    mock_page, locator = _make_locator_page(input_value="standard_user")

    reason = await check_fill_redundant(mock_page, 0, "standard_user")

    assert reason is not None
    assert "standard_user" in reason
    locator.input_value.assert_awaited_once()


@pytest.mark.asyncio
async def test_fill_not_redundant_when_value_differs():
    mock_page, _ = _make_locator_page(input_value="")

    reason = await check_fill_redundant(mock_page, 0, "standard_user")

    assert reason is None


@pytest.mark.asyncio
async def test_fill_redundant_check_fails_safe_on_error():
    """element หาย/frame ปิด/mock ที่ไม่ได้ config (bare AsyncMock) — ต้องไม่ throw และไม่
    เดาว่า redundant ทั้งที่เช็คไม่ได้จริง"""
    mock_page = AsyncMock()  # bare -- .locator() คืน coroutine ไม่ใช่ Locator จริง

    reason = await check_fill_redundant(mock_page, 0, "standard_user")

    assert reason is None


# ---------------- checkbox ----------------


@pytest.mark.asyncio
async def test_checkbox_redundant_when_already_checked():
    mock_page, locator = _make_locator_page(is_checked=True)

    reason = await check_checkbox_redundant(mock_page, 3)

    assert reason is not None
    locator.is_checked.assert_awaited_once()


@pytest.mark.asyncio
async def test_checkbox_not_redundant_when_unchecked():
    mock_page, _ = _make_locator_page(is_checked=False)

    reason = await check_checkbox_redundant(mock_page, 3)

    assert reason is None


# ---------------- click / disabled ----------------


@pytest.mark.asyncio
async def test_click_redundant_when_element_disabled():
    mock_page, locator = _make_locator_page(is_disabled=True)

    reason = await check_click_redundant(mock_page, 7)

    assert reason is not None
    locator.is_disabled.assert_awaited_once()


@pytest.mark.asyncio
async def test_click_not_redundant_when_element_enabled():
    mock_page, _ = _make_locator_page(is_disabled=False)

    reason = await check_click_redundant(mock_page, 7)

    assert reason is None


# ---------------- scroll ----------------


@pytest.mark.asyncio
async def test_scroll_down_redundant_at_bottom():
    mock_page = AsyncMock()
    mock_page.evaluate = AsyncMock(return_value=True)

    reason = await check_scroll_redundant(mock_page, "down")

    assert reason is not None
    assert "bottom" in reason


@pytest.mark.asyncio
async def test_scroll_up_redundant_at_top():
    mock_page = AsyncMock()
    mock_page.evaluate = AsyncMock(return_value=True)

    reason = await check_scroll_redundant(mock_page, "up")

    assert reason is not None
    assert "top" in reason


@pytest.mark.asyncio
async def test_scroll_not_redundant_when_not_at_edge():
    mock_page = AsyncMock()
    mock_page.evaluate = AsyncMock(return_value=False)

    reason = await check_scroll_redundant(mock_page, "down")

    assert reason is None


@pytest.mark.asyncio
async def test_scroll_redundant_check_fails_safe_on_non_bool_mock_result():
    """page.evaluate ที่ไม่ได้ config เฉพาะ (bare AsyncMock) คืน MagicMock() ซึ่ง truthy
    โดย default — ต้องไม่ถูกตีความว่า "อยู่ขอบแล้ว" (เทียบ `is True` ตรงๆ ไม่ใช่ truthy เฉยๆ)"""
    mock_page = AsyncMock()  # evaluate ไม่ได้ config -- คืน MagicMock ที่ truthy

    reason = await check_scroll_redundant(mock_page, "down")

    assert reason is None

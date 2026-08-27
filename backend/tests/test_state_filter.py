"""W19: core/state_filter.py — Deterministic State Filter (ดู W19.txt ข้อ 6)

Locator methods (input_value/is_checked/is_disabled) ต้อง mock ผ่าน page.locator ที่เป็น
MagicMock ธรรมดา (ไม่ใช่ AsyncMock ทั้งก้อน) เหมือน pattern ของ
test_actions.py::_make_select_mock_page — ไม่งั้น page.locator(selector) จะได้ coroutine
กลับมาแทน Locator object จริง"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.app.core.state_filter import (
    check_checkbox_redundant,
    check_click_invalidates_indexes,
    check_click_redundant,
    check_click_target_is_native_select,
    check_fill_is_empty_noop,
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


# ---------------- fill: empty-into-empty (W_empty_fill_noop) ----------------
# บั๊กจริงจาก live run ของ goal user เอง: โมเดลสั่ง fill(text="") ลงช่องที่ว่างอยู่แล้ว 3 step
# ติดกัน (index 21/22/23) check_fill_redundant() จับได้ถูกว่า "ช่องนี้มี '' อยู่แล้ว" แต่
# execute() คืน success=True โมเดลจึงอ่านว่าทำสำเร็จแล้วไล่สั่งช่องถัดไปแบบเดียวกันต่อ


@pytest.mark.asyncio
async def test_fill_empty_into_empty_field_is_rejected_not_skipped():
    mock_page, locator = _make_locator_page(input_value="")

    reason = await check_fill_is_empty_noop(mock_page, 21, "")

    assert reason is not None
    assert "already empty" in reason
    locator.input_value.assert_awaited_once()


@pytest.mark.asyncio
async def test_fill_empty_into_a_field_with_text_is_a_real_clear_and_stays_allowed():
    """fill("") ลงช่องที่ *มี* ข้อความอยู่คือการล้างค่า (เช่น เคลียร์ filter) ถูกต้องสมบูรณ์
    ห้ามบล็อก — จึงต้องอ่านค่าปัจจุบันจริง ไม่ตัดสินจาก text=="" อย่างเดียว"""
    mock_page, _ = _make_locator_page(input_value="ESS")

    assert await check_fill_is_empty_noop(mock_page, 21, "") is None


@pytest.mark.asyncio
async def test_fill_empty_check_skips_the_page_entirely_when_text_is_not_empty():
    mock_page, locator = _make_locator_page(input_value="")

    assert await check_fill_is_empty_noop(mock_page, 21, "ESS") is None

    locator.input_value.assert_not_awaited()


@pytest.mark.asyncio
async def test_fill_empty_check_fails_safe_on_error():
    """กฎประจำไฟล์นี้: อ่านสถานะจริงไม่ได้ = ถือว่าไม่ redundant เสมอ ห้าม throw/ห้ามเดา"""
    mock_page = AsyncMock()  # bare -- .locator() คืน coroutine ไม่ใช่ Locator จริง

    assert await check_fill_is_empty_noop(mock_page, 21, "") is None


# ---------------- click: dropdown (W_chain_stale_index) ----------------
# บั๊กจริงจาก live run: click(22) '-- Select --' + then_click_index=26 วน 5 รอบโดย filter
# Role=ESS ไม่เคยติด, 3 รอบคืน "element not found", และรอบที่ then click(29) "สำเร็จ" กลับ
# ไปโดน 'Demo Source [Profile/Account Menu]' ที่ไม่เกี่ยวเลย
#
# W_chain_stale_index_kind (บั๊กของ guard นี้เองรอบแรก, live run ebeec1c6): เวอร์ชันแรกจับการ
# คลิก *ตัวเลือก* ว่าเป็น trigger ด้วย แล้วตอบว่า "The dropdown is now OPEN" ทั้งที่การเลือก
# ตัวเลือกทำให้มันปิด — โมเดลจึงไปกด '-- Select --' เปิดใหม่ วนอยู่อย่างนั้น 13 ครั้ง


@pytest.mark.asyncio
async def test_click_on_a_dropdown_trigger_says_the_dropdown_is_now_open():
    mock_page, locator = _make_locator_page(evaluate="trigger")

    reason = await check_click_invalidates_indexes(mock_page, 22)

    assert reason is not None
    assert "now OPEN" in reason
    assert "separate next step" in reason  # ต้องชี้ทางต่อ ไม่ใช่แค่บอกว่าไม่ทำ
    locator.evaluate.assert_awaited_once()


@pytest.mark.asyncio
async def test_click_on_an_option_says_the_dropdown_closed_not_opened():
    """คนละข้อความกับ trigger โดยเจตนา — ถ้าบอกผิดข้าง โมเดลจะไปเปิด dropdown ใหม่แล้ววนไม่จบ
    (บั๊กจริงของ guard นี้เองรอบแรก) และต้องยืนยันด้วยว่า "การเลือกมีผลแล้ว" ไม่งั้นโมเดลจะ
    เข้าใจว่าเลือกไม่สำเร็จแล้วลองใหม่"""
    mock_page, _ = _make_locator_page(evaluate="option")

    reason = await check_click_invalidates_indexes(mock_page, 25)

    assert reason is not None
    assert "CLOSES the dropdown" in reason
    assert "now OPEN" not in reason
    assert "Your selection was applied" in reason


@pytest.mark.asyncio
async def test_click_on_a_plain_button_still_allows_chaining():
    mock_page, _ = _make_locator_page(evaluate="")

    assert await check_click_invalidates_indexes(mock_page, 5) is None


@pytest.mark.asyncio
async def test_click_invalidates_indexes_ignores_unknown_results():
    """เทียบกับ kind ที่รู้จักตรงๆ (ไม่ใช่ truthy) — evaluate() ที่คืนค่าที่ไม่ใช่ kind จริง
    (mock ที่ไม่ได้ config, หน้าที่ error) ต้องไม่ถูกตีความว่าเป็น dropdown"""
    for junk in (True, 1, "maybe", None):
        mock_page, _ = _make_locator_page(evaluate=junk)
        assert await check_click_invalidates_indexes(mock_page, 5) is None, junk


@pytest.mark.asyncio
async def test_click_invalidates_indexes_fails_safe_on_error():
    mock_page = AsyncMock()  # bare -- .locator() คืน coroutine ไม่ใช่ Locator จริง

    assert await check_click_invalidates_indexes(mock_page, 5) is None


# ---------------- W_click_native_select ----------------


@pytest.mark.asyncio
async def test_click_on_native_select_is_rejected_with_the_right_verb():
    """W_click_native_select: คลิก <select> จริงคือ no-op ที่ Playwright รายงานว่าสำเร็จ —
    ต้องถูกปฏิเสธก่อน dispatch พร้อมชี้ไป action ที่ถูก ('select' + label) ไม่งั้นโมเดลเห็น
    [OK] แล้ววนคลิกซ้ำจนโดน loop-detection ฆ่า task (บั๊กจริงบน saucedemo)"""
    mock_page, locator = _make_locator_page(evaluate="select")

    reason = await check_click_target_is_native_select(mock_page, 2)

    assert reason is not None
    assert "'select'" in reason
    assert "index 2" in reason
    locator.evaluate.assert_awaited_once()


@pytest.mark.asyncio
async def test_click_on_non_select_element_is_allowed():
    mock_page, _ = _make_locator_page(evaluate="button")

    assert await check_click_target_is_native_select(mock_page, 2) is None


@pytest.mark.asyncio
async def test_click_native_select_check_fails_safe_on_error():
    """อ่าน tag ไม่ได้ = ไม่บล็อก (ปล่อยให้ dispatch จริงไปเจอ error ของตัวเอง) เหมือน
    check ตัวอื่นในไฟล์นี้ทุกตัว"""
    mock_page = AsyncMock()  # bare -- .locator() คืน coroutine ไม่ใช่ Locator จริง

    assert await check_click_target_is_native_select(mock_page, 2) is None


# ---------------- W_inner_scroll ----------------
# ใช้ chromium จริงกับ HTML สังเคราะห์ — เคสนี้เป็นเรื่อง layout/CSS ล้วนๆ (body สูงเท่าจอ
# แล้ว pane ข้างในเป็นตัว scroll) mock page.evaluate ไม่พิสูจน์อะไรเลย

_APP_SHELL_HTML = """<!doctype html><html><body style="margin:0;height:100vh;overflow:hidden">
<div id="pane" style="height:100vh;overflow-y:auto"><div style="height:4000px">tall</div></div>
</body></html>"""

_NORMAL_PAGE_HTML = """<!doctype html><html><body><div style="height:4000px">tall</div></body></html>"""


@pytest.mark.asyncio
async def test_scroll_is_not_reported_as_redundant_on_app_shell_layout():
    """บั๊กจริงที่ทำให้เว็บทั้งกลุ่มใช้ไม่ได้: บน layout ที่ pane ข้างในเป็นตัว scroll (รูปแบบ
    มาตรฐานของ dashboard/mail/chat/data grid) window.scrollY เป็น 0 เสมอและ
    document.scrollHeight เท่ากับ innerHeight พอดี => guard เดิมตอบ "อยู่ล่างสุดแล้ว" ตลอด
    ทุกคำสั่ง scroll จึงถูก skip โดยไม่แตะ browser เลย — agent ไปดูเนื้อหาใต้ fold ไม่ได้เลย"""
    from playwright.async_api import async_playwright

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page()
        try:
            await page.set_content(_APP_SHELL_HTML)

            assert await check_scroll_redundant(page, "down") is None
            assert await check_scroll_redundant(page, "up") is not None  # อยู่บนสุดจริง
        finally:
            await browser.close()


@pytest.mark.asyncio
async def test_scroll_still_detects_a_page_that_genuinely_cannot_scroll():
    from playwright.async_api import async_playwright

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page()
        try:
            await page.set_content("<!doctype html><html><body>short</body></html>")

            assert await check_scroll_redundant(page, "down") is not None
        finally:
            await browser.close()


@pytest.mark.asyncio
async def test_ordinary_full_page_scrolling_still_works_as_before():
    from playwright.async_api import async_playwright

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page()
        try:
            await page.set_content(_NORMAL_PAGE_HTML)

            assert await check_scroll_redundant(page, "down") is None
            assert await check_scroll_redundant(page, "up") is not None
        finally:
            await browser.close()

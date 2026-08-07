from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from playwright.async_api import TimeoutError as PWTimeout, async_playwright

from backend.app.core.actions import (
    ActionResult,
    _DIALOG_CONTAINER_SELECTOR,
    _ELEMENT_ACTION_TIMEOUT_MS,
    _MODAL_CONFIRM_BUTTON_SELECTORS,
    _MODAL_CONFIRM_CLICK_RETRIES,
    _MODAL_DETACH_TIMEOUT_MS,
    _MODAL_RELOAD_TIMEOUT_MS,
    _detect_confirmation_modal,
    execute,
    resolve_confirmation_modal,
)
from backend.app.core.perception import get_snapshot


def _make_locator(count, visible, click_raises=False):
    """W23: mock ของ Locator (.first ที่ page.locator(selector).first คืนมา) — count()/
    is_visible()/click() เป็น async method จริงตาม Playwright API"""
    m = MagicMock()
    m.count = AsyncMock(return_value=count)
    m.is_visible = AsyncMock(return_value=visible)
    m.click = AsyncMock(side_effect=Exception("not clickable")) if click_raises else AsyncMock()
    return m


def _make_select_mock_page(option_texts, select_side_effect):
    """W42: select_option() ตอนนี้เรียก target.locator(selector).locator("option").
    evaluate_all(...) ก่อนเสมอ (ดึง {text, value} จริงจาก DOM มาเทียบแบบ normalize
    whitespace) — .locator() เป็น sync method ของ Playwright จริง (คืน Locator object
    ทันที ไม่ await) ต้อง mock ด้วย MagicMock ธรรมดา (ไม่ใช่ AsyncMock ทั้งก้อนเหมือน
    mock_page อื่นๆ ในไฟล์นี้) ไม่งั้นเรียก .locator(selector) จะได้ coroutine กลับมาแทน
    Locator object จริง — value ในเทสต์นี้ตั้งให้เท่ากับ text เฉยๆ (เพียงพอสำหรับเทสต์
    retry logic ที่ไม่ได้สนใจ whitespace/value แยกจาก text)"""
    mock_page = AsyncMock()
    option_locator = MagicMock()
    option_locator.evaluate_all = AsyncMock(
        return_value=[{"text": t, "value": t} for t in option_texts]
    )
    select_locator = MagicMock()
    select_locator.locator = MagicMock(return_value=option_locator)
    mock_page.locator = MagicMock(return_value=select_locator)
    mock_page.select_option = AsyncMock(side_effect=select_side_effect)
    return mock_page

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


# ---------------- W19 ("Safe Input Replacement"): focus -> select-all -> Backspace -> fill ----------------


@pytest.mark.asyncio
async def test_execute_fill_clears_existing_text_via_select_all_and_backspace_before_typing():
    mock_page = AsyncMock()

    result = await execute(mock_page, {"type": "fill", "index": 0, "text": "เพลงรักชาติ"})

    assert result.success is True
    mock_page.click.assert_awaited_once_with('[data-ai-index="0"]', timeout=_ELEMENT_ACTION_TIMEOUT_MS)
    mock_page.press.assert_any_call('[data-ai-index="0"]', "ControlOrMeta+a", timeout=_ELEMENT_ACTION_TIMEOUT_MS)
    mock_page.press.assert_any_call('[data-ai-index="0"]', "Backspace", timeout=_ELEMENT_ACTION_TIMEOUT_MS)
    mock_page.fill.assert_awaited_once_with('[data-ai-index="0"]', "เพลงรักชาติ", timeout=_ELEMENT_ACTION_TIMEOUT_MS)
    # ลำดับต้องเป็น click -> Ctrl+A -> Backspace -> fill เท่านั้น (ไม่ใช่แค่เรียกครบทุกตัว)
    call_order = [c[0] for c in mock_page.method_calls if c[0] in ("click", "press", "fill")]
    assert call_order == ["click", "press", "press", "fill"]


@pytest.mark.asyncio
async def test_execute_select_and_check_also_get_retried(_no_real_sleep):
    mock_page = _make_select_mock_page(["A"], [PWTimeout("boom"), None])
    result_select = await execute(mock_page, {"type": "select", "index": 1, "label": "A"})
    assert result_select.success is True
    assert mock_page.select_option.await_count == 2

    mock_page2 = AsyncMock()
    mock_page2.check = AsyncMock(side_effect=[PWTimeout("boom"), None])
    result_check = await execute(mock_page2, {"type": "check", "index": 3})
    assert result_check.success is True
    assert mock_page2.check.await_count == 2


# ---------------- hover: ปุ่ม hover-to-reveal (opacity:0/visibility:hidden จนกว่าจะ hover แถวแม่) ----------------
# บั๊กที่ user รายงานจริงบน uitestingplayground.com/scrolltoclick Case 4 — perception.py
# ตอนนี้ยังติด data-ai-index ให้ปุ่มพวกนี้แล้ว (ไม่กรองทิ้งเหมือนเดิม) แต่คลิกตรงๆ รอบแรกจะ
# พลาดเพราะ CSS ยังไม่เปลี่ยนสถานะจาก hover จริง — click retry ตั้งแต่รอบ 2 เป็นต้นไปต้อง
# hover() บน element เป้าหมายก่อนคลิกซ้ำเสมอ


@pytest.mark.asyncio
async def test_execute_hover_succeeds():
    mock_page = AsyncMock()

    result = await execute(mock_page, {"type": "hover", "index": 4})

    assert result.success is True
    mock_page.hover.assert_awaited_once_with('[data-ai-index="4"]', timeout=_ELEMENT_ACTION_TIMEOUT_MS, force=True)


@pytest.mark.asyncio
async def test_execute_hover_fails_gracefully_on_timeout():
    mock_page = AsyncMock()
    mock_page.hover = AsyncMock(side_effect=PWTimeout("not found"))

    result = await execute(mock_page, {"type": "hover", "index": 4})

    assert result.success is False


@pytest.mark.asyncio
async def test_execute_click_does_not_hover_on_first_attempt():
    """รอบแรกยังคลิกตรงๆ เหมือนเดิม ไม่ hover ก่อน — กัน overhead กับปุ่มทั่วไปที่ไม่ต้อง
    hover เลยตั้งแต่แรก (ส่วนใหญ่ของ click ทั้งหมด)"""
    mock_page = AsyncMock()

    result = await execute(mock_page, {"type": "click", "index": 5})

    assert result.success is True
    mock_page.hover.assert_not_awaited()
    mock_page.click.assert_awaited_once()


@pytest.mark.asyncio
async def test_execute_click_hovers_before_retrying_on_second_attempt(_no_real_sleep):
    """ปุ่ม hover-to-reveal: คลิกตรงๆ รอบแรกพลาด (CSS ยังไม่เปลี่ยนสถานะ) — รอบ retry ที่ 2
    ต้อง hover() บน element เป้าหมายก่อนคลิกซ้ำเสมอ (ยืนยัน call order: click -> hover -> click)"""
    mock_page = AsyncMock()
    mock_page.click = AsyncMock(side_effect=[PWTimeout("not ready yet"), None])

    result = await execute(mock_page, {"type": "click", "index": 5})

    assert result.success is True
    mock_page.hover.assert_awaited_once_with('[data-ai-index="5"]', timeout=_ELEMENT_ACTION_TIMEOUT_MS, force=True)
    assert mock_page.click.await_count == 2
    # กรองเอาแค่ click/hover (ตัด query_selector ที่ resolve_frame() เรียกแทรกก่อนทุกครั้งออก)
    call_order = [c[0] for c in mock_page.method_calls if c[0] in ("click", "hover")]
    assert call_order == ["click", "hover", "click"]


@pytest.mark.asyncio
async def test_execute_click_hovers_before_every_retry_attempt_until_giving_up(_no_real_sleep):
    """ถ้าคลิกพลาดต่อเนื่องจนครบโควตา ต้อง hover ก่อนคลิกทุกรอบตั้งแต่รอบ 2 เป็นต้นไป (ไม่ใช่
    แค่รอบแรกที่พลาดครั้งเดียว)"""
    mock_page = AsyncMock()
    mock_page.click = AsyncMock(side_effect=PWTimeout("still not there"))

    result = await execute(mock_page, {"type": "click", "index": 5})

    assert result.success is False
    assert mock_page.click.await_count == 3
    assert mock_page.hover.await_count == 2  # ก่อนรอบ 2 และรอบ 3 (ไม่ใช่ก่อนรอบแรก)
    # กรองเอาแค่ click/hover (ตัด query_selector ที่ resolve_frame() เรียกแทรกก่อนทุกครั้งออก)
    call_order = [c[0] for c in mock_page.method_calls if c[0] in ("click", "hover")]
    assert call_order == ["click", "hover", "click", "hover", "click"]


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


# ---------------- W23 ("Confirmation Modal Handler" / "Overlay Action Resolver") ----------------


@pytest.mark.asyncio
async def test_detect_confirmation_modal_true_when_dialog_visible():
    mock_page = MagicMock()
    dialog = _make_locator(count=1, visible=True)
    wrapper = MagicMock()
    wrapper.first = dialog
    mock_page.locator = MagicMock(return_value=wrapper)

    result = await _detect_confirmation_modal(mock_page)

    assert result is True
    mock_page.locator.assert_called_once_with(_DIALOG_CONTAINER_SELECTOR)


@pytest.mark.asyncio
async def test_detect_confirmation_modal_false_when_no_dialog_in_dom():
    mock_page = MagicMock()
    wrapper = MagicMock()
    wrapper.first = _make_locator(count=0, visible=False)
    mock_page.locator = MagicMock(return_value=wrapper)

    result = await _detect_confirmation_modal(mock_page)

    assert result is False


@pytest.mark.asyncio
async def test_detect_confirmation_modal_fails_safe_on_bare_mock_page():
    """page ที่ไม่ได้ config เฉพาะ (bare AsyncMock ทั้งก้อน) — ต้องคืน False เงียบๆ ไม่ throw
    (เหมือน orchestrator.py::_scan_validation_errors: เช็คไม่ได้ ถือว่าไม่มีโมดัล ปลอดภัยกว่า
    เสมอที่จะไม่บล็อก/หน่วง click ที่สำเร็จอยู่แล้ว)"""
    result = await _detect_confirmation_modal(AsyncMock())

    assert result is False


@pytest.mark.asyncio
async def test_resolve_confirmation_modal_clicks_most_specific_selector_first():
    """เจอปุ่ม "Yes, Delete" ตาม selector แรกสุด (เจาะจงที่สุด) ในลิสต์ priority — ไม่ต้องไล่
    ไปหา fallback ตัวอื่นต่อ"""
    mock_page = MagicMock()
    yes_delete = _make_locator(count=1, visible=True)

    def _locator_side_effect(selector):
        wrapper = MagicMock()
        if selector == _MODAL_CONFIRM_BUTTON_SELECTORS[0]:
            wrapper.first = yes_delete
        else:
            wrapper.first = _make_locator(count=0, visible=False)
        return wrapper

    mock_page.locator = MagicMock(side_effect=_locator_side_effect)
    mock_page.wait_for_selector = AsyncMock()
    mock_page.wait_for_load_state = AsyncMock()

    result = await resolve_confirmation_modal(mock_page)

    assert result is not None
    assert _MODAL_CONFIRM_BUTTON_SELECTORS[0] in result
    yes_delete.click.assert_awaited_once_with(force=True, timeout=_ELEMENT_ACTION_TIMEOUT_MS)
    mock_page.wait_for_selector.assert_awaited_once_with(
        _DIALOG_CONTAINER_SELECTOR, state="detached", timeout=_MODAL_DETACH_TIMEOUT_MS,
    )
    mock_page.wait_for_load_state.assert_awaited_once()


@pytest.mark.asyncio
async def test_resolve_confirmation_modal_falls_back_to_general_dialog_selector():
    """ปุ่ม OrangeHRM-specific ทุกตัวไม่เจอ (ไม่ใช่ OrangeHRM) แต่มี [role="dialog"] ปุ่ม
    "Confirm" ทั่วไป — ต้องเจอผ่าน fallback selector ตัวสุดท้ายในลิสต์"""
    mock_page = MagicMock()
    confirm_btn = _make_locator(count=1, visible=True)
    last_selector = _MODAL_CONFIRM_BUTTON_SELECTORS[-1]

    def _locator_side_effect(selector):
        wrapper = MagicMock()
        wrapper.first = confirm_btn if selector == last_selector else _make_locator(count=0, visible=False)
        return wrapper

    mock_page.locator = MagicMock(side_effect=_locator_side_effect)
    mock_page.wait_for_selector = AsyncMock()
    mock_page.wait_for_load_state = AsyncMock()

    result = await resolve_confirmation_modal(mock_page)

    assert result is not None
    assert last_selector in result
    confirm_btn.click.assert_awaited_once()


@pytest.mark.asyncio
async def test_resolve_confirmation_modal_returns_none_when_no_confirm_button_found():
    """ไม่เจอปุ่มยืนยันเลยสักตัว (ไม่ตรง selector ไหนในลิสต์เลย) -> คืน None เงียบๆ ไม่ throw
    ไม่เรียก wait_for_selector/wait_for_load_state เลย ปล่อยให้ LLM ตัดสินใจเองต่อในรอบถัดไป"""
    mock_page = MagicMock()
    wrapper = MagicMock()
    wrapper.first = _make_locator(count=0, visible=False)
    mock_page.locator = MagicMock(return_value=wrapper)
    mock_page.wait_for_selector = AsyncMock()
    mock_page.wait_for_load_state = AsyncMock()

    result = await resolve_confirmation_modal(mock_page)

    assert result is None
    mock_page.wait_for_selector.assert_not_awaited()
    mock_page.wait_for_load_state.assert_not_awaited()


@pytest.mark.asyncio
async def test_resolve_confirmation_modal_survives_detach_wait_timeout():
    """โมดัลบางตัวไม่ detach ออกจาก DOM จริง (แค่ซ่อนด้วย CSS) — wait_for_selector timeout
    ต้องไม่ทำให้ resolve_confirmation_modal() throw ออกไป ยังคืนข้อความสรุปสำเร็จตามปกติ"""
    mock_page = MagicMock()
    yes_delete = _make_locator(count=1, visible=True)

    def _locator_side_effect(selector):
        wrapper = MagicMock()
        wrapper.first = yes_delete if selector == _MODAL_CONFIRM_BUTTON_SELECTORS[0] else _make_locator(0, False)
        return wrapper

    mock_page.locator = MagicMock(side_effect=_locator_side_effect)
    mock_page.wait_for_selector = AsyncMock(side_effect=PWTimeout("still attached"))
    mock_page.wait_for_load_state = AsyncMock()

    result = await resolve_confirmation_modal(mock_page)

    assert result is not None
    mock_page.wait_for_load_state.assert_awaited_once()  # ยัง wait_stable ต่อแม้ detach-wait timeout


@pytest.mark.asyncio
async def test_resolve_confirmation_modal_retries_then_succeeds_without_reload(_no_real_sleep):
    """W24: รอบแรกคลิกแล้วโมดัลยังเปิดค้างอยู่จริง (ไม่ใช่ detach-wait timeout เฉยๆ) — retry
    รอบสองสำเร็จ ปิดโมดัลได้จริง -> ต้องไม่ไป reload หน้าเว็บเลย (reload คือ fallback สุดท้าย
    เท่านั้น ไม่ใช่ default behavior ตั้งแต่ retry แรกที่ยังไม่สำเร็จ)"""
    mock_page = MagicMock()
    yes_delete = _make_locator(count=1, visible=True)
    still_open_dialog = _make_locator(count=1, visible=True)

    def _locator_side_effect(selector):
        wrapper = MagicMock()
        if selector == _MODAL_CONFIRM_BUTTON_SELECTORS[0]:
            wrapper.first = yes_delete
        elif selector == _DIALOG_CONTAINER_SELECTOR:
            wrapper.first = still_open_dialog
        else:
            wrapper.first = _make_locator(count=0, visible=False)
        return wrapper

    mock_page.locator = MagicMock(side_effect=_locator_side_effect)
    # รอบแรก detach-wait timeout (โมดัลยังไม่ปิด) รอบสอง detach สำเร็จจริง
    mock_page.wait_for_selector = AsyncMock(side_effect=[PWTimeout("still attached"), None])
    mock_page.wait_for_load_state = AsyncMock()
    mock_page.reload = AsyncMock()

    result = await resolve_confirmation_modal(mock_page)

    assert result is not None
    assert "ไม่ตอบสนอง" not in result
    assert "ลองครั้งที่ 2" in result
    assert yes_delete.click.await_count == 2
    mock_page.reload.assert_not_awaited()


@pytest.mark.asyncio
async def test_resolve_confirmation_modal_reloads_page_after_button_unresponsive_for_all_retries(_no_real_sleep):
    """W24 (บั๊กจริงที่ user รายงาน — multi-batch operations): ปุ่มยืนยันไม่ตอบสนองจริงทุก
    ครั้งที่ลอง (โมดัลยังเปิดค้างอยู่แม้คลิกไปแล้ว _MODAL_CONFIRM_CLICK_RETRIES ครั้งเต็มโควตา)
    -> ต้อง page.reload() + รอ networkidle เป็น fallback สุดท้าย (จำลองพฤติกรรม "กด F5" ที่
    user ยืนยันว่าแก้ปัญหาได้จริงเวลาทำเอง) และข้อความคืนกลับต้องบอก LLM ให้รู้ว่าต้อง
    navigate/กรองข้อมูลใหม่เองต่อ"""
    mock_page = MagicMock()
    yes_delete = _make_locator(count=1, visible=True)
    still_open_dialog = _make_locator(count=1, visible=True)

    def _locator_side_effect(selector):
        wrapper = MagicMock()
        if selector == _MODAL_CONFIRM_BUTTON_SELECTORS[0]:
            wrapper.first = yes_delete
        elif selector == _DIALOG_CONTAINER_SELECTOR:
            wrapper.first = still_open_dialog
        else:
            wrapper.first = _make_locator(count=0, visible=False)
        return wrapper

    mock_page.locator = MagicMock(side_effect=_locator_side_effect)
    mock_page.wait_for_selector = AsyncMock(side_effect=PWTimeout("still attached"))
    mock_page.wait_for_load_state = AsyncMock()
    mock_page.reload = AsyncMock()

    result = await resolve_confirmation_modal(mock_page)

    assert result is not None
    assert "ไม่ตอบสนอง" in result
    assert "navigate" in result or "กรองข้อมูลใหม่" in result
    assert yes_delete.click.await_count == _MODAL_CONFIRM_CLICK_RETRIES
    mock_page.reload.assert_awaited_once()
    mock_page.wait_for_load_state.assert_awaited_once_with("networkidle", timeout=_MODAL_RELOAD_TIMEOUT_MS)


@pytest.mark.asyncio
async def test_resolve_confirmation_modal_reload_does_not_throw_if_reload_itself_fails(_no_real_sleep):
    """W24: page.reload() เองอาจ fail ได้ (เช่น network เพี้ยนชั่วคราว) — ต้องไม่ throw ออกไป
    ทำให้ action หลักที่เพิ่ง success (คลิก "Delete Selected") กลายเป็น fail ไปด้วย"""
    mock_page = MagicMock()
    yes_delete = _make_locator(count=1, visible=True)
    still_open_dialog = _make_locator(count=1, visible=True)

    def _locator_side_effect(selector):
        wrapper = MagicMock()
        if selector == _MODAL_CONFIRM_BUTTON_SELECTORS[0]:
            wrapper.first = yes_delete
        elif selector == _DIALOG_CONTAINER_SELECTOR:
            wrapper.first = still_open_dialog
        else:
            wrapper.first = _make_locator(count=0, visible=False)
        return wrapper

    mock_page.locator = MagicMock(side_effect=_locator_side_effect)
    mock_page.wait_for_selector = AsyncMock(side_effect=PWTimeout("still attached"))
    mock_page.reload = AsyncMock(side_effect=Exception("network hiccup"))
    mock_page.wait_for_load_state = AsyncMock()

    result = await resolve_confirmation_modal(mock_page)  # ต้องไม่ throw

    assert result is not None
    assert "ไม่ตอบสนอง" in result


@pytest.mark.asyncio
async def test_execute_click_auto_resolves_confirmation_modal_after_success():
    """W23 (บั๊กจริงที่ user รายงาน): คลิกสำเร็จแล้วเจอ confirmation modal เปิดขึ้นมา ต้อง
    resolve อัตโนมัติทันทีในระดับโค้ด (ไม่รอ LLM ตัดสินใจเรียก action แยกอีกรอบ) — ข้อความสรุป
    ต้องต่อท้าย message ของ action หลักให้เห็นแบบโปร่งใส"""
    mock_page = AsyncMock()
    with patch("backend.app.core.actions._detect_confirmation_modal", AsyncMock(return_value=True)), \
         patch(
             "backend.app.core.actions.resolve_confirmation_modal",
             AsyncMock(return_value=" [ตรวจพบ confirmation modal — กดยืนยันอัตโนมัติแล้ว (x)]"),
         ) as mock_resolve:
        result = await execute(mock_page, {"type": "click", "index": 5})

    assert result.success is True
    assert "ตรวจพบ confirmation modal" in result.message
    mock_resolve.assert_awaited_once()


@pytest.mark.asyncio
async def test_execute_click_skips_modal_resolution_when_no_modal_detected():
    mock_page = AsyncMock()
    with patch("backend.app.core.actions._detect_confirmation_modal", AsyncMock(return_value=False)) as mock_detect, \
         patch("backend.app.core.actions.resolve_confirmation_modal", AsyncMock()) as mock_resolve:
        result = await execute(mock_page, {"type": "click", "index": 5})

    assert result.success is True
    assert "ตรวจพบ confirmation modal" not in result.message
    mock_detect.assert_awaited_once()
    mock_resolve.assert_not_awaited()


@pytest.mark.asyncio
async def test_execute_delete_action_also_auto_resolves_confirmation_modal():
    """W23: submit/delete/purchase/pay dispatch ผ่าน _dispatch_click_with_retry() ตัวเดียวกับ
    plain click (ดู DEFAULT_NEEDS_CONFIRMATION branch ใน execute()) — modal handler ต้อง
    ทำงานเหมือนกันทุกประการ ไม่ใช่แค่ plain click"""
    mock_page = AsyncMock()
    ask_user_func = AsyncMock(return_value=True)
    with patch("backend.app.core.actions._detect_confirmation_modal", AsyncMock(return_value=True)), \
         patch(
             "backend.app.core.actions.resolve_confirmation_modal",
             AsyncMock(return_value=" [ตรวจพบ confirmation modal — กดยืนยันอัตโนมัติแล้ว (x)]"),
         ):
        result = await execute(mock_page, {"type": "delete", "index": 3}, ask_user_func=ask_user_func)

    assert result.success is True
    assert "ตรวจพบ confirmation modal" in result.message


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


# ---------------- W19: Deterministic State Filter short-circuits ก่อน dispatch จริง ----------------
# state_filter.py เองมีเทสต์ครบใน test_state_filter.py แล้ว — กลุ่มนี้เทสต์แค่ว่า execute()
# เรียกมันจริงและ short-circuit ตามผลลัพธ์ (ไม่แตะ page.click/fill/check/scroll เลยตอน
# redundant) ผ่าน patch ตรงจุดที่ actions.py import เข้ามา


@pytest.mark.asyncio
async def test_execute_fill_skips_dispatch_when_already_redundant():
    mock_page = AsyncMock()
    with patch(
        "backend.app.core.actions.state_filter.check_fill_redundant",
        AsyncMock(return_value="ช่องนี้มีข้อความอยู่แล้ว"),
    ):
        result = await execute(mock_page, {"type": "fill", "index": 0, "text": "standard_user"})

    assert result.success is True
    assert "[ข้าม]" in result.message
    mock_page.fill.assert_not_awaited()


@pytest.mark.asyncio
async def test_execute_check_skips_dispatch_when_already_redundant():
    mock_page = AsyncMock()
    with patch(
        "backend.app.core.actions.state_filter.check_checkbox_redundant",
        AsyncMock(return_value="ติ๊กอยู่แล้ว"),
    ):
        result = await execute(mock_page, {"type": "check", "index": 3})

    assert result.success is True
    assert "[ข้าม]" in result.message
    mock_page.check.assert_not_awaited()


@pytest.mark.asyncio
async def test_execute_scroll_skips_dispatch_when_already_at_edge():
    mock_page = AsyncMock()
    with patch(
        "backend.app.core.actions.state_filter.check_scroll_redundant",
        AsyncMock(return_value="เลื่อนหน้าจอถึงล่างสุดอยู่แล้ว"),
    ):
        result = await execute(mock_page, {"type": "scroll", "direction": "down"})

    assert result.success is True
    assert "[ข้าม]" in result.message
    mock_page.mouse.wheel.assert_not_awaited()


@pytest.mark.asyncio
async def test_execute_click_fails_without_dispatch_when_element_disabled():
    """ต่างจาก fill/check/scroll — click ที่ redundant เพราะ disabled ต้อง success=False
    (action ไม่ได้เกิดขึ้นจริง ไม่ใช่ "เป้าหมายบรรลุแล้ว")"""
    mock_page = AsyncMock()
    with patch(
        "backend.app.core.actions.state_filter.check_click_redundant",
        AsyncMock(return_value="element นี้อยู่ในสถานะ disabled แล้ว"),
    ):
        result = await execute(mock_page, {"type": "click", "index": 5})

    assert result.success is False
    assert "[ข้าม]" in result.message
    mock_page.click.assert_not_awaited()


@pytest.mark.asyncio
async def test_execute_fill_dispatches_normally_when_not_redundant():
    """sanity check: ไม่ redundant ต้อง dispatch ตามปกติทุกประการ (ไม่ใช่ short-circuit
    ทุกครั้งไม่ว่าผลจะเป็นยังไง)"""
    mock_page = AsyncMock()
    with patch(
        "backend.app.core.actions.state_filter.check_fill_redundant",
        AsyncMock(return_value=None),
    ):
        result = await execute(mock_page, {"type": "fill", "index": 0, "text": "standard_user"})

    assert result.success is True
    assert "[ข้าม]" not in result.message
    mock_page.fill.assert_awaited_once()


# ---------------- W40: execute() ต้องกด element ที่อยู่ใน <iframe> ได้จริง ----------------
# เทสต์กลุ่มนี้เปิด chromium จริง (ไม่ mock) — บั๊กที่ user รายงานคือ page.click(selector)
# หา element ข้าม frame boundary ไม่ได้เลย ต้องพิสูจน์กับ DOM จริงที่มี iframe จริง mock
# page ธรรมดาพิสูจน์เรื่องนี้ไม่ได้ (mock ไม่มี frame boundary ให้ล้มเหลวจริง)

_HTML_BUTTON_INSIDE_IFRAME = """
<html><body>
  <button id="main-btn">Main Button</button>
  <iframe srcdoc="<html><body><button id='inner-btn'>Inner Button</button></body></html>"></iframe>
</body></html>
"""


@pytest.mark.asyncio
async def test_execute_click_succeeds_on_button_inside_iframe():
    """W40: ปุ่มที่ perception.get_snapshot() เจอใน <iframe> ต้องกดผ่านได้จริงด้วย
    execute({"type": "click", ...}) — พิสูจน์ทั้ง 2 ฝั่งของ pipeline ทำงานร่วมกันจริง
    (perceive เห็น + dispatch กดได้ ไม่ใช่แค่เห็นแต่กดไม่ถึง)"""
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(_HTML_BUTTON_INSIDE_IFRAME)

        elements, _ = await get_snapshot(page)
        inner_btn = next(e for e in elements if e["label"] == "Inner Button")

        result = await execute(page, {"type": "click", "index": inner_btn["index"]})

        await browser.close()

    assert result.success is True


@pytest.mark.asyncio
async def test_execute_click_still_succeeds_on_main_frame_button_when_iframe_present():
    """ปุ่มในหน้าหลัก (ไม่ใช่ใน iframe) ต้องยังกดได้ตามปกติแม้หน้ามี iframe อื่นอยู่ด้วย —
    resolve_frame() ต้องลอง main frame ก่อนเสมอและเจอเลยไม่ต้องไล่ frame อื่นต่อ"""
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(_HTML_BUTTON_INSIDE_IFRAME)

        elements, _ = await get_snapshot(page)
        main_btn = next(e for e in elements if e["label"] == "Main Button")

        result = await execute(page, {"type": "click", "index": main_btn["index"]})

        await browser.close()

    assert result.success is True


# ---------------- W42: select_option() กับ option text ที่มี non-breaking space ----------------
# เทสต์กลุ่มนี้เปิด chromium จริง (ไม่ mock) — บั๊กที่ user รายงานคือ &nbsp; (U+00A0) ใน DOM
# จริง เทียบ mock string ธรรมดาพิสูจน์เรื่องนี้ไม่ได้ (mock ไม่มี whitespace encoding ให้
# ต่างจากที่พิมพ์เข้าไปเลย) ต้องใช้ <select> จริงที่ browser parse HTML entity ให้เอง

_HTML_SELECT_WITH_NBSP = """
<html><body>
  <select id="city">
    <option value="">Select a city</option>
    <option value="ny">New&nbsp;York</option>
    <option value="la">Los&nbsp;Angeles</option>
  </select>
</body></html>
"""


async def _select_and_read_value(html: str, index_finder, label: str):
    """เปิดหน้า/select ตาม index/กด select_option ด้วย label ที่กำหนด แล้วอ่านค่า .value
    ปัจจุบันของ <select> กลับมาด้วย — ใช้ร่วมกันในเทสต์ W42 ทุกตัวด้านล่าง"""
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(html)

        elements, _ = await get_snapshot(page)
        index = index_finder(elements)

        result = await execute(page, {"type": "select", "index": index, "label": label})
        selected_value = await page.input_value(f'[data-ai-index="{index}"]')

        await browser.close()

    return result, selected_value


@pytest.mark.asyncio
async def test_select_option_matches_nbsp_option_when_target_uses_regular_space():
    """W42: option text ใน DOM ใช้ &nbsp; คั่นคำ ("New\\u00a0York") แต่ label ที่ LLM ส่งมา
    ใช้ space ปกติ ("New York") — ต้อง select ได้ (เดิมจะ [FAIL] timeout เพราะเทียบ exact
    string ตรงๆ)"""
    result, selected_value = await _select_and_read_value(
        _HTML_SELECT_WITH_NBSP,
        lambda elements: next(e["index"] for e in elements if e["tag"] == "select"),
        "New York",
    )

    assert result.success is True
    assert selected_value == "ny"


@pytest.mark.asyncio
async def test_select_option_matches_regular_option_when_target_uses_nbsp():
    """W42: กลับกัน — DOM ใช้ space ปกติ แต่ label ที่ LLM ส่งมาดันมี &nbsp; ปนอยู่ (เช่น
    copy-paste มาจากที่อื่น) ก็ต้อง select ได้เหมือนกัน"""
    html = """
    <html><body>
      <select id="city">
        <option value="">Select a city</option>
        <option value="ny">New York</option>
      </select>
    </body></html>
    """
    result, selected_value = await _select_and_read_value(
        html,
        lambda elements: next(e["index"] for e in elements if e["tag"] == "select"),
        "New York",
    )

    assert result.success is True
    assert selected_value == "ny"


@pytest.mark.asyncio
async def test_select_option_matches_option_with_duplicate_and_trailing_whitespace():
    """W42: option text มี whitespace ซ้ำ/เว้นวรรคหัวท้ายเกิน (เช่นจาก HTML ที่จัด
    indentation ไม่เรียบร้อย) — ต้อง select ได้เมื่อ label ที่ LLM ส่งมาเป็นข้อความสะอาด"""
    html = """
    <html><body>
      <select id="city">
        <option value="">Select a city</option>
        <option value="ny">  New   York  </option>
      </select>
    </body></html>
    """
    result, selected_value = await _select_and_read_value(
        html,
        lambda elements: next(e["index"] for e in elements if e["tag"] == "select"),
        "New York",
    )

    assert result.success is True
    assert selected_value == "ny"


@pytest.mark.asyncio
async def test_select_option_fails_with_real_option_list_when_no_match_found():
    """W42: label ที่ไม่ตรงกับ option ไหนเลยแม้ normalize whitespace แล้ว ต้องคืน [FAIL]
    พร้อมแนบ list ตัวเลือกจริงที่มีอยู่ไปด้วย (ไม่ throw exception ออกมา)"""
    result, _ = await _select_and_read_value(
        _HTML_SELECT_WITH_NBSP,
        lambda elements: next(e["index"] for e in elements if e["tag"] == "select"),
        "Chicago",
    )

    assert result.success is False
    assert "New York" in result.message
    assert "Los Angeles" in result.message


@pytest.mark.asyncio
async def test_select_option_still_works_normally_for_plain_dropdown_without_nbsp():
    """W42: dropdown ปกติที่ไม่มี non-breaking space เลย ต้องยัง select ได้เหมือนเดิมทุก
    ประการ (regression check — ไม่กระทบ select ที่เคยผ่านอยู่แล้ว)"""
    html = """
    <html><body>
      <select id="language">
        <option value="">Select</option>
        <option value="py">Python</option>
        <option value="js">JavaScript</option>
      </select>
    </body></html>
    """
    result, selected_value = await _select_and_read_value(
        html,
        lambda elements: next(e["index"] for e in elements if e["tag"] == "select"),
        "JavaScript",
    )

    assert result.success is True
    assert selected_value == "js"


# ---------------- read_page_data: อ่านเนื้อหาหน้าเว็บ (Lane 1 นับ / Lane 2 อ่านตาราง) ----------------
# mock count_elements()/extract_table_data() ตรงๆ (ทั้งคู่เทสต์ครบแล้วใน test_perception.py
# ด้วย chromium จริง) — ที่นี่สนใจแค่ dispatch/lane-selection logic ของ execute() เอง


@pytest.mark.asyncio
async def test_execute_read_page_data_favors_count_for_counting_query():
    mock_page = AsyncMock()
    with patch("backend.app.core.actions.count_elements", AsyncMock(return_value=5)) as mock_count, \
         patch("backend.app.core.actions.extract_table_data", AsyncMock()) as mock_extract:
        result = await execute(
            mock_page,
            {"type": "read_page_data", "query": "มีสินค้ากี่ชิ้น", "target_hint": ".inventory_item"},
        )

    assert result.success is True
    assert "5" in result.message
    mock_count.assert_awaited_once_with(mock_page, ".inventory_item")
    mock_extract.assert_not_awaited()


@pytest.mark.asyncio
async def test_execute_read_page_data_uses_extract_table_data_for_non_counting_query():
    mock_page = AsyncMock()
    with patch("backend.app.core.actions.count_elements", AsyncMock()) as mock_count, \
         patch(
             "backend.app.core.actions.extract_table_data",
             AsyncMock(return_value="| Name |\n| --- |\n| Widget |"),
         ) as mock_extract:
        result = await execute(
            mock_page,
            {"type": "read_page_data", "query": "สรุปตารางสินค้าให้หน่อย", "target_hint": "#products"},
        )

    assert result.success is True
    assert "Widget" in result.message
    mock_extract.assert_awaited_once_with(mock_page, "#products", "สรุปตารางสินค้าให้หน่อย")
    mock_count.assert_not_awaited()


@pytest.mark.asyncio
async def test_execute_read_page_data_fails_without_target_hint():
    mock_page = AsyncMock()

    result = await execute(mock_page, {"type": "read_page_data", "query": "มีสินค้ากี่ชิ้น"})

    assert result.success is False


@pytest.mark.asyncio
async def test_execute_read_page_data_reports_failure_from_extract_table_data():
    mock_page = AsyncMock()
    with patch(
        "backend.app.core.actions.extract_table_data",
        AsyncMock(return_value="[FAIL] ไม่พบ element ที่ตรงกับ '#missing'"),
    ):
        result = await execute(
            mock_page, {"type": "read_page_data", "query": "สรุปให้หน่อย", "target_hint": "#missing"}
        )

    assert result.success is False
    assert "#missing" in result.message


@pytest.mark.asyncio
async def test_execute_read_page_data_waits_for_page_to_settle_before_reading():
    """Retry on State Change: ถ้าเพิ่ง edit/submit ข้อมูลในตารางไปเมื่อ step ก่อนหน้า DOM อาจ
    ยัง bind ค่าใหม่ไม่เสร็จตอน read_page_data ถูกเรียกตามมาติดๆ — ต้องรอหน้านิ่ง (wait_stable)
    ก่อนอ่านเสมอ ไม่ใช่อ่านทันทีแล้วอาจได้ข้อมูลเก่า/ว่างเปล่า"""
    mock_page = AsyncMock()
    with patch("backend.app.core.actions.wait_stable", AsyncMock()) as mock_wait_stable, \
         patch("backend.app.core.actions.extract_table_data", AsyncMock(return_value="| Name |\n| --- |\n| Widget |")):
        await execute(
            mock_page,
            {"type": "read_page_data", "query": "สรุปตารางสินค้าให้หน่อย", "target_hint": "#products"},
        )

    mock_wait_stable.assert_awaited_once_with(mock_page)


@pytest.mark.asyncio
async def test_execute_read_page_data_does_not_retry_on_failure(_no_real_sleep):
    """read_page_data ไม่ผ่าน _dispatch_with_retry เหมือน click/fill — target_hint ที่หา
    ไม่เจอเป็น deterministic mismatch ไม่ใช่ DOM-timing issue ที่ retry แล้วจะเปลี่ยนผล"""
    mock_page = AsyncMock()
    with patch(
        "backend.app.core.actions.extract_table_data",
        AsyncMock(return_value="[FAIL] ไม่พบ element ที่ตรงกับ '#missing'"),
    ) as mock_extract:
        await execute(mock_page, {"type": "read_page_data", "query": "สรุปให้หน่อย", "target_hint": "#missing"})

    assert mock_extract.await_count == 1
    _no_real_sleep.assert_not_awaited()

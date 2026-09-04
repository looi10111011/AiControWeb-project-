from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from playwright.async_api import TimeoutError as PWTimeout, async_playwright

from backend.app.core.actions import (
    _ACTION_RETRIES,
    ActionResult,
    _DIALOG_CONTAINER_SELECTOR,
    _DIALOG_CONTAINER_SELECTORS,
    _ELEMENT_ACTION_TIMEOUT_MS,
    _MODAL_CONFIRM_BUTTON_SELECTORS,
    _MODAL_CONFIRM_CLICK_RETRIES,
    _MODAL_DETACH_TIMEOUT_MS,
    _MODAL_RELOAD_TIMEOUT_MS,
    _SUCCESS_TOAST_SELECTOR,
    _deterministic_count_note,
    system_counted_conditions,
    _dispatch_click_with_retry,
    _detect_confirmation_modal,
    _detect_success_toast,
    _extracted_entries,
    execute,
    fill_secret,
    resolve_confirmation_modal,
)
from backend.app.core import actions
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
    assert "attempt 2/2" in result.message
    assert mock_page.click.await_count == 2
    _no_real_sleep.assert_awaited_once()  # หน่วงแค่ระหว่างครั้งที่ 1->2 ครั้งเดียว


@pytest.mark.asyncio
async def test_execute_fill_gives_up_after_max_retries(_no_real_sleep):
    mock_page = AsyncMock()
    mock_page.fill = AsyncMock(side_effect=PWTimeout("still not there"))

    result = await execute(mock_page, {"type": "fill", "index": 0, "text": "hello"})

    assert result.success is False
    assert "after 2 attempts" in result.message
    assert mock_page.fill.await_count == 2
    assert _no_real_sleep.await_count == 1  # หน่วงระหว่างแต่ละครั้ง ไม่หน่วงหลังครั้งสุดท้าย


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


# ---------------- W_chain ("Compound Actions") — fill+key, click/select/check+then_click_index ----------------


@pytest.mark.asyncio
async def test_execute_fill_with_key_chains_press_key_on_success():
    mock_page = AsyncMock()

    result = await execute(mock_page, {"type": "fill", "index": 0, "text": "hello", "key": "Enter"})

    assert result.success is True
    assert "press_key" in result.message
    mock_page.fill.assert_awaited_once_with('[data-ai-index="0"]', "hello", timeout=_ELEMENT_ACTION_TIMEOUT_MS)
    # ต้องเป็นการกด key จริง (ครั้งที่ 3 ต่อจาก Ctrl+A/Backspace ที่ fill() ทำเองอยู่แล้ว)
    mock_page.press.assert_any_call('[data-ai-index="0"]', "Enter", timeout=_ELEMENT_ACTION_TIMEOUT_MS)
    assert mock_page.press.await_count == 3


@pytest.mark.asyncio
async def test_execute_fill_without_key_does_not_press_extra_key():
    mock_page = AsyncMock()

    result = await execute(mock_page, {"type": "fill", "index": 0, "text": "hello"})

    assert result.success is True
    assert "press_key" not in result.message
    # แค่ Ctrl+A + Backspace ของ fill() เอง ไม่มีการกด key เพิ่ม
    assert mock_page.press.await_count == 2


def _make_fill_mock_page():
    """W_datepicker: fill() ตอนนี้เรียก target.locator(selector).evaluate(...) หลัง
    .fill() สำเร็จเสมอ (dismiss popup ที่อาจเปิดจาก focus) — เหมือน _make_select_mock_page
    ด้านบน ต้อง mock .locator() แบบ sync (MagicMock) ไม่ใช่ AsyncMock ทั้งก้อน ไม่งั้นเรียก
    .locator(selector) จะได้ coroutine กลับมาแทน Locator object จริง"""
    mock_page = AsyncMock()
    dismiss_locator = MagicMock()
    dismiss_locator.evaluate = AsyncMock()
    mock_page.locator = MagicMock(return_value=dismiss_locator)
    return mock_page, dismiss_locator


@pytest.mark.asyncio
async def test_execute_fill_dismisses_any_popup_opened_by_focus_after_success():
    """W_datepicker follow-up — false positive จริงที่เจอ: fill(From Date) บน OrangeHRM
    Leave List เปิด date-picker popup เป็นผลข้างเคียงของ focus (framework เอง ไม่ใช่
    intentional) popup แทรก element ใหม่เข้า DOM ทำให้ index ของ "To Date" เลื่อนหนี ถ้า
    ไม่ปิด popup ทันที agent (หรือ compound action อื่นในคำสั่งเดียวกัน) จะอ้าง index ผิด
    ไปกดปุ่มนำทางปฏิทินแทนช่องกรอกจริงเงียบๆ — ยืนยันแล้วว่าต้อง blur()+คลิก body จริง
    (Escape เพียงอย่างเดียวไม่ปิด popup นี้ เพราะ framework ผูก listener กับ outside-click)"""
    mock_page, dismiss_locator = _make_fill_mock_page()

    result = await execute(mock_page, {"type": "fill", "index": 0, "text": "2026-15-05"})

    assert result.success is True
    # W_file_input_guard: mock page คืน locator ตัวเดียวกันให้ทุก query — guard ที่เช็คว่า
    # เป้าหมายเป็น <input type=file> ไหม ก็เรียก evaluate() ผ่าน locator ตัวนี้ด้วย จำนวนครั้ง
    # จึงไม่ใช่ 1 อีกต่อไป สิ่งที่เทสต์นี้สนใจจริงๆ คือ "blur+body click ถูกยิงจริงหลัง fill"
    dismiss_locator.evaluate.assert_any_await(
        "el => { el.blur(); document.body.click(); }", timeout=500,
    )
    mock_page.wait_for_timeout.assert_awaited_once_with(200)


@pytest.mark.asyncio
async def test_execute_fill_still_succeeds_when_popup_dismissal_itself_fails():
    """dismiss เป็นแค่ best-effort cleanup — ต้องไม่ทำให้ fill() ที่สำเร็จไปแล้วกลายเป็น
    fail เพราะขั้นตอนเสริมนี้พัง (เช่น element หลุดจาก DOM ไปแล้วหลัง fill)"""
    mock_page, dismiss_locator = _make_fill_mock_page()
    dismiss_locator.evaluate = AsyncMock(side_effect=Exception("detached"))

    result = await execute(mock_page, {"type": "fill", "index": 0, "text": "hello"})

    assert result.success is True
    assert "hello" in result.message


@pytest.mark.asyncio
async def test_execute_fill_does_not_chain_key_when_fill_itself_fails():
    mock_page = AsyncMock()
    mock_page.fill = AsyncMock(side_effect=Exception("boom"))

    result = await execute(mock_page, {"type": "fill", "index": 0, "text": "hello", "key": "Enter"})

    assert result.success is False
    # Ctrl+A/Backspace เกิดก่อน fill() throw ได้ (2 ครั้ง) แต่ต้องไม่มีการกด "Enter" เพิ่มเลย
    enter_calls = [c for c in mock_page.press.await_args_list if c.args[1:2] == ("Enter",)]
    assert enter_calls == []


@pytest.mark.asyncio
async def test_execute_fill_chains_then_click_index_when_no_key_given():
    """W_chain follow-up — false positive จริงที่เจอ: MiniWoB "enter-text" ไม่มี
    Enter-to-submit เลย (input ไม่ได้อยู่ใน <form>, ไม่มี keypress listener) ทำให้
    fill+key:"Enter" กรอกค่าถูกต้องแต่ไม่เคย submit จริง (ปุ่ม Submit ไม่เคยถูกคลิก) —
    then_click_index ให้ fill "คลิกปุ่มจริง" แทนได้ในคำสั่งเดียวกัน เชื่อถือได้กว่า
    key:"Enter" เสมอเมื่อเห็นปุ่ม submit จริงในหน้า"""
    mock_page = AsyncMock()

    result = await execute(mock_page, {"type": "fill", "index": 0, "text": "Livia", "then_click_index": 1})

    assert result.success is True
    assert "then click(1)" in result.message
    mock_page.fill.assert_awaited_once_with('[data-ai-index="0"]', "Livia", timeout=_ELEMENT_ACTION_TIMEOUT_MS)
    # fill() เองเรียก .click() บน index 0 เพื่อ focus ก่อน clear อยู่แล้ว (1 ครั้ง) บวกกับ
    # click ที่ chain ไปยัง index 1 อีก 1 ครั้ง — เช็คว่ามีการคลิก index 1 เกิดขึ้นจริง
    # แยกต่างหาก (ไม่สนใจ call ของ index 0 ที่เป็นผลข้างเคียงปกติของ fill())
    click_selectors = [c.args[0] for c in mock_page.click.await_args_list]
    assert click_selectors == ['[data-ai-index="0"]', '[data-ai-index="1"]']


@pytest.mark.asyncio
async def test_execute_fill_can_chain_both_key_and_then_click_index_together():
    mock_page = AsyncMock()

    result = await execute(
        mock_page, {"type": "fill", "index": 0, "text": "hi", "key": "Enter", "then_click_index": 1},
    )

    assert result.success is True
    assert "press_key" in result.message
    assert "then click(1)" in result.message
    click_selectors = [c.args[0] for c in mock_page.click.await_args_list]
    assert click_selectors == ['[data-ai-index="0"]', '[data-ai-index="1"]']


@pytest.mark.asyncio
async def test_execute_fill_does_not_chain_click_when_fill_itself_fails():
    mock_page = AsyncMock()
    mock_page.fill = AsyncMock(side_effect=Exception("boom"))

    result = await execute(mock_page, {"type": "fill", "index": 0, "text": "hi", "then_click_index": 1})

    assert result.success is False
    # fill() เองเรียก .click() บน index 0 เพื่อ focus ก่อน clear (ปกติ) แต่ไม่มีทางคลิก
    # index 1 (then_click_index) เลยเพราะ fill ล้มเหลวก่อนถึงจุดนั้น
    click_selectors = {c.args[0] for c in mock_page.click.await_args_list}
    assert '[data-ai-index="1"]' not in click_selectors


@pytest.mark.asyncio
async def test_execute_click_chains_then_click_index_when_provided():
    mock_page = AsyncMock()

    result = await execute(mock_page, {"type": "click", "index": 0, "then_click_index": 5})

    assert result.success is True
    assert "then click(5)" in result.message
    click_selectors = [c.args[0] for c in mock_page.click.await_args_list]
    assert click_selectors == ['[data-ai-index="0"]', '[data-ai-index="5"]']


@pytest.mark.asyncio
async def test_execute_click_reports_success_when_only_the_chained_click_fails():
    """W_chain_partial_success (บั๊กจริง live-reproduce บน OrangeHRM กับ provider openai):
    primary คลิกสำเร็จและเปลี่ยนหน้าไปแล้วจริง แต่ chain ตัวที่สองพัง (index ค้างจาก snapshot
    ก่อนหน้า เพราะหน้าเพิ่งเปลี่ยนไปนั่นแหละ) — เดิมคืน success=False ทั้งก้อน ทำให้โมเดลอ่านว่า
    ล้มเหลวแล้ว "คลิก primary ซ้ำ" วนอยู่ 10 step ติดกันจนหมด max_steps ทั้งที่ไปถึงหน้า
    เป้าหมายตั้งแต่ step แรก — ต้องคืนความจริง: primary สำเร็จ, บอกให้สั่งคลิกที่สองแยก step"""
    mock_page = AsyncMock()
    call_count = {"n": 0}

    async def _click(selector, **kwargs):
        call_count["n"] += 1
        if selector == '[data-ai-index="5"]':
            raise Exception("element not found")

    mock_page.click = AsyncMock(side_effect=_click)

    result = await execute(mock_page, {"type": "click", "index": 0, "then_click_index": 5})

    assert result.success is True  # ความคืบหน้าจริงของ primary ต้องไม่หายไป
    assert "chained click(5) failed" in result.message
    assert "separate next step" in result.message
    assert "do not repeat the first action" in result.message


@pytest.mark.asyncio
async def test_execute_click_does_not_chain_when_primary_action_fails():
    mock_page = AsyncMock()
    mock_page.click = AsyncMock(side_effect=Exception("boom"))

    result = await execute(mock_page, {"type": "click", "index": 0, "then_click_index": 5})

    assert result.success is False
    assert "then click" not in result.message
    # ยังพยายามคลิก index 0 (retry ตามปกติ) แต่ไม่มีทางคลิก index 5 เลยเพราะ primary fail
    click_selectors = {c.args[0] for c in mock_page.click.await_args_list}
    assert '[data-ai-index="5"]' not in click_selectors


@pytest.mark.asyncio
async def test_execute_click_without_then_click_index_behaves_exactly_as_before():
    mock_page = AsyncMock()

    result = await execute(mock_page, {"type": "click", "index": 0})

    assert result.success is True
    mock_page.click.assert_awaited_once()


@pytest.mark.asyncio
async def test_execute_click_chain_skipped_when_secondary_needs_confirmation_and_user_rejects():
    """then_click_index ที่ label เข้าข่ายเสี่ยง (เช่น "Delete") ต้องผ่าน permission check
    เต็มรูปแบบเหมือน action เดี่ยวๆ ทุกประการ — ห้ามข้าม human-in-the-loop เด็ดขาดแค่เพราะ
    เป็น action ที่สองในคำสั่งเดียวกัน ผลลัพธ์ของ primary action ต้องไม่หายไปด้วย"""
    mock_page = AsyncMock()
    ask_user_func = AsyncMock(return_value=False)  # user ปฏิเสธ

    result = await execute(
        mock_page, {"type": "click", "index": 0, "then_click_index": 5},
        ask_user_func=ask_user_func, then_label="Delete",
    )

    assert result.success is True  # primary (index 0) ยังสำเร็จอยู่
    assert "did not go on to click" in result.message
    ask_user_func.assert_awaited_once()
    click_selectors = {c.args[0] for c in mock_page.click.await_args_list}
    assert '[data-ai-index="5"]' not in click_selectors  # ไม่เคยคลิกจริง


@pytest.mark.asyncio
async def test_execute_click_chain_proceeds_when_secondary_needs_confirmation_and_user_approves():
    mock_page = AsyncMock()
    ask_user_func = AsyncMock(return_value=True)  # user อนุมัติ

    result = await execute(
        mock_page, {"type": "click", "index": 0, "then_click_index": 5},
        ask_user_func=ask_user_func, then_label="Delete",
    )

    assert result.success is True
    assert "then click(5)" in result.message
    click_selectors = [c.args[0] for c in mock_page.click.await_args_list]
    assert click_selectors == ['[data-ai-index="0"]', '[data-ai-index="5"]']


@pytest.mark.asyncio
async def test_execute_check_chains_then_click_index():
    mock_page = AsyncMock()

    result = await execute(mock_page, {"type": "check", "index": 2, "then_click_index": 7})

    assert result.success is True
    assert "then click(7)" in result.message
    mock_page.click.assert_awaited_once_with('[data-ai-index="7"]', timeout=_ELEMENT_ACTION_TIMEOUT_MS)


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
    assert mock_page.click.await_count == _ACTION_RETRIES
    # hover ก่อนทุกรอบตั้งแต่รอบ 2 เป็นต้นไป = จำนวนรอบทั้งหมด - 1 (ไม่ hover ก่อนรอบแรก)
    assert mock_page.hover.await_count == _ACTION_RETRIES - 1
    # กรองเอาแค่ click/hover (ตัด query_selector ที่ resolve_frame() เรียกแทรกก่อนทุกครั้งออก)
    call_order = [c[0] for c in mock_page.method_calls if c[0] in ("click", "hover")]
    assert call_order == ["click"] + ["hover", "click"] * (_ACTION_RETRIES - 1)


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


# --- W_dialog_generic / W_modal_appear_race (C2+C3 จาก audit ของ P7/P8) ---
# เทสต์กลุ่มนี้ใช้ Chromium จริง (ไฟล์นี้ import async_playwright ไว้แล้ว) เพราะสิ่งที่ต้อง
# พิสูจน์คือ "selector ตรงกับ DOM จริงไหม" ซึ่ง mock พิสูจน์ไม่ได้เลย — mock ที่ตอบว่าเจอ
# ก็จะเจอเสมอไม่ว่า selector จะผิดแค่ไหน


@pytest.mark.asyncio
async def test_detect_confirmation_modal_finds_dialogs_that_appear_after_a_delay():
    """บั๊กจริงที่ user เจอ: dialog animate เข้ามาหลัง click ตอนเช็คยังไม่อยู่ใน DOM
    -> สรุปว่า "ไม่มีโมดัล" -> โมดัลโผล่มาบังทั้งหน้า -> ทุก click ถัดไป fail -> ไม่มีทาง
    กลับมาถึงจุดเช็คอีกเลย (จุดเรียกเดียวอยู่ใต้ if result.success) = ติด loop จนต้องกด Stop

    3 เคสนี้คือ modal ที่ selector ชุดเดิมตรวจไม่เจอเลยสักตัว — โดยเฉพาะ Bootstrap ที่
    markup อยู่ใน DOM อยู่แล้วและเปิดด้วยการสลับ class ซึ่งทำให้แนวคิดเดิมที่จะกรองด้วย
    ความยาว body.innerHTML ใช้ไม่ได้เลย (ความยาวไม่เปลี่ยนสักตัวอักษร)"""
    cases = {
        "bootstrap class toggle": (
            "<button id='go' onclick=\"setTimeout(()=>"
            "document.getElementById('m').classList.add('show'),120)\">Delete</button>"
            "<style>.modal{display:none}.modal.show{display:block}</style>"
            "<div id='m' class='modal'><button>No, Cancel</button>"
            "<button>Yes, Delete</button></div>"
        ),
        "native <dialog>": (
            "<button id='go' onclick=\"setTimeout(()=>"
            "document.getElementById('m').showModal(),120)\">Delete</button>"
            "<dialog id='m'><button>No, Cancel</button>"
            "<button>Yes, Delete</button></dialog>"
        ),
        "aria-modal without role": (
            "<button id='go' onclick=\"setTimeout(()=>"
            "document.getElementById('m').hidden=false,120)\">Delete</button>"
            "<div id='m' aria-modal='true' hidden><button>ตกลง</button></div>"
        ),
    }

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        try:
            for name, html in cases.items():
                await page.set_content(html)
                await page.click("#go")
                assert await _detect_confirmation_modal(page) is True, name
                # เจอแล้วต้องกดปุ่มยืนยันในนั้นได้จริงด้วย — ตรวจเจอแต่หาปุ่มไม่เจอ
                # แย่กว่าไม่ตรวจเจอตั้งแต่แรก เพราะ agent จะค้างอยู่หน้าโมดัลเหมือนเดิม
                assert await resolve_confirmation_modal(page) is not None, name
                assert await page.locator("#m").is_visible() is False, name
        finally:
            await browser.close()


@pytest.mark.asyncio
async def test_detect_confirmation_modal_stays_false_on_a_page_with_no_dialog():
    """ราคาที่จ่ายจากการรอต้องไม่แลกมาด้วย false positive — หน้าที่ไม่มี dialog เลยต้อง
    ตอบ False เสมอ ไม่ว่าจะรอนานแค่ไหน"""
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        try:
            await page.set_content("<button>Just a button</button><div>Are you sure?</div>")
            assert await _detect_confirmation_modal(page) is False
        finally:
            await browser.close()


def test_generic_confirm_button_selectors_are_scoped_to_every_known_container():
    """ตรวจเจอ dialog ของ framework ใหม่ได้ แต่หาปุ่มยืนยันในนั้นไม่เจอ = ยังค้างเหมือนเดิม
    ทั้งสองลิสต์จึงต้องมาจากชุด container เดียวกันเสมอ ไม่ hardcode แยกกัน"""
    joined = " ".join(_MODAL_CONFIRM_BUTTON_SELECTORS)

    for container in _DIALOG_CONTAINER_SELECTORS:
        assert container in _DIALOG_CONTAINER_SELECTOR, container
        assert f'{container} button:has-text(' in joined, container

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
    assert "was unresponsive" not in result
    assert "attempt 2" in result
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
    assert "was unresponsive" in result
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
    assert "was unresponsive" in result


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


# W63[7.1] ("Save Confirmation & Toast Wait", ticket Issue 7.1)


@pytest.mark.asyncio
async def test_detect_success_toast_returns_text_when_visible():
    mock_page = MagicMock()
    toast = MagicMock()
    toast.wait_for = AsyncMock()
    toast.inner_text = AsyncMock(return_value="Successfully Saved")
    wrapper = MagicMock()
    wrapper.first = toast
    mock_page.locator = MagicMock(return_value=wrapper)

    result = await _detect_success_toast(mock_page)

    assert result == "Successfully Saved"
    mock_page.locator.assert_called_once_with(_SUCCESS_TOAST_SELECTOR)


@pytest.mark.asyncio
async def test_detect_success_toast_returns_none_when_not_visible_in_time():
    mock_page = MagicMock()
    toast = MagicMock()
    toast.wait_for = AsyncMock(side_effect=PWTimeout("timeout"))
    wrapper = MagicMock()
    wrapper.first = toast
    mock_page.locator = MagicMock(return_value=wrapper)

    result = await _detect_success_toast(mock_page)

    assert result is None


@pytest.mark.asyncio
async def test_detect_success_toast_fails_safe_on_bare_mock_page():
    result = await _detect_success_toast(AsyncMock())

    assert result is None


# click ที่ label เป็นปุ่ม Save/Submit/Confirm ต้องเช็ค success toast อัตโนมัติหลังคลิกสำเร็จ —
# mirror รูปแบบเทสต์เดียวกับ confirmation-modal ด้านบนทุกประการ (patch
# _detect_confirmation_modal ให้ False เสมอกันชนกับ flow modal ที่คนละ elif กัน)


@pytest.mark.asyncio
async def test_execute_click_checks_toast_when_label_matches_save():
    mock_page = AsyncMock()
    with patch("backend.app.core.actions._detect_confirmation_modal", AsyncMock(return_value=False)), \
         patch(
             "backend.app.core.actions._detect_success_toast",
             AsyncMock(return_value="Successfully Saved"),
         ) as mock_toast:
        result = await execute(mock_page, {"type": "click", "index": 5}, label="Save")

    assert result.success is True
    assert 'Success confirmation found: "Successfully Saved"' in result.message
    assert result.toast_confirmed is True  # W64[7.2]
    mock_toast.assert_awaited_once()


@pytest.mark.asyncio
async def test_execute_click_notes_missing_toast_when_label_matches_save():
    mock_page = AsyncMock()
    with patch("backend.app.core.actions._detect_confirmation_modal", AsyncMock(return_value=False)), \
         patch("backend.app.core.actions._detect_success_toast", AsyncMock(return_value=None)):
        result = await execute(mock_page, {"type": "click", "index": 5}, label="บันทึก")

    assert result.success is True
    assert "No toast/success confirmation appeared" in result.message
    assert result.toast_confirmed is False  # W64[7.2]


@pytest.mark.asyncio
async def test_execute_click_skips_toast_check_for_non_save_label():
    mock_page = AsyncMock()
    with patch("backend.app.core.actions._detect_confirmation_modal", AsyncMock(return_value=False)), \
         patch("backend.app.core.actions._detect_success_toast", AsyncMock()) as mock_toast:
        result = await execute(mock_page, {"type": "click", "index": 5}, label="Next Page")

    assert result.success is True
    assert "toast" not in result.message
    assert result.toast_confirmed is False  # W64[7.2]: default เมื่อไม่ได้เช็ค toast เลย
    mock_toast.assert_not_awaited()


@pytest.mark.asyncio
async def test_execute_click_prefers_modal_over_toast_when_both_apply():
    """label ตรงกับ Save คำเดียวกับที่ modal อาจใช้ (เช่น "Confirm") แต่ modal ถูก detect ก่อน
    -> ต้องไม่เรียก toast check ซ้ำ (elif กันชนกัน คนละ flow)"""
    mock_page = AsyncMock()
    with patch("backend.app.core.actions._detect_confirmation_modal", AsyncMock(return_value=True)), \
         patch(
             "backend.app.core.actions.resolve_confirmation_modal",
             AsyncMock(return_value=" [ตรวจพบ confirmation modal — กดยืนยันอัตโนมัติแล้ว (x)]"),
         ), \
         patch("backend.app.core.actions._detect_success_toast", AsyncMock()) as mock_toast:
        result = await execute(
            mock_page, {"type": "click", "index": 5}, label="Confirm",
            ask_user_func=AsyncMock(return_value=True),
        )

    assert "ตรวจพบ confirmation modal" in result.message
    mock_toast.assert_not_awaited()


@pytest.mark.asyncio
async def test_execute_submit_action_preserves_toast_confirmed():
    """W64[7.2]: type="submit" (DEFAULT_NEEDS_CONFIRMATION) dispatch ผ่าน
    _dispatch_click_with_retry() ตัวเดียวกับ plain click แล้ว re-wrap ActionResult ใหม่ —
    ต้องคง toast_confirmed (และ locator_descriptor) จาก result เดิมไว้ด้วย ไม่ทิ้งไปเงียบๆ
    เหมือนที่เคยเป็นก่อนแก้ (bug จริงที่เจอระหว่างเขียนฟีเจอร์นี้)"""
    mock_page = AsyncMock()
    ask_user_func = AsyncMock(return_value=True)
    with patch("backend.app.core.actions._detect_confirmation_modal", AsyncMock(return_value=False)), \
         patch(
             "backend.app.core.actions._detect_success_toast",
             AsyncMock(return_value="Successfully Saved"),
         ):
        result = await execute(
            mock_page, {"type": "submit", "index": 3}, label="Save", ask_user_func=ask_user_func,
        )

    assert result.success is True
    assert result.toast_confirmed is True
    assert 'Success confirmation found: "Successfully Saved"' in result.message


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
    assert "there are only 1 tab" in result.message


@pytest.mark.asyncio
async def test_execute_does_not_retry_switch_tab_on_failure():
    """switch_tab ไม่ผ่าน _dispatch_with_retry (เหมือน goto/scroll/go_back/wait) —
    fail แล้วต้อง fail ทันทีไม่ retry"""
    mock_page = AsyncMock()
    mock_page.context.pages = []  # ไม่มี tab ให้สลับเลย

    result = await execute(mock_page, {"type": "switch_tab", "tab_index": 0})

    assert result.success is False
    assert "there are only 0 tab" in result.message


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
    assert "[Skipped]" in result.message
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
    assert "[Skipped]" in result.message
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
    assert "[Skipped]" in result.message
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
    assert "[Skipped]" in result.message
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
    assert "[Skipped]" not in result.message
    mock_page.fill.assert_awaited_once()


# ---------------- W65[3] ("Vault Expansion — Current Password Auto-fill") ----------------
# ค่าลับต้องไม่หลุดเข้า ActionResult.message เด็ดขาด (เทสต์สำคัญที่สุดของฟีเจอร์นี้) — mock
# site_learning.storage.load_credentials ที่ fill_secret() lazy-import มา (ดู comment ใน
# actions.py ว่าทำไมต้อง lazy import — กัน circular import กับ site_learning/crawler.py)


@pytest.mark.asyncio
async def test_fill_secret_fills_stored_password_without_leaking_it_in_message():
    mock_page = AsyncMock()
    mock_page.url = "https://demo.example.com/pim/changePasswordSave"
    with patch(
        "backend.app.site_learning.storage.load_credentials",
        return_value={"username": "admin", "password": "s3cr3t-real-password"},
    ) as mock_load:
        result = await fill_secret(mock_page, 4, "current_password")

    assert result.success is True
    assert "s3cr3t-real-password" not in result.message
    assert "s3cr3t-real-password" not in str(result)
    mock_load.assert_called_once_with("demo.example.com")
    mock_page.fill.assert_awaited_once_with(
        '[data-ai-index="4"]', "s3cr3t-real-password", timeout=_ELEMENT_ACTION_TIMEOUT_MS,
    )


@pytest.mark.asyncio
async def test_fill_secret_fails_gracefully_when_no_credential_stored():
    mock_page = AsyncMock()
    mock_page.url = "https://demo.example.com/pim/changePasswordSave"
    with patch("backend.app.site_learning.storage.load_credentials", return_value=None):
        result = await fill_secret(mock_page, 4, "current_password")

    assert result.success is False
    assert "no credential saved for this site" in result.message
    mock_page.fill.assert_not_awaited()


@pytest.mark.asyncio
async def test_fill_secret_rejects_unknown_secret_key():
    mock_page = AsyncMock()
    mock_page.url = "https://demo.example.com/pim/changePasswordSave"
    with patch("backend.app.site_learning.storage.load_credentials") as mock_load:
        result = await fill_secret(mock_page, 4, "new_password")

    assert result.success is False
    assert "unknown secret_key" in result.message
    mock_load.assert_not_called()  # ไม่ต้องเสีย I/O เรียก vault เลยถ้า secret_key ไม่รู้จักตั้งแต่แรก
    mock_page.fill.assert_not_awaited()


@pytest.mark.asyncio
async def test_execute_dispatches_fill_secret_type_to_fill_secret_function():
    mock_page = AsyncMock()
    mock_page.url = "https://demo.example.com/pim/changePasswordSave"
    with patch(
        "backend.app.site_learning.storage.load_credentials",
        return_value={"username": "admin", "password": "hunter2"},
    ):
        result = await execute(mock_page, {"type": "fill_secret", "index": 4, "secret": "current_password"})

    assert result.success is True
    assert "hunter2" not in result.message
    mock_page.fill.assert_awaited_once_with('[data-ai-index="4"]', "hunter2", timeout=_ELEMENT_ACTION_TIMEOUT_MS)


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
        AsyncMock(return_value="[FAIL] no element matching '#missing'"),
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
async def test_execute_read_page_data_does_not_retry_the_same_call_on_failure(_no_real_sleep):
    """read_page_data ไม่ผ่าน _dispatch_with_retry เหมือน click/fill — target_hint ที่หา
    ไม่เจอเป็น deterministic mismatch ไม่ใช่ DOM-timing issue ที่ retry แล้วจะเปลี่ยนผล

    W_query_is_a_question: มีการเรียก extract_table_data ครั้งที่สองจริง แต่ไม่ใช่ "retry"
    (ยิงคำถามเดิมซ้ำเผื่อฟลุก) — เป็นคำถามคนละข้อ คือ "ที่ hint นี้มีข้อมูลอะไรอยู่บ้างไหม
    ถ้าไม่เอา query ไปกรอง" ซึ่งตอบได้ต่างจากเดิมโดยไม่ต้องรอ DOM เปลี่ยน เหตุผลเดิมในชื่อ
    เทสต์ยังคงอยู่ครบ: argument ชุดเดิมเป๊ะไม่เคยถูกยิงซ้ำ และไม่มีการ sleep รออะไรทั้งสิ้น"""
    mock_page = AsyncMock()
    with patch(
        "backend.app.core.actions.extract_table_data",
        AsyncMock(return_value="[FAIL] no element matching '#missing'"),
    ) as mock_extract:
        result = await execute(
            mock_page, {"type": "read_page_data", "query": "สรุปให้หน่อย", "target_hint": "#missing"},
        )

    assert result.success is False
    called_queries = [call.args[2] for call in mock_extract.await_args_list]
    assert called_queries == ["สรุปให้หน่อย", ""]  # คนละ argument ไม่ใช่การยิงซ้ำ
    _no_real_sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_execute_read_page_data_returns_the_data_when_the_query_is_a_question(_no_real_sleep):
    """W_query_is_a_question (เจอจาก step trace): query ที่เป็นคำถามภาษาธรรมชาติไม่มีวัน
    ปรากฏเป็นข้อความในตาราง extract_table_data จึงคืน [FAIL] แล้วทิ้งข้อมูลที่อ่านมาได้ทั้งหมด
    — live run เสีย 3 step ติดกันกับเรื่องนี้ ต้องคืนข้อมูลให้โมเดลตอบเอง พร้อมบอกตรงๆ ว่า
    ไม่เจอข้อความนั้นแบบตรงตัว (ห้ามแกล้งทำเป็นเจอ ตามเจตนาเดิมของ W46)"""
    mock_page = AsyncMock()
    with patch(
        "backend.app.core.actions.extract_table_data",
        AsyncMock(side_effect=["[FAIL] nothing matching or close to 'first product name'",
                               '["Sauce Labs Onesie", "Sauce Labs Bike Light"]']),
    ):
        result = await execute(
            mock_page,
            {"type": "read_page_data", "query": "first product name", "target_hint": ".inventory_item_name"},
        )

    assert result.success is True
    assert "verbatim" in result.message
    assert "Sauce Labs Onesie" in result.message


@pytest.mark.asyncio
async def test_execute_select_on_custom_dropdown_is_rejected_with_actionable_hint():
    """W_custom_dropdown: OrangeHRM ทำ dropdown ด้วย div + role=combobox ไม่ใช่ <select> —
    select_option() เดิมจะไล่หา <option> ไม่เจอแล้วคืน "no options found in this dropdown"
    ซึ่งอ่านเหมือน "ตัวเลือกไม่มี" ทั้งที่ปัญหาคือ "ใช้ action ผิดชนิด" ต้องปฏิเสธก่อน dispatch
    พร้อมชี้ทางไป protocol W50 (คลิกเปิดก่อน แล้วคลิกตัวเลือก)"""
    mock_page = AsyncMock()

    with patch(
        "backend.app.core.actions.state_filter.check_select_target_is_native",
        AsyncMock(return_value=(
            "This element is a <div>, not a native <select> — the 'select' action only works "
            "on a real <select>. This is a custom dropdown: use type 'click' on this same index "
            "to OPEN it first, then look at the new indexed elements and 'click' the option whose "
            "label matches exactly what you want."
        )),
    ):
        result = await execute(mock_page, {"type": "select", "index": 22, "label": "ESS"})

    assert result.success is False
    assert "not a native <select>" in result.message
    assert "click" in result.message
    mock_page.select_option.assert_not_called()


@pytest.mark.asyncio
async def test_execute_select_on_real_native_select_still_dispatches():
    """ต้องไม่ไปบล็อก <select> จริงที่ทำงานถูกอยู่แล้ว"""
    mock_page = AsyncMock()

    with patch(
        "backend.app.core.actions.state_filter.check_select_target_is_native",
        AsyncMock(return_value=None),
    ), patch("backend.app.core.actions.select_option", AsyncMock(
        return_value=ActionResult(True, "select(2)", "selected 'Price' succeeded"),
    )) as mock_select:
        result = await execute(mock_page, {"type": "select", "index": 2, "label": "Price"})

    assert result.success is True
    mock_select.assert_awaited_once()


# --- W_chain_stale_index -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_click_drops_the_chained_click_when_it_opens_a_dropdown():
    """เคสจาก live run โดยตรง: click เปิด dropdown แล้ว chain ต่อทันที — คลิกที่สองต้องไม่ถูก
    dispatch เลย เพราะ index ของมันมาจาก snapshot ตอนที่ตัวเลือกยังไม่มีอยู่"""
    mock_page = AsyncMock()

    with patch(
        "backend.app.core.actions.state_filter.classify_click_index_disturbance",
        AsyncMock(return_value="trigger"),
    ):
        result = await execute(mock_page, {"type": "click", "index": 22, "then_click_index": 26})

    # คลิกหลักสำเร็จจริง (เปิด dropdown ได้ตามต้องการ) — ไม่ใช่ failure
    assert result.success is True
    assert "dropdown" in result.message
    click_selectors = [c.args[0] for c in mock_page.click.await_args_list]
    assert click_selectors == ['[data-ai-index="22"]']


@pytest.mark.asyncio
async def test_execute_click_still_chains_on_an_ordinary_button():
    """guard ต้องไม่แตะ compound action ปกติที่ทำงานถูกอยู่แล้ว (ปุ่ม Submit ที่เห็นในหน้าเดิม)"""
    mock_page = AsyncMock()

    with patch(
        "backend.app.core.actions.state_filter.classify_click_index_disturbance",
        AsyncMock(return_value=None),
    ):
        result = await execute(mock_page, {"type": "click", "index": 0, "then_click_index": 5})

    assert result.success is True
    assert "then click(5)" in result.message
    click_selectors = [c.args[0] for c in mock_page.click.await_args_list]
    assert click_selectors == ['[data-ai-index="0"]', '[data-ai-index="5"]']


@pytest.mark.asyncio
async def test_execute_click_without_a_chain_does_not_ask_for_a_chain_hint():
    """W_dropdown_sets_filter_dirty: การจำแนก dropdown ย้ายมาอยู่กับ *ทุก* click แล้ว (เพราะ
    orchestrator ต้องรู้ด้วยว่าคลิกนี้เลือกค่าใน filter หรือเปล่า — ดู ActionResult.
    dropdown_option_selected) แต่ click ที่ไม่มี then_click_index ต้องไม่ไปสร้างข้อความ
    "ไม่ chain ต่อเพราะ..." ติดมาในผลลัพธ์ ซึ่งไม่มีความหมายเลยเมื่อไม่มี chain ตั้งแต่ต้น"""
    mock_page = AsyncMock()

    with patch(
        "backend.app.core.actions.state_filter.classify_click_index_disturbance",
        AsyncMock(return_value="trigger"),
    ):
        result = await execute(mock_page, {"type": "click", "index": 0})

    assert "chained click" not in result.message


@pytest.mark.asyncio
async def test_execute_click_marks_a_dropdown_option_selection():
    """W_dropdown_sets_filter_dirty: คลิกที่เป็นการ "เลือกตัวเลือกใน dropdown" ต้องติดธงบน
    ActionResult ให้ orchestrator ยก filter_dirty_since_search ได้ — เดิมธงนั้นยกเฉพาะ
    fill/select ซึ่งไม่มีทางเกิดกับ custom dropdown เลย ทำให้ guard "ห้ามคลิก row action
    ก่อนกด Search" ตายสนิทบนเว็บ SPA"""
    mock_page = AsyncMock()

    with patch(
        "backend.app.core.actions.state_filter.classify_click_index_disturbance",
        AsyncMock(return_value="option"),
    ):
        result = await execute(mock_page, {"type": "click", "index": 5})

    assert result.success is True
    assert result.dropdown_option_selected is True


@pytest.mark.asyncio
async def test_execute_click_on_an_ordinary_button_is_not_a_dropdown_selection():
    """กระจกบานตรงข้ามของเทสต์ด้านบน — click ทั่วไปต้องไม่ติดธงนี้ ไม่งั้น guard จะยิงมั่ว"""
    mock_page = AsyncMock()

    with patch(
        "backend.app.core.actions.state_filter.classify_click_index_disturbance",
        AsyncMock(return_value=None),
    ):
        result = await execute(mock_page, {"type": "click", "index": 5})

    assert result.dropdown_option_selected is False


# ---------------- W_confident_zero: count=0 ไม่ใช่คำตอบจนกว่าจะพิสูจน์ได้ ----------------


@pytest.mark.asyncio
async def test_read_page_data_zero_count_falls_back_to_extraction_instead_of_answering_zero():
    """W_confident_zero (บั๊กจริงบน OrangeHRM): selector ที่ไม่ตรงอะไรเลยทำให้ count_elements()
    คืน 0 ซึ่งเดิมถูกรายงานเป็น "found 0 entries" (success=True) — โมเดลจึงตอบว่า "เจอ 0
    รายการ" อย่างมั่นใจทั้งที่ความจริงมีอยู่ 5 ต้องลองอ่านข้อมูลจริงด้วย hint เดิมก่อนเสมอ"""
    mock_page = AsyncMock()
    table = "| Username | User Role |\n| --- | --- |\n| a | ESS |"
    with patch("backend.app.core.actions.count_elements", AsyncMock(return_value=0)), \
         patch("backend.app.core.actions.extract_table_data", AsyncMock(return_value=table)) as mock_extract:
        result = await execute(
            mock_page,
            {"type": "read_page_data", "query": "how many users", "target_hint": "table tbody tr"},
        )

    assert result.success is True
    assert "0 entries" not in result.message
    assert table in result.message
    mock_extract.assert_awaited_once_with(mock_page, "table tbody tr", "")


@pytest.mark.asyncio
async def test_read_page_data_zero_count_with_nothing_readable_refuses_to_answer_zero():
    """ไม่มีอะไรอ่านได้เลย = selector ผิด ไม่ใช่ "คำตอบคือศูนย์" — ต้องคืน success=False และ
    บอกตรงๆ ว่าห้ามรายงาน 0 เป็นคำตอบ"""
    mock_page = AsyncMock()
    with patch("backend.app.core.actions.count_elements", AsyncMock(return_value=0)), \
         patch("backend.app.core.actions.extract_table_data", AsyncMock(return_value="[FAIL] nope")):
        result = await execute(
            mock_page,
            {"type": "read_page_data", "query": "how many users", "target_hint": "table tbody tr"},
        )

    assert result.success is False
    assert "does NOT mean the answer is zero" in result.message


@pytest.mark.asyncio
async def test_read_page_data_nonzero_count_still_answers_directly():
    """ทางเร็วเดิมต้องไม่เปลี่ยน: นับได้ > 0 ตอบตรงๆ ไม่ต้องแตะ extract_table_data เลย"""
    mock_page = AsyncMock()
    with patch("backend.app.core.actions.count_elements", AsyncMock(return_value=6)), \
         patch("backend.app.core.actions.extract_table_data", AsyncMock()) as mock_extract:
        result = await execute(
            mock_page,
            {"type": "read_page_data", "query": "how many users", "target_hint": '[role="row"]'},
        )

    assert result.success is True
    assert "6" in result.message
    mock_extract.assert_not_awaited()


# ---------------- W_deterministic_count: โค้ดนับให้ ไม่ปล่อยให้โมเดลนับด้วยตา ----------------

_USERS_TABLE = """|  | Username | User Role | Status |
| --- | --- | --- | --- |
|  | Admin | Admin | Enabled |
|  | DemoNonAdminPH | ESS | Disabled |
|  | ess.irhrg0 | ESS | Enabled |
|  | kabir | Admin | Enabled |"""


def test_extracted_entries_parses_markdown_table_without_header_or_separator():
    entries = _extracted_entries(_USERS_TABLE)

    assert len(entries) == 4
    assert all("---" not in e for e in entries)
    assert not any("Username" in e for e in entries)


def test_extracted_entries_parses_json_list():
    assert len(_extracted_entries('["Sauce Labs Backpack", "Sauce Labs Onesie"]')) == 2


def test_extracted_entries_returns_empty_when_it_cannot_parse():
    """นับไม่ได้ต้องไม่เดา — ไม่มีตัวเลขดีกว่าตัวเลขผิด"""
    assert _extracted_entries("just some prose from the page") == []


def test_deterministic_count_note_reports_total_entries():
    note = _deterministic_count_note(_USERS_TABLE, "how many users are there")

    assert "exactly 4 entries" in note
    assert "do not recount" in note


def test_deterministic_count_note_also_counts_a_key_value_condition_from_the_goal():
    """W_deterministic_count (บั๊กจริง): goal จริงของ user เขียนว่า "userrole=ess" — ดึงค่า
    ฝั่งขวาของ = มานับให้เลย แทนที่จะปล่อยให้โมเดลไล่นับแถวเอง (ซึ่งตอบผิด 6 จาก 7 จริง)"""
    note = _deterministic_count_note(_USERS_TABLE, "ลบ userrole=ess ออกให้หมด เจอกี่รายการ")

    assert "exactly 4 entries" in note
    assert "2 of those 4 entries contain 'ess'" in note


def test_deterministic_count_note_is_empty_when_data_cannot_be_parsed():
    assert _deterministic_count_note("just some prose", "how many") == ""


@pytest.mark.asyncio
async def test_read_page_data_attaches_the_system_computed_count_to_extracted_data():
    mock_page = AsyncMock()
    with patch("backend.app.core.actions.count_elements", AsyncMock(return_value=0)), \
         patch("backend.app.core.actions.extract_table_data", AsyncMock(return_value=_USERS_TABLE)):
        result = await execute(
            mock_page,
            {"type": "read_page_data", "query": "how many have userrole=ess",
             "target_hint": "table tbody tr"},
        )

    assert result.success is True
    assert "exactly 4 entries" in result.message
    assert "2 of those 4 entries contain 'ess'" in result.message
    assert _USERS_TABLE in result.message


# ---------------- W_click_navigated (P0 F1): click ที่พาไป navigate ไม่ใช่ failure ----------------


@pytest.mark.asyncio
async def test_click_that_times_out_but_navigates_is_reported_as_success():
    """บั๊กจริงบน OrangeHRM: คลิก "Admin" สำเร็จและหน้าเปลี่ยนไปแล้ว แต่ attempt 2/3 ไปหา
    data-ai-index เดิมบนหน้าใหม่ (ซึ่งไม่มีทางมี) จน timeout แล้วข้อความของรอบสุดท้ายชนะ —
    รายงานเป็น [FAIL] ทั้งที่คลิกได้ผลจริง"""
    mock_page = AsyncMock()
    mock_page.url = "https://app.example.com/admin/list"

    async def _click_then_navigate(selector, timeout=None):
        mock_page.url = "https://app.example.com/admin/users"
        raise PWTimeout("Timeout 3000ms exceeded")

    mock_page.click.side_effect = _click_then_navigate

    result = await _dispatch_click_with_retry(mock_page, 3)

    assert result.success is True
    assert "https://app.example.com/admin/users" in result.message
    # ต้องเลิก retry ทันทีที่รู้ว่า URL เปลี่ยนแล้ว — retry ต่อไม่มีทางสำเร็จและกินเวลาเปล่า
    assert mock_page.click.await_count == 1


@pytest.mark.asyncio
async def test_click_that_navigates_only_by_hash_route_still_counts_as_navigation():
    """SPA จำนวนมากใช้ hash router (#/admin/users) — ห้าม normalize fragment ทิ้งเหมือนที่
    crawler._normalize_url() ทำ (คนละหน้าที่กันสิ้นเชิง) ไม่งั้นจะเป็น false FAIL เหมือนเดิม"""
    mock_page = AsyncMock()
    mock_page.url = "https://app.example.com/#/admin/list"

    async def _click_then_navigate(selector, timeout=None):
        mock_page.url = "https://app.example.com/#/admin/users"
        raise PWTimeout("Timeout 3000ms exceeded")

    mock_page.click.side_effect = _click_then_navigate

    result = await _dispatch_click_with_retry(mock_page, 3)

    assert result.success is True


@pytest.mark.asyncio
async def test_click_that_fails_without_navigating_still_fails_but_reports_dom_change():
    """สัญญาณสำรองสำหรับ SPA ที่เปลี่ยนแค่ state ภายใน — ห้ามพลิกเป็น success จาก DOM signature
    เพราะ DOM อาจเปลี่ยนจาก toast/spinner ที่ไม่เกี่ยวเลย แต่ต้องแนบหลักฐานไปให้โมเดลเห็น"""
    mock_page = AsyncMock()
    mock_page.url = "https://app.example.com/admin/list"
    mock_page.click.side_effect = PWTimeout("Timeout 3000ms exceeded")
    mock_page.evaluate.side_effect = [1000, 4200]

    result = await _dispatch_click_with_retry(mock_page, 3)

    assert result.success is False
    assert "DOM did change" in result.message


@pytest.mark.asyncio
async def test_click_that_fails_with_no_change_at_all_reports_a_plain_failure():
    """กันการ regress: การล้มเหลวจริงๆ ต้องยังรายงานตรงไปตรงมาเหมือนเดิม ไม่มีโน้ตเสริมมั่ว"""
    mock_page = AsyncMock()
    mock_page.url = "https://app.example.com/admin/list"
    mock_page.click.side_effect = PWTimeout("Timeout 3000ms exceeded")
    mock_page.evaluate.side_effect = [1000, 1000]

    result = await _dispatch_click_with_retry(mock_page, 3)

    assert result.success is False
    assert "DOM did change" not in result.message
    assert "after 2 attempts" in result.message


# ---------------- W_conditional_count / W_count_answer_check (P1.1) ----------------


def test_deterministic_count_note_reports_a_zero_match_instead_of_staying_silent():
    """W_conditional_count: เดิมเงียบไปเลยตอนไม่มีแถวไหนตรงเงื่อนไข เหลือแต่บรรทัด "exactly N
    entries" ให้โมเดลอ่านแล้วรายงาน N เป็นคำตอบของคำถามที่มีเงื่อนไข ซึ่งตอบคนละคำถามกัน"""
    note = _deterministic_count_note(_USERS_TABLE, "มี userrole=manager กี่คน")

    assert "0 of those 4 entries contain 'manager'" in note
    # ต้องคงเจตนาของ W_confident_zero ไว้ด้วย — 0 ไม่ใช่คำตอบจนกว่าจะพิสูจน์ได้ว่าอ่านถูกตาราง
    assert "not the right table" in note


def test_system_counted_conditions_reads_back_what_the_note_wrote():
    """W_count_answer_check: ตัวเขียนกับตัวอ่าน format เดียวกันต้องตรงกันเสมอ (วางไว้ติดกันใน
    ไฟล์เดียวกันด้วยเหตุผลนี้) — เทสต์นี้คือสิ่งที่จะพังทันทีถ้ามีใครแก้ข้อความฝั่งเดียว"""
    note = _deterministic_count_note(_USERS_TABLE, "มี userrole=ess กี่คน")

    assert system_counted_conditions(note) == {"ess": 2}
    assert system_counted_conditions("[OK] click(3) -> click succeeded") == {}


@pytest.mark.asyncio
async def test_count_query_with_a_condition_never_answers_with_a_raw_selector_count():
    """W_conditional_count (บั๊กจริง พิสูจน์ซ้ำได้ 2026-08-26): เส้นทางหลักของคำถามเชิงนับ
    (count_elements คืนค่ามากกว่า 0) ไม่รู้จักเงื่อนไขใน query เลย — "มี user ที่ userrole=ess
    กี่คน" + '[role=row]' คืน "found 21 entries" ทั้งที่คำตอบจริงคือ 2 และคืน success=True ด้วย
    จึงไม่มีสัญญาณให้ใครจับได้ว่าตอบผิด"""
    mock_page = AsyncMock()
    with patch("backend.app.core.actions.count_elements", AsyncMock(return_value=21)) as mock_count, \
         patch("backend.app.core.actions.extract_table_data", AsyncMock(return_value=_USERS_TABLE)), \
         patch("backend.app.core.actions.wait_stable", AsyncMock()):
        result = await execute(
            mock_page,
            {"type": "read_page_data", "query": "มี user ที่ userrole=ess กี่คน",
             "target_hint": '[role="row"]'},
        )

    assert result.success is True
    assert "2 of those 4 entries contain 'ess'" in result.message
    assert "found 21 entries" not in result.message
    # อ่านแถวจริงได้แล้ว ไม่ต้องเสีย round-trip ไปนับ selector ดิบอีก
    mock_count.assert_not_awaited()


@pytest.mark.asyncio
async def test_count_query_with_a_condition_refuses_to_offer_the_raw_count_when_rows_are_unreadable():
    """อ่านแถวจริงไม่ได้แต่ selector ยังนับได้ — ต้องรายงานตามความจริงว่านี่คือจำนวน element
    ไม่ใช่จำนวนรายการที่ตรงเงื่อนไข (ธีมเดียวกับ W_click_native_select/W_confident_zero)"""
    mock_page = AsyncMock()
    with patch("backend.app.core.actions.count_elements", AsyncMock(return_value=21)), \
         patch("backend.app.core.actions.extract_table_data", AsyncMock(return_value="[FAIL] nothing")), \
         patch("backend.app.core.actions.wait_stable", AsyncMock()):
        result = await execute(
            mock_page,
            {"type": "read_page_data", "query": "มี user ที่ userrole=ess กี่คน",
             "target_hint": '[role="row"]'},
        )

    assert "Do NOT report 21 as the answer" in result.message


@pytest.mark.asyncio
async def test_count_query_without_any_condition_still_uses_the_cheap_selector_count():
    """กันการ regress: คำถามเชิงนับที่ไม่มีเงื่อนไขต้องยังนับด้วย selector ตรงๆ เหมือนเดิม
    (ถูกกว่าและถูกต้องอยู่แล้ว ไม่มีเหตุผลให้ไปอ่านตารางทั้งก้อนมา)"""
    mock_page = AsyncMock()
    with patch("backend.app.core.actions.count_elements", AsyncMock(return_value=6)), \
         patch("backend.app.core.actions.extract_table_data", AsyncMock()) as mock_extract, \
         patch("backend.app.core.actions.wait_stable", AsyncMock()):
        result = await execute(
            mock_page,
            {"type": "read_page_data", "query": "how many products are there",
             "target_hint": ".inventory_item"},
        )

    assert result.message == "found 6 entries matching '.inventory_item'"
    mock_extract.assert_not_awaited()


_MIXED_COLUMN_TABLE = """| Username | User Role | Status |
| --- | --- | --- |
| ess.irhrg0 | Admin | Enabled |
| jane | ESS | Enabled |"""


def test_count_uses_the_column_named_on_the_left_of_the_equals_sign():
    """W_column_aware_count: "userrole=ess" หมายถึงคอลัมน์ User Role ไม่ใช่ "แถวไหนก็ได้ที่มี
    คำว่า ess" — username "ess.irhrg0" ที่ Role เป็น Admin เคยถูกนับเป็น ESS ด้วย ทำให้ตอบ 2
    ทั้งที่ความจริงคือ 1 (ฝั่งซ้ายของ = ถูกทิ้งไปทั้งหมดในเวอร์ชันก่อน)"""
    note = _deterministic_count_note(_MIXED_COLUMN_TABLE, "มี userrole=ess กี่คน")

    assert "1 of those 2 entries contain 'ess' in the 'User Role' column" in note


def test_count_falls_back_to_the_whole_row_when_the_column_name_does_not_match():
    """ตารางที่ไม่มีคอลัมน์ชื่อนั้น (หรือ JSON list ที่ไม่มีคอลัมน์เลย) ต้องยังนับได้เหมือนเดิม
    — fallback ต้องไม่หายไปพร้อมกับการเพิ่มความแม่นยำ"""
    note = _deterministic_count_note('["Sauce Labs Backpack", "Sauce Labs Onesie"]', "how many name=Sauce")

    assert "2 of those 2 entries contain 'Sauce' anywhere in the row" in note


def test_column_matching_ignores_spacing_and_case_in_the_header():
    """user พิมพ์ชื่อคอลัมน์รูปแบบไหนก็ได้ ("userrole"/"user_role"/"User Role") ต้องเทียบติดหมด"""
    note = _deterministic_count_note(_MIXED_COLUMN_TABLE, "how many user_role=admin")

    assert "1 of those 2 entries contain 'admin' in the 'User Role' column" in note


# ---------------- W_fill_wrapper_resolves_to_inner_input ----------------
# บั๊กจริงหน้า Update Password ของ OrangeHRM (2026-09-03): fill_secret(40) ล้มด้วย
# "Element is not an <input>, <textarea>, <select> or [contenteditable]" เพราะ index ชี้ที่
# div ที่ห่อ <input> ไว้ (perception ติด index ให้ตัวห่อเพราะมันคือตัวที่มี label อ่านได้)


def _fill_target(is_fillable, inner_found=object()):
    target = AsyncMock()
    locator = MagicMock()
    locator.evaluate = AsyncMock(return_value=is_fillable)
    target.locator = MagicMock(return_value=locator)
    target.query_selector = AsyncMock(return_value=inner_found)
    return target


@pytest.mark.asyncio
async def test_effective_fill_selector_points_at_the_input_inside_a_wrapper():
    target = _fill_target(is_fillable=False)

    selector = await actions._effective_fill_selector(target, '[data-ai-index="40"]', 3000)

    assert selector.startswith('[data-ai-index="40"] ')
    assert "input" in selector


@pytest.mark.asyncio
async def test_effective_fill_selector_leaves_a_real_input_untouched():
    """ช่องกรอกปกติต้องไม่ถูกแตะเลย — ไม่งั้น selector ยาวขึ้นโดยไม่จำเป็นทุก fill"""
    target = _fill_target(is_fillable=True)

    assert await actions._effective_fill_selector(
        target, '[data-ai-index="2"]', 3000,
    ) == '[data-ai-index="2"]'


@pytest.mark.asyncio
async def test_effective_fill_selector_gives_up_honestly_when_nothing_is_fillable_inside():
    """ไม่มีช่องกรอกข้างในเลย = คืน selector เดิม ให้ Playwright บอก error ตามความจริง
    ดีกว่าเดาไปเรื่อยแล้วกรอกผิดที่ (หลักการเดียวกับ W_confident_zero)"""
    target = _fill_target(is_fillable=False, inner_found=None)

    assert await actions._effective_fill_selector(
        target, '[data-ai-index="3"]', 3000,
    ) == '[data-ai-index="3"]'


@pytest.mark.asyncio
async def test_effective_fill_selector_fails_open_when_the_dom_cannot_be_read():
    target = _fill_target(is_fillable=False)
    target.locator.return_value.evaluate = AsyncMock(side_effect=Exception("detached"))

    assert await actions._effective_fill_selector(
        target, '[data-ai-index="9"]', 3000,
    ) == '[data-ai-index="9"]'


# ---------------------------------------------------------------------------
# W_submit_before_confirm_password + W_menu_open_note_needs_no_chain
#
# ทั้งสองบั๊กมาจากรันสด 2026-09-03 บน OrangeHRM และทั้งคู่เป็นพฤติกรรมของ DOM จริง
# (ค่าในช่อง password ว่างหรือไม่ / เมนูที่เพิ่งเปิดสร้าง element ใหม่) — mock พิสูจน์ไม่ได้
# จึงใช้ Chromium จริง เหมือน W_fill_wrapper_resolves_to_inner_input
# ---------------------------------------------------------------------------

_PASSWORD_FORM_HTML = """
<html><body>
  <form onsubmit="document.title='SUBMITTED'; return false;">
    <input data-ai-index="1" type="password" />
    <input data-ai-index="2" type="password" />
    <input data-ai-index="3" type="password" />
    <button data-ai-index="4" type="submit">Save</button>
  </form>
</body></html>
"""

_MENU_HTML = """
<html><body>
  <div data-ai-index="1" role="button" aria-haspopup="true"
       onclick="document.getElementById('m').hidden = !document.getElementById('m').hidden">Profile</div>
  <ul id="m" role="menu" hidden>
    <li data-ai-index="2" role="menuitem">About</li>
    <li data-ai-index="3" role="menuitem">Change Password</li>
  </ul>
  <button data-ai-index="4">Elsewhere</button>
</body></html>
"""


async def _with_page(html):
    """context manager แบบง่ายๆ ไม่ได้ — คืน (playwright, browser, page) ให้ปิดเอง"""
    pw = await async_playwright().start()
    browser = await pw.chromium.launch()
    page = await browser.new_page()
    await page.set_content(html)
    return pw, browser, page


@pytest.mark.asyncio
async def test_fill_does_not_submit_while_other_password_fields_are_empty():
    pw, browser, page = await _with_page(_PASSWORD_FORM_HTML)
    try:
        result = await execute(
            page, {"type": "fill", "index": 1, "text": "abc",
                   "key": "Enter", "then_click_index": 4},
        )
        assert result.success is True          # การกรอกยังต้องสำเร็จตามปกติ
        assert "did not submit the form" in result.message
        assert await page.title() != "SUBMITTED"
        assert await page.input_value('[data-ai-index="1"]') == "abc"
    finally:
        await browser.close()
        await pw.stop()


@pytest.mark.asyncio
async def test_fill_still_submits_once_it_is_the_last_empty_password_field():
    """ห้ามตัดการส่งฟอร์มทิ้งเสมอ — ไม่งั้นงานเปลี่ยนรหัสผ่านจะกดบันทึกไม่ได้เลย"""
    pw, browser, page = await _with_page(_PASSWORD_FORM_HTML)
    try:
        await execute(page, {"type": "fill", "index": 1, "text": "abc"})
        await execute(page, {"type": "fill", "index": 2, "text": "abc"})
        result = await execute(
            page, {"type": "fill", "index": 3, "text": "abc",
                   "key": "Enter", "then_click_index": 4},
        )
        assert result.success is True
        assert "did not submit the form" not in result.message
        assert await page.title() == "SUBMITTED"
    finally:
        await browser.close()
        await pw.stop()


@pytest.mark.asyncio
async def test_click_that_opens_a_menu_says_the_indexes_are_stale_without_a_chained_click():
    pw, browser, page = await _with_page(_MENU_HTML)
    try:
        result = await execute(page, {"type": "click", "index": 1})
        assert result.success is True
        assert "now OPEN" in result.message
        assert "do NOT reuse the index you just clicked" in result.message
    finally:
        await browser.close()
        await pw.stop()


@pytest.mark.asyncio
async def test_ordinary_click_gets_no_extra_note():
    """โน้ตนี้กินโควตา token ทุก step ที่แนบ — ต้องไม่แถมให้คลิกที่ไม่เกี่ยวกับเมนู"""
    pw, browser, page = await _with_page(_MENU_HTML)
    try:
        result = await execute(page, {"type": "click", "index": 4})
        assert result.success is True
        assert result.message.strip() == "click succeeded"
    finally:
        await browser.close()
        await pw.stop()


# W_chained_submit_after_fill (บั๊กจริงจากรันสดสองเทิร์นผ่าน REST API 2026-09-04): เทิร์นแรก
# กรอก 12345678 ทั้งช่อง Password และ Confirm แล้วเว็บปฏิเสธเพราะไม่มีตัวพิมพ์เล็ก เทิร์นที่สอง
# user ตอบด้วยรหัสที่ผ่านนโยบาย โมเดลกรอกทับเฉพาะช่อง Password แล้วพ่วง click Save มาด้วย
# ช่อง Confirm ยังค้างค่าเดิม -> 'Passwords do not match'
#
# ต้องเช็คหลัง fill เท่านั้น: ก่อน fill ทั้งสองช่องยังถือค่าเก่าซึ่ง "ตรงกัน" พอดี guard ที่
# เช็คก่อน dispatch จึงมองไม่เห็นปัญหาเลย

_CHANGE_PASSWORD_HTML = """
<html><body>
  <form onsubmit="document.title='SUBMITTED'; return false;">
    <div><label>Current Password</label><input data-ai-index="1" type="password" /></div>
    <div><label>Password</label><input data-ai-index="2" type="password" /></div>
    <div><label>Confirm Password</label><input data-ai-index="3" type="password" /></div>
    <button data-ai-index="4" type="submit">Save</button>
  </form>
</body></html>
"""


@pytest.mark.asyncio
async def test_a_chained_submit_is_dropped_when_the_confirmation_still_holds_the_old_value():
    pw, browser, page = await _with_page(_CHANGE_PASSWORD_HTML)
    try:
        for i, v in [(1, "old"), (2, "12345678"), (3, "12345678")]:
            await page.fill(f'[data-ai-index="{i}"]', v)
        result = await execute(
            page, {"type": "fill", "index": 2, "text": "Abcd1234",
                   "key": "Tab", "then_click_index": 4},
        )
        assert result.success is True                     # การกรอกยังต้องสำเร็จ
        assert "do not hold the same value" in result.message
        assert await page.title() != "SUBMITTED"
    finally:
        await browser.close()
        await pw.stop()


@pytest.mark.asyncio
async def test_the_chained_submit_goes_through_once_both_fields_agree():
    """ห้ามตัดการส่งฟอร์มทิ้งเสมอ — ไม่งั้นงานเปลี่ยนรหัสผ่านจะจบไม่ได้เลย"""
    pw, browser, page = await _with_page(_CHANGE_PASSWORD_HTML)
    try:
        for i, v in [(1, "old"), (2, "Abcd1234"), (3, "12345678")]:
            await page.fill(f'[data-ai-index="{i}"]', v)
        result = await execute(
            page, {"type": "fill", "index": 3, "text": "Abcd1234",
                   "key": "Tab", "then_click_index": 4},
        )
        assert result.success is True
        assert "did not submit the form" not in result.message
        assert await page.title() == "SUBMITTED"
    finally:
        await browser.close()
        await pw.stop()

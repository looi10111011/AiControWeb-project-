from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from playwright.async_api import TimeoutError as PWTimeout, async_playwright

from backend.app.core.actions import ActionResult, _ELEMENT_ACTION_TIMEOUT_MS, execute
from backend.app.core.perception import get_snapshot


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

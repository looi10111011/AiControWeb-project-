"""W_prompt_sections (P4.1): SYSTEM_PROMPT ถูกแยกเป็น core + 4 บล็อกที่ฉีดตามบริบท

หลักฐานที่ทำให้ต้องทำ (step trace ของ release gate จริง): step แรกของทุก task เริ่มที่ ~11.4k
input token ทั้งที่หน้า saucedemo/MiniWoB มี element ไม่กี่ตัว — เกือบทั้งหมดคือ prompt
"""
import pytest

from backend.app.core import llm
from backend.app.core.orchestrator import _resolve_prompt_sections

# ผูกกับทะเบียนจริงเสมอ ไม่ hardcode ชื่อ section — ตอนเพิ่ม section ใหม่ (W_core_carries_
# situational_rules ย้ายกฎ 5 ก้อนออกจาก core) ลิสต์ที่ hardcode ไว้ค้างอยู่ที่ 4 ตัวเดิม
# แล้วเทสต์ fidelity ก็ล้มทันทีทั้งที่ prompt ครบถ้วนดี
_ALL = llm.ALL_PROMPT_SECTIONS
_NONE = frozenset()


# ---------------- ตัวประกอบ prompt ----------------

def test_full_prompt_keeps_every_original_line():
    """กันบรรทัดหายตอนแยกบล็อก — เทียบเป็น multiset เพราะบล็อกที่ gate ถูกย้ายไปต่อท้าย
    โดยตั้งใจ (core ต้องเป็น prefix คงที่ ไม่งั้น prefix cache ของ provider พลาดทุกครั้ง)"""
    from collections import Counter

    rebuilt = llm.build_system_prompt(_ALL)
    assert Counter(rebuilt.split("\n")) == Counter(llm.SYSTEM_PROMPT.split("\n"))
    assert len(rebuilt) == len(llm.SYSTEM_PROMPT)


def test_default_is_the_full_prompt_so_existing_callers_are_unchanged():
    assert llm.build_system_prompt() == llm.SYSTEM_PROMPT
    assert llm.build_system_prompt(None) == llm.SYSTEM_PROMPT


def test_core_only_prompt_is_substantially_smaller():
    core = llm.build_system_prompt(_NONE)
    assert len(core) < len(llm.SYSTEM_PROMPT) * 0.6


def test_core_is_a_stable_prefix_of_every_variant():
    """เหตุผลทั้งหมดที่ย้ายบล็อกที่ gate ไปต่อท้าย — ถ้า core ไม่ใช่ prefix ร่วม prefix cache
    ของ provider จะพลาดทุกครั้งที่ sections เปลี่ยน"""
    core = llm.build_system_prompt(_NONE)
    for name in ("plan", "table", "widget", "password"):
        assert llm.build_system_prompt(frozenset({name})).startswith(core)
    assert llm.SYSTEM_PROMPT.startswith(core)


def test_section_order_is_fixed_regardless_of_set_iteration_order():
    a = llm.build_system_prompt(frozenset({"password", "plan"}))
    b = llm.build_system_prompt(frozenset({"plan", "password"}))
    assert a == b


@pytest.mark.parametrize("name,needle", [
    ("table", "Batch/Bulk Action Protocol"),
    ("widget", "custom dropdown/menu"),
    ("password", "Account Security & Password Actions"),
    ("plan", "current plan confirmed by the user"),
])
def test_each_block_is_absent_from_core_and_present_when_requested(name, needle):
    assert needle not in llm.build_system_prompt(_NONE)
    assert needle in llm.build_system_prompt(frozenset({name}))


def test_anthropic_system_block_is_one_fixed_prefix_with_cache_control():
    """W_token_cut W2 เปลี่ยน _system_blocks() ให้ไม่รับ sections อีกต่อไป — system ของ
    Anthropic เป็น _PROMPT_CORE คงที่ทุกเทิร์นทุก task ส่วนบล็อกที่ gate ย้ายไปต่อท้าย user
    turn เพื่อไม่ให้ prefix cache ขาดกลาง task (เทสต์เดิมยังเรียกด้วยอาร์กิวเมนต์เก่าอยู่จึง
    ตกมาตั้งแต่ commit นั้น — TypeError ไม่ใช่ assertion)"""
    blocks = llm._system_blocks()
    assert blocks[0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert blocks[0]["text"] == llm._PROMPT_CORE
    # object เดิมซ้ำ — สำคัญกับ prefix cache ของ provider
    assert llm._system_blocks() is blocks


# ---------------- ตัวตัดสินว่าจะฉีดบล็อกไหน ----------------

def _resolve(previous=_NONE, goal="open the site", plan_text=None, elements=(), allow_fill_secret=False):
    return _resolve_prompt_sections(
        previous, goal=goal, plan_text=plan_text, elements=list(elements),
        allow_fill_secret=allow_fill_secret,
    )


def test_plain_navigation_goal_on_a_plain_page_needs_no_gated_block():
    assert _resolve() == _NONE


def test_plan_block_tracks_whether_a_plan_actually_exists():
    assert "plan" in _resolve(plan_text="1. login\n2. done")
    assert "plan" not in _resolve(plan_text=None)
    assert "plan" not in _resolve(plan_text="")


def test_password_block_reuses_the_fill_secret_gate_already_computed():
    assert "password" in _resolve(allow_fill_secret=True)
    assert "password" not in _resolve(allow_fill_secret=False)


# W_password_rules_arrive_too_late (บั๊กจริงพร้อมภาพหน้าจอ 2026-09-03): goal "เปลี่ยนรหัสผ่าน
# ใหม่เป็น 12345678" แต่ agent เดินไป PIM > Update Password ของพนักงาน แทนที่จะกดเมนูโปรไฟล์
# มุมขวาบน > Change Password — บล็อกนี้มีกฎ W20 สั่งเรื่องนี้ไว้ตรงตัว แต่เดิมส่งเฉพาะตอน
# allow_fill_secret ซึ่งเป็นจริงก็ต่อเมื่อ *ยืนอยู่บนฟอร์มเปลี่ยนรหัสผ่านแล้ว* คือหลังจากเลือก
# ทางผิดไปแล้ว โมเดลจึงไม่เคยเห็นกฎในจังหวะที่ต้องตัดสินใจเลือกทาง


def test_password_block_arrives_as_soon_as_the_goal_mentions_a_password_change():
    assert "password" in _resolve(
        goal="เปิดเว็บแล้วเปลี่ยนรหัสผ่านใหม่เป็น 12345678 ให้หน่อย", allow_fill_secret=False,
    )


def test_password_block_survives_thai_spacing():
    """W_thai_keyword_space: goal จริงของ user มีเว้นวรรคกลางคำ ("เปลี่ยน รหัสผ่าน") ซึ่งเดิม
    ทำให้ keyword match พลาดทั้งชุด บล็อก W20 จึงไม่ถูกส่งตอนโมเดลกำลังเลือกทาง"""
    assert "password" in _resolve(
        goal="เปิดเว็ป แล้วเปลี่ยน รหัสผ่านเป็น 12345678", allow_fill_secret=False,
    )


def test_password_block_also_arrives_when_only_the_plan_mentions_it():
    assert "password" in _resolve(
        goal="ทำตามแผน", plan_text="1. เลือก Change Password จากเมนูโปรไฟล์",
        allow_fill_secret=False,
    )


# W_secret_gate_stays_page_only (regression 2026-09-03): ตอนแก้ W_password_rules_arrive_too_late
# ผมไปเปิด `allow_fill_secret` จาก goal ด้วย ซึ่งเป็นคนละธงกัน — ธงตัวนั้นคุม *tool schema*
# ไม่ใช่ prompt พอเปิดตั้งแต่หน้า login แล้ว gpt-5.4-mini (ซึ่งกรอกทุก property ในสคีมาเสมอ)
# ก็ยิง fill_secret ใส่ index มั่วทั้ง task รันสดล้ม 2/2 รอบโดยไม่เคยกดเมนูโปรไฟล์เลย
# เทสต์นี้ตรึงไว้ว่าบล็อก prompt กับ schema gate ต้องแยกจากกัน: บล็อกมาจาก goal ได้ (ข้างบน)
# แต่ธง schema ต้องมาจากหน้าเว็บอย่างเดียว


def test_the_schema_gate_never_opens_from_the_goal_text_alone():
    """อ่านซอร์สตรงๆ เพราะสิ่งที่ต้องกันคือ *ที่มาของค่า* ไม่ใช่ผลลัพธ์ — pattern เดียวกับ
    เทสต์กัน drift ของ marker registry (W108)"""
    import re
    from pathlib import Path

    src = Path("backend/app/core/orchestrator.py").read_text(encoding="utf-8")
    assignment = re.search(r"^\s*allow_fill_secret = (.+)$", src, re.M)
    assert assignment is not None, "หา assignment ของ allow_fill_secret ไม่เจอ"
    assert "_page_looks_like_change_password_form" in assignment.group(1)
    assert "_goal_or_plan_requests_password_change" not in assignment.group(1)


def test_unrelated_goals_still_do_not_pay_for_the_password_block():
    """เหตุผลที่ gate นี้มีอยู่คือลด token — วัดแล้วบล็อกนี้ +2,548 ตัวอักษร (~640 token)
    ต่อ *ทุกเทิร์น* ของ task งานที่ไม่เกี่ยวกับรหัสผ่านจึงต้องไม่ถูกแถม"""
    assert "password" not in _resolve(goal="ลบ userrole=ess ออกให้หมด", allow_fill_secret=False)


@pytest.mark.parametrize("goal", [
    "ลบ userrole=ess ออกให้หมด",
    "delete all inactive users",
    "change the role of every ESS user to Admin",
    "มี user ที่ userrole=ess กี่คน",
])
def test_table_block_fires_for_bulk_and_counting_goals(goal):
    assert "table" in _resolve(goal=goal)


def test_table_block_fires_when_the_page_shows_a_selectable_table():
    assert "table" in _resolve(elements=[{"tag": "span", "label": "Select row"}])
    assert "table" in _resolve(elements=[{"tag": "span", "label": "(41) Records Found"}])


def test_widget_block_fires_for_native_select_and_for_custom_dropdown_triggers():
    assert "widget" in _resolve(elements=[{"tag": "select", "label": "A B C"}])
    assert "widget" in _resolve(elements=[{"tag": "div", "label": "-- Select --"}])
    assert "widget" not in _resolve(elements=[{"tag": "button", "label": "Login"}])


def test_sections_only_accumulate_and_are_never_dropped():
    """กฎที่โมเดลเคยเห็นแล้วหายไปกลางทางเป็นพฤติกรรมที่ไล่บั๊กยากมาก — และ prompt ที่กระพริบ
    ทำให้ prefix cache พลาดทุกครั้งที่สลับ"""
    after = _resolve(previous=frozenset({"widget", "table"}))
    assert {"widget", "table"} <= after


def test_resolver_ignores_elements_without_a_usable_label():
    assert _resolve(elements=[{"tag": "div"}, {"tag": "a", "label": None}]) == _NONE


# ---------------- W_tab_rebind (P3.6): ตามไปแท็บใหม่ที่ action เพิ่งเปิด ----------------

class _FakePage:
    def __init__(self, url, context=None):
        self.url = url
        self.context = context


class _FakeContext:
    def __init__(self, pages):
        self.pages = pages


def _ctx(*urls):
    ctx = _FakeContext([])
    ctx.pages = [_FakePage(u, ctx) for u in urls]
    return ctx


def test_detect_tab_switch_follows_a_tab_the_action_just_opened():
    """เคสที่เจ็บที่สุด: ลิงก์ target=_blank เปิดแท็บเอง โมเดลไม่เคยสั่ง switch_tab จึงไม่มีทาง
    แก้เองได้ ถ้าลูปไม่ตามไปให้ มันจะดึง snapshot ของแท็บเก่าซ้ำไปเรื่อยๆ"""
    from backend.app.core.orchestrator import _detect_tab_switch

    ctx = _ctx("https://a/", "https://b/")
    first, second = ctx.pages
    page, note = _detect_tab_switch(first, [first], {"type": "click", "index": 3})

    assert page is second
    assert "opened a new tab" in note
    assert "https://b/" in note


def test_detect_tab_switch_leaves_the_page_alone_when_nothing_opened():
    from backend.app.core.orchestrator import _detect_tab_switch

    ctx = _ctx("https://a/")
    only = ctx.pages[0]
    page, note = _detect_tab_switch(only, [only], {"type": "click", "index": 1})

    assert page is only
    assert note == ""


def test_detect_tab_switch_honours_an_explicit_switch_tab_action():
    from backend.app.core.orchestrator import _detect_tab_switch

    ctx = _ctx("https://a/", "https://b/")
    first, second = ctx.pages
    page, note = _detect_tab_switch(second, list(ctx.pages), {"type": "switch_tab", "tab_index": 0})

    assert page is first
    assert "Now operating on tab 0" in note


def test_detect_tab_switch_ignores_an_out_of_range_tab_index():
    from backend.app.core.orchestrator import _detect_tab_switch

    ctx = _ctx("https://a/")
    only = ctx.pages[0]
    page, note = _detect_tab_switch(only, list(ctx.pages), {"type": "switch_tab", "tab_index": 99})

    assert page is only
    assert note == ""


def test_detect_tab_switch_never_raises_when_the_context_is_gone():
    """browser/context ปิดไปแล้วตอนถูกเรียก (task กำลังจบ/ถูก stop) ต้องไม่ทำให้ step ที่
    ทำสำเร็จไปแล้วกลายเป็น error ย้อนหลัง"""
    from backend.app.core.orchestrator import _detect_tab_switch

    class _Dead:
        url = "https://a/"

        @property
        def context(self):
            raise RuntimeError("Target page, context or browser has been closed")

    dead = _Dead()
    page, note = _detect_tab_switch(dead, [], {"type": "click"})

    assert page is dead
    assert note == ""


# ---------------- W_listbox_container: container ของรายการตัวเลือกต้องไม่ได้ index ----------------

@pytest.mark.asyncio
async def test_open_dropdown_exposes_each_option_but_not_the_container_that_wraps_them():
    """บั๊กจริง live run 2026-08-27: role=listbox ที่ถูกเพิ่มเข้ามาใน W85 ได้ index เอง โดย label
    เป็น innerText ของทุก option ต่อกัน ('-- Select -- Admin ESS') ซึ่งมีคำว่า ESS ที่ goal ต้องการ
    อยู่ด้วย โมเดลจึงคลิกมันแล้วไปโดน option แรก (Admin) = กรองผิด role ทั้ง task

    เทสต์นี้รัน chromium จริงเพราะ _COLLECT_JS เป็น JS ทั้งก้อน — mock DOM พิสูจน์อะไรไม่ได้เลย
    """
    from playwright.async_api import async_playwright

    from backend.app.core.perception import get_snapshot

    html = """
      <body>
        <label>User Role</label>
        <div class="oxd-select-text" role="combobox" tabindex="0">-- Select --</div>
        <div role="listbox" tabindex="-1">
          <div role="option">-- Select --</div>
          <div role="option">Admin</div>
          <div role="option">ESS</div>
        </div>
        <div onclick="void 0">
          <div role="option">Alpha</div>
          <div role="option">Beta</div>
        </div>
      </body>
    """
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.set_content(html)
        elements, _ = await get_snapshot(page)
        await browser.close()

    labels = [e["label"] for e in elements]

    # ตัวเลือกแต่ละตัวยังต้องมี index ของตัวเอง (ไม่งั้น agent เลือกอะไรไม่ได้เลย)
    for option in ("Admin", "ESS", "Alpha", "Beta"):
        assert any(l.strip() == option for l in labels), f"option {option!r} หายไปจาก snapshot"

    # และต้องไม่มี element ไหนที่ label รวมหลาย option ไว้ด้วยกัน
    assert not [l for l in labels if "Admin" in l and "ESS" in l]

    # trigger ยังต้องอยู่ พร้อมชื่อ field นำหน้า (W_dropdown_field_label)
    assert any("User Role" in l for l in labels)


# ---------------- P3.9: ถอด hardcode OrangeHRM ออกจากทางเดินกลาง ----------------

def _parse_record_count(text):
    """เลียนแบบตรรกะใน _scan_remaining_target_records_once() ตรงส่วนที่แปลงข้อความเป็นตัวเลข"""
    from backend.app.core.orchestrator import (
        _RECORD_COUNT_PATTERNS,
        _RECORD_COUNT_ZERO_TEXTS,
    )

    if any(zero in text.lower() for zero in _RECORD_COUNT_ZERO_TEXTS):
        return 0
    for pattern in _RECORD_COUNT_PATTERNS:
        match = pattern.search(text)
        if match:
            try:
                return int(match.group(1).replace(",", ""))
            except ValueError:
                continue
    return None


@pytest.mark.parametrize("text,expected", [
    ("(41) Records Found", 41),          # OrangeHRM — ต้องไม่พัง
    ("No Records Found", 0),
    ("Showing 1-10 of 42", 42),          # pagination แบบมาตรฐาน
    ("1 - 10 of 1,234 results", 1234),   # มีตัวคั่นหลักพัน
    ("42 results found", 42),
    ("7 items", 7),
    ("ทั้งหมด 28 รายการ", 28),
    ("ไม่พบข้อมูล", 0),
    ("No results", 0),
])
def test_record_count_reads_common_result_summaries_not_just_orangehrm(text, expected):
    """W_record_count_generic: guard กัน false-completion ตัวเรือธงเคยผูกกับข้อความ
    '(N) Records Found' ของ OrangeHRM อย่างเดียว = ใช้ได้กับเว็บเดียวในโลก ที่เหลือคืน None
    เงียบๆ แปลว่าไม่มี guard เลย"""
    assert _parse_record_count(text) == expected


def test_record_count_prefers_the_result_total_over_a_page_number():
    """'of N' กว้างที่สุดจึงต้องอยู่ท้ายสุด — ไม่งั้นประโยคที่มีทั้งเลขหน้าและยอดรวมจะได้เลขหน้า"""
    assert _parse_record_count("Showing page 2 of 5 - 42 results") == 42


def test_record_count_returns_none_when_the_text_carries_no_count():
    """fail-safe เดิม: อ่านไม่ได้ = None = ไม่บล็อกอะไร ดีกว่าบล็อก finish_task ที่อาจถูกอยู่แล้ว"""
    assert _parse_record_count("Dashboard") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("html,expected_label", [
    # OrangeHRM เดิม ต้องไม่พัง
    ('<div class="oxd-dialog-container oxd-dialog-container-default">'
     '<button class="oxd-button--secondary">No, Cancel</button>'
     '<button class="oxd-button--label-danger">Yes, Delete</button></div>', "Yes, Delete"),
    ('<div role="dialog"><button>Cancel</button><button>OK</button></div>', "OK"),
    ('<div role="dialog"><button>No</button><button>Yes</button></div>', "Yes"),
    ('<div role="dialog"><button>ยกเลิก</button><button>ตกลง</button></div>', "ตกลง"),
    ('<div role="dialog"><button>Abbrechen</button><button>Löschen</button></div>', "Löschen"),
    ('<div role="dialog"><button>Annuler</button><button>Confirmer</button></div>', "Confirmer"),
    # ปุ่มที่ไม่ใช่ <button>
    ('<div role="dialog"><div role="button">Cancel</div>'
     '<div role="button">Confirm</div></div>', "Confirm"),
    # ปุ่มลวง: คำว่า delete โผล่ในปุ่มที่ *ไม่ควร* กด — คำยืนยันกลางๆ ต้องชนะ
    ('<div role="dialog"><button>Do not delete</button><button>Yes</button></div>', "Yes"),
])
async def test_modal_confirm_button_is_found_on_dialogs_that_are_not_orangehrm(html, expected_label):
    """W_modal_confirm_generic: fallback เดิมกว้างแค่ button:has-text("Confirm") — dialog ที่
    เขียนว่า Yes/OK/ตกลง/Löschen ไม่ match อะไรเลย แล้ว agent ค้างอยู่หน้าโมดัล ซึ่งเป็นบั๊ก
    ที่ W23 เขียนมาแก้พอดี แต่แก้ได้เฉพาะเว็บภาษาอังกฤษที่ใช้คำว่า Confirm"""
    from playwright.async_api import async_playwright

    from backend.app.core.actions import _find_visible_modal_confirm_button

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.set_content(f"<body>{html}</body>")
        _, locator = await _find_visible_modal_confirm_button(page)
        label = (await locator.inner_text()).strip() if locator else None
        await browser.close()

    assert label == expected_label


# W_password_field_has_no_label_attributes (บั๊กจริงที่วัดกับหน้าเว็บจริง 2026-09-03): ช่อง
# รหัสผ่านของ OrangeHRM ไม่มี label/aria-label/placeholder/name/id เลยสักตัว <label> เป็น
# พี่น้องอยู่ใน div.oxd-input-group ไม่ได้ผูกด้วย for= gate จึงคืน False บนหน้าเปลี่ยนรหัสผ่าน
# จริง -> fill_secret ถูกตัดจาก tool schema -> โมเดลกรอกรหัสปัจจุบันไม่ได้ ยิง fill(21, "")
# ซ้ำจนโดน loop detector ฆ่า (รันสด: 8 steps จบด้วย Failed)
#
# เทสต์นี้ทดสอบกับ Chromium จริง ไม่ mock — บั๊กนี้เกิดจากพฤติกรรมของ DOM API (el.labels ว่าง
# เมื่อ label ไม่ได้ผูกด้วย for=) ซึ่ง mock จะพิสูจน์ไม่ได้เลย เป็นบทเรียนเดียวกับ
# W_fill_wrapper_resolves_to_inner_input

_ORANGEHRM_SHAPED_FORM = """
<html><body><form>
  <div class="oxd-input-group">
    <label class="oxd-label">Current Password</label>
    <div class="oxd-input-group__label-wrapper"></div>
    <input type="password" />
  </div>
  <div class="oxd-input-group">
    <label class="oxd-label">Password</label>
    <input type="password" />
  </div>
  <div class="oxd-input-group">
    <label class="oxd-label">Confirm Password</label>
    <input type="password" />
  </div>
</form></body></html>
"""

# ฟอร์ม Add User: 2 ช่องเหมือนกันเป๊ะ แต่ไม่มี "Current Password" — ต้องยัง False
_ADD_USER_SHAPED_FORM = """
<html><body><form>
  <div class="oxd-input-group">
    <label class="oxd-label">Password</label>
    <input type="password" />
  </div>
  <div class="oxd-input-group">
    <label class="oxd-label">Confirm Password</label>
    <input type="password" />
  </div>
</form></body></html>
"""


async def _gate_on(html):
    from playwright.async_api import async_playwright

    from backend.app.core.orchestrator import _page_looks_like_change_password_form

    async with async_playwright() as pw:
        browser = await pw.chromium.launch()
        try:
            page = await browser.new_page()
            await page.set_content(html)
            return await _page_looks_like_change_password_form(page)
        finally:
            await browser.close()


@pytest.mark.asyncio
async def test_gate_finds_the_label_when_it_is_only_a_sibling_in_the_wrapper():
    assert await _gate_on(_ORANGEHRM_SHAPED_FORM) is True


@pytest.mark.asyncio
async def test_add_user_form_is_still_not_mistaken_for_a_change_password_form():
    """W_add_user_form_false_positive ต้องไม่หายไปกับการมองหา label ที่กว้างขึ้น"""
    assert await _gate_on(_ADD_USER_SHAPED_FORM) is False


# W_plan_counter_claims_a_password_change (บั๊กจริงจากรันสดผ่าน REST API 2026-09-04): task จบ
# ด้วย success=true ที่ step 4 ทั้งที่ยังไม่เคยกรอก Confirm Password และไม่เคยกดบันทึก —
# ground truth ด้วยสคริปต์ไม่ใช้ LLM ยืนยันว่ารหัสผ่านเดโมไม่ถูกเปลี่ยน goal-scope hard stop
# เชื่อ completed_plan_step ที่โมเดลรายงานเอง คลาสเดียวกับ W_plan_cursor_not_proof


async def _still_unfilled(html):
    from playwright.async_api import async_playwright

    from backend.app.core.orchestrator import _change_password_form_still_unfilled

    async with async_playwright() as pw:
        browser = await pw.chromium.launch()
        try:
            page = await browser.new_page()
            await page.set_content(html)
            return await _change_password_form_still_unfilled(page)
        finally:
            await browser.close()


@pytest.mark.asyncio
async def test_a_change_password_form_with_an_empty_field_is_proof_the_work_is_not_done():
    assert await _still_unfilled(_ORANGEHRM_SHAPED_FORM) is True


@pytest.mark.asyncio
async def test_nothing_blocks_completion_once_every_password_field_is_filled():
    filled = _ORANGEHRM_SHAPED_FORM.replace(
        '<input type="password" />', '<input type="password" value="x" />',
    )
    assert await _still_unfilled(filled) is False


@pytest.mark.asyncio
async def test_pages_that_are_not_change_password_forms_are_never_blocked():
    """หลักฐานนี้ต้องพูดเฉพาะเรื่องที่มันรู้จริง — งานอื่นทุกชนิดต้องไม่ถูกกันไม่ให้จบ"""
    assert await _still_unfilled("<html><body><input type='text'></body></html>") is False
    assert await _still_unfilled(_ADD_USER_SHAPED_FORM) is False


# W_core_carries_situational_rules (วัดจากงานจริง 2026-09-04): core ที่ส่งทุก call คือก้อนที่
# ใหญ่ที่สุดของ payload (6,286 tok เทียบกับ snapshot ~275 tok) และ 36% ของมันเป็นกฎที่ใช้ได้
# เฉพาะสถานการณ์ ย้ายออกมา gate ตาม marker ที่กฎนั้นพูดถึงเอง


def test_the_core_no_longer_carries_the_marker_rules():
    core = llm.build_system_prompt(frozenset())
    for marker in ("[required]", "[disabled]", "[already active]", "PRE_LEARNED_MANUAL"):
        assert marker not in core, marker
    assert len(core) < 20000          # เดิม 25,146 ตัวอักษร


def test_marker_rules_arrive_exactly_when_the_marker_is_on_the_page():
    assert "marker_required" in _resolve(elements=[{"label": "Username [required]"}])
    assert "marker_disabled" in _resolve(elements=[{"label": "Save [disabled]"}])
    assert "marker_active" in _resolve(elements=[{"label": "Admin [already active]"}])
    assert "save_toast" in _resolve(elements=[{"label": "Save"}])


def test_a_page_without_those_markers_pays_for_none_of_them():
    """เหตุผลทั้งหมดของการย้ายคือราคาต่อเทิร์น — หน้าที่ไม่มี marker ต้องไม่ถูกแถมสักบล็อก"""
    sections = _resolve(elements=[{"label": "Dashboard"}, {"label": "Search"}])
    for name in ("marker_required", "marker_disabled", "marker_active", "manual"):
        assert name not in sections, name


def test_the_manual_block_arrives_only_with_a_strict_manual():
    from backend.app.core.orchestrator import _resolve_prompt_sections

    with_manual = _resolve_prompt_sections(
        _NONE, goal="ไปหน้าแอดมิน", plan_text=None, elements=(), allow_fill_secret=False,
        manual_context="[PRE_LEARNED_MANUAL]\nroute: /admin",
    )
    assert "manual" in with_manual
    plain = _resolve_prompt_sections(
        _NONE, goal="ไปหน้าแอดมิน", plan_text=None, elements=(), allow_fill_secret=False,
        manual_context="สรุปคู่มือทั่วไปของเว็บนี้",
    )
    assert "manual" not in plain

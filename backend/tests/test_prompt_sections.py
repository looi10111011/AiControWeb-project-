"""W_prompt_sections (P4.1): SYSTEM_PROMPT ถูกแยกเป็น core + 4 บล็อกที่ฉีดตามบริบท

หลักฐานที่ทำให้ต้องทำ (step trace ของ release gate จริง): step แรกของทุก task เริ่มที่ ~11.4k
input token ทั้งที่หน้า saucedemo/MiniWoB มี element ไม่กี่ตัว — เกือบทั้งหมดคือ prompt
"""
import pytest

from backend.app.core import llm
from backend.app.core.orchestrator import _resolve_prompt_sections

_ALL = frozenset({"plan", "table", "widget", "password"})
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


def test_anthropic_system_blocks_keep_cache_control_and_are_cached_per_variant():
    blocks = llm._system_blocks(_NONE)
    assert blocks[0]["cache_control"] == {"type": "ephemeral"}
    assert blocks[0]["text"] == llm.build_system_prompt(_NONE)
    # object เดิมซ้ำ — สำคัญกับ prefix cache ของ provider
    assert llm._system_blocks(_NONE) is blocks


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

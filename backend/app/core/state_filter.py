"""core/state_filter.py — W19: เช็คแบบ deterministic (ไม่พึ่ง LLM) ก่อน dispatch ใน
actions.py::execute() ว่า action "จำเป็นจริงไหม/ใช้ถูกชนิดไหม" — fill ค่าเดิม, check ที่ติ๊กอยู่แล้ว,
scroll สุดขอบแล้ว, click ของที่ disabled, ใช้ action ผิดชนิดกับ element ฯลฯ
ต่างจาก actions.py::_dispatch_with_retry (W5) ที่ retry เพื่อให้สำเร็จ — ตัวนี้ตัดสินว่า "ไม่ต้องทำ/ทำไม่ได้"

ห้าม throw ออกไปเด็ดขาด — อ่านสถานะไม่ได้ (element หาย/frame ปิด/mock ไม่ได้ config) = คืนค่า
fail-safe (None/False/0 = ไม่ขวาง) ปล่อยให้ dispatch จริงไปเจอ error ของตัวเอง"""

from typing import Optional

from playwright.async_api import Page

from backend.app.core.perception import resolve_frame

_STATE_CHECK_TIMEOUT_MS = 500


def _sel(index: int) -> str:
    return f'[data-ai-index="{index}"]'


async def _locator(page: Page, index: int):
    """locator ของ element ที่ index นี้ (ข้าม frame ได้) — throw ได้ ผู้เรียกต้องห่อ try เอง"""
    selector = _sel(index)
    target = await resolve_frame(page, selector)
    return target.locator(selector)


async def _evaluate(page: Page, index: int, js: str, *args):
    """evaluate js บน element ที่ index นี้ — throw ได้ ผู้เรียกต้องห่อ try เอง"""
    locator = await _locator(page, index)
    return await locator.evaluate(js, *args, timeout=_STATE_CHECK_TIMEOUT_MS)


async def check_fill_redundant(page: Page, index: int, text: str) -> Optional[str]:
    """REDUNDANT ถ้าช่องมีข้อความ = text อยู่แล้วเป๊ะ"""
    try:
        current = await (await _locator(page, index)).input_value(timeout=_STATE_CHECK_TIMEOUT_MS)
    except Exception:
        return None
    if current == text:
        return f"This field already contains '{text}' — no need to fill it again"
    return None


async def check_fill_is_empty_noop(page: Page, index: int, text: str) -> Optional[str]:
    """W_empty_fill_noop (live run ของ user): โมเดล fill text="" ลงช่องที่ว่างอยู่แล้ว 3 step ติด —
    check_fill_redundant() คืน success จึงไม่มีสัญญาณว่าทำสิ่งไร้ความหมาย -> คืนเป็น failure แทน
    ต้องอ่านค่าจริงก่อน: fill("") ลงช่องที่ *มี* ข้อความคือการล้างค่า ซึ่งถูกต้อง ห้ามบล็อก"""
    if text != "":
        return None
    try:
        current = await (await _locator(page, index)).input_value(timeout=_STATE_CHECK_TIMEOUT_MS)
    except Exception:
        return None
    if current != "":
        return None
    return (
        "This field is already empty, and you asked to fill it with an empty string — that does "
        "nothing at all. If you meant to CLEAR it, it is already clear. If you meant to TYPE a "
        "value, put the actual value in 'text'. If you are unsure what to do next, do not fill "
        "more fields: look at the goal again and at the page you are on."
    )


async def check_select_target_is_native(page: Page, index: int) -> Optional[str]:
    """W_custom_dropdown (live บน OrangeHRM): "select" ใส่ dropdown แบบ div/combobox ได้ error "no options
    found" ที่อ่านเหมือนตัวเลือกไม่มีอยู่ ทั้งที่ใช้ action ผิดชนิด (filter Role=ESS ไม่เคยติด จน user
    ต้องกด Stop) -> ชี้ไป protocol W50 (คลิกเปิดก่อนแล้วคลิกตัวเลือก)"""
    try:
        tag = await _evaluate(page, index, "el => el.tagName.toLowerCase()")
    except Exception:
        return None
    if isinstance(tag, str) and tag and tag != "select":
        return (
            f"This element is a <{tag}>, not a native <select> — the 'select' action only works "
            "on a real <select>. This is a custom dropdown: use type 'click' on this same index "
            "to OPEN it first, then look at the new indexed elements and 'click' the option whose "
            "label matches exactly what you want."
        )
    return None


async def check_click_target_is_native_select(page: Page, index: int) -> Optional[str]:
    """W_click_native_select (live บน saucedemo 2026-08-26): กลับข้างกับ W_custom_dropdown — click ใส่
    <select> จริงคืน [OK] แต่ไม่มีอะไรเปลี่ยน ("sort by Price" -> click(2) 3 ครั้ง -> loop-detected)
    "สำเร็จแต่ไม่มีผล" อันตรายกว่าล้มเหลวชัดเจน -> คืน failure ชี้ไป select + label"""
    try:
        tag = await _evaluate(page, index, "el => el.tagName.toLowerCase()")
    except Exception:
        return None
    if tag == "select":
        return (
            "This element is a native <select>. Clicking it does nothing — it neither opens a "
            "list you can then click nor changes the selected value. Use action type 'select' "
            f"on this same index {index} with 'label' set to the exact option text you want "
            "(the options are listed in this element's label)."
        )
    return None


async def check_fill_target_is_file_input(page: Page, index: int) -> Optional[str]:
    """W_file_input_guard (P3.10): fill() ลง <input type="file"> throw ทุกครั้งจนหมด retry โดยไม่บอกทางออก
    *** ตั้งใจไม่เพิ่ม action อัปโหลดไฟล์ *** — เท่ากับให้ agent อ่านไฟล์ในเครื่องจาก path ที่โมเดลแต่ง
    เป็นการตัดสินใจด้านความปลอดภัยของเจ้าของโปรเจกต์ ตรงนี้แค่บอกให้ขอไฟล์จาก user"""
    try:
        input_type = await _evaluate(
            page, index,
            "el => (el.tagName || '').toLowerCase() === 'input' ? (el.type || '') : ''",
        )
    except Exception:
        return None
    if isinstance(input_type, str) and input_type.lower() == "file":
        return (
            "This is a file upload input. Text cannot be typed into it, and this agent is not "
            "allowed to pick files from the computer on its own. If the task genuinely needs a "
            "file here, call request_user_input to ask the person to attach or select the file "
            "themselves, then continue with the rest of the task."
        )
    return None


_TYPABLE_TARGET_JS = """(el) => {
    const tag = (el.tagName || "").toLowerCase();
    if (tag === "input" || tag === "textarea" || el.isContentEditable) return "";
    // fill() รองรับ wrapper ที่ห่อช่องกรอกไว้ข้างในอยู่แล้ว (W_fill_wrapper_resolves_to_inner_input)
    if (el.querySelector("input, textarea, [contenteditable]")) return "";
    const role = (el.getAttribute && el.getAttribute("role")) || "";
    const cls = typeof el.className === "string" ? el.className : "";
    const looksLikeDropdown = role === "combobox" || role === "listbox"
        || (el.getAttribute && el.getAttribute("aria-haspopup"))
        || /select-text|select-wrapper|dropdown/i.test(cls);
    return looksLikeDropdown ? "dropdown" : "other";
}"""


async def check_fill_target_is_not_typable(page: Page, index: int) -> Optional[str]:
    """W_fill_untypable_target (Add Candidate ของ OrangeHRM 2026-09-08): พิมพ์อีเมลลงตัวเปิด dropdown
    (div.oxd-select-text-input) ได้ error ดิบของ Playwright ที่ไม่บอกทางออก -> คืนทางที่ทำได้จริง
    ไม่ปฏิเสธ wrapper ที่มีช่องกรอกข้างใน (fill() แก้ให้เอง)"""
    try:
        kind = await _evaluate(page, index, _TYPABLE_TARGET_JS)
    except Exception:
        return None
    if kind == "dropdown":
        return (
            "This is a custom dropdown/menu trigger, not a text field — text cannot be typed "
            "into it. Click it to open the menu, then click the option whose label matches the "
            "value you want (never guess how many ArrowDown presses to send)."
        )
    if kind == "other":
        return (
            "This element is not a text field and cannot accept typed text. Look for the real "
            "input/textarea for this value in the indexed elements, or click this element if it "
            "is a button/link."
        )
    return None


_MENU_OVERLAY_JS = """(el) => {
    const r = el.getBoundingClientRect();
    if (!r.width || !r.height) return false;
    const top = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);
    if (!top || top === el || el.contains(top)) return false;
    const MENU = [
        '[role="listbox"]', '[role="menu"]', '[role="combobox"]',
        '[class*="select-dropdown"]', '[class*="dropdown-menu"]', '[class*="autocomplete"]',
    ].join(",");
    const menu = top.closest(MENU);
    // เป้าที่อยู่ *ใน* เมนูเองไม่ใช่เคสนี้ (นั่นคือการเลือกตัวเลือกตามปกติ)
    return !!menu && !menu.contains(el);
}"""


async def menu_overlay_covers_target(page: Page, index: int) -> bool:
    """เป้าหมายถูกเมนูที่เปิดค้างอยู่บังไว้หรือเปล่า (fail-safe คืน False)

    W_menu_overlay_blocks_target (OrangeHRM 2026-09-08): Tab ท้าย Last Name เปิดเมนู Vacancy คลุมช่อง
    Email -> fill/click รอ actionability หมดเวลา 6.6s แล้วโดน go_back จนงานพัง
    ต่างจาก marker [obscured] ของ perception (อาจเป็น tooltip ที่หายเอง) — ตัวนี้คือเมนูที่ปิดด้วย Escape"""
    try:
        covered = await _evaluate(page, index, _MENU_OVERLAY_JS)
        # is True ไม่ใช่ bool() — ค่าที่ไม่ใช่ boolean แท้ (mock/อ็อบเจ็กต์แปลก) = "ตอบไม่ได้" ไม่ใช่
        # "ถูกบัง" ไม่งั้นจะยิง Escape มั่ว
        return covered is True
    except Exception:
        return False


async def check_checkbox_redundant(page: Page, index: int) -> Optional[str]:
    """REDUNDANT ถ้า checkbox/radio ถูกติ๊กอยู่แล้ว ("check" ไม่ใช่ toggle)"""
    try:
        already_checked = await (await _locator(page, index)).is_checked(
            timeout=_STATE_CHECK_TIMEOUT_MS,
        )
    except Exception:
        return None
    if already_checked is True:
        return "This checkbox/radio is already ticked"
    return None


_SIBLING_CHECKBOX_JS = """(el) => {
    // ไล่ ancestor ขึ้นไปหา "กลุ่ม" ที่เล็กที่สุดที่มี checkbox ตั้งแต่ 2 ตัวขึ้นไป แล้วคืนจำนวน
    // ตั้งใจไม่ผูกกับ <form>/<fieldset> เพราะหน้าเว็บจำนวนมาก (รวม MiniWoB ที่เจอบั๊กนี้)
    // วาง checkbox ไว้ใน div เปล่าๆ ไม่มี form เลย
    const visible = (b) => b === el || (!b.disabled && b.offsetParent !== null);
    let scope = el.parentElement;
    let boxes = [];
    while (scope) {
        boxes = Array.from(scope.querySelectorAll('input[type="checkbox"]')).filter(visible);
        if (boxes.length >= 2) return boxes.length;
        scope = scope.parentElement;
    }
    return 0;
}"""


async def checkbox_group_size(page: Page, index: int) -> int:
    """จำนวน checkbox ในกลุ่มเดียวกับ index นี้ — 0 ถ้าเป็นช่องเดี่ยวหรืออ่านไม่ได้
    ใช้ตัดสินการพ่วง submit ต่อท้ายการติ๊ก (W_chained_submit_after_check ใน actions.py)

    ตั้งใจคืนแค่ขนาดกลุ่ม ไม่ใช่ "ช่องที่ยังว่าง" — เวอร์ชันแรกคืนรายชื่อแล้วโมเดลติ๊กเพิ่มให้ "ครบ"
    (2026-09-07: โจทย์ขอ 3 จาก 4 ช่อง โมเดลติ๊กครบ 4) ชั้นนี้ไม่รู้ goal จึงชี้นำผิดได้"""
    try:
        size = await _evaluate(page, index, _SIBLING_CHECKBOX_JS)
    except Exception:
        return 0
    return int(size) if isinstance(size, (int, float)) else 0


async def element_is_checkbox(page: Page, index: int) -> bool:
    """element นี้เป็น checkbox เองไหม — ตอบจาก DOM จริง เพราะ then_type เป็น optional (default "")
    เชื่อไม่ได้ ใช้แยก "พ่วงติ๊กช่องถัดไป" (ตั้งใจ) ออกจาก "พ่วงปุ่มส่งฟอร์ม" (บั๊ก)"""
    try:
        return bool(await _evaluate(
            page, index,
            "el => (el.tagName || '').toLowerCase() === 'input'"
            " && (el.type || '').toLowerCase() === 'checkbox'",
        ))
    except Exception:
        return False


_CHECKABLE_TARGET_JS = """(el) => {
    const tag = (el.tagName || "").toLowerCase();
    const type = (el.getAttribute("type") || "").toLowerCase();
    const role = (el.getAttribute("role") || "").toLowerCase();
    if (tag === "input" && (type === "checkbox" || type === "radio")) return "";
    if (role === "checkbox" || role === "radio" || role === "switch") return "";
    // custom checkbox ที่ซ่อน <input> จริงไว้ข้างใน (เช่น .oxd-checkbox-input) — check()
    // มีทางรองรับอยู่แล้วผ่าน JS-click + verify ห้ามปฏิเสธ
    if (el.querySelector('input[type="checkbox"], input[type="radio"], [role="checkbox"], [role="radio"]')) return "";
    if (el.closest && el.closest("label") && el.closest("label").querySelector('input[type="checkbox"], input[type="radio"]')) return "";
    if (tag === "button" || tag === "a" || type === "submit" || type === "button") return "button";
    return "other";
}"""


async def check_check_target_is_not_checkable(page: Page, index: int) -> Optional[str]:
    """W_check_fires_a_button (release-gate d29ed4a, MiniWoB click-checkboxes): "check" ใส่ปุ่ม Submit
    -> JS-click สำรอง *กดปุ่มไปจริง* แต่รายงานว่ายืนยันการติ๊กไม่ได้ โมเดลจึงทำซ้ำ (ฟอร์มส่งไปแล้ว)
    ปฏิเสธก่อน dispatch — แต่ไม่ปฏิเสธ custom checkbox ที่ check() รองรับ (role=checkbox/ห่อ input)"""
    try:
        kind = await _evaluate(page, index, _CHECKABLE_TARGET_JS)
    except Exception:
        return None
    if kind == "button":
        return (
            "This is a button/link, not a checkbox — ticking it would press it instead, and "
            "the result would say the tick could not be confirmed even though the button had "
            'already fired. Use type: "click" if you mean to press it.'
        )
    if kind == "other":
        return (
            "This element is not a checkbox or radio and cannot be ticked. Look for the real "
            'checkbox in the indexed elements, or use type: "click"/"fill" for what this '
            "element actually is."
        )
    return None


_ELEMENT_LABEL_JS = r"""(el) => ((el.innerText || el.value || el.getAttribute("aria-label")
    || el.getAttribute("placeholder") || "") + "").trim().replace(/\s+/g, " ").slice(0, 80)"""


async def element_text_at(page: Page, index: int) -> Optional[str]:
    """ข้อความของ element ที่ index นี้ *ตอนนี้* — None ถ้าอ่านไม่ได้

    W_select_reorders_the_page (release-gate 50eefd0, long_flow): select "Name (Z to A)" พ่วงคลิก
    index 10 — การเรียงสลับตำแหน่งสินค้า ผู้เรียกเทียบข้อความก่อน/หลังว่าเป้ายังเป็นตัวเดิมไหม"""
    try:
        return await _evaluate(page, index, _ELEMENT_LABEL_JS)
    except Exception:
        return None


async def check_click_redundant(page: Page, index: int) -> Optional[str]:
    """REDUNDANT ถ้า element disabled ไปแล้ว — perception กรอง disabled ตอน snapshot แต่สถานะอาจ
    เปลี่ยนระหว่างที่ LLM คิด (perceive กับ dispatch ไม่ atomic)"""
    try:
        disabled = await (await _locator(page, index)).is_disabled(
            timeout=_STATE_CHECK_TIMEOUT_MS,
        )
    except Exception:
        return None
    if disabled is True:
        return "This element is already disabled — it cannot be clicked"
    return None


_INDEX_DISTURBING_CLICK_JS = """(el) => {
    // คืน 'option' | 'trigger' | '' — ผู้เรียกแปลงเป็นข้อความคนละแบบ เพราะสองเคสนี้ทำให้ index
    // เดิมใช้ไม่ได้ "ด้วยเหตุผลตรงข้ามกัน" (อันหนึ่งเปิด list ขึ้นมาใหม่ อีกอันปิดมันลง) ถ้า
    // อธิบายผิดข้าง โมเดลจะแก้ผิดทาง — บั๊กจริงรอบแรกของ guard นี้เป็นแบบนั้นเป๊ะๆ
    const clsOf = (n) => {
        const c = n.className;
        if (c && c.baseVal !== undefined) return c.baseVal;       // SVG
        return typeof c === 'string' ? c : '';
    };
    const roleOf = (n) => (n.getAttribute && n.getAttribute('role')) || '';

    // (1) เช็คก่อนเสมอ: กำลังคลิก "ตัวเลือก" ที่อยู่ใน dropdown/menu ที่เปิดอยู่แล้วหรือเปล่า
    // ต้องมาก่อนข้อ (2) เพราะ library หลายตัว (react-select ฯลฯ) วาง menu ไว้ใน wrapper
    // เดียวกับ control ที่ถือ role/class ของ trigger — ไล่ ancestor ขึ้นไปจะเจอทั้งคู่
    const OPTION_ROLES = ['option', 'listbox', 'menu', 'menuitem'];
    const OPTION_CLASS_HINTS = ['option', 'menu-item', 'menuitem'];
    let node = el;
    for (let depth = 0; node && depth < 3; depth++, node = node.parentElement) {
        if (OPTION_ROLES.includes(roleOf(node))) return 'option';
        const cls = clsOf(node);
        if (OPTION_CLASS_HINTS.some(h => cls.includes(h))) return 'option';
    }

    // (2) trigger ของ dropdown/menu/combobox — ARIA ก่อน (มาตรฐาน ใช้ได้ข้ามเว็บ) แล้วค่อย
    // fallback ไป class ที่ library ยอดนิยมใช้กัน
    const CLASS_HINTS = [
        'oxd-select-text', 'oxd-select-wrapper',       // OrangeHRM
        'select__control', 'select__value-container',  // react-select
        'dropdown-toggle', 'ant-select-selector', 'v-select', 'MuiSelect',
    ];
    node = el;
    for (let depth = 0; node && depth < 3; depth++, node = node.parentElement) {
        if (node.tagName && node.tagName.toLowerCase() === 'select') return '';
        const role = roleOf(node);
        if (role === 'combobox') return 'trigger';
        const popup = (node.getAttribute && node.getAttribute('aria-haspopup')) || '';
        if (popup && popup !== 'false') return 'trigger';
        // W_expanded_alone_is_not_a_menu: aria-expanded เดี่ยวๆ แปลว่า "กดแล้วมีอะไรกางออก"
        // ซึ่งจริงกับ tab/accordion/disclosure ด้วย ไม่ใช่แค่ dropdown — คืนชนิดที่อ่อนกว่า
        // เพื่อให้ผู้เรียกเลือกได้ว่าจะเชื่อแค่ไหน (ดู index_shift_note_for_kind)
        if (node.hasAttribute && node.hasAttribute('aria-expanded')) return 'trigger_weak';
        const cls = clsOf(node);
        if (CLASS_HINTS.some(h => cls.includes(h))) return 'trigger';
    }
    return '';
}"""

_CHAIN_HINT_BY_KIND = {
    "trigger": (
        "the element you clicked opens a dropdown/menu, so its options did not exist yet when "
        "the indexes on screen were assigned — a chained click would land on an unrelated "
        "element. The dropdown is now OPEN: look at the new indexed elements and click the "
        "option whose label matches exactly what you want, as a separate next step"
    ),
    "option": (
        "you clicked an option inside a dropdown that was already open. Choosing it CLOSES the "
        "dropdown and removes every option from the page, so all the indexes you were given "
        "have shifted — a chained click would land on an unrelated element. Your selection was "
        "applied: look at the new indexed elements and issue your next click (Search, Save, "
        "etc.) as a separate step"
    ),
}


# W_menu_open_note_needs_no_chain (รันสด 2026-09-03): click(36) เมนูโปรไฟล์แล้วคลิก 36 ซ้ำอีก 4 ครั้งจน
# loop detector บังคับ recovery — ข้อความ _CHAIN_HINT_BY_KIND แนบเฉพาะตอนมี then_click_index คลิกเปล่า
# จึงไม่รู้ว่าเมนูเปิดและ index เลื่อนแล้ว ต้องเขียนคนละสำนวน (ชุด chain อธิบายเหตุที่ *ไม่ chain*)
_INDEX_SHIFT_NOTE_BY_KIND = {
    "trigger": (
        "this element opens a dropdown/menu and it is now OPEN. Its items only came into "
        "existence with this click, so every index you were given before it is stale — do NOT "
        "reuse the index you just clicked (clicking it again only closes the menu). Read the "
        "new indexed elements and click the item whose label matches what you want"
    ),
    "option": (
        "that was an option inside an open dropdown. Choosing it CLOSES the menu and removes "
        "every option from the page, so the indexes have shifted. Your selection was applied — "
        "read the new indexed elements before your next action"
    ),
}


def index_shift_note_for_kind(kind: Optional[str]) -> Optional[str]:
    """W_menu_open_note_needs_no_chain: note สำหรับคลิกที่ *ไม่มี* then_click_index

    W_expanded_alone_is_not_a_menu (release gate 2026-09-07, MiniWoB click-tab): tab ของ jQuery UI มี
    aria-expanded จึงได้ข้อความ "กดซ้ำจะปิดเมนู" แล้ววนคลิกแท็บเดิม -> trigger_weak ไม่แนบ note เลย
    แนบเฉพาะสัญญาณแรง (role=combobox / aria-haspopup / class ของ select library)"""
    return _INDEX_SHIFT_NOTE_BY_KIND.get(kind) if kind else None


def chain_hint_for_kind(kind: Optional[str]) -> Optional[str]:
    """W_dropdown_sets_filter_dirty: kind จาก classify_click_index_disturbance() -> ข้อความตัด chain
    (actions.py classify ครั้งเดียวแล้วใช้ทั้งตัด chain และยกธง filter dirty)"""
    # trigger_weak ใช้ข้อความเดียวกับ trigger — ไม่ chain ต่อปลอดภัยเสมอ (dropdown หรือ tab ก็ตาม)
    return _CHAIN_HINT_BY_KIND.get("trigger" if kind == "trigger_weak" else kind) if kind else None


async def check_click_invalidates_indexes(page: Page, index: int) -> Optional[str]:
    """W_chain_stale_index (live บน OrangeHRM): then_click_index ใช้ได้กับปุ่มที่เห็นอยู่แล้วตอนแจก index
    เท่านั้น — คลิก trigger สร้างตัวเลือกใหม่ / คลิกตัวเลือกทำให้ index เลื่อน chain จึงชี้ผิดตัวเสมอ
    (run จริง: วน 5 รอบ filter ไม่ติด และครั้งหนึ่ง "สำเร็จ" ไปโดน Profile/Account Menu)
    W_chain_stale_index_kind (live ebeec1c6): เวอร์ชัน bool บอก "dropdown OPEN" ตอนเพิ่งปิด วน 13 ครั้ง
    -> แยก option/trigger และเช็ค option ก่อนเสมอ

    คืน None (chain ต่อได้) ถ้าเป็น <select> จริง/ไม่เกี่ยวกับ dropdown/อ่าน DOM ไม่ได้"""
    return _CHAIN_HINT_BY_KIND.get(await classify_click_index_disturbance(page, index))


async def classify_click_index_disturbance(page: Page, index: int) -> Optional[str]:
    """W_dropdown_sets_filter_dirty: kind ดิบ — "trigger"/"trigger_weak" (คลิกนี้เปิด dropdown),
    "option" (เลือกตัวเลือกใน dropdown ที่เปิดอยู่) หรือ None (ไม่เกี่ยว/<select> จริง/อ่านไม่ได้)
    ต้องเรียก *ก่อน* dispatch — หลังคลิก dropdown ปิดไปแล้วแยก trigger/option ไม่ได้
    orchestrator ใช้ยกธง filter_dirty_since_search (ดู ActionResult.dropdown_option_selected)"""
    try:
        kind = await _evaluate(page, index, _INDEX_DISTURBING_CLICK_JS)
    except Exception:
        return None
    # เทียบกับ dict ตรงๆ — ค่าที่ไม่รู้จัก (mock/หน้า error) ต้องไม่ถูกตีความเป็น dropdown
    return kind if isinstance(kind, str) and kind in _CHAIN_HINT_BY_KIND else None


# W_inner_scroll: layout แบบ app-shell (body สูงเท่าจอ pane ข้างในเป็น overflow:auto) ทำให้
# window.scrollY=0 และ atBottom=true ตลอด -> scroll ทุกครั้งถูก skip "Already scrolled to the bottom"
# ใช้ไม่ได้ทั้งกลุ่ม dashboard/mail/chat/grid หา "ตัวที่ scroll จริง" ก่อนเสมอ และแชร์กับ
# actions.py::scroll() ผ่าน SCROLL_BY_JS ให้ตัวที่เช็คกับตัวที่เลื่อนเป็นตัวเดียวกัน
_FIND_SCROLLER_FN_JS = """
    const findScroller = () => {
      const doc = document.scrollingElement || document.documentElement;
      if (doc && doc.scrollHeight > doc.clientHeight + 2) return doc;
      let best = null;
      let bestArea = 0;
      for (const el of document.querySelectorAll('*')) {
        // เช็คตัวเลขก่อน getComputedStyle เสมอ — element ที่เลื่อนได้จริงมีน้อยมากต่อหน้า
        // การเรียก getComputedStyle ทุก element จะช้าโดยไม่จำเป็นบนหน้าที่มี element เยอะ
        if (el.scrollHeight <= el.clientHeight + 2) continue;
        const st = window.getComputedStyle(el);
        if (!/(auto|scroll)/.test(st.overflowY)) continue;
        const rect = el.getBoundingClientRect();
        const area = rect.width * rect.height;
        if (area > bestArea) { bestArea = area; best = el; }
      }
      return best || doc;
    };
"""

_SCROLL_EDGE_JS = "(dir) => {" + _FIND_SCROLLER_FN_JS + """
    const el = findScroller();
    const atBottom = (el.scrollTop + el.clientHeight) >= (el.scrollHeight - 2);
    const atTop = el.scrollTop <= 0;
    return dir === 'down' ? atBottom : atTop;
}"""

# actions.py::scroll() — คืนตำแหน่งก่อน/หลังให้รายงานได้ตามจริงว่าเลื่อนไปกี่พิกเซล
SCROLL_BY_JS = "(dy) => {" + _FIND_SCROLLER_FN_JS + """
    const el = findScroller();
    const before = el.scrollTop;
    el.scrollTop = before + dy;
    return { before: before, after: el.scrollTop };
}"""


async def check_scroll_redundant(page: Page, direction: str) -> Optional[str]:
    """REDUNDANT ถ้าอยู่สุดขอบตามทิศทางแล้ว — เทียบ `is True` เพราะค่าที่ไม่ใช่ bool จริง (mock/error)
    ต้องไม่ถูกตีความว่าอยู่ขอบแล้ว"""
    try:
        at_edge = await page.evaluate(_SCROLL_EDGE_JS, direction)
    except Exception:
        return None
    if at_edge is True:
        edge_label = "bottom" if direction == "down" else "top"
        return f"Already scrolled to the {edge_label} of the page"
    return None


# W_submit_before_confirm_password (รันสด 2026-09-03): fill index 22 พ่วง Enter + then_click_index 25
# ขณะ Confirm Password ยังว่าง -> ส่งฟอร์มไม่ครบ ได้ 'Passwords do not match' ปนกับ error จริง
# ตัดเฉพาะส่วนที่พ่วง (การกรอกถูกแล้ว) ตัดสินจาก DOM ล้วน อ่านแค่ "ว่างหรือไม่ว่าง" ไม่อ่านค่า
_OTHER_EMPTY_PASSWORD_FIELDS_JS = """(el) => {
    if (!el || (el.getAttribute('type') || '').toLowerCase() !== 'password') return 0;
    return Array.from(document.querySelectorAll('input[type="password"]')).filter(
        other => other !== el
            && other.getClientRects().length > 0
            && !(other.value || '').trim()
    ).length;
}"""


async def check_fill_submits_with_password_fields_left_empty(
    page: Page, index: int,
) -> Optional[str]:
    """เหตุผลที่ต้องตัดส่วนที่พ่วงมากับ fill นี้ทิ้ง — None ถ้าไม่มีปัญหาหรืออ่าน DOM ไม่ได้"""
    try:
        remaining = await _evaluate(page, index, _OTHER_EMPTY_PASSWORD_FIELDS_JS)
    except Exception:
        return None
    if not remaining:
        return None
    return (
        f"the value was typed in, but this form still has {remaining} empty password field(s) "
        "— submitting now would fail validation ('passwords do not match') and hide the real "
        "problem. Fill every remaining password field first, then submit as a separate step"
    )


# W_click_submits_with_empty_password_fields (REST API 2026-09-04): โมเดลเลี่ยง guard ข้างบนด้วยการกด
# Save เป็น action แยก (กรอกแค่ Current Password) -> ผูกกับ <form> เดียวกันเสมอ ไม่ใช่ทั้งหน้า
# (ปุ่ม Save ของฟอร์มอื่นต้องไม่โดน) และปุ่ม type="button" (Cancel) ไม่นับเป็นการส่ง
# W_password_confirm_mismatch (REST API 2026-09-04): เทิร์นสองกรอกรหัสใหม่ทับเฉพาะช่อง Password ช่อง
# Confirm ค้างค่าเดิม -> เทียบค่าในหน้าเว็บเท่านั้น คืนแค่ kind+index ห้ามส่งค่ารหัสออกมา
# (เหตุผลเดียวกับ W_password_value_leaks_into_label) ช่องรหัสปัจจุบันถูกคัดออกก่อนเทียบเสมอ
CURRENT_PASSWORD_LABEL_HINTS = (
    "current password", "old password", "existing password",
    "รหัสผ่านปัจจุบัน", "รหัสผ่านเดิม",
)

# W_password_field_has_no_label_attributes: บางเว็บไม่มี label/name/id เลย <label> เป็นพี่น้องในกล่อง
# ครอบ จึงต้องเดินขึ้น ancestor หา
PASSWORD_FIELD_LABEL_JS = r"""el => {
    const direct = (
        (el.labels && el.labels[0] && el.labels[0].innerText) ||
        el.getAttribute('aria-label') || el.getAttribute('placeholder') ||
        el.getAttribute('name') || el.id || ''
    );
    if (direct.trim()) return direct.toLowerCase();
    const byIds = (el.getAttribute('aria-labelledby') || '').split(/\s+/).filter(Boolean)
        .map(id => (document.getElementById(id) || {}).innerText || '').join(' ');
    if (byIds.trim()) return byIds.toLowerCase();
    let node = el;
    for (let i = 0; i < 4 && node; i++) {
        node = node.parentElement;
        if (!node) break;
        const lab = node.querySelector('label');
        if (lab && lab.innerText.trim()) return lab.innerText.toLowerCase();
    }
    return '';
}"""

_PASSWORD_FORM_SUBMIT_PROBLEM_JS = r"""(el, hints) => {
    const tag = (el.tagName || '').toLowerCase();
    const type = (el.getAttribute('type') || '').toLowerCase();
    const submits = (tag === 'button' && type !== 'button' && type !== 'reset')
        || (tag === 'input' && (type === 'submit' || type === 'image'));
    if (!submits) return null;
    const form = el.closest('form');
    if (!form) return null;
    const labelOf = """ + PASSWORD_FIELD_LABEL_JS + r""";
    const fields = Array.from(form.querySelectorAll('input[type="password"]'))
        .filter(f => f.getClientRects().length > 0);
    if (!fields.length) return null;
    const empty = fields.filter(f => !(f.value || '').trim())
        .map(f => f.getAttribute('data-ai-index'));
    if (empty.length) return {kind: 'empty', indexes: empty};
    const fresh = fields.filter(f => !hints.some(h => labelOf(f).includes(h)));
    if (fresh.length >= 2 && !fresh.every(f => f.value === fresh[0].value)) {
        return {kind: 'mismatch', indexes: fresh.map(f => f.getAttribute('data-ai-index'))};
    }
    return null;
}"""


async def password_form_submit_problem(page: Page, index: int):
    """เหตุผลที่ยังส่งฟอร์มรหัสผ่านนี้ไม่ได้ — {"kind": "empty"|"mismatch", "indexes": [...]}
    None ถ้าไม่ใช่ปุ่มส่งฟอร์ม/ไม่มี <form>/ไม่มีช่องรหัสผ่าน/อ่าน DOM ไม่ได้"""
    try:
        return await _evaluate(
            page, index, _PASSWORD_FORM_SUBMIT_PROBLEM_JS, list(CURRENT_PASSWORD_LABEL_HINTS),
        )
    except Exception:
        return None

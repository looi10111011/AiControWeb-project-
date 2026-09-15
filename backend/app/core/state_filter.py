"""core/state_filter.py — W19 (ดู W19.txt ข้อ 6 "Deterministic State Filter"): เช็คว่า
proposed action "จำเป็นจริงไหม" ก่อน dispatch จริงใน actions.py::execute() — ไม่พึ่ง LLM
เลย (deterministic ล้วนๆ เหมือน permission/rules.py) กัน round-trip ไปเบราว์เซอร์เปล่าๆ ตอน
สถานะปัจจุบันตรงกับที่ต้องการอยู่แล้ว (fill ข้อความเดิมซ้ำ, check checkbox ที่ติ๊กอยู่แล้ว,
scroll ทั้งที่สุดหน้าแล้ว, click element ที่ disabled ไปแล้ว)

ต่างจาก actions.py::_dispatch_with_retry (W5) ตรงที่ตัวนั้นแก้ปัญหา DOM ไม่นิ่ง (retry เพื่อ
ให้ "สำเร็จ") ส่วนตัวนี้ตัดสินว่า action "ไม่ต้องทำเลย" เพราะเป้าหมายบรรลุอยู่แล้ว/ทำไม่ได้
แน่นอน — คนละปัญหากัน ไม่ทับซ้อนกัน เรียกจาก execute() ก่อน dispatch จริงเสมอ (เฉพาะ type
ที่เช็คได้ตรงไปตรงมา: fill/check/click/scroll)

ห้าม throw ออกไปให้ execute() พังเด็ดขาด — error ระหว่างเช็ค (element หาย/frame ปิด/mock ที่
ไม่ได้ config ค่าไว้ตอนเทสต์ ฯลฯ) ถือว่า "ไม่ redundant" เสมอ (คืน None) ปล่อยให้ dispatch
จริงไปเจอ error ของตัวเองตามปกติ — ปลอดภัยกว่าการเดาว่า redundant ทั้งที่เช็คสถานะจริงไม่ได้"""

from typing import Optional

from playwright.async_api import Page

from backend.app.core.perception import resolve_frame

_STATE_CHECK_TIMEOUT_MS = 500


def _sel(index: int) -> str:
    return f'[data-ai-index="{index}"]'


async def check_fill_redundant(page: Page, index: int, text: str) -> Optional[str]:
    """REDUNDANT ถ้าช่อง input/textarea มีข้อความ = text อยู่แล้วเป๊ะ (fill ซ้ำไม่มีผล
    อะไรเพิ่ม แถมเสี่ยง trigger event ซ้ำโดยไม่จำเป็น)"""
    try:
        selector = _sel(index)
        target = await resolve_frame(page, selector)
        current = await target.locator(selector).input_value(timeout=_STATE_CHECK_TIMEOUT_MS)
    except Exception:
        return None
    if current == text:
        return f"This field already contains '{text}' — no need to fill it again"
    return None


async def check_fill_is_empty_noop(page: Page, index: int, text: str) -> Optional[str]:
    """W_empty_fill_noop (บั๊กจริงจาก live run ของ goal user เอง): โมเดลสั่ง fill ด้วย
    text="" ลงช่องที่ว่างอยู่แล้ว 3 step ติดกัน (index 21/22/23) — check_fill_redundant()
    ด้านบนจับได้ถูกต้องว่า "ช่องนี้มี '' อยู่แล้ว" แต่ execute() คืนเป็น success=True ทำให้
    โมเดลอ่านว่า "ทำสำเร็จ" แล้วเดินหน้าสั่งช่องถัดไปแบบเดียวกันต่อ เสีย step ฟรีไปเรื่อยๆ
    โดยไม่มีสัญญาณอะไรบอกว่ามันกำลังทำสิ่งที่ไม่มีความหมาย

    แยกออกมาจาก check_fill_redundant() เพราะ "กรอกค่าเดิมซ้ำ" กับ "กรอกค่าว่างลงช่องว่าง"
    คนละเรื่องกัน: อย่างแรกคือเป้าหมายบรรลุแล้วจริง (success ถูกต้อง) อย่างหลังคือ action ที่
    ไม่มีความหมายตั้งแต่ต้น ต้องตอบกลับเป็น failure พร้อมบอกทางเลือก — pattern เดียวกับ
    check_select_target_is_native() ที่แยก "ใช้ action ผิดชนิด" ออกจาก "ทำไปแล้ว"

    ยังต้องอ่านค่าปัจจุบันจริงก่อน ไม่ตัดสินจาก text=="" อย่างเดียว — fill("") ลงช่องที่ *มี*
    ข้อความอยู่คือการล้างค่า (เช่น เคลียร์ filter) ซึ่งถูกต้องสมบูรณ์ ห้ามบล็อก"""
    if text != "":
        return None
    try:
        selector = _sel(index)
        target = await resolve_frame(page, selector)
        current = await target.locator(selector).input_value(timeout=_STATE_CHECK_TIMEOUT_MS)
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
    """W_custom_dropdown (บั๊กจริง live-reproduce บน OrangeHRM): action "select" ใช้ได้กับ
    <select> จริงเท่านั้น — เว็บสมัยใหม่จำนวนมาก (รวม OrangeHRM) ทำ dropdown ด้วย div/button
    + role=combobox แทน พอ LLM สั่ง select ใส่ element พวกนี้ select_option() จะไล่หา <option>
    ไม่เจอสักตัวแล้วคืน "no option matching ... (no options found in this dropdown)" หลัง retry
    ครบ 3 รอบ — ข้อความนั้นอ่านเหมือน "ตัวเลือกที่ขอไม่มีอยู่" ทั้งที่ปัญหาจริงคือ "ใช้ action
    ผิดชนิด" ทำให้โมเดลไปหลงหาตัวเลือกอื่นแทนที่จะเปลี่ยนวิธีโต้ตอบ

    ผลจริงที่เจอ: filter Role=ESS ไม่เคยถูกตั้งเลย task เลยไม่มีเงื่อนไขจบที่ชัดเจนแล้ววน
    ติ๊ก checkbox ของแถวไปเรื่อยๆ จน user ต้องกด Stop เอง

    คืนข้อความชี้ทางไป protocol W50 ใน SYSTEM_PROMPT (คลิกเปิด dropdown ก่อน แล้วค่อยคลิก
    ตัวเลือกที่ label ตรงเป๊ะ) — fail-safe คืน None ถ้าอ่าน tag ไม่ได้จริงๆ (ปล่อยให้ dispatch
    ตามเดิม ปลอดภัยกว่าบล็อก action ที่อาจถูกต้องอยู่แล้ว)"""
    try:
        selector = _sel(index)
        target = await resolve_frame(page, selector)
        tag = await target.locator(selector).evaluate(
            "el => el.tagName.toLowerCase()", timeout=_STATE_CHECK_TIMEOUT_MS,
        )
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
    """W_click_native_select (บั๊กจริง live-reproduce บน saucedemo 2026-08-26): กระจกบานตรงข้าม
    ของ check_select_target_is_native() ด้านบน — คราวนี้คือสั่ง "click" ใส่ <select> จริง

    ต่างจากเคส select-บน-div ตรงที่เคสนั้นล้มเหลวอย่างเห็นได้ชัด (คืน [FAIL] หลัง retry ครบ)
    แต่เคสนี้ Playwright คลิก <select> ได้สำเร็จจริงและคืน [OK] — ทั้งที่ไม่มีอะไรเกิดขึ้นเลย
    (ค่าที่เลือกอยู่ไม่เปลี่ยน หน้าไม่เปลี่ยน) โมเดลจึงเห็น [OK] แล้วเข้าใจว่าเดินหน้าแล้ว
    วนคลิกซ้ำไปเรื่อยๆ จนโดน loop-detection ฆ่า task ทิ้ง — "สำเร็จแต่ไม่มีผล" อันตรายกว่า
    "ล้มเหลวชัดเจน" เพราะไม่มีสัญญาณอะไรให้โมเดลรู้ตัวเลย

    เหตุการณ์จริง: goal "sort the products by Price (low to high)" บน saucedemo หน้า inventory
    -> click(2) บน <select> ของ sort -> [OK] 3 ครั้งติด -> loop-detected -> task ตายที่ 3 step

    คืน success=False พร้อมชี้ทางไป action ที่ถูกต้อง (select + label) — pattern เดียวกับ
    check_select_target_is_native() เป๊ะ fail-safe คืน None ถ้าอ่าน tag ไม่ได้"""
    try:
        selector = _sel(index)
        target = await resolve_frame(page, selector)
        tag = await target.locator(selector).evaluate(
            "el => el.tagName.toLowerCase()", timeout=_STATE_CHECK_TIMEOUT_MS,
        )
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
    """W_file_input_guard (P3.10): <input type="file"> ถูก index ไปแล้วโดย perception (ตรง
    selector "input" เฉยๆ) โมเดลจึงเห็นและสั่ง fill ใส่ได้ — แต่ Playwright ไม่ยอมให้ fill()
    ลง file input (ต้องใช้ set_input_files) จึง throw แล้วเสีย retry ครบ 3 รอบทุกครั้ง โดย
    ข้อความ error ที่ได้ไม่ได้บอกเลยว่า "ต้องใช้วิธีอื่น"

    *** ตั้งใจไม่เพิ่ม action อัปโหลดไฟล์ *** — action แบบนั้นแปลว่า agent เลือกไฟล์ในเครื่อง
    ผู้ใช้เองได้จาก path ที่โมเดลแต่งขึ้นมา ซึ่งเป็นการเปิดช่องอ่านไฟล์ในเครื่องโดยไม่มีใคร
    ยืนยัน เป็นการตัดสินใจเชิงความปลอดภัยที่ต้องให้เจ้าของโปรเจกต์เลือกเอง ไม่ใช่ผลพลอยได้ของ
    การแก้บั๊ก — ตรงนี้แค่บอกความจริงว่าทำไม่ได้และให้ทางออกที่ปลอดภัย (ขอไฟล์จาก user)

    fail-safe คืน None ถ้าอ่าน DOM ไม่ได้ (pattern เดียวกับทุกฟังก์ชันในไฟล์นี้)"""
    try:
        selector = _sel(index)
        target = await resolve_frame(page, selector)
        input_type = await target.locator(selector).evaluate(
            "el => (el.tagName || '').toLowerCase() === 'input' ? (el.type || '') : ''",
            timeout=_STATE_CHECK_TIMEOUT_MS,
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
    """W_fill_untypable_target (วัดบนฟอร์ม Add Candidate ของ OrangeHRM 2026-09-08): agent
    พยายามพิมพ์อีเมลลงตัวเปิด dropdown (div.oxd-select-text-input) แล้วได้ error ดิบของ
    Playwright กลับไป ที่ขึ้นต้นว่า "Element is not an <input>, <textarea>, <select> or",
    ซึ่งบอกว่าอะไรผิดแต่ไม่บอกว่าต้องทำอะไรต่อ

    กระจกบานเดียวกับ check_click_target_is_native_select() (W_click_native_select) แค่สลับข้าง:
    ตัวนั้นคือสั่ง click กับ <select> จริง ตัวนี้คือสั่ง fill กับของที่ไม่ใช่ช่องกรอก คืนทางออก
    ที่ทำได้จริงแทน (คลิกเปิดเมนูแล้วเลือกตัวเลือก — เส้นทางเดียวกับ W50)

    ไม่ปฏิเสธ wrapper ที่มีช่องกรอกอยู่ข้างใน เพราะ fill() แก้ให้เองอยู่แล้ว
    fail-safe คืน None ถ้าอ่าน DOM ไม่ได้ (กฎเดียวกับทุกฟังก์ชันในไฟล์นี้)"""
    try:
        selector = _sel(index)
        target = await resolve_frame(page, selector)
        kind = await target.locator(selector).evaluate(
            _TYPABLE_TARGET_JS, timeout=_STATE_CHECK_TIMEOUT_MS,
        )
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
    """เป้าหมายถูกเมนู/รายการตัวเลือกที่เปิดค้างอยู่บังไว้หรือเปล่า

    W_menu_overlay_blocks_target (ทำซ้ำบนฟอร์ม Add Candidate ของ OrangeHRM 2026-09-08):
    กด Tab ท้ายช่อง Last Name ทำให้โฟกัสไปตกที่ dropdown Vacancy แล้วเมนูของมันเปิดคลุม
    ช่อง Email ที่อยู่ถัดลงไป — fill/click ช่อง Email จึงรอ actionability จนหมดเวลา 6.6 วินาที
    ต่อครั้ง แล้วโดนบังคับ go_back จนงานพัง ทั้งที่ไม่มีอะไรผิดกับเป้าหมายเลย

    ต่างจาก marker [obscured] ของ perception ตรงที่ตัวนั้นบอกแค่ "ถูกบัง" เฉยๆ (ซึ่งอาจเป็น
    tooltip ที่หายไปเองก่อนถึงเวลาคลิก — ดูคอมเมนต์ใน perception.py) ส่วนตัวนี้ตอบเฉพาะเจาะจง
    ว่าเป็น "เมนูที่เปิดค้างอยู่" ซึ่งปิดได้ด้วย Escape และไม่หายไปเอง

    fail-safe คืน False ถ้าอ่าน DOM ไม่ได้ (กฎเดียวกับทุกฟังก์ชันในไฟล์นี้)"""
    try:
        selector = _sel(index)
        target = await resolve_frame(page, selector)
        covered = await target.locator(selector).evaluate(
            _MENU_OVERLAY_JS, timeout=_STATE_CHECK_TIMEOUT_MS,
        )
        # เทียบ is True ไม่ใช่ bool() — ค่าที่ไม่ใช่ boolean แท้ (mock ในเทสต์ หรือ evaluate
        # ที่คืนอ็อบเจ็กต์แปลกๆ) ต้องแปลว่า "ตอบไม่ได้" = ไม่ขวาง ตามกฎ fail-safe ของไฟล์นี้
        # ไม่ใช่ "ถูกบัง" ซึ่งจะทำให้ยิง Escape มั่วในสถานการณ์ที่อ่านสถานะจริงไม่ได้
        return covered is True
    except Exception:
        return False

async def check_checkbox_redundant(page: Page, index: int) -> Optional[str]:
    """REDUNDANT ถ้า checkbox/radio ถูกติ๊กอยู่แล้ว (action นี้คือ "check" ล้วนๆ ไม่ใช่
    "toggle" — ไม่มีทางทำให้กลายเป็นติ๊กซ้อนสองครั้งจนหลุดเป็น unchecked)"""
    try:
        selector = _sel(index)
        target = await resolve_frame(page, selector)
        already_checked = await target.locator(selector).is_checked(timeout=_STATE_CHECK_TIMEOUT_MS)
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
    """จำนวน checkbox ในกลุ่มเดียวกับ index นี้ — คืน 0 ถ้าเป็นช่องเดี่ยว (ไม่มีกลุ่ม)

    ไม่ใช่การเช็ค redundant เหมือนตัวอื่นในไฟล์นี้ ผู้เรียกใช้ตัดสินว่าจะ "พ่วงปุ่ม submit
    ต่อท้ายการติ๊ก" ได้ไหม (ดู W_chained_submit_after_check ใน actions.py) — กฎการห้าม throw
    และการ fail-safe (ตรงนี้คือคืน 0 = ไม่ขวางอะไร) ยังเหมือนกันทุกประการ

    ตั้งใจรายงานแค่ "ขนาดกลุ่ม" ไม่ใช่ "ช่องไหนยังไม่ถูกติ๊ก" — เวอร์ชันแรกคืนรายชื่อช่องที่
    ยังว่างแล้วให้ actions เอาไปใส่ข้อความ ผลคือ gate รอบถัดมาโมเดลอ่านรายชื่อนั้นเป็นรายการ
    ที่ต้องทำ แล้วติ๊กช่องที่โจทย์ไม่ได้ขอเพิ่มเพื่อให้ "ครบ" (วัดเจอ 2026-09-07: โจทย์ขอ 3
    จาก 4 ช่อง โมเดลติ๊กครบ 4 ได้คะแนนบางส่วน) ชั้นนี้ไม่รู้จัก goal จึงไม่มีทางรู้ว่าช่องไหน
    ควรถูกติ๊ก — การบอกสถานะไปครึ่งๆ จึงชี้นำผิดมากกว่าไม่บอกเลย"""
    try:
        selector = _sel(index)
        target = await resolve_frame(page, selector)
        size = await target.locator(selector).evaluate(
            _SIBLING_CHECKBOX_JS, timeout=_STATE_CHECK_TIMEOUT_MS,
        )
    except Exception:
        return 0
    return int(size) if isinstance(size, (int, float)) else 0


async def element_is_checkbox(page: Page, index: int) -> bool:
    """element ตัวนี้เป็น checkbox เองหรือเปล่า — ผู้เรียกใช้แยก "พ่วงติ๊กช่องถัดไป" (ตั้งใจ)
    ออกจาก "พ่วงปุ่มส่งฟอร์ม" (บั๊ก) ตอบจาก DOM จริงไม่ใช่จาก metadata ที่ผู้เรียกส่งมา
    เพราะ then_type เป็น optional และ default เป็น "" — เชื่อค่าที่ไม่ได้ส่งมาไม่ได้"""
    try:
        selector = _sel(index)
        target = await resolve_frame(page, selector)
        return bool(await target.locator(selector).evaluate(
            "el => (el.tagName || '').toLowerCase() === 'input'"
            " && (el.type || '').toLowerCase() === 'checkbox'",
            timeout=_STATE_CHECK_TIMEOUT_MS,
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
    """W_check_fires_a_button (release-gate d29ed4a, MiniWoB click-checkboxes): โมเดลสั่ง
    type "check" ใส่ปุ่ม Submit — check() ตกไปทางสำรอง (JS-click แล้วค่อยยืนยันสถานะติ๊ก)
    ผลคือ **มันกดปุ่มนั้นไปจริงๆ** แล้วรายงานว่า "clicked, but the checked state could not
    be confirmed" ซึ่งอ่านเหมือนไม่มีอะไรเกิดขึ้น โมเดลจึงทำซ้ำ (trace ยืนยัน: ฟอร์มถูก
    ส่งไปแล้วตั้งแต่ครั้งที่รายงานว่าล้มเหลว)

    action ที่มี side effect จริงแต่รายงานว่าไม่มีอะไรเกิดขึ้น อันตรายกว่าการเสียเวลา —
    ปฏิเสธก่อน dispatch เหมือน check_click_target_is_native_select() (W_click_native_select)
    และ check_fill_target_is_not_typable() (W_fill_untypable_target) ซึ่งเป็นตระกูลเดียวกัน

    ระวังไม่ปฏิเสธ custom checkbox ที่ check() รองรับอยู่แล้ว (role=checkbox หรือห่อ input
    จริงไว้ข้างใน) fail-safe คืน None ถ้าอ่าน DOM ไม่ได้"""
    try:
        selector = _sel(index)
        target = await resolve_frame(page, selector)
        kind = await target.locator(selector).evaluate(
            _CHECKABLE_TARGET_JS, timeout=_STATE_CHECK_TIMEOUT_MS,
        )
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
    """ข้อความของ element ที่ index นี้ *ตอนนี้* — คืน None ถ้าอ่านไม่ได้/ไม่มีแล้ว

    W_select_reorders_the_page (release-gate 50eefd0, task long_flow): โมเดลสั่ง select
    "Name (Z to A)" พร้อมพ่วงคลิกสินค้า index 10 ต่อในคำสั่งเดียว — การเรียงลำดับสลับ
    ตำแหน่งสินค้าทั้งหน้า index 10 หลัง select จึงเป็นคนละตัวกับที่โมเดลเห็นตอนตัดสินใจ
    ผู้เรียกใช้เทียบข้อความก่อน/หลังเพื่อรู้ว่าเป้าที่พ่วงไว้ยังเป็นตัวเดิมไหม"""
    try:
        selector = _sel(index)
        target = await resolve_frame(page, selector)
        return await target.locator(selector).evaluate(
            _ELEMENT_LABEL_JS, timeout=_STATE_CHECK_TIMEOUT_MS,
        )
    except Exception:
        return None

async def check_click_redundant(page: Page, index: int) -> Optional[str]:
    """REDUNDANT (คลิกไม่ได้จริง) ถ้า element เป้าหมาย disabled ไปแล้ว — perception.py
    กรอง element ที่ disabled อยู่แล้วตอน snapshot ไม่ให้ติด index เลย แต่หน้าอาจเปลี่ยน
    สถานะไปแล้วระหว่างที่ LLM กำลังตัดสินใจ (perceive กับ dispatch ไม่ใช่ atomic กัน)"""
    try:
        selector = _sel(index)
        target = await resolve_frame(page, selector)
        disabled = await target.locator(selector).is_disabled(timeout=_STATE_CHECK_TIMEOUT_MS)
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


# W_menu_open_note_needs_no_chain (บั๊กจริงจากรันสด 2026-09-03 พร้อม log เต็ม): โมเดลสั่ง
# click(36) ที่เมนูโปรไฟล์ แล้วสั่ง click(36) ซ้ำอีก **4 ครั้งติด** จนโดน loop detector บังคับ
# recovery — เสียไป 5 step ก่อนจะเจอ "Change Password" ที่อยู่ในเมนูที่มันเปิดค้างไว้เอง
#
# สาเหตุ: ข้อความ _CHAIN_HINT_BY_KIND ที่อธิบายเรื่องนี้ไว้ครบอยู่แล้ว ถูกแนบเฉพาะตอนที่คำสั่ง
# มี then_click_index เท่านั้น คลิกเปล่าจึงได้ผลลัพธ์แค่ "click succeeded" ซึ่งไม่ได้บอกเลยว่า
# เมนูเปิดอยู่และ index ทั้งชุดเลื่อนไปแล้ว โมเดลจึงใช้ index เดิมซ้ำอย่างสมเหตุสมผลจากข้อมูล
# เท่าที่มัน "เห็น" — ความจริงเรื่อง index เลื่อนไม่ได้ขึ้นกับว่ามี chained click หรือไม่
# ต้องเขียนคนละสำนวนกับชุด chain ข้างบน (ชุดนั้นขึ้นต้นด้วยเหตุผลที่ *ไม่ chain ต่อ*
# ซึ่งไม่มีความหมายเลยเมื่อโมเดลไม่ได้ขอ chain มาตั้งแต่แรก)
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
    """W_menu_open_note_needs_no_chain: ข้อความเดียวกันในเชิงข้อเท็จจริงกับ chain_hint_for_kind()
    แต่สำหรับคลิกที่ *ไม่มี* then_click_index — ดูเหตุผลเต็มในคอมเมนต์เหนือตารางด้านบน

    W_expanded_alone_is_not_a_menu (regression ที่ release gate จับได้ 2026-09-07, งาน MiniWoB
    "click-tab"): tab ของ jQuery UI มี aria-expanded จึงถูกจัดเป็น trigger แล้วได้ข้อความว่า
    "กดซ้ำจะปิดเมนู" ซึ่งผิดสำหรับ tab (กดซ้ำไม่ปิด และการกดแท็บอื่นคือสิ่งที่ต้องทำ) โมเดลจึงวน
    คลิกแท็บเดิมจนโดน loop detector — งานนี้เคยผ่านก่อนหน้านี้
    ตอนที่ note นี้แนบเฉพาะกรณีมี chained click มันแทบไม่เคยยิง พอทำให้แนบทุกครั้งจึงไปโดน
    widget ชนิดอื่นที่ใช้ aria-expanded เหมือนกัน (tab/accordion/disclosure)
    -> แนบเฉพาะสัญญาณที่แรงจริง (role=combobox / aria-haspopup / class ของ select library)
    ส่วน trigger_weak ไม่แนบอะไรเลย ปล่อยให้โมเดลอ่าน snapshot ใหม่ตามปกติ"""
    return _INDEX_SHIFT_NOTE_BY_KIND.get(kind) if kind else None


def chain_hint_for_kind(kind: Optional[str]) -> Optional[str]:
    """W_dropdown_sets_filter_dirty: แปลง kind ที่ classify_click_index_disturbance() คืน
    เป็นข้อความอธิบายให้โมเดล — แยกออกมาเพื่อให้ actions.py เรียก classify ครั้งเดียวแล้วเอา
    ผลไปใช้ทั้งสองทาง (ตัด chain + ยกธง filter dirty) โดยไม่อ่าน DOM ซ้ำ"""
    # trigger_weak ใช้ข้อความเดียวกับ trigger สำหรับการ "ไม่ chain ต่อ" — การไม่ chain
    # ปลอดภัยเสมอไม่ว่าจะเป็น dropdown จริงหรือ tab (index ชุดเดิมใช้ไม่ได้ทั้งคู่)
    return _CHAIN_HINT_BY_KIND.get("trigger" if kind == "trigger_weak" else kind) if kind else None


async def check_click_invalidates_indexes(page: Page, index: int) -> Optional[str]:
    """W_chain_stale_index (บั๊กจริง live-reproduce บน OrangeHRM ด้วย goal ของ user เอง):
    then_click_index ("compound action") ถูกออกแบบมาสำหรับปุ่มที่ *เห็นอยู่แล้วในหน้าเดิม*
    ตอนที่ index ถูกแจก — ดู docstring ของ actions.py::_maybe_chain_click() ที่เขียนเงื่อนไข
    นี้ไว้เอง ("ปุ่ม Submit/OK ที่เห็นอยู่แล้วในหน้าเดิม ไม่ต้อง perceive ใหม่ก่อน")

    การแตะ dropdown ละเมิดเงื่อนไขนั้นทั้งขาไปและขากลับ: คลิก trigger = ตัวเลือกเพิ่งถูกสร้าง
    ขึ้นมาใหม่ (ไม่มี index เดิม) · คลิกตัวเลือก = ตัวเลือกทั้งชุดหายไปจากหน้า (index ที่เหลือ
    เลื่อนหมด) ทั้งสองทางทำให้ then_click_index ชี้ผิดตัวเสมอ

    ผลจริงที่บันทึกไว้ใน run เดียว: click(22) '-- Select --' -> then click(26) วน 5 รอบโดย
    filter ไม่เคยติด, 3 รอบคืน "element not found", และรอบที่ click(24) -> then click(29)
    "สำเร็จ" กลับไปโดน 'Demo Source [Profile/Account Menu]' ที่ไม่เกี่ยวอะไรเลย — ที่แย่กว่านั้น
    คือมันคืน success จึงไม่มีสัญญาณว่าล้มเหลว โมเดลเลยวนซ้ำแบบเดิมจนหมด max_steps

    W_chain_stale_index_kind (บั๊กจริงของ guard นี้เองรอบแรก, live run ebeec1c6): เวอร์ชันแรก
    คืนแค่ bool แล้วใช้ข้อความเดียวว่า "The dropdown is now OPEN" — แต่มันจับการคลิก *ตัวเลือก*
    ด้วย (ไล่ ancestor ไปเจอ role=listbox) ทำให้บอกโมเดลว่า dropdown เปิดอยู่ทั้งที่เพิ่งปิดไป
    โมเดลจึงไปกด '-- Select --' เพื่อ "เปิด" ใหม่ วนอยู่อย่างนั้น 13 ครั้งจนหมด max_steps
    ต้องแยกสองเคสออกจากกันและอธิบายให้ตรงข้าง — เช็ค "เป็นตัวเลือกไหม" ก่อน "เป็น trigger ไหม"
    เสมอ เพราะ library หลายตัววาง menu ไว้ใน wrapper เดียวกับ control

    คืน None (= chain ต่อได้ตามปกติ) ถ้า target เป็น <select> จริง, ไม่เกี่ยวกับ dropdown เลย
    หรืออ่าน DOM ไม่ได้ — fail-safe เหมือนทุกฟังก์ชันในไฟล์นี้"""
    return _CHAIN_HINT_BY_KIND.get(await classify_click_index_disturbance(page, index))


async def classify_click_index_disturbance(page: Page, index: int) -> Optional[str]:
    """W_dropdown_sets_filter_dirty: แกนกลางที่ check_click_invalidates_indexes() ด้านบนใช้อยู่
    — คืน kind ดิบ ("trigger" = คลิกนี้ *เปิด* dropdown / "option" = คลิกนี้ *เลือกตัวเลือก*
    ใน dropdown ที่เปิดอยู่ / None = ไม่เกี่ยวกับ dropdown, เป็น <select> จริง, หรืออ่าน DOM
    ไม่ได้)

    แยกออกมาเป็นฟังก์ชันของตัวเองเพราะมีผู้ใช้ที่สองที่ต้องการ "kind" ไม่ใช่ "ข้อความเตือน":
    orchestrator ต้องรู้ว่าคลิกที่เพิ่งเกิดขึ้นคือการ *เลือกค่าใน filter* หรือเปล่า เพื่อยกธง
    filter_dirty_since_search (ดู actions.py::ActionResult.dropdown_option_selected และจุดใช้
    ธงนั้นใน orchestrator.py) — ต้องอ่าน DOM *ก่อน* dispatch เท่านั้น เพราะหลังคลิกไปแล้ว
    dropdown ปิดไปพร้อมตัวเลือกทั้งชุด สถานะที่ใช้แยก trigger/option ออกจากกันหายไปหมด

    ผลข้างเคียงที่ได้ฟรีคือเรียก evaluate() ครั้งเดียวต่อ click แล้วใช้ผลได้ทั้งสองงาน (ตัด
    then_click_index + ยกธง filter dirty) แทนที่จะอ่าน DOM ซ้ำสองรอบ"""
    try:
        selector = _sel(index)
        target = await resolve_frame(page, selector)
        kind = await target.locator(selector).evaluate(
            _INDEX_DISTURBING_CLICK_JS, timeout=_STATE_CHECK_TIMEOUT_MS,
        )
    except Exception:
        return None
    # เทียบกับ dict ตรงๆ (ไม่ใช่ truthy) — evaluate() ที่คืนค่าที่ไม่ใช่ kind ที่รู้จัก (mock ที่
    # ไม่ได้ config, หน้าที่ error) ต้องไม่ถูกตีความว่าเป็น dropdown โดยไม่ตั้งใจ
    return kind if isinstance(kind, str) and kind in _CHAIN_HINT_BY_KIND else None


# W_inner_scroll (บั๊กจริงที่ทำให้เว็บทั้งกลุ่มใช้ไม่ได้ ไม่ใช่เว็บใดเว็บหนึ่ง): เดิมทั้งการ
# เช็ค "ถึงขอบหรือยัง" และ action scroll เอง มองแค่ window/document เท่านั้น — แต่ layout แบบ
# app-shell (body สูงเท่าจอ แล้ว pane ข้างในเป็น overflow:auto) คือรูปแบบมาตรฐานของ
# dashboard/mail/chat/data grid แทบทุกตัวในโลก บน layout นั้น window.scrollY เป็น 0 เสมอ และ
# document.scrollHeight เท่ากับ innerHeight พอดี => atBottom เป็น true ตลอดเวลา
#
# ผลคือคำสั่ง scroll ทุกครั้งถูก short-circuit เป็น "[Skipped] Already scrolled to the bottom"
# โดยไม่แตะ browser เลยสักครั้ง — agent จึงไปดูเนื้อหาใต้ fold ไม่ได้เลยบนเว็บกลุ่มนี้
#
# หา "ตัวที่ scroll จริง" ก่อนเสมอ: ถ้าหน้าเลื่อนได้ก็ใช้หน้า ถ้าไม่ ให้หา element ที่เลื่อนได้
# และกินพื้นที่จอมากที่สุด (pane หลักของ layout) — แชร์ JS ก้อนนี้กับ actions.py::scroll()
# ผ่าน SCROLL_BY_JS ด้านล่าง เพื่อให้ "ตัวที่เช็ค" กับ "ตัวที่เลื่อนจริง" เป็นตัวเดียวกันเสมอ
# (ถ้าแยกกันจะกลับไปเป็นบั๊กเดิมในรูปแบบใหม่ทันที)
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

# ใช้จาก actions.py::scroll() — เลื่อน "ตัวเดียวกับ" ที่ _SCROLL_EDGE_JS เช็ค แล้วคืนตำแหน่ง
# ก่อน/หลัง ให้ผู้เรียกรายงานได้ตามจริงว่าเลื่อนไปได้จริงกี่พิกเซล (ธีมเดียวกับ
# W_click_native_select: "สำเร็จแต่ไม่มีผล" อันตรายกว่า "ล้มเหลวชัดเจน")
SCROLL_BY_JS = "(dy) => {" + _FIND_SCROLLER_FN_JS + """
    const el = findScroller();
    const before = el.scrollTop;
    el.scrollTop = before + dy;
    return { before: before, after: el.scrollTop };
}"""


async def check_scroll_redundant(page: Page, direction: str) -> Optional[str]:
    """REDUNDANT ถ้าหน้าอยู่สุด บน/ล่าง อยู่แล้วตามทิศทางที่จะเลื่อน — เช็คด้วย
    scrollY/scrollHeight ตรงๆ ไม่ผ่าน LLM (ถูกกว่า/แม่นกว่าให้ LLM เดาจาก element ที่เห็น)

    เทียบ `is True` ตรงๆ (ไม่ใช่ truthy เฉยๆ) เพราะ page.evaluate() ที่ error/คืนค่าที่ไม่ใช่
    bool จริง (เช่น mock ที่ไม่ได้ config เฉพาะตอนเทสต์) ต้องไม่ถูกตีความว่า "อยู่ขอบแล้ว"
    โดยไม่ตั้งใจ — ปลอดภัยกว่าเสมอที่จะปล่อยให้ scroll dispatch จริงถ้าไม่แน่ใจ"""
    try:
        at_edge = await page.evaluate(_SCROLL_EDGE_JS, direction)
    except Exception:
        return None
    if at_edge is True:
        edge_label = "bottom" if direction == "down" else "top"
        return f"Already scrolled to the {edge_label} of the page"
    return None


# W_submit_before_confirm_password (บั๊กจริงจากรันสด 2026-09-03): โมเดลสั่ง
# {'type': 'fill', 'index': 22, 'text': '12345678', 'key': 'Enter', 'then_click_index': 25}
# ขณะที่ช่อง "Confirm Password" (index 23) ยังว่าง — ฟอร์มจึงถูกส่งไปทั้งที่กรอกไม่ครบ แล้วเว็บ
# ตอบ 'Passwords do not match' เพิ่มมาอีกข้อนอกเหนือจาก error จริงที่ต้องแก้ ทำให้ข้อความที่
# ส่งกลับไปให้ user กำกวมว่าต้องแก้อะไรกันแน่
#
# การ "กรอก" นั้นถูกแล้ว ผิดแค่ส่วนที่พ่วงมาส่งฟอร์ม — จึงตัดเฉพาะส่วนที่พ่วง ไม่ปฏิเสธทั้ง
# action (pattern เดียวกับ W_chain_stale_index ที่ตัด then_click_index ทิ้งแล้วบอกเหตุผล)
# ตัดสินจาก DOM ล้วน ไม่ต้องรู้ goal และไม่อ่านค่าในช่องไหนทั้งสิ้น อ่านแค่ "ว่างหรือไม่ว่าง"
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
    """เหตุผลที่ต้องตัดส่วนที่พ่วงมากับ fill นี้ทิ้ง — None ถ้าไม่มีปัญหา

    fail-safe คืน None ถ้าอ่าน DOM ไม่ได้ (เหมือนทุกตัวในไฟล์นี้): ปล่อยให้ทำตามที่โมเดลสั่ง
    ดีกว่าตัดคำสั่งของมันทิ้งเพราะเราอ่านสถานะไม่ออกเอง"""
    try:
        selector = _sel(index)
        target = await resolve_frame(page, selector)
        remaining = await target.locator(selector).evaluate(
            _OTHER_EMPTY_PASSWORD_FIELDS_JS, timeout=_STATE_CHECK_TIMEOUT_MS,
        )
    except Exception:
        return None
    if not remaining:
        return None
    return (
        f"the value was typed in, but this form still has {remaining} empty password field(s) "
        "— submitting now would fail validation ('passwords do not match') and hide the real "
        "problem. Fill every remaining password field first, then submit as a separate step"
    )


# W_click_submits_with_empty_password_fields (บั๊กจริงจากรันสดผ่าน REST API 2026-09-04):
# W_submit_before_confirm_password ปิดทางไว้เฉพาะ fill ที่พ่วง key="Enter"/then_click_index
# โมเดลจึงเลี่ยงด้วยการกด Save เป็น action แยกต่างหาก — step trace: fill_secret ช่อง Current
# Password แล้ว click(25) "Save" ทันที โดยไม่เคยกรอกช่อง Password/Confirm Password เลย ได้
# 'Passwords do not match' กลับมา (ทั้งสองช่องว่าง) task จบที่ 4 steps
#
# ผูกขอบเขตด้วย <form> เดียวกันเสมอ ไม่ใช่ทั้งหน้า: วัดกับหน้าจริงแล้วปุ่ม Save (type=submit)
# กับช่อง password ทั้งสามอยู่ใน form เดียวกัน ส่วนปุ่ม Upgrade/เมนูอยู่นอก form — การนับทั้งหน้า
# จะบล็อกปุ่ม Save ของฟอร์มอื่นที่ไม่เกี่ยวข้องกันเลย
# ปุ่มที่ type="button" (เช่น Cancel) ไม่นับเป็นการส่งฟอร์ม จึงไม่โดนแตะ
# W_password_confirm_mismatch (บั๊กจริงจากรันสดสองเทิร์นผ่าน REST API 2026-09-04): เทิร์นแรก
# กรอก 12345678 ลงทั้งช่อง Password และ Confirm Password แล้วเว็บปฏิเสธเพราะไม่มีตัวพิมพ์เล็ก
# เทิร์นที่สอง user ตอบกลับด้วยรหัสที่ผ่านนโยบาย โมเดลกรอกทับเฉพาะช่อง Password ช่องเดียว ช่อง
# Confirm ยังค้างค่าเดิมอยู่ -> 'Passwords do not match' guard ที่มีอยู่มองแค่ "ว่างหรือไม่ว่าง"
# จึงปล่อยผ่าน ทั้งที่ฟอร์มผิดตั้งแต่ก่อนส่งแล้ว
#
# เทียบค่าภายใน page context ทั้งหมด คืนออกมาแค่ kind กับ index — ห้ามส่งค่าจริงของช่องรหัสผ่าน
# ออกมานอกหน้าเว็บเด็ดขาด (เหตุผลเดียวกับ W_password_value_leaks_into_label)
#
# ช่อง "รหัสผ่านปัจจุบัน" ถูกคัดออกก่อนเทียบเสมอ — ค่าของมันไม่มีเหตุผลใดที่ต้องตรงกับรหัสใหม่
CURRENT_PASSWORD_LABEL_HINTS = (
    "current password", "old password", "existing password",
    "รหัสผ่านปัจจุบัน", "รหัสผ่านเดิม",
)

# W_password_field_has_no_label_attributes: ช่องรหัสผ่านของบางเว็บไม่มี label/name/id เลย
# <label> เป็นพี่น้องอยู่ในกล่องครอบ จึงต้องเดินขึ้น ancestor หา — orchestrator ใช้ตัวเดียวกันนี้
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
    const labelOf = el => {
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
};
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

    คืน None ถ้าไม่ใช่ปุ่มส่งฟอร์ม ไม่มี <form> ครอบ ฟอร์มไม่มีช่องรหัสผ่าน หรืออ่าน DOM ไม่ได้
    (fail-safe เหมือนทุกตัวในไฟล์นี้)"""
    try:
        selector = _sel(index)
        target = await resolve_frame(page, selector)
        return await target.locator(selector).evaluate(
            _PASSWORD_FORM_SUBMIT_PROBLEM_JS,
            list(CURRENT_PASSWORD_LABEL_HINTS),
            timeout=_STATE_CHECK_TIMEOUT_MS,
        )
    except Exception:
        return None

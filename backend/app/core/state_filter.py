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
        if (node.hasAttribute && node.hasAttribute('aria-expanded')) return 'trigger';
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


def chain_hint_for_kind(kind: Optional[str]) -> Optional[str]:
    """W_dropdown_sets_filter_dirty: แปลง kind ที่ classify_click_index_disturbance() คืน
    เป็นข้อความอธิบายให้โมเดล — แยกออกมาเพื่อให้ actions.py เรียก classify ครั้งเดียวแล้วเอา
    ผลไปใช้ทั้งสองทาง (ตัด chain + ยกธง filter dirty) โดยไม่อ่าน DOM ซ้ำ"""
    return _CHAIN_HINT_BY_KIND.get(kind) if kind else None


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

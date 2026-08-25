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


_SCROLL_EDGE_JS = """(dir) => {
    const atBottom = (window.innerHeight + window.scrollY) >= (document.documentElement.scrollHeight - 2);
    const atTop = window.scrollY <= 0;
    return dir === 'down' ? atBottom : atTop;
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

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
        return f"ช่องนี้มีข้อความ '{text}' อยู่แล้ว ไม่ต้องกรอกซ้ำ"
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
        return "checkbox/radio นี้ถูกติ๊กอยู่แล้ว"
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
        return "element นี้อยู่ในสถานะ disabled แล้ว คลิกไม่ได้"
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
        edge_label = "ล่างสุด" if direction == "down" else "บนสุด"
        return f"เลื่อนหน้าจอถึง{edge_label}อยู่แล้ว"
    return None

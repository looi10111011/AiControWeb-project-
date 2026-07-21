"""site_learning/auto_login.py — W15/W17: ตรวจจับ + auto-fill ฟอร์ม login แบบ
deterministic (หา field จาก input_type="password"/username-keyword ล้วนๆ ไม่ใช้ LLM
ตัดสินใจเลย) — ใช้ร่วมกันโดยสองที่:
  - crawler.py (W15 login bootstrap — กรอก+submit ครั้งเดียวตอนเรียนรู้เว็บไซต์ เพื่อผ่าน
    หน้า login แล้วสำรวจต่อได้)
  - core/orchestrator.py (W17 auto-login — ถ้ามี credential เก็บไว้แล้วจาก
    storage.save_credentials() และหน้าปัจจุบันเข้าข่ายเป็นหน้า login จริงตอนเริ่ม task)

แยกออกมาเป็นโมดูลกลางเพราะ crawler.py เอง import core/orchestrator.py อยู่แล้ว (ใช้
Orchestrator._llm_backend()) — ถ้า orchestrator.py import จาก crawler.py กลับไปจะเกิด
circular import ทันที โมดูลนี้ไม่ import ทั้งสองฝั่งเลย ปลอดภัยให้ทั้งคู่ import ได้อิสระ
"""

from typing import Optional

from playwright.async_api import Page

from backend.app.site_learning.schema import PageInfo

_USERNAME_FIELD_KEYWORDS = ("user", "email", "login", "name")
_LOGIN_SUBMIT_KEYWORDS = ("sign in", "log in", "login", "signin", "เข้าสู่ระบบ")


def find_login_fields(page_info: PageInfo) -> tuple[Optional[str], Optional[str]]:
    """หา (username_selector, password_selector) จาก form field ที่ extract มาแล้ว — คืน
    (None, None) ถ้าไม่เจอ password field เลย (ไม่ใช่หน้า login)"""
    password_field = next((f for f in page_info.forms if f.input_type == "password" and f.selector), None)
    if password_field is None:
        return None, None
    username_field = next(
        (
            f for f in page_info.forms
            if f.input_type in ("text", "email") and f.selector
            and any(kw in (f.field_name + f.label + f.placeholder).lower() for kw in _USERNAME_FIELD_KEYWORDS)
        ),
        None,
    )
    if username_field is None:
        # fallback: ฟอร์ม login ธรรมดาส่วนใหญ่มีแค่ 2 ช่อง (user/pass) ไม่ต้องพึ่งชื่อ
        # field ให้ตรง keyword เป๊ะ — เอา text/email field แรกที่ไม่ใช่ password
        username_field = next(
            (f for f in page_info.forms if f.input_type in ("text", "email") and f.selector),
            None,
        )
    if username_field is None:
        return None, None
    return username_field.selector, password_field.selector


def find_login_submit_selector(page_info: PageInfo) -> Optional[str]:
    for b in page_info.buttons:
        text = (b.text or b.aria_label or "").strip().lower()
        if any(kw in text for kw in _LOGIN_SUBMIT_KEYWORDS):
            return b.selector or None
    return None


async def attempt_login(page: Page, page_info: PageInfo, username: str, password: str) -> bool:
    """กรอก username/password แล้วกด sign in — ข้อยกเว้นเดียวที่อนุญาตให้ "submit" ได้
    ระหว่าง crawl (ดู crawler.py หัวไฟล์) หรือครั้งเดียวตอนต้น task จริง (ดู
    orchestrator.py::_maybe_auto_login) คืน True ถ้าลองกด submit สำเร็จจริง (ไม่ได้แปลว่า
    login สำเร็จเสมอไป — ผู้เรียกเช็คผลจริงจากการ re-extract หน้าถัดมาเอง) คืน False ถ้าไม่
    เจอ field/ปุ่มที่จำเป็นครบ หรือ fill/click ล้มเหลว (ไม่ throw ออกไป — 1 หน้า login พัง
    ไม่ควรทำทั้ง caller ล้มไปด้วย)"""
    username_selector, password_selector = find_login_fields(page_info)
    if not username_selector or not password_selector:
        return False
    submit_selector = find_login_submit_selector(page_info)
    if not submit_selector:
        return False
    try:
        await page.fill(username_selector, username, timeout=5000)
        await page.fill(password_selector, password, timeout=5000)
        await page.click(submit_selector, timeout=5000)
        await page.wait_for_load_state("networkidle", timeout=10000)
        return True
    except Exception:
        return False

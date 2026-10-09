"""site_learning/auto_login.py — W15/W17: ตรวจจับ + auto-fill ฟอร์ม login แบบ deterministic (ไม่ใช้ LLM).

ใช้ร่วมโดย crawler.py (W15 login bootstrap ตอนเรียนรู้เว็บ) และ core/orchestrator.py (W17 auto-login ด้วย
credential จาก storage.save_credentials()). แยกเป็นโมดูลกลางกัน circular import (crawler import orchestrator อยู่แล้ว).
"""

import urllib.parse
from typing import Optional

from playwright.async_api import Page, TimeoutError as PWTimeout

from backend.app.site_learning.extractor import extract_page
from backend.app.site_learning.schema import PageInfo

_USERNAME_FIELD_KEYWORDS = ("user", "email", "login", "name")
_LOGIN_SUBMIT_KEYWORDS = ("sign in", "log in", "login", "signin", "เข้าสู่ระบบ")


def find_login_fields(page_info: PageInfo) -> tuple[Optional[str], Optional[str]]:
    """คืน (username_selector, password_selector) หรือ (None, None) ถ้าไม่ใช่หน้า login"""
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
        # fallback: ฟอร์ม login ส่วนใหญ่มีแค่ 2 ช่อง — เอา text/email field แรก
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


def _normalize_url_for_compare(url: str) -> str:
    """ตัด fragment/trailing slash เพื่อเทียบ URL ก่อน-หลัง login — สำเนาย่อของ crawler.py::_normalize_url
    โดยเจตนา (นั่นเป็น private helper ไว้ dedupe BFS queue คนละจุดประสงค์)"""
    parsed = urllib.parse.urlparse(url)
    path = parsed.path.rstrip("/") or "/"
    return urllib.parse.urlunparse((parsed.scheme, parsed.netloc, path, "", parsed.query, ""))


async def verify_login_success(page: Page, pre_login_url: str) -> tuple[bool, str]:
    """W24: เช็คว่า session ผ่านจริงหลัง attempt_login() — ต้อง (1) URL เปลี่ยนจากก่อน submit และ
    (2) หน้าใหม่ไม่มีฟอร์ม login เหลือ (ยังเจอ = ถูก redirect กลับ). คืน (session_ok, reason);
    reason ว่างตอน True และไม่มี password ปน. ไม่ throw"""
    post_login_url = _normalize_url_for_compare(page.url)
    if post_login_url == _normalize_url_for_compare(pre_login_url):
        return False, "URL ไม่เปลี่ยนหลัง submit — เข้าใจว่า login ไม่ผ่าน"
    try:
        post_page_info, _ = await extract_page(page)
    except Exception:
        return False, "อ่านหน้าใหม่หลัง submit ไม่สำเร็จ"
    if find_login_fields(post_page_info) != (None, None):
        return False, "ยังเจอฟอร์ม login (username+password field) อยู่หลัง submit — เข้าใจว่าถูก redirect กลับหน้า login"
    return True, ""


async def login_with_verification(
    page: Page, page_info: PageInfo, username: str, password: str, retries: int = 1,
) -> tuple[bool, str]:
    """attempt_login() + verify_login_success() พร้อม retry (retries ครั้งหลังรอบแรก) สำหรับ caller ที่
    ต้องการแค่ pass/fail (เช่น orchestrator._maybe_auto_login). คืน (True, "") หรือ (False, reason
    ของรอบสุดท้าย). ไม่ throw"""
    reason = ""
    for attempt in range(retries + 1):
        pre_login_url = page.url
        did_login = await attempt_login(page, page_info, username, password)
        if not did_login:
            reason = "กรอกฟอร์ม/กดปุ่ม submit ไม่สำเร็จ (หา field/ปุ่มไม่ครบ หรือ fill/click ล้มเหลว)"
            continue
        session_ok, reason = await verify_login_success(page, pre_login_url)
        if session_ok:
            return True, ""
        if attempt < retries:
            try:
                page_info, _ = await extract_page(page)
            except Exception:
                break
    return False, reason


async def attempt_login(page: Page, page_info: PageInfo, username: str, password: str) -> bool:
    """กรอก username/password แล้วกด sign in — ข้อยกเว้นเดียวที่อนุญาตให้ "submit" ระหว่าง crawl.
    True = กด submit ได้ (ไม่ได้แปลว่า login ผ่าน — ใช้ verify_login_success()); False = หา field/ปุ่ม
    ไม่ครบหรือ fill/click ล้มเหลว. ไม่ throw"""
    username_selector, password_selector = find_login_fields(page_info)
    if not username_selector or not password_selector:
        return False
    submit_selector = find_login_submit_selector(page_info)
    if not submit_selector:
        return False
    try:
        await page.fill(username_selector, username, timeout=5000)
        await page.fill(password_selector, password, timeout=5000)
        pre_click_url = page.url
        await page.click(submit_selector, timeout=5000)
        # SPA ตรวจ credential ด้วย XHR ก่อน route — networkidle ทันทีจะ resolve บนหน้าเดิมแล้ว verify
        # เข้าใจผิดว่า login ไม่ผ่าน; รอ URL เปลี่ยนก่อน (timeout เงียบๆ ให้ verify ตัดสินเอง)
        try:
            await page.wait_for_url(lambda u: u != pre_click_url, timeout=10000)
        except PWTimeout:
            pass
        await page.wait_for_load_state("networkidle", timeout=10000)
        return True
    except Exception:
        return False

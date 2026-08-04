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

import urllib.parse
from typing import Optional

from playwright.async_api import Page, TimeoutError as PWTimeout

from backend.app.site_learning.extractor import extract_page
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


def _normalize_url_for_compare(url: str) -> str:
    """ตัด fragment/trailing slash ให้เทียบ URL ก่อน-หลัง login ได้แม่นยำ (URL ต่างกันแค่
    #section หรือ / ท้ายสุดไม่ควรนับว่าเป็นคนละหน้า) — สำเนาย่อของ crawler.py::
    _normalize_url โดยเจตนา (ไม่ import ข้ามมาเพราะเป็น private helper ของ crawler ที่ใช้
    เพื่อ dedupe BFS queue คนละจุดประสงค์ ไม่ใช่ shared utility และ logic สั้นพอที่จะซ้ำได้
    โดยไม่เสี่ยง drift)"""
    parsed = urllib.parse.urlparse(url)
    path = parsed.path.rstrip("/") or "/"
    return urllib.parse.urlunparse((parsed.scheme, parsed.netloc, path, "", parsed.query, ""))


async def verify_login_success(page: Page, pre_login_url: str) -> tuple[bool, str]:
    """หลัง attempt_login() คืน True แล้ว (แปลว่ากด submit ได้จริง) เรียกตัวนี้ต่อเพื่อเช็คว่า
    session ผ่านจริงไหม ไม่ใช่แค่กด submit ได้ — เกณฑ์ 2 ชั้นที่ต้องผ่านทั้งคู่ (ย้ายมาจาก
    crawler.py::_login_and_continue เดิม (W24) ที่มี logic นี้อยู่ก่อนแล้ว รวมจุดเดียวให้
    core/orchestrator.py::_maybe_auto_login เรียกใช้ร่วมได้ แทนที่จะไม่ verify อะไรเลยแบบ
    เดิม):
      (1) URL เปลี่ยนไปจาก URL ก่อน submit จริง (ไม่ใช่แค่ submit แล้ว reload หน้าเดิม)
      (2) หน้าใหม่ไม่มีฟอร์ม login (username+password field) เหลืออยู่แล้ว (find_login_
          fields คืน (None, None) — ถ้ายังเจอ = โดน redirect กลับมาหน้า login เดิม)

    คืน (session_ok, reason) — reason เป็นข้อความอธิบายเหตุผลตอน session_ok=False เท่านั้น
    (ว่างเปล่าตอน True) ไว้ log/แจ้ง user ต่อได้โดยไม่มี password ปนอยู่เลย ไม่ throw"""
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
    """attempt_login() + verify_login_success() รวมกัน พร้อม retry อัตโนมัติถ้ารอบแรกไม่
    ผ่าน (retries ครั้ง นับแยกจากความพยายามแรก) — ใช้ตอนที่ caller ไม่ต้องการ page_info ของ
    หน้าหลัง login ต่อ (แค่ต้องการรู้ผล pass/fail) เช่น core/orchestrator.py::
    _maybe_auto_login ที่ต่างจาก crawler.py::_login_and_continue ตรงที่ไม่ต้อง record หน้า
    หลัง login ลง manual คืน (True, "") ถ้า login ผ่านจริง (รอบใดก็ได้ใน retries+1 ครั้ง)
    คืน (False, reason) ของความพยายามครั้งสุดท้ายถ้ายังไม่ผ่านแม้ retry ครบแล้ว ไม่ throw"""
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
        pre_click_url = page.url
        await page.click(submit_selector, timeout=5000)
        # เว็บบางแห่ง (เช่น SPA ที่ยิง XHR ตรวจ credential ก่อนค่อย route เปลี่ยนหน้า) มีช่วง
        # หน่วงสั้นๆ ระหว่างกด submit กับ navigation จริงเริ่มต้น — ถ้าเรียก
        # wait_for_load_state("networkidle") ทันทีโดยไม่รอ URL เปลี่ยนก่อน มันอาจ resolve
        # ทันทีเพราะหน้า "เดิม" (ก่อน navigate) ก็ idle อยู่แล้วอยู่แล้ว ทำให้
        # verify_login_success() เห็น URL ยังไม่เปลี่ยนแล้วเข้าใจผิดว่า login ไม่ผ่านทั้งที่
        # จริงๆ แค่ยังไม่ทันเปลี่ยนหน้า — รอ URL เปลี่ยนก่อนเป็นอันดับแรก (เงียบๆ ถ้า timeout
        # เพราะ login ที่ล้มเหลวจริงก็ไม่มีทาง URL เปลี่ยนอยู่ดี ปล่อยให้ verify_login_success
        # ตัดสินจากสถานะสุดท้ายแทน)
        try:
            await page.wait_for_url(lambda u: u != pre_click_url, timeout=10000)
        except PWTimeout:
            pass
        await page.wait_for_load_state("networkidle", timeout=10000)
        return True
    except Exception:
        return False

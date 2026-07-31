import functools
import http.server
import threading

import pytest
from playwright.async_api import async_playwright

from backend.app.site_learning.auto_login import login_with_verification, verify_login_success

# ยิงจริงผ่าน chromium จริงเหมือน test_orchestrator_auto_login.py/test_site_learning_
# crawler.py — verify_login_success()/login_with_verification() พึ่ง extract_page()/
# page.url จริงบน DOM จริง mock ยากกว่าเปิด browser เปล่าตรงๆ

_PAGES = {
    "login.html": """
        <html><body>
          <input type="text" id="username" name="username" placeholder="Username" />
          <input type="password" id="password" name="password" placeholder="Password" />
          <button type="button" onclick="window.location.href='/welcome.html'">Sign In</button>
        </body></html>
    """,
    # login ที่ "กด submit ได้" แต่ redirect กลับมาหน้า login เดิม (เช่น password ผิด) —
    # ต้องถือว่า session_ok=False แม้ attempt_login() จะคืน True (กด submit สำเร็จ)
    "login-wrong-password.html": """
        <html><body>
          <input type="text" id="username" name="username" placeholder="Username" />
          <input type="password" id="password" name="password" placeholder="Password" />
          <button type="button" onclick="window.location.href='/login-wrong-password.html?err=1'">Sign In</button>
        </body></html>
    """,
    "welcome.html": "<html><body>Welcome!</body></html>",
}


@pytest.fixture
def fixture_server(tmp_path):
    for name, html in _PAGES.items():
        (tmp_path / name).write_text(html, encoding="utf-8")
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(tmp_path))
    httpd = http.server.HTTPServer(("127.0.0.1", 0), handler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    httpd.shutdown()


# --- verify_login_success() ---


@pytest.mark.asyncio
async def test_verify_login_success_true_when_url_changed_and_no_login_form_left(fixture_server):
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        try:
            await page.goto(f"{fixture_server}/login.html")
            pre_url = page.url
            await page.click("button")  # จำลอง submit สำเร็จ -> ไป welcome.html

            session_ok, reason = await verify_login_success(page, pre_url)

            assert session_ok is True
            assert reason == ""
        finally:
            await browser.close()


@pytest.mark.asyncio
async def test_verify_login_success_false_when_url_unchanged(fixture_server):
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        try:
            await page.goto(f"{fixture_server}/login.html")
            pre_url = page.url  # ไม่ navigate ไปไหนเลย จำลอง submit ไม่สำเร็จ

            session_ok, reason = await verify_login_success(page, pre_url)

            assert session_ok is False
            assert "URL ไม่เปลี่ยน" in reason
        finally:
            await browser.close()


@pytest.mark.asyncio
async def test_verify_login_success_false_when_redirected_back_to_login_form(fixture_server):
    """URL เปลี่ยนจริง แต่หน้าใหม่ยังมีฟอร์ม login (username+password) เหลืออยู่ — เข้าใจว่า
    โดน redirect กลับหน้า login เดิม (เช่น password ผิด) ต้องไม่นับว่า login สำเร็จ"""
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        try:
            await page.goto(f"{fixture_server}/login-wrong-password.html")
            pre_url = page.url
            await page.click("button")

            session_ok, reason = await verify_login_success(page, pre_url)

            assert session_ok is False
            assert "redirect กลับหน้า login" in reason
        finally:
            await browser.close()


# --- login_with_verification() ---


@pytest.mark.asyncio
async def test_login_with_verification_returns_true_on_first_success(fixture_server):
    from backend.app.site_learning.extractor import extract_page

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        try:
            await page.goto(f"{fixture_server}/login.html")
            page_info, _ = await extract_page(page)

            ok, reason = await login_with_verification(page, page_info, "alice", "s3cr3t")

            assert ok is True
            assert reason == ""
            assert "welcome.html" in page.url
        finally:
            await browser.close()


@pytest.mark.asyncio
async def test_login_with_verification_returns_false_after_retries_exhausted(fixture_server):
    """หน้าที่ redirect กลับ login form เสมอ (จำลอง password ผิดค้าง) — ต้องลองครบ
    retries+1 ครั้งแล้วคืน False พร้อมเหตุผล ไม่ throw"""
    from backend.app.site_learning.extractor import extract_page

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        try:
            await page.goto(f"{fixture_server}/login-wrong-password.html")
            page_info, _ = await extract_page(page)

            ok, reason = await login_with_verification(page, page_info, "alice", "wrong", retries=1)

            assert ok is False
            assert reason != ""
        finally:
            await browser.close()


@pytest.mark.asyncio
async def test_login_with_verification_never_raises_when_password_field_missing(fixture_server):
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        try:
            await page.goto(f"{fixture_server}/welcome.html")
            from backend.app.site_learning.extractor import extract_page
            page_info, _ = await extract_page(page)

            ok, reason = await login_with_verification(page, page_info, "alice", "s3cr3t")

            assert ok is False
        finally:
            await browser.close()

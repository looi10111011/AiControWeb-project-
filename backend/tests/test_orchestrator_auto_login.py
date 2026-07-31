import functools
import http.server
import threading

import pytest
from playwright.async_api import async_playwright

from backend.app.config import settings
from backend.app.core.orchestrator import _maybe_auto_login
from backend.app.site_learning import storage

# W17: เทสต์กลุ่มนี้ยิงจริงผ่าน chromium จริง (ไม่ mock Playwright) เหมือน
# test_site_learning_crawler.py — _maybe_auto_login() พึ่ง extract_page()/page.fill()/
# page.click() จริงบน DOM จริง mock ยากกว่าเปิด browser เปล่าตรงๆ

_PAGES = {
    "login.html": """
        <html><body>
          <input type="text" id="username" name="username" placeholder="Username" />
          <input type="password" id="password" name="password" placeholder="Password" />
          <button type="button" onclick="window.location.href='/welcome.html'">Sign In</button>
        </body></html>
    """,
    # login ที่ "กด submit ได้" แต่ redirect กลับมาหน้า login เดิม (เช่น credential ที่
    # บันทึกไว้ใช้ไม่ได้แล้ว/password ถูกเปลี่ยน) — ต้องคืนเหตุผล fail ไม่ใช่ None
    "login-fails.html": """
        <html><body>
          <input type="text" id="username" name="username" placeholder="Username" />
          <input type="password" id="password" name="password" placeholder="Password" />
          <button type="button" onclick="window.location.href='/login-fails.html?err=1'">Sign In</button>
        </body></html>
    """,
    "welcome.html": "<html><body>Welcome!</body></html>",
    "no-login.html": "<html><body><p>nothing to see here — no password field</p></body></html>",
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


@pytest.fixture(autouse=True)
def _isolated_manuals_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "site_manuals_dir", str(tmp_path / "manuals"))
    yield


@pytest.mark.asyncio
async def test_maybe_auto_login_fills_and_submits_when_credentials_stored(fixture_server):
    """มี credential เก็บไว้สำหรับโดเมนนี้แล้ว + หน้าปัจจุบันเป็นหน้า login จริง (มี
    password field) — ต้องกรอก+กด submit ให้เอง จนหลุดไปหน้าหลัง login สำเร็จ คืน None
    (ไม่มีเหตุผล fail ให้ caller แจ้ง user ต่อ)"""
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        try:
            await page.goto(f"{fixture_server}/login.html")
            storage.save_credentials("127.0.0.1", "alice", "s3cr3t")

            result = await _maybe_auto_login(page, verbose=False)

            assert result is None
            assert "welcome.html" in page.url
        finally:
            await browser.close()


@pytest.mark.asyncio
async def test_maybe_auto_login_returns_failure_reason_when_login_does_not_verify(fixture_server):
    """credential เก็บไว้ใช้ไม่ได้แล้ว (submit สำเร็จแต่ redirect กลับมาหน้า login เดิม) —
    ต้องคืนข้อความเหตุผล (ไม่ใช่ None) ให้ run_task() ยิง SSE แจ้ง user ต่อ ไม่ throw"""
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        try:
            await page.goto(f"{fixture_server}/login-fails.html")
            storage.save_credentials("127.0.0.1", "alice", "wrong-password")

            result = await _maybe_auto_login(page, verbose=False)

            assert isinstance(result, str)
            assert result != ""
            assert "wrong-password" not in result  # ห้ามมี password ปนออกมาในเหตุผล
        finally:
            await browser.close()


@pytest.mark.asyncio
async def test_maybe_auto_login_does_nothing_without_stored_credentials(fixture_server):
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        try:
            await page.goto(f"{fixture_server}/login.html")

            result = await _maybe_auto_login(page, verbose=False)

            assert result is None
            assert page.url.endswith("/login.html")  # ไม่ navigate ไปไหนเลย
        finally:
            await browser.close()


@pytest.mark.asyncio
async def test_maybe_auto_login_does_nothing_when_current_page_is_not_a_login_form(fixture_server):
    """มี credential เก็บไว้ แต่หน้าปัจจุบันไม่มี password field เลย (ไม่ใช่หน้า login) —
    ต้องไม่แตะอะไรเลย"""
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        try:
            await page.goto(f"{fixture_server}/no-login.html")
            storage.save_credentials("127.0.0.1", "alice", "s3cr3t")

            result = await _maybe_auto_login(page, verbose=False)

            assert result is None
            assert page.url.endswith("/no-login.html")
        finally:
            await browser.close()


@pytest.mark.asyncio
async def test_maybe_auto_login_never_raises_on_blank_page():
    """about:blank ไม่มี domain ที่ใช้ได้เลย (extract_domain คืนค่าว่าง) — ต้องไม่ throw
    ออกมาทำ run_task() ทั้งก้อนล้ม"""
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        try:
            result = await _maybe_auto_login(page, verbose=False)  # ต้องไม่ raise
            assert result is None
        finally:
            await browser.close()

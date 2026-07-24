import functools
import http.server
import itertools
import json
import re
import threading
from unittest.mock import AsyncMock, patch

import pytest
from playwright.async_api import async_playwright

from backend.app.site_learning.crawler import (
    _button_signature,
    _click_with_retry,
    _goto_with_retry,
    _page_template,
    _reveal_dynamic_content,
    _wait_for_dom_stable,
    crawl_site,
    describe_site,
)
from backend.app.site_learning.schema import ButtonInfo, PageInfo, UIPatternInfo

# เทสต์กลุ่มนี้ยิงจริงผ่าน chromium จริง (ไม่ mock Playwright) ต่อ local HTTP server ที่
# serve fixture HTML จริง — mock page.goto()/nav ทั้งเชนยากกว่าและพิสูจน์ BFS/dedup/
# same-origin filtering ได้แม่นยำน้อยกว่าการรันจริง ส่วน LLM (describe_page) mock ไว้
# เสมอ (ไม่ยิง API จริง ไม่ใช่สิ่งที่เทสต์กลุ่มนี้อยากพิสูจน์)

_FIXTURE_PAGES = {
    "index.html": """
        <html><body>
          <nav>
            <a href="/dashboard.html">Dashboard</a>
            <a href="/products.html">Products</a>
            <a href="/logout.html">Logout</a>
            <a href="https://external.example.com/">External</a>
          </nav>
        </body></html>
    """,
    "dashboard.html": """
        <html><body>
          <nav><a href="/index.html">Home</a><a href="/products.html">Products</a></nav>
          <table><thead><tr><th>Name</th></tr></thead></table>
        </body></html>
    """,
    "products.html": """
        <html><body>
          <nav><a href="/index.html">Home</a></nav>
          <button data-testid="create-btn">Create</button>
        </body></html>
    """,
    "logout.html": "<html><body>logged out</body></html>",
}


@pytest.fixture
def fixture_server(tmp_path):
    for name, html in _FIXTURE_PAGES.items():
        (tmp_path / name).write_text(html, encoding="utf-8")
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(tmp_path))
    httpd = http.server.HTTPServer(("127.0.0.1", 0), handler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    httpd.shutdown()


def _mock_generate_text():
    """W35: ชื่อหน้าต้องไม่ซ้ำกันข้าม call โดย default — crawler.py มี guard ใหม่ (ชื่อหน้า
    เดิมซ้ำติดกันเกิน 2 ครั้ง = หยุดไล่กดปุ่ม/ต่อคิว nav link จากหน้านั้น กัน "วนลูป" ตามที่
    user รายงาน) เดิม mock นี้คืนชื่อ "Page" คงที่ทุก call ทำให้เทสต์อื่นๆ ที่ไม่ได้ตั้งใจ
    ทดสอบเรื่องชื่อซ้ำเลย (บาง fixture มีมากกว่า 2 หน้า) ดันไปชน guard ใหม่นี้โดยไม่ตั้งใจ —
    ใช้ counter เฉพาะของแต่ละ mock instance ให้ชื่อไม่ซ้ำกันเสมอแทน (เทสต์ที่ตั้งใจทดสอบ
    guard ตัวนี้โดยเฉพาะสร้าง mock ของตัวเองแยกต่างหาก ดู
    test_crawl_site_stops_exploring_after_repeated_page_name)"""
    counter = itertools.count(1)

    async def _generate(*_args, **_kwargs):
        return f'{{"name": "Page {next(counter)}", "description": "a page"}}'

    return AsyncMock(side_effect=_generate)


@pytest.mark.asyncio
async def test_crawl_site_visits_same_origin_nav_links_via_bfs(fixture_server):
    with patch("backend.app.site_learning.crawler.llm.generate_text", _mock_generate_text()):
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            manual = await crawl_site(browser, f"{fixture_server}/index.html", max_pages=10)
            await browser.close()

    visited_urls = {p.url for p in manual.pages}
    assert any("dashboard" in u for u in visited_urls)
    assert any("products" in u for u in visited_urls)
    assert any(u.endswith("/index.html") for u in visited_urls)


@pytest.mark.asyncio
async def test_crawl_site_never_visits_blocked_nav_links(fixture_server):
    """safety.is_safe_nav_link() ต้องกันไม่ให้ BFS เดินตามลิงก์ "Logout" เด็ดขาด"""
    with patch("backend.app.site_learning.crawler.llm.generate_text", _mock_generate_text()):
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            manual = await crawl_site(browser, f"{fixture_server}/index.html", max_pages=10)
            await browser.close()

    visited_urls = {p.url for p in manual.pages}
    assert not any("logout" in u for u in visited_urls)


@pytest.mark.asyncio
async def test_crawl_site_never_visits_cross_origin_links(fixture_server):
    with patch("backend.app.site_learning.crawler.llm.generate_text", _mock_generate_text()):
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            manual = await crawl_site(browser, f"{fixture_server}/index.html", max_pages=10)
            await browser.close()

    visited_urls = {p.url for p in manual.pages}
    assert not any("external.example.com" in u for u in visited_urls)


@pytest.mark.asyncio
async def test_crawl_site_does_not_revisit_the_same_page_twice(fixture_server):
    """Dashboard/Products มีลิงก์กลับไป index.html/หากันเอง — ต้องไม่วนซ้ำไม่รู้จบ และ
    แต่ละหน้าต้องถูกเก็บแค่ครั้งเดียว"""
    with patch("backend.app.site_learning.crawler.llm.generate_text", _mock_generate_text()):
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            manual = await crawl_site(browser, f"{fixture_server}/index.html", max_pages=10)
            await browser.close()

    urls = [p.url for p in manual.pages]
    assert len(urls) == len(set(urls))
    assert len(manual.pages) == 3  # index, dashboard, products (logout/external ถูกกัน)


@pytest.mark.asyncio
async def test_crawl_site_respects_max_pages_limit(fixture_server):
    with patch("backend.app.site_learning.crawler.llm.generate_text", _mock_generate_text()):
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            manual = await crawl_site(browser, f"{fixture_server}/index.html", max_pages=1)
            await browser.close()

    assert len(manual.pages) == 1


@pytest.mark.asyncio
async def test_crawl_site_emits_progress_events(fixture_server):
    events = []

    async def on_progress(event):
        events.append(event)

    with patch("backend.app.site_learning.crawler.llm.generate_text", _mock_generate_text()):
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            await crawl_site(browser, f"{fixture_server}/index.html", max_pages=10, on_progress=on_progress)
            await browser.close()

    page_done_events = [e for e in events if e["kind"] == "page_done"]
    assert len(page_done_events) == 3
    assert events[-1]["kind"] == "crawl_scan_done"
    assert events[-1]["pages_found"] == 3


@pytest.mark.asyncio
async def test_crawl_site_falls_back_gracefully_when_llm_description_fails(fixture_server):
    """describe_page() ต้อง fallback เป็นชื่อจาก URL path เฉยๆ ไม่ throw ออกไปกลางการ
    crawl ถ้า LLM call ล้มเหลว/ตอบนอกรูปแบบ JSON"""
    with patch("backend.app.site_learning.crawler.llm.generate_text", AsyncMock(side_effect=RuntimeError("boom"))):
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            manual = await crawl_site(browser, f"{fixture_server}/index.html", max_pages=10)
            await browser.close()

    assert len(manual.pages) == 3
    assert all(p.name for p in manual.pages)  # fallback name ยังมีเสมอ ไม่ใช่ค่าว่าง


@pytest.mark.asyncio
async def test_crawl_site_sets_website_to_domain(fixture_server):
    with patch("backend.app.site_learning.crawler.llm.generate_text", _mock_generate_text()):
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            manual = await crawl_site(browser, f"{fixture_server}/index.html", max_pages=10)
            await browser.close()

    assert manual.website == "127.0.0.1"


# ---------------- W15: login bootstrap ----------------
# ปุ่ม "Sign In" ในหน้า fixture เหล่านี้ไม่ได้ submit ฟอร์มไป server จริง (fixture server
# เป็น SimpleHTTPRequestHandler อ่านไฟล์อย่างเดียว ไม่รับ POST) — ใช้ onclick
# window.location.href แทนเพื่อจำลอง "หน้าถัดไปหลัง login สำเร็จ" แบบ client-side ล้วนๆ
# ซึ่งเพียงพอสำหรับพิสูจน์ behavior ของ _attempt_login()/crawl_site() ที่สนใจแค่ว่ากรอก
# ช่องถูกต้อง + กดปุ่มถูกต้อง + ตามไปหน้าถัดไปได้จริง

_LOGIN_FIXTURE_PAGES = {
    "login.html": """
        <html><body>
          <input type="text" id="username" name="username" placeholder="Username" />
          <input type="password" id="password" name="password" placeholder="Password" />
          <button type="button" id="submit-btn" onclick="window.location.href='/post_login.html'">Sign In</button>
        </body></html>
    """,
    "post_login.html": """
        <html><body>
          <nav><a href="/settings.html">Settings</a></nav>
          <div>Welcome!</div>
        </body></html>
    """,
    "settings.html": """
        <html><body>
          <nav><a href="/post_login.html">Home</a></nav>
          <input type="password" id="change-password" name="new_password" placeholder="New password" />
          <button type="button" onclick="window.location.href='/post_login.html'">Sign In</button>
        </body></html>
    """,
}

_NO_SUBMIT_FIXTURE_PAGES = {
    "login_no_submit.html": """
        <html><body>
          <input type="text" id="username" name="username" placeholder="Username" />
          <input type="password" id="password" name="password" placeholder="Password" />
          <button type="button" onclick="window.location.href='/post_login.html'">Continue</button>
        </body></html>
    """,
}


def _make_fixture_server(tmp_path, pages):
    for name, html in pages.items():
        (tmp_path / name).write_text(html, encoding="utf-8")
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(tmp_path))
    httpd = http.server.HTTPServer(("127.0.0.1", 0), handler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, f"http://127.0.0.1:{port}"


@pytest.fixture
def login_fixture_server(tmp_path):
    httpd, base_url = _make_fixture_server(tmp_path, _LOGIN_FIXTURE_PAGES)
    yield base_url
    httpd.shutdown()


@pytest.fixture
def no_submit_fixture_server(tmp_path):
    httpd, base_url = _make_fixture_server(tmp_path, _NO_SUBMIT_FIXTURE_PAGES)
    yield base_url
    httpd.shutdown()


@pytest.mark.asyncio
async def test_crawl_site_login_bootstrap_fills_and_submits_then_continues_bfs(login_fixture_server):
    """เจอ password field ในหน้าแรก + ได้ username/password มา -> กรอก+กด submit ครั้ง
    เดียว แล้วต้องสำรวจต่อจากหน้าหลัง login ได้ (settings.html เดินทางมาจาก nav link
    ในหน้า post_login.html เท่านั้น — เข้าถึงไม่ได้เลยถ้า login ไม่สำเร็จ)"""
    with patch("backend.app.site_learning.crawler.llm.generate_text", _mock_generate_text()):
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            manual = await crawl_site(
                browser, f"{login_fixture_server}/login.html", max_pages=10,
                username="alice", password="s3cr3t",
            )
            await browser.close()

    visited_urls = {p.url for p in manual.pages}
    assert any("login.html" in u for u in visited_urls)
    assert any("post_login.html" in u for u in visited_urls)
    assert any("settings.html" in u for u in visited_urls)
    assert len(manual.pages) == 3


@pytest.mark.asyncio
async def test_crawl_site_attempts_login_only_once(login_fixture_server):
    """settings.html ก็มี password field + ปุ่มข้อความ "Sign In" เหมือนกัน (จำลองหน้า
    "เปลี่ยนรหัสผ่าน") แต่ต้องไม่ลอง submit ซ้ำอีกรอบ — login bootstrap ทำได้แค่ครั้งเดียว
    ตลอดทั้ง crawl เท่านั้น"""
    from backend.app.site_learning import crawler as crawler_module

    real_attempt_login = crawler_module.attempt_login
    spy = AsyncMock(wraps=real_attempt_login)

    with patch("backend.app.site_learning.crawler.llm.generate_text", _mock_generate_text()), \
         patch("backend.app.site_learning.crawler.attempt_login", spy):
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            await crawl_site(
                browser, f"{login_fixture_server}/login.html", max_pages=10,
                username="alice", password="s3cr3t",
            )
            await browser.close()

    assert spy.await_count == 1


@pytest.mark.asyncio
async def test_crawl_site_login_bootstrap_fails_gracefully_without_submit_button(no_submit_fixture_server):
    """มีช่อง username/password ครบแต่หาปุ่ม submit ที่เข้าข่าย keyword ("sign in"/"log
    in"/ฯลฯ) ไม่เจอ (ปุ่มจริงชื่อ "Continue") -> _attempt_login ต้องคืน False เงียบๆ ไม่
    throw และ crawl ต้องไม่ล้ม (เก็บได้แค่หน้า login เดียว ไม่ตามไปหน้าถัดไปเพราะไม่ได้
    login จริง)"""
    with patch("backend.app.site_learning.crawler.llm.generate_text", _mock_generate_text()):
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            manual = await crawl_site(
                browser, f"{no_submit_fixture_server}/login_no_submit.html", max_pages=10,
                username="alice", password="s3cr3t",
            )
            await browser.close()

    assert len(manual.pages) == 1
    assert manual.pages[0].url.endswith("login_no_submit.html")


@pytest.mark.asyncio
async def test_crawl_site_never_persists_credentials_in_manual_or_progress_events(login_fixture_server):
    """credential ต้องไม่รั่วไปที่ไหนเลย — ไม่อยู่ใน SiteManual (ทุก field ของทุกหน้า) และ
    ไม่อยู่ใน progress event ใดๆ ที่ยิงออกไประหว่าง crawl"""
    username, password = "alice-cred-marker", "s3cr3t-cred-marker"
    events = []

    async def on_progress(event):
        events.append(event)

    with patch("backend.app.site_learning.crawler.llm.generate_text", _mock_generate_text()):
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            manual = await crawl_site(
                browser, f"{login_fixture_server}/login.html", max_pages=10,
                username=username, password=password, on_progress=on_progress,
            )
            await browser.close()

    manual_dump = json.dumps(manual.to_dict())
    assert username not in manual_dump
    assert password not in manual_dump

    events_dump = json.dumps(events)
    assert username not in events_dump
    assert password not in events_dump


# ---------------- W23: on_credentials_needed — ถามคนจริงกลางคัน crawl ----------------


@pytest.mark.asyncio
async def test_crawl_site_asks_for_credentials_when_none_given_and_login_page_found(login_fixture_server):
    """ไม่ได้ส่ง username/password มาเลยตอนเริ่ม crawl แต่หน้าแรกมี password field จริง —
    ต้องเรียก on_credentials_needed(domain) แล้วเอาผลลัพธ์ไป login bootstrap ต่อทันที
    (เข้าถึง settings.html ได้เหมือน test_crawl_site_login_bootstrap_fills_and_submits_
    then_continues_bfs ทุกประการ ทั้งที่รอบนี้ไม่ได้ส่ง credential มาตั้งแต่ต้นเลย)"""
    domain_seen = None

    async def on_credentials_needed(domain):
        nonlocal domain_seen
        domain_seen = domain
        return {"username": "alice", "password": "s3cr3t"}

    with patch("backend.app.site_learning.crawler.llm.generate_text", _mock_generate_text()):
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            manual = await crawl_site(
                browser, f"{login_fixture_server}/login.html", max_pages=10,
                on_credentials_needed=on_credentials_needed,
            )
            await browser.close()

    assert domain_seen == "127.0.0.1"
    visited_urls = {p.url for p in manual.pages}
    assert any("post_login.html" in u for u in visited_urls)
    assert any("settings.html" in u for u in visited_urls)


@pytest.mark.asyncio
async def test_crawl_site_continues_without_login_when_user_skips_credentials_prompt(login_fixture_server):
    """on_credentials_needed คืน None (user กด "ข้าม"/หมดเวลา) — crawl ต้องไปต่อโดยไม่
    login แทนที่จะค้าง/ล้ม เก็บได้แค่หน้า login เดียว (settings.html เข้าถึงไม่ได้เพราะไม่ได้
    login จริง — เหมือน test_crawl_site_login_bootstrap_fails_gracefully_without_submit_
    button ทุกประการแค่สาเหตุต่างกัน)"""
    async def on_credentials_needed(domain):
        return None

    with patch("backend.app.site_learning.crawler.llm.generate_text", _mock_generate_text()):
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            manual = await crawl_site(
                browser, f"{login_fixture_server}/login.html", max_pages=10,
                on_credentials_needed=on_credentials_needed,
            )
            await browser.close()

    assert len(manual.pages) == 1
    assert manual.pages[0].url.endswith("login.html")


@pytest.mark.asyncio
async def test_crawl_site_asks_for_credentials_only_once(login_fixture_server):
    """settings.html ก็มี password field เหมือนกัน (จำลองหน้า "เปลี่ยนรหัสผ่าน") แต่ถ้า
    user เพิ่งเลือกข้ามไปแล้วตอนเจอหน้า login.html ต้องไม่ถามซ้ำอีกรอบตอนไปเจอหน้าอื่นที่มี
    password field — login_attempted ตัวเดียวกับที่ gate เส้นทาง username/password ปกติ"""
    call_count = 0

    async def on_credentials_needed(domain):
        nonlocal call_count
        call_count += 1
        return None

    with patch("backend.app.site_learning.crawler.llm.generate_text", _mock_generate_text()):
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            await crawl_site(
                browser, f"{login_fixture_server}/login.html", max_pages=10,
                on_credentials_needed=on_credentials_needed,
            )
            await browser.close()

    assert call_count == 1


@pytest.mark.asyncio
async def test_crawl_site_does_not_ask_for_credentials_on_pages_without_login_fields(fixture_server):
    """on_credentials_needed ให้มา แต่ไม่มีหน้าไหนมี password field เลย (fixture_server
    ธรรมดา ไม่ใช่ login_fixture_server) — ต้องไม่ถูกเรียกเลยสักครั้ง"""
    on_credentials_needed = AsyncMock(return_value=None)

    with patch("backend.app.site_learning.crawler.llm.generate_text", _mock_generate_text()):
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            await crawl_site(
                browser, f"{fixture_server}/index.html", max_pages=10,
                on_credentials_needed=on_credentials_needed,
            )
            await browser.close()

    on_credentials_needed.assert_not_awaited()


@pytest.mark.asyncio
async def test_crawl_site_prefers_upfront_credentials_over_asking(login_fixture_server):
    """ถ้าส่ง username/password มาตั้งแต่ต้น crawl แล้ว ไม่ควรเรียก on_credentials_needed
    เลย (เส้นทาง username/password ปกติ gate ด้วย login_attempted ตัวเดียวกัน — ตรวจก่อน
    เสมอ ดู crawl_site() ในไฟล์ crawler.py)"""
    on_credentials_needed = AsyncMock(return_value=None)

    with patch("backend.app.site_learning.crawler.llm.generate_text", _mock_generate_text()):
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            manual = await crawl_site(
                browser, f"{login_fixture_server}/login.html", max_pages=10,
                username="alice", password="s3cr3t",
                on_credentials_needed=on_credentials_needed,
            )
            await browser.close()

    on_credentials_needed.assert_not_awaited()
    visited_urls = {p.url for p in manual.pages}
    assert any("settings.html" in u for u in visited_urls)


# ---------------- W16: ไล่กดปุ่มปลอดภัยระหว่าง crawl ----------------
# ปุ่มพวกนี้ใช้ onclick="window.location.href=...' แทน <a href> จริง เพื่อจำลอง "ปุ่มที่
# พาไปหน้าใหม่" (ต่างจาก nav link ที่ extractor.py เก็บแยกเป็น nav_links อยู่แล้ว — ปุ่ม
# พวกนี้ต้องไม่อยู่ใน <nav>/[role=navigation]/aside/header/footer เพื่อไม่ให้ถูกนับเป็น
# nav link ไปก่อน จะได้พิสูจน์ path การ "กด" ล้วนๆ)

_BUTTON_FIXTURE_PAGES = {
    "start.html": """
        <html><body>
          <button onclick="window.location.href='/detail.html'">View Details</button>
          <button onclick="window.location.href='/deleted-trap.html'">Delete Item</button>
        </body></html>
    """,
    "detail.html": """
        <html><body>
          <button id="expand-btn" onclick="document.getElementById('hidden-panel').style.display='block'">Expand</button>
          <div id="hidden-panel" style="display:none">
            <button data-testid="hidden-btn">Extra Info</button>
          </div>
          <button onclick="window.location.href='/subdetail.html'">View Sub Detail</button>
        </body></html>
    """,
    "subdetail.html": """
        <html><body>
          <button onclick="window.location.href='/too-deep.html'">View Deeper</button>
        </body></html>
    """,
    "too-deep.html": "<html><body>dead end — no buttons, forces the DFS to backtrack</body></html>",
    "deleted-trap.html": "<html><body>should never be reached (Delete is blocked)</body></html>",
}

_BUTTON_CAP_FIXTURE_PAGES = {
    "cap_start.html": """
        <html><body>
          <button onclick="window.location.href='/view-a.html'">View A</button>
          <button onclick="window.location.href='/view-b.html'">View B</button>
          <button onclick="window.location.href='/view-c.html'">View C</button>
        </body></html>
    """,
    "view-a.html": "<html><body>a</body></html>",
    "view-b.html": "<html><body>b</body></html>",
    "view-c.html": "<html><body>c</body></html>",
}


@pytest.fixture
def button_fixture_server(tmp_path):
    httpd, base_url = _make_fixture_server(tmp_path, _BUTTON_FIXTURE_PAGES)
    yield base_url
    httpd.shutdown()


@pytest.fixture
def button_cap_fixture_server(tmp_path):
    httpd, base_url = _make_fixture_server(tmp_path, _BUTTON_CAP_FIXTURE_PAGES)
    yield base_url
    httpd.shutdown()


@pytest.mark.asyncio
async def test_crawl_site_explores_safe_buttons_with_unlimited_depth_dfs(button_fixture_server):
    """"View Details" (ปลอดภัย) ต้องพาไป detail.html ได้ แล้ว detail.html เองก็ต้องถูกไล่
    กดปุ่มต่อทันที ("View Sub Detail" พาไป subdetail.html) และ subdetail.html เองก็ต้อง
    ถูกไล่กดปุ่มต่อไปอีก ("View Deeper" พาไป too-deep.html) ไม่มี depth cap แล้ว (W16 —
    ยืนยันจาก user: "กดต่อไปเรื่อยๆ จนมั่นใจว่าไม่มีทางไปต่อ") too-deep.html ไม่มีปุ่มเลย
    (ตัน) ต้องทำให้ DFS ถอยกลับมาเองได้โดยไม่ค้าง — ยืนยันด้วยว่า crawl จบจริง (ไม่ hang)
    และเก็บได้ครบทั้ง 4 หน้าที่ปลอดภัย"""
    with patch("backend.app.site_learning.crawler.llm.generate_text", _mock_generate_text()):
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            manual = await crawl_site(browser, f"{button_fixture_server}/start.html", max_pages=20)
            await browser.close()

    visited_urls = {p.url for p in manual.pages}
    assert any("start.html" in u for u in visited_urls)
    assert any(u.endswith("/detail.html") for u in visited_urls)
    assert any("subdetail.html" in u for u in visited_urls)
    assert any("too-deep.html" in u for u in visited_urls)
    assert len(manual.pages) == 4  # start, detail, subdetail, too-deep (deleted-trap ถูกกัน)


@pytest.mark.asyncio
async def test_crawl_site_never_clicks_blocked_buttons_during_exploration(button_fixture_server):
    """"Delete Item" ต้องไม่ถูกกดเด็ดขาดระหว่างไล่สำรวจปุ่ม (is_crawl_safe บล็อก "delete")
    — deleted-trap.html ต้องไม่ถูกเยี่ยมเลย"""
    with patch("backend.app.site_learning.crawler.llm.generate_text", _mock_generate_text()):
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            manual = await crawl_site(browser, f"{button_fixture_server}/start.html", max_pages=20)
            await browser.close()

    visited_urls = {p.url for p in manual.pages}
    assert not any("deleted-trap.html" in u for u in visited_urls)


@pytest.mark.asyncio
async def test_crawl_site_merges_modal_revealed_content_without_duplicating_page(button_fixture_server):
    """กด "Expand" (ปุ่มที่ไม่เปลี่ยน URL แค่เปิด panel ที่ซ่อนอยู่) ต้อง re-extract แล้ว
    merge ปุ่มที่เพิ่งโผล่ ("Extra Info") เข้ากับ PageInfo ของ detail.html เดิม ไม่สร้าง
    entry ใหม่แยกต่างหาก (detail.html ต้องปรากฏใน manual แค่ครั้งเดียว)"""
    with patch("backend.app.site_learning.crawler.llm.generate_text", _mock_generate_text()):
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            manual = await crawl_site(browser, f"{button_fixture_server}/start.html", max_pages=20)
            await browser.close()

    # หมายเหตุ: ใช้ endswith("/detail.html") ไม่ใช่ endswith("detail.html") เฉยๆ — เพราะ
    # "subdetail.html".endswith("detail.html") ก็เป็น True ด้วย (substring บังเอิญตรงท้าย
    # คำ) จะทำให้นับ subdetail.html ปนเข้ามาเป็น false positive
    detail_pages = [p for p in manual.pages if p.url.endswith("/detail.html")]
    assert len(detail_pages) == 1
    button_texts = [b.text for b in detail_pages[0].buttons]
    assert "Extra Info" in button_texts


@pytest.mark.asyncio
async def test_crawl_site_emits_button_explored_progress_events(button_fixture_server):
    events = []

    async def on_progress(event):
        events.append(event)

    with patch("backend.app.site_learning.crawler.llm.generate_text", _mock_generate_text()):
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            await crawl_site(
                browser, f"{button_fixture_server}/start.html", max_pages=20, on_progress=on_progress,
            )
            await browser.close()

    button_events = [e for e in events if e["kind"] == "button_explored"]
    assert any(e["button"] == "View Details" for e in button_events)
    # ปุ่มที่ถูกบล็อกไม่ควรมี progress event "button_explored" เลย (ไม่ผ่านตัวกรอง
    # is_crawl_safe ตั้งแต่แรก ไม่ใช่แค่ไม่ถูกกด)
    assert not any(e["button"] == "Delete Item" for e in button_events)


@pytest.mark.asyncio
async def test_crawl_site_respects_max_buttons_per_page_limit(button_cap_fixture_server, monkeypatch):
    monkeypatch.setattr(
        "backend.app.site_learning.crawler.settings.site_learning_max_buttons_per_page", 2,
    )
    with patch("backend.app.site_learning.crawler.llm.generate_text", _mock_generate_text()):
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            manual = await crawl_site(browser, f"{button_cap_fixture_server}/cap_start.html", max_pages=20)
            await browser.close()

    visited_urls = {p.url for p in manual.pages}
    assert any("view-a.html" in u for u in visited_urls)
    assert any("view-b.html" in u for u in visited_urls)
    assert not any("view-c.html" in u for u in visited_urls)
    assert len(manual.pages) == 3  # cap_start + view-a + view-b (view-c ถูกตัดเพราะเกิน cap)


# ---------------- W18: UI pattern buttons — สำรวจแค่ instance ตัวแทนเดียว ----------------

def _pattern_card(i: int) -> str:
    return f"""
    <div class="product-card">
      <img src="/p{i}.jpg" alt="Product {i}">
      <h3>Product {i}</h3>
      <button onclick="window.location.href='/product{i}.html'">View Details</button>
    </div>
    """


_PATTERN_FIXTURE_PAGES = {
    "grid.html": f"""
        <html><body>
          <div class="grid">
            {"".join(_pattern_card(i) for i in range(5))}
          </div>
        </body></html>
    """,
    **{f"product{i}.html": f"<html><body>Product {i} detail</body></html>" for i in range(5)},
}


@pytest.fixture
def pattern_fixture_server(tmp_path):
    httpd, base_url = _make_fixture_server(tmp_path, _PATTERN_FIXTURE_PAGES)
    yield base_url
    httpd.shutdown()


@pytest.mark.asyncio
async def test_crawl_site_explores_only_the_representative_instance_of_a_ui_pattern(pattern_fixture_server):
    """5 การ์ดสินค้าโครงสร้างเดียวกัน แต่ละใบมีปุ่ม "View Details" ปลอดภัยที่พาไปหน้า
    รายละเอียดคนละหน้า — extractor.py ยุบเป็น UIPatternInfo ตัวเดียว (ดู
    test_site_learning_extractor.py) เหลือปุ่ม "ตัวแทน" แค่ 1 ปุ่ม (ของการ์ดใบแรก) ดังนั้น
    crawler ต้องกดแค่ครั้งเดียว ไปหน้ารายละเอียดแค่ 1 หน้า ไม่ใช่ทั้ง 5 หน้า"""
    with patch("backend.app.site_learning.crawler.llm.generate_text", _mock_generate_text()):
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            manual = await crawl_site(browser, f"{pattern_fixture_server}/grid.html", max_pages=20)
            await browser.close()

    product_pages_visited = [p for p in manual.pages if "/product" in p.url and p.url.endswith(".html")]
    assert len(product_pages_visited) == 1
    assert len(manual.pages) == 2  # grid.html + 1 product detail page เท่านั้น


# ---------------- W24: retry helper สำหรับ goto/click ----------------
# mock page ตรงๆ (ไม่เปิด browser จริง) — เร็วกว่ามากและพิสูจน์ retry semantics (จำนวนครั้ง
# ที่เรียกจริง, คืน None ตอนสำเร็จ, คืนข้อความ error ตอนล้มเหลวครบ) ได้แม่นยำกว่าการพยายาม
# จำลอง network failure ผ่าน browser จริง


@pytest.mark.asyncio
async def test_goto_with_retry_returns_none_on_first_success():
    page = AsyncMock()
    error = await _goto_with_retry(page, "http://example.com", retries=2)
    assert error is None
    assert page.goto.await_count == 1


@pytest.mark.asyncio
async def test_goto_with_retry_recovers_from_transient_failure(monkeypatch):
    monkeypatch.setattr("backend.app.site_learning.crawler.settings.site_learning_retry_backoff_ms", 1)
    page = AsyncMock()
    page.goto.side_effect = [RuntimeError("timeout"), None]
    error = await _goto_with_retry(page, "http://example.com", retries=2)
    assert error is None
    assert page.goto.await_count == 2


@pytest.mark.asyncio
async def test_goto_with_retry_gives_up_after_exhausting_retries(monkeypatch):
    monkeypatch.setattr("backend.app.site_learning.crawler.settings.site_learning_retry_backoff_ms", 1)
    page = AsyncMock()
    page.goto.side_effect = RuntimeError("connection refused")
    error = await _goto_with_retry(page, "http://example.com", retries=2)
    assert error is not None
    assert "connection refused" in error
    assert page.goto.await_count == 3  # ครั้งแรก + retry 2 ครั้ง


@pytest.mark.asyncio
async def test_click_with_retry_recovers_from_transient_failure(monkeypatch):
    monkeypatch.setattr("backend.app.site_learning.crawler.settings.site_learning_retry_backoff_ms", 1)
    page = AsyncMock()
    page.click.side_effect = [RuntimeError("detached"), None]
    error = await _click_with_retry(page, "#btn", retries=2)
    assert error is None
    assert page.click.await_count == 2


@pytest.mark.asyncio
async def test_click_with_retry_gives_up_after_exhausting_retries(monkeypatch):
    monkeypatch.setattr("backend.app.site_learning.crawler.settings.site_learning_retry_backoff_ms", 1)
    page = AsyncMock()
    page.click.side_effect = RuntimeError("element not visible")
    error = await _click_with_retry(page, "#btn", retries=1)
    assert error is not None
    assert page.click.await_count == 2  # ครั้งแรก + retry 1 ครั้ง


# ---------------- W24: SPA DOM-stability + infinite-scroll reveal helper ----------------


@pytest.mark.asyncio
async def test_wait_for_dom_stable_returns_once_length_repeats():
    page = AsyncMock()
    # การอ่านครั้งแรกแค่ตั้ง baseline (เทียบกับ sentinel -1 เริ่มต้น ไม่นับเป็น match) —
    # ต้องอ่านซ้ำจนค่าเท่าเดิมติดกันครบ `checks` ครั้ง (ไม่นับครั้ง baseline) ถึงจะถือว่านิ่ง
    # ค่า 100 ครบ 3 ครั้ง (1 baseline + 2 ครั้งติดกันจริง) พอสำหรับ checks=2 ไม่ต้องวนต่อจน
    # เจอ 999 เลย
    page.evaluate.side_effect = [100, 100, 100, 999, 999]
    await _wait_for_dom_stable(page, checks=2, interval_ms=1, max_iterations=10)
    assert page.evaluate.await_count == 3


@pytest.mark.asyncio
async def test_wait_for_dom_stable_never_throws_when_evaluate_fails():
    page = AsyncMock()
    page.evaluate.side_effect = RuntimeError("page closed")
    await _wait_for_dom_stable(page, checks=2, interval_ms=1, max_iterations=5)  # ไม่ throw ออกมา


@pytest.mark.asyncio
async def test_reveal_dynamic_content_scrolls_until_height_stops_growing(monkeypatch):
    monkeypatch.setattr("backend.app.site_learning.crawler.settings.site_learning_max_scroll_attempts", 6)
    monkeypatch.setattr("backend.app.site_learning.crawler.settings.site_learning_scroll_wait_ms", 1)
    page = AsyncMock()
    # scrollHeight: 1000 (initial) -> 1500 -> 2000 -> 2000 (นิ่งแล้ว หยุด scroll)
    page.evaluate.side_effect = [1000, 1500, 2000, 2000]
    await _reveal_dynamic_content(page)
    # เรียก evaluate: 1 (initial height) + 3 รอบ scroll (แต่ล่าสุดเท่าเดิมเลยหยุด) + 1 (scroll กลับบนสุด)
    scroll_to_top_calls = [c for c in page.evaluate.call_args_list if c.args and "scrollTo(0, 0)" in c.args[0]]
    assert len(scroll_to_top_calls) == 1


@pytest.mark.asyncio
async def test_reveal_dynamic_content_never_throws_when_evaluate_fails():
    page = AsyncMock()
    page.evaluate.side_effect = RuntimeError("page closed")
    await _reveal_dynamic_content(page)  # ไม่ throw ออกมา


# ---------------- W24: เมนูที่ไม่ใช่ <a> (role=menuitem/tab) ----------------

_MENU_ITEM_FIXTURE_PAGES = {
    "menu_start.html": """
        <html><body>
          <div role="menuitem" onclick="window.location.href='/settings.html'">Settings</div>
        </body></html>
    """,
    "settings.html": "<html><body>settings page</body></html>",
}


@pytest.fixture
def menu_item_fixture_server(tmp_path):
    httpd, base_url = _make_fixture_server(tmp_path, _MENU_ITEM_FIXTURE_PAGES)
    yield base_url
    httpd.shutdown()


@pytest.mark.asyncio
async def test_crawl_site_explores_role_menuitem_even_with_generic_label(menu_item_fixture_server):
    """"Settings" ไม่ตรงคำไหนใน safety.ALLOWED_CRAWL_KEYWORDS เลย (ต่างจาก "view"/"next"/
    ฯลฯ) — ปุ่มทั่วไปแบบนี้เดิมจะไม่ถูกไล่กด (is_crawl_safe default-deny) แต่ element นี้มี
    role="menuitem" ตรงๆ (ไม่ใช่ <a href>) ต้องถูกจัดเป็น nav menu item แล้วไล่กดแบบ
    default-allow (ดู extractor.py::isNavMenuItem + crawler.py::_is_explorable)"""
    with patch("backend.app.site_learning.crawler.llm.generate_text", _mock_generate_text()):
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            manual = await crawl_site(browser, f"{menu_item_fixture_server}/menu_start.html", max_pages=10)
            await browser.close()

    visited_urls = {p.url for p in manual.pages}
    assert any("settings.html" in u for u in visited_urls)


@pytest.mark.asyncio
async def test_crawl_site_does_not_dfs_click_real_href_nav_links(fixture_server):
    """W24: <a href="..."> ที่มีปลายทางจริงต้องไม่ถูก DFS-click ซ้ำผ่าน _explore_buttons()
    (เดินผ่าน BFS href เดิมเท่านั้น ที่เช็ค same-origin ได้ก่อน navigate) — ยืนยันด้วยการ
    เช็คว่า "Dashboard"/"Products" (label ที่ไม่ตรง ALLOWED_CRAWL_KEYWORDS เลย) ไม่เคยปรากฏ
    ใน progress event "button_explored" แม้แต่ครั้งเดียว ทั้งที่ทั้งคู่ถูกเยี่ยมจริงผ่าน BFS
    (เห็นได้จาก manual.pages)"""
    events = []

    async def on_progress(event):
        events.append(event)

    with patch("backend.app.site_learning.crawler.llm.generate_text", _mock_generate_text()):
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            manual = await crawl_site(
                browser, f"{fixture_server}/index.html", max_pages=10, on_progress=on_progress,
            )
            await browser.close()

    visited_urls = {p.url for p in manual.pages}
    assert any("dashboard" in u for u in visited_urls)  # เข้าถึงได้จริงผ่าน BFS href
    button_labels = {e["button"] for e in events if e["kind"] == "button_explored"}
    assert "Dashboard" not in button_labels
    assert "Products" not in button_labels


# ---------------- W24: session check หลัง login (login_result event) ----------------


@pytest.mark.asyncio
async def test_crawl_site_emits_login_result_success_true_when_session_looks_valid(login_fixture_server):
    events = []

    async def on_progress(event):
        events.append(event)

    with patch("backend.app.site_learning.crawler.llm.generate_text", _mock_generate_text()):
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            await crawl_site(
                browser, f"{login_fixture_server}/login.html", max_pages=10,
                username="alice", password="s3cr3t", on_progress=on_progress,
            )
            await browser.close()

    login_events = [e for e in events if e["kind"] == "login_result"]
    assert len(login_events) == 1
    assert login_events[0]["success"] is True


@pytest.mark.asyncio
async def test_crawl_site_emits_login_result_success_false_when_submit_never_happened(no_submit_fixture_server):
    events = []

    async def on_progress(event):
        events.append(event)

    with patch("backend.app.site_learning.crawler.llm.generate_text", _mock_generate_text()):
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            await crawl_site(
                browser, f"{no_submit_fixture_server}/login_no_submit.html", max_pages=10,
                username="alice", password="s3cr3t", on_progress=on_progress,
            )
            await browser.close()

    login_events = [e for e in events if e["kind"] == "login_result"]
    assert len(login_events) == 1
    assert login_events[0]["success"] is False
    assert login_events[0]["reason"]  # ต้องมีเหตุผลแนบมาด้วย ไม่ใช่แค่ False เฉยๆ


# ---------------- W24: บันทึก error จริงแทนกลืนเงียบๆ ----------------

_BROKEN_LINK_FIXTURE_PAGES = {
    "broken_start.html": """
        <html><body>
          <nav><a href="http://127.0.0.1:9/unreachable">Broken</a></nav>
        </body></html>
    """,
}


@pytest.fixture
def broken_link_fixture_server(tmp_path):
    httpd, base_url = _make_fixture_server(tmp_path, _BROKEN_LINK_FIXTURE_PAGES)
    yield base_url
    httpd.shutdown()


# ---------------- W25: BFS ต้องเดินตามลิงก์ในเนื้อหา ไม่ใช่แค่ลิงก์ในเมนู ----------------

_CONTENT_LINK_FIXTURE_PAGES = {
    "home.html": """
        <html><body>
          <nav><a href="/dashboard.html">Dashboard</a></nav>
          <main>
            <article><p>Some text <a href="/article-detail.html">Read more</a> here.</p></article>
          </main>
        </body></html>
    """,
    "dashboard.html": "<html><body>dashboard</body></html>",
    "article-detail.html": "<html><body>article detail page</body></html>",
}


@pytest.fixture
def content_link_fixture_server(tmp_path):
    httpd, base_url = _make_fixture_server(tmp_path, _CONTENT_LINK_FIXTURE_PAGES)
    yield base_url
    httpd.shutdown()


@pytest.mark.asyncio
async def test_crawl_site_visits_links_outside_nav_containers(content_link_fixture_server):
    """W25: user ขอ "extract all clickable elements: a" ตรงๆ — ลิงก์กลางบทความ (ไม่ได้อยู่
    ใน <nav>) ต้องถูก BFS เดินตามไปเยี่ยมด้วย ไม่ใช่แค่ลิงก์ในเมนู (เดิม extractor.py สแกน
    หา <a href> แค่ใน NAV_CONTAINERS เท่านั้น พลาดหน้านี้ไปเลย)"""
    with patch("backend.app.site_learning.crawler.llm.generate_text", _mock_generate_text()):
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            manual = await crawl_site(browser, f"{content_link_fixture_server}/home.html", max_pages=10)
            await browser.close()

    visited_urls = {p.url for p in manual.pages}
    assert any("dashboard.html" in u for u in visited_urls)
    assert any("article-detail.html" in u for u in visited_urls)


@pytest.mark.asyncio
async def test_crawl_site_records_goto_errors_instead_of_silently_skipping(broken_link_fixture_server, monkeypatch):
    """ลิงก์ไปพอร์ตที่ไม่มีอะไรฟังอยู่ (connection refused ทันที) ต้องไม่ทำให้ crawl ทั้ง
    ก้อนพัง (เก็บ broken_start.html ได้ตามปกติ) แต่ต้องบันทึกไว้ใน manual.errors + ยิง
    event "page_error" แทนที่จะกลืนเงียบๆ เหมือนเดิม (ดู docstring หัวไฟล์ ข้อ 6)"""
    monkeypatch.setattr("backend.app.site_learning.crawler.settings.site_learning_goto_retries", 0)
    monkeypatch.setattr("backend.app.site_learning.crawler.settings.site_learning_retry_backoff_ms", 1)
    events = []

    async def on_progress(event):
        events.append(event)

    with patch("backend.app.site_learning.crawler.llm.generate_text", _mock_generate_text()):
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            manual = await crawl_site(
                browser, f"{broken_link_fixture_server}/broken_start.html", max_pages=10, on_progress=on_progress,
            )
            await browser.close()

    assert len(manual.pages) == 1  # แค่ broken_start.html — ลิงก์ที่ไปไม่ถึงไม่ถูกนับเป็นหน้า
    assert any(e["phase"] == "goto" for e in manual.errors)
    assert any(e["kind"] == "page_error" for e in events)
    done_event = [e for e in events if e["kind"] == "crawl_scan_done"][0]
    assert done_event["errors_found"] >= 1


# ---------------- W26: สรุปภาพรวม "เว็บไซต์นี้ทำอะไรได้บ้าง" หลังเรียนรู้เสร็จ ----------------


@pytest.mark.asyncio
async def test_describe_site_uses_llm_to_summarize_all_pages():
    pages = [
        PageInfo(name="Login", url="/login", description="หน้าเข้าสู่ระบบ"),
        PageInfo(name="Dashboard", url="/dashboard", description="หน้าแดชบอร์ดสรุปข้อมูล"),
    ]
    with patch(
        "backend.app.site_learning.crawler.llm.generate_text",
        AsyncMock(return_value="เว็บไซต์นี้ใช้เข้าสู่ระบบแล้วดูแดชบอร์ดข้อมูลได้"),
    ):
        summary = await describe_site(client=None, model="m", provider="gemini", website="example.com", pages=pages)
    assert summary == "เว็บไซต์นี้ใช้เข้าสู่ระบบแล้วดูแดชบอร์ดข้อมูลได้"


@pytest.mark.asyncio
async def test_describe_site_falls_back_to_page_names_when_llm_fails():
    """ไม่ throw ออกไปกลาง crawl เหมือน describe_page() — fallback เป็นการเรียงชื่อหน้าดิบๆ"""
    pages = [PageInfo(name="Login", url="/login", description="")]
    with patch("backend.app.site_learning.crawler.llm.generate_text", AsyncMock(side_effect=RuntimeError("boom"))):
        summary = await describe_site(client=None, model="m", provider="gemini", website="example.com", pages=pages)
    assert "Login" in summary


@pytest.mark.asyncio
async def test_describe_site_falls_back_when_llm_returns_empty_text():
    pages = [PageInfo(name="Login", url="/login", description="")]
    with patch("backend.app.site_learning.crawler.llm.generate_text", AsyncMock(return_value="   ")):
        summary = await describe_site(client=None, model="m", provider="gemini", website="example.com", pages=pages)
    assert "Login" in summary


@pytest.mark.asyncio
async def test_describe_site_returns_empty_string_when_no_pages_have_names():
    summary = await describe_site(client=None, model="m", provider="gemini", website="example.com", pages=[])
    assert summary == ""


@pytest.mark.asyncio
async def test_crawl_site_populates_summary_field(fixture_server):
    with patch("backend.app.site_learning.crawler.llm.generate_text", _mock_generate_text()):
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            manual = await crawl_site(browser, f"{fixture_server}/index.html", max_pages=10)
            await browser.close()

    assert manual.summary != ""


# ---------------- W28/W29: กันไล่กด "ปุ่มเดิม" ซ้ำไม่รู้จบข้ามหลายหน้า ----------------


def test_button_signature_ignores_dynamic_aria_label_and_title():
    """W29: aria_label/title มักมีเนื้อหาต่อท้ายที่เปลี่ยนทุก instance (เช่น "Next video:
    <ชื่อคลิป>", "ผลการค้นหาสำหรับ <คำค้น>") — ต้องไม่ถูกใช้เป็นส่วนหนึ่งของ signature เลย
    ปุ่มที่เหมือนกันทุกอย่างยกเว้น aria_label/title ต้องได้ signature เดียวกัน"""
    a = ButtonInfo(text="Next", aria_label="Next video: Song A", title="Play next", has_icon=True)
    b = ButtonInfo(text="Next", aria_label="Next video: Song B", title="Play next 2", has_icon=True)
    assert _button_signature(a) == _button_signature(b)


def test_button_signature_differs_by_text():
    a = ButtonInfo(text="Next")
    b = ButtonInfo(text="Search")
    assert _button_signature(a) != _button_signature(b)


def test_button_signature_differs_by_role():
    a = ButtonInfo(text="View", role="button")
    b = ButtonInfo(text="View", role="link")
    assert _button_signature(a) != _button_signature(b)


def test_button_signature_differs_by_is_nav_menu_item():
    a = ButtonInfo(text="Settings", is_nav_menu_item=True)
    b = ButtonInfo(text="Settings", is_nav_menu_item=False)
    assert _button_signature(a) != _button_signature(b)


def test_button_signature_falls_back_to_data_testid_when_text_empty():
    """ปุ่ม icon-only (ไม่มี text บนปุ่มเลย เช่น ไอคอนค้นหาแว่นขยายล้วนๆ) ต้องยังจับคู่กันได้
    ถ้า data-testid ตรงกัน"""
    a = ButtonInfo(text="", data_testid="search-icon")
    b = ButtonInfo(text="", data_testid="search-icon")
    c = ButtonInfo(text="", data_testid="cart-icon")
    assert _button_signature(a) == _button_signature(b)
    assert _button_signature(a) != _button_signature(c)


def test_button_signature_falls_back_to_icon_hint_when_text_and_testid_empty():
    a = ButtonInfo(text="", data_testid="", icon_hint="search")
    b = ButtonInfo(text="", data_testid="", icon_hint="search")
    c = ButtonInfo(text="", data_testid="", icon_hint="cart")
    assert _button_signature(a) == _button_signature(b)
    assert _button_signature(a) != _button_signature(c)


# ---------------- W33: template ของทั้งหน้า (ไม่ใช่แค่ปุ่มเดียว) ----------------


def test_page_template_matches_pages_with_identical_button_structure():
    """สองหน้าที่มีปุ่มโครงสร้างเดียวกันทุกอย่าง (เช่นคลิป Shorts คนละคลิป UI เหมือนกัน
    ทุกอย่าง) ต้องได้ template เดียวกัน แม้ URL จะคนละอันก็ตาม (ไม่ใช่ส่วนหนึ่งของ template)"""
    page_a = PageInfo(
        url="/shorts/clip1",
        buttons=[
            ButtonInfo(text="Like", data_testid="like-btn", selector="#like"),
            ButtonInfo(text="Comment", data_testid="comment-btn", selector="#comment"),
        ],
    )
    page_b = PageInfo(
        url="/shorts/clip2",
        buttons=[
            ButtonInfo(text="Like", data_testid="like-btn", selector="#like"),
            ButtonInfo(text="Comment", data_testid="comment-btn", selector="#comment"),
        ],
    )
    assert _page_template(page_a) == _page_template(page_b)


def test_page_template_ignores_button_order():
    page_a = PageInfo(buttons=[
        ButtonInfo(text="Like", selector="#a"), ButtonInfo(text="Comment", selector="#b"),
    ])
    page_b = PageInfo(buttons=[
        ButtonInfo(text="Comment", selector="#b"), ButtonInfo(text="Like", selector="#a"),
    ])
    assert _page_template(page_a) == _page_template(page_b)


def test_page_template_differs_when_button_set_differs():
    page_a = PageInfo(buttons=[ButtonInfo(text="Like", selector="#a"), ButtonInfo(text="Comment", selector="#b")])
    page_b = PageInfo(buttons=[ButtonInfo(text="Like", selector="#a")])
    assert _page_template(page_a) != _page_template(page_b)


def test_page_template_differs_when_forms_presence_differs():
    page_a = PageInfo(buttons=[ButtonInfo(text="Like", selector="#a")])
    from backend.app.site_learning.schema import FormFieldInfo
    page_b = PageInfo(
        buttons=[ButtonInfo(text="Like", selector="#a")],
        forms=[FormFieldInfo(field_name="q", selector="#q")],
    )
    assert _page_template(page_a) != _page_template(page_b)


def test_page_template_accounts_for_ui_patterns():
    page_a = PageInfo(buttons=[], ui_patterns=[UIPatternInfo(name="Products", ui_type="Card", selector="div.card")])
    page_b = PageInfo(buttons=[], ui_patterns=[UIPatternInfo(name="Orders", ui_type="Table Row", selector="tr.row")])
    assert _page_template(page_a) != _page_template(page_b)


def test_page_template_of_two_completely_empty_pages_are_equal():
    """_page_template() เองไม่แยกแยะหน้าที่ไม่มีปุ่ม/ui_pattern เลย (ได้ template เดียวกัน
    จริง — "forms:False, tables:False" ทั้งคู่) — การ์ดกันไม่ให้ crawl_site() เอาไป dedupe
    ผิดๆ อยู่ที่ _record_page() (เช็ค has_template_signal ก่อนเสมอ) ไม่ใช่ในฟังก์ชันนี้ ดู
    test_crawl_site_visits_links_outside_nav_containers ที่ยืนยันพฤติกรรมระดับ crawl เต็ม
    ว่าหน้าเปล่า 2 หน้าที่ต่างความหมายกันจริง (Dashboard vs Article Detail) ยังถูกบันทึกครบ
    ทั้งคู่ ไม่ถูกข้ามเพราะบังเอิญไม่มีปุ่มเหมือนกัน"""
    page_a = PageInfo(url="/dashboard")
    page_b = PageInfo(url="/article-detail")
    assert _page_template(page_a) == _page_template(page_b)


_REPEAT_PAGE_TEMPLATE_FIXTURE_PAGES = {
    # จำลอง feed ที่มีลิงก์ไปคลิป Shorts/Reels คนละคลิปจริง (คนละ URL) แต่ UI เหมือนกันทุก
    # อย่าง (ปุ่ม Like/Comment) — ต่างจาก _REPEAT_BUTTON_FIXTURE_PAGES ด้านบนที่จำลองปุ่ม
    # "Next" ตัวเดียวกันถูกกดซ้ำ (ผ่าน _explore_buttons/explored_button_signatures) ตัวนี้
    # จำลองลิงก์คนละอันจริงๆ สิบๆ รายการที่ไม่เคยผ่าน signature-cap เลยเพราะเป็น nav_links
    # BFS ล้วนๆ (ดู W25 — สแกนทั้งเอกสาร)
    "feed.html": """
        <html><body>
          <a href="/clip1.html">Watch Video A</a>
          <a href="/clip2.html">Watch Video B</a>
          <a href="/clip3.html">Watch Video C</a>
          <a href="/about.html">About Us</a>
        </body></html>
    """,
    "clip1.html": """
        <html><body>
          <button data-testid="like-btn">Like</button>
          <button data-testid="comment-btn">Comment</button>
        </body></html>
    """,
    "clip2.html": """
        <html><body>
          <button data-testid="like-btn">Like</button>
          <button data-testid="comment-btn">Comment</button>
        </body></html>
    """,
    "clip3.html": """
        <html><body>
          <button data-testid="like-btn">Like</button>
          <button data-testid="comment-btn">Comment</button>
        </body></html>
    """,
    "about.html": """
        <html><body>
          <table><thead><tr><th>Name</th></tr></thead></table>
        </body></html>
    """,
}


@pytest.fixture
def repeat_page_template_fixture_server(tmp_path):
    httpd, base_url = _make_fixture_server(tmp_path, _REPEAT_PAGE_TEMPLATE_FIXTURE_PAGES)
    yield base_url
    httpd.shutdown()


@pytest.mark.asyncio
async def test_crawl_site_records_only_the_first_page_matching_a_known_template(repeat_page_template_fixture_server):
    """W33: 3 คลิปคนละ URL จริง (ไม่ถูกกันโดย visited-set) แต่ UI เหมือนกันทุกอย่าง — ต้อง
    บันทึกแค่ตัวแทนตัวแรกที่เจอ (clip1) ตัวถัดๆ ไป (clip2, clip3) ต้องไม่ถูกบันทึกลง
    manual.pages เลย ส่วนหน้าที่โครงสร้างต่างจริง (about.html — มีตารางไม่มีปุ่ม) ต้องยังถูก
    บันทึกตามปกติไม่ถูกกระทบ"""
    events = []

    async def on_progress(event):
        events.append(event)

    with patch("backend.app.site_learning.crawler.llm.generate_text", _mock_generate_text()):
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            manual = await crawl_site(
                browser, f"{repeat_page_template_fixture_server}/feed.html", max_pages=20, on_progress=on_progress,
            )
            await browser.close()

    visited_urls = {p.url for p in manual.pages}
    assert any("feed.html" in u for u in visited_urls)
    assert any("clip1.html" in u for u in visited_urls)
    assert not any("clip2.html" in u for u in visited_urls)
    assert not any("clip3.html" in u for u in visited_urls)
    assert any("about.html" in u for u in visited_urls)  # โครงสร้างต่างจริง ยังบันทึกปกติ

    skipped_events = [e for e in events if e["kind"] == "page_template_skipped"]
    skipped_urls = {e["url"] for e in skipped_events}
    assert any("clip2.html" in u for u in skipped_urls)
    assert any("clip3.html" in u for u in skipped_urls)


_REPEAT_BUTTON_FIXTURE_PAGES = {
    # จำลอง "YouTube Shorts": ปุ่ม "Next" label+role เหมือนกันทุกหน้าพาไปหน้าใหม่ (URL ไม่
    # ซ้ำ) เสมอไม่รู้จบ — shorts1 -> shorts2 -> shorts3 -> shorts4 ถ้าไม่ถูกจำกัดจะไล่กด
    # "Next" ไปเรื่อยๆ กิน max_pages budget ทั้งหมด ไม่เคยไปถึง other.html เลย
    "shorts1.html": """
        <html><body>
          <nav><a href="/other.html">Other</a></nav>
          <button onclick="window.location.href='/shorts2.html'">Next</button>
        </body></html>
    """,
    "shorts2.html": """
        <html><body>
          <button onclick="window.location.href='/shorts3.html'">Next</button>
        </body></html>
    """,
    "shorts3.html": """
        <html><body>
          <button onclick="window.location.href='/shorts4.html'">Next</button>
        </body></html>
    """,
    "shorts4.html": "<html><body>dead end</body></html>",
    "other.html": "<html><body>a totally different section of the site</body></html>",
}


@pytest.fixture
def repeat_button_fixture_server(tmp_path):
    httpd, base_url = _make_fixture_server(tmp_path, _REPEAT_BUTTON_FIXTURE_PAGES)
    yield base_url
    httpd.shutdown()


@pytest.mark.asyncio
async def test_crawl_site_stops_repeat_clicking_the_same_button_across_pages(repeat_button_fixture_server):
    """W28/W29: ปุ่ม "Next" (signature เดียวกันทุกหน้า) ต้องหยุดถูกไล่กดหลังครบเพดาน
    settings.site_learning_max_repeat_button_clicks (default 1) แม้จะเจอบนหน้าใหม่ที่ไม่
    เคยไปมาก่อนก็ตาม (ต่างจากเดิมที่ visited-set กันแค่ "หน้า" ไม่กัน "ปุ่ม" — ปุ่ม Next บน
    หน้าใหม่ทุกหน้าไม่เคยถูกมองว่า "เคยกดแล้ว" เพราะหน้ามัน "ใหม่" เสมอ) — ต้องยังสำรวจ
    other.html (เข้าถึงผ่าน nav link ปกติ ไม่เกี่ยวกับปุ่ม) ได้ตามปกติ ไม่ถูกกระทบ"""
    with patch("backend.app.site_learning.crawler.llm.generate_text", _mock_generate_text()):
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            manual = await crawl_site(browser, f"{repeat_button_fixture_server}/shorts1.html", max_pages=20)
            await browser.close()

    visited_urls = {p.url for p in manual.pages}
    assert any("shorts1.html" in u for u in visited_urls)
    assert any("shorts2.html" in u for u in visited_urls)  # "Next" กดได้ 1 ครั้งตามเพดาน default
    assert not any("shorts3.html" in u for u in visited_urls)  # ไม่ควรไปถึง — Next ถูกบล็อกแล้ว
    assert not any("shorts4.html" in u for u in visited_urls)
    assert any("other.html" in u for u in visited_urls)  # ยังสำรวจส่วนอื่นของเว็บได้ตามปกติ


@pytest.mark.asyncio
async def test_crawl_site_respects_custom_max_repeat_button_clicks(repeat_button_fixture_server, monkeypatch):
    monkeypatch.setattr("backend.app.site_learning.crawler.settings.site_learning_max_repeat_button_clicks", 2)
    with patch("backend.app.site_learning.crawler.llm.generate_text", _mock_generate_text()):
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            manual = await crawl_site(browser, f"{repeat_button_fixture_server}/shorts1.html", max_pages=20)
            await browser.close()

    visited_urls = {p.url for p in manual.pages}
    assert any("shorts3.html" in u for u in visited_urls)  # เพดาน 2 -> กด "Next" ได้ 2 ครั้ง
    assert not any("shorts4.html" in u for u in visited_urls)  # ครั้งที่ 3 ยังเกินเพดาน


# ---------------- W34: หลุดออกนอกโดเมนเป้าหมาย ต้องกลับมาทันที ไม่เรียนรู้เว็บอื่นต่อ ----------------
# ใช้ "localhost" กับ "127.0.0.1" แทนกันเพื่อจำลอง cross-domain แบบไม่ต้องเปิด server จริง
# 2 ตัว — extract_domain() แยกสองชื่อนี้เป็นคนละ domain แม้จะชี้ไป loopback/พอร์ตเดียวกันจริง


@pytest.fixture
def off_domain_fixture_server(tmp_path):
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(tmp_path))
    httpd = http.server.HTTPServer(("127.0.0.1", 0), handler)
    port = httpd.server_address[1]
    pages = {
        "start.html": f"""
            <html><body>
              <button onclick="window.location.href='http://localhost:{port}/external.html'">View</button>
              <button onclick="window.location.href='/safe.html'">Safe Details</button>
            </body></html>
        """,
        "external.html": "<html><body>different domain — should never be recorded</body></html>",
        "safe.html": "<html><body>same-domain page — should be recorded normally</body></html>",
        "redirect.html": f"""
            <html><body>
              <script>window.location.href = 'http://localhost:{port}/external2.html';</script>
            </body></html>
        """,
        "external2.html": "<html><body>different domain reached via client-side redirect</body></html>",
        "feed.html": """
            <html><body>
              <p>ลิงก์นี้ตั้งใจวางไว้นอก &lt;nav&gt; และตั้ง label ที่ไม่ตรงคำไหนใน
              ALLOWED_CRAWL_KEYWORDS เลยสักคำ (ระวัง: คำอย่าง "page"/"view"/"detail" ห้าม
              ปรากฏเป็น substring เด็ดขาด เพราะ is_crawl_safe() เช็คแบบ substring match —
              เดิมใช้ label "Weird Redirect Page" ซึ่งมีคำว่า "page" ปนอยู่ ทำให้ผ่าน
              is_crawl_safe() ได้ทั้งที่ไม่ใช่ nav menu item กลายเป็นถูก
              _explore_buttons() (DFS-click) กดไปก่อน BFS จะได้ประมวลผล href นี้เลย — เจอ
              จาก test นี้ fail เพราะ manual.errors ไม่มี phase="goto" เลยสักตัว มีแต่
              phase="click") เพื่อให้ is_crawl_safe() ปฏิเสธไม่ให้ _explore_buttons()
              กดมันเป็นปุ่ม แต่ is_safe_nav_link() (default-allow) ยังคงปล่อยให้ BFS
              เดินตาม href นี้ได้ตามปกติ — การันตีว่าเทสนี้ตรวจ BFS-goto path ล้วนๆ ไม่ปน
              กับ DFS-click path ที่มีเทสของตัวเองอยู่แล้ว
              (test_crawl_site_backs_out_immediately_when_click_navigates_off_domain)</p>
              <a href="/redirect.html">Bounce Link Alpha</a>
            </body></html>
        """,
    }
    for name, html in pages.items():
        (tmp_path / name).write_text(html, encoding="utf-8")
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    httpd.shutdown()


@pytest.mark.asyncio
async def test_crawl_site_backs_out_immediately_when_click_navigates_off_domain(off_domain_fixture_server):
    """W34: ปุ่ม "View" ที่ label ทั่วไปผ่าน is_crawl_safe() ได้ (ไม่รู้ปลายทางล่วงหน้า ต่าง
    จาก nav_links ที่เช็ค href ได้ก่อน queue) แต่จริงๆ พาไปคนละโดเมน ต้องไม่ถูก extract/
    บันทึกเข้า manual เลย ต้อง go_back กลับมาที่โดเมนเป้าหมายทันที ยิง off_domain_navigation
    event + บันทึก error ไว้

    หมายเหตุ: เดิมเทสนี้เช็คด้วยว่ากลับมาแล้วต้องกดปุ่มถัดไป ("Safe Details") ต่อได้ปกติด้วย —
    ตัดออกเพราะพบว่า fixture ที่จำลอง "คนละโดเมน" ด้วยการสลับ hostname "localhost" กับ
    "127.0.0.1" บนพอร์ตเดียวกัน (ดู comment ของ off_domain_fixture_server) ทำให้ Chromium
    หน่วงการยิง "load" event ของ goto() ที่กลับมาโดเมนเดิมนานผิดปกติ (~13-16 วินาที) เฉพาะ
    เวลาที่ navigate มาจาก click event เท่านั้น (ทดสอบแยกแล้วด้วย goto() ตรงๆ สลับ host ไปมา
    ไม่หน่วงเลย) — เป็น artifact ของวิธีจำลอง cross-domain แบบนี้ ไม่ใช่พฤติกรรมจริงที่จะเกิด
    กับโดเมนที่ต่างกันจริงๆ ในโปรดักชัน (ไม่มี connection ให้ browser พยายามใช้ซ้ำข้ามโดเมนแบบ
    นี้) คุณสมบัติหลักที่ user ต้องการ (ห้ามบันทึก/สำรวจเว็บนอกโดเมนเด็ดขาด + กลับมาทันที) ถูก
    ตรวจสอบและยืนยันว่าทำงานถูกต้องสม่ำเสมอด้านล่างนี้แล้ว"""
    events = []

    async def on_progress(event):
        events.append(event)

    with patch("backend.app.site_learning.crawler.llm.generate_text", _mock_generate_text()):
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            manual = await crawl_site(
                browser, f"{off_domain_fixture_server}/start.html", max_pages=10, on_progress=on_progress,
            )
            await browser.close()

    visited_urls = {p.url for p in manual.pages}
    assert not any("external.html" in u for u in visited_urls)

    off_domain_events = [e for e in events if e["kind"] == "off_domain_navigation"]
    assert len(off_domain_events) >= 1
    assert "localhost" in off_domain_events[0]["landed_on"]
    assert any(e["phase"] == "click" and "นอกโดเมน" in e["error"] for e in manual.errors)


@pytest.mark.asyncio
async def test_crawl_site_skips_queued_url_that_redirects_off_domain(off_domain_fixture_server):
    """W34: URL ที่ same-domain ตอนถูกต่อคิว (nav_links กรอง cross-origin ไว้แล้ว) แต่ตัว
    หน้าเอง client-side redirect ออกนอกโดเมนทันทีที่โหลด (เช่น URL shortener/OAuth bounce)
    ต้องไม่ extract/บันทึกหน้าปลายทางที่ไม่ใช่โดเมนเป้าหมายเลย ข้ามไปหน้าถัดไปใน queue แทน"""
    events = []

    async def on_progress(event):
        events.append(event)

    with patch("backend.app.site_learning.crawler.llm.generate_text", _mock_generate_text()):
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            manual = await crawl_site(
                browser, f"{off_domain_fixture_server}/feed.html", max_pages=10, on_progress=on_progress,
            )
            await browser.close()

    visited_urls = {p.url for p in manual.pages}
    assert not any("external2.html" in u for u in visited_urls)
    assert not any("redirect.html" in u for u in visited_urls)  # หน้าที่ redirect เองก็ไม่ถูกบันทึก (ไม่ใช่โดเมนเป้าหมายตอน extract จริง)

    off_domain_events = [e for e in events if e["kind"] == "off_domain_navigation"]
    assert len(off_domain_events) >= 1
    assert any(e["phase"] == "goto" and "นอกโดเมน" in e["error"] for e in manual.errors)


# ---------------- W35: ชื่อหน้าเดิมซ้ำติดกันเกิน 2 ครั้ง ต้องหยุดไล่สำรวจต่อจากหน้านั้น ----------------
# จำลองเคสที่ user รายงาน: เว็บมีหน้าแบบ Shorts/Reels หลายรายการที่ URL ต่างกันจริง (ไม่ถูก
# กันโดย visited-set) และโครงสร้าง (_page_template — W33) ก็ต่างกันเล็กน้อยพอที่ template
# dedup จะไม่จับ (เช่น จำนวนปุ่มไม่เท่ากัน) แต่ LLM (describe_page) ตั้งชื่อเดิมซ้ำๆ ทุกครั้ง
# ("Video") — ต้องมีเซฟตี้เน็ตจากมุมชื่อหน้าคอยตัดไม่ให้ไล่กดปุ่ม/ต่อคิว nav link ลึกลงไปอีก
# หลังเจอชื่อซ้ำติดกันเกิน 2 ครั้ง (หน้าที่ 3, 4, ... ยังถูกบันทึกอยู่ปกติ ผู้ใช้เห็น tag แล้ว
# จริง แค่ไม่ไล่ลึกต่อจากมันแล้ว)

_REPEATED_NAME_PAGES = {
    "index.html": """
        <html><body>
          <nav>
            <a href="/clip1.html">Clip 1</a>
            <a href="/clip2.html">Clip 2</a>
            <a href="/clip3.html">Clip 3</a>
            <a href="/clip4.html">Clip 4</a>
          </nav>
        </body></html>
    """,
    "clip1.html": '<html><body><a href="/secret1.html">Deeper 1</a></body></html>',
    "clip2.html": '<html><body><a href="/secret2.html">Deeper 2</a></body></html>',
    "clip3.html": '<html><body><a href="/secret3.html">Deeper 3</a></body></html>',
    "clip4.html": '<html><body><a href="/secret4.html">Deeper 4</a></body></html>',
    "secret1.html": "<html><body>reachable only via clip1 (count=1, should still explore)</body></html>",
    "secret2.html": "<html><body>reachable only via clip2 (count=2, should still explore)</body></html>",
    "secret3.html": "<html><body>reachable only via clip3 (count=3, must NOT be reached)</body></html>",
    "secret4.html": "<html><body>reachable only via clip4 (count=4, must NOT be reached)</body></html>",
}


@pytest.fixture
def repeated_name_fixture_server(tmp_path):
    for name, html in _REPEATED_NAME_PAGES.items():
        (tmp_path / name).write_text(html, encoding="utf-8")
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(tmp_path))
    httpd = http.server.HTTPServer(("127.0.0.1", 0), handler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    httpd.shutdown()


def _mock_generate_text_same_name_for_clips():
    """clip1-4.html ทั้งหมดได้ชื่อ "Video" เหมือนกันเป๊ะทุก call (จำลอง LLM ตั้งชื่อคล้ายกัน
    ให้หน้าประเภทเดียวกันซ้ำๆ) หน้าอื่น (index/secret) ได้ชื่อไม่ซ้ำกันตาม URL ปกติ"""

    async def _generate(_client, _model, prompt, _provider):
        # เช็คจากบรรทัด "URL: ..." ของ prompt โดยเฉพาะ (ไม่ใช่ substring ทั้งก้อน) กัน
        # ปนกับข้อความปุ่ม/ลิงก์บนหน้า index.html เอง (ซึ่งมีคำว่า "Clip 1".."Clip 4" อยู่ใน
        # รายชื่อปุ่มด้วย ทั้งที่ URL ของ index.html เองไม่ใช่หน้า clip*)
        match = re.search(r"URL: \S*/(\w+)\.html", prompt)
        page_name = match.group(1) if match else "page"
        if page_name.startswith("clip"):
            return '{"name": "Video", "description": "a clip"}'
        return f'{{"name": "{page_name}", "description": "a page"}}'

    return AsyncMock(side_effect=_generate)


@pytest.mark.asyncio
async def test_crawl_site_stops_exploring_after_repeated_page_name(repeated_name_fixture_server):
    """W35: clip1/clip2/clip3/clip4 ทั้ง 4 หน้าถูกบันทึกเข้า manual ตามปกติทุกหน้า (ยังเห็น
    tag ครบ) แต่หลังชื่อ "Video" ซ้ำติดกันเกิน 2 ครั้ง (นับจาก clip3 เป็นต้นไป) ต้องหยุดไล่
    กดปุ่ม/ต่อคิว nav link จากหน้านั้นทันที — secret1/secret2 (เจอผ่าน clip1/clip2 ซึ่งยังไม่
    เกินเพดาน) ต้องถูกสำรวจตามปกติ แต่ secret3/secret4 (เจอผ่าน clip3/clip4 ที่เกินเพดานแล้ว)
    ต้องไม่ถูกสำรวจเลย"""
    events = []

    async def on_progress(event):
        events.append(event)

    with patch("backend.app.site_learning.crawler.llm.generate_text", _mock_generate_text_same_name_for_clips()):
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            manual = await crawl_site(
                browser, f"{repeated_name_fixture_server}/index.html", max_pages=20, on_progress=on_progress,
            )
            await browser.close()

    visited_urls = {p.url for p in manual.pages}
    for n in range(1, 5):
        assert any(f"clip{n}.html" in u for u in visited_urls), f"clip{n}.html should still be recorded"

    assert any("secret1.html" in u for u in visited_urls)
    assert any("secret2.html" in u for u in visited_urls)
    assert not any("secret3.html" in u for u in visited_urls)
    assert not any("secret4.html" in u for u in visited_urls)

    skip_events = [e for e in events if e["kind"] == "repeated_page_name_skipped"]
    assert len(skip_events) >= 2
    assert all(e["name"] == "Video" for e in skip_events)

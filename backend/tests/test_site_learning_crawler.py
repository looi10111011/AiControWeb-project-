import functools
import http.server
import json
import threading
from unittest.mock import AsyncMock, patch

import pytest
from playwright.async_api import async_playwright

from backend.app.site_learning.crawler import (
    _click_with_retry,
    _goto_with_retry,
    _reveal_dynamic_content,
    _wait_for_dom_stable,
    crawl_site,
)

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
    return AsyncMock(return_value='{"name": "Page", "description": "a page"}')


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

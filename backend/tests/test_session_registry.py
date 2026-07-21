import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.app.core.session_registry import BrowserSession, SessionRegistry


def _fake_pool():
    """แทน BrowserPool จริง — session_registry.py ต้องการแค่ acquire_one()/release_one()
    ไม่จำเป็นต้องมี playwright chain เต็มเหมือน test_browser_pool.py"""
    pool = MagicMock()
    pool.acquire_one = AsyncMock(return_value=AsyncMock(name="pooled_browser"))
    pool.release_one = AsyncMock()
    return pool


def _patch_async_playwright():
    mock_playwright_instance = AsyncMock()
    mock_playwright_instance.stop = AsyncMock()
    mock_p_helper = MagicMock()
    mock_p_helper.start = AsyncMock(return_value=mock_playwright_instance)
    mock_async_playwright = MagicMock(return_value=mock_p_helper)
    return mock_async_playwright, mock_playwright_instance


@pytest.mark.asyncio
async def test_get_or_create_pool_mode_acquires_from_pool_and_creates_context():
    pool = _fake_pool()
    mock_browser = pool.acquire_one.return_value
    mock_context = AsyncMock()
    mock_page = AsyncMock()
    mock_context.new_page = AsyncMock(return_value=mock_page)
    mock_browser.new_context = AsyncMock(return_value=mock_context)

    registry = SessionRegistry()
    session = await registry.get_or_create(
        "sess-1", use_user_browser=False, headless=None, target_url="https://example.com",
        pool=pool, tab_reuse_policy=None, ask_user_func=None,
    )

    assert session.mode == "pool"
    assert session.page is mock_page
    assert session.context is mock_context
    assert session.browser is mock_browser
    assert session.playwright is None
    assert session.pool is pool
    pool.acquire_one.assert_awaited_once()
    mock_browser.new_context.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_or_create_owns_mode_launches_own_browser():
    mock_async_playwright, mock_playwright_ctx = _patch_async_playwright()
    mock_browser = AsyncMock()
    mock_page = AsyncMock()
    mock_browser.new_page = AsyncMock(return_value=mock_page)
    mock_launch = AsyncMock(return_value=mock_browser)

    with patch("backend.app.core.session_registry.async_playwright", mock_async_playwright), \
         patch("backend.app.core.session_registry._launch_chromium", mock_launch), \
         patch("backend.app.core.session_registry._detect_default_browser_channel", return_value=None):
        registry = SessionRegistry()
        session = await registry.get_or_create(
            "sess-1", use_user_browser=False, headless=False, target_url="https://example.com",
            pool=_fake_pool(), tab_reuse_policy=None, ask_user_func=None,
        )

    assert session.mode == "owns"
    assert session.page is mock_page
    assert session.context is None
    assert session.playwright is mock_playwright_ctx
    mock_launch.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_or_create_user_browser_mode_connects_via_cdp():
    mock_async_playwright, mock_playwright_ctx = _patch_async_playwright()
    mock_context = MagicMock()
    mock_browser = AsyncMock()
    mock_browser.contexts = [mock_context]
    mock_page = AsyncMock()
    mock_connect = AsyncMock(return_value=mock_browser)
    mock_resolve = AsyncMock(return_value=(mock_page, True))

    with patch("backend.app.core.session_registry.async_playwright", mock_async_playwright), \
         patch("backend.app.core.session_registry.connect_user_browser", mock_connect), \
         patch("backend.app.core.session_registry.resolve_target_page", mock_resolve):
        registry = SessionRegistry()
        session = await registry.get_or_create(
            "sess-1", use_user_browser=True, headless=None, target_url="https://example.com",
            pool=_fake_pool(), tab_reuse_policy="always_reuse", ask_user_func=None,
        )

    assert session.mode == "user_browser"
    assert session.page is mock_page
    assert session.context is mock_context
    mock_connect.assert_awaited_once()
    mock_resolve.assert_awaited_once_with(mock_context, "https://example.com", None, "always_reuse")


@pytest.mark.asyncio
async def test_get_or_create_returns_existing_session_without_reacquiring():
    pool = _fake_pool()
    mock_browser = pool.acquire_one.return_value
    # W19: get_or_create() ครั้งที่สองด้านล่างเจอ session ที่มีอยู่แล้ว -> ต้องผ่าน
    # is_healthy() ก่อนถึงจะคืนตัวเดิมโดยไม่แตะอะไรเพิ่ม (ไม่งั้นจะเข้าใจผิดว่า resource
    # พังแล้วเข้า path กู้คืน/สร้างใหม่แทน) — is_connected()/is_closed() ของ Playwright
    # จริงเป็น sync method ต้อง set เป็น MagicMock ตรงๆ ไม่ปล่อยให้ AsyncMock auto-mock
    # เป็น async (เรียกแล้วได้ coroutine truthy เสมอ)
    mock_browser.is_connected = MagicMock(return_value=True)
    mock_page = AsyncMock()
    mock_page.is_closed = MagicMock(return_value=False)
    mock_context = AsyncMock()
    mock_context.new_page = AsyncMock(return_value=mock_page)
    mock_browser.new_context = AsyncMock(return_value=mock_context)

    registry = SessionRegistry()
    first = await registry.get_or_create(
        "sess-1", use_user_browser=False, headless=None, target_url="https://example.com",
        pool=pool, tab_reuse_policy=None, ask_user_func=None,
    )
    second = await registry.get_or_create(
        "sess-1", use_user_browser=False, headless=None, target_url="https://other.com",
        pool=pool, tab_reuse_policy=None, ask_user_func=None,
    )

    assert first is second
    pool.acquire_one.assert_awaited_once()  # ครั้งเดียวเท่านั้น ไม่ acquire ซ้ำรอบสอง


@pytest.mark.asyncio
async def test_get_or_create_concurrent_calls_for_new_session_id_create_only_once():
    pool = _fake_pool()
    mock_browser = pool.acquire_one.return_value
    # W19: ดู comment เดียวกันใน test_get_or_create_returns_existing_session_without_reacquiring
    mock_browser.is_connected = MagicMock(return_value=True)
    mock_page = AsyncMock()
    mock_page.is_closed = MagicMock(return_value=False)
    mock_context = AsyncMock()
    mock_context.new_page = AsyncMock(return_value=mock_page)
    mock_browser.new_context = AsyncMock(return_value=mock_context)

    registry = SessionRegistry()
    results = await asyncio.gather(
        registry.get_or_create(
            "sess-race", use_user_browser=False, headless=None, target_url="https://example.com",
            pool=pool, tab_reuse_policy=None, ask_user_func=None,
        ),
        registry.get_or_create(
            "sess-race", use_user_browser=False, headless=None, target_url="https://example.com",
            pool=pool, tab_reuse_policy=None, ask_user_func=None,
        ),
    )

    assert results[0] is results[1]
    pool.acquire_one.assert_awaited_once()


# W19: is_healthy() + auto-recovery — session ที่ browser/page เดิมตายไปแล้วจริง (user ปิด
# หน้าต่าง/tab เอง, crash) ต้องไม่ถูกคืนกลับไปใช้เฉยๆ ต้องกู้คืนอัตโนมัติก่อนเสมอ


def test_is_healthy_true_when_browser_connected_and_page_open():
    page = MagicMock()
    page.is_closed = MagicMock(return_value=False)
    browser = MagicMock()
    browser.is_connected = MagicMock(return_value=True)
    session = BrowserSession("s", "pool", page, MagicMock(), browser, None)

    assert SessionRegistry.is_healthy(session) is True


def test_is_healthy_false_when_browser_disconnected():
    page = MagicMock()
    page.is_closed = MagicMock(return_value=False)
    browser = MagicMock()
    browser.is_connected = MagicMock(return_value=False)
    session = BrowserSession("s", "pool", page, MagicMock(), browser, None)

    assert SessionRegistry.is_healthy(session) is False


def test_is_healthy_false_when_page_closed():
    page = MagicMock()
    page.is_closed = MagicMock(return_value=True)
    browser = MagicMock()
    browser.is_connected = MagicMock(return_value=True)
    session = BrowserSession("s", "pool", page, MagicMock(), browser, None)

    assert SessionRegistry.is_healthy(session) is False


def test_is_healthy_false_instead_of_throwing_when_check_itself_errors():
    """ไม่ throw เด็ดขาดแม้ attribute หายไป/error ระหว่างเช็คเอง (เช่น object ถูกทำลาย
    ไปแล้วบางส่วน) — ถือว่า "ไม่ healthy" เงียบๆ แทน ให้ caller กู้คืนตามปกติ"""
    browser = MagicMock()
    browser.is_connected = MagicMock(side_effect=RuntimeError("boom"))
    session = BrowserSession("s", "pool", MagicMock(), MagicMock(), browser, None)

    assert SessionRegistry.is_healthy(session) is False


@pytest.mark.asyncio
async def test_get_or_create_recovers_by_opening_new_page_when_page_closed_but_browser_alive():
    """browser ยังต่ออยู่จริง (is_connected=True) แต่ page เดิมถูกปิดไปแล้ว (user ปิด tab
    เอง) — ต้องเปิด page ใหม่ในบริบท (context) เดิม ไม่ใช่รื้อ browser/context ทั้งชุดทิ้ง
    ไม่จำเป็น (session_id/mode/browser/context ต้องเป็นตัวเดิมเป๊ะ มีแค่ page เปลี่ยน)"""
    pool = _fake_pool()
    mock_browser = pool.acquire_one.return_value
    mock_browser.is_connected = MagicMock(return_value=True)
    stale_page = AsyncMock()
    stale_page.is_closed = MagicMock(return_value=False)
    fresh_page = AsyncMock()
    fresh_page.is_closed = MagicMock(return_value=False)
    mock_context = AsyncMock()
    mock_context.new_page = AsyncMock(side_effect=[stale_page, fresh_page])
    mock_browser.new_context = AsyncMock(return_value=mock_context)

    registry = SessionRegistry()
    first = await registry.get_or_create(
        "sess-1", use_user_browser=False, headless=None, target_url="https://example.com",
        pool=pool, tab_reuse_policy=None, ask_user_func=None,
    )
    assert first.page is stale_page

    # จำลอง user ปิด tab เดิมเอง — page เดิมตายไปแล้วแต่ browser/context ยังอยู่
    stale_page.is_closed = MagicMock(return_value=True)

    second = await registry.get_or_create(
        "sess-1", use_user_browser=False, headless=None, target_url="https://example.com",
        pool=pool, tab_reuse_policy=None, ask_user_func=None,
    )

    assert second is first  # session object เดิม แก้ไข in-place แค่ page
    assert second.page is fresh_page
    assert second.browser is mock_browser
    assert second.context is mock_context
    pool.acquire_one.assert_awaited_once()  # ไม่ acquire browser ใหม่จาก pool เลย


@pytest.mark.asyncio
async def test_get_or_create_recovers_by_launching_new_browser_when_disconnected():
    """browser หลุดการเชื่อมต่อไปแล้วจริง (is_connected=False) — ต้องทิ้งของเก่าทั้งชุด
    แล้วสร้าง session ใหม่ทั้งหมดด้วย session_id เดิม (mode เดิม: pool -> acquire browser
    ใหม่จาก pool อีกครั้ง) ไม่ release_one() browser ที่ตายไปแล้วกลับเข้า pool เด็ดขาด
    (จะ poison pool ให้ task อื่นในอนาคตได้ browser ตายไปด้วย)"""
    pool = _fake_pool()
    dead_browser = pool.acquire_one.return_value
    dead_browser.is_connected = MagicMock(return_value=False)
    dead_page = AsyncMock()
    dead_page.is_closed = MagicMock(return_value=False)
    dead_context = AsyncMock()
    dead_context.new_page = AsyncMock(return_value=dead_page)
    dead_browser.new_context = AsyncMock(return_value=dead_context)

    fresh_browser = AsyncMock()
    fresh_browser.is_connected = MagicMock(return_value=True)
    fresh_page = AsyncMock()
    fresh_page.is_closed = MagicMock(return_value=False)
    fresh_context = AsyncMock()
    fresh_context.new_page = AsyncMock(return_value=fresh_page)
    fresh_browser.new_context = AsyncMock(return_value=fresh_context)
    pool.acquire_one = AsyncMock(side_effect=[dead_browser, fresh_browser])

    registry = SessionRegistry()
    first = await registry.get_or_create(
        "sess-1", use_user_browser=False, headless=None, target_url="https://example.com",
        pool=pool, tab_reuse_policy=None, ask_user_func=None,
    )
    assert first.browser is dead_browser

    second = await registry.get_or_create(
        "sess-1", use_user_browser=False, headless=None, target_url="https://example.com",
        pool=pool, tab_reuse_policy=None, ask_user_func=None,
    )

    assert second.browser is fresh_browser
    assert second.page is fresh_page
    assert second.session_id == "sess-1"
    assert pool.acquire_one.await_count == 2
    pool.release_one.assert_not_awaited()  # ห้ามคืน browser ที่ตายไปแล้วกลับเข้า pool


@pytest.mark.asyncio
async def test_get_or_create_owns_mode_recovers_with_new_page_when_page_closed():
    """mode="owns" ไม่มี context แยกจาก browser (context=None) — กู้คืนด้วย
    browser.new_page() ตรงๆ แทน context.new_page()"""
    mock_async_playwright, mock_playwright_ctx = _patch_async_playwright()
    mock_browser = AsyncMock()
    mock_browser.is_connected = MagicMock(return_value=True)
    stale_page = AsyncMock()
    stale_page.is_closed = MagicMock(return_value=False)
    fresh_page = AsyncMock()
    fresh_page.is_closed = MagicMock(return_value=False)
    mock_browser.new_page = AsyncMock(side_effect=[stale_page, fresh_page])
    mock_launch = AsyncMock(return_value=mock_browser)

    with patch("backend.app.core.session_registry.async_playwright", mock_async_playwright), \
         patch("backend.app.core.session_registry._launch_chromium", mock_launch), \
         patch("backend.app.core.session_registry._detect_default_browser_channel", return_value=None):
        registry = SessionRegistry()
        first = await registry.get_or_create(
            "sess-1", use_user_browser=False, headless=False, target_url="https://example.com",
            pool=_fake_pool(), tab_reuse_policy=None, ask_user_func=None,
        )
        assert first.page is stale_page

        stale_page.is_closed = MagicMock(return_value=True)

        second = await registry.get_or_create(
            "sess-1", use_user_browser=False, headless=False, target_url="https://example.com",
            pool=_fake_pool(), tab_reuse_policy=None, ask_user_func=None,
        )

    assert second is first
    assert second.page is fresh_page
    mock_launch.assert_awaited_once()  # ไม่ relaunch browser ใหม่ทั้งตัว แค่เปิด page ใหม่


@pytest.mark.asyncio
async def test_close_pool_mode_returns_browser_and_closes_context():
    mock_context = AsyncMock()
    mock_browser = AsyncMock()
    pool = _fake_pool()
    session = BrowserSession("sess-1", "pool", AsyncMock(), mock_context, mock_browser, None, pool=pool)
    registry = SessionRegistry()
    registry._sessions["sess-1"] = session

    closed = await registry.close("sess-1")

    assert closed is True
    mock_context.close.assert_awaited_once()
    pool.release_one.assert_awaited_once_with(mock_browser)
    assert registry.get("sess-1") is None


@pytest.mark.asyncio
async def test_close_owns_mode_closes_browser_and_stops_playwright():
    mock_browser = AsyncMock()
    mock_playwright = AsyncMock()
    session = BrowserSession("sess-1", "owns", AsyncMock(), None, mock_browser, mock_playwright)
    registry = SessionRegistry()
    registry._sessions["sess-1"] = session

    closed = await registry.close("sess-1")

    assert closed is True
    mock_browser.close.assert_awaited_once()
    mock_playwright.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_close_user_browser_mode_only_stops_playwright_never_touches_browser():
    mock_context = MagicMock()
    mock_browser = AsyncMock()
    mock_playwright = AsyncMock()
    session = BrowserSession("sess-1", "user_browser", AsyncMock(), mock_context, mock_browser, mock_playwright)
    registry = SessionRegistry()
    registry._sessions["sess-1"] = session

    closed = await registry.close("sess-1")

    assert closed is True
    mock_playwright.stop.assert_awaited_once()
    mock_browser.close.assert_not_awaited()  # ห้ามปิด browser จริงของ user เด็ดขาด


@pytest.mark.asyncio
async def test_close_unknown_session_id_returns_false():
    registry = SessionRegistry()
    assert await registry.close("does-not-exist") is False


def test_list_returns_sessions_newest_first():
    registry = SessionRegistry()
    older = BrowserSession("a", "pool", AsyncMock(), AsyncMock(), AsyncMock(), None, created_at=1.0)
    newer = BrowserSession("b", "owns", AsyncMock(), None, AsyncMock(), AsyncMock(), created_at=2.0)
    registry._sessions["a"] = older
    registry._sessions["b"] = newer

    assert registry.list() == [newer, older]


@pytest.mark.asyncio
async def test_close_all_closes_every_session():
    pool = _fake_pool()
    sessions = {
        "a": BrowserSession("a", "pool", AsyncMock(), AsyncMock(), AsyncMock(), None, pool=pool),
        "b": BrowserSession("b", "owns", AsyncMock(), None, AsyncMock(), AsyncMock()),
    }
    registry = SessionRegistry()
    registry._sessions.update(sessions)

    await registry.close_all()

    assert registry.get("a") is None
    assert registry.get("b") is None
    sessions["a"].context.close.assert_awaited_once()
    sessions["b"].browser.close.assert_awaited_once()

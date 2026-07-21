from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.app.core.user_browser import (
    UserBrowserConnectError,
    connect_user_browser,
    resolve_target_page,
)


def _mock_page(url: str) -> MagicMock:
    page = MagicMock()
    page.url = url
    page.goto = AsyncMock()
    page.bring_to_front = AsyncMock()
    page.evaluate = AsyncMock()
    return page


class _FakeExpectPageCM:
    """จำลอง context.expect_page() ของจริง — async context manager ที่มี .value เป็น
    coroutine ให้ await แล้วได้ page ใหม่ที่ "เกิดขึ้น" ระหว่างอยู่ใน block (ของจริงรอ
    event "page" จาก context จริงๆ — เทสต์นี้ไม่ต้องรอ event จริง แค่คืน page ที่เตรียมไว้
    ล่วงหน้าตรงๆ)"""

    def __init__(self, new_page: MagicMock):
        self._new_page = new_page

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    @property
    def value(self):
        async def _get():
            return self._new_page
        return _get()


def _mock_context(pages: list, new_tab_page: MagicMock = None) -> MagicMock:
    context = MagicMock()
    context.pages = pages
    # fallback ตรงๆ ตอน context ว่างเปล่าจริงๆ (ไม่มี page ให้ evaluate window.open())
    context.new_page = AsyncMock(return_value=_mock_page("about:blank"))
    opened = new_tab_page if new_tab_page is not None else _mock_page("about:blank")
    context.expect_page = MagicMock(return_value=_FakeExpectPageCM(opened))
    return context


@pytest.mark.asyncio
async def test_connect_user_browser_calls_connect_over_cdp_with_url():
    mock_browser = AsyncMock()
    mock_playwright = AsyncMock()
    mock_playwright.chromium.connect_over_cdp = AsyncMock(return_value=mock_browser)

    result = await connect_user_browser(mock_playwright, "http://localhost:9222")

    mock_playwright.chromium.connect_over_cdp.assert_awaited_once_with("http://localhost:9222")
    assert result is mock_browser


@pytest.mark.asyncio
async def test_connect_user_browser_raises_clear_error_on_connect_failure():
    mock_playwright = AsyncMock()
    mock_playwright.chromium.connect_over_cdp = AsyncMock(side_effect=RuntimeError("connection refused"))

    with pytest.raises(UserBrowserConnectError) as exc_info:
        await connect_user_browser(mock_playwright, "http://localhost:9222")

    assert "9222" in str(exc_info.value)
    assert "--remote-debugging-port" in str(exc_info.value)


# tab ใหม่ต้องเปิดผ่าน window.open() ที่ evaluate จาก page ที่มีอยู่แล้วใน context เสมอ
# (ไม่ใช่ context.new_page() ตรงๆ) — เพราะ context.new_page() ผ่าน CDP
# (Target.createTarget) ไม่การันตีว่าจะแนบเข้า window เดิมที่ user กำลังใช้งานอยู่ (เคย
# เจอจริงว่า Chrome เปิดเป็น window แยกใหม่แทน) ส่วน window.open() ที่เรียกจาก page ของ
# window ไหน รับประกันว่า tab ใหม่จะอยู่ใน window นั้นเสมอตามพฤติกรรมมาตรฐานของ browser


@pytest.mark.asyncio
async def test_resolve_target_page_opens_new_tab_when_no_matching_domain_tab():
    existing = _mock_page("https://example.com/")
    new_tab = _mock_page("about:blank")
    context = _mock_context([existing], new_tab_page=new_tab)
    ask_user_func = AsyncMock()

    page, opened_new_tab = await resolve_target_page(
        context, "https://www.saucedemo.com/", ask_user_func, tab_reuse_policy="ask",
    )

    ask_user_func.assert_not_awaited()  # ไม่เจอ tab ตรง ไม่ต้องถาม
    assert opened_new_tab is True
    assert page is new_tab
    context.new_page.assert_not_awaited()  # ต้องไม่ใช้ context.new_page() ตรงๆ
    existing.evaluate.assert_awaited_once()  # เปิด tab ใหม่ผ่าน window.open() จาก page ที่มีอยู่แล้ว
    # ไม่ goto() เองในนี้ — ผู้เรียก (orchestrator.py) เป็นคน goto ต่อ (ดู docstring
    # ของ resolve_target_page())
    page.goto.assert_not_awaited()
    page.bring_to_front.assert_awaited_once()


@pytest.mark.asyncio
async def test_resolve_target_page_opens_new_tab_via_new_page_when_context_has_no_pages():
    """edge case: context ว่างเปล่าจริงๆ ไม่มี page ให้ evaluate window.open() เลย —
    ไม่มีทางเลือกอื่นแล้ว fallback เป็น context.new_page() ตรงๆ"""
    context = _mock_context([])
    ask_user_func = AsyncMock()

    page, opened_new_tab = await resolve_target_page(
        context, "https://www.saucedemo.com/", ask_user_func, tab_reuse_policy="ask",
    )

    assert opened_new_tab is True
    context.new_page.assert_awaited_once()


@pytest.mark.asyncio
async def test_resolve_target_page_asks_before_reuse_when_matching_tab_exists_and_policy_ask():
    matched = _mock_page("https://www.saucedemo.com/inventory.html")
    context = _mock_context([matched])
    ask_user_func = AsyncMock(return_value=True)

    await resolve_target_page(
        context, "https://www.saucedemo.com/", ask_user_func, tab_reuse_policy="ask",
    )

    ask_user_func.assert_awaited_once()
    cmd = ask_user_func.await_args.args[0]
    assert cmd["type"] == "confirm_tab_reuse"
    assert cmd["matched_tab_url"] == "https://www.saucedemo.com/inventory.html"


@pytest.mark.asyncio
async def test_resolve_target_page_reuses_tab_when_user_approves():
    matched = _mock_page("https://www.saucedemo.com/inventory.html")
    context = _mock_context([matched])
    ask_user_func = AsyncMock(return_value=True)

    page, opened_new_tab = await resolve_target_page(
        context, "https://www.saucedemo.com/", ask_user_func, tab_reuse_policy="ask",
    )

    assert page is matched
    assert opened_new_tab is False
    context.new_page.assert_not_awaited()
    matched.evaluate.assert_not_awaited()


@pytest.mark.asyncio
async def test_resolve_target_page_opens_new_tab_when_user_declines_reuse():
    matched = _mock_page("https://www.saucedemo.com/inventory.html")
    new_tab = _mock_page("about:blank")
    context = _mock_context([matched], new_tab_page=new_tab)
    ask_user_func = AsyncMock(return_value=False)

    page, opened_new_tab = await resolve_target_page(
        context, "https://www.saucedemo.com/", ask_user_func, tab_reuse_policy="ask",
    )

    assert page is new_tab
    assert opened_new_tab is True
    matched.evaluate.assert_awaited_once()  # opener คือ context.pages[0] ซึ่งคือ matched ในเคสนี้


@pytest.mark.asyncio
async def test_resolve_target_page_always_new_tab_policy_never_asks():
    matched = _mock_page("https://www.saucedemo.com/inventory.html")
    new_tab = _mock_page("about:blank")
    context = _mock_context([matched], new_tab_page=new_tab)
    ask_user_func = AsyncMock()

    page, opened_new_tab = await resolve_target_page(
        context, "https://www.saucedemo.com/", ask_user_func, tab_reuse_policy="always_new_tab",
    )

    ask_user_func.assert_not_awaited()
    assert opened_new_tab is True
    assert page is new_tab


@pytest.mark.asyncio
async def test_resolve_target_page_always_reuse_policy_never_asks():
    matched = _mock_page("https://www.saucedemo.com/inventory.html")
    context = _mock_context([matched])
    ask_user_func = AsyncMock()

    page, opened_new_tab = await resolve_target_page(
        context, "https://www.saucedemo.com/", ask_user_func, tab_reuse_policy="always_reuse",
    )

    ask_user_func.assert_not_awaited()
    assert opened_new_tab is False
    assert page is matched


@pytest.mark.asyncio
async def test_resolve_target_page_falls_back_to_terminal_input_when_ask_user_func_none():
    matched = _mock_page("https://www.saucedemo.com/inventory.html")
    context = _mock_context([matched])

    with patch("backend.app.core.user_browser.asyncio.to_thread", AsyncMock(return_value="y")):
        page, opened_new_tab = await resolve_target_page(
            context, "https://www.saucedemo.com/", None, tab_reuse_policy="ask",
        )

    assert page is matched
    assert opened_new_tab is False


@pytest.mark.asyncio
async def test_resolve_target_page_raises_on_invalid_tab_reuse_policy():
    context = _mock_context([])
    with pytest.raises(ValueError):
        await resolve_target_page(context, "https://www.saucedemo.com/", None, tab_reuse_policy="bogus")

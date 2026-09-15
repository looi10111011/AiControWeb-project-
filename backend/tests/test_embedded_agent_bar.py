"""The embedded bar must control its sender, including multiple localhost tabs."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from backend.app.api.schemas import CreateTaskRequest
from backend.app.core.session_registry import BrowserSession, SessionRegistry
from backend.app.core.user_browser import UserBrowserConnectError, resolve_target_page


def page(url, marker):
    result = MagicMock(url=url)
    result.evaluate = AsyncMock(return_value=marker)
    result.bring_to_front = AsyncMock()
    return result


@pytest.mark.asyncio
async def test_sender_selected_among_same_url_and_other_localhost_ports():
    console = page("http://localhost:8000/", "sender")
    other = page("http://localhost:8100/dashboard", "other")
    sender = page("http://localhost:8100/dashboard", "sender")
    context = MagicMock(pages=[console, other, sender])
    selected, opened = await resolve_target_page(
        context, sender.url, None, "always_reuse", target_tab_id="sender"
    )
    assert selected is sender and not opened
    console.evaluate.assert_not_awaited()
    context.new_page.assert_not_called()


@pytest.mark.asyncio
async def test_missing_sender_never_opens_another_tab():
    context = MagicMock(pages=[page("http://localhost:8100/dashboard", "other")])
    with pytest.raises(UserBrowserConnectError):
        await resolve_target_page(
            context, "http://localhost:8100/dashboard", None,
            "always_reuse", target_tab_id="missing",
        )
    context.new_page.assert_not_called()
    context.expect_page.assert_not_called()


@pytest.mark.asyncio
async def test_embedded_session_never_recovers_into_new_tab():
    registry = SessionRegistry()
    context = MagicMock(pages=[])
    session = BrowserSession("session", "user_browser", MagicMock(), context, MagicMock(), None, owner_token="owner")
    registry._sessions["session"] = session
    with patch.object(registry, "_recover", new_callable=AsyncMock) as recover:
        with pytest.raises(UserBrowserConnectError):
            await registry.get_or_create(
                "session", use_user_browser=True, headless=None,
                target_url="http://localhost:8100/dashboard", pool=MagicMock(),
                tab_reuse_policy="always_reuse", ask_user_func=None,
                owner_token="owner", target_tab_id="closed-tab",
            )
        recover.assert_not_awaited()


def test_embedded_request_requires_user_browser_session():
    with pytest.raises(ValidationError):
        CreateTaskRequest(url="http://localhost:8100", goal="PIM", target_tab_id="sender")

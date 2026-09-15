import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.api.routes import router
from backend.app.api.task_manager import TaskManager
from backend.app.config import settings
from backend.app.core.embedded_page import EmbeddedPage, PageExchange, PageSnapshot, run_embedded_task
from backend.app.core.llm import TokenUsage


def snapshot(document_id="home", url="http://localhost:8100/dashboard"):
    return PageSnapshot(document_id=document_id, url=url, title="Dashboard", elements=[
        {"index": 1, "tag": "a", "label": "PIM", "href": "http://localhost:8100/pim/employees"},
    ])


@pytest.mark.asyncio
async def test_navigation_ack_resolves_once_with_destination_snapshot():
    page = EmbeddedPage(snapshot())
    pending = asyncio.create_task(page.dispatch({"type": "click", "index": 1}))
    await asyncio.sleep(0)
    command = page.exchange(PageExchange(token=page.token, snapshot=snapshot()))
    destination = snapshot("pim", "http://localhost:8100/pim/employees")
    ack = PageExchange(token=page.token, snapshot=destination, command_id=command["id"], success=True, message="navigated")
    assert page.exchange(ack) is None
    assert (await pending)["success"]
    assert page.snapshot.document_id == "pim"
    assert page.exchange(ack) is None


def test_exchange_rejects_other_owner_and_origin():
    page = EmbeddedPage(snapshot())
    with pytest.raises(PermissionError):
        page.exchange(PageExchange(token="wrong", snapshot=snapshot()))
    with pytest.raises(PermissionError):
        page.exchange(PageExchange(token=page.token, snapshot=snapshot(url="http://localhost:8000")))


@pytest.mark.asyncio
async def test_loop_uses_page_actions_and_observes_destination_before_finishing():
    page = EmbeddedPage(snapshot())
    model = AsyncMock(side_effect=[
        ("browser_action", {"type": "click", "index": 1}, "first", [], TokenUsage()),
        ("finish_task", {"success": True, "message": "PIM opened"}, "last", [], TokenUsage()),
    ])
    captured = []

    async def dispatch(cmd):
        captured.append(cmd)
        if cmd["type"] == "click":
            page.snapshot = snapshot("pim", "http://localhost:8100/pim/employees")
        return {"success": True, "message": "OK"}

    page.dispatch = dispatch
    with patch("backend.app.core.orchestrator.Orchestrator._llm_backend", return_value=(None, "model", model, lambda *args: [], None)):
        result = await run_embedded_task(page, "ไปหน้า PIM", "openai", 5, AsyncMock(), AsyncMock())
    assert result["success"]
    assert [c["type"] for c in captured] == ["snapshot", "click"]
    assert "pim/employees" in model.call_args_list[1].args[3]
    assert page.closed


@pytest.mark.asyncio
async def test_cancel_closes_pending_bridge():
    page = EmbeddedPage(snapshot())
    pending = asyncio.create_task(page.dispatch({"type": "snapshot"}))
    await asyncio.sleep(0)
    page.close()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert page.command is None


def test_embedded_api_never_acquires_browser_or_cdp(monkeypatch):
    monkeypatch.setattr(settings, "api_key", "")
    app = FastAPI()
    app.include_router(router)
    app.state.task_manager = TaskManager()
    # Deliberately no browser_pool or session_registry: this path must not touch them.
    async def runner(page, *args):
        await page.dispatch({"type": "snapshot"})
        return {"success": True, "message": "done"}

    with patch("backend.app.api.routes.run_embedded_task", runner), TestClient(app) as client:
        response = client.post("/tasks", json={"url": snapshot().url, "goal": "PIM", "embedded_page": snapshot().model_dump()})
        assert response.status_code == 202
        task = response.json()
        assert task["embedded_token"]
        denied = client.post(f'/tasks/{task["task_id"]}/page', json={"token": "wrong", "snapshot": snapshot().model_dump()})
        assert denied.status_code == 403
        exchange = client.post(f'/tasks/{task["task_id"]}/page', json={"token": task["embedded_token"], "snapshot": snapshot().model_dump()})
        assert exchange.status_code == 200
        assert exchange.json()["command"]["action"]["type"] == "snapshot"
        client.post(f'/tasks/{task["task_id"]}/stop')

"""W10[A]: เทสต์ endpoint ใหม่ (/tasks, /pool/status) ผ่าน fastapi.testclient.TestClient
เหมือน test_health.py — ต่างตรงที่ endpoint พวกนี้ต้องมี app.state.browser_pool /
app.state.task_manager ซึ่งถูกสร้างใน main.py::lifespan ตอน server startup จริง —
TestClient ต้องใช้แบบ `with TestClient(app) as client:` (context manager) ถึงจะ trigger
lifespan startup/shutdown จริง (ต่างจาก test_health.py ที่ไม่ต้องเพราะ /health ไม่แตะ
app.state เลย)

ห้ามให้ lifespan เปิด browser จริง (BrowserPool.start() ของจริงจะ launch Chromium จริง)
— patch backend.app.main.BrowserPool ด้วย fake ที่มี interface เดียวกัน (size/available/
start/shutdown/acquire) ก่อนเข้า TestClient context เสมอ
"""

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app


class _FakeBrowserPool:
    def __init__(self, size: int = 2, headless: bool | None = None):
        self._size = size
        self._available = size

    @property
    def size(self) -> int:
        return self._size

    @property
    def available(self) -> int:
        return self._available

    async def start(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    @asynccontextmanager
    async def acquire(self):
        self._available -= 1
        try:
            yield object()  # แค่ placeholder แทน browser จริง ไม่มี test ไหนแตะ attribute ของมัน
        finally:
            self._available += 1


@pytest.fixture
def client():
    with patch("backend.app.main.BrowserPool", _FakeBrowserPool):
        with TestClient(app) as c:
            yield c


_FAKE_RESULT = {
    "success": True,
    "steps": 2,
    "message": "เสร็จแล้ว",
    "history": [],
    "tokens": {"input": 10, "output": 5, "cache_read": 0, "cache_creation": 0},
    "plan": None,
    "final_page_state": "หน้าสุดท้าย",
}


def _poll_until(client, task_id: str, *, not_status: str = "running", timeout_s: float = 5.0) -> dict:
    """endpoint create_task คืน task_id ทันที (202) แต่ Orchestrator รันเป็น background
    asyncio.Task — poll GET /tasks/{id} จนกว่าจะเปลี่ยนสถานะจริง (แทน sleep เดา)"""
    import time

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        resp = client.get(f"/tasks/{task_id}")
        body = resp.json()
        if body["status"] != not_status:
            return body
        time.sleep(0.02)
    pytest.fail(f"task {task_id} ยังเป็นสถานะ {not_status!r} ไม่จบภายใน {timeout_s}s")


def test_health_still_works_with_lifespan(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_pool_status_reflects_fake_pool(client):
    resp = client.get("/pool/status")
    assert resp.status_code == 200
    assert resp.json() == {"size": 2, "available": 2, "in_use": 0}


def test_get_unknown_task_returns_404(client):
    resp = client.get("/tasks/does-not-exist")
    assert resp.status_code == 404


def test_list_tasks_empty_by_default(client):
    resp = client.get("/tasks")
    assert resp.status_code == 200
    assert resp.json() == []


def test_create_task_returns_202_then_completes_successfully(client):
    with patch("backend.app.api.routes.Orchestrator") as MockOrchestrator:
        MockOrchestrator.return_value.run_task = AsyncMock(return_value=_FAKE_RESULT)

        resp = client.post("/tasks", json={"url": "https://example.com", "goal": "ทดสอบ"})
        assert resp.status_code == 202
        body = resp.json()
        assert body["status"] == "running"
        task_id = body["task_id"]

        final = _poll_until(client, task_id)
        assert final["status"] == "done"
        assert final["result"] == _FAKE_RESULT
        assert final["error"] is None
        assert final["url"] == "https://example.com"
        assert final["goal"] == "ทดสอบ"

    # task ต้องปรากฏใน GET /tasks ด้วย
    listed = client.get("/tasks").json()
    assert any(t["task_id"] == task_id for t in listed)


def test_create_task_records_error_status_on_failure(client):
    with patch("backend.app.api.routes.Orchestrator") as MockOrchestrator:
        MockOrchestrator.return_value.run_task = AsyncMock(side_effect=RuntimeError("boom"))

        resp = client.post("/tasks", json={"url": "https://example.com", "goal": "ทดสอบ"})
        task_id = resp.json()["task_id"]

        final = _poll_until(client, task_id)
        assert final["status"] == "error"
        assert final["error"] == "boom"
        assert final["result"] is None


def test_create_task_default_denies_permission_gated_actions(client):
    """auto_approve default = False -> ask_user_func ที่ส่งเข้า run_task() ต้องปฏิเสธ
    เสมอ (fail closed) — เพราะไม่มี human อยู่หน้าจอคอยตอบ REST request ตรงๆ"""
    with patch("backend.app.api.routes.Orchestrator") as MockOrchestrator:
        mock_run_task = AsyncMock(return_value=_FAKE_RESULT)
        MockOrchestrator.return_value.run_task = mock_run_task

        resp = client.post("/tasks", json={"url": "https://example.com", "goal": "ทดสอบ"})
        _poll_until(client, resp.json()["task_id"])

    ask_user_func = mock_run_task.await_args.kwargs["ask_user_func"]
    assert asyncio.run(ask_user_func({"type": "purchase"})) is False


def test_create_task_auto_approve_true_approves_everything(client):
    with patch("backend.app.api.routes.Orchestrator") as MockOrchestrator:
        mock_run_task = AsyncMock(return_value=_FAKE_RESULT)
        MockOrchestrator.return_value.run_task = mock_run_task

        resp = client.post(
            "/tasks", json={"url": "https://example.com", "goal": "ทดสอบ", "auto_approve": True},
        )
        _poll_until(client, resp.json()["task_id"])

    ask_user_func = mock_run_task.await_args.kwargs["ask_user_func"]
    assert asyncio.run(ask_user_func({"type": "purchase"})) is True


def test_create_task_passes_pooled_browser_into_run_task(client):
    with patch("backend.app.api.routes.Orchestrator") as MockOrchestrator:
        mock_run_task = AsyncMock(return_value=_FAKE_RESULT)
        MockOrchestrator.return_value.run_task = mock_run_task

        resp = client.post("/tasks", json={"url": "https://example.com", "goal": "ทดสอบ"})
        _poll_until(client, resp.json()["task_id"])

    # browser= ที่ส่งเข้า run_task() ต้องมาจาก pool.acquire() จริง ไม่ใช่ None (ไม่งั้น
    # orchestrator จะเปิด browser process ใหม่เองแทนที่จะยืมจาก pool — ผิดจุดประสงค์ W10[A])
    assert mock_run_task.await_args.kwargs["browser"] is not None

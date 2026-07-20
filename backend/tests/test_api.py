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
import json
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.app.config import settings
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


def test_create_task_default_routes_permission_prompt_through_human_in_the_loop(client):
    """W10[B]: auto_approve default = False -> ask_user_func ที่ส่งเข้า run_task() ต้อง
    "ถาม" ผ่าน TaskManager.request_approval() (push event เข้า SSE stream + รอ POST
    /tasks/{id}/respond) แทนที่จะ deny เองเงียบๆ ทันทีเหมือนก่อนมี human-in-the-loop UI
    บนหน้าเว็บจริง — mock request_approval ตรงนี้ (ไม่ต้องเล่น queue/future จริงข้าม event
    loop ของ TestClient) ดู test_task_manager.py สำหรับเทสต์กลไก request_approval จริง"""
    with (
        patch("backend.app.api.routes.Orchestrator") as MockOrchestrator,
        patch(
            "backend.app.api.task_manager.TaskManager.request_approval", new_callable=AsyncMock
        ) as mock_request_approval,
    ):
        mock_request_approval.return_value = False
        mock_run_task = AsyncMock(return_value=_FAKE_RESULT)
        MockOrchestrator.return_value.run_task = mock_run_task

        resp = client.post("/tasks", json={"url": "https://example.com", "goal": "ทดสอบ"})
        task_id = resp.json()["task_id"]
        _poll_until(client, task_id)

        # ต้องเรียก ask_user_func ขณะ patch ยังไม่หลุด (ไม่งั้นจะไปโดน request_approval()
        # ตัวจริงที่ await asyncio.Future ค้างตลอดกาล เพราะไม่มีใคร resolve ให้)
        ask_user_func = mock_run_task.await_args.kwargs["ask_user_func"]
        assert asyncio.run(ask_user_func({"type": "purchase"})) is False
        # W10[E]: ต้องแนบ timeout เสมอ (settings.approval_timeout_seconds) กัน task ที่
        # ไม่มีใครตอบยึด browser จาก pool ไว้ตลอดกาล (ดู task_manager.py::request_approval)
        mock_request_approval.assert_awaited_once_with(
            task_id, {"type": "purchase"}, timeout=settings.approval_timeout_seconds
        )


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


def test_create_task_defaults_confirm_plan_to_true(client):
    """W10[B]: ไม่ส่ง confirm_plan มาเลย -> run_task() ต้องได้ confirm_plan=True (ค่า
    default ของ CreateTaskRequest) ให้หน้าเว็บเห็นแผนก่อนเริ่มทำงานเสมอ เว้นแต่ผู้เรียก
    ปิดเองตรงๆ"""
    with patch("backend.app.api.routes.Orchestrator") as MockOrchestrator:
        mock_run_task = AsyncMock(return_value=_FAKE_RESULT)
        MockOrchestrator.return_value.run_task = mock_run_task

        resp = client.post("/tasks", json={"url": "https://example.com", "goal": "ทดสอบ"})
        _poll_until(client, resp.json()["task_id"])

    assert mock_run_task.await_args.kwargs["confirm_plan"] is True


def test_create_task_with_headless_false_bypasses_pool_for_a_visible_browser(client):
    """W10[C]: headless=False ตรงๆ = user ขอเห็นหน้าต่าง browser จริง ("เปิดหน้าเว็ปจริง
    ขึ้นมารันคู่ไปด้วย") — ต้อง bypass pool ไปเลย (browser ใน pool ถูก launch แบบ headless
    ไว้ล่วงหน้าตั้งแต่ startup แล้ว เปลี่ยนทีหลังไม่ได้) และต้องส่ง keep_browser_open=True
    ให้ orchestrator (ไม่ปิดหน้าต่างจนกว่า user จะปิดเอง)"""
    with patch("backend.app.api.routes.Orchestrator") as MockOrchestrator:
        mock_run_task = AsyncMock(return_value=_FAKE_RESULT)
        MockOrchestrator.return_value.run_task = mock_run_task

        resp = client.post(
            "/tasks", json={"url": "https://example.com", "goal": "ทดสอบ", "headless": False},
        )
        _poll_until(client, resp.json()["task_id"])

        # pool ต้องไม่ถูกแตะเลย (available ยังเต็ม 2/2 เหมือนเดิม — ไม่มี acquire() เกิดขึ้น)
        assert client.get("/pool/status").json() == {"size": 2, "available": 2, "in_use": 0}

    kwargs = mock_run_task.await_args.kwargs
    assert kwargs.get("browser") is None
    assert kwargs["headless"] is False
    assert kwargs["keep_browser_open"] is True


def test_create_task_default_headless_still_uses_pool(client):
    """headless ไม่ได้ส่งมา (None) หรือ True -> ยังใช้ pool เหมือนเดิม (ทางเร็ว/ไม่โชว์
    หน้าต่าง) ไม่ใช่แค่ headless=False เท่านั้นที่ bypass"""
    with patch("backend.app.api.routes.Orchestrator") as MockOrchestrator:
        mock_run_task = AsyncMock(return_value=_FAKE_RESULT)
        MockOrchestrator.return_value.run_task = mock_run_task

        resp = client.post("/tasks", json={"url": "https://example.com", "goal": "ทดสอบ"})
        _poll_until(client, resp.json()["task_id"])

    kwargs = mock_run_task.await_args.kwargs
    assert kwargs.get("browser") is not None
    assert "keep_browser_open" not in kwargs


def test_stop_task_cancels_a_running_task(client):
    with patch("backend.app.api.routes.Orchestrator") as MockOrchestrator:
        finish = asyncio.Event()

        async def _never_finishes_until_stopped(**kwargs):
            await finish.wait()
            return _FAKE_RESULT

        MockOrchestrator.return_value.run_task = _never_finishes_until_stopped

        resp = client.post("/tasks", json={"url": "https://example.com", "goal": "ทดสอบ"})
        task_id = resp.json()["task_id"]

        stop_resp = client.post(f"/tasks/{task_id}/stop")
        assert stop_resp.status_code == 200
        assert stop_resp.json() == {"status": "stopping"}

        final = _poll_until(client, task_id)
        assert final["status"] == "cancelled"
        assert final["result"] is None

        finish.set()  # ปลด _never_finishes_until_stopped() ที่โดน cancel ไปแล้วให้จบสนิท


def test_stop_unknown_task_returns_404(client):
    resp = client.post("/tasks/does-not-exist/stop")
    assert resp.status_code == 404


def test_stop_already_finished_task_returns_409(client):
    with patch("backend.app.api.routes.Orchestrator") as MockOrchestrator:
        mock_run_task = AsyncMock(return_value=_FAKE_RESULT)
        MockOrchestrator.return_value.run_task = mock_run_task

        resp = client.post("/tasks", json={"url": "https://example.com", "goal": "ทดสอบ"})
        task_id = resp.json()["task_id"]
        _poll_until(client, task_id)

    resp = client.post(f"/tasks/{task_id}/stop")
    assert resp.status_code == 409


def test_stream_unknown_task_returns_404(client):
    resp = client.get("/tasks/does-not-exist/stream")
    assert resp.status_code == 404


def test_stream_finished_task_replays_synthesized_done_event(client):
    """W10[B]: ต่อ SSE *หลัง* task จบไปแล้ว (เช่น รีเฟรชหน้าเว็บ) ต้องไม่ hang รอ event ที่
    ไม่มีวันมาอีก — ต้องได้ task_done สังเคราะห์จาก record.status/result ที่ยังอยู่ทันที"""
    with patch("backend.app.api.routes.Orchestrator") as MockOrchestrator:
        mock_run_task = AsyncMock(return_value=_FAKE_RESULT)
        MockOrchestrator.return_value.run_task = mock_run_task

        resp = client.post("/tasks", json={"url": "https://example.com", "goal": "ทดสอบ"})
        task_id = resp.json()["task_id"]
        _poll_until(client, task_id)

        stream_resp = client.get(f"/tasks/{task_id}/stream")
        assert stream_resp.status_code == 200
        assert stream_resp.text.startswith("data: ")
        # json.dumps() escapes non-ASCII เป็น \uXXXX โดย default (ensure_ascii=True) —
        # decode ผ่าน json.loads() แทนการเทียบ substring ไทยตรงๆ (frontend ใช้
        # JSON.parse() ซึ่ง decode \uXXXX กลับเป็นข้อความเดิมให้เองอยู่แล้ว)
        payload = json.loads(stream_resp.text.removeprefix("data: ").strip())
        assert payload["kind"] == "task_done"
        assert payload["status"] == "done"
        assert payload["result"] == _FAKE_RESULT


def test_respond_unknown_request_id_returns_404(client):
    with patch("backend.app.api.routes.Orchestrator") as MockOrchestrator:
        mock_run_task = AsyncMock(return_value=_FAKE_RESULT)
        MockOrchestrator.return_value.run_task = mock_run_task

        resp = client.post("/tasks", json={"url": "https://example.com", "goal": "ทดสอบ"})
        task_id = resp.json()["task_id"]
        _poll_until(client, task_id)

    resp = client.post(f"/tasks/{task_id}/respond", json={"request_id": "does-not-exist", "approved": True})
    assert resp.status_code == 404


def test_respond_unknown_task_id_returns_404(client):
    resp = client.post("/tasks/does-not-exist/respond", json={"request_id": "x", "approved": True})
    assert resp.status_code == 404

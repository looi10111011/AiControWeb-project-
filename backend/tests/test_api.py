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
from unittest.mock import AsyncMock, MagicMock, patch

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
        browser = await self.acquire_one()
        try:
            yield browser
        finally:
            await self.release_one(browser)

    async def acquire_one(self):
        self._available -= 1
        # AsyncMock (ไม่ใช่ object() เปล่าๆ) เพราะ session flow (W12,
        # core/session_registry.py) เรียก browser.new_context() ต่อจริง — เทสต์อื่นที่ไม่
        # แตะ session ไม่สนใจว่าเป็น mock อะไรอยู่แล้ว (ไม่เคยเรียก attribute ไหนของมันเลย)
        # W19: is_connected()/is_closed() ของ Playwright จริงเป็น sync method (ไม่ await)
        # — ต้อง set เป็น MagicMock ธรรมดาตรงๆ ทั้ง browser/context/page ที่ auto-vivify
        # ต่อกันมา ไม่งั้น AsyncMock auto-mock attribute พวกนี้เป็น async mock ไปด้วย
        # (เรียกแล้วได้ coroutine object กลับมา ซึ่ง truthy เสมอ ทำให้
        # SessionRegistry.is_healthy() เข้าใจผิดว่า resource พังตลอดแม้ไม่ได้ตั้งใจเช็คค่า
        # จริงเลย) เหมือน page.on = MagicMock() ที่ test_orchestrator.py ทำไว้แล้ว
        page = AsyncMock()
        page.is_closed = MagicMock(return_value=False)
        context = AsyncMock()
        context.new_page = AsyncMock(return_value=page)
        browser = AsyncMock()
        browser.is_connected = MagicMock(return_value=True)
        browser.new_context = AsyncMock(return_value=context)
        browser.new_page = AsyncMock(return_value=page)  # mode "owns" ใช้ตรงนี้แทน
        return browser

    async def release_one(self, browser) -> None:
        self._available += 1


@pytest.fixture
def _isolated_chroma(tmp_path, monkeypatch):
    """W20: routes.py::generate_plan/execute_plan เรียก core/plan_memory.py ตรงๆ (ไม่ผ่าน
    Orchestrator ที่ถูก mock ไว้ในเทสต์ส่วนใหญ่) ซึ่งแตะ ChromaDB จริงถ้าไม่ isolate —
    เจอบั๊กจริงระหว่างพัฒนา: เทสต์ที่ไม่ได้ mock plan_memory (เช่น
    test_execute_plan_threads_approved_plan_into_run_task) เขียนแผนจริงลง
    ./data/chroma (persist dir จริงของ production) แล้วเทสต์อื่นที่ใช้ domain/goal
    ซ้ำกันดันอ่านค่าที่หลุดมาจากเทสต์ก่อนหน้าแทนที่จะเรียก mock Orchestrator ตามที่ตั้งใจ —
    ต้อง monkeypatch ทั้ง settings.chroma_persist_dir (ให้ path ใหม่ตอน get_client() ถูก
    เรียกครั้งถัดไป) และ reset chroma_client._client ที่ cache client ไว้เป็น singleton
    ต่อ process (ดู comment ใน chroma_client.py) ไม่งั้น client เก่าที่ชี้ไป path จริงจะยัง
    ถูกใช้ซ้ำอยู่ดีแม้ setting จะเปลี่ยนไปแล้วก็ตาม"""
    from backend.app.rag import chroma_client

    monkeypatch.setattr(settings, "chroma_persist_dir", str(tmp_path / "chroma"))
    monkeypatch.setattr(chroma_client, "_client", None)
    yield


@pytest.fixture
def client(_isolated_chroma):
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


def test_create_task_with_use_user_browser_bypasses_pool_and_connects_via_cdp(client):
    """W12: use_user_browser=True = agent ต่อเข้า Chrome จริงของ user ผ่าน CDP (ดู
    core/user_browser.py) — ต้อง bypass pool เหมือน headless=False (browser เป็นของ
    user เอง ไม่ใช่ของ pool ให้ยืม) และส่ง connect_to_user_browser=True เข้า run_task()
    ไม่ใช่ browser=/keep_browser_open= (คนละกลไกกับ headless=False path)"""
    with patch("backend.app.api.routes.Orchestrator") as MockOrchestrator:
        mock_run_task = AsyncMock(return_value=_FAKE_RESULT)
        MockOrchestrator.return_value.run_task = mock_run_task

        resp = client.post(
            "/tasks", json={"url": "https://example.com", "goal": "ทดสอบ", "use_user_browser": True},
        )
        _poll_until(client, resp.json()["task_id"])

        # pool ต้องไม่ถูกแตะเลย เหมือน headless=False path
        assert client.get("/pool/status").json() == {"size": 2, "available": 2, "in_use": 0}

    kwargs = mock_run_task.await_args.kwargs
    assert kwargs.get("browser") is None
    assert kwargs["connect_to_user_browser"] is True
    assert "keep_browser_open" not in kwargs


def test_create_task_use_user_browser_ignores_headless_flag(client):
    """headless=False ส่งมาพร้อม use_user_browser=True — use_user_browser ต้องชนะ (ไม่
    เข้า path ของ wants_visible_browser ที่ launch Chromium แยกต่างหาก) เพราะ headless
    ไม่มีความหมายเลยตอนต่อเข้า browser จริงที่เปิดอยู่แล้ว"""
    with patch("backend.app.api.routes.Orchestrator") as MockOrchestrator:
        mock_run_task = AsyncMock(return_value=_FAKE_RESULT)
        MockOrchestrator.return_value.run_task = mock_run_task

        resp = client.post(
            "/tasks",
            json={"url": "https://example.com", "goal": "ทดสอบ", "headless": False, "use_user_browser": True},
        )
        _poll_until(client, resp.json()["task_id"])

    kwargs = mock_run_task.await_args.kwargs
    assert kwargs["connect_to_user_browser"] is True
    assert "keep_browser_open" not in kwargs


def test_create_task_passes_tab_reuse_policy_to_run_task(client):
    """tab_reuse_policy ส่งมาเอง (เช่น "always_reuse" ให้ follow-up command ในบทสนทนา
    เดียวกันต่อ tab เดิมได้เลยไม่มี prompt คั่นทุกเทิร์น) ต้องไหลเข้า run_task() ตรงๆ"""
    with patch("backend.app.api.routes.Orchestrator") as MockOrchestrator:
        mock_run_task = AsyncMock(return_value=_FAKE_RESULT)
        MockOrchestrator.return_value.run_task = mock_run_task

        resp = client.post(
            "/tasks",
            json={
                "url": "https://example.com", "goal": "ทดสอบ",
                "use_user_browser": True, "tab_reuse_policy": "always_reuse",
            },
        )
        _poll_until(client, resp.json()["task_id"])

    assert mock_run_task.await_args.kwargs["tab_reuse_policy"] == "always_reuse"


# W12: session_id — ต่างจาก tab_reuse_policy/use_user_browser ด้านบนที่คุมแค่ 1 task
# เดียว session_id ผูก "page เดิม" ข้ามหลาย POST /tasks (ดู core/session_registry.py) —
# เทสต์พวกนี้ยิงจริงผ่าน TestClient 2 ครั้งติดกัน (ไม่ mock session_registry เอง) เพื่อ
# ยืนยัน integration เต็มสาย routes.py -> session_registry.py -> orchestrator.run_task()


def test_create_task_with_session_id_reuses_same_page_across_calls(client):
    """2 POST /tasks ติดกันด้วย session_id เดียวกัน — ต้องได้ page object เดิมกลับมาทั้งคู่
    (ไม่ acquire จาก pool ซ้ำรอบสอง) ผ่าน run_task(page=...) ไม่ใช่ browser="""
    with patch("backend.app.api.routes.Orchestrator") as MockOrchestrator:
        mock_run_task = AsyncMock(return_value=_FAKE_RESULT)
        MockOrchestrator.return_value.run_task = mock_run_task

        resp1 = client.post(
            "/tasks", json={"url": "https://example.com", "goal": "เปิดเว็บ", "session_id": "sess-1"},
        )
        _poll_until(client, resp1.json()["task_id"])
        pool_after_first = client.get("/pool/status").json()

        resp2 = client.post(
            "/tasks", json={"url": "https://example.com", "goal": "sign in", "session_id": "sess-1"},
        )
        _poll_until(client, resp2.json()["task_id"])
        pool_after_second = client.get("/pool/status").json()

    assert len(mock_run_task.await_args_list) == 2
    first_kwargs = mock_run_task.await_args_list[0].kwargs
    second_kwargs = mock_run_task.await_args_list[1].kwargs
    assert "browser" not in first_kwargs
    assert first_kwargs["page"] is second_kwargs["page"]
    # pool เสียแค่ 1 ตัวให้ session นี้ ไม่ใช่ 2 ตัว (ไม่ acquire ซ้ำรอบสอง)
    assert pool_after_first == pool_after_second == {"size": 2, "available": 1, "in_use": 1}


def test_close_session_then_reusing_id_creates_new_page(client):
    with patch("backend.app.api.routes.Orchestrator") as MockOrchestrator:
        mock_run_task = AsyncMock(return_value=_FAKE_RESULT)
        MockOrchestrator.return_value.run_task = mock_run_task

        resp1 = client.post(
            "/tasks", json={"url": "https://example.com", "goal": "เปิดเว็บ", "session_id": "sess-close"},
        )
        _poll_until(client, resp1.json()["task_id"])

        close_resp = client.post("/sessions/sess-close/close")
        assert close_resp.status_code == 200
        assert client.get("/pool/status").json()["available"] == 2  # คืน browser กลับ pool แล้ว

        resp2 = client.post(
            "/tasks", json={"url": "https://example.com", "goal": "เปิดใหม่", "session_id": "sess-close"},
        )
        _poll_until(client, resp2.json()["task_id"])

    first_kwargs = mock_run_task.await_args_list[0].kwargs
    second_kwargs = mock_run_task.await_args_list[1].kwargs
    assert first_kwargs["page"] is not second_kwargs["page"]  # session ใหม่ (id เดิม) = page ใหม่


def test_close_unknown_session_returns_404(client):
    resp = client.post("/sessions/does-not-exist/close")
    assert resp.status_code == 404


def test_list_sessions_reflects_open_sessions(client):
    assert client.get("/sessions").json() == []

    with patch("backend.app.api.routes.Orchestrator") as MockOrchestrator:
        mock_run_task = AsyncMock(return_value=_FAKE_RESULT)
        MockOrchestrator.return_value.run_task = mock_run_task

        resp = client.post(
            "/tasks", json={"url": "https://example.com", "goal": "เปิดเว็บ", "session_id": "sess-list"},
        )
        _poll_until(client, resp.json()["task_id"])

    sessions = client.get("/sessions").json()
    assert len(sessions) == 1
    assert sessions[0]["session_id"] == "sess-list"
    assert sessions[0]["mode"] == "pool"


# W13: /api/generate_plan + /api/execute_plan — เฟสวางแผนแยกต่างหากจาก POST /tasks


def test_generate_plan_without_session_id_never_touches_browser(client):
    with patch("backend.app.api.routes.Orchestrator") as MockOrchestrator:
        mock_generate_plan = AsyncMock(return_value="1. Do X")
        MockOrchestrator.return_value.generate_plan = mock_generate_plan

        resp = client.post("/api/generate_plan", json={"url": "https://example.com", "goal": "ทดสอบ"})

    assert resp.status_code == 200
    assert resp.json() == {"plan": "1. Do X"}
    mock_generate_plan.assert_awaited_once_with(
        "https://example.com", "ทดสอบ", provider=None, page=None, site_manual_context="",
    )
    # ไม่มีทาง touch pool/session registry เลยจาก endpoint นี้
    assert client.get("/pool/status").json() == {"size": 2, "available": 2, "in_use": 0}
    assert client.get("/sessions").json() == []


def test_generate_plan_with_unknown_session_id_passes_none_page(client):
    with patch("backend.app.api.routes.Orchestrator") as MockOrchestrator:
        mock_generate_plan = AsyncMock(return_value="1. Do X")
        MockOrchestrator.return_value.generate_plan = mock_generate_plan

        resp = client.post(
            "/api/generate_plan",
            json={"url": "https://example.com", "goal": "ทดสอบ", "session_id": "does-not-exist-yet"},
        )

    assert resp.status_code == 200
    mock_generate_plan.assert_awaited_once_with(
        "https://example.com", "ทดสอบ", provider=None, page=None, site_manual_context="",
    )


def test_generate_plan_with_existing_session_perceives_that_page(client):
    """session_id ที่มี page เปิดค้างอยู่แล้วจริง (สร้างผ่าน execute_plan มาก่อน) —
    generate_plan ต้องส่ง page ตัวนั้นเข้า Orchestrator.generate_plan() ไม่ใช่ None"""
    with patch("backend.app.api.routes.Orchestrator") as MockOrchestrator:
        mock_run_task = AsyncMock(return_value=_FAKE_RESULT)
        mock_generate_plan = AsyncMock(return_value="1. Sign in")
        MockOrchestrator.return_value.run_task = mock_run_task
        MockOrchestrator.return_value.generate_plan = mock_generate_plan

        resp1 = client.post(
            "/api/execute_plan",
            json={"url": "https://example.com", "goal": "เปิดเว็บ", "session_id": "sess-plan"},
        )
        _poll_until(client, resp1.json()["task_id"])

        resp2 = client.post(
            "/api/generate_plan",
            json={"url": "https://example.com", "goal": "sign in", "session_id": "sess-plan"},
        )

    assert resp2.status_code == 200
    assert resp2.json() == {"plan": "1. Sign in"}
    page_arg = mock_generate_plan.await_args.kwargs["page"]
    assert page_arg is not None


def test_execute_plan_threads_approved_plan_into_run_task(client):
    with patch("backend.app.api.routes.Orchestrator") as MockOrchestrator:
        mock_run_task = AsyncMock(return_value=_FAKE_RESULT)
        MockOrchestrator.return_value.run_task = mock_run_task

        resp = client.post(
            "/api/execute_plan",
            json={"url": "https://example.com", "goal": "ทดสอบ", "plan": "1. Click X\n2. Click Y"},
        )
        _poll_until(client, resp.json()["task_id"])

    kwargs = mock_run_task.await_args.kwargs
    assert kwargs["approved_plan"] == "1. Click X\n2. Click Y"
    assert "confirm_plan" not in kwargs  # execute_plan ไม่มี confirm_plan gate อีกต่อไป


def test_execute_plan_without_plan_passes_none(client):
    with patch("backend.app.api.routes.Orchestrator") as MockOrchestrator:
        mock_run_task = AsyncMock(return_value=_FAKE_RESULT)
        MockOrchestrator.return_value.run_task = mock_run_task

        resp = client.post("/api/execute_plan", json={"url": "https://example.com", "goal": "ทดสอบ"})
        _poll_until(client, resp.json()["task_id"])

    assert mock_run_task.await_args.kwargs["approved_plan"] is None


def test_execute_plan_with_headless_false_bypasses_pool(client):
    with patch("backend.app.api.routes.Orchestrator") as MockOrchestrator:
        mock_run_task = AsyncMock(return_value=_FAKE_RESULT)
        MockOrchestrator.return_value.run_task = mock_run_task

        resp = client.post(
            "/api/execute_plan", json={"url": "https://example.com", "goal": "ทดสอบ", "headless": False},
        )
        _poll_until(client, resp.json()["task_id"])
        assert client.get("/pool/status").json() == {"size": 2, "available": 2, "in_use": 0}

    kwargs = mock_run_task.await_args.kwargs
    assert kwargs.get("browser") is None
    assert kwargs["headless"] is False
    assert kwargs["keep_browser_open"] is True


def test_execute_plan_reuses_session_page_across_calls(client):
    with patch("backend.app.api.routes.Orchestrator") as MockOrchestrator:
        mock_run_task = AsyncMock(return_value=_FAKE_RESULT)
        MockOrchestrator.return_value.run_task = mock_run_task

        resp1 = client.post(
            "/api/execute_plan",
            json={"url": "https://example.com", "goal": "เปิดเว็บ", "session_id": "sess-exec"},
        )
        _poll_until(client, resp1.json()["task_id"])
        resp2 = client.post(
            "/api/execute_plan",
            json={"url": "https://example.com", "goal": "sign in", "session_id": "sess-exec"},
        )
        _poll_until(client, resp2.json()["task_id"])

    first_kwargs = mock_run_task.await_args_list[0].kwargs
    second_kwargs = mock_run_task.await_args_list[1].kwargs
    assert first_kwargs["page"] is second_kwargs["page"]


# W20: Plan Memory (core/plan_memory.py) — /api/execute_plan บันทึกทุกแผนที่ user
# confirm เข้า Plan Memory เสมอ (ไม่ต้องมี flag แยกว่าแก้ไขหรือไม่) /api/generate_plan
# เช็ค Plan Memory ก่อนเรียก LLM เสมอ — เทสต์พวกนี้ mock ที่ตัว plan_memory module (ไม่
# แตะ ChromaDB จริง/ไม่โหลด embedding model จริง) เพราะสิ่งที่ต้องพิสูจน์คือ routes.py
# wiring ถูกต้อง (เรียก find_matching_plan/save_confirmed_plan ด้วย argument ที่ถูกต้อง
# ตรงจังหวะที่ถูกต้อง) ส่วน logic การจับคู่/versioning จริงมีเทสต์ของตัวเองแล้วใน
# test_plan_memory.py


def test_execute_plan_always_saves_confirmed_plan_to_plan_memory(client):
    """ทุกครั้งที่ user กด Approve (ไม่ว่าจะแก้ไขข้อความแผนมาก่อนหรือไม่) ต้องบันทึกเข้า
    Plan Memory เสมอ — ไม่มี flag แยกแบบ plan_edited ของระบบเดิม (W19) อีกต่อไป"""
    with patch("backend.app.api.routes.Orchestrator") as MockOrchestrator, \
         patch("backend.app.api.routes.plan_memory.save_confirmed_plan") as mock_save:
        mock_run_task = AsyncMock(return_value=_FAKE_RESULT)
        MockOrchestrator.return_value.run_task = mock_run_task

        resp = client.post(
            "/api/execute_plan",
            json={"url": "https://www.saucedemo.com/", "goal": "login", "plan": "1. Open site\n2. Log in"},
        )
        _poll_until(client, resp.json()["task_id"])

    mock_save.assert_called_once_with("www.saucedemo.com", "login", "1. Open site\n2. Log in")


def test_execute_plan_without_a_plan_does_not_touch_plan_memory(client):
    with patch("backend.app.api.routes.Orchestrator") as MockOrchestrator, \
         patch("backend.app.api.routes.plan_memory.save_confirmed_plan") as mock_save:
        mock_run_task = AsyncMock(return_value=_FAKE_RESULT)
        MockOrchestrator.return_value.run_task = mock_run_task

        resp = client.post("/api/execute_plan", json={"url": "https://example.com", "goal": "ทดสอบ"})
        _poll_until(client, resp.json()["task_id"])

    mock_save.assert_not_called()


def test_generate_plan_returns_matched_plan_memory_result_without_calling_llm(client):
    """Plan Priority: เจอ approved plan ที่ตรงพอใน Plan Memory ต้องคืนตรงๆ ข้าม
    Orchestrator.generate_plan() (เรียก LLM) ไปเลย"""
    with patch("backend.app.api.routes.Orchestrator") as MockOrchestrator, \
         patch(
             "backend.app.api.routes.plan_memory.find_matching_plan",
             return_value={"intent_key": "k1", "version": 2, "plan": "1. Reused step", "distance": 0.1},
         ):
        mock_generate_plan = AsyncMock(return_value="1. Should never be used")
        MockOrchestrator.return_value.generate_plan = mock_generate_plan

        resp = client.post(
            "/api/generate_plan", json={"url": "https://www.saucedemo.com/", "goal": "sign in"},
        )

    assert resp.status_code == 200
    assert resp.json() == {"plan": "1. Reused step"}
    mock_generate_plan.assert_not_awaited()


def test_generate_plan_falls_back_to_llm_when_no_plan_memory_match(client):
    with patch("backend.app.api.routes.Orchestrator") as MockOrchestrator, \
         patch("backend.app.api.routes.plan_memory.find_matching_plan", return_value=None):
        mock_generate_plan = AsyncMock(return_value="1. Fresh LLM draft")
        MockOrchestrator.return_value.generate_plan = mock_generate_plan

        resp = client.post(
            "/api/generate_plan", json={"url": "https://example.com", "goal": "ทดสอบ"},
        )

    assert resp.status_code == 200
    assert resp.json() == {"plan": "1. Fresh LLM draft"}
    mock_generate_plan.assert_awaited_once()


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


# W14: Website Learning & Manual Generation (/api/site-manual/*) — ระบบแยกต่างหากจาก
# RAG/ChromaDB เก็บ manual เป็น JSON บนดิสก์ (ดู backend/app/site_learning/)


@pytest.fixture
def isolated_manuals_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "site_manuals_dir", str(tmp_path))
    yield tmp_path


def test_site_manual_status_returns_false_when_no_manual(client, isolated_manuals_dir):
    resp = client.get("/api/site-manual/status", params={"url": "https://example.com/"})
    assert resp.status_code == 200
    assert resp.json() == {"exists": False, "version": None}


def test_site_manual_status_returns_true_with_version_when_manual_exists(client, isolated_manuals_dir):
    from backend.app.site_learning import storage
    from backend.app.site_learning.schema import PageInfo, SiteManual

    storage.save_manual(SiteManual(website="example.com", pages=[PageInfo(name="Home", url="/")]))

    resp = client.get("/api/site-manual/status", params={"url": "https://example.com/dashboard"})
    assert resp.status_code == 200
    assert resp.json() == {"exists": True, "version": 1}


def test_learn_site_returns_202_and_saves_manual(client, isolated_manuals_dir):
    from backend.app.site_learning.schema import ButtonInfo, PageInfo, SiteManual

    fake_manual = SiteManual(website="example.com", pages=[
        PageInfo(name="Home", url="https://example.com/", buttons=[ButtonInfo(text="Go")]),
    ])

    # W16: learn_site() เปิด browser ของตัวเองแบบมองเห็นได้ (headless=False) แยกจาก
    # BrowserPool ที่ fake ไว้แล้ว (_FakeBrowserPool) — ต้อง fake async_playwright() ตรงนี้
    # ด้วย ไม่งั้นเทสต์นี้จะเปิด Chromium จริงแบบมีหน้าต่างขึ้นมาจริงๆ ทุกครั้งที่รัน pytest
    # (ผิดหลักการของไฟล์นี้ทั้งไฟล์ — ดู docstring หัวไฟล์ "ห้ามให้ lifespan เปิด browser จริง")
    fake_browser = AsyncMock()
    fake_playwright_obj = AsyncMock()
    fake_playwright_obj.chromium.launch = AsyncMock(return_value=fake_browser)
    fake_playwright_cm = AsyncMock()
    fake_playwright_cm.__aenter__ = AsyncMock(return_value=fake_playwright_obj)
    fake_playwright_cm.__aexit__ = AsyncMock(return_value=False)

    with patch("backend.app.api.routes.async_playwright", return_value=fake_playwright_cm), \
         patch("backend.app.api.routes.crawl_site", AsyncMock(return_value=fake_manual)) as mock_crawl:
        resp = client.post("/api/site-manual/learn", json={"url": "https://example.com/"})
        assert resp.status_code == 202
        learn_id = resp.json()["learn_id"]
        assert resp.json()["status"] == "running"

        # ต่อ SSE หลัง crawl จบไปแล้ว (mock คืนผลทันที) — ต้องได้ learn_done ทันทีไม่ hang
        # (เหมือน pattern เดียวกับ test_stream_finished_task_replays_synthesized_done_event)
        import time
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            stream_resp = client.get(f"/api/site-manual/learn/{learn_id}/stream")
            if "learn_done" in stream_resp.text:
                break
            time.sleep(0.02)

    assert "learn_done" in stream_resp.text
    mock_crawl.assert_awaited_once()

    from backend.app.site_learning import storage
    assert storage.manual_exists("example.com") is True
    saved = storage.load_manual("example.com")
    assert saved.pages[0].name == "Home"


def test_learn_site_stream_unknown_learn_id_returns_404(client):
    resp = client.get("/api/site-manual/learn/does-not-exist/stream")
    assert resp.status_code == 404


def test_relearn_page_returns_404_when_no_manual_exists_yet(client, isolated_manuals_dir):
    resp = client.post(
        "/api/site-manual/example.com/relearn-page", json={"url": "https://example.com/dashboard"},
    )
    assert resp.status_code == 404


def test_relearn_page_updates_existing_manual_and_bumps_version(client, isolated_manuals_dir):
    from backend.app.site_learning import storage
    from backend.app.site_learning.schema import PageInfo, SiteManual

    storage.save_manual(SiteManual(website="example.com", pages=[
        PageInfo(name="Dashboard", url="https://example.com/dashboard", description="old"),
    ]))

    updated_page_info = PageInfo(name="Dashboard", url="https://example.com/dashboard", description="old")
    with patch("backend.app.api.routes.extract_page", AsyncMock(return_value=(updated_page_info, []))), \
         patch("backend.app.api.routes.describe_page", AsyncMock(return_value=("Dashboard", "refreshed"))):
        resp = client.post(
            "/api/site-manual/example.com/relearn-page", json={"url": "https://example.com/dashboard"},
        )

    assert resp.status_code == 200
    assert resp.json() == {"version": 2}
    saved = storage.load_manual("example.com")
    assert saved.version == 2
    assert saved.pages[0].description == "refreshed"


# ---------------- W17: เก็บ username/password แยกไฟล์จาก manual ----------------


def _fake_learn_playwright():
    """เหมือนใน test_learn_site_returns_202_and_saves_manual — fake async_playwright()
    กัน pytest เปิด Chromium จริงแบบมองเห็นได้ระหว่างรัน POST /api/site-manual/learn"""
    fake_browser = AsyncMock()
    fake_playwright_obj = AsyncMock()
    fake_playwright_obj.chromium.launch = AsyncMock(return_value=fake_browser)
    fake_playwright_cm = AsyncMock()
    fake_playwright_cm.__aenter__ = AsyncMock(return_value=fake_playwright_obj)
    fake_playwright_cm.__aexit__ = AsyncMock(return_value=False)
    return fake_playwright_cm


def test_learn_site_persists_credentials_when_username_and_password_given(client, isolated_manuals_dir):
    from backend.app.site_learning.schema import PageInfo, SiteManual

    fake_manual = SiteManual(website="example.com", pages=[PageInfo(name="Login", url="https://example.com/")])

    with patch("backend.app.api.routes.async_playwright", return_value=_fake_learn_playwright()), \
         patch("backend.app.api.routes.crawl_site", AsyncMock(return_value=fake_manual)):
        resp = client.post(
            "/api/site-manual/learn",
            json={"url": "https://example.com/", "username": "alice", "password": "s3cr3t"},
        )
        assert resp.status_code == 202
        learn_id = resp.json()["learn_id"]

        import time
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            stream_resp = client.get(f"/api/site-manual/learn/{learn_id}/stream")
            if "learn_done" in stream_resp.text:
                break
            time.sleep(0.02)

    from backend.app.site_learning import storage
    assert storage.credentials_exist("example.com") is True
    assert storage.load_credentials("example.com") == {"username": "alice", "password": "s3cr3t"}


def test_learn_site_does_not_persist_credentials_when_not_given(client, isolated_manuals_dir):
    from backend.app.site_learning.schema import PageInfo, SiteManual

    fake_manual = SiteManual(website="example.com", pages=[PageInfo(name="Home", url="https://example.com/")])

    with patch("backend.app.api.routes.async_playwright", return_value=_fake_learn_playwright()), \
         patch("backend.app.api.routes.crawl_site", AsyncMock(return_value=fake_manual)):
        resp = client.post("/api/site-manual/learn", json={"url": "https://example.com/"})
        learn_id = resp.json()["learn_id"]

        import time
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            stream_resp = client.get(f"/api/site-manual/learn/{learn_id}/stream")
            if "learn_done" in stream_resp.text:
                break
            time.sleep(0.02)

    from backend.app.site_learning import storage
    assert storage.credentials_exist("example.com") is False


# ---------------- W18: เลือกใช้บัญชีที่บันทึกไว้แทนการกรอกใหม่ ----------------


def test_learn_site_uses_saved_credentials_when_flag_set_and_none_given(client, isolated_manuals_dir):
    """ผู้ใช้เลือก "ใช้บัญชีที่บันทึกไว้" บน UI — ไม่ส่ง username/password มาเลย แค่
    use_saved_credentials: true — backend ต้องดึง credential ที่เก็บไว้แล้วมาป้อนให้
    crawl_site() เอง โดยไม่ต้องให้ frontend ส่งรหัสผ่านจริงกลับมา"""
    from backend.app.site_learning import storage
    from backend.app.site_learning.schema import PageInfo, SiteManual

    storage.save_credentials("example.com", "alice", "s3cr3t")
    fake_manual = SiteManual(website="example.com", pages=[PageInfo(name="Home", url="https://example.com/")])

    with patch("backend.app.api.routes.async_playwright", return_value=_fake_learn_playwright()), \
         patch("backend.app.api.routes.crawl_site", AsyncMock(return_value=fake_manual)) as mock_crawl:
        resp = client.post(
            "/api/site-manual/learn",
            json={"url": "https://example.com/", "use_saved_credentials": True},
        )
        learn_id = resp.json()["learn_id"]

        import time
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            stream_resp = client.get(f"/api/site-manual/learn/{learn_id}/stream")
            if "learn_done" in stream_resp.text:
                break
            time.sleep(0.02)

    mock_crawl.assert_awaited_once()
    call_kwargs = mock_crawl.await_args.kwargs
    assert call_kwargs["username"] == "alice"
    assert call_kwargs["password"] == "s3cr3t"


def test_learn_site_ignores_use_saved_credentials_flag_when_none_stored(client, isolated_manuals_dir):
    from backend.app.site_learning.schema import PageInfo, SiteManual

    fake_manual = SiteManual(website="example.com", pages=[PageInfo(name="Home", url="https://example.com/")])

    with patch("backend.app.api.routes.async_playwright", return_value=_fake_learn_playwright()), \
         patch("backend.app.api.routes.crawl_site", AsyncMock(return_value=fake_manual)) as mock_crawl:
        resp = client.post(
            "/api/site-manual/learn",
            json={"url": "https://example.com/", "use_saved_credentials": True},
        )
        learn_id = resp.json()["learn_id"]

        import time
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            stream_resp = client.get(f"/api/site-manual/learn/{learn_id}/stream")
            if "learn_done" in stream_resp.text:
                break
            time.sleep(0.02)

    call_kwargs = mock_crawl.await_args.kwargs
    assert call_kwargs["username"] is None
    assert call_kwargs["password"] is None


def test_learn_site_explicit_credentials_take_priority_over_saved_ones(client, isolated_manuals_dir):
    """ผู้ใช้เลือก "เข้าสู่ระบบด้วยบัญชีอื่น" แล้วกรอกใหม่ — ต้องใช้ค่าที่กรอกใหม่ ไม่ใช่
    ค่าที่เคยบันทึกไว้ก่อนหน้า แม้จะส่ง use_saved_credentials มาด้วยเผื่อไว้ก็ตาม"""
    from backend.app.site_learning import storage
    from backend.app.site_learning.schema import PageInfo, SiteManual

    storage.save_credentials("example.com", "old-user", "old-pass")
    fake_manual = SiteManual(website="example.com", pages=[PageInfo(name="Home", url="https://example.com/")])

    with patch("backend.app.api.routes.async_playwright", return_value=_fake_learn_playwright()), \
         patch("backend.app.api.routes.crawl_site", AsyncMock(return_value=fake_manual)) as mock_crawl:
        resp = client.post(
            "/api/site-manual/learn",
            json={
                "url": "https://example.com/", "use_saved_credentials": True,
                "username": "new-user", "password": "new-pass",
            },
        )
        learn_id = resp.json()["learn_id"]

        import time
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            stream_resp = client.get(f"/api/site-manual/learn/{learn_id}/stream")
            if "learn_done" in stream_resp.text:
                break
            time.sleep(0.02)

    call_kwargs = mock_crawl.await_args.kwargs
    assert call_kwargs["username"] == "new-user"
    assert call_kwargs["password"] == "new-pass"
    assert storage.load_credentials("example.com") == {"username": "new-user", "password": "new-pass"}


def test_save_site_credentials_endpoint_persists_and_status_reflects_it(client, isolated_manuals_dir):
    from backend.app.site_learning import storage

    assert client.get("/api/site-manual/example.com/credentials/status").json() == {"exists": False}

    resp = client.post(
        "/api/site-manual/example.com/credentials", json={"username": "alice", "password": "s3cr3t"},
    )
    assert resp.status_code == 204

    assert client.get("/api/site-manual/example.com/credentials/status").json() == {"exists": True}
    assert storage.load_credentials("example.com") == {"username": "alice", "password": "s3cr3t"}


def test_save_site_credentials_endpoint_never_echoes_password_back(client, isolated_manuals_dir):
    """response ของทั้ง POST (204) และ GET status ต้องไม่มี password หลุดออกมาเลย —
    เช็คทั้ง body ดิบๆ ไม่ใช่แค่ parse JSON เพราะอยากมั่นใจว่าไม่มีที่ไหนเผลอ echo กลับ"""
    resp = client.post(
        "/api/site-manual/example.com/credentials", json={"username": "alice", "password": "s3cr3t-marker"},
    )
    assert "s3cr3t-marker" not in resp.text

    status_resp = client.get("/api/site-manual/example.com/credentials/status")
    assert "s3cr3t-marker" not in status_resp.text
    assert "alice" not in status_resp.text


def test_delete_site_credentials_endpoint_removes_them(client, isolated_manuals_dir):
    from backend.app.site_learning import storage

    storage.save_credentials("example.com", "alice", "s3cr3t")
    resp = client.delete("/api/site-manual/example.com/credentials")
    assert resp.status_code == 204
    assert storage.credentials_exist("example.com") is False


def test_delete_site_credentials_endpoint_is_idempotent_when_none_exist(client, isolated_manuals_dir):
    resp = client.delete("/api/site-manual/does-not-exist.com/credentials")
    assert resp.status_code == 204

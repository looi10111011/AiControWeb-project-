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
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.app.config import settings
from backend.app.core.orchestrator import Orchestrator
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
    # Security 1.5: TestClient ยิงทุก request ด้วย "testclient" identity เดียวกันหมด (ไม่มี
    # IP จริงให้ get_remote_address แยก) — ถ้าไม่ reset limiter ระหว่างเทสต์ นับรวมข้ามเทสต์
    # ทั้งไฟล์จนชน rate limit ของ POST /tasks/POST /api/site-manual/learn (10-5 ต่อนาที)
    # กลางทาง ทำให้เทสต์ทีหลังได้ 429 ทั้งที่ไม่เกี่ยวกับสิ่งที่กำลังเทสต์เลย (ดู
    # routes.py::limiter) — reset ก่อนทุกเทสต์ให้เริ่มนับใหม่เสมอ เหมือนเป็นคนละ client จริง
    from backend.app.api.routes import limiter as api_limiter

    api_limiter.reset()
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


# --- W19-6 ("Master Controller" MODULE 1, "General QA / No-Browser Trigger") ---


def test_create_task_general_chat_goal_never_touches_browser_or_orchestrator(client):
    """goal ที่เป็นคำถามทั่วไป/ทักทาย (เช่น "สวัสดีครับ") ต้องไม่แตะ Orchestrator/pool/
    session เลยแม้แต่นิดเดียว — ตอบผ่าน llm.chat_response() ตรงๆ"""
    with patch(
        "backend.app.api.routes.llm.chat_response", new_callable=AsyncMock,
    ) as mock_chat_response, patch("backend.app.api.routes.Orchestrator") as MockOrchestrator:
        MockOrchestrator._llm_backend.return_value = (MagicMock(), "model-x", None, None, None)
        mock_chat_response.return_value = "สวัสดีครับ มีอะไรให้ช่วยไหมครับ"

        resp = client.post("/tasks", json={"url": "https://example.com", "goal": "สวัสดีครับ"})
        assert resp.status_code == 202
        task_id = resp.json()["task_id"]

        final = _poll_until(client, task_id)

    assert final["status"] == "done"
    assert final["result"]["message"] == "สวัสดีครับ มีอะไรให้ช่วยไหมครับ"
    assert final["result"]["steps"] == 0
    assert final["result"]["success"] is True
    MockOrchestrator.return_value.run_task.assert_not_called()
    mock_chat_response.assert_awaited_once()
    # ไม่มีทาง touch pool/session registry เลยจาก general-chat path นี้
    assert client.get("/pool/status").json() == {"size": 2, "available": 2, "in_use": 0}
    assert client.get("/sessions").json() == []


# --- W20 (MODULE 0): "/context" special command interceptor ---


def test_create_task_context_command_never_touches_browser_or_orchestrator(client):
    """goal ที่มีคำสั่ง "/context" ปน -> ต้องไม่แตะ Orchestrator/pool/session เลย ตอบผ่าน
    llm.context_inspection_reply() ตรงๆ (ไม่ใช่ chat_response/answer_file_query)"""
    with patch(
        "backend.app.api.routes.llm.context_inspection_reply", new_callable=AsyncMock,
    ) as mock_reply, patch("backend.app.api.routes.Orchestrator") as MockOrchestrator:
        MockOrchestrator._llm_backend.return_value = (MagicMock(), "model-x", None, None, None)
        mock_reply.return_value = "🎯 [ความเข้าใจของ Agent ต่อคำสั่งนี้]\n- Goal: ..."

        resp = client.post("/tasks", json={
            "url": "https://example.com", "goal": "/context เล่นเพลงที่ 3",
        })
        assert resp.status_code == 202
        task_id = resp.json()["task_id"]

        final = _poll_until(client, task_id)

    assert final["status"] == "done"
    assert "ความเข้าใจของ Agent" in final["result"]["message"]
    assert final["result"]["steps"] == 0
    assert final["result"]["success"] is True
    MockOrchestrator.return_value.run_task.assert_not_called()
    mock_reply.assert_awaited_once()
    # คำสั่ง "/context" เองต้องถูกตัดออกก่อนส่งเข้า LLM วิเคราะห์ ไม่ปนกับคำสั่งจริง
    assert mock_reply.await_args.args[2] == "เล่นเพลงที่ 3"
    assert client.get("/pool/status").json() == {"size": 2, "available": 2, "in_use": 0}
    assert client.get("/sessions").json() == []


def test_create_task_context_command_takes_priority_over_attached_file(client):
    """"/context" ต้องเช็คก่อนแม้แต่ attached_file — ไม่ไป parse ไฟล์/เรียก
    answer_file_query() เลยตอนอยู่ใน /context mode"""
    with (
        patch("backend.app.api.routes.load_manual_bytes") as mock_load_bytes,
        patch(
            "backend.app.api.routes.llm.answer_file_query", new_callable=AsyncMock,
        ) as mock_answer_file_query,
        patch(
            "backend.app.api.routes.llm.context_inspection_reply", new_callable=AsyncMock,
        ) as mock_reply,
        patch("backend.app.api.routes.Orchestrator") as MockOrchestrator,
    ):
        MockOrchestrator._llm_backend.return_value = (MagicMock(), "model-x", None, None, None)
        mock_reply.return_value = "🎯 [ความเข้าใจของ Agent ต่อคำสั่งนี้]\n- Goal: ..."

        resp = client.post("/tasks", json={
            "url": "https://example.com", "goal": "/context อ่านตารางหน้าแรกให้หน่อย",
            "attached_file_name": "report.xlsx", "attached_file_content_base64": _b64(b"x"),
        })
        _poll_until(client, resp.json()["task_id"])

    mock_load_bytes.assert_not_called()
    mock_answer_file_query.assert_not_called()
    mock_reply.assert_awaited_once()


def test_generate_plan_context_command_short_circuits_to_qa_without_calling_llm(client):
    """"/context" -> /api/generate_plan ต้องคืน is_qa=True ทันที ไม่เรียก
    Orchestrator.generate_plan() เลย (ให้ frontend ข้าม plan-approval ไปตอบ inspection ตรงๆ)"""
    with patch("backend.app.api.routes.Orchestrator") as MockOrchestrator:
        mock_generate_plan = AsyncMock(return_value="should not be called")
        MockOrchestrator.return_value.generate_plan = mock_generate_plan

        resp = client.post("/api/generate_plan", json={
            "url": "https://example.com", "goal": "/context ทำอะไรอยู่",
        })

    assert resp.status_code == 200
    assert resp.json()["is_qa"] is True
    assert resp.json()["plan"] == ""
    mock_generate_plan.assert_not_called()


# --- pdf/xlsx: attached-file query (never touches browser, priority over general-chat) ---


def _b64(raw: bytes) -> str:
    import base64
    return base64.b64encode(raw).decode("ascii")


def test_create_task_attached_file_never_touches_browser_or_orchestrator(client):
    """แนบไฟล์มา -> ต้องตอบจากเนื้อหาไฟล์ตรงๆ ไม่แตะ Orchestrator/pool/session เลย
    (เหมือน general-chat path แต่ผ่าน load_manual_bytes/answer_file_query แทน)"""
    with (
        patch("backend.app.api.routes.load_manual_bytes") as mock_load_bytes,
        patch(
            "backend.app.api.routes.llm.answer_file_query", new_callable=AsyncMock,
        ) as mock_answer_file_query,
        patch("backend.app.api.routes.Orchestrator") as MockOrchestrator,
    ):
        MockOrchestrator._llm_backend.return_value = (MagicMock(), "model-x", None, None, None)
        mock_load_bytes.return_value = "Invoice total: 1,250 THB"
        mock_answer_file_query.return_value = "ยอดรวม 1,250 บาทครับ"

        resp = client.post("/tasks", json={
            "url": "https://example.com", "goal": "ยอดรวมเท่าไหร่",
            "attached_file_name": "invoice.pdf", "attached_file_content_base64": _b64(b"fake pdf bytes"),
        })
        assert resp.status_code == 202
        task_id = resp.json()["task_id"]

        final = _poll_until(client, task_id)

    assert final["status"] == "done"
    assert final["result"]["message"] == "ยอดรวม 1,250 บาทครับ"
    assert final["result"]["steps"] == 0
    assert final["result"]["success"] is True
    MockOrchestrator.return_value.run_task.assert_not_called()
    mock_load_bytes.assert_called_once_with(b"fake pdf bytes", "invoice.pdf")
    mock_answer_file_query.assert_awaited_once()
    assert client.get("/pool/status").json() == {"size": 2, "available": 2, "in_use": 0}
    assert client.get("/sessions").json() == []


def test_create_task_attached_file_takes_priority_over_general_chat_check(client):
    """goal ที่จะ match is_general_chat_query() ด้วย (เช่น "สวัสดี") แต่มีไฟล์แนบมาด้วย —
    ต้องไปทาง attached-file path เสมอ ไม่ใช่ general-chat (เช็คไฟล์ก่อน keyword matching)"""
    with (
        patch("backend.app.api.routes.load_manual_bytes", return_value="doc text") as mock_load_bytes,
        patch(
            "backend.app.api.routes.llm.answer_file_query", new_callable=AsyncMock,
        ) as mock_answer_file_query,
        patch(
            "backend.app.api.routes.llm.chat_response", new_callable=AsyncMock,
        ) as mock_chat_response,
        patch("backend.app.api.routes.Orchestrator") as MockOrchestrator,
    ):
        MockOrchestrator._llm_backend.return_value = (MagicMock(), "model-x", None, None, None)
        mock_answer_file_query.return_value = "reply from file"

        resp = client.post("/tasks", json={
            "url": "https://example.com", "goal": "สวัสดีครับ",
            "attached_file_name": "note.pdf", "attached_file_content_base64": _b64(b"x"),
        })
        _poll_until(client, resp.json()["task_id"])

    mock_load_bytes.assert_called_once()
    mock_answer_file_query.assert_awaited_once()
    mock_chat_response.assert_not_called()


def test_create_task_attached_file_read_error_returns_graceful_message(client):
    """ไฟล์อ่านไม่ได้ (นามสกุลไม่รองรับ/เสีย) ต้องคืนข้อความ error ที่อ่านได้ผ่านช่องทาง
    ปกติเหมือนคำตอบสำเร็จ ไม่ใช่ 500/task status "error" ที่ frontend ไม่รู้จะแสดงอะไร"""
    with (
        patch(
            "backend.app.api.routes.load_manual_bytes",
            side_effect=ValueError("Unsupported file type: .docx"),
        ),
        patch(
            "backend.app.api.routes.llm.answer_file_query", new_callable=AsyncMock,
        ) as mock_answer_file_query,
        patch("backend.app.api.routes.Orchestrator") as MockOrchestrator,
    ):
        MockOrchestrator._llm_backend.return_value = (MagicMock(), "model-x", None, None, None)

        resp = client.post("/tasks", json={
            "url": "https://example.com", "goal": "อ่านให้หน่อย",
            "attached_file_name": "notes.docx", "attached_file_content_base64": _b64(b"x"),
        })
        final = _poll_until(client, resp.json()["task_id"])

    assert final["status"] == "done"
    assert final["result"]["success"] is True
    assert "notes.docx" in final["result"]["message"]
    mock_answer_file_query.assert_not_called()


def test_create_task_attached_image_routes_to_answer_image_query_not_answer_file_query(client):
    """ไฟล์แนบเป็นรูปภาพ (.png) -> ต้องไปทาง llm.answer_image_query() ตรงๆ (ส่ง bytes ดิบ)
    ไม่ผ่าน load_manual_bytes()/answer_file_query() เลย (ไม่มี "text" ให้ extract)"""
    with (
        patch("backend.app.api.routes.load_manual_bytes") as mock_load_bytes,
        patch(
            "backend.app.api.routes.llm.answer_file_query", new_callable=AsyncMock,
        ) as mock_answer_file_query,
        patch(
            "backend.app.api.routes.llm.answer_image_query", new_callable=AsyncMock,
        ) as mock_answer_image_query,
        patch("backend.app.api.routes.Orchestrator") as MockOrchestrator,
    ):
        MockOrchestrator._llm_backend.return_value = (MagicMock(), "model-x", None, None, None)
        mock_answer_image_query.return_value = "ในภาพเห็นใบเสร็จร้านกาแฟครับ"

        resp = client.post("/tasks", json={
            "url": "https://example.com", "goal": "ในภาพนี้มีอะไรบ้าง",
            "attached_file_name": "receipt.png", "attached_file_content_base64": _b64(b"fake png bytes"),
        })
        assert resp.status_code == 202
        final = _poll_until(client, resp.json()["task_id"])

    assert final["status"] == "done"
    assert final["result"]["message"] == "ในภาพเห็นใบเสร็จร้านกาแฟครับ"
    assert final["result"]["success"] is True
    MockOrchestrator.return_value.run_task.assert_not_called()
    mock_load_bytes.assert_not_called()
    mock_answer_file_query.assert_not_called()
    mock_answer_image_query.assert_awaited_once_with(
        ANY, "model-x", "ในภาพนี้มีอะไรบ้าง", b"fake png bytes", "receipt.png", ANY,
    )


def test_create_task_attached_file_base64_decode_error_returns_graceful_message(client):
    """base64 payload เสีย (decode ไม่ได้เลย) -> คืนข้อความ error ผ่านช่องทางปกติเหมือนกัน
    ไม่ใช่ 500 — ต้องไม่ไปถึง load_manual_bytes()/answer_image_query() เลย"""
    with (
        patch("backend.app.api.routes.load_manual_bytes") as mock_load_bytes,
        patch(
            "backend.app.api.routes.llm.answer_image_query", new_callable=AsyncMock,
        ) as mock_answer_image_query,
        patch("backend.app.api.routes.Orchestrator") as MockOrchestrator,
    ):
        MockOrchestrator._llm_backend.return_value = (MagicMock(), "model-x", None, None, None)

        resp = client.post("/tasks", json={
            "url": "https://example.com", "goal": "อ่านให้หน่อย",
            "attached_file_name": "broken.pdf", "attached_file_content_base64": "not-valid-base64!!!",
        })
        final = _poll_until(client, resp.json()["task_id"])

    assert final["status"] == "done"
    assert final["result"]["success"] is True
    assert "broken.pdf" in final["result"]["message"]
    mock_load_bytes.assert_not_called()
    mock_answer_image_query.assert_not_called()


def test_generate_plan_attached_file_short_circuits_to_qa_without_calling_llm(client):
    """แนบไฟล์มา -> /api/generate_plan ต้องคืน is_qa=True ทันที ไม่เรียก
    Orchestrator.generate_plan()/classify_intent() เลย (deterministic, ไม่มี LLM call)"""
    with patch("backend.app.api.routes.Orchestrator") as MockOrchestrator:
        mock_generate_plan = AsyncMock(return_value="should not be called")
        MockOrchestrator.return_value.generate_plan = mock_generate_plan

        resp = client.post("/api/generate_plan", json={
            "url": "https://example.com", "goal": "สรุปให้หน่อย",
            "attached_file_name": "report.xlsx", "attached_file_content_base64": _b64(b"x"),
        })

    assert resp.status_code == 200
    assert resp.json()["is_qa"] is True
    assert resp.json()["plan"] == ""
    mock_generate_plan.assert_not_called()


# --- pdf/xlsx (บั๊กจริงที่ user รายงาน): file_chat_memory follow-up ---
# เทิร์น 1 แนบไฟล์ถามสำเร็จ -> เทิร์น 2 ในเซสชันเดียวกันไม่ได้แนบไฟล์ใหม่มา (goal ล้วนๆ,
# ไม่มี url) แต่ควรตอบต่อจากไฟล์เดิมได้เลย ไม่ตกไปเปิด browser จริงด้วย url ว่างเปล่า


def test_create_task_file_chat_memory_answers_followup_without_url_or_browser(client):
    with (
        patch("backend.app.api.routes.load_manual_bytes", return_value="Day 1: 8h, Day 2: 8h") as mock_load_bytes,
        patch(
            "backend.app.api.routes.llm.answer_file_query", new_callable=AsyncMock,
        ) as mock_answer_file_query,
        patch("backend.app.api.routes.Orchestrator") as MockOrchestrator,
    ):
        MockOrchestrator._llm_backend.return_value = (MagicMock(), "model-x", None, None, None)
        mock_answer_file_query.side_effect = ["สรุปไฟล์ให้แล้วครับ", "แต่ละวันทำงาน 8 ชั่วโมงครับ"]

        turn1 = client.post("/tasks", json={
            "url": "", "goal": "อ่านไฟล์นี้หน่อย", "session_id": "sess-file-memory",
            "attached_file_name": "timesheet.xlsx", "attached_file_content_base64": _b64(b"x"),
        })
        _poll_until(client, turn1.json()["task_id"])

        turn2 = client.post("/tasks", json={
            "url": "", "goal": "แต่ละวันทำอะไรบ้าง", "session_id": "sess-file-memory",
        })
        final2 = _poll_until(client, turn2.json()["task_id"])

    assert final2["status"] == "done"
    assert final2["result"]["message"] == "แต่ละวันทำงาน 8 ชั่วโมงครับ"
    assert final2["result"]["success"] is True
    MockOrchestrator.return_value.run_task.assert_not_called()
    assert mock_answer_file_query.await_count == 2
    second_call_args = mock_answer_file_query.await_args_list[1].args
    assert second_call_args[2] == "แต่ละวันทำอะไรบ้าง"
    assert second_call_args[3] == "Day 1: 8h, Day 2: 8h"
    assert second_call_args[4] == "timesheet.xlsx"


def test_create_task_file_chat_memory_falls_through_when_goal_mentions_web_action(client):
    """เทิร์น 2 มีคำบ่งบอก web action จริงๆ ("ค้นหา") -> ต้องไม่ตอบจาก file memory เดิม
    ต้องผ่าน Orchestrator.run_task() ตามปกติ"""
    with (
        patch("backend.app.api.routes.load_manual_bytes", return_value="doc text"),
        patch("backend.app.api.routes.llm.answer_file_query", new_callable=AsyncMock, return_value="reply"),
        patch("backend.app.api.routes.Orchestrator") as MockOrchestrator,
    ):
        MockOrchestrator._llm_backend.return_value = (MagicMock(), "model-x", None, None, None)
        MockOrchestrator.return_value.run_task = AsyncMock(return_value=_FAKE_RESULT)

        turn1 = client.post("/tasks", json={
            "url": "", "goal": "อ่านไฟล์นี้หน่อย", "session_id": "sess-file-memory-2",
            "attached_file_name": "notes.txt", "attached_file_content_base64": _b64(b"x"),
        })
        _poll_until(client, turn1.json()["task_id"])

        turn2 = client.post("/tasks", json={
            "url": "https://example.com", "goal": "ค้นหาสินค้า iPhone ให้หน่อย",
            "session_id": "sess-file-memory-2",
        })
        final2 = _poll_until(client, turn2.json()["task_id"])

    assert final2["status"] == "done"
    assert final2["result"] == _FAKE_RESULT
    MockOrchestrator.return_value.run_task.assert_awaited_once()


def test_generate_plan_file_chat_memory_followup_short_circuits_to_qa(client):
    """generate_plan endpoint ต้องคืน is_qa=True ทันทีสำหรับเทิร์นต่อยอดจากไฟล์เดิมด้วย
    (ไม่ใช่แค่ execute_plan/_run_with_resolved_browser) — ไม่งั้น frontend จะโชว์ plan
    approval panel ที่ไม่จำเป็นก่อนตอบจาก memory"""
    with (
        patch("backend.app.api.routes.load_manual_bytes", return_value="doc text"),
        patch("backend.app.api.routes.llm.answer_file_query", new_callable=AsyncMock, return_value="reply"),
        patch("backend.app.api.routes.Orchestrator") as MockOrchestrator,
    ):
        MockOrchestrator._llm_backend.return_value = (MagicMock(), "model-x", None, None, None)
        mock_generate_plan = AsyncMock(return_value="should not be called")
        MockOrchestrator.return_value.generate_plan = mock_generate_plan

        turn1 = client.post("/tasks", json={
            "url": "", "goal": "อ่านไฟล์นี้หน่อย", "session_id": "sess-file-memory-3",
            "attached_file_name": "notes.txt", "attached_file_content_base64": _b64(b"x"),
        })
        _poll_until(client, turn1.json()["task_id"])

        resp = client.post("/api/generate_plan", json={
            "url": "", "goal": "แต่ละวันทำอะไรบ้าง", "session_id": "sess-file-memory-3",
        })

    assert resp.status_code == 200
    assert resp.json()["is_qa"] is True
    assert resp.json()["plan"] == ""
    mock_generate_plan.assert_not_called()


def test_close_session_clears_file_chat_memory_for_file_only_session(client):
    """session ที่เป็น file-chat ล้วนๆ (ไม่เคยแตะ session_registry/browser เลย) ต้องปิดได้
    โดยไม่ 404 (ปุ่ม "New Session" เรียก endpoint นี้เสมอ) และเคลียร์ memory จริง — เทิร์นถัดไป
    ด้วย session_id เดิมหลังปิดต้องกลับไปพฤติกรรมปกติ (ไม่มี memory ให้ตอบจากอีกต่อไป)"""
    with (
        patch("backend.app.api.routes.load_manual_bytes", return_value="doc text"),
        patch("backend.app.api.routes.llm.answer_file_query", new_callable=AsyncMock, return_value="reply"),
        patch("backend.app.api.routes.Orchestrator") as MockOrchestrator,
    ):
        MockOrchestrator._llm_backend.return_value = (MagicMock(), "model-x", None, None, None)
        MockOrchestrator.return_value.run_task = AsyncMock(return_value=_FAKE_RESULT)

        turn1 = client.post("/tasks", json={
            "url": "", "goal": "อ่านไฟล์นี้หน่อย", "session_id": "sess-file-memory-4",
            "attached_file_name": "notes.txt", "attached_file_content_base64": _b64(b"x"),
        })
        _poll_until(client, turn1.json()["task_id"])

        close_resp = client.post("/sessions/sess-file-memory-4/close")
        assert close_resp.status_code == 200

        # session_id เดิม แต่ memory ถูกเคลียร์ไปแล้ว — ต้อง fall through ไป Orchestrator
        # ตามปกติแม้ url ว่างเปล่าเหมือนเดิมและ goal เป็นคำถามต่อยอดแบบเดิมทุกประการ
        turn2 = client.post("/tasks", json={
            "url": "", "goal": "แต่ละวันทำอะไรบ้าง", "session_id": "sess-file-memory-4",
        })
        final2 = _poll_until(client, turn2.json()["task_id"])

    assert final2["status"] == "done"
    assert final2["result"] == _FAKE_RESULT
    MockOrchestrator.return_value.run_task.assert_awaited_once()


def test_create_task_ordinary_goal_still_uses_orchestrator_normally(client):
    """sanity check: goal ปกติที่ต้องใช้ browser จริง ต้องยังผ่าน Orchestrator.run_task()
    เหมือนเดิมทุกประการ ไม่ใช่ general-chat path ที่เพิ่งเพิ่มเข้ามาโดยไม่ตั้งใจ"""
    with patch("backend.app.api.routes.Orchestrator") as MockOrchestrator:
        MockOrchestrator.return_value.run_task = AsyncMock(return_value=_FAKE_RESULT)

        resp = client.post("/tasks", json={"url": "https://example.com", "goal": "ค้นหาสินค้า iPhone"})
        task_id = resp.json()["task_id"]

        final = _poll_until(client, task_id)

    assert final["status"] == "done"
    assert final["result"] == _FAKE_RESULT
    MockOrchestrator.return_value.run_task.assert_awaited_once()


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
            "/tasks", json={"url": "https://example.com", "goal": "เปิดเว็บ", "session_id": "sess-1",
                  "session_owner_token": "tok-1"},
        )
        _poll_until(client, resp1.json()["task_id"])
        pool_after_first = client.get("/pool/status").json()

        resp2 = client.post(
            "/tasks", json={"url": "https://example.com", "goal": "sign in", "session_id": "sess-1",
                  "session_owner_token": "tok-1"},
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


# --- W19-6 ("Master Controller" MODULE 2/3 — persistent extracted_memory buffer) ---
# route_multi_turn_strategy()/extract_structured_items()/chat_response() เองมีเทสต์ครบใน
# test_llm.py/test_intent_and_summarization.py แล้ว — กลุ่มนี้เทสต์แค่ว่า routes.py ต่อสาย
# เข้ากับ SessionRegistry.BrowserSession.extracted_memory จริง (อ่าน/เขียน/ใช้ตัดสินใจ)


# SEC-4 follow-up: session_id เดิมจะถูก "ใช้ต่อ" ได้ก็ต่อเมื่อแนบ owner_token เดิมมาด้วย
# (session_registry.py::BrowserSession.owner_token) — helper กับเทิร์นถัดไปจึงต้องใช้ค่าเดียวกัน
_MEMORY_SESSION_TOKEN = "tok-mem"


def _create_session_with_memory(client, session_id: str, memory: list[dict]) -> None:
    """สร้าง session ผ่าน POST /tasks จริง (mock run_task ธรรมดา) แล้ว inject
    extracted_memory เข้า session object ตรงๆ จำลองว่าเทิร์นก่อนหน้าเคย extract list ไว้
    แล้ว (ปกติ _update_extracted_memory() เป็นคนเติมให้ — เทสต์กลุ่มนี้ตั้งใจแยกเทสต์ส่วนนั้น
    ต่างหาก ไม่ปนกับเทสต์การ "ใช้" memory)

    ใช้ patch.object(Orchestrator, "run_task", ...) แทนการ patch ทั้ง class ตรงๆ (ต่างจาก
    เทสต์ทั่วไปในไฟล์นี้) เพราะโค้ดที่เทสต์กลุ่มนี้ใช้เรียก Orchestrator._llm_backend() (static
    method จริง) ด้วย — patch ทั้ง class จะทำให้ _llm_backend() กลายเป็น MagicMock ที่คืนค่า
    unpack ไม่ได้ไปด้วยโดยไม่ตั้งใจ"""
    with patch.object(Orchestrator, "run_task", AsyncMock(return_value=_FAKE_RESULT)):
        resp = client.post(
            "/tasks", json={"url": "https://example.com", "goal": "เปิดเว็บ", "session_id": session_id,
                  "session_owner_token": _MEMORY_SESSION_TOKEN},
        )
        _poll_until(client, resp.json()["task_id"])
    session = client.app.state.session_registry.get(session_id)
    session.page.url = "https://example.com/search?q=เพลง"
    session.extracted_memory = memory


_FAKE_SONG_LIST = [
    {"item_index": 1, "title": "เพลงที่ 1", "price": "", "status": "", "url": "", "attributes": {}},
    {"item_index": 2, "title": "เพลงที่ 2", "price": "", "status": "", "url": "", "attributes": {}},
    {"item_index": 3, "title": "เพลงรักชาติ", "price": "", "status": "", "url": "", "attributes": {}},
]


def test_create_task_replies_from_memory_without_touching_run_task(client):
    _create_session_with_memory(client, "sess-mem-reply", _FAKE_SONG_LIST)
    decision = {
        "context_analysis": {"is_continuation_of_previous_turn": True, "target_entity_from_memory": ""},
        "chosen_strategy": "REPLY_FROM_MEMORY",
        "reasoning": "คำตอบอยู่ใน buffer แล้ว",
        "planned_action": {"tool": "reply", "target_selector": "", "parameters": {}},
    }
    mock_run_task = AsyncMock(return_value=_FAKE_RESULT)
    with patch.object(Orchestrator, "run_task", mock_run_task), \
         patch("backend.app.api.routes.llm.route_multi_turn_strategy", AsyncMock(return_value=decision)) as mock_route, \
         patch("backend.app.api.routes.llm.chat_response", AsyncMock(return_value="มีเพลงทั้งหมด 3 เพลงครับ")) as mock_chat:
        resp = client.post(
            "/tasks", json={"url": "https://example.com", "goal": "มีเพลงกี่เพลง", "session_id": "sess-mem-reply",
                  "session_owner_token": _MEMORY_SESSION_TOKEN},
        )
        final = _poll_until(client, resp.json()["task_id"])

    assert final["status"] == "done"
    assert final["result"]["message"] == "มีเพลงทั้งหมด 3 เพลงครับ"
    assert final["result"]["steps"] == 0
    mock_route.assert_awaited_once()
    mock_run_task.assert_not_called()
    mock_chat.assert_awaited_once()


def test_create_task_augments_goal_with_target_entity_for_ordinal_selection(client):
    """"เล่นเพลงที่ 3" -> IN_PAGE_ACTION พร้อม target_entity_from_memory ที่ผูกกับรายการ
    จริงจาก buffer -> goal ที่ส่งเข้า run_task() ต้องมีรายละเอียดนั้นแนบไปด้วย ไม่ใช่แค่
    "เล่นเพลงที่ 3" ดิบๆ ที่ loop หลักต้องตีความ ordinal เองจาก DOM ใหม่"""
    _create_session_with_memory(client, "sess-mem-ordinal", _FAKE_SONG_LIST)
    decision = {
        "context_analysis": {"is_continuation_of_previous_turn": True, "target_entity_from_memory": "เพลงรักชาติ"},
        "chosen_strategy": "IN_PAGE_ACTION",
        "reasoning": "ต้องคลิกเล่นเพลงที่ 3 จาก buffer",
        "planned_action": {"tool": "click", "target_selector": "", "parameters": {}},
    }
    mock_run_task = AsyncMock(return_value=_FAKE_RESULT)
    with patch.object(Orchestrator, "run_task", mock_run_task), \
         patch("backend.app.api.routes.llm.route_multi_turn_strategy", AsyncMock(return_value=decision)):
        resp = client.post(
            "/tasks", json={"url": "https://example.com", "goal": "เล่นเพลงที่ 3", "session_id": "sess-mem-ordinal",
                  "session_owner_token": _MEMORY_SESSION_TOKEN},
        )
        _poll_until(client, resp.json()["task_id"])

    dispatched_goal = mock_run_task.await_args.kwargs["goal"]
    assert "เล่นเพลงที่ 3" in dispatched_goal
    assert "เพลงรักชาติ" in dispatched_goal


def test_create_task_clears_memory_on_new_navigation_decision(client):
    _create_session_with_memory(client, "sess-mem-newnav", _FAKE_SONG_LIST)
    decision = {
        "context_analysis": {"is_continuation_of_previous_turn": False, "target_entity_from_memory": ""},
        "chosen_strategy": "NEW_NAVIGATION",
        "reasoning": "user ขอหัวข้อใหม่",
        "planned_action": {"tool": "navigate", "target_selector": "", "parameters": {}},
    }
    mock_run_task = AsyncMock(return_value=_FAKE_RESULT)
    with patch.object(Orchestrator, "run_task", mock_run_task), \
         patch("backend.app.api.routes.llm.route_multi_turn_strategy", AsyncMock(return_value=decision)):
        resp = client.post(
            "/tasks", json={"url": "https://example.com", "goal": "ไปหาเสื้อผ้าแทน", "session_id": "sess-mem-newnav",
                  "session_owner_token": _MEMORY_SESSION_TOKEN},
        )
        _poll_until(client, resp.json()["task_id"])

    dispatched_goal = mock_run_task.await_args.kwargs["goal"]
    assert dispatched_goal == "ไปหาเสื้อผ้าแทน"  # ไม่ถูกแก้ไข (NEW_NAVIGATION ไม่ augment goal)
    session = client.app.state.session_registry.get("sess-mem-newnav")
    assert session.extracted_memory == []


def test_create_task_skips_multi_turn_strategy_when_no_memory_yet(client):
    """session ใหม่/ยังไม่เคย extract อะไรเลย (extracted_memory ว่างเปล่า default) — ต้อง
    ไม่เพิ่ม LLM call แถมโดยไม่จำเป็นสำหรับ task ปกติทั่วไป"""
    with patch("backend.app.api.routes.Orchestrator") as MockOrchestrator, \
         patch("backend.app.api.routes.llm.route_multi_turn_strategy", AsyncMock()) as mock_route:
        MockOrchestrator.return_value.run_task = AsyncMock(return_value=_FAKE_RESULT)

        resp = client.post(
            "/tasks", json={"url": "https://example.com", "goal": "ค้นหาสินค้า", "session_id": "sess-mem-fresh"},
        )
        _poll_until(client, resp.json()["task_id"])

    mock_route.assert_not_awaited()


def test_update_extracted_memory_populates_buffer_from_read_page_data_history():
    """_update_extracted_memory() หลัง run_task() จบต้องดึงผลลัพธ์ read_page_data ล่าสุดจาก
    history มาจัดโครงสร้างแล้วเก็บเข้า session.extracted_memory"""
    from backend.app.api.routes import _update_extracted_memory

    session = MagicMock()
    session.extracted_memory = []
    result = {
        "history": [
            {"cmd": {"type": "click", "index": 1}, "result": "[OK] click -> สำเร็จ", "success": True},
            {
                "cmd": {"type": "read_page_data", "query": "รายชื่อเพลง", "target_hint": "table"},
                "result": "[OK] read_page_data -> 1. เพลงที่ 1\n2. เพลงที่ 2\n3. เพลงรักชาติ",
                "success": True,
            },
        ],
    }
    with patch(
        "backend.app.api.routes.llm.extract_structured_items",
        AsyncMock(return_value=_FAKE_SONG_LIST),
    ) as mock_extract:
        asyncio.run(_update_extracted_memory(session, result, "anthropic"))

    assert session.extracted_memory == _FAKE_SONG_LIST
    mock_extract.assert_awaited_once()
    call_args = mock_extract.await_args.args
    assert "เพลงรักชาติ" in call_args[2]  # page_content ที่ส่งเข้าไปคือเนื้อหาจริงจาก history


def test_update_extracted_memory_leaves_buffer_untouched_when_no_read_page_data():
    from backend.app.api.routes import _update_extracted_memory

    session = MagicMock()
    session.extracted_memory = _FAKE_SONG_LIST
    result = {"history": [{"cmd": {"type": "click", "index": 1}, "result": "[OK] click -> สำเร็จ", "success": True}]}

    with patch("backend.app.api.routes.llm.extract_structured_items", AsyncMock()) as mock_extract:
        asyncio.run(_update_extracted_memory(session, result, "anthropic"))

    mock_extract.assert_not_awaited()
    assert session.extracted_memory == _FAKE_SONG_LIST  # ไม่ถูกล้างทิ้ง


def test_close_session_then_reusing_id_creates_new_page(client):
    with patch("backend.app.api.routes.Orchestrator") as MockOrchestrator:
        mock_run_task = AsyncMock(return_value=_FAKE_RESULT)
        MockOrchestrator.return_value.run_task = mock_run_task

        resp1 = client.post(
            "/tasks", json={"url": "https://example.com", "goal": "เปิดเว็บ", "session_id": "sess-close",
                  "session_owner_token": "tok-close"},
        )
        _poll_until(client, resp1.json()["task_id"])

        close_resp = client.post("/sessions/sess-close/close?session_owner_token=tok-close")
        assert close_resp.status_code == 200
        assert client.get("/pool/status").json()["available"] == 2  # คืน browser กลับ pool แล้ว

        resp2 = client.post(
            "/tasks", json={"url": "https://example.com", "goal": "เปิดใหม่", "session_id": "sess-close",
                  "session_owner_token": "tok-close"},
        )
        _poll_until(client, resp2.json()["task_id"])

    first_kwargs = mock_run_task.await_args_list[0].kwargs
    second_kwargs = mock_run_task.await_args_list[1].kwargs
    assert first_kwargs["page"] is not second_kwargs["page"]  # session ใหม่ (id เดิม) = page ใหม่


def test_close_unknown_session_returns_404(client):
    resp = client.post("/sessions/does-not-exist/close")
    assert resp.status_code == 404


# --- Security (SEC-4 follow-up): session_owner_token — session_id เดิมเป็นแค่ string ที่
# ใครก็ตามที่มี X-API-Key เดียวกันแนบเข้าไปใช้ต่อได้เลย ไม่มีการเช็คความเป็นเจ้าของ (ดู
# core/session_registry.py::BrowserSession.owner_token) — เทสต์กลุ่มนี้ยิงผ่าน endpoint
# จริงทั้งหมด (ไม่ใช่ unit test ของ SessionRegistry เอง — ดู test_session_registry.py)


def test_close_session_with_wrong_owner_token_returns_403(client):
    with patch("backend.app.api.routes.Orchestrator") as MockOrchestrator:
        MockOrchestrator.return_value.run_task = AsyncMock(return_value=_FAKE_RESULT)
        resp = client.post(
            "/tasks",
            json={
                "url": "https://example.com", "goal": "เปิดเว็บ",
                "session_id": "sess-owned", "session_owner_token": "correct-secret",
            },
        )
        _poll_until(client, resp.json()["task_id"])

    close_resp = client.post("/sessions/sess-owned/close?session_owner_token=wrong-secret")
    assert close_resp.status_code == 403


def test_close_session_with_correct_owner_token_succeeds(client):
    with patch("backend.app.api.routes.Orchestrator") as MockOrchestrator:
        MockOrchestrator.return_value.run_task = AsyncMock(return_value=_FAKE_RESULT)
        resp = client.post(
            "/tasks",
            json={
                "url": "https://example.com", "goal": "เปิดเว็บ",
                "session_id": "sess-owned-2", "session_owner_token": "correct-secret",
            },
        )
        _poll_until(client, resp.json()["task_id"])

    close_resp = client.post("/sessions/sess-owned-2/close?session_owner_token=correct-secret")
    assert close_resp.status_code == 200


def test_close_session_without_owner_token_is_rejected(client):
    """The HTTP layer must not allow a caller to omit the session secret."""
    with patch("backend.app.api.routes.Orchestrator") as MockOrchestrator:
        MockOrchestrator.return_value.run_task = AsyncMock(return_value=_FAKE_RESULT)
        resp = client.post(
            "/tasks", json={"url": "https://example.com", "goal": "เปิดเว็บ", "session_id": "sess-no-token"},
        )
        _poll_until(client, resp.json()["task_id"])

    close_resp = client.post("/sessions/sess-no-token/close")
    assert close_resp.status_code == 403


def test_create_task_with_mismatched_session_owner_token_fails_the_task(client):
    """create_task() รัน session resolution ข้างในตัว background task (ไม่ใช่ synchronous
    endpoint handler) — ownership mismatch เลยปรากฏเป็น task status="error" แทนที่จะเป็น
    403 ตรงๆ (ดู routes.py::_run_with_resolved_browser) แต่ยังคง block การเข้าถึงจริง
    (ไม่มีทาง run_task() ถูกเรียกด้วย browser/page ของ session คนอื่นเลย)"""
    with patch("backend.app.api.routes.Orchestrator") as MockOrchestrator:
        mock_run_task = AsyncMock(return_value=_FAKE_RESULT)
        MockOrchestrator.return_value.run_task = mock_run_task

        first = client.post(
            "/tasks",
            json={
                "url": "https://example.com", "goal": "เปิดเว็บ",
                "session_id": "sess-hijack-attempt", "session_owner_token": "owner-secret",
            },
        )
        _poll_until(client, first.json()["task_id"])

        second = client.post(
            "/tasks",
            json={
                "url": "https://example.com", "goal": "แอบใช้ต่อ",
                "session_id": "sess-hijack-attempt", "session_owner_token": "attacker-guess",
            },
        )
        final = _poll_until(client, second.json()["task_id"])

    assert final["status"] == "error"
    assert "owner_token" in final["error"]
    # run_task() ต้องถูกเรียกแค่รอบเดียว (ของ request แรกที่เป็นเจ้าของจริง) ไม่ใช่ 2 รอบ
    mock_run_task.assert_awaited_once()


def test_generate_plan_with_wrong_session_owner_token_returns_403(client):
    with patch("backend.app.api.routes.Orchestrator") as MockOrchestrator:
        MockOrchestrator.return_value.run_task = AsyncMock(return_value=_FAKE_RESULT)
        resp = client.post(
            "/tasks",
            json={
                "url": "https://example.com", "goal": "เปิดเว็บ",
                "session_id": "sess-plan-owned", "session_owner_token": "correct-secret",
            },
        )
        _poll_until(client, resp.json()["task_id"])

        plan_resp = client.post(
            "/api/generate_plan",
            json={
                "url": "https://example.com", "goal": "ทำต่อ",
                "session_id": "sess-plan-owned", "session_owner_token": "wrong-secret",
            },
        )
    assert plan_resp.status_code == 403


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
    assert resp.json() == {
        "plan": "1. Do X", "is_qa": False, "source": "llm",
        "template_id": None, "slot_values": None, "steps": None,
    }
    mock_generate_plan.assert_awaited_once_with(
        "https://example.com", "ทดสอบ", provider=None, page=None, site_manual_context="",
        previous_user_goal="", previous_assistant_message="",
    )
    # ไม่มีทาง touch pool/session registry เลยจาก endpoint นี้
    assert client.get("/pool/status").json() == {"size": 2, "available": 2, "in_use": 0}
    assert client.get("/sessions").json() == []


def test_generate_plan_forwards_previous_turn_context_for_anaphora_resolution(client):
    """W20 ("Context-Aware Implicit Execution", บั๊กจริงที่ user รายงาน): req.previous_user_goal/
    previous_assistant_message (เทิร์นก่อนหน้าจาก general-chat ที่แนะนำชื่อเพลง — ส่งมาจาก
    conversation history ฝั่ง client เอง) ต้องส่งต่อเข้า Orchestrator.generate_plan() ตรงๆ ให้
    LLM แก้คำอ้างอิงกำกวมอย่าง "okเปิดให้หน่อย" ได้"""
    with patch("backend.app.api.routes.Orchestrator") as MockOrchestrator:
        mock_generate_plan = AsyncMock(return_value="1. Open YouTube\n2. Search\n3. Play")
        MockOrchestrator.return_value.generate_plan = mock_generate_plan

        resp = client.post("/api/generate_plan", json={
            "url": "", "goal": "okเปิดให้หน่อย",
            "previous_user_goal": "ขอเพลงเศร้าๆหน่อย",
            "previous_assistant_message": "แนะนำเพลง \"โปรดส่งใครมารักฉันที\" ครับ",
        })

    assert resp.status_code == 200
    mock_generate_plan.assert_awaited_once_with(
        "", "okเปิดให้หน่อย", provider=None, page=None, site_manual_context="",
        previous_user_goal="ขอเพลงเศร้าๆหน่อย",
        previous_assistant_message="แนะนำเพลง \"โปรดส่งใครมารักฉันที\" ครับ",
    )


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
        previous_user_goal="", previous_assistant_message="",
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
            json={"url": "https://example.com", "goal": "เปิดเว็บ", "session_id": "sess-plan", "session_owner_token": "plan-owner"},
        )
        _poll_until(client, resp1.json()["task_id"])

        resp2 = client.post(
            "/api/generate_plan",
            json={"url": "https://example.com", "goal": "sign in", "session_id": "sess-plan", "session_owner_token": "plan-owner"},
        )

    assert resp2.status_code == 200
    assert resp2.json() == {
        "plan": "1. Sign in", "is_qa": False, "source": "llm",
        "template_id": None, "slot_values": None, "steps": None,
    }
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
            json={"url": "https://example.com", "goal": "เปิดเว็บ", "session_id": "sess-exec", "session_owner_token": "exec-owner"},
        )
        _poll_until(client, resp1.json()["task_id"])
        resp2 = client.post(
            "/api/execute_plan",
            json={"url": "https://example.com", "goal": "sign in", "session_id": "sess-exec", "session_owner_token": "exec-owner"},
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

    # extract_domain() ตัด "www." ออกโดยเจตนา (ดู permission/rules.py::extract_domain())
    mock_save.assert_called_once_with("saucedemo.com", "login", "1. Open site\n2. Log in")


def test_execute_plan_without_a_plan_does_not_touch_plan_memory(client):
    with patch("backend.app.api.routes.Orchestrator") as MockOrchestrator, \
         patch("backend.app.api.routes.plan_memory.save_confirmed_plan") as mock_save:
        mock_run_task = AsyncMock(return_value=_FAKE_RESULT)
        MockOrchestrator.return_value.run_task = mock_run_task

        resp = client.post("/api/execute_plan", json={"url": "https://example.com", "goal": "ทดสอบ"})
        _poll_until(client, resp.json()["task_id"])

    mock_save.assert_not_called()


# --- W_procmem: /api/execute_plan fast-path branch (settings.enable_procedural_memory) ---

_FASTPATH_RESULT = {
    "success": True, "steps": 2, "message": "fast-path done", "history": [], "tokens": {},
    "plan": "", "final_page_state": "", "execution_mode": "fastpath",
}
_FASTPATH_BODY = {
    "url": "https://example.com/login", "goal": "log in as alice",
    "plan": "1. Fill Username\n2. Click Login",
    "execution_mode": "fastpath", "template_id": "t1",
    "slot_values": {"username": "alice"},
    "steps": [{"action": "fill", "target": {"accessible_name": "Username"}, "value": "{{username}}"}],
}


def test_execute_plan_uses_run_fastpath_when_flag_enabled_and_fields_present(client, monkeypatch):
    monkeypatch.setattr(settings, "enable_procedural_memory", True)
    with patch("backend.app.api.routes.Orchestrator") as MockOrchestrator, \
         patch("backend.app.api.routes.plan_memory.save_confirmed_plan") as mock_save:
        mock_run_fastpath = AsyncMock(return_value=_FASTPATH_RESULT)
        MockOrchestrator.return_value.run_fastpath = mock_run_fastpath

        resp = client.post("/api/execute_plan", json=_FASTPATH_BODY)
        final = _poll_until(client, resp.json()["task_id"])

    assert final["result"] == _FASTPATH_RESULT
    kwargs = mock_run_fastpath.await_args.kwargs
    assert kwargs["template_id"] == "t1"
    assert kwargs["slot_values"] == {"username": "alice"}
    # แผนที่ render จาก steps ไม่ใช่ข้อความที่ user พิมพ์เอง — ต้องไม่ปนเข้า plan_memory
    mock_save.assert_not_called()


def test_execute_plan_falls_back_to_run_task_when_flag_disabled(client):
    """default settings.enable_procedural_memory=False — เพิกเฉยต่อ execution_mode/
    template_id/steps ที่ส่งมา วิ่งผ่าน run_task ปกติเหมือนไม่มี feature นี้เลย"""
    with patch("backend.app.api.routes.Orchestrator") as MockOrchestrator:
        mock_run_task = AsyncMock(return_value=_FAKE_RESULT)
        mock_run_fastpath = AsyncMock()
        MockOrchestrator.return_value.run_task = mock_run_task
        MockOrchestrator.return_value.run_fastpath = mock_run_fastpath

        resp = client.post("/api/execute_plan", json=_FASTPATH_BODY)
        _poll_until(client, resp.json()["task_id"])

    mock_run_task.assert_awaited_once()
    mock_run_fastpath.assert_not_called()


def test_execute_plan_falls_back_to_run_task_when_use_user_browser_requested(client, monkeypatch):
    """use_user_browser ยังไม่รองรับใน run_fastpath() (ดู docstring) — ต้อง fallback ไป
    slow path เงียบๆ แม้ flag เปิดและ template_id/steps มีมาครบก็ตาม"""
    monkeypatch.setattr(settings, "enable_procedural_memory", True)
    with patch("backend.app.api.routes.Orchestrator") as MockOrchestrator:
        mock_run_task = AsyncMock(return_value=_FAKE_RESULT)
        mock_run_fastpath = AsyncMock()
        MockOrchestrator.return_value.run_task = mock_run_task
        MockOrchestrator.return_value.run_fastpath = mock_run_fastpath

        body = {**_FASTPATH_BODY, "use_user_browser": True}
        resp = client.post("/api/execute_plan", json=body)
        _poll_until(client, resp.json()["task_id"])

    mock_run_task.assert_awaited_once()
    mock_run_fastpath.assert_not_called()


def test_execute_plan_fastpath_reuses_session_page(client, monkeypatch):
    monkeypatch.setattr(settings, "enable_procedural_memory", True)
    with patch("backend.app.api.routes.Orchestrator") as MockOrchestrator:
        mock_run_fastpath = AsyncMock(return_value=_FASTPATH_RESULT)
        MockOrchestrator.return_value.run_fastpath = mock_run_fastpath

        body = {**_FASTPATH_BODY, "session_id": "sess-fastpath", "session_owner_token": "fastpath-owner"}
        resp = client.post("/api/execute_plan", json=body)
        _poll_until(client, resp.json()["task_id"])

    kwargs = mock_run_fastpath.await_args.kwargs
    assert kwargs["session_id"] == "sess-fastpath"
    assert kwargs["page"] is not None


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
    assert resp.json() == {
        "plan": "1. Reused step", "is_qa": False, "source": "plan_memory",
        "template_id": None, "slot_values": None, "steps": None,
    }
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
    assert resp.json() == {
        "plan": "1. Fresh LLM draft", "is_qa": False, "source": "llm",
        "template_id": None, "slot_values": None, "steps": None,
    }
    mock_generate_plan.assert_awaited_once()


# --- W_procmem: procedural memory check in generate_plan() (settings.enable_procedural_memory) ---

_FAKE_TEMPLATE_CANDIDATE = {
    "template_id": "t1", "intent_key": "k1", "version": 1,
    "goal_pattern": "Log in with a username and password",
    "url_pattern": "https://example.com/login",
    "steps": [{"action": "fill", "target": {"accessible_name": "Username"}, "value": "{{username}}"}],
    "slots": [{"name": "username"}],
    "distance": 0.2,
}


def test_generate_plan_skips_procedural_check_when_flag_disabled(client):
    """default settings.enable_procedural_memory=False — ไม่แตะ find_candidate_templates
    เลยแม้แต่ครั้งเดียว"""
    with patch("backend.app.api.routes.Orchestrator") as MockOrchestrator, \
         patch("backend.app.api.routes.plan_memory.find_matching_plan", return_value=None), \
         patch("backend.app.api.routes.procedural_memory.find_candidate_templates") as mock_find:
        MockOrchestrator.return_value.generate_plan = AsyncMock(return_value="1. Fresh LLM draft")

        resp = client.post("/api/generate_plan", json={"url": "https://example.com", "goal": "log in"})

    assert resp.status_code == 200
    mock_find.assert_not_called()


def test_generate_plan_returns_procedural_reuse_when_confident_match_found(client, monkeypatch):
    monkeypatch.setattr(settings, "enable_procedural_memory", True)
    with patch("backend.app.api.routes.Orchestrator") as MockOrchestrator, \
         patch("backend.app.api.routes.plan_memory.find_matching_plan") as mock_plan_memory, \
         patch(
             "backend.app.api.routes.procedural_memory.find_candidate_templates",
             return_value=[_FAKE_TEMPLATE_CANDIDATE],
         ), \
         patch(
             "backend.app.api.routes.llm.plan_with_procedural_memory",
             AsyncMock(return_value={
                 "decision": "reuse", "template_id": "t1", "confidence": 0.9,
                 "slot_values": {"username": "alice"}, "patch": None, "reason": "match",
             }),
         ):
        MockOrchestrator._llm_backend.return_value = ("fake-client", "fake-model", None, None, None)
        resp = client.post("/api/generate_plan", json={"url": "https://example.com/login", "goal": "log in as alice"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["source"] == "procedural_reuse"
    assert body["template_id"] == "t1"
    assert body["slot_values"] == {"username": "alice"}
    assert "alice" in body["plan"]
    # เจอ procedural match มั่นใจพอแล้ว ต้องข้าม plan_memory ไปเลย ไม่เรียกซ้ำ
    mock_plan_memory.assert_not_called()


def test_generate_plan_passes_credentials_exist_as_has_auto_login_to_planner(client, monkeypatch):
    """W_procmem: บั๊กจริงที่เจอตอน Phase 4 validation — Planner ต้องรู้ว่าโดเมนนี้มี
    auto-login credential เก็บไว้ไหม (ดู orchestrator.py::_maybe_auto_login) ไม่งั้นมัน
    จะปฏิเสธ reuse candidate ที่ไม่มี step login ทั้งที่ auto-login จัดการให้แล้วนอก
    template เสมอ"""
    monkeypatch.setattr(settings, "enable_procedural_memory", True)
    with patch("backend.app.api.routes.Orchestrator") as MockOrchestrator, \
         patch(
             "backend.app.api.routes.procedural_memory.find_candidate_templates",
             return_value=[_FAKE_TEMPLATE_CANDIDATE],
         ), \
         patch("backend.app.api.routes.credentials_exist", return_value=True) as mock_creds_exist, \
         patch(
             "backend.app.api.routes.llm.plan_with_procedural_memory",
             AsyncMock(return_value={
                 "decision": "plan_fresh", "template_id": None, "confidence": 0.2,
                 "slot_values": {}, "patch": None, "reason": "n/a",
             }),
         ) as mock_planner:
        MockOrchestrator._llm_backend.return_value = ("fake-client", "fake-model", None, None, None)
        client.post("/api/generate_plan", json={"url": "https://example.com/login", "goal": "log in"})

    mock_creds_exist.assert_called_once_with("example.com")
    _, kwargs = mock_planner.call_args
    assert kwargs["has_auto_login"] is True


def test_generate_plan_falls_through_to_plan_memory_when_procedural_decision_is_plan_fresh(client, monkeypatch):
    monkeypatch.setattr(settings, "enable_procedural_memory", True)
    with patch("backend.app.api.routes.Orchestrator") as MockOrchestrator, \
         patch(
             "backend.app.api.routes.plan_memory.find_matching_plan",
             return_value={"intent_key": "k1", "version": 1, "plan": "1. Reused step", "distance": 0.1},
         ), \
         patch(
             "backend.app.api.routes.procedural_memory.find_candidate_templates",
             return_value=[_FAKE_TEMPLATE_CANDIDATE],
         ), \
         patch(
             "backend.app.api.routes.llm.plan_with_procedural_memory",
             AsyncMock(return_value={
                 "decision": "plan_fresh", "template_id": None, "confidence": 0.2,
                 "slot_values": {}, "patch": None, "reason": "no good match",
             }),
         ):
        MockOrchestrator._llm_backend.return_value = ("fake-client", "fake-model", None, None, None)
        resp = client.post("/api/generate_plan", json={"url": "https://example.com/login", "goal": "log in"})

    assert resp.status_code == 200
    assert resp.json()["source"] == "plan_memory"
    assert resp.json()["plan"] == "1. Reused step"


def test_generate_plan_applies_patch_for_adapt_decision(client, monkeypatch):
    monkeypatch.setattr(settings, "enable_procedural_memory", True)
    with patch("backend.app.api.routes.Orchestrator") as MockOrchestrator, \
         patch("backend.app.api.routes.plan_memory.find_matching_plan"), \
         patch(
             "backend.app.api.routes.procedural_memory.find_candidate_templates",
             return_value=[_FAKE_TEMPLATE_CANDIDATE],
         ), \
         patch(
             "backend.app.api.routes.llm.plan_with_procedural_memory",
             AsyncMock(return_value={
                 "decision": "adapt", "template_id": "t1", "confidence": 0.8,
                 "slot_values": {"username": "bob"},
                 "patch": [{"op": "insert", "index": 0, "step": {"action": "goto"}}],
                 "reason": "needs an extra goto step",
             }),
         ):
        MockOrchestrator._llm_backend.return_value = ("fake-client", "fake-model", None, None, None)
        resp = client.post("/api/generate_plan", json={"url": "https://example.com/login", "goal": "log in as bob"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["source"] == "procedural_adapt"
    assert body["steps"][0]["action"] == "goto"  # inserted by the patch
    assert len(body["steps"]) == 2


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


def test_learn_site_result_includes_summary_from_manual(client, isolated_manuals_dir):
    """W26: manual.summary (ภาพรวม "เว็บไซต์นี้ทำอะไรได้บ้าง" จาก describe_site()) ต้องไหล
    ออกไปถึง learn_done SSE event's result.summary ให้ frontend โชว์ได้"""
    from backend.app.site_learning.schema import PageInfo, SiteManual

    fake_manual = SiteManual(
        website="example.com",
        pages=[PageInfo(name="Home", url="https://example.com/")],
        summary="เว็บไซต์นี้ใช้ดูสินค้าและสั่งซื้อออนไลน์ได้",
    )

    fake_browser = AsyncMock()
    fake_playwright_obj = AsyncMock()
    fake_playwright_obj.chromium.launch = AsyncMock(return_value=fake_browser)
    fake_playwright_cm = AsyncMock()
    fake_playwright_cm.__aenter__ = AsyncMock(return_value=fake_playwright_obj)
    fake_playwright_cm.__aexit__ = AsyncMock(return_value=False)

    with patch("backend.app.api.routes.async_playwright", return_value=fake_playwright_cm), \
         patch("backend.app.api.routes.crawl_site", AsyncMock(return_value=fake_manual)):
        resp = client.post("/api/site-manual/learn", json={"url": "https://example.com/"})
        learn_id = resp.json()["learn_id"]

        import time
        deadline = time.monotonic() + 5.0
        stream_resp = None
        while time.monotonic() < deadline:
            stream_resp = client.get(f"/api/site-manual/learn/{learn_id}/stream")
            if "learn_done" in stream_resp.text:
                break
            time.sleep(0.02)

    # stream_resp.text เป็น SSE ("data: {...}\n\n") ที่ json.dumps escape ตัวอักษรไทยเป็น
    # \uXXXX เสมอ (ensure_ascii default) — เทียบ literal string ตรงๆ ไม่เจอ ต้อง parse JSON
    # ออกมาก่อนถึงจะเทียบได้ถูกต้อง
    done_line = next(line for line in stream_resp.text.splitlines() if "learn_done" in line)
    event = json.loads(done_line[len("data: "):])
    assert event["result"]["summary"] == "เว็บไซต์นี้ใช้ดูสินค้าและสั่งซื้อออนไลน์ได้"


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


# ---------------- W23: ถามคนจริงกลางคัน crawl แทนบังคับกรอกไว้ล่วงหน้า ----------------


def test_learn_site_passes_on_credentials_needed_callback_to_crawl_site(client, isolated_manuals_dir):
    """learn_site() ต้องผูก on_credentials_needed เข้ากับ crawl_site() เสมอ (แม้ตอนที่
    ผู้ใช้ไม่ได้กรอก username/password มาเลยตั้งแต่ต้น) ให้ crawler เรียกถามผ่าน SSE ได้ถ้า
    เจอหน้า login กลางคัน — เทสต์นี้เช็คแค่การผูก (wiring) ตรงๆ ผ่าน mock ไม่ได้เล่น
    concurrency จริงข้าม event loop (ดู test_learn_manager.py สำหรับพฤติกรรม request/
    resolve_credentials() เต็มๆ)"""
    from backend.app.site_learning.schema import PageInfo, SiteManual

    fake_manual = SiteManual(website="example.com", pages=[PageInfo(name="Home", url="https://example.com/")])

    with patch("backend.app.api.routes.async_playwright", return_value=_fake_learn_playwright()), \
         patch("backend.app.api.routes.crawl_site", AsyncMock(return_value=fake_manual)) as mock_crawl:
        resp = client.post("/api/site-manual/learn", json={"url": "https://example.com/"})
        learn_id = resp.json()["learn_id"]

        import time
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            stream_resp = client.get(f"/api/site-manual/learn/{learn_id}/stream")
            if "learn_done" in stream_resp.text:
                break
            time.sleep(0.02)

    call_kwargs = mock_crawl.await_args.kwargs
    assert callable(call_kwargs.get("on_credentials_needed"))


def test_respond_learn_credentials_returns_404_for_unknown_request_id(client):
    resp = client.post(
        "/api/site-manual/learn/does-not-exist/credentials",
        json={"request_id": "req-1", "username": "alice", "password": "s3cr3t"},
    )
    assert resp.status_code == 404


def test_save_site_credentials_endpoint_persists_and_status_reflects_it(client, isolated_manuals_dir):
    from backend.app.site_learning import storage

    assert client.get("/api/site-manual/example.com/credentials/status").json() == {"exists": False}

    resp = client.post(
        "/api/site-manual/example.com/credentials", json={"username": "alice", "password": "s3cr3t"},
    )
    assert resp.status_code == 204

    assert client.get("/api/site-manual/example.com/credentials/status").json() == {"exists": True}
    assert storage.load_credentials("example.com") == {"username": "alice", "password": "s3cr3t"}


def test_save_site_credentials_endpoint_normalizes_www_prefix(client, isolated_manuals_dir):
    """path param "domain" ไม่ผ่าน extract_domain(url) เหมือนจุดอื่น (รับ hostname ตรงๆ) —
    ต้อง normalize เอง (ตัด www./lowercase) ไม่งั้น credential ที่บันทึกผ่าน
    www.example.com จะหาไม่เจอตอน core/orchestrator.py::_maybe_auto_login ค้นด้วย
    extract_domain(page.url) ที่ตัด www. ออกแล้วเสมอ"""
    from backend.app.site_learning import storage

    resp = client.post(
        "/api/site-manual/www.Example.com/credentials", json={"username": "alice", "password": "s3cr3t"},
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

"""Security 1.1: verify_api_key() (routes.py) — ผูกไว้ที่ระดับ APIRouter(dependencies=...)
ครอบคลุมทุก route ใน api_router โดยอัตโนมัติ ไม่ต้องแก้ทีละ endpoint — settings.api_key
เป็น None (default) แปลว่า auth ปิดสำหรับ local dev เท่านั้น เทสต์ทั้งไฟล์นี้ตั้งค่าไว้
ชัดเจนก่อนเสมอ (monkeypatch) เพื่อไม่ให้กระทบเทสต์อื่นทั้งหมดใน test_api.py ที่ยิงโดยไม่มี
header เลย (ยังต้องผ่านเหมือนเดิมเพราะ api_key ไม่ได้ตั้งในสภาพแวดล้อมเทสต์ปกติ)

reuse fixtures จาก test_api.py (client/_isolated_chroma/_FakeBrowserPool) แทนการเขียนซ้ำ —
import ฟังก์ชันที่ @pytest.fixture มาตรงๆ ก็ยังทำงานเป็น fixture ได้ปกติ (pytest หา fixture
จาก function object ที่ registered ในโมดูล ไม่ใช่จาก "ที่ประกาศต้นทาง")
"""

from unittest.mock import patch

import pytest

from backend.app.config import settings
from backend.tests.test_api import _FAKE_RESULT, _isolated_chroma, client  # noqa: F401


@pytest.fixture
def _api_key_required(monkeypatch):
    monkeypatch.setattr(settings, "api_key", "test-secret-key")
    yield "test-secret-key"


def test_no_api_key_setting_leaves_endpoints_open(client):
    """ค่า default (settings.api_key = None) = auth ปิด — ไม่มี header เลยก็ต้องผ่านปกติ
    (พฤติกรรมเดิมของ dev ในเครื่อง ก่อนตั้งค่า .env)"""
    resp = client.get("/tasks")
    assert resp.status_code == 200


def test_network_exposed_server_without_api_key_is_rejected(client, monkeypatch):
    monkeypatch.setattr(settings, "api_host", "0.0.0.0")
    resp = client.get("/tasks")
    assert resp.status_code == 503


def test_missing_api_key_header_returns_401(client, _api_key_required):
    resp = client.get("/tasks")
    assert resp.status_code == 401


def test_wrong_api_key_header_returns_401(client, _api_key_required):
    resp = client.get("/tasks", headers={"X-API-Key": "wrong-key"})
    assert resp.status_code == 401


def test_correct_api_key_header_passes(client, _api_key_required):
    resp = client.get("/tasks", headers={"X-API-Key": _api_key_required})
    assert resp.status_code == 200


def test_create_task_requires_api_key(client, _api_key_required):
    with patch("backend.app.api.routes.Orchestrator") as MockOrchestrator:
        MockOrchestrator.return_value.run_task.return_value = _FAKE_RESULT
        resp = client.post("/tasks", json={"url": "https://example.com", "goal": "ทดสอบ"})
        assert resp.status_code == 401


def test_create_task_with_api_key_header_succeeds(client, _api_key_required):
    from unittest.mock import AsyncMock

    with patch("backend.app.api.routes.Orchestrator") as MockOrchestrator:
        MockOrchestrator.return_value.run_task = AsyncMock(return_value=_FAKE_RESULT)
        resp = client.post(
            "/tasks",
            json={"url": "https://example.com", "goal": "ทดสอบ"},
            headers={"X-API-Key": _api_key_required},
        )
        assert resp.status_code == 202


def test_stream_endpoint_accepts_valid_ticket_via_query_param(client, _api_key_required):
    """Security (SEC-5 follow-up): EventSource (frontend) ตั้ง custom header เองไม่ได้ —
    GET .../stream ต้องรับ ?ticket= (short-lived, ขอผ่าน POST /auth/stream-ticket ด้วย
    X-API-Key จริงก่อนเสมอ — ดู routes.py) แทน header ได้ด้วย"""
    ticket_resp = client.post("/auth/stream-ticket", headers={"X-API-Key": _api_key_required})
    assert ticket_resp.status_code == 200
    ticket = ticket_resp.json()["ticket"]

    resp = client.get(f"/tasks/does-not-exist/stream?ticket={ticket}")
    # request_id ไม่มีจริง -> 404 (ไม่ใช่ 401) พิสูจน์ว่า auth ผ่านไปแล้วก่อนถึง lookup logic
    assert resp.status_code == 404


def test_stream_endpoint_without_ticket_returns_401(client, _api_key_required):
    resp = client.get("/tasks/does-not-exist/stream")
    assert resp.status_code == 401


def test_stream_endpoint_rejects_bogus_ticket(client, _api_key_required):
    resp = client.get("/tasks/does-not-exist/stream?ticket=not-a-real-ticket")
    assert resp.status_code == 401


def test_stream_ticket_endpoint_itself_requires_api_key(client, _api_key_required):
    """ขอ ticket เองก็ยังต้องผ่าน X-API-Key header จริงก่อนเสมอ (router-level dependency
    เดียวกับทุก route อื่น) — ไม่งั้นใครก็ mint ticket เองได้โดยไม่มี key จริงเลย"""
    resp = client.post("/auth/stream-ticket")
    assert resp.status_code == 401


def test_stream_ticket_can_be_reused_within_ttl(client, _api_key_required):
    """ticket ไม่ใช่ single-use (ดู _issue_stream_ticket() docstring — EventSource
    auto-reconnect ได้) — ใช้ซ้ำได้หลายครั้งภายใน TTL เดียวกัน"""
    ticket = client.post("/auth/stream-ticket", headers={"X-API-Key": _api_key_required}).json()["ticket"]
    first = client.get(f"/tasks/does-not-exist/stream?ticket={ticket}")
    second = client.get(f"/tasks/does-not-exist/stream?ticket={ticket}")
    assert first.status_code == 404
    assert second.status_code == 404


def test_config_check_requires_api_key(client, _api_key_required):
    resp = client.get("/config/check")
    assert resp.status_code == 401
    resp_ok = client.get("/config/check", headers={"X-API-Key": _api_key_required})
    assert resp_ok.status_code == 200


def test_config_check_accepts_valid_ticket_too(client, _api_key_required):
    """ticket mechanism ไม่ได้ผูกเฉพาะ SSE endpoint — ใช้กับ route อื่นได้เหมือนกันเพราะ
    ผูกไว้ที่ verify_api_key() ระดับเดียวกันหมด (ดู routes.py)"""
    ticket = client.post("/auth/stream-ticket", headers={"X-API-Key": _api_key_required}).json()["ticket"]
    resp = client.get(f"/config/check?ticket={ticket}")
    assert resp.status_code == 200


def test_stream_ticket_expires_after_ttl(client, _api_key_required):
    """Security (SEC-5 follow-up): ticket ต้องใช้ไม่ได้อีกหลังหมดอายุ (TTL สั้นกว่า
    long-lived API key เดิมมาก — ดู _STREAM_TICKET_TTL_SECONDS) — แก้ expiry ที่เก็บไว้ใน
    store ตรงๆ ให้เป็นอดีตแทนการ sleep จริงหรือ mock time.time() ทั้ง module (ปลอดภัยกว่า
    ไม่กระทบโค้ดอื่นที่อ่านเวลาจริงระหว่างเทสต์)"""
    from backend.app.api import routes as routes_module

    ticket = client.post("/auth/stream-ticket", headers={"X-API-Key": _api_key_required}).json()["ticket"]
    routes_module._stream_tickets[ticket] = 0.0  # force-expire

    resp = client.get(f"/tasks/does-not-exist/stream?ticket={ticket}")
    assert resp.status_code == 401


def test_health_endpoint_not_gated_by_api_key(client, _api_key_required):
    """/health ประกาศตรงๆ ใน main.py (ไม่ผ่าน api_router) — ตั้งใจไม่ผูก auth (ใช้เช็ค
    liveness เฉยๆ ไม่มีข้อมูล sensitive อะไรให้ป้องกัน)"""
    resp = client.get("/health")
    assert resp.status_code == 200

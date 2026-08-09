"""Security 1.5: rate limit per-IP บน POST /tasks (10/นาที) และ POST /api/site-manual/learn
(5/นาที) เท่านั้น — endpoint ที่เปิด browser จริง/เปลือง resource (ดู routes.py::limiter) —
ไม่แตะ SSE stream/endpoint polling ปกติ กัน task ที่รันอยู่โดนตัดกลางคัน

Security (SEC-3 follow-up): POST /api/generate_plan เพิ่ม rate limit เข้ามาด้วย (20/นาที)
— เดิมไม่มีเลย ทั้งที่รับ attached_file_content_base64 (ไฟล์แนบ) ได้เหมือนกัน ยิงรัวๆ ด้วย
payload ใหญ่ (แม้จะมี size cap แล้ว — ดู schemas.py) ยังเปลือง CPU/memory ต่อ request ได้

TestClient ยิงทุก request ด้วย "testclient" identity เดียวกันหมด (get_remote_address
คืนค่าเดิมเสมอ) — ทุกเทสต์ในไฟล์นี้ reset limiter ก่อนเสมอ (ผ่าน client fixture ที่ import
มาจาก test_api.py) ให้เริ่มนับจากศูนย์ทุกครั้ง"""

from unittest.mock import AsyncMock, patch

from backend.tests.test_api import _FAKE_RESULT, _isolated_chroma, client  # noqa: F401


def test_create_task_allows_up_to_the_limit_then_429s(client):
    with patch("backend.app.api.routes.Orchestrator") as MockOrchestrator:
        MockOrchestrator.return_value.run_task = AsyncMock(return_value=_FAKE_RESULT)
        for i in range(10):
            resp = client.post("/tasks", json={"url": "https://example.com", "goal": f"task {i}"})
            assert resp.status_code == 202, f"request {i} should succeed, got {resp.status_code}"

        eleventh = client.post("/tasks", json={"url": "https://example.com", "goal": "one too many"})
        assert eleventh.status_code == 429


def test_pool_status_is_not_rate_limited(client):
    """endpoint polling ปกติ (ไม่เปิด browser ใหม่) ต้องไม่ถูกจำกัดเลย แม้ยิงเกิน 10 ครั้ง"""
    for _ in range(15):
        resp = client.get("/pool/status")
        assert resp.status_code == 200


def test_get_tasks_is_not_rate_limited(client):
    for _ in range(15):
        resp = client.get("/tasks")
        assert resp.status_code == 200


def test_generate_plan_allows_up_to_the_limit_then_429s(client):
    """ใช้ goal ที่มีคำสั่ง "/context" เพื่อ short-circuit ไป is_qa=True ทันที (ดู
    routes.py::generate_plan) ไม่ต้อง mock LLM เลย ยิงได้เร็ว/deterministic"""
    for i in range(20):
        resp = client.post(
            "/api/generate_plan", json={"url": "https://example.com", "goal": f"/context test {i}"},
        )
        assert resp.status_code == 200, f"request {i} should succeed, got {resp.status_code}"

    twenty_first = client.post(
        "/api/generate_plan", json={"url": "https://example.com", "goal": "/context one too many"},
    )
    assert twenty_first.status_code == 429

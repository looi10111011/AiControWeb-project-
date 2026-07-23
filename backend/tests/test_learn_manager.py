"""W23: LearnManager.request_credentials()/resolve_credentials()/push_event() — กลไก
human-in-the-loop สำหรับ "ขอ username/password กลางคัน crawl" mirror รูปแบบเดียวกับ
test_task_manager.py::test_request_approval_* ทุกประการ (เหตุผลเดียวกัน: ต้องควบคุม
concurrency ภายใน event loop *เดียวกัน* ตรงๆ ระหว่างสร้าง request ค้างไว้กับ resolve มัน
จาก "อีกฝั่ง" ในลูปเดียวกัน)
"""

import asyncio

import pytest

from backend.app.site_learning.learn_manager import LearnManager


def _controllable_coro(finish: asyncio.Event):
    """coro ที่ยัง "ทำงานอยู่" (record.status == "running") จนกว่า finish จะถูก set —
    กันไม่ให้ LearnManager._run() cleanup (ยกเลิก pending credential request ทั้งหมด +
    push learn_done) แทรกเข้ามากลางเทสต์ก่อนที่เราจะทันได้ resolve เอง"""

    async def _coro() -> dict:
        await finish.wait()
        return {"version": 1, "pages_found": 0}

    return _coro()


async def _drain_learn_done(record) -> None:
    """ปล่อยให้ background task (จาก submit()) จบแบบสะอาด กัน 'Task was destroyed but it
    is pending' warning ตอน event loop ปิดท้าย test"""
    event = await asyncio.wait_for(record.events.get(), timeout=1)
    assert event["kind"] == "learn_done"


@pytest.mark.asyncio
async def test_request_credentials_blocks_until_resolved_with_username_and_password():
    lm = LearnManager()
    finish = asyncio.Event()
    record = lm.submit("l1", "https://example.com", _controllable_coro(finish))

    req_task = asyncio.create_task(lm.request_credentials("l1", "example.com"))
    event = await asyncio.wait_for(record.events.get(), timeout=1)
    assert event["kind"] == "credentials_needed"
    assert event["domain"] == "example.com"
    assert "request_id" in event

    assert lm.resolve_credentials("l1", event["request_id"], "alice", "s3cr3t") is True
    result = await asyncio.wait_for(req_task, timeout=1)
    assert result == {"username": "alice", "password": "s3cr3t"}

    finish.set()
    await _drain_learn_done(record)


@pytest.mark.asyncio
async def test_request_credentials_resolves_to_none_when_user_skips():
    """username/password ว่างทั้งคู่ (ผู้ใช้กด "ข้าม") ต้อง resolve เป็น None ไม่ใช่ dict
    ว่างๆ — ให้ crawl_site() ไปต่อโดยไม่ login แทน"""
    lm = LearnManager()
    finish = asyncio.Event()
    record = lm.submit("l2", "https://example.com", _controllable_coro(finish))

    req_task = asyncio.create_task(lm.request_credentials("l2", "example.com"))
    event = await asyncio.wait_for(record.events.get(), timeout=1)

    assert lm.resolve_credentials("l2", event["request_id"], None, None) is True
    assert await asyncio.wait_for(req_task, timeout=1) is None

    finish.set()
    await _drain_learn_done(record)


@pytest.mark.asyncio
async def test_resolve_credentials_treats_partial_answer_as_skip():
    """username มาแต่ password ว่าง (หรือกลับกัน) ต้องนับเป็น "ข้าม" เหมือนกัน ไม่ใช่ครึ่งๆ
    กลางๆ — ไม่มีทาง login ด้วย credential ที่ไม่ครบคู่อยู่แล้ว"""
    lm = LearnManager()
    finish = asyncio.Event()
    record = lm.submit("l3", "https://example.com", _controllable_coro(finish))

    req_task = asyncio.create_task(lm.request_credentials("l3", "example.com"))
    event = await asyncio.wait_for(record.events.get(), timeout=1)

    assert lm.resolve_credentials("l3", event["request_id"], "alice", None) is True
    assert await asyncio.wait_for(req_task, timeout=1) is None

    finish.set()
    await _drain_learn_done(record)


@pytest.mark.asyncio
async def test_resolve_credentials_returns_false_for_unknown_request_id():
    lm = LearnManager()
    finish = asyncio.Event()
    record = lm.submit("l4", "https://example.com", _controllable_coro(finish))

    assert lm.resolve_credentials("l4", "does-not-exist", "alice", "s3cr3t") is False

    finish.set()
    await _drain_learn_done(record)


@pytest.mark.asyncio
async def test_resolve_credentials_returns_false_after_already_resolved():
    lm = LearnManager()
    finish = asyncio.Event()
    record = lm.submit("l5", "https://example.com", _controllable_coro(finish))

    req_task = asyncio.create_task(lm.request_credentials("l5", "example.com"))
    event = await asyncio.wait_for(record.events.get(), timeout=1)

    assert lm.resolve_credentials("l5", event["request_id"], "alice", "s3cr3t") is True
    assert lm.resolve_credentials("l5", event["request_id"], "bob", "hunter2") is False  # ตอบไปแล้ว
    assert await asyncio.wait_for(req_task, timeout=1) == {"username": "alice", "password": "s3cr3t"}

    finish.set()
    await _drain_learn_done(record)


def test_resolve_credentials_returns_false_for_unknown_learn_id():
    lm = LearnManager()
    assert lm.resolve_credentials("does-not-exist", "req-1", "alice", "s3cr3t") is False


@pytest.mark.asyncio
async def test_request_credentials_returns_none_immediately_for_unknown_learn_id():
    lm = LearnManager()
    assert await lm.request_credentials("does-not-exist", "example.com") is None


@pytest.mark.asyncio
async def test_learn_job_finishing_cancels_pending_credential_request():
    """crawl จบ/ถูก stop กลางคันระหว่างรอ user ตอบ credential prompt — Future ของ
    request_credentials() ต้องไม่ค้างรอตลอดไป (resolve เป็น None แทน) เหมือน
    TaskManager คู่กัน (test_request_approval_resolves_pending_when_task_finishes)"""
    lm = LearnManager()
    finish = asyncio.Event()
    record = lm.submit("l6", "https://example.com", _controllable_coro(finish))

    req_task = asyncio.create_task(lm.request_credentials("l6", "example.com"))
    await asyncio.wait_for(record.events.get(), timeout=1)  # credentials_needed — ทิ้งไป ไม่ resolve

    finish.set()  # ให้ _run() จบ -> ต้อง cleanup pending ทั้งหมดเอง
    await _drain_learn_done(record)
    assert await asyncio.wait_for(req_task, timeout=1) is None


@pytest.mark.asyncio
async def test_request_credentials_times_out_and_returns_none_if_nobody_responds():
    lm = LearnManager()
    finish = asyncio.Event()
    record = lm.submit("l7", "https://example.com", _controllable_coro(finish))

    req_task = asyncio.create_task(lm.request_credentials("l7", "example.com", timeout=0.05))
    event = await asyncio.wait_for(record.events.get(), timeout=1)
    request_id = event["request_id"]

    assert await asyncio.wait_for(req_task, timeout=1) is None
    timeout_event = await asyncio.wait_for(record.events.get(), timeout=1)
    assert timeout_event["kind"] == "credentials_timeout"
    assert timeout_event["request_id"] == request_id

    # หมดเวลาไปแล้ว — resolve ทีหลังต้องคืน False (เลิกนับเป็น pending request แล้ว)
    assert lm.resolve_credentials("l7", request_id, "alice", "s3cr3t") is False

    finish.set()
    await _drain_learn_done(record)

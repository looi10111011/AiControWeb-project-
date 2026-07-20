"""W10[B]: TaskManager.request_approval()/resolve_approval()/push_event() — กลไก
human-in-the-loop จริง (ask_user_func รอ asyncio.Future จนกว่า resolve_approval() จะถูก
เรียก) เทสต์แยกจาก test_api.py เพราะต้องควบคุม concurrency ภายใน event loop *เดียวกัน*
ตรงๆ (สร้าง approval request ค้างไว้พร้อมกับ resolve มันจาก "อีกฝั่ง" ในลูปเดียวกัน) —
pytest-asyncio รันทั้งฟังก์ชันทดสอบในลูปเดียวกันเสมอ ต่างจากการยิงผ่าน TestClient ที่
background task ของ jobจริงรันอยู่คนละ event loop (ดูเหตุผลที่ test_api.py เลือก mock
request_approval() แทนแทนที่จะเล่น queue/future จริงข้าม loop)
"""

import asyncio

import pytest

from backend.app.api.task_manager import TaskManager


def _controllable_coro(finish: asyncio.Event):
    """coro ที่ยัง "ทำงานอยู่" (record.status == "running") จนกว่า finish จะถูก set —
    กันไม่ให้ TaskManager._run() cleanup (ยกเลิก pending approval ทั้งหมด + push
    task_done) แทรกเข้ามากลางเทสต์ก่อนที่เราจะทันได้ resolve approval เอง"""

    async def _coro() -> dict:
        await finish.wait()
        return {"success": True, "steps": 0, "message": "ok", "history": [], "tokens": {}, "plan": None, "final_page_state": ""}

    return _coro()


async def _drain_task_done(record) -> None:
    """ปล่อยให้ background task (จาก submit()) จบแบบสะอาด กัน 'Task was destroyed but it
    is pending' warning ตอน event loop ปิดท้าย test"""
    event = await asyncio.wait_for(record.events.get(), timeout=1)
    assert event["kind"] == "task_done"


@pytest.mark.asyncio
async def test_request_approval_blocks_until_resolved():
    tm = TaskManager()
    finish = asyncio.Event()
    record = tm.submit("t1", "https://example.com", "goal", None, _controllable_coro(finish))

    approval_task = asyncio.create_task(tm.request_approval("t1", {"type": "purchase"}))
    event = await asyncio.wait_for(record.events.get(), timeout=1)
    assert event["kind"] == "approval_request"
    assert event["cmd"] == {"type": "purchase"}
    assert "request_id" in event

    assert tm.resolve_approval("t1", event["request_id"], True) is True
    assert await asyncio.wait_for(approval_task, timeout=1) is True

    finish.set()
    await _drain_task_done(record)


@pytest.mark.asyncio
async def test_request_approval_can_resolve_to_denied():
    tm = TaskManager()
    finish = asyncio.Event()
    record = tm.submit("t2", "https://example.com", "goal", None, _controllable_coro(finish))

    approval_task = asyncio.create_task(tm.request_approval("t2", {"type": "delete", "index": 5}))
    event = await asyncio.wait_for(record.events.get(), timeout=1)

    assert tm.resolve_approval("t2", event["request_id"], False) is True
    assert await asyncio.wait_for(approval_task, timeout=1) is False

    finish.set()
    await _drain_task_done(record)


@pytest.mark.asyncio
async def test_resolve_approval_returns_false_for_unknown_request_id():
    tm = TaskManager()
    finish = asyncio.Event()
    record = tm.submit("t3", "https://example.com", "goal", None, _controllable_coro(finish))

    assert tm.resolve_approval("t3", "does-not-exist", True) is False

    finish.set()
    await _drain_task_done(record)


@pytest.mark.asyncio
async def test_resolve_approval_returns_false_after_already_resolved():
    tm = TaskManager()
    finish = asyncio.Event()
    record = tm.submit("t4", "https://example.com", "goal", None, _controllable_coro(finish))

    approval_task = asyncio.create_task(tm.request_approval("t4", {"type": "pay"}))
    event = await asyncio.wait_for(record.events.get(), timeout=1)

    assert tm.resolve_approval("t4", event["request_id"], False) is True
    assert tm.resolve_approval("t4", event["request_id"], True) is False  # ตอบไปแล้ว
    await asyncio.wait_for(approval_task, timeout=1)

    finish.set()
    await _drain_task_done(record)


def test_resolve_approval_returns_false_for_unknown_task_id():
    tm = TaskManager()
    assert tm.resolve_approval("does-not-exist", "req-1", True) is False


@pytest.mark.asyncio
async def test_push_event_no_op_for_unknown_task_id():
    tm = TaskManager()
    await tm.push_event("does-not-exist", {"kind": "step"})  # ไม่ throw, เงียบๆ


@pytest.mark.asyncio
async def test_task_completion_cancels_pending_approval_as_denied():
    """ถ้า task จบ (สำเร็จ/ล้มเหลว) ระหว่างที่ยังมี approval ค้างอยู่ (เช่น run_task()
    โยน exception ระหว่างรอ human ตอบ) ต้องไม่ปล่อยให้ Future ค้างไปตลอดกาล —
    resolve เป็น False (ปฏิเสธ) ให้ทันทีตอน task จบ (ดู TaskManager._run())"""
    tm = TaskManager()
    finish = asyncio.Event()

    async def _coro() -> dict:
        await finish.wait()
        raise RuntimeError("boom")

    record = tm.submit("t5", "https://example.com", "goal", None, _coro())
    approval_task = asyncio.create_task(tm.request_approval("t5", {"type": "purchase"}))
    await asyncio.wait_for(record.events.get(), timeout=1)  # approval_request

    finish.set()
    done_event = await asyncio.wait_for(record.events.get(), timeout=1)
    assert done_event["kind"] == "task_done"
    assert done_event["status"] == "error"

    assert await asyncio.wait_for(approval_task, timeout=1) is False


@pytest.mark.asyncio
async def test_cancel_stops_a_running_task_and_marks_it_cancelled():
    tm = TaskManager()
    started = asyncio.Event()

    async def _coro() -> dict:
        started.set()
        await asyncio.Event().wait()  # ไม่มีวันจบเอง ต้องถูก cancel() เท่านั้น
        return {}

    record = tm.submit("t6", "https://example.com", "goal", None, _coro())
    await asyncio.wait_for(started.wait(), timeout=1)

    assert tm.cancel("t6") is True

    done_event = await asyncio.wait_for(record.events.get(), timeout=1)
    assert done_event["kind"] == "task_done"
    assert done_event["status"] == "cancelled"
    assert record.status == "cancelled"
    assert record.result is None


@pytest.mark.asyncio
async def test_cancel_resolves_pending_approval_as_denied():
    """หยุด task ระหว่างที่กำลังรอ human ตอบ permission/plan prompt อยู่ — cancel()
    ส่ง CancelledError เข้าไปตรง await point ปัจจุบันทันที (ไม่ต้องรอ event/future อื่น
    ก่อน) และ Future ของ request_approval() ต้องไม่ค้างรอตลอดไป (resolve เป็น False
    เหมือน task จบแบบอื่นๆ)"""
    tm = TaskManager()

    async def _coro() -> dict:
        await asyncio.Event().wait()  # ไม่มีวันจบเอง ต้องถูก cancel() เท่านั้น
        return {}

    record = tm.submit("t7", "https://example.com", "goal", None, _coro())
    approval_task = asyncio.create_task(tm.request_approval("t7", {"type": "purchase"}))
    await asyncio.wait_for(record.events.get(), timeout=1)  # approval_request

    assert tm.cancel("t7") is True

    assert await asyncio.wait_for(approval_task, timeout=1) is False


def test_cancel_returns_false_for_unknown_task_id():
    tm = TaskManager()
    assert tm.cancel("does-not-exist") is False


@pytest.mark.asyncio
async def test_cancel_returns_false_for_already_finished_task():
    tm = TaskManager()

    async def _coro() -> dict:
        return {"success": True}

    record = tm.submit("t8", "https://example.com", "goal", None, _coro())
    await asyncio.wait_for(record.events.get(), timeout=1)  # task_done

    assert tm.cancel("t8") is False


@pytest.mark.asyncio
async def test_request_approval_times_out_and_denies_if_nobody_responds():
    """W10[E]: ไม่มีใคร resolve_approval() เลยภายใน timeout ที่กำหนด — ต้องไม่ค้างรอ
    ตลอดกาล (ซึ่งจะยึด browser จาก pool ไว้ไม่มีวันคืน ดู config.py::approval_timeout_seconds)
    ต้อง auto-deny (False) แทน และเลิกนับเป็น pending request (resolve_approval() ทีหลัง
    ต้องคืน False เพราะหมดอายุไปแล้ว)"""
    tm = TaskManager()
    finish = asyncio.Event()
    record = tm.submit("t9", "https://example.com", "goal", None, _controllable_coro(finish))

    approval_task = asyncio.create_task(tm.request_approval("t9", {"type": "purchase"}, timeout=0.05))
    event = await asyncio.wait_for(record.events.get(), timeout=1)
    assert event["kind"] == "approval_request"
    request_id = event["request_id"]

    assert await asyncio.wait_for(approval_task, timeout=1) is False

    timeout_event = await asyncio.wait_for(record.events.get(), timeout=1)
    assert timeout_event == {"kind": "approval_timeout", "request_id": request_id, "cmd": {"type": "purchase"}}

    # request_id หมดอายุไปแล้ว (ถูก pop ออกจาก record.pending ตอน timeout) — ตอบทีหลังไม่มีผล
    assert tm.resolve_approval("t9", request_id, True) is False

    finish.set()
    await _drain_task_done(record)


@pytest.mark.asyncio
async def test_request_approval_no_timeout_waits_indefinitely():
    """timeout=None (ค่า default) ต้องยังคงพฤติกรรมเดิม — รอจนกว่า resolve_approval()
    จะถูกเรียกจริงๆ ไม่ auto-deny เอง"""
    tm = TaskManager()
    finish = asyncio.Event()
    record = tm.submit("t10", "https://example.com", "goal", None, _controllable_coro(finish))

    approval_task = asyncio.create_task(tm.request_approval("t10", {"type": "purchase"}))
    event = await asyncio.wait_for(record.events.get(), timeout=1)

    # ไม่มีใครตอบสักพัก (จำลองด้วย short sleep) — ต้องยังไม่ resolve เอง
    await asyncio.sleep(0.1)
    assert not approval_task.done()

    assert tm.resolve_approval("t10", event["request_id"], True) is True
    assert await asyncio.wait_for(approval_task, timeout=1) is True

    finish.set()
    await _drain_task_done(record)


@pytest.mark.asyncio
async def test_resolve_approval_with_edited_plan_mutates_the_pending_cmd():
    """W10[F]: edited_plan ต้องแก้ info["cmd"]["plan"] ใน-place ก่อน resolve future —
    orchestrator.py::_confirm_plan() ยังถือ reference ของ cmd dict ก้อนเดิมอยู่ (ส่งเข้า
    ask_user_func ไปแล้วแต่ยังไม่ทิ้ง) พอ future resolve กลับมาต้องอ่านแผนที่แก้แล้วออกไป
    ใช้แทนแผนเดิม ไม่ใช่แผนเดิมที่ AI ร่างไว้"""
    tm = TaskManager()
    finish = asyncio.Event()
    record = tm.submit("t11", "https://example.com", "goal", None, _controllable_coro(finish))

    cmd = {"type": "confirm_plan", "plan": "1. original plan"}
    approval_task = asyncio.create_task(tm.request_approval("t11", cmd))
    await asyncio.wait_for(record.events.get(), timeout=1)  # approval_request

    assert tm.resolve_approval("t11", list(record.pending.keys())[0], True, edited_plan="1. corrected plan") is True
    assert await asyncio.wait_for(approval_task, timeout=1) is True
    # cmd คือ dict ก้อนเดียวกับที่ _confirm_plan() ใน orchestrator.py ถืออยู่ — ต้องเห็นการ
    # แก้ไขสะท้อนกลับมาที่นี่ด้วย (ไม่ใช่แค่ใน record.pending ภายในเท่านั้น)
    assert cmd["plan"] == "1. corrected plan"

    finish.set()
    await _drain_task_done(record)


@pytest.mark.asyncio
async def test_resolve_approval_edited_plan_ignored_for_non_plan_requests():
    """edited_plan ไม่ควรมีผลอะไรกับ permission prompt ทั่วไป (cmd ไม่ใช่ type
    confirm_plan) — กัน misuse ที่อาจแอบเปลี่ยน field "plan" ที่ไม่มีความหมายอะไรสำหรับ
    action ปกติ"""
    tm = TaskManager()
    finish = asyncio.Event()
    record = tm.submit("t12", "https://example.com", "goal", None, _controllable_coro(finish))

    cmd = {"type": "purchase", "index": 3}
    approval_task = asyncio.create_task(tm.request_approval("t12", cmd))
    await asyncio.wait_for(record.events.get(), timeout=1)

    assert tm.resolve_approval("t12", list(record.pending.keys())[0], True, edited_plan="should be ignored") is True
    assert await asyncio.wait_for(approval_task, timeout=1) is True
    assert "plan" not in cmd

    finish.set()
    await _drain_task_done(record)

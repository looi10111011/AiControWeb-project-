"""W10[B]: TaskManager.request_approval()/resolve_approval()/push_event() — กลไก
human-in-the-loop จริง (ask_user_func รอ asyncio.Future จนกว่า resolve_approval() จะถูก
เรียก) เทสต์แยกจาก test_api.py เพราะต้องควบคุม concurrency ภายใน event loop *เดียวกัน*
ตรงๆ (สร้าง approval request ค้างไว้พร้อมกับ resolve มันจาก "อีกฝั่ง" ในลูปเดียวกัน) —
pytest-asyncio รันทั้งฟังก์ชันทดสอบในลูปเดียวกันเสมอ ต่างจากการยิงผ่าน TestClient ที่
background task ของ jobจริงรันอยู่คนละ event loop (ดูเหตุผลที่ test_api.py เลือก mock
request_approval() แทนแทนที่จะเล่น queue/future จริงข้าม loop)

W26 ("Broadcast live events to every subscribed tab"): TaskRecord.events (asyncio.Queue
เดียว) ถูกแทนที่ด้วย TaskRecord.event_subscribers (list ของ Queue หนึ่งอันต่อ SSE
connection ที่เปิดอยู่จริง — ดู task_manager.py::_broadcast()) — เทสต์ในไฟล์นี้ต้อง
"สมัคร" queue ของตัวเองก่อนเสมอ (ดู _subscribe() ด้านล่าง) แทนที่จะอ่านจาก record.events
ตรงๆ เหมือนเดิม
"""

import asyncio
import json

import pytest

from backend.app.api.routes import _stream_task_events
from backend.app.api.task_manager import TaskManager, _log_step_trace, _log_token_usage
from backend.app.config import settings


def _controllable_coro(finish: asyncio.Event):
    """coro ที่ยัง "ทำงานอยู่" (record.status == "running") จนกว่า finish จะถูก set —
    กันไม่ให้ TaskManager._run() cleanup (ยกเลิก pending approval ทั้งหมด + push
    task_done) แทรกเข้ามากลางเทสต์ก่อนที่เราจะทันได้ resolve approval เอง"""

    async def _coro() -> dict:
        await finish.wait()
        return {"success": True, "steps": 0, "message": "ok", "history": [], "tokens": {}, "plan": None, "final_page_state": ""}

    return _coro()


def _subscribe(record) -> asyncio.Queue:
    """W26: จำลอง SSE connection หนึ่งตัวสมัครเป็น subscriber ของ task นี้ — เทียบเท่ากับ
    ส่วนแรกของ routes.py::_stream_task_events() (สร้าง Queue + append เข้า
    record.event_subscribers) แต่ข้าม replay-pending-approval/SSE-string-formatting ไป
    เพราะเทสต์พวกนี้สนใจแค่กลไก push/broadcast ของ TaskManager เอง ไม่ใช่ HTTP layer"""
    queue: asyncio.Queue = asyncio.Queue()
    record.event_subscribers.append(queue)
    return queue


async def _drain_task_done(queue: asyncio.Queue) -> None:
    """ปล่อยให้ background task (จาก submit()) จบแบบสะอาด กัน 'Task was destroyed but it
    is pending' warning ตอน event loop ปิดท้าย test"""
    event = await asyncio.wait_for(queue.get(), timeout=1)
    assert event["kind"] == "task_done"


@pytest.mark.asyncio
async def test_request_approval_blocks_until_resolved():
    tm = TaskManager()
    finish = asyncio.Event()
    record = tm.submit("t1", "https://example.com", "goal", None, _controllable_coro(finish))
    queue = _subscribe(record)

    approval_task = asyncio.create_task(tm.request_approval("t1", {"type": "purchase"}))
    event = await asyncio.wait_for(queue.get(), timeout=1)
    assert event["kind"] == "approval_request"
    assert event["cmd"] == {"type": "purchase"}
    assert "request_id" in event

    assert tm.resolve_approval("t1", event["request_id"], True) is True
    assert await asyncio.wait_for(approval_task, timeout=1) is True

    finish.set()
    await _drain_task_done(queue)


@pytest.mark.asyncio
async def test_request_approval_can_resolve_to_denied():
    tm = TaskManager()
    finish = asyncio.Event()
    record = tm.submit("t2", "https://example.com", "goal", None, _controllable_coro(finish))
    queue = _subscribe(record)

    approval_task = asyncio.create_task(tm.request_approval("t2", {"type": "delete", "index": 5}))
    event = await asyncio.wait_for(queue.get(), timeout=1)

    assert tm.resolve_approval("t2", event["request_id"], False) is True
    assert await asyncio.wait_for(approval_task, timeout=1) is False

    finish.set()
    await _drain_task_done(queue)


@pytest.mark.asyncio
async def test_resolve_approval_returns_false_for_unknown_request_id():
    tm = TaskManager()
    finish = asyncio.Event()
    record = tm.submit("t3", "https://example.com", "goal", None, _controllable_coro(finish))
    queue = _subscribe(record)

    assert tm.resolve_approval("t3", "does-not-exist", True) is False

    finish.set()
    await _drain_task_done(queue)


@pytest.mark.asyncio
async def test_resolve_approval_returns_false_after_already_resolved():
    tm = TaskManager()
    finish = asyncio.Event()
    record = tm.submit("t4", "https://example.com", "goal", None, _controllable_coro(finish))
    queue = _subscribe(record)

    approval_task = asyncio.create_task(tm.request_approval("t4", {"type": "pay"}))
    event = await asyncio.wait_for(queue.get(), timeout=1)

    assert tm.resolve_approval("t4", event["request_id"], False) is True
    assert tm.resolve_approval("t4", event["request_id"], True) is False  # ตอบไปแล้ว
    await asyncio.wait_for(approval_task, timeout=1)

    finish.set()
    await _drain_task_done(queue)


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
    queue = _subscribe(record)
    approval_task = asyncio.create_task(tm.request_approval("t5", {"type": "purchase"}))
    await asyncio.wait_for(queue.get(), timeout=1)  # approval_request

    finish.set()
    done_event = await asyncio.wait_for(queue.get(), timeout=1)
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
    queue = _subscribe(record)
    await asyncio.wait_for(started.wait(), timeout=1)

    assert await tm.cancel("t6") is True

    done_event = await asyncio.wait_for(queue.get(), timeout=1)
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
    queue = _subscribe(record)
    approval_task = asyncio.create_task(tm.request_approval("t7", {"type": "purchase"}))
    await asyncio.wait_for(queue.get(), timeout=1)  # approval_request

    assert await tm.cancel("t7") is True

    assert await asyncio.wait_for(approval_task, timeout=1) is False


@pytest.mark.asyncio
async def test_cancel_waits_for_the_task_to_actually_stop_before_returning():
    """W27: แก้บั๊ก "ปุ่ม kill session ใช้งานจริงไม่ได้" — เดิม cancel() แค่ยิง
    CancelledError แล้ว return True ทันที ไม่รอให้ _run() ประมวลผลเสร็จก่อน (race กับ
    SessionRegistry.close() ที่ frontend เรียกตามมาทันทีหลัง stop ตอบกลับ — ดู
    routes.py::stop_task()/killSession() ใน index.html) — พิสูจน์ด้วยการเช็คว่า
    record.status เป็น "cancelled" ไปแล้วทันทีที่ cancel() คืนค่า โดยไม่ต้องรอ event อื่น
    เพิ่มก่อนเลย (ต่างจาก test_cancel_stops_a_running_task_and_marks_it_cancelled เดิมที่
    รอ event เพิ่มก่อนเช็ค — พิสูจน์ได้แค่ "sequence ถูกต้องในที่สุด" ไม่ได้พิสูจน์ timing ว่า
    cancel() เองรอจริงไหม)"""
    tm = TaskManager()
    started = asyncio.Event()

    async def _coro() -> dict:
        started.set()
        await asyncio.Event().wait()  # ไม่มีวันจบเอง ต้องถูก cancel() เท่านั้น
        return {}

    record = tm.submit("t9", "https://example.com", "goal", None, _coro())
    await asyncio.wait_for(started.wait(), timeout=1)

    assert await tm.cancel("t9") is True
    assert record.status == "cancelled"  # อัปเดตไปแล้วจริงๆ ก่อน cancel() จะคืนค่าด้วยซ้ำ


@pytest.mark.asyncio
async def test_cancel_returns_false_for_unknown_task_id():
    tm = TaskManager()
    assert await tm.cancel("does-not-exist") is False


@pytest.mark.asyncio
async def test_cancel_returns_false_for_already_finished_task():
    tm = TaskManager()

    async def _coro() -> dict:
        return {"success": True}

    record = tm.submit("t8", "https://example.com", "goal", None, _coro())
    queue = _subscribe(record)
    await asyncio.wait_for(queue.get(), timeout=1)  # task_done

    assert await tm.cancel("t8") is False


@pytest.mark.asyncio
async def test_request_approval_times_out_and_denies_if_nobody_responds():
    """W10[E]: ไม่มีใคร resolve_approval() เลยภายใน timeout ที่กำหนด — ต้องไม่ค้างรอ
    ตลอดกาล (ซึ่งจะยึด browser จาก pool ไว้ไม่มีวันคืน ดู config.py::approval_timeout_seconds)
    ต้อง auto-deny (False) แทน และเลิกนับเป็น pending request (resolve_approval() ทีหลัง
    ต้องคืน False เพราะหมดอายุไปแล้ว)"""
    tm = TaskManager()
    finish = asyncio.Event()
    record = tm.submit("t9", "https://example.com", "goal", None, _controllable_coro(finish))
    queue = _subscribe(record)

    approval_task = asyncio.create_task(tm.request_approval("t9", {"type": "purchase"}, timeout=0.05))
    event = await asyncio.wait_for(queue.get(), timeout=1)
    assert event["kind"] == "approval_request"
    request_id = event["request_id"]

    assert await asyncio.wait_for(approval_task, timeout=1) is False

    timeout_event = await asyncio.wait_for(queue.get(), timeout=1)
    assert timeout_event == {"kind": "approval_timeout", "request_id": request_id, "cmd": {"type": "purchase"}}

    # request_id หมดอายุไปแล้ว (ถูก pop ออกจาก record.pending ตอน timeout) — ตอบทีหลังไม่มีผล
    assert tm.resolve_approval("t9", request_id, True) is False

    finish.set()
    await _drain_task_done(queue)


@pytest.mark.asyncio
async def test_request_approval_no_timeout_waits_indefinitely():
    """timeout=None (ค่า default) ต้องยังคงพฤติกรรมเดิม — รอจนกว่า resolve_approval()
    จะถูกเรียกจริงๆ ไม่ auto-deny เอง"""
    tm = TaskManager()
    finish = asyncio.Event()
    record = tm.submit("t10", "https://example.com", "goal", None, _controllable_coro(finish))
    queue = _subscribe(record)

    approval_task = asyncio.create_task(tm.request_approval("t10", {"type": "purchase"}))
    event = await asyncio.wait_for(queue.get(), timeout=1)

    # ไม่มีใครตอบสักพัก (จำลองด้วย short sleep) — ต้องยังไม่ resolve เอง
    await asyncio.sleep(0.1)
    assert not approval_task.done()

    assert tm.resolve_approval("t10", event["request_id"], True) is True
    assert await asyncio.wait_for(approval_task, timeout=1) is True

    finish.set()
    await _drain_task_done(queue)


@pytest.mark.asyncio
async def test_resolve_approval_with_edited_plan_mutates_the_pending_cmd():
    """W10[F]: edited_plan ต้องแก้ info["cmd"]["plan"] ใน-place ก่อน resolve future —
    orchestrator.py::_confirm_plan() ยังถือ reference ของ cmd dict ก้อนเดิมอยู่ (ส่งเข้า
    ask_user_func ไปแล้วแต่ยังไม่ทิ้ง) พอ future resolve กลับมาต้องอ่านแผนที่แก้แล้วออกไป
    ใช้แทนแผนเดิม ไม่ใช่แผนเดิมที่ AI ร่างไว้"""
    tm = TaskManager()
    finish = asyncio.Event()
    record = tm.submit("t11", "https://example.com", "goal", None, _controllable_coro(finish))
    queue = _subscribe(record)

    cmd = {"type": "confirm_plan", "plan": "1. original plan"}
    approval_task = asyncio.create_task(tm.request_approval("t11", cmd))
    await asyncio.wait_for(queue.get(), timeout=1)  # approval_request

    assert tm.resolve_approval("t11", list(record.pending.keys())[0], True, edited_plan="1. corrected plan") is True
    assert await asyncio.wait_for(approval_task, timeout=1) is True
    # cmd คือ dict ก้อนเดียวกับที่ _confirm_plan() ใน orchestrator.py ถืออยู่ — ต้องเห็นการ
    # แก้ไขสะท้อนกลับมาที่นี่ด้วย (ไม่ใช่แค่ใน record.pending ภายในเท่านั้น)
    assert cmd["plan"] == "1. corrected plan"

    finish.set()
    await _drain_task_done(queue)


@pytest.mark.asyncio
async def test_resolve_approval_edited_plan_ignored_for_non_plan_requests():
    """edited_plan ไม่ควรมีผลอะไรกับ permission prompt ทั่วไป (cmd ไม่ใช่ type
    confirm_plan) — กัน misuse ที่อาจแอบเปลี่ยน field "plan" ที่ไม่มีความหมายอะไรสำหรับ
    action ปกติ"""
    tm = TaskManager()
    finish = asyncio.Event()
    record = tm.submit("t12", "https://example.com", "goal", None, _controllable_coro(finish))
    queue = _subscribe(record)

    cmd = {"type": "purchase", "index": 3}
    approval_task = asyncio.create_task(tm.request_approval("t12", cmd))
    await asyncio.wait_for(queue.get(), timeout=1)

    assert tm.resolve_approval("t12", list(record.pending.keys())[0], True, edited_plan="should be ignored") is True
    assert await asyncio.wait_for(approval_task, timeout=1) is True
    assert "plan" not in cmd


# ---------------- W_resume ("Mid-Task Input Request") ----------------
# บั๊กจริงที่ user รายงาน: agent ขอรหัสผ่านใหม่กลางทางแล้ว finish_task(false) จบ task
# ทั้งหมด ทำให้เทิร์นถัดไปที่ user ตอบค่ามาต้องเริ่มงานใหม่จากศูนย์ — answer_text mirrors
# edited_plan ข้างบนทุกประการ แค่คนละ cmd type/key


@pytest.mark.asyncio
async def test_resolve_approval_with_answer_text_mutates_the_pending_cmd():
    """answer_text ต้องแก้ info["cmd"]["answer"] ใน-place ก่อน resolve future —
    orchestrator.py::_request_user_input() ยังถือ reference ของ cmd dict ก้อนเดิมอยู่ พอ
    future resolve กลับมาต้องอ่านคำตอบที่แก้แล้วออกไปทำ task เดิมต่อทันที"""
    tm = TaskManager()
    finish = asyncio.Event()
    record = tm.submit("t13", "https://example.com", "goal", None, _controllable_coro(finish))
    queue = _subscribe(record)

    cmd = {"type": "request_user_input", "prompt": "รหัสผ่านใหม่คืออะไร?", "sensitive": True}
    approval_task = asyncio.create_task(tm.request_approval("t13", cmd))
    await asyncio.wait_for(queue.get(), timeout=1)  # approval_request

    assert tm.resolve_approval(
        "t13", list(record.pending.keys())[0], True, answer_text="Sup3rSecret!",
    ) is True
    assert await asyncio.wait_for(approval_task, timeout=1) is True
    # cmd คือ dict ก้อนเดียวกับที่ orchestrator.py ถืออยู่ — ต้องเห็นการแก้ไขสะท้อนกลับมาที่
    # นี่ด้วย (ไม่ใช่แค่ใน record.pending ภายในเท่านั้น)
    assert cmd["answer"] == "Sup3rSecret!"

    finish.set()
    await _drain_task_done(queue)


@pytest.mark.asyncio
async def test_resolve_approval_answer_text_ignored_for_non_input_requests():
    """answer_text ไม่ควรมีผลอะไรกับ permission prompt ทั่วไป/confirm_plan (cmd ไม่ใช่ type
    request_user_input) — กัน misuse ที่อาจแอบเติม key "answer" ที่ไม่มีความหมายอะไรสำหรับ
    action ปกติ"""
    tm = TaskManager()
    finish = asyncio.Event()
    record = tm.submit("t14", "https://example.com", "goal", None, _controllable_coro(finish))
    queue = _subscribe(record)

    cmd = {"type": "purchase", "index": 3}
    approval_task = asyncio.create_task(tm.request_approval("t14", cmd))
    await asyncio.wait_for(queue.get(), timeout=1)

    assert tm.resolve_approval(
        "t14", list(record.pending.keys())[0], True, answer_text="should be ignored",
    ) is True
    assert await asyncio.wait_for(approval_task, timeout=1) is True
    assert "answer" not in cmd

    finish.set()
    await _drain_task_done(queue)


# ---------------- W26 ("Broadcast live events to every subscribed tab") ----------------
# บั๊กจริงที่ user รายงาน (ยังเจออยู่แม้หลัง W25): approval prompt/log ไม่ขึ้นสด ต้องกด F5
# — root cause แท้จริง: frontend (index.html::ensureConversationFor) ตั้งใจให้ "ทุกแท็บที่
# เปิดค้างไว้เห็น task จากแท็บอื่นด้วย" (GET /tasks คืน task ทั้งระบบ ไม่กรองเฉพาะแท็บที่
# สร้าง แล้ว refreshTasks() เปิด SSE ให้ทุก task ที่ "running" โดยอัตโนมัติทุกแท็บ) ทำให้
# หลายแท็บ subscribe task เดียวกันพร้อมกันเป็นเรื่องปกติ ไม่ใช่ edge case หายาก — W25 เดิม
# แก้ด้วย "connection ล่าสุดชนะเสมอ" (สมมติว่ามีผู้ชมสดแค่คนเดียว) กลับกลายเป็นทำให้แท็บที่
# ผู้ใช้กำลังดูอยู่จริงถูกแท็บอื่นแย่ง event ไปเงียบๆ แทน (ยืนยันจากการทดสอบสด) — เทสต์ด้านล่าง
# พิสูจน์ว่าตอนนี้ "ทุก connection ที่เปิดอยู่จริงได้รับ event ครบทุกตัวพร้อมกัน" (broadcast
# จริง) แทนที่ test เดิมของ W25 ที่พิสูจน์พฤติกรรม "ล่าสุดชนะ" ซึ่งตอนนี้ถือเป็นพฤติกรรมที่
# ต้องการแก้ทิ้งแล้ว ไม่ใช่ของที่ต้องรักษาไว้อีกต่อไป


@pytest.mark.asyncio
async def test_stream_task_events_delivers_to_a_single_connection_as_before():
    """พฤติกรรมพื้นฐานที่ต้องยังทำงานถูกต้องเหมือนเดิม (ไม่ใช่แค่เคสหลายแท็บ) — connection
    เดียวต้องได้รับทุก event ที่ push เข้ามาหลังจากมันเปิดจริง"""
    tm = TaskManager()
    finish = asyncio.Event()
    record = tm.submit("s1", "https://example.com", "goal", None, _controllable_coro(finish))

    chunks: list = []

    async def _drain():
        # W26: ไม่ break เร็วเมื่อเจอ task_done — ปล่อยให้ async for วนต่ออีกรอบจน generator
        # คืน StopAsyncIteration เอง (_stream_task_events() break+return ภายในตัวมันเอง
        # หลัง task_done อยู่แล้ว) ให้แน่ใจว่า finally block ของ generator (ถอนตัวเองออกจาก
        # record.event_subscribers) รันจริงทันทีแบบ synchronous กับ iteration นี้ — break
        # จากฝั่ง consumer เองจะไม่ trigger aclose() ทันที (รอ GC/async-generator finalizer
        # ของ event loop แทน ซึ่ง timing ไม่แน่นอน)
        async for chunk in _stream_task_events(record):
            chunks.append(chunk)

    task = asyncio.create_task(_drain())
    await asyncio.sleep(0)  # ให้ generator รันจนถึง await queue.get() (สมัคร subscriber เสร็จแล้ว)

    await tm.push_event("s1", {"kind": "step", "step": 1, "cmd": {}, "result": "ok", "success": True})
    finish.set()
    await asyncio.wait_for(task, timeout=1)

    assert any('"kind": "step"' in c for c in chunks)
    assert any('"kind": "task_done"' in c for c in chunks)


@pytest.mark.asyncio
async def test_stream_task_events_broadcasts_to_every_overlapping_connection():
    """W26 (บั๊กหลักที่ user รายงาน): 2 แท็บเปิดดู task เดียวกันพร้อมกันจริง (สถานการณ์ปกติ
    ของแอปนี้ ไม่ใช่ edge case) — event ที่ push เข้ามาหลังจากนั้นต้องไปถึง *ทั้งสอง*
    connection ครบถ้วน ไม่มีตัวไหนถูกทิ้งไว้ข้างหลังเงียบๆ (ต่างจากพฤติกรรม W25 เดิมที่มีแค่
    connection ล่าสุดเท่านั้นที่ได้รับ)"""
    tm = TaskManager()
    finish = asyncio.Event()
    record = tm.submit("s2", "https://example.com", "goal", None, _controllable_coro(finish))

    tab_a_chunks: list = []
    tab_b_chunks: list = []

    async def _drain(out: list, gen):
        # W26: ไม่ break เร็วเมื่อเจอ task_done — ให้ generator คืน StopAsyncIteration เอง
        # (ดูเหตุผลเต็มในเทสต์ก่อนหน้า) กัน finally block (ถอนตัวเองออกจาก
        # record.event_subscribers) รันช้า/ไม่แน่นอน
        async for chunk in gen:
            out.append(chunk)

    tab_a_task = asyncio.create_task(_drain(tab_a_chunks, _stream_task_events(record)))
    await asyncio.sleep(0)  # แท็บ A สมัคร subscriber ของตัวเองเสร็จแล้ว

    tab_b_task = asyncio.create_task(_drain(tab_b_chunks, _stream_task_events(record)))
    await asyncio.sleep(0)  # แท็บ B สมัคร subscriber ของตัวเองเพิ่มเข้ามาด้วย (ไม่แทนที่แท็บ A)

    assert len(record.event_subscribers) == 2

    await tm.push_event("s2", {"kind": "step", "step": 1, "cmd": {}, "result": "ok", "success": True})
    await asyncio.sleep(0.05)

    assert any('"kind": "step"' in c for c in tab_a_chunks)
    assert any('"kind": "step"' in c for c in tab_b_chunks)  # ทั้งสองแท็บได้รับเหมือนกัน

    finish.set()
    await asyncio.wait_for(tab_a_task, timeout=1)
    await asyncio.wait_for(tab_b_task, timeout=1)

    # ทั้งสอง subscriber ต้องถอนตัวเองออกเรียบร้อยหลัง task_done (ดู _stream_task_events()
    # finally block) ไม่เหลือค้างให้ _broadcast() ป้อน event เข้าไปเปล่าๆ ต่อไปอีก
    assert record.event_subscribers == []


@pytest.mark.asyncio
async def test_stream_task_events_one_connection_disconnecting_does_not_affect_the_other():
    """แท็บหนึ่งปิด connection กลางคัน (จำลอง client disconnect ด้วย .cancel()) ต้องไม่
    กระทบแท็บที่ยังเปิดอยู่เลย — ยังคงได้รับ event ถัดไปตามปกติ"""
    tm = TaskManager()
    finish = asyncio.Event()
    record = tm.submit("s4", "https://example.com", "goal", None, _controllable_coro(finish))

    tab_b_chunks: list = []

    async def _drain(out: list, gen):
        # W26: ไม่ break เร็วเมื่อเจอ task_done — ให้ generator คืน StopAsyncIteration เอง
        # (ดูเหตุผลเต็มในเทสต์ก่อนหน้า) กัน finally block (ถอนตัวเองออกจาก
        # record.event_subscribers) รันช้า/ไม่แน่นอน
        async for chunk in gen:
            out.append(chunk)

    tab_a_task = asyncio.create_task(_drain([], _stream_task_events(record)))
    await asyncio.sleep(0)
    tab_b_task = asyncio.create_task(_drain(tab_b_chunks, _stream_task_events(record)))
    await asyncio.sleep(0)
    assert len(record.event_subscribers) == 2

    # จำลองแท็บ A ปิด connection กลางคัน (ASGI server จะ cancel generator ตอน client หลุด)
    tab_a_task.cancel()
    try:
        await tab_a_task
    except asyncio.CancelledError:
        pass
    await asyncio.sleep(0)
    assert len(record.event_subscribers) == 1  # แท็บ A ถอนตัวเองออกแล้วผ่าน finally block

    await tm.push_event("s4", {"kind": "step", "step": 1, "cmd": {}, "result": "ok", "success": True})
    finish.set()
    await asyncio.wait_for(tab_b_task, timeout=1)

    assert any('"kind": "step"' in c for c in tab_b_chunks)
    assert any('"kind": "task_done"' in c for c in tab_b_chunks)


@pytest.mark.asyncio
async def test_stream_task_events_replays_delivered_pending_approval_on_reconnect():
    """W10[B] (ไม่กระทบจาก W26): reconnect ระหว่างที่มี approval ค้างรออยู่ (delivered=True
    จากการส่งออกไปให้ subscriber ตัวเก่าไปแล้วอย่างน้อยหนึ่งครั้ง) ต้อง replay ให้ connection
    ใหม่เห็นทันที ไม่ต้องรอ event ใหม่จาก queue"""
    tm = TaskManager()
    finish = asyncio.Event()
    record = tm.submit("s3", "https://example.com", "goal", None, _controllable_coro(finish))
    queue = _subscribe(record)  # จำลอง subscriber ตัวเก่าที่เคยได้รับ event นี้ไปแล้ว

    approval_task = asyncio.create_task(tm.request_approval("s3", {"type": "delete", "index": 1}))
    await asyncio.wait_for(queue.get(), timeout=1)  # "ส่งออกไปแล้ว" ให้ subscriber ตัวเก่า (จำลอง)
    request_id = next(iter(record.pending))
    record.pending[request_id]["delivered"] = True

    chunks: list = []
    async for chunk in _stream_task_events(record):
        chunks.append(chunk)
        break  # แค่ chunk แรก (replay) พอสำหรับเทสต์นี้

    assert any('"kind": "approval_request"' in c for c in chunks)

    tm.resolve_approval("s3", request_id, True)
    await asyncio.wait_for(approval_task, timeout=1)
    finish.set()
    await _drain_task_done(queue)


# ---------------- W_step_trace: per-step trace + log ทุกเส้นทาง ----------------


class _FakeTaskRecord:
    """พอสำหรับ _log_token_usage/_log_step_trace ซึ่งอ่านแค่ attribute ไม่กี่ตัว"""

    def __init__(self, result, status="done", error=None):
        self.task_id = "t-1"
        self.url = "https://example.com"
        self.goal = "goal"
        self.provider = "openai"
        self.status = status
        self.error = error
        self.created_at = 0.0
        self.result = result


def test_log_step_trace_writes_one_line_per_step(tmp_path, monkeypatch):
    trace = tmp_path / "step_trace.jsonl"
    monkeypatch.setattr(settings, "step_trace_log_path", str(trace))
    record = _FakeTaskRecord({
        "history": [
            {"step": 1, "cmd": {"type": "click", "index": 2}, "label": "Login",
             "success": True, "failure_class": "ok",
             "timing": {"snapshot": 0.5, "llm": 3.2, "action": 1.1}, "result": "[OK] click(2)"},
            {"step": 2, "cmd": {"type": "fill", "index": 3}, "label": "User",
             "success": False, "failure_class": "element_not_found",
             "timing": {"snapshot": 0.4, "llm": 2.9, "action": 9.0}, "result": "[FAIL] element not found"},
        ],
    })

    _log_step_trace(record)

    lines = [json.loads(l) for l in trace.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 2
    assert lines[0]["action_type"] == "click"
    assert lines[0]["failure_class"] == "ok"
    assert lines[1]["failure_class"] == "element_not_found"
    assert lines[1]["timing"]["action"] == 9.0


def test_log_step_trace_does_nothing_without_history(tmp_path, monkeypatch):
    trace = tmp_path / "step_trace.jsonl"
    monkeypatch.setattr(settings, "step_trace_log_path", str(trace))

    _log_step_trace(_FakeTaskRecord(None, status="error", error="boom"))

    assert not trace.exists()


def test_log_step_trace_truncates_very_long_result_text(tmp_path, monkeypatch):
    trace = tmp_path / "step_trace.jsonl"
    monkeypatch.setattr(settings, "step_trace_log_path", str(trace))
    record = _FakeTaskRecord({
        "history": [{"step": 1, "cmd": {"type": "read_page_data"}, "success": True,
                     "result": "x" * 5000}],
    })

    _log_step_trace(record)

    entry = json.loads(trace.read_text(encoding="utf-8").splitlines()[0])
    assert len(entry["result"]) == 300


def test_log_token_usage_records_failed_and_cancelled_tasks_too(tmp_path, monkeypatch):
    """W_step_trace: เดิมเรียกเฉพาะเส้นทางสำเร็จ — task ที่ crash/ถูก Stop ไม่เคยถูกบันทึกเลย
    ทำให้ success rate ที่คำนวณจากไฟล์นี้เป็นเพดานบน ไม่ใช่ค่าจริง"""
    usage = tmp_path / "token_usage.jsonl"
    monkeypatch.setattr(settings, "token_usage_log_path", str(usage))

    _log_token_usage(_FakeTaskRecord(None, status="cancelled", error="หยุดโดยผู้ใช้ (Stop)"))

    entry = json.loads(usage.read_text(encoding="utf-8").splitlines()[0])
    assert entry["status"] == "cancelled"
    assert entry["success"] is False
    assert entry["tokens"] == {"input": 0, "output": 0, "cache_read": 0, "cache_creation": 0}

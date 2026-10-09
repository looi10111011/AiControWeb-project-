"""W10[A]: in-memory registry ของ task ที่ยิงผ่าน API — POST /tasks คืน task_id ทันที (202)
แล้ว poll/stream ทีหลัง เพราะ run_task() ใช้เวลาเป็นนาที; state หายเมื่อ restart (ยอมรับได้)
self._running เก็บ strong reference ของ asyncio.Task กัน GC เก็บกลางคัน

W10[B]: event subscribers + pending approval futures ผูก SSE (GET /tasks/{id}/stream) กับ
POST /tasks/{id}/respond ให้ human-in-the-loop เป็นปุ่มจริงบนหน้าเว็บ
W10[C]: TaskRecord.asyncio_task ให้ cancel() หา task จาก task_id ได้ (ปุ่ม Stop)
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Coroutine, Optional

from backend.app.core.telemetry import write_step_trace, write_token_usage


@dataclass
class TaskRecord:
    task_id: str
    url: str
    goal: str
    provider: Optional[str]
    status: str  # "running" | "done" | "error" | "cancelled"
    # W_live: headless ที่ resolve แล้ว (caller resolve ก่อน submit()) — frontend โชว์ live view เฉพาะ True
    headless: bool = True
    created_at: float = field(default_factory=time.time)
    result: Optional[dict] = None
    error: Optional[str] = None
    # W20 (Task4): ชื่อไฟล์แนบของเทิร์นนี้ (ไม่เก็บ base64) ให้ GET /tasks echo กลับหลัง page reload
    attached_file_name: Optional[str] = None
    # W10[C]: ตั้งใน submit() ทันทีหลังสร้าง ไม่มีทาง None ตอน "running"
    asyncio_task: Optional[asyncio.Task] = None
    # W26: บั๊กจริง approval prompt/log ไม่ขึ้นสดต้องกด F5 — เดิม Queue เดียวต่อ task แต่หลายแท็บ
    # subscribe task เดียวกันเป็นเรื่องปกติ (W_multitab) จึงเป็น pub-sub: Queue ต่อ SSE connection
    # (สมัคร/ถอนใน routes.py::_stream_task_events())
    event_subscribers: list = field(default_factory=list)
    # W10[B]: request_id -> {"future", "cmd", "delivered"} — เก็บ cmd ไว้ replay ให้ SSE connection
    # ใหม่ถ้าแท็บหลุดระหว่างรอ; "delivered" กัน double-delivery (replay เฉพาะที่เคยส่งแล้ว)
    pending: dict = field(default_factory=dict)


async def _broadcast(record: TaskRecord, event: dict) -> None:
    """W26: ส่ง event เข้า Queue ของทุก subscriber — copy list กัน "changed size during iteration"
    ถ้ามี subscriber ใหม่สมัครแทรก (ไม่จำเป็นต้องเห็น event รอบนี้)"""
    for queue in list(record.event_subscribers):
        await queue.put(event)


def _log_token_usage(record: TaskRecord) -> None:
    """W49: 1 บรรทัด JSON ต่อ task — W_step_trace: เรียกทุก exit path (crash/Stop ด้วย ไม่งั้น
    success rate เป็นแค่เพดานบน) W_eval_trace: writer จริงอยู่ core/telemetry.py (source="api")
    ห้าม throw — telemetry จับ Exception เองและผู้เรียกห่อ try/except อีกชั้น"""
    write_token_usage(
        task_id=record.task_id,
        url=record.url,
        goal=record.goal,
        provider=record.provider,
        result=record.result,
        status=record.status,
        error=record.error,
        duration_seconds=time.time() - record.created_at,
    )


def _log_step_trace(record: TaskRecord) -> None:
    """W_step_trace: 1 บรรทัดต่อ step เขียนครั้งเดียวตอน task จบ (ไม่แทรก disk I/O กลาง loop)
    W_eval_trace: writer อยู่ core/telemetry.py — ห้าม throw เหมือน _log_token_usage()"""
    write_step_trace(
        (record.result or {}).get("history"),
        task_id=record.task_id,
        provider=record.provider,
    )


class TaskManager:
    def __init__(self) -> None:
        self._tasks: dict[str, TaskRecord] = {}
        self._running: set[asyncio.Task] = set()

    def get(self, task_id: str) -> Optional[TaskRecord]:
        return self._tasks.get(task_id)

    def list(self) -> list[TaskRecord]:
        return sorted(self._tasks.values(), key=lambda t: t.created_at, reverse=True)

    def new_task_id(self) -> str:
        """routes.py ต้องรู้ task_id ก่อน submit() เพื่อประกอบ ask_user_func/on_event closure"""
        return str(uuid.uuid4())

    def submit(
        self, task_id: str, url: str, goal: str, provider: Optional[str],
        coro: Coroutine[Any, Any, dict], headless: bool = True,
        attached_file_name: Optional[str] = None,
    ) -> TaskRecord:
        """สร้าง TaskRecord "running" แล้วรัน coro เป็น background (ไม่ await) คืน record ทันที
        ให้ endpoint ตอบ 202 — headless ต้องเป็นค่าที่ caller resolve แล้ว (ไม่ใช่ None);
        attached_file_name (W20) เก็บไว้ echo ใน GET /tasks เท่านั้น"""
        record = TaskRecord(
            task_id=task_id, url=url, goal=goal, provider=provider, status="running",
            headless=headless, attached_file_name=attached_file_name,
        )
        self._tasks[task_id] = record
        task = asyncio.create_task(self._run(record, coro))
        record.asyncio_task = task
        self._running.add(task)
        task.add_done_callback(self._running.discard)
        return record

    async def _run(self, record: TaskRecord, coro: Coroutine[Any, Any, dict]) -> None:
        try:
            record.result = await coro
            record.status = "done"
        except asyncio.CancelledError:
            # W10[C]: จาก cancel() (ปุ่ม Stop) — ไม่ re-raise เพราะเป็น fire-and-forget ไม่มีใคร await;
            # ต้องอัพเดตสถานะ + push task_done ไม่งั้นค้าง "running" (CancelledError ไม่ใช่ Exception)
            record.status = "cancelled"
            record.error = "หยุดโดยผู้ใช้ (Stop)"
        except Exception as e:
            record.error = str(e)
            record.status = "error"
        # W_step_trace: log นอก try ให้ครอบทุกเส้นทาง (done/cancelled/error) — path error ส่วนใหญ่มี
        # result dict เพราะ run_task จับ exception ในลูปเอง (W_loop_crash)
        try:
            await asyncio.to_thread(_log_token_usage, record)
            await asyncio.to_thread(_log_step_trace, record)
        except Exception:
            pass
        # W10[B]: ปลด pending approval ที่ค้าง กัน Future ค้างตลอดกาล
        for info in record.pending.values():
            if not info["future"].done():
                info["future"].set_result(False)
        # sentinel ที่บอก SSE consumer ว่า stream จบแล้ว
        await _broadcast(record, {
            "kind": "task_done", "status": record.status,
            "result": record.result, "error": record.error,
        })

    async def cancel(self, task_id: str, wait_timeout: float = 5.0) -> bool:
        """POST /tasks/{id}/stop — คืน False ถ้า task ไม่ได้ "running" (endpoint คืน 409)

        W27: บั๊ก "ปุ่ม kill session ใช้ไม่ได้" — เดิมคืน True ทันทีหลัง .cancel() แล้ว frontend
        close session ต่อทันที race กับ task ที่ยังแตะ page อยู่; ตอนนี้ await task (มี timeout) ก่อน
        return ไม่ throw แม้ timeout (_run() ไม่ re-raise CancelledError)"""
        record = self._tasks.get(task_id)
        if record is None or record.status != "running" or record.asyncio_task is None:
            return False
        task = record.asyncio_task
        task.cancel()
        try:
            await asyncio.wait_for(task, timeout=wait_timeout)
        except asyncio.TimeoutError:
            pass
        return True

    async def push_event(self, task_id: str, event: dict) -> None:
        record = self._tasks.get(task_id)
        if record is not None:
            await _broadcast(record, event)

    async def request_approval(self, task_id: str, cmd: dict, timeout: Optional[float] = None) -> bool:
        """push "approval_request" แล้วรอ resolve_approval() (POST /tasks/{id}/respond) — คืน False
        ถ้าไม่มี record (fail closed)

        W10[E]: timeout (วินาที) หมดเวลา = ปฏิเสธอัตโนมัติ — task ที่ค้างรอ human ยังยึด browser
        จาก pool อยู่ ไม่มี timeout จะกัด browser_pool_size จน task ใหม่รอคิวไม่รู้จบ"""
        record = self._tasks.get(task_id)
        if record is None:
            return False
        request_id = str(uuid.uuid4())
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        record.pending[request_id] = {"future": future, "cmd": cmd, "delivered": False}
        await _broadcast(record, {"kind": "approval_request", "request_id": request_id, "cmd": cmd})
        try:
            if timeout is None:
                return await future
            try:
                return await asyncio.wait_for(future, timeout=timeout)
            except asyncio.TimeoutError:
                await _broadcast(record, {
                    "kind": "approval_timeout", "request_id": request_id, "cmd": cmd,
                })
                return False
        finally:
            record.pending.pop(request_id, None)

    def resolve_approval(
        self, task_id: str, request_id: str, approved: bool,
        edited_plan: Optional[str] = None, answer_text: Optional[str] = None,
    ) -> bool:
        """POST /tasks/{id}/respond — คืน False ถ้าไม่พบ request_id (endpoint คืน 404)

        W10[F] edited_plan / W_resume answer_text: mutate cmd dict เดิม (ไม่สร้างใหม่) เพราะ
        orchestrator (_confirm_plan()/_request_user_input()) ถือ reference เดียวกันและอ่าน
        cmd["plan"]/cmd["answer"] หลัง future resolve"""
        record = self._tasks.get(task_id)
        if record is None:
            return False
        info = record.pending.get(request_id)
        if info is None or info["future"].done():
            return False
        if edited_plan is not None and info["cmd"].get("type") == "confirm_plan":
            info["cmd"]["plan"] = edited_plan
        if answer_text is not None and info["cmd"].get("type") == "request_user_input":
            info["cmd"]["answer"] = answer_text
        info["future"].set_result(approved)
        return True

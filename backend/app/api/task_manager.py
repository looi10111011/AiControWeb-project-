"""W10[A]: registry ของ task ที่ยิงผ่าน API — Orchestrator.run_task() ใช้เวลาเป็นนาที
(หลาย step, เรียก LLM จริงทุก step) ถ้าให้ POST /tasks รอ await จนจบตรงๆ จะ block
request ค้างนานเกินไป (และ client ส่วนใหญ่มี HTTP timeout สั้นกว่านั้น) — เลยแยกเป็น
"submit แล้วคืน task_id ทันที (202) + poll สถานะทีหลังผ่าน GET /tasks/{id}" แทน เหมือน
pattern มาตรฐานของงานที่ใช้เวลานาน (job queue)

เก็บ state ใน memory ล้วนๆ (ไม่มี DB) — ตกเมื่อ process restart ได้ ยอมรับได้สำหรับ W10[A]
(ยังไม่มี requirement เรื่อง persistence ข้าม restart ใน roadmap) เก็บ reference ของ
asyncio.Task ไว้ใน self._running ด้วย (ไม่ใช่แค่ fire-and-forget) กัน task โดน garbage
collect กลางคันซึ่งเป็นปัญหาที่รู้จักกันดีของ asyncio.create_task() (ดู docs: "Task ที่ไม่
มี strong reference เก็บไว้อาจโดน GC เก็บก่อนรันเสร็จ")

W10[B]: เพิ่ม events (asyncio.Queue ต่อ task) + pending (asyncio.Future ต่อ approval
request ที่ยังค้างอยู่) ให้ routes.py ผูก Orchestrator.run_task(on_event=..., ask_user_func=...)
เข้ากับ SSE stream (GET /tasks/{id}/stream) + respond endpoint (POST /tasks/{id}/respond)
— ทำให้ human-in-the-loop (permission prompt + plan confirmation) เป็นปุ่มจริงบนหน้าเว็บ
แทนที่จะ fail-closed อัตโนมัติเหมือนเดิม (ไม่มี human อยู่หน้าจอรอตอบ REST ตรงๆ)

W10[C]: เก็บ reference ของ asyncio.Task ไว้ใน TaskRecord.asyncio_task ด้วย (แยกจาก
self._running ที่มีไว้กัน GC เฉยๆ ไม่ผูกกับ task_id) ให้ cancel() หา task ที่ต้อง
.cancel() ถูกตัวจาก task_id ได้ตรงๆ — ผูกกับปุ่ม Stop บนหน้าเว็บ (POST /tasks/{id}/stop)
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Coroutine, Optional

from backend.app.config import settings
from backend.app.core.telemetry import write_step_trace, write_token_usage


@dataclass
class TaskRecord:
    task_id: str
    url: str
    goal: str
    provider: Optional[str]
    status: str  # "running" | "done" | "error" | "cancelled"
    # W_live: ค่า headless ที่ resolve แล้ว (ไม่มีทางเป็น None ตรงนี้ — caller resolve
    # req.headless ก่อนส่งเข้า submit() แล้ว) ให้ frontend รู้ว่า task นี้เปิด browser
    # แบบมองเห็นได้หรือไม่ เพื่อตัดสินใจว่าจะโชว์ live view (headless=True) หรือปิดไปเลย
    # (headless=False — browser จริงเปิดโชว์อยู่แล้ว ไม่ต้อง stream screenshot ซ้ำซ้อน)
    headless: bool = True
    created_at: float = field(default_factory=time.time)
    result: Optional[dict] = None
    error: Optional[str] = None
    # W20 (Task4 "User Chat Bubble File Attachments"): filename of the file the user attached
    # to this turn's composer submission (pdf/xlsx/image, see routes.py::
    # CreateTaskRequest.attached_file_name) — kept here (not the base64 content, which is only
    # ever needed transiently for extraction/vision) purely so GET /tasks can echo it back and
    # the frontend can re-render the attachment card/thumbnail for historical turns after a
    # page reload, not just for the live SSE session that originally sent it.
    attached_file_name: Optional[str] = None
    # W10[C]: asyncio.Task ที่กำลังรัน _run(record, coro) ของ task นี้อยู่ — เก็บไว้ให้
    # cancel() เรียก .cancel() ถูกตัวได้ตรงๆ จาก task_id (ตั้งค่าใน submit() ทันทีหลังสร้าง
    # ไม่มีทาง None ตอน task ยัง "running" อยู่จริง)
    asyncio_task: Optional[asyncio.Task] = None
    # W26 ("Broadcast live events to every subscribed tab" — บั๊กจริงที่ user รายงาน:
    # approval prompt/log ไม่ขึ้นสด ต้องกด F5): เดิม (W10[B]) เป็น asyncio.Queue เดียว
    # ผูกกับ task ตลอดอายุ โดยตั้งใจไว้ว่า "ผู้ชมสดคนเดียวต่อ task ไม่ใช่ pub-sub" — แต่
    # frontend (index.html::ensureConversationFor, W_multitab) ตั้งใจให้ "ทุกแท็บที่เปิด
    # ค้างไว้เห็น task จากแท็บอื่นด้วย" (ฟีเจอร์จริง ไม่ใช่บั๊ก — GET /tasks คืน task ทั้ง
    # ระบบ ไม่กรองเฉพาะแท็บที่สร้าง แล้ว refreshTasks() เปิด SSE ให้ทุก task ที่ "running"
    # โดยอัตโนมัติ) ทำให้หลายแท็บ subscribe task เดียวกันพร้อมกันเป็นสถานการณ์ปกติ ไม่ใช่
    # edge case หายากอย่างที่ comment เดิมสันนิษฐานไว้ — ต้อง broadcast event ให้ subscriber
    # ทุกตัวจริงๆ (pub-sub) ไม่ใช่แค่ตัวเดียว — เปลี่ยนจาก Queue เดียวเป็น "รายชื่อผู้ฟัง"
    # (แต่ละ SSE connection ที่เปิดอยู่จริงมี Queue ของตัวเอง สมัคร/ถอนตัวเองใน
    # routes.py::_stream_task_events()) push_event()/request_approval()/_run() ด้านล่าง
    # ป้อน event เดียวกันเข้าทุก Queue ในลิสต์นี้พร้อมกัน
    event_subscribers: list = field(default_factory=list)
    # W10[B]: request_id -> {"future": Future[bool], "cmd": dict, "delivered": bool} ของ
    # approval ที่ยังรอ user ตอบอยู่ (ปกติมีแค่รายการเดียวพร้อมกัน เพราะ ask_user_func ถูก
    # await ทีละครั้งจาก loop เดียวใน run_task() แต่เก็บเป็น dict กัน race แปลกๆ ไว้เผื่อ
    # อนาคต) — เก็บ cmd ไว้ด้วย (ไม่ใช่แค่ future) เพื่อให้ stream_task() replay ให้ SSE
    # connection ใหม่เห็นได้ ถ้า tab เดิมหลุดไปกลางคันระหว่างรอ (เช่น ปิด/รีเฟรชแท็บตอนมี
    # permission prompt ค้างอยู่) — "delivered" กัน double-delivery: stream_task() replay
    # เฉพาะรายการที่เคยส่งออกไปแล้วอย่างน้อยหนึ่งครั้ง (True) เท่านั้น รายการที่ยังไม่เคย
    # ส่งเลย (False, เพิ่งถูกสร้างเกือบพร้อมกันกับตอน connection ใหม่เพิ่งต่อ) ปล่อยให้ไหล
    # ผ่าน queue drain ปกติด้านล่างแทน ไม่งั้น connection เดียวจะเห็น event ซ้ำสองครั้ง
    pending: dict = field(default_factory=dict)


async def _broadcast(record: TaskRecord, event: dict) -> None:
    """W26: ป้อน event เดียวกันเข้า Queue ของ subscriber (SSE connection ที่เปิดอยู่จริง)
    ทุกตัว — ใช้แทนที่ record.events.put() เดิมทุกจุด (_run()/push_event()/
    request_approval() ด้านล่าง) กัน subscriber ตัวใดตัวหนึ่งเห็น event ไม่ครบถ้วนถ้ามี
    หลายแท็บเปิดดู task เดียวกันพร้อมกัน (ดู TaskRecord.event_subscribers ด้านบนสำหรับ
    เหตุผลเต็ม) — list(record.event_subscribers) กัน RuntimeError "list changed size
    during iteration" เผื่อ subscriber ใหม่ลงทะเบียนตัวเองแทรกเข้ามาระหว่าง broadcast
    รอบนี้พอดี (ไม่จำเป็นต้องเห็น event รอบนี้ก็ได้ ปลอดภัยกว่าดักด้วย lock)"""
    for queue in list(record.event_subscribers):
        await queue.put(event)


def _log_token_usage(record: TaskRecord) -> None:
    """W49: เขียน 1 บรรทัด JSON ต่อ task (append-only, JSON Lines) — เรียกจาก _run() ทันทีหลัง
    coro คืนค่า เป็น choke point เดียวที่ครอบคลุมทุก return path ของ Orchestrator.run_task()
    (finish_task ปกติ, chat_reply, plan ถูกปฏิเสธ, หมด max_steps — ทุกอันคืน dict ที่มี key
    "tokens" เหมือนกัน ดู orchestrator.py::_tokens_dict())

    W_step_trace: เรียกทุกเส้นทางแล้ว ไม่ใช่เฉพาะตอนสำเร็จ — เดิม task ที่ crash หรือถูก Stop
    ไม่เคยถูกบันทึกเลย success rate ที่คำนวณจากไฟล์นี้จึงเป็น "เพดานบน" ไม่ใช่ค่าจริง

    W_eval_trace: เนื้อในย้ายไปอยู่ core/telemetry.py แล้ว (ดูเหตุผลที่หัวไฟล์นั้น) — ฟังก์ชันนี้
    เหลือหน้าที่เดียวคือแปลง TaskRecord เป็น argument ให้ writer ตัวกลาง พฤติกรรมและรูปแบบไฟล์
    เหมือนเดิมทุกประการ (source="api" เป็น default อยู่แล้ว)

    ห้าม throw ออกจากฟังก์ชันนี้เด็ดขาด — telemetry.write_token_usage() จับ Exception เองอยู่แล้ว
    และผู้เรียกยังห่อ try/except ไว้อีกชั้น"""
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
    """W_step_trace: เขียน 1 บรรทัดต่อ step ลง settings.step_trace_log_path — เขียนครั้งเดียว
    ตอน task จบ ไม่ใช่ทุก step เพื่อไม่ให้ disk I/O แทรกกลาง agent loop

    W_eval_trace: เนื้อในย้ายไป core/telemetry.py แล้ว เหลือแค่แปลง TaskRecord เป็น argument
    (ดูเหตุผลที่หัวไฟล์นั้น) — ห้าม throw เหมือน _log_token_usage() ด้านบน"""
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
        """สร้าง task_id ล่วงหน้าก่อนเรียก submit() — routes.py ต้องรู้ task_id นี้ตอน
        ประกอบ ask_user_func/on_event closure (ที่ต้องอ้างอิง task_id เพื่อ push event
        เข้า record ที่ถูกต้อง) ซึ่งเกิดขึ้น *ก่อน* record จะถูกสร้างจริงใน submit()"""
        return str(uuid.uuid4())

    def submit(
        self, task_id: str, url: str, goal: str, provider: Optional[str],
        coro: Coroutine[Any, Any, dict], headless: bool = True,
        attached_file_name: Optional[str] = None,
    ) -> TaskRecord:
        """สร้าง TaskRecord สถานะ "running" ทันที (ด้วย task_id ที่ caller สร้างไว้ล่วงหน้า
        ผ่าน new_task_id() แล้ว) แล้วสั่งรัน coro (โดยทั่วไปคือ Orchestrator.run_task() ที่
        ห่อด้วย BrowserPool.acquire() ดู routes.py) เป็น background — ไม่ await ตรงนี้ คืน
        record กลับทันทีให้ endpoint ส่ง response 202

        headless: ค่าที่ resolve แล้ว (ไม่ใช่ req.headless ดิบๆ ที่อาจเป็น None) ให้
        caller (routes.py) เป็นคนตัดสิน settings.browser_headless fallback เอง ก่อนส่งเข้า
        มาตรงนี้

        attached_file_name (W20, Task4): ชื่อไฟล์ที่ user แนบมากับ turn นี้ (ถ้ามี) — เก็บไว้
        เฉยๆ ให้ GET /tasks คืนกลับได้ ไม่มีผลอะไรกับการรัน coro เลย"""
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
            # W10[C]: มาจาก cancel() (ปุ่ม Stop บนหน้าเว็บ) — โดยปกติ convention ของ
            # CancelledError คือต้อง re-raise ต่อเสมอ แต่ตัว task นี้ (จาก submit()) เป็น
            # background job แบบ fire-and-forget ล้วนๆ ไม่มี caller ไหน await ตรงๆ ที่ต้อง
            # เห็น cancellation ส่งต่อ — สิ่งที่สำคัญกว่าคือต้องอัพเดต record.status +
            # push task_done event ให้ SSE consumer เห็นผลจริง (ไม่งั้นจะค้าง "running"
            # ตลอดไปเพราะ except Exception ด้านล่างจับ CancelledError ไม่ได้ตั้งแต่
            # Python 3.8 — มันสืบทอดจาก BaseException ไม่ใช่ Exception แล้ว)
            record.status = "cancelled"
            record.error = "หยุดโดยผู้ใช้ (Stop)"
        except Exception as e:
            record.error = str(e)
            record.status = "error"
        # W_step_trace: ย้าย logging ออกมานอก try ให้ครอบ *ทุก* เส้นทาง (done/cancelled/error)
        # — เดิมเรียกเฉพาะตอนสำเร็จ ทำให้ task ที่ crash หรือถูก Stop ไม่เคยถูกบันทึกเลย
        # success rate ที่คำนวณจากไฟล์นี้จึงเป็น "เพดานบน" ไม่ใช่ค่าจริง (สำคัญมากตอนใช้
        # ไฟล์นี้ประเมินว่าการแก้แต่ละรอบได้ผลจริงไหม)
        #
        # ตอนนี้ path error ก็มี result dict ให้บันทึกแล้วจริงในกรณีส่วนใหญ่ เพราะ
        # orchestrator.py::run_task จับ exception ในลูปแล้วคืน dict ปกติ (W_loop_crash) —
        # ที่เหลือ (exception นอกลูป/cancelled) ยังไม่มี result แต่ก็ยังบันทึกสถานะไว้ได้
        try:
            await asyncio.to_thread(_log_token_usage, record)
            await asyncio.to_thread(_log_step_trace, record)
        except Exception:
            pass
        # W10[B]: ยกเลิก pending approval ที่ยังค้างอยู่ (เช่น task ล้มเหลว/ถูก stop กลางคัน
        # ระหว่างรอ user ตอบ) กัน Future ค้างไม่มีใครมา resolve ไปตลอดกาล
        for info in record.pending.values():
            if not info["future"].done():
                info["future"].set_result(False)
        # sentinel เดียวที่บอก SSE consumer ว่า stream จบแล้ว (ดู stream_task() ใน routes.py)
        await _broadcast(record, {
            "kind": "task_done", "status": record.status,
            "result": record.result, "error": record.error,
        })

    async def cancel(self, task_id: str, wait_timeout: float = 5.0) -> bool:
        """เรียกจาก POST /tasks/{id}/stop — ส่ง asyncio.CancelledError เข้าไปใน task ที่
        กำลังรันอยู่ (ไม่ว่าจะกำลัง await LLM call, execute() action, หรือรอ
        request_approval() อยู่ก็ตาม — cancel() ทำงานได้ทุกจังหวะ await) คืน False ถ้า
        task ไม่ได้ "running" อยู่แล้ว (จบไปแล้ว/ถูก stop ไปแล้ว) ให้ endpoint คืน 409

        W27: แก้บั๊ก "ปุ่ม kill session ใช้งานจริงไม่ได้" — เดิม .cancel() แค่ยิง
        CancelledError เข้า task แล้ว "return True" ทันที ไม่รอให้ task หยุดใช้ page/browser
        จริงๆ ก่อน (asyncio.Task.cancel() แค่ "ขอ" ให้ cancel ที่จุด await ถัดไป ไม่ได้หยุด
        ทันที) — ฝั่ง frontend (index.html::killSession()) เรียก POST .../stop แล้วต่อด้วย
        POST /sessions/{id}/close ทันทีที่ stop ตอบกลับมา ถ้า _run() ยังไม่ทันประมวลผล
        CancelledError เสร็จ (ยังอยู่กลาง page.click()/page.evaluate() ฯลฯ) session.close()
        จะไปแตะ page/context/browser ตัวเดียวกันพร้อมกัน race กันจริง — ตอนนี้ await task
        ตัวเดิมหลังสั่ง cancel() (มี timeout กันค้างถ้า step ไหนไม่ยอมคืน control loop เลย)
        ก่อน return — ไม่ throw ออกมาแม้ await timeout (แค่ไม่รอต่อ ไม่ใช่ error) เพราะ
        _run() เองก็ catch CancelledError ไว้แล้วไม่ re-raise (ดู docstring ของ _run())
        await ตรงนี้จึงไม่มีทาง raise CancelledError กลับมาที่นี่เองอยู่แล้ว"""
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
        """เรียกจาก ask_user_func (routes.py) — push event "approval_request" เข้า
        stream ของ task นี้ (โชว์ปุ่ม Approve/Deny หรือ Confirm plan บนหน้าเว็บ) แล้วรอ
        (block เฉพาะ task coroutine นี้ ไม่บล็อก event loop รวม) จนกว่า resolve_approval()
        จะถูกเรียก (จาก POST /tasks/{id}/respond) — คืน False ถ้าไม่มี record นี้อยู่แล้ว
        (fail closed เหมือน default เดิม)

        W10[E]: timeout (วินาที) — ถ้าไม่มีใครตอบภายในเวลานี้ ถือว่าปฏิเสธอัตโนมัติ
        (คืน False เหมือนโดนกด Deny) แทนที่จะรอเฉยๆ ตลอดกาล — สำคัญมากเพราะ task ที่ค้าง
        รอ human ตอบอยู่ (เช่น user ปิดแท็บทิ้งกลางคันตอนรอ confirm plan) ยัง "ยึด" browser
        จาก pool ไว้อยู่ (pool.acquire() ยังไม่คืน context จนกว่า run_task() จะ return จริง)
        ถ้าไม่มี timeout เลย task ค้างพวกนี้จะกัด quota ของ browser_pool_size ไปเรื่อยๆ จน
        task ใหม่ทุกตัวต้องรอคิว browser ที่ไม่มีวันว่าง (ดูอาการจริงที่ routes.py::
        _make_ask_user_func เรียกใช้ค่านี้จาก settings.approval_timeout_seconds)"""
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
        """เรียกจาก POST /tasks/{id}/respond — คืน False ถ้าไม่พบ request_id นี้แล้ว
        (ตอบไปแล้ว/หมดอายุ/task_id ผิด) ให้ endpoint คืน 404 ต่อ

        W10[F]: edited_plan — ถ้า request นี้เป็น confirm_plan และ user แก้ไขข้อความแผน
        ก่อนกด Confirm ให้แก้ค่า "plan" ใน info["cmd"] ก่อน resolve future — mutate dict
        เดิม (ไม่สร้างใหม่) เพราะ orchestrator.py::_confirm_plan() ยังถือ reference ของ
        dict ก้อนเดียวกันนี้อยู่ (ส่งเข้า ask_user_func ไปแล้วแต่ยังไม่ทิ้ง) พอ future
        resolve กลับมา จะอ่าน cmd["plan"] ที่ถูกแก้แล้วออกไปใช้แทนแผนเดิมที่ AI ร่างไว้ —
        ไม่แตะ action อื่น (permission prompt ทั่วไปไม่มี key "plan" ให้แก้อยู่แล้ว)

        W_resume ("Mid-Task Input Request"): answer_text — เหมือน edited_plan ข้างบนทุก
        ประการ แค่คนละ cmd type (request_user_input) และคนละ key ("answer" แทน "plan") —
        orchestrator.py::_request_user_input() ยังถือ reference ของ cmd dict ก้อนเดียวกัน
        นี้อยู่เช่นกัน อ่าน cmd["answer"] กลับไปใช้ทำ task เดิมต่อทันทีโดยไม่จบ loop"""
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

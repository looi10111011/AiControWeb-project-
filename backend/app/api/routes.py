"""W10[A]: API endpoints ที่ขับ Orchestrator จริงผ่าน BrowserPool แทน CLI (run.py)

Task submission เป็นแบบ async (submit -> 202 + task_id -> poll GET /tasks/{id}) ไม่ใช่
sync request-response ตรงๆ เพราะ run_task() ใช้เวลานาน (หลาย step, เรียก LLM จริงทุก
step) ดู task_manager.py สำหรับเหตุผลเต็มๆ

W10[B]: เพิ่ม GET /tasks/{id}/stream (SSE) ให้หน้าเว็บเห็นความคืบหน้าสดๆ ทีละ step
(ไม่ใช่ poll ผลลัพธ์รวมท้าย task เดียวเหมือนเดิม) + POST /tasks/{id}/respond ให้กดปุ่ม
Approve/Deny หรือ Confirm plan บนหน้าเว็บได้จริง (human-in-the-loop ผ่าน REST จริงๆ
แทนที่จะ fail-closed/auto-approve อัตโนมัติ) — ทั้งสองผูกกับ TaskManager.push_event/
request_approval/resolve_approval (ดู task_manager.py)
"""

import json

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from backend.app.api.schemas import (
    CreateTaskRequest,
    PoolStatusResponse,
    RespondRequest,
    TaskCreatedResponse,
    TaskStatusResponse,
)
from backend.app.api.task_manager import TaskManager
from backend.app.config import settings
from backend.app.core.orchestrator import Orchestrator

router = APIRouter()


def _make_ask_user_func(task_manager: TaskManager, task_id: str, auto_approve: bool):
    """ask_user_func ที่ orchestrator ใช้ทั้งสองจุด: confirm_plan (ก่อนเริ่ม loop) และ
    permission-gated action ตอนรัน (actions.py) — cmd dict เดียวกันบอกได้อยู่แล้วว่าเป็น
    คำขอแบบไหน (cmd["type"] == "confirm_plan" หรือ action จริง เช่น "purchase"/"delete")
    ไม่ต้องแยก branch พิเศษ ส่งต่อให้ human ตัดสินใจจริงทั้งคู่ผ่านช่องทางเดียวกัน
    (TaskManager.request_approval -> push event "approval_request" เข้า SSE stream ->
    รอ POST /tasks/{id}/respond)

    auto_approve=True: ข้าม human-in-the-loop ทั้งหมด (ทั้ง plan และ action) อนุมัติเอง
    ทันที — ไว้ให้รันแบบไม่มีคนเฝ้าหน้าจอ (เช่น batch/CI) เหมือนพฤติกรรมเดิมก่อน W10[B]

    W10[E]: จำกัดเวลารอ human ตอบด้วย settings.approval_timeout_seconds เสมอ (ไม่ใช่รอ
    ตลอดกาล) — ไม่งั้น task ที่ user ปิดแท็บทิ้งกลางคันตอนรอ confirm plan จะยึด browser
    จาก pool ไว้ไม่มีวันคืน กัด quota ของ browser_pool_size จน task ใหม่ทุกตัวรอคิวไม่รู้จบ
    (ดู task_manager.py::request_approval() สำหรับรายละเอียดเต็ม)
    """

    async def ask_user_func(cmd: dict) -> bool:
        if auto_approve:
            await task_manager.push_event(task_id, {"kind": "auto_approved", "cmd": cmd})
            return True
        return await task_manager.request_approval(
            task_id, cmd, timeout=settings.approval_timeout_seconds
        )

    return ask_user_func


@router.post("/tasks", response_model=TaskCreatedResponse, status_code=202)
async def create_task(req: CreateTaskRequest, request: Request) -> TaskCreatedResponse:
    pool = request.app.state.browser_pool
    task_manager: TaskManager = request.app.state.task_manager
    orchestrator = Orchestrator()
    task_id = task_manager.new_task_id()
    ask_user_func = _make_ask_user_func(task_manager, task_id, req.auto_approve)

    async def _on_event(event: dict) -> None:
        await task_manager.push_event(task_id, event)

    # W10[C]: headless=False ตรงๆ (ไม่ใช่ None/True) = user ขอเห็นหน้าต่าง browser จริง
    # ("เปิดหน้าเว็ปจริงขึ้นมารันคู่ไปด้วย") — บายพาส pool ไปเลย เพราะ browser ใน pool
    # ถูก launch แบบ headless ไว้ล่วงหน้าตั้งแต่ตอน server startup แล้ว (ดู
    # browser_pool.py::BrowserPool.start()) เปลี่ยน headless ทีหลังผ่าน context ใหม่บน
    # browser เดิมไม่ได้ ต้อง launch browser ของตัวเองใหม่ทั้งตัว (owns_browser=True path
    # ใน orchestrator.py) ถึงจะเปิดหน้าต่างจริงได้ — ผูก keep_browser_open=True คู่กันเสมอ
    # (ไม่ต้องปิดหน้าต่างจนกว่า user จะปิดเอง) เพราะไม่มีประโยชน์อะไรที่จะเปิดหน้าต่างโชว์
    # แล้วรีบปิดทันทีที่ทำงานเสร็จ
    wants_visible_browser = req.headless is False

    async def _run() -> dict:
        if wants_visible_browser:
            return await orchestrator.run_task(
                url=req.url,
                goal=req.goal,
                max_steps=req.max_steps,
                headless=False,
                verbose=False,
                provider=req.provider,
                ask_user_func=ask_user_func,
                confirm_plan=req.confirm_plan,
                on_event=_on_event,
                keep_browser_open=True,
            )
        async with pool.acquire() as browser:
            return await orchestrator.run_task(
                url=req.url,
                goal=req.goal,
                max_steps=req.max_steps,
                headless=req.headless,
                verbose=False,
                provider=req.provider,
                ask_user_func=ask_user_func,
                confirm_plan=req.confirm_plan,
                browser=browser,
                on_event=_on_event,
            )

    record = task_manager.submit(task_id, req.url, req.goal, req.provider, _run())
    return TaskCreatedResponse(task_id=record.task_id, status=record.status)


@router.get("/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task(task_id: str, request: Request) -> TaskStatusResponse:
    task_manager = request.app.state.task_manager
    record = task_manager.get(task_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"ไม่พบ task_id: {task_id!r}")
    return TaskStatusResponse(
        task_id=record.task_id,
        url=record.url,
        goal=record.goal,
        provider=record.provider,
        status=record.status,
        created_at=record.created_at,
        result=record.result,
        error=record.error,
    )


@router.get("/tasks", response_model=list[TaskStatusResponse])
async def list_tasks(request: Request) -> list[TaskStatusResponse]:
    task_manager = request.app.state.task_manager
    return [
        TaskStatusResponse(
            task_id=r.task_id,
            url=r.url,
            goal=r.goal,
            provider=r.provider,
            status=r.status,
            created_at=r.created_at,
            result=r.result,
            error=r.error,
        )
        for r in task_manager.list()
    ]


@router.get("/tasks/{task_id}/stream")
async def stream_task(task_id: str, request: Request) -> StreamingResponse:
    """W10[B]: Server-Sent Events ของ task นี้ — step log สดๆ ระหว่างรัน +
    approval_request (permission prompt / plan confirmation) + task_done ปิดท้าย

    ออกแบบไว้สำหรับ "ผู้ชมสดคนเดียวต่อ task" (แท็บที่ยิง POST /tasks สร้าง task นี้ขึ้นมา
    เอง) ไม่ใช่ pub-sub หลายคน — ถ้า client มาเชื่อมต่อ *หลัง* task จบไปแล้ว จะไม่มี event
    เก่าให้ replay (ไม่ได้เก็บ log buffer แยก) เลยส่ง task_done สังเคราะห์กลับทันทีจาก
    record.status/result/error ที่ยังอยู่แทน เพื่อให้หน้าเว็บที่รีเฟรชทีหลังยังเห็นผลลัพธ์
    สุดท้ายได้ (ไม่ hang รอ event ที่ไม่มีวันมาอีกแล้ว)
    """
    task_manager: TaskManager = request.app.state.task_manager
    record = task_manager.get(task_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"ไม่พบ task_id: {task_id!r}")

    async def event_gen():
        if record.status != "running":
            done_event = {
                "kind": "task_done", "status": record.status,
                "result": record.result, "error": record.error,
            }
            yield f"data: {json.dumps(done_event)}\n\n"
            return
        # W10[B]: replay เฉพาะ approval ที่เคย "ส่งออกไปแล้ว" อย่างน้อยหนึ่งครั้ง
        # (delivered=True) — กรณี tab เดิมหลุดไปกลางคันระหว่างรอ permission prompt (event
        # เดิมถูก consume ออกจาก events queue ไปแล้วครั้งเดียว ไม่งั้น connection ใหม่จะไม่
        # รู้เลยว่ามีอะไรค้างรอ) ส่วนรายการที่ยังไม่เคยส่งเลย (เพิ่งถูกสร้างเกือบพร้อมกันกับ
        # connection นี้) ปล่อยให้ไหลผ่าน queue drain ปกติด้านล่างแทน — ไม่งั้น connection
        # เดียวกันนี้จะเห็น event ซ้ำสองครั้ง (ทั้งจาก replay และจาก queue)
        for request_id, info in list(record.pending.items()):
            if info["delivered"]:
                yield f"data: {json.dumps({'kind': 'approval_request', 'request_id': request_id, 'cmd': info['cmd']})}\n\n"
        while True:
            event = await record.events.get()
            if event.get("kind") == "approval_request":
                info = record.pending.get(event.get("request_id"))
                if info is not None:
                    info["delivered"] = True
            yield f"data: {json.dumps(event)}\n\n"
            if event.get("kind") == "task_done":
                break

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@router.post("/tasks/{task_id}/stop")
async def stop_task(task_id: str, request: Request) -> dict:
    """W10[C]: ปุ่ม Stop บนหน้าเว็บ — ยกเลิก task ที่กำลังรันอยู่กลางคัน (ไม่ว่าจะกำลังรอ
    LLM ตอบ, กำลัง execute() action, หรือกำลังรอ human ตอบ permission/plan prompt อยู่ก็
    ตาม — ดู TaskManager.cancel()) คืน 409 ถ้า task ไม่ได้ "running" อยู่แล้ว (จบไปแล้ว/
    ถูก stop ไปแล้วก่อนหน้านี้)"""
    task_manager: TaskManager = request.app.state.task_manager
    record = task_manager.get(task_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"ไม่พบ task_id: {task_id!r}")
    ok = task_manager.cancel(task_id)
    if not ok:
        raise HTTPException(status_code=409, detail="Task นี้ไม่ได้กำลังรันอยู่แล้ว")
    return {"status": "stopping"}


@router.post("/tasks/{task_id}/respond")
async def respond_task(task_id: str, req: RespondRequest, request: Request) -> dict:
    """W10[B]: ปุ่ม Approve/Deny (permission prompt) หรือ Confirm/Cancel (plan) บนหน้าเว็บ
    ยิงมาที่นี่ — request_id ต้องตรงกับ approval_request event ล่าสุดที่ยังไม่ถูกตอบ
    (ดู TaskManager.resolve_approval()) ไม่งั้นถือว่าหมดอายุ/ตอบไปแล้ว คืน 404"""
    task_manager: TaskManager = request.app.state.task_manager
    ok = task_manager.resolve_approval(task_id, req.request_id, req.approved, edited_plan=req.edited_plan)
    if not ok:
        raise HTTPException(status_code=404, detail="ไม่พบ pending request นี้ (อาจหมดอายุหรือตอบไปแล้ว)")
    return {"status": "ok"}


@router.get("/pool/status", response_model=PoolStatusResponse)
async def pool_status(request: Request) -> PoolStatusResponse:
    pool = request.app.state.browser_pool
    return PoolStatusResponse(size=pool.size, available=pool.available, in_use=pool.size - pool.available)

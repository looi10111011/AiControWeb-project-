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
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from playwright.async_api import async_playwright

from backend.app.api.schemas import (
    CreateTaskRequest,
    CredentialsStatusResponse,
    ExecutePlanRequest,
    GeneratePlanRequest,
    GeneratePlanResponse,
    LearnCreatedResponse,
    LearnCredentialsRequest,
    LearnSiteRequest,
    PoolStatusResponse,
    RelearnPageRequest,
    RelearnPageResponse,
    RespondRequest,
    SaveCredentialsRequest,
    SessionStatusResponse,
    SiteManualStatusResponse,
    TaskCreatedResponse,
    TaskStatusResponse,
)
from backend.app.api.task_manager import TaskManager
from backend.app.config import settings
from backend.app.core import plan_memory
from backend.app.core.orchestrator import Orchestrator
from backend.app.permission.rules import extract_domain
from backend.app.site_learning import crawl_site, describe_page, extract_page
from backend.app.site_learning.learn_manager import LearnManager
from backend.app.site_learning.storage import (
    credentials_exist,
    delete_credentials,
    load_credentials,
    load_knowledge_text,
    load_manual,
    manual_exists,
    save_credentials,
    save_manual,
    update_single_page,
)

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


async def _run_with_resolved_browser(
    req, orchestrator: Orchestrator, ask_user_func, on_event, pool, session_registry,
    extra_run_task_kwargs: dict,
) -> dict:
    """W13: ตรรกะร่วม "จะเอา page มาจากไหน" ระหว่าง POST /tasks (create_task,
    confirm_plan) และ POST /api/execute_plan (approved_plan) — ต่างกันแค่
    extra_run_task_kwargs ที่ผู้เรียกส่งมา (confirm_plan=... หรือ approved_plan=...)
    req รับได้ทั้ง CreateTaskRequest/ExecutePlanRequest (duck-typed — ใช้ attribute ชุด
    เดียวกันทั้งคู่: url/goal/max_steps/provider/headless/auto_approve/
    use_user_browser/tab_reuse_policy/session_id)

    ลำดับความสำคัญ: session_id ก่อน (ครอบคลุมทั้ง 3 โหมดในตัวผ่าน session_registry
    อยู่แล้ว) -> use_user_browser -> headless=False ตรงๆ (visible browser, launch เอง,
    ผูก keep_browser_open=True คู่กันเสมอเพราะไม่มีประโยชน์ที่จะเปิดหน้าต่างโชว์แล้วรีบ
    ปิดทันทีที่เสร็จ) -> fallback ไปยืมจาก pool (headless ตาม req.headless เป๊ะๆ)"""
    wants_visible_browser = req.headless is False
    # W14: โหลดคู่มือเว็บไซต์ที่ crawl มาอัตโนมัติครั้งเดียวตรงนี้ (ถ้ามี) แล้วส่งต่อเข้า
    # run_task() ทุก branch ด้านล่าง — ว่างเปล่าเงียบๆ ถ้าโดเมนนี้ยังไม่เคยถูกเรียนรู้
    site_manual_context = load_knowledge_text(extract_domain(req.url))

    # W12: session_id มา -> ผูก task นี้เข้ากับ session ที่มีชีวิตอยู่ข้ามหลาย request
    # (ดู core/session_registry.py) ครั้งแรกที่เจอ session_id นี้จะสร้าง page ใหม่ตาม
    # use_user_browser/headless ของ request นี้ ครั้งถัดๆ ไปด้วย session_id เดิมได้ page
    # ตัวเดิมกลับมาทันที (ไม่ต้อง acquire/launch/connect ซ้ำ) — ส่ง page= ตรงๆ ให้
    # orchestrator ข้าม acquisition/teardown ทั้งหมด (managed_externally=True)
    if req.session_id:
        session = await session_registry.get_or_create(
            req.session_id,
            use_user_browser=req.use_user_browser,
            headless=req.headless,
            target_url=req.url,
            pool=pool,
            tab_reuse_policy=req.tab_reuse_policy,
            ask_user_func=ask_user_func,
        )
        return await orchestrator.run_task(
            url=req.url,
            goal=req.goal,
            max_steps=req.max_steps,
            verbose=False,
            provider=req.provider,
            ask_user_func=ask_user_func,
            on_event=on_event,
            page=session.page,
            site_manual_context=site_manual_context,
            session_id=req.session_id,
            **extra_run_task_kwargs,
        )

    # W12: user_browser bypass pool เหมือน wants_visible_browser ด้านล่าง (browser ที่
    # ต่อผ่าน CDP เป็นของ user เอง ไม่ใช่ของ pool ให้ยืม) — ตรวจก่อน wants_visible_browser
    # เพราะ headless ไม่มีความหมายเลยในโหมดนี้ (ต่อเข้า browser จริงที่เปิดอยู่แล้ว ไม่
    # launch เอง) ไม่ต้องสน req.headless
    if req.use_user_browser:
        return await orchestrator.run_task(
            url=req.url,
            goal=req.goal,
            max_steps=req.max_steps,
            verbose=False,
            provider=req.provider,
            ask_user_func=ask_user_func,
            on_event=on_event,
            connect_to_user_browser=True,
            tab_reuse_policy=req.tab_reuse_policy,
            site_manual_context=site_manual_context,
            session_id=req.session_id,
            **extra_run_task_kwargs,
        )
    if wants_visible_browser:
        return await orchestrator.run_task(
            url=req.url,
            goal=req.goal,
            max_steps=req.max_steps,
            headless=False,
            verbose=False,
            provider=req.provider,
            ask_user_func=ask_user_func,
            on_event=on_event,
            keep_browser_open=True,
            site_manual_context=site_manual_context,
            session_id=req.session_id,
            **extra_run_task_kwargs,
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
            browser=browser,
            on_event=on_event,
            site_manual_context=site_manual_context,
            session_id=req.session_id,
            **extra_run_task_kwargs,
        )


@router.post("/tasks", response_model=TaskCreatedResponse, status_code=202)
async def create_task(req: CreateTaskRequest, request: Request) -> TaskCreatedResponse:
    pool = request.app.state.browser_pool
    session_registry = request.app.state.session_registry
    task_manager: TaskManager = request.app.state.task_manager
    orchestrator = Orchestrator()
    task_id = task_manager.new_task_id()
    ask_user_func = _make_ask_user_func(task_manager, task_id, req.auto_approve)

    async def _on_event(event: dict) -> None:
        await task_manager.push_event(task_id, event)

    async def _run() -> dict:
        return await _run_with_resolved_browser(
            req, orchestrator, ask_user_func, _on_event, pool, session_registry,
            extra_run_task_kwargs={"confirm_plan": req.confirm_plan},
        )

    record = task_manager.submit(task_id, req.url, req.goal, req.provider, _run())
    return TaskCreatedResponse(task_id=record.task_id, status=record.status)


@router.post("/api/generate_plan", response_model=GeneratePlanResponse)
async def generate_plan(req: GeneratePlanRequest, request: Request) -> GeneratePlanResponse:
    """W13: เฟสวางแผนแยกต่างหาก — synchronous ตรงๆ (ไม่ผ่าน TaskManager/SSE เพราะเป็น
    แค่ LLM call เดียว ไม่ใช่ agent loop หลายนาทีเหมือน execute_plan) ไม่เปิด/connect
    browser ใหม่เด็ดขาด (ดู orchestrator.py::Orchestrator.generate_plan()) — ถ้ามี
    session_id ที่มี page เปิดค้างอยู่แล้วจริง (จากเทิร์นก่อนหน้า) จะ perceive หน้านั้น
    มาช่วยร่างแผนให้ grounded กับสถานะปัจจุบัน (ใช้ session_registry.get() เฉยๆ ไม่ใช่
    get_or_create() — ไม่มีทางสร้าง session ใหม่จาก endpoint นี้)

    W20: เช็ค Plan Memory ก่อนเรียก LLM เสมอ (ดู core/plan_memory.py) — หา approved plan
    ที่ตรงกับ (domain, goal) นี้มากที่สุดด้วย semantic search (ไม่ใช่ exact text match —
    "Login"/"Sign in"/"เข้าสู่ระบบ" ควรจับคู่ lineage เดียวกันได้) เจอ = คืนแผนนั้นตรงๆ
    เลย ข้าม LLM ไปทั้งหมด (Plan Priority: user-approved มาก่อน LLM เสมอ) ไม่เจอ/ไม่ตรงพอ
    = fallback ไปให้ LLM ร่างใหม่ตามปกติด้านล่าง"""
    domain = extract_domain(req.url)
    matched = plan_memory.find_matching_plan(domain, req.goal)
    if matched is not None:
        return GeneratePlanResponse(plan=matched["plan"])

    # W19: เช็ค is_healthy() ก่อนหยิบ page มาใช้ — session ที่ browser/page ถูกปิดไปแล้ว
    # จริง (user ปิดหน้าต่าง/tab เอง, crash) จะถูกมองเหมือนไม่มี session เลย (page=None)
    # ไม่ใช่ auto-recover ที่นี่ (endpoint นี้ต้อง "ไม่แตะ browser เองเด็ดขาด" ตาม
    # docstring ด้านบน — recovery จริงเกิดตอน execute_plan()/create_task() ผ่าน
    # session_registry.get_or_create() เท่านั้น)
    page = None
    if req.session_id:
        session_registry = request.app.state.session_registry
        session = session_registry.get(req.session_id)
        if session is not None and session_registry.is_healthy(session):
            page = session.page
    site_manual_context = load_knowledge_text(domain)
    try:
        plan = await Orchestrator().generate_plan(
            req.url, req.goal, provider=req.provider, page=page, site_manual_context=site_manual_context,
        )
    except Exception as e:
        # ห่อ exception ทุกชนิด (LLM API error, page เดิมจาก session_id ถูกปิด/นำทางไปแล้ว
        # ระหว่าง perceive ฯลฯ) เป็น HTTPException ที่มี detail จริง — ไม่งั้น FastAPI จะคืน
        # 500 เปล่าๆ ("Internal Server Error" ไม่มี context) ให้ frontend เห็นแค่นั้น
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
    return GeneratePlanResponse(plan=plan)


@router.post("/api/execute_plan", response_model=TaskCreatedResponse, status_code=202)
async def execute_plan(req: ExecutePlanRequest, request: Request) -> TaskCreatedResponse:
    """W13: รันแผนที่อนุมัติแล้วจาก POST /api/generate_plan (req.plan อาจถูก user แก้ไข
    ข้อความมาก่อนก็ได้) — เหมือน create_task() ทุกประการยกเว้นไม่มี confirm_plan gate
    เลย (อนุมัติไปแล้วตั้งแต่ก่อนเรียก endpoint นี้)

    W20: ทุกครั้งที่ user กด Approve (ไม่ว่าจะแก้ไขข้อความแผนมาก่อนหรือไม่) บันทึกเข้า
    Plan Memory เสมอ (ดู core/plan_memory.py::save_confirmed_plan) — "Confirm" คือจุดที่
    ถือว่าแผนนี้ approved แล้วตามสเปค ไม่ต้องรอ flag แยกจาก frontend ว่าแก้ไขหรือไม่ (ถ้า
    เนื้อหาเหมือน version ล่าสุดเป๊ะอยู่แล้ว plan_memory จะไม่สร้าง version ซ้ำซ้อนเปล่าๆ
    เอง) — draft ที่ยังไม่ confirm/plan ที่ user cancel ไม่มีทางมาถึง endpoint นี้เลย"""
    if req.plan:
        plan_memory.save_confirmed_plan(extract_domain(req.url), req.goal, req.plan)

    pool = request.app.state.browser_pool
    session_registry = request.app.state.session_registry
    task_manager: TaskManager = request.app.state.task_manager
    orchestrator = Orchestrator()
    task_id = task_manager.new_task_id()
    ask_user_func = _make_ask_user_func(task_manager, task_id, req.auto_approve)

    async def _on_event(event: dict) -> None:
        await task_manager.push_event(task_id, event)

    async def _run() -> dict:
        return await _run_with_resolved_browser(
            req, orchestrator, ask_user_func, _on_event, pool, session_registry,
            extra_run_task_kwargs={"approved_plan": req.plan},
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
    # W27: cancel() ตอนนี้เป็น async แล้ว (รอให้ task หยุดใช้ page/browser จริงก่อน return
    # ไม่ใช่แค่ยิง CancelledError แล้วคืนทันที — ดู TaskManager.cancel() docstring) กัน race
    # กับ POST /sessions/{id}/close ที่ frontend (killSession()) ยิงตามมาทันทีหลัง stop ตอบ
    ok = await task_manager.cancel(task_id)
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


@router.post("/sessions/{session_id}/close")
async def close_session(session_id: str, request: Request) -> dict:
    """W12: ปุ่ม "New Session" บน Test Console ยิงมาที่นี่ก่อนเริ่มบทสนทนาใหม่ — ปิด
    page/context/browser ที่ session นี้ถืออยู่ (ดู core/session_registry.py::
    SessionRegistry.close() สำหรับรายละเอียดตาม mode) คืน 404 ถ้าไม่พบ session_id นี้
    (ปิดไปแล้ว/ไม่เคยมีอยู่จริง) — ไม่กระทบ task ที่กำลังรันอยู่บน session นี้เลยถ้ามี
    (เป็นหน้าที่ของ frontend ที่จะเช็คก่อนว่าไม่มี task รันค้างอยู่ก่อนเรียก endpoint นี้)"""
    session_registry = request.app.state.session_registry
    closed = await session_registry.close(session_id)
    if not closed:
        raise HTTPException(status_code=404, detail=f"ไม่พบ session_id: {session_id!r}")
    return {"status": "closed"}


@router.get("/sessions", response_model=list[SessionStatusResponse])
async def list_sessions(request: Request) -> list[SessionStatusResponse]:
    """ไว้ debug/monitor ว่าตอนนี้มี session ไหนถือ browser resource ค้างอยู่บ้าง (มีผลต่อ
    /pool/status ด้วย — session mode="pool" กิน browser จาก pool ไปจนกว่าจะปิดเอง)"""
    session_registry = request.app.state.session_registry
    return [
        SessionStatusResponse(
            session_id=s.session_id, mode=s.mode,
            created_at=s.created_at, last_active_at=s.last_active_at,
        )
        for s in session_registry.list()
    ]


# --- W14: Website Learning & Manual Generation (backend/app/site_learning/) — ระบบแยก
# ต่างหากสมบูรณ์จาก RAG/ChromaDB (backend/app/rag/) เก็บ manual ที่ crawl มาอัตโนมัติ
# เป็นไฟล์ JSON บนดิสก์ ไว้ให้ agent โหลดกลับมาใช้แทนการสำรวจซ้ำทุกครั้ง


@router.get("/api/site-manual/status", response_model=SiteManualStatusResponse)
async def site_manual_status(url: str) -> SiteManualStatusResponse:
    """ขับ banner "เว็บไซต์นี้ยังไม่มีคู่มือ" บน Test Console — เช็คเฉยๆ ไม่สร้าง/แตะ
    อะไรเลย (แค่ os.path.exists() ผ่าน storage.load_manual())"""
    manual = load_manual(extract_domain(url))
    if manual is None:
        return SiteManualStatusResponse(exists=False, version=None)
    return SiteManualStatusResponse(exists=True, version=manual.version)


@router.post("/api/site-manual/learn", response_model=LearnCreatedResponse, status_code=202)
async def learn_site(req: LearnSiteRequest, request: Request) -> LearnCreatedResponse:
    """เริ่ม crawl เว็บไซต์ที่ req.url — submit แบบเดียวกับ POST /tasks (202 + learn_id
    ทันที ไม่รอ crawl จบ เพราะเดินหลายหน้าอาจใช้เวลาเป็นนาที)

    W16: เปิด browser ของตัวเองแบบมองเห็นได้ (headless=False) แยกจาก BrowserPool โดย
    เจตนา — pool ตัวหลักถูก launch ไว้ล่วงหน้าตอน startup ด้วย headless mode ค่าเดียว
    (settings.browser_headless ปกติ True สำหรับ task ทั่วไปที่รันเงียบๆ) แต่ user อยาก
    "เห็น" ว่า crawler กำลังเดินอยู่หน้าไหนสดๆ ระหว่างเรียนรู้เว็บไซต์ — จึงเปิด Chromium
    process แยกเฉพาะ job นี้ ปิดเองเมื่อ crawl จบ (ไม่ยืม/ไม่คืน pool เลย ไม่กระทบ task
    อื่นที่ใช้ pool พร้อมกัน)"""
    learn_manager: LearnManager = request.app.state.learn_manager
    learn_id = learn_manager.new_learn_id()

    async def _on_progress(event: dict) -> None:
        await learn_manager.push_event(learn_id, event)

    # W23: crawler.py เจอหน้า login ระหว่างทางที่ยังไม่มี username/password ให้เลย —
    # หยุดรอถามคนจริงผ่าน SSE (credentials_needed event) + POST .../credentials แทน
    # การบังคับให้กรอกไว้ล่วงหน้าก่อน crawl เหมือนเดิม เก็บ credential ที่ได้ทันทีที่ user
    # ตอบ (ไม่รอให้ crawl จบก่อน — ถ้า crawl ถูก stop กลางคันหลังจากนี้ credential ที่กรอก
    # ไปแล้วก็ยังไม่หายไปเปล่าๆ)
    #
    # domain ที่ได้ต้องตรงกับ extract_domain(req.url) เท่านั้น (การันตีว่าไม่มีทางบันทึก
    # credential ผิดเว็บ ข้าม site_manuals กันได้ — ป้องกันสองชั้น: ชั้นแรกคือ crawl_site()
    # เอง generate domain นี้จาก extract_domain(start_url) ตรงๆ ไม่มีทางเป็นโดเมนอื่นอยู่
    # แล้ว เพราะกรอง nav link/ปุ่มที่พาออกนอกโดเมนทิ้งไปหมดตั้งแต่ต้น ชั้นที่สองคือ assert
    # ตรงนี้ กันไว้เผื่อ crawler เปลี่ยนพฤติกรรมในอนาคตแล้วลืมรักษา invariant นี้)
    target_domain = extract_domain(req.url)

    async def _on_credentials_needed(domain: str) -> Optional[dict]:
        assert domain == target_domain, (
            f"crawl job ของ {target_domain!r} เรียกขอ credential ของ {domain!r} ผิดเว็บ — "
            "ห้ามบันทึก/ใช้ credential ข้าม site_manuals เด็ดขาด"
        )
        creds = await learn_manager.request_credentials(
            learn_id, domain, timeout=settings.approval_timeout_seconds,
        )
        if creds:
            save_credentials(domain, creds["username"], creds["password"])
        return creds

    # W18: ถ้าผู้ใช้เลือก "ใช้บัญชีที่บันทึกไว้" บน UI แทนการกรอกใหม่ (req.username/password
    # จะเป็น None ทั้งคู่ในเคสนี้ — frontend ไม่มีทางส่งรหัสผ่านจริงที่ backend เก็บไว้แล้ว
    # กลับมาซ้ำได้อยู่แล้ว) โหลด credential ที่เก็บไว้ของโดเมนนี้มาใช้ login bootstrap แทน
    resolved_username, resolved_password = req.username, req.password
    if req.use_saved_credentials and not (resolved_username and resolved_password):
        saved_creds = load_credentials(extract_domain(req.url))
        if saved_creds:
            resolved_username, resolved_password = saved_creds["username"], saved_creds["password"]

    async def _run() -> dict:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(headless=False)
            try:
                manual = await crawl_site(
                    browser,
                    req.url,
                    provider=req.provider,
                    on_progress=_on_progress,
                    username=resolved_username,
                    password=resolved_password,
                    on_credentials_needed=_on_credentials_needed,
                )
            finally:
                await browser.close()
        version = save_manual(manual)
        # W17: ถ้ามี username/password ที่ใช้ login bootstrap จริง (ไม่ว่าจะกรอกใหม่หรือ
        # ดึงจาก credential เดิม) เขียนทับ credentials.json ของโดเมนนี้ไว้เหมือนเดิม — เขียน
        # ทับค่าเดิมด้วยค่าเดิมก็ไม่มีผลเสียอะไร (idempotent)
        if resolved_username and resolved_password:
            save_credentials(manual.website, resolved_username, resolved_password)
        # W24: ส่ง errors_count กลับไปด้วย — ให้ frontend โชว์ว่า crawl จบพร้อมปัญหาที่เจอ
        # ระหว่างทางกี่รายการ (ดู manual.errors — SiteManual.to_dict() มีรายละเอียดเต็ม
        # อยู่แล้วถ้าต้องการขุดดูทีหลัง ไม่ส่งรายละเอียดเต็มมาที่นี่เพราะ event นี้แค่สรุปผล)
        # W26: ส่ง summary (ภาพรวม "เว็บไซต์นี้ทำอะไรได้บ้าง" จาก describe_site()) กลับไปด้วย
        # ให้ frontend โชว์ทันทีที่เรียนรู้เสร็จ (ดู index.html::renderManualBanner)
        return {
            "version": version, "pages_found": len(manual.pages),
            "errors_count": len(manual.errors), "summary": manual.summary,
        }

    record = learn_manager.submit(learn_id, req.url, _run())
    return LearnCreatedResponse(learn_id=record.learn_id, status=record.status)


@router.get("/api/site-manual/learn/{learn_id}/stream")
async def stream_learn(learn_id: str, request: Request) -> StreamingResponse:
    """SSE ของ crawl job นี้ — "page_done" ทีละหน้าที่ crawl ผ่าน + "learn_done" ปิดท้าย
    (เหมือน stream_task() เกือบทุกประการแค่ event kind ต่างกัน) W23: เพิ่ม
    "credentials_needed"/"credentials_timeout" เข้ามาแล้ว — human-in-the-loop จริงๆ
    เหมือน approval_request ของ task ปกติ (ดู POST .../credentials ด้านล่าง)"""
    learn_manager: LearnManager = request.app.state.learn_manager
    record = learn_manager.get(learn_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"ไม่พบ learn_id: {learn_id!r}")

    async def event_gen():
        if record.status != "running":
            done_event = {
                "kind": "learn_done", "status": record.status,
                "result": record.result, "error": record.error,
            }
            yield f"data: {json.dumps(done_event)}\n\n"
            return
        while True:
            event = await record.events.get()
            yield f"data: {json.dumps(event)}\n\n"
            if event.get("kind") == "learn_done":
                break

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@router.post("/api/site-manual/learn/{learn_id}/credentials", status_code=204)
async def respond_learn_credentials(
    learn_id: str, req: LearnCredentialsRequest, request: Request,
) -> None:
    """W23: ตอบ "credentials_needed" event (ดู stream_learn() ด้านบน) — ปลดล็อก
    crawl_site() ที่กำลังรอ (await) อยู่ใน on_credentials_needed callback ของ
    learn_site() ให้ไปต่อได้ ปล่อย username/password ว่างทั้งคู่ = ผู้ใช้เลือกข้าม
    ("ไม่ต้อง login") คืน 404 ถ้า request_id ไม่ตรง/หมดอายุ/ตอบไปแล้ว (ดู
    LearnManager.resolve_credentials())"""
    learn_manager: LearnManager = request.app.state.learn_manager
    ok = learn_manager.resolve_credentials(learn_id, req.request_id, req.username, req.password)
    if not ok:
        raise HTTPException(status_code=404, detail=f"ไม่พบ request_id: {req.request_id!r} (หมดอายุ/ตอบไปแล้ว)")


@router.post("/api/site-manual/{domain}/relearn-page", response_model=RelearnPageResponse)
async def relearn_page(domain: str, req: RelearnPageRequest, request: Request) -> RelearnPageResponse:
    """Selector-repair (สเปค: "หาก Selector ใช้งานไม่ได้ ให้สำรวจเฉพาะหน้านั้น อัปเดต
    Version ไม่ต้องสร้าง Manual ใหม่ทั้งหมด") — สำรวจแค่หน้าเดียว (req.url) ใหม่ แทนที่จะ
    crawl ทั้งเว็บซ้ำ ต้องมี manual ของโดเมนนี้อยู่แล้วจาก POST /api/site-manual/learn
    มาก่อน (ไม่งั้นคืน 404 — ยังไม่รู้จักโครงสร้างเว็บนี้เลยสักหน้า จะ "ซ่อม" หน้าเดียว
    ไม่ได้) เป็น request แบบ sync ตรงๆ (ไม่ผ่าน LearnManager) เพราะสำรวจแค่หน้าเดียว เร็ว
    พอที่จะรอผลตรงๆ ได้ ไม่ต้อง submit-then-poll เหมือน crawl เต็มเว็บ"""
    if not manual_exists(domain):
        raise HTTPException(
            status_code=404,
            detail=f"ยังไม่มี manual ของ {domain!r} — ต้องเรียนรู้เว็บไซต์ทั้งหมดก่อน (POST /api/site-manual/learn)",
        )
    pool = request.app.state.browser_pool
    resolved_provider = req.provider or settings.llm_provider
    client, model, _, _, _ = Orchestrator._llm_backend(resolved_provider)

    async with pool.acquire() as browser:
        context = await browser.new_context()
        page = await context.new_page()
        try:
            await page.goto(req.url, timeout=15000)
            await page.wait_for_load_state("networkidle", timeout=8000)
            page_info, _ = await extract_page(page)
            page_info.name, page_info.description = await describe_page(client, model, resolved_provider, page_info)
            page_info.menu_path = page_info.menu_path or [page_info.name]
        finally:
            await context.close()

    version = update_single_page(domain, page_info)
    if version is None:
        raise HTTPException(status_code=404, detail=f"ยังไม่มี manual ของ {domain!r}")
    return RelearnPageResponse(version=version)


@router.post("/api/site-manual/{domain}/credentials", status_code=204)
async def save_site_credentials(domain: str, req: SaveCredentialsRequest) -> None:
    """W17: บันทึก/แก้ไข username-password ของโดเมนนี้ตรงๆ โดยไม่ต้อง crawl ทั้งเว็บใหม่
    (ต่างจาก POST /api/site-manual/learn ที่บันทึกให้อัตโนมัติเป็นผลพลอยได้จาก login
    bootstrap) — ไม่คืนค่า credential กลับเลย (204 เปล่าๆ) กันหลุดไปอยู่ใน response
    log/network tab โดยไม่จำเป็น"""
    save_credentials(domain, req.username, req.password)


@router.get("/api/site-manual/{domain}/credentials/status", response_model=CredentialsStatusResponse)
async def site_credentials_status(domain: str) -> CredentialsStatusResponse:
    """เช็คว่ามี credential เก็บไว้ให้โดเมนนี้ไหม — ไม่คืนค่า username/password จริงกลับมา
    เลย (แค่ exists: bool) กันไม่ให้ frontend/log ที่ไหนโชว์รหัสผ่านที่เก็บไว้แล้วออกมาซ้ำ"""
    return CredentialsStatusResponse(exists=credentials_exist(domain))


@router.delete("/api/site-manual/{domain}/credentials", status_code=204)
async def delete_site_credentials(domain: str) -> None:
    """ลบ credential ที่เก็บไว้ของโดเมนนี้ทิ้ง — ไม่ error ถ้าไม่มีอยู่แล้ว (idempotent)"""
    delete_credentials(domain)

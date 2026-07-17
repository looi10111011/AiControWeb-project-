"""W10[A]: API endpoints ที่ขับ Orchestrator จริงผ่าน BrowserPool แทน CLI (run.py)

Task submission เป็นแบบ async (submit -> 202 + task_id -> poll GET /tasks/{id}) ไม่ใช่
sync request-response ตรงๆ เพราะ run_task() ใช้เวลานาน (หลาย step, เรียก LLM จริงทุก
step) ดู task_manager.py สำหรับเหตุผลเต็มๆ
"""

from fastapi import APIRouter, HTTPException, Request

from backend.app.api.schemas import (
    CreateTaskRequest,
    PoolStatusResponse,
    TaskCreatedResponse,
    TaskStatusResponse,
)
from backend.app.core.orchestrator import Orchestrator

router = APIRouter()


async def _deny_all(cmd: dict) -> bool:
    """ask_user_func เริ่มต้นตอนเรียกผ่าน API (auto_approve=False) — ไม่มี human อยู่
    หน้าจอคอยตอบ REST request ตรงๆ เลย fail closed ปฏิเสธ action ที่ต้องขออนุมัติเสมอ
    (ดูเหตุผลเต็มใน schemas.py::CreateTaskRequest.auto_approve)"""
    return False


async def _approve_all(cmd: dict) -> bool:
    return True


@router.post("/tasks", response_model=TaskCreatedResponse, status_code=202)
async def create_task(req: CreateTaskRequest, request: Request) -> TaskCreatedResponse:
    pool = request.app.state.browser_pool
    task_manager = request.app.state.task_manager
    orchestrator = Orchestrator()
    ask_user_func = _approve_all if req.auto_approve else _deny_all

    async def _run() -> dict:
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
            )

    record = task_manager.submit(req.url, req.goal, req.provider, _run())
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


@router.get("/pool/status", response_model=PoolStatusResponse)
async def pool_status(request: Request) -> PoolStatusResponse:
    pool = request.app.state.browser_pool
    return PoolStatusResponse(size=pool.size, available=pool.available, in_use=pool.size - pool.available)

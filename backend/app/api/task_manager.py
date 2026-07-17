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
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Coroutine, Optional


@dataclass
class TaskRecord:
    task_id: str
    url: str
    goal: str
    provider: Optional[str]
    status: str  # "running" | "done" | "error"
    created_at: float = field(default_factory=time.time)
    result: Optional[dict] = None
    error: Optional[str] = None


class TaskManager:
    def __init__(self) -> None:
        self._tasks: dict[str, TaskRecord] = {}
        self._running: set[asyncio.Task] = set()

    def get(self, task_id: str) -> Optional[TaskRecord]:
        return self._tasks.get(task_id)

    def list(self) -> list[TaskRecord]:
        return sorted(self._tasks.values(), key=lambda t: t.created_at, reverse=True)

    def submit(
        self, url: str, goal: str, provider: Optional[str], coro: Coroutine[Any, Any, dict]
    ) -> TaskRecord:
        """สร้าง TaskRecord สถานะ "running" ทันที แล้วสั่งรัน coro (โดยทั่วไปคือ
        Orchestrator.run_task() ที่ห่อด้วย BrowserPool.acquire() ดู routes.py) เป็น
        background — ไม่ await ตรงนี้ คืน record กลับทันทีให้ endpoint ส่ง response 202"""
        record = TaskRecord(task_id=str(uuid.uuid4()), url=url, goal=goal, provider=provider, status="running")
        self._tasks[record.task_id] = record
        task = asyncio.create_task(self._run(record, coro))
        self._running.add(task)
        task.add_done_callback(self._running.discard)
        return record

    async def _run(self, record: TaskRecord, coro: Coroutine[Any, Any, dict]) -> None:
        try:
            record.result = await coro
            record.status = "done"
        except Exception as e:
            record.error = str(e)
            record.status = "error"

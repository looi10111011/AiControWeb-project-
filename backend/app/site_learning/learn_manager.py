"""site_learning/learn_manager.py — W14: registry ของ crawl job ที่ยิงผ่าน
POST /api/site-manual/learn — mirror รูปแบบเดียวกับ api/task_manager.py::TaskManager
(submit แล้วคืน learn_id ทันที ไม่รอ crawl จบ + poll/SSE ทีหลัง) แต่เป็นคลาสแยกต่างหาก
ไม่ใช้ TaskManager ร่วมกัน เพราะ crawl job มี field/lifecycle ไม่เหมือนกัน (page-checklist
progress ธรรมดา ไม่มี concept ของ human-in-the-loop approval/pending request เหมือน
task ปกติเลย — ผูก field พวกนั้นเข้าไปจะเกินความจำเป็นและสับสน)
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Coroutine, Optional


@dataclass
class LearnRecord:
    learn_id: str
    url: str
    status: str  # "running" | "done" | "error" | "cancelled"
    created_at: float = field(default_factory=time.time)
    result: Optional[dict] = None  # {"version": int, "pages_found": int} ตอนจบสำเร็จ
    error: Optional[str] = None
    asyncio_task: Optional[asyncio.Task] = None
    # ผู้บริโภคเดียวที่คาดหวังไว้คือ SSE connection เดียวต่อ crawl (เหมือน
    # task_manager.py::TaskRecord.events) ไม่ใช่ pub-sub หลายคน
    events: asyncio.Queue = field(default_factory=asyncio.Queue)


class LearnManager:
    def __init__(self) -> None:
        self._records: dict[str, LearnRecord] = {}
        self._running: set[asyncio.Task] = set()

    def get(self, learn_id: str) -> Optional[LearnRecord]:
        return self._records.get(learn_id)

    def new_learn_id(self) -> str:
        return str(uuid.uuid4())

    def submit(self, learn_id: str, url: str, coro: Coroutine[Any, Any, dict]) -> LearnRecord:
        """สร้าง LearnRecord สถานะ "running" ทันที แล้วสั่งรัน coro (โดยทั่วไปคือ
        crawl_site() ที่ห่อด้วย BrowserPool.acquire() — ดู routes.py) เป็น background —
        ไม่ await ตรงนี้ คืน record กลับทันทีให้ endpoint ส่ง response 202"""
        record = LearnRecord(learn_id=learn_id, url=url, status="running")
        self._records[learn_id] = record
        task = asyncio.create_task(self._run(record, coro))
        record.asyncio_task = task
        self._running.add(task)
        task.add_done_callback(self._running.discard)
        return record

    async def _run(self, record: LearnRecord, coro: Coroutine[Any, Any, dict]) -> None:
        try:
            record.result = await coro
            record.status = "done"
        except asyncio.CancelledError:
            record.status = "cancelled"
            record.error = "หยุดโดยผู้ใช้"
        except Exception as e:
            record.error = str(e)
            record.status = "error"
        # sentinel เดียวที่บอก SSE consumer ว่า stream จบแล้ว (เหมือน
        # task_manager.py::TaskManager._run() ที่ทำแบบเดียวกันทุกประการ)
        await record.events.put({
            "kind": "learn_done", "status": record.status,
            "result": record.result, "error": record.error,
        })

    async def push_event(self, learn_id: str, event: dict) -> None:
        record = self._records.get(learn_id)
        if record is not None:
            await record.events.put(event)

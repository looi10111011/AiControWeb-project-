"""site_learning/learn_manager.py — W14: registry ของ crawl job จาก POST /api/site-manual/learn.

Mirror TaskManager (submit คืน learn_id ทันที แล้ว poll/SSE) แต่แยกคลาสเพราะ lifecycle ต่างกัน (ไม่มี approval).
W23: request_credentials()/resolve_credentials() — crawler เจอหน้า login ที่ไม่มี credential ต้องหยุดถาม user
(mirror TaskManager.request_approval()/resolve_approval() แต่ payload เป็น {username, password} หรือ None)
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
    # ผู้บริโภคเดียว: SSE connection เดียวต่อ crawl (เหมือน TaskRecord.events) ไม่ใช่ pub-sub
    events: asyncio.Queue = field(default_factory=asyncio.Queue)
    # W23: request_id -> {"future": Future[Optional[dict]], "domain": str, "delivered": bool}
    # โครงสร้างเดียวกับ TaskRecord.pending (ดู task_manager.py)
    pending: dict = field(default_factory=dict)


class LearnManager:
    def __init__(self) -> None:
        self._records: dict[str, LearnRecord] = {}
        self._running: set[asyncio.Task] = set()

    def get(self, learn_id: str) -> Optional[LearnRecord]:
        return self._records.get(learn_id)

    def new_learn_id(self) -> str:
        return str(uuid.uuid4())

    def submit(self, learn_id: str, url: str, coro: Coroutine[Any, Any, dict]) -> LearnRecord:
        """สร้าง record "running" แล้วรัน coro เป็น background (ไม่ await) — คืนทันทีให้ endpoint ตอบ 202"""
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
        # W23: resolve pending credential request ที่ค้าง (crawl พัง/ถูก stop ระหว่างรอ user) กัน Future ค้างตลอดกาล
        for info in record.pending.values():
            if not info["future"].done():
                info["future"].set_result(None)
        # sentinel เดียวที่บอก SSE consumer ว่า stream จบแล้ว
        await record.events.put({
            "kind": "learn_done", "status": record.status,
            "result": record.result, "error": record.error,
        })

    async def push_event(self, learn_id: str, event: dict) -> None:
        record = self._records.get(learn_id)
        if record is not None:
            await record.events.put(event)

    async def request_credentials(
        self, learn_id: str, domain: str, timeout: Optional[float] = None,
    ) -> Optional[dict]:
        """push "credentials_needed" แล้วรอ (block แค่ crawl coroutine นี้) จน resolve_credentials().
        คืน None ถ้าไม่มี record/หมดเวลา/user ข้าม — caller ต้องรับ None ได้ (= ไปต่อโดยไม่ login).
        domain: โดเมนของเว็บที่ job นี้เรียนรู้ (extract_domain(start_url)) โชว์ให้ user รู้ว่ากรอกรหัสของเว็บไหน"""
        record = self._records.get(learn_id)
        if record is None:
            return None
        request_id = str(uuid.uuid4())
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        record.pending[request_id] = {"future": future, "domain": domain, "delivered": False}
        await record.events.put({"kind": "credentials_needed", "request_id": request_id, "domain": domain})
        try:
            if timeout is None:
                return await future
            try:
                return await asyncio.wait_for(future, timeout=timeout)
            except asyncio.TimeoutError:
                await record.events.put({
                    "kind": "credentials_timeout", "request_id": request_id, "domain": domain,
                })
                return None
        finally:
            record.pending.pop(request_id, None)

    def resolve_credentials(
        self, learn_id: str, request_id: str, username: Optional[str], password: Optional[str],
    ) -> bool:
        """คืน False ถ้าไม่พบ request ที่รออยู่ (endpoint ตอบ 404). username/password ไม่ครบ = user ข้าม
        (resolve เป็น None ให้ crawl ไปต่อโดยไม่ login)"""
        record = self._records.get(learn_id)
        if record is None:
            return False
        info = record.pending.get(request_id)
        if info is None or info["future"].done():
            return False
        if username and password:
            info["future"].set_result({"username": username, "password": password})
        else:
            info["future"].set_result(None)
        return True

"""site_learning/learn_manager.py — W14: registry ของ crawl job ที่ยิงผ่าน
POST /api/site-manual/learn — mirror รูปแบบเดียวกับ api/task_manager.py::TaskManager
(submit แล้วคืน learn_id ทันที ไม่รอ crawl จบ + poll/SSE ทีหลัง) แต่เป็นคลาสแยกต่างหาก
ไม่ใช้ TaskManager ร่วมกัน เพราะ crawl job มี field/lifecycle ไม่เหมือนกัน (page-checklist
progress ธรรมดา ไม่มี concept ของ human-in-the-loop approval/pending request เหมือน
task ปกติเลย — ผูก field พวกนั้นเข้าไปจะเกินความจำเป็นและสับสน)

W23: เพิ่ม pending/request_credentials()/resolve_credentials() เข้ามาแล้ว — crawler.py
เจอหน้า login ระหว่าง crawl ที่ยังไม่มี username/password ให้เลยต้อง "หยุดรอ" ถามคนจริง
ก่อนไปต่อได้ (ดู crawler.py::crawl_site() พารามิเตอร์ on_credentials_needed) mirror
TaskManager.request_approval()/resolve_approval() เกือบทุกประการ ต่างแค่ payload ที่ resolve
กลับมาเป็น dict {username, password} (หรือ None ถ้าข้าม) แทน bool
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
    # W23: request_id -> {"future": Future[Optional[dict]], "domain": str, "delivered": bool}
    # ของคำขอ credential ที่ยังรอ user ตอบอยู่ — โครงสร้างเดียวกับ
    # TaskRecord.pending ทุกประการ (ดู task_manager.py สำหรับเหตุผลเต็มๆ ของแต่ละ field)
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
        # W23: ยกเลิก pending credential request ที่ยังค้างอยู่ (เช่น crawl ล้มเหลว/ถูก stop
        # กลางคันระหว่างรอ user กรอก username/password) กัน Future ค้างไม่มีใครมา resolve ไป
        # ตลอดกาล (เหมือน TaskManager._run() ทุกประการ)
        for info in record.pending.values():
            if not info["future"].done():
                info["future"].set_result(None)
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

    async def request_credentials(
        self, learn_id: str, domain: str, timeout: Optional[float] = None,
    ) -> Optional[dict]:
        """เรียกจาก on_credentials_needed callback (routes.py) — push event
        "credentials_needed" เข้า stream ของ crawl job นี้ (โชว์ฟอร์มกรอก username/
        password บนหน้าเว็บ) แล้วรอ (block เฉพาะ crawl coroutine นี้ ไม่บล็อก event loop
        รวม — เหมือน TaskManager.request_approval() ทุกประการ) จนกว่า resolve_credentials()
        จะถูกเรียก (จาก POST .../credentials) คืน None ถ้าไม่มี record นี้อยู่แล้ว/หมดเวลา/
        user เลือกข้าม — ผู้เรียก (crawl_site()) ต้องรับมือกับ None ได้เสมอ (แปลว่า "ไปต่อ
        โดยไม่ login" ไม่ใช่ error)

        domain: ต้องเป็นโดเมนของเว็บที่ crawl job นี้กำลังเรียนรู้อยู่จริงเท่านั้น (มาจาก
        extract_domain(start_url) ใน crawl_site() ตรงๆ ไม่มีทางเป็นโดเมนอื่น เพราะ crawler
        กรอง nav link/ปุ่มที่พาออกนอกโดเมนทิ้งไปตั้งแต่ต้นแล้ว) — ส่งไปให้ event เห็นด้วย
        เพื่อโชว์บนหน้าเว็บว่ากำลังขอ credential ของเว็บไหน กัน user สับสนว่าจะกรอกรหัสผ่าน
        ของเว็บไหนกันแน่ ถ้ามีหลาย tab/learn job รันพร้อมกัน"""
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
        """เรียกจาก POST /api/site-manual/learn/{learn_id}/credentials — คืน False ถ้าไม่
        พบ request_id นี้แล้ว (ตอบไปแล้ว/หมดอายุ/learn_id ผิด) ให้ endpoint คืน 404 ต่อ —
        username/password ว่างทั้งคู่ (None) = user เลือกข้าม ส่ง None ให้ crawl_site()
        ไปต่อโดยไม่ login แทน"""
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

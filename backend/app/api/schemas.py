from typing import Optional

from pydantic import BaseModel, Field


class CreateTaskRequest(BaseModel):
    url: str
    goal: str
    max_steps: int = 30
    provider: Optional[str] = None  # None = ใช้ settings.llm_provider (ดู orchestrator.py)
    headless: Optional[bool] = None  # None = ใช้ settings.browser_headless
    # W10[A]: ไม่มี human อยู่หน้าจอคอยตอบ permission prompt ผ่าน REST ตรงๆ (ต่างจาก
    # run.py ที่ถาม terminal ได้) — ค่าเริ่มต้น False = action ที่ต้องขออนุมัติ (ดู
    # permission/rules.py) จะถูก "ปฏิเสธ" อัตโนมัติเสมอ (fail closed ปลอดภัยกว่า) ไม่ใช่
    # เงียบๆ อนุมัติให้เอง — ถ้าอยากให้ agent ทำ action พวกนี้ได้ ต้องส่ง true มาเอง
    # (รับผิดชอบเองว่าไม่มี human-in-the-loop จริงๆ ระหว่างรอบนี้)
    auto_approve: bool = False
    # ให้ LLM ร่างแผนคร่าวๆ ก่อนเริ่ม loop จริง (orchestrator.py::run_task) แล้วเก็บไว้ใน
    # result["plan"] ให้ console UI แสดงเป็น panel "Plan" — ค่า default True เพราะแค่
    # "โชว์แผน" ไม่ใช่ permission-gated action (ดู routes.py::_make_ask_user_func ที่
    # auto-approve confirm_plan เสมอ แยกจาก auto_approve ที่คุม action จริงบนหน้าเว็บ)
    confirm_plan: bool = True


class TaskCreatedResponse(BaseModel):
    task_id: str
    status: str


class TaskStatusResponse(BaseModel):
    task_id: str
    url: str
    goal: str
    provider: Optional[str]
    status: str
    created_at: float
    result: Optional[dict] = None
    error: Optional[str] = None


class PoolStatusResponse(BaseModel):
    size: int
    available: int
    in_use: int


class RespondRequest(BaseModel):
    """W10[B]: body ของ POST /tasks/{id}/respond — ผูกกับ approval_request event ที่
    ได้จาก GET /tasks/{id}/stream (request_id ต้องตรงกับที่ event ส่งมา ไม่งั้นถือว่า
    หมดอายุ/ตอบไปแล้ว ดู TaskManager.resolve_approval())"""

    request_id: str
    approved: bool
    # W10[F]: ถ้า request นี้เป็น confirm_plan (ไม่ใช่ permission prompt ของ action ทั่วไป)
    # user แก้ไขข้อความแผนเองก่อนกด Confirm ได้ — ส่งมาก็ต่อเมื่อมีการแก้ไขจริง (None =
    # ใช้แผนเดิมที่ AI ร่างไว้ไม่แก้) ดู TaskManager.resolve_approval()/orchestrator.py::
    # _confirm_plan() สำหรับตำแหน่งที่ใช้ค่านี้จริง
    edited_plan: Optional[str] = None

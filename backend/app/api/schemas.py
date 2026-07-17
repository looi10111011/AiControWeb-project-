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

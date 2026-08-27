"""core/orangehrm_eval.py — เหมือน core/evaluation.py (W12[B]) ทุกประการ แค่เปลี่ยน
url/tasks ไปเป็น OrangeHRM public demo instance แทน saucedemo.com — reuse
run_evaluation()/EvaluationReport/TaskEvalResult เดิมตรงๆ (parameterized ไว้แล้วสำหรับ
url/tasks ทั้งคู่ — ดู evaluation.py::run_evaluation()) ไม่ต้อง duplicate โครง harness ใหม่
เลย ต่างจาก core/miniwob_eval.py ที่ต้องมี Playwright plumbing ของตัวเอง (seeded
determinism, episode-timer override ฯลฯ) เพราะ MiniWoB มี reward signal จาก JS global ที่
evaluation.py เดิมไม่รองรับ — เว็บทั่วไปแบบนี้ตัดสิน success/fail จาก finish_task() ของ
agent เอง เหมือน SauceDemo อยู่แล้วทุกประการ

*** สถานะชั่วคราว: ใช้ public demo instance นี้เพราะ Docker ยังใช้งานไม่ได้บนเครื่อง dev
ตอนนี้ (Windows 11 Home ไม่มี Hyper-V, WSL2 ยังไม่ได้ติดตั้ง — ต้อง `wsl --install` +
reboot ก่อนถึงจะรัน self-hosted OrangeHRM/PrestaShop ผ่าน Docker ได้จริง) ย้ายไป
self-hosted instance ทันทีที่ Docker พร้อมใช้งาน (ดู requirements.txt/README สำหรับ
docker-compose ที่ scope ไว้แล้ว: orangehrm/orangehrm image + mysql, ใช้
installer/console install:on-new-database ของ OrangeHRM เอง — ยังไม่ implement เพราะรอ
Docker daemon ให้พร้อมก่อน)

opensource-demo.orangehrmlive.com เป็น **shared public demo** (multi-tenant — คนอื่น
ทดสอบพร้อมกันได้จริง, เคยสังเกตเห็น session identity สลับกลางทาง, มี employee record ของ
คนอื่นปนอยู่แล้วเพียบ) — ไม่ใช่ instance ส่วนตัว จึงต้องเลือก task อย่างระมัดระวัง:
  - ห้าม task ที่ mutate credential (เปลี่ยนรหัสผ่าน) หรือทำลาย/แก้ไขข้อมูลของคนอื่น
    (bulk delete, แก้ user คนอื่น ฯลฯ) เด็ดขาด
  - เลือกเฉพาะ task ที่ "additive" (สร้าง record ใหม่ของตัวเอง ไม่กระทบของเดิม เช่น เพิ่ม
    candidate สมัครงาน) หรือ "read-only" (login, ค้นหาที่คาดผลลัพธ์ได้แน่นอน) เท่านั้น
  - success rate จะมี noise มากกว่าปกติ (คนอื่นอาจเปลี่ยนสถานะเว็บกลางคันระหว่าง task
    กำลังรัน) — ยอมรับได้สำหรับ smoke test ชั่วคราว ไม่ใช่ benchmark ที่แม่นยำเทียบเท่า
    instance ส่วนตัว

username/password ("Admin"/"admin123") เป็น credential ทดสอบสาธารณะที่ OrangeHRM เผยแพร่
เองบนหน้า login ของ demo นี้โดยตรง (ไม่ใช่ความลับ ไม่ใช่ของ user จริง)
"""

import time
from typing import Optional

from backend.app.core.evaluation import EvaluationReport, run_evaluation

_ORANGEHRM_URL = "https://opensource-demo.orangehrmlive.com/web/index.php/auth/login"
_ORANGEHRM_USERNAME = "Admin"
_ORANGEHRM_PASSWORD = "admin123"

_TASK_LOGIN_DASHBOARD = (
    f"Log in with username '{_ORANGEHRM_USERNAME}' and password '{_ORANGEHRM_PASSWORD}', "
    "and confirm you land on the Dashboard page."
)
# ค้นหาชื่อที่ไม่น่าจะมีอยู่จริงบนหน้า shared demo — คาดผลลัพธ์ได้แน่นอน (ไม่มี record)
# ไม่ว่า tester คนอื่นจะเพิ่ม/ลบพนักงานไปกี่คนก็ตาม ต่างจากการค้นหาชื่อที่ "คาดว่ามีอยู่"
# ซึ่งอาจถูกลบไปจริงโดย tester คนอื่นระหว่างทางได้
_TASK_SEARCH_NO_RESULTS = (
    f"Log in with username '{_ORANGEHRM_USERNAME}' and password '{_ORANGEHRM_PASSWORD}', "
    "go to PIM, open the Employee List, search for the employee name "
    "'Zzzznonexistent999', and confirm the system shows no matching records found."
)


def _add_candidate_task() -> dict:
    """W_orangehrm: goal ต้องสร้างใหม่ทุกครั้งที่เรียก (ไม่ใช่ constant string เหมือน task
    อื่นในไฟล์นี้) — ผูก timestamp เข้าไปในชื่อ/อีเมลกันชนกับ candidate ที่ tester คนอื่น/
    การรันรอบก่อนหน้าสร้างไว้แล้วบน shared demo instance เดียวกัน (record ซ้ำอาจทำให้
    ระบบปฏิเสธ/พฤติกรรมไม่แน่นอน)"""
    tag = str(int(time.time()))
    goal = (
        f"Log in with username '{_ORANGEHRM_USERNAME}' and password '{_ORANGEHRM_PASSWORD}', "
        f"go to Recruitment, click Add Candidate, fill in First Name 'AgentTest{tag}', "
        f"Last Name 'Bench{tag}', Email 'agenttest{tag}@example.com', then save. "
        "Confirm the save succeeded."
    )
    return {"name": "add_candidate", "goal": goal, "max_steps": 15}


# task ที่ goal คงที่ (deterministic string เดียวกันทุกครั้ง) — add_candidate ไม่ได้อยู่ใน
# ลิสต์นี้เพราะ goal ต้องสร้างใหม่ทุก run (ดู _add_candidate_task() ด้านบน)
ORANGEHRM_TASKS: list[dict] = [
    {"name": "login_dashboard", "goal": _TASK_LOGIN_DASHBOARD, "max_steps": 10},
    {"name": "search_no_results", "goal": _TASK_SEARCH_NO_RESULTS, "max_steps": 15},
]


async def run_orangehrm_evaluation(
    provider: Optional[str] = None, run_id: Optional[str] = None,
) -> EvaluationReport:
    """รัน ORANGEHRM_TASKS + add_candidate (goal สร้างใหม่ทุกครั้ง) ผ่าน run_evaluation()
    ตัวเดียวกับ SauceDemo เป๊ะ (evaluation.py) แค่เปลี่ยน url — ดู docstring หัวไฟล์สำหรับ
    ข้อจำกัดของการทดสอบบน shared public demo instance นี้"""
    tasks = [*ORANGEHRM_TASKS, _add_candidate_task()]
    # W_eval_trace: ส่ง run_id ต่อให้ run_evaluation() เฉยๆ — suite นี้ reuse harness เดิม
    # ทั้งก้อนอยู่แล้ว ไม่มี loop ของตัวเองให้ต้องเขียน telemetry ซ้ำ
    return await run_evaluation(
        tasks=tasks, provider=provider, url=_ORANGEHRM_URL, run_id=run_id,
    )

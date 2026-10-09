"""core/orangehrm_eval.py — evaluation.py (W12[B]) บน OrangeHRM public demo แทน saucedemo.com
reuse run_evaluation()/EvaluationReport ตรงๆ (success ตัดสินจาก finish_task() ของ agent เหมือน SauceDemo)

W_gate_local_hrm: release gate ไม่รัน suite นี้เป็นค่าเริ่มต้นแล้ว — แทนด้วย core/hrm_local_eval.py
เพราะเดโมสาธารณะที่คนอื่นแก้ข้อมูลร่วมทำให้ commit เดียวกันได้ผลต่างกัน (เปิดเองได้ด้วย
`python run.py orangehrm` หรือ include_orangehrm=True) เดิมใช้ public demo เพราะ Docker/WSL2 ยังใช้ไม่ได้

shared multi-tenant demo: ห้าม task ที่แก้ credential หรือทำลาย/แก้ข้อมูลคนอื่น เลือกแค่ additive
(สร้าง record ของตัวเอง) หรือ read-only และยอมรับ noise ที่สูงกว่า instance ส่วนตัว
"Admin"/"admin123" เป็น credential ทดสอบที่ OrangeHRM เผยแพร่เองบนหน้า login (ไม่ใช่ความลับ)
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
# ค้นชื่อที่ไม่มีจริง — ผลคาดได้แน่นอนไม่ว่า tester คนอื่นจะเพิ่ม/ลบพนักงาน (ชื่อที่ "มีอยู่" อาจโดนลบ)
_TASK_SEARCH_NO_RESULTS = (
    f"Log in with username '{_ORANGEHRM_USERNAME}' and password '{_ORANGEHRM_PASSWORD}', "
    "go to PIM, open the Employee List, search for the employee name "
    "'Zzzznonexistent999', and confirm the system shows no matching records found."
)


def _add_candidate_task() -> dict:
    """W_orangehrm: goal สร้างใหม่ทุกครั้ง — ผูก timestamp ในชื่อ/อีเมลกันชนกับ candidate ที่มีอยู่
    แล้วบน shared demo (record ซ้ำทำให้พฤติกรรมไม่แน่นอน)"""
    tag = str(int(time.time()))
    goal = (
        f"Log in with username '{_ORANGEHRM_USERNAME}' and password '{_ORANGEHRM_PASSWORD}', "
        f"go to Recruitment, click Add Candidate, fill in First Name 'AgentTest{tag}', "
        f"Last Name 'Bench{tag}', Email 'agenttest{tag}@example.com', then save. "
        "Confirm the save succeeded."
    )
    return {"name": "add_candidate", "goal": goal, "max_steps": 15}


# เฉพาะ task ที่ goal คงที่ — add_candidate สร้างใหม่ทุก run (_add_candidate_task())
ORANGEHRM_TASKS: list[dict] = [
    {"name": "login_dashboard", "goal": _TASK_LOGIN_DASHBOARD, "max_steps": 10},
    {"name": "search_no_results", "goal": _TASK_SEARCH_NO_RESULTS, "max_steps": 15},
]


async def run_orangehrm_evaluation(
    provider: Optional[str] = None, run_id: Optional[str] = None,
) -> EvaluationReport:
    """รัน ORANGEHRM_TASKS + add_candidate ผ่าน run_evaluation() (telemetry/run_id จัดการที่นั่น)"""
    tasks = [*ORANGEHRM_TASKS, _add_candidate_task()]
    return await run_evaluation(
        tasks=tasks, provider=provider, url=_ORANGEHRM_URL, run_id=run_id,
    )

"""core/evaluation.py — W12[B]: Evaluation harness แนว WebVoyager — วัด success rate,
จำนวน step, และ token ต่อ task จริงบน saucedemo.com รันผ่าน Orchestrator.run_task() ตรงๆ
(ไม่ผ่าน API/BrowserPool — เหมือน demo อื่นๆ ใน run.py) headless=True + auto_approve=True
+ confirm_plan=False เสมอ ให้รันจบเป็น batch โดยไม่ต้องมีคนเฝ้าหน้าจอตอบ approve/confirm

ชุด task benchmark (BENCHMARK_TASKS) ใช้ข้อความ goal เดิมเป๊ะที่นิยามไว้แล้วใน run.py (ไม่
สร้างใหม่ซ้ำความหมาย) เลือกเฉพาะภารกิจที่ "ควรสำเร็จได้จริงถ้า agent ทำงานถูก" — ไม่รวมเคส
ที่จงใจให้ action พังเสมอ (เช่น W7[A] Test Case A ที่ทดสอบ long-term memory ไม่ใช่ทดสอบ
ความสามารถทำงานสำเร็จ) หรือเคสที่ต้องรันสองรอบต่อเนื่องกัน (Test Case B ที่รอบ 2 พึ่งผลจาก
รอบ 1) — ครอบคลุมความยาว/ความซับซ้อนต่างกัน 3 ระดับ: สั้น (login + เปลี่ยนสินค้าใน cart +
checkout), กลาง (RAG-based permission gate, บูรณาการ 3 สมอง), ยาว (sort สองทิศทาง + ใส่ของ
3 ชิ้น + ลบ 1 ชิ้น + checkout เต็ม flow)
"""

from dataclasses import dataclass, field
from typing import Optional

from backend.app.core.orchestrator import Orchestrator

_SAUCEDEMO_URL = "https://www.saucedemo.com/"

# เดียวกับ run.py::_DEFAULT_AGENT_GOAL เป๊ะ
_TASK_LOGIN_CHECKOUT = (
    "Log in, add first product, change item to second product , and proceed to checkout"
)
# เดียวกับ run.py::_TEST_CASE_D_GOAL เป๊ะ (W7[B]: ทดสอบว่า RAG manual สั่งขออนุมัติก่อน
# Checkout ได้จริงแม้ type="click" ธรรมดาไม่ตรง hardcoded rule ไหนเลย)
_TASK_RAG_PERMISSION = (
    "Log in as standard_user/secret_sauce, add the first product to the cart, "
    "click the shopping cart icon to open the Cart page, then click the 'Checkout' button"
)
# เดียวกับ run.py::_W8_INTEGRATION_GOAL เป๊ะ (W8: บูรณาการ perception + RAG manual + memory)
_TASK_RAG_INTEGRATION = (
    "Log in as standard_user/secret_sauce, add the first product to the cart, "
    "click the shopping cart icon to open the Cart page, then click the 'Checkout' "
    "button to go to the checkout information page (a page with First Name, Last "
    "Name, and Zip Code input fields). Fill in First Name, Last Name, and Zip/Postal "
    "Code exactly according to the store's official policy manual — do not invent "
    "your own values, check the reference manual for the exact values required — "
    "then click Continue"
)
# เดียวกับ run.py::_TEST_CASE_C_GOAL เป๊ะ (W7[A]: ยาวพอให้เห็น token/context compaction)
_TASK_LONG_FLOW = (
    "Log in as standard_user/secret_sauce, sort products by name Z to A, then sort back "
    "to name A to Z, add the first three products to the cart one at a time, go to the "
    "cart page, remove one item, then proceed to checkout, fill First Name with 'Test', "
    "Last Name with 'User', Zip Code with '10110', click Continue, then click Finish"
)

BENCHMARK_TASKS: list[dict] = [
    {"name": "login_checkout", "goal": _TASK_LOGIN_CHECKOUT, "max_steps": 15},
    {"name": "rag_permission", "goal": _TASK_RAG_PERMISSION, "max_steps": 15},
    {"name": "rag_integration", "goal": _TASK_RAG_INTEGRATION, "max_steps": 20},
    {"name": "long_flow", "goal": _TASK_LONG_FLOW, "max_steps": 25},
]


@dataclass
class TaskEvalResult:
    name: str
    goal: str
    success: bool
    steps: int
    total_tokens: int
    message: str
    error: Optional[str] = None


@dataclass
class EvaluationReport:
    results: list[TaskEvalResult] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.success) / len(self.results)

    @property
    def avg_steps(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.steps for r in self.results) / len(self.results)

    @property
    def avg_tokens(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.total_tokens for r in self.results) / len(self.results)


async def run_evaluation(
    tasks: Optional[list[dict]] = None,
    provider: Optional[str] = None,
    url: str = _SAUCEDEMO_URL,
) -> EvaluationReport:
    """รัน task ทีละตัวตามลำดับ (ไม่ concurrent ผ่าน pool) เพราะอยากวัด step/token ต่อ task
    ให้ตรงไปตรงมา ไม่ปนกับ rate-limit/คิวรอ browser ว่างที่จะทำให้ตัวเลขต่อ task เพี้ยน —
    ผ่าน Orchestrator.run_task() ตรงๆ (headless=True, auto_approve=True, confirm_plan=
    False เสมอ, ไม่ผ่าน ask_user_func คน ให้รันจบเป็น batch ได้เอง)

    task ไหนที่ run_task() เอง throw exception ขึ้นมาจริง (เช่น browser launch พัง, LLM
    API error ที่ไม่ถูกจับใน orchestrator) ไม่ทำให้ทั้ง batch หยุด — บันทึกเป็น
    success=False, steps=0, total_tokens=0 พร้อม error message แล้วรัน task ถัดไปต่อ (กฎ
    เดียวกับ retriever.py/long_term_memory.py: ส่วนหนึ่งพังไม่ควรทำทั้ง evaluation รอบนี้
    พังตาม — อยากได้ผลลัพธ์ของ task ที่เหลือครบเท่าที่ทำได้)"""
    tasks = tasks if tasks is not None else BENCHMARK_TASKS
    report = EvaluationReport()
    for task in tasks:
        try:
            result = await Orchestrator().run_task(
                url, task["goal"],
                max_steps=task.get("max_steps", 20),
                headless=True, auto_approve=True, confirm_plan=False, provider=provider,
            )
            tokens = result["tokens"]
            total_tokens = tokens["input"] + tokens["output"] + tokens["cache_read"] + tokens["cache_creation"]
            report.results.append(TaskEvalResult(
                name=task["name"], goal=task["goal"], success=result["success"],
                steps=result["steps"], total_tokens=total_tokens, message=result["message"],
            ))
        except Exception as e:
            report.results.append(TaskEvalResult(
                name=task["name"], goal=task["goal"], success=False, steps=0,
                total_tokens=0, message="", error=f"{type(e).__name__}: {e}",
            ))
    return report

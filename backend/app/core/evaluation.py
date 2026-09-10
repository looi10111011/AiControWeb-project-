"""core/evaluation.py — W12[B]: Evaluation harness แนว WebVoyager — วัด success rate,
จำนวน step, และ token ต่อ task จริงบน saucedemo.com รันผ่าน Orchestrator.run_task() ตรงๆ
(ไม่ผ่าน API/BrowserPool — เหมือน demo อื่นๆ ใน run.py) headless=True + confirm_plan=
False + ask_user_func ที่ auto-approve ทุกอย่างเสมอ ให้รันจบเป็น batch โดยไม่ต้องมีคนเฝ้า
หน้าจอตอบ approve/confirm (ดู _auto_approve() ด้านล่าง)

ชุด task benchmark (BENCHMARK_TASKS) ใช้ข้อความ goal เดิมเป๊ะที่นิยามไว้แล้วใน run.py (ไม่
สร้างใหม่ซ้ำความหมาย) เลือกเฉพาะภารกิจที่ "ควรสำเร็จได้จริงถ้า agent ทำงานถูก" — ไม่รวมเคส
ที่จงใจให้ action พังเสมอ (เช่น W7[A] Test Case A ที่ทดสอบ long-term memory ไม่ใช่ทดสอบ
ความสามารถทำงานสำเร็จ) หรือเคสที่ต้องรันสองรอบต่อเนื่องกัน (Test Case B ที่รอบ 2 พึ่งผลจาก
รอบ 1) — ครอบคลุมความยาว/ความซับซ้อนต่างกัน 3 ระดับ: สั้น (login + เปลี่ยนสินค้าใน cart +
checkout), กลาง (RAG-based permission gate, บูรณาการ 3 สมอง), ยาว (sort สองทิศทาง + ใส่ของ
3 ชิ้น + ลบ 1 ชิ้น + checkout เต็ม flow)
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional

from backend.app.config import settings
from backend.app.core.orchestrator import Orchestrator
from backend.app.core.telemetry import (
    SOURCE_EVAL, new_run_id, write_step_trace, write_token_usage,
)

_SAUCEDEMO_URL = "https://www.saucedemo.com/"


def _make_counting_auto_approve():
    """W_eval (release-gate follow-up): เหมือน _auto_approve() เดิมทุกประการ (auto-approve
    ทุก action ที่ต้องขอยืนยัน) แค่นับจำนวนครั้งที่ถูกเรียกไปด้วย — ใช้เป็น approval_count
    ใน TaskEvalResult ด้านล่าง (วัด "ต้องขออนุมัติกี่ครั้ง" ต่อ task จริง ไม่ใช่เดา) คืน
    (ask_user_func, get_count) คู่กัน — get_count เป็น closure อ่านค่า ณ ตอนนั้น ไม่ใช่
    ค่า snapshot ตอนสร้าง"""
    count = 0

    async def _counting_auto_approve(cmd: dict) -> bool:
        nonlocal count
        count += 1
        return True

    return _counting_auto_approve, lambda: count


async def _auto_approve(cmd: dict) -> bool:
    """ask_user_func ที่ auto-approve ทุก action ที่ต้องขอยืนยัน (submit/delete/purchase/
    pay ฯลฯ) เสมอ — eval รันแบบ batch ไม่มีคนเฝ้าหน้าจอตอบจริง ถ้าไม่ส่ง ask_user_func
    เข้า run_task() เลย (ปล่อยเป็น None ค่า default) actions.py จะ fallback ไป blocking
    input() ทาง terminal ซึ่งไม่มีคนตอบเลยในบริบทนี้ — ค้างตลอดไปเงียบๆ ไม่ error ให้เห็น
    ด้วยซ้ำ (เจอจริงตอนรัน BENCHMARK_TASKS ที่มี action ต้องขออนุมัติ เช่น checkout)"""
    return True

# เดียวกับ run.py::_DEFAULT_AGENT_GOAL เป๊ะ
#
# W_ambiguous_benchmark_goal (วัดจาก 6 รันของ task นี้ 2026-09-08): ข้อความเดิมคือ "add
# first product, change item to second product" ซึ่งไม่ได้บอกว่าสินค้าชิ้นไหน — agent จึง
# เรียก request_user_input **ทั้ง 6 รอบ** เพื่อถามว่าหมายถึงอันไหน เสียไปหนึ่ง step ทุกครั้ง
# ในงบ max_steps=15 ที่ต้องกรอกฟอร์ม 3 ช่องอยู่แล้ว และรอบที่ล้มก็ล้มเพราะเดินไม่ทันงบ
#
# นี่คือการแก้ *เครื่องวัด* ไม่ใช่แก้ agent: โจทย์ที่กำกวมวัดความสามารถในการเดาใจ ไม่ใช่
# ความสามารถในการทำงานตามสั่ง ซึ่งไม่ใช่สิ่งที่ suite นี้ตั้งใจวัด (task อื่นทุกตัวใน
# ไฟล์นี้ระบุค่าที่ต้องใช้ชัดเจนอยู่แล้ว) เจตนาของ task เหมือนเดิมทุกประการ: หยิบชิ้นแรก
# เปลี่ยนเป็นชิ้นที่สอง แล้วไป checkout — แค่บอกชื่อสองชิ้นนั้นตรงๆ ตามลำดับ default (A-Z)
# ของ saucedemo
#
# ผลที่ตามมาที่ต้องรู้: ตัวเลขของ login_checkout เทียบกับรันก่อนหน้านี้ไม่ได้อีกต่อไป
# เพราะเป็นคนละโจทย์ (เหมือนตอนที่ MiniWoB เปลี่ยนมาใช้ seed คงที่)
_TASK_LOGIN_CHECKOUT = (
    "Log in, add 'Sauce Labs Backpack' to the cart, then swap it for 'Sauce Labs Bike Light' (remove the backpack, add the bike light), and proceed to checkout"
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
    # W_eval (release-gate follow-up, ดู core/release_gate.py): 5 field ใหม่ต่อจากนี้
    latency_seconds: float = 0.0
    # llm_calls เป็นค่าประมาณ ไม่ใช่ตัวนับจริงทีละ LLM API call (run_task() ไม่มี counter
    # แบบนั้นให้ดึงตรงๆ) — fastpath task ที่ไม่ต้อง repair เลยไม่มี LLM call จริงสักครั้ง (ใช้
    # repairs แทน steps เพราะ steps ของ fastpath คือ "จำนวน step ที่ replay" ไม่ใช่ LLM call)
    # ส่วน slow-path (execution_mode ว่างเปล่า/ไม่ใช่ fastpath*) ใช้ steps ตรงๆ (แต่ละ step
    # ผูกกับ next_action() หนึ่งครั้งโดยประมาณ — คลาดเคลื่อนได้บ้างจาก nudge/retry guard ที่
    # เรียก next_action() ซ้ำโดยไม่เพิ่ม steps_taken เสมอไป แต่เป็นค่าประมาณที่ดีที่สุดที่ทำ
    # ได้โดยไม่ต้องเพิ่ม instrumentation ใหม่ใน orchestrator.py's main loop)
    llm_calls: int = 0
    approval_count: int = 0
    fastpath: bool = False
    recoveries: int = 0
    # W_gate_task_level_diff: ตัวนับจริงจาก run_task() (ไม่ใช่ค่าประมาณแบบ llm_calls
    # ด้านบน) — เก็บรายตัวเพื่อให้เทียบ task ต่อ task ได้ ไม่ใช่แค่ค่าเฉลี่ยรวม
    # ซึ่งกลบความต่างของแต่ละงานจนอ่านไม่ออกว่าอะไรเปลี่ยน
    action_calls: int = 0
    finish_task_calls: int = 0


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

    # W_eval (release-gate follow-up) — 5 property ใหม่ต่อจากนี้ ทุกอันคืน 0.0 เงียบๆ ถ้า
    # ไม่มี results เลย (เหมือน property เดิมด้านบนทุกประการ)

    def _latency_percentile(self, pct: float) -> float:
        """percentile คำนวณแบบ nearest-rank ธรรมดา (ไม่ interpolate) — พอสำหรับจำนวน task
        ต่อ batch ที่มีจริง (ระดับสิบ ไม่ใช่ระดับพัน ที่การ interpolate จะมีผลชัดเจนกว่านี้)"""
        if not self.results:
            return 0.0
        latencies = sorted(r.latency_seconds for r in self.results)
        idx = min(int(len(latencies) * pct) if pct < 1.0 else len(latencies) - 1, len(latencies) - 1)
        return latencies[idx]

    @property
    def p50_latency_seconds(self) -> float:
        return self._latency_percentile(0.5)

    @property
    def p95_latency_seconds(self) -> float:
        return self._latency_percentile(0.95)

    @property
    def avg_llm_calls(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.llm_calls for r in self.results) / len(self.results)

    @property
    def approval_rate(self) -> float:
        """approval_count เฉลี่ยต่อ task (ไม่ใช่ "สัดส่วน task ที่ต้องขออนุมัติ") — ชื่อ
        "_rate" ตามชื่อ metric ในสเปคเดิม (roadmap: "จำนวนครั้งที่ต้องขออนุมัติ") แต่ความ
        หมายจริงคือค่าเฉลี่ยจำนวนครั้ง"""
        if not self.results:
            return 0.0
        return sum(r.approval_count for r in self.results) / len(self.results)

    @property
    def fastpath_hit_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.fastpath) / len(self.results)

    @property
    def recovery_rate(self) -> float:
        """ในบรรดา task ที่ผ่าน fast-path จริง (fastpath=True) — สัดส่วนที่ "ต้องพึ่ง
        Repair อย่างน้อย 1 ครั้ง" (recoveries>0) แล้ว "ยังสำเร็จอยู่ดี" เทียบกับ task
        fast-path ที่ต้องพึ่ง Repair ทั้งหมด (ไม่ว่าจะสำเร็จหรือ escalate ไปสุดท้าย) — วัด
        "เมื่อ self-heal จำเป็น มันช่วยรอด task ได้บ่อยแค่ไหนจริงๆ" คืน 0.0 ถ้าไม่มี task
        ไหนต้องพึ่ง Repair เลย (ไม่มีอะไรให้วัด ไม่ใช่ "recovery ล้มเหลว 100%")"""
        needed_recovery = [r for r in self.results if r.fastpath and r.recoveries > 0]
        if not needed_recovery:
            return 0.0
        return sum(1 for r in needed_recovery if r.success) / len(needed_recovery)


async def run_evaluation(
    tasks: Optional[list[dict]] = None,
    provider: Optional[str] = None,
    url: str = _SAUCEDEMO_URL,
    run_id: Optional[str] = None,
) -> EvaluationReport:
    """รัน task ทีละตัวตามลำดับ (ไม่ concurrent ผ่าน pool) เพราะอยากวัด step/token ต่อ task
    ให้ตรงไปตรงมา ไม่ปนกับ rate-limit/คิวรอ browser ว่างที่จะทำให้ตัวเลขต่อ task เพี้ยน —
    ผ่าน Orchestrator.run_task() ตรงๆ (headless=True, confirm_plan=False เสมอ)

    *** W12[A] (แก้จากผลรันจริงครั้งแรก): run_task() ไม่มี kwarg ชื่อ auto_approve เลย
    (นั่นเป็นแนวคิดระดับ routes.py::_make_ask_user_func เท่านั้น — ห่อ ask_user_func ให้
    auto-approve เอง ไม่ใช่ parameter ตรงของ Orchestrator) เดิมโค้ดนี้ส่ง
    auto_approve=True ตรงๆ เข้า run_task() ทำให้ TypeError ทันทีทุก task (รันจริงครั้งแรก
    เจอ 0/4 สำเร็จหมด error เดียวกัน) — อันตรายกว่านั้นคือถ้าไม่ได้ตั้งใจส่ง ask_user_func
    เข้าไปเลย (ปล่อยเป็น None ค่า default) แล้วดันไปเจอ action ที่ต้องขออนุมัติจริง (เช่น
    checkout/purchase ใน BENCHMARK_TASKS) จะ fallback ไป blocking input() ทาง terminal
    ซึ่งไม่มีคนตอบเลยในบริบท batch eval แบบนี้ — ค้างตลอดไป ไม่ error ให้เห็นด้วยซ้ำ ต้อง
    ส่ง ask_user_func ที่ auto-approve เองตรงๆ แทน ***

    task ไหนที่ run_task() เอง throw exception ขึ้นมาจริง (เช่น browser launch พัง, LLM
    API error ที่ไม่ถูกจับใน orchestrator) ไม่ทำให้ทั้ง batch หยุด — บันทึกเป็น
    success=False, steps=0, total_tokens=0 พร้อม error message แล้วรัน task ถัดไปต่อ (กฎ
    เดียวกับ retriever.py/long_term_memory.py: ส่วนหนึ่งพังไม่ควรทำทั้ง evaluation รอบนี้
    พังตาม — อยากได้ผลลัพธ์ของ task ที่เหลือครบเท่าที่ทำได้)"""
    tasks = tasks if tasks is not None else BENCHMARK_TASKS
    report = EvaluationReport()
    # W_eval_trace: id ที่ผูกทุก task ของการรันครั้งนี้เข้าด้วยกัน — release_gate.py ส่งของมันเอง
    # ลงมาเพื่อให้ 3 suite ใช้ id เดียวกัน ส่วนการรัน suite เดี่ยวๆ (run.py eval/orangehrm)
    # สร้างเอง trace จึง group ได้เสมอไม่ว่าจะเรียกจากทางไหน
    resolved_run_id = run_id or new_run_id("eval")
    for index, task in enumerate(tasks):
        # settings.eval_task_delay_seconds: เว้นจังหวะก่อน task ถัดไป ไม่ใช่ก่อนตัวแรก
        if index and settings.eval_task_delay_seconds > 0:
            await asyncio.sleep(settings.eval_task_delay_seconds)
        counting_ask_user_func, get_approval_count = _make_counting_auto_approve()
        started_at = time.monotonic()
        task_id = f"{resolved_run_id}-{task['name']}"
        try:
            result = await Orchestrator().run_task(
                url, task["goal"],
                max_steps=task.get("max_steps", 20),
                headless=True, confirm_plan=False, provider=provider,
                ask_user_func=counting_ask_user_func,
            )
            latency = time.monotonic() - started_at
            tokens = result["tokens"]
            total_tokens = tokens["input"] + tokens["output"] + tokens["cache_read"] + tokens["cache_creation"]
            execution_mode = result.get("execution_mode", "")
            is_fastpath = execution_mode.startswith("fastpath")
            report.results.append(TaskEvalResult(
                name=task["name"], goal=task["goal"], success=result["success"],
                steps=result["steps"], total_tokens=total_tokens, message=result["message"],
                latency_seconds=latency,
                llm_calls=result.get("repairs", 0) if is_fastpath else result["steps"],
                approval_count=get_approval_count(),
                fastpath=is_fastpath,
                recoveries=result.get("repairs", 0),
                action_calls=result.get("action_calls", 0),
                finish_task_calls=result.get("finish_task_calls", 0),
            ))
            # W_eval_trace: เส้นทาง eval ไม่ผ่าน TaskManager จึงต้องเรียก writer เองตรงนี้
            # (ดู core/telemetry.py หัวไฟล์สำหรับบั๊กจริงที่ทำให้ต้องทำ) — เขียนหลังบันทึกผลลง
            # report แล้ว เพื่อให้ปัญหาการเขียน log ไม่มีทางทำให้ผล eval ที่วัดได้จริงหายไป
            write_step_trace(
                result.get("history"), task_id=task_id, provider=provider,
                run_id=resolved_run_id,
            )
            write_token_usage(
                task_id=task_id, url=url, goal=task["goal"], provider=provider,
                result=result, status="done", error=None, duration_seconds=latency,
                source=SOURCE_EVAL, run_id=resolved_run_id,
            )
        except Exception as e:
            report.results.append(TaskEvalResult(
                name=task["name"], goal=task["goal"], success=False, steps=0,
                total_tokens=0, message="", error=f"{type(e).__name__}: {e}",
                latency_seconds=time.monotonic() - started_at, approval_count=get_approval_count(),
            ))
            # W_eval_trace: ไม่มี result dict ให้ดึง history จึงไม่มี trace ให้เขียน แต่ยังบันทึก
            # แถว token_usage ไว้ด้วย status="error" — เหตุผลเดียวกับ W_step_trace ในฝั่ง API:
            # task ที่พังต้องปรากฏในไฟล์ ไม่งั้น success rate ที่คำนวณจากไฟล์นี้เป็นเพดานบน
            write_token_usage(
                task_id=task_id, url=url, goal=task["goal"], provider=provider,
                result=None, status="error", error=f"{type(e).__name__}: {e}",
                duration_seconds=time.monotonic() - started_at,
                source=SOURCE_EVAL, run_id=resolved_run_id,
            )
    return report

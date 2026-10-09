"""core/evaluation.py — W12[B]: Evaluation harness แนว WebVoyager บน saucedemo.com — วัด
success rate / step / token ต่อ task ผ่าน Orchestrator.run_task() ตรงๆ (ไม่ผ่าน API/BrowserPool)
headless + confirm_plan=False + ask_user_func auto-approve ให้รันเป็น batch ได้โดยไม่มีคนเฝ้า

BENCHMARK_TASKS ใช้ goal เดียวกับ demo ใน run.py เฉพาะงานที่ "ควรสำเร็จได้ถ้า agent ทำถูก" (ไม่รวม
W7[A] Test Case A ที่จงใจให้พัง หรือ Test Case B ที่ต้องรันสองรอบต่อกัน) ครอบคลุม สั้น/กลาง/ยาว
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
    """W_eval: คืน (ask_user_func, get_count) — ask_user_func approve ทุกอย่างและนับจำนวนครั้ง
    (approval_count ต่อ task) ส่วน get_count เป็น closure อ่านค่าปัจจุบัน ไม่ใช่ snapshot
    ใช้ร่วมกับ miniwob_eval.py และ benchmark_target/runner/agent_adapters.py"""
    count = 0

    async def _counting_auto_approve(cmd: dict) -> bool:
        nonlocal count
        count += 1
        return True

    return _counting_auto_approve, lambda: count


# เดียวกับ run.py::_DEFAULT_AGENT_GOAL เป๊ะ
# W_ambiguous_benchmark_goal (6 รัน 2026-09-08): เดิม "add first product, change item to second
# product" กำกวม agent เรียก request_user_input ทุกรอบจนงบ step ไม่พอ — แก้ *เครื่องวัด* ให้ระบุชื่อ
# สินค้าตรงๆ (เจตนาเดิม) ผลก่อนหน้าของ login_checkout จึงเทียบไม่ได้แล้ว (คนละโจทย์)
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
    # W_eval (release-gate follow-up, ดู core/release_gate.py)
    latency_seconds: float = 0.0
    # llm_calls เป็นค่าประมาณ (run_task() ไม่มี counter จริง): fastpath ใช้ repairs (steps คือจำนวน
    # step ที่ replay ไม่ใช่ LLM call), slow-path ใช้ steps (คลาดได้จาก nudge/retry guard)
    llm_calls: int = 0
    approval_count: int = 0
    fastpath: bool = False
    recoveries: int = 0
    # W_gate_task_level_diff: ตัวนับจริงจาก run_task() เก็บรายตัวให้เทียบ task ต่อ task ได้
    # (ค่าเฉลี่ยรวมกลบความต่างจนอ่านไม่ออกว่าอะไรเปลี่ยน)
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

    # W_eval (release-gate follow-up): ทุก property คืน 0.0 ถ้าไม่มี results

    def _latency_percentile(self, pct: float) -> float:
        """nearest-rank (ไม่ interpolate) — พอสำหรับ batch ระดับสิบ task"""
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
        """approval_count เฉลี่ยต่อ task (ไม่ใช่สัดส่วน task — ชื่อ "_rate" ตามสเปคเดิมใน roadmap)"""
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
        """สัดส่วน task fast-path ที่ต้อง Repair (recoveries>0) แล้วยังสำเร็จ — คืน 0.0 ถ้าไม่มี
        task ไหนต้อง Repair เลย (ไม่มีอะไรให้วัด ไม่ใช่ recovery ล้ม 100%)"""
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
    """รัน task ทีละตัวตามลำดับ (ไม่ผ่าน pool ให้ step/token ต่อ task ไม่ปนกับคิว/rate-limit)
    task ที่ run_task() throw บันทึกเป็น success=False พร้อม error แล้วรันตัวถัดไปต่อ

    W12[A] (รันจริงครั้งแรก 0/4): run_task() ไม่มี kwarg auto_approve (TypeError ทุก task) และ
    ask_user_func=None จะ fallback ไป input() ทาง terminal ซึ่งค้างเงียบใน batch — ต้องส่ง
    ask_user_func ที่ auto-approve เองเสมอ"""
    tasks = tasks if tasks is not None else BENCHMARK_TASKS
    report = EvaluationReport()
    # W_eval_trace: id ผูกทุก task ของรอบนี้ — release_gate.py ส่งของมันลงมาให้ทุก suite ใช้ร่วม
    # รันเดี่ยว (run.py eval/orangehrm) สร้างเอง
    resolved_run_id = run_id or new_run_id("eval")
    for index, task in enumerate(tasks):
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
            is_fastpath = result.get("execution_mode", "").startswith("fastpath")
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
            # W_eval_trace: เส้นทาง eval ไม่ผ่าน TaskManager จึงเรียก writer เอง (ดู core/telemetry.py)
            # เขียนหลังบันทึกผลลง report แล้ว ปัญหาการเขียน log จะได้ไม่ทำผล eval หาย
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
            # W_eval_trace: ไม่มี history ให้เขียน trace แต่ยังบันทึกแถว token_usage status="error"
            # (เหมือน W_step_trace ฝั่ง API) ไม่งั้น success rate จากไฟล์เป็นเพดานบน
            write_token_usage(
                task_id=task_id, url=url, goal=task["goal"], provider=provider,
                result=None, status="error", error=f"{type(e).__name__}: {e}",
                duration_seconds=time.monotonic() - started_at,
                source=SOURCE_EVAL, run_id=resolved_run_id,
            )
    return report

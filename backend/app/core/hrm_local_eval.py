"""core/hrm_local_eval.py — W_gate_local_hrm: suite ของ release gate บน benchmark_target/ (HRM ในเครื่อง)
แทน OrangeHRM public demo — เดโมสาธารณะ multi-tenant + success จากคำพูด agent ทำให้ commit เดียวกัน
ได้ 12/15 แล้ว 15/15 ที่นี่ทุก attempt เริ่มจาก fixture เดิม และ **success ตัดสินจาก DB**

ห้ามใส่ task read-only แบบ PIM-SEARCH-* (verifier จริงเสมอ agent ไม่ทำอะไรก็ผ่าน — เทสต์รัน stub
agent ที่ไม่ทำอะไรแล้วยืนยันว่าทั้ง 6 ตัวต้อง fail) Control Plane รันในโปรเซสผ่าน TestClient ส่วน
Target Surface (port 8100) เปิดให้เองถ้ายังไม่มีใครรัน (ไม่ใช้ --reload ตาม CLAUDE.md)
"""

import asyncio
import socket
import threading
import time
import urllib.request
from typing import Optional

from backend.app.config import settings
from backend.app.core.evaluation import EvaluationReport, TaskEvalResult
from backend.app.core.telemetry import new_run_id
from backend.app.permission.rules import allow_internal_navigation_here

# task ที่ user เลือก (6 ตัว) — ครอบคลุม แก้ไข / สร้าง / ปิดการใช้งาน / user account / ลา / สรรหา
HRM_LOCAL_TASK_IDS: tuple[str, ...] = (
    "PIM-EDIT-VERIFY-01",
    "PIM-CREATE-EMPLOYEE",
    "PIM-DEACTIVATE-EMPLOYEE",
    "ADMIN-CREATE-USER-ACCOUNT",
    "LEAVE-APPLY-01",
    "RECRUITMENT-SHORTLIST-01",
)

_TARGET_HOST = "127.0.0.1"
_TARGET_PORT = 8100
_TARGET_READY_TIMEOUT_SECONDS = 15.0
_TARGET_HEALTH_URL = f"http://{_TARGET_HOST}:{_TARGET_PORT}/login"


def load_hrm_local_tasks() -> list[dict]:
    """โหลด task จาก catalog ด้วย loader เดิมของ run_benchmark.py — id ไหนหาไม่เจอให้ล้มดังๆ
    ทันที ไม่ปล่อยให้ suite เงียบลงเหลือ 5 task แล้วตัวหารของ success_rate เปลี่ยนโดยไม่มีใครรู้"""
    from benchmark_target.run_benchmark import load_tasks

    by_id = {t["task_id"]: t for t in load_tasks(["L1", "L2"])}
    missing = [task_id for task_id in HRM_LOCAL_TASK_IDS if task_id not in by_id]
    if missing:
        raise RuntimeError(f"hrm_local suite: task ไม่อยู่ใน catalog: {missing}")
    return [by_id[task_id] for task_id in HRM_LOCAL_TASK_IDS]


def _target_is_up() -> bool:
    try:
        with urllib.request.urlopen(_TARGET_HEALTH_URL, timeout=2) as response:
            return response.status < 500
    except Exception:
        return False


def _port_is_taken() -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((_TARGET_HOST, _TARGET_PORT)) == 0


def ensure_target_running() -> None:
    """เปิด Target Surface ใน daemon thread ถ้ายังไม่มีใครรัน — ถ้า port ถูกใช้อยู่แต่ไม่ตอบ
    /login แปลว่าเป็นโปรแกรมอื่น ล้มดังๆ ดีกว่าเปิดซ้อนแล้ววัดผลบนเว็บผิดตัว"""
    if _target_is_up():
        return
    if _port_is_taken():
        raise RuntimeError(
            f"port {_TARGET_PORT} ถูกใช้อยู่แต่ไม่ใช่ benchmark target (GET /login ไม่ตอบ) — "
            "ปิดโปรแกรมนั้นก่อน"
        )
    import uvicorn

    config = uvicorn.Config(
        "benchmark_target.app.main:app", host=_TARGET_HOST, port=_TARGET_PORT, log_level="warning",
    )
    server = uvicorn.Server(config)
    threading.Thread(target=server.run, daemon=True, name="hrm-local-target").start()
    deadline = time.monotonic() + _TARGET_READY_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if _target_is_up():
            return
        time.sleep(0.3)
    raise RuntimeError(f"benchmark target ไม่พร้อมภายใน {_TARGET_READY_TIMEOUT_SECONDS:.0f} วินาที")


class _RecordingAdapter:
    """ห่อ adapter จริงเพื่อเก็บ AgentRunResult ล่าสุด — AttemptResult ไม่มี approval_count จึง
    ดึงจากตรงนี้แทนการไปแก้ attempt_runner ที่มีเทสต์คุมอยู่"""

    def __init__(self, inner):
        self._inner = inner
        self.last = None

    def run(self, task: dict, actor: str):
        # ล้างก่อนเรียก — ถ้า adapter จริง raise ต้องไม่เหลือผลของ task ก่อนหน้าค้างอยู่
        self.last = None
        self.last = self._inner.run(task, actor)
        return self.last


def attempt_to_eval_result(task: dict, attempt, agent_result) -> TaskEvalResult:
    """แปลงผล attempt เป็นแถวของ gate — success มาจาก DB (attempt.passed) เท่านั้น

    attempt ที่ล้มเพราะ infra/setup (fixture apply ไม่ได้, control plane ล่ม, browser พัง) ใส่
    steps=0 + error เพื่อให้ task_failed_on_infrastructure()/run_is_invalid() ของ gate จัดการเอง
    แทนที่จะนับเป็นความผิดของ agent"""
    tokens = attempt.tokens or {}
    total_tokens = sum(int(tokens.get(key, 0) or 0) for key in ("input", "output", "cache_read", "cache_creation"))
    infra = attempt.is_infra_failure()
    # W_gate_local_hrm: ต่อข้อความ agent ท้าย verifier detail ไม่งั้น provider ล่มกลางรัน ("429 quota")
    # ไม่มี marker ให้ task_failed_on_infrastructure() เห็น verifier อยู่ก่อนเพราะ _result_row ตัดที่ 400
    message = attempt.detail
    agent_message = getattr(agent_result, "message", "") if agent_result else ""
    if agent_message:
        message = f"{attempt.detail} | agent: {agent_message}"
    return TaskEvalResult(
        name=task["task_id"],
        goal=task["goal"]["description"],
        success=bool(attempt.passed),
        steps=0 if infra else attempt.steps,
        total_tokens=0 if infra else total_tokens,
        message=message,
        error=attempt.detail if infra else None,
        latency_seconds=attempt.duration_ms / 1000.0,
        llm_calls=0 if infra else attempt.steps,
        approval_count=getattr(agent_result, "approval_count", 0) if agent_result else 0,
    )


def _run_sync(provider: Optional[str], run_id: str, adapter=None) -> EvaluationReport:
    from fastapi.testclient import TestClient

    from benchmark_target.control.main import app as control_app
    from benchmark_target.runner.agent_adapters import HermesAgentAdapter
    from benchmark_target.runner.attempt_runner import run_attempt

    tasks = load_hrm_local_tasks()
    ensure_target_running()
    recording = _RecordingAdapter(
        adapter or HermesAgentAdapter(provider=provider, max_steps=20, headless=True, run_id=run_id)
    )
    report = EvaluationReport()

    # SSRF guard บล็อก localhost เป็นค่าเริ่มต้น — เปิดผ่าน ContextVar จึงมีผลแค่ thread นี้ (asyncio.run
    # ใน adapter คัดลอก context) task อื่นยังโดนบล็อก agent ถูกจำกัดอีกชั้นด้วย allowed_domains={"localhost"}
    with allow_internal_navigation_here(), TestClient(control_app) as control_client:
        for index, task in enumerate(tasks):
            if index and settings.eval_task_delay_seconds > 0:
                time.sleep(settings.eval_task_delay_seconds)
            attempt = run_attempt(task, recording, control_client)
            report.results.append(attempt_to_eval_result(task, attempt, recording.last))
    return report


async def run_hrm_local_evaluation(
    provider: Optional[str] = None, run_id: Optional[str] = None, adapter=None,
) -> EvaluationReport:
    """รัน 6 task บน benchmark_target ทีละตัว ตัดสินจาก DB — HermesAgentAdapter.run() เรียก
    asyncio.run() ข้างใน จึงต้องอยู่ใน thread แยก ไม่งั้นชน event loop ของ release gate

    adapter ไว้ให้เทสต์ฉีด StubAgentAdapter (ไม่แตะ LLM/browser)"""
    resolved_run_id = run_id or new_run_id("eval")
    return await asyncio.to_thread(_run_sync, provider, resolved_run_id, adapter)

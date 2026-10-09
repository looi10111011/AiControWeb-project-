"""core/miniwob_eval.py — MiniWoB++ evaluation harness; mirror ของ core/evaluation.py (W12[B])
(Orchestrator.run_task() ตรงๆ, auto-approve) ต่างแค่ที่มาของ task/url และวิธีตัดสิน success

ไม่ใช้ Gym/Selenium API ของ package `miniwob` (action space ไม่ตรงกับ agent ภาษาธรรมชาติตัวนี้)
ยืมแค่ไฟล์ HTML/JS ของ task (miniwob/html/miniwob/*.html) เปิดผ่าน file:// อ่านโจทย์จาก
core.getUtterance() และผลจาก window.WOB_DONE_GLOBAL / WOB_REWARD_GLOBAL ด้วย page.evaluate()

init script (_init_script) แก้ 3 ปัญหาที่เจอจากการเปิดไฟล์จริง:
  0. core.startEpisode() แค่โชว์ cover div "START" ต้องคลิกก่อนถึงจะ genProblem() — override ให้
     เรียก startEpisodeReal() ตรงๆ (ต้อง createDisplay() ก่อนเพื่อสร้าง #click-canvas และตั้ง
     core.cover_div เป็น dummy div เพราะ startEpisodeReal() ไม่เช็ค null) — one-shot ด้วย
     window.__miniwobStarted: endEpisode() เรียก startEpisode() ซ้ำทันที ถ้า auto-start ซ้ำจะ
     รีเซ็ต WOB_DONE/REWARD ก่อน Python ทันอ่าน (harness สนใจแค่ episode แรกต่อการโหลดหน้า)
  1. โจทย์สุ่มใหม่ทุกครั้งที่โหลด และ run_task() goto() ซ้ำเสมอ (extract_domain() ของ file:// ว่าง
     จึงไม่ skip) — ยึด Math.random ด้วย seeded PRNG ให้ทุกการโหลดได้โจทย์เดิม
  2. core.EPISODE_MAX_TIME default 10s สั้นเกินสำหรับ LLM agent — bump ใน 'load' listener ที่
     ลงทะเบียนก่อน window.onload ของหน้า
"""

import asyncio
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from playwright.async_api import async_playwright

from backend.app.config import settings
from backend.app.core.evaluation import EvaluationReport, _make_counting_auto_approve
from backend.app.core.orchestrator import Orchestrator
from backend.app.core.telemetry import (
    SOURCE_EVAL, new_run_id, write_step_trace, write_token_usage,
)

try:
    import miniwob as _miniwob_pkg
    MINIWOB_TASK_DIR: Optional[Path] = Path(_miniwob_pkg.__file__).parent / "html" / "miniwob"
except ImportError:
    MINIWOB_TASK_DIR = None

# ชุด smoke test ครอบคลุมรูปแบบการโต้ตอบต่างกัน (ไม่รันครบ 130 task เพราะยิง LLM จริงทุก step)
DEFAULT_TASKS: list[str] = [
    "click-test", "click-button", "click-checkboxes", "enter-text",
    "click-dialog", "focus-text", "click-tab", "choose-list",
]

# ยืดจาก default 10s ของ MiniWoB (docstring ข้อ 2)
_EPISODE_MAX_TIME_MS = 5 * 60 * 1000

# wall-clock กันรอบเดียวค้าง (เช่น LLM API แขวน) — คนละกลไกกับ max_steps
_TASK_WALL_CLOCK_TIMEOUT_SECONDS = 300


def _init_script(seed: int) -> str:
    """script สำหรับ page.add_init_script() — seeded Math.random (mulberry32) + override
    core.startEpisode/EPISODE_MAX_TIME (ดู docstring หัวไฟล์ ข้อ 0-2)"""
    return f"""
(function() {{
  var state = {seed} >>> 0;
  function mulberry32() {{
    state |= 0; state = (state + 0x6D2B79F5) | 0;
    var t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }}
  Math.random = mulberry32;

  window.addEventListener('load', function() {{
    if (window.core) {{
      core.EPISODE_MAX_TIME = {_EPISODE_MAX_TIME_MS};
      // ข้าม cover div — auto-start ครั้งแรกเท่านั้น (ดู docstring หัวไฟล์ ข้อ 0)
      core.startEpisode = function() {{
        core.createDisplay();
        if (!window.__miniwobStarted) {{
          window.__miniwobStarted = true;
          core.cover_div = document.createElement('div');
          core.startEpisodeReal();
        }}
      }};
    }}
  }});
}})();
"""


def task_seed(task_name: str) -> int:
    """seed 32-bit ที่ไม่ใช่ 0 และคงที่ต่อชื่อ task ทุก process/ทุกเครื่อง

    W_miniwob_seed_not_stable (2026-09-07, ไล่บั๊ก click-checkboxes): เดิมใช้ abs(hash()) ซึ่ง
    PYTHONHASHSEED สุ่มใหม่ทุก process — gate แต่ละรอบเจอคนละโจทย์ เทียบ regression ข้ามรอบไม่ได้
    แก้ด้วย crc32 (ต้องการแค่ determinism)"""
    return (zlib.crc32(task_name.encode("utf-8")) % 2_147_483_647) or 1


async def _auto_approve(cmd: dict) -> bool:
    """ask_user_func ที่ approve ทุกอย่าง — batch ไม่มีคนเฝ้า (ดู evaluation.py::run_evaluation)"""
    return True


@dataclass
class MiniWobResult:
    task: str
    utterance: str
    success: bool
    reward: float
    steps: int
    total_tokens: int
    message: str
    error: Optional[str] = None
    # W_eval / W_gate_task_level_diff: mirror ของ evaluation.py::TaskEvalResult (duck-typed คู่กัน)
    latency_seconds: float = 0.0
    llm_calls: int = 0
    approval_count: int = 0
    fastpath: bool = False
    recoveries: int = 0
    action_calls: int = 0
    finish_task_calls: int = 0


@dataclass
class MiniWobReport(EvaluationReport):
    """aggregate properties ทั้งหมดสืบจาก EvaluationReport (สูตรเดียวกัน duck-typed กับ MiniWobResult)"""
    results: list[MiniWobResult] = field(default_factory=list)


async def _run_one_task(
    task_name: str, max_steps: int, provider: Optional[str], headless: bool,
    run_id: Optional[str] = None, task_id: Optional[str] = None,
) -> MiniWobResult:
    if MINIWOB_TASK_DIR is None:
        raise RuntimeError(
            "ไม่พบ pip package 'miniwob' — ติดตั้งก่อนด้วย `pip install miniwob` "
            "(ยืมแค่ไฟล์ HTML/JS ของ task ที่ bundle มาด้วย ไม่ได้ใช้ Python/Gym API ของ "
            "package นี้เลย — ดู docstring หัวไฟล์)"
        )
    html_path = MINIWOB_TASK_DIR / f"{task_name}.html"
    if not html_path.exists():
        raise FileNotFoundError(f"ไม่พบ MiniWoB task {task_name!r} ที่ {html_path}")
    url = html_path.resolve().as_uri()

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=headless)
        try:
            context = await browser.new_context()
            page = await context.new_page()
            await page.add_init_script(_init_script(task_seed(task_name)))
            await page.goto(url)
            await page.wait_for_load_state("networkidle")

            utterance = await page.evaluate(
                "() => (window.core && core.getUtterance) ? core.getUtterance() "
                "        : ((document.getElementById('query') || {}).textContent || '').trim()"
            )
            goal = (
                f"{utterance.strip()}\n\n"
                "This is the entire instruction for this page (exact wording/casing "
                "matters — match it exactly, do not paraphrase). Call finish_task once "
                "you believe it is satisfied."
            )

            counting_ask_user_func, get_approval_count = _make_counting_auto_approve()
            started_at = time.monotonic()
            result = await Orchestrator().run_task(
                url, goal, max_steps=max_steps, page=page,
                confirm_plan=False, ask_user_func=counting_ask_user_func, provider=provider,
            )
            elapsed = time.monotonic() - started_at

            state = await page.evaluate(
                "() => ({done: !!window.WOB_DONE_GLOBAL, reward: window.WOB_REWARD_GLOBAL || 0})"
            )
            tokens = result["tokens"]
            total_tokens = tokens["input"] + tokens["output"] + tokens["cache_read"] + tokens["cache_creation"]
            is_fastpath = result.get("execution_mode", "").startswith("fastpath")
            # W_eval_trace: เส้นทาง eval ไม่ผ่าน TaskManager จึงเรียก writer เอง (ดู core/telemetry.py)
            write_step_trace(
                result.get("history"), task_id=task_id or task_name, provider=provider,
                run_id=run_id,
            )
            write_token_usage(
                task_id=task_id or task_name, url=url, goal=goal, provider=provider,
                result=result, status="done", error=None, duration_seconds=elapsed,
                source=SOURCE_EVAL, run_id=run_id,
            )
            return MiniWobResult(
                task=task_name, utterance=utterance.strip(),
                success=bool(state["done"]) and float(state["reward"]) > 0,
                reward=float(state["reward"]), steps=result["steps"],
                total_tokens=total_tokens,
                message=f'{result["message"]} ({elapsed:.1f}s)',
                latency_seconds=elapsed,
                llm_calls=result.get("repairs", 0) if is_fastpath else result["steps"],
                approval_count=get_approval_count(),
                fastpath=is_fastpath,
                recoveries=result.get("repairs", 0),
                action_calls=result.get("action_calls", 0),
                finish_task_calls=result.get("finish_task_calls", 0),
            )
        finally:
            await browser.close()


async def run_miniwob_evaluation(
    tasks: Optional[list[str]] = None,
    max_steps: int = 15,
    provider: Optional[str] = None,
    headless: bool = True,
    run_id: Optional[str] = None,
) -> MiniWobReport:
    """รัน task ทีละตัวตามลำดับ — task ที่พัง (ไฟล์หาไม่เจอ/timeout/exception) บันทึกเป็น
    success=False พร้อม error แล้วรันตัวถัดไปต่อ (กฎเดียวกับ evaluation.py::run_evaluation)"""
    tasks = tasks if tasks is not None else DEFAULT_TASKS
    report = MiniWobReport()
    # W_eval_trace: release_gate.py ส่ง run_id ร่วมของทุก suite ลงมา รันเดี่ยวสร้างเอง
    resolved_run_id = run_id or new_run_id("eval")
    for index, task_name in enumerate(tasks):
        if index and settings.eval_task_delay_seconds > 0:
            await asyncio.sleep(settings.eval_task_delay_seconds)
        task_id = f"{resolved_run_id}-{task_name}"
        started_at = time.monotonic()
        try:
            result = await asyncio.wait_for(
                _run_one_task(
                    task_name, max_steps, provider, headless,
                    run_id=resolved_run_id, task_id=task_id,
                ),
                timeout=_TASK_WALL_CLOCK_TIMEOUT_SECONDS,
            )
            report.results.append(result)
            continue
        except asyncio.TimeoutError:
            error = f"TimeoutError: เกิน {_TASK_WALL_CLOCK_TIMEOUT_SECONDS}s (wall-clock)"
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
        report.results.append(MiniWobResult(
            task=task_name, utterance="", success=False, reward=0.0, steps=0,
            total_tokens=0, message="", error=error,
        ))
        # W_eval_trace: ไม่มี history ให้เขียน trace แต่ยังต้องมีแถว token_usage ไม่งั้น task ที่พังหายจากไฟล์
        write_token_usage(
            task_id=task_id, url=task_name, goal=task_name, provider=provider,
            result=None, status="error", error=error,
            duration_seconds=time.monotonic() - started_at,
            source=SOURCE_EVAL, run_id=resolved_run_id,
        )
    return report

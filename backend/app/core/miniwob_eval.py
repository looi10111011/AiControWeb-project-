"""core/miniwob_eval.py — MiniWoB++ (https://github.com/Farama-Foundation/miniwob-plusplus)
evaluation harness สำหรับ agent ตัวนี้โดยเฉพาะ — mirror รูปแบบเดียวกับ core/evaluation.py
(W12[B]) ทุกประการ (Orchestrator.run_task() ตรงๆ, ไม่ผ่าน API/BrowserPool, auto-approve
ask_user_func) ต่างแค่ที่มาของ task/url และวิธีตัดสิน success

*** ตั้งใจไม่ใช้ package `miniwob` (pip install miniwob) ในแบบที่ตั้งใจไว้แต่แรก — ของเดิม
เป็น Gymnasium env ที่ควบคุมผ่าน Selenium ด้วย structured action space (env.step() รับ
CLICK_ELEMENT + element ref) ซึ่งไม่ตรงกับ agent ตัวนี้เลย (Playwright + LLM ที่รับ goal
เป็นภาษาธรรมชาติ ไม่ใช่ RL policy ที่เลือก action จาก observation dict) — เขียน adapter
แปลง action space จะเสียเวลาเปล่าและไม่ทดสอบความสามารถจริงของ agent ตัวนี้ด้วย

สิ่งที่ยืมมาจาก package `miniwob` มีแค่ไฟล์ HTML/JS ของแต่ละ task ที่ bundle มาด้วย
(miniwob/html/miniwob/*.html) ซึ่งเป็นหน้าเว็บ standalone สมบูรณ์ในตัวเอง ไม่พึ่ง Python/
Selenium เลย: เปิดตรงผ่าน file:// ได้, โจทย์ของแต่ละ episode มาจาก DOM element
id="query" (อ่านผ่าน core.getUtterance() ได้), และผลลัพธ์รายงานผ่าน global JS variable
window.WOB_DONE_GLOBAL / window.WOB_REWARD_GLOBAL (core.js::core.endEpisode()) — ทั้งหมดนี้
อ่าน/เขียนได้ตรงๆ ด้วย Playwright page.evaluate() อยู่แล้ว ไม่ต้องพึ่ง Selenium/Gym wrapper
เลยสักจุด (ตรวจ core.js ของจริงแล้ว — ไม่มีการเรียก Math.seedrandom()/query-string seed
ใดๆ เลย มีแต่ comment ตัวอย่างการใช้งาน seedrandom.js เฉยๆ)

ปัญหาที่ต้องแก้ไข 2 อย่างก่อนใช้งานได้จริงกับ agent LLM (ไม่ใช่ RL policy ที่ตอบสนองทันที):

  0. core.startEpisode() (เรียกจาก window.onload ของทุก task page) ไม่ได้เรียก genProblem()
     ตรงๆ — แค่โชว์ "cover div" สีเทาเขียนว่า START ทับหน้าไว้ (core.createDisplay() +
     คลิกที่ cover ถึงจะเรียก core.startEpisodeReal() ที่ genProblem()/สุ่มโจทย์จริงอยู่
     ข้างใน — ดู core.js) ไม่มีเอกสารพูดถึงจุดนี้เลย เจอจากการทดสอบเปิดไฟล์จริง (#query/
     #area ว่างเปล่าตลอดจนกว่าจะคลิก cover) — ปล่อยให้ agent ต้องคลิก "START" เป็น step
     แรกเสมอจะเสีย step/token เปล่าๆ ทุก task (แถม element นี้ไม่ใช่ของจริงของ task เลย)
     แก้ด้วย init script เดียวกับข้อ 1-2 ด้านล่าง: override core.startEpisode ให้เรียก
     core.startEpisodeReal() ตรงๆ ข้าม cover div ไปเลย (override ได้เพราะ listener นี้รอ
     จนถึง 'load' event ซึ่ง core.js โหลด/รัน exec เสร็จไปแล้วก่อนหน้านั้น — core.startEpisode
     มีอยู่จริงแล้วตอนนี้ ก่อนที่ handler window.onload=... ของหน้าเองจะถูกเรียก เพราะ listener
     ของเราลงทะเบียนก่อน (init script รันก่อน script ของหน้าทุกตัว) จึงทำงานก่อนตามลำดับ
     การลงทะเบียนของ event เดียวกัน) *** ต้องเรียก core.createDisplay() ก่อนด้วย (ทดสอบแล้ว
     เจอจริง) เพราะมันสร้าง <canvas id="click-canvas"> ที่ core.canvasClear()/
     startEpisodeReal() คาดว่ามีอยู่แล้วเสมอ (ไม่งั้น throw "Cannot read properties of
     null (reading 'getContext')" กลางคัน) และต้องตั้ง core.cover_div เป็น dummy
     <div> เปล่าๆ ที่ไม่ได้ append เข้า DOM จริง (แทนที่จะปล่อยเป็น null) เพราะ
     startEpisodeReal() เขียน core.cover_div.style.display='none' ตรงๆ โดยไม่เช็ค null
     ก่อนเลย ***

     *** ข้อควรระวังที่ทดสอบแล้วเจออีกจุด: core.endEpisode() (ถูกเรียกตอนคลิกถูก/ผิด)
     ปิดท้ายด้วย core.startEpisode() เสมอ (บรรทัดสุดท้ายของฟังก์ชัน — คอมเมนต์ในไฟล์บอกว่า
     เดิมมี setTimeout หน่วงไว้ 500ms แต่ปัจจุบัน comment ออกแล้วเพราะ "With the sync screen,
     the timeout above is redundant" คือ "sync screen" เดิมมันคือ cover div ที่รอคลิกอยู่
     แล้วนั่นเอง ที่เป็นตัวหน่วงให้ WOB_DONE_GLOBAL/WOB_REWARD_GLOBAL ค้างอยู่จนกว่าคนจะคลิก
     เริ่ม episode ใหม่เอง) — override ของเราที่ตัด cover ออกทำให้ startEpisode() เรียก
     startEpisodeReal() ซ้ำทันทีแบบ synchronous ทุกครั้งที่ endEpisode() จบ (แม้แต่ครั้ง
     แรกที่ agent เพิ่งทำสำเร็จ/พลาด) รีเซ็ต WOB_DONE_GLOBAL/WOB_REWARD_GLOBAL กลับเป็น
     false/0 ทันทีก่อนที่ page.evaluate() ฝั่ง Python จะทันอ่านค่าเดิมด้วยซ้ำ — แก้ด้วย flag
     window.__miniwobStarted (one-shot): auto-start เฉพาะครั้งแรกเท่านั้น (ตอน
     window.onload เรียกจริง) ครั้งถัดไปที่ endEpisode() เรียก startEpisode() ซ้ำ (เริ่ม
     episode ใหม่) ให้ "ค้าง" ไว้เฉยๆ (แค่ createDisplay() เหมือนเดิม ไม่เรียก
     startEpisodeReal() ซ้ำ) ให้ WOB_DONE_GLOBAL/WOB_REWARD_GLOBAL ของ episode แรกนิ่งอยู่
     แบบนั้นตลอดไป — ใช้ได้เพราะ harness นี้สนใจแค่ 1 episode ต่อการโหลดหน้า 1 ครั้งเท่านั้น
     (เปิดหน้าใหม่ทุก task run อยู่แล้ว ไม่มี "episode ที่ 2" ให้ต้องสนใจจริง) ***

  1. โจทย์ต้องรู้ "ล่วงหน้า" ก่อนเรียก run_task() (goal เป็น argument ธรรมดา ไม่ใช่ค่าที่
     เปลี่ยนได้กลางคัน) แต่ query text ของแต่ละ task สุ่มใหม่ทุกครั้งที่หน้าโหลด
     (window.onload -> core.startEpisode() -> genProblem() สุ่มเนื้อหาใหม่ทุกครั้ง) —
     run_task() รับ page ที่เปิดไว้แล้วได้ (param `page`) แต่ยังคง goto() ซ้ำเองเสมอ
     (skip_initial_goto ใน orchestrator.py เช็คแค่ domain ตรงกันไหม — extract_domain()
     ของ file:// คืนสตริงว่างเปล่าเสมอเพราะไม่มี netloc เลย ทำให้เงื่อนไข "หน้าเว็บเป้าหมาย
     เปิดอยู่แล้ว" เป็นเท็จเสมอสำหรับ file:// ไม่ว่าจะเป็น URL เดียวกันจริงแค่ไหน) — แก้โดย
     "ยึด randomness ให้นิ่ง" แทนที่จะพยายามเลี่ยง navigate ซ้ำ: inject Math.random()
     แบบ seeded (deterministic PRNG) ผ่าน page.add_init_script() ก่อนโหลดครั้งแรก ทำให้
     ทุกครั้งที่หน้านี้ (หรือ context เดียวกัน) โหลดใหม่ genProblem() สุ่มออกมาได้ "ผลลัพธ์
     เดียวกันเป๊ะ" เสมอ — อ่าน utterance รอบแรกได้ตรงกับ episode ที่ agent จะเจอจริงตอน
     run_task() navigate ซ้ำ (ไม่ต้องแก้ orchestrator.py เลยสักบรรทัด)

  2. core.EPISODE_MAX_TIME default 10 วินาที (คิดมาสำหรับ RL policy/scripted action ที่
     ตอบสนองทันที) หมดเวลาแล้ว core.js เรียก core.endEpisode(-1, false, 'timed out') เอง
     อัตโนมัติ ทำให้ episode ถูกตัดสินว่า "แพ้" ก่อน LLM agent (ที่แต่ละ step ใช้เวลาหลาย
     วินาที: perceive + LLM call + execute) จะมีโอกาสทำอะไรเลยด้วยซ้ำ — ยืดเวลาให้ผ่าน
     init script เดียวกัน (ข้อ 1) โดย addEventListener('load', ...) ดักไว้ก่อน (registration
     order ชนะ window.onload= ของหน้าเสมอเพราะ init script รันก่อนทุก script ของหน้า)
     bump core.EPISODE_MAX_TIME ก่อน core.startEpisode() จะอ่านค่านี้ไปตั้ง timer จริง
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from playwright.async_api import async_playwright

from backend.app.core.orchestrator import Orchestrator

try:
    import miniwob as _miniwob_pkg
    MINIWOB_TASK_DIR: Optional[Path] = Path(_miniwob_pkg.__file__).parent / "html" / "miniwob"
except ImportError:
    MINIWOB_TASK_DIR = None

# ชุด task เริ่มต้นสำหรับ smoke test — เลือกให้ครอบคลุมรูปแบบการโต้ตอบต่างกัน ไม่ใช่ทุก
# 130 task ในแพ็กเกจ (ยิง LLM call จริงทุก step ของทุก task — รันครบทุกตัวแพงและช้าเกินไป
# สำหรับ smoke test ปกติ): คลิกปุ่มตรงๆ, คลิกปุ่มที่ต้องอ่าน label เอง, ติ๊ก checkbox
# หลายตัว, พิมพ์ข้อความ, คลิกใน modal dialog, คลิก/focus ช่องกรอก, สลับ tab, เลือกจาก
# dropdown — ผู้เรียกส่ง `tasks=` เองเพื่อรันชุดอื่น/task เดี่ยวๆ ได้เสมอ (ดู run_evaluation())
DEFAULT_TASKS: list[str] = [
    "click-test", "click-button", "click-checkboxes", "enter-text",
    "click-dialog", "focus-text", "click-tab", "choose-list",
]

# ยืดจาก default ของ MiniWoB (10 วินาที) ให้พอสำหรับ LLM agent จริง — ดู docstring หัวไฟล์
# ข้อ 2 (5 นาทีเผื่อ task ที่ต้องหลาย step + LLM ช้าเป็นบางครั้ง)
_EPISODE_MAX_TIME_MS = 5 * 60 * 1000

# wall-clock กันรอบเดียวค้างไม่รู้จบ (เช่น LLM API แขวน) ไม่ให้ทั้ง batch ค้างตามไปด้วย —
# คนละกลไกกับ max_steps (จำกัดจำนวนรอบ perceive-plan-act ไม่ใช่เวลาจริง)
_TASK_WALL_CLOCK_TIMEOUT_SECONDS = 300


def _init_script(seed: int) -> str:
    """page.add_init_script() — รันก่อน script ของหน้าทุกตัวเสมอทุกครั้งที่หน้านี้โหลดใหม่
    (ดู docstring หัวไฟล์ ข้อ 1-2) mulberry32 เป็น seeded PRNG ธรรมดา ไม่ต้องพึ่ง library
    ภายนอกใดๆ (core.js เองก็ไม่ได้ import seedrandom.js ไว้ใช้งานจริงอยู่แล้ว มีแต่
    comment ตัวอย่าง)"""
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
      // ข้าม "cover div" ที่ต้องคลิก START ก่อนถึงจะสุ่มโจทย์จริง (ดู docstring หัวไฟล์
      // ข้อ 0) — auto-start เฉพาะครั้งแรกเท่านั้น (window.__miniwobStarted flag) ครั้ง
      // ถัดไปที่ endEpisode() เรียก startEpisode() ซ้ำเพื่อเริ่ม episode ใหม่ ให้ค้างไว้
      // เฉยๆ แทน (ไม่งั้น WOB_DONE_GLOBAL/WOB_REWARD_GLOBAL ของ episode แรกที่เพิ่งจบจะถูก
      // รีเซ็ตทันทีก่อน Python จะทันอ่าน — ดู docstring หัวไฟล์)
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


async def _auto_approve(cmd: dict) -> bool:
    """ask_user_func ที่ auto-approve ทุก action ที่ต้องขอยืนยัน — รันแบบ batch ไม่มีคน
    เฝ้าหน้าจอตอบจริง เหมือน core/evaluation.py::_auto_approve ทุกประการ (ดูที่นั่นสำหรับ
    เหตุผลเต็มว่าทำไมต้องส่งมาตรงๆ ไม่ปล่อยเป็น None)"""
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


@dataclass
class MiniWobReport:
    results: list[MiniWobResult] = field(default_factory=list)

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


async def _run_one_task(
    task_name: str, max_steps: int, provider: Optional[str], headless: bool,
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

    # seed คงที่ต่อ task name (ไม่ใช่สุ่มจาก wall-clock) — ต้องเหมือนเดิมทุกครั้งที่รันไฟล์
    # นี้ซ้ำ เพื่อให้ utterance ที่อ่านตอนนี้ตรงกับ episode ที่ agent จะเจอจริงตอน
    # run_task() navigate ซ้ำ (ดู docstring หัวไฟล์ ข้อ 1) — ไม่ต้องสุ่มข้าม task จริงจัง
    # (แค่ต้องการเลข 32-bit ที่ไม่ใช่ 0 เสมอ)
    seed = (abs(hash(task_name)) % 2_147_483_647) or 1

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=headless)
        try:
            context = await browser.new_context()
            page = await context.new_page()
            await page.add_init_script(_init_script(seed))
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

            started_at = time.monotonic()
            result = await Orchestrator().run_task(
                url, goal, max_steps=max_steps, page=page,
                confirm_plan=False, ask_user_func=_auto_approve, provider=provider,
            )
            elapsed = time.monotonic() - started_at

            state = await page.evaluate(
                "() => ({done: !!window.WOB_DONE_GLOBAL, reward: window.WOB_REWARD_GLOBAL || 0})"
            )
            tokens = result["tokens"]
            total_tokens = tokens["input"] + tokens["output"] + tokens["cache_read"] + tokens["cache_creation"]
            return MiniWobResult(
                task=task_name, utterance=utterance.strip(),
                success=bool(state["done"]) and float(state["reward"]) > 0,
                reward=float(state["reward"]), steps=result["steps"],
                total_tokens=total_tokens,
                message=f'{result["message"]} ({elapsed:.1f}s)',
            )
        finally:
            await browser.close()


async def run_miniwob_evaluation(
    tasks: Optional[list[str]] = None,
    max_steps: int = 15,
    provider: Optional[str] = None,
    headless: bool = True,
) -> MiniWobReport:
    """รัน MiniWoB task ทีละตัวตามลำดับผ่าน Orchestrator.run_task() ตรงๆ (เหมือน
    core/evaluation.py::run_evaluation ทุกประการ) — task ไหนพัง (ไฟล์หาไม่เจอ, timeout,
    exception จาก run_task()) ไม่ทำให้ทั้ง batch หยุด บันทึกเป็น success=False พร้อม error
    แล้วรัน task ถัดไปต่อ (กฎเดียวกับ evaluation.py)"""
    import asyncio

    tasks = tasks if tasks is not None else DEFAULT_TASKS
    report = MiniWobReport()
    for task_name in tasks:
        try:
            result = await asyncio.wait_for(
                _run_one_task(task_name, max_steps, provider, headless),
                timeout=_TASK_WALL_CLOCK_TIMEOUT_SECONDS,
            )
            report.results.append(result)
        except asyncio.TimeoutError:
            report.results.append(MiniWobResult(
                task=task_name, utterance="", success=False, reward=0.0, steps=0,
                total_tokens=0, message="",
                error=f"TimeoutError: เกิน {_TASK_WALL_CLOCK_TIMEOUT_SECONDS}s (wall-clock)",
            ))
        except Exception as e:
            report.results.append(MiniWobResult(
                task=task_name, utterance="", success=False, reward=0.0, steps=0,
                total_tokens=0, message="", error=f"{type(e).__name__}: {e}",
            ))
    return report

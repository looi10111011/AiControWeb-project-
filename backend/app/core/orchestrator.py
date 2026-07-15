"""Agent Loop: Perceive -> Plan -> Act -> Verify.

W1: skeleton only. W4: ทำ loop จริงกับเว็บง่าย 1 หน้า.
W5: retry action ที่ล้มเหลว (ดู actions.py::_dispatch_with_retry) + guard กัน
finish_task(false) ก่อนเวลาอันควร (ด้านล่าง) + permission layer/human-in-the-loop
"""

import asyncio
from typing import Optional

from playwright.async_api import Page, async_playwright

from backend.app.config import settings
from backend.app.core import llm
from backend.app.core import long_term_memory
from backend.app.core.actions import ActionResult, AskUserFunc, execute, goto, wait_stable
from backend.app.core.memory import ShortTermMemory
from backend.app.core.perception import get_snapshot
from backend.app.rag import retriever

# action ที่เปลี่ยนหน้า/DOM แบบมีนัยสำคัญ -> ต้องรอหน้านิ่งก่อน perceive รอบถัดไป
_PAGE_CHANGING_ACTIONS = {"click", "goto", "select", "go_back"}

# โมเดลบางตัว (โดยเฉพาะ Llama บน Groq) ชอบเรียก finish_task(success=false) เร็วเกินไป
# ทั้งที่ยังเหลือ step ให้ลองและยังไม่ได้ลองทางที่ชัดเจนอยู่ตรงหน้า (เช่น เห็นปุ่ม Add to
# cart แต่ไม่กด) — ไม่ยอมรับทันที ให้เตือนแล้วบังคับลองต่ออีกสูงสุด
# _MAX_PREMATURE_FALSE_FINISH_RETRIES ครั้งก่อน ถ้ายังยืนยัน false อีกถึงจะยอมรับจริง
_MAX_PREMATURE_FALSE_FINISH_RETRIES = 2
_PREMATURE_FALSE_FINISH_NUDGE = (
    "ยังไม่ยอมรับ finish_task(success=false) นี้ —ยังเหลือ step ให้ลองอยู่ และหน้าเว็บ"
    "ปัจจุบันอาจยังมี element ที่ทำต่อได้ (เช่น ปุ่มที่ยังไม่ได้กด, ช่องที่ยังว่าง) ให้ดู"
    "indexed elements ล่าสุดอีกครั้งแล้วลองทำ action ที่ยังไม่ได้ลอง ถ้าลองจริงๆ แล้วไปต่อ"
    "ไม่ได้จริง ค่อยเรียก finish_task(success=false) อีกครั้ง"
)

# W5[A] "Verify" (2026-07-15): W5 เดิมทำแค่ "Retry" (actions.py::_dispatch_with_retry)
# ไม่มี "Verify" เลย — ช่องโหว่ symmetric กับ guard ด้านบน: LLM อาจเรียก
# finish_task(success=true) เป็น action แรกสุดโดยไม่ทำอะไรเลย (steps_taken=0) แล้ว
# ระบบจะยอมรับทันทีโดยไม่มีการตรวจสอบใดๆ เลย (ต่างจาก false ที่มี guard คู่กันอยู่แล้ว)
# — SYSTEM_PROMPT ขอไว้แล้วว่า finish_task(true) ต้องมีหลักฐานจาก indexed elements
# แต่ไม่เคยมีการบังคับด้วยโค้ดเลย ไม่ block เด็ดขาด (บาง goal อาจสำเร็จอยู่แล้วตั้งแต่
# page แรกจริงๆ เช่น "verify ว่าอยู่หน้า login") แค่ให้ยืนยันอีกครั้งก่อนเหมือนกัน
_MAX_PREMATURE_TRUE_FINISH_RETRIES = 1
_PREMATURE_TRUE_FINISH_NUDGE = (
    "การเรียก finish_task(success=true) นี้ยังไม่มี action ใดๆ เกิดขึ้นเลยใน task นี้ "
    "(steps_taken=0) — ก่อนยืนยัน success ให้ตรวจสอบอีกครั้งว่า indexed elements ล่าสุด "
    "มีหลักฐานชัดเจนจริงๆ ว่า goal สำเร็จแล้ว ถ้าใช่จริง เรียก finish_task(success=true) "
    "อีกครั้งได้เลย ถ้าไม่แน่ใจ ให้ลองทำ action ที่เกี่ยวข้องกับ goal ก่อน"
)

# W5: loop-detection guard — บางโมเดล (เจอกับ Llama บน Groq) ถึงจะถูกเตือนแล้วก็ยัง
# วนเรียก browser_action เดิมเป๊ะๆ ซ้ำๆ (dict เดียวกันทุก field) ไม่ว่าจะสำเร็จหรือ fail
# ก็ตาม แปลว่าไม่มีความคืบหน้าจริง — กันไว้ไม่ให้เสีย step/token ไปเรื่อยๆ จนหมด max_steps
# โดยไม่ได้อะไรขึ้นมา ถ้าเจอ action เดิมติดกันครบจำนวนนี้ ให้หยุด task ทันที
_MAX_CONSECUTIVE_IDENTICAL_ACTIONS = 3

# (2026-07-13) เดิม guard ด้านบนจับได้แค่ pattern คาบ 1 (action เดิมเป๊ะๆ ซ้ำติดกัน
# เช่น AAAA) — แต่ agent บางครั้งวนสลับ 2 action ที่ไม่เหมือนกันไปมาแทน (คาบ 2 เช่น
# go_back -> click -> go_back -> click ซ้ำไปเรื่อยๆ) ซึ่งไม่ตรงเงื่อนไข "เดิมเป๊ะๆ
# ติดกัน" ของ guard เดิมเลยไม่เคย trigger — เพิ่ม guard ใหม่จับ pattern คาบ 2 (ABAB)
# โดยเฉพาะ แยกจาก guard เดิมที่จับคาบ 1 (AAAA) เพื่อไม่ให้ 2 เงื่อนไขทับซ้อนกันเอง
#
# (2026-07-15) generalize เพิ่มเติม: user ถามว่าถ้าโมเดลวนเป็นคาบ 3+ แทน (เช่น
# click ปุ่ม A -> scroll -> fill ค่า B -> click ปุ่ม A -> scroll -> fill ค่า B ...
# ที่ไม่ได้ทำให้หน้าเว็บเปลี่ยนสเตทจริง) guard เดิมที่เช็คแค่คาบ 2 ตรงๆ จะจับไม่ได้
# เลย (มีเทสต์ test_run_task_loop_guard_does_not_trigger_for_three_action_cycle
# ที่เดิมยืนยันไว้ตรงๆ ว่า "ยังไม่ scope ไว้") — generalize
# _is_alternating_pattern (เดิมเช็คเฉพาะคาบ 2) เป็น _is_repeating_cycle(history,
# period) เช็คได้ทุกคาบตั้งแต่ 2 ถึง _MAX_CYCLE_PERIOD แทน (คาบ 1 ยังคงแยกไปใช้
# _MAX_CONSECUTIVE_IDENTICAL_ACTIONS เดิมเหมือนเดิม เพราะ threshold หลวมกว่า — คาบ 1
# trigger ตั้งแต่ซ้ำครั้งที่ 3 ไม่ต้องรอครบ 2 รอบเต็มเหมือนคาบอื่น) — เลือก cap ที่ 4
# เพราะคาบยาวกว่านี้ทั้งเจอได้ยากขึ้นเรื่อยๆ ในทางปฏิบัติ และต้องใช้ window ยาวขึ้น
# เรื่อยๆ กว่าจะยืนยัน (period*2 action) ทำให้กว่าจะ trigger ก็เสีย step ไปเยอะแล้ว
# ไม่คุ้มจะเสีย step ต่อไปอีกเพื่อรอยืนยัน pattern ที่ยาวขึ้น
_MAX_CYCLE_PERIOD = 4
_MIN_CYCLE_REPEATS = 2  # ทุกคาบ (2 ขึ้นไป) ต้องเห็นครบกี่รอบถึงจะถือว่าติด loop
_MAX_CYCLE_WINDOW = _MAX_CYCLE_PERIOD * _MIN_CYCLE_REPEATS


def _is_repeating_cycle(window: list[dict], period: int) -> bool:
    """เช็คว่า window (ต้องยาวเท่ากับ period * _MIN_CYCLE_REPEATS พอดี) เป็นการวนซ้ำ
    คาบ `period` จริงหรือไม่ (เช่น period=3: A-B-C-A-B-C) — ต้องมีอย่างน้อย 2 ค่าที่
    ต่างกันในคาบเดียว ไม่งั้นคาบ p ของ [A, A, ..., A] จะ match ซ้ำกับคาบ 1 ที่มี guard
    แยกจับไปแล้วด้านบน (กันสอง guard ทับซ้อนกันเหมือนที่ตั้งใจไว้กับ ABAB เดิม)"""
    if len(window) != period * _MIN_CYCLE_REPEATS:
        return False
    cycle = window[:period]
    distinct: list[dict] = []
    for item in cycle:
        if item not in distinct:
            distinct.append(item)
    if len(distinct) < 2:
        return False
    return all(window[i] == cycle[i % period] for i in range(len(window)))


def _detect_repeating_cycle_period(recent_actions: list[dict]) -> Optional[int]:
    """เช็คคาบ 2 ถึง _MAX_CYCLE_PERIOD ตามลำดับ (คาบสั้นก่อน) บน recent_actions ที่
    ตัดมาแล้ว ("recent_actions[-window:]" ต่อคาบ) คืนคาบแรกที่เจอ หรือ None ถ้าไม่มี
    คาบไหน match เลย"""
    for period in range(2, _MAX_CYCLE_PERIOD + 1):
        window = period * _MIN_CYCLE_REPEATS
        if _is_repeating_cycle(recent_actions[-window:], period):
            return period
    return None

# W6[B]: จำนวน chunk คู่มือสูงสุดที่จะดึงมาแนบให้ LLM เห็นทุก step ของ per-step loop —
# ดึงใหม่ทุก step ตาม page_text ปัจจุบัน (ไม่ใช้กับ generate_plan ซึ่งเป็นแค่แผนคร่าวๆ
# ครั้งเดียวก่อนเริ่ม loop จริง เก็บ scope ไว้แค่ per-step planner ตามที่คุยกันไว้)
_RAG_CHUNKS_PER_STEP = 3

# W7[A] (long-term): เหมือน _RAG_CHUNKS_PER_STEP แต่สำหรับ long_term_memory.recall()
# (ประวัติ task run อื่นก่อนหน้า แทนคู่มือที่ user ป้อน) — ดึงใหม่ทุก step เหมือนกัน
_LONG_TERM_MEMORY_CHUNKS_PER_STEP = 3

# W7[B] (RAG-based permission): จำนวน chunk คู่มือที่ดึงมาเช็ค permission ของ action
# ที่กำลังจะทำ — ตั้งใจแยก query จาก manual_context ด้านบน (query=goal) เพราะรันจริง
# บน saucedemo.com พบว่า query ระดับ goal กว้างเกินไป: goal ที่พูดถึงคำว่า "Checkout"
# แค่ครั้งเดียวตอนท้ายสุด ทำให้ manual_context ดึง chunk เกี่ยวกับ Checkout ติดมาแทบ
# ทุก step (แม้แต่ตอน fill username ในหน้า login) ไม่ใช่แค่ step ที่กำลังจะกด Checkout
# จริง — เปลี่ยนมาใช้ query แคบตาม action ปัจจุบันแทน (ดู _build_permission_query())
# k=1 (ไม่ใช่ 3 แบบ manual_context) เพราะรันจริงยืนยันว่า k สูงกว่านี้ดึง chunk ที่
# ไม่เกี่ยวข้องติดมาด้วยได้ง่าย (คู่มือทดสอบมีแค่ ~11 chunk สั้นๆ — similarity ของ
# chunk อันดับ 2 อาจยังใกล้พอที่จะหลุดเข้ามาแบบผิดๆ)
_PERMISSION_RAG_CHUNKS_PER_STEP = 1


def _build_permission_query(cmd: dict, label: str) -> str:
    """ประกอบ query แคบเฉพาะ action นี้ (ไม่ใช่ทั้ง goal) ไว้ค้นคู่มือว่ามีกฎเกี่ยวกับ
    action นี้ไหม — goto ไม่มี label (ไม่มี index ให้จับคู่) ใช้ url แทน"""
    target = label or cmd.get("url", "")
    return f"{cmd.get('type', '')} {target}".strip()

# W7[A] (context compaction, Gemini เท่านั้น): stateless chat API ต้องส่ง messages
# ทั้งก้อนซ้ำทุก step (ไม่มี server-side session) — แต่ละ step ของ Gemini เพิ่ม page
# snapshot เต็มๆ + manual/memory/long-term context ทุกครั้งเข้าไปใน messages เรื่อยๆ
# ไม่เคยหดกลับเลย ทำให้ input token ต่อ step โตขึ้นเรื่อยๆ ตามจำนวน step (ไม่ใช่แค่
# ตามความยาว task จริง) — พอ step สะสมเกิน _GEMINI_COMPACT_AFTER_STEPS ให้ตัด step
# เก่ากว่า _GEMINI_KEEP_RECENT_STEPS ตัวล่าสุดออกจาก messages แล้วแทนที่ด้วย digest
# สั้นๆ (สร้างจาก ShortTermMemory.all() ที่มีข้อมูลสะอาดอยู่แล้ว ไม่ต้อง parse raw
# Gemini Content object เอง — ดู _build_gemini_history_digest()/_compact_gemini_messages()
# ด้านล่าง) — จำกัด scope แค่ Gemini ตามที่ user เลือก (Anthropic/Groq มี message
# format คนละแบบ ต้องเขียนแยกทีละตัว ยังไม่ทำตอนนี้)
_GEMINI_COMPACT_AFTER_STEPS = 6
_GEMINI_KEEP_RECENT_STEPS = 3


def _build_gemini_history_digest(memory: ShortTermMemory, upto_step: int) -> str:
    """สรุป step 1..upto_step (ไม่รวม step 0 ที่เป็น goto ตอนเริ่ม task) เป็น bullet
    list บรรทัดละ step สั้นๆ — สร้างใหม่จาก ShortTermMemory.all() ทุกครั้งที่บีบอัด
    (ไม่ใช่สะสมจาก digest รอบก่อน) เพราะ ShortTermMemory เก็บ history แบบไม่ตัดทิ้ง
    อยู่แล้วตลอด task จึงเป็นแหล่งความจริงที่สมบูรณ์กว่า raw messages ที่ถูกตัดไปแล้ว"""
    entries = [h for h in memory.all() if 0 < h.get("step", 0) <= upto_step]
    if not entries:
        return ""
    return "\n".join(f"- step {h['step']}: {h['cmd']} -> {h['result']}" for h in entries)


def _compact_gemini_messages(messages: list, cut_at: int, digest_text: str) -> list:
    """ตัด messages[:cut_at] ทิ้ง แล้วฝัง digest_text เข้าไปเป็นส่วนแรกของ text ใน
    turn แรกที่เหลืออยู่ (แทนที่จะแทรก turn ใหม่แยกต่างหาก) — messages[cut_at] ต้อง
    เป็น {"role": "user", "parts": [{"text": ...}]} เสมอ (จุดเริ่ม step ใหม่จาก
    next_action_gemini()) เพราะ cut_at มาจาก step boundary ที่ orchestrator เก็บเอง
    (ดู gemini_step_boundaries ใน run_task()) ไม่ใช่ตำแหน่งเดา — วิธีนี้ไม่ต้องแตะลำดับ
    role user/model ของ Gemini เลย กันปัญหา conversation structure ผิดเพี้ยนจากการ
    แทรก turn ใหม่ ถ้ารูปแบบไม่ตรงคาด (ผิดคาดจริงๆ) คืน messages เดิมไม่แก้อะไร
    ไม่ throw"""
    if cut_at <= 0 or not digest_text:
        return messages
    kept = messages[cut_at:]
    if not kept:
        return messages
    try:
        first = kept[0]
        original_text = first["parts"][0]["text"]
        new_first = {
            "role": "user",
            "parts": [{"text": f"[สรุป step ก่อนหน้าที่ถูกย่อไว้กันบทสนทนายาวเกินไป]\n{digest_text}\n\n{original_text}"}],
        }
        return [new_first] + kept[1:]
    except (KeyError, IndexError, TypeError):
        return messages


def _build_nudge_message(provider: str, text: str) -> dict:
    """ข้อความเตือนที่ต่อเข้า messages ตรงๆ (นอกเหนือจาก append_tool_result() ที่แต่ละ
    provider มี format ของตัวเองอยู่แล้วเป็นปกติ) — ต้องปรับ shape ตาม provider เหมือนกัน
    ไม่งั้น Gemini SDK จะ throw KeyError ตอนเจอ dict {"role","content"} แบบ Anthropic/
    Groq ปนอยู่ใน contents (ใช้กับ guard 2 จุดด้านล่าง: premature-false-finish และ
    premature-login-skip — เดิม hardcode format Anthropic/Groq ไว้จุดเดียวตั้งแต่ W4/W5
    ไม่เคยมีใครสังเกตเพราะไม่เคยรัน Gemini จนชนทั้ง 2 guard นี้พร้อมกันมาก่อน จนเจอจริง
    ตอนทดสอบ W7[A] Test Case A ผ่าน Gemini)"""
    if provider == "gemini":
        return {"role": "user", "parts": [{"text": text}]}
    return {"role": "user", "content": text}

# (2026-07-13) SYSTEM_PROMPT ขอไว้แล้วว่าห้าม wait คั่นกลางตอนกรอก login form แต่
# โมเดลเล็ก (เจอกับ Gemini flash-lite) ไม่ทำตามเสมอไป — สังเกตเห็นจริงว่าสั่ง wait
# เฉยๆ (ไม่มีความหมายเพราะหน้าไม่เปลี่ยน) แล้วรอบถัดไปข้ามไปกด element อื่น (เช่น ปุ่ม
# Login) ทั้งที่ยังไม่ได้กรอก password เลย — เพิ่ม code-level guard บังคับจริง:
# ถ้ามี input[type=password] ที่มองเห็นได้ยังว่างอยู่บนหน้าปัจจุบัน ห้ามทำ action อื่น
# นอกจาก "fill" (ไม่ว่าจะ fill ช่องไหนก็ตาม) เด็ดขาด — บล็อคทั้ง wait และการกด element
# อื่นๆ ทั้งหมด ไม่ใช่แค่ wait เพราะปัญหาจริงคือ "form ถูกทิ้งไว้ไม่ครบ" ไม่ใช่แค่ wait
# เฉยๆ กัน stall ตลอดไปด้วย retry จำกัดเหมือน guard อื่นๆ ในไฟล์นี้ ถ้าเกินโควตาแล้ว
# ยังไม่ยอมกรอก ปล่อยผ่านไปตามที่โมเดลเลือกแทนที่จะค้างไม่รู้จบ
_MAX_PREMATURE_LOGIN_SKIP_RETRIES = 2
_PREMATURE_LOGIN_SKIP_NUDGE = (
    "action นี้ถูกปฏิเสธ — หน้านี้ยังมีช่อง Password ที่ว่างอยู่ ห้ามข้ามไปทำ action อื่น "
    "(รวมถึง wait) จนกว่าจะกรอก Username และ Password ให้ครบก่อน ดู indexed elements "
    "แล้วเลือก fill ช่องที่ยังว่างอยู่ทันที"
)


async def _login_form_needs_password(page: Page) -> bool:
    """เช็คจาก DOM จริง (ไม่ใช่ label จาก snapshot เพราะแยกไม่ออกชัดพอระหว่าง
    placeholder กับค่าว่างจริง) ว่าหน้าปัจจุบันมี input[type=password] ที่มองเห็นได้
    และยังว่างอยู่ไหม — ใช้เป็นสัญญาณว่า login form ยังกรอกไม่ครบ"""
    try:
        password_inputs = page.locator('input[type="password"]:visible')
        count = await password_inputs.count()
        for i in range(count):
            value = await password_inputs.nth(i).input_value()
            if value == "":
                return True
        return False
    except Exception:
        return False

# หน่วงท้ายทุก step ที่ยังวนต่อ กันยิง LLM API ถี่เกิน free-tier quota ต่อนาที (RPM) —
# ไม่ใช่แค่ Gemini เจอ 429 ResourceExhausted เอง (ดู llm.py) provider อื่นก็มี rate
# limit เหมือนกัน แค่ชื่อ error ต่างกัน ค่านี้เป็น heuristic คร่าวๆ ไม่ได้ผูกกับ quota
# จริงเป๊ะๆ ของ key ไหน (แต่ละ key/โมเดลจำกัดไม่เท่ากัน)
_STEP_PACING_DELAY_SECONDS = 3


def _tokens_dict(usage: llm.TokenUsage) -> dict:
    return {
        "input": usage.input_tokens,
        "output": usage.output_tokens,
        "cache_read": usage.cache_read_tokens,
        "cache_creation": usage.cache_creation_tokens,
    }


async def _confirm_plan(plan_text: str, ask_user_func: Optional[AskUserFunc]) -> bool:
    """โชว์แผนแล้วรอ user ยืนยันก่อนเริ่ม loop จริง — ใช้ callback เดียวกับ permission
    layer (actions.AskUserFunc) เพื่อให้ชั้นบน (เช่น API server ใน W10) inject วิธีถาม
    ของตัวเองได้ (ส่ง event ไป UI แทน blocking input() ทาง terminal) โดยไม่ต้องแก้ตรงนี้
    """
    if ask_user_func is not None:
        return bool(await ask_user_func({"type": "confirm_plan", "plan": plan_text}))
    print("\n=== แผนที่ AI จะทำ ===", flush=True)
    print(plan_text, flush=True)
    print("========================", flush=True)
    choice = await asyncio.to_thread(input, "ยืนยันให้เริ่มทำงานตามแผนนี้หรือไม่? (y/n): ")
    return choice.strip().lower() in ("y", "yes")


class Orchestrator:
    def __init__(self):
        self.memory = ShortTermMemory()

    @staticmethod
    def _llm_backend(provider: str):
        """เลือก client/model/next_action/append_tool_result ตาม provider
        รองรับ "anthropic" (ตัวหลักตาม roadmap), "gemini" (provider สำรอง free tier
        กว้างกว่า) และ "groq" (ไว้ทดสอบตอนยังไม่มี Anthropic key จริง) — คืนรูปแบบ
        เดียวกันหมดให้ loop ข้างล่างเรียกแบบไม่ต้องรู้ว่าเป็น provider ไหน
        """
        if provider == "groq":
            return (
                llm.build_groq_client(settings.groq_api_key),
                settings.groq_model,
                llm.next_action_groq,
                llm.append_tool_result_groq,
            )
        if provider == "gemini":
            return (
                llm.build_gemini_client(settings.gemini_api_key),
                settings.gemini_model,
                llm.next_action_gemini,
                llm.append_tool_result_gemini,
            )
        if provider == "anthropic":
            return (
                llm.build_client(settings.anthropic_api_key),
                settings.anthropic_model,
                llm.next_action,
                llm.append_tool_result,
            )
        raise ValueError(f"ไม่รู้จัก LLM provider: {provider!r} (รองรับแค่ anthropic/gemini/groq)")

    async def run_task(
        self,
        url: str,
        goal: str,
        max_steps: int = 30,
        headless: bool | None = None,
        verbose: bool = False,
        provider: str | None = None,
        ask_user_func: Optional[AskUserFunc] = None,
        confirm_plan: bool = False,
    ) -> dict:
        """Perceive -> Plan -> Act loop บนหน้าเว็บเดียว จนกว่า LLM จะเรียก finish_task
        หรือครบ max_steps

        headless: None = ใช้ settings.browser_headless, True/False = บังคับ override
                  (เช่น run.py agent อยากเห็นหน้าต่าง browser จริงๆ ระหว่างรัน)
        verbose:  True = print แต่ละ step ลง terminal สดๆ ระหว่าง loop (ไว้ดูคู่กับ
                  หน้าต่าง browser ที่เปิดโชว์อยู่) — ปิดไว้ (False) ตอนเรียกจาก
                  API server ในอนาคต (W10) กัน log รก
        provider: None = ใช้ settings.llm_provider, หรือระบุ "anthropic"/"groq" ตรงๆ
        ask_user_func: callback (cmd/plan dict) -> bool ให้ชั้นบน (เช่น API server)
                  ตัดสินใจแทน blocking input() ทาง terminal — ใช้ร่วมกันทั้ง permission
                  layer (actions.execute) และ confirm_plan ด้านล่าง ถ้าไม่ส่งมา fallback
                  เป็น input() ทาง terminal ทั้งคู่
        confirm_plan: True = ก่อนเริ่ม loop จริง ให้ LLM ร่างแผนคร่าวๆ (llm.generate_plan)
                  โชว์ให้ user เห็นแล้วรอกดยืนยันก่อน — ถ้าไม่ยืนยัน จะไม่ลงมือทำ action
                  ใดๆ เลย (คืนผลลัพธ์ steps=0 ทันที) ไว้กัน agent เริ่มทำอะไรที่ user ยัง
                  ไม่ได้เห็นแผนมาก่อน

        W5: action ที่ fail จะถูก retry เงียบๆ ก่อนแล้ว (ดู actions.py::execute() ->
        _dispatch_with_retry) เฉพาะ click/fill/select/check — ถ้ายัง fail อยู่หลัง retry
        ครบ ผลลัพธ์สุดท้ายถึงจะถูกส่งกลับเข้าบทสนทนาให้ LLM เห็นแล้วตัดสินใจเองว่าจะลอง
        ทางอื่นยังไงในรอบถัดไป (เช่น index ผิดจริง ไม่ใช่แค่ DOM ยังไม่นิ่ง)

        W5 (verify, 2026-07-15): finish_task(success=true) ที่เรียกโดยยังไม่ทำ action
        ใดๆ เลย (steps_taken=0) จะไม่ถูกยอมรับทันที เตือนให้ยืนยันอีกครั้งก่อน (symmetric
        กับ guard ที่มีอยู่แล้วสำหรับ finish_task(false) ก่อนเวลาอันควร) — ผลลัพธ์ที่คืน
        กลับมามี key "final_page_state" เพิ่มด้วยเสมอ (page_text ของ get_snapshot() รอบ
        สุดท้ายก่อนจบ loop) ให้หลักฐานจริงจาก DOM เทียบกับ "message" ที่ LLM อ้างได้ ไม่
        ต้องเชื่อคำเคลมของ LLM ลอยๆ อย่างเดียว
        """
        is_headless = settings.browser_headless if headless is None else headless
        resolved_provider = provider or settings.llm_provider
        client, model, next_action, append_tool_result = self._llm_backend(resolved_provider)
        playwright = await async_playwright().start()
        browser = await playwright.chromium.launch(headless=is_headless)
        page = await browser.new_page()

        messages: list[dict] = []
        success = False
        final_message = "ครบ max_steps โดยยังไม่จบ task"
        steps_taken = 0
        total_usage = llm.TokenUsage()
        premature_false_finish_count = 0
        premature_true_finish_count = 0
        premature_login_skip_count = 0
        final_page_text = ""
        plan_text: Optional[str] = None
        last_action_cmd: Optional[dict] = None
        consecutive_repeat_count = 0
        recent_actions: list[dict] = []  # เก็บ action ล่าสุดไว้เช็ค pattern วนซ้ำ (คาบ 2-4)
        # W7[A] (context compaction, Gemini เท่านั้น): [(absolute_step_number,
        # len(messages) หลังจบ step นั้น), ...] — ใช้หา cut point ที่ปลอดภัย
        # (ตรงกับจุดเริ่ม turn ใหม่จริงๆ) ตอนบีบอัด ไม่ใช่ตำแหน่งเดา
        gemini_step_boundaries: list[tuple[int, int]] = []

        try:
            if verbose:
                print(f"[goto] {url}", flush=True)
            goto_result: ActionResult = await goto(page, url)
            self.memory.record({
                "step": 0,
                "cmd": {"type": "goto", "url": url},
                "result": str(goto_result),
                "success": goto_result.success,
            })
            if verbose:
                print(f"  -> {goto_result}", flush=True)
            await wait_stable(page)

            if confirm_plan:
                _, plan_page_text = await get_snapshot(page)
                plan_text = await llm.generate_plan(client, model, goal, plan_page_text, resolved_provider)
                if verbose:
                    print(f"[plan]\n{plan_text}", flush=True)
                approved = await _confirm_plan(plan_text, ask_user_func)
                if not approved:
                    if verbose:
                        print("[plan] ผู้ใช้ไม่ยืนยัน — ยกเลิกก่อนเริ่มทำงาน", flush=True)
                    return {
                        "success": False,
                        "steps": 0,
                        "message": "ผู้ใช้ไม่ยืนยันแผน — ยกเลิกก่อนเริ่มทำงาน",
                        "history": self.memory.recent(max_steps),
                        "tokens": _tokens_dict(total_usage),
                        "plan": plan_text,
                        "final_page_state": plan_page_text,
                    }

            for _ in range(max_steps):
                elements, page_text = await get_snapshot(page)
                # W5[A] verify: เก็บ page_text ล่าสุดไว้เป็นหลักฐานจริงจาก DOM ตอนจบ
                # task (ทุก path — finish_task/loop-detected/หมด max_steps) แนบไปกับ
                # result ให้ผู้ประเมิน (เช่น W12[B] eval script/human review) เทียบกับ
                # message ที่ LLM อ้างได้เอง ไม่ต้องเชื่อคำเคลมของ LLM ลอยๆ อย่างเดียว
                final_page_text = page_text

                # W6[B]: ดึงคู่มือที่เกี่ยวข้องกับ goal+หน้าปัจจุบันใหม่ทุก step (retrieve()
                # ไม่ throw เอง คืน [] เงียบๆ ถ้าไม่มีคู่มือ/error) — ใช้ to_thread เพราะ
                # เป็นงาน sync (local embedding inference + ChromaDB query) ไม่งั้นจะบล็อก
                # event loop ตัวเดียวกับที่ Playwright ใช้อยู่ (เหมือน _confirm_plan()
                # ที่ wrap input() ด้วย to_thread ด้วยเหตุผลเดียวกัน)
                manual_chunks = await asyncio.to_thread(
                    retriever.retrieve, query=goal, page_state=page_text, k=_RAG_CHUNKS_PER_STEP
                )
                manual_context = "\n".join(f"- {chunk}" for chunk in manual_chunks)

                # W7[A]: สรุป action ที่ล้มเหลวไปแล้วใน task นี้ (ดู
                # ShortTermMemory.failed_actions_summary() docstring) ป้อนกลับเข้า prompt
                # ทุก step เหมือน manual_context — ว่างเปล่าถ้ายังไม่เคย fail อะไรเลย
                memory_context = self.memory.failed_actions_summary()

                # W7[A] (long-term): ดึงประวัติ task run อื่นก่อนหน้าที่เกี่ยวข้องกับ
                # goal+หน้าปัจจุบัน (recall() ไม่ throw เอง คืน [] เงียบๆ เหมือน
                # retriever.retrieve()) — to_thread ด้วยเหตุผลเดียวกับ manual retrieve ด้านบน
                long_term_chunks = await asyncio.to_thread(
                    long_term_memory.recall,
                    query=goal, page_state=page_text, k=_LONG_TERM_MEMORY_CHUNKS_PER_STEP,
                )
                long_term_context = "\n".join(f"- {chunk}" for chunk in long_term_chunks)

                tool_name, tool_input, tool_use_id, messages, usage = await next_action(
                    client, model, goal, page_text, messages, manual_context, memory_context, long_term_context
                )
                total_usage += usage
                if verbose:
                    print(
                        f"  [tokens] input={usage.input_tokens} output={usage.output_tokens}"
                        f" cache_read={usage.cache_read_tokens} cache_write={usage.cache_creation_tokens}"
                        f" (รวม: input={total_usage.input_tokens} output={total_usage.output_tokens}"
                        f" cache_read={total_usage.cache_read_tokens} cache_write={total_usage.cache_creation_tokens})",
                        flush=True,
                    )

                if tool_name == "finish_task":
                    claimed_success = bool(tool_input.get("success", False))

                    # ยังเหลือ step ให้ลอง + เป็น finish_task call จริง (มี tool_use_id ให้
                    # ผูก tool_result กลับ ไม่ใช่ fallback ตอนโมเดลไม่ยอมเรียก tool เลย) +
                    # ยังไม่เกิน quota การเตือน -> ไม่ยอมรับ false ทันที เตือนแล้วให้ลองต่อ
                    if (
                        not claimed_success
                        and tool_use_id
                        and steps_taken < max_steps - 1
                        and premature_false_finish_count < _MAX_PREMATURE_FALSE_FINISH_RETRIES
                    ):
                        premature_false_finish_count += 1
                        if verbose:
                            print(
                                f"[finish_task(false) ไม่ยอมรับ {premature_false_finish_count}/"
                                f"{_MAX_PREMATURE_FALSE_FINISH_RETRIES}] message={tool_input.get('message', '')}",
                                flush=True,
                            )
                        # 1. ป้อนค่ากลับฝั่ง Tool ปกติเพื่อป้องกันโครงสร้างประวัติพัง
                        messages = append_tool_result(messages, tool_use_id, _PREMATURE_FALSE_FINISH_NUDGE)

                        # 2. ฉีด User Prompt ซ้ำเข้าไปท้ายบทสนทนา (ช่วยดึงสติโมเดลขนาดเล็กอย่าง Llama ได้ดีมาก)
                        messages.append(_build_nudge_message(
                            resolved_provider,
                            f"⚠️ [ระบบคำสั่งสำคัญ]: การเรียก finish_task(false) รอบล่าสุดถูกปฏิเสธอย่างสิ้นเชิง! "
                            f"ตรวจพบว่าเป้าหมาย '{goal}' ยังไม่สมบูรณ์ และหน้าเว็บยังมี Elements เหลืออยู่ "
                            f"ห้ามกดยอมแพ้จนกว่าจะลองพยายาม Action กับส่วนที่เหลือ ดูลิสต์ใหม่อีกครั้งแล้วทำต่อ!",
                        ))
                        continue

                    # W5[A] verify: symmetric กับ guard ด้านบนแต่ฝั่ง true — เรียก
                    # finish_task(success=true) เป็น action แรกสุด (steps_taken=0) ยัง
                    # ไม่มีหลักฐานว่าทำอะไรจริงเลย ให้ยืนยันอีกครั้งก่อนยอมรับ (ไม่ block
                    # เด็ดขาด เผื่อ goal สำเร็จอยู่แล้วตั้งแต่ page แรกจริงๆ)
                    if (
                        claimed_success
                        and tool_use_id
                        and steps_taken == 0
                        and premature_true_finish_count < _MAX_PREMATURE_TRUE_FINISH_RETRIES
                    ):
                        premature_true_finish_count += 1
                        if verbose:
                            print(
                                f"[finish_task(true) ไม่ยอมรับทันที {premature_true_finish_count}/"
                                f"{_MAX_PREMATURE_TRUE_FINISH_RETRIES}] message={tool_input.get('message', '')}",
                                flush=True,
                            )
                        messages = append_tool_result(messages, tool_use_id, _PREMATURE_TRUE_FINISH_NUDGE)
                        messages.append(_build_nudge_message(
                            resolved_provider,
                            f"⚠️ [ระบบคำสั่งสำคัญ]: การเรียก finish_task(true) โดยยังไม่ทำ action ใดๆ เลย "
                            f"ต้องมีหลักฐานชัดเจนจาก indexed elements ปัจจุบันว่าเป้าหมาย '{goal}' สำเร็จแล้ว "
                            f"จริงๆ ก่อนยืนยันอีกครั้ง",
                        ))
                        continue

                    success = claimed_success
                    final_message = tool_input.get("message", "")
                    if verbose:
                        print(f"[finish_task] success={success} message={final_message}", flush=True)
                    break

                # code-level guard (2026-07-13): ห้ามทำ action อื่นนอกจาก "fill" ถ้าหน้า
                # ปัจจุบันยังมีช่อง password ว่างอยู่ — กัน agent สั่ง wait/click ข้ามไป
                # ทั้งที่ login form ยังกรอกไม่ครบ (SYSTEM_PROMPT ขอไว้แล้วแต่โมเดลเล็ก
                # ไม่ทำตามเสมอไป จึงต้องบังคับด้วยโค้ดจริง ไม่ใช่แค่ขอทางคำสั่ง)
                #
                # *** ยกเว้น "goto" เสมอ — ระบบอาจจำเป็นต้อง goto ไปหน้าอื่นก่อน (เช่น
                # แก้เส้นทางที่ผิด, หรือ multi-hop กว่าจะถึงฟอร์ม login จริง) ห้ามดักเช็ค
                # สถานะฟอร์มของหน้าปัจจุบันจนบล็อก goto ไม่ให้ออกจากหน้านั้นได้เลย —
                # ปล่อยผ่านทันทีเสมอไม่ว่า password จะว่างอยู่หรือไม่ ***
                if (
                    tool_input.get("type") not in ("fill", "goto")
                    and await _login_form_needs_password(page)
                ):
                    if premature_login_skip_count < _MAX_PREMATURE_LOGIN_SKIP_RETRIES:
                        premature_login_skip_count += 1
                        if verbose:
                            print(
                                f"[login-form ยังไม่ครบ {premature_login_skip_count}/"
                                f"{_MAX_PREMATURE_LOGIN_SKIP_RETRIES}] ปฏิเสธ action={tool_input}",
                                flush=True,
                            )
                        messages = append_tool_result(messages, tool_use_id, _PREMATURE_LOGIN_SKIP_NUDGE)
                        messages.append(_build_nudge_message(
                            resolved_provider,
                            "⚠️ [ระบบคำสั่งสำคัญ]: หน้านี้ยังมีช่อง Password ที่ว่างอยู่ "
                            "ห้ามข้ามไปทำ action อื่น (รวมถึง wait) จนกว่าจะกรอก Username "
                            "และ Password ให้ครบก่อน ดู indexed elements แล้วเลือก fill "
                            "ช่องที่ยังว่างอยู่ทันที",
                        ))
                        continue
                    # เกินโควตาเตือนแล้วยังไม่ยอมกรอก ปล่อยผ่านไปตามที่โมเดลเลือกแทนที่จะ
                    # ค้างไม่รู้จบ (เหมือน escape valve ของ premature-false-finish guard)

                # loop-detection: action เดิมเป๊ะๆ ติดกันกี่ครั้งแล้ว (นับรวมทั้ง success/fail
                # เพราะแม้ execute() สำเร็จทุกครั้ง แต่ถ้า LLM สั่งซ้ำเดิมไม่เปลี่ยน ก็ไม่ใช่
                # ความคืบหน้าจริงอยู่ดี)
                if tool_input == last_action_cmd:
                    consecutive_repeat_count += 1
                else:
                    last_action_cmd = tool_input
                    consecutive_repeat_count = 1

                if consecutive_repeat_count >= _MAX_CONSECUTIVE_IDENTICAL_ACTIONS:
                    success = False
                    final_message = (
                        f"หยุด task: agent สั่ง action เดิมซ้ำติดกัน "
                        f"{consecutive_repeat_count} ครั้ง ({tool_input}) โดยไม่มีความคืบหน้า"
                    )
                    if verbose:
                        print(f"[loop-detected] {final_message}", flush=True)
                    break

                # loop-detection (2026-07-13, generalize 2026-07-15): จับ pattern วนซ้ำ
                # เป็นคาบ (คาบ 2 เช่น go_back -> click -> go_back -> click, คาบ 3 เช่น
                # click A -> scroll -> fill B -> click A -> scroll -> fill B, ...) ที่
                # guard ด้านบน (คาบ 1) จับไม่ได้เพราะ action แต่ละตัวไม่ได้ "เดิมเป๊ะๆ
                # ติดกัน" — เก็บ history แค่ _MAX_CYCLE_WINDOW ตัวล่าสุดพอ ไม่ต้องเก็บ
                # ทั้ง task (ดู _detect_repeating_cycle_period()/_is_repeating_cycle()
                # ด้านบนสุดของไฟล์)
                recent_actions.append(tool_input)
                if len(recent_actions) > _MAX_CYCLE_WINDOW:
                    recent_actions.pop(0)

                detected_period = _detect_repeating_cycle_period(recent_actions)
                if detected_period is not None:
                    success = False
                    cycle_desc = " -> ".join(str(a) for a in recent_actions[-detected_period:])
                    final_message = (
                        f"หยุด task: agent วน action ซ้ำเป็นคาบ {detected_period} "
                        f"({cycle_desc}) โดยไม่มีความคืบหน้า"
                    )
                    if verbose:
                        print(f"[loop-detected] {final_message}", flush=True)
                    break

                if verbose:
                    print(f"[step {steps_taken + 1}] {tool_input}", flush=True)

                # label ของ element เป้าหมาย (จาก snapshot เดียวกับที่ LLM เพิ่งเห็น) ส่ง
                # ให้ execute()/classify_action() เช็คคำเสี่ยงเป็นชั้นสำรอง เผื่อ LLM
                # เลือก type="click" ธรรมดากับปุ่มที่จริงๆ มีผลสำคัญ (เช่น "Remove")
                action_index = tool_input.get("index")
                action_label = next(
                    (e["label"] for e in elements if e["index"] == action_index), ""
                ) if action_index is not None else ""

                # W7[B]: RAG-based permission — ดึงคู่มือด้วย query แคบเฉพาะ action นี้
                # (ไม่ใช่ manual_context ด้านบนที่ query=goal กว้างทั้ง task) แล้วส่งให้
                # execute()/classify_action() เช็คว่าคู่มือระบุไว้ไหมว่า action นี้ต้องขอ
                # อนุมัติ (ดู _build_permission_query()/_PERMISSION_RAG_CHUNKS_PER_STEP
                # ด้านบนสำหรับเหตุผลที่แยก query)
                permission_query = _build_permission_query(tool_input, action_label)
                permission_chunks = (
                    await asyncio.to_thread(
                        retriever.retrieve, query=permission_query, k=_PERMISSION_RAG_CHUNKS_PER_STEP
                    )
                    if permission_query else []
                )
                manual_permission_guidance = "\n".join(f"- {c}" for c in permission_chunks)

                result: ActionResult = await execute(
                    page, tool_input, ask_user_func=ask_user_func, label=action_label,
                    manual_guidance=manual_permission_guidance,
                )
                steps_taken += 1
                self.memory.record({
                    "step": steps_taken,
                    "cmd": tool_input,
                    "result": str(result),
                    "success": result.success,
                    "tokens": _tokens_dict(usage),
                })
                if verbose:
                    print(f"  -> {result}", flush=True)

                messages = append_tool_result(messages, tool_use_id, str(result))

                # W7[A] (context compaction, Gemini เท่านั้น): เก็บ boundary ของ step
                # นี้ แล้วเช็คว่าต้องบีบอัดหรือยัง (ดูคอมเมนต์ยาวที่ _GEMINI_COMPACT_AFTER_STEPS
                # ด้านบนสุดของไฟล์ — เหตุผลที่ scope แค่ Gemini)
                if resolved_provider == "gemini":
                    gemini_step_boundaries.append((steps_taken, len(messages)))
                    if len(gemini_step_boundaries) > _GEMINI_COMPACT_AFTER_STEPS:
                        cut_list_index = len(gemini_step_boundaries) - _GEMINI_KEEP_RECENT_STEPS
                        cut_step_num, cut_at = gemini_step_boundaries[cut_list_index - 1]
                        digest = _build_gemini_history_digest(self.memory, upto_step=cut_step_num)
                        messages = _compact_gemini_messages(messages, cut_at, digest)
                        gemini_step_boundaries = [
                            (s, b - cut_at) for s, b in gemini_step_boundaries[cut_list_index:]
                        ]
                        if verbose:
                            print(
                                f"[gemini-compact] ย่อ step 1..{cut_step_num} เหลือ digest เดียว "
                                f"(เก็บ {_GEMINI_KEEP_RECENT_STEPS} step ล่าสุดแบบ raw)",
                                flush=True,
                            )

                if tool_input.get("type") in _PAGE_CHANGING_ACTIONS:
                    await wait_stable(page)

                # หน่วงท้าย step ก่อนวน next_action() รอบถัดไป กันยิง LLM API ถี่เกิน
                # quota ต่อนาที (ดู _STEP_PACING_DELAY_SECONDS ด้านบน)
                await asyncio.sleep(_STEP_PACING_DELAY_SECONDS)

            # W7[A] (long-term): บันทึกผลลัพธ์ของ task run นี้ไว้ให้ task run ถัดไป
            # (บน goal/หน้าเว็บที่เกี่ยวข้องกัน) recall() กลับมาใช้ได้ — เรียกครั้งเดียว
            # ตอนจบ loop จริง (ทุก path: finish_task, loop-detected, หมด max_steps)
            # ไม่ครอบ confirm_plan declined เพราะ return ไปก่อนถึงจุดนี้แล้ว (ไม่มี
            # action ใดๆ เกิดขึ้นจริงเลย ไม่มีอะไรให้บันทึกเป็น pattern)
            await asyncio.to_thread(
                long_term_memory.record_task,
                url=url, goal=goal, success=success, message=final_message,
                failed_actions=self.memory.failed_actions_summary(),
            )

            return {
                "success": success,
                "steps": steps_taken,
                "message": final_message,
                "history": self.memory.recent(max_steps),
                "tokens": _tokens_dict(total_usage),
                "plan": plan_text,
                "final_page_state": final_page_text,
            }
        finally:
            await browser.close()
            await playwright.stop()

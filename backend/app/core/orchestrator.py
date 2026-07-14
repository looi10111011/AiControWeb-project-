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
_MAX_ALTERNATING_CYCLES = 2  # ครบ 2 รอบ A-B-A-B (รวม 4 action) ถือว่าติด loop
_ALTERNATING_WINDOW = _MAX_ALTERNATING_CYCLES * 2


def _is_alternating_pattern(history: list[dict]) -> bool:
    """เช็คว่า _ALTERNATING_WINDOW action ล่าสุดเป็น A-B-A-B-... สลับกันแค่ 2 ค่า
    ไปเรื่อยๆ หรือไม่ — ไม่นับกรณี A กับ B เป็นค่าเดียวกัน (นั่นคือ AAAA ซึ่งมี
    _MAX_CONSECUTIVE_IDENTICAL_ACTIONS ด้านบนจับแยกไปแล้ว กันสอง guard ทับซ้อนกัน)"""
    if len(history) < _ALTERNATING_WINDOW:
        return False
    a, b = history[0], history[1]
    if a == b:
        return False
    return all(history[i] == (a if i % 2 == 0 else b) for i in range(_ALTERNATING_WINDOW))

# W6[B]: จำนวน chunk คู่มือสูงสุดที่จะดึงมาแนบให้ LLM เห็นทุก step ของ per-step loop —
# ดึงใหม่ทุก step ตาม page_text ปัจจุบัน (ไม่ใช้กับ generate_plan ซึ่งเป็นแค่แผนคร่าวๆ
# ครั้งเดียวก่อนเริ่ม loop จริง เก็บ scope ไว้แค่ per-step planner ตามที่คุยกันไว้)
_RAG_CHUNKS_PER_STEP = 3

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
        premature_login_skip_count = 0
        plan_text: Optional[str] = None
        last_action_cmd: Optional[dict] = None
        consecutive_repeat_count = 0
        recent_actions: list[dict] = []  # เก็บ action ล่าสุดไว้เช็ค pattern สลับกัน (ABAB)

        try:
            if verbose:
                print(f"[goto] {url}", flush=True)
            goto_result: ActionResult = await goto(page, url)
            self.memory.record({"step": 0, "cmd": {"type": "goto", "url": url}, "result": str(goto_result)})
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
                    }

            for _ in range(max_steps):
                elements, page_text = await get_snapshot(page)

                # W6[B]: ดึงคู่มือที่เกี่ยวข้องกับ goal+หน้าปัจจุบันใหม่ทุก step (retrieve()
                # ไม่ throw เอง คืน [] เงียบๆ ถ้าไม่มีคู่มือ/error) — ใช้ to_thread เพราะ
                # เป็นงาน sync (local embedding inference + ChromaDB query) ไม่งั้นจะบล็อก
                # event loop ตัวเดียวกับที่ Playwright ใช้อยู่ (เหมือน _confirm_plan()
                # ที่ wrap input() ด้วย to_thread ด้วยเหตุผลเดียวกัน)
                manual_chunks = await asyncio.to_thread(
                    retriever.retrieve, query=goal, page_state=page_text, k=_RAG_CHUNKS_PER_STEP
                )
                manual_context = "\n".join(f"- {chunk}" for chunk in manual_chunks)

                tool_name, tool_input, tool_use_id, messages, usage = await next_action(
                    client, model, goal, page_text, messages, manual_context
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
                        messages.append({
                            "role": "user",
                            "content": f"⚠️ [ระบบคำสั่งสำคัญ]: การเรียก finish_task(false) รอบล่าสุดถูกปฏิเสธอย่างสิ้นเชิง! "
                                       f"ตรวจพบว่าเป้าหมาย '{goal}' ยังไม่สมบูรณ์ และหน้าเว็บยังมี Elements เหลืออยู่ "
                                       f"ห้ามกดยอมแพ้จนกว่าจะลองพยายาม Action กับส่วนที่เหลือ ดูลิสต์ใหม่อีกครั้งแล้วทำต่อ!"
                        })
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
                        messages.append({
                            "role": "user",
                            "content": "⚠️ [ระบบคำสั่งสำคัญ]: หน้านี้ยังมีช่อง Password ที่ว่างอยู่ "
                                       "ห้ามข้ามไปทำ action อื่น (รวมถึง wait) จนกว่าจะกรอก Username "
                                       "และ Password ให้ครบก่อน ดู indexed elements แล้วเลือก fill "
                                       "ช่องที่ยังว่างอยู่ทันที",
                        })
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

                # loop-detection (2026-07-13): จับ pattern สลับ 2 action ไปมา (คาบ 2
                # เช่น go_back -> click -> go_back -> click) ที่ guard ด้านบน (คาบ 1)
                # จับไม่ได้เพราะ action แต่ละตัวไม่ได้ "เดิมเป๊ะๆ ติดกัน" — เก็บ history
                # แค่ _ALTERNATING_WINDOW ตัวล่าสุดพอ ไม่ต้องเก็บทั้ง task
                recent_actions.append(tool_input)
                if len(recent_actions) > _ALTERNATING_WINDOW:
                    recent_actions.pop(0)

                if _is_alternating_pattern(recent_actions):
                    success = False
                    final_message = (
                        f"หยุด task: agent วนสลับ 2 action ซ้ำๆ "
                        f"({recent_actions[-2]} <-> {recent_actions[-1]}) โดยไม่มีความคืบหน้า"
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

                result: ActionResult = await execute(
                    page, tool_input, ask_user_func=ask_user_func, label=action_label
                )
                steps_taken += 1
                self.memory.record({
                    "step": steps_taken,
                    "cmd": tool_input,
                    "result": str(result),
                    "tokens": _tokens_dict(usage),
                })
                if verbose:
                    print(f"  -> {result}", flush=True)

                messages = append_tool_result(messages, tool_use_id, str(result))

                if tool_input.get("type") in _PAGE_CHANGING_ACTIONS:
                    await wait_stable(page)

                # หน่วงท้าย step ก่อนวน next_action() รอบถัดไป กันยิง LLM API ถี่เกิน
                # quota ต่อนาที (ดู _STEP_PACING_DELAY_SECONDS ด้านบน)
                await asyncio.sleep(_STEP_PACING_DELAY_SECONDS)

            return {
                "success": success,
                "steps": steps_taken,
                "message": final_message,
                "history": self.memory.recent(max_steps),
                "tokens": _tokens_dict(total_usage),
                "plan": plan_text,
            }
        finally:
            await browser.close()
            await playwright.stop()

"""Agent Loop: Perceive -> Plan -> Act -> Verify.

W1: skeleton only. W4: ทำ loop จริงกับเว็บง่าย 1 หน้า. W5: เพิ่ม verify/retry.
"""

from playwright.async_api import async_playwright

from backend.app.config import settings
from backend.app.core import llm
from backend.app.core.actions import ActionResult, execute, goto, wait_stable
from backend.app.core.memory import ShortTermMemory
from backend.app.core.perception import get_snapshot

# action ที่เปลี่ยนหน้า/DOM แบบมีนัยสำคัญ -> ต้องรอหน้านิ่งก่อน perceive รอบถัดไป
_PAGE_CHANGING_ACTIONS = {"click", "goto", "select", "go_back"}


class Orchestrator:
    def __init__(self):
        self.memory = ShortTermMemory()

    @staticmethod
    def _llm_backend(provider: str):
        """เลือก client/model/next_action/append_tool_result ตาม provider
        รองรับ "anthropic" (ตัวหลักตาม roadmap) และ "groq" (ไว้ทดสอบตอนยังไม่มี
        Anthropic key จริง) — คืนรูปแบบเดียวกันหมดให้ loop ข้างล่างเรียกแบบไม่ต้อง
        รู้ว่าเป็น provider ไหน
        """
        if provider == "groq":
            return (
                llm.build_groq_client(settings.groq_api_key),
                settings.groq_model,
                llm.next_action_groq,
                llm.append_tool_result_groq,
            )
        if provider == "anthropic":
            return (
                llm.build_client(settings.anthropic_api_key),
                settings.anthropic_model,
                llm.next_action,
                llm.append_tool_result,
            )
        raise ValueError(f"ไม่รู้จัก LLM provider: {provider!r} (รองรับแค่ anthropic/groq)")

    async def run_task(
        self,
        url: str,
        goal: str,
        max_steps: int = 15,
        headless: bool | None = None,
        verbose: bool = False,
        provider: str | None = None,
    ) -> dict:
        """Perceive -> Plan -> Act loop บนหน้าเว็บเดียว จนกว่า LLM จะเรียก finish_task
        หรือครบ max_steps

        headless: None = ใช้ settings.browser_headless, True/False = บังคับ override
                  (เช่น run.py agent อยากเห็นหน้าต่าง browser จริงๆ ระหว่างรัน)
        verbose:  True = print แต่ละ step ลง terminal สดๆ ระหว่าง loop (ไว้ดูคู่กับ
                  หน้าต่าง browser ที่เปิดโชว์อยู่) — ปิดไว้ (False) ตอนเรียกจาก
                  API server ในอนาคต (W10) กัน log รก
        provider: None = ใช้ settings.llm_provider, หรือระบุ "anthropic"/"groq" ตรงๆ

        W4 v1: ไม่มี retry เมื่อ action ล้มเหลว (เป็นของ W5) — ผลลัพธ์ action ที่ fail
        จะถูกส่งกลับเข้าบทสนทนาให้ LLM เห็นแล้วตัดสินใจเองว่าจะลองทางอื่นยังไงในรอบถัดไป
        """
        is_headless = settings.browser_headless if headless is None else headless
        client, model, next_action, append_tool_result = self._llm_backend(
            provider or settings.llm_provider
        )
        playwright = await async_playwright().start()
        browser = await playwright.chromium.launch(headless=is_headless)
        page = await browser.new_page()

        messages: list[dict] = []
        success = False
        final_message = "ครบ max_steps โดยยังไม่จบ task"
        steps_taken = 0
        total_usage = llm.TokenUsage()

        try:
            if verbose:
                print(f"[goto] {url}", flush=True)
            goto_result: ActionResult = await goto(page, url)
            self.memory.record({"step": 0, "cmd": {"type": "goto", "url": url}, "result": str(goto_result)})
            if verbose:
                print(f"  -> {goto_result}", flush=True)
            await wait_stable(page)

            for _ in range(max_steps):
                _, page_text = await get_snapshot(page)

                tool_name, tool_input, tool_use_id, messages, usage = await next_action(
                    client, model, goal, page_text, messages
                )
                total_usage += usage
                if verbose:
                    print(
                        f"  [tokens] input={usage.input_tokens} output={usage.output_tokens}"
                        f" (รวม: input={total_usage.input_tokens} output={total_usage.output_tokens})",
                        flush=True,
                    )

                if tool_name == "finish_task":
                    success = bool(tool_input.get("success", False))
                    final_message = tool_input.get("message", "")
                    if verbose:
                        print(f"[finish_task] success={success} message={final_message}", flush=True)
                    break

                if verbose:
                    print(f"[step {steps_taken + 1}] {tool_input}", flush=True)

                result: ActionResult = await execute(page, tool_input)
                steps_taken += 1
                self.memory.record({
                    "step": steps_taken,
                    "cmd": tool_input,
                    "result": str(result),
                    "tokens": {"input": usage.input_tokens, "output": usage.output_tokens},
                })
                if verbose:
                    print(f"  -> {result}", flush=True)

                messages = append_tool_result(messages, tool_use_id, str(result))

                if tool_input.get("type") in _PAGE_CHANGING_ACTIONS:
                    await wait_stable(page)

            return {
                "success": success,
                "steps": steps_taken,
                "message": final_message,
                "history": self.memory.recent(max_steps),
                "tokens": {"input": total_usage.input_tokens, "output": total_usage.output_tokens},
            }
        finally:
            await browser.close()
            await playwright.stop()

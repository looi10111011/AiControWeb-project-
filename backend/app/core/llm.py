"""LLM planning: หน้าเว็บ (indexed elements) + goal -> action ถัดไป

W4: ใช้ tool-use / function calling แทนการให้ LLM ตอบ JSON เป็น text แล้วมาพาร์สเอง
    — response กลับมาเป็น tool call ที่ schema ถูกบังคับโดย API เลย ไม่ต้องกังวลเรื่อง
    markdown fence / คำอธิบายแถม / JSON ผิดรูปแบบ

ผูกกับ actions.execute()'s cmd dict โดยตรง: tool "browser_action" คืน dict ที่ยิงเข้า
execute(page, cmd) ได้ทันที ส่วน tool "finish_task" คือสัญญาณให้ orchestrator หยุด loop

รองรับ 2 provider:
  - Anthropic (Claude) — ตัวหลักตาม roadmap
  - Groq — ใช้ทดสอบ agent loop ชั่วคราวตอนยังไม่มี Anthropic key จริง (มี free tier)
ทั้งคู่คืนค่ารูปแบบเดียวกัน (tool_name, tool_input, tool_use_id, messages, usage) ให้
orchestrator.py เรียกใช้แบบไม่ต้องรู้ว่าข้างในเป็น provider ไหน — usage คือจำนวน token
ที่ใช้ไปในการเรียก LLM รอบนี้ (รวมทุก retry ถ้ามี) ไว้ให้ orchestrator log/สรุปได้
"""

import json
from dataclasses import dataclass
from typing import Any

from anthropic import AsyncAnthropic
from groq import AsyncGroq, BadRequestError as GroqBadRequestError


@dataclass
class TokenUsage:
    """จำนวน token ที่ใช้ไปในการเรียก LLM หนึ่งรอบ (รวมทุก retry ถ้ามี)"""

    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
        )

# บาง Llama model บน Groq บางครั้ง generate tool call ผิดรูปแบบ (เช่น
# "<function=...>" แทน JSON ที่ API คาดหวัง) ทำให้ได้ 400 tool_use_failed —
# ส่วนใหญ่เป็นเรื่อง sampling แบบสุ่ม ลองยิงซ้ำมักผ่าน ไม่ใช่บั๊กโค้ดเรา
_GROQ_TOOL_CALL_RETRIES = 3

# บางครั้ง Llama ตอบเป็นข้อความเฉยๆ โดยไม่เรียก tool เลย แม้ tool_choice="required"
# จะบังคับไว้แล้ว — แทนที่จะยอมแพ้แล้ว finish_task ทันที ให้เตือนแล้วลองใหม่ก่อน
_GROQ_NO_TOOL_CALL_RETRIES = 3
_NO_TOOL_CALL_NUDGE = (
    "คุณต้องเรียก tool (browser_action หรือ finish_task) เท่านั้น ห้ามพิมพ์ข้อความเฉยๆ "
    "โดยไม่เรียก tool ลองใหม่อีกครั้ง"
)

SYSTEM_PROMPT = """คุณคือ AI agent ที่ควบคุมหน้าเว็บผ่าน browser เพื่อทำ goal ที่ user สั่ง

ทุกครั้งคุณจะได้รับ "indexed elements" ของหน้าเว็บปัจจุบัน เช่น:
  [0] input(text) 'Username'
  [1] input(password) 'Password'
  [2] input(submit) 'Login'

กติกา:
- เลือก action ได้จาก index ที่เห็นในหน้าปัจจุบันเท่านั้น ห้ามเดา index ที่ไม่อยู่ในลิสต์
- ทำทีละ 1 action ต่อ 1 ครั้ง (ห้ามสั่งหลาย action พร้อมกัน)
- ถ้า action ก่อนหน้าล้มเหลว ([FAIL]/success=false) ให้ดู element ล่าสุดแล้วลองทางอื่น
  ห้ามยิง action เดิมซ้ำแบบไม่เปลี่ยนอะไรเลย
- ห้ามเรียก finish_task ก่อนที่จะลองทำ action จริงอย่างน้อย 1 ครั้ง เว้นแต่เห็นชัดเจน
  จากหน้าปัจจุบันว่า goal สำเร็จอยู่แล้ว — ถ้ายังไม่ได้ลองอะไรเลย ให้เรียก browser_action
  ก่อนเสมอ อย่าเพิ่งบอกว่า "จะทำ X" เฉยๆ โดยไม่เรียก tool จริง

- ถ้า goal มีหลายขั้นตอน/หลายส่วน (เช่น "login แล้วเพิ่มสินค้าลงตะกร้า" = 2 ส่วน:
  ล็อกอินให้สำเร็จ + เพิ่มสินค้าลงตะกร้า) ให้แตกเป็น sub-goal ในใจก่อนเริ่ม แล้วเช็คทีละ
  sub-goal ว่าทำครบจริงหรือยัง — กรอกฟอร์มเสร็จ (fill username/password) ไม่ใช่ "ล็อกอิน
  สำเร็จ" ต้องกดปุ่ม submit/login ด้วย แล้วเห็นว่าหน้าเปลี่ยนไปแล้ว (URL เปลี่ยน/เห็นเมนู
  ใหม่/ไม่เห็นฟอร์ม login แล้ว) ถึงจะถือว่า sub-goal นั้นเสร็จจริง
- ห้ามเรียก finish_task(success=true) จนกว่าจะเช็คจากหน้าเว็บปัจจุบัน (indexed elements
  ล่าสุดที่ได้รับ) แล้วเห็นหลักฐานชัดเจนว่า "ทุกส่วน" ของ goal สำเร็จแล้วจริงๆ — การกรอก
  ฟอร์มเสร็จ, การคลิกปุ่มไปแล้ว, หรือ action ก่อนหน้าคืน [OK] ไม่ได้แปลว่า goal สำเร็จ
  ต้องดู "ผลลัพธ์บนหน้าเว็บ" เป็นหลักฐาน ไม่ใช่แค่ว่า action ล่าสุดไม่ error
- ถ้ายังไม่สำเร็จแต่ในหน้าปัจจุบันยังมี element ที่ชัดเจนว่าต้องทำต่อ (เช่น เห็นช่อง
  password ว่างอยู่, เห็นปุ่ม Login ที่ยังไม่ได้กด, เห็นปุ่ม Add to cart ที่ยังไม่ได้กด)
  ให้เรียก browser_action ทำขั้นต่อไปทันที — ห้ามเรียก finish_task(success=false) แค่เพราะ
  "ยังไม่เสร็จ" ทั้งที่ยังมีทางไปต่อชัดเจนอยู่ตรงหน้า
- finish_task(success=false) ใช้เฉพาะตอนที่ลองหลายทางแล้วจริงๆ ไปต่อไม่ได้ (เช่น element
  ที่ต้องการหาไม่เจอซ้ำหลายรอบ, เจอ error ที่แก้ไม่ได้) ไม่ใช่ตัวเลือกเริ่มต้นเมื่อ goal
  ยังไม่เสร็จ
- เมื่อ goal สำเร็จครบทุกส่วนแล้ว (มีหลักฐานจากหน้าเว็บ) หรือทำต่อไม่ได้จริงๆ (ลองหลายทาง
  แล้วไม่สำเร็จ) ให้เรียก finish_task พร้อมสรุปผล
"""

# --- schema ของ tool ทั้ง 2 ตัว ใช้ร่วมกันระหว่าง Anthropic/Groq (แค่ห่อ format ต่างกัน) ---

_BROWSER_ACTION_PARAMS = {
    "type": "object",
    "properties": {
        "type": {
            "type": "string",
            "enum": [
                "click", "fill", "select", "check",
                "scroll", "goto", "go_back", "switch_tab", "wait",
            ],
            "description": "ชนิด action",
        },
        "index": {"type": "integer", "description": "index ของ element (click/fill/select/check)"},
        "text": {"type": "string", "description": "ข้อความที่จะกรอก (fill)"},
        "label": {"type": "string", "description": "ตัวเลือกที่จะเลือกใน dropdown (select)"},
        "direction": {"type": "string", "enum": ["up", "down"], "description": "ทิศทางเลื่อนจอ (scroll)"},
        "url": {"type": "string", "description": "URL ปลายทาง (goto)"},
        "tab_index": {"type": "integer", "description": "ลำดับ tab ที่จะสลับไป (switch_tab)"},
    },
    "required": ["type"],
}
_BROWSER_ACTION_DESC = (
    "สั่ง action บน browser หนึ่งครั้ง โดยอ้างอิง index จาก indexed elements "
    "ของหน้าปัจจุบันที่ให้ไปเท่านั้น"
)

_FINISH_TASK_PARAMS = {
    "type": "object",
    "properties": {
        "success": {"type": "boolean", "description": "goal สำเร็จไหม"},
        "message": {"type": "string", "description": "สรุปผลสั้นๆ ว่าทำอะไรไป/ทำไมหยุด"},
    },
    "required": ["success", "message"],
}
_FINISH_TASK_DESC = "เรียกเมื่อ goal สำเร็จแล้ว หรือเห็นชัดว่าทำต่อไม่ได้ — จบ loop"

# --- Anthropic tool format ---
BROWSER_ACTION_TOOL = {"name": "browser_action", "description": _BROWSER_ACTION_DESC, "input_schema": _BROWSER_ACTION_PARAMS}
FINISH_TASK_TOOL = {"name": "finish_task", "description": _FINISH_TASK_DESC, "input_schema": _FINISH_TASK_PARAMS}

# --- OpenAI-compatible (Groq) tool format ---
_GROQ_TOOLS = [
    {"type": "function", "function": {"name": "browser_action", "description": _BROWSER_ACTION_DESC, "parameters": _BROWSER_ACTION_PARAMS}},
    {"type": "function", "function": {"name": "finish_task", "description": _FINISH_TASK_DESC, "parameters": _FINISH_TASK_PARAMS}},
]


def build_client(api_key: str) -> AsyncAnthropic:
    return AsyncAnthropic(api_key=api_key)


async def next_action(
    client: AsyncAnthropic,
    model: str,
    goal: str,
    page_text: str,
    messages: list[dict],
) -> tuple[str, dict[str, Any], str, list[dict], TokenUsage]:
    """ส่ง page state ปัจจุบันเข้าไปในบทสนทนา แล้วขอ action ถัดไปจาก Claude

    คืนค่า (tool_name, tool_input, tool_use_id, messages_ใหม่, usage) — tool_use_id ต้อง
    ส่งเข้า append_tool_result() หลังทำ action เสร็จ, messages_ใหม่ต้องส่งกลับเข้า
    next_action() รอบถัดไป เพื่อให้ Claude เห็นบทสนทนา/ผลลัพธ์ action ก่อนหน้าต่อเนื่องกัน
    """
    messages = messages + [
        {"role": "user", "content": f"Goal: {goal}\n\nหน้าเว็บปัจจุบัน:\n{page_text}"}
    ]

    response = await client.messages.create(
        model=model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        tools=[BROWSER_ACTION_TOOL, FINISH_TASK_TOOL],
        tool_choice={"type": "any"},
        messages=messages,
    )
    usage = TokenUsage(response.usage.input_tokens, response.usage.output_tokens)

    messages = messages + [{"role": "assistant", "content": response.content}]

    tool_use = next((b for b in response.content if b.type == "tool_use"), None)
    if tool_use is None:
        # ไม่ควรเกิดขึ้นเพราะ tool_choice บังคับให้เรียก tool เสมอ — กันไว้เผื่อ API เปลี่ยนพฤติกรรม
        return "finish_task", {"success": False, "message": "LLM ไม่เรียก tool ใดๆ กลับมา"}, "", messages, usage

    return tool_use.name, tool_use.input, tool_use.id, messages, usage


def append_tool_result(messages: list[dict], tool_use_id: str, result_text: str) -> list[dict]:
    """ต่อผลลัพธ์ของ action ที่เพิ่งทำเข้าไปในบทสนทนา ก่อนเรียก next_action() รอบถัดไป (Anthropic)"""
    return messages + [
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": tool_use_id, "content": result_text}
            ],
        }
    ]


def build_groq_client(api_key: str) -> AsyncGroq:
    return AsyncGroq(api_key=api_key)


async def next_action_groq(
    client: AsyncGroq,
    model: str,
    goal: str,
    page_text: str,
    messages: list[dict],
) -> tuple[str, dict[str, Any], str, list[dict], TokenUsage]:
    """เหมือน next_action() แต่ยิงผ่าน Groq (OpenAI-compatible chat.completions + function calling)
    ใช้ทดสอบ agent loop ตอนยังไม่มี Anthropic key จริง

    Llama บางครั้งตอบเป็นข้อความเฉยๆ โดยไม่เรียก tool เลย แม้ tool_choice="required" —
    กรณีนี้ไม่ finish_task ทันที แต่เตือนให้เรียก tool แล้วลองใหม่สูงสุด
    _GROQ_NO_TOOL_CALL_RETRIES ครั้ง ก่อนจะ fallback เป็น finish_task(success=False)

    usage ที่คืนกลับ คือผลรวม token ของทุก request ที่ยิงจริง (รวม retry ที่สำเร็จด้วย)
    ไม่นับ request ที่ throw ก่อนได้ response กลับมา (เช่น tool_use_failed)
    """
    if not messages:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    messages = messages + [
        {"role": "user", "content": f"Goal: {goal}\n\nหน้าเว็บปัจจุบัน:\n{page_text}"}
    ]

    total_usage = TokenUsage()

    for attempt in range(_GROQ_NO_TOOL_CALL_RETRIES):
        response = None
        last_error: GroqBadRequestError | None = None
        for _ in range(_GROQ_TOOL_CALL_RETRIES):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    max_tokens=1024,
                    messages=messages,
                    tools=_GROQ_TOOLS,
                    tool_choice="required",
                )
                break
            except GroqBadRequestError as e:
                if getattr(e, "body", None) and e.body.get("error", {}).get("code") == "tool_use_failed":
                    last_error = e
                    continue
                raise
        if response is None:
            raise last_error

        if response.usage is not None:
            total_usage += TokenUsage(response.usage.prompt_tokens, response.usage.completion_tokens)

        message = response.choices[0].message
        messages = messages + [message.model_dump(exclude_none=True)]

        tool_calls = message.tool_calls or []
        if tool_calls:
            tool_call = tool_calls[0]
            tool_input = json.loads(tool_call.function.arguments)
            return tool_call.function.name, tool_input, tool_call.id, messages, total_usage

        if attempt < _GROQ_NO_TOOL_CALL_RETRIES - 1:
            messages = messages + [{"role": "user", "content": _NO_TOOL_CALL_NUDGE}]

    return (
        "finish_task",
        {"success": False, "message": f"LLM ไม่เรียก tool ใดๆ กลับมาแม้เตือนแล้ว {_GROQ_NO_TOOL_CALL_RETRIES} ครั้ง"},
        "",
        messages,
        total_usage,
    )


def append_tool_result_groq(messages: list[dict], tool_use_id: str, result_text: str) -> list[dict]:
    """ต่อผลลัพธ์ของ action ที่เพิ่งทำเข้าไปในบทสนทนา ก่อนเรียก next_action_groq() รอบถัดไป"""
    return messages + [{"role": "tool", "tool_call_id": tool_use_id, "content": result_text}]

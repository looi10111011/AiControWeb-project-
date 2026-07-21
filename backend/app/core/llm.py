"""LLM planning: หน้าเว็บ (indexed elements) + goal -> action ถัดไป

W4: ใช้ tool-use / function calling แทนการให้ LLM ตอบ JSON เป็น text แล้วมาพาร์สเอง
    — response กลับมาเป็น tool call ที่ schema ถูกบังคับโดย API เลย ไม่ต้องกังวลเรื่อง
    markdown fence / คำอธิบายแถม / JSON ผิดรูปแบบ

ผูกกับ actions.execute()'s cmd dict โดยตรง: tool "browser_action" คืน dict ที่ยิงเข้า
execute(page, cmd) ได้ทันที ส่วน tool "finish_task" คือสัญญาณให้ orchestrator หยุด loop

รองรับ 3 provider:
  - Anthropic (Claude) — ตัวหลักตาม roadmap
  - Gemini (Google) — provider สำรอง มี free tier กว้างกว่า Anthropic
  - Groq — ใช้ทดสอบ agent loop ชั่วคราวตอนยังไม่มี Anthropic key จริง (มี free tier)
ทั้งหมดคืนค่ารูปแบบเดียวกัน (tool_name, tool_input, tool_use_id, messages, usage) ให้
orchestrator.py เรียกใช้แบบไม่ต้องรู้ว่าข้างในเป็น provider ไหน — usage คือจำนวน token
ที่ใช้ไปในการเรียก LLM รอบนี้ (รวมทุก retry ถ้ามี) ไว้ให้ orchestrator log/สรุปได้

Anthropic path เปิด prompt caching ไว้ (system + tools มี cache_control) เพราะสอง
ก้อนนี้เหมือนเดิมทุก step ของ loop เดียวกัน ต่างแค่ messages ที่ยาวขึ้นเรื่อยๆ — Groq
ไม่ได้ทำตรงนี้ (ไม่รองรับ cache_control แบบเดียวกันผ่าน chat.completions)
"""

import asyncio
import json
from dataclasses import dataclass
from typing import Any

import google.generativeai as genai
from anthropic import AsyncAnthropic
from google.api_core.exceptions import ResourceExhausted
from groq import AsyncGroq, BadRequestError as GroqBadRequestError


@dataclass
class TokenUsage:
    """จำนวน token ที่ใช้ไปในการเรียก LLM หนึ่งรอบ (รวมทุก retry ถ้ามี)

    cache_creation_tokens/cache_read_tokens มีความหมายเฉพาะฝั่ง Anthropic (prompt
    caching) — Groq ไม่ได้ extract ค่านี้ เลยเป็น 0 เสมอในฝั่งนั้น
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens + self.cache_creation_tokens + self.cache_read_tokens

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_creation_tokens + other.cache_creation_tokens,
            self.cache_read_tokens + other.cache_read_tokens,
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

# Gemini free tier มี quota เป็นนาที (RPM) — ยิงถี่เกินจะได้ 429 ResourceExhausted
# กลับมา ถ้าไม่ดักไว้ agent loop จะ crash ทั้ง process กลางคันแทนที่จะแค่หน่วงแล้วลองใหม่
# (quota มักรีเซ็ตในหลักนาที ไม่ใช่วินาที เลย backoff แบบ exponential เริ่มจากค่าเยอะพอ)
_GEMINI_RATE_LIMIT_RETRIES = 3
_GEMINI_RATE_LIMIT_BACKOFF_SECONDS = 20

SYSTEM_PROMPT = """คุณคือ AI agent ควบคุมหน้าเว็บผ่าน browser ให้ทำ goal ที่ user สั่ง

ทุกครั้งได้รับ "indexed elements" ของหน้าปัจจุบัน เช่น:
  [0] input(text) 'Username'
  [1] input(submit) 'Login'

กติกา:
- เลือก action จาก index ที่เห็นในหน้าปัจจุบันเท่านั้น ทำทีละ 1 action ต่อครั้ง
- action ก่อนหน้า fail แล้ว ให้ดู element ล่าสุดแล้วลองทางอื่น ห้ามยิงซ้ำแบบเดิมเป๊ะๆ
- ห้าม finish_task ก่อนลอง action จริงอย่างน้อย 1 ครั้ง เว้นแต่เห็นชัดจากหน้าปัจจุบันว่า
  goal สำเร็จอยู่แล้ว
- goal ที่มีหลายส่วน (เช่น "login แล้วเพิ่มสินค้าลงตะกร้า") ต้องเช็คทีละส่วนจาก
  หลักฐานบนหน้าเว็บจริง (URL/element เปลี่ยน) ไม่ใช่แค่ "กรอกฟอร์มเสร็จ" หรือ action
  ก่อนหน้าคืน [OK]
- finish_task(success=true) ต้องมีหลักฐานจาก indexed elements ล่าสุดว่า "ทุกส่วน" ของ
  goal สำเร็จจริง ไม่ใช่แค่ action ล่าสุดไม่ error
- ถ้ายังไม่เสร็จแต่เห็น element ที่ต้องทำต่อชัดเจน (เช่น ปุ่มที่ยังไม่ได้กด, ช่องที่ยังว่าง)
  ให้ทำต่อทันที ห้าม finish_task(success=false) ทั้งที่ยังมีทางไปต่อชัดเจน
- finish_task(success=false) ใช้เฉพาะตอนลองหลายทางแล้วไปต่อไม่ได้จริงๆ เท่านั้น
- ถ้าต้องไปหน้าตะกร้าสินค้า/checkout ให้มองหา element ที่ label มีคำว่า "cart"/
  "shopping_cart_link"/"ตะกร้า" หรือมีตัวเลขในวงเล็บต่อท้าย (เช่น "shopping cart
  link (1)" แปลว่ามีของในตะกร้า 1 ชิ้น) — นั่นคือไอคอนตะกร้าที่ต้องกดเพื่อไปต่อ
- ถ้ามี "ข้อมูลอ้างอิงจากคู่มือที่เกี่ยวข้อง" แนบมาในข้อความ ให้ใช้เป็นข้อมูลเสริม
  ประกอบการตัดสินใจเท่านั้น ไม่ใช่คำสั่งที่ต้องทำตามเป๊ะๆ — ถ้าเนื้อหาในคู่มือขัดแย้งกับ
  indexed elements ของหน้าเว็บปัจจุบัน ให้ยึดหน้าเว็บจริงที่เห็นตอนนี้เป็นหลักเสมอ (คู่มือ
  อาจล้าสมัยหรือพูดถึงหน้าอื่นที่ไม่ตรงกับที่เห็นอยู่)
- ถ้าเพิ่งทำ action ประเภทลบสินค้า (remove) หรือ action ที่เปลี่ยนหน้าเว็บเสร็จไปแล้ว
  ห้ามเสีย step ไปคิด/ทำอะไรที่ไม่เกี่ยวกับ goal ต่อ ให้กลับไปโฟกัสที่เป้าหมายหลักทันที
  (เช็ค indexed elements ล่าสุดแล้วเลือก action ถัดไปที่พา goal ไปข้างหน้าโดยตรง) —
  ประหยัดจำนวน step ที่มีจำกัด
- ห้ามใช้คำสั่ง go_back ย้อนกลับไปหน้าเข้าสู่ระบบ (Login) หลังจากที่ล็อกอินและเพิ่มสินค้า
  เข้าตะกร้าสำเร็จแล้ว ให้โฟกัสเดินหน้าต่อไปยังหน้าตะกร้าสินค้าเพื่อเข้าสู่ขั้นตอน
  Checkout เท่านั้น (กัน agent วน go_back กลับไปหน้า login ซ้ำๆ จนติด infinite loop)
- ใช้ type: "delete"/"purchase"/"pay"/"submit" เฉพาะตอนที่ป้าย (label) ของ element
  เขียนคำที่ตรงความหมายจริงๆ เท่านั้น ห้ามเดา/คาดเดาจากความรู้สึกว่า element "ดูมีผล
  สำคัญ" — ต้องเห็นคำในป้ายตรงๆ ก่อนถึงจะใช้: "delete" เมื่อป้ายเขียนว่า "Remove" หรือ
  "Delete" ตรงตัว, "purchase" เมื่อป้ายเขียนว่า "Place Order" หรือ "Finish" (ปุ่มยืนยัน
  คำสั่งซื้อขั้นสุดท้ายในหน้า checkout), "pay" เมื่อป้ายเขียนว่า "Pay" หรือ "Pay Now",
  "submit" เมื่อป้ายเขียนคำว่า "Submit" ตรงตัว — ถ้าป้ายไม่ได้เขียนคำเหล่านี้ตรงๆ (เช่น
  "Open Menu", "Continue Shopping", "Add to cart", ไอคอนไม่มีข้อความ) ให้ใช้ "click"
  เสมอ ไม่ว่า element นั้นจะดูสำคัญแค่ไหนก็ตาม ห้ามใช้ 4 type นี้ "เผื่อไว้ก่อน"
  เด็ดขาด เพราะระบบจะหยุดขอยืนยันจาก human ทุกครั้งที่เจอ ใช้พร่ำเพรื่อจะทำให้ user
  ต้องกดอนุมัติบ่อยเกินจำเป็น
- หากกรอกฟอร์มเข้าสู่ระบบ (Login Form) ให้กรอกข้อมูลให้ครบทั้ง Username และ Password
  ทันที ห้ามสั่ง wait คั่นกลางหากหน้าเว็บไม่มีการเปลี่ยนแปลง
- ถ้า goal ต้องการหาข้อมูลเฉพาะเจาะจง (เช่น ราคา/ชื่อ/รายละเอียดสินค้า) ที่ยังไม่เห็นชัด
  ในหน้าปัจจุบัน ห้าม scroll ไปเรื่อยๆ แบบไม่มีทิศทางเพื่อ "หาไปเรื่อยๆ" — ให้คลิกเข้าไป
  ที่ element ที่เจาะจงกว่า (เช่น ชื่อ/รูปสินค้าที่พาไปหน้า product detail) ก่อน เพราะ
  ข้อมูลที่ต้องการมักอยู่ครบและชัดเจนกว่าในหน้าเจาะจงนั้น เทียบกับการกวาดหาในหน้ารวม/
  หน้า catalog
- ห้ามใช้ goto ไปยัง URL ของหน้าที่กำลังอยู่อยู่แล้วเด็ดขาด (เช็คก่อนเสมอว่า element ที่
  ต้องการทำ action ด้วยปรากฏอยู่ใน indexed elements ปัจจุบันอยู่แล้วหรือยัง ถ้าอยู่แล้ว
  แปลว่าไม่ต้อง goto) — goto จะโหลดหน้าใหม่ทั้งหมดจากศูนย์ ล้างข้อมูลที่เพิ่งกรอกในฟอร์ม
  ทิ้งทั้งหมด (เช่น ชื่อ/นามสกุล/รหัสไปรษณีย์ที่กรอกไปแล้วจะหายไปต้องกรอกใหม่) ถ้าไม่แน่ใจ
  ว่าอยู่หน้าไหน ให้ดู element ใน indexed elements ล่าสุดตัดสินใจแทนการ goto ซ้ำเพื่อ
  "เช็คให้ชัวร์"
- action ใดๆ ที่ผลลัพธ์ล่าสุดออกมาเป็น [OK] แล้ว ถือว่าสำเร็จสมบูรณ์แล้วจริง แม้จะเป็น
  action ที่มีข้อมูลอ้างอิงจากคู่มือบอกว่าต้องขออนุมัติจาก human ก่อน (เช่น "ต้องขอ
  อนุมัติ") ก็ตาม — ผลลัพธ์ [OK] แปลว่า human อนุมัติให้ทำไปแล้วจริงในตอนนั้น ห้ามสงสัย/
  go_back/พยายามทำซ้ำ/หยุดงาน (finish_task) เพราะคิดว่ายังไม่ได้รับอนุมัติ ให้เดินหน้า
  ทำ action ถัดไปตาม goal ต่อไปตามปกติ
- หาก Action ใดได้รับการปฏิเสธจากมนุษย์ (human-in-the-loop ตอบไม่อนุญาตต่อ action ที่
  ต้องขอยืนยันก่อน — จะเห็นข้อความ "ผู้ใช้ปฏิเสธการทำ Action นี้" แนบมาใน "Action ที่เคย
  ลองแล้วล้มเหลว") ห้ามพยายามทำ Action นั้นซ้ำอีกเด็ดขาดในรอบการทำงานปัจจุบัน (task นี้)
  ให้พิจารณาทางเลือกอื่นที่ยังไม่ได้ลอง (เช่น element อื่นที่พาไปสู่เป้าหมายเดียวกันได้)
  หรือถ้าไม่มีทางเลือกอื่นจริงๆ ให้ยุติงานด้วย finish_task(success=false) พร้อมอธิบาย
  เหตุผลที่ทำต่อไม่ได้ให้ user เข้าใจชัดเจน — ต่างจาก action ที่ล้มเหลวเพราะเหตุผลทาง
  เทคนิค (เช่น timeout/index ผิด) ที่ยังลองทางอื่นได้ตามปกติ การถูกปฏิเสธคือคำตัดสินใจ
  ของมนุษย์ ไม่ใช่ปัญหาทางเทคนิคที่แก้ด้วยการลองซ้ำ
"""

# W6[B]: ต่อ user turn เดียวกันนี้ใช้ร่วมกันทั้ง 3 provider (Anthropic/Groq ใช้ตรงๆ เป็น
# plain string content, Gemini เอาไปห่อเป็น parts[0]["text"] — สุดท้ายเป็น plain text
# เหมือนกันหมด) — ต่อ section คู่มือ (จาก retriever.retrieve() ที่ orchestrator เรียกให้
# ทุก step) เฉพาะตอนมีผลลัพธ์จริง กัน prompt รกด้วย section เปล่าๆ ทุก step ที่หาไม่เจอ
# ในคู่มือ (retrieve() คืน [] เงียบๆ เสมอ ไม่ throw)
#
# W7[A]: เพิ่ม memory_context เดียวกัน — สรุป action ที่ล้มเหลวไปแล้วใน task นี้ (จาก
# ShortTermMemory.failed_actions_summary() ที่ orchestrator เรียกให้ทุก step) ต่อกัน
# ท้ายสุด เฉพาะตอนมีผลลัพธ์จริงเหมือนกัน (ว่างเปล่าถ้ายังไม่เคย fail อะไรเลย)
#
# W7[A] (long-term): เพิ่ม long_term_context — เหมือน manual_context ทุกประการแค่มา
# จาก long_term_memory.recall() (จาก task run อื่นที่เคยทำมาก่อน) แทนคู่มือที่ user
# ป้อน — คนละ section กับ memory_context (ตัวนั้นจำได้แค่ภายใน task ปัจจุบันเดียวเท่านั้น
# ตัวนี้จำข้ามหลาย task run)
#
# W9[A] (vision fallback): เพิ่ม vision_context — คำอธิบายจาก Gemini vision (ดู
# describe_screenshot() ด้านล่าง) ตอน action ที่ต้องพึ่ง element visibility (click/
# fill/select/check) ล้มเหลวซ้ำแม้ retry ครบแล้ว ทั้งที่ index มีอยู่จริงใน DOM — สงสัย
# ว่ามี popup/overlay บัง element ที่ perception (DOM-based ล้วนๆ) มองไม่เห็นครบ (ดู
# marker "[ถูกบังอยู่]" ใน perception.py ที่เป็นสัญญาณเสริมอีกชั้นแบบไม่ต้องพึ่ง vision)
# — ว่างเปล่าถ้าไม่มี action ล้มเหลวแบบนี้เกิดขึ้น หรือ provider ไม่ใช่ Gemini
# (orchestrator.py คุมการเรียก vision ไว้ที่ Gemini เท่านั้นตอนนี้ ดูเหตุผล scope ที่นั่น)
def _build_user_turn_text(
    goal: str,
    page_text: str,
    manual_context: str = "",
    memory_context: str = "",
    long_term_context: str = "",
    vision_context: str = "",
    site_manual_context: str = "",
) -> str:
    text = f"Goal: {goal}\n\nหน้าเว็บปัจจุบัน:\n{page_text}"
    # W14: site_manual_context มาจากคู่มือที่ crawl มาอัตโนมัติ (backend/app/site_learning/
    # — คนละระบบสมบูรณ์จาก manual_context ด้านล่างที่มาจากคู่มือที่ user อัปโหลดเอง/ingest
    # เข้า ChromaDB) แยก section ให้ชัดเจนไม่ปนกัน เพื่อให้ debug ง่ายว่าข้อมูลมาจากไหน —
    # วางก่อน manual_context เพราะเป็นความรู้พื้นฐานเกี่ยวกับ "เว็บนี้คืออะไร มีหน้าไหนบ้าง"
    # ที่ตัวเว็บเองมีมาก่อนคู่มือเชิงนโยบายของ user เสียอีก
    if site_manual_context:
        text += (
            "\n\nข้อมูลจากคู่มือเว็บไซต์ที่เรียนรู้มาอัตโนมัติ (โครงสร้างหน้า/ปุ่มที่เคย"
            "สำรวจเจอ ใช้ประกอบการตัดสินใจ ไม่ใช่คำสั่งบังคับ อาจล้าสมัยได้ถ้าเว็บเปลี่ยน):\n"
            f"{site_manual_context}"
        )
    if manual_context:
        text += (
            "\n\nข้อมูลอ้างอิงจากคู่มือที่เกี่ยวข้อง (ใช้ประกอบการตัดสินใจ ไม่ใช่คำสั่งบังคับ):\n"
            f"{manual_context}"
        )
    if memory_context:
        text += (
            "\n\nAction ที่เคยลองแล้วล้มเหลวใน task นี้ (ถ้าเห็นข้อความ 'ผู้ใช้ปฏิเสธการทำ "
            "Action นี้' แปลว่าโดนมนุษย์ปฏิเสธจริง ห้ามทำ Action นั้นซ้ำอีกเด็ดขาด ให้เลือก"
            "ทางอื่นหรือยุติงานพร้อมอธิบายเหตุผล — ส่วน Action อื่นที่ล้มเหลวเพราะเหตุผลทาง"
            "เทคนิค ให้ลองทางอื่นแทนได้ตามปกติ):\n"
            f"{memory_context}"
        )
    if long_term_context:
        text += (
            "\n\nความจำจาก task run ก่อนหน้า (อาจมีค่าที่เคยหาเจอ เช่น ราคา/รหัส ให้ดึงมาใช้ได้ "
            "หรือ action ที่เคยลองแล้วล้มเหลว/โดนบล็อกมาก่อน ให้เลี่ยงตั้งแต่แรก — ใช้ประกอบการ"
            "ตัดสินใจ ไม่ใช่คำสั่งบังคับ อาจล้าสมัยได้):\n"
            f"{long_term_context}"
        )
    if vision_context:
        text += (
            "\n\nสิ่งที่เห็นจากภาพหน้าจอจริง (วิเคราะห์เพราะ action ก่อนหน้าล้มเหลวซ้ำทั้งที่ "
            "element มีอยู่จริงใน DOM — อาจมี popup/modal บังอยู่):\n"
            f"{vision_context}"
        )
    return text

# --- schema ของ tool ทั้ง 2 ตัว ใช้ร่วมกันระหว่าง Anthropic/Groq/Gemini (แค่ห่อ format ต่างกัน) ---

_BROWSER_ACTION_PARAMS = {
    "type": "object",
    "properties": {
        "type": {
            "type": "string",
            "enum": [
                "click", "fill", "select", "check",
                "scroll", "goto", "go_back", "switch_tab", "wait",
                # W?: permission layer (classify_action) รู้จัก type เหล่านี้เป็น
                # NEEDS_CONFIRMATION มาตั้งแต่ W4/W5 แต่ก่อนหน้านี้ไม่เคยอยู่ใน enum
                # ที่ LLM เรียกได้จริงเลย — human-in-the-loop เลย unreachable ผ่าน
                # agent loop จริง (trigger ได้แค่ตอนยิง execute() ตรงๆ ใน demo/test)
                # เพิ่มเข้ามาให้เป็น alias ของ click ที่มีความหมายชัดเจนกว่า (index
                # เหมือนเดิม) — actions.py::execute() dispatch ให้แล้ว (เห็นได้จาก
                # DEFAULT_NEEDS_CONFIRMATION check)
                "submit", "delete", "purchase", "pay",
            ],
            "description": "ชนิด action",
        },
        "index": {
            "type": "integer",
            "description": "index ของ element (click/fill/select/check/submit/delete/purchase/pay)",
        },
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
# cache_control อยู่บน tool ตัวสุดท้าย -> Anthropic cache ทั้ง prefix (tools + system
# ที่ตามมา) เป็นก้อนเดียว เพราะ tools/system เหมือนเดิมทุก step ของ loop เดียวกัน
FINISH_TASK_TOOL = {
    "name": "finish_task",
    "description": _FINISH_TASK_DESC,
    "input_schema": _FINISH_TASK_PARAMS,
    "cache_control": {"type": "ephemeral"},
}

# --- OpenAI-compatible (Groq) tool format ---
_GROQ_TOOLS = [
    {"type": "function", "function": {"name": "browser_action", "description": _BROWSER_ACTION_DESC, "parameters": _BROWSER_ACTION_PARAMS}},
    {"type": "function", "function": {"name": "finish_task", "description": _FINISH_TASK_DESC, "parameters": _FINISH_TASK_PARAMS}},
]

# --- Gemini (google-generativeai) tool format ---
_GEMINI_TOOLS = [
    {
        "function_declarations": [
            {"name": "browser_action", "description": _BROWSER_ACTION_DESC, "parameters": _BROWSER_ACTION_PARAMS},
            {"name": "finish_task", "description": _FINISH_TASK_DESC, "parameters": _FINISH_TASK_PARAMS},
        ]
    }
]


# system ส่งเป็น content block (ไม่ใช่ string เฉยๆ) พร้อม cache_control -> Anthropic
# cache ทั้ง tools+system prefix ไว้ (เหมือนกันทุก step ของ loop เดียวกัน ต่างแค่
# messages ที่ยาวขึ้นเรื่อยๆ) ลด input token cost ของทุก step หลังจากตัวแรก
# หมายเหตุ: ต้อง prompt ยาวพอถึง minimum cacheable length ของโมเดลนั้นๆ ไม่งั้น API
# จะเมิน cache_control เงียบๆ (ไม่ error) — เช็คได้จาก usage.cache_read/creation_tokens
_SYSTEM_BLOCKS = [{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}]


def build_client(api_key: str) -> AsyncAnthropic:
    return AsyncAnthropic(api_key=api_key)


async def next_action(
    client: AsyncAnthropic,
    model: str,
    goal: str,
    page_text: str,
    messages: list[dict],
    manual_context: str = "",
    memory_context: str = "",
    long_term_context: str = "",
    vision_context: str = "",
    site_manual_context: str = "",
) -> tuple[str, dict[str, Any], str, list[dict], TokenUsage]:
    """ส่ง page state ปัจจุบันเข้าไปในบทสนทนา แล้วขอ action ถัดไปจาก Claude

    คืนค่า (tool_name, tool_input, tool_use_id, messages_ใหม่, usage) — tool_use_id ต้อง
    ส่งเข้า append_tool_result() หลังทำ action เสร็จ, messages_ใหม่ต้องส่งกลับเข้า
    next_action() รอบถัดไป เพื่อให้ Claude เห็นบทสนทนา/ผลลัพธ์ action ก่อนหน้าต่อเนื่องกัน

    manual_context (W6[B]): chunk คู่มือที่เกี่ยวข้อง (จาก retriever.retrieve()) ที่
    orchestrator ดึงมาให้ทุก step — ว่างเปล่าได้ตามปกติถ้าไม่มีคู่มือ ingest ไว้/ไม่เจอ
    อะไรตรงกับหน้านี้

    memory_context (W7[A]): สรุป action ที่ล้มเหลวไปแล้วใน task นี้ (จาก
    ShortTermMemory.failed_actions_summary() ที่ orchestrator ดึงมาให้ทุก step) —
    ว่างเปล่าได้ตามปกติถ้ายังไม่เคย fail อะไรเลย

    long_term_context (W7[A] long-term): เหมือน manual_context แต่มาจาก
    long_term_memory.recall() (ประวัติ task run อื่นก่อนหน้า) แทนคู่มือ

    vision_context (W9[A]): คำอธิบายจาก Gemini vision ตอน action ก่อนหน้าล้มเหลวซ้ำ —
    ดู _build_user_turn_text() ด้านบน (ปัจจุบัน orchestrator.py ยิง vision fallback
    เฉพาะ provider=gemini เท่านั้น เลย path นี้ (Anthropic) จะได้ "" เสมอในทางปฏิบัติ
    แต่รับ parameter ไว้เผื่อขยาย provider อื่นทีหลัง)

    site_manual_context (W14): เนื้อหาย่อจากคู่มือเว็บไซต์ที่ crawl มาอัตโนมัติ (ดู
    backend/app/site_learning/) — orchestrator ดึงมาครั้งเดียวตอนเริ่ม task (ไม่ใช่ทุก
    step แบบ manual_context เพราะไม่ได้ผูกกับ page state ปัจจุบัน) ว่างเปล่าถ้าโดเมนนี้
    ยังไม่เคยถูกเรียนรู้/สร้าง manual ไว้
    """
    messages = messages + [
        {
            "role": "user",
            "content": _build_user_turn_text(
                goal, page_text, manual_context, memory_context, long_term_context, vision_context,
                site_manual_context,
            ),
        }
    ]

    response = await client.messages.create(
        model=model,
        max_tokens=1024,
        system=_SYSTEM_BLOCKS,
        tools=[BROWSER_ACTION_TOOL, FINISH_TASK_TOOL],
        tool_choice={"type": "any"},
        messages=messages,
    )
    usage = TokenUsage(
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        cache_creation_tokens=getattr(response.usage, "cache_creation_input_tokens", 0) or 0,
        cache_read_tokens=getattr(response.usage, "cache_read_input_tokens", 0) or 0,
    )

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
    manual_context: str = "",
    memory_context: str = "",
    long_term_context: str = "",
    vision_context: str = "",
    site_manual_context: str = "",
) -> tuple[str, dict[str, Any], str, list[dict], TokenUsage]:
    """เหมือน next_action() แต่ยิงผ่าน Groq (OpenAI-compatible chat.completions + function calling)
    ใช้ทดสอบ agent loop ตอนยังไม่มี Anthropic key จริง

    Llama บางครั้งตอบเป็นข้อความเฉยๆ โดยไม่เรียก tool เลย แม้ tool_choice="required" —
    กรณีนี้ไม่ finish_task ทันที แต่เตือนให้เรียก tool แล้วลองใหม่สูงสุด
    _GROQ_NO_TOOL_CALL_RETRIES ครั้ง ก่อนจะ fallback เป็น finish_task(success=False)

    usage ที่คืนกลับ คือผลรวม token ของทุก request ที่ยิงจริง (รวม retry ที่สำเร็จด้วย)
    ไม่นับ request ที่ throw ก่อนได้ response กลับมา (เช่น tool_use_failed)

    manual_context/memory_context/long_term_context/vision_context: ดู next_action() —
    เหมือนกัน (vision_context จะเป็น "" เสมอในทางปฏิบัติ เพราะ vision fallback ปัจจุบัน
    scope แค่ provider=gemini)
    """
    if not messages:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    messages = messages + [
        {
            "role": "user",
            "content": _build_user_turn_text(
                goal, page_text, manual_context, memory_context, long_term_context, vision_context,
                site_manual_context,
            ),
        }
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


def build_gemini_client(api_key: str):
    """google-generativeai ใช้ global config (genai.configure) ไม่มี client object
    แยกต่างหากเหมือน Anthropic/Groq — configure() ครั้งเดียวแล้วคืน genai module กลับไป
    ให้ next_action_gemini() ใช้สร้าง GenerativeModel ต่อ (tools/system_instruction
    เหมือนเดิมทุกครั้ง แค่ constructor local object เฉยๆ ไม่มี network call)"""
    genai.configure(api_key=api_key)
    return genai


def _normalize_gemini_args(args: dict) -> dict[str, Any]:
    """Gemini คืนตัวเลขทุกตัวเป็น float ผ่าน protobuf Struct เสมอ แม้ schema จะระบุ
    "integer" ไว้ก็ตาม (เช่น index: 0.0 แทน 0) — ถ้าไม่แปลงกลับ selector ที่ยิงเข้า
    Playwright จะพัง ('[data-ai-index="0.0"]' ไม่ตรงกับ element จริงที่ index="0")"""
    return {
        key: int(value) if isinstance(value, float) and value.is_integer() else value
        for key, value in args.items()
    }


async def next_action_gemini(
    client,
    model: str,
    goal: str,
    page_text: str,
    messages: list,
    manual_context: str = "",
    memory_context: str = "",
    long_term_context: str = "",
    vision_context: str = "",
    site_manual_context: str = "",
) -> tuple[str, dict[str, Any], str, list, TokenUsage]:
    """เหมือน next_action() แต่ยิงผ่าน Gemini (google-generativeai function calling)

    messages เก็บ Content ของ Gemini เอง (dict {"role": ..., "parts": [...]} หรือ
    Content proto ที่ SDK คืนมาตรงๆ ก็ใส่ต่อ list ได้เลย) — คนละ shape กับ
    Anthropic/Groq แต่ orchestrator.py ไม่แคร์ เพราะแค่ถือ opaque state ส่งเข้า-ออก

    tool_use_id ที่คืนกลับ คือชื่อ function ("browser_action"/"finish_task") ไม่ใช่ id
    จริงแบบ Anthropic/Groq เพราะ Gemini SDK เวอร์ชันนี้ไม่มี call id ให้ — ใช้เป็น "name"
    ที่ append_tool_result_gemini() ต้องผูก function_response กลับด้วย

    manual_context/memory_context/long_term_context/vision_context: ดู next_action() —
    เหมือนกัน (vision_context (W9[A]) จะมีค่าจริงเฉพาะ provider นี้ — orchestrator.py
    ยิง vision fallback (llm.describe_screenshot()) scope แค่ Gemini เท่านั้นตอนนี้)
    """
    gemini_model = client.GenerativeModel(
        model_name=model,
        tools=_GEMINI_TOOLS,
        tool_config={"function_calling_config": {"mode": "ANY"}},
        system_instruction=SYSTEM_PROMPT,
    )

    messages = messages + [
        {
            "role": "user",
            "parts": [{
                "text": _build_user_turn_text(
                    goal, page_text, manual_context, memory_context, long_term_context, vision_context,
                    site_manual_context,
                )
            }],
        }
    ]

    response = None
    for attempt in range(_GEMINI_RATE_LIMIT_RETRIES):
        try:
            response = await gemini_model.generate_content_async(contents=messages)
            break
        except ResourceExhausted:
            if attempt == _GEMINI_RATE_LIMIT_RETRIES - 1:
                raise
            # exponential backoff: 20s, 40s, ... กัน retry ถี่เกินไปจนโดน 429 ซ้ำอีก
            await asyncio.sleep(_GEMINI_RATE_LIMIT_BACKOFF_SECONDS * (attempt + 1))

    usage = TokenUsage(
        response.usage_metadata.prompt_token_count,
        response.usage_metadata.candidates_token_count,
    )

    content = response.candidates[0].content
    messages = messages + [content]

    part = next((p for p in content.parts if p.function_call and p.function_call.name), None)
    if part is None:
        # ไม่ควรเกิดขึ้นเพราะ tool_config mode="ANY" บังคับให้เรียก function เสมอ — กันไว้
        # เผื่อ API เปลี่ยนพฤติกรรม (เหมือน next_action() ฝั่ง Anthropic)
        return "finish_task", {"success": False, "message": "LLM ไม่เรียก tool ใดๆ กลับมา"}, "", messages, usage

    fc = part.function_call
    tool_input = _normalize_gemini_args(dict(fc.args))
    return fc.name, tool_input, fc.name, messages, usage


def append_tool_result_gemini(messages: list, tool_use_id: str, result_text: str) -> list:
    """ต่อผลลัพธ์ของ action ที่เพิ่งทำเข้าไปในบทสนทนา ก่อนเรียก next_action_gemini() รอบ
    ถัดไป — tool_use_id ตรงนี้คือชื่อ function (ดู next_action_gemini())"""
    return messages + [
        {
            "role": "user",
            "parts": [{"function_response": {"name": tool_use_id, "response": {"result": result_text}}}],
        }
    ]
_PLAN_PROMPT_TEMPLATE = (
    "Goal: {goal}\n\nหน้าเว็บเริ่มต้นที่เห็นตอนนี้:\n{page_text}\n\n"
    "เขียนแผนคร่าวๆ เป็น bullet สั้นๆ (ไม่เกิน 5-6 ข้อ) ว่าจะทำ goal นี้ให้สำเร็จด้วย"
    "ขั้นตอนอะไรบ้าง — สรุประดับสูงพอให้ user อ่านแล้วเข้าใจและตัดสินใจอนุมัติได้ ไม่ต้อง"
    "เรียก tool ไม่ต้องระบุ index ของ element เป๊ะๆ ตอบเป็นข้อความธรรมดา ไม่ต้องมี markdown"
)


async def generate_text(client, model: str, prompt: str, provider: str) -> str:
    """เรียก LLM แบบ plain text call เดียว (ไม่ใช้ tool-use) — primitive ที่ใช้ร่วมกันทั้ง
    generate_plan() ด้านล่าง (ห่อ prompt ด้วย _PLAN_PROMPT_TEMPLATE) และ
    site_learning/crawler.py (ห่อ prompt ของตัวเองเพื่อขอ LLM เขียนชื่อ/คำอธิบายหน้า
    สั้นๆ ตอน crawl — ไม่เกี่ยวกับ plan/goal เลย) แยกออกมาเป็น primitive กัน logic
    per-provider ซ้ำ 2 ที่"""
    if provider == "anthropic":
        response = await client.messages.create(
            model=model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in response.content if b.type == "text").strip()

    if provider == "groq":
        response = await client.chat.completions.create(
            model=model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        return (response.choices[0].message.content or "").strip()

    if provider == "gemini":
        gemini_model = client.GenerativeModel(model_name=model)
        response = await gemini_model.generate_content_async(
            contents=[{"role": "user", "parts": [{"text": prompt}]}],
        )
        return response.text.strip()

    raise ValueError(f"ไม่รู้จัก LLM provider: {provider!r} (รองรับแค่ anthropic/gemini/groq)")


async def generate_plan(client, model: str, goal: str, page_text: str, provider: str) -> str:
    """ให้ LLM ร่างแผนระดับสูง (plain text, ไม่เรียก tool) ก่อนเริ่ม agent loop จริง —
    ใช้กับ Orchestrator.run_task(..., confirm_plan=True) เพื่อโชว์ user ก่อนแล้วรอกดยืนยัน
    ค่อยเริ่ม perceive->plan->act loop จริง (ป้องกันไม่ให้ agent ลงมือทำอะไรที่ user ไม่ได้
    เห็นแผนมาก่อน)
    """
    prompt = _PLAN_PROMPT_TEMPLATE.format(goal=goal, page_text=page_text)
    return await generate_text(client, model, prompt, provider)


# --- W9[A] vision fallback (Gemini เท่านั้นตอนนี้) ---
# scope แค่ Gemini ตามที่ project ทำมาตลอด (ดู context compaction ของ W7[A] ที่ scope
# เดียวกัน) — Anthropic/Groq รองรับ vision ได้เหมือนกันในทางเทคนิค แต่ยังไม่ได้ทดสอบ
# จริง เพิ่มทีหลังได้ถ้าต้องการ ไม่ใช่ข้อจำกัดทางสถาปัตยกรรม
_VISION_FALLBACK_PROMPT_TEMPLATE = (
    "Action ประเภท {action_type} (index {index}) ล้มเหลวซ้ำแม้ retry ครบแล้ว ทั้งที่ "
    "element นี้มีอยู่จริงใน DOM ตอน perceive — อาจมี popup/modal/cookie banner บัง "
    "element นี้อยู่จริงที่ perception (อ่านจาก DOM อย่างเดียว) ตรวจไม่พบครบ นี่คือ"
    "ภาพหน้าจอปัจจุบันจริง ช่วยดูว่าเห็นอะไรผิดปกติไหม (เช่น popup บัง, หน้ายังโหลดไม่เสร็จ, "
    "error message ที่ไม่ได้อยู่ใน indexed elements) แล้วแนะนำสั้นๆ ว่าควรทำอะไรต่อ "
    "(ไม่เกิน 3 ประโยค ตอบเป็นข้อความธรรมดา ไม่ต้องมี markdown)"
)


async def describe_screenshot(client, model: str, screenshot_png: bytes, action_type: str, index: Any) -> str:
    """เรียกตอน action ที่ต้องพึ่ง element visibility (click/fill/select/check และ
    alias submit/delete/purchase/pay) ล้มเหลวซ้ำแม้ retry ครบแล้ว (actions.py::
    _dispatch_with_retry หมดโควตา) ทั้งที่ index มีอยู่จริงใน DOM ตอน perceive — สงสัยว่า
    มี popup/overlay บัง element ที่ perception (DOM-based ล้วนๆ ไม่เช็ค z-index/overlap
    เต็มรูปแบบ แม้จะมี marker "[ถูกบังอยู่]" เสริมแล้วก็ตาม) ตรวจไม่เจอครบ — ส่ง
    screenshot จริงให้ Gemini vision อธิบายสิ่งที่เห็น + คำแนะนำ ไม่ใช้ tool-use (เหมือน
    generate_plan()) แค่ตอบข้อความธรรมดา ให้ orchestrator.py เอาไปป้อนกลับเข้า prompt
    step ถัดไปเป็น context เสริม (vision_context ใน _build_user_turn_text())

    ห้าม throw ออกไปเด็ดขาด (เหมือน retriever.retrieve()/long_term_memory.recall()) —
    ถ้า vision call พังเอง (เช่น quota/network) ต้องไม่ทำให้ agent loop หลักพังตาม คืน ""
    เงียบๆ แทน
    """
    try:
        prompt = _VISION_FALLBACK_PROMPT_TEMPLATE.format(action_type=action_type, index=index)
        gemini_model = client.GenerativeModel(model_name=model)
        response = await gemini_model.generate_content_async(
            contents=[{
                "role": "user",
                "parts": [{"text": prompt}, {"mime_type": "image/png", "data": screenshot_png}],
            }],
        )
        return (response.text or "").strip()
    except Exception as e:
        print(f"⚠️ Vision fallback error: {e}", flush=True)
        return ""

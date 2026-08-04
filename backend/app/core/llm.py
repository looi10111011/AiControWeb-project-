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

W43: user ขอ real-time checkbox ในหน้า plan (Test Console UI) — ติ๊กทีละ step ตอน agent
ทำ step นั้นสำเร็จจริงระหว่างรัน task (ไม่ใช่แค่ตอนจบ task ทั้งหมด) เพิ่ม 2 อย่างที่นี่:
  1. _BROWSER_ACTION_PARAMS ได้ property "completed_plan_step" ใหม่ (optional เสมอ ไม่
     required เด็ดขาด — ad-hoc task ที่ไม่มีแผนต้องยังทำงานเหมือนเดิมทุกประการ) ให้ LLM
     ใส่เลขข้อ (1-based) ถ้า action ที่เพิ่งเรียกทำให้ step นั้นของแผนเสร็จสมบูรณ์แล้ว —
     orchestrator.py อ่านค่านี้แล้วยิง SSE event "plan_step_done" เฉพาะตอน execute()
     สำเร็จจริงเท่านั้น (ดู orchestrator.py สำหรับ logic เต็ม)
  2. _build_user_turn_text()/next_action()/next_action_groq()/next_action_gemini() ทั้ง 3
     provider ได้ parameter ใหม่ plan_context — แนบแผนที่ user ยืนยันแล้วเป็น section แยก
     "แพลนปัจจุบัน" ให้ LLM เห็นเลขข้อจริงก่อนตัดสินใจว่า action นี้ทำให้ step ไหนเสร็จ (ว่าง
     เปล่าเสมอสำหรับ ad-hoc task — ไม่มี section นี้โผล่มาปนเลย)
  3. _PLAN_PROMPT_TEMPLATE เปลี่ยนจากขอ "bullet สั้นๆ" (ไม่บังคับ format จริงจัง) เป็น
     บังคับ format เลขข้อ "1. ... \n2. ..." ตรงๆ เพราะ frontend (index.html) ต้อง parse
     แต่ละบรรทัดเป็น step แยกเพื่อ render checklist ที่ index ตรงกับที่ LLM อ้างอิงใน
     completed_plan_step ได้แน่นอน (เดิมโมเดลบังเอิญมักตอบแบบเลขข้ออยู่แล้วในทางปฏิบัติ
     แต่ไม่ใช่สัญญาที่บังคับได้ — ต้องบังคับชัดเจนไม่งั้น parsing ฝั่ง frontend จะพลาด)
"""

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional
from zoneinfo import ZoneInfo

import google.generativeai as genai
from anthropic import AsyncAnthropic
from google.api_core.exceptions import ResourceExhausted
from groq import AsyncGroq, BadRequestError as GroqBadRequestError

from backend.app.config import settings


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
- การคลิกเลือกรายการจากผลการค้นหา/ลิสต์ (เช่น คลิกวิดีโอ YouTube, การ์ดบทความ, ผลลัพธ์
  การค้นหาสินค้า) เพื่อเปิดดู/เล่น ให้ใช้ "click" เสมอ แม้ว่า plan step จะใช้คำว่า
  "เลือก"/"select" ก็ตาม — คำว่า "เลือก" ในที่นี้แปลว่า "คลิกเพื่อเปิดดู/นำทางไป" ไม่ใช่
  การยืนยันคำสั่งซื้อ/ลบ/จ่ายเงิน อย่าตีความคำว่า "เลือก" ว่าต้องเป็น "submit"/"purchase"
  เด็ดขาด (ป้ายของ element พวกนี้มักเป็นชื่อวิดีโอ/หัวข้อบทความ ไม่ใช่คำสั่งเสี่ยงใดๆ)
- การกดปุ่ม Enter บนคีย์บอร์ดเพื่อยืนยันคำค้นหาที่พิมพ์ไว้ในช่องค้นหา (เช่น กด Enter
  หลังพิมพ์คำค้นหาใน YouTube/Google) ให้ใช้ type: "press_key" เสมอ ห้ามใช้ "submit"
  เด็ดขาดแม้จะรู้สึกว่า "กด Enter = submit ฟอร์ม" ก็ตาม — การค้นหาไม่ใช่การ submit ที่มี
  ผลจริงแบบ checkout/ลบ/จ่ายเงิน ย้อนกลับได้ง่ายมาก
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
- ก่อนเลือก action ทุกครั้ง (โดยเฉพาะ click/fill/select/check) ให้ยึด "URL ปัจจุบันจริง
  ของหน้าเว็บ" และ indexed elements ที่แนบมาในข้อความนี้เท่านั้นเป็นความจริงล่าสุด ห้าม
  อ้างอิง index/สมมติสถานะจากหน้าเว็บของ step ก่อนหน้าเด็ดขาด แม้จะดูคล้ายกับที่วางแผนไว้
  ก็ตาม (นี่คือ "Action Trap" — ทำ action ต่อจากแผนเดิมทั้งที่หน้าเว็บเปลี่ยนไปแล้วจริง) —
  ถ้าเห็นข้อความ "[หน้าเว็บเปลี่ยนไปเองหลัง action นี้: จาก ... เป็น ...]" ต่อท้ายผลลัพธ์
  action ก่อนหน้า ต้องตรวจสอบ URL ปัจจุบันและ indexed elements ของหน้าใหม่นี้ใหม่ทั้งหมด
  ก่อนตัดสินใจ action ถัดไปเสมอ ห้ามเดินหน้าตามแผนเดิมที่ร่างไว้ก่อนหน้าเปลี่ยนต่อ
- ถ้าเห็นข้อความ "[ระบบตรวจพบการวนซ้ำ: ... ระบบจึงบังคับทำ ... แทน action ที่คุณเพิ่งขอ
  โดยอัตโนมัติ ...]" ต่อท้ายผลลัพธ์ action ก่อนหน้า แปลว่าระบบเพิ่งบังคับทำ action อื่น
  แทน action ที่คุณเพิ่งขอไปจริงๆ (ไม่ใช่ action เดิมของคุณที่สำเร็จ) — ห้ามเลือก action
  ประเภทเดิม/element เดิมที่ทำให้ติด loop ซ้ำอีกเด็ดขาดในรอบถัดไป ให้ตรวจสอบ URL ปัจจุบัน
  และ indexed elements ของหน้าใหม่หลัง recovery นี้ก่อน แล้วเลือก action ที่ต่างออกไป
  จริงๆ (เช่น element อื่นที่ยังไม่เคยลอง) หรือถ้าเห็นชัดว่าไม่มีทางไปต่อจริงๆ ให้
  finish_task พร้อมอธิบายเหตุผล
- ถ้ามี "แพลนปัจจุบันที่ user ยืนยันแล้ว" แนบมาในข้อความ (เลขข้อ 1, 2, 3, ...) ให้ดูว่า
  action ที่คุณกำลังจะเรียกตอนนี้ทำให้ step ไหนของแพลน "เสร็จสมบูรณ์แล้วจริง" หรือไม่ (ต้อง
  เสร็จจริงตามหลักฐานที่จะเห็นหลัง action นี้ทำงาน ไม่ใช่แค่ "กำลังจะทำ") ถ้าใช่ ให้ใส่เลขข้อ
  นั้น (1-based ตามที่แสดงในแพลน) ลงใน parameter "completed_plan_step" ของ action นี้ด้วย —
  ถ้า action นี้ยังไม่ทำให้ step ไหนเสร็จ (เช่น เป็นแค่ขั้นตอนย่อยระหว่างทางของ step เดียวกัน)
  ห้ามใส่ completed_plan_step มาเลย (ละ parameter นี้ไว้) ห้ามเดา/ใส่เผื่อไว้ก่อน และห้ามใส่
  เลขข้อเดิมซ้ำสำหรับ step ที่เคยระบุว่าเสร็จไปแล้วในรอบก่อนหน้า — ถ้าไม่มี "แพลนปัจจุบัน"
  แนบมาเลย (ad-hoc task ไม่ผ่าน Confirm plan) ไม่ต้องสนใจ parameter นี้เลย
  - W_planbug: ระวังเป็นพิเศษกับ action type "fill"/"select"/"check" — ถ้า step ในแผนบรรยาย
    การกรอก/เลือก/ติ๊กค่านั้นตรงๆ อยู่แล้ว (เช่น "พิมพ์คำว่า X ลงในช่องค้นหา", "กรอกอีเมล",
    "เลือก Y จาก dropdown") ให้ถือว่า action fill/select/check นั้น "ทำให้ step นั้นเสร็จ
    สมบูรณ์แล้วทันที" ใส่ completed_plan_step ที่ action นี้เลย ห้ามรอไปใส่ที่ action ถัดไป
    (เช่น กด Enter/คลิกปุ่มค้นหา) เพราะ step ที่บรรยายแค่ "พิมพ์/กรอก/เลือก" เฉยๆ ไม่ได้รวม
    การกดส่ง/คลิกถัดไปด้วย — ยกเว้นถ้า step นั้นบรรยายรวมทั้งสองอย่างไว้ในข้อเดียวกันจริงๆ
    (เช่น "พิมพ์คำค้นหาแล้วกด Enter") ถึงจะรอใส่ที่ action ที่กดส่งจริง
- ถ้าต้องการอ่าน "เนื้อหา" บนหน้าเว็บ (เช่น นับจำนวนสินค้า, อ่าน/สรุปตาราง, หาค่าที่ปรากฏ
  อยู่บนหน้า) ไม่ใช่แค่หา element เพื่อกด/กรอก ให้ใช้ type: "read_page_data" พร้อม "query"
  (คำถามที่ต้องการคำตอบ) และ "target_hint" (CSS selector ที่คาดว่าตรงกับ element/แถวตาราง/
  รายการที่มีข้อมูลนั้น เช่น ".inventory_item" หรือ "table tbody tr") — ถ้าคำถามตอบได้ด้วย
  การนับจำนวนล้วนๆ (เช่น "มีสินค้ากี่ชิ้น") ให้ favor การนับตรงๆ เสมอ (ระบบจะนับให้จาก
  target_hint โดยตรง เร็ว/ประหยัด token กว่าให้ดึงตารางทั้งก้อนมานับเอง) ไม่ต้องขอให้ดึง
  เนื้อหาเต็มมาก่อนแล้วค่อยนับ — เรียก read_page_data เฉพาะตอนจำเป็นจริงๆ เท่านั้น ไม่ต้อง
  เรียกทุก step ถ้าไม่มีคำถามเกี่ยวกับเนื้อหาหน้าเว็บที่ยังตอบไม่ได้
- W_listformat: ตอนสรุปผลลัพธ์จาก read_page_data (ชื่อคน/username/รายการใดๆ) ใน finish_task
  ต้อง "คัดลอกตัวสะกดตรงตามที่ระบบส่งกลับมาทุกตัวอักษร" ห้ามพิมพ์จากความจำ/เดาการสะกดใหม่/
  แก้ไขให้ดู "ถูกต้องกว่า" เด็ดขาด (เช่น เห็น "Cierra Vaga" ต้องตอบ "Cierra Vaga" ไม่ใช่เปลี่ยน
  เป็น "Cierra Vega" ทั้งที่ดูเหมือนชื่อที่คุ้นเคยกว่า) — ถ้าข้อมูลที่ดึงมามีคำอธิบายกำกับว่า
  "ใกล้เคียงกับคำค้น ... ไม่ตรงกันเป๊ะ" ให้บอก user ตรงๆ ว่าเป็นการเดา ไม่ใช่ตรงกันเป๊ะ ไม่ใช่
  เงียบๆ ปัดเป็นคำตอบที่มั่นใจ เมื่อ goal ขอ "รายชื่อ"/list ของหลายรายการ ให้เรียงลำดับตาม
  ตัวอักษร (A-Z) ก่อนตอบเสมอเพื่อให้อ่านง่าย เว้นแต่ goal ระบุลำดับอื่นชัดเจน (เช่น "เรียงตาม
  วันที่") — การเรียงลำดับใหม่ทำได้เฉพาะ "ลำดับที่แสดง" เท่านั้น ห้ามเปลี่ยนตัวสะกด/เนื้อหาของ
  แต่ละรายการระหว่างเรียงเด็ดขาด
- W46: ก่อนเรียก finish_task พร้อมข้อความทำนอง "ไม่มีข้อมูล"/"ไม่พบ"/"หาไม่เจอ" ต้องทำ 2 อย่างนี้
  ก่อนเสมอ: (ก) ตรวจ conversation history ของ session นี้ (ผลลัพธ์ action ก่อนหน้า/
  "Action ล่าสุดที่คุณเพิ่งทำไป" ที่แนบมาในข้อความ) ว่าเคยค้นหา/เจอข้อมูลที่เกี่ยวข้องกับ
  คำถามนี้มาก่อนหรือยัง (ข) ถ้ายังไม่เคยลองค้นหาเลยสักครั้ง ต้องเรียก action ที่มีอยู่ (fill
  ช่องค้นหาแล้วกด/read_page_data) อย่างน้อย 1 ครั้งก่อนเสมอ ถึงจะ finish_task ว่าไม่พบได้ —
  ห้ามสรุปว่า "ไม่มีข้อมูล" จากการดูหน้าปัจจุบันเฉยๆ โดยไม่เคยลองค้นหาเลย
- ค้นหาแล้วจริงๆ (ทำตามข้อข้างบนครบแล้ว) แต่ยังไม่พบเป้าหมายที่ goal ระบุมาตรงๆ (เช่น
  username/ชื่อ/รหัสที่เจาะจงเป็นตัวๆ) ห้ามลงมือ "แก้ปัญหาแทน" ด้วยการทำ action ที่ไม่ได้อยู่
  ใน scope ของ goal เดิมเด็ดขาด เช่น ไปหน้า Add/Create เพื่อสร้างรายการใหม่ทดแทนของที่หาไม่เจอ,
  แก้ไข/ลบรายการอื่นที่ไม่ใช่เป้าหมายที่ระบุ, หรือ เดา/เลือกรายการอื่นที่ "ดูใกล้เคียง" มาทำแทน —
  goal ที่บอกให้แก้ไขของที่มีอยู่แล้ว (เช่น "แก้ไข role ของ user X") ไม่ได้แปลว่า "สร้าง X ถ้ายังไม่มี"
  ไม่ว่ากรณีใด สิ่งเดียวที่ทำได้คือ finish_task(success=false) รายงานตรงๆ ว่าไม่พบเป้าหมายที่ระบุ
  ให้ user ตัดสินใจเองว่าจะเอาอย่างไรต่อ
- คำถามที่ไม่มี verb สั่งงานตรงๆ (เช่น "อายุเท่าไหร่", "ราคาเท่าไหร่") ห้ามตีความว่าเป็น
  "แค่ถามเฉยๆ ไม่ต้องลงมือทำอะไร" — ทุกคำถามที่ต้องใช้ข้อมูลจากหน้าเว็บที่ยังไม่เห็นชัดในหน้า
  ปัจจุบัน นับเป็นคำสั่งให้ค้นหาโดยปริยายเสมอ (เทียบเท่ากับมีคำว่า "ค้นหา"/"หา" นำหน้า)
- ถ้าเห็น element ที่ label ต่อท้ายด้วย "[ซ่อนอยู่ — อาจต้อง hover แถวก่อน]" (ปุ่ม/ลิงก์ที่
  ยังไม่แสดงผลเต็มที่จนกว่าจะ hover แถว/บริเวณรอบๆ ก่อน เช่น ปุ่ม action ในแถวอีเมลที่โผล่มา
  ตอน hover เท่านั้น) ให้เรียก type: "hover" กับ index นั้นก่อน 1 ครั้ง แล้วค่อยคลิกต่อได้เลย
  (ไม่จำเป็นต้อง get_snapshot ใหม่ก่อนก็ได้ — ถ้าคลิกตรงๆ โดยไม่ hover ก่อน ระบบ retry จะ
  ลอง hover ให้อัตโนมัติตั้งแต่รอบที่ 2 อยู่แล้วเช่นกัน)
- W50: dropdown/menu ที่เห็นบนหน้าเว็บมี 2 แบบ ต้องแยกให้ออกก่อนเลือกวิธีโต้ตอบ:
  (ก) native dropdown จริง (element tag เป็น "select") — ใช้ type: "select" พร้อม
  "label" ตามปกติเหมือนเดิม (ก) นี้ยังทำงานถูกต้องอยู่แล้ว ไม่ต้องเปลี่ยน
  (ข) custom dropdown/menu (element ที่ label/ป้ายดูเหมือนตัวเลือก/dropdown แต่ tag ไม่ใช่
  "select" — เช่น div/button ที่มี role=combobox, หรือหลังคลิกเปิดแล้วเห็น element
  role=option/menuitem โผล่ขึ้นมาใหม่ในลิสต์) — วิธีที่เสถียรที่สุดสำหรับแบบนี้คือลำดับ
  คีย์บอร์ด ไม่ใช่การไล่คลิก selector ลึกๆ: (1) type: "click" ที่ index ของตัว dropdown
  เพื่อเปิดมันก่อน (2) type: "press_key" ที่ index เดิมนั้น พร้อม key: "ArrowDown" (ทำซ้ำ
  ได้หลายครั้งถ้าต้องเลื่อนผ่านหลายตัวเลือก) (3) type: "press_key" ที่ index เดิม พร้อม
  key: "Enter" เพื่อยืนยันตัวเลือกที่ไฮไลต์อยู่ — ให้ใช้ลำดับนี้ทันทีถ้าคลิกตัวเลือกตรงๆ
  ไม่สำเร็จ หรือถ้าเห็นชัดว่า element เป็น custom dropdown ตั้งแต่แรก (ไม่ต้องเสีย step
  ลองคลิกตัวเลือกก่อนก็ได้ถ้ามั่นใจ)
- บรรทัด "เวลาปัจจุบัน (Asia/Bangkok)" ที่แนบมาในข้อความทุกครั้งคือเวลาจริงจากเซิร์ฟเวอร์
  ณ ขณะนั้น ให้ยึดเป็นความจริงเสมอเมื่อต้องอ้างอิงวันที่/เวลาปัจจุบัน ห้ามเดาหรืออ้างอิง
  วันที่จาก training data ของตัวเองเด็ดขาด แม้คำถามจะดูเหมือนต้องใช้ "ความรู้ทั่วไป"
  เกี่ยวกับวันที่ก็ตาม (เช่น "วันนี้วันอะไร", "ตอนนี้กี่โมง", "ปีนี้ปีอะไร")
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
_THAI_WEEKDAYS = ("วันจันทร์", "วันอังคาร", "วันพุธ", "วันพฤหัสบดี", "วันศุกร์", "วันเสาร์", "วันอาทิตย์")
_THAI_MONTHS = (
    "มกราคม", "กุมภาพันธ์", "มีนาคม", "เมษายน", "พฤษภาคม", "มิถุนายน",
    "กรกฎาคม", "สิงหาคม", "กันยายน", "ตุลาคม", "พฤศจิกายน", "ธันวาคม",
)


def _current_bangkok_time_text() -> str:
    """เวลาจริงจากเซิร์ฟเวอร์ ณ ขณะเรียก (Asia/Bangkok) — เรียกสดทุกครั้งที่
    _build_user_turn_text() ถูกเรียก (ทุก step ของ loop) ไม่ cache ค่าไว้ข้ามรอบ เพราะ LLM
    เองไม่มีการรับรู้เวลาจริง ต้องฉีดเข้า context ทุก turn ไม่งั้นจะเดา/อ้างอิงวันที่จาก
    training data ผิดๆ (ดู SYSTEM_PROMPT ข้อสุดท้ายที่สั่งให้ยึดบรรทัดนี้เป็นความจริงเสมอ)"""
    now = datetime.now(tz=ZoneInfo("Asia/Bangkok"))
    weekday = _THAI_WEEKDAYS[now.weekday()]
    month = _THAI_MONTHS[now.month - 1]
    buddhist_year = now.year + 543
    return f"{weekday}ที่ {now.day} {month} {buddhist_year} เวลา {now.strftime('%H:%M')} น."


def _build_user_turn_text(
    goal: str,
    page_text: str,
    manual_context: str = "",
    memory_context: str = "",
    long_term_context: str = "",
    vision_context: str = "",
    site_manual_context: str = "",
    current_url: str = "",
    action_history_context: str = "",
    plan_context: str = "",
    verification_context: str = "",
) -> str:
    text = f"Goal: {goal}"
    # แนบเวลาจริงของเซิร์ฟเวอร์ทุก turn (ไม่ใช่แค่ตอนเริ่ม session) — LLM ไม่มีการรับรู้
    # เวลาจริงในตัวเอง ต้องฉีดเข้า context ทุกครั้งที่เรียก _build_user_turn_text() (ดู
    # _current_bangkok_time_text() ด้านบน — อ่านเวลาสดทุกครั้ง ไม่ cache ค่าเดิมค้างไว้)
    text += f"\n\nเวลาปัจจุบัน (Asia/Bangkok): {_current_bangkok_time_text()}"
    # W43: plan_context มีค่าเฉพาะ task ที่ผ่าน Confirm plan (confirm_plan=True/
    # approved_plan) มาก่อนเท่านั้น — ad-hoc task (ไม่มีแพลนเลย) ได้ "" เสมอ ไม่มี section
    # นี้โผล่มาปนเลย (backward compatible ทุกประการกับ prompt เดิม) วางไว้ก่อน "หน้าเว็บ
    # ปัจจุบัน" เพราะเป็นบริบทระดับ task (เหมือน Goal) ไม่ใช่ข้อมูลเฉพาะ step นี้แบบ
    # manual_context/memory_context ด้านล่าง — ให้ LLM เห็นเลขข้อของแผนก่อนตัดสินใจว่า action
    # ที่กำลังจะทำ "ทำให้ step ไหนเสร็จ" (ดู completed_plan_step ใน _BROWSER_ACTION_PARAMS)
    if plan_context:
        text += f"\n\nแพลนปัจจุบันที่ user ยืนยันแล้ว (แต่ละบรรทัดคือ 1 step ตามเลขข้อ):\n{plan_context}"
    # W30 (recovered from an earlier exploratory branch — ดู roadmap.txt): เพิ่มหลัง user
    # รายงานว่า agent บางครั้งดูเหมือนตัดสินใจจาก state เก่า (เช่นหน้าเว็บเปลี่ยนไปเองระหว่าง
    # ทาง แต่ยังพูดถึงหน้าเดิม) — get_snapshot() ที่ orchestrator.py เรียกทุก step อยู่แล้ว
    # เป็นการอ่านสด (live) จาก page จริงเสมออยู่แล้ว ไม่มี cache ทางโค้ด แต่ก่อนหน้านี้
    # page.url ไม่เคยถูกโชว์เป็นข้อความชัดๆ ให้ LLM เห็นเลย (มีแค่ indexed elements list) —
    # โมเดลเลยต้องเดาว่า "นี่หน้าเดิมหรือหน้าใหม่" จาก element ที่หน้าตาอาจคล้ายกันได้ ใส่
    # URL ปัจจุบันจริงตรงๆ ทุก step (อ่านจาก page.url สดๆ ไม่ใช่ค่าที่จำมาจาก step ก่อน) ให้
    # หลักฐานชัดเจนกว่าการเดาจาก element เพียงอย่างเดียว
    if current_url:
        text += f"\n\nURL ปัจจุบันจริงของหน้าเว็บ (อ่านสดจาก browser ทุก step): {current_url}"
    text += f"\n\nหน้าเว็บปัจจุบัน:\n{page_text}"
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
    # W32: action ล่าสุดไม่กี่ step (ทั้งสำเร็จและล้มเหลว) แยกจาก memory_context ด้านบนที่
    # กรองเฉพาะ fail — ให้เห็นชัดๆ ว่า "ตัวเองเพิ่งทำอะไรไปบ้าง" กันเลือก action เดิมซ้ำ
    # (เช่น กดปุ่มเดิมสำเร็จซ้ำหลายครั้งแต่ไม่มีความคืบหน้าจริงต่อ goal — memory_context
    # เปล่าๆ เพราะไม่มี action ไหน fail เลยสักครั้ง)
    if action_history_context:
        text += (
            "\n\nAction ล่าสุดที่คุณเพิ่งทำไป (เรียงตามลำดับ ไม่ว่าจะสำเร็จหรือล้มเหลว) — "
            "ถ้าเห็นว่ากำลังจะเลือก action ซ้ำ/คล้ายกับที่เพิ่งทำไปโดยไม่มีความคืบหน้าใหม่ "
            "จริงต่อ goal ให้เปลี่ยนไปทำ action ที่ต่างออกไปแทน:\n"
            f"{action_history_context}"
        )
    # W50: client-side action verification — สัญญาณเสริมจากโค้ด (ไม่ต้องพึ่ง LLM สังเกต
    # เอง) ว่า action ก่อนหน้าที่คืน [OK] จริงๆ แล้วอาจไม่มีผลอะไรกับหน้าเว็บเลย (ดู
    # orchestrator.py::run_task() จุดคำนวณ verification_context — เทียบ element
    # count/เนื้อหาหน้าก่อน-หลัง action) ว่างเปล่าถ้าไม่มีสัญญาณผิดปกติ
    if verification_context:
        text += f"\n\n{verification_context}"
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
                # W45: อ่านเนื้อหาบนหน้าเว็บ (นับจำนวน/อ่านตาราง/สรุปข้อมูล) — ต่างจาก
                # click/fill/select ตรงที่ไม่ได้กด/แก้ไข element ใดๆ เลย แค่ query
                # เนื้อหาที่มองเห็นอยู่แล้วกลับมาตอบ ดู "query"/"target_hint" ด้านล่าง
                "read_page_data",
                # W47: เลื่อนเมาส์ไปวางไว้บน element (ไม่คลิก) — ใช้ trigger CSS :hover
                # ของ element/บรรพบุรุษก่อนกด element ที่ซ่อนอยู่จนกว่าจะ hover แถวแม่
                # (ดู label marker "[ซ่อนอยู่ — อาจต้อง hover แถวก่อน]" ด้านล่าง) — ปกติ
                # ไม่ต้องเรียกเองเพราะ click retry รอบ 2 เป็นต้นไปจะ hover ให้อัตโนมัติ
                # อยู่แล้ว เรียกเองได้ถ้าต้องการ get_snapshot ใหม่หลัง hover ก่อนตัดสินใจ
                "hover",
                # W50: ส่ง key ไปยัง element ตาม index — ใช้กับ custom dropdown/menu ที่
                # ไม่ใช่ <select><option> จริง (ดู "key" parameter ด้านล่าง + กติกาการใช้
                # ใน SYSTEM_PROMPT)
                "press_key",
            ],
            "description": "ชนิด action",
        },
        "index": {
            "type": "integer",
            "description": "index ของ element (click/fill/select/check/submit/delete/purchase/pay/hover/press_key)",
        },
        "text": {"type": "string", "description": "ข้อความที่จะกรอก (fill)"},
        "label": {"type": "string", "description": "ตัวเลือกที่จะเลือกใน dropdown (select)"},
        "key": {
            "type": "string",
            "enum": ["ArrowDown", "ArrowUp", "Enter", "Escape", "Tab", "Space"],
            "description": (
                "ปุ่มคีย์บอร์ดที่จะกด (press_key เท่านั้น) — ใช้กับ custom dropdown/menu "
                "ที่ไม่ใช่ <select><option> จริง: กด ArrowDown/ArrowUp เพื่อเลื่อนตัวเลือก "
                "ที่ไฮไลต์ แล้วกด Enter เพื่อยืนยันตัวเลือกนั้น"
            ),
        },
        "direction": {"type": "string", "enum": ["up", "down"], "description": "ทิศทางเลื่อนจอ (scroll)"},
        "url": {"type": "string", "description": "URL ปลายทาง (goto)"},
        "tab_index": {"type": "integer", "description": "ลำดับ tab ที่จะสลับไป (switch_tab)"},
        "query": {
            "type": "string",
            "description": (
                "คำถามที่ต้องการคำตอบจากเนื้อหาบนหน้าเว็บ (read_page_data เท่านั้น) "
                "เช่น 'มีสินค้ากี่ชิ้น' หรือ 'ราคาสินค้าชิ้นนี้เท่าไหร่'"
            ),
        },
        "target_hint": {
            "type": "string",
            "description": (
                "CSS selector ที่คาดว่าตรงกับ element/แถวตาราง/รายการที่มีข้อมูลที่ต้องการ "
                "(read_page_data เท่านั้น) เช่น '.inventory_item' หรือ 'table tbody tr'"
            ),
        },
        # W43: optional เสมอ (ไม่อยู่ใน "required" ด้านล่าง) — ไม่ส่งมาก็ได้ถ้า action นี้ไม่
        # เกี่ยวกับ plan step ไหนเลย/ยังไม่มี plan ให้ทำตาม (ad-hoc task ที่ไม่ผ่าน Confirm
        # plan) เห็นได้จาก orchestrator.py ที่ตอนนี้อ่านค่านี้ผ่าน tool_input.get(...) เฉยๆ
        # (คืน None ถ้าไม่มี ไม่ throw) — ห้าม LLM ทำเป็น required เด็ดขาด กันพัง backward
        # compat กับ task ที่ไม่มีแพลนเลย
        "completed_plan_step": {
            "type": "integer",
            "description": (
                "ใส่เลขข้อ (1-based ตามที่แสดงใน \"แพลนปัจจุบัน\") ถ้า action ที่เพิ่งเรียกนี้"
                " ทำให้ step นั้นของแพลนเสร็จสมบูรณ์แล้ว — ไม่ต้องใส่ (ละไว้) ถ้า action นี้ยัง"
                " ไม่ทำให้ step ไหนเสร็จ หรือไม่มีแพลนแนบมาในบทสนทนานี้เลย"
            ),
        },
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
        "message": {"type": "string", "description": "สรุปผลสั้นๆ ว่าทำอะไรไป/ทำไมหยุด — รายชื่อ/ข้อมูลที่ดึงมาต้องคัดลอกตัวสะกดตรงตามต้นฉบับ ห้ามเดา/แก้สะกด และถ้าเป็นรายการหลายรายการให้เรียงตามตัวอักษรก่อนตอบ (ดู W_listformat ใน system prompt)"},
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


# --- W_procmem: Abstractor tool (llm.abstract_trajectory()) — single-shot call ต่างหาก
# ไม่ใช่ tool ที่อยู่ใน agent loop หลัก (browser_action/finish_task ด้านบน) เรียกแค่ครั้ง
# เดียวแบบ fire-and-forget หลัง task สำเร็จ (ดู orchestrator.py) เพื่อกลั่น trajectory
# เป็น template ให้ core/procedural_memory.py เก็บไว้ — "target" ของแต่ละ step ต้องมี
# shape เดียวกับ locator descriptor ที่ core/dom_locator.py::compute_locator_descriptor()
# คำนวณไว้แล้วในแต่ละ step ของ trajectory (ดู _format_trajectory_for_abstractor()
# ด้านล่าง) เพื่อให้ fastpath_executor.py เอาไป resolve_locator() ต่อได้ตรงๆ ไม่ต้องแปลง
# รูปแบบอีกชั้น
_ABSTRACTOR_TARGET_SCHEMA = {
    "type": "object",
    "description": (
        "locator ของ element เป้าหมาย — คัดลอกมาจาก locator_descriptor ของ step ที่ตรงกัน"
        "ใน TRAJECTORY ตรงๆ ห้ามแต่งค่าขึ้นเอง"
    ),
    "properties": {
        "tag": {"type": "string"},
        "explicit_role": {"type": "string"},
        "implicit_role": {"type": "string"},
        "accessible_name": {"type": "string"},
        "data_testid": {"type": "string"},
        "css_fallback": {"type": "string"},
    },
}
# W_procmem: step schema เดียวที่ใช้ร่วมกันทั้ง ABSTRACTOR_TOOL (steps ทั้งชุด) และ
# PROCEDURAL_PLANNER_TOOL (แค่ตอน decision="adapt" ต้องส่ง patch step เดี่ยวๆ กลับมา) —
# กันไม่ให้ schema สอง tool เพี้ยนไปคนละแบบทั้งที่ต้อง resolve_locator() ด้วยตรรกะเดียวกัน
_TEMPLATE_STEP_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["goto", "click", "fill", "select", "check", "press_key", "hover"],
        },
        "target": _ABSTRACTOR_TARGET_SCHEMA,
        "value": {
            "type": "string",
            "description": "ค่าที่จะกรอก/เลือก — ต้องเป็น {{slot_name}} เท่านั้น ห้ามมีค่าจริงหลงเหลืออยู่เด็ดขาด",
        },
        "widget": {
            "type": "string",
            "description": "ระบุถ้า element เป็น custom widget ที่ไม่ใช่ native (เช่น 'vue_dropdown', 'autocomplete', 'date_picker')",
        },
        "sensitive": {
            "type": "boolean",
            "description": "true ถ้า step นี้เกี่ยวข้องกับรหัสผ่าน/ข้อมูลลับ — ต้อง omit ค่าจริงออกจาก value โดยสิ้นเชิง",
        },
    },
    "required": ["action"],
}
_ABSTRACTOR_PARAMS = {
    "type": "object",
    "properties": {
        "goal_pattern": {
            "type": "string",
            "description": "คำอธิบาย task class แบบทั่วไป (paraphrase) ไม่ใช่ถ้อยคำ/ค่าเฉพาะของ goal ตัวนี้ตัวเดียว",
        },
        "url_pattern": {"type": "string", "description": "URL เริ่มต้นของ task class นี้"},
        "slots": {
            "type": "array",
            "description": "รายชื่อ slot ทั้งหมดที่ใช้ใน steps เรียงตามลำดับที่ปรากฏครั้งแรก",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["name"],
            },
        },
        "steps": {"type": "array", "items": _TEMPLATE_STEP_SCHEMA},
    },
    "required": ["goal_pattern", "steps", "slots"],
}
_ABSTRACTOR_DESC = (
    "กลั่น GOAL + TRAJECTORY ของ action ที่ทำสำเร็จแล้วให้เป็น template ที่นำกลับมาใช้ซ้ำ "
    "ได้ (steps + locator + {{slot}} placeholder แทนค่าจริงเสมอ)"
)
ABSTRACTOR_TOOL = {"name": "emit_template", "description": _ABSTRACTOR_DESC, "input_schema": _ABSTRACTOR_PARAMS}
_GROQ_ABSTRACTOR_TOOLS = [
    {"type": "function", "function": {"name": "emit_template", "description": _ABSTRACTOR_DESC, "parameters": _ABSTRACTOR_PARAMS}},
]
_GEMINI_ABSTRACTOR_TOOLS = [
    {"function_declarations": [{"name": "emit_template", "description": _ABSTRACTOR_DESC, "parameters": _ABSTRACTOR_PARAMS}]},
]

_ABSTRACTOR_SYSTEM_PROMPT = (
    "You are a Workflow Abstractor for a browser-automation agent.\n"
    "Given a GOAL and a TRAJECTORY of actions that successfully completed it,\n"
    "distill a REUSABLE, PARAMETERIZED template.\n\n"
    "RULES\n"
    "- Separate STRUCTURE from DATA. Replace every concrete input value with a\n"
    "  named slot {{slot_name}} in the step's \"value\" field. NEVER keep real\n"
    "  values, PII, passwords, IDs, or record-specific data in the template.\n"
    "- For each step's \"target\", copy the locator fields (tag/explicit_role/\n"
    "  implicit_role/accessible_name/data_testid/css_fallback) directly from the\n"
    "  matching TRAJECTORY entry's locator_descriptor — do not invent new values.\n"
    "- Keep only steps with action in [goto, click, fill, select, check, press_key,\n"
    "  hover] that a TRAJECTORY entry actually shows executing successfully.\n"
    "- CRITICAL: Emit ONE template step per TRAJECTORY entry, in the SAME order, and\n"
    "  NO MORE than that. Never add an extra step just because the GOAL implies it\n"
    "  must have happened (e.g. a login form) — if TRAJECTORY does not contain a\n"
    "  matching entry for it, that step happened outside this trajectory (such as an\n"
    "  automated login bootstrap) and must NOT appear in the template at all.\n"
    "- Mark any password/secret step with \"sensitive\": true and omit its literal\n"
    "  value from \"value\" entirely (reference only the slot name).\n"
    "- \"slots\" must list every slot used, in order of first appearance.\n"
    "- \"goal_pattern\" must describe the TASK CLASS, not this one instance — AND must\n"
    "  describe ONLY what the STEPS you are emitting actually do. If part of the\n"
    "  original GOAL (e.g. \"log in\") is not reflected in any step (because it\n"
    "  happened outside this trajectory, such as an automated login bootstrap), do\n"
    "  NOT mention that part in goal_pattern at all — a future task matched against\n"
    "  this template should get exactly what these steps do, no more, no less.\n"
    "- Output ONLY valid JSON via the emit_template tool call. No prose."
)


def _format_trajectory_for_abstractor(trajectory: list[dict]) -> str:
    """แปลง self.memory.all() (orchestrator.py) เป็นข้อความสั้นๆ ให้ Abstractor อ่าน —
    เอาเฉพาะ action ที่กระทำ element จริงและสำเร็จเท่านั้น (ข้าม read_page_data/
    finish_task/action ที่ fail — ไม่มี locator_descriptor ให้อ้างอิงอยู่แล้วเพราะ
    actions.py คำนวณแค่ตอนสำเร็จ ดู core/actions.py) แต่ละบรรทัดเป็น JSON ก้อนเดียว
    (action + locator_descriptor + ค่าที่กรอก/เลือกจริง) ให้ LLM คัดลอก target ตรงๆ ได้
    ไม่ต้องตีความจาก prose"""
    lines = []
    for entry in trajectory:
        if not entry.get("success"):
            continue
        cmd = entry.get("cmd") or {}
        action = cmd.get("type")
        if action == "goto":
            lines.append(json.dumps({"action": "goto", "url": cmd.get("url", "")}, ensure_ascii=False))
            continue
        if action not in ("click", "fill", "select", "check", "press_key", "hover"):
            continue
        detail: dict[str, Any] = {"action": action, "locator_descriptor": entry.get("locator_descriptor") or {}}
        if action == "fill":
            detail["typed_value"] = cmd.get("text", "")
        elif action == "select":
            detail["selected_label"] = cmd.get("label", "")
        elif action == "press_key":
            detail["key"] = cmd.get("key", "")
        lines.append(json.dumps(detail, ensure_ascii=False))
    return "\n".join(lines) if lines else "(no successful element-targeting actions recorded)"


async def abstract_trajectory(
    client, model: str, goal: str, url: str, trajectory: list[dict], provider: str,
) -> Optional[dict]:
    """W_procmem: กลั่น trajectory ของ task ที่สำเร็จแล้ว (self.memory.all() จาก
    orchestrator.py) ให้เป็น template ที่มีโครงสร้าง (ดู core/procedural_memory.py) —
    เรียกครั้งเดียวแบบ fire-and-forget หลัง finish_task(success=True) เท่านั้น (ดู
    orchestrator.py) ห้าม throw ออกไปเด็ดขาดไม่ว่ากรณีใด (provider error/parse ผิดพลาด/
    ไม่เรียก tool กลับมา) — คืน None แทนเสมอ ให้ผู้เรียก skip การบันทึกเงียบๆ (เหมือน
    fallback pattern อื่นๆ ทั้งระบบ — ดู plan_memory.py/long_term_memory.py)"""
    trajectory_text = _format_trajectory_for_abstractor(trajectory)
    prompt = (
        f"GOAL: {goal}\nURL: {url}\nTRAJECTORY:\n{trajectory_text}\n\n"
        "Call emit_template now with the distilled reusable template."
    )
    try:
        if provider == "anthropic":
            response = await client.messages.create(
                model=model,
                max_tokens=2048,
                system=_ABSTRACTOR_SYSTEM_PROMPT,
                tools=[ABSTRACTOR_TOOL],
                tool_choice={"type": "tool", "name": "emit_template"},
                messages=[{"role": "user", "content": prompt}],
            )
            tool_use = next((b for b in response.content if b.type == "tool_use"), None)
            return tool_use.input if tool_use is not None else None

        if provider == "groq":
            response = await client.chat.completions.create(
                model=model,
                max_tokens=2048,
                messages=[
                    {"role": "system", "content": _ABSTRACTOR_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                tools=_GROQ_ABSTRACTOR_TOOLS,
                tool_choice={"type": "function", "function": {"name": "emit_template"}},
            )
            tool_calls = response.choices[0].message.tool_calls or []
            if not tool_calls:
                return None
            return json.loads(tool_calls[0].function.arguments)

        if provider == "gemini":
            gemini_model = client.GenerativeModel(
                model_name=model,
                tools=_GEMINI_ABSTRACTOR_TOOLS,
                tool_config={"function_calling_config": {"mode": "ANY"}},
                system_instruction=_ABSTRACTOR_SYSTEM_PROMPT,
            )
            response = await gemini_model.generate_content_async(
                contents=[{"role": "user", "parts": [{"text": prompt}]}],
            )
            for part in response.candidates[0].content.parts:
                fc = getattr(part, "function_call", None)
                if fc and fc.name == "emit_template":
                    return _gemini_struct_to_plain_python(fc.args)
            return None

        return None
    except Exception as e:
        print(f"⚠️ abstract_trajectory error: {e}", flush=True)
        return None


# --- W_procmem: Memory-augmented Planner tool (llm.plan_with_procedural_memory()) —
# single-shot call ต่างหาก เรียกจาก routes.py::generate_plan ก่อน plan_memory/LLM
# ร่างใหม่เสมอ (ดู module docstring ของ core/procedural_memory.py สำหรับลำดับความ
# สำคัญเต็มๆ) ตัดสินใจว่าจะ reuse/adapt/plan_fresh จาก candidate template ที่
# core/procedural_memory.py::find_candidate_templates() ดึงมาให้แล้ว
_PROCEDURAL_PLANNER_PARAMS = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["reuse", "adapt", "plan_fresh"]},
        "template_id": {
            "type": "string",
            "description": "template_id ของ candidate ที่เลือก (เฉพาะ reuse/adapt) — ต้องเป็นค่าที่มีอยู่จริงใน CANDIDATE_TEMPLATES เท่านั้น",
        },
        "confidence": {"type": "number", "description": "0.0-1.0 ความมั่นใจว่า candidate ตรงกับ task class + URL pattern + form fields ของ NEW_TASK จริง"},
        "slot_values": {
            "type": "object",
            "description": "ค่าที่จะแทน {{slot_name}} แต่ละตัว ดึงมาจาก NEW_TASK เท่านั้น ห้ามแต่งขึ้นเอง — slot ที่ NEW_TASK ไม่ได้ระบุมาให้เว้นว่าง/ไม่ต้องใส่ key นั้น",
        },
        "patch": {
            "type": "array",
            "description": "เฉพาะ decision=adapt — รายการแก้ไข steps ของ candidate ทีละจุด",
            "items": {
                "type": "object",
                "properties": {
                    "op": {"type": "string", "enum": ["replace", "insert", "remove"]},
                    "index": {"type": "integer", "description": "0-based index ใน steps ของ candidate ที่ op นี้กระทำ"},
                    "step": _TEMPLATE_STEP_SCHEMA,
                },
                "required": ["op", "index"],
            },
        },
        "reason": {"type": "string", "description": "เหตุผลสั้นๆ ของการตัดสินใจนี้"},
    },
    "required": ["decision", "confidence", "reason"],
}
_PROCEDURAL_PLANNER_DESC = (
    "ตัดสินใจว่าจะ reuse/adapt template ที่มีอยู่แล้ว หรือปล่อยให้ร่างแผนใหม่ (plan_fresh) "
    "จาก candidate template ที่ค้นมาให้แล้ว"
)
PROCEDURAL_PLANNER_TOOL = {
    "name": "plan_decision", "description": _PROCEDURAL_PLANNER_DESC, "input_schema": _PROCEDURAL_PLANNER_PARAMS,
}
_GROQ_PROCEDURAL_PLANNER_TOOLS = [
    {"type": "function", "function": {"name": "plan_decision", "description": _PROCEDURAL_PLANNER_DESC, "parameters": _PROCEDURAL_PLANNER_PARAMS}},
]
_GEMINI_PROCEDURAL_PLANNER_TOOLS = [
    {"function_declarations": [{"name": "plan_decision", "description": _PROCEDURAL_PLANNER_DESC, "parameters": _PROCEDURAL_PLANNER_PARAMS}]},
]

_PROCEDURAL_PLANNER_SYSTEM_PROMPT = (
    "You are the Planner of a browser agent with PROCEDURAL MEMORY.\n"
    "You receive a NEW_TASK and up to K CANDIDATE_TEMPLATES retrieved by\n"
    "similarity. Choose the fastest CORRECT way to act.\n\n"
    "DECIDE exactly one:\n"
    "- \"reuse\": a candidate matches the task class AND the current page fits.\n"
    "  Return its template_id, confidence 0-1, and slot_values filled ONLY from\n"
    "  NEW_TASK. Leave a slot out of slot_values if NEW_TASK doesn't provide it.\n"
    "- \"adapt\": a candidate is close but needs small changes. Return\n"
    "  template_id, slot_values, and a \"patch\" (steps to add/replace/remove).\n"
    "- \"plan_fresh\": no candidate is good enough. Leave template_id out; the\n"
    "  slow path will handle it.\n\n"
    "RULES\n"
    "- Match on task class + URL pattern + form fields, NOT surface wording.\n"
    "- NEVER invent slot values. Use only data present in NEW_TASK.\n"
    "- If your best confidence would be < 0.6, choose \"plan_fresh\" instead.\n"
    "- Output ONLY valid JSON via the plan_decision tool call. No prose."
)

_PROCEDURAL_PLANNER_SAFE_DEFAULT: dict[str, Any] = {
    "decision": "plan_fresh", "template_id": None, "confidence": 0.0, "slot_values": {}, "patch": None,
    "reason": "",
}


def _format_candidates_for_planner(candidates: list[dict]) -> str:
    """สรุป candidate ให้ Planner อ่าน — ตัดรายละเอียด locator/target ของแต่ละ step
    ออก (เหลือแค่ลำดับ action type) กันไม่ให้ prompt บวมโดยไม่จำเป็น เพราะ Planner
    แค่ต้อง "ตัดสินใจ" ว่า candidate ไหนตรงกับ task class เท่านั้น — steps เต็มๆ พร้อม
    locator จริง ผู้เรียก (routes.py) ค่อยไปดึงจาก candidates list เดิม (ที่
    find_candidate_templates() คืนมาให้ตั้งแต่แรก) มาประกอบเป็น template สุดท้ายเอง
    หลัง Planner ตัดสินใจแล้ว ไม่ต้องให้ LLM คัดลอก locator กลับมาเองให้เสี่ยงพิมพ์ผิด"""
    lines = []
    for c in candidates:
        step_sequence = "/".join(str(s.get("action", "")) for s in c.get("steps", []))
        slot_names = [s.get("name") for s in c.get("slots", []) if isinstance(s, dict) and s.get("name")]
        lines.append(json.dumps({
            "template_id": c.get("template_id"),
            "goal_pattern": c.get("goal_pattern", ""),
            "url_pattern": c.get("url_pattern", ""),
            "slots": slot_names,
            "step_sequence": step_sequence,
        }, ensure_ascii=False))
    return "\n".join(lines) if lines else "(no candidates)"


async def plan_with_procedural_memory(
    client, model: str, goal: str, url: str, page_fingerprint: str, candidates: list[dict], provider: str,
    *, has_auto_login: bool = False,
) -> dict:
    """W_procmem: ตัดสินใจ reuse/adapt/plan_fresh จาก candidate template ที่
    core/procedural_memory.py::find_candidate_templates() ดึงมาให้ — เรียกจาก
    routes.py::generate_plan ก่อน plan_memory/LLM ร่างใหม่เสมอ (ดู module docstring
    ของ core/procedural_memory.py) ห้าม throw ออกไปเด็ดขาดไม่ว่ากรณีใด — คืน
    _PROCEDURAL_PLANNER_SAFE_DEFAULT (decision=plan_fresh, confidence=0.0) แทนเสมอถ้า
    provider error/parse ผิดพลาด/ไม่เรียก tool กลับมา ให้ caller fallback ไปทาง
    plan_memory/LLM ร่างใหม่ตามปกติ (เหมือนไม่มี procedural memory เลย)

    page_fingerprint: label/role ของ element บนหน้าปัจจุบัน (จาก
    perception.get_snapshot() text_repr) ถ้า session มี page เปิดค้างอยู่แล้ว — ว่างเปล่า
    ถ้าเป็น task ใหม่ที่ยังไม่เคยเปิดหน้าเลย ใช้ช่วยยืนยันว่า candidate ที่ดูตรงกันจาก
    goal text เพียวๆ ยังตรงกับสภาพหน้าเว็บจริงตอนนี้ด้วยหรือไม่ (ป้องกันกรณีเว็บถูก
    redesign ไปแล้วทั้งที่ goal ยังพิมพ์เหมือนเดิม)

    has_auto_login (W_procmem, แก้ปัญหาจริงที่เจอตอน Phase 4 validation): True ถ้า
    โดเมนนี้มี credential เก็บไว้แล้ว (ดู site_learning/storage.py::credentials_exist —
    ผู้เรียก routes.py เป็นคนเช็คให้) — auto_login.py จะ login ให้อัตโนมัติ "นอก" LLM
    loop เสมอไม่ว่าทางไหน (ดู orchestrator.py::_maybe_auto_login) ทำให้ template ที่ไม่มี
    step login เลยยังถือว่า "ครบ" สำหรับ task ที่ implies ว่าต้อง login ก่อน — ถ้าไม่บอก
    Planner เรื่องนี้ มันจะเดา (ผิด) ว่า template ขาด step login ไปแล้วปฏิเสธ reuse ทั้งที่
    จริงๆ ใช้ได้ปกติ (เจอบั๊กนี้จริงกับ OrangeHRM ระหว่างทดสอบ)"""
    candidates_text = _format_candidates_for_planner(candidates)
    auto_login_note = (
        "AUTO_LOGIN: this domain has stored credentials — login happens automatically "
        "and invisibly before any task starts, completely outside any template's steps. "
        "A candidate template lacking login steps is NOT incomplete because of that; do "
        "not penalize it or require login steps to be present.\n"
        if has_auto_login else
        "AUTO_LOGIN: none configured for this domain — if the task needs login, a "
        "matching template should actually contain those steps.\n"
    )
    prompt = (
        f"NEW_TASK: {goal}\nCURRENT_URL: {url}\n"
        f"PAGE_FINGERPRINT: {page_fingerprint or '(no live page yet)'}\n"
        f"{auto_login_note}"
        f"CANDIDATE_TEMPLATES:\n{candidates_text}\n\n"
        "Call plan_decision now."
    )
    try:
        if provider == "anthropic":
            response = await client.messages.create(
                model=model,
                max_tokens=1024,
                system=_PROCEDURAL_PLANNER_SYSTEM_PROMPT,
                tools=[PROCEDURAL_PLANNER_TOOL],
                tool_choice={"type": "tool", "name": "plan_decision"},
                messages=[{"role": "user", "content": prompt}],
            )
            tool_use = next((b for b in response.content if b.type == "tool_use"), None)
            decision = tool_use.input if tool_use is not None else None
        elif provider == "groq":
            response = await client.chat.completions.create(
                model=model,
                max_tokens=1024,
                messages=[
                    {"role": "system", "content": _PROCEDURAL_PLANNER_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                tools=_GROQ_PROCEDURAL_PLANNER_TOOLS,
                tool_choice={"type": "function", "function": {"name": "plan_decision"}},
            )
            tool_calls = response.choices[0].message.tool_calls or []
            decision = json.loads(tool_calls[0].function.arguments) if tool_calls else None
        elif provider == "gemini":
            gemini_model = client.GenerativeModel(
                model_name=model,
                tools=_GEMINI_PROCEDURAL_PLANNER_TOOLS,
                tool_config={"function_calling_config": {"mode": "ANY"}},
                system_instruction=_PROCEDURAL_PLANNER_SYSTEM_PROMPT,
            )
            response = await gemini_model.generate_content_async(
                contents=[{"role": "user", "parts": [{"text": prompt}]}],
            )
            decision = None
            for part in response.candidates[0].content.parts:
                fc = getattr(part, "function_call", None)
                if fc and fc.name == "plan_decision":
                    decision = _gemini_struct_to_plain_python(fc.args)
                    break
        else:
            decision = None

        if decision is None:
            return dict(_PROCEDURAL_PLANNER_SAFE_DEFAULT)

        # W_procmem defense-in-depth: ไม่เชื่อ confidence/decision ของ LLM ตรงๆ 100% —
        # บังคับ plan_fresh เองถ้า confidence ต่ำกว่าเกณฑ์ แม้ LLM จะเผลอตอบ
        # decision="reuse"/"adapt" มาก็ตาม (กันโมเดลมั่นใจเกินจริง)
        confidence = float(decision.get("confidence", 0.0) or 0.0)
        if confidence < settings.procedural_memory_min_confidence:
            return {**_PROCEDURAL_PLANNER_SAFE_DEFAULT, "confidence": confidence, "reason": decision.get("reason", "")}

        chosen_decision = decision.get("decision", "plan_fresh")
        template_id = decision.get("template_id")
        # W_procmem: "template_id" ไม่ได้อยู่ใน required ของ schema (ไม่มีทางบังคับแบบ
        # "required เฉพาะตอน decision=reuse/adapt" ข้าม provider ได้เนียนพอ) — เจอจริง
        # ตอนทดสอบว่า LLM ตอบ decision="reuse" มาแต่ลืมใส่ template_id มาด้วย ถ้ามี
        # candidate แค่ตัวเดียวไม่มีความกำกวม เดาแทนให้ได้อย่างปลอดภัย แต่ถ้ามีหลายตัวและ
        # ไม่ระบุมาเลย ไม่มีทางรู้ว่าหมายถึงตัวไหน ปลอดภัยกว่าที่จะ plan_fresh แทนการเดา
        if chosen_decision in ("reuse", "adapt") and not template_id:
            if len(candidates) == 1:
                template_id = candidates[0].get("template_id")
            else:
                return {
                    **_PROCEDURAL_PLANNER_SAFE_DEFAULT, "confidence": confidence,
                    "reason": "LLM chose reuse/adapt but did not specify which template_id (ambiguous with multiple candidates)",
                }

        return {
            "decision": chosen_decision,
            "template_id": template_id,
            "confidence": confidence,
            "slot_values": decision.get("slot_values") or {},
            "patch": decision.get("patch"),
            "reason": decision.get("reason", ""),
        }
    except Exception as e:
        print(f"⚠️ plan_with_procedural_memory error: {e}", flush=True)
        return dict(_PROCEDURAL_PLANNER_SAFE_DEFAULT)


# --- W_procmem: Repair tool (llm.repair_step()) — single-shot call ต่างหาก เรียกจาก
# core/fastpath_executor.py เฉพาะตอน step ของ template ที่กำลัง replay ล้มเหลว (resolve
# locator ไม่ได้/dispatch พัง/verify ไม่ผ่าน) — เป้าหมายคือแก้ step "เดียว" ให้ยังทำ
# sub-goal เดิมสำเร็จบนหน้าเว็บปัจจุบัน ไม่ใช่ plan ใหม่ทั้งชุด (นั่นคือหน้าที่ของ
# "replan" — escalate กลับไปให้ orchestrator.run_task() เต็มรูปแบบแทน)
_REPAIR_STEP_PARAMS = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["click", "fill", "select", "check", "press_key", "hover", "replan"],
            "description": "\"replan\" ถ้า element ที่ต้องการไม่มีอยู่บนหน้านี้จริงๆ (ไม่ใช่แค่ locator เดิมใช้ไม่ได้)",
        },
        "target": _ABSTRACTOR_TARGET_SCHEMA,
        "value": {
            "type": "string",
            "description": "ค่าที่จะกรอก/เลือก — ต้องเป็น {{slot_name}} เดิมจาก FAILED_STEP เท่านั้น ห้ามเปลี่ยนว่าข้อมูลไหนไปช่องไหน",
        },
        "widget": {
            "type": "string",
            "description": "ระบุถ้าต้องใช้ custom widget handling (เช่น 'vue_dropdown' สำหรับ dropdown ที่ไม่ใช่ native <select>: click เปิดก่อน แล้วค่อย click ตัวเลือก)",
        },
        "slot": {
            "type": "string",
            "description": "ชื่อ slot เดิมที่ step นี้อ้างอิง (ต้องตรงกับ FAILED_STEP เป๊ะ ไม่เปลี่ยน — ว่างเปล่าถ้า step เดิมไม่มี slot เช่น click เฉยๆ)",
        },
    },
    "required": ["action"],
}
_REPAIR_STEP_DESC = "แก้ template step หนึ่งที่ล้มเหลวระหว่าง replay ให้ยังทำ sub-goal เดิมสำเร็จบนหน้าเว็บปัจจุบัน หรือส่งสัญญาณ replan ถ้าทำไม่ได้จริง"
REPAIR_STEP_TOOL = {"name": "emit_repaired_step", "description": _REPAIR_STEP_DESC, "input_schema": _REPAIR_STEP_PARAMS}
_GROQ_REPAIR_STEP_TOOLS = [
    {"type": "function", "function": {"name": "emit_repaired_step", "description": _REPAIR_STEP_DESC, "parameters": _REPAIR_STEP_PARAMS}},
]
_GEMINI_REPAIR_STEP_TOOLS = [
    {"function_declarations": [{"name": "emit_repaired_step", "description": _REPAIR_STEP_DESC, "parameters": _REPAIR_STEP_PARAMS}]},
]

_REPAIR_STEP_SYSTEM_PROMPT = (
    "You are the Repair module. One template step failed during execution.\n"
    "Given the FAILED_STEP, the ERROR, and the CURRENT_PAGE, produce a\n"
    "corrected SINGLE step that achieves the same sub-goal on THIS page.\n\n"
    "RULES\n"
    "- Keep the same intent and the SAME slot (do not change which data\n"
    "  goes where — copy the \"slot\" field from FAILED_STEP verbatim if present).\n"
    "- Prefer robust locators (role+name, label, data-testid) over long CSS.\n"
    "- For custom dropdowns/menus that are not a native <select> (e.g. Vue/React\n"
    "  component libraries), emit widget:\"vue_dropdown\" (click to open, then\n"
    "  click the option) instead of assuming a native select.\n"
    "- If the element truly isn't on the page at all, return action:\"replan\" to\n"
    "  hand back to the full planner — do not guess wildly.\n"
    "- Output ONLY the emit_repaired_step tool call. No prose."
)

REPLAN_SIGNAL: dict = {"action": "replan"}


async def repair_step(
    client, model: str, failed_step: dict, error: str, current_page_text: str, provider: str,
) -> dict:
    """W_procmem: แก้ template step เดียวที่ล้มเหลวระหว่าง fast-path replay (ดู
    core/fastpath_executor.py) — เรียกเฉพาะตอน resolve_locator()/dispatch/verify ของ
    step นั้นไม่ผ่าน ไม่ใช่ทุก step

    **สำคัญ**: ผู้เรียก (fastpath_executor.py) ต้อง mask ค่าจริงของ step ที่
    sensitive=True ออกจาก failed_step ก่อนส่งเข้าฟังก์ชันนี้เสมอ (เช่นแทนด้วย
    "••••••") — ฟังก์ชันนี้เองไม่ mask ให้ ป้องกันไม่ให้ raw secret (รหัสผ่าน) เข้าไปใน
    LLM prompt โดยไม่จำเป็น

    ห้าม throw ออกไปเด็ดขาดไม่ว่ากรณีใด (provider error/parse ผิดพลาด/ไม่เรียก tool
    กลับมา) — คืน REPLAN_SIGNAL ({"action": "replan"}) แทนเสมอ ซึ่งเป็นค่าที่ปลอดภัย
    ที่สุดอยู่แล้ว (escalate กลับไปให้ slow-path loop เต็มรูปแบบจัดการต่อ แทนที่จะเสี่ยง
    ทำ action ผิดๆ ต่อ)"""
    prompt = (
        f"FAILED_STEP: {json.dumps(failed_step, ensure_ascii=False)}\n"
        f"ERROR: {error}\n"
        f"CURRENT_PAGE:\n{current_page_text}\n\n"
        "Call emit_repaired_step now with the corrected step (or replan)."
    )
    try:
        if provider == "anthropic":
            response = await client.messages.create(
                model=model,
                max_tokens=1024,
                system=_REPAIR_STEP_SYSTEM_PROMPT,
                tools=[REPAIR_STEP_TOOL],
                tool_choice={"type": "tool", "name": "emit_repaired_step"},
                messages=[{"role": "user", "content": prompt}],
            )
            tool_use = next((b for b in response.content if b.type == "tool_use"), None)
            result = tool_use.input if tool_use is not None else None
        elif provider == "groq":
            response = await client.chat.completions.create(
                model=model,
                max_tokens=1024,
                messages=[
                    {"role": "system", "content": _REPAIR_STEP_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                tools=_GROQ_REPAIR_STEP_TOOLS,
                tool_choice={"type": "function", "function": {"name": "emit_repaired_step"}},
            )
            tool_calls = response.choices[0].message.tool_calls or []
            result = json.loads(tool_calls[0].function.arguments) if tool_calls else None
        elif provider == "gemini":
            gemini_model = client.GenerativeModel(
                model_name=model,
                tools=_GEMINI_REPAIR_STEP_TOOLS,
                tool_config={"function_calling_config": {"mode": "ANY"}},
                system_instruction=_REPAIR_STEP_SYSTEM_PROMPT,
            )
            response = await gemini_model.generate_content_async(
                contents=[{"role": "user", "parts": [{"text": prompt}]}],
            )
            result = None
            for part in response.candidates[0].content.parts:
                fc = getattr(part, "function_call", None)
                if fc and fc.name == "emit_repaired_step":
                    result = _gemini_struct_to_plain_python(fc.args)
                    break
        else:
            result = None

        return result if result is not None else dict(REPLAN_SIGNAL)
    except Exception as e:
        print(f"⚠️ repair_step error: {e}", flush=True)
        return dict(REPLAN_SIGNAL)


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
    current_url: str = "",
    action_history_context: str = "",
    plan_context: str = "",
    *,
    verification_context: str = "",
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

    current_url (W30): page.url จริงตอน perceive step นี้ (orchestrator.py ดึงมาให้ทุก
    step เหมือน page_text — ดู _build_user_turn_text() สำหรับเหตุผลที่เพิ่ม)

    action_history_context (W32): action ล่าสุดไม่กี่ step (ทั้งสำเร็จและล้มเหลว) จาก
    ShortTermMemory.recent_actions_summary() — ต่างจาก memory_context ที่กรองเฉพาะ fail

    plan_context (W43): แผนที่ user ยืนยันแล้ว (เลขข้อ "1. ... 2. ...") ถ้า task นี้ผ่าน
    Confirm plan มา — ว่างเปล่าถ้าเป็น ad-hoc task ไม่มีแผนเลย ดู _build_user_turn_text()

    verification_context (W50): สัญญาณเสริมจากโค้ดว่า action ก่อนหน้าอาจไม่มีผลจริงกับ
    หน้าเว็บแม้จะคืน [OK] — orchestrator.py คำนวณให้ทุก step ดู _build_user_turn_text()
    """
    messages = messages + [
        {
            "role": "user",
            "content": _build_user_turn_text(
                goal, page_text, manual_context, memory_context, long_term_context, vision_context,
                site_manual_context, current_url, action_history_context, plan_context,
                verification_context,
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
    current_url: str = "",
    action_history_context: str = "",
    plan_context: str = "",
    *,
    verification_context: str = "",
) -> tuple[str, dict[str, Any], str, list[dict], TokenUsage]:
    """เหมือน next_action() แต่ยิงผ่าน Groq (OpenAI-compatible chat.completions + function calling)
    ใช้ทดสอบ agent loop ตอนยังไม่มี Anthropic key จริง

    Llama บางครั้งตอบเป็นข้อความเฉยๆ โดยไม่เรียก tool เลย แม้ tool_choice="required" —
    กรณีนี้ไม่ finish_task ทันที แต่เตือนให้เรียก tool แล้วลองใหม่สูงสุด
    _GROQ_NO_TOOL_CALL_RETRIES ครั้ง ก่อนจะ fallback เป็น finish_task(success=False)

    usage ที่คืนกลับ คือผลรวม token ของทุก request ที่ยิงจริง (รวม retry ที่สำเร็จด้วย)
    ไม่นับ request ที่ throw ก่อนได้ response กลับมา (เช่น tool_use_failed)

    manual_context/memory_context/long_term_context/vision_context/current_url/
    action_history_context/plan_context: ดู next_action() — เหมือนกัน (vision_context
    จะเป็น "" เสมอในทางปฏิบัติ เพราะ vision fallback ปัจจุบัน scope แค่ provider=gemini)
    """
    if not messages:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    messages = messages + [
        {
            "role": "user",
            "content": _build_user_turn_text(
                goal, page_text, manual_context, memory_context, long_term_context, vision_context,
                site_manual_context, current_url, action_history_context, plan_context,
                verification_context,
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


def _gemini_struct_to_plain_python(value: Any) -> Any:
    """W_procmem: Gemini function_call().args คืน protobuf Struct/ListValue
    (MapComposite/RepeatedComposite จาก proto-plus) ที่มี nested composite ซ้อนอยู่ลึกๆ
    เสมอ ไม่ใช่แค่ชั้นบนสุด — dict()/list() ตรงๆ (แบบที่ next_action_gemini() ใช้กับ
    _BROWSER_ACTION_PARAMS ที่เป็น flat schema เดียว พอแปลงชั้นเดียว) แปลงได้แค่ชั้นบนสุด
    ไม่พอสำหรับ ABSTRACTOR_TOOL ที่มี array ซ้อน (steps/slots) — json.dumps() ของ
    RepeatedComposite ที่หลงเหลืออยู่ข้างในจะพัง ("Object of type RepeatedComposite is
    not JSON serializable", เจอบั๊กจริงตอนทดสอบ) ฟังก์ชันนี้ไล่แปลงทุกชั้น recursively
    ด้วย duck-typing (เช็ค .items() ก่อนสำหรับ mapping, แล้วค่อยเช็ค iterable สำหรับ
    sequence) แทนที่จะ import internal type ของ proto-plus ตรงๆ (เปราะบางกว่าข้าม
    เวอร์ชัน library)"""
    if hasattr(value, "items"):
        return {k: _gemini_struct_to_plain_python(v) for k, v in value.items()}
    if isinstance(value, (str, bytes)):
        return value
    if isinstance(value, (list, tuple)) or hasattr(value, "__iter__"):
        return [_gemini_struct_to_plain_python(v) for v in value]
    return value


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
    current_url: str = "",
    action_history_context: str = "",
    plan_context: str = "",
    *,
    verification_context: str = "",
) -> tuple[str, dict[str, Any], str, list, TokenUsage]:
    """เหมือน next_action() แต่ยิงผ่าน Gemini (google-generativeai function calling)

    messages เก็บ Content ของ Gemini เอง (dict {"role": ..., "parts": [...]} หรือ
    Content proto ที่ SDK คืนมาตรงๆ ก็ใส่ต่อ list ได้เลย) — คนละ shape กับ
    Anthropic/Groq แต่ orchestrator.py ไม่แคร์ เพราะแค่ถือ opaque state ส่งเข้า-ออก

    tool_use_id ที่คืนกลับ คือชื่อ function ("browser_action"/"finish_task") ไม่ใช่ id
    จริงแบบ Anthropic/Groq เพราะ Gemini SDK เวอร์ชันนี้ไม่มี call id ให้ — ใช้เป็น "name"
    ที่ append_tool_result_gemini() ต้องผูก function_response กลับด้วย

    manual_context/memory_context/long_term_context/vision_context/current_url/
    action_history_context/plan_context: ดู next_action() — เหมือนกัน (vision_context
    (W9[A]) จะมีค่าจริงเฉพาะ provider นี้ — orchestrator.py ยิง vision fallback
    (llm.describe_screenshot()) scope แค่ Gemini เท่านั้นตอนนี้)
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
                    site_manual_context, current_url, action_history_context, plan_context,
                    verification_context,
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
# W43: บังคับ format เลขข้อ "1. ... \n2. ..." ตรงๆ (เดิมขอแค่ "bullet สั้นๆ" ซึ่งไม่ได้
# การันตี format นี้จริงจัง — โมเดลบังเอิญมักตอบแบบเลขข้อเองอยู่แล้วในทางปฏิบัติ แต่ไม่ใช่
# สัญญาที่บังคับได้) จำเป็นเพราะตอนนี้ frontend (index.html) ต้อง parse plan text นี้เป็น
# step แยกทีละข้อเพื่อ render เป็น checklist ที่ติ๊กได้ real-time ระหว่าง task รันจริง (ดู
# orchestrator.py::completed_plan_step) ถ้า format ไม่ตรง parsing จะแมตช์ index ผิดข้อ
_PLAN_PROMPT_TEMPLATE = (
    "Goal: {goal}\n\nหน้าเว็บเริ่มต้นที่เห็นตอนนี้:\n{page_text}\n\n"
    "เขียนแผนคร่าวๆ ว่าจะทำ goal นี้ให้สำเร็จด้วยขั้นตอนอะไรบ้าง (ไม่เกิน 5-6 ข้อ) — สรุป"
    "ระดับสูงพอให้ user อ่านแล้วเข้าใจและตัดสินใจอนุมัติได้ ไม่ต้องเรียก tool ไม่ต้องระบุ "
    "index ของ element เป๊ะๆ ตอบเป็นข้อความธรรมดา ไม่ต้องมี markdown\n\n"
    "*** ต้องตอบเป็นรายการเลขข้อเท่านั้น แต่ละข้อขึ้นต้นด้วยเลข ตามด้วยจุด แล้วเว้นวรรค "
    "เช่น '1. ค้นหาปุ่ม Login แล้วคลิก' บนบรรทัดของตัวเอง ห้ามใช้ bullet แบบอื่น (-, •, ก., "
    "ก) ฯลฯ) เด็ดขาด และห้ามมีข้อความอื่นก่อน/หลังรายการเลขข้อเลย เพราะระบบจะ parse แต่ละ"
    "บรรทัดเป็น step แยกเพื่อโชว์ความคืบหน้าให้ user เห็นทีละข้อระหว่างทำงานจริง ***"
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


# --- Intent Classification & Page Summarization ---
_CLASSIFY_INTENT_PROMPT = """วิเคราะห์ความต้องการ (Intent) ของผู้ใช้จากคำขอ (User Goal/Question) ด้านล่างนี้:
- ตอบว่า "qa_summary" หากผู้ใช้ต้องการถามคำถาม, สรุปเนื้อหา, อ่านข้อมูล, แปลความหมาย, สอบถามราคา/รายละเอียด, สอบถามสินค้า/ข้อมูล หรือประมวลผลข้อมูลจากหน้าเว็บ โดยไม่ต้องการให้ทำการคลิก/กรอกฟอร์ม/นำทาง
- ตอบว่า "action_task" หากผู้ใช้สั่งให้เบราว์เซอร์ทำ Action หรือกระบวนการใดๆ บนหน้าเว็บ เช่น คลิกปุ่ม, กรอกฟอร์ม, ค้นหา, สั่งซื้อสินค้า, ล็อกอิน, นำทางไปหน้าอื่น

User Goal/Question: {goal}
Page Content (ย่อ): {page_text_short}

ตอบเพียงคำเดียวเท่านั้น: qa_summary หรือ action_task"""


async def classify_intent(client, model: str, goal: str, page_text: str = "", provider: str = "gemini") -> str:
    """วิเคราะห์ Intent ของผู้ใช้ว่าเป็น qa_summary (การถามตอบ/ขอสรุปเนื้อหา) หรือ action_task (การสั่งงาน/automation บนเว็บ)"""
    goal_lower = goal.lower().strip()

    # Action imperatives (สั่งให้เบราว์เซอร์กระทำ)
    action_keywords = [
        "คลิก", "click", "กด", "กรอก", "fill", "พิมพ์", "type", "ซื้อ", "buy", "submit",
        "login", "ล็อกอิน", "เข้าสู่ระบบ", "สมัคร", "register", "search", "ค้นหา",
        "select", "เลือก", "check", "uncheck", "scroll", "ไปที่", "goto", "go to",
        "ป้อน", "ใส่ข้อมูล", "สั่งซื้อ", "เพิ่มลงตะกร้า", "add to cart", "checkout"
    ]

    # Q&A & Summarization markers (ถาม/ขอสรุปข้อมูล)
    qa_keywords = [
        "สรุป", "คืออะไร", "หมายถึงอะไร", "ราคากี่บาท", "ราคาเท่าไหร่", "มีรายละเอียดอะไรบ้าง",
        "อ่าน", "แปล", "แปลภาษา", "ตอบคำถาม", "ช่วยอ่าน", "ย่อความ", "หน้านี้เกี่ยวกับอะไร",
        "มีสินค้าอะไรบ้าง", "สรุปข้อมูล", "บอกหน่อย", "มีอะไรบ้าง", "ใคร", "ที่ไหน", "เมื่อไหร่",
        "ทำไม", "อย่างไร", "เท่าไร", "กี่", "รายละเอียด", "รายละเอียดสินค้า", "รายละเอียดของ",
        "summarize", "explain", "what is", "how much", "tell me", "what does", "describe"
    ]

    has_action = any(kw in goal_lower for kw in action_keywords)
    has_qa = any(kw in goal_lower for kw in qa_keywords)

    # 1. Clear Intent via heuristics
    if has_qa and not has_action:
        return "qa_summary"
    if has_action and not has_qa:
        return "action_task"

    # 2. Priority heuristic when both or neither match
    # If starting with pure question phrase
    if any(goal_lower.startswith(kw) for kw in ["สรุป", "หน้านี้", "คืออะไร", "มีอะไร", "ราคา", "แปล", "what", "how", "tell"]):
        if not any(goal_lower.startswith(kw) for kw in ["คลิก", "กด", "กรอก", "ค้นหา", "ไปที่", "click", "fill"]):
            return "qa_summary"

    # 3. LLM classification fallback for ambiguous/conversational cases
    try:
        page_text_short = page_text[:500] if page_text else ""
        prompt = _CLASSIFY_INTENT_PROMPT.format(goal=goal, page_text_short=page_text_short)
        result = await generate_text(client, model, prompt, provider)
        result_clean = result.strip().lower()
        if "qa_summary" in result_clean or "qa" in result_clean or "summary" in result_clean:
            return "qa_summary"
        return "action_task"
    except Exception as e:
        print(f"⚠️ classify_intent error ({e}) — fallback to action_task", flush=True)
        return "action_task"


_SUMMARIZE_SYSTEM_PROMPT = (
    "คุณคือ AI Assistant ที่มีความสามารถในการอ่านหน้าเว็บ ปัจจุบันผู้ใช้อยู่ที่หน้าเว็บนี้ และต้องการถามคำถามหรือขอสรุปข้อมูล\n"
    "โปรดอ่านเนื้อหาเว็บต่อไปนี้แล้วตอบคำถามของผู้ใช้ให้กระชับ เข้าใจง่าย และใช้ภาษาไทยที่เป็นกันเอง"
)


async def summarize_page(client, model: str, page_text: str, user_prompt: str, provider: str = "gemini") -> str:
    """สรุปเนื้อหาหน้าเว็บหรือตอบคำถามตาม Prompt รูปแบบเฉพาะที่กำหนดให้ออกมาเป็นภาษาไทยอย่างเป็นธรรมชาติ"""
    full_prompt = f"{_SUMMARIZE_SYSTEM_PROMPT}\n\nPage Content: {page_text}\n\nUser Question: {user_prompt}"
    try:
        return await generate_text(client, model, full_prompt, provider)
    except Exception as e:
        print(f"⚠️ summarize_page error: {e}", flush=True)
        return f"ขออภัยด้วยครับ ไม่สามารถสรุปข้อมูลจากหน้าเว็บได้ในขณะนี้เนื่องจากเกิดข้อผิดพลาด: {e}"



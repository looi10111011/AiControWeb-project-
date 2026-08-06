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
import base64
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
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
  เงียบๆ ปัดเป็นคำตอบที่มั่นใจ
  - ลิสต์ธรรมดาที่มีแค่ field เดียว (เช่น แค่รายชื่อคน ไม่มีข้อมูลอื่นประกอบต่อรายการ) ให้
    เรียงลำดับตามตัวอักษร (A-Z) ก่อนตอบเสมอเพื่อให้อ่านง่าย เว้นแต่ goal ระบุลำดับอื่นชัดเจน
    (เช่น "เรียงตามวันที่") — การเรียงลำดับใหม่ทำได้เฉพาะ "ลำดับที่แสดง" เท่านั้น ห้ามเปลี่ยน
    ตัวสะกด/เนื้อหาของแต่ละรายการระหว่างเรียงเด็ดขาด
  - W19 ("Table Data Extractor & Presenter"): ข้อมูลที่มีหลาย field ต่อแถว/รายการ (เช่น
    ตารางที่มี Username+Employee Name+Role+Status ในแถวเดียวกัน) ให้ยึดกฎตรงข้ามกับข้อบน —
    "ห้ามเรียงลำดับใหม่เด็ดขาด" รักษาลำดับแถวตามที่ปรากฏบนหน้าจอจริง (DOM order, บนลงล่าง)
    เสมอ ไม่ว่ากรณีใด เว้นแต่ user ขอให้เรียงแบบอื่นชัดเจนเท่านั้น — เหตุผล: การเรียงข้อมูล
    หลาย field ใหม่ (เช่น เรียง username ตาม A-Z) ทำให้ผู้ใช้เทียบคำตอบกับสิ่งที่เห็นบนจอจริง
    ไม่ได้อีกต่อไป ผิดจุดประสงค์ของการ "แสดงข้อมูลตามที่ปรากฏจริง" ไปเลย
  - ห้ามแยก field ของแถว/รายการเดียวกันออกจากกันเป็นคนละลิสต์เด็ดขาด (เช่น แยก username
    ทั้งหมดไว้ลิสต์หนึ่ง แล้วแยก employee name ไว้อีกลิสต์หนึ่งต่างหาก) — แต่ละแถวต้องนำเสนอ
    เป็นก้อนข้อมูลเดียว (1 atomic object ต่อแถว) เสมอ เช่น:
      1. Admin (Employee: Surya king, Role: Admin)
      2. AutoUser_2335 (Employee: Manoj B, Role: Admin)
      3. ayush123 (Employee: Ayush Saha, Role: Admin)
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
- W19 ("Scoped Search Context"): ถ้า indexed elements มี label ซ้ำกันหลายตัว (เช่น
  "Search" โผล่ทั้งใน sidebar เมนูหลักและในฟอร์ม/ตัวกรองของเนื้อหาหลัก) ให้สังเกตว่า element
  ไหนมี marker "(navigation)" ต่อท้าย (แปลว่าอยู่ใน sidebar/menu/nav) — ถ้า goal ต้องการ
  กรอกฟอร์ม/ค้นหาข้อมูล/ทำงานกับเนื้อหาหลักของหน้า ให้เลือก element ที่ "ไม่มี" marker นี้
  (อยู่ใน main content) เสมอ ใช้ตัวที่มี "(navigation)" เฉพาะตอน goal ตั้งใจจะเปิดเมนู/
  นำทางผ่าน sidebar จริงๆ เท่านั้น
- W19 ("Exact Element Matching"): เลือก index จาก label ที่สื่อความหมายจริง (ชื่อ
  field/ปุ่มที่มองเห็น เช่น "Employee Name", "User Role") ไม่ใช่จำเลข index จาก step
  ก่อนหน้า — index เปลี่ยนใหม่ทุกครั้งที่ perceive จริง ห้ามสมมติว่า index เดิมยังชี้ไปที่
  element เดิมข้าม step เด็ดขาด ต้องอ่าน indexed elements ล่าสุดที่แนบมาทุกครั้งเสมอ
- W19 (Autocomplete field เช่น "Employee Name" บน OrangeHRM): ห้าม fill ข้อความลงช่อง
  autocomplete แล้วถือว่าจบเลย — ต้อง (1) fill ข้อความค้นหาลงช่องก่อน (2) รอ/perceive หน้า
  ใหม่ให้เห็นตัวเลือกที่ popup ขึ้นมา (มักเป็น element role=option/menuitem ใหม่ในลิสต์) แล้ว
  (3) click ตัวเลือกแรกที่ตรงจากลิสต์ popup นั้น การ fill เฉยๆ โดยไม่คลิกเลือกจาก popup มักไม่
  ถูกฟอร์มยอมรับจริง แม้ข้อความจะแสดงอยู่ในช่องแล้วก็ตาม
- W19 ("Autocomplete Disambiguation", ต่างจากข้อบน): ถ้าตั้งใจ "กด Enter เพื่อค้นหา" (เช่น
  ช่องค้นหา YouTube/Google ที่ไม่ใช่ autocomplete ที่ต้องเลือกจาก popup) ให้เลือก
  type="press_key" key="Enter" ที่ index ของ "ช่อง input เดิม" ที่เพิ่ง fill ไปตรงๆ เท่านั้น
  ห้ามสับสนไปเลือก index ของ suggestion/option ที่โผล่ขึ้นมาใน popup โดยไม่ตั้งใจ (จะกลาย
  เป็นเลือก suggestion นั้นแทนการค้นหาคำที่พิมพ์จริง) — ยกเว้นตั้งใจจะเลือก suggestion นั้น
  จริงๆ (ตามข้อ autocomplete field ด้านบน) จึงค่อย click ที่ index ของ suggestion แทน
- W19 ("Task Completion Verifier"): ก่อนเรียก finish_task(success=true) ให้ตรวจสอบ
  indexed elements/ข้อความบนหน้าปัจจุบันว่ามีข้อความ error/validation โผล่อยู่ไหม (เช่น
  "Required", "Invalid", "Already Exists", หรือคำแปลไทย) ถ้ามี แปลว่า step ที่ทำไปยังไม่
  สำเร็จจริง ห้ามเรียก finish_task(success=true) ให้แก้ field ที่มีปัญหาก่อน — มองหา
  สัญญาณความสำเร็จจริง (navigate กลับไปหน้า list, toast/ข้อความ "Successfully Saved") แทน
  ก่อนยืนยันว่าสำเร็จ
- W19 ("Log Cleanliness"): element ที่มี marker "[active อยู่แล้ว]" ต่อท้าย label (เมนู/
  แท็บที่เลือก/active อยู่แล้ว) ห้ามคลิกซ้ำเด็ดขาด เพราะบาง framework ไม่ trigger การ
  เปลี่ยนแปลงอะไรเลยถ้าคลิกทับตัวเดิมที่ active อยู่แล้ว (โครงสร้างหน้าเหมือนเดิมทุก
  ประการ) ทำให้เสีย step ไปเปล่าๆ รอหน้าเปลี่ยนที่จะไม่มีวันเกิดขึ้น ให้ข้ามไปทำ action ถัดไป
  ที่เกี่ยวกับ goal บนหน้าปัจจุบันได้เลย (element นี้ "อยู่แล้ว" ตามที่ต้องการ ไม่ต้องกดซ้ำ) —
  ยกเว้น goal สั่งให้ "รีเฟรช"/"เปิดใหม่" ชัดเจนเท่านั้นถึงคลิกซ้ำได้
- W20 ("No Redundant Search Submission"): ตอนส่งคำค้นหา/คำกรองที่พิมพ์ไว้ในช่อง ให้เลือก
  วิธีเดียวเท่านั้นระหว่าง (ก) type: "press_key" key: "Enter" ที่ index ของช่อง input นั้น
  หรือ (ข) type: "click" ที่ปุ่ม "Search"/"ค้นหา" — ห้ามทำทั้งสองอย่างติดกันสำหรับคำค้นหา
  เดียวกันเด็ดขาด (ยิง Enter แล้วยังไปคลิกปุ่ม Search ซ้ำอีกที ถือเป็น submit ซ้ำซ้อนที่อาจ
  ค้นหาซ้ำ/รีเซ็ตผลลัพธ์เดิม) — หลังจากยิง press_key Enter แล้ว ให้ไปดูผลลัพธ์การค้นหาที่
  หน้าเว็บเปลี่ยนไปทันที ข้าม step "คลิกปุ่ม Search" ที่วางแผนไว้ก่อนหน้าไปเลยโดยอัตโนมัติ
- W20 ("Account Security & Password Actions", HIGHEST PRIORITY): goal ที่เกี่ยวกับ "เปลี่ยน
  รหัสผ่าน"/"แก้ไขข้อมูลโปรไฟล์ของฉัน"/"ตั้งค่าความปลอดภัย" ของ user ที่ login อยู่ปัจจุบัน —
  ห้ามคลิกเมนู "My Info" ในแถบเมนูหลัก (sidebar) เด็ดขาด (เมนูนี้มักเป็นข้อมูล directory ของ
  พนักงาน ไม่ใช่การตั้งค่าบัญชีผู้ใช้ระบบ) ให้ทำตามลำดับนี้เสมอแทน: (1) คลิก element ที่เป็น
  User Dropdown/Profile Menu มุมขวาบนของหน้าเว็บ (มักโชว์ avatar/ชื่อผู้ใช้ที่ login อยู่) (2)
  รอให้ dropdown menu แสดงผล แล้วดู indexed elements ใหม่ (3) คลิก "Change Password" หรือ
  "Profile Settings" จากตัวเลือกที่โผล่มาในนั้น — sidebar menu ใช้สำหรับ navigation ทั่วไป
  เท่านั้น ส่วน dropdown มุมขวาบนใช้สำหรับการตั้งค่าที่ผูกกับ session/user คนนี้โดยเฉพาะ ถ้าเผลอ
  ลองเส้นทาง "My Info" ไปแล้วไม่เจอฟังก์ชันเปลี่ยนรหัสผ่านที่ต้องการ ให้รับรู้ทันทีว่าผิดทาง
  แล้ว fallback ไปทำตามลำดับ mandatory protocol นี้แทน ห้ามวนกลับไปลองเส้นทางเดิมที่ล้มเหลว
  ซ้ำอีก
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
        "message": {"type": "string", "description": "สรุปผลสั้นๆ ว่าทำอะไรไป/ทำไมหยุด — รายชื่อ/ข้อมูลที่ดึงมาต้องคัดลอกตัวสะกดตรงตามต้นฉบับ ห้ามเดา/แก้สะกด — ลิสต์ field เดียวเรียง A-Z ก่อนตอบ, ตารางหลาย field ต่อแถวห้ามเรียงใหม่เด็ดขาด (รักษา DOM order) และห้ามแยก field ของแถวเดียวกันออกจากกัน (ดู W_listformat ใน system prompt)"},
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


# --- W19 (ดู W19.txt ข้อ 8 "Semantic Redundancy Evaluator"): single-shot call ต่างหาก
# ประเมินว่า proposed action ที่ next_action() เพิ่งเลือก "มีประโยชน์จริง" ต่อ goal ไหม
# หรือเป็นแค่ side-step ที่ไม่จำเป็น (เช่น scroll/อ่านข้อมูลที่ไม่เกี่ยว ทั้งที่ปุ่มที่ต้อง
# กดอยู่ตรงหน้าแล้ว) — เรียกจาก orchestrator.py ก่อน dispatch จริงทุก step **เฉพาะตอน
# settings.enable_semantic_redundancy_check เปิดอยู่เท่านั้น** (ปิดไว้ default เหมือน
# enable_procedural_memory — เพิ่ม LLM call ต่อ step 1 ครั้ง มีต้นทุน latency/token จริง
# ต้อง validate คุณภาพก่อนเปิดเป็น default)
#
# ต่างจาก state_filter.py (W19 ข้อ 6) ตรงที่ตัวนั้นเช็ค "สถานะ DOM" แบบ deterministic
# (ไม่พึ่ง LLM เลย, เร็ว, แม่นยำ 100% แต่ตอบได้แค่คำถามแคบๆ เช่น "ค่าซ้ำไหม") ส่วนตัวนี้เช็ค
# "เจตนา" เทียบกับ goal ทั้ง task (ต้องใช้ LLM ตัดสิน ไม่มีทาง deterministic ได้จริง) — สอง
# ชั้นทำงานคนละจุด ไม่ทับซ้อนกัน
_SEMANTIC_REDUNDANCY_PARAMS = {
    "type": "object",
    "properties": {
        "is_semantically_redundant": {
            "type": "boolean",
            "description": "true ถ้า action นี้ไม่ทำให้ USER_GOAL คืบหน้าเลย (side-step ที่ข้ามได้)",
        },
        "value_score": {
            "type": "number",
            "description": "0.0-1.0 — ความเกี่ยวข้องของ action นี้กับ USER_GOAL (1.0 = จำเป็นมาก, 0.0 = ไม่เกี่ยวเลย)",
        },
        "action_decision": {
            "type": "string",
            "enum": ["PASS", "SKIP_STEP", "FORCE_REPLAN"],
            "description": (
                "PASS = ปล่อยให้ dispatch ตามปกติ (ค่า default เมื่อไม่แน่ใจ). "
                "SKIP_STEP = action นี้เจาะจงไม่มีประโยชน์ ข้าม step นี้ไปเลือก action อื่นแทน. "
                "FORCE_REPLAN = ทั้งแนวทางตอนนี้ดูหลงทางไปไกลจาก goal มาก ควรคิดแผนใหม่ทั้งหมด"
            ),
        },
        "reasoning": {"type": "string", "description": "เหตุผลสั้นๆ กระชับ 1-2 ประโยค"},
    },
    "required": ["is_semantically_redundant", "value_score", "action_decision", "reasoning"],
}
_SEMANTIC_REDUNDANCY_DESC = "ประเมินว่า proposed action ทำให้ USER_GOAL คืบหน้าจริงไหม หรือเป็น side-step ที่ข้ามได้"
SEMANTIC_REDUNDANCY_TOOL = {
    "name": "evaluate_action_value", "description": _SEMANTIC_REDUNDANCY_DESC, "input_schema": _SEMANTIC_REDUNDANCY_PARAMS,
}
_GROQ_SEMANTIC_REDUNDANCY_TOOLS = [
    {"type": "function", "function": {"name": "evaluate_action_value", "description": _SEMANTIC_REDUNDANCY_DESC, "parameters": _SEMANTIC_REDUNDANCY_PARAMS}},
]
_GEMINI_SEMANTIC_REDUNDANCY_TOOLS = [
    {"function_declarations": [{"name": "evaluate_action_value", "description": _SEMANTIC_REDUNDANCY_DESC, "parameters": _SEMANTIC_REDUNDANCY_PARAMS}]},
]

_SEMANTIC_REDUNDANCY_SYSTEM_PROMPT = (
    "You are a Semantic Redundancy Evaluator for a browser automation agent.\n"
    "You review ONE proposed action right before it is dispatched — you do not\n"
    "plan, you only judge whether THIS SPECIFIC action moves the agent closer\n"
    "to USER_GOAL.\n\n"
    "RULES\n"
    "- Default to PASS whenever the action is plausibly useful — you are a\n"
    "  cheap sanity check, not the planner. Being wrong and blocking a useful\n"
    "  action is worse than letting a mildly wasteful one through.\n"
    "- Only choose SKIP_STEP when the action is CLEARLY unrelated to the goal\n"
    "  (e.g. reading unrelated footer text, scrolling with no target in mind,\n"
    "  re-navigating to a page already open) while a more direct path is\n"
    "  visible in TARGET_CONTEXT/STEP_SUMMARY.\n"
    "- Only choose FORCE_REPLAN when the entire recent direction looks lost\n"
    "  (not just this one action) — this is rare, reserve it for clear cases.\n"
    "- Output ONLY the evaluate_action_value tool call. No prose."
)

_SEMANTIC_REDUNDANCY_SAFE_DEFAULT: dict[str, Any] = {
    "is_semantically_redundant": False,
    "value_score": 1.0,
    "action_decision": "PASS",
    "reasoning": "evaluator error/uncertain — ไม่บล็อกความคืบหน้า (fail-open)",
}


async def evaluate_semantic_redundancy(
    client, model: str, goal: str, step_summary: str, page_title: str, target_context: str,
    tool_name: str, tool_input: dict, provider: str,
) -> dict:
    """เรียก 1 ครั้งต่อ step (เฉพาะตอน settings.enable_semantic_redundancy_check เปิด) —
    ประเมิน proposed action (tool_name/tool_input ที่ next_action() เพิ่งเลือกมา) เทียบ
    กับ goal ทั้ง task

    ห้าม throw ออกไปให้ orchestrator loop พังเด็ดขาดไม่ว่ากรณีใด (provider error/parse
    ผิดพลาด/ไม่เรียก tool กลับมา) — คืน _SEMANTIC_REDUNDANCY_SAFE_DEFAULT (action_decision
    PASS) แทนเสมอ เป็นค่าที่ปลอดภัยที่สุด (ปล่อยให้ dispatch ตามปกติเหมือนไม่มี evaluator
    นี้อยู่เลย ดีกว่าเสี่ยง block action ที่จริงๆ มีประโยชน์เพราะ evaluator เองพัง)"""
    prompt = (
        f"USER_GOAL: {goal}\n"
        f"STEP_SUMMARY: {step_summary}\n"
        f"CURRENT_PAGE: {page_title}\n"
        f"TARGET_CONTEXT: {target_context}\n"
        f"PROPOSED_ACTION: {tool_name} {json.dumps(tool_input, ensure_ascii=False)}\n\n"
        "Call evaluate_action_value now."
    )
    try:
        if provider == "anthropic":
            response = await client.messages.create(
                model=model,
                max_tokens=512,
                system=_SEMANTIC_REDUNDANCY_SYSTEM_PROMPT,
                tools=[SEMANTIC_REDUNDANCY_TOOL],
                tool_choice={"type": "tool", "name": "evaluate_action_value"},
                messages=[{"role": "user", "content": prompt}],
            )
            tool_use = next((b for b in response.content if b.type == "tool_use"), None)
            result = tool_use.input if tool_use is not None else None
        elif provider == "groq":
            response = await client.chat.completions.create(
                model=model,
                max_tokens=512,
                messages=[
                    {"role": "system", "content": _SEMANTIC_REDUNDANCY_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                tools=_GROQ_SEMANTIC_REDUNDANCY_TOOLS,
                tool_choice={"type": "function", "function": {"name": "evaluate_action_value"}},
            )
            tool_calls = response.choices[0].message.tool_calls or []
            result = json.loads(tool_calls[0].function.arguments) if tool_calls else None
        elif provider == "gemini":
            gemini_model = client.GenerativeModel(
                model_name=model,
                tools=_GEMINI_SEMANTIC_REDUNDANCY_TOOLS,
                tool_config={"function_calling_config": {"mode": "ANY"}},
                system_instruction=_SEMANTIC_REDUNDANCY_SYSTEM_PROMPT,
            )
            response = await gemini_model.generate_content_async(
                contents=[{"role": "user", "parts": [{"text": prompt}]}],
            )
            result = None
            for part in response.candidates[0].content.parts:
                fc = getattr(part, "function_call", None)
                if fc and fc.name == "evaluate_action_value":
                    result = _gemini_struct_to_plain_python(fc.args)
                    break
        else:
            result = None

        return result if result is not None else dict(_SEMANTIC_REDUNDANCY_SAFE_DEFAULT)
    except Exception as e:
        print(f"⚠️ evaluate_semantic_redundancy error: {e}", flush=True)
        return dict(_SEMANTIC_REDUNDANCY_SAFE_DEFAULT)


# --- W19-2: "Safety & Performance Middleware" — single-shot call ที่รวม redundancy check
# (เหมือน evaluate_semantic_redundancy ด้านบน) กับ permission check (เหมือน
# permission/rules.py::classify_action) เข้าเป็น 1 LLM call เดียว ประหยัด round-trip กว่า
# เรียกแยก 2 ครั้ง — เรียกจาก orchestrator.py ก่อน dispatch จริง **เฉพาะตอน
# settings.enable_middleware_evaluator เปิดอยู่เท่านั้น** (ปิดไว้ default เหมือนโมดูล LLM
# ตัวอื่นๆ ในไฟล์นี้ — ต้อง validate คุณภาพก่อนเปิดเป็น default)
#
# *** สำคัญ: เป็น "โมดูลที่ 4" แบบ additive ล้วนๆ ไม่ได้แทนที่/ลดทอนระบบความปลอดภัยเดิมเลย
# แม้แต่น้อย — permission/rules.py::classify_action() ยังคงเป็นผู้ตัดสินสุดท้ายเสมอ
# (final say) ต่อทุก action เหมือนเดิมทุกประการ ตัวนี้ทำได้แค่ "เพิ่มความระมัดระวัง"
# (escalate-only): ผลลัพธ์ risk_level=REQUIRES_CONSENT/BLOCKED จากตัวนี้ถูกส่งต่อเข้า
# classify_action() ผ่าน manual_guidance string เดียวกับที่ RAG คู่มือ (W7[B]) ใช้อยู่แล้ว
# (ต่อท้ายวลีที่ตรงกับ MANUAL_CONFIRMATION_KEYWORDS) ทำให้ classify_action() escalate เป็น
# NEEDS_CONFIRMATION ตามกลไกเดิมที่มีอยู่แล้ว/เทสต์ไว้แล้ว — ไม่มีทาง "ลดระดับ" ความเสี่ยงที่
# classify_action() ตัดสินไปแล้วได้เลย (risk_level=AUTO_APPROVE ของตัวนี้ = ไม่ต่อท้ายอะไร
# เข้า manual_guidance เลย = พฤติกรรมเดิมเป๊ะ) และไม่เรียก ask_user_func เองตรงๆ ด้วย —
# ปล่อยให้ execute()'s classify_action()/_confirm_action() (ที่ทดสอบไว้แล้ว) เป็นคนถามจริง
# กันการถามซ้ำสองครั้งสำหรับ action เดียวกัน
_MIDDLEWARE_PARAMS = {
    "type": "object",
    "properties": {
        "redundancy_evaluation": {
            "type": "object",
            "properties": {
                "is_redundant": {"type": "boolean", "description": "true ถ้า action นี้ทำซ้ำสถานะเดิม/เป็น side-step ที่ไม่เกี่ยวกับ USER_GOAL"},
                "redundancy_reason": {"type": "string", "description": "เหตุผลถ้า redundant, ไม่งั้นเว้นว่าง"},
            },
            "required": ["is_redundant", "redundancy_reason"],
        },
        "permission_evaluation": {
            "type": "object",
            "properties": {
                "risk_level": {
                    "type": "string", "enum": ["AUTO_APPROVE", "REQUIRES_CONSENT", "BLOCKED"],
                    "description": (
                        "REQUIRES_CONSENT เฉพาะ: financial (สั่งซื้อ/pay now/โอนเงิน/เพิ่มบัตร), "
                        "account security (เปลี่ยนรหัสผ่าน/ตั้งค่าความปลอดภัย/MFA), destructive "
                        "(ลบไฟล์/ยกเลิก subscription/purge repo/ล้างตะกร้า), PII (เลขบัตรประชาชน/"
                        "เงินเดือน/ข้อมูลสุขภาพ/รหัสผ่าน), download executable (.exe/.bat/.sh/.zip "
                        "จากโดเมนที่ไม่รู้จัก) เท่านั้น ที่เหลือ AUTO_APPROVE เสมอไม่ว่าเว็บไหน"
                    ),
                },
                "permission_reason": {"type": "string", "description": "เหตุผลของ risk_level นี้ อ้างอิงผลกระทบจริงของ action"},
            },
            "required": ["risk_level", "permission_reason"],
        },
        "final_action_decision": {
            "type": "string", "enum": ["EXECUTE", "SKIP_REDUNDANT", "PROMPT_USER_PERMISSION"],
        },
    },
    "required": ["redundancy_evaluation", "permission_evaluation", "final_action_decision"],
}
_MIDDLEWARE_DESC = (
    "ประเมิน proposed action ครั้งเดียวทั้ง redundancy (มีประโยชน์ต่อ goal ไหม) และ "
    "permission (ต้องขออนุมัติจาก user ก่อนไหม) — ใช้ได้ทุกเว็บ ไม่เจาะจงเว็บใดเว็บหนึ่ง"
)
MIDDLEWARE_EVALUATOR_TOOL = {
    "name": "middleware_evaluate", "description": _MIDDLEWARE_DESC, "input_schema": _MIDDLEWARE_PARAMS,
}
_GROQ_MIDDLEWARE_TOOLS = [
    {"type": "function", "function": {"name": "middleware_evaluate", "description": _MIDDLEWARE_DESC, "parameters": _MIDDLEWARE_PARAMS}},
]
_GEMINI_MIDDLEWARE_TOOLS = [
    {"function_declarations": [{"name": "middleware_evaluate", "description": _MIDDLEWARE_DESC, "parameters": _MIDDLEWARE_PARAMS}]},
]

_MIDDLEWARE_SYSTEM_PROMPT = (
    "You are the Safety & Performance Middleware for a Universal AI Browser Automation\n"
    "System. Evaluate proposed actions across ANY website (e-commerce, social media,\n"
    "cloud storage, banking, government, internal tools, etc.) to optimize speed and\n"
    "safety. Perform a SINGLE-PASS EVALUATION of ONE proposed action.\n\n"
    "REDUNDANCY RULES (all sites)\n"
    "- is_redundant=true if: re-typing the exact same value into an input field;\n"
    "  clicking/toggling an option already in the desired state; performing an\n"
    "  irrelevant side-action (social share links, footer nav, unrelated ads) that\n"
    "  does not move closer to USER_GOAL.\n"
    "- Default to is_redundant=false when plausibly useful — being wrong and\n"
    "  blocking a useful action is worse than letting a mildly wasteful one through.\n\n"
    "CONTEXT-AWARE PERMISSION RULES (generic risk assessment)\n"
    "- risk_level=AUTO_APPROVE for routine, non-destructive interactions: searching\n"
    "  (search bars, filter checkboxes, sorting dropdowns), reading/extracting\n"
    "  (next page, expanding accordions, scrolling), navigation (links, category\n"
    "  menus, tab switching), non-sensitive inputs (search queries, comments,\n"
    "  non-financial forms).\n"
    "- risk_level=REQUIRES_CONSENT ONLY for high-risk impact: financial (placing\n"
    "  orders, \"Pay Now\", transferring funds, adding credit cards); account\n"
    "  security (changing passwords, security settings, MFA); destructive\n"
    "  (deleting items, canceling subscriptions, purging files/repos, emptying\n"
    "  carts); sensitive PII (national ID, salary, health records, private\n"
    "  passwords); downloads of executables (.exe/.bat/.sh/.zip from unverified\n"
    "  domains).\n"
    "- risk_level=BLOCKED only for clearly malicious/irreversible-and-forbidden intent.\n\n"
    "Output ONLY the middleware_evaluate tool call. No prose."
)

_MIDDLEWARE_SAFE_DEFAULT: dict[str, Any] = {
    "redundancy_evaluation": {"is_redundant": False, "redundancy_reason": ""},
    "permission_evaluation": {
        "risk_level": "AUTO_APPROVE",
        "permission_reason": "evaluator error/uncertain — ไม่บล็อกความคืบหน้า (fail-open)",
    },
    "final_action_decision": "EXECUTE",
}


async def evaluate_safety_and_performance(
    client, model: str, goal: str, current_domain: str, action_type: str, element_description: str,
    action_value: str, provider: str,
) -> dict:
    """เรียก 1 ครั้งต่อ step (เฉพาะตอน settings.enable_middleware_evaluator เปิด) — รวม
    redundancy check + permission check เป็น LLM call เดียว (ดู module comment ด้านบน
    สำหรับเหตุผลที่เป็น "โมดูลที่ 4" แบบ additive ไม่แทนที่ classify_action())

    current_domain: โดเมนของเว็บปัจจุบัน (extract_domain(page.url)) — ใช้แค่บอกบริบทเว็บ
    ปัจจุบันให้ LLM เห็น ไม่ได้ผูก logic เฉพาะเว็บไหนเว็บหนึ่งเลย (generic ข้าม platform)

    ห้าม throw ออกไปให้ orchestrator loop พังเด็ดขาดไม่ว่ากรณีใด — คืน
    _MIDDLEWARE_SAFE_DEFAULT (EXECUTE/AUTO_APPROVE) แทนเสมอ เป็นค่าที่ปลอดภัยที่สุด
    (เหมือนไม่มี middleware นี้อยู่เลย — classify_action() ที่ dispatch จริงยังทำงานตามปกติ)"""
    prompt = (
        f"OVERALL_GOAL: {goal}\n"
        f"ACTIVE_SITE_DOMAIN: {current_domain}\n"
        f"ACTION_PROPOSED: {action_type} on target element {element_description} with value {action_value}\n\n"
        "Call middleware_evaluate now."
    )
    try:
        if provider == "anthropic":
            response = await client.messages.create(
                model=model,
                max_tokens=512,
                system=_MIDDLEWARE_SYSTEM_PROMPT,
                tools=[MIDDLEWARE_EVALUATOR_TOOL],
                tool_choice={"type": "tool", "name": "middleware_evaluate"},
                messages=[{"role": "user", "content": prompt}],
            )
            tool_use = next((b for b in response.content if b.type == "tool_use"), None)
            result = tool_use.input if tool_use is not None else None
        elif provider == "groq":
            response = await client.chat.completions.create(
                model=model,
                max_tokens=512,
                messages=[
                    {"role": "system", "content": _MIDDLEWARE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                tools=_GROQ_MIDDLEWARE_TOOLS,
                tool_choice={"type": "function", "function": {"name": "middleware_evaluate"}},
            )
            tool_calls = response.choices[0].message.tool_calls or []
            result = json.loads(tool_calls[0].function.arguments) if tool_calls else None
        elif provider == "gemini":
            gemini_model = client.GenerativeModel(
                model_name=model,
                tools=_GEMINI_MIDDLEWARE_TOOLS,
                tool_config={"function_calling_config": {"mode": "ANY"}},
                system_instruction=_MIDDLEWARE_SYSTEM_PROMPT,
            )
            response = await gemini_model.generate_content_async(
                contents=[{"role": "user", "parts": [{"text": prompt}]}],
            )
            result = None
            for part in response.candidates[0].content.parts:
                fc = getattr(part, "function_call", None)
                if fc and fc.name == "middleware_evaluate":
                    result = _gemini_struct_to_plain_python(fc.args)
                    break
        else:
            result = None

        return result if result is not None else dict(_MIDDLEWARE_SAFE_DEFAULT)
    except Exception as e:
        print(f"⚠️ evaluate_safety_and_performance error: {e}", flush=True)
        return dict(_MIDDLEWARE_SAFE_DEFAULT)


# --- W19-3: "Voice & Persona Interface" — แปลงสถานะ agent (ดิบๆ เช่น "[OK] click ->
# สำเร็จ") ให้เป็นข้อความไทยธรรมชาติแบบผู้ช่วยส่วนตัว ให้ Test Console UI โชว์แทน raw log
# (ดู module comment: ห้ามพูดแบบ system log "Status: Executing command click on selector
# #search-btn" ต้องพูดธรรมชาติแบบคนคุยกัน) — เรียกจาก orchestrator.py **เฉพาะตอน
# settings.enable_persona_voice เปิดอยู่เท่านั้น** (ปิดไว้ default เหมือนโมดูล LLM ตัวอื่น)
#
# ต่างจาก 3 โมดูลก่อนหน้า (state_filter/semantic_redundancy/middleware) ตรงที่ตัวนี้ไม่มีผล
# ต่อ control flow ของ agent loop เลยแม้แต่น้อย (ไม่ skip/ไม่ block/ไม่ replan อะไรทั้งสิ้น) —
# เป็นแค่ "ชั้นการสื่อสาร" (presentation layer) ล้วนๆ คืนข้อความเสริมให้ UI แสดงคู่กับ raw
# log เดิม (ไม่ได้แทนที่ — raw log/history ยังคงส่งครบเหมือนเดิมทุกประการ เผื่อ debug จริง)
#
# ความถี่การเรียก: ไม่ได้เรียกทุก step ของ browser action (จะแพงเกินไปโดยไม่จำเป็น เพราะเป็น
# แค่ข้อความคุย ไม่ใช่การตัดสินใจที่กระทบผลลัพธ์) — เรียกเฉพาะจังหวะสำคัญที่ user จะได้เห็น/
# สนใจจริงๆ ตามที่ RULES ระบุ: PROGRESS (เริ่ม task), PERMISSION (ขออนุมัติ), ERROR (action
# ล้มเหลว), COMPLETION (จบ task) — ดู orchestrator.py สำหรับจุดที่เรียกจริง (ปัจจุบันต่อสาย
# แค่ COMPLETION/ERROR ตอนจบ task เท่านั้น เป็นจุดที่คุ้มค่าที่สุด/ความถี่ต่ำสุด — PROGRESS/
# PERMISSION ยังไม่ต่อสาย รอ validate ของจริงก่อน)
_PERSONA_PARAMS = {
    "type": "object",
    "properties": {
        "user_message": {"type": "string", "description": "ข้อความไทยธรรมชาติ สุภาพ กระชับ 1 ประโยค แสดงบน UI"},
        "action_status": {
            "type": "string", "enum": ["IN_PROGRESS", "WAITING_APPROVAL", "COMPLETED", "FAILED"],
        },
    },
    "required": ["user_message", "action_status"],
}
_PERSONA_DESC = "แปลงสถานะ agent ดิบๆ เป็นข้อความไทยธรรมชาติแบบผู้ช่วยส่วนตัว สำหรับแสดงบน UI"
PERSONA_VOICE_TOOL = {"name": "speak_to_user", "description": _PERSONA_DESC, "input_schema": _PERSONA_PARAMS}
_GROQ_PERSONA_TOOLS = [
    {"type": "function", "function": {"name": "speak_to_user", "description": _PERSONA_DESC, "parameters": _PERSONA_PARAMS}},
]
_GEMINI_PERSONA_TOOLS = [
    {"function_declarations": [{"name": "speak_to_user", "description": _PERSONA_DESC, "parameters": _PERSONA_PARAMS}]},
]

_PERSONA_SYSTEM_PROMPT = (
    "You are the Voice & Persona Interface for a Universal AI Browser Agent.\n"
    "Communicate with the user in natural, polite, friendly, human-like Thai —\n"
    "like a smart digital personal assistant, not a system log.\n\n"
    "TONE & STYLE\n"
    "- Friendly, concise (1 sentence), helpful, natural. Use ครับ/ค่ะ naturally.\n"
    "- NEVER speak like a raw log (e.g. do NOT say \"Status: Executing command\n"
    "  click on selector #search-btn\").\n\n"
    "RULES BY AGENT_STATUS\n"
    "- IN_PROGRESS: state the action on CURRENT_DOMAIN simply, 1 sentence\n"
    "  (e.g. \"กำลังเข้าไปดูสินค้าที่สนใจบน Shopee ให้เลยครับ...\").\n"
    "- WAITING_APPROVAL: contextualize WHY approval is needed from the action\n"
    "  type, without jargon (e.g. \"ปุ่มนี้เป็นปุ่มกดยืนยันการชำระเงิน เพื่อความ\n"
    "  ปลอดภัย ให้ผมกดชำระเงินต่อเลยไหมครับ?\").\n"
    "- FAILED: be encouraging, transparent, solution-oriented (e.g. \"เอ๊ะ\n"
    "  เหมือนหน้าเว็บนี้จะโหลดช้าหน่อย เดี๋ยวผมลองใหม่อีกทางนะครับ\").\n"
    "- COMPLETED: summarize clearly what was achieved on that site (e.g.\n"
    "  \"เรียบร้อยครับ! ผมจองคิวบนเว็บให้เสร็จแล้ว\").\n\n"
    "Output ONLY the speak_to_user tool call. No prose, no markdown."
)

_PERSONA_SAFE_DEFAULT: dict[str, Any] = {"user_message": "", "action_status": "IN_PROGRESS"}


async def generate_persona_message(
    client, model: str, domain_name: str, user_goal: str, agent_status: str, status_detail: str, provider: str,
) -> dict:
    """agent_status (input): "starting"/"waiting_approval"/"failed"/"completed" — สถานะ
    ดิบที่ orchestrator รู้อยู่แล้ว ใช้บอกบริบทให้ LLM เลือกโทนที่เหมาะสม (ดู RULES ด้านบน)
    status_detail: รายละเอียดเสริมเฉพาะจังหวะนั้น (เช่น final_message ตอน COMPLETED, error
    text ตอน FAILED, ชื่อ action ตอน WAITING_APPROVAL) — เว้นว่างได้ถ้าไม่มี

    คืน {user_message, action_status} เสมอ — user_message="" หมายถึง "ไม่มีข้อความ persona
    ให้แสดง" (ผู้เรียกควร fallback ไปโชว์ raw log/message เดิมแทน ไม่ใช่โชว์อะไรว่างเปล่า)

    ห้าม throw ออกไปให้ orchestrator loop พังเด็ดขาด เป็นแค่ presentation layer เสริม
    ไม่กระทบผลลัพธ์จริงของ task เลยไม่ว่าจะพังแค่ไหน — error ใดๆ คืน _PERSONA_SAFE_DEFAULT"""
    prompt = (
        f"CURRENT_DOMAIN: {domain_name}\n"
        f"USER_GOAL: {user_goal}\n"
        f"AGENT_STATUS: {agent_status}\n"
        f"STATUS_DETAIL: {status_detail}\n\n"
        "Call speak_to_user now."
    )
    try:
        if provider == "anthropic":
            response = await client.messages.create(
                model=model,
                max_tokens=256,
                system=_PERSONA_SYSTEM_PROMPT,
                tools=[PERSONA_VOICE_TOOL],
                tool_choice={"type": "tool", "name": "speak_to_user"},
                messages=[{"role": "user", "content": prompt}],
            )
            tool_use = next((b for b in response.content if b.type == "tool_use"), None)
            result = tool_use.input if tool_use is not None else None
        elif provider == "groq":
            response = await client.chat.completions.create(
                model=model,
                max_tokens=256,
                messages=[
                    {"role": "system", "content": _PERSONA_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                tools=_GROQ_PERSONA_TOOLS,
                tool_choice={"type": "function", "function": {"name": "speak_to_user"}},
            )
            tool_calls = response.choices[0].message.tool_calls or []
            result = json.loads(tool_calls[0].function.arguments) if tool_calls else None
        elif provider == "gemini":
            gemini_model = client.GenerativeModel(
                model_name=model,
                tools=_GEMINI_PERSONA_TOOLS,
                tool_config={"function_calling_config": {"mode": "ANY"}},
                system_instruction=_PERSONA_SYSTEM_PROMPT,
            )
            response = await gemini_model.generate_content_async(
                contents=[{"role": "user", "parts": [{"text": prompt}]}],
            )
            result = None
            for part in response.candidates[0].content.parts:
                fc = getattr(part, "function_call", None)
                if fc and fc.name == "speak_to_user":
                    result = _gemini_struct_to_plain_python(fc.args)
                    break
        else:
            result = None

        return result if result is not None else dict(_PERSONA_SAFE_DEFAULT)
    except Exception as e:
        print(f"⚠️ generate_persona_message error: {e}", flush=True)
        return dict(_PERSONA_SAFE_DEFAULT)


# --- W19-4: "Orchestrator & Planner Agent for Multi-Turn Conversations" — decides how a
# NEW user instruction (turn N ของ conversation เดียวกัน) ควรถูกจัดการ โดยไม่เสียบริบทเดิม
# (ไม่ navigate กลับหน้าแรก/ค้นหาใหม่ทั้งที่ user อ้างถึงของที่เจอไปแล้ว)
#
# *** สถานะปัจจุบัน: ต่อสายจริงแล้วใน routes.py (ดู _run_with_resolved_browser, W19-6
# MODULE 3 "Ordinal Selection") — SessionRegistry.BrowserSession.extracted_memory (list[dict],
# ดู session_registry.py) เป็น buffer ที่ persist ข้าม turn ตามที่ comment เดิมรอไว้ ก่อนเรียก
# orchestrator.run_task() เลยด้วยซ้ำ (REPLY_FROM_MEMORY ข้าม run_task()/การเปิด browser
# ไปทั้งหมดจริงตามที่ออกแบบไว้)
async def route_multi_turn_strategy(
    client, model: str, overall_goal: str, current_user_instruction: str, current_domain: str,
    current_url: str, extracted_memory_buffer: str, recent_action_history: str, provider: str,
) -> dict:
    """extracted_memory_buffer: สรุปข้อมูลที่เคย extract ไว้จาก turn ก่อนๆ ของ conversation
    เดียวกัน (เช่น ผลลัพธ์จาก extract_structured_items() ด้านล่าง) — ส่งเป็น string
    (caller เป็นคนตัดสินใจ format เอง เช่น JSON dump ของ list รายการ) ว่างเปล่าได้ถ้ายังไม่
    เคย extract อะไรมาก่อนในเทิร์นก่อนหน้า

    recent_action_history: สรุป action 3 ครั้งล่าสุด (เช่น จาก ShortTermMemory.recent(3))
    ว่างเปล่าได้ถ้าเพิ่งเริ่ม session

    ห้าม throw ออกไปพังเด็ดขาดไม่ว่ากรณีใด — คืน _MULTI_TURN_SAFE_DEFAULT
    (chosen_strategy=NEW_NAVIGATION) แทนเสมอ ซึ่งเท่ากับ "พฤติกรรมเดิมของระบบทุกวันนี้"
    (ทุก turn ทำ task ใหม่อิสระ ไม่มี strategy router เลย) — ไม่ใช่ค่าที่สุ่มเดา แต่เป็นค่าที่
    ปลอดภัยที่สุดเพราะเท่ากับปิด feature นี้ไปเฉยๆ เมื่อตัดสินใจไม่ได้จริง"""
    prompt = (
        f"OVERALL_CONVERSATION_GOAL: {overall_goal}\n"
        f"CURRENT_USER_INSTRUCTION_TURN_N: {current_user_instruction}\n"
        f"ACTIVE_DOMAIN: {current_domain}\n"
        f"ACTIVE_URL: {current_url}\n"
        f"EXTRACTED_MEMORY_BUFFER:\n{extracted_memory_buffer or '(ว่างเปล่า — ยังไม่เคย extract อะไรมาก่อน)'}\n\n"
        f"RECENT_ACTION_HISTORY (last 3 steps):\n{recent_action_history or '(ว่างเปล่า — เพิ่งเริ่ม session)'}\n\n"
        "Call route_strategy now."
    )
    try:
        if provider == "anthropic":
            response = await client.messages.create(
                model=model,
                max_tokens=768,
                system=_MULTI_TURN_SYSTEM_PROMPT,
                tools=[MULTI_TURN_STRATEGY_TOOL],
                tool_choice={"type": "tool", "name": "route_strategy"},
                messages=[{"role": "user", "content": prompt}],
            )
            tool_use = next((b for b in response.content if b.type == "tool_use"), None)
            result = tool_use.input if tool_use is not None else None
        elif provider == "groq":
            response = await client.chat.completions.create(
                model=model,
                max_tokens=768,
                messages=[
                    {"role": "system", "content": _MULTI_TURN_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                tools=_GROQ_MULTI_TURN_TOOLS,
                tool_choice={"type": "function", "function": {"name": "route_strategy"}},
            )
            tool_calls = response.choices[0].message.tool_calls or []
            result = json.loads(tool_calls[0].function.arguments) if tool_calls else None
        elif provider == "gemini":
            gemini_model = client.GenerativeModel(
                model_name=model,
                tools=_GEMINI_MULTI_TURN_TOOLS,
                tool_config={"function_calling_config": {"mode": "ANY"}},
                system_instruction=_MULTI_TURN_SYSTEM_PROMPT,
            )
            response = await gemini_model.generate_content_async(
                contents=[{"role": "user", "parts": [{"text": prompt}]}],
            )
            result = None
            for part in response.candidates[0].content.parts:
                fc = getattr(part, "function_call", None)
                if fc and fc.name == "route_strategy":
                    result = _gemini_struct_to_plain_python(fc.args)
                    break
        else:
            result = None

        return result if result is not None else dict(_MULTI_TURN_SAFE_DEFAULT)
    except Exception as e:
        print(f"⚠️ route_multi_turn_strategy error: {e}", flush=True)
        return dict(_MULTI_TURN_SAFE_DEFAULT)


_MULTI_TURN_PARAMS = {
    "type": "object",
    "properties": {
        "context_analysis": {
            "type": "object",
            "properties": {
                "is_continuation_of_previous_turn": {
                    "type": "boolean",
                    "description": "true ถ้า CURRENT_USER_INSTRUCTION_TURN_N อ้างถึงรายการ/ข้อมูลที่เจอไปแล้วใน EXTRACTED_MEMORY_BUFFER หรือหน้าปัจจุบัน",
                },
                "target_entity_from_memory": {"type": "string", "description": "คำอธิบาย/ID ของรายการที่ถูกอ้างถึง ถ้ามี ไม่งั้นเว้นว่าง"},
            },
            "required": ["is_continuation_of_previous_turn", "target_entity_from_memory"],
        },
        "chosen_strategy": {
            "type": "string",
            "enum": ["REPLY_FROM_MEMORY", "IN_PAGE_ACTION", "NEW_NAVIGATION"],
            "description": (
                "REPLY_FROM_MEMORY = คำตอบอยู่ใน buffer แล้ว ไม่ต้องทำ browser action เลย. "
                "IN_PAGE_ACTION = ต้องดู/คลิก element บนหน้าปัจจุบัน ไม่ต้อง navigate ไปไหน. "
                "NEW_NAVIGATION = user ขอหัวข้อ/เว็บใหม่จริงๆ เท่านั้นถึงเลือกอันนี้"
            ),
        },
        "reasoning": {"type": "string", "description": "เหตุผลสั้นๆ ว่าทำไมเลือก strategy นี้"},
        "planned_action": {
            "type": "object",
            "properties": {
                "tool": {"type": "string", "description": "เช่น reply, click, extract, type, navigate"},
                "target_selector": {"type": "string", "description": "selector/element identity ที่ชัดเจน ถ้ามี ไม่งั้นเว้นว่าง"},
                "parameters": {"type": "object", "description": "พารามิเตอร์เสริมของ tool นี้"},
            },
            "required": ["tool"],
        },
    },
    "required": ["context_analysis", "chosen_strategy", "reasoning", "planned_action"],
}
_MULTI_TURN_DESC = "ตัดสินใจว่า user instruction ใหม่ (turn N) ควรจัดการด้วย REPLY_FROM_MEMORY/IN_PAGE_ACTION/NEW_NAVIGATION"
MULTI_TURN_STRATEGY_TOOL = {
    "name": "route_strategy", "description": _MULTI_TURN_DESC, "input_schema": _MULTI_TURN_PARAMS,
}
_GROQ_MULTI_TURN_TOOLS = [
    {"type": "function", "function": {"name": "route_strategy", "description": _MULTI_TURN_DESC, "parameters": _MULTI_TURN_PARAMS}},
]
_GEMINI_MULTI_TURN_TOOLS = [
    {"function_declarations": [{"name": "route_strategy", "description": _MULTI_TURN_DESC, "parameters": _MULTI_TURN_PARAMS}]},
]

_MULTI_TURN_SYSTEM_PROMPT = (
    "You are the Orchestrator & Planner Agent for a Universal Multi-Turn AI Browser\n"
    "System. Execute user instructions continuously across multi-turn conversations on\n"
    "ANY website without losing context, navigating away unnecessarily, or resetting\n"
    "session state.\n\n"
    "STRICT BEHAVIORAL RULES\n"
    "1. READ MEMORY BUFFER FIRST: before planning any new navigation, check\n"
    "   EXTRACTED_MEMORY_BUFFER. If the instruction refers to previously found items\n"
    "   (\"ราคาเท่าไหร่\", \"ขอรายละเอียดอันแรก\", \"เปรียบเทียบ 2 อันนี้\", \"เอาเข้าตระกร้า\n"
    "   ให้หน่อย\") you MUST use the existing buffer or current page state — do NOT\n"
    "   refresh, do NOT navigate back to the home page, do NOT start a brand-new search.\n"
    "1b. PRONOUN / ENTITY-SWITCH CONTINUATION: a follow-up may name a DIFFERENT entity\n"
    "    than the previous turn while implicitly repeating the same question about it\n"
    "    (Thai ellipsis pattern ending in \"ละ\", e.g. \"Cedric Kelly ละ\", \"คนต่อไปละ\",\n"
    "    \"คนนี้ได้เท่าไหร่\"). If that named/implied entity (by name, row position, or\n"
    "    \"next one\") already exists as a row in EXTRACTED_MEMORY_BUFFER, this is STILL\n"
    "    REPLY_FROM_MEMORY — set target_entity_from_memory to that row's identity and\n"
    "    answer from the buffer, do NOT treat the new entity name as a reason to search\n"
    "    or navigate. Only fall through to IN_PAGE_ACTION/NEW_NAVIGATION if that entity is\n"
    "    genuinely absent from EXTRACTED_MEMORY_BUFFER.\n"
    "2. STATEFUL ACTION DECISION: choose exactly one of 3 strategies —\n"
    "   REPLY_FROM_MEMORY (answer exists in the buffer, no browser action at all),\n"
    "   IN_PAGE_ACTION (need to inspect/click something on the CURRENT page's DOM,\n"
    "   no navigation), NEW_NAVIGATION (ONLY when the user explicitly asks for a new\n"
    "   topic or site, e.g. \"ไปหาเสื้อผ้าแทน\", \"เริ่มค้นหาใหม่\", \"เปิดเว็บอื่น\").\n"
    "3. UNIVERSAL SITE AGNOSTIC: apply these rules the same way on e-commerce, travel/\n"
    "   booking, social networks, productivity tools, and internal dashboards.\n"
    "4. Default to IN_PAGE_ACTION over NEW_NAVIGATION when uncertain whether the\n"
    "   instruction is a continuation — losing context is worse than one extra\n"
    "   in-page inspection step.\n\n"
    "Output ONLY the route_strategy tool call. No prose."
)

_MULTI_TURN_SAFE_DEFAULT: dict[str, Any] = {
    "context_analysis": {"is_continuation_of_previous_turn": False, "target_entity_from_memory": ""},
    "chosen_strategy": "NEW_NAVIGATION",
    "reasoning": "evaluator error/uncertain — fallback ไปพฤติกรรมเดิมของระบบ (ทำ task ใหม่อิสระทุก turn เหมือนไม่มี router นี้อยู่เลย)",
    "planned_action": {"tool": "", "target_selector": "", "parameters": {}},
}


# --- W19-4 (ต่อ): "Structured Data Extractor" — คู่กับ route_multi_turn_strategy() ด้านบน:
# แปลง raw page content (เช่น ผลลัพธ์ดิบจาก perception.extract_table_data()) ให้เป็น list
# ของ "complete package" ต่อรายการ (title+price+status+url+attributes เสริม) แทนที่จะเป็น
# text/list ของ field เดียวโดดๆ — ใช้เป็น input ของ extracted_memory_buffer ที่
# route_multi_turn_strategy() ด้านบนอ่าน ต่อสายจริงแล้วใน routes.py::_update_extracted_memory
# (เก็บผลลัพธ์ลง SessionRegistry.BrowserSession.extracted_memory)
_STRUCTURED_EXTRACT_ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "item_index": {"type": "integer", "description": "ลำดับรายการ เริ่มจาก 1"},
        "title": {"type": "string", "description": "ชื่อ/หัวข้อของรายการ"},
        "price": {"type": "string", "description": "ราคา/มูลค่า ถ้ามีบนหน้านี้ ไม่งั้นเว้นว่าง"},
        "status": {"type": "string", "description": "สถานะ เช่น In Stock/Sold Out/Available ถ้ามี ไม่งั้นเว้นว่าง"},
        "url": {"type": "string", "description": "ลิงก์/ID ของรายการ ถ้ามี ไม่งั้นเว้นว่าง"},
        "attributes": {
            "type": "object",
            "description": "ฟิลด์เสริมอื่นๆ ที่มีอยู่จริงบนหน้านี้แต่ไม่เข้าฟิลด์มาตรฐานด้านบน (เช่น rating, badge, quantity, date)",
        },
    },
    "required": ["item_index", "title"],
}
_STRUCTURED_EXTRACT_PARAMS = {
    "type": "object",
    "properties": {
        "items": {"type": "array", "items": _STRUCTURED_EXTRACT_ITEM_SCHEMA},
    },
    "required": ["items"],
}
_STRUCTURED_EXTRACT_DESC = "แปลง raw page content ให้เป็น structured item array (title+price+status+url ต่อรายการ)"
STRUCTURED_EXTRACTOR_TOOL = {
    "name": "emit_structured_items", "description": _STRUCTURED_EXTRACT_DESC, "input_schema": _STRUCTURED_EXTRACT_PARAMS,
}
_GROQ_STRUCTURED_EXTRACT_TOOLS = [
    {"type": "function", "function": {"name": "emit_structured_items", "description": _STRUCTURED_EXTRACT_DESC, "parameters": _STRUCTURED_EXTRACT_PARAMS}},
]
_GEMINI_STRUCTURED_EXTRACT_TOOLS = [
    {"function_declarations": [{"name": "emit_structured_items", "description": _STRUCTURED_EXTRACT_DESC, "parameters": _STRUCTURED_EXTRACT_PARAMS}]},
]

_STRUCTURED_EXTRACT_SYSTEM_PROMPT = (
    "You are the Structured Data Extractor. When extracting data from ANY web page:\n"
    "1. Always capture entities as COMPLETE PACKAGES (e.g. Name + Price + Rating + Link)\n"
    "   — never extract a standalone attribute without pairing it to its parent item.\n"
    "2. Never invent data that is not present in PAGE_CONTENT — leave a field empty\n"
    "   (\"\") rather than guessing.\n"
    "3. Fields that don't fit title/price/status/url go into \"attributes\" (e.g. rating,\n"
    "   badge, quantity, date) — keep them tied to the same item_index.\n"
    "4. STRICT TOP-TO-BOTTOM ORDER (W19): item_index must follow the exact order items\n"
    "   appear in PAGE_CONTENT, top to bottom. Never reorder, sort, or rank items\n"
    "   yourself for any reason.\n"
    "5. FILTER OUT NON-ORGANIC ITEMS (W19): skip entries that are clearly ads,\n"
    "   sponsored/promoted listings, \"recommended for you\"/\"people also viewed\"\n"
    "   sections, or navigation/badge clutter mixed into PAGE_CONTENT — only extract the\n"
    "   main organic list/search results the user actually asked for. If PAGE_CONTENT\n"
    "   marks something as \"Ad\"/\"Sponsored\"/\"แนะนำ\", exclude it entirely (do not just\n"
    "   flag it — leave it out of the array).\n\n"
    "Output ONLY the emit_structured_items tool call. No prose."
)


async def extract_structured_items(client, model: str, page_content: str, extraction_hint: str, provider: str) -> list[dict]:
    """page_content: raw text/markdown ที่ได้จาก perception.extract_table_data() หรือ
    เนื้อหาดิบอื่นที่ต้องการให้จัดโครงสร้าง — extraction_hint: บริบทเสริมสั้นๆ ว่ากำลังมองหา
    อะไร (เช่น "รายการสินค้าในผลค้นหา") ว่างเปล่าได้

    คืน list ของ dict เสมอ (ไม่ใช่ dict ห่อ "items" — unwrap ให้ผู้เรียกใช้ตรงๆ) — ห้าม throw
    ออกไปพังเด็ดขาดไม่ว่ากรณีใด คืน [] เปล่าๆ แทนเสมอตอน error (ผู้เรียก fallback ไปใช้
    page_content ดิบต่อได้ตามปกติ เหมือนไม่มีตัวจัดโครงสร้างนี้อยู่เลย)"""
    if not (page_content or "").strip():
        return []
    prompt = (
        f"EXTRACTION_HINT: {extraction_hint or '(ไม่มี — จัดโครงสร้างทุกรายการที่เห็น)'}\n\n"
        f"PAGE_CONTENT:\n{page_content}\n\n"
        "Call emit_structured_items now."
    )
    try:
        if provider == "anthropic":
            response = await client.messages.create(
                model=model,
                max_tokens=2048,
                system=_STRUCTURED_EXTRACT_SYSTEM_PROMPT,
                tools=[STRUCTURED_EXTRACTOR_TOOL],
                tool_choice={"type": "tool", "name": "emit_structured_items"},
                messages=[{"role": "user", "content": prompt}],
            )
            tool_use = next((b for b in response.content if b.type == "tool_use"), None)
            result = tool_use.input if tool_use is not None else None
        elif provider == "groq":
            response = await client.chat.completions.create(
                model=model,
                max_tokens=2048,
                messages=[
                    {"role": "system", "content": _STRUCTURED_EXTRACT_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                tools=_GROQ_STRUCTURED_EXTRACT_TOOLS,
                tool_choice={"type": "function", "function": {"name": "emit_structured_items"}},
            )
            tool_calls = response.choices[0].message.tool_calls or []
            result = json.loads(tool_calls[0].function.arguments) if tool_calls else None
        elif provider == "gemini":
            gemini_model = client.GenerativeModel(
                model_name=model,
                tools=_GEMINI_STRUCTURED_EXTRACT_TOOLS,
                tool_config={"function_calling_config": {"mode": "ANY"}},
                system_instruction=_STRUCTURED_EXTRACT_SYSTEM_PROMPT,
            )
            response = await gemini_model.generate_content_async(
                contents=[{"role": "user", "parts": [{"text": prompt}]}],
            )
            result = None
            for part in response.candidates[0].content.parts:
                fc = getattr(part, "function_call", None)
                if fc and fc.name == "emit_structured_items":
                    result = _gemini_struct_to_plain_python(fc.args)
                    break
        else:
            result = None

        if result is None:
            return []
        items = result.get("items")
        return items if isinstance(items, list) else []
    except Exception as e:
        print(f"⚠️ extract_structured_items error: {e}", flush=True)
        return []


# --- W19-5: "Structured Data Extractor Engine" (Query Normalizer) — ต่างจาก
# extract_structured_items() ด้านบน (แปลง raw page CONTENT ที่ดึงมาแล้วให้เป็น item
# array) ตัวนี้ทำงาน "ก่อน" การดึงข้อมูลเลย: แปลงคำถามภาษาธรรมชาติยาวๆ ของ user (เช่น
# "อ่านรายชื่อผู้ใช้งานระบบในหน้าแอดมินทั้งหมด") ให้เป็น target scope/fields ที่เจาะจง
# พอจะใช้เป็น input ของ perception.extract_table_data()/actions.read_page_data() ได้ตรงๆ
# แทนที่จะให้ LLM หลักเดา CSS selector ดิบๆ เองจาก query ยาวๆ
#
# *** สถานะปัจจุบัน: standalone + tested เท่านั้น "ยังไม่ต่อสาย" เข้า actions.py::
# read_page_data()/orchestrator.py loop เลย (ต่างจาก route_multi_turn_strategy/
# extract_structured_items ด้านบนที่ต่อสายจริงแล้วใน routes.py — ตัวนี้ยังไม่มีจุดต่อสาย
# เพราะเพิ่ม LLM call แยกก่อน read_page_data ทุกครั้งมีต้นทุน latency จริง ต้องตัดสินใจจุด
# ต่อสายที่เหมาะสมก่อน ไม่ใช่แค่เพิ่ม if-branch เข้า loop เดิม) ***
_EXTRACTION_QUERY_PARAMS = {
    "type": "object",
    "properties": {
        "normalized_target_scope": {
            "type": "string",
            "description": "CSS selector ที่เจาะจงพอจะเป็น container ของข้อมูลที่ต้องการ (เช่น 'div.oxd-table-body', 'table', 'main')",
        },
        "extraction_type": {
            "type": "string",
            "enum": ["TABLE_MULTI_ROW", "LIST", "SINGLE_VALUE", "COUNT"],
            "description": "TABLE_MULTI_ROW/LIST = หลายแถว/รายการ, SINGLE_VALUE = ค่าเดียว, COUNT = แค่นับจำนวน",
        },
        "data_fields": {
            "type": "array", "items": {"type": "string"},
            "description": "รายชื่อ field ที่ query ต้องการจริงๆ (เช่น ['Username', 'User Role', 'Employee Name', 'Status']) — [] ถ้า query ไม่ได้เจาะจง field ใดเป็นพิเศษ",
        },
    },
    "required": ["normalized_target_scope", "extraction_type", "data_fields"],
}
_EXTRACTION_QUERY_DESC = "แปลงคำถามภาษาธรรมชาติยาวๆ ให้เป็น target scope/extraction type/field ที่เจาะจงสำหรับดึงข้อมูลจาก DOM"
EXTRACTION_QUERY_NORMALIZER_TOOL = {
    "name": "emit_normalized_query", "description": _EXTRACTION_QUERY_DESC, "input_schema": _EXTRACTION_QUERY_PARAMS,
}
_GROQ_EXTRACTION_QUERY_TOOLS = [
    {"type": "function", "function": {"name": "emit_normalized_query", "description": _EXTRACTION_QUERY_DESC, "parameters": _EXTRACTION_QUERY_PARAMS}},
]
_GEMINI_EXTRACTION_QUERY_TOOLS = [
    {"function_declarations": [{"name": "emit_normalized_query", "description": _EXTRACTION_QUERY_DESC, "parameters": _EXTRACTION_QUERY_PARAMS}]},
]

_EXTRACTION_QUERY_SYSTEM_PROMPT = (
    "You are the Structured Data Extractor Engine. Convert long natural language read\n"
    "requests into precise DOM extraction queries.\n\n"
    "RULES\n"
    "- Do NOT pass the raw natural language string through as a selector.\n"
    "- Normalize into a targeted scope: identify the container that most likely holds\n"
    "  the requested data (e.g. a table body, card grid, or list container) using\n"
    "  TARGET_DOM_SCOPE as a hint when given.\n"
    "- List the specific data fields the query is actually asking for (e.g. Username,\n"
    "  User Role, Employee Name, Status) — leave data_fields empty only if the query\n"
    "  is genuinely unspecific about which fields matter.\n"
    "- Prefer extraction_type=COUNT when the query is purely about how many items\n"
    "  exist, not their contents.\n"
    "- Output ONLY the emit_normalized_query tool call. No prose."
)

_EXTRACTION_QUERY_SAFE_DEFAULT: dict[str, Any] = {
    "normalized_target_scope": "",
    "extraction_type": "TABLE_MULTI_ROW",
    "data_fields": [],
}


async def normalize_extraction_query(
    client, model: str, raw_user_query: str, main_content_container: str, provider: str,
) -> dict:
    """raw_user_query: คำถามภาษาธรรมชาติดิบๆ จาก user (เช่น "อ่านรายชื่อผู้ใช้งานระบบใน
    หน้าแอดมินทั้งหมด") — main_content_container: hint ของ container หลักที่ข้อมูลน่าจะ
    อยู่ (เช่น จาก region="main" ใน perception.py, ดู "Scoped Search Context") ว่างเปล่า
    ได้ถ้าไม่รู้

    ห้าม throw ออกไปพังเด็ดขาดไม่ว่ากรณีใด — คืน _EXTRACTION_QUERY_SAFE_DEFAULT แทนเสมอ
    (normalized_target_scope="" = ให้ผู้เรียก fallback ไปใช้ target_hint/query เดิมที่มี
    อยู่แล้วตรงๆ เหมือนไม่มีตัว normalize นี้อยู่เลย)"""
    if not (raw_user_query or "").strip():
        return dict(_EXTRACTION_QUERY_SAFE_DEFAULT)
    prompt = (
        f"RAW_USER_QUERY: {raw_user_query}\n"
        f"TARGET_DOM_SCOPE: {main_content_container or '(ไม่ทราบ)'}\n\n"
        "Call emit_normalized_query now."
    )
    try:
        if provider == "anthropic":
            response = await client.messages.create(
                model=model,
                max_tokens=512,
                system=_EXTRACTION_QUERY_SYSTEM_PROMPT,
                tools=[EXTRACTION_QUERY_NORMALIZER_TOOL],
                tool_choice={"type": "tool", "name": "emit_normalized_query"},
                messages=[{"role": "user", "content": prompt}],
            )
            tool_use = next((b for b in response.content if b.type == "tool_use"), None)
            result = tool_use.input if tool_use is not None else None
        elif provider == "groq":
            response = await client.chat.completions.create(
                model=model,
                max_tokens=512,
                messages=[
                    {"role": "system", "content": _EXTRACTION_QUERY_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                tools=_GROQ_EXTRACTION_QUERY_TOOLS,
                tool_choice={"type": "function", "function": {"name": "emit_normalized_query"}},
            )
            tool_calls = response.choices[0].message.tool_calls or []
            result = json.loads(tool_calls[0].function.arguments) if tool_calls else None
        elif provider == "gemini":
            gemini_model = client.GenerativeModel(
                model_name=model,
                tools=_GEMINI_EXTRACTION_QUERY_TOOLS,
                tool_config={"function_calling_config": {"mode": "ANY"}},
                system_instruction=_EXTRACTION_QUERY_SYSTEM_PROMPT,
            )
            response = await gemini_model.generate_content_async(
                contents=[{"role": "user", "parts": [{"text": prompt}]}],
            )
            result = None
            for part in response.candidates[0].content.parts:
                fc = getattr(part, "function_call", None)
                if fc and fc.name == "emit_normalized_query":
                    result = _gemini_struct_to_plain_python(fc.args)
                    break
        else:
            result = None

        return result if result is not None else dict(_EXTRACTION_QUERY_SAFE_DEFAULT)
    except Exception as e:
        print(f"⚠️ normalize_extraction_query error: {e}", flush=True)
        return dict(_EXTRACTION_QUERY_SAFE_DEFAULT)


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
    "Goal: {goal}\n\n"
    "URL/หน้าปัจจุบันจริงตอนนี้: {current_url}\n\n"
    "หน้าเว็บเริ่มต้นที่เห็นตอนนี้:\n{page_text}\n\n"
    "เขียนแผนคร่าวๆ ว่าจะทำ goal นี้ให้สำเร็จด้วยขั้นตอนอะไรบ้าง (ไม่เกิน 5-6 ข้อ) — สรุป"
    "ระดับสูงพอให้ user อ่านแล้วเข้าใจและตัดสินใจอนุมัติได้ ไม่ต้องเรียก tool ไม่ต้องระบุ "
    "index ของ element เป๊ะๆ ตอบเป็นข้อความธรรมดา ไม่ต้องมี markdown\n\n"
    "*** Navigation Deduplication (สำคัญ): ตรวจสอบ URL/หน้าปัจจุบันด้านบนก่อนเสมอ — ถ้า"
    "อยู่บนหน้า/module เป้าหมายอยู่แล้ว (เช่น goal พูดถึง 'หน้า Admin' และ URL ปัจจุบันคือ "
    ".../admin/viewSystemUsers อยู่แล้ว) ห้ามใส่ขั้นตอนคลิกเมนู/ลิงก์ navigation ไปหน้านั้นซ้ำ "
    "ให้ข้ามไปขั้นตอนที่ทำบนหน้านี้ได้เลย (เช่น ค้นหา/แก้ไข/กรอกฟอร์ม) — ยกเว้น goal สั่งให้ "
    "'รีเฟรช'/'เปิดใหม่' ชัดเจนเท่านั้นถึงใส่ขั้นตอน navigate ซ้ำได้ ***\n\n"
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


async def generate_plan(client, model: str, goal: str, page_text: str, provider: str, current_url: str = "") -> str:
    """ให้ LLM ร่างแผนระดับสูง (plain text, ไม่เรียก tool) ก่อนเริ่ม agent loop จริง —
    ใช้กับ Orchestrator.run_task(..., confirm_plan=True) เพื่อโชว์ user ก่อนแล้วรอกดยืนยัน
    ค่อยเริ่ม perceive->plan->act loop จริง (ป้องกันไม่ให้ agent ลงมือทำอะไรที่ user ไม่ได้
    เห็นแผนมาก่อน)

    current_url (W19, "Navigation Deduplication"): URL จริงของหน้าปัจจุบัน ณ ตอนร่างแผน
    (ถ้ามี — ผู้เรียกส่งมาจาก page.url จริงถ้ามี page เปิดค้างอยู่แล้ว) ใช้ให้ LLM เช็คว่า
    "อยู่หน้าเป้าหมายอยู่แล้วหรือยัง" ก่อนร่างขั้นตอน navigate ซ้ำที่ไม่จำเป็น — ว่างเปล่าได้
    (default "") ถ้าไม่มี page เปิดอยู่เลย (ad-hoc task ที่ยังไม่เคย perceive อะไร)"""
    prompt = _PLAN_PROMPT_TEMPLATE.format(
        goal=goal, page_text=page_text, current_url=current_url or "(ไม่ทราบ — ยังไม่มีหน้าเว็บเปิดอยู่)",
    )
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


# --- W19-6 ("Master Controller" MODULE 1 — General QA / No-Browser Trigger): เช็คก่อน
# classify_intent() ด้านล่างเสมอ (เร็วกว่า/ถูกกว่า — deterministic ล้วนๆ ไม่เรียก LLM เลย)
# ว่า goal เป็นคำถามทั่วไป/ทักทาย/เวลา/คำนวณ ที่ไม่ต้องแตะ browser เลยไหม — เรียกจาก
# routes.py::_run_with_resolved_browser() ก่อนจะ resolve session/pool ใดๆ ทั้งสิ้น (ข้าม
# การเปิด browser ไปเลยทั้งกระบวนการ ไม่ใช่แค่ข้าม action ภายใน page ที่เปิดอยู่แล้วแบบ
# qa_summary เดิม)
#
# ตั้งใจให้ "แคบ/อนุรักษ์นิยม" มาก (ผิดพลาดแบบ false negative ปลอดภัยกว่า false positive
# เสมอ — เดาว่า "ต้องใช้ browser" ทั้งที่จริงไม่ต้องใช้ แค่เสีย browser session เปล่าๆ แต่
# เดาว่า "ไม่ต้องใช้ browser" ทั้งที่จริงต้องใช้ = ตอบผิด/ตอบไม่ได้เลยทั้งที่ user ต้องการ
# ให้ไปทำ action จริง) — เช็คคำที่บ่งบอกว่าเกี่ยวกับเว็บ/browser ก่อนเสมอ ถ้าเจอคืน False
# ทันทีไม่ว่าจะดูเหมือนคำถามทั่วไปแค่ไหนก็ตาม (เช่น "hi ช่วยค้นหา iPhone ให้หน่อย" มีคำ
# ทักทายนำหน้าแต่จริงๆ ต้องการ browser action)
_GENERAL_CHAT_WEB_EXCLUSION_KEYWORDS = (
    "http://", "https://", "www.", "เว็บ", "หน้าเว็บ", "หน้านี้", "คลิก", "click", "กด",
    "ค้นหา", "search", "กรอก", "fill", "ไปที่", "ไปยัง", "เข้าไปหน้า", "เปิดเว็บ", "goto",
    "go to", "navigate", "ล็อกอิน", "login", "สั่งซื้อ", "ซื้อ", "checkout",
)
# (บั๊กจริงที่ user รายงาน — session log จริง): พิมพ์ตามด้วยคำถาม date/time เต็มรูปแบบ
# ("วันนี้วันที่เท่าไหร่") ก่อนแล้ว "เวลา" คำเดียวโดดๆ เป็น follow-up ครั้งถัดไปในบทสนทนา
# เดียวกัน (พึ่งบริบทก่อนหน้าแทนพิมพ์เต็มซ้ำ) — pattern เดิมที่มีแต่วลียาวๆ ("เวลาเท่าไหร่")
# ไม่ match คำเดี่ยวๆ นี้เลย ทำให้ตกไปเปิด browser ทั้งที่ควรตอบจาก chat ตรงๆ — เพิ่มคำเดี่ยว
# เข้าไปด้วย (ปลอดภัย ไม่กระทบ false positive เพราะ _GENERAL_CHAT_WEB_EXCLUSION_KEYWORDS
# เช็คก่อนเสมออยู่แล้ว — goal ที่มีคำว่า "เวลา"/"วันที่" ปนกับคำเกี่ยวกับเว็บ เช่น "ค้นหาเวลา
# เปิดร้านในหน้าเว็บนี้" จะโดน exclusion keyword กรองออกไปก่อนถึงจุดนี้อยู่ดี)
_GENERAL_CHAT_TIME_DATE_PATTERNS = (
    "วันนี้วันที่", "วันนี้วันอะไร", "วันนี้คือวันที่", "ตอนนี้กี่โมง", "กี่โมงแล้ว",
    "เวลาเท่าไหร่", "เวลาเท่าไร", "เวลาปัจจุบัน", "วันนี้กี่", "ปีนี้ปีอะไร", "ปีนี้ พ.ศ.",
    "เวลา", "วันที่", "วันนี้",
    "what time is it", "what's the time", "what day is it", "what is today's date",
    "today's date", "current time", "current date",
)
_GENERAL_CHAT_GREETING_EXACT_PHRASES = (
    "สวัสดี", "สวัสดีครับ", "สวัสดีค่ะ", "หวัดดี", "หวัดดีครับ", "หวัดดีค่ะ",
    "hello", "hi", "hey", "good morning", "good afternoon", "good evening",
)
# (บั๊กจริงที่ user รายงาน): "อยากฟังเพลงแนวอกหักๆ หามาสัก 4-5 เพลงหน่อย" เปิด browser ทั้งที่
# จริงๆ ตอบได้จากความรู้ทั่วไปของโมเดลเอง (แนะนำชื่อเพลง ไม่ได้ขอ "ค้นหา"/ลิงก์จริงจากเว็บไหน
# เลย) — คำขอเชิง "แนะนำ/อยากฟัง-ดู-อ่าน" ล้วนๆ (ไม่มี exclusion keyword ที่เจาะจงเว็บ/การ
# กระทำบนเว็บปน) นับเป็น general chat ได้เหมือน date/time/greeting — ปลอดภัยเพราะ
# _GENERAL_CHAT_WEB_EXCLUSION_KEYWORDS เช็คก่อนเสมอ (เช่น "แนะนำสินค้าในเว็บนี้หน่อย" มีคำว่า
# "เว็บ" อยู่แล้ว โดน exclusion กรองทิ้งไปก่อนถึงจุดนี้)
_GENERAL_CHAT_RECOMMENDATION_PATTERNS = (
    "แนะนำ", "อยากฟัง", "อยากดู", "อยากอ่าน", "ช่วยแต่ง", "แต่งเพลง", "แต่งกลอน", "แต่งนิทาน",
    "แต่งเรื่อง", "recommend", "suggest",
)
# "1+1 ได้เท่าไหร่", "2*3=", "(4+5)/3 เท่ากับเท่าไหร่" — ตัวเลข/เครื่องหมายคำนวณล้วนๆ
# (บวก ×/÷ ที่บางคนพิมพ์แทน */÷) ตามด้วยคำถามเสริมได้ (หรือไม่มีก็ได้) ไม่มีตัวอักษร
# อื่นปนเลย — เข้มงวดตั้งใจ กัน false positive กับ goal ที่มีตัวเลขปนแต่ไม่ใช่คำนวณจริง (เช่น
# "ไปหน้า 2" ซึ่งจะโดน exclusion keyword ด้านบนกรองออกไปก่อนอยู่ดี)
_MATH_EXPRESSION_RE = re.compile(
    r"^[\d\s\.\+\-\*/×÷()]+(ได้เท่าไหร่|ได้เท่าไร|เท่ากับเท่าไหร่|เท่ากับเท่าไร|=\s*\??|\?)?$"
)


def goal_mentions_web_action(goal: str) -> bool:
    """True ถ้า goal มีคำที่บ่งบอกว่าต้องการ browser action จริงๆ (เว็บ/คลิก/ค้นหา/นำทาง/ฯลฯ
    — ดู _GENERAL_CHAT_WEB_EXCLUSION_KEYWORDS) — factored ออกมาจาก is_general_chat_query()
    ด้านล่างเป็นฟังก์ชัน public แยกต่างหาก เพราะ routes.py ต้องใช้เช็คเดียวกันนี้ตรงๆ ด้วย
    (ดู _run_with_resolved_browser()/generate_plan() ส่วน "file-chat memory follow-up":
    เทิร์นที่ไม่ได้แนบไฟล์ใหม่มา แต่เคยแนบไว้ในเทิร์นก่อนหน้าของ session เดียวกัน ต้องเช็คคำ
    ชุดเดียวกันนี้ก่อนตัดสินใจว่าเป็นคำถามต่อยอดจากไฟล์เดิม หรือ user ต้องการ browser action
    ใหม่จริงๆ)"""
    lower = (goal or "").strip().lower()
    return any(kw in lower for kw in _GENERAL_CHAT_WEB_EXCLUSION_KEYWORDS)


def is_general_chat_query(goal: str) -> bool:
    """True ถ้า goal เป็นคำถามทั่วไป/ทักทาย/ถามวันเวลา/คำนวณเลข/ขอคำแนะนำจากความรู้ทั่วไป
    ที่ตอบได้โดยไม่ต้องแตะ browser เลยแม้แต่นิดเดียว — deterministic ล้วนๆ ไม่เรียก LLM
    (ต่างจาก classify_intent ที่มี LLM fallback สำหรับกรณีกำกวม เพราะ False Positive ของ
    ฟังก์ชันนี้มีต้นทุนสูงกว่า classify_intent มาก — ดู module comment ด้านบน — และเรียกจาก
    routes.py ก่อนแม้แต่จะรู้ว่าจะใช้ provider ไหน/มี client พร้อมหรือยัง เพิ่มชั้น LLM
    fallback ตรงนี้เคยลองแล้วจริงพบว่าทำให้ทุก task (แม้ที่ไม่ใช่ general-chat เลย) ต้องเสีย
    LLM round-trip ก่อนเริ่มเสมอ ไม่ใช่แค่กรณีกำกวมจริงๆ — ต้องคงเป็น deterministic ล้วนๆ)"""
    stripped = (goal or "").strip()
    if not stripped:
        return False
    if goal_mentions_web_action(stripped):
        return False
    lower = stripped.lower()
    if any(p in lower for p in _GENERAL_CHAT_TIME_DATE_PATTERNS):
        return True
    if any(
        lower == g or lower.startswith(f"{g} ") or lower.startswith(f"{g},")
        for g in _GENERAL_CHAT_GREETING_EXACT_PHRASES
    ):
        return True
    if any(p in lower for p in _GENERAL_CHAT_RECOMMENDATION_PATTERNS):
        return True
    if _MATH_EXPRESSION_RE.match(stripped) and any(ch.isdigit() for ch in stripped):
        return True
    return False


_CONTEXT_INSPECTION_PREFIX = "/context"


def is_context_inspection_command(goal: str) -> bool:
    """W20 (MODULE 0 "Special Command Interceptor"): True ถ้า goal ขึ้นต้นด้วยหรือมีคำสั่ง
    "/context" ปน — ตรวจก่อนทุก check อื่นเสมอ (ก่อนแม้แต่ attached_file/
    is_general_chat_query) เพราะ /context คือ debug/inspection mode ที่ user ต้องการ "ดูว่า
    agent เข้าใจคำสั่งว่าอะไร" โดยไม่ต้องการให้ลงมือทำจริงไม่ว่ากรณีใด (ไม่เปิด browser, ไม่
    parse ไฟล์, ไม่เรียก external API ใดๆ) — เช็คแบบ "contains" ไม่ใช่แค่ "startswith"
    ตามสเปค (เผื่อ user พิมพ์นำหน้าด้วยคำอื่นก่อน เช่น "ช่วย /context หน่อย")"""
    return _CONTEXT_INSPECTION_PREFIX in (goal or "").strip().lower()


def strip_context_inspection_command(goal: str) -> str:
    """ตัดคำสั่ง "/context" ออกจาก goal เหลือแค่คำสั่งจริงที่ user ต้องการให้วิเคราะห์ —
    ใช้ก่อนส่งเข้า context_inspection_reply() กันคำว่า "/context" เองไปปนกับคำสั่งจริงตอน
    LLM วิเคราะห์ Goal/Extracted Parameters"""
    stripped = (goal or "").strip()
    lower = stripped.lower()
    idx = lower.find(_CONTEXT_INSPECTION_PREFIX)
    if idx == -1:
        return stripped
    return (stripped[:idx] + stripped[idx + len(_CONTEXT_INSPECTION_PREFIX):]).strip()


# W20 (MODULE 0): system prompt บังคับรูปแบบผลลัพธ์ตามสเปคเป๊ะๆ (หัวข้อ/emoji/ภาษาไทย) —
# ไม่ใช่ format ที่โมเดลจะเดาได้เองแม่นยำพอ ต้องล็อกด้วย system prompt ตรงๆ พร้อมตัวอย่าง
# โครงสร้างชัดเจน ห้ามลงมือทำ action ใดๆ จริง (แค่ "อธิบายความเข้าใจ + แผนที่ตั้งใจจะทำ"
# เท่านั้น ไม่ใช่ลงมือทำจริง)
_CONTEXT_INSPECTION_SYSTEM_PROMPT = (
    "คุณคือ AI agent ที่กำลังอยู่ใน CONTEXT_INSPECTION_MODE — user ต้องการดูว่าคุณเข้าใจ"
    "คำสั่งของเขาว่าอะไร และวางแผนจะตอบอย่างไร โดย \"ห้ามลงมือทำจริง\" เด็ดขาด (ห้ามคลิก/"
    "พิมพ์/นำทางเว็บ, ห้ามอ่าน/parse ไฟล์จริง, ห้ามเรียก API ภายนอกใดๆ) แค่วิเคราะห์คำสั่ง"
    "แล้วอธิบายความเข้าใจ + แผนที่ตั้งใจจะใช้เท่านั้น\n\n"
    "ตอบเป็นภาษาไทยตามโครงสร้างนี้เป๊ะๆ ห้ามเพิ่ม/ตัดหัวข้อ ห้ามใส่ markdown อื่นนอกจากนี้:\n\n"
    "🎯 [ความเข้าใจของ Agent ต่อคำสั่งนี้]\n"
    "- Goal: (อธิบายว่าผู้ใช้ต้องการให้ทำอะไร เป้าหมายหลักคืออะไร)\n"
    "- Target System: (ระบุว่าเป็นงาน Browser, งานไฟล์เอกสาร หรืองานสนทนาทั่วไป)\n"
    "- Extracted Parameters: (ระบุตัวแปรสำคัญ เช่น คำค้นหา, ลำดับ index ที่ต้องการ, ชื่อไฟล์)\n\n"
    "💡 [แนวทางการตอบคำถาม / Plan ที่จะใช้ดำเนินการ]\n"
    "- Strategy: (สรุปขั้นตอนสั้นๆ ที่ Agent ตั้งใจจะทำเพื่อหาคำตอบ)\n"
    "- Expected Output: (ระบุรูปแบบคำตอบที่ Agent เตรียมจะส่งกลับให้ผู้ใช้)\n"
    "- Data Source: (ระบุว่าจะดึงข้อมูลจาก Chat History, Live DOM หรือ File Memory)"
)


async def context_inspection_reply(client, model: str, user_input: str, provider: str) -> str:
    """W20 (MODULE 0): วิเคราะห์คำสั่งที่ user พิมพ์ตาม /context แล้วคืนคำอธิบายตามฟอร์แมต
    Thai structured ที่ตายตัว — ไม่แตะ browser/session/pool/file parser เลย (เหมือน
    chat_response/answer_file_query ด้านบนทุกประการ แค่ system prompt/โครงสร้างคำตอบต่างกัน)

    ห้าม throw ออกไปพังเด็ดขาด — คืนข้อความขอโทษสั้นๆ แทนตอน error"""
    try:
        if provider == "anthropic":
            response = await client.messages.create(
                model=model, max_tokens=512, system=_CONTEXT_INSPECTION_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_input}],
            )
            return "".join(b.text for b in response.content if b.type == "text").strip()
        if provider == "groq":
            response = await client.chat.completions.create(
                model=model, max_tokens=512,
                messages=[
                    {"role": "system", "content": _CONTEXT_INSPECTION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_input},
                ],
            )
            return (response.choices[0].message.content or "").strip()
        if provider == "gemini":
            gemini_model = client.GenerativeModel(
                model_name=model, system_instruction=_CONTEXT_INSPECTION_SYSTEM_PROMPT,
            )
            response = await gemini_model.generate_content_async(
                contents=[{"role": "user", "parts": [{"text": user_input}]}],
            )
            return (response.text or "").strip()
        return "ขออภัยครับ ระบบไม่รู้จัก provider นี้"
    except Exception as e:
        print(f"⚠️ context_inspection_reply error: {e}", flush=True)
        return "ขออภัยครับ ตอนนี้ระบบขัดข้องชั่วคราว ลองใหม่อีกครั้งนะครับ"


_CHAT_RESPONSE_SYSTEM_PROMPT = (
    "คุณคือผู้ช่วย AI ที่เป็นมิตร ตอบคำถามทั่วไป/ทักทาย/บอกวันเวลา/คำนวณเลขง่ายๆ แบบสั้น "
    "กระชับ เป็นธรรมชาติ ไม่ต้องมี markdown"
)


async def chat_response(client, model: str, user_input: str, provider: str, current_time_text: str = "") -> str:
    """ตอบคำถามทั่วไปแบบสนทนาตรงๆ ไม่แตะ browser/DOM เลย (ดู is_general_chat_query ด้านบน
    สำหรับตัวตัดสินใจว่าควรเรียกฟังก์ชันนี้เมื่อไหร่) — current_time_text (optional):
    วันเวลาจริงจากเซิร์ฟเวอร์ (เช่นจาก _current_bangkok_time_text()) ให้คำถามเกี่ยวกับ
    วันที่/เวลาตอบถูกจริง ไม่เดาจาก training data — ว่างเปล่าได้ถ้าคำถามไม่เกี่ยวกับเวลา

    ห้าม throw ออกไปพังเด็ดขาด — คืนข้อความขอโทษสั้นๆ แทนตอน error"""
    prompt = user_input
    if current_time_text:
        prompt = f"เวลาปัจจุบันจริง (Asia/Bangkok): {current_time_text}\n\nคำถามจากผู้ใช้: {user_input}"
    try:
        if provider == "anthropic":
            response = await client.messages.create(
                model=model, max_tokens=512, system=_CHAT_RESPONSE_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            return "".join(b.text for b in response.content if b.type == "text").strip()
        if provider == "groq":
            response = await client.chat.completions.create(
                model=model, max_tokens=512,
                messages=[
                    {"role": "system", "content": _CHAT_RESPONSE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            return (response.choices[0].message.content or "").strip()
        if provider == "gemini":
            gemini_model = client.GenerativeModel(model_name=model, system_instruction=_CHAT_RESPONSE_SYSTEM_PROMPT)
            response = await gemini_model.generate_content_async(
                contents=[{"role": "user", "parts": [{"text": prompt}]}],
            )
            return (response.text or "").strip()
        return "ขออภัยครับ ระบบไม่รู้จัก provider นี้"
    except Exception as e:
        print(f"⚠️ chat_response error: {e}", flush=True)
        return "ขออภัยครับ ตอนนี้ระบบขัดข้องชั่วคราว ลองใหม่อีกครั้งนะครับ"


# --- pdf/xlsx: "Attached File Query" — user แนบไฟล์ PDF/XLSX เข้ามาตรงๆ ผ่าน composer
# (ต่างจาก rag/ingestion.py::ingest_manual ที่เป็นการอัปโหลด manual ไว้ล่วงหน้าเพื่อ
# chunk+embed เข้า ChromaDB — อันนี้เป็น one-shot query เดียว ไม่มี RAG/chunking เลย)
# ดู routes.py::_file_query_result สำหรับจุดต่อสาย (short-circuit เหมือน chat_response
# ด้านบน ไม่แตะ browser/session/pool เลย)
_ANSWER_FILE_QUERY_SYSTEM_PROMPT = (
    "คุณคือผู้ช่วย AI ที่อ่านเอกสารที่ user แนบมาให้ แล้วตอบคำถาม/สรุป/ดึงข้อมูลตามที่ user "
    "ขอ โดยอ้างอิงจาก \"เนื้อหาเอกสาร\" ด้านล่างเท่านั้น ห้ามเดาหรือแต่งข้อมูลที่ไม่มีในเอกสาร "
    "— ถ้าสิ่งที่ user ถามหาไม่มีอยู่ในเอกสารจริงๆ ให้บอกตรงๆ ว่าไม่พบ ตอบแบบกระชับ เป็น"
    "ธรรมชาติ ไม่ต้องมี markdown"
)

# ไม่มี RAG/chunking ในฟีเจอร์นี้ (ดู comment ด้านบน) — จำกัดความยาวเนื้อหาที่ส่งเข้า LLM
# ตรงๆ กันไฟล์ใหญ่มากทำให้ context ล้น/ค่าใช้จ่ายพุ่ง เอกสารที่ยาวเกินนี้จะถูกตัดท้ายทิ้ง
# พร้อมบอก LLM ตรงๆ ว่าเนื้อหาไม่ครบ (กันตอบราวกับเห็นทั้งไฟล์ทั้งที่จริงเห็นแค่บางส่วน)
_ANSWER_FILE_QUERY_MAX_CHARS = 40_000


async def answer_file_query(
    client, model: str, goal: str, file_text: str, filename: str, provider: str,
) -> str:
    """ตอบคำถาม/สรุป/ดึงข้อมูลจากไฟล์ PDF/XLSX ที่ user แนบมา — ไม่แตะ browser/DOM เลย
    (เหมือน chat_response ด้านบนทุกประการ แค่มีเนื้อหาไฟล์เป็น context เพิ่ม)

    ห้าม throw ออกไปพังเด็ดขาด — คืนข้อความขอโทษสั้นๆ แทนตอน error (เหมือน chat_response)"""
    truncated = len(file_text) > _ANSWER_FILE_QUERY_MAX_CHARS
    content = file_text[:_ANSWER_FILE_QUERY_MAX_CHARS]
    truncation_note = (
        "\n\n[หมายเหตุ: เอกสารยาวเกินไป ตัดแสดงแค่บางส่วนด้านบน ไม่ใช่เนื้อหาทั้งหมดของไฟล์]"
        if truncated else ""
    )
    prompt = (
        f"ชื่อไฟล์: {filename}\n\nเนื้อหาเอกสาร:\n{content}{truncation_note}\n\n"
        f"คำขอจากผู้ใช้: {goal}"
    )
    try:
        if provider == "anthropic":
            response = await client.messages.create(
                model=model, max_tokens=1024, system=_ANSWER_FILE_QUERY_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            return "".join(b.text for b in response.content if b.type == "text").strip()
        if provider == "groq":
            response = await client.chat.completions.create(
                model=model, max_tokens=1024,
                messages=[
                    {"role": "system", "content": _ANSWER_FILE_QUERY_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            return (response.choices[0].message.content or "").strip()
        if provider == "gemini":
            gemini_model = client.GenerativeModel(
                model_name=model, system_instruction=_ANSWER_FILE_QUERY_SYSTEM_PROMPT,
            )
            response = await gemini_model.generate_content_async(
                contents=[{"role": "user", "parts": [{"text": prompt}]}],
            )
            return (response.text or "").strip()
        return "ขออภัยครับ ระบบไม่รู้จัก provider นี้"
    except Exception as e:
        print(f"⚠️ answer_file_query error: {e}", flush=True)
        return "ขออภัยครับ ตอนนี้ระบบขัดข้องชั่วคราว ลองใหม่อีกครั้งนะครับ"


# pdf/xlsx (ต่อ): รูปภาพที่ user แนบมาผ่าน composer เดียวกัน — ต่างจาก answer_file_query()
# ด้านบน (extract เป็น text ก่อนด้วย load_manual_bytes()) เพราะรูปภาพไม่มี "text" ให้ extract
# ล่วงหน้า ต้องส่ง base64 ตรงๆ ให้ LLM แบบ multimodal — ทำ 3 provider เต็มรูปแบบ (ไม่ใช่แค่
# Gemini แบบ describe_screenshot() ด้านบน) เพราะ endpoint นี้ user เลือก provider เองได้ผ่าน
# request ปกติ ต่างจาก vision fallback ที่เป็น internal retry mechanism เดียวที่ scope แคบไว้
# ตั้งใจได้
_IMAGE_EXTENSION_MIME_TYPES = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".webp": "image/webp", ".gif": "image/gif",
}

_ANSWER_IMAGE_QUERY_SYSTEM_PROMPT = (
    "คุณคือผู้ช่วย AI ที่ดูภาพที่ user แนบมาให้ แล้วตอบคำถาม/อธิบาย/สรุปสิ่งที่เห็นในภาพตามที่ "
    "user ขอ โดยอ้างอิงจากสิ่งที่เห็นในภาพจริงเท่านั้น ห้ามเดาหรือแต่งสิ่งที่ไม่เห็นในภาพ ตอบแบบ"
    "กระชับ เป็นธรรมชาติ ไม่ต้องมี markdown"
)


async def answer_image_query(
    client, model: str, goal: str, image_bytes: bytes, filename: str, provider: str,
) -> str:
    """ตอบคำถามเกี่ยวกับรูปภาพที่ user แนบมาผ่าน composer — ไม่แตะ browser/DOM เลย (เหมือน
    answer_file_query() ด้านบนทุกประการ แค่ input เป็นรูปภาพ base64 ตรงๆ แทน text ที่ extract
    มาแล้ว) ดู routes.py::_file_query_result สำหรับจุดต่อสาย (แยกสาขาตามนามสกุลไฟล์ว่าเป็น
    รูปภาพหรือเอกสาร ก่อนจะเลือกเรียกฟังก์ชันนี้หรือ answer_file_query())

    ห้าม throw ออกไปพังเด็ดขาด — คืนข้อความขอโทษสั้นๆ แทนตอน error (เหมือน answer_file_query)"""
    mime_type = _IMAGE_EXTENSION_MIME_TYPES.get(Path(filename).suffix.lower(), "image/png")
    prompt = f"คำขอจากผู้ใช้เกี่ยวกับภาพที่แนบมา (ชื่อไฟล์: {filename}): {goal}"
    try:
        if provider == "anthropic":
            image_b64 = base64.b64encode(image_bytes).decode("ascii")
            response = await client.messages.create(
                model=model, max_tokens=1024, system=_ANSWER_IMAGE_QUERY_SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": mime_type, "data": image_b64}},
                        {"type": "text", "text": prompt},
                    ],
                }],
            )
            return "".join(b.text for b in response.content if b.type == "text").strip()
        if provider == "groq":
            image_b64 = base64.b64encode(image_bytes).decode("ascii")
            response = await client.chat.completions.create(
                model=model, max_tokens=1024,
                messages=[
                    {"role": "system", "content": _ANSWER_IMAGE_QUERY_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_b64}"}},
                        ],
                    },
                ],
            )
            return (response.choices[0].message.content or "").strip()
        if provider == "gemini":
            gemini_model = client.GenerativeModel(
                model_name=model, system_instruction=_ANSWER_IMAGE_QUERY_SYSTEM_PROMPT,
            )
            response = await gemini_model.generate_content_async(
                contents=[{
                    "role": "user",
                    "parts": [{"text": prompt}, {"mime_type": mime_type, "data": image_bytes}],
                }],
            )
            return (response.text or "").strip()
        return "ขออภัยครับ ระบบไม่รู้จัก provider นี้"
    except Exception as e:
        print(f"⚠️ answer_image_query error: {e}", flush=True)
        return "ขออภัยครับ ตอนนี้ระบบขัดข้องชั่วคราว ลองใหม่อีกครั้งนะครับ"


# --- Intent Classification & Page Summarization ---
_CLASSIFY_INTENT_PROMPT = """วิเคราะห์ความต้องการ (Intent) ของผู้ใช้จากคำขอ (User Goal/Question) ด้านล่างนี้:
- ตอบว่า "qa_summary" หากผู้ใช้ต้องการถามคำถาม, สรุปเนื้อหา, อ่านข้อมูล, แปลความหมาย, สอบถามราคา/รายละเอียด, สอบถามสินค้า/ข้อมูล หรือประมวลผลข้อมูลจากหน้าเว็บ โดยไม่ต้องการให้ทำการคลิก/กรอกฟอร์ม/นำทาง
- ตอบว่า "action_task" หากผู้ใช้สั่งให้เบราว์เซอร์ทำ Action หรือกระบวนการใดๆ บนหน้าเว็บ เช่น คลิกปุ่ม, กรอกฟอร์ม, ค้นหา, สั่งซื้อสินค้า, ล็อกอิน, นำทางไปหน้าอื่น
- W19: ถ้าคำขอมีทั้งคำสั่ง navigate/คลิก ("เข้าไปหน้า...", "กดปุ่ม...", "เปิดเว็บ...", "คลิก...", "go to...", "navigate to...") ปนกับคำขอให้อ่าน/สรุปข้อมูล ("...แล้วอ่าน...", "...แล้วสรุป...", "...and read...", "...and extract...") ในประโยคเดียวกัน ให้ตอบ "action_task" เสมอ ไม่ว่ากรณีใด (compound command ที่ต้อง navigate ก่อนถึงจะอ่านข้อมูลได้จริง ไม่ใช่ qa_summary)

User Goal/Question: {goal}
Page Content (ย่อ): {page_text_short}

ตอบเพียงคำเดียวเท่านั้น: qa_summary หรือ action_task"""


async def classify_intent(client, model: str, goal: str, page_text: str = "", provider: str = "gemini") -> str:
    """วิเคราะห์ Intent ของผู้ใช้ว่าเป็น qa_summary (การถามตอบ/ขอสรุปเนื้อหา) หรือ action_task (การสั่งงาน/automation บนเว็บ)"""
    goal_lower = goal.lower().strip()

    # Action imperatives (สั่งให้เบราว์เซอร์กระทำ)
    # W19 ("Intent Classification Router"): เพิ่ม "เข้าไปหน้า"/"เปิดเว็บ"/"เปิด"/"ไปยัง" —
    # เดิมมีแค่ "ไปที่" ทำให้วลี navigation แบบอื่นที่ user พิมพ์จริง (เช่น "เข้าไปหน้า Admin
    # แล้วอ่าน...") ไม่ match action_keywords เลยสักตัว ทั้งที่ "อ่าน" match qa_keywords —
    # กลายเป็น has_qa=True, has_action=False ผิดๆ แล้วโดน step 1 ด้านล่างตัดสินเป็น
    # qa_summary ทันทีทั้งที่ user สั่ง navigate จริง (compound command rule ด้านล่างจะทำงาน
    # ไม่ได้เลยถ้า action_keywords ยังจับคำเหล่านี้ไม่ได้ตั้งแต่ต้น)
    action_keywords = [
        "คลิก", "click", "กด", "กรอก", "fill", "พิมพ์", "type", "ซื้อ", "buy", "submit",
        "login", "ล็อกอิน", "เข้าสู่ระบบ", "สมัคร", "register", "search", "ค้นหา",
        "select", "เลือก", "check", "uncheck", "scroll", "ไปที่", "ไปยัง", "เข้าไปหน้า",
        "เข้าหน้า", "เปิดเว็บ", "เปิดหน้า", "เปิด", "goto", "go to", "navigate",
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

    # 2.5 W19 ("Intent Classification Router" ROUTING RULE): ถึงจุดนี้แปลว่า goal match ทั้ง
    # action_keywords และ qa_keywords พร้อมกันจริงๆ (ไม่ถูก step 2 ด้านบนจับว่าเป็น qa เพียวๆ
    # ที่บังเอิญมี action keyword ปนมาแบบไม่ตั้งใจไปแล้ว) — นี่คือ compound command แท้ๆ (เช่น
    # "เข้าไปหน้า Admin แล้วอ่านรายชื่อผู้ใช้", "Go to X and read Y") ต้องไป action_task เสมอ
    # ตัดสินใจแบบ deterministic ตรงนี้เลย ไม่รอ LLM fallback ที่ step 3 (โมเดล compliance ไม่
    # การันตี 100% — เหตุผลเดียวกับที่ permission/rules.py ต้องมีชั้นสำรองระดับโค้ดคู่กับ prompt
    # เสมอ ไม่ใช่พึ่ง prompt อย่างเดียว)
    if has_action and has_qa:
        return "action_task"

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



"""Agent Loop: Perceive -> Plan -> Act -> Verify.

W1: skeleton only. W4: ทำ loop จริงกับเว็บง่าย 1 หน้า.
W5: retry action ที่ล้มเหลว (ดู actions.py::_dispatch_with_retry) + guard กัน
finish_task(false) ก่อนเวลาอันควร (ด้านล่าง) + permission layer/human-in-the-loop
"""

import asyncio
import base64
import re
import sys
import time
from typing import Awaitable, Callable, Optional

from playwright.async_api import Browser, Page, Playwright, async_playwright

from backend.app.config import settings
from backend.app.core import fastpath_executor
from backend.app.core import llm
from backend.app.core import long_term_memory
from backend.app.core import procedural_memory
from backend.app.core.actions import (
    REJECTED_BY_USER_MESSAGE,
    ActionResult,
    AskUserFunc,
    execute,
    goto,
    wait_stable,
)
from backend.app.core.memory import ShortTermMemory
from backend.app.core.perception import get_snapshot
from backend.app.core.user_browser import connect_user_browser, resolve_target_page
from backend.app.permission.rules import extract_domain
from backend.app.rag import retriever

# action ที่เปลี่ยนหน้า/DOM แบบมีนัยสำคัญ -> ต้องรอหน้านิ่งก่อน perceive รอบถัดไป
# W50: press_key เพิ่มเข้ามา — กด Enter บน custom dropdown อาจ submit form/navigate ได้
# เหมือนกัน (ดู actions.py::press_key/perception.py W50)
_PAGE_CHANGING_ACTIONS = {"click", "goto", "select", "go_back", "press_key"}

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

# W44: qa_summary เดิมตอบด้วย llm.summarize_page() ตัวเดียว (ไม่มี tool ให้เรียกเลย) เห็น
# แค่ page_text จาก get_snapshot() (interactive elements ล้วนๆ ไม่มีเนื้อหาตาราง/list) —
# คำถามแบบ "เห็นชื่อ X ในตารางไหม" เลยตอบไม่ได้เสมอแม้ read_page_data/extract_table_data()
# จะมีอยู่แล้วก็ตาม (unreachable เพราะ path นี้ไม่เคยเข้า next_action()/execute() loop เลย)
# ตอนนี้ให้ qa_summary วน next_action() แบบจำกัดสูงสุด _QA_SUMMARY_MAX_STEPS รอบแทน
# อนุญาตแค่ type="read_page_data" (ไม่ mutate อะไรบนหน้าเว็บ) + finish_task เท่านั้น —
# action อื่น (click/fill/...) ถูกปฏิเสธเงียบๆ ไม่ dispatch จริง (แค่ tool_result บอกเหตุผล)
# กันโมเดลหลุดไปทำ action ทั้งที่ user แค่ถามคำถาม ถ้าครบโควตาแล้วยังไม่เรียก finish_task
# เลย fallback กลับไปใช้ summarize_page() แบบเดิมเป็น safety net (ไม่แย่ไปกว่าพฤติกรรมเดิม)
#
# W46 (ต่อยอด W44) user รายงานว่า agent ยอมแพ้เร็วเกินไป (finish_task บอก "ไม่มีข้อมูล" ทั้งที่
# ยังไม่เคยลองค้นหาเลย) — สาเหตุหนึ่งคือ loop นี้ห้าม fill/click เด็ดขาดแม้แต่การกรอกช่อง
# ค้นหา/กดปุ่มค้นหา (คำถามที่คำตอบอยู่หลัง search flow ตอบไม่ได้เลยไม่ว่ากรณีไหน) — ผ่อนให้
# "fill"/"click" ทำได้เพิ่มเติม "เฉพาะ" ตอน label ของ element เป้าหมายดูเป็นช่อง/ปุ่มค้นหา/
# กรองข้อมูลจริงๆ เท่านั้น (ดู _label_looks_like_search()) ยังคงกันโมเดลไม่ให้หลุดไปกด
# element อื่นที่ไม่เกี่ยวกับการค้นหา (เช่น login/checkout/delete) เหมือนเจตนาเดิมของ W44
# ทุกประการ — เพิ่ม _QA_SUMMARY_MAX_STEPS จาก 3 เป็น 4 ด้วย เพราะ flow ค้นหาจริงต้องใช้อย่าง
# น้อย 3 turn (fill -> click -> read_page_data) ก่อนจะเหลือ turn ให้เรียก finish_task ได้
_QA_SUMMARY_MAX_STEPS = 4
# W19 ("Guard Compatibility Rule"): เพิ่ม "หรือคลิกเมนู/nav เพื่อไปหน้าอื่นที่มีข้อมูลที่
# ต้องการได้" ต่อท้ายข้อความเดิม — บอก LLM ตรงๆ ว่านำทางไปหน้าอื่นเพื่อหาคำตอบได้แล้ว (ดู
# region="navigation" ใน qa_is_nav_click ด้านล่างที่อนุญาตจริง) ไม่ใช่แค่ fill/click ช่อง
# ค้นหาเหมือนเดิม
_QA_SUMMARY_ACTION_REJECTED_NUDGE = (
    "[ปฏิเสธ] คำถามนี้คือ qa_summary (ถามข้อมูล ไม่ใช่สั่งงาน) ไม่อนุญาตให้ทำ action ที่มีผล"
    "ต่อหน้าเว็บ (fill/select/goto/...) ยกเว้น fill/click กับช่องค้นหา/ปุ่มค้นหา/ตัวกรอง"
    "ข้อมูล หรือคลิกเมนู/nav เพื่อไปหน้าอื่นที่มีข้อมูลที่ต้องการได้เท่านั้น — ถ้าต้องการอ่าน"
    "เนื้อหาเพิ่มเติมให้ใช้ type: 'read_page_data' แล้วเรียก finish_task พร้อมคำตอบสุดท้าย"
    "ทันทีที่พอตอบคำถามได้ ห้ามสรุปว่า 'ไม่มีข้อมูล' ก่อนลองค้นหา/นำทางไปหาอย่างน้อย 1 ครั้ง "
    "ถ้ายังไม่เคยลองเลย"
)

# label ที่บ่งบอกว่า element เป้าหมายเป็นช่อง/ปุ่มค้นหา/กรองข้อมูลจริงๆ — ใช้เป็นชั้นสำรอง
# ระดับโค้ด (ไม่พึ่ง LLM เลือกถูกเพียงอย่างเดียว เหมือน pattern เดียวกับ RISKY_LABEL_KEYWORDS
# ใน permission/rules.py) จำกัดคำให้เจาะจงพอ ไม่เอาคำสั้นๆ ที่ match ผิดคำอื่นได้ง่าย (เช่น
# "หา" เดี่ยวๆ จะไปแมตช์ "หาย"/"หาก" โดยไม่ตั้งใจ)
_QA_SUMMARY_SEARCH_LABEL_KEYWORDS = ("search", "ค้นหา", "filter", "กรอง", "find", "query")


def _label_looks_like_search(label: str) -> bool:
    lower = (label or "").lower()
    return any(keyword in lower for keyword in _QA_SUMMARY_SEARCH_LABEL_KEYWORDS)


# user รายงานว่า agent ตอบ "list รายชื่อ" ด้วยการแปะรายละเอียดอื่นที่ไม่มีใครถาม (ตำแหน่ง/
# office/salary) ปนมาด้วยทุกคน และเขียนรวมเป็นย่อหน้าเดียวยาว ("1. X (...) 2. Y (...)")
# แทนที่จะขึ้นบรรทัดใหม่ทีละข้อ — แปะ guidance นี้ต่อท้าย goal เฉพาะตอนเข้า qa_summary
# mini-loop เท่านั้น (ไม่แตะ SYSTEM_PROMPT หลักที่ action_task ทั่วไปก็ใช้ร่วมกัน เพราะ
# guidance นี้เกี่ยวกับ "การตอบคำถาม" ล้วนๆ ไม่เกี่ยวกับ action_task ที่ finish_task แค่สรุป
# สั้นๆ ว่าทำอะไรไป)
_QA_ANSWER_FORMAT_GUIDANCE = (
    "\n\n[คำแนะนำการตอบ — สำคัญ]: ตอบเฉพาะข้อมูลที่ผู้ใช้ถามจริงๆ เท่านั้น อย่าแปะรายละเอียด"
    "อื่นที่ไม่มีใครถามมาด้วย (เช่น ถ้าถามแค่ \"รายชื่อ\" ให้ตอบแค่ชื่อ ไม่ต้องพ่วงตำแหน่ง/"
    "office/salary ที่ไม่ได้ถาม) ถ้าคำตอบมีหลายรายการ ให้จัดเป็นลิสต์แบบขึ้นบรรทัดใหม่ทีละข้อ"
    "เสมอ (เช่น \"1. ...\\n2. ...\\n3. ...\") ห้ามเขียนรวมเป็นย่อหน้าเดียวยาวๆ"
)

_PREMATURE_TRUE_FINISH_NUDGE = (
    "การเรียก finish_task(success=true) นี้ยังไม่มี action ใดๆ เกิดขึ้นเลยใน task นี้ "
    "(steps_taken=0) — ก่อนยืนยัน success ให้ตรวจสอบอีกครั้งว่า indexed elements ล่าสุด "
    "มีหลักฐานชัดเจนจริงๆ ว่า goal สำเร็จแล้ว ถ้าใช่จริง เรียก finish_task(success=true) "
    "อีกครั้งได้เลย ถ้าไม่แน่ใจ ให้ลองทำ action ที่เกี่ยวข้องกับ goal ก่อน"
)

# Task4 ("Task Completion Verifier", W19): symmetric กับ guard ด้านบนแต่เช็คคนละสัญญาณ —
# guard ด้านบนเช็คแค่ "steps_taken==0" (ไม่มีหลักฐานว่าทำอะไรเลย) ตัวนี้เช็ค "หน้าเว็บ
# ปัจจุบันมี validation error โผล่อยู่จริงไหม" (เช่น กรอกฟอร์มแล้วกด Save แต่ field ยังไม่
# ผ่าน validation — steps_taken > 0 แล้วแต่ยังไม่สำเร็จจริง guard เดิมด้านบนจับไม่ได้เพราะ
# เช็คแค่ step แรกสุด) — ทำงานอิสระจาก guard เดิม ไม่ทับซ้อนกัน (เช็คคนละเงื่อนไข ทำงาน
# พร้อมกันได้ทั้งคู่) ไม่ว่า steps_taken จะเท่าไหร่ก็ตาม
_MAX_PREMATURE_VALIDATION_ERROR_RETRIES = 2
_PREMATURE_VALIDATION_ERROR_NUDGE_TEMPLATE = (
    "การเรียก finish_task(success=true) นี้ถูกปฏิเสธ — ตรวจพบข้อความ error/validation ที่ยัง"
    "แสดงอยู่บนหน้าปัจจุบัน: {errors} ห้ามถือว่า task สำเร็จทั้งที่ยังมี error พวกนี้ค้างอยู่ "
    "ให้แก้ field ที่เกี่ยวข้องตามข้อความ error ก่อน (เช่น กรอกช่องที่ว่าง/แก้ค่าที่ไม่ถูกต้อง/"
    "เปลี่ยนค่าที่ซ้ำ) แล้วลองใหม่ ถ้าแก้แล้ว error หายไปแล้วจริงๆ ค่อยเรียก "
    "finish_task(success=true) อีกครั้ง"
)

# ARIA [role=alert] เป็นมาตรฐานข้ามเว็บ (ใช้ได้ทุกเว็บที่ทำ a11y ไว้) ส่วน [class*=error/
# invalid] เป็นชั้นสำรองสำหรับเว็บที่ไม่ได้ใช้ ARIA (พบบ่อยมาก) — .oxd-input-field-error-
# message เจาะจง OrangeHRM ตรงๆ (user รายงานปัญหานี้มาจากเว็บนี้โดยตรง) :not(:empty) กัน
# นับ element placeholder ที่ framework render ทิ้งไว้เสมอแต่ว่างอยู่ตอนไม่มี error จริง
#
# W20 (Task12, "UI Validation Error Detection"): ยืนยันด้วยการ trigger validation error
# จริงบน opensource-demo.orangehrmlive.com (กรอกรหัสผ่านอ่อนในฟอร์ม Change Password) — ข้อความ
# error จริง ("Should have at least 7 characters") อยู่ใน
# `<span class="oxd-text oxd-text--span oxd-input-field-error-message oxd-input-group__message">`
# ซึ่ง .oxd-input-field-error-message (มีอยู่แล้ว) แมตช์อยู่แล้วจริงๆ — เพิ่ม
# .oxd-input-group__message/.text-danger/.invalid-feedback/.oxd-input--error เป็นชั้นสำรอง
# เพิ่มเติมสำหรับเว็บอื่นที่อาจไม่มี class "error"/"invalid" ติดมาด้วย (Bootstrap ใช้
# .text-danger/.invalid-feedback เป็นชื่อ class มาตรฐานของตัวเอง ไม่มีคำว่า error/invalid
# ปนเลย ตัว [class*=...] ด้านบนจะจับไม่ได้ถ้าไม่เพิ่มตรงๆ)
_VALIDATION_ERROR_SELECTOR = (
    '[role="alert"]:not(:empty), [class*="error" i]:not(:empty), '
    '[class*="invalid" i]:not(:empty), .oxd-input-field-error-message, '
    '.oxd-input-group__message, .text-danger, .invalid-feedback, .oxd-input--error'
)

# W20 (Task12 follow-up, บั๊กจริงที่พบ live บน opensource-demo.orangehrmlive.com): หน้า login
# ของเว็บนี้มี `<div class="orangehrm-login-error">Username : Admin / Password : admin123</div>`
# เป็น "คำใบ้ demo credentials" คงที่อยู่นอก <form> เสมอ ไม่เกี่ยวอะไรกับ action ที่เพิ่งทำเลย —
# แต่ชื่อ class มีคำว่า "error" ติดมาด้วย (ตั้งชื่อผิดโดย OrangeHRM เอง) ทำให้ [class*="error" i]
# เดิมแมตช์ผิดพลาด hard-stop guard ใหม่ (หลัง fill) ทันทีตั้งแต่ step แรกโดยไม่มี validation
# error จริงเกิดขึ้นเลย — ยืนยันด้วย DOM จริง: element นี้ไม่ได้อยู่ใน <form class="oxd-form">
# (closest('form') === null) ต่างจาก validation error จริงทุกตัวที่เจอมา (.oxd-input-field-
# error-message, Bootstrap .invalid-feedback ฯลฯ) ซึ่ง render อยู่ใน <form> เสมอเพราะผูกกับ
# input field ของฟอร์มนั้นโดยตรง — จำกัด scope การสแกนให้อยู่แค่ใน <form> เท่านั้นสำหรับ hard-
# stop guard ใหม่ (ดู _scan_validation_errors()'s within_form param) กรอง false positive แบบ
# นี้ทิ้งได้โดยทั่วไป ไม่ใช่ hack เฉพาะเว็บนี้เว็บเดียว — ไม่แตะ guard เดิมก่อน finish_task
# (ยังคง scope ทั้งหน้าเหมือนเดิมทุกประการ เป็น backstop สำหรับหน้าที่ไม่มี <form> tag จริงๆ
# เช่นบาง SPA ที่ handle submit ด้วย onClick แทน)
_VALIDATION_ERROR_SELECTOR_IN_FORM = ", ".join(
    f"form {part.strip()}" for part in _VALIDATION_ERROR_SELECTOR.split(",")
)

# W19 (latency): timeout สั้นๆ สำหรับ Playwright locator call ที่เป็นแค่ "เช็คสถานะ DOM
# เฉยๆ" (ไม่ใช่การรอ element โผล่มาจริงจากการกระทำ เช่น click/fill) — ไม่ระบุ timeout เอง
# Playwright จะ default เป็น 30000ms ต่อ call เดียว ซึ่งนานเกินจำเป็นมากสำหรับ element ที่
# ควรจะพร้อมอยู่แล้วตั้งแต่ perceive ผ่านมาก่อนหน้านี้ — ใช้กับ _login_form_needs_password()/
# _scan_validation_errors() ด้านล่าง (คนละค่ากับ state_filter.py::_STATE_CHECK_TIMEOUT_MS
# ที่ 500ms เพราะจุดประสงค์ต่างกัน: state_filter เช็คก่อน dispatch ทุก step ต้องเร็วที่สุด
# ส่วนตัวนี้เช็คตอน finish_task/login-guard เท่านั้น ความถี่ต่ำกว่ามาก ให้เวลาเผื่อหน้าที่โหลด
# ช้าได้มากกว่าหน่อย)
_DOM_CHECK_TIMEOUT_MS = 3000


async def _scan_validation_errors(page: Page, within_form: bool = False) -> list[str]:
    """สแกนหา element ที่บ่งบอกว่ามี validation error ปรากฏอยู่จริงบนหน้าปัจจุบัน (มองเห็น
    ได้ + มีข้อความ) — เรียกก่อนยอมรับ finish_task(success=true) เท่านั้น (ไม่ใช่ทุก step
    เพื่อไม่ให้เสีย overhead โดยไม่จำเป็น) คืน list ข้อความที่เจอ (สูงสุด 5 รายการ) หรือ []
    ถ้าไม่เจอเลย/error ระหว่างสแกน (ไม่ throw ให้ finish_task guard พัง — ปลอดภัยกว่าเสมอที่
    จะถือว่า "ไม่เจอ error" ถ้าสแกนไม่ได้จริงๆ ดีกว่าบล็อก finish_task ที่อาจถูกต้องอยู่แล้ว)

    within_form (W20, Task12 follow-up — ดู comment เหนือ _VALIDATION_ERROR_SELECTOR_IN_FORM):
    True = จำกัด scope ให้อยู่แค่ภายใน <form> เท่านั้น กัน false positive จาก static hint/
    banner นอกฟอร์มที่ชื่อ class บังเอิญมีคำว่า "error"/"invalid" ปนอยู่ — ใช้กับ hard-stop
    guard ใหม่หลัง fill/click เท่านั้น ไม่ใช่ default (False = scope ทั้งหน้าเหมือนเดิม ใช้กับ
    guard เดิมก่อน finish_task)

    W19 (latency): .is_visible()/.inner_text() ของ Playwright มี actionability wait ในตัว
    ที่ default เป็น 30000ms ถ้าไม่ระบุ timeout เอง — ถ้า element ตัวไหนหลุด/detach ไประหว่าง
    ทาง (เช่น re-render พอดีตอนกำลังสแกน) การรอ default 30s ต่อ element เดียวจะทำให้
    finish_task guard นี้ช้าเกินจำเป็นไปมาก ใส่ _DOM_CHECK_TIMEOUT_MS (3s) ตรงๆ ให้ทุกจุด"""
    try:
        selector = _VALIDATION_ERROR_SELECTOR_IN_FORM if within_form else _VALIDATION_ERROR_SELECTOR
        locator = page.locator(selector)
        count = await locator.count()
        found: list[str] = []
        for i in range(min(count, 20)):
            item = locator.nth(i)
            try:
                if not await item.is_visible(timeout=_DOM_CHECK_TIMEOUT_MS):
                    continue
                text = (await item.inner_text(timeout=_DOM_CHECK_TIMEOUT_MS)).strip()
            except Exception:
                continue
            if text:
                found.append(text)
            if len(found) >= 5:
                break
        return found
    except Exception:
        return []


# W20 (Task12, "UI Validation Error Detection" — บั๊กจริงที่ user รายงาน): _scan_validation_
# errors() เดิม (ด้านบน) ถูกเรียกแค่จุดเดียวคือก่อนยอมรับ finish_task(success=true) เท่านั้น —
# ถ้า agent ไม่เคยเรียก finish_task เลย (แค่วน fill/click/refresh ไม่จบ ตามที่ user รายงาน)
# guard เดิมไม่มีทางทำงานเลยตลอด task นั้น ต้องเช็คทันทีหลัง action ที่ "ยืนยันว่าจะส่งฟอร์มนี้
# จริงๆ" ด้วย ไม่ใช่รอถึงตอน finish_task เท่านั้น
#
# ตอนแรกตั้งใจ "ไม่" เช็คหลัง fill (แค่ click/submit ปุ่ม Save เท่านั้น) เพราะทดสอบจริงบน
# opensource-demo.orangehrmlive.com พบว่าฟอร์มหลายช่อง (เช่น Password + Confirm Password)
# จะโชว์ "* Required" ให้ช่องที่ "ยังไม่ได้กรอกเลย" ควบคู่ไปกับ error ของช่องที่เพิ่งกรอกจริง —
# hard-stop ทันทีหลัง fill ช่องแรกจะ false-positive ใส่ "Required" ของช่องถัดไปที่ยังไม่ทันกรอก
#
# (2026-08-06) user ขอให้ครอบคลุม fill ด้วยตามสเปคเดิม ("แก้ไขให้ครบ") — เปิดให้ fill trigger
# เช็คนี้ด้วยแล้ว แต่แก้ false-positive เดิมด้วย _is_bare_required_message() ด้านล่าง (กรอง
# "* Required" ล้วนๆ ทิ้งก่อน hard-stop — ช่องพี่น้องที่ยังไม่ได้กรอกไม่ใช่ปัญหาจริง) ส่วน error
# ที่มีเนื้อหาจริง (เช่น "Should have at least 7 characters") ยัง hard-stop เหมือนเดิมทุกกรณี
# ไม่ว่าจะเกิดจาก fill หรือ click/submit ก็ตาม — ผลคือ nudge-retry เดิมของ
# _scan_validation_errors() (ก่อน finish_task, ดูคอมเมนต์เหนือ _MAX_PREMATURE_VALIDATION_
# ERROR_RETRIES) แทบจะไม่มีโอกาสถูกใช้งานอีกแล้วสำหรับ error ที่เกิดจาก fill โดยตรง (hard-stop
# นี้ดักไว้ก่อนเสมอ) — คงมันไว้เป็น backstop เฉยๆ เผื่อ error ที่โผล่จาก action type อื่นที่ไม่
# อยู่ใน _should_check_validation_error_after_action() (เช่น select/check) หรือปุ่มที่ label
# ไม่ match _FORM_SUBMIT_LABEL_KEYWORDS
_FORM_SUBMIT_LABEL_KEYWORDS = (
    "save", "submit", "update", "change password", "confirm",
    "บันทึก", "ยืนยัน", "เปลี่ยนรหัสผ่าน", "อัปเดต", "แก้ไข",
)

# "* Required"/"Required"/"Required." ล้วนๆ ไม่มีเนื้อหาอื่น (เห็นจริงบน OrangeHRM) — ต่างจาก
# error ที่มีเนื้อหาจริงเช่น "Should have at least 7 characters" ซึ่งต้องไม่ถูกกรองทิ้ง
_BARE_REQUIRED_MESSAGE_RE = re.compile(r'^[\*\s]*required[\.\!]?$', re.IGNORECASE)


def _is_bare_required_message(text: str) -> bool:
    """True ถ้าข้อความเป็นแค่ "* Required"/"Required" เฉยๆ (ไม่มีรายละเอียดว่าค่าที่กรอกผิด
    ยังไง) — เป็น false positive ที่โผล่ให้ช่องพี่น้องที่ "ยังไม่ได้กรอกเลย" เท่านั้น ไม่ใช่
    ปัญหาของค่าที่เพิ่ง fill จริงๆ ต้องกรองทิ้งก่อน hard-stop"""
    return bool(_BARE_REQUIRED_MESSAGE_RE.match((text or "").strip()))


def _label_looks_like_form_submit(label: str) -> bool:
    lowered = (label or "").lower()
    return any(kw in lowered for kw in _FORM_SUBMIT_LABEL_KEYWORDS)


def _should_check_validation_error_after_action(action_type: str, label: str) -> bool:
    """True ถ้า action ที่เพิ่ง dispatch สำเร็จ (result.success) นี้ควรเช็ค validation error
    ทันที — "fill" เช็คทุกครั้ง (กรอง "* Required" ล้วนๆ ทิ้งที่จุดเรียกใช้แทน ดู
    _is_bare_required_message()) ส่วน "click"/"submit" เฉพาะตอน label ดูเป็นปุ่ม Save/Submit/
    Confirm/Update เท่านั้น (ไม่เช็คทุก click ทั่วไป เสี่ยง false positive จาก error/alert อื่น
    ที่ไม่เกี่ยวกับฟอร์มบนหน้าที่เพิ่ง navigate ไป)"""
    if action_type == "fill":
        return True
    if action_type in ("click", "submit"):
        return _label_looks_like_form_submit(label)
    return False


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

# W31: ทั้ง 2 guard ด้านบน (คาบ 1 และคาบ 2-4) เดิมพอ trigger แล้วจบ task ทันที
# (success=False) — user ขอให้ลองบังคับทำ action อื่นแทนก่อน (เช่น scroll/go_back) ให้
# โอกาส agent กู้สถานการณ์เอง แทนที่จะยอมแพ้ทันทีที่เจอ loop ครั้งแรก — เลือก go_back เป็น
# ตัวแรกเสมอ (ทางที่น่าเชื่อถือที่สุดที่จะพา agent ออกจาก sub-flow ที่ติดอยู่ เช่น Shorts/
# Reels viewer หรือ modal ที่วนเปิด-ปิดซ้ำ) ถ้ายังวนซ้ำอีกหลังจากนั้น (บังคับ go_back ไปแล้ว
# ก็ยังไม่หลุด) ลองครั้งที่ 2 ด้วย scroll แทน (เผื่อเป็นกรณี "มีตัวเลือกอื่นอยู่นอกจอ ไม่ใช่
# ติดอยู่ใน sub-flow") — เกินจำนวนนี้แล้วยังวนซ้ำไม่หายค่อยยอมแพ้จริง (escape valve เดียวกับ
# guard อื่นๆ ในไฟล์นี้ กัน force ไม่รู้จบถ้า forcing เองก็ไม่ช่วยอะไร)
_MAX_FORCED_LOOP_RECOVERIES = 2
_LOOP_RECOVERY_ACTIONS: list[dict] = [{"type": "go_back"}, {"type": "scroll", "direction": "down"}]


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

# W9[A] vision fallback (Gemini เท่านั้นตอนนี้ — ดูเหตุผล scope ที่ llm.py::
# describe_screenshot()): action ประเภทเหล่านี้เท่านั้นที่ต้องพึ่ง element visibility
# จริงๆ (click/fill/select/check + alias submit/delete/purchase/pay ที่ dispatch ไป
# click ตัวเดิม) — scroll/goto/go_back/switch_tab/wait ล้มเหลวด้วยเหตุผลอื่น ไม่เกี่ยว
# กับ popup/overlay บัง ไม่ต้อง trigger vision
_VISION_FALLBACK_ACTION_TYPES = {
    "click", "fill", "select", "check", "submit", "delete", "purchase", "pay",
    # W50: press_key พึ่ง element visibility เหมือน click (ต้อง focus element ที่มองเห็น
    # ได้จริงก่อนถึงจะกด key ได้ผล) — failure mode เดียวกับ click ที่มี popup/overlay บัง
    "press_key",
}

# W50 (client-side action verification): action ประเภทเหล่านี้ "ควรจะ" ทำให้หน้าเว็บ
# เปลี่ยนแปลงบางอย่างเสมอถ้าทำงานได้จริง (เปิด dropdown, ติ๊ก checkbox, กด option ฯลฯ) —
# ต่างจาก scroll/wait/goto/go_back/switch_tab/read_page_data ที่ "ไม่เปลี่ยน" ก็เป็นเรื่อง
# ปกติ (เช่น scroll ถึงสุดหน้าแล้ว, wait บนหน้าที่นิ่งอยู่แล้ว) — reuse
# _VISION_FALLBACK_ACTION_TYPES เพราะเป็น action กลุ่มเดียวกันที่พึ่ง "มีผลจริงบน DOM"
# เหมือนกันทั้งคู่ (fill รวมอยู่ในนี้ด้วยเพราะ label ของ input เปลี่ยนตามค่าที่กรอกจริง)
_VERIFICATION_SIGNAL_ACTION_TYPES = _VISION_FALLBACK_ACTION_TYPES

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

# W7[A] (context compaction) / W22 (generalize จาก Gemini-only มาทุก provider):
# stateless chat API ต้องส่ง messages ทั้งก้อนซ้ำทุก step (ไม่มี server-side session)
# — ทุก step เพิ่ม page snapshot เต็มๆ + manual/memory/long-term context เข้าไปใน
# messages เรื่อยๆ ไม่เคยหดกลับเลย ทำให้ input token ต่อ step โตขึ้นเรื่อยๆ ตามจำนวน
# step (ไม่ใช่แค่ตามความยาว task จริง) — พอ step สะสมเกิน _COMPACT_AFTER_STEPS ให้ตัด
# step เก่ากว่า _KEEP_RECENT_STEPS ตัวล่าสุดออกจาก messages แล้วแทนที่ด้วย digest สั้นๆ
# (สร้างจาก ShortTermMemory.all() ที่มีข้อมูลสะอาดอยู่แล้ว ไม่ต้อง parse raw message
# object ของแต่ละ provider เอง — ดู _build_history_digest() ด้านล่าง)
#
# เดิม (W7[A]) จำกัด scope แค่ Gemini เพราะ Anthropic/Groq มี message format คนละแบบ
# ต้องเขียน splicing แยกทีละตัว — ตัว digest (provider-agnostic อยู่แล้วเพราะอ่านจาก
# ShortTermMemory ไม่ใช่ raw messages) ใช้ร่วมกันได้ทั้ง 3 ตัว มีแค่ฟังก์ชัน splice
# raw messages (_compact_gemini_messages/_compact_anthropic_messages/
# _compact_groq_messages ด้านล่าง) ที่ต้องแยกตาม wire format ของแต่ละเจ้า — เลือกจาก
# _llm_backend() เหมือน next_action/append_tool_result ที่มีอยู่แล้ว (ดู
# Orchestrator._llm_backend())
_COMPACT_AFTER_STEPS = 6
_KEEP_RECENT_STEPS = 3

# W50 (delta history / bounded digest): เดิม _build_history_digest() ถูกเรียกซ้ำทุกรอบ
# compaction ด้วย upto_step ที่โตขึ้นเรื่อยๆ แต่ from_step เริ่มที่ 1 เสมอ (implicit) —
# แปลว่า task ที่ยาวพอจะมีหลายรอบ compaction (ทุก _COMPACT_AFTER_STEPS step) แต่ละรอบ
# สรุปซ้ำทุก step ตั้งแต่ต้น task ใหม่หมด ทำให้ digest text โตไม่มีเพดานตามความยาว task
# (ไม่ใช่ตามจำนวน step ใหม่ที่เพิ่งถูกตัดออกจริง) แล้วก้อนที่โตขึ้นเรื่อยๆ นี้ถูกส่งซ้ำเข้า
# LLM ทุก step ที่เหลือของ task จนกว่าจะ compact รอบถัดไป — เป็นสาเหตุจริงของ token cost
# ที่โตเร็วกว่า task เอง ไม่ใช่แค่โตตามสัดส่วนปกติ — แก้โดยสะสม "delta" เท่านั้น (ดู
# digest_lines/digest_upto_step ใน run_task()) แล้ว cap ด้วย _MAX_DIGEST_LINES กันไม่ให้
# โตไม่มีเพดานแม้จะเป็น delta ก็ตาม (task ที่ยาวมากๆ จริงๆ ก็ยังต้องมีเพดาน)
_MAX_DIGEST_LINES = 20


def _build_history_digest(memory: ShortTermMemory, upto_step: int, from_step: int = 1) -> str:
    """สรุป step from_step..upto_step (ไม่รวม step 0 ที่เป็น goto ตอนเริ่ม task) เป็น
    bullet list บรรทัดละ step สั้นๆ — สร้างจาก ShortTermMemory.all() เพราะเก็บ history
    แบบไม่ตัดทิ้งอยู่แล้วตลอด task จึงเป็นแหล่งความจริงที่สมบูรณ์กว่า raw messages ที่ถูก
    ตัดไปแล้ว — provider-agnostic (ไม่แตะ raw messages เลย) เลยใช้ร่วมกันได้ทั้ง
    Anthropic/Groq/Gemini

    from_step (W50, default 1 = พฤติกรรมเดิมทุกประการ): จุดเริ่มของช่วงที่จะสรุป — ใช้
    ตอน compaction รอบที่ 2 เป็นต้นไปเพื่อสรุปเฉพาะ step "ใหม่" ที่ยังไม่เคยถูกสรุปมาก่อน
    (ดู digest_upto_step ใน run_task()) แทนที่จะสรุปซ้ำตั้งแต่ step 1 ทุกรอบ"""
    entries = [h for h in memory.all() if from_step <= h.get("step", 0) <= upto_step]
    if not entries:
        return ""
    return "\n".join(f"- step {h['step']}: {h['cmd']} -> {h['result']}" for h in entries)


_DIGEST_PREFIX = "[สรุป step ก่อนหน้าที่ถูกย่อไว้กันบทสนทนายาวเกินไป]"


def _compact_gemini_messages(messages: list, cut_at: int, digest_text: str) -> list:
    """ตัด messages[:cut_at] ทิ้ง แล้วฝัง digest_text เข้าไปเป็นส่วนแรกของ text ใน
    turn แรกที่เหลืออยู่ (แทนที่จะแทรก turn ใหม่แยกต่างหาก) — messages[cut_at] ต้อง
    เป็น {"role": "user", "parts": [{"text": ...}]} เสมอ (จุดเริ่ม step ใหม่จาก
    next_action_gemini()) เพราะ cut_at มาจาก step boundary ที่ orchestrator เก็บเอง
    (ดู step_boundaries ใน run_task()) ไม่ใช่ตำแหน่งเดา — วิธีนี้ไม่ต้องแตะลำดับ
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
            "parts": [{"text": f"{_DIGEST_PREFIX}\n{digest_text}\n\n{original_text}"}],
        }
        return [new_first] + kept[1:]
    except (KeyError, IndexError, TypeError):
        return messages


def _compact_anthropic_messages(messages: list, cut_at: int, digest_text: str) -> list:
    """W22: เหมือน _compact_gemini_messages() ทุกประการแค่ shape ต่างกัน — Anthropic
    เก็บ user turn เป็น {"role": "user", "content": "<text ล้วนๆ>"} (ไม่ใช่ list of
    content block เหมือน tool_result/assistant turn) messages[cut_at] ต้องเป็น turn
    แบบนี้เสมอเพราะ cut_at มาจาก step boundary ที่บันทึกหลัง append_tool_result() พอดี
    (จุดเริ่ม step ถัดไปคือ user text turn จาก next_action() เสมอ) — ถ้ารูปแบบไม่ตรงคาด
    (เช่นโดน nudge message แทรกกลาง ทำให้ turn แรกไม่ใช่ plain text) คืน messages เดิม
    ไม่แก้อะไร ไม่ throw เหมือนกัน"""
    if cut_at <= 0 or not digest_text:
        return messages
    kept = messages[cut_at:]
    if not kept:
        return messages
    try:
        first = kept[0]
        original_text = first["content"]
        if first.get("role") != "user" or not isinstance(original_text, str):
            return messages
        new_first = {"role": "user", "content": f"{_DIGEST_PREFIX}\n{digest_text}\n\n{original_text}"}
        return [new_first] + kept[1:]
    except (KeyError, IndexError, TypeError, AttributeError):
        return messages


def _compact_groq_messages(messages: list, cut_at: int, digest_text: str) -> list:
    """W22: เหมือน _compact_anthropic_messages() แต่ Groq เก็บ system prompt เป็น
    messages[0] เอง (llm.next_action_groq(): "if not messages: messages = [{"role":
    "system", ...}]") ต่างจาก Anthropic/Gemini ที่ส่ง system แยกนอก messages เสมอ —
    ถ้าตัด messages[:cut_at] ตรงๆ แบบเดียวกับสองตัวบนจะกิน system message ทิ้งไปด้วย
    (cut_at มาจาก step boundary ที่บันทึก "หลัง" step แรกจบเสมอ ซึ่งมากกว่า index 0
    อยู่แล้ว) ต้องกัน messages[0] ไว้เสมอ ไม่ให้หลุดไปอยู่ใน "ส่วนที่ตัดทิ้ง"""
    if cut_at <= 0 or not digest_text or not messages:
        return messages
    if messages[0].get("role") != "system":
        return messages
    system_msg = messages[0]
    kept = messages[cut_at:]
    if not kept:
        return messages
    try:
        first = kept[0]
        original_text = first["content"]
        if first.get("role") != "user" or not isinstance(original_text, str):
            return messages
        new_first = {"role": "user", "content": f"{_DIGEST_PREFIX}\n{digest_text}\n\n{original_text}"}
        return [system_msg, new_first] + kept[1:]
    except (KeyError, IndexError, TypeError, AttributeError):
        return messages


def _make_dialog_handler(memory: ShortTermMemory, verbose: bool):
    """W9[A] "handle error states (popup)": auto-dismiss JS dialog (alert/confirm/
    prompt/beforeunload) — ถ้าไม่ handle เอง Playwright จะปล่อยให้ dialog ค้างบล็อก
    หน้าเว็บทั้งหมดจนกว่าจะมีใคร accept/dismiss เอง ทำให้ action ถัดไปทุกตัว timeout
    เงียบๆ โดยไม่มีใครรู้ว่าสาเหตุจริงคือ dialog ค้างอยู่ ไม่ใช่ DOM ยังไม่นิ่ง — เลือก
    dismiss เสมอ (ไม่ accept) เพราะปลอดภัยกว่า: confirm()/prompt() บางเว็บใช้คู่กับ
    action ทำลายข้อมูล (เช่น "แน่ใจนะว่าจะลบ?") การ accept ให้เองโดยไม่ถามมนุษย์ก่อนขัด
    กับหลัก human-in-the-loop ของ permission layer ทั้งระบบ — บันทึกเข้า short-term
    memory ด้วย (ผ่าน pipe เดียวกับ failed_actions_summary() ที่มีอยู่แล้วจาก W7[A]
    ไม่ต้องเพิ่ม context section ใหม่) ให้ LLM step ถัดไปรู้ตัวว่าเพิ่งมี dialog โผล่มา
    แล้วถูกปิดอัตโนมัติ เผื่อ dialog นั้นมีข้อความสำคัญ (เช่น error จากฟอร์ม)

    W12: page ที่มาจาก session (page= param ของ run_task()) ถูกใช้ซ้ำข้ามหลาย
    run_task() call — handler นี้ถูก page.on("dialog", ...) ผูกเพิ่มเข้าไปใหม่ทุกครั้งที่
    เรียก (ไม่มีทาง deregister handler ของเทิร์นก่อนหน้าได้ง่ายๆ ข้าม call แยกกัน) ทำให้
    dialog เดียวกันอาจโดนหลาย handler (จากคนละเทิร์น คนละ ShortTermMemory) เรียกพร้อมกัน
    — ตัวแรกที่เรียก dismiss() สำเร็จ ตัวถัดๆ ไปจะเจอ error เพราะ dialog ถูกจัดการไปแล้ว
    (ปกติของ Playwright) ห่อด้วย try/except กันไม่ให้ handler เก่าที่ค้างอยู่พัง task
    ปัจจุบันเงียบๆ (ไม่ใช่ error ที่ควร fail ทั้ง task)"""
    async def _handle_dialog(dialog):
        message = f"[POPUP] เจอ {dialog.type} dialog: '{dialog.message}' — ปิดอัตโนมัติแล้ว (dismiss)"
        if verbose:
            print(f"  {message}", flush=True)
        memory.record({
            "step": -1,  # ไม่ผูกกับ step ไหนโดยเฉพาะ (เกิดขึ้นได้ทุกเมื่อระหว่าง action)
            "cmd": {"type": "dialog", "dialog_type": dialog.type},
            "result": message,
            "success": False,
        })
        try:
            await dialog.dismiss()
        except Exception:
            pass  # dialog ถูก handler อื่น (เทิร์นก่อนหน้าที่ยังค้าง listener อยู่) จัดการไปแล้ว
    return _handle_dialog


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
    และยังว่างอยู่ไหม — ใช้เป็นสัญญาณว่า login form ยังกรอกไม่ครบ

    W19 (latency): .input_value() ไม่ระบุ timeout เองจะ default เป็น 30000ms ของ
    Playwright — ใส่ _DOM_CHECK_TIMEOUT_MS (3s) ตรงๆ กันรอนานเกินจำเป็นถ้า element หลุด/
    detach ระหว่างทาง"""
    try:
        password_inputs = page.locator('input[type="password"]:visible')
        count = await password_inputs.count()
        for i in range(count):
            value = await password_inputs.nth(i).input_value(timeout=_DOM_CHECK_TIMEOUT_MS)
            if value == "":
                return True
        return False
    except Exception:
        return False

# ระยะห่างต่ำสุดที่ต้องการระหว่างการเรียก next_action() (LLM) 2 ครั้งติดกัน กันยิง LLM
# API ถี่เกิน free-tier quota ต่อนาที (RPM) — ไม่ใช่แค่ Gemini เจอ 429 ResourceExhausted
# เอง (ดู llm.py) provider อื่นก็มี rate limit เหมือนกัน แค่ชื่อ error ต่างกัน ค่านี้เป็น
# heuristic คร่าวๆ ไม่ได้ผูกกับ quota จริงเป๊ะๆ ของ key ไหน (แต่ละ key/โมเดลจำกัดไม่เท่ากัน)
#
# W41: user รายงานว่า agent ใช้เวลานานเกินไปกว่าจะ "แจ้งสถานะเสร็จสิ้น" หลัง action สุดท้าย
# เสร็จจริงแล้ว — เดิม sleep(_STEP_PACING_DELAY_SECONDS) แบบตรงๆ ท้ายทุก step (ไม่ว่า step
# นั้นจะกินเวลาไปแล้วเท่าไหร่ก็ตามจาก execute()/wait_stable()/get_snapshot()/retrieve() ฯลฯ)
# บวกเพิ่มเข้าไปอีกทุกครั้งแบบ "ไม่หักลบ" เวลาที่ผ่านไปแล้วเลย — เปลี่ยนมาวัด wall-clock
# จริงตั้งแต่ next_action() ครั้งก่อนจบ แล้ว sleep แค่ส่วนที่ยังขาดให้ครบ
# _STEP_PACING_DELAY_SECONDS เท่านั้น (ดู last_llm_call_at ใน run_task()) — ยังการันตี
# ระยะห่างขั้นต่ำเท่าเดิมทุกประการ (ไม่ลดความปลอดภัยจาก rate-limit เลย) แค่ไม่เสียเวลาเปล่า
# ซ้ำกับงานที่ทำไปแล้วจริงระหว่าง step นั้น — ผลคือ step สุดท้ายก่อนจะรู้ว่า LLM ตัดสินใจ
# เรียก finish_task (ซึ่งงานจริงของ step ก่อนหน้ามักกินเวลาไปเกิน 3 วินาทีอยู่แล้วจาก
# wait_stable()/network) มักไม่ต้องรอเพิ่มเลยหรือรอสั้นลงมาก
_STEP_PACING_DELAY_SECONDS = 3

# W41: user รายงานว่า agent ใช้เวลานานเกินไปกว่าจะ "แจ้งสถานะเสร็จสิ้น" หลัง action
# สุดท้ายเสร็จจริงแล้ว — สาเหตุหนึ่งที่แก้ได้ตรงๆ ไม่มี trade-off เลย: long_term_memory.
# record_task() (บันทึกประวัติ task run ไว้ให้ task ถัดไป recall() ใช้ — ดู
# long_term_memory.py) เดิม await ตรงๆ ก่อน return ผลลัพธ์กลับไปเสมอ ทั้งที่เป็นแค่
# embedding + ChromaDB write ที่ "ไม่มีใครรอผลลัพธ์" เลย (คืน None, ไม่ throw ออกมาเอง
# อยู่แล้วตามสัญญาของ record_task() เอง) ทำให้ user เห็นสถานะ "เสร็จสิ้น" ช้าไปอีก
# เท่ากับเวลาที่ embedding+write ใช้จริงโดยไม่จำเป็น — ยิงเป็น background task แทน (ไม่
# await ก่อน return) เก็บ reference ไว้ใน _background_tasks กัน asyncio garbage collect
# ทิ้งก่อนทำงานเสร็จ (python เตือนเรื่องนี้ไว้ตรงๆ ใน asyncio.create_task() docs) ใช้
# add_done_callback ลบ reference ทิ้งเองอัตโนมัติเมื่อเสร็จแล้ว ไม่ต้องมีใครมาคอย clear
_background_tasks: set = set()


def _fire_and_forget(coro) -> None:
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


# W10[B]: callback (event dict) -> None ให้ชั้นบน (API server) รับรู้ความคืบหน้าสดๆ
# ระหว่าง loop กำลังรัน (ต่างจาก history ใน return value ท้าย run_task() ที่มาถึงทีเดียว
# ตอนจบเท่านั้น) — ใช้แพทเทิร์นเดียวกับ ask_user_func: optional, ไม่ส่งมาก็ไม่ทำอะไร
# (fallback เงียบๆ ไม่ throw) ไม่ผูกกับ transport ใดๆ (SSE/WebSocket เป็นเรื่องของชั้นบน)
OnEventFunc = Callable[[dict], Awaitable[None]]


# W11[A]: เปิด browser ที่มองเห็น (headless=False) ด้วย browser ตัวจริงที่ user ตั้งเป็น
# ค่าเริ่มต้นของเครื่อง (Chrome/Edge) แทน Chromium เปล่าๆ ที่ Playwright ติดตั้งมาเอง (ไม่มี
# bookmark/extension/login ของ user) — ตรวจผ่าน registry key เดียวกับที่ Windows ใช้ตอน
# double-click ไฟล์ .html/ลิงก์ (HKCU...UrlAssociations\https\UserChoice ProgId) แล้ว map
# เป็น Playwright "channel" (chromium.launch(channel=...) ใช้ binary ของ Chrome/Edge ที่
# ติดตั้งจริงในเครื่อง แทน bundled Chromium)
#
# รองรับแค่ Chrome/Edge เพราะทั้งคู่เป็น Chromium-based มี CDP ให้ Playwright เกาะควบคุมได้
# จริง — Safari ทำไม่ได้เลยไม่ว่า OS ไหน (ไม่มี Windows build ด้วย, ส่วน webkit ที่
# Playwright bundle มาเป็นคนละตัวกับ Safari.app จริง ไม่มี CDP ให้เกาะ) และ Firefox ต้อง
# ใช้ playwright.firefox คนละ browser type กับ chromium (นอก scope ตอนนี้) — เจอกรณีพวกนี้
# คืน None แล้วปล่อยให้ fallback ไป Chromium ของ Playwright เอง (ยังใช้งานได้ปกติ แค่ไม่ใช่
# แอปที่ user คุ้นเคย)
def _detect_default_browser_channel() -> Optional[str]:
    if sys.platform != "win32":
        return None
    try:
        import winreg
        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\Shell\Associations\UrlAssociations\https\UserChoice",
        )
        prog_id, _ = winreg.QueryValueEx(key, "ProgId")
    except OSError:
        return None
    prog_id = (prog_id or "").lower()
    if "chrome" in prog_id:
        return "chrome"
    if "edge" in prog_id:
        return "msedge"
    return None


async def _launch_chromium(playwright: Playwright, headless: bool, channel: Optional[str]) -> Browser:
    """channel: ผลจาก _detect_default_browser_channel() — ถ้า launch ด้วย channel ที่
    ระบุไม่สำเร็จ (เช่น ตรวจเจอว่า default browser คือ Chrome แต่เครื่องนี้ไม่ได้ติดตั้ง
    Chrome จริงๆ ติดตั้งแค่ Chromium/พาธเพี้ยน) fallback ไป Chromium ของ Playwright เอง
    เงียบๆ แทนที่จะทำให้ task ทั้งก้อนพังเพราะเรื่องเครื่องสำอาง (เลือกโชว์เบราว์เซอร์ไหน)"""
    if channel:
        try:
            return await playwright.chromium.launch(headless=headless, channel=channel)
        except Exception:
            pass
    return await playwright.chromium.launch(headless=headless)


async def _maybe_auto_login(page: Page, verbose: bool) -> Optional[str]:
    """W17: เติม username/password ให้อัตโนมัติถ้ามี credential เก็บไว้สำหรับโดเมนนี้แล้ว
    (จาก POST /api/site-manual/learn หรือ .../credentials — ดู site_learning/storage.py::
    save_credentials) และหน้าปัจจุบัน (หลัง goto/skip_initial_goto ตอนต้น run_task())
    เข้าข่ายเป็นหน้า login จริง (เจอ password field จริงจาก extract_page()) — ทำครั้งเดียว
    ตอนต้น task ก่อนเข้า loop หลัก กัน agent เสียเวลา/token กรอกฟอร์ม login เองทุกครั้งที่
    เจอเว็บเดิม ไม่เคยส่ง credential เข้า prompt/context ของ LLM เลย (fill/click ทำตรงๆ
    ผ่าน Playwright ก่อนที่ agent จะเห็นหน้าเลยด้วยซ้ำ)

    import แบบ lazy (ในฟังก์ชัน ไม่ใช่หัวไฟล์) โดยเจตนา — site_learning/crawler.py เอง
    import core/orchestrator.py อยู่แล้ว (ใช้ Orchestrator._llm_backend()) ถ้า import จาก
    site_learning ไว้หัวไฟล์นี้จะเกิด circular import ตอนโหลดโมดูลทันที เลื่อนมา import ตอน
    เรียกจริง (runtime, หลังทั้งสองโมดูลโหลดเสร็จแล้ว) แก้ปัญหานี้โดยไม่ต้องแตะโครงสร้าง
    site_learning/__init__.py เลย

    คืน None ถ้าไม่มี credential เก็บไว้ / หน้าปัจจุบันไม่ใช่หน้า login / login สำเร็จจริง
    (ไม่ต้องแจ้ง user อะไรเลย) คืนข้อความเหตุผล (mask password เสมอ ไม่มีรหัสผ่านปนออกมา
    เด็ดขาด) ถ้าเจอ credential + เป็นหน้า login จริง แต่ login ไม่ผ่านแม้ retry แล้ว — ให้
    caller (run_task ด้านล่าง) ยิง SSE event แจ้ง user ต่อ ไม่ throw ในทุกกรณี (agent ยัง
    fallback ไปกรอกเองผ่าน action ปกติได้อยู่แล้วถ้า auto-login ไม่สำเร็จ — แค่ต้องให้ user
    รู้ตัวว่า credential ที่บันทึกไว้ใช้ไม่ได้แล้ว ไม่ใช่ปล่อยผ่านเงียบๆ เหมือนเดิม)"""
    try:
        from backend.app.site_learning import storage as site_storage
        from backend.app.site_learning.auto_login import find_login_fields, login_with_verification
        from backend.app.site_learning.extractor import extract_page as site_extract_page

        domain = extract_domain(page.url)
        creds = site_storage.load_credentials(domain)
        if not creds:
            return None
        page_info, _ = await site_extract_page(page)
        username_selector, password_selector = find_login_fields(page_info)
        if not username_selector or not password_selector:
            return None
        if verbose:
            print(f"[auto-login] พบ credential ที่เก็บไว้สำหรับ {domain} — ลอง login อัตโนมัติ", flush=True)
        # retries=1: ลองซ้ำอีก 1 ครั้งถ้ารอบแรกไม่ผ่าน (เช่น หน้าโหลดช้า/DOM ยังไม่นิ่งตอน
        # fill รอบแรก) ก่อนยอมรับว่า login ไม่ผ่านจริง — login_with_verification() ตรวจ
        # session_ok จริง (URL เปลี่ยน + ไม่เจอฟอร์ม login เหลือ) ไม่ใช่แค่ "กด submit ได้"
        did_login, reason = await login_with_verification(
            page, page_info, creds["username"], creds["password"], retries=1,
        )
        if not did_login:
            return reason or "ล็อกอินไม่สำเร็จด้วย credential ที่บันทึกไว้สำหรับเว็บนี้"
        await wait_stable(page)
        return None
    except Exception:
        return None


def _tokens_dict(usage: llm.TokenUsage) -> dict:
    return {
        "input": usage.input_tokens,
        "output": usage.output_tokens,
        "cache_read": usage.cache_read_tokens,
        "cache_creation": usage.cache_creation_tokens,
    }


async def _confirm_plan(plan_text: str, ask_user_func: Optional[AskUserFunc]) -> tuple[bool, str]:
    """โชว์แผนแล้วรอ user ยืนยันก่อนเริ่ม loop จริง — ใช้ callback เดียวกับ permission
    layer (actions.AskUserFunc) เพื่อให้ชั้นบน (เช่น API server ใน W10) inject วิธีถาม
    ของตัวเองได้ (ส่ง event ไป UI แทน blocking input() ทาง terminal) โดยไม่ต้องแก้ตรงนี้

    คืนค่า (approved, plan_text) — W10[F]: plan_text ที่คืนอาจไม่ใช่ตัวเดิมที่ส่งเข้ามา
    ถ้า user แก้ไขข้อความแผนก่อนกด Confirm (ดู routes.py::respond_task ->
    TaskManager.resolve_approval(edited_plan=...) ที่ mutate key "plan" ใน cmd dict
    ก้อนเดียวกับที่เรา await อยู่นี้ตรงๆ ก่อน future resolve กลับมา — ต้องอ่านจาก cmd
    หลัง await เสร็จแล้ว ไม่ใช่เชื่อตัวแปร plan_text เดิมที่ปิด scope ไปแล้วตอนส่งเข้า
    ask_user_func) ให้ caller (run_task) เอาไปใช้แทนแผนเดิมที่ AI ร่างไว้เอง"""
    if ask_user_func is not None:
        cmd = {"type": "confirm_plan", "plan": plan_text}
        approved = bool(await ask_user_func(cmd))
        return approved, cmd.get("plan", plan_text)
    print("\n=== แผนที่ AI จะทำ ===", flush=True)
    print(plan_text, flush=True)
    print("========================", flush=True)
    choice = await asyncio.to_thread(input, "ยืนยันให้เริ่มทำงานตามแผนนี้หรือไม่? (y/n): ")
    return choice.strip().lower() in ("y", "yes"), plan_text


class Orchestrator:
    def __init__(self):
        self.memory = ShortTermMemory()

    @staticmethod
    def _llm_backend(provider: str):
        """เลือก client/model/next_action/append_tool_result/compact_messages ตาม
        provider รองรับ "anthropic" (ตัวหลักตาม roadmap), "gemini" (provider สำรอง
        free tier กว้างกว่า) และ "groq" (ไว้ทดสอบตอนยังไม่มี Anthropic key จริง) —
        คืนรูปแบบเดียวกันหมดให้ loop ข้างล่างเรียกแบบไม่ต้องรู้ว่าเป็น provider ไหน
        (W22: เพิ่ม compact_messages เข้าชุดนี้ด้วย — เดิม context compaction เคย
        hardcode เฉพาะ Gemini ในตัว loop เอง ตอนนี้ generalize ผ่าน dispatch ตรงนี้แทน
        เหมือน next_action/append_tool_result ทุกประการ)
        """
        if provider == "groq":
            return (
                llm.build_groq_client(settings.groq_api_key),
                settings.groq_model,
                llm.next_action_groq,
                llm.append_tool_result_groq,
                _compact_groq_messages,
            )
        if provider == "gemini":
            return (
                llm.build_gemini_client(settings.gemini_api_key),
                settings.gemini_model,
                llm.next_action_gemini,
                llm.append_tool_result_gemini,
                _compact_gemini_messages,
            )
        if provider == "anthropic":
            return (
                llm.build_client(settings.anthropic_api_key),
                settings.anthropic_model,
                llm.next_action,
                llm.append_tool_result,
                _compact_anthropic_messages,
            )
        raise ValueError(f"ไม่รู้จัก LLM provider: {provider!r} (รองรับแค่ anthropic/gemini/groq)")

    async def generate_plan(
        self, url: str, goal: str, provider: Optional[str] = None, page: Optional[Page] = None,
        site_manual_context: str = "", previous_user_goal: str = "", previous_assistant_message: str = "",
    ) -> tuple[str, bool]:
        """W13: ร่างแผนคร่าวๆ (llm.generate_plan) แยกเป็นเฟสของตัวเอง ไม่ผูกกับ
        run_task() เลย — ต่างจาก confirm_plan=True เดิมที่ต้อง acquire/launch/connect
        browser ก่อนแล้วค่อย goto+perceive มาร่างแผน ฟังก์ชันนี้ "ไม่เปิด/ไม่ connect
        อะไรเองเด็ดขาด": ถ้า caller (routes.py::generate_plan endpoint) ส่ง page มาให้
        (เช่น session ที่มี page เปิดค้างอยู่แล้วจากเทิร์นก่อนหน้า) จะ perceive หน้านั้น
        จริงเพื่อร่างแผนที่ grounded กับสถานะปัจจุบัน — ถ้าไม่ส่งมา (None, เช่น
        session_id ยังไม่เคยมี page เลย) จะร่างแผนจาก goal เพียวๆ (page_text="") ไม่มี
        browser เกี่ยวข้องในฟังก์ชันนี้เลยไม่ว่ากรณีไหน

        site_manual_context (W14): เนื้อหาย่อจากคู่มือเว็บไซต์ที่ crawl มาอัตโนมัติ (ดู
        backend/app/site_learning/) — ผู้เรียก (routes.py) ดึงมาเองจาก
        site_learning.storage.load_knowledge_text(domain) ก่อนเรียกฟังก์ชันนี้ ถ้ามีจะ
        แปะไว้ก่อน page_text ให้ LLM เห็นโครงสร้างเว็บที่รู้จักอยู่แล้วตอนร่างแผน (ไม่ต้อง
        เดาจาก page_text อย่างเดียว) ว่างเปล่า (default) ถ้าโดเมนนี้ยังไม่เคยถูกเรียนรู้

        คืนค่าเป็น tuple (plan_text, is_qa) — ถ้า Intent เป็น qa_summary จะคืน (" ", True)
        เพื่อให้ frontend ข้ามหน้าต่างอนุมัติ PLAN แล้วตอบคำถามได้ทันที

        previous_user_goal/previous_assistant_message (W20, "Context-Aware Implicit
        Execution"): เทิร์นก่อนหน้าล่าสุดในเซสชันเดียวกัน (ถ้ามี — ผู้เรียกส่งมาจาก
        conversation history ฝั่ง client เอง ดู routes.py::generate_plan endpoint docstring)
        ส่งต่อเข้า llm.generate_plan() ตรงๆ ให้ LLM แก้คำอ้างอิงกำกวมอย่าง "เปิดให้หน่อย"/
        "play it" ได้ แม้เทิร์นก่อนหน้าจะเป็น general-chat ล้วนๆ ที่ไม่เคยแตะ browser/session
        เลยก็ตาม (route_multi_turn_strategy/session.extracted_memory ด้านล่างจับ entity จาก
        เทิร์นแบบนี้ไม่ได้เลย เพราะมันทำงานเฉพาะกับ session ที่มี page เปิดอยู่จริงและเคยมี
        read_page_data สำเร็จเท่านั้น) ว่างเปล่าได้ทั้งคู่ (default) ถ้าเป็นเทิร์นแรกของ
        session หรือไม่มีเทิร์นก่อนหน้าจริงๆ"""
        resolved_provider = provider or settings.llm_provider
        client, model, _, _, _ = self._llm_backend(resolved_provider)
        page_text = ""
        # W19 (Navigation Deduplication): URL จริงของหน้าที่ perceive สำเร็จ (page.url ถ้ามี
        # page เปิดอยู่จริง) ไม่ใช่ url param ดิบที่ user พิมพ์มาตอนแรก (session ที่มี page
        # เปิดค้างจากเทิร์นก่อนอาจอยู่คนละหน้ากับ url param แล้วจริงๆ — page.url สะท้อน
        # สถานะปัจจุบันจริงเสมอ) ว่างเปล่าถ้า perceive ไม่สำเร็จ/ไม่มี page เลย
        current_url_for_plan = ""
        if page is not None:
            try:
                _, page_text = await get_snapshot(page)
                current_url_for_plan = page.url
            except Exception as e:
                print(f"⚠️ generate_plan: get_snapshot ล้มเหลว ({e!r}) — ใช้ page_text ว่างแทน", flush=True)

        user_intent = await llm.classify_intent(client, model, goal, page_text=page_text, provider=resolved_provider)
        if user_intent == "qa_summary":
            return "", True

        if site_manual_context:
            page_text = f"[คู่มือเว็บไซต์ที่เรียนรู้มาก่อนแล้ว]\n{site_manual_context}\n\n{page_text}".strip()
        plan_text = await llm.generate_plan(
            client, model, goal, page_text, resolved_provider, current_url=current_url_for_plan,
            previous_user_goal=previous_user_goal, previous_assistant_message=previous_assistant_message,
        )
        return plan_text, False


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
        browser: Optional[Browser] = None,
        on_event: Optional[OnEventFunc] = None,
        keep_browser_open: bool = False,
        connect_to_user_browser: bool = False,
        user_browser_cdp_url: Optional[str] = None,
        allowed_domains: Optional[set] = None,
        tab_reuse_policy: Optional[str] = None,
        page: Optional[Page] = None,
        approved_plan: Optional[str] = None,
        site_manual_context: str = "",
        session_id: Optional[str] = None,
    ) -> dict:
        """Perceive -> Plan -> Act loop บนหน้าเว็บเดียว จนกว่า LLM จะเรียก finish_task
        หรือครบ max_steps

        session_id (W23): ส่งต่อให้ long_term_memory.recall()/record_task() ใช้ scope
        ความจำข้าม task run ให้อยู่แค่ภายใน session เดียวกัน (ดู core/long_term_memory.py
        module docstring) — ไม่ระบุมา (None) = ไม่มี session context ให้ scope ปลอดภัยได้
        recall() จะคืน [] เสมอ (ไม่ query แบบไม่กรองเด็ดขาด) ส่วน record_task() ยังบันทึก
        ได้ปกติแค่ session_id ว่างเปล่า (recall กลับมาไม่เจอในทางปฏิบัติ)

        headless: None = ใช้ settings.browser_headless, True/False = บังคับ override
                  (เช่น run.py agent อยากเห็นหน้าต่าง browser จริงๆ ระหว่างรัน) — ไม่มีผล
                  ถ้าส่ง browser เข้ามาเอง (เพราะ browser launch ไปแล้วตั้งแต่ตอนเปิด pool)
        verbose:  True = print แต่ละ step ลง terminal สดๆ ระหว่าง loop (ไว้ดูคู่กับ
                  หน้าต่าง browser ที่เปิดโชว์อยู่) — ปิดไว้ (False) ตอนเรียกจาก
                  API server (W10) กัน log รก
        provider: None = ใช้ settings.llm_provider, หรือระบุ "anthropic"/"groq" ตรงๆ
        ask_user_func: callback (cmd/plan dict) -> bool ให้ชั้นบน (เช่น API server)
                  ตัดสินใจแทน blocking input() ทาง terminal — ใช้ร่วมกันทั้ง permission
                  layer (actions.execute) และ confirm_plan ด้านล่าง ถ้าไม่ส่งมา fallback
                  เป็น input() ทาง terminal ทั้งคู่
        confirm_plan: True = ก่อนเริ่ม loop จริง ให้ LLM ร่างแผนคร่าวๆ (llm.generate_plan)
                  โชว์ให้ user เห็นแล้วรอกดยืนยันก่อน — ถ้าไม่ยืนยัน จะไม่ลงมือทำ action
                  ใดๆ เลย (คืนผลลัพธ์ steps=0 ทันที) ไว้กัน agent เริ่มทำอะไรที่ user ยัง
                  ไม่ได้เห็นแผนมาก่อน
        on_event: W10[B] — callback (event dict) -> None ให้ API server สตรีมความคืบหน้า
                  สดๆ ระหว่าง loop กำลังรัน (goto ตอนเริ่ม + ทุก step ที่ execute()
                  จริง) ไปหน้าเว็บได้แบบ real-time แทนที่จะรอ poll ผลลัพธ์รวมท้าย task
                  เดียว — ไม่ส่งมาก็ไม่ทำอะไร (ค่าเดิมของ W1-W9)
        keep_browser_open: W10[C] — True = ไม่ปิด browser window ตอนจบ task (finish_task/
                  loop-detected/max_steps/cancelled ทุก path) ปล่อยให้ user ปิดหน้าต่างเอง
                  ทีหลัง — มีผลเฉพาะตอน owns_browser=True (ไม่ได้ยืม browser จาก pool มา
                  เพราะ context ที่ยืมจาก pool ต้องคืนกลับเสมอให้ task อื่นใช้ต่อได้ ไม่งั้น
                  pool จะรั่วทีละ context ทุก task ที่ตั้งค่านี้) — ปกติใช้คู่กับ headless=
                  False เท่านั้น (เปิด browser แบบไม่ซ่อนหน้าต่างค้างไว้ให้ user เฝ้าดูต่อ
                  หลังงานเสร็จ ถ้าเป็น headless=True ด้วยจะแค่รั่ว process เปล่าๆ ไม่มี
                  ประโยชน์ — เป็นหน้าที่ของผู้เรียก (routes.py) ที่จะไม่ตั้ง flag คู่นี้ผิดกัน)
        browser: W10[A] — ถ้าไม่ส่งมา (None, ค่าเดิมของ W1-W9) เปิด/ปิด playwright +
                  browser process เองทั้งหมดเหมือนเดิมทุกประการ ถ้าส่งมา (ยืมมาจาก
                  core/browser_pool.py::BrowserPool.acquire() — ตัว browser เป็น process
                  ที่เปิดค้างไว้ล่วงหน้า reuse ข้าม task ได้) จะเปิดแค่ BrowserContext
                  ใหม่ (session แยกต่างหาก ไม่แชร์ cookie/localStorage กับ task อื่นที่ยืม
                  browser ตัวเดียวกัน) แล้วปิดแค่ context ตอนจบ ไม่ปิด/ไม่ stop
                  playwright ของ browser ที่ยืมมา (ผู้ให้ยืม คือ BrowserPool เป็นคนคุม
                  lifecycle ของตัว browser process เอง)
        connect_to_user_browser: ต่อเข้า Chrome จริงที่ user เปิดใช้งานอยู่แล้ว (มี
                  cookie/login ค้างอยู่จริง เช่น mail) ผ่าน CDP (ดู core/user_browser.py)
                  แทนที่จะ launch Chromium ว่างๆ เอง — mutually exclusive กับ param
                  `browser` ด้านบน (ส่งมาพร้อมกันทั้งคู่จะ raise ValueError ทันที) ใช้
                  BrowserContext เดิมของ user จริง (browser.contexts[0]) ไม่เคย
                  new_context()/close() บน browser จริงเด็ดขาด — ปิดแค่ tab ที่ agent
                  เปิดเอง (ถ้าเปิดจริง) ตอนจบ task เท่านั้น
        user_browser_cdp_url: None = ใช้ settings.user_browser_cdp_url — มีผลเฉพาะตอน
                  connect_to_user_browser=True
        allowed_domains: จำกัด goto/navigation ให้อยู่แค่โดเมนในนี้เท่านั้น (ส่งต่อเข้า
                  actions.execute()/classify_action() ทุก step ของ task นี้) ไม่ใช่ของ
                  connect_to_user_browser โดยเฉพาะ (ใช้กับ owns_browser/pool ปกติได้ด้วย)
                  แต่เป็น use case หลัก — ถ้า connect_to_user_browser=True และไม่ระบุมา
                  (None) จะ auto-derive เป็น {extract_domain(url)} ให้เอง (default-deny
                  ทุกโดเมนอื่นแม้ผู้เรียกลืมระบุ กันไม่ให้ agent หลุดไปแตะ session อื่นที่
                  login ไว้ในเครื่องเดียวกัน เช่น mail) None บน owns_browser/pool ปกติ
                  หมายถึง "ไม่จำกัด" (พฤติกรรมเดิมทุกประการ)
        tab_reuse_policy: "ask"(default)/"always_new_tab"/"always_reuse" — มีผลเฉพาะตอน
                  connect_to_user_browser=True (ดู core/user_browser.py::
                  resolve_target_page) None = ใช้ settings.user_browser_tab_reuse_policy
        page: session-managed page — ผู้เรียก (core/session_registry.py::SessionRegistry
                  ผ่าน routes.py) resolve หน้าเว็บที่จะใช้ไว้ให้แล้วเองล่วงหน้า (อาจมาจาก
                  pool/owns/CDP โหมดไหนก็ได้ แต่ resolve ไปแล้วครั้งเดียวตอน session ถูก
                  สร้าง ไม่ใช่ทุกครั้งที่เรียก run_task()) — mutually exclusive กับทั้ง
                  `browser` และ `connect_to_user_browser` (ส่งมาพร้อมกันจะ raise
                  ValueError ทันที) เมื่อส่งมา run_task() จะไม่ acquire/launch/connect
                  อะไรเองเลย และจะไม่ปิด/คืนอะไรตอนจบ task ด้วย (session registry เป็นคน
                  คุม lifecycle เต็มๆ ข้ามหลาย run_task() call จนกว่า user จะปิด session
                  เอง) — ใช้คู่กับ "detect หน้าปัจจุบัน" ด้านล่าง (skip_initial_goto)
                  เพื่อให้ turn ถัดไปในบทสนทนาเดียวกันทำงานต่อจากหน้าที่ turn ก่อนทิ้งไว้
                  แทนที่จะโหลดหน้าแรกซ้ำเหมือนเริ่มใหม่ทั้งหมด
        approved_plan: W13 — แผนที่ user อนุมัติแล้ว (อาจแก้ไขข้อความมาก่อน) จากเฟส
                  วางแผนแยกต่างหาก (ดู generate_plan() ด้านบน + routes.py::
                  POST /api/generate_plan) — ต่างจาก confirm_plan ด้านบนตรงที่ไม่มีการ
                  เรียก LLM ร่างแผน/รอ ask_user_func ข้างในนี้เลย (อนุมัติไปแล้วตั้งแต่
                  ก่อนเรียก run_task()) แค่ผนวกเข้า effective_goal ทันทีแล้วเริ่ม loop
                  จริงเลย — mutually exclusive กับ confirm_plan=True (ส่งมาพร้อมกันจะ
                  raise ValueError ทันที เพราะเป็นคนละกลไกกันสำหรับจุดประสงค์เดียวกัน)
                  None (default) = ไม่มีแผนที่อนุมัติมาก่อน ทำงานตาม goal เดิมตรงๆ (หรือ
                  ตาม confirm_plan ถ้าตั้งไว้)
        site_manual_context: W14 — เนื้อหาย่อจากคู่มือเว็บไซต์ที่ crawl มาอัตโนมัติ (ดู
                  backend/app/site_learning/) ส่งเข้า llm.next_action() ทุก step เป็น
                  section แยกจาก manual_context (ที่มาจากคู่มือ user อัปโหลดเองผ่าน
                  RAG/ChromaDB — คนละระบบกันสมบูรณ์) — ผู้เรียก (routes.py) เป็นคนดึงจาก
                  site_learning.storage.load_knowledge_text(domain) มาเองครั้งเดียวก่อน
                  เรียก run_task() ไม่ใช่ orchestrator.py ไปโหลดเอง (เหมือน pattern เดียว
                  กับ page= — เก็บ orchestrator.py ให้ไม่ต้องรู้จัก storage โดยตรง) ค่า
                  คงที่ตลอด task เดียว ไม่ re-fetch ทุก step แบบ manual_context (เพราะ
                  ไม่ได้ผูกกับ page state ปัจจุบันที่เปลี่ยนไปเรื่อยๆ) ว่างเปล่า (default)
                  ถ้าโดเมนนี้ยังไม่เคยถูกเรียนรู้/ไม่มี manual เลย

        W12: "detect หน้าปัจจุบัน" แทนการบังคับ goto(url) เสมอ — หลัง resolve page ได้
        แล้ว (ไม่ว่าจากโหมดไหน) เช็ค page.url ตรงๆ ก่อนตัดสินใจ: ถ้ายังเป็น "about:blank"/
        ว่างเปล่า (หน้าใหม่ที่เพิ่งเปิด ยังไม่มีอะไรให้ perceive) จะ goto(url) ตามปกติ แต่ถ้า
        page.url มีเนื้อหาจริงอยู่แล้ว (session ที่ reuse หน้ามาจาก turn ก่อนหน้า หรือ tab
        ที่ resolve_target_page() เลือก reuse มา) จะข้าม goto ไปเลย ปล่อยให้ agent
        perceive หน้าปัจจุบันตรงๆ แล้วเริ่ม loop ต่อจากจุดเดิม — ถ้า agent ประเมินเองว่า
        ต้อง navigate จริงๆ ก็มี action "goto" ให้เรียกเองได้อยู่แล้วในทุก step ปกติ

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
        if connect_to_user_browser and browser is not None:
            raise ValueError(
                "run_task() รับ connect_to_user_browser=True พร้อมกับ browser param ไม่ได้ "
                "— ทั้งคู่เป็นคนละแหล่งของ browser (CDP ต่อเข้า Chrome จริงของ user vs. "
                "ยืมมาจาก BrowserPool) เลือกอย่างใดอย่างหนึ่ง"
            )
        if page is not None and (browser is not None or connect_to_user_browser):
            raise ValueError(
                "run_task() รับ page= พร้อมกับ browser=/connect_to_user_browser=True "
                "ไม่ได้ — page= หมายความว่า caller (เช่น core/session_registry.py::"
                "SessionRegistry ผ่าน routes.py) resolve หน้าเว็บที่จะใช้ไว้ให้แล้วเอง "
                "ไม่ต้องให้ run_task() ไป acquire/launch/connect หา browser/page เองอีก"
            )
        if approved_plan and confirm_plan:
            raise ValueError(
                "run_task() รับ approved_plan พร้อมกับ confirm_plan=True ไม่ได้ — ทั้งคู่"
                "เป็นกลไกขออนุมัติแผนคนละแบบสำหรับจุดประสงค์เดียวกัน (approved_plan = "
                "อนุมัติไปแล้วจากภายนอกก่อนเรียก run_task(), confirm_plan=True = ให้ "
                "run_task() ร่างแผน+รอ ask_user_func เองข้างใน) เลือกอย่างใดอย่างหนึ่ง"
            )

        is_headless = settings.browser_headless if headless is None else headless
        resolved_provider = provider or settings.llm_provider
        client, model, next_action, append_tool_result, compact_messages = self._llm_backend(resolved_provider)

        async def _emit(event: dict) -> None:
            if on_event is not None:
                await on_event(event)

        async def _emit_screenshot(step: int) -> None:
            """W_live: ถ่าย screenshot ของ page ปัจจุบันส่งเป็น SSE event "screenshot"
            ให้ Test Console แสดง live view ระหว่าง task รันอยู่ (แทนที่จะต้องสลับไปดู
            หน้าต่างเบราว์เซอร์จริงเอง) — best-effort ล้วนๆ ไม่ throw เด็ดขาด ถ้า
            screenshot ล้มเหลว (เช่น page กำลัง navigate/ปิดพอดี) แค่ข้ามเงียบๆ ไม่ใช่
            error ที่ควร fail ทั้ง step/task จริง — jpeg quality ต่ำ (55) ตั้งใจให้ไฟล์เล็ก
            พอส่งผ่าน SSE ทุก step โดยไม่หน่วง loop มาก ไม่ใช่ไว้ดูรายละเอียดคมชัด

            is_headless=False (uncheck "Headless" บน Test Console) แปลว่า user ตั้งใจเปิด
            หน้าต่าง browser จริงให้เห็น (เช่น จะแก้ CAPTCHA เอง) — ข้าม live view ไปเลยใน
            เคสนี้ ไม่ต้อง stream screenshot ซ้ำซ้อนกับหน้าต่างจริงที่เปิดโชว์อยู่แล้ว"""
            if on_event is None or not is_headless:
                return
            try:
                raw = await page.screenshot(type="jpeg", quality=55)
            except Exception:
                return
            b64 = base64.b64encode(raw).decode("ascii")
            await _emit({
                "kind": "screenshot", "step": step,
                "image": f"data:image/jpeg;base64,{b64}",
            })

        async def _force_loop_recovery(reason: str) -> bool:
            """W31: เรียกตอน loop-detection guard (คาบ 1 หรือคาบ 2-4 — ดู
            _MAX_CONSECUTIVE_IDENTICAL_ACTIONS/_detect_repeating_cycle_period ด้านบนสุด
            ของไฟล์) trigger — บังคับทำ recovery action (go_back ก่อน แล้ว scroll ถ้ายัง
            ไม่หาย — ดู _LOOP_RECOVERY_ACTIONS) แทน action ที่ agent เพิ่งขอไป โดยไม่ผ่าน
            การตัดสินใจของ LLM รอบนี้เลย แทนที่จะจบ task ทันทีเหมือนเดิม — คืน True ถ้า
            บังคับสำเร็จ (caller ควร continue loop ต่อ ให้ agent ลองใหม่จาก state หลัง
            recovery) คืน False ถ้าเกิน _MAX_FORCED_LOOP_RECOVERIES แล้ว (caller ควร
            fallback ไปจบ task แบบเดิม — escape valve กัน force ไม่รู้จบถ้า forcing เองก็
            ไม่ช่วยอะไร)

            ผลลัพธ์ของ recovery action ถูกป้อนกลับเข้า messages ผ่าน tool_use_id ของ
            action เดิมที่ agent เพิ่งขอ (ต้องตอบทุก tool_use ด้วย tool_result เสมอ ไม่งั้น
            Anthropic/Groq API จะ error) พร้อมอธิบายตรงๆ ว่าเกิดอะไรขึ้น ไม่ใช่แกล้งทำเป็น
            ว่า action เดิมที่ agent ขอไปสำเร็จ"""
            nonlocal forced_recovery_count, messages, last_action_cmd, consecutive_repeat_count, steps_taken
            if forced_recovery_count >= _MAX_FORCED_LOOP_RECOVERIES:
                return False
            forced_cmd = _LOOP_RECOVERY_ACTIONS[min(forced_recovery_count, len(_LOOP_RECOVERY_ACTIONS) - 1)]
            forced_recovery_count += 1
            forced_result: ActionResult = await execute(
                page, forced_cmd, ask_user_func=ask_user_func, label="",
                manual_guidance="", allowed_domains=effective_allowed_domains,
            )
            steps_taken += 1
            self.memory.record({
                "step": steps_taken,
                "cmd": forced_cmd,
                "label": "[ระบบบังคับ - กันวนซ้ำ]",
                "result": str(forced_result),
                "success": forced_result.success,
                "tokens": _tokens_dict(llm.TokenUsage()),
            })
            await _emit({
                "kind": "step", "step": steps_taken, "cmd": forced_cmd,
                "label": "[ระบบบังคับ - กันวนซ้ำ]",
                "result": str(forced_result), "success": forced_result.success,
            })
            forced_text = (
                f"[ระบบตรวจพบการวนซ้ำ: {reason} — ระบบจึงบังคับทำ {forced_cmd} แทน action ที่"
                f"คุณเพิ่งขอโดยอัตโนมัติ (ครั้งที่ {forced_recovery_count}/"
                f"{_MAX_FORCED_LOOP_RECOVERIES}) ผลลัพธ์: {forced_result}] ตรวจสอบ URL ปัจจุบัน"
                "และ indexed elements ของหน้าเว็บใหม่ทั้งหมด แล้วเลือก action ที่ต่างจากที่วน"
                "ซ้ำมาก่อนหน้านี้จริงๆ"
            )
            messages = append_tool_result(messages, tool_use_id, forced_text)
            last_action_cmd = forced_cmd
            consecutive_repeat_count = 1
            recent_actions.append(forced_cmd)
            if len(recent_actions) > _MAX_CYCLE_WINDOW:
                recent_actions.pop(0)
            return True

        # W10[A]: owns_browser=True (browser ไม่ได้ถูกส่งมา, ไม่ใช่โหมด user browser, ไม่ใช่
        # โหมด session-managed) = พฤติกรรมเดิมของ W1-W9 เปิด/ปิด playwright + browser
        # process เองทั้งหมด — owns_browser=False (ยืมมาจาก BrowserPool) เปิดแค่ context
        # ใหม่บน browser ที่มีอยู่แล้ว แล้วปิดแค่ context ตอนจบ (ดู finally ท้าย method —
        # browser process เป็นของ pool ไม่ใช่ของ task นี้) — connect_to_user_browser=True
        # เป็น branch แยกต่างหาก (ดูด้านล่าง) ไม่นับเป็น owns_browser เพราะ browser จริงของ
        # user ไม่มีวันถูกปิดจากโค้ดฝั่งนี้เด็ดขาด — managed_externally=True (page ถูกส่ง
        # เข้ามาแล้ว) เป็น branch ที่ 4: ไม่ต้อง acquire/launch/connect อะไรเองเลย แล้วก็
        # ไม่ปิด/คืนอะไรตอนจบด้วย (ผู้เรียกเป็นคนคุม lifecycle เต็มๆ ข้ามหลาย call)
        managed_externally = page is not None
        owns_browser = browser is None and not connect_to_user_browser and not managed_externally
        playwright = None
        context = None
        opened_new_tab = False
        effective_allowed_domains = allowed_domains
        browser_channel = _detect_default_browser_channel() if (owns_browser and not is_headless) else None
        # W11[A]: ถ้าจะเปิดหน้าต่างให้เห็น (is_headless=False) *และ* ต้องรอ user ยืนยัน
        # แผนก่อน (confirm_plan=True) — อย่าเพิ่งเปิดหน้าต่างจริงตอนนี้ ไปเปิดแบบซ่อน
        # (headless=True ชั่วคราว) เพื่อไป goto+อ่านหน้าเว็บมาร่างแผนเท่านั้น แล้วค่อยเปิด
        # หน้าต่างจริงทีหลัง *หลัง* จากที่ user กด "Confirm & start" แล้วเท่านั้น (ดูจุด
        # relaunch ด้านล่าง หลัง _confirm_plan) — ไม่งั้นหน้าต่าง browser จะเด้งขึ้นมาโชว์
        # การ navigate ไปหน้าเว็บเป้าหมายให้ user เห็นก่อนที่ user จะกดยืนยันด้วยซ้ำ ทั้งที่
        # ในตอนนั้น user ยังไม่ได้ตกลงจะให้ agent เริ่มทำงานเลย
        defer_visible_window = owns_browser and confirm_plan and not is_headless
        if managed_externally:
            pass  # page ถูก resolve มาให้แล้ว ไม่ต้องทำอะไรเพิ่ม
        elif connect_to_user_browser:
            playwright = await async_playwright().start()
            browser = await connect_user_browser(
                playwright, user_browser_cdp_url or settings.user_browser_cdp_url,
            )
            # ห้าม browser.new_context() เด็ดขาด — ต้องใช้ context จริงที่มี cookie/login
            # ของ user อยู่แล้ว (contexts[0]) ไม่ใช่ context ว่างเปล่าใหม่
            context = browser.contexts[0]
            resolved_tab_reuse_policy = tab_reuse_policy or settings.user_browser_tab_reuse_policy
            page, opened_new_tab = await resolve_target_page(
                context, url, ask_user_func, resolved_tab_reuse_policy,
            )
            if effective_allowed_domains is None:
                # default-deny ทุกโดเมนอื่นนอกจาก target ของ task นี้เอง แม้ผู้เรียกลืม
                # ระบุ allowed_domains มาเอง — กัน agent หลุดไปแตะ session อื่นที่ login
                # ค้างไว้ในเครื่องเดียวกัน (เช่น mail) โดยไม่ตั้งใจ
                effective_allowed_domains = {extract_domain(url)}
        elif owns_browser:
            playwright = await async_playwright().start()
            browser = await _launch_chromium(
                playwright, headless=(True if defer_visible_window else is_headless), channel=browser_channel,
            )
            page = await browser.new_page()
        else:
            context = await browser.new_context()
            page = await context.new_page()
        page.on("dialog", _make_dialog_handler(self.memory, verbose))

        messages: list[dict] = []
        success = False
        final_message = "ครบ max_steps โดยยังไม่จบ task"
        steps_taken = 0
        total_usage = llm.TokenUsage()
        premature_false_finish_count = 0
        premature_true_finish_count = 0
        premature_login_skip_count = 0
        premature_validation_error_count = 0
        # Task4 (W19, ดู _scan_validation_errors ด้านบนสุดของไฟล์): ผลของการ verify ครั้ง
        # สุดท้ายก่อนจบ task — "OK" default เสมอ เปลี่ยนเป็น "EXECUTION_FAILED_NEEDS_REPAIR"
        # เฉพาะตอนที่ยอมรับ finish_task(success=true) ไปทั้งที่ retry ครบโควตาแล้วยังเจอ
        # validation error ค้างอยู่ (escape valve เดียวกับ guard อื่นในไฟล์นี้ — ปล่อยผ่านไป
        # ตามที่โมเดลยืนยัน แทนที่จะค้างไม่รู้จบ แต่ tag ผลลัพธ์ไว้ให้ผู้เรียกรู้ว่าน่าสงสัย)
        completion_verification = "OK"
        final_page_text = ""
        # W9[A] vision fallback: คำอธิบายจาก describe_screenshot() ของ step ก่อนหน้า
        # (ถ้ามี action ที่ต้องพึ่ง visibility ล้มเหลวซ้ำแม้ retry ครบแล้ว) — ใช้ครั้งเดียว
        # แล้วเคลียร์ทิ้ง (ไม่ persist ข้าม step เพราะเป็น diagnostic ของสถานการณ์ตอนนั้น
        # ไม่ใช่ fact ถาวรแบบ manual/memory context)
        pending_vision_context = ""
        plan_text: Optional[str] = None
        # W10[F]: goal ที่ next_action() เห็นจริงทุก step — ปกติเท่ากับ goal เดิมเป๊ะ แต่ถ้า
        # confirm_plan=True จะถูกผนวกด้วยแผน (ที่อาจถูก user แก้ไขก่อน confirm) เข้าไปด้วย
        # หลัง plan ผ่านการยืนยันแล้ว (ดูด้านล่าง) — แยกจาก goal ตัวเดิมเพราะ goal ยังต้อง
        # ใช้แบบดิบๆ ต่อ (RAG query, long-term memory query, log) ไม่อยากให้ข้อความแผนที่
        # อาจยาวมากปนเข้าไปทำให้ query เพี้ยน
        effective_goal = goal
        # W41: wall-clock เวลาที่ next_action() ครั้งก่อนหน้า "จบ" (คืนค่ามาแล้ว) — None
        # ตอนยังไม่เคยเรียกเลย (ครั้งแรกไม่ต้องรอ pacing delay อะไรทั้งนั้น) ใช้คำนวณว่ายัง
        # ต้องหน่วงอีกแค่ไหนให้ครบ _STEP_PACING_DELAY_SECONDS ก่อนเรียกครั้งถัดไป (ดู
        # docstring ของ _STEP_PACING_DELAY_SECONDS ด้านบนสุดของไฟล์)
        last_llm_call_at: Optional[float] = None
        last_action_cmd: Optional[dict] = None
        consecutive_repeat_count = 0
        recent_actions: list[dict] = []  # เก็บ action ล่าสุดไว้เช็ค pattern วนซ้ำ (คาบ 2-4)
        # W31: จำนวนครั้งที่บังคับทำ recovery action (go_back/scroll) แทน action ที่ agent
        # เพิ่งขอไปแล้ว เพราะตรวจพบว่ากำลังวนซ้ำ — ดู _force_loop_recovery()/
        # _MAX_FORCED_LOOP_RECOVERIES ด้านบนสุดของไฟล์
        forced_recovery_count = 0
        # W7[A] (context compaction) / W22 (ทุก provider แล้ว ไม่ใช่แค่ Gemini):
        # [(absolute_step_number, len(messages) หลังจบ step นั้น), ...] — ใช้หา cut
        # point ที่ปลอดภัย (ตรงกับจุดเริ่ม turn ใหม่จริงๆ) ตอนบีบอัด ไม่ใช่ตำแหน่งเดา
        step_boundaries: list[tuple[int, int]] = []
        # W50 (delta history): digest_lines สะสมทีละ "delta" ข้ามหลายรอบ compaction
        # (ไม่ใช่สร้างใหม่ทั้งก้อนทุกรอบ — ดู _build_history_digest()/_MAX_DIGEST_LINES
        # ด้านบนสุดของไฟล์) digest_upto_step คือ step สุดท้ายที่เคยถูกสรุปไปแล้ว (0 =
        # ยังไม่เคย compact เลย) ใช้เป็น from_step ของรอบถัดไป กันสรุปซ้ำ step เดิม
        digest_lines: list[str] = []
        digest_upto_step = 0

        # W22: dedupe manual_context/long_term_context ข้าม step ที่ page_text ไม่เปลี่ยน
        # (เช่น action step ก่อนหน้า fail หรือเป็น fill ที่ไม่ navigate ไปไหน) — ทั้งสองคำนวณ
        # จาก (goal, page_text) ล้วนๆ (goal คงที่ตลอด run_task() นี้อยู่แล้ว) เลย deterministic
        # ถ้า page_text เดิมเป๊ะ = ผลลัพธ์ retrieval ต้องเหมือนเดิมเป๊ะด้วย ไม่มีประโยชน์ต้อง
        # เรียก ChromaDB ซ้ำ (ประหยัด compute) หรือส่งข้อความ context ก้อนเดิมซ้ำเข้า prompt
        # อีกรอบ (ประหยัด token จริง — provider-agnostic ไม่ผูกกับ cache feature ของเจ้าไหน
        # เพราะเป็นการลด byte ที่ส่งจริง ไม่ใช่การลดราคาแบบ cache_control) ตัวแปรก้อนนี้เก็บผล
        # ของ step ล่าสุดที่ page_text เปลี่ยนจริงไว้ใช้ซ้ำ
        last_page_text_for_context: Optional[str] = None
        last_manual_context = ""
        last_long_term_context = ""
        _CONTEXT_UNCHANGED_NOTE = "(เหมือนกับ step ก่อนหน้า — หน้าเว็บยังไม่เปลี่ยน)"

        # W12: "detect หน้าปัจจุบัน" แทนการบังคับ goto(url) เสมอ — เช็คจากสถานะจริงของ
        # page.url ตรงๆ (ไม่ผูกกับโหมดไหนเจาะจง) แทนเดิมที่เคยเช็คแค่ connect_to_user_browser
        # + opened_new_tab (ใช้ได้แค่โหมด CDP โหมดเดียว)
        #
        # W19: เดิมเช็คแค่ "page.url ว่างเปล่าไหม" (about:blank/"") — ไม่ได้เทียบกับ url
        # เป้าหมายของ task นี้เลย ทำให้ session ที่ reuse page ข้ามเทิร์นมา (หรือ tab ที่
        # resolve_target_page() เลือก reuse ในโหมด CDP) ถ้าเทิร์นใหม่สั่ง url อื่นที่ไม่ใช่
        # เว็บเดิม จะไม่ถูก navigate ไปเว็บใหม่เลย (ค้างอยู่หน้าเก่าทั้งที่ user ต้องการเว็บ
        # อื่นจริงๆ) — เทียบ domain กับ target url ตรงๆ แทน: match กัน = "เว็บเป้าหมายเปิด
        # อยู่แล้วจริง" ปล่อยให้ agent perceive หน้าปัจจุบันต่อจากจุดเดิมเลย (เช่นสั่ง
        # "เปิดเว็บ" สำเร็จแล้ว เทิร์นถัดมาสั่ง "sign in" บนเว็บเดิม — ต้องการให้กดปุ่ม sign
        # in บนหน้าที่เปิดค้างไว้ ไม่ใช่เปิดหน้าใหม่เหมือนเริ่มต้นทั้งหมด) ไม่ match กัน (คนละ
        # domain หรือ page ยังว่างเปล่าอยู่) = ต้อง goto(url) ตามปกติเพื่อไปเว็บเป้าหมายจริง
        # ถ้า agent ประเมินเองว่าต้อง navigate เพิ่มเติมอีกก็มี action "goto" ให้เรียกเองได้
        # อยู่แล้วในทุก step ปกติ
        current_domain = extract_domain(page.url) if page.url not in ("about:blank", "") else ""
        target_domain = extract_domain(url)
        site_already_open = bool(target_domain) and current_domain == target_domain
        skip_initial_goto = site_already_open

        try:
            if skip_initial_goto:
                continue_msg = "Website is already open. Reusing the existing browser session."
                if verbose:
                    print(f"[continue] {continue_msg} — หน้าปัจจุบัน: {page.url}", flush=True)
                self.memory.record({
                    "step": 0,
                    "cmd": {"type": "continue", "url": page.url},
                    "result": continue_msg,
                    "success": True,
                })
                await _emit({
                    "kind": "step", "step": 0, "cmd": {"type": "continue", "url": page.url},
                    "result": continue_msg, "success": True,
                })
            else:
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
                await _emit({
                    "kind": "step", "step": 0, "cmd": {"type": "goto", "url": url},
                    "result": str(goto_result), "success": goto_result.success,
                })
            await wait_stable(page)

            # W17: auto-login ครั้งเดียวตอนต้น task ก่อนวางแผน/เข้า loop หลัก — ใช้ได้ทั้ง
            # กรณี goto สดๆ และกรณี skip_initial_goto (tab เดิมจากเทิร์นก่อนหน้าดันมาเจอ
            # หน้า login พอดี เช่น session หลุด) ไม่มีผลอะไรถ้าหน้าปัจจุบันไม่ใช่หน้า login
            # หรือไม่มี credential เก็บไว้สำหรับโดเมนนี้ (ดู _maybe_auto_login())
            #
            # ถ้ามี credential เก็บไว้จริงแต่ login ไม่ผ่าน (retry แล้วก็ยังไม่ผ่าน) ต้องแจ้ง
            # user ชัดเจนผ่าน SSE ก่อน — เดิมล้มเหลวแบบเงียบๆ ทำให้ agent เดินหน้า task ต่อ
            # ทั้งที่ไม่มี permission ที่ถูกต้องโดย user ไม่รู้ตัว ไม่ throw/ไม่หยุด task เพราะ
            # agent ยัง fallback ไปกรอกฟอร์ม login เองผ่าน action ปกติได้อยู่แล้ว — แค่ต้อง
            # ให้ user เห็นว่า credential ที่บันทึกไว้ใช้ไม่ได้แล้ว
            auto_login_failure_reason = await _maybe_auto_login(page, verbose)
            if auto_login_failure_reason:
                await _emit({
                    "kind": "auto_login_failed",
                    "message": "ล็อกอินไม่สำเร็จด้วย credential ที่บันทึกไว้สำหรับเว็บนี้",
                    "reason": auto_login_failure_reason,
                })

            # Intent Classification: ตรวจจับ Intent ของผู้ใช้ก่อนเริ่ม Planner Loop
            initial_elements, initial_page_text = await get_snapshot(page)
            user_intent = await llm.classify_intent(client, model, goal, page_text=initial_page_text, provider=resolved_provider)
            if user_intent == "qa_summary":
                if verbose:
                    print(f"[intent] ตรวจพบ Intent: qa_summary — ตอบคำถาม/สรุปข้อมูลจากหน้าเว็บ (read_page_data + ค้นหา)", flush=True)
                qa_messages: list = []
                summary_text = ""
                # เก็บ elements/page_text ล่าสุดไว้ใช้ต่อ (ทั้ง lookup label ของ fill/click รอบ
                # ถัดไป และ fallback/final_page_state ท้ายบล็อกนี้) — ต้องอัปเดตทุกครั้งที่ทำ
                # action ที่เปลี่ยนหน้าเว็บจริง (fill/click) ไม่งั้น next_action() รอบถัดไปจะ
                # เห็น page_text เดิมก่อนค้นหาอยู่ ทั้งที่หน้าเปลี่ยนไปแล้วจริงหลัง submit ค้นหา
                qa_elements, qa_page_text = initial_elements, initial_page_text
                qa_goal = f"{goal}{_QA_ANSWER_FORMAT_GUIDANCE}"
                for _ in range(_QA_SUMMARY_MAX_STEPS):
                    qa_tool_name, qa_tool_input, qa_tool_use_id, qa_messages, qa_usage = await next_action(
                        client, model, qa_goal, qa_page_text, qa_messages, plan_context="",
                    )
                    total_usage += qa_usage
                    if qa_tool_name == "finish_task":
                        summary_text = qa_tool_input.get("message", "")
                        break

                    qa_action_type = qa_tool_input.get("type")
                    if qa_action_type == "read_page_data":
                        qa_result: ActionResult = await execute(
                            page, qa_tool_input, ask_user_func=ask_user_func, label="",
                            manual_guidance="", allowed_domains=effective_allowed_domains,
                        )
                        qa_messages = append_tool_result(qa_messages, qa_tool_use_id, str(qa_result))
                        continue

                    # ผ่อนให้ fill/click ทำได้เพิ่ม "เฉพาะ" ตอน label ของ element เป้าหมายดู
                    # เป็นช่อง/ปุ่มค้นหา/กรองข้อมูลจริงๆ (ดู _label_looks_like_search() —
                    # ต่อยอด W44) action อื่นที่ไม่เข้าเงื่อนไขนี้ยังถูกปฏิเสธเหมือนเดิมทุก
                    # ประการ (login/checkout/delete/... ยังทำไม่ได้จาก intent นี้)
                    #
                    # W19 ("Guard Compatibility Rule"): user รายงานว่าคำถามที่ต้อง navigate
                    # ไปหน้าย่อยก่อนถึงจะเห็นข้อมูล (เช่น "มีผู้ใช้กี่คนในหน้า Admin" ทั้งที่
                    # ยังไม่ได้อยู่หน้า Admin) ตอบไม่ได้เลย เพราะคลิกเมนู "Admin" ไม่ใช่ช่อง
                    # ค้นหา ไม่เข้าเงื่อนไข _label_looks_like_search() เลย โดน
                    # _QA_SUMMARY_ACTION_REJECTED_NUDGE ปฏิเสธทุกครั้ง — ผ่อนเพิ่มให้ click
                    # (เฉพาะ click ไม่รวม fill) ที่ region="navigation" (sidebar/nav/menu, ดู
                    # perception.py::getRegion) ทำได้ด้วย เพราะเป็นแค่การนำทางเปลี่ยนหน้า
                    # ไม่ mutate ข้อมูลอะไรบนเว็บเลย (คนละเรื่องกับ submit/delete/purchase —
                    # ยังถูก classify_action() ใน execute() ด้านล่างเช็คซ้ำอีกชั้นอยู่ดี ถ้า
                    # label ดันเป็นคำเสี่ยงจริงๆ ก็ยังโดนขอ confirm ตามปกติ ไม่ได้ข้าม
                    # permission layer ไปเลย)
                    qa_index = qa_tool_input.get("index")
                    qa_label = next(
                        (e["label"] for e in qa_elements if e["index"] == qa_index), ""
                    ) if qa_index is not None else ""
                    qa_region = next(
                        (e.get("region", "") for e in qa_elements if e["index"] == qa_index), ""
                    ) if qa_index is not None else ""
                    qa_is_nav_click = qa_action_type == "click" and qa_region == "navigation"
                    if (qa_action_type in ("fill", "click") and _label_looks_like_search(qa_label)) or qa_is_nav_click:
                        qa_result: ActionResult = await execute(
                            page, qa_tool_input, ask_user_func=ask_user_func, label=qa_label,
                            manual_guidance="", allowed_domains=effective_allowed_domains,
                        )
                        qa_messages = append_tool_result(qa_messages, qa_tool_use_id, str(qa_result))
                        if qa_action_type in _PAGE_CHANGING_ACTIONS:
                            await wait_stable(page)
                        qa_elements, qa_page_text = await get_snapshot(page)
                        continue

                    qa_messages = append_tool_result(qa_messages, qa_tool_use_id, _QA_SUMMARY_ACTION_REJECTED_NUDGE)
                if not summary_text:
                    # ครบโควตาแล้วยังไม่เรียก finish_task (หรือโมเดลไม่ยอมเรียก tool ที่
                    # อนุญาตเลยสักครั้ง) — fallback กลับไปใช้ summarize_page() เดิม แทนที่
                    # จะคืนคำตอบว่างเปล่าให้ user ใช้ qa_page_text ล่าสุด (หลังลองค้นหาไปแล้ว
                    # ถ้ามี) ไม่ใช่ initial_page_text เดิมก่อนค้นหา ไม่งั้นจะเสียผลลัพธ์การ
                    # ค้นหาที่เพิ่งทำไปทิ้งไปเฉยๆ
                    summary_text = await llm.summarize_page(
                        client, model, page_text=qa_page_text, user_prompt=qa_goal, provider=resolved_provider
                    )
                self.memory.record({
                    "step": 1,
                    "cmd": {"type": "chat_reply"},
                    "result": summary_text,
                    "success": True,
                })
                await _emit({
                    "kind": "chat_reply",
                    "message": summary_text,
                    "status": "chat_reply",
                })
                return {
                    "status": "chat_reply",
                    "success": True,
                    "steps": 1,
                    "message": summary_text,
                    "history": self.memory.recent(max_steps),
                    "tokens": _tokens_dict(total_usage),
                    "plan": None,
                    "final_page_state": qa_page_text,
                }


            if approved_plan:
                # W13: แผนถูกอนุมัติไปแล้วจากภายนอก (routes.py::POST /api/generate_plan
                # -> user review -> POST /api/execute_plan) ก่อนจะเรียก run_task() ด้วย
                # ซ้ำ — ไม่ต้องเรียก llm.generate_plan()/รอ ask_user_func ข้างในนี้เลย แค่
                # ผนวกเข้า effective_goal ทันทีแล้วเริ่ม loop จริงต่อได้เลย (ใช้ตัวแปร
                # effective_goal/plan_text ชุดเดียวกับที่ confirm_plan ด้านล่างใช้ ให้ผล
                # ต่อ loop/result["plan"] เหมือนกันทุกประการ ไม่ว่าแผนจะมาจากทางไหน)
                plan_text = approved_plan
                effective_goal = f"{goal}\n\nFollow this confirmed plan:\n{plan_text}"
            elif confirm_plan:
                _, plan_page_text = await get_snapshot(page)
                plan_text = await llm.generate_plan(client, model, goal, plan_page_text, resolved_provider)
                if verbose:
                    print(f"[plan]\n{plan_text}", flush=True)
                approved, plan_text = await _confirm_plan(plan_text, ask_user_func)
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

                # W10[F]: จากนี้ไปทุก step ให้ next_action() เห็นแผนที่ยืนยันแล้ว (ซึ่งอาจ
                # ถูก user แก้ไขไปแล้วจากที่ AI ร่างไว้เอง) เป็นส่วนหนึ่งของเป้าหมายด้วย —
                # ไม่งั้นต่อให้ user แก้ plan_text ถูกต้องแค่ไหน ก็ไม่มีผลอะไรกับพฤติกรรม
                # จริงเลย เพราะ per-step loop ไม่เคยอ่าน plan_text อยู่แล้ว (ใช้แค่โชว์ตอน
                # confirm เฉยๆ) — ต่อท้าย goal เดิมแทนที่จะแทนที่ ให้ยังอ่านออกว่าเป้าหมาย
                # หลักคืออะไร บวกกับแผนที่ต้องทำตามคืออะไร
                effective_goal = f"{goal}\n\nFollow this confirmed plan:\n{plan_text}"

                # W11[A]: user ยืนยันแผนแล้ว — ถึงเวลาเปิดหน้าต่างจริงที่ซ่อนไว้ก่อนหน้านี้
                # (ดู defer_visible_window ด้านบน) ปิดตัว headless ชั่วคราวทิ้ง แล้วเปิด
                # browser ที่มองเห็นได้ตัวใหม่แทน (Playwright เปลี่ยน headless<->headed
                # กลางคันของ process เดิมไม่ได้ ต้อง launch ใหม่) — ยังไม่มี action จริงเกิด
                # ขึ้นเลยตอนนี้ (steps_taken ยังเป็น 0) แค่ goto ซ้ำหน้าเดิมบนหน้าต่างใหม่
                # ก็เพียงพอ ไม่มีอะไรให้เสียหาย
                if defer_visible_window:
                    await browser.close()
                    browser = await _launch_chromium(playwright, headless=False, channel=browser_channel)
                    page = await browser.new_page()
                    page.on("dialog", _make_dialog_handler(self.memory, verbose))
                    await goto(page, url)
                    await wait_stable(page)

            for _ in range(max_steps):
                elements, page_text = await get_snapshot(page)
                await _emit_screenshot(steps_taken)
                # W5[A] verify: เก็บ page_text ล่าสุดไว้เป็นหลักฐานจริงจาก DOM ตอนจบ
                # task (ทุก path — finish_task/loop-detected/หมด max_steps) แนบไปกับ
                # result ให้ผู้ประเมิน (เช่น W12[B] eval script/human review) เทียบกับ
                # message ที่ LLM อ้างได้เอง ไม่ต้องเชื่อคำเคลมของ LLM ลอยๆ อย่างเดียว
                final_page_text = page_text

                # W6[B]: ดึงคู่มือที่เกี่ยวข้องกับ goal+หน้าปัจจุบันใหม่ทุก step ที่หน้าเปลี่ยน
                # จริง (retrieve() ไม่ throw เอง คืน [] เงียบๆ ถ้าไม่มีคู่มือ/error) — ใช้
                # to_thread เพราะเป็นงาน sync (local embedding inference + ChromaDB query)
                # ไม่งั้นจะบล็อก event loop ตัวเดียวกับที่ Playwright ใช้อยู่ (เหมือน
                # _confirm_plan() ที่ wrap input() ด้วย to_thread ด้วยเหตุผลเดียวกัน)
                #
                # W22: ถ้า page_text เหมือน step ก่อนหน้าเป๊ะ (เช่น action ก่อนหน้า fail/ไม่
                # navigate ไปไหน) ข้าม retrieval ทั้งคู่ไปเลย ใช้ marker สั้นๆ แทนก้อนข้อความ
                # เดิมที่ LLM เห็นไปแล้วในเทิร์นก่อนหน้า (ดู comment ของ
                # last_page_text_for_context ด้านบนสุดของ run_task())
                page_changed_for_context = page_text != last_page_text_for_context
                if page_changed_for_context:
                    manual_chunks = await asyncio.to_thread(
                        retriever.retrieve, query=goal, page_state=page_text, k=_RAG_CHUNKS_PER_STEP
                    )
                    manual_context = "\n".join(f"- {chunk}" for chunk in manual_chunks)

                    long_term_chunks = await asyncio.to_thread(
                        long_term_memory.recall,
                        query=goal, page_state=page_text, k=_LONG_TERM_MEMORY_CHUNKS_PER_STEP,
                        session_id=session_id or "",
                    )
                    long_term_context = "\n".join(f"- {chunk}" for chunk in long_term_chunks)

                    last_page_text_for_context = page_text
                    last_manual_context = manual_context
                    last_long_term_context = long_term_context
                else:
                    manual_context = _CONTEXT_UNCHANGED_NOTE if last_manual_context else ""
                    long_term_context = _CONTEXT_UNCHANGED_NOTE if last_long_term_context else ""

                # W50 (client-side action verification): สัญญาณเสริมจากโค้ด (ไม่ต้องพึ่ง
                # LLM สังเกตเอง) ว่า action ก่อนหน้าที่คืน [OK] แล้วจริงๆ อาจไม่มีผลอะไรกับ
                # หน้าเว็บเลย — ใช้ page_changed_for_context ที่คำนวณไปแล้วด้านบน (เทียบ
                # page_text ของรอบนี้กับรอบก่อนหน้า) ไม่ต้องยิง browser เพิ่มเลย: ถ้า action
                # ก่อนหน้า (self.memory.recent(1) — บันทึกไว้แล้วท้าย iteration ก่อนหน้า)
                # สำเร็จ (success=True) เป็นประเภทที่ "ควรจะ" เปลี่ยนอะไรบนหน้าเว็บ (ดู
                # _VERIFICATION_SIGNAL_ACTION_TYPES) แต่ page_text เหมือนเดิมทุกตัวอักษร —
                # แจ้งเตือน LLM รอบนี้ว่า action นั้นอาจเป็น no-op ทั้งที่ดูเหมือนสำเร็จ กัน
                # การเสีย step ต่อๆ ไปคิดว่า "ทำไปแล้ว" ทั้งที่จริงไม่มีผล
                verification_context = ""
                if steps_taken > 0 and not page_changed_for_context:
                    last_record = self.memory.recent(1)
                    if last_record:
                        last_cmd = last_record[0].get("cmd", {}) or {}
                        if (
                            last_record[0].get("success") is True
                            and last_cmd.get("type") in _VERIFICATION_SIGNAL_ACTION_TYPES
                        ):
                            verification_context = (
                                f"[ตรวจสอบผล action ก่อนหน้า ({last_cmd}): ไม่พบการเปลี่ยนแปลง"
                                "บนหน้าเว็บเลย (element/เนื้อหาของหน้าเหมือนเดิมทุกตัวอักษร) — "
                                "action นี้อาจไม่มีผลจริงแม้จะได้ [OK] ก็ตาม ลองพิจารณาทางอื่น "
                                "(เช่น hover ก่อนคลิก, คลิกตำแหน่ง/index อื่น หรือถ้าเป็น custom "
                                "dropdown ให้ลองใช้ press_key แทน)]"
                            )

                # W7[A]: สรุป action ที่ล้มเหลวไปแล้วใน task นี้ (ดู
                # ShortTermMemory.failed_actions_summary() docstring) ป้อนกลับเข้า prompt
                # ทุก step เหมือน manual_context — ว่างเปล่าถ้ายังไม่เคย fail อะไรเลย
                memory_context = self.memory.failed_actions_summary()

                # W32: action ล่าสุดไม่กี่ step (ทั้งสำเร็จและล้มเหลว) แยกจาก memory_context
                # ด้านบนที่กรองเฉพาะ fail — ดู ShortTermMemory.recent_actions_summary()
                action_history_context = self.memory.recent_actions_summary()

                # W9[A]: ใช้ vision_context ของรอบนี้แล้วเคลียร์ทิ้งทันที (one-shot —
                # ดู pending_vision_context ด้านบนสุดของ run_task())
                vision_context, pending_vision_context = pending_vision_context, ""

                # W41: หน่วงเฉพาะส่วนที่ยังขาดให้ครบ _STEP_PACING_DELAY_SECONDS นับจากที่
                # next_action() ครั้งก่อนจบ (ไม่ใช่ sleep เต็มจำนวนทุกครั้งแบบเดิม) — งาน
                # จริงที่ทำไปแล้วตั้งแต่ครั้งก่อน (execute()/wait_stable()/get_snapshot()/
                # retrieve()/recall() ด้านบน) นับรวมเข้าไปในระยะห่างนี้ด้วย ระยะห่างขั้นต่ำ
                # ระหว่างการเรียก LLM 2 ครั้งยังเท่าเดิมทุกประการ (ไม่ลดความปลอดภัยจาก
                # rate-limit) แค่ไม่ sleep ซ้ำกับเวลาที่ผ่านไปแล้วจริง
                if last_llm_call_at is not None:
                    elapsed = time.monotonic() - last_llm_call_at
                    remaining = _STEP_PACING_DELAY_SECONDS - elapsed
                    if remaining > 0:
                        await asyncio.sleep(remaining)

                # W43: plan_text (ดู confirm_plan/approved_plan ด้านบน) เป็น None สำหรับ
                # ad-hoc task ที่ไม่มีแผนเลย — ส่งเป็น "" ให้ next_action()/
                # _build_user_turn_text() ไม่ต้องรู้จัก Optional เอง (plan_context="" =
                # ไม่มี section "แพลนปัจจุบัน" โผล่มาปนเลย ตรงกับพฤติกรรมเดิมทุกประการ)
                tool_name, tool_input, tool_use_id, messages, usage = await next_action(
                    client, model, effective_goal, page_text, messages,
                    manual_context, memory_context, long_term_context, vision_context,
                    site_manual_context, page.url, action_history_context, plan_text or "",
                    verification_context=verification_context,
                )
                last_llm_call_at = time.monotonic()
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

                    # Task4 (W19, ดู _scan_validation_errors ด้านบนสุดของไฟล์): เช็คทุกครั้ง
                    # ที่ claimed_success (ไม่ผูกกับ steps_taken เหมือน guard ด้านบน) เพราะ
                    # error อาจโผล่ขึ้นมาหลัง action ผ่านไปหลาย step แล้วก็ได้ ไม่ใช่แค่ step
                    # แรกสุด
                    detected_errors: list[str] = []
                    if claimed_success and tool_use_id:
                        detected_errors = await _scan_validation_errors(page)
                    if (
                        detected_errors
                        and premature_validation_error_count < _MAX_PREMATURE_VALIDATION_ERROR_RETRIES
                    ):
                        premature_validation_error_count += 1
                        errors_text = "; ".join(detected_errors)
                        if verbose:
                            print(
                                f"[finish_task(true) พบ validation error {premature_validation_error_count}/"
                                f"{_MAX_PREMATURE_VALIDATION_ERROR_RETRIES}] {errors_text}",
                                flush=True,
                            )
                        nudge_text = _PREMATURE_VALIDATION_ERROR_NUDGE_TEMPLATE.format(errors=errors_text)
                        messages = append_tool_result(messages, tool_use_id, nudge_text)
                        messages.append(_build_nudge_message(resolved_provider, f"⚠️ [ระบบคำสั่งสำคัญ]: {nudge_text}"))
                        continue
                    if detected_errors:
                        # retry ครบโควตาแล้วยังเจอ error ค้างอยู่ — ปล่อยผ่านไปตามที่โมเดล
                        # ยืนยัน (escape valve เดียวกับ guard อื่นในไฟล์นี้) แต่ tag ผลลัพธ์
                        # ไว้ให้ผู้เรียกรู้ว่าน่าสงสัย แทนที่จะค้างไม่รู้จบ
                        completion_verification = "EXECUTION_FAILED_NEEDS_REPAIR"

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
                    loop_reason = (
                        f"agent สั่ง action เดิมซ้ำติดกัน {consecutive_repeat_count} ครั้ง "
                        f"({tool_input}) โดยไม่มีความคืบหน้า"
                    )
                    # W31: บังคับ recovery action แทนการจบ task ทันที ให้โอกาส agent กู้
                    # สถานการณ์เอง (ดู _force_loop_recovery() — คืน False ถ้าเกิน
                    # _MAX_FORCED_LOOP_RECOVERIES แล้ว ค่อย fallback ไปจบ task แบบเดิม)
                    if await _force_loop_recovery(loop_reason):
                        if verbose:
                            print(f"[loop-detected] {loop_reason} -> บังคับ recovery action แทน", flush=True)
                        continue
                    success = False
                    final_message = f"หยุด task: {loop_reason} (บังคับ recovery action ไปแล้วแต่ยังไม่หาย)"
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
                    cycle_desc = " -> ".join(str(a) for a in recent_actions[-detected_period:])
                    loop_reason = f"agent วน action ซ้ำเป็นคาบ {detected_period} ({cycle_desc}) โดยไม่มีความคืบหน้า"
                    if await _force_loop_recovery(loop_reason):
                        if verbose:
                            print(f"[loop-detected] {loop_reason} -> บังคับ recovery action แทน", flush=True)
                        continue
                    success = False
                    final_message = f"หยุด task: {loop_reason} (บังคับ recovery action ไปแล้วแต่ยังไม่หาย)"
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
                # W_search follow-up: tag จริงของ element (เช่น "a") ส่งให้
                # execute()/classify_action() เช็คสัญญาณโครงสร้างเป็นชั้นสำรองอีกชั้น —
                # ดักเคส label เป็นเนื้อหาอิสระ (ชื่อวิดีโอ/บทความ) ที่ไม่ match ทั้ง
                # SAFE_ACTION_LABEL_KEYWORDS และ RISKY_LABEL_KEYWORDS เลย (ดู
                # permission/rules.py::ANCHOR_TAG)
                action_tag = next(
                    (e.get("tag", "") for e in elements if e["index"] == action_index), ""
                ) if action_index is not None else ""
                # W_search follow-up 2: attribute "type" ของ element (เช่น input ที่
                # type="text"/"search") ส่งคู่กับ tag ให้ execute()/classify_action() แยก
                # ช่องกรอกข้อความ/ค้นหาธรรมดาออกจาก input ที่แท้จริงอาจเสี่ยง (ดู
                # permission/rules.py::SAFE_INPUT_TAG/RISKY_INPUT_TYPES)
                action_element_type = next(
                    (e.get("type", "") for e in elements if e["index"] == action_index), ""
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

                # W19 (ดู W19.txt ข้อ 8 "Semantic Redundancy Evaluator") / W19-2 ("Safety &
                # Performance Middleware", ดู llm.py::evaluate_safety_and_performance) —
                # สองตัวนี้ mutually exclusive กัน (ไม่เรียก LLM ซ้ำสองครั้งเพื่อเช็ค
                # redundancy เรื่องเดียวกัน): เปิด enable_middleware_evaluator แล้วใช้ตัวนั้น
                # (รวม permission check มาด้วยในตัว) แทน enable_semantic_redundancy_check
                # เดิม ถ้าไม่ได้เปิด middleware ค่อย fallback ไปใช้ evaluate_semantic_redundancy
                # ตามปกติ (เฉพาะ tool_input ที่ไม่ใช่ navigate/รอ/อ่านข้อมูล — มักมีเหตุผล
                # ชัดเจนอยู่แล้วว่าทำไมต้องทำ ไม่ใช่กลุ่ม action ที่ตัวนี้ถูกออกแบบมาจับ)
                if settings.enable_middleware_evaluator and tool_input.get("type") not in (
                    "goto", "go_back", "wait", "switch_tab", "read_page_data",
                ):
                    element_description = (
                        f"<{action_tag or 'element'}> '{action_label}'" if action_label
                        else f"<{action_tag}>" if action_tag else "(ไม่ทราบ label)"
                    )
                    middleware = await llm.evaluate_safety_and_performance(
                        client, model, effective_goal, extract_domain(page.url),
                        tool_input.get("type", ""), element_description,
                        str(tool_input.get("text") or tool_input.get("label") or ""), resolved_provider,
                    )
                    if verbose:
                        print(f"[middleware] {middleware}", flush=True)
                    if middleware.get("final_action_decision") == "SKIP_REDUNDANT":
                        skip_reason = middleware.get("redundancy_evaluation", {}).get("redundancy_reason", "")
                        messages = append_tool_result(
                            messages, tool_use_id,
                            f"[ระบบข้าม step นี้อัตโนมัติ] {skip_reason} "
                            "เลือก action อื่นที่ทำให้เป้าหมายคืบหน้าจริงแทน",
                        )
                        continue
                    # escalate-only (ดู module comment ใน llm.py): risk_level REQUIRES_CONSENT/
                    # BLOCKED ต่อวลีที่ตรงกับ permission/rules.py::MANUAL_CONFIRMATION_KEYWORDS
                    # เข้า manual_permission_guidance เดิม (ตัวเดียวกับที่ RAG คู่มือใช้อยู่
                    # แล้ว) ให้ classify_action() ที่ dispatch จริงด้านล่าง escalate เป็น
                    # NEEDS_CONFIRMATION ผ่านกลไกเดิมที่ทดสอบไว้แล้ว — ไม่เรียก ask_user_func
                    # เองตรงๆ ที่นี่ (กันถามซ้ำสองครั้งสำหรับ action เดียวกัน) และไม่มีทาง
                    # "ลดระดับ" ความเสี่ยงที่ classify_action() จะตัดสินเองอยู่ดี (risk_level
                    # AUTO_APPROVE = ไม่ต่อท้ายอะไรเลย = พฤติกรรมเดิมเป๊ะ)
                    permission_eval = middleware.get("permission_evaluation", {})
                    if permission_eval.get("risk_level") in ("REQUIRES_CONSENT", "BLOCKED"):
                        escalation_note = (
                            f"[Safety Middleware] {permission_eval.get('permission_reason', '')} "
                            "— requires confirmation before proceeding."
                        )
                        manual_permission_guidance = (
                            f"{manual_permission_guidance}\n- {escalation_note}"
                            if manual_permission_guidance else f"- {escalation_note}"
                        )
                elif settings.enable_semantic_redundancy_check and tool_input.get("type") not in (
                    "goto", "go_back", "wait", "switch_tab", "read_page_data",
                ):
                    step_summary = (
                        f"{tool_input.get('type')} -> '{action_label}'" if action_label
                        else str(tool_input.get("type"))
                    )
                    redundancy = await llm.evaluate_semantic_redundancy(
                        client, model, effective_goal, step_summary, page.url, action_label,
                        tool_name, tool_input, resolved_provider,
                    )
                    if redundancy.get("action_decision") in ("SKIP_STEP", "FORCE_REPLAN"):
                        if verbose:
                            print(f"[semantic-redundancy] {redundancy}", flush=True)
                        skip_reason = redundancy.get("reasoning", "")
                        messages = append_tool_result(
                            messages, tool_use_id,
                            f"[ระบบข้าม step นี้อัตโนมัติ] {skip_reason} "
                            "เลือก action อื่นที่ทำให้เป้าหมายคืบหน้าจริงแทน",
                        )
                        continue

                # W30: เก็บ URL ก่อน dispatch action ไว้เทียบหลัง action จบ (ดู
                # url_changed_unexpectedly ด้านล่าง) — เฉพาะ action ที่ "ไม่ได้ตั้งใจจะ
                # navigate" เอง (goto/switch_tab/go_back คือจุดประสงค์หลักคือเปลี่ยนหน้า
                # อยู่แล้ว ไม่ต้องเตือนซ้ำ)
                url_before_action = page.url

                result: ActionResult = await execute(
                    page, tool_input, ask_user_func=ask_user_func, label=action_label,
                    manual_guidance=manual_permission_guidance, allowed_domains=effective_allowed_domains,
                    element_tag=action_tag, element_type=action_element_type,
                )
                steps_taken += 1
                # W10[D]: แนบ label ของ element เป้าหมาย (ชื่อปุ่ม/ช่องกรอกจริงบนหน้าเว็บ
                # เช่น "Login", "Username" — มาจาก perception.py::get_snapshot() ตัวเดียว
                # กับที่ action_label ด้านบนใช้เช็ค permission อยู่แล้ว) เข้า history/event
                # ด้วย ให้ UI (Log panel) โชว์ชื่อจริงแทน index เปล่าๆ ที่มนุษย์อ่านไม่รู้
                # เรื่องว่ากดอะไร/กรอกช่องไหน — ไม่มีผลกับ dispatch จริง (ยังใช้ tool_input
                # เดิมเป๊ะ) แค่ข้อมูลเสริมไว้แสดงผล
                self.memory.record({
                    "step": steps_taken,
                    "cmd": tool_input,
                    "label": action_label,
                    "result": str(result),
                    "success": result.success,
                    "tokens": _tokens_dict(usage),
                    # W_procmem: locator ที่ "อยู่รอด" ข้าม task run ได้ (ดู
                    # core/dom_locator.py/actions.py) — None เสมอยกเว้น
                    # click/fill/select/check ที่สำเร็จจริง ใช้เป็น input ของ
                    # llm.abstract_trajectory() ตอน task จบสำเร็จ (ดูจุดเรียกด้านล่าง)
                    "locator_descriptor": result.locator_descriptor,
                })
                if verbose:
                    print(f"  -> {result}", flush=True)
                await _emit({
                    "kind": "step", "step": steps_taken, "cmd": tool_input,
                    "label": action_label,
                    "result": str(result), "success": result.success,
                    # W49: cumulative token usage ของ task นี้จนถึง step นี้ — ให้ frontend
                    # โชว์ token count สดๆ ตอน task ยัง "running" อยู่ (ก่อนหน้านี้มีแค่ใน
                    # result["tokens"] ตอนจบ task เท่านั้น ซึ่ง SSE consumer เห็นช้าเกินไป)
                    "tokens": _tokens_dict(total_usage),
                })

                # W43: LLM ระบุว่า action นี้ทำให้ step ของแผนเสร็จสมบูรณ์แล้ว (ดู
                # completed_plan_step ใน llm.py::_BROWSER_ACTION_PARAMS) — ยิง SSE event
                # ใหม่ให้ frontend ติ๊ก checkbox ของ step นั้นแบบ real-time เฉพาะตอน
                # execute() สำเร็จจริงเท่านั้น (result.success — กันติ๊กผิดว่าทำสำเร็จ
                # ทั้งที่ action พัง) และเฉพาะ task ที่มีแผนจริงๆ เท่านั้น (plan_text ไม่ใช่
                # None/ว่างเปล่า — ad-hoc task ไม่มีแผนไม่ควรยิง event นี้เลยแม้ LLM จะใส่
                # completed_plan_step มาผิดๆ ก็ตาม เพราะไม่มี checkbox ให้ติ๊กอยู่แล้วฝั่ง UI)
                completed_plan_step = tool_input.get("completed_plan_step")
                if plan_text and result.success and completed_plan_step is not None:
                    await _emit({"kind": "plan_step_done", "step": completed_plan_step})

                # W10[F]: human ปฏิเสธ action นี้ตรงๆ (กด Deny บน permission prompt) —
                # ต้องจบ task ทันที ไม่ใช่ป้อนผลลัพธ์กลับเข้า messages แล้ววน loop ต่อให้
                # LLM ลองทางอื่น (พฤติกรรมเดิม ผิดจุดประสงค์ของ human-in-the-loop: การ
                # ปฏิเสธคือคำสั่ง "หยุด" ไม่ใช่ "ลองทางอื่น") — ต่างจาก REJECTED_BY_USER_
                # MESSAGE ที่เกิดจาก timeout (ask_user_func คืน False เพราะไม่มีใครตอบ
                # ทัน) ซึ่งก็ควรจบทันทีเหมือนกัน เพราะจากมุมมอง user คือ "ยังไม่ได้อนุมัติ"
                # ไม่ต่างจากปฏิเสธเลย
                if not result.success and result.message == REJECTED_BY_USER_MESSAGE:
                    success = False
                    final_message = (
                        f"หยุด task ทันที: ผู้ใช้ปฏิเสธ action นี้ ({tool_input}) — "
                        "ไม่ลองทำทางอื่นต่อตามหลัก human-in-the-loop (การปฏิเสธคือคำสั่งหยุด)"
                    )
                    if verbose:
                        print(f"[human-denied] {final_message}", flush=True)
                    break

                # W20 (Task12, "UI Validation Error Detection" — Early Termination
                # Guardrail, บั๊กจริงที่ user รายงาน): เช็คทันทีถ้า action ที่เพิ่งสำเร็จนี้
                # (fill ทุกครั้ง หรือ click/submit ปุ่ม Save/Submit/Confirm/Update — ดู
                # _should_check_validation_error_after_action ด้านบนสุดของไฟล์) ทำให้มี
                # validation error ปรากฏอยู่บนหน้าปัจจุบันจริง — ต่างจาก
                # _scan_validation_errors() เดิม (เรียกอยู่แล้วด้านล่างก่อนยอมรับ
                # finish_task(success=true) เท่านั้น) ตัวนี้ทำงานได้แม้ agent จะไม่เคยเรียก
                # finish_task เลยสักครั้ง (แค่วน fill/click/refresh ไม่จบตามที่ user รายงาน —
                # guard เดิมไม่มีทางถูกเรียกเลยในสถานการณ์นั้น) STOP ทันที ไม่ลอง
                # go_back()/refresh/retry ต่อเด็ดขาด (ต่างจาก guard เดิมที่ยังให้ nudge-retry
                # ได้ 2 ครั้งก่อนตอน finish_task) เพราะ validation error พวกนี้ต้องแก้ด้วยการ
                # เปลี่ยนค่าที่กรอกจริงๆ เท่านั้น ไม่ใช่สิ่งที่ retry/refresh เดิมซ้ำแล้วจะหายไป
                # เอง — กรอง "* Required" ล้วนๆ ทิ้งก่อน (ดู _is_bare_required_message()) กัน
                # false positive จากช่องพี่น้องที่ยังไม่ได้กรอก แล้วคัดลอกข้อความ error ตรงตามที่
                # ระบบแสดงจริงส่งกลับให้ user เห็นเป๊ะๆ
                #
                # (2026-08-06) user ขอเพิ่ม: แทนที่จะจบด้วยการรายงาน error เฉยๆ ให้ข้อความชวน
                # user ตอบกลับมาด้วยค่าใหม่ที่ต้องการใช้แทน — page/session (ตอน session_id ผูก
                # กับ conversation) ยังเปิดค้างอยู่หลัง break นี้เหมือน "human-denied" break
                # ด้านบน ทำให้ turn ถัดไปของ user ใน conversation เดียวกันไหลเข้า
                # generate_plan() พร้อม previous_assistant_message = ข้อความนี้ (กลไก
                # "Context-Aware Implicit Execution" ที่มีอยู่แล้ว) แล้ว perceive หน้าเดิมที่
                # ฟอร์มยังค้างอยู่ได้ต่อ — LLM planner จึงกรอกค่าใหม่แทนที่ในช่องเดิมแล้วส่งฟอร์ม
                # ต่อให้ได้เองโดยไม่ต้องมี infrastructure ใหม่ (ไม่ใช่ ask_user_func เดิมที่รองรับ
                # แค่ approve/deny — เจตนาจริงคือ "รับ input ใหม่จาก user" ซึ่งคือข้อความ chat
                # ตอบกลับปกติ ไม่ใช่ permission prompt)
                if result.success and _should_check_validation_error_after_action(
                    tool_input.get("type"), action_label,
                ):
                    validation_errors = [
                        e for e in await _scan_validation_errors(page, within_form=True)
                        if not _is_bare_required_message(e)
                    ]
                    if validation_errors:
                        errors_text = " | ".join(f"'{e}'" for e in validation_errors)
                        success = False
                        completion_verification = "TASK_FAILED_USER_INPUT_ERROR"
                        final_message = (
                            "ไม่สามารถดำเนินการต่อได้ เนื่องจากข้อมูลที่กรอกไม่ผ่านการตรวจสอบ"
                            f"ของระบบ: {errors_text} — กรุณาตอบกลับมาด้วยค่าใหม่ที่ต้องการใช้แทน "
                            "ระบบจะกรอกค่านั้นแทนที่ในช่องเดิมแล้วดำเนินการต่อให้ทันที"
                        )
                        if verbose:
                            print(f"[validation-error] {final_message}", flush=True)
                        break

                # W9[A] vision fallback (Gemini เท่านั้น): action ที่ต้องพึ่ง element
                # visibility ล้มเหลวซ้ำแม้ retry ครบแล้ว (actions.py::
                # _dispatch_with_retry หมดโควตา) ทั้งที่ index มีอยู่จริงใน DOM ตอน
                # perceive — สงสัยว่ามี popup/overlay บัง element ที่ perception
                # (DOM-based ล้วนๆ) ตรวจไม่เจอครบ (แม้จะมี marker "[ถูกบังอยู่]" เสริม
                # จาก perception.py แล้วก็ตาม — ยังมีเคสที่ elementFromPoint() พลาดได้
                # เช่น overlay ที่มี pointer-events: none) ถ่าย screenshot จริงส่งให้
                # Gemini vision วิเคราะห์ ป้อนผลลัพธ์เข้า step ถัดไปเป็น context เสริม
                # (pending_vision_context ด้านบนสุดของ run_task()) — ห้าม throw ออกไป
                # กระทบ loop หลักเด็ดขาด (เหมือนทุก fallback อื่นในไฟล์นี้)
                if (
                    resolved_provider == "gemini"
                    and not result.success
                    and tool_input.get("type") in _VISION_FALLBACK_ACTION_TYPES
                ):
                    try:
                        screenshot_png = await page.screenshot(type="png")
                        pending_vision_context = await llm.describe_screenshot(
                            client, model, screenshot_png, tool_input.get("type"), tool_input.get("index"),
                        )
                        if verbose and pending_vision_context:
                            print(f"  [vision-fallback] {pending_vision_context}", flush=True)
                    except Exception as e:
                        if verbose:
                            print(f"  [vision-fallback] ล้มเหลว: {e}", flush=True)

                # W30: แจ้งเตือนชัดๆ ถ้าหน้าเว็บเปลี่ยนไปเองหลัง action ที่ไม่ได้ตั้งใจจะ
                # navigate (เช่น click ธรรมดาที่ดันมี redirect/JS navigation ซ่อนอยู่) —
                # user รายงานว่า agent บางครั้งดูเหมือนตัดสินใจ step ถัดไปจาก state เก่า
                # (คิดว่ายังอยู่หน้าเดิม) ทั้งที่จริงๆ หน้าเปลี่ยนไปแล้ว — get_snapshot()
                # ของ step ถัดไปอ่านสดจาก page จริงเสมออยู่แล้ว (ไม่มี cache ทางโค้ด) แต่
                # ไม่เคยมีสัญญาณชัดๆ บอกโมเดลตรงๆ ว่า "หน้าเปลี่ยนไปแล้วนะ อย่าเพิ่งเชื่อ
                # แผนเดิม" มาก่อน — เช็คแม้ result.success=True ด้วย (การ "สำเร็จ" ที่แอบ
                # พาไปหน้าอื่นโดยไม่ตั้งใจอันตรายกว่า fail ธรรมดา เพราะโมเดลอาจไม่รู้ตัวว่า
                # ต้อง re-evaluate) ไม่เช็คกับ goto/switch_tab/go_back เพราะ navigate คือ
                # จุดประสงค์หลักของ action พวกนี้อยู่แล้ว ไม่ต้องเตือนซ้ำ
                result_text = str(result)
                if tool_input.get("type") not in ("goto", "switch_tab", "go_back") and page.url != url_before_action:
                    result_text += (
                        f"\n[หน้าเว็บเปลี่ยนไปเองหลัง action นี้: จาก {url_before_action} เป็น "
                        f"{page.url} — แผนเดิมอาจไม่ตรงกับหน้าปัจจุบันแล้ว ตรวจสอบ indexed "
                        "elements ของหน้าใหม่นี้ก่อนตัดสินใจ action ถัดไป]"
                    )
                messages = append_tool_result(messages, tool_use_id, result_text)

                # W7[A] (context compaction) / W22 (ทุก provider): เก็บ boundary ของ
                # step นี้ แล้วเช็คว่าต้องบีบอัดหรือยัง (ดูคอมเมนต์ยาวที่
                # _COMPACT_AFTER_STEPS ด้านบนสุดของไฟล์) — compact_messages มาจาก
                # _llm_backend() ตาม resolved_provider เอง (Anthropic/Groq/Gemini คนละ
                # ฟังก์ชัน คนละ wire format แต่ contract เดียวกัน)
                step_boundaries.append((steps_taken, len(messages)))
                if len(step_boundaries) > _COMPACT_AFTER_STEPS:
                    cut_list_index = len(step_boundaries) - _KEEP_RECENT_STEPS
                    cut_step_num, cut_at = step_boundaries[cut_list_index - 1]
                    # W50: สรุปเฉพาะ step "ใหม่" (delta) ตั้งแต่ digest_upto_step+1 ถึง
                    # cut_step_num นี้ ต่อท้าย digest_lines ที่สะสมมาจากรอบก่อนๆ แทนที่จะ
                    # สรุปซ้ำตั้งแต่ step 1 ทุกรอบ (ดู _build_history_digest() ด้านบนสุด
                    # ของไฟล์) แล้ว cap ด้วย _MAX_DIGEST_LINES กันไม่ให้ digest text โต
                    # ไม่มีเพดานตามความยาว task — คำนวณเป็น candidate ก่อน ยังไม่ commit
                    # เข้า digest_lines/digest_upto_step จริงจนกว่าจะรู้ว่า compact_messages()
                    # ด้านล่างสำเร็จจริง (ไม่ no-op) ไม่งั้นถ้า no-op แล้ว advance ไปก่อน
                    # จะเสีย step ที่เพิ่งสรุปไปฟรีๆ (ไม่เคยถูกแทรกเข้า messages จริงเลย)
                    delta_text = _build_history_digest(
                        self.memory, upto_step=cut_step_num, from_step=digest_upto_step + 1,
                    )
                    candidate_lines = list(digest_lines)
                    if delta_text:
                        candidate_lines.extend(delta_text.split("\n"))
                    dropped_count = 0
                    if len(candidate_lines) > _MAX_DIGEST_LINES:
                        dropped_count = len(candidate_lines) - _MAX_DIGEST_LINES
                        candidate_lines = candidate_lines[-_MAX_DIGEST_LINES:]
                    digest_text_lines = candidate_lines
                    if dropped_count > 0:
                        digest_text_lines = [
                            f"- (มี step ก่อนหน้าอีก {dropped_count} step ถูกย่อทิ้งแล้ว ไม่แสดงรายละเอียด)",
                        ] + digest_text_lines
                    digest = "\n".join(digest_text_lines)
                    len_before_compact = len(messages)
                    messages = compact_messages(messages, cut_at, digest)
                    # W22: อ้างจากความยาวจริงก่อน/หลัง แทนที่จะสมมติว่าลดลงเท่า cut_at
                    # เป๊ะ — compact_messages() อาจ no-op (คืน messages เดิมเป๊ะถ้ารูปแบบ
                    # ไม่ตรงคาด, removed=0) หรือลดลงน้อยกว่า cut_at จริง (Groq ต้องกัน
                    # system message ไว้ที่ index 0 เสมอ ดู _compact_groq_messages())
                    # ถ้าสมมติผิดจะได้ boundary เพี้ยนสะสมไปเรื่อยๆ ทำให้รอบบีบอัดถัดไป
                    # ตัดผิดตำแหน่ง (กลางบทสนทนา ไม่ใช่ต้น turn จริง)
                    removed = len_before_compact - len(messages)
                    if removed > 0:
                        # W50: commit candidate digest เข้า state สะสมจริง เฉพาะตอนที่
                        # compact_messages() แทรก digest เข้า messages สำเร็จจริงเท่านั้น
                        digest_lines = candidate_lines
                        digest_upto_step = cut_step_num
                        step_boundaries = [
                            (s, b - removed) for s, b in step_boundaries[cut_list_index:]
                        ]
                        if verbose:
                            print(
                                f"[context-compact] ย่อ step ..{cut_step_num} เหลือ digest สะสม "
                                f"{len(digest_lines)} บรรทัด (เก็บ {_KEEP_RECENT_STEPS} step ล่าสุดแบบ raw)",
                                flush=True,
                            )

                if tool_input.get("type") in _PAGE_CHANGING_ACTIONS:
                    await wait_stable(page)

                    # domain guard: classify_action() เช็ค allowlist แค่ตอน type=="goto"
                    # เท่านั้น — click ที่พาออกนอกโดเมน (เช่นลิงก์ "Sign in with Google"/
                    # OAuth, โฆษณา, redirect ในหน้าเดิม) ไม่ถูกจับตอน permission check
                    # เลย เพราะ perception.py ไม่ได้เก็บ href ของ element ไว้เช็คล่วงหน้า
                    # — เช็คซ้ำอีกชั้นหลัง action จบแล้วจริงแทน (defense-in-depth) ถ้าหลุด
                    # ออกนอก allowlist ให้ดึงกลับทันทีก่อนจะ perceive/ส่งให้ LLM เห็นหน้า
                    # นอกขอบเขต — ทำงานเฉพาะตอน effective_allowed_domains ถูกตั้งไว้จริง
                    # (ไม่ใช่ None) ไม่กระทบ task ปกติที่ไม่ได้จำกัดโดเมน
                    if effective_allowed_domains is not None:
                        current_domain = extract_domain(page.url)
                        if current_domain not in effective_allowed_domains:
                            try:
                                await page.go_back()
                                await wait_stable(page)
                            except Exception:
                                pass
                            domain_guard_msg = (
                                f"[BLOCKED] Action นี้พาไปยังโดเมนนอกขอบเขตที่อนุญาต "
                                f"({current_domain!r}) — ถูกดึงกลับอัตโนมัติแล้ว โดเมนที่"
                                f" อนุญาตสำหรับ task นี้: {sorted(effective_allowed_domains)}"
                                " ห้ามพยายามไปโดเมนนี้ซ้ำอีก"
                            )
                            if verbose:
                                print(f"  [domain-guard] {domain_guard_msg}", flush=True)
                            messages.append(_build_nudge_message(resolved_provider, domain_guard_msg))

                # W41: pacing delay ย้ายไปอยู่ก่อน next_action() แทน (ดู comment ตรงจุดที่
                # เรียก next_action() ด้านบน) — ไม่ sleep ซ้ำท้าย step แล้ว

            # W7[A] (long-term): บันทึกผลลัพธ์ของ task run นี้ไว้ให้ task run ถัดไป
            # (บน goal/หน้าเว็บที่เกี่ยวข้องกัน) recall() กลับมาใช้ได้ — เรียกครั้งเดียว
            # ตอนจบ loop จริง (ทุก path: finish_task, loop-detected, หมด max_steps)
            # ไม่ครอบ confirm_plan declined เพราะ return ไปก่อนถึงจุดนี้แล้ว (ไม่มี
            # action ใดๆ เกิดขึ้นจริงเลย ไม่มีอะไรให้บันทึกเป็น pattern)
            #
            # W41: ยิงเป็น background task (ไม่ await) — ดู docstring ของ
            # _fire_and_forget/_background_tasks ด้านบนสุดของไฟล์ ผลลัพธ์ของ task run
            # นี้ (return ด้านล่าง) ไม่ต้องรอ record_task() เสร็จก่อนเลย
            _fire_and_forget(asyncio.to_thread(
                long_term_memory.record_task,
                url=url, goal=goal, success=success, message=final_message,
                failed_actions=self.memory.failed_actions_summary(),
                session_id=session_id or "",
            ))

            # W_procmem: กลั่น trajectory ของ task ที่สำเร็จแล้วเป็น template ให้
            # core/procedural_memory.py เก็บไว้ reuse ในอนาคต (ดู core/llm.py::
            # abstract_trajectory, core/procedural_memory.py::save_template) — เรียกแค่
            # ตอน success=True เท่านั้น (template จาก task ที่ล้มเหลวไม่มีประโยชน์ให้
            # replay ซ้ำ) ยิงเป็น background task เหมือน record_task ด้านบนทุกประการ —
            # ไม่ await ผลลัพธ์ของ task run นี้ไม่ต้องรอ Abstractor เสร็จก่อนเลย และ
            # settings.enable_procedural_memory_capture (default True, ดู config.py)
            # เป็นแค่ "ฝั่งเขียน" ปิดแยกจาก enable_procedural_memory (ฝั่งอ่าน) ได้
            if settings.enable_procedural_memory_capture and success:
                async def _run_abstractor() -> None:
                    try:
                        template = await llm.abstract_trajectory(
                            client, model, goal, url, self.memory.all(), resolved_provider,
                        )
                        if template is not None:
                            await asyncio.to_thread(
                                procedural_memory.save_template, extract_domain(url), template,
                            )
                    except Exception as e:
                        print(f"⚠️ Procedural Memory abstractor hook error: {e}", flush=True)

                _fire_and_forget(_run_abstractor())

            # W19-3 (ดู llm.py::generate_persona_message, config.py::enable_persona_voice):
            # แปลง final_message ดิบให้เป็นข้อความไทยธรรมชาติ เรียกแค่ตอนจบ task เท่านั้น
            # (ความถี่ต่ำสุด ไม่ได้ผูกกับทุก browser action step) — additive ล้วนๆ: เพิ่ม
            # key "persona_message"/"persona_status" ต่อจาก "message"/"success" เดิม ไม่
            # แก้/ลบอะไรที่มีอยู่แล้วเลย (raw message/history ยังส่งครบเหมือนเดิมทุกประการ)
            persona_message = ""
            persona_status = "COMPLETED" if success else "FAILED"
            if settings.enable_persona_voice:
                persona = await llm.generate_persona_message(
                    client, model, extract_domain(url), goal, persona_status, final_message, resolved_provider,
                )
                persona_message = persona.get("user_message", "")
                persona_status = persona.get("action_status", persona_status)
                if persona_message:
                    await _emit({
                        "kind": "persona_message", "user_message": persona_message, "action_status": persona_status,
                    })

            return {
                "success": success,
                "steps": steps_taken,
                "message": final_message,
                "history": self.memory.recent(max_steps),
                "tokens": _tokens_dict(total_usage),
                "plan": plan_text,
                "final_page_state": final_page_text,
                "persona_message": persona_message,
                "persona_status": persona_status,
                "completion_verification": completion_verification,
            }
        finally:
            if managed_externally:
                # page= มาจาก session registry (routes.py::create_task ผ่าน
                # core/session_registry.py::SessionRegistry) — ผู้เรียกเป็นคนคุม
                # lifecycle เต็มๆ ข้ามหลาย run_task() call จนกว่า user จะปิด session เอง
                # (POST /sessions/{id}/close) ห้ามปิด/คืนอะไรที่นี่เด็ดขาดไม่ว่ากรณีใด
                pass
            elif connect_to_user_browser:
                # ห้าม browser.close()/context.close()/page.close() บน browser จริงของ
                # user เด็ดขาด ไม่ว่า tab นั้นจะเป็น tab ที่ agent เปิดเองหรือไม่ก็ตาม —
                # เดิมเคย page.close() ตอน opened_new_tab=True แต่กลายเป็นบั๊กจริง: task/
                # เทิร์นถัดไปในบทสนทนาเดียวกัน (เช่น follow-up command ใน Test Console)
                # หา tab เดิมด้วย domain matching (resolve_target_page()) ไม่เจอเลยเพราะ
                # ถูกปิดไปแล้ว เลยต้องเปิด tab ใหม่ทุกครั้ง ดูเหมือน "ทำงานต่อจากเดิมไม่ได้
                # เปิดหน้าต่างใหม่ตลอด" ทั้งที่ resolve_target_page() ทำงานถูกอยู่แล้ว —
                # แก้โดยปล่อย tab ไว้เสมอ (เหมือน keep_browser_open=True ของ owns_browser
                # ด้านล่าง) ให้ทั้ง user และเทิร์นถัดไปกลับมาใช้ tab เดิมต่อได้ — user ปิด
                # tab เองเมื่อไม่ต้องการแล้ว keep_browser_open ไม่มีผลใดๆ ในโหมดนี้เพราะ
                # ไม่มีอะไรให้ "ปิด/ไม่ปิด" ตั้งแต่แรกอยู่แล้ว (ไม่เคยปิดอะไรเลยไม่ว่ากรณีใด)
                #
                # playwright.stop() แค่ตัดการเชื่อมต่อ CDP ของ driver ตัวนี้เอง ไม่ใช่การ
                # สั่งปิด browser จริง (คนละความหมายกับ playwright.stop() ใน owns_browser
                # ด้านล่างที่ปิด process ที่ตัวเอง launch เอง)
                await playwright.stop()
            elif owns_browser:
                if not keep_browser_open:
                    await browser.close()
                    await playwright.stop()
                # keep_browser_open=True: ปล่อย browser/playwright ค้างไว้โดยตั้งใจ — ไม่มี
                # ใคร close() ให้อีกจากโค้ดฝั่งนี้ต่อจากนี้ (ผู้ใช้ปิดหน้าต่าง browser เอง
                # ทีหลัง) รู้อยู่แล้วว่า playwright driver process จะค้างอยู่เบื้องหลังจนกว่า
                # จะปิด แลกกับ requirement ที่ user ขอไว้ตรงๆ ว่าไม่ต้องปิดจนกว่าจะปิดเอง
            else:
                # context ที่ยืมจาก pool ต้องคืนกลับเสมอไม่ว่า keep_browser_open จะเป็นอะไร
                # (ดู docstring ของ keep_browser_open ด้านบน) ไม่งั้น pool จะรั่วทีละ context
                await context.close()

    async def run_fastpath(
        self,
        url: str,
        goal: str,
        template_id: str,
        steps: list[dict],
        slot_values: dict,
        max_steps: int = 30,
        provider: Optional[str] = None,
        ask_user_func: Optional[AskUserFunc] = None,
        on_event: Optional[OnEventFunc] = None,
        browser: Optional[Browser] = None,
        page: Optional[Page] = None,
        session_id: Optional[str] = None,
    ) -> dict:
        """W_procmem: รัน procedural template (ดู core/procedural_memory.py) ผ่าน
        fastpath_executor.execute_template() แทน perceive->plan->act loop เต็มรูปแบบ
        ของ run_task() — ประหยัด LLM call ต่อ step ทุกตัวตราบใดที่ replay สำเร็จตรงๆ

        รองรับ resource model แค่ 2 แบบใน v1 นี้ (ตรงกับ 2 branch ที่ใช้บ่อยที่สุดใน
        routes.py::_run_with_resolved_browser): page= (session-managed, จาก
        session_registry — ผู้เรียกคุม lifecycle เอง ไม่ปิด/ไม่คืนอะไรที่นี่) หรือ
        browser= (ยืมจาก BrowserPool — เปิด context+page ใหม่ แล้วปิดแค่ context ตอน
        จบเสมอ) — โหมดอื่น (connect_to_user_browser, launch หน้าต่างเองแบบ
        headless=False ฯลฯ) ยังไม่รองรับ ผู้เรียกต้องเช็คเองก่อนเรียกฟังก์ชันนี้ ไม่งั้น
        fallback ไปเรียก run_task() ตรงๆ แทน (เท่ากับปิด fast-path ไปเงียบๆ สำหรับโหมด
        ที่ยังไม่รองรับ ไม่ raise error)

        escalation (ดู core/fastpath_executor.py module docstring): ถ้า template
        replay ล้มเหลวเกิน repair quota จะเรียก run_task() แบบเต็มรูปแบบ "บน page
        เดียวกัน" เสมอ (ไม่ acquire browser ซ้ำสองรอบ) แล้วคืนผลลัพธ์นั้นตรงๆ — page
        อาจ navigate ไปแล้วบางส่วนจาก step ที่ทำสำเร็จก่อนล้มเหลว แต่ run_task()'s W12
        "detect หน้าปัจจุบัน" จะ perceive ต่อจากจุดนั้นเองอยู่แล้ว ไม่ใช่เริ่มนับหนึ่งใหม่

        W_procmem (แก้ไขหลังพบบั๊กจริงระหว่าง Phase 4 validation): navigate + เรียก
        _maybe_auto_login() เองที่นี่ก่อนส่งต่อให้ fastpath_executor.execute_template()
        เสมอ (mirror ลำดับเดียวกับ run_task() ด้านบนทุกประการ — goto/skip_initial_goto
        -> wait_stable -> _maybe_auto_login) — เดิม execute_template() navigate เองแต่
        ไม่เคยเรียก auto-login เลย ทำให้ domain ที่มี credential เก็บไว้ (ดู
        site_learning/auto_login.py) replay ไม่ได้เลยถ้าหน้าเป้าหมายต้อง login ก่อน
        (แม้ template จะไม่มี step login เองเพราะ auto-login เกิด "นอก" LLM loop เสมอ
        ไม่ว่าทางไหน)"""
        if page is not None and browser is not None:
            raise ValueError("run_fastpath: ส่ง page และ browser มาพร้อมกันไม่ได้ (เลือกอย่างใดอย่างหนึ่ง)")
        if page is None and browser is None:
            raise ValueError("run_fastpath: ต้องส่ง page หรือ browser มาอย่างใดอย่างหนึ่ง")

        owns_context = False
        context = None
        if page is None:
            context = await browser.new_context()
            page = await context.new_page()
            owns_context = True

        resolved_provider = provider or settings.llm_provider
        client, model, _, _, _ = self._llm_backend(resolved_provider)

        async def _emit(event: dict) -> None:
            if on_event is not None:
                await on_event(event)

        async def _run_task_fallback() -> dict:
            return await self.run_task(
                url=url, goal=goal, max_steps=max_steps, provider=resolved_provider,
                ask_user_func=ask_user_func, on_event=on_event, page=page, session_id=session_id,
            )

        try:
            current_domain = extract_domain(page.url) if page.url not in ("about:blank", "") else ""
            target_domain = extract_domain(url)
            site_already_open = bool(target_domain) and current_domain == target_domain
            if not site_already_open:
                await goto(page, url)
            await wait_stable(page)

            auto_login_failure_reason = await _maybe_auto_login(page, verbose=False)
            if auto_login_failure_reason:
                await _emit({
                    "kind": "auto_login_failed",
                    "message": "ล็อกอินไม่สำเร็จด้วย credential ที่บันทึกไว้สำหรับเว็บนี้",
                    "reason": auto_login_failure_reason,
                })

            return await fastpath_executor.execute_template(
                page=page, url=url, goal=goal, template_id=template_id, steps=steps,
                slot_values=slot_values, client=client, model=model, provider=resolved_provider,
                on_event=on_event, run_task_fallback=_run_task_fallback,
            )
        finally:
            if owns_context:
                await context.close()

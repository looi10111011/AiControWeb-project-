"""Agent Loop: Perceive -> Plan -> Act -> Verify.

W1: skeleton only. W4: ทำ loop จริงกับเว็บง่าย 1 หน้า.
W5: retry action ที่ล้มเหลว (ดู actions.py::_dispatch_with_retry) + guard กัน
finish_task(false) ก่อนเวลาอันควร (ด้านล่าง) + permission layer/human-in-the-loop
"""

import asyncio
from difflib import SequenceMatcher
import base64
import re
import sys
import time
import urllib.parse
from typing import Awaitable, Callable, Optional

from playwright.async_api import Browser, Page, Playwright, async_playwright

from backend.app.config import settings
from backend.app.core import fastpath_executor
from backend.app.core import llm
from backend.app.core import long_term_memory
from backend.app.core import procedural_memory
from backend.app.core import state_filter
from backend.app.core.actions import (
    REJECTED_BY_USER_MESSAGE,
    ActionResult,
    AskUserFunc,
    execute,
    goto,
    wait_stable,
)
# W_delete_all_intent: ใช้ตัวดึง key=value ตัวเดียวกับที่ W_deterministic_count ใช้ใน
# read_page_data (ดู _goal_condition_values ด้านล่าง) — import ชื่อเดียวแทนที่จะ copy regex
# มาไว้อีกที่ กัน 2 ที่ที่ต้องแก้พร้อมกันแบบ dom_locator/extractor
from backend.app.core.actions import _KEY_VALUE_IN_QUERY_RE
# W_count_answer_check: ทั้งตัวอ่านตัวเลขที่โค้ดนับไว้ (system_counted_conditions) และชุดคำที่
# บ่งบอกว่า goal เป็นคำถามเชิงนับ อยู่ที่ actions.py ที่เดียวกับที่สร้างข้อความนั้นขึ้นมา —
# import ชื่อมาใช้ ไม่ก็อป format/keyword มาไว้อีกที่
from backend.app.core.actions import _COUNT_QUERY_KEYWORDS, system_counted_conditions
from backend.app.core.goal_intent import canonical_intent, contains_keyword
from backend.app.core.memory import ShortTermMemory, clip_result
from backend.app.core.perception import get_snapshot
from backend.app.core.user_browser import connect_user_browser, resolve_target_page
from backend.app.permission.rules import DEFAULT_NEEDS_CONFIRMATION, extract_domain, install_ssrf_guard
from backend.app.rag import retriever
from backend.app.rag.chroma_client import _embedding_function

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
    "This finish_task(success=false) is not accepted yet — steps remain, and the current "
    "page may still hold elements you can act on (e.g. a button not yet pressed, a field "
    "still empty). Look at the latest indexed elements again and try an action you haven't "
    "tried. If you genuinely cannot proceed after that, call finish_task(success=false) again."
)

# W5[A] "Verify" (2026-07-15): W5 เดิมทำแค่ "Retry" (actions.py::_dispatch_with_retry)
# ไม่มี "Verify" เลย — ช่องโหว่ symmetric กับ guard ด้านบน: LLM อาจเรียก
# finish_task(success=true) เป็น action แรกสุดโดยไม่ทำอะไรเลย (steps_taken=0) แล้ว
# ระบบจะยอมรับทันทีโดยไม่มีการตรวจสอบใดๆ เลย (ต่างจาก false ที่มี guard คู่กันอยู่แล้ว)
# — SYSTEM_PROMPT ขอไว้แล้วว่า finish_task(true) ต้องมีหลักฐานจาก indexed elements
# แต่ไม่เคยมีการบังคับด้วยโค้ดเลย ไม่ block เด็ดขาด (บาง goal อาจสำเร็จอยู่แล้วตั้งแต่
# page แรกจริงๆ เช่น "verify ว่าอยู่หน้า login") แค่ให้ยืนยันอีกครั้งก่อนเหมือนกัน
_MAX_PREMATURE_TRUE_FINISH_RETRIES = 1

# W_resume ("Mid-Task Input Request"): request_user_input หยุด loop รอ human ตอบแล้วทำต่อ
# (ดู llm.py::REQUEST_USER_INPUT_TOOL) — จำกัดจำนวนครั้งที่เรียก tool นี้ได้ต่อ task กัน
# LLM วนถามไม่รู้จบโดยไม่มีความคืบหน้าจริง (เช่น ถามซ้ำเพราะไม่ยอมอ่านคำตอบที่ได้มาแล้ว) —
# เกินโควตานี้แล้วปฏิเสธไม่ให้หยุดรออีก ป้อน tool_result บอกเหตุผลแล้วบังคับให้ตัดสินใจเอง
# ต่อ (ไม่ finish_task ทันทีเงียบๆ — ให้ LLM เห็นเหตุผลแล้วเลือก action ต่อไปเอง เหมือน
# escape valve อื่นในไฟล์นี้)
_MAX_REQUEST_USER_INPUT_CALLS = 3

# W_token_cut W3 (หลักฐานจาก live baseline 2026-09-02): เทิร์น LLM ที่ไม่ได้ลงมือทำอะไร
# เกือบทั้งหมดคือ guard ปฏิเสธ action ซ้ำ + finish_task ที่ถูกตีกลับด้วยเหตุผลเดิม —
# ไม่ลดความปลอดภัย แค่ไม่ให้ "เตือนเรื่องเดิมซ้ำ" กินเทิร์น LLM เต็มๆ อีก
#
# _MAX_TASK_GUARD_REJECTIONS: guard ปฏิเสธรวมทุกเหตุผลทั้ง task เกินนี้ = โมเดลติดลูป
#   แก้ตัวจริง จบ task ตามความจริงแทนการเผา step budget ต่อ (ตัวนับ guard_rejections เป็น
#   monotonic ไม่ reset ต่างจากโควตาต่อ guard ที่ reset ตอน action สำเร็จ — ดู W1)
# _MAX_SAME_GUARD_REASON_REJECTIONS: เหตุผลเดียวโดนซ้ำเกินนี้ (ข้ามรอบ reset ได้) = ลูป
_MAX_TASK_GUARD_REJECTIONS = 9
_MAX_SAME_GUARD_REASON_REJECTIONS = 4

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
    "[Rejected] This is a qa_summary question (asking for information, not issuing a "
    "command), so actions that change the page (fill/select/goto/...) are not allowed — the "
    "only exceptions are fill/click on a search box/search button/filter, or clicking a "
    "menu/nav item to reach another page holding the information you need. To read more "
    "content use type: 'read_page_data', then call finish_task with your final answer as "
    "soon as you can answer the question. Never conclude 'there is no data' before trying to "
    "search or navigate at least once."
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
    "\n\n[Answer guidance — important]: answer only what the user actually asked for. Never "
    "bolt on details nobody asked about (if they asked only for \"the names\", give just the "
    "names — no job title/office/salary they didn't ask for). If the answer has multiple "
    "entries, always format them as a list with one item per line (e.g. \"1. ...\\n2. ...\\n"
    "3. ...\"); never run them together into one long paragraph."
)

_PREMATURE_TRUE_FINISH_NUDGE = (
    "This finish_task(success=true) comes with no action having happened at all in this task "
    "(steps_taken=0) — before confirming success, check again that the latest indexed "
    "elements really do give clear evidence the goal is done. If they genuinely do, call "
    "finish_task(success=true) again. If you are unsure, perform an action related to the "
    "goal first."
)

# ACC-3 (accuracy audit follow-up): symmetric กับ guard ด้านบน (steps_taken==0) และ
# validation-error guard ด้านล่าง แต่เช็คสัญญาณอีกแบบที่ทั้งคู่จับไม่ได้ — โมเดลอาจลอง
# action จริงหลาย step (steps_taken > 0 ผ่าน guard แรกไปแล้ว) แต่ action ที่ "มีผลจริงต่อ
# หน้าเว็บ" (fill/click/select/...) ล้มเหลวทุกครั้งเลยสักครั้งเดียว (เช่น index ผิด/element
# หาไม่เจอซ้ำๆ) แล้วยังเรียก finish_task(success=true) — ไม่มี validation error ให้
# _scan_validation_errors() จับด้วย (เพราะ action ไม่เคยสำเร็จจนถึงจะ trigger validation
# ได้ด้วยซ้ำ) ต้องเช็คจาก ShortTermMemory ตรงๆ ว่ามี mutating action ไหนสำเร็จอย่างน้อย 1
# ครั้งไหมตลอดทั้ง task — ไม่นับ read_page_data/wait/hover/scroll (ไม่ใช่ "ความคืบหน้า"
# ต่อ goal โดยตรง แค่สำรวจ/รอ/เลื่อนจอเฉยๆ) เป็น escape valve เดียวกับ guard อื่นในไฟล์นี้
# (ปล่อยผ่านตามที่โมเดลยืนยันถ้า retry ครบโควตาแล้วยังไม่เปลี่ยนใจ ไม่ block เด็ดขาด)
_MAX_PREMATURE_ALL_FAILED_RETRIES = 2
# หมายเหตุ: ห้ามใส่ "goto"/"go_back"/"switch_tab" — แค่เปลี่ยนหน้า/สลับ tab ไม่นับเป็น
# "ทำอะไรสำเร็จต่อ goal" แถม goto แรกสุด (ไปที่ url ตั้งต้นก่อนเข้า loop) ถูก record เป็น
# success=True เสมอทุก task อยู่แล้ว ถ้านับรวมด้วย guard นี้จะไม่มีวันยิงเลยในทางปฏิบัติ
_MUTATING_ACTION_TYPES = {
    "click", "fill", "select", "check", "press_key",
    "submit", "delete", "purchase", "pay",
}
_PREMATURE_ALL_FAILED_NUDGE = (
    "This finish_task(success=true) is rejected — every page-changing action "
    "(fill/click/select/...) attempted in this task failed; not one succeeded. The goal "
    "cannot plausibly be complete when no action has ever succeeded even once. Re-check the "
    "index/element you chose (the index may be wrong, or the element not found) and try "
    "another route. Once something genuinely succeeds, call finish_task(success=true) again."
)


# W_readonly_goal_evidence (บั๊กจริง live-reproduce บน OrangeHRM 2026-08-26): guard ACC-3
# ด้านบนถามว่า "มี action ไหนสำเร็จบ้างไหม" แต่ไปวัดด้วย _MUTATING_ACTION_TYPES อย่างเดียว —
# goal ประเภทถามอย่างเดียว (เช่น "เจอกี่รายการ", "ชื่อสินค้าตัวแรกคืออะไร") ไม่มีวันมี
# mutating action สำเร็จได้เลยโดยธรรมชาติ เส้นทางที่ถูกต้องของมันคือ goto -> read_page_data
# -> finish_task เท่านั้น guard จึงปฏิเสธ finish_task(true) ที่ถูกต้องทุกครั้งจนหมดโควตา
# เสีย LLM call ทิ้ง 2 รอบต่อ task ฟรีๆ (เห็นในรัน live: "[finish_task(true) ไม่มี mutating
# action ไหนสำเร็จเลย 1/2]" ทั้งที่ read_page_data เพิ่งดึงตารางจริงมาได้สำเร็จ)
#
# read_page_data ที่สำเร็จคือ "หลักฐานว่าได้ข้อมูลจริงจากหน้าเว็บแล้ว" ซึ่งตรงกับเจตนาเดิมของ
# guard นี้เป๊ะ (กัน claim สำเร็จทั้งที่ไม่เคยมีอะไรทำงานเลย) — ต่างจาก goto/go_back/switch_tab
# ที่ตั้งใจไม่นับ เพราะ goto แรกสุดถูก record เป็น success=True ทุก task อยู่แล้ว ส่วน
# read_page_data ไม่เคยถูก record เองอัตโนมัติ จะโผล่มาก็ต่อเมื่อโมเดลสั่งเองและได้ข้อมูลจริง
_EVIDENCE_ACTION_TYPES = _MUTATING_ACTION_TYPES | {"read_page_data"}


def _has_any_successful_mutating_action(history: list[dict]) -> bool:
    return any(
        (h.get("cmd") or {}).get("type") in _EVIDENCE_ACTION_TYPES and h.get("success") is True
        for h in history
    )

# Task4 ("Task Completion Verifier", W19): symmetric กับ guard ด้านบนแต่เช็คคนละสัญญาณ —
# guard ด้านบนเช็คแค่ "steps_taken==0" (ไม่มีหลักฐานว่าทำอะไรเลย) ตัวนี้เช็ค "หน้าเว็บ
# ปัจจุบันมี validation error โผล่อยู่จริงไหม" (เช่น กรอกฟอร์มแล้วกด Save แต่ field ยังไม่
# ผ่าน validation — steps_taken > 0 แล้วแต่ยังไม่สำเร็จจริง guard เดิมด้านบนจับไม่ได้เพราะ
# เช็คแค่ step แรกสุด) — ทำงานอิสระจาก guard เดิม ไม่ทับซ้อนกัน (เช็คคนละเงื่อนไข ทำงาน
# พร้อมกันได้ทั้งคู่) ไม่ว่า steps_taken จะเท่าไหร่ก็ตาม
_MAX_PREMATURE_VALIDATION_ERROR_RETRIES = 2
_PREMATURE_VALIDATION_ERROR_NUDGE_TEMPLATE = (
    "This finish_task(success=true) is rejected — error/validation messages are still shown "
    "on the current page: {errors} Never treat the task as successful while these remain. "
    "Fix the relevant field per the error message first (e.g. fill an empty field, correct "
    "an invalid value, change a duplicate value), then retry. Once the errors are genuinely "
    "gone, call finish_task(success=true) again."
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


# W22 ("DOM-Based Post-Action Verification Guardrail" — hallucinated false-completion บั๊ก
# จริงที่ user รายงาน: agent ตอบ "No users with Role ESS found (or all have been deleted)"
# ทั้งที่หน้าเว็บจริงยังโชว์ "(3) Records Found" พร้อมแถว ESS user เหลืออยู่ครบ 3 แถว — สาเหตุ
# เดียวกับ validation-error guard ด้านบน: LLM สรุปจาก conversation history/ผลลัพธ์ action
# ก่อนหน้า (ซึ่ง hallucinate ได้) แทนที่จะอ่านสถานะ DOM จริง ณ ตอนเรียก finish_task — ต่างจาก
# guard เดิมตรงที่ guard เดิมมองหา "error message ที่ปรากฏ" (สัญญาณว่าฟอร์มยังไม่ผ่าน) ส่วน
# ตัวนี้มองหา "record count ที่เหลืออยู่จริงในตาราง" (สัญญาณว่า deletion goal ยังไม่เสร็จ) —
# เปิดใช้เฉพาะ deletion-intent goal เท่านั้น (ดู _is_deletion_intent_goal ด้านล่าง) ไม่ใช่ทุก
# goal เพราะข้อความ "X Records Found" ไม่เกี่ยวข้องกับ goal อื่นเลย (เช่น login, navigation)
# เช็คแล้วจะเป็นแค่ noise เปล่าๆ ไม่มีประโยชน์
_MAX_PREMATURE_DELETION_INCOMPLETE_RETRIES = 2

# W64[7.1]: ต่อท้าย {action_hint} ด้วยคำแนะนำที่ต่างกันตามว่า goal เป็นงานลบหรือแก้ไข (ดู
# _premature_mutation_action_hint() ด้านล่าง) — เดิม (W22) hardcode คำแนะนำ "กลับไปทำขั้นตอน
# ลบ" ตรงๆ ในนี้ ผิดถ้าใช้กับ edit-all-intent goal (ไม่มีอะไรให้ "ลบ")
_PREMATURE_DELETION_INCOMPLETE_NUDGE_TEMPLATE = (
    "This finish_task(success=true) is rejected — checking the current page's real DOM shows "
    "{count} entries still matching the condition (actual text on the page: \"{text}\"). "
    "Never treat the job as complete, or claim no matching entries exist, while the table "
    "genuinely still shows a count above 0 — {action_hint}until the count really reaches 0 "
    "or the page shows \"No Records Found\", only then may you call finish_task(success=true)."
)

# W22: คำที่บ่งบอกว่า goal นี้เป็นงานลบข้อมูล/รายการ — ครอบคลุมทั้งไทย/อังกฤษ (ไม่ผูกกับคำว่า
# "ทั้งหมด"/"all" เหมือน keyword ของ Batch/Bulk Action Protocol ด้านบน เพราะแม้แต่การลบรายการ
# เดียว/บางรายการ ก็เจอปัญหา false-completion แบบเดียวกันได้เหมือนกัน ไม่ใช่แค่ bulk delete)
_DELETION_INTENT_KEYWORDS = ("ลบ", "delete", "remove", "ล้าง")


def _is_deletion_intent_goal(goal: str) -> bool:
    """W22: True ถ้า goal มีคำที่บ่งบอกว่าเป็นงานลบข้อมูล/รายการ — ใช้เป็นเงื่อนไขเปิดใช้
    _scan_remaining_target_records() ก่อนยอมรับ finish_task(success=true) เท่านั้น (ดู
    docstring ของ _MAX_PREMATURE_DELETION_INCOMPLETE_RETRIES ด้านบนว่าทำไมต้อง scope แคบ)"""
    lower = (goal or "").lower()
    return any(kw in lower for kw in _DELETION_INTENT_KEYWORDS)


# W_plan_keeps_goal_verb: แผนที่ LLM ร่างมาต้องยังทำ "สิ่งเดียวกับที่ goal สั่ง" — บั๊กจริง
# live run 2026-08-27: goal สั่ง "ลบ user ที่ userrole=ess ออกให้หมด" แต่ planner ร่างแผนว่า
# "เปิดรายการผู้ใช้ที่พบแต่ละรายการเพื่อแก้ไข และเปลี่ยน Role ออกจาก ESS" แล้ว agent ก็เดินตาม
# แผนนั้นจริงๆ (กด Edit -> เปลี่ยน User Role) — user รายงานว่า "ไม่เดินตาม planner เลย" แต่
# ความจริงคือเดินตามแผนเป๊ะ ปัญหาอยู่ที่แผนผิดชนิดงานมาตั้งแต่ต้น
#
# scope แคบไว้ที่ deletion อย่างเดียวโดยตั้งใจ: "ลบ" มีทางเลือกที่ดูสมเหตุสมผลแต่ผิด (แก้ค่า
# แทนการลบ) และผลลัพธ์ต่างกันถาวร ส่วนงานอ่าน/ค้นหาไม่มี failure mode แบบนี้
# W_plan_warn_not_abort: ข้อความที่ยัดเข้า messages ตั้งแต่เทิร์นแรกเมื่อแผนที่ user ยืนยันมา
# ยังไม่ตรงชนิดงาน — เตือนโมเดลตรงๆ ว่าอย่าเดินตามส่วนที่ผิดของแผน ไม่ใช่หยุดทั้ง task
# W_plan_progress_stall: stall detector ที่มีอยู่ 3 ตัว (action ซ้ำเป๊ะ / label เดิม / วนเป็นคาบ
# 2-4) จับได้แต่ "ทำ action ซ้ำ" — agent ที่ทำ action *ต่างกันทุกครั้ง* แต่ไม่คืบหน้าตามแผนเลย
# ไม่มีตัวไหนจับได้ และ counter ที่มีทั้งหมดในไฟล์นี้เป็นแบบ "reset เมื่อสำเร็จ" ไม่ใช่ "นับว่า
# ผ่านไปกี่ครั้งแล้ว" จึงต้องมีตัวใหม่ — แต่ผูกกับ plan_cursor ที่มีอยู่แล้ว ไม่ได้สร้างนิยาม
# "ความคืบหน้า" ตัวที่สอง
#
# advisory ล้วน ไม่บล็อก: จุดที่รู้ผลอยู่ *หลัง* dispatch ซึ่ง nudge แบบ reject+continue ใช้ไม่ได้
# (จะทิ้ง tool_use ไว้โดยไม่มี tool_result ตอบ -> Anthropic/Groq error ดู docstring ของ
# _force_loop_recovery) จึงแนบข้อความไปกับผลของ action แทน — pattern เดียวกับ blocked_note
# ของ W_modal_check_on_failure ไม่เสียเทิร์น LLM เพิ่มและโมเดลเห็นทันที
# W_action_matches_plan_step (W111): W_plan_progress_stall ด้านล่างเป็นแค่ *ตัวนับ* — มันถาม
# ว่า "cursor ขยับไหม" ไม่ใช่ "สิ่งที่ทำตรงกับ step ที่กำลังทำอยู่ไหม" และ live run แสดงจุดอ่อน
# ของมันตรงๆ: โมเดลใส่ completed_plan_step มาแทบทุก action ทำให้ cursor ขยับตลอด ตัวนับจึงไม่มี
# วันถึงเกณฑ์ ทั้งที่งานไม่คืบเลย
#
# ตัวนี้จึง **ห้ามรีเซ็ตเมื่อ cursor ขยับ** (นั่นคือรูโหว่ของตัวนั้นพอดี) — รีเซ็ตเฉพาะเมื่อ
# action *ตรงกับ step* จริงๆ เท่านั้น
#
# advisory ล้วน ไม่บล็อก และเป็น "สัญญาณอ่อน" โดยเจตนา: ข้อความของ step เขียนโดย LLM จึงกว้าง/
# กำกวมได้ตลอด การเอาไปบล็อก action จะพังงานที่ถูกต้อง — เป้าหมายคือเตือนโมเดลให้กลับไปอ่าน step
_MIN_PLAN_STEP_WORDS_TO_JUDGE = 3
_MAX_ACTIONS_MISMATCHING_PLAN_STEP = 5
# W_plan_cursor_needs_a_matching_action: หน่วงการเดินหน้า cursor ได้มากสุดกี่ครั้งติดกัน —
# 2 พอที่จะกันการไล่ติ๊กรวดเดียว แต่ไม่มากจนแผนดูค้างในสายตาคนดู
_MAX_BLOCKED_CURSOR_ADVANCES = 2
_MAX_PLAN_MISMATCH_NOTES = 1

_PLAN_MISMATCH_NOTE_TEMPLATE = (
    " [None of your last {count} actions matched what the current plan step actually asks. "
    "Step {step} of {total} says: {step_text!r}. Read it again and do that — or, if this page "
    "cannot do it, go where it can instead of trying more variations here.]"
)

# คำที่ไม่ช่วยแยกแยะอะไรเลย ตัดทิ้งก่อนเทียบ ไม่งั้นเกือบทุก action จะ "ตรง" เพราะบังเอิญมีคำ
# เชื่อมเหมือนกัน — ชุดเล็กโดยเจตนา เอาเฉพาะคำที่โผล่ในแผนแทบทุกข้อของโปรเจกต์นี้
_PLAN_STEP_STOPWORDS = frozenset({
    "หน้า", "แล้ว", "และ", "ที่", "ของ", "ให้", "ไป", "จาก", "เพื่อ", "การ", "ใน", "กับ",
    "the", "and", "then", "for", "with", "from", "into", "page", "click", "on", "to", "a", "of",
})


def _plan_step_keywords(step_text: str) -> set[str]:
    """คำที่พอจะใช้บอกได้ว่า step นี้พูดถึงอะไร — ตัดคำเชื่อมและคำสั้นทิ้ง"""
    words = re.findall(r"[\w\u0e00-\u0e7f]+", (step_text or "").lower())
    return {w for w in words if len(w) >= 3 and w not in _PLAN_STEP_STOPWORDS}


def _action_matches_plan_step(step_text: str, action_label: str, action_type: str) -> Optional[bool]:
    """action นี้ดูเหมือนกำลังทำ step นี้อยู่ไหม

    None = ตัดสินไม่ได้ (step กว้าง/สั้นเกินไป หรือ action ไม่มี label ให้เทียบ) — ผู้เรียกต้อง
    ไม่นับเป็น mismatch เพราะ "อ่านไม่ออก" กับ "ทำผิด" คนละเรื่องกัน
    ใช้ _field_names_match() ที่ทนพิมพ์ผิด/คำไทยติดกันอยู่แล้ว ไม่สร้างตัวเทียบชุดที่สอง"""
    if not (action_label or "").strip():
        return None

    # W_action_matches_plan_step: แผนถูกเขียนเป็นภาษาไทยแต่ label ของเว็บเป็นอังกฤษเกือบเสมอ
    # ในโปรเจกต์นี้ — การเทียบ token ตรงๆ จึงตอบ "ไม่ตรง" ให้กับคู่ที่ถูกต้องอย่าง
    # step "แล้วกดค้นหา" กับปุ่ม "Search" ซึ่งจะทำให้ guard นี้เตือนผิดเป็นปกติ
    # ใช้ regex สองภาษาที่มีอยู่แล้วในไฟล์นี้เป็นสะพานข้ามภาษา แทนการสร้างพจนานุกรมใหม่
    # (ทุกตัวครอบทั้งไทย/อังกฤษอยู่แล้วเพราะถูกเขียนมาเพื่ออ่าน label ของเว็บไทยตั้งแต่ต้น)
    # ต้องเช็คก่อนเกณฑ์จำนวนคำด้านล่าง เพราะภาษาไทยไม่มีเว้นวรรค การตัดคำแบบ regex จึงได้
    # token เดียวยาวๆ ต่อประโยค ทำให้ step ภาษาไทยเกือบทุกข้อมีคำน้อยกว่าเกณฑ์และกลายเป็น
    # "ตัดสินไม่ได้" ทั้งหมด — guard จะไม่มีวันได้ทำงานเลยกับแผนที่ planner เขียนจริง
    for step_pattern, label_pattern in (
        (_SEARCH_LABEL_RE, _SEARCH_LABEL_RE),
        (_DESTRUCTIVE_LABEL_RE, _DESTRUCTIVE_LABEL_RE),
        (_ROW_ACTION_LABEL_RE, _ROW_ACTION_LABEL_RE),
        # ฝั่ง step ใช้ตัวไม่ผูก ^ (คำอยู่กลางประโยคเสมอ) ฝั่ง label ใช้ตัวผูก ^ ตามเดิม
        # เพื่อไม่ให้ "Saved Searches" นับเป็นปุ่มบันทึก — นี่คือเหตุผลที่ W103 แยกสองตัวไว้
        (_PLAN_COMMIT_STEP_RE, _RECORD_COMMIT_LABEL_RE),
    ):
        if step_pattern.search(step_text or "") and label_pattern.search(action_label or ""):
            return True

    keywords = _plan_step_keywords(step_text)
    if len(keywords) < _MIN_PLAN_STEP_WORDS_TO_JUDGE:
        return None
    haystack = _plan_step_keywords(f"{action_label} {action_type}")
    if not haystack:
        return None
    for word in haystack:
        if any(_field_names_match(word, key) for key in keywords):
            return True
    return False


_MAX_ACTIONS_WITHOUT_PLAN_PROGRESS = 4
_MAX_PLAN_STALL_NOTES = 2

_PLAN_STALL_NOTE_TEMPLATE = (
    " [You have taken {count} actions since the last plan step was completed, and the plan is "
    "still on step {step} of {total}: {step_text!r}. Re-read that step and do what it actually "
    "asks — if it is already done, say so by passing completed_plan_step on your next action; "
    "if it cannot be done on this page, go where it can be done instead of trying more "
    "variations here.]"
)

_PLAN_MISMATCH_WARNING_TEMPLATE = (
    "The confirmed plan does not match what the goal asks for: {reason}. Follow the goal, "
    "not that part of the plan — this goal only asks you to DELETE records. Never open an "
    "edit form and save it: changing a record is not deleting it, and it cannot be undone. "
    "Use each row's own delete action instead."
)

_PLAN_KEEPS_GOAL_VERB_CORRECTION = (
    "The plan you drafted does not delete anything. This goal is a DELETE task: the user "
    "asked for the matching records to be removed, not edited. Never substitute changing a "
    "field's value (e.g. switching a role to something else) for deleting the record — those "
    "have permanently different outcomes. Redraft the plan so it uses the row's own delete "
    "action and confirms the deletion, keeping every other step as it was."
)


def _plan_drops_goal_operation(goal: str, plan_text: str) -> Optional[str]:
    """คืน "เหตุผลที่แผนผิด" ถ้าแผนทิ้งชนิดงานที่ goal สั่งไป หรือ None ถ้าแผนใช้ได้

    (คืนเป็นข้อความ ไม่ใช่ bool เพราะผู้เรียกต้องบอกได้ว่าผิดกฎข้อไหน — ผู้เรียกทั้งหมดใช้ค่านี้
    เป็น boolean อยู่แล้ว จึงไม่กระทบพฤติกรรมเดิม)

    กฎ A — goal มีคำกลุ่มลบ แต่ทั้งแผนไม่มีคำกลุ่มลบเลยสักคำ
    เกณฑ์แคบและ deterministic: ไม่ตัดสินจากการที่แผน "มีคำแก้ไขด้วย" เพราะแผนลบที่ถูกต้องอาจมี
    ขั้นตอนตั้ง filter ที่ใช้คำว่า "เลือก/set" ได้ตามปกติ

    กฎ B (W_plan_commits_a_record_edit) — goal เป็นงานลบ *ล้วนๆ* แต่แผนมีขั้นตอนที่ "บันทึก"
    การแก้ไข record: งานลบล้วนไม่มีวันต้องกด Save ฟอร์มเลยแม้แต่ครั้งเดียว
    กฎ A อย่างเดียวไม่พอจริงตามที่ audit ท้วง — แผนที่ทำให้ agent ไปเปลี่ยน Role ของ user จริง
    ในรันที่ user รายงาน ("เปิดแต่ละรายการเพื่อแก้ไข -> เปลี่ยน Role -> บันทึก -> ลบสิทธิ์ ESS")
    มีคำว่า "ลบ" อยู่ด้วย จึงผ่านกฎ A ฉลุยทั้งที่ workflow เป็น edit ล้วน
    นี่คือกฎเดียวกับที่ runtime บังคับอยู่แล้วใน W_no_record_edit_for_delete_goal เพียงแต่ย้ายมา
    ตรวจตั้งแต่ตอนร่างแผน แทนที่จะรอไปบล็อกตอนจะกด Save จริง

    กฎ B gate ด้วย _goal_is_deletion_only() ไม่ใช่ _is_deletion_intent_goal() — goal ที่สั่งแก้ไข
    จริง ("เปลี่ยน Role ของทุกคนที่เป็น ESS เป็น Admin") ต้องไม่โดนกฎนี้เลย"""
    if not plan_text:
        return None
    if _is_deletion_intent_goal(goal) and not any(
        kw in plan_text.lower() for kw in _DELETION_INTENT_KEYWORDS
    ):
        return (
            "แผนไม่มีขั้นตอนลบเลยสักข้อ ทั้งที่ goal สั่งให้ลบ"
        )
    if _goal_is_deletion_only(goal):
        for line in _plan_step_lines(plan_text):
            if _PLAN_COMMIT_STEP_RE.search(line):
                return (
                    f"แผนมีขั้นตอนบันทึกการแก้ไข record ({line!r}) ทั้งที่ goal สั่งให้ลบอย่างเดียว "
                    "— งานลบล้วนไม่มีวันต้องกด Save ฟอร์ม"
                )
    return None


# W64[7.1] ("Filter Order & False Completion" — ticket Issue 7.1, บั๊กจริง: goal สั่งเปลี่ยน
# Role ของ user ที่ Role=ESS ทั้งหมดเป็น Admin — agent เห็น 1 แถว ESS เหลืออยู่จริงในตารางที่
# กรองแล้ว แต่ไม่กด Edit ให้ครบ ดันเรียก finish_task(success=true) เลย): ใช้ keyword ตาม
# W21 "Batch/Bulk Action Protocol — Edit All" ใน llm.py SYSTEM_PROMPT เดียวกัน — สาเหตุ
# false-completion เหมือน deletion เป๊ะ (LLM สรุปจากประวัติ conversation แทนอ่านสถานะ DOM
# จริง) เพราะสัญญาณ "งานเสร็จ" ของทั้งสองแบบเหมือนกันทุกประการในทางปฏิบัติ: filter ตาม
# เงื่อนไข target (เช่น Role=ESS) แล้วนับแถวที่ตรงเงื่อนไขในตารางที่กรองแล้ว ต้องเหลือ 0 ถึงจะ
# ถือว่าเสร็จ — ต่างแค่ action ที่ต้องทำกับแต่ละแถว (ลบ vs แก้ไขค่า) ไม่ใช่เงื่อนไขความสำเร็จ
#
# W68 (บั๊กจริงที่ user รายงาน: goal พิมพ์ว่า "...เปลี่ยน Role ของทุกคนในผลการค้นหาให้เป็น
# Admin..." — agent ตอบ "ไม่พบผู้ใช้ Role ESS (0 รายการ)" ทั้งที่ตารางจริงโชว์ "(16) Records
# Found" พร้อมแถว ESS อยู่จริง): keyword เดิมด้านบนต้องเจอ "เปลี่ยนทุก"/"แก้ทุก" ติดกันเป๊ะ
# เท่านั้น แต่คำพูดธรรมชาติจริงมักแยกคำ "เปลี่ยน...ทุกคน..." ห่างกันด้วยคำอื่น (เช่น "Role
# ของ") ทำให้ exact-phrase match พลาด — _is_edit_all_intent_goal() คืน False ทั้งที่ goal
# เป็น edit-all จริง เลยไม่เปิด _scan_remaining_target_records() guard ปล่อยให้คำตอบ
# hallucinate ของ LLM หลุดผ่านไปโดยไม่มีการตรวจ DOM จริงเลย — เปลี่ยนจาก exact-phrase
# matching เป็น "มี mutation verb + มี bulk marker อยู่ที่ไหนก็ได้ใน goal" แทน (ไม่ต้องติดกัน)
# ยัง cover keyword ชุดเดิมทั้งหมดได้อยู่ (เป็น superset) — ความเสี่ยง false-positive ต่ำ
# เพราะ guard ที่เปิดใช้ (_scan_remaining_target_records) เอง fail-safe อยู่แล้ว (ไม่เจอ
# element "(N) Records Found" บนหน้า = ปล่อยผ่านเงียบๆ ไม่ block อะไรเลย)
_EDIT_ALL_MUTATION_VERBS = ("เปลี่ยน", "แก้ไข", "แก้", "ปรับ", "update", "change", "edit", "set")
# "ทุก" คำเดียวพอ (ครอบคลุม "ทุกคน"/"ทุกราย"/"ทุกแถว"/"ทุกรายการ"/"เปลี่ยนทุก" ที่เป็น substring
# ของมันอยู่แล้วทั้งหมด — ไม่ต้องแจกแจงแยกทีละคำ) บวก "ทั้งหมด"/"ให้หมด" ที่ไม่มีคำว่า "ทุก" ปน
_EDIT_ALL_BULK_MARKERS = ("ทุก", "ทั้งหมด", "ให้หมด", "all", "every", "each")


def _is_edit_all_intent_goal(goal: str) -> bool:
    """W64[7.1]/W68: True ถ้า goal มีทั้งคำกริยาบ่งบอกการแก้ไข/เปลี่ยนค่า (_EDIT_ALL_
    MUTATION_VERBS) และคำบ่งบอกขอบเขต "ทุกแถว/ทั้งหมด" (_EDIT_ALL_BULK_MARKERS) อยู่ในข้อความ
    เดียวกัน ไม่จำเป็นต้องติดกันเป็นวลีเดียว (แก้บั๊ก W68 ที่คำพูดธรรมชาติมักแยกคำสองกลุ่มนี้
    ด้วยคำอื่นคั่นกลาง เช่น "เปลี่ยน Role ของทุกคน") — ใช้ร่วมกับ _is_deletion_intent_goal()
    เป็นเงื่อนไข OR เปิดใช้ _scan_remaining_target_records() ก่อนยอมรับ
    finish_task(success=true) (ดู docstring ของ _is_deletion_intent_goal ด้านบนสำหรับที่มา
    ของกลไกเดิม)"""
    lower = (goal or "").lower()
    has_verb = any(v in lower for v in _EDIT_ALL_MUTATION_VERBS)
    has_bulk_marker = any(m in lower for m in _EDIT_ALL_BULK_MARKERS)
    return has_verb and has_bulk_marker


_DELETE_MUTATION_ACTION_HINT = (
    "go back and continue deleting (tick the select-all checkbox + Delete Selected, or click "
    "delete row by row, as SYSTEM_PROMPT describes) from the entries still remaining "
)
_EDIT_ALL_MUTATION_ACTION_HINT = (
    "go back and continue editing (click the Edit icon on a row still remaining, change the "
    "value the goal asks for, then press Save, as SYSTEM_PROMPT describes) from the entries "
    "still remaining "
)


def _premature_mutation_action_hint(goal: str) -> str:
    """W64[7.1]: เลือกคำแนะนำที่ต่อท้าย _PREMATURE_DELETION_INCOMPLETE_NUDGE_TEMPLATE ตาม
    intent จริงของ goal — deletion เป็น default (เข้ากันได้กับพฤติกรรมเดิมของ W22 เป๊ะถ้า
    goal เข้าเงื่อนไข deletion-intent ด้วย แม้จะเข้าเงื่อนไข edit-all-intent พร้อมกันก็ตาม
    เพราะ keyword สองชุดแทบไม่ overlap กันในทางปฏิบัติ)"""
    if _is_deletion_intent_goal(goal):
        return _DELETE_MUTATION_ACTION_HINT
    return _EDIT_ALL_MUTATION_ACTION_HINT


# W64[7.1] ("Filter Order & False Completion" — ticket Issue 7.1, บั๊กจริง: agent เลือก
# Role=ESS ใน filter dropdown แล้วคลิกปุ่มแก้ไข (Edit) บนตารางทันที "ก่อน" คลิก Search เลย —
# แถวที่กด Edit จึงเป็นแถวเก่าจากตารางที่ยังไม่กรอง ไม่ใช่แถวที่ตรงเงื่อนไข ESS จริง): เดิมมีแค่
# คำแนะนำใน SYSTEM_PROMPT (W20 "No Redundant Search Submission", W63[3.1]) ให้ agent "ควร"
# กด Search ก่อน แต่ไม่มีอะไรบังคับจริงถ้าโมเดลเผลอข้ามไป — เพิ่ม hard guard ระดับโค้ด บล็อก
# การคลิกปุ่ม action ของแถวตาราง (Edit/View/Delete ฯลฯ — label ที่ ICON_CLASS_LABEL_RULES ใน
# perception.py W21 เดา/แปะให้แล้ว) ถ้า "1 step ก่อนหน้าทันที" คือ fill/select ที่สำเร็จ (บ่ง
# บอกว่าเพิ่งแก้ filter/dropdown แต่ยังไม่ได้กด Search ยืนยันเลย)
#
# ทำไม scope แคบแค่ "1 step ก่อนหน้าทันที" เท่านั้น (ไม่ใช่ "ตั้งแต่ fill ล่าสุดจนกว่าจะกด
# Search ไม่ว่าจะผ่านไปกี่ step"): กันไม่ให้ block ผิดกรณีที่ agent fill/select field ที่ไม่ใช่
# filter จริง (เช่น กรอกฟอร์ม Edit ที่เปิดอยู่แล้ว) แล้วบังเอิญ action ถัดไปห่างออกไปหลาย step
# เป็นปุ่ม Edit ของแถวอื่นที่ไม่เกี่ยวข้องกันเลย — window แคบสุดเท่าที่จำเป็นสำหรับ pattern
# ที่ user รายงานจริง (fill/select ตามด้วยคลิก row-action ทันที ไม่มี action คั่นกลาง) ลด
# false-positive ได้มากสุดโดยยังจับบั๊กจริงได้ครบ
# W_delete_all_intent (บั๊กจริงที่ user รายงาน + step trace 2026-08-26 ยืนยัน: goal "ลบ user
# ที่ userrole=ess ออกให้หมด" — agent ลบไป 1 แถวจาก 7 แถวที่เป็น ESS แล้ว claim success=True
# โดยไม่เคยกดปุ่ม Search เลยสักครั้งตลอด run): bulk marker (_EDIT_ALL_BULK_MARKERS ด้านบน) ถูก
# ใช้โดย _is_edit_all_intent_goal() เท่านั้น ซึ่ง *ต้องมี verb แก้ไข* ด้วย goal ลบทั้งหมดจึงไม่
# เข้าเงื่อนไขไหนเลย — สุทธิคือคำว่า "ให้หมด" ใน goal มีผลเป็นศูนย์ ไม่มีอะไรในระบบรู้ว่านี่คือ
# งาน "ลบทั้งหมด" ไม่ใช่ "ลบชิ้นเดียว"
#
# ระดับการบังคับที่ตกลงกับ user: บังคับ "ลำดับ" แต่ไม่บังคับ "วิธี" — ต้องเห็นว่าตารางกรองตรง
# เงื่อนไขจริงก่อนลบ และห้าม claim สำเร็จถ้ายังเหลือแถว แต่จะใช้ Select All หรือลบทีละแถว
# ปล่อยให้โมเดลเลือกเอง (เว็บที่ไม่มี Select All ต้องยังทำงานได้)
def _is_delete_all_intent_goal(goal: str) -> bool:
    """W_delete_all_intent: True ถ้า goal เป็นทั้งงานลบ (_is_deletion_intent_goal) และมีขอบเขต
    "ทุก/ทั้งหมด" (_EDIT_ALL_BULK_MARKERS) — ใช้ค่าคงที่เดิมทั้งสองชุด ไม่สร้าง keyword ใหม่
    ซ้อนขึ้นมาอีกชุด"""
    lower = (goal or "").lower()
    return _is_deletion_intent_goal(goal) and any(m in lower for m in _EDIT_ALL_BULK_MARKERS)


# W_count_answer_check (ปิดครึ่งหลังของ W_deterministic_count/W_conditional_count): โค้ดนับ
# ให้แล้วและแนบตัวเลขไปกับผลลัพธ์ทุกครั้งก็จริง แต่ไม่มีอะไรตรวจเลยว่าโมเดล "ใช้" ตัวเลขนั้นจริง
# หรือไม่ — บั๊กที่ user เจอสดคือ "ตารางมี 7 แถวที่เป็น ESS แต่ agent ตอบ 6" ซึ่งเป็นความผิด
# ที่เกิด *หลัง* ข้อมูลถูกอ่านมาถูกต้องแล้ว การแนบตัวเลขเป็นแค่คำแนะนำใน prompt ที่โมเดลจะ
# เพิกเฉยก็ได้ (และ P0 รอบนี้ก็เพิ่งพิสูจน์อีกรอบว่ากฎที่เขียนถูกครบแล้วโมเดลก็ยังทำไม่ครบ)
#
# ยิงเฉพาะตอนเงื่อนไขครบทั้ง 3 ข้อพร้อมกันเท่านั้น กัน false positive: goal เป็นคำถามเชิงนับ
# จริง + ค่าเงื่อนไขที่โค้ดนับไว้ปรากฏใน goal จริง + ตัวเลขที่โค้ดนับได้ "ไม่โผล่ในคำตอบเลย
# สักตัว" (ถ้าโมเดลตอบ 4 ตอนโค้ดนับได้ 4 ก็ผ่านทันที ไม่ต้องตีความประโยคภาษาธรรมชาติเลย)
_MAX_COUNT_ANSWER_MISMATCH_RETRIES = 2

_COUNT_ANSWER_MISMATCH_NUDGE_TEMPLATE = (
    "This answer is rejected. The system already counted this for you from "
    "the page's real data: there are exactly {count} entries matching '{value}'. Your answer "
    "does not contain that number anywhere, so it contradicts what was actually counted — and "
    "counting is not something to do by eye when the number has already been computed for you. "
    "Call finish_task again with {count} as the number, or read the page again if you believe "
    "the data has changed since it was counted."
)


def _count_answer_contradiction(
    goal: str, answer: str, system_counted: dict[str, int],
) -> Optional[tuple[str, int]]:
    """W_count_answer_check: คืน (ค่าเงื่อนไข, จำนวนที่โค้ดนับได้) คู่แรกที่คำตอบขัดกับตัวเลข
    ที่โค้ดนับไว้ หรือ None ถ้าไม่ขัด/ยังไม่เข้าเงื่อนไขให้ตรวจ

    ตรวจเฉพาะตอนเงื่อนไขครบทั้ง 3 ข้อพร้อมกัน (ดู _MAX_COUNT_ANSWER_MISMATCH_RETRIES ด้านบน)
    — จงใจไม่ตีความประโยคภาษาธรรมชาติเลยแม้แต่นิดเดียว แค่ถามว่า "ตัวเลขที่โค้ดนับได้โผล่อยู่ใน
    คำตอบไหม" ซึ่งเป็นคำถามที่ตอบได้แบบ deterministic 100% เหมือน guard อื่นในไฟล์นี้

    ใช้ร่วมกันทั้ง main loop และ qa_summary mini-loop โดยตั้งใจ — คำถามเชิงนับส่วนใหญ่ถูก
    classify_intent() ส่งเข้า mini-loop (ซึ่งเป็นที่ที่บั๊กจริงเกิด) แต่ goal ที่มีทั้งคำถาม
    และ action ปนกันจะไปจบที่ main loop แทน ถ้าเขียนแยกสองที่จะเพี้ยนออกจากกันแน่นอน"""
    if not system_counted or not _goal_asks_for_a_count(goal):
        return None
    numbers_in_answer = set(re.findall(r"\d+", answer or ""))
    goal_lower = (goal or "").lower()
    return next(
        (
            (value, count) for value, count in system_counted.items()
            if value in goal_lower and str(count) not in numbers_in_answer
        ),
        None,
    )


def _goal_asks_for_a_count(goal: str) -> bool:
    """W_count_answer_check: True ถ้า goal เป็นคำถามเชิงนับ — ใช้ keyword ชุดเดียวกับที่
    read_page_data ใช้เลือก lane อยู่แล้ว (actions._COUNT_QUERY_KEYWORDS) ไม่สร้างชุดใหม่ซ้อน"""
    lower = (goal or "").lower()
    return any(kw in lower for kw in _COUNT_QUERY_KEYWORDS)

# W_prompt_sections (P4.1): ตัดสินว่า step นี้ต้องส่งบล็อกไหนของ SYSTEM_PROMPT บ้าง (ดู
# llm.py::build_system_prompt สำหรับเหตุผลเต็มและเกณฑ์ว่าอะไร gate ได้)
#
# *** สะสมอย่างเดียว ไม่ถอดออก *** — ผู้เรียกส่ง set เดิมเข้ามาแล้ว union กับของรอบนี้ เหตุผล:
# (1) prompt ที่กระพริบไปมาระหว่าง step ทำให้ prefix cache ของ provider พลาดทุกครั้งที่สลับ
#     ซึ่งแพงกว่าการส่งบล็อกที่ไม่ได้ใช้ต่ออีกไม่กี่ step
# (2) กฎที่โมเดล "เคยเห็น" แล้วหายไปกลางทางเป็นพฤติกรรมที่ไล่บั๊กยากมาก (เช่น เปิด dropdown
#     ที่ step 3 แล้วกฎ W50 หายไปตอน step 4 เพราะ dropdown ปิดไปแล้ว)
#
# สัญญาณทุกตัวเป็น deterministic ที่ไฟล์นี้คำนวณอยู่แล้ว ไม่มี heuristic ใหม่:
#   plan     — มี plan_text จริงหรือไม่ (ตรงตัว ไม่ต้องเดา)
#   table    — goal เป็นงาน bulk/ลบ/แก้ทั้งหมด/คำถามเชิงนับ หรือหน้ามี checkbox เลือกแถว
#   widget   — snapshot มี <select> จริง หรือมี trigger ของ custom dropdown
#   password — ใช้ค่า allow_fill_secret ที่คำนวณไว้แล้วก่อนเรียก LLM (W_fill_secret_schema_gate)
_TABLE_ELEMENT_LABEL_HINTS = ("select row", "select all", "records found")
_WIDGET_ELEMENT_LABEL_HINTS = ("-- select --", "--select--")


# W_tab_rebind (P3.6): actions.switch_tab() เรียก bring_to_front() แล้ว return เฉยๆ — ตัวแปร
# `page` ที่ลูปหลักถืออยู่ไม่เคยถูกเปลี่ยน ทุก get_snapshot()/execute() หลังจากนั้นจึงยังยิงไป
# ที่แท็บเดิม ผลคือ agent "จ้องแท็บเก่า" แล้วดึง snapshot เดิมซ้ำๆ ไปเรื่อยๆ โดยไม่รู้ตัว
#
# เคสที่เจ็บกว่าคือแท็บที่เปิดเองจากการคลิกลิงก์ target="_blank" — โมเดลไม่เคยสั่ง switch_tab
# เลยด้วยซ้ำ จึงไม่มีทางแก้เองได้ (ไม่มี action ไหนที่มันเรียกแล้วจะกลับมาถูกแท็บ)
#
# แก้ที่ลูปหลักแทนที่จะแก้ใน actions.py: ตัว action ไม่ได้เป็นเจ้าของตัวแปร `page` ของลูป และ
# การให้ ActionResult พก Page object กลับมาจะทำให้ dataclass ที่ถูก str()/บันทึกลง history
# ต้องแบก object ที่ serialize ไม่ได้ไปด้วยโดยไม่จำเป็น
def _detect_tab_switch(page: Page, tabs_before: list, cmd: dict):
    """คืน (page ที่ควรใช้ต่อ, ข้อความอธิบาย) — คืน (page เดิม, "") ถ้าไม่ต้องเปลี่ยนอะไร

    ห้าม throw: browser/context อาจถูกปิดไปแล้วตอนถูกเรียก (task ที่กำลังจบ/ถูก stop) ซึ่ง
    ไม่ควรทำให้ step ที่ทำสำเร็จไปแล้วกลายเป็น error ย้อนหลัง"""
    try:
        tabs_after = list(page.context.pages)
    except Exception:
        return page, ""

    if cmd.get("type") == "switch_tab":
        index = cmd.get("tab_index")
        if isinstance(index, int) and 0 <= index < len(tabs_after) and tabs_after[index] is not page:
            target = tabs_after[index]
            return target, f"\n[Now operating on tab {index}: {target.url}]"
        return page, ""

    # แท็บใหม่โผล่มาเองจาก action นี้ (คลิกลิงก์ target="_blank" ฯลฯ) — ตามไปแท็บล่าสุดเสมอ
    # เพราะนั่นคือสิ่งที่ผู้ใช้จริงจะเห็นอยู่ตรงหน้าหลังคลิก
    opened = [t for t in tabs_after if t not in tabs_before]
    if opened:
        target = opened[-1]
        return target, (
            f"\n[This action opened a new tab and the agent has switched to it: {target.url} "
            "— the indexed elements you get next are from this new tab, not the previous one]"
        )
    return page, ""


# W_core_carries_situational_rules (รอบสอง): ปุ่ม/ช่องค้นหา — คนละชุดกับ
# _FORM_SUBMIT_LABEL_KEYWORDS โดยเจตนา ชุดนั้นคือ "ปุ่มที่บันทึกข้อมูล" ซึ่งไม่รวม Search
# (การค้นหาไม่ใช่การบันทึก และ permission layer ก็แยกสองอย่างนี้ออกจากกันด้วยเหตุผลเดียวกัน)
_SEARCH_CONTROL_LABEL_KEYWORDS = ("search", "ค้นหา", "filter", "กรอง", "go", "ok")


def _resolve_prompt_sections(
    previous: frozenset,
    *,
    goal: str,
    plan_text: Optional[str],
    elements: list[dict],
    allow_fill_secret: bool,
    manual_context: str = "",
) -> frozenset:
    sections = set(previous)
    if plan_text:
        sections.add("plan")
    # W_password_rules_arrive_too_late (บั๊กจริงที่ user รายงานพร้อมภาพหน้าจอ 2026-09-03:
    # goal "เปลี่ยนรหัสผ่านใหม่เป็น ..." แต่ agent เดินไป PIM > Update Password ของพนักงาน
    # แทนที่จะกดเมนูโปรไฟล์มุมขวาบน > Change Password): บล็อกกฎเรื่องรหัสผ่านมี W20 ที่สั่ง
    # เรื่องนี้ไว้ตรงตัวอยู่แล้ว ("ห้ามคลิก My Info — ให้กด User Dropdown มุมขวาบนก่อน") แต่
    # เดิมส่งเฉพาะตอน allow_fill_secret ซึ่งเป็น True ก็ต่อเมื่อ *ยืนหน้าฟอร์มเปลี่ยนรหัสผ่าน
    # อยู่แล้ว* — คือหลังจากเลือกทางผิดไปแล้ว โมเดลจึงไม่เคยเห็นกฎตอนที่ต้องตัดสินใจเลือกทาง
    # ส่งตั้งแต่ตอนที่ goal/แผนพูดถึงการเปลี่ยนรหัสผ่าน (sections สะสมข้าม step อยู่แล้ว
    # ส่งครั้งเดียวก็อยู่ยาวทั้ง task)
    if allow_fill_secret or _goal_or_plan_requests_password_change(
        f"{goal} {plan_text or ''}"
    ):
        sections.add("password")
    if (
        _is_deletion_intent_goal(goal)
        or _is_edit_all_intent_goal(goal)
        or _goal_targets_existing_records_only(goal)
        or _goal_asks_for_a_count(goal)
    ):
        sections.add("table")
    # W_core_carries_situational_rules: บล็อกที่ย้ายออกจาก core มา gate ตาม marker ที่กฎนั้น
    # พูดถึงเอง — ทริกเกอร์ตรงตัวกับสิ่งที่อยู่บนหน้าจริง ไม่ใช่การเดาจากถ้อยคำของ goal จึงไม่มี
    # ทางส่งไม่ทันเวลาที่ต้องใช้ (marker มาพร้อม snapshot ของ step เดียวกับที่โมเดลจะตัดสินใจ)
    # sections สะสมข้าม step อยู่แล้ว เห็นครั้งเดียวก็อยู่ยาวทั้ง task
    if manual_context and "[PRE_LEARNED_MANUAL]" in manual_context:
        sections.add("manual")
    for element in elements:
        label = str(element.get("label", "")).lower()
        tag = element.get("tag")
        if tag == "select" or any(h in label for h in _WIDGET_ELEMENT_LABEL_HINTS):
            sections.add("widget")
        if any(h in label for h in _TABLE_ELEMENT_LABEL_HINTS):
            sections.add("table")
        if "[already active]" in label:
            sections.add("marker_active")
        if "[disabled]" in label:
            sections.add("marker_disabled")
        if "[required]" in label:
            sections.add("marker_required")
        if _RECORD_COMMIT_LABEL_RE.search(label) or any(
            k in label for k in _FORM_SUBMIT_LABEL_KEYWORDS
        ):
            sections.add("save_toast")
            sections.add("search_submit")
        if any(k in label for k in _SEARCH_CONTROL_LABEL_KEYWORDS):
            sections.add("search_submit")
        if "may need to hover" in label:
            sections.add("marker_hover")
        if tag in ("input", "textarea", "select") or element.get("contenteditable"):
            sections.add("form_input")
            if "search" in label or element.get("type") == "search":
                sections.add("search_submit")
    # W_core_carries_situational_rules (รอบสอง): กฎ "label ซ้ำกันหลายตัว" ใช้ได้ก็ต่อเมื่อหน้านี้
    # มี label ซ้ำจริง — นับจาก snapshot ตรงๆ ไม่ต้องเดาจากถ้อยคำของ goal
    labels = [str(e.get("label", "")).strip().lower() for e in elements]
    labels = [l for l in labels if l]
    if len(labels) != len(set(labels)):
        sections.add("dup_labels")
    return frozenset(sections)


_URL_IN_GOAL_RE = re.compile(r"https?://\S+", re.IGNORECASE)


def _normalized_field_name(text: str) -> str:
    """W_filter_scope_guard: ยุบชื่อ field ให้เทียบกันได้ระหว่างที่ user พิมพ์ใน goal กับที่
    เว็บแสดงจริง — "userrole" (goal) กับ "User Role" (label บนหน้า) คือ field เดียวกัน
    ตัดทุกอย่างที่ไม่ใช่ตัวอักษร/ตัวเลขทิ้ง (ช่องว่าง ขีด ขีดล่าง) แล้วเทียบตัวพิมพ์เล็ก"""
    return re.sub(r"[^0-9a-z\u0e00-\u0e7f]+", "", (text or "").lower())


def _goal_condition_pairs(goal: str) -> list[tuple[str, str]]:
    """W_column_aware_rows: (ชื่อ field ที่ normalize แล้ว, ค่าที่ต้องการ) ทุกคู่ที่ user เขียนไว้
    ใน goal แบบ "key=value" — เป็น **แหล่งความจริงเดียว** ของทั้ง _goal_condition_fields()
    และ _goal_condition_values() ด้านล่าง (เดิมสองตัวนั้นวน regex เองคนละรอบ ซึ่งแปลว่ากฎการ
    ตัดURL/กันค่าซ้ำต้องแก้สองที่พร้อมกันตลอด)

    ตัด URL ทิ้งก่อนเสมอ — goal จริงมักมี URL ปนอยู่ด้วย ("ไปที่ https://x/?id=9 แล้วลบ user
    ที่ userrole=ess ให้หมด") ซึ่ง query string ของมันเข้าเงื่อนไข key=value เป๊ะๆ ทั้งที่ไม่ใช่
    เงื่อนไขของงานเลย ถ้าไม่ตัดจะได้เงื่อนไข AND ที่ไม่มีแถวไหนตรงได้เลย แล้ว guard จะบล็อก
    การลบที่ถูกต้องทิ้งไปเปล่าๆ

    คืน [] ถ้า goal ไม่ได้ระบุเงื่อนไขแบบนี้ = ไม่เปิด guard ที่พึ่งมันเลยสักตัว (ไม่เดาเงื่อนไข
    เองจากภาษาธรรมชาติ)"""
    goal_without_urls = _URL_IN_GOAL_RE.sub(" ", goal or "")
    pairs: list[tuple[str, str]] = []
    seen: set[str] = set()
    for raw_field, raw_value in _KEY_VALUE_IN_QUERY_RE.findall(goal_without_urls):
        value = raw_value.strip()
        if not value or value.lower() in seen:
            continue
        seen.add(value.lower())
        pairs.append((_normalized_field_name(raw_field), value))
    return pairs


def _goal_condition_fields(goal: str) -> list[str]:
    """W_filter_scope_guard: ฝั่งซ้ายของ "key=value" = field ที่ user "อนุญาต" ให้กรอง
    ("userrole=ess" -> ["userrole"])"""
    fields: list[str] = []
    for field, _ in _goal_condition_pairs(goal):
        if field and field not in fields:
            fields.append(field)
    return fields


# W_filter_scope_guard: perception ใส่ชื่อ field เป็น prefix ให้ dropdown trigger อยู่แล้ว
# (perception.py — "User Role: -- Select --", "Status: Enabled") ตัวคั่นคือ ": " ตัวแรก
# เท่านั้น เพราะค่าที่ตามมาอาจมี ":" ของตัวเองได้
_FIELD_LABEL_PREFIX_RE = re.compile(r"^([^:]{1,40}):\s")


def _filter_field_from_label(label: str, action_type: str = "") -> str:
    """ชื่อ field ที่ action นี้กำลังจะไปแตะ (normalize แล้ว) — "" ถ้า label ไม่ได้บอกชื่อ
    field มาเลย ซึ่งแปลว่าตัดสินไม่ได้ ต้องปล่อยผ่าน ไม่ใช่เดาแล้วบล็อก

    W_field_label_without_value (บั๊กจริง live stability check 2026-08-31): prefix
    "ชื่อ field: ค่า" จะมีก็ต่อเมื่อช่องนั้น *มีค่าอยู่แล้ว* — ช่องที่ยังว่าง label คือชื่อ field
    เปล่าๆ ("Username") ไม่มี ":" เลย เพราะตัวเติม prefix ข้ามไปเมื่อ label มีชื่อนั้นอยู่แล้ว
    ผลคือ guard ตาบอดพอดีตอนที่ช่องยังไม่ถูกแตะ — ซึ่งคือจังหวะเดียวที่มันต้องทำงาน
    (รันจริง: goal บอกแค่ userrole=ess แต่ agent ไป fill ช่อง "Username" ผ่านฉลุย)

    จำกัดไว้ที่ fill/select เท่านั้น: การ fill เล็งไปที่ form field เสมอ label ที่ไม่มี ":" จึงคือ
    ชื่อช่องแน่ๆ — ส่วน click ห้ามตีความแบบนี้เด็ดขาด ปุ่มชื่อ "Search"/"Delete" จะกลายเป็น
    "ชื่อ field" ทันทีแล้ว guard จะบล็อกทุกปุ่มบนหน้า"""
    # W_label_marker_key: "Username [required]" ต้องให้ชื่อ field เป็น "username" ไม่ใช่
    # "usernamerequired" ซึ่งจะไม่ match อะไรเลยแล้ว guard ก็เงียบไปเฉยๆ
    cleaned = _label_without_markers(label)
    match = _FIELD_LABEL_PREFIX_RE.match(cleaned)
    if match:
        return _normalized_field_name(match.group(1))
    if action_type in ("fill", "select"):
        return _normalized_field_name(cleaned)
    return ""


# W_filter_scope_guard: goal ที่ user พิมพ์จริงไม่ได้สะอาดเหมือนตัวอย่างในเทสต์ — ของจริงคือ
# "แล้บลบuserole=ess" (ไม่เว้นวรรคก่อน key เลย และสะกด userole ตัว r เดียว) ส่วนหน้าเว็บเขียนว่า
# "User Role" การเทียบแบบตรงตัวจึงไม่ match แล้ว guard จะไปบล็อก *การกดที่ถูกต้อง* ซึ่งแย่กว่า
# ไม่มี guard เลย — เทียบ 3 ชั้นจากเข้มไปหลวม และ fail-open เสมอเมื่อตัดสินไม่ได้
_FIELD_NAME_SIMILARITY_THRESHOLD = 0.8
# ดู W_field_match_min_length ใน _field_names_match() — ชื่อ field ที่สั้นกว่านี้ห้ามตัดสินด้วย
# กฎ "ครอบกันอยู่" เพราะมันเป็นส่วนประกอบของชื่ออื่นได้ง่ายเกินไป
_MIN_FIELD_NAME_CONTAINMENT_LENGTH = 5


def _field_names_match(goal_field: str, page_field: str) -> bool:
    """goal_field มาจากที่ user พิมพ์ page_field มาจาก label ที่ perception อ่านได้จริง
    (normalize มาแล้วทั้งคู่) — True = ถือว่าเป็น field เดียวกัน"""
    if not goal_field or not page_field:
        return False
    if goal_field == page_field:
        return True
    # "แล้บลบuserole" ครอบ "userole" อยู่ — คำไทยที่ติดมาหน้า key ไม่ควรทำให้ไม่ match
    #
    # W_field_match_min_length (MR3 จาก audit): กฎ "ครอบกันอยู่" หลวมเกินไปสำหรับชื่อสั้น —
    # goal "name=john" จะทำให้ "name" ครอบอยู่ใน "employeename"/"username"/"nationality"
    # ทั้งหมด แล้ว guard ก็ปล่อยให้ตั้งค่าช่องผิดผ่านไปได้ (false negative เงียบๆ)
    # ชื่อ field จริงที่สั้นกว่า 5 ตัวอักษรมีน้อยมาก (id/tel/url/name/type) แต่ชื่อที่ *มี*
    # คำสั้นพวกนี้เป็นส่วนประกอบมีเยอะมาก — ต่ำกว่านี้ให้ตัดสินด้วยกฎที่เข้มกว่าเท่านั้น
    shorter = min(len(goal_field), len(page_field))
    if shorter >= _MIN_FIELD_NAME_CONTAINMENT_LENGTH and (
        page_field in goal_field or goal_field in page_field
    ):
        return True
    # ตัดส่วนที่ไม่ใช่ ASCII ทิ้งแล้วลองใหม่ ("แล้บลบuserole" -> "userole")
    # W_field_match_min_length: ชั้นนี้ก็ใช้ containment เหมือนกัน จึงต้องมีเพดานความสั้น
    # เดียวกัน ไม่งั้น "name" ก็ยังรั่วไปตรงกับ "employeename" ผ่านทางนี้อยู่ดี
    ascii_goal = re.sub(r"[^0-9a-z]+", "", goal_field)
    if ascii_goal == page_field:
        return True
    if (
        ascii_goal
        and min(len(ascii_goal), len(page_field)) >= _MIN_FIELD_NAME_CONTAINMENT_LENGTH
        and (ascii_goal in page_field or page_field in ascii_goal)
    ):
        return True
    # เหลือแค่พิมพ์ผิดจริงๆ ("userole" vs "userrole") — ใช้ ASCII ฝั่ง goal เทียบ ไม่งั้น
    # คำไทยที่ติดมาจะถ่วง ratio ให้ต่ำจนไม่ match
    return SequenceMatcher(None, ascii_goal or goal_field, page_field).ratio() >= _FIELD_NAME_SIMILARITY_THRESHOLD


# W_prefer_row_delete: การกด "Edit" ถูกตั้งใจปล่อยผ่านมาตลอด (ดูเหตุผลเหนือ
# _RECORD_COMMIT_LABEL_RE: บางเว็บใช้หน้า Edit เป็นทางผ่านไปหาปุ่มลบ) — ยังคงเหตุผลนั้นไว้
# ทุกประการ เพียงแต่เพิ่มเงื่อนไข: ถ้า *บนหน้าเดียวกันนี้* มีปุ่ม/ไอคอนลบให้กดอยู่แล้ว การเข้า
# หน้า Edit ก็ไม่ใช่ "ทางผ่านที่จำเป็น" อีกต่อไป มันคือการเดินผิดทางเฉยๆ — และเดินผิดทางบน goal
# ที่สั่งลบเคยจบลงด้วยการเปลี่ยน Role ของ user จริงมาแล้ว (live run 2026-08-27)
# ไม่มีปุ่มลบในหน้า = ปล่อยผ่านเหมือนเดิม ห้ามบล็อก
# W_reject_obscured_click (P8/M3): perception ติดป้าย [obscured] ให้ element ที่ถูกของอื่นวางทับ
# มาตั้งแต่ต้น แต่ค้นทั้ง backend/ แล้ว **ไม่มีโค้ดส่วนไหนอ่านป้ายนี้เลยสักบรรทัด** (มีแต่ตัวที่
# สร้างป้าย + เทสต์ที่ assert ว่าป้ายมีอยู่) โมเดลจึงคลิกของที่ถูกบังได้เรื่อยๆ ครั้งละ ~12-18
# วินาทีที่รู้ล่วงหน้าอยู่แล้วว่าจะ timeout — ต่างจากป้ายพี่น้อง ([already active]/[disabled]/
# [hidden — may need to hover]) ที่มีกฎรองรับครบ
#
# *** ต้องมี dialog เปิดอยู่ด้วยถึงจะปฏิเสธ *** — comment ที่ perception.py อธิบายไว้ถูกต้องแล้ว
# ว่าจงใจเก็บ element ที่ถูกบังไว้ใน snapshot เพราะ overlay อาจหายไปเองก่อนถึงเวลาคลิกจริง
# (dropdown/tooltip ที่ปิดตัวเอง) การมี dialog เปิดค้างต่างหากคือสิ่งที่ทำให้ "ถูกบัง" กลายเป็น
# ถาวรจนคลิกไม่ได้แน่ๆ
# W_marker_registry (W108): perception เติม marker ต่อท้าย label 7 ตัว แต่ฝั่ง Python เคยตั้งชื่อ
# ไว้แค่ 3 ตัว ที่เหลือถูกพิมพ์เป็น literal ซ้ำหลายที่ ("[already active]" อยู่ 5 จุด) และ
# [disabled]/[required]/[hidden — ...] ไม่มีชื่อเลย — วันที่มีใครเพิ่ม marker ตัวที่ 8 จะไม่มีอะไร
# เตือนว่าต้องมาแก้ตัวตัดด้านล่างด้วย
#
# *** ทะเบียนนี้ไม่ใช่ของสำคัญที่สุด เทสต์ที่คู่กับมันต่างหาก ***
# test_orchestrator.py อ่านซอร์สของ perception.py จริงแล้วยืนยันว่า marker ทุกตัวที่ JS เติม
# มีอยู่ในทะเบียนนี้ครบ — นั่นคือสิ่งเดียวที่ทำให้ drift "ดัง" ขึ้นมาแทนที่จะเงียบ ซึ่งเป็นรูปแบบ
# ที่เจอซ้ำมาแล้วหลายครั้งในโปรเจกต์นี้ (comment/เจตนาถูก แต่โค้ดอีกฝั่งไม่ทำตาม)
_PERCEPTION_LABEL_MARKERS = (
    "[in open dialog]",
    "[Profile/Account Menu]",
    "[obscured]",
    "[hidden — may need to hover the row first]",
    "[disabled]",
    "[required]",
    "[already active]",
)

_ALREADY_ACTIVE_LABEL_MARKER = "[already active]"

# W_label_marker_key (W107): marker เป็น "สถานะชั่วคราวของ element" ไม่ใช่ "ตัวตน" ของมัน —
# ปุ่มเดิมที่บังเอิญถูก hover/ถูกบัง/อยู่ใน dialog ได้ label คนละสตริง โค้ดที่ใช้ label เป็นกุญแจ
# เทียบจึงมองว่าเป็นคนละ element
# บั๊กจริงที่ user เจอ: loop detector ใช้ (type, label) เป็นกุญแจ พอ agent สลับคลิกระหว่าง
# "Select row" กับ "Select row [hidden — may need to hover the row first]" ตัวนับก็รีเซ็ตทุกครั้ง
# ไม่มีวันถึงเกณฑ์ 4 -> เผา step จนหมด max_steps ทั้งที่ guard ตัวนี้ถูกเขียนมาเพื่อเคสนี้โดยตรง
#
# ตัดเฉพาะ marker ที่รู้จัก **ไม่ตัด [...] ทั่วไป** เพราะ label จริงของเว็บมีวงเล็บเหลี่ยมของ
# ตัวเองได้ (เช่นปุ่ม "[Beta] Export") การตัดมั่วจะทำให้ element คนละตัวกลายเป็นตัวเดียวกัน
def _label_without_markers(label: str) -> str:
    """label ที่ตัด marker ของ perception ออกหมดแล้ว — ใช้ตอนต้องการ "ตัวตน" ของ element
    เท่านั้น ห้ามใช้แทน label ดิบในที่ที่ตั้งใจตรวจ marker (ดู _OBSCURED_LABEL_MARKER ฯลฯ)

    marker ต่อท้ายเป็นปกติ แต่กลายเป็น label ทั้งก้อนได้ถ้า label เดิมว่าง และซ้อนกันได้หลายตัว
    จึงตัดทุกตำแหน่ง ไม่ใช่แค่ท้ายสตริง"""
    cleaned = label or ""
    for marker in _PERCEPTION_LABEL_MARKERS:
        cleaned = cleaned.replace(marker, " ")
    return " ".join(cleaned.split())


_OBSCURED_LABEL_MARKER = "[obscured]"
# ป้ายที่ perception ติดให้ element ที่อยู่ในกล่องโต้ตอบที่เปิดค้าง (W_dialog_in_snapshot)
_DIALOG_LABEL_MARKER = "[in open dialog]"
_MAX_OBSCURED_CLICK_RETRIES = 2

_OBSCURED_CLICK_NUDGE_TEMPLATE = (
    "[Rejected] '{label}' is behind a dialog that is currently open, so clicking it cannot "
    "work — it would only wait and time out. Deal with the dialog first: pick one of its own "
    "buttons to close it{dialog_hint}. Everything behind the dialog becomes clickable again "
    "once it is closed."
)


_MAX_PREFER_ROW_DELETE_RETRIES = 2

_PREFER_ROW_DELETE_NUDGE_TEMPLATE = (
    "[Rejected] '{label}' opens an edit form, but this goal only asks you to DELETE — and "
    "this page already shows a delete control you can use directly ('{delete_label}'). Going "
    "through the edit form risks changing a record instead of removing it, which cannot be "
    "undone. Use the delete action on the row you want removed instead."
)

# W_prefer_row_delete: perception ติดป้าย [Profile/Account Menu] ให้ element กลุ่มนี้อยู่แล้ว
# แต่ของเดิมมีที่ใช้ป้ายนี้อยู่ที่เดียวคือตอนระบบเลือก element ให้เองใน forced recovery —
# คลิกที่ "โมเดลเลือกเอง" ไม่เคยถูกกันเลย ทั้งที่เป็นทางเดินออกนอกงานที่เห็นซ้ำๆ (live run
# 2026-08-27: กด "William Little [Profile/Account Menu]" กลางงานลบ user)
_PROFILE_MENU_LABEL_MARKER = "[Profile/Account Menu]"
# W_account_keyword_scope (MR4 จาก audit): "setting"/"ตั้งค่า" เดี่ยวๆ กว้างเกินไป — goal ที่
# พูดถึง settings *ของระบบ* ("ไปที่หน้า Configuration แล้วตั้งค่า...") จะปิด guard นี้ทิ้งฟรีๆ
# ทั้งที่ไม่ได้เกี่ยวกับบัญชีของผู้ใช้ที่ล็อกอินอยู่เลย — ใช้เฉพาะรูปที่ระบุว่าเป็นของตัวผู้ใช้เอง
_ACCOUNT_GOAL_KEYWORDS = (
    "profile", "account", "logout", "log out", "sign out", "password",
    "my settings", "account settings", "personal settings",
    "โปรไฟล์", "บัญชี", "ออกจากระบบ", "รหัสผ่าน", "ตั้งค่าบัญชี", "ตั้งค่าส่วนตัว",
)
_MAX_PROFILE_MENU_RETRIES = 2

_PROFILE_MENU_NUDGE = (
    "[Rejected] That element is the user's own profile/account menu (logout, change "
    "password, personal settings). This goal has nothing to do with the signed-in user's "
    "account, so opening it cannot move the task forward — and it leads to flows that log "
    "you out or change credentials. Stay on the page's own content and pick an element that "
    "belongs to the goal."
)


def _goal_is_about_the_signed_in_account(goal: str) -> bool:
    lower = (goal or "").lower()
    return any(kw in lower for kw in _ACCOUNT_GOAL_KEYWORDS)


# W_empty_table_needs_right_filter (บั๊กจริง live stability check 2026-08-31): guard ฝั่ง
# "ยืนยันว่าจบงาน" รับ "ตารางที่แสดงอยู่เหลือ 0 แถว" เป็นหลักฐานความสำเร็จ โดย **ไม่เคยตรวจว่า
# ตัวกรองบนหน้ายังตั้งเป็นค่าที่ goal สั่งอยู่จริงไหม** — ตารางว่างเพราะกรองผิดกับตารางว่างเพราะ
# ลบครบ หน้าตาเหมือนกันเป๊ะสำหรับมัน
# (รันจริง: agent คลิก dropdown ซ้ำซ้อนจนตัวกรองเพี้ยน กด Search ได้ตารางว่าง แล้วรายงานว่า
# "ลบครบแล้ว" ทั้งที่ ground truth ก่อนและหลังเท่ากันที่ 1 ESS = ไม่ได้ลบอะไรเลย)
#
# น่าสังเกตว่าฝั่ง *ทำลายข้อมูล* มี guard คู่นี้อยู่แล้ว (W_delete_all_intent guard B ตรวจว่าแถว
# ที่เห็นตรงเงื่อนไขก่อนยอมให้ลบ) แต่ฝั่ง *ยืนยันว่าจบงาน* ไม่เคยมีคู่ของมัน
#
# อ่านจาก label ที่ perception ทำไว้ให้แล้วล้วนๆ ("User Role: ESS" — W_dropdown_field_label +
# W_field_label_for_plain_inputs) ไม่เรียก LLM ไม่ยิง DOM เพิ่มสักครั้ง
# ค่าที่แปลว่า "ยังไม่ได้ตั้ง" ของ dropdown ตัวกรอง — ต้องถือว่า "ไม่ตรงเงื่อนไข" ไม่ใช่ "ไม่รู้"
_FILTER_UNSET_VALUE_TEXTS = ("-- select --", "--select--", "select...", "all", "any", "ทั้งหมด", "")


def _filter_value_from_label(label: str) -> Optional[str]:
    """ค่าที่ตัวกรองตัวนี้ถืออยู่ตอนนี้ ("User Role: ESS" -> "ESS") — None ถ้า label ไม่ได้อยู่ใน
    รูป "ชื่อ field: ค่า" เลย (ตัดสินไม่ได้)"""
    # W_label_marker_key: ตัด marker ก่อนเสมอ ไม่งั้นค่าที่ได้กลายเป็น "ESS [in open dialog]"
    # แล้ว _cell_matches_value()/_FILTER_UNSET_VALUE_TEXTS อ่านผิดทั้งคู่
    cleaned = _label_without_markers(label)
    match = _FIELD_LABEL_PREFIX_RE.match(cleaned)
    return cleaned[match.end():].strip() if match else None


def _page_filter_matches_goal(
    elements: Optional[list], pairs: list[tuple[str, str]],
) -> Optional[bool]:
    """ตัวกรองบนหน้าตอนนี้ตรงกับที่ goal สั่งไหม

    True  = ทุก field ที่ goal ระบุ แสดงค่าที่ถูกต้องอยู่บนหน้าจริง
    False = เจอ field นั้นแต่ค่าไม่ตรง (รวมกรณีค่ากลับไปเป็น "-- Select --" = ตัวกรองหลุด)
    None  = อ่านตัวกรองไม่ได้เลย -> **ผู้เรียกต้อง fail-open** คงพฤติกรรมเดิมทุกประการ
            (เว็บที่ perception อ่าน label ตัวกรองไม่ได้ต้องไม่พังเพราะ guard นี้)
    """
    if not pairs or not elements:
        return None
    decided = False
    for field, value in pairs:
        for el in elements:
            label = str(el.get("label") or "")
            page_field = _filter_field_from_label(label)
            if not page_field or not _field_names_match(field, page_field):
                continue
            current = _filter_value_from_label(label)
            if current is None:
                continue
            decided = True
            if current.strip().lower() in _FILTER_UNSET_VALUE_TEXTS:
                return False
            if not _cell_matches_value(current, value):
                return False
            break
    return True if decided else None


def _extra_filters_set_on_page(elements: Optional[list], pairs: list[tuple[str, str]]) -> list[str]:
    """ชื่อ+ค่าของตัวกรองตัวอื่นที่ "ถูกตั้งค่าไว้" ทั้งที่ goal ไม่ได้พูดถึง — ตารางว่างขณะที่มี
    ตัวกรองส่วนเกินตั้งอยู่ เชื่อไม่ได้เหมือนกัน (เคส Status=Enabled ที่ user เจอตั้งแต่ต้น:
    ESS ที่ถูก disable หายจากตาราง แล้ว "ลบให้หมด" จะจบทั้งที่ยังเหลือ)
    W_filter_scope_guard กันตอน *จะตั้ง* ตัวกรองอยู่แล้ว ตัวนี้กันตอน *จะสรุปผล*"""
    if not pairs or not elements:
        return []
    extras = []
    for el in elements:
        label = str(el.get("label") or "")
        page_field = _filter_field_from_label(label)
        if not page_field or any(_field_names_match(f, page_field) for f, _ in pairs):
            continue
        current = _filter_value_from_label(label)
        if current is not None and current.strip().lower() not in _FILTER_UNSET_VALUE_TEXTS:
            extras.append(label)
    return extras


_EMPTY_TABLE_WRONG_FILTER_NUDGE_TEMPLATE = (
    "[Rejected] The table is empty, but that is not evidence the job is done: {problem}. "
    "An empty table only proves the work is complete when the filter on screen is exactly "
    "the one the goal asked for ({condition}). Set the filter to that, press Search, and "
    "look at the rows that come back before claiming anything."
)


_MAX_FILTER_SCOPE_RETRIES = 2

# W_filter_already_satisfied: โควตาเท่ากับ guard พี่น้องด้วยเหตุผลเดียวกัน — บางเว็บต้องเปิด
# dropdown ซ้ำจริงๆ (ค่าที่โชว์อยู่เป็นค่า default ที่ยังไม่ถูก apply) การบล็อกตายจะทำให้เว็บ
# กลุ่มนั้นใช้งานไม่ได้เลย
_MAX_FILTER_SATISFIED_RETRIES = 2

_FILTER_SATISFIED_NUDGE_TEMPLATE = (
    "[Rejected] '{label}' already holds the value the goal asked for ({field}={value}), so "
    "clicking it again cannot change anything — you have re-opened this same filter "
    "{count} times already. The filter is set: press Search now (or, if you already "
    "searched, work on the rows in the result table)."
)

_FILTER_SCOPE_NUDGE_TEMPLATE = (
    "[Rejected] This action sets the field '{label}', but the goal only asks you to filter "
    "by {allowed} — it never mentions that field at all. Setting an extra filter narrows the "
    "table by a condition the user did not ask for, so rows they DO care about disappear and "
    "the job looks finished when it is not. Leave every other filter untouched: set only "
    "{allowed}, then press Search."
)


def _goal_condition_values(goal: str) -> list[str]:
    """W_delete_all_intent: ดึงค่าเงื่อนไขแบบ key=value ออกจาก goal ("userrole=ess" -> ["ess"])
    — ใช้ตัวดึงตัวเดียวกับที่ W_deterministic_count ใช้อยู่แล้ว (actions._KEY_VALUE_IN_QUERY_RE)
    ไม่เขียน regex ใหม่ซ้อน คืน [] ถ้า goal ไม่ได้ระบุเงื่อนไขแบบนี้ (= ไม่เปิด guard เลย
    ปล่อยผ่านตามปกติ ไม่เดาเงื่อนไขเองจากภาษาธรรมชาติ)"""
    # ตัด URL ทิ้งก่อนเสมอ — goal จริงมักมี URL ปนอยู่ด้วย ("ไปที่ https://x/?id=9 แล้วลบ user
    # ที่ userrole=ess ให้หมด") ซึ่ง query string ของมันเข้าเงื่อนไข key=value เป๊ะๆ ทั้งที่
    # ไม่ใช่เงื่อนไขของงานเลย ถ้าไม่ตัดจะได้เงื่อนไข AND ที่ไม่มีแถวไหนตรงได้เลย แล้ว guard
    # จะบล็อกการลบที่ถูกต้องทิ้งไปเปล่าๆ
    values: list[str] = []
    seen: set[str] = set()
    # W_column_aware_rows: ข้อจำกัดเดิมที่เขียนไว้ตรงนี้ ("แยกคอลัมน์ไม่ได้ จึงเล็งคอลัมน์
    # ไม่ได้") **หมดไปแล้ว** — _scan_visible_table_rows() คืนเซลล์แยกคอลัมน์พร้อมหัวตารางแล้ว
    # ฟังก์ชันนี้เหลือหน้าที่แค่ "ค่าที่ user ขอ" สำหรับข้อความรายงาน ส่วนการตัดสินว่าแถวไหน
    # ตรงเงื่อนไขย้ายไปที่ _row_matches_condition() ซึ่งใช้ทั้ง field และ value
    for _, value in _goal_condition_pairs(goal):
        if value.lower() not in seen:
            seen.add(value.lower())
            values.append(value)
    return values


# W_delete_all_intent: อ่าน "แถวข้อมูลที่มองเห็นอยู่จริง" ของตารางที่ใหญ่ที่สุดบนหน้า — generic
# ล้วนๆ (ไม่มี .oxd-* เลย) รองรับทั้ง <table><tr> และ ARIA grid (div[role=table|grid] +
# [role=row]) ตาม pattern เดียวกับ perception._EXTRACT_TABLE_JS ที่พิสูจน์แล้วว่าถูกต้อง —
# ตัดแถวหัวตารางทิ้งเสมอ (แถวที่มี th/[role=columnheader] อยู่ข้างใน) เพราะหัวตารางมีคำว่า
# "User Role" อยู่ด้วยจะทำให้นับ match เกินจริง
# W_column_aware_rows: คืน "เซลล์แยกคอลัมน์ + ชื่อหัวตาราง" ไม่ใช่ innerText ทั้งแถวก้อนเดียว
# เหมือนเดิม — เพราะการเทียบเงื่อนไขกับข้อความทั้งแถวทำให้ค่าอย่าง "ess" ไปตรงกับชื่อคน
# ("Jessica"), username ("ess.irhrg0") หรือสถานะ ("Assessed") ได้หมด ซึ่งเป็นเคสที่ comment
# ของ _KEY_VALUE_IN_QUERY_RE (W_column_aware_count) เตือนไว้ตรงตัวแล้วแต่ยังไม่เคยถูกแก้จริง
# ผลของการนับผิดคือของที่กู้คืนไม่ได้ทั้งสองทิศ: นับเกิน -> ปล่อยให้ลบจากตารางที่ยังไม่กรอง /
# นับเกินตอนตรวจงานที่เสร็จแล้ว -> task ที่จบแล้วจบไม่ได้
_VISIBLE_TABLE_ROWS_JS = r"""
() => {
  const clean = (s) => (s || '').replace(/\s+/g, ' ').trim();
  let best = null;
  for (const t of document.querySelectorAll('table, [role="table"], [role="grid"]')) {
    let rowEls = Array.from(t.querySelectorAll('tr'));
    if (rowEls.length === 0) rowEls = Array.from(t.querySelectorAll('[role="row"]'));
    // W_column_headers_fallback: ชั้นที่ 1 คือหัวตารางจริง (th/[role=columnheader]) — ถ้าไม่มี
    // ตารางจำนวนมากยังบอกชื่อคอลัมน์ไว้ที่ตัวเซลล์เอง โดยเฉพาะตาราง responsive ที่ต้องโชว์ชื่อ
    // คอลัมน์บนมือถือ ลองไล่ต่ออีก 2 ชั้นก่อนจะยอมแพ้
    const headerEl = rowEls.find((r) => r.querySelector('th, [role="columnheader"]'));
    let headers = headerEl
      ? Array.from(headerEl.querySelectorAll('th, [role="columnheader"]')).map((h) => clean(h.innerText))
      : [];
    let headerSource = headers.length ? 'header-row' : '';
    if (!headers.length) {
      // ชั้นที่ 2: เซลล์ชี้ไปหาหัวคอลัมน์ที่ประกาศไว้เอง (aria-labelledby / headers=)
      const firstBody = rowEls.find(
        (r) => !r.querySelector('th, [role="columnheader"]') &&
               r.querySelector('td, [role="cell"], [role="gridcell"]'),
      );
      const bodyCells = firstBody
        ? Array.from(firstBody.querySelectorAll('td, [role="cell"], [role="gridcell"]'))
        : [];
      const viaIds = bodyCells.map((c) => {
        const ref = c.getAttribute('headers') || c.getAttribute('aria-labelledby') || '';
        const target = ref ? document.getElementById(ref.split(/\s+/)[0]) : null;
        return target ? clean(target.innerText) : '';
      });
      if (viaIds.some(Boolean)) { headers = viaIds; headerSource = 'aria'; }
      if (!headers.length) {
        // ชั้นที่ 3: data-* ที่ชื่อคอลัมน์ (พบบ่อยในตาราง responsive ที่โชว์ label ผ่าน CSS)
        const viaData = bodyCells.map(
          (c) => clean(c.getAttribute('data-col') || c.getAttribute('data-field') ||
                       c.getAttribute('data-label') || c.getAttribute('data-title') || ''),
        );
        if (viaData.some(Boolean)) { headers = viaData; headerSource = 'data-attr'; }
      }
    }
    const rows = [];
    for (const r of rowEls) {
      if (r.querySelector('th, [role="columnheader"]')) continue;
      let cellEls = Array.from(r.querySelectorAll('td, [role="cell"], [role="gridcell"]'));
      // ตารางที่ไม่ได้ใช้ td/[role=cell] เลย (div ล้วน) — ถอยไปใช้ทั้งแถวเป็นเซลล์เดียว
      // ซึ่งให้พฤติกรรมเท่าเดิมกับก่อน W_column_aware_rows ไม่ใช่คืนแถวว่าง
      const cells = cellEls.length > 0
        ? cellEls.map((c) => clean(c.innerText))
        : [clean(r.innerText)];
      if (cells.some(Boolean)) rows.push(cells);
    }
    if (rows.length > 0 && (!best || rows.length > best.rows.length)) {
      best = { headers, rows, headerSource };
    }
  }
  return best;
}
"""


async def _scan_visible_table_rows(page: Page) -> Optional[tuple[list[str], list[list[str]]]]:
    """W_delete_all_intent / W_column_aware_rows: คืน (ชื่อหัวตาราง, แถวข้อมูลแยกเป็นเซลล์) ของ
    ตารางที่ใหญ่ที่สุดบนหน้า หรือ None ถ้าหน้านี้ไม่มีตาราง/อ่านไม่ได้ — ห้าม throw ออกไปพัง loop
    เด็ดขาด (หลักการเดียวกับ guard อื่นในไฟล์นี้) หัวตารางว่างได้ (ตารางที่ไม่มี header จริง)
    ซึ่งจะทำให้ผู้เรียกถอยไปเทียบทั้งแถวเองตามเดิม"""
    try:
        table = await page.evaluate(_VISIBLE_TABLE_ROWS_JS)
    except Exception:
        return None
    if not isinstance(table, dict) or not table.get("rows"):
        return None
    headers = [str(h) for h in (table.get("headers") or [])]
    rows = [[str(c) for c in row] for row in table["rows"] if isinstance(row, list)]
    return (headers, rows) if rows else None


def _table_columns_are_addressable(headers: list[str], rows: list[list[str]]) -> bool:
    """W_column_headers_fallback: เล็งคอลัมน์ได้จริงไหม — ต้องมีชื่อคอลัมน์ *และ* แถวถูกแยก
    เป็นเซลล์จริง (ตาราง div ล้วนถูกยัดทั้งแถวเป็นเซลล์เดียวโดย _VISIBLE_TABLE_ROWS_JS
    ซึ่งเทียบเท่ากับไม่มีคอลัมน์เลย)"""
    return bool(headers) and any(len(row) > 1 for row in rows)


def _column_index_for_field(headers: list[str], field: str) -> Optional[int]:
    """W_column_aware_rows: index ของคอลัมน์ที่ชื่อตรงกับ field ที่ user เขียนใน goal —
    ใช้ตัวเทียบชื่อตัวเดียวกับ W_filter_scope_guard (_field_names_match) ที่ทนการพิมพ์ผิด/
    คำไทยติดหน้าอยู่แล้ว ไม่สร้างกฎการเทียบชุดที่สอง

    None = หาไม่เจอ ซึ่งแปลว่า "ตัดสินไม่ได้" ไม่ใช่ "ไม่ตรง" — ผู้เรียกต้อง fail-open"""
    if not field:
        return None
    for index, header in enumerate(headers):
        if _field_names_match(field, _normalized_field_name(header)):
            return index
    return None


def _cell_matches_value(cell: str, value: str) -> bool:
    """W_column_aware_rows: เซลล์นี้มีค่าที่ user ขอไหม — ยอมรับ 2 แบบเท่านั้น: ตรงทั้งเซลล์
    หรือเป็น "คำเต็มคำหนึ่ง" ที่คั่นด้วยช่องว่างในเซลล์ (เช่นเซลล์ "Senior ESS" กับค่า "ess")

    จงใจใช้การตัดด้วยช่องว่าง ไม่ใช่ word boundary ของ regex — `` ถือว่า "." เป็นตัวคั่นด้วย
    ทำให้ username "ess.irhrg0" ยังตรงกับค่า "ess" อยู่ดี ซึ่งเป็นเคสตัวอย่างที่ comment ของ
    _KEY_VALUE_IN_QUERY_RE (W_column_aware_count) ยกไว้ตรงตัวว่าเคยนับผิดมาแล้วจริง
    ที่ตัดทิ้งไปพร้อมกัน: "Jessica" / "Assessed" (ค่าอยู่กลางคำ)"""
    cell_norm = (cell or "").strip().lower()
    value_norm = (value or "").strip().lower()
    if not value_norm:
        return False
    return cell_norm == value_norm or value_norm in cell_norm.split()


def _row_matches_condition(
    headers: list[str], cells: list[str], pairs: list[tuple[str, str]],
) -> bool:
    """W_column_aware_rows: ทุกคู่ field=value ต้องตรงในแถวเดียวกัน (AND) ตรงกับความหมายของ
    "userrole=ess status=enabled" ที่ user เขียนจริง — เล็งคอลัมน์ได้ก็เทียบเฉพาะเซลล์นั้น
    เล็งไม่ได้ (ตารางไม่มีหัว/ชื่อหัวไม่ตรงอะไรเลย) ถึงค่อยถอยไปเทียบทั้งแถวแบบเดิม"""
    row_text = " ".join(cells)
    for field, value in pairs:
        index = _column_index_for_field(headers, field)
        if index is not None and index < len(cells):
            if not _cell_matches_value(cells[index], value):
                return False
        elif not _cell_matches_value(row_text, value):
            return False
    return True


async def _count_rows_matching_condition(
    page: Page, pairs: list[tuple[str, str]], *, require_columns: bool = False,
) -> Optional[tuple[int, int]]:
    """W_delete_all_intent: คืน (จำนวนแถวที่ตรงเงื่อนไขทุกคู่, จำนวนแถวทั้งหมดที่เห็น) ของตาราง
    ที่แสดงอยู่ หรือ None ถ้าเช็คไม่ได้

    W_column_headers_fallback: `require_columns=True` = "ถ้าเล็งคอลัมน์ไม่ได้ ให้ตอบว่าตัดสิน
    ไม่ได้ (None) แทนที่จะถอยไปเทียบทั้งแถว" — ตารางที่ไม่มีหัวคอลัมน์เลยทำให้
    _row_matches_condition() fail-open กลับไปเทียบข้อความทั้งแถว ซึ่งคือบั๊ก W97 เดิมเป๊ะ
    (ค่า "ess" ไปตรงกับชื่อคน "Jessica") เพียงแต่เงียบกว่าเพราะไม่มีใครเห็น

    ผู้เรียกต้องเลือกเองตามทิศทางของการตัดสินใจ:
      - ทิศที่ปลอดภัย (บล็อกไว้ก่อน เช่น กันลบจากตารางที่ยังไม่กรอง) -> require_columns=False
        เทียบทั้งแถวได้ เพราะเดาเกินไปแล้วบล็อก อย่างมากก็เสีย step
      - ทิศที่อันตราย (ใช้เป็นหลักฐานว่า "จบงานแล้ว") -> require_columns=True เพราะการนับเกิน
        จริงในทิศนี้แปลว่าอ้างว่าเสร็จทั้งที่ยังเหลือ"""
    if not pairs:
        return None
    scanned = await _scan_visible_table_rows(page)
    if scanned is None:
        return None
    headers, rows = scanned
    if require_columns and not _table_columns_are_addressable(headers, rows):
        return None
    matching = sum(1 for cells in rows if _row_matches_condition(headers, cells, pairs))
    return matching, len(rows)


# W_delete_all_intent: click ที่ "ทำลายข้อมูล" — ครอบคลุมทั้ง action type ที่ permission layer
# จัดเป็นกลุ่มยืนยันอยู่แล้ว (delete/submit/purchase/pay) และ click ธรรมดาที่ label บอกเองว่า
# เป็นการลบ (โมเดลเลือก type "click" ให้ปุ่ม Delete ได้เสมอ — defense-in-depth เดียวกับที่
# permission/rules.py ใช้ label keyword เสริม action type)
_DESTRUCTIVE_LABEL_RE = re.compile(r"\b(delete|remove|destroy|trash)\b|ลบ", re.IGNORECASE)

_MAX_DESTRUCTIVE_BEFORE_FILTER_RETRIES = 2

_DESTRUCTIVE_BEFORE_FILTER_NUDGE_TEMPLATE = (
    "This destructive action is BLOCKED for now. The goal says to delete only the entries "
    "matching {condition}, but the table on screen right now shows {total} rows and only "
    "{matching} of them match that condition — so the list you are about to delete from is NOT "
    "filtered yet. Deleting from this list risks destroying data the goal never asked you to "
    "touch. Set the filter for {condition} first, then press the Search button and wait for the "
    "table to reload, and only act on the rows that come back."
)

_DELETE_ALL_NO_SEARCH_NUDGE = (
    "This finish_task(success=true) is rejected. You changed a filter field but never pressed "
    "the Search button afterwards, so the table you were looking at was never filtered by the "
    "condition the goal asks for — whatever you did, it cannot be the complete job. Press "
    "Search, look at the rows that come back, and finish the deletion on those rows."
)

_DELETE_ALL_UNVERIFIED_NUDGE_TEMPLATE = (
    "This finish_task(success=true) is rejected. The goal is to delete EVERY entry matching "
    "{condition}, but the page you are on right now shows no result table at all, so there is "
    "no evidence left that the job is complete — you may have navigated away from the list. Go "
    "back to the filtered list, look at how many rows still match {condition}, and only call "
    "finish_task(success=true) once that list is genuinely empty."
)

# W_undefined_quota: โควตาของ guard 2 ตัวข้างบน (_DELETE_ALL_NO_SEARCH_NUDGE และ
# _DELETE_ALL_UNVERIFIED_NUDGE_TEMPLATE) — ชื่อนี้ถูก "ใช้" มาตั้งแต่ W80 ที่จุดตรวจ
# finish_task แต่ไม่เคยถูก "ประกาศ" เลยสักที่ ทั้งสอง guard จึงโยน NameError ทันทีที่ควรจะ
# ทำงาน คือตอน agent อ้างว่าลบครบทั้งที่ยังไม่เคยกด Search ซึ่งเป็นเคสที่ W80 สร้างมาเพื่อจับ
# โดยเฉพาะ — แล้ว W_loop_crash (W77) รับ NameError ไว้เงียบๆ แปลงเป็น success=False ทั้ง task
# guard ชุด W80 จึงไม่เคยได้ทำงานจริงสักครั้งนับตั้งแต่เขียนมา
#
# เทสต์มองไม่เห็นเพราะ assert แค่ result["success"] is False ซึ่งเป็นจริงทั้งตอน guard ทำงาน
# ถูกและตอน task crash — เทสต์ของ guard พวกนี้ต้อง assert "ข้อความ nudge" เสมอ ไม่ใช่แค่ผลลัพธ์
# 2 = เท่ากับ _MAX_DESTRUCTIVE_BEFORE_FILTER_RETRIES ด้านบน (guard ตระกูลเดียวกัน ให้โมเดลแก้ตัว
# ได้ 2 ครั้งแล้วยอมรับคำตอบของมัน ไม่ขังไว้จนหมด step budget)
_MAX_DELETE_ALL_UNVERIFIED_RETRIES = 2

_ROW_ACTION_LABEL_RE = re.compile(r"\b(edit|view details|delete|download|pencil)\b", re.IGNORECASE)

# W64[7.1]: ใช้เช็คว่า click ที่เพิ่งสำเร็จคือการกด Search จริงหรือไม่ (ถ้าใช่ ล้าง
# filter_dirty_since_search ทันที) — ครอบคลุมทั้งไทย/อังกฤษเหมือน keyword set อื่นในไฟล์นี้
_SEARCH_LABEL_RE = re.compile(r"\bsearch\b|ค้นหา", re.IGNORECASE)

# W_rowaction_own_quota: guard นี้เคยถูกจำกัดด้วย _MAX_PREMATURE_TABLE_VERIFY_RETRIES ซึ่ง
# เป็นโควตาของ guard คนละตัว (W63[7.2] "Strict Table Assertion") — คนละบั๊ก คนละเงื่อนไข
# ปรับค่าของ guard หนึ่งแล้วไปเปลี่ยนพฤติกรรมของอีก guard หนึ่งโดยไม่ตั้งใจ ให้โควตาของตัวเอง
# ค่าเท่าเดิม (2) พฤติกรรมจึงไม่เปลี่ยนจากเดิมเลย แค่แยกปุ่มปรับออกจากกัน
_MAX_PREMATURE_ROW_ACTION_BEFORE_SEARCH_RETRIES = 2

# W_unknown_tool: ชื่อ tool ทั้งหมดที่มีอยู่จริง (ตรงกับที่ llm.py ประกาศให้ทุก provider) —
# ใช้ปฏิเสธชื่อที่โมเดลมโนขึ้นเองก่อนจะหลุดไปถึง actions.execute() ดูจุดใช้งานในลูปหลัก
_KNOWN_TOOL_NAMES = frozenset({"browser_action", "request_user_input", "finish_task"})

# W_step_trace (failure taxonomy): ก่อนหน้านี้ทั้งระบบไม่มี field ไหนบอกเลยว่า step หนึ่ง
# "ล้มเพราะอะไร" — มีแค่ success: true/false กับข้อความอิสระที่แต่ละ action เขียนเอง ทำให้
# ตอบคำถามพื้นฐานอย่าง "task ที่ล้ม 15 ครั้งล่าสุด ล้มเพราะหา element ไม่เจอ หรือเพราะโมเดล
# เลือก action ผิด" ไม่ได้เลยโดยไม่ไล่อ่าน log ทีละบรรทัด
#
# จัดหมวดจากข้อความของ ActionResult (deterministic ล้วนๆ ไม่เรียก LLM) — เรียงจากเฉพาะเจาะจง
# ไปกว้าง เพราะข้อความหนึ่งอาจตรงหลายรูปแบบ (เช่น "[Skipped]" ที่เกิดจาก guard ต่างกัน)
_FAILURE_TAXONOMY_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("human_denied", (REJECTED_BY_USER_MESSAGE,)),
    ("permission_blocked", ("blocked by the permission layer", "ถูกบล็อก")),
    ("wrong_action_type", ("not a native <select>", "is a native <select>", "unknown action")),
    ("skipped_no_op", ("[Skipped]",)),
    ("element_not_found", ("element not found", "no option matching", "not clickable")),
    ("timeout", ("timeout", "Timeout")),
    ("missing_parameter", ("missing parameter",)),
    ("read_failed", ("nothing matching or close to", "no element matching")),
)


def _classify_step_failure(result_text: str, success: bool) -> str:
    """คืนชื่อหมวดความล้มเหลวของ step นี้ — "ok" ถ้าสำเร็จ, "other" ถ้าล้มแต่ไม่ตรงหมวดไหนเลย

    ห้าม throw (ใช้ตอนประกอบ trace เท่านั้น ไม่ควรมีทางทำให้ task พัง)"""
    if success:
        return "ok"
    lowered = (result_text or "").lower()
    for name, needles in _FAILURE_TAXONOMY_PATTERNS:
        if any(needle.lower() in lowered for needle in needles):
            return name
    return "other"

# W_step_budget: ข้อความ default ตอนลูปจบเพราะหมดรอบ — ดึงเป็นค่าคงที่เพื่อให้ตอนจบ task
# เช็คได้ว่า "ยังไม่มีใครตั้งข้อความจริงให้เลย" แล้วเติมเหตุผลล่าสุดของโมเดลต่อท้ายได้
_MAX_STEPS_EXHAUSTED_MESSAGE = "ครบ max_steps โดยยังไม่จบ task"

_PREMATURE_ROW_ACTION_BEFORE_SEARCH_NUDGE_TEMPLATE = (
    "This action is rejected — you filled/selected a value in the field '{prev_label}' on the "
    "previous step but have not yet pressed the Search button (or Enter) to apply that "
    "filter. The table you see now is still the OLD result from before the new filter, not "
    "the genuinely filtered one — do NOT click the '{label}' row action on this row right "
    "now. Press Search first, then check the new round of indexed elements to confirm the "
    "table really is filtered to the condition you wanted, before choosing an action on a row."
)


# W63[7.2]: ต่างจาก guard อื่นในไฟล์นี้ (เปิดใช้ตาม intent keyword ของ goal) guard นี้เปิดใช้
# ตามการมี tool_input["verify_text"] มาจาก LLM เอง (ดู llm.py::_FINISH_TASK_PARAMS) แทน — LLM
# เป็นคนตัดสินใจว่า goal นี้ "ควร" ยืนยันด้วยข้อความในตารางไหม ไม่ต้องเดา intent จาก keyword
# เอง (ยืดหยุ่นกว่า เพราะครอบคลุมทั้งงานสร้าง/แก้ไข/บันทึกที่คาดว่าจะโผล่ในตาราง ไม่ใช่แค่
# "สร้าง" เพียงอย่างเดียว)
_MAX_PREMATURE_TABLE_VERIFY_RETRIES = 2

_PREMATURE_TABLE_VERIFY_NUDGE_TEMPLATE = (
    "This finish_task(success=true) is rejected — checking the real DOM of the results table "
    "on the current page finds the text \"{text}\" (given in verify_text) in no row at all. "
    "Never treat the entry as created/saved while the table genuinely does not show it. Check "
    "whether the submit actually went through (is a validation error still showing?), whether "
    "you have navigated back to the correct list page, or whether you need to refresh/search "
    "again before it appears, before calling finish_task(success=true) again."
)

# W64[7.2] ("Add-Action Idempotency Lock" — ticket Issue 7.2, บั๊กจริง: agent บันทึกพนักงาน
# ใหม่สำเร็จจริง (มี toast ยืนยัน) แต่ค้นหาเพื่อ verify แล้วไม่เจอเพราะยังไม่รอ AJAX table
# reload ให้เสร็จ — "ตื่นตระหนก" กด Reset แล้วกรอกฟอร์ม Add Employee ใหม่ทั้งหมดซ้ำอีกรอบ จน
# เกิด error ข้อมูลซ้ำ): ต่อท้าย nudge เดิมเฉพาะตอนที่ task นี้เคยมี action ที่ toast ยืนยัน
# สำเร็จแล้วจริง (ดู any_toast_confirmed_this_task) — ห้ามตีความ "หาไม่เจอในตาราง" เป็น
# "ยังไม่ได้บันทึก" แล้วย้อนกลับไปกรอกฟอร์มใหม่เด็ดขาด เพราะมีหลักฐาน toast ยืนยันสำเร็จจริง
# อยู่แล้ว การหาไม่เจอครั้งนี้น่าจะเป็นปัญหาการค้นหา/filter/pagination มากกว่า
_TOAST_CONFIRMED_NO_RECREATE_SUFFIX = (
    " *** IMPORTANT: earlier in this same task an action genuinely detected a toast/save-"
    "success confirmation (see the earlier step in the history) — the data really was saved. "
    "Not finding it in the table right now is NOT evidence that it wasn't. NEVER press "
    "Reset/clear the form and fill in the creation form again (that produces duplicate "
    "entries and duplicate-data errors) — the only thing to do is wait a moment and re-query "
    "the table (e.g. press Search again). ***"
)


# W22 (ORANGEHRM SPECIFIC ตามสเปคที่ user ให้มา): ข้อความ "(N) Records Found"/"No Records
# Found" ปรากฏแทบทุกหน้าที่มี list/filter ของ OrangeHRM (Admin > User Management, PIM >
# Employee List, Recruitment > Candidates ฯลฯ) — .orangehrm-horizontal-padding span คือ
# selector ที่ user ยืนยันมาจาก DOM จริง ใช้ :has-text() (Playwright selector engine เอง ไม่ใช่
# CSS มาตรฐาน) เป็นชั้นสำรองกว้างๆ เผื่อ layout เปลี่ยนไปในเวอร์ชันอื่นของ OrangeHRM ที่ยังคง
# ข้อความนี้ไว้แต่ขยับ element/class ไป — ตั้งใจไม่ทำให้ generic ข้ามเว็บเหมือน
# _VALIDATION_ERROR_SELECTOR เพราะ "Records Found" ไม่ใช่ข้อความมาตรฐานที่เว็บอื่นใช้ร่วมกันเลย
# W_record_count_generic (P3.9): เดิม selector + regex ผูกกับข้อความ "(N) Records Found"
# ของ OrangeHRM อย่างเดียว — guard กัน false-completion ตัวเรือธง (ยืนยันว่าลบ/แก้ครบจริง)
# จึงทำงานได้กับเว็บเดียวในโลก ที่เหลือ _scan_remaining_target_records() คืน None เงียบๆ
# = ไม่มี guard เลย
#
# เพิ่มชั้น generic ที่ครอบรูปแบบที่เว็บทั่วไปใช้จริง โดยยังคง fail-safe เดิมทุกประการ:
# อ่านไม่ได้/ไม่เจอ = None = ไม่บล็อกอะไร (ดีกว่าบล็อก finish_task ที่อาจถูกต้องอยู่แล้ว)
# W_record_count_picks_wrapper: เดิมเป็น selector เดียวคั่นด้วย comma แล้วหยิบ `.first` โดยเชื่อ
# ว่า "ของเจาะจงที่วางไว้ก่อนจะชนะ" — **ผิด** CSS ที่คั่นด้วย comma คืน element ตาม *ลำดับใน DOM*
# ไม่ใช่ลำดับที่เขียนใน selector และ `:has-text()` ก็ match บรรพบุรุษทุกชั้นที่มีข้อความนั้นอยู่ข้างใน
# `.first` จึงได้ `<div>` ก้อนใหญ่ที่ครอบทั้งหน้าเสมอ (เห็นจริงใน live run 2026-08-28: ข้อความ
# ผลลัพธ์ของ task ยัด innerText ทั้งหน้าเข้าไปทั้งก้อน)
# ไม่ใช่แค่เรื่องความสวย: ข้อความก้อนนั้นถูกส่งต่อให้ _RECORD_COUNT_PATTERNS หาตัวเลข ซึ่งอาจไป
# เจอเลขอื่นบนหน้าที่ไม่เกี่ยวเลย = guard กัน hallucination ตัวเรือธงอ่านค่าผิด
#
# แยกเป็นลิสต์แล้วไล่ทีละตัว (pattern เดียวกับ actions.py::_find_visible_modal_confirm_button)
# ลำดับใน list ถึงจะมีความหมายจริง และในแต่ละ selector เลือก element ที่ข้อความสั้นที่สุด =
# ตัวที่เป็นข้อความสรุปเอง ไม่ใช่ container ที่ห่อมันอยู่
_RECORD_COUNT_SELECTORS = (
    # OrangeHRM (เจาะจงที่สุด เก็บไว้ก่อนเสมอ)
    '.orangehrm-horizontal-padding span:has-text("Records Found")',
    'span:has-text("No Records Found")',
    # generic — ข้อความสรุปผลลัพธ์ที่ framework/เว็บทั่วไปใช้ วางไว้ทีหลังเพื่อให้ของเจาะจง
    # ชนะก่อนถ้ามีทั้งคู่บนหน้าเดียวกัน
    ':is(span, div, p, h2, h3):has-text("Records Found")',
    ':is(span, div, p, h2, h3):has-text("results found")',
    ':is(span, div, p, h2, h3):has-text("No results")',
    ':is(span, div, p, h2, h3):has-text("รายการ")',
)
# element ที่ match ได้ต่อ selector หนึ่งตัว — ดูแค่ไม่กี่ตัวแรกพอ (ห่วงโซ่บรรพบุรุษที่ห่อข้อความ
# เดียวกันยาวได้ แต่ตัวที่สั้นที่สุดอยู่ในกลุ่มแรกๆ เสมอ) กัน DOM query บานบนหน้าที่ใหญ่มาก
_RECORD_COUNT_MAX_CANDIDATES = 8

# รูปแบบตัวเลขที่ยอมรับ เรียงจากเจาะจงไปกว้าง — ตัวแรกที่ match ชนะ
# ตั้งใจไม่รับ "ตัวเลขลอยๆ" ที่ไม่มีคำบอกบริบทกำกับเลย เพราะหน้าเว็บมีตัวเลขเต็มไปหมด
# (ราคา/วันที่/เลขหน้า) การเดาผิดที่นี่แปลว่า guard ไปบล็อก finish_task ที่ถูกต้อง
_RECORD_COUNT_PATTERNS = (
    # OrangeHRM: "(41) Records Found"
    re.compile(r"\((\d+)\)\s*Records?\s*Found", re.IGNORECASE),
    # "42 results found" / "42 results" / "42 items" — เจาะจงกว่า "of N" ด้านล่างจึงมาก่อน
    re.compile(r"\b([\d,]+)\s+(?:results?|items?|records?|entries)\b", re.IGNORECASE),
    # ไทย: "42 รายการ" / "ทั้งหมด 42 รายการ"
    re.compile(r"([\d,]+)\s*รายการ"),
    # "Showing 1-10 of 42" — เอาตัวหลัง "of" ซึ่งคือยอดรวมจริง ไว้ท้ายสุดเพราะกว้างที่สุด:
    # ข้อความแบบ "page 2 of 5" ก็ match ได้ ถ้าเอาขึ้นก่อนจะแย่งเคสที่มีทั้งเลขหน้าและยอดรวม
    # อยู่ในประโยคเดียวกัน (selector ด้านบนกรองแล้วว่าต้องเป็นข้อความสรุปผลลัพธ์ถึงจะมาถึงตรงนี้
    # แต่ลำดับยังต้องถูกอยู่ดี)
    re.compile(r"\bof\s+([\d,]+)\b", re.IGNORECASE),
)

# ข้อความที่แปลว่า "ไม่มีผลลัพธ์เลย" (= 0) — ต้องเช็คก่อน pattern ตัวเลขเสมอ เพราะบางอันมี
# เลข 0 อยู่ในประโยคอยู่แล้ว บางอันไม่มีเลขเลย
_RECORD_COUNT_ZERO_TEXTS = (
    "no records found", "no results", "no matching records", "no data",
    "ไม่พบข้อมูล", "ไม่พบรายการ", "ไม่มีข้อมูล",
)

# W68b (บั๊กจริงที่ user รายงานซ้ำหลัง W68: goal "เปลี่ยน Role ของทุกคนที่ไม่ใช่ Admin เป็น
# Admin" — agent ยัง claim "ผลการค้นหาแสดง 0 รายการ" ทั้งที่ตารางจริงโชว์ "(16) Records Found"
# แม้ W68 จะเปิด _is_edit_all_intent_goal() ให้ตรงแล้วก็ตาม): สาเหตุที่สอง ต่างจาก W68 —
# _scan_remaining_target_records() เดิมอ่าน DOM แค่ครั้งเดียวทันทีตอน finish_task ถูกเรียก
# ถ้าจังหวะนั้นตรงกับช่วง AJAX ของปุ่ม Search ยังไม่ update DOM เสร็จ (wait_stable() ก่อนหน้า
# ใช้ networkidle timeout 4s แต่ไม่รับประกัน 100% ว่า DOM re-render เสร็จภายในนั้นเสมอ) จะอ่าน
# ได้ "(0)"/"No Records Found" ของสถานะเก่า/ชั่วคราว — อันตรายเฉพาะเคส "0" เท่านั้น (แปลว่า
# ปล่อยผ่านให้ finish_task(success=true) ทันทีไม่มี retry เลย) ต่างจากเคส ">0" ที่อ่านผิดแค่
# เสีย nudge retry เปล่าๆ ไม่อันตราย (ดู caller) — เพิ่ม double-check เฉพาะตอนอ่านได้ "0"
# เท่านั้น: รอสั้นๆ แล้วอ่านซ้ำอีกรอบ ถ้ารอบสองเจอ >0 (มาสาย/ยืนยันว่าเพิ่ง update จริง) ให้เชื่อ
# รอบสองแทน (สดกว่า น่าเชื่อถือกว่า) — ไม่กระทบ latency ปกติเลยเพราะ retry นี้เกิดเฉพาะตอนได้
# ผล "0" เท่านั้น (เคสที่กำลังจะยอมรับ finish_task อยู่แล้ว เสีย delay สั้นๆ ครั้งเดียวคุ้มกว่า
# false-completion)
_ZERO_RECORD_RECHECK_DELAY_SECONDS = 0.8


async def _scan_remaining_target_records_once(page: Page) -> Optional[tuple[int, str]]:
    """W22: อ่านข้อความ "(N) Records Found"/"No Records Found" จาก DOM จริง ณ ตอนนี้ — คืน
    (จำนวนที่เหลือจริง, ข้อความดิบที่เจอ) ถ้าเจอ element นี้จริง หรือ None ถ้าหน้าปัจจุบันไม่มี
    element แบบนี้เลย (ไม่ใช่หน้าตารางแบบ OrangeHRM/เว็บนี้ไม่รองรับ UI แบบนี้ — ปล่อยผ่านเสมอ
    ไม่ block เหมือนหลักการเดียวกับ _scan_validation_errors ด้านบน: เช็คไม่ได้ ดีกว่าบล็อก
    finish_task ที่อาจถูกต้องอยู่แล้ว) "No Records Found" ตีความเป็น 0 เสมอ ไม่ throw ออกไปพัง
    guard เด็ดขาด (เหมือน _scan_validation_errors — จับ Exception กว้างๆ คืน None แทน)"""
    text = ""
    try:
        for selector in _RECORD_COUNT_SELECTORS:
            locator = page.locator(selector)
            total = await locator.count()
            if total == 0:
                continue
            candidates = []
            for i in range(min(total, _RECORD_COUNT_MAX_CANDIDATES)):
                try:
                    candidate = (await locator.nth(i).inner_text(
                        timeout=_DOM_CHECK_TIMEOUT_MS,
                    )).strip()
                except Exception:
                    continue
                if candidate:
                    candidates.append(candidate)
            if candidates:
                # สั้นที่สุด = ข้อความสรุปเอง ไม่ใช่ container ที่ห่อมันอยู่
                text = min(candidates, key=len)
                break
    except Exception:
        return None
    if not text:
        return None
    lowered = text.lower()
    if any(zero_text in lowered for zero_text in _RECORD_COUNT_ZERO_TEXTS):
        return 0, text
    # W_record_count_generic: ไล่ pattern จากเจาะจงไปกว้าง ตัวแรกที่ match ชนะ — ตัวคั่นหลักพัน
    # ต้องถอดก่อนแปลงเป็น int ("1,234 results")
    for pattern in _RECORD_COUNT_PATTERNS:
        match = pattern.search(text)
        if match:
            try:
                return int(match.group(1).replace(",", "")), text
            except ValueError:
                continue
    return None


async def _scan_remaining_target_records(page: Page) -> Optional[tuple[int, str]]:
    """W68b: เหมือน _scan_remaining_target_records_once() ทุกประการ แต่ double-check เฉพาะ
    ตอนอ่านได้ผล "0 รายการเหลือ" เท่านั้น (ดู docstring ของ _ZERO_RECORD_RECHECK_DELAY_SECONDS
    ด้านบนสำหรับเหตุผลเต็ม) — ผล None/>0 คืนค่าทันทีไม่หน่วงเพิ่ม"""
    result = await _scan_remaining_target_records_once(page)
    if result is not None and result[0] == 0:
        await asyncio.sleep(_ZERO_RECORD_RECHECK_DELAY_SECONDS)
        recheck = await _scan_remaining_target_records_once(page)
        if recheck is not None and recheck[0] > 0:
            return recheck
    return result


# W63[7.2] ("Strict Table Assertion & Truth Reporting" — ticket Issue 7.2): เรียงจากเจาะจง
# ที่สุด (OrangeHRM .oxd-table-body) ไปกว้างสุด (<table><tbody>/ARIA rowgroup/class ที่มีคำว่า
# table+body มาตรฐานที่ CSS framework ทั่วไปใช้ร่วมกัน — Material/Bootstrap/Ant Design ฯลฯ)
# ตั้งใจไม่ผูกกับ OrangeHRM เพียงเว็บเดียว ต่างจาก _RECORD_COUNT_SELECTORS ด้านบนที่ข้อความ
# "Records Found" ไม่ใช่ pattern ที่เว็บอื่นใช้ร่วมกันเลย แต่ <tbody>/[role=rowgroup] เป็น
# มาตรฐาน HTML/ARIA ตรงๆ
_TABLE_BODY_SELECTOR = (
    '.oxd-table-body, table tbody, tbody, [role="rowgroup"], '
    '[class*="table-body" i], [class*="tablebody" i]'
)


async def _scan_created_item_in_table(page: Page, verify_text: str) -> bool:
    """W63[7.2]: True ถ้าเจอ verify_text (substring, case-insensitive) อยู่จริงใน table body
    ของหน้าปัจจุบัน หรือถ้าหน้านี้ไม่มี table body ให้เช็คเลย (ปล่อยผ่านเสมอ — เช็คไม่ได้ ดีกว่า
    บล็อก finish_task ที่อาจถูกต้องอยู่แล้ว หลักการเดียวกับ guard อื่นในไฟล์นี้) — False เฉพาะ
    ตอนมี table body จริงแต่เนื้อหาไม่มี verify_text อยู่เลย (รวมถึงตารางว่างเปล่า/"No Records
    Found" — ถือเป็นหลักฐานว่ายังไม่พบรายการนี้จริงเหมือนกัน ไม่ใช่แค่ "เช็คไม่ได้")"""
    try:
        locator = page.locator(_TABLE_BODY_SELECTOR).first
        if await locator.count() == 0:
            return True
        text = (await locator.inner_text(timeout=_DOM_CHECK_TIMEOUT_MS)).strip()
    except Exception:
        return True
    return verify_text.lower() in text.lower()


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
# W_bare_invalid_is_a_field_hint (release gate จับได้ 2026-09-07, งาน "search_no_results"
# ตกซ้ำได้ 100% ทั้งสองรอบ): ช่อง Employee Name ของ OrangeHRM เป็น autocomplete พอพิมพ์ชื่อที่
# ไม่มีอยู่จริง มันขึ้นคำว่า "Invalid" ใต้ช่อง ตัวสแกน validation error เห็นแล้วยุติ task ทั้งงาน
# เพื่อขอค่าใหม่จาก user — ทั้งที่ goal ของงานนั้นคือ "ค้นหาชื่อที่ไม่มีอยู่แล้วยืนยันว่าไม่พบ"
# คำว่า "Invalid" จึงเป็นผลลัพธ์ที่ถูกต้อง ไม่ใช่ความล้มเหลว (จบที่ 1 step ทุกครั้ง)
#
# เป็น false positive ชนิดเดียวกับ "* Required" เป๊ะ: คำเดียวโดดๆ ที่เป็นป้ายบอกสถานะของช่อง
# ไม่ได้บอกว่าค่าที่กรอกผิดยังไง ต่างจากข้อความจริงอย่าง "Invalid email format" หรือ
# "Should have at least 7 characters" ซึ่งบอกรายละเอียดและยังต้องหยุดเหมือนเดิม
_BARE_FIELD_HINT_MESSAGE_RE = re.compile(
    r'^[\*\s]*(?:required|invalid)[\.\!]?$', re.IGNORECASE,
)


def _is_bare_required_message(text: str) -> bool:
    """True ถ้าข้อความเป็นแค่ป้ายบอกสถานะของช่องคำเดียว ("* Required"/"Required"/"Invalid")
    ไม่มีรายละเอียดว่าค่าที่กรอกผิดยังไง — false positive ที่โผล่ให้ช่องพี่น้องที่ยังไม่ได้กรอก
    หรือให้ autocomplete ที่หาคำที่พิมพ์ไม่เจอ ไม่ใช่ปัญหาของค่าที่เพิ่ง fill จริงๆ
    ต้องกรองทิ้งก่อน hard-stop (ดู W_bare_invalid_is_a_field_hint เหนือ regex)"""
    return bool(_BARE_FIELD_HINT_MESSAGE_RE.match((text or "").strip()))


# W_required_error_survives_the_fix (release gate จับได้ 2026-09-07, งาน rag_permission /
# rag_integration / long_flow ตกด้วยอาการเดียวกันทั้งสามงาน): agent กด Continue บนฟอร์ม checkout
# ของ SauceDemo ทั้งที่ยังไม่ได้กรอก เว็บขึ้น "Error: First Name is required" — agent แก้ถูกต้อง
# ด้วยการกรอกช่องนั้นในเทิร์นถัดมา แต่ตัวสแกนที่รันทันทีหลัง fill ยังเห็นแบนเนอร์เดิมค้างอยู่
# (SauceDemo ไม่ล้างจนกว่าจะกดส่งใหม่) แล้วเอา error ที่ล้าสมัยไปแล้วมาฆ่างานทิ้ง
#
# ข้อความแบบนี้มีรายละเอียดจริงจึงไม่เข้าตัวกรอง bare-hint — ต้องใช้เงื่อนไขที่ตรงกว่า:
# error ที่บอกว่า "ช่อง X ต้องกรอก" ย่อมล้าสมัยทันทีที่เพิ่งกรอกช่อง X สำเร็จ
# แคบโดยเจตนา: ต้องเป็นข้อความชนิด required *และ* ต้องอ้างถึงชื่อช่องที่เพิ่งกรอกจริงเท่านั้น
# ข้อความที่บอกว่าค่าที่กรอก "ผิดรูปแบบ" (invalid format / at least N characters) ไม่เข้าเงื่อนไข
# นี้และยังหยุด task เหมือนเดิม เพราะการกรอกใหม่ไม่ได้ทำให้มันหายไปเอง
_REQUIRED_ERROR_WORDS = ("required", "ต้องกรอก", "จำเป็นต้องระบุ", "ห้ามเว้นว่าง")


def _is_required_field_error(text: str) -> bool:
    """True ถ้าข้อความบอกว่า "ช่องนี้ต้องกรอก" — เป็น error ที่ agent แก้เองได้เสมอ

    W_required_error_is_not_a_dead_end (release gate จับได้ 2026-09-07, rag_integration และ
    long_flow ตกด้วยข้อความเดียวกันเป๊ะ 'Error: Postal Code is required'): agent กรอก
    First/Last Name เองแล้วพ่วงคลิก Continue ทั้งที่ยังไม่ได้กรอก Postal Code — error นี้เป็น
    ความจริง ไม่ใช่ของค้าง แต่ระบบยุติงานทั้งงานเพื่อ "ขอค่าใหม่จาก user" ทั้งที่ค่าที่ขาดคือ
    สิ่งที่ agent เติมเองได้ (มันเพิ่งเติมชื่อเองไปสองช่องในเทิร์นก่อนหน้า)
    hard-stop ตัวนี้มีไว้สำหรับ error ที่ "แก้ได้ด้วยค่าใหม่จาก user เท่านั้น" — ข้อความชนิด
    required ไม่เข้าข่ายนั้นเลย มันบอกชัดว่าต้องทำอะไรต่อและ agent ทำได้เอง ปล่อยให้ loop เดินต่อ
    (แบนเนอร์ยังอยู่บนหน้า โมเดลเห็นใน snapshot ถัดไปอยู่แล้ว)
    ส่วนข้อความที่บอกว่า "ค่าที่กรอกผิด" (invalid format / at least N characters / already
    exists) ยังหยุดเหมือนเดิม เพราะกรอกใหม่เองมั่วๆ ไม่ได้ ต้องรู้ค่าที่ถูกจริงๆ"""
    return any(w in (text or "").lower() for w in _REQUIRED_ERROR_WORDS)


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


# W65[2] ("Error Passthrough" — fatal-class short-circuit): hard-stop guard หลัง fill/click
# (ดู _should_check_validation_error_after_action ด้านบน, เรียกใช้จริงหลัง action สำเร็จ) มี
# พฤติกรรมถูกต้องอยู่แล้ว — break ทันทีพร้อมข้อความ error จริง ไม่ผ่าน LLM ตีความ ยกเว้น bare
# "* Required" (กรองด้วย _is_bare_required_message แล้ว) จุดที่ยังขาดคือ guard ก่อน
# finish_task(success=true) (เรียกใช้จริงในลูปหลักของ run_task ด้านล่าง) ซึ่งยังให้ LLM วน
# nudge/retry ก่อนเสมอ (สูงสุด _MAX_PREMATURE_VALIDATION_ERROR_RETRIES ครั้ง) แม้ error จะเป็น
# ประเภทที่ retry ไปก็ไม่มีทางหาย (เช่น login ผิด, ข้อมูลซ้ำ, ไม่มีสิทธิ์) — keyword list นี้ใช้
# แยก error 2 กลุ่ม: "fatal" (ต้องข้อมูลใหม่จาก user เท่านั้นถึงจะแก้ได้ ไม่มีทาง retry แล้วหาย
# เอง) vs error อื่นๆ ทั้งหมด (อาจเป็น timing/DOM ไม่นิ่ง ให้ LLM ลองแก้เองก่อนตามเดิม)
_FATAL_VALIDATION_ERROR_KEYWORDS = (
    "invalid credentials", "incorrect password", "invalid username or password",
    "already exists", "unauthorized", "not authorized", "permission denied",
    "ไม่ถูกต้อง", "ผิดพลาด", "มีอยู่แล้ว", "ไม่มีสิทธิ์",
)


def _is_fatal_validation_error(text: str) -> bool:
    """W65[2]: True ถ้าข้อความ error ตรงกับ keyword ที่บ่งบอกว่าเป็นปัญหาที่ agent แก้เองไม่ได้
    ด้วยการลองใหม่ (ต้องข้อมูล/สิทธิ์ใหม่จาก user เท่านั้น) — ใช้ก่อนยอมรับ
    finish_task(success=true) เพื่อข้าม nudge-retry loop ไปบังคับความจริงลง final result ทันที
    แทนที่จะเสีย round-trip LLM หลายครั้งไปกับ error ที่รู้อยู่แล้วว่าไม่มีทางหายเอง"""
    lower = (text or "").lower()
    return any(kw in lower for kw in _FATAL_VALIDATION_ERROR_KEYWORDS)


# W5: loop-detection guard — บางโมเดล (เจอกับ Llama บน Groq) ถึงจะถูกเตือนแล้วก็ยัง
# วนเรียก browser_action เดิมเป๊ะๆ ซ้ำๆ (dict เดียวกันทุก field) ไม่ว่าจะสำเร็จหรือ fail
# ก็ตาม แปลว่าไม่มีความคืบหน้าจริง — กันไว้ไม่ให้เสีย step/token ไปเรื่อยๆ จนหมด max_steps
# โดยไม่ได้อะไรขึ้นมา ถ้าเจอ action เดิมติดกันครบจำนวนนี้ ให้หยุด task ทันที
_MAX_CONSECUTIVE_IDENTICAL_ACTIONS = 3

# W_same_label_loop (บั๊กจริงที่ user รายงาน live บน OrangeHRM, goal "เปิดเว็ป แล้วไปที่เมนู
# แอดมิน แล้วลบ user role=ess ออกให้หมด"): guard คาบ 1 ด้านบนเทียบ cmd ทั้ง dict รวม "index"
# ด้วย — agent วนติ๊ก checkbox ของ "แถว" ไปเรื่อยๆ (click(35) -> click(36) -> click(34) ->
# check(37) ...) ซึ่ง label เหมือนกันหมดว่า "Select row" แต่ index ต่างกันทุกครั้ง เลยไม่เข้า
# เงื่อนไข "เดิมเป๊ะๆ" สักรอบ และ cycle detector (คาบ 2-4) ก็ไม่เห็นเป็นคาบเพราะ index ไม่ซ้ำ
# เป็นแบบแผน — ผลคือวนได้ไม่จำกัดจนกว่าจะหมด max_steps หรือ user กด Stop เอง (รอบที่รายงาน
# user กด Stop)
#
# element ที่ label เหมือนกันเป๊ะคือ "ของชนิดเดียวกันคนละแถว" เสมอในทางปฏิบัติ — สั่ง action
# ชนิดเดิมใส่ label เดิมติดกันเกินจำนวนนี้โดยไม่ทำอย่างอื่นคั่นเลย แปลว่าไม่ได้เดินหน้าเข้าหา
# goal จริง threshold ตั้งหลวมกว่าคาบ 1 (3) เพราะการติ๊กหลายแถวก่อนกดลบทีเดียวเป็นรูปแบบที่
# ถูกต้องอยู่บ้าง — และเมื่อ trip ก็ใช้ _force_loop_recovery() เหมือน guard อื่น (บังคับ action
# แล้วไปต่อ ไม่ใช่ฆ่า task ทิ้งทันที) งานที่ยาวจริงจึงยังมีทางไปต่อได้
_MAX_CONSECUTIVE_SAME_LABEL_ACTIONS = 4

# W_already_logged_in_but_told_to_log_in: บอกโมเดลว่าระบบล็อกอินให้แล้วเฉพาะช่วง step แรกๆ
# พอเดินไปได้สักพักมันเห็นหน้าหลังล็อกอินเองแล้ว ไม่ต้องจ่ายค่าบรรทัดนี้ทุกเทิร์นจนจบงาน
_MAX_ALREADY_LOGGED_IN_REMINDER_STEPS = 3

# W_session_drift: จำนวนครั้งสูงสุดที่ระบบจะ login ใหม่ให้เองกลางทาง (ดู guard ต้นลูปหลัก) —
# เผื่อ session หมดอายุจริงระหว่าง task ยาว แต่ไม่ปล่อยให้วน login ไม่รู้จบถ้า credential ใช้
# ไม่ได้จริง (กรณีนั้น _maybe_auto_login() จะคืนเหตุผลความล้มเหลวออกมาอยู่แล้ว)
_MAX_MID_TASK_RELOGINS = 2

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


# W29 ("Loop-guard blind spot" — บั๊กจริงที่ user รายงาน: agent ติดลูปกดตัวเลือก dropdown
# ซ้ำๆ ไม่หยุด (สลับ 2 index ไปมา) ทั้งที่ loop-guard คาบ 1/2-4 ด้านล่างควรจับได้): dict ของ
# action (tool_input) มี key "completed_plan_step" ติดมาด้วย "เฉพาะครั้งแรก" ที่ action นั้น
# ทำให้ step ของแผนเสร็จสมบูรณ์ (ดู llm.py SYSTEM_PROMPT "W_planbug" — ห้าม mark step เดิม
# ซ้ำสองครั้ง) รอบถัดๆ ไปของ action ที่ "เหมือนเดิมเป๊ะ" ทุกอย่าง (type/index/ฯลฯ) จะไม่มี
# key นี้อีกแล้ว — ถ้าเทียบ dict ทั้งก้อนตรงๆ (รวม completed_plan_step) รอบแรกกับรอบถัดไปจะ
# "ไม่เท่ากัน" ทั้งที่จริงๆ คือ action เดิมเป๊ะที่ agent สั่งซ้ำ ทำให้ทั้ง guard คาบ 1
# (consecutive_repeat_count) และคาบ 2-4 (_detect_repeating_cycle_period ด้านล่าง) มองไม่เห็น
# การวนซ้ำเลย (เจอจริง: click(22) -> click(25) -> click(22) -> click(25) ซ้ำไม่รู้จบ เพราะ
# click(22) รอบแรกมี completed_plan_step ติดมาด้วย รอบหลังๆ ไม่มี — equality เพี้ยนทุกรอบ) —
# ตัด completed_plan_step ออกก่อนเทียบ/เก็บเข้า recent_actions เสมอ (ไม่กระทบการ dispatch
# จริงเลย — จุดที่ยังใช้ tool_input ดิบเดิมเพื่อ execute()/บันทึก plan_step_done ไม่ได้ถูกแตะ)
def _cmd_for_repeat_comparison(cmd: dict) -> dict:
    if "completed_plan_step" not in cmd:
        return cmd
    return {k: v for k, v in cmd.items() if k != "completed_plan_step"}


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
# W_token_trim (P1/Q2): 3 -> 2 — chunk อันดับ 3 ถูก rank ว่า marginal อยู่แล้ว
# (retriever เรียงตาม relevance) ตัดออก 1 ประหยัด ~1 chunk (~500 char) ต่อ step
# ที่หน้าเปลี่ยน โดยแทบไม่เสีย recall จริง
_RAG_CHUNKS_PER_STEP = 2

# W7[A] (long-term): เหมือน _RAG_CHUNKS_PER_STEP แต่สำหรับ long_term_memory.recall()
# (ประวัติ task run อื่นก่อนหน้า แทนคู่มือที่ user ป้อน) — ดึงใหม่ทุก step เหมือนกัน
_LONG_TERM_MEMORY_CHUNKS_PER_STEP = 2  # W_token_trim (P1/Q2): 3 -> 2

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
# W_token_trim (P2/M1): 6 -> 4 — digest replay + stale-snapshot stub ด้านล่างจัดการ
# bulk แล้ว compact ถี่ขึ้นเพื่อตัด raw turn เก่าออกเร็วขึ้น (_KEEP_RECENT_STEPS คงที่ 3)
_COMPACT_AFTER_STEPS = 4
_KEEP_RECENT_STEPS = 3

# W_token_trim (P2/M1): stub เนื้อ "Current page:" ของ user turn เก่าทุกอันยกเว้น N
# อันล่าสุด — page snapshot เก่าไร้ค่าทันทีที่มี snapshot ใหม่กว่า (SYSTEM_PROMPT W19
# "Exact Element Matching"/"Action Trap" สั่งให้โมเดลยึดเฉพาะ snapshot ล่าสุดอยู่แล้ว
# ไม่ให้อ้าง index เก่าข้าม step) — เก็บ 2 อันล่าสุดไว้เต็ม (อันก่อนหน้า + อันปัจจุบัน
# ที่ next_action จะ append) เผื่อโมเดลต้องเทียบ "หน้าก่อน action ล่าสุด" กับ "หน้าตอนนี้"
_STALE_SNAPSHOT_MARKER = "\n\nCurrent page:\n"
_SUPERSEDED_SNAPSHOT_STUB = (
    "[snapshot from an earlier step — superseded; act only on the latest snapshot below]"
)


def _stub_snapshot_in_text(text: str) -> Optional[str]:
    """คืน text ที่แทนเนื้อ page snapshot ด้วย stub — None ถ้าไม่มี snapshot ในนี้เลย
    page_text = "\\n".join(element lines) ไม่มี "\\n\\n" ข้างในเลย (perception.py) — block
    จึงจบพอดีที่ "\\n\\n" ตัวถัดไป (section หน้าถัดไป) หรือท้าย string — parse ง่าย/ทน"""
    i = text.find(_STALE_SNAPSHOT_MARKER)
    if i == -1:
        return None
    start = i + len(_STALE_SNAPSHOT_MARKER)
    j = text.find("\n\n", start)
    end = len(text) if j == -1 else j
    if text[start:end] == _SUPERSEDED_SNAPSHOT_STUB:
        return None  # stubbed อยู่แล้ว — idempotent
    return text[:start] + _SUPERSEDED_SNAPSHOT_STUB + text[end:]


def _dedupe_stale_snapshots(messages: list, keep_last_full: int = 1) -> list:
    """W_token_trim (P2/M1): provider-agnostic — จับ user turn ที่มี page snapshot
    (content เป็น str ของ Anthropic/Groq/OpenAI หรือ parts[0].text ของ Gemini; tool_result
    turn ของ Anthropic เป็น list ไม่ใช่ str จึงข้ามเอง) แล้ว stub ทุกอันยกเว้น
    keep_last_full อันท้าย — ไม่แตะ turn อื่น ไม่ throw"""
    hits: list[tuple[int, str, str]] = []
    for k, m in enumerate(messages):
        if not isinstance(m, dict) or m.get("role") != "user":
            continue
        content = m.get("content")
        if isinstance(content, str):
            if _STALE_SNAPSHOT_MARKER in content:
                hits.append((k, "content", content))
        elif isinstance(m.get("parts"), list) and m["parts"]:
            part_text = m["parts"][0].get("text") if isinstance(m["parts"][0], dict) else None
            if isinstance(part_text, str) and _STALE_SNAPSHOT_MARKER in part_text:
                hits.append((k, "parts", part_text))
    if len(hits) <= keep_last_full:
        return messages
    targets = hits[:-keep_last_full] if keep_last_full > 0 else hits
    out = list(messages)
    for k, kind, text in targets:
        stubbed = _stub_snapshot_in_text(text)
        if stubbed is None:
            continue
        if kind == "content":
            out[k] = {**messages[k], "content": stubbed}
        else:
            new_parts = list(messages[k]["parts"])
            new_parts[0] = {**new_parts[0], "text": stubbed}
            out[k] = {**messages[k], "parts": new_parts}
    return out


# W_token_cut W5 (หลักฐาน W_prompt_audit 2026-09-02): "assistant history" (turn เก่าที่สะสม
# ใน messages) โตขึ้น ~4.5-5k tok ต่อ LLM call แบบไม่มีเพดาน และเป็น component ที่ใหญ่ที่สุด
# ใน task ยาว — _dedupe_stale_snapshots ยุบแค่บล็อก "Current page:" ของ user turn เก่า ส่วน
# ที่เหลือ (บล็อกกฎ gated ~3.9k tok, plan, scaffolding, manual, action-history) ยังค้างเต็ม
# ทุก turn เก่า ทั้งที่ทุกอย่างถูกส่ง "สด" ใหม่ใน turn ปัจจุบันอยู่แล้ว + digest เก็บสิ่งที่
# เกิดขึ้นไว้ครบ -> สำเนาเก่าไม่มีค่าเชิงข้อมูล มีแต่กิน token
#
# W5 = ยุบ user turn ของ step เก่า (เกิน _W5_KEEP_RECENT_FULL_TURNS อันท้าย) ให้เหลือแค่
# บรรทัด Goal + stub สั้นๆ. idempotent, provider-agnostic (str content ของ Anthropic/Groq/
# OpenAI + parts[0].text ของ Gemini), ไม่ throw. ไม่แตะ tool_result / assistant function_call
# (API ต้องการ call_id ที่จับคู่กัน) และไม่แตะ nudge turn (ไม่มี snapshot marker)
#
# keep_last_full=1: เก็บ turn ของ step ล่าสุดใน messages ไว้เต็ม 1 อัน + turn ปัจจุบันที่
# next_action จะ append เต็มอีก 1 = โมเดลเห็น 2 turn ล่าสุดครบทั้ง rules/plan/snapshot
# (นโยบายเดียวกับ _dedupe_stale_snapshots) ทุกอย่างในนั้นถูกส่งสดใหม่ทุก turn อยู่แล้ว
_W5_KEEP_RECENT_FULL_TURNS = 1
_W5_SUPERSEDED_TURN_STUB = (
    "[an earlier step's full context — page snapshot, indexed elements, rules, plan, "
    "recent-action list — has been omitted here to keep the conversation short. It is "
    "superseded. Act only on the latest turn below plus the digest of earlier steps.]"
)
# user turn ของ step จริงขึ้นต้นด้วยบรรทัดนี้เสมอ (ดู llm._build_user_turn_text) และมี
# snapshot marker (หรือ stub ของมันหลัง _dedupe_stale_snapshots) อยู่ข้างใน — ใช้แยกออกจาก
# nudge turn / _NO_TOOL_CALL_NUDGE / tool_result ที่ไม่ควรแตะ
_W5_STEP_TURN_PREFIX = "Goal: "


def _w5_step_turn_text(m) -> tuple[Optional[str], Optional[str]]:
    """คืน (kind, text) ถ้า m คือ user turn ของ step จริงที่ยุบได้ — ไม่งั้น (None, None)"""
    if not isinstance(m, dict) or m.get("role") != "user":
        return None, None
    content = m.get("content")
    if isinstance(content, str):
        text = content
        kind = "content"
    elif isinstance(m.get("parts"), list) and m["parts"] and isinstance(m["parts"][0], dict):
        text = m["parts"][0].get("text")
        kind = "parts"
    else:
        return None, None
    if not isinstance(text, str) or not text.startswith(_W5_STEP_TURN_PREFIX):
        return None, None
    has_snapshot = (_STALE_SNAPSHOT_MARKER in text) or (_SUPERSEDED_SNAPSHOT_STUB in text)
    if not has_snapshot:
        return None, None  # nudge/other user turn — ไม่แตะ
    return kind, text


def _compact_stale_user_turns(messages: list, goal: str,
                              keep_last_full: int = _W5_KEEP_RECENT_FULL_TURNS) -> tuple[list, int]:
    """W_token_cut W5: ยุบ user turn ของ step เก่าให้เหลือ "Goal: ...\\n\\n<stub>"

    คืน (messages_ใหม่, จำนวนตัวอักษรที่ตัดออกได้จริงรอบนี้). idempotent — turn ที่ยุบแล้ว
    (content == goal+stub) ถูกข้าม ไม่ throw"""
    try:
        stub_body = f"{_W5_STEP_TURN_PREFIX}{goal}\n\n{_W5_SUPERSEDED_TURN_STUB}"
        hits = [k for k, m in enumerate(messages) if _w5_step_turn_text(m)[0] is not None]
        if len(hits) <= keep_last_full:
            return messages, 0
        targets = hits[:-keep_last_full] if keep_last_full > 0 else hits
        out = list(messages)
        removed = 0
        for k in targets:
            kind, text = _w5_step_turn_text(messages[k])
            if text is None or text == stub_body:
                continue
            removed += len(text) - len(stub_body)
            if kind == "content":
                out[k] = {**messages[k], "content": stub_body}
            else:
                new_parts = list(messages[k]["parts"])
                new_parts[0] = {**new_parts[0], "text": stub_body}
                out[k] = {**messages[k], "parts": new_parts}
        return out, max(0, removed)
    except Exception:
        return messages, 0


# W_token_cut W7: บล็อกกฎที่ gate (~3.9k tok บนหน้าตาราง) ถูก W2 ย้ายจาก system prompt
# (cache ได้) มาต่อท้าย user turn ทุก turn — เนื้อเหมือนกันหมด model ถูกสั่งให้ยึด turn
# ล่าสุดอยู่แล้ว (W19 Action Trap) สำเนาใน turn เก่าจึงไม่มีค่า ตัดออกเหลือ 1 บรรทัดอ้างอิง
# บล็อกอยู่ท้ายสุดของ user turn เสมอ (ดู llm._build_user_turn_text) หา header line แล้วตัด
# ตั้งแต่ตรงนั้นถึงจบ string — keep_last_full=0 = ทุก turn ใน messages (turn ปัจจุบันที่
# next_action จะ append ยังส่งเต็ม) idempotent, provider-agnostic, ไม่ throw
def _dedupe_stale_gated(messages: list, keep_last_full: int = 0) -> tuple[list, int]:
    try:
        hdr = llm.GATED_BLOCK_HEADER
        deref = llm._GATED_BLOCK_DEREF

        def _turn_text(m):
            if not isinstance(m, dict) or m.get("role") != "user":
                return None, None
            c = m.get("content")
            if isinstance(c, str):
                return ("content", c) if ("\n\n" + hdr + "\n") in c else (None, None)
            if isinstance(m.get("parts"), list) and m["parts"] and isinstance(m["parts"][0], dict):
                t = m["parts"][0].get("text")
                return ("parts", t) if (isinstance(t, str) and ("\n\n" + hdr + "\n") in t) else (None, None)
            return None, None

        hits = [k for k, m in enumerate(messages) if _turn_text(m)[0] is not None]
        if len(hits) <= keep_last_full:
            return messages, 0
        targets = hits[:-keep_last_full] if keep_last_full > 0 else hits
        out = list(messages)
        removed = 0
        for k in targets:
            kind, text = _turn_text(messages[k])
            cut = text.find("\n\n" + hdr + "\n")
            new_text = text[:cut] + "\n\n" + deref
            if new_text == text:
                continue
            removed += len(text) - len(new_text)
            if kind == "content":
                out[k] = {**messages[k], "content": new_text}
            else:
                new_parts = list(messages[k]["parts"])
                new_parts[0] = {**new_parts[0], "text": new_text}
                out[k] = {**messages[k], "parts": new_parts}
        return out, max(0, removed)
    except Exception:
        return messages, 0


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
    # W_token_trim (P1/Q1): clip result หัว+ท้ายเหมือน memory.py summaries — digest
    # ค้างอยู่ทั้ง task ยิ่งต้องไม่ฝัง result เต็ม (เช่น ตาราง read_page_data)
    return "\n".join(f"- step {h['step']}: {h['cmd']} -> {clip_result(h['result'])}" for h in entries)


_DIGEST_PREFIX = "[Digest of earlier steps, compacted to keep the conversation from growing too long]"


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
    "This action is rejected — this page still has an empty Password field. Do not move on to "
    "any other action (including wait) until both Username and Password are filled in. Look "
    "at the indexed elements and fill the empty field right now."
)


async def _login_form_needs_password(page: Page) -> bool:
    """เช็คจาก DOM จริง (ไม่ใช่ label จาก snapshot เพราะแยกไม่ออกชัดพอระหว่าง
    placeholder กับค่าว่างจริง) ว่าหน้าปัจจุบันมี input[type=password] ที่มองเห็นได้
    และยังว่างอยู่ไหม — ใช้เป็นสัญญาณว่า login form ยังกรอกไม่ครบ

    W19 (latency): .input_value() ไม่ระบุ timeout เองจะ default เป็น 30000ms ของ
    Playwright — ใส่ _DOM_CHECK_TIMEOUT_MS (3s) ตรงๆ กันรอนานเกินจำเป็นถ้า element หลุด/
    detach ระหว่างทาง

    W_change_password_form_is_not_a_login_form (บั๊กจริงจากรันสด 2026-09-03): หน้าเปลี่ยน
    รหัสผ่านมีช่อง password ว่าง 3 ช่อง ตัวตรวจนี้จึงตอบ True แล้วผู้เรียกทั้งสองรายเข้าใจผิด
    ว่าหลุดกลับมาหน้า login — guard session-drift ยิง auto-login ซ้ำ 2 รอบ และ guard
    login-form ปฏิเสธทุก action ที่ไม่ใช่ fill/goto อีก 4 ครั้งจน backstop ฆ่า task ทิ้ง
    (4 steps, ไม่มี action ไหนผิดเลยสักตัว) — ฟอร์มเปลี่ยนรหัสผ่านไม่ใช่ฟอร์ม login และ
    หน้า login จริงมีช่อง password ช่องเดียวจึงไม่มีทางเข้าเงื่อนไขนี้ ความปลอดภัยเดิมคงอยู่ครบ
    """
    try:
        if await _page_looks_like_change_password_form(page):
            return False
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
# settings.step_pacing_delay_seconds เท่านั้น (ดู last_llm_call_at ใน run_task()) — ยัง
# การันตีระยะห่างขั้นต่ำเท่าเดิมทุกประการ (ไม่ลดความปลอดภัยจาก rate-limit เลย) แค่ไม่เสีย
# เวลาเปล่าซ้ำกับงานที่ทำไปแล้วจริงระหว่าง step นั้น — ผลคือ step สุดท้ายก่อนจะรู้ว่า LLM
# ตัดสินใจเรียก finish_task (ซึ่งงานจริงของ step ก่อนหน้ามักกินเวลาไปเกิน 3 วินาทีอยู่แล้ว
# จาก wait_stable()/network) มักไม่ต้องรอเพิ่มเลยหรือรอสั้นลงมาก
#
# Speed 2.4: ย้ายจาก module constant (hardcode 3 เสมอ) เข้า Settings เพื่อปรับได้ตาม
# provider/tier โดยไม่ต้องแก้โค้ด (ดู config.py::step_pacing_delay_seconds)

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


# W_consent_banner (บั๊กจริง live-reproduce บน opensource-demo.orangehrmlive.com หลายรอบ —
# ไม่คงที่ โผล่บ้างไม่โผล่บ้าง): แบนเนอร์ขอความยินยอมคุกกี้ (Cookiebot) วางทับหน้า login อยู่
# indexed elements 11 ตัวแรกกลายเป็นปุ่มของแบนเนอร์ทั้งหมด ทำให้ _maybe_auto_login() หาฟอร์ม
# login ไม่เจอ -> ไม่ล็อกอินให้ -> agent หลงไปยิง action ใส่ element ของแบนเนอร์แทน (เจอทั้ง
# fill_secret ใส่ div ของแบนเนอร์ และ recovery ที่คลิกปุ่ม "Show details" ของแบนเนอร์)
#
# ไม่ใช่ปัญหาเฉพาะเว็บนี้เลย — CMP รายใหญ่ (Cookiebot/OneTrust/Didomi/Usercentrics/CookieYes)
# ใช้รูปแบบเดียวกันหมดและอยู่บนเว็บจริงจำนวนมาก จึงจัดการที่นี่ครั้งเดียวก่อนเข้า loop แทนที่จะ
# หวังให้ LLM คลำเอาเองทุกครั้ง
#
# *** เลือก "ปฏิเสธ" เสมอ ไม่ใช่ "ยอมรับทั้งหมด" *** — ค่า default ที่เคารพความเป็นส่วนตัวของ
# user มากที่สุด (ไม่เปิด tracking/marketing cookie ให้โดยที่เจ้าของงานไม่ได้สั่ง) ถ้าไม่เจอปุ่ม
# ปฏิเสธจริงๆ ค่อย fallback ไปปุ่มปิด (×/Close) ซึ่งไม่ได้ให้ความยินยอมอะไรเลยเช่นกัน — ไม่มี
# เส้นทางไหนในฟังก์ชันนี้ที่กด "Allow all" ให้เด็ดขาด
_CONSENT_REJECT_TEXTS = (
    "reject all", "reject cookies", "reject", "deny all", "deny", "decline all", "decline",
    "only necessary", "necessary only", "use necessary cookies only", "essential only",
    "ปฏิเสธทั้งหมด", "ปฏิเสธ", "ไม่ยอมรับ", "เฉพาะที่จำเป็น", "ที่จำเป็นเท่านั้น",
)
_CONSENT_CLOSE_TEXTS = ("close banner", "close", "ปิด")

# W_consent_banner_midtask (บั๊กจริง live run 3): CMP บางเจ้าไม่ได้โผล่ตอนโหลดหน้าแรก แต่โผล่
# กลางทาง (หลัง navigate ภายในแอป/หลัง auto-login) — การปิดตอนเริ่ม task อย่างเดียวจึงไม่พอ
# ผลที่เจอ: step 3 โมเดลกดปุ่ม "Allow all" ของแบนเนอร์เอง (ให้ความยินยอม tracking cookie
# แทน user โดยไม่มีใครสั่ง) แล้วหลังจากนั้นหลุดไปคลิกลิงก์โฆษณาบนหน้าจนหมด max_steps
#
# ตรวจจาก elements snapshot ที่มีอยู่แล้วในมือทุก step (ไม่ต้องยิง JS เพิ่มถ้าไม่มีแบนเนอร์) —
# เจอ label ที่เป็นลายเซ็นของแบนเนอร์คุกกี้เมื่อไหร่ ค่อยเรียก _dismiss_consent_banner()
_CONSENT_LABEL_SIGNATURES = (
    "allow all", "accept all", "accept cookies", "allow selection", "reject all",
    "deny all", "manage cookies", "cookie settings", "this website uses cookies",
    "we use cookies", "ยอมรับคุกกี้", "เว็บไซต์นี้ใช้คุกกี้",
)


# W_captcha_detect (P3.8): ก่อนหน้านี้ทั้งโปรเจกต์ไม่มีการตรวจจับ CAPTCHA เลยสักบรรทัด —
# เจอ reCAPTCHA/Turnstile/Cloudflare interstitial เมื่อไหร่ snapshot จะแทบว่างเปล่า (widget
# อยู่ใน iframe ของ provider ที่อ่านข้ามไม่ได้) แล้ว agent จะไล่คลิกสิ่งที่เหลือจนหมด
# step budget แล้วรายงานเหตุผลผิด (เช่น "ไม่พบปุ่มที่ต้องการ") — ทั้งที่ความจริงคือติดกำแพงบอท
#
# นโยบายชัดเจน: *ไม่* พยายามแก้ CAPTCHA เอง — แจ้ง user แล้วให้คนทำ ใช้ request_user_input
# ที่มีอยู่แล้ว (กลไกเดียวกับที่ W_resume ใช้ขอข้อมูลกลางทาง) ไม่สร้างช่องทางใหม่
#
# ตรวจจากข้อมูลที่ perceive มาแล้วเท่านั้น ไม่แตะ browser เพิ่ม (หลักการเดียวกับ
# _snapshot_shows_consent_banner) — ใช้ทั้งชื่อ frame/label ที่ CMP-like ของ provider ทิ้งไว้
_CAPTCHA_LABEL_SIGNATURES = (
    "recaptcha", "hcaptcha", "captcha", "turnstile",
    "i'm not a robot", "im not a robot", "ยืนยันว่าไม่ใช่บอท",
    "verify you are human", "verifying you are human",
    "checking your browser", "cloudflare",
)


def _snapshot_shows_captcha(elements: list[dict], page_text: str = "") -> bool:
    """True ถ้า snapshot รอบนี้มีลายเซ็นของ CAPTCHA/bot wall — ไม่ throw ไม่แตะ browser"""
    for element in elements or []:
        label = str(element.get("label", "")).lower()
        if label and any(sig in label for sig in _CAPTCHA_LABEL_SIGNATURES):
            return True
    lowered = (page_text or "").lower()
    return any(sig in lowered for sig in _CAPTCHA_LABEL_SIGNATURES)


_CAPTCHA_USER_PROMPT = (
    "This page is showing a CAPTCHA / bot check, which I am not allowed to solve. Please "
    "complete it yourself in the browser window, then reply here (any text) so I can continue "
    "from where the task left off."
)


def _snapshot_shows_consent_banner(elements: list[dict]) -> bool:
    """W_consent_banner_midtask: True ถ้า elements ที่ perceive มารอบนี้มีลายเซ็นของแบนเนอร์
    ขอความยินยอมคุกกี้ — เช็คจากข้อมูลที่มีอยู่แล้ว ไม่แตะ browser เลย ไม่ throw"""
    for element in elements or []:
        label = str(element.get("label", "")).lower()
        if label and any(sig in label for sig in _CONSENT_LABEL_SIGNATURES):
            return True
    return False
# container ของ CMP รายใหญ่ — ใช้ยืนยันว่าปุ่มที่เจออยู่ใน "แบนเนอร์คุกกี้" จริง ไม่ใช่ปุ่ม
# "Reject"/"Deny" ของฟีเจอร์อื่นบนหน้า (เช่นหน้าอนุมัติใบลาที่มีปุ่ม Reject ของมันเอง —
# กดผิดจะกลายเป็นการปฏิเสธคำขอจริงของ user)
_CONSENT_CONTAINER_HINTS = (
    "cookie", "consent", "gdpr", "cookiebot", "onetrust", "didomi", "usercentrics",
    "cookieyes", "truste", "privacy-banner",
)

_CONSENT_DISMISS_JS = """(payload) => {
  const { rejectTexts, closeTexts, containerHints } = payload;
  const inConsentContainer = (el) => {
    let node = el;
    for (let depth = 0; node && depth < 8; depth++, node = node.parentElement) {
      const id = (node.id || "").toLowerCase();
      const cls = (typeof node.className === "string" ? node.className : "").toLowerCase();
      const label = (node.getAttribute && (node.getAttribute("aria-label") || "") || "").toLowerCase();
      if (containerHints.some(h => id.includes(h) || cls.includes(h) || label.includes(h))) return true;
    }
    return false;
  };
  const visible = (el) => {
    const r = el.getBoundingClientRect();
    if (r.width < 1 || r.height < 1) return false;
    const s = getComputedStyle(el);
    return s.visibility !== "hidden" && s.display !== "none";
  };
  const candidates = Array.from(document.querySelectorAll('button, a, input[type="button"], [role="button"]'));
  const pick = (texts) => candidates.find(el => {
    if (!visible(el) || !inConsentContainer(el)) return false;
    const t = ((el.innerText || el.value || el.getAttribute("aria-label") || "").trim()).toLowerCase();
    return t && texts.some(x => t === x || t.startsWith(x));
  });
  // ปฏิเสธก่อนเสมอ ปิดแบนเนอร์เป็นทางเลือกสุดท้าย — ไม่มี branch ไหนกด "accept/allow"
  const target = pick(rejectTexts) || pick(closeTexts);
  if (!target) return "";
  const shown = (target.innerText || target.value || "").trim().slice(0, 60);
  target.click();
  return shown || "(unlabelled)";
}"""


async def _dismiss_consent_banner(page: Page, verbose: bool = False) -> Optional[str]:
    """ปิดแบนเนอร์ขอความยินยอมคุกกี้ถ้ามี — คืนข้อความบนปุ่มที่กด หรือ None ถ้าไม่เจอ/ทำไม่ได้

    ห้าม throw เด็ดขาด (เหมือน _maybe_auto_login/perception): นี่คือส่วนเสริม ไม่ใช่สิ่งที่ task
    ต้องพึ่งพา เจอ error อะไรก็เดินหน้าต่อตามปกติ

    W_consent_banner_iframe (บั๊กจริง live run: เรียกฟังก์ชันนี้แล้วคืน None ทุกครั้งทั้งที่
    perception เห็นปุ่ม "Allow all"/"Deny" ของแบนเนอร์เต็มหน้า — โมเดลถึงกดปุ่มแบนเนอร์เอง
    ที่ step 4): CMP หลายเจ้า (รวม Cookiebot) render แบนเนอร์ใน iframe แยก — page.evaluate()
    มองเห็นแค่ main document เท่านั้น ต่างจาก perception.py ที่เดินเข้าไปอ่านทุก frame อยู่แล้ว
    (ดู resolve_frame) จึงเกิดสภาพ "เห็นแต่กดไม่ได้" — ต้องไล่ยิงทุก frame เหมือนกัน"""
    clicked = None
    for frame in [page, *page.frames]:
        try:
            clicked = await frame.evaluate(_CONSENT_DISMISS_JS, {
                "rejectTexts": list(_CONSENT_REJECT_TEXTS),
                "closeTexts": list(_CONSENT_CLOSE_TEXTS),
                "containerHints": list(_CONSENT_CONTAINER_HINTS),
            })
        except Exception as e:
            if verbose:
                print(f"  [consent-banner] ข้าม frame หนึ่ง (evaluate ล้มเหลว): {e}", flush=True)
            continue
        if clicked:
            break
    if not clicked:
        return None
    if verbose:
        print(f"  [consent-banner] กดปุ่มปฏิเสธ/ปิดแบนเนอร์: {clicked!r}", flush=True)
    try:
        await wait_stable(page)
    except Exception:
        pass
    return clicked



async def _maybe_auto_login(
    page: Page, verbose: bool, outcome: Optional[dict] = None,
) -> Optional[str]:
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
    รู้ตัวว่า credential ที่บันทึกไว้ใช้ไม่ได้แล้ว ไม่ใช่ปล่อยผ่านเงียบๆ เหมือนเดิม)

    W_auto_login_outcome_is_invisible: reason ที่คืนออกไปเป็น None ได้ทั้งกรณี "ข้าม" และ
    "สำเร็จ" แยกสองกรณีนี้ออกจากกันไม่ได้เลย และบรรทัด log ที่มีอยู่ก็ผูกกับ verbose ซึ่ง
    เส้นทาง API ส่ง False เสมอ ผลคือเวลา guard login_skip ยิงในงานจริง (เจอ 2 ใน 3 รอบของ
    goal เดียวกัน ห่างกันไม่กี่นาที) ตอบไม่ได้ว่า auto-login ล้มเหลวหรือทำงานปกติ —
    วัดก่อนแก้ตามกฎเดิมของโปรเจกต์นี้

    รายงานผลผ่าน dict `outcome` ที่ผู้เรียกส่งเข้ามา (คีย์ "result": "skipped"/"ok"/"failed")
    **โดยเจตนา ไม่เปลี่ยนชนิดของค่าที่คืน** — ลองเปลี่ยนเป็น tuple มาแล้วและเทสต์ล้ม 166 เคส
    เพราะมี 13 จุดที่ patch ฟังก์ชันนี้ด้วย AsyncMock ที่คืนค่าเดี่ยว การ unpack จึงพังตั้งแต่
    ก่อนเข้า try ของ run_task แล้วลามเป็นลูกโซ่ไปทั้งไฟล์ พารามิเตอร์ที่มีค่า default ทำให้
    ผู้เรียกเดิมและ mock เดิมใช้ได้เหมือนเดิมทุกประการ"""
    def _report(result: str) -> None:
        if outcome is not None:
            outcome["result"] = result

    try:
        from backend.app.site_learning import storage as site_storage
        from backend.app.site_learning.auto_login import find_login_fields, login_with_verification
        from backend.app.site_learning.extractor import extract_page as site_extract_page

        domain = extract_domain(page.url)
        creds = site_storage.load_credentials(domain)
        if not creds:
            _report("skipped")
            return None
        page_info, _ = await site_extract_page(page)
        username_selector, password_selector = find_login_fields(page_info)
        if not username_selector or not password_selector:
            _report("skipped")
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
            _report("failed")
            return reason or "ล็อกอินไม่สำเร็จด้วย credential ที่บันทึกไว้สำหรับเว็บนี้"
        await wait_stable(page)
        _report("ok")
        return None
    except Exception:
        _report("skipped")
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


async def _request_user_input(
    prompt_text: str, sensitive: bool, ask_user_func: Optional[AskUserFunc],
) -> tuple[bool, str]:
    """W_resume ("Mid-Task Input Request" — บั๊กจริงที่ user รายงาน: agent ขอรหัสผ่านใหม่
    กลางทางแล้ว finish_task(false) จบ task ทั้งหมดทิ้ง plan/messages/browser state เดิม
    ทำให้เทิร์นถัดไปที่ user ตอบค่ามาต้องเริ่มงานใหม่จากศูนย์) — หยุดรอคำตอบจาก human จริงๆ
    "กลางทาง" โดยไม่จบ loop เลย (คนละกลไกจาก finish_task โดยสิ้นเชิง — ผู้เรียกยังคง await
    coroutine เดิมอยู่ที่จุดนี้ ไม่คืน control กลับไปให้ caller ของ run_task() จนกว่าจะได้
    คำตอบ) ใช้ ask_user_func เดียวกับ _confirm_plan()/permission prompt ทุกประการ (cmd
    dict บอกชนิดคำขอผ่าน "type" — ask_user_func เดิม/routes.py::_make_ask_user_func ไม่
    ต้องแก้อะไรเลย เพราะ forward cmd แบบ generic อยู่แล้ว ไม่ special-case type ไหนเป็น
    พิเศษ)

    คืนค่า (provided, answer) — provided=False ถ้า user ปฏิเสธ/หมดเวลา (ask_user_func คืน
    False) answer มาจาก cmd["answer"] ที่ resolve_approval() (task_manager.py) mutate เข้า
    cmd dict ก้อนเดียวกับที่เรา await อยู่นี้ตรงๆ ก่อน future resolve กลับมา — pattern
    เดียวกับที่ _confirm_plan() อ่าน cmd["plan"] กลับมาทุกประการ (ห้ามเชื่อตัวแปรที่ปิด
    scope ไปแล้วตอนส่งเข้า ask_user_func ต้องอ่านจาก cmd หลัง await เสร็จเท่านั้น)"""
    if ask_user_func is not None:
        cmd = {"type": "request_user_input", "prompt": prompt_text, "sensitive": sensitive}
        provided = bool(await ask_user_func(cmd))
        return provided, (cmd.get("answer", "") if provided else "")
    print(f"\n=== Agent ต้องการข้อมูลเพิ่มเติม ===\n{prompt_text}", flush=True)
    answer = await asyncio.to_thread(input, "คำตอบ: ")
    return True, answer



def _plan_step_lines(plan_text: Optional[str]) -> list[str]:
    """W_plan_step_cursor: บรรทัดที่ไม่ว่างของแผน = 1 step ต่อ 1 บรรทัด (นิยามเดียวกับที่
    _total_plan_steps ใช้มาตลอด แยกออกมาเพื่อให้ทั้งการนับและการแสดงผลอ่านจากที่เดียวกัน)"""
    return [line.strip() for line in (plan_text or "").splitlines() if line.strip()]


def _total_plan_steps(plan_text: str) -> int:
    return len(_plan_step_lines(plan_text))


def _focused_plan_context(plan_text: Optional[str], cursor: int) -> str:
    """W_plan_step_cursor: แผนที่ส่งให้โมเดลทุก step ต้องบอกด้วยว่า "ตอนนี้อยู่ข้อไหน"

    เดิมส่งแผนทั้งก้อนดิบๆ ซ้ำทุก step โดยไม่มีอะไรบอกลำดับเลย โมเดลเล็กจึงกระโดดไปทำข้อ
    ท้ายๆ หรือรายงานว่าข้อ 5 เสร็จตั้งแต่ action แรกได้ (บั๊กจริง live run 2026-08-27:
    read_page_data มากับ completed_plan_step=5 แล้ว Goal Boundary Gate หยุด task พร้อม
    อ้างว่าสำเร็จทั้งที่ยังไม่ได้ลบอะไรเลย)

    ทำเครื่องหมายตามความจริงเท่านั้น: ข้อก่อน cursor = เสร็จแล้ว, ข้อที่ cursor = กำลังทำ,
    ที่เหลือ = ยังไม่ถึง — ไม่บังคับ "วิธี" บังคับแค่ "ลำดับ"

    W_token_trim (P1/Q6): เดิมส่งทุกข้อของแผนเต็มๆ ทุก step (แค่สลับ marker [done]/
    CURRENT/[not yet]) — แผนยาวๆ เปลืองซ้ำทุก step โดยไม่จำเป็น ตอนนี้ส่งแค่
    CURRENT + NEXT + FINAL step แบบเต็มข้อความ ข้อที่ทำไปแล้วยุบเป็นบรรทัดนับเดียว
    ข้อระหว่าง NEXT กับ FINAL ยุบเป็นบรรทัดช่วงเดียว — ยังคง "เห็นปลายทาง" (FINAL step
    เต็มข้อความ) ตามเหตุผลเดิมที่โมเดลต้องรู้จุดหมายถึงจะเลือกวิธีของข้อปัจจุบันได้ถูก

    cursor เกินจำนวนข้อ = แผนจบครบแล้ว ยังคืนค่าที่ไม่ว่าง (guard ที่ตามมายังต้องเห็นว่า
    มีแผนอยู่จริง)"""
    steps = _plan_step_lines(plan_text)
    if not steps:
        return ""
    total = len(steps)
    done = cursor - 1
    lines: list[str] = []
    if done > 0:
        shown = min(done, total)
        lines.append(f"[steps 1-{shown} already done ({shown}/{total})]")
    if cursor > total:
        lines.append("(every step is already complete)")
        return "\n".join(lines)
    lines.append(f">>> CURRENT STEP ({cursor}/{total}): {steps[cursor - 1]}")
    if cursor + 1 <= total:
        lines.append(f"NEXT STEP ({cursor + 1}/{total}): {steps[cursor]}")
    # ข้อระหว่าง NEXT (cursor+1) กับ FINAL (total) — ยุบเป็นบรรทัดเดียวถ้ามี >= 2 ข้อ
    if total - (cursor + 1) >= 2:
        lines.append(
            f"[steps {cursor + 2}-{total - 1} not shown to save space — still do them in order]"
        )
    # FINAL step เต็มข้อความ (ถ้ายังไม่ถูกแสดงไปแล้วในบรรทัด CURRENT/NEXT)
    if total >= cursor + 2:
        lines.append(f"FINAL STEP ({total}/{total}): {steps[total - 1]}")
    return "\n".join(lines)


# W_goal_scope ("Goal Boundary Gate" — hard-enforcement companion to W_stop_when_done above:
# real bug the user reported, same root cause as W_stop_when_done's own example: goal "go to
# Admin page" -> agent clicks Admin -> Admin page loads (goal done) -> agent keeps going,
# clicks into "Nationalities", one click away from creating a record nobody asked for.
# W_stop_when_done only *nudges* the model to stop, and only when an explicit multi-step plan
# was confirmed (plan_text set) — the literal "go to Admin page" ad-hoc goal in the bug report
# has no plan at all, so that nudge never even fires for it. This guard is a bounded, code-level
# HARD stop that covers both gaps: (1) an ad-hoc/no-plan path using a narrow heuristic for
# single-objective navigation goals, (2) reuses the plan-based signal but actually blocks
# dispatch instead of just asking nicely. See the enforcement guard further down (right after
# the finish_task block) for where this is actually applied.
_MAX_PREMATURE_GOAL_SCOPE_RETRIES = 1
# Read-only/non-mutating action types still allowed once the goal is judged scope-satisfied —
# same category this file already treats as "not real progress toward/away from a goal"
# elsewhere (see _MUTATING_ACTION_TYPES's docstring above: "ไม่นับ read_page_data/wait/hover/
# scroll").
_GOAL_SCOPE_ALLOWED_ACTION_TYPES = {"read_page_data", "wait", "scroll", "hover"}

# W_plan_panel_lags_the_log (user รายงาน 2026-09-07): PLAN panel ติ๊กครบทุกข้อพร้อมกันตอน task จบ
# ทั้งที่ LOG เดินสดปกติ — สาเหตุคือ event เดียวที่บอกความคืบหน้าของแผน (plan_step_done) ยิงก็ต่อ
# เมื่อโมเดลรายงาน completed_plan_step มาเอง และส่งค่า plan_cursor-1 ซึ่งเป็น 0 ตราบใดที่ cursor
# ยังไม่ขยับ (แถวในแผนนับจาก 1 หน้าเว็บจึงไม่มีอะไรติ๊ก) ส่วน LOG ทันเพราะ event "step" ยิงทุก
# action ไม่มีเงื่อนไข
#
# ตัวที่ทำให้ PLAN ทันคือ cursor ตัวที่สองที่ตัดสินจาก "หลักฐานที่มองเห็นบนหน้าเว็บ" (URL เปลี่ยน
# หรือ action ตรงกับข้อความของข้อนั้น) ตามที่ user เลือกไว้ว่าเอาความทันใจ
#
# *** ข้อจำกัดที่ห้ามละเมิด: cursor ตัวนี้ใช้ "แสดงผลอย่างเดียว" ***
# plan_cursor ตัวเดิมป้อน plan_fully_completed -> goal-scope hard stop ซึ่งเป็นต้นเหตุบั๊ก
# false completion (W_plan_counter_claims_a_password_change: รายงานสำเร็จทั้งที่ยังไม่ได้กดบันทึก)
# การเร่ง cursor เดิมให้ขยับง่ายขึ้นเพื่อให้ UI ทันจะเปิดบั๊กนั้นกลับมาทันที จึงต้องแยกกันเด็ดขาด
_PLAN_NAV_STEP_KEYWORDS = (
    "ไปที่", "ไปยัง", "เปิดหน้า", "เข้าหน้า", "เข้าสู่ระบบ", "ล็อกอิน", "หน้า",
    "go to", "open", "navigate", "login", "log in", "sign in",
)


def _step_is_navigational(step_text: str) -> bool:
    """ข้อนี้เป็นขั้น "ไปที่หน้า X" หรือเปล่า — ใช้ contains_keyword() ที่ทนการเว้นวรรคของภาษาไทย
    ("ไป ที่ หน้า Admin" ต้องนับได้) ห้ามใช้ substring ดิบ ดูบทเรียนใน W_thai_keyword_space"""
    return contains_keyword(step_text, _PLAN_NAV_STEP_KEYWORDS)


def _display_step_evidence(
    step_text: str, action_label: str, action_type: str, url_changed: bool, success: bool,
) -> str:
    """หลักฐานว่า action นี้กำลังทำข้อที่ cursor แสดงผลชี้อยู่ — "match" / "url" / "action" / ""

    ฟังก์ชันบริสุทธิ์ ไม่มี side effect ทดสอบตรงๆ ได้ และไม่เรียก LLM (กฎเดิมของเส้นทางนี้)

    ใช้ _action_matches_plan_step() ซ้ำโดยเจตนา — มันมีสะพานไทย<->อังกฤษผ่าน regex ที่มีอยู่แล้ว
    (step "แล้วกดค้นหา" กับปุ่ม "Search") docstring ของมันห้ามเอาไป *บล็อก* dispatch ซึ่งไม่ใช่
    กรณีนี้: cursor แสดงผลไม่ gate อะไรเลยสักอย่าง"""
    if not success:
        return ""
    if action_type in _GOAL_SCOPE_ALLOWED_ACTION_TYPES:
        return ""      # การอ่านหน้าไม่ใช่ความคืบหน้า — ใช้ชุดเดิม ไม่สร้างชุดที่สาม
    if _action_matches_plan_step(
        step_text, _label_without_markers(action_label or ""), action_type,
    ) is True:
        return "match"
    if _step_is_navigational(step_text):
        # goto/go_back/switch_tab นับเป็นการนำทางเสมอ แม้ URL จะเท่าเดิม (reload หน้าเดิม)
        if url_changed or action_type in ("goto", "go_back", "switch_tab"):
            return "url"
    # W_plan_ticks_every_row (user เลือกเอง 2026-09-07 หลังเห็นของจริงว่าติ๊กสดได้แค่ 1 ใน 4 แถว):
    # action ที่เปลี่ยนสถานะหน้าเว็บสำเร็จแล้ว แต่ผูกกับข้อความของข้อนี้ไม่ได้ ให้ถือว่าเดินหน้า
    # ไปหนึ่งข้อ วัดจริงแล้วสองสาขาข้างบนครอบได้แค่ส่วนน้อย เพราะแถวส่วนใหญ่ในแผนที่ LLM ร่างมา
    # เขียนกว้างเกินกว่าจะ match กับ label ของปุ่ม ("กรอกข้อมูลผู้ใช้", "ตรวจสอบผลลัพธ์") และ
    # ไม่ใช่ขั้นนำทางจึงไม่มี URL เปลี่ยนให้จับ
    #
    # ราคาที่จ่ายโดยรู้ตัว: แถวติ๊กเร็วกว่าความจริงเมื่อข้อเดียวกินหลาย action และเมื่อจำนวน action
    # มากกว่าจำนวนข้อ cursor จะไปค้างที่ข้อสุดท้ายจนกว่างานจะจบ — ทั้งสองอย่างไม่กระทบความถูกต้อง
    # ของงานเลย เพราะ cursor ตัวนี้ไม่เคยป้อน plan_fully_completed / prompt / guard ใด ๆ (ดู
    # ข้อจำกัดที่หัวข้อ W_plan_panel_lags_the_log ด้านบน) และการ cap ไว้ที่ข้อสุดท้ายทำให้มัน
    # พูดว่า "แผนจบแล้ว" ไม่ได้อยู่ดี ข้อสุดท้ายยังต้องรอ task สำเร็จจริงถึงจะติ๊ก
    return "action"

# W_verify_text_needs_a_write (บั๊กจริงที่ user เจอจาก token 2026-09-04): goal "เปิดเว็ปแล้วไป
# ที่หน้าแอดมิน" ใช้ LLM ไป 4 ครั้งเพื่อให้ได้ action เดียว หนึ่งในเทิร์นที่เสียไปคือ finish_task
# ที่ถูก guard ตาราง (W63[7.2]) ตีกลับ เพราะโมเดลส่ง verify_text มาด้วยทั้งที่งานนี้เป็นการ
# นำทางล้วน ไม่เคยสร้าง/แก้อะไรให้ไปโผล่ในตารางได้เลย
#
# comment เดิมของ guard นั้นเขียนไว้ว่า "เช็คเฉพาะตอนที่ LLM ระบุ verify_text มาเอง" ซึ่งตั้งอยู่
# บนสมมติฐานว่าโมเดลจะใส่มาเมื่อจำเป็นเท่านั้น — สมมติฐานนี้ผิดกับ gpt-5.4-mini บน endpoint
# ChatGPT OAuth ซึ่งกรอกทุก property ในสคีมาเสมอ (เหตุผลเดียวกับ W_fill_secret_schema_gate
# และ W_secret_stays_in_schema_forever) verify_text จึงมาทุกครั้งไม่ว่างานจะเป็นชนิดไหน
#
# หลักฐานที่ใช้แทนคือ "งานนี้เคยเขียนค่าลงฟอร์มไหม" — ถ้าไม่เคย ก็ไม่มีอะไรที่จะไปโผล่ในตาราง
# ต้องใช้สองสัญญาณ ไม่ใช่สัญญาณเดียว: เกณฑ์ "เคยเขียนค่าลงฟอร์มไหม" อย่างเดียวไม่พอ เพราะงาน
# สร้าง record ที่เดินด้วยการคลิกล้วนก็มีจริง (เทสต์ 5 ตัวที่ตรึงสัญญาหลักของ guard นี้ใช้ลำดับ
# แบบนั้นพอดี) จึงเช็คเจตนาของ goal ก่อน แล้วค่อยตกมาที่หลักฐานจาก action
#   - operation เป็น create/edit -> guard ทำงานเสมอ ไม่ว่าจะเดินด้วย action ชนิดไหน
#   - operation อ่านไม่ออก (unknown) -> เชื่อหลักฐาน: เคยเขียนค่าลงฟอร์มจริงไหม
# เลือกให้ปลอดภัยไว้ก่อนโดยตั้งใจ — ปิด guard เฉพาะตอนที่มั่นใจทั้งสองทางว่างานนี้ไม่มีอะไร
# ไปโผล่ในตารางได้เลย
_VALUE_WRITING_ACTION_TYPES = {"fill", "fill_secret", "select", "check"}
_TABLE_VERIFY_RELEVANT_OPERATIONS = {"create", "edit"}
_GOAL_SCOPE_GATE_NUDGE_TEMPLATE = (
    "[Rejected] The goal appears to already be satisfied ({reason}) — this action "
    "({action_type}) is outside what the goal actually asked for. Never explore other "
    "menus/pages or create/edit/delete data nobody asked for once the goal is done. If the "
    "goal is genuinely complete, call finish_task(success=true) on your very next turn "
    "instead."
)
_GOAL_SCOPE_GATE_HARD_STOP_MESSAGE_TEMPLATE = (
    "Stopping task: the goal was already satisfied ({reason}), but the model kept trying to "
    "act outside the goal's scope even after being told to stop — forcing "
    "finish_task(success=true) instead of returning control for another action."
)
# Goals containing any of these are treated as compound/multi-objective and are never
# classified as a simple navigation goal below (see _extract_simple_navigation_target) —
# required so an ad-hoc goal like "go to Admin page and delete the ESS user" still gets to
# execute every step it actually asked for, instead of being cut short the moment the
# navigation half is done.
_COMPOUND_GOAL_MARKERS = (
    " and ", ",", " then ", "และ", "แล้ว", "จากนั้น", " before ", " after that",
)
# Recognized "just go somewhere, nothing else" phrasings (English + Thai — this codebase's own
# tests already use phrasing like "ไปหน้า Admin จัดการผู้ใช้"). Checked as a literal prefix of
# the (lowercased) goal.
_SIMPLE_NAV_PREFIXES = (
    "navigate to ", "go to ", "goto ", "open ",
    "ไปที่หน้า", "ไปยังหน้า", "ไปหน้า", "ไปที่", "ไปยัง", "เปิดหน้า", "เข้าหน้า",
)
_SIMPLE_NAV_TRAILING_WORDS = ("page", "หน้า")


def _extract_simple_navigation_target(goal: str) -> Optional[str]:
    """W_goal_scope: True เฉพาะ goal ที่เป็น "ไปที่หน้า X" ล้วนๆ ไม่มี objective อื่นปนมาด้วย
    (เช่น "ไปหน้า Admin แล้วลบ user" ต้องคืน None — ดู _COMPOUND_GOAL_MARKERS) คืน target
    keyword ที่ตัด nav prefix + trailing "page"/"หน้า" ออกแล้ว (เช่น "Admin") หรือ None ถ้า
    goal ไม่ตรงรูปแบบนี้เลย — ใช้คู่กับ _navigation_target_reached() ด้านล่างเป็นหลักฐานว่า
    ถึงเป้าหมายจริงหรือยัง

    W_goal_scope_regression: also strips a leading English definite article ("the") right
    after the nav prefix — "go to the Admin page" was extracting "the Admin" (never a
    substring of any real URL path), which silently broke _navigation_target_reached()'s
    match for this common phrasing even though the destination genuinely was reached."""
    stripped = (goal or "").strip()
    if not stripped:
        return None
    lower = stripped.lower()
    if any(marker in lower for marker in _COMPOUND_GOAL_MARKERS):
        return None
    for prefix in _SIMPLE_NAV_PREFIXES:
        if lower.startswith(prefix.lower()):
            remainder = stripped[len(prefix):].strip(" .!ๆ")
            if remainder[:4].lower() == "the ":
                remainder = remainder[4:].strip()
            for trailing in _SIMPLE_NAV_TRAILING_WORDS:
                if remainder.lower().endswith(trailing) and len(remainder) > len(trailing):
                    remainder = remainder[: -len(trailing)].strip()
            return remainder or None
    return None


# W_thai_nav_target_never_matches_url (วัดจากงานจริง 2026-09-04): user เขียน goal เป็นภาษาไทย
# ("ไปที่หน้าแอดมิน") แต่ URL ของเว็บเป็นอังกฤษเสมอ (/web/admin/viewSystemUsers) การเทียบ
# target กับ path ตรงๆ จึงเป็นเท็จตลอดกาลสำหรับ goal ภาษาไทย — gate ที่ควรปิดงานให้เองไม่เคย
# ทำงานเลย แล้วงานก็ต้องจ่ายเทิร์นเพิ่มให้โมเดลเรียก finish_task เอง (และมักเสียอีกเทิร์นให้
# guard already_active_skip ระหว่างนั้น)
#
# ตารางนี้ตั้งใจให้เล็กและเพิ่มจากหลักฐานเท่านั้น — ใส่เฉพาะคำที่เจอในงานจริงแล้ว ห้ามเดาเติม
# ล่วงหน้าเป็นพจนานุกรม (บทเรียนเดียวกับที่ normalize_goal_for_matching เขียนไว้ว่าไม่แก้คำสะกด
# ด้วยพจนานุกรม) ถ้าวันหลังเจอคำใหม่ในรันจริง ค่อยเติมพร้อมอ้างรันนั้น
_NAV_TARGET_URL_ALIASES = {
    "แอดมิน": "admin",
    "ผู้ดูแลระบบ": "admin",
}


def _navigation_target_reached(target: str, page_url: str, last_action_record: list[dict]) -> bool:
    """W_goal_scope: หลักฐานว่าถึงหน้าเป้าหมายจริง (ไม่ใช่แค่เดา) — ต้องมีทั้ง (1) action
    ล่าสุดที่ execute() จริง (self.memory.recent(1)) สำเร็จและเป็นประเภท navigate จริงๆ
    (click/goto — ไม่นับ fill/wait/... ที่ไม่ได้ตั้งใจเปลี่ยนหน้า) และ (2) target keyword
    ปรากฏใน URL PATH จริง (ไม่ใช่ page_text) — จงใจไม่เช็คจาก page_text เพราะเมนู sidebar ของ
    เว็บพวกนี้มักโชว์ลิงก์ "Admin" ค้างอยู่ตลอดไม่ว่าจะอยู่หน้าไหน ถ้าเช็คจาก text จะ false
    positive ทันทีตั้งแต่ก่อนคลิกด้วยซ้ำ"""
    if not target or not last_action_record:
        return False
    record = last_action_record[0]
    if record.get("success") is not True:
        return False
    if (record.get("cmd") or {}).get("type") not in ("click", "goto"):
        return False
    path = urllib.parse.urlparse(page_url or "").path.lower()
    # W_thai_nav_target_never_matches_url: เทียบทั้งคำเดิมและคำที่ map เป็นอังกฤษแล้ว
    candidates = {target.lower()}
    for thai, english in _NAV_TARGET_URL_ALIASES.items():
        if thai in target.lower():
            candidates.add(english)
    return any(c in path for c in candidates)


# W_goal_scope_compound_login (ส่วนที่ backup ไม่มี — เพิ่มหลัง live-reproduce บั๊กเดิมซ้ำด้วย goal
# จริงของ user): _extract_simple_navigation_target() ปฏิเสธ goal ที่มี compound marker ทุกกรณี
# ซึ่งถูกต้องสำหรับ "ไปหน้า Admin แล้วลบ user" (ยังมีงานให้ทำต่อจริง) แต่ทำให้ goal อย่าง
# "login then goto adminmenu" ไม่เคยถูก gate เลย ทั้งที่ clause แรกเป็นแค่ login ที่ระบบทำเอง
# อยู่แล้วผ่าน _maybe_auto_login() ไม่ใช่งานที่โมเดลต้องลงมือ — ผลจริงที่เจอ: agent ถึงหน้า Admin
# ตั้งแต่ step 1 แล้วไปกด Edit/Save/Delete บนข้อมูลจริงต่อ
#
# รับเป็น nav target ก็ต่อเมื่อ clause สุดท้ายเป็น nav ล้วนๆ *และ* ทุก clause ก่อนหน้าเป็น login
# ล้วน — เงื่อนไขที่สองคือสิ่งที่กัน "ไปหน้า Admin แล้วลบ user" ไม่ให้ถูกตัดกลางคัน
_LOGIN_ONLY_CLAUSE_KEYWORDS = (
    "login", "log in", "sign in", "signin", "เข้าสู่ระบบ", "ล็อกอิน", "ล๊อกอิน", "ลงชื่อเข้าใช้",
)


def _is_login_only_clause(clause: str) -> bool:
    """True ถ้า clause นี้เป็นแค่คำสั่ง login เฉยๆ (ไม่มี objective อื่นปนมา) — ตัดคำ login ออก
    แล้วต้องไม่เหลือตัวอักษร/ตัวเลขอะไรที่สื่อถึงงานอื่นอีกเลย เพื่อไม่ให้ clause อย่าง
    "login and delete the ESS user" หลุดผ่านไปเป็น login-only"""
    lower = (clause or "").strip().lower()
    if not lower:
        return False
    if not any(kw in lower for kw in _LOGIN_ONLY_CLAUSE_KEYWORDS):
        return False
    for kw in _LOGIN_ONLY_CLAUSE_KEYWORDS:
        lower = lower.replace(kw, " ")
    return not re.search(r"[a-zA-Z\u0e00-\u0e7f0-9]", lower)


# W_open_site_prefix_blocks_nav_gate (วัดจากงานจริงของ user 2026-09-04): goal
# "เปิดเว็ปแล้วไปที่หน้าแอดมิน" ไม่เคยได้ nav target เลย เพราะ clause แรกคือ "เปิดเว็ป" ซึ่งไม่ใช่
# login จึงตกเงื่อนไขของ W_goal_scope_compound_login ทั้งที่เป็น no-op แบบเดียวกันเป๊ะ —
# run_task() ยิง goto ไปที่ url ให้ตั้งแต่ก่อนเข้า loop อยู่แล้ว โมเดลไม่ต้องทำอะไรกับ clause นี้
#
# ผลของการไม่มี nav target: goal-scope gate ไม่เคยปิดงานให้เอง พอโมเดลกดเมนูซ้ำที่ตัวเอง
# ยืนอยู่แล้วก็เสียเทิร์นไปกับ guard already_active_skip (วัดได้ 2 ใน 6 รอบ ครั้งละ ~15k token)
# แล้วต้องรออีกเทิร์นให้โมเดลเรียก finish_task เอง
_OPEN_SITE_ONLY_CLAUSE_KEYWORDS = (
    "open the website", "open the site", "open the page", "open website", "open site",
    "go to the website", "go to the site", "visit the site", "visit the website",
    "เปิดเว็ปไซต์", "เปิดเว็บไซต์", "เปิดหน้าเว็บ", "เปิดเว็ป", "เปิดเว็บ",
)


def _is_open_site_only_clause(clause: str) -> bool:
    """True ถ้า clause นี้แค่บอกให้ "เปิดเว็บ" เฉยๆ — รูปแบบเดียวกับ _is_login_only_clause()
    เป๊ะ (ตัดคำออกแล้วต้องไม่เหลืออะไรที่สื่อถึงงานอื่น) จึงกัน "เปิดเว็บแล้วลบ user" ได้เหมือนกัน"""
    lower = (clause or "").strip().lower()
    if not lower:
        return False
    if not any(kw in lower for kw in _OPEN_SITE_ONLY_CLAUSE_KEYWORDS):
        return False
    for kw in _OPEN_SITE_ONLY_CLAUSE_KEYWORDS:
        lower = lower.replace(kw, " ")
    return not re.search(r"[a-zA-Z฀-๿0-9]", lower)


def _is_noop_prefix_clause(clause: str) -> bool:
    """clause นำหน้าที่ระบบทำให้เองอยู่แล้ว ไม่ใช่งานที่โมเดลต้องลงมือ — login หรือ เปิดเว็บ"""
    return _is_login_only_clause(clause) or _is_open_site_only_clause(clause)


def _extract_goal_navigation_target(goal: str) -> Optional[str]:
    """W_goal_scope_compound_login (ดู comment ด้านบน): ลอง _extract_simple_navigation_target()
    ก่อนเสมอ (พฤติกรรมเดิมทุกประการ) — ถ้าไม่ผ่านค่อยตัด goal ตาม compound marker แล้วรับเฉพาะ
    กรณี "login แล้วไปหน้า X" ตามเงื่อนไขที่อธิบายไว้ข้างบน คืน None ในกรณีอื่นทั้งหมด"""
    direct = _extract_simple_navigation_target(goal)
    if direct:
        return direct
    text = (goal or "").strip()
    if not text:
        return None
    pattern = "|".join(re.escape(marker) for marker in _COMPOUND_GOAL_MARKERS)
    clauses = [c.strip() for c in re.split(pattern, text, flags=re.IGNORECASE) if c.strip()]
    if len(clauses) < 2:
        return None
    if not all(_is_noop_prefix_clause(c) for c in clauses[:-1]):
        return None
    return _extract_simple_navigation_target(clauses[-1])


# W_no_create_for_existing_goal (บั๊กจริงจากภาพหน้าจอที่ user ส่งมา, goal "...ลบ user
# role=ess ออกให้หมด"): กรองแล้วไม่เจอแถว ESS เหลือ — agent กลับ "แก้ปัญหาแทน user" ด้วยการ
# navigate ไปหน้า Add User (/web/admin/saveSystemUser) แล้วกรอกฟอร์มสร้าง user ใหม่ จน
# ระบบตอบ "Username: Already exists" กลับมา แทนที่จะรายงานตรงๆ ว่าไม่พบเป้าหมาย
#
# SYSTEM_PROMPT มีกฎนี้อยู่แล้วเป๊ะๆ ("NEVER solve the problem for the user ... e.g. going to
# an Add/Create page to create a replacement for what you couldn't find") — แต่เป็น prompt
# ล้วน ไม่มีอะไรบังคับระดับโค้ด ซึ่งเป็นเหตุผลประจำของไฟล์นี้ที่ทุกกฎสำคัญต้องมี guard หนุน
#
# ขอบเขตแคบโดยเจตนา: เปิดใช้เฉพาะ goal ที่เป็นงาน "ลบ/แก้ของที่มีอยู่" และ *ไม่มี* คำสั่งสร้าง
# ปนอยู่เลย — goal อย่าง "สร้าง user ใหม่แล้วลบคนเก่า" จะไม่โดนบล็อก
_CREATE_INTENT_KEYWORDS = (
    "add", "create", "new user", "register", "สร้าง", "เพิ่ม", "ลงทะเบียน",
)
# label/URL ที่บ่งบอกว่ากำลังจะเข้าสู่ flow สร้างรายการใหม่
# "add to ..." ตั้งใจไม่นับ — เป็นการ "เอาของที่มีอยู่ไปใส่ที่ไหนสักแห่ง" (Add to cart/Add to
# list) ไม่ใช่การสร้าง record ใหม่ ต่างจาก "Add User"/"+ Add"/"Create" ที่เปิดฟอร์มสร้างจริง
_CREATE_ACTION_LABEL_RE = re.compile(
    r"^\s*\+?\s*(?:add|create|new)\b(?!\s+to\b)|^\s*(?:เพิ่ม|สร้าง)", re.IGNORECASE,
)
_CREATE_URL_MARKERS = ("/add", "/create", "/new", "saveuser", "savesystemuser", "adduser")


def _goal_wants_to_create(goal: str) -> bool:
    lower = (goal or "").lower()
    return any(kw in lower for kw in _CREATE_INTENT_KEYWORDS)


def _goal_targets_existing_records_only(goal: str) -> bool:
    """True ถ้า goal เป็นงานกับ "ของที่มีอยู่แล้ว" ล้วนๆ (ลบ/แก้ทั้งหมด) โดยไม่มีคำสั่งสร้างปน"""
    if _goal_wants_to_create(goal):
        return False
    return _is_deletion_intent_goal(goal) or _is_edit_all_intent_goal(goal)


def _action_starts_create_flow(tool_input: dict, label: str) -> bool:
    """True ถ้า action นี้กำลังพาไปสู่ flow สร้างรายการใหม่ — ดูจาก label ของปุ่ม (Add/Create/
    New/เพิ่ม/สร้าง ที่ "ขึ้นต้น" ด้วยคำพวกนี้เท่านั้น กัน label อย่าง "Add to cart"/"Address"
    ที่ไม่เกี่ยว) หรือ URL ปลายทางของ goto"""
    if _CREATE_ACTION_LABEL_RE.search(label or ""):
        return True
    url = str(tool_input.get("url") or "").lower()
    return bool(url) and any(marker in url for marker in _CREATE_URL_MARKERS)


_NO_CREATE_NUDGE_TEMPLATE = (
    "[Rejected] This goal is about EXISTING records ({reason}) — it never asked you to create "
    "anything. Going to an Add/Create page ({what}) to make a new record is outside the goal's "
    "scope, even if the target you were looking for cannot be found. If the filtered table "
    "genuinely has no matching rows left, that IS the answer: call finish_task and report "
    "plainly what you found (or didn't find). Never invent a replacement record."
)


# W_no_credential_flow (บั๊กจริงจาก live run ของ goal user เอง "...ไปที่เมนูแอดมิน แล้วลบ user
# role=ess ออกให้หมด" รอบล่าสุด): หลังลบไป 2 ราย agent หลงทาง แล้วแทนที่จะกรอง Role=ESS ใหม่
# มันคลิก "Demo Source [Profile/Account Menu]" -> "Change Password" แล้วพยายามกรอกช่องรหัสผ่าน
# จนระบบตอบ "Invalid" — task จบด้วยการ "ขอให้ user ป้อนรหัสผ่านใหม่" ทั้งที่ goal ไม่เคยพูดถึง
# รหัสผ่านเลยสักคำ (เกิดซ้ำ 2 รอบใน run เดียว: step 9-13 และ step 17-18)
#
# ทำไม guard ที่มีอยู่ไม่ครอบ: exclusion ของ "[Profile/Account Menu]" ใน
# _find_goal_matching_nav_element() คุมเฉพาะ element ที่ "ระบบ" เลือกเองตอน forced recovery
# ส่วน fill_secret context guard คุมเฉพาะ action type fill_secret — ทั้งคู่ไม่แตะ "click ที่
# โมเดลเลือกเอง" ซึ่งเป็นทางที่บั๊กนี้เดินจริง
#
# ทำไมต้องเป็น hard reject ไม่มีโควตาปล่อยผ่าน (เหมือน W_no_create_for_existing_goal): การ
# เปลี่ยนรหัสผ่านของบัญชีที่ agent login อยู่ = ล็อกตัวเองออกจากระบบของ user จริง กู้คืนไม่ได้
# ด้วยการ go_back — ต่างจาก guard ประเภท "เสีย step เปล่า" ที่ยอมให้ลองผิดได้สองสามครั้ง
#
# ขอบเขตแคบโดยเจตนา: ปิด guard ทันทีถ้า goal พูดถึงรหัสผ่าน/บัญชีเอง (รวมถึง goal ที่ให้
# credential มาสำหรับ login ด้วย — ยอมปิด guard ในเคสนั้นดีกว่าไปบล็อกงานที่ user สั่งจริง)
_CREDENTIAL_GOAL_KEYWORDS = (
    "password", "passwd", "credential", "รหัสผ่าน", "พาสเวิร์ด", "เปลี่ยนรหัส",
    "account setting", "my account", "ตั้งค่าบัญชี", "โปรไฟล์ของฉัน",
)
# label ที่บ่งบอกว่ากำลังเข้า flow เปลี่ยน credential — จับทั้งคำเต็มเท่านั้น กัน label อย่าง
# "Password Policy" (หน้ารายงาน/ตั้งค่าระบบ ไม่ใช่การเปลี่ยนรหัสของตัวเอง) ไม่โดนไปด้วย
_CREDENTIAL_FLOW_LABEL_RE = re.compile(
    r"\b(?:change|update|reset|edit|forgot)\s+(?:your\s+|my\s+)?password\b"
    r"|\bpassword\s+(?:change|reset)\b"
    r"|เปลี่ยนรหัสผ่าน|ลืมรหัสผ่าน|ตั้งรหัสผ่านใหม่",
    re.IGNORECASE,
)
_CREDENTIAL_FLOW_URL_MARKERS = (
    "updatepassword", "changepassword", "resetpassword", "change-password",
    "reset-password", "/password",
)


def _goal_mentions_credentials(goal: str) -> bool:
    lower = (goal or "").lower()
    return any(kw in lower for kw in _CREDENTIAL_GOAL_KEYWORDS)


def _action_enters_credential_flow(tool_input: dict, label: str) -> bool:
    """True ถ้า action นี้กำลังพาไปสู่หน้าเปลี่ยนรหัสผ่าน — ดูจาก label ของปุ่ม/เมนู หรือ URL
    ปลายทางของ goto"""
    if _CREDENTIAL_FLOW_LABEL_RE.search(label or ""):
        return True
    url = str(tool_input.get("url") or "").lower()
    return bool(url) and any(marker in url for marker in _CREDENTIAL_FLOW_URL_MARKERS)


_NO_CREDENTIAL_FLOW_NUDGE = (
    "[Rejected] '{what}' opens a credential/account-settings flow, and this goal never mentions "
    "passwords or account settings at all. Changing the password of the account you are logged "
    "in with would lock the user out of their own system — that is not recoverable by going "
    "back. This is almost always a sign you lost track of the task: re-read the goal, look at "
    "the page you are actually on, and go back to the records you were working on (re-apply the "
    "filter if it was cleared). If you believe the goal is already complete, call finish_task "
    "instead."
)


# W_no_record_edit_for_delete_goal (บั๊กจริงจาก live run 227b771e ด้วย goal ของ user เอง
# "...ไปที่เมนูแอดมิน แล้วลบ user role=ess ออกให้หมด"): agent เข้า flow Edit -> Save แล้ว
# *เขียนทับ user record จริง 2 ครั้ง* — step 20 click 'Edit', step 22 click 'Save' ได้
# [Success confirmation found: "Successfully Updated"] กลับมาทั้งคู่ ทั้งที่ goal ไม่มีคำว่า
# แก้ไข/อัปเดตอยู่เลยสักคำ
#
# หนักกว่าบั๊ก Add User ที่ W_no_create_for_existing_goal แก้ไว้ด้วยซ้ำ: Add User ยังโดนระบบ
# ปฏิเสธเอง ("Username: Already exists") แต่ Save สำเร็จจริงและ go_back() กู้ไม่ได้
#
# ทำไม W_no_create_for_existing_goal ไม่ครอบ: มันจับ "flow สร้างรายการใหม่" (Add/Create/New)
# ส่วนนี่คือการแก้ record ที่มีอยู่แล้ว — mutation คนละชนิดที่ goal ก็ไม่ได้ขอเหมือนกัน
#
# เลือกบล็อกที่จังหวะ "Save" ไม่ใช่ที่ "Edit" โดยเจตนา: คลิก Edit เฉยๆ ไม่ทำอะไรเสียหาย และ
# บางเว็บใช้หน้า Edit เป็นทางผ่านไปหาปุ่มลบด้วยซ้ำ แต่ Save คือจุดที่เขียนจริงและไม่มีทางเป็น
# ส่วนหนึ่งของงานลบได้เลย
#
# ใช้ regex "ขึ้นต้นด้วย + word boundary" ไม่ใช่ substring (แบบเดียวกับ _CREATE_ACTION_LABEL_RE):
# save\b ไม่ match "Saved Searches" (ลิงก์นำทาง) และ update\b ไม่ match "Updated on" (หัวคอลัมน์)
#
# คำที่จงใจ *ไม่* ใส่ — ทุกตัวเป็น false positive ที่จะทำให้ guard นี้พังงานลบเสียเอง:
#   submit/confirm/ยืนยัน  เป็น label ของ *dialog ยืนยันการลบ* บนหลายเว็บ บล็อกแล้วลบไม่สำเร็จเลย
#   change password        W_no_credential_flow ครอบไปแล้ว
#   แก้ไข                   นั่นคือ "Edit" = ทางเข้า ไม่ใช่จุด commit (ดูย่อหน้าด้านบน)
# ด้วยเหตุผลข้อแรกนี้เองจึงไม่เอา _FORM_SUBMIT_LABEL_KEYWORDS ที่มีอยู่แล้วมาใช้ซ้ำ — มันมี
# confirm/ยืนยัน/submit ครบ เพราะถูกออกแบบมาตอบคำถาม "ควรสแกน validation error หลัง action ไหน"
# ซึ่งกว้างได้อย่างปลอดภัย คนละเจตนากับการบล็อก action
# W_plan_commits_a_record_edit: คำชุดเดียวกัน แต่ใช้คนละ anchoring เพราะตอบคำถามคนละข้อ —
# ตัวมี ^ ใช้กับ *label ของปุ่ม* (label ขึ้นต้นด้วยคำนี้ = ปุ่ม commit จริง ไม่ใช่ "Saved Searches")
# ส่วนตัวไม่มี ^ ใช้กับ *ข้อความของแผน* ซึ่งคำจะอยู่กลางประโยคเสมอ ("3. บันทึกการเปลี่ยนแปลง")
# แยกคำออกมาเป็นค่าคงที่เดียวเพื่อไม่ให้มีคลังคำ 2 ชุดที่ต้องแก้พร้อมกัน
_RECORD_COMMIT_WORDS = ("save", "update", "บันทึก", "อัปเดต")
_RECORD_COMMIT_ALTERNATION = "|".join(_RECORD_COMMIT_WORDS)

_RECORD_COMMIT_LABEL_RE = re.compile(
    rf"^\s*(?:{_RECORD_COMMIT_ALTERNATION})\b", re.IGNORECASE,
)
_PLAN_COMMIT_STEP_RE = re.compile(
    rf"(?:{_RECORD_COMMIT_ALTERNATION})", re.IGNORECASE,
)


def _goal_is_deletion_only(goal: str) -> bool:
    """True ถ้า goal เป็นงาน "ลบ" ล้วนๆ โดยไม่มีคำสั่งแก้ไข/สร้างปนอยู่เลย — แคบกว่า
    _goal_targets_existing_records_only() (ที่รวม edit-all ด้วย) เพราะ guard นี้ต้องไม่แตะ
    goal ที่สั่งแก้ไขจริงอย่าง "เปลี่ยน Role ของทุกคนที่เป็น ESS เป็น Admin" """
    if _goal_wants_to_create(goal) or _is_edit_all_intent_goal(goal):
        return False
    return _is_deletion_intent_goal(goal)


def _action_commits_a_record_edit(tool_input: dict, label: str) -> bool:
    """True ถ้า action นี้คือการกด "บันทึก" ฟอร์มแก้ไข — ดูจาก label เท่านั้น ไม่ดู URL เลย
    (ต่างจาก _action_starts_create_flow) เพราะ _CREATE_URL_MARKERS มี "savesystemuser"/
    "saveuser" อยู่แล้ว ถ้าตัวนี้ดู URL ด้วยจะยิงซ้ำกับ W_no_create บน action เดียวกัน แล้ว
    nudge สองข้อความจะขัดกันเอง"""
    return bool(_RECORD_COMMIT_LABEL_RE.search(label or ""))


_NO_RECORD_EDIT_NUDGE = (
    "[Rejected] '{what}' commits an edit to an existing record, but this goal only asks you to "
    "DELETE things — it never asks you to change or update anything. Saving here would "
    "permanently overwrite real data, and going back cannot undo it. Being on an edit form at "
    "all means you took a wrong turn: leave it (Cancel, or go_back), return to the list, and "
    "use the row's own Delete action instead. If the filtered list genuinely has no matching "
    "rows left, that IS the answer — call finish_task and report it plainly."
)


# --- W_fill_secret_hardening: fill_secret context guard ------------------------------------
# Ported from the w77-w91 line of work after this exact failure was live-reproduced again on
# opensource-demo.orangehrmlive.com with the OpenAI provider, goal "login then goto adminmenu":
# auto-login had ALREADY succeeded (Dashboard visible, sidebar showing "[3] a 'Admin'"), yet
# the model kept proposing fill_secret over and over — across 3 separate runs it burned 8-10
# steps and 220k-350k input tokens, at one point typing the real saved password into three
# unrelated fields (index 21/22/23) before giving up without ever calling finish_task.
#
# There is never a legitimate reason to dispatch fill_secret outside a genuine change-password
# context, so every attempt outside one is rejected here BEFORE dispatch — SYSTEM_PROMPT already
# says so (W65[3]), but relying on prompt compliance alone is precisely the failure this guard
# exists for. Provider-agnostic on purpose: Gemini happens not to make this mistake today, which
# is exactly why the difference showed up as "openai is broken" rather than as a missing guard.
_PASSWORD_CHANGE_INTENT_KEYWORDS = (
    "change password", "reset password", "update password", "new password",
    "change my password", "security settings",
    "เปลี่ยนรหัสผ่าน", "เปลี่ยนรหัส", "ตั้งรหัสผ่านใหม่", "รีเซ็ตรหัสผ่าน", "รหัสผ่านปัจจุบัน",
)

# ป้ายที่บอกว่าช่องนั้นคือ "รหัสผ่านปัจจุบัน" จริงๆ (ไม่ใช่ช่องตั้งรหัสใหม่) — เก็บไว้ที่นี่
# เพราะเวอร์ชันนี้ยังไม่มี state_filter.CURRENT_PASSWORD_LABEL_HINTS ให้ใช้ร่วมกัน ถ้าวันหลัง
# เพิ่มเข้าไปใน state_filter ให้ย้ายไปใช้ตัวเดียวกัน อย่าปล่อยให้มี 2 ชุด drift ออกจากกัน
_CURRENT_PASSWORD_LABEL_HINTS = (
    "current password", "old password", "existing password",
    "รหัสผ่านปัจจุบัน", "รหัสผ่านเดิม",
)


def _goal_or_plan_requests_password_change(text: str) -> bool:
    # W_thai_keyword_space: เทียบผ่าน contains_keyword() ที่ทนการเว้นวรรคของภาษาไทย —
    # ดูเหตุผลเต็ม (พร้อมบั๊กจริงที่มันแก้) ใน goal_intent.contains_keyword()
    return contains_keyword(text, _PASSWORD_CHANGE_INTENT_KEYWORDS)


# W_password_field_has_no_label_attributes (บั๊กจริงที่วัดกับหน้าเว็บจริงแล้ว 2026-09-03):
# ช่องรหัสผ่านของ OrangeHRM บนหน้า /web/pim/updatePassword ไม่มี label/aria-label/placeholder/
# name/id เลยสักตัว (วัดแล้วได้ '' ทั้ง 3 ช่อง) เพราะ <label> เป็น *พี่น้อง* อยู่ใน
# div.oxd-input-group ไม่ได้ผูกด้วย for= และไม่ได้ห่อ input ไว้ el.labels จึงว่างเปล่า
# ผลคือ gate ด้านล่างคืน False บนหน้าเปลี่ยนรหัสผ่านจริง -> fill_secret ถูกตัดออกจาก tool
# schema -> โมเดลไม่มีทางกรอกรหัสปัจจุบันได้เลย จึงยิง fill(21, "") ซ้ำจนโดน loop detector
# ฆ่าทิ้ง (รันสด 2026-09-03: 8 steps, จบด้วย Failed)
#
# เดินขึ้น ancestor หา <label> ตัวแรก — วิธีเดียวกับที่ perception.py ใช้อยู่แล้ว
# (getPrecedingSiblingLabelText) จำกัด 4 ชั้นเพื่อไม่ให้ไปคว้า label ของ field อื่นในฟอร์ม
# ความเข้มงวดของ W_add_user_form_false_positive ไม่หายไป: ยืนยันกับหน้า Add User จริงแล้วว่า
# ได้ 'password' / 'confirm password' เท่านั้น ไม่มี 'current password' -> ยัง False ตามเดิม
_PASSWORD_FIELD_LABEL_JS = r"""el => {
    const direct = (
        (el.labels && el.labels[0] && el.labels[0].innerText) ||
        el.getAttribute('aria-label') || el.getAttribute('placeholder') ||
        el.getAttribute('name') || el.id || ''
    );
    if (direct.trim()) return direct.toLowerCase();
    const byIds = (el.getAttribute('aria-labelledby') || '').split(/\s+/).filter(Boolean)
        .map(id => (document.getElementById(id) || {}).innerText || '').join(' ');
    if (byIds.trim()) return byIds.toLowerCase();
    let node = el;
    for (let i = 0; i < 4 && node; i++) {
        node = node.parentElement;
        if (!node) break;
        const lab = node.querySelector('label');
        if (lab && lab.innerText.trim()) return lab.innerText.toLowerCase();
    }
    return '';
}"""


async def _page_looks_like_change_password_form(page: Page) -> bool:
    """True ถ้าหน้าปัจจุบันมี input[type=password] ที่มองเห็นได้ >= 2 ช่อง (current/new[/confirm])
    *และ* อย่างน้อย 1 ช่องมี label/placeholder/name/id สื่อว่าเป็น "Current Password" จริงๆ

    W_add_user_form_false_positive: เช็คแค่ "มีช่อง password >= 2" ไม่พอ — ฟอร์ม "Add User"
    (Password + Confirm Password สำหรับตั้งรหัสให้ user ใหม่ ไม่ใช่รหัสปัจจุบันของคนที่ login
    อยู่) มี 2 ช่องเหมือนกันเป๊ะ ถ้านับแค่จำนวนจะถูกเข้าใจผิดว่าเป็น change-password form

    fail-safe คืน False ถ้าเช็คไม่ได้จริงๆ (ปลอดภัยกว่าเดาว่าใช่ — เดาว่าใช่แปลว่าปล่อยให้
    fill_secret พิมพ์รหัสผ่านจริงลงช่องที่ไม่รู้ว่าคืออะไร)"""
    try:
        password_inputs = await page.locator('input[type="password"]:visible').all()
        if len(password_inputs) < 2:
            return False
        for el in password_inputs:
            label = await el.evaluate(_PASSWORD_FIELD_LABEL_JS)
            if any(hint in label for hint in _CURRENT_PASSWORD_LABEL_HINTS):
                return True
        return False
    except Exception:
        return False


async def _current_password_field_is_empty(page: Page) -> bool:
    """True ถ้าหน้านี้เป็นฟอร์มเปลี่ยนรหัสผ่าน *และ* ช่อง Current Password ยังว่างอยู่

    W_secret_stays_in_schema_forever (บั๊กจริง วัดจาก debug ของรันสด 2026-09-03 รอบที่ 5):
    บนหน้าเปลี่ยนรหัสผ่าน ทุก tool call ที่โมเดลส่งมาคือ
    {'secret': 'current_password', 'type': 'fill_secret', 'index': 21} เหมือนกันหมดทุกเทิร์น
    แม้ระบบจะปฏิเสธพร้อมชี้ index ของช่องที่ยังว่าง ([22] Password, [23] Confirm Password)
    ไปแล้ว 2 ครั้งติด — นี่คืออาการเดิมที่ W_fill_secret_schema_gate บันทึกไว้เป๊ะ: gpt-5.4-mini
    กรอกทุก property ในสคีมาเสมอ พอ `secret` มี enum ค่าเดียวมันจึงส่งมาทุกครั้งแล้วลาก `type`
    เป็น fill_secret ไปด้วย หลักฐานยืนยัน: step ที่โมเดลเลือก click ได้ปกติ ล้วนเป็น step บน
    หน้าที่ fill_secret ไม่อยู่ในสคีมา
    บทสรุปเดิมของ W_fill_secret_schema_gate จึงใช้ได้ตรงตัว — nudge เอาไม่อยู่ ต้อง *ตัดออกจาก
    สคีมา* และตัดทันทีที่ช่อง Current Password ถูกกรอกแล้ว เพราะจากจุดนั้นไปมันไม่มีประโยชน์อีก"""
    try:
        password_inputs = await page.locator('input[type="password"]:visible').all()
        if len(password_inputs) < 2:
            return False
        for el in password_inputs:
            label = await el.evaluate(_PASSWORD_FIELD_LABEL_JS)
            if any(hint in label for hint in _CURRENT_PASSWORD_LABEL_HINTS):
                return not (await el.input_value(timeout=_DOM_CHECK_TIMEOUT_MS)).strip()
        return False
    except Exception:
        return False


async def _change_password_form_still_unfilled(page: Page) -> bool:
    """True ถ้ายังยืนอยู่บนฟอร์มเปลี่ยนรหัสผ่านที่มีช่องว่างเหลือ — หลักฐานตรงๆ ว่างานยังไม่จบ

    W_plan_counter_claims_a_password_change (บั๊กจริงจากรันสดผ่าน REST API 2026-09-04):
    task จบด้วย **success=true** ที่ step 4 ทั้งที่ยังไม่เคยกรอกช่อง Confirm Password และไม่เคย
    กดบันทึกเลย — ground truth ด้วยสคริปต์ไม่ใช้ LLM ยืนยันว่ารหัสผ่านของเดโมไม่ถูกเปลี่ยน
    (รหัสเดิมยังล็อกอินได้ ส่วน 12345678 ไม่ได้) สาเหตุคือ goal-scope hard stop เชื่อ
    plan_fully_completed ซึ่งมาจาก completed_plan_step ที่ *โมเดลรายงานเอง*

    นี่คือบั๊กคลาสเดียวกับ W_plan_cursor_not_proof เป๊ะ ต่างแค่ชนิดงาน — และกฎเดียวกันใช้ได้:
    "การวัดต้องชนะตัวนับเสมอ" งานเปลี่ยนรหัสผ่านมีการวัดตรงๆ อยู่แล้วเหมือนที่งานลบแบบมีเงื่อนไขมี
    คือ "ฟอร์มยังมีช่องรหัสผ่านว่างอยู่ไหม" ซึ่งอ่านจาก DOM ได้ตรงๆ ไม่ต้องเชื่อใคร

    ห้าม raise ตามกฎของไฟล์นี้ — คืน False (= ไม่ขัดขวาง) ถ้าอ่านไม่ได้"""
    try:
        if not await _page_looks_like_change_password_form(page):
            return False
        return any(not st.get("filled") for st in await _password_field_states(page))
    except Exception:
        return False


# W_secret_refilled_forever (บั๊กจริงจากรันสด 2026-09-03 ซ้ำ 2 รอบติดด้วยผลเหมือนกันเป๊ะ):
# พอ fill_secret กรอกช่อง Current Password สำเร็จ โมเดลสั่ง fill_secret ที่ index เดิมซ้ำทันที
# ทุกครั้ง ไม่เคยขยับไปช่อง Password/Confirm Password (ซึ่งอยู่ใน snapshot ครบพร้อม index ที่
# ถูกต้อง — ยืนยันด้วย probe หน้าจริงแล้ว) จนโดน same-label-loop ฆ่าที่ step 12 ทั้งสองรอบ
# ทุก action "สำเร็จ" หมดแต่ไม่มีความคืบหน้าเลยแม้แต่นิดเดียว
#
# รูปแบบเดียวกับ W_state_guard_shortcut: บอกว่า "อย่าทำซ้ำ" ไม่พอ ต้องชี้ index จริงของช่อง
# ถัดไปให้ ความว่าง/ไม่ว่างอ่านจาก DOM ตรงๆ ไม่ใช่จาก label (label ปิดค่าไว้แล้วตาม
# W_password_value_leaks_into_label) และไม่ส่งค่าจริงของช่องไหนออกไปทั้งสิ้น
_MAX_SECRET_REFILL_RETRIES = 2

# W_click_submits_with_empty_password_fields: โควตาเหมือน guard อื่นในไฟล์นี้ ไม่บล็อกตาย —
# ฟอร์มบางแบบมีช่องรหัสผ่านที่ "เว้นว่างได้" จริง (หน้าแก้โปรไฟล์ที่รวมการเปลี่ยนรหัสผ่านไว้
# ด้วย) ถ้าบล็อกถาวรจะทำให้เว็บกลุ่มนั้นกดบันทึกไม่ได้เลย
_MAX_EMPTY_PASSWORD_SUBMIT_RETRIES = 2

_PASSWORD_FIELD_STATE_JS = """() => Array.from(
    document.querySelectorAll('input[type="password"]')
).filter(
    el => el.getClientRects().length > 0
).map(el => ({
    index: el.getAttribute('data-ai-index'),
    filled: !!(el.value || '').trim(),
}))"""


# W_index_drift_measure (2026-09-07): จาก release gate เห็น action ที่ยิงใส่ index แล้วไปโดน
# element คนละตัวกับที่ตั้งใจ (fill(24) ไปโดน '-- Select --' แทน Email) แต่ trace ที่มีแยกไม่ออก
# ระหว่างสองสาเหตุที่แก้คนละทาง:
#   (1) หน้า re-render ระหว่าง snapshot กับ dispatch (LLM คิดอยู่หลายวินาที) -> index เดิมชี้
#       คนละ element กับที่โมเดลเห็นตอนตัดสินใจ
#   (2) โมเดลอ้าง index จากเทิร์นเก่าที่จำมาจาก history -> snapshot ปัจจุบันถูกต้องอยู่แล้ว
# ตัววัดนี้เทียบ label ตอน snapshot กับ label สดตอนจะ dispatch: ต่างกัน = สาเหตุ (1)
# เหมือนกันแต่ action ยังผิดเป้า = สาเหตุ (2) — วัดก่อน ค่อยตัดสินว่าจะแก้ทางไหน
_LIVE_LABEL_JS = """(el) => (el.getAttribute('data-ai-label') || el.innerText || el.value || '').trim()"""


async def _live_label_at_index(page: Page, index) -> Optional[str]:
    """label สดของ element ที่ index นั้น ณ ตอนนี้ — None ถ้าไม่มี element นั้นแล้ว/อ่านไม่ได้"""
    try:
        found = await page.query_selector(f'[data-ai-index="{index}"]')
        if found is None:
            return None
        return (await found.evaluate(_LIVE_LABEL_JS)) or ""
    except Exception:
        return None


async def _password_field_states(page: Page) -> list[dict]:
    """[{index, filled}] ของช่อง password ที่มองเห็นได้ — ห้าม raise ตามกฎของไฟล์นี้"""
    try:
        return await page.evaluate(_PASSWORD_FIELD_STATE_JS)
    except Exception:
        return []


def _label_for_index(elements: list[dict], index) -> str:
    for el in elements or []:
        if str(el.get("index")) == str(index):
            return str(el.get("label") or "")
    return ""


# W_state_guard_shortcut ("Point at the actual answer, not just 'try something else'" — real
# bug the user reported and live-reproduced (3 separate live runs) against OrangeHRM with the
# OpenAI provider: on a goal like "goto admin page", with the sidebar plainly showing
# "[3] a 'Admin' (navigation)", the model kept choosing fill_secret over and over — turn after
# turn, even across forced go_back/scroll recoveries (see
# _MAX_FILL_SECRET_CONTEXT_REJECT_RETRIES above). The generic nudge text alone ("take the
# action that actually matches the goal instead") wasn't enough to redirect it. Two distinct
# patterns were observed across the live runs, so the hint below checks both, cheaply and
# deterministically (never changes what gets dispatched, never overrides the guard):
#   (1) Sometimes the model picks the WRONG element entirely — e.g. index 35, "span 'manda
#       userTester [Profile/Account Menu]'", unrelated to "admin". Fix: scan the current
#       indexed elements for a navigation-region link whose label already matches a keyword
#       from the goal, and name its exact index/label.
#   (2) Sometimes the model picks the RIGHT element (index 3, the actual "Admin" link) but the
#       WRONG ACTION VERB — it tries to fill_secret a plain <a> tag as if authenticating with
#       it, instead of clicking it. Fix (1) is useless here since the target was never wrong —
#       this needs its own check: look up the model's own chosen index, and if that element
#       isn't a real <input>, tell it directly to use 'click' on that SAME index instead.
# _fill_secret_context_hint() below tries (2) first (it's the more specific, more actionable
# diagnosis when it applies), then falls back to (1).
# A third pattern was found later (W_fill_secret_recovery_excludes_already_active — the goal's
# own nav target is already "[already active]", i.e. the agent has ARRIVED and neither (1) nor
# (2) applies) and is checked ahead of both; see _fill_secret_context_hint()'s docstring.
_GOAL_KEYWORD_STOPWORDS = frozenset({
    "go", "goto", "to", "the", "a", "an", "page", "pages", "navigate", "open", "click",
    "on", "into", "and", "then", "please", "menu", "section", "screen", "view",
})


def _find_goal_matching_nav_element(
    elements: list[dict], goal: str, skip_already_active: bool = False,
) -> Optional[dict]:
    """คืน element แรกที่ region="navigation" (ดู perception.py::getRegion) ที่ label มีคำจาก
    goal ปนอยู่ (คำที่ยาว >= 3 ตัวอักษร ตัด stopword ทั่วไปทิ้งก่อน) — คืน None ถ้าไม่มี
    keyword ให้เทียบเลย/ไม่เจอ element ไหนตรงเลย ไม่ throw

    skip_already_active (default False, W_fill_secret_recovery_excludes_already_active): ข้าม
    element ที่ label มี "[already active]" (perception.py แปะให้เมนู/แท็บที่เป็นหน้าปัจจุบัน
    อยู่แล้ว) แล้ว "ไล่หาต่อ" ไม่ใช่เลิกหาทันที — nav element อื่นที่ตรง goal เหมือนกันแต่ยัง
    ไม่ active (เช่น เมนูย่อยของหน้าเดียวกัน) ยังควรถูกเลือกได้ตามปกติ ดู
    _fill_secret_recovery_target() สำหรับเหตุผลว่าทำไมเฉพาะ caller นั้นถึงต้องข้าม

    W_goal_token_substring (บั๊กจริงที่เจอตอน live test บน OrangeHRM): เดิมเทียบทางเดียว
    (goal word อยู่ใน label) — user พิมพ์ goal ว่า "login then goto adminmenu" ติดกันเป็นคำ
    เดียว ส่วน label บนหน้าเว็บคือ "Admin" ทำให้ "adminmenu" ไม่มีวัน match "admin" ได้เลย
    (คำที่ยาวกว่าหา substring ในคำที่สั้นกว่าไม่เจอเสมอ) แล้วตกไปเลือก element มั่วแทน —
    เทียบสองทางแทน โดยฝั่ง label ต้องเป็นคำยาว >= 4 ตัวอักษรถึงจะยอมให้ match กลับด้าน
    (กันคำสั้นอย่าง "PIM"/"Buzz" ไป match คำใน goal โดยบังเอิญ)

    W_goal_token_wordboundary (บั๊กจริงที่เจอตอน live test ต่อมา): เทียบแบบ substring ดิบๆ
    ทำให้ goal ภาษาไทย "...ลบuserole=ess ออกให้หมด" ดึงคำ ASCII ได้ ["userole", "ess"] แล้ว
    "ess" ไป match ข้างใน "businESS Solutions" — recovery เลยคลิกลิงก์ Business Solutions ที่
    ไม่เกี่ยวอะไรเลย ต้องเทียบที่ "ขอบเขตคำ" ไม่ใช่ substring กลางคำ ทั้งสองทิศทาง"""
    words = [w for w in re.findall(r"[a-zA-Z]{3,}", (goal or "").lower()) if w not in _GOAL_KEYWORD_STOPWORDS]
    if not words:
        return None
    for element in elements:
        if element.get("region") != "navigation":
            continue
        label = str(element.get("label", ""))
        if skip_already_active and "[already active]" in label:
            continue
        label_lower = label.lower()
        # ทิศทางที่ 1: คำจาก goal ปรากฏใน label แบบเป็น "คำ" จริงๆ (ไม่ใช่กลางคำอื่น)
        if any(re.search(rf"\b{re.escape(word)}\b", label_lower) for word in words):
            return element
        # ทิศทางที่ 2 (W_goal_token_substring): คำใน label เป็นส่วนหนึ่งของคำใน goal ที่ user
        # พิมพ์ติดกัน (เช่น label "admin" อยู่ใน goal word "adminmenu") — ต้องเป็นคำยาว >= 4
        # และต้องอยู่ต้นคำของ goal word เท่านั้น กันการชนกลางคำแบบ ess/business ซ้ำอีก
        label_words = [w for w in re.findall(r"[a-z]{4,}", label_lower) if w not in _GOAL_KEYWORD_STOPWORDS]
        if any(word.startswith(lw) for lw in label_words for word in words):
            return element
    return None


def _fill_secret_recovery_target(elements: list[dict], tool_input: dict, goal: str) -> Optional[dict]:
    """คืน element ที่ควร click แทน fill_secret ที่ถูกปฏิเสธ ถ้า heuristic แบบ deterministic
    หาเจอจริง (ไม่งั้นคืน None) — ใช้ pattern เดียวกับที่ _fill_secret_context_hint() ใช้สร้าง
    ข้อความเตือน (ดู comment เหนือ _fill_secret_context_hint สำหรับ pattern ทั้ง 2 แบบที่เจอจริง)
    แยกออกมาต่างหากเพื่อให้ _force_loop_recovery() เอาไปสั่ง click จริงได้ (W_state_guard_shortcut
    "Targeted Recovery") ไม่ใช่แค่บอกในข้อความเฉยๆ แล้วหวังว่าโมเดลจะทำตาม

    W_fill_secret_recovery_target_priority (real bug, live-reproduced twice on
    opensource-demo.orangehrmlive.com with the OpenAI provider — the very "pattern (1)" case
    the comment above _fill_secret_context_hint() already documented seeing live ("span
    'manda userTester [Profile/Account Menu]'"), now reproduced again as "span 'Automation QE'"
    — same account-dropdown-menu confusion, not a coincidence): unlike
    _fill_secret_context_hint() (pure suggestion text — the model can ignore it and try
    something else next turn, so a wrong guess there just wastes one hint, not one dispatch),
    this function's result gets clicked BLINDLY with no LLM double-check in between. Trying
    "pattern (2)" (trust the model's own chosen index) FIRST meant that whenever the model's
    fill_secret guess pointed at ANY non-<input> element — including one entirely unrelated to
    the goal, like an account/profile dropdown — it got force-clicked as-is, sending the
    recovery further from the goal instead of toward it (live-reproduced: 2 wasted rounds
    clicking the account dropdown before recovery finally reached the real "Admin" link on
    attempt 3). Goal-keyword matching is tried FIRST now instead — it is the safer signal for a
    BLIND dispatch (matches something the goal actually asked for), falling back to trusting the
    model's own chosen index only when no goal-matching nav element exists at all (preserves
    "pattern (2)": right element index, wrong action verb — e.g. the model correctly targeted
    the real "Admin" link by index but chose fill_secret instead of click).

    W_fill_secret_recovery_excludes_profile_menu (real bug, live-reproduced a THIRD time on
    the same site: "span 'Amber Floyd [Profile/Account Menu]'" this time — same account-
    dropdown-menu confusion again): the goal-keyword fix above did not close this case, because
    the goal itself never shared any vocabulary with the page's English nav labels — goal
    keyword extraction is ASCII-only (see _find_goal_matching_nav_element()'s regex), so a Thai
    goal (or any non-Latin-script goal) always returns None from goal-matching, falling through
    to blindly trust the model's own chosen index again with nothing left to stop it. The model
    can also target the SAME wrong profile-menu element turn after turn without self-correcting,
    exhausting the whole recovery budget on one repeated wrong guess. "[Profile/Account Menu]"
    (perception.py, W20/Task10) is a deterministic, non-site-specific marker — attached
    programmatically to whichever element genuinely opens the account/profile menu on ANY site,
    specifically so the model can find it for an intentional password-change flow. That is
    exactly the OPPOSITE of what this recovery path should ever target: it only fires once we
    already know this is NOT a password-change flow, so force-clicking the one element whose
    entire purpose is password-related account actions is actively counterproductive here, not
    merely unhelpful. Excluding it means a goal giving neither signal now falls through to None
    — the caller's existing generic go_back/scroll escape valve (_force_loop_recovery()'s own
    fallback for forced_cmd=None) takes over instead, which is bounded and terminates, unlike
    repeating the same wrong click indefinitely.

    W_fill_secret_recovery_excludes_already_active (real bug, live-reproduced on
    opensource-demo.orangehrmlive.com, goal "เปิดเว็บ แล้วไปที่หน้าadmin"): the agent had ALREADY
    ARRIVED — page was /web/admin/viewSystemUsers with the System Users list rendered — yet the
    task ran 5 steps, burned 227k input tokens and ended "Stopping task: fill_secret was rejected
    3 times in a row ... (a recovery action was forced but the loop persisted)". Activity Log
    showed the same forced click three times over: click(3), the sidebar "Admin" link, whose
    label was literally "Admin [already active]" (perception.py marks the current nav item, see
    W19 "Log Cleanliness" there). _find_goal_matching_nav_element() substring-matches that same
    label and had no reason to care about the marker, so goal-matching picked the link to the page
    we were standing on. Every forced click was therefore a guaranteed no-op: execute() returned
    success, the page never changed, the model saw an identical snapshot and re-proposed
    fill_secret — until _MAX_AUTHENTICATED_FILL_SECRET_FAST_RECOVERIES, then
    _MAX_FILL_SECRET_CONTEXT_REJECT_RETRIES, then _MAX_FORCED_LOOP_RECOVERIES were all drained and
    the whole task hard-failed on work it had actually completed.

    The main loop already refuses to click "[already active]" elements (see
    _MAX_ALREADY_ACTIVE_SKIP_RETRIES at its dispatch site), but forced recovery calls execute()
    directly and never passes through that check — so the exclusion has to be repeated here. Same
    reasoning as the profile-menu paragraph above and it is the stronger case of the two: an
    already-active nav element cannot possibly break the loop it was chosen to break, because
    clicking it is definitionally a no-op. Excluded on BOTH paths (goal-match and the model's own
    chosen index), since the model re-proposing the same already-active index every turn is
    exactly what was observed."""
    goal_match = _find_goal_matching_nav_element(elements, goal, skip_already_active=True)
    if goal_match is not None:
        return goal_match
    chosen_index = tool_input.get("index")
    chosen_element = next((e for e in elements if e.get("index") == chosen_index), None)
    chosen_label = str(chosen_element.get("label", "")) if chosen_element is not None else ""
    if (
        chosen_element is not None
        and chosen_element.get("tag") != "input"
        and "[Profile/Account Menu]" not in chosen_label
        and "[already active]" not in chosen_label
    ):
        return chosen_element
    return None


def _fill_secret_context_hint(elements: list[dict], tool_input: dict, goal: str) -> str:
    """คืนข้อความเสริม (เติมต่อท้าย _FILL_SECRET_NOT_PASSWORD_CONTEXT_NUDGE) ที่ชี้เป้าให้
    ตรงจุดที่สุดเท่าที่ heuristic แบบ deterministic ทำได้ — คืน "" ถ้าไม่มีอะไรให้ชี้ได้จริง
    (ไม่ throw ไม่ว่ากรณีใด) ดู comment เหนือฟังก์ชันนี้สำหรับ pattern ทั้ง 2 แบบที่เจอจริง

    W_fill_secret_recovery_excludes_already_active (pattern (3), เช็คก่อน pattern (2)/(1) —
    ดูเหตุการณ์จริงเต็มๆ ใน docstring ของ _fill_secret_recovery_target()): ถ้า nav element ที่
    ตรงกับ goal ถูกแปะ "[already active]" อยู่แล้ว แปลว่า "ไปถึงหน้าที่ goal ขอแล้ว" ไม่ใช่
    "ยังหาทางไปไม่เจอ" — hint ที่ถูกต้องคือบอกให้ปิดงาน (finish_task) ไม่ใช่ชี้ให้คลิกลิงก์
    เดิมซ้ำเหมือน pattern (1) (ซึ่งในเคสจริงนั้นคือคำแนะนำที่ผิดและช่วยเลี้ยง loop ไว้ด้วยซ้ำ)
    เช็คก่อน pattern (2) เพราะ (2) วินิจฉัยแค่ "action verb ผิด" ซึ่งเป็นข้อสังเกตที่เล็กกว่า
    และไม่ขัดกัน — ถ้าไปถึงหน้าปลายทางแล้วจริงๆ นั่นคือสิ่งที่โมเดลต้องรู้ก่อนเรื่องอื่น

    ยังเป็นแค่ "ข้อความ" เหมือน pattern อื่นทุกประการ — ไม่เปลี่ยน action ที่ dispatch จริง
    ไม่บังคับ finish_task ให้เอง และไม่รายงานสำเร็จแทนโมเดล (SYSTEM_PROMPT มีกฎนี้อยู่แล้ว —
    ดู W_goal_precheck ใน llm.py — แต่เคสจริงข้างบนพิสูจน์แล้วว่า prompt อย่างเดียวไม่พอ ซึ่ง
    เป็นเหตุผลประจำของไฟล์นี้ที่ต้องมี code หนุนกฎใน prompt อีกชั้น)"""
    already_active_goal_match = _find_goal_matching_nav_element(elements, goal)
    if already_active_goal_match is not None and "[already active]" in str(
        already_active_goal_match.get("label", "")
    ):
        return (
            f" Note: index {already_active_goal_match.get('index')} labeled "
            f"{already_active_goal_match.get('label')!r} is the navigation entry matching this "
            "goal, and it is already marked [already active] — you are ALREADY on the page the "
            "goal asked for, so there is nothing left to navigate to. Do not click it again and "
            "do not call fill_secret. Read what is on the page right now: if it satisfies the "
            "goal, call finish_task with that as your evidence; if some part genuinely remains, "
            "act on that remaining part specifically."
        )
    chosen_index = tool_input.get("index")
    chosen_element = next((e for e in elements if e.get("index") == chosen_index), None)
    # W_login_form_wrong_verb (pattern (4), บั๊กจริง live-reproduce บน saucedemo.com — คนละเว็บ
    # กับที่ pattern อื่นเจอมา จึงเป็นเรื่องทั่วไปไม่ใช่ของเว็บใดเว็บหนึ่ง): goal บอกรหัสผ่านมา
    # ตรงๆ ("login as standard_user with password secret_sauce") โมเดลเล็ง "ช่อง password ถูก
    # ตัวแล้ว" แต่ใช้ verb ผิดเป็น fill_secret (ซึ่งสงวนไว้ให้ฟอร์มเปลี่ยนรหัสผ่านเท่านั้น)
    # — guard ปฏิเสธถูกต้อง แต่ pattern (2) ด้านล่างข้าม element ที่เป็น <input> ทั้งหมด เลย
    # ไม่มี hint อะไรกลับไปเลยสักคำ โมเดลจึงเสนอ fill_secret ซ้ำจนครบโควตาแล้ว task ตายใน 2
    # step ทั้งที่ทางแก้ชัดเจนมาก: ใช้ "fill" ธรรมดากับ index เดิมนั่นแหละ
    if chosen_element is not None and chosen_element.get("tag") == "input":
        return (
            f" Index {chosen_index} labeled {chosen_element.get('label', '')!r} is an "
            "<input> on what looks like an ordinary LOGIN form, not a Change Password form. "
            "fill_secret is only for the Current Password field of a genuine change-password "
            f"flow — to log in, use action type 'fill' on this same index {chosen_index} with "
            "the password value taken from the goal or from earlier in this conversation. If "
            "the goal never provided one, call request_user_input to ask for it."
        )
    if chosen_element is not None and chosen_element.get("tag") != "input":
        return (
            f" Index {chosen_index} labeled {chosen_element.get('label', '')!r} is a "
            f"<{chosen_element.get('tag', '')}> element, not a password input — it can never "
            f"accept fill_secret no matter what. If that's genuinely the element the goal "
            f"needs, use action type 'click' on index {chosen_index} instead of fill_secret."
        )
    matching_nav_element = _find_goal_matching_nav_element(elements, goal)
    if matching_nav_element is not None:
        return (
            f" There is already a navigation link matching the goal right here: index "
            f"{matching_nav_element.get('index')} labeled {matching_nav_element.get('label')!r} "
            "— click that instead."
        )
    return ""


_FILL_SECRET_NOT_PASSWORD_CONTEXT_NUDGE = (
    "This action is rejected — fill_secret is only for the Current Password field on a "
    "genuine Change Password form, and nothing about the current goal/plan or this page "
    "indicates that's what's happening right now. If you're already authenticated and the "
    "goal doesn't ask to change a password, drop this approach entirely and take the action "
    "that actually matches the goal instead. If you do intend to change the password, first "
    "navigate to the real Change Password form (see the Account Security & Password Actions "
    "protocol) before calling fill_secret."
)

_MAX_FILL_SECRET_CONTEXT_REJECT_RETRIES = 2

# W_goal_precheck ("Already-Achieved Pre-check" escape valve): guard นี้มีโควตาเหมือนทุก
# pre-dispatch guard ในไฟล์นี้ — โมเดลที่อ่อนกว่าอาจเสนอ click เดิมซ้ำไปเรื่อยๆ โดยไม่ยอม
# เปลี่ยนใจ ถ้าบล็อกไม่มีที่สิ้นสุด loop จะไม่มีวันคืบหน้าเลย (เผา token ปฏิเสธข้อเสนอเดิมจน
# หมด max_steps โดยไม่มีอะไรถูกบันทึกลง history เพราะ record() เกิดตอน dispatch จริงเท่านั้น)
# — เกินโควตาแล้วปล่อยผ่านไป dispatch จริง ปลอดภัยกว่าเพราะคลิก element ที่ active อยู่แล้ว
# เป็น no-op ตามนิยาม
_MAX_ALREADY_ACTIVE_SKIP_RETRIES = 2


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
        # W_openai_oauth: auth ผ่าน OAuth token (ดู core/openai_oauth.py หัวไฟล์สำหรับ risk
        # disclosure เต็ม) ไม่ใช่ api key จาก settings เหมือน 3 provider ข้างบน —
        # build_openai_client() แค่สร้าง client shell (ไม่มี network call, ไม่มี token จริง
        # ตอนนี้) เพราะ _llm_backend() เป็น sync แต่การขอ/refresh OAuth token ต้อง await
        # httpx call ได้ — เลี่ยงการทำให้ _llm_backend() เป็น async (จะกระทบ call site อื่น
        # ในไฟล์นี้อีกหลายจุด) ด้วยการผลัก async work ลงไปใน llm.next_action_openai() แทน
        # (มันเป็น async def อยู่แล้ว ถูก await ทุก step โดย loop ข้างล่างอยู่แล้ว)
        if provider == "openai":
            return (
                llm.build_openai_client(),
                settings.openai_model,
                llm.next_action_openai,
                llm.append_tool_result_openai,
                # W_openai_oauth: OpenAI provider เก็บ user turn เป็น {"role": "user",
                # "content": "<str>"} เหมือน Anthropic เป๊ะ (ดู llm.next_action_openai —
                # ตั้งใจให้ shape นี้ตรงกัน) เลย reuse _compact_anthropic_messages ได้ตรงๆ
                # ไม่ต้องเขียน _compact_openai_messages แยกซ้ำโค้ดเดิม (function_call/
                # function_call_output item ใช้ key "type" ไม่ใช่ "role" เลยไม่ถูกเข้าใจผิด
                # ว่าเป็น user turn โดยไม่ตั้งใจ)
                _compact_anthropic_messages,
            )
        raise ValueError(f"ไม่รู้จัก LLM provider: {provider!r} (รองรับแค่ anthropic/gemini/groq/openai)")

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
        # W_plan_keeps_goal_verb: ร่างใหม่ *ครั้งเดียว* พร้อมบอกตรงๆ ว่าผิดตรงไหน — ถ้ารอบสอง
        # ยังผิดอีกก็ไม่ร่างซ้ำไปเรื่อยๆ ปล่อยแผนนั้นกลับไปให้ user เห็นบนหน้าจอยืนยันแผน
        # แล้ว run_task() จะเป็นคนหยุดถามเองก่อนเริ่มลงมือ (ดู guard ที่นั่น)
        if _plan_drops_goal_operation(goal, plan_text):
            print("⚠️ แผนที่ร่างมาไม่ตรงชนิดงานที่สั่ง (goal สั่งลบ แต่แผนไม่ลบ) — ร่างใหม่อีกครั้ง", flush=True)
            plan_text = await llm.generate_plan(
                client, model, f"{goal}\n\n{_PLAN_KEEPS_GOAL_VERB_CORRECTION}", page_text,
                resolved_provider, current_url=current_url_for_plan,
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
        nav_target_page_query: Optional[str] = None,
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
            # W_screenshot_never_throws: try เดิมครอบแค่ page.screenshot() ทั้งที่ docstring
            # ด้านบนสัญญาว่า "ไม่ throw เด็ดขาด" — base64.b64encode() และ _emit() อยู่นอก try
            # ทั้งคู่ ทำให้ค่าที่ screenshot คืนมาผิดชนิด หรือ subscriber ที่พังตอน emit สามารถ
            # ฆ่า task ทั้งตัวได้ ทั้งที่ live view เป็นแค่ของประดับ ไม่ใช่ส่วนหนึ่งของงาน
            # (พบจากเทสต์ 4 เคสที่ล้มค้างมานาน: mock คืน AsyncMock ให้ screenshot แล้ว
            # b64encode โยน TypeError ออกมาจนทั้ง run ตาย — บนของจริงเกิดได้เหมือนกันเวลา
            # page กำลังปิด/หน่วยความจำไม่พอ)
            try:
                raw = await page.screenshot(type="jpeg", quality=55)
                b64 = base64.b64encode(raw).decode("ascii")
                await _emit({
                    "kind": "screenshot", "step": step,
                    "image": f"data:image/jpeg;base64,{b64}",
                })
            except Exception:
                return

        async def _force_loop_recovery(
            reason: str, forced_cmd: Optional[dict] = None, forced_target: Optional[dict] = None,
        ) -> bool:
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
            ว่า action เดิมที่ agent ขอไปสำเร็จ

            forced_cmd (optional, W_state_guard_shortcut "Targeted Recovery"): ถ้า caller ระบุ
            command มาเอง (เช่น click index ที่ heuristic หาเจอว่าตรงกับ goal จริงๆ — ดู
            _fill_secret_recovery_target()) ใช้ command นั้นแทนการไล่ทีละตัวจาก
            _LOOP_RECOVERY_ACTIONS — go_back/scroll เป็นแค่ "หนีออกจาก state ที่ค้าง" ทั่วไป
            ไม่เคยพาไปใกล้ goal จริงเลย ถ้ารู้เป้าหมายที่ถูกต้องชัดเจนแล้วควรคลิกตรงนั้นแทน

            forced_target (optional): element dict ที่ forced_cmd ชี้ไป — ส่ง label/tag/type
            ต่อให้ execute()/classify_action() เหมือน dispatch ปกติทุกประการ กัน forced click
            หลุด risk-keyword/anchor-tag heuristic ไปเพราะไม่มี label (go_back/scroll ไม่มี
            element เป้าหมายอยู่แล้ว label="" เดิมจึงปลอดภัยสำหรับ 2 type นั้นเท่านั้น)"""
            nonlocal forced_recovery_count, messages, last_action_cmd, consecutive_repeat_count, steps_taken
            if forced_recovery_count >= _MAX_FORCED_LOOP_RECOVERIES:
                return False
            if forced_cmd is None:
                forced_cmd = _LOOP_RECOVERY_ACTIONS[min(forced_recovery_count, len(_LOOP_RECOVERY_ACTIONS) - 1)]
            forced_recovery_count += 1
            forced_result: ActionResult = await execute(
                page, forced_cmd, ask_user_func=ask_user_func,
                label=(forced_target.get("label", "") if forced_target else ""),
                manual_guidance="", allowed_domains=effective_allowed_domains,
                element_tag=(forced_target.get("tag", "") if forced_target else ""),
                element_type=(forced_target.get("type", "") if forced_target else ""),
            )
            steps_taken += 1
            self.memory.record({
                "step": steps_taken,
                "cmd": forced_cmd,
                "label": "[System forced — loop prevention]",
                "result": str(forced_result),
                "success": forced_result.success,
                "tokens": _tokens_dict(llm.TokenUsage()),
            })
            await _emit({
                "kind": "step", "step": steps_taken, "cmd": forced_cmd,
                "label": "[System forced — loop prevention]",
                "result": str(forced_result), "success": forced_result.success,
            })
            forced_text = (
                f"[The system detected a repeat loop: {reason} — so it automatically forced "
                f"{forced_cmd} instead of the action you just requested (attempt "
                f"{forced_recovery_count}/{_MAX_FORCED_LOOP_RECOVERIES}) result: "
                f"{forced_result}] Check the current URL and all indexed elements of the new "
                "page, then choose an action genuinely different from the one that was looping."
            )
            messages = append_tool_result(messages, tool_use_id, forced_text)
            # W29: route ผ่าน _cmd_for_repeat_comparison() เหมือนจุดอื่นเพื่อความสอดคล้องกัน
            # (forced_cmd ไม่มี completed_plan_step อยู่แล้วในทางปฏิบัติ — no-op จริงๆ แต่กัน
            # ไว้เผื่อ _LOOP_RECOVERY_ACTIONS เปลี่ยนแปลงในอนาคต)
            normalized_forced_cmd = _cmd_for_repeat_comparison(forced_cmd)
            last_action_cmd = normalized_forced_cmd
            consecutive_repeat_count = 1
            recent_actions.append(normalized_forced_cmd)
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
        # default-deny ทุกโดเมนอื่นนอกจาก target ของ task นี้เอง แม้ผู้เรียกลืมระบุ
        # allowed_domains มาเอง — กัน agent หลุดไปแตะ session อื่นที่ login ค้างไว้ในเครื่อง
        # เดียวกัน (เช่น mail) โดยไม่ตั้งใจ
        #
        # W_domain_guard_default (บั๊กจริง live-reproduce 3 รอบด้วย goal ของ user เอง): เดิม
        # default นี้ตั้งอยู่ "ข้างใน" branch ของ connect_to_user_browser เท่านั้น ส่วนเส้นทาง
        # BrowserPool (ที่ POST /tasks ใช้จริงทุก task และไม่เคยส่ง allowed_domains มาเลย)
        # ปล่อยค้างเป็น None — และ domain guard ท้ายลูปทำงานเฉพาะตอนไม่ใช่ None จึงตายสนิท
        # กับทุก task ที่ยิงผ่าน API ผลจริง: agent คลิกลิงก์โปรโมทบนหน้า login ของ
        # opensource-demo.orangehrmlive.com หลุดออกไป orangehrm.com (เว็บการตลาด คนละโดเมน)
        # แล้วเสีย step ที่เหลือทั้งหมดคลิก "Contact Sales"/"Start Your 30 Day Free Trial"/
        # "Asset Tracking" จนหมด max_steps — เกิดซ้ำ 3 รอบติดกัน
        #
        # ย้ายมาตั้งที่นี่ให้ครอบทุกเส้นทาง (pool/self-launched/CDP/session) — เจตนาเดิมที่
        # คอมเมนต์ด้านบนเขียนไว้ก็คือ "แม้ผู้เรียกลืมระบุ" อยู่แล้ว ไม่ได้ตั้งใจให้ CDP เท่านั้น
        # ผู้เรียกที่ต้องการข้ามโดเมนจริง (SSO/OAuth redirect) ยังส่ง allowed_domains เองได้
        # เหมือนเดิมทุกประการ ค่านี้แค่เป็น default ตอนไม่ได้ระบุ
        effective_allowed_domains = allowed_domains
        if effective_allowed_domains is None:
            # ถ้า url ผิดรูป/ว่าง extract_domain() คืน "" — อย่าตั้ง allowlist เป็น {""}
            # เพราะนั่นแปลว่า "บล็อกทุกโดเมนรวมทั้งของตัวเอง" ปล่อยเป็น None (ไม่มี guard)
            # เหมือนพฤติกรรมเดิมดีกว่า เคสนี้เกิดกับ task ที่ caller ส่ง page มาเองแล้ว
            # ไม่ได้ระบุ url
            _self_domain = extract_domain(url)
            if _self_domain:
                effective_allowed_domains = {_self_domain}
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
            await install_ssrf_guard(context)
            resolved_tab_reuse_policy = tab_reuse_policy or settings.user_browser_tab_reuse_policy
            page, opened_new_tab = await resolve_target_page(
                context, url, ask_user_func, resolved_tab_reuse_policy,
            )
        elif owns_browser:
            playwright = await async_playwright().start()
            browser = await _launch_chromium(
                playwright, headless=(True if defer_visible_window else is_headless), channel=browser_channel,
            )
            page = await browser.new_page()
            await install_ssrf_guard(page)
        else:
            context = await browser.new_context()
            await install_ssrf_guard(context)
            page = await context.new_page()
        page.on("dialog", _make_dialog_handler(self.memory, verbose))

        messages: list[dict] = []
        success = False
        final_message = _MAX_STEPS_EXHAUSTED_MESSAGE
        # W_step_budget: งบจริงของลูปคือ "จำนวนรอบ" (for _ in range(max_steps)) แต่ steps_taken
        # เพิ่มเฉพาะตอน dispatch action จริง — เส้นทาง guard ~14 จุดที่ `continue` กินรอบไปโดย
        # ไม่เพิ่ม steps_taken เลย สองตัวนี้จึงห่างกันเรื่อยๆ ระหว่าง task
        #
        # ต้องแยกให้ชัดเพราะ guard กัน premature finish_task(false) เอา steps_taken (ตัวนับ
        # action) ไปเทียบกับ max_steps (งบรอบ) — พอใกล้จบ เงื่อนไข "ยังเหลือ step ให้ลอง"
        # ยังเป็นจริงอยู่ทั้งที่รอบหมดแล้วจริง ผลคือ finish_task(false) ที่ถูกต้องถูกปฏิเสธ
        # แล้วลูปจบเองด้วยข้อความ default ด้านบน — คำอธิบายจริงของโมเดลว่าทำไมทำไม่ได้ถูกทิ้ง
        iterations_used = 0
        # ข้อความล่าสุดจาก finish_task(false) ที่ guard ปฏิเสธไป — ใช้แทนข้อความ default ถ้า
        # สุดท้ายลูปจบเพราะหมดรอบจริงๆ (โมเดลอธิบายไว้แล้วว่าติดอะไร ไม่มีเหตุผลให้ทิ้ง)
        last_rejected_finish_message = ""
        steps_taken = 0
        total_usage = llm.TokenUsage()
        # W_llm_call_count: จำนวน "เทิร์น" ที่ยิงไปหา LLM จริง — ไม่เท่ากับจำนวน step เพราะ
        # เทิร์นที่ถูก guard ปฏิเสธ/เทิร์นที่จบงาน ไม่ได้ลงมือทำ action จึงไม่มีบรรทัดใน
        # step_trace (หนึ่งครั้งนี้อาจรวม retry ภายใน next_action ด้วย — ดู W_notoolcall)
        llm_turns = 0
        # W_token_cut W1: แยกให้เห็นว่าเทิร์น LLM ถูกใช้ไปกับอะไร — ไป token_usage.jsonl
        # action_calls = browser_action ที่ dispatch จริง; finish_task_calls = ทุกครั้งที่
        # โมเดลเรียก finish_task (รับหรือไม่ก็นับ); guard_rejections = {ชื่อ guard: จำนวนครั้ง}
        # ที่เทิร์นถูกตีกลับโดยไม่ได้ลงมือ (W3 ใช้ตารางนี้ตัดสินว่าจะไปรวบเทิร์นตรงไหน)
        action_calls = 0
        finish_task_calls = 0
        guard_rejections: dict[str, int] = {}
        cache_hit_turns = 0
        cache_miss_turns = 0
        # W_prompt_audit: 1 entry ต่อการเรียก LLM — char count ของ request สุดท้ายแยกตาม
        # หมวด + input/cache_read/output token จริงของ call นั้น (แปลง char->token ตอน
        # วิเคราะห์โดยเทียบสัดส่วน ไม่เดา ratio ล่วงหน้า) ดู llm._char_payload_audit
        payload_audits: list[dict] = []
        # W_token_cut W5: การยุบ user turn ของ step เก่า (ดู _compact_stale_user_turns)
        history_compaction_events = 0
        history_chars_saved = 0
        _W5_CHARS_PER_TOKEN = 4.3  # calibrated จาก W_prompt_audit (4.23-4.36 คงที่)
        # W_token_cut W7: การยุบบล็อกกฎที่ gate ใน turn เก่า (ดู _dedupe_stale_gated)
        gated_deref_events = 0
        gated_chars_saved = 0

        def _record_payload_audit(_usage: "llm.TokenUsage") -> None:
            pc = getattr(_usage, "payload_chars", None)
            if not pc:
                return
            payload_audits.append({
                **pc,
                "_input_tokens": _usage.input_tokens,
                "_cache_read": _usage.cache_read_tokens,
                "_output_tokens": _usage.output_tokens,
            })

        def _bump_guard(_name: str) -> None:
            guard_rejections[_name] = guard_rejections.get(_name, 0) + 1

        # W_token_cut W3: finish_task ที่ถูก guard ตีกลับด้วยเหตุผลไหนไปแล้ว 1 ครั้ง — ครั้งที่
        # สองที่โมเดลกลับมาเรียก finish ด้วยสถานการณ์เดิม ไม่เตือนซ้ำ (เสียเทิร์น LLM ฟรี —
        # หลักฐาน c2 ของ baseline: nudge รอบสองไม่เปลี่ยนผล) ตกไปเส้นทาง "ยอมรับพร้อม tag
        # ความจริง" ที่ guard นั้นมีอยู่แล้วทันที — เฉพาะ guard ที่ไม่ใช่ safety ของข้อมูล
        finish_reject_reasons_seen: set[str] = set()
        finish_loop_prevented = 0

        def _first_guard_hit(_reason: str) -> bool:
            """W_token_cut W3: เรียก *หลัง* _bump_guard(_reason) แล้ว — 1 = ครั้งแรกของ
            เหตุผลนี้ (แนบ user-turn nudge เสริมได้), >1 = ซ้ำ (tool_result อย่างเดียวพอ
            ไม่ทบ history เปล่าๆ ทุกรอบ)"""
            return guard_rejections.get(_reason, 0) <= 1

        _AUDIT_BASE_CATS = (
            "system_prompt", "tool_schema", "page_snapshot", "action_history", "plan",
            "tool_result", "user_message", "gated_prompt", "other",
        )

        def _apportion_audit_tokens(_cat: str) -> int:
            """W_prompt_audit/W5: รวม token ของหมวดหนึ่งข้ามทุก call โดย apportion char
            ของหมวดนั้นเทียบ char รวมของ call แล้วคูณ _input_tokens จริงของ call"""
            total = 0.0
            for _c in payload_audits:
                _ct = sum(_c.get(k, 0) for k in _AUDIT_BASE_CATS) or 1
                total += _c.get(_cat, 0) / _ct * _c.get("_input_tokens", 0)
            return round(total)

        def _run_stats() -> dict:
            calls = llm_turns or 1
            tok = _tokens_dict(total_usage)
            return {
                "llm_calls": llm_turns,
                "action_calls": action_calls,
                "finish_task_calls": finish_task_calls,
                "guard_rejections": dict(guard_rejections),
                # W_token_cut W3 telemetry
                "guard_reason_counts": dict(guard_rejections),
                "repeated_guard_count": sum(max(0, v - 1) for v in guard_rejections.values()),
                "finish_loop_prevented": finish_loop_prevented,
                "notool_retries": total_usage.notool_retries,
                "cache_hit_turns": cache_hit_turns,
                "cache_miss_turns": cache_miss_turns,
                # W_prompt_audit calibration (2026-09-02): the provider's input_tokens IS the
                # full prompt count (cached portion included) — measured 4.23-4.36 chars/token
                # against input_tokens alone; adding cache_read double-counted ~1.5-2x. Use
                # input alone. cache_read reported separately below for the discount view.
                "avg_input_tokens_per_call": round(tok["input"] / calls, 1),
                "avg_cached_tokens_per_call": round(tok["cache_read"] / calls, 1),
                "avg_output_tokens_per_call": round(tok["output"] / calls, 1),
                # W_prompt_audit: char count ของทุก request แยกตามหมวด + token จริงต่อ call
                "payload_audit": list(payload_audits),
                # W_token_cut W5 telemetry
                "history_compaction_events": history_compaction_events,
                "history_chars_saved": history_chars_saved,
                "history_tokens_saved": round(history_chars_saved / _W5_CHARS_PER_TOKEN),
                # W_token_cut W7 telemetry
                "gated_deref_events": gated_deref_events,
                "gated_tokens_saved": round(gated_chars_saved / _W5_CHARS_PER_TOKEN),
                # assistant_history ที่โมเดลเห็นจริง (หลัง W5) — apportion char->token ต่อ
                # call จาก payload_audit; compacted = ค่านี้ (post), saved = ที่ตัดออกไป
                "assistant_history_tokens": _apportion_audit_tokens("other_assistant_history"),
                "assistant_history_compacted_tokens": _apportion_audit_tokens("other_assistant_history"),
            }

        premature_false_finish_count = 0
        premature_true_finish_count = 0
        premature_all_failed_count = 0
        premature_login_skip_count = 0
        premature_validation_error_count = 0
        premature_deletion_incomplete_count = 0
        premature_table_verify_count = 0
        premature_row_action_before_search_count = 0
        premature_destructive_before_filter_count = 0
        premature_count_answer_mismatch_count = 0
        # W_count_answer_check: {ค่าเงื่อนไข: จำนวนที่โค้ดนับได้} ที่สะสมมาจาก read_page_data
        # ของ task นี้ — ค่าใหม่ทับค่าเก่าเสมอ (ตารางเปลี่ยนได้ระหว่าง task เช่นหลังลบไปบางแถว
        # ตัวเลขล่าสุดจึงเป็นตัวที่จริงที่สุด)
        system_counted: dict[str, int] = {}
        premature_delete_all_unverified_count = 0
        # W_delete_all_intent: เงื่อนไขแบบ key=value ที่ user เขียนไว้ใน goal เอง (เช่น
        # "userrole=ess" -> ["ess"]) — ว่างเปล่า = goal ไม่ได้ระบุเงื่อนไขชัดเจน guard ทั้งชุด
        # นี้จะไม่ทำงานเลย (ไม่เดาเงื่อนไขเองจากภาษาธรรมชาติ)
        delete_all_condition_values = (
            _goal_condition_values(goal) if _is_delete_all_intent_goal(goal) else []
        )
        # W_column_aware_rows: guard ที่ต้องตัดสินว่า "แถวไหนตรงเงื่อนไข" ต้องรู้ทั้งชื่อคอลัมน์
        # และค่า ไม่ใช่ค่าอย่างเดียว — ส่วน *_values ด้านบนเหลือไว้ใช้กับข้อความรายงานเท่านั้น
        delete_all_condition_pairs = (
            _goal_condition_pairs(goal) if _is_delete_all_intent_goal(goal) else []
        )
        # W_delete_all_intent: sticky ต่อ task — True หลังยืนยันแล้วครั้งแรกว่าตารางที่เห็น
        # กรองตรงเงื่อนไขจริง (หรือหลังหมดโควตา nudge) ไม่ต้องอ่าน DOM ซ้ำทุกครั้งที่ลบแถวถัดไป
        destructive_filter_verified = False
        # W_delete_all_intent: sticky ต่อ task ต่างจาก filter_dirty_since_search ที่ scope แค่
        # 1 step — True ตั้งแต่มีการเปลี่ยนค่า filter แล้วยังไม่เคยกด Search ตามหลังเลย ใช้
        # ปฏิเสธ finish_task(success=true) ของงานลบทั้งหมด (trace ยืนยันว่า run ที่ claim
        # สำเร็จผิดๆ ไม่เคยกด Search เลยสักครั้งตลอด run)
        filter_changed_without_search = False
        # W_resume: จำนวนครั้งที่เรียก request_user_input ไปแล้วใน task นี้ (ดู
        # _MAX_REQUEST_USER_INPUT_CALLS ด้านบนสุดของไฟล์)
        request_user_input_count = 0
        # W64[7.1]: True เฉพาะช่วง "1 step ถัดไปทันที" หลัง fill/select ที่สำเร็จ — reset เป็น
        # False หลัง action ถัดไปเสมอไม่ว่าจะเป็น action อะไร (ดู docstring ของ
        # _ROW_ACTION_LABEL_RE ด้านบนสุดของไฟล์สำหรับเหตุผลเต็มว่าทำไม scope แคบแค่ 1 step)
        filter_dirty_since_search = False
        # W64[7.2]: True ตั้งแต่ครั้งแรกที่ action ใดๆ ใน task นี้มี toast_confirmed=True (ดู
        # actions.py::ActionResult) — ใช้ตัดสินว่า table-verify guard (ดู
        # _scan_created_item_in_table ด้านล่าง) ควร "ผ่อนปรน" แค่ไหนตอนหา verify_text ไม่เจอ
        # ในตารางแม้ retry ครบแล้ว (ดู docstring เหนือจุดใช้งานจริงสำหรับเหตุผลเต็ม)
        any_toast_confirmed_this_task = False
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
        # W_token_trim (P3/M3): the site manual is constant per task — send it in full once
        # (first step + first step after every compaction), then reference it by a stable
        # id + summary. site_manual_full/_ref are "" when there is no manual, so the whole
        # scheme collapses to "pass '' every step" exactly as before.
        site_manual_full, site_manual_ref = llm.site_manual_blocks(
            site_manual_context, extract_domain(url),
        )
        site_manual_full_sent = False
        force_full_site_manual = False
        plan_text: Optional[str] = None
        # W10[F]: goal ที่ next_action() เห็นจริงทุก step — ปกติเท่ากับ goal เดิมเป๊ะ แต่ถ้า
        # confirm_plan=True จะถูกผนวกด้วยแผน (ที่อาจถูก user แก้ไขก่อน confirm) เข้าไปด้วย
        # หลัง plan ผ่านการยืนยันแล้ว (ดูด้านล่าง) — แยกจาก goal ตัวเดิมเพราะ goal ยังต้อง
        # ใช้แบบดิบๆ ต่อ (RAG query, long-term memory query, log) ไม่อยากให้ข้อความแผนที่
        # อาจยาวมากปนเข้าไปทำให้ query เพี้ยน
        effective_goal = goal
        # W41: wall-clock เวลาที่ next_action() ครั้งก่อนหน้า "จบ" (คืนค่ามาแล้ว) — None
        # ตอนยังไม่เคยเรียกเลย (ครั้งแรกไม่ต้องรอ pacing delay อะไรทั้งนั้น) ใช้คำนวณว่ายัง
        # ต้องหน่วงอีกแค่ไหนให้ครบ settings.step_pacing_delay_seconds ก่อนเรียกครั้งถัดไป
        # (ดู comment เต็มด้านบนสุดของไฟล์ + config.py::step_pacing_delay_seconds)
        last_llm_call_at: Optional[float] = None
        last_action_cmd: Optional[dict] = None
        consecutive_repeat_count = 0
        # W_same_label_loop: นับซ้ำด้วย (type, label) แทน (type, index) — ดู docstring ของ
        # _MAX_CONSECUTIVE_SAME_LABEL_ACTIONS ด้านบนสุดของไฟล์
        last_same_label_key: Optional[tuple] = None
        consecutive_same_label_count = 0
        # W_index_drift_measure: นับว่า element ที่ index ชี้เปลี่ยนไป/หายไปกี่ครั้งต่อ task
        index_drift_changed = 0
        index_drift_gone = 0
        # W_session_drift: จำนวนครั้งที่ login ใหม่ให้กลางทาง (ดู guard ต้นลูป)
        mid_task_relogin_count = 0
        # W_fill_secret_hardening: จำนวนครั้งติดกันที่ fill_secret ถูกปฏิเสธเพราะไม่ใช่
        # change-password context (รีเซ็ตทันทีที่ dispatch อย่างอื่นผ่าน — ดู guard ในลูป)
        consecutive_fill_secret_context_reject_count = 0
        secret_refill_reject_count = 0
        # W_retry_value_has_no_home: label ของช่องที่ agent กรอกค่าเองในงานนี้ ตามลำดับ
        # ที่กรอก — ใช้บอก user ตอนขอค่าใหม่ว่าจะเอาไปแทนที่ช่องไหน (label ไม่ใช่ index
        # เพราะ index ถูกแจกใหม่ทุก snapshot จึงข้ามเทิร์นไม่ได้)
        agent_filled_field_labels: list[str] = []
        # ว่าง = ไม่ได้กำลังรอค่าใหม่จาก user
        retry_value_field_labels: list[str] = []
        # W_auto_login_outcome_is_invisible: ค่าเริ่มต้นสำหรับ path ที่ไม่เคยเรียก auto-login
        auto_login_outcome = "skipped"
        # W_verify_text_needs_a_write: งานนี้เคยเขียนค่าลงฟอร์มจริงไหม
        wrote_a_value_this_task = False
        goal_wants_a_record_change = (
            canonical_intent(goal).operation in _TABLE_VERIFY_RELEVANT_OPERATIONS
        )
        empty_password_submit_count = 0
        # W_goal_precheck: จำนวนครั้งติดกันที่ข้ามการคลิก element ที่มี marker "[already active]"
        consecutive_already_active_skip_count = 0
        # W_goal_scope: sticky ทั้งคู่ — ไม่ reset กลับ False อีกเลยตลอด task ("แผนเสร็จครบแล้ว"/
        # "ถึงหน้าเป้าหมายแล้ว" เป็นความจริงถาวร ไม่ใช่สถานะชั่วคราวของ step เดียว) โดยเฉพาะตัว
        # nav: ถ้าคำนวณใหม่ทุก step จะพังทันทีที่ action ถัดไปเป็น read-only เพราะ
        # self.memory.recent(1) จะกลายเป็น read_page_data ไม่ใช่ click/goto อีกต่อไป ทำให้
        # _navigation_target_reached() คืน False ทั้งที่ยังอยู่หน้าเป้าหมายอยู่
        plan_fully_completed = False
        # W_plan_step_cursor: "ตอนนี้อยู่ข้อไหนของแผน" ต้องเป็นของโค้ด ไม่ใช่ตัวเลขที่โมเดล
        # ใส่มาเอง — completed_plan_step เป็น self-report ล้วนๆ มาตลอด ไม่มีตัวนับฝั่งโค้ด
        # เลยสักตัว โมเดลจึงรายงานข้อสุดท้ายมาเป็นค่าแรกได้ แล้ว plan_fully_completed ติดทันที
        plan_cursor = 1
        # W_plan_panel_lags_the_log: cursor "สำหรับแสดงผลเท่านั้น" — ตัดสินจากหลักฐานบน
        # หน้าเว็บ (URL/action) เพื่อให้ PLAN panel เดินทันกับ LOG
        # ค่านี้ *ไม่เคย* ป้อน plan_fully_completed, ไม่เคยเข้า prompt, และไม่มีวันเกิน
        # จำนวนข้อของแผน — ข้อสุดท้ายจึงยังต้องรอ task สำเร็จจริงถึงจะติ๊ก (ดู allDone ฝั่ง
        # frontend) โครงสร้างนี้ทำให้มันแปลว่า "แผนจบแล้ว" ไม่ได้เลยแม้จะเดินไปสุดทาง
        display_plan_cursor = 1
        # W_plan_cursor_needs_a_matching_action: กี่ครั้งติดกันแล้วที่ไม่ยอมให้ cursor เดินหน้า
        # เพราะ action ดู "ไม่ตรง" กับ step ปัจจุบัน — ต้องมีเพดาน ไม่งั้นล็อกตาย (ดูจุดใช้งาน)
        blocked_cursor_advances = 0
        # W_plan_progress_stall: กี่ action ที่สำเร็จแล้วผ่านไปโดย plan_cursor ไม่ขยับเลย
        actions_since_plan_progress = 0
        plan_stall_notes_sent = 0
        # W_action_matches_plan_step: นับแยกจากตัวบน และ *ไม่* รีเซ็ตเมื่อ cursor ขยับ
        actions_mismatching_plan_step = 0
        plan_mismatch_notes_sent = 0
        # W_filter_scope_guard: field ที่ goal อนุญาตให้กรอง (ว่าง = ไม่เปิด guard นี้เลย)
        goal_filter_fields = _goal_condition_fields(goal)
        # W_filter_already_satisfied: คู่ field=value เต็มๆ (ไม่ใช่แค่ชื่อ field) — ใช้ตัวเดียว
        # กับที่ _goal_condition_fields() อ่านมา ไม่เรียก regex ซ้ำทุก step
        goal_condition_pairs = _goal_condition_pairs(goal)
        filter_satisfied_reject_count = 0
        filter_scope_reject_count = 0
        prefer_row_delete_reject_count = 0
        profile_menu_reject_count = 0
        obscured_click_reject_count = 0
        # W_prefer_row_delete: goal นี้เกี่ยวกับบัญชีของผู้ใช้เองหรือเปล่า (คำนวณครั้งเดียว)
        goal_is_about_account = _goal_is_about_the_signed_in_account(goal)
        nav_target_reached_confirmed = False
        # W_goal_scope: จำนวนครั้งติดกันที่ action ถูกปฏิเสธเพราะ goal ถือว่าสำเร็จแล้ว — เงื่อนไข
        # reset ไม่เหมือน counter อื่นในไฟล์นี้ (ดู comment ตรงจุดใช้งานจริง)
        consecutive_goal_scope_reject_count = 0
        # W21 ("Batch/Bulk Action Protocol"): ผลลัพธ์ (success/fail) ของ action ล่าสุดที่
        # เพิ่ง execute() จริงไปเมื่อ step ก่อนหน้า — ใช้คู่กับ _BULK_SAFE_REPEAT_TYPES
        # ด้านล่างตอนเช็ค consecutive_repeat_count (ดู docstring ตรงจุดเช็คด้านล่าง)
        last_action_succeeded: Optional[bool] = None
        # W64[7.1]: label ของ field ล่าสุดที่ fill/select สำเร็จ (ตอนที่ filter_dirty_since_
        # search เพิ่งถูกตั้งเป็น True) — ใช้แสดงใน nudge message เท่านั้น (ดู
        # _PREMATURE_ROW_ACTION_BEFORE_SEARCH_NUDGE_TEMPLATE ด้านบนสุดของไฟล์)
        last_filter_field_label = ""
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
        _CONTEXT_UNCHANGED_NOTE = "(identical to the previous step — the page has not changed)"

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
            # W_consent_banner: ปิดแบนเนอร์คุกกี้ก่อนเสมอ — ถ้ามันบังอยู่ _maybe_auto_login()
            # จะหาฟอร์ม login ไม่เจอแล้วไม่ล็อกอินให้ ทั้งที่มี credential เก็บไว้จริง
            await _dismiss_consent_banner(page, verbose)
            _auto_login_box: dict = {}
            auto_login_failure_reason = await _maybe_auto_login(page, verbose, _auto_login_box)
            auto_login_outcome = _auto_login_box.get("result", "skipped")
            # W_consent_banner (รอบสอง): CMP หลายเจ้าโหลด script แบบ async แล้ว render
            # แบนเนอร์ "หลัง" wait_stable คืนค่าไปแล้ว — ยิงซ้ำอีกรอบตรงนี้จึงจับเคสนั้นได้
            # (ยืนยันจาก live run: เรียกครั้งเดียวก่อน auto-login คืน None เพราะยังไม่มีแบนเนอร์
            # แต่ step ถัดๆ มาโมเดลเห็นปุ่มของแบนเนอร์เต็มหน้าไปแล้ว) — no-op ถ้าไม่มีอะไรให้ปิด
            await _dismiss_consent_banner(page, verbose)
            if auto_login_failure_reason:
                await _emit({
                    "kind": "auto_login_failed",
                    "message": "ล็อกอินไม่สำเร็จด้วย credential ที่บันทึกไว้สำหรับเว็บนี้",
                    "reason": auto_login_failure_reason,
                })

            # W66[C] ("Fast-Path Navigation", manual trigger — opt-in เท่านั้น ค่า default
            # None = พฤติกรรมเดิมทุกประการ ไม่กระทบ caller เดิมที่ไม่รู้จัก parameter นี้เลย):
            # ลองเดินตาม nav path ที่เรียนรู้ไว้ (site_learning/) ไปหน้าเป้าหมายก่อนเข้า loop
            # ปกติ — วางไว้หลัง auto-login เสมอ (ต้อง login ให้เสร็จก่อนถึงจะเห็นเมนู Admin/
            # หน้าที่ต้อง auth) ไม่มี manual/หาไม่เจอ/replay ล้มเหลว -> fallback เงียบๆ กลับไป
            # เริ่มจาก url เดิม (goto ซ้ำ) แล้วปล่อยให้ loop ปกติด้านล่างทำงานเหมือนไม่เคยระบุ
            # nav_target_page_query เลย (ปลอดภัย ไม่แย่กว่าเดิม — pattern เดียวกับ fastpath
            # เดิมทั้งไฟล์) *** lazy import กัน circular import (site_learning/__init__.py ->
            # crawler.py -> orchestrator.py) — pattern เดียวกับ _maybe_auto_login() ***
            #
            # W67[D] (auto-decide): ถ้าไม่ได้ระบุ nav_target_page_query มาเอง (explicit
            # trigger) แต่เปิด enable_nav_fastpath_auto_decide ไว้ (default True) ให้ใช้
            # goal ตรงๆ เป็น query แทน — auto-decide ต้องมั่นใจกว่า explicit trigger (ที่
            # user/caller ตั้งใจระบุมาเองแล้วเชื่อได้เต็มที่ ใช้ threshold หลวมเดิม=1) เพราะ
            # จะลงมือคลิกจริงตามผล match โดยไม่มีใครยืนยันอีกชั้น จึงใช้
            # nav_fastpath_min_match_score (default 2) ที่เข้มกว่า
            effective_nav_query = nav_target_page_query
            nav_min_score = 1
            if effective_nav_query is None and settings.enable_nav_fastpath_auto_decide:
                effective_nav_query = goal
                nav_min_score = settings.nav_fastpath_min_match_score
            if effective_nav_query:
                nav_manual = None
                nav_target_page = None
                try:
                    from backend.app.site_learning import storage as site_storage
                    nav_domain = extract_domain(page.url)
                    nav_manual = site_storage.load_manual(nav_domain)
                    if nav_manual is not None:
                        nav_target_page = site_storage.find_matching_page(
                            nav_manual, effective_nav_query, min_score=nav_min_score
                        )
                except Exception:
                    nav_manual, nav_target_page = None, None
                if nav_manual is not None and nav_target_page is not None:
                    nav_result = await fastpath_executor.execute_navigation(
                        page, goal, nav_manual, nav_target_page, client, model, resolved_provider,
                        ask_user_func=ask_user_func, on_event=_emit,
                    )
                    if verbose:
                        print(
                            f"[nav-fastpath] target={nav_target_page.name!r} "
                            f"success={nav_result.get('success')} message={nav_result.get('message')}",
                            flush=True,
                        )
                    if nav_result.get("success"):
                        await wait_stable(page)
                    else:
                        # replay ล้มเหลว/ไม่มี path ให้ใช้ — กลับไปจุดเริ่มต้นเดิมให้แน่ใจ
                        # (page อาจค้างอยู่กลางทางถ้า repair/escalate ยังไม่จบดี)
                        await goto(page, url)
                        await wait_stable(page)

            # Intent Classification: ตรวจจับ Intent ของผู้ใช้ก่อนเริ่ม Planner Loop
            initial_elements, initial_page_text = await get_snapshot(page)
            # Speed 2.1: ในเส้นทางปกติ (ไม่มี confirm_plan) ไม่มีอะไรเปลี่ยนหน้าเว็บระหว่าง
            # snapshot นี้กับ snapshot แรกของ loop หลักด้านล่าง (auto-login + wait_stable()
            # รันเสร็จไปแล้วตั้งแต่ก่อนบรรทัดนี้) — เก็บไว้ใช้ซ้ำแทนที่จะ get_snapshot() อีก
            # รอบซ้ำซ้อนตอนเข้า loop ครั้งแรก invalidate (set เป็น None) เฉพาะ path
            # confirm_plan ด้านล่างที่ page state เปลี่ยนแน่นอน (รอ user ตอบไม่จำกัดเวลา +
            # อาจ relaunch browser ใหม่ทั้งหมดถ้า defer_visible_window)
            cached_elements, cached_page_text = initial_elements, initial_page_text
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
                # W_count_answer_check: mini-loop นี้คือที่ที่คำถามเชิงนับเกือบทั้งหมดไปจบจริง
                # (classify_intent ส่ง "มี ... กี่คน" มาที่ qa_summary) และเป็นที่ที่บั๊กที่ user
                # เจอสดเกิดขึ้น — ตารางมี 7 แถวที่เป็น ESS แต่ agent ตอบ 6 โดยไม่มี guard ตัวไหน
                # จับได้เลย เพราะทุก guard ที่มีตรวจ "ทำ action สำเร็จไหม" ไม่ใช่ "ตัวเลขในคำตอบ
                # ถูกไหม" (main loop มี guard เดียวกันนี้ด้วย ใช้ helper ตัวเดียวกัน)
                qa_system_counted: dict[str, int] = {}
                qa_count_mismatch_retries = 0
                for _ in range(_QA_SUMMARY_MAX_STEPS):
                    qa_tool_name, qa_tool_input, qa_tool_use_id, qa_messages, qa_usage = await next_action(
                        client, model, qa_goal, qa_page_text, qa_messages, plan_context="",
                        # W_fill_secret_schema_gate: qa_summary เป็น mini-loop แบบอ่านอย่างเดียว
                        # (อนุญาตแค่ read_page_data/fill-ค้นหา/click อยู่แล้ว ดูด้านล่าง) —
                        # fill_secret ไม่มีวันถูกต้องที่นี่ ตัดออกจาก schema ไปเลย
                        allow_fill_secret=False,
                        # W_token_cut W2: qa ไม่ได้วน _resolve_prompt_sections — ส่งบล็อก gate
                        # ครบเหมือนที่เคยได้จาก build_system_prompt(None) เดิม (table สำคัญกับ
                        # คำถามเชิงนับ/สรุปตาราง) แค่ย้ายไปอยู่ท้าย user turn
                        prompt_sections=llm.ALL_PROMPT_SECTIONS,
                    )
                    total_usage += qa_usage
                    llm_turns += 1
                    _record_payload_audit(qa_usage)  # W_prompt_audit
                    if qa_usage.cache_read_tokens > 0:  # W_token_cut W1
                        cache_hit_turns += 1
                    else:
                        cache_miss_turns += 1
                    if qa_tool_name == "finish_task":
                        finish_task_calls += 1
                        qa_answer = qa_tool_input.get("message", "")
                        # W_count_answer_check: คำตอบต้องมีตัวเลขที่โค้ดนับไว้จริง ไม่งั้นตีกลับ
                        # ให้ตอบใหม่ (มีโควตา escape valve เหมือน guard อื่นในไฟล์นี้ — ตอบ
                        # คำถามผิดยังดีกว่าไม่ได้คำตอบเลย ถ้าโมเดลยืนยันซ้ำๆ)
                        qa_contradiction = _count_answer_contradiction(
                            goal, qa_answer, qa_system_counted,
                        )
                        if (
                            qa_contradiction is not None
                            and qa_count_mismatch_retries < _MAX_COUNT_ANSWER_MISMATCH_RETRIES
                        ):
                            qa_count_mismatch_retries += 1
                            qa_value, qa_count = qa_contradiction
                            if verbose:
                                print(
                                    f"[qa: คำตอบขัดกับตัวเลขที่โค้ดนับ "
                                    f"{qa_count_mismatch_retries}/"
                                    f"{_MAX_COUNT_ANSWER_MISMATCH_RETRIES}] "
                                    f"{qa_value!r} นับได้ {qa_count} แต่ไม่มีในคำตอบ",
                                    flush=True,
                                )
                            qa_messages = append_tool_result(
                                qa_messages, qa_tool_use_id,
                                _COUNT_ANSWER_MISMATCH_NUDGE_TEMPLATE.format(
                                    count=qa_count, value=qa_value,
                                ),
                            )
                            continue
                        summary_text = qa_answer
                        break

                    qa_action_type = qa_tool_input.get("type")
                    if qa_action_type == "read_page_data":
                        action_calls += 1  # W_token_cut W1
                        qa_result: ActionResult = await execute(
                            page, qa_tool_input, ask_user_func=ask_user_func, label="",
                            manual_guidance="", allowed_domains=effective_allowed_domains,
                        )
                        if qa_result.success:
                            qa_system_counted.update(system_counted_conditions(qa_result.message))
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
                        action_calls += 1  # W_token_cut W1
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
                    **_run_stats(),
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
                # Speed 2.1: page state เปลี่ยนแน่นอนหลังจากนี้ (รอ user ตอบ confirm ไม่
                # จำกัดเวลา + อาจ relaunch browser ทั้งหมดถ้า defer_visible_window ด้านล่าง)
                # — invalidate cache บังคับให้ loop หลัก get_snapshot() ใหม่เสมอ
                cached_elements = cached_page_text = None
                _, plan_page_text = await get_snapshot(page)
                plan_text = await llm.generate_plan(client, model, goal, plan_page_text, resolved_provider)
                # W_plan_keeps_goal_verb: เหมือนเส้นทาง API ด้านบน — ร่างใหม่ครั้งเดียว
                if _plan_drops_goal_operation(goal, plan_text):
                    if verbose:
                        print("[plan] ไม่ตรงชนิดงานที่สั่ง — ร่างใหม่อีกครั้ง", flush=True)
                    plan_text = await llm.generate_plan(
                        client, model, f"{goal}\n\n{_PLAN_KEEPS_GOAL_VERB_CORRECTION}",
                        plan_page_text, resolved_provider,
                    )
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
                        **_run_stats(),
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
                    await install_ssrf_guard(page)
                    page.on("dialog", _make_dialog_handler(self.memory, verbose))
                    await goto(page, url)
                    await wait_stable(page)

            # W_goal_scope: resolve ครั้งเดียว ไม่ใช่ทุก iteration — คำนวณจากตัว goal ล้วนๆ
            # W_plan_keeps_goal_verb: จุดนี้ plan_text นิ่งแล้วทั้งเส้นทาง approved_plan (user
            # กดยืนยันบนหน้าจอ อาจแก้ข้อความมาเองด้วย) และเส้นทาง confirm_plan ในลูป
            #
            # W_plan_warn_not_abort: เดิมจุดนี้ `return {"steps": 0}` หยุดทั้ง task ทันที —
            # แต่กว่าจะมาถึงตรงนี้ plan_text มาจาก approved_plan หรือ _confirm_plan() ซึ่ง
            # **ทั้งสองทางคือแผนที่ user ยืนยันมาแล้ว** (และอาจแก้ข้อความเองด้วยซ้ำ) การหยุดจึง
            # เท่ากับตัดสินใจแทน user บนสิ่งที่เขาเพิ่งอ่านและกดยืนยันไปเอง
            # ตามที่ user เลือกไว้: เตือนให้ดังตั้งแต่เทิร์นแรก แต่ไม่หยุด
            #
            # ความปลอดภัยไม่ได้หายไปไหน — guard ตอน execution ยังบล็อกการกด Save จริงอยู่ครบ
            # (W_no_record_edit_for_delete_goal เป็น hard reject ไม่มีโควตา + W_prefer_row_delete)
            # ที่เปลี่ยนคือ "ไม่ตัดสินใจแทน user ตั้งแต่ยังไม่เริ่ม" ไม่ใช่ "ปล่อยให้เขียนทับข้อมูล"
            plan_mismatch_reason = (
                _plan_drops_goal_operation(goal, plan_text) if plan_text else None
            )
            if plan_mismatch_reason:
                # ต่อท้าย effective_goal ไม่ใช่ยัดเข้า messages: ตรงนี้ messages ยังว่างอยู่
                # (ประกาศไว้ต้น run_task) การใส่ข้อความแรกเป็น nudge จะได้ user turn สองอัน
                # ติดกันก่อน goal ซึ่ง provider บางเจ้าไม่รับ — และการอยู่ใน goal ทำให้คำเตือน
                # ติดไปกับ *ทุก* step ไม่ใช่หายไปหลังเทิร์นแรก
                print(f"\u26a0\ufe0f [plan] {plan_mismatch_reason}", flush=True)
                effective_goal = (
                    f"{effective_goal}\n\n"
                    + _PLAN_MISMATCH_WARNING_TEMPLATE.format(reason=plan_mismatch_reason)
                )

            # ไม่ข้ามแม้จะมี confirmed plan อยู่ (plan_fully_completed อาจไม่มีวันเป็น True ถ้า
            # planner เติมบรรทัดสุดท้ายเป็นการ "ยืนยันผล" ที่ไม่มี action ไหนทำให้เสร็จได้ —
            # goal_nav_target จึงต้องเป็น signal สำรองที่ใช้ได้เสมอ)
            # W_prompt_sections: สะสมข้าม step ของ task นี้ (ดู _resolve_prompt_sections)
            prompt_sections: frozenset = frozenset()
            # W_captcha_detect: ถามคนเรื่อง CAPTCHA แค่ครั้งเดียวต่อ task
            captcha_reported = False
            goal_nav_target = _extract_goal_navigation_target(goal)
            # W_no_create_for_existing_goal: resolve ครั้งเดียวเหมือน goal_nav_target
            goal_targets_existing_only = _goal_targets_existing_records_only(goal)
            # W_no_credential_flow: resolve ครั้งเดียวเหมือนกัน (goal ไม่เปลี่ยนกลาง task)
            goal_mentions_credentials = _goal_mentions_credentials(goal)
            # W_no_record_edit_for_delete_goal: resolve ครั้งเดียวเหมือนกัน
            goal_is_deletion_only = _goal_is_deletion_only(goal)

            # W_plan_panel_lags_the_log: ยิงความคืบหน้าครั้งแรกก่อนเข้าลูป เพื่อให้ spinner ไป
            # อยู่ข้อ 1 ตั้งแต่ต้น ไม่ต้องรอ action แรกจบก่อน (แผนถูกยืนยันเสร็จแล้วตรงนี้)
            if plan_text:
                await _emit({
                    "kind": "plan_progress", "current": 1, "done_through": 0,
                    "total": _total_plan_steps(plan_text), "confirmed": 0,
                    "evidence": "start", "url": page.url,
                })

            for _ in range(max_steps):
                # W_step_budget: นับ "รอบ" แยกจาก steps_taken (ดูคำอธิบายที่จุดประกาศตัวแปร)
                iterations_used += 1

                # W_token_cut W3: backstop ลูปแก้ตัวไม่รู้จบ — guard ปฏิเสธ action ของโมเดล
                # ซ้ำเกินเพดาน (รวมทุกเหตุผล หรือเหตุผลเดียวข้ามรอบ reset) โดยไม่คืบหน้า =
                # จบ task ตามความจริง แทนการเผา step budget ที่เหลือทั้งหมด (เช็คที่หัวลูป
                # จุดเดียวที่ break ได้โดยไม่มี tool_use ค้างไม่มี tool_result ตอบ)
                if guard_rejections and (
                    sum(guard_rejections.values()) >= _MAX_TASK_GUARD_REJECTIONS
                    or max(guard_rejections.values()) >= _MAX_SAME_GUARD_REASON_REJECTIONS
                ):
                    _worst = max(guard_rejections, key=lambda k: guard_rejections[k])
                    success = False
                    final_message = (
                        f"หยุด task: ระบบปฏิเสธ action ของโมเดลซ้ำหลายครั้งโดยไม่คืบหน้า "
                        f"(guard '{_worst}' x{guard_rejections[_worst]}, รวมทุกเหตุผล "
                        f"{sum(guard_rejections.values())} ครั้ง) — โมเดลหาวิธีทำงานที่ผ่าน"
                        f"ข้อจำกัดไม่ได้"
                    )
                    completion_verification = "EXECUTION_FAILED_NEEDS_REPAIR"
                    if verbose:
                        print(f"[W3 guard-loop backstop] {final_message}", flush=True)
                    break

                # W_login_check_once (P4.7): _login_form_needs_password() ถูกเรียก 2 ครั้งต่อ
                # รอบ (guard session-drift ก่อนเรียก LLM + guard login-form ก่อน dispatch) และ
                # แต่ละครั้งรอได้ถึง _DOM_CHECK_TIMEOUT_MS ต่อช่อง password ที่มองเห็น
                #
                # ระหว่างสองจุดนั้นมีแค่การเรียก LLM คั่น ไม่มี action ใดแตะหน้าเว็บเลย DOM จึง
                # เหมือนเดิมแน่นอน — อ่านซ้ำได้คำตอบเดิมเสมอ cache ต่อรอบจึงปลอดภัย (ไม่ cache
                # ข้ามรอบเด็ดขาด: หลัง execute() หน้าเปลี่ยนได้ตลอด)
                login_form_state: Optional[bool] = None

                async def _login_form_needs_password_cached() -> bool:
                    nonlocal login_form_state
                    if login_form_state is None:
                        login_form_state = await _login_form_needs_password(page)
                    return login_form_state
                # Speed 2.1: รอบแรกของ loop (steps_taken ยังเป็น 0) ใช้ snapshot ที่ cache
                # ไว้ตอน intent classification แทน get_snapshot() ซ้ำ ถ้ายังไม่ถูก invalidate
                # (ดู comment ตอนตั้งค่า cached_elements/cached_page_text ด้านบน) — รอบถัดๆ
                # ไปยังคง get_snapshot() ใหม่ทุกครั้งเหมือนเดิมทุกประการ

                # W_goal_scope: หลักฐานว่า goal สำเร็จแล้ว ณ ตอนนี้ — เช็คก่อน next_action() ทุก
                # step (ใช้บังคับ gate ด้านล่าง หลัง finish_task block) page.url เป็นสถานะจริง
                # ล่าสุดของ browser เสมอ ไม่ต้องรอ get_snapshot() ของรอบนี้ — steps_taken > 0
                # กันไม่ให้ lock ตั้งแต่ step แรกสุดที่ยังไม่ได้ทำอะไรเลย
                goal_scope_satisfied_reason: Optional[str] = None
                if steps_taken > 0:
                    # W_plan_cursor_not_proof (บั๊กจริง live run 2026-08-28, เจอตอน verify P7):
                    # plan_cursor เป็น "ตัวนับ action ที่สำเร็จ" ไม่ใช่ "หลักฐานว่างานเสร็จ" —
                    # รันนั้นเดินถูกทุกอย่าง (Admin -> dropdown -> ESS -> Search -> Select All)
                    # แต่ Select All คือ action ที่ 5 พอดี cursor จึงผ่านจำนวนข้อของแผน แล้ว
                    # goal-scope gate ก็ไป **บล็อกปุ่ม Delete ที่ตามมา** และปิด task ด้วย
                    # success=True ทั้งที่ ESS ยังอยู่ครบ 9 คน (ยืนยันด้วยสคริปต์ไม่ใช้ LLM
                    # ก่อน/หลังรัน) — W93 แก้ "โมเดลอ้างเลขข้อสุดท้ายทันที" ไปแล้ว แต่ยังเหลือ
                    # "นับครบจำนวนข้อ = จบ" ซึ่งผิดด้วยเหตุผลเดียวกัน
                    #
                    # goal ลบแบบมีเงื่อนไขมี "การวัดตรงๆ" อยู่แล้ว (เหลือกี่แถวที่ตรงเงื่อนไข)
                    # การวัดต้องชนะตัวนับเสมอ จึงข้าม shortcut ของแผนไปใช้เส้นทางที่อ่านหลักฐาน
                    # จริงด้านล่างแทน — goal อื่นที่ไม่มีวิธีวัดตรงๆ ยังใช้ธงของแผนเหมือนเดิม
                    # W_plan_counter_claims_a_password_change: ฟอร์มเปลี่ยนรหัสผ่านที่ยังกรอก
                    # ไม่ครบคือหลักฐานตรงๆ ว่างานยังไม่จบ — ต้องชนะตัวนับของแผนเช่นเดียวกับที่
                    # การนับแถวที่เหลือชนะมันในงานลบ (ดู docstring ของ helper)
                    if (
                        plan_fully_completed
                        and not delete_all_condition_values
                        and not await _change_password_form_still_unfilled(page)
                    ):
                        goal_scope_satisfied_reason = "the confirmed plan's last step is already complete"
                    else:
                        if not nav_target_reached_confirmed and goal_nav_target and _navigation_target_reached(
                            goal_nav_target, page.url, self.memory.recent(1)
                        ):
                            nav_target_reached_confirmed = True
                        if nav_target_reached_confirmed:
                            goal_scope_satisfied_reason = (
                                f"the goal's navigation target ('{goal_nav_target}') has already been reached"
                            )
                        elif _goal_targets_existing_records_only(goal):
                            # W_zero_records_done: งานลบ/แก้ตามเงื่อนไข "เสร็จ" เมื่อตารางที่
                            # กรองแล้วไม่เหลือแถวที่ตรงเงื่อนไข — สัญญาณเดียวกับที่ guard
                            # premature-deletion ใช้ปฏิเสธ finish_task ตอนยังเหลือ >0 อยู่แล้ว
                            # (_scan_remaining_target_records_once) แค่ใช้ในทิศทางตรงข้าม
                            # คืน None ถ้าหน้านี้ไม่ใช่หน้าตาราง = ไม่มีสัญญาณ ไม่ gate อะไรเลย
                            remaining = await _scan_remaining_target_records_once(page)
                            if remaining is not None and remaining[0] == 0:
                                goal_scope_satisfied_reason = (
                                    "the filtered table has no matching rows left "
                                    f"({remaining[1]!r}), so there is nothing more to act on"
                                )

                # W_step_trace: จับเวลาของ 3 ส่วนที่กินเวลาจริงต่อ step (snapshot / LLM /
                # action) แยกกัน — ทั้งระบบไม่เคยมี instrumentation เวลาเลยสักจุด จึงตอบไม่ได้
                # ว่า task ที่ใช้ 693 วินาทีหมดเวลาไปกับอะไร (ดู config.py::step_trace_log_path)
                _snapshot_started_at = time.monotonic()
                if steps_taken == 0 and cached_elements is not None:
                    elements, page_text = cached_elements, cached_page_text
                else:
                    elements, page_text = await get_snapshot(page)
                step_snapshot_seconds = time.monotonic() - _snapshot_started_at
                await _emit_screenshot(steps_taken)
                # W5[A] verify: เก็บ page_text ล่าสุดไว้เป็นหลักฐานจริงจาก DOM ตอนจบ
                # task (ทุก path — finish_task/loop-detected/หมด max_steps) แนบไปกับ
                # result ให้ผู้ประเมิน (เช่น W12[B] eval script/human review) เทียบกับ
                # message ที่ LLM อ้างได้เอง ไม่ต้องเชื่อคำเคลมของ LLM ลอยๆ อย่างเดียว
                final_page_text = page_text

                # W_consent_banner_midtask (ดู docstring ของ _snapshot_shows_consent_banner):
                # แบนเนอร์คุกกี้โผล่กลางทางได้ ไม่ใช่แค่ตอนโหลดหน้าแรก — ปิดให้ทันทีที่เห็นใน
                # snapshot แล้ว perceive ใหม่ ไม่งั้นโมเดลจะเสีย step ไปกดปุ่มของแบนเนอร์เอง
                # (live run: กด "Allow all" ให้ tracking cookie แทน user โดยไม่มีใครสั่ง)
                if _snapshot_shows_consent_banner(elements):
                    if await _dismiss_consent_banner(page, verbose):
                        elements, page_text = await get_snapshot(page)
                        final_page_text = page_text

                # W_captcha_detect (P3.8): เช็คหลังปิดแบนเนอร์คุกกี้แล้ว เพราะแบนเนอร์บางเจ้า
                # บังหน้าไว้จน snapshot ดูว่างเปล่าคล้าย bot wall — เช็คก่อนจะเข้าใจผิดได้
                # ถามคนแค่ครั้งเดียวต่อ task (โควตาเดียวกับ request_user_input ปกติ) แล้ว
                # perceive ใหม่ ถ้ายังติดอยู่ก็ปล่อยให้ลูปเดินต่อตามปกติ ไม่ฆ่า task ทิ้ง
                if (
                    not captcha_reported
                    and _snapshot_shows_captcha(elements, page_text)
                    and request_user_input_count < _MAX_REQUEST_USER_INPUT_CALLS
                ):
                    captcha_reported = True
                    request_user_input_count += 1
                    if verbose:
                        print("[captcha] เจอ CAPTCHA/bot wall — ขอให้ user ทำเองแล้วรอ", flush=True)
                    await _request_user_input(_CAPTCHA_USER_PROMPT, False, ask_user_func)
                    elements, page_text = await get_snapshot(page)
                    final_page_text = page_text

                # W_session_drift (บั๊กจริง live-reproduce ด้วย goal ของ user เอง: "เปิดเว็ป
                # แล้วไปที่เมนูแอดมิน แล้วลบ user role=ess ออกให้หมด"): auto-login สำเร็จและ
                # agent ไปถึงหน้า Admin คลิกแถวได้แล้วจริง แต่ step ถัดมาโมเดลสั่ง go_back เอง
                # แล้วเด้งกลับไปหน้า login/หน้าโฆษณาของเว็บ — ไม่มีอะไรพากลับเข้าระบบอีกเลย
                # agent เลยใช้ step ที่เหลือทั้งหมด (18 จาก 22) ไปคลิกลิงก์การตลาดบนหน้านั้น
                #
                # ไม่ใช่ปัญหาเฉพาะเว็บนี้: session หมดอายุกลางทาง/กด go_back ข้ามขอบเขตแอป เป็น
                # เรื่องปกติของเว็บที่ต้อง auth ทุกตัว — พอเจอฟอร์ม login โผล่มาอีกครั้งทั้งที่มี
                # credential เก็บไว้แล้ว ให้ปิดแบนเนอร์ + login ใหม่ให้เลย (กลไกเดิมทั้งคู่
                # ไม่ได้สร้างใหม่) แล้ว perceive ใหม่ก่อนถาม LLM
                #
                # มีโควตาเหมือน guard อื่นในไฟล์นี้: ถ้า login ซ้ำแล้วยังเด้งกลับมาอีก แปลว่า
                # credential ใช้ไม่ได้จริง/เว็บบังคับ logout อยู่ — เลิกพยายามแล้วปล่อยให้ loop
                # ปกติจัดการต่อ ไม่วน re-login ไม่รู้จบ
                if (
                    steps_taken > 0
                    and mid_task_relogin_count < _MAX_MID_TASK_RELOGINS
                    and await _login_form_needs_password_cached()
                ):
                    mid_task_relogin_count += 1
                    if verbose:
                        print(
                            f"[session-drift {mid_task_relogin_count}/{_MAX_MID_TASK_RELOGINS}] "
                            f"เจอฟอร์ม login กลางทาง ({page.url}) — ปิดแบนเนอร์แล้วลอง login ใหม่",
                            flush=True,
                        )
                    await _dismiss_consent_banner(page, verbose)
                    relogin_reason = await _maybe_auto_login(page, verbose)
                    if relogin_reason is None:
                        elements, page_text = await get_snapshot(page)
                        final_page_text = page_text

                # W6[B]: ดึงคู่มือที่เกี่ยวข้องกับ goal+หน้าปัจจุบันใหม่ทุก step ที่หน้าเปลี่ยน
                # จริง (retrieve()/recall() ไม่ throw เอง คืน [] เงียบๆ ถ้าไม่มีคู่มือ/error)
                # — ใช้ to_thread เพราะเป็นงาน sync (local embedding inference + ChromaDB
                # query) ไม่งั้นจะบล็อก event loop ตัวเดียวกับที่ Playwright ใช้อยู่ (เหมือน
                # _confirm_plan() ที่ wrap input() ด้วย to_thread ด้วยเหตุผลเดียวกัน)
                #
                # Speed 2.2: retriever.retrieve()/long_term_memory.recall() คำนวณ
                # embed_input จาก (goal, page_text) แบบเดียวกันเป๊ะ แล้ว query ด้วย
                # embedding function เดียวกัน (_embedding_function singleton ใน
                # chroma_client.py) แยกกันคนละ collection — เดิม embed ซ้ำ 2 รอบทั้งที่ผล
                # embedding เหมือนกัน (sequential await ด้วย) เปลี่ยนเป็น embed ครั้งเดียว
                # แล้วส่ง vector สำเร็จรูปเข้าทั้งคู่ (query_embedding=) + ยิง query ของทั้ง
                # สอง collection พร้อมกันผ่าน asyncio.gather() แทน await ทีละตัว — ไม่แตะ
                # _client_lock ที่มีอยู่แล้วใน chroma_client.py (กัน race ตอน init ครั้งแรก
                # จากหลาย thread) การ gather ยังปลอดภัยเพราะ lock นั้นยังทำงานอยู่เหมือนเดิม
                # embed ล้มเหลว (model โหลดไม่ผ่าน ฯลฯ) fallback เป็น query_embedding=None
                # ให้ retrieve()/recall() embed เองจาก query_texts ตามเดิมทุกประการ (ไม่ throw)
                #
                # W22: ถ้า page_text เหมือน step ก่อนหน้าเป๊ะ (เช่น action ก่อนหน้า fail/ไม่
                # navigate ไปไหน) ข้าม retrieval ทั้งคู่ไปเลย ใช้ marker สั้นๆ แทนก้อนข้อความ
                # เดิมที่ LLM เห็นไปแล้วในเทิร์นก่อนหน้า (ดู comment ของ
                # last_page_text_for_context ด้านบนสุดของ run_task())
                page_changed_for_context = page_text != last_page_text_for_context
                if page_changed_for_context:
                    step_embed_input = f"{goal}\n\nCurrent page:\n{page_text}"
                    try:
                        step_embedding = (
                            await asyncio.to_thread(_embedding_function, [step_embed_input])
                        )[0]
                    except Exception:
                        step_embedding = None

                    manual_chunks, long_term_chunks = await asyncio.gather(
                        asyncio.to_thread(
                            retriever.retrieve, query=goal, page_state=page_text,
                            k=_RAG_CHUNKS_PER_STEP, query_embedding=step_embedding,
                        ),
                        asyncio.to_thread(
                            long_term_memory.recall,
                            query=goal, page_state=page_text, k=_LONG_TERM_MEMORY_CHUNKS_PER_STEP,
                            session_id=session_id or "", query_embedding=step_embedding,
                        ),
                    )
                    manual_context = "\n".join(f"- {chunk}" for chunk in manual_chunks)
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
                # W_already_logged_in_but_told_to_log_in (release gate จับได้ 2026-09-07, งาน
                # add_candidate): goal ขึ้นต้นว่า "Log in with username 'Admin' and password
                # 'admin123', go to Recruitment..." แต่ _maybe_auto_login() พาเข้าระบบไปแล้ว
                # ตั้งแต่ก่อนเข้า loop จึงไม่มีฟอร์ม login ให้กรอก โมเดลไม่รู้เรื่องนี้เลยจึงพยายาม
                # ทำตามคำสั่งแรกของ goal ด้วยการยัด username/password ลงช่อง Search ใน sidebar
                # แล้วหลงทางต่ออีก 5 step จนไม่เคยไปถึงฟอร์มเป้าหมาย
                #
                # ระบบรู้คำตอบอยู่แล้ว (auto_login_outcome) แค่ไม่เคยบอกโมเดล — บอกเฉพาะตอนที่
                # login สำเร็จจริง *และ* goal พูดถึงการ login เท่านั้น งานที่ไม่เกี่ยวไม่ต้องจ่าย
                # ค่าบรรทัดนี้ และหยุดบอกหลังพ้น step แรกๆ ไปแล้ว (โมเดลเห็นหน้าหลังล็อกอินเองแล้ว)
                if (
                    auto_login_outcome == "ok"
                    and steps_taken < _MAX_ALREADY_LOGGED_IN_REMINDER_STEPS
                    and contains_keyword(goal, _LOGIN_ONLY_CLAUSE_KEYWORDS)
                ):
                    verification_context = (
                        "[The system already signed in with the stored credentials before this "
                        "task started — the log-in part of the goal is DONE. There is no log-in "
                        "form on screen; never type a username or password into a search box or "
                        "any other field. Carry on from the next part of the goal.]"
                    )
                if steps_taken > 0 and not page_changed_for_context:
                    last_record = self.memory.recent(1)
                    if last_record:
                        last_cmd = last_record[0].get("cmd", {}) or {}
                        if (
                            last_record[0].get("success") is True
                            and last_cmd.get("type") in _VERIFICATION_SIGNAL_ACTION_TYPES
                        ):
                            verification_context = (
                                f"[Verification of the previous action ({last_cmd}): no change "
                                "was detected on the page at all (its elements/content are "
                                "byte-for-byte identical) — this action may have had no real "
                                "effect even though it returned [OK]. Consider another route "
                                "(e.g. hover before clicking, click a different position/index, "
                                "or for a custom dropdown try press_key instead)]"
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

                # W41: หน่วงเฉพาะส่วนที่ยังขาดให้ครบ settings.step_pacing_delay_seconds นับจากที่
                # next_action() ครั้งก่อนจบ (ไม่ใช่ sleep เต็มจำนวนทุกครั้งแบบเดิม) — งาน
                # จริงที่ทำไปแล้วตั้งแต่ครั้งก่อน (execute()/wait_stable()/get_snapshot()/
                # retrieve()/recall() ด้านบน) นับรวมเข้าไปในระยะห่างนี้ด้วย ระยะห่างขั้นต่ำ
                # ระหว่างการเรียก LLM 2 ครั้งยังเท่าเดิมทุกประการ (ไม่ลดความปลอดภัยจาก
                # rate-limit) แค่ไม่ sleep ซ้ำกับเวลาที่ผ่านไปแล้วจริง
                # W_timing_gap: การรอนี้ตั้งใจ (กัน rate limit) แต่ต้อง "เห็นได้" ในรายงาน
                # ไม่ใช่หายไปในช่องว่างที่ไม่มีใครวัด — ตอนวิเคราะห์ทีหลังจะได้แยกออกจาก
                # ความช้าที่แก้ได้จริง
                step_pacing_seconds = 0.0
                if last_llm_call_at is not None:
                    elapsed = time.monotonic() - last_llm_call_at
                    remaining = settings.step_pacing_delay_seconds - elapsed
                    if remaining > 0:
                        step_pacing_seconds = remaining
                        await asyncio.sleep(remaining)

                # W43: plan_text (ดู confirm_plan/approved_plan ด้านบน) เป็น None สำหรับ
                # ad-hoc task ที่ไม่มีแผนเลย — ส่งเป็น "" ให้ next_action()/
                # _build_user_turn_text() ไม่ต้องรู้จัก Optional เอง (plan_context="" =
                # ไม่มี section "แพลนปัจจุบัน" โผล่มาปนเลย ตรงกับพฤติกรรมเดิมทุกประการ)
                # W_fill_secret_schema_gate (ดูเหตุผลเต็มใน llm.py ที่ค่าคงที่ชื่อเดียวกัน):
                # คำนวณ "หน้านี้ใช้ fill_secret ได้จริงไหม" *ก่อน* เรียก LLM แล้วส่งเข้าไปให้
                # llm.py ตัด fill_secret ออกจาก tool schema เลยถ้าใช้ไม่ได้ — แทนที่จะเสนอ
                # ตลอดเวลาแล้วให้ guard ด้านล่างไล่ปฏิเสธทีหลัง (ซึ่งเสีย step/token และจบด้วย
                # loop-detected ทุกครั้งในเคสจริง)
                #
                # เงื่อนไขเดียวกันเป๊ะกับ guard ด้านล่าง และคำนวณที่นี่ที่เดียว แล้ว guard
                # ใช้ค่าเดิมซ้ำ — schema กับ guard จึงพูดตรงกันเสมอ และไม่ต้องยิง DOM check
                # ซ้ำสองรอบต่อ step
                #
                # W_secret_gate_stays_page_only (regression 2026-09-03 ที่ผมทำเองแล้ว user
                # เจอจากการรันสด 2 รอบติด): เคยแก้บรรทัดนี้ให้เปิด fill_secret ตั้งแต่ตอน
                # goal/แผน *พูดถึง* การเปลี่ยนรหัสผ่าน เพื่อให้กฎ W20 ถูกส่งเร็วขึ้น — ผิดจุด
                # และรื้อ W_fill_secret_schema_gate ทิ้งพอดี: gpt-5.4-mini บน endpoint
                # ChatGPT OAuth กรอกทุก property ในสคีมาทุกครั้ง พอ `secret` มี enum ค่าเดียว
                # มันจึงส่งมาตลอดแล้วลาก `type` เป็น fill_secret ไปด้วย ผลคือ agent ยิง
                # fill_secret ใส่ index มั่วตั้งแต่หน้า login (หลุดไปถึงหน้า Help & Support)
                # ไม่เคยกดเมนูโปรไฟล์เลยสักรอบ — คือ 5/5 failure แบบเดิมเป๊ะ
                #
                # กฎ W20 ไม่ต้องพึ่งธงตัวนี้อยู่แล้ว: _resolve_prompt_sections() เปิดบล็อก
                # password จาก _goal_or_plan_requests_password_change() ของมันเองแยกต่างหาก
                # (ดู W_password_rules_arrive_too_late) ธงตัวนี้จึงกลับไปถามแค่ข้อเดียวตามเดิม
                # — "ตอนนี้ยืนอยู่บนฟอร์มเปลี่ยนรหัสผ่านจริงไหม"
                allow_fill_secret = await _page_looks_like_change_password_form(page)
                # W_secret_stays_in_schema_forever: สคีมาแคบกว่าบริบทหนึ่งขั้น — พอช่อง Current
                # Password ถูกกรอกแล้ว fill_secret ไม่มีประโยชน์อีกเลย ตัดออกทันทีเพื่อให้โมเดล
                # ไม่มีทางส่งมันมาซ้ำได้ ส่วน guard/บล็อก prompt ยังใช้ allow_fill_secret ตัวกว้าง
                # ต่อไป (ไม่งั้น guard hardening จะเข้าใจผิดว่า "ไม่ใช่หน้าเปลี่ยนรหัสผ่าน" แล้ว
                # บังคับ recovery ทั้งที่หน้าถูกแล้ว)
                fill_secret_in_schema = (
                    allow_fill_secret and await _current_password_field_is_empty(page)
                )

                # W_steptimeout: ครอบ timeout เหมือนที่ routes.py::generate_plan ทำกับ
                # generate_plan อยู่แล้ว (ดู config.py::llm_step_timeout_seconds) — ปล่อยให้
                # TimeoutError ทะลุขึ้นไปหา except ของ run_task (W_loop_crash) ซึ่งจะรายงาน
                # ว่าค้างที่ step ไหนพร้อมคืนงานที่ทำไปแล้วครบ แทนที่จะค้างเงียบตลอดไป
                _llm_started_at = time.monotonic()
                prompt_sections = _resolve_prompt_sections(
                    prompt_sections, goal=goal, plan_text=plan_text, elements=elements,
                    allow_fill_secret=allow_fill_secret,
                    # site_manual_context เป็นพารามิเตอร์ของ run_task จึงมีค่าเสมอ —
                    # ห้ามใช้ effective_site_manual ตรงนี้ มันถูกกำหนดค่าทีหลังในลูป
                    manual_context=site_manual_context or manual_context or "",
                )

                # W_token_trim (P2/M1): ยุบ page snapshot ของ turn เก่าใน history ก่อน
                # ยิง LLM — เก็บอันก่อนหน้าไว้เต็ม 1 อัน (+ อันปัจจุบันที่ next_action จะ
                # append) = เห็น snapshot เต็ม 2 อันล่าสุดในทุก request
                messages = _dedupe_stale_snapshots(messages, keep_last_full=1)

                # W_token_cut W5: ยุบ *ทั้ง* user turn ของ step เก่า (เกิน 2 อันท้าย) เหลือ
                # แค่ Goal + stub — บล็อกกฎ/plan/scaffolding/manual ในนั้นถูกส่งสดใหม่ทุก
                # turn อยู่แล้ว + digest เก็บผลลัพธ์ไว้ครบ (หลักฐาน W_prompt_audit: turn เก่า
                # ที่สะสม = component ที่ใหญ่ที่สุดในงานยาว โต ~4.5-5k tok/call ไม่มีเพดาน)
                messages, _w5_removed = _compact_stale_user_turns(messages, goal)
                if _w5_removed:
                    history_compaction_events += 1
                    history_chars_saved += _w5_removed

                # W_token_cut W7: บล็อกกฎที่ gate ใน turn เก่าทุกอันเหลือ 1 บรรทัดอ้างอิง
                # (turn ปัจจุบันที่ next_action จะ append ยังส่งเต็ม) — เนื้อกฎเหมือนเดิม
                # ทุก turn, model ถูกสั่งให้ยึด turn ล่าสุด
                messages, _w7_removed = _dedupe_stale_gated(messages, keep_last_full=0)
                if _w7_removed:
                    gated_deref_events += 1
                    gated_chars_saved += _w7_removed

                # W_token_trim (P3/M3): full manual on the first step and on the first step
                # after any compaction (which would have spliced the earlier full copy
                # out); a short id+summary reference every other step
                if not site_manual_full:
                    effective_site_manual = ""
                elif force_full_site_manual or not site_manual_full_sent:
                    effective_site_manual = site_manual_full
                    site_manual_full_sent = True
                    force_full_site_manual = False
                else:
                    effective_site_manual = site_manual_ref

                tool_name, tool_input, tool_use_id, messages, usage = await asyncio.wait_for(
                    next_action(
                        client, model, effective_goal, page_text, messages,
                        manual_context, memory_context, long_term_context, vision_context,
                        effective_site_manual, page.url, action_history_context,
                        # W_plan_step_cursor: ส่งแผนพร้อมเครื่องหมายว่าอยู่ข้อไหน แทนแผนดิบ
                        _focused_plan_context(plan_text, plan_cursor),
                        verification_context=verification_context,
                        allow_fill_secret=fill_secret_in_schema,
                        prompt_sections=prompt_sections,
                    ),
                    timeout=settings.llm_step_timeout_seconds,
                )
                last_llm_call_at = time.monotonic()
                # W_step_trace: เวลาที่ใช้กับ LLM ของ step นี้ล้วนๆ (ไม่รวม pacing delay ด้านบน
                # ซึ่งเป็นการรอโดยตั้งใจ ไม่ใช่ความช้าของ provider)
                step_llm_seconds = last_llm_call_at - _llm_started_at
                total_usage += usage
                llm_turns += 1
                _record_payload_audit(usage)  # W_prompt_audit
                # W_token_cut W1: cache ติดไหมต่อเทิร์น — บน endpoint openai ที่รันจริงมีแค่
                # cached_tokens ให้ดู (ดู W_token_cut ในหมายเหตุ) ตัวเลขนี้ x/llm_calls = hit ratio
                if usage.cache_read_tokens > 0:
                    cache_hit_turns += 1
                else:
                    cache_miss_turns += 1
                if verbose:
                    print(
                        f"  [tokens] input={usage.input_tokens} output={usage.output_tokens}"
                        f" cache_read={usage.cache_read_tokens} cache_write={usage.cache_creation_tokens}"
                        f" (รวม: input={total_usage.input_tokens} output={total_usage.output_tokens}"
                        f" cache_read={total_usage.cache_read_tokens} cache_write={total_usage.cache_creation_tokens})",
                        flush=True,
                    )

                # W_unknown_tool: เดิมเช็คแค่ "request_user_input" กับ "finish_task" ที่เหลือ
                # ตกลงไปเส้นทาง browser_action หมดโดยไม่ตรวจอะไรเลย — ชื่อ tool ที่โมเดลมโนขึ้น
                # เอง (เกิดได้จริง โดยเฉพาะกับ provider ที่เราเห็นแล้วว่ากรอกสคีมามั่ว) จึงถูก
                # ส่งเข้า actions.execute() แล้วได้ ActionResult(False, "unknown action") กลับมา
                # ซึ่งไม่ได้บอกโมเดลเลยว่า "ชื่อ tool ผิด" และมี tool อะไรให้ใช้บ้าง — เสีย step
                # ฟรีๆ แล้ววนผิดซ้ำได้เรื่อยๆ ตอบให้ตรงจุดแทน แล้วไปต่อโดยไม่นับเป็น step
                if tool_name not in _KNOWN_TOOL_NAMES:
                    _bump_guard("unknown_tool")  # W_token_cut W1
                    if verbose:
                        print(f"[unknown-tool] โมเดลเรียก tool ที่ไม่มีอยู่จริง: {tool_name!r}", flush=True)
                    messages = append_tool_result(
                        messages, tool_use_id,
                        f"[Rejected by the system] there is no tool named {tool_name!r}. The only "
                        f"tools that exist are: {', '.join(sorted(_KNOWN_TOOL_NAMES))}. Call one of "
                        "those instead — use browser_action for anything you want to do on the page.",
                    )
                    continue

                # W_resume ("Mid-Task Input Request"): ตรวจก่อน finish_task เสมอ (คนละ tool
                # กันเลย ไม่ใช่ alias/ไม่ผ่าน guard ของ finish_task ด้านล่างเลยสักจุด) —
                # หยุด loop รอคำตอบจาก human จริงๆ ผ่าน _request_user_input() (mechanism
                # เดียวกับ _confirm_plan()/permission prompt) แล้ว "ทำ loop เดิมต่อทันที"
                # ด้วยคำตอบที่ได้ (ป้อนกลับเป็น tool_result ของ tool_use นี้เอง) — ไม่ return
                # ไม่ reset plan ไม่ต้องรอเทิร์นถัดไปเหมือน finish_task(false) เดิม
                if tool_name == "request_user_input":
                    prompt_text = str(tool_input.get("prompt", "")).strip()
                    sensitive = bool(tool_input.get("sensitive", False))

                    if request_user_input_count >= _MAX_REQUEST_USER_INPUT_CALLS:
                        _bump_guard("request_user_input_quota")  # W_token_cut W1
                        if verbose:
                            print(
                                f"[request_user_input เกินโควตา {_MAX_REQUEST_USER_INPUT_CALLS} "
                                "ครั้ง — ปฏิเสธไม่ให้หยุดรออีก]", flush=True,
                            )
                        messages = append_tool_result(
                            messages, tool_use_id,
                            f"[Rejected by the system] you have already asked this kind of "
                            f"question {_MAX_REQUEST_USER_INPUT_CALLS} times in this task — do "
                            "not call request_user_input again. Decide from the information you "
                            "already have, or call finish_task(success=false) if you genuinely "
                            "cannot continue.",
                        )
                        continue

                    request_user_input_count += 1
                    steps_taken += 1
                    provided, answer = await _request_user_input(prompt_text, sensitive, ask_user_func)
                    log_cmd = {"type": "request_user_input", "prompt": prompt_text, "sensitive": sensitive}
                    result_text = (
                        f"the user answered: {answer}" if provided
                        else "[No answer] the user declined or did not answer within the time limit"
                    )
                    # W_secret_answer_not_logged: คำตอบต้องไปถึงโมเดล (ไม่งั้นกรอกไม่ได้) แต่ไม่
                    # ควรไปโผล่ในที่ที่ถูกเก็บไว้ยาวๆ — panel LOG บนหน้าจอ, data/step_trace.jsonl
                    # บนดิสก์ และ ShortTermMemory ที่ถูกสรุปกลับเข้า prompt ทุก step ล้วนไม่จำเป็น
                    # ต้องรู้ค่าจริง (ค่าจริงอยู่ใน messages ของเทิร์นนี้อยู่แล้ว) — ปิดบังเฉพาะตอน
                    # sensitive=True เท่านั้น คำตอบทั่วไป (ชื่อ/ตัวเลข) ยังเห็นได้เหมือนเดิม
                    logged_result_text = (
                        "the user answered: [hidden]" if provided and sensitive else result_text
                    )
                    if verbose:
                        print(f"[request_user_input] {prompt_text!r} -> {logged_result_text}", flush=True)
                    self.memory.record({
                        "step": steps_taken,
                        "cmd": log_cmd,
                        "label": "",
                        "result": logged_result_text,
                        "success": provided,
                        "tokens": _tokens_dict(usage),
                        "locator_descriptor": None,
                    })
                    await _emit({
                        "kind": "step", "step": steps_taken, "cmd": log_cmd,
                        "label": "", "result": logged_result_text, "success": provided,
                        "tokens": _tokens_dict(total_usage),
                        "llm_calls": llm_turns,
                    })
                    messages = append_tool_result(messages, tool_use_id, result_text)
                    continue

                if tool_name == "finish_task":
                    finish_task_calls += 1  # W_token_cut W1
                    claimed_success = bool(tool_input.get("success", False))

                    # ยังเหลือ step ให้ลอง + เป็น finish_task call จริง (มี tool_use_id ให้
                    # ผูก tool_result กลับ ไม่ใช่ fallback ตอนโมเดลไม่ยอมเรียก tool เลย) +
                    # ยังไม่เกิน quota การเตือน -> ไม่ยอมรับ false ทันที เตือนแล้วให้ลองต่อ
                    if (
                        not claimed_success
                        and tool_use_id
                        # W_step_budget: เทียบ "รอบที่ใช้ไป" กับงบรอบจริง ไม่ใช่ steps_taken
                        # (ตัวนับ action ซึ่งน้อยกว่าเสมอ) — เดิมเงื่อนไขนี้ยังเป็นจริงอยู่ตอน
                        # รอบใกล้หมดแล้ว ทำให้ finish_task(false) ที่ถูกต้องถูกปฏิเสธ แล้วลูปจบ
                        # เองด้วยข้อความ default ทิ้งคำอธิบายจริงของโมเดลไปเปล่าๆ
                        and iterations_used < max_steps
                        and premature_false_finish_count < _MAX_PREMATURE_FALSE_FINISH_RETRIES
                    ):
                        premature_false_finish_count += 1
                        _bump_guard("premature_false_finish")  # W_token_cut W1
                        # W_step_budget: เก็บคำอธิบายของโมเดลไว้ เผื่อสุดท้ายลูปจบเพราะหมดรอบ
                        # จริงๆ — ดีกว่ารายงานแค่ "ครบ max_steps โดยยังไม่จบ task" ลอยๆ
                        last_rejected_finish_message = str(tool_input.get("message", "")).strip()
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
                            f"⚠️ [Important system command]: your latest finish_task(false) was "
                            f"rejected outright! The goal '{goal}' is not complete and the page "
                            "still has elements left. Do not give up until you have tried acting "
                            "on what remains — look at the list again and continue!",
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
                        _bump_guard("premature_true_finish")  # W_token_cut W1
                        if verbose:
                            print(
                                f"[finish_task(true) ไม่ยอมรับทันที {premature_true_finish_count}/"
                                f"{_MAX_PREMATURE_TRUE_FINISH_RETRIES}] message={tool_input.get('message', '')}",
                                flush=True,
                            )
                        messages = append_tool_result(messages, tool_use_id, _PREMATURE_TRUE_FINISH_NUDGE)
                        messages.append(_build_nudge_message(
                            resolved_provider,
                            f"⚠️ [Important system command]: calling finish_task(true) without "
                            "having performed any action requires clear evidence from the "
                            f"current indexed elements that the goal '{goal}' really is "
                            "complete, before you confirm again.",
                        ))
                        continue

                    # ACC-3 (accuracy audit follow-up): symmetric กับ guard ด้านบน (steps_taken
                    # ==0) แต่เช็ค steps_taken > 0 แทน — โมเดลลอง action จริงมาแล้วหลาย step
                    # (ผ่าน guard แรกไปแล้ว) แต่ถ้าไม่มี mutating action ไหนสำเร็จเลยสักครั้ง
                    # ตลอดทั้ง task ก็ยังน่าสงสัยเหมือนกัน (ดู _has_any_successful_mutating_
                    # action() ด้านบนสุดของไฟล์สำหรับเหตุผลเต็ม)
                    if (
                        claimed_success
                        and tool_use_id
                        and steps_taken > 0
                        and not _has_any_successful_mutating_action(self.memory.all())
                        and premature_all_failed_count < _MAX_PREMATURE_ALL_FAILED_RETRIES
                    ):
                        premature_all_failed_count += 1
                        _bump_guard("premature_all_failed")  # W_token_cut W1
                        if verbose:
                            print(
                                f"[finish_task(true) ไม่มี mutating action ไหนสำเร็จเลย "
                                f"{premature_all_failed_count}/{_MAX_PREMATURE_ALL_FAILED_RETRIES}] "
                                f"message={tool_input.get('message', '')}",
                                flush=True,
                            )
                        messages = append_tool_result(messages, tool_use_id, _PREMATURE_ALL_FAILED_NUDGE)
                        messages.append(_build_nudge_message(
                            resolved_provider,
                            f"⚠️ [Important system command]: {_PREMATURE_ALL_FAILED_NUDGE}",
                        ))
                        continue

                    # Task4 (W19, ดู _scan_validation_errors ด้านบนสุดของไฟล์): เช็คทุกครั้ง
                    # ที่ claimed_success (ไม่ผูกกับ steps_taken เหมือน guard ด้านบน) เพราะ
                    # error อาจโผล่ขึ้นมาหลัง action ผ่านไปหลาย step แล้วก็ได้ ไม่ใช่แค่ step
                    # แรกสุด
                    detected_errors: list[str] = []
                    if claimed_success and tool_use_id:
                        detected_errors = await _scan_validation_errors(page)
                    # W65[2] ("Error Passthrough"): แยก error ที่ "fatal" (retry ไปก็ไม่มีทาง
                    # หายเอง ต้องข้อมูล/สิทธิ์ใหม่จาก user เท่านั้น) ออกจาก error อื่นทั้งหมด —
                    # fatal ข้าม nudge-retry loop ไปเลย บังคับความจริงลง final result ทันที
                    # (ไม่ต้องรอ retry quota) ต่างจาก error ทั่วไปด้านล่างที่ยังให้ LLM ลองแก้เอง
                    # ก่อนตามเดิมทุกประการ (backward compatible 100%)
                    fatal_errors = [e for e in detected_errors if _is_fatal_validation_error(e)]
                    if fatal_errors:
                        fatal_text = "; ".join(fatal_errors)
                        if verbose:
                            print(f"[finish_task(true) พบ fatal validation error — ข้าม retry] {fatal_text}", flush=True)
                        completion_verification = "TASK_FAILED_USER_INPUT_ERROR"
                        claimed_success = False
                        retry_value_field_labels = list(agent_filled_field_labels)
                        tool_input["message"] = (
                            "ไม่สามารถดำเนินการต่อได้ เนื่องจากข้อมูลที่กรอกไม่ผ่านการตรวจสอบ"
                            f"ของระบบ: {fatal_text} — กรุณาตอบกลับมาด้วยค่าใหม่ที่ต้องการใช้แทน "
                            "ระบบจะกรอกค่านั้นแทนที่ในช่องเดิมแล้วดำเนินการต่อให้ทันที"
                        )
                    elif (
                        detected_errors
                        and premature_validation_error_count < _MAX_PREMATURE_VALIDATION_ERROR_RETRIES
                    ):
                        premature_validation_error_count += 1
                        _bump_guard("validation_error")  # W_token_cut W1
                        errors_text = "; ".join(detected_errors)
                        if verbose:
                            print(
                                f"[finish_task(true) พบ validation error {premature_validation_error_count}/"
                                f"{_MAX_PREMATURE_VALIDATION_ERROR_RETRIES}] {errors_text}",
                                flush=True,
                            )
                        nudge_text = _PREMATURE_VALIDATION_ERROR_NUDGE_TEMPLATE.format(errors=errors_text)
                        messages = append_tool_result(messages, tool_use_id, nudge_text)
                        messages.append(_build_nudge_message(resolved_provider, f"⚠️ [Important system command]: {nudge_text}"))
                        continue
                    elif detected_errors:
                        # retry ครบโควตาแล้วยังเจอ error ค้างอยู่ — ปล่อยผ่านไปตามที่โมเดล
                        # ยืนยัน (escape valve เดียวกับ guard อื่นในไฟล์นี้) แต่ tag ผลลัพธ์
                        # ไว้ให้ผู้เรียกรู้ว่าน่าสงสัย แทนที่จะค้างไม่รู้จบ
                        completion_verification = "EXECUTION_FAILED_NEEDS_REPAIR"

                    # W22 ("DOM-Based Post-Action Verification Guardrail", ขยายรวม edit-all
                    # ใน W64[7.1]): เช็คเฉพาะ deletion-intent หรือ edit-all-intent goal (ดู
                    # _is_deletion_intent_goal/_is_edit_all_intent_goal ด้านบนสุดของไฟล์) —
                    # อ่านจำนวนแถวที่เหลืออยู่จริงจาก DOM (ไม่ใช่เชื่อคำอธิบายของ LLM) ก่อน
                    # ยอมรับ finish_task(success=true) กันปัญหา hallucinated false-completion
                    # ที่ user รายงานจริงทั้งสองแบบ (deletion: agent ตอบ "ไม่พบ/ลบครบแล้ว"
                    # ทั้งที่ตารางยังโชว์ "(3) Records Found" — edit-all: agent เห็น 1 แถว
                    # ตรงเงื่อนไข target เหลืออยู่แต่ไม่กด Edit ให้ครบ) — สัญญาณ "เสร็จ" ของ
                    # ทั้งสองแบบเหมือนกันเป๊ะ: แถวที่ตรงเงื่อนไข filter เดิมต้องเหลือ 0
                    remaining_records: Optional[tuple[int, str]] = None
                    if (
                        claimed_success and tool_use_id
                        and (_is_deletion_intent_goal(goal) or _is_edit_all_intent_goal(goal))
                    ):
                        remaining_records = await _scan_remaining_target_records(page)

                    # W_delete_all_intent: guard สองชั้นที่ _scan_remaining_target_records()
                    # เดิมจับไม่ได้ ทั้งคู่เจอจริงใน run ที่ claim สำเร็จผิดๆ เมื่อ 2026-08-26
                    #
                    # (1) ไม่เคยกด Search เลย — ตัวเลข "(N) Records Found" ที่ guard เดิมอ่าน
                    #     จึงเป็นของตารางที่ยังไม่ถูกกรอง ไม่มีความหมายกับเงื่อนไขของ goal เลย
                    # (2) finish_task ถูกเรียกบนหน้าที่ไม่มีตารางแล้ว (run จริงเผลอคลิกลิงก์
                    #     footer "OrangeHRM, Inc" ก่อนจบ) — guard เดิมคืน None = "เช็คไม่ได้"
                    #     แล้วปล่อยผ่านเงียบๆ ซึ่งสำหรับงาน "ลบทั้งหมด" คือการยอมรับคำกล่าวอ้าง
                    #     ที่ไม่มีหลักฐานรองรับเลยสักชิ้น ต่างจาก goal ทั่วไปตรงที่ความเสียหาย
                    #     ของการเชื่อผิดคือข้อมูลที่ยังไม่ถูกลบจริง แต่ user เข้าใจว่าลบแล้ว
                    if claimed_success and tool_use_id and delete_all_condition_values:
                        condition_text = " + ".join(repr(v) for v in delete_all_condition_values)
                        row_match = (
                            None if remaining_records is not None
                            else await _count_rows_matching_condition(page, delete_all_condition_pairs)
                        )
                        blocking_nudge = None
                        if filter_changed_without_search:
                            blocking_nudge = _DELETE_ALL_NO_SEARCH_NUDGE
                        elif remaining_records is None and row_match is None:
                            blocking_nudge = _DELETE_ALL_UNVERIFIED_NUDGE_TEMPLATE.format(
                                condition=condition_text,
                            )
                        if (
                            blocking_nudge
                            and premature_delete_all_unverified_count < _MAX_DELETE_ALL_UNVERIFIED_RETRIES
                        ):
                            premature_delete_all_unverified_count += 1
                            _bump_guard("delete_all_unverified")  # W_token_cut W1
                            if verbose:
                                print(
                                    f"[ลบทั้งหมดยังพิสูจน์ไม่ได้ "
                                    f"{premature_delete_all_unverified_count}/"
                                    f"{_MAX_DELETE_ALL_UNVERIFIED_RETRIES}] {blocking_nudge[:80]}",
                                    flush=True,
                                )
                            messages = append_tool_result(messages, tool_use_id, blocking_nudge)
                            messages.append(_build_nudge_message(
                                resolved_provider, f"⚠️ [Important system command]: {blocking_nudge}",
                            ))
                            continue

                        # W_delete_all_intent: ถ้าหน้านี้ไม่มี "(N) Records Found" (เว็บส่วนใหญ่
                        # ในโลกไม่มี — ดู _RECORD_COUNT_SELECTOR) แต่มีตารางจริงอยู่ ให้นับแถวที่
                        # ยังตรงเงื่อนไขเองแบบ generic แล้วป้อนเข้า guard เดิมด้านล่างตามปกติ
                        # (ใช้ทางเดิมทั้งหมด: nudge, โควตา, และการเขียนทับ claimed_success)
                        if remaining_records is None:
                            if row_match is not None and row_match[0] > 0:
                                remaining_records = (
                                    row_match[0],
                                    f"{row_match[0]} of the {row_match[1]} rows on this page still "
                                    f"match {condition_text}",
                                )
                    if (
                        remaining_records is not None
                        and remaining_records[0] > 0
                        and premature_deletion_incomplete_count < _MAX_PREMATURE_DELETION_INCOMPLETE_RETRIES
                    ):
                        premature_deletion_incomplete_count += 1
                        _bump_guard("deletion_incomplete")  # W_token_cut W1
                        remaining_count, remaining_text = remaining_records
                        if verbose:
                            print(
                                f"[finish_task(true) ยังลบไม่ครบ "
                                f"{premature_deletion_incomplete_count}/"
                                f"{_MAX_PREMATURE_DELETION_INCOMPLETE_RETRIES}] {remaining_text}",
                                flush=True,
                            )
                        nudge_text = _PREMATURE_DELETION_INCOMPLETE_NUDGE_TEMPLATE.format(
                            count=remaining_count, text=remaining_text,
                            action_hint=_premature_mutation_action_hint(goal),
                        )
                        messages = append_tool_result(messages, tool_use_id, nudge_text)
                        messages.append(_build_nudge_message(resolved_provider, f"⚠️ [Important system command]: {nudge_text}"))
                        continue
                    # W_empty_table_needs_right_filter: ตารางว่างเป็นหลักฐานความสำเร็จได้ก็
                    # ต่อเมื่อตัวกรองบนหน้าคือตัวที่ goal สั่งจริง — ไม่งั้น "ว่างเพราะกรองผิด"
                    # จะถูกนับเป็น "ว่างเพราะลบครบ" (ดูเหตุผลเต็มที่จุดประกาศ helper)
                    if (
                        claimed_success
                        and tool_use_id
                        and delete_all_condition_pairs
                        and remaining_records is not None
                        and remaining_records[0] == 0
                        and premature_deletion_incomplete_count
                        < _MAX_PREMATURE_DELETION_INCOMPLETE_RETRIES
                    ):
                        filter_ok = _page_filter_matches_goal(elements, delete_all_condition_pairs)
                        extra_filters = _extra_filters_set_on_page(
                            elements, delete_all_condition_pairs,
                        )
                        problem = ""
                        if filter_ok is False:
                            problem = "the filter on screen is not set to what the goal asked for"
                        elif extra_filters:
                            problem = (
                                "an extra filter the goal never mentioned is still narrowing the "
                                f"table ({', '.join(repr(f) for f in extra_filters)})"
                            )
                        if problem:
                            premature_deletion_incomplete_count += 1
                            _bump_guard("empty_table_wrong_filter")  # W_token_cut W1
                            nudge_text = _EMPTY_TABLE_WRONG_FILTER_NUDGE_TEMPLATE.format(
                                problem=problem,
                                condition=" + ".join(
                                    f"{f}={v}" for f, v in delete_all_condition_pairs
                                ),
                            )
                            if verbose:
                                print(
                                    f"[ตารางว่างแต่ตัวกรองไม่ตรง "
                                    f"{premature_deletion_incomplete_count}/"
                                    f"{_MAX_PREMATURE_DELETION_INCOMPLETE_RETRIES}] {problem}",
                                    flush=True,
                                )
                            messages = append_tool_result(messages, tool_use_id, nudge_text)
                            messages.append(_build_nudge_message(
                                resolved_provider,
                                f"\u26a0\ufe0f [Important system command]: {nudge_text}",
                            ))
                            continue

                    if remaining_records is not None and remaining_records[0] > 0:
                        # retry ครบโควตาแล้วยังลบ/แก้ไขไม่ครบจริง — ต่างจาก validation-error
                        # guard ด้านบนที่ปล่อยผ่านตามคำยืนยันของโมเดล (error message ตีความได้
                        # หลายแบบ) ตัวเลขแถวที่เหลือนับได้ตรงๆ ไม่มีทางตีความผิด ต้อง "บังคับ
                        # ความจริง" ลง final result เสมอ (TRUTH-BASED RESPONSE GENERATION ตาม
                        # ที่ user สั่ง) — เขียนทับทั้ง claimed_success และ message ของ LLM เอง
                        # ไม่ปล่อยให้คำอธิบาย hallucinate ของ LLM หลุดออกไปถึง user เด็ดขาด
                        completion_verification = "EXECUTION_FAILED_NEEDS_REPAIR"
                        remaining_count, _ = remaining_records
                        claimed_success = False
                        # W64[7.1]: ข้อความต่างกันตาม intent — deletion พูดว่า "ยังไม่ถูกลบ"
                        # ส่วน edit-all พูดว่า "ยังไม่ถูกแก้ไข" ไม่งั้นข้อความจะผิดความจริงถ้า
                        # goal จริงๆ เป็นงานแก้ไข ไม่ใช่งานลบ
                        if _is_deletion_intent_goal(goal):
                            tool_input["message"] = (
                                f"พบผู้ใช้งาน/รายการที่ตรงเงื่อนไขเหลืออยู่ {remaining_count} รายการในระบบ "
                                f"และยังไม่ได้ถูกลบออก (พยายามลบซ้ำแล้ว "
                                f"{premature_deletion_incomplete_count} ครั้งแต่ยังไม่สำเร็จ) — โปรดลองสั่ง"
                                f"ลบอีกครั้งหรือดำเนินการต่อด้วยตนเอง"
                            )
                        else:
                            tool_input["message"] = (
                                f"พบผู้ใช้งาน/รายการที่ตรงเงื่อนไขเหลืออยู่ {remaining_count} รายการในระบบ "
                                f"และยังไม่ได้ถูกแก้ไขค่าตามที่ต้องการ (พยายามแก้ไขซ้ำแล้ว "
                                f"{premature_deletion_incomplete_count} ครั้งแต่ยังไม่สำเร็จ) — โปรดลองสั่ง"
                                f"แก้ไขอีกครั้งหรือดำเนินการต่อด้วยตนเอง"
                            )

                    # W_count_answer_check: คำตอบของ goal เชิงนับต้องมีตัวเลขที่โค้ดนับไว้
                    # อยู่จริง — ดู docstring ของ _MAX_COUNT_ANSWER_MISMATCH_RETRIES ด้านบนสุด
                    # ของไฟล์สำหรับเงื่อนไขทั้ง 3 ข้อที่ต้องครบพร้อมกันก่อนจะยิง
                    if (
                        claimed_success and tool_use_id and system_counted
                        and _goal_asks_for_a_count(goal)
                        and premature_count_answer_mismatch_count < _MAX_COUNT_ANSWER_MISMATCH_RETRIES
                    ):
                        contradicted = _count_answer_contradiction(
                            goal, str(tool_input.get("message") or ""), system_counted,
                        )
                        if contradicted is not None:
                            premature_count_answer_mismatch_count += 1
                            _bump_guard("count_answer_mismatch")  # W_token_cut W1
                            value, count = contradicted
                            if verbose:
                                print(
                                    f"[คำตอบขัดกับตัวเลขที่โค้ดนับ "
                                    f"{premature_count_answer_mismatch_count}/"
                                    f"{_MAX_COUNT_ANSWER_MISMATCH_RETRIES}] "
                                    f"{value!r} นับได้ {count} แต่ไม่มีในคำตอบ",
                                    flush=True,
                                )
                            nudge_text = _COUNT_ANSWER_MISMATCH_NUDGE_TEMPLATE.format(
                                count=count, value=value,
                            )
                            messages = append_tool_result(messages, tool_use_id, nudge_text)
                            messages.append(_build_nudge_message(
                                resolved_provider, f"⚠️ [Important system command]: {nudge_text}",
                            ))
                            continue

                    # W63[7.2] ("Strict Table Assertion & Truth Reporting"): เช็คเฉพาะตอนที่
                    # LLM ระบุ verify_text มาเอง (ดู llm.py::_FINISH_TASK_PARAMS) — อ่าน DOM
                    # จริงของ table body ก่อนยอมรับว่ารายการที่สร้าง/บันทึกไปโผล่ในตารางจริง
                    # (ไม่ใช่เชื่อคำอธิบายของ LLM เฉยๆ — หลักการเดียวกับ deletion-verification
                    # guard ด้านบน)
                    verify_text = str(tool_input.get("verify_text") or "").strip()
                    table_item_found = True
                    if (
                        claimed_success and tool_use_id and verify_text
                        and (goal_wants_a_record_change or wrote_a_value_this_task)
                    ):
                        # W_verify_text_on_delete_goal (บั๊กจริง live stability check 2026-08-31):
                        # verify_text ถูกออกแบบมาสำหรับงาน *สร้าง* รายการ ("ชื่อที่เพิ่งสร้างต้อง
                        # โผล่ในตารางจริง") — SYSTEM_PROMPT (W63[7.2]) ก็สั่งไว้ตรงตัวว่างานลบให้
                        # เว้นว่าง แต่โมเดลส่ง verify_text="No Records Found" มาบนงานลบ แล้ว guard
                        # ก็ไล่หาข้อความนั้นเป็น *แถวหนึ่งในตาราง* ตามหน้าที่ ไม่เจอ (มันเป็น
                        # ข้อความสถานะ ไม่ใช่แถว) จึงพลิกงานที่สำเร็จจริงให้กลายเป็น
                        # VERIFICATION_FAILED — ตารางว่างเปล่าคือ *หลักฐานว่าสำเร็จ* ของงานลบ
                        # ไม่ใช่หลักฐานว่าล้มเหลว
                        #
                        # สองชั้น ชั้นแรกตรงตามสัญญาที่ prompt เขียนไว้แล้ว ชั้นสองกันเคสทั่วไป
                        # (โมเดลส่งวลี "ไม่มีผลลัพธ์" มาบน goal ชนิดอื่น) ใช้ชุดคำเดิม
                        # _RECORD_COUNT_ZERO_TEXTS ไม่สร้างชุดใหม่
                        verify_text_is_zero_phrase = any(
                            zero_text in verify_text.lower()
                            for zero_text in _RECORD_COUNT_ZERO_TEXTS
                        )
                        if goal_is_deletion_only or verify_text_is_zero_phrase:
                            if verbose:
                                print(
                                    f"[verify_text] ข้ามการหาแถวในตาราง ({verify_text!r}) — "
                                    "งานลบใช้ 'ตารางว่าง' เป็นหลักฐานความสำเร็จ ไม่ใช่ความล้มเหลว",
                                    flush=True,
                                )
                        else:
                            table_item_found = await _scan_created_item_in_table(page, verify_text)
                    if (
                        not table_item_found
                        and premature_table_verify_count < _MAX_PREMATURE_TABLE_VERIFY_RETRIES
                    ):
                        if "table_verify" in finish_reject_reasons_seen:
                            # W_token_cut W3: เคยตีกลับ finish ด้วยเหตุผลนี้แล้ว 1 ครั้ง —
                            # ไม่เสียเทิร์นเตือนซ้ำ ตกไปเส้นทาง "ยอมรับพร้อม tag ความจริง"
                            # ด้านล่าง (EXECUTION_FAILED_NEEDS_REPAIR / OK_SAVE_CONFIRMED...)
                            finish_loop_prevented += 1
                            if verbose:
                                print("[W3] finish_task(table_verify) collapse — ไม่เตือนซ้ำ", flush=True)
                        else:
                            premature_table_verify_count += 1
                            finish_reject_reasons_seen.add("table_verify")
                            _bump_guard("table_verify")  # W_token_cut W1
                            if verbose:
                                print(
                                    f"[finish_task(true) ไม่พบ verify_text ในตาราง "
                                    f"{premature_table_verify_count}/"
                                    f"{_MAX_PREMATURE_TABLE_VERIFY_RETRIES}] {verify_text}",
                                    flush=True,
                                )
                            nudge_text = _PREMATURE_TABLE_VERIFY_NUDGE_TEMPLATE.format(text=verify_text)
                            if any_toast_confirmed_this_task:
                                nudge_text += _TOAST_CONFIRMED_NO_RECREATE_SUFFIX
                            messages = append_tool_result(messages, tool_use_id, nudge_text)
                            messages.append(_build_nudge_message(resolved_provider, f"⚠️ [Important system command]: {nudge_text}"))
                            continue
                    if not table_item_found and any_toast_confirmed_this_task:
                        # W64[7.2]: มีหลักฐาน toast ยืนยันสำเร็จจริงมาก่อนหน้านี้ใน task
                        # เดียวกัน — ต่างจาก branch ด้านล่าง (ไม่มีหลักฐานอะไรเลยนอกจากคำยืนยัน
                        # ของ LLM เอง) ตรงนี้ไม่ force claimed_success=False (คงค่า True ที่
                        # ผ่านเงื่อนไข claimed_success ด้านบนมาแล้ว) แค่เขียนทับ message ด้วย
                        # วลีตรงสเปคที่ user ระบุ ("บันทึกสำเร็จแล้ว แต่ไม่พบในตาราง") — ไม่ใช่
                        # VERIFICATION_FAILED เพราะนี่ไม่ใช่ความล้มเหลว มีหลักฐานจริงว่าบันทึก
                        # สำเร็จแล้ว แค่ตรวจไม่เจอในตาราง (อาจเป็นปัญหา search/filter/
                        # pagination มากกว่า)
                        completion_verification = "OK_SAVE_CONFIRMED_NOT_IN_TABLE"
                        tool_input["message"] = "บันทึกข้อมูลเรียบร้อยแล้ว แต่ไม่พบรายการในตารางการค้นหา"
                    elif not table_item_found:
                        # retry ครบโควตาแล้วยังไม่เจอในตารางจริง และไม่มีหลักฐาน toast ยืนยัน
                        # เลยด้วย — บังคับความจริงลง final result เสมอ (TRUTH-BASED RESPONSE
                        # GENERATION เหมือน guard ด้านบน) ใช้ข้อความ "VERIFICATION_FAILED: Item
                        # not found in results table." ตรงตามสเปคที่ user ระบุ ให้ผู้เรียก/
                        # ผู้ตรวจสอบ log เห็นสัญญาณนี้ชัดเจน
                        completion_verification = "EXECUTION_FAILED_NEEDS_REPAIR"
                        claimed_success = False
                        tool_input["message"] = (
                            f'VERIFICATION_FAILED: Item not found in results table. '
                            f'(ค้นหา "{verify_text}" ในตารางผลลัพธ์แล้วไม่พบจริง หลังพยายามแล้ว '
                            f"{premature_table_verify_count} ครั้ง)"
                        )

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
                # W_secret_fill_is_a_fill: fill_secret คือ *วิธีเดียว* ที่ระบบเปิดให้กรอก
                # ช่องรหัสผ่านด้วยค่าที่เก็บไว้ การไม่มีชื่อมันอยู่ในรายการนี้แปลว่า guard ที่
                # ตั้งใจบังคับ "กรอกรหัสผ่านให้ครบก่อน" กลับไปปฏิเสธการกรอกรหัสผ่านเสียเอง
                if (
                    tool_input.get("type") not in ("fill", "fill_secret", "goto")
                    and await _login_form_needs_password_cached()
                ):
                    if premature_login_skip_count < _MAX_PREMATURE_LOGIN_SKIP_RETRIES:
                        premature_login_skip_count += 1
                        _bump_guard("login_skip")  # W_token_cut W1
                        if verbose:
                            print(
                                f"[login-form ยังไม่ครบ {premature_login_skip_count}/"
                                f"{_MAX_PREMATURE_LOGIN_SKIP_RETRIES}] ปฏิเสธ action={tool_input}",
                                flush=True,
                            )
                        messages = append_tool_result(messages, tool_use_id, _PREMATURE_LOGIN_SKIP_NUDGE)
                        # W_token_cut W3: ครั้งแรกของเหตุผลนี้เท่านั้นที่แนบ user-turn nudge
                        # เสริม (~500 tok) — ครั้งซ้ำ tool_result อย่างเดียวพอ ไม่ทบ history
                        if _first_guard_hit("login_skip"):
                            messages.append(_build_nudge_message(
                                resolved_provider,
                                "⚠️ [Important system command]: this page still has an empty "
                                "Password field. Do not move on to any other action (including "
                                "wait) until both Username and Password are filled in. Look at "
                                "the indexed elements and fill the empty field right now.",
                            ))
                        continue
                    # เกินโควตาเตือนแล้วยังไม่ยอมกรอก ปล่อยผ่านไปตามที่โมเดลเลือกแทนที่จะ
                    # ค้างไม่รู้จบ (เหมือน escape valve ของ premature-false-finish guard)


                # W_fill_secret_hardening (ดู comment เหนือ _PASSWORD_CHANGE_INTENT_KEYWORDS
                # สำหรับบั๊กจริงที่แก้): ปฏิเสธ fill_secret ก่อน dispatch เสมอ ถ้าไม่มีสัญญาณว่านี่
                # คือ change-password context จริง — ไม่มี escape valve แบบ "เกินโควตาแล้วปล่อย
                # ผ่าน" เหมือน guard อื่นในไฟล์นี้ เพราะปล่อยผ่านแปลว่าพิมพ์รหัสผ่านจริงลง element
                # ที่ไม่รู้ว่าคืออะไร — โควตาที่นี่จึงใช้ "บังคับ recovery action" แทน (กลไก
                # _force_loop_recovery ตัวเดียวกับ loop-detection ด้านล่าง ไม่ใช่กลไกใหม่)
                # W_fill_secret_schema_gate: ใช้ค่าที่คำนวณไว้แล้วก่อนเรียก LLM (ดูจุดนั้น) —
                # guard นี้ยังต้องอยู่แม้ schema จะตัด fill_secret ออกไปแล้ว เพราะ (1) provider
                # อาจ "หลุด" ส่ง type ที่ไม่มีใน enum มาได้อยู่ดี (model compliance ไม่การันตี
                # — เหตุผลเดียวกับที่ไฟล์นี้มี guard เกือบทุกตัว) และ (2) fastpath/แผนที่บันทึก
                # ไว้ก่อนหน้าอาจ replay action นี้เข้ามาโดยไม่ผ่าน tool schema เลย
                if tool_input.get("type") == "fill_secret" and not allow_fill_secret:
                    consecutive_fill_secret_context_reject_count += 1
                    _bump_guard("fill_secret_context")  # W_token_cut W1
                    if verbose:
                        print(
                            f"[fill-secret-context {consecutive_fill_secret_context_reject_count}/"
                            f"{_MAX_FILL_SECRET_CONTEXT_REJECT_RETRIES}] ปฏิเสธ (ไม่ใช่ change-password context) action={tool_input}",
                            flush=True,
                        )
                    if consecutive_fill_secret_context_reject_count > _MAX_FILL_SECRET_CONTEXT_REJECT_RETRIES:
                        loop_reason = (
                            f"fill_secret was rejected {consecutive_fill_secret_context_reject_count} times in "
                            "a row because there is still no genuine change-password context (the goal/plan "
                            "doesn't ask for one, and this page doesn't look like a Change Password form)"
                        )
                        # W_state_guard_shortcut: ก่อน fallback ไป go_back/scroll ทั่วไป (ซึ่งไม่
                        # เคยพาไปใกล้ goal เลย) ลองหา element ที่ตรงกับ goal จริงๆ ด้วย heuristic
                        # เดียวกับที่ใช้สร้าง nudge text — เจอแล้วสั่ง click ตรงนั้นแทนเลย
                        # (ปลอดภัยกว่า fill_secret มาก เพราะ click ไม่มีทางเขียน credential ผิดที่)
                        recovery_target = _fill_secret_recovery_target(elements, tool_input, goal)
                        forced_click_cmd = (
                            {"type": "click", "index": recovery_target["index"]}
                            if recovery_target is not None
                            else None
                        )
                        if await _force_loop_recovery(
                            loop_reason, forced_cmd=forced_click_cmd, forced_target=recovery_target,
                        ):
                            consecutive_fill_secret_context_reject_count = 0
                            if verbose:
                                print(f"[fill-secret-context] {loop_reason} -> forcing recovery action instead", flush=True)
                            continue
                        success = False
                        final_message = f"Stopping task: {loop_reason} (a recovery action was forced but the loop persisted)"
                        if verbose:
                            print(f"[fill-secret-context] {final_message}", flush=True)
                        break
                    context_nudge_text = _FILL_SECRET_NOT_PASSWORD_CONTEXT_NUDGE + _fill_secret_context_hint(
                        elements, tool_input, goal,
                    )
                    messages = append_tool_result(messages, tool_use_id, context_nudge_text)
                    messages.append(_build_nudge_message(
                        resolved_provider,
                        f"⚠️ [Important system command]: {context_nudge_text}",
                    ))
                    continue
                consecutive_fill_secret_context_reject_count = 0

                # W_secret_refilled_forever: ช่องที่ fill_secret เล็งอยู่ถูกกรอกไปแล้ว การกรอกซ้ำ
                # จึงไม่ใช่ความคืบหน้า — ปฏิเสธแล้วชี้ index ของช่องที่ยังว่างอยู่จริงให้
                if tool_input.get("type") == "fill_secret":
                    _pw_states = await _password_field_states(page)
                    _target_index = str(tool_input.get("index"))
                    _already_filled = any(
                        str(st.get("index")) == _target_index and st.get("filled")
                        for st in _pw_states
                    )
                    _still_empty = [
                        st for st in _pw_states if not st.get("filled") and st.get("index")
                    ]
                    if (
                        _already_filled
                        and _still_empty
                        and secret_refill_reject_count < _MAX_SECRET_REFILL_RETRIES
                    ):
                        secret_refill_reject_count += 1
                        _bump_guard("secret_refill")  # W_token_cut W1
                        _empty_desc = ", ".join(
                            f"[{st['index']}] {_label_for_index(elements, st['index'])}"
                            for st in _still_empty
                        )
                        _refill_nudge = (
                            f"[Rejected] The Current Password field (index {_target_index}) is "
                            "ALREADY filled — the system typed the saved password into it and it "
                            "worked. Filling it again changes nothing and wastes a step. The "
                            "password fields still EMPTY right now are: "
                            f"{_empty_desc}. Put the new password into those with a normal "
                            "'fill' action (fill_secret only ever works on Current Password), "
                            "then submit the form."
                        )
                        if verbose:
                            print(
                                f"[secret-refill {secret_refill_reject_count}/"
                                f"{_MAX_SECRET_REFILL_RETRIES}] ปฏิเสธ กรอกซ้ำช่องที่เต็มแล้ว "
                                f"— ช่องที่ยังว่าง: {_empty_desc}",
                                flush=True,
                            )
                        messages = append_tool_result(messages, tool_use_id, _refill_nudge)
                        continue

                # W_click_submits_with_empty_password_fields: กดส่งฟอร์มทั้งที่ช่องรหัสผ่าน
                # ในฟอร์มเดียวกันยังว่าง = ล้ม validation แน่นอน แล้วบดบัง error จริงที่ต้องแก้
                if tool_input.get("type") == "click":
                    _pw_problem = await state_filter.password_form_submit_problem(
                        page, tool_input.get("index"),
                    )
                    _empty_pw = (_pw_problem or {}).get("indexes") or []
                    if _pw_problem and empty_password_submit_count < _MAX_EMPTY_PASSWORD_SUBMIT_RETRIES:
                        empty_password_submit_count += 1
                        _bump_guard("empty_password_submit")  # W_token_cut W1
                        _empty_pw_desc = ", ".join(
                            f"[{idx}] {_label_for_index(elements, idx)}" for idx in _empty_pw
                        )
                        _empty_pw_nudge = (
                            "[Rejected] This button submits a form whose password field(s) are "
                            f"still empty: {_empty_pw_desc}. Submitting now fails validation "
                            "and hides the real problem. Fill every one of those fields first, "
                            "then submit."
                        ) if _pw_problem.get("kind") == "empty" else (
                            "[Rejected] The new-password fields on this form do not hold the "
                            f"same value: {_empty_pw_desc}. Submitting now fails with "
                            "'Passwords do not match'. Type the SAME new password into every "
                            "one of them — including the confirmation field, which may still "
                            "hold a value you typed earlier — then submit."
                        )
                        if verbose:
                            print(
                                f"[empty-password-submit {empty_password_submit_count}/"
                                f"{_MAX_EMPTY_PASSWORD_SUBMIT_RETRIES}] ปฏิเสธ กดส่งฟอร์มทั้งที่"
                                f"ช่องรหัสผ่านยังว่าง: {_empty_pw_desc}",
                                flush=True,
                            )
                        messages = append_tool_result(messages, tool_use_id, _empty_pw_nudge)
                        continue

                # loop-detection: action เดิมเป๊ะๆ ติดกันกี่ครั้งแล้ว (นับรวมทั้ง success/fail
                # เพราะแม้ execute() สำเร็จทุกครั้ง แต่ถ้า LLM สั่งซ้ำเดิมไม่เปลี่ยน ก็ไม่ใช่
                # ความคืบหน้าจริงอยู่ดี)
                #
                # W21 ("Batch/Bulk Action Protocol"): ข้อยกเว้นเดียว — action ประเภทที่ต้อง
                # ขอยืนยันจาก human ทุกครั้งอยู่แล้ว (submit/delete/purchase/pay,
                # DEFAULT_NEEDS_CONFIRMATION) ที่ "เดิมเป๊ะๆ" (dict เท่ากันทุก field รวม
                # index) และครั้งก่อนหน้าสำเร็จจริง (last_action_succeeded) มักเกิดจาก
                # bulk-delete fallback loop ที่ถูกต้อง ไม่ใช่ agent ค้างวน — เช่น "ลบ user
                # ทั้งหมด" ต้องคลิกถังขยะ "แถวแรก" ซ้ำๆ แต่พอลบแถวแรกสำเร็จ แถวถัดไปเลื่อน
                # ขึ้นมาแทนที่ตำแหน่งเดิมพอดี ได้ data-ai-index ตัวเดิมซ้ำทุกรอบ (perception.py
                # คำนวณ index ใหม่จากลำดับ DOM ทุกครั้ง ไม่ใช่ identity ของ element เดิม) —
                # เป็นความคืบหน้าจริง (แถวถูกลบไปแล้วจริงทุกรอบ) ต่างจาก loop ที่ guard นี้
                # ตั้งใจจะจับ (agent ค้างซ้ำโดยไม่มีผลอะไรเปลี่ยนแปลงเลย) — ปลอดภัยเพราะ human
                # ยังต้องกดอนุมัติทุกครั้งอยู่ดี (ask_user_func ใน execute()) ไม่มีทางวนไม่รู้จบ
                # แบบไม่มีใครควบคุม ถ้าครั้งก่อนหน้า "fail" ซ้ำๆ (เช่น index ผิด/element หาไม่
                # เจอ) ยังนับเป็น repeat ตามปกติเหมือนเดิมทุกประการ (ไม่เข้าเงื่อนไขยกเว้นนี้)
                is_bulk_safe_repeat = (
                    tool_input.get("type") in DEFAULT_NEEDS_CONFIRMATION and last_action_succeeded is True
                )
                # W29: เทียบด้วย _cmd_for_repeat_comparison() (ตัด completed_plan_step ทิ้ง
                # ก่อน) ไม่ใช่ tool_input ดิบ — ดู docstring เหนือฟังก์ชันนั้นสำหรับเหตุผลเต็ม
                normalized_tool_input = _cmd_for_repeat_comparison(tool_input)
                if normalized_tool_input == last_action_cmd and not is_bulk_safe_repeat:
                    consecutive_repeat_count += 1
                else:
                    last_action_cmd = normalized_tool_input
                    consecutive_repeat_count = 1

                # W_same_label_loop (ดู docstring ของ _MAX_CONSECUTIVE_SAME_LABEL_ACTIONS
                # ด้านบนสุดของไฟล์): นับซ้ำอีกชั้นด้วย (type, label) แทน (type, index) —
                # จับเคสที่ agent วนทำ "ของชนิดเดียวกันคนละแถว" ไปเรื่อยๆ ซึ่ง guard คาบ 1
                # ด้านบนมองไม่เห็นเลยเพราะ index ต่างกันทุกครั้ง ข้ามไปถ้าไม่มี label ให้เทียบ
                # (ตัดสินไม่ได้ ปลอดภัยกว่าปล่อย) หรือเป็น bulk-safe repeat แบบเดียวกับด้านบน
                # หา label เองตรงนี้ ไม่รอ action_label ที่คำนวณทีหลัง (อยู่หลัง guard นี้ในลูป
                # — ย้ายขึ้นมาจะไปสลับลำดับ guard อื่นที่พึ่งตำแหน่งเดิม) ใช้ snapshot ชุดเดียวกัน
                _same_label_index = tool_input.get("index")
                _same_label_text = next(
                    (e["label"] for e in elements if e["index"] == _same_label_index), ""
                ) if _same_label_index is not None else ""
                # W_label_marker_key: เทียบด้วย "ตัวตน" ของ element ไม่ใช่ label ดิบ —
                # ดูเหตุผลเต็มที่จุดประกาศ _label_without_markers()
                _same_label_identity = _label_without_markers(_same_label_text)
                same_label_key = (
                    (tool_input.get("type"), _same_label_identity) if _same_label_identity else None
                )
                if same_label_key is None or is_bulk_safe_repeat:
                    last_same_label_key = None
                    consecutive_same_label_count = 0
                elif same_label_key == last_same_label_key:
                    consecutive_same_label_count += 1
                else:
                    last_same_label_key = same_label_key
                    consecutive_same_label_count = 1

                if consecutive_repeat_count >= _MAX_CONSECUTIVE_IDENTICAL_ACTIONS:
                    loop_reason = (
                        f"the agent issued the same action {consecutive_repeat_count} times "
                        f"in a row ({tool_input}) with no progress"
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

                if consecutive_same_label_count >= _MAX_CONSECUTIVE_SAME_LABEL_ACTIONS:
                    loop_reason = (
                        f"the agent issued '{tool_input.get('type')}' on an element labelled "
                        f"{_same_label_text.strip()!r} {consecutive_same_label_count} times in a row "
                        "(different indexes, same element kind) with no progress toward the goal"
                    )
                    if await _force_loop_recovery(loop_reason):
                        consecutive_same_label_count = 0
                        if verbose:
                            print(f"[same-label-loop] {loop_reason} -> บังคับ recovery action แทน", flush=True)
                        continue
                    success = False
                    final_message = f"หยุด task: {loop_reason} (บังคับ recovery action ไปแล้วแต่ยังไม่หาย)"
                    if verbose:
                        print(f"[same-label-loop] {final_message}", flush=True)
                    break

                # loop-detection (2026-07-13, generalize 2026-07-15): จับ pattern วนซ้ำ
                # เป็นคาบ (คาบ 2 เช่น go_back -> click -> go_back -> click, คาบ 3 เช่น
                # click A -> scroll -> fill B -> click A -> scroll -> fill B, ...) ที่
                # guard ด้านบน (คาบ 1) จับไม่ได้เพราะ action แต่ละตัวไม่ได้ "เดิมเป๊ะๆ
                # ติดกัน" — เก็บ history แค่ _MAX_CYCLE_WINDOW ตัวล่าสุดพอ ไม่ต้องเก็บ
                # ทั้ง task (ดู _detect_repeating_cycle_period()/_is_repeating_cycle()
                # ด้านบนสุดของไฟล์)
                # W29: เก็บ normalized_tool_input (ตัด completed_plan_step ทิ้งแล้ว) ไม่ใช่
                # tool_input ดิบ — เหตุผลเดียวกับ guard คาบ 1 ด้านบน
                recent_actions.append(normalized_tool_input)
                if len(recent_actions) > _MAX_CYCLE_WINDOW:
                    recent_actions.pop(0)

                detected_period = _detect_repeating_cycle_period(recent_actions)
                if detected_period is not None:
                    cycle_desc = " -> ".join(str(a) for a in recent_actions[-detected_period:])
                    loop_reason = f"the agent is cycling actions with period {detected_period} ({cycle_desc}) with no progress"
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
                # W_index_drift_measure: element ที่ index นี้ยังเป็นตัวเดิมกับตอน snapshot ไหม
                if action_index is not None and action_label:
                    _live = await _live_label_at_index(page, action_index)
                    if _live is None:
                        index_drift_gone += 1
                    elif _live and _label_without_markers(action_label) not in _live:
                        index_drift_changed += 1
                        if verbose:
                            print(
                                f"[index-drift] index {action_index}: snapshot={action_label!r} "
                                f"-> ตอน dispatch={_live[:40]!r}",
                                flush=True,
                            )
                # W_search follow-up 2: attribute "type" ของ element (เช่น input ที่
                # type="text"/"search") ส่งคู่กับ tag ให้ execute()/classify_action() แยก
                # ช่องกรอกข้อความ/ค้นหาธรรมดาออกจาก input ที่แท้จริงอาจเสี่ยง (ดู
                # permission/rules.py::SAFE_INPUT_TAG/RISKY_INPUT_TYPES)
                action_element_type = next(
                    (e.get("type", "") for e in elements if e["index"] == action_index), ""
                ) if action_index is not None else ""

                # W_chain ("Compound Actions"): เดียวกับ action_label/action_tag/
                # action_element_type ด้านบนทุกประการ แค่ resolve ให้ then_click_index
                # (element ที่สองที่จะคลิกต่อทันทีถ้ามี — ดู llm.py::_BROWSER_ACTION_PARAMS
                # "then_click_index", actions.py::_maybe_chain_click) จาก elements snapshot
                # เดียวกัน ให้ classify_action() ของ action ที่สองมีสัญญาณเสริมเหมือน action
                # หลักทุกประการ ไม่ใช่แค่ index เปล่าๆ
                then_click_index = tool_input.get("then_click_index")
                then_label = next(
                    (e["label"] for e in elements if e["index"] == then_click_index), ""
                ) if then_click_index is not None else ""
                then_tag = next(
                    (e.get("tag", "") for e in elements if e["index"] == then_click_index), ""
                ) if then_click_index is not None else ""
                then_element_type = next(
                    (e.get("type", "") for e in elements if e["index"] == then_click_index), ""
                ) if then_click_index is not None else ""

                # W_goal_scope ("Goal Boundary Gate" — ดู docstring เต็มใกล้
                # _extract_simple_navigation_target/_navigation_target_reached ด้านบน): พอ
                # goal_scope_satisfied_reason (คำนวณต้น iteration นี้) บอกว่า goal สำเร็จแล้ว
                # เหลือให้ทำได้แค่ action แบบอ่านอย่างเดียว (_GOAL_SCOPE_ALLOWED_ACTION_TYPES)
                # อย่างอื่นทั้งหมด (เดินเข้าเมนู/โมดูลอื่น, สร้าง/แก้/ลบข้อมูล) ถูกปฏิเสธที่นี่
                # ก่อน dispatch เหมือน pre-dispatch guard ตัวอื่นในไฟล์นี้ — มี nudge ให้แค่
                # 1 ครั้ง (_MAX_PREMATURE_GOAL_SCOPE_RETRIES) แล้ว HARD stop เลย จงใจไม่ทำ
                # escape valve แบบ "เกินโควตาแล้วปล่อยผ่าน" เหมือน guard อื่น เพราะการปล่อย
                # action นอก scope ผ่านไปคือ failure mode ที่ guard นี้ถูกสร้างมาแก้พอดี
                #
                # edge case ที่รู้อยู่และตั้งใจไม่จัดการ: ถ้าฟอร์ม login โผล่กลับมาหลัง goal ถูก
                # ตัดสินว่าสำเร็จแล้ว (เช่น session หมดอายุพอดีหลังถึงหน้าเป้าหมาย) guard นี้จะ
                # บล็อก "fill" ที่ใช้กู้สถานะด้วย — หายากพอที่จะบันทึกไว้เฉยๆ แทนการเขียนเคสพิเศษ
                if goal_scope_satisfied_reason and tool_input.get("type") not in _GOAL_SCOPE_ALLOWED_ACTION_TYPES:
                    if consecutive_goal_scope_reject_count < _MAX_PREMATURE_GOAL_SCOPE_RETRIES:
                        consecutive_goal_scope_reject_count += 1
                        _bump_guard("goal_scope")  # W_token_cut W1
                        if verbose:
                            print(
                                f"[goal-scope {consecutive_goal_scope_reject_count}/"
                                f"{_MAX_PREMATURE_GOAL_SCOPE_RETRIES}] ปฏิเสธ action นอก scope "
                                f"({goal_scope_satisfied_reason}): {tool_input}",
                                flush=True,
                            )
                        nudge_text = _GOAL_SCOPE_GATE_NUDGE_TEMPLATE.format(
                            reason=goal_scope_satisfied_reason, action_type=tool_input.get("type"),
                        )
                        messages = append_tool_result(messages, tool_use_id, nudge_text)
                        messages.append(_build_nudge_message(
                            resolved_provider, f"⚠️ [Important system command]: {nudge_text}",
                        ))
                        continue
                    # W_goal_scope_false_success (บั๊กจริง live run 2026-08-27): เส้นทางนี้
                    # ตั้ง success=True ตรงๆ แล้ว break — *ไม่ผ่าน finish_task เลย* guard กัน
                    # false-completion ทั้งชุดของ W_delete_all_intent จึงไม่มีโอกาสได้ตรวจสัก
                    # ตัว ผลจริง: agent กรอง Role ผิด (ได้ Admin แทน ESS) ไม่ได้ลบอะไรเลย
                    # แต่ UI ขึ้น Done + success
                    #
                    # ใช้หลักฐาน deterministic ชุดเดียวกับที่ guard ของ finish_task ใช้อยู่แล้ว
                    # (_scan_remaining_target_records) — ถ้า goal ระบุเงื่อนไขไว้ชัดและยังอ่าน
                    # ได้ว่าเหลือแถวตรงเงื่อนไข ห้ามอ้างว่าสำเร็จ ให้รายงานตามความจริงแทน
                    # fail-safe: อ่านไม่ได้/หน้านี้ไม่ใช่หน้าตาราง คืน None = ไม่มีหลักฐานขัดแย้ง
                    # ก็คงพฤติกรรมเดิมทุกประการ
                    success = True
                    final_message = _GOAL_SCOPE_GATE_HARD_STOP_MESSAGE_TEMPLATE.format(
                        reason=goal_scope_satisfied_reason,
                    )
                    if delete_all_condition_values:
                        # W_empty_table_needs_right_filter: เช็คตัวกรองก่อนตัวนับแถว — ถ้ากรอง
                        # ผิดอยู่ ตัวเลข 0 ที่อ่านได้ไม่มีความหมายเลยตั้งแต่ต้น ไม่ต้องไปดูมัน
                        # (elements ตรงนี้เป็น snapshot ของ iteration ปัจจุบันแล้ว — คนละกรณี
                        # กับจุดที่ตั้ง goal_scope_satisfied_reason ซึ่งเกิดก่อน get_snapshot)
                        _filter_ok = _page_filter_matches_goal(elements, delete_all_condition_pairs)
                        _extra_filters = _extra_filters_set_on_page(
                            elements, delete_all_condition_pairs,
                        )
                        if _filter_ok is False or _extra_filters:
                            success = False
                            final_message = (
                                "Stopping task without completing it: the table looks empty, but "
                                "that is not evidence the job is done — "
                                + (
                                    "the filter on screen is not set to what the goal asked for"
                                    if _filter_ok is False
                                    else "an extra filter the goal never mentioned is still "
                                    f"narrowing it ({', '.join(repr(f) for f in _extra_filters)})"
                                )
                                + f". Nothing is being claimed as done for "
                                f"{delete_all_condition_values!r}."
                            )
                            if verbose:
                                print(f"[goal-scope] {final_message}", flush=True)
                            break

                        remaining = await _scan_remaining_target_records(page)
                        evidence = None
                        if remaining is not None and remaining[0] > 0:
                            evidence = remaining[1]
                        elif remaining is None:
                            # W_plan_cursor_not_proof: ข้อความนับของหน้านั้นอาจไม่ใช่รูปแบบที่
                            # _RECORD_COUNT_PATTERNS รู้จัก (ของจริงที่เจอ: หลังกด Select All
                            # OrangeHRM เปลี่ยนจาก "(9) Records Found" เป็น "(9) Records
                            # Selected") — ตกลงมานับแถวจากตารางตรงๆ แทน ซึ่งเป็นหลักฐานที่
                            # แข็งกว่าข้อความสรุปอยู่แล้ว ใช้ helper ตัวเดียวกับ guard ของ
                            # finish_task ไม่เขียนตัวนับใหม่
                            # W_column_headers_fallback: ตรงนี้ใช้ตัดสินว่า "ยังเหลืองานไหม"
                            # ซึ่งเป็นทิศที่ผิดแล้วอ้างว่าเสร็จ -> ต้องเล็งคอลัมน์ได้จริงเท่านั้น
                            row_match = await _count_rows_matching_condition(
                                page, delete_all_condition_pairs, require_columns=True,
                            )
                            if row_match is not None and row_match[0] > 0:
                                evidence = (
                                    f"{row_match[0]} of {row_match[1]} visible rows still match"
                                )
                        if evidence is not None:
                            success = False
                            final_message = (
                                f"Stopping task without completing it: the goal asked to delete "
                                f"every record matching {delete_all_condition_values!r}, but the "
                                f"table still reports {evidence!r}. Nothing is being claimed "
                                "as done — re-check that the filter was applied to the right "
                                "column and that the rows were actually deleted."
                            )
                            if verbose:
                                print(f"[goal-scope] ปฏิเสธการอ้างสำเร็จ: {final_message}", flush=True)
                    if verbose:
                        print(f"[goal-scope] {final_message}", flush=True)
                    break
                elif not goal_scope_satisfied_reason:
                    consecutive_goal_scope_reject_count = 0
                # else: goal_scope_satisfied_reason ถูกตั้งแล้วแต่ action นี้อยู่ในชุด read-only ที่
                # อนุญาต — จงใจ *ไม่* reset counter ตรงนี้ ถ้า reset ทุก allowed action โมเดลจะ
                # หนี hard-stop ได้ตลอดกาลด้วยการแทรก read_page_data/wait/scroll/hover คั่นระหว่าง
                # การละเมิดแต่ละครั้ง (guard อื่นในไฟล์นี้ reset ได้ปลอดภัยเพราะไม่มี action
                # ประเภท "ผ่านฟรี" แบบนี้ให้ใช้)

                # W_no_credential_flow (ดูคอมเมนต์เหนือ _CREDENTIAL_GOAL_KEYWORDS ด้านบน
                # สำหรับบั๊กจริง): goal ที่ไม่ได้พูดถึงรหัสผ่าน/บัญชีเลย ต้องไม่เข้าหน้า
                # Change Password เด็ดขาด — hard reject ไม่มีโควตา เพราะเปลี่ยนรหัสของบัญชี
                # ที่ login อยู่ = ล็อก user ออกจากระบบจริง go_back() กู้ไม่ได้
                if (
                    not goal_mentions_credentials
                    and tool_input.get("type") in ({"click", "goto"} | DEFAULT_NEEDS_CONFIRMATION)
                    and _action_enters_credential_flow(tool_input, action_label)
                ):
                    what = action_label.strip() or str(tool_input.get("url") or tool_input.get("type"))
                    nudge_text = _NO_CREDENTIAL_FLOW_NUDGE.format(what=what)
                    if verbose:
                        print(f"[no-credential-flow] ปฏิเสธ action ที่พาไปหน้าเปลี่ยนรหัสผ่าน: {what!r}", flush=True)
                    messages = append_tool_result(messages, tool_use_id, nudge_text)
                    messages.append(_build_nudge_message(
                        resolved_provider, f"⚠️ [Important system command]: {nudge_text}",
                    ))
                    continue

                # W_reject_obscured_click (P8/M3, ดูเหตุผลเต็มที่จุดประกาศค่าคงที่): เป้าที่ถูก
                # บังอยู่ + มี dialog เปิดค้าง = คลิกไปก็ timeout แน่นอน ไม่ต้องเสียเวลาไปพิสูจน์
                # อ่าน "มี dialog ไหม" จาก snapshot ที่เพิ่งดึงมาแล้ว (W_dialog_in_snapshot ติด
                # ป้าย [in open dialog] ให้) ไม่ยิง DOM query เพิ่มต่อ action
                if (
                    _OBSCURED_LABEL_MARKER in (action_label or "")
                    and tool_input.get("type") in ({"click"} | DEFAULT_NEEDS_CONFIRMATION)
                    and obscured_click_reject_count < _MAX_OBSCURED_CLICK_RETRIES
                ):
                    dialog_labels = [
                        str(el.get("label") or "")
                        for el in (elements or [])
                        if _DIALOG_LABEL_MARKER in str(el.get("label") or "")
                    ]
                    if dialog_labels:
                        obscured_click_reject_count += 1
                        _bump_guard("obscured_click")  # W_token_cut W1
                        nudge_text = _OBSCURED_CLICK_NUDGE_TEMPLATE.format(
                            label=action_label,
                            dialog_hint=(
                                " (the dialog currently shows: "
                                + ", ".join(repr(l) for l in dialog_labels[:5])
                                + ")"
                            ),
                        )
                        if verbose:
                            print(
                                f"[obscured-click {obscured_click_reject_count}/"
                                f"{_MAX_OBSCURED_CLICK_RETRIES}] {action_label!r} "
                                f"ถูก dialog บังอยู่ ({len(dialog_labels)} element ใน dialog)",
                                flush=True,
                            )
                        messages = append_tool_result(messages, tool_use_id, nudge_text)
                        messages.append(_build_nudge_message(
                            resolved_provider, f"\u26a0\ufe0f [Important system command]: {nudge_text}",
                        ))
                        continue

                # W_prefer_row_delete: เมนูโปรไฟล์/บัญชีของผู้ใช้เองไม่เคยเป็นทางไปสู่งานที่
                # goal สั่ง (นอกจาก goal จะพูดถึงบัญชีเอง) แถมนำไปสู่ flow logout/เปลี่ยน
                # รหัสผ่าน — perception ติดป้ายให้แล้ว ใช้ป้ายนั้นตรงๆ ไม่ต้องเดา
                if (
                    _PROFILE_MENU_LABEL_MARKER in (action_label or "")
                    and not goal_is_about_account
                    and tool_input.get("type") in ({"click"} | DEFAULT_NEEDS_CONFIRMATION)
                    and profile_menu_reject_count < _MAX_PROFILE_MENU_RETRIES
                ):
                    profile_menu_reject_count += 1
                    _bump_guard("profile_menu")  # W_token_cut W1
                    if verbose:
                        print(
                            f"[profile-menu {profile_menu_reject_count}/{_MAX_PROFILE_MENU_RETRIES}] "
                            f"ปฏิเสธการเปิดเมนูบัญชีบน goal ที่ไม่เกี่ยวกับบัญชี: {action_label!r}",
                            flush=True,
                        )
                    messages = append_tool_result(messages, tool_use_id, _PROFILE_MENU_NUDGE)
                    messages.append(_build_nudge_message(
                        resolved_provider, f"\u26a0\ufe0f [Important system command]: {_PROFILE_MENU_NUDGE}",
                    ))
                    continue

                # W_prefer_row_delete (ดูคอมเมนต์ที่จุดประกาศค่าคงที่): goal สั่งลบล้วนๆ แล้ว
                # โมเดลจะกด Edit ทั้งที่หน้านี้มีปุ่มลบให้กดอยู่แล้ว = เดินผิดทาง ไม่ใช่ทางผ่าน
                # ที่จำเป็น — หา element ที่ label บอกว่าเป็นการลบจาก snapshot ปัจจุบันตรงๆ
                # (ไม่ถาม LLM) ถ้าไม่มีเลยก็ปล่อยผ่านเหมือนเดิม รักษาเคสเว็บที่ต้องลบผ่านหน้า Edit
                if (
                    goal_is_deletion_only
                    and tool_input.get("type") == "click"
                    and action_label
                    and _ROW_ACTION_LABEL_RE.search(action_label)
                    and not _DESTRUCTIVE_LABEL_RE.search(action_label)
                    and prefer_row_delete_reject_count < _MAX_PREFER_ROW_DELETE_RETRIES
                ):
                    delete_label = next(
                        (
                            str(el.get("label") or "")
                            for el in (elements or [])
                            if _DESTRUCTIVE_LABEL_RE.search(str(el.get("label") or ""))
                        ),
                        "",
                    )
                    if delete_label:
                        prefer_row_delete_reject_count += 1
                        _bump_guard("prefer_row_delete")  # W_token_cut W1
                        nudge_text = _PREFER_ROW_DELETE_NUDGE_TEMPLATE.format(
                            label=action_label, delete_label=delete_label,
                        )
                        if verbose:
                            print(
                                f"[prefer-row-delete {prefer_row_delete_reject_count}/"
                                f"{_MAX_PREFER_ROW_DELETE_RETRIES}] {action_label!r} "
                                f"ทั้งที่มี {delete_label!r} ให้กดอยู่แล้ว",
                                flush=True,
                            )
                        messages = append_tool_result(messages, tool_use_id, nudge_text)
                        messages.append(_build_nudge_message(
                            resolved_provider, f"\u26a0\ufe0f [Important system command]: {nudge_text}",
                        ))
                        continue

                # W_no_record_edit_for_delete_goal (ดูคอมเมนต์เหนือ _RECORD_COMMIT_LABEL_RE
                # ด้านบนสำหรับบั๊กจริง): goal ที่สั่ง "ลบ" ล้วนๆ ต้องไม่กดบันทึกฟอร์มแก้ไข
                # เด็ดขาด — hard reject ไม่มีโควตา เหมือน W_no_create_for_existing_goal/
                # W_no_credential_flow เพราะเขียนทับ record จริงกู้คืนไม่ได้ ไม่ใช่แค่เสีย step
                #
                # goto ไม่อยู่ในชุด type โดยเจตนา (URL ไม่เคย commit ฟอร์ม และ URL ที่มีคำว่า
                # save เป็นของหน้า Add ซึ่ง W_no_create จับไปแล้ว) ส่วน DEFAULT_NEEDS_CONFIRMATION
                # ใส่ไว้ครอบ action type "submit" ที่โมเดลอาจเลือกแทน click
                if (
                    goal_is_deletion_only
                    and tool_input.get("type") in ({"click"} | DEFAULT_NEEDS_CONFIRMATION)
                    and _action_commits_a_record_edit(tool_input, action_label)
                ):
                    what = action_label.strip() or str(tool_input.get("type"))
                    nudge_text = _NO_RECORD_EDIT_NUDGE.format(what=what)
                    if verbose:
                        print(f"[no-record-edit] ปฏิเสธการบันทึกฟอร์มแก้ไขบน goal ที่สั่งลบ: {what!r}", flush=True)
                    messages = append_tool_result(messages, tool_use_id, nudge_text)
                    messages.append(_build_nudge_message(
                        resolved_provider, f"⚠️ [Important system command]: {nudge_text}",
                    ))
                    continue

                # W_no_create_for_existing_goal (ดู docstring ของ
                # _goal_targets_existing_records_only ด้านบนสำหรับบั๊กจริง): goal ที่พูดถึง
                # "ของที่มีอยู่" (ลบ/แก้ทั้งหมด) ต้องไม่แตะ flow สร้างรายการใหม่เด็ดขาด —
                # ปฏิเสธก่อน dispatch เหมือน pre-dispatch guard ตัวอื่นในไฟล์นี้ ไม่มีโควตา
                # ปล่อยผ่าน เพราะ "สร้างของใหม่แทนของที่หาไม่เจอ" คือความเสียหายที่กู้คืนยาก
                # (สร้าง record จริงในระบบจริง) ไม่ใช่แค่เสีย step เปล่า
                if (
                    goal_targets_existing_only
                    and tool_input.get("type") in ({"click", "goto"} | DEFAULT_NEEDS_CONFIRMATION)
                    and _action_starts_create_flow(tool_input, action_label)
                ):
                    what = action_label.strip() or str(tool_input.get("url") or tool_input.get("type"))
                    nudge_text = _NO_CREATE_NUDGE_TEMPLATE.format(
                        reason="a delete/edit goal, with no create instruction anywhere in it",
                        what=what,
                    )
                    if verbose:
                        print(f"[no-create] ปฏิเสธ action ที่พาไปสร้างรายการใหม่: {what!r}", flush=True)
                    messages = append_tool_result(messages, tool_use_id, nudge_text)
                    messages.append(_build_nudge_message(
                        resolved_provider, f"⚠️ [Important system command]: {nudge_text}",
                    ))
                    continue

                # W_goal_precheck ("Already-Achieved Pre-check" — โค้ดหนุนกฎ W19 "Log
                # Cleanliness" ใน SYSTEM_PROMPT ที่สั่งไว้แล้วว่า "ห้ามคลิก element ที่มี marker
                # [already active] ซ้ำ"): marker นี้ perception.py คำนวณจาก DOM จริง เป็น
                # deterministic signal อยู่แล้ว — คลิกซ้ำไม่มีผลอะไรแน่นอน (โครงสร้างหน้าเหมือน
                # เดิมทุกตัวอักษร) ไม่มีเหตุผลจะพึ่ง prompt compliance อย่างเดียวกับสิ่งที่เช็คใน
                # โค้ดได้ถูกๆ แบบนี้
                #
                # บั๊กจริงที่ทำให้ต้องเพิ่ม (live-reproduce บน OrangeHRM กับ provider openai,
                # goal "login then goto adminmenu"): agent ไปถึงหน้า Admin ตั้งแต่ step 1-2 จริง
                # (label ขึ้น "Admin [already active]") แต่ยังคลิกซ้ำอีก 5 ครั้งจนหมด max_steps
                # และระหว่างนั้นเผลอไปกด Edit/Save บนข้อมูลจริงของระบบ — เสีย step ไปเปล่าๆ
                # และเสี่ยงแก้ข้อมูลที่ goal ไม่ได้สั่งเลย
                #
                # มีโควตาเหมือน guard อื่นในไฟล์นี้: เกินแล้วปล่อย dispatch จริงแทนที่จะบล็อก
                # ต่อไม่รู้จบ (คลิก element ที่ active อยู่แล้วเป็น no-op ปลอดภัยกว่าค้างทั้ง task)
                if (
                    tool_input.get("type") == "click"
                    and "[already active]" in action_label
                    and consecutive_already_active_skip_count < _MAX_ALREADY_ACTIVE_SKIP_RETRIES
                ):
                    consecutive_already_active_skip_count += 1
                    _bump_guard("already_active_skip")  # W_token_cut W1
                    if verbose:
                        print(
                            f"[already-active {consecutive_already_active_skip_count}/"
                            f"{_MAX_ALREADY_ACTIVE_SKIP_RETRIES}] ข้ามการคลิกซ้ำ label={action_label!r}",
                            flush=True,
                        )
                    messages = append_tool_result(
                        messages, tool_use_id,
                        "[Skipped] This element is already active/selected (marked "
                        '"[already active]") — clicking it again would have no effect. That '
                        "part of the goal is ALREADY DONE. Read what is on the page right now: "
                        "if it satisfies the goal, call finish_task with that as your evidence; "
                        "otherwise move on to a genuinely different action toward the goal.",
                    )
                    continue

                # W_filter_scope_guard: goal ระบุเงื่อนไขไว้ชัดเจนแบบ field=value แล้ว
                # การไปตั้งค่า filter *ช่องอื่น* ที่ goal ไม่เคยพูดถึงไม่ใช่แค่เสียเวลา —
                # มันกรองแถวที่ user ต้องการออกจากตารางไปด้วย (บั๊กจริง live run 2026-08-27:
                # goal บอกแค่ userrole=ess แต่ agent ไปตั้ง Status=Enabled ด้วย ทำให้ ESS ที่
                # ถูก disable หายไปจากตาราง แล้ว "ลบให้หมด" จะลบไม่ครบโดยที่ทุกฝ่ายเข้าใจว่าครบ)
                #
                # ทำได้ deterministic ล้วนๆ เพราะ perception ใส่ชื่อ field เป็น prefix ให้
                # dropdown trigger อยู่แล้ว ("Status: Enabled") ไม่ต้องถาม LLM ว่าช่องนี้คือช่องอะไร
                #
                # ใช้โควตาไม่ใช่บล็อกตาย: บางเว็บบังคับให้ต้องเลือกค่าบางช่องก่อนถึงจะกด Search
                # ได้จริง ถ้าบล็อกตายจะทำให้เว็บกลุ่มนั้นใช้งานไม่ได้เลย
                #
                # W_filter_scope_via_dropdown (live 2026-09-01: agent ตั้ง Status=Enabled ได้
                # ทั้งที่ goal บอกแค่ userrole=ess — trace: press_key(Status: -- Select --)
                # แล้วตามด้วย click(Enabled)): ลิสต์ชนิด action เดิมมีแค่ fill/select/click
                # จึงข้าม press_key ที่เลือกค่าใน dropdown ได้จริง และ check ที่ติ๊ก filter
                # แบบ checkbox/toggle ได้ — ต้องครอบ *ทุก* ชนิดที่ตั้งค่า filter ได้
                #
                # อีกทางที่เคยสงสัยว่าหลุด คือ "กดตัวเลือกในลิสต์" (label เป็นแค่ "Enabled"
                # ไม่มีชื่อ field นำหน้า guard จึงอ่านไม่ออก) — ตรวจแล้วว่าปิดเองโดยอัตโนมัติ
                # เมื่อครอบ trigger ครบทุกชนิด: ลิสต์ตัวเลือกจะเปิดขึ้นมาได้ก็ต่อเมื่อ action ที่
                # เปิดมันผ่าน guard นี้ไปแล้ว ซึ่งแปลว่าช่องนั้นอยู่ในขอบเขต goal หรือโควตาหมด
                # ไปแล้วทั้งคู่ การไล่ตามหา "ตัวเลือกของ trigger ตัวไหน" จึงเป็นโค้ดที่ยิงไม่ได้จริง
                touched_field = _filter_field_from_label(
                    action_label or "", str(tool_input.get("type") or ""),
                )
                # W_filter_already_satisfied: ตัวกรองตัวนี้ถือค่าที่ goal ขอไว้อยู่แล้ว การกด
                # ซ้ำจึงเป็น no-op — เป็นวงวนที่ guard เดิมมองไม่เห็น (ดู scratchpad/บันทึกของ
                # W_filter_already_satisfied: loop detector เทียบ index ที่เปลี่ยนทุก snapshot
                # และตัวนับ label ซ้ำตั้งไว้ที่ 4 เพราะ "Select row" ต้องกดซ้ำหลายแถวได้จริง)
                # อ่านค่าจาก label ที่ perception เติมให้อยู่แล้ว ไม่ต้องถาม LLM และไม่ต้องอ่าน DOM
                satisfied_pair = None
                if touched_field and tool_input.get("type") in ("click", "press_key"):
                    _current_value = _filter_value_from_label(action_label or "")
                    if _current_value:
                        satisfied_pair = next(
                            (
                                (f, v) for f, v in goal_condition_pairs
                                if _field_names_match(f, touched_field)
                                and _cell_matches_value(_current_value, v)
                            ),
                            None,
                        )
                if satisfied_pair and filter_satisfied_reject_count < _MAX_FILTER_SATISFIED_RETRIES:
                    filter_satisfied_reject_count += 1
                    nudge_text = _FILTER_SATISFIED_NUDGE_TEMPLATE.format(
                        label=action_label,
                        field=satisfied_pair[0],
                        value=satisfied_pair[1],
                        count=filter_satisfied_reject_count,
                    )
                    if verbose:
                        print(
                            f"[filter ถูกตั้งไว้แล้ว {filter_satisfied_reject_count}/"
                            f"{_MAX_FILTER_SATISFIED_RETRIES}] label={action_label!r}",
                            flush=True,
                        )
                    messages = append_tool_result(messages, tool_use_id, nudge_text)
                    messages.append(_build_nudge_message(
                        resolved_provider, f"\u26a0\ufe0f [Important system command]: {nudge_text}",
                    ))
                    continue

                if (
                    goal_filter_fields
                    and touched_field
                    and not any(_field_names_match(f, touched_field) for f in goal_filter_fields)
                    and tool_input.get("type") in (
                        "fill", "select", "click", "press_key", "check",
                    )
                    and filter_scope_reject_count < _MAX_FILTER_SCOPE_RETRIES
                ):
                    filter_scope_reject_count += 1
                    _bump_guard("filter_scope")  # W_token_cut W1
                    nudge_text = _FILTER_SCOPE_NUDGE_TEMPLATE.format(
                        label=action_label,
                        allowed=", ".join(goal_filter_fields),
                    )
                    if verbose:
                        print(
                            f"[filter นอกขอบเขต {filter_scope_reject_count}/"
                            f"{_MAX_FILTER_SCOPE_RETRIES}] label={action_label!r} "
                            f"field={touched_field!r} allowed={goal_filter_fields}",
                            flush=True,
                        )
                    messages = append_tool_result(messages, tool_use_id, nudge_text)
                    messages.append(_build_nudge_message(
                        resolved_provider, f"\u26a0\ufe0f [Important system command]: {nudge_text}",
                    ))
                    continue

                # W64[7.1] ("Filter Order & False Completion" — ดู docstring เต็มของ
                # _ROW_ACTION_LABEL_RE ด้านบนสุดของไฟล์): บล็อกการคลิกปุ่ม row-action
                # (Edit/View/Delete/Download) ทันทีถ้า step ก่อนหน้าคือ fill/select ที่สำเร็จ
                # โดยยังไม่ได้กด Search/ค้นหา/Enter ยืนยัน filter นั้นเลย — เช็คก่อน dispatch
                # จริง (เหมือน login-password guard ด้านบน) ไม่ผ่าน RAG/middleware evaluator
                # ที่พึ่ง LLM เพราะนี่คือ deterministic state ล้วนๆ ไม่ต้องเดา
                if (
                    filter_dirty_since_search
                    and tool_input.get("type") in ({"click"} | DEFAULT_NEEDS_CONFIRMATION)
                    and action_label
                    and _ROW_ACTION_LABEL_RE.search(action_label)
                ):
                    if premature_row_action_before_search_count < _MAX_PREMATURE_ROW_ACTION_BEFORE_SEARCH_RETRIES:
                        premature_row_action_before_search_count += 1
                        _bump_guard("row_action_before_search")  # W_token_cut W1
                        if verbose:
                            print(
                                f"[row-action ก่อน search {premature_row_action_before_search_count}/"
                                f"{_MAX_PREMATURE_ROW_ACTION_BEFORE_SEARCH_RETRIES}] label={action_label!r} "
                                f"prev_field={last_filter_field_label!r}",
                                flush=True,
                            )
                        nudge_text = _PREMATURE_ROW_ACTION_BEFORE_SEARCH_NUDGE_TEMPLATE.format(
                            prev_label=last_filter_field_label or "(field name unknown)", label=action_label,
                        )
                        messages = append_tool_result(messages, tool_use_id, nudge_text)
                        messages.append(_build_nudge_message(resolved_provider, f"⚠️ [Important system command]: {nudge_text}"))
                        filter_dirty_since_search = False
                        continue
                    # เกินโควตาเตือนแล้วยังไม่ยอมกด Search ก่อน ปล่อยผ่านไปตามที่โมเดลเลือก
                    # แทนที่จะค้างไม่รู้จบ (escape valve เดียวกับ guard อื่นในไฟล์นี้)

                # W_delete_all_intent (safety gate สำคัญที่สุดของชุดนี้): ก่อน click ที่ทำลาย
                # ข้อมูล "ครั้งแรก" ของ goal ลบแบบมีเงื่อนไข ต้องเห็นก่อนว่าตารางที่กำลังจะลบ
                # ออกมานั้นกรองตรงเงื่อนไขจริงแล้ว — อ่าน DOM ตรงๆ (ไม่ถาม LLM, ไม่เชื่อคำ
                # อธิบายของมัน) แล้วเทียบกับค่าที่ user เขียนไว้ใน goal เอง
                #
                # นี่คือสถานการณ์ที่ W21 เขียนมาป้องกันโดยตรงและเป็นความเสียหายจริงที่เกิดไป
                # แล้ว 1 ครั้งบนเดโมสาธารณะ: agent ลบแถวจากตารางที่ยังไม่ได้กรอง = ลบข้อมูล
                # ที่ goal ไม่เคยสั่งให้แตะ ซึ่งกู้คืนไม่ได้ ต่างจาก guard อื่นในไฟล์นี้ที่แค่
                # เสีย step เปล่า
                #
                # เจตนาคือบังคับ "ลำดับ" (ต้องกรองก่อนลบ) ไม่ใช่ "วิธี" — จะลบด้วย Select All
                # หรือลบทีละแถวก็ได้ทั้งคู่ ตราบใดที่ตารางที่เห็นตรงเงื่อนไขแล้ว
                if (
                    delete_all_condition_values
                    and not destructive_filter_verified
                    and (
                        tool_input.get("type") in DEFAULT_NEEDS_CONFIRMATION
                        or (
                            tool_input.get("type") == "click" and action_label
                            and _DESTRUCTIVE_LABEL_RE.search(action_label)
                        )
                    )
                ):
                    row_match = await _count_rows_matching_condition(page, delete_all_condition_pairs)
                    condition_text = " + ".join(repr(v) for v in delete_all_condition_values)
                    if (
                        row_match is not None and row_match[0] < row_match[1]
                        and premature_destructive_before_filter_count < _MAX_DESTRUCTIVE_BEFORE_FILTER_RETRIES
                    ):
                        premature_destructive_before_filter_count += 1
                        _bump_guard("destructive_before_filter")  # W_token_cut W1
                        matching, total = row_match
                        if verbose:
                            print(
                                f"[ลบก่อนกรอง {premature_destructive_before_filter_count}/"
                                f"{_MAX_DESTRUCTIVE_BEFORE_FILTER_RETRIES}] label={action_label!r} "
                                f"ตรงเงื่อนไข {matching}/{total} แถว",
                                flush=True,
                            )
                        nudge_text = _DESTRUCTIVE_BEFORE_FILTER_NUDGE_TEMPLATE.format(
                            condition=condition_text, matching=matching, total=total,
                        )
                        messages = append_tool_result(messages, tool_use_id, nudge_text)
                        messages.append(_build_nudge_message(
                            resolved_provider, f"⚠️ [Important system command]: {nudge_text}",
                        ))
                        continue
                    # ตารางตรงเงื่อนไขครบแล้ว / เช็คไม่ได้ (หน้านี้ไม่มีตาราง — fail-safe
                    # เหมือน guard อื่นในไฟล์นี้: เช็คไม่ได้ ดีกว่าบล็อก action ที่อาจถูกอยู่
                    # แล้ว) / หมดโควตาเตือน — ปล่อยผ่านและไม่เช็คซ้ำอีกตลอด task นี้ (การลบแถว
                    # ถัดๆ ไปทำกับตารางชุดเดียวกันที่เพิ่งยืนยันไปแล้ว)
                    destructive_filter_verified = True

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
                        else f"<{action_tag}>" if action_tag else "(label unknown)"
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
                            f"[The system skipped this step automatically] {skip_reason} "
                            "Choose a different action that genuinely advances the goal instead.",
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
                            f"[The system skipped this step automatically] {skip_reason} "
                            "Choose a different action that genuinely advances the goal instead.",
                        )
                        continue

                # W30: เก็บ URL ก่อน dispatch action ไว้เทียบหลัง action จบ (ดู
                # url_changed_unexpectedly ด้านล่าง) — เฉพาะ action ที่ "ไม่ได้ตั้งใจจะ
                # navigate" เอง (goto/switch_tab/go_back คือจุดประสงค์หลักคือเปลี่ยนหน้า
                # อยู่แล้ว ไม่ต้องเตือนซ้ำ)
                url_before_action = page.url

                # W_tab_rebind: ต้องเก็บ "ก่อน" ไว้เทียบ ไม่งั้นแยกแท็บที่ action นี้เพิ่งเปิด
                # ออกจากแท็บที่ user เปิดค้างไว้เองตั้งแต่ก่อนเริ่ม task ไม่ได้
                try:
                    tabs_before_action = list(page.context.pages)
                except Exception:
                    tabs_before_action = []
                _action_started_at = time.monotonic()
                action_calls += 1  # W_token_cut W1
                result: ActionResult = await execute(
                    page, tool_input, ask_user_func=ask_user_func, label=action_label,
                    manual_guidance=manual_permission_guidance, allowed_domains=effective_allowed_domains,
                    element_tag=action_tag, element_type=action_element_type,
                    then_label=then_label, then_tag=then_tag, then_type=then_element_type,
                )
                # W_step_trace: เวลาที่ใช้กับ browser จริงของ step นี้ (รวม retry ภายใน
                # actions.py และ wait ต่างๆ ที่ execute() ทำเอง)
                step_action_seconds = time.monotonic() - _action_started_at

                # W_retry_value_has_no_home: จำ label ของช่องที่ agent กรอกค่าเอง ไว้บอก user
                # ตอนขอค่าใหม่ว่าจะเอาไปแทนที่ตรงไหน — เก็บเฉพาะ fill ธรรมดา เพราะ fill_secret
                # คือรหัสปัจจุบันที่ระบบกรอกให้เอง ไม่ใช่ค่าที่ user จะเปลี่ยน
                if result.success and tool_input.get("type") in _VALUE_WRITING_ACTION_TYPES:
                    wrote_a_value_this_task = True  # W_verify_text_needs_a_write
                if (
                    result.success
                    and tool_input.get("type") == "fill"
                    and action_label
                    and action_label not in agent_filled_field_labels
                ):
                    agent_filled_field_labels.append(action_label)

                # W_tab_rebind: เปลี่ยน page ที่ลูปถืออยู่ *ก่อน* อย่างอื่นจะอ่านหน้าเว็บต่อ
                # (W30 url-changed check, wait_stable, get_snapshot ของ step ถัดไป) — ผูก
                # dialog handler ให้แท็บใหม่ด้วย ไม่งั้น alert() บนแท็บนั้นจะค้างไม่มีใครปิด
                # (ดู _make_dialog_handler) แล้วทุก action หลังจากนั้น timeout เงียบๆ
                _switched_page, _tab_note = _detect_tab_switch(page, tabs_before_action, tool_input)
                if _switched_page is not page:
                    page = _switched_page
                    page.on("dialog", _make_dialog_handler(self.memory, verbose))
                    if verbose:
                        print(f"[tab-rebind] {_tab_note.strip()}", flush=True)
                # W21: เก็บผลลัพธ์ของ action นี้ไว้ให้ loop-guard ตอนต้น step ถัดไปเช็ค
                # is_bulk_safe_repeat (ดู docstring ตรงจุดเช็คด้านบน)
                last_action_succeeded = result.success
                steps_taken += 1

                # W_guard_quota_reset: ตัวนับ premature_* ทุกตัว (และ already-active) เดิมมีแต่
                # `+= 1` ไม่มีจุด reset ที่ไหนเลยทั้งไฟล์ — โควตา 2 ครั้งจึงเป็น "โควตาตลอดทั้ง
                # task" ไม่ใช่ "โควตาต่อครั้งที่โมเดลหลงทาง" ผลคือ guard กัน false-completion
                # ทุกตัวตายถาวรตั้งแต่กลาง task เป็นต้นไป: งานยาวๆ ที่โมเดลเผลอ claim สำเร็จ
                # ตอนต้น 2 ครั้ง จะไม่มีอะไรกันการ claim สำเร็จผิดๆ ในตอนท้ายอีกเลย ทั้งที่นั่น
                # คือจุดที่ verification สำคัญที่สุด
                #
                # reset เมื่อมี action ที่ "สำเร็จจริง" คั่น (result.success) เท่านั้น — นั่นคือ
                # นิยามของความคืบหน้าที่เชื่อถือได้ที่สุดที่มีในลูปนี้ ไม่ใช่แค่ "ยิง action ไป"
                # (action ที่ล้มเหลวไม่ควรคืนโควตาให้ ไม่งั้นวนขอ nudge ได้ไม่จำกัด) — pattern
                # เดียวกับที่ consecutive_same_label_count/consecutive_fill_secret_context_reject_count
                # ทำถูกอยู่แล้ว เพดานรวมยังคุมด้วย max_steps และ loop-detection เหมือนเดิมทุกประการ
                if result.success:
                    premature_false_finish_count = 0
                    premature_true_finish_count = 0
                    premature_all_failed_count = 0
                    premature_login_skip_count = 0
                    premature_validation_error_count = 0
                    premature_deletion_incomplete_count = 0
                    premature_table_verify_count = 0
                    premature_row_action_before_search_count = 0
                    consecutive_already_active_skip_count = 0

                # W64[7.1]: อัปเดต filter_dirty_since_search — True เฉพาะตอน fill/select
                # สำเร็จรอบนี้เท่านั้น (reset เป็น False เสมอไม่ว่า action อื่นจะเป็นอะไร รวมถึง
                # ตอน guard ด้านบนเพิ่งบล็อกไปเอง — ดู docstring ของ _ROW_ACTION_LABEL_RE
                # สำหรับเหตุผลที่ scope แคบแค่ 1 step)
                #
                # W_dropdown_sets_filter_dirty (บั๊กจริง live-reproduce บน OrangeHRM ผ่าน step
                # trace 2026-08-26: agent เปลี่ยน User Role เป็น ESS แล้วกด Edit ต่อทันทีโดย
                # ไม่เคยกด Search เลย และไม่เคยถูก nudge สักครั้ง): เงื่อนไข fill/select เดิม
                # ไม่มีทางเป็นจริงบน custom dropdown เลย เพราะ W50 + state_filter.
                # check_select_target_is_native() *บังคับ* ให้โมเดลใช้ "click" กับ dropdown
                # พวกนี้ — แถม else ด้านล่างยัง reset ธงเป็น False ด้วยการคลิกตัวเลือกนั้นเอง
                # ผลคือ guard "ห้ามคลิก row action ก่อนกด Search" ตายสนิทบนเว็บ SPA สมัยใหม่
                # แทบทั้งหมด ซึ่งเป็นกลุ่มที่ต้องการ guard นี้ที่สุด — นับ click ที่ actions
                # ยืนยันว่าเป็นการ "เลือกตัวเลือกใน dropdown" จริง (ดู ActionResult.
                # dropdown_option_selected) เป็น filter change เท่ากับ fill/select
                if result.success and (
                    tool_input.get("type") in ("fill", "select") or result.dropdown_option_selected
                ):
                    filter_dirty_since_search = True
                    last_filter_field_label = action_label
                    filter_changed_without_search = True
                else:
                    filter_dirty_since_search = False

                # W_delete_all_intent: ธง sticky ตัวนี้ล้างได้ทางเดียวเท่านั้น — กด Search
                # สำเร็จจริง (_SEARCH_LABEL_RE ครอบคลุมไทย/อังกฤษ เดิมประกาศไว้แต่ไม่เคยถูกใช้
                # เลยสักที่) ไม่ล้างตาม action อื่นเหมือน filter_dirty_since_search เพราะ
                # คำถามที่มันตอบคือ "ตลอด task นี้ เคยกรองจริงหรือยัง" ไม่ใช่ "step ที่แล้วทำอะไร"
                if result.success and action_label and _SEARCH_LABEL_RE.search(action_label):
                    filter_changed_without_search = False

                # W64[7.2]: บันทึกว่า task นี้เคยมี action ที่ toast ยืนยันสำเร็จจริงแล้วหรือยัง
                # (ดู actions.py::ActionResult.toast_confirmed) — ใช้ตัดสิน leniency ของ
                # table-verify guard ด้านล่าง (ครั้งเดียวพอ ไม่ต้อง reset กลับ False เพราะ
                # "เคยยืนยันสำเร็จแล้วอย่างน้อย 1 ครั้งใน task นี้" ยังเป็นความจริงตลอดไปไม่ว่า
                # action ถัดๆ ไปจะเป็นอะไรก็ตาม)
                if result.toast_confirmed:
                    any_toast_confirmed_this_task = True
                # W_count_answer_check: ดูดตัวเลขที่ _deterministic_count_note() นับไว้ออกจาก
                # ข้อความผลลัพธ์ (read_page_data เท่านั้นที่มีบรรทัดนั้น — action อื่นคืน {})
                if result.success:
                    system_counted.update(system_counted_conditions(result.message))
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
                result_text = str(result) + _tab_note
                # W_plan_panel_lags_the_log: อ่านครั้งเดียว ใช้สองที่ (โน้ต W30 ด้านล่าง และ
                # การขยับ cursor แสดงผล) ไม่เพิ่มการอ่าน DOM
                url_changed_this_action = page.url != url_before_action
                if tool_input.get("type") not in ("goto", "switch_tab", "go_back") and url_changed_this_action:
                    result_text += (
                        f"\n[The page changed by itself after this action: from "
                        f"{url_before_action} to {page.url} — the earlier plan may no longer "
                        "match the current page; check this new page's indexed elements before "
                        "deciding your next action]"
                    )
                # W10[D]: แนบ label ของ element เป้าหมาย (ชื่อปุ่ม/ช่องกรอกจริงบนหน้าเว็บ
                # เช่น "Login", "Username" — มาจาก perception.py::get_snapshot() ตัวเดียว
                # กับที่ action_label ด้านบนใช้เช็ค permission อยู่แล้ว) เข้า history/event
                # ด้วย ให้ UI (Log panel) โชว์ชื่อจริงแทน index เปล่าๆ ที่มนุษย์อ่านไม่รู้
                # เรื่องว่ากดอะไร/กรอกช่องไหน — ไม่มีผลกับ dispatch จริง (ยังใช้ tool_input
                # เดิมเป๊ะ) แค่ข้อมูลเสริมไว้แสดงผล
                # W_timing_gap: ถือ reference ของ dict ไว้ (ShortTermMemory.record เก็บ object
                # ตัวเดียวกัน ไม่ได้ copy) เพื่อเติมเวลา wait_stable ลงไปทีหลังได้ — wait_stable
                # เกิดหลังบันทึก step ไปแล้ว จะย้ายการบันทึกไปไว้ทีหลังไม่ได้เพราะ guard หลายตัว
                # ด้านล่างอ่าน memory ของ step นี้
                step_record = {
                    "step": steps_taken,
                    "cmd": tool_input,
                    "label": action_label,
                    # W_click_navigated: บันทึก result_text (ตัวเดียวกับที่ส่งให้ LLM
                    # ด้านบน) ไม่ใช่ str(result) เปล่าๆ — ShortTermMemory.failed_actions_
                    # summary() (ดู core/memory.py) ยัดบรรทัด "[FAIL] click(3)" เข้า prompt
                    # ทุก step ที่เหลือของ task ถ้าเก็บแต่ข้อความดิบ โน้ต W30 "หน้าเปลี่ยนไป
                    # เอง" ที่คำนวณไว้แล้วจะหายไปจาก memory ทั้งที่เป็นหลักฐานชิ้นเดียวที่
                    # อธิบายว่า action นั้นได้ผลจริง — memory poisoning ที่ขยายผลบั๊ก
                    # W_click_navigated ให้ลามไปทั้ง run
                    "result": result_text,
                    "success": result.success,
                    "tokens": _tokens_dict(usage),
                    # W_procmem: locator ที่ "อยู่รอด" ข้าม task run ได้ (ดู
                    # core/dom_locator.py/actions.py) — None เสมอยกเว้น
                    # click/fill/select/check ที่สำเร็จจริง ใช้เป็น input ของ
                    # llm.abstract_trajectory() ตอน task จบสำเร็จ (ดูจุดเรียกด้านล่าง)
                    "locator_descriptor": result.locator_descriptor,
                    # W_step_trace: 3 field ใหม่ ไว้ให้ task_manager เขียนลง
                    # settings.step_trace_log_path ตอน task จบ (ดู config.py ที่ค่านั้น) —
                    # ไม่ถูกใช้ตัดสินใจอะไรในลูปเลย เป็นข้อมูลวินิจฉัยล้วนๆ
                    "failure_class": _classify_step_failure(str(result), result.success),
                    "timing": {
                        "snapshot": round(step_snapshot_seconds, 3),
                        "llm": round(step_llm_seconds, 3),
                        "action": round(step_action_seconds, 3),
                        # W_timing_gap: pacing วัดได้ตั้งแต่ต้น step ส่วน wait เติมทีหลัง
                        # (wait_stable เกิดหลังจุดนี้ — ดูด้านล่าง)
                        "pacing": round(step_pacing_seconds, 3),
                        "wait": 0.0,
                    },
                }
                self.memory.record(step_record)
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
                    "llm_calls": llm_turns,
                })

                # W43: LLM ระบุว่า action นี้ทำให้ step ของแผนเสร็จสมบูรณ์แล้ว (ดู
                # completed_plan_step ใน llm.py::_BROWSER_ACTION_PARAMS) — ยิง SSE event
                # ใหม่ให้ frontend ติ๊ก checkbox ของ step นั้นแบบ real-time เฉพาะตอน
                # execute() สำเร็จจริงเท่านั้น (result.success — กันติ๊กผิดว่าทำสำเร็จ
                # ทั้งที่ action พัง) และเฉพาะ task ที่มีแผนจริงๆ เท่านั้น (plan_text ไม่ใช่
                # None/ว่างเปล่า — ad-hoc task ไม่มีแผนไม่ควรยิง event นี้เลยแม้ LLM จะใส่
                # completed_plan_step มาผิดๆ ก็ตาม เพราะไม่มี checkbox ให้ติ๊กอยู่แล้วฝั่ง UI)
                completed_plan_step = tool_input.get("completed_plan_step")
                # W_plan_step_cursor: การที่โมเดลรายงานเลขมา = หลักฐานว่า *หนึ่ง* step จบ
                # ไม่ใช่ว่าทุกข้อจนถึงเลขนั้นจบ — cursor จึงเดินหน้าได้ทีละ 1 เสมอ ต่อให้
                # โมเดลส่ง 5 มาตอนที่ยังอยู่ข้อ 1 (และไม่ถอยหลังเด็ดขาด) เลขที่ยิงออก SSE
                # จึงเป็น cursor จริง ไม่ใช่เลขที่โมเดลอ้าง — Test Console จะติ๊กช้าลงแต่ตรง
                # กับงานที่ทำจริง
                # เงื่อนไข action type ใช้ชุดเดียวกับ W_goal_scope_false_success ด้านล่าง
                # ไม่สร้างชุดใหม่ซ้อน (อ่านอย่างเดียวไม่ทำให้ step ไหน "เสร็จ" ได้จริง)
                # W_plan_progress_stall: นิยาม "action ที่ควรทำให้แผนคืบ" ใช้ชุดเดียวกับที่
                # cursor ใช้ (ไม่อยู่ใน _GOAL_SCOPE_ALLOWED_ACTION_TYPES = ไม่ใช่การอ่านเฉยๆ) —
                # ในไฟล์นี้มีนิยาม "mutating" อยู่ 2 ชุดแล้วและต่างกัน อย่าสร้างชุดที่สาม
                plan_progressing_action = (
                    bool(plan_text)
                    and result.success
                    and tool_input.get("type") not in _GOAL_SCOPE_ALLOWED_ACTION_TYPES
                )
                # W_plan_cursor_needs_a_matching_action: completed_plan_step เป็น self-report
                # ล้วนๆ และ W111 วัดมาแล้วว่าโมเดลแนบมากับแทบทุก action — cursor จึงวิ่งจนแผน
                # ติ๊กครบก่อนงานจริงจะเสร็จ (user รายงานพร้อมภาพหน้าจอ 2026-09-03: แผนติ๊กครบ
                # 5 ข้อขณะที่ยังลบไม่เสร็จ) ใช้ตัวเทียบตัวเดียวกับ W111 เป็นประตู: ถ้า action
                # ที่เพิ่งทำ "ไม่ตรงกับ step ปัจจุบัน" อย่างชัดเจน (False) ห้ามเดินหน้า cursor
                # ส่วน None = ตัดสินไม่ได้ ยังปล่อยผ่านเหมือนเดิม เพราะตัวเทียบนี้เป็นสัญญาณอ่อน
                # โดยเจตนา (step ที่ LLM เขียนกว้าง/กำกวมได้เสมอ — ดู docstring ของมัน)
                # การบล็อกทุกเคสที่อ่านไม่ออกจะทำให้แผนไม่มีวันติ๊กเลยบนงานจริงส่วนใหญ่
                _cursor_step_text = ""
                if plan_text:
                    _cursor_steps = _plan_step_lines(plan_text)
                    if 0 < plan_cursor <= len(_cursor_steps):
                        _cursor_step_text = _cursor_steps[plan_cursor - 1]
                _cursor_action_matches = _action_matches_plan_step(
                    _cursor_step_text, action_label or "", str(tool_input.get("type") or ""),
                ) if _cursor_step_text else None

                # W_plan_cursor_needs_a_matching_action (แก้รอบสอง — บั๊กจริงที่ user เจอทันที
                # หลังรอบแรก 2026-09-03): รอบแรกบล็อกการเดินหน้าทุกครั้งที่ตัวเทียบตอบ "ไม่ตรง"
                # ซึ่งกลายเป็น **ล็อกตาย** — พอ cursor ไม่ขยับ step ปัจจุบันก็ยังเป็นข้อเดิม
                # action ถัดไปก็ยิ่งไม่ตรงกับข้อเดิมนั้น วนแบบนี้ตลอด task (หน้าจอ: แผนค้างอยู่
                # ข้อ 1 หมุนยาวๆ ทั้งที่ log เดินไปถึง step 13 แล้ว)
                #
                # ต้นเหตุที่แท้จริงคือเอาสัญญาณที่ docstring ของ _action_matches_plan_step()
                # เขียนไว้เองว่าเป็น "สัญญาณอ่อน ห้ามเอาไปบล็อก" มาใช้เป็นประตูแข็ง — ใช้เป็น
                # ตัวหน่วงแทน: บล็อกได้ไม่เกิน _MAX_BLOCKED_CURSOR_ADVANCES ครั้งติดกัน แล้ว
                # ปล่อยผ่าน ทำให้ยังกันโมเดลไล่ติ๊กแผนรวดเดียวได้ (ปัญหาเดิม) โดยไม่มีทางค้าง
                if plan_progressing_action and completed_plan_step is not None:
                    if (
                        _cursor_action_matches is False
                        and blocked_cursor_advances < _MAX_BLOCKED_CURSOR_ADVANCES
                    ):
                        blocked_cursor_advances += 1
                        actions_since_plan_progress += 1
                    else:
                        blocked_cursor_advances = 0
                        plan_cursor = min(plan_cursor + 1, _total_plan_steps(plan_text) + 1)
                        actions_since_plan_progress = 0
                elif plan_progressing_action:
                    actions_since_plan_progress += 1

                # W_plan_panel_lags_the_log: ขยับ cursor แสดงผลจากหลักฐานบนหน้าเว็บ แล้วยิง
                # ความคืบหน้า *ทุก action* — จังหวะเดียวกับ event "step" ที่ทำให้ LOG ทัน สอง
                # panel จึงเดินคู่กันโดยโครงสร้าง ไม่มีทางหลุดจากกันได้
                #
                # ยิงทุกครั้งแม้ไม่มีความคืบหน้า เพราะแถว "กำลังทำข้อนี้" ต้องรีเฟรชให้ตรงเสมอ
                # และ event นี้พา done_through ไปด้วย ทำให้ tab ที่เพิ่งเชื่อมสายกลางคัน
                # กู้ติ๊กที่พลาดไปได้ครบ (ระบบไม่มี replay buffer — ดู routes.py::_stream_task_events)
                #
                # ค่าใช้จ่ายอยู่บน SSE ล้วน ไม่มีอะไรเข้า prompt ของ LLM
                if plan_text:
                    _plan_total = _total_plan_steps(plan_text)
                    _plan_lines = _plan_step_lines(plan_text)
                    _display_text = (
                        _plan_lines[display_plan_cursor - 1]
                        if 0 < display_plan_cursor <= len(_plan_lines) else ""
                    )
                    _display_evidence = _display_step_evidence(
                        _display_text, action_label or "", str(tool_input.get("type") or ""),
                        url_changed_this_action, bool(result.success),
                    )
                    # cap ที่ _plan_total ไม่ใช่ _plan_total + 1 — ข้อสุดท้ายยังต้องรอ task สำเร็จ
                    if _display_evidence and display_plan_cursor < _plan_total:
                        display_plan_cursor += 1
                    # floor: ฝั่งแสดงผลต้องไม่ตามหลัง cursor ตัวอนุรักษ์นิยม
                    display_plan_cursor = max(
                        display_plan_cursor, min(plan_cursor, _plan_total),
                    )
                    await _emit({
                        "kind": "plan_progress",
                        "current": display_plan_cursor,
                        "done_through": display_plan_cursor - 1,
                        "total": _plan_total,
                        "confirmed": min(plan_cursor - 1, _plan_total),
                        "evidence": _display_evidence or "none",
                        "url": page.url,
                    })

                # W_action_matches_plan_step: เทียบกับ step ที่กำลังทำอยู่ *หลัง* cursor ขยับแล้ว
                # (ถ้า cursor เพิ่งขยับ แปลว่า step ปัจจุบันคือข้อถัดไป ซึ่งเป็นข้อที่ต้องเทียบจริง
                # ในเทิร์นหน้า) — รีเซ็ตเฉพาะตอน "ตรง" เท่านั้น ไม่ใช่ตอน cursor ขยับ
                if plan_progressing_action:
                    # W_action_matches_plan_step: cursor วิ่งเลยข้อสุดท้ายได้ (โมเดลใส่
                    # completed_plan_step มาทุก action) — ตรึงไว้ที่ข้อสุดท้ายแทนที่จะปล่อยให้
                    # ไม่มี step ให้เทียบ ซึ่งจะทำให้ guard เงียบพอดีในเคสที่ต้องการมันที่สุด
                    # (แผน "เสร็จ" ตามตัวนับแล้วแต่ agent ยังทำอะไรอยู่ = ต้องเกี่ยวกับข้อสุดท้าย)
                    _steps_now = _plan_step_lines(plan_text)
                    _current_now = (
                        _steps_now[min(plan_cursor, len(_steps_now)) - 1] if _steps_now else ""
                    )
                    _matched = _action_matches_plan_step(
                        _current_now, action_label or "", str(tool_input.get("type") or ""),
                    )
                    if _matched is True:
                        actions_mismatching_plan_step = 0
                    elif _matched is False:
                        actions_mismatching_plan_step += 1
                if plan_text and result.success and completed_plan_step is not None:
                    await _emit({"kind": "plan_step_done", "step": min(plan_cursor - 1, _total_plan_steps(plan_text))})
                    # W_goal_scope: step สุดท้ายของแผนเสร็จแล้ว = goal ถือว่าสำเร็จ ใช้เป็น
                    # สัญญาณให้ Goal Boundary Gate ด้านบน (sticky — ดู declaration) เงื่อนไข
                    # ผูกกับ result.success อยู่แล้วจาก if ด้านบน จึงไม่มีทางติดธงจาก action
                    # ที่ล้มเหลว หรือจาก completed_plan_step ที่ LLM ใส่มาทั้งที่ไม่มีแผนเลย
                    # W_goal_scope_false_success: action แบบอ่านอย่างเดียวไม่ทำให้ step
                    # สุดท้ายของแผน "เสร็จ" ได้ในทางปฏิบัติ — comment ด้านบนบอกว่าปลอดภัยเพราะ
                    # ผูกกับ result.success แต่ read_page_data ที่ "สำเร็จ" คือการอ่านล้วนๆ
                    # ไม่ได้พิสูจน์อะไรเกี่ยวกับงานที่ต้องทำเลย (บั๊กจริง live run 2026-08-27:
                    # โมเดลส่ง completed_plan_step=5 มากับ read_page_data แล้ว Goal Boundary
                    # Gate ก็หยุด task พร้อมอ้างว่าสำเร็จทั้งที่ยังไม่ได้ลบอะไร)
                    # ใช้ชุด action เดียวกับ gate เองใช้ ไม่สร้างชุดใหม่ซ้อน
                    # W_plan_step_cursor: ผูกกับ cursor ของโค้ด ไม่ใช่ตัวเลขดิบจากโมเดล —
                    # เดิมโมเดลส่งเลขข้อสุดท้ายมาเป็น action แรกก็ทำให้ธงนี้ติดได้ทันที
                    if plan_cursor > _total_plan_steps(plan_text):
                        plan_fully_completed = True

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
                        # W_required_error_is_not_a_dead_end: ข้อความชนิด "ช่องนี้ต้องกรอก"
                        # ไม่ใช่ทางตัน — agent เติมเองได้ ไม่ว่าจะเป็นของค้างจากช่องที่กรอกไปแล้ว
                        # (W_required_error_survives_the_fix) หรือเป็นช่องที่ยังไม่ได้กรอกจริงๆ
                        and not _is_required_field_error(e)
                    ]
                    if validation_errors:
                        errors_text = " | ".join(f"'{e}'" for e in validation_errors)
                        success = False
                        completion_verification = "TASK_FAILED_USER_INPUT_ERROR"
                        retry_value_field_labels = list(agent_filled_field_labels)
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
                # (DOM-based ล้วนๆ) ตรวจไม่เจอครบ (แม้จะมี marker "[obscured]" เสริม
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

                # W_plan_progress_stall: แนบตรงนี้เพราะเป็นจุดเดียวที่ตอบ tool_result ของเทิร์นนี้
                # — การ continue ก่อนถึงบรรทัดนี้จะทิ้ง tool_use ไว้โดยไม่มีคำตอบ
                if (
                    plan_text
                    and actions_since_plan_progress >= _MAX_ACTIONS_WITHOUT_PLAN_PROGRESS
                    and plan_stall_notes_sent < _MAX_PLAN_STALL_NOTES
                ):
                    plan_stall_notes_sent += 1
                    _plan_steps = _plan_step_lines(plan_text)
                    _current_step_text = (
                        _plan_steps[plan_cursor - 1] if 0 < plan_cursor <= len(_plan_steps) else ""
                    )
                    if _current_step_text:
                        if verbose:
                            print(
                                f"[plan-stall {plan_stall_notes_sent}/{_MAX_PLAN_STALL_NOTES}] "
                                f"{actions_since_plan_progress} action แล้วยังอยู่ข้อ {plan_cursor}",
                                flush=True,
                            )
                        result_text = result_text + _PLAN_STALL_NOTE_TEMPLATE.format(
                            count=actions_since_plan_progress,
                            step=plan_cursor,
                            total=len(_plan_steps),
                            step_text=_current_step_text,
                        )
                        # นับใหม่หลังเตือน ไม่งั้นจะเตือนซ้ำทุก action ที่เหลือ
                        actions_since_plan_progress = 0

                # W_action_matches_plan_step: แนบที่เดียวกับ W_plan_progress_stall ด้วยเหตุผล
                # เดียวกัน — จุดนี้คือที่เดียวที่ตอบ tool_result ของเทิร์นนี้
                if (
                    plan_text
                    and actions_mismatching_plan_step >= _MAX_ACTIONS_MISMATCHING_PLAN_STEP
                    and plan_mismatch_notes_sent < _MAX_PLAN_MISMATCH_NOTES
                ):
                    _mismatch_steps = _plan_step_lines(plan_text)
                    _mismatch_text = (
                        _mismatch_steps[min(plan_cursor, len(_mismatch_steps)) - 1]
                        if _mismatch_steps else ""
                    )
                    if _mismatch_text:
                        plan_mismatch_notes_sent += 1
                        if verbose:
                            print(
                                f"[plan-mismatch {plan_mismatch_notes_sent}/"
                                f"{_MAX_PLAN_MISMATCH_NOTES}] "
                                f"{actions_mismatching_plan_step} action \u0e44\u0e21\u0e48\u0e15\u0e23\u0e07\u0e01\u0e31\u0e1a step {plan_cursor}",
                                flush=True,
                            )
                        result_text = result_text + _PLAN_MISMATCH_NOTE_TEMPLATE.format(
                            count=actions_mismatching_plan_step,
                            step=plan_cursor,
                            total=len(_mismatch_steps),
                            step_text=_mismatch_text,
                        )
                        actions_mismatching_plan_step = 0

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
                            f"- (there are {dropped_count} earlier steps, compacted away and not shown in detail)",
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
                        # W_token_trim (P3/M3): the compaction just spliced away the turns
                        # that held the full site manual — re-send it in full next step so
                        # the id reference always has something earlier to point back to
                        # ([PRE_LEARNED_MANUAL] strict mode in particular must reach the
                        # model at least once per compaction window)
                        force_full_site_manual = True
                        if verbose:
                            print(
                                f"[context-compact] ย่อ step ..{cut_step_num} เหลือ digest สะสม "
                                f"{len(digest_lines)} บรรทัด (เก็บ {_KEEP_RECENT_STEPS} step ล่าสุดแบบ raw)",
                                flush=True,
                            )

                if tool_input.get("type") in _PAGE_CHANGING_ACTIONS:
                    _wait_started_at = time.monotonic()
                    await wait_stable(page)
                    # W_timing_gap: เติมย้อนเข้า record ของ step นี้ (ดู step_record ด้านบน)
                    step_record["timing"]["wait"] = round(time.monotonic() - _wait_started_at, 3)

                    # domain guard: classify_action() เช็ค allowlist แค่ตอน type=="goto"
                    # เท่านั้น — click ที่พาออกนอกโดเมน (เช่นลิงก์ "Sign in with Google"/
                    # OAuth, โฆษณา, redirect ในหน้าเดิม) ไม่ถูกจับตอน permission check
                    # เลย เพราะ perception.py ไม่ได้เก็บ href ของ element ไว้เช็คล่วงหน้า
                    # — เช็คซ้ำอีกชั้นหลัง action จบแล้วจริงแทน (defense-in-depth) ถ้าหลุด
                    # ออกนอก allowlist ให้ดึงกลับทันทีก่อนจะ perceive/ส่งให้ LLM เห็นหน้า
                    # นอกขอบเขต — ตั้งแต่ W_domain_guard_default เป็นต้นมา
                    # effective_allowed_domains มีค่าเสมอ (default = โดเมนของ url เอง)
                    # จึงครอบ task ทุกเส้นทางรวมทั้งที่ยิงผ่าน API ไม่ใช่แค่ CDP เหมือนเดิม
                    # ยังคงเช็ค None ไว้เผื่อ url ผิดรูปจน extract_domain() คืน ""
                    if effective_allowed_domains is not None:
                        current_domain = extract_domain(page.url)
                        if current_domain not in effective_allowed_domains:
                            try:
                                await page.go_back()
                                await wait_stable(page)
                            except Exception:
                                pass
                            domain_guard_msg = (
                                "[BLOCKED] this action leads to a domain outside the allowed "
                                f"scope ({current_domain!r}) — it was reverted automatically. "
                                f"Domains allowed for this task: {sorted(effective_allowed_domains)}"
                                " Do not try to reach that domain again."
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

            # W_step_budget: ลูปจบเพราะหมดรอบ แต่โมเดลเคยอธิบายไว้แล้วว่าติดอะไร (ตอนที่ guard
            # ปฏิเสธ finish_task(false) ไป) — เอาเหตุผลนั้นมาบอกผู้ใช้ ดีกว่าข้อความ default ลอยๆ
            if final_message == _MAX_STEPS_EXHAUSTED_MESSAGE and last_rejected_finish_message:
                final_message = (
                    f"{_MAX_STEPS_EXHAUSTED_MESSAGE} — เหตุผลล่าสุดที่โมเดลรายงาน: "
                    f"{last_rejected_finish_message}"
                )

            return {
                "success": success,
                "steps": steps_taken,
                "message": final_message,
                "history": self.memory.recent(max_steps),
                "tokens": _tokens_dict(total_usage),
                **_run_stats(),
                "plan": plan_text,
                "final_page_state": final_page_text,
                "persona_message": persona_message,
                "persona_status": persona_status,
                "completion_verification": completion_verification,
                # W_retry_value_has_no_home: ข้อความ TASK_FAILED_USER_INPUT_ERROR สัญญากับ user
                # ว่า "ตอบค่าใหม่มาแล้วระบบจะกรอกแทนที่ในช่องเดิมให้ทันที" — ฟิลด์นี้คือสิ่งที่
                # ทำให้คำสัญญานั้นเป็นจริงได้ ผู้เรียก (api/routes.py) จำไว้กับ session แล้วเทิร์น
                # ถัดไปที่ user ตอบมาเป็น "ค่า" เปล่าๆ จะถูกแปลงเป็นคำสั่งที่ระบุช่องชัดเจน
                "retry_value_field_labels": retry_value_field_labels,
                # W_auto_login_outcome_is_invisible: "skipped" | "ok" | "failed"
                "auto_login": auto_login_outcome,
                # W_index_drift_measure: element ที่ index ชี้เปลี่ยนตัว/หายไประหว่าง
                # snapshot กับ dispatch กี่ครั้ง
                "index_drift_changed": index_drift_changed,
                "index_drift_gone": index_drift_gone,
            }
        except Exception as e:
            # W_loop_crash: เดิม try ก้อนนี้มีแต่ finally ไม่มี except เลยสักตัว — exception
            # ใดๆ จาก get_snapshot()/next_action()/execute()/wait_stable()/compact_messages()
            # จึงทะลุออกจาก run_task() ไปทั้งดุ้น ทำให้ return dict ด้านบนไม่ถูกรันเลย: history
            # ทุก step ที่ทำสำเร็จมาแล้ว, token ที่จ่ายไปจริง, final_page_state, และการเขียน
            # long_term_memory.record_task() หายทั้งหมด — ผู้ใช้เห็นแค่ข้อความ Python ดิบๆ จาก
            # task_manager.py (record.error = str(e)) โดยไม่มีผลงานบางส่วนอะไรเลย
            #
            # เคสจริงที่เจอได้บ่อย: TargetClosedError ตอน page ถูกปิดกลางคัน, "Execution context
            # was destroyed" ตอน SPA re-render พอดีจังหวะ, OAuthLoginRequired ตอน token หมดอายุ
            # กลาง task ยาว, Gemini ResourceExhausted ที่ retry ครบโควตาแล้ว re-raise
            #
            # จับแล้วรายงานตามความจริง (success=False + ข้อความที่บอกว่าพังที่ step ไหน) แต่
            # *ยังคืน dict เดิมครบทุก field* — งานที่ทำไปแล้วไม่หาย และ caller ทุกตัว (routes/
            # task_manager/evaluation) ไม่ต้องรู้จักเส้นทางพิเศษอะไรใหม่เลย
            #
            # ไม่พยายาม "ทำ step ต่อ" หลัง exception: สาเหตุส่วนใหญ่ที่หลุดมาถึงตรงนี้คือ
            # browser/page ตายไปแล้ว (action-level error ธรรมดาถูกจับเป็น ActionResult(False)
            # ใน actions.py อยู่แล้ว ไม่เคยมาถึงที่นี่) การวนต่อจึงมีแต่จะพังซ้ำจนหมด max_steps
            success = False
            final_message = (
                f"Task stopped by an unexpected error at step {steps_taken}: "
                f"{type(e).__name__}: {e}"
            )
            completion_verification = "EXECUTION_FAILED_NEEDS_REPAIR"
            if verbose:
                print(f"[crash] {final_message}", flush=True)
            self.memory.record({
                "step": steps_taken,
                "cmd": {"type": "crash"},
                "result": final_message,
                "success": False,
            })
            return {
                "success": success,
                "steps": steps_taken,
                "message": final_message,
                "history": self.memory.recent(max_steps),
                "tokens": _tokens_dict(total_usage),
                **_run_stats(),
                "plan": plan_text,
                "final_page_state": final_page_text,
                "persona_message": "",
                "persona_status": "FAILED",
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
            await install_ssrf_guard(context)
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

            await _dismiss_consent_banner(page, verbose=False)  # W_consent_banner (ดู main loop)
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
                ask_user_func=ask_user_func, on_event=on_event, run_task_fallback=_run_task_fallback,
            )
        finally:
            if owns_context:
                await context.close()

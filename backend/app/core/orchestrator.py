"""Agent Loop: Perceive -> Plan -> Act -> Verify.

W1 skeleton, W4 loop จริง, W5 retry (actions.py::_dispatch_with_retry) + guard กัน
finish_task(false) ก่อนเวลา + permission layer/human-in-the-loop

แผนที่โซน (ค้นหา "# โซน N" ในไฟล์) — โซน 1-18 คือค่าคงที่/helper ที่ run_task ใช้:
  โซน 1   ค่าคงที่ลูป + guard ก่อนยอมรับ finish_task + qa_summary
  โซน 2   ติดตามแผน (plan ตรงชนิดงาน / action ตรง step)
  โซน 3   จำแนก intent ของ goal (ลบ / แก้ทั้งหมด / นับ)
  โซน 4   ประกอบ prompt ต่อ step + ย้ายแท็บ
  โซน 5   เงื่อนไข field=value ของ goal + marker + guard ตัวกรอง
  โซน 6   อ่านตารางบนหน้า + guard ก่อนลบ
  โซน 7   ชื่อ tool + จัดหมวดความล้มเหลว (telemetry)
  โซน 8   ตรวจ DOM จริงก่อนยอมรับ finish_task
  โซน 9   validation error หลัง action
  โซน 10  จับลูป + recovery
  โซน 11  RAG / memory / vision / permission query
  โซน 12  บีบ context / ลด token
  โซน 13  utility หน้าเว็บ + background task
  โซน 14  เปิด browser + แบนเนอร์คุกกี้ + CAPTCHA + auto-login
  โซน 15  คุยกับ user + แผนที่ส่งให้โมเดล
  โซน 16  Goal Boundary Gate (goal สำเร็จแล้วหยุด)
  โซน 17  guard ห้ามทำเกิน goal (สร้าง / รหัสผ่าน / save ในงานลบ / ค่าที่ขาด)
  โซน 18  รหัสผ่าน + fill_secret
  โซน 19  class Orchestrator — run_task แบ่งเป็นขั้น [run_task 1]-[11] และลูปหลักขั้นย่อย (ก)-(ฐ)

ลำดับการทำงานของ run_task:
  [1-4] ตั้งค่า + resolve browser + state -> [5] goto/แบนเนอร์/auto-login -> [6] qa_summary?
  -> [7] แผน -> [9] ลูป: (ข) perceive -> (ง) LLM -> (จ) finish_task/tool พิเศษ -> (ฉ)(ซ) guard
  -> (ฌ) dispatch -> (ญ)(ฎ) บันทึกผล/แผน -> (ฐ) ส่งผลกลับ + compaction -> [10] บันทึก memory + คืนผล
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
# W_delete_all_intent: import regex key=value ตัวเดียวกับ W_deterministic_count ไม่ copy (กันต้องแก้ 2 ที่)
from backend.app.core.actions import _KEY_VALUE_IN_QUERY_RE
# W_count_answer_check: ตัวอ่านตัวเลขที่โค้ดนับ + keyword คำถามเชิงนับ อยู่ที่ actions.py ที่สร้างข้อความ — import ไม่ copy
from backend.app.core.actions import _COUNT_QUERY_KEYWORDS, system_counted_conditions
from backend.app.core.goal_intent import canonical_intent, contains_keyword
from backend.app.core.memory import ShortTermMemory, clip_result
from backend.app.core.perception import LABEL_MARKERS, get_snapshot, label_without_markers
from backend.app.core.user_browser import connect_user_browser, resolve_target_page
from backend.app.permission.rules import DEFAULT_NEEDS_CONFIRMATION, extract_domain, install_ssrf_guard
from backend.app.rag import retriever
from backend.app.rag.chroma_client import _embedding_function

# ══════════════════════════════════════════════════════════════════════
# โซน 1: ค่าคงที่ของลูป + guard ก่อนยอมรับ finish_task + qa_summary
#   ทำอะไร: โควตา/ข้อความเตือนที่ใช้ตีกลับ finish_task ที่ยังไม่มีหลักฐาน และตั้งค่า mini-loop ถามตอบ
#   ทำงานยังไง: guard แต่ละตัวมีโควตา nudge แล้วปล่อยผ่าน (escape valve) ยกเว้นตัวที่ระบุว่า hard
# ══════════════════════════════════════════════════════════════════════
# action ที่เปลี่ยนหน้า/DOM -> รอหน้านิ่งก่อน perceive รอบถัดไป
# W50: press_key ด้วย — Enter บน custom dropdown อาจ submit/navigate
_PAGE_CHANGING_ACTIONS = {"click", "goto", "select", "go_back", "press_key"}

# โมเดลบางตัว (Llama บน Groq) เรียก finish_task(false) เร็วเกินทั้งที่ยังมีทางให้ลอง —
# เตือนแล้วบังคับลองต่อสูงสุด _MAX_PREMATURE_FALSE_FINISH_RETRIES ครั้งก่อนยอมรับ
_MAX_PREMATURE_FALSE_FINISH_RETRIES = 2
_PREMATURE_FALSE_FINISH_NUDGE = (
    "This finish_task(success=false) is not accepted yet — steps remain, and the current "
    "page may still hold elements you can act on (e.g. a button not yet pressed, a field "
    "still empty). Look at the latest indexed elements again and try an action you haven't "
    "tried. If you genuinely cannot proceed after that, call finish_task(success=false) again."
)

# W5[A] Verify (2026-07-15): คู่ของ guard ด้านบน — finish_task(true) ตอน steps_taken=0 เคยผ่านทันที
# ไม่ block เด็ดขาด (บาง goal สำเร็จตั้งแต่หน้าแรกจริง) แค่ให้ยืนยันอีกครั้ง
_MAX_PREMATURE_TRUE_FINISH_RETRIES = 1

# W_resume: จำกัด request_user_input ต่อ task กัน LLM วนถามไม่รู้จบ — เกินโควตาป้อน tool_result
# บอกเหตุผลให้ตัดสินใจเอง (ไม่ finish_task เงียบๆ)
_MAX_REQUEST_USER_INPUT_CALLS = 3

# W_token_cut W3 (live baseline 2026-09-02): เทิร์นที่ไม่ได้ทำอะไรส่วนใหญ่คือ guard ปฏิเสธซ้ำ
# เหตุผลเดิม — ไม่ลดความปลอดภัย แค่ไม่ให้เตือนซ้ำกินเทิร์น
# _MAX_TASK_GUARD_REJECTIONS: ปฏิเสธรวมทั้ง task เกินนี้ = ติดลูป จบ task ตามจริง (ตัวนับ
#   monotonic ไม่ reset ต่างจากโควตาต่อ guard ที่ reset เมื่อ action สำเร็จ)
# _MAX_SAME_GUARD_REASON_REJECTIONS: เหตุผลเดียวซ้ำเกินนี้ (ข้ามรอบ reset ได้) = ลูป
_MAX_TASK_GUARD_REJECTIONS = 9
_MAX_SAME_GUARD_REASON_REJECTIONS = 4

# W44: qa_summary เดิมใช้ summarize_page() อย่างเดียว เห็นแค่ interactive elements ตอบคำถาม
# เรื่องเนื้อหาตารางไม่ได้ — ตอนนี้วน next_action() สูงสุด _QA_SUMMARY_MAX_STEPS รอบ อนุญาตแค่
# read_page_data + finish_task (action อื่นปฏิเสธ ไม่ dispatch) ครบโควตาแล้ว fallback summarize_page()
# W46: agent ยอมแพ้ก่อนลองค้นหา — ผ่อนให้ fill/click ได้เฉพาะ label ที่เป็นช่อง/ปุ่มค้นหา
# (_label_looks_like_search) ขยายเป็น 4 step เพราะ search flow ต้อง fill -> click -> read -> finish
_QA_SUMMARY_MAX_STEPS = 4
# W19 (Guard Compatibility Rule): บอก LLM ว่าคลิก nav ไปหน้าอื่นได้ด้วย (ดู qa_is_nav_click)
_QA_SUMMARY_ACTION_REJECTED_NUDGE = (
    "[Rejected] This is a qa_summary question (asking for information, not issuing a "
    "command), so actions that change the page (fill/select/goto/...) are not allowed — the "
    "only exceptions are fill/click on a search box/search button/filter, or clicking a "
    "menu/nav item to reach another page holding the information you need. To read more "
    "content use type: 'read_page_data', then call finish_task with your final answer as "
    "soon as you can answer the question. Never conclude 'there is no data' before trying to "
    "search or navigate at least once."
)

# label ช่อง/ปุ่มค้นหา — ชั้นสำรองระดับโค้ด (pattern เดียวกับ RISKY_LABEL_KEYWORDS)
# คำต้องเจาะจงพอ ("หา" เดี่ยวๆ จะแมตช์ "หาย"/"หาก")
_QA_SUMMARY_SEARCH_LABEL_KEYWORDS = ("search", "ค้นหา", "filter", "กรอง", "find", "query")


def _label_looks_like_search(label: str) -> bool:
    lower = (label or "").lower()
    return any(keyword in lower for keyword in _QA_SUMMARY_SEARCH_LABEL_KEYWORDS)


# agent ตอบ list พร้อมรายละเอียดที่ไม่มีใครถาม เป็นย่อหน้าเดียว — ต่อท้าย goal เฉพาะ qa_summary
# ไม่แตะ SYSTEM_PROMPT หลักที่ action_task ใช้ร่วม
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

# ACC-3: finish_task(true) ทั้งที่ mutating action ล้มเหลวทุกครั้ง (index ผิดซ้ำๆ) — guard
# steps_taken==0 และ validation-error จับไม่ได้ เช็คจาก ShortTermMemory ว่ามีอย่างน้อย 1 ครั้งที่สำเร็จ
# ไม่นับ read_page_data/wait/hover/scroll escape valve เหมือน guard อื่น (ครบโควตาแล้วปล่อยผ่าน)
_MAX_PREMATURE_ALL_FAILED_RETRIES = 2
# ห้ามใส่ goto/go_back/switch_tab — goto แรกสุดถูก record success=True ทุก task guard จะไม่มีวันยิง
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


# W_readonly_goal_evidence (OrangeHRM 2026-08-26): goal ถามอย่างเดียว (goto -> read_page_data ->
# finish) ไม่มีวันมี mutating action guard ACC-3 จึงปฏิเสธคำตอบถูกทุกครั้ง เสีย LLM call ฟรี
# read_page_data ที่สำเร็จ = หลักฐานว่าได้ข้อมูลจริง (ต่างจาก goto ไม่ถูก record อัตโนมัติ)
_EVIDENCE_ACTION_TYPES = _MUTATING_ACTION_TYPES | {"read_page_data"}


def _has_any_successful_mutating_action(history: list[dict]) -> bool:
    return any(
        (h.get("cmd") or {}).get("type") in _EVIDENCE_ACTION_TYPES and h.get("success") is True
        for h in history
    )

# Task4 (W19 Task Completion Verifier): ปฏิเสธ finish_task(true) เมื่อหน้ายังมี validation error
# อิสระจาก guard steps_taken==0 (ทำงานพร้อมกันได้ทุก steps_taken)
_MAX_PREMATURE_VALIDATION_ERROR_RETRIES = 2
_PREMATURE_VALIDATION_ERROR_NUDGE_TEMPLATE = (
    "This finish_task(success=true) is rejected — error/validation messages are still shown "
    "on the current page: {errors} Never treat the task as successful while these remain. "
    "Fix the relevant field per the error message first (e.g. fill an empty field, correct "
    "an invalid value, change a duplicate value), then retry. Once the errors are genuinely "
    "gone, call finish_task(success=true) again."
)

# [role=alert] = มาตรฐาน a11y, [class*=error/invalid] = สำรองเว็บไม่ใช้ ARIA,
# .oxd-input-field-error-message = OrangeHRM, :not(:empty) กัน placeholder ว่างที่ render ทิ้งไว้
# W20 (Task12): เพิ่ม .oxd-input-group__message/.text-danger/.invalid-feedback/.oxd-input--error
# (Bootstrap ไม่มีคำ error/invalid ในชื่อ class)
_VALIDATION_ERROR_SELECTOR = (
    '[role="alert"]:not(:empty), [class*="error" i]:not(:empty), '
    '[class*="invalid" i]:not(:empty), .oxd-input-field-error-message, '
    '.oxd-input-group__message, .text-danger, .invalid-feedback, .oxd-input--error'
)

# W20 (Task12 follow-up): หน้า login OrangeHRM มี div.orangehrm-login-error (คำใบ้ demo
# credentials นอก <form>) แมตช์ [class*=error] ทำ hard-stop guard หลัง fill ยิงผิดตั้งแต่ step แรก
# validation error จริงอยู่ใน <form> เสมอ — guard หลัง fill สแกนเฉพาะใน form (within_form)
# guard ก่อน finish_task ยังสแกนทั้งหน้า (backstop SPA ที่ไม่มี <form>)
_VALIDATION_ERROR_SELECTOR_IN_FORM = ", ".join(
    f"form {part.strip()}" for part in _VALIDATION_ERROR_SELECTOR.split(",")
)

# W19 (latency): timeout สำหรับ locator ที่แค่เช็คสถานะ DOM (default Playwright 30s นานเกิน)
# ใช้กับ _login_form_needs_password/_scan_validation_errors — ต่างจาก state_filter 500ms เพราะ
# ตัวนี้เช็คแค่ตอน finish_task/login-guard ความถี่ต่ำ เผื่อหน้าช้าได้
_DOM_CHECK_TIMEOUT_MS = 3000


# W22 (Post-Action Verification): agent ตอบ "ลบ ESS หมดแล้ว" ทั้งที่หน้ายังโชว์ "(3) Records Found"
# — LLM สรุปจาก history แทน DOM จริง guard นี้นับ record ที่เหลือจริงก่อนยอมรับ finish_task
# เปิดเฉพาะ deletion-intent goal ("X Records Found" ไม่เกี่ยวกับ goal อื่น)
_MAX_PREMATURE_DELETION_INCOMPLETE_RETRIES = 2

# W64[7.1]: {action_hint} ต่างกันตามงานลบ/แก้ไข (_premature_mutation_action_hint) — เดิม hardcode "ลบ"
_PREMATURE_DELETION_INCOMPLETE_NUDGE_TEMPLATE = (
    "This finish_task(success=true) is rejected — checking the current page's real DOM shows "
    "{count} entries still matching the condition (actual text on the page: \"{text}\"). "
    "Never treat the job as complete, or claim no matching entries exist, while the table "
    "genuinely still shows a count above 0 — {action_hint}until the count really reaches 0 "
    "or the page shows \"No Records Found\", only then may you call finish_task(success=true)."
)

# W22: keyword งานลบ (ไม่ผูกกับ "ทั้งหมด" — ลบรายการเดียวก็เจอ false-completion ได้)
_DELETION_INTENT_KEYWORDS = ("ลบ", "delete", "remove", "ล้าง")


def _is_deletion_intent_goal(goal: str) -> bool:
    """W22: goal เป็นงานลบ -> เปิด _scan_remaining_target_records() ก่อนยอมรับ finish_task(true)"""
    lower = (goal or "").lower()
    return any(kw in lower for kw in _DELETION_INTENT_KEYWORDS)


# ══════════════════════════════════════════════════════════════════════
# โซน 2: ติดตามแผน (plan)
#   ทำอะไร: ตรวจว่าแผนตรงชนิดงานของ goal และ action ตรงกับ step ปัจจุบัน
#   ทำงานยังไง: _plan_drops_goal_operation() ตรวจตอนร่างแผน, _action_matches_plan_step() เป็นสัญญาณอ่อน ใช้เตือนเท่านั้น ไม่บล็อก
# ══════════════════════════════════════════════════════════════════════
# W_plan_keeps_goal_verb (live run 2026-08-27): goal "ลบ user ESS ให้หมด" แต่ planner ร่างแผน
# "แก้ไข Role ออกจาก ESS" แล้ว agent เดินตามเป๊ะ — ปัญหาคือแผนผิดชนิดงาน scope แค่ deletion
# (ทางเลือกผิดที่ดูสมเหตุสมผลและผลถาวร)
# W_plan_warn_not_abort: แผนที่ user ยืนยันแต่ผิดชนิดงาน -> เตือนตั้งแต่เทิร์นแรก ไม่หยุด task
# W_plan_progress_stall: stall detector เดิมจับแค่ action ซ้ำ ไม่จับ action ต่างกันที่ไม่คืบหน้า
# ผูกกับ plan_cursor ที่มีอยู่ advisory ล้วน — รู้ผลหลัง dispatch reject+continue ไม่ได้ (tool_use
# ไม่มี tool_result -> Anthropic/Groq error) จึงแนบข้อความกับผล action แบบ blocked_note
# W_action_matches_plan_step (W111): ตัวนับ stall ถามแค่ "cursor ขยับไหม" แต่โมเดลใส่
# completed_plan_step แทบทุก action ตัวนับไม่มีวันถึงเกณฑ์ — ตัวนี้ห้าม reset เมื่อ cursor ขยับ
# reset เฉพาะเมื่อ action ตรง step จริง advisory/สัญญาณอ่อนโดยเจตนา (step เขียนโดย LLM กำกวมได้)
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
    """action นี้ดูเหมือนทำ step นี้อยู่ไหม — None = ตัดสินไม่ได้ (ผู้เรียกต้องไม่นับเป็น mismatch)
    ใช้ _field_names_match() ที่ทนพิมพ์ผิด/คำไทยติดกันอยู่แล้ว"""
    if not (action_label or "").strip():
        return None

    # W_action_matches_plan_step: แผนไทย vs label อังกฤษ ("กดค้นหา" vs "Search") เทียบ token ไม่ได้
    # ใช้ regex สองภาษาที่มีอยู่เป็นสะพาน — ต้องเช็คก่อนเกณฑ์จำนวนคำ เพราะไทยไม่มีเว้นวรรค
    # step ไทยเกือบทุกข้อจะ "ตัดสินไม่ได้"
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
    """คืนเหตุผลที่แผนทิ้งชนิดงานของ goal หรือ None ถ้าใช้ได้ (ผู้เรียกใช้เป็น boolean)

    กฎ A: goal มีคำลบ แต่แผนไม่มีคำลบเลย (ไม่ตัดสินจากคำแก้ไข — แผนลบที่ถูกอาจมี "เลือก/set" filter)
    กฎ B (W_plan_commits_a_record_edit): goal ลบ *ล้วน* แต่แผนมีขั้นตอนบันทึกการแก้ไข record
    — แผนผิดจริงที่ user รายงานมีคำ "ลบ" จึงผ่านกฎ A กฎเดียวกับ W_no_record_edit_for_delete_goal
    ตอน runtime แค่ตรวจเร็วขึ้น gate ด้วย _goal_is_deletion_only() กัน goal แก้ไขจริงโดน"""
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


# ══════════════════════════════════════════════════════════════════════
# โซน 3: จำแนก intent ของ goal
#   ทำอะไร: บอกว่า goal เป็นงานลบ / แก้ทั้งหมด / ลบทั้งหมด / คำถามเชิงนับ
#   ทำงานยังไง: keyword matching ล้วน (ไม่เรียก LLM) — ใช้เปิด/ปิด guard ในโซนอื่น
# ══════════════════════════════════════════════════════════════════════
# W64[7.1]: goal เปลี่ยน Role ESS ทั้งหมดเป็น Admin แต่ agent finish ทั้งที่ยังเหลือแถว — สาเหตุ
# เดียวกับ deletion (สรุปจาก history) เงื่อนไขสำเร็จเหมือนกัน: แถวที่ตรงเงื่อนไขต้องเหลือ 0
# W68: goal "เปลี่ยน Role ของทุกคน..." ไม่ติดกันเป็นวลี exact-phrase พลาด guard ไม่เปิด —
# เปลี่ยนเป็น "มี verb + มี bulk marker ที่ไหนก็ได้" (superset ของเดิม) false-positive เสี่ยงต่ำ
# เพราะ _scan_remaining_target_records fail-safe (ไม่เจอ "(N) Records Found" = ปล่อยผ่าน)
_EDIT_ALL_MUTATION_VERBS = ("เปลี่ยน", "แก้ไข", "แก้", "ปรับ", "update", "change", "edit", "set")
# "ทุก" คำเดียวพอ (ครอบคลุม "ทุกคน"/"ทุกราย"/"ทุกแถว"/"ทุกรายการ"/"เปลี่ยนทุก" ที่เป็น substring
# ของมันอยู่แล้วทั้งหมด — ไม่ต้องแจกแจงแยกทีละคำ) บวก "ทั้งหมด"/"ให้หมด" ที่ไม่มีคำว่า "ทุก" ปน
_EDIT_ALL_BULK_MARKERS = ("ทุก", "ทั้งหมด", "ให้หมด", "all", "every", "each")


def _is_edit_all_intent_goal(goal: str) -> bool:
    """W64[7.1]/W68: goal มีทั้ง verb แก้ไข และ bulk marker (ไม่ต้องติดกัน) — OR กับ
    _is_deletion_intent_goal() เป็นเงื่อนไขเปิด _scan_remaining_target_records()"""
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
    """W64[7.1]: คำแนะนำต่อท้าย nudge ตาม intent — deletion เป็น default (ตรง W22 เดิม)"""
    if _is_deletion_intent_goal(goal):
        return _DELETE_MUTATION_ACTION_HINT
    return _EDIT_ALL_MUTATION_ACTION_HINT


# W64[7.1] (Filter Order): agent เลือก filter แล้วกด Edit บนแถวทันทีก่อนกด Search (แถวเก่าที่ยัง
# ไม่กรอง) — SYSTEM_PROMPT แนะนำอย่างเดียวไม่พอ hard guard บล็อกปุ่ม row-action ถ้า step ก่อนหน้า
# *ทันที* คือ fill/select ที่สำเร็จ (window แคบสุด กัน block ผิดตอนกรอกฟอร์ม Edit ที่เปิดอยู่)
# W_delete_all_intent (2026-08-26): goal "ลบ ESS ให้หมด" agent ลบ 1 จาก 7 แล้ว claim success
# ไม่เคยกด Search — bulk marker ใช้แค่ใน _is_edit_all_intent_goal (ต้องมี verb แก้ไข) "ให้หมด"
# จึงไม่มีผล ตกลงกับ user: บังคับ "ลำดับ" (กรองก่อนลบ, ห้าม claim ถ้ายังเหลือ) ไม่บังคับ "วิธี"
# (Select All หรือทีละแถวก็ได้ เว็บที่ไม่มี Select All ต้องทำงานได้)
def _is_delete_all_intent_goal(goal: str) -> bool:
    """W_delete_all_intent: งานลบ + ขอบเขต "ทุก/ทั้งหมด" (ใช้ค่าคงที่เดิม ไม่สร้าง keyword ใหม่)"""
    lower = (goal or "").lower()
    return _is_deletion_intent_goal(goal) and any(m in lower for m in _EDIT_ALL_BULK_MARKERS)


# W_count_answer_check: โค้ดนับให้แล้วแต่ไม่มีใครเช็คว่าโมเดลใช้ตัวเลขนั้น (บั๊กสด: ESS 7 แถว agent
# ตอบ 6) ยิงเฉพาะเมื่อครบ 3 ข้อ: goal เป็นคำถามเชิงนับ + ค่าเงื่อนไขอยู่ใน goal + ตัวเลขที่นับได้
# ไม่โผล่ในคำตอบเลย (ไม่ตีความภาษาธรรมชาติ)
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
    """W_count_answer_check: (ค่าเงื่อนไข, จำนวนที่นับได้) คู่แรกที่คำตอบขัด หรือ None

    deterministic ล้วน: แค่ถามว่าตัวเลขที่นับได้โผล่ในคำตอบไหม ใช้ร่วมทั้ง main loop และ
    qa_summary mini-loop โดยตั้งใจ (เขียนแยกสองที่จะเพี้ยนออกจากกัน)"""
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

# ══════════════════════════════════════════════════════════════════════
# โซน 4: ประกอบ prompt ต่อ step + ย้ายแท็บ
#   ทำอะไร: เลือกบล็อก SYSTEM_PROMPT ที่ต้องส่ง และตามแท็บใหม่ที่ action เปิด
#   ทำงานยังไง: _resolve_prompt_sections() สะสม section จากสัญญาณ deterministic, _detect_tab_switch() เปลี่ยน page ที่ลูปถือ
# ══════════════════════════════════════════════════════════════════════
# W_prompt_sections (P4.1): เลือกบล็อก SYSTEM_PROMPT ที่ต้องส่ง (ดู llm.py::build_system_prompt)
# สะสมอย่างเดียว ไม่ถอด: (1) prompt สลับไปมาทำ prefix cache พลาด (2) กฎที่หายกลางทางไล่บั๊กยาก
# สัญญาณ deterministic ทั้งหมด:
#   plan     — มี plan_text
#   table    — goal bulk/ลบ/แก้ทั้งหมด/นับ หรือหน้ามี checkbox เลือกแถว
#   widget   — มี <select> หรือ trigger ของ custom dropdown
#   password — allow_fill_secret (W_fill_secret_schema_gate)
_TABLE_ELEMENT_LABEL_HINTS = ("select row", "select all", "records found")
_WIDGET_ELEMENT_LABEL_HINTS = ("-- select --", "--select--")


# W_tab_rebind (P3.6): switch_tab() แค่ bring_to_front() ตัวแปร `page` ของลูปไม่เปลี่ยน agent จ้อง
# แท็บเก่า — แท็บจาก target="_blank" หนักกว่า (โมเดลไม่เคยสั่ง switch_tab) แก้ที่ลูป ไม่ให้
# ActionResult พก Page (serialize ไม่ได้)
def _detect_tab_switch(page: Page, tabs_before: list, cmd: dict):
    """คืน (page ที่ควรใช้ต่อ, ข้อความอธิบาย) หรือ (page เดิม, "") — ห้าม throw
    (browser อาจปิดแล้วตอน task จบ/ถูก stop)"""
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


# W_core_carries_situational_rules: ปุ่ม/ช่องค้นหา — แยกจาก _FORM_SUBMIT_LABEL_KEYWORDS
# โดยเจตนา (ค้นหาไม่ใช่บันทึก)
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
    # W_password_rules_arrive_too_late (2026-09-03): agent ไป PIM > Update Password แทนเมนูโปรไฟล์
    # เพราะกฎ W20 ส่งเฉพาะตอน allow_fill_secret (อยู่หน้าฟอร์มแล้ว = เลือกทางผิดไปแล้ว)
    # ส่งตั้งแต่ goal/แผนพูดถึงการเปลี่ยนรหัสผ่าน
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
    # W_core_carries_situational_rules: gate ตาม marker บนหน้าจริง มาพร้อม snapshot ของ step
    # เดียวกันจึงไม่มีทางส่งไม่ทัน
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
        if "[focused]" in label:
            sections.add("marker_focused")
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


# ══════════════════════════════════════════════════════════════════════
# โซน 5: เงื่อนไข field=value ของ goal + marker + guard ตัวกรอง
#   ทำอะไร: อ่านเงื่อนไขจาก goal เทียบกับชื่อ/ค่าตัวกรองบนหน้า และข้อความเตือนของ guard ก่อน dispatch
#   ทำงานยังไง: ดึง key=value จาก goal (ตัด URL) -> เทียบชื่อ field แบบทนพิมพ์ผิด 3 ชั้น -> อ่านค่าตัวกรองจาก label ของ perception
# ══════════════════════════════════════════════════════════════════════
_URL_IN_GOAL_RE = re.compile(r"https?://\S+", re.IGNORECASE)


def _normalized_field_name(text: str) -> str:
    """W_filter_scope_guard: ยุบชื่อ field ให้เทียบได้ ("userrole" == "User Role") —
    ตัดทุกอย่างที่ไม่ใช่ตัวอักษร/ตัวเลข แล้วเทียบตัวพิมพ์เล็ก"""
    return re.sub(r"[^0-9a-z\u0e00-\u0e7f]+", "", (text or "").lower())


def _goal_condition_pairs(goal: str) -> list[tuple[str, str]]:
    """W_column_aware_rows: คู่ (field ที่ normalize แล้ว, ค่า) จาก "key=value" ใน goal — แหล่ง
    ความจริงเดียวของ _goal_condition_fields()/_goal_condition_values()

    ตัด URL ก่อนเสมอ: query string ของ URL ใน goal เข้ารูป key=value ได้ ถ้าไม่ตัดจะได้เงื่อนไข
    AND ที่ไม่มีแถวไหนตรง แล้ว guard บล็อกการลบที่ถูกต้อง
    คืน [] ถ้าไม่มีเงื่อนไข = ไม่เปิด guard (ไม่เดาจากภาษาธรรมชาติ)"""
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


# W_filter_scope_guard: perception เติม prefix ชื่อ field ("User Role: -- Select --") — ตัวคั่นคือ
# ": " ตัวแรกเท่านั้น (ค่าอาจมี ":" เอง)
_FIELD_LABEL_PREFIX_RE = re.compile(r"^([^:]{1,40}):\s")


def _filter_field_from_label(label: str, action_type: str = "") -> str:
    """ชื่อ field ที่ action นี้จะแตะ (normalize แล้ว) — "" = ตัดสินไม่ได้ ต้องปล่อยผ่าน

    W_field_label_without_value (2026-08-31): ช่องว่างไม่มี prefix "ชื่อ: ค่า" label คือชื่อเปล่าๆ
    guard จึงตาบอดตอนช่องยังไม่ถูกแตะ (agent fill "Username" ผ่านฉลุย) — จำกัดที่ fill/select
    (เล็ง form field เสมอ) ห้ามตีความ click แบบนี้ ไม่งั้นปุ่ม "Search" กลายเป็นชื่อ field"""
    # W_label_marker_key: "Username [required]" ต้องให้ชื่อ field เป็น "username" ไม่ใช่
    # "usernamerequired" ซึ่งจะไม่ match อะไรเลยแล้ว guard ก็เงียบไปเฉยๆ
    cleaned = _label_without_markers(label)
    match = _FIELD_LABEL_PREFIX_RE.match(cleaned)
    if match:
        return _normalized_field_name(match.group(1))
    if action_type in ("fill", "select"):
        return _normalized_field_name(cleaned)
    return ""


# W_filter_scope_guard: goal จริงสกปรก ("แล้บลบuserole=ess" ไม่เว้นวรรค สะกดผิด) เทียบตรงตัวไม่ match
# แล้วบล็อกการกดที่ถูก (แย่กว่าไม่มี guard) — เทียบ 3 ชั้นจากเข้มไปหลวม fail-open เมื่อตัดสินไม่ได้
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
    # "แล้บลบuserole" ครอบ "userole" — คำไทยนำหน้า key ไม่ควรทำให้ไม่ match
    # W_field_match_min_length (MR3): ชื่อสั้น ("name") ครอบอยู่ใน employeename/username/nationality
    # ปล่อยตั้งค่าช่องผิดผ่าน — ต่ำกว่าเกณฑ์ใช้กฎเข้มเท่านั้น
    shorter = min(len(goal_field), len(page_field))
    if shorter >= _MIN_FIELD_NAME_CONTAINMENT_LENGTH and (
        page_field in goal_field or goal_field in page_field
    ):
        return True
    # ตัดส่วนที่ไม่ใช่ ASCII แล้วลองใหม่ ("แล้บลบuserole" -> "userole")
    # W_field_match_min_length: ชั้นนี้ใช้ containment ด้วย ต้องมีเพดานความสั้นเดียวกัน
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


# W_prefer_row_delete: ปล่อย "Edit" ผ่านเพราะบางเว็บใช้หน้า Edit เป็นทางไปปุ่มลบ — แต่ถ้าหน้านี้
# มีปุ่มลบอยู่แล้ว เข้า Edit คือเดินผิดทาง (live run 2026-08-27 จบด้วยเปลี่ยน Role จริง)
# ไม่มีปุ่มลบ = ปล่อยผ่านเหมือนเดิม
# W_marker_registry (W108): marker เคยพิมพ์ literal ซ้ำหลายที่ ของสำคัญคือเทสต์ใน test_orchestrator.py
# ที่อ่านซอร์ส perception.py แล้วยืนยันว่าทุก marker อยู่ในทะเบียน (ทำให้ drift ดังแทนเงียบ)
# W_marker_hides_a_shared_label: ทะเบียนอยู่ที่ perception.py คู่กับ JS ที่สร้าง marker
_PERCEPTION_LABEL_MARKERS = LABEL_MARKERS

# W_label_marker_key (W107): marker = สถานะชั่วคราว ไม่ใช่ตัวตน — loop detector ใช้ (type, label)
# เป็นกุญแจ agent สลับคลิก "Select row" กับ "Select row [hidden — ...]" ตัวนับ reset ทุกครั้ง
# เผา step จนหมด ตัดเฉพาะ marker ที่รู้จัก ไม่ตัด [...] ทั่วไป (ปุ่มจริงอย่าง "[Beta] Export")
# ใช้เมื่อต้องการ "ตัวตน" ของ element เท่านั้น ห้ามใช้แทน label ดิบในที่ที่ตั้งใจตรวจ marker
# ตัวเดียวกับ perception.label_without_markers (เดิม copy ไว้สองที่)
_label_without_markers = label_without_markers

# W_reject_obscured_click (P8/M3): ป้าย [obscured] ไม่เคยมีโค้ดอ่านเลย โมเดลคลิกของที่ถูกบัง
# ~12-18 วินาทีต่อครั้งจน timeout — ปฏิเสธเฉพาะตอนมี dialog เปิด (overlay อื่นอาจหายเองก่อนคลิก
# ส่วน dialog ค้างทำให้ถูกบังถาวร)
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

# W_prefer_row_delete: ป้าย [Profile/Account Menu] เดิมใช้แค่ใน forced recovery — คลิกที่โมเดล
# เลือกเองไม่เคยถูกกัน (live run 2026-08-27: กดเมนูโปรไฟล์กลางงานลบ user)
_PROFILE_MENU_LABEL_MARKER = "[Profile/Account Menu]"
# W_account_keyword_scope (MR4): "setting"/"ตั้งค่า" เดี่ยวๆ กว้างเกิน (settings ของระบบก็ปิด guard)
# ใช้เฉพาะรูปที่เป็นของตัวผู้ใช้เอง
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


# W_empty_table_needs_right_filter (2026-08-31): guard ยืนยันจบงานรับ "ตารางเหลือ 0 แถว" โดยไม่เช็ค
# ว่าตัวกรองยังตั้งตาม goal — ตารางว่างเพราะกรองผิดกับเพราะลบครบหน้าตาเหมือนกัน (รันจริง: ตัวกรอง
# เพี้ยน ได้ตารางว่าง รายงานว่าลบครบ ทั้งที่ไม่ได้ลบอะไร) ฝั่งทำลายมี guard คู่นี้แล้ว (W_delete_all_intent
# guard B) ฝั่งยืนยันไม่เคยมี อ่านจาก label ของ perception ล้วน ไม่เรียก LLM/DOM เพิ่ม
# ค่าที่แปลว่า "ยังไม่ได้ตั้ง" = ไม่ตรงเงื่อนไข (ไม่ใช่ "ไม่รู้")
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
    """ตัวกรองบนหน้าตรงกับ goal ไหม
    True = ทุก field ที่ goal ระบุมีค่าถูก, False = เจอ field แต่ค่าไม่ตรง (รวมหลุดเป็น "-- Select --"),
    None = อ่านไม่ได้ -> ผู้เรียกต้อง fail-open"""
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
    """ตัวกรองส่วนเกินที่ตั้งไว้ทั้งที่ goal ไม่ได้พูดถึง (เคส Status=Enabled: ESS ที่ถูก disable หาย
    จากตาราง "ลบให้หมด" จึงจบทั้งที่ยังเหลือ) — W_filter_scope_guard กันตอนตั้ง ตัวนี้กันตอนสรุปผล"""
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

# W_filter_already_satisfied: โควตาเท่า guard พี่น้อง — บางเว็บต้องเปิด dropdown ซ้ำจริง
# (ค่าที่โชว์เป็น default ที่ยังไม่ apply) บล็อกตายจะใช้งานไม่ได้
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
    """W_delete_all_intent: ค่าเงื่อนไข key=value ใน goal ("userrole=ess" -> ["ess"]) หรือ []"""
    # ตัด URL แล้วใน _goal_condition_pairs (query string ของ URL เข้ารูป key=value)
    values: list[str] = []
    seen: set[str] = set()
    # W_column_aware_rows: ตัวนี้เหลือแค่ให้ค่าสำหรับข้อความรายงาน การตัดสินแถวที่ตรงเงื่อนไข
    # ย้ายไป _row_matches_condition() ซึ่งใช้ทั้ง field และ value
    for _, value in _goal_condition_pairs(goal):
        if value.lower() not in seen:
            seen.add(value.lower())
            values.append(value)
    return values


# ══════════════════════════════════════════════════════════════════════
# โซน 6: อ่านตารางบนหน้า + guard ก่อนลบ
#   ทำอะไร: นับแถวที่ตรงเงื่อนไข (เล็งคอลัมน์) และค่าคงที่ของ guard งานลบ
#   ทำงานยังไง: _VISIBLE_TABLE_ROWS_JS อ่านหัวตาราง+เซลล์ -> _count_rows_matching_condition() นับ AND ทุกคู่ (require_columns ตามทิศของการตัดสิน)
# ══════════════════════════════════════════════════════════════════════
# W_delete_all_intent: แถวข้อมูลที่มองเห็นของตารางที่ใหญ่สุด — generic (<table> + ARIA grid
# pattern เดียวกับ perception._EXTRACT_TABLE_JS) ตัดแถวหัวตาราง (มีคำ "User Role" นับเกิน)
# W_column_aware_rows: คืนเซลล์แยกคอลัมน์ + หัวตาราง — เทียบทั้งแถว "ess" ตรงกับ "Jessica"/
# "ess.irhrg0"/"Assessed" นับผิดทั้งสองทิศ (ปล่อยลบจากตารางยังไม่กรอง / งานที่เสร็จแล้วจบไม่ได้)
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
    """W_delete_all_intent/W_column_aware_rows: (หัวตาราง, แถวแยกเซลล์) ของตารางใหญ่สุด หรือ None
    ห้าม throw หัวตารางว่างได้ (ผู้เรียกถอยไปเทียบทั้งแถว)"""
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
    """W_column_headers_fallback: เล็งคอลัมน์ได้ไหม — ต้องมีหัวตาราง *และ* แถวแยกเซลล์จริง
    (ตาราง div ล้วนถูกยัดทั้งแถวเป็นเซลล์เดียว = ไม่มีคอลัมน์)"""
    return bool(headers) and any(len(row) > 1 for row in rows)


def _column_index_for_field(headers: list[str], field: str) -> Optional[int]:
    """W_column_aware_rows: index คอลัมน์ที่ตรง field ใน goal (ใช้ _field_names_match ตัวเดียวกับ
    W_filter_scope_guard) None = ตัดสินไม่ได้ -> ผู้เรียกต้อง fail-open"""
    if not field:
        return None
    for index, header in enumerate(headers):
        if _field_names_match(field, _normalized_field_name(header)):
            return index
    return None


def _cell_matches_value(cell: str, value: str) -> bool:
    """W_column_aware_rows: เซลล์มีค่าที่ขอไหม — ตรงทั้งเซลล์ หรือเป็นคำเต็มคั่นด้วยช่องว่าง
    ("Senior ESS") ตัดด้วยช่องว่าง ไม่ใช่ regex word boundary (ซึ่งถือ "." เป็นตัวคั่น
    "ess.irhrg0" จะตรง "ess") ตัด "Jessica"/"Assessed" ด้วย"""
    cell_norm = (cell or "").strip().lower()
    value_norm = (value or "").strip().lower()
    if not value_norm:
        return False
    return cell_norm == value_norm or value_norm in cell_norm.split()


def _row_matches_condition(
    headers: list[str], cells: list[str], pairs: list[tuple[str, str]],
) -> bool:
    """W_column_aware_rows: ทุกคู่ field=value ต้องตรงในแถวเดียวกัน (AND) — เล็งคอลัมน์ได้
    เทียบเฉพาะเซลล์ ไม่ได้ถอยไปเทียบทั้งแถว"""
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
    """W_delete_all_intent: (แถวที่ตรงทุกคู่, แถวทั้งหมดที่เห็น) หรือ None ถ้าเช็คไม่ได้

    W_column_headers_fallback: require_columns=True = เล็งคอลัมน์ไม่ได้ให้คืน None แทนเทียบทั้งแถว
    (บั๊ก W97 "ess" ตรง "Jessica" แบบเงียบ) ผู้เรียกเลือกตามทิศ:
      - ทิศปลอดภัย (บล็อกไว้ก่อน) -> False: เดาเกินอย่างมากเสีย step
      - ทิศอันตราย (หลักฐานว่าจบงาน) -> True: นับเกิน = อ้างว่าเสร็จทั้งที่ยังเหลือ"""
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


# W_delete_all_intent: click ที่ทำลายข้อมูล — ทั้ง action type กลุ่มยืนยัน และ click ที่ label
# บอกว่าลบ (defense-in-depth แบบ permission/rules.py)
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

# W_undefined_quota: ชื่อนี้ถูกใช้ตั้งแต่ W80 แต่ไม่เคยประกาศ guard ทั้งสองโยน NameError แล้ว
# W_loop_crash (W77) กลืนเป็น success=False — guard ชุด W80 ไม่เคยทำงานจริง
# เทสต์มองไม่เห็นเพราะ assert แค่ success is False — เทสต์ guard ต้อง assert ข้อความ nudge เสมอ
# 2 = เท่า _MAX_DESTRUCTIVE_BEFORE_FILTER_RETRIES (guard ตระกูลเดียวกัน)
_MAX_DELETE_ALL_UNVERIFIED_RETRIES = 2

_ROW_ACTION_LABEL_RE = re.compile(r"\b(edit|view details|delete|download|pencil)\b", re.IGNORECASE)

# W64[7.1]: ใช้เช็คว่า click ที่เพิ่งสำเร็จคือการกด Search จริงหรือไม่ (ถ้าใช่ ล้าง
# filter_dirty_since_search ทันที) — ครอบคลุมทั้งไทย/อังกฤษเหมือน keyword set อื่นในไฟล์นี้
_SEARCH_LABEL_RE = re.compile(r"\bsearch\b|ค้นหา", re.IGNORECASE)

# W_rowaction_own_quota: เคยใช้โควตาของ W63[7.2] (คนละ guard) ปรับอันหนึ่งกระทบอีกอัน —
# แยกโควตา ค่าเท่าเดิม (2)
_MAX_PREMATURE_ROW_ACTION_BEFORE_SEARCH_RETRIES = 2

# ══════════════════════════════════════════════════════════════════════
# โซน 7: ชื่อ tool + จัดหมวดความล้มเหลว (telemetry)
#   ทำอะไร: รู้จัก tool ที่มีจริง และจัดหมวดสาเหตุที่ step ล้ม ลง step_trace
#   ทำงานยังไง: _classify_step_failure() จับ pattern ข้อความ ActionResult จากเจาะจงไปกว้าง
# ══════════════════════════════════════════════════════════════════════
# W_unknown_tool: ชื่อ tool ทั้งหมดที่มีอยู่จริง (ตรงกับที่ llm.py ประกาศให้ทุก provider) —
# ใช้ปฏิเสธชื่อที่โมเดลมโนขึ้นเองก่อนจะหลุดไปถึง actions.execute() ดูจุดใช้งานในลูปหลัก
_KNOWN_TOOL_NAMES = frozenset({"browser_action", "request_user_input", "finish_task"})

# W_step_trace (failure taxonomy): เดิมไม่มี field บอกว่า step ล้มเพราะอะไร — จัดหมวดจากข้อความ
# ActionResult (deterministic) เรียงจากเจาะจงไปกว้าง เพราะข้อความหนึ่งอาจตรงหลายรูปแบบ
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
    """หมวดความล้มเหลวของ step — "ok" ถ้าสำเร็จ, "other" ถ้าไม่ตรงหมวดไหน ห้าม throw"""
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

# ══════════════════════════════════════════════════════════════════════
# โซน 8: ตรวจ DOM จริงก่อนยอมรับ finish_task
#   ทำอะไร: หาหลักฐานจากหน้าเว็บ: record ที่เหลือ / รายการที่สร้างอยู่ในตาราง / validation error
#   ทำงานยังไง: อ่าน "(N) Records Found" หรือรูปแบบ generic (อ่านซ้ำเมื่อได้ 0), table body, error ในฟอร์ม — อ่านไม่ได้ = ไม่บล็อก (fail-safe)
# ══════════════════════════════════════════════════════════════════════
_PREMATURE_ROW_ACTION_BEFORE_SEARCH_NUDGE_TEMPLATE = (
    "This action is rejected — you filled/selected a value in the field '{prev_label}' on the "
    "previous step but have not yet pressed the Search button (or Enter) to apply that "
    "filter. The table you see now is still the OLD result from before the new filter, not "
    "the genuinely filtered one — do NOT click the '{label}' row action on this row right "
    "now. Press Search first, then check the new round of indexed elements to confirm the "
    "table really is filtered to the condition you wanted, before choosing an action on a row."
)


# W63[7.2]: เปิดตาม tool_input["verify_text"] ที่ LLM ส่งมาเอง (llm.py::_FINISH_TASK_PARAMS) ไม่ใช่
# intent keyword — ครอบคลุมงานสร้าง/แก้ไข/บันทึกที่คาดว่าจะโผล่ในตาราง
_MAX_PREMATURE_TABLE_VERIFY_RETRIES = 2

_PREMATURE_TABLE_VERIFY_NUDGE_TEMPLATE = (
    "This finish_task(success=true) is rejected — checking the real DOM of the results table "
    "on the current page finds the text \"{text}\" (given in verify_text) in no row at all. "
    "Never treat the entry as created/saved while the table genuinely does not show it. Check "
    "whether the submit actually went through (is a validation error still showing?), whether "
    "you have navigated back to the correct list page, or whether you need to refresh/search "
    "again before it appears, before calling finish_task(success=true) again."
)

# W64[7.2] (Add-Action Idempotency Lock): agent บันทึกสำเร็จ (มี toast) แต่ค้นหาก่อน AJAX reload
# เสร็จ ไม่เจอ แล้วกรอกฟอร์ม Add ใหม่ซ้ำจน error ข้อมูลซ้ำ — ต่อท้าย nudge เมื่อ task เคยเห็น
# toast ยืนยันแล้ว: ห้ามตีความ "หาไม่เจอ" เป็น "ยังไม่ได้บันทึก"
_TOAST_CONFIRMED_NO_RECREATE_SUFFIX = (
    " *** IMPORTANT: earlier in this same task an action genuinely detected a toast/save-"
    "success confirmation (see the earlier step in the history) — the data really was saved. "
    "Not finding it in the table right now is NOT evidence that it wasn't. NEVER press "
    "Reset/clear the form and fill in the creation form again (that produces duplicate "
    "entries and duplicate-data errors) — the only thing to do is wait a moment and re-query "
    "the table (e.g. press Search again). ***"
)


# W22: "(N) Records Found" ของ OrangeHRM — selector ที่ user ยืนยันจาก DOM จริง + :has-text() สำรอง
# W_record_count_generic (P3.9): เดิมผูก OrangeHRM อย่างเดียว เว็บอื่นไม่มี guard เลย — เพิ่มชั้น
# generic คง fail-safe (อ่านไม่ได้ = None = ไม่บล็อก)
# W_record_count_picks_wrapper (live run 2026-08-28): selector คั่น comma คืนตามลำดับ DOM ไม่ใช่ลำดับ
# ที่เขียน และ :has-text() match บรรพบุรุษทุกชั้น .first จึงได้ div ครอบทั้งหน้า -> regex อาจเจอเลขอื่น
# แยกเป็นลิสต์ไล่ทีละตัว (แบบ actions.py::_find_visible_modal_confirm_button) เลือกข้อความสั้นสุด
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

# เรียงจากเจาะจงไปกว้าง ตัวแรกที่ match ชนะ — ไม่รับตัวเลขลอยๆ (ราคา/วันที่/เลขหน้า)
# เดาผิด = บล็อก finish_task ที่ถูก
_RECORD_COUNT_PATTERNS = (
    # OrangeHRM: "(41) Records Found"
    re.compile(r"\((\d+)\)\s*Records?\s*Found", re.IGNORECASE),
    # "42 results found" / "42 results" / "42 items" — เจาะจงกว่า "of N" ด้านล่างจึงมาก่อน
    re.compile(r"\b([\d,]+)\s+(?:results?|items?|records?|entries)\b", re.IGNORECASE),
    # ไทย: "42 รายการ" / "ทั้งหมด 42 รายการ"
    re.compile(r"([\d,]+)\s*รายการ"),
    # "Showing 1-10 of 42" — กว้างสุด ("page 2 of 5" ก็ match) จึงไว้ท้ายสุด
    re.compile(r"\bof\s+([\d,]+)\b", re.IGNORECASE),
)

# ข้อความที่แปลว่า "ไม่มีผลลัพธ์เลย" (= 0) — ต้องเช็คก่อน pattern ตัวเลขเสมอ เพราะบางอันมี
# เลข 0 อยู่ในประโยคอยู่แล้ว บางอันไม่มีเลขเลย
_RECORD_COUNT_ZERO_TEXTS = (
    "no records found", "no results", "no matching records", "no data",
    "ไม่พบข้อมูล", "ไม่พบรายการ", "ไม่มีข้อมูล",
)

# W68b: agent claim "0 รายการ" ทั้งที่ตารางโชว์ "(16) Records Found" — อ่าน DOM ครั้งเดียวตอน
# finish_task อาจตรงช่วง AJAX ยังไม่ re-render (networkidle 4s ไม่รับประกัน) อันตรายเฉพาะ "0"
# (ปล่อยผ่านทันที) ">0" ผิดแค่เสีย nudge — ได้ "0" ให้รอแล้วอ่านซ้ำ เชื่อรอบสอง
_ZERO_RECORD_RECHECK_DELAY_SECONDS = 0.8


async def _scan_remaining_target_records_once(page: Page) -> Optional[tuple[int, str]]:
    """W22: อ่าน "(N) Records Found"/"No Records Found" จาก DOM -> (จำนวนที่เหลือ, ข้อความดิบ)
    หรือ None ถ้าหน้าไม่มี element นี้ (ปล่อยผ่าน) "No Records Found" = 0 ห้าม throw"""
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
    """W68b: เหมือน _scan_remaining_target_records_once() แต่อ่านซ้ำเมื่อได้ "0" — None/>0 คืนทันที"""
    result = await _scan_remaining_target_records_once(page)
    if result is not None and result[0] == 0:
        await asyncio.sleep(_ZERO_RECORD_RECHECK_DELAY_SECONDS)
        recheck = await _scan_remaining_target_records_once(page)
        if recheck is not None and recheck[0] > 0:
            return recheck
    return result


# W63[7.2] (Strict Table Assertion): เรียงจากเจาะจง (OrangeHRM) ไปกว้าง (<tbody>/[role=rowgroup]/
# class table-body) — generic โดยตั้งใจ เพราะ tbody/rowgroup เป็นมาตรฐาน HTML/ARIA
_TABLE_BODY_SELECTOR = (
    '.oxd-table-body, table tbody, tbody, [role="rowgroup"], '
    '[class*="table-body" i], [class*="tablebody" i]'
)


async def _scan_created_item_in_table(page: Page, verify_text: str) -> bool:
    """W63[7.2]: True ถ้าเจอ verify_text (case-insensitive) ใน table body หรือหน้าไม่มี table body
    (ปล่อยผ่าน) — False เฉพาะมี table body แต่ไม่มีข้อความ (รวมตารางว่าง/"No Records Found")"""
    try:
        locator = page.locator(_TABLE_BODY_SELECTOR).first
        if await locator.count() == 0:
            return True
        text = (await locator.inner_text(timeout=_DOM_CHECK_TIMEOUT_MS)).strip()
    except Exception:
        return True
    return verify_text.lower() in text.lower()


async def _scan_validation_errors(page: Page, within_form: bool = False) -> list[str]:
    """ข้อความ validation error ที่มองเห็นบนหน้า (สูงสุด 5) หรือ [] — เรียกเฉพาะก่อนยอมรับ
    finish_task(true) และหลัง fill/click (ไม่ทุก step) ห้าม throw สแกนไม่ได้ = ไม่เจอ

    within_form (W20 Task12): True = สแกนเฉพาะใน <form> กัน banner นอกฟอร์มที่ class มีคำ error
    ใช้กับ hard-stop guard หลัง fill/click; False = ทั้งหน้า (guard ก่อน finish_task)
    W19 (latency): ใส่ _DOM_CHECK_TIMEOUT_MS ทุกจุด — element ที่ detach ระหว่างสแกนจะรอ default 30s"""
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


# ══════════════════════════════════════════════════════════════════════
# โซน 9: validation error หลัง action
#   ทำอะไร: ตัดสินว่า error หลัง fill/กดบันทึก ต้องหยุด task หรือให้ agent แก้เอง
#   ทำงานยังไง: กรอง hint คำเดียว (Required/Invalid) และ error ชนิด required ที่ agent เติมเองได้ error fatal (login ผิด/ข้อมูลซ้ำ) บังคับความจริงทันที
# ══════════════════════════════════════════════════════════════════════
# W20 (Task12, UI Validation Error Detection): _scan_validation_errors() เดิมเรียกแค่ก่อน finish_task
# agent ที่วน fill/click ไม่จบจึงไม่เคยโดนเช็ค — เช็คทันทีหลัง action ที่ยืนยันส่งฟอร์มด้วย
# เดิมไม่เช็คหลัง fill เพราะฟอร์มหลายช่องโชว์ "* Required" ให้ช่องที่ยังไม่กรอก (false-positive)
# (2026-08-06) user ขอให้ครอบ fill ด้วย — กรอง bare "* Required" ด้วย _is_bare_required_message()
# error ที่มีเนื้อหาจริงยัง hard-stop เสมอ nudge-retry ก่อน finish_task เหลือเป็น backstop
# (action อื่นอย่าง select/check หรือปุ่มที่ label ไม่ตรง _FORM_SUBMIT_LABEL_KEYWORDS)
_FORM_SUBMIT_LABEL_KEYWORDS = (
    "save", "submit", "update", "change password", "confirm",
    "บันทึก", "ยืนยัน", "เปลี่ยนรหัสผ่าน", "อัปเดต", "แก้ไข",
)

# bare "* Required"/"Required." (OrangeHRM) — ต่างจาก error มีเนื้อหา ("Should have at least 7 characters")
# W_bare_invalid_is_a_field_hint (gate 2026-09-07, search_no_results ตก 100%): autocomplete
# Employee Name ขึ้น "Invalid" เมื่อชื่อไม่มีอยู่ — ซึ่งคือผลที่ goal ต้องการ ไม่ใช่ความล้มเหลว
# ป้ายคำเดียวไม่บอกว่าผิดยังไง ต่างจาก "Invalid email format" ที่ยังต้องหยุด
_BARE_FIELD_HINT_MESSAGE_RE = re.compile(
    r'^[\*\s]*(?:required|invalid)[\.\!]?$', re.IGNORECASE,
)


def _is_bare_required_message(text: str) -> bool:
    """ข้อความเป็นแค่ป้ายสถานะคำเดียว ("* Required"/"Invalid") — false positive จากช่องพี่น้องที่ยัง
    ไม่กรอก หรือ autocomplete ที่หาไม่เจอ กรองทิ้งก่อน hard-stop"""
    return bool(_BARE_FIELD_HINT_MESSAGE_RE.match((text or "").strip()))


# W_required_error_survives_the_fix (gate 2026-09-07, rag_permission/rag_integration/long_flow):
# SauceDemo ไม่ล้างแบนเนอร์ "First Name is required" จนกว่าจะส่งใหม่ agent กรอกแก้แล้วแต่สแกน
# หลัง fill เห็น error ล้าสมัยแล้วฆ่างาน — error ชนิด required ที่อ้างชื่อช่องที่เพิ่งกรอกถือว่าล้าสมัย
# แคบโดยเจตนา: error ค่าผิดรูปแบบยังหยุดเหมือนเดิม
_REQUIRED_ERROR_WORDS = ("required", "ต้องกรอก", "จำเป็นต้องระบุ", "ห้ามเว้นว่าง")


def _is_required_field_error(text: str) -> bool:
    """ข้อความบอกว่า "ช่องนี้ต้องกรอก" — agent แก้เองได้เสมอ

    W_required_error_is_not_a_dead_end (gate 2026-09-07, 'Error: Postal Code is required'): agent
    ลืมกรอกช่องแล้วกด Continue ระบบยุติงานเพื่อขอค่าจาก user ทั้งที่ agent เติมเองได้ — hard-stop
    มีไว้สำหรับ error ที่แก้ได้ด้วยค่าใหม่จาก user เท่านั้น required จึงปล่อยให้ loop เดินต่อ
    error ค่าผิด (invalid format/at least N/already exists) ยังหยุดเหมือนเดิม"""
    return any(w in (text or "").lower() for w in _REQUIRED_ERROR_WORDS)


def _label_looks_like_form_submit(label: str) -> bool:
    lowered = (label or "").lower()
    return any(kw in lowered for kw in _FORM_SUBMIT_LABEL_KEYWORDS)


def _should_check_validation_error_after_action(action_type: str, label: str) -> bool:
    """ควรเช็ค validation error ทันทีหลัง action ที่สำเร็จนี้ไหม — fill ทุกครั้ง (กรอง bare
    "* Required" ที่จุดเรียก) click/submit เฉพาะ label ปุ่ม Save/Submit/Confirm/Update
    (click ทั่วไปเสี่ยง false positive จาก alert บนหน้าที่เพิ่ง navigate ไป)"""
    if action_type == "fill":
        return True
    if action_type in ("click", "submit"):
        return _label_looks_like_form_submit(label)
    return False


# W65[2] (Error Passthrough): guard ก่อน finish_task ยังให้ LLM nudge/retry แม้ error ประเภทที่ retry
# ไม่มีวันหาย (login ผิด, ข้อมูลซ้ำ, ไม่มีสิทธิ์) — keyword นี้แยก error "fatal" (ต้องข้อมูลใหม่จาก
# user) ออกจาก error อื่น (อาจเป็น timing ให้ LLM ลองแก้เองตามเดิม)
_FATAL_VALIDATION_ERROR_KEYWORDS = (
    "invalid credentials", "incorrect password", "invalid username or password",
    "already exists", "unauthorized", "not authorized", "permission denied",
    "ไม่ถูกต้อง", "ผิดพลาด", "มีอยู่แล้ว", "ไม่มีสิทธิ์",
)


def _is_fatal_validation_error(text: str) -> bool:
    """W65[2]: error ที่ agent แก้เองไม่ได้ — ข้าม nudge-retry บังคับความจริงลงผลลัพธ์ทันที"""
    lower = (text or "").lower()
    return any(kw in lower for kw in _FATAL_VALIDATION_ERROR_KEYWORDS)


# ══════════════════════════════════════════════════════════════════════
# โซน 10: จับลูป + recovery
#   ทำอะไร: ตรวจ action ซ้ำเดิม / label เดิม / วนเป็นคาบ 2-4
#   ทำงานยังไง: ตัด completed_plan_step ก่อนเทียบ -> trip แล้วบังคับ go_back/scroll (_force_loop_recovery) ก่อนยอมจบ task
# ══════════════════════════════════════════════════════════════════════
# W5: loop guard — บางโมเดล (Llama บน Groq) วนสั่ง action เดิมเป๊ะแม้ถูกเตือน ครบจำนวนนี้หยุด
_MAX_CONSECUTIVE_IDENTICAL_ACTIONS = 3

# W_same_label_loop (OrangeHRM live): agent ติ๊ก "Select row" คนละแถวไปเรื่อยๆ index ต่างทุกครั้ง
# guard คาบ 1 และ cycle detector จับไม่ได้ วนจน user กด Stop — label เดิม action ชนิดเดิมติดกัน
# เกินนี้ = ไม่คืบหน้า threshold หลวมกว่าคาบ 1 เพราะติ๊กหลายแถวก่อนลบเป็นรูปแบบถูกต้อง
# trip แล้วใช้ _force_loop_recovery() (ไม่ฆ่า task ทันที)
_MAX_CONSECUTIVE_SAME_LABEL_ACTIONS = 4

# W_already_logged_in_but_told_to_log_in: บอกโมเดลว่าระบบล็อกอินให้แล้วเฉพาะช่วง step แรกๆ
# พอเดินไปได้สักพักมันเห็นหน้าหลังล็อกอินเองแล้ว ไม่ต้องจ่ายค่าบรรทัดนี้ทุกเทิร์นจนจบงาน
_MAX_ALREADY_LOGGED_IN_REMINDER_STEPS = 3

# W_session_drift: login ใหม่ให้เองกลางทางได้สูงสุดเท่านี้ (session หมดอายุใน task ยาว)
# ไม่วนไม่รู้จบถ้า credential ใช้ไม่ได้จริง
_MAX_MID_TASK_RELOGINS = 2

# (2026-07-13) guard คาบ 1 จับแค่ AAAA — agent วนสลับ ABAB (go_back -> click ...) ไม่โดน
# (2026-07-15) generalize เป็น _is_repeating_cycle() คาบ 2.._MAX_CYCLE_PERIOD (คาบ 1 แยกใช้
# _MAX_CONSECUTIVE_IDENTICAL_ACTIONS เพราะ trigger เร็วกว่า) cap 4: คาบยาวกว่านี้เจอยาก และต้อง
# รอ period*2 action กว่าจะยืนยัน ไม่คุ้ม step ที่เสีย
_MAX_CYCLE_PERIOD = 4
_MIN_CYCLE_REPEATS = 2  # ทุกคาบ (2 ขึ้นไป) ต้องเห็นครบกี่รอบถึงจะถือว่าติด loop
_MAX_CYCLE_WINDOW = _MAX_CYCLE_PERIOD * _MIN_CYCLE_REPEATS

# W31: guard loop เดิมจบ task ทันที — user ขอให้บังคับ action อื่นก่อน: go_back (หลุดจาก sub-flow
# เช่น Shorts/modal) แล้ว scroll (ตัวเลือกอยู่นอกจอ) เกินนี้แล้วยังวนค่อยยอมแพ้
_MAX_FORCED_LOOP_RECOVERIES = 2
_LOOP_RECOVERY_ACTIONS: list[dict] = [{"type": "go_back"}, {"type": "scroll", "direction": "down"}]


# W29 (Loop-guard blind spot): completed_plan_step ติดมาเฉพาะครั้งแรกที่ step เสร็จ การเทียบ dict
# ทั้งก้อนจึงเห็น action เดิมเป็นคนละตัว (click(22)->click(25)->click(22)... ไม่รู้จบ) — ตัด key นี้
# ก่อนเทียบ/เก็บเข้า recent_actions (ไม่กระทบ dispatch จริงที่ใช้ tool_input ดิบ)
def _cmd_for_repeat_comparison(cmd: dict) -> dict:
    if "completed_plan_step" not in cmd:
        return cmd
    return {k: v for k, v in cmd.items() if k != "completed_plan_step"}


def _is_repeating_cycle(window: list[dict], period: int) -> bool:
    """window (ยาว period * _MIN_CYCLE_REPEATS พอดี) วนคาบ `period` จริงไหม — ต้องมีค่าต่างกัน
    อย่างน้อย 2 ในคาบ ไม่งั้นทับกับ guard คาบ 1"""
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
    """คาบแรก (2.._MAX_CYCLE_PERIOD สั้นก่อน) ที่ recent_actions วนซ้ำ หรือ None"""
    for period in range(2, _MAX_CYCLE_PERIOD + 1):
        window = period * _MIN_CYCLE_REPEATS
        if _is_repeating_cycle(recent_actions[-window:], period):
            return period
    return None

# ══════════════════════════════════════════════════════════════════════
# โซน 11: RAG / memory / vision / permission query
#   ทำอะไร: ค่าคงที่ของ context เสริมที่แนบให้ LLM ทุก step
#   ทำงานยังไง: คู่มือ + long-term memory ดึงใหม่เมื่อหน้าเปลี่ยน, permission query แคบตาม action ปัจจุบัน
# ══════════════════════════════════════════════════════════════════════
# W6[B]: chunk คู่มือที่แนบทุก step ของ per-step loop (ดึงใหม่ตาม page_text)
# W_token_trim (P1/Q2): 3 -> 2 — chunk อันดับ 3 marginal อยู่แล้ว ประหยัด ~500 char ต่อ step
_RAG_CHUNKS_PER_STEP = 2

# W7[A] (long-term): เหมือน _RAG_CHUNKS_PER_STEP แต่สำหรับ long_term_memory.recall()
# (ประวัติ task run อื่นก่อนหน้า แทนคู่มือที่ user ป้อน) — ดึงใหม่ทุก step เหมือนกัน
_LONG_TERM_MEMORY_CHUNKS_PER_STEP = 2  # W_token_trim (P1/Q2): 3 -> 2

# W9[A] vision fallback (Gemini เท่านั้น — ดู llm.py::describe_screenshot): เฉพาะ action ที่พึ่ง
# element visibility (scroll/goto/wait ล้มด้วยเหตุผลอื่น ไม่เกี่ยวกับ overlay)
_VISION_FALLBACK_ACTION_TYPES = {
    "click", "fill", "select", "check", "submit", "delete", "purchase", "pay",
    # W50: press_key พึ่ง element visibility เหมือน click (ต้อง focus element ที่มองเห็น
    # ได้จริงก่อนถึงจะกด key ได้ผล) — failure mode เดียวกับ click ที่มี popup/overlay บัง
    "press_key",
}

# W50 (client-side action verification): action ที่ถ้าทำงานจริงควรเปลี่ยนหน้าเสมอ (scroll/wait/
# goto ไม่เปลี่ยนก็ปกติ) — กลุ่มเดียวกับ vision fallback (fill รวมด้วย เพราะ label เปลี่ยนตามค่า)
_VERIFICATION_SIGNAL_ACTION_TYPES = _VISION_FALLBACK_ACTION_TYPES

# W7[B] (RAG permission): query แคบตาม action ปัจจุบัน (_build_permission_query) — query ระดับ goal
# กว้างเกิน (saucedemo: goal พูด "Checkout" ครั้งเดียว chunk Checkout ติดมาแทบทุก step)
# k=1 เพราะคู่มือมี ~11 chunk สั้น อันดับ 2 หลุดเข้ามาแบบผิดๆ ได้ง่าย
_PERMISSION_RAG_CHUNKS_PER_STEP = 1


def _build_permission_query(cmd: dict, label: str) -> str:
    """ประกอบ query แคบเฉพาะ action นี้ (ไม่ใช่ทั้ง goal) ไว้ค้นคู่มือว่ามีกฎเกี่ยวกับ
    action นี้ไหม — goto ไม่มี label (ไม่มี index ให้จับคู่) ใช้ url แทน"""
    target = label or cmd.get("url", "")
    return f"{cmd.get('type', '')} {target}".strip()

# ══════════════════════════════════════════════════════════════════════
# โซน 12: บีบ context / ลด token
#   ทำอะไร: ยุบ history เก่าไม่ให้ token โตตามจำนวน step
#   ทำงานยังไง: stub snapshot เก่า -> ยุบ user turn เก่าเหลือ Goal+stub -> ตัดบล็อกกฎเก่า -> compaction แทน step เก่าด้วย digest (splice ตาม wire format ของ provider)
# ══════════════════════════════════════════════════════════════════════
# W7[A]/W22 (context compaction ทุก provider): stateless API ส่ง messages ทั้งก้อนทุก step token
# โตตามจำนวน step — เกิน _COMPACT_AFTER_STEPS ตัด step เก่า (เก็บ _KEEP_RECENT_STEPS ล่าสุด) แทนด้วย
# digest จาก ShortTermMemory (provider-agnostic) เฉพาะฟังก์ชัน splice แยกตาม wire format
# (_compact_gemini/anthropic/groq_messages เลือกผ่าน _llm_backend())
# W_token_trim (P2/M1): 6 -> 4 — digest + stale-snapshot stub จัดการ bulk แล้ว compact ถี่ขึ้นได้
_COMPACT_AFTER_STEPS = 4
_KEEP_RECENT_STEPS = 3

# W_token_trim (P2/M1): stub "Current page:" ของ user turn เก่า — snapshot เก่าไร้ค่าเมื่อมีใหม่กว่า
# (W19 สั่งให้ยึด snapshot ล่าสุด) เก็บ 2 อันล่าสุดเต็มไว้ให้เทียบ "ก่อน/หลัง action ล่าสุด"
_STALE_SNAPSHOT_MARKER = "\n\nCurrent page:\n"
_SUPERSEDED_SNAPSHOT_STUB = (
    "[snapshot from an earlier step — superseded; act only on the latest snapshot below]"
)


def _stub_snapshot_in_text(text: str) -> Optional[str]:
    """text ที่แทน page snapshot ด้วย stub หรือ None ถ้าไม่มี snapshot — page_text ไม่มี "\\n\\n"
    ข้างใน block จึงจบที่ "\\n\\n" ตัวถัดไปหรือท้าย string"""
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
    """W_token_trim (P2/M1): stub page snapshot ของ user turn ทุกอันยกเว้น keep_last_full อันท้าย
    provider-agnostic (str content หรือ Gemini parts[0].text; tool_result เป็น list จึงข้ามเอง) ไม่ throw"""
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


# W_token_cut W5 (W_prompt_audit 2026-09-02): assistant history โต ~4.5-5k tok ต่อ call ไม่มีเพดาน
# (_dedupe_stale_snapshots ยุบแค่ snapshot ส่วนกฎ/plan/manual ค้างทุก turn ทั้งที่ส่งสดทุก turn)
# ยุบ user turn เก่า (เกิน _W5_KEEP_RECENT_FULL_TURNS) เหลือ Goal + stub — idempotent,
# provider-agnostic, ไม่ throw ไม่แตะ tool_result/function_call (ต้องจับคู่ call_id) และ nudge turn
# keep=1: turn ล่าสุด + turn ปัจจุบัน = โมเดลเห็นครบ 2 turn
_W5_KEEP_RECENT_FULL_TURNS = 1
_W5_SUPERSEDED_TURN_STUB = (
    "[an earlier step's full context — page snapshot, indexed elements, rules, plan, "
    "recent-action list — has been omitted here to keep the conversation short. It is "
    "superseded. Act only on the latest turn below plus the digest of earlier steps.]"
)
# user turn ของ step จริงขึ้นต้นด้วยบรรทัดนี้ (llm._build_user_turn_text) — แยกจาก nudge/tool_result
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
    """W_token_cut W5: ยุบ user turn เก่าเป็น "Goal: ...\\n\\n<stub>" -> (messages, ตัวอักษรที่ตัดได้)
    idempotent ไม่ throw"""
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


# W_token_cut W7: บล็อกกฎ gated (~3.9k tok) ต่อท้ายทุก user turn สำเนาเก่าไร้ค่า (W19 ยึด turn
# ล่าสุด) — ตัดจาก header ถึงท้าย string เหลือบรรทัดอ้างอิง keep_last_full=0 (turn ปัจจุบันยังเต็ม)
# idempotent, provider-agnostic, ไม่ throw
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


# W50 (delta digest): เดิม compaction ทุกรอบสรุปซ้ำตั้งแต่ step 1 digest โตไม่มีเพดานและส่งซ้ำทุก step
# — สะสมเฉพาะ delta (digest_lines/digest_upto_step ใน run_task) และ cap ด้วยค่านี้
_MAX_DIGEST_LINES = 20


def _build_history_digest(memory: ShortTermMemory, upto_step: int, from_step: int = 1) -> str:
    """สรุป step from_step..upto_step (ไม่รวม goto step 0) เป็น bullet บรรทัดละ step
    จาก ShortTermMemory.all() (ไม่เคยตัด ครบกว่า raw messages) provider-agnostic
    from_step (W50): compaction รอบหลังสรุปเฉพาะ step ใหม่"""
    entries = [h for h in memory.all() if from_step <= h.get("step", 0) <= upto_step]
    if not entries:
        return ""
    # W_token_trim (P1/Q1): clip result หัว+ท้ายเหมือน memory.py summaries — digest
    # ค้างอยู่ทั้ง task ยิ่งต้องไม่ฝัง result เต็ม (เช่น ตาราง read_page_data)
    return "\n".join(f"- step {h['step']}: {h['cmd']} -> {clip_result(h['result'])}" for h in entries)


_DIGEST_PREFIX = "[Digest of earlier steps, compacted to keep the conversation from growing too long]"


def _compact_gemini_messages(messages: list, cut_at: int, digest_text: str) -> list:
    """ตัด messages[:cut_at] แล้วฝัง digest เป็นส่วนแรกของ turn แรกที่เหลือ (ไม่แทรก turn ใหม่ กัน
    ลำดับ user/model ของ Gemini เพี้ยน) cut_at มาจาก step_boundaries จึงเป็น user turn เสมอ
    รูปแบบไม่ตรงคาด คืน messages เดิม ไม่ throw"""
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
    """W22: แบบเดียวกับ _compact_gemini_messages() แต่ shape Anthropic ({"role":"user","content":str})
    turn แรกไม่ใช่ plain text (เช่นโดน nudge แทรก) คืน messages เดิม ไม่ throw"""
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
    """W22: แบบ Anthropic แต่ Groq เก็บ system prompt เป็น messages[0] — ต้องกันไว้ไม่ให้ถูกตัดทิ้ง"""
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


# ══════════════════════════════════════════════════════════════════════
# โซน 13: utility หน้าเว็บ + background
#   ทำอะไร: ปิด JS dialog อัตโนมัติ, ข้อความ nudge ตาม provider, guard ฟอร์ม login, งานเบื้องหลัง
#   ทำงานยังไง: ทุกตัวห้าม throw — เป็นส่วนเสริม ไม่ใช่สิ่งที่ loop พึ่ง
# ══════════════════════════════════════════════════════════════════════
def _make_dialog_handler(memory: ShortTermMemory, verbose: bool):
    """W9[A]: auto-dismiss JS dialog — ถ้าค้าง action ถัดไปทุกตัว timeout เงียบๆ เลือก dismiss เสมอ
    (confirm() บางเว็บใช้กับการลบ accept เองขัด human-in-the-loop) บันทึกลง memory ให้ LLM รู้

    W12: page จาก session ถูกใช้ซ้ำข้าม run_task() handler จึงผูกซ้อนหลายตัว ตัวหลังเจอ error
    เพราะ dialog ถูกจัดการแล้ว — ห่อ try/except กัน handler เก่าพัง task ปัจจุบัน"""
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
    """ข้อความเตือนที่ต่อเข้า messages ตรงๆ — shape ตาม provider (Gemini SDK throw KeyError เมื่อ
    เจอ {"role","content"}) เจอตอนทดสอบ W7[A] Test Case A ผ่าน Gemini"""
    if provider == "gemini":
        return {"role": "user", "parts": [{"text": text}]}
    return {"role": "user", "content": text}

# (2026-07-13) โมเดลเล็ก (Gemini flash-lite) wait กลาง login form แล้วกด Login ก่อนกรอก password —
# code guard: มีช่อง password ว่างที่มองเห็น ห้าม action อื่นนอกจาก fill (ปัญหาคือฟอร์มไม่ครบ
# ไม่ใช่แค่ wait) retry จำกัด เกินโควตาปล่อยผ่าน
_MAX_PREMATURE_LOGIN_SKIP_RETRIES = 2
_PREMATURE_LOGIN_SKIP_NUDGE = (
    "This action is rejected — this page still has an empty Password field. Do not move on to "
    "any other action (including wait) until both Username and Password are filled in. Look "
    "at the indexed elements and fill the empty field right now."
)


async def _login_form_needs_password(page: Page) -> bool:
    """หน้ามี input[type=password] ที่มองเห็นและว่างไหม (อ่าน DOM จริง snapshot แยก placeholder
    กับค่าว่างไม่ออก) = login form ยังไม่ครบ
    W19: ใส่ _DOM_CHECK_TIMEOUT_MS (default 30s)
    W_change_password_form_is_not_a_login_form (2026-09-03): หน้าเปลี่ยนรหัสผ่านมีช่องว่าง 3 ช่อง
    ทำให้ auto-login ยิงซ้ำและ login guard ปฏิเสธจน task ตาย — ข้ามฟอร์มนี้ (login จริงมีช่องเดียว)
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

# ระยะห่างต่ำสุดระหว่าง next_action() กัน rate limit (RPM) ทุก provider — heuristic ไม่ผูก quota จริง
# W41: เดิม sleep เต็มท้ายทุก step โดยไม่หักเวลาที่ผ่านไป ทำให้สถานะ "เสร็จ" ช้า — วัด wall-clock
# จาก next_action() ครั้งก่อน sleep เฉพาะส่วนที่ขาด (last_llm_call_at ใน run_task) ระยะขั้นต่ำเท่าเดิม
# Speed 2.4: ค่าอยู่ที่ config.py::step_pacing_delay_seconds

# W41: long_term_memory.record_task() ไม่มีใครรอผล (คืน None ไม่ throw) แต่เดิม await ก่อน return
# ทำให้สถานะ "เสร็จ" ช้า — ยิงเป็น background task เก็บ reference กัน GC ทิ้งก่อนเสร็จ
# (ตาม asyncio.create_task docs) ลบเองผ่าน add_done_callback
_background_tasks: set = set()


def _fire_and_forget(coro) -> None:
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


# ══════════════════════════════════════════════════════════════════════
# โซน 14: เปิด browser + แบนเนอร์คุกกี้ + CAPTCHA + auto-login
#   ทำอะไร: เตรียมหน้าเว็บให้พร้อมก่อน/ระหว่าง loop
#   ทำงานยังไง: default browser ของเครื่อง -> ปิดแบนเนอร์ (เลือกปฏิเสธเสมอ ทุก frame) -> ตรวจ CAPTCHA -> login ด้วย credential ที่บันทึกไว้ (ไม่เข้า prompt)
# ══════════════════════════════════════════════════════════════════════
# W10[B]: callback แจ้งความคืบหน้าสดระหว่าง loop (ต่างจาก history ที่มาตอนจบ) — optional แบบ
# ask_user_func ไม่ส่งมาก็ไม่ทำอะไร ไม่ผูก transport (SSE/WebSocket เป็นเรื่องชั้นบน)
OnEventFunc = Callable[[dict], Awaitable[None]]


# W11[A]: browser ที่มองเห็นใช้ default browser ของเครื่อง (Chrome/Edge มี bookmark/login ของ user)
# แทน bundled Chromium — อ่าน registry UrlAssociations\https\UserChoice ProgId แล้ว map เป็น channel
# รองรับแค่ Chromium-based (มี CDP) Safari/Firefox คืน None -> fallback Chromium ของ Playwright
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
    """launch ด้วย channel ไม่สำเร็จ (default คือ Chrome แต่ไม่ได้ติดตั้งจริง) -> fallback
    Chromium เงียบๆ ไม่ให้ task พังเพราะเรื่องเลือกเบราว์เซอร์"""
    if channel:
        try:
            return await playwright.chromium.launch(headless=headless, channel=channel)
        except Exception:
            pass
    return await playwright.chromium.launch(headless=headless)


# W_consent_banner (OrangeHRM live, ไม่คงที่): Cookiebot ทับหน้า login element 11 ตัวแรกเป็นปุ่ม
# แบนเนอร์ _maybe_auto_login หาฟอร์มไม่เจอ agent ยิง action ใส่แบนเนอร์ — CMP รายใหญ่ใช้รูปแบบ
# เดียวกัน จัดการครั้งเดียวก่อนเข้า loop
# เลือก "ปฏิเสธ" เสมอ (เคารพความเป็นส่วนตัว) ไม่เจอค่อยปุ่มปิด — ไม่มีทางไหนกด "Allow all"
_CONSENT_REJECT_TEXTS = (
    "reject all", "reject cookies", "reject", "deny all", "deny", "decline all", "decline",
    "only necessary", "necessary only", "use necessary cookies only", "essential only",
    "ปฏิเสธทั้งหมด", "ปฏิเสธ", "ไม่ยอมรับ", "เฉพาะที่จำเป็น", "ที่จำเป็นเท่านั้น",
)
_CONSENT_CLOSE_TEXTS = ("close banner", "close", "ปิด")

# W_consent_banner_midtask (live run 3): บาง CMP โผล่กลางทาง โมเดลกด "Allow all" เอง (ยินยอมแทน
# user) แล้วหลุดไปคลิกโฆษณาจนหมด step — ตรวจ label ลายเซ็นใน snapshot ทุก step (ไม่ยิง JS เพิ่ม)
_CONSENT_LABEL_SIGNATURES = (
    "allow all", "accept all", "accept cookies", "allow selection", "reject all",
    "deny all", "manage cookies", "cookie settings", "this website uses cookies",
    "we use cookies", "ยอมรับคุกกี้", "เว็บไซต์นี้ใช้คุกกี้",
)


# W_captcha_detect (P3.8): CAPTCHA (widget ใน iframe ที่อ่านไม่ได้) ทำ snapshot แทบว่าง agent
# คลิกมั่วจนหมด step แล้วรายงานเหตุผลผิด — ไม่แก้ CAPTCHA เอง แจ้ง user ผ่าน request_user_input
# ตรวจจากข้อมูลที่ perceive แล้วเท่านั้น (ชื่อ frame/label)
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
# container ของ CMP — ยืนยันว่าปุ่ม Reject อยู่ในแบนเนอร์จริง ไม่ใช่ Reject ของฟีเจอร์อื่น
# (เช่นอนุมัติใบลา กดผิด = ปฏิเสธคำขอจริง)
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
    """ปิดแบนเนอร์คุกกี้ถ้ามี -> ข้อความบนปุ่มที่กด หรือ None ห้าม throw (ส่วนเสริม)

    W_consent_banner_iframe (live run): Cookiebot render ใน iframe page.evaluate() เห็นแค่ main
    document จึง "เห็นแต่กดไม่ได้" (โมเดลกดแบนเนอร์เองที่ step 4) — ไล่ทุก frame แบบ perception"""
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
    """W17: เติม credential ที่บันทึกไว้ (site_learning/storage.py::save_credentials) ถ้าหน้าปัจจุบัน
    เป็นหน้า login จริง — ครั้งเดียวก่อนเข้า loop credential ไม่เข้า prompt LLM เลย

    import site_learning แบบ lazy โดยเจตนา: crawler.py import orchestrator อยู่แล้ว (circular)

    คืน None = ไม่มี credential/ไม่ใช่หน้า login/login สำเร็จ; คืนเหตุผล (mask password เสมอ)
    เมื่อ login ไม่ผ่านแม้ retry ให้ caller แจ้ง user ไม่ throw

    W_auto_login_outcome_is_invisible: None แยก "ข้าม" กับ "สำเร็จ" ไม่ได้ และ log ผูก verbose
    (API ส่ง False) — รายงานผ่าน dict `outcome` ("skipped"/"ok"/"failed") โดยไม่เปลี่ยนชนิดค่าคืน
    (ลองเป็น tuple แล้วเทสต์ล้ม 166 เคส: 13 จุด mock คืนค่าเดี่ยว)"""
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
        # retries=1 เผื่อหน้าช้า — login_with_verification() ตรวจ session_ok จริง (URL เปลี่ยน +
        # ไม่มีฟอร์ม login เหลือ) ไม่ใช่แค่กด submit ได้
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


# ══════════════════════════════════════════════════════════════════════
# โซน 15: คุยกับ user + แผนที่ส่งให้โมเดล
#   ทำอะไร: ยืนยันแผน, ขอค่ากลางทาง, บอกโมเดลว่าอยู่ข้อไหนของแผน
#   ทำงานยังไง: ใช้ ask_user_func ตัวเดียวกับ permission (อ่านค่าที่ user แก้จาก cmd หลัง await) _focused_plan_context() ส่ง CURRENT/NEXT/FINAL
# ══════════════════════════════════════════════════════════════════════
def _tokens_dict(usage: llm.TokenUsage) -> dict:
    return {
        "input": usage.input_tokens,
        "output": usage.output_tokens,
        "cache_read": usage.cache_read_tokens,
        "cache_creation": usage.cache_creation_tokens,
    }


async def _confirm_plan(plan_text: str, ask_user_func: Optional[AskUserFunc]) -> tuple[bool, str]:
    """โชว์แผนแล้วรอ user ยืนยัน — ใช้ ask_user_func แบบ permission layer (ชั้นบน inject วิธีถามเอง)

    คืน (approved, plan_text) — W10[F]: user แก้แผนได้ (TaskManager.resolve_approval mutate
    cmd["plan"]) ต้องอ่านจาก cmd หลัง await ไม่ใช่ตัวแปรเดิม"""
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
    """W_resume (Mid-Task Input Request): agent ขอค่ากลางทางแล้ว finish_task(false) ทิ้ง state
    ทั้งหมด — หยุดรอ human "กลางทาง" โดยไม่จบ loop ใช้ ask_user_func เดียวกับ _confirm_plan
    (routes.py forward cmd แบบ generic ไม่ต้องแก้)

    คืน (provided, answer) — answer มาจาก cmd["answer"] ที่ resolve_approval() mutate ต้องอ่าน
    หลัง await (pattern เดียวกับ _confirm_plan)"""
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
    """W_plan_step_cursor: บอกโมเดลว่าตอนนี้อยู่ข้อไหนของแผน

    เดิมส่งแผนดิบทุก step โมเดลเล็กกระโดดไปข้อท้ายๆ (live run 2026-08-27: completed_plan_step=5
    ตั้งแต่ action แรก Goal Boundary Gate หยุด task อ้างว่าสำเร็จ) — บังคับ "ลำดับ" ไม่บังคับ "วิธี"
    W_token_trim (P1/Q6): ส่งเต็มแค่ CURRENT + NEXT + FINAL (ยังเห็นปลายทาง) ข้อที่เสร็จ/ข้อระหว่าง
    ยุบเป็นบรรทัดเดียว cursor เกินจำนวนข้อ (แผนจบ) ยังคืนค่าไม่ว่าง (guard ต้องเห็นว่ามีแผน)"""
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


# ══════════════════════════════════════════════════════════════════════
# โซน 16: Goal Boundary Gate — goal สำเร็จแล้วหยุด
#   ทำอะไร: ตัดสินว่า goal ถึงเป้าหมายแล้ว (นำทางถึงหน้า / สั่งซื้อเสร็จ / แผนจบ) แล้วบล็อก action นอก scope
#   ทำงานยังไง: ดึง nav target จาก goal (ข้าม clause login/เปิดเว็บ) -> เทียบกับ URL path จริง + cursor แสดงผลของ PLAN panel
# ══════════════════════════════════════════════════════════════════════
# W_goal_scope (Goal Boundary Gate): hard stop คู่กับ W_stop_when_done — goal "go to Admin page"
# ถึงแล้วแต่ agent เดินต่อเกือบสร้าง record ที่ไม่มีใครขอ W_stop_when_done แค่ nudge และทำงาน
# เฉพาะมีแผน ตัวนี้ครอบทั้ง (1) goal นำทางไม่มีแผน (heuristic แคบ) (2) สัญญาณจากแผน แต่บล็อกจริง
# ดู enforcement หลังบล็อก finish_task
_MAX_PREMATURE_GOAL_SCOPE_RETRIES = 1
# action ไม่ mutate ที่ยังอนุญาตหลัง goal ครบ scope (กลุ่มเดียวกับที่ ACC-3 ไม่นับเป็นความคืบหน้า)
_GOAL_SCOPE_ALLOWED_ACTION_TYPES = {"read_page_data", "wait", "scroll", "hover"}

# W_plan_panel_lags_the_log (2026-09-07): PLAN panel ติ๊กครบตอนจบ task เพราะ plan_step_done ยิงเฉพาะ
# เมื่อโมเดลรายงาน completed_plan_step — ใช้ cursor ตัวที่สองจากหลักฐานบนหน้า (URL เปลี่ยน/action ตรงข้อ)
# ห้ามละเมิด: cursor นี้ "แสดงผลอย่างเดียว" — plan_cursor เดิมป้อน goal-scope hard stop เร่งให้ขยับ
# ง่ายจะเปิดบั๊ก W_plan_counter_claims_a_password_change (สำเร็จทั้งที่ยังไม่กดบันทึก) กลับมา
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
    """หลักฐานว่า action นี้ทำข้อที่ cursor แสดงผลชี้อยู่ — "match"/"url"/"action"/""
    ฟังก์ชันบริสุทธิ์ ไม่เรียก LLM ใช้ _action_matches_plan_step() (สะพานไทย<->อังกฤษ) ได้ เพราะ
    cursor แสดงผลไม่ gate อะไร"""
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
    # W_plan_ticks_every_row (2026-09-07 ติ๊กสดได้แค่ 1 ใน 4 แถว): action เปลี่ยนสถานะสำเร็จแต่ผูก
    # กับข้อไม่ได้ ให้เดินหน้าหนึ่งข้อ (แผนของ LLM กว้างเกิน match label และไม่มี URL เปลี่ยน)
    # ราคาที่ยอมรับ: ติ๊กเร็วกว่าจริงได้ ไม่กระทบความถูกต้องเพราะไม่ป้อน guard ใด และ cap ข้อสุดท้าย
    return "action"

# W_verify_text_needs_a_write (2026-09-04): goal นำทางล้วนเสีย LLM call เพราะ guard ตาราง W63[7.2]
# ตีกลับ — gpt-5.4-mini บน ChatGPT OAuth กรอกทุก property ในสคีมาเสมอ verify_text จึงมาทุกครั้ง
# (เหตุผลเดียวกับ W_fill_secret_schema_gate) เช็คสองสัญญาณ (ปลอดภัยไว้ก่อน):
#   - operation create/edit -> guard ทำงานเสมอ (งานสร้างที่เดินด้วยคลิกล้วนมีจริง)
#   - operation unknown -> เชื่อหลักฐาน: เคยเขียนค่าลงฟอร์มไหม
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
# goal ที่มีคำเหล่านี้ = หลาย objective ไม่ใช่ goal นำทางล้วน ("ไปหน้า Admin แล้วลบ user" ต้องทำครบ)
_COMPOUND_GOAL_MARKERS = (
    " and ", ",", " then ", "และ", "แล้ว", "จากนั้น", " before ", " after that",
)
# รูปประโยค "แค่ไปที่หน้า X" (ไทย/อังกฤษ) เช็คเป็น prefix ของ goal ตัวพิมพ์เล็ก
_SIMPLE_NAV_PREFIXES = (
    "navigate to ", "go to ", "goto ", "open ",
    "ไปที่หน้า", "ไปยังหน้า", "ไปหน้า", "ไปที่", "ไปยัง", "เปิดหน้า", "เข้าหน้า",
)
_SIMPLE_NAV_TRAILING_WORDS = ("page", "หน้า")


def _extract_simple_navigation_target(goal: str) -> Optional[str]:
    """W_goal_scope: target ของ goal "ไปที่หน้า X" ล้วน (ตัด prefix + "page"/"หน้า") หรือ None
    ถ้ามี objective อื่น (_COMPOUND_GOAL_MARKERS) — คู่กับ _navigation_target_reached()
    W_goal_scope_regression: ตัด "the" หลัง prefix ด้วย ("go to the Admin page" เคยได้ "the Admin")"""
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


# W_thai_nav_target_never_matches_url (2026-09-04): goal ไทย ("หน้าแอดมิน") vs URL อังกฤษ gate ไม่เคย
# ทำงาน — ตารางเล็ก เพิ่มจากหลักฐานในรันจริงเท่านั้น ห้ามเดาเติมเป็นพจนานุกรม
_NAV_TARGET_URL_ALIASES = {
    "แอดมิน": "admin",
    "ผู้ดูแลระบบ": "admin",
}


# W_order_complete_is_the_end (gate 3aaf213, login_checkout): ถึง checkout-complete แล้วเดินเล่นต่อ
# จนชน max_steps task นับว่าล้ม — goal-scope gate สามทางเดิมไม่ครอบงานสั่งซื้อ
# ต้องเข้าทั้ง goal (สั่งซื้อ) และ URL (success) — URL อย่างเดียวไม่พอ
# ไม่ใส่ "order" เดี่ยว: benchmark "sort by Price (low to high)" ใช้ order แปลว่าลำดับ
_CHECKOUT_GOAL_KEYWORDS = (
    "checkout", "check out", "purchase", "place the order", "buy",
    "ชำระเงิน", "สั่งซื้อ", "เช็คเอาท์",
)
_ORDER_COMPLETE_URL_MARKERS = (
    "checkout-complete", "checkout_complete", "order-complete", "order_complete",
    "order-confirmation", "ordercomplete", "thank-you", "thankyou",
    "purchase-complete", "payment-success", "payment_success",
)


def _order_is_complete(goal: str, page_url: str) -> bool:
    """งานสั่งซื้อ/ชำระเงินที่หน้าปัจจุบันคือหน้ายืนยันว่าสั่งซื้อเสร็จแล้ว"""
    if not contains_keyword(goal or "", _CHECKOUT_GOAL_KEYWORDS):
        return False
    url = (page_url or "").lower()
    return any(marker in url for marker in _ORDER_COMPLETE_URL_MARKERS)

def _navigation_target_reached(target: str, page_url: str, last_action_record: list[dict]) -> bool:
    """W_goal_scope: ถึงหน้าเป้าหมายจริง — ต้องมี (1) action ล่าสุดเป็น click/goto ที่สำเร็จ และ
    (2) target อยู่ใน URL path ไม่ใช่ page_text (sidebar โชว์ลิงก์ "Admin" ตลอด จะ false positive)"""
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


# W_goal_scope_compound_login: goal "login then goto adminmenu" ไม่เคยถูก gate (compound) ทั้งที่
# login ระบบทำเองผ่าน _maybe_auto_login — agent ถึง Admin ตั้งแต่ step 1 แล้วไปกด Edit/Save/Delete
# รับเป็น nav target เมื่อ clause สุดท้ายเป็น nav ล้วน *และ* ทุก clause ก่อนหน้าเป็น login ล้วน
_LOGIN_ONLY_CLAUSE_KEYWORDS = (
    "login", "log in", "sign in", "signin", "เข้าสู่ระบบ", "ล็อกอิน", "ล๊อกอิน", "ลงชื่อเข้าใช้",
)


def _is_login_only_clause(clause: str) -> bool:
    """clause เป็น login ล้วน — ตัดคำ login แล้วต้องไม่เหลือตัวอักษรใดเลย
    (กัน "login and delete the ESS user")"""
    lower = (clause or "").strip().lower()
    if not lower:
        return False
    if not any(kw in lower for kw in _LOGIN_ONLY_CLAUSE_KEYWORDS):
        return False
    for kw in _LOGIN_ONLY_CLAUSE_KEYWORDS:
        lower = lower.replace(kw, " ")
    return not re.search(r"[a-zA-Z\u0e00-\u0e7f0-9]", lower)


# W_open_site_prefix_blocks_nav_gate (2026-09-04): "เปิดเว็ปแล้วไปที่หน้าแอดมิน" ไม่ได้ nav target
# เพราะ "เปิดเว็ป" ไม่ใช่ login ทั้งที่เป็น no-op เหมือนกัน (goto url ทำก่อนเข้า loop แล้ว) — ผลคือ
# เสียเทิร์นกับ already_active_skip (2 ใน 6 รอบ ~15k token ต่อครั้ง)
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
    """W_goal_scope_compound_login: ลอง _extract_simple_navigation_target() ก่อน แล้วค่อยรับกรณี
    "login/เปิดเว็บ แล้วไปหน้า X" ที่เหลือคืน None"""
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


# ══════════════════════════════════════════════════════════════════════
# โซน 17: guard ห้ามทำเกิน goal
#   ทำอะไร: กันสร้าง record ใหม่ / เข้า flow รหัสผ่าน / กดบันทึกฟอร์มแก้ไขในงานลบ / บันทึกก่อนกรอกค่าครบ
#   ทำงานยังไง: ตัดสินจาก label ปุ่ม + URL + intent ของ goal — ส่วนใหญ่เป็น hard reject ไม่มีโควตา เพราะความเสียหายกู้คืนไม่ได้
# ══════════════════════════════════════════════════════════════════════
# W_no_create_for_existing_goal: goal "ลบ ESS ให้หมด" ไม่เจอแถว agent ไปหน้า Add User สร้าง user
# ใหม่จน "Already exists" — SYSTEM_PROMPT ห้ามอยู่แล้วแต่ไม่มีโค้ดบังคับ
# แคบโดยเจตนา: เฉพาะ goal ลบ/แก้ของที่มีอยู่ที่ไม่มีคำสั่งสร้างปน
_CREATE_INTENT_KEYWORDS = (
    "add", "create", "new user", "register", "สร้าง", "เพิ่ม", "ลงทะเบียน",
)
# label/URL ที่เข้า flow สร้างรายการใหม่ — ไม่นับ "add to ..." (Add to cart ไม่ใช่สร้าง record)
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
    """action พาเข้า flow สร้างรายการ — label *ขึ้นต้น* ด้วย Add/Create/New/เพิ่ม/สร้าง (กัน
    "Add to cart"/"Address") หรือ URL ปลายทางของ goto"""
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


# W_no_credential_flow (live run): goal ลบ ESS agent หลงไปเมนูโปรไฟล์ -> Change Password จน task
# ขอรหัสใหม่จาก user (เกิดซ้ำ 2 รอบในรันเดียว) — guard เดิมคุมแค่ forced recovery และ fill_secret
# ไม่ครอบ click ที่โมเดลเลือกเอง
# hard reject ไม่มีโควตา: เปลี่ยนรหัสบัญชีที่ login อยู่ = ล็อก user ออก กู้ด้วย go_back ไม่ได้
# ปิด guard ถ้า goal พูดถึงรหัสผ่าน/บัญชีเอง (รวม goal ที่ให้ credential มา login)
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


# W_no_record_edit_for_delete_goal (live run 227b771e): goal ลบ ESS agent กด Edit -> Save เขียนทับ
# record จริง 2 ครั้ง ("Successfully Updated") — หนักกว่า Add User (ระบบปฏิเสธเอง) Save สำเร็จจริง
# กู้ไม่ได้ W_no_create ไม่ครอบเพราะนี่คือแก้ record ที่มีอยู่
# บล็อกที่ Save ไม่ใช่ Edit: Edit ไม่เสียหาย และบางเว็บใช้เป็นทางไปปุ่มลบ
# regex ขึ้นต้น + word boundary: save\b ไม่ match "Saved Searches", update\b ไม่ match "Updated on"
# ไม่ใส่โดยเจตนา: submit/confirm/ยืนยัน (label ของ dialog ยืนยันการลบ), change password
# (W_no_credential_flow ครอบแล้ว), แก้ไข (= Edit) จึงไม่ใช้ _FORM_SUBMIT_LABEL_KEYWORDS ซ้ำ
# W_plan_commits_a_record_edit: คำชุดเดียว สอง anchoring — มี ^ สำหรับ label ปุ่ม, ไม่มี ^ สำหรับ
# ข้อความแผน (คำอยู่กลางประโยค)
_RECORD_COMMIT_WORDS = ("save", "update", "บันทึก", "อัปเดต")
_RECORD_COMMIT_ALTERNATION = "|".join(_RECORD_COMMIT_WORDS)

_RECORD_COMMIT_LABEL_RE = re.compile(
    rf"^\s*(?:{_RECORD_COMMIT_ALTERNATION})\b", re.IGNORECASE,
)
_PLAN_COMMIT_STEP_RE = re.compile(
    rf"(?:{_RECORD_COMMIT_ALTERNATION})", re.IGNORECASE,
)


# W_goal_values_before_save (gate 2026-09-07, add_candidate): กรอก 2 ใน 3 ช่องแล้วกด Save ซ้ำจน
# หมด step — ค่าในเครื่องหมายคำพูดของ goal คือสิ่งที่ user พิมพ์ตรงๆ เช็คได้ deterministic ว่ายัง
# ไม่ถูกกรอก ไม่เดาค่าที่ไม่มีคำพูด
_GOAL_QUOTED_VALUE_RE = re.compile(r"""['"“‘]([^'"”’
]{2,80})['"”’]""")
# ค่าที่ยาวเกินไปหรือมีช่องว่างเยอะมักเป็นประโยคที่ user ยกมาอ้าง ไม่ใช่ค่าที่ต้องกรอก
_MAX_GOAL_VALUE_WORDS = 4


def _goal_literal_values(goal: str) -> list[str]:
    """ค่าที่ goal ระบุไว้ตรงๆ ว่าต้องกรอก — จาก 'ค่าในเครื่องหมายคำพูด' และฝั่งขวาของ
    field=value (ใช้ _goal_condition_pairs() ตัวเดิม ไม่เขียน regex ที่สาม)"""
    values: list[str] = []
    for raw in _GOAL_QUOTED_VALUE_RE.findall(goal or ""):
        value = raw.strip()
        if value and len(value.split()) <= _MAX_GOAL_VALUE_WORDS:
            values.append(value)
    for _field, value in _goal_condition_pairs(goal or ""):
        if value:
            values.append(value)
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        key = value.casefold()
        if key not in seen:
            seen.add(key)
            unique.append(value)
    return unique


_FORM_VALUES_JS = """() => ({
    values: Array.from(document.querySelectorAll("input, textarea, select"))
        .map((el) => (el.value == null ? "" : String(el.value))).filter(Boolean),
    hasPassword: !!document.querySelector('input[type="password"]'),
})"""

# ค่าที่ goal ให้ไว้ "ล็อกอิน" — auto-login (W17) กรอกนอกลูป ไม่ผ่าน action และไม่อยู่ใน DOM หน้าถัดไป
# guard จะรายงานว่ายังไม่กรอกตลอดกาล (probe 2026-09-08: ขาด ['Admin', 'admin123'] ค้าง)
_GOAL_CREDENTIAL_VALUE_RE = re.compile(
    r"(?:username|user\s*name|password|pass|ชื่อผู้ใช้|รหัสผ่าน)\s*"
    r"(?:is|=|:|คือ)?\s*['\"“‘]([^'\"”’\n]{1,80})['\"”’]",
    re.IGNORECASE,
)


async def _values_missing_before_commit(
    page: Page, goal: str, written: set,
) -> list[str]:
    """ค่าจาก goal ที่ไม่เคยถูกกรอกใน task นี้ และไม่อยู่ในฟอร์มตอนนี้ — เช็คสองแหล่งเพราะพลาด
    คนละทาง (ค่าหน้าก่อนไม่อยู่ใน DOM, ค่าที่ระบบกรอกเองไม่อยู่ใน written) ห้าม throw"""
    wanted = _goal_literal_values(goal)
    if not wanted:
        return []
    try:
        state = await page.evaluate(_FORM_VALUES_JS)
    except Exception:
        return []
    if not isinstance(state, dict):
        return []
    present = {
        str(v).strip().casefold() for v in (state.get("values") or []) if str(v).strip()
    }
    present |= {str(v).strip().casefold() for v in written if str(v).strip()}
    if not state.get("hasPassword"):
        # ไม่มีช่องรหัสผ่านบนหน้านี้ = ไม่ใช่หน้า login แล้ว ค่า credential จาก goal จึงไม่มี
        # ที่ให้กรอกและไม่ควรถูกนับว่าขาด (ดูคอมเมนต์เหนือ _GOAL_CREDENTIAL_VALUE_RE)
        present |= {
            v.strip().casefold() for v in _GOAL_CREDENTIAL_VALUE_RE.findall(goal or "")
        }
    return [value for value in wanted if value.casefold() not in present]


_MAX_MISSING_VALUE_REJECTIONS = 2
_MISSING_GOAL_VALUES_NUDGE = (
    "[Rejected] '{what}' would submit this form, but {count} value(s) the goal explicitly "
    "asked for have not been entered anywhere yet: {values}. Find the field for each one on "
    "this page and fill it in first, then submit. Never submit a form with the goal's own "
    "values still missing — the site rejects it (a 'Required' message appears next to the "
    "empty field) and pressing the button again cannot fix it."
)

def _goal_is_deletion_only(goal: str) -> bool:
    """goal ลบล้วน ไม่มีแก้ไข/สร้างปน — แคบกว่า _goal_targets_existing_records_only() ต้องไม่แตะ
    goal แก้ไขจริง ("เปลี่ยน Role ของ ESS เป็น Admin")"""
    if _goal_wants_to_create(goal) or _is_edit_all_intent_goal(goal):
        return False
    return _is_deletion_intent_goal(goal)


def _action_commits_a_record_edit(tool_input: dict, label: str) -> bool:
    """action นี้คือกดบันทึกฟอร์มแก้ไข — ดู label อย่างเดียว ไม่ดู URL (ไม่งั้นยิงซ้ำกับ W_no_create
    เพราะ _CREATE_URL_MARKERS มี savesystemuser แล้ว nudge สองข้อความขัดกัน)"""
    return bool(_RECORD_COMMIT_LABEL_RE.search(label or ""))


_NO_RECORD_EDIT_NUDGE = (
    "[Rejected] '{what}' commits an edit to an existing record, but this goal only asks you to "
    "DELETE things — it never asks you to change or update anything. Saving here would "
    "permanently overwrite real data, and going back cannot undo it. Being on an edit form at "
    "all means you took a wrong turn: leave it (Cancel, or go_back), return to the list, and "
    "use the row's own Delete action instead. If the filtered list genuinely has no matching "
    "rows left, that IS the answer — call finish_task and report it plainly."
)


# ══════════════════════════════════════════════════════════════════════
# โซน 18: รหัสผ่าน + fill_secret
#   ทำอะไร: คุมว่า fill_secret ใช้ได้เฉพาะฟอร์มเปลี่ยนรหัสผ่านจริง และชี้ทางเมื่อโมเดลใช้ผิด
#   ทำงานยังไง: ตรวจฟอร์มจาก DOM (ช่อง password >= 2 + มี Current Password) -> gate สคีมา/guard -> recovery คลิก nav ที่ตรง goal (ไม่เลือก profile menu / [already active])
# ══════════════════════════════════════════════════════════════════════
# --- W_fill_secret_hardening: fill_secret context guard ------------------------------------
# Live-reproduced on OrangeHRM (OpenAI, "login then goto adminmenu"): auto-login had already
# succeeded, yet the model kept proposing fill_secret — 8-10 steps and 220k-350k tokens per run,
# once typing the real saved password into three unrelated fields. fill_secret outside a genuine
# change-password context is never legitimate, so reject it BEFORE dispatch (SYSTEM_PROMPT W65[3]
# alone wasn't enough). Provider-agnostic on purpose.
_PASSWORD_CHANGE_INTENT_KEYWORDS = (
    "change password", "reset password", "update password", "new password",
    "change my password", "security settings",
    "เปลี่ยนรหัสผ่าน", "เปลี่ยนรหัส", "ตั้งรหัสผ่านใหม่", "รีเซ็ตรหัสผ่าน", "รหัสผ่านปัจจุบัน",
)

_CURRENT_PASSWORD_LABEL_HINTS = state_filter.CURRENT_PASSWORD_LABEL_HINTS


def _goal_or_plan_requests_password_change(text: str) -> bool:
    # W_thai_keyword_space: เทียบผ่าน contains_keyword() ที่ทนการเว้นวรรคของภาษาไทย —
    # ดูเหตุผลเต็ม (พร้อมบั๊กจริงที่มันแก้) ใน goal_intent.contains_keyword()
    return contains_keyword(text, _PASSWORD_CHANGE_INTENT_KEYWORDS)


# W_password_field_has_no_label_attributes (2026-09-03): ช่องรหัสผ่านของ OrangeHRM updatePassword ไม่มี
# label/aria/placeholder/name/id เลย (<label> เป็นพี่น้อง ไม่ผูก for=) gate จึงตัด fill_secret ออกจาก
# สคีมาบนหน้าจริง โมเดลยิง fill(21,"") จนโดน loop detector — เดินขึ้น ancestor หา <label> (แบบ
# perception) จำกัด 4 ชั้น หน้า Add User ยังได้แค่ 'password'/'confirm password' -> False ตามเดิม
_PASSWORD_FIELD_LABEL_JS = state_filter.PASSWORD_FIELD_LABEL_JS


async def _page_looks_like_change_password_form(page: Page) -> bool:
    """หน้ามี password ที่มองเห็น >= 2 ช่อง *และ* อย่างน้อย 1 ช่องสื่อว่าเป็น "Current Password"
    W_add_user_form_false_positive: ฟอร์ม Add User (Password + Confirm) ก็มี 2 ช่อง นับจำนวนอย่างเดียวไม่พอ
    fail-safe คืน False (เดาว่าใช่ = ปล่อย fill_secret พิมพ์รหัสจริงลงช่องที่ไม่รู้จัก)"""
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
    """หน้าฟอร์มเปลี่ยนรหัสผ่าน *และ* ช่อง Current Password ยังว่าง

    W_secret_stays_in_schema_forever (2026-09-03): gpt-5.4-mini กรอกทุก property ในสคีมา `secret`
    enum ค่าเดียวจึงถูกส่งทุกเทิร์นและลาก type เป็น fill_secret แม้ถูกปฏิเสธ — nudge ไม่พอ ต้อง
    ตัดออกจากสคีมาทันทีที่ช่อง Current Password ถูกกรอก (W_fill_secret_schema_gate)"""
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
    """ยังอยู่บนฟอร์มเปลี่ยนรหัสผ่านที่มีช่องว่าง = งานยังไม่จบ

    W_plan_counter_claims_a_password_change (REST API 2026-09-04): task จบ success=true ที่ step 4
    ทั้งที่ยังไม่กรอก Confirm และไม่กดบันทึก (ground truth: รหัสไม่ถูกเปลี่ยน) เพราะ goal-scope เชื่อ
    completed_plan_step ที่โมเดลรายงานเอง — คลาสเดียวกับ W_plan_cursor_not_proof: การวัดชนะตัวนับเสมอ
    ห้าม raise อ่านไม่ได้คืน False"""
    try:
        if not await _page_looks_like_change_password_form(page):
            return False
        return any(not st.get("filled") for st in await _password_field_states(page))
    except Exception:
        return False


# W_secret_refilled_forever (2026-09-03 ซ้ำ 2 รอบ): fill_secret สำเร็จแล้วโมเดลสั่งซ้ำที่ index เดิม
# ไม่ขยับไปช่อง Password/Confirm จนโดน same-label-loop — แบบ W_state_guard_shortcut: ชี้ index จริง
# ของช่องถัดไป อ่านความว่างจาก DOM (label ปิดค่าไว้แล้ว) ไม่ส่งค่าจริงออกไป
_MAX_SECRET_REFILL_RETRIES = 2

# W_click_submits_with_empty_password_fields: มีโควตา ไม่บล็อกตาย — บางฟอร์มเว้นช่องรหัสว่างได้จริง
# (หน้าแก้โปรไฟล์ที่รวมเปลี่ยนรหัส)
_MAX_EMPTY_PASSWORD_SUBMIT_RETRIES = 2

_PASSWORD_FIELD_STATE_JS = """() => Array.from(
    document.querySelectorAll('input[type="password"]')
).filter(
    el => el.getClientRects().length > 0
).map(el => ({
    index: el.getAttribute('data-ai-index'),
    filled: !!(el.value || '').trim(),
}))"""


# W_index_drift_measure (2026-09-07): fill(24) ไปโดน '-- Select --' แทน Email แยกไม่ออกว่าสาเหตุคือ
#   (1) หน้า re-render ระหว่าง snapshot กับ dispatch หรือ (2) โมเดลอ้าง index จากเทิร์นเก่า
# เทียบ label ตอน snapshot กับ label สดตอน dispatch: ต่าง = (1), เหมือนแต่ผิดเป้า = (2) — วัดก่อนแก้
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


# W_state_guard_shortcut: point at the actual answer, not just "try something else" (3 live runs,
# OrangeHRM/OpenAI: model kept choosing fill_secret while "[3] a 'Admin' (navigation)" was on screen).
# Deterministic hint patterns (never change what gets dispatched):
#   (1) wrong element (e.g. the profile-menu span) -> name the nav link whose label matches the goal
#   (2) right element, wrong verb (fill_secret on a plain <a>) -> say "click that same index"
#   (3) goal's nav target already "[already active]" -> the agent has arrived (checked first)
# _fill_secret_context_hint() tries (3), then (2), then (1).
_GOAL_KEYWORD_STOPWORDS = frozenset({
    "go", "goto", "to", "the", "a", "an", "page", "pages", "navigate", "open", "click",
    "on", "into", "and", "then", "please", "menu", "section", "screen", "view",
})


def _find_goal_matching_nav_element(
    elements: list[dict], goal: str, skip_already_active: bool = False,
) -> Optional[dict]:
    """nav element แรกที่ label มีคำจาก goal (>= 3 ตัว ตัด stopword) หรือ None ไม่ throw

    skip_already_active (W_fill_secret_recovery_excludes_already_active): ข้ามตัวที่ "[already active]"
    แล้วหาต่อ (ดู _fill_secret_recovery_target)
    W_goal_token_substring: "adminmenu" (goal) ต้อง match "Admin" (label) — เทียบสองทาง ฝั่ง label
    ต้อง >= 4 ตัว (กัน "PIM"/"Buzz")
    W_goal_token_wordboundary: "ess" เคย match กลาง "businESS Solutions" — เทียบที่ขอบเขตคำทั้งสองทิศ"""
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
        # ทิศ 2 (W_goal_token_substring): คำใน label (>= 4 ตัว) อยู่ *ต้น* คำใน goal ("adminmenu")
        label_words = [w for w in re.findall(r"[a-z]{4,}", label_lower) if w not in _GOAL_KEYWORD_STOPWORDS]
        if any(word.startswith(lw) for lw in label_words for word in words):
            return element
    return None


def _fill_secret_recovery_target(elements: list[dict], tool_input: dict, goal: str) -> Optional[dict]:
    """Element to force-click instead of a rejected fill_secret (used blindly by
    _force_loop_recovery(), unlike the hint text), or None.

    Order and exclusions, each from a live-reproduced OrangeHRM bug:
    - W_fill_secret_recovery_target_priority: goal-keyword nav match FIRST. Trusting the model's
      own index first force-clicked the account dropdown twice before reaching "Admin". The
      model's index is only a fallback (pattern (2): right element, wrong verb).
    - W_fill_secret_recovery_excludes_profile_menu: never the "[Profile/Account Menu]" element.
      Goal matching is ASCII-only so a Thai goal falls through to the model's index, which kept
      pointing at the profile menu — and that element exists for password flows, the opposite of
      what this path (known NOT a password flow) wants. None lets the generic go_back/scroll take over.
    - W_fill_secret_recovery_excludes_already_active: never an "[already active]" element (goal
      "เปิดเว็บ แล้วไปที่หน้าadmin": agent had arrived, recovery clicked "Admin [already active]"
      three times — a guaranteed no-op — and the task hard-failed on completed work). The main
      loop's already-active check is bypassed by forced recovery, so it's repeated here, on both
      the goal-match and the model's-index paths."""
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
    """ข้อความเสริมท้าย _FILL_SECRET_NOT_PASSWORD_CONTEXT_NUDGE ที่ชี้เป้าให้ตรงที่สุด หรือ "" ไม่ throw

    pattern (3) เช็คก่อน: nav target ของ goal เป็น "[already active]" = ถึงแล้ว บอกให้ finish_task
    (ชี้ให้คลิกซ้ำแบบ (1) คือเลี้ยง loop) — ยังเป็นแค่ข้อความ ไม่บังคับ finish_task แทนโมเดล
    (W_goal_precheck ใน prompt มีอยู่แล้วแต่ไม่พอ)"""
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
    # W_login_form_wrong_verb (pattern (4), saucedemo): goal ให้รหัสมาตรงๆ โมเดลเล็งช่อง password
    # ถูกแต่ใช้ fill_secret — pattern (2) ข้าม <input> ทั้งหมดจึงไม่มี hint task ตายใน 2 step
    # บอกให้ใช้ "fill" กับ index เดิม
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

# W_goal_precheck: guard already-active มีโควตา — โมเดลอ่อนเสนอ click เดิมไม่หยุด บล็อกไม่สิ้นสุด
# เผา step โดยไม่มี history เกินโควตาปล่อย dispatch (คลิก element ที่ active อยู่เป็น no-op)
_MAX_ALREADY_ACTIVE_SKIP_RETRIES = 2


# ══════════════════════════════════════════════════════════════════════
# โซน 19: class Orchestrator — generate_plan / run_task / run_fastpath
#   ทำอะไร: จุดเริ่มงานของ agent
#   ทำงานยังไง: _llm_backend เลือก provider -> generate_plan ร่างแผน -> run_task วนลูปหลัก (ดูขั้น [1]-[11] ข้างใน) -> run_fastpath replay template
# ══════════════════════════════════════════════════════════════════════
class Orchestrator:
    def __init__(self):
        self.memory = ShortTermMemory()

    @staticmethod
    def _llm_backend(provider: str):
        """(client, model, next_action, append_tool_result, compact_messages) ของ provider
        (anthropic/gemini/groq/openai) รูปเดียวกัน loop ไม่ต้องรู้ว่าเป็นตัวไหน
        W22: compact_messages dispatch ที่นี่ (เดิม hardcode Gemini ใน loop)"""
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
        # W_openai_oauth (risk disclosure ใน core/openai_oauth.py): สร้างแค่ client shell ไม่มี
        # network call — _llm_backend() เป็น sync แต่ขอ/refresh token ต้อง await จึงผลักไปทำใน
        # llm.next_action_openai() (async อยู่แล้ว) แทนการทำให้ตัวนี้ async
        if provider == "openai":
            return (
                llm.build_openai_client(),
                settings.openai_model,
                llm.next_action_openai,
                llm.append_tool_result_openai,
                # user turn ของ OpenAI shape เดียวกับ Anthropic -> reuse ได้ (function_call item
                # ใช้ key "type" ไม่ใช่ "role" จึงไม่ถูกเข้าใจเป็น user turn)
                _compact_anthropic_messages,
            )
        raise ValueError(f"ไม่รู้จัก LLM provider: {provider!r} (รองรับแค่ anthropic/gemini/groq/openai)")

    async def generate_plan(
        self, url: str, goal: str, provider: Optional[str] = None, page: Optional[Page] = None,
        site_manual_context: str = "", previous_user_goal: str = "", previous_assistant_message: str = "",
    ) -> tuple[str, bool]:
        """W13: ร่างแผน (llm.generate_plan) เป็นเฟสแยก ไม่เปิด/connect browser เอง — มี page
        (session เดิม) ก็ perceive ให้แผน grounded ไม่มีก็ร่างจาก goal ล้วน

        site_manual_context (W14): คู่มือเว็บที่ crawl มา (routes.py โหลดให้) แปะก่อน page_text
        previous_user_goal/previous_assistant_message (W20 Context-Aware Implicit Execution):
        เทิร์นก่อนหน้าให้ LLM แก้คำอ้างอิงกำกวม ("เปิดให้หน่อย") แม้เทิร์นนั้นเป็น general-chat
        (route_multi_turn_strategy ทำงานเฉพาะ session ที่มี page + extracted_memory)

        คืน (plan_text, is_qa) — qa_summary คืน (" ", True) ให้ frontend ข้ามหน้าอนุมัติแผน"""
        resolved_provider = provider or settings.llm_provider
        client, model, _, _, _ = self._llm_backend(resolved_provider)
        page_text = ""
        # W19 (Navigation Deduplication): ใช้ page.url จริง ไม่ใช่ url param (session อาจอยู่หน้าอื่นแล้ว)
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
        # W_plan_keeps_goal_verb: ร่างใหม่ *ครั้งเดียว* พร้อมบอกว่าผิดตรงไหน — ยังผิดก็ปล่อยให้ user
        # เห็นบนหน้ายืนยัน run_task() จะหยุดถามเองก่อนลงมือ
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
        """Perceive -> Plan -> Act loop จนกว่า LLM เรียก finish_task หรือครบ max_steps

        session_id (W23): scope long-term memory ต่อ session; None = recall() คืน [] เสมอ
        headless: None = settings.browser_headless ไม่มีผลถ้าส่ง browser มาเอง
        verbose: print ทุก step ลง terminal (API ปิด)
        provider: None = settings.llm_provider
        ask_user_func: callback (cmd) -> bool ใช้ร่วม permission layer และ confirm_plan;
            ไม่ส่งมา = input() ทาง terminal
        confirm_plan: ร่างแผนแล้วรอ user ยืนยันก่อน ไม่ยืนยัน = steps=0
        on_event: W10[B] สตรีมความคืบหน้าสด (goto + ทุก step)
        keep_browser_open: W10[C] ไม่ปิด browser ตอนจบ มีผลเฉพาะ owns_browser (context จาก pool
            ต้องคืนเสมอไม่งั้นรั่ว) ใช้คู่ headless=False (routes.py รับผิดชอบไม่ตั้งผิดคู่)
        browser: W10[A] ยืมจาก BrowserPool -> เปิด/ปิดแค่ context ใหม่ (session แยก) None = เปิด/ปิดเอง
        connect_to_user_browser: ต่อ Chrome จริงของ user ผ่าน CDP (core/user_browser.py) ห้ามใช้
            พร้อม browser ใช้ context เดิมของ user ไม่เคย new_context()/close() ปิดแค่ tab ที่เปิดเอง
        user_browser_cdp_url: None = settings.user_browser_cdp_url
        allowed_domains: จำกัด navigation ทุก step — None จะ derive เป็น {extract_domain(url)}
            (W_domain_guard_default)
        tab_reuse_policy: "ask"/"always_new_tab"/"always_reuse" (CDP เท่านั้น)
        page: session-managed (SessionRegistry) ห้ามใช้พร้อม browser/connect_to_user_browser —
            ไม่ acquire/ปิดอะไรเอง registry คุม lifecycle ข้ามหลาย call
        approved_plan: W13 แผนที่อนุมัติแล้ว ผนวกเข้า effective_goal ทันที ห้ามใช้พร้อม confirm_plan
        site_manual_context: W14 คู่มือที่ crawl มา (routes.py โหลดให้) คงที่ทั้ง task แยกจาก
            manual_context (RAG ที่ user อัปโหลด)

        W12: ไม่ goto(url) เสมอ — page.url ว่าง/about:blank ถึง goto ไม่งั้น perceive หน้าปัจจุบันต่อเลย
        W5: action ที่ fail retry เงียบก่อน (actions._dispatch_with_retry) ยัง fail ค่อยส่งให้ LLM เห็น
        W5 verify: finish_task(true) ตอน steps_taken=0 ต้องยืนยันซ้ำ ผลลัพธ์มี "final_page_state"
            (snapshot สุดท้าย) ไว้เทียบกับ message ที่ LLM อ้าง
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

        # ────────────────────────────────────────────────────────────
        # [run_task 1] ตรวจ param + เลือก provider + helper ภายใน (_emit/_force_loop_recovery)
        # ────────────────────────────────────────────────────────────
        is_headless = settings.browser_headless if headless is None else headless
        resolved_provider = provider or settings.llm_provider
        client, model, next_action, append_tool_result, compact_messages = self._llm_backend(resolved_provider)

        async def _emit(event: dict) -> None:
            if on_event is not None:
                await on_event(event)

        async def _emit_screenshot(step: int) -> None:
            """W_live: screenshot ส่ง SSE event "screenshot" ให้ Test Console — best-effort ไม่ throw
            jpeg quality 55 ให้เล็กพอส่งทุก step ข้ามเมื่อ is_headless=False (มีหน้าต่างจริงอยู่แล้ว)"""
            if on_event is None or not is_headless:
                return
            # W_screenshot_never_throws: b64encode/_emit เคยอยู่นอก try ค่าผิดชนิดหรือ subscriber
            # พังฆ่า task ได้ (เทสต์ 4 เคส mock คืน AsyncMock) — ครอบทั้งหมด
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
            """W31: loop guard trigger -> บังคับ recovery action (go_back แล้ว scroll) โดยไม่ผ่าน LLM
            True = บังคับแล้ว caller continue; False = เกิน _MAX_FORCED_LOOP_RECOVERIES caller จบ task

            ผลป้อนกลับผ่าน tool_use_id เดิม (ทุก tool_use ต้องมี tool_result ไม่งั้น Anthropic/Groq
            error) พร้อมบอกตรงๆ ว่าเกิดอะไร
            forced_cmd (W_state_guard_shortcut Targeted Recovery): command ที่ heuristic หาเจอ
            (_fill_secret_recovery_target) ใช้แทน go_back/scroll ทั่วไป
            forced_target: element ของ forced_cmd — ส่ง label/tag/type ให้ execute()/classify_action()
            เหมือน dispatch ปกติ กันหลุด risk-keyword heuristic"""
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
            # W29: ผ่าน _cmd_for_repeat_comparison() เพื่อความสอดคล้อง (no-op ในทางปฏิบัติ)
            normalized_forced_cmd = _cmd_for_repeat_comparison(forced_cmd)
            last_action_cmd = normalized_forced_cmd
            consecutive_repeat_count = 1
            recent_actions.append(normalized_forced_cmd)
            if len(recent_actions) > _MAX_CYCLE_WINDOW:
                recent_actions.pop(0)
            return True

        # ────────────────────────────────────────────────────────────
        # [run_task 2] resolve browser/page (owns/pool/CDP/session) + domain guard
        # ────────────────────────────────────────────────────────────
        # W10[A] browser 4 โหมด:
        #   owns_browser — เปิด/ปิด playwright + browser เอง (W1-W9)
        #   pool (browser ส่งมา) — เปิด/ปิดแค่ context browser เป็นของ pool
        #   connect_to_user_browser — browser จริงของ user ไม่มีวันถูกปิดจากฝั่งนี้
        #   managed_externally (page ส่งมา) — ไม่ acquire/ปิดอะไรเลย ผู้เรียกคุม lifecycle
        managed_externally = page is not None
        owns_browser = browser is None and not connect_to_user_browser and not managed_externally
        playwright = None
        context = None
        opened_new_tab = False
        # W_domain_guard_default (live 3 รอบ): default-deny โดเมนอื่นแม้ผู้เรียกลืมระบุ — เดิมตั้งค่านี้
        # เฉพาะ CDP เส้นทาง pool (POST /tasks) จึงไม่มี guard agent หลุดไป orangehrm.com คลิก
        # "Contact Sales" จนหมด step ย้ายมาครอบทุกเส้นทาง งานข้ามโดเมนจริง (SSO) ส่ง allowed_domains เอง
        effective_allowed_domains = allowed_domains
        if effective_allowed_domains is None:
            # url ผิดรูป/ว่าง -> "" ห้ามตั้ง {""} (บล็อกทุกโดเมนรวมตัวเอง) ปล่อย None
            # (เกิดกับ task ที่ส่ง page มาเองไม่ระบุ url)
            _self_domain = extract_domain(url)
            if _self_domain:
                effective_allowed_domains = {_self_domain}
        browser_channel = _detect_default_browser_channel() if (owns_browser and not is_headless) else None
        # W11[A]: headed + confirm_plan -> เปิดแบบซ่อนก่อนเพื่อร่างแผน แล้วค่อยเปิดหน้าต่างจริงหลัง
        # user กด Confirm (ดู relaunch หลัง _confirm_plan) ไม่ให้หน้าต่างเด้งก่อน user ตกลง
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

        # ────────────────────────────────────────────────────────────
        # [run_task 3] state ของ task: ตัวนับ guard, แผน, สถิติ token/เวลา
        # ────────────────────────────────────────────────────────────
        messages: list[dict] = []
        success = False
        final_message = _MAX_STEPS_EXHAUSTED_MESSAGE
        # W_step_budget: งบจริงคือจำนวนรอบ แต่ steps_taken เพิ่มเฉพาะ dispatch จริง (guard ~14 จุด
        # continue โดยไม่เพิ่ม) — guard premature finish_task(false) เทียบ steps_taken กับ max_steps
        # ใกล้จบจึงปฏิเสธ finish(false) ที่ถูกแล้วทิ้งคำอธิบายจริงของโมเดล ต้องแยกตัวนับ
        iterations_used = 0
        # ข้อความล่าสุดจาก finish_task(false) ที่ guard ปฏิเสธไป — ใช้แทนข้อความ default ถ้า
        # สุดท้ายลูปจบเพราะหมดรอบจริงๆ (โมเดลอธิบายไว้แล้วว่าติดอะไร ไม่มีเหตุผลให้ทิ้ง)
        last_rejected_finish_message = ""
        steps_taken = 0
        total_usage = llm.TokenUsage()
        # W_llm_call_count: จำนวนเทิร์นที่ยิง LLM จริง (ไม่เท่า step — เทิร์นที่ถูกปฏิเสธ/จบงานไม่มีใน
        # step_trace อาจรวม retry ใน next_action ด้วย)
        llm_turns = 0
        # W_token_cut W1: เทิร์น LLM ใช้ไปกับอะไร (token_usage.jsonl) — action_calls = dispatch จริง,
        # finish_task_calls = ทุกครั้งที่เรียก, guard_rejections = {guard: ครั้งที่ตีกลับ}
        action_calls = 0
        finish_task_calls = 0
        guard_rejections: dict[str, int] = {}
        cache_hit_turns = 0
        cache_miss_turns = 0
        # W_prompt_audit: 1 entry ต่อ LLM call — char count แยกหมวด + token จริง (ดู llm._char_payload_audit)
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

        # W_token_cut W3: finish_task ที่ถูกตีกลับด้วยเหตุผลเดิมเป็นครั้งที่สอง ไม่เตือนซ้ำ (nudge รอบสอง
        # ไม่เปลี่ยนผล) ไปทาง "ยอมรับพร้อม tag ความจริง" เลย — เฉพาะ guard ที่ไม่ใช่ safety ของข้อมูล
        finish_reject_reasons_seen: set[str] = set()
        finish_loop_prevented = 0

        def _first_guard_hit(_reason: str) -> bool:
            """W_token_cut W3: เรียกหลัง _bump_guard(_reason) — True = ครั้งแรกของเหตุผลนี้
            (แนบ nudge เสริมได้), False = ซ้ำ (tool_result อย่างเดียว)"""
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
                # W_prompt_audit calibration (2026-09-02): input_tokens already includes the cached
                # part (4.23-4.36 chars/token); adding cache_read double-counted. cache_read reported
                # separately for the discount view.
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
        # W_count_answer_check: {ค่าเงื่อนไข: จำนวนที่นับได้} จาก read_page_data — ค่าใหม่ทับเก่าเสมอ
        # (ตารางเปลี่ยนระหว่าง task ได้)
        system_counted: dict[str, int] = {}
        premature_delete_all_unverified_count = 0
        # W_delete_all_intent: ค่า key=value ใน goal ("userrole=ess" -> ["ess"]) ว่าง = guard ชุดนี้ไม่ทำงาน
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
        # W_delete_all_intent: sticky ทั้ง task (ต่างจาก filter_dirty_since_search ที่ 1 step) — เปลี่ยน
        # filter แล้วยังไม่กด Search ใช้ปฏิเสธ finish_task(true) ของงานลบทั้งหมด
        filter_changed_without_search = False
        # W_resume: จำนวนครั้งที่เรียก request_user_input ไปแล้วใน task นี้ (ดู
        # _MAX_REQUEST_USER_INPUT_CALLS ด้านบนสุดของไฟล์)
        request_user_input_count = 0
        # W64[7.1]: True เฉพาะ 1 step ถัดไปหลัง fill/select ที่สำเร็จ (ดู _ROW_ACTION_LABEL_RE)
        filter_dirty_since_search = False
        # W64[7.2]: True ตั้งแต่มี action ที่ toast_confirmed — table-verify guard ผ่อนปรนตามนี้
        any_toast_confirmed_this_task = False
        # Task4 (W19): "EXECUTION_FAILED_NEEDS_REPAIR" เมื่อยอมรับ finish_task(true) หลัง retry ครบ
        # แต่ยังเจอ validation error (escape valve แต่ tag ให้ผู้เรียกรู้ว่าน่าสงสัย)
        completion_verification = "OK"
        final_page_text = ""
        # W9[A]: คำอธิบายจาก describe_screenshot() ของ step ก่อน — ใช้ครั้งเดียวแล้วเคลียร์ (diagnostic)
        pending_vision_context = ""
        # W_token_trim (P3/M3): site manual is constant per task — full once (first step and first
        # step after each compaction), then a stable id + summary. Both "" when there is no manual.
        site_manual_full, site_manual_ref = llm.site_manual_blocks(
            site_manual_context, extract_domain(url),
        )
        site_manual_full_sent = False
        force_full_site_manual = False
        plan_text: Optional[str] = None
        # W10[F]: goal ที่ next_action() เห็น = goal + แผนที่ยืนยัน (ถ้ามี) แยกจาก goal ดิบที่ใช้กับ
        # RAG/long-term query/log (แผนยาวทำ query เพี้ยน)
        effective_goal = goal
        # W41: เวลาที่ next_action() ครั้งก่อนจบ (None = ยังไม่เคยเรียก) ใช้คำนวณ pacing delay ที่เหลือ
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
        # W_retry_value_has_no_home: label ช่องที่ agent กรอกเองตามลำดับ — บอก user ว่าค่าใหม่จะไปช่องไหน
        # (label ไม่ใช่ index เพราะ index เปลี่ยนทุก snapshot)
        agent_filled_field_labels: list[str] = []
        # ว่าง = ไม่ได้กำลังรอค่าใหม่จาก user
        retry_value_field_labels: list[str] = []
        # W_auto_login_outcome_is_invisible: ค่าเริ่มต้นสำหรับ path ที่ไม่เคยเรียก auto-login
        auto_login_outcome = "skipped"
        # W_verify_text_needs_a_write: งานนี้เคยเขียนค่าลงฟอร์มจริงไหม
        wrote_a_value_this_task = False
        # W_goal_values_before_save: ค่าที่ agent เขียนลงหน้าไปแล้วใน task นี้ (ค่าที่กรอกใน
        # หน้าก่อนหน้าไม่เหลืออยู่ใน DOM ของหน้าปัจจุบัน จึงต้องจำไว้เอง)
        values_written_this_task: set = set()
        missing_value_rejection_count = 0
        goal_wants_a_record_change = (
            canonical_intent(goal).operation in _TABLE_VERIFY_RELEVANT_OPERATIONS
        )
        empty_password_submit_count = 0
        # W_goal_precheck: จำนวนครั้งติดกันที่ข้ามการคลิก element ที่มี marker "[already active]"
        consecutive_already_active_skip_count = 0
        # W_goal_scope: sticky ทั้ง task — ถ้าคำนวณ nav ใหม่ทุก step action read-only ถัดไปจะทำให้
        # memory.recent(1) ไม่ใช่ click/goto แล้ว _navigation_target_reached() คืน False
        plan_fully_completed = False
        # W_plan_step_cursor: ตำแหน่งในแผนเป็นของโค้ด — completed_plan_step เป็น self-report โมเดล
        # รายงานข้อสุดท้ายเป็นค่าแรกได้
        plan_cursor = 1
        # W_plan_panel_lags_the_log: cursor แสดงผลอย่างเดียว (จากหลักฐาน URL/action) — ไม่ป้อน
        # plan_fully_completed/prompt และไม่เกินจำนวนข้อ ข้อสุดท้ายรอ task สำเร็จจริง (allDone ฝั่ง frontend)
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
        # W_order_complete_is_the_end: sticky เหมือน nav_target_reached_confirmed — พอกด
        # Back Home หน้าก็เปลี่ยนไปแล้ว ถ้าอ่านจาก URL ปัจจุบันอย่างเดียวสัญญาณจะหายทันที
        order_completed_confirmed = False
        # W_goal_scope: จำนวนครั้งติดกันที่ action ถูกปฏิเสธเพราะ goal ถือว่าสำเร็จแล้ว — เงื่อนไข
        # reset ไม่เหมือน counter อื่นในไฟล์นี้ (ดู comment ตรงจุดใช้งานจริง)
        consecutive_goal_scope_reject_count = 0
        # W21 (Batch/Bulk): ผลของ action ล่าสุด ใช้คู่ _BULK_SAFE_REPEAT_TYPES ตอนเช็ค repeat
        last_action_succeeded: Optional[bool] = None
        # W64[7.1]: label field ล่าสุดที่ fill/select — ใช้ใน nudge เท่านั้น
        last_filter_field_label = ""
        recent_actions: list[dict] = []  # เก็บ action ล่าสุดไว้เช็ค pattern วนซ้ำ (คาบ 2-4)
        # W31: จำนวนครั้งที่บังคับ recovery แล้ว (_MAX_FORCED_LOOP_RECOVERIES)
        forced_recovery_count = 0
        # W7[A]/W22: [(step, len(messages) หลังจบ step)] — cut point ที่ตรงจุดเริ่ม turn จริง
        step_boundaries: list[tuple[int, int]] = []
        # W50 (delta digest): digest_lines สะสมข้ามรอบ compaction digest_upto_step = step สุดท้ายที่สรุปแล้ว
        digest_lines: list[str] = []
        digest_upto_step = 0

        # W22: page_text ไม่เปลี่ยน -> ใช้ manual/long-term context ของรอบก่อน (deterministic จาก
        # goal+page_text) ประหยัด ChromaDB call และ token ที่ส่งจริง (ไม่พึ่ง cache ของ provider)
        last_page_text_for_context: Optional[str] = None
        last_manual_context = ""
        last_long_term_context = ""
        _CONTEXT_UNCHANGED_NOTE = "(identical to the previous step — the page has not changed)"

        # ────────────────────────────────────────────────────────────
        # [run_task 4] ตัดสินว่าต้อง goto(url) ไหม (เว็บเป้าหมายเปิดอยู่แล้วหรือยัง)
        # ────────────────────────────────────────────────────────────
        # W12: ตัดสิน goto จาก page.url จริง (ทุกโหมด ไม่ใช่แค่ CDP)
        # W19: เทียบ domain กับ url เป้าหมาย — เดิมเช็คแค่ว่าง session ที่ reuse page จึงไม่ไปเว็บใหม่
        # ตรงกัน = perceive ต่อจากหน้าเดิม ("sign in" บนเว็บที่เปิดค้าง); ไม่ตรง/ว่าง = goto(url)
        current_domain = extract_domain(page.url) if page.url not in ("about:blank", "") else ""
        target_domain = extract_domain(url)
        site_already_open = bool(target_domain) and current_domain == target_domain
        skip_initial_goto = site_already_open

        try:
            # ────────────────────────────────────────────────────────────
            # [run_task 5] เปิดหน้าแรก -> ปิดแบนเนอร์ -> auto-login -> fast-path nav
            # ────────────────────────────────────────────────────────────
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

            # W17: auto-login ครั้งเดียวตอนต้น task (ทั้ง goto สดและ skip_initial_goto) — credential
            # มีแต่ login ไม่ผ่าน ต้องแจ้ง user ผ่าน SSE (เดิมเงียบ) ไม่หยุด task (agent กรอกเองได้)
            # W_consent_banner: ปิดแบนเนอร์ก่อนเสมอ ไม่งั้นหาฟอร์ม login ไม่เจอ
            await _dismiss_consent_banner(page, verbose)
            _auto_login_box: dict = {}
            auto_login_failure_reason = await _maybe_auto_login(page, verbose, _auto_login_box)
            auto_login_outcome = _auto_login_box.get("result", "skipped")
            # W_consent_banner (รอบสอง): CMP โหลด async render หลัง wait_stable — ยิงซ้ำ (no-op ถ้าไม่มี)
            await _dismiss_consent_banner(page, verbose)
            if auto_login_failure_reason:
                await _emit({
                    "kind": "auto_login_failed",
                    "message": "ล็อกอินไม่สำเร็จด้วย credential ที่บันทึกไว้สำหรับเว็บนี้",
                    "reason": auto_login_failure_reason,
                })

            # W66[C] (Fast-Path Navigation, opt-in): เดินตาม nav path ที่เรียนรู้ไว้ก่อนเข้า loop —
            # หลัง auto-login เสมอ ล้มเหลว -> fallback goto url เดิมเงียบๆ lazy import กัน circular
            # W67[D] (auto-decide): ไม่ระบุ query แต่เปิด enable_nav_fastpath_auto_decide -> ใช้ goal
            # เป็น query ด้วย threshold เข้มกว่า (nav_fastpath_min_match_score) เพราะไม่มีใครยืนยัน
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

            # ────────────────────────────────────────────────────────────
            # [run_task 6] classify intent — qa_summary เข้า mini-loop ถามตอบแล้ว return
            # ────────────────────────────────────────────────────────────
            # Intent Classification: ตรวจจับ Intent ของผู้ใช้ก่อนเริ่ม Planner Loop
            initial_elements, initial_page_text = await get_snapshot(page)
            # Speed 2.1: ไม่มีอะไรเปลี่ยนหน้าระหว่างนี้กับ snapshot แรกของ loop — cache ไว้ใช้ซ้ำ
            # invalidate เฉพาะ path confirm_plan (รอ user + อาจ relaunch browser)
            cached_elements, cached_page_text = initial_elements, initial_page_text
            user_intent = await llm.classify_intent(client, model, goal, page_text=initial_page_text, provider=resolved_provider)
            if user_intent == "qa_summary":
                if verbose:
                    print(f"[intent] ตรวจพบ Intent: qa_summary — ตอบคำถาม/สรุปข้อมูลจากหน้าเว็บ (read_page_data + ค้นหา)", flush=True)
                qa_messages: list = []
                summary_text = ""
                # อัปเดต elements/page_text ทุกครั้งหลัง fill/click (ใช้ lookup label และ fallback)
                # ไม่งั้นรอบถัดไปเห็นหน้าก่อนค้นหา
                qa_elements, qa_page_text = initial_elements, initial_page_text
                qa_goal = f"{goal}{_QA_ANSWER_FORMAT_GUIDANCE}"
                # W_count_answer_check: คำถามเชิงนับส่วนใหญ่จบที่ mini-loop นี้ (บั๊กสด: ESS 7 ตอบ 6)
                # ใช้ helper ตัวเดียวกับ main loop
                qa_system_counted: dict[str, int] = {}
                qa_count_mismatch_retries = 0
                for _ in range(_QA_SUMMARY_MAX_STEPS):
                    qa_tool_name, qa_tool_input, qa_tool_use_id, qa_messages, qa_usage = await next_action(
                        client, model, qa_goal, qa_page_text, qa_messages, plan_context="",
                        # W_fill_secret_schema_gate: qa อ่านอย่างเดียว fill_secret ไม่มีวันถูก
                        allow_fill_secret=False,
                        # W_token_cut W2: qa ไม่ผ่าน _resolve_prompt_sections — ส่งบล็อก gate ครบ
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
                        # W_count_answer_check: ตีกลับถ้าไม่มีตัวเลขที่นับได้ (มีโควตา escape valve)
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

                    # W46: fill/click ได้เฉพาะ label ช่อง/ปุ่มค้นหา (_label_looks_like_search)
                    # W19 (Guard Compatibility Rule): คำถามที่ต้องนำทางก่อน ("มีผู้ใช้กี่คนในหน้า Admin")
                    # ตอบไม่ได้ — ผ่อนให้ click (ไม่รวม fill) ที่ region="navigation" ได้ ไม่ mutate ข้อมูล
                    # และ classify_action() ยังเช็คซ้ำใน execute()
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
                    # ครบโควตาแล้วไม่เรียก finish_task -> fallback summarize_page() ด้วย qa_page_text
                    # ล่าสุด (ไม่ทิ้งผลการค้นหา)
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


            # ────────────────────────────────────────────────────────────
            # [run_task 7] แผน: approved_plan / confirm_plan + เตือนถ้าแผนผิดชนิดงาน
            # ────────────────────────────────────────────────────────────
            if approved_plan:
                # W13: แผนอนุมัติจากภายนอกแล้ว (POST /api/generate_plan -> review -> execute_plan)
                # ผนวกเข้า effective_goal ทันที ใช้ตัวแปรชุดเดียวกับ confirm_plan
                plan_text = approved_plan
                effective_goal = f"{goal}\n\nFollow this confirmed plan:\n{plan_text}"
            elif confirm_plan:
                # Speed 2.1: page state จะเปลี่ยน (รอ user/relaunch) — invalidate cache
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

                # W10[F]: ทุก step เห็นแผนที่ยืนยัน (อาจถูก user แก้) — ต่อท้าย goal ไม่แทนที่
                effective_goal = f"{goal}\n\nFollow this confirmed plan:\n{plan_text}"

                # W11[A]: user ยืนยันแล้ว เปิดหน้าต่างจริง (Playwright สลับ headless กลาง process
                # ไม่ได้ ต้อง launch ใหม่) ยังไม่มี action goto หน้าเดิมซ้ำพอ
                if defer_visible_window:
                    await browser.close()
                    browser = await _launch_chromium(playwright, headless=False, channel=browser_channel)
                    page = await browser.new_page()
                    await install_ssrf_guard(page)
                    page.on("dialog", _make_dialog_handler(self.memory, verbose))
                    await goto(page, url)
                    await wait_stable(page)

            # W_goal_scope: resolve ครั้งเดียว (จาก goal ล้วน)
            # W_plan_keeps_goal_verb + W_plan_warn_not_abort: เดิม return steps=0 ทันทีเมื่อแผนผิดชนิดงาน
            # แต่แผนตรงนี้คือแผนที่ user ยืนยันแล้ว การหยุด = ตัดสินใจแทน user — เตือนตั้งแต่เทิร์นแรก
            # แต่ไม่หยุด guard ตอน execution ยังบล็อก Save จริงครบ (W_no_record_edit_for_delete_goal,
            # W_prefer_row_delete)
            plan_mismatch_reason = (
                _plan_drops_goal_operation(goal, plan_text) if plan_text else None
            )
            if plan_mismatch_reason:
                # ต่อท้าย effective_goal ไม่ใช่ใส่ messages: messages ยังว่าง nudge จะเป็น user turn
                # ซ้อนก่อน goal (บาง provider ไม่รับ) และอยู่ใน goal ทำให้ติดไปทุก step
                print(f"\u26a0\ufe0f [plan] {plan_mismatch_reason}", flush=True)
                effective_goal = (
                    f"{effective_goal}\n\n"
                    + _PLAN_MISMATCH_WARNING_TEMPLATE.format(reason=plan_mismatch_reason)
                )

            # ────────────────────────────────────────────────────────────
            # [run_task 8] ค่าที่คำนวณจาก goal ครั้งเดียวก่อนเข้าลูป
            # ────────────────────────────────────────────────────────────
            # goal_nav_target ใช้แม้มี confirmed plan (planner อาจเติมข้อ "ยืนยันผล" ที่ไม่มีวันเสร็จ)
            # W_prompt_sections: สะสมข้าม step (ดู _resolve_prompt_sections)
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

            # ────────────────────────────────────────────────────────────
            # [run_task 9] ลูปหลัก: Perceive -> LLM -> guard -> Act -> Verify
            #   ขั้นย่อย (ก)-(ฐ) ด้านใน
            # ────────────────────────────────────────────────────────────
            for _ in range(max_steps):
                # ────────────────────────────────────────────────────────────
                # (ก) backstop guard ที่หัวลูป + เช็คว่า goal สำเร็จแล้วหรือยัง
                # ────────────────────────────────────────────────────────────
                # W_step_budget: นับ "รอบ" แยกจาก steps_taken (ดูคำอธิบายที่จุดประกาศตัวแปร)
                iterations_used += 1

                # W_token_cut W3: backstop — guard ปฏิเสธเกินเพดาน = จบ task ตามจริง เช็คที่หัวลูป
                # จุดเดียวที่ break ได้โดยไม่มี tool_use ค้าง
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

                # W_login_check_once (P4.7): _login_form_needs_password() ถูกเรียก 2 ครั้งต่อรอบ ระหว่างนั้น
                # มีแค่ LLM call DOM ไม่เปลี่ยน — cache ต่อรอบปลอดภัย ห้าม cache ข้ามรอบ
                login_form_state: Optional[bool] = None

                async def _login_form_needs_password_cached() -> bool:
                    nonlocal login_form_state
                    if login_form_state is None:
                        login_form_state = await _login_form_needs_password(page)
                    return login_form_state
                # Speed 2.1: รอบแรกใช้ snapshot ที่ cache ไว้ตอน classify intent (ถ้ายังไม่ invalidate)

                # W_goal_scope: หลักฐานว่า goal สำเร็จ เช็คก่อน next_action() ทุก step (page.url สดเสมอ)
                # steps_taken > 0 กัน lock ตั้งแต่ step แรก
                goal_scope_satisfied_reason: Optional[str] = None
                if steps_taken > 0:
                    # W_plan_cursor_not_proof (live run 2026-08-28): plan_cursor คือตัวนับ ไม่ใช่หลักฐาน —
                    # Select All เป็น action ที่ 5 พอดี cursor ครบ gate บล็อกปุ่ม Delete แล้วปิด task
                    # success=True ทั้งที่ ESS ยังอยู่ 9 คน goal ลบแบบมีเงื่อนไขมีการวัดตรง (แถวที่เหลือ)
                    # การวัดชนะตัวนับเสมอ goal อื่นใช้ธงของแผนตามเดิม
                    # W_plan_counter_claims_a_password_change: ฟอร์มเปลี่ยนรหัสที่ยังไม่ครบก็ชนะตัวนับเช่นกัน
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
                        elif _order_is_complete(goal, page.url) or order_completed_confirmed:
                            order_completed_confirmed = True
                            goal_scope_satisfied_reason = (
                                "the order was already placed and the site showed its "
                                "order-complete page"
                            )
                        elif _goal_targets_existing_records_only(goal):
                            # W_zero_records_done: งานลบ/แก้ตามเงื่อนไขเสร็จเมื่อแถวที่ตรงเหลือ 0 —
                            # สัญญาณเดียวกับ guard premature-deletion ใช้ทิศตรงข้าม None = ไม่ใช่หน้าตาราง
                            remaining = await _scan_remaining_target_records_once(page)
                            if remaining is not None and remaining[0] == 0:
                                goal_scope_satisfied_reason = (
                                    "the filtered table has no matching rows left "
                                    f"({remaining[1]!r}), so there is nothing more to act on"
                                )

                # ────────────────────────────────────────────────────────────
                # (ข) Perceive: snapshot + ปิดแบนเนอร์ / CAPTCHA / login ใหม่ถ้า session หลุด
                # ────────────────────────────────────────────────────────────
                # W_step_trace: จับเวลา snapshot / LLM / action แยกกัน (เดิมไม่มี instrumentation
                # ตอบไม่ได้ว่า task 693 วินาทีหมดไปกับอะไร)
                _snapshot_started_at = time.monotonic()
                if steps_taken == 0 and cached_elements is not None:
                    elements, page_text = cached_elements, cached_page_text
                else:
                    elements, page_text = await get_snapshot(page)
                step_snapshot_seconds = time.monotonic() - _snapshot_started_at
                await _emit_screenshot(steps_taken)
                # W5[A] verify: page_text ล่าสุดเป็นหลักฐาน DOM แนบกับผลลัพธ์ทุก path ให้เทียบกับ
                # message ที่ LLM อ้าง
                final_page_text = page_text

                # W_consent_banner_midtask: เห็นแบนเนอร์ใน snapshot ปิดทันทีแล้ว perceive ใหม่
                if _snapshot_shows_consent_banner(elements):
                    if await _dismiss_consent_banner(page, verbose):
                        elements, page_text = await get_snapshot(page)
                        final_page_text = page_text

                # W_captcha_detect (P3.8): เช็คหลังปิดแบนเนอร์ (แบนเนอร์ทำ snapshot ดูว่างคล้าย bot wall)
                # ถามคนครั้งเดียวต่อ task ยังติดก็ปล่อยลูปเดินต่อ ไม่ฆ่า task
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

                # W_session_drift (live, goal ลบ ESS): ถึงหน้า Admin แล้วโมเดล go_back เด้งกลับหน้า
                # login ไม่มีอะไรพากลับเข้าระบบ เสีย 18 จาก 22 step คลิกลิงก์การตลาด — เจอฟอร์ม login
                # อีกครั้งขณะมี credential: ปิดแบนเนอร์ + login ใหม่ (กลไกเดิม) แล้ว perceive ใหม่
                # มีโควตา: login ซ้ำแล้วยังเด้ง = credential ใช้ไม่ได้ เลิกพยายาม
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

                # ────────────────────────────────────────────────────────────
                # (ค) เตรียม context: คู่มือ/memory/สัญญาณ no-op/pacing
                # ────────────────────────────────────────────────────────────
                # W6[B]: ดึงคู่มือ + long-term memory ใหม่เมื่อหน้าเปลี่ยน (ไม่ throw) — to_thread
                # เพราะ embedding/ChromaDB เป็น sync จะบล็อก event loop ของ Playwright
                # Speed 2.2: ทั้งคู่ embed input เดียวกันด้วย function เดียวกัน — embed ครั้งเดียวแล้ว
                # gather query สอง collection พร้อมกัน (_client_lock ยังกัน race ตอน init) embed ล้มเหลว
                # -> query_embedding=None ให้ embed เองตามเดิม
                # W22: page_text เหมือนเดิม -> ข้าม retrieval ใช้ marker สั้นแทน
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

                # W50 (client-side action verification): action ก่อนหน้า [OK] และควรเปลี่ยนหน้า
                # (_VERIFICATION_SIGNAL_ACTION_TYPES) แต่ page_text เหมือนเดิมทุกตัวอักษร -> เตือนว่าอาจ
                # เป็น no-op (ใช้ page_changed_for_context ไม่ยิง browser เพิ่ม)
                verification_context = ""
                # W_already_logged_in_but_told_to_log_in (gate 2026-09-07, add_candidate): goal ขึ้นต้น
                # "Log in with ..." แต่ auto-login ทำไปแล้ว โมเดลยัด username/password ลงช่อง Search
                # แล้วหลงทาง — บอกเฉพาะเมื่อ login สำเร็จ *และ* goal พูดถึง login ในช่วง step แรกๆ
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

                # W7[A]: สรุป action ที่ล้มเหลวใน task นี้ ป้อนเข้า prompt ทุก step
                memory_context = self.memory.failed_actions_summary()

                # W32: action ล่าสุดไม่กี่ step (ทั้งสำเร็จและล้มเหลว) แยกจาก memory_context
                # ด้านบนที่กรองเฉพาะ fail — ดู ShortTermMemory.recent_actions_summary()
                action_history_context = self.memory.recent_actions_summary()

                # W9[A]: ใช้ vision_context ของรอบนี้แล้วเคลียร์ทิ้งทันที (one-shot —
                # ดู pending_vision_context ด้านบนสุดของ run_task())
                vision_context, pending_vision_context = pending_vision_context, ""

                # W41: หน่วงเฉพาะส่วนที่ขาดให้ครบ step_pacing_delay_seconds นับจาก next_action() ครั้งก่อน
                # W_timing_gap: การรอนี้ตั้งใจ (rate limit) แต่ต้องเห็นในรายงาน แยกจากความช้าที่แก้ได้
                step_pacing_seconds = 0.0
                if last_llm_call_at is not None:
                    elapsed = time.monotonic() - last_llm_call_at
                    remaining = settings.step_pacing_delay_seconds - elapsed
                    if remaining > 0:
                        step_pacing_seconds = remaining
                        await asyncio.sleep(remaining)

                # ────────────────────────────────────────────────────────────
                # (ง) เรียก LLM: gate สคีมา fill_secret + เลือก prompt sections + ยุบ history
                # ────────────────────────────────────────────────────────────
                # W43: plan_text None (ad-hoc) -> "" ให้ next_action ไม่ต้องรู้จัก Optional
                # W_fill_secret_schema_gate (ดู llm.py): คำนวณก่อนเรียก LLM แล้วตัด fill_secret ออกจาก
                # สคีมาถ้าใช้ไม่ได้ — guard ด้านล่างใช้ค่าเดียวกัน schema กับ guard จึงตรงกันเสมอ
                # W_secret_gate_stays_page_only (regression 2026-09-03): เคยเปิดตั้งแต่ goal *พูดถึง*
                # การเปลี่ยนรหัส -> gpt-5.4-mini ยิง fill_secret มั่วตั้งแต่หน้า login (5/5 failure)
                # กฎ W20 เปิดผ่าน _resolve_prompt_sections แยกต่างหากแล้ว ธงนี้ถามแค่
                # "ยืนอยู่บนฟอร์มเปลี่ยนรหัสจริงไหม"
                allow_fill_secret = await _page_looks_like_change_password_form(page)
                # W_secret_stays_in_schema_forever: สคีมาแคบกว่าบริบท — Current Password กรอกแล้วตัด
                # fill_secret ทันที ส่วน guard/prompt ยังใช้ allow_fill_secret ตัวกว้าง (ไม่งั้น guard
                # เข้าใจผิดว่าไม่ใช่หน้าเปลี่ยนรหัสแล้วบังคับ recovery)
                fill_secret_in_schema = (
                    allow_fill_secret and await _current_password_field_is_empty(page)
                )

                # W_steptimeout: timeout ต่อ LLM call (config.py::llm_step_timeout_seconds) — TimeoutError
                # ทะลุไป except ของ run_task (W_loop_crash) รายงาน step ที่ค้างพร้อมงานที่ทำแล้ว
                _llm_started_at = time.monotonic()
                prompt_sections = _resolve_prompt_sections(
                    prompt_sections, goal=goal, plan_text=plan_text, elements=elements,
                    allow_fill_secret=allow_fill_secret,
                    # site_manual_context เป็นพารามิเตอร์ของ run_task จึงมีค่าเสมอ —
                    # ห้ามใช้ effective_site_manual ตรงนี้ มันถูกกำหนดค่าทีหลังในลูป
                    manual_context=site_manual_context or manual_context or "",
                )

                # W_token_trim (P2/M1): ยุบ snapshot เก่า เก็บเต็ม 2 อันล่าสุด
                messages = _dedupe_stale_snapshots(messages, keep_last_full=1)

                # W_token_cut W5: ยุบ user turn เก่าทั้งก้อนเหลือ Goal + stub (ดู _compact_stale_user_turns)
                messages, _w5_removed = _compact_stale_user_turns(messages, goal)
                if _w5_removed:
                    history_compaction_events += 1
                    history_chars_saved += _w5_removed

                # W_token_cut W7: บล็อกกฎ gated ใน turn เก่าเหลือ 1 บรรทัดอ้างอิง
                messages, _w7_removed = _dedupe_stale_gated(messages, keep_last_full=0)
                if _w7_removed:
                    gated_deref_events += 1
                    gated_chars_saved += _w7_removed

                # W_token_trim (P3/M3): full manual on the first step and the first step after any
                # compaction (which spliced the earlier copy out); short id+summary otherwise
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

                # ────────────────────────────────────────────────────────────
                # (จ) tool พิเศษ: unknown / request_user_input / finish_task (guard ก่อนยอมรับจบ)
                # ────────────────────────────────────────────────────────────
                # W_unknown_tool: ชื่อ tool ที่โมเดลมโนเคยตกไป execute() ได้ "unknown action" ที่ไม่บอก
                # ว่าชื่อผิด — ตอบตรงจุดพร้อมรายชื่อ tool แล้วไปต่อโดยไม่นับ step
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

                # W_resume: ตรวจก่อน finish_task (คนละ tool) — หยุดรอ human ผ่าน _request_user_input()
                # แล้วทำ loop ต่อด้วยคำตอบเป็น tool_result ไม่ return ไม่ reset plan
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
                    # W_nobody_is_watching (step_trace 2026-09-08): 51 ครั้งได้คำตอบว่างทั้งหมด (eval/gate
                    # ไม่มีคนเฝ้า harness อนุมัติเสมอแต่ไม่มีข้อความ) — บอกความจริงและปิดทางถามซ้ำทั้ง task
                    if provided and not (answer or "").strip():
                        provided = False
                        request_user_input_count = _MAX_REQUEST_USER_INPUT_CALLS
                        # W_no_answer_is_not_an_exit (gate fedd2ed): ข้อความแรกเสนอทาง finish_task(false)
                        # โมเดลหยิบทันที rag_integration จบใน 3 step — "ไม่มีใครตอบ" ต้องไม่ใช่ใบอนุญาตยอมแพ้
                        result_text = (
                            "[No answer] nobody is available to answer in this run — this is normal "
                            "and not a failure. Carry on with the task using what is already on the "
                            "page, the goal, and any attached manual; pick sensible values yourself "
                            "for anything routine (a name, a postcode). Do not call request_user_input "
                            "again. Give up only if the goal genuinely cannot be done without a value "
                            "that exists nowhere — and say which value that is."
                        )
                    else:
                        result_text = (
                            f"the user answered: {answer}" if provided
                            else "[No answer] the user declined or did not answer within the time limit"
                        )
                    # W_secret_answer_not_logged: คำตอบ sensitive ไปถึงโมเดล (อยู่ใน messages) แต่ปิดใน
                    # LOG/step_trace/ShortTermMemory คำตอบทั่วไปเห็นได้ตามเดิม
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

                    # ยังเหลือรอบ + finish_task call จริง (มี tool_use_id) + ยังไม่เกินโควตา -> เตือนให้ลองต่อ
                    if (
                        not claimed_success
                        and tool_use_id
                        # W_step_budget: เทียบรอบที่ใช้ไปกับงบรอบ ไม่ใช่ steps_taken
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

                    # W5[A] verify: finish_task(true) ตอน steps_taken=0 ให้ยืนยันอีกครั้ง (ไม่ block เด็ดขาด)
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

                    # ACC-3: steps_taken > 0 แต่ไม่มี mutating/evidence action สำเร็จเลย ก็น่าสงสัย
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

                    # Task4 (W19): เช็คทุกครั้งที่ claimed_success (error อาจโผล่หลังหลาย step)
                    detected_errors: list[str] = []
                    if claimed_success and tool_use_id:
                        detected_errors = await _scan_validation_errors(page)
                    # W65[2] (Error Passthrough): error fatal ข้าม nudge-retry บังคับความจริงลงผลทันที
                    # error ทั่วไปยังให้ LLM ลองแก้ก่อนตามเดิม
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
                        # retry ครบแล้วยังมี error — ปล่อยผ่าน (escape valve) แต่ tag ว่าน่าสงสัย
                        completion_verification = "EXECUTION_FAILED_NEEDS_REPAIR"

                    # W22/W64[7.1] (DOM-Based Post-Action Verification): goal ลบหรือแก้ทั้งหมด อ่านแถวที่
                    # เหลือจาก DOM ก่อนยอมรับ finish_task(true) — สัญญาณ "เสร็จ" เดียวกัน: แถวที่ตรง
                    # เงื่อนไขต้องเหลือ 0
                    remaining_records: Optional[tuple[int, str]] = None
                    if (
                        claimed_success and tool_use_id
                        and (_is_deletion_intent_goal(goal) or _is_edit_all_intent_goal(goal))
                    ):
                        remaining_records = await _scan_remaining_target_records(page)

                    # W_delete_all_intent: สองชั้นที่ _scan_remaining_target_records() จับไม่ได้ (2026-08-26):
                    # (1) ไม่เคยกด Search — "(N) Records Found" เป็นของตารางที่ยังไม่กรอง
                    # (2) finish บนหน้าที่ไม่มีตาราง (คลิก footer ก่อนจบ) — None เดิมปล่อยผ่าน ซึ่งสำหรับ
                    #     "ลบทั้งหมด" คือยอมรับคำอ้างที่ไม่มีหลักฐาน
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

                        # W_delete_all_intent: ไม่มี "(N) Records Found" (เว็บส่วนใหญ่) แต่มีตาราง -> นับแถว
                        # ที่ตรงเงื่อนไขเองแล้วป้อน guard เดิม
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
                    # W_empty_table_needs_right_filter: ตารางว่างเป็นหลักฐานได้เมื่อตัวกรองตรง goal เท่านั้น
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
                        # retry ครบแล้วยังเหลือแถว — ต่างจาก validation guard (ตีความได้หลายแบบ) ตัวเลขแถว
                        # นับตรง ต้องบังคับความจริงเสมอ เขียนทับ claimed_success และ message ของ LLM
                        completion_verification = "EXECUTION_FAILED_NEEDS_REPAIR"
                        remaining_count, _ = remaining_records
                        claimed_success = False
                        # W64[7.1]: ข้อความตาม intent — "ยังไม่ถูกลบ" vs "ยังไม่ถูกแก้ไข"
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

                    # W_count_answer_check: ดู _MAX_COUNT_ANSWER_MISMATCH_RETRIES สำหรับเงื่อนไข 3 ข้อ
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

                    # W63[7.2] (Strict Table Assertion): มี verify_text -> อ่าน table body จริงก่อนยอมรับ
                    verify_text = str(tool_input.get("verify_text") or "").strip()
                    table_item_found = True
                    if (
                        claimed_success and tool_use_id and verify_text
                        and (goal_wants_a_record_change or wrote_a_value_this_task)
                    ):
                        # W_verify_text_on_delete_goal (2026-08-31): โมเดลส่ง verify_text="No Records Found"
                        # บนงานลบ guard หาเป็นแถวในตารางไม่เจอ พลิกงานสำเร็จเป็น VERIFICATION_FAILED —
                        # ตารางว่างคือหลักฐานว่าสำเร็จของงานลบ ข้ามทั้งงานลบและวลี "ไม่มีผลลัพธ์"
                        # (_RECORD_COUNT_ZERO_TEXTS)
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
                            # W_token_cut W3: เคยตีกลับเหตุผลนี้แล้ว ไม่เตือนซ้ำ ไปทาง "ยอมรับพร้อม tag"
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
                        # W64[7.2]: มี toast ยืนยันก่อนหน้า = บันทึกสำเร็จจริง ไม่ force success=False
                        # แค่เขียน message ตามสเปค user (อาจเป็นปัญหา search/filter/pagination)
                        completion_verification = "OK_SAVE_CONFIRMED_NOT_IN_TABLE"
                        tool_input["message"] = "บันทึกข้อมูลเรียบร้อยแล้ว แต่ไม่พบรายการในตารางการค้นหา"
                    elif not table_item_found:
                        # retry ครบ ไม่เจอในตาราง และไม่มี toast — บังคับความจริง
                        # ("VERIFICATION_FAILED: Item not found in results table." ตามสเปค user)
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

                # ────────────────────────────────────────────────────────────
                # (ฉ) guard ก่อน dispatch: ฟอร์ม login / fill_secret / action ซ้ำและวนลูป
                # ────────────────────────────────────────────────────────────
                # (2026-07-13) code guard: ช่อง password ยังว่าง -> ห้าม action อื่นนอกจาก fill
                # (โมเดลเล็กไม่ทำตาม prompt) ยกเว้น goto เสมอ (อาจต้องแก้เส้นทาง/multi-hop)
                # W_secret_fill_is_a_fill: fill_secret คือวิธีเดียวที่กรอกรหัสที่เก็บไว้ ต้องอยู่ในรายการ
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


                # W_fill_secret_hardening: ปฏิเสธ fill_secret นอกบริบทเปลี่ยนรหัสก่อน dispatch เสมอ —
                # ไม่มี escape valve (ปล่อยผ่าน = พิมพ์รหัสจริงลง element ที่ไม่รู้จัก) โควตาใช้บังคับ
                # recovery แทน (_force_loop_recovery)
                # W_fill_secret_schema_gate: ยังต้องมี guard แม้ schema ตัดแล้ว — provider หลุดส่ง type
                # นอก enum ได้ และ fastpath/แผนเก่าอาจ replay โดยไม่ผ่าน schema
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
                        # W_state_guard_shortcut: ลอง click element ที่ตรง goal ก่อน fallback go_back/scroll
                        # (click ไม่มีทางเขียน credential ผิดที่)
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

                # loop-detection: action เดิมเป๊ะติดกัน (นับทั้ง success/fail — สั่งซ้ำไม่เปลี่ยนก็ไม่คืบหน้า)
                # W21 (Batch/Bulk): ยกเว้น action ที่ต้องยืนยันจาก human อยู่แล้ว (DEFAULT_NEEDS_CONFIRMATION)
                # ที่ครั้งก่อนสำเร็จ — ลบแถวแรกซ้ำๆ แถวถัดไปเลื่อนขึ้นมาได้ index เดิม เป็นความคืบหน้าจริง
                # และ human กดอนุมัติทุกครั้ง ถ้าครั้งก่อน fail ยังนับตามปกติ
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

                # W_same_label_loop: นับซ้ำด้วย (type, label) แทน index — จับ "ของชนิดเดียวกันคนละแถว"
                # ข้ามถ้าไม่มี label หรือเป็น bulk-safe repeat หา label เองตรงนี้ (action_label คำนวณ
                # ทีหลัง ย้ายขึ้นมาจะสลับลำดับ guard อื่น)
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
                    # W31: บังคับ recovery แทนจบ task (False = เกินโควตา ค่อยจบ)
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

                # loop-detection คาบ 2-4 (2026-07-13/07-15): เก็บแค่ _MAX_CYCLE_WINDOW ตัวล่าสุด
                # W29: เก็บ normalized_tool_input (ตัด completed_plan_step)
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

                # ────────────────────────────────────────────────────────────
                # (ช) resolve label/tag/type ของเป้าหมาย (และ then_click_index)
                # ────────────────────────────────────────────────────────────
                # label ของเป้าหมาย ส่งให้ classify_action() เช็คคำเสี่ยง (LLM เลือก click กับปุ่ม "Remove" ได้)
                action_index = tool_input.get("index")
                action_label = next(
                    (e["label"] for e in elements if e["index"] == action_index), ""
                ) if action_index is not None else ""
                # W_search follow-up: tag จริง (เช่น "a") เป็นสัญญาณโครงสร้างสำรอง — label เนื้อหาอิสระ
                # (ชื่อวิดีโอ) ไม่ match keyword ใดเลย (permission/rules.py::ANCHOR_TAG)
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
                # W_search follow-up 2: attribute type แยกช่องค้นหาธรรมดาออกจาก input เสี่ยง
                # (permission/rules.py::SAFE_INPUT_TAG/RISKY_INPUT_TYPES)
                action_element_type = next(
                    (e.get("type", "") for e in elements if e["index"] == action_index), ""
                ) if action_index is not None else ""

                # W_chain (Compound Actions): resolve then_click_index แบบเดียวกัน ให้ action ที่สอง
                # มีสัญญาณครบใน classify_action() (llm.py::_BROWSER_ACTION_PARAMS, actions._maybe_chain_click)
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

                # ────────────────────────────────────────────────────────────
                # (ซ) guard ตาม goal: scope / credential / obscured / profile / ลบ / save / สร้าง / active / ตัวกรอง / กรองก่อนลบ
                # ────────────────────────────────────────────────────────────
                # W_goal_scope (Goal Boundary Gate): goal สำเร็จแล้ว -> อนุญาตแค่ action read-only
                # nudge 1 ครั้งแล้ว HARD stop — ไม่มี escape valve เพราะปล่อย action นอก scope คือ failure
                # mode ที่ guard นี้แก้ edge case ที่ยอมรับ: session หมดอายุหลังถึงเป้าหมายจะบล็อก fill
                # ที่ใช้กู้ด้วย (หายาก)
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
                    # W_goal_scope_false_success (live run 2026-08-27): ทางนี้ตั้ง success=True แล้ว break
                    # ไม่ผ่าน finish_task guard ของ W_delete_all_intent จึงไม่ได้ตรวจ (กรองผิดเป็น Admin
                    # ไม่ได้ลบอะไร แต่ขึ้น Done) — ใช้หลักฐานชุดเดียวกัน (_scan_remaining_target_records)
                    # ยังเหลือแถวห้ามอ้างสำเร็จ อ่านไม่ได้ = คงพฤติกรรมเดิม
                    success = True
                    final_message = _GOAL_SCOPE_GATE_HARD_STOP_MESSAGE_TEMPLATE.format(
                        reason=goal_scope_satisfied_reason,
                    )
                    if delete_all_condition_values:
                        # W_empty_table_needs_right_filter: เช็คตัวกรองก่อนตัวนับ (กรองผิด 0 ไม่มีความหมาย)
                        # elements ตรงนี้เป็น snapshot ของ iteration ปัจจุบันแล้ว
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
                            # W_plan_cursor_not_proof: ข้อความนับอาจไม่ตรง pattern (หลัง Select All เป็น
                            # "(9) Records Selected") — นับแถวจากตารางตรงด้วย helper เดียวกับ finish_task
                            # W_column_headers_fallback: ทิศ "อ้างว่าเสร็จ" -> require_columns=True
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
                # else: action read-only ที่อนุญาต — ห้าม reset counter (ไม่งั้นโมเดลแทรก read/wait
                # คั่นหนี hard-stop ได้ตลอด)

                # W_no_credential_flow: goal ไม่พูดถึงรหัส/บัญชี ห้ามเข้า Change Password — hard reject
                # (ล็อก user ออก go_back กู้ไม่ได้)
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

                # W_reject_obscured_click (P8/M3): ถูกบัง + มี dialog เปิด = timeout แน่นอน อ่านจาก
                # snapshot ([in open dialog]) ไม่ยิง DOM เพิ่ม
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

                # W_prefer_row_delete: เมนูโปรไฟล์ไม่เคยเป็นทางไปงานของ goal (เว้น goal พูดถึงบัญชี)
                # และนำไป logout/เปลี่ยนรหัส — ใช้ป้ายจาก perception
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

                # W_prefer_row_delete: goal ลบล้วน กด Edit ทั้งที่หน้ามีปุ่มลบ = เดินผิดทาง หาปุ่มลบจาก
                # snapshot ไม่มีก็ปล่อยผ่าน (เว็บที่ต้องลบผ่านหน้า Edit)
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

                # W_goal_values_before_save (gate 2026-09-07): ครอบทั้ง click บันทึกตรงๆ และการพ่วงกับ
                # fill (then_click_index) — โมเดลพ่วง submit แทบทุกครั้ง ดูแค่ click จะพลาดทางหลัก
                # (บทเรียน W_chained_submit_after_check) โควตา 2 ครั้ง เดาผิดเสียแค่สองเทิร์น
                _commits_form = (
                    tool_input.get("type") in ({"click"} | DEFAULT_NEEDS_CONFIRMATION)
                    and _action_commits_a_record_edit(tool_input, action_label)
                ) or (
                    tool_input.get("then_click_index") is not None
                    and bool(_RECORD_COMMIT_LABEL_RE.search(then_label or ""))
                )
                if _commits_form and missing_value_rejection_count < _MAX_MISSING_VALUE_REJECTIONS:
                    _pending = {
                        v.strip() for v in (tool_input.get("text"), tool_input.get("label"))
                        if isinstance(v, str) and v.strip()
                    }
                    missing_values = await _values_missing_before_commit(
                        page, goal, values_written_this_task | _pending,
                    )
                    if missing_values:
                        missing_value_rejection_count += 1
                        nudge_text = _MISSING_GOAL_VALUES_NUDGE.format(
                            what=(action_label.strip() or str(tool_input.get("type"))),
                            count=len(missing_values),
                            values=", ".join(f"'{v}'" for v in missing_values),
                        )
                        if verbose:
                            print(f"[missing-values] ยังไม่ได้กรอก: {missing_values}")
                        messages = append_tool_result(messages, tool_use_id, nudge_text)
                        messages.append(_build_nudge_message(
                            resolved_provider, f"⚠️ [Important system command]: {nudge_text}",
                        ))
                        continue

                # W_no_record_edit_for_delete_goal: goal ลบล้วน ห้ามกดบันทึกฟอร์มแก้ไข — hard reject
                # (เขียนทับ record กู้ไม่ได้) goto ไม่อยู่ในชุดโดยเจตนา (URL ไม่ commit ฟอร์ม
                # W_no_create จับหน้า Add แล้ว) DEFAULT_NEEDS_CONFIRMATION ครอบ type "submit"
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

                # W_no_create_for_existing_goal: goal ลบ/แก้ของที่มีอยู่ ห้ามเข้า flow สร้าง — ไม่มีโควตา
                # (สร้าง record จริงกู้ยาก)
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

                # W_goal_precheck: โค้ดหนุนกฎ W19 "ห้ามคลิก [already active] ซ้ำ" (marker deterministic)
                # live (OrangeHRM/openai): ถึง Admin ตั้งแต่ step 1-2 แต่คลิกซ้ำ 5 ครั้งจนหมด step และ
                # เผลอกด Edit/Save บนข้อมูลจริง มีโควตา เกินแล้วปล่อย dispatch (no-op)
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

                # W_filter_scope_guard (live run 2026-08-27): goal บอกแค่ userrole=ess agent ตั้ง
                # Status=Enabled ด้วย ESS ที่ถูก disable หายจากตาราง "ลบให้หมด" จึงไม่ครบ — deterministic
                # จาก prefix ชื่อ field ของ perception มีโควตา (บางเว็บบังคับเลือกช่องอื่นก่อน Search)
                # W_filter_scope_via_dropdown (live 2026-09-01): ต้องครอบทุกชนิดที่ตั้ง filter ได้
                # (press_key บน dropdown, check บน toggle) — ตัวเลือกในลิสต์ ("Enabled" ไม่มีชื่อ field)
                # ปิดเองอัตโนมัติ เพราะลิสต์เปิดได้ก็ต่อเมื่อ trigger ผ่าน guard นี้แล้ว
                touched_field = _filter_field_from_label(
                    action_label or "", str(tool_input.get("type") or ""),
                )
                # W_filter_already_satisfied: ตัวกรองมีค่าที่ goal ขอแล้ว กดซ้ำ = no-op ที่ guard เดิม
                # มองไม่เห็น (index เปลี่ยนทุก snapshot, same-label ตั้งไว้ 4) อ่านจาก label ของ perception
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

                # W64[7.1] (Filter Order): บล็อก row-action ทันทีหลัง fill/select ที่ยังไม่กด Search
                # deterministic ล้วน ไม่ผ่าน RAG/middleware
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

                # W_delete_all_intent (safety gate สำคัญสุด): ก่อน click ทำลายข้อมูลครั้งแรกของ goal ลบ
                # แบบมีเงื่อนไข ตารางต้องกรองตรงเงื่อนไขแล้ว (อ่าน DOM เทียบกับ goal) — เคยลบแถวจากตาราง
                # ที่ยังไม่กรองบนเดโมจริง กู้คืนไม่ได้ บังคับ "ลำดับ" ไม่บังคับ "วิธี"
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
                    # ตรงเงื่อนไข / เช็คไม่ได้ (fail-safe) / หมดโควตา -> ปล่อยผ่านและไม่เช็คซ้ำทั้ง task
                    # (แถวถัดไปอยู่ในตารางชุดเดียวกันที่ยืนยันแล้ว)
                    destructive_filter_verified = True

                # ────────────────────────────────────────────────────────────
                # (ฌ) permission RAG + middleware แล้ว dispatch จริงผ่าน actions.execute()
                # ────────────────────────────────────────────────────────────
                # W7[B]: RAG permission ด้วย query แคบเฉพาะ action นี้ (_build_permission_query)
                permission_query = _build_permission_query(tool_input, action_label)
                permission_chunks = (
                    await asyncio.to_thread(
                        retriever.retrieve, query=permission_query, k=_PERMISSION_RAG_CHUNKS_PER_STEP
                    )
                    if permission_query else []
                )
                manual_permission_guidance = "\n".join(f"- {c}" for c in permission_chunks)

                # W19-8 (Semantic Redundancy) / W19-2 (Safety & Performance Middleware) mutually exclusive
                # (ไม่เรียก LLM สองครั้งเรื่องเดียวกัน) — เปิด middleware ใช้ตัวนั้น (รวม permission) ไม่งั้น
                # fallback semantic redundancy ข้าม navigate/รอ/อ่าน
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
                    # escalate-only: REQUIRES_CONSENT/BLOCKED -> ต่อวลี MANUAL_CONFIRMATION_KEYWORDS เข้า
                    # manual_permission_guidance ให้ classify_action() escalate ผ่านกลไกเดิม ไม่ถาม user
                    # ตรงนี้ (กันถามซ้ำ) และไม่มีทางลดระดับความเสี่ยง
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

                # W30: URL ก่อน dispatch ไว้เทียบหลัง action (ไม่เตือนสำหรับ goto/switch_tab/go_back)
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

                # ────────────────────────────────────────────────────────────
                # (ญ) หลัง action: สถานะ/แท็บ/reset โควตา/ธงตัวกรอง/บันทึก history + SSE
                # ────────────────────────────────────────────────────────────
                # W_retry_value_has_no_home: จำ label ช่องที่ agent กรอกเอง (ไม่รวม fill_secret =
                # รหัสปัจจุบันที่ระบบกรอก)
                if result.success and tool_input.get("type") in _VALUE_WRITING_ACTION_TYPES:
                    wrote_a_value_this_task = True  # W_verify_text_needs_a_write
                    for _written in (tool_input.get("text"), tool_input.get("label")):
                        if isinstance(_written, str) and _written.strip():
                            values_written_this_task.add(_written.strip())
                if (
                    result.success
                    and tool_input.get("type") == "fill"
                    and action_label
                    and action_label not in agent_filled_field_labels
                ):
                    agent_filled_field_labels.append(action_label)

                # W_tab_rebind: เปลี่ยน page ก่อนอ่านหน้าต่อ และผูก dialog handler ให้แท็บใหม่
                # (ไม่งั้น alert() ค้าง action หลังจากนั้น timeout เงียบ)
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

                # W_guard_quota_reset: ตัวนับ premature_* เดิมไม่เคย reset โควตาจึงเป็น "ทั้ง task" —
                # guard false-completion ตายถาวรตั้งแต่กลาง task ตรงจุดที่ verification สำคัญที่สุด
                # reset เมื่อ action สำเร็จจริงเท่านั้น (fail ไม่คืนโควตา ไม่งั้นวนขอ nudge ไม่จำกัด)
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

                # W64[7.1]: filter_dirty_since_search = True เฉพาะเมื่อ fill/select สำเร็จรอบนี้ ไม่งั้น False
                # W_dropdown_sets_filter_dirty (OrangeHRM 2026-08-26): custom dropdown ถูกบังคับใช้ click
                # (W50 + state_filter.check_select_target_is_native) เงื่อนไขเดิมไม่เคยจริง guard ตายบน SPA
                # — นับ click ที่ ActionResult.dropdown_option_selected ยืนยันด้วย
                if result.success and (
                    tool_input.get("type") in ("fill", "select") or result.dropdown_option_selected
                ):
                    filter_dirty_since_search = True
                    last_filter_field_label = action_label
                    filter_changed_without_search = True
                else:
                    filter_dirty_since_search = False

                # W_delete_all_intent: ธง sticky ล้างได้ทางเดียว — กด Search สำเร็จ (_SEARCH_LABEL_RE)
                # ตอบคำถาม "ทั้ง task เคยกรองจริงหรือยัง"
                if result.success and action_label and _SEARCH_LABEL_RE.search(action_label):
                    filter_changed_without_search = False

                # W64[7.2]: task นี้เคยมี toast ยืนยันสำเร็จ (sticky — เป็นความจริงตลอดไป)
                if result.toast_confirmed:
                    any_toast_confirmed_this_task = True
                # W_count_answer_check: ดูดตัวเลขที่ _deterministic_count_note() นับไว้ออกจาก
                # ข้อความผลลัพธ์ (read_page_data เท่านั้นที่มีบรรทัดนั้น — action อื่นคืน {})
                if result.success:
                    system_counted.update(system_counted_conditions(result.message))
                # W30: หน้าเปลี่ยนเองหลัง action ที่ไม่ตั้งใจ navigate (redirect/JS ซ่อน) — โมเดลเคยตัดสินจาก
                # state เก่า แจ้งตรงๆ แม้ success=True (สำเร็จแต่พาไปหน้าอื่นอันตรายกว่า fail)
                # ไม่เช็คกับ goto/switch_tab/go_back
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
                # W10[D]: แนบ label จริงของเป้าหมายเข้า history/event ให้ Log panel โชว์ชื่อแทน index
                # (ไม่กระทบ dispatch)
                # W_timing_gap: ถือ reference dict ไว้เติมเวลา wait_stable ทีหลัง (ShortTermMemory เก็บ
                # object เดียวกัน) ย้ายการบันทึกไปทีหลังไม่ได้เพราะ guard ด้านล่างอ่าน memory ของ step นี้
                step_record = {
                    "step": steps_taken,
                    "cmd": tool_input,
                    "label": action_label,
                    # W_click_navigated: เก็บ result_text (มีโน้ต W30) ไม่ใช่ str(result) — failed_actions_summary()
                    # ยัด "[FAIL] click(3)" เข้า prompt ทุก step ที่เหลือ ตัดโน้ตทิ้ง = memory poisoning
                    "result": result_text,
                    "success": result.success,
                    "tokens": _tokens_dict(usage),
                    # W_procmem: locator ที่อยู่รอดข้าม task (dom_locator.py) — None ยกเว้น
                    # click/fill/select/check ที่สำเร็จ ใช้กับ llm.abstract_trajectory()
                    "locator_descriptor": result.locator_descriptor,
                    # W_step_trace: ข้อมูลวินิจฉัยล้วน ไม่ใช้ตัดสินใจในลูป (task_manager เขียนลง step_trace)
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
                    # W49: token สะสมให้ frontend โชว์สดระหว่าง running
                    "tokens": _tokens_dict(total_usage),
                    "llm_calls": llm_turns,
                })

                # ────────────────────────────────────────────────────────────
                # (ฎ) ความคืบหน้าแผน: plan_cursor (ของจริง) + display cursor (แสดงผล)
                # ────────────────────────────────────────────────────────────
                # W43: completed_plan_step -> SSE ให้ frontend ติ๊ก step แบบ real-time เฉพาะ execute()
                # สำเร็จและ task มีแผนจริง
                completed_plan_step = tool_input.get("completed_plan_step")
                # W_plan_step_cursor: เลขที่โมเดลรายงาน = หนึ่ง step จบ ไม่ใช่ทุกข้อถึงเลขนั้น — cursor
                # เดินทีละ 1 ไม่ถอยหลัง SSE ส่ง cursor จริง
                # action type ใช้ชุดเดียวกับ W_goal_scope_false_success / W_plan_progress_stall (ไม่ใช่
                # read-only) — มีนิยาม "mutating" 2 ชุดแล้ว อย่าสร้างชุดที่สาม
                plan_progressing_action = (
                    bool(plan_text)
                    and result.success
                    and tool_input.get("type") not in _GOAL_SCOPE_ALLOWED_ACTION_TYPES
                )
                # W_plan_cursor_needs_a_matching_action (2026-09-03 แผนติ๊กครบ 5 ข้อขณะยังลบไม่เสร็จ):
                # action ไม่ตรง step ชัดเจน (False) ห้ามเดิน cursor None = ตัดสินไม่ได้ ปล่อยผ่าน
                # (สัญญาณอ่อนโดยเจตนา)
                _cursor_step_text = ""
                if plan_text:
                    _cursor_steps = _plan_step_lines(plan_text)
                    if 0 < plan_cursor <= len(_cursor_steps):
                        _cursor_step_text = _cursor_steps[plan_cursor - 1]
                _cursor_action_matches = _action_matches_plan_step(
                    _cursor_step_text, action_label or "", str(tool_input.get("type") or ""),
                ) if _cursor_step_text else None

                # W_plan_cursor_needs_a_matching_action รอบสอง (2026-09-03): บล็อกทุกครั้ง = ล็อกตาย
                # (แผนค้างข้อ 1 ขณะ log ถึง step 13) — ใช้สัญญาณอ่อนเป็นตัวหน่วง บล็อกไม่เกิน
                # _MAX_BLOCKED_CURSOR_ADVANCES ครั้งติดกันแล้วปล่อยผ่าน
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

                # W_plan_panel_lags_the_log: ขยับ cursor แสดงผลแล้วยิงทุก action (จังหวะเดียวกับ event
                # "step") พา done_through ไปด้วยให้ tab ที่เชื่อมกลางคันกู้ติ๊กได้ (ไม่มี replay buffer —
                # routes.py::_stream_task_events) ค่าใช้จ่ายอยู่บน SSE ไม่เข้า prompt
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

                # W_action_matches_plan_step: เทียบกับ step ปัจจุบัน *หลัง* cursor ขยับ reset เฉพาะตอนตรง
                if plan_progressing_action:
                    # cursor เลยข้อสุดท้ายได้ — ตรึงไว้ที่ข้อสุดท้าย ไม่งั้น guard เงียบตอนที่ต้องการที่สุด
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
                    # W_goal_scope: step สุดท้ายเสร็จ = goal สำเร็จ (sticky) ผูก result.success
                    # W_goal_scope_false_success (live run 2026-08-27): read_page_data ที่ "สำเร็จ" ไม่พิสูจน์อะไร
                    # (completed_plan_step=5 มากับ read_page_data แล้ว gate หยุด task) — ใช้ชุด action ของ gate
                    # W_plan_step_cursor: ผูกกับ cursor ของโค้ด ไม่ใช่เลขดิบจากโมเดล
                    if plan_cursor > _total_plan_steps(plan_text):
                        plan_fully_completed = True

                # ────────────────────────────────────────────────────────────
                # (ฏ) จบ task ทันที (human deny / validation error) + vision fallback + โน้ตแผน
                # ────────────────────────────────────────────────────────────
                # W10[F]: human กด Deny (หรือ timeout = ยังไม่อนุมัติ) -> จบ task ทันที ไม่วนให้ลองทางอื่น
                # (ปฏิเสธคือ "หยุด")
                if not result.success and result.message == REJECTED_BY_USER_MESSAGE:
                    success = False
                    final_message = (
                        f"หยุด task ทันที: ผู้ใช้ปฏิเสธ action นี้ ({tool_input}) — "
                        "ไม่ลองทำทางอื่นต่อตามหลัก human-in-the-loop (การปฏิเสธคือคำสั่งหยุด)"
                    )
                    if verbose:
                        print(f"[human-denied] {final_message}", flush=True)
                    break

                # W20 (Task12, Early Termination): หลัง fill หรือ click ปุ่มบันทึกที่สำเร็จ ถ้ามี validation
                # error ให้ STOP ทันที (ทำงานแม้ไม่เคยเรียก finish_task) ไม่ retry/refresh — ต้องเปลี่ยนค่า
                # เท่านั้น กรอง bare "* Required" แล้วส่งข้อความ error จริงให้ user
                # (2026-08-06) ข้อความชวน user ตอบค่าใหม่ — page/session ยังค้าง เทิร์นถัดไปเข้า
                # generate_plan() พร้อม previous_assistant_message (Context-Aware Implicit Execution)
                # แล้วกรอกค่าใหม่ในฟอร์มเดิม ไม่ต้องมี infrastructure ใหม่
                if result.success and _should_check_validation_error_after_action(
                    tool_input.get("type"), action_label,
                ):
                    validation_errors = [
                        e for e in await _scan_validation_errors(page, within_form=True)
                        if not _is_bare_required_message(e)
                        # W_required_error_is_not_a_dead_end: "ช่องนี้ต้องกรอก" agent เติมเองได้ ไม่ใช่ทางตัน
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

                # W9[A] vision fallback (Gemini เท่านั้น): action ที่พึ่ง visibility ล้มเหลวหลัง retry ครบ
                # ทั้งที่ index มีจริง — สงสัย overlay ที่ elementFromPoint พลาด (pointer-events: none)
                # ส่ง screenshot ให้ Gemini vision ป้อนเป็น context step ถัดไป ห้าม throw
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

                # ────────────────────────────────────────────────────────────
                # (ฐ) ส่งผลกลับ LLM -> compaction -> รอหน้านิ่ง + domain guard
                # ────────────────────────────────────────────────────────────
                messages = append_tool_result(messages, tool_use_id, result_text)

                # W7[A]/W22: เก็บ boundary แล้วเช็คว่าต้อง compact ไหม (compact_messages ตาม provider)
                step_boundaries.append((steps_taken, len(messages)))
                if len(step_boundaries) > _COMPACT_AFTER_STEPS:
                    cut_list_index = len(step_boundaries) - _KEEP_RECENT_STEPS
                    cut_step_num, cut_at = step_boundaries[cut_list_index - 1]
                    # W50: สรุปเฉพาะ delta (digest_upto_step+1..cut_step_num) cap _MAX_DIGEST_LINES —
                    # เป็น candidate ก่อน commit เมื่อ compact_messages() ไม่ no-op (ไม่งั้นเสีย step ที่สรุปฟรี)
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
                    # W22: ใช้ความยาวจริงก่อน/หลัง — compact อาจ no-op หรือลดน้อยกว่า cut_at (Groq กัน
                    # system ที่ index 0) สมมติผิดทำ boundary เพี้ยนสะสม
                    removed = len_before_compact - len(messages)
                    if removed > 0:
                        # W50: commit candidate digest เข้า state สะสมจริง เฉพาะตอนที่
                        # compact_messages() แทรก digest เข้า messages สำเร็จจริงเท่านั้น
                        digest_lines = candidate_lines
                        digest_upto_step = cut_step_num
                        step_boundaries = [
                            (s, b - removed) for s, b in step_boundaries[cut_list_index:]
                        ]
                        # W_token_trim (P3/M3): compaction spliced out the full manual — re-send it next
                        # step ([PRE_LEARNED_MANUAL] strict mode must reach the model once per window)
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

                    # domain guard: classify_action() เช็ค allowlist เฉพาะ goto — click ที่พาออกนอกโดเมน
                    # (OAuth/โฆษณา/redirect) จับไม่ได้เพราะ perception ไม่เก็บ href เช็คหลัง action แล้วดึงกลับ
                    # ก่อน perceive ครอบทุกเส้นทางตั้งแต่ W_domain_guard_default (None เมื่อ url ผิดรูป)
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

            # ────────────────────────────────────────────────────────────
            # [run_task 10] หลังลูป: long-term/procedural memory + persona + คืนผล
            # ────────────────────────────────────────────────────────────
            # W7[A]: บันทึกผล task ให้ recall() ของ task ถัดไป ครั้งเดียวตอนจบ loop (ทุก path ยกเว้น
            # confirm_plan declined) W41: background task ไม่ await
            _fire_and_forget(asyncio.to_thread(
                long_term_memory.record_task,
                url=url, goal=goal, success=success, message=final_message,
                failed_actions=self.memory.failed_actions_summary(),
                session_id=session_id or "",
            ))

            # W_procmem: กลั่น trajectory ที่สำเร็จเป็น template (procedural_memory.save_template) แบบ
            # background — enable_procedural_memory_capture คือฝั่งเขียน แยกจาก enable_procedural_memory
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

            # W19-3 (enable_persona_voice): แปลง final_message เป็นไทยธรรมชาติตอนจบ task — additive
            # เพิ่ม persona_message/persona_status ไม่แก้ key เดิม
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
                # W_retry_value_has_no_home: ทำให้คำสัญญา "ตอบค่าใหม่แล้วกรอกช่องเดิมให้" เป็นจริง —
                # routes.py จำกับ session แล้วแปลงค่าเปล่าๆ ของเทิร์นถัดไปเป็นคำสั่งที่ระบุช่อง
                "retry_value_field_labels": retry_value_field_labels,
                # W_auto_login_outcome_is_invisible: "skipped" | "ok" | "failed"
                "auto_login": auto_login_outcome,
                # W_index_drift_measure: element ที่ index ชี้เปลี่ยนตัว/หายไประหว่าง
                # snapshot กับ dispatch กี่ครั้ง
                "index_drift_changed": index_drift_changed,
                "index_drift_gone": index_drift_gone,
            }
        # ────────────────────────────────────────────────────────────
        # [run_task 11] error ที่ไม่คาดคิด (คืนงานที่ทำแล้วครบ) + finally ปิด/คืน browser
        # ────────────────────────────────────────────────────────────
        except Exception as e:
            # W_loop_crash: เดิม try มีแต่ finally — exception (TargetClosedError, "Execution context was
            # destroyed", OAuthLoginRequired, Gemini ResourceExhausted) ทะลุออกไป history/token/
            # final_page_state/record_task หายหมด ผู้ใช้เห็นแค่ข้อความ Python ดิบ
            # จับแล้วรายงานตามจริง (success=False + step ที่พัง) คืน dict ครบทุก field ไม่ทำ step ต่อ
            # (ถึงตรงนี้ส่วนใหญ่ browser ตายแล้ว action error ปกติเป็น ActionResult(False) อยู่แล้ว)
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
                # page= จาก session registry — ผู้เรียกคุม lifecycle ข้าม call ห้ามปิด/คืนอะไร
                pass
            elif connect_to_user_browser:
                # ห้ามปิดอะไรบน browser จริงของ user — เดิม page.close() tab ที่เปิดเอง เทิร์นถัดไปหา
                # tab เดิม (resolve_target_page) ไม่เจอ เปิด tab ใหม่ทุกครั้ง ปล่อย tab ไว้เสมอ
                # playwright.stop() แค่ตัด CDP ของ driver นี้ ไม่ใช่ปิด browser
                await playwright.stop()
            elif owns_browser:
                if not keep_browser_open:
                    await browser.close()
                    await playwright.stop()
                # keep_browser_open=True: ปล่อย browser/playwright ค้างโดยตั้งใจ (user ปิดเอง)
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
        """W_procmem: รัน procedural template ผ่าน fastpath_executor.execute_template() แทน
        perceive->plan->act ประหยัด LLM call ทุก step ที่ replay สำเร็จ

        v1 รองรับแค่ page= (session-managed ไม่ปิดอะไร) หรือ browser= (pool เปิด/ปิด context)
        โหมดอื่นผู้เรียกต้อง fallback เป็น run_task() เอง
        escalation: replay ล้มเกิน repair quota -> run_task() เต็มบน page เดียวกัน (W12 perceive ต่อ)
        navigate + _maybe_auto_login() ที่นี่ก่อนส่งต่อ (ลำดับเดียวกับ run_task) — เดิม
        execute_template() ไม่เคย auto-login domain ที่ต้อง login จึง replay ไม่ได้"""
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

"""
actions.py  —  W3: Browser Actions (ชุดเครื่องมือให้ Agent สั่งงาน)
------------------------------------------------------------------
หน้าที่: ห่อการกระทำบน browser ให้เป็น "action มาตรฐาน" ที่ Agent Loop (W4)
         เรียกใช้ได้ด้วย index ที่ได้จาก perception.get_snapshot()

ออกแบบให้ทุก action:
  1. รับ index (หรือ params) เดียวกันกับที่ LLM เห็นตอน perceive
  2. คืน ActionResult (success/ข้อความ) เสมอ
  3. ไม่ throw ดิบๆ ออกไป -> จับ error แล้วรายงานกลับแทน (agent จะได้ไม่ตาย)

W5: execute() retry click/fill/select/check ให้เองในนี้ (_dispatch_with_retry) ก่อน
ส่งผลลัพธ์กลับ orchestrator — กัน false negative จาก DOM ที่ยังไม่นิ่ง โดยไม่เสีย
LLM token สักรอบเดียว

W19: execute() เช็ค core/state_filter.py ก่อน dispatch จริงสำหรับ click/fill/check/scroll
— action ที่ "สถานะเป้าหมายบรรลุอยู่แล้ว" (fill ข้อความเดิมซ้ำ/check ที่ติ๊กอยู่แล้ว/scroll
ที่สุดหน้าแล้ว) หรือ "ทำไม่ได้แน่นอน" (click element disabled) จะไม่แตะ browser เลย
short-circuit กลับ ActionResult ทันที — deterministic ล้วนๆ ไม่พึ่ง LLM

ใช้คู่กับ perception.py (ไฟล์เดียวกับ W2)

W40: click/fill/select_option/check เดิม dispatch ผ่าน page.click()/page.fill()/
page.select_option()/page.check() ตรงๆ เสมอ — ถ้า element ที่ index ชี้ไปอยู่ใน <iframe>
(perception.py::get_snapshot() ตอนนี้ perceive เห็นแล้วตั้งแต่ W40 ฝั่งนั้น) จะหา element
ไม่เจอเลย เพราะ page.click(selector) query แค่ document หลัก ข้าม frame boundary ไม่ได้ —
เพิ่มการเรียก perception.resolve_frame() ก่อน dispatch จริงทุกจุด หา Frame object ที่ถูก
ต้องแล้วเรียก .click()/.fill()/.select_option()/.check() กับ target นั้นแทน page ตรงๆ

W42: select_option() เดิมเทียบ label ที่ LLM ส่งมากับ option ใน DOM แบบ exact string ผ่าน
Playwright ตรงๆ (select_option(selector, label=label)) — พังกับเว็บที่ option text ใช้
non-breaking space (U+00A0, &nbsp; ใน HTML) คั่นคำแทน space ปกติ (พบจริงบน
uitestingplayground.com/select — ตั้งใจทำมาเทสต์ automation tool โดยเฉพาะ) LLM ส่ง label
มาด้วย space ปกติเสมอ (เห็นจาก perception.py::get_snapshot() ที่ normalizeข้อความผ่าน
JS .innerText/label ธรรมดา) ไม่มีทาง match ได้เลยไม่ว่า retry กี่ครั้ง (_dispatch_with_retry
เดิม — ไม่ได้แตะ) เพราะเป็น deterministic mismatch ไม่ใช่ timing issue — แก้ด้วยการดึง option
text จริงจาก DOM มาก่อน (target.locator(selector).locator("option").all_text_contents())
แล้วเทียบแบบ normalize whitespace ทั้งสองฝั่ง (ดู _normalize_option_text) เจอ match แล้วค่อย
เรียก select_option ด้วย text จริงจาก DOM (ไม่ใช่ label ดิบจาก LLM) กัน mismatch แบบอื่นที่
อาจเจอเว็บอื่นด้วย (trailing space, case ต่าง ฯลฯ) — หา match ไม่เจอเลยแม้ normalize แล้ว
คืน [FAIL] พร้อม list ตัวเลือกจริงที่มีอยู่แนบไปด้วย (debug ง่ายกว่า timeout เฉยๆ แบบเดิม)
"""

import asyncio
import json
import re
from dataclasses import dataclass, replace
from typing import Awaitable, Callable, Optional, Union
from playwright.async_api import Frame, Page, TimeoutError as PWTimeout

from backend.app.core import state_filter
from backend.app.core.dom_locator import compute_locator_descriptor
from backend.app.core.perception import count_elements, extract_table_data, resolve_frame
from backend.app.permission.rules import (
    DEFAULT_NEEDS_CONFIRMATION, ActionRisk, classify_action, extract_domain, install_ssrf_guard,
)
# W65[3] ("Vault Expansion"): site_learning.storage ไม่ได้ import ตรงนี้ระดับ module — ยืนยัน
# แล้วว่าจะเกิด circular import จริง (site_learning/__init__.py -> crawler.py ->
# orchestrator.py -> fastpath_executor.py -> กลับมา actions.py ที่ยังโหลดไม่เสร็จ) ต้อง lazy
# import ข้างในฟังก์ชันแทน (ดู fill_secret() ด้านล่าง) — pattern เดียวกับที่ orchestrator.py::
# _maybe_auto_login() ใช้อยู่แล้วสำหรับปัญหาเดียวกันเป๊ะ

# ask_user_func: callback ให้ orchestrator/UI ชั้นบนตัดสินใจแทน blocking input()
# เช่น API server (W10) จะ inject callback ที่ส่ง event ไป UI แล้วรอ user กดยืนยันจริง
# แทนที่จะพึ่ง terminal input() ตรงๆ — รับ cmd dict คืน bool (True = อนุญาต)
AskUserFunc = Callable[[dict], Awaitable[bool]]


# ------------------------------------------------------------
# ผลลัพธ์มาตรฐานของทุก action
# ------------------------------------------------------------
@dataclass
class ActionResult:
    success: bool
    action: str
    message: str = ""
    # W_procmem: locator ที่ "อยู่รอด" ข้าม task run ได้ (ดู core/dom_locator.py) —
    # คำนวณเฉพาะตอน action สำเร็จจริงใน click/fill/select_option/check เท่านั้น (ดูจุด
    # เรียก compute_locator_descriptor() ในแต่ละฟังก์ชันด้านล่าง) None สำหรับ action อื่น
    # ทั้งหมด (goto/scroll/go_back/switch_tab/wait_stable/read_page_data — ไม่มี element
    # เดี่ยวๆ ให้ abstract) และ None ตอน action ล้มเหลวด้วย (ไม่มีอะไรให้อธิบาย element
    # ที่ยังไม่เกิดผลจริง) — default None ท้ายสุดไม่กระทบ call site เดิมที่สร้าง
    # ActionResult(success, action, message) แบบ positional 3 ตัวเลยสักที่เดียว
    locator_descriptor: Optional[dict] = None
    # W64[7.2] ("Add-Action Idempotency Lock" — ticket Issue 7.2): True เฉพาะตอน
    # _detect_success_toast() (ดู _dispatch_click_with_retry ด้านล่าง) เจอ toast/ข้อความ
    # ยืนยันสำเร็จจริงหลัง click ที่ label เป็น Save/Submit/Confirm — สัญญาณที่ตรวจสอบได้จริง
    # (ไม่ใช่แค่คำอธิบายของ LLM) ว่า "ข้อมูลถูกบันทึกจริงแล้ว" ให้ orchestrator.py ใช้ตัดสินใจ
    # ว่า finish_task guard (ดู _scan_created_item_in_table) ควร "เชื่อ" ว่างานสำเร็จแล้วแม้
    # ตรวจไม่เจอใน table body ภายหลัง (อาจเป็นปัญหา search/filter/pagination ไม่ใช่ว่าไม่ได้
    # ถูกสร้างจริง) แทนที่จะบังคับ VERIFICATION_FAILED เหมือนตอนไม่มีหลักฐานยืนยันเลย — string
    # matching ข้อความ toast โดยตรงเปราะบางกว่า (ต้องคง format ให้ตรงกันข้าม 2 ไฟล์) จึงใช้
    # field ที่ type-checked แทน (pattern เดียวกับ locator_descriptor ด้านบน)
    toast_confirmed: bool = False
    # W_dropdown_sets_filter_dirty: True เฉพาะตอน click นี้คือการ "เลือกตัวเลือกใน custom
    # dropdown ที่เปิดอยู่" จริง (ตรวจจาก DOM ก่อน dispatch — ดู state_filter.
    # classify_click_index_disturbance) — orchestrator ใช้ยกธง filter_dirty_since_search
    # เหมือนที่ fill/select ยกอยู่แล้ว เพราะ W50 + check_select_target_is_native() บังคับให้
    # โมเดลใช้ "click" กับ custom dropdown ทุกกรณี ธงจึงไม่เคยถูกยกเลยบนเว็บ SPA สมัยใหม่
    # (= เว็บแทบทั้งหมด) ทำให้ guard "ห้ามคลิก row action ก่อนกด Search" ตายสนิทในเคสที่
    # ต้องการมันที่สุด — ส่งสัญญาณผ่าน field ที่ type-checked แทน string matching ข้อความ
    # ผลลัพธ์ (pattern เดียวกับ toast_confirmed/locator_descriptor ด้านบน)
    dropdown_option_selected: bool = False

    def __str__(self):
        mark = "OK" if self.success else "FAIL"
        return f"[{mark}] {self.action} -> {self.message}"


# selector ที่ผูกกับ index ที่ perception ติดไว้บน element
def _sel(index: int) -> str:
    return f'[data-ai-index="{index}"]'


def _normalize_option_text(text: str) -> str:
    """W42: แทน non-breaking space (U+00A0 — &nbsp; ใน HTML) เป็น space ปกติ แล้วยุบ
    whitespace ที่เหลือ (ซ้ำ/tab/newline) ให้เป็น space เดียว + ตัดช่องว่างหัวท้ายทิ้ง — ใช้
    เทียบ label ที่ LLM ส่งมากับ option text จริงจาก DOM ใน select_option() ด้านล่าง กัน
    mismatch จาก whitespace ล้วนๆ (nbsp/trailing space/ซ้ำ) โดยไม่กระทบการเทียบเนื้อหาจริง"""
    return re.sub(r"\s+", " ", (text or "").replace(" ", " ")).strip()


# W5: timeout สั้นลงสำหรับ action ที่ต้องหา/รอ element (click/fill/select/check) — ถ้า
# element หาไม่เจอหรือมองไม่เห็น (not visible/not actionable) ภายในเวลานี้ ให้ดีด [FAIL]
# กลับเข้า loop หลักทันที ไม่รอค้างนาน โดยเฉพาะตอนรวมกับ _dispatch_with_retry ด้านล่างที่
# ยิงซ้ำอยู่แล้ว (5s เดิม x 3 ครั้ง = รอได้ถึง 15s ต่อ 1 action เดียว นานเกินไป)
_ELEMENT_ACTION_TIMEOUT_MS = 3000
# W_check_evaluate_timeout: ชั้น fallback ของ check() (force click / JS click) ไม่ได้รอ
# actionability เหมือนชั้นแรก — force=True และ el.click() ข้ามการรอนั้นไปเลย เวลาที่ตั้งไว้จึง
# ครอบแค่ "หา element เจอไหม" ซึ่งชั้นแรกเพิ่งพิสูจน์ไปแล้วว่าเจอ (มันล้มเพราะ Playwright
# ไม่ยอมรับว่า element นี้ checkable ไม่ใช่เพราะหาไม่เจอ) — 3 วินาทีต่อชั้นจึงเป็นการรอเปล่า
# ที่คูณด้วย _ACTION_RETRIES อีกรอบ
_ELEMENT_FALLBACK_TIMEOUT_MS = 1000
# อ่านสถานะ DOM ล้วนๆ หลังคลิก — ค่าเดียวกับ state_filter._STATE_CHECK_TIMEOUT_MS ด้วยเหตุผล
# เดียวกัน (เช็คก่อน/หลัง dispatch ทุก step ต้องเร็วที่สุด)
_STATE_READ_TIMEOUT_MS = 500


# ------------------------------------------------------------
# W5: Verify + Retry — action พวก click/fill/select/check พังบ่อยเพราะ DOM ยัง
# ไม่นิ่ง (element ยัง render/animate ไม่เสร็จ) ไม่ใช่เพราะ index ผิดจริงๆ เสมอไป
# retry เงียบๆ ระดับนี้ก่อน ไม่เสีย token เพราะไม่ต้องถาม LLM จนกว่าจะลองครบ —
# ถ้ายัง fail อยู่หลัง retry ครบ ค่อยส่งกลับให้ LLM ตัดสินใจเหมือน W4 เดิม
# ------------------------------------------------------------
# W_retry_never_paid_off: วัดจาก data/step_trace.jsonl ทั้งไฟล์ (422 แถว, 2026-09-03) —
# action ที่สำเร็จในรอบ retry ที่ 2 หรือ 3 = 0 ครั้ง ส่วนที่เผาครบทุกรอบแล้วล้มเหลว = 29 ครั้ง
# (click 15, fill 10, select 2, press_key 1, check 1) รอบที่สามจึงยังไม่เคยกู้อะไรได้เลยใน
# ประวัติที่บันทึกไว้ แต่คูณเวลาหางของทุก action ที่ล้มเหลว — เหลือการลองซ้ำอีกหนึ่งรอบไว้
# สำหรับหน้าเว็บที่ render ไม่ทันจริงๆ ซึ่งเป็นเหตุผลตั้งต้นของ retry
_ACTION_RETRIES = 2  # ครั้งแรก + retry อีก 1 ครั้ง
_ACTION_RETRY_DELAY_SEC = 0.5


async def _dispatch_with_retry(action_func, *args) -> ActionResult:
    """เรียก action_func(*args) สูงสุด _ACTION_RETRIES ครั้ง คั่นด้วย delay สั้นๆ ถ้า fail
    คืนผลลัพธ์แรกที่สำเร็จทันที หรือผลลัพธ์ของความพยายามครั้งสุดท้ายถ้าไม่สำเร็จเลย —
    แนบจำนวนครั้งที่ลองไว้ใน message ด้วย เผื่อ debug ว่า action นี้ flaky แค่ไหน"""
    result: ActionResult = None
    for attempt in range(1, _ACTION_RETRIES + 1):
        result = await action_func(*args)
        if result.success:
            if attempt > 1:
                result = ActionResult(
                    True, result.action, f"{result.message} (attempt {attempt}/{_ACTION_RETRIES})"
                )
            return result
        if attempt < _ACTION_RETRIES:
            await asyncio.sleep(_ACTION_RETRY_DELAY_SEC)
    return ActionResult(False, result.action, f"{result.message} (after {_ACTION_RETRIES} attempts)")


# ------------------------------------------------------------
# ACTIONS
# ------------------------------------------------------------

async def click(page: Page, index: int, timeout: int = _ELEMENT_ACTION_TIMEOUT_MS) -> ActionResult:
    """คลิก element ตาม index"""
    try:
        selector = _sel(index)
        target = await resolve_frame(page, selector)
        await target.click(selector, timeout=timeout)
        descriptor = await compute_locator_descriptor(target, selector)
        return ActionResult(True, f"click({index})", "click succeeded", locator_descriptor=descriptor)
    except PWTimeout:
        return ActionResult(False, f"click({index})", "element not found / not clickable (timeout)")
    except Exception as e:
        return ActionResult(False, f"click({index})", f"error: {e}")


# W47: hover-to-reveal action buttons (เช่น flag icon ที่โผล่มาตอน hover แถวแม่ในอีเมล
# client ทั่วไป — เจอจริงบน uitestingplayground.com/scrolltoclick Case 4) มี
# data-ai-index แล้วจาก perception.py (แก้ไปแล้วให้ไม่กรอง element ที่ซ่อนด้วย
# opacity:0/visibility:hidden ของตัวเองทิ้ง ตราบใดที่ bounding box ไม่เป็น 0x0) แต่คลิก
# ตรงๆ รอบแรกจะพลาดเสมอเพราะ CSS ยังไม่เปลี่ยนสถานะจาก hover จริง — Playwright's
# locator.click() เองไม่ trigger :hover ของ ancestor ให้ก่อนอัตโนมัติ (มันเล็ง element
# เป้าหมายแล้วคลิกตรงจุดกึ่งกลางทันที ไม่ได้จำลอง mouse move ผ่าน ancestor เหมือนคนจริง)
async def hover(page: Page, index: int, timeout: int = _ELEMENT_ACTION_TIMEOUT_MS) -> ActionResult:
    """เลื่อนเมาส์ไปวางไว้บน element ตาม index (ไม่คลิก) — trigger CSS :hover ของ element
    เอง/บรรพบุรุษทั้งสาย (ไม่ต้องหา parent row เอง แค่ hover ตัว element เป้าหมายตรงๆ ก็พอ
    เพราะ browser จะ trigger :hover ของ ancestor ทั้งสายตามธรรมชาติของการขยับเมาส์จริง)

    force=True จำเป็นมาก — พิสูจน์จริงบน uitestingplayground.com/scrolltoclick Case 4 ว่า
    ถ้าไม่ใส่ .hover() ธรรมดาจะ timeout เหมือน click() ทุกประการ เพราะ Playwright เองมี
    actionability check ภายใน (รอ element "visible" ตามนิยามของ Playwright เอง) ก่อนจะ
    ยอมส่ง mouse event เข้าไปจริง — element ที่ visibility:hidden (เจอจริงบนเว็บนี้ ต่างจาก
    opacity:0 ที่ Playwright ยัง "visible" ปกติเพราะไม่เช็ค opacity) ไม่ผ่านเงื่อนไขนี้เลย
    ไม่ว่าจะ retry กี่ครั้ง — force=True ข้าม actionability check พวกนี้ทั้งหมด ส่ง mouse
    move ไปที่พิกัดกึ่งกลางของ element ตรงๆ (ไปโดน parent ที่ visible จริงแทนถ้าตัว element
    เองไม่ hit-testable) trigger :hover ของ ancestor ได้จริงเหมือนเมาส์ขยับจริง แล้วค่อยให้
    click() รอบถัดไปเจอ element ที่ visibility เปลี่ยนเป็น visible แล้วสำเร็จตามปกติ"""
    try:
        selector = _sel(index)
        target = await resolve_frame(page, selector)
        await target.hover(selector, timeout=timeout, force=True)
        return ActionResult(True, f"hover({index})", "hover succeeded")
    except PWTimeout:
        return ActionResult(False, f"hover({index})", "element not found / not hoverable (timeout)")
    except Exception as e:
        return ActionResult(False, f"hover({index})", f"error: {e}")


# W63[7.1] ("Save Confirmation & Toast Wait" — ticket Issue 7.1): เดิมมีแค่คำแนะนำใน
# SYSTEM_PROMPT (W19 "Task Completion Verifier") ให้ LLM "มองหา" toast เองตอนจะเรียก
# finish_task เท่านั้น ไม่มีการเช็คระดับโค้ดเลย — ปัญหาคือ toast มักเป็น element ชั่วคราว
# (auto-dismiss ไม่กี่วินาที) ถ้าไปเช็คตอน finish_task (ซึ่งอาจเกิดขึ้นหลาย step ถัดมา หลัง
# LLM ทำ action อื่นต่อไปแล้ว) toast อาจหายไปแล้วจริงๆ ทั้งที่ save สำเร็จ ทำให้เช็คตอนนั้นไม่
# น่าเชื่อถือ — ย้ายจุดเช็คมาไว้ทันทีหลัง click ที่ label บ่งบอกว่าเป็นปุ่ม Save/Submit/Confirm
# (จุดเดียวกับที่ _dispatch_click_with_retry() เช็ค confirmation modal อยู่แล้วด้านล่าง) รอ
# สั้นๆ (bounded, ไม่ throw ถ้าไม่เจอ) แล้วแนบผลลัพธ์ต่อท้าย message ให้ LLM เห็นทันทีว่า toast
# ปรากฏจริงไหม แทนที่จะฝากความหวังไว้กับการเดาของ LLM เองล้วนๆ (defense-in-depth เหมือน
# pattern อื่นในไฟล์นี้ เช่น modal auto-resolve ด้านล่าง)
_SAVE_LABEL_RE = re.compile(r"\b(save|submit|confirm|update)\b|บันทึก|ยืนยัน|อัปเดต|อัพเดท", re.IGNORECASE)

# เรียงจากเจาะจงที่สุด (OrangeHRM .oxd-toast) ไปกว้างสุด (ARIA live region/toast framework
# ทั่วไป) — ตั้งใจไม่ผูกกับ OrangeHRM เพียงเว็บเดียว เพราะ role="status"/role="alert" และ
# class ที่มีคำว่า toast/snackbar/notification เป็น pattern มาตรฐานที่ web framework ทั่วไปใช้
# ร่วมกันจริง (Material/Bootstrap/Ant Design ฯลฯ) ต่างจาก _RECORD_COUNT_SELECTORS ใน
# orchestrator.py ที่ข้อความ "Records Found" ไม่ใช่ pattern ที่เว็บอื่นใช้ร่วมกันเลย
_SUCCESS_TOAST_SELECTOR = (
    '.oxd-toast--success, .oxd-toast-container, '
    '[role="status"]:not(:empty), [role="alert"]:not(:empty), '
    '[class*="toast" i]:not(:empty), [class*="snackbar" i]:not(:empty), '
    '[class*="notification" i][class*="success" i]:not(:empty)'
)

_TOAST_WAIT_TIMEOUT_MS = 2500


async def _detect_success_toast(page: Page) -> Optional[str]:
    """W63[7.1]: เช็คว่ามี success toast/confirmation message โผล่ขึ้นมาจริงหลัง action นี้
    ไหม — คืนข้อความที่เจอ (ตัดสั้นๆ ไม่เกิน 200 ตัวอักษร) หรือ None ถ้าไม่เจอ/เช็คไม่ได้
    (ไม่ throw ให้ click ที่เพิ่ง success พัง — หลักการเดียวกับ _detect_confirmation_modal
    ด้านล่าง) รอสั้นๆ (_TOAST_WAIT_TIMEOUT_MS) ให้ animation/network เข้ามาแสดงผลก่อนถ้ายังไม่
    เจอทันที เพราะ toast มักปรากฏหลัง response กลับมาไม่กี่ร้อย ms ไม่ใช่ทันทีที่คลิก

    W_toast_container_is_empty (วัดกับหน้าจริง 2026-09-07): เดิมใช้ .first ซึ่งหยิบ element แรก
    ที่ match ตาม DOM order — บน OrangeHRM นั่นคือ ".oxd-toast-container" ซึ่งเป็น *กล่องครอบ*
    ที่มีอยู่ก่อนแล้วและยังว่างเปล่า โค้ดจึงอ่านข้อความได้ "" แล้วคืน None ทั้งที่ toast ขึ้นจริง
    วัดได้ว่าโผล่ที่ 156 ms และอยู่ถึง 3500 ms คือทันเวลาที่รออยู่ (2500 ms) สบายๆ
    ผลคือ action ที่บันทึกสำเร็จถูกรายงานว่า "No toast/success confirmation appeared" แล้ว agent
    ก็เผา step ไล่หาคำยืนยันที่ระบบมองข้ามไปเอง (เห็นในงาน add_candidate/login_checkout/
    rag_permission ของ release gate)
    -> รอ element ตัวแรกที่ *มีข้อความจริง* ไม่ใช่ตัวแรกที่ match selector"""
    try:
        handle = await page.wait_for_function(
            """(sel) => {
                for (const el of document.querySelectorAll(sel)) {
                    if (!el.getClientRects().length) continue;
                    const text = (el.innerText || '').trim();
                    if (text) return text;
                }
                return null;
            }""",
            arg=_SUCCESS_TOAST_SELECTOR,
            timeout=_TOAST_WAIT_TIMEOUT_MS,
        )
        text = (await handle.json_value() or "").strip()
        return text[:200] if text else None
    except Exception:
        return None


# W_click_navigated (บั๊กจริง live-reproduce บน OrangeHRM 2026-08-26 ผ่าน step trace: agent
# คลิก "Admin" สำเร็จจริง หน้าเปลี่ยนไปแล้ว แต่ log รายงาน [FAIL] "element not found /
# not clickable (timeout)" ทำให้รวนทั้ง run): _dispatch_click_with_retry() วน 3 รอบโดยเขียนทับ
# `result` ทุกรอบ ข้อความของรอบสุดท้ายจึงชนะเสมอ — พอ attempt 1 คลิกสำเร็จและ SPA router
# เปลี่ยนหน้า node เดิมหลุดจาก DOM แล้ว attempt 2/3 ไปหา [data-ai-index="N"] ที่ *ไม่มีทาง*
# มีอยู่บนหน้าใหม่ (perception.py ล้างและแปะ index ใหม่ทุก snapshot) จึง timeout แน่นอน 100%
# แล้วรายงานว่า FAIL ทั้งที่คลิกได้ผลจริง (เสียเวลา retry เปล่าอีก ~10 วินาทีด้วย)
#
# เป็นบั๊กคลาสเดียวกับที่ crawler.py W34 (_explore_buttons) เจอและแก้ไปแล้ว: "error ถือว่าเป็น
# error จริงก็ต่อเมื่อ URL ไม่เปลี่ยน" — ยก pattern นั้นมาใช้ แต่ *ห้าม* ใช้
# crawler._normalize_url() ตัวนั้นซ้ำเด็ดขาด เพราะมันตัด fragment (#...) ทิ้งโดยตั้งใจ (หน้าที่
# ของมันคือ dedup ตอน crawl: "URL ต่างกันแค่ #section ไม่ควรถือว่าเป็นคนละหน้า") ซึ่งเป็น
# semantics *ตรงข้าม* กับที่ตรงนี้ต้องการ — SPA จำนวนมากใช้ hash router (#/admin/users) หรือ
# เปลี่ยนแค่ query param ถ้าใช้ฟังก์ชันนั้นตรงๆ การ navigate แบบนั้นจะยังถูกมองว่า "URL ไม่
# เปลี่ยน" แล้วเป็น false FAIL เหมือนเดิมทุกประการ จึงเทียบ URL เต็ม (รวม query + fragment)
# normalize แค่ trailing slash เท่านั้น
def _normalize_click_url(url: str) -> str:
    """W_click_navigated: normalize เบาที่สุดเท่าที่จำเป็น — ตัดแค่ trailing slash ท้าย URL
    (https://x/a/ กับ https://x/a คือหน้าเดียวกันจริง) คง query + fragment ไว้ครบเสมอ

    รับค่าที่ไม่ใช่ str ได้ด้วย (คืน "" ไปเลย) — Page.url จริงเป็น str property เสมอ แต่ page
    ที่ถูก mock ในเทสต์คืน mock object ให้แทน ซึ่งไม่ควรทำให้ click พังทั้งฟังก์ชัน และการที่
    ทั้งก่อน/หลังคืน "" เท่ากันแปลว่า "ไม่ได้ navigate" ซึ่งเป็น default ที่ปลอดภัยอยู่แล้ว"""
    text = url.strip() if isinstance(url, str) else ""
    return text[:-1] if len(text) > 1 and text.endswith("/") else text


# W_click_navigated: SPA router บางตัว transition ช้ากว่า attempt แรกของ click (พบใน W34 ว่า
# navigate จริงมาดีเลย์ได้หลายวินาที) — เผื่อเวลา poll สั้นๆ อีกครั้งหลัง retry หมดโควตา ก่อน
# สรุปว่าคลิกไม่สำเร็จจริง ตั้งสั้นกว่า W34 (5 วินาที) มากเพราะคนละ budget: crawler รันแบบ
# offline ครั้งเดียวต่อเว็บ แต่ตรงนี้อยู่ใน agent loop ที่ user รออยู่จริง และ click ที่ล้มจริงๆ
# (index หลุด/element หาย) ก็เจอบ่อยพอๆ กัน — 2 วินาทีพอสำหรับ router transition ที่ค้างอยู่
_CLICK_NAV_POLL_ATTEMPTS = 10
_CLICK_NAV_POLL_INTERVAL_SEC = 0.2


async def _dom_signature(page: Page) -> Optional[int]:
    """W_click_navigated: ความยาวของ document.body.innerHTML — สัญญาณสำรองสำหรับ SPA ที่
    เปลี่ยนแค่ state ภายในโดยไม่แตะ URL เลย (วิธีเดียวกับที่ crawler.py::_wait_for_dom_stable
    ใช้อยู่แล้ว) คืน None ถ้าอ่านไม่ได้ (หน้าปิดไปแล้ว/execution context ถูกทำลายกลาง
    navigation) — ห้าม throw ออกไปทำให้ click พังเด็ดขาด"""
    try:
        value = await page.evaluate("document.body ? document.body.innerHTML.length : 0")
        return int(value)
    except Exception:
        return None


async def _dispatch_click_with_retry(page: Page, index: int, label: str = "") -> ActionResult:
    """เหมือน _dispatch_with_retry() ทั่วไป (ครั้งแรก + retry อีก _ACTION_RETRIES-1 ครั้ง)
    แต่เฉพาะ click(): ตั้งแต่รอบ retry ที่ 2 เป็นต้นไป hover() บน element เป้าหมายก่อนคลิก
    ซ้ำเสมอ 1 ครั้ง — แก้ปัญหาปุ่ม hover-to-reveal ที่ perception.py ติด index ให้แล้วแต่
    คลิกตรงๆ รอบแรกจะพลาดเพราะ CSS ยังไม่เปลี่ยนสถานะจาก hover จริง (ดู hover() ด้านบน)

    รอบแรกยังคลิกตรงๆ เหมือนเดิมทุกประการ ไม่ hover ก่อนเด็ดขาด — กัน overhead (เวลา +
    round-trip ไป Playwright เพิ่ม) กับปุ่มทั่วไปที่ไม่ต้อง hover เลยตั้งแต่แรก (ส่วนใหญ่
    ของ click ทั้งหมด) ผลของ hover() เองไม่ถูกนำมาตัดสิน success/fail ของรอบนั้น (แค่เป็น
    ขั้นเตรียมก่อนคลิก — hover ไม่เจอ/ล้มเหลวก็ปล่อยให้ click() ลองต่อแล้วรายงานผลจริงของ
    click() เอง ไม่ใช่ของ hover())

    W23 ("Confirmation Modal Handler"): จุดเดียวที่ click-family ทั้งหมด (plain "click" และ
    "submit"/"delete"/"purchase"/"pay" ที่ execute() dispatch ผ่านฟังก์ชันนี้เหมือนกันทุก
    ประการ — ดู execute() ด้านล่าง) วิ่งผ่านเสมอ เหมาะเป็นจุดเดียวที่จะเช็ค+resolve
    confirmation modal ("Are you Sure?") ที่อาจเพิ่งเปิดขึ้นมาจาก click นี้ ก่อนคืนผลลัพธ์
    กลับไปให้ LLM ตัดสินใจ action ถัดไป (ดู _detect_confirmation_modal()/
    resolve_confirmation_modal() ด้านล่าง สำหรับเหตุผลเต็ม)"""
    # W_click_navigated: อ่านสถานะ "ก่อนคลิก" ไว้ก่อนเสมอ ทั้ง URL และ DOM signature — ใช้
    # ตัดสินตอนท้ายว่า timeout ที่ได้เป็น failure จริงหรือแค่ผลข้างเคียงของ navigation ที่
    # สำเร็จไปแล้ว (ดู docstring ของ _normalize_click_url ด้านบนสำหรับบั๊กจริงเต็มๆ)
    url_before = _normalize_click_url(page.url)
    dom_before = await _dom_signature(page)
    result: ActionResult = None
    for attempt in range(1, _ACTION_RETRIES + 1):
        if attempt > 1:
            await hover(page, index)
        result = await click(page, index)
        if result.success:
            if attempt > 1:
                result = ActionResult(
                    True, result.action, f"{result.message} (attempt {attempt}/{_ACTION_RETRIES})"
                )
            if await _detect_confirmation_modal(page):
                modal_note = await resolve_confirmation_modal(page)
                if modal_note:
                    result = ActionResult(
                        result.success, result.action, f"{result.message}{modal_note}",
                        locator_descriptor=result.locator_descriptor,
                    )
            elif label and _SAVE_LABEL_RE.search(label):
                # W63[7.1]: ไม่เช็ค toast ถ้าเพิ่งเจอ confirmation modal ไปแล้วด้านบน (คนละ
                # flow กัน — modal คือปุ่ม Delete/Remove ที่ต้องยืนยันซ้ำ ไม่ใช่ปุ่ม Save) —
                # จำกัดเฉพาะ label ที่ตรงคำ Save/Submit/Confirm กัน overhead การรอ toast บน
                # click ทั่วไปที่ไม่เกี่ยวข้องเลย (เช่น navigation link)
                toast_text = await _detect_success_toast(page)
                toast_note = (
                    f' [Success confirmation found: \"{toast_text}"]' if toast_text
                    else " [No toast/success confirmation appeared within the time limit after the click — check for a validation error, or whether the page already navigated back to the list by itself, before treating it as successful]"
                )
                result = ActionResult(
                    result.success, result.action, f"{result.message}{toast_note}",
                    locator_descriptor=result.locator_descriptor, toast_confirmed=bool(toast_text),
                )
            return result
        # W_click_navigated: attempt นี้ล้มเหลว แต่ถ้า URL เปลี่ยนไปแล้ว = attempt ก่อนหน้า
        # คลิกโดนจริงและพาไปหน้าใหม่แล้ว — retry ต่อไม่มีทางสำเร็จได้เลย (index ชุดเดิมไม่มี
        # อยู่บนหน้าใหม่) ออกจากลูปทันที ประหยัดเวลารอ timeout ที่รู้ผลล่วงหน้าอยู่แล้ว
        if _normalize_click_url(page.url) != url_before:
            break
        if attempt < _ACTION_RETRIES:
            await asyncio.sleep(_ACTION_RETRY_DELAY_SEC)

    # W_click_navigated: หมดโควตา retry (หรือ break ออกมาเพราะ URL เปลี่ยนแล้ว) — เผื่อเวลา
    # ให้ SPA router ที่ transition มาช้าอีกครั้งก่อนสรุปว่าล้มเหลวจริง (W34 พบว่า navigate
    # จริงมาดีเลย์ได้หลายวินาทีหลัง retry ครบแล้ว) — poll เฉพาะตอนที่ URL ยังไม่เปลี่ยนเท่านั้น
    # ไม่หน่วงเพิ่มเลยในเคสที่รู้ผลแล้ว
    navigated = _normalize_click_url(page.url) != url_before
    if not navigated:
        for _ in range(_CLICK_NAV_POLL_ATTEMPTS):
            await asyncio.sleep(_CLICK_NAV_POLL_INTERVAL_SEC)
            if _normalize_click_url(page.url) != url_before:
                navigated = True
                break
    if navigated:
        # รายงานตามความจริงทั้งสองส่วน: Playwright บอกว่า timeout จริง *และ* หน้าเปลี่ยนไป
        # จริง — ไม่กลบข้อความเดิมทิ้ง (หลักการเดียวกับ W_click_native_select/W_confident_zero
        # ที่ผลลัพธ์ต้องสะท้อนสิ่งที่เกิดขึ้นจริง ไม่ใช่สิ่งที่โค้ดอยากให้เป็น)
        return ActionResult(
            True, result.action,
            f"the click reported a timeout, but the page navigated from {url_before} to "
            f"{_normalize_click_url(page.url)} — the click DID take effect. Read the new page's "
            f"indexed elements before deciding your next action (original error: {result.message})",
        )

    # W_click_navigated: สัญญาณสำรองสำหรับ SPA ที่เปลี่ยนแค่ state ภายในโดย URL คงเดิม —
    # *ไม่* พลิกเป็น success จากสัญญาณนี้เด็ดขาด เพราะ DOM อาจเปลี่ยนจาก toast/spinner/
    # lazy-load ที่ไม่เกี่ยวกับคลิกนี้เลย — ยังคืน fail ตามเดิม แต่แนบหลักฐานไปด้วยให้โมเดล
    # ตัดสินใจบนข้อมูลจริง แทนที่จะเข้าใจว่า "ไม่มีอะไรเกิดขึ้นเลย" แล้วคลิกซ้ำจนโดน loop guard
    dom_after = await _dom_signature(page)
    dom_note = ""
    if dom_before is not None and dom_after is not None and dom_after != dom_before:
        dom_note = (
            " [The page did not navigate, but its DOM did change after this click "
            f"({dom_before} -> {dom_after} characters) — this click may already have taken "
            "effect (a panel, dropdown or dialog may have opened). Look at the page's current "
            "indexed elements before repeating the same action]"
        )
    # W_modal_check_on_failure (P8/M4): จุดเช็ค modal ทั้งหมดอยู่ใต้ `if result.success:` —
    # พอ modal บล็อกจนคลิกไม่สำเร็จ ระบบก็ไม่มีทางรู้ว่ามี modal อยู่ กลายเป็น dead-end ที่
    # ป้อนตัวเอง: คลิกไม่ได้ -> ไม่เช็ค -> ไม่รู้ -> คลิกที่เดิมไม่ได้อีก
    # เจตนาจำกัดไว้แค่ "บอกความจริง" ไม่กดปุ่มยืนยันให้เอง — การกดปุ่มใน dialog ที่ agent
    # ไม่ได้เปิดเองคือการตัดสินใจทำลายข้อมูลโดยโค้ด ซึ่งเกินขอบเขตที่ตกลงกันไว้
    # pattern เดียวกับ W_click_native_select/W_confident_zero: รายงานตามความจริงให้โมเดล
    # ตัดสินใจบนข้อมูลจริง ดีกว่าปล่อยให้เข้าใจว่า "element หาไม่เจอ" แล้วไล่คลิกตัวอื่นต่อ
    blocked_note = ""
    if await _detect_confirmation_modal(page):
        blocked_note = (
            " [A dialog is open on top of the page and is blocking this element — nothing "
            "behind it can be clicked. Act on the dialog first: choose one of its own buttons "
            "(confirm or cancel) to close it, then continue.]"
        )
    return ActionResult(
        False,
        result.action,
        f"{result.message} (after {_ACTION_RETRIES} attempts){dom_note}{blocked_note}",
    )


# W23 ("Confirmation Modal Handler" — บั๊กจริงที่ user รายงาน): agent คลิก "Delete Selected"/
# "Remove" สำเร็จ เปิด confirmation modal ("Are you Sure?") ขึ้นมาจริง แต่แล้ว "ค้าง"/
# "หยุดนิ่ง" อยู่ตรงนั้น ไม่กดปุ่มยืนยัน ("Yes, Delete") ในโมดัลต่อ — สาเหตุ: modal เป็น
# element ใหม่ที่เพิ่ง render ขึ้นมาหลัง click แต่ agent loop ปกติต้องรอ LLM ตัดสินใจเรียก
# action ถัดไปเองก่อนถึงจะเห็น/กด (เสีย round-trip เต็มๆ ต่อโมดัลหนึ่งอัน) ถ้า LLM ตีความ
# index/label ผิด/ไม่รู้ว่าต้องกดต่อ จะค้างจริงๆ ตามที่ user รายงาน — แก้ด้วยการ resolve
# modal นี้ "อัตโนมัติในระดับโค้ด" ทันทีหลัง click สำเร็จ ไม่ต้องรอ LLM ตัดสินใจเรียก action
# แยกอีกรอบเลย (เหมือน pattern เดียวกับ auto-hover-on-retry ด้านบน — เติมเต็ม "สิ่งที่ควร
# เกิดขึ้นจริง" ด้วยโค้ดกำหนดตายตัว แทนที่จะฝากความหวังไว้กับการตัดสินใจของ LLM ล้วนๆ)
#
# ไม่ต้องขอ human-in-the-loop ซ้ำอีกรอบสำหรับปุ่มยืนยันในโมดัลนี้ — human อนุมัติ action ที่
# เปิดโมดัลนี้ไปแล้วครั้งเดียว (ผ่าน classify_action()/ask_user_func ปกติที่ execute() เช็คก่อน
# dispatch action เดิมอยู่แล้ว ก่อนจะมาถึง _dispatch_click_with_retry() นี้เลยด้วยซ้ำ) ปุ่ม
# ยืนยันในโมดัลเป็นแค่ UX ของเว็บที่ถาม "ซ้ำ" สำหรับ action เดียวกันที่อนุมัติไปแล้ว ไม่ใช่การ
# ตัดสินใจใหม่ที่ต้องขออนุมัติเพิ่ม
# W_dialog_generic (C3 จาก audit ของ P7/P8): ของเดิมมี 3 ตัวและ 2 ใน 3 เป็นของ OrangeHRM ล้วน
# (.oxd-dialog-container, .orangehrm-modal-header) เหลือของมาตรฐานแค่ [role="dialog"] ตัวเดียว
# — dialog ที่พบบ่อยที่สุดในโลกจริงจึงตรวจไม่เจอเลยสักตัว: <dialog> ของ HTML เอง, aria-modal,
# Bootstrap, MUI, antd, Radix, SweetAlert ทั้งหมดนี้กระทบทุกอย่างที่ยืนอยู่บน "รู้ไหมว่ามี
# dialog เปิดอยู่" ไม่ใช่แค่ auto-confirm
#
# เรียง generic ก่อน framework เสมอ (หลักการเดียวกับ _MODAL_CONFIRM_BUTTON_SELECTORS ที่เรียง
# เจาะจง->กว้าง แต่คนละเจตนา: ตัวนั้นเลือก "ปุ่มไหน" จึงต้องเจาะจงก่อน ตัวนี้แค่ตอบว่า "มีไหม")
_DIALOG_CONTAINER_SELECTORS = (
    "dialog[open]",
    '[role="dialog"]',
    '[role="alertdialog"]',
    '[aria-modal="true"]',
    ".modal.show",           # Bootstrap
    ".MuiDialog-root",       # MUI
    ".ant-modal-wrap",       # Ant Design
    ".swal2-container",      # SweetAlert2
    ".oxd-dialog-container", # OrangeHRM
    ".orangehrm-modal-header",
)
_DIALOG_CONTAINER_SELECTOR = ", ".join(_DIALOG_CONTAINER_SELECTORS)

# W_modal_confirm_generic (P3.9): ชั้น fallback เดิมกว้างแค่ 'button:has-text("Confirm")'
# เท่านั้น — dialog ที่เขียนว่า "Yes" / "OK" / "ตกลง" / "Löschen" / "Supprimer" จึงไม่ match
# อะไรเลยสักตัว แล้ว resolve_confirmation_modal() คืน None ทำให้ agent ค้างอยู่หน้าโมดัล
# ซึ่งเป็นบั๊กที่ W23 เขียนมาแก้พอดี แต่แก้ได้เฉพาะเว็บภาษาอังกฤษที่ใช้คำว่า Confirm
#
# คำยืนยันด้านล่างครอบภาษาชุดเดียวกับ permission/rules.py::RISKY_LABEL_KEYWORDS (ไทย/ญี่ปุ่น/
# จีน/เยอรมัน/ฝรั่งเศส/สเปน/โปรตุเกส) — ตั้งใจ *ไม่* import มาใช้ซ้ำ เพราะชุดนั้นตอบคำถามคนละ
# ข้อ ("action นี้เสี่ยงไหม" ซึ่งรวม pay/purchase ที่ไม่ใช่ปุ่มยืนยันในโมดัล) การผูกสองชุดเข้า
# ด้วยกันจะทำให้แก้ชุดหนึ่งแล้วอีกชุดเปลี่ยนพฤติกรรมตามโดยไม่ตั้งใจ
#
# *** ลำดับสำคัญมาก *** — คำยืนยันกลางๆ (yes/ok/confirm) มาก่อนคำทำลายข้อมูล (delete/remove)
# เสมอ เพราะ has-text() เป็น substring match: dialog ที่มีปุ่ม "Do not delete" จะ match คำว่า
# delete ด้วย ถ้าเอาคำทำลายขึ้นก่อนมีโอกาสกดผิดปุ่ม ส่วนคำปฏิเสธ (cancel/ยกเลิก/no) ไม่อยู่ใน
# ลิสต์นี้เลยโดยตั้งใจ
_MODAL_CONFIRM_TEXTS = (
    # อังกฤษ — ยืนยันกลางๆ ก่อน
    "Yes, Delete", "Yes", "OK", "Confirm", "Proceed", "Continue",
    # ไทย
    "ตกลง", "ยืนยัน", "ใช่",
    # ญี่ปุ่น / จีน
    "はい", "確認", "确定", "确认", "是",
    # เยอรมัน / ฝรั่งเศส / สเปน / โปรตุเกส
    "Ja", "Bestätigen", "Oui", "Confirmer", "Sí", "Si", "Confirmar", "Sim",
    # คำทำลายข้อมูล — ท้ายสุดเสมอ (ดูเหตุผลเรื่องลำดับด้านบน)
    "Delete", "Remove", "ลบ", "削除", "删除", "Löschen", "Supprimer", "Eliminar", "Excluir",
)

# เรียงจากเจาะจงที่สุด (OrangeHRM "Yes, Delete" ปุ่มสีแดง) ไปหากว้างที่สุด (fallback ทั่วไป
# สำหรับ dialog framework อื่นที่ไม่ใช่ OrangeHRM) — ลองทีละตัวจนกว่าจะเจอปุ่มที่ visible จริง
_MODAL_CONFIRM_BUTTON_SELECTORS = [
    "div.oxd-dialog-container-default button.oxd-button--label-danger",
    ".oxd-button--label-danger",
    'button:has-text("Yes, Delete")',
    '[role="dialog"] button.oxd-button--secondary',
    # W_modal_confirm_generic: ชั้น generic — จำกัดขอบเขตอยู่ใน dialog container เสมอ (ทั้ง
    # [role=dialog] มาตรฐานและ container ของ framework ที่ไม่ได้ใส่ role ให้) กันไปโดนปุ่มชื่อ
    # เดียวกันที่อยู่บนหน้าเว็บปกตินอกโมดัล
    # รวม container x tag ของคำเดียวกันไว้ใน selector เดียว (comma-separated) — ไล่ทีละคำ
    # ไม่ใช่ทีละ combination เพื่อคง "ลำดับความสำคัญของคำ" ไว้ครบโดยยิง locator แค่ 30 ครั้ง
    # แทน 120 ครั้ง (ฟังก์ชันนี้ถูกเรียกทุกครั้งที่เจอโมดัล จะช้าไม่ได้)
    *[
        ", ".join(
            f'{container} {tag}:has-text("{text}")'
            # W_dialog_generic: ใช้ชุด container เดียวกับ _DIALOG_CONTAINER_SELECTORS
            # ด้านบน ไม่ hardcode ซ้ำ — ไม่งั้นเพิ่ม framework ใหม่แล้วตรวจ "เจอ dialog" ได้
            # แต่หาปุ่มยืนยันในนั้นไม่เจอ ซึ่งแย่กว่าไม่ตรวจเจอตั้งแต่แรก
            for container in _DIALOG_CONTAINER_SELECTORS
            for tag in ("button", '[role="button"]')
        )
        for text in _MODAL_CONFIRM_TEXTS
    ],
]

# W_modal_appear_race: เวลารอ dialog ที่กำลัง animate เข้ามา (ดู _detect_confirmation_modal)
# 600ms พอสำหรับ transition ของ UI framework ทั่วไป (ส่วนใหญ่ 150-300ms) และถ้าสะสมทุก click
# ของ task หนึ่งก็ยังน้อยกว่าการคลิกพลาดเพราะโดน modal บังแค่ครั้งเดียว
_MODAL_APPEAR_TIMEOUT_MS = 600

_MODAL_DETACH_TIMEOUT_MS = 5000

# W24 ("Auto-Refresh & Re-attachment Guardrail" — บั๊กจริงที่ user รายงาน: ในงาน batch หลาย
# รอบ (ลบ user หลายชุดติดกัน) โมดัลยืนยัน/ปุ่มของรอบที่ 2 เป็นต้นไป "ไม่ตอบสนอง" ทั้งที่รอบแรก
# ทำงานปกติ — กด F5 มือแล้วหายเสมอ แปลว่า DOM node หลุด event binding หรือ UI desync กับ
# state จริงหลัง AJAX table reload ของรอบก่อนหน้า ไม่ใช่ปัญหา selector ผิด/element หาไม่เจอ
# (ถ้าเป็นแบบนั้น count()==0 จะกรองออกไปตั้งแต่ _find_visible_modal_confirm_button() แล้ว) —
# คลิกซ้ำเฉยๆ ไม่ช่วยเพราะปัญหาไม่ใช่ timing แต่เป็น state desync ระดับหน้าเว็บ ต้องจำลอง
# พฤติกรรม "กด F5" จริงๆ (page.reload()) ถึงจะ sync กลับมาได้ — retry ธรรมดาก่อน (เผื่อเป็น
# แค่ animation/timing ปกติ) แล้วค่อย fallback ไป reload ถ้า retry ครบแล้วยังไม่หาย
_MODAL_CONFIRM_CLICK_RETRIES = 3
_MODAL_CONFIRM_RETRY_DELAY_SEC = 1.0
_MODAL_RELOAD_TIMEOUT_MS = 15000


async def _detect_confirmation_modal(page: Page) -> bool:
    """W23: True ถ้ามี dialog/modal container ปรากฏอยู่จริงบนหน้าตอนนี้ (มองเห็นได้) — ไม่
    throw ออกไปพัง (เหมือนหลักการเดียวกับ orchestrator.py::_scan_validation_errors: เช็ค
    ไม่ได้ ถือว่า "ไม่มีโมดัล" ปลอดภัยกว่าเสมอ ดีกว่าไปบล็อก/หน่วง click ที่สำเร็จอยู่แล้ว)

    W_modal_appear_race (C2 จาก audit ของ P7/P8): ฟังก์ชันนี้ถูกเรียก *ทันที* หลัง click สำเร็จ
    แต่ dialog ส่วนใหญ่ animate เข้ามา — ตอนเช็ค node ยังไม่อยู่ใน DOM จึงได้ count()==0 แล้ว
    สรุปว่า "ไม่มีโมดัล" ทั้งที่อีกเสี้ยววินาทีมันจะโผล่ขึ้นมาบังทั้งหน้า
    นี่คือสาเหตุที่ user เจอ agent ติด loop: modal เปิดค้าง -> ทุก click ถัดไป fail -> ไม่มี
    ทางกลับมาถึงบรรทัดที่เรียกฟังก์ชันนี้อีกเลย (จุดเรียกเดียวอยู่ใต้ `if result.success:`)
    = dead-end ที่ออกเองไม่ได้

    จึงรอสั้นๆ ก่อนสรุปว่าไม่มี — เจตนา *ไม่* ใช้ _dom_signature() มากรองว่า "ควรรอไหม" ตามที่
    เคยร่างไว้ เพราะมันคือความยาว body.innerHTML: modal ที่ markup อยู่ใน DOM อยู่แล้วและเปิด
    ด้วยการสลับ class (display:none -> block, .modal.show, MUI keepMounted) ความยาวไม่เปลี่ยน
    เลยสักตัวอักษร ตัวกรองนั้นจึงพลาด modal ทั้งตระกูล
    ราคาที่จ่าย: click ที่ไม่เปิดอะไรเลยเสียเพิ่ม _MODAL_APPEAR_TIMEOUT_MS — ตั้งไว้สั้นพอที่
    สะสมทั้ง task แล้วยังน้อยกว่า *การคลิกพลาดครั้งเดียว* ที่กินสูงสุด ~18 วินาที"""
    if await _is_modal_still_open(page):
        return True
    try:
        await page.locator(_DIALOG_CONTAINER_SELECTOR).first.wait_for(
            state="visible", timeout=_MODAL_APPEAR_TIMEOUT_MS,
        )
        return True
    except Exception:
        return False


async def _is_modal_still_open(page: Page) -> bool:
    """W24: ใช้ตรรกะเดียวกับ _detect_confirmation_modal() ทุกประการ แยกฟังก์ชันเพราะ
    resolve_confirmation_modal() ด้านล่างต้องเรียกซ้ำหลายจุด (เช็คตอนเข้า/เช็คซ้ำหลัง
    detach-wait timeout) — ตั้งชื่อสื่อบริบทการใช้งานที่ต่างกันให้อ่านง่ายกว่าเรียก
    _detect_confirmation_modal() ตรงๆ ซ้ำๆ"""
    try:
        locator = page.locator(_DIALOG_CONTAINER_SELECTOR).first
        if await locator.count() == 0:
            return False
        return await locator.is_visible(timeout=_ELEMENT_ACTION_TIMEOUT_MS)
    except Exception:
        return False


async def _find_visible_modal_confirm_button(page: Page):
    """W23/W24: ไล่ตาม _MODAL_CONFIRM_BUTTON_SELECTORS ทีละตัว (เจาะจงที่สุดก่อน) คืน
    (selector, locator) คู่แรกที่เจอ+visible จริง หรือ (None, None) ถ้าไม่เจอเลยสักตัว —
    แยกออกมาจาก resolve_confirmation_modal() เพราะ W24 ต้อง "หาปุ่มครั้งเดียว แล้วคลิกซ้ำได้
    หลายรอบ" (ปุ่มเดิมตัวเดียวกัน ไม่ใช่ query หา element ใหม่ทุกรอบ retry — การ query ใหม่ทุก
    รอบเสี่ยงหยิบปุ่มที่ "หน้าตาเหมือนเดิมแต่จริงๆ คือ element คนละตัว" หลัง desync ได้เช่นกัน)"""
    for selector in _MODAL_CONFIRM_BUTTON_SELECTORS:
        try:
            candidate = page.locator(selector).first
            if await candidate.count() == 0:
                continue
            if not await candidate.is_visible(timeout=_ELEMENT_ACTION_TIMEOUT_MS):
                continue
            return selector, candidate
        except Exception:
            continue
    return None, None


async def resolve_confirmation_modal(page: Page) -> Optional[str]:
    """W23/W24: หาปุ่มยืนยันของ confirmation modal (_find_visible_modal_confirm_button())
    แล้วคลิกด้วย force=True (ข้าม actionability check ของ Playwright — modal บางตัว animate
    เข้ามาทำให้ element "ยังไม่ visible ตามนิยามของ Playwright" ชั่วขณะแม้จะมองเห็นได้จริงบนจอ
    แล้วก็ตาม) — ลองคลิก+รอ detach สูงสุด _MODAL_CONFIRM_CLICK_RETRIES ครั้ง ห่างกันครั้งละ
    _MODAL_CONFIRM_RETRY_DELAY_SEC วินาที (เผื่อเป็นแค่ animation/timing ปกติ) ถ้าครบโควตา
    แล้วโมดัลยังไม่ปิดจริง (ปุ่ม "ไม่ตอบสนอง" ตามที่ user รายงาน — ไม่ใช่ timing ธรรมดาแล้ว
    แต่เป็น UI state desync หลัง AJAX reload ของรอบก่อนหน้า) ให้ page.reload() จำลอง "กด F5"
    จริง แล้วรอ networkidle ก่อน return

    คืนข้อความสรุปสั้นๆ ให้ต่อท้าย message ของ action หลักที่ trigger โมดัลนี้ (เช่น
    "delete(3)") ให้ LLM/log เห็นว่าเกิดอะไรขึ้นเพิ่มเติมหลัง action นั้นแบบโปร่งใส (รวมถึงกรณี
    reload — ข้อความจะบอก LLM ให้รู้ว่าต้อง navigate/กรองข้อมูลใหม่เองต่อ เพราะ reload ล้าง
    client-side state เช่นคำค้นหาที่กรองไว้ทิ้งไปด้วย) — None ถ้าไม่เจอปุ่มยืนยันเลยสักตัวตั้งแต่
    แรก (ปล่อยผ่านเงียบๆ ให้ LLM ตัดสินใจเองต่อในรอบถัดไปตามปกติ ไม่ throw/ไม่ทำให้ action หลัก
    ที่เพิ่ง success กลายเป็น fail ไปด้วยเพราะเหตุนี้)"""
    clicked_selector, candidate = await _find_visible_modal_confirm_button(page)
    if clicked_selector is None:
        return None

    for attempt in range(1, _MODAL_CONFIRM_CLICK_RETRIES + 1):
        try:
            await candidate.click(force=True, timeout=_ELEMENT_ACTION_TIMEOUT_MS)
        except Exception:
            pass  # ปุ่ม "ไม่ตอบสนอง" ก็เข้าเงื่อนไขนี้ได้เหมือนกัน — ยัง retry ต่อได้ ไม่ throw ทันที

        try:
            await page.wait_for_selector(
                _DIALOG_CONTAINER_SELECTOR, state="detached", timeout=_MODAL_DETACH_TIMEOUT_MS,
            )
            await wait_stable(page)
            retry_note = "" if attempt == 1 else f" (attempt {attempt}/{_MODAL_CONFIRM_CLICK_RETRIES})"
            return f" [Confirmation modal detected — confirmed automatically ({clicked_selector}){retry_note}]"
        except Exception:
            # detach-wait timeout: อาจเป็นเพราะโมดัลปิดจริงแล้วแค่ไม่ detach ออกจาก DOM (บาง
            # framework ซ่อนด้วย CSS อย่างเดียว ไม่ลบ element) หรืออาจเป็นเพราะยังเปิดค้างอยู่
            # จริงๆ (ปุ่มไม่ตอบสนอง) — เช็คแยกให้ชัดก่อนตัดสินใจ retry/reload ต่อ ไม่เดาว่า
            # timeout = ปิดสำเร็จเหมือนพฤติกรรมเดิม (W23) อีกต่อไป
            if not await _is_modal_still_open(page):
                await wait_stable(page)
                return f" [Confirmation modal detected — confirmed automatically ({clicked_selector})]"
            if attempt < _MODAL_CONFIRM_CLICK_RETRIES:
                await asyncio.sleep(_MODAL_CONFIRM_RETRY_DELAY_SEC)

    # W24: ครบโควตา retry แล้วโมดัลยังเปิดค้างอยู่จริง — ปุ่มยืนยัน "ไม่ตอบสนอง" จริงๆ ตามที่
    # user รายงาน จำลองพฤติกรรม "กด F5" ด้วย page.reload() แทน ไม่ throw ออกไปแม้ reload เอง
    # จะ fail (เช่น network เพี้ยนชั่วคราว) — ปลอดภัยกว่าเสมอที่จะแจ้ง LLM ให้รู้สถานการณ์ต่อ
    # ดีกว่าทำให้ action หลักที่เพิ่ง success (คลิก "Delete Selected") กลายเป็น fail ไปด้วย
    try:
        await page.reload(timeout=_MODAL_RELOAD_TIMEOUT_MS)
        await page.wait_for_load_state("networkidle", timeout=_MODAL_RELOAD_TIMEOUT_MS)
    except Exception:
        pass
    return (
        f" [The confirmation modal's confirm button was unresponsive after {_MODAL_CONFIRM_CLICK_RETRIES} attempts — the system reloaded the page automatically to resync state (as if pressing F5). Check the indexed elements of this freshly reloaded page, then navigate/re-apply the filter the goal needs before continuing, because the reload wiped the previous state (e.g. the search term you had filtered by)]"
    )


# W50: keyboard-based interaction สำหรับ custom dropdown/menu widget (MUI/Ant Design/
# React-select/Headless UI ฯลฯ) ที่ implement เองด้วย <div>/<li> role=option/menuitem
# (ดู perception.py W50) ไม่ใช่ <select><option> จริง — คลิกเลือก option ตรงๆ บางทีพลาด
# เพราะ DOM ซับซ้อน/มี animation/ตัวเลือกซ้อนอยู่ใต้ overlay — sequence ที่เสถียรกว่าคือ
# click เปิด dropdown ก่อน แล้วส่ง key (ArrowDown/ArrowUp/Enter ฯลฯ) ไปยัง element เดิม
# (เบราว์เซอร์ native keyboard navigation ของ widget เอง) แทนการไล่หา selector ของตัวเลือก
async def press_key(page: Page, index: int, key: str, timeout: int = _ELEMENT_ACTION_TIMEOUT_MS) -> ActionResult:
    """ส่ง key event ไปยัง element ตาม index (Playwright's locator.press() focus element
    ให้ก่อนส่ง key เองอยู่แล้ว — ต่างจาก page.keyboard.press() ที่ยิงไปที่ element ที่มี
    focus อยู่ ณ ขณะนั้นเฉยๆ ไม่รับประกันว่าเป็น element ที่ LLM ตั้งใจสั่ง)"""
    try:
        selector = _sel(index)
        target = await resolve_frame(page, selector)
        await target.press(selector, key, timeout=timeout)
        return ActionResult(True, f"press_key({index}, {key})", f"pressed key '{key}' succeeded")
    except PWTimeout:
        return ActionResult(False, f"press_key({index}, {key})", "element not found / key press failed (timeout)")
    except Exception as e:
        return ActionResult(False, f"press_key({index}, {key})", f"error: {e}")


# W19 ("Safe Input Replacement"): เว็บที่มี custom autocomplete/controlled input (React/
# Vue state เช่น ช่องค้นหา YouTube) บางเว็บไม่เห็นการเคลียร์แบบ CDP-level ล้วนๆ ของ
# Playwright's .fill() ว่าเป็น "การกดจริง" — ค่าที่เห็นในช่องอาจดูเหมือนเปลี่ยนแล้ว แต่
# state ภายในของ widget (เช่น autocomplete popup) ยังอ้างอิงคำค้นหาเดิมอยู่ ก่อนพิมพ์
# ข้อความใหม่ทุกครั้งจึงต้อง focus -> select-all (Ctrl+A) -> Backspace ก่อนเสมอ (จำลอง
# การกดจริงของมนุษย์ trigger keyboard event ที่ widget พวกนี้ฟังอยู่จริง) แล้วค่อย .fill()
# ข้อความใหม่ลงในช่องที่ว่างแล้ว (เร็วกว่า/เชื่อถือได้กว่าการพิมพ์ทีละตัวอักษร เพราะช่อง
# ว่างเปล่าแล้วไม่มีอะไรให้ .fill() ต้องเคลียร์ซ้ำอีก)
# W_fill_wrapper_resolves_to_inner_input: element ที่กรอกได้จริงตามนิยามของ Playwright เอง
# (ข้อความ error ของมันบอกไว้ตรงๆ ว่ารับอะไรบ้าง) — เช็คกับตัว element ก่อน ไม่ใช่เดาจาก tag
# ที่ perception รายงาน เพราะ index ชี้ไปที่ DOM node จริงเสมอ
_IS_FILLABLE_JS = """(el) => {
    const tag = (el.tagName || '').toLowerCase();
    if (tag === 'textarea' || tag === 'select') return true;
    if (tag === 'input') return (el.type || 'text').toLowerCase() !== 'hidden';
    return !!el.isContentEditable;
}"""

# ช่องกรอกตัวแรกที่อยู่ *ข้างใน* ตัวห่อ — เรียง input ก่อน textarea/contenteditable ตามความถี่
# ที่พบจริงบนฟอร์ม และตัด hidden ออกเพราะกรอกไม่ได้อยู่แล้ว
_INNER_FILLABLE_SELECTOR = 'input:not([type="hidden"]), textarea, [contenteditable="true"]'


async def _effective_fill_selector(
    target: Union[Page, Frame], selector: str, timeout: int,
) -> str:
    """คืน selector ของ element ที่กรอกได้จริง — ตัวเดิมถ้ามันกรอกได้อยู่แล้ว

    ถ้า element ที่ index ชี้ไปเป็นแค่กล่องครอบ (พบบ่อยบน SPA ที่ห่อ <input> ไว้ใน div ที่ถือ
    label ของช่องนั้น) ให้เล็งช่องกรอกตัวแรกข้างในแทน — ถ้าไม่มีข้างในเลยก็คืนตัวเดิมไป ให้
    Playwright เป็นคนบอก error ตามความจริง ไม่ใช่เงียบไปเฉยๆ (fail-safe เหมือนทุกตัวในไฟล์นี้)"""
    try:
        fillable = await target.locator(selector).evaluate(_IS_FILLABLE_JS, timeout=timeout)
    except Exception:
        return selector
    if fillable:
        return selector
    inner = f"{selector} :is({_INNER_FILLABLE_SELECTOR})"
    try:
        found = await target.query_selector(inner)
    except Exception:
        return selector
    return inner if found is not None else selector


async def fill(page: Page, index: int, text: str, timeout: int = _ELEMENT_ACTION_TIMEOUT_MS) -> ActionResult:
    """พิมพ์ข้อความลงช่อง input/textarea ตาม index — เคลียร์ข้อความเดิมด้วย
    focus -> select-all -> Backspace ก่อนเสมอ (ดู module comment ด้านบน)

    W_datepicker ("Popup Dismissal After Fill" — บั๊กจริงที่ user รายงาน: agent กรอกวันที่
    ในช่อง "From Date" สำเร็จ แต่ "To Date" กลับไม่ถูกกรอกแล้วแจ้ง success เท่านั้น) —
    ยืนยันจากการทดสอบจริงบน OrangeHRM Leave List ว่า focus (จาก target.click() ด้านบน)
    ทำให้ date-picker calendar popup เปิดขึ้นมาเป็นผลข้างเคียงของ framework เอง (ไม่ใช่
    intentional — เราแค่ต้องการ focus ก่อน clear text) popup นี้แทรก element ใหม่ (ปุ่ม
    นำทางเดือน/ปี) เข้า DOM ทำให้ data-ai-index ของ element ถัดๆ ไปใน DOM order เลื่อน
    หนีจากตำแหน่งเดิม (สลับ index ของ "To Date" ไปเป็น index ของปุ่มนำทางปฏิทินแทน) ถ้า
    agent (หรือ compound action อื่นในคำสั่งเดียวกัน) อ้าง index จาก snapshot ก่อนหน้าที่
    ยังไม่มี popup — พลาด target ไปกดปุ่มปฏิทินแทนช่องกรอกจริงเงียบๆ โดยไม่มี error ให้เห็น
    เลย (ปุ่มปฏิทินไม่ error แค่ไม่มีผลตามที่ agent ตั้งใจ) — ปิด popup ทันทีหลัง fill()
    สำเร็จเสมอ กัน state ที่ไม่คาดคิดแบบนี้หลุดไปถึงรอบ perceive ถัดไป

    ทดสอบแล้ว: Escape เพียงอย่างเดียวไม่ปิด popup นี้ (framework ผูก listener กับ
    outside-click ไม่ใช่ keydown) ต้องเป็นการคลิกจริงเท่านั้น — blur() + คลิกที่ <body>
    ตรงๆ (ไม่ใช่คลิกตำแหน่งพิกัดบนหน้าจอที่อาจไปโดน element อื่นที่มี handler ไม่พึงประสงค์
    เช่น link ที่พาไป navigate โดยไม่ตั้งใจ — body ไม่มีทางมี handler ที่ทำอะไรเองอยู่แล้ว)
    จำลอง "คลิกออกไปข้างนอก" แบบที่มนุษย์จริงทำหลังพิมพ์เสร็จตามธรรมชาติอยู่แล้ว ปลอดภัยกับ
    input ทั่วไปที่ไม่มี popup อะไรเลยด้วย (ไม่มีผลข้างเคียง) — best-effort เท่านั้น ห่อ
    try/except กันไม่ให้ fill() ที่สำเร็จไปแล้วกลายเป็น fail เพราะขั้นตอนเสริมนี้พังเฉยๆ
    (เช่น element หลุดจาก DOM ไปแล้วหลัง fill)"""
    try:
        selector = _sel(index)
        target = await resolve_frame(page, selector)
        # W_fill_wrapper_resolves_to_inner_input: index อาจชี้ที่กล่องครอบ ไม่ใช่ช่องกรอก
        selector = await _effective_fill_selector(target, selector, timeout)
        await target.click(selector, timeout=timeout)
        await target.press(selector, "ControlOrMeta+a", timeout=timeout)
        await target.press(selector, "Backspace", timeout=timeout)
        await target.fill(selector, text, timeout=timeout)
        try:
            # W_check_evaluate_timeout: จุดเดียวกันกับใน check() — Locator.evaluate() ที่ไม่ระบุ
            # timeout รอได้ถึง 30 วินาที ทั้งที่นี่เป็นแค่ขั้นตอนเสริม best-effort หลัง fill สำเร็จ
            # ไปแล้ว (element ที่หลุดจาก DOM หลัง fill คือเคสที่ทำให้รอเต็มเวลาโดยไม่ได้อะไรเลย)
            await target.locator(selector).evaluate(
                "el => { el.blur(); document.body.click(); }", timeout=_STATE_READ_TIMEOUT_MS,
            )
            # popup ปิดจริง (ยืนยันจากการทดสอบ) แต่ไม่ synchronous — framework ใช้
            # transition/nextTick ก่อนถอด element ออกจาก DOM จริง (~100-300ms) ไม่รอตรงนี้
            # จะคืนผลลัพธ์ก่อน popup หายจริง ทำให้ get_snapshot() รอบถัดไป (ที่ orchestrator
            # เรียกทันทีหลัง action นี้) ยังเห็น popup/ปุ่มนำทางค้างอยู่
            await target.wait_for_timeout(200)
        except Exception:
            pass
        descriptor = await compute_locator_descriptor(target, selector)
        return ActionResult(True, f"fill({index})", f"filled '{text}' succeeded", locator_descriptor=descriptor)
    except PWTimeout:
        return ActionResult(False, f"fill({index})", "could not fill (timeout)")
    except Exception as e:
        return ActionResult(False, f"fill({index})", f"error: {e}")


# W65[3] ("Vault Expansion — Current Password Auto-fill"): แทนที่จะสร้าง multi-named-secret
# store ใหม่ (ต้องเปลี่ยน schema credentials.json จาก single-pair เป็น dict — เสี่ยง/effort
# สูงเกินจำเป็นสำหรับ use case นี้) reuse credential login ที่บันทึกไว้ต่อโดเมนอยู่แล้ว
# (site_learning/storage.py::save_credentials/load_credentials) เป็น "current password"
# โดยตรง — ตรงกับความหมายจริง (current password ก็คือรหัสที่ใช้ login อยู่ตอนนี้)
#
# ***ต้องไม่หลุดเข้า LLM context เด็ดขาด*** (หลักการเดียวกับ orchestrator.py::
# _maybe_auto_login) — ค่าจริงไม่เคยถูกส่งผ่าน cmd["text"] จาก LLM เลย (LLM ส่งแค่ secret_key
# ที่เป็นชื่อ symbolic เช่น "current_password" มา) และ ActionResult.message ต้องไม่ echo ค่า
# จริงกลับไปด้วย (message บอกแค่ "สำเร็จ"/"ล้มเหลว" เฉยๆ) — ต่างจาก fill() ธรรมดาด้านบนที่ log
# ข้อความที่กรอกไว้ตรงๆ ได้เพราะเป็นข้อมูลที่ LLM ให้มาเองอยู่แล้ว ไม่ใช่ความลับ
_SUPPORTED_SECRET_KEYS = {"current_password"}


async def fill_secret(page: Page, index: int, secret_key: str, timeout: int = _ELEMENT_ACTION_TIMEOUT_MS) -> ActionResult:
    """กรอกค่าลับที่บันทึกไว้ (ตอนนี้รองรับแค่ secret_key="current_password" — รหัสผ่านที่ใช้
    login เว็บนี้อยู่) ลงช่อง input ตาม index — คืน [FAIL] แบบ fail-safe (ไม่มี credential
    บันทึกไว้/secret_key ที่ไม่รู้จัก) ให้ LLM fallback ไปถาม user เองตามกติกา W65[1] ปกติ
    แทนที่จะ throw หรือค้าง"""
    if secret_key not in _SUPPORTED_SECRET_KEYS:
        return ActionResult(False, f"fill_secret({index})", f"unknown secret_key '{secret_key}' — you must ask the user yourself")

    # Lazy import กัน circular import (site_learning -> crawler.py -> orchestrator.py ->
    # fastpath_executor.py -> actions.py) — pattern เดียวกับ orchestrator.py::_maybe_auto_login()
    from backend.app.site_learning import storage as site_storage

    domain = extract_domain(page.url)
    creds = site_storage.load_credentials(domain)  # sync call, pattern เดียวกับ _maybe_auto_login
    if not creds or not creds.get("password"):
        return ActionResult(False, f"fill_secret({index})", "no credential saved for this site — you must ask the user yourself")

    try:
        selector = _sel(index)
        target = await resolve_frame(page, selector)
        # W_fill_wrapper_resolves_to_inner_input (บั๊กจริงหน้า Update Password ของ OrangeHRM:
        # "Element is not an <input>, <textarea>, <select> or [contenteditable]")
        selector = await _effective_fill_selector(target, selector, timeout)
        await target.click(selector, timeout=timeout)
        await target.press(selector, "ControlOrMeta+a", timeout=timeout)
        await target.press(selector, "Backspace", timeout=timeout)
        await target.fill(selector, creds["password"], timeout=timeout)
        descriptor = await compute_locator_descriptor(target, selector)
        # ***ห้าม echo ค่าจริงกลับใน message เด็ดขาด*** ต่างจาก fill() ปกติด้านบน
        return ActionResult(True, f"fill_secret({index})", "filled the saved password successfully", locator_descriptor=descriptor)
    except PWTimeout:
        return ActionResult(False, f"fill_secret({index})", "could not fill (timeout)")
    except Exception as e:
        return ActionResult(False, f"fill_secret({index})", f"error: {e}")


async def select_option(page: Page, index: int, label: str, timeout: int = _ELEMENT_ACTION_TIMEOUT_MS) -> ActionResult:
    """เลือกตัวเลือกใน dropdown (<select>) ตาม index — เลือกด้วยข้อความที่เห็น

    W42: ดึง option {text, value} จริงจาก DOM มาก่อนเทียบ text แบบ normalize whitespace
    (nbsp/ซ้ำ/trailing) กับ label ที่ LLM ส่งมา แทนที่จะยิง select_option(label=label) แบบ
    exact string ตรงๆ (พังกับเว็บที่ใช้ &nbsp; คั่นคำใน option เช่น
    uitestingplayground.com/select — ดู docstring หัวไฟล์) เจอ match แล้วเลือกด้วย
    select_option(value=...) ของ option นั้น (ไม่ใช่ label=matched_text) — ***เหตุผลที่ใช้
    value ไม่ใช่ text แม้จะ normalize แล้ว: select_option(label=...) ของ Playwright เองก็
    เทียบแบบ exact string ภายในอีกที ถ้า text จริงมี whitespace ยุ่งๆ (เช่น "  New   York  "
    จาก HTML indentation) การส่ง text ที่ normalize แล้วหรือ text ดิบกลับไปก็ยังไม่ตรงกับที่
    Playwright คาดหวังอยู่ดี (เจอจากการทดสอบจริง) — value attribute เป็น token สั้นๆ ที่ไม่มี
    ปัญหา whitespace แบบนี้ตั้งแต่ต้น (หรือถ้าไม่มี value= ระบุไว้ใน HTML เลย browser จะ
    default value เป็น text เดียวกันเป๊ะ ก็ยัง match ได้ปกติ) เชื่อถือได้กว่า***"""
    selector = _sel(index)
    target = await resolve_frame(page, selector)

    try:
        options = await target.locator(selector).locator("option").evaluate_all(
            "elements => elements.map(el => ({text: el.textContent, value: el.value}))"
        )
    except Exception as e:
        return ActionResult(False, f"select({index})", f"error: {e}")

    normalized_target = _normalize_option_text(label)
    matched = next(
        (opt for opt in options if _normalize_option_text(opt["text"]) == normalized_target), None,
    )

    if matched is not None:
        try:
            await target.select_option(selector, value=matched["value"], timeout=timeout)
            descriptor = await compute_locator_descriptor(target, selector)
            return ActionResult(
                True, f"select({index})", f"selected '{_normalize_option_text(matched['text'])}' succeeded",
                locator_descriptor=descriptor,
            )
        except PWTimeout:
            return ActionResult(False, f"select({index})", "could not select (timeout)")
        except Exception as e:
            return ActionResult(False, f"select({index})", f"error: {e}")

    # ไม่เจอ text ที่ตรงกันเลยแม้ normalize แล้ว — เผื่อ label ที่ LLM ส่งมาคือ value
    # attribute ไม่ใช่ text ที่เห็น (ของเดิมมี fallback นี้อยู่แล้ว ยังคงไว้เหมือนเดิม)
    try:
        await target.select_option(selector, value=label, timeout=timeout)
        descriptor = await compute_locator_descriptor(target, selector)
        return ActionResult(True, f"select({index})", f"selected (by value) '{label}' succeeded", locator_descriptor=descriptor)
    except Exception:
        options_repr = ", ".join(repr(_normalize_option_text(o["text"])) for o in options)
        options_repr = options_repr or "(no options found in this dropdown)"
        return ActionResult(
            False, f"select({index})",
            f"no option matching '{label}' even after normalising whitespace — options actually available: {options_repr}",
        )


async def _is_effectively_checked(target: Union[Page, Frame], selector: str) -> bool:
    """W21: เช็คว่า element (หรือ input/state ที่เกี่ยวข้อง) อยู่ในสถานะ "ติ๊กแล้ว" จริงหรือไม่
    หลัง force-click/JS-click ด้านล่างใน check() — ใช้แทน Playwright's is_checked() ตรงๆ
    เพราะ custom checkbox (เช่น OrangeHRM .oxd-checkbox-input) มักไม่ใช่ <input> ที่ index ชี้
    ไปตรงๆ (Playwright's is_checked() ต้องการ input/[role=checkbox] เป๊ะๆ ไม่งั้น throw) —
    ไล่เช็คหลายสัญญาณตามลำดับความน่าเชื่อถือ: (1) ตัวเองเป็น input/[role=checkbox] จริง ใช้
    .checked/aria-checked ตรงๆ (2) มี <input type=checkbox> ซ้อนอยู่ข้างใน (custom wrapper ที่
    ห่อ native input ไว้แต่ซ่อนด้วย CSS) (3) มี aria-checked บน element เอง (4) สุดท้ายเดาจาก
    class ที่บ่งบอกสถานะ active/checked (บาง framework สลับแค่ class ผ่าน JS ล้วนๆ ไม่มี aria
    เลย) — คืน False ถ้าเช็คอะไรไม่ได้เลย (ปลอดภัยกว่าเดาว่าติ๊กแล้วทั้งที่ไม่แน่ใจ)"""
    try:
        return await target.locator(selector).evaluate(
            """el => {
                const readCheckbox = (node) => {
                    if (!node) return null;
                    if ('checked' in node && typeof node.checked === 'boolean') return node.checked;
                    const ariaChecked = (node.getAttribute && node.getAttribute('aria-checked') || '').toLowerCase();
                    if (ariaChecked === 'true') return true;
                    if (ariaChecked === 'false') return false;
                    return null;
                };
                let result = readCheckbox(el);
                if (result === null) {
                    const nestedInput = el.querySelector && el.querySelector('input[type="checkbox"], input[type="radio"]');
                    result = readCheckbox(nestedInput);
                }
                if (result === null) {
                    const cls = (el.className || '').toString().toLowerCase();
                    result = cls.includes('checked') || cls.includes('--active') || cls.includes(' active');
                }
                return !!result;
            }""",
            # W_check_evaluate_timeout: ต้องระบุเสมอ — ไม่ระบุ = 30 วินาทีของ Playwright
            # ต่อการเรียกหนึ่งครั้ง และ check() เรียกฟังก์ชันนี้ได้ถึง 2 ครั้งต่อความพยายาม
            timeout=_STATE_READ_TIMEOUT_MS,
        )
    except Exception:
        return False


async def check(page: Page, index: int, timeout: int = _ELEMENT_ACTION_TIMEOUT_MS) -> ActionResult:
    """ติ๊ก checkbox/radio ตาม index

    W21 ("Custom UI Checkbox"): OrangeHRM และ SPA framework ทั่วไปมักซ่อน native
    <input type="checkbox"> จริงด้วย CSS (display:none/opacity:0) แล้วแทนที่ด้วย span/div
    ห่อหุ้มที่ styled เอง (เช่น .oxd-checkbox-input) — target.check() ของ Playwright ปฏิเสธ
    ทันที (throw) ถ้า element ที่ selector ชี้ไปไม่ใช่ input/[role=checkbox] ที่ "visible" ตาม
    นิยามของ Playwright เอง ไม่ว่าจะ retry กี่ครั้งก็ตาม (deterministic mismatch เหมือน W42 ใน
    select_option ด้านบน ไม่ใช่ timing issue) — perception.py ตอนนี้ติด index ให้ wrapper
    พวกนี้ได้แล้ว (ดู CHECKBOX_WRAPPER_SELECTOR) แต่ยังต้องมีทาง "คลิก" ที่ไม่ใช่ .check() ตรงๆ
    รองรับด้วย จึงเพิ่ม fallback 2 ชั้นเรียงจากรุกน้อยไปมาก แล้ว verify สถานะจริงหลังคลิกทุกครั้ง
    (ต่างจาก click()/fill() อื่นที่เชื่อว่า Playwright ไม่ throw = สำเร็จ เพราะ custom checkbox
    "คลิกได้ไม่ error" ไม่ได้แปลว่า "ติ๊กแล้วจริง" เสมอไป)"""
    selector = _sel(index)

    # ทางหลัก: native check() ปกติ — เร็วและตรงกับ <input type=checkbox>/[role=checkbox]
    # ทั่วไปส่วนใหญ่อยู่แล้ว ไม่ต้อง fallback เลยถ้าสำเร็จ
    try:
        target = await resolve_frame(page, selector)
        await target.check(selector, timeout=timeout)
        descriptor = await compute_locator_descriptor(target, selector)
        return ActionResult(True, f"check({index})", "checked successfully", locator_descriptor=descriptor)
    except Exception:
        pass

    # Fallback 1: force click — ข้าม Playwright's actionability check (element visible ตาม
    # นิยามของ Playwright) เหมือน hover(force=True) ด้านบน คลิกที่พิกัดกึ่งกลางของ element
    # ตรงๆ ไม่ว่า Playwright จะมองว่า element นี้ "checkable" หรือไม่
    try:
        target = await resolve_frame(page, selector)
        await target.click(selector, timeout=min(timeout, _ELEMENT_FALLBACK_TIMEOUT_MS), force=True)
        if await _is_effectively_checked(target, selector):
            descriptor = await compute_locator_descriptor(target, selector)
            return ActionResult(True, f"check({index})", "checked successfully (force click)", locator_descriptor=descriptor)
    except Exception:
        pass

    # Fallback 2: JS dispatch ตรงบน element ผ่าน el.click() — ข้าม actionability check และ
    # การจำลอง mouse event ของ Playwright ทั้งหมด ใช้เป็นทางสุดท้ายสำหรับ wrapper ที่ force
    # click ก็ยังคลิกไม่โดน (เช่น wrapper ที่มีขนาด 0x0 จริงๆ ตัวอย่างการมองเห็นมาจาก
    # pseudo-element ล้วนๆ)
    try:
        target = await resolve_frame(page, selector)
        # W_check_evaluate_timeout: ต้องส่ง timeout เองเสมอ — Locator.evaluate() ที่ไม่ระบุ
        # ใช้ default 30 วินาทีของ Playwright ทำให้ action สุดท้ายของ task ค้างครึ่งนาที
        # ทั้งที่งานจริงเสร็จไปแล้ว (รันจริง 2026-09-03: "Timeout 30000ms exceeded" ที่
        # step สุดท้าย หลังลบข้อมูลครบตั้งแต่ step ก่อนหน้า) — บั๊กคลาสเดียวกับ
        # W_descriptor_timeout ที่ compute_locator_descriptor() เคยโดนมาแล้ว
        await target.locator(selector).evaluate(
            "el => el.click()", timeout=min(timeout, _ELEMENT_FALLBACK_TIMEOUT_MS),
        )
        if await _is_effectively_checked(target, selector):
            descriptor = await compute_locator_descriptor(target, selector)
            return ActionResult(True, f"check({index})", "checked successfully (JS click)", locator_descriptor=descriptor)
        return ActionResult(False, f"check({index})", "clicked, but the checked state could not be confirmed")
    except Exception as e:
        return ActionResult(False, f"check({index})", f"error: {e}")


async def scroll(page: Page, direction: str = "down", amount: int = 600) -> ActionResult:
    """เลื่อนหน้าจอ ('down'/'up') — ใช้ตอน element ที่ต้องการอยู่นอกจอ

    W_inner_scroll (ดูเหตุผลเต็มที่ state_filter.py::_FIND_SCROLLER_FN_JS): เดิมใช้
    page.mouse.wheel() ซึ่งเลื่อน "อะไรก็ตามที่อยู่ใต้ตำแหน่งเมาส์ปัจจุบัน" — ตำแหน่งนั้นไม่มี
    ใครคุมเลยในระบบนี้ (ไม่เคย move mouse ไปไหนโดยตั้งใจ) บน layout ที่ pane ข้างในเป็นตัว
    scroll ผลจึงขึ้นกับความบังเอิญล้วนๆ

    เลื่อน element ตัวเดียวกับที่ check_scroll_redundant() ใช้ตัดสินว่า "ถึงขอบหรือยัง" แทน
    แล้วรายงานระยะที่เลื่อนได้จริง — ถ้าเลื่อนไม่ได้เลยต้องบอกตามตรง (success=False) ไม่ใช่
    คืน [OK] ลอยๆ ให้โมเดลเข้าใจผิดว่าเลื่อนแล้ว (ธีมเดียวกับ W_click_native_select)

    ถ้า evaluate ล้มเหลว (หน้าแปลก/CSP) ยัง fallback ไป mouse.wheel แบบเดิมทุกประการ"""
    dy = amount if direction == "down" else -amount
    try:
        moved = await page.evaluate(state_filter.SCROLL_BY_JS, dy)
        await page.wait_for_timeout(300)
        delta = int(moved["after"]) - int(moved["before"])
        if delta == 0:
            return ActionResult(
                False, f"scroll({direction})",
                "nothing scrolled — the scrollable area is already at that end, or this page "
                "does not scroll at all. Do not repeat this scroll; act on what is already "
                "visible, or open the item you need directly.",
            )
        return ActionResult(True, f"scroll({direction})", f"scrolled {delta}px")
    except Exception:
        pass
    try:
        await page.mouse.wheel(0, dy)
        await page.wait_for_timeout(300)
        return ActionResult(True, f"scroll({direction})", f"scrolled {dy}px")
    except Exception as e:
        return ActionResult(False, f"scroll({direction})", f"error: {e}")


# W_blank_navigation_destroys_the_task (release gate จับได้ 2026-09-07, งาน MiniWoB
# "click-checkboxes"): หลังโดนบังคับ go_back หน้าเว็บกลายเป็นหน้าว่าง โมเดลจึงสั่ง goto ด้วย url
# ว่าง ซึ่งเดิมพาไป about:blank แล้วรายงานว่า "navigated to" สำเร็จ — จากจุดนั้นทุก action ที่
# เหลือล้มหมด (read_page_data: "no element matching 'body'") task กู้ตัวเองไม่ได้อีกเลย
# ไปต่อไม่ได้แต่ยังเผา step จนหมดงบ
#
# ไม่มีกรณีไหนที่การไปหน้าว่างเป็นความตั้งใจของ goal — ปฏิเสธตรงๆ ดีกว่าปล่อยให้ทำลาย context
# ของ task ทิ้ง (หลักเดียวกับ check_fill_is_empty_noop: ปฏิเสธ action ที่ไม่มีทางให้ผลที่ต้องการ)
_BLANK_URLS = ("", "about:blank", "about:", "blank")


async def goto(page: Page, url: str, timeout: int = 15000) -> ActionResult:
    """เปิด URL ใหม่"""
    if (url or "").strip().lower() in _BLANK_URLS:
        return ActionResult(
            False, "goto",
            "[Rejected] that is a blank page, not a destination — navigating there would throw "
            "away the page this task is working on and every later action would fail. If you "
            "need to start over, goto the task's original URL; otherwise read the indexed "
            "elements on the current page and continue from there",
        )
    try:
        await page.goto(url, timeout=timeout)
        return ActionResult(True, "goto", f"navigated to {url}")
    except Exception as e:
        return ActionResult(False, "goto", f"error: {e}")


async def go_back(page: Page) -> ActionResult:
    """ย้อนกลับหน้าก่อนหน้า

    W_blank_navigation_destroys_the_task: Playwright คืน None เฉยๆ (ไม่ throw) เมื่อไม่มี
    ประวัติให้ย้อน เดิมจึงรายงาน "went back successfully" ทุกครั้งแม้ไม่ได้ไปไหนเลยหรือหลุดไป
    หน้าว่าง — recovery ที่ล้มเหลวถูกนับเป็นสำเร็จ แล้ว loop ก็เดินต่อบนหน้าที่ใช้อะไรไม่ได้
    เทียบ URL ก่อน/หลังแล้วรายงานตามความจริง"""
    before = page.url
    try:
        await page.go_back()
    except Exception as e:
        return ActionResult(False, "go_back", f"error: {e}")
    after = page.url
    # เช็ค "ไม่ได้ขยับ" ก่อนเสมอ — หน้าที่สร้างด้วย set_content มี url เป็น about:blank อยู่แล้ว
    # ทั้งที่เนื้อหาปกติดี ถ้าเช็คหน้าว่างก่อนจะรายงานผิดว่า "หน้าหายไปแล้ว"
    if after == before:
        return ActionResult(False, "go_back", "there was no previous page to go back to")
    if (after or "").strip().lower() in _BLANK_URLS:
        return ActionResult(
            False, "go_back",
            "going back left a blank page — the page this task was working on is gone. "
            "Navigate to the task's URL again to carry on",
        )
    return ActionResult(True, "go_back", "went back successfully")


async def switch_tab(page: Page, tab_index: int) -> ActionResult:
    """สลับไป tab อื่นในหน้าต่างเดียวกัน (บาง action เปิด tab ใหม่)"""
    try:
        pages = page.context.pages
        if tab_index >= len(pages):
            return ActionResult(False, f"switch_tab({tab_index})", f"there are only {len(pages)} tab")
        await pages[tab_index].bring_to_front()
        return ActionResult(True, f"switch_tab({tab_index})", "switched tab successfully")
    except Exception as e:
        return ActionResult(False, f"switch_tab({tab_index})", f"error: {e}")


# W19 (latency): ลดจาก 8000ms เดิม — "networkidle" ไม่มีวัน resolve เร็วเลยบนเว็บที่มี
# analytics/polling/websocket ยิงต่อเนื่องตลอด (พบได้บ่อยมาก) ทำให้ page-changing action
# ทุกตัว (click/goto/select/go_back/press_key — ดู orchestrator.py::_PAGE_CHANGING_ACTIONS)
# ต้องรอเต็ม timeout เปล่าๆ ก่อนไป step ถัดไปเสมอบนเว็บพวกนี้ — timeout ที่นี่ไม่เคยทำให้
# task fail อยู่แล้ว (PWTimeout ด้านล่างถูกจับเป็น success เสมอ) แค่กำหนด "เพดานเวลาสูงสุดที่
# ยอมเสียไปเปล่าๆ" ต่อ 1 การเรียก ลดเพดานนี้ลงตรงๆ คือวิธีลด latency ที่ปลอดภัยที่สุด
# (ไม่กระทบ correctness เลย เพราะพฤติกรรม "ไม่ fail" เหมือนเดิมทุกประการ)
async def wait_stable(page: Page, timeout: int = 4000) -> ActionResult:
    """รอให้หน้าเว็บนิ่ง — เรียกหลังทุก action ที่ทำให้หน้าเปลี่ยน ก่อน snapshot รอบใหม่"""
    try:
        await page.wait_for_load_state("networkidle", timeout=timeout)
        return ActionResult(True, "wait_stable", "the page is stable")
    except PWTimeout:
        # ไม่ถือเป็น fail ร้ายแรง — บางหน้ามี network ยิงตลอด
        return ActionResult(True, "wait_stable", "timed out, but it is safe to continue")
    except Exception as e:
        return ActionResult(False, "wait_stable", f"error: {e}")


# W45: อ่านเนื้อหาบนหน้าเว็บตอบคำถามที่ perception.py::get_snapshot() (กรองเฉพาะ element
# คลิกได้) ตอบไม่ได้เลย — แยกเป็น 2 lane คนละแบบเพราะต้นทุน token ต่างกันมาก: Lane 1 นับ/
# lookup ตรงไปตรงมา (count_elements — deterministic, token แทบเป็นศูนย์) vs Lane 2 อ่าน/
# สรุปเนื้อหาที่ต้องตีความ (extract_table_data — LLM ต้องอ่านผลลัพธ์ไปตีความเองต่อ) —
# ตัดสินใจเลือก lane ที่นี่ (ระดับโค้ด ไม่ใช่ให้ LLM สั่งตรงๆ ว่าจะเรียกตัวไหน) เพราะ query
# ที่ตอบได้ด้วยการนับล้วนๆ ควรนับตรงๆ เสมอ ถูกกว่าให้ LLM อ่านตารางทั้งก้อนมานับเอง — เป็น
# ชั้นสำรองระดับโค้ดคู่กับกติกาเดียวกันใน llm.py::SYSTEM_PROMPT (defense-in-depth เหมือน
# pattern อื่นในไฟล์นี้ เช่น RISKY_LABEL_KEYWORDS — ไม่พึ่ง LLM เลือกถูกเพียงอย่างเดียว)
_COUNT_QUERY_KEYWORDS = ("กี่", "จำนวน", "นับ", "how many", "count", "number of")

# W_deterministic_count (บั๊กจริง live-reproduce บน OrangeHRM 2026-08-26, ต่อจาก
# W_confident_zero): หลังแก้ให้ read_page_data คืน "ข้อมูลจริง" แทน "0 ที่ฟังดูน่าเชื่อถือ"
# แล้ว โมเดลอ่านตารางถูกต้องแต่ยัง "นับด้วยตา" ผิด — ตารางมี 7 แถวที่เป็น ESS แต่ตอบ 6
# ไม่มี guard ตัวไหนในระบบจับได้เลย เพราะทุก guard ที่มีตรวจ "ทำ action สำเร็จไหม" ไม่ใช่
# "ตัวเลขในคำตอบถูกไหม"
#
# การนับเป็นงาน deterministic 100% ไม่มีเหตุผลให้โมเดลทำเอง — เมื่อ query เป็นคำถามเชิงนับ
# ให้โค้ดนับจากข้อมูลที่ดึงมาได้จริงแล้วแนบตัวเลขไปด้วย (pattern เดียวกับ guard อื่นในไฟล์นี้:
# ไม่พึ่ง LLM compliance เพียงอย่างเดียว)
#
# ตั้งใจ conservative เรื่อง "นับเฉพาะที่ตรงเงื่อนไข": ดึงเงื่อนไขจาก query เฉพาะรูปแบบ
# "key=value"/"key = value" ที่ชัดเจนเท่านั้น (ตรงกับที่ user พิมพ์จริง เช่น "userrole=ess",
# "Role = ESS") ไม่พยายามเดาจากคำทั่วไปในประโยค เพราะคำอย่าง "user"/"role" โผล่ในทุกแถว
# อยู่แล้ว จะได้ตัวเลขที่ไม่มีความหมายแล้วทำให้โมเดลสับสนหนักกว่าเดิม
# W_column_aware_count: จับ *ทั้งสองฝั่ง* ของ "=" (เดิมทิ้งฝั่งซ้ายไปเลย เก็บแต่ค่า) — ฝั่ง
# ซ้ายคือชื่อคอลัมน์ที่ user ตั้งใจกรอง ("userrole=ess" = คอลัมน์ User Role ไม่ใช่ "แถวไหนก็ได้
# ที่มีคำว่า ess") ทิ้งไปแล้วนับผิดจริง: ตารางที่มี username "ess.irhrg0" ซึ่ง Role เป็น Admin
# จะถูกนับเป็น ESS ด้วย ทั้งที่ไม่ใช่ — ดู _matching_entry_count() ด้านล่าง
_KEY_VALUE_IN_QUERY_RE = re.compile(r"([\w฀-๿]+)\s*=\s*([\w.\-@]+)")
_MARKDOWN_SEPARATOR_CELL_RE = re.compile(r"^:?-{2,}:?$")


def _extracted_entries(data: str) -> list[str]:
    """แปลงผลลัพธ์ของ extract_table_data() กลับเป็น "รายการต่อ entry" เพื่อนับ

    รองรับ 2 รูปแบบที่ extract_table_data() คืนได้จริง — markdown table (`| a | b |`) และ
    JSON list (`["x", "y"]` ซึ่งอาจมีบรรทัด annotation นำหน้าตอน fuzzy match) — คืน [] ถ้า
    parse ไม่ได้ ผู้เรียกจะข้ามการนับไปเฉยๆ (ไม่มีตัวเลขดีกว่าตัวเลขผิด)"""
    table_lines = [ln.strip() for ln in data.splitlines() if ln.strip().startswith("|")]
    if table_lines:
        rows = []
        for line in table_lines:
            cells = [c.strip() for c in line.strip("|").split("|")]
            if cells and all(_MARKDOWN_SEPARATOR_CELL_RE.match(c) for c in cells if c):
                continue  # บรรทัดคั่น header ของ markdown
            rows.append(line)
        return rows[1:] if len(rows) > 1 else []  # แถวแรกคือ header

    start = data.find("[")
    if start == -1:
        return []
    try:
        parsed = json.loads(data[start:])
    except (json.JSONDecodeError, ValueError):
        return []
    return [str(item) for item in parsed] if isinstance(parsed, list) else []


# W_count_answer_check: อ่านตัวเลขที่ _deterministic_count_note() ด้านล่างเพิ่งเขียนลงใน
# ข้อความผลลัพธ์กลับออกมา — ตั้งใจวางไว้ *ติดกัน* กับฟังก์ชันที่สร้างข้อความนั้น เพราะทั้งสอง
# ต้องเปลี่ยนพร้อมกันเสมอถ้ารูปแบบข้อความเปลี่ยน (กับดักเดียวกับที่ dom_locator.py/
# site_learning/extractor.py เตือนไว้ว่า "แก้ทั้งคู่หรือไม่แก้เลย" — ที่นี่แก้ด้วยการวางชิดกัน
# แทนที่จะให้ orchestrator ไปเดา format เอาเองอีกไฟล์หนึ่ง)
_SYSTEM_COUNT_IN_RESULT_RE = re.compile(
    r"\[counted by the system\] (\d+) of those \d+ entries contain '([^']+)'"
)


def system_counted_conditions(message: str) -> dict[str, int]:
    """W_count_answer_check: คืน {ค่าเงื่อนไข: จำนวนที่โค้ดนับได้} จากข้อความผลลัพธ์ของ
    read_page_data — {} ถ้าไม่มีบรรทัดที่โค้ดนับเองอยู่เลย (action อื่นทั้งหมด)"""
    return {
        value.strip().lower(): int(count)
        for count, value in _SYSTEM_COUNT_IN_RESULT_RE.findall(message or "")
    }


def _extracted_table_cells(data: str) -> tuple[list[str], list[list[str]]]:
    """W_column_aware_count: แยก markdown table ที่ extract_table_data() คืนมาเป็น
    (เซลล์หัวตาราง, แถวข้อมูลแบบแยกเซลล์) — คืน ([], []) ถ้าไม่ใช่ตาราง (เช่น JSON list ของ
    รายการสินค้า ซึ่งไม่มีคอลัมน์ให้เล็งอยู่แล้ว) ผู้เรียกจะ fallback ไปนับทั้งแถวแทน"""
    table_lines = [ln.strip() for ln in data.splitlines() if ln.strip().startswith("|")]
    parsed = []
    for line in table_lines:
        cells = [c.strip() for c in line.strip("|").split("|")]
        if cells and all(_MARKDOWN_SEPARATOR_CELL_RE.match(c) for c in cells if c):
            continue
        parsed.append(cells)
    if len(parsed) < 2:
        return [], []
    return parsed[0], parsed[1:]


def _normalize_column_name(text: str) -> str:
    """W_column_aware_count: "User Role" / "user_role" / "userrole" ต้องเทียบกันติด — ตัด
    อักขระที่ไม่ใช่ตัวอักษร/ตัวเลขทิ้งทั้งหมดแล้ว lowercase (user พิมพ์ชื่อคอลัมน์ในรูปแบบไหน
    ก็ได้ ไม่มีทางบังคับให้ตรงกับหัวตารางเป๊ะๆ)"""
    return re.sub(r"[^a-z0-9ก-๙]", "", (text or "").lower())


def _matching_entry_count(
    entries: list[str], header: list[str], rows: list[list[str]], key: str, value: str,
) -> tuple[int, str]:
    """W_column_aware_count: คืน (จำนวนแถวที่ตรงเงื่อนไข, คำอธิบายว่านับจากตรงไหน)

    ถ้าหัวตารางมีคอลัมน์ที่ชื่อตรงกับฝั่งซ้ายของ "=" ให้เทียบเฉพาะเซลล์ในคอลัมน์นั้น — ไม่งั้น
    fallback ไปเทียบทั้งแถวแบบเดิม (ตารางที่ไม่มีหัว/JSON list/ชื่อคอลัมน์ที่เดาไม่ตรง)

    ตัวอย่างที่ fallback เดิมนับผิดจริง: แถว "| ess.irhrg0 | Admin | Enabled |" ถูกนับเป็น
    userrole=ess ด้วย เพราะคำว่า ess อยู่ในคอลัมน์ Username ไม่ใช่ User Role"""
    needle = value.lower()
    target = _normalize_column_name(key)
    if target and rows:
        for column, name in enumerate(header):
            if _normalize_column_name(name) != target:
                continue
            matching = sum(
                1 for row in rows if column < len(row) and needle in row[column].lower()
            )
            return matching, f"in the '{name.strip()}' column"
    return sum(1 for entry in entries if needle in entry.lower()), "anywhere in the row"


def _deterministic_count_note(data: str, query: str) -> str:
    """คืนบรรทัดสรุปจำนวนที่โค้ดนับเองจาก data — คืน "" ถ้านับไม่ได้ (ดู _extracted_entries)"""
    entries = _extracted_entries(data)
    if not entries:
        return ""
    total = len(entries)
    notes = [
        f"[counted by the system, not by you] the data below contains exactly {total} entries. "
        "Use this number — do not recount the rows yourself."
    ]
    header, rows = _extracted_table_cells(data)
    seen: set[str] = set()
    for key, value in _KEY_VALUE_IN_QUERY_RE.findall(query):
        needle = value.strip().lower()
        if not needle or needle in seen:
            continue
        seen.add(needle)
        matching, where = _matching_entry_count(entries, header, rows, key, value.strip())
        if matching:
            notes.append(
                f"[counted by the system] {matching} of those {total} entries contain "
                f"'{value.strip()}' {where}."
            )
        else:
            # W_conditional_count: เดิมเงียบไปเลยตอน matching เป็น 0 ซึ่งทิ้งให้โมเดลเห็นแต่
            # บรรทัด "exactly N entries" แล้วรายงาน N เป็นคำตอบของคำถามที่มีเงื่อนไข — ผิด
            # คนละเรื่องกันเลย ต้องบอกตามจริงว่านับได้ 0 แต่ต้องพ่วงเงื่อนไขของ W_confident_zero
            # ไว้ด้วยเสมอ (0 ไม่ใช่คำตอบจนกว่าจะพิสูจน์ได้ว่าอ่านตารางถูกตัว)
            notes.append(
                f"[counted by the system] 0 of those {total} entries contain "
                f"'{value.strip()}' {where}. If that contradicts what you can see on the page, the data "
                "below is not the right table — read it again with a different target_hint "
                "instead of reporting 0."
            )
    return "\n".join(notes) + "\n"


async def read_page_data(page: Page, query: str, target_hint: str) -> ActionResult:
    """query: คำถามที่ต้องการคำตอบ — ใช้ตัดสินใจเลือก lane ด้านล่าง (นับ vs อ่านตาราง) และ
    ถ้าไม่ใช่คำถามเชิงนับ ยังถูกส่งต่อเข้า extract_table_data() เป็นค่าที่จะ lookup ในแถว/
    รายการด้วย (exact match ก่อน ไม่เจอค่อย fuzzy_find — ดู perception.py) ไม่ถูกส่งเข้า LLM
    เพิ่มรอบใหม่เอง — target_hint: CSS selector ที่คาดว่าตรงกับ element/แถวตาราง/รายการที่มี
    ข้อมูลนั้นจริง (LLM เป็นคนเดามาจาก indexed elements/URL ที่เห็น)

    ไม่ผ่าน _dispatch_with_retry เหมือน click/fill — query ที่ตอบไม่ได้เพราะ target_hint
    ไม่ตรงกับอะไรเลยเป็น deterministic mismatch (เหมือนกันทุกครั้ง) ไม่ใช่ timing issue
    ที่ retry แล้วจะเปลี่ยนผลลัพธ์

    รอหน้านิ่งก่อนอ่านเสมอ (wait_stable) — ถ้าเพิ่ง edit/submit ข้อมูลในตารางไปเมื่อ step
    ก่อนหน้า (เช่น React re-render/re-fetch ข้อมูลใหม่หลัง submit form) DOM อาจยังไม่ bind
    ค่าใหม่เสร็จตอนที่ LLM สั่ง read_page_data ตามมาติดๆ กัน อ่านทันทีอาจได้ข้อมูลเก่า/ว่าง
    เปล่า — wait_stable() timeout แล้วเดินต่อได้เสมอ (ไม่ throw) ไม่ทำให้ query ที่หน้านิ่ง
    อยู่แล้วช้าลงมาก"""
    if not target_hint:
        return ActionResult(False, "read_page_data", "target_hint (a CSS selector) is required")

    await wait_stable(page)

    is_count_query = any(kw in query.lower() for kw in _COUNT_QUERY_KEYWORDS)
    try:
        if is_count_query:
            # W_conditional_count (ช่องที่ W_deterministic_count ยังปิดไม่ถึง — พิสูจน์ซ้ำได้
            # 2026-08-26): W_deterministic_count แนบตัวเลขที่โค้ดนับให้เฉพาะ "เส้นทางสำรอง"
            # (count_elements คืน 0 หรือ extract คืน [FAIL]) เท่านั้น ส่วนเส้นทางหลักของคำถาม
            # เชิงนับ — count_elements คืนค่ามากกว่า 0 — คืน "found N entries matching
            # '<selector>'" ดิบๆ โดยไม่รู้จักเงื่อนไขใน query เลยสักนิด
            #
            # ผลคือคำถาม "มี user ที่ userrole=ess กี่คน" + target_hint '[role="row"]' คืน
            # "found 21 entries" (ทุกแถวรวมหัวตาราง) ทั้งที่คำตอบจริงคือ 4 — success=True
            # ด้วย จึงไม่มีสัญญาณอะไรให้ใครจับได้เลย เป็น "ตอบผิดแบบมั่นใจ" บนเส้นทางที่ใช้
            # บ่อยที่สุดของ lane นี้ (คำถามเชิงนับเกือบทั้งหมดเข้าทางนี้)
            #
            # เมื่อ query มีเงื่อนไขแบบ key=value ชัดเจน การนับ element ดิบๆ ตอบคำถามนั้นไม่ได้
            # ตามนิยาม — ต้องอ่านแถวจริงแล้วนับเฉพาะที่ตรงเงื่อนไข (ตัวนับเดียวกับ
            # _deterministic_count_note ไม่ได้เขียนตรรกะการนับขึ้นใหม่)
            condition_values = [v.strip() for _, v in _KEY_VALUE_IN_QUERY_RE.findall(query) if v.strip()]
            if condition_values:
                rows = await extract_table_data(page, target_hint, "")
                if not rows.startswith("[FAIL]"):
                    note = _deterministic_count_note(rows, query)
                    if note:
                        return ActionResult(True, "read_page_data", f"{note}{rows}")

            count = await count_elements(page, target_hint)
            if count > 0:
                if condition_values:
                    # อ่านแถวจริงไม่ได้ (ไม่มีตาราง/parse ไม่ออก) แต่ selector ยังนับได้ —
                    # รายงานตามความจริงว่านี่คือ "จำนวน element ที่ตรง selector" ไม่ใช่
                    # "จำนวนรายการที่ตรงเงื่อนไข" แทนที่จะปล่อยตัวเลขที่ตอบคนละคำถามออกไป
                    # เฉยๆ (ธีมเดียวกับ W_click_native_select/W_confident_zero)
                    condition_text = " + ".join(repr(v) for v in condition_values)
                    return ActionResult(
                        True, "read_page_data",
                        f"the selector '{target_hint}' matches {count} elements, but that is the "
                        f"raw element count — it is NOT the number of entries matching "
                        f"{condition_text}, and the rows themselves could not be read to check. "
                        f"Do NOT report {count} as the answer. Read the table with a different "
                        "target_hint (for sites that build tables out of <div>, try "
                        "role=row) so the matching rows can actually be counted.",
                    )
                return ActionResult(True, "read_page_data", f"found {count} entries matching '{target_hint}'")
            # W_confident_zero (บั๊กจริง live-reproduce บน OrangeHRM 2026-08-26): count_elements()
            # คืน 0 ทั้งกรณี "มีศูนย์รายการจริง" และกรณี "selector ไม่ตรงอะไรเลยบนหน้านี้" —
            # เดิมทั้งสองกรณีคืน success=True พร้อมข้อความ "found 0 entries" ซึ่งอ่านเหมือน
            # ข้อเท็จจริงที่ยืนยันแล้ว โมเดลจึงปิดงานด้วยคำตอบ "0 รายการ" อย่างมั่นใจ
            #
            # เหตุการณ์จริง: goal ถามจำนวน user ที่ Role=ESS บน OrangeHRM โมเดลเดา
            # target_hint="table tbody tr" แต่ OrangeHRM ไม่มี <table> จริงเลย (เป็น ARIA grid
            # ด้วย div[role=row]) -> count=0 -> ตอบว่า "เจอ 0 รายการ" ทั้งที่ความจริงมี 5 —
            # ตอบผิดแบบมั่นใจ อันตรายกว่าตอบว่าทำไม่ได้มาก เพราะไม่มีสัญญาณให้ใครจับได้เลย
            #
            # แก้: 0 ไม่ใช่คำตอบจนกว่าจะพิสูจน์ได้ — ลอง extract_table_data() ด้วย hint เดิมก่อน
            # (ตัวนั้นมี fallback ครบ: querySelectorAll หลายตัว, ARIA grid, <table>/<ul> ทั้งหน้า)
            # ถ้ามันอ่านข้อมูลได้จริง แปลว่า count=0 มาจาก selector ผิด ไม่ใช่ของจริง
            fallback = await extract_table_data(page, target_hint, "")
            if not fallback.startswith("[FAIL]"):
                # W_deterministic_count: แนบตัวเลขที่โค้ดนับเองไปด้วยเสมอ — เดิมบอกให้โมเดล
                # "นับเอาเองจากข้อมูลข้างล่าง" ซึ่งมันนับผิดจริง (7 แถว ESS ตอบ 6)
                return ActionResult(
                    True, "read_page_data",
                    f"the selector '{target_hint}' matched 0 elements directly, so counting it "
                    "would have been wrong. Here is the data actually found on this page "
                    f"instead:\n{_deterministic_count_note(fallback, query)}{fallback}",
                )
            return ActionResult(
                False, "read_page_data",
                f"the selector '{target_hint}' matched 0 elements on this page, and no table or "
                "list could be read from it either. This means the selector is wrong — it does "
                "NOT mean the answer is zero. Do not report 0 as the answer. Pick a different "
                "target_hint based on what you can see on the page (for sites that build tables "
                "out of <div> instead of <table>, try '[role=\"row\"]').",
            )
        data = await extract_table_data(page, target_hint, query)
        if data.startswith("[FAIL]"):
            # W_query_is_a_question (เจอจาก step trace ตัวใหม่: read_page_data ล้มติดกัน 3 step
            # บน saucedemo ก่อนจะบังเอิญสำเร็จ): extract_table_data() ตีความ query ว่าเป็น
            # "ค่าที่ต้องหาให้เจอในตาราง" (เช่นชื่อคน) แล้วคืน [FAIL] ถ้าหาไม่เจอ — แต่โมเดล
            # ส่งคำถามภาษาธรรมชาติมาเป็นปกติ ("first product name", "how many rows...") ซึ่ง
            # ไม่มีวันปรากฏเป็นข้อความในตารางอยู่แล้ว ผลคือ [FAIL] ทั้งที่ดึงข้อมูลมาได้ครบ
            # แล้วจริงๆ แล้วทิ้งข้อมูลนั้นไปเปล่าๆ โมเดลก็เดา target_hint ใหม่วนไปเรื่อยๆ
            #
            # ลองอีกรอบแบบไม่ส่ง query (= ขอข้อมูลเฉยๆ) ถ้าได้ข้อมูลจริงก็คืนไปให้ตอบเอง
            # พร้อมบอกตรงๆ ว่าไม่เจอข้อความนั้นแบบตรงตัว — รักษาเจตนาเดิมของ W46 (ห้ามแกล้ง
            # ทำเป็นเจอ) ไว้ครบ แค่ไม่ทิ้งข้อมูลที่อ่านมาได้แล้ว
            without_query = await extract_table_data(page, target_hint, "")
            if without_query.startswith("[FAIL]"):
                return ActionResult(False, "read_page_data", data)
            return ActionResult(
                True, "read_page_data",
                f"no cell matched '{query}' verbatim, so treat that as a question to answer "
                "from the data below rather than as a value that exists on the page. If the "
                f"answer genuinely is not in here, say so.\n"
                f"{_deterministic_count_note(without_query, query)}{without_query}",
            )
        # W_deterministic_count: goal ที่ถามจำนวนแต่ hint ตรงพอดี (count lane ไม่ทำงาน เพราะ
        # query ไม่มีคำเชิงนับ) ยังต้องได้ตัวเลขที่นับด้วยโค้ดเหมือนกัน — แนบเฉพาะตอนนับได้จริง
        return ActionResult(True, "read_page_data", f"{_deterministic_count_note(data, query)}{data}")
    except Exception as e:
        return ActionResult(False, "read_page_data", f"error: {e}")


# ------------------------------------------------------------
# Permission layer: กัน action เสี่ยง/บล็อก ก่อนถึง dispatch จริง
# adapted จาก PR "permission-ab" — จุดต่างจาก PR เดิม: ask_user_func ถูกใช้งานจริง
# (PR เดิมรับ param นี้มาแต่ไม่ได้เรียกใช้เลย ยังเรียก input() ตรงๆ เสมอ)
# ------------------------------------------------------------

# ข้อความนี้เป็น "สัญญาณ" เดียวที่บ่งบอกว่า action ถูกมนุษย์ปฏิเสธจริง (ต่างจาก failure
# ทั่วไป เช่น timeout/index ผิด/BLOCKED) — ดึงเป็น constant แยกแทนที่จะ hardcode ข้อความ
# ซ้ำในหลายที่ เพราะ memory.py::ShortTermMemory ต้องเช็คข้อความนี้เพื่อแยก "refusal"
# ออกจาก failure อื่นๆ (ดู rejected_actions_summary())
REJECTED_BY_USER_MESSAGE = "The user refused to perform this action (human-in-the-loop)"


async def _confirm_action(cmd: dict, ask_user_func: Optional[AskUserFunc], label: str = "") -> bool:
    """label: ชื่อ element เป้าหมายจริงบนหน้าเว็บ (เช่น "Place Order") — แนบเข้า cmd
    สำเนา (ไม่แตะ cmd ตัวจริงที่ยังต้องใช้ dispatch ต่อ) ภายใต้ key "element_label" แยก
    จาก key "label" เดิมของ cmd (ซึ่งมีความหมายอื่นสำหรับ action type="select" คือ
    ตัวเลือกที่จะเลือก ไม่ใช่ชื่อ element) ให้ชั้นบน (API server -> UI) โชว์ "กดปุ่มอะไร"
    เป็นชื่อจริงแทน index เปล่าๆ ในการ์ด permission prompt — เหมือน pattern เดียวกับที่
    orchestrator.py แนบ "label" คู่กับ "cmd" เข้า history/event อยู่แล้ว (ดู W10[D])"""
    if ask_user_func is not None:
        confirm_cmd = dict(cmd, element_label=label) if label else cmd
        return bool(await ask_user_func(confirm_cmd))
    print(f"\n[HUMAN-IN-THE-LOOP] The agent wants to run a risky command: {cmd}", flush=True)
    # ใช้ asyncio.to_thread เพื่อให้รับ input() ได้โดยไม่บล็อก async event loop หลัก
    choice = await asyncio.to_thread(input, "Do you want to allow this action? (y/n): ")
    approved = choice.strip().lower() in ("y", "yes")
    if approved:
        print("[APPROVED] proceeding...", flush=True)
    return approved


# ------------------------------------------------------------
# ทางเข้าเดียวสำหรับ W4: agent ส่ง action มาเป็น dict แล้ว dispatch
# ------------------------------------------------------------
async def _check_permission(
    cmd: dict, ask_user_func: Optional[AskUserFunc], label: str, manual_guidance: str,
    allowed_domains: Optional[set], element_tag: str, element_type: str,
) -> Optional[str]:
    """W_chain ("Compound Actions" — ลด step ของ form/list task): เดิมเช็ค permission
    inline อยู่ที่หัว execute() เท่านั้น (ครั้งเดียวต่อ cmd) — แยกเป็นฟังก์ชันย่อยเพื่อเรียก
    ซ้ำได้กับ "action ที่สอง" ที่ chain ต่อท้ายใน cmd เดียวกัน (ดู then_click_index ด้านล่าง)
    โดยไม่ต้อง duplicate logic — คืน None ถ้าอนุญาตให้ทำต่อได้ (SAFE หรือ NEEDS_CONFIRMATION
    ที่ approve แล้ว) หรือคืนข้อความ error ถ้าถูกบล็อก/ถูกปฏิเสธ (ให้ผู้เรียกคืน ActionResult
    ที่ fail เอง — ฟังก์ชันนี้ไม่รู้จัก "action" string ของ ActionResult)"""
    risk = classify_action(
        cmd, label=label, manual_guidance=manual_guidance, allowed_domains=allowed_domains,
        element_tag=element_tag, element_type=element_type,
    )
    if risk == ActionRisk.BLOCKED:
        return "Action blocked by the security layer (blocklist)"
    if risk == ActionRisk.NEEDS_CONFIRMATION:
        approved = await _confirm_action(cmd, ask_user_func, label)
        if not approved:
            return REJECTED_BY_USER_MESSAGE
    return None


async def _maybe_chain_click(
    page: Page, cmd: dict, primary: ActionResult, ask_user_func: Optional[AskUserFunc],
    manual_guidance: str, allowed_domains: Optional[set],
    then_label: str, then_tag: str, then_type: str,
) -> ActionResult:
    """W_chain: ถ้า cmd มี "then_click_index" (LLM ขอคลิก element ที่สองต่อทันทีในคำสั่ง
    เดียวกัน — ใช้กับ fill/select/check ที่ตามด้วยปุ่ม Submit/OK ที่ "เห็นอยู่แล้ว" ในหน้า
    เดิม ไม่ต้อง perceive ใหม่ก่อน) ให้ dispatch คลิกที่สองต่อทันที รวมเป็น ActionResult
    เดียว ลด LLM round-trip จาก 2-3 step เหลือ 1 step ต่อ interaction pattern แบบนี้ — คู่กับ
    fill โดยเฉพาะ นี่คือทางเลือกที่ "เชื่อถือได้กว่า" fill+key:"Enter" เสมอเมื่อเห็นปุ่ม
    submit จริงในหน้า เพราะบางเว็บไม่มี Enter-to-submit เลย (ดู docstring จุดเรียกใน
    execute() ส่วน fill — ยืนยันจากการทดสอบจริงว่า fill() เขียนค่าถูกต้องเสมอ ปัญหาที่เจอ
    จริงคือ Enter บางหน้าไม่มีผลอะไรเลย ไม่ใช่ fill() พังหรือ event มาไม่ทัน)

    ไม่ chain เลยถ้า primary action เอง fail (ไม่มีเหตุผลจะคลิกต่อถ้าขั้นแรกยังไม่สำเร็จ) หรือ
    cmd ไม่มี then_click_index มาเลย (คืน primary เดิมตรงๆ ไม่กระทบ caller เดิมที่ไม่เคยใช้
    ฟีเจอร์นี้แม้แต่นิดเดียว)

    ความปลอดภัย: action ที่สองยังผ่าน classify_action()/ask_user_func เต็มรูปแบบเหมือน
    action เดี่ยวๆ ทุกประการ (เรียก _check_permission() ซ้ำ ไม่ได้ auto-approve เพราะเป็น
    action ที่สอง) — ถ้าต้องขออนุมัติ/ถูกบล็อก จะไม่ทำต่อ แค่คืนผลของ primary เฉยๆ (ความคืบ
    หน้าที่เกิดขึ้นจริงแล้วไม่หายไป) พร้อมข้อความบอกให้สั่ง click ที่สองแยกเป็น step ถัดไปแทน
    — ไม่มีทางข้าม human-in-the-loop ผ่านช่องทางนี้ได้เลย

    manual_guidance/allowed_domains ใช้ค่าเดียวกับที่ primary action ได้รับ (ไม่ query RAG
    ซ้ำรอบสองสำหรับ target ที่สอง — เสีย latency ที่เพิ่งลดไปกลับคืนหมด ขัดจุดประสงค์ของ
    feature นี้เอง) ยอมรับว่าอาจไม่ specific เท่าที่ควรสำหรับ target ที่สอง แต่ปลอดภัยกว่า
    ไม่มี guidance เลย"""
    then_index = cmd.get("then_click_index")
    # W_chain_partial_success: index ติดลบไม่มีทางเป็น element จริง — บาง provider ส่ง -1 มาเป็น
    # sentinel แทน "ไม่มี chain" (ดู llm._normalize_openai_args) กรองที่นี่อีกชั้นให้ทุก provider
    # ไม่ใช่แค่ตัวที่รู้จัก ไม่งั้นเสีย retry ของ _dispatch_click_with_retry() 3 รอบเปล่าๆ ทุกครั้ง
    if then_index is None or (isinstance(then_index, int) and then_index < 0) or not primary.success:
        return primary

    synthetic_cmd = {"type": "click", "index": then_index}
    denial = await _check_permission(
        synthetic_cmd, ask_user_func, then_label, manual_guidance, allowed_domains, then_tag, then_type,
    )
    if denial is not None:
        return ActionResult(
            primary.success,
            primary.action,
            f"{primary.message} (did not go on to click index {then_index}: {denial} — issue that click as a separate next step instead)",
            locator_descriptor=primary.locator_descriptor, toast_confirmed=primary.toast_confirmed,
        )

    second = await _dispatch_click_with_retry(page, then_index, then_label)
    if second.success:
        return ActionResult(
            True,
            primary.action,
            f"{primary.message} + then click({then_index}): {second.message}",
            locator_descriptor=primary.locator_descriptor,
            toast_confirmed=primary.toast_confirmed or second.toast_confirmed,
        )
    # W_chain_partial_success (บั๊กจริง live-reproduce บน OrangeHRM กับ provider openai):
    # เดิมคืน primary.success and second.success — primary ที่ "สำเร็จจริงและเปลี่ยนหน้าไปแล้ว"
    # ถูกรายงานรวมเป็น [FAIL] เพราะ chain ตัวที่สองพลาด โมเดลอ่านว่าล้มเหลวทั้งก้อนแล้ว "ลอง
    # คลิก primary ซ้ำ" รอบแล้วรอบเล่า (เห็นจริง 10 step ติดกัน: click(3) สำเร็จทุกครั้ง แต่
    # then_click_index ที่ค้างจาก snapshot ก่อนหน้าพังทุกครั้ง จน task หมด max_steps ทั้งที่
    # ไปถึงหน้าเป้าหมายตั้งแต่ step แรก) — chain ตัวที่สองพลาดเป็นเรื่องปกติมากเพราะ primary
    # มักทำให้หน้าเปลี่ยน แล้ว index ที่โมเดลจำมาจาก snapshot เดิมก็ค้างทันที
    #
    # ยึดหลักเดียวกับ branch permission-denial ด้านบนทุกประการ: ความคืบหน้าที่เกิดขึ้นจริงแล้ว
    # ต้องไม่หายไป คืน primary.success ตามจริง แล้วบอกให้สั่งคลิกตัวที่สองแยกเป็น step ถัดไป
    return ActionResult(
        primary.success,
        primary.action,
        f"{primary.message} — but the chained click({then_index}) failed: {second.message}. "
        "The first action DID succeed; the page has most likely changed, so that second index "
        "is stale. Look at the new indexed elements and issue that click as a separate next "
        "step (do not repeat the first action).",
        locator_descriptor=primary.locator_descriptor,
        toast_confirmed=primary.toast_confirmed or second.toast_confirmed,
    )


async def execute(
    page: Page, cmd: dict, ask_user_func: Optional[AskUserFunc] = None, label: str = "",
    manual_guidance: str = "", allowed_domains: Optional[set] = None, element_tag: str = "",
    element_type: str = "", then_label: str = "", then_tag: str = "", then_type: str = "",
) -> ActionResult:
    """
    รับคำสั่งจาก LLM ในรูป dict เช่น:
        {"type": "fill",   "index": 0, "text": "standard_user"}
        {"type": "click",  "index": 2}
        {"type": "select", "index": 2, "label": "Price (low to high)"}
        {"type": "scroll", "direction": "down"}
        {"type": "goto",   "url": "https://..."}
    แล้ว dispatch ไป action ที่ถูกต้อง — นี่คือจุดที่ W4 จะเรียกใช้

    ก่อน dispatch จริง เช็ค permission ก่อนเสมอ (classify_action จาก
    backend/app/permission/rules.py): BLOCKED -> ปฏิเสธทันทีไม่ถาม, NEEDS_CONFIRMATION ->
    ถาม user ก่อน (ผ่าน ask_user_func ถ้ามี ไม่งั้น fallback เป็น input() ทาง terminal)

    label: ข้อความของ element เป้าหมาย (จาก indexed elements ตอน perceive) — ส่งต่อให้
    classify_action() เช็คคำเสี่ยง (เช่น "Remove") เป็นชั้นสำรองนอกจาก type ล้วนๆ กัน LLM
    ต้องเลือก type=delete/submit/purchase/pay ให้ถูกเองเพียงอย่างเดียว (ดู
    permission/rules.py::RISKY_LABEL_KEYWORDS) ไม่ส่งมาก็ได้ (default "")

    manual_guidance (W7[B]): เนื้อหาคู่มือที่เกี่ยวข้องกับ step นี้ — ส่งต่อให้
    classify_action() เช็คว่าคู่มือระบุไว้ไหมว่า action แบบนี้ต้องขออนุมัติก่อน (ดู
    permission/rules.py::MANUAL_CONFIRMATION_KEYWORDS) ไม่ส่งมาก็ได้ (default "")

    allowed_domains: ส่งต่อให้ classify_action() override ALLOWED_DOMAINS เฉพาะ call
    นี้ (ดู permission/rules.py::classify_action ส่วน allowed_domains) ไม่ส่งมาก็ได้
    (default None = พฤติกรรมเดิม ใช้ ALLOWED_DOMAINS ของ module) — ใช้ตอน orchestrator
    ต่อ agent เข้า browser จริงของ user เพื่อจำกัด goto แค่โดเมนของ task นั้นๆ

    element_tag (W_search follow-up): ชื่อ HTML tag ของ element เป้าหมาย (เช่น "a") —
    ส่งต่อให้ classify_action() เช็คสัญญาณโครงสร้างเป็นชั้นสำรองอีกชั้นนอกจาก label (ดู
    permission/rules.py::ANCHOR_TAG) ไม่ส่งมาก็ได้ (default "")

    element_type (W_search follow-up 2): ค่า attribute "type" ของ element เป้าหมาย
    (เช่น input ที่ type="text"/"search"/"submit") — ส่งต่อให้ classify_action() คู่กับ
    element_tag (ดู permission/rules.py::SAFE_INPUT_TAG/RISKY_INPUT_TYPES) ไม่ส่งมาก็ได้
    (default "")

    then_label/then_tag/then_type (W_chain, "Compound Actions"): label/tag/type ของ
    element ที่ cmd["then_click_index"] ชี้ไป (ถ้ามี — ใช้ได้กับ type="fill"/"select"/
    "check"/"click" ดู _maybe_chain_click() ด้านล่าง)
    เหมือน label/element_tag/element_type ด้านบนทุกประการแค่สำหรับ target ตัวที่สองของ
    action เดียวกัน — orchestrator.py resolve มาจาก elements list เดียวกับที่ resolve
    label/element_tag/element_type ของ action หลัก ไม่ส่งมาก็ได้ (default "" ทั้งคู่ —
    then_click_index ที่ไม่มี label/tag/type แนบมาก็ยังเช็ค permission ได้ปกติ แค่ไม่มี
    สัญญาณเสริมให้ classify_action() ใช้)
    """
    t = cmd.get("type")

    denial = await _check_permission(
        cmd, ask_user_func, label, manual_guidance, allowed_domains, element_tag, element_type,
    )
    if denial is not None:
        return ActionResult(False, f"{t}", denial)

    try:
        # click/fill/select/check ผ่าน retry wrapper (W5) เพราะพังบ่อยจาก DOM ยังไม่นิ่ง
        # ไม่ใช่ index ผิดเสมอไป — scroll/goto/go_back/switch_tab/wait ไม่ retry เพราะ
        # failure mode ต่างกัน (เช่น goto ผิด URL ก็จะผิดซ้ำทุกครั้ง ไม่ใช่เรื่อง timing)
        # click ใช้ _dispatch_click_with_retry() แยกต่างหาก (ไม่ใช่ _dispatch_with_retry()
        # ทั่วไป) เพราะตั้งแต่รอบ retry ที่ 2 เป็นต้นไปต้อง hover() บน element เป้าหมายก่อน
        # คลิกซ้ำเสมอ (ดู hover-to-reveal action button — hover()/_dispatch_click_with_retry()
        # ด้านบน) fill/select/check ไม่ต้องการ hover ก่อนเลย ยังใช้ _dispatch_with_retry()
        # ทั่วไปเหมือนเดิมทุกประการ
        # W19: Deterministic State Filter (ดู core/state_filter.py) — เช็คก่อน dispatch
        # จริงว่า action นี้ "จำเป็นไหม" จากสถานะ DOM ปัจจุบัน (ไม่พึ่ง LLM) เจอว่า
        # redundant ให้ short-circuit ไม่แตะ browser เลย คืน [OK] ทันที (fill/check/scroll
        # = เป้าหมายบรรลุอยู่แล้ว ไม่ใช่ error) ยกเว้น click ที่ disabled ซึ่งคือ action
        # ที่ "ทำไม่ได้จริง" ไม่ใช่ "ทำไปแล้ว" จึงคืน success=False ให้ agent รู้ว่าต้องหา
        # ทางอื่น (เช่น กรอกช่องที่ทำให้ปุ่มนี้ enable ก่อน) แทนที่จะเข้าใจผิดว่าคลิกสำเร็จ
        if t == "click":
            redundant = await state_filter.check_click_redundant(page, cmd["index"])
            if redundant is not None:
                return ActionResult(False, f"click({cmd['index']})", f"[Skipped] {redundant}")
            # W_click_native_select: คลิก <select> จริงคือ no-op ที่คืน [OK] (ดู docstring ของ
            # check_click_target_is_native_select สำหรับ task ที่ตายเพราะเรื่องนี้จริง) —
            # ต้องเช็คก่อน dispatch เพราะหลังคลิกไปแล้วแยกไม่ออกจากคลิกที่ได้ผลจริง
            wrong_kind = await state_filter.check_click_target_is_native_select(page, cmd["index"])
            if wrong_kind is not None:
                return ActionResult(False, f"click({cmd['index']})", f"[Skipped] {wrong_kind}")
            # W_chain_stale_index: ตัด then_click_index ทิ้งถ้าคลิกนี้ทำให้ index ชุดเดิมใช้
            # ไม่ได้ — ทั้ง "เปิด" dropdown (ตัวเลือกเพิ่งเกิด ไม่มี index เดิม) และ "เลือก
            # ตัวเลือก" (ตัวเลือกทั้งชุดหายไป index ที่เหลือเลื่อนหมด) ดู docstring ของ
            # state_filter.check_click_invalidates_indexes สำหรับบั๊กจริงทั้งสองเคส ต้องเช็ค
            # *ก่อน* dispatch เพราะหลังคลิกแล้วสถานะที่ใช้แยกสองเคสนี้ออกจากกันหายไปแล้ว
            #
            # W_dropdown_sets_filter_dirty: เรียกกับ *ทุก* click แล้ว (เดิมเฉพาะตอนมี
            # then_click_index) เพราะ orchestrator ต้องรู้ด้วยว่าคลิกนี้เป็นการเลือกค่าใน
            # dropdown หรือไม่ ไม่ใช่แค่ตอนจะ chain — ผลของ evaluate() ครั้งเดียวใช้ได้ทั้ง
            # สองงาน (timeout สั้น 500ms + fail-safe คืน None อยู่แล้ว)
            disturb_kind = await state_filter.classify_click_index_disturbance(page, cmd["index"])
            stale_chain_note = (
                state_filter.chain_hint_for_kind(disturb_kind)
                if cmd.get("then_click_index") is not None else None
            )
            result = await _dispatch_click_with_retry(page, cmd["index"], label)
            if result.success and disturb_kind == "option":
                result = replace(result, dropdown_option_selected=True)
            # W_menu_open_note_needs_no_chain: คลิกที่ไม่มี chain ก็ต้องรู้ว่า index เลื่อนแล้ว
            if stale_chain_note is None and result.success:
                shift_note = state_filter.index_shift_note_for_kind(disturb_kind)
                if shift_note is not None:
                    result = replace(result, message=f"{result.message} ({shift_note})")
            if stale_chain_note is not None:
                # คลิกหลักสำเร็จจริง (เปิด dropdown/เลือกตัวเลือกได้ตามต้องการ) — รายงานตาม
                # ความจริง แล้วบอก
                # เหตุผลที่ไม่ chain ต่อ pattern เดียวกับ W_chain_partial_success
                return replace(
                    result,
                    message=f"{result.message} (did not go on to the chained click: {stale_chain_note})",
                )
            return await _maybe_chain_click(
                page, cmd, result, ask_user_func, manual_guidance, allowed_domains,
                then_label, then_tag, then_type,
            )
        if t == "fill":
            # W_empty_fill_noop: เช็คก่อน check_fill_redundant() เพราะเคส "ว่าง -> ว่าง"
            # เข้าเงื่อนไข redundant ด้วย (current == text == "") แต่ต้องตอบเป็น failure
            # ไม่ใช่ success — ดู docstring ของ check_fill_is_empty_noop()
            # W_file_input_guard (P3.10): เช็คก่อนทุกอย่าง — fill ลง <input type=file> ไม่มี
            # ทางสำเร็จ ต่อให้ข้อความว่าง/ซ้ำหรือไม่ก็ตาม (ดู state_filter สำหรับเหตุผลที่ไม่
            # เพิ่ม action อัปโหลดไฟล์ให้ agent)
            file_input = await state_filter.check_fill_target_is_file_input(page, cmd["index"])
            if file_input is not None:
                return ActionResult(False, f"fill({cmd['index']})", f"[Skipped] {file_input}")
            empty_noop = await state_filter.check_fill_is_empty_noop(page, cmd["index"], cmd["text"])
            if empty_noop is not None:
                return ActionResult(False, f"fill({cmd['index']})", f"[Rejected] {empty_noop}")
            redundant = await state_filter.check_fill_redundant(page, cmd["index"], cmd["text"])
            if redundant is not None:
                return ActionResult(True, f"fill({cmd['index']})", f"[Skipped] {redundant}")
            # W_submit_before_confirm_password: ตัดเฉพาะส่วนที่พ่วงมาส่งฟอร์ม ถ้าฟอร์มยังมี
            # ช่องรหัสผ่านอื่นว่างอยู่ — การกรอกยังทำตามปกติ ดูเหตุผลเต็มใน state_filter
            early_submit = None
            if cmd.get("key") == "Enter" or cmd.get("then_click_index") is not None:
                early_submit = await state_filter.check_fill_submits_with_password_fields_left_empty(
                    page, cmd["index"],
                )
            if early_submit is not None:
                cmd = {k: v for k, v in cmd.items() if k not in ("key", "then_click_index")}
            result = await _dispatch_with_retry(fill, page, cmd["index"], cmd["text"])
            if early_submit is not None and result.success:
                result = replace(
                    result,
                    message=f"{result.message} (did not submit the form: {early_submit})",
                )
            # W_chain ("Compound Actions"): "key" (เดิมมีไว้ใช้กับ press_key เท่านั้น) ใช้
            # ร่วมกับ fill ได้ด้วย — กด key นี้ (ปกติ "Enter") ทันทีหลัง fill สำเร็จ รวม
            # "Focus + Type + Press Enter" เป็น 1 step เดียว (fill() เองก็ focus element
            # อยู่แล้วตั้งแต่ต้น — ดู fill() ด้านบน) แทนที่จะต้องรอ perceive ใหม่แล้วสั่ง
            # press_key แยกอีก step — ไม่ต้องเช็ค permission ซ้ำ (press_key ไม่เคยอยู่ใน
            # DEFAULT_NEEDS_CONFIRMATION เลย ปลอดภัยเสมอไม่ว่า index ไหน)
            if result.success and cmd.get("key"):
                key_result = await _dispatch_with_retry(press_key, page, cmd["index"], cmd["key"])
                result = ActionResult(
                    result.success and key_result.success, result.action,
                    f"{result.message} + then press_key({cmd['key']}): {key_result.message}",
                    locator_descriptor=result.locator_descriptor,
                )
            # W_chain follow-up (false-positive fix — reported after enabling "key" above):
            # ยืนยันจากการทดสอบจริงบน MiniWoB "enter-text" ว่า fill() เขียนค่าลง DOM ถูกต้อง
            # เป๊ะเสมอ (ไม่มีปัญหาเรื่อง event/state ไม่ทัน — Playwright's .fill() dispatch
            # input event ให้เองอยู่แล้วแบบ synchronous) แต่บางหน้าเว็บ "ไม่ฟัง" Enter เลย
            # (<input> ไม่ได้อยู่ใน <form> จริง ไม่มี keypress listener) ทำให้ agent ที่เดา
            # ใช้ "key":"Enter" กับฟอร์มแบบนี้เข้าใจผิดว่า submit ไปแล้วทั้งที่ปุ่ม Submit
            # จริงยังไม่เคยถูกคลิกเลย (ค่าที่กรอกไปถูกต้องอยู่ตลอด ไม่ใช่ปัญหา timing/event
            # แต่เป็นปัญหา "เลือก action ผิด") — เพิ่ม then_click_index ให้ใช้กับ fill ได้
            # ด้วย (เดิมมีแค่ click/select/check) ให้ agent สั่ง "fill + คลิกปุ่ม Submit ที่
            # เห็นอยู่แล้ว" เป็น 1 คำสั่งได้ตรงๆ แทนที่จะเดาว่า Enter ใช้ได้ไหม — ทางเลือกที่
            # เชื่อถือได้กว่า key:"Enter" เสมอเมื่อเห็นปุ่ม submit จริงอยู่ในหน้า (ดู
            # SYSTEM_PROMPT ใน llm.py สำหรับลำดับความสำคัญที่แนะนำ agent)
            # W_chained_submit_after_fill (บั๊กจริงจากรันสดสองเทิร์น 2026-09-04): เช็ค
            # *หลัง* fill เท่านั้น — ก่อน fill ช่องยังถือค่าเก่าอยู่ทั้งคู่ซึ่ง "ตรงกัน" พอดี
            # guard ที่เช็คก่อน dispatch จึงมองไม่เห็นปัญหาเลย เทิร์นที่สองกรอกทับเฉพาะช่อง
            # Password ช่องเดียว ช่อง Confirm ยังค้างค่าจากเทิร์นแรก แล้ว chained click ก็กด
            # Save ต่อทันที -> 'Passwords do not match'
            # (guard ฝั่ง orchestrator คุมได้เฉพาะ action type "click" ที่โมเดลสั่งแยก —
            # การส่งฟอร์มที่พ่วงมากับ fill ไม่เคยผ่านตรงนั้นเลย)
            if result.success and cmd.get("then_click_index") is not None:
                chained_problem = await state_filter.password_form_submit_problem(
                    page, cmd["then_click_index"],
                )
                if chained_problem is not None:
                    fields = ", ".join(str(i) for i in chained_problem.get("indexes") or [])
                    reason = (
                        f"password fields {fields} are still empty"
                        if chained_problem.get("kind") == "empty"
                        else f"the new-password fields {fields} do not hold the same value "
                             "(the confirmation field may still hold a value typed earlier)"
                    )
                    return replace(
                        result,
                        message=(
                            f"{result.message} (did not submit the form: {reason} — type the "
                            "SAME new password into every one of them, then submit)"
                        ),
                    )
            return await _maybe_chain_click(
                page, cmd, result, ask_user_func, manual_guidance, allowed_domains,
                then_label, then_tag, then_type,
            )
        if t == "fill_secret":
            # W65[3]: ไม่เช็ค state_filter.check_fill_redundant() เหมือน "fill" ด้านบน — ฟังก์ชัน
            # นั้นต้องการ cmd["text"] (ค่าจริงที่จะเทียบกับค่าปัจจุบันในช่อง) แต่ fill_secret ไม่มี
            # ค่าจริงให้ LLM เห็นเลยตั้งแต่ต้น (แค่ secret_key เป็นชื่อ symbolic) ข้ามการเช็คนี้ไป
            # เป็นแค่ optimization ที่เสียไป ไม่กระทบ correctness (retry wrapper ด้านล่างยังกัน
            # DOM ไม่นิ่งได้ตามปกติ)
            return await _dispatch_with_retry(fill_secret, page, cmd["index"], cmd.get("secret", ""))
        if t == "select":
            # W_custom_dropdown: ปฏิเสธก่อน dispatch ถ้า target ไม่ใช่ <select> จริง (ดู
            # state_filter.check_select_target_is_native) — success=False เพราะนี่คือ "ใช้
            # action ผิดชนิด" ไม่ใช่ "ทำไปแล้ว" pattern เดียวกับ click-on-disabled ด้านบน
            wrong_kind = await state_filter.check_select_target_is_native(page, cmd["index"])
            if wrong_kind is not None:
                return ActionResult(False, f"select({cmd['index']})", f"[Skipped] {wrong_kind}")
            result = await _dispatch_with_retry(select_option, page, cmd["index"], cmd["label"])
            return await _maybe_chain_click(
                page, cmd, result, ask_user_func, manual_guidance, allowed_domains,
                then_label, then_tag, then_type,
            )
        if t == "check":
            redundant = await state_filter.check_checkbox_redundant(page, cmd["index"])
            if redundant is not None:
                return ActionResult(True, f"check({cmd['index']})", f"[Skipped] {redundant}")
            result = await _dispatch_with_retry(check, page, cmd["index"])
            return await _maybe_chain_click(
                page, cmd, result, ask_user_func, manual_guidance, allowed_domains,
                then_label, then_tag, then_type,
            )
        if t == "scroll":
            direction = cmd.get("direction", "down")
            redundant = await state_filter.check_scroll_redundant(page, direction)
            if redundant is not None:
                return ActionResult(True, f"scroll({direction})", f"[Skipped] {redundant}")
            return await scroll(page, direction)
        if t == "goto":        return await goto(page, cmd["url"])
        if t == "go_back":     return await go_back(page)
        if t == "switch_tab":  return await switch_tab(page, cmd["tab_index"])
        if t == "wait":        return await wait_stable(page)
        if t == "hover":       return await hover(page, cmd["index"])
        if t == "press_key":   return await _dispatch_with_retry(press_key, page, cmd["index"], cmd["key"])
        if t == "read_page_data":
            return await read_page_data(page, cmd.get("query", ""), cmd.get("target_hint", ""))
        if t in DEFAULT_NEEDS_CONFIRMATION:
            # submit/delete/purchase/pay ไม่ใช่ action จริงแยกต่างหาก — เป็นแค่ risk
            # category ของ classify_action() (เช็คผ่านไปแล้วด้านบนตอนมาถึงตรงนี้) ที่จริง
            # แล้วคือคลิก element ตัวเดิม แค่ต้องขอยืนยันจาก human ก่อนเพราะเสี่ยงกว่า
            # click ธรรมดา — คืน label เดิม (เช่น "submit(2)") ไม่ใช่ "click(2)" กันสับสน
            # (ใช้ retry wrapper เดียวกับ click ปกติ รวม hover-on-retry ด้วย — ปุ่ม hover-to-
            # reveal ก็อาจเป็นปุ่มความเสี่ยงสูงได้เหมือนกัน เช่น "Delete" ที่โผล่มาตอน hover)
            result = await _dispatch_click_with_retry(page, cmd["index"], label)
            # W64[7.2]: คง locator_descriptor/toast_confirmed จาก result เดิมไว้ด้วย (เดิม
            # re-wrap ทิ้งทั้งสอง field นี้ไปเงียบๆ — "Save"/"Submit" ที่ classify_action()
            # มองว่าเสี่ยงพอต้องขออนุมัติ (เช่น label มีคำว่า "Confirm Purchase") ก็ยัง
            # ต้องการให้ toast_confirmed สะท้อนความจริงถูกต้องเหมือน plain click ทุกประการ)
            return ActionResult(
                result.success, f"{t}({cmd['index']})", result.message,
                locator_descriptor=result.locator_descriptor, toast_confirmed=result.toast_confirmed,
            )
        return ActionResult(False, f"unknown({t})", "unknown action")
    except KeyError as e:
        return ActionResult(False, f"{t}", f"missing parameter: {e}")


# ------------------------------------------------------------
# DEMO: ทดสอบ actions ครบชุดบน saucedemo (login -> เลือก dropdown -> checkout)
# ------------------------------------------------------------
async def demo():
    from playwright.async_api import async_playwright
    from perception import get_snapshot   # ใช้ perception จาก W2

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()
        await install_ssrf_guard(page)
        await page.goto("https://www.saucedemo.com/")

        # ทุก step: perceive -> execute -> log ผล (นี่คือตัวอย่างย่อของ W4)
        steps = [
            {"type": "fill",   "index": 0, "text": "standard_user"},
            {"type": "fill",   "index": 1, "text": "secret_sauce"},
            {"type": "click",  "index": 2},
            {"type": "wait"},
        ]

        for cmd in steps:
            res = await execute(page, cmd)
            print(res)

        # perceive หน้าใหม่ แล้วลอง action ที่ยังไม่เคยเทสต์: select dropdown + scroll
        elements, text_repr = await get_snapshot(page)
        print("\n--- หน้า inventory ---")
        print(text_repr[:400], "...\n")

        sel_idx = next(e['index'] for e in elements if e['tag'] == 'select')
        print(await execute(page, {"type": "select", "index": sel_idx, "label": "Price (low to high)"}))
        print(await execute(page, {"type": "scroll", "direction": "down"}))

        await asyncio.sleep(3)
        await browser.close()


if __name__ == "__main__":
    asyncio.run(demo())
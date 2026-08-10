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
import re
from dataclasses import dataclass
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


# ------------------------------------------------------------
# W5: Verify + Retry — action พวก click/fill/select/check พังบ่อยเพราะ DOM ยัง
# ไม่นิ่ง (element ยัง render/animate ไม่เสร็จ) ไม่ใช่เพราะ index ผิดจริงๆ เสมอไป
# retry เงียบๆ ระดับนี้ก่อน ไม่เสีย token เพราะไม่ต้องถาม LLM จนกว่าจะลองครบ —
# ถ้ายัง fail อยู่หลัง retry ครบ ค่อยส่งกลับให้ LLM ตัดสินใจเหมือน W4 เดิม
# ------------------------------------------------------------
_ACTION_RETRIES = 3  # ครั้งแรก + retry อีก 2 ครั้ง
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
                    True, result.action, f"{result.message} (ลองครั้งที่ {attempt}/{_ACTION_RETRIES})"
                )
            return result
        if attempt < _ACTION_RETRIES:
            await asyncio.sleep(_ACTION_RETRY_DELAY_SEC)
    return ActionResult(False, result.action, f"{result.message} (ลองแล้ว {_ACTION_RETRIES} ครั้ง)")


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
        return ActionResult(True, f"click({index})", "คลิกสำเร็จ", locator_descriptor=descriptor)
    except PWTimeout:
        return ActionResult(False, f"click({index})", "หา element ไม่เจอ/คลิกไม่ได้ (timeout)")
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
        return ActionResult(True, f"hover({index})", "hover สำเร็จ")
    except PWTimeout:
        return ActionResult(False, f"hover({index})", "หา element ไม่เจอ/hover ไม่ได้ (timeout)")
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
# ร่วมกันจริง (Material/Bootstrap/Ant Design ฯลฯ) ต่างจาก _RECORD_COUNT_SELECTOR ใน
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
    เจอทันที เพราะ toast มักปรากฏหลัง response กลับมาไม่กี่ร้อย ms ไม่ใช่ทันทีที่คลิก"""
    try:
        locator = page.locator(_SUCCESS_TOAST_SELECTOR).first
        await locator.wait_for(state="visible", timeout=_TOAST_WAIT_TIMEOUT_MS)
        text = (await locator.inner_text(timeout=_ELEMENT_ACTION_TIMEOUT_MS)).strip()
        return text[:200] if text else None
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
    result: ActionResult = None
    for attempt in range(1, _ACTION_RETRIES + 1):
        if attempt > 1:
            await hover(page, index)
        result = await click(page, index)
        if result.success:
            if attempt > 1:
                result = ActionResult(
                    True, result.action, f"{result.message} (ลองครั้งที่ {attempt}/{_ACTION_RETRIES})"
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
                    f' [พบข้อความยืนยันสำเร็จ: "{toast_text}"]' if toast_text
                    else " [ไม่พบ toast/ข้อความยืนยันสำเร็จภายในเวลาที่กำหนดหลังคลิก — "
                         "ตรวจสอบ validation error หรือดูว่าหน้าเปลี่ยนกลับไปหน้ารายการเองแล้ว"
                         "หรือยังก่อนถือว่าสำเร็จ]"
                )
                result = ActionResult(
                    result.success, result.action, f"{result.message}{toast_note}",
                    locator_descriptor=result.locator_descriptor, toast_confirmed=bool(toast_text),
                )
            return result
        if attempt < _ACTION_RETRIES:
            await asyncio.sleep(_ACTION_RETRY_DELAY_SEC)
    return ActionResult(False, result.action, f"{result.message} (ลองแล้ว {_ACTION_RETRIES} ครั้ง)")


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
_DIALOG_CONTAINER_SELECTOR = '.oxd-dialog-container, [role="dialog"], .orangehrm-modal-header'

# เรียงจากเจาะจงที่สุด (OrangeHRM "Yes, Delete" ปุ่มสีแดง) ไปหากว้างที่สุด (fallback ทั่วไป
# สำหรับ dialog framework อื่นที่ไม่ใช่ OrangeHRM) — ลองทีละตัวจนกว่าจะเจอปุ่มที่ visible จริง
_MODAL_CONFIRM_BUTTON_SELECTORS = [
    "div.oxd-dialog-container-default button.oxd-button--label-danger",
    ".oxd-button--label-danger",
    'button:has-text("Yes, Delete")',
    '[role="dialog"] button.oxd-button--secondary',
    '[role="dialog"] button:has-text("Confirm")',
]

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
    ไม่ได้ ถือว่า "ไม่มีโมดัล" ปลอดภัยกว่าเสมอ ดีกว่าไปบล็อก/หน่วง click ที่สำเร็จอยู่แล้ว)"""
    return await _is_modal_still_open(page)


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
            retry_note = "" if attempt == 1 else f" (ลองครั้งที่ {attempt}/{_MODAL_CONFIRM_CLICK_RETRIES})"
            return f" [ตรวจพบ confirmation modal — กดยืนยันอัตโนมัติแล้ว ({clicked_selector}){retry_note}]"
        except Exception:
            # detach-wait timeout: อาจเป็นเพราะโมดัลปิดจริงแล้วแค่ไม่ detach ออกจาก DOM (บาง
            # framework ซ่อนด้วย CSS อย่างเดียว ไม่ลบ element) หรืออาจเป็นเพราะยังเปิดค้างอยู่
            # จริงๆ (ปุ่มไม่ตอบสนอง) — เช็คแยกให้ชัดก่อนตัดสินใจ retry/reload ต่อ ไม่เดาว่า
            # timeout = ปิดสำเร็จเหมือนพฤติกรรมเดิม (W23) อีกต่อไป
            if not await _is_modal_still_open(page):
                await wait_stable(page)
                return f" [ตรวจพบ confirmation modal — กดยืนยันอัตโนมัติแล้ว ({clicked_selector})]"
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
        f" [ปุ่มยืนยัน confirmation modal ไม่ตอบสนองหลังลองแล้ว {_MODAL_CONFIRM_CLICK_RETRIES} "
        f"ครั้ง — ระบบ reload หน้าเว็บอัตโนมัติเพื่อ sync สถานะใหม่ (เหมือนกด F5) ต้องตรวจสอบ "
        f"indexed elements ล่าสุดหลังจากนี้แล้ว navigate/กรองข้อมูลใหม่ตามที่ goal ต้องการก่อน"
        f"ทำงานต่อ เพราะ reload ล้าง state เดิม (เช่นคำค้นหาที่กรองไว้) ทิ้งไปแล้ว]"
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
        return ActionResult(True, f"press_key({index}, {key})", f"กดปุ่ม '{key}' สำเร็จ")
    except PWTimeout:
        return ActionResult(False, f"press_key({index}, {key})", "หา element ไม่เจอ/กดปุ่มไม่ได้ (timeout)")
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
async def fill(page: Page, index: int, text: str, timeout: int = _ELEMENT_ACTION_TIMEOUT_MS) -> ActionResult:
    """พิมพ์ข้อความลงช่อง input/textarea ตาม index — เคลียร์ข้อความเดิมด้วย
    focus -> select-all -> Backspace ก่อนเสมอ (ดู module comment ด้านบน)"""
    try:
        selector = _sel(index)
        target = await resolve_frame(page, selector)
        await target.click(selector, timeout=timeout)
        await target.press(selector, "ControlOrMeta+a", timeout=timeout)
        await target.press(selector, "Backspace", timeout=timeout)
        await target.fill(selector, text, timeout=timeout)
        descriptor = await compute_locator_descriptor(target, selector)
        return ActionResult(True, f"fill({index})", f"กรอก '{text}' สำเร็จ", locator_descriptor=descriptor)
    except PWTimeout:
        return ActionResult(False, f"fill({index})", "กรอกไม่ได้ (timeout)")
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
        return ActionResult(False, f"fill_secret({index})", f"ไม่รู้จัก secret_key '{secret_key}' — ต้องถาม user เอง")

    # Lazy import กัน circular import (site_learning -> crawler.py -> orchestrator.py ->
    # fastpath_executor.py -> actions.py) — pattern เดียวกับ orchestrator.py::_maybe_auto_login()
    from backend.app.site_learning import storage as site_storage

    domain = extract_domain(page.url)
    creds = site_storage.load_credentials(domain)  # sync call, pattern เดียวกับ _maybe_auto_login
    if not creds or not creds.get("password"):
        return ActionResult(False, f"fill_secret({index})", "ไม่มี credential ที่บันทึกไว้สำหรับเว็บนี้ — ต้องถาม user เอง")

    try:
        selector = _sel(index)
        target = await resolve_frame(page, selector)
        await target.click(selector, timeout=timeout)
        await target.press(selector, "ControlOrMeta+a", timeout=timeout)
        await target.press(selector, "Backspace", timeout=timeout)
        await target.fill(selector, creds["password"], timeout=timeout)
        descriptor = await compute_locator_descriptor(target, selector)
        # ***ห้าม echo ค่าจริงกลับใน message เด็ดขาด*** ต่างจาก fill() ปกติด้านบน
        return ActionResult(True, f"fill_secret({index})", "กรอกรหัสผ่านที่บันทึกไว้สำเร็จ", locator_descriptor=descriptor)
    except PWTimeout:
        return ActionResult(False, f"fill_secret({index})", "กรอกไม่ได้ (timeout)")
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
                True, f"select({index})", f"เลือก '{_normalize_option_text(matched['text'])}' สำเร็จ",
                locator_descriptor=descriptor,
            )
        except PWTimeout:
            return ActionResult(False, f"select({index})", "เลือกไม่ได้ (timeout)")
        except Exception as e:
            return ActionResult(False, f"select({index})", f"error: {e}")

    # ไม่เจอ text ที่ตรงกันเลยแม้ normalize แล้ว — เผื่อ label ที่ LLM ส่งมาคือ value
    # attribute ไม่ใช่ text ที่เห็น (ของเดิมมี fallback นี้อยู่แล้ว ยังคงไว้เหมือนเดิม)
    try:
        await target.select_option(selector, value=label, timeout=timeout)
        descriptor = await compute_locator_descriptor(target, selector)
        return ActionResult(True, f"select({index})", f"เลือก (by value) '{label}' สำเร็จ", locator_descriptor=descriptor)
    except Exception:
        options_repr = ", ".join(repr(_normalize_option_text(o["text"])) for o in options)
        options_repr = options_repr or "(ไม่พบ option ใดๆ ใน dropdown นี้)"
        return ActionResult(
            False, f"select({index})",
            f"ไม่พบตัวเลือกที่ตรงกับ '{label}' แม้ normalize whitespace แล้ว — "
            f"ตัวเลือกที่มีจริง: {options_repr}",
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
            }"""
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
        return ActionResult(True, f"check({index})", "ติ๊กสำเร็จ", locator_descriptor=descriptor)
    except Exception:
        pass

    # Fallback 1: force click — ข้าม Playwright's actionability check (element visible ตาม
    # นิยามของ Playwright) เหมือน hover(force=True) ด้านบน คลิกที่พิกัดกึ่งกลางของ element
    # ตรงๆ ไม่ว่า Playwright จะมองว่า element นี้ "checkable" หรือไม่
    try:
        target = await resolve_frame(page, selector)
        await target.click(selector, timeout=timeout, force=True)
        if await _is_effectively_checked(target, selector):
            descriptor = await compute_locator_descriptor(target, selector)
            return ActionResult(True, f"check({index})", "ติ๊กสำเร็จ (force click)", locator_descriptor=descriptor)
    except Exception:
        pass

    # Fallback 2: JS dispatch ตรงบน element ผ่าน el.click() — ข้าม actionability check และ
    # การจำลอง mouse event ของ Playwright ทั้งหมด ใช้เป็นทางสุดท้ายสำหรับ wrapper ที่ force
    # click ก็ยังคลิกไม่โดน (เช่น wrapper ที่มีขนาด 0x0 จริงๆ ตัวอย่างการมองเห็นมาจาก
    # pseudo-element ล้วนๆ)
    try:
        target = await resolve_frame(page, selector)
        await target.locator(selector).evaluate("el => el.click()")
        if await _is_effectively_checked(target, selector):
            descriptor = await compute_locator_descriptor(target, selector)
            return ActionResult(True, f"check({index})", "ติ๊กสำเร็จ (JS click)", locator_descriptor=descriptor)
        return ActionResult(False, f"check({index})", "คลิกแล้วแต่ยืนยันสถานะ checked ไม่ได้")
    except Exception as e:
        return ActionResult(False, f"check({index})", f"error: {e}")


async def scroll(page: Page, direction: str = "down", amount: int = 600) -> ActionResult:
    """เลื่อนหน้าจอ ('down'/'up') — ใช้ตอน element ที่ต้องการอยู่นอกจอ"""
    try:
        dy = amount if direction == "down" else -amount
        await page.mouse.wheel(0, dy)
        await page.wait_for_timeout(300)
        return ActionResult(True, f"scroll({direction})", f"เลื่อน {dy}px")
    except Exception as e:
        return ActionResult(False, f"scroll({direction})", f"error: {e}")


async def goto(page: Page, url: str, timeout: int = 15000) -> ActionResult:
    """เปิด URL ใหม่"""
    try:
        await page.goto(url, timeout=timeout)
        return ActionResult(True, "goto", f"ไปที่ {url}")
    except Exception as e:
        return ActionResult(False, "goto", f"error: {e}")


async def go_back(page: Page) -> ActionResult:
    """ย้อนกลับหน้าก่อนหน้า"""
    try:
        await page.go_back()
        return ActionResult(True, "go_back", "ย้อนกลับสำเร็จ")
    except Exception as e:
        return ActionResult(False, "go_back", f"error: {e}")


async def switch_tab(page: Page, tab_index: int) -> ActionResult:
    """สลับไป tab อื่นในหน้าต่างเดียวกัน (บาง action เปิด tab ใหม่)"""
    try:
        pages = page.context.pages
        if tab_index >= len(pages):
            return ActionResult(False, f"switch_tab({tab_index})", f"มีแค่ {len(pages)} tab")
        await pages[tab_index].bring_to_front()
        return ActionResult(True, f"switch_tab({tab_index})", "สลับ tab สำเร็จ")
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
        return ActionResult(True, "wait_stable", "หน้านิ่งแล้ว")
    except PWTimeout:
        # ไม่ถือเป็น fail ร้ายแรง — บางหน้ามี network ยิงตลอด
        return ActionResult(True, "wait_stable", "timeout แต่เดินต่อได้")
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
        return ActionResult(False, "read_page_data", "ต้องระบุ target_hint (CSS selector)")

    await wait_stable(page)

    is_count_query = any(kw in query.lower() for kw in _COUNT_QUERY_KEYWORDS)
    try:
        if is_count_query:
            count = await count_elements(page, target_hint)
            return ActionResult(True, "read_page_data", f"พบ {count} รายการที่ตรงกับ '{target_hint}'")
        data = await extract_table_data(page, target_hint, query)
        return ActionResult(not data.startswith("[FAIL]"), "read_page_data", data)
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
REJECTED_BY_USER_MESSAGE = "ผู้ใช้ปฏิเสธการทำ Action นี้ (Human-in-the-loop)"


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
    print(f"\n[HUMAN-IN-THE-LOOP] Agent ต้องการเรียกใช้คำสั่งที่มีความเสี่ยง: {cmd}", flush=True)
    # ใช้ asyncio.to_thread เพื่อให้รับ input() ได้โดยไม่บล็อก async event loop หลัก
    choice = await asyncio.to_thread(input, "คุณต้องการอนุญาตให้ทำ Action นี้หรือไม่? (y/n): ")
    approved = choice.strip().lower() in ("y", "yes")
    if approved:
        print("[APPROVED] อนุญาตให้ดำเนินการต่อ...", flush=True)
    return approved


# ------------------------------------------------------------
# ทางเข้าเดียวสำหรับ W4: agent ส่ง action มาเป็น dict แล้ว dispatch
# ------------------------------------------------------------
async def execute(
    page: Page, cmd: dict, ask_user_func: Optional[AskUserFunc] = None, label: str = "",
    manual_guidance: str = "", allowed_domains: Optional[set] = None, element_tag: str = "",
    element_type: str = "",
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
    """
    t = cmd.get("type")

    risk = classify_action(
        cmd, label=label, manual_guidance=manual_guidance, allowed_domains=allowed_domains,
        element_tag=element_tag, element_type=element_type,
    )
    if risk == ActionRisk.BLOCKED:
        return ActionResult(False, f"{t}", "Action ถูกบล็อกโดยระบบรักษาความปลอดภัย (Blocklist)")
    if risk == ActionRisk.NEEDS_CONFIRMATION:
        approved = await _confirm_action(cmd, ask_user_func, label)
        if not approved:
            return ActionResult(False, f"{t}", REJECTED_BY_USER_MESSAGE)

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
                return ActionResult(False, f"click({cmd['index']})", f"[ข้าม] {redundant}")
            return await _dispatch_click_with_retry(page, cmd["index"], label)
        if t == "fill":
            redundant = await state_filter.check_fill_redundant(page, cmd["index"], cmd["text"])
            if redundant is not None:
                return ActionResult(True, f"fill({cmd['index']})", f"[ข้าม] {redundant}")
            return await _dispatch_with_retry(fill, page, cmd["index"], cmd["text"])
        if t == "fill_secret":
            # W65[3]: ไม่เช็ค state_filter.check_fill_redundant() เหมือน "fill" ด้านบน — ฟังก์ชัน
            # นั้นต้องการ cmd["text"] (ค่าจริงที่จะเทียบกับค่าปัจจุบันในช่อง) แต่ fill_secret ไม่มี
            # ค่าจริงให้ LLM เห็นเลยตั้งแต่ต้น (แค่ secret_key เป็นชื่อ symbolic) ข้ามการเช็คนี้ไป
            # เป็นแค่ optimization ที่เสียไป ไม่กระทบ correctness (retry wrapper ด้านล่างยังกัน
            # DOM ไม่นิ่งได้ตามปกติ)
            return await _dispatch_with_retry(fill_secret, page, cmd["index"], cmd.get("secret", ""))
        if t == "select":      return await _dispatch_with_retry(select_option, page, cmd["index"], cmd["label"])
        if t == "check":
            redundant = await state_filter.check_checkbox_redundant(page, cmd["index"])
            if redundant is not None:
                return ActionResult(True, f"check({cmd['index']})", f"[ข้าม] {redundant}")
            return await _dispatch_with_retry(check, page, cmd["index"])
        if t == "scroll":
            direction = cmd.get("direction", "down")
            redundant = await state_filter.check_scroll_redundant(page, direction)
            if redundant is not None:
                return ActionResult(True, f"scroll({direction})", f"[ข้าม] {redundant}")
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
        return ActionResult(False, f"unknown({t})", "ไม่รู้จัก action นี้")
    except KeyError as e:
        return ActionResult(False, f"{t}", f"ขาด parameter: {e}")


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
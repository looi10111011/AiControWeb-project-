"""
actions.py — W3: Browser Actions
--------------------------------
ห่อการกระทำบน browser เป็น action มาตรฐานที่ agent loop เรียกด้วย index จาก
perception.get_snapshot() — ทุก action คืน ActionResult เสมอ ไม่ throw ดิบ

W5: execute() retry click/fill/select/check เอง (_dispatch_with_retry) กัน false negative
    จาก DOM ที่ยังไม่นิ่ง โดยไม่เสีย LLM token
W19: เช็ค core/state_filter.py ก่อน dispatch — action ที่สำเร็จอยู่แล้ว (fill ค่าเดิม, check ที่
    ติ๊กแล้ว, scroll สุดหน้า) หรือทำไม่ได้แน่ (click element disabled) short-circuit ทันที
W40: resolve_frame() ก่อน dispatch ทุกจุด — page.click(selector) ข้าม <iframe> ไม่ได้
W42: select_option() เทียบ label แบบ normalize whitespace (_normalize_option_text) — option
    ที่ใช้ &nbsp; (uitestingplayground.com/select) ไม่มีวัน match แบบ exact แล้วเลือกด้วย text
    จริงจาก DOM ไม่เจอคืน [FAIL] พร้อมรายการตัวเลือกจริง

แผนที่โซน (ค้นหา "# โซน N" ในไฟล์):
  โซน 1   ActionResult + ค่าคงที่
  โซน 2   retry engine (_dispatch_with_retry)
  โซน 3   click / hover
  โซน 4   ตรวจผลหลังคลิก (_dispatch_click_with_retry: toast / error / navigation)
  โซน 5   confirmation modal (ตรวจ + กดยืนยันอัตโนมัติ)
  โซน 6   press_key / fill / fill_secret / select_option / check
  โซน 7   scroll / goto / go_back / switch_tab / wait_stable
  โซน 8   read_page_data + นับแบบ deterministic
  โซน 9   permission layer
  โซน 10  execute() — ทางเข้าเดียวของ agent loop

ลำดับการทำงานของ 1 action: orchestrator -> execute() (โซน 10) -> permission (โซน 9)
-> state_filter -> action (โซน 3/6/7/8) ผ่าน retry (โซน 2) -> ตรวจผล (โซน 4/5) -> ActionResult
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
    DEFAULT_NEEDS_CONFIRMATION, ActionRisk, classify_action, extract_domain,
)
# W65[3]: site_learning.storage ต้อง lazy import ในฟังก์ชัน (ดู fill_secret) — circular import
# (site_learning -> crawler -> orchestrator -> fastpath_executor -> actions) แบบ _maybe_auto_login

# ══════════════════════════════════════════════════════════════════════
# โซน 1: ชนิดข้อมูลพื้นฐาน + ค่าคงที่
#   ทำอะไร: ActionResult ที่ทุก action คืน, callback ขออนุมัติ, timeout ต่างๆ
#   ทำงานยังไง: ActionResult มี field เสริม (locator_descriptor/toast_confirmed/dropdown_option_selected) ส่งสัญญาณให้ orchestrator แบบ type-checked
# ══════════════════════════════════════════════════════════════════════
# ask_user_func: callback ให้ชั้นบน (API server W10) ตัดสินใจแทน blocking input() — รับ cmd คืน True = อนุญาต
AskUserFunc = Callable[[dict], Awaitable[bool]]


# --- ผลลัพธ์มาตรฐานของทุก action ---
@dataclass
class ActionResult:
    success: bool
    action: str
    message: str = ""
    # W_procmem: locator ที่อยู่รอดข้าม task (core/dom_locator.py) — เฉพาะ click/fill/select/check
    # ที่สำเร็จ นอกนั้น None default ท้ายสุดไม่กระทบ call site แบบ positional 3 ตัว
    locator_descriptor: Optional[dict] = None
    # W64[7.2] (Add-Action Idempotency Lock): True เมื่อ _detect_success_toast() เจอ toast หลัง
    # click Save/Submit/Confirm — หลักฐานจริงว่าบันทึกแล้ว orchestrator ใช้ผ่อน table-verify guard
    # field type-checked แทน string matching ข้ามไฟล์
    toast_confirmed: bool = False
    # W_dropdown_sets_filter_dirty: True เมื่อ click นี้เลือกตัวเลือกใน custom dropdown ที่เปิดอยู่
    # (state_filter.classify_click_index_disturbance) — W50 บังคับ click กับ custom dropdown ธง
    # filter_dirty_since_search จึงไม่เคยถูกยกบน SPA guard "ห้ามคลิก row action ก่อนกด Search" ตาย
    dropdown_option_selected: bool = False

    def __str__(self):
        mark = "OK" if self.success else "FAIL"
        return f"[{mark}] {self.action} -> {self.message}"


# selector ที่ผูกกับ index ที่ perception ติดไว้บน element
def _sel(index: int) -> str:
    return f'[data-ai-index="{index}"]'


def _normalize_option_text(text: str) -> str:
    """W42: nbsp -> space แล้วยุบ whitespace ซ้ำ + trim — เทียบ label ของ LLM กับ option จริง"""
    return re.sub(r"\s+", " ", (text or "").replace(" ", " ")).strip()


# W5: timeout สั้นของ action ที่หา element — คูณกับ retry แล้ว 5s เดิมรอได้ถึง 15s ต่อ action
_ELEMENT_ACTION_TIMEOUT_MS = 3000
# W_check_evaluate_timeout: fallback ของ check() (force/JS click) ไม่รอ actionability timeout ครอบแค่
# "หาเจอไหม" ซึ่งชั้นแรกพิสูจน์แล้ว 3s ต่อชั้นคือรอเปล่าคูณ _ACTION_RETRIES
_ELEMENT_FALLBACK_TIMEOUT_MS = 1000
# อ่านสถานะ DOM ล้วนๆ หลังคลิก — ค่าเดียวกับ state_filter._STATE_CHECK_TIMEOUT_MS ด้วยเหตุผล
# เดียวกัน (เช็คก่อน/หลัง dispatch ทุก step ต้องเร็วที่สุด)
_STATE_READ_TIMEOUT_MS = 500


# ══════════════════════════════════════════════════════════════════════
# โซน 2: retry engine
#   ทำอะไร: ลองซ้ำ action ที่พังเพราะ DOM ยังไม่นิ่ง ก่อนส่งผลให้ LLM
#   ทำงานยังไง: _dispatch_with_retry() ลองสูงสุด _ACTION_RETRIES ครั้ง คั่น delay สั้นๆ
# ══════════════════════════════════════════════════════════════════════
# W_retry_never_paid_off (step_trace 422 แถว 2026-09-03): สำเร็จในรอบ 2-3 = 0 ครั้ง ล้มครบทุกรอบ 29
# ครั้ง — รอบสามไม่เคยกู้อะไรได้ แต่คูณเวลาหางของทุก action ที่ล้ม เหลือ retry หนึ่งรอบ
_ACTION_RETRIES = 2  # ครั้งแรก + retry อีก 1 ครั้ง
_ACTION_RETRY_DELAY_SEC = 0.5
# W_menu_overlay_blocks_target: รอสั้นๆ ให้เมนูปิดจริงก่อนวัดซ้ำ (transition ของ UI library)
_MENU_DISMISS_WAIT_MS = 200


async def _dispatch_with_retry(action_func, *args) -> ActionResult:
    """เรียก action_func สูงสุด _ACTION_RETRIES ครั้ง คืนผลแรกที่สำเร็จหรือผลครั้งสุดท้าย
    แนบจำนวนครั้งที่ลองใน message (debug ความ flaky)"""
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


# ══════════════════════════════════════════════════════════════════════
# โซน 3: click / hover
#   ทำอะไร: action พื้นฐานบน element ตาม index
#   ทำงานยังไง: resolve_frame() หา frame ก่อน -> สั่ง Playwright -> คืน ActionResult
# ══════════════════════════════════════════════════════════════════════

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


# W47: ปุ่ม hover-to-reveal (uitestingplayground scrolltoclick Case 4) — locator.click() ไม่ trigger
# :hover ของ ancestor ให้ คลิกตรงรอบแรกจึงพลาดเสมอ
async def hover(page: Page, index: int, timeout: int = _ELEMENT_ACTION_TIMEOUT_MS) -> ActionResult:
    """เลื่อนเมาส์ไปวางบน element (ไม่คลิก) — trigger :hover ของ ancestor ทั้งสายเอง

    force=True จำเป็น: element visibility:hidden ไม่ผ่าน actionability check ของ Playwright
    (.hover() ธรรมดา timeout เหมือน click) force ส่ง mouse move ไปพิกัดกึ่งกลางตรงๆ แล้ว click()
    รอบถัดไปเจอ element ที่ visible แล้ว"""
    try:
        selector = _sel(index)
        target = await resolve_frame(page, selector)
        await target.hover(selector, timeout=timeout, force=True)
        return ActionResult(True, f"hover({index})", "hover succeeded")
    except PWTimeout:
        return ActionResult(False, f"hover({index})", "element not found / not hoverable (timeout)")
    except Exception as e:
        return ActionResult(False, f"hover({index})", f"error: {e}")


# ══════════════════════════════════════════════════════════════════════
# โซน 4: ตรวจผลหลังคลิก (click-family ทั้งหมดผ่านที่นี่)
#   ทำอะไร: ยืนยันว่าคลิกได้ผลจริง: toast สำเร็จ / error ที่เพิ่งโผล่ / หน้าเปลี่ยน / modal
#   ทำงานยังไง: _dispatch_click_with_retry() อ่านสถานะก่อนคลิก -> คลิก (retry + hover) -> เทียบ URL/DOM/error ก่อน-หลัง -> resolve modal / เช็ค toast -> รายงานตามความจริง
# ══════════════════════════════════════════════════════════════════════
# W63[7.1] (Save Confirmation & Toast Wait): toast หายเองในไม่กี่วินาที เช็คตอน finish_task (หลาย step
# ถัดมา) ไม่น่าเชื่อถือ — เช็คทันทีหลัง click ที่ label เป็น Save/Submit/Confirm รอสั้นๆ ไม่ throw
# แล้วแนบผลใน message
_SAVE_LABEL_RE = re.compile(r"\b(save|submit|confirm|update)\b|บันทึก|ยืนยัน|อัปเดต|อัพเดท", re.IGNORECASE)

# เรียงจากเจาะจง (OrangeHRM .oxd-toast) ไปกว้าง (role=status/alert, class toast/snackbar) — generic
# โดยตั้งใจ ต่างจาก "Records Found" ใน orchestrator ที่เป็นของ OrangeHRM เท่านั้น
_SUCCESS_TOAST_SELECTOR = (
    '.oxd-toast--success, .oxd-toast-container, '
    '[role="status"]:not(:empty), [role="alert"]:not(:empty), '
    '[class*="toast" i]:not(:empty), [class*="snackbar" i]:not(:empty), '
    '[class*="notification" i][class*="success" i]:not(:empty)'
)

_TOAST_WAIT_TIMEOUT_MS = 2500


async def _detect_success_toast(page: Page) -> Optional[str]:
    """W63[7.1]: ข้อความ success toast หลัง action (<= 200 ตัว) หรือ None ไม่ throw — รอสั้นๆ
    (_TOAST_WAIT_TIMEOUT_MS) เพราะ toast มาหลัง response ไม่กี่ร้อย ms

    W_toast_container_is_empty (2026-09-07): .first หยิบ .oxd-toast-container ที่ว่าง ได้ "" ทั้งที่
    toast ขึ้นจริง (156 ms ถึง 3500 ms) agent เผา step หาคำยืนยัน — รอ element แรกที่ *มีข้อความ*"""
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


# W_click_navigated (OrangeHRM 2026-08-26): คลิก "Admin" สำเร็จหน้าเปลี่ยนแล้ว แต่ retry รอบถัดไปหา
# [data-ai-index="N"] ที่ไม่มีบนหน้าใหม่ timeout แล้วรายงาน [FAIL] (เสียเวลาอีก ~10s) — คลาสเดียวกับ
# crawler.py W34: error จริงเมื่อ URL ไม่เปลี่ยน ห้ามใช้ crawler._normalize_url() เพราะตัด fragment
# (hash router #/admin เปลี่ยนแค่ตรงนั้น) — เทียบ URL เต็ม normalize แค่ trailing slash
def _normalize_click_url(url: str) -> str:
    """W_click_navigated: ตัดแค่ trailing slash คง query + fragment ไว้
    ค่าที่ไม่ใช่ str (mock ในเทสต์) คืน "" — ก่อน/หลังเท่ากัน = ไม่ได้ navigate (default ปลอดภัย)"""
    text = url.strip() if isinstance(url, str) else ""
    return text[:-1] if len(text) > 1 and text.endswith("/") else text


# W_click_navigated: SPA router บางตัว transition ช้า poll อีก 2 วินาทีหลัง retry หมด (สั้นกว่า W34
# 5s เพราะอยู่ใน loop ที่ user รอ)
_CLICK_NAV_POLL_ATTEMPTS = 10
_CLICK_NAV_POLL_INTERVAL_SEC = 0.2


async def _dom_signature(page: Page) -> Optional[int]:
    """W_click_navigated: ความยาว body.innerHTML — สัญญาณสำรองของ SPA ที่ไม่เปลี่ยน URL
    (วิธีเดียวกับ crawler._wait_for_dom_stable) None ถ้าอ่านไม่ได้ ห้าม throw"""
    try:
        value = await page.evaluate("document.body ? document.body.innerHTML.length : 0")
        return int(value)
    except Exception:
        return None


# W_rejected_submit_reports_success (gate 8c0f68a, login_checkout): กด Continue ฟอร์มว่าง หน้าขึ้น
# "Error: First Name is required" แต่ผลคือ "[OK] click succeeded" โมเดลกดซ้ำจนหมด step — toast
# detector ตอบแค่ "สำเร็จไหม" และ gate ด้วย _SAVE_LABEL_RE (ไม่มี Continue/Next)
# รายงานเฉพาะข้อความที่เพิ่งโผล่หลังคลิก (เทียบ before/after)
_ERROR_MESSAGE_SELECTOR = (
    '[role="alert"]:not(:empty), [aria-live="assertive"]:not(:empty), '
    '[class*="error" i]:not(:empty), [class*="invalid" i]:not(:empty), '
    '[class*="danger" i]:not(:empty)'
)
_MAX_ERROR_MESSAGE_CHARS = 160


async def _visible_error_texts(page: Page) -> set:
    """ข้อความ error ที่ผู้ใช้มองเห็นอยู่ตอนนี้ — ห้าม throw (กฎเดียวกับ state_filter)"""
    try:
        texts = await page.evaluate(
            """(sel) => Array.from(document.querySelectorAll(sel))
                 .filter((el) => el.offsetParent !== null)
                 .map((el) => (el.innerText || "").trim())
                 .filter((t) => t && t.length <= 160)""",
            _ERROR_MESSAGE_SELECTOR,
        )
    except Exception:
        return set()
    # ค่าที่ไม่ใช่ list (mock ในเทสต์ / evaluate ที่คืนอย่างอื่น) = อ่านไม่ได้ ไม่ใช่ "ไม่มี
    # error" — คืน set ว่างเหมือนกัน แต่ห้าม throw ออกไปให้ execute() พังเด็ดขาด
    if not isinstance(texts, list):
        return set()
    return {t for t in texts if isinstance(t, str)}

async def _dispatch_click_with_retry(page: Page, index: int, label: str = "") -> ActionResult:
    """_dispatch_with_retry เฉพาะ click: รอบ retry ที่ 2 ขึ้นไป hover() ก่อนคลิก (ปุ่ม hover-to-reveal)
    รอบแรกคลิกตรงไม่ hover (ปุ่มส่วนใหญ่ไม่ต้อง) ผลของ hover ไม่นับ

    W23: click-family ทุกตัว (click/submit/delete/purchase/pay) ผ่านที่นี่ จึงเป็นจุดเดียวที่เช็ค+resolve
    confirmation modal ก่อนคืนผลให้ LLM"""
    # W_click_navigated: อ่าน URL/DOM/error "ก่อนคลิก" ไว้ตัดสิน timeout ตอนท้าย
    url_before = _normalize_click_url(page.url)
    dom_before = await _dom_signature(page)
    errors_before = await _visible_error_texts(page)
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
            # W63[7.1]: เช็ค toast เฉพาะ label Save/Submit/Confirm และไม่ใช่ flow modal
            else:
                toast_text = (
                    await _detect_success_toast(page)
                    if label and _SAVE_LABEL_RE.search(label) else ""
                )
                stayed = _normalize_click_url(page.url) == url_before
                new_errors = (
                    await _visible_error_texts(page) - errors_before if stayed else set()
                )
                note = ""
                if toast_text:
                    note = f' [Success confirmation found: "{toast_text}"]'
                elif new_errors:
                    # W_rejected_submit_reports_success: ฟอร์มที่ถูกปฏิเสธไม่พาไปไหน และ
                    # ข้อความที่มันขึ้นคือคำตอบว่าทำไมคลิกนี้ไม่ได้ผล
                    shown = "; ".join(sorted(new_errors))[:_MAX_ERROR_MESSAGE_CHARS]
                    note = (
                        f' [The page rejected this: "{shown}" — the click went through but '
                        "nothing was submitted. Fix what the message asks for, then try again; "
                        "clicking the same button again changes nothing.]"
                    )
                elif label and _SAVE_LABEL_RE.search(label):
                    # W_no_toast_is_not_a_reason_to_repeat (gate bf637ce, MiniWoB click-checkboxes): ได้คะแนน
                    # เต็มแล้วแต่ข้อความ "ไม่พบ toast — ตรวจ error" ทำให้กด Submit ซ้ำจนหมด step
                    # (add_candidate กด Save ซ้ำ 4 ครั้ง) มาถึงตรงนี้ = ไม่มีอะไรปฏิเสธ หลายเว็บไม่มี toast
                    note = (
                        " [No confirmation message appeared, and nothing on the page rejected "
                        "the click either — many sites simply show no toast. Pressing the same "
                        "button again will not make one appear: look at the data itself (the "
                        "list/table/page you changed) for evidence, or move on to the next step "
                        "of the goal.]"
                    )
                if note:
                    result = ActionResult(
                        result.success, result.action, f"{result.message}{note}",
                        locator_descriptor=result.locator_descriptor,
                        toast_confirmed=bool(toast_text),
                    )
            return result
        # W_click_navigated: URL เปลี่ยนแล้ว = attempt ก่อนคลิกโดน retry ไม่มีทางสำเร็จ ออกทันที
        if _normalize_click_url(page.url) != url_before:
            break
        if attempt < _ACTION_RETRIES:
            await asyncio.sleep(_ACTION_RETRY_DELAY_SEC)

    # W_click_navigated: poll รอ router ที่ transition ช้า เฉพาะตอน URL ยังไม่เปลี่ยน
    navigated = _normalize_click_url(page.url) != url_before
    if not navigated:
        for _ in range(_CLICK_NAV_POLL_ATTEMPTS):
            await asyncio.sleep(_CLICK_NAV_POLL_INTERVAL_SEC)
            if _normalize_click_url(page.url) != url_before:
                navigated = True
                break
    if navigated:
        # รายงานความจริงทั้งสองส่วน (timeout *และ* หน้าเปลี่ยน) ไม่กลบข้อความเดิม
        return ActionResult(
            True, result.action,
            f"the click reported a timeout, but the page navigated from {url_before} to "
            f"{_normalize_click_url(page.url)} — the click DID take effect. Read the new page's "
            f"indexed elements before deciding your next action (original error: {result.message})",
        )

    # W_click_navigated: SPA เปลี่ยน state ไม่เปลี่ยน URL — *ไม่* พลิกเป็น success (DOM เปลี่ยนจาก
    # toast/spinner ได้) คืน fail พร้อมหลักฐาน กันโมเดลคลิกซ้ำจนโดน loop guard
    dom_after = await _dom_signature(page)
    dom_note = ""
    if dom_before is not None and dom_after is not None and dom_after != dom_before:
        dom_note = (
            " [The page did not navigate, but its DOM did change after this click "
            f"({dom_before} -> {dom_after} characters) — this click may already have taken "
            "effect (a panel, dropdown or dialog may have opened). Look at the page's current "
            "indexed elements before repeating the same action]"
        )
    # W_modal_check_on_failure (P8/M4): เช็ค modal เดิมอยู่ใต้ if result.success — modal บล็อกคลิก
    # จึงไม่เคยถูกตรวจ (dead-end) บอกความจริงอย่างเดียว ไม่กดยืนยันให้ (dialog ที่ agent ไม่ได้เปิด)
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


# ══════════════════════════════════════════════════════════════════════
# โซน 5: confirmation modal
#   ทำอะไร: ตรวจ dialog ที่เพิ่งเปิดและกดยืนยันให้อัตโนมัติ (action ถูกอนุมัติไปแล้ว)
#   ทำงานยังไง: _detect_confirmation_modal() รอ dialog animate -> หาปุ่มยืนยัน (คำกลางๆ ก่อนคำทำลาย) -> คลิก force + รอปิด -> ยังค้างก็ page.reload()
# ══════════════════════════════════════════════════════════════════════
# W23 (Confirmation Modal Handler): agent เปิด "Are you Sure?" แล้วค้าง ไม่กด "Yes, Delete" —
# resolve อัตโนมัติทันทีหลัง click ไม่ขอ human ซ้ำ (อนุมัติ action ที่เปิด modal ไปแล้ว ปุ่มยืนยัน
# เป็นแค่ UX ถามซ้ำ)
# W_dialog_generic (C3): เดิม 2 ใน 3 selector เป็นของ OrangeHRM <dialog>/aria-modal/Bootstrap/MUI/
# antd/Radix/SweetAlert ตรวจไม่เจอ เรียง generic ก่อน framework (ตัวนี้ตอบแค่ "มีไหม")
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

# W_modal_confirm_generic (P3.9): fallback เดิมแค่ "Confirm" — "Yes"/"OK"/"ตกลง"/"Löschen" ไม่ match
# agent ค้างที่ modal ภาษาครอบชุดเดียวกับ RISKY_LABEL_KEYWORDS แต่ไม่ import (คนละคำถาม ผูกกันแล้ว
# แก้ชุดหนึ่งกระทบอีกชุด)
# ลำดับสำคัญ: ยืนยันกลางๆ (yes/ok/confirm) ก่อนคำทำลาย เพราะ has-text() เป็น substring
# ("Do not delete" match delete) คำปฏิเสธไม่อยู่ในลิสต์โดยตั้งใจ
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
    # W_modal_confirm_generic: จำกัดใน dialog container (กันปุ่มชื่อเดียวกันนอกโมดัล) รวม
    # container x tag เป็น selector เดียวต่อคำ — คงลำดับคำ ยิง locator 30 ครั้งแทน 120
    *[
        ", ".join(
            f'{container} {tag}:has-text("{text}")'
            # W_dialog_generic: ใช้ container ชุดเดียวกับ _DIALOG_CONTAINER_SELECTORS (เจอ dialog
            # แต่หาปุ่มไม่เจอแย่กว่าไม่เจอเลย)
            for container in _DIALOG_CONTAINER_SELECTORS
            for tag in ("button", '[role="button"]')
        )
        for text in _MODAL_CONFIRM_TEXTS
    ],
]

# W_modal_appear_race: รอ dialog ที่ animate เข้ามา 600ms (transition ทั่วไป 150-300ms)
_MODAL_APPEAR_TIMEOUT_MS = 600

_MODAL_DETACH_TIMEOUT_MS = 5000

# W24 (Auto-Refresh & Re-attachment): batch ลบหลายรอบ ปุ่ม modal รอบ 2+ ไม่ตอบสนอง กด F5 แล้วหาย
# = state desync หลัง AJAX reload ไม่ใช่ selector — retry ก่อน แล้ว fallback page.reload()
_MODAL_CONFIRM_CLICK_RETRIES = 3
_MODAL_CONFIRM_RETRY_DELAY_SEC = 1.0
_MODAL_RELOAD_TIMEOUT_MS = 15000


async def _detect_confirmation_modal(page: Page) -> bool:
    """W23: มี dialog ที่มองเห็นอยู่ไหม ไม่ throw (เช็คไม่ได้ = ไม่มี)

    W_modal_appear_race (C2): เรียกทันทีหลัง click แต่ dialog ยัง animate ไม่เข้า DOM สรุปว่าไม่มี
    แล้ว click ถัดไป fail ทั้งหมด (dead-end) — รอสั้นๆ ก่อนสรุป ไม่กรองด้วย _dom_signature()
    (modal ที่สลับ class ความยาว innerHTML ไม่เปลี่ยน) ราคา: click ธรรมดาเสีย _MODAL_APPEAR_TIMEOUT_MS
    ซึ่งน้อยกว่าคลิกพลาดครั้งเดียว (~18s)"""
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
    """W24: ตรรกะเดียวกับ _detect_confirmation_modal() (ไม่รอ) ใช้เช็คซ้ำใน resolve_confirmation_modal()"""
    try:
        locator = page.locator(_DIALOG_CONTAINER_SELECTOR).first
        if await locator.count() == 0:
            return False
        return await locator.is_visible(timeout=_ELEMENT_ACTION_TIMEOUT_MS)
    except Exception:
        return False


async def _find_visible_modal_confirm_button(page: Page):
    """W23/W24: ปุ่มยืนยันตัวแรกที่ visible (เจาะจงก่อน) -> (selector, locator) หรือ (None, None)
    หาครั้งเดียวแล้วคลิกซ้ำตัวเดิม (query ใหม่ทุกรอบเสี่ยงได้ element คนละตัวหลัง desync)"""
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
    """W23/W24: คลิกปุ่มยืนยันด้วย force=True (modal ที่ animate ยังไม่ visible ตามนิยาม Playwright)
    คลิก+รอ detach สูงสุด _MODAL_CONFIRM_CLICK_RETRIES ครั้ง ยังค้าง = desync -> page.reload()

    คืนข้อความต่อท้าย message ของ action หลัก (กรณี reload บอกให้กรองใหม่เพราะ client state หาย)
    None ถ้าไม่เจอปุ่มยืนยันเลย ไม่ throw ไม่ทำให้ action หลักกลายเป็น fail"""
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
            # detach timeout: modal อาจปิดแล้วแค่ซ่อนด้วย CSS หรือยังค้างจริง — เช็คก่อน ไม่เดา
            if not await _is_modal_still_open(page):
                await wait_stable(page)
                return f" [Confirmation modal detected — confirmed automatically ({clicked_selector})]"
            if attempt < _MODAL_CONFIRM_CLICK_RETRIES:
                await asyncio.sleep(_MODAL_CONFIRM_RETRY_DELAY_SEC)

    # W24: retry ครบยังค้าง -> reload (จำลอง F5) ไม่ throw แม้ reload fail — แจ้ง LLM ดีกว่าทำให้
    # action หลักที่สำเร็จแล้วกลายเป็น fail
    try:
        await page.reload(timeout=_MODAL_RELOAD_TIMEOUT_MS)
        await page.wait_for_load_state("networkidle", timeout=_MODAL_RELOAD_TIMEOUT_MS)
    except Exception:
        pass
    return (
        f" [The confirmation modal's confirm button was unresponsive after {_MODAL_CONFIRM_CLICK_RETRIES} attempts — the system reloaded the page automatically to resync state (as if pressing F5). Check the indexed elements of this freshly reloaded page, then navigate/re-apply the filter the goal needs before continuing, because the reload wiped the previous state (e.g. the search term you had filtered by)]"
    )


# ══════════════════════════════════════════════════════════════════════
# โซน 6: กดคีย์ / กรอก / ค่าลับ / เลือก / ติ๊ก
#   ทำอะไร: press_key, fill, fill_secret, select_option, check
#   ทำงานยังไง: fill เคลียร์ด้วยคีย์บอร์ดก่อนพิมพ์และปิด popup, select เทียบ option แบบ normalize, check มี fallback force/JS click แล้ว verify สถานะจริง
# ══════════════════════════════════════════════════════════════════════
# W50: custom dropdown/menu (role=option/menuitem) คลิกตัวเลือกตรงๆ พลาดบ่อย — click เปิดแล้วส่ง
# ArrowDown/Enter ไปที่ element เดิม (keyboard navigation ของ widget) เสถียรกว่า
async def press_key(page: Page, index: int, key: str, timeout: int = _ELEMENT_ACTION_TIMEOUT_MS) -> ActionResult:
    """ส่ง key ไปยัง element ตาม index — locator.press() focus element ให้ก่อน (page.keyboard.press()
    ยิงไปที่ focus ปัจจุบัน ไม่รับประกันว่าเป็นตัวที่ LLM สั่ง)"""
    try:
        selector = _sel(index)
        target = await resolve_frame(page, selector)
        await target.press(selector, key, timeout=timeout)
        return ActionResult(True, f"press_key({index}, {key})", f"pressed key '{key}' succeeded")
    except PWTimeout:
        return ActionResult(False, f"press_key({index}, {key})", "element not found / key press failed (timeout)")
    except Exception as e:
        return ActionResult(False, f"press_key({index}, {key})", f"error: {e}")


# W19 (Safe Input Replacement): controlled input (React/Vue เช่นช่องค้นหา YouTube) ไม่เห็นการเคลียร์
# แบบ CDP ของ .fill() state ภายในยังอ้างคำเดิม — focus -> Ctrl+A -> Backspace (keyboard event จริง)
# แล้วค่อย .fill() ลงช่องที่ว่างแล้ว
# W_fill_wrapper_resolves_to_inner_input: เช็คว่ากรอกได้ตามนิยามของ Playwright กับ element จริง
# ไม่เดาจาก tag ที่ perception รายงาน
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
    """selector ของ element ที่กรอกได้จริง — ตัวเดิม หรือช่องกรอกตัวแรกข้างในถ้าเป็นกล่องครอบ
    (SPA ห่อ <input> ใน div ที่ถือ label) ไม่มีเลยคืนตัวเดิมให้ Playwright บอก error ตามจริง"""
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
    """พิมพ์ข้อความลงช่องตาม index — เคลียร์ด้วย focus -> select-all -> Backspace ก่อนเสมอ

    W_datepicker (OrangeHRM Leave List): focus เปิด date-picker popup ที่แทรกปุ่มเข้า DOM index ถัดไป
    เลื่อน "To Date" ไปชี้ปุ่มปฏิทินเงียบๆ — ปิด popup ทันทีหลัง fill สำเร็จ
    Escape ไม่ได้ผล (listener เป็น outside-click) ใช้ blur() + คลิก <body> (ไม่คลิกพิกัดที่อาจโดนลิงก์)
    best-effort ห่อ try/except ไม่ให้ fill ที่สำเร็จกลายเป็น fail"""
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
            # W_check_evaluate_timeout: evaluate() ไม่ระบุ timeout รอได้ 30s ทั้งที่เป็นแค่ขั้นเสริม
            await target.locator(selector).evaluate(
                "el => { el.blur(); document.body.click(); }", timeout=_STATE_READ_TIMEOUT_MS,
            )
            # popup ปิดแบบ async (~100-300ms transition) ไม่รอ get_snapshot() ถัดไปยังเห็นปุ่มค้าง
            await target.wait_for_timeout(200)
        except Exception:
            pass
        descriptor = await compute_locator_descriptor(target, selector)
        return ActionResult(True, f"fill({index})", f"filled '{text}' succeeded", locator_descriptor=descriptor)
    except PWTimeout:
        return ActionResult(False, f"fill({index})", "could not fill (timeout)")
    except Exception as e:
        return ActionResult(False, f"fill({index})", f"error: {e}")


# W65[3] (Vault Expansion — Current Password Auto-fill): reuse credential login ต่อโดเมน
# (site_learning/storage.py) เป็น current password แทนสร้าง multi-secret store ใหม่
# ค่าจริงห้ามหลุดเข้า LLM — LLM ส่งแค่ชื่อ symbolic ("current_password") และ message ไม่ echo ค่า
_SUPPORTED_SECRET_KEYS = {"current_password"}


async def fill_secret(page: Page, index: int, secret_key: str, timeout: int = _ELEMENT_ACTION_TIMEOUT_MS) -> ActionResult:
    """กรอกค่าลับที่บันทึกไว้ (รองรับแค่ "current_password") — ไม่มี credential/key ไม่รู้จัก
    คืน [FAIL] ให้ LLM ถาม user ตาม W65[1] ไม่ throw"""
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
    """เลือกตัวเลือกใน <select> ตาม index ด้วยข้อความที่เห็น

    W42: ดึง option {text, value} จริงมาเทียบแบบ normalize whitespace แล้วเลือกด้วย value= ของ
    option นั้น — select_option(label=...) เทียบ exact ภายในอีกที text ที่มี whitespace ยุ่ง
    ("  New   York  ") ยังไม่ตรง value ไม่มีปัญหานี้ (ไม่มี value= browser ใช้ text เดียวกัน)"""
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
    """W21: element ติ๊กแล้วจริงไหมหลัง force/JS click — is_checked() ต้องการ input/[role=checkbox]
    เป๊ะ (custom wrapper throw) ไล่สัญญาณ: (1) ตัวเองเป็น input/role=checkbox (2) มี checkbox
    ซ้อนข้างใน (3) aria-checked (4) class active/checked เช็คไม่ได้คืน False"""
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

    W21 (Custom UI Checkbox): native checkbox ถูกซ่อนแทนด้วย span (.oxd-checkbox-input)
    target.check() throw แน่นอน (deterministic ไม่ใช่ timing) — fallback 2 ชั้นจากรุกน้อยไปมาก
    แล้ว verify สถานะจริงทุกครั้ง (custom checkbox คลิกไม่ error ไม่ได้แปลว่าติ๊กแล้ว)"""
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

    # Fallback 1: force click — ข้าม actionability check คลิกกึ่งกลาง element
    try:
        target = await resolve_frame(page, selector)
        await target.click(selector, timeout=min(timeout, _ELEMENT_FALLBACK_TIMEOUT_MS), force=True)
        if await _is_effectively_checked(target, selector):
            descriptor = await compute_locator_descriptor(target, selector)
            return ActionResult(True, f"check({index})", "checked successfully (force click)", locator_descriptor=descriptor)
    except Exception:
        pass

    # Fallback 2: JS el.click() — สำหรับ wrapper ที่ force click ไม่โดน (0x0 วาดด้วย pseudo-element)
    try:
        target = await resolve_frame(page, selector)
        # W_check_evaluate_timeout (2026-09-03): evaluate() ไม่ระบุ timeout ค้าง 30s ที่ step สุดท้าย
        # หลังงานเสร็จแล้ว — คลาสเดียวกับ W_descriptor_timeout
        await target.locator(selector).evaluate(
            "el => el.click()", timeout=min(timeout, _ELEMENT_FALLBACK_TIMEOUT_MS),
        )
        if await _is_effectively_checked(target, selector):
            descriptor = await compute_locator_descriptor(target, selector)
            return ActionResult(True, f"check({index})", "checked successfully (JS click)", locator_descriptor=descriptor)
        return ActionResult(False, f"check({index})", "clicked, but the checked state could not be confirmed")
    except Exception as e:
        return ActionResult(False, f"check({index})", f"error: {e}")


# ══════════════════════════════════════════════════════════════════════
# โซน 7: นำทาง + สถานะหน้า
#   ทำอะไร: scroll, goto, go_back, switch_tab, wait_stable
#   ทำงานยังไง: ทุกตัวรายงานผลตามจริง (เลื่อนได้เท่าไหร่ / URL ก่อน-หลัง) ปฏิเสธการไปหน้าว่าง
# ══════════════════════════════════════════════════════════════════════
async def scroll(page: Page, direction: str = "down", amount: int = 600) -> ActionResult:
    """เลื่อนหน้าจอ ('down'/'up')

    W_inner_scroll (ดู state_filter.py::_FIND_SCROLLER_FN_JS): mouse.wheel() เลื่อนอะไรก็ได้ใต้เมาส์
    (ไม่มีใครคุมตำแหน่ง) — เลื่อน element ตัวเดียวกับที่ check_scroll_redundant() ใช้ รายงานระยะจริง
    เลื่อนไม่ได้ = success=False evaluate ล้มเหลว fallback mouse.wheel แบบเดิม"""
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


# W_blank_navigation_destroys_the_task (gate 2026-09-07, MiniWoB click-checkboxes): หลัง go_back
# หน้าว่าง โมเดล goto url ว่างไป about:blank แล้ว "สำเร็จ" ทุก action หลังจากนั้นล้ม — ไม่มี goal
# ไหนตั้งใจไปหน้าว่าง ปฏิเสธตรงๆ (หลักเดียวกับ check_fill_is_empty_noop)
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
    W_blank_navigation_destroys_the_task: go_back() คืน None เมื่อไม่มีประวัติ เดิมรายงานสำเร็จเสมอ
    — เทียบ URL ก่อน/หลังแล้วรายงานตามจริง"""
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


# W19 (latency): 8000 -> 4000ms — networkidle ไม่ resolve บนเว็บที่ polling/analytics ตลอด ทุก
# page-changing action รอเต็ม timeout timeout ไม่เคยทำให้ fail (PWTimeout = success) ลดเพดานจึงปลอดภัย
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


# ══════════════════════════════════════════════════════════════════════
# โซน 8: read_page_data + นับแบบ deterministic
#   ทำอะไร: ตอบคำถามเรื่องเนื้อหาหน้าเว็บ (นับ/อ่านตาราง) และแนบตัวเลขที่โค้ดนับเอง
#   ทำงานยังไง: คำถามเชิงนับ -> count_elements (มีเงื่อนไข key=value อ่านแถวแล้วนับเฉพาะคอลัมน์) / คำถามอื่น -> extract_table_data -> _deterministic_count_note แนบ "[counted by the system]"
# ══════════════════════════════════════════════════════════════════════
# W45: read_page_data แยก 2 lane ตามต้นทุน token — Lane 1 นับ (count_elements, deterministic)
# vs Lane 2 อ่านตาราง (extract_table_data ให้ LLM ตีความ) เลือก lane ในโค้ด คำถามเชิงนับนับตรงเสมอ
# (คู่กับกฎใน SYSTEM_PROMPT defense-in-depth)
_COUNT_QUERY_KEYWORDS = ("กี่", "จำนวน", "นับ", "how many", "count", "number of")

# W_deterministic_count (OrangeHRM 2026-08-26): โมเดลอ่านตารางถูกแต่นับผิด (ESS 7 ตอบ 6) — การนับ
# deterministic โค้ดนับแล้วแนบตัวเลขไป เงื่อนไขดึงเฉพาะรูป "key=value" ชัดเจน (ไม่เดาจากคำทั่วไป
# "user"/"role" ที่อยู่ทุกแถว)
# W_column_aware_count: เก็บทั้งสองฝั่ง "=" — ฝั่งซ้ายคือคอลัมน์ ("ess.irhrg0" ที่ Role เป็น Admin
# เคยถูกนับเป็น ESS) ดู _matching_entry_count()
_KEY_VALUE_IN_QUERY_RE = re.compile(r"([\w฀-๿]+)\s*=\s*([\w.\-@]+)")
_MARKDOWN_SEPARATOR_CELL_RE = re.compile(r"^:?-{2,}:?$")


def _extracted_entries(data: str) -> list[str]:
    """แปลงผล extract_table_data() (markdown table หรือ JSON list) เป็นรายการต่อ entry
    parse ไม่ได้คืน [] ผู้เรียกข้ามการนับ (ไม่มีตัวเลขดีกว่าตัวเลขผิด)"""
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


# W_count_answer_check: อ่านตัวเลขที่ _deterministic_count_note() เขียนกลับออกมา — วางชิดกันเพราะต้อง
# เปลี่ยน format พร้อมกันเสมอ (orchestrator ไม่ต้องเดา format เอง)
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
    """W_column_aware_count: markdown table -> (หัวตาราง, แถวแยกเซลล์) ไม่ใช่ตาราง ([], [])
    ผู้เรียก fallback นับทั้งแถว"""
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
    """W_column_aware_count: "User Role"/"user_role"/"userrole" เทียบติด — ตัดที่ไม่ใช่ตัวอักษร/ตัวเลข + lower"""
    return re.sub(r"[^a-z0-9ก-๙]", "", (text or "").lower())


def _matching_entry_count(
    entries: list[str], header: list[str], rows: list[list[str]], key: str, value: str,
) -> tuple[int, str]:
    """W_column_aware_count: (จำนวนแถวที่ตรง, คำอธิบายที่มา) — มีคอลัมน์ชื่อตรงฝั่งซ้ายของ "="
    เทียบเฉพาะเซลล์นั้น ไม่งั้น fallback ทั้งแถว (ที่เคยนับ "ess.irhrg0 | Admin" เป็น ESS)"""
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
            # W_conditional_count: เดิมเงียบตอนได้ 0 โมเดลรายงาน "exactly N" เป็นคำตอบ — บอก 0 ตามจริง
            # พ่วงเงื่อนไขของ W_confident_zero (0 ไม่ใช่คำตอบจนกว่าจะแน่ใจว่าอ่านตารางถูกตัว)
            notes.append(
                f"[counted by the system] 0 of those {total} entries contain "
                f"'{value.strip()}' {where}. If that contradicts what you can see on the page, the data "
                "below is not the right table — read it again with a different target_hint "
                "instead of reporting 0."
            )
    return "\n".join(notes) + "\n"


async def read_page_data(page: Page, query: str, target_hint: str) -> ActionResult:
    """query: คำถาม — ใช้เลือก lane (นับ vs อ่านตาราง) และถ้าไม่ใช่คำถามนับ ส่งเป็นค่าที่ lookup ใน
    extract_table_data() target_hint: CSS selector ที่ LLM เดาว่าตรงข้อมูล

    ไม่ผ่าน _dispatch_with_retry — hint ไม่ตรงเป็น deterministic mismatch ไม่ใช่ timing
    wait_stable() ก่อนอ่านเสมอ (หลัง submit DOM อาจยังไม่ bind ค่าใหม่) timeout แล้วเดินต่อ"""
    if not target_hint:
        return ActionResult(False, "read_page_data", "target_hint (a CSS selector) is required")

    await wait_stable(page)

    is_count_query = any(kw in query.lower() for kw in _COUNT_QUERY_KEYWORDS)
    try:
        if is_count_query:
            # W_conditional_count (2026-08-26): เส้นทางหลัก (count > 0) เคยคืน "found N entries" ดิบ
            # ไม่รู้จักเงื่อนไข ("userrole=ess" + '[role="row"]' ได้ 21 ทั้งที่จริง 4 — success=True
            # ตอบผิดแบบมั่นใจ) มีเงื่อนไข key=value ต้องอ่านแถวแล้วนับเฉพาะที่ตรง (ตัวนับเดียวกับ
            # _deterministic_count_note)
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
                    # อ่านแถวไม่ได้แต่ selector นับได้ — บอกตามจริงว่าเป็นจำนวน element ไม่ใช่รายการที่ตรงเงื่อนไข
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
            # W_confident_zero (OrangeHRM 2026-08-26): count=0 ทั้ง "ไม่มีจริง" และ "selector ผิด" (เดา
            # "table tbody tr" บน ARIA grid) เดิมตอบ "found 0" อย่างมั่นใจทั้งที่มี 5 — 0 ไม่ใช่คำตอบ
            # จนกว่าจะพิสูจน์: ลอง extract_table_data() (มี fallback ครบ) อ่านได้ = selector ผิด
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
            # W_query_is_a_question (saucedemo ล้ม 3 step ติด): extract_table_data() ตีความ query เป็น
            # ค่าที่ต้องหาในตาราง แต่โมเดลส่งคำถามธรรมชาติ ("first product name") ได้ [FAIL] ทั้งที่มีข้อมูล
            # — ลองซ้ำไม่ส่ง query ได้ข้อมูลก็คืนพร้อมบอกว่าไม่เจอข้อความตรงตัว (เจตนา W46 ห้ามแกล้งเจอ)
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


# ══════════════════════════════════════════════════════════════════════
# โซน 9: permission layer
#   ทำอะไร: กัน action เสี่ยงก่อน dispatch และขออนุมัติจากคน
#   ทำงานยังไง: classify_action() -> BLOCKED ปฏิเสธ / NEEDS_CONFIRMATION ถาม ask_user_func (หรือ input())
# ══════════════════════════════════════════════════════════════════════
# ask_user_func จริง — PR เดิมรับ param แล้วยังเรียก input() ตรงๆ) ---

# สัญญาณเดียวว่ามนุษย์ปฏิเสธจริง (ต่างจาก timeout/BLOCKED) — memory.ShortTermMemory ใช้แยก refusal
# (rejected_actions_summary) จึงเป็น constant ไม่ hardcode ซ้ำ
REJECTED_BY_USER_MESSAGE = "The user refused to perform this action (human-in-the-loop)"


async def _confirm_action(cmd: dict, ask_user_func: Optional[AskUserFunc], label: str = "") -> bool:
    """label: ชื่อ element จริง (เช่น "Place Order") แนบในสำเนา cmd เป็น "element_label" (key "label"
    ของ cmd มีความหมายอื่นสำหรับ select) ให้การ์ด permission โชว์ชื่อแทน index (แบบ W10[D])"""
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


# ══════════════════════════════════════════════════════════════════════
# โซน 10: execute() — ทางเข้าเดียวของ agent loop + compound action
#   ทำอะไร: รับ cmd dict จาก LLM แล้ว dispatch ไป action ที่ถูกตัว
#   ทำงานยังไง: เช็ค permission -> state_filter (ข้าม action ที่ไม่จำเป็น/ผิดชนิด) -> dispatch ผ่าน retry -> then_click_index/key พ่วงต่อ (ผ่าน permission ซ้ำ) -> คืน ActionResult
# ══════════════════════════════════════════════════════════════════════
async def _check_permission(
    cmd: dict, ask_user_func: Optional[AskUserFunc], label: str, manual_guidance: str,
    allowed_domains: Optional[set], element_tag: str, element_type: str,
) -> Optional[str]:
    """W_chain: permission check แยกเป็นฟังก์ชันเพื่อเรียกซ้ำกับ action ที่สอง (then_click_index)
    คืน None = ทำต่อได้ (SAFE หรือได้รับอนุมัติ) หรือข้อความ error ถ้าถูกบล็อก/ปฏิเสธ"""
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
    """W_chain: cmd มี then_click_index -> คลิก element ที่สองต่อทันทีรวมเป็น ActionResult เดียว
    (ลด round-trip) — fill + คลิก Submit ที่เห็นอยู่เชื่อถือได้กว่า key:"Enter" (บางหน้าไม่มี
    Enter-to-submit) ไม่ chain ถ้า primary fail หรือไม่มี then_click_index

    ความปลอดภัย: action ที่สองผ่าน _check_permission() เต็มรูปแบบ ต้องอนุมัติ/ถูกบล็อก -> คืนผล
    primary พร้อมบอกให้สั่งแยก ไม่มีทางข้าม human-in-the-loop
    manual_guidance/allowed_domains ใช้ค่าของ primary (ไม่ query RAG ซ้ำ เสีย latency ที่เพิ่งลด)"""
    then_index = cmd.get("then_click_index")
    # W_chain_partial_success: index ติดลบคือ sentinel "ไม่มี chain" ของบาง provider
    # (llm._normalize_openai_args) กรองทุก provider ไม่งั้นเสีย retry เปล่า
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
    # W_chain_partial_success (OrangeHRM/openai): เดิมคืน primary.success and second.success —
    # primary ที่เปลี่ยนหน้าแล้วถูกรายงาน [FAIL] เพราะ index ที่สองค้าง โมเดลคลิก primary ซ้ำ 10 step
    # ความคืบหน้าจริงต้องไม่หาย คืน primary.success แล้วบอกให้คลิกตัวที่สองแยก
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


async def _close_menu_covering(page: Page, index: int) -> bool:
    """ปิดเมนูที่เปิดค้างและบังเป้าหมาย -> True ถ้าปิดได้

    W_menu_overlay_blocks_target (gate 6a9f051, Add Candidate): Tab ท้าย Last Name เปิดเมนู Vacancy
    คลุมช่อง Email fill รอ 6.6s แล้วล้ม — เมนูไม่ได้เปิดจากเจตนาโมเดล ปิดให้เลยไม่เสียเทิร์น LLM
    Escape ยิงเฉพาะเมื่อยืนยันว่าเมนูบังจริง (dialog ไม่เข้าเงื่อนไข menu_overlay_covers_target)"""
    if not await state_filter.menu_overlay_covers_target(page, index):
        return False
    try:
        await page.keyboard.press("Escape")
        await page.wait_for_timeout(_MENU_DISMISS_WAIT_MS)
    except Exception:
        return False
    return not await state_filter.menu_overlay_covers_target(page, index)

async def _chained_button_blocked_by_checkbox_group(page, cmd: dict, then_type: str):
    """เหตุผลที่ห้ามพ่วงปุ่มต่อท้าย action นี้ (None = พ่วงได้)
    W_chained_submit_after_check: ใช้กับทั้ง check และ click — โมเดลสลับมา click checkbox (toggle)
    แล้วพ่วง Submit ทันที ปิดทางเดียวเท่ากับไม่ได้ปิด"""
    if cmd.get("then_click_index") is None:
        return None
    if not await state_filter.element_is_checkbox(page, cmd["index"]):
        return None
    group_size = await state_filter.checkbox_group_size(page, cmd["index"])
    if group_size < 2:
        return None
    if (then_type or "").lower() == "checkbox" or await state_filter.element_is_checkbox(
        page, cmd["then_click_index"]
    ):
        return None      # พ่วง checkbox ตัวอื่น = ติ๊กสองช่องรวด ไม่ใช่การส่งฟอร์มก่อนเวลา
    return (
        f"did not press the chained button: this is one of {group_size} checkboxes in the "
        "same group. Tick exactly the boxes the instruction names — no others — then "
        "press the button on its own turn."
    )

async def execute(
    page: Page, cmd: dict, ask_user_func: Optional[AskUserFunc] = None, label: str = "",
    manual_guidance: str = "", allowed_domains: Optional[set] = None, element_tag: str = "",
    element_type: str = "", then_label: str = "", then_tag: str = "", then_type: str = "",
) -> ActionResult:
    """รับคำสั่ง dict จาก LLM แล้ว dispatch เช่น
        {"type": "fill",   "index": 0, "text": "standard_user"}
        {"type": "click",  "index": 2}
        {"type": "select", "index": 2, "label": "Price (low to high)"}
        {"type": "scroll", "direction": "down"}
        {"type": "goto",   "url": "https://..."}

    เช็ค permission ก่อนเสมอ (permission/rules.py::classify_action): BLOCKED ปฏิเสธทันที,
    NEEDS_CONFIRMATION ถาม user (ask_user_func หรือ input())

    label: ข้อความของเป้าหมาย — เช็คคำเสี่ยงสำรอง (RISKY_LABEL_KEYWORDS)
    manual_guidance (W7[B]): คู่มือที่เกี่ยวกับ step นี้ (MANUAL_CONFIRMATION_KEYWORDS)
    allowed_domains: override ALLOWED_DOMAINS เฉพาะ call นี้ (None = ของ module)
    element_tag (W_search follow-up): tag ของเป้าหมาย (ANCHOR_TAG)
    element_type (W_search follow-up 2): attribute type (SAFE_INPUT_TAG/RISKY_INPUT_TYPES)
    then_label/then_tag/then_type (W_chain): สัญญาณเดียวกันของ then_click_index
    """
    t = cmd.get("type")

    denial = await _check_permission(
        cmd, ask_user_func, label, manual_guidance, allowed_domains, element_tag, element_type,
    )
    if denial is not None:
        return ActionResult(False, f"{t}", denial)

    try:
        # retry wrapper (W5) เฉพาะ click/fill/select/check — scroll/goto/wait fail แบบไม่ใช่ timing
        # click ใช้ _dispatch_click_with_retry() (hover ก่อนคลิกซ้ำ)
        # W19 State Filter: redundant -> short-circuit [OK] ไม่แตะ browser ยกเว้น click บน disabled
        # คืน success=False (ทำไม่ได้ ไม่ใช่ทำแล้ว) ให้หาทางอื่น
        if t == "click":
            redundant = await state_filter.check_click_redundant(page, cmd["index"])
            if redundant is not None:
                return ActionResult(False, f"click({cmd['index']})", f"[Skipped] {redundant}")
            # W_click_native_select: คลิก <select> จริงคือ no-op ที่คืน [OK] — เช็คก่อน dispatch
            # (หลังคลิกแยกไม่ออก)
            wrong_kind = await state_filter.check_click_target_is_native_select(page, cmd["index"])
            if wrong_kind is not None:
                return ActionResult(False, f"click({cmd['index']})", f"[Skipped] {wrong_kind}")
            # W_menu_overlay_blocks_target: เหมือนฝั่ง fill — เป้าที่ถูกเมนูเปิดค้างบังไว้จะ
            # ล้มด้วย timeout ทุกครั้ง ไม่ใช่เพราะ index ผิด
            menu_closed = await _close_menu_covering(page, cmd["index"])
            # W_chain_stale_index: ตัด then_click_index ถ้าคลิกนี้ทำให้ index เดิมใช้ไม่ได้ (เปิด dropdown
            # หรือเลือกตัวเลือก — ดู state_filter.check_click_invalidates_indexes) เช็คก่อน dispatch
            # W_dropdown_sets_filter_dirty: เรียกกับทุก click (orchestrator ต้องรู้ว่าเลือกค่าใน dropdown)
            # evaluate() ครั้งเดียวใช้สองงาน
            disturb_kind = await state_filter.classify_click_index_disturbance(page, cmd["index"])
            stale_chain_note = (
                state_filter.chain_hint_for_kind(disturb_kind)
                if cmd.get("then_click_index") is not None else None
            )
            result = await _dispatch_click_with_retry(page, cmd["index"], label)
            if menu_closed:
                result = replace(
                    result,
                    message=(f"{result.message} (an open dropdown menu was covering this "
                             "element — it was closed first)"),
                )
            if result.success and disturb_kind == "option":
                result = replace(result, dropdown_option_selected=True)
            # W_menu_open_note_needs_no_chain: คลิกที่ไม่มี chain ก็ต้องรู้ว่า index เลื่อนแล้ว
            if stale_chain_note is None and result.success:
                shift_note = state_filter.index_shift_note_for_kind(disturb_kind)
                if shift_note is not None:
                    result = replace(result, message=f"{result.message} ({shift_note})")
            if result.success:
                blocked = await _chained_button_blocked_by_checkbox_group(page, cmd, then_type)
                if blocked is not None:
                    return replace(result, message=f"{result.message} ({blocked})")
            if stale_chain_note is not None:
                # คลิกหลักสำเร็จ — รายงานตามจริงพร้อมเหตุผลที่ไม่ chain (แบบ W_chain_partial_success)
                return replace(
                    result,
                    message=f"{result.message} (did not go on to the chained click: {stale_chain_note})",
                )
            return await _maybe_chain_click(
                page, cmd, result, ask_user_func, manual_guidance, allowed_domains,
                then_label, then_tag, then_type,
            )
        if t == "fill":
            # W_file_input_guard (P3.10): fill ลง <input type=file> ไม่มีทางสำเร็จ เช็คก่อนทุกอย่าง
            # W_empty_fill_noop: เช็คก่อน check_fill_redundant() ("ว่าง -> ว่าง" ต้องเป็น failure)
            file_input = await state_filter.check_fill_target_is_file_input(page, cmd["index"])
            if file_input is not None:
                return ActionResult(False, f"fill({cmd['index']})", f"[Skipped] {file_input}")
            # W_fill_untypable_target: ตัวเปิด dropdown/ปุ่ม ไม่ใช่ช่องกรอก — แทน error ดิบของ Playwright
            # ด้วยทางออกที่ทำได้จริง (กระจกของ W_click_native_select)
            not_typable = await state_filter.check_fill_target_is_not_typable(page, cmd["index"])
            if not_typable is not None:
                return ActionResult(False, f"fill({cmd['index']})", f"[Skipped] {not_typable}")
            # W_menu_overlay_blocks_target: เมนูที่ Tab เปิดค้างไว้บังช่องถัดไป (ดู
            # _close_menu_covering) — ปิดก่อน ไม่งั้นรอ actionability จนหมดเวลา 6.6 วิแล้วล้ม
            menu_closed = await _close_menu_covering(page, cmd["index"])
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
            if menu_closed:
                result = replace(
                    result,
                    message=(f"{result.message} (an open dropdown menu was covering this "
                             "field — it was closed first)"),
                )
            if early_submit is not None and result.success:
                result = replace(
                    result,
                    message=f"{result.message} (did not submit the form: {early_submit})",
                )
            # W_chain: "key" ใช้กับ fill ได้ — กด key (ปกติ Enter) หลัง fill สำเร็จใน step เดียว
            # ไม่ต้องเช็ค permission ซ้ำ (press_key ไม่อยู่ใน DEFAULT_NEEDS_CONFIRMATION)
            if result.success and cmd.get("key"):
                key_result = await _dispatch_with_retry(press_key, page, cmd["index"], cmd["key"])
                result = ActionResult(
                    result.success and key_result.success, result.action,
                    f"{result.message} + then press_key({cmd['key']}): {key_result.message}",
                    locator_descriptor=result.locator_descriptor,
                )
            # W_chain follow-up (MiniWoB enter-text): fill ถูกเสมอ แต่บางหน้าไม่ฟัง Enter (<input> นอก
            # <form>) — then_click_index ใช้กับ fill ได้ "fill + คลิก Submit" เชื่อถือได้กว่า
            # W_chained_submit_after_fill (2026-09-04): เช็ค *หลัง* fill — เทิร์นที่สองกรอกทับแค่ Password
            # Confirm ยังค่าเก่า chained click กด Save -> 'Passwords do not match' (guard ของ orchestrator
            # คุมแค่ click ที่สั่งแยก)
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
            # W65[3]: ไม่เช็ค check_fill_redundant() — ต้องใช้ค่าจริงซึ่ง fill_secret ไม่มี (เสียแค่ optimization)
            return await _dispatch_with_retry(fill_secret, page, cmd["index"], cmd.get("secret", ""))
        if t == "select":
            # W_custom_dropdown: target ไม่ใช่ <select> จริง -> success=False (ใช้ action ผิดชนิด)
            wrong_kind = await state_filter.check_select_target_is_native(page, cmd["index"])
            if wrong_kind is not None:
                return ActionResult(False, f"select({cmd['index']})", f"[Skipped] {wrong_kind}")
            # W_select_reorders_the_page (gate 50eefd0, long_flow): select ตัวเรียง/กรองทำหน้าสลับ index
            # ที่พ่วงมาชี้ element อื่น — เทียบข้อความเป้าที่พ่วงก่อน/หลัง แทนการเดาว่า select ไหนเรียง
            then_index = cmd.get("then_click_index")
            then_text_before = (
                await state_filter.element_text_at(page, then_index)
                if then_index is not None else None
            )
            result = await _dispatch_with_retry(select_option, page, cmd["index"], cmd["label"])
            if result.success and then_index is not None:
                then_text_after = await state_filter.element_text_at(page, then_index)
                if then_text_before != then_text_after:
                    return replace(
                        result,
                        message=(
                            f"{result.message} (did not go on to the chained click: choosing "
                            f"this value re-ordered the page, so index {then_index} is now "
                            f'"{then_text_after or "gone"}" instead of '
                            f'"{then_text_before or "unknown"}" — take a fresh look before '
                            "clicking)"
                        ),
                    )
            return await _maybe_chain_click(
                page, cmd, result, ask_user_func, manual_guidance, allowed_domains,
                then_label, then_tag, then_type,
            )
        if t == "check":
            # W_check_fires_a_button: ต้องเช็คก่อน redundant — เป้าที่ไม่ใช่ checkbox เลย
            # ตอบ "ติ๊กอยู่แล้วหรือยัง" ไม่ได้ตั้งแต่ต้น และ check() จะไปกดมันจริง
            not_checkable = await state_filter.check_check_target_is_not_checkable(
                page, cmd["index"])
            if not_checkable is not None:
                return ActionResult(
                    False, f"check({cmd['index']})", f"[Skipped] {not_checkable}")
            redundant = await state_filter.check_checkbox_redundant(page, cmd["index"])
            if redundant is not None:
                return ActionResult(True, f"check({cmd['index']})", f"[Skipped] {redundant}")
            result = await _dispatch_with_retry(check, page, cmd["index"])
            # W_chained_submit_after_check (gate 2026-09-07, MiniWoB click-checkboxes): พ่วง Submit มากับ
            # การติ๊กช่องแรก episode จบได้ 0 — ต้นทางคือกฎ prompt (_PROMPT_SEARCH_SUBMIT) ที่ถูกกับช่อง
            # ค้นหาแต่ผิดกับกลุ่ม checkbox แก้ prompt อย่างเดียวไม่พอ
            # ราคาเมื่อเดาผิด: หนึ่ง step (ถูกกว่าส่งฟอร์มก่อนเวลาที่กู้ไม่ได้) ไม่ตัด chain ไป checkbox
            # ตัวอื่น ตรวจจาก DOM ไม่เชื่อ then_type (optional default "")
            # เวอร์ชันแรกบอกรายชื่อช่องที่ยังไม่ติ๊ก โมเดลติ๊กเกินโจทย์ — กฎที่ถูก: "กลุ่มหลายช่อง ไม่พ่วงเลย"
            if result.success:
                blocked = await _chained_button_blocked_by_checkbox_group(page, cmd, then_type)
                if blocked is not None:
                    return replace(result, message=f"{result.message} ({blocked})")
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
            # submit/delete/purchase/pay = risk category ของ click ตัวเดิม (อนุมัติแล้ว) — คืน label
            # เดิม ("submit(2)") ใช้ wrapper เดียวกับ click รวม hover-on-retry
            result = await _dispatch_click_with_retry(page, cmd["index"], label)
            # W64[7.2]: คง locator_descriptor/toast_confirmed ไว้ (เดิม re-wrap ทิ้งเงียบๆ)
            return ActionResult(
                result.success, f"{t}({cmd['index']})", result.message,
                locator_descriptor=result.locator_descriptor, toast_confirmed=result.toast_confirmed,
            )
        return ActionResult(False, f"unknown({t})", "unknown action")
    except KeyError as e:
        return ActionResult(False, f"{t}", f"missing parameter: {e}")

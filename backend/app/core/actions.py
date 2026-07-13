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

ใช้คู่กับ perception.py (ไฟล์เดียวกับ W2)
"""

import asyncio
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional
from playwright.async_api import Page, TimeoutError as PWTimeout

from backend.app.permission.rules import DEFAULT_NEEDS_CONFIRMATION, ActionRisk, classify_action

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

    def __str__(self):
        mark = "OK" if self.success else "FAIL"
        return f"[{mark}] {self.action} -> {self.message}"


# selector ที่ผูกกับ index ที่ perception ติดไว้บน element
def _sel(index: int) -> str:
    return f'[data-ai-index="{index}"]'


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

async def click(page: Page, index: int, timeout: int = 5000) -> ActionResult:
    """คลิก element ตาม index"""
    try:
        await page.click(_sel(index), timeout=timeout)
        return ActionResult(True, f"click({index})", "คลิกสำเร็จ")
    except PWTimeout:
        return ActionResult(False, f"click({index})", "หา element ไม่เจอ/คลิกไม่ได้ (timeout)")
    except Exception as e:
        return ActionResult(False, f"click({index})", f"error: {e}")


async def fill(page: Page, index: int, text: str, timeout: int = 5000) -> ActionResult:
    """พิมพ์ข้อความลงช่อง input/textarea ตาม index"""
    try:
        await page.fill(_sel(index), text, timeout=timeout)
        return ActionResult(True, f"fill({index})", f"กรอก '{text}' สำเร็จ")
    except PWTimeout:
        return ActionResult(False, f"fill({index})", "กรอกไม่ได้ (timeout)")
    except Exception as e:
        return ActionResult(False, f"fill({index})", f"error: {e}")


async def select_option(page: Page, index: int, label: str, timeout: int = 5000) -> ActionResult:
    """เลือกตัวเลือกใน dropdown (<select>) ตาม index — เลือกด้วยข้อความที่เห็น"""
    try:
        await page.select_option(_sel(index), label=label, timeout=timeout)
        return ActionResult(True, f"select({index})", f"เลือก '{label}' สำเร็จ")
    except PWTimeout:
        return ActionResult(False, f"select({index})", "เลือกไม่ได้ (timeout)")
    except Exception as e:
        # เผื่อ label ไม่ตรงเป๊ะ -> ลองเลือกด้วย value แทน
        try:
            await page.select_option(_sel(index), value=label, timeout=timeout)
            return ActionResult(True, f"select({index})", f"เลือก (by value) '{label}' สำเร็จ")
        except Exception as e2:
            return ActionResult(False, f"select({index})", f"error: {e2}")


async def check(page: Page, index: int, timeout: int = 5000) -> ActionResult:
    """ติ๊ก checkbox/radio ตาม index"""
    try:
        await page.check(_sel(index), timeout=timeout)
        return ActionResult(True, f"check({index})", "ติ๊กสำเร็จ")
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


async def wait_stable(page: Page, timeout: int = 8000) -> ActionResult:
    """รอให้หน้าเว็บนิ่ง — เรียกหลังทุก action ที่ทำให้หน้าเปลี่ยน ก่อน snapshot รอบใหม่"""
    try:
        await page.wait_for_load_state("networkidle", timeout=timeout)
        return ActionResult(True, "wait_stable", "หน้านิ่งแล้ว")
    except PWTimeout:
        # ไม่ถือเป็น fail ร้ายแรง — บางหน้ามี network ยิงตลอด
        return ActionResult(True, "wait_stable", "timeout แต่เดินต่อได้")
    except Exception as e:
        return ActionResult(False, "wait_stable", f"error: {e}")


# ------------------------------------------------------------
# Permission layer: กัน action เสี่ยง/บล็อก ก่อนถึง dispatch จริง
# adapted จาก PR "permission-ab" — จุดต่างจาก PR เดิม: ask_user_func ถูกใช้งานจริง
# (PR เดิมรับ param นี้มาแต่ไม่ได้เรียกใช้เลย ยังเรียก input() ตรงๆ เสมอ)
# ------------------------------------------------------------
async def _confirm_action(cmd: dict, ask_user_func: Optional[AskUserFunc]) -> bool:
    if ask_user_func is not None:
        return bool(await ask_user_func(cmd))
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
async def execute(page: Page, cmd: dict, ask_user_func: Optional[AskUserFunc] = None) -> ActionResult:
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
    """
    t = cmd.get("type")

    risk = classify_action(cmd)
    if risk == ActionRisk.BLOCKED:
        return ActionResult(False, f"{t}", "Action ถูกบล็อกโดยระบบรักษาความปลอดภัย (Blocklist)")
    if risk == ActionRisk.NEEDS_CONFIRMATION:
        approved = await _confirm_action(cmd, ask_user_func)
        if not approved:
            return ActionResult(False, f"{t}", "ผู้ใช้ปฏิเสธการทำ Action นี้ (Human-in-the-loop)")

    try:
        # click/fill/select/check ผ่าน retry wrapper (W5) เพราะพังบ่อยจาก DOM ยังไม่นิ่ง
        # ไม่ใช่ index ผิดเสมอไป — scroll/goto/go_back/switch_tab/wait ไม่ retry เพราะ
        # failure mode ต่างกัน (เช่น goto ผิด URL ก็จะผิดซ้ำทุกครั้ง ไม่ใช่เรื่อง timing)
        if t == "click":       return await _dispatch_with_retry(click, page, cmd["index"])
        if t == "fill":        return await _dispatch_with_retry(fill, page, cmd["index"], cmd["text"])
        if t == "select":      return await _dispatch_with_retry(select_option, page, cmd["index"], cmd["label"])
        if t == "check":       return await _dispatch_with_retry(check, page, cmd["index"])
        if t == "scroll":      return await scroll(page, cmd.get("direction", "down"))
        if t == "goto":        return await goto(page, cmd["url"])
        if t == "go_back":     return await go_back(page)
        if t == "switch_tab":  return await switch_tab(page, cmd["tab_index"])
        if t == "wait":        return await wait_stable(page)
        if t in DEFAULT_NEEDS_CONFIRMATION:
            # submit/delete/purchase/pay ไม่ใช่ action จริงแยกต่างหาก — เป็นแค่ risk
            # category ของ classify_action() (เช็คผ่านไปแล้วด้านบนตอนมาถึงตรงนี้) ที่จริง
            # แล้วคือคลิก element ตัวเดิม แค่ต้องขอยืนยันจาก human ก่อนเพราะเสี่ยงกว่า
            # click ธรรมดา — คืน label เดิม (เช่น "submit(2)") ไม่ใช่ "click(2)" กันสับสน
            result = await _dispatch_with_retry(click, page, cmd["index"])
            return ActionResult(result.success, f"{t}({cmd['index']})", result.message)
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
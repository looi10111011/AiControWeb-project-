"""core/user_browser.py — เชื่อม agent เข้ากับ Chrome จริงที่ user เปิดใช้งานอยู่แล้ว
(มี cookie/login ค้างอยู่ เช่น mail) แทนที่จะ launch Chromium ว่างๆ เองเหมือน
_launch_chromium()/BrowserPool เดิม

หลักการ: user ต้องเปิด Chrome เองล่วงหน้าด้วย flag `--remote-debugging-port` (ไม่ใช่
launch_persistent_context ที่ต้องปิด Chrome จริงก่อนรัน) แล้ว agent ต่อเข้าไปผ่าน
Chrome DevTools Protocol (CDP) ด้วย playwright.chromium.connect_over_cdp() — ได้
BrowserContext เดิมที่มี cookie จริงอยู่แล้ว (browser.contexts[0]) ไม่ใช่ context ว่าง
เปล่าแบบที่ browser.new_context() จะสร้างให้

ทุกฟังก์ชันในไฟล์นี้ "ห้าม" ปิด/ทำลาย browser หรือ context ที่ user ใช้งานอยู่จริงเด็ดขาด
(เป็นหน้าที่ของผู้เรียก orchestrator.py จะปิดแค่ page ที่ตัวเองเปิดเอง ถ้าเปิดจริง) —
ดู resolve_target_page() ด้านล่างที่คืนค่า opened_new_tab บอกผู้เรียกตรงๆ
"""

import asyncio
from typing import Awaitable, Callable, Optional

from playwright.async_api import Browser, BrowserContext, Page, Playwright

from backend.app.permission.rules import extract_domain

# ใช้ signature เดียวกับ actions.py::AskUserFunc (cmd dict -> bool) — ไม่ import ตรงๆ
# จาก actions.py กัน circular import (actions.py ไม่ได้ต้องรู้จักไฟล์นี้เลย)
AskUserFunc = Callable[[dict], Awaitable[bool]]

VALID_TAB_REUSE_POLICIES = {"ask", "always_new_tab", "always_reuse"}


class UserBrowserConnectError(RuntimeError):
    """เชื่อมต่อ CDP ไปยัง Chrome จริงของ user ไม่สำเร็จ — ข้อความ error อธิบายสาเหตุที่
    พบบ่อยที่สุดตรงๆ (Chrome ยังไม่ได้เปิดด้วย --remote-debugging-port) แทนที่จะปล่อย
    exception ดิบของ Playwright (ConnectionError ทั่วไป) ที่ไม่บอกวิธีแก้"""


async def connect_user_browser(playwright: Playwright, cdp_url: str) -> Browser:
    """ต่อเข้า Chrome จริงที่ user เปิดไว้แล้วผ่าน CDP — ไม่ launch process ใหม่เอง
    (ต่างจาก _launch_chromium() ใน orchestrator.py) ถ้าต่อไม่ได้ (Chrome ไม่ได้เปิด
    debug port ไว้จริง/พอร์ตผิด) โยน UserBrowserConnectError ที่บอกวิธีแก้ตรงๆ"""
    try:
        return await playwright.chromium.connect_over_cdp(cdp_url)
    except Exception as e:
        raise UserBrowserConnectError(
            f"เชื่อมต่อ Chrome จริงของ user ที่ {cdp_url} ไม่สำเร็จ ({e}) — เช็คว่าปิด "
            "Chrome ทุกหน้าต่าง/process ให้หมดก่อนแล้วเปิดใหม่ด้วย flag "
            "--remote-debugging-port=9222 (ดู docstring หัวไฟล์ run.py คำสั่ง "
            "real-browser) แล้วลองใหม่"
        ) from e


async def _open_new_tab_in_same_window(context: BrowserContext) -> Page:
    """เปิด tab ใหม่แบบรับประกันว่าโผล่ใน window เดียวกับที่ user กำลังใช้งานอยู่จริง —
    "ห้ามใช้ context.new_page() ตรงๆ" เพราะภายในเรียก CDP Target.createTarget() ซึ่ง
    Chrome ไม่การันตีว่าจะแนบ tab ใหม่เข้า window ที่ user กำลังดูอยู่เสมอ (โดยเฉพาะถ้า
    Chrome instance นั้นมีมากกว่า 1 window เปิดอยู่พร้อมกัน) ทำให้ user เห็นเป็น "window
    ใหม่ผุดขึ้นมา" แทนที่จะเป็น tab ใหม่ข้างๆ tab เดิมที่กำลังใช้อยู่ (นี่คือปัญหาที่ user
    รายงานมาจริง) — แก้ด้วยการเรียก window.open() ผ่าน page ที่เปิดอยู่แล้วจริงแทน (เช่น
    tab ของ Test Console เอง หรือ tab อื่นที่ user เปิดค้างไว้) ซึ่งเป็นพฤติกรรมมาตรฐานของ
    ทุก browser: window.open() ที่ถูกเรียกจาก page ของ window ไหน จะเปิด tab ใหม่ใน
    window นั้นเสมอ ไม่มีทางหลุดไป window อื่น"""
    if not context.pages:
        # context ว่างเปล่าจริงๆ ไม่มี page ให้ evaluate เลย (ไม่ควรเกิดในทางปฏิบัติ —
        # Chrome ที่เปิดปกติมี "New Tab" อย่างน้อย 1 อันเสมอ) — ไม่มีทางเลือกอื่นแล้ว
        # fallback เป็น context.new_page() ตรงๆ
        return await context.new_page()

    opener = context.pages[0]
    async with context.expect_page() as new_page_info:
        await opener.evaluate("() => { window.open('about:blank', '_blank'); }")
    return await new_page_info.value


async def _find_matching_tab(context: BrowserContext, target_domain: str) -> Optional[Page]:
    """หา tab ที่เปิดอยู่แล้วใน context ที่ domain ตรงกับ target_domain (ใช้จับคู่แบบ
    domain ล้วนๆ ไม่ใช่ URL เป๊ะๆ — tab ที่อยู่ path อื่นของ domain เดียวกันก็นับว่า
    ตรง) คืน None ถ้าไม่เจอเลย"""
    for p in context.pages:
        if extract_domain(p.url) == target_domain:
            return p
    return None


async def resolve_target_page(
    context: BrowserContext,
    target_url: str,
    ask_user_func: Optional[AskUserFunc],
    tab_reuse_policy: str = "ask",
) -> tuple[Page, bool]:
    """หา/เปิด page ที่จะใช้ทำ task บน context จริงของ user (ห้าม context.new_context()
    เด็ดขาด — context ที่ส่งเข้ามาต้องเป็นตัวเดียวกับที่มี cookie จริงอยู่แล้วเสมอ) คืน
    (page, opened_new_tab) — opened_new_tab=True เฉพาะตอนฟังก์ชันนี้เป็นคนเปิด tab ใหม่
    เอง (ผู้เรียก orchestrator.py ใช้ค่านี้ตัดสินว่าต้อง page.close() ตอนจบ task ไหม —
    tab ที่ user เปิดค้างไว้เองห้าม agent ปิดทิ้งเด็ดขาด)

    tab_reuse_policy:
      - ไม่เจอ tab ที่ domain ตรงกับ target เลย -> เปิด tab ใหม่เสมอ ไม่ถาม (ไม่ว่า
        policy จะเป็นอะไร)
      - เจอ tab ที่ตรง + "always_new_tab" -> เปิด tab ใหม่แทน ไม่ถาม (ไม่แตะ tab เดิม)
      - เจอ tab ที่ตรง + "always_reuse" -> ใช้ tab เดิมเลย ไม่ถาม
      - เจอ tab ที่ตรง + "ask" (default) -> ถาม user ก่อน (ผ่าน ask_user_func — ไม่มีก็
        fallback เป็น input() ทาง terminal เหมือน pattern เดิมของ permission layer)
        อนุมัติ -> ใช้ tab เดิม, ปฏิเสธ/timeout -> เปิด tab ใหม่แทน (ไม่ throw ไม่ค้าง)

    ไม่ goto(target_url) ในนี้เอง — แค่เลือก/เปิด page แล้วคืนกลับให้ผู้เรียก
    (orchestrator.py) ไป goto ต่อผ่านจุด goto() เดียวกับ 2 path เดิม (owns_browser/
    pool) เพื่อให้ memory record/on_event ของ step แรกเหมือนกันทุก path"""
    if tab_reuse_policy not in VALID_TAB_REUSE_POLICIES:
        raise ValueError(
            f"tab_reuse_policy ไม่รู้จัก: {tab_reuse_policy!r} (ต้องเป็นหนึ่งใน "
            f"{sorted(VALID_TAB_REUSE_POLICIES)})"
        )

    target_domain = extract_domain(target_url)
    matched = await _find_matching_tab(context, target_domain)

    if matched is None or tab_reuse_policy == "always_new_tab":
        page = await _open_new_tab_in_same_window(context)
        opened_new_tab = True
    elif tab_reuse_policy == "always_reuse":
        page = matched
        opened_new_tab = False
    else:  # "ask"
        approved = await _confirm_tab_reuse(matched, target_url, ask_user_func)
        if approved:
            page = matched
            opened_new_tab = False
        else:
            page = await _open_new_tab_in_same_window(context)
            opened_new_tab = True

    await page.bring_to_front()
    return page, opened_new_tab


async def _confirm_tab_reuse(
    matched_tab: Page, target_url: str, ask_user_func: Optional[AskUserFunc],
) -> bool:
    """ถาม user ก่อนให้ agent เข้าไปใช้ tab ที่ user เปิดค้างไว้เอง — ใช้
    AskUserFunc pattern เดียวกับ permission layer (actions.py::_confirm_action) ไม่มี
    ask_user_func ก็ fallback เป็น input() ทาง terminal เหมือนกัน"""
    cmd = {
        "type": "confirm_tab_reuse",
        "matched_tab_url": matched_tab.url,
        "target_url": target_url,
    }
    if ask_user_func is not None:
        return bool(await ask_user_func(cmd))

    print(
        f"\n[USER-BROWSER] เจอ tab ที่เปิดอยู่แล้วตรงกับโดเมนเป้าหมาย: {matched_tab.url}",
        flush=True,
    )
    choice = await asyncio.to_thread(
        input, "ให้ agent ใช้ tab นี้ต่อเลยไหม (ไม่งั้นจะเปิด tab ใหม่แทน)? (y/n): "
    )
    return choice.strip().lower() in ("y", "yes")

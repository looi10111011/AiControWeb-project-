"""core/user_browser.py — ต่อ agent เข้า Chrome จริงของ user (มี cookie/login อยู่แล้ว) ผ่าน CDP
(connect_over_cdp) แทนการ launch Chromium ว่างๆ — user ต้องเปิด Chrome ด้วย --remote-debugging-port เอง

ห้ามปิด/ทำลาย browser หรือ context ของ user ในไฟล์นี้ — resolve_target_page() คืน opened_new_tab
ให้ผู้เรียกตัดสินว่าจะปิดเฉพาะ page ที่ agent เปิดเอง
"""

import asyncio
from urllib.parse import urlsplit
from typing import Awaitable, Callable, Optional

from playwright.async_api import Browser, BrowserContext, Page, Playwright

from backend.app.permission.rules import extract_domain

# signature เดียวกับ actions.py::AskUserFunc — ไม่ import ตรงกัน circular import
AskUserFunc = Callable[[dict], Awaitable[bool]]

VALID_TAB_REUSE_POLICIES = {"ask", "always_new_tab", "always_reuse"}


class UserBrowserConnectError(RuntimeError):
    """ต่อ CDP ไม่สำเร็จ — ข้อความบอกวิธีแก้ (เปิด Chrome ด้วย --remote-debugging-port) แทน error ดิบ"""


async def connect_user_browser(playwright: Playwright, cdp_url: str) -> Browser:
    """ต่อ Chrome ที่เปิดอยู่แล้วผ่าน CDP (ไม่ launch เอง) — ล้มเหลวโยน UserBrowserConnectError"""
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
    """เปิด tab ใหม่ใน window เดียวกับที่ user ใช้อยู่ — ห้าม context.new_page() ตรงๆ เพราะ CDP
    Target.createTarget() อาจโผล่เป็น window ใหม่เมื่อมีหลาย window (bug ที่ user รายงานจริง)
    แก้ด้วย window.open() จาก page ที่เปิดอยู่ ซึ่งเปิด tab ใน window ของ opener เสมอ"""
    if not context.pages:
        # ไม่มี page ให้ evaluate (ไม่ควรเกิดจริง) — fallback new_page()
        return await context.new_page()

    opener = context.pages[0]
    async with context.expect_page() as new_page_info:
        await opener.evaluate("() => { window.open('about:blank', '_blank'); }")
    return await new_page_info.value


async def _find_matching_tab(context: BrowserContext, target_domain: str) -> Optional[Page]:
    """tab แรกที่ domain ตรง (ไม่สน path) หรือ None"""
    for p in context.pages:
        if extract_domain(p.url) == target_domain:
            return p
    return None


async def resolve_target_page(
    context: BrowserContext,
    target_url: str,
    ask_user_func: Optional[AskUserFunc],
    tab_reuse_policy: str = "ask",
    target_tab_id: Optional[str] = None,
) -> tuple[Page, bool]:
    """เลือก/เปิด page บน context จริงของ user คืน (page, opened_new_tab) — opened_new_tab=True
    เฉพาะเมื่อฟังก์ชันนี้เปิด tab เอง (tab ของ user ห้าม agent ปิด)

    ไม่เจอ tab domain ตรง หรือ "always_new_tab" -> tab ใหม่; "always_reuse" -> ใช้ tab เดิม;
    "ask" -> ถาม user (ask_user_func หรือ input()) ปฏิเสธ/timeout -> tab ใหม่ (ไม่ throw)
    target_tab_id: หา tab ที่มี Agent Bar marker ตรงกันพอดี 1 tab ไม่งั้น UserBrowserConnectError
    ไม่ goto() เอง — ผู้เรียกทำผ่านจุด goto เดียวกับทุก path"""
    if tab_reuse_policy not in VALID_TAB_REUSE_POLICIES:
        raise ValueError(
            f"tab_reuse_policy ไม่รู้จัก: {tab_reuse_policy!r} (ต้องเป็นหนึ่งใน "
            f"{sorted(VALID_TAB_REUSE_POLICIES)})"
        )

    if target_tab_id:
        origin = urlsplit(target_url)
        matches = []
        for candidate in context.pages:
            candidate_url = urlsplit(candidate.url)
            if (candidate_url.scheme, candidate_url.netloc) != (origin.scheme, origin.netloc):
                continue
            try:
                marker = await candidate.evaluate(
                    "() => document.getElementById('hermes-ai-bar-host')?.dataset.tabId"
                )
                if marker == target_tab_id:
                    matches.append(candidate)
            except Exception:
                continue
        if len(matches) != 1:
            raise UserBrowserConnectError(
                "ไม่พบแท็บ Agent Bar ที่ส่งคำสั่งใน browser ที่เชื่อมต่อ CDP "
                "กรุณาเปิดหน้า benchmark ใน browser ที่เชื่อมต่อแล้วลองใหม่"
            )
        await matches[0].bring_to_front()
        return matches[0], False

    target_domain = extract_domain(target_url)
    matched = await _find_matching_tab(context, target_domain)

    if matched is None or tab_reuse_policy == "always_new_tab":
        reuse = False
    elif tab_reuse_policy == "always_reuse":
        reuse = True
    else:  # "ask"
        reuse = await _confirm_tab_reuse(matched, target_url, ask_user_func)

    page = matched if reuse else await _open_new_tab_in_same_window(context)
    await page.bring_to_front()
    return page, not reuse


async def _confirm_tab_reuse(
    matched_tab: Page, target_url: str, ask_user_func: Optional[AskUserFunc],
) -> bool:
    """ถาม user ก่อนใช้ tab ที่เปิดค้างไว้ — AskUserFunc pattern เดียวกับ permission layer, ไม่มีก็ input()"""
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

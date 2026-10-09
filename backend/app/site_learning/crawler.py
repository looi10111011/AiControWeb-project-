"""site_learning/crawler.py — W14: BFS deterministic crawler ที่สร้าง SiteManual. LLM ไม่ตัดสินใจ navigate เลย
(ลด token cost) — เรียกแค่ครั้งเดียวต่อหน้าเพื่อตั้งชื่อ/คำอธิบาย; ที่เหลือเป็น DOM data ล้วน.

W15: ข้อยกเว้น "ห้ามกด Submit" — login bootstrap (auto_login.attempt_login) เพราะหน้า login ไม่มี nav link ให้เดินต่อ;
     เฉพาะเมื่อ caller ส่ง credential มาเองและเจอ password field จริง; ไม่บันทึก/log credential ที่ไหนเลย
W16: ไล่กดปุ่มปลอดภัย (is_crawl_safe) แบบ DFS ไม่จำกัดความลึก แล้ว go_back()/goto() กลับก่อนปุ่มถัดไป; จบได้ด้วย
     visited-set + site_learning_max_pages + site_learning_max_buttons_per_page
W24: contract ของ "เรียนรู้เว็บ" บังคับในโค้ด (ไม่ใช่ prompt): queue BFS; _wait_for_dom_stable() สำหรับ SPA routing
     ที่ไม่ยิง network; เมนูที่ไม่ใช่ <a> (is_nav_menu_item) default-allow; _goto/_click_with_retry + SiteManual.errors
     + event page_error/button_click_failed แทนการกลืนเงียบ; ตรวจ session หลัง login (login_result); retry/scroll
     ย้ายเป็น settings.site_learning_*; _reveal_dynamic_content() สำหรับ lazy/infinite scroll; newly-revealed pass
     1 ชั้น (_MAX_REVEAL_DEPTH) สำหรับ dropdown/accordion ที่ซ่อนด้วย display:none ตอน extract แรก
W25: <a href> สแกนทั้งเอกสาร (เดิมแค่ NAV_CONTAINERS) — ยัง goto()-BFS ไม่ใช่ click+go_back (ผลเหมือนกันแต่เร็วกว่า)
W28/W29: ลูปกดปุ่มเดิมข้าม URL ไม่รู้จบ (YouTube Shorts "Next video") — explored_button_signatures นับต่อ crawl
     จำกัดด้วย settings.site_learning_max_repeat_button_clicks (ดู _button_signature)
W33: feed ลิงก์ไปคลิปคนละ URL แต่ UI เหมือนกัน — _page_template()/known_page_templates ข้ามหน้าโครงสร้างซ้ำใน
     _record_page() (ไม่ต้องมี logic "กด back" แยก — caller กลับเองอยู่แล้ว)
W34: บั๊กจริง — DFS-click ที่หลุดนอกโดเมนเคยตกไป branch "modal" แล้ว merge โครงสร้างเว็บอื่นเข้า manual และไม่กลับมา;
     แยก branch off-domain (event off_domain_navigation + goto กลับ) + guard redirect นอกโดเมนใน BFS (ข้าม URL)
W35: ชื่อหน้าซ้ำติดกัน > 2 ครั้ง -> หยุดไล่ปุ่ม/ต่อคิวจากหน้านั้น (template ไม่ตรงเป๊ะแต่ LLM ตั้งชื่อเหมือน)
W36: tier filter — ตัด "decorative" ก่อน _is_explorable() + เพดาน core ต่อหน้า (top-K ตาม button_core_priority);
     ไม่ใช่ safety gate. "nav" ยังผ่าน _explore_buttons() เพราะเมนู SPA ไม่มี href (W24) — user ยืนยันแล้ว
W37/W38: ไม่เข้าหน้าวีดีโอ/แฮชแท็กเลย (_excluded_content_reason) ใน 3 จุด: nav_links loop, classify_button_tier,
     post-click URL check (event f"{reason}_content_skipped")
W39: ปุ่มใน <iframe> — extractor แปะ frame_index; _resolve_click_target() คืน Frame ที่ถูก (index ไม่ใช่ url เพราะ
     srcdoc iframe ได้ "about:srcdoc" เหมือนกันหมด). ขอบเขต: auto_login ยังไม่รองรับฟอร์มใน iframe และ navigation
     ภายใน iframe ถูกจัดการเหมือน modal (re-extract ทุก frame แล้ว merge)
"""

import json
import re
import time
import urllib.parse
from typing import Awaitable, Callable, Optional, Union

from playwright.async_api import Browser, Frame, Page

from backend.app.config import settings
from backend.app.core import llm
from backend.app.core.dom_locator import compute_locator_descriptor
from backend.app.core.orchestrator import Orchestrator
from backend.app.permission.rules import extract_domain, install_ssrf_guard
from backend.app.site_learning.auto_login import attempt_login, find_login_fields, verify_login_success
from backend.app.site_learning.extractor import extract_page
from backend.app.site_learning.safety import (
    button_core_priority,
    is_crawl_safe,
    is_hashtag_label,
    is_hashtag_url,
    is_safe_nav_link,
    is_video_content_label,
    is_video_content_url,
)
from backend.app.site_learning.schema import PageInfo, SiteManual

OnProgressFunc = Callable[[dict], Awaitable[None]]
# W23: เรียกเมื่อเจอหน้า login แต่ไม่มี credential — รับ domain ของเว็บนี้ คืน {"username","password"} หรือ None (ข้าม/หมดเวลา)
OnCredentialsNeededFunc = Callable[[str], Awaitable[Optional[dict]]]

_DESCRIBE_PROMPT_TEMPLATE = (
    "นี่คือโครงสร้างของหน้าเว็บหน้าหนึ่ง สกัดจาก DOM จริงล้วนๆ (ไม่ใช่จินตนาการ):\n"
    "URL: {url}\n"
    "Breadcrumb: {breadcrumb}\n"
    "ปุ่มที่เจอ: {buttons}\n"
    "ช่องกรอกในฟอร์ม: {forms}\n"
    "ตาราง: {tables}\n"
    "UI pattern ที่ซ้ำกันหลาย instance: {ui_patterns}\n\n"
    "ตอบเป็น JSON เท่านั้น ไม่มีข้อความอื่นเลย รูปแบบ: "
    '{{"name": "ชื่อหน้าสั้นๆ ไม่เกิน 4 คำ", "description": "คำอธิบายหน้าที่ของหน้านี้ 1 ประโยค"}}'
)


async def describe_page(client, model: str, provider: str, page_info: PageInfo) -> tuple[str, str]:
    """คืน (name, description) จาก LLM ครั้งเดียวต่อหน้า; parse ไม่ได้ -> ชื่อจาก URL path, "". ไม่ throw"""
    buttons = ", ".join(
        (b.text or b.aria_label or b.icon_hint) for b in page_info.buttons[:15]
        if (b.text or b.aria_label or b.icon_hint)
    ) or "(none)"
    forms = ", ".join(
        (f.label or f.field_name) for f in page_info.forms[:10] if (f.label or f.field_name)
    ) or "(none)"
    tables = ", ".join(f"{len(t.columns)} columns" for t in page_info.tables[:5]) or "(none)"
    ui_patterns = ", ".join(
        f"{p.name} ({p.ui_type} x{p.item_count})" for p in page_info.ui_patterns[:10] if p.name
    ) or "(none)"
    prompt = _DESCRIBE_PROMPT_TEMPLATE.format(
        url=page_info.url,
        breadcrumb=" > ".join(page_info.breadcrumb) or "(none)",
        buttons=buttons, forms=forms, tables=tables, ui_patterns=ui_patterns,
    )
    try:
        text = await llm.generate_text(client, model, prompt, provider)
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            name = str(data.get("name", "")).strip()
            description = str(data.get("description", "")).strip()
            if name:
                return name, description
    except Exception:
        pass
    fallback = urllib.parse.urlparse(page_info.url).path.strip("/").split("/")[-1] or "Home"
    return fallback.replace("-", " ").replace("_", " ").title(), ""


_SITE_SUMMARY_PROMPT_TEMPLATE = (
    "นี่คือรายชื่อหน้าเว็บทั้งหมดที่สำรวจเจอในเว็บไซต์ {website} พร้อมคำอธิบายสั้นๆ ของแต่ละหน้า "
    "(สกัดจากโครงสร้าง DOM จริงของแต่ละหน้า ไม่ใช่จินตนาการ):\n{pages_summary}\n\n"
    "เขียนสรุปสั้นๆ ไม่เกิน 4 ประโยค อธิบายให้ผู้ใช้ทั่วไป (ที่ยังไม่เคยเห็นเว็บนี้มาก่อน) "
    "เข้าใจง่ายว่าเว็บไซต์นี้ทำอะไรได้บ้าง (สรุปภาพรวมความสามารถหลัก ไม่ใช่แค่ไล่ชื่อหน้าทีละ"
    "หน้า) ตอบเป็นภาษาไทยล้วนๆ เป็นข้อความธรรมดา ไม่ต้องมี markdown/bullet/หัวข้อ"
)


async def describe_site(client, model: str, provider: str, website: str, pages: list[PageInfo]) -> str:
    """W26: สรุปภาพรวมเว็บหลัง crawl จบ (LLM ครั้งเดียว ใช้แค่ name/description กัน token). LLM ล้ม -> รายชื่อหน้า;
    "" ถ้าไม่มีหน้าที่มีชื่อ. ไม่ throw"""
    named_pages = [p for p in pages if p.name]
    fallback = (
        f"เว็บไซต์นี้มีทั้งหมด {len(pages)} หน้า: " + ", ".join(p.name for p in named_pages[:15])
        if named_pages else ""
    )
    if not named_pages:
        return fallback
    pages_summary = "\n".join(f"- {p.name}: {p.description}" for p in named_pages if p.description)
    if not pages_summary:
        pages_summary = "\n".join(f"- {p.name}" for p in named_pages)
    prompt = _SITE_SUMMARY_PROMPT_TEMPLATE.format(website=website, pages_summary=pages_summary)
    try:
        text = (await llm.generate_text(client, model, prompt, provider)).strip()
        if text:
            return text
    except Exception:
        pass
    return fallback


def _merge_page_info(base: PageInfo, extra: PageInfo) -> None:
    """merge โครงสร้างที่โผล่จาก modal/panel/tab (URL เดิม) เข้า base — dedup ด้วย selector (buttons/forms)
    และค่าตรงๆ (tables/modals/tabs)"""
    known_button_selectors = {b.selector for b in base.buttons if b.selector}
    base.buttons.extend(b for b in extra.buttons if b.selector and b.selector not in known_button_selectors)
    known_form_selectors = {f.selector for f in base.forms if f.selector}
    base.forms.extend(f for f in extra.forms if f.selector and f.selector not in known_form_selectors)
    known_table_sigs = {tuple(t.columns) for t in base.tables}
    base.tables.extend(t for t in extra.tables if tuple(t.columns) not in known_table_sigs)
    for modal in extra.modals:
        if modal not in base.modals:
            base.modals.append(modal)
    for tab in extra.tabs:
        if tab not in base.tabs:
            base.tabs.append(tab)
    base.search_box = base.search_box or extra.search_box


def _button_label(button) -> str:
    """text > aria_label > title > icon_hint > data_testid (humanized). data_testid: เว็บ QA (saucedemo) มีปุ่ม
    icon-only ที่ไม่มี label อื่น -> is_crawl_safe() ปฏิเสธตลอด ทั้งที่ "shopping-cart-link" บอกชัด. ไม่กระทบ selector"""
    return (
        button.text or button.aria_label or button.title or button.icon_hint
        or button.data_testid.replace("-", " ").replace("_", " ").strip()
    )


def _button_signature(button) -> str:
    """W28/W29: identity ของปุ่มข้ามหน้า = text > data_testid > icon_hint + role/has_icon/is_nav_menu_item.
    W29 ตัด aria_label/title ออก — YouTube ใส่ "Next video: <ชื่อคลิป>" ทำให้ signature ไม่เคยซ้ำ เพดาน W28 ไร้ผล.
    trade-off: ปุ่ม text เหมือนกันคนละหน้าที่ (เช่น "View" คนละตาราง) ถูกนับเป็นปุ่มเดียว — ยอมเพื่อกันลูป"""
    identity = (
        (button.text or "").strip().lower()
        or (button.data_testid or "").strip().lower()
        or (button.icon_hint or "").strip().lower()
    )
    return "|".join([
        identity,
        (getattr(button, "role", "") or "").strip().lower(),
        str(bool(button.has_icon)),
        str(bool(getattr(button, "is_nav_menu_item", False))),
    ])


def _page_template(page_info: PageInfo) -> frozenset[str]:
    """W33: ลายนิ้วมือโครงสร้างหน้า = set ของ _button_signature + ui_pattern + มี/ไม่มี form/table (ไม่สนลำดับ/จำนวน).
    เทียบ exact เท่านั้น — ยอม false-negative ดีกว่า false-positive ที่ข้ามหน้าที่ควรบันทึก"""
    button_sigs = frozenset(_button_signature(b) for b in page_info.buttons if b.selector)
    pattern_sigs = frozenset(f"pattern:{p.ui_type}:{p.selector}" for p in page_info.ui_patterns)
    coarse = frozenset({
        f"forms:{bool(page_info.forms)}",
        f"tables:{bool(page_info.tables)}",
    })
    return button_sigs | pattern_sigs | coarse


def _excluded_content_reason(url: str, label: str = "") -> Optional[str]:
    """W37/W38: "video" | "hashtag" | None — จุดเดียวสำหรับ nav_links loop และ post-click check
    (classify_button_tier เรียก is_*_label ตรงๆ เพราะต้องการแค่ bool)"""
    if is_video_content_url(url) or is_video_content_label(label):
        return "video"
    if is_hashtag_url(url) or is_hashtag_label(label):
        return "hashtag"
    return None


def _normalize_url(url: str) -> str:
    """ตัด fragment และ trailing slash กันนับหน้าเดิมซ้ำ"""
    parsed = urllib.parse.urlparse(url)
    path = parsed.path.rstrip("/") or "/"
    return urllib.parse.urlunparse((parsed.scheme, parsed.netloc, path, "", parsed.query, ""))


# W24: ไล่กดปุ่มที่โผล่จาก modal/dropdown/accordion แค่ 1 ชั้น กันเปิด/ปิดซ้อนไม่รู้จบ
_MAX_REVEAL_DEPTH = 1


async def _wait_for_dom_stable(
    page: Page, checks: int = 3, interval_ms: Optional[int] = None, max_iterations: int = 12,
) -> None:
    """W24: SPA client-side routing อาจไม่ยิง network -> networkidle ผ่านเร็วเกิน; poll innerHTML.length จนเท่ากัน
    `checks` ครั้งติดหรือครบ max_iterations. best-effort ไม่ throw"""
    interval = settings.site_learning_retry_backoff_ms // 3 if interval_ms is None else interval_ms
    interval = max(interval, 50)
    try:
        last_length = -1
        stable_count = 0
        for _ in range(max_iterations):
            length = await page.evaluate("document.body.innerHTML.length")
            if length == last_length:
                stable_count += 1
                if stable_count >= checks:
                    return
            else:
                stable_count = 0
            last_length = length
            await page.wait_for_timeout(interval)
    except Exception:
        pass


async def _settle_url(page: Page, max_iterations: int = 8, interval_ms: int = 200) -> None:
    """W34: cascading redirect (A -> B -> นอกโดเมน) ทำให้ _wait_for_dom_stable() exit เร็วตอน context ถูกทำลาย แล้ว
    อ่าน page.url กลางทาง -> ตัดสิน same/off-domain ผิด. poll page.url จนนิ่ง 2 รอบหรือครบ max_iterations"""
    last_url = page.url
    stable = 0
    for _ in range(max_iterations):
        await page.wait_for_timeout(interval_ms)
        current = page.url
        if current == last_url:
            stable += 1
            if stable >= 2:
                return
        else:
            stable = 0
        last_url = current


async def _reveal_dynamic_content(page: Page) -> None:
    """W24: scroll จน scrollHeight นิ่งหรือครบ site_learning_max_scroll_attempts (lazy/infinite scroll) แล้วกลับบนสุด.
    best-effort ไม่ throw"""
    try:
        previous_height = await page.evaluate("document.body.scrollHeight")
        for _ in range(settings.site_learning_max_scroll_attempts):
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await page.wait_for_timeout(settings.site_learning_scroll_wait_ms)
            try:
                await page.wait_for_load_state("networkidle", timeout=2000)
            except Exception:
                pass
            new_height = await page.evaluate("document.body.scrollHeight")
            if new_height <= previous_height:
                break
            previous_height = new_height
    except Exception:
        pass
    finally:
        try:
            await page.evaluate("window.scrollTo(0, 0)")
        except Exception:
            pass


async def _goto_with_retry(page: Page, url: str, retries: int) -> Optional[str]:
    """W24: goto+networkidle สูงสุด retries+1 ครั้ง — คืน None ถ้าสำเร็จ หรือ error ตัวสุดท้าย. ไม่ throw"""
    last_error = ""
    for attempt in range(retries + 1):
        try:
            await page.goto(url, timeout=15000)
            await page.wait_for_load_state("networkidle", timeout=8000)
            return None
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
            if attempt < retries:
                await page.wait_for_timeout(settings.site_learning_retry_backoff_ms)
    return last_error


# W39: Page และ Frame มี .click()/.wait_for_timeout() หน้าตาเดียวกัน ใช้แทนกันได้ (frame_index 0 = page)
ClickTarget = Union[Page, Frame]


def _resolve_click_target(page: Page, frame_index: int) -> ClickTarget:
    """W39: Frame ที่ page.frames[frame_index] (page.click หา element ข้าม frame ไม่เจอ); 0 หรือ index เกินขอบเขต
    (frame เปลี่ยนหลัง extract) -> page ให้ click ล้มตามปกติ"""
    if not frame_index:
        return page
    frames = page.frames
    if 0 <= frame_index < len(frames):
        return frames[frame_index]
    return page


async def _click_with_retry(target: ClickTarget, selector: str, retries: int) -> Optional[str]:
    """W24: click สูงสุด retries+1 ครั้ง — None ถ้าสำเร็จ หรือ error ตัวสุดท้าย (W39: target เป็น Page/Frame).
    W34: ปุ่มที่ navigate ทันทีอาจ "ดูเหมือน fail" (context ถูกทำลาย แล้ว retry หา selector บนหน้าใหม่ไม่เจอ);
    no_wait_after=True ลองแล้วแย่กว่า — caller ต้องเช็ค page.url ก่อนถือว่า fail จริง"""
    last_error = ""
    for attempt in range(retries + 1):
        try:
            await target.click(selector, timeout=5000)
            return None
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
            if attempt < retries:
                await target.wait_for_timeout(settings.site_learning_retry_backoff_ms)
    return last_error


async def crawl_site(
    browser: Browser,
    start_url: str,
    max_pages: Optional[int] = None,
    provider: Optional[str] = None,
    on_progress: Optional[OnProgressFunc] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    on_credentials_needed: Optional[OnCredentialsNeededFunc] = None,
) -> SiteManual:
    """BFS crawl จาก start_url (same-origin, กรองด้วย is_safe_nav_link) — คืน SiteManual ที่ยังไม่ save
    (caller เรียก storage.save_manual()). ใช้ BrowserContext ของตัวเองและปิดเสมอ.

    username/password (W15): login bootstrap ครั้งเดียวบนหน้าแรกที่ถึง.
    on_credentials_needed (W23): ไม่มี credential แต่เจอหน้า login -> await callback (LearnManager.request_credentials);
        dict = login ต่อ, None = บันทึกหน้าแล้วสำรวจต่อโดยไม่ login. ถามครั้งเดียวต่อ crawl (login_attempted เดียวกัน).
        ไม่ส่งอะไรเลย = ไม่แตะฟอร์มใดๆ.
    W16: ทุกหน้าที่บันทึกถูกไล่กดปุ่มปลอดภัยแบบ DFS (จำกัดด้วย max_buttons_per_page และ max_pages)"""
    domain = extract_domain(start_url)
    resolved_provider = provider or settings.llm_provider
    client, model, _, _, _ = Orchestrator._llm_backend(resolved_provider)
    effective_max_pages = max_pages or settings.site_learning_max_pages

    context = await browser.new_context()
    await install_ssrf_guard(context)
    page = await context.new_page()
    pages: list[PageInfo] = []
    # W24: error จริง (goto/click ล้มครบ retry, login ไม่ผ่าน) ติดไปกับ SiteManual แทนการกลืนเงียบ
    manual_errors: list[dict] = []
    try:
        visited: set[str] = set()
        # W66[A]: queue พก (url, parent_url, arrived_via_descriptor) ให้ตั้ง PageInfo.parent_url/arrived_via ตอน dequeue;
        # root = ("", {}). v1 เฉพาะ BFS — หน้าจาก DFS-click/login ยังว่าง (fastpath_executor fail-safe คืน None เอง)
        queue: list[tuple[str, str, dict]] = [(start_url, "", {})]
        queued: set[str] = {_normalize_url(start_url)}
        estimated_total = 1
        login_attempted = False
        # W28: นับการกดปุ่ม signature เดียวกันข้ามทุก URL (_button_signature)
        explored_button_signatures: dict[str, int] = {}
        # W33: template ของหน้าที่บันทึกแล้ว (_page_template) — หน้าโครงสร้างซ้ำถูกข้ามใน _record_page()
        known_page_templates: set[frozenset[str]] = set()
        # W35: ยังวนลูปแม้มี W33 — คลิป Shorts/Reels ต่างกันเล็กน้อย template ไม่ตรงเป๊ะ แต่ LLM ตั้งชื่อเหมือนเดิม;
        # ชื่อซ้ำติดกัน > 2 ครั้ง -> หยุดไล่ปุ่ม/ต่อคิวจากหน้านั้น (_record_page)
        last_recorded_page_name = ""
        consecutive_same_name_count = 0

        def _is_explorable(button) -> bool:
            """W24: nav menu item -> default-allow (is_safe_nav_link) เหมือน <a> nav; ปุ่มอื่น -> is_crawl_safe() default-deny"""
            label = _button_label(button)
            if getattr(button, "is_nav_menu_item", False) and is_safe_nav_link(label):
                return True
            return is_crawl_safe(label, cmd_type="click")

        async def _record_page(
            page_info: PageInfo, nav_links: list[dict], check_page_template: bool = True,
            explore_buttons: bool = True,
        ) -> None:
            """describe + เก็บเข้า pages + event page_done + ไล่กดปุ่ม + ต่อคิว nav link — ใช้ร่วมโดย BFS, หลัง login,
            และ DFS-click.

            W33: ข้ามหน้าที่ template ซ้ำ (ไม่ describe/ไล่ปุ่ม). false-positive จริง 2 จุดที่แก้แล้ว: (1) หน้าไม่มีปุ่ม/
            ui_pattern ได้ template ว่างตรงกันโดยบังเอิญ -> ต้องมี signal ก่อน dedup (has_template_signal);
            (2) DFS-click มี signature-cap ของตัวเอง (W28/W29) — check_page_template=False กัน template dedup ทับเพดาน
            ที่ user ตั้ง (ยังจำ template ไว้ให้ BFS)"""
            nonlocal estimated_total, last_recorded_page_name, consecutive_same_name_count
            template = _page_template(page_info)
            has_template_signal = bool(page_info.buttons) or bool(page_info.ui_patterns)
            if check_page_template and has_template_signal and template in known_page_templates:
                if on_progress:
                    await on_progress({"kind": "page_template_skipped", "url": page_info.url})
                return
            if has_template_signal:
                known_page_templates.add(template)

            page_info.name, page_info.description = await describe_page(
                client, model, resolved_provider, page_info,
            )
            page_info.menu_path = page_info.menu_path or [page_info.name]
            pages.append(page_info)

            if on_progress:
                await on_progress({
                    "kind": "page_done",
                    "name": page_info.name,
                    "url": page_info.url,
                    "done": len(pages),
                    "total": max(estimated_total, len(pages)),
                })

            # W35: หน้านี้บันทึกแล้ว แค่หยุดไล่ปุ่ม/ต่อคิวจากหน้านี้ ตัดวงจรหน้าประเภทเดิมซ้ำๆ
            if page_info.name and page_info.name == last_recorded_page_name:
                consecutive_same_name_count += 1
            else:
                consecutive_same_name_count = 1
                last_recorded_page_name = page_info.name

            if consecutive_same_name_count > 2:
                if on_progress:
                    await on_progress({
                        "kind": "repeated_page_name_skipped",
                        "name": page_info.name,
                        "url": page_info.url,
                        "consecutive_count": consecutive_same_name_count,
                    })
                return

            if explore_buttons and len(pages) < effective_max_pages:
                await _explore_buttons(page_info)

            for link in nav_links:
                href = link.get("href", "")
                if not href:
                    continue
                if not is_safe_nav_link(link.get("text", "")):
                    continue
                absolute = urllib.parse.urljoin(page.url, href)
                if extract_domain(absolute) != domain:
                    continue  # ข้าม cross-origin เด็ดขาด — นอกขอบเขตการเรียนรู้เว็บนี้
                # W37/W38: ลิงก์ไปหน้าวีดีโอ/แฮชแท็กไม่ต่อคิว (เนื้อหาไม่ใช่โครงสร้าง + แนะนำต่อไม่รู้จบ)
                if _excluded_content_reason(absolute, link.get("text", "")):
                    continue
                normalized_link = _normalize_url(absolute)
                if normalized_link in visited or normalized_link in queued:
                    continue
                # W66[A]: คำนวณ descriptor ตอนนี้ขณะ page ยังอยู่บนหน้าที่ extract nav_links มา;
                # compute_locator_descriptor() ไม่ throw (คืน {} ถ้าหาไม่เจอ)
                arrived_via_descriptor = {}
                link_selector = link.get("selector", "")
                if link_selector:
                    arrived_via_descriptor = await compute_locator_descriptor(page, link_selector)
                queue.append((absolute, page_info.url, arrived_via_descriptor))
                queued.add(normalized_link)
                estimated_total = max(estimated_total, len(pages) + len(queue))

        async def _explore_buttons(base_page_info: PageInfo, depth: int = 0) -> None:
            """ไล่กดปุ่มที่ _is_explorable() อนุญาตบนหน้านี้ทีละปุ่ม:
            - URL เปลี่ยนในโดเมน -> _record_page ถ้ายังไม่เคยเจอ (DFS ไม่จำกัดความลึก) แล้ว go_back()/goto() กลับ
            - URL เปลี่ยนนอกโดเมน (W34) -> บันทึก error แล้ว goto กลับ
            - URL ไม่เปลี่ยน -> modal/panel: re-extract + _merge_page_info, ไล่ปุ่มที่เพิ่งโผล่อีก depth < _MAX_REVEAL_DEPTH,
              กด Escape
            ไม่ throw; กลับหน้าตั้งต้นไม่ได้ -> เลิกไล่ปุ่มที่เหลือของหน้านี้. depth=1 = ปุ่มที่โผล่จาก modal"""
            before_url = _normalize_url(page.url)
            # W18: ปุ่มระดับหน้า + ปุ่มของ instance ตัวแทนของแต่ละ UI pattern
            candidate_buttons = list(base_page_info.buttons)
            for pattern in base_page_info.ui_patterns:
                candidate_buttons.extend(pattern.buttons)

            # W36: ตัด decorative ก่อนถึง _is_explorable(); nav ยังผ่านเพราะเมนู SPA ไม่มี href ให้ BFS (W24)
            tier_filtered = [b for b in candidate_buttons if b.tier != "decorative"]
            safe_buttons = [b for b in tier_filtered if b.selector and _is_explorable(b)]

            # W36: core เกิน site_learning_max_core_buttons_per_page -> top-K ตาม button_core_priority (nav ไม่มีเพดานนี้);
            # เทียบด้วย id() เพราะ dataclass __eq__ เทียบค่า ปุ่มคนละ element ที่ค่าเหมือนกันจะชนกัน
            core_buttons = [b for b in safe_buttons if b.tier == "core"]
            if len(core_buttons) > settings.site_learning_max_core_buttons_per_page:
                ranked_core = sorted(core_buttons, key=button_core_priority)
                keep_core_ids = {id(b) for b in ranked_core[:settings.site_learning_max_core_buttons_per_page]}
                safe_buttons = [b for b in safe_buttons if b.tier != "core" or id(b) in keep_core_ids]

            safe_buttons = safe_buttons[:settings.site_learning_max_buttons_per_page]

            for button in safe_buttons:
                if len(pages) >= effective_max_pages:
                    break
                # W28: signature นี้ถูกกดครบเพดานทั้ง crawl แล้ว -> ข้าม
                signature = _button_signature(button)
                click_count = explored_button_signatures.get(signature, 0)
                if click_count >= settings.site_learning_max_repeat_button_clicks:
                    continue
                explored_button_signatures[signature] = click_count + 1
                label = _button_label(button)
                if on_progress:
                    await on_progress({"kind": "button_explored", "url": before_url, "button": label})

                # W39: ปุ่มใน iframe ต้องกดผ่าน Frame ที่ถูก
                click_target = _resolve_click_target(page, getattr(button, "frame_index", 0))
                click_error = await _click_with_retry(click_target, button.selector, settings.site_learning_click_retries)
                # W34: click_error อาจมาจาก retry บนหน้าที่ navigate ไปแล้ว — URL เปลี่ยน = ถือว่ากดสำเร็จ
                if click_error is not None and _normalize_url(page.url) == before_url:
                    # W34: navigate จริงบางทีดีเลย์หลัง retry ครบ (>15s) — รออีก ~5s ก่อนสรุปว่า fail
                    for _ in range(10):
                        await page.wait_for_timeout(500)
                        if _normalize_url(page.url) != before_url:
                            break
                if click_error is not None and _normalize_url(page.url) == before_url:
                    manual_errors.append({"url": before_url, "phase": "click", "button": label, "error": click_error})
                    if on_progress:
                        await on_progress({
                            "kind": "button_click_failed", "url": before_url, "button": label, "error": click_error,
                        })
                    continue  # กดปุ่มนี้ไม่ได้แม้ retry ครบแล้ว (element หาย/ถูกบัง/detach ฯลฯ) ข้ามไปปุ่มถัดไป

                # target="_blank" เปิดแท็บใหม่ — ปิดทิ้ง กัน page สะสมใน context
                for extra_page in list(context.pages):
                    if extra_page != page:
                        try:
                            await extra_page.close()
                        except Exception:
                            pass

                try:
                    await page.wait_for_load_state("networkidle", timeout=3000)
                except Exception:
                    pass
                await _wait_for_dom_stable(page)  # W24: SPA client-side routing (ดู docstring หัวไฟล์ ข้อ 3)

                after_url = _normalize_url(page.url)
                if after_url != before_url:
                    # W34: รอ URL นิ่งก่อนตัดสิน same/off-domain (cascading redirect); เรียกเฉพาะเมื่อ navigate จริง
                    # เพราะเรียกทุกปุ่มจะช้าลงมาก
                    await _settle_url(page)
                    after_url = _normalize_url(page.url)
                if after_url != before_url and extract_domain(page.url) == domain:
                    # W37/W38: thumbnail ที่ label ไม่มีคำใบ้หลุด tier filter มาได้ — เช็ค URL ปลายทาง; ไม่บันทึก ไม่ใช่ error
                    excluded_reason = _excluded_content_reason(page.url)
                    if excluded_reason:
                        if on_progress:
                            await on_progress({
                                "kind": f"{excluded_reason}_content_skipped", "url": before_url,
                                "button": label, "landed_on": page.url,
                            })
                    elif after_url not in visited and after_url not in queued:
                        visited.add(after_url)
                        await _reveal_dynamic_content(page)
                        new_page_info, new_nav_links = await extract_page(page)
                        # W33: ปิด template dedup บนเส้นทาง DFS (มี signature-cap W28/W29 ของตัวเอง)
                        await _record_page(new_page_info, new_nav_links, check_page_template=False)

                    try:
                        await page.go_back(timeout=8000)
                        await page.wait_for_load_state("networkidle", timeout=5000)
                    except Exception:
                        pass
                    if _normalize_url(page.url) != before_url:
                        try:
                            await page.goto(before_url, timeout=15000)
                            await page.wait_for_load_state("networkidle", timeout=8000)
                        except Exception:
                            break  # กลับหน้าตั้งต้นไม่ได้จริงๆ — เลิกไล่ปุ่มที่เหลือของหน้านี้
                elif after_url != before_url:
                    # W34: หลุดนอกโดเมน (ลิงก์ label "View"/"Continue" ผ่าน is_crawl_safe แต่ DFS-click ไม่รู้ปลายทางล่วงหน้า)
                    # — ห้ามสำรวจเว็บอื่น กลับทันที
                    manual_errors.append({
                        "url": before_url, "phase": "click", "button": label,
                        "error": f"navigate ออกนอกโดเมนเป้าหมาย ({domain}): {page.url}",
                    })
                    if on_progress:
                        await on_progress({
                            "kind": "off_domain_navigation", "url": before_url,
                            "button": label, "landed_on": page.url,
                        })
                    # W34: goto แทน go_back() — bfcache restore ข้าม cross-origin (localhost -> 127.0.0.1) ทำให้ปุ่มถัดไป
                    # ใช้ 15-20s กว่า click จะติด; goto ได้ DOM สด
                    try:
                        await page.goto(before_url, timeout=15000)
                        await page.wait_for_load_state("networkidle", timeout=8000)
                    except Exception:
                        break  # กลับหน้าตั้งต้นไม่ได้จริงๆ — เลิกไล่ปุ่มที่เหลือของหน้านี้
                else:
                    # ไม่ navigate — modal/panel/tab/dropdown ในหน้าเดิม: re-extract แล้ว merge
                    known_selectors_before = {b.selector for b in base_page_info.buttons if b.selector}
                    try:
                        revealed_info, _ = await extract_page(page)
                        _merge_page_info(base_page_info, revealed_info)
                        # W24: ไล่กดเฉพาะปุ่มที่เพิ่งโผล่ (ซ่อนด้วย display:none ตอนแรก) อีก 1 ชั้น ผ่าน PageInfo ชั่วคราว
                        if depth < _MAX_REVEAL_DEPTH:
                            newly_revealed_buttons = [
                                b for b in revealed_info.buttons
                                if b.selector and b.selector not in known_selectors_before
                            ]
                            if newly_revealed_buttons:
                                await _explore_buttons(PageInfo(buttons=newly_revealed_buttons), depth=depth + 1)
                    except Exception:
                        pass
                    try:
                        await page.keyboard.press("Escape")
                    except Exception:
                        pass

        async def _login_and_continue(
            login_username: str, login_password: str, page_info: PageInfo, nav_links: list[dict],
        ) -> None:
            """บันทึกหน้า login, attempt_login(), ตรวจ session ด้วย verify_login_success() (W24: URL เปลี่ยน + ไม่มีฟอร์ม
            login เหลือ) แล้วบันทึกหน้าหลัง login. cookie/storage token เป็นแค่ข้อมูลใน event "login_result" ไม่ใช่เงื่อนไข
            (เว็บ token-based ไม่มี cookie; fixture ทดสอบก็ไม่มี).

            W40: หน้า login ต้อง explore_buttons=False — ปุ่ม submit อย่าง "Continue" (อยู่ใน ALLOWED) ไม่ตรง
            _LOGIN_SUBMIT_KEYWORDS ทำให้ attempt_login() คืน False ถูกแล้ว แต่ _explore_buttons() จะไปกด submit แทน
            (test_crawl_site_login_bootstrap_fails_gracefully_without_submit_button)"""
            await _record_page(page_info, nav_links, explore_buttons=False)
            pre_login_url = page.url
            did_login = await attempt_login(page, page_info, login_username, login_password)

            session_ok = False
            reason = ""
            post_page_info: Optional[PageInfo] = None
            post_nav_links: list[dict] = []
            if not did_login:
                reason = "กรอกฟอร์ม/กดปุ่ม submit ไม่สำเร็จ (หา field/ปุ่มไม่ครบ หรือ fill/click ล้มเหลว)"
            else:
                session_ok, reason = await verify_login_success(page, pre_login_url)
                if session_ok:
                    # reveal + รอ DOM นิ่ง + extract ซ้ำ ให้ได้โครงสร้างหน้าหลัง login ครบที่สุด
                    await _reveal_dynamic_content(page)
                    await _wait_for_dom_stable(page)
                    post_page_info, post_nav_links = await extract_page(page)

            cookie_count = 0
            try:
                cookie_count = len(await context.cookies())
            except Exception:
                pass
            has_storage_token = False
            try:
                has_storage_token = bool(await page.evaluate(
                    "() => Object.keys(window.localStorage||{}).length + Object.keys(window.sessionStorage||{}).length > 0"
                ))
            except Exception:
                pass
            if on_progress:
                await on_progress({
                    "kind": "login_result", "success": session_ok, "url": page.url,
                    "reason": reason, "cookie_count": cookie_count, "has_storage_token": has_storage_token,
                })

            if session_ok and post_page_info is not None:
                post_login_url = _normalize_url(page.url)
                if post_login_url not in visited:
                    visited.add(post_login_url)
                    await page.bring_to_front()
                    if on_progress:
                        await on_progress({"kind": "page_start", "url": page.url})
                    await _record_page(post_page_info, post_nav_links)
            elif did_login and not session_ok:
                manual_errors.append({"url": page.url, "phase": "login", "error": reason})

        while queue and len(pages) < effective_max_pages:
            url, parent_url, arrived_via = queue.pop(0)
            normalized = _normalize_url(url)
            if normalized in visited:
                continue
            visited.add(normalized)

            # W24: retry ก่อนยอมแพ้ (แยก transient กับพังจริง) แล้วบันทึก error + event แทนกลืนเงียบ
            goto_error = await _goto_with_retry(page, url, settings.site_learning_goto_retries)
            if goto_error is not None:
                manual_errors.append({"url": url, "phase": "goto", "error": goto_error})
                if on_progress:
                    await on_progress({"kind": "page_error", "url": url, "phase": "goto", "error": goto_error})
                continue  # หน้านี้ไปไม่ถึงแม้ retry ครบแล้ว (404/timeout/DNS ฯลฯ) — ข้ามไปหน้าถัดไป ไม่ล้มทั้ง crawl

            # W34: same-domain ตอนต่อคิวแต่อาจ redirect ออกนอกโดเมน (URL shortener, OAuth bounce) — แค่ข้าม URL นี้
            # (BFS ไม่มีหน้าเดิมให้กลับ) ไม่บันทึกหน้านอกโดเมนเด็ดขาด
            await _settle_url(page)  # W34: กัน cascading redirect ที่ยังไม่จบตอนเช็ค
            if extract_domain(page.url) != domain:
                manual_errors.append({
                    "url": url, "phase": "goto", "error": f"redirect ออกนอกโดเมนเป้าหมาย ({domain}): {page.url}",
                })
                if on_progress:
                    await on_progress({"kind": "off_domain_navigation", "url": url, "landed_on": page.url})
                continue

            # W16: page_start ก่อน extract/describe ให้ UI (browser headless=False) โชว์หน้าที่กำลังเรียนทันที
            await page.bring_to_front()
            if on_progress:
                await on_progress({"kind": "page_start", "url": page.url})

            # W24: เผย lazy content + รอ DOM นิ่ง (SPA) ก่อน extract
            await _reveal_dynamic_content(page)
            await _wait_for_dom_stable(page)
            page_info, nav_links = await extract_page(page)
            # W66[A]: ผูก parent/arrived_via ครั้งเดียว ครอบทุก _record_page(page_info) ในลูปนี้
            page_info.parent_url = parent_url
            page_info.arrived_via = arrived_via

            # W15: login bootstrap ครั้งเดียวต่อ crawl (หน้า "เปลี่ยนรหัสผ่าน" หลัง login ไม่ควรถูก submit)
            if not login_attempted and username and password:
                login_attempted = True
                await _login_and_continue(username, password, page_info, nav_links)
                continue

            # W23: ถามคนจริงเฉพาะหน้าที่เป็นหน้า login จริง (find_login_fields ครบ) และถามครั้งเดียว (login_attempted)
            if (
                not login_attempted
                and on_credentials_needed is not None
                and find_login_fields(page_info) != (None, None)
            ):
                login_attempted = True
                creds = await on_credentials_needed(domain)
                if creds and creds.get("username") and creds.get("password"):
                    await _login_and_continue(creds["username"], creds["password"], page_info, nav_links)
                    continue
                # user ข้าม/หมดเวลา — บันทึกหน้านี้แล้วสำรวจต่อโดยไม่ login (ไม่ใช่ error)

            # W?: เจอฟอร์ม login อีกหลังลองไปแล้ว (session หลุด/login ของ section อื่น) — ห้าม submit ซ้ำ; ถือเป็น
            # dead-end: บันทึกหน้าไว้แต่ไม่ต่อคิว nav_links ไม่ไล่ปุ่ม และบันทึก manual_errors ว่าส่วนนี้อาจไม่สมบูรณ์
            if login_attempted and find_login_fields(page_info) != (None, None):
                # check_page_template=False: หน้า login ซ้ำมี template เหมือนรอบแรกเป๊ะ W33 จะข้ามเงียบ ทำให้ manual
                # มีแค่ error dead-end โดยไม่มีหน้านั้น
                await _record_page(
                    page_info, [], check_page_template=False, explore_buttons=False,
                )
                reason = "เจอฟอร์ม login ซ้ำระหว่าง crawl หลัง login ไปแล้วก่อนหน้านี้ — ไม่ login ซ้ำ ถือเป็น dead-end (manual ส่วนนี้อาจไม่สมบูรณ์เพราะสำรวจได้ในสถานะยังไม่ login)"
                manual_errors.append({"url": page.url, "phase": "login", "error": reason})
                if on_progress:
                    await on_progress({
                        "kind": "login_result", "success": False, "url": page.url, "reason": reason,
                        "cookie_count": 0, "has_storage_token": False,
                    })
                continue

            await _record_page(page_info, nav_links)
    finally:
        await context.close()

    # W26: สรุปภาพรวมเว็บครั้งเดียวหลังสำรวจครบ
    site_summary = await describe_site(client, model, resolved_provider, domain, pages)
    manual = SiteManual(
        website=domain, pages=pages, generated_at=time.time(), errors=manual_errors, summary=site_summary,
    )
    if on_progress:
        # W24: errors_found บอกว่าจบเพราะครบจริง หรือจบทั้งที่เจอปัญหา
        await on_progress({
            "kind": "crawl_scan_done", "pages_found": len(pages), "errors_found": len(manual_errors),
        })
    return manual

"""site_learning/crawler.py — W14: BFS deterministic crawler ที่สร้าง SiteManual — เดิน
DOM หา nav link เองทั้งหมด (ไม่ให้ LLM ตัดสินใจว่าจะคลิก/ไปหน้าไหนต่อ ตามที่ user ยืนยัน
ไว้ตอนคุยแผน เพื่อลด token cost ตามเป้าหมายหลักของฟีเจอร์นี้) เรียก LLM แค่ครั้งเดียวต่อ
หน้าเพื่อเขียนชื่อ+คำอธิบายสั้นๆ เท่านั้น — ทุกอย่างอื่น (nav structure, button/form/
table extraction, selector/xpath) เป็น DOM data ล้วนๆ ไม่มี LLM เกี่ยวข้องเลย

W15: ข้อยกเว้นแรกจากกฎ "ห้ามกด Submit" — login bootstrap (ดู
site_learning/auto_login.py::attempt_login()) เพราะเว็บส่วนใหญ่ไม่มี nav link ใดๆ ให้เดินต่อเลยจนกว่าจะ sign in ก่อน (หน้า
login มีแค่ฟอร์ม ไม่มีเมนู) ถ้าไม่ยอม submit ฟอร์มนี้เลย crawler จะสำรวจได้แค่หน้าแรกหน้า
เดียวเสมอ — อนุญาตเฉพาะตอนที่ caller ส่ง username/password มาเองตรงๆ (ไม่ใช่ agent/LLM
ตัดสินใจเอง) และเจอ password field จริงบนหน้าเท่านั้น ไม่บันทึก credential ไว้ที่ไหนเลย
(ไม่ใส่ใน SiteManual, ไม่ log, ไม่ปรากฏใน progress event)

W16: นอกจากเดินตาม nav link แล้ว ตอนนี้ crawler ยัง "ไล่กด" ปุ่มที่ปลอดภัย (ดู
safety.is_crawl_safe — allowlist เดิม ไม่แตะ Delete/Submit/Purchase/Logout ฯลฯ) ทีละปุ่ม
บนแต่ละหน้า เพื่อสำรวจ path/สถานะที่ nav link เดินไม่ถึง (เช่น ปุ่ม "View" ในตารางที่พาไป
หน้ารายละเอียด, ปุ่ม "Expand" ที่เปิด panel) — เป็น DFS แบบไม่จำกัดความลึก (ตามที่ user
ยืนยัน: "กดต่อไปเรื่อยๆ จนมั่นใจว่าไม่มีทางไปต่อ ... พอตันแล้วให้ถอยกลับแล้วเปลี่ยนปุ่ม")
หน้าใหม่ที่เจอจากการกดปุ่มก็ถูกไล่กดปุ่มของมันต่อเองเสมอ ไม่มี depth cap แล้ว — ตันเมื่อไหร่
(ไม่มีปุ่มปลอดภัยเหลือ/ทุกปุ่มพาไปหน้าที่เคยเจอแล้ว) ก็ถอยกลับไปลองปุ่มอื่นของหน้าก่อนหน้า
โดยอัตโนมัติ (ดู _explore_buttons() — หลังกดแต่ละปุ่มจะย้อนกลับมาหน้าตั้งต้นเสมอก่อนลองปุ่ม
ถัดไป: page.go_back() ก่อน มี page.goto() เป็น fallback ถ้า go_back ไม่พากลับไป URL เดิม
จริง) การเดินจึงจบเองได้แน่นอนด้วย visited-set (ไม่เดินซ้ำหน้าที่เคยเจอ) +
settings.site_learning_max_pages (เพดานรวมทั้ง crawl) + settings.site_learning_max_buttons_per_page
(เพดานต่อหน้า กันหน้าที่มีปุ่มเยอะผิดปกติทำให้ตันช้าเกินไป)
"""

import json
import re
import time
import urllib.parse
from typing import Awaitable, Callable, Optional

from playwright.async_api import Browser, Page

from backend.app.config import settings
from backend.app.core import llm
from backend.app.core.orchestrator import Orchestrator
from backend.app.permission.rules import extract_domain
from backend.app.site_learning.auto_login import attempt_login, find_login_fields
from backend.app.site_learning.extractor import extract_page
from backend.app.site_learning.safety import is_crawl_safe, is_safe_nav_link
from backend.app.site_learning.schema import PageInfo, SiteManual

OnProgressFunc = Callable[[dict], Awaitable[None]]
# W23: เรียกตอนเจอหน้าที่มี password field จริง แต่ยังไม่มี username/password ให้ใช้เลย
# (ไม่ได้ส่งมาตอนเริ่ม crawl) — รับ domain (ของเว็บที่กำลัง crawl อยู่นี้เท่านั้น) คืน
# {"username":..., "password":...} ถ้า user กรอกจริง หรือ None ถ้า user เลือกข้าม/หมดเวลา
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
    """เรียก LLM ครั้งเดียวต่อหน้าเพื่อตั้งชื่อ+เขียนคำอธิบายจากโครงสร้างที่สกัดมาแล้ว —
    ไม่เคยใช้ LLM ตัดสินใจ navigate เลย ถ้า parse ผลลัพธ์ไม่ได้ (โมเดลตอบนอกรูปแบบ JSON)
    fallback เป็นชื่อจาก URL path เฉยๆ ไม่ throw ออกไปกลางการ crawl (1 หน้าพังไม่ควรทำ
    ทั้ง crawl ล้มเหลวไปด้วย)"""
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


def _merge_page_info(base: PageInfo, extra: PageInfo) -> None:
    """รวมโครงสร้างที่เพิ่งโผล่มาใหม่ (เช่น modal/panel/tab ที่เปิดจากปุ่มที่เพิ่งกด แต่ URL
    ไม่เปลี่ยน — ดู _explore_buttons) เข้ากับ PageInfo เดิมของหน้านี้ — dedup ด้วย selector
    (buttons/forms) และด้วยค่าตรงๆ (modals/tabs/tables) กันซ้ำกับที่เจอไปแล้วตอน extract
    ครั้งแรก ไม่สร้าง PageInfo/entry ใหม่แยกต่างหาก เพราะยังเป็น URL เดียวกัน (ไม่ใช่หน้า
    ใหม่จริงๆ) — เป้าหมายคือให้ manual จับโครงสร้างที่ซ่อนอยู่หลังปุ่มได้ครบถ้วนกว่าเดิม"""
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
    """label ที่ดีที่สุดเท่าที่มีของปุ่มนี้ — text > aria_label > title > icon_hint (ลำดับ
    เดิมตามที่ ButtonInfo.icon_hint กำหนดไว้) เพิ่ม data_testid (humanized) เป็น fallback
    สุดท้ายอีกชั้น: เว็บที่ทำ QA อัตโนมัติ (เช่น saucedemo) มักใส่ data-test/data-testid ไว้
    บนปุ่ม icon-only ที่ไม่มี text/aria-label/title/svg-title/icon-font class เลยสักอย่าง
    (เช่น ปุ่มตะกร้าสินค้ามุมขวาบนที่เป็นแค่ svg ไม่มี label ให้ inferIconHint() เดาได้ —
    ดู extractor.py) ทำให้ label ออกมาว่างเปล่า แล้ว is_crawl_safe() เห็นว่า "ไม่แน่ใจ"
    ปฏิเสธไม่กดตลอดไป (ดู safety.py) ทั้งที่ data-test บอกไว้ชัดเจนอยู่แล้วว่าปุ่มนี้คืออะไร
    เช่น "shopping-cart-link" -> "shopping cart link" — ไม่กระทบ selector ที่ใช้ dispatch
    จริง (ยังคง button.selector เดิมเป๊ะ) แค่ทำให้ตัดสินใจ "ควรกดไหม" ได้แม่นขึ้นเท่านั้น"""
    return (
        button.text or button.aria_label or button.title or button.icon_hint
        or button.data_testid.replace("-", " ").replace("_", " ").strip()
    )


def _normalize_url(url: str) -> str:
    """ตัด fragment ออกกันนับซ้ำ (URL ต่างกันแค่ #section ไม่ควรถือว่าเป็นคนละหน้า) และ
    ตัด trailing slash ให้เหมือนกันเสมอ"""
    parsed = urllib.parse.urlparse(url)
    path = parsed.path.rstrip("/") or "/"
    return urllib.parse.urlunparse((parsed.scheme, parsed.netloc, path, "", parsed.query, ""))


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
    """BFS deterministic crawl เริ่มจาก start_url — เดินเฉพาะลิงก์ same-origin ที่เจอใน
    nav/menu (ดู extractor.py::extract_page) ทีละหน้า กรองด้วย
    safety.is_safe_nav_link() ก่อนเสมอ (ห้ามเดินตามลิงก์ที่ label เข้าข่ายทำลาย/
    เปลี่ยนแปลงข้อมูล เช่น "Logout") คืน SiteManual ที่ยังไม่ได้ save ลงดิสก์ (caller เป็น
    คนเรียก storage.save_manual() เอง — ดู routes.py) เปิด BrowserContext ของตัวเองแยก
    ต่างหาก (ไม่แตะ context/page อื่นของ browser ที่ยืมมา) ปิดให้เสมอก่อน return

    username/password (W15, optional): ถ้าให้มาทั้งคู่ และหน้าแรกที่เจอมี password
    field จริง จะลองกรอก+กด sign in ครั้งเดียว (ดู auto_login.py::attempt_login()) ก่อนสำรวจต่อ —
    จำเป็นเพราะเว็บส่วนใหญ่ไม่มี nav link ให้เดินต่อเลยจนกว่าจะ login (หน้า login มีแค่
    ฟอร์ม)

    on_credentials_needed (W23, optional): ถ้าไม่ได้ให้ username/password มาเลยตอนเริ่ม
    crawl (ทั้งคู่ None) แต่ crawler เจอหน้าที่มี password field จริง (find_login_fields()
    เจอครบทั้งคู่) ระหว่างทาง จะเรียก callback นี้ (ส่ง domain ของเว็บนี้ไปด้วย) แล้ว "หยุด
    รอ" (await) จน user ตอบกลับผ่าน UI จริง (ดู routes.py::learn_site() ที่ผูก callback นี้
    เข้ากับ LearnManager.request_credentials()) — ได้ dict {username, password} กลับมา =
    ใช้ login bootstrap ต่อทันที, ได้ None กลับมา (user เลือกข้าม/หมดเวลา) = บันทึกหน้านี้
    ตามปกติแล้วสำรวจต่อโดยไม่ login (เหมือนไม่เคยมี callback นี้เลย) ถามแค่ครั้งเดียวตลอด
    ทั้ง crawl เท่ากับ username/password (login_attempted ตัวเดียวกัน) ไม่ระบุอะไรเลย (ทั้ง
    username/password และ callback นี้เป็น None หมด) = พฤติกรรมเดิมทุกประการ ไม่แตะฟอร์ม
    ใดๆ เลย ไม่ถามใคร

    W16: นอกจากเดิน nav link แล้ว ทุกหน้าที่บันทึก (ผ่าน _record_page) จะถูกไล่กดปุ่ม
    "ปลอดภัย" ด้วย (ดู _explore_buttons/is_crawl_safe) เพื่อสำรวจ path ที่ nav link เดิน
    ไม่ถึง — ไม่จำกัดความลึก (DFS กดไปเรื่อยๆ จนตัน แล้วถอยกลับไปลองปุ่มอื่น) จำกัดแค่
    จำนวนปุ่มต่อหน้าไว้ที่ settings.site_learning_max_buttons_per_page และเพดานรวมทั้ง
    crawl ที่ settings.site_learning_max_pages (ตัวเดียวกับที่คุม nav-link BFS)"""
    domain = extract_domain(start_url)
    resolved_provider = provider or settings.llm_provider
    client, model, _, _, _ = Orchestrator._llm_backend(resolved_provider)
    effective_max_pages = max_pages or settings.site_learning_max_pages

    context = await browser.new_context()
    page = await context.new_page()
    pages: list[PageInfo] = []
    try:
        visited: set[str] = set()
        queue: list[str] = [start_url]
        queued: set[str] = {_normalize_url(start_url)}
        estimated_total = 1
        login_attempted = False

        async def _record_page(page_info: PageInfo, nav_links: list[dict]) -> None:
            """describe + เก็บเข้า pages + ยิง progress event + ไล่กดปุ่มปลอดภัย + ต่อคิว
            nav link ที่ปลอดภัย — logic ร่วมที่ใช้ทั้งกับหน้าที่เจอจาก BFS ปกติ, หน้าหลัง
            login bootstrap, และหน้าที่เจอจากการไล่กดปุ่ม (ดู docstring หัวไฟล์) ไล่กดปุ่ม
            ของทุกหน้าที่บันทึกเสมอ ไม่ว่าจะเจอหน้านั้นจากทางไหน (nav link/ปุ่ม/login) —
            ไม่มี depth cap แล้ว (W16) การันตีว่าจบได้จริงด้วย visited-set +
            effective_max_pages เท่านั้น"""
            nonlocal estimated_total
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

            if len(pages) < effective_max_pages:
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
                normalized_link = _normalize_url(absolute)
                if normalized_link in visited or normalized_link in queued:
                    continue
                queue.append(absolute)
                queued.add(normalized_link)
                estimated_total = max(estimated_total, len(pages) + len(queue))

        async def _explore_buttons(base_page_info: PageInfo) -> None:
            """ไล่กดปุ่มที่ is_crawl_safe() อนุญาตทีละปุ่มบนหน้านี้ (base_page_info.url) —
            กดแล้วเช็คว่า URL เปลี่ยนไหม: เปลี่ยน = เจอหน้าใหม่ (บันทึกผ่าน _record_page ถ้า
            ยังไม่เคยเจอ — ซึ่งจะไล่กดปุ่มของหน้าใหม่นั้นต่อทันที เป็น DFS แบบไม่จำกัดความลึก
            — แล้ว go_back()/goto() กลับมาหน้าตั้งต้นก่อนลองปุ่มถัดไปเสมอ, ถ้าเคยเจอแล้ว
            (visited/queued) ก็แค่กลับมาเฉยๆ ไม่สำรวจซ้ำ นี่คือจุดที่ทำให้ DFS "ตัน" แล้ว
            ถอยกลับไปลองปุ่มอื่นของหน้าก่อนหน้าโดยธรรมชาติ) ไม่เปลี่ยน = แค่ modal/panel/tab
            เปิดในหน้าเดิม (re-extract แล้ว merge เข้า base_page_info ผ่าน _merge_page_info
            แทนที่จะสร้างหน้าใหม่ + กด Escape ปิดแบบ best-effort ก่อนไปปุ่มถัดไป) ไม่ throw
            ออกไปแม้ปุ่มไหนกด/กลับไม่ได้ — ข้ามไปปุ่มถัดไปเงียบๆ เว้นแต่ "กลับหน้าตั้งต้น
            ไม่ได้เลย" ซึ่งเลิกไล่ปุ่มที่เหลือของหน้านี้ทันที (state ของหน้าพังไปแล้ว ไล่ต่อ
            ไม่มีประโยชน์)"""
            before_url = _normalize_url(page.url)
            # W18: รวมปุ่มระดับหน้า + ปุ่ม "ตัวแทน" ของ UI pattern แต่ละแบบ (ดู
            # extractor.py::UIPatternInfo — เช่น ปุ่ม "View" ใน product card ที่ซ้ำกัน 100
            # ใบ ตอนนี้เหลือ instance เดียวให้ลองกด) เข้าลิสต์เดียวกัน กรองด้วย
            # is_crawl_safe() เหมือนกันทุกประการ — label ใช้ text > aria_label > title >
            # icon_hint (เผื่อเป็นปุ่ม icon-only ที่ไม่มี text/aria-label/title เลย)
            candidate_buttons = list(base_page_info.buttons)
            for pattern in base_page_info.ui_patterns:
                candidate_buttons.extend(pattern.buttons)
            safe_buttons = [
                b for b in candidate_buttons
                if b.selector and is_crawl_safe(_button_label(b), cmd_type="click")
            ][:settings.site_learning_max_buttons_per_page]

            for button in safe_buttons:
                if len(pages) >= effective_max_pages:
                    break
                label = _button_label(button)
                if on_progress:
                    await on_progress({"kind": "button_explored", "url": before_url, "button": label})
                try:
                    await page.click(button.selector, timeout=5000)
                    await page.wait_for_timeout(400)
                    try:
                        await page.wait_for_load_state("networkidle", timeout=3000)
                    except Exception:
                        pass
                except Exception:
                    continue  # กดปุ่มนี้ไม่ได้ (element หาย/ถูกบัง/detach ฯลฯ) ข้ามไปปุ่มถัดไป

                # ปุ่มบาง element เป็น target="_blank" เปิดแท็บใหม่แทนที่จะ navigate หน้า
                # เดิม — ปิดแท็บที่เพิ่งเปิดทิ้งทันที (ไม่ตามไปสำรวจ) กัน context สะสม page
                # ค้างเป็นสิบๆ ตัวถ้าไล่กดหลายร้อยปุ่มตลอด crawl
                for extra_page in list(context.pages):
                    if extra_page != page:
                        try:
                            await extra_page.close()
                        except Exception:
                            pass

                after_url = _normalize_url(page.url)
                if after_url != before_url and extract_domain(page.url) == domain:
                    if after_url not in visited and after_url not in queued:
                        visited.add(after_url)
                        new_page_info, new_nav_links = await extract_page(page)
                        await _record_page(new_page_info, new_nav_links)

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
                else:
                    # ไม่ navigate ไปไหน — น่าจะเป็น modal/expand panel/tab ที่เปิดในหน้าเดิม
                    # re-extract แล้ว merge โครงสร้างใหม่เข้าไปในหน้านี้ (ไม่ใช่หน้าใหม่จริง)
                    try:
                        revealed_info, _ = await extract_page(page)
                        _merge_page_info(base_page_info, revealed_info)
                    except Exception:
                        pass
                    try:
                        await page.keyboard.press("Escape")
                    except Exception:
                        pass

        async def _login_and_continue(
            login_username: str, login_password: str, page_info: PageInfo, nav_links: list[dict],
        ) -> None:
            """บันทึกหน้า login เอง (มีประโยชน์ต่อ manual) แล้วลอง attempt_login() —
            สำเร็จ (did_login=True — แค่แปลว่ากด submit ได้จริง ไม่ได้การันตีว่า login
            ผ่าน) ก็ extract+บันทึกหน้าถัดจาก login ต่อด้วยเลย ใช้ร่วมกันทั้ง 2 เส้นทางที่มี
            username/password มาใช้ได้ (ส่งมาตั้งแต่ต้น crawl กับได้จาก
            on_credentials_needed ระหว่างทาง — ดู docstring ของ crawl_site())"""
            await _record_page(page_info, nav_links)
            did_login = await attempt_login(page, page_info, login_username, login_password)
            if did_login:
                post_login_url = _normalize_url(page.url)
                if post_login_url not in visited:
                    visited.add(post_login_url)
                    await page.bring_to_front()
                    if on_progress:
                        await on_progress({"kind": "page_start", "url": page.url})
                    post_page_info, post_nav_links = await extract_page(page)
                    await _record_page(post_page_info, post_nav_links)

        while queue and len(pages) < effective_max_pages:
            url = queue.pop(0)
            normalized = _normalize_url(url)
            if normalized in visited:
                continue
            visited.add(normalized)

            try:
                await page.goto(url, timeout=15000)
                await page.wait_for_load_state("networkidle", timeout=8000)
            except Exception:
                # หน้านี้ไปไม่ถึง (404/timeout/DNS ฯลฯ) — ข้ามไปหน้าถัดไปเงียบๆ ไม่ล้ม
                # ทั้ง crawl เพราะแค่หน้าเดียวมีปัญหา
                continue

            # W16: ยิงก่อน extract/describe (ซึ่งกินเวลาจาก LLM call) เพื่อให้ UI โชว์ "กำลัง
            # เรียนรู้หน้านี้อยู่" ได้ทันทีที่หน้าโหลดเสร็จ ไม่ต้องรอ page_done — คู่กับ
            # browser ที่เปิดแบบมองเห็นได้ (headless=False, ดู routes.py::learn_site()) ให้
            # user เห็นจริงๆ ว่ากำลังเดินอยู่หน้าไหน
            await page.bring_to_front()
            if on_progress:
                await on_progress({"kind": "page_start", "url": page.url})

            page_info, nav_links = await extract_page(page)

            # W15: login bootstrap — ลองแค่ครั้งเดียวตลอดทั้ง crawl (login_attempted)
            # ตรงหน้าแรกที่เจอ password field จริงเท่านั้น ไม่ใช่ทุกหน้าที่มี password
            # field (เช่น หน้า "เปลี่ยนรหัสผ่าน" หลัง login ไปแล้วไม่ควรลอง submit ซ้ำ)
            if not login_attempted and username and password:
                login_attempted = True
                await _login_and_continue(username, password, page_info, nav_links)
                continue

            # W23: ไม่มี username/password ให้มาตั้งแต่ต้นเลย แต่มี on_credentials_needed
            # ให้ "ถามคนจริง" ได้ — เช็คว่าหน้านี้เข้าข่ายหน้า login จริงก่อน
            # (find_login_fields เจอทั้ง username+password field ครบ) ค่อยเรียก ไม่งั้นจะ
            # ถามทุกครั้งที่ยังไม่เคย login แม้หน้านั้นไม่ใช่หน้า login เลยก็ตาม (ถามครั้ง
            # เดียวตลอด crawl เหมือนกับ username/password ด้านบน — login_attempted ตัว
            # เดียวกัน กันถามซ้ำถ้า user เพิ่งเลือกข้ามไปแล้ว)
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
                # user เลือกข้าม/หมดเวลา — บันทึกหน้านี้ตามปกติแล้วสำรวจต่อโดยไม่ login
                # (เหมือนไม่เคยมี callback นี้เลย ไม่ใช่ error)

            await _record_page(page_info, nav_links)
    finally:
        await context.close()

    manual = SiteManual(website=domain, pages=pages, generated_at=time.time())
    if on_progress:
        await on_progress({"kind": "crawl_scan_done", "pages_found": len(pages)})
    return manual

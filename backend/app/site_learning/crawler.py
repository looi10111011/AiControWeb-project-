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

W24 — ปรับปรุงตามข้อร้องขอชุด "แก้ไข self learning" (ทำให้ contract ของการ "เรียนรู้เว็บไซต์"
ชัดเจน+ตรวจสอบได้ แทนคำสั่งกว้างๆ แบบ "เข้าเว็บและเรียนรู้" — ระบบนี้เป็น deterministic
crawler ไม่ใช่ LLM agent ที่ตัดสินใจเองจาก prompt ตามที่ W14 ตั้งใจไว้ตั้งแต่ต้น ดู
docstring บรรทัดแรกของไฟล์นี้ — เลยแปล requirement เป็นการบังคับ+ตรวจสอบ "ขั้นตอน" ในโค้ด
ตรงๆ แทนที่จะเขียนเป็นข้อความ prompt ให้ LLM ตีความเอง) ครบทุกขั้นตอนที่ระบุ:
  1. Login -> สำรวจทุกเมนู -> คลิกทุกหน้าที่เข้าถึงได้ -> อ่านข้อมูลทุกหน้า -> บันทึกข้อมูล
     -> จบเมื่อ queue ว่าง+retry ครบเท่านั้น (ของเดิมมีอยู่แล้วเกือบทั้งหมด ยกเว้นจุดที่ระบุ
     ด้านล่าง — ไม่ใช่จบทันทีหลัง login/เข้า dashboard เพราะ loop หลักเดินตาม queue ต่อเสมอ)
  2. Queue-based BFS (มีอยู่แล้วตั้งแต่ W14 — queue/queued/visited ด้านล่าง)
  3. SPA support: เดิมพึ่ง wait_for_load_state("networkidle") อย่างเดียว ซึ่งไม่พอสำหรับ
     client-side routing ที่ไม่ยิง network request ใหม่เลย (route เปลี่ยนจาก cache/state
     ล้วนๆ) — เพิ่ม _wait_for_dom_stable() (poll ความยาว DOM จนนิ่ง) เรียกคู่กับ
     networkidle ทุกจุดที่ navigate/click แล้ว
  0. (W25, ต่อยอดจากรอบนี้) "Extract all clickable elements (a/button/role=button/
     role=link) -> queue -> click ทีละตัว -> wait for load -> learn -> go back -> ทำจนกว่า
     queue จะว่าง": button/role=button/role=link เป็นแบบนี้อยู่แล้วทุกประการผ่าน
     _explore_buttons() (ดูข้อ 4 ด้านล่าง) — ส่วน <a href> เดิม extractor.py สแกนหาแค่ที่
     อยู่ใน NAV_CONTAINERS (nav/aside/header/footer/[role=tablist]/[role=menu]) เท่านั้น
     พลาดลิงก์ในเนื้อหา (content area) ที่ไม่มีทางอื่นเข้าถึงได้เลย — แก้เป็นสแกนทั้ง
     เอกสาร (ดู extractor.py::_EXTRACT_JS ส่วน nav links) ให้ "extract all clickable a" จริง
     *** ตั้งใจไม่เปลี่ยน <a href> จาก goto()-based BFS มาเป็น click()+go_back() แบบ
     button — สองแบบนี้ให้ manual ผลลัพธ์เหมือนกันทุกประการ (เข้าเว็บ/เรียนรู้/บันทึกครบ)
     แค่ goto() ตรงไปที่ URL ที่รู้อยู่แล้วจาก href ได้เลย ไม่ต้องเสียเวลา click+กลับมาที่
     หน้าเดิมก่อนไปหน้าถัดไป (ไม่มี "หน้าเดิม" ให้ต้องกลับเลยในการ goto ตามคิวแบบ BFS
     เรียงต่อกัน) — เปลี่ยนแค่ตอนนี้ scope กว้างขึ้น (ทั้งเอกสาร ไม่ใช่แค่ nav) ***
  4. เมนูที่ไม่ใช่ <a>: extractor.py เพิ่ม role=menuitem/role=tab/router-link เข้า
     BUTTON_SELECTOR + ธง is_nav_menu_item — ปุ่ม/element ที่ธงนี้ true จะถูกไล่กดแบบ
     default-allow (เหมือน nav link ปกติ ดู _is_explorable ด้านล่าง) แทนที่จะต้องผ่าน
     keyword allowlist เข้มแบบปุ่มทั่วไป — ครอบคลุม sidebar/dropdown/tab ที่ไม่ได้ทำเป็น
     <a href> จริง (<a href="..."> ที่มีปลายทางจริงไม่นับเป็น nav menu item ในความหมายนี้
     — ยังเดินผ่าน BFS href เดิมที่เช็ค same-origin ได้ก่อน navigate เท่านั้น กัน
     _explore_buttons() คลิกลิงก์เดิมซ้ำแล้วเสี่ยงหลุดไปนอกโดเมนก่อนรู้ปลายทาง — ดู
     extractor.py::isNavMenuItem() สำหรับเหตุผลเต็ม)
  5. ไม่ปิด browser ก่อนเวลา: ตรวจสอบแล้ว — context.close()/browser.close() (routes.py)
     อยู่ใน finally หลัง crawl loop จบสมบูรณ์เท่านั้น ไม่มีจุดไหนปิดกลางคัน (ไม่ต้องแก้)
  6. Error handling: page.goto()/page.click() เดิมไม่มี retry เลย (fail ครั้งเดียว = ข้าม
     เงียบๆ ทันที) — เพิ่ม _goto_with_retry()/_click_with_retry() (จำนวนครั้งปรับได้จาก
     settings.site_learning_goto_retries/click_retries) + บันทึกลง SiteManual.errors +
     ยิง progress event "page_error"/"button_click_failed" แทนการกลืน exception เงียบๆ
  7. Finish condition: มีอยู่แล้ว (while queue and len(pages) < max — ดู main loop) แค่เพิ่ม
     "ลอง retry ครบแล้ว" เป็นเงื่อนไขที่ทำให้ "ข้าม" หน้านั้น (ไม่ใช่ทำให้ crawl ทั้งหมดจบ)
  8. ตรวจ session หลัง login: เดิมไม่เช็คอะไรเลยนอกจาก "กด submit ได้ไหม" (attempt_login()
     คืนแค่นั้น ไม่รู้ว่า login ผ่านจริงหรือไม่) — เพิ่มการตรวจใน _login_and_continue():
     URL เปลี่ยนจริงไหม, ยังเจอฟอร์ม login อยู่ไหม (find_login_fields), จำนวน cookie,
     มี localStorage/sessionStorage token ไหม — ยิงเป็น event "login_result" (ดูเหตุผลที่
     cookie/token เป็นแค่สัญญาณ informational ไม่ใช่เงื่อนไขบังคับ ในคอมเมนต์ของ
     _login_and_continue เอง — เว็บจำนวนมากใช้ token-based auth ไม่มี cookie เลย)
  9. Logging: progress event เดิมมีแค่ page_start/page_done/button_explored/
     crawl_scan_done — เพิ่ม login_result/page_error/button_click_failed ครบตามที่ระบุ
     (login สำเร็จไหม, error อะไรเกิดขึ้น) ส่วน "พบเมนู/หน้ากี่รายการ" มีอยู่แล้วใน
     page_done (done/total) + crawl_scan_done (pages_found) เดิม เพิ่ม errors_found เข้าไป
     ด้วย
 10. Config: ย้าย retry/scroll count จาก magic number เป็น settings.site_learning_*
     ปรับได้จาก .env (ดู config.py) พร้อมคอมเมนต์อธิบายว่าทำไม "ตั้งน้อยเกินไปจะหยุดเร็ว"
 11. Dynamic content: เพิ่ม _reveal_dynamic_content() (เลื่อนจอจน scroll height นิ่ง — รองรับ
     infinite scroll/lazy loading) เรียกก่อน extract ทุกจุด — modal/popup/accordion/
     dropdown/tab ที่เปิดจากปุ่มมีอยู่แล้ว (ดู _explore_buttons ฝั่ง "ไม่เปลี่ยน URL") แต่
     เดิมไม่ไล่กด element ที่เพิ่ง "โผล่มา" จากการเปิดนั้นต่อ (ถูกซ่อนด้วย display:none ตอน
     extract ครั้งแรก มองไม่เห็นเลย) — เพิ่ม newly-revealed pass ความลึกจำกัด 1 ชั้น (ดู
     _MAX_REVEAL_DEPTH) ให้ dropdown/accordion ที่เพิ่งเปิดถูกไล่กดต่อได้จริง
 12. กันเรียนรู้ซ้ำ: มีอยู่แล้ว (visited/queued set ทั้ง BFS และ DFS ปุ่ม — ไม่ต้องแก้)

W28: แก้ปัญหา "self learning วนลูป" ที่ user รายงาน (เจอบน YouTube Shorts — เรียนรู้ได้แค่
หน้า Shorts วนไปเรื่อยๆ, ปุ่มค้นหา/ปุ่มอื่นที่อยู่ตำแหน่งเดิมถูกกดซ้ำไม่รู้จบข้ามหลายหน้า) —
สาเหตุจริง: visited/queued (ข้อ 12 ข้างบน) กันแค่ "หน้า URL เดิม" ไม่ให้สำรวจซ้ำ แต่ไม่เคยกัน
"ปุ่มเดิม" (label+role เดียวกัน เช่น ไอคอนค้นหาบน header, ปุ่ม "Previous/Next video" บน
player) ไม่ให้ถูกไล่กดซ้ำข้าม URL ที่ต่างกัน — ผสมกับ W16 ที่ตั้งใจเอา depth cap ของ DFS
ปุ่มออกไปแล้ว (กด "Next video" ได้ URL วิดีโอใหม่ = หน้าใหม่ที่ visited-set ไม่เคยเห็น =
recurse ไล่กดปุ่มของหน้านั้นต่อ = เจอ "Next video" อีก = กด = ได้ URL ใหม่อีก ไม่รู้จบ) ทำให้
เว็บที่มีเนื้อหาไม่จำกัด (ทุกคลิป Shorts มี URL ไม่ซ้ำกันเอง) ไล่กดปุ่มเดิมพาไปหน้าใหม่ที่
"ไม่เคยเจอ" ได้เรื่อยๆ จนกิน max_pages budget ทั้งหมดไปกับหมวดเดียว ไม่เคยย้อนกลับไปสำรวจ
ส่วนอื่นของเว็บเลย (ดูย่อหน้า W16 ด้านบนที่อธิบาย DFS ไม่จำกัดความลึกไว้ตั้งแต่ต้น) — แก้ด้วย
explored_button_signatures (dict ระดับ crawl เดียวกับ visited/queued ไม่ใช่แค่ต่อหน้า) นับ
จำนวนครั้งที่ปุ่ม "label+role เดียวกัน" ถูกไล่กดข้ามทุกหน้าทั้ง crawl (ดู _button_signature) —
ถึงเพดาน settings.site_learning_max_repeat_button_clicks แล้วข้ามปุ่มนั้นไปเลยไม่ว่าจะเจอบน
หน้าไหนอีก ทำให้ DFS ไล่ตามปุ่มประเภท pagination/chrome ที่ซ้ำกันได้จำกัดจำนวนครั้งจริง แทนที่
จะไม่จำกัดเหมือนเดิม (ของเดิมที่ dedup ด้วย selector ในหน้าเดียวกัน — เช่น _merge_page_info,
known_button_selectors — ยังอยู่เหมือนเดิมทุกจุด นี่เป็นเพดานเพิ่มเติมข้าม URL เท่านั้น)

W33: user เสนอทางแก้เพิ่มเติมสำหรับเคส Shorts/Reels โดยเฉพาะ — "กดเข้าดูแค่ 1 รอบ แล้วเก็บ
เทมเพลตโครงสร้างไว้ ต่อไปหากอ่านได้โครงสร้างตรงกับ template (ต่างแค่รายละเอียดแต่ปุ่มเหมือนกัน)
ให้กด back แล้วไปทำ step ต่อไปได้เลย" — ช่องว่างจริงที่ W28/W29 (ข้างบน) ยังไม่ครอบคลุม: ถ้าเว็บ
มี feed ที่มีลิงก์ไปคลิปคนละอันจริงๆ สิบๆ รายการ (คนละ href/URL จริง ไม่ใช่ปุ่ม "Next" ตัวเดียว
ที่ถูกกดซ้ำ) explored_button_signatures ไม่ช่วยอะไรเลย เพราะแต่ละลิงก์เป็น nav_links ที่ต่อคิว
BFS ตรงๆ (ดู W25 — สแกนทั้งเอกสารแล้ว) ไม่ผ่าน _explore_buttons()/signature-cap เลย แต่ละคลิป
เลยยังถูก describe+บันทึกลง manual ครบทุกอันอยู่ดี ทั้งที่ UI เหมือนกันทุกอย่าง (ปุ่ม
Like/Comment/Share/Next) ต่างแค่เนื้อหาคลิป — เสีย LLM describe call + กิน max_pages budget
ไปกับหน้าที่ให้ข้อมูลโครงสร้างซ้ำซ้อนกันจริง

แก้ด้วย _page_template()/known_page_templates (set ระดับ crawl เดียวกับ
explored_button_signatures) — ก่อน describe+บันทึกหน้าใหม่ทุกครั้งใน _record_page() (จุดร่วม
ของทั้ง BFS/DFS-ปุ่ม/หลัง-login) เทียบ "ลายนิ้วมือโครงสร้าง" ของหน้านี้ (set ของ signature ปุ่ม
ทุกปุ่มบนหน้า + สัญญาณหยาบๆ ว่ามีฟอร์ม/ตารางไหม — ไม่สนเนื้อหา/ลำดับ) กับที่เคยเจอมาแล้ว ตรงเป๊ะ
= ข้ามไปเลย (ไม่ describe ไม่บันทึกลง pages ไม่ไล่กดปุ่มต่อ ไม่ต่อคิว nav link ของหน้านี้ —
ถือว่าได้ตัวแทนเพียงพอแล้วจากครั้งแรก) ไม่ตรง = บันทึกตามปกติ + จำ template นี้ไว้เป็นตัวแทนใหม่
— ไม่ต้องมี logic "กด back" แยกต่างหากเลยตามที่ user ขอ เพราะ _explore_buttons() (เส้นทาง DFS)
go_back()/goto(before_url) อยู่แล้วเสมอไม่ว่า _record_page() จะทำอะไรข้างในก็ตาม ส่วนเส้นทาง BFS
ก็แค่ไปหยิบ URL ถัดไปจาก queue ต่อเองอยู่แล้วโดยธรรมชาติ (ตรงกับ "ไปทำ step ต่อไปได้เลย")
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


_SITE_SUMMARY_PROMPT_TEMPLATE = (
    "นี่คือรายชื่อหน้าเว็บทั้งหมดที่สำรวจเจอในเว็บไซต์ {website} พร้อมคำอธิบายสั้นๆ ของแต่ละหน้า "
    "(สกัดจากโครงสร้าง DOM จริงของแต่ละหน้า ไม่ใช่จินตนาการ):\n{pages_summary}\n\n"
    "เขียนสรุปสั้นๆ ไม่เกิน 4 ประโยค อธิบายให้ผู้ใช้ทั่วไป (ที่ยังไม่เคยเห็นเว็บนี้มาก่อน) "
    "เข้าใจง่ายว่าเว็บไซต์นี้ทำอะไรได้บ้าง (สรุปภาพรวมความสามารถหลัก ไม่ใช่แค่ไล่ชื่อหน้าทีละ"
    "หน้า) ตอบเป็นภาษาไทยล้วนๆ เป็นข้อความธรรมดา ไม่ต้องมี markdown/bullet/หัวข้อ"
)


async def describe_site(client, model: str, provider: str, website: str, pages: list[PageInfo]) -> str:
    """W26: เรียก LLM อีกครั้งเดียว (แยกจาก describe_page ที่เรียกต่อหน้า) หลัง crawl จบทั้ง
    เว็บแล้ว เพื่อสรุปภาพรวม "เว็บไซต์นี้ทำอะไรได้บ้าง" ให้ user อ่านทันทีที่เรียนรู้เสร็จ ไม่
    ต้องไล่เปิดดูทุกหน้าเอง — ใช้แค่ name/description ของแต่ละหน้าที่มีอยู่แล้วจาก
    describe_page() (ไม่ส่ง buttons/forms/tables ดิบๆ ซ้ำ กันกิน token เกินจำเป็น สรุประดับ
    "ภาพรวมเว็บไซต์" ไม่ต้องมีรายละเอียดลึกขนาดนั้น)

    fallback ถ้า LLM ล้มเหลว/ตอบว่างเปล่า: เรียงชื่อหน้าดิบๆ แทน (ไม่ throw ออกไปกลาง crawl —
    กฎเดียวกับ describe_page()) คืนสตริงว่างเปล่าถ้าไม่มีหน้าไหนมีชื่อเลย (crawl ไม่เจออะไร)"""
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


def _button_signature(button) -> str:
    """W28/W29: ลายเซ็นของปุ่มที่ถือว่า "เหมือนกันจริง" ข้ามหลายหน้า — W29 เปลี่ยนจากใช้
    label เต็ม (_button_label: text > aria_label > title > icon_hint > data_testid) มาใช้
    แค่ text ที่เห็นจริงบนปุ่ม (มักคงที่ เช่น "Search"/"Subscribe") ตกมาถึง data_testid แล้ว
    icon_hint แทน — ***ตัด aria_label/title ออกไปเลย ไม่ใช้เป็น identity อีกต่อไป*** เพราะ
    สองฟิลด์นี้เป็น "description" ที่เว็บจำนวนมาก (เช่น YouTube) generate มาแบบมีเนื้อหาต่อ
    ท้ายที่เปลี่ยนทุก instance (เช่น aria-label="Next video: <ชื่อคลิป>", "ผลการค้นหาสำหรับ
    <คำค้น>") — ตอน W28 ยังใช้ label เต็มที่รวม aria_label/title เข้าไปด้วย ทำให้ปุ่มที่จริงๆ
    เป็นปุ่มเดียวกันทุกประการ (icon/role/ตำแหน่งในหน้าเหมือนกันเป๊ะ เช่นปุ่ม "Next video" บน
    player ของทุกคลิป Shorts) ได้ signature ไม่ตรงกันสักครั้งเพราะส่วนท้ายของ aria_label
    เปลี่ยนไปเรื่อยๆ ตามคลิป — เพดาน settings.site_learning_max_repeat_button_clicks เลยไม่มี
    ผลจริง (ไม่เคยนับว่าเป็นปุ่ม "เดิม" สักที ยังคงไล่กด "Next video"/"Search" ซ้ำข้ามหน้าได้
    ไม่รู้จบเหมือน W28 ตั้งใจแก้แต่แก้ไม่หมด) — W29 หันไปอ้างอิง "องค์ประกอบ" ของปุ่มแทน
    (role, has_icon, icon_hint, data_testid, is_nav_menu_item) ซึ่งคงที่ข้ามหน้าจริงสำหรับ
    ปุ่ม chrome/pagination ประเภทนี้ ตามที่ user ยืนยัน: "อ้างอิงจากองค์ประกอบปุ่ม แบบไม่สนใจ
    description" — แลกกับ false-positive ที่เป็นไปได้: ปุ่ม text ธรรมดาที่ไม่มี icon/
    data-testid เลยและข้อความบนปุ่มดันซ้ำกันโดยบังเอิญระหว่างปุ่มที่ทำหน้าที่ต่างกันจริง (เช่น
    "View" ในตาราง Products กับตาราง Orders) จะถูกมองว่าเป็นปุ่มเดียวกัน — ยอมรับ trade-off
    นี้เพราะเป้าหมายหลักคือกัน loop ไม่รู้จบก่อน"""
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
    """W33: ลายนิ้วมือของ "โครงสร้าง" หน้านี้ (ไม่ใช่เนื้อหา) — ใช้เทียบว่าหน้าใหม่ที่เพิ่ง
    เจอ "หน้าตาเหมือนหน้าที่เคยบันทึกไปแล้วไหม ต่างแค่รายละเอียด" (ตามที่ user ระบุ — ตัวอย่าง
    ที่เจอบ่อยคือหน้า Shorts/Reels: แต่ละคลิปเป็นคนละ URL จริง แต่ UI ทั้งหมด (ปุ่ม
    Like/Comment/Share/Next ฯลฯ) เหมือนกันเป๊ะทุกคลิป) — ต่างจาก _button_signature() ที่
    เทียบ "ปุ่มเดียว" ข้ามหน้า ตัวนี้เทียบ "ทั้งหน้า" โดยรวม signature ของทุกปุ่มบนหน้าเป็น
    set เดียว (ไม่สนลำดับ/ตำแหน่งเป๊ะ — แค่ "มีปุ่มหน้าตาแบบนี้ครบชุดไหม") บวกกับสัญญาณหยาบๆ
    ว่ามีฟอร์ม/ตารางอยู่ไหม (เฉพาะมี/ไม่มี ไม่สนจำนวนเป๊ะ กัน noise จาก UI pattern ที่
    item_count ไม่เท่ากันเพราะโหลดเนื้อหามาไม่พร้อมกัน)

    exact match เท่านั้น (frozenset ทั้งก้อนต้องเท่ากันเป๊ะ) — ยอมรับ false-negative ได้
    (หน้าที่ต่างกันจริงเล็กน้อย เช่น มีปุ่ม "Pinned Comment" เพิ่มมาบางคลิป จะไม่ถูกมองว่า
    "template เดียวกัน" ทั้งที่โดยรวมก็คล้ายกันมาก) ดีกว่าเสี่ยง false-positive (มองว่าเป็น
    หน้าเดียวกันทั้งที่จริงๆ ต่างกัน แล้วข้ามไปไม่บันทึกหน้าที่ควรบันทึก) — เป้าหมายหลักคือกัน
    การไล่บันทึก/สำรวจหน้าที่ซ้ำซ้อนกันจริงชัดๆ (เช่น feed ที่มีลิงก์ไปคลิปคนละอันนับสิบๆ
    รายการ) ไม่ใช่การจัดกลุ่มหน้าเว็บแบบสมบูรณ์แบบ"""
    button_sigs = frozenset(_button_signature(b) for b in page_info.buttons if b.selector)
    pattern_sigs = frozenset(f"pattern:{p.ui_type}:{p.selector}" for p in page_info.ui_patterns)
    coarse = frozenset({
        f"forms:{bool(page_info.forms)}",
        f"tables:{bool(page_info.tables)}",
    })
    return button_sigs | pattern_sigs | coarse


def _normalize_url(url: str) -> str:
    """ตัด fragment ออกกันนับซ้ำ (URL ต่างกันแค่ #section ไม่ควรถือว่าเป็นคนละหน้า) และ
    ตัด trailing slash ให้เหมือนกันเสมอ"""
    parsed = urllib.parse.urlparse(url)
    path = parsed.path.rstrip("/") or "/"
    return urllib.parse.urlunparse((parsed.scheme, parsed.netloc, path, "", parsed.query, ""))


# W24: ไล่กดปุ่มที่เพิ่งโผล่มาจาก modal/dropdown/accordion อีกแค่ 1 ชั้นเท่านั้น (ไม่
# recurse ไม่จำกัด) กันเปิด/ปิด dropdown ซ้อนกันไม่รู้จบ — ดู _explore_buttons()
_MAX_REVEAL_DEPTH = 1


async def _wait_for_dom_stable(
    page: Page, checks: int = 3, interval_ms: Optional[int] = None, max_iterations: int = 12,
) -> None:
    """W24: SPA (React/Vue/Next.js/Angular) ที่ navigate ด้วย client-side routing มักไม่
    ยิง network request ใหม่เลย (ข้อมูล prefetch/cache ไว้แล้ว) ทำให้
    wait_for_load_state("networkidle") อย่างเดียวผ่านเร็วเกินไปทั้งที่ DOM ยังเรนเดอร์ไม่
    เสร็จ — poll ความยาวของ document.body.innerHTML จนนิ่ง (เท่ากัน `checks` ครั้งติดกัน)
    หรือครบ max_iterations ก่อน ไม่ throw ออกไปเลย (best-effort — หน้าที่ evaluate ไม่ได้/
    ปิดไปแล้วก็แค่ข้าม ไม่ควรทำทั้ง crawl ล้มเพราะจุดนี้จุดเดียว)"""
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


async def _reveal_dynamic_content(page: Page) -> None:
    """W24: เลื่อนจอลงมาเรื่อยๆ จนความสูงของหน้า (scrollHeight) ไม่ขยับอีกแล้ว หรือครบ
    settings.site_learning_max_scroll_attempts ก่อน — เพื่อให้ extract_page() เห็นเนื้อหา
    ที่โหลดแบบ lazy/infinite-scroll (เช่น product grid ที่โหลดสินค้าเพิ่มตอน scroll ถึง
    ล่างสุด) ซึ่งเดิมไม่มีการ scroll เลยระหว่าง crawl เห็นแค่เนื้อหาที่โหลดมาตั้งแต่แรก —
    best-effort ล้วนๆ ไม่ throw ออกไปแม้ evaluate ล้มเหลว กลับขึ้นบนสุดก่อนจบเสมอ (เผื่อ
    fixed header/lazy image ที่ผูกกับ scroll position ตอน extract จริง)"""
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
    """W24: retry page.goto()+networkidle สูงสุด `retries` ครั้ง (รวมครั้งแรกทั้งหมด
    retries+1 ครั้ง) ก่อนยอมแพ้ — คืน None ถ้าสำเร็จ, คืนข้อความ error ตัวสุดท้ายถ้าล้มเหลว
    ครบทุกครั้ง (ไม่ throw ออกไปให้ caller เอง — แค่หน้าเดียวพังไม่ควรทำทั้ง crawl ล้มไปด้วย
    เหมือนพฤติกรรมเดิม แค่ตอนนี้ retry ก่อนค่อยยอมแพ้ + บอกเหตุผลที่แท้จริงกลับไปแทนที่จะ
    เงียบข้ามไปเฉยๆ)"""
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


async def _click_with_retry(page: Page, selector: str, retries: int) -> Optional[str]:
    """W24: เหมือน _goto_with_retry() แค่สำหรับ page.click() ระหว่างไล่สำรวจปุ่ม — คืน
    None ถ้าสำเร็จ, ข้อความ error ตัวสุดท้ายถ้าล้มเหลวครบทุกครั้ง"""
    last_error = ""
    for attempt in range(retries + 1):
        try:
            await page.click(selector, timeout=5000)
            return None
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
            if attempt < retries:
                await page.wait_for_timeout(settings.site_learning_retry_backoff_ms)
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
    # W24: เก็บ error ที่เกิดจริง (goto/click ที่ retry ครบแล้วยังล้ม, login ที่ดูเหมือน
    # ไม่ผ่าน) ติดไปกับ SiteManual แทนที่จะกลืนเงียบๆ แล้วสรุปว่า "เรียนรู้เสร็จ" (ดู
    # docstring หัวไฟล์ ข้อ 6)
    manual_errors: list[dict] = []
    try:
        visited: set[str] = set()
        queue: list[str] = [start_url]
        queued: set[str] = {_normalize_url(start_url)}
        estimated_total = 1
        login_attempted = False
        # W28: กันไล่กด "ปุ่มเดิม" (label+role เดียวกัน) ซ้ำไม่รู้จบข้าม URL ที่ต่างกัน — ดู
        # docstring หัวไฟล์ ข้อ W28 และ _button_signature ด้านบน
        explored_button_signatures: dict[str, int] = {}
        # W33: template ของหน้าที่เคยบันทึกไปแล้ว (ดู _page_template()) — หน้าใหม่ที่โครงสร้าง
        # ตรงกับ template ที่เคยเจอ (เช่น คลิป Shorts/Reels อื่นที่ UI เหมือนกันทุกอย่าง ต่าง
        # แค่เนื้อหา) จะถูกข้ามไม่บันทึก/ไม่ไล่กดปุ่มต่อ (ดู _record_page())
        known_page_templates: set[frozenset[str]] = set()

        def _is_explorable(button) -> bool:
            """W24: เมนู/nav item (is_nav_menu_item — ดู extractor.py::isNavMenuItem)
            ตัดสินใจแบบ default-allow เหมือน safety.is_safe_nav_link() ที่ใช้กับ <a> nav
            link ปกติ (บล็อกเฉพาะคำที่ชัดเจนว่าทำลาย/เปลี่ยนแปลงข้อมูล เช่น "Logout") —
            ปุ่มอื่นๆ ทั้งหมดยังต้องผ่าน safety.is_crawl_safe() แบบ default-deny เข้มเหมือน
            เดิมทุกประการ (ไม่ลดความเข้มงวดของ Safety Rule เดิมลงเลย แค่ให้เมนูที่ไม่ใช่
            <a> ได้สิทธิ์เดียวกับเมนูที่เป็น <a>)"""
            label = _button_label(button)
            if getattr(button, "is_nav_menu_item", False) and is_safe_nav_link(label):
                return True
            return is_crawl_safe(label, cmd_type="click")

        async def _record_page(
            page_info: PageInfo, nav_links: list[dict], check_page_template: bool = True,
        ) -> None:
            """describe + เก็บเข้า pages + ยิง progress event + ไล่กดปุ่มปลอดภัย + ต่อคิว
            nav link ที่ปลอดภัย — logic ร่วมที่ใช้ทั้งกับหน้าที่เจอจาก BFS ปกติ, หน้าหลัง
            login bootstrap, และหน้าที่เจอจากการไล่กดปุ่ม (ดู docstring หัวไฟล์) ไล่กดปุ่ม
            ของทุกหน้าที่บันทึกเสมอ ไม่ว่าจะเจอหน้านั้นจากทางไหน (nav link/ปุ่ม/login) —
            ไม่มี depth cap แล้ว (W16) การันตีว่าจบได้จริงด้วย visited-set +
            effective_max_pages เท่านั้น

            W33: หน้าที่โครงสร้างตรงกับ template ที่เคยบันทึกไปแล้ว (ดู _page_template()) —
            เช่น เจอ feed ที่มีลิงก์ไปคลิป Shorts/Reels คนละอันสิบๆ รายการ แต่ละคลิปเป็น URL
            จริงต่างกัน (ไม่ถูกกันโดย visited-set) แต่ UI เหมือนกันทุกอย่าง — บันทึกแค่ตัวแทน
            ตัวแรกที่เจอ (ถือว่า "กดเข้าดูแล้ว 1 รอบ") ตัวถัดๆ ไปข้ามไปเลย ไม่ describe (ประหยัด
            LLM call) ไม่ไล่กดปุ่มต่อ (ผู้เรียก — ทั้ง main BFS loop และ _explore_buttons — จะ
            ไปทำ item ถัดไปเองตามปกติอยู่แล้ว ไม่ต้องมี logic "กด back" แยกต่างหากตรงนี้เลย
            เพราะ _explore_buttons() go_back()/goto(before_url) เสมอหลัง _record_page() คืน
            ไม่ว่าจะทำอะไรข้างในก็ตาม)

            *** เจอ false-positive จริงระหว่างทดสอบ 2 จุด แก้แล้ว: (1) หน้าที่ไม่มีปุ่ม/
            ui_pattern เลยสักตัว (เช่นหน้าเปล่าๆ ที่มีแค่ข้อความ) ได้ template ที่ "เหมือนกัน
            โดยบังเอิญ" กับหน้าเปล่าอื่นๆ ทั้งที่เนื้อหา/ความหมายต่างกันจริง (เช่น "Dashboard"
            vs "Article Detail") — ไม่ใช่เคส Shorts/Reels ที่ UI ซับซ้อนพอจะเป็นสัญญาณจริง —
            ต้องมีอย่างน้อย 1 ปุ่มหรือ 1 ui_pattern ถึงจะเอามาเทียบ dedup เลย (ดู has_signal)
            (2) เส้นทาง _explore_buttons() (DFS-click) มีกลไก signature-cap ของตัวเองอยู่แล้ว
            (W28/W29 — settings.site_learning_max_repeat_button_clicks ปรับได้) ถ้าให้
            page-template dedup ทำงานที่นั่นด้วยจะไปทับ/ขัดกับเพดานที่ user ปรับตั้งใจไว้ (เช่น
            ตั้งเพดานไว้ 2 ครั้ง แต่ template dedup ตัดจบไปตั้งแต่ครั้งที่ 1) — พารามิเตอร์
            check_page_template=False ให้ _explore_buttons() ปิด layer นี้เฉพาะเส้นทางของ
            ตัวเอง (ยังคงบันทึก template นี้ไว้ให้เส้นทาง BFS ใช้เทียบต่อได้ปกติ แค่ไม่ตัดสินใจ
            ข้ามด้วยตัวเอง) ***"""
            nonlocal estimated_total
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

        async def _explore_buttons(base_page_info: PageInfo, depth: int = 0) -> None:
            """ไล่กดปุ่มที่ _is_explorable() อนุญาตทีละปุ่มบนหน้านี้ (base_page_info.url) —
            กดแล้วเช็คว่า URL เปลี่ยนไหม: เปลี่ยน = เจอหน้าใหม่ (บันทึกผ่าน _record_page ถ้า
            ยังไม่เคยเจอ — ซึ่งจะไล่กดปุ่มของหน้าใหม่นั้นต่อทันที เป็น DFS แบบไม่จำกัดความลึก
            — แล้ว go_back()/goto() กลับมาหน้าตั้งต้นก่อนลองปุ่มถัดไปเสมอ, ถ้าเคยเจอแล้ว
            (visited/queued) ก็แค่กลับมาเฉยๆ ไม่สำรวจซ้ำ นี่คือจุดที่ทำให้ DFS "ตัน" แล้ว
            ถอยกลับไปลองปุ่มอื่นของหน้าก่อนหน้าโดยธรรมชาติ) ไม่เปลี่ยน = แค่ modal/panel/tab/
            dropdown/accordion เปิดในหน้าเดิม (re-extract แล้ว merge เข้า base_page_info
            ผ่าน _merge_page_info แทนที่จะสร้างหน้าใหม่ — W24: แล้วไล่กด element ที่ "เพิ่ง
            โผล่มาจริง" ต่ออีก 1 ชั้น ดู depth/_MAX_REVEAL_DEPTH ด้านล่าง — เพราะ element ที่
            ถูกซ่อนด้วย display:none ตอน extract ครั้งแรกไม่เคยถูกมองเห็น/กดเลยมาก่อน + กด
            Escape ปิดแบบ best-effort ก่อนไปปุ่มถัดไป) ไม่ throw ออกไปแม้ปุ่มไหนกด/กลับไม่ได้
            — ข้ามไปปุ่มถัดไปเงียบๆ เว้นแต่ "กลับหน้าตั้งต้นไม่ได้เลย" ซึ่งเลิกไล่ปุ่มที่
            เหลือของหน้านี้ทันที (state ของหน้าพังไปแล้ว ไล่ต่อไม่มีประโยชน์)

            W24: click/goto ที่ล้มเหลวตอนนี้ retry ก่อน (settings.site_learning_
            click_retries) แล้วค่อยบันทึกลง manual_errors + ยิง "button_click_failed"
            แทนการกลืนเงียบๆ เหมือนเดิม — depth=0 คือปุ่มระดับหน้าโดยตรง, depth=1 คือปุ่มที่
            เพิ่งโผล่มาจาก modal/dropdown/accordion (ไม่ recurse ลึกกว่านี้)"""
            before_url = _normalize_url(page.url)
            # W18: รวมปุ่มระดับหน้า + ปุ่ม "ตัวแทน" ของ UI pattern แต่ละแบบ (ดู
            # extractor.py::UIPatternInfo — เช่น ปุ่ม "View" ใน product card ที่ซ้ำกัน 100
            # ใบ ตอนนี้เหลือ instance เดียวให้ลองกด) เข้าลิสต์เดียวกัน กรองด้วย
            # _is_explorable() (W24: เมนู/nav item default-allow, ปุ่มอื่น default-deny
            # เหมือนเดิม — ดู docstring ของ _is_explorable) — label ใช้ text > aria_label >
            # title > icon_hint (เผื่อเป็นปุ่ม icon-only ที่ไม่มี text/aria-label/title เลย)
            candidate_buttons = list(base_page_info.buttons)
            for pattern in base_page_info.ui_patterns:
                candidate_buttons.extend(pattern.buttons)
            safe_buttons = [
                b for b in candidate_buttons if b.selector and _is_explorable(b)
            ][:settings.site_learning_max_buttons_per_page]

            for button in safe_buttons:
                if len(pages) >= effective_max_pages:
                    break
                # W28: ปุ่มนี้ (label+role เดียวกัน) ถูกไล่กดไปแล้วครบเพดานหรือยัง (นับรวม
                # ทั้ง crawl ข้าม URL ไม่ใช่แค่หน้านี้) — ถ้าครบแล้วข้ามไปเลย กัน DFS ไล่ตาม
                # ปุ่ม pagination/chrome ที่โผล่ซ้ำทุกหน้า (เช่น "Next video") ไม่รู้จบ
                signature = _button_signature(button)
                click_count = explored_button_signatures.get(signature, 0)
                if click_count >= settings.site_learning_max_repeat_button_clicks:
                    continue
                explored_button_signatures[signature] = click_count + 1
                label = _button_label(button)
                if on_progress:
                    await on_progress({"kind": "button_explored", "url": before_url, "button": label})

                click_error = await _click_with_retry(page, button.selector, settings.site_learning_click_retries)
                if click_error is not None:
                    manual_errors.append({"url": before_url, "phase": "click", "button": label, "error": click_error})
                    if on_progress:
                        await on_progress({
                            "kind": "button_click_failed", "url": before_url, "button": label, "error": click_error,
                        })
                    continue  # กดปุ่มนี้ไม่ได้แม้ retry ครบแล้ว (element หาย/ถูกบัง/detach ฯลฯ) ข้ามไปปุ่มถัดไป

                # ปุ่มบาง element เป็น target="_blank" เปิดแท็บใหม่แทนที่จะ navigate หน้า
                # เดิม — ปิดแท็บที่เพิ่งเปิดทิ้งทันที (ไม่ตามไปสำรวจ) กัน context สะสม page
                # ค้างเป็นสิบๆ ตัวถ้าไล่กดหลายร้อยปุ่มตลอด crawl
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
                if after_url != before_url and extract_domain(page.url) == domain:
                    if after_url not in visited and after_url not in queued:
                        visited.add(after_url)
                        await _reveal_dynamic_content(page)
                        new_page_info, new_nav_links = await extract_page(page)
                        # W33: ปิด page-template dedup เฉพาะเส้นทางนี้ (DFS-click) — มี
                        # explored_button_signatures (W28/W29) เป็นกลไก dedup ของตัวเอง
                        # อยู่แล้วที่ปรับเพดานได้ผ่าน settings ไม่อยากให้ template dedup มา
                        # ทับ/ขัดเพดานที่ user ตั้งใจไว้ (ดู docstring ของ _record_page())
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
                else:
                    # ไม่ navigate ไปไหน — น่าจะเป็น modal/expand panel/tab/dropdown/accordion
                    # ที่เปิดในหน้าเดิม re-extract แล้ว merge โครงสร้างใหม่เข้าไปในหน้านี้
                    # (ไม่ใช่หน้าใหม่จริง)
                    known_selectors_before = {b.selector for b in base_page_info.buttons if b.selector}
                    try:
                        revealed_info, _ = await extract_page(page)
                        _merge_page_info(base_page_info, revealed_info)
                        # W24: element ที่ "เพิ่งโผล่มาจริง" (ไม่เคยอยู่ใน base_page_info มา
                        # ก่อนเลย — ถูกซ่อนด้วย display:none ตอน extract ครั้งแรก มองไม่
                        # เห็น/กดไม่ได้เลยมาก่อน) ไล่กดต่ออีกแค่ 1 ชั้น (depth < _MAX_
                        # REVEAL_DEPTH) กันเปิด/ปิด dropdown ซ้อนกันไม่รู้จบ — ใช้ before_url
                        # เดียวกัน (หน้ายังไม่ navigate ไปไหนเลย) รีเคิร์สผ่าน PageInfo ปลอม
                        # ที่มีแค่ปุ่มใหม่พวกนี้ ไม่ต้องเขียน loop ซ้ำ
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
            """บันทึกหน้า login เอง (มีประโยชน์ต่อ manual) แล้วลอง attempt_login() แล้ว
            ตรวจสอบว่า session ใช้ได้จริงหรือไม่ (W24 — ดู docstring หัวไฟล์ ข้อ 8) ก่อนค่อย
            extract+บันทึกหน้าถัดจาก login ต่อ ใช้ร่วมกันทั้ง 2 เส้นทางที่มี username/
            password มาใช้ได้ (ส่งมาตั้งแต่ต้น crawl กับได้จาก on_credentials_needed
            ระหว่างทาง — ดู docstring ของ crawl_site())

            W24 การตรวจ session: attempt_login() คืนแค่ "กด submit ได้จริงไหม" ไม่รู้ว่า
            login ผ่านจริง — เพิ่มเช็ค 2 ชั้นที่ตัดสิน "session_ok" จริงๆ (ทั้งคู่ต้องผ่าน):
            (1) URL เปลี่ยนไปจาก URL ก่อน submit จริง (ไม่ใช่แค่ submit แล้ว reload หน้าเดิม)
            (2) หน้าใหม่ไม่มีฟอร์ม login เหลืออยู่แล้ว (find_login_fields คืน (None, None)
            — ถ้ายังเจอ = โดน redirect กลับมาหน้า login เดิม ถือว่า login ไม่ผ่าน)
            ส่วนจำนวน cookie / มี localStorage-sessionStorage token ไหม เป็นแค่สัญญาณ
            informational แนบไปกับ event "login_result" เท่านั้น *** ไม่ใช้ตัดสิน pass/fail
            เพราะเว็บจำนวนมากใช้ token-based auth ไม่มี cookie เลย (fixture ทดสอบในโปรเจกต์
            นี้เองก็ไม่มี server จริงตั้ง cookie ให้ — ถ้าเอา cookie เป็นเงื่อนไขบังคับจะทำให้
            false-negative ทุกเว็บที่ไม่ใช้ cookie ทันที) ***"""
            await _record_page(page_info, nav_links)
            pre_login_url = _normalize_url(page.url)
            did_login = await attempt_login(page, page_info, login_username, login_password)

            session_ok = False
            reason = ""
            post_page_info: Optional[PageInfo] = None
            post_nav_links: list[dict] = []
            if not did_login:
                reason = "กรอกฟอร์ม/กดปุ่ม submit ไม่สำเร็จ (หา field/ปุ่มไม่ครบ หรือ fill/click ล้มเหลว)"
            else:
                post_login_url = _normalize_url(page.url)
                if post_login_url == pre_login_url:
                    reason = "URL ไม่เปลี่ยนหลัง submit — เข้าใจว่า login ไม่ผ่าน"
                else:
                    await _reveal_dynamic_content(page)
                    await _wait_for_dom_stable(page)
                    post_page_info, post_nav_links = await extract_page(page)
                    if find_login_fields(post_page_info) != (None, None):
                        reason = "ยังเจอฟอร์ม login (username+password field) อยู่หลัง submit — เข้าใจว่าถูก redirect กลับหน้า login"
                    else:
                        session_ok = True

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
            url = queue.pop(0)
            normalized = _normalize_url(url)
            if normalized in visited:
                continue
            visited.add(normalized)

            # W24: retry ก่อนยอมแพ้ (settings.site_learning_goto_retries) — เดิมล้มครั้งเดียว
            # = ข้ามทันที ไม่แยกว่าเป็น transient failure (network กระตุก/DOM ยังไม่นิ่ง) หรือ
            # พังจริง (404/DNS ผิด) บันทึก error จริงลง manual_errors + ยิง event แทนกลืนเงียบๆ
            goto_error = await _goto_with_retry(page, url, settings.site_learning_goto_retries)
            if goto_error is not None:
                manual_errors.append({"url": url, "phase": "goto", "error": goto_error})
                if on_progress:
                    await on_progress({"kind": "page_error", "url": url, "phase": "goto", "error": goto_error})
                continue  # หน้านี้ไปไม่ถึงแม้ retry ครบแล้ว (404/timeout/DNS ฯลฯ) — ข้ามไปหน้าถัดไป ไม่ล้มทั้ง crawl

            # W16: ยิงก่อน extract/describe (ซึ่งกินเวลาจาก LLM call) เพื่อให้ UI โชว์ "กำลัง
            # เรียนรู้หน้านี้อยู่" ได้ทันทีที่หน้าโหลดเสร็จ ไม่ต้องรอ page_done — คู่กับ
            # browser ที่เปิดแบบมองเห็นได้ (headless=False, ดู routes.py::learn_site()) ให้
            # user เห็นจริงๆ ว่ากำลังเดินอยู่หน้าไหน
            await page.bring_to_front()
            if on_progress:
                await on_progress({"kind": "page_start", "url": page.url})

            # W24: เผยเนื้อหา lazy/infinite-scroll ก่อน แล้วรอ DOM นิ่ง (SPA client routing)
            # ก่อนสกัดโครงสร้างจริง — ดู docstring หัวไฟล์ ข้อ 3, 11
            await _reveal_dynamic_content(page)
            await _wait_for_dom_stable(page)
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

    # W26: สรุปภาพรวม "เว็บไซต์นี้ทำอะไรได้บ้าง" ครั้งเดียวหลังสำรวจครบทุกหน้าแล้ว (ดู
    # describe_site() — ใช้ client/model เดียวกับที่ describe_page() ใช้ต่อหน้าอยู่แล้ว)
    site_summary = await describe_site(client, model, resolved_provider, domain, pages)
    manual = SiteManual(
        website=domain, pages=pages, generated_at=time.time(), errors=manual_errors, summary=site_summary,
    )
    if on_progress:
        # W24: errors_found เพิ่มเข้ามา — สรุปว่า crawl "จบเพราะสำรวจครบจริง" หรือ "จบทั้งที่
        # เจอปัญหาระหว่างทาง" ไม่ใช่แค่ pages_found เฉยๆ (ดู docstring หัวไฟล์ ข้อ 9)
        await on_progress({
            "kind": "crawl_scan_done", "pages_found": len(pages), "errors_found": len(manual_errors),
        })
    return manual

"""site_learning/storage.py — W14: อ่าน/เขียน manual ที่ crawler.py สร้างลงดิสก์ — เก็บ
เป็นไฟล์ JSON ล้วนๆ ใต้ settings.site_manuals_dir แยกโฟลเดอร์ต่อโดเมน ไม่มี ChromaDB/
embedding เกี่ยวข้องเลย (คนละระบบกับ backend/app/rag/ ที่เก็บคู่มือที่ user อัปโหลดเอง —
ตั้งใจไม่ใช้ path "data/manuals" เดิมเพราะชื่อนั้นถูก chroma_collection_name="manuals"
จับจองความหมายไว้แล้ว)

โครงสร้างโฟลเดอร์ต่อโดเมน (settings.site_manuals_dir/{domain}/):
    latest.json    — version ล่าสุดเสมอ, ตัวที่ orchestrator โหลดไปใช้จริง — save_manual()
                     เขียนทับไฟล์นี้ตรงๆ ทุกครั้ง ไม่เก็บไฟล์ประวัติแยกต่อเวอร์ชัน (vN.json)
                     อีกต่อไป (ตามที่ user ขอ — กันไฟล์สะสมไม่รู้จบบนดิสก์ที่ commit เข้า
                     git) manual.version ยังนับเพิ่มไว้เป็น metadata ปกติ แค่ไม่มีไฟล์แยก
    ui-map.json    — tree โครงสร้างเมนู (derive จาก menu_path ของทุกหน้า)
    selectors.json — flat lookup {"หน้า > ปุ่ม": {css, xpath, aria, data_testid}}
    knowledge.json — {page_name: description} ฉบับย่อ ไว้ยัด prompt ถูกๆ
"""

import json
import re
import time
from pathlib import Path
from typing import Optional

from cryptography.fernet import Fernet, InvalidToken

from backend.app.config import settings
from backend.app.site_learning.schema import PageInfo, SiteManual


def _domain_dir(domain: str) -> Path:
    return Path(settings.site_manuals_dir) / domain


def manual_exists(domain: str) -> bool:
    return (_domain_dir(domain) / "latest.json").exists()


def load_manual(domain: str) -> Optional[SiteManual]:
    path = _domain_dir(domain) / "latest.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return SiteManual.from_dict(data)


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_ui_map(manual: SiteManual) -> dict:
    """แปลง menu_path ของทุกหน้าเป็น tree เดียว —
    {label: {"children": {...}, "page": page_name|None}}"""
    root: dict = {}
    for page in manual.pages:
        node = root
        path = page.menu_path or ([page.name] if page.name else [])
        for i, label in enumerate(path):
            node = node.setdefault(label, {"children": {}, "page": None})
            if i == len(path) - 1:
                node["page"] = page.name
            node = node["children"]
    return root


def _build_selectors(manual: SiteManual) -> dict:
    """flat lookup: "{page_name} > {button_text}" -> {css, xpath, aria, data_testid} —
    ไว้ให้ selector-repair (update_single_page) หรือ debug tooling ค้นหาเร็วๆ โดยไม่ต้อง
    ไล่ทั้ง manual

    W18: รวม selector ของ UI pattern ด้วย (key รูปแบบ "{page_name} > [{pattern_name}]
    {button_label}") — pattern.selector คือ selector ที่ใช้ซ้ำได้กับทุก instance ของ
    pattern นั้น (ไม่ใช่แค่ instance ตัวแทนที่ extract มา) ส่วน button.selector ภายในยังชี้
    ไปที่ปุ่มของ instance แรกเท่านั้น — ผู้ใช้ที่อยากกดปุ่มแบบนี้ใน instance อื่นต้องใช้
    pattern.selector หา container แล้ว query ปุ่มที่เข้าข่ายภายในเอง"""
    out = {}
    for page in manual.pages:
        for b in page.buttons:
            label = b.text or b.aria_label or b.icon_hint
            if not label:
                continue
            key = f"{page.name} > {label}"
            out[key] = {"css": b.selector, "xpath": b.xpath, "aria": b.aria_label, "data_testid": b.data_testid}
        for pattern in page.ui_patterns:
            out[f"{page.name} > [{pattern.name}]"] = {
                "css": pattern.selector, "xpath": "", "aria": "", "data_testid": "",
            }
            for b in pattern.buttons:
                label = b.text or b.aria_label or b.icon_hint
                if not label:
                    continue
                key = f"{page.name} > [{pattern.name}] {label}"
                out[key] = {"css": b.selector, "xpath": b.xpath, "aria": b.aria_label, "data_testid": b.data_testid}
    return out


def _build_knowledge(manual: SiteManual) -> dict:
    return {p.name: p.description for p in manual.pages if p.name}


def save_manual(manual: SiteManual) -> int:
    """บันทึก manual ใหม่ทั้งก้อน ทับ latest.json ตัวเดิมตรงๆ ไม่เก็บไฟล์ประวัติ vN.json
    แยกต่างหากอีกต่อไป (เดิมเขียน v{N}.json ทุกครั้งที่ save ไม่เคยลบ — ไฟล์สะสมไม่รู้จบ
    บนดิสก์ที่ commit เข้า git ตามที่ user ขอให้เปลี่ยน) bump version ต่อจาก version
    ล่าสุดที่มีอยู่จริงบนดิสก์เสมอ (ไม่ใช่แค่ manual.version ที่ caller ส่งมา กันลืม
    อัปเดต) — ยังคงเลขเวอร์ชันไว้เป็น metadata (ใช้แสดงผล/comparison เท่านั้น ไม่มีไฟล์
    ต่อเวอร์ชันให้ย้อนดูอีกแล้ว) คืน version number ใหม่"""
    domain_dir = _domain_dir(manual.website)
    existing = load_manual(manual.website)
    new_version = (existing.version + 1) if existing else 1
    manual.version = new_version
    manual.generated_at = time.time()

    data = manual.to_dict()
    _write_json(domain_dir / "latest.json", data)
    _write_json(domain_dir / "ui-map.json", _build_ui_map(manual))
    _write_json(domain_dir / "selectors.json", _build_selectors(manual))
    _write_json(domain_dir / "knowledge.json", _build_knowledge(manual))
    return new_version


def update_single_page(domain: str, page_info: PageInfo) -> Optional[int]:
    """selector-repair path (สเปค: "หาก Selector ใช้งานไม่ได้ ให้สำรวจเฉพาะหน้านั้น
    อัปเดต Version ไม่ต้องสร้าง Manual ใหม่ทั้งหมด") — แทนที่หน้าเดียว (จับคู่ด้วย url)
    ใน manual ที่มีอยู่แล้ว แล้ว re-derive ui-map/selectors/knowledge + bump version คืน
    None ถ้าโดเมนนี้ยังไม่มี manual เลย (ต้อง crawl เต็มรูปแบบก่อนครั้งแรกเสมอ)"""
    manual = load_manual(domain)
    if manual is None:
        return None
    for i, p in enumerate(manual.pages):
        if p.url == page_info.url:
            manual.pages[i] = page_info
            break
    else:
        manual.pages.append(page_info)
    return save_manual(manual)


def _credentials_path(domain: str) -> Path:
    return _domain_dir(domain) / "credentials.json"


def _credential_key_path() -> Path:
    # Security 1.4: key ต่อเครื่อง เก็บที่ระดับ "data/" เดียว (parent ของ site_manuals_dir)
    # ไม่ใช่ต่อโดเมน — credentials.json ของทุกโดเมนใช้ key เดียวกัน
    return Path(settings.site_manuals_dir).parent / ".credential_key"


def _get_fernet() -> Fernet:
    """Security 1.4: เข้ารหัส/ถอดรหัส credentials.json ด้วย key แบบ local-machine — generate
    ครั้งแรกที่ต้องใช้แล้วเก็บไว้ที่ data/.credential_key (gitignored) ไม่ต้องให้ user ตั้ง
    env var เพิ่มเอง ไฟล์นี้เป็น secret ต่อเครื่อง — ย้ายเครื่อง/ลบไฟล์นี้ทิ้งแล้ว
    credentials.json เก่าที่เข้ารหัสไว้แล้วจะถอดรหัสไม่ได้อีก (ต้องกรอก credential ใหม่)"""
    path = _credential_key_path()
    if path.exists():
        key = path.read_bytes()
    else:
        key = Fernet.generate_key()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(key)
    return Fernet(key)


def save_credentials(domain: str, username: str, password: str) -> None:
    """W17: เก็บ username/password สำหรับโดเมนนี้ไว้ให้ orchestrator ดึงไปใช้ auto-login
    ตอนรัน task จริง (ดู core/orchestrator.py::_maybe_auto_login, site_learning/
    auto_login.py) — เขียนคนละไฟล์ (credentials.json) แยกจาก latest.json ของ manual โดย
    เจตนา ไม่ปนกับ manual ที่ save_manual() เขียนทับ (กันหลุดปนไปด้วยความไม่ตั้งใจถ้ามีคน
    แก้ save_manual()/schema ในอนาคต) — ไฟล์นี้เขียนทับตัวเดิมเสมอ ไม่มีประวัติเวอร์ชัน

    Security 1.4: username/password เข้ารหัสด้วย Fernet ก่อนเขียนลงดิสก์เสมอ (marker
    "encrypted": true ให้ load_credentials() แยกจากไฟล์เก่าที่ยังเป็น plaintext ได้)"""
    fernet = _get_fernet()
    encrypted = {
        "encrypted": True,
        "username": fernet.encrypt(username.encode("utf-8")).decode("ascii"),
        "password": fernet.encrypt(password.encode("utf-8")).decode("ascii"),
    }
    _write_json(_credentials_path(domain), encrypted)


def load_credentials(domain: str) -> Optional[dict]:
    """คืน {"username":..., "password":...} หรือ None ถ้ายังไม่เคยเก็บไว้/อ่านไม่ได้ (ไม่
    throw — โดเมนที่ไม่มี credential เก็บไว้เป็นเรื่องปกติ ไม่ใช่ error)

    Security 1.4: ไฟล์ที่มี marker "encrypted": true ถอดรหัสก่อนคืนค่า — ไฟล์เก่าที่ยังเป็น
    plaintext (ไม่มี marker นี้เลย จาก storage.py เวอร์ชันก่อนหน้า) อ่านตรงๆ แบบเดิมเพื่อ
    backward-compat แล้ว re-save แบบเข้ารหัสทันที (migration แบบเนียน ไม่ต้องมี script
    แยกต่างหาก ไม่ต้องให้ user ทำอะไรเอง)"""
    path = _credentials_path(domain)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    username = data.get("username")
    password = data.get("password")
    if not username or not password:
        return None
    if not data.get("encrypted"):
        # legacy plaintext — migrate เงียบๆ (ไม่ throw ถ้า migrate ไม่สำเร็จ เพราะยังคืนค่า
        # ที่อ่านได้แล้วอยู่ดี ไม่ควรทำให้ caller เห็น error จากการ migrate ที่ไม่ใช่ requirement)
        try:
            save_credentials(domain, username, password)
        except OSError:
            pass
        return {"username": username, "password": password}
    try:
        fernet = _get_fernet()
        username = fernet.decrypt(username.encode("ascii")).decode("utf-8")
        password = fernet.decrypt(password.encode("ascii")).decode("utf-8")
    except (InvalidToken, ValueError):
        return None
    return {"username": username, "password": password}


def credentials_exist(domain: str) -> bool:
    return _credentials_path(domain).exists()


def delete_credentials(domain: str) -> bool:
    path = _credentials_path(domain)
    if not path.exists():
        return False
    path.unlink()
    return True


def find_matching_page(manual: SiteManual, goal: str) -> Optional[PageInfo]:
    """W21 ("Self-Learned Site Manual Integration"): หาแค่ "หน้าเดียวที่น่าจะตรงกับ goal
    ที่สุด" จาก manual ที่ crawl มาแล้ว ให้ routes.py ใช้ตัดสินใจว่าจะฉีด Strict Guided Plan
    context (ดู llm.py::build_strict_manual_context) หรือไม่ — matching แบบ keyword overlap
    ล้วนๆ (นับจำนวนคำใน goal ที่ยาว >= 3 ตัวอักษรที่ปรากฏใน name/description/breadcrumb ของ
    แต่ละหน้า) ไม่ใช้ embedding/ChromaDB เลย เพราะ manual นี้เป็น JSON แบนบนดิสก์อยู่แล้ว (ดู
    docstring หัวไฟล์) มีจำนวนหน้าต่อเว็บไซต์น้อยพอที่ keyword scoring ตรงไปตรงมาก็เพียงพอ ไม่
    คุ้มเพิ่ม dependency ใหม่ — คืน None ถ้าไม่มีหน้าไหนได้คะแนนเลย (ไม่มีคำไหนตรงกันแม้แต่คำ
    เดียว) ให้ caller fallback ไป dynamic planner ตามปกติ"""
    goal_tokens = {t for t in re.split(r"[^\w]+", (goal or "").lower()) if len(t) >= 3}
    if not goal_tokens:
        return None

    best_page: Optional[PageInfo] = None
    best_score = 0
    for page in manual.pages:
        haystack = " ".join([page.name, page.description, " ".join(page.breadcrumb), " ".join(page.menu_path)]).lower()
        score = sum(1 for t in goal_tokens if t in haystack)
        if score > best_score:
            best_score = score
            best_page = page
    return best_page


def _page_flow_steps(page: PageInfo) -> list[str]:
    """W21: ลำดับหน้า/เมนูที่ต้องผ่านเพื่อไปถึง page นี้ — breadcrumb (ลำดับที่ crawler
    เห็นจริงตอนสำรวจ เช่น "Home > Admin > User Management") น่าเชื่อถือกว่า menu_path
    (แค่ตำแหน่งในเมนู sidebar เฉยๆ ไม่รับประกันว่าตรงกับลำดับ navigation จริง) ให้ breadcrumb
    ชนะถ้ามี ตกไป menu_path ถ้าไม่มี breadcrumb เลย ตกไปแค่ชื่อหน้าเดี่ยวๆ ถ้าไม่มีทั้งคู่"""
    return list(page.breadcrumb) or list(page.menu_path) or ([page.name] if page.name else [])


def build_learned_page_flow_text(page: PageInfo) -> str:
    """W21 ("Self-Learned Site Manual Integration" ข้อ 1, Manual Lookup & Context
    Injection): ประกอบ block "📍 Learned Page Flow Sequence" ตามฟอร์แมตที่สเปคกำหนดตายตัว
    (Task/W21.txt Task5) — ใช้ทั้งใน /context (llm.py::context_inspection_reply) และแปะ
    ไว้เป็นส่วนหัวของ build_strict_manual_context() ด้านล่าง เขียนแยกจาก build_strict_
    manual_context เพราะ /context ต้องการแค่ block นี้เฉยๆ (โหมด inspection ไม่ลงมือทำจริง)
    ในขณะที่ planner ต้องการรายละเอียด selector เพิ่มเติมด้วย"""
    steps = _page_flow_steps(page)
    flow = " ➔ ".join(f"[{s}]" for s in steps) if steps else f"[{page.name or 'หน้าเป้าหมาย'}]"
    return f"📍 **Learned Page Flow Sequence:**\n`{flow}`"


def build_strict_manual_context(page: PageInfo) -> str:
    """W21 ("Self-Learned Site Manual Integration" ข้อ 2, Strict Guided Planner
    Generation): ประกอบข้อความที่ฉีดเข้า site_manual_context (ช่องทางเดียวกับที่
    load_knowledge_text() ใช้อยู่แล้ว — ดู orchestrator.py::generate_plan()) แต่ขึ้นต้นด้วย
    marker "[PRE_LEARNED_MANUAL]" ตรงตามที่ SYSTEM_PROMPT (llm.py, กติกา W21 "PRE_LEARNED_
    MANUAL Strict Mode") ตรวจหา — ต่างจาก load_knowledge_text() เดิม (แค่ name: description
    สั้นๆ ของทุกหน้า ใช้เป็นข้อมูลอ้างอิงกว้างๆ) block นี้ scope แคบลงเหลือ "หน้าเดียวที่
    ตรงกับ goal" พร้อมรายละเอียด route/ปุ่ม/selector ที่บันทึกไว้จริงจาก crawl ให้ planner
    ยึดเป็นหลักแทนการเดา — selector/xpath ที่แนบมาเป็นข้อมูลอ้างอิงให้ LLM ใช้ตัดสินใจว่า
    element ไหนใน indexed elements ตรงกับที่คู่มือพูดถึง (สถาปัตยกรรมนี้ยังคง index-based
    เดิมทั้งหมด ไม่มีการยิง selector ตรงๆ ข้าม perception layer)"""
    flow_block = build_learned_page_flow_text(page)
    lines = [
        "[PRE_LEARNED_MANUAL]",
        flow_block,
        f"Target Page: {page.name or '(ไม่ทราบชื่อ)'} — {page.url or '(ไม่ทราบ URL)'}",
    ]
    if page.description:
        lines.append(f"Description: {page.description}")
    if page.buttons:
        lines.append("Recorded buttons on this page (label — selector):")
        for b in page.buttons[:20]:
            label = b.text or b.aria_label or b.title or b.icon_hint or "(ไม่มี label)"
            selector_hint = b.selector or b.xpath or "(ไม่มี selector บันทึกไว้)"
            lines.append(f"  - {label} — {selector_hint}")
    return "\n".join(lines)


def load_knowledge_text(domain: str) -> str:
    """ข้อความสั้นๆ (page_name: description ต่อบรรทัด) ไว้ฉีดเข้า prompt ตรงๆ ผ่าน
    site_manual_context (ดู llm.py::_build_user_turn_text) — คืนสตริงว่างเปล่าถ้ายังไม่มี
    manual สำหรับโดเมนนี้ (ไม่ throw)"""
    manual = load_manual(domain)
    if manual is None:
        return ""
    lines = [f"- {p.name}: {p.description}" for p in manual.pages if p.name and p.description]
    return "\n".join(lines)

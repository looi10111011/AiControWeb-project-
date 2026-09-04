"""site_learning/storage.py — W14: อ่าน/เขียน manual ที่ crawler.py สร้างลงดิสก์ — เก็บ
เป็นไฟล์ JSON ล้วนๆ ใต้ settings.site_manuals_dir แยกโฟลเดอร์ต่อโดเมน ไม่มี ChromaDB/
embedding เกี่ยวข้องเลย (คนละระบบกับ backend/app/rag/ ที่เก็บคู่มือที่ user อัปโหลดเอง —
ตั้งใจไม่ใช้ path "data/manuals" เดิมเพราะชื่อนั้นถูก chroma_collection_name="manuals"
จับจองความหมายไว้แล้ว)

โครงสร้างโฟลเดอร์ต่อโดเมน (settings.site_manuals_dir/{domain}/):
    latest.json     — version ล่าสุดเสมอ, ตัวที่ orchestrator โหลดไปใช้จริง — save_manual()
                     เขียนทับไฟล์นี้ตรงๆ ทุกครั้ง ไม่เก็บไฟล์ประวัติแยกต่อเวอร์ชัน (vN.json)
                     อีกต่อไป (ตามที่ user ขอ — กันไฟล์สะสมไม่รู้จบบนดิสก์ที่ commit เข้า
                     git) manual.version ยังนับเพิ่มไว้เป็น metadata ปกติ แค่ไม่มีไฟล์แยก
    ui-map.json     — tree โครงสร้างเมนู (derive จาก menu_path ของทุกหน้า)
    selectors.json  — flat lookup {"หน้า > ปุ่ม": {css, xpath, aria, data_testid}}
    knowledge.json  — {page_name: description} ฉบับย่อ ไว้ยัด prompt ถูกๆ
    llm-manual.json — W69: reshape ของ latest.json แบบเดียวกับคู่มือ QA ที่มนุษย์เขียนเอง
                     ({page_key: {url, elements: {semantic_key: css_selector}}} แบนๆ) ไว้
                     paste เข้า prompt LLM ตรงๆ ได้โดยไม่ต้องรู้จัก schema ภายในของระบบนี้
                     เลย — ดู build_llm_manual() ด้านล่างสำหรับขอบเขต (ไม่มี assertions/
                     notes/api เพราะข้อมูลพวกนั้นต้องมาจากการทดสอบจริงที่ crawler นี้ตั้งใจ
                     ไม่ทำ)
"""

import json
import re
import time
import urllib.parse
from collections import Counter
from pathlib import Path
from typing import Optional

from cryptography.fernet import InvalidToken

from backend.app.config import settings
from backend.app.core import goal_intent
from backend.app.core.crypto_store import get_fernet
from backend.app.site_learning.schema import ButtonInfo, FormFieldInfo, PageInfo, SiteManual


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


# --- W69: llm-manual.json — reshape SiteManual เป็นฟอร์แมตคู่มือ QA แบบมนุษย์เขียน (semantic
# key -> css selector แบนๆ ต่อหน้า) ตามที่ user ขอ — เก็บ selector/label เท่านั้น (จาก crawl
# ล้วนๆ), ไม่มี assertions/notes/api เพราะข้อมูลพวกนั้น (เช่น "submit ผิดรหัสผ่านแล้วเจอ error
# ว่าอะไร") ต้องมาจากการทดสอบจริงที่ crawler นี้ตั้งใจไม่ทำ (deterministic, ห้ามกด Submit — ดู
# crawler.py หัวไฟล์)


def _slugify(text: str) -> str:
    """แปลง label เป็น snake_case ล้วนๆ (a-z0-9 + underscore) ไว้ใช้เป็น key ของ
    pages/elements ใน llm-manual.json — ตัดอักขระอื่นทั้งหมดออกรวมถึงภาษาไทย/unicode อื่นๆ
    เพราะ key ต้องเป็น ASCII อ่าน/ค้นหาได้ง่ายสำหรับ LLM หรือ tooling downstream"""
    return re.sub(r"[^a-z0-9]+", "_", (text or "").strip().lower()).strip("_")


def _dedupe_key(base: str, used: set[str]) -> str:
    key = base
    i = 2
    while key in used:
        key = f"{base}_{i}"
        i += 1
    used.add(key)
    return key


def _semantic_key(label: str, suffix: str, used: set[str]) -> str:
    slug = _slugify(label) or suffix or "field"
    if suffix and not slug.endswith(suffix):
        slug = f"{slug}_{suffix}"
    return _dedupe_key(slug, used)


def _page_key(page: PageInfo, used: set[str]) -> str:
    base = _slugify(page.name) or _slugify(urllib.parse.urlparse(page.url).path) or "page"
    if not base.endswith("page"):
        base = f"{base}_page"
    return _dedupe_key(base, used)


def _button_label(b: ButtonInfo) -> str:
    """ลำดับความสำคัญเดียวกับ crawler.py::_button_label — text > aria_label > title >
    icon_hint > data_testid (humanized) เขียนแยกไว้ที่นี่แทนการ import ข้ามไฟล์ กัน circular
    import (crawler.py import storage.py อยู่แล้วสำหรับ save_manual())"""
    return (
        b.text or b.aria_label or b.title or b.icon_hint
        or b.data_testid.replace("-", " ").replace("_", " ").strip()
    )


def _button_suffix(b: ButtonInfo) -> str:
    return "link" if b.is_nav_menu_item else "button"


_FIELD_SUFFIX_BY_INPUT_TYPE = {
    "checkbox": "checkbox", "radio": "radio", "select": "select",
    "select-one": "select", "select-multiple": "select",
    "textarea": "textarea", "file": "upload",
}


def _field_label(f: FormFieldInfo) -> str:
    return f.label or f.field_name or f.placeholder


def _field_suffix(f: FormFieldInfo) -> str:
    return _FIELD_SUFFIX_BY_INPUT_TYPE.get((f.input_type or "").strip().lower(), "input")


def _build_page_elements(page: PageInfo) -> dict[str, str]:
    """semantic_key -> css selector ของหน้าเดียว — รวมปุ่ม/ช่องฟอร์มระดับหน้าปกติ บวก
    container selector + ปุ่มของ ui_patterns (การ์ด/แถวตารางที่ซ้ำกันหลาย instance — ใช้
    selector ที่ match ได้กับทุก instance ไม่ใช่แค่ตัวแทนตัวแรก) ข้าม element ที่ไม่มี
    label หรือไม่มี selector เลย (ไม่มีประโยชน์ให้ LLM อ้างอิง)"""
    elements: dict[str, str] = {}
    used: set[str] = set()
    for b in page.buttons:
        label = _button_label(b)
        if not label or not b.selector:
            continue
        elements[_semantic_key(label, _button_suffix(b), used)] = b.selector
    for f in page.forms:
        label = _field_label(f)
        if not label or not f.selector:
            continue
        elements[_semantic_key(label, _field_suffix(f), used)] = f.selector
    for pattern in page.ui_patterns:
        if pattern.selector:
            elements[_semantic_key(pattern.name or "item", "list", used)] = pattern.selector
        for b in pattern.buttons:
            label = _button_label(b)
            if not label or not b.selector:
                continue
            full_label = f"{pattern.name} {label}" if pattern.name else label
            elements[_semantic_key(full_label, _button_suffix(b), used)] = b.selector
    return elements


def _relative_page_url(url: str) -> str:
    if not url:
        return ""
    parsed = urllib.parse.urlparse(url)
    path = parsed.path or "/"
    return f"{path}?{parsed.query}" if parsed.query else path


def _detect_scheme(manual: SiteManual) -> str:
    for page in manual.pages:
        scheme = urllib.parse.urlparse(page.url).scheme if page.url else ""
        if scheme:
            return scheme
    return "https"


# ต้องเจอ selector เดียวกันเป๊ะบนหน้าต่างกันอย่างน้อยเท่านี้ก่อนถือว่าเป็น "component ที่ใช้
# ร่วมกันทั้งเว็บ" (nav/toast/spinner ฯลฯ) แทนที่จะเป็น element เฉพาะหน้าใดหน้าหนึ่งที่บังเอิญ
# ซ้ำกัน — เลข 3 เลือกแบบ heuristic กันเว็บที่มีแค่ 1-2 หน้าโดนดันทุก element เข้า shared
# หมดทั้งที่ยังไม่เห็นรูปแบบซ้ำจริงๆ
_SHARED_SELECTOR_MIN_PAGES = 3


def _build_shared_selectors(pages_elements: dict[str, dict[str, str]]) -> dict[str, str]:
    """selector ที่ปรากฏ (string ตรงกันเป๊ะ) บนหน้าต่างกันตั้งแต่ _SHARED_SELECTOR_MIN_PAGES
    หน้าขึ้นไป น่าจะเป็น component ที่ใช้ร่วมกันทั้งเว็บ — ยกออกมาไว้ต่างหากให้อ้างอิงได้โดยไม่
    ต้องเจาะจงหน้า (ยังคงอยู่ในแต่ละหน้าใน pages[].elements ตามเดิมด้วย ไม่ตัดออก — เก็บซ้ำได้
    ไม่เสียหาย) ทำงานแบบ best-effort จากข้อมูล crawl ที่มีอยู่แล้วล้วนๆ ไม่เดา/เติมความรู้จาก
    ภายนอก คืน {} เฉยๆ ถ้าไม่มี selector ไหนถึงเกณฑ์ (เว็บที่ crawl ได้น้อยหน้า)"""
    selector_pages: dict[str, set[str]] = {}
    selector_labels: dict[str, list[str]] = {}
    for page_key, elements in pages_elements.items():
        for key, selector in elements.items():
            selector_pages.setdefault(selector, set()).add(page_key)
            selector_labels.setdefault(selector, []).append(key)

    shared: dict[str, str] = {}
    used: set[str] = set()
    for selector, pages_seen in selector_pages.items():
        if len(pages_seen) < _SHARED_SELECTOR_MIN_PAGES:
            continue
        canonical = Counter(selector_labels[selector]).most_common(1)[0][0]
        shared[_dedupe_key(canonical, used)] = selector
    return shared


def build_llm_manual(manual: SiteManual) -> dict:
    """W69: reshape SiteManual (schema.py — โครงสร้างภายในที่ orchestrator ใช้จริง) เป็น
    ฟอร์แมต {app, base_url, pages: {page_key: {url, elements}}, shared_selectors} แบบเดียว
    กับคู่มือ QA ที่มนุษย์เขียนเอง (ดู scratchpad ตัวอย่างที่ user ส่งมา) — page_key/
    semantic_key ได้จาก _slugify() ของ name/label ที่ crawl มา ไม่มี LLM call เพิ่ม (ข้อมูล
    ทุกอย่างมีอยู่แล้วใน manual จาก describe_page()/extract_page() ตอน crawl) ตั้งใจไม่ใส่
    assertions/notes/instance_notes/api ตามที่ user ยืนยัน — ฟิลด์พวกนั้นต้องมาจากการทดสอบ
    จริง (submit ฟอร์มผิดดูข้อความ error, สังเกต network request) ซึ่งขัดกับกติกาเดิมของ
    crawler นี้ (deterministic, ห้ามกด Submit)"""
    used_page_keys: set[str] = set()
    pages: dict[str, dict] = {}
    pages_elements: dict[str, dict[str, str]] = {}
    for page in manual.pages:
        page_key = _page_key(page, used_page_keys)
        elements = _build_page_elements(page)
        pages_elements[page_key] = elements
        entry: dict = {"url": _relative_page_url(page.url)}
        if page.name:
            entry["name"] = page.name
        if page.description:
            entry["description"] = page.description
        entry["elements"] = elements
        pages[page_key] = entry

    scheme = _detect_scheme(manual)
    return {
        "app": manual.website,
        "base_url": f"{scheme}://{manual.website}" if manual.website else "",
        "version": manual.version,
        "generated_at": manual.generated_at,
        "summary": manual.summary,
        "pages": pages,
        "shared_selectors": _build_shared_selectors(pages_elements),
    }


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
    _write_json(domain_dir / "llm-manual.json", build_llm_manual(manual))
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


def save_credentials(domain: str, username: str, password: str) -> None:
    """W17: เก็บ username/password สำหรับโดเมนนี้ไว้ให้ orchestrator ดึงไปใช้ auto-login
    ตอนรัน task จริง (ดู core/orchestrator.py::_maybe_auto_login, site_learning/
    auto_login.py) — เขียนคนละไฟล์ (credentials.json) แยกจาก latest.json ของ manual โดย
    เจตนา ไม่ปนกับ manual ที่ save_manual() เขียนทับ (กันหลุดปนไปด้วยความไม่ตั้งใจถ้ามีคน
    แก้ save_manual()/schema ในอนาคต) — ไฟล์นี้เขียนทับตัวเดิมเสมอ ไม่มีประวัติเวอร์ชัน

    Security 1.4: username/password เข้ารหัสด้วย Fernet ก่อนเขียนลงดิสก์เสมอ (marker
    "encrypted": true ให้ load_credentials() แยกจากไฟล์เก่าที่ยังเป็น plaintext ได้)"""
    fernet = get_fernet()
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
        fernet = get_fernet()
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


def find_matching_page(manual: SiteManual, goal: str, min_score: int = 1) -> Optional[PageInfo]:
    """W21 ("Self-Learned Site Manual Integration"): หาแค่ "หน้าเดียวที่น่าจะตรงกับ goal
    ที่สุด" จาก manual ที่ crawl มาแล้ว ให้ routes.py ใช้ตัดสินใจว่าจะฉีด Strict Guided Plan
    context (ดู llm.py::build_strict_manual_context) หรือไม่ — matching แบบ keyword overlap
    ล้วนๆ (นับจำนวนคำใน goal ที่ยาว >= 3 ตัวอักษรที่ปรากฏใน name/description/breadcrumb ของ
    แต่ละหน้า) ไม่ใช้ embedding/ChromaDB เลย เพราะ manual นี้เป็น JSON แบนบนดิสก์อยู่แล้ว (ดู
    docstring หัวไฟล์) มีจำนวนหน้าต่อเว็บไซต์น้อยพอที่ keyword scoring ตรงไปตรงมาก็เพียงพอ ไม่
    คุ้มเพิ่ม dependency ใหม่ — คืน None ถ้าไม่มีหน้าไหนได้คะแนนถึง min_score (default 1 =
    พฤติกรรมเดิมทุกประการ ไม่กระทบ caller เดิม) ให้ caller fallback ไป dynamic planner ตามปกติ
    W67: เพิ่ม min_score ให้ caller ที่ต้องการความมั่นใจสูงกว่า (เช่น nav-fastpath auto-decide
    ที่ลงมือคลิกจริงตาม match ไม่ใช่แค่โชว์ context ให้ LLM อ่านเฉยๆ) ปรับ threshold เข้มขึ้นได้
    โดยไม่กระทบ caller เดิมที่ยังใช้ default 1"""
    # T2: ภาษาไทยไม่มีเว้นวรรค การตัดคำด้วย [^\w]+ จึงได้ token ก้อนเดียวยาวๆ ต่อประโยค ซึ่ง
    # ไม่มีทางตรงกับชื่อหน้า/breadcrumb ของคู่มือเลย = หา page ไม่เจอสำหรับ goal ภาษาไทยทุกอัน
    # goal_intent.matching_tokens() เติม token ที่เป็น ASCII จากคู่ field=value เข้ามาให้ ซึ่ง
    # เป็นส่วนที่เป็นภาษาอังกฤษเสมอแม้ประโยครอบๆ จะเป็นภาษาไทย — union ไม่ใช่แทนที่ เพื่อไม่ให้
    # พฤติกรรมของ goal ภาษาอังกฤษเดิมเปลี่ยนแม้แต่นิดเดียว
    goal_tokens = {t for t in re.split(r"[^\w]+", (goal or "").lower()) if len(t) >= 3}
    goal_tokens |= goal_intent.matching_tokens(goal or "")
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
    if best_score < min_score:
        return None
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
    # W65[1] ("Required-Field Validation"): FormFieldInfo.required ถูก crawl เก็บไว้ตั้งแต่
    # extractor.py::_EXTRACT_JS แล้ว (form field's HTML `required`/`aria-required`) แต่ก่อน
    # หน้านี้ไม่เคยมีใครอ่านเลย (dead data) — เพิ่ม block เดียวกับ buttons ด้านบนให้ planner
    # เห็นล่วงหน้าว่าหน้านี้มีฟิลด์อะไรบ้าง/ฟิลด์ไหนบังคับ ก่อนจะร่างแผน (ดู _PLAN_PROMPT_
    # TEMPLATE ใน llm.py ที่ใช้ข้อมูลนี้ตัดสินใจว่าต้องถาม user หาค่าที่ขาดก่อนไหม)
    if page.forms:
        lines.append("Recorded form fields on this page (label — required?):")
        for f in page.forms[:20]:
            field_label = f.label or f.field_name or f.placeholder or "(ไม่มี label)"
            req_note = " *จำเป็น" if f.required else ""
            lines.append(f"  - {field_label}{req_note}")
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

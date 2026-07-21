"""site_learning/storage.py — W14: อ่าน/เขียน manual ที่ crawler.py สร้างลงดิสก์ — เก็บ
เป็นไฟล์ JSON ล้วนๆ ใต้ settings.site_manuals_dir แยกโฟลเดอร์ต่อโดเมน ไม่มี ChromaDB/
embedding เกี่ยวข้องเลย (คนละระบบกับ backend/app/rag/ ที่เก็บคู่มือที่ user อัปโหลดเอง —
ตั้งใจไม่ใช้ path "data/manuals" เดิมเพราะชื่อนั้นถูก chroma_collection_name="manuals"
จับจองความหมายไว้แล้ว)

โครงสร้างโฟลเดอร์ต่อโดเมน (settings.site_manuals_dir/{domain}/):
    latest.json    — version ล่าสุดเสมอ, ตัวที่ orchestrator โหลดไปใช้จริง
    v1.json, v2.json, ...  — ประวัติทุกเวอร์ชัน ไม่เคยลบทิ้ง
    ui-map.json    — tree โครงสร้างเมนู (derive จาก menu_path ของทุกหน้า)
    selectors.json — flat lookup {"หน้า > ปุ่ม": {css, xpath, aria, data_testid}}
    knowledge.json — {page_name: description} ฉบับย่อ ไว้ยัด prompt ถูกๆ
"""

import json
import time
from pathlib import Path
from typing import Optional

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
    """บันทึก manual ใหม่ทั้งก้อน — bump version ต่อจาก version ล่าสุดที่มีอยู่จริงบน
    ดิสก์เสมอ (ไม่ใช่แค่ manual.version ที่ caller ส่งมา กันลืมอัปเดต) เขียน
    latest.json + v{N}.json (ประวัติ ไม่เคยลบ) + ui-map/selectors/knowledge.json คืน
    version number ใหม่"""
    domain_dir = _domain_dir(manual.website)
    existing = load_manual(manual.website)
    new_version = (existing.version + 1) if existing else 1
    manual.version = new_version
    manual.generated_at = time.time()

    data = manual.to_dict()
    _write_json(domain_dir / "latest.json", data)
    _write_json(domain_dir / f"v{new_version}.json", data)
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


def save_credentials(domain: str, username: str, password: str) -> None:
    """W17: เก็บ username/password สำหรับโดเมนนี้ไว้ให้ orchestrator ดึงไปใช้ auto-login
    ตอนรัน task จริง (ดู core/orchestrator.py::_maybe_auto_login, site_learning/
    auto_login.py) — เขียนคนละไฟล์ (credentials.json) แยกจาก latest.json/vN.json ของ
    manual โดยเจตนา: manual มีระบบ versioning (v1.json, v2.json, ... ไม่เคยลบทิ้ง) ถ้าฝัง
    credential ปนไปด้วยจะมีสำเนารหัสผ่านกระจายอยู่หลายไฟล์บนดิสก์ตลอดกาล — ไฟล์นี้เขียนทับ
    ตัวเดิมเสมอ ไม่มีประวัติเวอร์ชัน"""
    _write_json(_credentials_path(domain), {"username": username, "password": password})


def load_credentials(domain: str) -> Optional[dict]:
    """คืน {"username":..., "password":...} หรือ None ถ้ายังไม่เคยเก็บไว้/อ่านไม่ได้ (ไม่
    throw — โดเมนที่ไม่มี credential เก็บไว้เป็นเรื่องปกติ ไม่ใช่ error)"""
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
    return {"username": username, "password": password}


def credentials_exist(domain: str) -> bool:
    return _credentials_path(domain).exists()


def delete_credentials(domain: str) -> bool:
    path = _credentials_path(domain)
    if not path.exists():
        return False
    path.unlink()
    return True


def load_knowledge_text(domain: str) -> str:
    """ข้อความสั้นๆ (page_name: description ต่อบรรทัด) ไว้ฉีดเข้า prompt ตรงๆ ผ่าน
    site_manual_context (ดู llm.py::_build_user_turn_text) — คืนสตริงว่างเปล่าถ้ายังไม่มี
    manual สำหรับโดเมนนี้ (ไม่ throw)"""
    manual = load_manual(domain)
    if manual is None:
        return ""
    lines = [f"- {p.name}: {p.description}" for p in manual.pages if p.name and p.description]
    return "\n".join(lines)

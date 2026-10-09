"""site_learning/storage.py — W14: อ่าน/เขียน manual ของ crawler เป็น JSON ใต้ settings.site_manuals_dir/{domain}/
(ไม่มี ChromaDB; ไม่ใช้ "data/manuals" เพราะชื่อนั้นเป็นของ chroma_collection_name="manuals").

    latest.json     — version ล่าสุดที่ orchestrator โหลด; เขียนทับทุกครั้ง ไม่เก็บ vN.json (user ขอ กันไฟล์สะสมใน git)
    ui-map.json     — tree เมนู (จาก menu_path)
    selectors.json  — {"หน้า > ปุ่ม": {css, xpath, aria, data_testid}}
    knowledge.json  — {page_name: description}
    llm-manual.json — W69: {page_key: {url, elements: {semantic_key: css}}} แบบคู่มือ QA ที่มนุษย์เขียน (build_llm_manual)
    credentials.json — W17/Security 1.4: credential เข้ารหัส Fernet แยกจาก manual
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
    """{label: {"children": {...}, "page": page_name|None}} จาก menu_path ของทุกหน้า"""
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


def _selector_entry(b: ButtonInfo) -> dict:
    return {"css": b.selector, "xpath": b.xpath, "aria": b.aria_label, "data_testid": b.data_testid}


def _build_selectors(manual: SiteManual) -> dict:
    """flat lookup "{page_name} > {label}" -> {css, xpath, aria, data_testid} สำหรับ selector-repair/debug.

    W18: รวม UI pattern ("{page} > [{pattern}]" = selector ที่ match ทุก instance; "{page} > [{pattern}] {label}"
    = ปุ่มของ instance แรกเท่านั้น — instance อื่นต้องหา container ด้วย pattern.selector ก่อน)"""
    out = {}
    for page in manual.pages:
        for b in page.buttons:
            label = b.text or b.aria_label or b.icon_hint
            if label:
                out[f"{page.name} > {label}"] = _selector_entry(b)
        for pattern in page.ui_patterns:
            out[f"{page.name} > [{pattern.name}]"] = {
                "css": pattern.selector, "xpath": "", "aria": "", "data_testid": "",
            }
            for b in pattern.buttons:
                label = b.text or b.aria_label or b.icon_hint
                if label:
                    out[f"{page.name} > [{pattern.name}] {label}"] = _selector_entry(b)
    return out


def _build_knowledge(manual: SiteManual) -> dict:
    return {p.name: p.description for p in manual.pages if p.name}


# W69: llm-manual.json — selector/label จาก crawl ล้วน; ไม่มี assertions/notes/api เพราะต้องมาจากการทดสอบจริง
# (เช่น submit ผิดแล้วเจอ error อะไร) ซึ่ง crawler ตั้งใจไม่ทำ (deterministic, ห้ามกด Submit)


def _slugify(text: str) -> str:
    """snake_case ASCII ล้วน (a-z0-9_) — ตัดภาษาไทย/unicode ทิ้ง ให้ key ค้นหาง่ายสำหรับ LLM/tooling"""
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
    """ลำดับเดียวกับ crawler.py::_button_label (text > aria_label > title > icon_hint > data_testid) —
    สำเนาไว้กัน circular import (crawler.py import storage.py)"""
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
    """semantic_key -> css ของหน้าเดียว: ปุ่ม, ช่องฟอร์ม, container + ปุ่มของ ui_patterns; ข้ามที่ไม่มี label/selector"""
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


# selector เดียวกันเป๊ะบน >= N หน้า = component ร่วมทั้งเว็บ (nav/toast); 3 กันเว็บ 1-2 หน้าโดนดันเข้า shared หมด
_SHARED_SELECTOR_MIN_PAGES = 3


def _build_shared_selectors(pages_elements: dict[str, dict[str, str]]) -> dict[str, str]:
    """selector ที่อยู่บน >= _SHARED_SELECTOR_MIN_PAGES หน้า -> {label ที่พบบ่อยสุด: selector} (ยังคงอยู่ใน
    pages[].elements ด้วย). คืน {} ถ้าไม่มีตัวไหนถึงเกณฑ์"""
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
    """W69: SiteManual -> {app, base_url, version, generated_at, summary, pages: {page_key: {url, name?,
    description?, elements}}, shared_selectors} แบบคู่มือ QA ที่มนุษย์เขียน. ไม่มี LLM call (key จาก _slugify())"""
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
    """เขียนทับ latest.json + ไฟล์ derive ทั้งหมด. version = version บนดิสก์ + 1 (ไม่เชื่อ manual.version ของ caller)
    — เป็นแค่ metadata ไม่มีไฟล์ vN.json แล้ว. คืน version ใหม่"""
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
    """selector-repair: แทนที่ (จับคู่ด้วย url) หรือเพิ่มหน้าเดียว แล้ว save_manual() (bump version).
    คืน None ถ้ายังไม่มี manual (ต้อง crawl เต็มก่อน)"""
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
    """W17: credential สำหรับ auto-login (orchestrator._maybe_auto_login) — แยกไฟล์จาก latest.json โดยเจตนา
    กันหลุดปนกับ manual. Security 1.4: เข้ารหัส Fernet เสมอ + marker "encrypted": true แยกจากไฟล์ plaintext เก่า"""
    fernet = get_fernet()
    encrypted = {
        "encrypted": True,
        "username": fernet.encrypt(username.encode("utf-8")).decode("ascii"),
        "password": fernet.encrypt(password.encode("utf-8")).decode("ascii"),
    }
    _write_json(_credentials_path(domain), encrypted)


def load_credentials(domain: str) -> Optional[dict]:
    """คืน {"username", "password"} หรือ None ถ้าไม่มี/อ่านไม่ได้/ถอดรหัสไม่ได้. ไม่ throw.
    Security 1.4: ไฟล์ plaintext เก่า (ไม่มี marker "encrypted") อ่านตรงๆ แล้ว re-save แบบเข้ารหัสทันที (migration เงียบ)"""
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
        # legacy plaintext — migrate ล้มก็ไม่ throw (ยังคืนค่าที่อ่านได้อยู่ดี)
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
    """W21: หน้าเดียวที่ตรง goal ที่สุดด้วย keyword overlap (token >= 3 ตัวอักษรใน name/description/breadcrumb/
    menu_path) — ไม่ใช้ embedding เพราะ manual เป็น JSON เล็ก. คืน None ถ้าคะแนนสูงสุด < min_score.
    W67: min_score ให้ caller ที่ลงมือคลิกจริง (nav-fastpath auto-decide) ตั้ง threshold เข้มขึ้นได้; default 1 = เดิม"""
    # T2: ภาษาไทยไม่มีเว้นวรรค split ได้ token ก้อนเดียวไม่ตรงอะไรเลย — union token ASCII จาก field=value
    # (goal_intent.matching_tokens) ไม่ใช่แทนที่ เพื่อไม่เปลี่ยนพฤติกรรม goal ภาษาอังกฤษ
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
    """W21: breadcrumb (ลำดับ navigation จริง) > menu_path (แค่ตำแหน่งใน sidebar) > [page.name]"""
    return list(page.breadcrumb) or list(page.menu_path) or ([page.name] if page.name else [])


def build_learned_page_flow_text(page: PageInfo) -> str:
    """W21: block "📍 Learned Page Flow Sequence" (ฟอร์แมตตายตัวตามสเปค W21 Task5) — ใช้ใน /context
    (llm.py::context_inspection_reply) และเป็นหัวของ build_strict_manual_context()"""
    steps = _page_flow_steps(page)
    flow = " ➔ ".join(f"[{s}]" for s in steps) if steps else f"[{page.name or 'หน้าเป้าหมาย'}]"
    return f"📍 **Learned Page Flow Sequence:**\n`{flow}`"


def build_strict_manual_context(page: PageInfo) -> str:
    """W21: site_manual_context ของหน้าเดียวที่ตรง goal (route/ปุ่ม/selector จาก crawl) ขึ้นต้นด้วย marker
    "[PRE_LEARNED_MANUAL]" ที่ SYSTEM_PROMPT (llm.py, Strict Mode) ตรวจหา. selector เป็นแค่ข้อมูลอ้างอิงให้ LLM
    จับคู่กับ indexed elements — การกดยังเป็น index-based ไม่ยิง selector ข้าม perception"""
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
    # W65[1]: FormFieldInfo.required เคยเป็น dead data — แสดงให้ planner เห็นฟิลด์บังคับก่อนร่างแผน
    # (_PLAN_PROMPT_TEMPLATE ใน llm.py ใช้ตัดสินใจว่าต้องถามค่าที่ขาดจาก user ไหม)
    if page.forms:
        lines.append("Recorded form fields on this page (label — required?):")
        for f in page.forms[:20]:
            field_label = f.label or f.field_name or f.placeholder or "(ไม่มี label)"
            req_note = " *จำเป็น" if f.required else ""
            lines.append(f"  - {field_label}{req_note}")
    return "\n".join(lines)


def load_knowledge_text(domain: str) -> str:
    """บรรทัดละ "- page_name: description" สำหรับ site_manual_context; "" ถ้ายังไม่มี manual. ไม่ throw"""
    manual = load_manual(domain)
    if manual is None:
        return ""
    lines = [f"- {p.name}: {p.description}" for p in manual.pages if p.name and p.description]
    return "\n".join(lines)

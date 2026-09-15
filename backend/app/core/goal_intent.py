"""T1-T3: ภาษาของ goal, การปรับรูป goal สำหรับจับคู่, และ intent กลางตัวเดียว

ทำไมต้องมีไฟล์นี้ (หลักฐานจากโค้ดที่มีอยู่ ไม่ใช่การคาดเดา):

1. `plan_memory._UNSUPPORTED_SCRIPT_RE` ปิด Plan Memory ทั้ง find และ save สำหรับไทย/จีน/
   ญี่ปุ่น/เกาหลี/อาหรับ/ซีริลลิก เพราะ embedding ที่ใช้ทั้งแอปเป็น English-only tokenizer
   (goal ไทยสองอันที่ไม่เกี่ยวกันเลยได้ cosine distance ≈ 0 — บั๊กจริงที่บันทึกไว้ในไฟล์นั้น)
   ผลคืองานภาษาไทยจ่ายค่าร่างแผนใหม่ทุกครั้งตลอดไป
2. `site_learning/storage.py::find_matching_page()` ตัดคำด้วย `[^\\w]+` แล้วทิ้ง token ที่สั้น
   กว่า 3 ตัวอักษร — ภาษาไทยไม่มีเว้นวรรค จึงได้ token ก้อนเดียวยาวๆ ต่อประโยค = หา page
   ในคู่มือที่เรียนรู้มาไม่เจอเลย
3. orchestrator มี predicate ที่อ่าน goal อยู่ 15 ตัว แต่ละตัวมีคลังคำของตัวเอง และไม่มีที่ไหน
   รวมคำตอบไว้เป็นก้อนเดียว

**กฎที่ยึดตลอดทั้งไฟล์:**

- ผลของ normalize/canonical ใช้เป็น *กุญแจสำหรับจับคู่* เท่านั้น ห้ามเอาไปแทน goal ที่ส่งให้
  LLM หรือที่โชว์ให้ user เห็น — ข้อความที่ user พิมพ์เองคือเจตนาต้นฉบับ การเขียนใหม่ให้
  "สะอาด" คือการเดาแทนเขา
- deterministic ล้วน ไม่เรียก LLM (กฎเดิมของเส้นทาง `api/routes.py` — เคยมีคนใส่ LLM call
  เข้าไปแล้วพัง 26 เทสต์)
- ห้ามเขียน logic ตัดสิน intent ใหม่ — `canonical_intent()` **ประกอบ** จาก predicate เดิมของ
  orchestrator ทั้งหมด เพื่อให้พฤติกรรมของ guard ที่ใช้ predicate เหล่านั้นอยู่ไม่เปลี่ยนเลย
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional

# สคริปต์ที่ embedding model ปัจจุบันไม่รองรับจริง — ย้ายมาจาก plan_memory.py เพื่อให้มีที่เดียว
# (เดิมช่วงอักษรชุดนี้อยู่ในไฟล์นั้นไฟล์เดียว พอมีผู้ใช้ที่สองก็จะกลายเป็นสองชุดที่ drift ออกจากกัน
# ได้ทันที — รูปแบบเดียวกับที่ dom_locator.py/extractor.py เตือนไว้เรื่อง selector chain)
UNSUPPORTED_SCRIPT_RE = re.compile(
    "["
    "฀-๿"  # Thai
    "一-鿿"  # CJK Unified Ideographs (Chinese)
    "぀-ヿ"  # Hiragana/Katakana (Japanese)
    "가-힯"  # Hangul (Korean)
    "؀-ۿ"  # Arabic
    "Ѐ-ӿ"  # Cyrillic
    "]"
)

_THAI_RE = re.compile("[฀-๿]")
_LATIN_RE = re.compile("[A-Za-z]")


def uses_unsupported_script(text: str) -> bool:
    """True ถ้าข้อความมีตัวอักษรจากสคริปต์ที่ embedding ปัจจุบันแยกความหมายไม่ออก"""
    return bool(UNSUPPORTED_SCRIPT_RE.search(text or ""))


# --------------------------------------------------------------------------
# T1 · Goal Language Telemetry
# --------------------------------------------------------------------------

def detect_goal_language(goal: str) -> dict:
    """สรุป "รูปร่าง" ของ goal ที่ user พิมพ์มา — ใช้เป็น dimension ของ telemetry

    ตอบคำถามที่ตอนนี้ตอบไม่ได้เลย: งานภาษาไทยสำเร็จ/ใช้ token ต่างจากภาษาอังกฤษไหม ซึ่งสำคัญ
    กับโปรเจกต์นี้เป็นพิเศษเพราะกลไก "จำ goal" ทุกตัวถูกออกแบบมาสำหรับภาษาอังกฤษ (ดู docstring
    หัวไฟล์) — ต้องมีตัวเลขก่อนถึงจะพิสูจน์ได้ว่า T2/T3 ช่วยจริง (บทเรียนเดียวกับ W109 ที่ต้อง
    มาก่อน W110/W111)

    word_count ของภาษาไทยจะได้ 1 เสมอต่อประโยค (ไม่มีเว้นวรรค) — นั่นคือ *ข้อเท็จจริงที่ต้อง
    เห็น* ไม่ใช่ข้อบกพร่องของตัววัด เพราะมันคือสาเหตุตรงๆ ที่ตัวจับคู่แบบ token-based ใช้กับ
    ภาษาไทยไม่ได้"""
    text = goal or ""
    has_thai = bool(_THAI_RE.search(text))
    has_latin = bool(_LATIN_RE.search(text))
    if has_thai and has_latin:
        script = "mixed"
    elif has_thai:
        script = "thai"
    elif has_latin:
        script = "latin"
    else:
        script = "other"
    return {
        "script": script,
        "chars": len(text),
        "word_count": len([t for t in re.split(r"\s+", text.strip()) if t]),
    }


# --------------------------------------------------------------------------
# T2 · Goal Normalizer
# --------------------------------------------------------------------------

def normalize_goal_for_matching(goal: str) -> str:
    """รูปของ goal ที่ใช้ "จับคู่" เท่านั้น — ไม่ใช่ข้อความที่จะส่งให้ LLM หรือโชว์ให้ user

    สิ่งที่ทำ: lowercase, ตัดเครื่องหมายวรรคตอน, ยุบช่องว่างซ้ำ และ **แยกคู่ field=value ออกมา
    เป็น token ของตัวเอง** — goal จริงของ user คือ "แล้บลบuserole=ess" ซึ่งไม่มีเว้นวรรคเลย
    สักตัว ตัวจับคู่แบบ token-based จึงมองเห็นเป็นก้อนเดียวและหาอะไรไม่เจอ

    สิ่งที่จงใจ **ไม่** ทำ: ไม่แก้คำสะกดผิดด้วยพจนานุกรม (`userole` -> `userrole`) —
    `orchestrator._field_names_match()` ทนพิมพ์ผิดอยู่แล้วด้วย similarity 0.8 + กฎ containment
    ขั้นต่ำ 5 ตัวอักษร การเดาคำสะกดสองชั้นจะทำให้ debug ไม่ออกว่าชั้นไหนเดาผิด"""
    text = (goal or "").lower()
    parts: list[str] = []
    for key, value in goal_condition_pairs(goal):
        parts.extend([key, value])
    cleaned = re.sub(r"[^\w฀-๿]+", " ", text)
    parts.append(cleaned)
    return " ".join(" ".join(parts).split())


def matching_tokens(goal: str) -> set[str]:
    """token ที่ใช้ให้คะแนนความเข้ากันได้ — เติมคู่ field=value เข้าไปเสมอ

    `site_learning/storage.py::find_matching_page()` ตัด token ที่สั้นกว่า 3 ตัวอักษรทิ้งและ
    ตัดคำด้วยช่องว่าง ซึ่งกับภาษาไทยแปลว่าได้ token ก้อนเดียวที่ไม่ตรงกับชื่อหน้าอะไรเลย —
    คู่ field=value เป็นส่วนที่เป็น ASCII เสมอแม้ประโยคจะเป็นภาษาไทย จึงเป็นสะพานที่ใช้ได้จริง
    โดยไม่ต้องมี tokenizer ภาษาไทย"""
    return {t for t in normalize_goal_for_matching(goal).split() if len(t) >= 3}


# --------------------------------------------------------------------------
# T3 · Intent Canonicalizer
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class GoalIntent:
    """คำตอบรวมของคำถาม "goal นี้สั่งให้ทำอะไร" ที่เดิมกระจายอยู่ใน predicate 15 ตัว"""

    operation: str                      # delete | edit | create | read | unknown
    scope: str                          # all | conditional | single
    conditions: tuple[tuple[str, str], ...] = ()
    about_own_account: bool = False
    asks_for_count: bool = False
    script: str = "other"

    def as_telemetry(self) -> dict:
        return {
            "goal_operation": self.operation,
            "goal_scope": self.scope,
            "goal_has_condition": bool(self.conditions),
        }


_ASCII_FIELD_TAIL_RE = re.compile(r"[a-z0-9_]+$")


def _ascii_field_tail(field_name: str) -> str:
    """ตัดคำภาษาไทยที่ติดมาข้างหน้าชื่อ field ออก

    วัดจาก goal จริงของ user: `"แล้บลบuserole=ess"` — ภาษาไทยไม่มีเว้นวรรค คำว่า "แล้บลบ" จึง
    ติดมากับชื่อ field และ `\\w` ของ regex ก็นับอักษรไทยเป็นตัวอักษรด้วย ผลคือได้ชื่อ field ว่า
    "แล้บลบuserole" ฝั่ง orchestrator ไม่เป็นไรเพราะ `_field_names_match()` ทนเรื่องนี้อยู่แล้ว
    (กฎ containment) แต่ *กุญแจ* ต้องเป็นสตริงที่คงที่ — ถ้าเจตนาเดียวกันแต่พิมพ์นำหน้าต่างกัน
    แล้วได้คนละกุญแจ ระบบจำแผนก็ไม่มีประโยชน์

    ชื่อ field บนหน้าเว็บเป็น ASCII เสมอ (มาจาก label ใน HTML) ส่วนที่เป็นภาษาไทยจึงเป็น
    ประโยครอบข้างที่ติดมา ไม่ใช่ชื่อ field — คืนตัวเดิมถ้าไม่มีหาง ASCII ให้ตัด (ไม่เดาต่อ)"""
    match = _ASCII_FIELD_TAIL_RE.search(field_name or "")
    return match.group(0) if match else (field_name or "")


def goal_condition_pairs(goal: str) -> list[tuple[str, str]]:
    """คู่ field=value ใน goal — ห่อ `orchestrator._goal_condition_pairs()` ไว้ชั้นเดียว

    import แบบ lazy เพราะ orchestrator จะ import ไฟล์นี้กลับ (telemetry/plan_memory เรียกผ่าน
    ไฟล์นี้) — import ที่ระดับ module จะกลายเป็นวงกลมทันที"""
    from backend.app.core import orchestrator as _orch

    return [
        (_ascii_field_tail(key), value)
        for key, value in _orch._goal_condition_pairs(goal or "")
    ]


def canonical_intent(goal: str) -> GoalIntent:
    """ประกอบคำตอบจาก predicate เดิมของ orchestrator — ไม่มี logic ตัดสินใหม่ในนี้เลย

    ที่ไม่มี operation "navigate": มันต้องอาศัยคลังคำชุดใหม่ (predicate เดิมไม่มีตัวไหนตอบ
    คำถามนี้) ซึ่งขัดกับกฎของไฟล์นี้ที่ว่าห้ามเขียน logic ตัดสิน intent ใหม่ — goal ที่เป็นการ
    นำทางล้วนจึงได้ operation="unknown" และตกไปใช้เส้นทาง fallback เดิมทุกประการ"""
    from backend.app.core import orchestrator as _orch

    text = goal or ""
    conditions = tuple(goal_condition_pairs(text))

    if _orch._is_deletion_intent_goal(text):
        operation = "delete"
    elif _orch._is_edit_all_intent_goal(text):
        operation = "edit"
    elif _orch._goal_wants_to_create(text):
        operation = "create"
    elif _orch._goal_asks_for_a_count(text):
        operation = "read"
    else:
        operation = "unknown"

    if _orch._is_delete_all_intent_goal(text) or _orch._is_edit_all_intent_goal(text):
        scope = "all"
    elif conditions:
        scope = "conditional"
    else:
        scope = "single"

    return GoalIntent(
        operation=operation,
        scope=scope,
        conditions=conditions,
        about_own_account=_orch._goal_is_about_the_signed_in_account(text),
        asks_for_count=_orch._goal_asks_for_a_count(text),
        script=detect_goal_language(text)["script"],
    )


def plan_memory_intent_key(goal: str) -> Optional[str]:
    """กุญแจ lineage ของ Plan Memory ที่ไม่ต้องพึ่ง embedding — คืน None ถ้าตัดสินไม่ได้

    เงื่อนไขเข้มโดยตั้งใจ: ต้องรู้ทั้ง operation และมีเงื่อนไข field=value อย่างน้อยหนึ่งคู่
    ถ้ากุญแจกว้างกว่านี้ goal คนละเรื่องจะชนกัน ซึ่งคือบั๊กเดิมที่ทำให้ต้องปิด Plan Memory
    สำหรับภาษาเหล่านี้ตั้งแต่แรก (ดู plan_memory.py) — ยอมไม่ match ดีกว่า match ผิดเจตนา
    ค่าใน key มาจาก _goal_condition_pairs() ตรงๆ จึงแยก userrole=ess ออกจาก userrole=admin
    ได้เอง (เคสเดียวกับที่ _is_value_substitution_only() กันไว้ฝั่ง semantic)

    ข้อจำกัดที่ยอมรับโดยตั้งใจ: key ไม่ทนคำสะกดผิด — goal ที่พิมพ์ "userole" (r ตัวเดียว ซึ่ง
    เป็นสิ่งที่ user คนนี้พิมพ์จริง) ได้คนละ key กับ "userrole" ผลคือ lineage แยกกัน แล้วอันที่
    ไม่ตรงก็ fallback ไปให้ LLM ร่างใหม่ = พฤติกรรมเดิมทุกประการ ไม่มีแผนผิดเจตนาถูกเสิร์ฟ
    การเดาคำสะกดตรงนี้จะทำให้ key ไม่นิ่ง ซึ่งอันตรายกว่าการไม่ match"""
    intent = canonical_intent(goal)
    if intent.operation == "unknown" or not intent.conditions:
        return None
    pairs = ",".join(f"{k}={v}" for k, v in sorted(intent.conditions))
    return f"{intent.operation}:{intent.scope}:{pairs}"


def spaceless(text: str) -> str:
    """ข้อความที่ตัดช่องว่างออกทั้งหมด + lowercase — ใช้เทียบคำสำคัญเท่านั้น"""
    return re.sub(r"\s+", "", (text or "").lower())


def contains_keyword(text: str, keywords: Iterable[str]) -> bool:
    """คำสำคัญโผล่ใน text ไหม — เทียบทั้งแบบตรงตัวและแบบตัดช่องว่างออกทั้งสองฝั่ง

    W_thai_keyword_space (บั๊กจริงที่วัดได้จากรันสด 2026-09-03): goal ของ user คือ
    "เปิดเว็ป แล้วเปลี่ยน รหัสผ่านเป็น 12345678" แต่คำสำคัญที่เก็บไว้คือ "เปลี่ยนรหัสผ่าน"
    (ไม่มีเว้นวรรค) — substring match จึงเป็นเท็จเพราะ **เว้นวรรคเดียว** ผลคือบล็อกกฎ W20
    ("ห้ามคลิก My Info ให้กดเมนูโปรไฟล์มุมขวาบนก่อน") ไม่ถูกส่งเข้า prompt เลยตอนที่โมเดล
    กำลังเลือกทาง แล้ว agent ก็เดินเข้า My Info -> Memberships -> Delete ตรงข้ามกับกฎเป๊ะ

    ภาษาไทยไม่มีขอบเขตคำ ผู้ใช้เว้นวรรคตรงไหนก็ได้ตามใจ การเทียบ substring ดิบจึงเปราะกับ
    ภาษากลุ่มนี้โดยธรรมชาติ — เทียบเวอร์ชันตัดช่องว่างเพิ่มอีกชั้นแก้ได้ทั้งกลุ่ม
    ฝั่งอังกฤษไม่กระทบ: การเทียบตรงตัวยังทำงานก่อนเสมอ และเวอร์ชันตัดช่องว่างไม่เคยทำให้
    ข้อความที่เดิม "ไม่ match" กลายเป็น match ข้ามเครื่องหมายวรรคตอน (ตัดเฉพาะ whitespace)"""
    lower = (text or "").lower()
    if any(keyword in lower for keyword in keywords):
        return True
    packed = spaceless(text)
    return any(spaceless(keyword) in packed for keyword in keywords)

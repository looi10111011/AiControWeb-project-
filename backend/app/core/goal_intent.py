"""T1-T3: ภาษาของ goal, การปรับรูป goal สำหรับจับคู่, และ intent กลางตัวเดียว

เหตุผล: (1) plan_memory ปิด find/save สำหรับไทย/CJK/อาหรับ/ซีริลลิก (embedding English-only) งานไทยจึง
ร่างแผนใหม่ทุกครั้ง (2) storage.py::find_matching_page() ตัดคำด้วยช่องว่าง ภาษาไทยได้ token ก้อนเดียว
หา page ไม่เจอ (3) orchestrator มี predicate อ่าน goal 15 ตัว ไม่มีที่รวมคำตอบ

กฎ: ผล normalize/canonical ใช้เป็น *กุญแจจับคู่* เท่านั้น ห้ามแทน goal ที่ส่ง LLM/โชว์ user;
deterministic ล้วน ไม่เรียก LLM (เคยใส่แล้วพัง 26 เทสต์); canonical_intent() **ประกอบ** จาก predicate
เดิมของ orchestrator ห้ามเขียน logic ตัดสิน intent ใหม่
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional

# สคริปต์ที่ embedding ไม่รองรับ — ย้ายมาจาก plan_memory.py ให้มีที่เดียว (ผู้ใช้สองที่จะ drift ออกจากกัน)
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


def detect_goal_language(goal: str) -> dict:
    """T1: {script, chars, word_count} ของ goal เป็น dimension ของ telemetry (วัดก่อนว่า T2/T3 ช่วยจริง)
    word_count ภาษาไทยได้ 1 ต่อประโยคโดยตั้งใจ — นั่นคือสาเหตุที่ตัวจับคู่แบบ token ใช้กับไทยไม่ได้"""
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


def normalize_goal_for_matching(goal: str) -> str:
    """T2: รูปของ goal สำหรับจับคู่เท่านั้น — lowercase, ตัดวรรคตอน, ยุบช่องว่าง และแยกคู่ field=value
    เป็น token ของตัวเอง (goal จริง "แล้บลบuserole=ess" ไม่มีเว้นวรรคเลย) จงใจไม่แก้คำสะกดผิด
    (_field_names_match() ทนอยู่แล้ว เดาสองชั้นจะ debug ไม่ออก)"""
    text = (goal or "").lower()
    parts: list[str] = []
    for key, value in goal_condition_pairs(goal):
        parts.extend([key, value])
    cleaned = re.sub(r"[^\w฀-๿]+", " ", text)
    parts.append(cleaned)
    return " ".join(" ".join(parts).split())


def matching_tokens(goal: str) -> set[str]:
    """token (>=3 ตัวอักษร) สำหรับให้คะแนนจับคู่ — รวมคู่ field=value ที่เป็น ASCII เสมอแม้ประโยคเป็นไทย
    จึงเป็นสะพานให้ find_matching_page() โดยไม่ต้องมี tokenizer ภาษาไทย"""
    return {t for t in normalize_goal_for_matching(goal).split() if len(t) >= 3}


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
    """ตัดคำไทยที่ติดหน้าชื่อ field (goal จริง "แล้บลบuserole=ess" -> \\w นับอักษรไทยด้วย ได้ "แล้บลบuserole")
    กุญแจต้องคงที่ไม่ขึ้นกับคำนำหน้า ชื่อ field บนเว็บเป็น ASCII เสมอ — คืนตัวเดิมถ้าไม่มีหาง ASCII"""
    match = _ASCII_FIELD_TAIL_RE.search(field_name or "")
    return match.group(0) if match else (field_name or "")


def goal_condition_pairs(goal: str) -> list[tuple[str, str]]:
    """คู่ field=value ใน goal (ห่อ orchestrator._goal_condition_pairs()) — import lazy กัน import วงกลม"""
    from backend.app.core import orchestrator as _orch

    return [
        (_ascii_field_tail(key), value)
        for key, value in _orch._goal_condition_pairs(goal or "")
    ]


def canonical_intent(goal: str) -> GoalIntent:
    """T3: ประกอบจาก predicate เดิมของ orchestrator — ไม่มี operation "navigate" เพราะต้องใช้คลังคำใหม่
    (ขัดกฎหัวไฟล์) goal นำทางล้วนจึงได้ "unknown" และใช้ fallback เดิม"""
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
    """กุญแจ lineage ของ Plan Memory ไม่พึ่ง embedding — คืน None ถ้าไม่รู้ operation หรือไม่มีคู่
    field=value (เข้มโดยตั้งใจ: กุญแจกว้างไป goal คนละเรื่องจะชนกัน = บั๊กเดิม) ค่าใน key แยก
    userrole=ess กับ =admin ได้เอง จงใจไม่ทนคำสะกดผิด ("userole" ได้ key แยก -> fallback LLM ตามเดิม)
    เพราะ key ที่ไม่นิ่งอันตรายกว่าการไม่ match"""
    intent = canonical_intent(goal)
    if intent.operation == "unknown" or not intent.conditions:
        return None
    pairs = ",".join(f"{k}={v}" for k, v in sorted(intent.conditions))
    return f"{intent.operation}:{intent.scope}:{pairs}"


def spaceless(text: str) -> str:
    """ข้อความที่ตัดช่องว่างออกทั้งหมด + lowercase — ใช้เทียบคำสำคัญเท่านั้น"""
    return re.sub(r"\s+", "", (text or "").lower())


def contains_keyword(text: str, keywords: Iterable[str]) -> bool:
    """คำสำคัญโผล่ใน text ไหม — เทียบตรงตัวก่อน แล้วเทียบแบบตัด whitespace ทั้งสองฝั่ง

    W_thai_keyword_space (รันสด 2026-09-03): "แล้วเปลี่ยน รหัสผ่านเป็น..." ไม่ match "เปลี่ยนรหัสผ่าน" เพราะ
    เว้นวรรคเดียว บล็อกกฎ W20 จึงไม่เข้า prompt แล้ว agent เดินเข้า My Info -> Delete ตรงข้ามกฎ"""
    lower = (text or "").lower()
    if any(keyword in lower for keyword in keywords):
        return True
    packed = spaceless(text)
    return any(spaceless(keyword) in packed for keyword in keywords)

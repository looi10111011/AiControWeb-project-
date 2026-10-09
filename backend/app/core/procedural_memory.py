"""core/procedural_memory.py — W_procmem: template แบบมีโครงสร้าง (ordered steps + locator
descriptor จาก dom_locator.py + {{slot}} placeholder ไม่เคยเก็บค่าจริง) ให้ fastpath_executor.py replay
ข้าม LLM loop — ต่างจาก plan_memory.py ที่เก็บแค่ข้อความแผนไว้ข้าม LLM ตอนร่างแผน
ลำดับตอนหาแผน (routes.py::generate_plan): procedural template -> plan_memory -> LLM ร่างใหม่

Data model เหมือน plan_memory (lineage = intent_key, versioned, ไม่ลบ) แต่ steps/slots เก็บเป็น JSON
string (Chroma metadata รับแค่ scalar) Versioning เทียบ structural signature (ลำดับ action + ชื่อ slot)
แทนเนื้อหาเป๊ะ เพราะ locator ของ run เดียวกันต่างกันเล็กน้อยได้ — ไม่งั้นได้ lineage ซ้ำทุกครั้งที่สำเร็จ

ห้าม throw ออกไปให้ endpoint/orchestrator พัง (เป็น enhancement ไม่ใช่ requirement)"""

import json
import re
import time
import uuid
from typing import Optional

from backend.app.config import settings
from backend.app.core.plan_memory import _uses_unsupported_script
from backend.app.rag.chroma_client import get_procedural_memory_collection

_SLOT_RE = re.compile(r"\{\{(\w+)\}\}")

# คำกริยาไทยสำหรับ render_steps_as_plan_text() ให้แผนหน้าตาเหมือนที่ LLM ร่าง (llm.py::_PLAN_PROMPT_TEMPLATE)
_ACTION_VERB_TH = {
    "click": "คลิก",
    "fill": "กรอกข้อมูลลงใน",
    "select": "เลือกค่าใน",
    "check": "ติ๊กเลือก",
    "press_key": "กดปุ่มคีย์บอร์ดที่",
    "hover": "เลื่อนเมาส์ไปที่",
    "goto": "ไปที่หน้า",
}


def substitute_slots(text: str, slot_values: dict) -> str:
    """แทน {{slot_name}} ด้วยค่าจาก slot_values — slot ที่ไม่มีค่าปล่อยเป็น literal {{...}} ให้เห็นชัด"""
    return _SLOT_RE.sub(lambda m: str(slot_values.get(m.group(1), m.group(0))), text)


def _step_signature(steps: list[dict]) -> str:
    """ลำดับ action type ล้วนๆ (ไม่รวม locator/value) — ส่วนหนึ่งของ versioning signature"""
    return "/".join(str(step.get("action", "")) for step in steps)


def _slot_names(slots: list) -> tuple:
    """ชื่อ slot เรียงตามตัวอักษร — slot ต่างกันแม้ action เหมือนกันถือเป็นคนละรุ่นของ template"""
    names = []
    for slot in slots or []:
        name = slot.get("name") if isinstance(slot, dict) else slot
        if name:
            names.append(str(name))
    return tuple(sorted(names))


def _best_match(domain: str, goal_pattern: str) -> Optional[tuple[str, float]]:
    """คืน (intent_key, distance) ของ approved document ที่ใกล้ที่สุดในโดเมน หรือ None"""
    collection = get_procedural_memory_collection()
    results = collection.query(
        query_texts=[goal_pattern], n_results=1, where={"$and": [{"domain": domain}, {"status": "approved"}]}
    )
    ids = results.get("ids") or [[]]
    if not ids or not ids[0]:
        return None
    metadata = results["metadatas"][0][0]
    distance = results["distances"][0][0]
    return metadata["intent_key"], distance


def _latest_version(domain: str, intent_key: str) -> Optional[dict]:
    """คืน metadata ของ version ล่าสุดของ lineage นี้ หรือ None ถ้าไม่มี document เลย"""
    collection = get_procedural_memory_collection()
    got = collection.get(where={"$and": [{"domain": domain}, {"intent_key": intent_key}]})
    metadatas = got.get("metadatas") or []
    if not metadatas:
        return None
    return max(metadatas, key=lambda m: m["version"])


def save_template(domain: str, template: dict) -> Optional[dict]:
    """บันทึก template จาก Abstractor (schema ABSTRACTOR_TOOL ใน llm.py) หลัง task สำเร็จ
    คืน {template_id, intent_key, version} หรือ None ถ้าขาด goal_pattern/steps, ใช้สคริปต์ที่
    embedding ไม่รองรับ หรือ error ใดๆ — never raises"""
    goal_pattern = template.get("goal_pattern", "")
    url_pattern = template.get("url_pattern", "")
    steps = template.get("steps") or []
    slots = template.get("slots") or []
    if not goal_pattern or not steps:
        return None
    if _uses_unsupported_script(goal_pattern):
        return None
    try:
        signature = _step_signature(steps)
        slot_names = _slot_names(slots)

        intent_key = None
        new_version = 1
        success_count = 1
        # W_procmem versioning: reuse plan_memory_max_distance (embedding model ตัวเดียวกัน)
        match = _best_match(domain, goal_pattern)
        if match is not None and match[1] <= settings.plan_memory_max_distance:
            candidate_key = match[0]
            latest = _latest_version(domain, candidate_key)
            if (
                latest is not None
                and latest.get("step_signature") == signature
                and tuple(json.loads(latest.get("slot_names_json", "[]"))) == slot_names
            ):
                intent_key = candidate_key
                new_version = latest["version"] + 1
                success_count = latest.get("success_count", 0) + 1

        if intent_key is None:
            intent_key = str(uuid.uuid4())
            new_version = 1
            success_count = 1

        template_id = str(uuid.uuid4())
        now = time.time()
        collection = get_procedural_memory_collection()
        collection.add(
            documents=[goal_pattern],
            metadatas=[{
                "domain": domain,
                "url_pattern": url_pattern,
                "intent_key": intent_key,
                "version": new_version,
                "status": "approved",
                "created_at": now,
                "last_used_at": now,
                "success_count": success_count,
                "failure_count": 0,
                "goal_pattern": goal_pattern,
                "steps_json": json.dumps(steps, ensure_ascii=False),
                "slots_json": json.dumps(slots, ensure_ascii=False),
                "step_signature": signature,
                "slot_names_json": json.dumps(list(slot_names), ensure_ascii=False),
                "template_id": template_id,
            }],
            ids=[str(uuid.uuid4())],
        )
        return {"template_id": template_id, "intent_key": intent_key, "version": new_version}
    except Exception as e:
        print(f"⚠️ Procedural Memory save_template error: {e}", flush=True)
        return None


def find_candidate_templates(domain: str, goal: str, k: Optional[int] = None) -> list[dict]:
    """W_procmem Step 1: top-K candidate ในโดเมน ไม่มี threshold (llm.plan_with_procedural_memory()
    ตัดสิน reuse/adapt/plan_fresh เอง) คืน [] ถ้าไม่มี/สคริปต์ไม่รองรับ/error — never raises"""
    if _uses_unsupported_script(goal):
        return []
    try:
        collection = get_procedural_memory_collection()
        n = k or settings.procedural_memory_max_candidates
        results = collection.query(
            query_texts=[goal], n_results=n, where={"$and": [{"domain": domain}, {"status": "approved"}]}
        )
        ids = results.get("ids") or [[]]
        if not ids or not ids[0]:
            return []
        candidates = []
        for meta, distance in zip(results["metadatas"][0], results["distances"][0]):
            try:
                steps = json.loads(meta.get("steps_json", "[]"))
                slots = json.loads(meta.get("slots_json", "[]"))
            except (TypeError, ValueError):
                continue
            candidates.append({
                "template_id": meta.get("template_id"),
                "intent_key": meta.get("intent_key"),
                "version": meta.get("version"),
                "goal_pattern": meta.get("goal_pattern", ""),
                "url_pattern": meta.get("url_pattern", ""),
                "steps": steps,
                "slots": slots,
                "distance": distance,
                # ACC-1: ส่ง track record ให้ Planner แยก template ที่บันทึกครั้งเดียวออกจากที่ reuse
                # สำเร็จหลายครั้ง (เดิมไม่มี field นี้ confidence มาจาก semantic match ล้วนๆ)
                "success_count": meta.get("success_count", 0),
                "failure_count": meta.get("failure_count", 0),
            })
        return candidates
    except Exception as e:
        print(f"⚠️ Procedural Memory find_candidate_templates error: {e}", flush=True)
        return []


def record_template_outcome(template_id: str, success: bool) -> None:
    """อัปเดต success_count/failure_count/last_used_at หลัง fast-path run (สำเร็จผ่าน Repair ก็นับ
    success) — template_id ที่หาไม่เจอไม่มีผล; never raises"""
    if not template_id:
        return
    try:
        collection = get_procedural_memory_collection()
        got = collection.get(where={"template_id": template_id})
        ids = got.get("ids") or []
        metadatas = got.get("metadatas") or []
        if not ids:
            return
        meta = dict(metadatas[0])
        meta["success_count"] = meta.get("success_count", 0) + (1 if success else 0)
        meta["failure_count"] = meta.get("failure_count", 0) + (0 if success else 1)
        meta["last_used_at"] = time.time()
        collection.update(ids=[ids[0]], metadatas=[meta])
    except Exception as e:
        print(f"⚠️ Procedural Memory record_template_outcome error: {e}", flush=True)


def render_steps_as_plan_text(steps: list[dict], slot_values: Optional[dict] = None) -> str:
    """steps (+ แทน slot) -> ข้อความแผน "1. ...\\n2. ..." รูปแบบเดียวกับ llm.py::_PLAN_PROMPT_TEMPLATE
    ให้ frontend parsePlanSteps() ใช้ได้ตรงๆ step ที่ sensitive=True แสดง "••••••" เสมอ"""
    slot_values = slot_values or {}
    lines = []
    for i, step in enumerate(steps, start=1):
        action = step.get("action", "")
        verb = _ACTION_VERB_TH.get(action, action)
        target = step.get("target")
        if isinstance(target, dict):
            target_label = target.get("accessible_name") or target.get("css_fallback") or target.get("tag", "")
        else:
            target_label = str(target or "")

        line = f'{verb} "{target_label}"' if target_label else verb
        raw_value = step.get("value")
        if raw_value is not None and str(raw_value):
            if step.get("sensitive"):
                value_text = "••••••"
            else:
                value_text = substitute_slots(str(raw_value), slot_values)
            line += f' ด้วยค่า "{value_text}"'
        lines.append(f"{i}. {line}")
    return "\n".join(lines)


def apply_template_patch(steps: list[dict], patch: Optional[list[dict]]) -> list[dict]:
    """ใช้ patch (op: replace/insert/remove, index, step?) จาก decision="adapt" ของ Planner
    คืน list ใหม่เสมอ (ไม่แก้ของเดิม) op ที่ index ผิด/ไม่มี step ถูกข้ามทีละ op ไม่ throw"""
    result = list(steps)
    for op in patch or []:
        action = op.get("op")
        index = op.get("index")
        if not isinstance(index, int):
            continue
        if action == "remove":
            if 0 <= index < len(result):
                result.pop(index)
        elif action == "replace":
            if 0 <= index < len(result) and op.get("step"):
                result[index] = op["step"]
        elif action == "insert":
            if 0 <= index <= len(result) and op.get("step"):
                result.insert(index, op["step"])
    return result

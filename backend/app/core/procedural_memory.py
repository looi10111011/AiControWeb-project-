"""core/procedural_memory.py — W_procmem: template แบบมีโครงสร้าง (ordered steps +
stable locator + {{slot}} placeholder) ต่อยอดจาก core/plan_memory.py

ต่างจาก plan_memory.py ตรงไหน: plan_memory เก็บ "ข้อความแผน" ดิบๆ ที่ user confirm แล้ว
reuse ได้แค่ตอนร่างแผน (ข้าม LLM call เดียว) — ตัวนี้เก็บ "step ที่รันได้จริง" (action +
locator descriptor จาก core/dom_locator.py + ค่าที่แทนด้วย {{slot}} เสมอ ไม่เคยเป็นค่า
จริง) ให้ fastpath_executor.py replay ได้ตรงๆ ข้าม step-by-step LLM loop ทั้งหมด ไม่ใช่
แค่ตอนร่างแผน

ลำดับความสำคัญตอนหาแผนให้ user (ดู routes.py::generate_plan): procedural template
(ไฟล์นี้) ก่อน -> plan_memory (ข้อความแผนเดิม) -> LLM ร่างใหม่สดๆ — plan_memory ไม่ถูก
แทนที่ ยังทำงานเป็น fallback ชั้นถัดไปเหมือนเดิมทุกประการ

Data model: เหมือน plan_memory.py ทุกประการ (lineage = intent_key สุ่มครั้งแรก, versioned,
ไม่เคยลบทิ้ง) ต่างแค่ metadata ที่เก็บเพิ่ม — steps/slots เป็น list/dict ซึ่ง Chroma
metadata รองรับแค่ scalar (str/int/float/bool) เท่านั้น จึงต้อง json.dumps() ก่อนเก็บ
เสมอ แล้ว json.loads() กลับตอนอ่าน (ดู steps_json/slots_json ด้านล่าง)

Versioning ต่างจาก plan_memory.py ตรงที่ plan_memory เทียบ "เนื้อหาแผนเป๊ะๆ" ว่าเหมือน
version ล่าสุดไหม (ข้อความ plain text, exact match เดียวกันได้ทุกครั้งถ้า user ไม่แก้
อะไรเลย) — ตัวนี้เทียบ "structural signature" แทน (ลำดับ action type ของ steps + ชื่อ
slot ทั้งหมด) เพราะสอง task run ที่ "ทำงานเดียวกัน" จริงๆ แทบไม่มีทางได้ JSON เหมือนกัน
เป๊ะไบต์ต่อไบต์ (locator descriptor อาจต่างกันเล็กน้อยแม้ action sequence จะเหมือนกัน) —
signature ตรงกัน = ถือว่าเป็น run เดิมของ intent เดิม บันทึกเพิ่ม version/success_count
ให้ lineage นั้น แทนที่จะสร้าง lineage ใหม่ที่แทบซ้ำกันทุกครั้งที่ task class เดิมสำเร็จ
ซ้ำ (จะทำให้ find_candidate_templates() คืน candidate ซ้ำๆ กันเปล่าๆ ให้ Planner เลือก)

ห้าม throw ออกไปให้ endpoint/orchestrator loop พังเด็ดขาด (กฎเดียวกับ plan_memory.py/
long_term_memory.py) — เป็นแค่ enhancement ไม่ใช่ requirement"""

import json
import re
import time
import uuid
from typing import Optional

from backend.app.config import settings
from backend.app.core.plan_memory import _uses_unsupported_script
from backend.app.rag.chroma_client import get_procedural_memory_collection

_SLOT_RE = re.compile(r"\{\{(\w+)\}\}")

# คำอธิบาย action type เป็นภาษาไทยสั้นๆ สำหรับ render_steps_as_plan_text() — ให้ template
# ที่มาจาก procedural memory โชว์เป็นแผนอ่านง่ายแบบเดียวกับที่ LLM ร่างสดๆ ทุกประการ
# (ดู llm.py::_PLAN_PROMPT_TEMPLATE) ไม่ต้องแก้อะไรฝั่ง frontend เลย
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
    """แทน {{slot_name}} ด้วยค่าจริงจาก slot_values — placeholder ที่ไม่มีใน slot_values
    (Planner ไม่ได้เติมมาให้ หรือพิมพ์ชื่อ slot ผิด) ปล่อยไว้เป็น literal เดิม ({{...}})
    ให้เห็นชัดว่ายังไม่ถูกแทนค่า แทนที่จะแอบหายไปเงียบๆ"""
    return _SLOT_RE.sub(lambda m: str(slot_values.get(m.group(1), m.group(0))), text)


def _step_signature(steps: list[dict]) -> str:
    """ลำดับ action type ล้วนๆ ของ steps (ไม่รวม locator/value) — ใช้เทียบว่า 2 template
    เป็น "ขั้นตอนแบบเดียวกัน" ไหม (ดู module docstring ส่วน Versioning ด้านบน)"""
    return "/".join(str(step.get("action", "")) for step in steps)


def _slot_names(slots: list) -> tuple:
    """ชื่อ slot ทั้งหมด เรียงตามตัวอักษร (ไม่สนลำดับที่ปรากฏ) — รวมเป็นส่วนหนึ่งของ
    signature เพราะ action sequence เดียวกันเป๊ะแต่จำนวน/ชื่อ slot ต่างกัน (เช่น เพิ่ม
    field ใหม่) ถือว่าเป็นคนละ "รุ่น" ของ template ที่ควรบันทึกแยก version ให้เห็นชัด"""
    names = []
    for slot in slots or []:
        name = slot.get("name") if isinstance(slot, dict) else slot
        if name:
            names.append(str(name))
    return tuple(sorted(names))


def _best_match(domain: str, goal_pattern: str) -> Optional[tuple[str, float]]:
    """คืน (intent_key, distance) ของ document ที่ใกล้เคียงที่สุดในโดเมนนี้ หรือ None ถ้า
    โดเมนนี้ไม่มี document เลย (เหมือน plan_memory.py::_best_match ทุกประการ แค่คนละ
    collection)"""
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
    """บันทึก template ที่ Abstractor สร้างเสร็จหลัง task สำเร็จ (เรียกจาก
    orchestrator.py หลัง finish_task(success=True) เท่านั้น — ดู module docstring หัวไฟล์
    llm.py::abstract_trajectory()) template ต้องมี goal_pattern/url_pattern/steps/slots
    ตรงตาม schema ของ ABSTRACTOR_TOOL (llm.py) — คืน None เงียบๆ ถ้าขาด goal_pattern/
    steps หรือ goal_pattern ใช้สคริปต์ที่ embedding ไม่รองรับ (ดู
    plan_memory._uses_unsupported_script) หรือ error ระหว่างทางไม่ว่ากรณีใด (ห้าม throw
    ให้ orchestrator loop ที่กำลังจะ return ผลลัพธ์ของ task ที่เพิ่งสำเร็จพังตาม)"""
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
        # W_procmem versioning (ดู module docstring): reuse ระยะห่างเดียวกับ
        # plan_memory_max_distance — เป็น embedding model ตัวเดียวกันเป๊ะ ค่าที่คาลิเบรต
        # ไว้แล้วยังใช้ได้ตรงๆ ไม่ต้องคาลิเบรตใหม่แยกต่างหาก
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
    """W_procmem Step 1: ดึง top-K candidate template ที่ใกล้เคียงกับ goal นี้ที่สุดใน
    โดเมนนี้ (ดึงแบบหลวมๆ ไม่มี threshold ตัดสินใจเด็ดขาดแบบ
    plan_memory.find_matching_plan — ปล่อยให้ llm.plan_with_procedural_memory()
    ตัดสินใจ reuse/adapt/plan_fresh เองจาก candidate list นี้) คืน [] เงียบๆ ถ้าไม่มี
    candidate เลย/goal ใช้สคริปต์ที่ embedding ไม่รองรับ/error ระหว่างทาง"""
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
                # ACC-1 (accuracy audit follow-up): track record จริงของ template นี้ (ดู
                # record_template_outcome() ด้านล่าง) — เดิม dict นี้ไม่มี field พวกนี้เลย
                # ทำให้ llm.plan_with_procedural_memory() (ผู้เรียกฟังก์ชันนี้) ไม่มีทาง
                # แยกแยะ template ที่เพิ่งถูกบันทึกครั้งเดียว (success_count=1, ยังไม่เคย
                # ถูก verify ซ้ำ) ออกจาก template ที่ reuse สำเร็จมาแล้วหลายครั้งจริง —
                # ประเมิน confidence จากแค่ semantic/pattern match ล้วนๆ เหมือนกันหมด
                "success_count": meta.get("success_count", 0),
                "failure_count": meta.get("failure_count", 0),
            })
        return candidates
    except Exception as e:
        print(f"⚠️ Procedural Memory find_candidate_templates error: {e}", flush=True)
        return []


def record_template_outcome(template_id: str, success: bool) -> None:
    """อัปเดต success_count/failure_count/last_used_at หลัง fast-path run ที่ใช้
    template_id นี้จบ (ไม่ว่าจะจบด้วยตัวเองหรือผ่าน Repair สำเร็จก็ยังนับเป็น success —
    ดู fastpath_executor.py) — ไม่มีผลอะไรถ้าไม่เจอ template_id นี้แล้ว (เช่นถูกลบ/error
    ระหว่างบันทึกครั้งก่อน) เงียบๆ ไม่ throw"""
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
    """แปลง steps ที่มีโครงสร้าง (+ แทนค่า slot แล้ว) เป็นข้อความแผนรูปแบบเดียวกับที่
    llm.py::_PLAN_PROMPT_TEMPLATE บังคับ LLM ตอบอยู่แล้ว ("1. ...\\n2. ...") — ให้
    index.html::parsePlanSteps()/renderPlanPreviewList() ใช้ได้ตรงๆ โดยไม่ต้องแก้อะไร
    ฝั่ง frontend เลย ไม่ว่าแผนจะมาจาก procedural memory หรือ LLM ร่างสดๆ

    step ที่ sensitive=True (เช่น กรอกรหัสผ่าน) จะไม่โชว์ค่าจริงเด็ดขาด แสดงเป็น
    "••••••" แทนเสมอ ไม่ว่า slot_values จะมีค่าจริงส่งมาด้วยหรือไม่ก็ตาม"""
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
    """ใช้ patch (op: replace/insert/remove, index, step?) จาก
    llm.plan_with_procedural_memory()'s decision="adapt" กับ steps ของ candidate ที่
    Planner เลือกไว้ — เรียกจาก routes.py::generate_plan ก่อนส่ง final_steps กลับไปให้
    user review คืน list ใหม่เสมอ (ไม่แก้ steps เดิมของ candidate ที่อาจถูกอ้างอิงที่อื่น
    อยู่ด้วย) op ที่ index ไม่ถูกต้อง (นอกช่วง/ไม่ใช่ int) หรือไม่มี "step" มาด้วยตอน
    replace/insert จะถูกข้ามเงียบๆ ทีละ op (ไม่ throw ทั้งฟังก์ชัน — LLM อาจส่ง index ผิด
    มาได้ ดีกว่าทำให้ endpoint ทั้งตัวพัง)"""
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

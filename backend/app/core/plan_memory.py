"""core/plan_memory.py — W20: Plan Memory ("ทำครั้งแรกให้ AI คิด ครั้งต่อไปให้จำ") แทนที่
core/plan_store.py (W19) ทั้งระบบ — เดิม plan_store.py จับคู่ (domain, goal) แบบ exact
text match ล้วนๆ (แค่ strip/lowercase/ยุบช่องว่าง) "Login" กับ "Sign in" ถือเป็นคนละ goal
กันทันที ทั้งที่เป็นเจตนาเดียวกัน — ตัวนี้ใช้ semantic search (ChromaDB, embedding function
เดียวกับคู่มือ/long-term memory) แทน จับคู่ตาม "เจตนา" ไม่ใช่ตัวอักษร

Data model: ทุก document ในนี้คือ "1 version ของ 1 lineage" — lineage หนึ่ง (ระบุด้วย
intent_key ที่สุ่มขึ้นครั้งแรกที่พบ intent นี้) มีได้หลาย version สะสมไว้ตลอด (ไม่เคยลบทิ้ง
— ดู "Plan Versioning" ใน requirement) แต่ละ document เก็บ:
    domain, intent_key, version (int), status ("approved" เท่านั้น — ห้ามมี draft/
    rejected ปนอยู่ในนี้เด็ดขาด), created_by ("user" เสมอ), created_at, goal (ข้อความ
    ที่ user พิมพ์ตอน confirm ครั้งนั้น — ใช้เป็น embedding document ด้วย), plan (เนื้อหา
    แผนแบบ plain text เต็ม)

หา lineage ที่ตรงกันด้วย semantic search ต่อ (domain, goal ใหม่) ก่อนเสมอ ทั้งตอนจะ "หา
แผนมาใช้" (find_matching_plan, เรียกจาก routes.py::generate_plan) และตอนจะ "บันทึก
แผนที่เพิ่ง confirm" (save_confirmed_plan, เรียกจาก routes.py::execute_plan) — เกณฑ์
เดียวกัน (settings.plan_memory_max_distance) ทำให้แผนของ intent เดียวกันสะสม version
ไปเรื่อยๆ ใน lineage เดิม แทนที่จะกลายเป็น lineage ใหม่ทุกครั้งที่ user พิมพ์ถ้อยคำต่างไป
เล็กน้อย

ห้าม throw ออกไปให้ endpoint พังเด็ดขาด (กฎเดียวกับ retriever.py/long_term_memory.py) —
Plan Memory เป็นแค่ enhancement (ประหยัดการเรียก LLM) ไม่ใช่ requirement ที่ต้องมีถึงจะ
ทำงานได้ ถ้า Chroma ล่ม/error ระหว่างทาง ต้อง fallback เงียบๆ (find_matching_plan คืน
None ให้ generate_plan ไปร่างจาก LLM ตามปกติ, save_confirmed_plan แค่ไม่ได้บันทึกอะไร
task ที่กำลังจะรันก็ยังรันต่อได้ปกติ)
"""

import time
import uuid
from typing import Optional

from backend.app.config import settings
from backend.app.rag.chroma_client import get_plan_memory_collection


def _best_match(domain: str, goal: str) -> Optional[tuple[str, float]]:
    """คืน (intent_key, distance) ของ document ที่ใกล้เคียงที่สุดในโดเมนนี้ (ทุก document
    ในนี้เป็น status="approved" อยู่แล้วเสมอ ไม่ต้อง filter status ซ้ำ) คืน None ถ้าโดเมน
    นี้ไม่มี document เลย"""
    collection = get_plan_memory_collection()
    results = collection.query(query_texts=[goal], n_results=1, where={"domain": domain})
    ids = results.get("ids") or [[]]
    if not ids or not ids[0]:
        return None
    metadata = results["metadatas"][0][0]
    distance = results["distances"][0][0]
    return metadata["intent_key"], distance


def _latest_version(domain: str, intent_key: str) -> Optional[dict]:
    """คืน metadata ของ version ล่าสุด (เลข version มากสุด) ของ lineage นี้ หรือ None ถ้า
    ไม่มี document เลย (ไม่ควรเกิดถ้า _best_match() เพิ่งเจอ intent_key นี้มาเอง แต่กันไว้
    เผื่อ race กับ _best_effort เขียนพร้อมกัน)"""
    collection = get_plan_memory_collection()
    got = collection.get(where={"$and": [{"domain": domain}, {"intent_key": intent_key}]})
    metadatas = got.get("metadatas") or []
    if not metadatas:
        return None
    return max(metadatas, key=lambda m: m["version"])


def find_matching_plan(domain: str, goal: str) -> Optional[dict]:
    """W20 Step 1: หา approved plan ที่ตรงกับ goal นี้มากที่สุด (semantic ไม่ใช่ exact text
    — ดู module docstring สำหรับตัวเลข distance จริงที่ใช้คาลิเบรต threshold) คืน dict
    {intent_key, version, plan, distance} ของ version ล่าสุดของ lineage ที่ match ถ้า
    distance อยู่ในเกณฑ์ (settings.plan_memory_max_distance) คืน None ถ้าไม่เจอ/ไม่ตรงพอ/
    error ระหว่างทาง — ให้ caller (routes.py::generate_plan) fallback ไปให้ LLM ร่างใหม่
    เอง (ตรงตาม Plan Priority: user-approved ก่อนเสมอ, LLM เป็นแค่ fallback ตอนไม่มี
    lineage ไหนตรงพอ)"""
    try:
        match = _best_match(domain, goal)
        if match is None:
            return None
        intent_key, distance = match
        if distance > settings.plan_memory_max_distance:
            return None
        version_meta = _latest_version(domain, intent_key)
        if version_meta is None:
            return None
        return {
            "intent_key": intent_key,
            "version": version_meta["version"],
            "plan": version_meta["plan"],
            "distance": distance,
        }
    except Exception as e:
        print(f"⚠️ Plan Memory find_matching_plan error: {e}", flush=True)
        return None


def save_confirmed_plan(domain: str, goal: str, plan: str) -> Optional[dict]:
    """บันทึกแผนที่ user "Confirm" แล้วเท่านั้น — เรียกจาก routes.py::execute_plan() ทุก
    ครั้งที่ task เริ่มจริง ไม่ว่า user จะแก้ไขข้อความแผนมาก่อนหรือไม่ก็ตาม (draft ที่ยังไม่
    confirm/แผนที่ user cancel ไม่มีทางเรียกฟังก์ชันนี้เลย — cancelPlan() ฝั่ง frontend ไม่
    เคยยิง request ออกไป ดู index.html)

    หา lineage ที่ตรงกันก่อนเสมอ (เกณฑ์เดียวกับ find_matching_plan()):
      - เจอ lineage เดิม: ถ้าเนื้อหาแผนเหมือน version ล่าสุดเป๊ะ (user confirm โดยไม่ได้
        แก้อะไรเลยจากแผนที่โหลดมาจาก Plan Memory เดิม) ไม่สร้าง version ซ้ำซ้อนเปล่าๆ คืน
        version เดิมตรงๆ (created=False) — สร้าง version ใหม่ (ล่าสุด+1) เฉพาะตอนเนื้อหา
        ต่างจริง (ตรงตาม Editing Behavior: แก้ไข = canonical version ใหม่)
      - ไม่เจอ lineage ไหนตรงพอ: เป็น intent ใหม่จริง (intent_key สุ่มใหม่, version=1)
    คืน None เงียบๆ ถ้า error ระหว่างทาง (ไม่ throw — ห้ามทำให้ execute_plan ทั้ง endpoint
    พังแค่เพราะบันทึกความจำไม่สำเร็จ, task ที่กำลังจะรันต้องรันต่อได้ปกติเสมอ)"""
    try:
        match = _best_match(domain, goal)
        if match is not None and match[1] <= settings.plan_memory_max_distance:
            intent_key = match[0]
            latest = _latest_version(domain, intent_key)
            if latest is not None and latest["plan"] == plan:
                return {"intent_key": intent_key, "version": latest["version"], "plan": plan, "created": False}
            new_version = (latest["version"] + 1) if latest is not None else 1
        else:
            intent_key = str(uuid.uuid4())
            new_version = 1

        collection = get_plan_memory_collection()
        collection.add(
            documents=[goal],
            metadatas=[{
                "domain": domain,
                "intent_key": intent_key,
                "version": new_version,
                "status": "approved",
                "created_by": "user",
                "created_at": time.time(),
                "goal": goal,
                "plan": plan,
            }],
            ids=[str(uuid.uuid4())],
        )
        return {"intent_key": intent_key, "version": new_version, "plan": plan, "created": True}
    except Exception as e:
        print(f"⚠️ Plan Memory save_confirmed_plan error: {e}", flush=True)
        return None

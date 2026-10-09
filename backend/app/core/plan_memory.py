"""core/plan_memory.py — W20: Plan Memory ("ครั้งแรกให้ AI คิด ครั้งต่อไปให้จำ") แทน plan_store.py (W19)
ที่จับคู่ goal แบบ exact text ("Login" กับ "Sign in" เป็นคนละ goal) — ตัวนี้ใช้ semantic search (ChromaDB)

แต่ละ document = 1 version ของ 1 lineage (intent_key สุ่มครั้งแรก, ไม่เคยลบ) metadata: domain,
intent_key, version, status ("approved" เท่านั้น), created_by ("user"), created_at, goal (ใช้เป็น
embedding document ด้วย), plan ทั้ง find_matching_plan และ save_confirmed_plan ใช้เกณฑ์เดียวกัน
(settings.plan_memory_max_distance) ให้ intent เดียวกันสะสม version ใน lineage เดิม

ห้าม throw ออกไปให้ endpoint พัง — error ใดๆ fallback เงียบๆ (find คืน None ให้ LLM ร่างใหม่, save ไม่บันทึก)
"""

import difflib
import re
import time
import uuid
from typing import Optional

from backend.app.config import settings
from backend.app.core import goal_intent
from backend.app.rag.chroma_client import get_plan_memory_collection

# W21 (re-applied; บั๊กจริง: goal "ซื้อของทั้งหมด...จ่ายเงิน" ได้แผน "ไปที่หน้าเข้าสู่ระบบ"): embedding
# default เป็น English-only สคริปต์ที่ไม่รู้จัก (ไทย/จีน/ญี่ปุ่น/เกาหลี/อาหรับ ฯลฯ) ยุบเป็น embedding
# แทบเดียวกัน (cosine ≈ 0) จึงไม่เชื่อ semantic match ของสคริปต์กลุ่มนี้เลย ทั้ง find และ save
# (T1: ช่วงอักษรย้ายไป goal_intent.UNSUPPORTED_SCRIPT_RE; T3: ยังจับคู่ด้วย intent key ตายตัวได้)


def _uses_unsupported_script(text: str) -> bool:
    """True ถ้ามีตัวอักษรจากสคริปต์ที่ embedding ไม่รองรับ (W21 ด้านบน)"""
    return bool(goal_intent.UNSUPPORTED_SCRIPT_RE.search(text))


# W_planvalue (บั๊กจริง): "edit role Cody55 to admin" แล้วพิมพ์ "edit role username::gfgfgf to admin"
# embedding ใกล้กันมาก จึงคืนแผนเก่าที่ฝัง "Cody55" ให้ user review/อนุมัติผิดเป้าหมาย — ประโยคเกือบ
# เหมือนเดิมต่างแค่ค่าเฉพาะ 1 จุด (username/ID) ต้องไม่ reuse; ถ้อยคำต่างทั้งประโยค ("Login" vs
# "Sign in") ratio ต่ำ ไม่เข้าเงื่อนไขนี้ ยัง reuse ได้ตามปกติ
def _is_value_substitution_only(old_goal: str, new_goal: str) -> bool:
    """True ถ้าสองประโยคเหมือนกันแทบทุกคำ ต่างแค่ช่วงเดียว <=2 token (ค่าเป้าหมายเปลี่ยน) — คืน
    False ถ้าฝั่งใดว่าง (lineage เก่าก่อนมี field goal)"""
    if not old_goal or not new_goal:
        return False
    old_tokens = old_goal.split()
    new_tokens = new_goal.split()
    sm = difflib.SequenceMatcher(None, old_tokens, new_tokens)
    if sm.ratio() < 0.6:
        return False
    replaced = [op for op in sm.get_opcodes() if op[0] == "replace"]
    if len(replaced) != 1:
        return False
    _, i1, i2, j1, j2 = replaced[0]
    return (i2 - i1) <= 2 and (j2 - j1) <= 2


def _best_match(domain: str, goal: str) -> Optional[tuple[str, float]]:
    """คืน (intent_key, distance) ของ document ที่ใกล้ที่สุดในโดเมน (ทุกตัว approved อยู่แล้ว) หรือ None"""
    collection = get_plan_memory_collection()
    results = collection.query(query_texts=[goal], n_results=1, where={"domain": domain})
    ids = results.get("ids") or [[]]
    if not ids or not ids[0]:
        return None
    metadata = results["metadatas"][0][0]
    distance = results["distances"][0][0]
    return metadata["intent_key"], distance


def _latest_version(domain: str, intent_key: str) -> Optional[dict]:
    """คืน metadata ของ version ล่าสุดของ lineage นี้ หรือ None (กัน race กับการเขียนพร้อมกัน)"""
    collection = get_plan_memory_collection()
    got = collection.get(where={"$and": [{"domain": domain}, {"intent_key": intent_key}]})
    metadatas = got.get("metadatas") or []
    if not metadatas:
        return None
    return max(metadatas, key=lambda m: m["version"])


def _find_by_intent_key(domain: str, goal: str) -> Optional[dict]:
    """T3: หา lineage ด้วยกุญแจที่ถอดจาก intent ไม่ผ่าน embedding (สำหรับสคริปต์ที่ไม่รองรับ)
    distance=0.0 หมายถึงตรงแบบตายตัว ไม่ใช่ผล semantic; never raises"""
    intent_key = goal_intent.plan_memory_intent_key(goal)
    if intent_key is None:
        return None
    try:
        version_meta = _latest_version(domain, intent_key)
        if version_meta is None:
            return None
        return {
            "intent_key": intent_key,
            "version": version_meta["version"],
            "plan": version_meta["plan"],
            "distance": 0.0,
        }
    except Exception as e:
        print(f"⚠️ Plan Memory _find_by_intent_key error: {e}", flush=True)
        return None


def find_matching_plan(domain: str, goal: str) -> Optional[dict]:
    """W20 Step 1: คืน {intent_key, version, plan, distance} ของ version ล่าสุดของ lineage ที่ใกล้ที่สุด
    ภายใน settings.plan_memory_max_distance หรือ None (ไม่เจอ/ไม่ตรงพอ/error) ให้ generate_plan
    fallback ไป LLM — never raises"""
    if _uses_unsupported_script(goal):
        # T3: เดิมยอมแพ้เสมอ งานภาษาไทยไม่เคยได้ reuse — ไม่เชื่อ semantic distance แต่ยังจับคู่ด้วย
        # intent key ตายตัว (goal_intent.plan_memory_intent_key(); None = พฤติกรรมเดิม)
        return _find_by_intent_key(domain, goal)
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
        if _is_value_substitution_only(version_meta.get("goal", ""), goal):
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
    """บันทึกแผนที่ user Confirm แล้ว (เรียกจาก routes.py::execute_plan) คืน
    {intent_key, version, plan, created}:
      - เจอ lineage เดิม + แผนเหมือน version ล่าสุดเป๊ะ -> คืน version เดิม (created=False)
      - เจอ lineage เดิม + แผนต่าง -> version ล่าสุด+1
      - ไม่เจอ -> intent_key ใหม่ version=1
    คืน None ถ้า error — never raises (task ต้องรันต่อได้เสมอ)"""
    forced_intent_key = None
    if _uses_unsupported_script(goal):
        # T3: บันทึกได้เฉพาะเมื่อถอด intent key ได้ (find/save ของ goal กลุ่มนี้ใช้ where filter ไม่ใช่ embedding)
        forced_intent_key = goal_intent.plan_memory_intent_key(goal)
        if forced_intent_key is None:
            return None
    try:
        match = None if forced_intent_key else _best_match(domain, goal)
        intent_key = forced_intent_key
        new_version = 1
        if forced_intent_key is not None:
            latest = _latest_version(domain, forced_intent_key)
            if latest is not None and latest["plan"] == plan:
                return {"intent_key": forced_intent_key, "version": latest["version"], "plan": plan, "created": False}
            new_version = (latest["version"] + 1) if latest is not None else 1
        if match is not None and match[1] <= settings.plan_memory_max_distance:
            candidate_key = match[0]
            latest = _latest_version(domain, candidate_key)
            # W_planvalue: ต่างแค่ค่าเป้าหมาย 1 จุด -> แยก lineage ใหม่ ไม่งั้น find ครั้งหน้ายัง reuse แผนผิดเป้า
            if latest is not None and _is_value_substitution_only(latest.get("goal", ""), goal):
                latest = None
            else:
                intent_key = candidate_key
                if latest is not None and latest["plan"] == plan:
                    return {"intent_key": intent_key, "version": latest["version"], "plan": plan, "created": False}
                new_version = (latest["version"] + 1) if latest is not None else 1
        if intent_key is None:
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

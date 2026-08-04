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

import difflib
import re
import time
import uuid
from typing import Optional

from backend.app.config import settings
from backend.app.rag.chroma_client import get_plan_memory_collection

# W21 (re-applied — เคยแก้ไปแล้วรอบหนึ่งแต่ไฟล์นี้กลับไปเป็นเวอร์ชันก่อนแก้โดยไม่ทราบสาเหตุ
# แน่ชัด — ดูเหตุผลที่คุยกันตอนเจอบั๊กจริง: goal "ซื้อของทั้งหมด...จ่ายเงิน" ดันได้แผนของ
# goal เก่าคนละเรื่องเลย "ไปที่หน้าเข้าสู่ระบบ" กลับมา): embedding function ที่ใช้ทั่ว
# ทั้งแอป (chromadb DefaultEmbeddingFunction, ดู rag/chroma_client.py) เป็น English-only
# tokenizer — พิสูจน์แล้วว่า text ที่เป็นสคริปต์ที่โมเดลไม่รู้จัก (ไทย, จีน, ญี่ปุ่น,
# เกาหลี, อาหรับ ฯลฯ) จะยุบเหลือ embedding ที่แทบเหมือนกันหมดไม่ว่าความหมายจริงจะต่างกันแค่
# ไหน (cosine distance ≈ 0 ระหว่าง goal ภาษาไทยสองอันที่ไม่เกี่ยวกันเลย) ผลคือ goal
# ภาษาไทยใดๆ จะ "match" กับ lineage แรกที่เคยบันทึกไว้ในโดเมนนี้เสมอ ไม่ว่าเจตนาจะตรงกัน
# จริงหรือไม่ — ก่อนจะมี multilingual embedding model จริง ทางที่ปลอดภัยที่สุดคือ "ไม่เชื่อ"
# semantic match ของ goal ที่ใช้สคริปต์กลุ่มนี้เลย ข้าม Plan Memory ไปทั้ง find และ save
# (บันทึกไปก็ค้นไม่เจอถูกต้องอยู่ดี แถมกลายเป็น lineage ที่จะไป false-match กับ goal
# ภาษาเดียวกันตัวอื่นๆ ในอนาคตซ้ำอีก) fallback ไปให้ LLM ร่างใหม่ทุกครั้งแทน (เหมือนไม่มี
# Plan Memory เลยสำหรับภาษากลุ่มนี้ — งานพัง 0 ครั้ง ดีกว่าประหยัด LLM call แล้วได้แผนผิด
# เจตนา)
_UNSUPPORTED_SCRIPT_RE = re.compile(
    "["
    "฀-๿"  # Thai
    "一-鿿"  # CJK Unified Ideographs (Chinese)
    "぀-ヿ"  # Hiragana/Katakana (Japanese)
    "가-힣"  # Hangul (Korean)
    "؀-ۿ"  # Arabic
    "Ѐ-ӿ"  # Cyrillic
    "]"
)


def _uses_unsupported_script(text: str) -> bool:
    """True ถ้า goal มีตัวอักษรจากสคริปต์ที่ embedding model ปัจจุบันไม่รองรับจริง (ดู
    comment ด้านบน) — ใช้เช็คก่อนทั้ง find_matching_plan และ save_confirmed_plan"""
    return bool(_UNSUPPORTED_SCRIPT_RE.search(text))


# W_planvalue: บั๊กจริงที่ user เจอ — goal เดิมที่เคย confirm ไปแล้ว "edit role Cody55 to
# admin" ถูกบันทึกเป็น lineage หนึ่ง พอ user พิมพ์ goal ใหม่ "edit role username::gfgfgf
# to admin" (เปลี่ยนแค่ username เป้าหมาย ประโยคที่เหลือเหมือนเดิมทุกคำ) embedding
# distance ระหว่างสองประโยคนี้ใกล้กันมาก (ต่างกันแค่ 1 token ท่ามกลางคำเดิมทั้งหมด) เลย
# "match" แล้วคืนแผนเก่าที่มีคำว่า "Cody55" ฝังอยู่ในทุก step ตรงๆ กลับไปให้ user เห็นตอน
# review plan (แม้ execution จริงจะ grounded กับ goal สดใหม่ ไม่ได้พังจริง แต่ plan ที่โชว์
# ให้ user "อนุมัติ" ผิดเป้าหมายไปเลย — ทำลายจุดประสงค์ของการให้ review ก่อน) ปัญหานี้
# เฉพาะเจาะจงกว่าเคส _UNSUPPORTED_SCRIPT_RE ด้านบน (ข้ามภาษาทั้งประโยค) — ตรงนี้คือ "ประโยค
# แทบจะเหมือนกันเป๊ะ ต่างกันแค่คำ/ช่วงสั้นๆ 1 จุด" ซึ่งมักจะเป็น "ค่าเฉพาะเจาะจง" (username/
# ID) ที่เปลี่ยนไป ไม่ใช่แค่ถ้อยคำที่ต่างกันแบบ "Login" vs "Sign in" (ต่างกันเกือบทั้งประโยค
# ratio ต่ำ ไม่เข้าเงื่อนไขนี้ ปล่อยให้ semantic matching ทำงานตามปกติ เพราะนั่นคือ use case
# ที่ Plan Memory ถูกออกแบบมาให้ reuse ได้จริงๆ)
def _is_value_substitution_only(old_goal: str, new_goal: str) -> bool:
    """True ถ้า old_goal (ที่ผูกกับแผนที่ semantic match เจอ) กับ new_goal (ที่ user พิมพ์
    ตอนนี้) เป็น "ประโยคแม่แบบ" เดียวกันแทบทุกคำ ต่างกันแค่ token สั้นๆ ช่วงเดียว (<=2 token)
    — เป็นสัญญาณว่า "เป้าหมายเฉพาะเจาะจง" (เช่นชื่อ user ที่จะแก้ไข) เปลี่ยนไปจริง แม้
    embedding distance จะยังใกล้กันมากก็ตาม ต้องปฏิเสธการ reuse แผนเดิม (คืน False ถ้า
    old_goal ว่างเปล่า — lineage เก่าที่บันทึกไว้ก่อนมี field นี้ ไม่มีอะไรให้เทียบ)"""
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
    if _uses_unsupported_script(goal):
        return None
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
    if _uses_unsupported_script(goal):
        return None
    try:
        match = _best_match(domain, goal)
        intent_key = None
        new_version = 1
        if match is not None and match[1] <= settings.plan_memory_max_distance:
            candidate_key = match[0]
            latest = _latest_version(domain, candidate_key)
            # W_planvalue (ดู _is_value_substitution_only ด้านบน): goal นี้ต่างจาก goal
            # เดิมของ lineage นี้แค่ "ค่าเฉพาะเจาะจง" 1 จุด (เช่น username เป้าหมาย) —
            # ต้องแยกเป็น lineage ใหม่ ไม่ใช่เพิ่ม version ให้ lineage เดิม ไม่งั้น
            # find_matching_plan() ครั้งถัดไปจะยังคง reuse แผนที่ผูกกับเป้าหมายที่เปลี่ยน
            # ไปแล้วอยู่ดี (แค่เลื่อนปัญหาไปอีก version หนึ่ง)
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

"""ChromaDB connection.

W1: skeleton — เชื่อมต่อและสร้าง/เปิด collection ได้.
W2: ออกแบบ schema จริงของ metadata (source, section, page, ฯลฯ).
W3: เปลี่ยนมาใช้ local embedding (all-MiniLM-L6-v2 ผ่าน ChromaDB DefaultEmbeddingFunction)
    แทน Gemini API — รันได้ offline ไม่ต้องมี API key, โหลดโมเดลครั้งแรก ~90MB
    (เก็บ cache ไว้ในเครื่อง ครั้งต่อไปไม่โหลดซ้ำ)
"""

import threading

import chromadb
from chromadb import Collection
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from backend.app.config import settings

# ทุก collection ต้องใช้ embedding function เดียวกันเสมอ (ingest กับ query ต้องมิติตรงกัน)
_embedding_function = DefaultEmbeddingFunction()

# W11[C]: cache client เป็น singleton ต่อ process (เดิม get_client() สร้าง
# chromadb.PersistentClient(...) ใหม่ทุกครั้งที่เรียก) — เจอบั๊กจริงระหว่างทดสอบ
# concurrency (run.py isolation/concurrency): ยิง 2 task พร้อมกันในโปรเซสเดียวกัน แต่ละ
# task เรียก retrieve() ทุก step ผ่าน asyncio.to_thread() พร้อมๆ กัน ทำให้เกิดการสร้าง
# PersistentClient หลายตัวชี้ไป path เดียวกัน (./data/chroma) พร้อมกันจากคนละ thread —
# เจอ error จริง ("'RustBindingsAPI' object has no attribute 'bindings'") หลุดออกมา
# (retrieve() ดักไว้แล้วคืน [] เสมอ ไม่ทำให้ task พัง แต่ silently เสีย manual context
# ของ step นั้นไปเฉยๆ — กระทบ RAG-based permission ของ W7[B] ได้ตรงๆ ถ้า query ที่ควรเจอ
# RULE-04 ดันพังจังหวะนี้พอดี) — cache instance เดียวใช้ซ้ำกันทุก call แก้ที่ต้นเหตุ
# (การเปิด connection ซ้อนกันหลายตัวพร้อมกัน) แทนที่จะพึ่ง try/except ปลายทางอย่างเดียว
#
# *** เจอรอบสองหลังใส่ cache ตัวแรกแล้ว: แค่ cache เฉยๆ (if _client is None: _client = ...)
# ยังไม่พอ เพราะ asyncio.to_thread() ใช้ thread pool จริง (ไม่ใช่แค่ event loop) — ถ้า 2
# task ยิง retrieve() ครั้ง "แรกสุด" ของ process พร้อมกันเป๊ะ ทั้งคู่เห็น `_client is None`
# เป็น True ก่อนที่อีกฝั่งจะทันเซ็ตค่า (classic check-then-set race ข้าม thread) เลยยังสร้าง
# PersistentClient ซ้อนกัน 2 ตัวได้เหมือนเดิม (ยืนยันจริงจาก run.py isolation ที่ error
# ยังโผล่อยู่แม้ใส่ cache แล้ว) — ต้องใส่ lock ครอบช่วง check-then-set ด้วยถึงจะปิด race
# ได้จริง ***
_client: chromadb.ClientAPI | None = None
_client_lock = threading.Lock()


def get_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:  # เช็คซ้ำในนี้ เผื่อ thread อื่นสร้างเสร็จไปแล้วระหว่างรอ lock
                _client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    return _client


def get_collection() -> Collection:
    client = get_client()
    return client.get_or_create_collection(
        name=settings.chroma_collection_name,
        embedding_function=_embedding_function,
    )


# W7[A]: collection แยกต่างหากจากคู่มือ (manuals) — เก็บ "ความจำ" ข้าม task run แทน
# เนื้อหาคู่มือที่ user ป้อน ใช้ client/embedding function เดียวกัน (persist_dir เดียวกัน
# แค่คนละ collection name) ดู backend/app/core/long_term_memory.py
def get_long_term_collection() -> Collection:
    client = get_client()
    return client.get_or_create_collection(
        name=settings.chroma_long_term_collection_name,
        embedding_function=_embedding_function,
    )


# W20: Plan Memory (ดู backend/app/core/plan_memory.py) — แผนที่ user "Confirm" แล้ว
# เก็บแยก collection ต่างหากจากทั้งคู่มือและ long-term memory ข้างบน ตั้ง hnsw:space เป็น
# "cosine" ตรงๆ (แทน default ของ chromadb ที่ไม่ใช่ cosine) เพราะ
# settings.plan_memory_max_distance ถูกคาลิเบรตไว้โดยสมมติว่าเป็น cosine distance
# เท่านั้น — ตั้งตอน get_or_create_collection() ครั้งแรกที่สร้าง collection เท่านั้น
# (เปลี่ยนทีหลังไม่ได้ ถ้าจะเปลี่ยนต้องลบ collection เดิมทิ้งแล้วสร้างใหม่)
def get_plan_memory_collection() -> Collection:
    client = get_client()
    return client.get_or_create_collection(
        name=settings.chroma_plan_memory_collection_name,
        embedding_function=_embedding_function,
        metadata={"hnsw:space": "cosine"},
    )

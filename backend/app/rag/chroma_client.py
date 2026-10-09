"""ChromaDB connection + collections (manuals, long-term memory, plan memory, procedural memory).

W3: ใช้ local embedding (all-MiniLM-L6-v2 ผ่าน DefaultEmbeddingFunction) แทน Gemini API —
    รัน offline ไม่ต้องมี API key, โหลดโมเดลครั้งแรก ~90MB แล้ว cache ไว้
"""

import threading

import chromadb
from chromadb import Collection
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from backend.app.config import settings

# ทุก collection ต้องใช้ embedding function เดียวกันเสมอ (ingest กับ query ต้องมิติตรงกัน)
_embedding_function = DefaultEmbeddingFunction()

# W11[C]: client singleton ต่อ process + lock ครอบ check-then-set — เดิมสร้าง PersistentClient ใหม่ทุก call
# 2 task พร้อมกัน (to_thread) เปิดซ้อน path เดียวกัน -> "'RustBindingsAPI' object has no attribute 'bindings'"
# (retrieve() คืน [] เงียบๆ เสีย manual context/RAG permission W7[B]); cache อย่างเดียวยังแข่งกันข้าม thread ได้ ต้องมี lock
_client: chromadb.ClientAPI | None = None
_client_lock = threading.Lock()


def get_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:  # เช็คซ้ำในนี้ เผื่อ thread อื่นสร้างเสร็จไปแล้วระหว่างรอ lock
                _client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    return _client


def _get_or_create(name: str, metadata: dict | None = None) -> Collection:
    kwargs = {"name": name, "embedding_function": _embedding_function}
    if metadata is not None:
        kwargs["metadata"] = metadata
    return get_client().get_or_create_collection(**kwargs)


def get_collection() -> Collection:
    return _get_or_create(settings.chroma_collection_name)


# W7[A]: collection แยกจากคู่มือ — เก็บ "ความจำ" ข้าม task run (ดู core/long_term_memory.py)
def get_long_term_collection() -> Collection:
    return _get_or_create(settings.chroma_long_term_collection_name)


# W20: Plan Memory (core/plan_memory.py) — hnsw:space="cosine" เพราะ plan_memory_max_distance คาลิเบรตเป็น
# cosine distance; ตั้งได้ตอนสร้าง collection ครั้งแรกเท่านั้น (เปลี่ยนต้องลบ collection แล้วสร้างใหม่)
def get_plan_memory_collection() -> Collection:
    return _get_or_create(settings.chroma_plan_memory_collection_name, {"hnsw:space": "cosine"})


# W_procmem: Procedural Memory (core/procedural_memory.py) — template steps+locator+slot แยกจาก plan_memory;
# ตั้ง cosine ตั้งแต่สร้าง (เปลี่ยนทีหลังไม่ได้) เผื่ออนาคตใช้ threshold gating แม้ตอนนี้แค่ดึง top-K ให้ Planner
def get_procedural_memory_collection() -> Collection:
    return _get_or_create(settings.chroma_procedural_memory_collection_name, {"hnsw:space": "cosine"})

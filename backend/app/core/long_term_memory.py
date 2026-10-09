"""W7[A]: long-term memory ข้าม task run (ShortTermMemory ใน memory.py อยู่แค่ใน 1 run)

1 task run = 1 document ใน get_long_term_collection() (แยกจากคู่มือของ retriever.py) เก็บ summary จาก
finish_task (อาจมีค่าที่หาเจอ เช่น ราคา/OTP) + action ที่ fail ให้ task ถัดไปเลี่ยง ดึงกลับด้วย semantic
search ต่อ goal+page_state ห้าม throw ออกไปให้ agent loop พัง (record_task/recall ดักทุก exception)

W23: เดิม collection เป็น global ข้าม session/site — session หนึ่งดึงความจำ (รวม action ที่ fail) ของอีก
session ไปใช้ (บั๊กจริง ซ้ำเติมด้วย embedding ที่ยุบภาษาไทย) ตอนนี้ recall() ต้องมี session_id (ไม่มี = คืน
[] ไม่ query) และกรองด้วย where={"session_id": ...} ที่ ChromaDB เอง
"""

import uuid
from typing import Optional

from backend.app.rag.chroma_client import get_long_term_collection


def record_task(
    url: str, goal: str, success: bool, message: str, failed_actions: str = "", session_id: str = "",
) -> None:
    """บันทึก 1 task run เป็น document ใหม่เสมอ (สะสมประวัติ ไม่ upsert) — session_id ว่างก็ยังบันทึก
    แต่ recall() จะหาไม่เจอ; never raises"""
    try:
        collection = get_long_term_collection()

        doc_lines = [
            f"URL: {url}",
            f"Goal: {goal}",
            f"Outcome: {'success' if success else 'failed'}",
            f"Summary: {message}",
        ]
        if failed_actions:
            doc_lines.append("Failed actions during this task (avoid repeating these):")
            doc_lines.append(failed_actions)
        document = "\n".join(doc_lines)

        collection.add(
            documents=[document],
            metadatas=[{"url": url, "goal": goal, "success": success, "session_id": session_id}],
            ids=[str(uuid.uuid4())],
        )
    except Exception as e:
        print(f"⚠️ Long-term Memory Record Error: {e}")


def recall(
    query: str, page_state: str = "", k: int = 3, session_id: str = "",
    query_embedding: Optional[list[float]] = None,
) -> list[str]:
    """ดึง document ของ run ก่อนหน้าที่เกี่ยวกับ goal+page_state ภายใน session เดียวกันเท่านั้น
    (session_id ว่าง = คืน [] ไม่ query) คืน [] ถ้า error/ไม่เจอ; never raises

    query_embedding: Speed 2.2 — embedding ที่ orchestrator คำนวณครั้งเดียวต่อ step ใช้ร่วมกับ
    retriever.retrieve() แทนการ embed ซ้ำ"""
    if not session_id:
        return []
    try:
        collection = get_long_term_collection()

        embed_input = f"{query}\n\nCurrent page:\n{page_state}" if page_state else query

        if query_embedding is not None:
            results = collection.query(
                query_embeddings=[query_embedding], n_results=k, where={"session_id": session_id},
            )
        else:
            results = collection.query(
                query_texts=[embed_input], n_results=k, where={"session_id": session_id},
            )

        documents = results.get("documents", [])
        if documents:
            return documents[0]
        return []
    except Exception as e:
        print(f"⚠️ Long-term Memory Recall Error: {e}")
        return []

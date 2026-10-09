"""Query คู่มือทุกครั้งที่ Planner วางแผน (เชื่อมกับ orchestrator ใน W6).

W3: ChromaDB embed query_texts เองด้วย local model — ต้องเป็น embedding function เดียวกับตอน ingest
    (ดู chroma_client.py)
"""

from typing import Optional

from backend.app.rag.chroma_client import get_collection


def retrieve(query: str, page_state: str = "", k: int = 5, query_embedding: Optional[list[float]] = None) -> list[str]:
    """คืน chunk คู่มือสูงสุด k ชิ้น — ไม่เคย raise (พังคืน [] เสมอ).

    page_state: ถ้ามีจะรวมเข้ากับ query ก่อน embed. ชื่อพารามิเตอร์ 'k' เป็นสัญญา ห้ามเปลี่ยน.
    query_embedding: Speed 2.2 — vector ที่ orchestrator embed ไว้แล้ว (ใช้ร่วมกับ
        long_term_memory.recall() กัน embed ซ้ำ 2 รอบต่อ step); None = ให้ ChromaDB embed เอง
    """
    try:
        collection = get_collection()
        embed_input = f"{query}\n\nCurrent page:\n{page_state}" if page_state else query

        if query_embedding is not None:
            results = collection.query(query_embeddings=[query_embedding], n_results=k)
        else:
            results = collection.query(query_texts=[embed_input], n_results=k)

        documents = results.get("documents", [])
        if documents:
            return documents[0]
        return []

    except Exception as e:
        # กฎเหล็กข้อที่ [2]: ห้าม throw error ออกมา ถ้าภายในพังให้ดักแล้วคืน [] เสมอ เพื่อไม่ให้ระบบ Agent พัง
        print(f"⚠️ RAG Retriever Error: {e}")
        return []

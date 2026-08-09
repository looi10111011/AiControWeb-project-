"""Query คู่มือทุกครั้งที่ Planner วางแผน (เชื่อมกับ orchestrator ใน W6).

W1: skeleton only. W3: Updated for Gemini Embedding v2 & Contract Compliance
W3: เปลี่ยนมาใช้ local embedding (all-MiniLM-L6-v2 ผ่าน ChromaDB) แทน Gemini API
    — ไม่ต้อง embed เองแล้ว ส่ง query_texts ให้ collection.query() embed ให้เอง
    (ต้อง embedding function เดียวกับตอน ingest เสมอ ดู chroma_client.py)
"""

from typing import Optional

from backend.app.rag.chroma_client import get_collection


def retrieve(query: str, page_state: str = "", k: int = 5, query_embedding: Optional[list[float]] = None) -> list[str]:
    """ดึงข้อมูล chunk จากคู่มือตามข้อตกลงสัญญา Contract

    Args:
        query (str): คำค้นหาเป้าหมาย
        page_state (str): โครงสร้างหน้าเว็บปัจจุบัน — ถ้ามีจะถูกรวมเข้ากับ query
            ก่อน embed เพื่อให้ผลลัพธ์ตรงกับบริบทหน้าเว็บปัจจุบันมากขึ้น
        k (int): จำนวน chunk สูงสุดที่ต้องการให้คืนค่า (ชื่อพารามิเตอร์ต้องเป็น 'k')
        query_embedding (Optional[list[float]]): Speed 2.2 — vector ที่ embed มาแล้วจาก
            ภายนอก (orchestrator.py คำนวณครั้งเดียวต่อ step ใช้ร่วมกับ
            long_term_memory.recall() แทนที่จะให้ embedding model รันซ้ำ 2 รอบต่อ step
            ทั้งที่ embed_input เหมือนกันเป๊ะ) ไม่ส่งมา (None, default) = พฤติกรรมเดิมทุก
            ประการ (ให้ ChromaDB embed จาก query_texts เอง) เผื่อที่อื่นเรียก retrieve()
            ตรงๆ โดยไม่มี embedding สำเร็จรูปมาให้
    """
    try:
        collection = get_collection()

        # รวม page_state เข้ากับ query ถ้ามี แล้วให้ collection.query() embed เอง
        # ด้วย local model (all-MiniLM-L6-v2) ตัวเดียวกับตอน ingest (ยกเว้นมี
        # query_embedding สำเร็จรูปมาให้แล้ว — ข้ามการ embed ซ้ำ)
        embed_input = f"{query}\n\nCurrent page:\n{page_state}" if page_state else query

        # ค้นหาข้อมูลใน ChromaDB และจำกัดผลลัพธ์ด้วยตัวแปร k
        if query_embedding is not None:
            results = collection.query(query_embeddings=[query_embedding], n_results=k)
        else:
            results = collection.query(
                query_texts=[embed_input],
                n_results=k  # นำค่า k ที่ได้รับจากฟังก์ชันมาส่งต่อให้ ChromaDB
            )

        # แกะข้อมูลส่งคืนกลับเป็น list[str]
        documents = results.get("documents", [])
        if documents and len(documents) > 0:
            return documents[0]

        return []

    except Exception as e:
        # กฎเหล็กข้อที่ [2]: ห้าม throw error ออกมา ถ้าภายในพังให้ดักแล้วคืน [] เสมอ เพื่อไม่ให้ระบบ Agent พัง
        print(f"⚠️ RAG Retriever Error: {e}")
        return []
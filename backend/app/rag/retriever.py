"""Query คู่มือทุกครั้งที่ Planner วางแผน (เชื่อมกับ orchestrator ใน W6).

W1: skeleton only. W3: Updated for Gemini Embedding v2 & Contract Compliance
W3: เปลี่ยนมาใช้ local embedding (all-MiniLM-L6-v2 ผ่าน ChromaDB) แทน Gemini API
    — ไม่ต้อง embed เองแล้ว ส่ง query_texts ให้ collection.query() embed ให้เอง
    (ต้อง embedding function เดียวกับตอน ingest เสมอ ดู chroma_client.py)
"""

from backend.app.rag.chroma_client import get_collection


def retrieve(query: str, page_state: str = "", k: int = 5) -> list[str]:
    """ดึงข้อมูล chunk จากคู่มือตามข้อตกลงสัญญา Contract

    Args:
        query (str): คำค้นหาเป้าหมาย
        page_state (str): โครงสร้างหน้าเว็บปัจจุบัน — ถ้ามีจะถูกรวมเข้ากับ query
            ก่อน embed เพื่อให้ผลลัพธ์ตรงกับบริบทหน้าเว็บปัจจุบันมากขึ้น
        k (int): จำนวน chunk สูงสุดที่ต้องการให้คืนค่า (ชื่อพารามิเตอร์ต้องเป็น 'k')
    """
    try:
        collection = get_collection()

        # รวม page_state เข้ากับ query ถ้ามี แล้วให้ collection.query() embed เอง
        # ด้วย local model (all-MiniLM-L6-v2) ตัวเดียวกับตอน ingest
        embed_input = f"{query}\n\nCurrent page:\n{page_state}" if page_state else query

        # ค้นหาข้อมูลใน ChromaDB และจำกัดผลลัพธ์ด้วยตัวแปร k
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
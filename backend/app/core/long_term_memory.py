"""W7[A]: long-term memory — จำ pattern/ข้อมูลข้าม task run (คนละเรื่องกับ
ShortTermMemory ใน memory.py ที่จำได้แค่ภายใน 1 task run เดียวแล้วหายไปพร้อม
Orchestrator instance)

เก็บสรุปผลลัพธ์ของแต่ละ task run เป็น 1 document ใน ChromaDB collection แยกต่างหาก
(get_long_term_collection() — ดู chroma_client.py) ใช้ embedding function เดียวกับ
คู่มือ (local all-MiniLM-L6-v2) แล้วดึงกลับมาด้วย semantic search ต่อ goal+page_state
ปัจจุบันเหมือน retriever.retrieve() ทุกประการ (ฟังก์ชันนี้จงใจแยกจาก retriever.py เพราะ
คนละ collection/domain: retriever.py = คู่มือที่ user ป้อน, ตัวนี้ = ประวัติที่ agent
สร้างเอง)

record_task() เก็บทั้ง 2 อย่างไว้ในเอกสารเดียวกัน:
1. ข้อความสรุปจาก finish_task (อาจมีค่าที่ AI หาเจอระหว่างทาง เช่น ราคา/OTP อยู่ในนั้น
   — ให้ task ถัดไปดึงมาใช้ได้)
2. action ที่ fail ระหว่างทาง (จาก ShortTermMemory.failed_actions_summary() ของ task
   นั้น — เตือน task ถัดไปให้เลี่ยง ไม่ต้องลองซ้ำสิ่งที่รู้อยู่แล้วว่าพัง/โดนบล็อก)

กฎเดียวกับ retriever.py: ห้าม throw ออกไปให้ agent loop พังเด็ดขาด ทั้ง record_task()
และ recall() ดักทุก exception เอง
"""

import uuid

from backend.app.rag.chroma_client import get_long_term_collection


def record_task(url: str, goal: str, success: bool, message: str, failed_actions: str = "") -> None:
    """บันทึกผลลัพธ์ของ 1 task run — เรียกครั้งเดียวตอนจบ run_task() แต่ละครั้ง
    (ไม่ upsert ทับ id เดิมเหมือน manual ingestion — ทุก task run คือ document ใหม่
    เสมอ เพราะต้องการสะสมประวัติหลายรอบไว้ ไม่ใช่แค่ค่าล่าสุด)"""
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
            metadatas=[{"url": url, "goal": goal, "success": success}],
            ids=[str(uuid.uuid4())],
        )
    except Exception as e:
        # กฎเดียวกับ retriever.py: ห้าม throw ออกไปให้ agent loop พังเด็ดขาด
        print(f"⚠️ Long-term Memory Record Error: {e}")


def recall(query: str, page_state: str = "", k: int = 3) -> list[str]:
    """ดึง document ของ task run ก่อนหน้าที่เกี่ยวข้องกับ goal+page ปัจจุบัน —
    รูปแบบเดียวกับ retriever.retrieve() ทุกประการ (รวม page_state เข้า query ก่อน
    embed, คืน [] เสมอถ้า error/ไม่มีอะไรตรง ไม่ throw)"""
    try:
        collection = get_long_term_collection()

        embed_input = f"{query}\n\nCurrent page:\n{page_state}" if page_state else query

        results = collection.query(query_texts=[embed_input], n_results=k)

        documents = results.get("documents", [])
        if documents and len(documents) > 0:
            return documents[0]

        return []

    except Exception as e:
        print(f"⚠️ Long-term Memory Recall Error: {e}")
        return []

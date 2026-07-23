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

W23: session_id เข้ามาแล้ว — เดิม collection นี้เป็น global เดียวข้าม session/site/user
ทั้งหมด ไม่มีการกรองใดๆ เลย ทำให้ session หนึ่งดึงความจำ (รวมถึง "action ที่เคย fail")
ของอีก session ที่ทำคนละเว็บ/คนละเจตนาไปใช้ได้ (เจอบั๊กจริงคู่กับ plan_memory.py — เพราะ
ใช้ embedding function ตัวเดียวกันที่ยุบ text ภาษาไทยให้ใกล้เคียงกันหมด ยิ่งซ้ำเติมปัญหา
นี้ให้เกิดง่ายขึ้นไปอีก) — recall() ตอนนี้ "ต้อง" มี session_id ถึงจะคืนอะไรเลย (ไม่มี =
คืน [] เงียบๆ ทันที ไม่ query เลยด้วยซ้ำ ปลอดภัยไว้ก่อนดีกว่าเสี่ยง unscoped query) กรอง
ด้วย where={"session_id": ...} ตรงๆ ที่ ChromaDB เอง (ไม่ใช่กรองทีหลังในโค้ด) การันตีว่า
document ที่คืนมาเป็นของ session นี้เท่านั้นจริงๆ ไม่มีทางหลุดข้าม session ได้เลย
"""

import uuid

from backend.app.rag.chroma_client import get_long_term_collection


def record_task(
    url: str, goal: str, success: bool, message: str, failed_actions: str = "", session_id: str = "",
) -> None:
    """บันทึกผลลัพธ์ของ 1 task run — เรียกครั้งเดียวตอนจบ run_task() แต่ละครั้ง
    (ไม่ upsert ทับ id เดิมเหมือน manual ingestion — ทุก task run คือ document ใหม่
    เสมอ เพราะต้องการสะสมประวัติหลายรอบไว้ ไม่ใช่แค่ค่าล่าสุด)

    session_id: ผูก document นี้เข้ากับ session ที่สร้างมันขึ้นมา ให้ recall() กรองกลับ
    มาได้เฉพาะของ session เดียวกันเท่านั้น (ดู module docstring) ว่างเปล่าได้ (เช่น task
    ที่ไม่มี session concept เลย) — document ยังถูกบันทึกอยู่ (ไม่เสียประวัติ) แค่จะไม่มี
    ทาง recall กลับมาเจอได้อีกเลยในทางปฏิบัติ เพราะ recall() ปฏิเสธ query แบบไม่มี
    session_id ไปตั้งแต่ต้น"""
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
        # กฎเดียวกับ retriever.py: ห้าม throw ออกไปให้ agent loop พังเด็ดขาด
        print(f"⚠️ Long-term Memory Record Error: {e}")


def recall(query: str, page_state: str = "", k: int = 3, session_id: str = "") -> list[str]:
    """ดึง document ของ task run ก่อนหน้าที่เกี่ยวข้องกับ goal+page ปัจจุบัน "ภายใน
    session เดียวกันเท่านั้น" (session_id ว่างเปล่า = ไม่มี session context ให้ scope
    ปลอดภัยได้ คืน [] ทันทีโดยไม่ query เลย — ดู module docstring) กรองด้วย
    where={"session_id": ...} ที่ ChromaDB ตรงๆ ก่อน semantic search เสมอ ไม่ใช่กรอง
    ทีหลังในโค้ด (รูปแบบเดียวกับ retriever.retrieve() ทุกประการนอกจากนี้ — รวม page_state
    เข้า query ก่อน embed, คืน [] เสมอถ้า error/ไม่มีอะไรตรง ไม่ throw)"""
    if not session_id:
        return []
    try:
        collection = get_long_term_collection()

        embed_input = f"{query}\n\nCurrent page:\n{page_state}" if page_state else query

        results = collection.query(
            query_texts=[embed_input], n_results=k, where={"session_id": session_id},
        )

        documents = results.get("documents", [])
        if documents and len(documents) > 0:
            return documents[0]

        return []

    except Exception as e:
        print(f"⚠️ Long-term Memory Recall Error: {e}")
        return []

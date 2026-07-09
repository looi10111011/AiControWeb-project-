"""Manual ingestion: PDF/DOCX/TXT -> chunk -> embed -> เก็บใน ChromaDB.
W1: skeleton only. W3: implement จริง.
W3: เปลี่ยนมาใช้ local embedding (all-MiniLM-L6-v2 ผ่าน ChromaDB) แทน Gemini API
    — collection.upsert() ส่ง documents ดิบไป ให้ ChromaDB embed ให้เองอัตโนมัติ
"""

from pathlib import Path
import hashlib
import re
from typing import List

# สำหรับอ่านไฟล์
from pypdf import PdfReader
from docx import Document

from backend.app.rag.chroma_client import get_collection

# จบประโยคด้วย . ! ? (ตามด้วยเว้นวรรค) หรือขึ้นบรรทัดใหม่ — ใช้หาจุดตัดที่ไม่ทำให้ประโยคขาด
_SENTENCE_BOUNDARY_RE = re.compile(r'(?<=[.!?])\s+|\n+')


def load_manual(path: Path) -> str:
    """โหลดไฟล์ PDF/DOCX/TXT เป็น text"""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    ext = path.suffix.lower()
    
    if ext == '.txt':
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    
    elif ext == '.docx':
        doc = Document(path)
        text = '\n'.join([para.text for para in doc.paragraphs])
        return text
    
    elif ext == '.pdf':
        reader = PdfReader(path)
        text = ''
        for page in reader.pages:
            text += page.extract_text() + '\n'
        return text
    
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def _tail_at_word_boundary(text: str, size: int) -> str:
    """เอาท้ายข้อความมาไม่เกิน size ตัวอักษร แต่ขยับไปขอบเขตคำ (เว้นวรรค) แรกที่เจอ
    กันคำถูกตัดครึ่งตอนใช้เป็น overlap ต่อท้าย chunk ถัดไป"""
    if len(text) <= size:
        return text
    tail = text[-size:]
    space_idx = tail.find(' ')
    return tail[space_idx + 1:] if space_idx != -1 else tail


def _split_by_words(text: str, size: int) -> List[str]:
    """ตัดข้อความยาวตามขอบเขตคำ (เว้นวรรค) แทนการตัดกลางคำตรงๆ ตามจำนวนตัวอักษร
    ใช้เป็น fallback สุดท้ายเมื่อประโยคเดียวก็ยังยาวเกิน chunk_size (พบยาก)"""
    words = text.split(' ')
    pieces = []
    current = ''
    for word in words:
        candidate = f'{current} {word}'.strip() if current else word
        if len(candidate) > size and current:
            pieces.append(current)
            current = word
        else:
            current = candidate
    if current:
        pieces.append(current)

    # กรณีคำเดียวยาวเกิน size จริงๆ (แทบไม่เกิดขึ้น) ค่อย hard-split ตามตัวอักษร
    result = []
    for piece in pieces:
        if len(piece) <= size:
            result.append(piece)
        else:
            result.extend(piece[i:i + size] for i in range(0, len(piece), size))
    return result


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """แบ่งข้อความเป็น chunks ตามย่อหน้า (บรรทัดว่างคั่น) — แต่ละหัวข้อ/ย่อหน้า
    จะไม่ถูกรวมเข้ากับหัวข้ออื่นในบรรทัดเดียวกัน กัน chunk ใหญ่เกินไปแบบไม่จำเป็น
    ถ้าย่อหน้าไหนยาวเกิน chunk_size เอง จะแบ่งตามขอบเขตประโยค (ไม่ตัดกลางประโยค/กลางคำ)
    """
    if not text:
        return []

    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    if not paragraphs:
        return []

    chunks: List[str] = []
    current = ''

    for para in paragraphs:
        # ย่อหน้าพอดีขนาด -> เป็น chunk แยกของตัวเอง ไม่รวมกับย่อหน้าอื่น
        units = [para] if len(para) <= chunk_size else _SENTENCE_BOUNDARY_RE.split(para)

        for unit in units:
            unit = unit.strip()
            if not unit:
                continue

            # ประโยคเดียวก็ยังยาวเกิน chunk_size (พบยาก) -> ตัดตามขอบเขตคำแทน
            pieces = [unit] if len(unit) <= chunk_size else _split_by_words(unit, chunk_size)

            for piece in pieces:
                if current and len(current) + len(piece) + 1 > chunk_size:
                    chunks.append(current)
                    # overlap: เอาท้ายประโยคสุดท้ายของ chunk ก่อนหน้ามาต่อ ไม่ตัดกลางคำ
                    current = _tail_at_word_boundary(current, overlap).strip() if overlap > 0 else ''
                    current = f'{current} {piece}'.strip() if current else piece
                else:
                    current = f'{current} {piece}'.strip() if current else piece

        # จบย่อหน้าแล้ว ตัด chunk ทันที ไม่ดึงย่อหน้าถัดไปมารวม
        if current:
            chunks.append(current)
            current = ''

    if current:
        chunks.append(current)

    return chunks


def ingest_manual(path: Path):
    """load -> chunk -> add to collection (ChromaDB embed ให้เองด้วย local model)"""
    print(f"📄 Loading: {path}")

    # 1. Load text
    text = load_manual(path)
    print(f"   ✅ Loaded {len(text)} characters")

    # 2. Chunk
    chunks = chunk_text(text)
    print(f"   ✅ Created {len(chunks)} chunks")

    if not chunks:
        print("⚠️ No chunks created, skipping...")
        return

    # 3. Add to ChromaDB — ไม่ embed เองแล้ว ส่ง documents ดิบไปให้ collection
    #    embed ด้วย local model (all-MiniLM-L6-v2) ที่ผูกไว้กับ collection ใน chroma_client.py
    #    (โหลดโมเดลครั้งแรก ~90MB แล้ว cache ไว้ใช้ครั้งต่อไป)
    print("   🔧 Generating embeddings (local all-MiniLM-L6-v2)...")
    collection = get_collection()

    # สร้าง ids, documents, metadatas
    ids = []
    documents = []
    metadatas = []

    for i, chunk in enumerate(chunks):
        # สร้าง id จาก hash ของชื่อไฟล์ + index (deterministic -> ingest ไฟล์เดิมซ้ำแล้ว upsert ทับของเดิมได้)
        doc_id = hashlib.md5(f"{path.stem}_{i}".encode()).hexdigest()
        ids.append(doc_id)
        documents.append(chunk)
        metadatas.append({
            "source": path.name,
            "chunk_index": i,
            "total_chunks": len(chunks),
            "path": str(path)
        })

    # upsert แทน add: กัน error/ถูกข้ามเงียบๆ ตอน ingest ไฟล์เดิมซ้ำ (เช่น อัปเดตคู่มือ)
    collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

    print(f"✅ Done! Added {len(chunks)} chunks to collection '{collection.name}'")
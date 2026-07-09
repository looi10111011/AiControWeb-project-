"""Manual ingestion: PDF/DOCX/TXT -> chunk -> embed -> เก็บใน ChromaDB.
W1: skeleton only. W3: implement จริง.
W3: เปลี่ยนมาใช้ local embedding (all-MiniLM-L6-v2 ผ่าน ChromaDB) แทน Gemini API
    — collection.upsert() ส่ง documents ดิบไป ให้ ChromaDB embed ให้เองอัตโนมัติ
"""

from pathlib import Path
import hashlib
from typing import List

# สำหรับอ่านไฟล์
from pypdf import PdfReader
from docx import Document

from backend.app.rag.chroma_client import get_collection


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


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """แบ่งข้อความเป็น chunks โดยใช้ newline หรือ sentence boundary"""
    if not text:
        return []

    # ถ้าข้อความสั้นกว่าขนาด chunk ให้คืนค่าเป็น chunk เดียว
    if len(text) <= chunk_size:
        return [text]

    # แบ่งด้วย newline ก่อน (เพื่อให้ chunk สมบูรณ์ตามบรรทัด)
    # ถ้าบรรทัดไหนยาวเกิน chunk_size เอง (พบบ่อยกับ PDF ที่ extract_text() ไม่มี newline)
    # ให้ตัดแบ่งบรรทัดนั้นก่อน กัน chunk บวมเกินขนาดที่ตั้งไว้
    raw_lines = text.split('\n')
    lines = []
    for line in raw_lines:
        if len(line) <= chunk_size:
            lines.append(line)
        else:
            lines.extend(line[i:i + chunk_size] for i in range(0, len(line), chunk_size))

    chunks = []
    current_chunk = ''
    
    for line in lines:
        # ถ้าเพิ่มบรรทัดนี้แล้วเกิน chunk_size ให้เก็บ chunk เดิม แล้วเริ่มใหม่
        if len(current_chunk) + len(line) + 1 > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            # overlap: เก็บส่วนท้ายของ chunk เดิมไว้
            overlap_text = current_chunk[-overlap:] if overlap > 0 else ''
            current_chunk = overlap_text + ' ' + line
        else:
            if current_chunk:
                current_chunk += '\n' + line
            else:
                current_chunk = line
    
    # เก็บ chunk สุดท้าย
    if current_chunk:
        chunks.append(current_chunk.strip())
    
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
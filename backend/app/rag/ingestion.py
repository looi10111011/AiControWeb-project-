"""Manual ingestion: PDF/DOCX/TXT (+XLSX/CSV for attached bytes) -> chunk -> ChromaDB.

W3: collection.upsert() ส่ง documents ดิบ ให้ ChromaDB embed เองด้วย local model (all-MiniLM-L6-v2)
"""

from io import BytesIO
from pathlib import Path
import csv
import hashlib
import re
from typing import List

from pypdf import PdfReader
from docx import Document
from openpyxl import load_workbook

from backend.app.rag.chroma_client import get_collection

# จบประโยคด้วย . ! ? (ตามด้วยเว้นวรรค) หรือขึ้นบรรทัดใหม่ — ใช้หาจุดตัดที่ไม่ทำให้ประโยคขาด
_SENTENCE_BOUNDARY_RE = re.compile(r'(?<=[.!?])\s+|\n+')


def _docx_to_text(source) -> str:
    doc = Document(source)
    return '\n'.join([para.text for para in doc.paragraphs])


def _pdf_to_text(source) -> str:
    reader = PdfReader(source)
    text = ''
    for page in reader.pages:
        text += page.extract_text() + '\n'
    return text


def load_manual(path: Path) -> str:
    """โหลดไฟล์ PDF/DOCX/TXT บนดิสก์เป็น text (FileNotFoundError / ValueError ถ้าไม่รองรับ)"""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ext = path.suffix.lower()

    if ext == '.txt':
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    elif ext == '.docx':
        return _docx_to_text(path)
    elif ext == '.pdf':
        return _pdf_to_text(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


# Task7 (W20): header row ของ timesheet ไทยไม่ได้อยู่แถว 1 เสมอ (metadata ชื่อ-สกุล/สังกัดแผนก อยู่ข้างบน)
# — สแกนหา header ด้วย label ที่รู้จักแทนการสมมติ index 0
_XLSX_HEADER_KEYWORDS = ("วันที่", "รายละเอียด", "เวลาทำงาน", "จำนวนชั่วโมง")


def _label_row_cells(header_values: List[str], row_values: List[str]) -> str:
    """Task8-follow-up: คืน "label: value | ..." ต่อ cell — แถวแบบ positional ล้วนทำให้ LLM นับคอลัมน์ผิด
    (bug จริง W20: อ้างว่าคอลัมน์ที่มีข้อมูลทุกแถว "ว่างทุกแถว" ในตาราง 20+ แถวที่ header merge 2 ชั้น)"""
    pairs = []
    for i, value in enumerate(row_values):
        label = header_values[i] if i < len(header_values) and header_values[i] else f"col{i + 1}"
        pairs.append(f"{label}: {value}")
    return ' | '.join(pairs)


def _split_metadata_and_table(rows_values: List[List[str]]) -> List[str]:
    """Task7 + Task8: header = แถวแรกที่มี keyword ใน _XLSX_HEADER_KEYWORDS อย่างน้อย 2 ตัว *ต่างกัน*.
    แถวเหนือ header -> "## Document Metadata"; header คง "a | b | c"; แถวข้อมูล -> _label_row_cells().
    ไม่เจอ header -> "a | b | c" ทุกแถวเหมือนเดิม.

    ต้อง 2+ distinct: bug จริง — title merge ทั้งแถว "ใบลงเวลาทำงานนักศึกษาฝึกงาน" มี "เวลาทำงาน" ซ้ำทุก cell
    ชนะกฎ ">=1 keyword" ก่อน header จริง แล้ว label ผิดทั้งตาราง"""
    header_idx = next(
        (
            i for i, row in enumerate(rows_values)
            if len({kw for cell in row for kw in _XLSX_HEADER_KEYWORDS if kw in cell}) >= 2
        ),
        None,
    )
    if header_idx is None:
        return [' | '.join(row) for row in rows_values]

    header_values = rows_values[header_idx]
    metadata_lines = [' | '.join(row) for row in rows_values[:header_idx]]
    table_lines = [
        ' | '.join(header_values),
        *(_label_row_cells(header_values, row) for row in rows_values[header_idx + 1:]),
    ]
    if not metadata_lines:
        return table_lines
    return ["## Document Metadata", *metadata_lines, "## Table", *table_lines]


def _xlsx_bytes_to_text(content: bytes) -> str:
    """XLSX -> text ตามรายชีต ("# Sheet: <title>"), ข้ามแถวว่างล้วน, ผ่าน _split_metadata_and_table().
    data_only=True: ใช้ค่า cache ของ formula (ตรงกับที่ user เห็นใน Excel).
    merged cell: openpyxl เก็บค่าไว้แค่ top-left (cell อื่นเป็น None) — ต้อง forward-fill ทั้ง range ไม่งั้น
    sub-row ที่ merge จาก parent (ชื่อพนักงาน/แผนก) กลายเป็นว่าง"""
    workbook = load_workbook(BytesIO(content), data_only=True)
    sections = []
    for sheet in workbook.worksheets:
        merged_values = {}
        for merged_range in sheet.merged_cells.ranges:
            top_left_value = sheet.cell(merged_range.min_row, merged_range.min_col).value
            for r in range(merged_range.min_row, merged_range.max_row + 1):
                for c in range(merged_range.min_col, merged_range.max_col + 1):
                    merged_values[(r, c)] = top_left_value

        rows_values: List[List[str]] = []
        for row in sheet.iter_rows():
            values = [
                merged_values.get((cell.row, cell.column)) if cell.value is None else cell.value
                for cell in row
            ]
            if all(v is None for v in values):
                continue
            rows_values.append(['' if v is None else str(v) for v in values])
        if rows_values:
            lines = _split_metadata_and_table(rows_values)
            sections.append(f"# Sheet: {sheet.title}\n" + '\n'.join(lines))
    return '\n\n'.join(sections)


def _csv_bytes_to_text(content: bytes) -> str:
    """CSV -> "a | b | c" ต่อแถว (ข้ามแถวว่างล้วน). ใช้ csv.reader รองรับ comma/quote ในค่า;
    utf-8-sig ตัด BOM ที่ Excel ใส่มา"""
    text = content.decode('utf-8-sig')
    rows_text = []
    for row in csv.reader(text.splitlines()):
        if not row or all(not cell.strip() for cell in row):
            continue
        rows_text.append(' | '.join(row))
    return '\n'.join(rows_text)


def load_manual_bytes(content: bytes, filename: str) -> str:
    """เหมือน load_manual() แต่รับ bytes ของไฟล์ที่แนบผ่าน API (+ .xlsx/.csv) — ไม่แตะดิสก์;
    filename ใช้ดู extension เท่านั้น ไม่เคยเป็น path (กัน path traversal). ValueError ถ้าไม่รองรับ"""
    ext = Path(filename).suffix.lower()

    if ext == '.txt':
        return content.decode('utf-8')
    elif ext == '.docx':
        return _docx_to_text(BytesIO(content))
    elif ext == '.pdf':
        return _pdf_to_text(BytesIO(content))
    elif ext == '.xlsx':
        return _xlsx_bytes_to_text(content)
    elif ext == '.csv':
        return _csv_bytes_to_text(content)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def _tail_at_word_boundary(text: str, size: int) -> str:
    """ท้ายข้อความไม่เกิน size ตัวอักษร ขยับไปขอบคำแรก (ใช้เป็น overlap ไม่ให้คำขาดครึ่ง)"""
    if len(text) <= size:
        return text
    tail = text[-size:]
    space_idx = tail.find(' ')
    return tail[space_idx + 1:] if space_idx != -1 else tail


def _split_by_words(text: str, size: int) -> List[str]:
    """ตัดตามขอบคำ — fallback เมื่อประโยคเดียวยาวเกิน chunk_size; คำเดียวที่ยาวเกินค่อย hard-split"""
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

    result = []
    for piece in pieces:
        if len(piece) <= size:
            result.append(piece)
        else:
            result.extend(piece[i:i + size] for i in range(0, len(piece), size))
    return result


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """แบ่งเป็น chunk ตามย่อหน้า (บรรทัดว่างคั่น) ไม่รวมข้ามย่อหน้า; ย่อหน้ายาวเกินแบ่งตามประโยค
    แล้วตามคำ (ไม่ตัดกลางประโยค/คำ)"""
    if not text:
        return []

    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    if not paragraphs:
        return []

    chunks: List[str] = []
    current = ''

    for para in paragraphs:
        units = [para] if len(para) <= chunk_size else _SENTENCE_BOUNDARY_RE.split(para)

        for unit in units:
            unit = unit.strip()
            if not unit:
                continue

            pieces = [unit] if len(unit) <= chunk_size else _split_by_words(unit, chunk_size)

            for piece in pieces:
                if current and len(current) + len(piece) + 1 > chunk_size:
                    chunks.append(current)
                    # overlap: เอาท้ายประโยคสุดท้ายของ chunk ก่อนหน้ามาต่อ ไม่ตัดกลางคำ
                    current = _tail_at_word_boundary(current, overlap).strip() if overlap > 0 else ''
                current = f'{current} {piece}'.strip() if current else piece

        # จบย่อหน้าแล้ว ตัด chunk ทันที ไม่ดึงย่อหน้าถัดไปมารวม
        if current:
            chunks.append(current)
            current = ''

    return chunks


def ingest_manual(path: Path):
    """load -> chunk -> upsert เข้า collection (ChromaDB embed ให้เองด้วย local model)"""
    print(f"📄 Loading: {path}")

    text = load_manual(path)
    print(f"   ✅ Loaded {len(text)} characters")

    chunks = chunk_text(text)
    print(f"   ✅ Created {len(chunks)} chunks")

    if not chunks:
        print("⚠️ No chunks created, skipping...")
        return

    print("   🔧 Generating embeddings (local all-MiniLM-L6-v2)...")
    collection = get_collection()

    ids = []
    documents = []
    metadatas = []

    for i, chunk in enumerate(chunks):
        # id deterministic จากชื่อไฟล์ + index -> ingest ไฟล์เดิมซ้ำแล้ว upsert ทับของเดิม
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

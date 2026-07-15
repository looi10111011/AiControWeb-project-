import re
from pathlib import Path
from unittest.mock import patch

import pytest

from backend.app.rag.ingestion import chunk_text, ingest_manual, load_manual

MANUAL_TXT = Path(__file__).parent / "manual_test.txt"


# --- load_manual ---

def test_load_manual_txt_reads_content():
    text = load_manual(MANUAL_TXT)
    assert "Facebook" in text


def test_load_manual_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_manual(Path("does_not_exist.txt"))


def test_load_manual_unsupported_extension_raises():
    with pytest.raises(ValueError):
        load_manual(Path(__file__))  # .py ไม่รองรับ


# --- chunk_text ---

def test_chunk_text_empty_returns_empty_list():
    assert chunk_text("") == []


def test_chunk_text_shorter_than_chunk_size_returns_single_chunk():
    assert chunk_text("hello world", chunk_size=500) == ["hello world"]


def test_chunk_text_splits_multiline_text():
    text = "\n".join(f"line {i}" for i in range(100))
    chunks = chunk_text(text, chunk_size=50, overlap=10)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c) <= 50 + 10 + 1  # เผื่อ overlap prefix ต่อท้าย


def test_chunk_text_splits_oversized_single_line():
    # บรรทัดเดียวยาว 1200 ตัวอักษร ไม่มี newline (เช่น PDF ที่ extract ไม่มี newline)
    text = "a" * 1200
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 500 + 50 + 1 for c in chunks)
    assert "".join(chunks).replace(" ", "").count("a") >= 1200


def test_chunk_text_keeps_each_paragraph_as_its_own_chunk():
    # แต่ละหัวข้อ (คั่นด้วยบรรทัดว่าง) ต้องไม่ถูกรวมเข้า chunk เดียวกับหัวข้ออื่น
    # แม้จะรวมกันแล้วยังพอดี chunk_size ก็ตาม (บั๊กเดิม: ยัดหลายหัวข้อใน chunk เดียว)
    text = "topic A: do X\n\ntopic B: do Y\n\ntopic C: do Z"
    chunks = chunk_text(text, chunk_size=500)
    assert chunks == ["topic A: do X", "topic B: do Y", "topic C: do Z"]


def test_chunk_text_real_manual_produces_one_chunk_per_topic():
    text = load_manual(MANUAL_TXT)
    chunks = chunk_text(text)
    # manual_test.txt มี 11 หัวข้อคั่นด้วยบรรทัดว่าง (นับล่าสุด W7[B]: เพิ่ม
    # RULE-04 เข้าไปอีก 1 หัวข้อ) — ตัวเลขนี้ค้างมาก่อนแล้ว (เคย fail จากการแก้ไฟล์
    # ครั้งก่อนหน้าที่เพิ่ม RULE-01..03 โดยไม่ได้อัปเดตเทสต์นี้ตาม) แก้ให้ตรงของจริง
    assert len(chunks) == 11
    assert "Facebook" in chunks[0]
    assert "Alt+F4" in chunks[1]


def test_chunk_text_never_splits_a_word_across_chunks():
    # ย่อหน้าเดียวยาวมาก หลายประโยค ไม่มีบรรทัดว่างคั่น -> ต้อง fallback ไปแบ่งตามประโยค/คำ
    # และห้ามมี chunk ไหนขึ้นต้นหรือจบกลางคำ (บั๊กเดิม: overlap ตัดกลางคำ)
    words = [f"word{i}" for i in range(200)]
    text = ". ".join(" ".join(words[i:i + 8]) for i in range(0, len(words), 8)) + "."
    chunks = chunk_text(text, chunk_size=120, overlap=20)
    assert len(chunks) > 1
    for c in chunks:
        # ทุก token ในทุก chunk ต้องเป็น "wordN" เต็มคำเสมอ ห้ามมีเศษคำที่ถูกตัดครึ่ง
        for token in c.replace(".", " ").split():
            assert re.fullmatch(r"word\d+", token), f"found broken token: {token!r} in chunk {c!r}"


# --- ingest_manual (mock chroma กันยิง DB จริง/โหลดโมเดล embedding ตอนรัน unit test) ---
# หมายเหตุ: ingestion.py ไม่ embed เองแล้ว (local embedding ผูกไว้กับ collection ใน
# chroma_client.py) — ingest_manual แค่ต้องส่ง documents ดิบให้ collection.upsert()

def test_ingest_manual_upserts_with_matching_ids_and_documents():
    with patch("backend.app.rag.ingestion.get_collection") as mock_get_collection, \
         patch("backend.app.rag.ingestion.chunk_text", return_value=["chunk A", "chunk B"]):
        mock_collection = mock_get_collection.return_value
        mock_collection.name = "manuals"

        ingest_manual(MANUAL_TXT)

        mock_collection.upsert.assert_called_once()
        _, kwargs = mock_collection.upsert.call_args
        assert kwargs["documents"] == ["chunk A", "chunk B"]
        assert "embeddings" not in kwargs  # ให้ ChromaDB embed เองด้วย local model
        assert len(kwargs["ids"]) == 2
        assert len(set(kwargs["ids"])) == 2  # id ไม่ชนกัน
        assert all(m["source"] == MANUAL_TXT.name for m in kwargs["metadatas"])


def test_ingest_manual_skips_when_no_chunks():
    with patch("backend.app.rag.ingestion.chunk_text", return_value=[]), \
         patch("backend.app.rag.ingestion.get_collection") as mock_get_collection:
        ingest_manual(MANUAL_TXT)

        mock_get_collection.return_value.upsert.assert_not_called()

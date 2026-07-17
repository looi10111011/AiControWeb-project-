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
    # manual_test.txt มี 62 ย่อหน้าคั่นด้วยบรรทัดว่าง (นับรวมหัวข้อใหม่ D-J)
    assert len(chunks) == 62
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


# --- edge cases เพิ่มเติม (chunk_text) ---

def test_chunk_text_whitespace_only_returns_empty_list():
    """ข้อความที่มีแต่ whitespace/newline ต้องคืน [] เหมือน empty string"""
    assert chunk_text("   \n\n   \t  \n") == []


def test_chunk_text_single_word_longer_than_chunk_size_does_not_crash():
    """คำเดียวที่ยาวกว่า chunk_size จริงๆ — fallback hard-split ตัวอักษร ไม่ควร crash"""
    long_word = "x" * 1500
    chunks = chunk_text(long_word, chunk_size=500, overlap=0)
    assert len(chunks) > 1
    # ต้องครอบคลุมตัวอักษรครบ
    assert sum(len(c) for c in chunks) >= 1500


def test_chunk_text_overlap_zero_no_prefix_added():
    """overlap=0 ต้องไม่มี prefix ต่อท้ายขึ้นต้น chunk ถัดไป (ไม่มีเศษ overlap)"""
    # ใช้ประโยคสั้นๆ สองย่อหน้า ให้ chunk ที่ 1 จบแล้วไม่ต้อง overlap ไป chunk ที่ 2
    text = "A " * 300   # บรรทัดเดียว ยาวพอให้แตกหลาย chunk
    chunks = chunk_text(text, chunk_size=100, overlap=0)
    assert len(chunks) > 1
    # ทุก chunk ต้องไม่ยาวเกิน chunk_size (ไม่มี overlap เพิ่ม)
    for c in chunks:
        assert len(c) <= 100


def test_chunk_text_new_topics_present_in_real_manual():
    """หัวข้อใหม่ที่เพิ่มเข้า manual_test.txt ต้องถูก chunk แยกและมีเนื้อหาครบ"""
    text = load_manual(MANUAL_TXT)
    chunks = chunk_text(text)
    combined = " ".join(chunks)
    # ตรวจว่าหัวข้อใหม่ทั้ง 5 มีอยู่ใน chunks จริง
    assert "Clear browsing data" in combined   # ล้าง Cache
    assert "Incognito" in combined             # Incognito/Private
    assert "Ctrl+D" in combined                # Bookmark
    assert "Print Preview" in combined         # Print
    assert "Ctrl+0" in combined                # Zoom


def test_ingest_manual_metadata_contains_path_field():
    """metadata ของทุก chunk ต้องมี field 'path' เป็น string (ไม่ใช่ None/missing)"""
    with patch("backend.app.rag.ingestion.get_collection") as mock_get_collection, \
         patch("backend.app.rag.ingestion.chunk_text", return_value=["chunk X"]):
        mock_collection = mock_get_collection.return_value
        mock_collection.name = "manuals"

        ingest_manual(MANUAL_TXT)

        _, kwargs = mock_collection.upsert.call_args
        for meta in kwargs["metadatas"]:
            assert "path" in meta
            assert isinstance(meta["path"], str)


def test_ingest_manual_ids_differ_across_files():
    """ไฟล์ต่างกัน ต้องได้ id ต่างกัน (hash ขึ้นกับ stem ของไฟล์) กัน id ชนข้ามไฟล์"""
    import hashlib
    other_path = MANUAL_TXT.parent / "other_manual.txt"
    id_a = hashlib.md5(f"{MANUAL_TXT.stem}_0".encode()).hexdigest()
    id_b = hashlib.md5(f"{other_path.stem}_0".encode()).hexdigest()
    assert id_a != id_b


# --- edge cases เพิ่มเติม รอบ 2 ---

def test_chunk_text_only_newlines_returns_empty_list():
    """string ที่มีแต่ newlines หลายบรรทัด ต้องคืน [] เหมือน empty"""
    assert chunk_text("\n\n\n\n\n") == []


def test_chunk_text_exactly_chunk_size_is_single_chunk():
    """ข้อความที่ยาวพอดี chunk_size ต้องได้ 1 chunk ไม่ถูกแตกออก"""
    text = "A" * 500
    chunks = chunk_text(text, chunk_size=500, overlap=0)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_overlap_larger_than_piece_does_not_crash():
    """overlap ใหญ่กว่าเนื้อหาของ piece ต้องไม่ crash และคืน list"""
    text = "short sentence. next sentence."
    chunks = chunk_text(text, chunk_size=10, overlap=50)
    assert isinstance(chunks, list)
    assert len(chunks) > 0


def test_chunk_text_mixed_thai_english_preserves_all_content():
    """ข้อความผสม TH/EN ต้องไม่ทำให้เนื้อหาหายหรือบิดเบือน"""
    text = "วิธีใช้ browser: กด Ctrl+T เพื่อเปิด new tab"
    chunks = chunk_text(text, chunk_size=500)
    combined = " ".join(chunks)
    assert "Ctrl+T" in combined
    assert "browser" in combined


def test_chunk_text_group_a_topics_present_in_real_manual():
    """หัวข้อกลุ่ม A (Tab/Reload/DevTools/URL/Navigation) ต้องอยู่ใน chunks จริง"""
    text = load_manual(MANUAL_TXT)
    chunks = chunk_text(text)
    combined = " ".join(chunks)
    assert "Ctrl+T" in combined          # เปิด Tab ใหม่
    assert "Ctrl+Shift+T" in combined    # กู้คืน Tab
    assert "Alt+Left Arrow" in combined  # Browser History
    assert "F5" in combined              # Reload
    assert "F12" in combined             # Developer Tools
    assert "Ctrl+L" in combined          # Copy URL
    assert "Ctrl+Tab" in combined        # สลับ Tab


def test_chunk_text_group_b_topics_present_in_real_manual():
    """หัวข้อกลุ่ม B (Form/Dropdown/Upload/Copy) ต้องอยู่ใน chunks จริง"""
    text = load_manual(MANUAL_TXT)
    chunks = chunk_text(text)
    combined = " ".join(chunks)
    assert "Submit" in combined          # กรอกฟอร์ม
    assert "dropdown" in combined        # Dropdown
    assert "Choose File" in combined     # Upload
    assert "select all" in combined      # คัดลอกข้อความ


def test_chunk_text_group_c_topics_present_in_real_manual():
    """หัวข้อกลุ่ม C (Logout/Password/HTTPS) ต้องอยู่ใน chunks จริง"""
    text = load_manual(MANUAL_TXT)
    chunks = chunk_text(text)
    combined = " ".join(chunks)
    assert "Sign out" in combined        # Logout
    assert "current password" in combined  # เปลี่ยนรหัสผ่าน
    assert "padlock" in combined         # SSL/HTTPS
    assert "https://" in combined        # SSL/HTTPS


def test_ingest_manual_chunk_index_is_sequential():
    """metadata chunk_index ต้องเรียงจาก 0 ถึง N-1 ต่อเนื่องไม่มีช่องว่าง"""
    with patch("backend.app.rag.ingestion.get_collection") as mock_get_collection, \
         patch("backend.app.rag.ingestion.chunk_text",
               return_value=["a", "b", "c", "d", "e"]):
        mock_collection = mock_get_collection.return_value
        mock_collection.name = "manuals"

        ingest_manual(MANUAL_TXT)

        _, kwargs = mock_collection.upsert.call_args
        indices = [m["chunk_index"] for m in kwargs["metadatas"]]
        assert indices == list(range(5))
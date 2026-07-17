import pytest
from unittest.mock import patch

from backend.app.rag.retriever import retrieve


# --- happy path ---

def test_retrieve_returns_first_document_list():
    with patch("backend.app.rag.retriever.get_collection") as mock_get_collection:
        mock_collection = mock_get_collection.return_value
        mock_collection.query.return_value = {"documents": [["chunk A", "chunk B"]]}

        result = retrieve("วิธีเปิด Facebook")

        assert result == ["chunk A", "chunk B"]


def test_retrieve_passes_k_as_n_results():
    with patch("backend.app.rag.retriever.get_collection") as mock_get_collection:
        mock_collection = mock_get_collection.return_value
        mock_collection.query.return_value = {"documents": [["chunk"]]}

        retrieve("query", k=3)

        _, kwargs = mock_collection.query.call_args
        assert kwargs["n_results"] == 3


def test_retrieve_without_page_state_queries_with_query_only():
    with patch("backend.app.rag.retriever.get_collection") as mock_get_collection:
        mock_get_collection.return_value.query.return_value = {"documents": [["c"]]}

        retrieve("วิธีปิด browser")

        _, kwargs = mock_get_collection.return_value.query.call_args
        assert kwargs["query_texts"] == ["วิธีปิด browser"]


def test_retrieve_with_page_state_combines_into_query_texts():
    with patch("backend.app.rag.retriever.get_collection") as mock_get_collection:
        mock_get_collection.return_value.query.return_value = {"documents": [["c"]]}

        retrieve("วิธีปิด browser", page_state="[0] button 'Close'")

        _, kwargs = mock_get_collection.return_value.query.call_args
        combined = kwargs["query_texts"][0]
        assert "วิธีปิด browser" in combined
        assert "[0] button 'Close'" in combined


# --- edge cases: ต้องไม่ throw ออกไปให้ agent loop พังเด็ดขาด ---

def test_retrieve_returns_empty_list_when_no_documents_key():
    with patch("backend.app.rag.retriever.get_collection") as mock_get_collection:
        mock_get_collection.return_value.query.return_value = {}

        assert retrieve("query") == []


def test_retrieve_returns_empty_list_when_documents_empty():
    with patch("backend.app.rag.retriever.get_collection") as mock_get_collection:
        mock_get_collection.return_value.query.return_value = {"documents": []}

        assert retrieve("query") == []


def test_retrieve_swallows_collection_lookup_errors_and_returns_empty_list():
    with patch("backend.app.rag.retriever.get_collection", side_effect=RuntimeError("Chroma down")):
        assert retrieve("query") == []


def test_retrieve_swallows_chroma_query_errors_and_returns_empty_list():
    with patch("backend.app.rag.retriever.get_collection") as mock_get_collection:
        mock_get_collection.return_value.query.side_effect = RuntimeError("Chroma query failed")

        assert retrieve("query") == []


# --- edge cases เพิ่มเติม ---

def test_retrieve_empty_query_does_not_throw():
    """query ว่าง ('') ต้องไม่ throw ออกมา ต้องคืน list เสมอ (กฎเหล็ก retriever)"""
    with patch("backend.app.rag.retriever.get_collection") as mock_get_collection:
        mock_get_collection.return_value.query.return_value = {"documents": [[]]}

        result = retrieve("")
        assert isinstance(result, list)


def test_retrieve_very_long_page_state_does_not_throw():
    """page_state ยาวมากๆ (เช่น DOM ขนาดใหญ่) ต้องไม่ throw และคืน list เสมอ"""
    with patch("backend.app.rag.retriever.get_collection") as mock_get_collection:
        mock_get_collection.return_value.query.return_value = {"documents": [["chunk"]]}

        long_page_state = "<div>" * 5000  # ~30 000 ตัวอักษร
        result = retrieve("query", page_state=long_page_state, k=3)
        assert isinstance(result, list)
        # ตรวจว่า page_state ถูกรวมเข้า query_texts จริง
        _, kwargs = mock_get_collection.return_value.query.call_args
        assert long_page_state in kwargs["query_texts"][0]


def test_retrieve_k_equals_one_returns_at_most_one_result():
    """k=1 ต้องส่ง n_results=1 ให้ ChromaDB และคืนผลลัพธ์เป็น list ไม่เกิน 1 element"""
    with patch("backend.app.rag.retriever.get_collection") as mock_get_collection:
        mock_get_collection.return_value.query.return_value = {"documents": [["only chunk"]]}

        result = retrieve("query", k=1)

        _, kwargs = mock_get_collection.return_value.query.call_args
        assert kwargs["n_results"] == 1
        assert result == ["only chunk"]


def test_retrieve_returns_empty_list_when_inner_list_is_empty():
    """ChromaDB คืน documents=[[]] (inner list ว่าง) ต้องได้ [] กลับ ไม่ใช่ [[]]"""
    with patch("backend.app.rag.retriever.get_collection") as mock_get_collection:
        mock_get_collection.return_value.query.return_value = {"documents": [[]]}

        result = retrieve("query")
        assert result == []


def test_retrieve_query_with_special_characters_does_not_throw():
    """query ที่มี special chars (เช่น \n \t <> / \\) ต้องไม่ throw และคืน list เสมอ"""
    with patch("backend.app.rag.retriever.get_collection") as mock_get_collection:
        mock_get_collection.return_value.query.return_value = {"documents": [["result"]]}

        special_query = "test\n\t<script>alert('xss')</script>/path\\back"
        result = retrieve(special_query)
        assert isinstance(result, list)


# --- edge cases เพิ่มเติม รอบ 2 ---

def test_retrieve_default_k_sends_n_results_5():
    """ไม่ระบุ k → n_results ต้องเป็น 5 (default ของฟังก์ชัน)"""
    with patch("backend.app.rag.retriever.get_collection") as mock_get_collection:
        mock_get_collection.return_value.query.return_value = {"documents": [["c"]]}

        retrieve("query")

        _, kwargs = mock_get_collection.return_value.query.call_args
        assert kwargs["n_results"] == 5


def test_retrieve_returns_flat_list_not_nested():
    """ต้องคืน list[str] แบน ไม่ใช่ list[list[str]] (documents[0] ไม่ใช่ทั้ง documents)"""
    with patch("backend.app.rag.retriever.get_collection") as mock_get_collection:
        mock_get_collection.return_value.query.return_value = {
            "documents": [["chunk 1", "chunk 2", "chunk 3"]]
        }

        result = retrieve("query", k=3)

        assert result == ["chunk 1", "chunk 2", "chunk 3"]
        assert all(isinstance(item, str) for item in result)


def test_retrieve_empty_page_state_uses_query_only():
    """page_state='' (empty string) → query_texts ต้องมีแค่ query ล้วนๆ ไม่มีส่วน page state"""
    with patch("backend.app.rag.retriever.get_collection") as mock_get_collection:
        mock_get_collection.return_value.query.return_value = {"documents": [["c"]]}

        retrieve("my query", page_state="")

        _, kwargs = mock_get_collection.return_value.query.call_args
        # page_state ว่าง → embed_input ต้องเท่ากับ query ล้วนๆ ไม่มี "Current page:" แนบ
        assert kwargs["query_texts"] == ["my query"]


def test_retrieve_multiple_chunks_all_returned():
    """ChromaDB คืน 5 chunks → ต้องได้ครบทั้ง 5 ไม่ถูก slice หรือ truncate"""
    chunks = [f"chunk {i}" for i in range(5)]
    with patch("backend.app.rag.retriever.get_collection") as mock_get_collection:
        mock_get_collection.return_value.query.return_value = {"documents": [chunks]}

        result = retrieve("query", k=5)

        assert result == chunks
        assert len(result) == 5


def test_retrieve_unicode_thai_arabic_emoji_query_does_not_throw():
    """query ที่มีภาษาไทย, อาหรับ, emoji ต้องไม่ throw และคืน list เสมอ"""
    with patch("backend.app.rag.retriever.get_collection") as mock_get_collection:
        mock_get_collection.return_value.query.return_value = {"documents": [["result"]]}

        unicode_queries = [
            "วิธีเปิด browser บน Windows",   # ภาษาไทย
            "كيفية فتح المتصفح",              # ภาษาอาหรับ
            "🔍 search web 🌐",               # emoji
        ]
        for q in unicode_queries:
            result = retrieve(q)
            assert isinstance(result, list), f"failed on query: {q!r}"


def test_retrieve_none_in_document_list_does_not_crash():
    """ถ้า ChromaDB คืน documents ที่มี None ปนมา ต้องไม่ throw ออกมา"""
    with patch("backend.app.rag.retriever.get_collection") as mock_get_collection:
        mock_get_collection.return_value.query.return_value = {
            "documents": [[None, "chunk text"]]
        }

        # ไม่ throw → แค่คืน list กลับมา (อาจมี None หรือไม่ก็ได้ แต่ห้าม throw)
        try:
            result = retrieve("query")
            assert isinstance(result, list)
        except Exception as e:
            pytest.fail(f"retrieve() should not throw but raised: {e}")

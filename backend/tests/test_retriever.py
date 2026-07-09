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

from unittest.mock import patch

from backend.app.core.long_term_memory import record_task, recall


# --- recall() ---
#
# W23: recall() ตอนนี้ "ต้อง" มี session_id ถึงจะ query อะไรเลย (ดู module docstring —
# ป้องกัน session หนึ่งดึงความจำของอีก session ที่ทำคนละเว็บ/คนละเจตนามาปนกัน) ทุกเทสต์
# กลุ่ม "query จริง" ด้านล่างเลยต้องส่ง session_id="s1" ตรงๆ เสมอ


def test_recall_returns_first_document_list():
    with patch("backend.app.core.long_term_memory.get_long_term_collection") as mock_get_collection:
        mock_collection = mock_get_collection.return_value
        mock_collection.query.return_value = {"documents": [["task A", "task B"]]}

        result = recall("login goal", session_id="s1")

        assert result == ["task A", "task B"]


def test_recall_passes_k_as_n_results():
    with patch("backend.app.core.long_term_memory.get_long_term_collection") as mock_get_collection:
        mock_collection = mock_get_collection.return_value
        mock_collection.query.return_value = {"documents": [["x"]]}

        recall("goal", k=5, session_id="s1")

        _, kwargs = mock_collection.query.call_args
        assert kwargs["n_results"] == 5


def test_recall_with_page_state_combines_into_query_texts():
    with patch("backend.app.core.long_term_memory.get_long_term_collection") as mock_get_collection:
        mock_get_collection.return_value.query.return_value = {"documents": [["x"]]}

        recall("goal", page_state="[0] button 'Apply Code'", session_id="s1")

        _, kwargs = mock_get_collection.return_value.query.call_args
        combined = kwargs["query_texts"][0]
        assert "goal" in combined
        assert "[0] button 'Apply Code'" in combined


def test_recall_returns_empty_list_when_no_documents_key():
    with patch("backend.app.core.long_term_memory.get_long_term_collection") as mock_get_collection:
        mock_get_collection.return_value.query.return_value = {}

        assert recall("goal", session_id="s1") == []


def test_recall_swallows_collection_lookup_errors_and_returns_empty_list():
    with patch(
        "backend.app.core.long_term_memory.get_long_term_collection", side_effect=RuntimeError("Chroma down")
    ):
        assert recall("goal", session_id="s1") == []


def test_recall_swallows_query_errors_and_returns_empty_list():
    with patch("backend.app.core.long_term_memory.get_long_term_collection") as mock_get_collection:
        mock_get_collection.return_value.query.side_effect = RuntimeError("query failed")

        assert recall("goal", session_id="s1") == []


# --- W23: session scoping (the actual isolation fix) ---


def test_recall_filters_query_by_session_id():
    with patch("backend.app.core.long_term_memory.get_long_term_collection") as mock_get_collection:
        mock_get_collection.return_value.query.return_value = {"documents": [["x"]]}

        recall("goal", session_id="session-abc")

        _, kwargs = mock_get_collection.return_value.query.call_args
        assert kwargs["where"] == {"session_id": "session-abc"}


def test_recall_returns_empty_list_without_querying_when_session_id_missing():
    """session_id ว่างเปล่า (ค่า default) = ไม่มี session context ให้ scope ปลอดภัยได้ —
    ต้องคืน [] ทันทีโดยไม่แตะ ChromaDB เลยด้วยซ้ำ (ไม่ใช่แค่คืนผลลัพธ์ว่างหลัง query แบบไม่
    กรอง) กันความเสี่ยงดึงความจำข้าม session มาปนกันแบบไม่ตั้งใจ"""
    with patch("backend.app.core.long_term_memory.get_long_term_collection") as mock_get_collection:
        assert recall("goal") == []
        mock_get_collection.assert_not_called()


def test_recall_returns_empty_list_without_querying_when_session_id_explicitly_empty():
    with patch("backend.app.core.long_term_memory.get_long_term_collection") as mock_get_collection:
        assert recall("goal", session_id="") == []
        mock_get_collection.assert_not_called()


def test_recall_never_leaks_across_different_session_ids():
    """เอกสารเดียวกัน (mock คืนค่าเดิมไม่ว่า where จะเป็นอะไร — จำลอง ChromaDB จริงที่จะ
    กรองด้วย where เองก่อน) แต่จุดที่ต้องพิสูจน์คือ session คนละอันต้องส่ง where คนละค่ากัน
    เสมอ ไม่มีทาง query แบบไม่กรอง/กรองผิด session ได้เลย"""
    with patch("backend.app.core.long_term_memory.get_long_term_collection") as mock_get_collection:
        mock_get_collection.return_value.query.return_value = {"documents": [["irrelevant here"]]}

        recall("goal", session_id="session-1")
        recall("goal", session_id="session-2")

        wheres = [call.kwargs["where"] for call in mock_get_collection.return_value.query.call_args_list]
        assert wheres == [{"session_id": "session-1"}, {"session_id": "session-2"}]


# --- record_task() ---


def test_record_task_adds_document_with_url_goal_outcome_and_message():
    with patch("backend.app.core.long_term_memory.get_long_term_collection") as mock_get_collection:
        mock_collection = mock_get_collection.return_value

        record_task(
            url="https://www.saucedemo.com/",
            goal="login",
            success=True,
            message="Login สำเร็จแล้ว",
        )

        _, kwargs = mock_collection.add.call_args
        document = kwargs["documents"][0]
        assert "https://www.saucedemo.com/" in document
        assert "login" in document
        assert "success" in document
        assert "Login สำเร็จแล้ว" in document
        assert kwargs["metadatas"][0] == {
            "url": "https://www.saucedemo.com/", "goal": "login", "success": True, "session_id": "",
        }
        assert len(kwargs["ids"][0]) > 0


def test_record_task_stores_the_given_session_id():
    with patch("backend.app.core.long_term_memory.get_long_term_collection") as mock_get_collection:
        mock_collection = mock_get_collection.return_value

        record_task(url="https://example.com", goal="goal", success=True, message="ok", session_id="session-xyz")

        assert mock_collection.add.call_args.kwargs["metadatas"][0]["session_id"] == "session-xyz"


def test_record_task_marks_failed_outcome_when_not_success():
    with patch("backend.app.core.long_term_memory.get_long_term_collection") as mock_get_collection:
        mock_collection = mock_get_collection.return_value

        record_task(url="https://example.com", goal="goal", success=False, message="ทำต่อไม่ได้")

        _, kwargs = mock_collection.add.call_args
        assert "failed" in kwargs["documents"][0]
        assert kwargs["metadatas"][0]["success"] is False


def test_record_task_includes_failed_actions_when_provided():
    with patch("backend.app.core.long_term_memory.get_long_term_collection") as mock_get_collection:
        mock_collection = mock_get_collection.return_value

        record_task(
            url="https://example.com",
            goal="goal",
            success=False,
            message="โดนบล็อก",
            failed_actions="- {'type': 'click', 'index': 3} -> [FAIL] click(3) -> ปุ่มนี้ถูกบล็อก",
        )

        document = mock_collection.add.call_args.kwargs["documents"][0]
        assert "click(3)" in document
        assert "ปุ่มนี้ถูกบล็อก" in document


def test_record_task_omits_failed_actions_section_when_empty():
    with patch("backend.app.core.long_term_memory.get_long_term_collection") as mock_get_collection:
        mock_collection = mock_get_collection.return_value

        record_task(url="https://example.com", goal="goal", success=True, message="สำเร็จ")

        document = mock_collection.add.call_args.kwargs["documents"][0]
        assert "Failed actions" not in document


def test_record_task_swallows_errors_without_throwing():
    with patch(
        "backend.app.core.long_term_memory.get_long_term_collection", side_effect=RuntimeError("Chroma down")
    ):
        record_task(url="https://example.com", goal="goal", success=True, message="ok")  # ไม่ throw = ผ่าน

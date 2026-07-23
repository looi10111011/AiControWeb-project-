from unittest.mock import patch

from backend.app.config import settings
from backend.app.core.plan_memory import _uses_unsupported_script, find_matching_plan, save_confirmed_plan

# ทุกเทสต์ mock get_plan_memory_collection() ตรงๆ (เหมือน test_long_term_memory.py) —
# ไม่โหลด embedding model จริง/ไม่แตะ ChromaDB จริงเลย เพราะสิ่งที่ต้องพิสูจน์คือ logic การ
# หา lineage/version ของโมดูลนี้ ไม่ใช่พฤติกรรมจริงของ semantic search (นั่นคุมด้วยการ
# calibrate settings.plan_memory_max_distance เอง — ดู comment ใน config.py)


def _query_result(intent_key: str, distance: float) -> dict:
    return {
        "ids": [["doc-1"]],
        "metadatas": [[{"intent_key": intent_key, "version": 1, "plan": "irrelevant here"}]],
        "distances": [[distance]],
    }


# --- find_matching_plan() ---


def test_find_matching_plan_returns_none_when_domain_has_no_documents():
    with patch("backend.app.core.plan_memory.get_plan_memory_collection") as mock_get:
        mock_get.return_value.query.return_value = {"ids": [[]], "metadatas": [[]], "distances": [[]]}

        assert find_matching_plan("example.com", "login") is None


def test_find_matching_plan_returns_none_when_distance_exceeds_threshold():
    with patch("backend.app.core.plan_memory.get_plan_memory_collection") as mock_get:
        mock_get.return_value.query.return_value = _query_result("k1", settings.plan_memory_max_distance + 0.01)

        assert find_matching_plan("example.com", "checkout") is None


def test_find_matching_plan_returns_latest_version_when_within_threshold():
    with patch("backend.app.core.plan_memory.get_plan_memory_collection") as mock_get:
        collection = mock_get.return_value
        collection.query.return_value = _query_result("k1", 0.2)
        collection.get.return_value = {
            "metadatas": [
                {"intent_key": "k1", "version": 1, "plan": "1. old plan"},
                {"intent_key": "k1", "version": 3, "plan": "1. newest plan"},
                {"intent_key": "k1", "version": 2, "plan": "1. middle plan"},
            ]
        }

        result = find_matching_plan("example.com", "sign in")

        assert result == {"intent_key": "k1", "version": 3, "plan": "1. newest plan", "distance": 0.2}


def test_find_matching_plan_passes_domain_filter_to_query():
    with patch("backend.app.core.plan_memory.get_plan_memory_collection") as mock_get:
        collection = mock_get.return_value
        collection.query.return_value = {"ids": [[]], "metadatas": [[]], "distances": [[]]}

        find_matching_plan("www.saucedemo.com", "login")

        _, kwargs = collection.query.call_args
        assert kwargs["where"] == {"domain": "www.saucedemo.com"}
        assert kwargs["query_texts"] == ["login"]


def test_find_matching_plan_swallows_errors_and_returns_none():
    with patch(
        "backend.app.core.plan_memory.get_plan_memory_collection", side_effect=RuntimeError("Chroma down"),
    ):
        assert find_matching_plan("example.com", "login") is None


def test_find_matching_plan_returns_none_when_lineage_has_no_versions():
    """edge case: _best_match เจอ intent_key จาก query() แต่ get() (ดึง version ทั้งหมด
    ของ lineage นั้น) กลับว่างเปล่า (ไม่ควรเกิดจริง แต่กันไว้ไม่ throw/คืนค่าผิด)"""
    with patch("backend.app.core.plan_memory.get_plan_memory_collection") as mock_get:
        collection = mock_get.return_value
        collection.query.return_value = _query_result("k1", 0.1)
        collection.get.return_value = {"metadatas": []}

        assert find_matching_plan("example.com", "login") is None


# --- save_confirmed_plan() ---


def test_save_confirmed_plan_creates_new_lineage_when_no_match():
    with patch("backend.app.core.plan_memory.get_plan_memory_collection") as mock_get:
        collection = mock_get.return_value
        collection.query.return_value = {"ids": [[]], "metadatas": [[]], "distances": [[]]}

        result = save_confirmed_plan("example.com", "login", "1. Open site\n2. Log in")

        assert result["version"] == 1
        assert result["created"] is True
        _, kwargs = collection.add.call_args
        metadata = kwargs["metadatas"][0]
        assert metadata["domain"] == "example.com"
        assert metadata["version"] == 1
        assert metadata["status"] == "approved"
        assert metadata["created_by"] == "user"
        assert metadata["plan"] == "1. Open site\n2. Log in"
        assert metadata["intent_key"] == result["intent_key"]
        assert kwargs["documents"] == ["login"]


def test_save_confirmed_plan_bumps_version_when_matching_lineage_has_different_content():
    with patch("backend.app.core.plan_memory.get_plan_memory_collection") as mock_get:
        collection = mock_get.return_value
        collection.query.return_value = _query_result("k1", 0.2)
        collection.get.return_value = {"metadatas": [{"intent_key": "k1", "version": 2, "plan": "1. old plan"}]}

        result = save_confirmed_plan("example.com", "sign in", "1. New edited step")

        assert result == {"intent_key": "k1", "version": 3, "plan": "1. New edited step", "created": True}
        _, kwargs = collection.add.call_args
        assert kwargs["metadatas"][0]["version"] == 3
        assert kwargs["metadatas"][0]["intent_key"] == "k1"


def test_save_confirmed_plan_does_not_duplicate_when_content_identical_to_latest():
    """user โหลดแผนจาก Plan Memory มา (ไม่ได้แก้อะไรเลย) แล้วกด Confirm ตรงๆ — ต้องไม่
    สร้าง version ใหม่ซ้ำซ้อนเปล่าๆ คืน version เดิมตรงๆ พร้อม created=False"""
    with patch("backend.app.core.plan_memory.get_plan_memory_collection") as mock_get:
        collection = mock_get.return_value
        collection.query.return_value = _query_result("k1", 0.1)
        collection.get.return_value = {"metadatas": [{"intent_key": "k1", "version": 5, "plan": "1. same plan"}]}

        result = save_confirmed_plan("example.com", "login", "1. same plan")

        assert result == {"intent_key": "k1", "version": 5, "plan": "1. same plan", "created": False}
        collection.add.assert_not_called()


def test_save_confirmed_plan_swallows_errors_and_returns_none():
    with patch(
        "backend.app.core.plan_memory.get_plan_memory_collection", side_effect=RuntimeError("Chroma down"),
    ):
        assert save_confirmed_plan("example.com", "login", "1. Do X") is None


# --- W23 (re-applied): unsupported-script goals skip Plan Memory entirely ---
#
# เจอบั๊กจริง: chromadb DefaultEmbeddingFunction (English-only tokenizer) ยุบ text
# ภาษาไทยสองอันที่ไม่เกี่ยวข้องกันเลยให้ cosine distance ≈ 0 (แทบเหมือนกันหมด) ทำให้
# goal ภาษาไทยใหม่ใดๆ "match" กับ lineage แรกที่เคยบันทึกไว้ในโดเมนเดิมเสมอ ไม่ว่าเจตนา
# จะตรงกันจริงหรือไม่ (เช่น "ซื้อของทั้งหมด...จ่ายเงิน" ดันได้แผน "ไปที่หน้าเข้าสู่ระบบ"
# ของ goal เก่าคนละเรื่องกลับมา) — ทดสอบว่า find/save ทั้งคู่ข้าม Plan Memory ไปเลยสำหรับ
# goal ที่ใช้สคริปต์กลุ่มนี้ ไม่ไปแตะ collection.query()/add() เลยด้วยซ้ำ


def test_uses_unsupported_script_detects_thai():
    assert _uses_unsupported_script("เปิดเว็บ") is True
    assert _uses_unsupported_script("ซื้อของทั้งหมดบนหน้าเว็บ แล้วหยุดรอหน้าจ่ายเงิน") is True


def test_uses_unsupported_script_detects_other_non_latin_scripts():
    assert _uses_unsupported_script("打开网站") is True  # Chinese
    assert _uses_unsupported_script("ログイン") is True  # Japanese
    assert _uses_unsupported_script("로그인") is True  # Korean
    assert _uses_unsupported_script("تسجيل الدخول") is True  # Arabic
    assert _uses_unsupported_script("Войти") is True  # Cyrillic


def test_uses_unsupported_script_false_for_latin_text():
    assert _uses_unsupported_script("log in") is False
    assert _uses_unsupported_script("Ouvrir le site web") is False  # accented Latin (French)


def test_find_matching_plan_skips_lookup_entirely_for_thai_goal():
    with patch("backend.app.core.plan_memory.get_plan_memory_collection") as mock_get:
        result = find_matching_plan("www.saucedemo.com", "ซื้อของทั้งหมดบนหน้าเว็บ แล้วหยุดรอหน้าจ่ายเงิน")

        assert result is None
        mock_get.assert_not_called()  # ไม่แตะ ChromaDB เลยด้วยซ้ำ ไม่ใช่แค่ทิ้งผลลัพธ์


def test_save_confirmed_plan_skips_saving_entirely_for_thai_goal():
    with patch("backend.app.core.plan_memory.get_plan_memory_collection") as mock_get:
        result = save_confirmed_plan("www.saucedemo.com", "เปิดเว็บ", "1. ไปที่หน้าเข้าสู่ระบบของเว็บไซต์")

        assert result is None
        mock_get.assert_not_called()

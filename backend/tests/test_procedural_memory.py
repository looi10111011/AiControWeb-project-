from unittest.mock import patch

from backend.app.core.procedural_memory import (
    apply_template_patch,
    find_candidate_templates,
    record_template_outcome,
    render_steps_as_plan_text,
    save_template,
)

# ทุกเทสต์ mock get_procedural_memory_collection() ตรงๆ (เหมือน test_plan_memory.py) —
# ไม่โหลด embedding model จริง/ไม่แตะ ChromaDB จริงเลย เพราะสิ่งที่ต้องพิสูจน์คือ logic
# การหา lineage/version ของโมดูลนี้ ไม่ใช่พฤติกรรมจริงของ semantic search

_STEPS = [
    {"action": "goto", "target": {}},
    {"action": "fill", "target": {"accessible_name": "Username"}, "value": "{{username}}"},
    {"action": "click", "target": {"accessible_name": "Login"}},
]
_SLOTS = [{"name": "username", "description": "the username"}]


def _query_result(intent_key: str, distance: float, **extra_meta) -> dict:
    return {
        "ids": [["doc-1"]],
        "metadatas": [[{"intent_key": intent_key, "version": 1, **extra_meta}]],
        "distances": [[distance]],
    }


# --- save_template() ---


def test_save_template_creates_new_lineage_when_no_match():
    with patch("backend.app.core.procedural_memory.get_procedural_memory_collection") as mock_get:
        collection = mock_get.return_value
        collection.query.return_value = {"ids": [[]], "metadatas": [[]], "distances": [[]]}

        result = save_template("example.com", {
            "goal_pattern": "Log in with a username",
            "url_pattern": "https://example.com/login",
            "steps": _STEPS,
            "slots": _SLOTS,
        })

        assert result["version"] == 1
        _, kwargs = collection.add.call_args
        metadata = kwargs["metadatas"][0]
        assert metadata["domain"] == "example.com"
        assert metadata["version"] == 1
        assert metadata["success_count"] == 1
        assert metadata["step_signature"] == "goto/fill/click"
        assert kwargs["documents"] == ["Log in with a username"]


def test_save_template_bumps_version_when_same_signature_and_slots():
    with patch("backend.app.core.procedural_memory.get_procedural_memory_collection") as mock_get:
        collection = mock_get.return_value
        collection.query.return_value = _query_result("k1", 0.1)
        collection.get.return_value = {
            "metadatas": [{
                "intent_key": "k1", "version": 2, "success_count": 5,
                "step_signature": "goto/fill/click",
                "slot_names_json": '["username"]',
            }]
        }

        result = save_template("example.com", {
            "goal_pattern": "Log in with a username",
            "url_pattern": "https://example.com/login",
            "steps": _STEPS,
            "slots": _SLOTS,
        })

        assert result["intent_key"] == "k1"
        assert result["version"] == 3
        _, kwargs = collection.add.call_args
        assert kwargs["metadatas"][0]["success_count"] == 6  # carried forward + 1


def test_save_template_creates_new_lineage_when_step_signature_differs():
    """แม้ goal ใกล้เคียงกันมาก (distance ต่ำ) ถ้าลำดับ action ต่างกันจริง (task คนละแบบ)
    ต้องแยก lineage ใหม่ ไม่ใช่บันทึกทับ version ของ lineage เดิมที่จริงๆ เป็นคนละ
    ขั้นตอนกัน"""
    with patch("backend.app.core.procedural_memory.get_procedural_memory_collection") as mock_get:
        collection = mock_get.return_value
        collection.query.return_value = _query_result("k1", 0.1)
        collection.get.return_value = {
            "metadatas": [{
                "intent_key": "k1", "version": 1, "success_count": 1,
                "step_signature": "goto/click",  # คนละลำดับ action กับ _STEPS
                "slot_names_json": "[]",
            }]
        }

        result = save_template("example.com", {
            "goal_pattern": "Log in with a username",
            "url_pattern": "https://example.com/login",
            "steps": _STEPS,
            "slots": _SLOTS,
        })

        assert result["intent_key"] != "k1"
        assert result["version"] == 1


def test_save_template_returns_none_when_missing_goal_pattern_or_steps():
    assert save_template("example.com", {"goal_pattern": "", "steps": _STEPS}) is None
    assert save_template("example.com", {"goal_pattern": "goal", "steps": []}) is None


def test_save_template_skips_entirely_for_thai_goal_pattern():
    with patch("backend.app.core.procedural_memory.get_procedural_memory_collection") as mock_get:
        result = save_template("example.com", {
            "goal_pattern": "เข้าสู่ระบบด้วยชื่อผู้ใช้",
            "steps": _STEPS,
            "slots": _SLOTS,
        })

        assert result is None
        mock_get.assert_not_called()


def test_save_template_swallows_errors_and_returns_none():
    with patch(
        "backend.app.core.procedural_memory.get_procedural_memory_collection",
        side_effect=RuntimeError("Chroma down"),
    ):
        assert save_template("example.com", {"goal_pattern": "goal", "steps": _STEPS}) is None


# --- find_candidate_templates() ---


def test_find_candidate_templates_parses_stringified_json_fields():
    with patch("backend.app.core.procedural_memory.get_procedural_memory_collection") as mock_get:
        collection = mock_get.return_value
        collection.query.return_value = {
            "ids": [["doc-1"]],
            "metadatas": [[{
                "template_id": "t1", "intent_key": "k1", "version": 2,
                "goal_pattern": "Log in", "url_pattern": "https://example.com/login",
                "steps_json": '[{"action": "click"}]',
                "slots_json": "[]",
            }]],
            "distances": [[0.3]],
        }

        candidates = find_candidate_templates("example.com", "please log me in")

        assert len(candidates) == 1
        assert candidates[0]["template_id"] == "t1"
        assert candidates[0]["steps"] == [{"action": "click"}]
        assert candidates[0]["distance"] == 0.3


def test_find_candidate_templates_returns_empty_list_when_no_documents():
    with patch("backend.app.core.procedural_memory.get_procedural_memory_collection") as mock_get:
        mock_get.return_value.query.return_value = {"ids": [[]], "metadatas": [[]], "distances": [[]]}

        assert find_candidate_templates("example.com", "login") == []


def test_find_candidate_templates_skips_entirely_for_thai_goal():
    with patch("backend.app.core.procedural_memory.get_procedural_memory_collection") as mock_get:
        assert find_candidate_templates("example.com", "เข้าสู่ระบบ") == []
        mock_get.assert_not_called()


def test_find_candidate_templates_swallows_errors_and_returns_empty_list():
    with patch(
        "backend.app.core.procedural_memory.get_procedural_memory_collection",
        side_effect=RuntimeError("Chroma down"),
    ):
        assert find_candidate_templates("example.com", "login") == []


# --- record_template_outcome() ---


def test_record_template_outcome_increments_success_count():
    with patch("backend.app.core.procedural_memory.get_procedural_memory_collection") as mock_get:
        collection = mock_get.return_value
        collection.get.return_value = {
            "ids": ["doc-1"],
            "metadatas": [{"success_count": 3, "failure_count": 1}],
        }

        record_template_outcome("t1", success=True)

        _, kwargs = collection.update.call_args
        assert kwargs["metadatas"][0]["success_count"] == 4
        assert kwargs["metadatas"][0]["failure_count"] == 1


def test_record_template_outcome_increments_failure_count():
    with patch("backend.app.core.procedural_memory.get_procedural_memory_collection") as mock_get:
        collection = mock_get.return_value
        collection.get.return_value = {
            "ids": ["doc-1"],
            "metadatas": [{"success_count": 3, "failure_count": 1}],
        }

        record_template_outcome("t1", success=False)

        _, kwargs = collection.update.call_args
        assert kwargs["metadatas"][0]["success_count"] == 3
        assert kwargs["metadatas"][0]["failure_count"] == 2


def test_record_template_outcome_noop_when_template_id_not_found():
    with patch("backend.app.core.procedural_memory.get_procedural_memory_collection") as mock_get:
        collection = mock_get.return_value
        collection.get.return_value = {"ids": [], "metadatas": []}

        record_template_outcome("missing", success=True)

        collection.update.assert_not_called()


def test_record_template_outcome_swallows_errors():
    with patch(
        "backend.app.core.procedural_memory.get_procedural_memory_collection",
        side_effect=RuntimeError("Chroma down"),
    ):
        record_template_outcome("t1", success=True)  # must not raise


# --- render_steps_as_plan_text() ---


def test_render_steps_as_plan_text_substitutes_slots():
    text = render_steps_as_plan_text(_STEPS, {"username": "alice"})

    assert "alice" in text
    assert text.startswith("1. ")
    assert "\n2. " in text
    assert "\n3. " in text


def test_render_steps_as_plan_text_redacts_sensitive_values():
    steps = [{"action": "fill", "target": {"accessible_name": "Password"}, "value": "{{password}}", "sensitive": True}]

    text = render_steps_as_plan_text(steps, {"password": "hunter2"})

    assert "hunter2" not in text
    assert "••••••" in text


def test_render_steps_as_plan_text_leaves_unresolved_slot_as_literal():
    text = render_steps_as_plan_text(_STEPS, {})  # ไม่ส่ง username มาเลย

    assert "{{username}}" in text


# --- apply_template_patch() ---


def test_apply_template_patch_replace():
    steps = [{"action": "click"}, {"action": "fill"}]
    patch = [{"op": "replace", "index": 1, "step": {"action": "fill", "value": "new"}}]

    result = apply_template_patch(steps, patch)

    assert result == [{"action": "click"}, {"action": "fill", "value": "new"}]
    assert steps == [{"action": "click"}, {"action": "fill"}]  # original list untouched


def test_apply_template_patch_insert():
    steps = [{"action": "click"}]
    patch = [{"op": "insert", "index": 0, "step": {"action": "goto"}}]

    result = apply_template_patch(steps, patch)

    assert result == [{"action": "goto"}, {"action": "click"}]


def test_apply_template_patch_remove():
    steps = [{"action": "click"}, {"action": "hover"}]
    patch = [{"op": "remove", "index": 1}]

    result = apply_template_patch(steps, patch)

    assert result == [{"action": "click"}]


def test_apply_template_patch_ignores_out_of_range_index():
    steps = [{"action": "click"}]
    patch = [{"op": "replace", "index": 99, "step": {"action": "fill"}}]

    result = apply_template_patch(steps, patch)

    assert result == steps


def test_apply_template_patch_ignores_replace_without_step():
    steps = [{"action": "click"}]
    patch = [{"op": "replace", "index": 0}]

    result = apply_template_patch(steps, patch)

    assert result == steps


def test_apply_template_patch_handles_none_and_empty_patch():
    steps = [{"action": "click"}]

    assert apply_template_patch(steps, None) == steps
    assert apply_template_patch(steps, []) == steps

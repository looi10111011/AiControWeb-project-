"""T1-T3: ภาษาของ goal, ตัวปรับรูป goal สำหรับจับคู่, และ intent กลางตัวเดียว

goal ที่ใช้ในเทสต์ชุดนี้เป็น goal ที่ user พิมพ์จริง ไม่ใช่ตัวอย่างที่แต่งให้สวย — ทั้งการสะกด
"userole" (r ตัวเดียว) และการไม่เว้นวรรคก่อน key คือสิ่งที่เกิดขึ้นจริงและเป็นเหตุผลที่ไฟล์
goal_intent.py มีอยู่
"""

import pytest

from backend.app.core import goal_intent
from backend.app.core.goal_intent import (
    GoalIntent,
    canonical_intent,
    detect_goal_language,
    matching_tokens,
    normalize_goal_for_matching,
    plan_memory_intent_key,
    uses_unsupported_script,
)

_THAI_GOAL = "เปิดเว็ป แล้บลบuserole=ess ออกให้หมด"
_ENGLISH_GOAL = "open the site and delete all users with userrole=ess"


# ---------------- T1 ----------------

def test_detect_goal_language_separates_thai_latin_and_mixed():
    assert detect_goal_language("ลบผู้ใช้ทั้งหมด")["script"] == "thai"
    assert detect_goal_language("delete every user")["script"] == "latin"
    assert detect_goal_language(_THAI_GOAL)["script"] == "mixed"
    assert detect_goal_language("12345")["script"] == "other"


def test_word_count_shows_the_thai_tokenisation_problem_instead_of_hiding_it():
    """ภาษาไทยไม่มีเว้นวรรค — ประโยคยาวๆ จึงได้ word_count ต่ำมาก นี่คือข้อเท็จจริงที่ต้องเห็น
    ในตัวเลข ไม่ใช่ข้อบกพร่องของตัววัด เพราะมันคือสาเหตุตรงๆ ที่ตัวจับคู่แบบ token-based
    (site_learning/storage.py::find_matching_page) ใช้กับภาษาไทยไม่ได้"""
    thai = detect_goal_language("ลบผู้ใช้ทุกคนที่มีสิทธิ์เป็นพนักงานทั่วไปออกให้หมด")
    english = detect_goal_language("delete every user whose role is employee")
    assert thai["word_count"] == 1
    assert english["word_count"] > 5


def test_unsupported_script_check_still_answers_the_same_as_before():
    """plan_memory ใช้ตัวนี้เป็นประตูเดิม — ย้ายที่อยู่ของ regex ได้ แต่คำตอบต้องไม่เปลี่ยน"""
    assert uses_unsupported_script(_THAI_GOAL) is True
    assert uses_unsupported_script(_ENGLISH_GOAL) is False


# ---------------- T2 ----------------

def test_normalizer_splits_the_condition_out_of_an_unspaced_thai_sentence():
    """goal จริงคือ "แล้บลบuserole=ess" ติดกันหมด — ตัวจับคู่แบบ token เห็นเป็นก้อนเดียว
    ตัว normalizer ต้องดึงคู่ field=value ออกมาเป็น token ของตัวเองให้ได้"""
    tokens = matching_tokens(_THAI_GOAL)
    assert "ess" in tokens
    assert any("user" in t for t in tokens)


def test_normalizer_never_rewrites_the_goal_itself():
    """ผลของ normalizer ใช้เป็นกุญแจจับคู่เท่านั้น — ข้อความที่ user พิมพ์ต้องไม่ถูกแตะ
    (ถ้าเอาไปแทน goal ที่ส่งให้ LLM = เดาเจตนาแทน user)"""
    before = _THAI_GOAL
    normalize_goal_for_matching(before)
    assert before == "เปิดเว็ป แล้บลบuserole=ess ออกให้หมด"


# ---------------- T3 ----------------

def test_canonical_intent_reads_the_users_real_thai_goal():
    intent = canonical_intent(_THAI_GOAL)
    assert intent.operation == "delete"
    assert intent.scope == "all"
    assert intent.conditions == (("userole", "ess"),)
    assert intent.script == "mixed"


def test_english_goal_gives_the_same_intent_except_the_script():
    """เจตนาเดียวกันคนละภาษา ต้องได้ operation/scope เหมือนกัน — ต่างแค่ field ที่บอกภาษา"""
    thai = canonical_intent(_THAI_GOAL)
    english = canonical_intent(_ENGLISH_GOAL)
    assert (thai.operation, thai.scope) == (english.operation, english.scope)
    assert english.script == "latin"


def test_goal_without_a_condition_gets_no_plan_memory_key():
    """กุญแจต้องเข้มพอ — goal ที่ไม่มี field=value ต้องตกไปใช้เส้นทาง fallback เดิมทุกประการ
    ยอมไม่ match ดีกว่า match ผิดเจตนา (นั่นคือบั๊กที่ทำให้ต้องปิด Plan Memory ตั้งแต่แรก)"""
    assert plan_memory_intent_key("ลบผู้ใช้ทั้งหมด") is None
    assert plan_memory_intent_key("สวัสดี") is None


def test_plan_memory_key_separates_different_values_of_the_same_field():
    """userrole=ess กับ userrole=admin คือคนละงาน — กุญแจต้องแยกกัน (เคสเดียวกับที่
    _is_value_substitution_only() กันไว้ฝั่ง semantic matching)"""
    ess = plan_memory_intent_key("ลบ userrole=ess ออกให้หมด")
    admin = plan_memory_intent_key("ลบ userrole=admin ออกให้หมด")
    assert ess is not None and admin is not None
    assert ess != admin


def test_plan_memory_key_is_stable_when_thai_words_touch_the_field_name():
    """ภาษาไทยไม่มีเว้นวรรค ชื่อ field จึงมีคำไทยติดมาข้างหน้าได้ ("แล้บลบuserole") — กุญแจ
    ต้องออกมาเท่ากันไม่ว่าจะพิมพ์นำหน้าด้วยคำอะไร ไม่งั้นความจำก็ไม่มีประโยชน์"""
    assert plan_memory_intent_key("แล้บลบuserole=ess ออกให้หมด") == plan_memory_intent_key(
        "ลบ userole=ess ออกให้หมด"
    )


def test_canonical_intent_does_not_invent_a_navigate_operation():
    """ไฟล์นี้ห้ามมี logic ตัดสิน intent ใหม่ — goal ที่เป็นการนำทางล้วนจึงต้องได้ unknown
    แล้วตกไปเส้นทางเดิม ไม่ใช่ถูกเดาด้วยคลังคำชุดที่ 16"""
    assert canonical_intent("เปิดเว็บแล้วไปที่หน้าแอดมิน").operation == "unknown"


def test_as_telemetry_exposes_only_flat_json_safe_fields():
    """แถวใน token_usage.jsonl ต้อง serialize ได้ตรงๆ — ห้ามมี tuple/dataclass หลุดเข้าไป"""
    payload = canonical_intent(_THAI_GOAL).as_telemetry()
    assert payload == {
        "goal_operation": "delete",
        "goal_scope": "all",
        "goal_has_condition": True,
    }
    assert all(isinstance(v, (str, bool)) for v in payload.values())


def test_goal_intent_is_frozen_so_callers_cannot_mutate_a_shared_answer():
    with pytest.raises(Exception):
        canonical_intent(_THAI_GOAL).operation = "edit"


def test_empty_goal_does_not_raise():
    """telemetry เขียนทุกเส้นทางรวมถึง task ที่พังก่อนมี goal — ห้าม throw"""
    assert canonical_intent("") == GoalIntent(operation="unknown", scope="single", script="other")
    assert matching_tokens("") == set()
    assert goal_intent.plan_memory_intent_key("") is None


# W_thai_keyword_space (บั๊กจริงจากรันสด 2026-09-03): goal ของ user คือ
# "เปิดเว็ป แล้วเปลี่ยน รหัสผ่านเป็น 12345678" แต่คำสำคัญที่เก็บไว้คือ "เปลี่ยนรหัสผ่าน"
# — เว้นวรรคเดียวทำให้ substring match เป็นเท็จ บล็อกกฎ W20 จึงไม่ถูกส่งตอนโมเดลเลือกทาง
# แล้ว agent เดินเข้า My Info -> Memberships -> Delete ซึ่งเป็นสิ่งที่กฎนั้นห้ามไว้ตรงตัว


def test_thai_keyword_matches_no_matter_where_the_user_puts_spaces():
    keywords = ("เปลี่ยนรหัสผ่าน",)
    assert goal_intent.contains_keyword("เปิดเว็ป แล้วเปลี่ยน รหัสผ่านเป็น 12345678", keywords)
    assert goal_intent.contains_keyword("เปิดเว็ปแล้วเปลี่ยนรหัสผ่านเป็น 12345678", keywords)
    assert goal_intent.contains_keyword("เปลี่ยน  รหัส  ผ่าน ใหม่", ("เปลี่ยนรหัสผ่าน",))


def test_english_keyword_matching_is_unchanged():
    keywords = ("change password", "reset password")
    assert goal_intent.contains_keyword("Please change password now", keywords)
    assert not goal_intent.contains_keyword("open the admin page", keywords)


def test_spaceless_matching_does_not_jump_over_punctuation():
    """ตัดเฉพาะ whitespace ไม่ใช่เครื่องหมายวรรคตอน — ไม่งั้นประโยคคนละประโยคจะเชื่อมกันเอง"""
    assert not goal_intent.contains_keyword("nothing to change. Password rules are strict",
                                            ("change password",))


def test_unrelated_goals_do_not_match_by_accident():
    assert not goal_intent.contains_keyword("ลบ userrole=ess ออกให้หมด", ("เปลี่ยนรหัสผ่าน",))

from backend.app.core.actions import REJECTED_BY_USER_MESSAGE
from backend.app.core.memory import ShortTermMemory


def test_failed_actions_summary_empty_when_no_history():
    mem = ShortTermMemory()

    assert mem.failed_actions_summary() == ""


def test_failed_actions_summary_empty_when_all_steps_succeeded():
    mem = ShortTermMemory()
    mem.record({"step": 1, "cmd": {"type": "click", "index": 1}, "result": "[OK] click(1) -> สำเร็จ", "success": True})
    mem.record({"step": 2, "cmd": {"type": "fill", "index": 2}, "result": "[OK] fill(2) -> สำเร็จ", "success": True})

    assert mem.failed_actions_summary() == ""


def test_failed_actions_summary_includes_failed_step_cmd_and_result():
    mem = ShortTermMemory()
    mem.record({
        "step": 1,
        "cmd": {"type": "click", "index": 9},
        "result": "[FAIL] click(9) -> หา element ไม่เจอ",
        "success": False,
    })

    summary = mem.failed_actions_summary()

    assert "{'type': 'click', 'index': 9}" in summary
    assert "[FAIL] click(9) -> หา element ไม่เจอ" in summary


def test_failed_actions_summary_ignores_successful_steps_mixed_in():
    mem = ShortTermMemory()
    mem.record({"step": 1, "cmd": {"type": "click", "index": 1}, "result": "[OK] click(1) -> สำเร็จ", "success": True})
    mem.record({"step": 2, "cmd": {"type": "fill", "index": 2}, "result": "[FAIL] fill(2) -> พัง", "success": False})

    summary = mem.failed_actions_summary()

    assert "click(1)" not in summary
    assert "fill(2)" in summary


def test_failed_actions_summary_caps_at_max_items_keeping_most_recent():
    mem = ShortTermMemory()
    for i in range(7):
        mem.record({
            "step": i,
            "cmd": {"type": "click", "index": i},
            "result": f"[FAIL] click({i}) -> พัง",
            "success": False,
        })

    summary = mem.failed_actions_summary(max_items=3)
    lines = summary.split("\n")

    assert len(lines) == 3
    assert "click(4)" in summary
    assert "click(5)" in summary
    assert "click(6)" in summary
    assert "click(0)" not in summary


def test_failed_actions_summary_ignores_entries_missing_success_key():
    """record() ที่ไม่มี key "success" (เช่น goto record เดิมก่อน W7[A]) ต้องไม่ถูกนับ
    เป็น failure และไม่ throw"""
    mem = ShortTermMemory()
    mem.record({"step": 0, "cmd": {"type": "goto", "url": "https://example.com"}, "result": "[OK] ไปที่ url"})

    assert mem.failed_actions_summary() == ""


# --- Refusal memory (2026-07-15): action ที่ถูกมนุษย์ปฏิเสธ (human-in-the-loop)
# ต้องไม่ถูกลืม/evict ออกจาก summary เหมือน failure ทั่วไป เพราะความหมายต่างกัน —
# failure ทั่วไปเป็นแค่คำใบ้ให้ลองทางอื่น ส่วนการถูกปฏิเสธคือคำสั่งห้ามเด็ดขาดไม่ให้ทำ
# action เดิมซ้ำอีกตลอด task นี้ (ดูกติกาใหม่ใน llm.py::SYSTEM_PROMPT)


def test_failed_actions_summary_never_evicts_rejected_action_beyond_max_items():
    mem = ShortTermMemory()
    mem.record({
        "step": 1,
        "cmd": {"type": "click", "index": 9},
        "result": f"[FAIL] click(9) -> {REJECTED_BY_USER_MESSAGE}",
        "success": False,
    })
    # ตามด้วย generic failure อีกหลายตัว เกิน max_items=3 พอที่จะ evict รายการเก่าออก
    # ถ้าเป็น failure ธรรมดา (ดู test_failed_actions_summary_caps_at_max_items เดิม)
    for i in range(5):
        mem.record({
            "step": i + 2,
            "cmd": {"type": "click", "index": i},
            "result": f"[FAIL] click({i}) -> พัง",
            "success": False,
        })

    summary = mem.failed_actions_summary(max_items=3)

    assert "click(9)" in summary
    assert REJECTED_BY_USER_MESSAGE in summary
    # generic failure ยังถูก cap ตามปกติ เหลือแค่ 3 ตัวล่าสุด (index 2,3,4)
    assert "click(0)" not in summary
    assert "click(1)" not in summary
    assert "click(2)" in summary
    assert "click(3)" in summary
    assert "click(4)" in summary


def test_failed_actions_summary_dedupes_repeated_rejection_of_same_action():
    """ถ้าโมเดลฝ่าฝืนกติกาแล้วลองทำ action เดิมซ้ำ โดนปฏิเสธซ้ำอีก ต้องไม่โผล่ซ้ำกัน
    หลายบรรทัดใน summary (กัน prompt บวมไม่จำกัด)"""
    mem = ShortTermMemory()
    action = {"type": "click", "index": 9}
    for step in (1, 2, 3):
        mem.record({
            "step": step,
            "cmd": action,
            "result": f"[FAIL] click(9) -> {REJECTED_BY_USER_MESSAGE}",
            "success": False,
        })

    summary = mem.failed_actions_summary()

    assert len(summary.split("\n")) == 1


def test_failed_actions_summary_keeps_all_distinct_rejected_actions():
    mem = ShortTermMemory()
    mem.record({
        "step": 1,
        "cmd": {"type": "click", "index": 5},
        "result": f"[FAIL] click(5) -> {REJECTED_BY_USER_MESSAGE}",
        "success": False,
    })
    mem.record({
        "step": 2,
        "cmd": {"type": "delete", "index": 7},
        "result": f"[FAIL] delete(7) -> {REJECTED_BY_USER_MESSAGE}",
        "success": False,
    })

    summary = mem.failed_actions_summary()

    assert "click(5)" in summary
    assert "delete(7)" in summary

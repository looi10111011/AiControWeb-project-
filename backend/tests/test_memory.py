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

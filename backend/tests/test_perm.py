from unittest.mock import AsyncMock

import pytest

from backend.app.core.actions import execute
from backend.app.permission.rules import ActionRisk, classify_action

# adapted จาก PR "permission-ab" (origin/permission-ab) — ไฟล์เดิมเป็น manual script
# (print + if __name__ == "__main__") ไม่ใช่ pytest test จริง เขียนใหม่เป็น
# pytest.mark.asyncio + assert ให้รันเป็นส่วนหนึ่งของ test suite ได้จริง
#
# execute(None, cmd) ใช้ page=None ได้เฉพาะเคส BLOCKED/NEEDS_CONFIRMATION-rejected
# เท่านั้น เพราะ permission check คืนค่าก่อนจะแตะ page เลย — เคส SAFE ต้องมี page จริง


def test_classify_action_blocks_blocked_domain():
    assert classify_action({"type": "goto", "url": "https://malicious.com/x"}) == ActionRisk.BLOCKED


def test_classify_action_needs_confirmation_for_submit():
    assert classify_action({"type": "submit"}) == ActionRisk.NEEDS_CONFIRMATION


def test_classify_action_safe_for_normal_click():
    assert classify_action({"type": "click", "index": 0}) == ActionRisk.SAFE


@pytest.mark.asyncio
async def test_execute_blocks_goto_to_blocked_domain_without_asking_user():
    ask_user_func = AsyncMock(return_value=True)

    result = await execute(None, {"type": "goto", "url": "https://malicious.com"}, ask_user_func=ask_user_func)

    assert result.success is False
    assert "บล็อก" in result.message
    ask_user_func.assert_not_awaited()  # BLOCKED ปฏิเสธทันที ไม่ต้องถามด้วยซ้ำ


@pytest.mark.asyncio
async def test_execute_asks_user_before_needs_confirmation_action_and_respects_approval():
    ask_user_func = AsyncMock(return_value=True)
    cmd = {"type": "submit"}

    result = await execute(None, cmd, ask_user_func=ask_user_func)

    ask_user_func.assert_awaited_once_with(cmd)
    # submit ไม่ได้ implement ใน dispatch จริง (ไม่มี t == "submit" case) เลยตกไป
    # "ไม่รู้จัก action นี้" — แต่สำคัญคือผ่าน permission gate ไปถึง dispatch ได้ (ไม่ถูก
    # ปฏิเสธเพราะ human-in-the-loop) ยืนยันว่า approval ถูก respect จริง
    assert result.message != "ผู้ใช้ปฏิเสธการทำ Action นี้ (Human-in-the-loop)"


@pytest.mark.asyncio
async def test_execute_rejects_needs_confirmation_action_when_user_declines():
    ask_user_func = AsyncMock(return_value=False)

    result = await execute(None, {"type": "submit"}, ask_user_func=ask_user_func)

    assert result.success is False
    assert "ปฏิเสธ" in result.message

from unittest.mock import AsyncMock

import pytest

from backend.app.core.actions import execute
from backend.app.permission.rules import ActionRisk, classify_action

# adapted จาก PR "permission-ab" (origin/permission-ab) — ไฟล์เดิมเป็น manual script
# (print + if __name__ == "__main__") ไม่ใช่ pytest test จริง เขียนใหม่เป็น
# pytest.mark.asyncio + assert ให้รันเป็นส่วนหนึ่งของ test suite ได้จริง
#
# execute(None, cmd) ใช้ page=None ได้เฉพาะเคส BLOCKED/NEEDS_CONFIRMATION-rejected
# เท่านั้น เพราะ permission check คืนค่าก่อนจะแตะ page เลย — เคส SAFE/approved ต้องมี
# page จริง (หรือ mock ที่มี click()/fill()) เพราะ dispatch ไปเรียก page.click() จริง


def test_classify_action_blocks_blocked_domain():
    assert classify_action({"type": "goto", "url": "https://malicious.com/x"}) == ActionRisk.BLOCKED


def test_classify_action_safe_for_allowed_goto_even_with_risky_looking_label():
    """goto ที่ผ่าน domain check แล้วต้องเป็น SAFE ทันที ไม่ตกไปเช็ค label ต่อ —
    ระบบต้อง goto ไปหน้าเว็บก่อนถึงจะเห็นฟอร์ม/element อะไรเลย เอา label (ซึ่งปกติ
    ว่างเปล่าสำหรับ goto อยู่แล้วเพราะไม่มี index) มาตัดสิน risk ของการ "ไปหน้าเว็บ"
    ไม่ได้ — เทสต์นี้จงใจส่ง label ที่ตรงคำเสี่ยงมาด้วยเพื่อพิสูจน์ว่าไม่มีผลกับ goto"""
    cmd = {"type": "goto", "url": "https://www.saucedemo.com/"}
    assert classify_action(cmd, label="Remove") == ActionRisk.SAFE


def test_classify_action_needs_confirmation_for_submit():
    assert classify_action({"type": "submit"}) == ActionRisk.NEEDS_CONFIRMATION


def test_classify_action_safe_for_normal_click():
    assert classify_action({"type": "click", "index": 0}) == ActionRisk.SAFE


# ชั้นสำรอง (defense-in-depth): LLM อาจส่ง type="click" ธรรมดาสำหรับปุ่มที่จริงๆ มีผล
# สำคัญ (เช่น saucedemo "Remove" เป็นแค่ <button>Remove</button> ไม่มี type พิเศษเลย) —
# classify_action() ต้องจับได้จาก label แม้ type จะเป็นแค่ "click"


def test_classify_action_needs_confirmation_for_click_with_risky_label():
    assert classify_action({"type": "click", "index": 5}, label="Remove") == ActionRisk.NEEDS_CONFIRMATION


def test_classify_action_needs_confirmation_for_click_with_risky_label_case_insensitive():
    assert classify_action({"type": "click", "index": 5}, label="FINISH") == ActionRisk.NEEDS_CONFIRMATION


def test_classify_action_safe_for_click_with_ordinary_label():
    assert classify_action({"type": "click", "index": 5}, label="Add to cart") == ActionRisk.SAFE


def test_classify_action_safe_when_label_not_provided():
    # ไม่ส่ง label มาเลย (default "") ต้องไม่ throw และไม่ถือว่าเสี่ยง
    assert classify_action({"type": "click", "index": 5}) == ActionRisk.SAFE


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
    mock_page = AsyncMock()
    cmd = {"type": "submit", "index": 3}

    result = await execute(mock_page, cmd, ask_user_func=ask_user_func)

    ask_user_func.assert_awaited_once_with(cmd)
    # submit/delete/purchase/pay ไม่ใช่ action จริงแยกต่างหาก — เป็นแค่ risk category
    # ที่ alias ไปเรียก click() ตัวเดิม (เช็ค permission ผ่านแล้วด้านบน) แค่ต้องขอยืนยัน
    # ก่อนเพราะเสี่ยงกว่า click ธรรมดา
    assert result.success is True
    assert result.action == "submit(3)"
    mock_page.click.assert_awaited_once()


@pytest.mark.asyncio
async def test_execute_rejects_needs_confirmation_action_when_user_declines():
    ask_user_func = AsyncMock(return_value=False)

    result = await execute(None, {"type": "submit", "index": 3}, ask_user_func=ask_user_func)

    assert result.success is False
    assert "ปฏิเสธ" in result.message


@pytest.mark.asyncio
async def test_execute_asks_user_for_plain_click_with_risky_label():
    """type="click" ธรรมดา (ไม่ใช่ submit/delete/purchase/pay) แต่ label ตรงคำเสี่ยง —
    ต้องขอยืนยันเหมือนกัน ไม่ใช่ผ่านฉลุยเพราะ type ดูไม่เสี่ยง"""
    ask_user_func = AsyncMock(return_value=False)
    cmd = {"type": "click", "index": 7}

    result = await execute(None, cmd, ask_user_func=ask_user_func, label="Remove")

    ask_user_func.assert_awaited_once_with(cmd)
    assert result.success is False
    assert "ปฏิเสธ" in result.message


@pytest.mark.asyncio
async def test_execute_does_not_ask_user_for_plain_click_with_ordinary_label():
    mock_page = AsyncMock()
    ask_user_func = AsyncMock()
    cmd = {"type": "click", "index": 8}

    result = await execute(mock_page, cmd, ask_user_func=ask_user_func, label="Add to cart")

    assert result.success is True
    mock_page.click.assert_awaited_once()
    ask_user_func.assert_not_awaited()


# W7[B]: RAG-based permission — คู่มือ (manual_guidance, มาจาก manual_context ที่
# orchestrator ดึงมาให้ planner อยู่แล้วตั้งแต่ W6[B]) อาจกำหนดเองว่า action ไหนต้อง
# ขออนุมัติเพิ่มจาก DEFAULT_NEEDS_CONFIRMATION/RISKY_LABEL_KEYWORDS ที่ hardcode ไว้


def test_classify_action_needs_confirmation_when_manual_says_requires_approval():
    cmd = {"type": "click", "index": 9}
    manual = "- นโยบายร้าน: การสั่งซื้อเกิน $100 requires approval จากผู้จัดการก่อนเสมอ"
    assert classify_action(cmd, manual_guidance=manual) == ActionRisk.NEEDS_CONFIRMATION


def test_classify_action_needs_confirmation_when_manual_says_requires_approval_thai():
    cmd = {"type": "click", "index": 9}
    manual = "- คำสั่งซื้อทุกรายการต้องขออนุมัติจากหัวหน้างานก่อนกดยืนยัน"
    assert classify_action(cmd, manual_guidance=manual) == ActionRisk.NEEDS_CONFIRMATION


def test_classify_action_safe_when_manual_guidance_unrelated_to_approval():
    cmd = {"type": "click", "index": 9}
    manual = "- หน้านี้แสดงรายการสินค้าเรียงตามราคา"
    assert classify_action(cmd, manual_guidance=manual) == ActionRisk.SAFE


def test_classify_action_safe_when_manual_guidance_not_provided():
    # ไม่ส่ง manual_guidance มาเลย (default "") ต้องไม่ throw และไม่ถือว่าเสี่ยง
    assert classify_action({"type": "click", "index": 9}) == ActionRisk.SAFE


def test_classify_action_needs_confirmation_for_goto_when_manual_requires_approval():
    """goto ที่ผ่าน domain check แล้วยังต้องเช็คคู่มือต่อ (ต่างจาก label ที่ข้ามไปเลย) —
    เช่น คู่มือบอกว่าการไปหน้า admin ต้องขออนุมัติก่อน"""
    cmd = {"type": "goto", "url": "https://www.saucedemo.com/admin"}
    manual = "- การเข้าหน้า admin ต้องได้รับอนุมัติจากทีมความปลอดภัยก่อนเสมอ"
    assert classify_action(cmd, manual_guidance=manual) == ActionRisk.NEEDS_CONFIRMATION


@pytest.mark.asyncio
async def test_execute_asks_user_for_plain_click_when_manual_requires_approval():
    """type="click" ธรรมดา + label ปกติ (ไม่เสี่ยง) แต่คู่มือระบุว่าต้องขออนุมัติ —
    ต้องขอยืนยันเหมือนกัน ไม่ใช่พึ่ง label/type อย่างเดียว"""
    ask_user_func = AsyncMock(return_value=False)
    cmd = {"type": "click", "index": 9}

    result = await execute(
        None, cmd, ask_user_func=ask_user_func, label="Checkout",
        manual_guidance="- การกด Checkout ทุกครั้ง requires approval จากหัวหน้างาน",
    )

    ask_user_func.assert_awaited_once_with(cmd)
    assert result.success is False
    assert "ปฏิเสธ" in result.message


@pytest.mark.asyncio
async def test_execute_does_not_ask_user_when_manual_guidance_not_provided():
    mock_page = AsyncMock()
    ask_user_func = AsyncMock()
    cmd = {"type": "click", "index": 10}

    result = await execute(mock_page, cmd, ask_user_func=ask_user_func, label="Checkout")

    assert result.success is True
    mock_page.click.assert_awaited_once()
    ask_user_func.assert_not_awaited()

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from playwright.async_api import async_playwright

from backend.app.core.actions import execute
from backend.app.permission.rules import (
    ALLOWED_DOMAINS,
    ActionRisk,
    classify_action,
    install_ssrf_guard,
    is_private_or_internal,
)

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


# W_search: บั๊กจริงที่ user รายงาน — LLM บางครั้งเลือก action type "submit" ให้ปุ่ม
# ค้นหา/เปิดดูวิดีโอ (ตีความ "ค้นหา" ว่าเป็นการ "ส่งฟอร์ม" ทางความหมาย) ทั้งที่ย้อนกลับ
# ได้ง่ายมาก ไม่มีผลถาวรใดๆ ต่างจาก submit ที่แท้จริงเสี่ยง (เช่น place order/checkout)
# — ต้องเช็ค label ก่อนเสมอแม้ action_type จะดูเสี่ยง (defense-in-depth สวนทางกับ
# RISKY_LABEL_KEYWORDS)


def test_classify_action_safe_for_submit_with_search_label():
    assert classify_action({"type": "submit"}, label="Search") == ActionRisk.SAFE


def test_classify_action_safe_for_submit_with_watch_video_label():
    assert classify_action({"type": "submit"}, label="Watch") == ActionRisk.SAFE


def test_classify_action_safe_for_submit_with_thai_search_label():
    assert classify_action({"type": "submit"}, label="ค้นหา") == ActionRisk.SAFE


def test_classify_action_still_needs_confirmation_for_submit_with_no_label():
    # ไม่มี label ให้เช็คเลย (ว่างเปล่า) — ไม่มีข้อมูลพอจะลดระดับ ต้อง fail-safe เป็น
    # NEEDS_CONFIRMATION เหมือนเดิมทุกประการ (ไม่ใช่ auto-safe ทุก submit)
    assert classify_action({"type": "submit"}, label="") == ActionRisk.NEEDS_CONFIRMATION


def test_classify_action_needs_confirmation_when_label_matches_both_safe_and_risky_keywords():
    # label ที่ดู "ปลอดภัย" บางส่วนแต่ก็มีคำเสี่ยงปนอยู่ด้วย (เช่นอยู่ในหน้า checkout จริง)
    # ต้องระวังไว้ก่อนเสมอ — ฝั่งเสี่ยงชนะ
    assert classify_action({"type": "submit"}, label="Confirm and Search Orders") == ActionRisk.NEEDS_CONFIRMATION


def test_classify_action_manual_guidance_still_wins_over_safe_label():
    # คู่มือที่ user ตั้งไว้เองต้องชนะเสมอ แม้ label จะดูปลอดภัยแค่ไหนก็ตาม
    cmd = {"type": "submit"}
    manual = "- การค้นหาทุกครั้งต้องขออนุมัติจากหัวหน้างานก่อนเสมอ"
    assert classify_action(cmd, label="Search", manual_guidance=manual) == ActionRisk.NEEDS_CONFIRMATION


def test_classify_action_safe_for_delete_type_with_safe_label():
    # ครอบคลุมทั้ง 4 action type ใน DEFAULT_NEEDS_CONFIRMATION ไม่ใช่แค่ submit —
    # ในทางปฏิบัติ delete/purchase/pay จริงๆ แทบไม่มีทาง match safe label ได้เลย (label
    # ของปุ่มลบ/ซื้อ/จ่ายเงินจริงไม่ใช่คำว่า "search"/"watch") แต่ยืนยันว่า logic ใช้ร่วมกัน
    # ทั้ง 4 ประเภทสม่ำเสมอ
    assert classify_action({"type": "delete"}, label="View details") == ActionRisk.SAFE


def test_classify_action_safe_for_normal_click():
    assert classify_action({"type": "click", "index": 0}) == ActionRisk.SAFE


# W_search follow-up: บั๊กจริงที่ user รายงานต่อ — คลิกวิดีโอ YouTube จากผลการค้นหา (LLM
# เผลอเลือก type="submit"/"purchase" ให้กับการคลิกเลือกรายการ) label ในเคสนี้เป็นชื่อ
# วิดีโอดิบๆ (เช่น "เพลงรัก - Three Man Down |Official MV|") ไม่ match ทั้ง
# SAFE_ACTION_LABEL_KEYWORDS และ RISKY_LABEL_KEYWORDS เลย — ต้องใช้ element_tag (tag=="a")
# เป็นสัญญาณสำรองชั้นสุดท้ายแทน


def test_classify_action_safe_for_submit_type_on_anchor_with_arbitrary_title_label():
    cmd = {"type": "submit", "index": 12}
    label = "เพลงรัก - Three Man Down |Official MV|"
    assert classify_action(cmd, label=label, element_tag="a") == ActionRisk.SAFE


def test_classify_action_needs_confirmation_for_submit_type_on_non_anchor_with_arbitrary_label():
    # tag ไม่ใช่ "a" (เช่น <button>) + label ไม่ match ทั้งสองฝั่ง — ไม่มีสัญญาณปลอดภัยเลย
    # ต้อง fail-safe เป็น NEEDS_CONFIRMATION เหมือนเดิม
    cmd = {"type": "submit", "index": 12}
    assert classify_action(cmd, label="เพลงรัก - Three Man Down |Official MV|", element_tag="button") == (
        ActionRisk.NEEDS_CONFIRMATION
    )


def test_classify_action_risky_label_still_wins_over_anchor_tag():
    # ป้องกันเคสหายาก: ปุ่ม "Place Order" ที่ทำเป็น <a> ตกแต่งด้วย CSS ให้ดูเหมือนปุ่ม —
    # label เสี่ยงต้องชนะ tag=="a" เสมอ
    cmd = {"type": "purchase", "index": 12}
    assert classify_action(cmd, label="Place Order", element_tag="a") == ActionRisk.NEEDS_CONFIRMATION


def test_classify_action_no_anchor_signal_without_element_tag():
    # element_tag ไม่ส่งมา (default "") ต้องไม่ auto-safe จาก tag เปล่าๆ — พฤติกรรมเดิม
    cmd = {"type": "submit", "index": 12}
    assert classify_action(cmd, label="เพลงรัก - Three Man Down |Official MV|") == ActionRisk.NEEDS_CONFIRMATION


# W_search follow-up 2: บั๊กจริงอีกเคส — LLM เลือก type="submit" ให้กับ action "กด Enter
# เพื่อค้นหา" (เป้าหมายยังเป็นช่องค้นหาเดิม) หลังจากพิมพ์คำค้นหาไปแล้ว label ของช่องนั้น
# กลายเป็นคำค้นหาดิบๆ (เช่น "เพลงรัก") ไม่ match ทั้ง SAFE_ACTION_LABEL_KEYWORDS และ
# RISKY_LABEL_KEYWORDS เลย (เหมือนเคส anchor ด้านบนแต่คนละ element type) — ต้องใช้
# element_tag=="input" (ไม่ใช่ type เสี่ยง) เป็นสัญญาณสำรองชั้นสุดท้ายแทน


def test_classify_action_safe_for_submit_type_on_text_input_with_arbitrary_query_label():
    cmd = {"type": "submit", "index": 3}
    assert classify_action(cmd, label="เพลงรัก", element_tag="input", element_type="text") == ActionRisk.SAFE


def test_classify_action_safe_for_submit_type_on_search_input_with_no_type_attribute():
    # <input> ที่ไม่มี type attribute ระบุตรงๆ เลย (getAttribute คืนค่าว่าง) — ยังถือว่า
    # ปลอดภัย (ไม่ใช่ submit/image/password ที่แท้จริง)
    cmd = {"type": "submit", "index": 3}
    assert classify_action(cmd, label="เพลงรัก", element_tag="input", element_type="") == ActionRisk.SAFE


def test_classify_action_needs_confirmation_for_submit_type_input_with_risky_input_type():
    # <input type="submit"> จริงๆ (ปุ่ม submit ของฟอร์มในคราบ input) ต้อง fail-safe
    cmd = {"type": "submit", "index": 3}
    assert classify_action(cmd, label="เพลงรัก", element_tag="input", element_type="submit") == (
        ActionRisk.NEEDS_CONFIRMATION
    )


def test_classify_action_needs_confirmation_for_purchase_type_password_input():
    # <input type="password"> ระวังไว้ก่อน (อาจเป็นส่วนหนึ่งของ flow ยืนยันตัวตนก่อนจ่ายเงิน)
    cmd = {"type": "purchase", "index": 3}
    assert classify_action(cmd, label="", element_tag="input", element_type="password") == (
        ActionRisk.NEEDS_CONFIRMATION
    )


def test_classify_action_risky_label_still_wins_over_safe_input_tag():
    cmd = {"type": "purchase", "index": 3}
    assert classify_action(cmd, label="Confirm Payment", element_tag="input", element_type="text") == (
        ActionRisk.NEEDS_CONFIRMATION
    )


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

    # element_label แนบเข้าไปให้ ask_user_func เห็นชื่อ element จริงด้วย (ไม่ใช่แค่
    # index) — cmd ต้นฉบับที่ dispatch จริงยังไม่ถูกแตะ (ดู actions.py::_confirm_action)
    ask_user_func.assert_awaited_once_with({**cmd, "element_label": "Remove"})
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

    ask_user_func.assert_awaited_once_with({**cmd, "element_label": "Checkout"})
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


# per-call allowed_domains override (real-user-browser mode, core/user_browser.py) —
# ต้อง override ALLOWED_DOMAINS เฉพาะ call นั้นๆ โดยไม่แตะ module-level global เลย กัน
# task อื่น/thread อื่นที่ใช้ classify_action() พร้อมกันไม่ได้รับผลกระทบ


def test_classify_action_blocks_domain_not_in_per_call_allowed_domains():
    cmd = {"type": "goto", "url": "https://www.saucedemo.com/"}
    assert classify_action(cmd, allowed_domains={"mail.google.com"}) == ActionRisk.BLOCKED


def test_classify_action_allows_domain_in_per_call_allowed_domains():
    cmd = {"type": "goto", "url": "https://mail.google.com/mail/u/0/"}
    assert classify_action(cmd, allowed_domains={"mail.google.com"}) == ActionRisk.SAFE


def test_classify_action_per_call_allowed_domains_overrides_global_allowed_domains():
    # module-level ALLOWED_DOMAINS ว่างเปล่า (อนุญาตทุกโดเมนที่ไม่ได้ blocklist) แต่
    # per-call allowed_domains ต้อง "แคบกว่า" เดิมได้จริง ไม่ใช่แค่ขยายเพิ่ม
    assert ALLOWED_DOMAINS == set()  # sanity check ค่า default ของ module ตอนนี้
    cmd = {"type": "goto", "url": "https://www.saucedemo.com/"}
    assert classify_action(cmd, allowed_domains={"mail.google.com"}) == ActionRisk.BLOCKED


def test_classify_action_none_allowed_domains_preserves_legacy_global_behavior():
    # allowed_domains=None (ไม่ส่งมา) = พฤติกรรมเดิมทุกประการ (ใช้ ALLOWED_DOMAINS ของ
    # module ซึ่งว่างเปล่า = ไม่จำกัดโดเมนเลย)
    cmd = {"type": "goto", "url": "https://www.saucedemo.com/"}
    assert classify_action(cmd) == ActionRisk.SAFE
    assert classify_action(cmd, allowed_domains=None) == ActionRisk.SAFE


def test_classify_action_blocked_domains_global_still_applies_with_per_call_allowed_domains():
    # BLOCKED_DOMAINS (module-level) ยังคงเป็น hard block เสมอ ต่อให้ per-call
    # allowed_domains จะอนุญาตโดเมนนั้นไว้ก็ตาม
    cmd = {"type": "goto", "url": "https://malicious.com/x"}
    assert classify_action(cmd, allowed_domains={"malicious.com"}) == ActionRisk.BLOCKED


@pytest.mark.asyncio
async def test_execute_forwards_allowed_domains_to_classify_action():
    result = await execute(
        None, {"type": "goto", "url": "https://mail.google.com/"},
        allowed_domains={"www.saucedemo.com"},
    )
    assert result.success is False
    assert "บล็อก" in result.message


# --- Security 1.2 (SSRF): is_private_or_internal() + goto ไปยัง private/internal IP ---


def test_is_private_or_internal_true_for_cloud_metadata_ip():
    assert is_private_or_internal("169.254.169.254") is True


def test_is_private_or_internal_true_for_rfc1918_ranges():
    assert is_private_or_internal("10.0.0.5") is True
    assert is_private_or_internal("172.16.0.5") is True
    assert is_private_or_internal("192.168.1.1") is True


def test_is_private_or_internal_true_for_loopback():
    assert is_private_or_internal("127.0.0.1") is True
    assert is_private_or_internal("localhost") is True
    assert is_private_or_internal("::1") is True


def test_is_private_or_internal_false_for_public_ip():
    assert is_private_or_internal("8.8.8.8") is False


def test_is_private_or_internal_false_for_public_domain():
    assert is_private_or_internal("example.com") is False


def test_is_private_or_internal_false_for_unresolvable_hostname():
    assert is_private_or_internal("this-domain-does-not-exist-xyz123.invalid") is False


def test_is_private_or_internal_blocks_any_private_dns_answer():
    answers = [
        (None, None, None, None, ("8.8.8.8", 0)),
        (None, None, None, None, ("fd00::1", 0, 0, 0)),
    ]
    with patch("backend.app.permission.rules.socket.getaddrinfo", return_value=answers):
        assert is_private_or_internal("mixed.example") is True


def test_classify_action_blocks_goto_to_cloud_metadata_ip():
    cmd = {"type": "goto", "url": "http://169.254.169.254/"}
    assert classify_action(cmd) == ActionRisk.BLOCKED


def test_classify_action_blocks_goto_to_private_lan_ip():
    cmd = {"type": "goto", "url": "http://192.168.1.1/admin"}
    assert classify_action(cmd) == ActionRisk.BLOCKED


def test_classify_action_blocks_goto_to_localhost():
    cmd = {"type": "goto", "url": "http://localhost:8000/"}
    assert classify_action(cmd) == ActionRisk.BLOCKED


def test_classify_action_ssrf_block_wins_even_when_domain_in_allowed_domains():
    """defense-in-depth: SSRF block ต้องเช็คก่อน ALLOWED_DOMAINS เสมอ — แม้ caller จะ
    allowlist โดเมนนี้ไว้ (ไม่มีทาง allowlist internal IP ได้จริง)"""
    cmd = {"type": "goto", "url": "http://169.254.169.254/"}
    assert classify_action(cmd, allowed_domains={"169.254.169.254"}) == ActionRisk.BLOCKED


def test_classify_action_allows_internal_navigation_when_setting_enabled(monkeypatch):
    from backend.app.config import settings

    monkeypatch.setattr(settings, "allow_internal_navigation", True)
    cmd = {"type": "goto", "url": "http://localhost:8000/"}
    assert classify_action(cmd) == ActionRisk.SAFE


def test_classify_action_goto_public_domain_still_safe():
    """sanity: goto ไปโดเมนสาธารณะปกติต้องไม่ถูกกระทบจากการเพิ่ม SSRF check เลย"""
    cmd = {"type": "goto", "url": "https://example.com/"}
    assert classify_action(cmd) == ActionRisk.SAFE


# --- Security 1.3: SAFE-downgrade heuristic (tag/structure-based) จำกัดเฉพาะ
# action_type=="submit" เท่านั้น — delete/purchase/pay ต้อง fallthrough ไป
# NEEDS_CONFIRMATION เสมอถ้าไม่ match risky/safe label แม้จะเป็น <a>/plain <input> ก็ตาม
# (เดิม heuristic นี้ครอบคลุมทั้ง 4 action type ทำให้หน้าเว็บทำปุ่ม "ลบ"/"สั่งซื้อ" เป็น <a>
# label ทั่วไปหลบ confirmation ได้ — บั๊กจริงที่พบ)


def test_classify_action_needs_confirmation_for_delete_type_anchor_with_arbitrary_label():
    cmd = {"type": "delete", "index": 1}
    assert classify_action(cmd, label="Continue", element_tag="a") == ActionRisk.NEEDS_CONFIRMATION


def test_classify_action_needs_confirmation_for_purchase_type_anchor_with_arbitrary_label():
    cmd = {"type": "purchase", "index": 1}
    assert classify_action(cmd, label="Continue", element_tag="a") == ActionRisk.NEEDS_CONFIRMATION


def test_classify_action_needs_confirmation_for_pay_type_anchor_with_arbitrary_label():
    cmd = {"type": "pay", "index": 1}
    assert classify_action(cmd, label="Continue", element_tag="a") == ActionRisk.NEEDS_CONFIRMATION


def test_classify_action_needs_confirmation_for_delete_type_plain_input_with_arbitrary_label():
    cmd = {"type": "delete", "index": 1}
    assert classify_action(
        cmd, label="xyz", element_tag="input", element_type="text",
    ) == ActionRisk.NEEDS_CONFIRMATION


def test_classify_action_needs_confirmation_for_purchase_type_plain_input_with_arbitrary_label():
    cmd = {"type": "purchase", "index": 1}
    assert classify_action(
        cmd, label="xyz", element_tag="input", element_type="text",
    ) == ActionRisk.NEEDS_CONFIRMATION


def test_classify_action_delete_type_still_safe_when_label_matches_safe_keyword():
    """sanity: delete type ที่ label match SAFE_ACTION_LABEL_KEYWORDS จริงๆ (ไม่ใช่ tag
    downgrade) ยังต้องเป็น SAFE เหมือนเดิม — ไม่ถูกกระทบจากการจำกัด scope นี้เลย"""
    cmd = {"type": "delete", "index": 1}
    assert classify_action(cmd, label="View details") == ActionRisk.SAFE


def test_classify_action_submit_type_anchor_downgrade_still_works_after_narrowing():
    """sanity: 2 เคส false-positive เดิม (W_search follow-up) เกิดกับ action_type=="submit"
    เท่านั้น — ต้องยังทำงานเหมือนเดิมทุกประการหลังจำกัด scope การ downgrade"""
    cmd = {"type": "submit", "index": 12}
    label = "เพลงรัก - Three Man Down |Official MV|"
    assert classify_action(cmd, label=label, element_tag="a") == ActionRisk.SAFE


# --- Security follow-up: install_ssrf_guard()/_ssrf_route_handler() — SSRF check ที่ชั้น
# เครือข่ายจริง แก้ 2 ช่องโหว่ที่ classify_action() (เช็คแค่ action_type=="goto") ปิดไม่ถึง:
# (1) agent คลิกลิงก์ไป internal IP แทนที่จะ goto ตรงๆ (2) goto ไปโดเมนสาธารณะที่ redirect
# ไปยัง internal IP (open redirect/DNS rebinding) — ดู module comment เต็มใน rules.py


def _make_mock_route(url: str, is_navigation: bool):
    request = MagicMock()
    request.url = url
    request.is_navigation_request = MagicMock(return_value=is_navigation)
    route = MagicMock()
    route.request = request
    route.abort = AsyncMock()
    route.continue_ = AsyncMock()
    return route


@pytest.mark.asyncio
async def test_ssrf_route_handler_aborts_navigation_to_private_ip():
    from backend.app.permission.rules import _ssrf_route_handler

    route = _make_mock_route("http://169.254.169.254/latest/meta-data/", is_navigation=True)
    await _ssrf_route_handler(route)
    route.abort.assert_awaited_once()
    route.continue_.assert_not_awaited()


@pytest.mark.asyncio
async def test_ssrf_route_handler_allows_navigation_to_public_domain():
    from backend.app.permission.rules import _ssrf_route_handler

    route = _make_mock_route("https://example.com/", is_navigation=True)
    await _ssrf_route_handler(route)
    route.continue_.assert_awaited_once()
    route.abort.assert_not_awaited()


@pytest.mark.asyncio
async def test_ssrf_route_handler_skips_check_for_non_navigation_subresource():
    """W: จำกัดเช็คแค่ is_navigation_request() เท่านั้น (ดู module comment ใน rules.py
    เรื่อง trade-off performance) — subresource (css/js/image/xhr) แม้ไป private IP ก็
    ปล่อยผ่านไม่เช็คเลย ไม่เสียเวลา resolve DNS เพิ่มทุก request ย่อยของทุกหน้า"""
    from backend.app.permission.rules import _ssrf_route_handler

    route = _make_mock_route("http://169.254.169.254/evil.js", is_navigation=False)
    await _ssrf_route_handler(route)
    route.continue_.assert_awaited_once()
    route.abort.assert_not_awaited()


@pytest.mark.asyncio
async def test_ssrf_route_handler_respects_allow_internal_navigation_setting(monkeypatch):
    from backend.app.config import settings
    from backend.app.permission.rules import _ssrf_route_handler

    monkeypatch.setattr(settings, "allow_internal_navigation", True)
    route = _make_mock_route("http://169.254.169.254/", is_navigation=True)
    await _ssrf_route_handler(route)
    route.continue_.assert_awaited_once()
    route.abort.assert_not_awaited()


@pytest.mark.asyncio
async def test_install_ssrf_guard_blocks_real_goto_to_internal_ip():
    """integration เต็มสาย: install_ssrf_guard() บน Page จริง + page.goto() จริงไป
    127.0.0.1 (private ตามจริง — ไม่ต้อง mock is_private_or_internal เลย) ต้องถูก block
    จริง (route.abort() ทำให้ Playwright โยน error แทนที่จะโหลดหน้าสำเร็จ)"""
    import http.server
    import threading

    httpd = http.server.HTTPServer(("127.0.0.1", 0), http.server.SimpleHTTPRequestHandler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            await install_ssrf_guard(page)
            with pytest.raises(Exception):
                await page.goto(f"http://127.0.0.1:{port}/", timeout=5000)
            await browser.close()
    finally:
        httpd.shutdown()


@pytest.mark.asyncio
async def test_install_ssrf_guard_allows_goto_when_internal_navigation_enabled(monkeypatch):
    """sanity: settings.allow_internal_navigation=True (dev ที่ตั้งใจทดสอบเว็บ local) ต้อง
    ไม่บล็อก goto ไปยัง 127.0.0.1 เหมือนเดิม — guard เป็น escape hatch จริง ไม่ใช่ hard-block
    ตายตัวที่ปิดไม่ได้เลย"""
    import http.server
    import threading

    from backend.app.config import settings

    monkeypatch.setattr(settings, "allow_internal_navigation", True)

    httpd = http.server.HTTPServer(("127.0.0.1", 0), http.server.SimpleHTTPRequestHandler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            await install_ssrf_guard(page)
            resp = await page.goto(f"http://127.0.0.1:{port}/", timeout=5000)
            assert resp is not None and resp.ok
            await browser.close()
    finally:
        httpd.shutdown()


@pytest.mark.asyncio
async def test_install_ssrf_guard_blocks_click_driven_navigation_to_internal_ip(tmp_path):
    """Security SEC-1: SSRF ผ่านการ "คลิก" ลิงก์ (ไม่ใช่ goto ตรงๆ) ต้องถูก block เหมือนกัน
    — จำลองหน้าเว็บที่มีลิงก์ไป internal target (ทั้งคู่อยู่บน 127.0.0.1 เดียวกัน ดังนั้น
    "คลิกลิงก์" นี้คือ navigation ไปยัง private IP จริง เหมือนกรณี prompt injection ที่ฝัง
    <a href="http://169.254.169.254/..."> หลอกให้ agent คลิก)"""
    import functools
    import http.server
    import threading

    index_path = tmp_path / "index.html"
    index_path.write_text('<html><body><a id="go" href="/target.html">click</a></body></html>', encoding="utf-8")
    (tmp_path / "target.html").write_text("<html><body>should never load</body></html>", encoding="utf-8")

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(tmp_path))
    httpd = http.server.HTTPServer(("127.0.0.1", 0), handler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            # โหลดหน้าแรกได้ตามปกติก่อน (allow_internal_navigation ชั่วคราวแค่ตอน goto นี้
            # ไม่งั้นแม้แต่หน้า index เองก็โหลดไม่ได้เพราะอยู่บน 127.0.0.1 เหมือนกัน) —
            # จำลองสถานการณ์จริงที่ agent อยู่บนหน้าเว็บสาธารณะที่ถูกฝัง prompt injection
            # แล้วโดนหลอกให้คลิกลิงก์ไป private IP โดยไม่ได้ตั้งใจ goto ไปเองตรงๆ
            await page.goto(f"http://127.0.0.1:{port}/index.html", timeout=5000)
            await install_ssrf_guard(page)
            await page.click("#go")
            await page.wait_for_timeout(300)
            assert "should never load" not in await page.content()
            await browser.close()
    finally:
        httpd.shutdown()

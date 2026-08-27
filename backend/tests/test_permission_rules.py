import pytest

from backend.app.permission.rules import ActionRisk, classify_action, extract_domain, normalize_domain
from backend.app.site_learning import storage


# --- extract_domain(): www./ไม่มี www. ต้องถือเป็น domain เดียวกันเป๊ะ (กันบั๊ก credential
# หาไม่เจอข้าม www./non-www ของเว็บเดียวกัน — ดู docstring extract_domain()) ---


def test_extract_domain_strips_www_prefix():
    assert extract_domain("https://www.example.com/login") == "example.com"


def test_extract_domain_without_www_unchanged():
    assert extract_domain("https://example.com/login") == "example.com"


def test_extract_domain_www_and_non_www_are_equal():
    assert extract_domain("https://www.example.com/a") == extract_domain("https://example.com/b")


def test_extract_domain_only_strips_leading_www_not_other_subdomains():
    """ต้องตัดเฉพาะ "www." นำหน้าจริงๆ เท่านั้น ห้ามตัด subdomain อื่นที่บังเอิญขึ้นต้น
    ด้วย www ปนกัน (เช่น "www2.example.com" ไม่ใช่ "www.example.com")"""
    assert extract_domain("https://www2.example.com/") == "www2.example.com"
    assert extract_domain("https://wwwexample.com/") == "wwwexample.com"


def test_extract_domain_lowercases_and_strips_port():
    assert extract_domain("https://WWW.Example.COM:8080/path") == "example.com"


def test_extract_domain_returns_empty_string_on_malformed_url():
    assert extract_domain("not a url at all ::::") == ""


# --- normalize_domain(): เวอร์ชันของ extract_domain() สำหรับ input ที่เป็น bare hostname
# อยู่แล้ว (เช่น path param ของ REST endpoint) ไม่ใช่ URL เต็มรูปแบบ ---


def test_normalize_domain_strips_www_and_lowercases():
    assert normalize_domain("WWW.Example.COM") == "example.com"


def test_normalize_domain_leaves_non_www_domain_unchanged_besides_lowercasing():
    assert normalize_domain("Example.COM") == "example.com"


def test_normalize_domain_agrees_with_extract_domain_for_the_same_host():
    assert normalize_domain("www.example.com") == extract_domain("https://www.example.com/x")


# --- ผลกระทบจริงต่อ credential lookup (storage.py) — บั๊กเดิม: login ผ่าน URL ที่มี www.
# แต่ task จริงเริ่มจาก URL ที่ไม่มี www. (หรือกลับกัน) ของเว็บเดียวกัน ทำให้หา credential
# ไม่เจอ ---


def test_credentials_saved_via_www_domain_are_found_via_non_www_domain(tmp_path, monkeypatch):
    from backend.app.config import settings

    monkeypatch.setattr(settings, "site_manuals_dir", str(tmp_path / "manuals"))

    save_domain = extract_domain("https://www.example.com/login")
    storage.save_credentials(save_domain, "alice", "s3cr3t")

    lookup_domain = extract_domain("https://example.com/dashboard")
    creds = storage.load_credentials(lookup_domain)

    assert creds == {"username": "alice", "password": "s3cr3t"}


# --- W_risky_multilingual: ปุ่มทำลายข้อมูลที่ไม่ได้เขียนเป็นภาษาอังกฤษต้องถูกยกระดับด้วย ---
# เดิม RISKY_LABEL_KEYWORDS เป็นอังกฤษล้วน ขณะที่ SAFE_ACTION_LABEL_KEYWORDS เป็นสองภาษา
# อยู่แล้ว — human-in-the-loop จึงไม่เคยทำงานกับปุ่มลบบนเว็บภาษาอื่นเลย


@pytest.mark.parametrize("label", ["ลบ", "Löschen", "削除", "删除", "Supprimer", "Eliminar", "ชำระเงิน"])
def test_destructive_label_in_any_language_needs_confirmation(label):
    assert classify_action({"type": "click", "index": 1}, label=label) == ActionRisk.NEEDS_CONFIRMATION


@pytest.mark.parametrize("label", ["ค้นหา", "Search", "ถัดไป", "ดูรายละเอียด"])
def test_safe_labels_are_still_safe_after_adding_multilingual_risky_keywords(label):
    """2 เคส false-positive เดิมที่ W_search แก้ไว้ (ปุ่มค้นหา/กรอง, การ์ดผลการค้นหา) ต้องยัง
    เป็น SAFE เหมือนเดิมทุกประการ — การเพิ่มคำเสี่ยงภาษาอื่นต้องไม่ไปกวนของเดิม"""
    assert classify_action({"type": "click", "index": 1}, label=label) == ActionRisk.SAFE


def test_english_risky_keywords_still_work():
    for label in ("Delete", "Remove", "Place Order", "Pay now"):
        assert classify_action({"type": "click", "index": 1}, label=label) == ActionRisk.NEEDS_CONFIRMATION

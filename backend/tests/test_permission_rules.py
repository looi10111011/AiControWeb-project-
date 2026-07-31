from backend.app.permission.rules import extract_domain, normalize_domain
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

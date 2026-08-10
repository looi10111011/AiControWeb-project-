import os

import pytest

from backend.app.config import settings
from backend.app.site_learning import storage
from backend.app.site_learning.schema import ButtonInfo, FormFieldInfo, PageInfo, SiteManual, UIPatternInfo


@pytest.fixture(autouse=True)
def _isolated_manuals_dir(tmp_path, monkeypatch):
    """ทุกเทสต์ในไฟล์นี้เขียน/อ่านไฟล์จริงลงดิสก์ — ใช้ tmp_path ของ pytest กัน
    settings.site_manuals_dir ตัวจริงบนเครื่อง dev โดนเขียนทับ/ปนกับข้อมูลเทสต์"""
    monkeypatch.setattr(settings, "site_manuals_dir", str(tmp_path))
    yield


def _sample_manual(website: str = "example.com") -> SiteManual:
    return SiteManual(website=website, pages=[
        PageInfo(
            name="Dashboard", url="/dashboard", description="home page", menu_path=["Dashboard"],
            buttons=[ButtonInfo(text="Export", selector="button.export")],
        ),
    ])


def test_manual_exists_and_load_manual_return_falsy_when_never_saved():
    assert storage.manual_exists("example.com") is False
    assert storage.load_manual("example.com") is None


def test_save_manual_creates_version_1_and_all_derived_files():
    version = storage.save_manual(_sample_manual())

    assert version == 1
    assert storage.manual_exists("example.com") is True
    domain_dir = os.path.join(settings.site_manuals_dir, "example.com")
    # ไม่มีไฟล์ประวัติแยกต่อเวอร์ชัน (vN.json) อีกต่อไป — save_manual() ทับ latest.json
    # ตรงๆ ทุกครั้งตามที่ user ขอ (กันไฟล์สะสมไม่รู้จบบนดิสก์ที่ commit เข้า git)
    assert set(os.listdir(domain_dir)) == {
        "latest.json", "ui-map.json", "selectors.json", "knowledge.json",
    }


def test_load_manual_round_trips_pages():
    storage.save_manual(_sample_manual())
    loaded = storage.load_manual("example.com")

    assert loaded is not None
    assert loaded.website == "example.com"
    assert loaded.version == 1
    assert loaded.pages[0].name == "Dashboard"
    assert loaded.pages[0].buttons[0].text == "Export"


def test_save_manual_bumps_version_on_subsequent_saves():
    storage.save_manual(_sample_manual())
    v2 = storage.save_manual(_sample_manual())

    assert v2 == 2
    loaded = storage.load_manual("example.com")
    assert loaded.version == 2
    # version number ยังนับเพิ่มไว้เป็น metadata ปกติ แต่ไม่มีไฟล์ vN.json แยกต่างหากอีก
    # ต่อไป (latest.json ถูกทับตรงๆ) — ดู docstring save_manual()
    domain_dir = os.path.join(settings.site_manuals_dir, "example.com")
    assert set(os.listdir(domain_dir)) == {
        "latest.json", "ui-map.json", "selectors.json", "knowledge.json",
    }


def test_load_knowledge_text_summarizes_pages_with_descriptions():
    storage.save_manual(_sample_manual())
    text = storage.load_knowledge_text("example.com")
    assert "Dashboard" in text
    assert "home page" in text


def test_load_knowledge_text_empty_when_no_manual():
    assert storage.load_knowledge_text("does-not-exist.com") == ""


def test_update_single_page_replaces_matching_page_and_bumps_version():
    storage.save_manual(_sample_manual())

    updated_page = PageInfo(
        name="Dashboard", url="/dashboard", description="home page (updated)", menu_path=["Dashboard"],
        buttons=[ButtonInfo(text="Export CSV", selector="button.export-csv")],
    )
    version = storage.update_single_page("example.com", updated_page)

    assert version == 2
    loaded = storage.load_manual("example.com")
    assert len(loaded.pages) == 1  # แทนที่ ไม่ใช่เพิ่ม (จับคู่ด้วย url)
    assert loaded.pages[0].description == "home page (updated)"
    assert loaded.pages[0].buttons[0].text == "Export CSV"


def test_update_single_page_appends_when_url_not_found():
    storage.save_manual(_sample_manual())

    new_page = PageInfo(name="Settings", url="/settings", description="settings page")
    storage.update_single_page("example.com", new_page)

    loaded = storage.load_manual("example.com")
    assert len(loaded.pages) == 2
    assert {p.name for p in loaded.pages} == {"Dashboard", "Settings"}


def test_update_single_page_returns_none_when_no_manual_exists_yet():
    result = storage.update_single_page("never-learned.com", PageInfo(name="X", url="/x"))
    assert result is None


def test_load_manual_returns_none_on_corrupt_json(tmp_path):
    domain_dir = os.path.join(settings.site_manuals_dir, "broken.com")
    os.makedirs(domain_dir, exist_ok=True)
    with open(os.path.join(domain_dir, "latest.json"), "w", encoding="utf-8") as f:
        f.write("{not valid json")

    assert storage.load_manual("broken.com") is None


# ---------------- W17: เก็บ username/password แยกไฟล์จาก manual ----------------


def test_credentials_exist_and_load_return_falsy_when_never_saved():
    assert storage.credentials_exist("example.com") is False
    assert storage.load_credentials("example.com") is None


def test_save_and_load_credentials_round_trips():
    storage.save_credentials("example.com", "alice", "s3cr3t")

    assert storage.credentials_exist("example.com") is True
    creds = storage.load_credentials("example.com")
    assert creds == {"username": "alice", "password": "s3cr3t"}


# --- Security 1.4: credentials.json เข้ารหัสบนดิสก์ (Fernet) + migration จากไฟล์เก่า ---


def test_saved_credentials_file_is_encrypted_on_disk():
    """ไฟล์ credentials.json บนดิสก์จริงต้องไม่มี username/password เป็น plaintext เลย —
    เข้ารหัสก่อนเขียนเสมอ (ดู storage.py::save_credentials)"""
    import json as _json

    storage.save_credentials("example.com", "alice", "s3cr3t-password")

    raw = open(os.path.join(settings.site_manuals_dir, "example.com", "credentials.json"), encoding="utf-8").read()
    assert "s3cr3t-password" not in raw
    assert "alice" not in raw
    data = _json.loads(raw)
    assert data["encrypted"] is True


def test_load_credentials_migrates_legacy_plaintext_file_transparently(tmp_path):
    """ไฟล์เก่าที่ยังเป็น plaintext (จาก storage.py เวอร์ชันก่อนหน้า Security 1.4 — ไม่มี
    marker "encrypted" เลย) ต้องยังอ่านได้ปกติ (backward-compat) แล้วถูก re-save เป็นแบบ
    เข้ารหัสทันทีเงียบๆ โดยไม่ต้องมี migration script แยก"""
    import json as _json

    domain_dir = os.path.join(settings.site_manuals_dir, "legacy.com")
    os.makedirs(domain_dir, exist_ok=True)
    with open(os.path.join(domain_dir, "credentials.json"), "w", encoding="utf-8") as f:
        _json.dump({"username": "legacy-user", "password": "legacy-pass"}, f)

    creds = storage.load_credentials("legacy.com")
    assert creds == {"username": "legacy-user", "password": "legacy-pass"}

    # ไฟล์บนดิสก์ต้องถูกเขียนทับเป็นแบบเข้ารหัสแล้วหลัง load ครั้งแรก
    raw = open(os.path.join(domain_dir, "credentials.json"), encoding="utf-8").read()
    assert "legacy-pass" not in raw
    migrated = _json.loads(raw)
    assert migrated["encrypted"] is True

    # โหลดซ้ำรอบสอง (จากไฟล์ที่ migrate แล้ว) ต้องยังได้ค่าเดิมกลับมาถูกต้อง
    assert storage.load_credentials("legacy.com") == {"username": "legacy-user", "password": "legacy-pass"}


def test_load_credentials_returns_none_for_tampered_ciphertext():
    """ไฟล์ที่มี marker encrypted แต่ ciphertext ถูกแก้ไข/เสียหาย (เช่น key เปลี่ยน/ไฟล์เสีย)
    ต้องคืน None แทนที่จะ throw ออกไปทำให้ caller พัง"""
    import json as _json

    domain_dir = os.path.join(settings.site_manuals_dir, "tampered.com")
    os.makedirs(domain_dir, exist_ok=True)
    with open(os.path.join(domain_dir, "credentials.json"), "w", encoding="utf-8") as f:
        _json.dump({"encrypted": True, "username": "not-a-valid-fernet-token", "password": "also-not-valid"}, f)

    assert storage.load_credentials("tampered.com") is None


def test_save_credentials_writes_a_separate_file_from_the_manual():
    """credentials.json ต้องไม่ปนกับ latest.json ของ manual — เก็บคนละไฟล์เจตนา กัน
    credential หลุดปนเข้าไปใน manual โดยไม่ตั้งใจ"""
    storage.save_manual(_sample_manual())
    storage.save_credentials("example.com", "alice", "s3cr3t")

    domain_dir = os.path.join(settings.site_manuals_dir, "example.com")
    assert "credentials.json" in os.listdir(domain_dir)
    manual_dump = open(os.path.join(domain_dir, "latest.json"), encoding="utf-8").read()
    assert "s3cr3t" not in manual_dump
    assert "alice" not in manual_dump


def test_save_credentials_overwrites_in_place_no_versioning():
    storage.save_credentials("example.com", "alice", "old-pass")
    storage.save_credentials("example.com", "alice", "new-pass")

    creds = storage.load_credentials("example.com")
    assert creds["password"] == "new-pass"
    domain_dir = os.path.join(settings.site_manuals_dir, "example.com")
    assert os.listdir(domain_dir) == ["credentials.json"]  # ไม่มีไฟล์ประวัติเวอร์ชันเลย


def test_delete_credentials_removes_the_file_and_is_idempotent():
    storage.save_credentials("example.com", "alice", "s3cr3t")

    assert storage.delete_credentials("example.com") is True
    assert storage.credentials_exist("example.com") is False
    assert storage.delete_credentials("example.com") is False  # ลบซ้ำไม่ error


def test_credentials_are_isolated_per_domain():
    """W23: credential ของเว็บหนึ่งต้องไม่มีทางไปโผล่/ถูกดึงมาใช้กับอีกเว็บหนึ่งได้เลย —
    แต่ละโดเมนเก็บใน _domain_dir(domain)/credentials.json แยกโฟลเดอร์กันเด็ดขาดอยู่แล้ว
    โดยโครงสร้าง (ดู storage.py หัวไฟล์) เทสต์นี้ยืนยันตรงๆ ว่า save/load จริงไม่รั่วข้ามกัน"""
    storage.save_credentials("site-a.com", "alice", "alice-pass")
    storage.save_credentials("site-b.com", "bob", "bob-pass")

    assert storage.load_credentials("site-a.com") == {"username": "alice", "password": "alice-pass"}
    assert storage.load_credentials("site-b.com") == {"username": "bob", "password": "bob-pass"}

    # ลบของเว็บหนึ่งต้องไม่กระทบอีกเว็บหนึ่งเลย
    assert storage.delete_credentials("site-a.com") is True
    assert storage.credentials_exist("site-a.com") is False
    assert storage.load_credentials("site-b.com") == {"username": "bob", "password": "bob-pass"}

    domain_a_dir = os.path.join(settings.site_manuals_dir, "site-a.com")
    domain_b_dir = os.path.join(settings.site_manuals_dir, "site-b.com")
    assert not os.path.exists(os.path.join(domain_a_dir, "credentials.json"))
    assert os.path.exists(os.path.join(domain_b_dir, "credentials.json"))


def test_save_and_load_manual_round_trips_icon_hint_and_ui_patterns():
    manual = SiteManual(website="example.com", pages=[
        PageInfo(
            name="Products", url="/products",
            buttons=[ButtonInfo(text="", icon_hint="shopping cart", selector="#cart-btn")],
            ui_patterns=[
                UIPatternInfo(
                    name="Product Card", ui_type="Card", components=["Image", "Title", "Price"],
                    buttons=[ButtonInfo(text="Add to Cart", selector="button.add-to-cart")],
                    selector="div.product-card", item_count=42,
                ),
            ],
        ),
    ])
    storage.save_manual(manual)
    loaded = storage.load_manual("example.com")

    assert loaded.pages[0].buttons[0].icon_hint == "shopping cart"
    assert len(loaded.pages[0].ui_patterns) == 1
    pattern = loaded.pages[0].ui_patterns[0]
    assert pattern.name == "Product Card"
    assert pattern.ui_type == "Card"
    assert pattern.components == ["Image", "Title", "Price"]
    assert pattern.selector == "div.product-card"
    assert pattern.item_count == 42
    assert pattern.buttons[0].text == "Add to Cart"


def test_build_selectors_includes_ui_pattern_selectors():
    manual = SiteManual(website="example.com", pages=[
        PageInfo(
            name="Products", url="/products",
            ui_patterns=[
                UIPatternInfo(
                    name="Product Card", ui_type="Card",
                    buttons=[ButtonInfo(text="Add to Cart", selector="button.add-to-cart")],
                    selector="div.product-card", item_count=42,
                ),
            ],
        ),
    ])
    storage.save_manual(manual)
    selectors_path = os.path.join(settings.site_manuals_dir, "example.com", "selectors.json")
    import json
    selectors = json.loads(open(selectors_path, encoding="utf-8").read())

    assert selectors["Products > [Product Card]"]["css"] == "div.product-card"
    assert selectors["Products > [Product Card] Add to Cart"]["css"] == "button.add-to-cart"


def test_load_credentials_returns_none_on_corrupt_json():
    domain_dir = os.path.join(settings.site_manuals_dir, "broken.com")
    os.makedirs(domain_dir, exist_ok=True)
    with open(os.path.join(domain_dir, "credentials.json"), "w", encoding="utf-8") as f:
        f.write("{not valid json")

    assert storage.load_credentials("broken.com") is None


def test_save_and_load_manual_round_trips_errors_list():
    """W24: SiteManual.errors (goto/click/login ที่ล้มเหลวระหว่าง crawl) ต้อง persist ผ่าน
    save_manual()/load_manual() ได้ครบ ไม่หายไปตอน round-trip ผ่าน JSON บนดิสก์"""
    manual = SiteManual(
        website="example.com",
        pages=[PageInfo(name="Home", url="/")],
        errors=[{"url": "/broken", "phase": "goto", "error": "TimeoutError: boom"}],
    )
    storage.save_manual(manual)

    loaded = storage.load_manual("example.com")
    assert loaded.errors == [{"url": "/broken", "phase": "goto", "error": "TimeoutError: boom"}]


def test_save_and_load_manual_round_trips_summary():
    """W26: SiteManual.summary (ภาพรวม "เว็บไซต์นี้ทำอะไรได้บ้าง" จาก describe_site()) ต้อง
    persist ผ่าน save_manual()/load_manual() ได้ครบ ไม่หายไปตอน round-trip ผ่าน JSON บนดิสก์"""
    manual = SiteManual(
        website="example.com",
        pages=[PageInfo(name="Home", url="/")],
        summary="เว็บไซต์นี้ใช้ดูสินค้าและสั่งซื้อออนไลน์ได้",
    )
    storage.save_manual(manual)

    loaded = storage.load_manual("example.com")
    assert loaded.summary == "เว็บไซต์นี้ใช้ดูสินค้าและสั่งซื้อออนไลน์ได้"


def test_load_manual_defaults_summary_to_empty_string_for_old_manuals_without_it():
    domain_dir = os.path.join(settings.site_manuals_dir, "legacy2.com")
    os.makedirs(domain_dir, exist_ok=True)
    legacy_data = {
        "website": "legacy2.com", "version": 1, "generated_at": 0.0,
        "pages": [{"name": "Home", "url": "/"}],
    }
    import json
    with open(os.path.join(domain_dir, "latest.json"), "w", encoding="utf-8") as f:
        json.dump(legacy_data, f)

    loaded = storage.load_manual("legacy2.com")
    assert loaded.summary == ""


def test_load_manual_defaults_errors_to_empty_list_for_old_manuals_without_it():
    """manual ที่ save ไว้ก่อนมี W24 (ไม่มี key "errors" เลยใน JSON) ต้องโหลดได้ปกติ ไม่
    throw — errors default เป็น [] เงียบๆ"""
    domain_dir = os.path.join(settings.site_manuals_dir, "legacy.com")
    os.makedirs(domain_dir, exist_ok=True)
    legacy_data = {
        "website": "legacy.com", "version": 1, "generated_at": 0.0,
        "pages": [{"name": "Home", "url": "/"}],
    }
    import json
    with open(os.path.join(domain_dir, "latest.json"), "w", encoding="utf-8") as f:
        json.dump(legacy_data, f)

    loaded = storage.load_manual("legacy.com")
    assert loaded.errors == []


# --- W65[1] ("Required-Field Validation"): build_strict_manual_context() เดิมมีแค่ block
# page.buttons — page.forms (มี FormFieldInfo.required เก็บไว้ตั้งแต่ crawl) ไม่เคยถูกอ่าน
# เลยจนถึงตอนนี้ (dead data) — เพิ่ม block ใหม่ให้ planner เห็นฟิลด์ที่ required ล่วงหน้า


def test_build_strict_manual_context_lists_required_and_optional_form_fields():
    page = PageInfo(
        name="Change Password", url="/changePasswordSave",
        forms=[
            FormFieldInfo(label="Current Password", required=True),
            FormFieldInfo(label="New Password", required=True),
            FormFieldInfo(label="Middle Name", required=False),
        ],
    )

    context = storage.build_strict_manual_context(page)

    assert "Recorded form fields on this page (label — required?):" in context
    assert "- Current Password *จำเป็น" in context
    assert "- New Password *จำเป็น" in context
    assert "- Middle Name" in context
    assert "Middle Name *จำเป็น" not in context


def test_build_strict_manual_context_falls_back_to_field_name_or_placeholder():
    page = PageInfo(
        name="Signup", url="/signup",
        forms=[
            FormFieldInfo(field_name="email", required=True),
            FormFieldInfo(placeholder="Zip Code", required=False),
        ],
    )

    context = storage.build_strict_manual_context(page)

    assert "- email *จำเป็น" in context
    assert "- Zip Code" in context


def test_build_strict_manual_context_omits_forms_block_when_no_forms_recorded():
    page = PageInfo(name="Dashboard", url="/dashboard")

    context = storage.build_strict_manual_context(page)

    assert "Recorded form fields" not in context


# --- W67[D]: find_matching_page(manual, goal, min_score) — min_score default=1 คือ
# พฤติกรรมเดิมทุกประการ (caller เดิม 3 จุดใน routes.py ไม่ส่ง min_score เลย) เพิ่มเข้ามาให้
# caller ที่ต้องการความมั่นใจสูงกว่า (nav-fastpath auto-decide) กรอง match ที่ไม่มั่นใจออกได้


def _manual_with_admin_page() -> SiteManual:
    return SiteManual(website="example.com", pages=[
        PageInfo(name="Dashboard", url="/dashboard", description="home page"),
        PageInfo(
            name="User Management", url="/admin/users",
            description="Manage system users", breadcrumb=["Home", "Admin", "User Management"],
        ),
    ])


def test_find_matching_page_default_min_score_matches_loosely_like_before():
    manual = _manual_with_admin_page()

    page = storage.find_matching_page(manual, "users")

    assert page is not None
    assert page.name == "User Management"


def test_find_matching_page_returns_none_when_score_below_min_score():
    manual = _manual_with_admin_page()

    # "users" เจอแค่ 1 token match (score=1) — min_score=2 ต้องกรองออก แม้ default (1) จะผ่าน
    page = storage.find_matching_page(manual, "users", min_score=2)

    assert page is None


def test_find_matching_page_min_score_still_returns_page_when_score_meets_threshold():
    manual = _manual_with_admin_page()

    page = storage.find_matching_page(manual, "admin users management", min_score=2)

    assert page is not None
    assert page.name == "User Management"

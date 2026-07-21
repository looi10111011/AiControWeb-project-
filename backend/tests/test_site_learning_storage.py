import os

import pytest

from backend.app.config import settings
from backend.app.site_learning import storage
from backend.app.site_learning.schema import ButtonInfo, PageInfo, SiteManual, UIPatternInfo


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
    assert set(os.listdir(domain_dir)) == {
        "latest.json", "v1.json", "ui-map.json", "selectors.json", "knowledge.json",
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
    # v1.json (ประวัติ) ต้องยังอยู่ ไม่เคยลบทิ้ง
    domain_dir = os.path.join(settings.site_manuals_dir, "example.com")
    assert "v1.json" in os.listdir(domain_dir)
    assert "v2.json" in os.listdir(domain_dir)


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


def test_save_credentials_writes_a_separate_file_from_the_manual():
    """credentials.json ต้องไม่ปนกับ latest.json/vN.json ของ manual — เก็บคนละไฟล์
    เพราะ manual มีระบบ versioning (ไม่เคยลบ vN.json เก่า) ถ้าฝัง credential ปนไปด้วยจะมี
    สำเนารหัสผ่านกระจายอยู่หลายไฟล์บนดิสก์ตลอดกาล"""
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

from backend.app.site_learning.safety import is_crawl_safe, is_safe_nav_link

# is_crawl_safe(): default-deny (allowlist-first) — ตรงกับสเปค Safety Rules ตรงๆ
# ("หากไม่แน่ใจ ห้ามกด") ใช้ตัดสินใจว่าจะกดปุ่ม/element นี้ระหว่าง crawl ไหม


def test_is_crawl_safe_allows_view_label():
    assert is_crawl_safe("View Details") is True


def test_is_crawl_safe_allows_next_previous_pagination():
    assert is_crawl_safe("Next") is True
    assert is_crawl_safe("Previous") is True
    assert is_crawl_safe("Pagination") is True


def test_is_crawl_safe_blocks_delete_save_submit():
    assert is_crawl_safe("Delete") is False
    assert is_crawl_safe("Save") is False
    assert is_crawl_safe("Submit") is False


def test_is_crawl_safe_blocked_wins_over_allowed_when_both_match():
    # "View & Delete" มีทั้งคำ allow (view) และคำ block (delete) พร้อมกัน — BLOCKED ต้องชนะ
    assert is_crawl_safe("View & Delete") is False


def test_is_crawl_safe_ambiguous_label_defaults_to_unsafe():
    # ไม่ตรงคำไหนใน allowlist/blocklist เลย ("ไม่แน่ใจ ห้ามกด")
    assert is_crawl_safe("Random Button") is False
    assert is_crawl_safe("") is False


def test_is_crawl_safe_case_insensitive():
    assert is_crawl_safe("DELETE") is False
    assert is_crawl_safe("view details") is True


def test_is_crawl_safe_scroll_and_wait_are_inherently_safe_regardless_of_label():
    assert is_crawl_safe("", "scroll") is True
    assert is_crawl_safe("anything at all", "wait") is True


def test_is_crawl_safe_goto_type_still_checks_label():
    # goto ไม่ได้อยู่ใน _INHERENTLY_SAFE_TYPES ตั้งใจ — ลิงก์ label "Logout" ที่ navigate
    # ไปด้วย goto ก็ยังต้องถูกบล็อกเหมือน click ปกติ
    assert is_crawl_safe("Logout", "goto") is False
    assert is_crawl_safe("Next", "goto") is True


def test_is_crawl_safe_cmd_type_itself_can_be_blocked_keyword():
    assert is_crawl_safe("", "submit") is False


def test_is_crawl_safe_allows_back_and_forward():
    # W18: ความหมายเดียวกับ next/previous ที่มีอยู่แล้ว — icon ปุ่มย้อนกลับ/ไปต่อ
    assert is_crawl_safe("Back") is True
    assert is_crawl_safe("Go forward") is True


# is_safe_nav_link(): default-allow (blocklist-only) — สำหรับตัดสินใจว่าจะเดินตามลิงก์
# nav/menu ระหว่าง BFS ไหม (คนละคำถามจาก is_crawl_safe())


def test_is_safe_nav_link_allows_generic_menu_labels():
    # ชื่อ feature ทั่วไปที่ไม่ได้อยู่ใน ALLOWED_CRAWL_KEYWORDS เลย แต่ต้องปลอดภัยที่จะ
    # เดินตามอยู่ดี (เมนูปกติของเว็บ ไม่ใช่คำ action)
    assert is_safe_nav_link("Dashboard") is True
    assert is_safe_nav_link("Products") is True
    assert is_safe_nav_link("Orders") is True


def test_is_safe_nav_link_blocks_logout_and_destructive_labels():
    assert is_safe_nav_link("Logout") is False
    assert is_safe_nav_link("Delete Account") is False


def test_is_safe_nav_link_empty_text_is_safe():
    assert is_safe_nav_link("") is True

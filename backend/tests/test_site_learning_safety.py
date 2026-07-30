from backend.app.site_learning.safety import (
    is_crawl_safe,
    is_hashtag_label,
    is_hashtag_url,
    is_safe_nav_link,
    is_video_content_label,
    is_video_content_url,
)

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


# is_video_content_url()/is_video_content_label() (W37): กันไม่ให้ crawler navigate เข้า
# หน้า "ดูวีดีโอ" เลย (YouTube/Facebook/Instagram ฯลฯ) เพราะคลิปแนะนำคลิปถัดไปไม่รู้จบ


def test_is_video_content_url_matches_youtube_watch_and_shorts():
    assert is_video_content_url("https://www.youtube.com/watch?v=abc123") is True
    assert is_video_content_url("https://www.youtube.com/shorts/abc123") is True


def test_is_video_content_url_matches_facebook_and_instagram_reels():
    assert is_video_content_url("https://www.facebook.com/reel/123456") is True
    assert is_video_content_url("https://www.instagram.com/reel/abc123/") is True


def test_is_video_content_url_matches_generic_video_path_regardless_of_domain():
    # ไม่ผูกกับโดเมนใดโดเมนหนึ่ง — ใช้ได้กับเว็บวีดีโออื่นๆ ที่ใช้ path convention คล้ายกัน
    assert is_video_content_url("https://example.com/videos/42") is True
    assert is_video_content_url("https://example.com/embed/42") is True


def test_is_video_content_url_does_not_match_unrelated_paths():
    assert is_video_content_url("https://example.com/products/123") is False
    assert is_video_content_url("https://example.com/dashboard") is False


def test_is_video_content_label_matches_watch_and_shorts_keywords():
    assert is_video_content_label("Watch Now") is True
    assert is_video_content_label("Reels") is True
    assert is_video_content_label("Play Video") is True


def test_is_video_content_label_does_not_match_unrelated_labels():
    assert is_video_content_label("View Details") is False
    assert is_video_content_label("") is False


# is_hashtag_url()/is_hashtag_label() (W38): กันไม่ให้ crawler navigate เข้าหน้ารวมโพสต์ของ
# แฮชแท็ก เพราะแนะนำแฮชแท็ก/โพสต์อื่นไม่รู้จบ (ปัญหาเดียวกับวีดีโอ แค่คนละประเภทเนื้อหา)


def test_is_hashtag_label_matches_hash_prefixed_text():
    assert is_hashtag_label("#travel") is True
    assert is_hashtag_label("#Cat_Lovers") is True
    assert is_hashtag_label("  #trending123") is True  # เว้นวรรคนำหน้าไม่ควรมีผล


def test_is_hashtag_label_does_not_match_bare_or_spaced_hash():
    # "#" เดี่ยวๆ หรือ "# " (เว้นวรรค) ไม่ใช่แฮชแท็กจริง อาจเป็นแค่สัญลักษณ์ตกแต่ง
    assert is_hashtag_label("#") is False
    assert is_hashtag_label("# ") is False


def test_is_hashtag_label_does_not_match_unrelated_labels():
    assert is_hashtag_label("View Details") is False
    assert is_hashtag_label("") is False


def test_is_hashtag_url_matches_hashtag_path_segment():
    assert is_hashtag_url("https://twitter.com/hashtag/travel") is True
    assert is_hashtag_url("https://example.com/hashtags/trending") is True


def test_is_hashtag_url_does_not_match_generic_blog_tags_path():
    # "/tags/xxx" เฉยๆ (ไม่มีคำว่า "hashtag" ตรงๆ) มักเป็นหมวดหมู่บทความปกติของ blog/เอกสาร
    # ไม่ใช่แฮชแท็กแบบโซเชียล — ตั้งใจไม่กันเพื่อไม่ให้ตัดโครงสร้างที่ควรเรียนรู้จริงทิ้ง
    assert is_hashtag_url("https://example.com/blog/tags/python") is False
    assert is_hashtag_url("https://example.com/products/123") is False

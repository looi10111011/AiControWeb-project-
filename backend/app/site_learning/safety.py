"""site_learning/safety.py — W14: กัน crawler กด action ที่เปลี่ยนข้อมูลระหว่างเรียนรู้เว็บ.

ต่างจาก permission/rules.py (default-allow + human-in-the-loop): crawl ไม่มีคนดู จึง default-deny ("หากไม่แน่ใจ ห้ามกด").
W36: classify_button_tier()/button_core_priority() ตอบ "ควรไล่กดก่อนไหม" (ลดจำนวนปุ่มต่อหน้า) — ไม่ใช่ safety gate;
     ปุ่ม core ยังต้องผ่าน is_crawl_safe() เดิมทุกประการ
W37: is_video_content_url/label — วีดีโอแนะนำคลิปต่อไม่รู้จบ พา crawler ไม่กลับมาสำรวจส่วนอื่น (user ขอแค่โครงสร้าง);
     กันไม่ให้เข้าเลยแม้ครั้งเดียว ใช้ 3 จุดใน crawler.py (BFS nav_links, tier, post-click URL check)
W38: is_hashtag_url/label — ลูปกดแฮชแท็กแบบเดียวกับวีดีโอ; label "#..." เชื่อถือได้สุด, URL segment เป็น fallback
"""

import re
import urllib.parse

from backend.app.site_learning.schema import ButtonInfo

# คำที่บ่งบอกว่า action นี้ "แค่เดินสำรวจ/ดูข้อมูล" ปลอดภัยให้ crawler กดได้
# W18: back/forward — จาก icon_hint/aria-label ของปุ่มย้อนกลับ/ไปต่อ
ALLOWED_CRAWL_KEYWORDS = {
    "menu", "view", "detail", "next", "previous", "expand", "collapse",
    "filter", "search", "pagination", "page", "tabs", "tab", "open", "close",
    "back", "forward",
    # W20: "Continue" (wizard/checkout) = "next"; "Continue to Payment" ยังโดน BLOCKED (BLOCKED ชนะ ALLOWED เสมอ)
    "continue",
    # W19: ไอคอนตะกร้า = แค่ไปดูตะกร้า (เดิมโดน default-deny manual เลยไม่มีหน้า cart/checkout);
    # "add to cart"/"add to bag" กันด้วย "add to " ใน BLOCKED
    "cart", "bag", "basket",
    # W20: "Checkout" แค่ไปหน้ากรอกที่อยู่ ไม่ใช่ commit (Place Order/Confirm/Pay/Submit ยัง BLOCKED);
    # เดิมอยู่ใน BLOCKED ทำให้ไปได้แค่หน้า cart
    "checkout",
}

# action ที่อาจเปลี่ยน/ทำลายข้อมูล — ห้ามกดเด็ดขาด แม้ label ตรง ALLOWED ด้วย (เช่น "View & Delete")
BLOCKED_CRAWL_KEYWORDS = {
    "delete", "remove", "save", "submit", "confirm", "purchase",
    "payment", "reset", "logout", "approve", "reject", "execute", "send", "sync",
    # W19: "add to " (เว้นวรรคท้ายตั้งใจ กันชน "Address"/"Additional") — add to cart/bag/wishlist เปลี่ยนสถานะจริง
    "add to ",
}

# ปลอดภัยโดยธรรมชาติไม่ต้องดู label. ไม่รวม "goto" โดยตั้งใจ — ลิงก์ "Logout" แบบ GET-triggers-action ยังมี effect
_INHERENTLY_SAFE_TYPES = {"scroll", "wait"}


def is_crawl_safe(label: str, cmd_type: str = "") -> bool:
    """ "จะกดปุ่มนี้ระหว่าง crawl ไหม" — default-deny: True เฉพาะเมื่อ label/cmd_type ตรง ALLOWED_CRAWL_KEYWORDS
    (หรือ type scroll/wait) และไม่ตรง BLOCKED_CRAWL_KEYWORDS; label ว่าง/ไม่ตรงอะไร = False; BLOCKED ชนะเสมอ.
    คนละคำถามกับ is_safe_nav_link() (default-allow สำหรับเมนูชื่อเฉพาะเว็บอย่าง "Dashboard")"""
    lower_label = (label or "").lower()
    lower_type = (cmd_type or "").lower()

    if lower_type in _INHERENTLY_SAFE_TYPES:
        return True

    if lower_type in BLOCKED_CRAWL_KEYWORDS or any(word in lower_label for word in BLOCKED_CRAWL_KEYWORDS):
        return False
    if lower_type in ALLOWED_CRAWL_KEYWORDS or any(word in lower_label for word in ALLOWED_CRAWL_KEYWORDS):
        return True
    return False


def is_safe_nav_link(text: str) -> bool:
    """ "เดินตาม (goto) ลิงก์ nav นี้ไหม" ตอน BFS — default-allow, บล็อกเฉพาะที่ตรง BLOCKED_CRAWL_KEYWORDS
    (เช่น <a href="/logout">)"""
    lower = (text or "").lower()
    return not any(word in lower for word in BLOCKED_CRAWL_KEYWORDS)


# W36: ฟังก์ชันหลักของหน้า — ใช้จัด tier/priority เท่านั้น ไม่ใช่ safety gate; มี submit/save/login ที่ยังโดน
# BLOCKED โดยตั้งใจ ("core" = ถ้ากดได้ควรกดก่อน)
CORE_ACTION_KEYWORDS = {
    "search", "filter", "sort", "add to cart", "add to bag", "checkout",
    "save", "apply", "submit", "login", "log in", "sign in",
}

# W36: ปุ่มรอง/ตกแต่ง — ไม่คุ้มเวลาสำรวจ ข้ามตั้งแต่ต้นใน _explore_buttons() (ไม่ใช่เพราะอันตราย)
DECORATIVE_KEYWORDS = {
    "share", "like", "unlike", "favorite", "favourite", "wishlist",
    "upvote", "downvote", "notification", "bell", "theme", "dark mode",
    "light mode", "language", "locale", "avatar", "profile menu",
    "account menu", "breadcrumb",
}

# W36: เลขหน้า > 1 ("2", "Page 4") = decorative — ไล่กดไม่รู้จบโดยไม่ได้โครงสร้างใหม่ ("Next" ยังอนุญาต)
_PAGE_NUMBER_PATTERN = re.compile(r"^(?:page\s*)?(\d+)$", re.IGNORECASE)

# W37: path segment ของหน้าดูวีดีโอ (ไม่ผูกโดเมน). ต้องตรงทั้ง segment ไม่ใช่ substring — substring ชน
# fixture "/shorts1.html" ของ W28 จริง และเสี่ยง "/shorts-collection" ในเว็บขายเสื้อผ้า
VIDEO_CONTENT_URL_PATH_SEGMENTS = {
    "watch", "shorts", "reel", "reels", "video", "videos", "embed", "clip", "clips",
}

# W37: label ที่บอกว่ากดแล้วเล่นวีดีโอ — สำหรับปุ่มที่ไม่มี href ให้เช็ค URL ล่วงหน้า (substring match เหมือน set อื่น)
VIDEO_CONTENT_LABEL_KEYWORDS = {"watch", "play video", "play now", "shorts", "reel", "reels"}


# W38: ไม่รวม "tag"/"tags" โดยตั้งใจ — blog/e-commerce ใช้ "/tags/xxx" เป็นหมวดหมู่ปกติที่ควรเรียนรู้
HASHTAG_URL_PATH_SEGMENTS = {"hashtag", "hashtags"}

# W38: "#" ตามด้วย word char (เช่น "#travel") — แทบทุกเว็บโซเชียล render แบบนี้; "#" เดี่ยว/"# " ไม่นับ
_HASHTAG_LABEL_PATTERN = re.compile(r"^#\w")


def _url_path_segments(url: str) -> list[str]:
    """W37/W38: path segment ตัวพิมพ์เล็ก (ไม่รวม segment ว่าง) สำหรับเทียบแบบทั้งชิ้น"""
    path = urllib.parse.urlparse(url).path.lower()
    return [seg for seg in path.split("/") if seg]


def is_video_content_url(url: str) -> bool:
    """W37: "/watch", "/shorts/abc123", "/reel/42" ตรง; "/shorts-collection", "/watchlist" ไม่ตรง"""
    return any(seg in VIDEO_CONTENT_URL_PATH_SEGMENTS for seg in _url_path_segments(url))


def is_video_content_label(label: str) -> bool:
    """W37: label บอกตรงๆ ว่ากดแล้วเล่นวีดีโอ (ใช้ใน classify_button_tier() และ crawler.py::_record_page())"""
    lower = (label or "").lower()
    return any(word in lower for word in VIDEO_CONTENT_LABEL_KEYWORDS)


def is_hashtag_url(url: str) -> bool:
    """W38: "/hashtag/travel" ตรง; "/tags/travel" ไม่ตรง — fallback รองจาก is_hashtag_label()"""
    return any(seg in HASHTAG_URL_PATH_SEGMENTS for seg in _url_path_segments(url))


def is_hashtag_label(label: str) -> bool:
    """W38: สัญญาณหลักกันแฮชแท็ก (ไม่พึ่ง URL convention เฉพาะเว็บ)"""
    return bool(_HASHTAG_LABEL_PATTERN.match((label or "").strip()))


def _button_label_text(button_info: ButtonInfo) -> str:
    """รวม text/aria_label/title/icon_hint เป็นสตริงพิมพ์เล็กเดียว (keyword อาจอยู่ฟิลด์ไหนก็ได้ เช่น icon-only)
    — ต่างจาก crawler.py::_button_label() ที่คืนแค่ฟิลด์แรกที่ไม่ว่าง"""
    return " ".join(
        part for part in (button_info.text, button_info.aria_label, button_info.title, button_info.icon_hint)
        if part
    ).strip().lower()


def classify_button_tier(button_info: ButtonInfo) -> str:
    """W36/W37/W38: "nav" / "core" / "decorative" — heuristic ล้วน ไม่เรียก LLM (crawler deterministic, W14/W24).

    ลำดับ: (0) วีดีโอ/แฮชแท็ก -> decorative ก่อนเช็ค nav (ชนะแม้เป็น tab "Watch"/"Reels" — user ไม่ต้องการเลย)
    (1) is_nav_menu_item -> nav (แม้ label มีคำ decorative เช่น "Share Settings")
    (2) เลขหน้า > 1 หรือ DECORATIVE_KEYWORDS -> decorative
    (3) อื่นๆ -> core เสมอ (form-submit/CORE keyword/role=search ก็ core) — สำคัญ: ปุ่ม "View"/"Expand" ที่ W16
    พึ่งต้องไม่หาย; is_crawl_safe() ยังเป็นตัวตัดสินสุดท้ายว่ากดได้ไหม"""
    label = _button_label_text(button_info)
    if is_video_content_label(label) or is_hashtag_label(label):
        return "decorative"

    if button_info.is_nav_menu_item:
        return "nav"

    page_number_match = _PAGE_NUMBER_PATTERN.match((button_info.text or "").strip())
    if page_number_match and int(page_number_match.group(1)) > 1:
        return "decorative"
    if any(word in label for word in DECORATIVE_KEYWORDS):
        return "decorative"

    return "core"


def button_core_priority(button_info: ButtonInfo) -> int:
    """W36: ลำดับตอนตัด top-K ปุ่ม core (settings.site_learning_max_core_buttons_per_page) — น้อย = สำคัญกว่า:
    0 form-submit > 1 label ตรง CORE_ACTION_KEYWORDS เป๊ะ > 2 มีเป็น substring > 3 อื่นๆ"""
    if button_info.is_form_submit:
        return 0
    label = _button_label_text(button_info)
    if label in CORE_ACTION_KEYWORDS:
        return 1
    if any(word in label for word in CORE_ACTION_KEYWORDS):
        return 2
    return 3

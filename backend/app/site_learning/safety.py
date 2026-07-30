"""site_learning/safety.py — W14: กันไม่ให้ crawler กด action ที่อาจเปลี่ยนแปลงข้อมูล
ระหว่าง "เรียนรู้เว็บไซต์" — ต่างจาก permission/rules.py (blocklist-first, default-allow,
ใช้กับ agent loop ปกติที่ทำงานจริงตาม goal ของ user โดยมี human-in-the-loop คอยขอ
อนุมัติ action เสี่ยงอยู่แล้ว) เพราะโหมด crawl ไม่มี human คอยดูอยู่ทุกคลิก ต้อง
"default-deny" เข้มกว่ามาก (ตามสเปค: "หากไม่แน่ใจ ห้ามกด") — อนุญาตเฉพาะ action ที่
ชัดเจนว่าเป็นแค่การเดินสำรวจ/ดูข้อมูล (navigation) เท่านั้น ไม่ใช่ default อนุญาตแล้วเช็ค
blocklist ทีหลังแบบ permission/rules.py

W36: เพิ่มชั้น "core function classification" แยกต่างหากจาก is_crawl_safe()/
is_safe_nav_link() ข้างบนโดยสิ้นเชิง — เป้าหมายคนละอย่างกัน: is_crawl_safe() ตอบว่า "กด
ปุ่มนี้ได้อย่างปลอดภัยไหม" (ยังทำงานเหมือนเดิมทุกประการ ไม่ถูกแก้เลย) ส่วน
classify_button_tier()/button_core_priority() ด้านล่างตอบว่า "ปุ่มนี้ควรค่าแก่การไล่กด
ก่อนไหม" (ลดจำนวนปุ่มที่ _explore_buttons() ต้องไล่กดต่อหน้า แก้ปัญหา self-learning กดปุ่ม
เยอะเกินความจำเป็นบนหน้าที่มีปุ่มรอง/ตกแต่งเยอะ เช่น filter/sort/share/like/notification)
— ปุ่มที่ tier="core" ยังต้องผ่าน is_crawl_safe() แบบเดิมทุกประการก่อนกดจริงอยู่ดี (เช่น
CORE_ACTION_KEYWORDS ด้านล่างมี "submit"/"save"/"login" อยู่ด้วยเพื่อจัดชั้นให้ถูกหมวด แต่
ปุ่มพวกนี้ยังถูก BLOCKED_CRAWL_KEYWORDS ข้างบนบล็อกไม่ให้กดจริงเหมือนเดิม — tier ไม่ใช่
safety gate ตัวใหม่ ไม่ได้ลด/เพิ่มความเข้มงวดของ Safety Rule เดิมเลย)

W37: user ระบุตรงๆ ว่าต้องการแค่ "เรียนรู้โครงสร้างของหน้าเว็บ" ไม่ต้องการให้ crawler กดดู
วีดีโอเลย (ยกตัวอย่าง YouTube/Facebook/Instagram) เพราะวีดีโอแต่ละคลิปแนะนำคลิปถัดไปต่อเรื่อยๆ
ไม่รู้จบ (recommendation feed) พาให้ crawler เดินเข้าไปแล้วไม่ถอยกลับมาสำรวจส่วนอื่นของเว็บอีก
เลย ทั้งที่คลิปวีดีโอเป็น "เนื้อหา" ไม่ใช่ "โครงสร้าง" ของเว็บที่ user อยากได้จริงๆ — W28-W36
(ข้างบนทั้งหมด) บรรเทาอาการ "วนลูป" ได้ระดับหนึ่ง (จำกัดจำนวนครั้ง/ตัดปุ่มรอง) แต่ยังปล่อยให้
เดินเข้าไปดูวีดีโอได้บ้างอยู่ดี (แค่จำกัดจำนวน ไม่ได้ตัดออกทั้งหมด) — เพิ่ม
is_video_content_url()/is_video_content_label() ด้านล่าง กันไม่ให้ crawler navigate เข้าหน้า
"ดูวีดีโอ" เลยแม้แต่ครั้งเดียว (ต่างจากทุกชั้นก่อนหน้าที่แค่จำกัดจำนวน) ใช้ที่ 3 จุดใน
crawler.py: (1) nav_links BFS ก่อนต่อคิว (2) classify_button_tier() ให้ปุ่ม/label ที่บ่งบอก
วีดีโอเป็น tier="decorative" (ไม่กดเลย) (3) หลัง DFS-click navigate สำเร็จแล้ว เช็ค URL
ปลายทางอีกชั้น (เผื่อ label เดิมไม่มีคำใบ้ตรงๆ เช่น thumbnail ที่ label เป็นแค่ชื่อคลิป) —
เช็คจาก URL path เป็นหลัก (ไม่ผูกกับโดเมนใดโดเมนหนึ่ง ใช้ได้กับเว็บวีดีโออื่นๆ ที่ใช้ path
convention คล้ายกันด้วย ไม่ใช่แค่ 3 เว็บที่ user ยกตัวอย่าง)

W38: user ยืนยันว่า W37 (กันดูวีดีโอ) แก้ได้ถูกต้องแล้ว แต่เจอปัญหาคล้ายกันแบบใหม่ — crawler
"ติดลูปกดดูแฮชแท็ก" แทน (กด "#คำ" ในโพสต์ -> ไปหน้ารวมโพสต์ที่ติด hashtag เดียวกันนับพันโพสต์
-> มีลิงก์แฮชแท็กอื่นในหน้านั้นอีก -> กดต่อไปเรื่อยๆ ไม่รู้จบ เหมือนปัญหาวีดีโอเป๊ะ แค่คนละ
ประเภทเนื้อหา) — เพิ่ม is_hashtag_url()/is_hashtag_label() ด้านล่าง ใช้แพทเทิร์นเดียวกับ W37
ทุกจุด (nav_links BFS/classify_button_tier/post-click safety net ใน crawler.py) สัญญาณที่
เชื่อถือได้สุดคือ label ของลิงก์แฮชแท็กเองเกือบทุกเว็บขึ้นต้นด้วย "#" ตรงๆ ให้เห็น (ไม่ต้องพึ่ง
URL convention เฉพาะเว็บเลยด้วยซ้ำ ต่างจากวีดีโอที่ label บางทีไม่มีคำใบ้เลย ต้องพึ่ง URL เป็น
หลัก) — ยังเพิ่ม URL path segment check คู่กันไว้เป็น fallback (path segment "hashtag"/
"hashtags" เท่านั้น ไม่รวม "tag"/"tags" เฉยๆ เพราะเว็บทั่วไปจำนวนมาก เช่น blog/เอกสาร ใช้
"/tags/xxx" เป็นหมวดหมู่บทความปกติ ไม่ใช่ hashtag แบบโซเชียล การรวมเข้าไปจะตัดโครงสร้างที่
ควรเรียนรู้จริงทิ้งไปโดยไม่จำเป็น)
"""

import re
import urllib.parse

from backend.app.site_learning.schema import ButtonInfo

# คำที่บ่งบอกว่า action นี้ "แค่เดินสำรวจ/ดูข้อมูล" ปลอดภัยให้ crawler กดได้ระหว่างเรียนรู้
# W18: เพิ่ม back/forward — คำที่ icon ปุ่มย้อนกลับ/ไปต่อมักใช้ (มาจาก icon_hint หรือ
# aria-label เช่น "Go back") ความหมายเดียวกับ previous/next ที่มีอยู่แล้ว
ALLOWED_CRAWL_KEYWORDS = {
    "menu", "view", "detail", "next", "previous", "expand", "collapse",
    "filter", "search", "pagination", "page", "tabs", "tab", "open", "close",
    "back", "forward",
    # W20: ปุ่ม "Continue" (multi-step form/wizard เช่น checkout) ความหมายเดียวกับ
    # "next" ที่อนุญาตอยู่แล้ว — แค่ไปหน้าถัดไปของ flow ไม่ใช่ commit อะไร (label ที่มีคำ
    # เสี่ยงร่วมด้วย เช่น "Continue to Payment" ยังโดน BLOCKED_CRAWL_KEYWORDS บล็อกตามปกติ
    # เพราะ BLOCKED ชนะ ALLOWED เสมอ)
    "continue",
    # W19: ไอคอนตะกร้า/กระเป๋าสินค้า (มุมขวาบนของเว็บ e-commerce ทั่วไป) เป็นแค่ "ไป
    # ดูตะกร้า" — read-only navigation ล้วนๆ ไม่ต่างจาก "view"/"detail" ที่อนุญาตอยู่แล้ว
    # ไม่ได้เปลี่ยนแปลงข้อมูลอะไรเลย จำเป็นสำหรับให้ crawler เดินไปสำรวจหน้า cart/checkout
    # ต่อได้ (ก่อนหน้านี้ label ว่างเปล่า + ไม่มีคำไหนตรง allowlist เลย ทำให้ปุ่มนี้โดน
    # default-deny ตลอด ไม่เคยถูกกดสักครั้ง — manual เลยไม่มีหน้า cart/checkout บันทึกไว้
    # เลย) *** ระวัง: "add to cart"/"add to bag" ก็มีคำว่า cart/bag อยู่ในนั้นด้วย แต่เป็น
    # action ที่เปลี่ยนสถานะจริง (เพิ่มสินค้า) ไม่ใช่แค่เดินดู — กันด้วยการเติม "add to" ใน
    # BLOCKED_CRAWL_KEYWORDS ด้านล่าง (BLOCKED ชนะ ALLOWED เสมอ ดู is_crawl_safe()) ***
    "cart", "bag", "basket",
    # W20: ปุ่ม "Checkout" (บนหน้า cart) เป็นแค่การไปหน้าถัดไปของ flow (กรอกที่อยู่จัดส่ง)
    # ไม่ใช่การ "ยืนยันคำสั่งซื้อ" จริง — ปุ่มที่ commit คำสั่งซื้อจริงๆ (Place Order/
    # Confirm/Pay/Submit) ยังอยู่ใน BLOCKED_CRAWL_KEYWORDS ด้านล่างเหมือนเดิม เอาออกจาก
    # blocked มาไว้ allowed ตรงนี้แทน ให้ crawler เดินสำรวจหน้าฟอร์ม checkout ต่อจากหน้า
    # cart ได้ (ก่อนหน้านี้ "checkout" อยู่ใน BLOCKED ทำให้ไปได้แค่หน้า cart หน้าเดียว)
    "checkout",
}

# คำที่บ่งบอกว่า action นี้อาจเปลี่ยนแปลง/ทำลายข้อมูลจริง — ห้ามกดเด็ดขาดระหว่าง crawl
# ไม่ว่า label จะเข้าข่าย ALLOWED_CRAWL_KEYWORDS ด้วยพร้อมกันหรือไม่ก็ตาม (BLOCKED ชนะเสมอ
# เช่น label "View & Delete" ต้องถือว่าไม่ปลอดภัย)
BLOCKED_CRAWL_KEYWORDS = {
    "delete", "remove", "save", "submit", "confirm", "purchase",
    "payment", "reset", "logout", "approve", "reject", "execute", "send", "sync",
    # W19: "add to " (เว้นวรรคท้ายตั้งใจ กันชนกับคำอื่นที่บังเอิญมี "add" เป็น substring
    # เช่น "Address"/"Additional") ครอบคลุม "Add to cart"/"Add to bag"/"Add to wishlist"
    # ฯลฯ — ปุ่มพวกนี้เปลี่ยนสถานะจริง (เพิ่มสินค้า/รายการ) ไม่ใช่แค่เดินสำรวจดูเฉยๆ ต้อง
    # ยังถูกบล็อกอยู่แม้จะเพิ่ง allow "cart"/"bag"/"basket" ไปด้านบน (BLOCKED ชนะ ALLOWED
    # เสมอเมื่อ label ตรงทั้งคู่พร้อมกัน — ดู is_crawl_safe())
    "add to ",
}

# ชนิด action ที่ปลอดภัยโดยธรรมชาติเสมอ ไม่ต้องพึ่ง label เลย — เลื่อนจอ/รอเฉยๆ ไม่มีทาง
# เปลี่ยนแปลงข้อมูลบน server ได้ ("goto" ตั้งใจไม่รวมไว้ตรงนี้ — แม้ปกติจะปลอดภัย แต่ปุ่ม/
# ลิงก์ที่ label ตรง BLOCKED_CRAWL_KEYWORDS (เช่น "Logout") ก็ยัง goto/navigate ไปกระตุ้น
# effect ได้จริงถ้าปลายทางเป็น URL แบบ GET-triggers-action — ให้ตกไปเช็ค label ตามปกติ
# เหมือน click แทนที่จะยกเว้นเฉยๆ)
_INHERENTLY_SAFE_TYPES = {"scroll", "wait"}


def is_crawl_safe(label: str, cmd_type: str = "") -> bool:
    """ใช้ตัดสินใจว่า "จะกดปุ่ม/element นี้ระหว่าง crawl ไหม" (ตรงกับสเปค Safety Rules
    ตรงๆ) — label: ข้อความ/aria-label/title ของ element เป้าหมาย, cmd_type: ประเภท
    action (เช่น "click") ไม่ส่งมาก็ได้ (default "")

    Default-deny: คืน True ก็ต่อเมื่อ label หรือ cmd_type ตรงคำใน ALLOWED_CRAWL_KEYWORDS
    จริงๆ (หรือ cmd_type เป็นชนิดที่ปลอดภัยโดยธรรมชาติ) และไม่ตรงคำใน
    BLOCKED_CRAWL_KEYWORDS เลย — ถ้า label ว่างเปล่า หรือไม่ตรงคำไหนใน allowlist เลย
    (ไม่แน่ใจ) ให้ถือว่าไม่ปลอดภัยเสมอ ("หากไม่แน่ใจ ห้ามกด") — BLOCKED ชนะ ALLOWED เสมอ
    ถ้าตรงทั้งคู่พร้อมกัน

    หมายเหตุ: นี่คนละคำถามกับ is_safe_nav_link() ด้านล่าง — ฟังก์ชันนี้ตอบว่า "กดปุ่มนี้ไหม"
    (default-deny เข้ม) ส่วน is_safe_nav_link() ตอบว่า "เดินตามลิงก์เมนู/nav นี้ไหม"
    (default-allow — เมนูทั่วไปอย่าง "Dashboard"/"Products" ไม่มีทางอยู่ใน
    ALLOWED_CRAWL_KEYWORDS ตรงๆ เพราะเป็นชื่อ feature เฉพาะเว็บ ไม่ใช่คำ action ทั่วไป)"""
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
    """ต่างจาก is_crawl_safe() ตรงที่นี่คือ default-allow (บล็อกเฉพาะที่ตรง
    BLOCKED_CRAWL_KEYWORDS ชัดเจนเท่านั้น) — ใช้ตัดสินใจว่า "จะเดินตาม (goto) ลิงก์ nav
    นี้ไหม" ตอน BFS สำรวจเมนู (ดู crawler.py) ซึ่งเป็นคำถามคนละแบบจาก "จะกดปุ่มนี้ไหม":
    เมนู/nav item ทั่วไป (Dashboard, Products, Orders) ปลอดภัยที่จะเดินตามเสมอแม้จะไม่
    ตรงคำไหนใน ALLOWED_CRAWL_KEYWORDS เลยก็ตาม (เพราะนั่นเป็นชื่อ feature เฉพาะเว็บ ไม่ใช่
    คำ action ทั่วไป) ยกเว้นชัดเจนว่าเป็นลิงก์ที่กด GET แล้ว trigger effect ทันที (เช่น
    "Logout" ที่มักเป็นแค่ <a href="/logout">)"""
    lower = (text or "").lower()
    return not any(word in lower for word in BLOCKED_CRAWL_KEYWORDS)


# W36: คำที่บ่งบอกว่าปุ่มนี้เป็น "ฟังก์ชันหลักของหน้า" — แยกต่างหากจาก ALLOWED_CRAWL_KEYWORDS
# ข้างบน (อันนั้นใช้กับ is_crawl_safe()/is_safe_nav_link() เป็น safety gate อยู่แล้ว ไม่แตะ)
# เซตนี้ใช้แค่จัดชั้น (tier) + จัดลำดับความสำคัญตอนต้องตัด top-K เมื่อปุ่ม core เกินเพดาน
# ต่อหน้า (ดู button_core_priority/settings.site_learning_max_core_buttons_per_page) — มีคำ
# ที่ยังโดน BLOCKED_CRAWL_KEYWORDS บล็อกอยู่ (submit/save/login) โดยตั้งใจ: การจัดเป็น
# tier="core" ไม่ได้แปลว่ากดได้ แค่บอกว่า "ถ้ากดได้ (ผ่าน is_crawl_safe ด้วย) นี่คือปุ่ม
# สำคัญที่ควรกดก่อนปุ่มรอง/ตกแต่ง"
CORE_ACTION_KEYWORDS = {
    "search", "filter", "sort", "add to cart", "add to bag", "checkout",
    "save", "apply", "submit", "login", "log in", "sign in",
}

# W36: คำที่บ่งบอกว่าปุ่มนี้เป็น "ปุ่มรอง/ตกแต่ง" — ไม่ใช่ฟังก์ชันหลักของหน้า ไม่ควรเสียเวลา
# ไล่กดระหว่าง crawl เลย (ข้ามตั้งแต่ต้นใน _explore_buttons() ไม่ต้องเช็ค is_crawl_safe()
# ด้วยซ้ำ — ต่างจาก BLOCKED_CRAWL_KEYWORDS ที่ยังต้องเช็คแล้วปฏิเสธ เซตนี้ไม่เข้าเงื่อนไข
# "อาจเปลี่ยนแปลงข้อมูล" เลย แค่ไม่คุ้มเวลาสำรวจ)
DECORATIVE_KEYWORDS = {
    "share", "like", "unlike", "favorite", "favourite", "wishlist",
    "upvote", "downvote", "notification", "bell", "theme", "dark mode",
    "light mode", "language", "locale", "avatar", "profile menu",
    "account menu", "breadcrumb",
}

# W36: ปุ่ม pagination ที่เป็นเลขหน้ามากกว่า 1 ตรงๆ (เช่น "2", "3", "Page 4") — ถือเป็น
# decorative เพราะเป็นแค่การดูเนื้อหาชุดเดียวกันหน้าถัดไป ไม่ใช่ path/ฟีเจอร์ใหม่ของเว็บ (ต่าง
# จากปุ่ม "Next"/"Previous"/"pagination" ทั่วไปใน ALLOWED_CRAWL_KEYWORDS ที่ยังอนุญาตอยู่ —
# เลขหน้าเจาะจงแบบนี้ไล่กดได้ไม่รู้จบถ้าเนื้อหามีหลายสิบหน้า เสียเวลาโดยไม่ได้ข้อมูลใหม่)
_PAGE_NUMBER_PATTERN = re.compile(r"^(?:page\s*)?(\d+)$", re.IGNORECASE)

# W37: path segment ที่บ่งบอกว่า URL นี้เป็นหน้า "ดูวีดีโอ" (watch/shorts/reel ฯลฯ) —
# ตรวจจาก path เท่านั้น (ไม่ผูกกับโดเมนใดโดเมนหนึ่ง) ให้ครอบคลุมเว็บวีดีโอ/โซเชียลทั่วไปที่ใช้
# path convention คล้ายกัน ไม่ใช่แค่ YouTube/Facebook/Instagram ที่ user ยกตัวอย่างมา — ใช้กัน
# ไม่ให้ crawler navigate เข้าไปเลยแม้แต่ครั้งเดียว (ดู is_video_content_url) ต่างจาก W28/W33
# ที่แค่จำกัดจำนวนครั้ง/หน้าซ้ำ
#
# *** ต้องเทียบแบบ "segment ทั้งชิ้นตรงกันเป๊ะ" (ดู is_video_content_url) ไม่ใช่ substring
# ธรรมดา — ตอนแรกใช้ substring แล้วพบจากการรัน test suite จริงว่าไปชนกับ URL ที่บังเอิญมีคำนี้
# เป็นส่วนหนึ่งของคำยาวกว่า เช่น "/shorts1.html" (fixture เดิมของ W28 ที่จำลอง YouTube Shorts
# ด้วยชื่อไฟล์ธรรมดา ไม่ใช่ path จริง) และในเว็บจริงก็เสี่ยง false-positive กับเว็บขายเสื้อผ้าที่
# มี path เช่น "/shorts-collection" ได้เหมือนกัน — segment ที่ตรงกันเป๊ะเท่านั้นถึงจะนับ ***
VIDEO_CONTENT_URL_PATH_SEGMENTS = {
    "watch", "shorts", "reel", "reels", "video", "videos", "embed", "clip", "clips",
}

# W37: label ของปุ่ม/ลิงก์ที่บ่งบอกตรงๆ ว่า "กดแล้วจะเล่นวีดีโอ" — ต่างจาก
# VIDEO_CONTENT_URL_PATH_SEGMENTS ข้างบนที่เช็คจาก URL ปลายทาง (รู้ได้แน่นอนกว่าแต่ต้อง
# navigate ไปก่อน) อันนี้เช็คจาก label ก่อน navigate เลย จำเป็นสำหรับปุ่ม (ต่างจาก <a href> ที่
# รู้ปลายทางล่วงหน้าได้จาก href) เพราะปุ่มส่วนใหญ่ไม่มี href ให้เช็ค URL ล่วงหน้า — เป็น label
# (ข้อความที่มนุษย์อ่าน ไม่ใช่ path segment) เลยยังใช้ substring match แบบเดิม (เหมือน
# DECORATIVE_KEYWORDS/CORE_ACTION_KEYWORDS ทุกตัวในไฟล์นี้)
VIDEO_CONTENT_LABEL_KEYWORDS = {"watch", "play video", "play now", "shorts", "reel", "reels"}


# W38: path segment ที่บ่งบอกว่า URL นี้เป็นหน้ารวมโพสต์ของ "แฮชแท็ก" (hashtag/topic feed) —
# ตั้งใจไม่รวม "tag"/"tags" เฉยๆ (ต่างจาก "hashtag"/"hashtags" ที่ไม่กำกวม) เพราะเว็บทั่วไป
# จำนวนมาก (blog/เอกสาร/E-commerce) ใช้ "/tags/xxx" เป็นหมวดหมู่บทความ/สินค้าปกติ ไม่ใช่
# hashtag แบบโซเชียลที่แนะนำโพสต์อื่นไม่รู้จบ — รวมเข้าไปจะตัดโครงสร้างที่ควรเรียนรู้จริงทิ้ง
# ไปโดยไม่จำเป็น (ดู is_hashtag_url — segment ต้องตรงเป๊ะเหมือน VIDEO_CONTENT_URL_PATH_SEGMENTS)
HASHTAG_URL_PATH_SEGMENTS = {"hashtag", "hashtags"}

# W38: pattern จับ label ที่ขึ้นต้นด้วย "#" ตามด้วยตัวอักษร/ตัวเลข/underscore อย่างน้อย 1 ตัว
# (เช่น "#travel", "#Cat_Lovers") — นี่คือสัญญาณที่เชื่อถือได้ที่สุดสำหรับแฮชแท็ก เพราะแทบทุก
# เว็บ (Instagram/X/TikTok/Facebook ฯลฯ) render แฮชแท็กด้วย "#" นำหน้าให้เห็นตรงๆ ในข้อความ
# เสมอ ไม่ต้องพึ่ง URL convention เฉพาะเว็บเลยด้วยซ้ำ (ต่างจากวีดีโอที่บางทีปุ่ม/thumbnail ไม่มี
# คำใบ้ใน label เลย ต้องพึ่ง URL เป็นหลัก — ดู is_video_content_label) ไม่ใช้ .match กับ "#"
# เดี่ยวๆ หรือ "# " (เว้นวรรค) เพราะนั่นไม่ใช่แฮชแท็กจริง อาจเป็นแค่สัญลักษณ์ตกแต่ง
_HASHTAG_LABEL_PATTERN = re.compile(r"^#\w")


def _url_path_segments(url: str) -> list[str]:
    """W37/W38: ตัด path ของ url เป็น segment (คั่นด้วย "/") กรอง segment ว่างทิ้ง — ใช้ร่วม
    กันทั้ง is_video_content_url และ is_hashtag_url เทียบ segment ทั้งชิ้นเป๊ะ กันปัญหา
    false-positive จาก substring บังเอิญ (ดู comment ของ VIDEO_CONTENT_URL_PATH_SEGMENTS)"""
    path = urllib.parse.urlparse(url).path.lower()
    return [seg for seg in path.split("/") if seg]


def is_video_content_url(url: str) -> bool:
    """W37: True ถ้า URL นี้น่าจะเป็นหน้า "ดูวีดีโอ" (watch/shorts/reel ฯลฯ) — ใช้กันไม่ให้
    crawler เดินเข้าไปสำรวจ (ทั้ง BFS nav_links ก่อนต่อคิว และหลัง DFS-click navigate ไปแล้ว
    จริง — ดู crawler.py) เพราะวีดีโอแนะนำวีดีโอถัดไปไม่รู้จบ ไม่ใช่ "โครงสร้างเว็บ" ที่ user
    ต้องการให้เรียนรู้ — เช็คว่า path segment ไหนตรงกับ VIDEO_CONTENT_URL_PATH_SEGMENTS แบบ
    เป๊ะทั้งชิ้น (ไม่ใช่ substring — ดู comment ของ set นั้นสำหรับเหตุผล) เช่น
    "/watch"/"/shorts/abc123"/"/reel/42" ตรง แต่ "/shorts-collection"/"/watchlist" ไม่ตรง"""
    return any(seg in VIDEO_CONTENT_URL_PATH_SEGMENTS for seg in _url_path_segments(url))


def is_video_content_label(label: str) -> bool:
    """W37: True ถ้า label (text/aria-label/title ของปุ่มหรือข้อความของ nav link) บ่งบอก
    ตรงๆ ว่ากดแล้วจะเล่นวีดีโอ — ใช้กับปุ่มใน classify_button_tier() (ก่อนเช็ค
    is_nav_menu_item ด้วยซ้ำ) และกับ nav link text ใน crawler.py::_record_page()"""
    lower = (label or "").lower()
    return any(word in lower for word in VIDEO_CONTENT_LABEL_KEYWORDS)


def is_hashtag_url(url: str) -> bool:
    """W38: True ถ้า URL นี้น่าจะเป็นหน้ารวมโพสต์ของแฮชแท็ก — เช็คว่า path segment ไหนตรงกับ
    HASHTAG_URL_PATH_SEGMENTS แบบเป๊ะทั้งชิ้น (เหมือน is_video_content_url) เช่น
    "/hashtag/travel" ตรง แต่ "/tags/travel" (หมวดหมู่บทความทั่วไป ไม่ใช่แฮชแท็กโซเชียล)
    ไม่ตรง — เป็น fallback รองจาก is_hashtag_label() (label เชื่อถือได้กว่ามาก)"""
    return any(seg in HASHTAG_URL_PATH_SEGMENTS for seg in _url_path_segments(url))


def is_hashtag_label(label: str) -> bool:
    """W38: True ถ้า label ขึ้นต้นด้วย "#" ตามด้วยตัวอักษร/ตัวเลข (ดู
    _HASHTAG_LABEL_PATTERN) — สัญญาณหลักที่ใช้กันแฮชแท็ก เชื่อถือได้มากกว่า
    is_hashtag_url() เพราะไม่ต้องพึ่ง URL convention เฉพาะเว็บเลย"""
    return bool(_HASHTAG_LABEL_PATTERN.match((label or "").strip()))


def _button_label_text(button_info: ButtonInfo) -> str:
    """รวม text/aria_label/title/icon_hint ของปุ่มเป็นสตริงเดียวตัวพิมพ์เล็ก ไว้เช็ค keyword
    ใน classify_button_tier()/button_core_priority() — ต่างจาก crawler.py::_button_label()
    ที่คืนแค่ตัวแรกที่ไม่ว่าง (ใช้แสดงผล/ตัดสิน is_crawl_safe ทีละคำ) ตัวนี้รวมทุกฟิลด์เข้า
    ด้วยกันเพราะ keyword ที่บ่งบอก tier อาจอยู่ในฟิลด์ไหนก็ได้ (เช่น icon-only button ที่
    ความหมายอยู่ใน icon_hint ล้วนๆ ไม่มี text เลย)"""
    return " ".join(
        part for part in (button_info.text, button_info.aria_label, button_info.title, button_info.icon_hint)
        if part
    ).strip().lower()


def classify_button_tier(button_info: ButtonInfo) -> str:
    """W36/W37/W38: จัดชั้นปุ่มเป็น "nav" (เมนู/nav item) / "core" (ฟังก์ชันหลักของหน้า) /
    "decorative" (ปุ่มรอง/ตกแต่ง — รวมถึงปุ่มที่กดแล้วเล่นวีดีโอ (W37) หรือเปิดหน้าแฮชแท็ก
    (W38)) — heuristic ล้วนๆ จาก keyword + สัญญาณ DOM ที่ extract มาแล้ว (is_nav_menu_item,
    is_form_submit, role) ไม่เรียก LLM เลย (ตามแนวที่ตัดสินใจไว้ตั้งแต่ W14/W24 ว่า crawler
    ต้อง deterministic กัน token cost)

    ลำดับการเช็ค: (0) W37/W38: label ตรง VIDEO_CONTENT_LABEL_KEYWORDS หรือขึ้นต้นด้วย "#"
    (is_hashtag_label) -> "decorative" ทันที ก่อนเช็ค is_nav_menu_item ด้วยซ้ำ — ตั้งใจให้ชนะ
    แม้เป็นเมนู/tab จริง (เช่น tab "Watch"/"Reels" ในแถบเมนูหลักของ Facebook, หรือแฮชแท็ก
    "#Trending" ที่ปักหมุดไว้ในเมนู) เพราะ user ระบุตรงๆ ว่าไม่ต้องการให้กดเข้าไปดูวีดีโอ/
    แฮชแท็กเลยไม่ว่าจะมาในรูปแบบเมนูหรือปุ่มทั่วไปก็ตาม (1) is_nav_menu_item -> "nav" (เมนู/
    nav item อื่นๆ ที่ไม่ใช่วีดีโอ/แฮชแท็ก ยังคงเป็น "nav" เสมอไม่ว่า label จะมีคำ decorative/
    core ปนอยู่ด้วยหรือไม่ เช่น เมนู "Share Settings" ในแถบเมนูหลักยังนับเป็น nav ไม่ใช่
    decorative) (2) เลขหน้า pagination > 1 หรือ label ตรง DECORATIVE_KEYWORDS -> "decorative"
    (3) is_form_submit หรือ label ตรง CORE_ACTION_KEYWORDS หรือ role="search" -> "core" (4)
    fallback ตั้งใจให้เป็น "core" เสมอ (ไม่มี tier ที่ 4 ให้เลือก) — สำคัญมาก: ปุ่มที่ไม่เข้า
    เงื่อนไขไหนชัดเจนเลย (เช่น "View"/"Expand" ที่ W16 ตั้งใจให้ไล่กดสำรวจ path ที่ nav เดินไม่
    ถึง) ต้องยังเป็น "core" ไม่ใช่ "decorative" มิฉะนั้นความสามารถเดิมของ W16 จะหายไปหมด —
    is_crawl_safe() (ไม่ถูกแก้เลยในงานนี้) ยังคงเป็นตัวตัดสินสุดท้ายเหมือนเดิมว่าปุ่ม
    tier="core" กดได้จริงไหม"""
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

    if button_info.is_form_submit:
        return "core"
    if any(word in label for word in CORE_ACTION_KEYWORDS):
        return "core"
    if (button_info.role or "").strip().lower() == "search":
        return "core"

    return "core"


def button_core_priority(button_info: ButtonInfo) -> int:
    """W36: ลำดับความสำคัญของปุ่ม tier="core" ใช้ตอนต้องตัด top-K เมื่อจำนวนปุ่ม core บน
    หน้าเดียวเกิน settings.site_learning_max_core_buttons_per_page (ดู
    crawler.py::_explore_buttons) — ยิ่งค่าน้อยยิ่งสำคัญกว่า (0 = สูงสุด) ตามลำดับที่ระบุ:
    form-submit > exact keyword match (label ทั้งก้อนตรงคำใน CORE_ACTION_KEYWORDS เป๊ะ) >
    partial match (คำใน CORE_ACTION_KEYWORDS เป็นแค่ substring ของ label) > อื่นๆ (fallback
    core ที่ไม่ตรง keyword ไหนเลย เช่น "View"/"Expand")"""
    if button_info.is_form_submit:
        return 0
    label = _button_label_text(button_info)
    if label in CORE_ACTION_KEYWORDS:
        return 1
    if any(word in label for word in CORE_ACTION_KEYWORDS):
        return 2
    return 3

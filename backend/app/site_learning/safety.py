"""site_learning/safety.py — W14: กันไม่ให้ crawler กด action ที่อาจเปลี่ยนแปลงข้อมูล
ระหว่าง "เรียนรู้เว็บไซต์" — ต่างจาก permission/rules.py (blocklist-first, default-allow,
ใช้กับ agent loop ปกติที่ทำงานจริงตาม goal ของ user โดยมี human-in-the-loop คอยขอ
อนุมัติ action เสี่ยงอยู่แล้ว) เพราะโหมด crawl ไม่มี human คอยดูอยู่ทุกคลิก ต้อง
"default-deny" เข้มกว่ามาก (ตามสเปค: "หากไม่แน่ใจ ห้ามกด") — อนุญาตเฉพาะ action ที่
ชัดเจนว่าเป็นแค่การเดินสำรวจ/ดูข้อมูล (navigation) เท่านั้น ไม่ใช่ default อนุญาตแล้วเช็ค
blocklist ทีหลังแบบ permission/rules.py
"""

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

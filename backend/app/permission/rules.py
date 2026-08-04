"""Permission layer: allowlist / blocklist action + human-in-the-loop.

W1: skeleton only. W4-5: implement จริง (ดู roadmap.txt เฟส 1).

Adapted from PR "permission-ab" (origin/permission-ab): เดิม classify_action()
รับ action_name: str เฉยๆ — เปลี่ยนเป็นรับ cmd dict ทั้งก้อน เพราะ risk ของบาง action
(เช่น goto) ขึ้นกับ parameter อื่นด้วย ไม่ใช่แค่ type (เช่น goto ไป domain ที่ถูกบล็อก)
"""

import urllib.parse
from enum import Enum


class ActionRisk(str, Enum):
    SAFE = "safe"
    NEEDS_CONFIRMATION = "needs_confirmation"
    BLOCKED = "blocked"


# ตัวอย่าง action ที่ต้องขอยืนยันก่อนเสมอ (เช่น submit ฟอร์ม, ลบข้อมูล, ชำระเงิน)
DEFAULT_NEEDS_CONFIRMATION = {"submit", "delete", "purchase", "pay"}
DEFAULT_BLOCKED_ACTIONS: set[str] = set()

# โดเมนที่ไม่อนุญาตให้เข้าถึงเด็ดขาด (Blocklist)
BLOCKED_DOMAINS = {
    "malicious.com",
    "phishing.net",
}

# ถ้า ALLOWED_DOMAINS มีค่า จะบล็อกโดเมนที่ไม่อยู่ในนี้ทั้งหมด (ถ้าว่าง แปลว่าอนุญาตทั้งหมดที่ไม่ได้ถูกบล็อก)
ALLOWED_DOMAINS: set[str] = set()

# ชั้นสำรอง (defense-in-depth) นอกจาก type: LLM อาจส่ง type="click" ธรรมดาสำหรับปุ่มที่
# จริงๆ แล้วมีผลสำคัญ/ย้อนกลับยาก (เช่น saucedemo ปุ่ม "Remove" ในตะกร้าเป็นแค่
# <button>Remove</button> ธรรมดา ไม่มี type พิเศษอะไรให้สังเกตเลยนอกจากป้ายข้อความ) —
# ไม่ควรพึ่งแค่ LLM เลือก type (submit/delete/purchase/pay) ให้ถูกต้องเพียงอย่างเดียว
# เพราะเป็นเรื่อง model compliance ที่ไม่การันตี — เช็คจากคำในป้าย element ประกอบด้วย
RISKY_LABEL_KEYWORDS = {
    "remove", "delete", "place order", "finish", "pay", "purchase", "confirm",
}

# W_search: บั๊กจริงที่ user รายงาน — action ที่ย้อนกลับได้ง่ายมากและไม่มีผลถาวรใดๆ เลย
# (กดปุ่มค้นหา, คลิกเข้าไปดูวิดีโอ/บทความ) มักถูกจัดเป็น NEEDS_CONFIRMATION ผิดๆ เพราะ
# LLM บางครั้งเลือก action type "submit" ให้ปุ่มค้นหา (ตีความ "ค้นหา" ว่าเป็นการ "ส่งฟอร์ม"
# ทางความหมาย ทั้งที่ SYSTEM_PROMPT (llm.py) สั่งห้ามเดาแบบนี้ไว้แล้ว — model compliance
# ไม่การันตี 100% เหมือนที่ RISKY_LABEL_KEYWORDS ด้านบนก็มีไว้เพราะเหตุผลเดียวกัน ฝั่งตรงข้าม)
# — เช็ค label สวนทางกัน: ถ้าดูชัดเจนว่าเป็นแค่ค้นหา/กรอง/เปิดดูเนื้อหา (ไม่ใช่ฟอร์มที่มี
# ผลจริงเช่น สั่งซื้อ/ลบ/จ่ายเงิน) ให้ลดระดับกลับเป็น SAFE แม้ action_type จะดูเสี่ยง —
# RISKY_LABEL_KEYWORDS ยังคงชนะเสมอถ้า label match ทั้งสองฝั่ง (เช่น "Confirm and Search"
# ที่จริงๆ อยู่ในหน้า checkout) ระวังไว้ก่อนดีกว่า
SAFE_ACTION_LABEL_KEYWORDS = {
    "search", "ค้นหา", "ค้น", "find", "filter", "กรอง",
    "watch", "ดู", "ชม", "play", "เล่น", "view", "read", "อ่าน",
    "browse", "next", "ถัดไป", "previous", "ก่อนหน้า", "go to", "open",
}

# W_search follow-up: บั๊กจริงที่ user รายงานต่อ — คลิกเลือกวิดีโอ/บทความจากผลการค้นหา
# (เช่น การ์ดวิดีโอ YouTube) ป้ายของ element มักเป็น "ชื่อเรื่อง" ดิบๆ (เช่น "เพลงรัก -
# Three Man Down |Official MV|") ซึ่งไม่มีทางไป match SAFE_ACTION_LABEL_KEYWORDS ได้เลย
# (เนื้อหาอิสระ ไม่ใช่คำกริยาสั่งงาน) แม้ตัวมันจะปลอดภัยมากก็ตาม (แค่ navigate ไปดู/เล่น) —
# ใช้สัญญาณเชิงโครงสร้างแทน label: ปุ่ม/action ที่มีผลจริง (submit ฟอร์ม, ลบ, สั่งซื้อ,
# จ่ายเงิน) แทบไม่มีทางเป็น <a> (anchor/ลิงก์ธรรมดา) เพราะ anchor แค่พาไปหน้าอื่น (GET,
# ย้อนกลับได้ง่าย) ไม่ได้ submit ข้อมูลอะไรเลย — ปุ่มพวกนี้เกือบทั้งหมดเป็น <button>/input
# ที่แท้จริง สั่งให้ tag == "a" ลดระดับ type ที่ดูเสี่ยงกลับเป็น SAFE ได้เสมอ (ยกเว้น label
# หรือคู่มือยังคงชนะถ้าดูเสี่ยงจริง — กันกรณีหายากที่ปุ่ม "Place Order" ถูกทำเป็น <a> ที่
# ตกแต่งด้วย CSS ให้ดูเหมือนปุ่ม)
ANCHOR_TAG = "a"

# W_search follow-up 2: บั๊กจริงอีกเคส — หลัง fill คำค้นหาลงในช่องค้นหาสำเร็จแล้ว label ของ
# ช่องนั้น (จาก perception.py::get_snapshot) จะกลายเป็น "ค่าที่พิมพ์ไปแล้ว" (เช่น
# "เพลงรัก") แทนที่จะเป็น placeholder/ชื่อช่องเดิม ("Search"/"ค้นหา") เพราะ label
# computation fallback ไป el.value เมื่อ innerText ว่าง — ถ้า LLM เผลอเลือก type="submit"
# ให้กับ action "กด Enter เพื่อค้นหา" (แทนที่จะเป็น "press_key" ธรรมดา) โดยเป้าหมายยังเป็น
# ช่องกรอกข้อความ/ค้นหาเดิม label ที่เห็นตอนนั้นจะเป็นคำค้นหาดิบๆ ไม่ match ทั้งสองฝั่งอีกเช่น
# กัน (เหมือน ANCHOR_TAG ด้านบนแต่คนละ element type) — <input type="text/search"> ธรรมดา
# (ไม่ใช่ type="submit"/"password"/"image" ที่แท้จริงอาจเป็นส่วนหนึ่งของฟอร์มอันตราย) ก็แทบ
# ไม่มีทางเป็นการ submit ฟอร์ม/ลบ/สั่งซื้อ/จ่ายเงินได้เองเช่นกัน (แค่กรอก/ค้นหา ย้อนกลับได้ง่าย)
SAFE_INPUT_TAG = "input"
RISKY_INPUT_TYPES = {"submit", "image", "password"}

# W7[B]: RAG-based permission — คู่มือที่ user ป้อน (ผ่าน ingestion ตั้งแต่ W3) อาจ
# กำหนดเองว่า action ไหนต้องขออนุมัติเพิ่มเติมจาก DEFAULT_NEEDS_CONFIRMATION/
# RISKY_LABEL_KEYWORDS ที่ hardcode ไว้ข้างบน (เช่น คู่มือเขียนว่า "การสั่งซื้อเกิน
# $100 ต้องขออนุมัติจากผู้จัดการก่อน") — ไม่ได้ให้ LLM ตัดสินเอง (พึ่ง model compliance
# ไม่ได้ ดูเหตุผลเดียวกับ RISKY_LABEL_KEYWORDS ด้านบน) แต่สแกนหาคำที่บ่งบอกว่าคู่มือ
# กำลังขอให้ขออนุมัติ/ยืนยันก่อนทำ เป็นชั้นสำรองระดับโค้ดเหมือนกัน
MANUAL_CONFIRMATION_KEYWORDS = {
    "ต้องขออนุมัติ", "ต้องได้รับอนุมัติ", "ต้องขอความยินยอม", "ต้องยืนยันก่อน",
    "requires approval", "needs approval", "require confirmation",
    "requires confirmation", "ask for confirmation", "confirm before", "ask before",
}


def _label_looks_risky(label: str) -> bool:
    lower = (label or "").lower()
    return any(keyword in lower for keyword in RISKY_LABEL_KEYWORDS)


def _label_looks_safe(label: str) -> bool:
    lower = (label or "").lower()
    return any(keyword in lower for keyword in SAFE_ACTION_LABEL_KEYWORDS)


def _manual_requires_confirmation(manual_guidance: str) -> bool:
    lower = (manual_guidance or "").lower()
    return any(keyword in lower for keyword in MANUAL_CONFIRMATION_KEYWORDS)


def normalize_domain(domain: str) -> str:
    """lowercase + ตัด "www." นำหน้าออก — ใช้กับ domain string ที่เป็น hostname ล้วนๆ
    อยู่แล้ว (เช่น path param ของ REST endpoint อย่าง /api/site-manual/{domain}/...) ต่าง
    จาก extract_domain() ด้านล่างที่รับ URL เต็มรูปแบบ (มี scheme) แล้ว parse หา netloc
    เอง — แยกออกมาเป็นฟังก์ชันกลางให้ทั้งคู่เรียกใช้ร่วมกัน กันไม่ให้กติกา "www. กับไม่มี
    www. ถือเป็นโดเมนเดียวกัน" ไป drift กันระหว่าง 2 จุด"""
    domain = domain.lower()
    if domain.startswith("www."):
        domain = domain[len("www."):]
    return domain


def extract_domain(url: str) -> str:
    """แยก domain ล้วนๆ ออกจาก URL (lowercase, ตัด port ออก, ตัด "www." นำหน้าออก) — ใช้
    ร่วมกันทุกจุดที่ต้องเทียบ/เก็บ domain ในระบบ (classify_action() เช็ค goto,
    core/user_browser.py จับคู่ tab ที่เปิดอยู่, site_learning/storage.py เก็บ/ค้นหา
    credential ต่อโดเมน ฯลฯ) กันไม่ให้ logic parse URL ซ้ำกันหลายที่ ถ้า URL ผิดรูปแบบ
    มากๆ คืนสตริงว่างเปล่า (ให้ผู้เรียกตัดสินใจเองว่าจะปฏิบัติยังไงกับ domain ว่าง แทนที่จะ
    throw ออกไป)

    ตัด "www." ออกโดยเจตนา (ไม่ใช่แค่ lowercase+ตัด port): ก่อนหน้านี้ "www.example.com"
    กับ "example.com" ถูกมองเป็นคนละ domain กันเป๊ะๆ ทำให้ credential ที่บันทึกไว้ตอน
    login bootstrap ผ่าน URL หนึ่ง (เช่น มี www.) หาไม่เจอตอนรัน task จริงที่เริ่มจาก URL
    อีกแบบ (ไม่มี www.) ของเว็บเดียวกัน — เห็นได้จาก storage.py::save_credentials/
    load_credentials ที่ key ด้วยค่าจากฟังก์ชันนี้ตรงๆ"""
    try:
        domain = urllib.parse.urlparse(url).netloc.lower()
        if ":" in domain:
            domain = domain.split(":")[0]
        return normalize_domain(domain)
    except Exception:
        return ""


def classify_action(
    cmd: dict, label: str = "", manual_guidance: str = "", allowed_domains: "set[str] | None" = None,
    element_tag: str = "", element_type: str = "",
) -> ActionRisk:
    """label (optional): ข้อความของ element ที่จะโดน action นี้ (จาก indexed elements
    ตอน perceive) — ใช้เช็คคำเสี่ยงเป็นชั้นสำรองนอกจาก type ล้วนๆ (ดู RISKY_LABEL_KEYWORDS)
    ไม่ส่งมาก็ได้ (default "") จะข้ามการเช็คชั้นนี้ไปเฉยๆ ไม่ throw

    manual_guidance (optional, W7[B]): เนื้อหาคู่มือที่เกี่ยวข้องกับ step นี้ (ตัวเดียว
    กับ manual_context ที่ orchestrator ดึงมาป้อน planner อยู่แล้วใน W6[B] — ไม่ยิง
    ChromaDB ซ้ำ) ใช้เช็คว่าคู่มือระบุไว้ไหมว่า action แบบนี้ต้องขออนุมัติก่อน ไม่ส่งมา
    ก็ได้ (default "") จะข้ามการเช็คชั้นนี้ไปเฉยๆ เหมือน label

    allowed_domains (optional): ชุดโดเมนที่อนุญาต override เฉพาะ call นี้ ไม่แตะ
    module-level ALLOWED_DOMAINS เลย — ไม่ส่งมา (None, default) = พฤติกรรมเดิมทุก
    ประการ (ใช้ ALLOWED_DOMAINS/BLOCKED_DOMAINS ของ module) ส่งมาเป็น set (แม้จะว่าง
    เปล่า) = ใช้ set นี้แทน ALLOWED_DOMAINS เดิมทั้งหมดสำหรับ call นี้เท่านั้น (เพจ semantic
    เดียวกับ ALLOWED_DOMAINS เดิม: ว่างเปล่า = ไม่จำกัด ไม่ใช่ deny-all) — ใช้ตอนต่อ agent
    เข้า browser จริงของ user (core/user_browser.py) ที่ต้องจำกัดแค่โดเมนของ task นั้นๆ
    โดยไม่กระทบ task/thread อื่นที่ใช้ classify_action() พร้อมกัน BLOCKED_DOMAINS
    (module-level) ยังคงเป็น hard block เสมอไม่ว่าจะ override หรือไม่

    element_tag (optional, W_search follow-up): ชื่อ HTML tag ของ element เป้าหมาย
    (เช่น "a", "button", "input") จาก indexed elements ตอน perceive — ใช้เป็นสัญญาณ
    โครงสร้างเพิ่มเติมนอกจาก label (ดู ANCHOR_TAG/SAFE_INPUT_TAG ด้านบน) ไม่ส่งมาก็ได้
    (default "") จะข้ามการเช็คชั้นนี้ไปเฉยๆ

    element_type (optional, W_search follow-up 2): ค่า attribute "type" ของ element
    เป้าหมาย (เช่น input ที่ type="text"/"search"/"submit"/"password") จาก indexed
    elements ตอน perceive — ใช้คู่กับ element_tag=="input" เพื่อแยกช่องกรอกข้อความ/
    ค้นหาธรรมดา (ปลอดภัย) ออกจาก input ที่แท้จริงอาจเสี่ยง (ดู RISKY_INPUT_TYPES ด้านบน)
    ไม่ส่งมาก็ได้ (default "")"""
    action_type = cmd.get("type", "")

    if action_type in DEFAULT_BLOCKED_ACTIONS:
        return ActionRisk.BLOCKED

    if action_type in DEFAULT_NEEDS_CONFIRMATION:
        # W_search: คู่มือ (ถ้ามี) ยังคงเป็นกฎที่ user ตั้งไว้เองโดยตรง ชนะเสมอไม่ว่า
        # label จะดูปลอดภัยแค่ไหน — เช็คก่อนอันดับแรก
        if _manual_requires_confirmation(manual_guidance):
            return ActionRisk.NEEDS_CONFIRMATION
        # label เองก็ match คำเสี่ยงด้วย (เช่น "Confirm and Search" ในหน้า checkout หรือ
        # "Place Order" ที่ทำเป็น <a> ตกแต่งด้วย CSS) ให้ฝั่งเสี่ยงชนะเสมอไม่ว่า label
        # หรือ tag จะดูปลอดภัยแค่ไหน — เช็คก่อนอันดับสอง (กันไว้ก่อนเสมอ)
        if _label_looks_risky(label):
            return ActionRisk.NEEDS_CONFIRMATION
        # label ที่ชัดเจนว่าเป็นแค่ค้นหา/เปิดดูเนื้อหา (ย้อนกลับได้ง่าย ไม่มีผลถาวร) ให้
        # ลดระดับกลับเป็น SAFE แม้ LLM จะเผลอเลือก action type ที่ดูเสี่ยงมาก็ตาม (ดู
        # SAFE_ACTION_LABEL_KEYWORDS ด้านบน)
        if _label_looks_safe(label):
            return ActionRisk.SAFE
        # W_search follow-up: label เป็นเนื้อหาอิสระ (เช่น ชื่อวิดีโอ/บทความ) ไม่ match
        # คำปลอดภัยหรือคำเสี่ยงเลย — เช็ค tag เป็นสัญญาณสุดท้าย: <a> ธรรมดาแทบไม่มีทางเป็น
        # การ submit/ลบ/สั่งซื้อ/จ่ายเงินจริง (ดู ANCHOR_TAG ด้านบน)
        if (element_tag or "").lower() == ANCHOR_TAG:
            return ActionRisk.SAFE
        # W_search follow-up 2: เช่นเดียวกัน — label อาจกลายเป็น "ค่าที่พิมพ์ไปแล้ว" ใน
        # ช่องกรอกข้อความ/ค้นหา (เช่น คำค้นหาดิบๆ) หลัง fill สำเร็จ ไม่ match ทั้งสองฝั่ง
        # เหมือนกัน — <input> ที่ไม่ใช่ type เสี่ยง (submit/image/password) ก็แทบไม่มีทาง
        # เป็นการ submit/ลบ/สั่งซื้อ/จ่ายเงินได้เองเช่นกัน (ดู SAFE_INPUT_TAG/
        # RISKY_INPUT_TYPES ด้านบน)
        if (element_tag or "").lower() == SAFE_INPUT_TAG and (element_type or "").lower() not in RISKY_INPUT_TYPES:
            return ActionRisk.SAFE
        return ActionRisk.NEEDS_CONFIRMATION

    if action_type == "goto":
        url = cmd.get("url", "")
        domain = extract_domain(url)

        if domain in BLOCKED_DOMAINS:
            return ActionRisk.BLOCKED

        effective_allowed = ALLOWED_DOMAINS if allowed_domains is None else allowed_domains
        if effective_allowed and domain not in effective_allowed:
            return ActionRisk.BLOCKED

        # goto ที่ผ่าน domain check แล้ว (ไม่ได้อยู่ blocklist) ยังต้องเช็คคู่มือต่อ
        # (เช่น คู่มือบอกว่าการไปหน้า admin ต้องขออนุมัติก่อน) — แต่ไม่ตกไปเช็ค label
        # ต่อด้านล่างเหมือนเดิม (label ปกติว่างเปล่าสำหรับ goto อยู่แล้วเพราะไม่มี
        # index ให้จับคู่ เอามาตัดสิน risk ของการ "ไปหน้าเว็บ" ไม่ได้)
        if _manual_requires_confirmation(manual_guidance):
            return ActionRisk.NEEDS_CONFIRMATION
        return ActionRisk.SAFE

    if _label_looks_risky(label) or _manual_requires_confirmation(manual_guidance):
        return ActionRisk.NEEDS_CONFIRMATION

    return ActionRisk.SAFE

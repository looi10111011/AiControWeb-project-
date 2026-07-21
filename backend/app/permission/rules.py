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


def _manual_requires_confirmation(manual_guidance: str) -> bool:
    lower = (manual_guidance or "").lower()
    return any(keyword in lower for keyword in MANUAL_CONFIRMATION_KEYWORDS)


def extract_domain(url: str) -> str:
    """แยก domain ล้วนๆ ออกจาก URL (lowercase, ตัด port ออก) — ใช้ร่วมกันทั้ง
    classify_action() (เช็ค goto) และ core/user_browser.py (จับคู่ tab ที่เปิดอยู่กับ
    target domain ตอนต่อเข้า browser จริงของ user) กันไม่ให้ logic parse URL ซ้ำกัน
    2 ที่ ถ้า URL ผิดรูปแบบมากๆ คืนสตริงว่างเปล่า (ให้ผู้เรียกตัดสินใจเองว่าจะปฏิบัติ
    ยังไงกับ domain ว่าง แทนที่จะ throw ออกไป)"""
    try:
        domain = urllib.parse.urlparse(url).netloc.lower()
        if ":" in domain:
            domain = domain.split(":")[0]
        return domain
    except Exception:
        return ""


def classify_action(
    cmd: dict, label: str = "", manual_guidance: str = "", allowed_domains: "set[str] | None" = None,
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
    (module-level) ยังคงเป็น hard block เสมอไม่ว่าจะ override หรือไม่"""
    action_type = cmd.get("type", "")

    if action_type in DEFAULT_BLOCKED_ACTIONS:
        return ActionRisk.BLOCKED

    if action_type in DEFAULT_NEEDS_CONFIRMATION:
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

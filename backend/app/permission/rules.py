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


def _label_looks_risky(label: str) -> bool:
    lower = (label or "").lower()
    return any(keyword in lower for keyword in RISKY_LABEL_KEYWORDS)


def classify_action(cmd: dict, label: str = "") -> ActionRisk:
    """label (optional): ข้อความของ element ที่จะโดน action นี้ (จาก indexed elements
    ตอน perceive) — ใช้เช็คคำเสี่ยงเป็นชั้นสำรองนอกจาก type ล้วนๆ (ดู RISKY_LABEL_KEYWORDS)
    ไม่ส่งมาก็ได้ (default "") จะข้ามการเช็คชั้นนี้ไปเฉยๆ ไม่ throw"""
    action_type = cmd.get("type", "")

    if action_type in DEFAULT_BLOCKED_ACTIONS:
        return ActionRisk.BLOCKED

    if action_type in DEFAULT_NEEDS_CONFIRMATION:
        return ActionRisk.NEEDS_CONFIRMATION

    if action_type == "goto":
        url = cmd.get("url", "")
        try:
            parsed_url = urllib.parse.urlparse(url)
            domain = parsed_url.netloc.lower()

            # ตัด port ออกถ้ามี
            if ":" in domain:
                domain = domain.split(":")[0]

            if domain in BLOCKED_DOMAINS:
                return ActionRisk.BLOCKED

            if ALLOWED_DOMAINS and domain not in ALLOWED_DOMAINS:
                return ActionRisk.BLOCKED
        except Exception:
            # ถ้าระบุ URL มาผิดรูปแบบมากๆ ให้บล็อกไปก่อนเพื่อความปลอดภัย
            return ActionRisk.BLOCKED

        # goto ที่ผ่าน domain check แล้ว (ไม่ได้อยู่ blocklist) ต้องถือว่า SAFE ทันที
        # ไม่ตกไปเช็ค label ต่อด้านล่าง — ระบบต้อง goto ไปหน้าเว็บก่อนเป็นอันดับแรก
        # ถึงจะเห็นฟอร์ม/element อะไรเลย จะเอา label (ที่ปกติว่างเปล่าสำหรับ goto
        # อยู่แล้วเพราะไม่มี index ให้จับคู่) มาตัดสิน risk ของการ "ไปหน้าเว็บ" ไม่ได้
        return ActionRisk.SAFE

    if _label_looks_risky(label):
        return ActionRisk.NEEDS_CONFIRMATION

    return ActionRisk.SAFE

"""Permission layer: allowlist / blocklist action + human-in-the-loop.

W1: skeleton only. W4-5: implement จริง (ดู roadmap.txt เฟส 1).
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


def classify_action(cmd: dict) -> ActionRisk:
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

    return ActionRisk.SAFE

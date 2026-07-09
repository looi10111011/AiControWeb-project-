"""Permission layer: allowlist / blocklist action + human-in-the-loop.

W1: skeleton only. W4-5: implement จริง (ดู roadmap.txt เฟส 1).
"""

from enum import Enum


class ActionRisk(str, Enum):
    SAFE = "safe"
    NEEDS_CONFIRMATION = "needs_confirmation"
    BLOCKED = "blocked"


# ตัวอย่าง action ที่ต้องขอยืนยันก่อนเสมอ (เช่น submit ฟอร์ม, ลบข้อมูล, ชำระเงิน)
DEFAULT_NEEDS_CONFIRMATION = {"submit", "delete", "purchase", "pay"}
DEFAULT_BLOCKED: set[str] = set()


def classify_action(action_name: str) -> ActionRisk:
    if action_name in DEFAULT_BLOCKED:
        return ActionRisk.BLOCKED
    if action_name in DEFAULT_NEEDS_CONFIRMATION:
        return ActionRisk.NEEDS_CONFIRMATION
    return ActionRisk.SAFE

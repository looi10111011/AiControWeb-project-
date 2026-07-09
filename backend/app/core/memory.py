"""Agent memory: short-term (กันทำซ้ำ) + long-term (pattern).

W1: skeleton only. W7: ทำจริง.
"""


class ShortTermMemory:
    def __init__(self):
        self._history: list[dict] = []

    def record(self, step: dict):
        self._history.append(step)

    def recent(self, n: int = 5) -> list[dict]:
        return self._history[-n:]


class LongTermMemory:
    """Nice-to-have — ตัดก่อนถ้าเวลาไม่พอ (ดู roadmap.txt ข้อควรระวัง #4)."""

    def __init__(self):
        self._patterns: list[dict] = []

"""Agent memory: short-term (กันทำซ้ำ) + long-term (pattern).

W1: skeleton only. W7[A]: short-term ทำจริงแล้ว (ดู failed_actions_summary()
ด้านล่าง) — long-term ยังเป็น skeleton เหมือนเดิม (nice-to-have, ตัดก่อนถ้าเวลาไม่พอ
ตามที่ตกลงกับ user ไว้)
"""


class ShortTermMemory:
    def __init__(self):
        self._history: list[dict] = []

    def record(self, step: dict):
        self._history.append(step)

    def recent(self, n: int = 5) -> list[dict]:
        return self._history[-n:]

    def failed_actions_summary(self, max_items: int = 5) -> str:
        """สรุป action ที่ล้มเหลว (success is False) ใน task ปัจจุบัน (ล่าสุด
        max_items รายการ) เป็น bullet list สั้นๆ — ใช้ป้อนกลับเข้า prompt ทุก step
        (ดู orchestrator.py) กัน LLM ลองซ้ำ action/แนวทางที่รู้อยู่แล้วว่าไม่เวิร์ค

        ต่างจาก loop-detection guard ใน orchestrator.py (ที่หยุด task ทันทีเมื่อ
        action เดิมเป๊ะๆ ซ้ำติดกันครบจำนวน) — ตัวนี้เป็นการเตือนเชิงรุกทุก step
        ตั้งแต่ล้มเหลวครั้งแรก ครอบคลุม action ที่ fail แต่ไม่ได้ซ้ำเป๊ะๆ ทุกฟิลด์ด้วย
        (เช่น fill index เดิมด้วยข้อความคนละแบบ) ซึ่ง loop-detection แยกไม่ออกว่าเป็น
        action ใหม่ — แม้ conversation history เดิมจะมีผลลัพธ์นี้อยู่แล้วในตัว แต่สรุป
        แบบเจาะจงช่วยให้โมเดลเล็ก (เช่น Gemini flash-lite) สังเกตเห็นชัดกว่าต้องไล่อ่าน
        history ทั้งหมดเอง (ดูรูปแบบ defense-in-depth เดียวกับ permission layer ที่
        เช็คทั้ง type และ label)

        record() ที่ไม่มี key "success" (เช่น ถ้าถูกเรียกโดยไม่ตั้งใจไม่ครบฟิลด์) จะไม่
        ถูกนับเป็น failure (ไม่ throw)
        """
        failed = [h for h in self._history if h.get("success") is False]
        if not failed:
            return ""
        lines = [f"- {h.get('cmd')} -> {h.get('result', '')}" for h in failed[-max_items:]]
        return "\n".join(lines)


class LongTermMemory:
    """Nice-to-have — ตัดก่อนถ้าเวลาไม่พอ (ดู roadmap.txt ข้อควรระวัง #4)."""

    def __init__(self):
        self._patterns: list[dict] = []

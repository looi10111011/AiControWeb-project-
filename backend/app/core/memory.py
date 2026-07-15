"""Agent memory: short-term (กันทำซ้ำ) + long-term (pattern).

W1: skeleton only. W7[A]: short-term ทำจริงแล้ว (ดู failed_actions_summary()
ด้านล่าง) — long-term ยังเป็น skeleton เหมือนเดิม (nice-to-have, ตัดก่อนถ้าเวลาไม่พอ
ตามที่ตกลงกับ user ไว้)
"""

from backend.app.core.actions import REJECTED_BY_USER_MESSAGE


class ShortTermMemory:
    def __init__(self):
        self._history: list[dict] = []

    def record(self, step: dict):
        self._history.append(step)

    def recent(self, n: int = 5) -> list[dict]:
        return self._history[-n:]

    def all(self) -> list[dict]:
        """คืน history ทั้งหมดของ task นี้ (สำเนา ไม่ใช่ reference ตรง) — ใช้โดย
        Gemini conversation compaction ใน orchestrator.py (W7[A]) เพื่อสร้าง digest
        ของ step ที่ถูกตัดออกจาก raw messages ไปแล้ว โดยไม่ต้อง parse raw Gemini
        Content object เอง (recent() อย่างเดียวไม่พอเพราะต้องเลือกช่วง step ตาม
        เกณฑ์ของตัวเอง ไม่ใช่แค่ n ตัวล่าสุด)"""
        return list(self._history)

    def failed_actions_summary(self, max_items: int = 5) -> str:
        """สรุป action ที่ล้มเหลว (success is False) ใน task ปัจจุบัน เป็น bullet
        list สั้นๆ — ใช้ป้อนกลับเข้า prompt ทุก step (ดู orchestrator.py) กัน LLM
        ลองซ้ำ action/แนวทางที่รู้อยู่แล้วว่าไม่เวิร์ค

        ต่างจาก loop-detection guard ใน orchestrator.py (ที่หยุด task ทันทีเมื่อ
        action เดิมเป๊ะๆ ซ้ำติดกันครบจำนวน) — ตัวนี้เป็นการเตือนเชิงรุกทุก step
        ตั้งแต่ล้มเหลวครั้งแรก ครอบคลุม action ที่ fail แต่ไม่ได้ซ้ำเป๊ะๆ ทุกฟิลด์ด้วย
        (เช่น fill index เดิมด้วยข้อความคนละแบบ) ซึ่ง loop-detection แยกไม่ออกว่าเป็น
        action ใหม่ — แม้ conversation history เดิมจะมีผลลัพธ์นี้อยู่แล้วในตัว แต่สรุป
        แบบเจาะจงช่วยให้โมเดลเล็ก (เช่น Gemini flash-lite) สังเกตเห็นชัดกว่าต้องไล่อ่าน
        history ทั้งหมดเอง (ดูรูปแบบ defense-in-depth เดียวกับ permission layer ที่
        เช็คทั้ง type และ label)

        (2026-07-15) Refusal memory: action ที่ถูกมนุษย์ปฏิเสธจริง (result มี
        REJECTED_BY_USER_MESSAGE จาก actions.py) มีความหมายต่างจาก failure ทั่วไป
        (timeout/index ผิด) — failure ทั่วไปเป็นแค่คำใบ้ให้ลองทางอื่น แต่การถูกปฏิเสธ
        คือคำสั่งห้ามเด็ดขาดไม่ให้ทำ action เดิมซ้ำอีกตลอด task นี้ (ดูกติกาใหม่ใน
        llm.py::SYSTEM_PROMPT) — ถ้าปล่อยให้ถูก evict ออกจาก max_items เหมือน failure
        ทั่วไป โมเดลอาจ "ลืม" แล้วย้อนกลับไปลองซ้ำ action ที่มนุษย์เพิ่งปฏิเสธไปแล้วอีก
        ครั้งหลังผ่านไปหลาย step (ทั้งดื้อดึงกดปุ่มเดิม/สร้างความรำคาญให้ user ถูกถามซ้ำ)
        — แยก rejected ออกมารวมทุกตัวเสมอ (ไม่ตัดทิ้งด้วย max_items, dedupe ตาม cmd กัน
        prompt บวมถ้าโมเดลยังฝ่าฝืนลองซ้ำแล้วโดนปฏิเสธซ้ำอีก) ส่วน failure อื่นที่ไม่ใช่
        การถูกปฏิเสธยังคงถูกจำกัดด้วย max_items ล่าสุดเหมือนเดิม

        record() ที่ไม่มี key "success" (เช่น ถ้าถูกเรียกโดยไม่ตั้งใจไม่ครบฟิลด์) จะไม่
        ถูกนับเป็น failure (ไม่ throw)
        """
        failed = [h for h in self._history if h.get("success") is False]
        if not failed:
            return ""

        rejected = [h for h in failed if REJECTED_BY_USER_MESSAGE in str(h.get("result", ""))]
        other = [h for h in failed if h not in rejected]

        deduped_rejected: list[dict] = []
        seen_cmds: list[dict] = []
        for h in rejected:
            if h.get("cmd") not in seen_cmds:
                seen_cmds.append(h.get("cmd"))
                deduped_rejected.append(h)

        lines = [f"- {h.get('cmd')} -> {h.get('result', '')}" for h in deduped_rejected]
        lines += [f"- {h.get('cmd')} -> {h.get('result', '')}" for h in other[-max_items:]]
        return "\n".join(lines)


class LongTermMemory:
    """Nice-to-have — ตัดก่อนถ้าเวลาไม่พอ (ดู roadmap.txt ข้อควรระวัง #4)."""

    def __init__(self):
        self._patterns: list[dict] = []

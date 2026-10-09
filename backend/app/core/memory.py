"""Agent short-term memory (ภายใน 1 task run) — W7[A]: สรุป action ที่ล้มเหลว/ล่าสุดป้อนกลับเข้า prompt
กันทำซ้ำ (long-term ข้าม task อยู่ที่ long_term_memory.py)
"""

from backend.app.core.actions import REJECTED_BY_USER_MESSAGE

# W_token_trim (P1/Q1): summary ที่ป้อนกลับทุก step เดิมฝัง result เต็ม (read_page_data ได้ถึง 60 แถว)
# ค้างในหน้าต่าง last-5 และ digest กินหลายพัน token/step — clip แบบหัว+ท้าย คง prefix ชนิด result
# และท้ายที่มักเป็น row-count ไว้ (result เต็มยังอยู่ใน raw tool_result ของ step ล่าสุด)
_RESULT_CLIP_HEAD = 200
_RESULT_CLIP_TAIL = 120


def clip_result(s: object) -> str:
    text = str(s or "")
    if len(text) <= _RESULT_CLIP_HEAD + _RESULT_CLIP_TAIL + 32:
        return text
    omitted = len(text) - _RESULT_CLIP_HEAD - _RESULT_CLIP_TAIL
    return f"{text[:_RESULT_CLIP_HEAD]} …[{omitted} chars omitted]… {text[-_RESULT_CLIP_TAIL:]}"


class ShortTermMemory:
    def __init__(self):
        self._history: list[dict] = []

    def record(self, step: dict):
        self._history.append(step)

    def recent(self, n: int = 5) -> list[dict]:
        return self._history[-n:]

    def all(self) -> list[dict]:
        """สำเนา history ทั้งหมด — ใช้สร้าง digest ตอน context compaction (W7[A])"""
        return list(self._history)

    def failed_actions_summary(self, max_items: int = 5) -> str:
        """bullet list ของ action ที่ success is False — เตือนเชิงรุกทุก step (ครอบคลุม action ที่ fail
        แต่ไม่ซ้ำเป๊ะ ซึ่ง loop-detection ใน orchestrator แยกไม่ออก) step ที่ไม่มี key "success" ไม่นับ

        (2026-07-15) Refusal memory: action ที่มนุษย์ปฏิเสธ (REJECTED_BY_USER_MESSAGE) คือคำสั่งห้ามทำซ้ำ
        ตลอด task — แสดงทุกตัวเสมอ (dedupe ตาม cmd, ไม่ตัดด้วย max_items) ไม่งั้นโมเดลลืมแล้วกดซ้ำ
        ส่วน failure อื่นจำกัดที่ max_items ล่าสุด
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

        lines = [f"- {h.get('cmd')} -> {clip_result(h.get('result', ''))}" for h in deduped_rejected]
        lines += [f"- {h.get('cmd')} -> {clip_result(h.get('result', ''))}" for h in other[-max_items:]]
        return "\n".join(lines)

    def recent_actions_summary(self, n: int = 5) -> str:
        """W32: bullet list ของ n action ล่าสุด (ทั้งสำเร็จและ fail) — ไม่ถูกตัดตอน context compaction
        (W7[A]/W22) และรวม action ที่สำเร็จด้วย ให้โมเดลเห็นว่ากำลังวนซ้ำ (เช่น กด "Next" สำเร็จ 3 ครั้ง)"""
        recent = self._history[-n:]
        if not recent:
            return ""
        return "\n".join(
            f"- step {h.get('step', '?')}: {h.get('cmd')} -> {clip_result(h.get('result', ''))}"
            for h in recent
        )

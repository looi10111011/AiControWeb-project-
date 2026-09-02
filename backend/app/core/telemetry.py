"""core/telemetry.py — W_eval_trace: ตัวเขียน log วิเคราะห์ 2 ไฟล์ (token_usage.jsonl และ
step_trace.jsonl) แยกออกมาจาก api/task_manager.py เพื่อให้ "ทุกเส้นทางที่เรียก
Orchestrator.run_task()" ใช้ร่วมกันได้ ไม่ใช่เฉพาะเส้นทาง HTTP API

ทำไมต้องย้ายลงมาที่ core/ (ไม่ใช่ให้ eval import ขึ้นไปหา api/): core/evaluation.py,
core/miniwob_eval.py, core/release_gate.py เป็น layer ล่างกว่า api/ — ถ้าให้มัน import
api/task_manager.py จะเป็น layer inversion และดึง TaskRecord (dataclass ที่ผูกกับ SSE/
approval/asyncio.Task ของฝั่ง HTTP ทั้งก้อน) เข้ามาโดยไม่จำเป็นเลย ทั้งที่ตัวเขียน log จริงๆ
ต้องการแค่ history + task_id + provider

บั๊กจริงที่ทำให้ต้องแยก (W82 baseline ครั้งแรก): release gate รัน 15 task แล้วรายงานได้แค่
"14/15 ผ่าน" โดยไม่มีข้อมูลเลยว่า task ไหนตกและตกที่ step ไหน — ยืนยันแล้วว่า trace rows
จาก gate run นั้น = 0 บรรทัด เพราะ _log_step_trace() ถูกเรียกจาก TaskManager._run() ที่เดียว
ส่วน evaluation.py/miniwob_eval.py เรียก run_task() ตรงๆ ไม่ผ่าน TaskManager จึงข้าม telemetry
ทั้งชุด (ปัญหาเดียวกับที่ W_step_trace ตั้งใจปิด แต่เส้นทาง eval เลี่ยงไป)

*** ห้ามฟังก์ชันในไฟล์นี้ throw ออกไปเด็ดขาด *** — ปัญหาการเก็บ log ไม่ควรมีทางทำให้ task ที่
ทำเสร็จไปแล้วจริงกลายเป็น error ย้อนหลัง (ผู้เรียกทุกตัวห่อ try/except ไว้อีกชั้นแล้ว แต่ตั้งใจ
ให้ปลอดภัยด้วยตัวเองด้วย)

เขียนแบบ sync ล้วนๆ (blocking file I/O) — ผู้เรียกต้องห่อ asyncio.to_thread() เสมอ กันบล็อก
event loop ตอน disk ช้า
"""

import json
import time
from pathlib import Path
from typing import Any, Optional

from backend.app.config import settings

# W_eval_trace: แยก "งานจริงของ user ที่ยิงผ่าน HTTP API" ออกจาก "การรัน benchmark" ในไฟล์
# เดียวกัน — จำเป็นเพราะ P4 (งานลด token/latency) วัดผลจาก token_usage.jsonl และถ้าแถวของ
# benchmark ปนเข้าไปโดยแยกไม่ออก ตัวเลข "ต้นทุนต่อ step ของงานจริง" จะเพี้ยนทันที
# (ไฟล์นี้ปนแถว fixture จาก unit test อยู่แล้วด้วย — มี field นี้แล้วกรองได้จริงทั้งสองแบบ)
SOURCE_API = "api"
SOURCE_EVAL = "eval"

_EMPTY_TOKENS = {"input": 0, "output": 0, "cache_read": 0, "cache_creation": 0}


def write_token_usage(
    *,
    task_id: str,
    url: str,
    goal: str,
    provider: Optional[str],
    result: Optional[dict],
    status: str,
    error: Optional[str],
    duration_seconds: float,
    source: str = SOURCE_API,
    run_id: Optional[str] = None,
) -> None:
    """W49: เขียน 1 บรรทัด JSON ต่อ task (append-only, JSON Lines)

    W_step_trace: บันทึก status สุดท้ายจริงด้วย — แยก "ล้มเหลวเพราะ agent ทำไม่สำเร็จ"
    (done + success=false) ออกจาก "พังกลางคัน" (error) และ "ผู้ใช้กดหยุด" (cancelled)
    ซึ่งเดิมไม่ปรากฏในไฟล์นี้เลยสักบรรทัด

    result เป็น None ได้ (เส้นทาง exception/cancelled ที่ run_task() ไม่ได้คืน dict) —
    บันทึก tokens เป็นศูนย์แต่ยังเก็บ status/error ไว้"""
    try:
        def _stat(key: str, default: Any) -> Any:
            val = result.get(key) if result else None
            return default if val is None else val

        entry: dict[str, Any] = {
            "timestamp": time.time(),
            "task_id": task_id,
            "url": url,
            "goal": goal,
            "provider": provider,
            "steps": result.get("steps") if result else 0,
            "success": result.get("success") if result else False,
            "duration_seconds": round(duration_seconds, 2),
            "tokens": (result.get("tokens") if result else None) or dict(_EMPTY_TOKENS),
            "status": status,
            "error": error,
            "source": source,
            # W_llm_call_count: กี่เทิร์นที่ยิงไปหา LLM จริง เทียบกับ steps ด้านบนแล้วเห็นทันที
            # ว่าโดนเผาไปกับเทิร์นที่ไม่ได้ลงมือทำอะไรกี่ครั้ง (ดู orchestrator.py::llm_turns)
            "llm_calls": _stat("llm_calls", 0),
            # W_token_cut W1: แยกว่าเทิร์น LLM ถูกใช้ไปกับอะไร — action_calls + finish_task_calls
            # + sum(guard_rejections) + notool_retries ควรเข้าใกล้ llm_calls (ส่วนต่างคือเทิร์น
            # อื่นที่ยังไม่ได้ tag) kpi.py สรุป median/p95 ของกลุ่มนี้
            "action_calls": _stat("action_calls", 0),
            "finish_task_calls": _stat("finish_task_calls", 0),
            "guard_rejections": _stat("guard_rejections", {}),
            # W_token_cut W3: guard_reason_counts = alias ของ guard_rejections (ชื่อที่ชัดกว่า);
            # repeated_guard_count = ผลรวม (count-1) ต่อเหตุผล = จำนวนเทิร์นที่ guard เตือน
            # เรื่องเดิมซ้ำ; finish_loop_prevented = จำนวนครั้งที่ W3 ตัดวงจร finish->reject->LLM
            "guard_reason_counts": _stat("guard_reason_counts", {}),
            "repeated_guard_count": _stat("repeated_guard_count", 0),
            "finish_loop_prevented": _stat("finish_loop_prevented", 0),
            # W_prompt_audit: 1 entry ต่อ LLM call — char count ของ request แยกตามหมวด
            # (system / tool_schema / page_snapshot / action_history / plan / tool_result /
            # user_message / gated_prompt / other) + _input_tokens/_cache_read ของ call นั้น
            "payload_audit": _stat("payload_audit", []),
            # W_token_cut W5: การยุบ user turn ของ step เก่า (assistant history compaction)
            "history_compaction_events": _stat("history_compaction_events", 0),
            "history_chars_saved": _stat("history_chars_saved", 0),
            "history_tokens_saved": _stat("history_tokens_saved", 0),
            "assistant_history_tokens": _stat("assistant_history_tokens", 0),
            "assistant_history_compacted_tokens": _stat("assistant_history_compacted_tokens", 0),
            # W_token_cut W7: บล็อกกฎที่ gate ใน turn เก่าถูกยุบเหลือ 1 บรรทัดอ้างอิง
            "gated_deref_events": _stat("gated_deref_events", 0),
            "gated_tokens_saved": _stat("gated_tokens_saved", 0),
            "notool_retries": _stat("notool_retries", 0),
            "cache_hit_turns": _stat("cache_hit_turns", 0),
            "cache_miss_turns": _stat("cache_miss_turns", 0),
            "avg_input_tokens_per_call": _stat("avg_input_tokens_per_call", 0),
            "avg_cached_tokens_per_call": _stat("avg_cached_tokens_per_call", 0),
            "avg_output_tokens_per_call": _stat("avg_output_tokens_per_call", 0),
        }
        if run_id:
            entry["run_id"] = run_id
        path = Path(settings.token_usage_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        return


def write_step_trace(
    history: Optional[list],
    *,
    task_id: str,
    provider: Optional[str],
    run_id: Optional[str] = None,
) -> None:
    """W_step_trace: เขียน 1 บรรทัดต่อ step ลง settings.step_trace_log_path (ดูเหตุผลเต็มที่
    config.py ตรงค่านั้น) — เขียนครั้งเดียวตอน task จบ ไม่ใช่ทุก step เพื่อไม่ให้ disk I/O
    แทรกกลาง agent loop

    history ว่าง/None (เช่น run_task() throw ตั้งแต่ยังไม่ได้ทำ step แรก) = ไม่เขียนอะไรเลย"""
    try:
        if not history:
            return
        path = Path(settings.step_trace_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            for step in history:
                if not isinstance(step, dict):
                    continue
                cmd = step.get("cmd") or {}
                line: dict[str, Any] = {
                    "timestamp": time.time(),
                    "task_id": task_id,
                    "provider": provider,
                    "step": step.get("step"),
                    "action_type": cmd.get("type"),
                    "index": cmd.get("index"),
                    "label": step.get("label"),
                    "success": step.get("success"),
                    "failure_class": step.get("failure_class"),
                    "timing": step.get("timing"),
                    "tokens": step.get("tokens"),
                    # ตัดข้อความยาวๆ ทิ้ง เก็บแค่พอให้ไล่ดูได้ว่าเกิดอะไร (ตารางเต็มๆ จาก
                    # read_page_data ยาวเป็นพันตัวอักษร ไม่มีประโยชน์ในไฟล์วิเคราะห์)
                    "result": str(step.get("result", ""))[:300],
                }
                if run_id:
                    line["run_id"] = run_id
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
    except Exception:
        return


def new_run_id(prefix: str) -> str:
    """W_eval_trace: id ที่ผูก "การรัน eval หนึ่งครั้ง" เข้ากับทุกบรรทัด trace/token ที่มันสร้าง

    จำเป็นเพราะ release_gate.save_summary() เกิด *หลัง* suite ทั้งหมดรันจบแล้ว และ timestamp
    ในชื่อไฟล์ JSON มาจาก build_summary() ตอนนั้น — ดึงย้อนหลังมาผูกกับ trace ที่เขียนไปก่อน
    หน้าไม่ได้เลย จึงต้องสร้าง id ตั้งแต่ต้นแล้วร้อยลงไปทุกชั้นแทน"""
    return f"{prefix}-{int(time.time() * 1000)}"

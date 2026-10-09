"""core/telemetry.py — W_eval_trace: ตัวเขียนเดียวของ token_usage.jsonl + step_trace.jsonl
อยู่ใน core/ (ไม่ใช่ api/task_manager.py) เพื่อให้ทุกเส้นทางที่เรียก run_task() ใช้ร่วมได้ — W82: gate
run แรกมี trace 0 บรรทัด เพราะ eval เรียก run_task() ตรงๆ ไม่ผ่าน TaskManager (และ eval import api/
= layer inversion)

*** ห้าม throw เด็ดขาด *** — ปัญหา log ต้องไม่ทำให้ task ที่เสร็จแล้วกลายเป็น error
blocking file I/O ล้วน — ผู้เรียกต้องห่อ asyncio.to_thread() เสมอ
"""

import json
import time
from pathlib import Path
from typing import Any, Optional

from backend.app.config import settings

# W_eval_trace: แยกงานจริง (HTTP API) ออกจาก benchmark ในไฟล์เดียวกัน — P4 วัดต้นทุนต่อ step
# จาก token_usage.jsonl ถ้าปนกันตัวเลขจะเพี้ยน
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
    """W49: append 1 บรรทัด JSON ต่อ task ลง settings.token_usage_log_path — ไม่ throw
    W_step_trace: status จริง (done/error/cancelled) แยก agent ทำไม่สำเร็จออกจากพัง/ผู้ใช้กดหยุด
    result=None ได้ (exception/cancelled) -> tokens เป็นศูนย์ แต่ยังเก็บ status/error"""
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
            # W_llm_call_count: เทียบกับ steps -> เทิร์นที่ยิง LLM แล้วไม่ได้ลงมือ (orchestrator::llm_turns)
            "llm_calls": _stat("llm_calls", 0),
            # W_token_cut W1: action_calls + finish_task_calls + sum(guard_rejections) +
            # notool_retries ควรเข้าใกล้ llm_calls (ส่วนต่าง = เทิร์นที่ยังไม่ได้ tag)
            # W_auto_login_outcome_is_invisible: "skipped"|"ok"|"failed" — จับคู่กับ
            # guard_rejections["login_skip"] ว่าเกิดตอน auto-login ล้มจริงหรือทั้งที่ล็อกอินแล้ว
            "auto_login": _stat("auto_login", "skipped"),
            # W_index_drift_measure: element ที่ index ชี้เปลี่ยนตัว/หายไประหว่าง snapshot กับ
            # dispatch — แยก "หน้า re-render" ออกจาก "โมเดลอ้าง index เก่า" (แก้คนละทาง)
            "index_drift_changed": _stat("index_drift_changed", 0),
            "index_drift_gone": _stat("index_drift_gone", 0),
            "action_calls": _stat("action_calls", 0),
            "finish_task_calls": _stat("finish_task_calls", 0),
            "guard_rejections": _stat("guard_rejections", {}),
            # W_token_cut W3: guard_reason_counts = alias ของ guard_rejections; repeated_guard_count
            # = sum(count-1) ต่อเหตุผล; finish_loop_prevented = ครั้งที่ตัดวงจร finish->reject->LLM
            "guard_reason_counts": _stat("guard_reason_counts", {}),
            "repeated_guard_count": _stat("repeated_guard_count", 0),
            "finish_loop_prevented": _stat("finish_loop_prevented", 0),
            # W_prompt_audit: 1 entry ต่อ LLM call — char count แยกหมวดของ request + tokens ของ call
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
        # T1/T3: ภาษา/intent ของ goal คำนวณตรงนี้ (pure function ของ goal) ให้แถวเส้นทาง exception
        # ได้ field ครบด้วย — ห่อ try แยกเพราะห้าม throw แม้จะเป็น pure function
        try:
            from backend.app.core import goal_intent as _goal_intent

            language = _goal_intent.detect_goal_language(goal)
            entry["goal_script"] = language["script"]
            entry["goal_chars"] = language["chars"]
            entry["goal_word_count"] = language["word_count"]
            entry.update(_goal_intent.canonical_intent(goal).as_telemetry())
        except Exception:
            pass
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
    """W_step_trace: 1 บรรทัดต่อ step ลง settings.step_trace_log_path เขียนครั้งเดียวตอน task จบ
    (disk I/O ไม่แทรกกลาง loop) — history ว่าง/None = ไม่เขียน; ไม่ throw"""
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
                    # ตัดให้สั้น — ตารางจาก read_page_data ยาวเป็นพันตัวอักษร ไม่มีประโยชน์ที่นี่
                    "result": str(step.get("result", ""))[:300],
                }
                if run_id:
                    line["run_id"] = run_id
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
    except Exception:
        return


def new_run_id(prefix: str) -> str:
    """W_eval_trace: id ผูกการรัน eval หนึ่งครั้งกับทุกบรรทัด trace/token — สร้างตั้งแต่ต้น
    เพราะ summary ของ release_gate เกิดหลัง trace ถูกเขียนไปแล้ว"""
    return f"{prefix}-{int(time.time() * 1000)}"

"""core/kpi.py — W_production_kpi: สรุป telemetry ของ *งานจริง* ให้ตอบได้ว่า "ดีขึ้นจริงไหม"

ทำไมต้องมี: telemetry เขียนครบมาตั้งแต่ W78/W83 (`data/token_usage.jsonl` 1 บรรทัดต่อ task,
`data/step_trace.jsonl` 1 บรรทัดต่อ step พร้อม failure_class/timing) แต่ **ไม่มีอะไรอ่านมันเพื่อ
สรุปผลเลย** — `grep` หาผู้ใช้ path ทั้งสองเจอแต่ตัวเขียนกับเทสต์ ส่วน release_gate อ่านเฉพาะ
`data/eval_results/*.json` จึงตอบได้แค่ "benchmark ดีขึ้นไหม" ไม่ใช่ "งานจริงของ user ดีขึ้นไหม"

หลักการที่ยกมาจาก W91 โดยตรง (gate รันซ้ำบน commit เดิมยังแกว่งเกินเกณฑ์ 10% ด้วยตัวมันเอง):
**ห้ามโชว์ค่าเฉลี่ยเดี่ยวๆ แล้วสรุปว่าดีขึ้น** ทุกตัวเลขต้องมาพร้อม median + p95 + จำนวนตัวอย่าง
เพื่อให้คนอ่านตัดสินเองได้ว่าความต่างที่เห็นใหญ่กว่า noise หรือยัง

อ่านอย่างเดียว ไม่แตะ agent loop และไม่ throw ออกไปทำให้ผู้เรียกพัง (ไฟล์ยังไม่มี/บรรทัดเสีย =
ข้ามเงียบๆ เหมือนหลักการของ telemetry.py เอง)
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

from backend.app.config import settings

SECONDS_PER_DAY = 86400

# W_production_kpi: แถวที่ไม่มี field "source" คือของเก่าก่อน W83 (ตอนนั้นยังไม่มีการแยก
# งานจริงออกจาก benchmark) — นับแยกไว้ให้เห็น ไม่เอามาปนกับสถิติของงานจริง เพราะเราไม่รู้ว่า
# มันคืออะไร การเดาแล้วเอามารวมจะทำให้ตัวเลขดูเยอะขึ้นโดยไม่มีความหมาย
_SOURCE_UNKNOWN = "(legacy — ก่อนมี field source)"

# W_production_kpi: `source="api"` อย่างเดียวแยกงานจริงออกจาก fixture ของเทสต์ไม่ได้ — วัดจริง
# แล้วพบว่า 564 จาก 642 แถวที่ source=api ชี้ไป example.com (เทสต์รุ่นเก่าที่เขียนลงไฟล์จริง)
# ทำให้รายงานออกมาเป็น success 99% / duration 0.0s / 10 tokens ซึ่งไม่ใช่ความจริงของงานจริงเลย
#
# ใช้กฎที่ไม่ต้องเดา: RFC 2606 + RFC 6761 สงวนโดเมนกลุ่มนี้ไว้สำหรับเอกสาร/ทดสอบโดยเฉพาะ
# "ไม่มีงานจริงของ user ที่ไหนวิ่งไปโดเมนพวกนี้ได้" — ต่างจากการเดาว่า "แถวที่ duration=0
# น่าจะเป็นของปลอม" ซึ่งจะไปตัดงานจริงที่จบเร็วทิ้งด้วย
_RESERVED_TEST_HOST_SUFFIXES = (
    "example.com", "example.net", "example.org",
    ".test", ".example", ".invalid", ".localhost",
)


def _is_reserved_test_url(url: str) -> bool:
    host = (urlparse(str(url or "")).hostname or "").lower()
    if not host:
        return False
    return any(
        host == suffix or host.endswith("." + suffix) if not suffix.startswith(".")
        else host.endswith(suffix)
        for suffix in _RESERVED_TEST_HOST_SUFFIXES
    )


def _read_jsonl(path: str) -> list[dict]:
    """อ่าน JSONL แบบทนพัง — ไฟล์ไม่มี/บรรทัดเสีย ข้ามเงียบๆ ไม่ throw"""
    rows: list[dict] = []
    try:
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    parsed = json.loads(line)
                except Exception:
                    continue
                if isinstance(parsed, dict):
                    rows.append(parsed)
    except Exception:
        return rows
    return rows


def _percentile(values: list[float], fraction: float) -> Optional[float]:
    """p50/p95 แบบ nearest-rank — ตั้งใจไม่ interpolate เพราะกลุ่มตัวอย่างเล็ก (หลักสิบ)
    การ interpolate จะสร้างตัวเลขที่ไม่เคยเกิดขึ้นจริงสักครั้ง"""
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def _stat_block(values: list[float]) -> dict[str, Any]:
    """ทุกตัวเลขมาพร้อม n เสมอ — n น้อยแปลว่ายังสรุปอะไรไม่ได้ ต้องเห็นคู่กันตลอด"""
    clean = [v for v in values if isinstance(v, (int, float))]
    return {
        "n": len(clean),
        "median": _percentile(clean, 0.5),
        "p95": _percentile(clean, 0.95),
    }


def summarise_tasks(rows: list[dict]) -> dict[str, Any]:
    """สรุปแถวของ token_usage.jsonl ที่กรองมาแล้ว

    W_production_kpi: แยก "task ที่ใช้เบราว์เซอร์จริง" ออกจาก "task ที่ตอบแบบแชท" เสมอ —
    routes.py มีทางลัด 4 ทางที่คืนผลโดยไม่แตะเบราว์เซอร์เลย (คำสั่ง /context, ไฟล์แนบ,
    general chat, follow-up จากไฟล์ที่จำไว้) ทุกทางคืน steps=0 ตามออกแบบ
    ถ้าเอามารวมกัน median ของ steps/duration จะกลายเป็น 0 ทันทีที่ traffic ส่วนใหญ่เป็นแชท
    (วัดจริงแล้วเป็นแบบนั้น) ซึ่งอ่านแล้วเข้าใจผิดว่า agent ทำงานเสร็จใน 0 step"""
    if not rows:
        return {"n": 0}
    done = [r for r in rows if r.get("status") == "done"]
    successes = [r for r in done if r.get("success") is True]
    browser = [r for r in rows if (r.get("steps") or 0) > 0]
    browser_done = [r for r in browser if r.get("status") == "done"]
    browser_success = [r for r in browser_done if r.get("success") is True]

    # W_token_cut W1: llm_calls vs steps — ส่วนต่างคือเทิร์นที่ยิง LLM แล้วไม่ได้ลงมือทำ
    # (guard ปฏิเสธ / finish_task ที่ถูกตีกลับ / no-tool retry) ตัวเลขที่ W3 ใช้เล็งเป้า
    def _guard_total(r: dict) -> int:
        gr = r.get("guard_rejections") or {}
        return sum(v for v in gr.values() if isinstance(v, (int, float)))

    guard_by_name: Counter = Counter()
    for r in browser:
        for name, count in (r.get("guard_rejections") or {}).items():
            if isinstance(count, (int, float)):
                guard_by_name[name] += count
    cache_hits = sum(r.get("cache_hit_turns") or 0 for r in browser)
    cache_misses = sum(r.get("cache_miss_turns") or 0 for r in browser)
    finish_loops_prevented = sum(r.get("finish_loop_prevented") or 0 for r in browser)  # W3
    history_events = sum(r.get("history_compaction_events") or 0 for r in browser)  # W5
    history_tokens_saved = sum(r.get("history_tokens_saved") or 0 for r in browser)  # W5
    gated_deref_events = sum(r.get("gated_deref_events") or 0 for r in browser)  # W7
    gated_tokens_saved = sum(r.get("gated_tokens_saved") or 0 for r in browser)  # W7
    return {
        "n": len(rows),
        "n_done": len(done),
        "success_rate": (len(successes) / len(done)) if done else None,
        "n_browser": len(browser),
        "n_chat_shaped": len(rows) - len(browser),
        "browser_success_rate": (
            (len(browser_success) / len(browser_done)) if browser_done else None
        ),
        "status": dict(Counter(str(r.get("status")) for r in rows).most_common()),
        # สถิติด้านล่างคิดจาก task ที่ใช้เบราว์เซอร์เท่านั้น (ดู docstring)
        "steps": _stat_block([r.get("steps") for r in browser]),
        "duration_seconds": _stat_block([r.get("duration_seconds") for r in browser]),
        "input_tokens": _stat_block([(r.get("tokens") or {}).get("input") for r in browser]),
        # W_token_cut W1
        "llm_calls": _stat_block([r.get("llm_calls") for r in browser]),
        "action_calls": _stat_block([r.get("action_calls") for r in browser]),
        "guard_rejections_total": _stat_block([_guard_total(r) for r in browser]),
        "repeated_guard_count": _stat_block([r.get("repeated_guard_count") for r in browser]),
        "finish_loops_prevented": finish_loops_prevented,  # W_token_cut W3
        "history_compaction_events": history_events,  # W_token_cut W5
        "history_tokens_saved": history_tokens_saved,  # W_token_cut W5
        "gated_deref_events": gated_deref_events,  # W_token_cut W7
        "gated_tokens_saved": gated_tokens_saved,  # W_token_cut W7
        "assistant_history_tokens": _stat_block(
            [r.get("assistant_history_tokens") for r in browser]
        ),
        "notool_retries": _stat_block([r.get("notool_retries") for r in browser]),
        "avg_input_tokens_per_call": _stat_block(
            [r.get("avg_input_tokens_per_call") for r in browser]
        ),
        "avg_cached_tokens_per_call": _stat_block(
            [r.get("avg_cached_tokens_per_call") for r in browser]
        ),
        "avg_output_tokens_per_call": _stat_block(
            [r.get("avg_output_tokens_per_call") for r in browser]
        ),
        "cache_hit_ratio": (
            cache_hits / (cache_hits + cache_misses)
            if (cache_hits + cache_misses) else None
        ),
        "guard_rejections_by_name": dict(guard_by_name.most_common()),
    }


def summarise_steps(rows: list[dict]) -> dict[str, Any]:
    """สรุปแถวของ step_trace.jsonl — failure_class คือของที่ตอบได้ว่า "ล้มเพราะอะไร"
    ซึ่ง token_usage บอกไม่ได้ และ timing แยก phase ตอบว่าเวลาหมดไปกับอะไร"""
    if not rows:
        return {"n": 0}
    phases = ("snapshot", "llm", "action", "pacing", "wait")
    totals = {phase: 0.0 for phase in phases}
    for row in rows:
        timing = row.get("timing") or {}
        for phase in phases:
            value = timing.get(phase)
            if isinstance(value, (int, float)):
                totals[phase] += value
    return {
        "n": len(rows),
        "failure_class": dict(
            Counter(str(r.get("failure_class") or "(none)") for r in rows).most_common()
        ),
        "seconds_by_phase": {k: round(v, 1) for k, v in totals.items()},
    }


def build_kpi_report(
    *,
    token_usage_path: Optional[str] = None,
    step_trace_path: Optional[str] = None,
    source: str = "api",
    window_days: int = 7,
    now: Optional[float] = None,
) -> dict[str, Any]:
    """รายงาน KPI ของงานจริง + เทียบกับช่วงก่อนหน้าที่ยาวเท่ากัน

    source="api" คือค่าเริ่มต้นโดยเจตนา — คำถามที่ต้องตอบคือ "งานจริงของ user ดีขึ้นไหม"
    ไม่ใช่ผลของ benchmark (ซึ่ง release_gate ตอบอยู่แล้ว)
    """
    now = time.time() if now is None else now
    usage = _read_jsonl(token_usage_path or settings.token_usage_log_path)
    steps = _read_jsonl(step_trace_path or settings.step_trace_log_path)

    source_counts = Counter(str(r.get("source") or _SOURCE_UNKNOWN) for r in usage)
    in_source = [r for r in usage if r.get("source") == source]
    scoped = [r for r in in_source if not _is_reserved_test_url(r.get("url"))]
    excluded_reserved = len(in_source) - len(scoped)

    recent_cutoff = now - window_days * SECONDS_PER_DAY
    previous_cutoff = now - 2 * window_days * SECONDS_PER_DAY

    def _in(rows: list[dict], low: float, high: float) -> list[dict]:
        return [r for r in rows if low <= float(r.get("timestamp") or 0) < high]

    recent = _in(scoped, recent_cutoff, now + 1)
    previous = _in(scoped, previous_cutoff, recent_cutoff)
    recent_run_ids = {r.get("run_id") for r in recent if r.get("run_id")}

    return {
        "source": source,
        "window_days": window_days,
        "rows_by_source": dict(source_counts.most_common()),
        # โชว์ให้เห็นเสมอว่าตัดอะไรออกไปเท่าไร — การกรองเงียบๆ ทำให้คนอ่านเชื่อตัวเลขผิด
        "excluded_reserved_test_urls": excluded_reserved,
        "all_time": summarise_tasks(scoped),
        "recent": summarise_tasks(recent),
        "previous": summarise_tasks(previous),
        "recent_steps": summarise_steps(
            [r for r in steps if r.get("run_id") in recent_run_ids] if recent_run_ids else []
        ),
        "all_steps": summarise_steps(steps),
    }


def _fmt(value: Optional[float], digits: int = 1) -> str:
    return "-" if value is None else f"{value:,.{digits}f}"


def format_kpi_report(report: dict[str, Any]) -> str:
    """ข้อความสำหรับ terminal — จงใจโชว์ n ติดกับทุกตัวเลข และไม่สรุปให้ว่า "ดีขึ้น/แย่ลง"
    เพราะ noise ของ metric พวกนี้สูงพอที่การสรุปอัตโนมัติจะหลอกคนอ่านได้ (ดู W91)"""
    lines = [
        f"=== KPI ของงานจริง (source={report['source']}) ===",
        "แถวทั้งหมดในไฟล์แยกตาม source: "
        + ", ".join(f"{k}={v}" for k, v in report["rows_by_source"].items()),
        f"ตัดออกเพราะเป็นโดเมนที่สงวนไว้ทดสอบ (RFC 2606/6761): "
        f"{report['excluded_reserved_test_urls']} แถว",
        "",
    ]
    for title, key in (
        ("ทั้งหมดเท่าที่มี", "all_time"),
        (f"{report['window_days']} วันล่าสุด", "recent"),
        (f"{report['window_days']} วันก่อนหน้านั้น", "previous"),
    ):
        block = report[key]
        if not block.get("n"):
            lines.append(f"{title}: (ไม่มีข้อมูล)")
            continue
        rate = block.get("browser_success_rate")
        lines.append(
            f"{title}: {block['n']} task "
            f"(ใช้เบราว์เซอร์ {block.get('n_browser', 0)} / ตอบแบบแชท "
            f"{block.get('n_chat_shaped', 0)}) — สำเร็จเฉพาะงานเบราว์เซอร์ "
            f"{'-' if rate is None else f'{rate:.0%}'}"
        )
        for label, metric in (("steps", "steps"), ("วินาที", "duration_seconds"),
                              ("input tokens", "input_tokens"),
                              ("llm_calls", "llm_calls"), ("action_calls", "action_calls"),
                              ("guard rejections", "guard_rejections_total"),
                              ("repeated guard", "repeated_guard_count"),
                              ("notool_retries", "notool_retries"),
                              ("in tok/call", "avg_input_tokens_per_call"),
                              ("cached tok/call", "avg_cached_tokens_per_call"),
                              ("out tok/call", "avg_output_tokens_per_call")):
            # ทุกตัวคิดจากงานเบราว์เซอร์เท่านั้น
            stat = block.get(metric) or {"median": None, "p95": None, "n": 0}
            lines.append(
                f"    {label:<16} median={_fmt(stat['median'])}  p95={_fmt(stat['p95'])}  n={stat['n']}"
            )
        ratio = block.get("cache_hit_ratio")
        lines.append(
            f"    cache hit ratio (เทิร์นที่ cache ติด / เทิร์นทั้งหมด): "
            f"{'-' if ratio is None else f'{ratio:.0%}'}"
        )
        by_name = block.get("guard_rejections_by_name") or {}
        if by_name:
            lines.append(f"    guard rejections แยกตามชื่อ: {by_name}")
        flp = block.get("finish_loops_prevented") or 0
        if flp:
            lines.append(f"    finish->reject->LLM loops ที่ W3 ตัดออก: {flp}")
        hce = block.get("history_compaction_events") or 0
        if hce or block.get("history_tokens_saved"):
            ah = block.get("assistant_history_tokens") or {}
            lines.append(
                f"    W5 history compaction: {hce} ครั้ง, ~{block.get('history_tokens_saved', 0):,} "
                f"tok ตัดออก · assistant history ที่เหลือ median={_fmt(ah.get('median'))} "
                f"p95={_fmt(ah.get('p95'))}"
            )
        gde = block.get("gated_deref_events") or 0
        if gde or block.get("gated_tokens_saved"):
            lines.append(
                f"    W7 gated-rule deref: {gde} ครั้ง, ~{block.get('gated_tokens_saved', 0):,} tok ตัดออก"
            )
        lines.append(f"    status: {block['status']}")
    steps_block = report["recent_steps"] if report["recent_steps"].get("n") else report["all_steps"]
    if steps_block.get("n"):
        scope = "ช่วงล่าสุด" if report["recent_steps"].get("n") else "ทั้งหมด"
        lines += [
            "",
            f"step trace ({scope}, {steps_block['n']} step):",
            f"    failure_class: {steps_block['failure_class']}",
            f"    เวลารวมแยก phase (วินาที): {steps_block['seconds_by_phase']}",
        ]
    lines += [
        "",
        "อ่านอย่างระวัง: n น้อย = ยังสรุปไม่ได้ และ metric กลุ่มนี้แกว่งเองสูง",
        "(W91: gate รันซ้ำบน commit เดิมยังต่างกันเกิน 10%) — ดู median/p95 คู่กับ n เสมอ",
    ]
    return "\n".join(lines)

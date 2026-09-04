"""W_production_kpi (W109) — สรุป telemetry ของงานจริง

เทสต์กลุ่มนี้เขียน JSONL ปลอมลง tmp_path แล้วอ่านกลับ ไม่แตะไฟล์จริงของโปรเจกต์เด็ดขาด
(บทเรียนตรงจากบั๊กที่ KPI ตัวนี้เจอเอง: เทสต์รุ่นเก่าเคยเขียนลง data/token_usage.jsonl จริง
จน 564 จาก 642 แถวเป็น fixture ปลอม)
"""

import json

from backend.app.core.kpi import (
    _is_reserved_test_url,
    build_kpi_report,
    format_kpi_report,
    summarise_steps,
    summarise_tasks,
)

DAY = 86400
NOW = 1_800_000_000.0


def _write(path, rows):
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8",
    )
    return str(path)


def _task(**kw):
    row = {
        "timestamp": NOW - DAY, "task_id": "t", "url": "https://real.example-site.io/a",
        "goal": "g", "provider": "openai", "steps": 5, "success": True,
        "duration_seconds": 10.0, "tokens": {"input": 1000}, "status": "done",
        "source": "api", "run_id": "r1",
    }
    row.update(kw)
    return row


def test_reserved_test_urls_are_recognised():
    """`source="api"` อย่างเดียวแยกงานจริงออกจาก fixture ไม่ได้ — วัดจริงแล้ว 564/642 แถวชี้ไป
    example.com ทำให้รายงานออกมา success 99% / duration 0.0s ซึ่งไม่จริงเลย

    ใช้กฎที่ไม่ต้องเดา: RFC 2606/6761 สงวนโดเมนกลุ่มนี้ไว้สำหรับเอกสาร/ทดสอบ — ต่างจากการเดาว่า
    "แถวที่ duration=0 น่าจะปลอม" ซึ่งจะไปตัดงานจริงที่จบเร็วทิ้งด้วย"""
    assert _is_reserved_test_url("https://example.com/x") is True
    assert _is_reserved_test_url("http://app.example.com") is True
    assert _is_reserved_test_url("https://foo.test/") is True
    assert _is_reserved_test_url("https://x.invalid/") is True

    assert _is_reserved_test_url("https://opensource-demo.orangehrmlive.com/a") is False
    assert _is_reserved_test_url("https://www.saucedemo.com/") is False
    # ชื่อโดเมนจริงที่บังเอิญมีคำว่า example อยู่ข้างใน ต้องไม่โดนตัด
    assert _is_reserved_test_url("https://real.example-site.io/a") is False
    assert _is_reserved_test_url("") is False


def test_report_excludes_reserved_urls_and_says_how_many():
    """การกรองเงียบๆ ทำให้คนอ่านเชื่อตัวเลขผิด — ต้องบอกเสมอว่าตัดออกไปกี่แถว"""
    usage = _write(
        __import__("pathlib").Path(_tmp()) / "u.jsonl",
        [_task(), _task(url="https://example.com/x", steps=1)],
    )
    report = build_kpi_report(
        token_usage_path=usage, step_trace_path=usage + ".missing", now=NOW,
    )

    assert report["excluded_reserved_test_urls"] == 1
    assert report["all_time"]["n"] == 1


def test_browser_tasks_are_separated_from_chat_shaped_ones():
    """routes.py มีทางลัด 4 ทางที่คืนผลโดยไม่แตะเบราว์เซอร์เลย (/context, ไฟล์แนบ, general
    chat, follow-up จากไฟล์) ทุกทางคืน steps=0 ตามออกแบบ — ถ้าเอามารวม median ของ
    steps/duration จะกลายเป็น 0 ทันทีที่ traffic ส่วนใหญ่เป็นแชท (วัดจริงแล้วเป็นแบบนั้น)
    ซึ่งอ่านแล้วเข้าใจผิดว่า agent ทำงานเสร็จใน 0 step"""
    rows = [
        _task(steps=0, duration_seconds=0.1, tokens={"input": 10}),
        _task(steps=0, duration_seconds=0.1, tokens={"input": 10}),
        _task(steps=4, duration_seconds=60.0, tokens={"input": 50_000}),
    ]
    summary = summarise_tasks(rows)

    assert summary["n"] == 3
    assert summary["n_browser"] == 1
    assert summary["n_chat_shaped"] == 2
    # สถิติต้องคิดจากงานเบราว์เซอร์เท่านั้น ไม่ถูกแชทลากลงเป็น 0
    assert summary["steps"]["median"] == 4
    assert summary["duration_seconds"]["median"] == 60.0
    assert summary["steps"]["n"] == 1


def test_w1_call_breakdown_is_summarised_with_median_p95_and_by_name():
    """W_token_cut W1: llm_calls / guard_rejections / cache hit ratio ต้องสรุปแบบ
    median+p95+n และ guard_rejections รวมข้ามทุก task แยกตามชื่อ guard"""
    rows = [
        _task(steps=3, llm_calls=10, action_calls=3, notool_retries=0,
              cache_hit_turns=2, cache_miss_turns=8, repeated_guard_count=1, finish_loop_prevented=1,
              avg_input_tokens_per_call=6000.0, avg_output_tokens_per_call=30.0,
              guard_rejections={"filter_scope": 2, "premature_true_finish": 1}),
        _task(steps=4, llm_calls=20, action_calls=4, notool_retries=1,
              cache_hit_turns=6, cache_miss_turns=14, repeated_guard_count=2, finish_loop_prevented=0,
              avg_input_tokens_per_call=6200.0, avg_output_tokens_per_call=35.0,
              guard_rejections={"filter_scope": 3}),
        _task(steps=0, duration_seconds=0.1),  # chat-shaped — ต้องไม่ถูกนับ
    ]
    summary = summarise_tasks(rows)
    assert summary["finish_loops_prevented"] == 1
    assert summary["repeated_guard_count"]["n"] == 2

    assert summary["llm_calls"]["n"] == 2
    # nearest-rank ไม่ interpolate: median ของ 2 ค่าคือค่าล่าง (ดู _percentile)
    assert summary["llm_calls"]["median"] == 10
    assert summary["llm_calls"]["p95"] == 20
    assert summary["guard_rejections_total"]["median"] == 3  # 3 กับ 3
    assert summary["guard_rejections_by_name"] == {"filter_scope": 5, "premature_true_finish": 1}
    # cache hit ratio = (2+6) / (2+6+8+14)
    assert summary["cache_hit_ratio"] == 8 / 30
    assert summary["notool_retries"]["p95"] == 1


def test_success_rate_counts_only_finished_tasks():
    """task ที่ถูก cancel/error ไม่ควรนับเป็น "ล้มเหลว" ของ agent — มันไม่เคยได้ทำจนจบ"""
    rows = [
        _task(steps=3, success=True, status="done"),
        _task(steps=3, success=False, status="done"),
        _task(steps=3, success=False, status="cancelled"),
    ]
    summary = summarise_tasks(rows)

    assert summary["n_done"] == 2
    assert summary["browser_success_rate"] == 0.5
    assert summary["status"]["cancelled"] == 1


def test_step_summary_reports_failure_classes_and_phase_time():
    """failure_class คือสิ่งเดียวที่ตอบได้ว่า "ล้มเพราะอะไร" ซึ่ง token_usage บอกไม่ได้"""
    rows = [
        {"failure_class": "ok", "timing": {"llm": 1.0, "action": 2.0}},
        {"failure_class": "element_not_found", "timing": {"llm": 3.0, "wait": 0.5}},
        {"timing": {}},
    ]
    summary = summarise_steps(rows)

    assert summary["n"] == 3
    assert summary["failure_class"]["element_not_found"] == 1
    assert summary["failure_class"]["(none)"] == 1
    assert summary["seconds_by_phase"]["llm"] == 4.0
    assert summary["seconds_by_phase"]["action"] == 2.0


def test_window_comparison_splits_recent_from_previous():
    """คำถามคือ "ดีขึ้นไหม" จึงต้องเทียบสองช่วงที่ยาวเท่ากัน ไม่ใช่ดูค่ารวมค่าเดียว"""
    from pathlib import Path

    usage = _write(
        Path(_tmp()) / "u2.jsonl",
        [
            _task(timestamp=NOW - 2 * DAY, run_id="recent"),
            _task(timestamp=NOW - 10 * DAY, run_id="old"),
        ],
    )
    report = build_kpi_report(
        token_usage_path=usage, step_trace_path=usage + ".missing", window_days=7, now=NOW,
    )

    assert report["recent"]["n"] == 1
    assert report["previous"]["n"] == 1


def test_missing_files_do_not_raise():
    """อ่านอย่างเดียวและห้าม throw — หลักการเดียวกับ telemetry.py ที่เขียนมัน"""
    report = build_kpi_report(
        token_usage_path="/definitely/missing.jsonl",
        step_trace_path="/definitely/missing2.jsonl",
        now=NOW,
    )

    assert report["all_time"] == {"n": 0}
    assert "KPI" in format_kpi_report(report)


_TMP = []


def _tmp():
    import tempfile

    if not _TMP:
        _TMP.append(tempfile.mkdtemp())
    return _TMP[0]


def test_report_splits_by_goal_script(tmp_path):
    """T1: คำถาม "งานภาษาไทยสำเร็จ/แพงต่างจากภาษาอังกฤษไหม" ต้องอ่านออกจากรายงานได้ตรงๆ
    แถวเก่าที่ยังไม่มี goal_script ต้องไปอยู่กลุ่ม unknown ตามความจริง ไม่เดาย้อนหลังให้"""
    usage = tmp_path / "token_usage.jsonl"
    rows = [
        {"timestamp": 1000, "source": "api", "url": "https://real.example-site.io/", "status": "done",
         "success": True, "steps": 3, "goal_script": "thai", "tokens": {"input": 90000}},
        {"timestamp": 1001, "source": "api", "url": "https://real.example-site.io/", "status": "done",
         "success": True, "steps": 2, "goal_script": "latin", "tokens": {"input": 30000}},
        {"timestamp": 1002, "source": "api", "url": "https://real.example-site.io/", "status": "done",
         "success": False, "steps": 4, "tokens": {"input": 50000}},
    ]
    usage.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    report = build_kpi_report(
        token_usage_path=str(usage), step_trace_path=str(tmp_path / "missing.jsonl"), now=2000,
    )

    by_script = report["by_goal_script"]
    assert set(by_script) == {"thai", "latin", "unknown"}
    assert by_script["thai"]["input_tokens"]["median"] == 90000
    assert by_script["latin"]["input_tokens"]["median"] == 30000
    assert "แยกตามภาษาของ goal" in format_kpi_report(report)

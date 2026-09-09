"""core/release_gate.py — W_eval: ยกระดับ core/evaluation.py + core/miniwob_eval.py +
core/orangehrm_eval.py (ที่มีอยู่แล้ว ก่อนหน้านี้แค่พิมพ์ผลลง stdout ใน run.py) ให้เป็น
release gate จริง — รัน 3 suite รวมกัน, เขียนสรุปเป็น JSON ต่อ run (tag ด้วย git commit +
model + provider + timestamp) ไว้ที่ settings.release_gate_results_dir, แล้วเทียบกับผลรัน
ล่าสุดก่อนหน้า (หรือ baseline ที่ระบุเอง) — metric ไหน regress เกิน
settings.release_gate_max_regression_pct ถือว่า "ไม่ผ่าน" คืน exit code ไม่เท่ากับ 0 (ผ่าน
run.py::run_release_gate ที่เรียก sys.exit() เอง) พร้อมกันได้ — ยังไม่ผูก CI อัตโนมัติ
(repo นี้ยังไม่มี .github workflow เลย เป็นการตัดสินใจแยกต่างหาก) แค่ให้ exit code พร้อมต่อ
CI ทันทีที่ต้องการ

ทำไม reuse EvaluationReport ตรงๆ แทนที่จะเขียน aggregator ใหม่: EvaluationReport's
properties (success_rate/p50_latency_seconds/.../recovery_rate) เข้าถึง self.results ผ่าน
attribute access ล้วนๆ (duck-typed) — TaskEvalResult/MiniWobResult มี field ชื่อตรงกันครบ
ทุกตัวที่ property พวกนี้ต้องใช้ (success/steps/total_tokens/latency_seconds/llm_calls/
approval_count/fastpath/recoveries) ต่างกันแค่ field เสริมที่ property พวกนี้ไม่แตะ (reward/
utterance/task ของ MiniWobResult, name/goal ของ TaskEvalResult) — ผสม 2 ชนิด dataclass ใน
list เดียวกันแล้วสร้าง EvaluationReport(results=...) ได้ตรงๆ โดยไม่ต้อง duplicate สูตร
คำนวณเดิมเลยสักบรรทัด"""

import json
import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from backend.app.config import settings
from backend.app.core.evaluation import EvaluationReport, run_evaluation
from backend.app.core.miniwob_eval import run_miniwob_evaluation
from backend.app.core.orangehrm_eval import run_orangehrm_evaluation
from backend.app.core.telemetry import new_run_id

# metric ไหน "ยิ่งสูงยิ่งดี" (regression = ลดลง) เทียบกับ "ยิ่งต่ำยิ่งดี" (regression =
# เพิ่มขึ้น) — ต้องรู้ทิศทางถึงจะตัดสิน pass/fail ต่อ metric ได้ถูก ไม่ใช่ทุก metric ที่
# "ตัวเลขเปลี่ยน" จะแปลว่า "แย่ลง" เหมือนกันหมด
_HIGHER_IS_BETTER = {"success_rate", "fastpath_hit_rate", "recovery_rate"}
_LOWER_IS_BETTER = {
    "p50_latency_seconds", "p95_latency_seconds", "avg_llm_calls", "avg_tokens", "approval_rate",
}
METRIC_NAMES = sorted(_HIGHER_IS_BETTER | _LOWER_IS_BETTER)
# W_gate_is_noisy: ความถูกต้องคือ metric เดียวที่ตัดสิน commit ได้
_CORRECTNESS_METRICS = {"success_rate"}


def _git_commit_short() -> str:
    """คืน short SHA ของ HEAD ปัจจุบัน หรือ "unknown" เงียบๆ ถ้าไม่ใช่ git repo/git ไม่มีใน
    PATH (ห้าม throw ทำให้ release gate ทั้งตัวพังเพราะแค่ tag ผลลัพธ์ไม่ได้)"""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _sanitize_for_filename(text: str) -> str:
    """แทนอักขระที่ใช้เป็นชื่อไฟล์ไม่ได้ (Windows โดยเฉพาะ: / \\ : * ? " < > |) ด้วย "_" —
    model string มักมีจุด/ขีดคั่นอยู่แล้วซึ่งใช้ได้ปกติ ไม่ต้องแตะ"""
    bad_chars = '/\\:*?"<>| '
    return "".join("_" if c in bad_chars else c for c in text) or "unknown"


def _result_row(result) -> dict:
    """W_gate_per_task: หนึ่งแถวต่อ task สำหรับเก็บลง summary JSON

    duck-typed เหมือนที่ EvaluationReport ทำกับ TaskEvalResult/MiniWobResult (ดู docstring
    หัวไฟล์) — ใช้ getattr มีค่า default ทุกตัว เพราะสอง dataclass มี field เสริมไม่ตรงกัน
    (name/goal ของ TaskEvalResult vs task/utterance/reward ของ MiniWobResult)

    ทำไมต้องมี: baseline 2 รอบแรก (W82/W83) บอกได้แค่ "14/15 ผ่าน" แต่ไม่รู้ว่า task ไหนตก
    และตอน W84 วัดผลจริง approval_rate เด้ง 2 เท่าโดยอธิบายไม่ได้เลยว่ามาจาก task ไหน เพราะ
    approval_count รายตัวไม่เคยถูกบันทึกลงไฟล์ — metric รวมอย่างเดียวชี้ตัวการไม่ได้"""
    return {
        "name": getattr(result, "name", None) or getattr(result, "task", None),
        "success": getattr(result, "success", False),
        "steps": getattr(result, "steps", 0),
        "total_tokens": getattr(result, "total_tokens", 0),
        "latency_seconds": round(getattr(result, "latency_seconds", 0.0), 2),
        "llm_calls": getattr(result, "llm_calls", 0),
        "approval_count": getattr(result, "approval_count", 0),
        "fastpath": getattr(result, "fastpath", False),
        "recoveries": getattr(result, "recoveries", 0),
        # W_gate_task_level_diff: ตัวนับจริงต่อ task — ค่าเฉลี่ยรวมกลบความต่าง
        # รายงานได้ว่า "avg_llm_calls +20%" แต่บอกไม่ได้ว่างานไหนเปลี่ยน
        "action_calls": getattr(result, "action_calls", 0),
        "finish_task_calls": getattr(result, "finish_task_calls", 0),
        # W_gate_run_invalid: เก็บข้อความสรุปของ task ไว้ด้วย — รันที่ล้มยกชุดเพราะ
        # provider ปฏิเสธ ไม่มี error field ให้ดูเลย (run_task คืน dict ปกติ status=done)
        "message": str(getattr(result, "message", "") or "")[:160],
        "error": getattr(result, "error", None),
    }


def build_summary(
    report: EvaluationReport, *, git_commit: str, model: str, provider: str,
    run_id: Optional[str] = None,
) -> dict:
    """แปลง EvaluationReport (รวมทุก suite แล้ว) เป็น dict ที่ serialize เป็น JSON ได้ตรงๆ —
    เก็บ per_suite แยกไว้ด้วย (ไม่ใช่แค่ aggregate รวม) ให้ยังสืบสาวได้ว่า suite ไหนเป็นตัว
    ฉุด metric ลงถ้า gate ไม่ผ่าน"""
    summary = {
        "git_commit": git_commit,
        "model": model,
        "provider": provider,
        "timestamp": time.time(),
        "task_count": len(report.results),
        # W_gate_per_task: เก็บรายตัวด้วย ไม่ใช่แค่ aggregate (ดู _result_row)
        "results": [_result_row(r) for r in report.results],
        "aggregate": {
            "success_rate": report.success_rate,
            "p50_latency_seconds": report.p50_latency_seconds,
            "p95_latency_seconds": report.p95_latency_seconds,
            "avg_llm_calls": report.avg_llm_calls,
            "avg_tokens": report.avg_tokens,
            "approval_rate": report.approval_rate,
            "fastpath_hit_rate": report.fastpath_hit_rate,
            "recovery_rate": report.recovery_rate,
        },
    }
    # W_eval_trace: id เดียวกับที่ทุกบรรทัดใน step_trace.jsonl/token_usage.jsonl ของ gate run
    # นี้ถืออยู่ — ทำให้ join กลับได้ว่า "gate run ที่ตกรอบนั้น task ไหนตก ตกที่ step ไหน"
    # (สร้างตั้งแต่ต้น run_release_gate() ไม่ใช่ตรงนี้ เพราะ save_summary() เกิดหลัง suite
    # รันจบหมดแล้ว — ดึงย้อนหลังมาผูกกับ trace ที่เขียนไปก่อนหน้าไม่ได้)
    if run_id:
        summary["run_id"] = run_id
    return summary


def save_summary(summary: dict, results_dir: Optional[str] = None) -> Path:
    """เขียน summary เป็น JSON ไฟล์ใหม่ 1 ไฟล์ต่อ run (ไม่เคย overwrite ไฟล์เก่า — ต้องเก็บ
    ประวัติไว้เทียบย้อนหลังได้เสมอ) ชื่อไฟล์รวม commit+model+timestamp กันชนกันเองถ้ารัน
    ติดกันเร็วมาก"""
    directory = Path(results_dir or settings.release_gate_results_dir)
    directory.mkdir(parents=True, exist_ok=True)
    filename = (
        f"{_sanitize_for_filename(summary['git_commit'])}"
        f"_{_sanitize_for_filename(summary['model'])}"
        f"_{int(summary['timestamp'] * 1000)}.json"
    )
    path = directory / filename
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_latest_summary(results_dir: Optional[str] = None, *, exclude_path: Optional[Path] = None) -> Optional[dict]:
    """หา summary JSON ล่าสุดใน results_dir (เรียงตาม field "timestamp" ข้างในไฟล์เอง ไม่ใช่
    mtime ของไฟล์ — เผื่อไฟล์ถูก copy/sync มาจากที่อื่นแล้ว mtime ไม่ตรงกับตอนที่ eval รันจริง)
    คืน None เงียบๆ ถ้า dir ไม่มีอยู่/ไม่มีไฟล์ JSON ที่อ่านได้เลย (เช่น รัน release gate เป็น
    ครั้งแรกไม่เคยมี baseline มาก่อน) — exclude_path กันไม่ให้เทียบไฟล์ล่าสุดกับตัวเองถ้า
    caller เพิ่ง save_summary() ของ run นี้ไปแล้วก่อนเรียกฟังก์ชันนี้"""
    directory = Path(results_dir or settings.release_gate_results_dir)
    if not directory.is_dir():
        return None
    candidates = []
    for path in directory.glob("*.json"):
        if exclude_path is not None and path.resolve() == exclude_path.resolve():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            # W_gate_is_noisy: โฟลเดอร์เดียวกันนี้เก็บรายงาน flakiness ด้วย ซึ่งไม่ใช่
            # ผลรันเดี่ยวและไม่มี "aggregate" — ถ้าหลุดเข้าไปเป็น baseline จะกลายเป็น
            # การเทียบกับศูนย์ทุก metric เงียบๆ
            if "aggregate" not in data:
                continue
            candidates.append((data.get("timestamp", 0), data))
        except Exception:
            continue
    if not candidates:
        return None
    candidates.sort(key=lambda pair: pair[0])
    return candidates[-1][1]


def load_recent_summaries(
    results_dir: Optional[str] = None, *, limit: int = 5, exclude_path: Optional[Path] = None,
) -> list[dict]:
    """คืน summary JSON ล่าสุดไม่เกิน limit ไฟล์ เรียงเก่า->ใหม่ (เกณฑ์เดียวกับ
    load_latest_summary: เรียงตาม field "timestamp" ข้างในไฟล์ ไม่ใช่ mtime)"""
    directory = Path(results_dir or settings.release_gate_results_dir)
    if not directory.is_dir():
        return []
    candidates = []
    for path in directory.glob("*.json"):
        if exclude_path is not None and path.resolve() == exclude_path.resolve():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            # W_gate_is_noisy: โฟลเดอร์เดียวกันนี้เก็บรายงาน flakiness ด้วย ซึ่งไม่ใช่
            # ผลรันเดี่ยวและไม่มี "aggregate" — ถ้าหลุดเข้าไปเป็น baseline จะกลายเป็น
            # การเทียบกับศูนย์ทุก metric เงียบๆ
            if "aggregate" not in data:
                continue
            candidates.append((data.get("timestamp", 0), data))
        except Exception:
            continue
    candidates.sort(key=lambda pair: pair[0])
    return [data for _, data in candidates[-limit:]] if limit > 0 else []


def build_noise_baseline(summaries: list[dict]) -> tuple[dict, dict[str, float]]:
    """W_gate_noise_floor: ยุบ summary หลายรันเป็น baseline เดียว + วัด noise ของ harness เอง

    baseline ต่อ metric = median (ไม่ใช่ mean — รันเดียวที่ timeout/ดวงดีไม่ควรลาก baseline
    ทั้งก้อน) และ spread = (max-min)/|median|*100 = "ตัวเลขนี้แกว่งได้เองแค่ไหนโดยไม่ต้องมี
    ใครแก้โค้ดเลย" ซึ่งเป็นตัวตัดสินว่า metric นั้นเอามา gate ได้จริงไหม

    summaries เดียว: spread=0.0 ทุกตัว (วัด noise ไม่ได้จากจุดข้อมูลเดียว) — จงใจให้ผลลัพธ์
    เท่ากับพฤติกรรมเดิมทุกประการ ไม่ใช่ให้ "ผ่านหมด" เพราะข้อมูลไม่พอ
    """
    aggregate: dict[str, float] = {}
    spread: dict[str, float] = {}
    for metric in METRIC_NAMES:
        values = [
            float(summary.get("aggregate", {}).get(metric, 0.0)) for summary in summaries
        ]
        if not values:
            aggregate[metric], spread[metric] = 0.0, 0.0
            continue
        median = statistics.median(values)
        aggregate[metric] = median
        spread[metric] = (
            (max(values) - min(values)) / abs(median) * 100.0 if median else 0.0
        )
    newest = summaries[-1] if summaries else {}
    baseline = {
        "git_commit": newest.get("git_commit", "unknown"),
        "model": newest.get("model", "unknown"),
        "provider": newest.get("provider"),
        "timestamp": newest.get("timestamp", 0),
        "aggregate": aggregate,
        "baseline_run_count": len(summaries),
        "baseline_run_ids": [s.get("run_id") for s in summaries],
    }
    return baseline, spread


@dataclass
class MetricComparison:
    metric: str
    current: float
    baseline: float
    pct_change: float
    passed: bool
    # W_gate_noise_floor: gating=False แปลว่า metric นี้ "รายงานได้ แต่ตัดสินไม่ได้" —
    # spread ของมันเองระหว่างรันที่ไม่มีอะไรเปลี่ยนเลย กว้างกว่าเกณฑ์ regression ที่ตั้งไว้
    # ปล่อยให้มัน fail gate = สร้าง false alarm ประจำจนคนเลิกเชื่อ gate ทั้งตัว
    # (ค่า default ทั้งคู่ทำให้ caller เดิมที่ไม่ส่ง history เข้ามาได้พฤติกรรมเหมือนเดิมเป๊ะ)
    gating: bool = True
    spread_pct: Optional[float] = None


def compare_against_baseline(
    current: dict, baseline: dict, max_regression_pct: Optional[float] = None,
    *, noise_pct: Optional[dict[str, float]] = None,
) -> list[MetricComparison]:
    """เทียบ current["aggregate"] กับ baseline["aggregate"] ทีละ metric — pct_change เป็น
    "ทิศทางจริง" เสมอ (บวก = ตัวเลขเพิ่มขึ้น, ลบ = ลดลง ไม่ว่า metric นั้นจะเป็น higher-
    is-better หรือ lower-is-better) ส่วน passed ตีความตาม _HIGHER_IS_BETTER/_LOWER_IS_BETTER
    ด้านบนแยกต่างหาก — baseline metric ที่เป็น 0 พอดี (เช่น recovery_rate ที่ยังไม่เคยมี
    fastpath task ต้องพึ่ง repair เลย) กัน division by zero ด้วยการถือว่า "ผ่าน" เสมอถ้า
    current ก็ไม่ได้แย่ลงในทิศทางที่วัดได้ (0 -> 0 นับเป็นไม่เปลี่ยน ไม่ใช่ regression)"""
    threshold = max_regression_pct if max_regression_pct is not None else settings.release_gate_max_regression_pct
    comparisons = []
    for metric in METRIC_NAMES:
        current_value = float(current.get("aggregate", {}).get(metric, 0.0))
        baseline_value = float(baseline.get("aggregate", {}).get(metric, 0.0))
        if baseline_value == 0.0:
            pct_change = 0.0 if current_value == 0.0 else float("inf")
        else:
            pct_change = (current_value - baseline_value) / abs(baseline_value) * 100.0

        if metric in _HIGHER_IS_BETTER:
            # regression = ลดลงเกิน threshold (pct_change ติดลบมากกว่า -threshold)
            passed = pct_change >= -threshold
        else:
            # regression = เพิ่มขึ้นเกิน threshold
            passed = pct_change <= threshold

        # W_gate_noise_floor: metric ที่ noise ของตัวเองกว้างกว่าเกณฑ์ ตัดสิน regression
        # ไม่ได้ — ยังคำนวณ passed ตามปกติเพื่อให้รายงานอ่านได้ แค่ไม่ให้มัน gate
        #
        # W_gate_is_noisy: ยิ่งกว่านั้น metric ประสิทธิภาพ (step/token/latency/llm_calls/
        # approval) ไม่ตัดสิน pass/fail อีกต่อไปไม่ว่า spread จะแคบแค่ไหน — task ที่
        # สำเร็จใน 9 step กับ 22 step คือ task ที่สำเร็จเหมือนกัน ความถูกต้องคือ
        # success_rate ตัวเดียว ที่เหลือเป็นข้อมูลประกอบ (แบนด์ที่แคบเพราะ 5 รอบ
        # ล่าสุดบังเอิญนิ่ง เคยตีธง FAIL ให้ approval_rate มาแล้วสองครั้งในวันเดียว)
        metric_spread = None if noise_pct is None else float(noise_pct.get(metric, 0.0))
        gates = metric in _CORRECTNESS_METRICS and (
            metric_spread is None or metric_spread <= threshold
        )
        comparisons.append(MetricComparison(
            metric=metric, current=current_value, baseline=baseline_value,
            pct_change=pct_change, passed=passed,
            gating=gates,
            spread_pct=metric_spread,
        ))
    return comparisons


# W_gate_is_noisy (2026-09-09, หลังวัดกับของจริงทั้งวัน): gate รอบเดียวตัดสิน commit ไม่ได้
# หลักฐานที่ปิดเรื่องนี้: commit 3ddb18a รันสองครั้งติดโดยไม่แตะโค้ดเลย ได้ 12/15 (ธง FAIL)
# แล้ว 15/15 (ผ่าน) — task ที่ล้มรอบแรกทั้งสามตัวผ่านหมดในรอบสอง ก่อนหน้านั้น ec556ba ก็
# ให้ 1.000 แล้ว 0.800 มาแล้วเช่นกัน
#
# สามอย่างที่เปลี่ยนตามหลักฐานนี้:
#   1. ความถูกต้อง (success) ตัดสินด้วย "median ของหลายรัน" ไม่ใช่รันเดียว
#   2. metric ประสิทธิภาพ (step/token/latency/llm_calls) รายงานอย่างเดียว ไม่ตัดสิน pass/fail
#      — task ที่สำเร็จใน 9 step กับ 22 step ก็คือ task ที่สำเร็จเหมือนกัน
#   3. task ที่ผ่านบ้างล้มบ้างในโค้ดเดียวกันถูกทำเครื่องหมาย FLAKY เพื่อไม่ให้มีใครเสียเวลา
#      ไล่ regression ที่ไม่มีอยู่จริง
_FLAKY_LOW = 0.2
_FLAKY_HIGH = 0.8


# W_gate_infra_failure (2026-09-09): รัน flakiness ครั้งแรกที่วัดได้จริงจบด้วย 4 task ขึ้น
# FLAKY (4/5) — พอเปิด message ดูพบว่าทั้งสี่ล้มที่ step 0 ด้วย "ResourceExhausted: 429 You
# exceeded your current quota" ในรอบที่ 5 รอบเดียว ส่วนรอบ 1-4 ผ่านครบ 15/15 ทุกรอบ
#
# นี่คือ infrastructure คนละชนิดกับ run_is_invalid() ด้านบน: ตรงนั้นคือ "ทั้งรันตายหมด"
# ส่วนนี่คือ "โควตาหมดกลางรัน" — task ที่เหลือยังเดินได้ รันจึงยัง valid แต่ task ที่โดน
# ไม่ได้บอกอะไรเลยเกี่ยวกับ commit การนับมันเป็น "ล้ม" คือการสร้าง flaky ปลอมขึ้นมาเอง
_INFRA_FAILURE_MARKERS = (
    "resourceexhausted", "429", "exceeded your current quota", "rate limit",
    "quota", "insufficient_quota", "authentication_error", "api key is invalid",
    "not supported when using codex", "connection error", "temporarily unavailable",
    "503", "502",
)


def task_failed_on_infrastructure(row: dict) -> bool:
    """task ที่ล้มโดยไม่ได้ลงมือทำอะไรเลย และข้อความบอกว่าเป็นปัญหาฝั่ง provider/เครือข่าย

    ต้องครบทั้งสามอย่าง: ล้ม + steps == 0 + ข้อความเข้าเงื่อนไข — task ที่เดินไปได้หลาย
    step แล้วค่อยเจอ 429 ตอนท้ายยังนับเป็นผลจริง เพราะมันได้ทำงานจริงไปแล้วส่วนหนึ่ง"""
    if row.get("success"):
        return False
    if int(row.get("steps", 0) or 0) != 0:
        return False
    text = f"{row.get('message') or ''} {row.get('error') or ''}".lower()
    return any(marker in text for marker in _INFRA_FAILURE_MARKERS)


def run_is_invalid(summary: dict) -> bool:
    """รันที่ไม่มี task ไหนได้ลงมือทำอะไรเลย = วัดอะไรไม่ได้ ไม่ใช่ regression

    W_gate_run_invalid (2026-09-09): endpoint ของ ChatGPT OAuth ปฏิเสธทุกโมเดลกลางวัน
    ("The 'gpt-5.4-mini' model is not supported when using Codex with a ChatGPT account")
    ผลคือ flakiness 5 รอบได้ 0/5 ทุก task และ gate อ่านออกมาเป็น success_rate 0.000 —
    ซึ่งจะไป fail commit ให้กับ provider ที่ล่ม นี่คือ false positive ที่แย่ที่สุดของ gate

    เกณฑ์ที่ใช้คือ "ทุก task จบด้วย steps == 0" เพราะ task ใน benchmark ทุกตัวต้องเปิด
    เบราว์เซอร์และลงมือทำอย่างน้อยหนึ่ง action เสมอ — ไม่มีทางที่โค้ดจะพังจนทุกงานได้ศูนย์
    step พร้อมกันโดยที่ยังไม่ใช่ปัญหาโครงสร้าง (ต่างจาก task เดี่ยวที่ได้ 0 step ซึ่งเกิด
    ได้ปกติจาก short-circuit ของ api/routes.py)"""
    rows = summary.get("results") or []
    if not rows:
        return True
    return all(int(row.get("steps", 0) or 0) == 0 for row in rows)


def task_flakiness(summaries: list[dict]) -> list[dict]:
    """อัตราการผ่านต่อ task จากหลายรันของ *commit เดียวกัน* เรียงจากผ่านน้อยไปมาก

    verdict: "stable-pass" (ผ่านทุกรอบ) / "stable-fail" (ล้มทุกรอบ) / "flaky" (อยู่ระหว่าง
    20-80%) — เกณฑ์ตามที่ user กำหนด สิ่งที่อยู่นอกช่วงนั้นแต่ไม่ใช่ 0/1 พอดี (เช่น 1/5)
    ถือว่า "mostly-fail"/"mostly-pass" ซึ่งยังต้องดู แต่ไม่ใช่ noise เต็มตัว"""
    runs: dict[str, list[bool]] = {}
    skipped: dict[str, int] = {}
    for summary in summaries:
        for row in summary.get("results", []) or []:
            name = row.get("name")
            if not name:
                continue
            # W_gate_infra_failure: โควตาหมด/คีย์เสีย = ไม่มีข้อมูลเกี่ยวกับ task นี้เลย
            # ในรอบนั้น ไม่ใช่ "ล้ม" — นับเป็นล้มเมื่อไหร่ก็ได้ flaky ปลอมทันที
            if task_failed_on_infrastructure(row):
                skipped[name] = skipped.get(name, 0) + 1
                continue
            runs.setdefault(name, []).append(bool(row.get("success")))
    report = []
    for name, outcomes in runs.items():
        rate = sum(outcomes) / len(outcomes)
        if rate == 1.0:
            verdict = "stable-pass"
        elif rate == 0.0:
            verdict = "stable-fail"
        elif _FLAKY_LOW <= rate <= _FLAKY_HIGH:
            verdict = "flaky"
        else:
            verdict = "mostly-pass" if rate > _FLAKY_HIGH else "mostly-fail"
        report.append({
            "name": name, "runs": len(outcomes), "passed": sum(outcomes),
            "pass_rate": round(rate, 3), "verdict": verdict,
            "skipped_infra": skipped.get(name, 0),
        })
    return sorted(report, key=lambda row: (row["pass_rate"], row["name"]))


def success_rate_stats(summaries: list[dict]) -> dict:
    """min / median / max ของ success_rate ข้ามหลายรัน — median คือค่าที่ใช้ตัดสิน

    รันเดียวเป็นตัวอย่างเดียวจากการแจกแจงที่กว้างพอจะกินทั้ง 0.8 ถึง 1.0 ได้ (วัดแล้ว
    บน commit เดียวกัน) การเทียบตัวอย่างเดียวกับตัวอย่างเดียวจึงบอกอะไรไม่ได้เลย"""
    rates = [float(s.get("aggregate", {}).get("success_rate", 0.0)) for s in summaries]
    if not rates:
        return {"min": 0.0, "median": 0.0, "max": 0.0, "runs": 0}
    return {
        "min": min(rates), "median": statistics.median(rates), "max": max(rates),
        "runs": len(rates),
    }


def compare_tasks(current: dict, baseline: dict) -> list[dict]:
    """เทียบ task ต่อ task ระหว่างสอง summary — คืนเฉพาะแถวที่ผลลัพธ์ (success) ต่างกัน
    หรือตัวนับขยับเกิน 50% เพราะค่าเฉลี่ยรวมบอกได้แค่ "อะไรบางอย่างเปลี่ยน"

    ตัวนับที่เทียบเป็น step/llm_calls/action_calls/finish_task_calls — ทั้งหมดเป็น
    metric ประสิทธิภาพ จึงไม่มีอันไหนตัดสิน pass/fail (ดู W_gate_is_noisy ด้านบน)"""
    base_rows = {r.get("name"): r for r in (baseline.get("results") or [])}
    diffs = []
    for row in current.get("results") or []:
        name = row.get("name")
        before = base_rows.get(name)
        if before is None:
            continue
        counters = {}
        for field in ("steps", "llm_calls", "action_calls", "finish_task_calls"):
            was, now = int(before.get(field, 0) or 0), int(row.get(field, 0) or 0)
            if was != now:
                counters[field] = (was, now)
        success_changed = bool(before.get("success")) != bool(row.get("success"))
        big_move = any(
            abs(now - was) > max(1, was * 0.5) for was, now in counters.values()
        )
        if success_changed or big_move:
            diffs.append({
                "name": name,
                "success_before": bool(before.get("success")),
                "success_after": bool(row.get("success")),
                "counters": counters,
            })
    return diffs


async def run_release_gate(
    provider: Optional[str] = None,
    results_dir: Optional[str] = None,
    baseline_path: Optional[str] = None,
    max_regression_pct: Optional[float] = None,
    include_orangehrm: bool = True,
    include_miniwob: bool = True,
) -> dict[str, Any]:
    """รัน SauceDemo (เสมอ) + OrangeHRM + MiniWoB (ปิดได้ทีละตัวผ่าน include_*, เผื่อเครื่อง
    dev ยังไม่ได้ pip install miniwob/เจอ shared demo ล่ม) รวมผลเป็น EvaluationReport เดียว
    (ดู module docstring สำหรับเหตุผลที่ mix TaskEvalResult/MiniWobResult ในลิสต์เดียวกันได้)
    save_summary() เสมอไม่ว่า baseline จะเจอไหม (ทุก run คือ baseline ของ run ถัดไป) แล้ว
    เทียบกับ baseline_path ที่ระบุเอง หรือถ้าไม่ระบุ ใช้ไฟล์ล่าสุดก่อนหน้า (ไม่นับไฟล์ที่
    เพิ่ง save ไปเอง) — ไม่มี baseline เลย (ครั้งแรก) ถือว่า "ผ่าน" เสมอ (ไม่มีอะไรให้ regress
    เทียบกับ) พร้อม comparisons=[] ให้ผู้เรียกรู้ว่าเป็นกรณีนี้"""
    resolved_provider = provider or settings.llm_provider
    # W_eval_trace: สร้างก่อนรัน suite แรกเสมอ แล้วส่งลงไปให้ทั้ง 3 suite ใช้ร่วมกัน — ทุกบรรทัด
    # trace/token ของ gate run นี้จึงถือ id เดียวกันและ join กับ summary JSON ได้
    run_id = new_run_id("gate")

    combined_results = []
    sauce_report = await run_evaluation(provider=provider, run_id=run_id)
    combined_results.extend(sauce_report.results)
    if include_orangehrm:
        orangehrm_report = await run_orangehrm_evaluation(provider=provider, run_id=run_id)
        combined_results.extend(orangehrm_report.results)
    if include_miniwob:
        miniwob_report = await run_miniwob_evaluation(provider=provider, run_id=run_id)
        combined_results.extend(miniwob_report.results)

    combined_report = EvaluationReport(results=combined_results)
    from backend.app.core.orchestrator import Orchestrator
    model = Orchestrator._llm_backend(resolved_provider)[1]
    summary = build_summary(
        combined_report, git_commit=_git_commit_short(), model=model,
        provider=resolved_provider, run_id=run_id,
    )
    saved_path = save_summary(summary, results_dir)

    # W_gate_noise_floor: baseline_path ที่ระบุเองยังหมายถึง "ไฟล์เดียวนี้เท่านั้น" เหมือนเดิม
    # (ผู้เรียกจงใจปักหมุดไว้แล้ว ห้ามไปเฉลี่ยกับไฟล์อื่นให้) — เฉพาะเส้นทาง auto เท่านั้นที่
    # เปลี่ยนไปใช้ median ของหลายรัน
    noise_pct: Optional[dict[str, float]] = None
    if baseline_path:
        baseline = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    else:
        history = load_recent_summaries(
            results_dir, limit=settings.release_gate_baseline_runs, exclude_path=saved_path,
        )
        baseline, noise_pct = (None, None) if not history else build_noise_baseline(history)

    if baseline is None:
        return {
            "summary": summary, "saved_path": str(saved_path), "baseline": None,
            "comparisons": [], "passed": True,
        }

    comparisons = compare_against_baseline(
        summary, baseline, max_regression_pct, noise_pct=noise_pct,
    )
    return {
        "summary": summary, "saved_path": str(saved_path), "baseline": baseline,
        "comparisons": comparisons,
        "passed": all(c.passed for c in comparisons if c.gating),
    }


async def run_release_gate_repeated(
    repeats: int = 3,
    provider: Optional[str] = None,
    results_dir: Optional[str] = None,
    max_regression_pct: Optional[float] = None,
    include_orangehrm: bool = True,
    include_miniwob: bool = True,
) -> dict[str, Any]:
    """รัน gate ซ้ำ `repeats` รอบบนโค้ดชุดเดียวกัน แล้วตัดสินด้วย median

    W_gate_is_noisy: รันเดียวตัดสิน commit ไม่ได้ (ดูคอมเมนต์เหนือ task_flakiness) —
    ตัวเลขที่คืนออกไปจึงเป็น min/median/max ของ success_rate พร้อมอัตราการผ่านราย task
    ข้ามทุกรอบ ส่วน pass/fail มาจาก median เทียบ baseline ด้วยเกณฑ์เดิม

    baseline ที่ใช้เทียบคือ baseline ของรอบแรก (ก่อนรอบนี้จะเขียนผลของตัวเองลงไป) —
    ไม่งั้นรอบที่ 2-3 จะไปเทียบกับรอบที่ 1 ของตัวเอง ซึ่งไม่ใช่การเทียบข้าม commit อีกต่อไป
    """
    runs: list[dict] = []
    summaries: list[dict] = []
    baseline: Optional[dict] = None
    for attempt in range(max(1, repeats)):
        outcome = await run_release_gate(
            provider=provider, results_dir=results_dir,
            max_regression_pct=max_regression_pct,
            include_orangehrm=include_orangehrm, include_miniwob=include_miniwob,
        )
        runs.append(outcome)
        summaries.append(outcome["summary"])
        if attempt == 0:
            baseline = outcome.get("baseline")

    # W_gate_run_invalid: รันที่ provider ล่ม (ทุก task 0 step) ไม่ถูกนับทั้งใน median
    # และใน flakiness — ไม่งั้น outage หนึ่งครั้งจะกลายเป็น "ทุก task เป็น stable-fail"
    valid = [s for s in summaries if not run_is_invalid(s)]
    invalid_runs = len(summaries) - len(valid)
    stats = success_rate_stats(valid)
    threshold = (
        max_regression_pct if max_regression_pct is not None
        else settings.release_gate_max_regression_pct
    )
    baseline_rate = (
        None if not baseline else float(baseline.get("aggregate", {}).get("success_rate", 0.0))
    )
    if baseline_rate and valid:
        drop_pct = (stats["median"] - baseline_rate) / abs(baseline_rate) * 100.0
        passed = drop_pct >= -threshold
    else:
        drop_pct, passed = 0.0, True

    return {
        "runs": runs,
        "summaries": summaries,
        "success_rate": stats,
        "baseline_success_rate": baseline_rate,
        "median_change_pct": drop_pct,
        "flakiness": task_flakiness(valid),
        "task_diffs": (
            [] if baseline is None or not valid else compare_tasks(valid[-1], baseline)
        ),
        "invalid_runs": invalid_runs,
        # ไม่มีรันที่ใช้ได้เลย = ตอบคำถามว่า "commit นี้ดีไหม" ไม่ได้ ไม่ใช่ตอบว่า "แย่ลง"
        "measured": bool(valid),
        "passed": passed if valid else True,
    }


def save_flakiness_report(report: dict, results_dir: Optional[str] = None) -> Path:
    """เก็บรายงาน flakiness แยกจาก summary ปกติ — ตั้งชื่อขึ้นต้นด้วย "flakiness_" เพื่อไม่ให้
    load_recent_summaries() (ที่ glob "*.json" ในโฟลเดอร์เดียวกัน) หยิบไปทำ baseline"""
    directory = Path(results_dir or settings.release_gate_results_dir)
    directory.mkdir(parents=True, exist_ok=True)
    name = (
        f"flakiness_{_sanitize_for_filename(report.get('git_commit', 'unknown'))}"
        f"_{int(time.time() * 1000)}.json"
    )
    path = directory / name
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path

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
            candidates.append((data.get("timestamp", 0), data))
        except Exception:
            continue
    if not candidates:
        return None
    candidates.sort(key=lambda pair: pair[0])
    return candidates[-1][1]


@dataclass
class MetricComparison:
    metric: str
    current: float
    baseline: float
    pct_change: float
    passed: bool


def compare_against_baseline(
    current: dict, baseline: dict, max_regression_pct: Optional[float] = None,
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

        comparisons.append(MetricComparison(
            metric=metric, current=current_value, baseline=baseline_value,
            pct_change=pct_change, passed=passed,
        ))
    return comparisons


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

    if baseline_path:
        baseline = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    else:
        baseline = load_latest_summary(results_dir, exclude_path=saved_path)

    if baseline is None:
        return {
            "summary": summary, "saved_path": str(saved_path), "baseline": None,
            "comparisons": [], "passed": True,
        }

    comparisons = compare_against_baseline(summary, baseline, max_regression_pct)
    return {
        "summary": summary, "saved_path": str(saved_path), "baseline": baseline,
        "comparisons": comparisons, "passed": all(c.passed for c in comparisons),
    }

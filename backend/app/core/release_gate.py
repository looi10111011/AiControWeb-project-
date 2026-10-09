"""core/release_gate.py — W_eval: รวม suite saucedemo + hrm_local + miniwob (+ orangehrm แบบ opt-in)
เป็น release gate — เขียน summary JSON ต่อ run (tag commit/model/provider/timestamp) ลง
settings.release_gate_results_dir แล้วเทียบกับ baseline; exit code ให้ run.py::run_release_gate
(ยังไม่ผูก CI — repo ไม่มี .github workflow)

reuse EvaluationReport ตรงๆ: property ของมันเข้าถึง self.results แบบ duck-typed และ
TaskEvalResult/MiniWobResult มี field ที่ property ใช้ครบ จึงผสมสองชนิดใน list เดียวได้"""

import json
import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from backend.app.config import settings
from backend.app.core.evaluation import EvaluationReport, run_evaluation
from backend.app.core.hrm_local_eval import run_hrm_local_evaluation
from backend.app.core.miniwob_eval import run_miniwob_evaluation
from backend.app.core.orangehrm_eval import run_orangehrm_evaluation
from backend.app.core.telemetry import new_run_id

_HIGHER_IS_BETTER = {"success_rate", "fastpath_hit_rate", "recovery_rate"}
_LOWER_IS_BETTER = {
    "p50_latency_seconds", "p95_latency_seconds", "avg_llm_calls", "avg_tokens", "approval_rate",
}
METRIC_NAMES = sorted(_HIGHER_IS_BETTER | _LOWER_IS_BETTER)
# W_gate_is_noisy: ความถูกต้องคือ metric เดียวที่ตัดสิน commit ได้
_CORRECTNESS_METRICS = {"success_rate"}


def _git_commit_short() -> str:
    """short SHA ของ HEAD หรือ "unknown" — ห้าม throw (แค่ tag ผลไม่ได้ ไม่ควรทำให้ gate พัง)"""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _sanitize_for_filename(text: str) -> str:
    """แทนอักขระที่ใช้เป็นชื่อไฟล์ไม่ได้ (Windows: / \\ : * ? " < > | และช่องว่าง) ด้วย _"""
    bad_chars = '/\\:*?"<>| '
    return "".join("_" if c in bad_chars else c for c in text) or "unknown"


def _threshold(max_regression_pct: Optional[float]) -> float:
    return (
        max_regression_pct if max_regression_pct is not None
        else settings.release_gate_max_regression_pct
    )


def _write_json(results_dir: Optional[str], filename: str, data: dict) -> Path:
    directory = Path(results_dir or settings.release_gate_results_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _result_row(result) -> dict:
    """W_gate_per_task: หนึ่งแถวต่อ task (duck-typed, getattr มี default ทุกตัว)
    baseline W82/W83 บอกได้แค่ "14/15 ผ่าน" และ W84 approval_rate เด้ง 2 เท่าโดยชี้ task ไม่ได้"""
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
        # W_gate_task_level_diff: ตัวนับจริงต่อ task — ค่าเฉลี่ยรวมบอกไม่ได้ว่างานไหนเปลี่ยน
        "action_calls": getattr(result, "action_calls", 0),
        "finish_task_calls": getattr(result, "finish_task_calls", 0),
        # W_gate_run_invalid: รันที่ provider ปฏิเสธไม่มี error field (status=done) ต้องดูจาก message
        # 2026-09-10: 160 -> 400 เพราะข้อความ 404 ของ codex ถูกตัดกลางคำ "access" จน marker ของ
        # task_failed_on_infrastructure() match ไม่ติด
        "message": str(getattr(result, "message", "") or "")[:400],
        "error": getattr(result, "error", None),
    }


# W_gate_suite_baseline: ผลต่าง suite เทียบกันไม่ได้ (สลับ OrangeHRM -> hrm_local แล้วรายงานดีขึ้น/แย่ลง
# ทั้งที่โค้ดไม่เปลี่ยน) ไฟล์เก่าที่ไม่มี field "suites" ถูกสร้างตอน gate มีแค่ชุดนี้เสมอ
LEGACY_SUITES = ["miniwob", "orangehrm", "saucedemo"]


def build_summary(
    report: EvaluationReport, *, git_commit: str, model: str, provider: str,
    run_id: Optional[str] = None, suites: Optional[list[str]] = None,
) -> dict:
    """EvaluationReport (รวมทุก suite) -> dict ที่ serialize เป็น JSON ได้ พร้อมแถวราย task"""
    summary = {
        "git_commit": git_commit,
        "model": model,
        "provider": provider,
        "timestamp": time.time(),
        "suites": sorted(suites) if suites is not None else LEGACY_SUITES,
        "task_count": len(report.results),
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
    # W_eval_trace: id เดียวกับทุกบรรทัด step_trace/token_usage ของ gate run นี้ (สร้างตั้งแต่ต้น
    # run_release_gate() เพราะ summary เกิดหลัง trace ถูกเขียนไปแล้ว)
    if run_id:
        summary["run_id"] = run_id
    return summary


def save_summary(summary: dict, results_dir: Optional[str] = None) -> Path:
    """เขียน summary เป็นไฟล์ใหม่ 1 ไฟล์ต่อ run (ไม่ overwrite — ต้องเก็บประวัติ)"""
    filename = (
        f"{_sanitize_for_filename(summary['git_commit'])}"
        f"_{_sanitize_for_filename(summary['model'])}"
        f"_{int(summary['timestamp'] * 1000)}.json"
    )
    return _write_json(results_dir, filename, summary)


# W_gate_model_baseline (2026-09-10): baseline เคยหยิบไฟล์ล่าสุดโดยไม่สนโมเดล — เปลี่ยน
# gpt-5.4-mini -> gemini-flash-lite -> gpt-5.5 ในสองวันแล้วรายงาน "แย่ลง 40%" ทั้งที่ commit ไม่ผิด
# (โค้ดเดียวกัน 12/15 vs 15/15) โมเดลใหม่ที่ไม่มีประวัติ = ไม่มี baseline (ผ่านและบอกว่าไม่มีอะไรเทียบ)
def _summaries_in(
    results_dir: Optional[str], exclude_path: Optional[Path], model: Optional[str],
    suites: Optional[list[str]] = None,
) -> list[tuple[float, dict]]:
    directory = Path(results_dir or settings.release_gate_results_dir)
    if not directory.is_dir():
        return []
    candidates: list[tuple[float, dict]] = []
    for path in directory.glob("*.json"):
        if exclude_path is not None and path.resolve() == exclude_path.resolve():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        # W_gate_is_noisy: รายงาน flakiness ในโฟลเดอร์เดียวกันไม่มี "aggregate" — ถ้าหลุดเป็น
        # baseline จะเทียบกับศูนย์ทุก metric เงียบๆ
        if "aggregate" not in data:
            continue
        if model is not None and data.get("model") != model:
            continue
        if suites is not None and sorted(data.get("suites") or LEGACY_SUITES) != sorted(suites):
            continue
        # รันที่วัดอะไรไม่ได้ (run_is_invalid) ห้ามเข้า baseline — ไม่งั้นรันปกติถัดไปดู "ดีขึ้น 300%"
        if run_is_invalid(data):
            continue
        candidates.append((data.get("timestamp", 0), data))
    candidates.sort(key=lambda pair: pair[0])
    return candidates


def load_latest_summary(
    results_dir: Optional[str] = None, *, exclude_path: Optional[Path] = None,
    model: Optional[str] = None, suites: Optional[list[str]] = None,
) -> Optional[dict]:
    """summary ล่าสุด (เรียงตาม field "timestamp" ในไฟล์ ไม่ใช่ mtime — ไฟล์อาจถูก copy มา)
    None ถ้าไม่มี; exclude_path กันเทียบกับไฟล์ที่เพิ่ง save ของ run นี้เอง"""
    candidates = _summaries_in(results_dir, exclude_path, model, suites)
    return candidates[-1][1] if candidates else None


def load_recent_summaries(
    results_dir: Optional[str] = None, *, limit: int = 5, exclude_path: Optional[Path] = None,
    model: Optional[str] = None, suites: Optional[list[str]] = None,
) -> list[dict]:
    """summary ล่าสุดไม่เกิน limit ไฟล์ เรียงเก่า->ใหม่ (ตาม field "timestamp")"""
    candidates = _summaries_in(results_dir, exclude_path, model, suites)
    return [data for _, data in candidates[-limit:]] if limit > 0 else []


def build_noise_baseline(summaries: list[dict]) -> tuple[dict, dict[str, float]]:
    """W_gate_noise_floor: ยุบหลายรันเป็น baseline (median ต่อ metric) + spread
    = (max-min)/|median|*100 = metric แกว่งเองแค่ไหนโดยไม่มีใครแก้โค้ด
    summary เดียว -> spread 0.0 ทุกตัว (เท่าพฤติกรรมเดิม)"""
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
    # W_gate_noise_floor: gating=False = "รายงานได้ แต่ตัดสินไม่ได้" (noise กว้างกว่าเกณฑ์ ->
    # false alarm ประจำ) default ทั้งคู่ทำให้ caller เดิมได้พฤติกรรมเดิม
    gating: bool = True
    spread_pct: Optional[float] = None


def compare_against_baseline(
    current: dict, baseline: dict, max_regression_pct: Optional[float] = None,
    *, noise_pct: Optional[dict[str, float]] = None,
) -> list[MetricComparison]:
    """เทียบ aggregate ทีละ metric — pct_change เป็นทิศทางจริง (บวก=เพิ่ม) ส่วน passed ตีความตาม
    _HIGHER_IS_BETTER/_LOWER_IS_BETTER; baseline 0 -> pct 0 (ถ้า current 0) หรือ inf"""
    threshold = _threshold(max_regression_pct)
    comparisons = []
    for metric in METRIC_NAMES:
        current_value = float(current.get("aggregate", {}).get(metric, 0.0))
        baseline_value = float(baseline.get("aggregate", {}).get(metric, 0.0))
        if baseline_value == 0.0:
            pct_change = 0.0 if current_value == 0.0 else float("inf")
        else:
            pct_change = (current_value - baseline_value) / abs(baseline_value) * 100.0

        if metric in _HIGHER_IS_BETTER:
            passed = pct_change >= -threshold
        else:
            passed = pct_change <= threshold

        # W_gate_noise_floor: metric ที่ noise กว้างกว่าเกณฑ์ไม่ gate (ยังคำนวณ passed ไว้รายงาน)
        # W_gate_is_noisy: metric ประสิทธิภาพไม่ gate เลย — สำเร็จใน 9 กับ 22 step ก็สำเร็จเหมือนกัน
        # (แบนด์แคบเพราะ 5 รอบบังเอิญนิ่ง เคยตีธง FAIL ให้ approval_rate สองครั้งในวันเดียว)
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


# W_gate_is_noisy (2026-09-09): gate รอบเดียวตัดสิน commit ไม่ได้ — 3ddb18a ได้ 12/15 (FAIL) แล้ว 15/15
# โดยไม่แตะโค้ด, ec556ba ได้ 1.000 แล้ว 0.800 -> ตัดสินด้วย median หลายรัน, metric ประสิทธิภาพรายงาน
# อย่างเดียว, task ที่ผ่าน 20-80% ติดป้าย FLAKY
_FLAKY_LOW = 0.2
_FLAKY_HIGH = 0.8


# W_gate_infra_failure (2026-09-09): flakiness รันแรก 4 task ขึ้น FLAKY เพราะ 429 quota ที่ step 0 ในรอบ
# ที่ 5 เท่านั้น — "โควตาหมดกลางรัน" ไม่บอกอะไรเกี่ยวกับ commit (ต่างจาก run_is_invalid ที่ทั้งรันตาย)
# 2026-09-10 เพิ่ม model_not_found: codex endpoint ตอบ 404 "does not exist or you do not have access"
# เมื่อโควตาหมดกลางรัน ทั้งที่โมเดลเพิ่งใช้ได้ (ตัวชี้ขาดคือเปลี่ยนกลางรัน)
_INFRA_FAILURE_MARKERS = (
    "resourceexhausted", "429", "exceeded your current quota", "rate limit",
    "quota", "insufficient_quota", "authentication_error", "api key is invalid",
    "not supported when using codex", "connection error", "temporarily unavailable",
    "model_not_found", "does not exist or you do not have",
    "503", "502",
)


def task_failed_on_infrastructure(row: dict) -> bool:
    """True เมื่อครบสามอย่าง: ล้ม + steps == 0 + ข้อความตรง marker ฝั่ง provider/เครือข่าย
    (task ที่เดินไปหลาย step แล้วค่อยเจอ 429 ยังนับเป็นผลจริง)"""
    if row.get("success"):
        return False
    if int(row.get("steps", 0) or 0) != 0:
        return False
    text = f"{row.get('message') or ''} {row.get('error') or ''}".lower()
    return any(marker in text for marker in _INFRA_FAILURE_MARKERS)


def run_is_invalid(summary: dict) -> bool:
    """รันที่วัดอะไรไม่ได้ ไม่ใช่ regression

    W_gate_run_invalid (2026-09-09): ChatGPT OAuth ปฏิเสธทุกโมเดลกลางวัน -> success_rate 0.000 จะไป
    fail commit แทน provider ที่ล่ม เกณฑ์: ทุก task steps == 0 (benchmark ทุกตัวต้องลงมืออย่างน้อย 1 action)
    2026-09-10: โควตาหมดกลางรันได้ 5/15 โดย 10 ตัวตาย 404 ที่ step 0 -> ถ้า infra failure เกินครึ่ง = invalid"""
    rows = summary.get("results") or []
    if not rows:
        return True
    if all(int(row.get("steps", 0) or 0) == 0 for row in rows):
        return True
    infra = sum(1 for row in rows if task_failed_on_infrastructure(row))
    return infra * 2 > len(rows)


def task_flakiness(summaries: list[dict]) -> list[dict]:
    """อัตราการผ่านต่อ task จากหลายรันของ commit เดียวกัน เรียงจากผ่านน้อยไปมาก
    verdict: stable-pass / stable-fail / flaky (20-80% ตามที่ user กำหนด) / mostly-pass / mostly-fail"""
    runs: dict[str, list[bool]] = {}
    skipped: dict[str, int] = {}
    for summary in summaries:
        for row in summary.get("results", []) or []:
            name = row.get("name")
            if not name:
                continue
            # W_gate_infra_failure: โควตาหมด/คีย์เสีย = ไม่มีข้อมูล ไม่ใช่ "ล้ม" (กัน flaky ปลอม)
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
    (รันเดียวแกว่งได้ 0.8-1.0 บน commit เดียวกัน)"""
    rates = [float(s.get("aggregate", {}).get("success_rate", 0.0)) for s in summaries]
    if not rates:
        return {"min": 0.0, "median": 0.0, "max": 0.0, "runs": 0}
    return {
        "min": min(rates), "median": statistics.median(rates), "max": max(rates),
        "runs": len(rates),
    }


def compare_tasks(current: dict, baseline: dict) -> list[dict]:
    """เทียบราย task — คืนเฉพาะแถวที่ success ต่างกันหรือตัวนับ (steps/llm_calls/action_calls/
    finish_task_calls) ขยับเกิน 50% เป็นข้อมูลประกอบ ไม่ตัดสิน pass/fail (W_gate_is_noisy)"""
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
    include_orangehrm: bool = False,
    include_miniwob: bool = True,
    include_hrm_local: bool = True,
) -> dict[str, Any]:
    """รัน saucedemo (เสมอ) + hrm_local + miniwob (+ orangehrm ถ้า include_orangehrm) เป็น report เดียว
    W_gate_local_hrm: OrangeHRM สาธารณะปิดเป็นค่าเริ่มต้น (ข้อมูลร่วมกัน ผลไม่นิ่ง ตัดสินจากคำของ agent)
    ส่วน hrm_local ตัดสินจาก DB หลัง reset fixture

    save_summary() เสมอ แล้วเทียบกับ baseline_path หรือรันก่อนหน้า (ไม่นับไฟล์ที่เพิ่ง save)
    ไม่มี baseline -> passed=True, comparisons=[]"""
    resolved_provider = provider or settings.llm_provider
    # W_eval_trace: สร้างก่อน suite แรก แล้วส่งให้ทุก suite ใช้ร่วม (join trace กับ summary ได้)
    run_id = new_run_id("gate")

    combined_results = []
    sauce_report = await run_evaluation(provider=provider, run_id=run_id)
    combined_results.extend(sauce_report.results)
    suites = ["saucedemo"]
    if include_hrm_local:
        suites.append("hrm_local")
        hrm_report = await run_hrm_local_evaluation(provider=provider, run_id=run_id)
        combined_results.extend(hrm_report.results)
    if include_orangehrm:
        suites.append("orangehrm")
        orangehrm_report = await run_orangehrm_evaluation(provider=provider, run_id=run_id)
        combined_results.extend(orangehrm_report.results)
    if include_miniwob:
        suites.append("miniwob")
        miniwob_report = await run_miniwob_evaluation(provider=provider, run_id=run_id)
        combined_results.extend(miniwob_report.results)

    combined_report = EvaluationReport(results=combined_results)
    from backend.app.core.orchestrator import Orchestrator
    model = Orchestrator._llm_backend(resolved_provider)[1]
    summary = build_summary(
        combined_report, git_commit=_git_commit_short(), model=model,
        provider=resolved_provider, run_id=run_id, suites=suites,
    )
    saved_path = save_summary(summary, results_dir)

    # W_gate_noise_floor: baseline_path ที่ระบุเอง = ไฟล์เดียวนั้นเท่านั้น; เฉพาะทาง auto ใช้ median หลายรัน
    noise_pct: Optional[dict[str, float]] = None
    if baseline_path:
        baseline = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    else:
        history = load_recent_summaries(
            results_dir, limit=settings.release_gate_baseline_runs, exclude_path=saved_path,
            model=model, suites=suites,
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
    include_orangehrm: bool = False,
    include_miniwob: bool = True,
    include_hrm_local: bool = True,
) -> dict[str, Any]:
    """W_gate_is_noisy: รัน gate `repeats` รอบบนโค้ดเดียวกัน ตัดสินด้วย median success_rate เทียบ
    baseline ของรอบแรก (ก่อนรอบนี้เขียนผลตัวเอง — ไม่งั้นรอบ 2-3 เทียบกับรอบ 1 ของตัวเอง)"""
    runs: list[dict] = []
    summaries: list[dict] = []
    baseline: Optional[dict] = None
    for attempt in range(max(1, repeats)):
        outcome = await run_release_gate(
            provider=provider, results_dir=results_dir,
            max_regression_pct=max_regression_pct,
            include_orangehrm=include_orangehrm, include_miniwob=include_miniwob,
            include_hrm_local=include_hrm_local,
        )
        runs.append(outcome)
        summaries.append(outcome["summary"])
        if attempt == 0:
            baseline = outcome.get("baseline")

    # W_gate_run_invalid: รันที่ provider ล่มไม่นับทั้งใน median และ flakiness
    valid = [s for s in summaries if not run_is_invalid(s)]
    invalid_runs = len(summaries) - len(valid)
    stats = success_rate_stats(valid)
    threshold = _threshold(max_regression_pct)
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
        # ไม่มีรันที่ใช้ได้ = ตอบไม่ได้ ไม่ใช่ "แย่ลง"
        "measured": bool(valid),
        "passed": passed if valid else True,
    }


def save_flakiness_report(report: dict, results_dir: Optional[str] = None) -> Path:
    """ขึ้นต้นชื่อด้วย "flakiness_" แยกจาก summary ปกติ (ไม่มี "aggregate" จึงไม่ถูกหยิบเป็น baseline)"""
    name = (
        f"flakiness_{_sanitize_for_filename(report.get('git_commit', 'unknown'))}"
        f"_{int(time.time() * 1000)}.json"
    )
    return _write_json(results_dir, name, report)

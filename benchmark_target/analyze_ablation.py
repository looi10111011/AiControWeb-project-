"""W_ablation: สรุปผลของ run_ablation.py เป็นตาราง Markdown + CSV สำหรับรายงาน (อ่านอย่างเดียว ไม่ใช้ LLM/browser)

นับเฉพาะ attempt ที่ valid (ไม่ได้ล้มเพราะ infra) และ join telemetry ด้วย telemetry_task_id
(= f"{run_id}-{task_id}" ที่ HermesAgentAdapter เขียน) — dry-run ไม่มี telemetry ช่องที่มาจาก
token_usage/step_trace จึงว่าง

การนับ token: provider "openai" คืน input_tokens ที่ **รวม** cached แล้ว (llm.py ใช้
input_tokens_details.cached_tokens เป็นส่วนย่อย) แต่ eval suite เดิม (hrm_local_eval/evaluation/
miniwob_eval) บวก input + output + cache_read + cache_creation จึงนับ cached ซ้ำสำหรับ openai
ที่นี่คิดเองตาม provider ใน manifest — ห้ามเอา avg_tokens ของ release gate มาเทียบกับตัวเลขจากไฟล์นี้

Usage:
    python benchmark_target/analyze_ablation.py data/ablation/<session>.jsonl [--curve-task TASK_ID]
"""

import argparse
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from backend.app.config import settings  # noqa: E402
from benchmark_target.runner.stats import wilson_interval  # noqa: E402

# provider ที่ input_tokens รวม cached แล้ว — ตัวอื่น (anthropic) input ไม่รวม cache read/creation
_INPUT_INCLUDES_CACHED = {"openai"}

_TIMING_PHASES = ("snapshot", "llm", "action", "pacing", "wait")

# ข้อความที่ run_task() คืนเมื่อ exception หลุดออกจากลูป (orchestrator.py W_loop_crash)
_CRASH_PREFIX = "task stopped by an unexpected error at step "
# Gemini ตอบ "InternalServerError: 500 Internal error encountered." — marker ของ gate มีแค่ 502/503
_EXTRA_PROVIDER_MARKERS = ("internalservererror", "internal error encountered")


def is_provider_failure(success, steps: int, message: str) -> bool:
    """attempt ที่ provider ล่ม ไม่ใช่ agent ทำพลาด — ไม่นับเป็น sample ของ arm ไหนเลย

    กฎของ release gate (task_failed_on_infrastructure) จับเฉพาะ steps == 0 เพราะ gate ตัดสิน commit
    แต่ ablation เทียบ arm กัน: rep 1 (2026-10-22) มี 429 ที่ step 3 ใน BASELINE และ 500 ที่ step 5 ใน
    FULL — ถ้านับเป็นผลจริง arm ที่บังเอิญรันตอนโควตาใกล้หมดจะโดนลงโทษ จึงนับเพิ่มเมื่อลูป crash
    ด้วย error ฝั่ง provider ไม่ว่าจะอยู่ step ไหน"""
    if success:
        return False
    from backend.app.core.release_gate import _INFRA_FAILURE_MARKERS, task_failed_on_infrastructure

    if task_failed_on_infrastructure({"success": success, "steps": steps, "message": message}):
        return True
    text = (message or "").lower()
    return text.startswith(_CRASH_PREFIX) and any(
        marker in text for marker in _INFRA_FAILURE_MARKERS + _EXTRA_PROVIDER_MARKERS
    )


def total_tokens(tokens: dict, provider: str) -> int:
    def val(key: str) -> int:
        return int(tokens.get(key, 0) or 0)

    if provider in _INPUT_INCLUDES_CACHED:
        return val("input") + val("output")
    return val("input") + val("output") + val("cache_read") + val("cache_creation")


def uncached_input_tokens(tokens: dict, provider: str) -> int:
    value = int(tokens.get("input", 0) or 0)
    if provider in _INPUT_INCLUDES_CACHED:
        value -= int(tokens.get("cache_read", 0) or 0)
    return max(value, 0)


def load_session(results_path: Path) -> tuple[dict, list[dict]]:
    manifest = json.loads(results_path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    records = [
        json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    for record in records:
        # จัดใหม่ทุกครั้ง — ไฟล์เก่าเขียนด้วยเกณฑ์ที่หลวมกว่า (smoke ไม่มี field นี้, rep 1 นับแค่ step 0)
        record["provider_failure"] = bool(record.get("provider_failure")) or (
            record["claimed_success"] is not None
            and is_provider_failure(record["claimed_success"], record["steps"], record["agent_message"])
        )
        record["valid"] = record["valid"] and not record["provider_failure"]
        record.setdefault("session_id", manifest["session_id"])
    return manifest, records


# ตัวรัน/ตัววิเคราะห์ (run_ablation.py, analyze_ablation.py) อยู่ใน benchmark_target/ เหมือนกัน แต่ไม่ใช่โค้ดที่ถูกวัด
_MEASURED_CODE_PATHS = (
    "backend/", "benchmark_target/app/", "benchmark_target/catalog/", "benchmark_target/control/",
    "benchmark_target/runner/",
)
_MUST_MATCH_ACROSS_SESSIONS = ("provider", "model", "max_steps", "dry_run")


def _measured_code_changes(commits: list[str]) -> list[str] | None:
    """ไฟล์ของ agent/target ที่ต่างกันระหว่าง commit ของแต่ละ session — None = ถาม git ไม่ได้"""
    import subprocess

    changed: set[str] = set()
    for other in commits[1:]:
        try:
            out = subprocess.run(
                ["git", "diff", "--name-only", commits[0], other, "--", *_MEASURED_CODE_PATHS],
                cwd=REPO_ROOT, capture_output=True, text=True, timeout=30, check=True,
            ).stdout
        except Exception:
            return None
        changed.update(line for line in out.splitlines() if line.strip())
    return sorted(changed)


def load_sessions(results_paths: list[Path]) -> tuple[dict, list[dict]]:
    """W_ablation: รวมหลาย session เป็นการทดลองเดียว — โควตา Gemini free tier (500 request/วัน) ทำให้
    rep 1 รัน 5 arm แต่ rep 2-3 รันแค่ FULL+BASELINE เป็นอีก session (run_ablation.py --first-rep 2)
    รวมได้เฉพาะ provider/model/max_steps เดียวกัน และช่อง (arm, task, rep) ห้ามซ้ำข้าม session"""
    loaded = [load_session(path) for path in results_paths]
    if len(loaded) == 1:
        return loaded[0]
    manifests = [manifest for manifest, _ in loaded]
    records = [record for _, session_records in loaded for record in session_records]
    for key in _MUST_MATCH_ACROSS_SESSIONS:
        values = {str(m.get(key)) for m in manifests}
        if len(values) > 1:
            raise SystemExit(f"รวม session ไม่ได้: {key} ไม่ตรงกัน {sorted(values)}")
    owner: dict[tuple[str, str, int], str] = {}
    clashes = []
    for record in records:
        if not record["valid"]:
            continue
        key = (record["arm"], record["task_id"], record["rep"])
        if owner.setdefault(key, record["session_id"]) != record["session_id"]:
            clashes.append(key)
    if clashes:
        raise SystemExit(f"ช่องเดียวกันอยู่หลาย session (เลข rep ชนกัน — ใช้ --first-rep): {clashes[:5]}")

    arms = list(dict.fromkeys(arm for m in manifests for arm in m["arms"]))
    commits = list(dict.fromkeys(m["git_commit"] for m in manifests))
    combined = {
        **{key: manifests[0][key] for key in _MUST_MATCH_ACROSS_SESSIONS},
        "session_id": " + ".join(m["session_id"] for m in manifests),
        "sessions": manifests,
        "git_commit": ", ".join(commits),
        "measured_code_changes": _measured_code_changes(commits) if len(commits) > 1 else [],
        "uncommitted_files": sorted({f for m in manifests for f in m.get("uncommitted_files", [])}),
        "arms": arms,
        "tasks": list(dict.fromkeys(task for m in manifests for task in m["tasks"])),
        "reference_arm": "FULL" if "FULL" in arms else arms[0],
    }
    return combined, records


def load_telemetry(telemetry_task_ids: set[str], token_usage_path: Path, step_trace_path: Path):
    """คืน (token_usage row ต่อ task, list ของ step ต่อ task) เฉพาะ id ที่ต้องการ — ไฟล์ใหญ่ อ่านทีละบรรทัด"""
    usage: dict[str, dict] = {}
    steps: dict[str, list[dict]] = defaultdict(list)
    if not telemetry_task_ids:
        return usage, steps
    for path, sink in ((token_usage_path, "usage"), (step_trace_path, "steps")):
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                task_id = row.get("task_id")
                if task_id not in telemetry_task_ids:
                    continue
                if sink == "usage":
                    usage[task_id] = row
                else:
                    steps[task_id].append(row)
    for rows in steps.values():
        rows.sort(key=lambda r: (r.get("step") is None, r.get("step") or 0))
    return usage, steps


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _p95(values: list[float]) -> float | None:
    """nearest-rank เหมือน EvaluationReport._latency_percentile — n ระดับสิบไม่ต้อง interpolate"""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, -(-95 * len(ordered) // 100))
    return ordered[rank - 1]


def enrich(records: list[dict], provider: str, usage: dict, steps: dict) -> list[dict]:
    """1 แถวต่อ attempt ที่ valid พร้อม metric ที่คำนวณแล้ว"""
    rows = []
    for record in records:
        if not record["valid"]:
            continue
        tokens = record.get("tokens") or {}
        telemetry_id = record.get("telemetry_task_id")
        usage_row = usage.get(telemetry_id) if telemetry_id else None
        step_rows = steps.get(telemetry_id, []) if telemetry_id else []
        timing = {phase: 0.0 for phase in _TIMING_PHASES}
        for step in step_rows:
            for phase in _TIMING_PHASES:
                timing[phase] += float((step.get("timing") or {}).get(phase) or 0.0)
        rows.append({
            "session_id": record.get("session_id"),
            "arm": record["arm"],
            "task_id": record["task_id"],
            "rep": record["rep"],
            "passed": record["passed"],
            "claimed_success": record["claimed_success"],
            "false_success": bool(record["claimed_success"]) and not record["passed"],
            # rep 1 (2026-10-22) RECRUITMENT-SHORTLIST-01: shortlist สำเร็จจริง แต่ agent อ่านฟอร์ม Schedule
            # Interview ที่โผล่หลัง shortlist เป็นส่วนของงานแล้วรายงานว่าทำไม่ได้ — ความผิดพลาดของคำรายงาน
            # อีกทิศ นับแยกจาก false success
            "false_failure": record["claimed_success"] is False and record["passed"],
            "steps": record["steps"],
            "llm_calls": usage_row.get("llm_calls") if usage_row else None,
            "input_tokens": int(tokens.get("input", 0) or 0),
            "cached_tokens": int(tokens.get("cache_read", 0) or 0),
            "uncached_input_tokens": uncached_input_tokens(tokens, provider),
            "output_tokens": int(tokens.get("output", 0) or 0),
            "total_tokens": total_tokens(tokens, provider),
            "duration_seconds": record["duration_seconds"],
            **{f"time_{phase}": round(timing[phase], 2) if step_rows else None for phase in _TIMING_PHASES},
            "failure_classes": Counter(
                s.get("failure_class") for s in step_rows if s.get("failure_class") not in (None, "ok")
            ),
            "has_telemetry": usage_row is not None,
        })
    return rows


def summarize_arms(rows: list[dict], arms: list[str], invalid_counts: Counter) -> list[dict]:
    summaries = []
    for arm in arms:
        arm_rows = [r for r in rows if r["arm"] == arm]
        n = len(arm_rows)
        passed = sum(1 for r in arm_rows if r["passed"])
        ci_low, ci_high = wilson_interval(passed, n) if n else (None, None)
        llm_calls = [r["llm_calls"] for r in arm_rows if r["llm_calls"] is not None]
        input_sum = sum(r["input_tokens"] for r in arm_rows)
        summaries.append({
            "arm": arm,
            "valid_attempts": n,
            "infra_failures": invalid_counts.get(arm, 0),
            "passed": passed,
            "success_rate": passed / n if n else None,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "claimed_success": sum(1 for r in arm_rows if r["claimed_success"]),
            "false_success": sum(1 for r in arm_rows if r["false_success"]),
            "false_failure": sum(1 for r in arm_rows if r["false_failure"]),
            "median_steps": _median([r["steps"] for r in arm_rows]),
            "median_llm_calls": _median(llm_calls),
            "median_total_tokens": _median([r["total_tokens"] for r in arm_rows]),
            "median_input_tokens": _median([r["input_tokens"] for r in arm_rows]),
            "median_output_tokens": _median([r["output_tokens"] for r in arm_rows]),
            "cache_share": (sum(r["cached_tokens"] for r in arm_rows) / input_sum) if input_sum else None,
            "median_duration_seconds": _median([r["duration_seconds"] for r in arm_rows]),
            "p95_duration_seconds": _p95([r["duration_seconds"] for r in arm_rows]),
        })
    return summaries


_DELTA_METRICS = ("total_tokens", "steps", "duration_seconds")


def paired_deltas(rows: list[dict], arms: list[str], reference: str) -> list[dict]:
    """เทียบ arm กับ reference แบบจับคู่ราย task: median ของ task นั้นใน arm หาร median ใน reference
    แล้วเอา median ของ % ส่วนต่างข้าม task — ทนต่อ task ยาว/สั้นปนกันกว่าเทียบค่าเฉลี่ยรวม"""
    by_cell: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        by_cell[(row["arm"], row["task_id"])].append(row)
    tasks = sorted({row["task_id"] for row in rows})
    out = []
    for arm in arms:
        if arm == reference:
            continue
        entry = {"arm": arm, "paired_tasks": 0}
        pct: dict[str, list[float]] = defaultdict(list)
        success_diff = []
        for task_id in tasks:
            mine, ref = by_cell.get((arm, task_id)), by_cell.get((reference, task_id))
            if not mine or not ref:
                continue
            entry["paired_tasks"] += 1
            success_diff.append(
                sum(r["passed"] for r in mine) / len(mine) - sum(r["passed"] for r in ref) / len(ref)
            )
            for metric in _DELTA_METRICS:
                ref_median = statistics.median(r[metric] for r in ref)
                if ref_median:
                    pct[metric].append((statistics.median(r[metric] for r in mine) - ref_median) / ref_median * 100)
        entry["mean_success_rate_diff"] = statistics.mean(success_diff) if success_diff else None
        for metric in _DELTA_METRICS:
            entry[f"median_pct_change_{metric}"] = _median(pct[metric])
        out.append(entry)
    return out


def token_curve(records: list[dict], steps: dict, provider: str) -> list[dict]:
    """token ต่อ step (input = ขนาด prompt ของ call นั้น) — ไว้วาดกราฟเส้นว่า compaction ทำให้ไม่โตตาม step"""
    out = []
    for record in records:
        telemetry_id = record.get("telemetry_task_id")
        if not record["valid"] or not telemetry_id:
            continue
        for step in steps.get(telemetry_id, []):
            tokens = step.get("tokens") or {}
            out.append({
                "session_id": record.get("session_id"),
                "arm": record["arm"],
                "task_id": record["task_id"],
                "rep": record["rep"],
                "step": step.get("step"),
                "input_tokens": int(tokens.get("input", 0) or 0),
                "cached_tokens": int(tokens.get("cache_read", 0) or 0),
                "output_tokens": int(tokens.get("output", 0) or 0),
                "total_tokens": total_tokens(tokens, provider),
            })
    return out


def _fmt(value, kind: str = "num") -> str:
    if value is None:
        return "—"
    if kind == "pct":
        return f"{value * 100:.0f}%"
    if kind == "signed_pct":
        return f"{value:+.0f}%"
    if kind == "signed_pp":
        return f"{value * 100:+.0f} pp"
    if isinstance(value, float) and not value.is_integer():
        return f"{value:,.1f}"
    return f"{int(value):,}"


def render_markdown(manifest: dict, summaries: list[dict], deltas: list[dict], rows: list[dict],
                    failure_counts: dict[str, Counter]) -> str:
    lines = [
        f"# ผล ablation — {manifest['session_id']}",
        "",
        f"- commit `{manifest['git_commit']}`, provider `{manifest['provider']}` / model `{manifest['model']}`, "
        f"max_steps={manifest['max_steps']}{' — **DRY RUN (stub agent)**' if manifest['dry_run'] else ''}",
    ]
    for session in manifest.get("sessions") or [manifest]:
        first_rep = session.get("first_rep", 1)
        last_rep = first_rep + session["reps"] - 1
        lines.append(
            f"- session `{session['session_id']}` @ `{session['git_commit']}`: {len(session['tasks'])} task × "
            f"arm {', '.join(session['arms'])} × rep {first_rep}{f'–{last_rep}' if last_rep != first_rep else ''}"
            f", seed={session['seed']}"
        )
    # session เดียวไม่มี key นี้ = ไม่มีอะไรต้องเทียบ; None = รวมหลาย session แต่ถาม git ไม่ได้
    changes = manifest.get("measured_code_changes", [])
    if changes is None:
        lines.append("- ⚠️ เทียบโค้ดระหว่าง commit ของแต่ละ session ไม่ได้ (git ไม่ตอบ)")
    elif changes:
        lines.append(f"- ⚠️ โค้ด agent/target ต่างกันระหว่าง session: {', '.join(changes)}")
    elif len(manifest.get("sessions") or []) > 1 and "," in manifest["git_commit"]:
        lines.append("- commit ต่างกันแต่โค้ด agent/target เหมือนกัน (เปลี่ยนแค่ตัวรัน/ตัววิเคราะห์)")
    if manifest.get("uncommitted_files"):
        lines.append(f"- ⚠️ มีไฟล์ยังไม่ commit ตอนรัน: {', '.join(manifest['uncommitted_files'])}")
    if rows and not any(r["has_telemetry"] for r in rows):
        lines.append("- ไม่มี telemetry (dry-run หรือไฟล์ถูกย้าย) — llm_calls/เวลาแยกช่วง/failure_class ว่าง")
    lines += [
        "",
        "## สรุปราย arm",
        "",
        "| Arm | valid (infra) | ผ่าน (DB) | success rate [95% CI] | agent บอกสำเร็จ | false success | "
        "false failure | median steps | median LLM calls | median tokens | cache share | median เวลา (s) | "
        "p95 เวลา (s) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        ci = f" [{_fmt(s['ci_low'], 'pct')}–{_fmt(s['ci_high'], 'pct')}]" if s["valid_attempts"] else ""
        lines.append(
            f"| {s['arm']} | {s['valid_attempts']} ({s['infra_failures']}) | {s['passed']} | "
            f"{_fmt(s['success_rate'], 'pct')}{ci} | {s['claimed_success']} | {s['false_success']} | "
            f"{s['false_failure']} | "
            f"{_fmt(s['median_steps'])} | {_fmt(s['median_llm_calls'])} | {_fmt(s['median_total_tokens'])} | "
            f"{_fmt(s['cache_share'], 'pct')} | {_fmt(s['median_duration_seconds'])} | "
            f"{_fmt(s['p95_duration_seconds'])} |"
        )
    reference = manifest["reference_arm"]
    if deltas:
        lines += [
            "",
            f"## เทียบกับ {reference} แบบจับคู่ราย task",
            "",
            "ค่าบวก = มากกว่า reference (tokens/steps/เวลา มากขึ้น = แย่ลง)",
            "",
            "| Arm | task ที่จับคู่ได้ | ส่วนต่าง success rate | median Δ tokens | median Δ steps | median Δ เวลา |",
            "|---|---|---|---|---|---|",
        ]
        for d in deltas:
            lines.append(
                f"| {d['arm']} | {d['paired_tasks']} | {_fmt(d['mean_success_rate_diff'], 'signed_pp')} | "
                f"{_fmt(d['median_pct_change_total_tokens'], 'signed_pct')} | "
                f"{_fmt(d['median_pct_change_steps'], 'signed_pct')} | "
                f"{_fmt(d['median_pct_change_duration_seconds'], 'signed_pct')} |"
            )

    tasks = list(manifest["tasks"])
    arms = list(manifest["arms"])
    cell: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for row in rows:
        cell[(row["task_id"], row["arm"])].append(row["passed"])
    lines += ["", "## ผ่าน/valid ราย task", "", "| task | " + " | ".join(arms) + " |",
              "|---|" + "---|" * len(arms)]
    for task_id in tasks:
        parts = []
        for arm in arms:
            results = cell.get((task_id, arm), [])
            parts.append(f"{sum(results)}/{len(results)}" if results else "—")
        lines.append(f"| {task_id} | " + " | ".join(parts) + " |")

    if any(failure_counts.values()):
        classes = sorted({c for counter in failure_counts.values() for c in counter})
        lines += ["", "## failure_class ราย step (ไม่นับ ok)", "", "| Arm | " + " | ".join(classes) + " |",
                  "|---|" + "---|" * len(classes)]
        for arm in arms:
            counter = failure_counts.get(arm, Counter())
            lines.append(f"| {arm} | " + " | ".join(str(counter.get(c, 0)) for c in classes) + " |")

    lines += [
        "",
        "## การตีความ",
        "",
        "- success rate มาจาก DB หลัง reset fixture ไม่ใช่จากคำพูด agent; false success = agent บอกสำเร็จแต่ DB บอกไม่,"
        " false failure = agent บอกไม่สำเร็จแต่ DB บอกสำเร็จ",
        "- infra = provider ล่ม (429/500/โมเดลถูกปฏิเสธ) ไม่ว่าจะเกิดที่ step ไหน — ไม่นับเป็น sample ของ arm ใด",
        "- benchmark นี้ผันผวน (commit เดียวกันเคยได้ 12/15 และ 15/15) — ส่วนต่าง success rate ที่ CI ยังซ้อนกัน"
        " ให้ถือว่าแยกไม่ออกจาก noise",
        "- arm ที่มีจำนวน valid น้อยกว่า (รันแค่บาง rep) มี CI กว้างกว่า — ดูคอลัมน์ valid ก่อนเทียบ",
        f"- tokens = {'input + output (input ของ openai รวม cached แล้ว)' if manifest['provider'] in _INPUT_INCLUDES_CACHED else 'input + output + cache_read + cache_creation'}",
    ]
    return "\n".join(lines) + "\n"


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:  # utf-8-sig ให้ Excel เปิดภาษาไทยได้
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def analyze(results_paths: Path | list[Path], token_usage_path: Path | None = None,
            step_trace_path: Path | None = None, curve_task: str | None = None,
            out_stem: Path | None = None) -> dict:
    if isinstance(results_paths, Path):
        results_paths = [results_paths]
    manifest, records = load_sessions(results_paths)
    provider = manifest["provider"]
    telemetry_ids = {r["telemetry_task_id"] for r in records if r.get("telemetry_task_id")}
    usage, steps = load_telemetry(
        telemetry_ids,
        token_usage_path or Path(settings.token_usage_log_path),
        step_trace_path or Path(settings.step_trace_log_path),
    )
    rows = enrich(records, provider, usage, steps)
    arms = list(manifest["arms"])
    invalid_counts = Counter(r["arm"] for r in records if not r["valid"])
    summaries = summarize_arms(rows, arms, invalid_counts)
    deltas = paired_deltas(rows, arms, manifest["reference_arm"])
    failure_counts: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        failure_counts[row["arm"]].update(row["failure_classes"])
    curve = token_curve(records, steps, provider)
    if curve_task:
        curve = [c for c in curve if c["task_id"] == curve_task]

    if out_stem is not None:
        stem = out_stem
    elif len(results_paths) == 1:
        stem = results_paths[0].with_suffix("")
    else:
        stem = Path(f"{results_paths[0].with_suffix('')}_combined")
    report_path = Path(f"{stem}_report.md")
    report_path.write_text(render_markdown(manifest, summaries, deltas, rows, failure_counts), encoding="utf-8")
    _write_csv(Path(f"{stem}_arms.csv"), summaries)
    _write_csv(Path(f"{stem}_deltas.csv"), deltas)
    _write_csv(Path(f"{stem}_attempts.csv"),
               [{**r, "failure_classes": json.dumps(dict(r["failure_classes"]))} for r in rows])
    _write_csv(Path(f"{stem}_token_curve.csv"), curve)
    _write_csv(Path(f"{stem}_failure_classes.csv"),
               [{"arm": arm, "failure_class": cls, "count": count}
                for arm, counter in failure_counts.items() for cls, count in counter.items()])
    return {"report": report_path, "summaries": summaries, "deltas": deltas, "rows": rows}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="สรุปผล run_ablation.py")
    parser.add_argument("results", nargs="+", help="path ของ <session>.jsonl (หลายไฟล์ = รวมเป็นการทดลองเดียว)")
    parser.add_argument("--curve-task", default=None, help="กรองกราฟ token ต่อ step ให้เหลือ task เดียว")
    parser.add_argument("--token-usage", default=None, help=f"default {settings.token_usage_log_path}")
    parser.add_argument("--step-trace", default=None, help=f"default {settings.step_trace_log_path}")
    parser.add_argument("--out", default=None,
                        help="prefix ของไฟล์ผล (default: <ไฟล์แรก> หรือ <ไฟล์แรก>_combined เมื่อหลายไฟล์)")
    args = parser.parse_args(argv)
    result = analyze(
        [Path(path) for path in args.results],
        Path(args.token_usage) if args.token_usage else None,
        Path(args.step_trace) if args.step_trace else None,
        args.curve_task,
        Path(args.out) if args.out else None,
    )
    print(result["report"].read_text(encoding="utf-8"))
    print(f"ไฟล์: {result['report']} (+ _arms/_deltas/_attempts/_token_curve/_failure_classes .csv)")


if __name__ == "__main__":
    main()

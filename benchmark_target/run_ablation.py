"""W_ablation: controlled benchmark + ablation สำหรับรายงาน (แผนเต็ม: docs/benchmark_ablation_plan.md)

รัน task ชุดเดียวกันบน benchmark_target ภายใต้หลาย arm ที่ปิดฟีเจอร์บางกลุ่มด้วย mock.patch — ไม่แก้
orchestrator.py ทุกจุดที่ปิดเป็นชื่อระดับโมดูลหรือค่าใน settings ที่ถูกอ่านตอนรัน (patch.object ล้มทันที
ถ้าชื่อหายไป จึงรู้ตัวเมื่อโค้ดเปลี่ยน) ผลตัดสินจาก DB ผ่าน run_attempt() เหมือน hrm_local_eval และเก็บ
คำพูดของ agent แยกไว้นับ false success (agent บอกสำเร็จ แต่ DB บอกไม่)

ลำดับรัน: for rep: for task: for arm in shuffle(arms) — ไม่รัน arm เดียวต่อกันยาวๆ ไม่งั้นช่วงเวลาและสถานะ
โควต้าไปผูกกับ arm ใด arm หนึ่ง ทุก rep จึงเป็นบล็อกที่สมดุล หยุดกลางทางก็ยังวิเคราะห์ได้ และ --resume
สร้าง schedule เดิมจาก seed ใน manifest แล้วข้ามช่องที่ได้ผลใช้ได้แล้ว

Usage:
    python benchmark_target/run_ablation.py --dry-run                 # stub agent ไม่ใช้ LLM/browser
    python benchmark_target/run_ablation.py --arms FULL,BASELINE --reps 3
    python benchmark_target/run_ablation.py --resume data/ablation/<session>.jsonl
    python benchmark_target/analyze_ablation.py data/ablation/<session>.jsonl
"""

import argparse
import json
import random
import subprocess
import sys
import time
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from backend.app.config import settings  # noqa: E402
from backend.app.core import llm, orchestrator  # noqa: E402
from backend.app.core.release_gate import _git_commit_short  # noqa: E402
from benchmark_target.analyze_ablation import is_provider_failure  # noqa: E402

DEFAULT_OUT_DIR = REPO_ROOT / "data" / "ablation"

# 6 ตัวแรก = HRM_LOCAL_TASK_IDS (ผ่าน 6/6 เมื่อ 2026-10-14) + 4 ตัวที่ actor เดียว ค่าในเป้าหมายระบุชัด
# และยาวพอให้ compaction ทำงาน (เกิน _COMPACT_AFTER_STEPS) — เหตุผลที่ตัด task อื่นอยู่ใน docs ข้อ 1
DEFAULT_TASK_IDS = (
    "PIM-EDIT-VERIFY-01",
    "PIM-CREATE-EMPLOYEE",
    "PIM-DEACTIVATE-EMPLOYEE",
    "ADMIN-CREATE-USER-ACCOUNT",
    "LEAVE-APPLY-01",
    "RECRUITMENT-SHORTLIST-01",
    "PIM-REACTIVATE-EMPLOYEE",
    "TIME-CREATE-SUBMIT-TIMESHEET",
    "TIME-APPROVE-BY-SUPERVISOR",
    "RECRUITMENT-FULL-HIRE-WORKFLOW-02",
)

# http_redirect/download_artifact คืน INCONCLUSIVE เสมอใน attempt_runner — นับเป็น "ไม่ผ่าน" จะทำให้ตัวเลขผิด
_SUPPORTED_VERIFICATION_TYPES = {"custom", "state_diff"}

ARM_NAMES = ("FULL", "NO_CONTEXT", "NO_PROMPT_SECTIONS", "NO_VERIFY", "BASELINE")
REFERENCE_ARM = "FULL"

# ทุกตัวถูกใช้เป็น `count < _MAX_...` ในบล็อก finish_task ของ run_task() — ตั้ง 0 = ไม่ตีกลับเลย
_VERIFY_RETRY_CONSTANTS = (
    "_MAX_PREMATURE_TRUE_FINISH_RETRIES",
    "_MAX_PREMATURE_FALSE_FINISH_RETRIES",
    "_MAX_PREMATURE_ALL_FAILED_RETRIES",
    "_MAX_PREMATURE_VALIDATION_ERROR_RETRIES",
    "_MAX_PREMATURE_DELETION_INCOMPLETE_RETRIES",
    "_MAX_DELETE_ALL_UNVERIFIED_RETRIES",
    "_MAX_PREMATURE_TABLE_VERIFY_RETRIES",
    "_MAX_COUNT_ANSWER_MISMATCH_RETRIES",
)

_UNLIMITED = 10**9


async def _table_item_always_found(page, verify_text: str) -> bool:  # noqa: ARG001
    """W63[7.2] ตั้ง _MAX_PREMATURE_TABLE_VERIFY_RETRIES=0 อย่างเดียวไม่พอ — retry หมดแล้วหาไม่เจอ
    run_task() บังคับ claimed_success=False ซึ่งยังเป็นการ verify อยู่ ต้องให้การสแกนตารางผ่านเสมอด้วย"""
    return True


# capture เขียน template แล้วยิง LLM (abstractor) แบบ background หลัง task สำเร็จ — กินโควต้าและทำให้
# attempt ถัดไปโดน throttle ได้ ไม่ใช่สิ่งที่วัด จึงปิดเท่ากันทุก arm (replay ปิดอยู่แล้วตาม config)
COMMON_PATCHES = [(settings, "enable_procedural_memory_capture", False)]

_CONTEXT_PATCHES = [
    (orchestrator, "_COMPACT_AFTER_STEPS", _UNLIMITED),
    (orchestrator, "_compact_stale_user_turns", lambda messages, goal, **_: (messages, 0)),
    (orchestrator, "_dedupe_stale_snapshots", lambda messages, **_: messages),
    # W_token_cut W7 — smoke 2026-10-22 พบว่าตัวนี้ยังทำงานใน NO_CONTEXT (gated_deref_events=19) จนกว่าจะ patch
    (orchestrator, "_dedupe_stale_gated", lambda messages, **_: (messages, 0)),
    (settings, "snapshot_max_elements", _UNLIMITED),
    (settings, "read_page_data_max_rows", _UNLIMITED),
]
_PROMPT_SECTION_PATCHES = [
    (orchestrator, "_resolve_prompt_sections", lambda previous, **_: llm.ALL_PROMPT_SECTIONS),
]
_VERIFY_PATCHES = [(orchestrator, name, 0) for name in _VERIFY_RETRY_CONSTANTS] + [
    (orchestrator, "_scan_created_item_in_table", _table_item_always_found),
]

ARM_PATCHES = {
    "FULL": [],
    "NO_CONTEXT": _CONTEXT_PATCHES,
    "NO_PROMPT_SECTIONS": _PROMPT_SECTION_PATCHES,
    "NO_VERIFY": _VERIFY_PATCHES,
    "BASELINE": _CONTEXT_PATCHES + _PROMPT_SECTION_PATCHES + _VERIFY_PATCHES,
}


@contextmanager
def arm_patches(arm: str):
    with ExitStack() as stack:
        for target, attribute, value in COMMON_PATCHES + ARM_PATCHES[arm]:
            stack.enter_context(mock.patch.object(target, attribute, value))
        yield


def build_schedule(task_ids: list[str], arms: list[str], reps: int, seed: int) -> list[tuple[int, str, str]]:
    rng = random.Random(seed)
    cells = []
    for rep in range(1, reps + 1):
        for task_id in task_ids:
            order = list(arms)
            rng.shuffle(order)
            cells.extend((rep, task_id, arm) for arm in order)
    return cells


def load_selected_tasks(task_ids: list[str]) -> dict[str, dict]:
    from benchmark_target.run_benchmark import load_tasks

    by_id = {t["task_id"]: t for t in load_tasks(["L1", "L2", "L3", "L4", "L5"])}
    missing = [task_id for task_id in task_ids if task_id not in by_id]
    if missing:
        raise SystemExit(f"task ไม่อยู่ใน catalog: {missing}")
    unsupported = [
        task_id for task_id in task_ids
        if by_id[task_id]["verification"]["type"] not in _SUPPORTED_VERIFICATION_TYPES
    ]
    if unsupported:
        raise SystemExit(f"task ที่ attempt_runner ตัดสินจาก DB ไม่ได้ (ได้ INCONCLUSIVE เสมอ): {unsupported}")
    return {task_id: by_id[task_id] for task_id in task_ids}


def _uncommitted_source_files() -> list[str]:
    """ไฟล์ที่ยังไม่ commit นอก data/ — data/*.jsonl ถูก track และโตทุกรันอยู่แล้ว ไม่นับ"""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT,
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout
    except Exception:
        return []
    paths = [line[3:] for line in out.splitlines() if line.strip()]
    return [p for p in paths if not p.startswith(("data/", "benchmark_target/data/"))]


def _make_adapter(dry_run: bool, provider: str | None, max_steps: int, headless: bool, run_id: str | None):
    from benchmark_target.runner.agent_adapters import AgentRunResult, HermesAgentAdapter, StubAgentAdapter

    if dry_run:
        return StubAgentAdapter([AgentRunResult(success=True, steps=0, message="stub: no action taken", tokens={})])
    return HermesAgentAdapter(provider=provider, max_steps=max_steps, headless=headless, run_id=run_id)


def provider_failed(agent_result) -> bool:
    """run_task() ไม่ raise เมื่อ provider ล่ม — คืน success=False พร้อมข้อความ error attempt_runner
    จึงเห็นเป็น FUNCTIONAL_FAIL ปกติ (smoke 2026-10-22: codex ตอบ 400 "gpt-5.5 not supported" ทั้ง 5 arm
    ถูกนับเป็นความผิดของ agent) เกณฑ์อยู่ที่ analyze_ablation.is_provider_failure ให้สองไฟล์ตรงกัน"""
    if agent_result is None:
        return False
    return is_provider_failure(agent_result.success, agent_result.steps, agent_result.message)


def attempt_record(*, session_id: str, arm: str, task: dict, rep: int, try_index: int,
                   telemetry_run_id: str | None, attempt, agent_result) -> dict:
    provider_failure = provider_failed(agent_result)
    return {
        "session_id": session_id,
        "arm": arm,
        "task_id": task["task_id"],
        "task_revision": task["revision"],
        "level": task.get("level"),
        "rep": rep,
        "try": try_index,
        # HermesAgentAdapter เขียน telemetry ด้วย task_id = f"{run_id}-{task_id}" — เก็บไว้ join ตอนวิเคราะห์
        "telemetry_task_id": f"{telemetry_run_id}-{task['task_id']}" if telemetry_run_id else None,
        "valid": not attempt.is_infra_failure() and not provider_failure,
        "provider_failure": provider_failure,
        "passed": bool(attempt.passed),
        # None = agent ไม่ได้เริ่ม (fixture/precondition/infra ล้มก่อน)
        "claimed_success": bool(agent_result.success) if agent_result is not None else None,
        "failure_class": attempt.failure_class.value if attempt.failure_class else None,
        "state": attempt.state.value,
        "detail": (attempt.detail or "")[:500],
        "agent_message": (getattr(agent_result, "message", "") or "")[:500],
        "steps": attempt.steps,
        "tokens": attempt.tokens or {},
        "approval_count": getattr(agent_result, "approval_count", 0) if agent_result is not None else 0,
        "duration_seconds": round(attempt.duration_ms / 1000.0, 2),
        "started_at": attempt.started_at,
        "finished_at": attempt.finished_at,
    }


def _append_jsonl(path: Path, record: dict) -> None:
    # เขียนทีละ attempt — Ctrl+C/โควต้าหมดกลางทาง ผลที่ได้แล้วไม่หาย และ --resume อ่านต่อได้
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _load_resume(results_path: Path) -> tuple[dict, set, dict]:
    manifest_path = results_path.with_suffix(".manifest.json")
    if not manifest_path.exists():
        raise SystemExit(f"ไม่พบ manifest ของ session: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    done: set[tuple[str, str, int]] = set()
    tries_made: dict[tuple[str, str, int], int] = {}
    if results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row["arm"], row["task_id"], row["rep"])
            tries_made[key] = max(tries_made.get(key, 0), row["try"])
            # ไฟล์ที่เขียนก่อนเกณฑ์ปัจจุบันอาจบอก valid ทั้งที่ provider ล่มกลาง task — ช่องนั้นต้องรันใหม่
            reclassified = row["claimed_success"] is not None and is_provider_failure(
                row["claimed_success"], row["steps"], row["agent_message"],
            )
            if row["valid"] and not reclassified:
                done.add(key)
    return manifest, done, tries_made


def _parse_list(value: str | None, default: tuple[str, ...]) -> list[str]:
    if not value:
        return list(default)
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Controlled benchmark + ablation บน benchmark_target")
    parser.add_argument("--arms", default=None, help=f"comma list จาก {','.join(ARM_NAMES)} (default ทั้งหมด)")
    parser.add_argument("--tasks", default=None, help="comma list ของ task_id (default 10 ตัวตามแผน)")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0, help="seed ของการสลับลำดับ arm")
    parser.add_argument("--provider", default=None, help="openai/anthropic/gemini (default = settings.llm_provider)")
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--max-infra-retries", type=int, default=2,
                        help="รันซ้ำช่องที่ล้มเพราะ infra ได้อีกกี่ครั้งก่อนข้าม")
    parser.add_argument("--max-consecutive-infra", type=int, default=5,
                        help="หยุด session เมื่อ infra ล้มติดกันครบจำนวนนี้ (รันต่อด้วย --resume)")
    parser.add_argument("--delay", type=float, default=settings.eval_task_delay_seconds,
                        help="วินาทีที่เว้นระหว่าง attempt")
    parser.add_argument("--headed", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="StubAgentAdapter — ไม่ใช้ LLM/browser")
    parser.add_argument("--allow-dirty", action="store_true", help="ยอมรันจริงทั้งที่มีไฟล์ยังไม่ commit")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--resume", default=None, help="path ของ <session>.jsonl ที่จะรันต่อ")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)

    if args.resume:
        results_path = Path(args.resume)
        manifest, done, tries_made = _load_resume(results_path)
        current_model = getattr(settings, f"{manifest['provider']}_model", "unknown")
        if not manifest["dry_run"] and current_model != manifest["model"]:
            # adapter อ่านโมเดลจาก settings ตอนรัน ไม่ใช่จาก manifest — resume หลังเปลี่ยนโมเดลจะปนสองโมเดล
            # ใน session เดียวโดยที่ manifest ยังบอกโมเดลเดิม
            raise SystemExit(
                f"โมเดลเปลี่ยนจาก {manifest['model']!r} เป็น {current_model!r} — เริ่ม session ใหม่แทน --resume"
            )
        print(f"resume {manifest['session_id']}: ใช้ arms/tasks/reps/seed จาก manifest, เสร็จแล้ว {len(done)} ช่อง")
    else:
        arms = _parse_list(args.arms, ARM_NAMES)
        unknown = [arm for arm in arms if arm not in ARM_PATCHES]
        if unknown:
            raise SystemExit(f"ไม่รู้จัก arm: {unknown} (มี {list(ARM_NAMES)})")
        # .env ในเครื่องตั้ง LLM_PROVIDER=gemini ทั้งที่ default ของ config คือ openai — รันจริงต้องเลือกเอง
        # ไม่งั้นผลทั้ง session ผูกกับ provider ที่ไม่ได้ตั้งใจ (และ gemini free tier โควต้าหมดกลางทางได้)
        if not args.dry_run and not args.provider:
            raise SystemExit(f"รันจริงต้องใส่ --provider (ค่าใน settings ตอนนี้คือ {settings.llm_provider!r})")
        provider = args.provider or settings.llm_provider
        session_id = f"abl-{'dry-' if args.dry_run else ''}{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        manifest = {
            "session_id": session_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_commit_short(),
            "uncommitted_files": _uncommitted_source_files(),
            "dry_run": args.dry_run,
            "provider": provider,
            "model": getattr(settings, f"{provider}_model", "unknown"),
            "arms": arms,
            "reference_arm": REFERENCE_ARM if REFERENCE_ARM in arms else arms[0],
            "tasks": _parse_list(args.tasks, DEFAULT_TASK_IDS),
            "reps": args.reps,
            "seed": args.seed,
            "max_steps": args.max_steps,
            "step_pacing_delay_seconds": settings.step_pacing_delay_seconds,
        }
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        results_path = out_dir / f"{session_id}.jsonl"
        results_path.with_suffix(".manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8",
        )
        done, tries_made = set(), {}

    dry_run = manifest["dry_run"]
    if not dry_run and manifest["uncommitted_files"] and not args.allow_dirty:
        raise SystemExit(
            "มีไฟล์ยังไม่ commit — ผลต้องผูกกับ commit เดียว (ใส่ --allow-dirty ถ้าตั้งใจ):\n  "
            + "\n  ".join(manifest["uncommitted_files"])
        )

    tasks_by_id = load_selected_tasks(manifest["tasks"])
    schedule = build_schedule(manifest["tasks"], manifest["arms"], manifest["reps"], manifest["seed"])

    from fastapi.testclient import TestClient

    from backend.app.core.hrm_local_eval import _RecordingAdapter, ensure_target_running
    from backend.app.permission.rules import allow_internal_navigation_here
    from benchmark_target.control.main import app as control_app
    from benchmark_target.runner.attempt_runner import run_attempt

    if not dry_run:
        ensure_target_running()

    print(f"session {manifest['session_id']} — {len(schedule)} ช่อง "
          f"({len(manifest['tasks'])} task x {len(manifest['arms'])} arm x {manifest['reps']} rep) -> {results_path}")
    attempts_made = 0
    consecutive_infra = 0
    # SSRF guard บล็อก localhost เป็นค่าเริ่มต้น — เปิดเฉพาะ context นี้ เหมือน hrm_local_eval._run_sync()
    with allow_internal_navigation_here(), TestClient(control_app) as control_client:
        for position, (rep, task_id, arm) in enumerate(schedule, 1):
            key = (arm, task_id, rep)
            if key in done:
                continue
            task = tasks_by_id[task_id]
            first_try = tries_made.get(key, 0) + 1
            for try_index in range(first_try, first_try + args.max_infra_retries + 1):
                if attempts_made and args.delay > 0:
                    time.sleep(args.delay)
                telemetry_run_id = None if dry_run else f"{manifest['session_id']}-{arm}-r{rep}-t{try_index}"
                recording = _RecordingAdapter(
                    _make_adapter(dry_run, manifest["provider"], manifest["max_steps"], not args.headed, telemetry_run_id)
                )
                with arm_patches(arm):
                    attempt = run_attempt(task, recording, control_client)
                attempts_made += 1
                record = attempt_record(
                    session_id=manifest["session_id"], arm=arm, task=task, rep=rep, try_index=try_index,
                    telemetry_run_id=telemetry_run_id, attempt=attempt, agent_result=recording.last,
                )
                _append_jsonl(results_path, record)
                verdict = "PASS" if record["passed"] else ("INFRA" if not record["valid"] else "FAIL")
                print(f"  [{position}/{len(schedule)}] r{rep} {task_id:36s} {arm:19s} {verdict:5s} "
                      f"claimed={record['claimed_success']} steps={record['steps']} "
                      f"{record['duration_seconds']:.1f}s "
                      f"{'' if record['valid'] else (record['agent_message'] if record['provider_failure'] else record['detail'])[:100]}")
                if record["valid"]:
                    consecutive_infra = 0
                    break
                # provider ล่มทั้งรัน (โมเดลถูกเลิกรับ/โควตาหมด) — ไล่ทุกช่องต่อก็ได้แต่แถว infra ไม่ต้องเผาเวลา
                # แก้ต้นเหตุแล้ว --resume ช่องที่เหลือ (เทียบ run_is_invalid ของ release gate)
                consecutive_infra += 1
                if consecutive_infra >= args.max_consecutive_infra:
                    raise SystemExit(
                        f"infra ล้มติดกัน {consecutive_infra} ครั้ง — หยุด session ไว้ก่อน ล่าสุด: "
                        f"{(record['agent_message'] or record['detail'])[:200]}\n"
                        f"แก้แล้วรันต่อ: python benchmark_target/run_ablation.py --resume {results_path}"
                    )

    print(f"\nผลอยู่ที่ {results_path}\nวิเคราะห์: python benchmark_target/analyze_ablation.py {results_path}")
    return results_path


if __name__ == "__main__":
    main()

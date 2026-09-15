"""Run Engine entrypoint — ties queue + catalog + stability sampling + artifacts together
against a REAL running Control Plane over HTTP (not TestClient — this is the actual usage
path, unlike test_runner.py's in-process tests).

Usage:
    python benchmark_target/run_control.py &        # start the Control Plane first
    python benchmark_target/run_benchmark.py --level L1 --n 3

Defaults to StubAgentAdapter (always reports success, does nothing to the browser) — this
proves the Run Engine's wiring against a live Control Plane without needing an LLM key or
a real browser. Every task will come back FUNCTIONAL_FAIL (or, for a handful of read-only
tasks whose verification is tautologically true regardless of agent action, PASS) — the
stub took no action, and this engine trusts authoritative DB state over the agent's
self-report (spec 16).

Pass --hermes-provider to run the REAL Orchestrator instead (Phase 7 — live browser
validation). This spends real LLM API/OAuth quota and launches a real headless Chromium
per attempt: scope --level/--task-id narrowly and keep --n small. Verified live (2026-09):
python run_benchmark.py --hermes-provider --task-id AUTH-LOGIN-SUCCESS-ADMIN --n 1
    -> real agent logged in, reached dashboard, PASS in 5 steps / ~58s / ~51k tokens
python run_benchmark.py --hermes-provider --task-id PIM-EDIT-VERIFY-01 --n 1
    -> real agent reported success in 4 steps, but the DB shows the field was never saved
       -> correctly reported FUNCTIONAL_FAIL: exactly the class of gap this benchmark
       exists to catch (agent self-report says "done", authoritative state disagrees).
"""

import argparse
import sys
import uuid
from pathlib import Path

import httpx
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark_target.runner.agent_adapters import AgentRunResult, HermesAgentAdapter, StubAgentAdapter
from benchmark_target.runner.artifacts import write_run_artifacts
from benchmark_target.runner.models import RunState
from benchmark_target.runner.queue import RunManager
from benchmark_target.runner.stability_runner import run_stability

CATALOG_TASKS_DIR = Path(__file__).resolve().parent / "catalog" / "v1" / "tasks"


def load_tasks(levels: list[str]) -> list[dict]:
    tasks = []
    for level in levels:
        path = CATALOG_TASKS_DIR / f"{level}.yaml"
        if path.exists():
            tasks.extend(yaml.safe_load(path.read_text()) or [])
    return tasks


def main():
    parser = argparse.ArgumentParser(description="Run the Hermes HRM benchmark catalog")
    parser.add_argument("--level", action="append", default=None, help="Catalog level(s) to run, e.g. L1 (repeatable)")
    parser.add_argument("--task-id", default=None, help="Run a single task_id only")
    parser.add_argument("--n", type=int, default=3, help="Stability samples per task (spec default is 5)")
    parser.add_argument("--control-url", default="http://127.0.0.1:8101")
    parser.add_argument("--priority", default="NORMAL", choices=["HIGH", "NORMAL", "LOW"])
    parser.add_argument(
        "--hermes-provider", nargs="?", const="__default__", default=None, metavar="PROVIDER",
        help='Use the real Hermes Orchestrator instead of the stub (e.g. "openai", "anthropic", '
             '"gemini"; omit provider name after the flag to use the configured default). '
             "Spends real LLM API/quota usage and launches a real (headless) browser per attempt "
             "— costs scale with --n and the number of tasks selected, so scope --level/--task-id "
             "narrowly when using this.",
    )
    parser.add_argument("--max-steps", type=int, default=20, help="Per-attempt step budget for --hermes-provider")
    parser.add_argument("--headed", action="store_true", help="Show the browser window (--hermes-provider only)")
    args = parser.parse_args()

    levels = args.level or ["L1", "L2", "L3", "L4", "L5"]
    tasks = load_tasks(levels)
    if args.task_id:
        tasks = [t for t in tasks if t["task_id"] == args.task_id]
    if not tasks:
        print("No matching tasks found.")
        sys.exit(1)

    manager = RunManager()
    run_id = uuid.uuid4().hex[:12]
    manager.submit(run_id, args.priority)
    record = manager.start_next()
    assert record is not None and record.run_id == run_id

    manager.transition(run_id, RunState.RUNNING)

    control_client = httpx.Client(base_url=args.control_url, timeout=30.0)
    if args.hermes_provider is not None:
        provider = None if args.hermes_provider == "__default__" else args.hermes_provider
        adapter = HermesAgentAdapter(provider=provider, max_steps=args.max_steps, headless=not args.headed)
        agent_profile = f"hermes:{provider or 'default'}"
    else:
        adapter = StubAgentAdapter([AgentRunResult(success=True, steps=0, message="stub: no action taken", tokens={})])
        agent_profile = "stub"

    print(f"Run {run_id}: {len(tasks)} task(s), n={args.n}, control-plane={args.control_url}")
    results = []
    for task in tasks:
        result = run_stability(task, adapter, control_client, n=args.n)
        results.append(result)
        print(f"  {task['task_id']:55s} {result.functional_result.value:12s} "
              f"{result.execution_health.value:12s} "
              f"({result.valid_sample_count}/{result.total_attempt_count} attempts) {result.reason}")

    manager.transition(run_id, RunState.COMPLETING)
    manager.transition(run_id, RunState.COMPLETED)

    run_dir = write_run_artifacts(
        run_id, results,
        manifest={
            "catalog_version": "v1",
            "levels": levels,
            "agent_profile": agent_profile,
            "stability_n": args.n,
        },
    )
    print(f"\nArtifacts written to {run_dir}")


if __name__ == "__main__":
    main()

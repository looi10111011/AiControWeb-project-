"""Run artifacts (spec 48-49): run_manifest.json, task_results.json, attempt_results.json,
metrics.json, written under data/runs/{run_id}/. Screenshots/video/downloads are optional
per spec 42 and not produced here — this engine's own tests don't drive a real browser
(Phase 7 does), so there is nothing to capture yet.
"""

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from benchmark_target.runner.models import AttemptResult, StabilityResult

RUNS_DIR = Path(__file__).resolve().parents[1] / "data" / "runs"


def _attempt_to_dict(a: AttemptResult) -> dict:
    d = asdict(a)
    d["state"] = a.state.value
    d["failure_class"] = a.failure_class.value if a.failure_class else None
    return d


def _stability_to_task_result(s: StabilityResult) -> dict:
    return {
        "task_id": s.task_id,
        "task_revision": s.task_revision,
        "functional_result": s.functional_result.value,
        "execution_health": s.execution_health.value,
        "success_rate": s.success_rate,
        "confidence_interval": list(s.confidence_interval) if s.confidence_interval else None,
        "valid_sample_count": s.valid_sample_count,
        "total_attempt_count": s.total_attempt_count,
        "n_requested": s.n_requested,
        "reason": s.reason,
    }


def compute_metrics(stability_results: list[StabilityResult]) -> dict:
    all_attempts = [a for s in stability_results for a in s.attempts]
    passed_attempts = [a for a in all_attempts if a.passed]

    by_result = {}
    for s in stability_results:
        key = s.functional_result.value
        by_result[key] = by_result.get(key, 0) + 1

    def _avg(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    return {
        "task_count": len(stability_results),
        "functional_result_distribution": by_result,
        "total_attempts": len(all_attempts),
        "total_passed_attempts": len(passed_attempts),
        "avg_steps_on_pass": _avg([a.steps for a in passed_attempts]),
        "avg_duration_ms_on_pass": _avg([a.duration_ms for a in passed_attempts]),
        "avg_input_tokens_on_pass": _avg([a.tokens.get("input", 0) for a in passed_attempts if a.tokens]),
        "avg_output_tokens_on_pass": _avg([a.tokens.get("output", 0) for a in passed_attempts if a.tokens]),
    }


def write_run_artifacts(run_id: str, stability_results: list[StabilityResult], manifest: dict,
                         out_dir: Path | None = None) -> Path:
    run_dir = (out_dir or RUNS_DIR) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    full_manifest = {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        **manifest,
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(full_manifest, indent=2, default=str))

    task_results = [_stability_to_task_result(s) for s in stability_results]
    (run_dir / "task_results.json").write_text(json.dumps(task_results, indent=2, default=str))

    attempt_results = [_attempt_to_dict(a) for s in stability_results for a in s.attempts]
    (run_dir / "attempt_results.json").write_text(json.dumps(attempt_results, indent=2, default=str))

    metrics = compute_metrics(stability_results)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))

    return run_dir

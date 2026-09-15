"""Stability sampling (spec 30-32): N valid functional samples, auto-fill through infra
failures up to a hard cap, deterministic classification, Wilson interval as a diagnostic
only. This is where the error-class-aware retry policy (spec 28) actually shows up in
practice — an infra-class failure here is not "attempt failed", it's "this sample didn't
count, try again fresh" (attempt_runner.run_attempt already does a full reset/fixture/
precondition cycle per call, so "fresh" is automatic, not something this function manages).
"""

from benchmark_target.runner.attempt_runner import run_attempt
from benchmark_target.runner.models import ExecutionHealth, FunctionalResult, StabilityResult
from benchmark_target.runner.stats import wilson_interval

DEFAULT_N = 5
MIN_VALID_FOR_CLASSIFICATION = 3


def run_stability(task: dict, agent_adapter, control_client, n: int = DEFAULT_N,
                   max_total_attempts: int | None = None) -> StabilityResult:
    max_total = max_total_attempts if max_total_attempts is not None else 2 * n

    all_attempts = []
    valid_attempts = []
    total_made = 0

    while len(valid_attempts) < n and total_made < max_total:
        result = run_attempt(task, agent_adapter, control_client)
        all_attempts.append(result)
        total_made += 1
        if not result.is_infra_failure():
            valid_attempts.append(result)

    valid_count = len(valid_attempts)
    infra_events = total_made - valid_count

    if valid_count < MIN_VALID_FOR_CLASSIFICATION:
        functional_result = FunctionalResult.INCONCLUSIVE
        reason = (
            "INSUFFICIENT_VALID_ATTEMPTS" if total_made >= max_total
            else "INSUFFICIENT_VALID_SAMPLES"
        )
        success_rate = None
        ci = None
    else:
        passed_count = sum(1 for a in valid_attempts if a.passed)
        success_rate = passed_count / valid_count
        ci = wilson_interval(passed_count, valid_count)
        if passed_count == valid_count:
            functional_result = FunctionalResult.PASS
            reason = "STABLE_PASS"
        elif passed_count == 0:
            functional_result = FunctionalResult.FAIL
            reason = "STABLE_FAIL"
        else:
            functional_result = FunctionalResult.FLAKY
            reason = "FLAKY"

    if functional_result == FunctionalResult.INCONCLUSIVE and valid_count == 0:
        execution_health = ExecutionHealth.FAILED_INFRA
    elif infra_events == 0:
        execution_health = ExecutionHealth.CLEAN
    elif valid_count >= n:
        execution_health = ExecutionHealth.RECOVERED
    else:
        execution_health = ExecutionHealth.DEGRADED

    return StabilityResult(
        task_id=task["task_id"],
        task_revision=task["revision"],
        n_requested=n,
        attempts=all_attempts,
        functional_result=functional_result,
        execution_health=execution_health,
        success_rate=success_rate,
        confidence_interval=ci,
        valid_sample_count=valid_count,
        total_attempt_count=total_made,
        reason=reason,
    )

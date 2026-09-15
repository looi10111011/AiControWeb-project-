"""Run Engine data model (spec 25-29): state machines, failure taxonomy, result model.
Plain enums + dataclasses — no framework needed for something this small.
"""

from dataclasses import dataclass, field
from enum import Enum


class RunState(str, Enum):
    QUEUED = "QUEUED"
    PREPARING = "PREPARING"
    RUNNING = "RUNNING"
    PAUSING = "PAUSING"
    PAUSED = "PAUSED"
    RESUMING = "RESUMING"
    CANCELLING = "CANCELLING"
    CANCELLED = "CANCELLED"
    INTERRUPTED = "INTERRUPTED"
    RECOVERING = "RECOVERING"
    COMPLETING = "COMPLETING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


RUN_TERMINAL_STATES = {RunState.CANCELLED, RunState.COMPLETED, RunState.FAILED}

# spec 25: "Terminal states must not transition back." Every other edge in the diagram is
# allowed; this is the one invariant worth enforcing in code rather than trusting callers.
RUN_VALID_TRANSITIONS: dict[RunState, set[RunState]] = {
    RunState.QUEUED: {RunState.PREPARING, RunState.CANCELLING},
    RunState.PREPARING: {RunState.RUNNING, RunState.FAILED, RunState.CANCELLING},
    RunState.RUNNING: {RunState.PAUSING, RunState.CANCELLING, RunState.COMPLETING,
                        RunState.INTERRUPTED, RunState.FAILED},
    RunState.PAUSING: {RunState.PAUSED, RunState.CANCELLING},
    RunState.PAUSED: {RunState.RESUMING, RunState.CANCELLING},
    RunState.RESUMING: {RunState.RUNNING, RunState.FAILED},
    RunState.CANCELLING: {RunState.CANCELLED},
    RunState.INTERRUPTED: {RunState.RECOVERING, RunState.FAILED},
    RunState.RECOVERING: {RunState.RUNNING, RunState.FAILED},
    RunState.COMPLETING: {RunState.COMPLETED, RunState.FAILED},
}


def can_transition(current: RunState, target: RunState) -> bool:
    if current in RUN_TERMINAL_STATES:
        return False
    return target in RUN_VALID_TRANSITIONS.get(current, set())


class AttemptState(str, Enum):
    PENDING = "PENDING"
    RESETTING = "RESETTING"
    FIXTURING = "FIXTURING"
    PRECONDITION_CHECK = "PRECONDITION_CHECK"
    STARTING_AGENT = "STARTING_AGENT"
    RUNNING = "RUNNING"
    VERIFYING = "VERIFYING"
    INTEGRITY_CHECK = "INTEGRITY_CHECK"
    COMPLETED = "COMPLETED"
    # failure states (spec 26)
    SETUP_ERROR = "SETUP_ERROR"
    AGENT_ERROR = "AGENT_ERROR"
    TARGET_APP_ERROR = "TARGET_APP_ERROR"
    INFRA_ERROR = "INFRA_ERROR"
    BROWSER_ERROR = "BROWSER_ERROR"
    NETWORK_ERROR = "NETWORK_ERROR"
    TIMEOUT = "TIMEOUT"
    POLICY_VIOLATION = "POLICY_VIOLATION"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"
    INTERRUPTED = "INTERRUPTED"
    RECOVERING = "RECOVERING"


class FailureClass(str, Enum):
    """spec 27 — never collapse everything into 'agent failed'."""
    FUNCTIONAL_FAIL = "FUNCTIONAL_FAIL"
    VERIFICATION_FAIL = "VERIFICATION_FAIL"
    SCENARIO_SETUP_ERROR = "SCENARIO_SETUP_ERROR"
    TARGET_APP_ERROR = "TARGET_APP_ERROR"
    INFRA_ERROR = "INFRA_ERROR"
    BROWSER_ERROR = "BROWSER_ERROR"
    NETWORK_ERROR = "NETWORK_ERROR"
    AGENT_STARTUP_ERROR = "AGENT_STARTUP_ERROR"
    POLICY_VIOLATION = "POLICY_VIOLATION"
    TIMEOUT = "TIMEOUT"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"
    INCONCLUSIVE = "INCONCLUSIVE"


# spec 28: error-class-aware retry. Missing from this set = never auto-retried.
RETRY_ELIGIBLE: set[FailureClass] = {
    FailureClass.SCENARIO_SETUP_ERROR,
    FailureClass.TARGET_APP_ERROR,
    FailureClass.INFRA_ERROR,
    FailureClass.BROWSER_ERROR,
    FailureClass.NETWORK_ERROR,
    FailureClass.AGENT_STARTUP_ERROR,
}

# Maps a FailureClass onto the AttemptState it leaves the attempt in — the two enums use
# different vocabularies (spec 26 vs 27) for the same underlying idea, so this is the one
# place that reconciles them instead of every caller guessing a name match.
FAILURE_CLASS_TO_ATTEMPT_STATE: dict[FailureClass, AttemptState] = {
    FailureClass.FUNCTIONAL_FAIL: AttemptState.VERIFYING,
    FailureClass.VERIFICATION_FAIL: AttemptState.INTEGRITY_CHECK,
    FailureClass.SCENARIO_SETUP_ERROR: AttemptState.SETUP_ERROR,
    FailureClass.TARGET_APP_ERROR: AttemptState.TARGET_APP_ERROR,
    FailureClass.INFRA_ERROR: AttemptState.INFRA_ERROR,
    FailureClass.BROWSER_ERROR: AttemptState.BROWSER_ERROR,
    FailureClass.NETWORK_ERROR: AttemptState.NETWORK_ERROR,
    FailureClass.AGENT_STARTUP_ERROR: AttemptState.AGENT_ERROR,
    FailureClass.POLICY_VIOLATION: AttemptState.POLICY_VIOLATION,
    FailureClass.TIMEOUT: AttemptState.TIMEOUT,
    FailureClass.BUDGET_EXCEEDED: AttemptState.BUDGET_EXCEEDED,
    FailureClass.INCONCLUSIVE: AttemptState.AGENT_ERROR,
}


# spec 30: infra-class failures don't count toward the N valid functional samples a
# stability run needs — they get resampled instead (spec 31), up to a hard cap.
INFRA_FAILURE_CLASSES: set[FailureClass] = {
    FailureClass.SCENARIO_SETUP_ERROR,
    FailureClass.TARGET_APP_ERROR,
    FailureClass.INFRA_ERROR,
    FailureClass.BROWSER_ERROR,
    FailureClass.NETWORK_ERROR,
    FailureClass.AGENT_STARTUP_ERROR,
    # TIMEOUT is ambiguous but treated as infra here: a timeout says nothing about whether
    # the agent's approach was right, only that something didn't finish in budget.
    FailureClass.TIMEOUT,
}


class FunctionalResult(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    FLAKY = "FLAKY"
    INCONCLUSIVE = "INCONCLUSIVE"


class ExecutionHealth(str, Enum):
    CLEAN = "CLEAN"
    RECOVERED = "RECOVERED"
    DEGRADED = "DEGRADED"
    INTERRUPTED = "INTERRUPTED"
    FAILED_INFRA = "FAILED_INFRA"


@dataclass
class AttemptResult:
    attempt_id: str
    task_id: str
    task_revision: int
    state: AttemptState
    passed: bool
    failure_class: FailureClass | None = None
    detail: str = ""
    steps: int = 0
    duration_ms: float = 0.0
    tokens: dict = field(default_factory=dict)
    verify_result: dict | None = None
    integrity_violations: list | None = None
    started_at: str = ""
    finished_at: str = ""

    def is_infra_failure(self) -> bool:
        return self.failure_class in INFRA_FAILURE_CLASSES


@dataclass
class StabilityResult:
    task_id: str
    task_revision: int
    n_requested: int
    attempts: list[AttemptResult]
    functional_result: FunctionalResult
    execution_health: ExecutionHealth
    success_rate: float | None
    confidence_interval: tuple[float, float] | None
    valid_sample_count: int
    total_attempt_count: int
    reason: str = ""

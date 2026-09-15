"""Run queue + single-active-run enforcement (spec 25). HIGH/NORMAL/LOW, FIFO within a
priority. Only one Benchmark Run may be RUNNING (or PREPARING/COMPLETING, i.e. actively
occupying the environment) at a time in a Benchmark Environment; the rest wait queued.
"""

from collections import deque
from dataclasses import dataclass, field

from benchmark_target.runner.models import RunState, can_transition

PRIORITIES = ("HIGH", "NORMAL", "LOW")

# States that count as "occupying" the environment — a second run may not enter PREPARING
# while any of these is active (spec 25: "Only one active Benchmark Run may execute").
ACTIVE_STATES = {
    RunState.PREPARING, RunState.RUNNING, RunState.PAUSING, RunState.PAUSED,
    RunState.RESUMING, RunState.CANCELLING, RunState.INTERRUPTED, RunState.RECOVERING,
    RunState.COMPLETING,
}


class InvalidTransitionError(Exception):
    pass


@dataclass
class RunRecord:
    run_id: str
    priority: str
    state: RunState = RunState.QUEUED
    history: list = field(default_factory=list)


class RunManager:
    def __init__(self):
        self._queues: dict[str, deque[str]] = {p: deque() for p in PRIORITIES}
        self._runs: dict[str, RunRecord] = {}

    def submit(self, run_id: str, priority: str = "NORMAL") -> RunRecord:
        if priority not in PRIORITIES:
            raise ValueError(f"invalid priority: {priority}")
        if run_id in self._runs:
            raise ValueError(f"run_id already submitted: {run_id}")
        record = RunRecord(run_id=run_id, priority=priority)
        self._runs[run_id] = record
        self._queues[priority].append(run_id)
        return record

    def active_run_id(self) -> str | None:
        for run_id, record in self._runs.items():
            if record.state in ACTIVE_STATES:
                return run_id
        return None

    def start_next(self) -> RunRecord | None:
        """Pops the next queued run (HIGH first, FIFO within a priority) and moves it to
        PREPARING, but only if no other run currently occupies the environment."""
        if self.active_run_id() is not None:
            return None
        for priority in PRIORITIES:
            queue = self._queues[priority]
            while queue:
                run_id = queue.popleft()
                record = self._runs[run_id]
                if record.state == RunState.QUEUED:
                    self.transition(run_id, RunState.PREPARING)
                    return record
        return None

    def transition(self, run_id: str, target: RunState) -> RunRecord:
        record = self._runs[run_id]
        if not can_transition(record.state, target):
            raise InvalidTransitionError(f"{run_id}: cannot go {record.state} -> {target}")
        record.history.append((record.state, target))
        record.state = target
        return record

    def get(self, run_id: str) -> RunRecord:
        return self._runs[run_id]

    def queue_depth(self) -> dict[str, int]:
        return {p: len(q) for p, q in self._queues.items()}

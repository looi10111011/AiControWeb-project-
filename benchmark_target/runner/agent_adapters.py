"""Pluggable agent adapters. The Run Engine never talks to an LLM/browser directly — it
calls an adapter's `run(task, actor)` and classifies whatever comes back. This is what
lets Phase 6 (queue/state machines/retry/stability/artifacts) be fully unit-tested without
a real browser or API key, while Phase 7 wires the real thing through the same interface.
"""

from dataclasses import dataclass
from typing import Protocol


@dataclass
class AgentRunResult:
    success: bool
    steps: int
    message: str
    tokens: dict


class AgentAdapterError(Exception):
    """Raised by an adapter when the agent/browser/infra itself failed, as opposed to the
    agent completing a task run that didn't achieve the goal (that's AgentRunResult.success
    = False, not an exception). `failure_class` must be one of the INFRA_ERROR-family
    FailureClass values — attempt_runner maps it straight through."""

    def __init__(self, message: str, failure_class: str):
        super().__init__(message)
        self.failure_class = failure_class


class AgentAdapter(Protocol):
    def run(self, task: dict, actor: str) -> AgentRunResult: ...


class StubAgentAdapter:
    """Deterministic, scripted adapter for testing the Run Engine itself (spec: Phase 6
    is queue/state-machine/retry/stability/artifacts — it does not require a live browser
    or LLM to be correct). Pass a list of outcomes; each call to `run()` consumes the next
    one. An outcome is either an AgentRunResult, or an (exception_message, failure_class)
    tuple to raise as AgentAdapterError. Repeats the last outcome once the list is exhausted.
    """

    def __init__(self, outcomes: list):
        if not outcomes:
            raise ValueError("StubAgentAdapter needs at least one scripted outcome")
        self._outcomes = outcomes
        self._call_count = 0

    def run(self, task: dict, actor: str) -> AgentRunResult:
        idx = min(self._call_count, len(self._outcomes) - 1)
        outcome = self._outcomes[idx]
        self._call_count += 1
        if isinstance(outcome, tuple):
            message, failure_class = outcome
            raise AgentAdapterError(message, failure_class)
        return outcome

    @property
    def call_count(self) -> int:
        return self._call_count


def default_password_for(username: str) -> str:
    """Reconstructs the seed's fixed password convention (see app/seed.py's module
    docstring: "{FirstName}@123", and "admin"/"Admin@123" as the one exception — which
    still fits the same shape once you capitalize the username's first token). Not a
    secret lookup, just arithmetic on a documented, deterministic fixture convention — the
    agent still has to actually find and use the login form itself."""
    first_token = username.split(".")[0]
    return f"{first_token.capitalize()}@123"


class HermesAgentAdapter:
    """Phase 7: wraps the real Hermes Orchestrator (backend/app/core/orchestrator.py).
    Not exercised by Phase 6's unit tests — those use StubAgentAdapter — but the interface
    is identical, so the Run Engine (attempt_runner/stability_runner/queue/artifacts)
    doesn't change at all between a stubbed and a live run.

    Deliberately does NOT use BrowserPool: `Orchestrator.run_task(browser=None)` already
    launches and tears down its own Playwright/Chromium per call (see the docstring on
    run_task's `browser` param) — a benchmark run isn't trying to reuse warm browser
    processes across tasks the way the live API server is, so the extra pool machinery
    would be complexity with no payoff here.
    """

    def __init__(self, provider: str | None = None, max_steps: int = 20, headless: bool = True):
        self._provider = provider
        self._max_steps = max_steps
        self._headless = headless

    def run(self, task: dict, actor: str) -> AgentRunResult:
        import asyncio

        from backend.app.core.orchestrator import Orchestrator

        password = default_password_for(actor)
        goal = (
            f"{task['goal']['description']}\n\n"
            f'If you need to log in, use username "{actor}" and password "{password}".'
        )

        async def _run():
            orchestrator = Orchestrator()
            return await orchestrator.run_task(
                url=task["url"],
                goal=goal,
                max_steps=self._max_steps,
                headless=self._headless,
                provider=self._provider,
                confirm_plan=False,
                # W11-style auto-approve (see CLAUDE.md): a live benchmark run must never
                # block on a human answering a NEEDS_CONFIRMATION prompt via terminal input().
                ask_user_func=lambda _cmd: True,
                allowed_domains={"localhost"},
            )

        try:
            result = asyncio.run(_run())
        except Exception as e:  # noqa: BLE001 — classify, don't let raw exceptions escape
            raise AgentAdapterError(f"{type(e).__name__}: {e}", "BROWSER_ERROR") from e

        return AgentRunResult(
            success=result.get("success", False),
            steps=result.get("steps", 0),
            message=result.get("message", ""),
            tokens=result.get("tokens", {}),
        )

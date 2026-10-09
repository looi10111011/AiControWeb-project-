"""Shared pytest fixtures.

W_test_telemetry_isolation: telemetry.write_token_usage()/write_step_trace() append to the paths in
settings, and nothing redirected them during tests — one full run added ~121 example.com rows to the
real data/token_usage.jsonl (the source of the "564 of 642 rows" that kpi.py has to filter out).
Every test now writes to its own tmp_path; tests that set these paths themselves still win.
"""

import pytest

from backend.app.config import settings


@pytest.fixture(autouse=True)
def _isolate_telemetry_files(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "token_usage_log_path", str(tmp_path / "token_usage.jsonl"))
    monkeypatch.setattr(settings, "step_trace_log_path", str(tmp_path / "step_trace.jsonl"))

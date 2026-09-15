"""Entrypoint for the Target Surface only — python benchmark_target/run_target.py

Deliberately not wired into the main project's run.py yet (that file is the agent's
entrypoint; this is the thing the agent gets pointed at). No --reload, matching this repo's
own documented reason for avoiding it (WindowsSelectorEventLoopPolicy / asyncio subprocess
issue) even though this app itself doesn't spawn subprocesses — consistency, not necessity.
"""

import uvicorn

from benchmark_target.app.config import TARGET_HOST, TARGET_PORT

if __name__ == "__main__":
    uvicorn.run("benchmark_target.app.main:app", host=TARGET_HOST, port=TARGET_PORT)

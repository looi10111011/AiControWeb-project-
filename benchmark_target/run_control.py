"""Entrypoint for the Control Plane only — python benchmark_target/run_control.py

Separate process, separate port from run_target.py, on purpose (spec 2). Run both when
using this benchmark: the Target Surface is what the Agent's browser points at, the
Control Plane is what the benchmark runner calls to reset/seed/verify/inspect.
"""

import uvicorn

from benchmark_target.control.config import CONTROL_HOST, CONTROL_PORT

if __name__ == "__main__":
    uvicorn.run("benchmark_target.control.main:app", host=CONTROL_HOST, port=CONTROL_PORT)

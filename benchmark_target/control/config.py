"""Control Plane config. Separate process, separate port, loopback-only — never linked
from any Target Surface page (spec 2). The Agent has no route from one to the other.
"""

from pathlib import Path

CONTROL_HOST = "127.0.0.1"
CONTROL_PORT = 8101

# Spec 22: verify environment identity before a destructive op, fail closed otherwise.
# The DB path itself is hardcoded to this repo's own fixture file (never a real deployment's
# database), but /reset and /fixtures/apply still require the caller to echo this string —
# a cheap, explicit guard against a benchmark-runner bug pointed at the wrong environment.
ENVIRONMENT_IDENTITY = "hermes-benchmark-local"

BASE_DIR = Path(__file__).resolve().parent.parent  # benchmark_target/
CONTROL_DATA_DIR = BASE_DIR / "data" / "control"
SNAPSHOTS_DIR = CONTROL_DATA_DIR / "snapshots"
DIFFS_DIR = CONTROL_DATA_DIR / "diffs"

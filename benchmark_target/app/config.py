"""Target Surface config.

Everything here is fixed/local-only on purpose (spec: deterministic, local-first, easy to
reset). No .env — this is a benchmark fixture app, not a deployable product.
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # benchmark_target/
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "target.db"

TEMPLATES_DIR = BASE_DIR / "app" / "templates"
STATIC_DIR = BASE_DIR / "app" / "static"

# Session cookie signing key. Fixed on purpose: the whole point of this app is that a reset
# reproduces the exact same environment every time, and a random-per-boot secret would
# invalidate every session across a Control Plane restart mid-run. This is a benchmark
# fixture, never exposed beyond localhost — not a secrets-management concern here.
SESSION_SECRET = "benchmark-target-fixed-session-secret-v1"
SESSION_MAX_AGE_SECONDS = 30 * 60  # 30 min — used by the session-expiry benchmark tasks (L4)

TARGET_HOST = "127.0.0.1"
TARGET_PORT = 8100

EMPLOYEES_PER_PAGE = 10

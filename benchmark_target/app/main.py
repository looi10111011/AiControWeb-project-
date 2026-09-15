"""Target Surface — the only thing the Browser Agent may ever reach.

Deliberately a separate FastAPI app/process from any future Control Plane (spec 2): no
import of, or route into, control-plane code lives here, and none ever should.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from benchmark_target.app.config import SESSION_MAX_AGE_SECONDS, SESSION_SECRET, STATIC_DIR
from benchmark_target.app.db import get_conn, init_db
from benchmark_target.app.routers import admin, auth, dashboard, documents, leave, performance, pim, recruitment, reports, time
from benchmark_target.app.seed import seed


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    with get_conn() as conn:
        has_data = conn.execute("SELECT 1 FROM employees LIMIT 1").fetchone() is not None
    if not has_data:
        seed()
    yield


app = FastAPI(title="Hermes Benchmark HRM — Target Surface", lifespan=lifespan)

app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET,
    max_age=SESSION_MAX_AGE_SECONDS,
    session_cookie="hermes_hrm_session",
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

app.include_router(dashboard.router)
app.include_router(auth.router)
app.include_router(pim.router)
app.include_router(admin.router)
app.include_router(leave.router)
app.include_router(time.router)
app.include_router(recruitment.router)
app.include_router(performance.router)
app.include_router(documents.router)
app.include_router(reports.router)


@app.get("/health")
def health():
    return {"status": "ok"}

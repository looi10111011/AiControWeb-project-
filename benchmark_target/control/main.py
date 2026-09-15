"""Benchmark Control Plane (spec 2B). A separate FastAPI app/process from the Target
Surface — different port, no shared router, no link from any Target Surface page. It talks
to the same SQLite file as infrastructure (reset/seed/inspect it), which is not the same
thing as being reachable by the Agent: the Agent only ever gets a browser pointed at the
Target Surface's port, and nothing here is served on that port.

Never expose this port to anything but the benchmark runner.
"""

import json
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import Body, FastAPI, HTTPException, Query

from benchmark_target.app.db import TABLE_NAMES, get_conn, init_db, reset_db
from benchmark_target.app.faults import arm_fault, sweep_expired
from benchmark_target.app.seed import seed as seed_db
from benchmark_target.control import integrity, preconditions, verifiers
from benchmark_target.control.config import DIFFS_DIR, ENVIRONMENT_IDENTITY, SNAPSHOTS_DIR
from benchmark_target.control.snapshot import capture, diff as diff_snapshots


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    DIFFS_DIR.mkdir(parents=True, exist_ok=True)
    with get_conn() as conn:
        # spec 19: "if Runner crashes -> startup recovery -> disable stale faults"
        sweep_expired(conn)
    yield


app = FastAPI(title="Hermes Benchmark HRM — Control Plane", lifespan=lifespan)


def _require_environment(confirm_environment: str) -> None:
    if confirm_environment != ENVIRONMENT_IDENTITY:
        # spec 22: fail closed if environment identity can't be verified.
        raise HTTPException(
            400,
            f"Environment identity mismatch — refusing a destructive operation. "
            f'Expected confirm_environment="{ENVIRONMENT_IDENTITY}".',
        )


@app.get("/health")
def health():
    checks = {}
    try:
        with get_conn() as conn:
            counts = {t: conn.execute(f"SELECT COUNT(*) c FROM {t}").fetchone()["c"] for t in TABLE_NAMES}
            checks["db_reachable"] = True
            checks["schema_ok"] = True
            checks["seed_present"] = counts.get("employees", 0) > 0
            checks["table_counts"] = counts
    except Exception as e:  # noqa: BLE001 — health check must report, never crash the caller
        checks["db_reachable"] = False
        checks["error"] = str(e)

    healthy = checks.get("db_reachable") and checks.get("schema_ok")
    return {"status": "healthy" if healthy else "unhealthy", "checks": checks}


@app.get("/fixtures")
def list_fixtures():
    # V1: one fixture — the full deterministic seed. Named and listed explicitly so the
    # catalog/task layer (spec 13) has a stable contract to grow into, even before there
    # are multiple named fixtures to choose from.
    return {"fixtures": ["base"]}


@app.post("/fixtures/apply")
def apply_fixture(name: str = Body(...), confirm_environment: str = Body(...)):
    _require_environment(confirm_environment)
    if name != "base":
        raise HTTPException(404, f"Unknown fixture: {name}")
    reset_db()
    seed_db()
    return {"applied": name, "at": datetime.now(timezone.utc).isoformat()}


@app.post("/reset")
def reset(confirm_environment: str = Body(..., embed=True)):
    _require_environment(confirm_environment)
    reset_db()
    seed_db()
    return {"reset": True, "at": datetime.now(timezone.utc).isoformat()}


@app.post("/snapshot")
def create_snapshot(tables: list[str] | None = Body(None, embed=True)):
    try:
        with get_conn() as conn:
            data = capture(conn, tables)
    except ValueError as e:
        raise HTTPException(400, str(e))

    snapshot_id = uuid.uuid4().hex[:12]
    payload = {"snapshot_id": snapshot_id, "captured_at": datetime.now(timezone.utc).isoformat(), "data": data}
    (SNAPSHOTS_DIR / f"{snapshot_id}.json").write_text(json.dumps(payload, indent=2, default=str))
    return {"snapshot_id": snapshot_id, "captured_at": payload["captured_at"], "tables": list(data.keys())}


@app.get("/snapshot/{snapshot_id}")
def get_snapshot(snapshot_id: str):
    path = SNAPSHOTS_DIR / f"{snapshot_id}.json"
    if not path.exists():
        raise HTTPException(404, f"Unknown snapshot: {snapshot_id}")
    return json.loads(path.read_text())


@app.post("/state-diff")
def state_diff(before_snapshot_id: str = Body(...), after_snapshot_id: str | None = Body(None)):
    before_path = SNAPSHOTS_DIR / f"{before_snapshot_id}.json"
    if not before_path.exists():
        raise HTTPException(404, f"Unknown snapshot: {before_snapshot_id}")
    before_payload = json.loads(before_path.read_text())

    if after_snapshot_id:
        after_path = SNAPSHOTS_DIR / f"{after_snapshot_id}.json"
        if not after_path.exists():
            raise HTTPException(404, f"Unknown snapshot: {after_snapshot_id}")
        after_payload = json.loads(after_path.read_text())
    else:
        # No after-snapshot given: capture live state now, scoped to the same tables the
        # before-snapshot covered (spec 17: scoped diff, not a whole-DB comparison by default).
        with get_conn() as conn:
            after_data = capture(conn, list(before_payload["data"].keys()))
        after_snapshot_id = uuid.uuid4().hex[:12]
        after_payload = {
            "snapshot_id": after_snapshot_id,
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "data": after_data,
        }
        (SNAPSHOTS_DIR / f"{after_snapshot_id}.json").write_text(json.dumps(after_payload, indent=2, default=str))

    result = diff_snapshots(before_payload["data"], after_payload["data"])
    diff_id = uuid.uuid4().hex[:12]
    record = {
        "diff_id": diff_id,
        "before_snapshot_id": before_snapshot_id,
        "after_snapshot_id": after_snapshot_id,
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "diff": result,
    }
    (DIFFS_DIR / f"{diff_id}.json").write_text(json.dumps(record, indent=2, default=str))
    return record


@app.post("/preconditions/check")
def check_preconditions(conditions: list[dict] = Body(..., embed=True)):
    with get_conn() as conn:
        results = preconditions.evaluate(conn, conditions)
    return {"all_passed": all(r["passed"] for r in results), "results": results}


@app.post("/verify")
def verify(verifier: str = Body(...), args: dict = Body(default_factory=dict)):
    with get_conn() as conn:
        return verifiers.run(conn, verifier, args)


@app.get("/integrity")
def check_integrity():
    with get_conn() as conn:
        violations = integrity.check(conn)
    return {"healthy": len(violations) == 0, "violations": violations}


@app.get("/audit")
def audit_log(
    actor: str | None = None,
    action: str | None = None,
    resource: str | None = None,
    since: str | None = None,
    limit: int = Query(200, le=1000),
):
    where = ["1=1"]
    params: list = []
    if actor:
        where.append("actor_username = ?")
        params.append(actor)
    if action:
        where.append("action = ?")
        params.append(action)
    if resource:
        where.append("resource = ?")
        params.append(resource)
    if since:
        where.append("timestamp >= ?")
        params.append(since)
    where_sql = " AND ".join(where)

    with get_conn() as conn:
        rows = conn.execute(
            f"SELECT * FROM audit_log WHERE {where_sql} ORDER BY id DESC LIMIT ?", [*params, limit]
        ).fetchall()
    return {"count": len(rows), "entries": [dict(r) for r in rows]}


@app.post("/faults")
def create_fault(trigger: str = Body(...), fault_type: str = Body(...), ttl_seconds: int = Body(300)):
    if fault_type not in ("conflict", "error"):
        raise HTTPException(400, "fault_type must be 'conflict' or 'error'")
    with get_conn() as conn:
        fault_id = arm_fault(conn, trigger, fault_type, ttl_seconds)
    return {"fault_id": fault_id, "trigger": trigger, "fault_type": fault_type, "ttl_seconds": ttl_seconds}


@app.get("/faults")
def list_faults():
    with get_conn() as conn:
        sweep_expired(conn)
        rows = conn.execute(
            "SELECT * FROM fault_injections WHERE consumed = 0 ORDER BY id"
        ).fetchall()
    return {"active": [dict(r) for r in rows]}


@app.delete("/faults/{fault_id}")
def delete_fault(fault_id: int):
    with get_conn() as conn:
        cur = conn.execute("DELETE FROM fault_injections WHERE id = ?", (fault_id,))
    if cur.rowcount == 0:
        raise HTTPException(404, f"Unknown fault: {fault_id}")
    return {"deleted": fault_id}

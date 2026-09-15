# Architecture Audit — HRM Benchmark Target

Phase 1 of the Hermes benchmark-web master spec. No application code written yet.

## 1. Current repository

`Aiagentcontrolbrowser` is the **agent**, not a target app: FastAPI backend (`backend/app/`)
driving Playwright, an Orchestrator perceive→plan→act→verify loop, RAG manuals, memory,
permission layer. Frontend is a single-file vanilla Test Console. No SQL database anywhere —
persistence is ChromaDB (RAG/memory) + JSONL telemetry files.

Existing benchmark surface (`backend/app/core/evaluation.py`, `orangehrm_eval.py`,
`miniwob_eval.py`, `release_gate.py`) already runs `BENCHMARK_TASKS` against **live third-party
sites** — SauceDemo and, critically, `opensource-demo.orangehrmlive.com`, a **shared public
OrangeHRM demo** other people also hit concurrently. `orangehrm_eval.py` line ~18 already flags
this in its own comment as multi-tenant/non-deterministic. This is exactly the problem the new
spec exists to fix: no reset, no fixtures, no isolation, no ground-truth verification beyond
scraping the live page. `release_gate.py`'s own settings.py comments (`W_gate_is_noisy`) record
the same commit scoring 12/15 then 15/15 with zero code change — a symptom of benchmarking
against an environment nobody controls.

Stack already in place and reusable: FastAPI, Playwright 1.62, pytest+pytest-asyncio, pydantic.
No SQLAlchemy/sqlite3 dependency yet (stdlib `sqlite3` needs no new dependency).

## 2. Reusable components

- **FastAPI app pattern** (`main.py` lifespan, `APIRouter(dependencies=[...])` auth pattern) —
  reuse the same shape for the Control Plane API, on a **different port**, never mounted on the
  agent's router.
- **Playwright** is already a project dependency — Phase 7 browser validation needs nothing new.
- **pytest / pytest-asyncio**, already strict-mode (no `conftest.py`/`pytest.ini` in repo) — new
  target-app tests follow the same explicit-marker convention.
- **JSONL telemetry pattern** (`core/telemetry.py`) — the benchmark's `trace.jsonl` /
  `step_trace.jsonl`-equivalent artifacts should follow the same "write once at end, one writer"
  rule that already proved itself here (`data/token_usage.jsonl` design).
- **`run.py` single-entrypoint convention** — the benchmark gets its own `run_benchmark.py`
  entrypoint (or a `python run.py benchmark ...` subcommand) rather than scattering scripts.

## 3. Gaps

- No SQL persistence layer at all — must add one (plain `sqlite3`, no ORM, per spec §44 "simple
  over unnecessary microservices"; stdlib only, zero new dependency).
- No concept of "two worlds" (target vs. control plane) anywhere in this repo — must be built
  from scratch as a **separate FastAPI app + separate port**, not a mode flag on the existing one.
- No fixture/seed/reset system, no task catalog, no verifier, no run/attempt state machines —
  all net-new (spec §13-31).
- Existing `orangehrm_eval.py` targets the live public demo; once the new local target exists it
  is the obvious replacement, but that swap is out of scope for this session (flagged as a
  follow-up, not silently done — swapping eval targets changes benchmark semantics elsewhere).

## 4. Proposed location & decision record

**User decision (asked directly, not inferred):** new target application lives in a **new
subfolder inside this repo**, not a separate repo. Rationale given the spec's own priority order
(§ FINAL PRIORITY: correctness > isolation > determinism > reproducibility > verifiability >
realism > maintainability > performance > breadth) and §44 ("simple... easy for Claude/Codex to
modify"): one repo to check out, one place for Hermes to point its browser at
`http://localhost:<port>`, no cross-repo path juggling for either the agent or future maintainers.

Proposed structure (not yet created):

```
benchmark_target/
  app/                    Target Surface — FastAPI + server-rendered or minimal-JS HTML,
                           the only thing the Agent may ever reach (its own port, e.g. 8100)
    main.py
    db.py                 sqlite3, WAL mode, one file per environment
    auth/ pim/ leave/ time/ recruitment/ performance/ reports/ documents/ admin/
    templates/  static/
  control/                Benchmark Control Plane — separate FastAPI app, separate port
                           (e.g. 8101), never reachable from the Target Surface's own links/nav
    reset.py fixtures.py preconditions.py verify.py state_diff.py integrity.py
    audit.py faults.py health.py runs.py
  catalog/
    v1/                   immutable once promoted — tasks as YAML, per spec §11-13
  fixtures/                deterministic seed data + fixture files for upload tests
  tests/                   unit + integration + benchmark-determinism tests (pytest, matches
                           repo's strict-asyncio convention)
  data/                    per-environment sqlite file(s), run artifacts — gitignored like
                           the existing data/ dir already is for chroma/eval_results
run_benchmark.py           or a `benchmark` subcommand added to the existing run.py
```

Target Surface and Control Plane are two separate FastAPI processes/ports from day one — this is
the single most consequential architectural decision (spec §2) and is not something to retrofit
later, since retrofitting risk is exactly "the Agent could reach the control plane" which is a
security boundary, not a refactor.

## 5. Risks / open questions carried forward (non-blocking, engineering calls to be made in
   Phase 2+ and documented inline, not asked about again per spec §54)

- RBAC scope model (spec §3.4) will be implemented as the minimal `resource.action` + scope enum
  described in the spec, not a general policy engine.
- 110–160 tasks is a target, not a hard requirement (spec says so explicitly) — actual count
  will follow from what each module can meaningfully support.
- Fault injection (spec §19) and concurrency tests (spec §20) are the highest-complexity, lowest-
  immediate-value items relative to Phase 2/3 foundation — they land in Phase 4/6 as planned, not
  pulled forward.
- SQLite is fine for V1 per spec §44; no Postgres migration path work happens now.

## 6. Next step

Per the user's explicit scope choice for this session, work stops here (Phase 1 only). Phase 2
(Target Application Foundation: auth, layout, dashboard, database, deterministic seed, Employees,
Admin/Users) is the next session's starting point, building on this audit.

# Implementation Report — Hermes Benchmark HRM

Built per `CLAUDE MASTER PROMPT — AUTOMATIC BROWSER AGENT — BENCHMARK WEB v1.0`. Covers
Phases 1–8 of that spec's Phase 1 audit (see [ARCHITECTURE_AUDIT.md](ARCHITECTURE_AUDIT.md)
for the Phase 1 findings this was built on).

## What was implemented

An OrangeHRM-style HRM web application (`benchmark_target/`) that exists purely as a
controlled, deterministic benchmark target for the Hermes browser agent (`backend/`,
this repo's original project) — plus the Control Plane infrastructure to reset/seed/
verify/inspect it, a declarative task catalog, and a Run Engine to execute the catalog
against either a scripted stub or the real agent.

Two hard separations were kept throughout, per the spec's own priority order (correctness
> isolation/security > determinism > reproducibility > verifiability > realism >
maintainability > performance > breadth):

- **Target Surface** (`benchmark_target/app/`, port 8100) — the only thing the Agent's
  browser ever reaches. No route into the Control Plane exists anywhere in its templates
  or routers.
- **Control Plane** (`benchmark_target/control/`, port 8101) — separate FastAPI process,
  separate port, loopback-only, never linked from a Target Surface page.

## Architecture

```
benchmark_target/
  app/                  Target Surface — FastAPI + Jinja2 templates, session-cookie auth
    db.py                 sqlite3 schema (15 tables), TABLE_NAMES as the single source of
                           truth the Control Plane's snapshot/integrity code reads from
    seed.py                Deterministic seed: 50 employees, 19 users, leave/timesheet/
                           recruitment/performance fixtures — every value a pure function
                           of a fixed index, fixed timestamp, no wall-clock, no os.urandom
    security.py             Deterministic password hashing (salt derived from the password
                           itself, not os.urandom — see Phase 8 finding below)
    deps.py                  Minimal RBAC: resource+action+scope, default-deny
    faults.py                 Shared fault-injection table + one-shot consume, read by
                           both the Target Surface (leave/timesheet approval routes) and
                           armed by the Control Plane
    routers/                  auth, dashboard, pim (employees), admin (users), leave,
                           time (timesheets), recruitment, performance, documents, reports
  control/                Control Plane — separate FastAPI app
    main.py                  /health /reset /fixtures /snapshot /state-diff
                           /preconditions/check /verify /integrity /audit /faults
    snapshot.py, preconditions.py, verifiers.py, integrity.py
    config.py                 ENVIRONMENT_IDENTITY fail-closed guard for /reset
  catalog/
    generate_catalog.py     Generates catalog/v1/tasks/L*.yaml from a live seed — every
                           employee_code/username/email a task references is pulled from
                           seed(), not hand-typed, so fixture existence holds by construction
    schema.py, validate_catalog.py
    v1/tasks/L1.yaml..L5.yaml   117 tasks (see distribution below)
  runner/                  Run Engine
    models.py                 Run/Attempt state machines, failure taxonomy, retry policy,
                           result model (spec 25-29)
    attempt_runner.py           Single-attempt lifecycle: reset->fixture->precondition->
                           agent->verify->integrity
    stability_runner.py         N=5 valid-sample stability sampling, auto-fill through
                           infra failures up to 2N, Wilson interval (diagnostic only)
    queue.py                     HIGH/NORMAL/LOW FIFO + single-active-run enforcement
    artifacts.py                  run_manifest/task_results/attempt_results/metrics.json
    agent_adapters.py              StubAgentAdapter (deterministic, no LLM/browser) and
                           HermesAgentAdapter (wraps the real Orchestrator)
  tests/                   test_runner.py (23), test_determinism.py (14)
  run_target.py, run_control.py, run_benchmark.py
```

## Benchmark task catalog

**117 tasks** generated from the live seed, validated against the schema and the Control
Plane's own verifier/precondition registries (so a task can never reference a verifier or
precondition type that doesn't exist).

| Level | Count | Spec target |
|-------|-------|-------------|
| L1 — Basic | 29 | 20-30 |
| L2 — Structured | 30 | 30-40 (low end) |
| L3 — Complex Workflow | 28 | 30-40 (close, slightly under) |
| L4 — Recovery | 20 | 20-30 (low end) |
| L5 — Expert | 10 | 10-20 (low end) |
| **Total** | **117** | 110-160 |

Spec explicitly permits not forcing exact per-level counts ("do not force exact counts if
quality suffers") — L3/L4/L5 sit at or slightly under target; every task is still meaningful
and grounded in real seed data, which was prioritized over hitting an exact number.

## Test results

```
benchmark_target/tests/test_runner.py        23 passed
benchmark_target/tests/test_determinism.py   14 passed
                                              ────────────
                                              37 passed, 0 failed
```

Both suites run against a real Control Plane (`fastapi.testclient.TestClient`, in-process,
no subprocess) and real SQLite — only the agent itself is stubbed in these suites, which is
this phase's intended scope (queue/state-machine/retry/stability/artifacts/determinism, not
a live browser or LLM).

## Browser validation results (real agent, real LLM, real browser)

Two live runs against the real Hermes `Orchestrator` (OpenAI ChatGPT-OAuth provider,
headless Chromium), kept deliberately small since each attempt spends real API/OAuth quota:

| Task | Result | Steps | Duration | Tokens |
|------|--------|-------|----------|--------|
| `AUTH-LOGIN-SUCCESS-ADMIN` | **PASS** | 5-6 | ~40-58s | ~51k |
| `PIM-EDIT-VERIFY-01` | **FUNCTIONAL_FAIL** | 4 | ~38s | ~52k |

The second result is a genuine finding, not a benchmark bug: the agent reported completing
the edit, but the authoritative DB state showed the field was never saved. The benchmark's
verify-against-DB-state design (spec 16 — "business state is authoritative") caught exactly
this class of self-report-vs-reality gap, which is the whole point of building it this way.

## Known limitations

- **`download_artifact` / `http_redirect` verification types are stubs.** The Control
  Plane's job is authoritative DB-state verification; inspecting a downloaded CSV's
  content or an HTTP redirect is explicitly the benchmark *runner's* job per spec 9, and
  `attempt_runner.py` reports these honestly as `INCONCLUSIVE` rather than faking a pass.
  A handful of catalog tasks (CSV export, some redirect-only L1 tasks) use these types.
- **Fault injection covers two triggers** (`leave.approval.submit`, `timesheet.approval.submit`),
  each with `conflict`/`error` types — proven end-to-end (armed via Control Plane, consumed
  exactly once by the Target Surface, retry succeeds), but not exhaustively wired into every
  mutating route.
- **RBAC is the minimal model the spec explicitly asks for** (resource+action+scope,
  default-deny) — not a general policy engine. Scopes are `own`/`direct_reports`/`all` only.
- **No screenshots/video capture** — spec marks both optional; not needed for this
  phase's validation.
- **Single fixture** (`base`) — spec's fixture/reset separation exists structurally
  (`/fixtures/apply` vs `/reset` are separate endpoints) but there's only one fixture to
  apply in v1.
- **Only 2 tasks have been run live end-to-end.** The other 115 are validated for schema
  correctness, precondition truth, and determinism, but not yet proven against a real
  agent run — that's a cost/time tradeoff, not a design gap.

## Remaining risks

- **LLM agent behavior is inherently non-deterministic** — Phase 8 deliberately proved
  determinism at the fixture/seed/catalog/verifier layer instead of the agent layer, since
  the repo's own `optimize.txt`/`release_gate.py` already document the agent-level success
  rate as noisy run-to-run. A future stability run (`run_stability`, N=5) against the real
  agent will show that noise directly; this is expected, not a bug to chase.
- **Reproducibility manifest (spec 49) is partial.** `run_manifest.json` captures
  catalog_version, levels, agent_profile, and stability_n; it does not yet capture browser
  version, OS, or a prompt_version pin (Hermes's own prompt-versioning didn't exist as a
  citable artifact at generation time).
- **Catalog generator correctness depends on the seed staying in sync with it** — Phase 8
  already caught one instance of this class of risk (the password-salt determinism bug);
  a similar drift could recur if `seed.py`'s deterministic indices ever change without
  regenerating the catalog. `validate_catalog.py` catches schema/verifier drift but not
  semantic drift like the tautological-task bug fixed in Phase 6.

## Commands

Setup (run once):
```bash
.venv\Scripts\python.exe -m benchmark_target.catalog.generate_catalog
```

Run the Target Surface (what the Agent's browser points at):
```bash
.venv\Scripts\python.exe benchmark_target\run_target.py
# -> http://127.0.0.1:8100
```

Run the Control Plane (separate terminal — what the benchmark runner calls):
```bash
.venv\Scripts\python.exe benchmark_target\run_control.py
# -> http://127.0.0.1:8101
```

Reset the environment to a clean deterministic state:
```bash
curl -X POST http://127.0.0.1:8101/reset -H "Content-Type: application/json" -d "{\"confirm_environment\": \"hermes-benchmark-local\"}"
```

Run the benchmark (stub agent — proves wiring, no LLM cost):
```bash
.venv\Scripts\python.exe benchmark_target\run_benchmark.py --level L1 --n 3
```

Reproduce one specific task with the real Hermes agent (spends real API/OAuth quota):
```bash
.venv\Scripts\python.exe benchmark_target\run_benchmark.py --hermes-provider --task-id AUTH-LOGIN-SUCCESS-ADMIN --n 1
```

Validate the catalog (schema, uniqueness, verifier/precondition existence):
```bash
.venv\Scripts\python.exe -m benchmark_target.catalog.validate_catalog
```

Run the full benchmark_target test suite:
```bash
.venv\Scripts\python.exe -m pytest benchmark_target\tests\ -q
```

## Recommended next steps

1. Run a full stability sample (`--n 5`, default) with `--hermes-provider` across a
   representative slice of each level to get a first real success-rate baseline —
   budget for real API cost and wall-clock time (each attempt is ~40-90s).
2. Wire `download_artifact` verification into the runner (fetch the actual CSV response
   and check headers/content) rather than leaving it `INCONCLUSIVE`.
3. Consider replacing `orangehrm_eval.py`'s target (the shared public
   `opensource-demo.orangehrmlive.com`) with this local, deterministic target — that was
   flagged in the Phase 1 audit as the exact problem this benchmark exists to fix, but the
   swap itself was left out of scope for this build.
4. Expand fault injection triggers to more mutating routes once the first triggers prove
   useful signal in real runs.
5. Populate the reproducibility manifest's remaining fields (browser version, OS,
   prompt_version) once Hermes's own prompt versioning is a citable artifact.

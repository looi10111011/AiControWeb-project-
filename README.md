# AI Browser Agent

AI ที่ควบคุมหน้าเว็บได้เอง โดย (1) เรียนรู้หน้าเว็บด้วยตัวเอง (Perception) และ (2) ใช้คู่มือที่ user ป้อนเพิ่ม (RAG)

รายละเอียดแผนงานเต็ม: [`roadmap.txt`](./roadmap.txt)

## โครงสร้างโปรเจกต์

```
backend/
  app/
    main.py               # FastAPI entrypoint — lifespan สร้าง BrowserPool/TaskManager/
                           # SessionRegistry/LearnManager
    config.py              # Settings (pydantic-settings, อ่าน .env) — provider/feature flags
    api/                    # HTTP layer — [B]
      routes.py              # ทุก endpoint (task submit/poll/SSE, site-manual learn, credentials)
      schemas.py               # Pydantic request/response models
      task_manager.py           # รัน Orchestrator.run_task() แบบ background task
    core/                    # Agent core — [A]
      orchestrator.py          # Perceive -> Plan -> Act -> Verify loop (หัวใจของ agent)
      perception.py             # หน้าเว็บ -> indexed elements snapshot
      actions.py                 # dispatch action ไปยัง Playwright
      llm.py                       # prompt + tool-calling ต่อ provider (Anthropic/Gemini/OpenAI)
      fastpath_executor.py         # เล่นซ้ำ site manual ที่เรียนรู้ไว้ ข้าม LLM ต่อ step
      state_filter.py                # กัน action ซ้ำ/ไม่มีผลก่อนถึง LLM
      dom_locator.py                  # descriptor ของ element ที่ข้าม step/ข้าม run ได้จริง
      memory.py / long_term_memory.py / plan_memory.py / procedural_memory.py
                                        # ความจำระยะสั้น/ยาว/แผน/ทำซ้ำ
      browser_pool.py                   # pool ของ Playwright browser ที่เปิดค้างไว้ใช้ซ้ำ
      session_registry.py                # track session แบบ interactive ที่ยืม/คืน browser
      user_browser.py                     # ต่อเข้า Chrome จริงของ user ผ่าน CDP
      crypto_store.py / openai_oauth.py    # เก็บ credential เข้ารหัส / OAuth "Sign in with ChatGPT"
      release_gate.py / telemetry.py        # gate วัดผล / เขียน token_usage.jsonl + step_trace.jsonl
      evaluation.py / miniwob_eval.py / orangehrm_eval.py  # ชุด benchmark
    permission/               # Permission layer — [B]
      rules.py                  # classify_action(): SAFE / NEEDS_CONFIRMATION / BLOCKED
    rag/                      # Knowledge (RAG manual) — [B]
      chroma_client.py           # ChromaDB connection + embedding function
      ingestion.py                 # PDF/DOCX/TXT/XLSX -> chunk -> embed
      retriever.py                   # query คู่มือที่ ingest ไว้
    site_learning/            # เรียนรู้เว็บอัตโนมัติ — [A]
      crawler.py, extractor.py, learn_manager.py, auto_login.py, storage.py, safety.py
  tests/                    # pytest (ต้องมี @pytest.mark.asyncio ทุกเทสต์ async — ไม่มี conftest.py)
frontend/                  # Test Console (UI) — index.html ไฟล์เดียว ไม่มี build step
benchmark_target/           # เว็บ HRM จำลอง (deterministic) สำหรับ benchmark agent เอง แทน
                             # เว็บ demo สาธารณะ — ดู W_benchmark_target ใน roadmap.txt
  app/                        # Target Surface (port 8100) — หน้าเดียวที่ agent เข้าถึงได้
  control/                     # Control Plane (port 8101) — reset/fixtures/verify แยก process
  catalog/                      # task catalog (YAML, generate จาก seed จริง)
  runner/                        # Run Engine: state machine, stub/real agent adapter
  tests/
docs/
  SRS.md                    # Scope / Software Requirements
  embedded-agent-bar.md
data/                       # ChromaDB persistence, eval results, manuals, token usage log
  manuals/ site_manuals/       # คู่มือ (RAG ingest) / คู่มือเว็บที่เรียนรู้อัตโนมัติ
  screenshots/                  # ภาพหน้าจอตอน vision fallback
  eval_results/                   # ผลรัน eval/release-gate/flakiness (ไม่ track ใน git)
  chroma/                          # vector DB ของ ChromaDB (ไม่ track ใน git)
run.py                     # entrypoint เดียวของทุกคำสั่ง (server/test/agent/eval/... ดู README/CLAUDE.md)
roadmap.txt                # แผนงานรายสัปดาห์เต็ม + log ความคืบหน้าจริง (W<n> tag)
optimize.txt               # แผน/ปัญหาที่กำลังแก้อยู่ (living doc)
```

## Setup

```bash
# สร้าง virtual environment
py -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# ติดตั้ง dependencies
pip install -r requirements.txt
playwright install chromium

# ตั้งค่า environment variables
copy .env.example .env
# แล้วกรอก ANTHROPIC_API_KEY / GEMINI_API_KEY

# รัน API server
uvicorn backend.app.main:app --reload
```

เปิด http://127.0.0.1:8000/health เพื่อเช็คว่า server รันอยู่

## Tech Stack

| ส่วน | เทคโนโลยี |
|---|---|
| Browser control | Playwright (Python) |
| Agent framework | Loop เขียนเอง |
| LLM | Claude (หลัก) + Gemini (สำรอง) |
| RAG / Vector DB | ChromaDB |
| Backend / API | FastAPI |
| UI | Test Console (web) |

## บทบาท

- **[A] Agent Core** — Orchestrator, Perception, Playwright, LLM integration, Memory
- **[B] Knowledge & Interface** — RAG คู่มือ, Permission logic, Test Console (UI), Evaluation, เอกสาร

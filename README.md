# AI Browser Agent

AI ที่ควบคุมหน้าเว็บได้เอง โดย (1) เรียนรู้หน้าเว็บด้วยตัวเอง (Perception) และ (2) ใช้คู่มือที่ user ป้อนเพิ่ม (RAG)

รายละเอียดแผนงานเต็ม: [`roadmap.txt`](./roadmap.txt)

## โครงสร้างโปรเจกต์

```
backend/
  app/
    main.py            # FastAPI entrypoint
    config.py          # Settings (env vars)
    core/               # Agent core — [A]
      orchestrator.py   # Perceive -> Plan -> Act -> Verify loop
      perception.py      # หน้าเว็บ -> indexed elements snapshot (W2 ✅)
      actions.py         # Browser actions ผ่าน Playwright
      memory.py          # short-term / long-term memory
    rag/                 # Knowledge — [B]
      chroma_client.py   # ChromaDB connection
      ingestion.py        # PDF/DOCX/TXT -> chunk -> embed
      retriever.py         # query คู่มือ
    permission/          # Permission logic — [B]
      rules.py            # allowlist / blocklist / human-in-the-loop
  tests/
frontend/               # Test Console (UI) — เริ่ม W10
docs/
  SRS.md                # Scope / Software Requirements
data/
  manuals/               # คู่มือทดสอบ (PDF/DOCX/TXT) สำหรับ ingest
  screenshots/           # เก็บภาพหน้าจอ agent (ถ้าใช้ vision fallback)
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

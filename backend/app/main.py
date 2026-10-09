import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

# กัน UnicodeEncodeError ตอน print ไทย/emoji บน Windows console (cp1252) — run.py แก้แล้วฝั่ง CLI
# แต่ uvicorn ที่ import ไฟล์นี้ตรงๆ ไม่เคยผ่านโค้ดนั้น
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

# หมายเหตุ: ตั้ง asyncio event loop policy ที่นี่ไม่ได้ผล — uvicorn สร้าง loop (Selector ตอน --reload)
# ก่อน import ไฟล์นี้เสมอ ต้องแก้ที่ run.py (ห้ามใช้ --reload) แทน

from backend.app.api.routes import limiter as api_limiter
from backend.app.api.routes import router as api_router
from backend.app.api.routes import verify_api_key
from backend.app.api.task_manager import TaskManager
from backend.app.config import settings
from backend.app.core.browser_pool import BrowserPool
from backend.app.core.session_registry import SessionRegistry
from backend.app.site_learning.learn_manager import LearnManager

# W20: หน้าเว็บย้ายจาก backend/app/static/ ไป frontend/ (repo root) — vanilla ไม่มี build step
STATIC_DIR = Path(__file__).parent.parent.parent / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # W10[A]: เปิด pool ตอน startup ให้ task แรกไม่ต้องรอ Chromium launch
    app.state.browser_pool = BrowserPool(size=settings.browser_pool_size)
    await app.state.browser_pool.start()
    app.state.task_manager = TaskManager()
    # W12: session อาจถือ browser ที่ยืมจาก pool — ต้อง close_all() ก่อน pool.shutdown() เสมอ
    app.state.session_registry = SessionRegistry()
    # W14: crawl job ยืม/คืน browser เองใน routes.py ไม่ต้อง cleanup ตอน shutdown
    app.state.learn_manager = LearnManager()
    # pdf/xlsx: session_id -> {"filename", "text"} ของไฟล์ล่าสุด ให้เทิร์นถัดไปถามต่อจากไฟล์ได้
    # โดยไม่ตกไปเปิด browser (routes.py::_file_chat_memory_reply) — text ล้วน ไม่ต้อง cleanup
    app.state.file_chat_memory = {}
    # W_retry_value_has_no_home: session_id -> {"labels": [...]} ของช่องที่รอค่าใหม่หลัง
    # TASK_FAILED_USER_INPUT_ERROR — ทำให้คำสัญญา "ตอบค่าใหม่มา จะกรอกช่องเดิมให้" เป็นจริง
    app.state.pending_value_request = {}
    yield
    await app.state.session_registry.close_all()
    await app.state.browser_pool.shutdown()


app = FastAPI(title="AI Browser Agent", lifespan=lifespan)
# benchmark_target (port 8100) ฝัง "AI bar" widget ที่เรียก API นี้ข้าม origin — อนุญาตเฉพาะ
# 2 origin นั้น (ไม่ใช่ "*"; ไม่ต้อง allow_credentials เพราะไม่มี cookie ข้าม origin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8100", "http://localhost:8100"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)
# Security 1.5: @limiter.limit() ใน routes.py หา limiter จาก app.state — จำกัดเฉพาะ endpoint ที่แปะ decorator
app.state.limiter = api_limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.include_router(api_router)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/config/check", dependencies=[Depends(verify_api_key)])
async def config_check():
    return {
        "primary_llm_provider": settings.primary_llm_provider,
        "fallback_llm_provider": settings.fallback_llm_provider,
        # UI โชว์ model ของ provider ที่เลือก — ส่งตาราง model ทุก provider เพราะสลับได้ใน Settings
        "llm_provider": settings.llm_provider,
        "models": {
            "anthropic": settings.anthropic_model,
            "gemini": settings.gemini_model,
            "groq": settings.groq_model,
            "openai": settings.openai_model,
        },
        "chroma_collection_name": settings.chroma_collection_name,
        "browser_headless": settings.browser_headless,
        "browser_pool_size": settings.browser_pool_size,
    }


# Mounted last so it never shadows the API routes above — serves the console UI from frontend/.
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

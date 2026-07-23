import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

# แก้ปัญหา UnicodeEncodeError เวลา print ข้อความไทย/emoji (orchestrator.py, llm.py, ฯลฯ
# มี print() debug log หลายจุดที่ไม่ได้ gate ด้วย verbose) บน Windows console (cp1252/
# charmap default) — run.py มี fix เดียวกันนี้อยู่แล้วสำหรับตอนรันผ่าน CLI แต่ตอนรัน
# uvicorn ตรงๆ (ไม่ผ่าน run.py) ไม่เคยผ่านโค้ดนั้นเลย ต้อง reconfigure ที่นี่ด้วยเพราะนี่
# คือ entrypoint จริงที่ uvicorn import
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

# หมายเหตุ: เคยลองแก้ NotImplementedError ของ Playwright บน Windows (ดู
# run.py::run_server() สำหรับ root cause จริง — uvicorn --reload บังคับ
# WindowsSelectorEventLoopPolicy) ด้วยการตั้ง asyncio.set_event_loop_policy(...) ตรงนี้
# แต่ไม่ได้ผล เพราะ uvicorn.Server.run() เรียก config.setup_event_loop() (ซึ่งตั้ง
# policy เป็น Selector ตอน --reload) แล้วค่อย asyncio.run(...) สร้าง event loop จริง
# ก่อนที่ backend.app.main จะถูก import ด้วยซ้ำ (import เกิดทีหลังสุดตอน Config.load())
# — ตั้ง policy ในไฟล์นี้จึงสายเกินไปเสมอ ไม่มีผลอะไรกับ loop ที่สร้างไปแล้ว ต้องแก้ที่
# run.py (ไม่ยิง --reload) แทน ไม่ใช่ที่นี่

from backend.app.api.routes import router as api_router
from backend.app.api.task_manager import TaskManager
from backend.app.config import settings
from backend.app.core.browser_pool import BrowserPool
from backend.app.core.session_registry import SessionRegistry
from backend.app.site_learning.learn_manager import LearnManager

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # W10[A]: เปิด browser pool ล่วงหน้าตอน server startup (ไม่ใช่ตอน request แรกเข้ามา)
    # เพื่อให้ task แรกที่เข้ามาไม่ต้องรอ Chromium process launch เหมือนกับ task ถัดๆ ไป
    app.state.browser_pool = BrowserPool(size=settings.browser_pool_size)
    await app.state.browser_pool.start()
    app.state.task_manager = TaskManager()
    # W12: session ที่ยังเปิดค้างอยู่ (ดู core/session_registry.py) อาจถือ browser ที่ยืม
    # มาจาก pool อยู่ — ปิด session ทั้งหมดก่อนเสมอ (คืน browser กลับ pool /ตัด CDP
    # connection /ปิด browser ที่ launch เอง) ก่อนที่ pool.shutdown() จะปิด browser ทุกตัว
    # ทิ้ง ไม่งั้น session ที่เหลืออยู่จะพยายามคืน browser ที่ปิดไปแล้ว
    app.state.session_registry = SessionRegistry()
    # W14: registry ของ crawl job ที่ POST /api/site-manual/learn สร้าง — ไม่มี
    # browser/session ผูกไว้ยาว (crawl ยืม/คืน browser จาก pool เองใน routes.py, ปิดทันที
    # ที่ crawl จบ) เลยไม่ต้องปิดอะไรตอน shutdown เหมือน session_registry ด้านบน
    app.state.learn_manager = LearnManager()
    yield
    await app.state.session_registry.close_all()
    await app.state.browser_pool.shutdown()


app = FastAPI(title="AI Browser Agent", lifespan=lifespan)
app.include_router(api_router)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/config/check")
async def config_check():
    return {
        "primary_llm_provider": settings.primary_llm_provider,
        "fallback_llm_provider": settings.fallback_llm_provider,
        "chroma_collection_name": settings.chroma_collection_name,
        "browser_headless": settings.browser_headless,
        "browser_pool_size": settings.browser_pool_size,
    }


# Mounted last so it never shadows the API routes above — serves the single-page
# console UI (index.html) at "/" and any other files under backend/app/static/.
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

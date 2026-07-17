from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.app.api.routes import router as api_router
from backend.app.api.task_manager import TaskManager
from backend.app.config import settings
from backend.app.core.browser_pool import BrowserPool


@asynccontextmanager
async def lifespan(app: FastAPI):
    # W10[A]: เปิด browser pool ล่วงหน้าตอน server startup (ไม่ใช่ตอน request แรกเข้ามา)
    # เพื่อให้ task แรกที่เข้ามาไม่ต้องรอ Chromium process launch เหมือนกับ task ถัดๆ ไป
    app.state.browser_pool = BrowserPool(size=settings.browser_pool_size)
    await app.state.browser_pool.start()
    app.state.task_manager = TaskManager()
    yield
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

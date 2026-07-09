from fastapi import FastAPI

from backend.app.config import settings

app = FastAPI(title="AI Browser Agent")


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
    }

"""W10[A]: Browser Pool (persistent).

เปิด browser process ล่วงหน้าตอน API server startup (main.py::lifespan) ให้แต่ละ task ยืมแทนการ launch
Chromium ใหม่ทุก request (~1-2 วินาที) — pool คุมระดับ Browser (ส่วนที่แพง) ผู้ยืมต้องเปิด/ปิด
BrowserContext ของตัวเอง (ไม่แชร์ cookie ข้าม task) ขนาดคงที่จาก settings.browser_pool_size
เกินโควตา request จะ await ใน asyncio.Queue จนมีตัวว่าง
"""

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator

from playwright.async_api import Browser, Playwright, async_playwright

from backend.app.config import settings


class BrowserPool:
    def __init__(self, size: int = 2, headless: bool | None = None):
        self._size = size
        self._headless = headless
        self._playwright: Playwright | None = None
        self._browsers: list[Browser] = []
        self._available: asyncio.Queue[Browser] = asyncio.Queue()
        self._started = False

    @property
    def size(self) -> int:
        return self._size

    @property
    def available(self) -> int:
        """จำนวน browser ที่ว่างอยู่ (GET /pool/status)"""
        return self._available.qsize()

    async def start(self) -> None:
        """launch browser ครบ size ตัว — เรียกซ้ำเป็น no-op"""
        if self._started:
            return
        is_headless = settings.browser_headless if self._headless is None else self._headless
        self._playwright = await async_playwright().start()
        for _ in range(self._size):
            browser = await self._playwright.chromium.launch(headless=is_headless)
            self._browsers.append(browser)
            await self._available.put(browser)
        self._started = True

    async def shutdown(self) -> None:
        """ปิด browser ทุกตัว + playwright — ต้องเรียกตอน shutdown ไม่งั้น Chromium process ค้าง"""
        if not self._started:
            return
        for browser in self._browsers:
            await browser.close()
        await self._playwright.stop()
        self._browsers = []
        self._available = asyncio.Queue()
        self._started = False

    async def acquire_one(self) -> Browser:
        """ยืม browser แบบไม่ auto-return (สำหรับ resource ข้าม request เช่น SessionRegistry) —
        ผู้เรียกต้อง release_one() เอง ไม่งั้น browser หายจาก pool ถาวร"""
        if not self._started:
            raise RuntimeError("BrowserPool ยังไม่ได้ start() — เรียก start() ตอน app startup ก่อน")
        return await self._available.get()

    async def release_one(self, browser: Browser) -> None:
        """คืน browser กลับเข้า pool — คู่กับ acquire_one()"""
        await self._available.put(browser)

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[Browser]:
        """ยืม browser (await ถ้าไม่มีตัวว่าง) แล้วคืนเสมอตอนออกจาก block แม้ throw — ไม่ปิด browser"""
        browser = await self.acquire_one()
        try:
            yield browser
        finally:
            await self.release_one(browser)

"""W10[A]: Browser Pool (persistent).

Orchestrator.run_task() เดิม (W1-W9) เปิด/ปิด playwright + browser process ใหม่ทุกครั้ง
ที่เรียก (async_playwright().start() -> chromium.launch() -> ... -> browser.close() ->
playwright.stop() ใน finally) — โอเคสำหรับ CLI demo ที่รันทีละ task แล้วจบโปรแกรม แต่ถ้า
เป็น API server ที่รับ request ต่อเนื่อง การเปิด Chromium process ใหม่ทุก request (~1-2
วินาที) เป็นต้นทุนที่ไม่จำเป็น — BrowserPool นี้เปิด browser process ไว้ล่วงหน้าตอน API
server startup (ดู main.py::lifespan) แล้วให้แต่ละ task "ยืม" browser ที่มีอยู่แล้วผ่าน
acquire() แทนที่จะเปิดใหม่ทุกครั้ง

ระดับที่ pool คุม = Browser (process) ไม่ใช่ Page/Context เพราะ process คือส่วนที่แพง
ที่สุดที่จะ reuse ได้จริง — แต่ละ task ที่ยืม browser ไปยังต้องได้ BrowserContext ของ
ตัวเอง (session แยกกัน ไม่แชร์ cookie/localStorage ข้าม task) ซึ่งเป็นหน้าที่ของฝั่งที่
เรียก acquire() (ดู orchestrator.py::run_task() เมื่อรับ browser param เข้ามา — เปิด
context ใหม่เอง ปิดแค่ context ตอนจบ ไม่ปิด browser)

ขนาด pool คงที่ (ไม่ auto-scale) — ตั้งจาก settings.browser_pool_size ตอน startup เกิน
โควตานี้ request ใหม่จะ await อยู่ใน queue จนกว่าจะมี browser ว่างคืนกลับมา (acquire()
เป็น async context manager ที่ block เองผ่าน asyncio.Queue.get() ไม่ต้อง busy-poll)
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
        """จำนวน browser ที่ว่างอยู่ตอนนี้ (ไม่ได้ถูกยืมไป) — ไว้ debug/monitor ผ่าน
        GET /pool/status"""
        return self._available.qsize()

    async def start(self) -> None:
        """เปิด playwright + launch browser ให้ครบ size ตัวล่วงหน้า — เรียกครั้งเดียวตอน
        API server startup (main.py::lifespan) เรียกซ้ำได้แบบ no-op ถ้า start ไปแล้ว"""
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
        """ปิด browser ทุกตัว + playwright — เรียกตอน API server shutdown (main.py::
        lifespan) ห้ามลืมเรียก ไม่งั้น Chromium process ค้างอยู่เบื้องหลัง"""
        if not self._started:
            return
        for browser in self._browsers:
            await browser.close()
        await self._playwright.stop()
        self._browsers = []
        self._available = asyncio.Queue()
        self._started = False

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[Browser]:
        """ยืม browser ตัวหนึ่งจาก pool — ถ้าทุกตัวถูกยืมไปหมด await จนกว่าจะมีตัวว่าง
        คืนกลับ (ผ่าน asyncio.Queue) คืน browser กลับเข้า pool เสมอตอนออกจาก block นี้
        (แม้ task ข้างในจะ throw ก็ตาม — finally) ไม่ปิด browser เอง (ยังใช้ต่อ task
        อื่นได้อีก)"""
        if not self._started:
            raise RuntimeError("BrowserPool ยังไม่ได้ start() — เรียก start() ตอน app startup ก่อน")
        browser = await self._available.get()
        try:
            yield browser
        finally:
            await self._available.put(browser)

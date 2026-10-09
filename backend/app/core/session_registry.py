"""core/session_registry.py — W12: Stateful Agent — session_id หนึ่งตัวผูกกับ Page/Context/Browser
ชุดเดียวที่มีชีวิตข้ามหลาย POST /tasks (in-memory, ไม่มี DB) ปิดเฉพาะตอน close()/close_all()
("New Session" หรือ server shutdown) — run_task(page=...) ข้าม teardown สำหรับ session ที่ลงทะเบียนไว้

ไม่ goto() เองตอนสร้าง session — ปล่อยให้ run_task()'s skip_initial_goto (W19 ใน orchestrator.py) ตัดสิน
W19: get_or_create() เช็ค is_healthy() ก่อนคืนเสมอ (browser/page อาจตายเพราะ user ปิดเอง/crash)
แล้วกู้คืนผ่าน _recover() เบาไปหนัก: page ใหม่ -> context ใหม่ -> browser ใหม่ทั้งชุด
"""

import asyncio
import secrets
import time
from dataclasses import dataclass, field
from typing import Optional

from playwright.async_api import Browser, BrowserContext, Page, Playwright, async_playwright

from backend.app.config import settings
from backend.app.core.browser_pool import BrowserPool
from backend.app.core.orchestrator import _detect_default_browser_channel, _launch_chromium
from backend.app.permission.rules import install_ssrf_guard
from backend.app.core.user_browser import (
    AskUserFunc,
    _open_new_tab_in_same_window,
    connect_user_browser,
    resolve_target_page,
)


class SessionOwnershipError(Exception):
    """Security (SEC-4 follow-up): session_id ที่มีอยู่แล้วถูกเรียกโดยไม่มี/ไม่ตรง owner_token"""


@dataclass
class BrowserSession:
    session_id: str
    mode: str  # "pool" | "owns" | "user_browser"
    page: Page
    context: Optional[BrowserContext]
    browser: Browser
    # None เฉพาะ mode="pool" (playwright เป็นของ BrowserPool) — mode อื่นต้อง stop() เองตอนปิด
    playwright: Optional[Playwright]
    # ตั้งเฉพาะ mode="pool" — ใช้คืน browser กลับ pool ตอนปิด
    pool: Optional[BrowserPool] = None
    created_at: float = field(default_factory=time.time)
    last_active_at: float = field(default_factory=time.time)
    # Security (SEC-4 follow-up): session_id เป็นแค่ string ที่ client เลือก — หลุดแล้วใครมี API key
    # เดียวกันก็แนบเข้า session คนอื่นได้ (รวม user_browser ที่มี login จริง) จึงเพิ่ม token สุ่ม 256-bit
    # ที่ต้องแนบกลับมาทุกครั้งที่ใช้ session ต่อ — คุม "ใช้ session ไหนได้" แยกจาก settings.api_key
    owner_token: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    # W19-6 (extracted_memory_buffer): structured items ล่าสุดจาก llm.extract_structured_items()
    # persist ข้ามเทิร์นให้อ้างแบบ ordinal ได้ ("เพลงที่ 3") — routes.py อ่าน/เขียนตรงๆ, [] ถ้ายังไม่เคย extract
    extracted_memory: list[dict] = field(default_factory=list)


def _check_owner(
    session: Optional[BrowserSession], session_id: str, owner_token: Optional[str], require_owner_token: bool,
) -> None:
    if session is not None and (
        (require_owner_token and not owner_token) or
        (owner_token is not None and session.owner_token != owner_token)
    ):
        raise SessionOwnershipError(f"session_id {session_id!r} ไม่ใช่ของ owner_token นี้")


class SessionRegistry:
    """เก็บ BrowserSession ต่อ session_id ใน memory ล้วนๆ (หายเมื่อ process restart — ยอมรับได้)"""

    def __init__(self) -> None:
        self._sessions: dict[str, BrowserSession] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def get(
        self, session_id: str, owner_token: Optional[str] = None, *, require_owner_token: bool = False,
    ) -> Optional[BrowserSession]:
        """owner_token=None = ไม่เช็คเจ้าของ (เฉพาะ internal call) — endpoint จริงต้องส่งมาเสมอ
        ไม่ตรง -> SessionOwnershipError (ไม่คืน None เงียบๆ ให้ดูเหมือนไม่มี session)"""
        session = self._sessions.get(session_id)
        _check_owner(session, session_id, owner_token, require_owner_token)
        return session

    def list(self) -> list[BrowserSession]:
        """สำหรับ GET /sessions — ล่าสุดก่อน"""
        return sorted(self._sessions.values(), key=lambda s: s.created_at, reverse=True)

    def _lock_for(self, session_id: str) -> asyncio.Lock:
        lock = self._locks.get(session_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[session_id] = lock
        return lock

    async def get_or_create(
        self,
        session_id: str,
        *,
        use_user_browser: bool,
        headless: Optional[bool],
        target_url: str,
        pool: BrowserPool,
        tab_reuse_policy: Optional[str],
        ask_user_func: Optional[AskUserFunc],
        owner_token: Optional[str] = None,
        require_owner_token: bool = False,
        target_tab_id: Optional[str] = None,
    ) -> BrowserSession:
        """มีอยู่แล้ว -> คืนตัวเดิม (กู้คืนก่อนถ้าไม่ healthy) ไม่มี -> สร้างตาม mode (use_user_browser >
        headless is False > pool) ด้วย double-checked lock ต่อ session_id

        Security (SEC-4 follow-up): owner_token เช็คเฉพาะ session ที่มีอยู่แล้ว ไม่ตรง -> SessionOwnershipError
        (ไม่สร้างทับ) ตอนสร้างใหม่ใช้ token ที่ caller ส่งมาเป็น secret, None = server generate เอง"""
        existing = self._sessions.get(session_id)
        if target_tab_id:
            if not use_user_browser:
                raise ValueError("target_tab_id requires use_user_browser")
            async with self._lock_for(session_id):
                existing = self._sessions.get(session_id)
                if existing is not None:
                    if not owner_token or existing.owner_token != owner_token:
                        raise SessionOwnershipError("Invalid session owner token")
                    if existing.mode != "user_browser" or existing.context is None:
                        raise ValueError("Agent Bar requires a user browser session")
                    # Re-resolve the sender on every request; never recover by opening a tab.
                    existing.page, _ = await resolve_target_page(
                        existing.context, target_url, ask_user_func, "always_reuse",
                        target_tab_id=target_tab_id,
                    )
                    existing.last_active_at = time.time()
                    return existing
                session = await self._create(
                    session_id, use_user_browser=True, headless=headless,
                    target_url=target_url, pool=pool, tab_reuse_policy="always_reuse",
                    ask_user_func=ask_user_func, owner_token=owner_token,
                    target_tab_id=target_tab_id,
                )
                self._sessions[session_id] = session
                return session
        if existing is not None:
            _check_owner(existing, session_id, owner_token, require_owner_token)
            return await self._reuse_or_recover(
                existing, target_url=target_url, pool=pool,
                tab_reuse_policy=tab_reuse_policy, ask_user_func=ask_user_func,
            )

        async with self._lock_for(session_id):
            existing = self._sessions.get(session_id)
            if existing is not None:
                _check_owner(existing, session_id, owner_token, require_owner_token)
                return await self._reuse_or_recover(
                    existing, target_url=target_url, pool=pool,
                    tab_reuse_policy=tab_reuse_policy, ask_user_func=ask_user_func,
                )
            session = await self._create(
                session_id,
                use_user_browser=use_user_browser,
                headless=headless,
                target_url=target_url,
                pool=pool,
                tab_reuse_policy=tab_reuse_policy,
                ask_user_func=ask_user_func,
                owner_token=owner_token,
            )
            self._sessions[session_id] = session
            return session

    async def _reuse_or_recover(
        self,
        existing: BrowserSession,
        *,
        target_url: str,
        pool: BrowserPool,
        tab_reuse_policy: Optional[str],
        ask_user_func: Optional[AskUserFunc],
    ) -> BrowserSession:
        existing.last_active_at = time.time()
        if self.is_healthy(existing):
            return existing
        recovered = await self._recover(
            existing, target_url=target_url, pool=pool,
            tab_reuse_policy=tab_reuse_policy, ask_user_func=ask_user_func,
        )
        self._sessions[recovered.session_id] = recovered
        return recovered

    @staticmethod
    def is_healthy(session: BrowserSession) -> bool:
        """browser ยังต่ออยู่และ page ยังไม่ปิด — sync (Playwright เช็คแค่ flag ใน memory) ไม่ throw
        (error ระหว่างเช็ค = ไม่ healthy) caller ใช้เช็คอย่างเดียวโดยไม่ recover ได้"""
        try:
            if not session.browser.is_connected():
                return False
            if session.page.is_closed():
                return False
            return True
        except Exception:
            return False

    async def _recover(
        self,
        session: BrowserSession,
        *,
        target_url: str,
        pool: BrowserPool,
        tab_reuse_policy: Optional[str],
        ask_user_func: Optional[AskUserFunc],
    ) -> BrowserSession:
        """กู้คืน session ที่ไม่ healthy จากเบาไปหนัก (Playwright reconnect page ที่ปิดแล้วไม่ได้):
          1) browser ยังต่ออยู่ -> page ใหม่ในบริบทเดิม
          2) ไม่สำเร็จ + mode "pool" -> context ใหม่บน browser เดิม
          3) ไม่งั้น -> ปิดของเก่าแบบ best-effort แล้วสร้างใหม่ทั้งชุด session_id/owner_token เดิม
        คืน session ที่ healthy เสมอ — โปร่งใสต่อ caller"""
        browser_alive = False
        try:
            browser_alive = session.browser.is_connected()
        except Exception:
            browser_alive = False

        if browser_alive:
            try:
                if session.mode == "user_browser" and session.context is not None:
                    # W_fix: ห้าม context.new_page() ใน Chrome จริงของ user — CDP Target.createTarget()
                    # อาจเปิด window ใหม่แทน tab (bug ที่ user รายงานจริง) ใช้ helper ที่การันตี window เดิม
                    session.page = await _open_new_tab_in_same_window(session.context)
                elif session.context is not None:
                    session.page = await session.context.new_page()
                else:
                    # Security (SSRF follow-up): browser.new_page() สร้าง context ใหม่โดยนัย ต้อง install guard ใหม่
                    session.page = await session.browser.new_page()
                    await install_ssrf_guard(session.page)
                session.last_active_at = time.time()
                return session
            except Exception:
                pass

            if session.mode == "pool":
                try:
                    new_context = await session.browser.new_context()
                    await install_ssrf_guard(new_context)
                    session.context = new_context
                    session.page = await new_context.new_page()
                    session.last_active_at = time.time()
                    return session
                except Exception:
                    pass

        await self._best_effort_close(session)
        return await self._create(
            session.session_id,
            use_user_browser=(session.mode == "user_browser"),
            headless=(False if session.mode == "owns" else None),
            target_url=target_url,
            pool=pool,
            tab_reuse_policy=tab_reuse_policy,
            ask_user_func=ask_user_func,
            # Security (SEC-4 follow-up): ต้องคง owner_token เดิม ไม่งั้น client ใช้ session ต่อไม่ได้หลัง recovery
            owner_token=session.owner_token,
        )

    async def _best_effort_close(self, session: BrowserSession) -> None:
        """ปิด resource ตาม mode โดยกลืน exception ทุกจุด — ใช้ทั้ง _recover() และ close()
        (W27: close() เดิมไม่มี error handling -> race กับ task ที่เพิ่ง stop/browser ที่ user ปิดเอง
        ทำให้ 500 ทั้งที่ pop ออกจาก registry แล้ว resource รั่ว)

        mode="pool": คืน browser กลับ pool เฉพาะตัวที่ยัง is_connected() — release_one() ไม่เช็คสถานะ
        คืนตัวที่ตายแล้วจะ poison ทั้ง pool (ยอมเสียสล็อตถาวรแทน)"""
        try:
            if session.mode == "pool" and session.context is not None:
                await session.context.close()
        except Exception:
            pass
        try:
            if session.mode == "user_browser":
                if session.playwright is not None:
                    await session.playwright.stop()
            elif session.mode == "owns":
                await session.browser.close()
                if session.playwright is not None:
                    await session.playwright.stop()
            elif session.pool is not None and session.browser.is_connected():
                await session.pool.release_one(session.browser)
        except Exception:
            pass

    async def _create(
        self,
        session_id: str,
        *,
        use_user_browser: bool,
        headless: Optional[bool],
        target_url: str,
        pool: BrowserPool,
        tab_reuse_policy: Optional[str],
        ask_user_func: Optional[AskUserFunc],
        owner_token: Optional[str] = None,
        target_tab_id: Optional[str] = None,
    ) -> BrowserSession:
        session_kwargs = {"owner_token": owner_token} if owner_token is not None else {}
        if use_user_browser:
            playwright = await async_playwright().start()
            browser = await connect_user_browser(playwright, settings.user_browser_cdp_url)
            # ห้าม browser.new_context() — ต้องใช้ context จริงที่มี cookie/login ของ user
            context = browser.contexts[0]
            # Security (SSRF follow-up): install ระดับ context ครอบคลุมทุก tab ทั้งเดิมและอนาคต
            # (รวม _open_new_tab_in_same_window() ตอน recover — ไม่ต้องติดตั้งซ้ำ)
            await install_ssrf_guard(context)
            try:
                page, _opened_new_tab = await resolve_target_page(
                    context, target_url, ask_user_func,
                    tab_reuse_policy or settings.user_browser_tab_reuse_policy,
                    target_tab_id=target_tab_id,
                )
            except Exception:
                await playwright.stop()
                raise
            return BrowserSession(session_id, "user_browser", page, context, browser, playwright, **session_kwargs)

        if headless is False:
            playwright = await async_playwright().start()
            channel = _detect_default_browser_channel()
            browser = await _launch_chromium(playwright, headless=False, channel=channel)
            page = await browser.new_page()
            await install_ssrf_guard(page)
            return BrowserSession(session_id, "owns", page, None, browser, playwright, **session_kwargs)

        browser = await pool.acquire_one()
        context = await browser.new_context()
        await install_ssrf_guard(context)
        page = await context.new_page()
        return BrowserSession(session_id, "pool", page, context, browser, None, pool=pool, **session_kwargs)

    async def close(
        self, session_id: str, owner_token: Optional[str] = None, *, require_owner_token: bool = False,
    ) -> bool:
        """ปิด session — False ถ้าไม่พบ session_id ไม่ throw จากการปิด resource (W27: ใช้ _best_effort_close())
        Security (SEC-4 follow-up): owner_token ไม่ตรง -> SessionOwnershipError (ปิด = kill browser จริง)"""
        _check_owner(self._sessions.get(session_id), session_id, owner_token, require_owner_token)
        session = self._sessions.pop(session_id, None)
        self._locks.pop(session_id, None)
        if session is None:
            return False
        await self._best_effort_close(session)
        return True

    async def close_all(self) -> None:
        """เรียกตอน shutdown (main.py::lifespan) ก่อน browser_pool.shutdown() เสมอ — คืน browser กลับ pool ก่อน"""
        for session_id in list(self._sessions):
            await self.close(session_id)

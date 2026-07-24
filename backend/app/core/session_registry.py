"""core/session_registry.py — W12: Stateful Agent (persistent Page/Context/Browser
ข้าม HTTP request)

เดิม (W1-W11): ทุก POST /tasks acquire browser/context/page ของตัวเองใหม่ทุกครั้ง (จาก
BrowserPool หรือ launch เองหรือต่อ CDP) แล้วปิด/คืนกลับตอนจบ task เดียวกันนั้นเสมอ (ดู
orchestrator.py::run_task() finally block) — ทำให้ follow-up command ในบทสนทนาเดียวกัน
(เช่น "เปิดเว็บ" แล้วต่อด้วย "sign in") ไม่มีทางทำงานต่อจากหน้าเดิมได้เลย เพราะหน้าเดิม
ถูกปิด/คืนไปแล้วตั้งแต่ task ก่อนหน้าจบ

ใหม่: session_id หนึ่งตัว (จาก Test Console — ดู backend/app/static/index.html) ผูกกับ
Page/Context/Browser ชุดเดียวที่ "มีชีวิตอยู่ข้ามหลาย POST /tasks" — สร้างครั้งแรกตอน
session_id ยังไม่เคยเจอ แล้วถูกเก็บไว้ใน memory ของ process (ไม่มี DB เหมือน
task_manager.py) ให้ request ถัดๆ ไปที่ session_id เดิมมาเจอ page ตัวเดิมได้ทันที — ปิด
ก็ต่อเมื่อ close()/close_all() ถูกเรียกตรงๆ (ปุ่ม "New Session" หรือตอน server shutdown)
เท่านั้น ไม่มีการปิดอัตโนมัติจาก run_task() อีกต่อไปสำหรับ session ที่ลงทะเบียนไว้ (ดู
orchestrator.py::run_task(page=...) — managed_externally=True ข้าม teardown ทั้งหมด)

ไม่ทำ goto() เองตอนสร้าง session ใหม่ — แค่คืน Page กลับไป (ว่างเปล่าสำหรับโหมด pool/owns,
อาจมีเนื้อหาอยู่แล้วถ้าเป็น tab ที่ resolve_target_page() เลือก reuse ในโหมด CDP) แล้วปล่อย
ให้ orchestrator.run_task()'s skip_initial_goto (เช็ค domain ปัจจุบันเทียบกับ url เป้าหมาย
จริง — ดู W19 ใน orchestrator.py) ตัดสินใจเองว่าต้อง goto ไหม — ไม่ duplicate logic นั้นซ้ำ
ในนี้

W19: get_or_create() เดิมคืน session ที่เจอใน dict ตรงๆ โดยไม่เช็คว่ายัง "ใช้งานได้จริง"
ไหมเลย — ถ้า browser process ถูกปิดไปแล้วเอง (user ปิดหน้าต่างตรงๆ, crash, ถูก OS ฆ่าทิ้ง)
หรือ page ถูกปิดไปแล้ว (user ปิด tab เอง) แต่ session_id เดิมยังถูกเรียกใช้ซ้ำ จะได้ page/
browser object ที่ตายไปแล้วกลับมา ทำให้ทุก operation ถัดไป (goto/perceive/action) พังหมด
— ตอนนี้เช็ค is_healthy() ก่อนคืนทุกครั้ง ถ้าไม่ healthy จะกู้คืนอัตโนมัติผ่าน _recover()
(ไล่ลองเบาไปหนัก: เปิด page ใหม่ในบริบทเดิม -> เปิด context ใหม่บน browser เดิม -> ปิดของ
เก่าทิ้งแล้วเปิด browser ใหม่ทั้งชุด) ไม่ fail ทันทีจากแค่ resource เดียวหลุด
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional

from playwright.async_api import Browser, BrowserContext, Page, Playwright, async_playwright

from backend.app.config import settings
from backend.app.core.browser_pool import BrowserPool
from backend.app.core.orchestrator import _detect_default_browser_channel, _launch_chromium
from backend.app.core.user_browser import AskUserFunc, connect_user_browser, resolve_target_page


@dataclass
class BrowserSession:
    session_id: str
    mode: str  # "pool" | "owns" | "user_browser"
    page: Page
    context: Optional[BrowserContext]
    browser: Browser
    # None เฉพาะ mode="pool" — playwright driver instance เป็นของ BrowserPool ไม่ใช่ของ
    # session (pool เปิด/ปิด playwright ของตัวเองแยกต่างหาก) mode อื่นเปิด playwright
    # driver ของตัวเองตอนสร้าง session เลยต้องเก็บไว้ stop() เองตอนปิด
    playwright: Optional[Playwright]
    # ตั้งเฉพาะ mode="pool" — ต้องใช้คืน browser กลับ pool ตอนปิด session
    pool: Optional[BrowserPool] = None
    created_at: float = field(default_factory=time.time)
    last_active_at: float = field(default_factory=time.time)


class SessionRegistry:
    """เก็บ BrowserSession ต่อ session_id ใน memory ล้วนๆ (ตกเมื่อ process restart ได้
    เหมือน task_manager.py::TaskManager — ยอมรับได้ ยังไม่มี requirement เรื่อง
    persistence ข้าม restart)"""

    def __init__(self) -> None:
        self._sessions: dict[str, BrowserSession] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def get(self, session_id: str) -> Optional[BrowserSession]:
        return self._sessions.get(session_id)

    def list(self) -> list[BrowserSession]:
        """ไว้ debug/monitor ผ่าน GET /sessions — session ล่าสุดก่อน (เหมือน
        task_manager.py::TaskManager.list())"""
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
    ) -> BrowserSession:
        """session_id เคยเจอมาก่อน -> คืนตัวเดิมถ้ายัง healthy (ดู is_healthy()) ถ้าไม่
        healthy แล้วจะกู้คืนอัตโนมัติก่อนคืน (ดู _recover() — ไม่ fail ทันที) ไม่เคยเจอ ->
        สร้างใหม่ตาม mode ที่กำหนด (ลำดับความสำคัญเดียวกับ routes.py::create_task() เดิม:
        use_user_browser ก่อน, ตามด้วย headless is False, สุดท้าย fallback ไป pool) แล้ว
        ลงทะเบียนไว้ — ใช้ double-checked locking ต่อ session_id กัน 2 request ที่มาถึง
        พร้อมกันด้วย session_id ใหม่ตัวเดียวกันสร้างซ้ำ 2 รอบ (ไม่ควรเกิดจาก UI ปกติเพราะ
        frontend รอ task ก่อนจบก่อนส่ง follow-up อยู่แล้ว แต่กันไว้)
        """
        existing = self._sessions.get(session_id)
        if existing is not None:
            return await self._reuse_or_recover(
                existing, target_url=target_url, pool=pool,
                tab_reuse_policy=tab_reuse_policy, ask_user_func=ask_user_func,
            )

        async with self._lock_for(session_id):
            existing = self._sessions.get(session_id)
            if existing is not None:
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
        """เช็คว่า browser/page ของ session นี้ยังใช้งานได้จริงไหม — sync ล้วนๆ
        (is_connected()/is_closed() ของ Playwright แค่ดู flag ที่ driver จำไว้ในหน่วยความจำ
        ไม่ได้ยิงไปเช็ค IPC จริงฝั่ง browser process ถึงไม่ต้อง await) ครอบคลุมทั้ง browser
        disconnected/closed และ page ถูกปิด (user ปิด tab/window เอง, crash, ฯลฯ) — ไม่
        throw เด็ดขาด ถือว่า "ไม่ healthy" ถ้าเช็คแล้ว error เอง (เช่น attribute หายไปเพราะ
        object ถูกทำลายไปแล้วบางส่วน) ให้ผู้เรียก (routes.py::generate_plan ที่ไม่อยากแตะ
        browser เองเลย) ใช้แค่เช็คอย่างเดียวโดยไม่ต้อง recover ก็ได้"""
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
        """กู้คืน session ที่ is_healthy() ล้มเหลว — ไล่ลองจากทางที่ "เบา" ที่สุดไปหา
        "หนัก" ที่สุดเสมอ ไม่รื้อของทั้งชุดถ้ายังไม่จำเป็นจริงๆ (ตรงตาม requirement:
        1. reconnect เข้า page เดิม, 2. เปิด page ใหม่ในบริบทเดิม, 3. เปิด context ใหม่,
        4. เปิด browser ใหม่ — Playwright ไม่มี API "reconnect" เข้า page ที่ปิดไปแล้วจริง
        เลยข้ามขั้นตอน 1 ไปเริ่มที่ขั้น 2 ตรงๆ):
          1) browser ยังต่ออยู่จริง (is_connected()) แค่ page เดิมปิดไปแล้ว -> เปิด page
             ใหม่ในบริบทเดิม (context เดิมถ้ามี — mode "pool"/"user_browser", ไม่งั้น
             browser.new_page() ตรงๆ สำหรับ mode "owns" ที่ไม่มี context แยกจาก browser)
          2) ข้อ 1 ทำไม่สำเร็จด้วย (context เองก็พังไปด้วย) -> เปิด context ใหม่บน browser
             เดิม (เฉพาะ mode "pool" ที่มี context แยกเป็นของตัวเองจริง)
          3) browser หลุดการเชื่อมต่อไปแล้วจริง หรือกู้ข้อ 1-2 ไม่สำเร็จเลย -> ปิดของเก่า
             เท่าที่ยังทำได้ (เงียบๆ ไม่ throw ต่อให้ปิดไม่สำเร็จ — ของเดิมมักพังอยู่แล้ว)
             แล้วสร้าง session ใหม่ทั้งชุดด้วยพารามิเตอร์เดิมที่ session นี้ถูกสร้างครั้งแรก
             (mode เดิม -> use_user_browser/headless ที่ตรงกัน) session_id เดิมเป๊ะ ผู้เรียก
             (เช่น orchestrator.run_task ที่กำลังจะเริ่ม task ด้วย session นี้) ไม่มีทางรู้
             เลยว่าข้างหลังมีการกู้คืนเกิดขึ้น
        คืน BrowserSession ที่ผ่าน is_healthy() แล้วเสมอ ไม่มีทาง "fail ทันที" จากแค่
        resource เดียวหลุด"""
        browser_alive = False
        try:
            browser_alive = session.browser.is_connected()
        except Exception:
            browser_alive = False

        if browser_alive:
            try:
                if session.context is not None:
                    session.page = await session.context.new_page()
                else:
                    session.page = await session.browser.new_page()
                session.last_active_at = time.time()
                return session
            except Exception:
                pass

            if session.mode == "pool":
                try:
                    new_context = await session.browser.new_context()
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
        )

    async def _best_effort_close(self, session: BrowserSession) -> None:
        """ปิด resource ของ session นี้ตาม mode โดยกลืน exception ทุกจุด — เรียกจาก 2 ที่:
        (1) _recover() ตอนกู้คืน session ที่รู้อยู่แล้วว่า resource เดิมพังบางส่วน/ทั้งหมด
        (ปิดของที่พังไปแล้วซ้ำมักจะ throw เอง — เช่น context.close() บน browser ที่
        disconnect ไปแล้ว) ไม่ให้เรื่องนั้นบล็อกการสร้าง session ใหม่ทดแทน (2) W27:
        close() (ปุ่ม "kill session" บน Test Console) — เดิม close() เขียนตรรกะปิดซ้ำเองแบบ
        ไม่มี error handling เลย ทำให้ race กับ task ที่เพิ่งถูก stop (ดู
        task_manager.py::cancel()) หรือ browser ที่ user ปิดเองด้วยมือไปก่อนแล้ว ทำให้
        endpoint 500 ทั้งที่ session ถูก pop ออกจาก registry ไปแล้ว (ดูเหมือนปิดสำเร็จแต่
        resource จริงรั่ว) — รวมเป็น method เดียวกัน ให้ทั้ง 2 เส้นทางได้ error handling
        เดียวกัน

        mode="pool": ห้าม release_one() browser ที่ disconnect ไปแล้วกลับเข้า pool
        เด็ดขาด — release_one() ไม่เช็คสถานะ browser เลย (แค่ put ลง queue ตรงๆ) ถ้าคืน
        ตัวที่ตายแล้วกลับไป task อื่นในอนาคตที่ acquire_one() ได้ตัวนี้ไปจะพังตามทันที
        (poison ทั้ง pool) — ปล่อยตัวที่ตายแล้วทิ้งไปเฉยๆ (pool เสียสล็อตนี้ถาวร ยอมรับได้
        มากกว่าทำ pool เสียหายทั้งระบบ) คืนกลับ pool เฉพาะตัวที่ยัง is_connected() จริง"""
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
    ) -> BrowserSession:
        if use_user_browser:
            playwright = await async_playwright().start()
            browser = await connect_user_browser(playwright, settings.user_browser_cdp_url)
            # ห้าม browser.new_context() เด็ดขาด — ต้องใช้ context จริงที่มี cookie/login
            # ของ user อยู่แล้ว (ดูเหตุผลเดียวกับ orchestrator.py::run_task())
            context = browser.contexts[0]
            page, _opened_new_tab = await resolve_target_page(
                context, target_url, ask_user_func,
                tab_reuse_policy or settings.user_browser_tab_reuse_policy,
            )
            return BrowserSession(session_id, "user_browser", page, context, browser, playwright)

        if headless is False:
            playwright = await async_playwright().start()
            channel = _detect_default_browser_channel()
            browser = await _launch_chromium(playwright, headless=False, channel=channel)
            page = await browser.new_page()
            return BrowserSession(session_id, "owns", page, None, browser, playwright)

        browser = await pool.acquire_one()
        context = await browser.new_context()
        page = await context.new_page()
        return BrowserSession(session_id, "pool", page, context, browser, None, pool=pool)

    async def close(self, session_id: str) -> bool:
        """ปิด session — คืน False ถ้าไม่พบ session_id นี้ (ปิดไปแล้ว/ไม่เคยมีอยู่จริง)
        ปิดเฉพาะ resource ที่ session นี้เป็นเจ้าของเองจริงๆ ตาม mode

        W27: แก้บั๊ก "ปุ่ม kill session ใช้งานจริงไม่ได้" — เดิม method นี้ไม่มี try/except
        เลยสักจุด (ต่างจาก _best_effort_close() ด้านบนที่กลืน exception ทุกจุดอยู่แล้ว) ถ้า
        browser.close()/context.close() ล้มเหลว (เช่น race กับ task ที่เพิ่งถูก
        TaskManager.cancel() แต่ยังไม่ทันหยุดใช้ page จริงๆ — ดู task_manager.py::cancel()
        กับ routes.py::stop_task() ที่แก้คู่กัน หรือ user ปิดหน้าต่าง browser จริงเองด้วยมือ
        ก่อนกดปุ่มนี้) exception จะหลุดออกไปจาก endpoint ตรงๆ เป็น 500 — แต่ session ก็ถูก
        pop() ออกจาก self._sessions ไปแล้วก่อนหน้านั้น (บรรทัดบน) ทำให้ดูเหมือน "ปิดแล้ว"
        จาก state ของ registry แต่ resource จริง (browser process/หน้าต่างที่มองเห็นได้/
        browser ที่ยืมจาก pool) อาจไม่ถูกปิด/คืนจริงเลย — ตอนนี้ delegate ไปที่
        _best_effort_close() ตัวเดียวกับที่ _recover() ใช้อยู่แล้ว (กลืน exception ทุกจุด +
        เช็ค is_connected() ก่อนคืน browser กลับ pool กัน poison pool ด้วย browser ที่ตายไป
        แล้ว) แทนที่จะเขียนตรรกะเดิมซ้ำแบบไม่มี error handling"""
        session = self._sessions.pop(session_id, None)
        self._locks.pop(session_id, None)
        if session is None:
            return False
        await self._best_effort_close(session)
        return True

    async def close_all(self) -> None:
        """ปิดทุก session ที่เหลืออยู่ — เรียกตอน API server shutdown (main.py::lifespan)
        ก่อน browser_pool.shutdown() เสมอ (คืน browser ที่ session ถือไว้กลับ pool ก่อน
        ที่ pool จะปิด browser ทุกตัวทิ้ง)"""
        for session_id in list(self._sessions):
            await self.close(session_id)

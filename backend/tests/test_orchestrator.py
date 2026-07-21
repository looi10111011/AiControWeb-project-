from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.app.core import llm
from backend.app.core.actions import ActionResult
from backend.app.core.memory import ShortTermMemory
from backend.app.core.orchestrator import (
    Orchestrator,
    _GEMINI_COMPACT_AFTER_STEPS,
    _GEMINI_KEEP_RECENT_STEPS,
    _LONG_TERM_MEMORY_CHUNKS_PER_STEP,
    _MAX_CONSECUTIVE_IDENTICAL_ACTIONS,
    _MAX_PREMATURE_FALSE_FINISH_RETRIES,
    _MAX_PREMATURE_TRUE_FINISH_RETRIES,
    _PREMATURE_FALSE_FINISH_NUDGE,
    _PREMATURE_TRUE_FINISH_NUDGE,
    _RAG_CHUNKS_PER_STEP,
    _build_gemini_history_digest,
    _build_nudge_message,
    _compact_gemini_messages,
    _make_dialog_handler,
)

# ทุกเทสต์ mock ทั้ง Playwright และ llm.next_action — ไม่เปิด browser จริง ไม่ยิง LLM API จริง
# สำคัญ: ต้องส่ง provider="anthropic" ให้ run_task() ตรงๆ เสมอ ห้ามปล่อยให้ fallback ไป
# settings.llm_provider เพราะค่านั้นอ่านจาก .env ของเครื่อง dev แต่ละคน — ถ้า .env ตั้ง
# LLM_PROVIDER=groq ไว้ (เช่นตอนทดสอบ) แล้วเทสต์ไม่ pin provider ให้ตรงกับที่ mock ไว้
# (llm.next_action) จะหลุดไปเรียก llm.next_action_groq ตัวจริงที่ไม่ได้ mock -> ยิง Groq
# API จริงระหว่างรัน pytest (เคยเกิดขึ้นมาแล้ว)

_GOTO_OK = ActionResult(True, "goto", "ไปที่ url")
_WAIT_OK = ActionResult(True, "wait_stable", "หน้านิ่งแล้ว")


# pacing delay ท้ายทุก step (_STEP_PACING_DELAY_SECONDS) กันไม่ให้ test suite ช้าจริง —
# เหมือน test_actions.py ที่ mock asyncio.sleep กัน _ACTION_RETRY_DELAY_SEC ค้าง
@pytest.fixture(autouse=True)
def _no_real_sleep():
    with patch("backend.app.core.orchestrator.asyncio.sleep", AsyncMock()) as mock_sleep:
        yield mock_sleep


# _login_form_needs_password() เรียก page.locator() จริง (sync ใน Playwright จริง)
# แต่ mock_page เป็น AsyncMock เปล่าๆ ที่ทำให้ทุก attribute เป็น async หมด — เรียกจริง
# จะได้ RuntimeWarning (coroutine ไม่ถูก await) แล้ว fallback เป็น False อยู่ดีเพราะ
# ครอบด้วย try/except — mock ให้ตรงๆ แทนกันเทสต์อื่นๆ ที่ไม่ได้ตั้งใจทดสอบ guard นี้รก
# ด้วย warning, default False (ไม่มี password field ว่างอยู่ เหมือนหน้าเว็บทั่วไป) —
# เทสต์เฉพาะของ guard นี้ override เป็น True เองในเทสต์
@pytest.fixture(autouse=True)
def _no_password_field_by_default():
    with patch("backend.app.core.orchestrator._login_form_needs_password", AsyncMock(return_value=False)):
        yield


# W18: _maybe_auto_login() อ่าน storage.load_credentials() ผ่าน lazy import ตอนรัน
# จริง — ไม่ได้ผูกกับ mock chain ไหนในไฟล์นี้เลย ถ้าไม่ mock ทิ้งไว้จะไปอ่าน
# settings.site_manuals_dir ตัวจริงบนเครื่อง dev (ไม่ได้ isolate เหมือน
# test_site_learning_storage.py) — mock_page.url บางเทสต์ตั้งเป็นโดเมนจริง (เช่น
# www.saucedemo.com สำหรับโหมด user_browser) ซึ่งอาจมี credentials.json จริงเก็บไว้
# จากการใช้งานจริงของ user ทำให้ _maybe_auto_login พยายาม extract_page()/fill() บน
# mock_page (AsyncMock เปล่าๆ ไม่ใช่ page จริง) เกิด RuntimeWarning รกไม่เกี่ยวกับสิ่งที่
# เทสต์ไฟล์นี้ตั้งใจพิสูจน์เลย — mock เป็น no-op default เหมือนแพทเทิร์นเดียวกับ
# _no_password_field_by_default ด้านบน
@pytest.fixture(autouse=True)
def _no_auto_login_by_default():
    with patch("backend.app.core.orchestrator._maybe_auto_login", AsyncMock(return_value=None)):
        yield


# W7[A] (long-term): long_term_memory.recall()/record_task() ทั้งคู่เป็นงาน sync ที่
# แตะ ChromaDB จริง (disk I/O + local embedding model) — mock default ไว้ให้ทุกเทสต์
# กันไม่ให้ pytest ไปเขียน/อ่าน collection จริงบนเครื่อง dev โดยไม่ตั้งใจ (ต่างจาก
# retriever.retrieve ของ W6[B] ที่เทสต์เก่าบางเคสไม่ได้ mock — ตัวนั้นเป็นแค่ read
# ส่วน record_task() เป็น write จริง ปล่อยไม่ mock จะสะสม test noise ในข้อมูลจริง)
# เทสต์เฉพาะของ long-term memory override เป็นค่าที่ต้องการเองภายใน with patch(...)
@pytest.fixture(autouse=True)
def _no_real_long_term_memory():
    with patch("backend.app.core.orchestrator.long_term_memory.recall", return_value=[]) as mock_recall, \
         patch("backend.app.core.orchestrator.long_term_memory.record_task", return_value=None) as mock_record:
        yield mock_recall, mock_record


def _patch_browser():
    """mock chain ให้ตรงกับของจริง:
    async_playwright() -> (sync) helper -> await .start() -> playwright
    -> await playwright.chromium.launch() -> browser -> await browser.new_page() -> page
    """
    mock_page = AsyncMock()
    # page.on() เป็น sync method จริงใน Playwright (ลงทะเบียน event listener เฉยๆ ไม่
    # await) — mock_page เป็น AsyncMock เปล่าๆ ทำให้ .on() กลายเป็น async mock ไปด้วย
    # โดยไม่ตั้งใจ (W9[A] เพิ่ม page.on("dialog", ...) ใน run_task() แล้วไม่เคย await
    # ผลลัพธ์เพราะของจริงไม่ต้อง await) ทิ้ง RuntimeWarning ไว้ทุกเทสต์ที่ใช้ fixture นี้
    # — แก้ให้ตรงกับพฤติกรรมจริงเหมือนที่เคยทำกับ page.locator() ใน W5/W6
    mock_page.on = MagicMock()
    # W12: page.url เป็น plain string property จริงใน Playwright — หน้าใหม่ที่เพิ่งเปิด
    # (browser.new_page()) เริ่มที่ "about:blank" เสมอ ต้อง set ตรงๆ ไม่งั้น AsyncMock()
    # auto-mock .url เป็น child mock object ที่ไม่เท่ากับ "about:blank"/"" เลย ทำให้
    # skip_initial_goto (เช็คจาก page.url ตรงๆ ดู orchestrator.py) เข้าใจผิดว่าหน้านี้มี
    # เนื้อหาอยู่แล้ว ข้าม goto() ทั้งที่ควร goto จริงเหมือนพฤติกรรมเดิมทุกประการ
    mock_page.url = "about:blank"
    mock_browser = AsyncMock()
    mock_browser.new_page = AsyncMock(return_value=mock_page)
    mock_browser.close = AsyncMock()

    mock_playwright_instance = AsyncMock()
    mock_playwright_instance.chromium.launch = AsyncMock(return_value=mock_browser)
    mock_playwright_instance.stop = AsyncMock()

    mock_p_helper = MagicMock()
    mock_p_helper.start = AsyncMock(return_value=mock_playwright_instance)

    mock_async_playwright = MagicMock(return_value=mock_p_helper)

    return mock_async_playwright, mock_browser, mock_playwright_instance


# --- W9[A] "handle error states (popup)": auto-dismiss JS dialog (alert/confirm/
# prompt/beforeunload) กันไม่ให้ dialog ที่ไม่มีใคร handle ค้างบล็อกหน้าเว็บทั้งหมด


@pytest.mark.asyncio
async def test_dialog_handler_dismisses_and_records_to_memory():
    """dialog handler ต้อง dismiss() เสมอ (ไม่ accept — ปลอดภัยกว่า เพราะ confirm()
    บางเว็บผูกกับ action ทำลายข้อมูล) + บันทึกเข้า short-term memory ให้ step ถัดไป
    เห็นผ่าน failed_actions_summary() pipe เดิมจาก W7[A] (ไม่ต้องเพิ่ม context section
    ใหม่)"""
    memory = ShortTermMemory()
    mock_dialog = AsyncMock()
    mock_dialog.type = "confirm"
    mock_dialog.message = "แน่ใจนะว่าจะออกจากหน้านี้?"

    handler = _make_dialog_handler(memory, verbose=False)
    await handler(mock_dialog)

    mock_dialog.dismiss.assert_awaited_once()
    mock_dialog.accept.assert_not_awaited()
    summary = memory.failed_actions_summary()
    assert "confirm" in summary
    assert "แน่ใจนะว่าจะออกจากหน้านี้?" in summary


@pytest.mark.asyncio
async def test_run_task_registers_dialog_handler_on_page():
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    mock_page = mock_browser.new_page.return_value

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=[]), \
         patch(
             "backend.app.core.orchestrator.llm.next_action",
             AsyncMock(return_value=("finish_task", {"success": True, "message": "เสร็จ"}, "", [], llm.TokenUsage())),
         ):
        await Orchestrator().run_task("https://example.com", "goal", provider="anthropic")

    mock_page.on.assert_called_once()
    assert mock_page.on.call_args.args[0] == "dialog"


def _patch_pooled_browser():
    """W10[A]: mock chain สำหรับ browser ที่ยืมมาจาก BrowserPool.acquire() (ต่างจาก
    _patch_browser() ด้านบนที่จำลอง async_playwright().start() ทั้งสาย) — ตัวนี้ไม่มี
    playwright/chromium.launch() เกี่ยวข้องเลย เพราะ browser ถูกส่งเข้ามาสำเร็จรูปแล้ว
    ต้องเปิดแค่ context ใหม่: await browser.new_context() -> context
    -> await context.new_page() -> page"""
    mock_page = AsyncMock()
    mock_page.on = MagicMock()
    mock_page.url = "about:blank"  # W12: หน้าใหม่จาก context.new_page() เริ่มว่างเปล่าเสมอ
    mock_context = AsyncMock()
    mock_context.new_page = AsyncMock(return_value=mock_page)
    mock_context.close = AsyncMock()
    mock_browser = AsyncMock()
    mock_browser.new_context = AsyncMock(return_value=mock_context)
    return mock_browser, mock_context, mock_page


@pytest.mark.asyncio
async def test_run_task_with_pooled_browser_uses_context_not_new_browser_process():
    """เมื่อส่ง browser= เข้ามาเอง (จำลอง BrowserPool.acquire()) ห้ามเปิด
    async_playwright()/chromium.launch() ใหม่เด็ดขาด (นั่นคือทั้งจุดของ pool — reuse
    browser process เดิม) ต้องเปิดแค่ BrowserContext ใหม่แทน แล้วปิดแค่ context ตอนจบ
    ไม่แตะ browser (ของ pool ต้องคืนกลับให้ยืมต่อได้ ไม่ถูกปิดทิ้ง)"""
    mock_browser, mock_context, mock_page = _patch_pooled_browser()

    with patch("backend.app.core.orchestrator.async_playwright") as mock_async_playwright, \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=[]), \
         patch(
             "backend.app.core.orchestrator.llm.next_action",
             AsyncMock(return_value=("finish_task", {"success": True, "message": "เสร็จ"}, "", [], llm.TokenUsage())),
         ):
        result = await Orchestrator().run_task(
            "https://example.com", "goal", provider="anthropic", browser=mock_browser,
        )

    assert result["success"] is True
    mock_async_playwright.assert_not_called()
    mock_browser.new_context.assert_awaited_once()
    mock_context.new_page.assert_awaited_once()
    mock_page.on.assert_called_once()
    mock_context.close.assert_awaited_once()
    mock_browser.close.assert_not_called()


def _patch_user_browser(matched_page=None):
    """mock chain สำหรับโหมด connect_to_user_browser=True: async_playwright() ->
    .start() -> playwright (ไม่มี chromium.launch()/connect_over_cdp เกี่ยวข้องตรงๆ ใน
    chain นี้ เพราะ connect_user_browser()/resolve_target_page() ถูก patch แยกเป็น
    ฟังก์ชันระดับโมดูลไปเลย — ไม่ต้อง mock รายละเอียด CDP ซ้ำในนี้ เพราะมี
    test_user_browser.py ทดสอบฟังก์ชันพวกนั้นเองอยู่แล้วโดยตรง)"""
    mock_page = AsyncMock()
    mock_page.on = MagicMock()
    mock_page.close = AsyncMock()
    # page.url เป็น plain string property จริงใน Playwright (ไม่ใช่ coroutine) — ต้อง
    # set ตรงๆ ไม่งั้น AsyncMock() auto-mock .url เป็น AsyncMock ลูกไปด้วย ทำให้ domain
    # guard (extract_domain(page.url) ใน orchestrator.py) ไปเรียก .decode() บน mock
    # แบบไม่ await จน pytest เตือน RuntimeWarning ทุกเทสต์ที่มี page-changing action
    mock_page.url = "https://www.saucedemo.com/inventory.html"
    mock_context = MagicMock()
    mock_context.pages = [] if matched_page is None else [matched_page]
    mock_browser = AsyncMock()
    mock_browser.contexts = [mock_context]

    mock_playwright_instance = AsyncMock()
    mock_playwright_instance.stop = AsyncMock()
    mock_p_helper = MagicMock()
    mock_p_helper.start = AsyncMock(return_value=mock_playwright_instance)
    mock_async_playwright = MagicMock(return_value=mock_p_helper)

    return mock_async_playwright, mock_playwright_instance, mock_browser, mock_context, mock_page


def _finish_task_only():
    return AsyncMock(
        return_value=("finish_task", {"success": True, "message": "เสร็จ"}, "", [], llm.TokenUsage())
    )


@pytest.mark.asyncio
async def test_run_task_raises_when_both_browser_and_connect_to_user_browser_given():
    mock_browser, _, _ = _patch_pooled_browser()

    with pytest.raises(ValueError):
        await Orchestrator().run_task(
            "https://example.com", "goal", provider="anthropic",
            browser=mock_browser, connect_to_user_browser=True,
        )


@pytest.mark.asyncio
async def test_run_task_user_browser_mode_connects_via_cdp_not_launch():
    mock_async_playwright, mock_playwright_ctx, mock_browser, mock_context, mock_page = _patch_user_browser()
    mock_connect = AsyncMock(return_value=mock_browser)
    mock_resolve = AsyncMock(return_value=(mock_page, True))

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.connect_user_browser", mock_connect), \
         patch("backend.app.core.orchestrator.resolve_target_page", mock_resolve), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=[]), \
         patch("backend.app.core.orchestrator.llm.next_action", _finish_task_only()):
        result = await Orchestrator().run_task(
            "https://example.com", "goal", provider="anthropic", connect_to_user_browser=True,
        )

    assert result["success"] is True
    mock_connect.assert_awaited_once()
    assert mock_connect.await_args.args[1] == "http://localhost:9222"  # settings default
    mock_playwright_ctx.chromium.launch.assert_not_called()


@pytest.mark.asyncio
async def test_run_task_user_browser_mode_uses_existing_context_not_new_context():
    mock_async_playwright, mock_playwright_ctx, mock_browser, mock_context, mock_page = _patch_user_browser()
    mock_resolve = AsyncMock(return_value=(mock_page, True))

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.connect_user_browser", AsyncMock(return_value=mock_browser)), \
         patch("backend.app.core.orchestrator.resolve_target_page", mock_resolve), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=[]), \
         patch("backend.app.core.orchestrator.llm.next_action", _finish_task_only()):
        await Orchestrator().run_task(
            "https://example.com", "goal", provider="anthropic", connect_to_user_browser=True,
        )

    mock_browser.new_context.assert_not_awaited()
    # resolve_target_page() ต้องได้ context จริงที่มาจาก browser.contexts[0] เป๊ะๆ
    mock_resolve.assert_awaited_once()
    assert mock_resolve.await_args.args[0] is mock_context


@pytest.mark.asyncio
async def test_run_task_user_browser_mode_never_calls_browser_close():
    mock_async_playwright, mock_playwright_ctx, mock_browser, mock_context, mock_page = _patch_user_browser()

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.connect_user_browser", AsyncMock(return_value=mock_browser)), \
         patch("backend.app.core.orchestrator.resolve_target_page", AsyncMock(return_value=(mock_page, True))), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=[]), \
         patch("backend.app.core.orchestrator.llm.next_action", _finish_task_only()):
        await Orchestrator().run_task(
            "https://example.com", "goal", provider="anthropic", connect_to_user_browser=True,
        )

    mock_browser.close.assert_not_awaited()
    mock_playwright_ctx.stop.assert_awaited_once()  # ตัด CDP connection เฉยๆ ไม่ปิด browser จริง


@pytest.mark.asyncio
async def test_run_task_user_browser_mode_never_closes_page_it_opened_itself():
    """เดิม opened_new_tab=True เคยสั่ง page.close() ตอนจบ task — กลายเป็นบั๊กจริง: เทิร์น
    ถัดไปในบทสนทนาเดียวกัน (follow-up command ใน Test Console) หา tab เดิมด้วย domain
    matching ไม่เจอเลยเพราะถูกปิดไปแล้ว ต้องเปิด tab ใหม่ทุกครั้ง (ดูเหมือน "ทำงานต่อจาก
    เดิมไม่ได้") — ตอนนี้ต้องปล่อย tab ไว้เสมอไม่ว่า opened_new_tab จะเป็นอะไร ให้เทิร์น
    ถัดไปกลับมาใช้ต่อได้ (เหมือน keep_browser_open=True ของ owns_browser)"""
    mock_async_playwright, mock_playwright_ctx, mock_browser, mock_context, mock_page = _patch_user_browser()

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.connect_user_browser", AsyncMock(return_value=mock_browser)), \
         patch("backend.app.core.orchestrator.resolve_target_page", AsyncMock(return_value=(mock_page, True))), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=[]), \
         patch("backend.app.core.orchestrator.llm.next_action", _finish_task_only()):
        await Orchestrator().run_task(
            "https://example.com", "goal", provider="anthropic", connect_to_user_browser=True,
        )

    mock_page.close.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_task_user_browser_mode_never_closes_reused_existing_page():
    mock_async_playwright, mock_playwright_ctx, mock_browser, mock_context, mock_page = _patch_user_browser()

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.connect_user_browser", AsyncMock(return_value=mock_browser)), \
         patch("backend.app.core.orchestrator.resolve_target_page", AsyncMock(return_value=(mock_page, False))), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=[]), \
         patch("backend.app.core.orchestrator.llm.next_action", _finish_task_only()):
        await Orchestrator().run_task(
            "https://example.com", "goal", provider="anthropic", connect_to_user_browser=True,
        )

    # opened_new_tab=False -> tab นี้เป็นของ user เอง (agent แค่ขอใช้ต่อ) ห้ามปิดทิ้งอยู่แล้ว
    mock_page.close.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_task_user_browser_mode_passes_explicit_tab_reuse_policy_to_resolve_target_page():
    mock_async_playwright, mock_playwright_ctx, mock_browser, mock_context, mock_page = _patch_user_browser()
    mock_resolve = AsyncMock(return_value=(mock_page, False))

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.connect_user_browser", AsyncMock(return_value=mock_browser)), \
         patch("backend.app.core.orchestrator.resolve_target_page", mock_resolve), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=[]), \
         patch("backend.app.core.orchestrator.llm.next_action", _finish_task_only()):
        await Orchestrator().run_task(
            "https://example.com", "goal", provider="anthropic", connect_to_user_browser=True,
            tab_reuse_policy="always_reuse",
        )

    assert mock_resolve.await_args.args[-1] == "always_reuse"


@pytest.mark.asyncio
async def test_run_task_user_browser_mode_skips_goto_when_reusing_existing_tab():
    """W12: opened_new_tab=False (resolve_target_page() reuse tab จากเทิร์นก่อนหน้า) —
    ต้องไม่ goto(url) ซ้ำตอนเริ่ม task เด็ดขาด ไม่งั้นจะรีโหลดหน้าทิ้ง progress ที่ทำค้าง
    ไว้จากเทิร์นก่อน (บั๊กที่ user รายงาน: สั่ง "เปิดเว็บ" สำเร็จแล้ว เทิร์นถัดมาสั่ง
    "sign in" กลับเห็นหน้าเปิดใหม่เหมือนเริ่มต้นใหม่ทั้งหมดแทนที่จะกดปุ่ม sign in ต่อ)"""
    mock_async_playwright, mock_playwright_ctx, mock_browser, mock_context, mock_page = _patch_user_browser()
    mock_page.url = "https://example.com/dashboard"
    mock_goto = AsyncMock(return_value=_GOTO_OK)

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.connect_user_browser", AsyncMock(return_value=mock_browser)), \
         patch("backend.app.core.orchestrator.resolve_target_page", AsyncMock(return_value=(mock_page, False))), \
         patch("backend.app.core.orchestrator.goto", mock_goto), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=[]), \
         patch("backend.app.core.orchestrator.llm.next_action", _finish_task_only()):
        result = await Orchestrator().run_task(
            "https://example.com", "sign in", provider="anthropic", connect_to_user_browser=True,
        )

    mock_goto.assert_not_awaited()
    assert result["history"][0]["cmd"] == {"type": "continue", "url": "https://example.com/dashboard"}


@pytest.mark.asyncio
async def test_run_task_user_browser_mode_gotos_when_opening_a_fresh_tab():
    """opened_new_tab=True (ไม่มี tab เดิมให้ reuse) — ต้อง goto(url) ตามปกติ เพราะ tab
    ใหม่ว่างเปล่า (about:blank) ยังไม่มีอะไรให้ perceive เลยจนกว่าจะ navigate ก่อน"""
    mock_async_playwright, mock_playwright_ctx, mock_browser, mock_context, mock_page = _patch_user_browser()
    mock_page.url = "about:blank"  # tab ใหม่ที่เพิ่งเปิด ยังไม่มีเนื้อหาอะไรเลย
    mock_goto = AsyncMock(return_value=_GOTO_OK)

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.connect_user_browser", AsyncMock(return_value=mock_browser)), \
         patch("backend.app.core.orchestrator.resolve_target_page", AsyncMock(return_value=(mock_page, True))), \
         patch("backend.app.core.orchestrator.goto", mock_goto), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=[]), \
         patch("backend.app.core.orchestrator.llm.next_action", _finish_task_only()):
        result = await Orchestrator().run_task(
            "https://example.com", "goal", provider="anthropic", connect_to_user_browser=True,
        )

    mock_goto.assert_awaited_once_with(mock_page, "https://example.com")
    assert result["history"][0]["cmd"] == {"type": "goto", "url": "https://example.com"}


@pytest.mark.asyncio
async def test_run_task_user_browser_mode_derives_allowed_domains_from_url_when_not_provided():
    mock_async_playwright, mock_playwright_ctx, mock_browser, mock_context, mock_page = _patch_user_browser()
    click_result = ActionResult(True, "click(1)", "คลิกสำเร็จ")
    next_action_calls = [
        ("browser_action", {"type": "click", "index": 1}, "tool_1", ["m1"], llm.TokenUsage()),
        ("finish_task", {"success": True, "message": "เสร็จ"}, "", ["m2"], llm.TokenUsage()),
    ]

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.connect_user_browser", AsyncMock(return_value=mock_browser)), \
         patch("backend.app.core.orchestrator.resolve_target_page", AsyncMock(return_value=(mock_page, True))), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=[]), \
         patch("backend.app.core.orchestrator.execute", AsyncMock(return_value=click_result)) as mock_execute, \
         patch("backend.app.core.orchestrator.llm.append_tool_result", side_effect=lambda m, tid, r: m + [r]), \
         patch("backend.app.core.orchestrator.llm.next_action", AsyncMock(side_effect=next_action_calls)):
        await Orchestrator().run_task(
            "https://www.saucedemo.com/inventory.html", "goal", provider="anthropic",
            connect_to_user_browser=True,
        )

    mock_execute.assert_awaited_once_with(
        mock_page, {"type": "click", "index": 1},
        ask_user_func=None, label="", manual_guidance="",
        allowed_domains={"www.saucedemo.com"},
    )


@pytest.mark.asyncio
async def test_run_task_user_browser_mode_passes_explicit_allowed_domains_to_execute():
    mock_async_playwright, mock_playwright_ctx, mock_browser, mock_context, mock_page = _patch_user_browser()
    click_result = ActionResult(True, "click(1)", "คลิกสำเร็จ")
    next_action_calls = [
        ("browser_action", {"type": "click", "index": 1}, "tool_1", ["m1"], llm.TokenUsage()),
        ("finish_task", {"success": True, "message": "เสร็จ"}, "", ["m2"], llm.TokenUsage()),
    ]

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.connect_user_browser", AsyncMock(return_value=mock_browser)), \
         patch("backend.app.core.orchestrator.resolve_target_page", AsyncMock(return_value=(mock_page, True))), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=[]), \
         patch("backend.app.core.orchestrator.execute", AsyncMock(return_value=click_result)) as mock_execute, \
         patch("backend.app.core.orchestrator.llm.append_tool_result", side_effect=lambda m, tid, r: m + [r]), \
         patch("backend.app.core.orchestrator.llm.next_action", AsyncMock(side_effect=next_action_calls)):
        await Orchestrator().run_task(
            "https://www.saucedemo.com/inventory.html", "goal", provider="anthropic",
            connect_to_user_browser=True, allowed_domains={"custom.example.com"},
        )

    mock_execute.assert_awaited_once_with(
        mock_page, {"type": "click", "index": 1},
        ask_user_func=None, label="", manual_guidance="",
        allowed_domains={"custom.example.com"},
    )


def _mock_session_page(url: str) -> AsyncMock:
    """page ที่ "resolve มาแล้ว" โดย caller ภายนอก (จำลอง core/session_registry.py::
    SessionRegistry) — ส่งเข้า run_task(page=...) ตรงๆ"""
    page = AsyncMock()
    page.on = MagicMock()
    page.url = url
    return page


@pytest.mark.asyncio
async def test_run_task_raises_when_page_given_with_browser():
    mock_browser, _, _ = _patch_pooled_browser()
    session_page = _mock_session_page("about:blank")

    with pytest.raises(ValueError):
        await Orchestrator().run_task(
            "https://example.com", "goal", provider="anthropic",
            page=session_page, browser=mock_browser,
        )


@pytest.mark.asyncio
async def test_run_task_raises_when_page_given_with_connect_to_user_browser():
    session_page = _mock_session_page("about:blank")

    with pytest.raises(ValueError):
        await Orchestrator().run_task(
            "https://example.com", "goal", provider="anthropic",
            page=session_page, connect_to_user_browser=True,
        )


@pytest.mark.asyncio
async def test_run_task_with_page_skips_all_acquisition():
    """page= (session-managed) ต้องไม่ acquire/launch/connect หา browser เองเลย —
    ไม่เรียก async_playwright()/chromium.launch()/pool อะไรทั้งนั้น"""
    session_page = _mock_session_page("about:blank")
    mock_async_playwright = MagicMock()

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=[]), \
         patch("backend.app.core.orchestrator.llm.next_action", _finish_task_only()):
        result = await Orchestrator().run_task(
            "https://example.com", "goal", provider="anthropic", page=session_page,
        )

    mock_async_playwright.assert_not_called()
    assert result["success"] is True
    session_page.on.assert_called_once()  # dialog handler ยังต้องผูกให้ task นี้เสมอ


@pytest.mark.asyncio
async def test_run_task_with_page_never_closes_or_returns_anything():
    """session registry (ผ่าน routes.py) เป็นคนคุม lifecycle เต็มๆ ข้ามหลาย call — ห้าม
    run_task() ปิด/คืนอะไรที่นี่เด็ดขาดไม่ว่า path ไหน (finish_task ปกติ, loop-detected,
    exception กลาง loop ก็ตาม)"""
    session_page = _mock_session_page("about:blank")

    with patch("backend.app.core.orchestrator.async_playwright"), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=[]), \
         patch("backend.app.core.orchestrator.llm.next_action", _finish_task_only()):
        await Orchestrator().run_task(
            "https://example.com", "goal", provider="anthropic", page=session_page,
        )

    session_page.close.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_task_with_page_skips_goto_when_page_already_has_content():
    """W12: session ที่ reuse page มาจากเทิร์นก่อนหน้า (page.url ไม่ใช่ about:blank แล้ว)
    ต้องไม่ goto(url) ซ้ำ — ปล่อยให้ agent perceive หน้าปัจจุบันตรงๆ ต่อจากจุดเดิม"""
    session_page = _mock_session_page("https://example.com/dashboard")
    mock_goto = AsyncMock(return_value=_GOTO_OK)

    with patch("backend.app.core.orchestrator.async_playwright"), \
         patch("backend.app.core.orchestrator.goto", mock_goto), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=[]), \
         patch("backend.app.core.orchestrator.llm.next_action", _finish_task_only()):
        result = await Orchestrator().run_task(
            "https://example.com", "sign in", provider="anthropic", page=session_page,
        )

    mock_goto.assert_not_awaited()
    assert result["history"][0]["cmd"] == {"type": "continue", "url": "https://example.com/dashboard"}


@pytest.mark.asyncio
async def test_run_task_with_page_gotos_when_domain_differs_from_target():
    """W19: page ที่ reuse มามีเนื้อหาอยู่แล้วจริง (ไม่ blank) แต่เป็นคนละ domain กับ url
    เป้าหมายของ task นี้ (เช่น session เดิมค้างอยู่หน้า other-site.com แต่เทิร์นใหม่สั่ง
    ให้ไป example.com) — ต้อง goto(url) ไปเว็บเป้าหมายจริง ไม่ใช่ข้ามไปเพราะแค่ไม่ blank
    (บั๊กเดิมก่อนแก้: เช็คแค่ "blank หรือไม่" ไม่เทียบ domain เลย)"""
    session_page = _mock_session_page("https://other-site.com/some-page")
    mock_goto = AsyncMock(return_value=_GOTO_OK)

    with patch("backend.app.core.orchestrator.async_playwright"), \
         patch("backend.app.core.orchestrator.goto", mock_goto), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=[]), \
         patch("backend.app.core.orchestrator.llm.next_action", _finish_task_only()):
        result = await Orchestrator().run_task(
            "https://example.com", "goal", provider="anthropic", page=session_page,
        )

    mock_goto.assert_awaited_once_with(session_page, "https://example.com")
    assert result["history"][0]["cmd"] == {"type": "goto", "url": "https://example.com"}


@pytest.mark.asyncio
async def test_run_task_with_page_gotos_when_page_is_blank():
    """session ใหม่ (page ที่เพิ่ง acquire มา ยังไม่เคย navigate เลย, about:blank) ต้อง
    goto(url) ตามปกติเหมือนเดิม — ไม่ใช่ทุก page= จะข้าม goto เสมอไป ขึ้นกับสถานะจริง"""
    session_page = _mock_session_page("about:blank")
    mock_goto = AsyncMock(return_value=_GOTO_OK)

    with patch("backend.app.core.orchestrator.async_playwright"), \
         patch("backend.app.core.orchestrator.goto", mock_goto), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=[]), \
         patch("backend.app.core.orchestrator.llm.next_action", _finish_task_only()):
        result = await Orchestrator().run_task(
            "https://example.com", "goal", provider="anthropic", page=session_page,
        )

    mock_goto.assert_awaited_once_with(session_page, "https://example.com")
    assert result["history"][0]["cmd"] == {"type": "goto", "url": "https://example.com"}


@pytest.mark.asyncio
async def test_run_task_stops_immediately_on_finish_task():
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "[0] button 'Go'"))), \
         patch("backend.app.core.orchestrator.execute") as mock_execute, \
         patch("backend.app.core.llm.build_client", return_value="fake-client"), \
         patch(
             "backend.app.core.orchestrator.llm.next_action",
             AsyncMock(return_value=(
                 "finish_task", {"success": True, "message": "เสร็จแล้ว"}, "", [],
                 llm.TokenUsage(input_tokens=100, output_tokens=20),
             )),
         ):
        result = await Orchestrator().run_task("https://example.com", "some goal", provider="anthropic")

    assert result["success"] is True
    assert result["steps"] == 0
    assert result["message"] == "เสร็จแล้ว"
    # history มี record ของ goto เริ่มต้นเสมอ แม้ finish_task ทันทีโดยไม่มี action อื่น
    assert result["history"] == [
        {
            "step": 0,
            "cmd": {"type": "goto", "url": "https://example.com"},
            "result": str(_GOTO_OK),
            "success": True,
        }
    ]
    # token ของรอบ next_action ที่นำไปสู่ finish_task ต้องถูกนับรวมด้วย แม้ไม่มี browser action เกิดขึ้นเลย
    assert result["tokens"] == {"input": 100, "output": 20, "cache_read": 0, "cache_creation": 0}
    mock_execute.assert_not_called()
    mock_browser.close.assert_awaited_once()
    mock_playwright_ctx.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_task_executes_action_then_finishes():
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    click_result = ActionResult(True, "click(2)", "คลิกสำเร็จ")

    next_action_calls = [
        ("browser_action", {"type": "click", "index": 2}, "tool_1", ["m1"], llm.TokenUsage(input_tokens=50, output_tokens=10)),
        ("finish_task", {"success": True, "message": "เพิ่มลงตะกร้าแล้ว"}, "", ["m2"], llm.TokenUsage(input_tokens=60, output_tokens=15)),
    ]

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "[2] button 'Add to cart'"))), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=[]), \
         patch("backend.app.core.orchestrator.execute", AsyncMock(return_value=click_result)) as mock_execute, \
         patch("backend.app.core.orchestrator.llm.append_tool_result", side_effect=lambda m, tid, r: m + [r]), \
         patch("backend.app.core.orchestrator.llm.next_action", AsyncMock(side_effect=next_action_calls)):
        result = await Orchestrator().run_task("https://example.com", "add item to cart", provider="anthropic")

    assert result["success"] is True
    assert result["steps"] == 1
    assert result["message"] == "เพิ่มลงตะกร้าแล้ว"
    mock_execute.assert_awaited_once_with(
        mock_browser.new_page.return_value, {"type": "click", "index": 2},
        ask_user_func=None, label="", manual_guidance="", allowed_domains=None,
    )
    assert result["history"] == [
        {
            "step": 0,
            "cmd": {"type": "goto", "url": "https://example.com"},
            "result": str(_GOTO_OK),
            "success": True,
        },
        {
            "step": 1,
            "cmd": {"type": "click", "index": 2},
            "label": "",  # get_snapshot() mock คืน elements=[] ในเทสต์นี้ เลยไม่มี label ให้จับคู่
            "result": str(click_result),
            "success": True,
            "tokens": {"input": 50, "output": 10, "cache_read": 0, "cache_creation": 0},
        },
    ]
    # ต้องรวม token ของทั้ง 2 รอบ next_action (browser_action + finish_task) ไม่ใช่แค่รอบสุดท้าย
    assert result["tokens"] == {"input": 110, "output": 25, "cache_read": 0, "cache_creation": 0}


@pytest.mark.asyncio
async def test_run_task_stops_at_max_steps_without_finish_task():
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    scroll_result = ActionResult(True, "scroll(down)", "เลื่อนแล้ว")

    # สลับ direction ทุกครั้งกัน loop-detection guard (W5) เข้าใจผิดว่าเป็น action เดิม
    # ซ้ำติดกัน — เทสต์นี้อยากวัดพฤติกรรม max_steps ตรงๆ ไม่ใช่ loop guard
    next_action_calls = [
        (
            "browser_action", {"type": "scroll", "direction": "down" if i % 2 == 0 else "up"}, f"tool_{i}", [],
            llm.TokenUsage(input_tokens=30, output_tokens=5),
        )
        for i in range(3)
    ]

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.execute", AsyncMock(return_value=scroll_result)) as mock_execute, \
         patch("backend.app.core.orchestrator.llm.append_tool_result", side_effect=lambda m, tid, r: m), \
         patch("backend.app.core.orchestrator.llm.next_action", AsyncMock(side_effect=next_action_calls)):
        result = await Orchestrator().run_task(
            "https://example.com", "goal that never finishes", max_steps=3, provider="anthropic"
        )

    assert result["success"] is False
    assert result["steps"] == 3
    assert mock_execute.await_count == 3
    # token สะสมของ next_action ต้องนับทุกรอบ (3 รอบ) ไม่ใช่แค่รอบเดียว
    assert result["tokens"] == {"input": 90, "output": 15, "cache_read": 0, "cache_creation": 0}


@pytest.mark.asyncio
async def test_run_task_closes_browser_even_if_action_raises():
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(side_effect=RuntimeError("boom"))):
        with pytest.raises(RuntimeError):
            await Orchestrator().run_task("https://example.com", "goal", provider="anthropic")

    mock_browser.close.assert_awaited_once()
    mock_playwright_ctx.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_task_overrides_premature_finish_task_false_then_succeeds():
    """เจอบ่อยกับ Llama บน Groq: เรียก finish_task(success=false) ทั้งที่ยังมี action
    ที่ทำต่อได้ชัดเจน (เช่น เห็นปุ่ม Add to cart แต่ไม่กด) — ต้องไม่ยอมรับทันที เตือนแล้ว
    บังคับให้ลองต่อ ไม่ใช่หยุด task กลางคันทั้งที่ยังทำได้"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    click_result = ActionResult(True, "click(5)", "เพิ่มลงตะกร้าสำเร็จ")

    next_action_calls = [
        ("finish_task", {"success": False, "message": "ทำต่อไม่ได้"}, "tool_f1", ["m1"], llm.TokenUsage()),
        ("browser_action", {"type": "click", "index": 5}, "tool_2", ["m2"], llm.TokenUsage()),
        ("finish_task", {"success": True, "message": "เพิ่มลงตะกร้าแล้ว"}, "", ["m3"], llm.TokenUsage()),
    ]
    append_tool_result_mock = MagicMock(side_effect=lambda m, tid, r: m + [r])

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "[5] button 'Add to cart'"))), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=[]), \
         patch("backend.app.core.orchestrator.execute", AsyncMock(return_value=click_result)) as mock_execute, \
         patch("backend.app.core.orchestrator.llm.append_tool_result", append_tool_result_mock), \
         patch("backend.app.core.orchestrator.llm.next_action", AsyncMock(side_effect=next_action_calls)):
        result = await Orchestrator().run_task("https://example.com", "add item to cart", provider="anthropic")

    assert result["success"] is True
    assert result["steps"] == 1
    mock_execute.assert_awaited_once_with(
        mock_browser.new_page.return_value, {"type": "click", "index": 5},
        ask_user_func=None, label="", manual_guidance="", allowed_domains=None,
    )
    # ต้องเตือนกลับเข้า tool_f1 (finish_task call ที่ถูกปฏิเสธ) ก่อนลองต่อ
    append_tool_result_mock.assert_any_call(["m1"], "tool_f1", _PREMATURE_FALSE_FINISH_NUDGE)


@pytest.mark.asyncio
async def test_run_task_accepts_finish_task_false_after_max_premature_retries():
    """ถ้าโมเดลยืนยัน finish_task(success=false) ซ้ำเกิน quota การเตือนจริงๆ ต้องยอม
    รับว่าทำต่อไม่ได้จริง ไม่ใช่บังคับลองต่อไม่มีที่สิ้นสุด"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.execute") as mock_execute, \
         patch("backend.app.core.orchestrator.llm.append_tool_result", side_effect=lambda m, tid, r: m + [r]), \
         patch(
             "backend.app.core.orchestrator.llm.next_action",
             AsyncMock(return_value=(
                 "finish_task", {"success": False, "message": "ไปต่อไม่ได้จริงๆ"}, "tool_f", [],
                 llm.TokenUsage(),
             )),
         ) as mock_next_action:
        result = await Orchestrator().run_task("https://example.com", "goal", provider="anthropic")

    assert result["success"] is False
    assert result["message"] == "ไปต่อไม่ได้จริงๆ"
    # เตือนไป _MAX_PREMATURE_FALSE_FINISH_RETRIES ครั้ง + ครั้งสุดท้ายที่ยอมรับ = +1
    assert mock_next_action.await_count == _MAX_PREMATURE_FALSE_FINISH_RETRIES + 1
    mock_execute.assert_not_called()


@pytest.mark.asyncio
async def test_run_task_accepts_finish_task_false_immediately_when_no_tool_use_id():
    """finish_task(success=false) จาก fallback ตอนโมเดลไม่ยอมเรียก tool เลย (tool_use_id
    ว่าง) ไม่มี tool call จริงให้ผูก tool_result กลับ — ต้องยอมรับทันที ห้ามพยายามเตือน"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch(
             "backend.app.core.orchestrator.llm.next_action",
             AsyncMock(return_value=(
                 "finish_task", {"success": False, "message": "no tool call"}, "", [],
                 llm.TokenUsage(),
             )),
         ) as mock_next_action:
        result = await Orchestrator().run_task("https://example.com", "goal", provider="anthropic")

    assert result["success"] is False
    assert result["message"] == "no tool call"
    assert mock_next_action.await_count == 1


# --- W5[A] verify (2026-07-16): finish_task(success=true) เรียกทันทีโดยยังไม่ทำ
# action ใดๆ เลย (steps_taken=0) ต้องไม่ถูกยอมรับทันที — symmetric กับ guard ฝั่ง
# false ด้านบน


@pytest.mark.asyncio
async def test_run_task_overrides_premature_finish_task_true_with_zero_steps():
    """finish_task(success=true) เป็น action แรกสุด (steps_taken=0) ต้องไม่ถูกยอมรับ
    ทันที — เตือนแล้วให้ยืนยันอีกครั้งก่อน ไม่ใช่ปล่อยผ่านลอยๆ ไม่มีหลักฐาน"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()

    next_action_calls = [
        ("finish_task", {"success": True, "message": "สำเร็จแล้ว"}, "tool_t1", ["m1"], llm.TokenUsage()),
        ("finish_task", {"success": True, "message": "ยืนยันสำเร็จจริง"}, "tool_t2", ["m2"], llm.TokenUsage()),
    ]
    append_tool_result_mock = MagicMock(side_effect=lambda m, tid, r: m + [r])

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=[]), \
         patch("backend.app.core.orchestrator.llm.append_tool_result", append_tool_result_mock), \
         patch(
             "backend.app.core.orchestrator.llm.next_action", AsyncMock(side_effect=next_action_calls)
         ) as mock_next_action:
        result = await Orchestrator().run_task("https://example.com", "goal", provider="anthropic")

    assert result["success"] is True
    assert result["message"] == "ยืนยันสำเร็จจริง"
    assert mock_next_action.await_count == 2
    # ต้องเตือนกลับเข้า tool_t1 (call แรกที่ถูกปฏิเสธ) ก่อนยอมรับ call ที่สอง
    append_tool_result_mock.assert_any_call(["m1"], "tool_t1", _PREMATURE_TRUE_FINISH_NUDGE)


@pytest.mark.asyncio
async def test_run_task_accepts_finish_task_true_after_max_premature_retries():
    """ถ้าโมเดลยืนยัน finish_task(true) ซ้ำอีกหลังโดนเตือนแล้ว (เกิน
    _MAX_PREMATURE_TRUE_FINISH_RETRIES) ต้องยอมรับจริง ไม่บังคับลองต่อไม่มีที่สิ้นสุด"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=[]), \
         patch("backend.app.core.orchestrator.llm.append_tool_result", side_effect=lambda m, tid, r: m + [r]), \
         patch(
             "backend.app.core.orchestrator.llm.next_action",
             AsyncMock(return_value=(
                 "finish_task", {"success": True, "message": "ยืนยันสำเร็จจริงแน่นอน"}, "tool_t", [],
                 llm.TokenUsage(),
             )),
         ) as mock_next_action:
        result = await Orchestrator().run_task("https://example.com", "goal", provider="anthropic")

    assert result["success"] is True
    assert result["message"] == "ยืนยันสำเร็จจริงแน่นอน"
    # เตือนไป _MAX_PREMATURE_TRUE_FINISH_RETRIES ครั้ง + ครั้งสุดท้ายที่ยอมรับ = +1
    assert mock_next_action.await_count == _MAX_PREMATURE_TRUE_FINISH_RETRIES + 1


@pytest.mark.asyncio
async def test_run_task_accepts_finish_task_true_immediately_when_no_tool_use_id():
    """finish_task(success=true) จาก fallback (tool_use_id ว่าง) ต้องยอมรับทันทีแม้
    steps_taken=0 — ไม่มี tool call จริงให้ผูก tool_result กลับ ห้ามพยายามเตือน"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=[]), \
         patch(
             "backend.app.core.orchestrator.llm.next_action",
             AsyncMock(return_value=(
                 "finish_task", {"success": True, "message": "no tool call"}, "", [],
                 llm.TokenUsage(),
             )),
         ) as mock_next_action:
        result = await Orchestrator().run_task("https://example.com", "goal", provider="anthropic")

    assert result["success"] is True
    assert result["message"] == "no tool call"
    assert mock_next_action.await_count == 1


@pytest.mark.asyncio
async def test_run_task_does_not_nudge_finish_task_true_when_steps_already_taken():
    """finish_task(success=true) หลังทำ action จริงไปแล้วอย่างน้อย 1 step (steps_taken>0)
    ต้องยอมรับทันที ไม่ใช่โดน guard ฝั่ง zero-steps เตือนเลย"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    click_result = ActionResult(True, "click", "สำเร็จ")

    next_action_calls = [
        ("browser_action", {"type": "click", "index": 1}, "t1", [], llm.TokenUsage()),
        ("finish_task", {"success": True, "message": "เสร็จแล้ว"}, "tool_t1", [], llm.TokenUsage()),
    ]

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=[]), \
         patch("backend.app.core.orchestrator.execute", AsyncMock(return_value=click_result)), \
         patch("backend.app.core.orchestrator.llm.append_tool_result", side_effect=lambda m, tid, r: m), \
         patch(
             "backend.app.core.orchestrator.llm.next_action", AsyncMock(side_effect=next_action_calls)
         ) as mock_next_action:
        result = await Orchestrator().run_task("https://example.com", "goal", provider="anthropic")

    assert result["success"] is True
    assert mock_next_action.await_count == 2


@pytest.mark.asyncio
async def test_run_task_result_includes_final_page_state():
    """W5[A] verify: result ต้องมี key "final_page_state" เป็น page_text ของ
    get_snapshot() รอบสุดท้ายก่อนจบ loop — ให้หลักฐานจริงจาก DOM เทียบกับ message
    ที่ LLM อ้างได้ ไม่ต้องเชื่อคำเคลมลอยๆ อย่างเดียว"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch(
             "backend.app.core.orchestrator.get_snapshot",
             AsyncMock(return_value=([], "[0] button 'Order Confirmed'")),
         ), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=[]), \
         patch(
             "backend.app.core.orchestrator.llm.next_action",
             AsyncMock(return_value=("finish_task", {"success": True, "message": "เสร็จ"}, "", [], llm.TokenUsage())),
         ):
        result = await Orchestrator().run_task("https://example.com", "goal", provider="anthropic")

    assert result["final_page_state"] == "[0] button 'Order Confirmed'"


# --- W9[A] vision fallback (Gemini เท่านั้น): action ที่ต้องพึ่ง element visibility
# (click/fill/select/check) ล้มเหลว -> ถ่าย screenshot + เรียก describe_screenshot()
# แล้วป้อนผลลัพธ์เข้า vision_context ของ next_action() รอบถัดไป


@pytest.mark.asyncio
async def test_run_task_triggers_vision_fallback_when_visible_action_fails_on_gemini():
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    mock_page = mock_browser.new_page.return_value
    mock_page.screenshot = AsyncMock(return_value=b"fakepngbytes")
    fail_result = ActionResult(False, "click(5)", "หา element ไม่เจอ")

    next_action_calls = [
        ("browser_action", {"type": "click", "index": 5}, "t1", [], llm.TokenUsage()),
        ("finish_task", {"success": True, "message": "เสร็จ"}, "", [], llm.TokenUsage()),
    ]

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=[]), \
         patch("backend.app.core.orchestrator.execute", AsyncMock(return_value=fail_result)), \
         patch("backend.app.core.llm.build_gemini_client", return_value="fake-client"), \
         patch("backend.app.core.orchestrator.llm.append_tool_result_gemini", side_effect=lambda m, tid, r: m), \
         patch(
             "backend.app.core.orchestrator.llm.describe_screenshot",
             AsyncMock(return_value="เห็น cookie banner บังปุ่มอยู่"),
         ) as mock_describe, \
         patch(
             "backend.app.core.orchestrator.llm.next_action_gemini", AsyncMock(side_effect=next_action_calls)
         ) as mock_next_action:
        await Orchestrator().run_task("https://example.com", "goal", provider="gemini")

    mock_describe.assert_awaited_once()
    describe_args = mock_describe.await_args.args
    assert describe_args[2] == b"fakepngbytes"
    assert describe_args[3] == "click"
    assert describe_args[4] == 5

    # W14: args[-1] เป็น site_manual_context ตัวใหม่ (ว่างเปล่าในเทสต์นี้) —
    # vision_context ขยับไป args[-2]
    second_call_vision_context = mock_next_action.await_args_list[1].args[-2]
    assert second_call_vision_context == "เห็น cookie banner บังปุ่มอยู่"


@pytest.mark.asyncio
async def test_run_task_does_not_trigger_vision_fallback_for_non_gemini_provider():
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    fail_result = ActionResult(False, "click(5)", "หา element ไม่เจอ")

    next_action_calls = [
        ("browser_action", {"type": "click", "index": 5}, "t1", [], llm.TokenUsage()),
        ("finish_task", {"success": True, "message": "เสร็จ"}, "", [], llm.TokenUsage()),
    ]

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=[]), \
         patch("backend.app.core.orchestrator.execute", AsyncMock(return_value=fail_result)), \
         patch("backend.app.core.orchestrator.llm.append_tool_result", side_effect=lambda m, tid, r: m), \
         patch("backend.app.core.orchestrator.llm.describe_screenshot") as mock_describe, \
         patch("backend.app.core.orchestrator.llm.next_action", AsyncMock(side_effect=next_action_calls)):
        await Orchestrator().run_task("https://example.com", "goal", provider="anthropic")

    mock_describe.assert_not_called()


@pytest.mark.asyncio
async def test_run_task_does_not_trigger_vision_fallback_for_non_visibility_action():
    """scroll/goto/go_back/switch_tab/wait ล้มเหลวด้วยเหตุผลอื่น ไม่เกี่ยวกับ
    popup/overlay บัง — ไม่ต้อง trigger vision"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    fail_result = ActionResult(False, "scroll(down)", "error: boom")

    next_action_calls = [
        ("browser_action", {"type": "scroll", "direction": "down"}, "t1", [], llm.TokenUsage()),
        ("finish_task", {"success": True, "message": "เสร็จ"}, "", [], llm.TokenUsage()),
    ]

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=[]), \
         patch("backend.app.core.orchestrator.execute", AsyncMock(return_value=fail_result)), \
         patch("backend.app.core.llm.build_gemini_client", return_value="fake-client"), \
         patch("backend.app.core.orchestrator.llm.append_tool_result_gemini", side_effect=lambda m, tid, r: m), \
         patch("backend.app.core.orchestrator.llm.describe_screenshot") as mock_describe, \
         patch("backend.app.core.orchestrator.llm.next_action_gemini", AsyncMock(side_effect=next_action_calls)):
        await Orchestrator().run_task("https://example.com", "goal", provider="gemini")

    mock_describe.assert_not_called()


@pytest.mark.asyncio
async def test_run_task_does_not_trigger_vision_fallback_when_action_succeeds():
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    click_result = ActionResult(True, "click(5)", "คลิกสำเร็จ")

    next_action_calls = [
        ("browser_action", {"type": "click", "index": 5}, "t1", [], llm.TokenUsage()),
        ("finish_task", {"success": True, "message": "เสร็จ"}, "", [], llm.TokenUsage()),
    ]

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=[]), \
         patch("backend.app.core.orchestrator.execute", AsyncMock(return_value=click_result)), \
         patch("backend.app.core.llm.build_gemini_client", return_value="fake-client"), \
         patch("backend.app.core.orchestrator.llm.append_tool_result_gemini", side_effect=lambda m, tid, r: m), \
         patch("backend.app.core.orchestrator.llm.describe_screenshot") as mock_describe, \
         patch("backend.app.core.orchestrator.llm.next_action_gemini", AsyncMock(side_effect=next_action_calls)):
        await Orchestrator().run_task("https://example.com", "goal", provider="gemini")

    mock_describe.assert_not_called()


@pytest.mark.asyncio
async def test_run_task_confirm_plan_stops_before_any_action_when_user_declines():
    """confirm_plan=True: ต้องโชว์แผนแล้วรอ user ยืนยันก่อน — ถ้า user ปฏิเสธ ห้ามลงมือ
    ทำ action ใดๆ เลย (ห้ามเรียก next_action/execute เลยแม้แต่ครั้งเดียว)"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    ask_user_func = AsyncMock(return_value=False)

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "[0] button 'Go'"))), \
         patch("backend.app.core.orchestrator.execute") as mock_execute, \
         patch("backend.app.core.orchestrator.llm.generate_plan", AsyncMock(return_value="1. ทำ A\n2. ทำ B")) as mock_generate_plan, \
         patch("backend.app.core.orchestrator.retriever.retrieve") as mock_retrieve, \
         patch("backend.app.core.orchestrator.llm.next_action") as mock_next_action:
        result = await Orchestrator().run_task(
            "https://example.com", "some goal", provider="anthropic",
            confirm_plan=True, ask_user_func=ask_user_func,
        )

    assert result["success"] is False
    assert result["steps"] == 0
    assert result["plan"] == "1. ทำ A\n2. ทำ B"
    mock_generate_plan.assert_awaited_once()
    ask_user_func.assert_awaited_once_with({"type": "confirm_plan", "plan": "1. ทำ A\n2. ทำ B"})
    mock_next_action.assert_not_called()
    mock_execute.assert_not_called()
    # W6[B]: retrieve() ต่อเข้าแค่ per-step loop เท่านั้น ไม่ใช่ generate_plan — ถ้า loop
    # ไม่เคยเริ่มเลย (user ปฏิเสธแผน) retrieve() ก็ต้องไม่ถูกเรียกเลยเช่นกัน
    mock_retrieve.assert_not_called()


@pytest.mark.asyncio
async def test_run_task_confirm_plan_proceeds_when_user_approves():
    """confirm_plan=True + user ยืนยัน -> loop ต้องทำงานตามปกติต่อ ไม่ต่างจากไม่เปิด
    confirm_plan เลย นอกจากมี plan text แนบมาด้วยตอนจบ"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    ask_user_func = AsyncMock(return_value=True)

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "[0] button 'Go'"))), \
         patch("backend.app.core.orchestrator.llm.generate_plan", AsyncMock(return_value="1. ทำ A")), \
         patch(
             "backend.app.core.orchestrator.llm.next_action",
             AsyncMock(return_value=("finish_task", {"success": True, "message": "เสร็จแล้ว"}, "", [], llm.TokenUsage())),
         ) as mock_next_action:
        result = await Orchestrator().run_task(
            "https://example.com", "some goal", provider="anthropic",
            confirm_plan=True, ask_user_func=ask_user_func,
        )

    assert result["success"] is True
    assert result["plan"] == "1. ทำ A"
    mock_next_action.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_task_uses_user_edited_plan_text_when_confirmed():
    """W10[F]: user แก้ไขข้อความแผนก่อนกด Confirm (จำลองพฤติกรรมของ
    TaskManager.resolve_approval(edited_plan=...) ที่ mutate cmd["plan"] ใน-place ก่อน
    ask_user_func คืนค่ากลับมา) — ผลลัพธ์ต้องสะท้อนแผนที่แก้แล้ว (ไม่ใช่แผนเดิมที่ AI ร่าง)
    ทั้งใน result["plan"] และใน goal ที่ next_action() เห็นทุก step ต่อจากนี้ (ไม่งั้นแก้
    plan ไปก็ไม่มีผลอะไรกับพฤติกรรมจริงเลย)"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()

    async def _ask_user_edits_the_plan(cmd: dict) -> bool:
        # จำลอง TaskManager.resolve_approval(request_id, True, edited_plan="...")
        cmd["plan"] = "1. แผนที่ user แก้ไขเอง"
        return True

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "[0] button 'Go'"))), \
         patch("backend.app.core.orchestrator.llm.generate_plan", AsyncMock(return_value="1. แผนเดิมที่ AI ร่าง")), \
         patch(
             "backend.app.core.orchestrator.llm.next_action",
             AsyncMock(return_value=("finish_task", {"success": True, "message": "เสร็จแล้ว"}, "", [], llm.TokenUsage())),
         ) as mock_next_action:
        result = await Orchestrator().run_task(
            "https://example.com", "some goal", provider="anthropic",
            confirm_plan=True, ask_user_func=_ask_user_edits_the_plan,
        )

    assert result["plan"] == "1. แผนที่ user แก้ไขเอง"
    effective_goal_seen_by_next_action = mock_next_action.await_args.args[2]
    assert "แผนที่ user แก้ไขเอง" in effective_goal_seen_by_next_action
    assert "แผนเดิมที่ AI ร่าง" not in effective_goal_seen_by_next_action


@pytest.mark.asyncio
async def test_run_task_without_confirm_plan_skips_plan_generation_entirely():
    """confirm_plan=False (default) -> ห้ามเรียก llm.generate_plan เลย กันเสีย token
    เปล่าๆ กับ use case ที่ไม่ต้องการ gate นี้"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.llm.generate_plan") as mock_generate_plan, \
         patch(
             "backend.app.core.orchestrator.llm.next_action",
             AsyncMock(return_value=("finish_task", {"success": True, "message": "ok"}, "", [], llm.TokenUsage())),
         ):
        result = await Orchestrator().run_task("https://example.com", "goal", provider="anthropic")

    assert result["plan"] is None
    mock_generate_plan.assert_not_called()


# W13: Orchestrator.generate_plan() — เฟสวางแผนแยกต่างหาก ไม่ผูกกับ run_task()/browser
# lifecycle เลย (ดู routes.py::POST /api/generate_plan)


@pytest.mark.asyncio
async def test_generate_plan_without_page_touches_no_browser():
    mock_generate_plan = AsyncMock(return_value="1. Do X\n2. Do Y")
    with patch("backend.app.core.orchestrator.llm.generate_plan", mock_generate_plan), \
         patch("backend.app.core.orchestrator.get_snapshot") as mock_get_snapshot, \
         patch("backend.app.core.orchestrator.async_playwright") as mock_async_playwright:
        result = await Orchestrator().generate_plan("https://example.com", "goal", provider="anthropic")

    mock_get_snapshot.assert_not_called()
    mock_async_playwright.assert_not_called()
    assert result == "1. Do X\n2. Do Y"
    call_args = mock_generate_plan.await_args.args
    assert call_args[2] == "goal"
    assert call_args[3] == ""  # page_text ว่างเปล่า ไม่มี page ให้ perceive


@pytest.mark.asyncio
async def test_generate_plan_with_page_perceives_current_state():
    mock_page = AsyncMock()
    mock_generate_plan = AsyncMock(return_value="1. Sign in")
    with patch("backend.app.core.orchestrator.llm.generate_plan", mock_generate_plan), \
         patch(
             "backend.app.core.orchestrator.get_snapshot",
             AsyncMock(return_value=([], "[1] button 'Sign in'")),
         ) as mock_get_snapshot:
        result = await Orchestrator().generate_plan(
            "https://example.com", "sign in", provider="anthropic", page=mock_page,
        )

    mock_get_snapshot.assert_awaited_once_with(mock_page)
    call_args = mock_generate_plan.await_args.args
    assert call_args[3] == "[1] button 'Sign in'"
    assert result == "1. Sign in"


# W13: run_task(approved_plan=...) — แผนที่อนุมัติไปแล้วจากภายนอก (generate_plan() +
# user review ผ่าน routes.py) ก่อนเรียก run_task() ด้วยซ้ำ


@pytest.mark.asyncio
async def test_run_task_raises_when_approved_plan_given_with_confirm_plan():
    with pytest.raises(ValueError):
        await Orchestrator().run_task(
            "https://example.com", "goal", provider="anthropic",
            approved_plan="1. Do X", confirm_plan=True,
        )


@pytest.mark.asyncio
async def test_run_task_approved_plan_skips_internal_generation_and_confirmation():
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.llm.generate_plan") as mock_generate_plan, \
         patch(
             "backend.app.core.orchestrator.llm.next_action",
             AsyncMock(return_value=("finish_task", {"success": True, "message": "เสร็จแล้ว"}, "", [], llm.TokenUsage())),
         ) as mock_next_action:
        result = await Orchestrator().run_task(
            "https://example.com", "sign in", provider="anthropic", approved_plan="1. Click sign in",
        )

    mock_generate_plan.assert_not_called()  # ไม่ต้องร่างแผนเองอีก อนุมัติมาแล้ว
    assert result["plan"] == "1. Click sign in"
    effective_goal_seen_by_next_action = mock_next_action.await_args.args[2]
    assert "1. Click sign in" in effective_goal_seen_by_next_action


@pytest.mark.asyncio
async def test_run_task_stops_on_repeated_identical_action():
    """loop-detection guard: บาง provider (เจอกับ Llama บน Groq) ยังวนเรียก
    browser_action เดิมเป๊ะๆ ซ้ำๆ แม้ execute() จะสำเร็จทุกครั้ง — ต้องหยุด task เอง
    ไม่ปล่อยให้วนจนหมด max_steps เสีย token ไปเรื่อยๆ โดยไม่มีความคืบหน้า"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    click_result = ActionResult(True, "click(5)", "คลิกสำเร็จ")
    same_action = ("browser_action", {"type": "click", "index": 5}, "tool_x", [], llm.TokenUsage())

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.execute", AsyncMock(return_value=click_result)) as mock_execute, \
         patch("backend.app.core.orchestrator.llm.append_tool_result", side_effect=lambda m, tid, r: m), \
         patch("backend.app.core.orchestrator.llm.next_action", AsyncMock(return_value=same_action)) as mock_next_action:
        result = await Orchestrator().run_task(
            "https://example.com", "goal", max_steps=10, provider="anthropic"
        )

    assert result["success"] is False
    assert "ซ้ำ" in result["message"]
    # ยิงจริงแค่ (N-1) ครั้ง เพราะการเรียกซ้ำครั้งที่ N ถูกสกัดไว้ก่อน execute()
    assert result["steps"] == _MAX_CONSECUTIVE_IDENTICAL_ACTIONS - 1
    assert mock_execute.await_count == _MAX_CONSECUTIVE_IDENTICAL_ACTIONS - 1
    assert mock_next_action.await_count == _MAX_CONSECUTIVE_IDENTICAL_ACTIONS


@pytest.mark.asyncio
async def test_run_task_loop_guard_does_not_trigger_for_varied_actions():
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    click_result = ActionResult(True, "click", "สำเร็จ")

    next_action_calls = [
        ("browser_action", {"type": "click", "index": 1}, "t1", [], llm.TokenUsage()),
        ("browser_action", {"type": "click", "index": 2}, "t2", [], llm.TokenUsage()),
        ("browser_action", {"type": "click", "index": 1}, "t3", [], llm.TokenUsage()),
        ("finish_task", {"success": True, "message": "เสร็จ"}, "", [], llm.TokenUsage()),
    ]

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.execute", AsyncMock(return_value=click_result)) as mock_execute, \
         patch("backend.app.core.orchestrator.llm.append_tool_result", side_effect=lambda m, tid, r: m), \
         patch("backend.app.core.orchestrator.llm.next_action", AsyncMock(side_effect=next_action_calls)):
        result = await Orchestrator().run_task(
            "https://example.com", "goal", max_steps=10, provider="anthropic"
        )

    assert result["success"] is True
    assert mock_execute.await_count == 3


@pytest.mark.asyncio
async def test_run_task_loop_guard_resets_count_after_different_action():
    """A, A, B, A, A -> ไม่มีช่วงไหนซ้ำติดกันครบ _MAX_CONSECUTIVE_IDENTICAL_ACTIONS
    ครั้ง (สูงสุดคือ 2 ติดกัน) ต้องไม่ trigger — พิสูจน์ว่า count reset จริงตอนเจอ
    action ต่างจากเดิม ไม่ใช่แค่นับสะสมรวมทั้ง task"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    click_result = ActionResult(True, "click", "สำเร็จ")
    action_a = {"type": "click", "index": 1}
    action_b = {"type": "click", "index": 2}

    next_action_calls = [
        ("browser_action", action_a, "t1", [], llm.TokenUsage()),
        ("browser_action", action_a, "t2", [], llm.TokenUsage()),
        ("browser_action", action_b, "t3", [], llm.TokenUsage()),
        ("browser_action", action_a, "t4", [], llm.TokenUsage()),
        ("browser_action", action_a, "t5", [], llm.TokenUsage()),
        ("finish_task", {"success": True, "message": "เสร็จ"}, "", [], llm.TokenUsage()),
    ]

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.execute", AsyncMock(return_value=click_result)) as mock_execute, \
         patch("backend.app.core.orchestrator.llm.append_tool_result", side_effect=lambda m, tid, r: m), \
         patch("backend.app.core.orchestrator.llm.next_action", AsyncMock(side_effect=next_action_calls)):
        result = await Orchestrator().run_task(
            "https://example.com", "goal", max_steps=10, provider="anthropic"
        )

    assert result["success"] is True
    assert mock_execute.await_count == 5


@pytest.mark.asyncio
async def test_run_task_stops_on_alternating_two_action_pattern():
    """loop-detection (คาบ 2, 2026-07-13): agent วนสลับ 2 action ที่ไม่เหมือนกันไปมา
    (เช่น go_back <-> click) — guard เดิม (_MAX_CONSECUTIVE_IDENTICAL_ACTIONS) จับได้
    แค่ action เดิมเป๊ะๆ ซ้ำติดกัน (คาบ 1) ไม่ตรงเงื่อนไขนี้เลยไม่เคย trigger ต้องมี
    guard ใหม่จับคาบ 2 (ABAB) แยกต่างหาก"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    click_result = ActionResult(True, "click", "สำเร็จ")
    action_a = {"type": "go_back"}
    action_b = {"type": "click", "index": 3}

    next_action_calls = [
        ("browser_action", action_a, "t1", [], llm.TokenUsage()),
        ("browser_action", action_b, "t2", [], llm.TokenUsage()),
        ("browser_action", action_a, "t3", [], llm.TokenUsage()),
        ("browser_action", action_b, "t4", [], llm.TokenUsage()),
    ]

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.execute", AsyncMock(return_value=click_result)) as mock_execute, \
         patch("backend.app.core.orchestrator.llm.append_tool_result", side_effect=lambda m, tid, r: m), \
         patch(
             "backend.app.core.orchestrator.llm.next_action", AsyncMock(side_effect=next_action_calls)
         ) as mock_next_action:
        result = await Orchestrator().run_task(
            "https://example.com", "goal", max_steps=10, provider="anthropic"
        )

    assert result["success"] is False
    assert "คาบ 2" in result["message"]
    # การสลับครั้งที่ 4 (B ตัวที่ 2) ถูกสกัดไว้ก่อน execute() เหมือน guard คาบ 1
    assert mock_execute.await_count == 3
    assert mock_next_action.await_count == 4


@pytest.mark.asyncio
async def test_run_task_stops_on_repeating_three_action_cycle():
    """(2026-07-15) generalize: guard เดิมจับได้แค่คาบ 2 (ABAB) ตรงๆ — ตอนนั้นมีเทสต์
    (test_run_task_loop_guard_does_not_trigger_for_three_action_cycle เดิม) ยืนยันไว้
    ตรงๆ ว่าคาบ 3 (ABC-ABC) "ยังไม่ scope ไว้" ไม่ trigger — user ถามว่า pattern ที่
    ไม่ใช่แค่คาบ 1/2 (เช่น click ปุ่มเดิม/scroll/fill สลับกันเป็นคาบยาวกว่านั้นที่ไม่ทำ
    ให้หน้าเว็บเปลี่ยนสเตทจริง) จะจับได้ไหม — generalize guard ให้ครอบคลุมถึงคาบ 4
    (_MAX_CYCLE_PERIOD) แล้ว พลิกกลับเทสต์นี้ให้ยืนยันว่าคาบ 3 ต้อง trigger จริง"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    click_result = ActionResult(True, "click", "สำเร็จ")
    action_a = {"type": "click", "index": 1}
    action_b = {"type": "click", "index": 2}
    action_c = {"type": "click", "index": 3}

    next_action_calls = [
        ("browser_action", action_a, "t1", [], llm.TokenUsage()),
        ("browser_action", action_b, "t2", [], llm.TokenUsage()),
        ("browser_action", action_c, "t3", [], llm.TokenUsage()),
        ("browser_action", action_a, "t4", [], llm.TokenUsage()),
        ("browser_action", action_b, "t5", [], llm.TokenUsage()),
        ("browser_action", action_c, "t6", [], llm.TokenUsage()),
    ]

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.execute", AsyncMock(return_value=click_result)) as mock_execute, \
         patch("backend.app.core.orchestrator.llm.append_tool_result", side_effect=lambda m, tid, r: m), \
         patch(
             "backend.app.core.orchestrator.llm.next_action", AsyncMock(side_effect=next_action_calls)
         ) as mock_next_action:
        result = await Orchestrator().run_task(
            "https://example.com", "goal", max_steps=10, provider="anthropic"
        )

    assert result["success"] is False
    assert "คาบ 3" in result["message"]
    # ครบ 2 รอบเต็ม (ABC-ABC = 6 action) ถึงจะ trigger — ครั้งที่ 6 ถูกสกัดไว้ก่อน execute()
    assert mock_execute.await_count == 5
    assert mock_next_action.await_count == 6


@pytest.mark.asyncio
async def test_run_task_stops_on_repeating_four_action_cycle():
    """คาบ 4 (ABCD-ABCD, ตรงกับ _MAX_CYCLE_PERIOD พอดี) ต้อง trigger เหมือนกัน —
    ยืนยันว่า generalize ไม่ได้ทำแค่คาบ 3 แต่ครอบคลุมทุกคาบใน range ที่ตั้งใจไว้จริง"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    click_result = ActionResult(True, "click", "สำเร็จ")
    action_a = {"type": "click", "index": 1}
    action_b = {"type": "click", "index": 2}
    action_c = {"type": "click", "index": 3}
    action_d = {"type": "click", "index": 4}

    next_action_calls = [
        ("browser_action", action_a, "t1", [], llm.TokenUsage()),
        ("browser_action", action_b, "t2", [], llm.TokenUsage()),
        ("browser_action", action_c, "t3", [], llm.TokenUsage()),
        ("browser_action", action_d, "t4", [], llm.TokenUsage()),
        ("browser_action", action_a, "t5", [], llm.TokenUsage()),
        ("browser_action", action_b, "t6", [], llm.TokenUsage()),
        ("browser_action", action_c, "t7", [], llm.TokenUsage()),
        ("browser_action", action_d, "t8", [], llm.TokenUsage()),
    ]

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.execute", AsyncMock(return_value=click_result)) as mock_execute, \
         patch("backend.app.core.orchestrator.llm.append_tool_result", side_effect=lambda m, tid, r: m), \
         patch(
             "backend.app.core.orchestrator.llm.next_action", AsyncMock(side_effect=next_action_calls)
         ) as mock_next_action:
        result = await Orchestrator().run_task(
            "https://example.com", "goal", max_steps=10, provider="anthropic"
        )

    assert result["success"] is False
    assert "คาบ 4" in result["message"]
    assert mock_execute.await_count == 7
    assert mock_next_action.await_count == 8


@pytest.mark.asyncio
async def test_run_task_loop_guard_does_not_trigger_for_five_action_cycle():
    """เกินขอบเขตที่ตั้งใจไว้ (_MAX_CYCLE_PERIOD=4) โดยเจตนา — คาบ 5 (ABCDE-ABCDE)
    ไม่ควร trigger เพราะยังไม่ scope ไว้ (เอกสารขอบเขตของ guard ไว้ตรงๆ เหมือนที่เทสต์
    คาบ 3 เดิมเคยทำก่อนจะขยายมาถึงคาบ 4)"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    click_result = ActionResult(True, "click", "สำเร็จ")
    actions = [{"type": "click", "index": i} for i in range(1, 6)]  # A..E

    next_action_calls = [
        ("browser_action", a, f"t{i}", [], llm.TokenUsage())
        for i, a in enumerate(actions + actions, start=1)
    ] + [("finish_task", {"success": True, "message": "เสร็จ"}, "", [], llm.TokenUsage())]

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.execute", AsyncMock(return_value=click_result)) as mock_execute, \
         patch("backend.app.core.orchestrator.llm.append_tool_result", side_effect=lambda m, tid, r: m), \
         patch("backend.app.core.orchestrator.llm.next_action", AsyncMock(side_effect=next_action_calls)):
        result = await Orchestrator().run_task(
            "https://example.com", "goal", max_steps=15, provider="anthropic"
        )

    assert result["success"] is True
    assert mock_execute.await_count == 10


# --- code-level guard: ห้ามข้าม login form ที่ยังกรอกไม่ครบ (2026-07-13) ---
# SYSTEM_PROMPT ขอไว้แล้วว่าห้าม wait คั่นกลางตอน login แต่โมเดลเล็ก (Gemini flash-lite)
# ไม่ทำตามเสมอไป — เจอจริงว่าสั่ง wait เฉยๆ แล้วรอบถัดไปข้ามไปกด element อื่นทั้งที่ยังไม่
# ได้กรอก password เลย เทสต์กลุ่มนี้ patch _login_form_needs_password() ตรงๆ (ไม่ใช้ page
# จริง) เพื่อควบคุม scenario ได้แน่นอน


@pytest.mark.asyncio
async def test_run_task_rejects_non_fill_action_when_password_field_still_empty():
    """ถ้า password field ยังว่างอยู่ (login form ยังกรอกไม่ครบ) ต้องปฏิเสธ action ที่
    ไม่ใช่ fill (เช่น wait) แล้วเตือนให้กรอกก่อน — ไม่เรียก execute() เลยสำหรับ action
    ที่ถูกปฏิเสธ"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    fill_result = ActionResult(True, "fill(1)", "กรอกสำเร็จ")

    next_action_calls = [
        ("browser_action", {"type": "wait"}, "t1", [], llm.TokenUsage()),
        ("browser_action", {"type": "fill", "index": 1, "text": "secret_sauce"}, "t2", [], llm.TokenUsage()),
        ("finish_task", {"success": True, "message": "เสร็จ"}, "", [], llm.TokenUsage()),
    ]

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=[]), \
         patch("backend.app.core.orchestrator.execute", AsyncMock(return_value=fill_result)) as mock_execute, \
         patch("backend.app.core.orchestrator.llm.append_tool_result", side_effect=lambda m, tid, r: m), \
         patch("backend.app.core.orchestrator._login_form_needs_password", AsyncMock(return_value=True)), \
         patch(
             "backend.app.core.orchestrator.llm.next_action", AsyncMock(side_effect=next_action_calls)
         ) as mock_next_action:
        result = await Orchestrator().run_task("https://example.com", "login goal", provider="anthropic")

    assert result["success"] is True
    # step 1 (wait) ถูกปฏิเสธ ไม่เรียก execute() เลย, step 2 (fill) ผ่านปกติ
    assert mock_execute.await_count == 1
    assert mock_next_action.await_count == 3


@pytest.mark.asyncio
async def test_run_task_login_form_guard_gives_up_after_max_retries():
    """ถ้าโมเดลยืนกรานทำ action ที่ไม่ใช่ fill ต่อไปเรื่อยๆ แม้เตือนแล้ว (เช่น สั่ง wait
    ซ้ำ) guard ต้องไม่ค้างตลอดไป — ปล่อยผ่านหลังเตือนครบ _MAX_PREMATURE_LOGIN_SKIP_RETRIES
    ครั้ง"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    wait_result = ActionResult(True, "wait_stable", "หน้านิ่งแล้ว")

    next_action_calls = [
        ("browser_action", {"type": "wait"}, "t1", [], llm.TokenUsage()),
        ("browser_action", {"type": "wait"}, "t2", [], llm.TokenUsage()),
        ("browser_action", {"type": "wait"}, "t3", [], llm.TokenUsage()),
        ("finish_task", {"success": False, "message": "หมดหวัง"}, "", [], llm.TokenUsage()),
    ]

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=[]), \
         patch("backend.app.core.orchestrator.execute", AsyncMock(return_value=wait_result)) as mock_execute, \
         patch("backend.app.core.orchestrator.llm.append_tool_result", side_effect=lambda m, tid, r: m), \
         patch("backend.app.core.orchestrator._login_form_needs_password", AsyncMock(return_value=True)), \
         patch(
             "backend.app.core.orchestrator.llm.next_action", AsyncMock(side_effect=next_action_calls)
         ) as mock_next_action:
        result = await Orchestrator().run_task("https://example.com", "login goal", provider="anthropic")

    # 2 ครั้งแรกถูกปฏิเสธ (nudge), ครั้งที่ 3 ถูกปล่อยผ่านให้ execute() จริง (กัน stall
    # ตลอดไป) แล้ว finish_task(false) หลังจากนั้นค่อยจบ
    assert result["success"] is False
    assert mock_execute.await_count == 1
    assert mock_next_action.await_count == 4


@pytest.mark.asyncio
async def test_run_task_login_form_guard_exempts_goto():
    """goto ต้องไม่โดน guard นี้บล็อกเด็ดขาด แม้ password field ยังว่างอยู่ — ระบบ
    อาจจำเป็นต้อง goto ไปหน้าอื่นก่อน (แก้เส้นทาง/multi-hop กว่าจะถึงฟอร์ม login จริง)
    ห้ามติดอยู่ที่หน้าเดิมแบบออกไปไหนไม่ได้เลย"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    goto_result = ActionResult(True, "goto", "ไปที่ url")

    next_action_calls = [
        ("browser_action", {"type": "goto", "url": "https://example.com/login"}, "t1", [], llm.TokenUsage()),
        ("finish_task", {"success": True, "message": "เสร็จ"}, "", [], llm.TokenUsage()),
    ]

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=[]), \
         patch("backend.app.core.orchestrator.execute", AsyncMock(return_value=goto_result)) as mock_execute, \
         patch("backend.app.core.orchestrator.llm.append_tool_result", side_effect=lambda m, tid, r: m), \
         patch("backend.app.core.orchestrator._login_form_needs_password", AsyncMock(return_value=True)), \
         patch(
             "backend.app.core.orchestrator.llm.next_action", AsyncMock(side_effect=next_action_calls)
         ) as mock_next_action:
        result = await Orchestrator().run_task("https://example.com", "goal", provider="anthropic")

    # goto ต้องผ่าน execute() ทันที ไม่ถูกปฏิเสธ/นับเป็นการ nudge เลยแม้แต่ครั้งเดียว
    assert result["success"] is True
    assert mock_execute.await_count == 1
    assert mock_next_action.await_count == 2


@pytest.mark.asyncio
async def test_run_task_calls_retrieve_with_goal_page_text_and_k_then_passes_result_into_next_action():
    """W6[B]: ทุก step ต้องดึงคู่มือด้วย retrieve(query=goal, page_state=page_text ปัจจุบัน,
    k=_RAG_CHUNKS_PER_STEP) แล้วเอาผลลัพธ์ (join เป็น bullet list) ส่งต่อเข้า next_action()
    เป็น manual_context"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "[0] button 'Go'"))), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=["chunk1", "chunk2", "chunk3"]) as mock_retrieve, \
         patch(
             "backend.app.core.orchestrator.llm.next_action",
             AsyncMock(return_value=("finish_task", {"success": True, "message": "เสร็จแล้ว"}, "", [], llm.TokenUsage())),
         ) as mock_next_action:
        await Orchestrator().run_task("https://example.com", "some goal", provider="anthropic")

    mock_retrieve.assert_called_once_with(query="some goal", page_state="[0] button 'Go'", k=_RAG_CHUNKS_PER_STEP)
    # W14: args[-1] เป็น site_manual_context ตัวใหม่ — manual_context ขยับไป args[-5]
    manual_context = mock_next_action.await_args.args[-5]
    assert manual_context == "- chunk1\n- chunk2\n- chunk3"


@pytest.mark.asyncio
async def test_run_task_calls_retrieve_every_step_with_that_steps_page_text():
    """retrieve() ต้องถูกเรียกใหม่ทุก step ตาม page_text ของ step นั้นๆ (ไม่ใช่แค่ครั้งเดียว
    ตอนเริ่ม task) — พิสูจน์ด้วยการให้ get_snapshot คืน page_text ต่างกันทุก step"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    click_result = ActionResult(True, "click", "สำเร็จ")

    page_texts = [([], "[0] step1 page"), ([], "[0] step2 page"), ([], "[0] step3 page")]
    next_action_calls = [
        ("browser_action", {"type": "click", "index": 1}, "t1", [], llm.TokenUsage()),
        ("browser_action", {"type": "click", "index": 2}, "t2", [], llm.TokenUsage()),
        ("finish_task", {"success": True, "message": "เสร็จ"}, "", [], llm.TokenUsage()),
    ]

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(side_effect=page_texts)), \
         patch("backend.app.core.orchestrator.execute", AsyncMock(return_value=click_result)), \
         patch("backend.app.core.orchestrator.llm.append_tool_result", side_effect=lambda m, tid, r: m), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=["c"]) as mock_retrieve, \
         patch("backend.app.core.orchestrator.llm.next_action", AsyncMock(side_effect=next_action_calls)):
        await Orchestrator().run_task("https://example.com", "goal", max_steps=10, provider="anthropic")

    # 2 calls ต่อ step ที่มี action จริง (manual_context สำหรับ planner query=goal +
    # permission-specific query แคบเฉพาะ action — ดู _build_permission_query()) x 2
    # step ที่เป็น browser_action + 1 call เดียวของ step สุดท้าย (finish_task ไม่ผ่าน
    # execute() เลยไม่มี permission-specific call)
    assert mock_retrieve.call_count == 5
    # manual_context calls เท่านั้นที่ส่ง page_state มาด้วย (permission-specific ไม่ส่ง)
    manual_calls = [c for c in mock_retrieve.call_args_list if "page_state" in c.kwargs]
    called_page_states = [c.kwargs["page_state"] for c in manual_calls]
    assert called_page_states == ["[0] step1 page", "[0] step2 page", "[0] step3 page"]


@pytest.mark.asyncio
async def test_run_task_manual_context_is_empty_string_when_retrieve_returns_no_chunks():
    """ยังไม่มีคู่มือ ingest ไว้ (หรือหาไม่เจออะไรตรงกัน) -> retrieve() คืน [] -> manual_context
    ต้องเป็น "" เฉยๆ ไม่ใช่ None หรือข้อความ placeholder"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=[]), \
         patch(
             "backend.app.core.orchestrator.llm.next_action",
             AsyncMock(return_value=("finish_task", {"success": True, "message": "เสร็จแล้ว"}, "", [], llm.TokenUsage())),
         ) as mock_next_action:
        await Orchestrator().run_task("https://example.com", "goal", provider="anthropic")

    # W14: args[-1] เป็น site_manual_context ตัวใหม่ — manual_context ขยับไป args[-5]
    manual_context = mock_next_action.await_args.args[-5]
    assert manual_context == ""


@pytest.mark.asyncio
async def test_run_task_memory_context_reflects_previous_step_failure():
    """W7[A]: action ที่ fail ใน step ก่อนหน้า ต้องโผล่ใน memory_context ที่ส่งเข้า
    next_action() ของ step ถัดไป (ผ่าน ShortTermMemory.failed_actions_summary()) —
    step แรก (ยังไม่มี failure ใดๆ นอกจาก goto ที่สำเร็จ) ต้องยังว่างเปล่าอยู่"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    fail_result = ActionResult(False, "click", "หา element ไม่เจอ")

    next_action_calls = [
        ("browser_action", {"type": "click", "index": 9}, "t1", [], llm.TokenUsage()),
        ("finish_task", {"success": True, "message": "เสร็จ"}, "", [], llm.TokenUsage()),
    ]

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=[]), \
         patch("backend.app.core.orchestrator.execute", AsyncMock(return_value=fail_result)), \
         patch("backend.app.core.orchestrator.llm.append_tool_result", side_effect=lambda m, tid, r: m), \
         patch(
             "backend.app.core.orchestrator.llm.next_action", AsyncMock(side_effect=next_action_calls)
         ) as mock_next_action:
        await Orchestrator().run_task("https://example.com", "goal", provider="anthropic")

    # W14: args[-1] เป็น site_manual_context ตัวใหม่ — memory_context ขยับไป args[-4]
    first_call_memory_context = mock_next_action.await_args_list[0].args[-4]
    second_call_memory_context = mock_next_action.await_args_list[1].args[-4]
    assert first_call_memory_context == ""
    assert "[FAIL]" in second_call_memory_context
    assert "หา element ไม่เจอ" in second_call_memory_context


@pytest.mark.asyncio
async def test_run_task_memory_context_is_empty_string_when_no_failures_yet():
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=[]), \
         patch(
             "backend.app.core.orchestrator.llm.next_action",
             AsyncMock(return_value=("finish_task", {"success": True, "message": "เสร็จแล้ว"}, "", [], llm.TokenUsage())),
         ) as mock_next_action:
        await Orchestrator().run_task("https://example.com", "goal", provider="anthropic")

    # W14: args[-1] เป็น site_manual_context ตัวใหม่ — memory_context ขยับไป args[-4]
    memory_context = mock_next_action.await_args.args[-4]
    assert memory_context == ""


@pytest.mark.asyncio
async def test_run_task_calls_long_term_memory_recall_with_goal_page_text_and_k_then_passes_into_next_action():
    """W7[A] (long-term): ทุก step ต้อง recall(query=goal, page_state=page_text ปัจจุบัน,
    k=_LONG_TERM_MEMORY_CHUNKS_PER_STEP) แล้วเอาผลลัพธ์ (join เป็น bullet list) ส่งต่อเข้า
    next_action() เป็น long_term_context (arg สุดท้าย)"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "[0] button 'Apply Code'"))), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=[]), \
         patch(
             "backend.app.core.orchestrator.long_term_memory.recall",
             return_value=["task1: เคยกด Apply Code แล้วโดนบล็อก"],
         ) as mock_recall, \
         patch(
             "backend.app.core.orchestrator.llm.next_action",
             AsyncMock(return_value=("finish_task", {"success": True, "message": "เสร็จแล้ว"}, "", [], llm.TokenUsage())),
         ) as mock_next_action:
        await Orchestrator().run_task("https://example.com", "some goal", provider="anthropic")

    mock_recall.assert_called_once_with(
        query="some goal", page_state="[0] button 'Apply Code'", k=_LONG_TERM_MEMORY_CHUNKS_PER_STEP
    )
    # W14: args[-1] เป็น site_manual_context ตัวใหม่ — long_term_context ขยับไป args[-3]
    long_term_context = mock_next_action.await_args.args[-3]
    assert long_term_context == "- task1: เคยกด Apply Code แล้วโดนบล็อก"


@pytest.mark.asyncio
async def test_run_task_long_term_context_is_empty_string_when_recall_returns_no_chunks():
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=[]), \
         patch("backend.app.core.orchestrator.long_term_memory.recall", return_value=[]), \
         patch(
             "backend.app.core.orchestrator.llm.next_action",
             AsyncMock(return_value=("finish_task", {"success": True, "message": "เสร็จแล้ว"}, "", [], llm.TokenUsage())),
         ) as mock_next_action:
        await Orchestrator().run_task("https://example.com", "goal", provider="anthropic")

    # W14: args[-1] เป็น site_manual_context ตัวใหม่ — long_term_context ขยับไป args[-3]
    long_term_context = mock_next_action.await_args.args[-3]
    assert long_term_context == ""


@pytest.mark.asyncio
async def test_run_task_records_task_outcome_into_long_term_memory_at_the_end():
    """W7[A] (long-term): record_task() ต้องถูกเรียกครั้งเดียวตอนจบ loop จริง ด้วย
    url/goal/success/message ที่ตรงกับผลลัพธ์สุดท้าย + failed_actions จาก
    ShortTermMemory.failed_actions_summary() ของ task นั้น"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    fail_result = ActionResult(False, "click", "หา element ไม่เจอ")

    # tool_use_id="" ตัวที่สอง เพื่อให้ finish_task(false) ถูกยอมรับทันที (ไม่ตกไปเจอ
    # premature-false-finish guard ที่ต้องมี tool_use_id จริงถึงจะเตือน — ดู
    # test_run_task_accepts_finish_task_false_immediately_when_no_tool_use_id ด้านบน)
    next_action_calls = [
        ("browser_action", {"type": "click", "index": 9}, "t1", [], llm.TokenUsage()),
        ("finish_task", {"success": False, "message": "ทำต่อไม่ได้"}, "", [], llm.TokenUsage()),
    ]

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=[]), \
         patch("backend.app.core.orchestrator.execute", AsyncMock(return_value=fail_result)), \
         patch("backend.app.core.orchestrator.llm.append_tool_result", side_effect=lambda m, tid, r: m), \
         patch("backend.app.core.orchestrator.llm.next_action", AsyncMock(side_effect=next_action_calls)), \
         patch("backend.app.core.orchestrator.long_term_memory.record_task") as mock_record_task:
        result = await Orchestrator().run_task("https://example.com", "goal", provider="anthropic")

    assert result["success"] is False
    mock_record_task.assert_called_once_with(
        url="https://example.com",
        goal="goal",
        success=False,
        message="ทำต่อไม่ได้",
        failed_actions=mock_record_task.call_args.kwargs["failed_actions"],
    )
    assert "หา element ไม่เจอ" in mock_record_task.call_args.kwargs["failed_actions"]


@pytest.mark.asyncio
async def test_run_task_does_not_record_long_term_memory_when_plan_declined():
    """confirm_plan=True + user ปฏิเสธ -> return ก่อนถึง loop จริงเลย ไม่มี action ใดๆ
    เกิดขึ้น -> ไม่ควรบันทึกอะไรเข้า long-term memory (ไม่มี pattern ให้จำ)"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    ask_user_func = AsyncMock(return_value=False)

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.llm.generate_plan", AsyncMock(return_value="1. ทำ X\n2. ทำ Y")), \
         patch("backend.app.core.orchestrator.long_term_memory.record_task") as mock_record_task:
        result = await Orchestrator().run_task(
            "https://example.com", "goal", provider="anthropic",
            confirm_plan=True, ask_user_func=ask_user_func,
        )

    assert result["steps"] == 0
    mock_record_task.assert_not_called()


# --- Gemini-aware nudge messages (bug found via W7[A] Test Case A live run) ---
# ก่อนแก้: nudge message ที่ฉีดเข้า messages ตรงๆ (นอกเหนือจาก append_tool_result())
# ของ guard 2 ตัว (premature-false-finish, premature-login-skip) hardcode เป็น
# {"role":"user","content":...} แบบ Anthropic/Groq เสมอ — ใช้กับ provider="gemini"
# แล้ว Gemini SDK จริงจะ throw KeyError เพราะ contents ต้องการ key "parts" ไม่ใช่
# "content" — ไม่เคยมี unit test เดิมจับได้เพราะ next_action_gemini() ถูก mock ทั้งก้อน
# เสมอ ไม่เคยมี test ตรวจ shape ของ nudge message ที่ฉีดกลับเข้า messages เอง


def test_build_nudge_message_uses_gemini_shape_for_gemini_provider():
    result = _build_nudge_message("gemini", "เตือนนะ")

    assert result == {"role": "user", "parts": [{"text": "เตือนนะ"}]}


def test_build_nudge_message_uses_content_shape_for_other_providers():
    assert _build_nudge_message("anthropic", "เตือนนะ") == {"role": "user", "content": "เตือนนะ"}
    assert _build_nudge_message("groq", "เตือนนะ") == {"role": "user", "content": "เตือนนะ"}


@pytest.mark.asyncio
async def test_run_task_premature_false_finish_nudge_uses_gemini_message_shape_for_gemini_provider():
    """บั๊กที่เจอจริงจากการทดสอบ W7[A] Test Case A ผ่าน Gemini (ดูหมายเหตุด้านบน) —
    nudge message ของ guard นี้ต้องเป็น {"role":"user","parts":[{"text":...}]} เมื่อ
    provider="gemini" ไม่ใช่ {"role":"user","content":...} แบบเดิม มิฉะนั้น Gemini SDK
    จริงจะ throw KeyError ตอนส่ง messages เข้า generate_content_async() รอบถัดไป"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()

    next_action_calls = [
        ("finish_task", {"success": False, "message": "ทำต่อไม่ได้"}, "call_1", [], llm.TokenUsage()),
        ("finish_task", {"success": True, "message": "จบแล้ว"}, "", [], llm.TokenUsage()),
    ]

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=[]), \
         patch("backend.app.core.llm.build_gemini_client", return_value="fake-client"), \
         patch(
             "backend.app.core.orchestrator.llm.next_action_gemini", AsyncMock(side_effect=next_action_calls)
         ) as mock_next_action:
        result = await Orchestrator().run_task("https://example.com", "goal", provider="gemini")

    assert result["success"] is True
    second_call_messages = mock_next_action.await_args_list[1].args[4]
    user_messages = [m for m in second_call_messages if isinstance(m, dict) and m.get("role") == "user"]
    assert user_messages  # ต้องมี nudge จริงถูกฉีดเข้าไป ไม่ใช่ list ว่างเปล่า
    assert all("content" not in m for m in user_messages)  # ต้องไม่มี key แบบ Anthropic/Groq หลงเหลือ
    assert any(
        "ถูกปฏิเสธ" in part.get("text", "") for m in user_messages for part in m.get("parts", [])
    )


# --- Gemini context compaction (W7[A], Test Case C) ---


def test_build_gemini_history_digest_summarizes_steps_up_to_cutoff():
    memory = ShortTermMemory()
    memory.record({"step": 0, "cmd": {"type": "goto", "url": "https://x"}, "result": "[OK] ไปที่ url", "success": True})
    memory.record({"step": 1, "cmd": {"type": "fill", "index": 0}, "result": "[OK] fill(0) -> login สำเร็จ", "success": True})
    memory.record({"step": 2, "cmd": {"type": "click", "index": 1}, "result": "[OK] click(1) -> เพิ่มสินค้า", "success": True})
    memory.record({"step": 3, "cmd": {"type": "click", "index": 2}, "result": "[FAIL] click(2) -> พัง", "success": False})

    digest = _build_gemini_history_digest(memory, upto_step=2)

    assert "step 1" in digest
    assert "login สำเร็จ" in digest
    assert "step 2" in digest
    assert "เพิ่มสินค้า" in digest
    assert "step 3" not in digest  # เกิน upto_step ไม่ควรโผล่
    assert "goto" not in digest  # step 0 ไม่นับ (ไม่ใช่ step ของ action จริง)


def test_build_gemini_history_digest_returns_empty_string_when_no_matching_steps():
    memory = ShortTermMemory()
    memory.record({"step": 0, "cmd": {"type": "goto", "url": "https://x"}, "result": "[OK]", "success": True})

    assert _build_gemini_history_digest(memory, upto_step=5) == ""


def test_compact_gemini_messages_drops_old_turns_and_prepends_digest_to_kept_turn():
    messages = [
        {"role": "user", "parts": [{"text": "old step 1"}]},
        {"role": "model", "parts": [{"function_call": {"name": "browser_action", "args": {}}}]},
        {"role": "user", "parts": [{"text": "old step 2"}]},
        {"role": "model", "parts": [{"function_call": {"name": "browser_action", "args": {}}}]},
        {"role": "user", "parts": [{"text": "recent step 3 goal here"}]},
        {"role": "model", "parts": [{"function_call": {"name": "browser_action", "args": {}}}]},
    ]

    result = _compact_gemini_messages(messages, cut_at=4, digest_text="- step 1: ok\n- step 2: ok")

    assert len(result) == 2  # เท่ากับ len(messages) - cut_at เสมอ (แทนที่ text ไม่ตัด turn)
    assert "step 1: ok" in result[0]["parts"][0]["text"]
    assert "recent step 3 goal here" in result[0]["parts"][0]["text"]
    assert result[1] == messages[5]  # turn ที่เหลือไม่ถูกแตะเลย


def test_compact_gemini_messages_is_noop_when_cut_at_zero_or_digest_empty():
    messages = [{"role": "user", "parts": [{"text": "x"}]}]

    assert _compact_gemini_messages(messages, cut_at=0, digest_text="something") == messages
    assert _compact_gemini_messages(messages, cut_at=1, digest_text="") == messages


def test_compact_gemini_messages_falls_back_to_original_on_unexpected_shape():
    """ถ้า messages[cut_at] ไม่ใช่รูปแบบ {"role":"user","parts":[{"text":...}]} ที่คาดไว้
    (ผิดคาดจริงๆ) ต้องคืน messages เดิมไม่แก้อะไร ไม่ throw"""
    messages = [{"role": "user", "parts": [{"function_response": {"name": "x", "response": {}}}]}]

    assert _compact_gemini_messages(messages, cut_at=0, digest_text="x") == messages


@pytest.mark.asyncio
async def test_run_task_compacts_gemini_history_once_step_count_exceeds_threshold():
    """W7[A] (Test Case C): เกิน _GEMINI_COMPACT_AFTER_STEPS step แล้ว messages ที่ส่ง
    เข้า next_action_gemini() ต้องไม่โตต่อเนื่องไม่มีเพดานตามจำนวน step อีกต่อไป (ถูก
    ตัด step เก่ากว่า _GEMINI_KEEP_RECENT_STEPS ตัวล่าสุดออก) — digest ของ step แรกๆ
    ต้องยังโผล่อยู่ในบทสนทนาที่เหลือ (ไม่ได้หายไปเฉยๆ พิสูจน์ assertion #2 ของ Test Case C
    ที่ต้องการให้ agent ยังจำ step แรกๆ ได้)"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    click_result = ActionResult(True, "click", "สำเร็จ")

    total_action_steps = _GEMINI_COMPACT_AFTER_STEPS + 2  # ต้องเกิน threshold แน่ๆ
    step_actions = [
        ("browser_action", {"type": "click", "index": i}, f"call_{i}") for i in range(total_action_steps)
    ]
    step_actions.append(("finish_task", {"success": True, "message": "เสร็จ"}, ""))

    captured_messages_per_call: list[list] = []

    async def _next_action_side_effect(
        client, model, goal, page_text, messages, manual_context="", memory_context="",
        long_term_context="", vision_context="", site_manual_context="",
    ):
        captured_messages_per_call.append(messages)
        i = len(captured_messages_per_call) - 1
        tool_name, tool_input, tool_use_id = step_actions[i]
        # จำลอง messages โตขึ้นจริงเหมือน implementation จริง (append 1 user ctx turn
        # + 1 model turn ต่อ call) — ไม่งั้น mock คืนค่าคงที่จะไม่พิสูจน์อะไรเกี่ยวกับ
        # compaction เลย
        new_messages = messages + [
            {"role": "user", "parts": [{"text": f"Goal: {goal}\n\nหน้าเว็บปัจจุบัน:\n{page_text} #{i}"}]},
            {"role": "model", "parts": [{"function_call": {"name": tool_name, "args": tool_input}}]},
        ]
        return tool_name, tool_input, tool_use_id, new_messages, llm.TokenUsage()

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=[]), \
         patch("backend.app.core.orchestrator.execute", AsyncMock(return_value=click_result)), \
         patch("backend.app.core.llm.build_gemini_client", return_value="fake-client"), \
         patch("backend.app.core.orchestrator.llm.next_action_gemini", AsyncMock(side_effect=_next_action_side_effect)):
        result = await Orchestrator().run_task(
            "https://example.com", "goal", provider="gemini", max_steps=total_action_steps + 2
        )

    assert result["success"] is True
    assert result["steps"] == total_action_steps

    # ถ้าไม่มี compaction เลย messages ของ call สุดท้าย (finish_task) จะยาว 3 ตัว/step
    # x total_action_steps = (6+2)*3 = 24 ตัว — ต้องน้อยกว่านี้มากถ้า compaction ทำงานจริง
    last_call_messages = captured_messages_per_call[-1]
    uncompacted_would_be = total_action_steps * 3
    assert len(last_call_messages) < uncompacted_would_be

    # digest ของ step แรกๆ (ที่ถูกบีบอัดไปแล้ว) ต้องยังโผล่อยู่ในบทสนทนาที่เหลือ
    all_text = " ".join(
        part.get("text", "")
        for msg in last_call_messages
        for part in msg.get("parts", [])
        if isinstance(part, dict)
    )
    assert "step 1" in all_text
    assert "สรุป step ก่อนหน้า" in all_text


@pytest.mark.asyncio
async def test_run_task_does_not_compact_for_non_gemini_provider():
    """scope จำกัดแค่ Gemini ตามที่ user เลือก — provider อื่น (Anthropic/Groq) ต้องไม่
    ถูกตัด messages เลยไม่ว่าจะกี่ step ก็ตาม (ยังไม่ได้ implement ให้ provider อื่น)"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    click_result = ActionResult(True, "click", "สำเร็จ")

    total_action_steps = _GEMINI_COMPACT_AFTER_STEPS + 2
    step_actions = [
        ("browser_action", {"type": "click", "index": i}, f"t{i}") for i in range(total_action_steps)
    ]
    step_actions.append(("finish_task", {"success": True, "message": "เสร็จ"}, ""))

    captured_messages_per_call: list[list] = []

    async def _next_action_side_effect(
        client, model, goal, page_text, messages, manual_context="", memory_context="",
        long_term_context="", vision_context="", site_manual_context="",
    ):
        captured_messages_per_call.append(messages)
        i = len(captured_messages_per_call) - 1
        tool_name, tool_input, tool_use_id = step_actions[i]
        new_messages = messages + [{"role": "user", "content": f"turn {i}"}]
        return tool_name, tool_input, tool_use_id, new_messages, llm.TokenUsage()

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=[]), \
         patch("backend.app.core.orchestrator.execute", AsyncMock(return_value=click_result)), \
         patch("backend.app.core.orchestrator.llm.append_tool_result", side_effect=lambda m, tid, r: m + [r]), \
         patch("backend.app.core.orchestrator.llm.next_action", AsyncMock(side_effect=_next_action_side_effect)):
        result = await Orchestrator().run_task(
            "https://example.com", "goal", provider="anthropic", max_steps=total_action_steps + 2
        )

    assert result["success"] is True
    # ทุก step เพิ่ม 2 message (1 จาก next_action mock, 1 จาก append_tool_result จริง) ไม่มี
    # การบีบอัดใดๆ เกิดขึ้นเลย (ฟีเจอร์นี้ scope แค่ gemini) — โตเป็นเส้นตรงตามจำนวน step เป๊ะ
    last_call_messages = captured_messages_per_call[-1]
    assert len(last_call_messages) == total_action_steps * 2


# --- Permission layer connected to the real per-step loop ---
# เทสต์กลุ่มนี้ไม่ mock backend.app.core.orchestrator.execute เหมือนเทสต์อื่นๆ ด้านบน —
# ปล่อยให้ actions.py::execute() ตัวจริงทำงาน (รวม classify_action() + _confirm_action())
# กับ mock_page (AsyncMock เฉยๆ ไม่ raise) เพื่อพิสูจน์ว่า permission layer ต่อเข้ากับ
# loop จริงของ orchestrator ได้จริง ไม่ใช่แค่ต่อกับ execute() แบบแยกส่วนใน test_perm.py


@pytest.mark.asyncio
async def test_run_task_needs_confirmation_action_calls_ask_user_func_and_executes_when_approved():
    """purchase/delete/pay/submit (NEEDS_CONFIRMATION) ต้องขอยืนยันจาก ask_user_func
    ก่อนเสมอ ผ่าน execute() ตัวจริง — อนุมัติแล้วต้อง dispatch จริงต่อ (สำเร็จ)"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    ask_user_func = AsyncMock(return_value=True)

    next_action_calls = [
        ("browser_action", {"type": "purchase", "index": 3}, "t1", [], llm.TokenUsage()),
        ("finish_task", {"success": True, "message": "เสร็จ"}, "", [], llm.TokenUsage()),
    ]

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=[]), \
         patch("backend.app.core.orchestrator.llm.append_tool_result", side_effect=lambda m, tid, r: m), \
         patch("backend.app.core.orchestrator.llm.next_action", AsyncMock(side_effect=next_action_calls)):
        result = await Orchestrator().run_task(
            "https://example.com", "goal", provider="anthropic", ask_user_func=ask_user_func
        )

    ask_user_func.assert_awaited_once_with({"type": "purchase", "index": 3})
    assert result["success"] is True
    assert result["steps"] == 1
    # history[0] คือ goto ตอนเริ่ม task, history[1] คือ step ของ purchase ที่เพิ่งอนุมัติ
    assert "[OK]" in result["history"][1]["result"]
    assert "purchase(3)" in result["history"][1]["result"]


@pytest.mark.asyncio
async def test_run_task_needs_confirmation_action_rejected_when_ask_user_func_declines():
    """ถ้า ask_user_func ปฏิเสธ ต้องไม่ dispatch action จริง (ไม่กด element) และ
    ผลลัพธ์ที่บันทึกต้องสะท้อนว่าโดนปฏิเสธ ไม่ใช่ error อื่น"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    ask_user_func = AsyncMock(return_value=False)

    next_action_calls = [
        ("browser_action", {"type": "delete", "index": 5}, "t1", [], llm.TokenUsage()),
        # tool_use_id="" กัน premature-false-finish guard (W4) เตือนแล้วลองใหม่ —
        # ไม่ใช่สิ่งที่เทสต์นี้อยากวัด (ดู test_run_task_accepts_finish_task_false_
        # immediately_when_no_tool_use_id ด้านบนสำหรับพฤติกรรมของ guard นั้นโดยเฉพาะ)
        ("finish_task", {"success": False, "message": "หยุดหลังโดนปฏิเสธ"}, "", [], llm.TokenUsage()),
    ]

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=[]), \
         patch("backend.app.core.orchestrator.llm.append_tool_result", side_effect=lambda m, tid, r: m), \
         patch("backend.app.core.orchestrator.llm.next_action", AsyncMock(side_effect=next_action_calls)):
        result = await Orchestrator().run_task(
            "https://example.com", "goal", provider="anthropic", ask_user_func=ask_user_func
        )

    ask_user_func.assert_awaited_once_with({"type": "delete", "index": 5})
    assert "[FAIL]" in result["history"][1]["result"]
    assert "ปฏิเสธ" in result["history"][1]["result"]


@pytest.mark.asyncio
async def test_run_task_stops_immediately_when_action_rejected_by_human():
    """(2026-07-17) เดิม (ก่อนแก้): action ที่ถูกมนุษย์ปฏิเสธ (ask_user_func คืน False)
    จะถูกป้อนกลับเข้า messages แล้วปล่อยให้ LLM วน loop ลองทางอื่นต่อไปเรื่อยๆ — ผิด
    เจตนาของ human-in-the-loop (การกด Deny ควรแปลว่า "หยุด" ไม่ใช่ "ลองทางอื่น") — ตอนนี้
    ต้องจบ task ทันทีที่โดนปฏิเสธ ไม่เรียก next_action() รอบถัดไปอีกเลย (ต่างจาก
    test_run_task_rejected_action_flows_into_memory_context_next_step เดิมที่ยืนยันไว้
    ตรงข้ามกัน — พฤติกรรมเปลี่ยนไปตามที่ user ขอ)"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    ask_user_func = AsyncMock(return_value=False)

    next_action_calls = [
        ("browser_action", {"type": "delete", "index": 5}, "t1", [], llm.TokenUsage()),
        # ไม่ควรถูกเรียกเลย — ถ้า mock ถูก consume ตัวนี้แปลว่า loop ยังวนต่อทั้งที่ถูก
        # ปฏิเสธไปแล้ว (บั๊กเดิมที่กำลังกันไว้)
        ("browser_action", {"type": "click", "index": 2}, "t2", [], llm.TokenUsage()),
    ]

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=[]), \
         patch("backend.app.core.orchestrator.llm.append_tool_result", side_effect=lambda m, tid, r: m), \
         patch(
             "backend.app.core.orchestrator.llm.next_action", AsyncMock(side_effect=next_action_calls)
         ) as mock_next_action:
        result = await Orchestrator().run_task(
            "https://example.com", "goal", provider="anthropic", ask_user_func=ask_user_func
        )

    assert mock_next_action.await_count == 1  # ไม่มีการลองทางอื่นต่อหลังโดนปฏิเสธ
    assert result["success"] is False
    assert "ปฏิเสธ" in result["message"]
    assert "[FAIL]" in result["history"][1]["result"]
    assert "ผู้ใช้ปฏิเสธการทำ Action นี้" in result["history"][1]["result"]


@pytest.mark.asyncio
async def test_run_task_blocked_domain_goto_never_calls_ask_user_func():
    """BLOCKED (goto ไปโดเมนใน blocklist) ต้องถูกปฏิเสธทันทีโดยไม่ถาม human เลย —
    ต่างจาก NEEDS_CONFIRMATION ที่ต้องถาม"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    ask_user_func = AsyncMock(return_value=True)

    next_action_calls = [
        ("browser_action", {"type": "goto", "url": "https://malicious.com/login"}, "t1", [], llm.TokenUsage()),
        ("finish_task", {"success": False, "message": "โดนบล็อก"}, "", [], llm.TokenUsage()),
    ]

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=[]), \
         patch("backend.app.core.orchestrator.llm.append_tool_result", side_effect=lambda m, tid, r: m), \
         patch("backend.app.core.orchestrator.llm.next_action", AsyncMock(side_effect=next_action_calls)):
        result = await Orchestrator().run_task(
            "https://example.com", "goal", provider="anthropic", ask_user_func=ask_user_func
        )

    ask_user_func.assert_not_called()
    assert "Blocklist" in result["history"][1]["result"]


@pytest.mark.asyncio
async def test_run_task_plain_click_on_risky_labeled_element_still_asks_for_confirmation():
    """defense-in-depth: LLM ส่ง type="click" ธรรมดา (ไม่ใช่ delete/submit/purchase/pay)
    กับ element ที่ label ตรงคำเสี่ยง (เช่น "Remove" บน saucedemo) — orchestrator ต้อง
    หา label จาก elements ของ snapshot รอบนั้นแล้วส่งให้ execute() เช็คด้วย ไม่ใช่พึ่ง
    ให้ LLM เลือก type ให้ถูกเพียงอย่างเดียว"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    ask_user_func = AsyncMock(return_value=True)

    elements = [{"index": 7, "tag": "button", "type": "", "label": "Remove"}]
    next_action_calls = [
        ("browser_action", {"type": "click", "index": 7}, "t1", [], llm.TokenUsage()),
        ("finish_task", {"success": True, "message": "เสร็จ"}, "", [], llm.TokenUsage()),
    ]

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=(elements, "page"))), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=[]), \
         patch("backend.app.core.orchestrator.llm.append_tool_result", side_effect=lambda m, tid, r: m), \
         patch("backend.app.core.orchestrator.llm.next_action", AsyncMock(side_effect=next_action_calls)):
        result = await Orchestrator().run_task(
            "https://example.com", "goal", provider="anthropic", ask_user_func=ask_user_func
        )

    # element_label แนบเข้าไปให้ ask_user_func เห็นชื่อ element จริงด้วย (ไม่ใช่แค่ index)
    # — cmd ต้นฉบับที่ dispatch จริงยังไม่ถูกแตะ (ดู actions.py::_confirm_action)
    ask_user_func.assert_awaited_once_with({"type": "click", "index": 7, "element_label": "Remove"})
    assert result["success"] is True
    assert "[OK]" in result["history"][1]["result"]
    # W10[D]: history ต้องเก็บ label ของ element เป้าหมายไว้ด้วย (ไม่ใช่แค่ index) ให้ UI
    # โชว์ชื่อจริง (เช่น "Remove") แทน index เปล่าๆ — ดึงจาก elements ของ snapshot รอบ
    # เดียวกับที่ action_label ด้านบนใช้เช็ค permission อยู่แล้ว ไม่ต้องคำนวณซ้ำ
    assert result["history"][1]["label"] == "Remove"


@pytest.mark.asyncio
async def test_run_task_on_event_step_includes_element_label():
    """W10[D]: on_event() (สตรีมสดๆ ไปหน้าเว็บ W10[B]) ต้องได้ label ของ element
    เป้าหมายเหมือนกับที่ history เก็บไว้ ไม่ใช่แค่ index เปล่าๆ — ให้ Log panel ที่ฟัง
    stream สดๆ โชว์ชื่อปุ่ม/ช่องกรอกได้แบบ real-time เหมือนกับตอน task จบแล้ว"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    on_event = AsyncMock()

    elements = [{"index": 3, "tag": "button", "type": "", "label": "Checkout"}]
    next_action_calls = [
        ("browser_action", {"type": "click", "index": 3}, "t1", [], llm.TokenUsage()),
        ("finish_task", {"success": True, "message": "เสร็จ"}, "", [], llm.TokenUsage()),
    ]

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=(elements, "page"))), \
         patch(
             "backend.app.core.orchestrator.execute",
             AsyncMock(return_value=ActionResult(True, "click(3)", "สำเร็จ")),
         ), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=[]), \
         patch("backend.app.core.orchestrator.llm.append_tool_result", side_effect=lambda m, tid, r: m), \
         patch("backend.app.core.orchestrator.llm.next_action", AsyncMock(side_effect=next_action_calls)):
        result = await Orchestrator().run_task(
            "https://example.com", "goal", provider="anthropic", on_event=on_event
        )

    assert result["success"] is True
    step_events = [c.args[0] for c in on_event.await_args_list if c.args[0].get("kind") == "step"]
    action_step_event = next(e for e in step_events if e.get("cmd", {}).get("type") == "click")
    assert action_step_event["label"] == "Checkout"


# --- W7[B]: RAG-based permission — manual_context (ดึงมาแล้วสำหรับ planner ตั้งแต่
# W6[B]) ต้องถูกส่งต่อให้ execute() เช็คด้วยว่าคู่มือระบุไว้ไหมว่า action นี้ต้องขอ
# อนุมัติ — ไม่ยิง retriever.retrieve() ซ้ำอีกครั้งเพื่อเช็ค permission โดยเฉพาะ


@pytest.mark.asyncio
async def test_run_task_passes_permission_specific_manual_guidance_into_execute():
    """manual_guidance ที่ execute() ได้รับต้องมาจาก query แคบเฉพาะ action นี้
    (type+label ผ่าน _build_permission_query()) ไม่ใช่ manual_context (query=goal)
    ตัวเดียวกับที่ป้อน planner — เดิม (ก่อนแก้) reuse manual_context ตรงๆ แต่รันจริง
    บน saucedemo.com พบว่ากว้างเกินไป (ดูคอมเมนต์ที่ _PERMISSION_RAG_CHUNKS_PER_STEP
    ใน orchestrator.py) พิสูจน์ด้วยการให้ retrieve() คืนค่าต่างกันตามลำดับการเรียก
    (side_effect) แล้วเช็คว่า execute() ได้รับผลลัพธ์ของ call ที่ 2 (permission-specific)
    ไม่ใช่ call ที่ 1 (manual_context)"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    click_result = ActionResult(True, "click", "สำเร็จ")

    elements = [{"index": 1, "tag": "button", "type": "", "label": "Some Button"}]
    next_action_calls = [
        ("browser_action", {"type": "click", "index": 1}, "t1", [], llm.TokenUsage()),
        ("finish_task", {"success": True, "message": "เสร็จ"}, "", [], llm.TokenUsage()),
    ]

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=(elements, "page"))), \
         patch(
             "backend.app.core.orchestrator.retriever.retrieve",
             # call ที่ 3 คือ manual_context ของ step ที่ 2 (finish_task) — ไม่มี
             # permission-specific call คู่กัน เพราะ finish_task break ก่อนถึง execute()
             side_effect=[["manual chunk for planner"], ["permission-specific chunk"], []],
         ) as mock_retrieve, \
         patch("backend.app.core.orchestrator.llm.append_tool_result", side_effect=lambda m, tid, r: m), \
         patch("backend.app.core.orchestrator.execute", AsyncMock(return_value=click_result)) as mock_execute, \
         patch("backend.app.core.orchestrator.llm.next_action", AsyncMock(side_effect=next_action_calls)):
        await Orchestrator().run_task("https://example.com", "goal", provider="anthropic")

    assert mock_execute.await_args.kwargs["manual_guidance"] == "- permission-specific chunk"
    # call แรก (manual_context ของ planner) ต้อง query ด้วย goal ทั้งก้อน, call ที่สอง
    # (permission-specific) ต้อง query แคบด้วย type+label ของ action นี้เท่านั้น
    assert mock_retrieve.call_args_list[0].kwargs["query"] == "goal"
    assert mock_retrieve.call_args_list[1].kwargs["query"] == "click Some Button"


@pytest.mark.asyncio
async def test_run_task_asks_for_confirmation_when_manual_requires_approval_for_safe_looking_action():
    """type="click" ธรรมดา + label ปกติ (ไม่เสี่ยง) แต่คู่มือของ step นั้นบอกว่าต้องขอ
    อนุมัติก่อน — ต้อง trigger NEEDS_CONFIRMATION ผ่าน loop จริง (execute() ตัวจริง
    ไม่ mock) แม้ LLM จะไม่ได้เลือก type=submit/delete/purchase/pay เองเลยก็ตาม"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    ask_user_func = AsyncMock(return_value=True)

    elements = [{"index": 9, "tag": "button", "type": "", "label": "Checkout"}]
    next_action_calls = [
        ("browser_action", {"type": "click", "index": 9}, "t1", [], llm.TokenUsage()),
        ("finish_task", {"success": True, "message": "เสร็จ"}, "", [], llm.TokenUsage()),
    ]

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=(elements, "page"))), \
         patch(
             "backend.app.core.orchestrator.retriever.retrieve",
             return_value=["การกด Checkout ทุกครั้ง requires approval จากหัวหน้างานก่อนเสมอ"],
         ), \
         patch("backend.app.core.orchestrator.llm.append_tool_result", side_effect=lambda m, tid, r: m), \
         patch("backend.app.core.orchestrator.llm.next_action", AsyncMock(side_effect=next_action_calls)):
        result = await Orchestrator().run_task(
            "https://example.com", "goal", provider="anthropic", ask_user_func=ask_user_func
        )

    ask_user_func.assert_awaited_once_with({"type": "click", "index": 9, "element_label": "Checkout"})
    assert result["success"] is True
    assert "[OK]" in result["history"][1]["result"]

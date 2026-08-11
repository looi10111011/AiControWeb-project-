import asyncio
import itertools
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.app.core import llm
from backend.app.core import orchestrator as orchestrator_module
from backend.app.core.actions import ActionResult
from backend.app.core.memory import ShortTermMemory
from backend.app.core.orchestrator import (
    Orchestrator,
    _COMPACT_AFTER_STEPS,
    _KEEP_RECENT_STEPS,
    _LONG_TERM_MEMORY_CHUNKS_PER_STEP,
    _MAX_CONSECUTIVE_IDENTICAL_ACTIONS,
    _MAX_PREMATURE_ALL_FAILED_RETRIES,
    _MAX_PREMATURE_DELETION_INCOMPLETE_RETRIES,
    _MAX_PREMATURE_FALSE_FINISH_RETRIES,
    _MAX_PREMATURE_TABLE_VERIFY_RETRIES,
    _MAX_PREMATURE_TRUE_FINISH_RETRIES,
    _MAX_PREMATURE_VALIDATION_ERROR_RETRIES,
    _MAX_REQUEST_USER_INPUT_CALLS,
    _PREMATURE_ALL_FAILED_NUDGE,
    _PREMATURE_FALSE_FINISH_NUDGE,
    _PREMATURE_TRUE_FINISH_NUDGE,
    _QA_ANSWER_FORMAT_GUIDANCE,
    _QA_SUMMARY_ACTION_REJECTED_NUDGE,
    _QA_SUMMARY_MAX_STEPS,
    _RAG_CHUNKS_PER_STEP,
    _build_history_digest,
    _build_nudge_message,
    _compact_anthropic_messages,
    _compact_gemini_messages,
    _compact_groq_messages,
    _make_dialog_handler,
    _login_form_needs_password,
    _is_deletion_intent_goal,
    _is_edit_all_intent_goal,
    _is_fatal_validation_error,
    _scan_created_item_in_table,
    _scan_remaining_target_records,
    _scan_validation_errors,
    _is_bare_required_message,
    _label_looks_like_form_submit,
    _should_check_validation_error_after_action,
)


async def _await_background_tasks(before: set) -> None:
    """W41: long_term_memory.record_task() ตอนนี้ยิงเป็น background task (ไม่ await ก่อน
    run_task() return แล้ว — ดู orchestrator.py::_fire_and_forget) เทสต์ที่เช็ค call args
    ของ record_task() ต้องรอ background task ที่ยังค้างอยู่ให้เสร็จก่อนเสมอ ไม่งั้น assert
    อาจทำงานก่อน task ได้รันจริง (race condition) — รับ `before` (snapshot ของ
    _background_tasks ก่อนเรียก run_task()) มาด้วย แล้ว gather() เฉพาะ task ที่ "ใหม่" ตั้งแต่
    เทสต์นี้เริ่มเท่านั้น (ไม่แตะ task ค้างจากเทสต์ก่อนหน้าที่ event loop ของมันปิดไปแล้ว —
    pytest-asyncio สร้าง event loop ใหม่แยกทุกเทสต์ตาม default — gather() ข้าม task ของ loop
    อื่นจะ throw ValueError ทันที)"""
    pending = orchestrator_module._background_tasks - before
    if pending:
        await asyncio.gather(*pending)

# ทุกเทสต์ mock ทั้ง Playwright และ llm.next_action — ไม่เปิด browser จริง ไม่ยิง LLM API จริง
# สำคัญ: ต้องส่ง provider="anthropic" ให้ run_task() ตรงๆ เสมอ ห้ามปล่อยให้ fallback ไป
# settings.llm_provider เพราะค่านั้นอ่านจาก .env ของเครื่อง dev แต่ละคน — ถ้า .env ตั้ง
# LLM_PROVIDER=groq ไว้ (เช่นตอนทดสอบ) แล้วเทสต์ไม่ pin provider ให้ตรงกับที่ mock ไว้
# (llm.next_action) จะหลุดไปเรียก llm.next_action_groq ตัวจริงที่ไม่ได้ mock -> ยิง Groq
# API จริงระหว่างรัน pytest (เคยเกิดขึ้นมาแล้ว)

_GOTO_OK = ActionResult(True, "goto", "ไปที่ url")
_WAIT_OK = ActionResult(True, "wait_stable", "หน้านิ่งแล้ว")


# pacing delay ท้ายทุก step (settings.step_pacing_delay_seconds) กันไม่ให้ test suite ช้าจริง —
# เหมือน test_actions.py ที่ mock asyncio.sleep กัน _ACTION_RETRY_DELAY_SEC ค้าง
@pytest.fixture(autouse=True)
def _no_real_sleep():
    with patch("backend.app.core.orchestrator.asyncio.sleep", AsyncMock()) as mock_sleep:
        yield mock_sleep


@pytest.mark.asyncio
async def test_run_task_step_pacing_delay_reads_from_settings(monkeypatch, _no_real_sleep):
    """Speed 2.4: pacing delay ต้องอ่านจาก settings.step_pacing_delay_seconds (ปรับได้จาก
    .env) ไม่ใช่ hardcode module constant ตายตัวเหมือนเดิมอีกต่อไป"""
    from backend.app.config import settings

    monkeypatch.setattr(settings, "step_pacing_delay_seconds", 0.05)
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    next_action_calls = [
        ("browser_action", {"type": "click", "index": 1}, "t1", [], llm.TokenUsage()),
        ("finish_task", {"success": True, "message": "เสร็จ"}, "", [], llm.TokenUsage()),
    ]

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=[]), \
         patch("backend.app.core.orchestrator.long_term_memory.recall", return_value=[]), \
         patch("backend.app.core.orchestrator.execute", AsyncMock(return_value=ActionResult(True, "click", "สำเร็จ"))), \
         patch("backend.app.core.orchestrator.llm.append_tool_result", side_effect=lambda m, tid, r: m), \
         patch("backend.app.core.orchestrator.llm.next_action", AsyncMock(side_effect=next_action_calls)):
        await Orchestrator().run_task("https://example.com", "goal", max_steps=10, provider="anthropic")

    # elapsed ระหว่าง step แทบเป็นศูนย์ (ทุกอย่าง mock ให้จบทันที) — remaining ที่ส่งเข้า
    # sleep() ต้องใกล้เคียงค่าที่ตั้งไว้ (0.05) ไม่ใช่ค่า hardcode เดิม (3)
    _no_real_sleep.assert_awaited_once()
    remaining_arg = _no_real_sleep.await_args.args[0]
    assert 0 < remaining_arg <= 0.05


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


# Task4 (W19, "Task Completion Verifier"): _scan_validation_errors() เรียก page.locator()
# จริงเหมือน _login_form_needs_password ด้านบน — pattern เดียวกันทุกประการ (mock ให้ตรงๆ
# เป็น [] ว่างเปล่า/ไม่พบ error เลย เป็น default ของทุกเทสต์ กันรก RuntimeWarning ใน
# เทสต์อื่นที่ไม่ได้ตั้งใจทดสอบ guard นี้ — เทสต์เฉพาะของ guard นี้ override เอง)
@pytest.fixture(autouse=True)
def _no_validation_errors_by_default():
    with patch("backend.app.core.orchestrator._scan_validation_errors", AsyncMock(return_value=[])):
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


# --- W17[hardened]: _maybe_auto_login() คืนข้อความเหตุผลเมื่อ login ไม่ผ่าน (แทนที่จะ
# เงียบๆ) — run_task() ต้องยิง SSE event "auto_login_failed" แจ้ง user ต่อ แต่ยังรัน task
# ต่อไปตามปกติ (ไม่ throw/ไม่หยุด — agent ยัง fallback กรอกฟอร์ม login เองได้)


@pytest.mark.asyncio
async def test_run_task_emits_auto_login_failed_event_when_login_does_not_verify():
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    on_event = AsyncMock()

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=[]), \
         patch(
             "backend.app.core.orchestrator._maybe_auto_login",
             AsyncMock(return_value="ล็อกอินไม่สำเร็จด้วย credential ที่บันทึกไว้สำหรับเว็บนี้"),
         ), \
         patch(
             "backend.app.core.orchestrator.llm.next_action",
             AsyncMock(return_value=("finish_task", {"success": True, "message": "เสร็จ"}, "", [], llm.TokenUsage())),
         ):
        result = await Orchestrator().run_task(
            "https://example.com", "goal", provider="anthropic", on_event=on_event,
        )

    # task ยังรันต่อจนจบตามปกติ ไม่ถูกหยุดเพราะ auto-login fail
    assert result["success"] is True
    failed_events = [c.args[0] for c in on_event.await_args_list if c.args[0].get("kind") == "auto_login_failed"]
    assert len(failed_events) == 1
    assert failed_events[0]["reason"] == "ล็อกอินไม่สำเร็จด้วย credential ที่บันทึกไว้สำหรับเว็บนี้"


@pytest.mark.asyncio
async def test_run_task_does_not_emit_auto_login_failed_event_when_login_succeeds_or_not_attempted():
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    on_event = AsyncMock()

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=[]), \
         patch("backend.app.core.orchestrator._maybe_auto_login", AsyncMock(return_value=None)), \
         patch(
             "backend.app.core.orchestrator.llm.next_action",
             AsyncMock(return_value=("finish_task", {"success": True, "message": "เสร็จ"}, "", [], llm.TokenUsage())),
         ):
        await Orchestrator().run_task(
            "https://example.com", "goal", provider="anthropic", on_event=on_event,
        )

    failed_events = [c.args[0] for c in on_event.await_args_list if c.args[0].get("kind") == "auto_login_failed"]
    assert failed_events == []


# --- W66[C] ("Fast-Path Navigation", manual trigger): nav_target_page_query is opt-in
# (default None = พฤติกรรมเดิมทุกประการ, เทสต์เดิมทั้งไฟล์ที่ไม่ระบุ parameter นี้ยังผ่านหมด
# — ดูผลรัน). ทั้ง 3 เทสต์ mock _maybe_auto_login/site_learning.storage/
# fastpath_executor.execute_navigation ให้ไม่แตะ browser/LLM จริงเลย


@pytest.mark.asyncio
async def test_run_task_skips_nav_fastpath_silently_when_no_manual_exists():
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)) as mock_goto, \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=[]), \
         patch("backend.app.core.orchestrator._maybe_auto_login", AsyncMock(return_value=None)), \
         patch("backend.app.site_learning.storage.load_manual", return_value=None) as mock_load_manual, \
         patch("backend.app.core.orchestrator.fastpath_executor.execute_navigation", AsyncMock()) as mock_exec_nav, \
         patch(
             "backend.app.core.orchestrator.llm.next_action",
             AsyncMock(return_value=("finish_task", {"success": True, "message": "เสร็จ"}, "", [], llm.TokenUsage())),
         ):
        result = await Orchestrator().run_task(
            "https://example.com", "goal", provider="anthropic", nav_target_page_query="Admin",
        )

    assert result["success"] is True
    mock_load_manual.assert_called_once()
    mock_exec_nav.assert_not_awaited()  # ไม่มี manual -> ไม่ลอง replay เลย
    assert mock_goto.await_count == 1  # goto เริ่มต้นครั้งเดียวตามปกติ ไม่มี fallback goto ซ้ำ


@pytest.mark.asyncio
async def test_run_task_continues_normally_after_successful_nav_fastpath():
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    mock_manual = MagicMock()
    mock_target_page = MagicMock()

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)) as mock_goto, \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=[]), \
         patch("backend.app.core.orchestrator._maybe_auto_login", AsyncMock(return_value=None)), \
         patch("backend.app.site_learning.storage.load_manual", return_value=mock_manual), \
         patch("backend.app.site_learning.storage.find_matching_page", return_value=mock_target_page), \
         patch(
             "backend.app.core.orchestrator.fastpath_executor.execute_navigation",
             AsyncMock(return_value={"success": True, "steps": 2, "message": "ok", "execution_mode": "nav"}),
         ) as mock_exec_nav, \
         patch(
             "backend.app.core.orchestrator.llm.next_action",
             AsyncMock(return_value=("finish_task", {"success": True, "message": "เสร็จ"}, "", [], llm.TokenUsage())),
         ):
        result = await Orchestrator().run_task(
            "https://example.com", "goal", provider="anthropic", nav_target_page_query="Admin",
        )

    assert result["success"] is True
    mock_exec_nav.assert_awaited_once()
    assert mock_goto.await_count == 1  # nav สำเร็จ -> ไม่ต้อง goto กลับจุดเริ่มต้นซ้ำ


@pytest.mark.asyncio
async def test_run_task_falls_back_to_original_url_when_nav_fastpath_fails():
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    mock_manual = MagicMock()
    mock_target_page = MagicMock()

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)) as mock_goto, \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=[]), \
         patch("backend.app.core.orchestrator._maybe_auto_login", AsyncMock(return_value=None)), \
         patch("backend.app.site_learning.storage.load_manual", return_value=mock_manual), \
         patch("backend.app.site_learning.storage.find_matching_page", return_value=mock_target_page), \
         patch(
             "backend.app.core.orchestrator.fastpath_executor.execute_navigation",
             AsyncMock(return_value={"success": False, "steps": 0, "message": "[FAIL]", "execution_mode": "nav_unavailable"}),
         ) as mock_exec_nav, \
         patch(
             "backend.app.core.orchestrator.llm.next_action",
             AsyncMock(return_value=("finish_task", {"success": True, "message": "เสร็จ"}, "", [], llm.TokenUsage())),
         ):
        result = await Orchestrator().run_task(
            "https://example.com", "goal", provider="anthropic", nav_target_page_query="Admin",
        )

    # ล้มเหลว -> ยัง fallback ไปทำงานปกติต่อได้ ไม่แย่กว่าเดิม (ไม่ throw/ไม่หยุด task)
    assert result["success"] is True
    mock_exec_nav.assert_awaited_once()
    assert mock_goto.await_count == 2  # goto เริ่มต้น + goto กลับไป url เดิมหลัง nav ล้มเหลว
    assert mock_goto.await_args_list[-1].args[1] == "https://example.com"


# --- W67[D] ("Fast-Path Navigation", auto-decide): ไม่ระบุ nav_target_page_query เอง แต่
# settings.enable_nav_fastpath_auto_decide เปิดอยู่ (default True) -> ใช้ goal ตรงๆ เป็น query
# แทน โดยใช้ threshold เข้มกว่า (nav_fastpath_min_match_score) จาก find_matching_page() —
# explicit nav_target_page_query ยัง override ได้เหมือนเดิมด้วย threshold หลวม (1)


@pytest.mark.asyncio
async def test_run_task_auto_decides_nav_fastpath_using_goal_when_query_not_given():
    from backend.app.config import settings

    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    mock_manual = MagicMock()
    mock_target_page = MagicMock()

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)) as mock_goto, \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=[]), \
         patch("backend.app.core.orchestrator._maybe_auto_login", AsyncMock(return_value=None)), \
         patch("backend.app.site_learning.storage.load_manual", return_value=mock_manual), \
         patch("backend.app.site_learning.storage.find_matching_page", return_value=mock_target_page) as mock_find, \
         patch(
             "backend.app.core.orchestrator.fastpath_executor.execute_navigation",
             AsyncMock(return_value={"success": True, "steps": 2, "message": "ok", "execution_mode": "nav"}),
         ) as mock_exec_nav, \
         patch(
             "backend.app.core.orchestrator.llm.next_action",
             AsyncMock(return_value=("finish_task", {"success": True, "message": "เสร็จ"}, "", [], llm.TokenUsage())),
         ):
        result = await Orchestrator().run_task(
            "https://example.com", "ไปหน้า Admin จัดการผู้ใช้", provider="anthropic",
        )

    assert result["success"] is True
    mock_exec_nav.assert_awaited_once()  # ไม่ระบุ nav_target_page_query เอง แต่ auto-decide ยัง trigger ได้
    mock_find.assert_called_once_with(
        mock_manual, "ไปหน้า Admin จัดการผู้ใช้", min_score=settings.nav_fastpath_min_match_score,
    )
    assert mock_goto.await_count == 1


@pytest.mark.asyncio
async def test_run_task_does_not_auto_decide_nav_fastpath_when_setting_disabled(monkeypatch):
    from backend.app.config import settings

    monkeypatch.setattr(settings, "enable_nav_fastpath_auto_decide", False)
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)) as mock_goto, \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=[]), \
         patch("backend.app.core.orchestrator._maybe_auto_login", AsyncMock(return_value=None)), \
         patch("backend.app.site_learning.storage.load_manual") as mock_load_manual, \
         patch("backend.app.core.orchestrator.fastpath_executor.execute_navigation", AsyncMock()) as mock_exec_nav, \
         patch(
             "backend.app.core.orchestrator.llm.next_action",
             AsyncMock(return_value=("finish_task", {"success": True, "message": "เสร็จ"}, "", [], llm.TokenUsage())),
         ):
        result = await Orchestrator().run_task(
            "https://example.com", "ไปหน้า Admin จัดการผู้ใช้", provider="anthropic",
        )

    assert result["success"] is True
    mock_load_manual.assert_not_called()  # setting ปิด -> ไม่แม้แต่จะพยายามหา manual เลย
    mock_exec_nav.assert_not_awaited()
    assert mock_goto.await_count == 1


@pytest.mark.asyncio
async def test_run_task_explicit_nav_query_uses_loose_threshold_not_auto_decide_threshold():
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    mock_manual = MagicMock()
    mock_target_page = MagicMock()

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=[]), \
         patch("backend.app.core.orchestrator._maybe_auto_login", AsyncMock(return_value=None)), \
         patch("backend.app.site_learning.storage.load_manual", return_value=mock_manual), \
         patch("backend.app.site_learning.storage.find_matching_page", return_value=mock_target_page) as mock_find, \
         patch(
             "backend.app.core.orchestrator.fastpath_executor.execute_navigation",
             AsyncMock(return_value={"success": True, "steps": 2, "message": "ok", "execution_mode": "nav"}),
         ), \
         patch(
             "backend.app.core.orchestrator.llm.next_action",
             AsyncMock(return_value=("finish_task", {"success": True, "message": "เสร็จ"}, "", [], llm.TokenUsage())),
         ):
        await Orchestrator().run_task(
            "https://example.com", "goal", provider="anthropic", nav_target_page_query="Admin",
        )

    mock_find.assert_called_once_with(mock_manual, "Admin", min_score=1)


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

    # extract_domain() ตัด "www." ออกโดยเจตนา (กัน credential/allowed_domains แยกกันเป็นคนละ
    # โดเมนทั้งที่เป็นเว็บเดียวกัน — ดู permission/rules.py::extract_domain())
    mock_execute.assert_awaited_once_with(
        mock_page, {"type": "click", "index": 1},
        ask_user_func=None, label="", manual_guidance="",
        allowed_domains={"saucedemo.com"}, element_tag="", element_type="",
        then_label="", then_tag="", then_type="",
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
        allowed_domains={"custom.example.com"}, element_tag="", element_type="",
        then_label="", then_tag="", then_type="",
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
        ask_user_func=None, label="", manual_guidance="", allowed_domains=None, element_tag="", element_type="",
        then_label="", then_tag="", then_type="",
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
            # W_procmem: click_result เป็น ActionResult ที่สร้างขึ้นตรงๆ ในเทสต์นี้ (ไม่ผ่าน
            # actions.py จริง) เลยไม่มี locator_descriptor แนบมา (default None)
            "locator_descriptor": None,
        },
    ]
    # ต้องรวม token ของทั้ง 2 รอบ next_action (browser_action + finish_task) ไม่ใช่แค่รอบสุดท้าย
    assert result["tokens"] == {"input": 110, "output": 25, "cache_read": 0, "cache_creation": 0}


@pytest.mark.asyncio
async def test_run_task_resolves_label_tag_type_for_then_click_index():
    """W_chain ("Compound Actions"): then_click_index ต้อง resolve label/tag/type จาก
    elements snapshot เดียวกับที่ action หลัก (index) ใช้ — เหมือน action_label/
    action_tag/action_element_type ทุกประการแค่สำหรับ target ตัวที่สอง (ดู
    orchestrator.py บริเวณที่ resolve action_label ก่อน execute())"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    click_result = ActionResult(True, "select(1)", "เลือกสำเร็จ + then click(5): คลิกสำเร็จ")
    elements = [
        {"index": 1, "label": "Carolynn", "tag": "li", "type": ""},
        {"index": 5, "label": "Submit", "tag": "button", "type": ""},
    ]

    next_action_calls = [
        (
            "browser_action",
            {"type": "select", "index": 1, "label": "Carolynn", "then_click_index": 5},
            "tool_1", ["m1"], llm.TokenUsage(),
        ),
        ("finish_task", {"success": True, "message": "เสร็จ"}, "", ["m2"], llm.TokenUsage()),
    ]

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=(elements, "elements"))), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=[]), \
         patch("backend.app.core.orchestrator.execute", AsyncMock(return_value=click_result)) as mock_execute, \
         patch("backend.app.core.orchestrator.llm.append_tool_result", side_effect=lambda m, tid, r: m + [r]), \
         patch("backend.app.core.orchestrator.llm.next_action", AsyncMock(side_effect=next_action_calls)):
        result = await Orchestrator().run_task("https://example.com", "choose from list", provider="anthropic")

    assert result["success"] is True
    mock_execute.assert_awaited_once_with(
        mock_browser.new_page.return_value,
        {"type": "select", "index": 1, "label": "Carolynn", "then_click_index": 5},
        ask_user_func=None, label="Carolynn", manual_guidance="", allowed_domains=None,
        element_tag="li", element_type="",
        then_label="Submit", then_tag="button", then_type="",
    )


# --- W_resume ("Mid-Task Input Request") — บั๊กจริงที่ user รายงาน: agent ขอรหัสผ่านใหม่
# กลางทางแล้ว finish_task(false) จบ task ทั้งหมด ทำให้เทิร์นถัดไปที่ user ตอบค่ามาต้องเริ่ม
# งานใหม่จากศูนย์แทนที่จะทำ plan เดิมต่อ — request_user_input ต้อง "หยุดรอ" ผ่าน
# ask_user_func เดียวกับ permission prompt แล้วทำ loop เดิมต่อทันทีด้วยคำตอบที่ได้ (ไม่
# return/ไม่จบ task)


@pytest.mark.asyncio
async def test_run_task_request_user_input_pauses_then_continues_same_loop_with_answer():
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    click_result = ActionResult(True, "fill(0)", "กรอกสำเร็จ")

    next_action_calls = [
        (
            "request_user_input",
            {"prompt": "รหัสผ่านใหม่คืออะไร?", "sensitive": True},
            "tool_ask", ["m1"], llm.TokenUsage(),
        ),
        ("browser_action", {"type": "fill", "index": 0, "text": "answer-goes-here"}, "tool_fill", ["m2"], llm.TokenUsage()),
        ("finish_task", {"success": True, "message": "เสร็จ"}, "", ["m3"], llm.TokenUsage()),
    ]

    async def fake_ask_user_func(cmd):
        assert cmd["type"] == "request_user_input"
        cmd["answer"] = "Sup3rSecret!"  # mutate in place, mirrors resolve_approval() mutating cmd["answer"]
        return True

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "elements"))), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=[]), \
         patch("backend.app.core.orchestrator.execute", AsyncMock(return_value=click_result)), \
         patch("backend.app.core.orchestrator.llm.append_tool_result", side_effect=lambda m, tid, r: m + [r]) as mock_append, \
         patch("backend.app.core.orchestrator.llm.next_action", AsyncMock(side_effect=next_action_calls)):
        result = await Orchestrator().run_task(
            "https://example.com", "change my password", provider="anthropic",
            ask_user_func=fake_ask_user_func,
        )

    # ไม่จบ task ตอนเจอ request_user_input เลย — ทำ loop เดิมต่อจน finish_task(true) จริง
    assert result["success"] is True
    assert result["message"] == "เสร็จ"
    # คำตอบที่ user ให้มาต้องถูกป้อนกลับเป็น tool_result ของ tool_use "tool_ask" (ไม่ใช่
    # ถูกทิ้ง/เพิกเฉย) ให้ LLM เห็นแล้วใช้ทำ step ถัดไปต่อ
    answer_result_calls = [c for c in mock_append.call_args_list if c.args[1] == "tool_ask"]
    assert len(answer_result_calls) == 1
    assert "Sup3rSecret!" in answer_result_calls[0].args[2]
    # history ต้องมี entry ของ request_user_input step ด้วย (ไม่ใช่แค่ fill/finish)
    request_steps = [h for h in result["history"] if h["cmd"].get("type") == "request_user_input"]
    assert len(request_steps) == 1
    assert request_steps[0]["success"] is True


@pytest.mark.asyncio
async def test_run_task_request_user_input_declined_still_continues_without_hanging():
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()

    next_action_calls = [
        ("request_user_input", {"prompt": "ต้องการค่าอะไรสักอย่าง"}, "tool_ask", ["m1"], llm.TokenUsage()),
        ("finish_task", {"success": False, "message": "ทำต่อไม่ได้"}, "tool_f", ["m2"], llm.TokenUsage()),
    ]

    async def declining_ask_user_func(cmd):
        return False  # ผู้ใช้ปฏิเสธ/หมดเวลา — ไม่ mutate cmd["answer"] เลย

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "elements"))), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=[]), \
         patch("backend.app.core.orchestrator.llm.append_tool_result", side_effect=lambda m, tid, r: m + [r]), \
         patch("backend.app.core.orchestrator.llm.next_action", AsyncMock(side_effect=next_action_calls)):
        result = await asyncio.wait_for(
            Orchestrator().run_task(
                "https://example.com", "goal", provider="anthropic",
                # max_steps=2: หลัง request_user_input 1 step (steps_taken=1), เงื่อนไข
                # "premature false finish" guard (steps_taken < max_steps-1) ต้องเป็นเท็จ
                # ทันที กันไม่ให้ finish_task(false) ที่ next_action_calls จบไว้ให้ต้องผ่าน
                # retry-nudge cycle ของ guard นั้นเพิ่ม (คนละเรื่องกับสิ่งที่เทสต์นี้สนใจ)
                max_steps=2, ask_user_func=declining_ask_user_func,
            ),
            timeout=5,
        )

    # ไม่ค้าง (มี timeout ป้องกันไว้ด้านบนแล้ว) และจบด้วยผลลัพธ์ที่สมเหตุสมผล ไม่ throw
    assert result["success"] is False


@pytest.mark.asyncio
async def test_run_task_request_user_input_repeat_cap_forces_rejection():
    """ป้องกัน LLM วนถามไม่รู้จบโดยไม่มีความคืบหน้าจริง — เกินโควตา
    _MAX_REQUEST_USER_INPUT_CALLS แล้วต้องถูกปฏิเสธ ไม่ใช่หยุดรอเพิ่มอีก"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()

    # เรียก request_user_input ซ้ำเกินโควตา 1 ครั้ง แล้วค่อย finish_task
    next_action_calls = [
        ("request_user_input", {"prompt": f"คำถามที่ {i}"}, f"tool_{i}", [f"m{i}"], llm.TokenUsage())
        for i in range(_MAX_REQUEST_USER_INPUT_CALLS + 1)
    ] + [("finish_task", {"success": False, "message": "ทำต่อไม่ได้"}, "tool_f", ["mf"], llm.TokenUsage())]

    ask_user_func = AsyncMock(return_value=False)

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "elements"))), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=[]), \
         patch("backend.app.core.orchestrator.llm.append_tool_result", side_effect=lambda m, tid, r: m + [r]), \
         patch("backend.app.core.orchestrator.llm.next_action", AsyncMock(side_effect=next_action_calls)):
        await asyncio.wait_for(
            Orchestrator().run_task(
                "https://example.com", "goal", provider="anthropic",
                # max_steps=4: หลัง request_user_input ที่ "ยอมรับ" ครบ
                # _MAX_REQUEST_USER_INPUT_CALLS (3) ครั้ง steps_taken=3 — ต้องให้เงื่อนไข
                # premature-false-finish guard เป็นเท็จทันทีตอนถึง finish_task(false)
                # เหมือนเทสต์ข้างบน (คนละเรื่องกับ repeat cap ที่เทสต์นี้สนใจจริงๆ)
                max_steps=4, ask_user_func=ask_user_func,
            ),
            timeout=5,
        )

    # ask_user_func ต้องถูกเรียกไม่เกิน quota เลย (ครั้งที่เกินโควตาต้องถูกปฏิเสธก่อนจะไป
    # เรียก ask_user_func เลยด้วยซ้ำ ไม่ใช่แค่ไม่หยุดรอ)
    assert ask_user_func.await_count == _MAX_REQUEST_USER_INPUT_CALLS


# W44: qa_summary ตอนนี้วน next_action() แบบจำกัด (ดู _QA_SUMMARY_MAX_STEPS) แทนที่จะเรียก
# llm.summarize_page() ตัวเดียวแบบเดิม — goal ทุกเทสต์ด้านล่างใช้คำใน qa_keywords
# (llm.classify_intent()) ตรงๆ ("มีสินค้าอะไรบ้าง") ไม่ปนคำใน action_keywords เลย เพื่อให้
# classify_intent() คืน "qa_summary" ผ่าน heuristic ทันที ไม่ต้องยิง LLM จริง (กัน network
# call ระหว่าง pytest)
_QA_GOAL = "ตารางนี้มีสินค้าอะไรบ้าง"


@pytest.mark.asyncio
async def test_run_task_qa_summary_answers_directly_when_finish_task_called_immediately():
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "[0] button 'Go'"))), \
         patch("backend.app.core.orchestrator.execute", AsyncMock()) as mock_execute, \
         patch("backend.app.core.orchestrator.llm.summarize_page", AsyncMock()) as mock_summarize, \
         patch(
             "backend.app.core.orchestrator.llm.next_action",
             AsyncMock(return_value=(
                 "finish_task", {"success": True, "message": "มีสินค้า 3 ชิ้น"}, "", [],
                 llm.TokenUsage(input_tokens=30, output_tokens=6),
             )),
         ):
        result = await Orchestrator().run_task("https://example.com", _QA_GOAL, provider="anthropic")

    assert result["status"] == "chat_reply"
    assert result["success"] is True
    assert result["message"] == "มีสินค้า 3 ชิ้น"
    assert result["tokens"] == {"input": 30, "output": 6, "cache_read": 0, "cache_creation": 0}
    mock_execute.assert_not_called()
    mock_summarize.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_task_qa_summary_calls_read_page_data_via_execute_then_answers():
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    mock_page = mock_browser.new_page.return_value
    read_result = ActionResult(True, "read_page_data", "พบ 1 รายการที่ตรงกับ 'table tbody tr': Cierra Vega")

    qa_next_action_calls = [
        (
            "browser_action",
            {"type": "read_page_data", "query": "มีชื่อ Cierra ไหม", "target_hint": "table tbody tr"},
            "qa_tool_1", ["m1"], llm.TokenUsage(input_tokens=40, output_tokens=8),
        ),
        (
            "finish_task", {"success": True, "message": "เห็นชื่อ Cierra ในตารางค่ะ"}, "", ["m2"],
            llm.TokenUsage(input_tokens=45, output_tokens=9),
        ),
    ]

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "[table not shown here]"))), \
         patch("backend.app.core.orchestrator.execute", AsyncMock(return_value=read_result)) as mock_execute, \
         patch("backend.app.core.orchestrator.llm.append_tool_result", side_effect=lambda m, tid, r: m + [r]), \
         patch("backend.app.core.orchestrator.llm.summarize_page", AsyncMock()) as mock_summarize, \
         patch("backend.app.core.orchestrator.llm.next_action", AsyncMock(side_effect=qa_next_action_calls)):
        result = await Orchestrator().run_task("https://example.com", _QA_GOAL, provider="anthropic")

    assert result["status"] == "chat_reply"
    assert result["message"] == "เห็นชื่อ Cierra ในตารางค่ะ"
    mock_execute.assert_awaited_once_with(
        mock_page, {"type": "read_page_data", "query": "มีชื่อ Cierra ไหม", "target_hint": "table tbody tr"},
        ask_user_func=None, label="", manual_guidance="", allowed_domains=None,
    )
    mock_summarize.assert_not_awaited()
    assert result["tokens"] == {"input": 85, "output": 17, "cache_read": 0, "cache_creation": 0}


@pytest.mark.asyncio
async def test_run_task_qa_summary_rejects_non_read_page_data_action_without_dispatching():
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()

    qa_next_action_calls = [
        ("browser_action", {"type": "click", "index": 2}, "qa_tool_1", ["m1"], llm.TokenUsage(input_tokens=10, output_tokens=2)),
        ("finish_task", {"success": True, "message": "ตอบได้จากข้อมูลที่เห็นอยู่แล้ว"}, "", ["m2"], llm.TokenUsage(input_tokens=11, output_tokens=3)),
    ]
    append_tool_result_calls = []

    def _fake_append_tool_result(messages, tool_use_id, result_text):
        append_tool_result_calls.append((tool_use_id, result_text))
        return messages + [result_text]

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "[2] button 'Delete'"))), \
         patch("backend.app.core.orchestrator.execute", AsyncMock()) as mock_execute, \
         patch("backend.app.core.orchestrator.llm.append_tool_result", side_effect=_fake_append_tool_result), \
         patch("backend.app.core.orchestrator.llm.summarize_page", AsyncMock()) as mock_summarize, \
         patch("backend.app.core.orchestrator.llm.next_action", AsyncMock(side_effect=qa_next_action_calls)):
        result = await Orchestrator().run_task("https://example.com", _QA_GOAL, provider="anthropic")

    assert result["message"] == "ตอบได้จากข้อมูลที่เห็นอยู่แล้ว"
    mock_execute.assert_not_called()
    mock_summarize.assert_not_awaited()
    assert append_tool_result_calls == [("qa_tool_1", _QA_SUMMARY_ACTION_REJECTED_NUDGE)]


@pytest.mark.asyncio
async def test_run_task_qa_summary_falls_back_to_summarize_page_when_steps_exhausted():
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    read_result = ActionResult(True, "read_page_data", "พบ 40 รายการ")

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "[some elements]"))), \
         patch("backend.app.core.orchestrator.execute", AsyncMock(return_value=read_result)) as mock_execute, \
         patch("backend.app.core.orchestrator.llm.append_tool_result", side_effect=lambda m, tid, r: m + [r]), \
         patch("backend.app.core.orchestrator.llm.summarize_page", AsyncMock(return_value="สรุปแบบเดิมจาก fallback")) as mock_summarize, \
         patch(
             "backend.app.core.orchestrator.llm.next_action",
             AsyncMock(return_value=(
                 "browser_action", {"type": "read_page_data", "query": "q", "target_hint": "table"}, "qa_tool", ["m"],
                 llm.TokenUsage(input_tokens=5, output_tokens=1),
             )),
         ) as mock_next_action:
        result = await Orchestrator().run_task("https://example.com", _QA_GOAL, provider="anthropic")

    assert result["message"] == "สรุปแบบเดิมจาก fallback"
    assert mock_next_action.await_count == _QA_SUMMARY_MAX_STEPS
    assert mock_execute.await_count == _QA_SUMMARY_MAX_STEPS
    mock_summarize.assert_awaited_once()


# user รายงานว่า agent ตอบ "list รายชื่อ" ด้วยการแปะรายละเอียดอื่นที่ไม่มีใครถามปนมา (ตำแหน่ง/
# office/salary) และเขียนรวมเป็นย่อหน้าเดียวยาวแทนที่จะขึ้นบรรทัดใหม่ทีละข้อ — ตอนนี้ qa_summary
# ต่อ _QA_ANSWER_FORMAT_GUIDANCE เข้าไปในทั้ง goal ที่ next_action() เห็น และ user_prompt ที่
# summarize_page() fallback เห็น (ดู orchestrator.py qa_goal)
@pytest.mark.asyncio
async def test_run_task_qa_summary_appends_answer_format_guidance_to_goal_seen_by_next_action():
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "[0] button 'Go'"))), \
         patch("backend.app.core.orchestrator.execute", AsyncMock()), \
         patch(
             "backend.app.core.orchestrator.llm.next_action",
             AsyncMock(return_value=(
                 "finish_task", {"success": True, "message": "1. A\n2. B"}, "", [],
                 llm.TokenUsage(input_tokens=10, output_tokens=2),
             )),
         ) as mock_next_action:
        await Orchestrator().run_task("https://example.com", _QA_GOAL, provider="anthropic")

    goal_seen_by_next_action = mock_next_action.await_args.args[2]
    assert goal_seen_by_next_action == f"{_QA_GOAL}{_QA_ANSWER_FORMAT_GUIDANCE}"


@pytest.mark.asyncio
async def test_run_task_qa_summary_appends_answer_format_guidance_to_summarize_page_fallback():
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "[some elements]"))), \
         patch("backend.app.core.orchestrator.execute", AsyncMock()), \
         patch("backend.app.core.orchestrator.llm.summarize_page", AsyncMock(return_value="สรุป")) as mock_summarize, \
         patch(
             "backend.app.core.orchestrator.llm.next_action",
             AsyncMock(return_value=(
                 "finish_task", {"success": False}, "", [], llm.TokenUsage(),
             )),
         ):
        await Orchestrator().run_task("https://example.com", _QA_GOAL, provider="anthropic")

    mock_summarize.assert_awaited_once()
    assert mock_summarize.await_args.kwargs["user_prompt"] == f"{_QA_GOAL}{_QA_ANSWER_FORMAT_GUIDANCE}"


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
        ask_user_func=None, label="", manual_guidance="", allowed_domains=None, element_tag="", element_type="",
        then_label="", then_tag="", then_type="",
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


# --- ACC-3 (accuracy audit follow-up): finish_task(success=true) hard guard when every
# mutating action attempted in the task has failed (steps_taken > 0, so the zero-steps
# guard above doesn't already catch it, and no validation-error text on the page either) ---


@pytest.mark.asyncio
async def test_run_task_rejects_finish_task_true_when_every_mutating_action_failed():
    """click ล้มเหลวทุกครั้ง (steps_taken > 0 แต่ไม่มี mutating action ไหนสำเร็จเลย) แล้ว
    เรียก finish_task(success=true) — ต้องถูกปฏิเสธและเตือนก่อน ไม่ใช่ยอมรับทันที"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    failed_click = ActionResult(False, "click", "หา element ไม่เจอ")

    next_action_calls = [
        ("browser_action", {"type": "click", "index": 1}, "t1", [], llm.TokenUsage()),
        ("finish_task", {"success": True, "message": "สำเร็จแล้ว"}, "tool_t1", ["m1"], llm.TokenUsage()),
        ("finish_task", {"success": True, "message": "ยืนยันสำเร็จจริง"}, "tool_t2", ["m2"], llm.TokenUsage()),
        ("finish_task", {"success": True, "message": "ยืนยันสำเร็จจริงอีกครั้ง"}, "tool_t3", ["m3"], llm.TokenUsage()),
    ]
    append_tool_result_mock = MagicMock(side_effect=lambda m, tid, r: m + [r])

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=[]), \
         patch("backend.app.core.orchestrator.execute", AsyncMock(return_value=failed_click)), \
         patch("backend.app.core.orchestrator.llm.append_tool_result", append_tool_result_mock), \
         patch(
             "backend.app.core.orchestrator.llm.next_action", AsyncMock(side_effect=next_action_calls)
         ) as mock_next_action:
        result = await Orchestrator().run_task("https://example.com", "goal", provider="anthropic")

    assert result["success"] is True
    assert result["message"] == "ยืนยันสำเร็จจริงอีกครั้ง"
    # 1 click (fail) + 2 finish_task ที่ถูกเตือน (nudge) + 1 finish_task ที่ยอมรับจริง
    assert mock_next_action.await_count == 1 + _MAX_PREMATURE_ALL_FAILED_RETRIES + 1
    append_tool_result_mock.assert_any_call(["m1"], "tool_t1", _PREMATURE_ALL_FAILED_NUDGE)
    append_tool_result_mock.assert_any_call(["m2"], "tool_t2", _PREMATURE_ALL_FAILED_NUDGE)


@pytest.mark.asyncio
async def test_run_task_accepts_finish_task_true_after_max_all_failed_retries():
    """โมเดลยืนยัน finish_task(true) ซ้ำหลังโดนเตือนแล้วเกิน
    _MAX_PREMATURE_ALL_FAILED_RETRIES — ต้องยอมรับจริง ไม่บังคับลองต่อไม่มีที่สิ้นสุด
    (escape valve เดียวกับ guard อื่นในไฟล์นี้)"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    failed_click = ActionResult(False, "click", "หา element ไม่เจอ")

    next_action_calls = [
        ("browser_action", {"type": "click", "index": 1}, "t1", [], llm.TokenUsage()),
    ] + [
        ("finish_task", {"success": True, "message": "ยืนยันสำเร็จจริงแน่นอน"}, f"tool_t{i}", [], llm.TokenUsage())
        for i in range(_MAX_PREMATURE_ALL_FAILED_RETRIES + 1)
    ]

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=[]), \
         patch("backend.app.core.orchestrator.execute", AsyncMock(return_value=failed_click)), \
         patch("backend.app.core.orchestrator.llm.append_tool_result", side_effect=lambda m, tid, r: m + [r]), \
         patch(
             "backend.app.core.orchestrator.llm.next_action", AsyncMock(side_effect=next_action_calls)
         ) as mock_next_action:
        result = await Orchestrator().run_task("https://example.com", "goal", provider="anthropic")

    assert result["success"] is True
    assert result["message"] == "ยืนยันสำเร็จจริงแน่นอน"
    # 1 click (fail) + (_MAX_PREMATURE_ALL_FAILED_RETRIES + 1) finish_task calls
    assert mock_next_action.await_count == 1 + _MAX_PREMATURE_ALL_FAILED_RETRIES + 1


@pytest.mark.asyncio
async def test_run_task_does_not_reject_finish_task_true_when_one_mutating_action_succeeded():
    """sanity: บาง action fail แต่มีอย่างน้อย 1 อันสำเร็จจริง — guard ใหม่ต้องไม่ยิงเลย
    (แค่บาง action fail ไม่ได้แปลว่า task ล้มเหลว — retry ปกติของแต่ละ action เองจัดการอยู่
    แล้ว ดู actions.py::_dispatch_with_retry)"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    results = [ActionResult(False, "click", "fail"), ActionResult(True, "click", "สำเร็จ")]

    next_action_calls = [
        ("browser_action", {"type": "click", "index": 1}, "t1", [], llm.TokenUsage()),
        ("browser_action", {"type": "click", "index": 2}, "t2", [], llm.TokenUsage()),
        ("finish_task", {"success": True, "message": "เสร็จแล้ว"}, "tool_t1", [], llm.TokenUsage()),
    ]

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=[]), \
         patch("backend.app.core.orchestrator.execute", AsyncMock(side_effect=results)), \
         patch("backend.app.core.orchestrator.llm.append_tool_result", side_effect=lambda m, tid, r: m), \
         patch(
             "backend.app.core.orchestrator.llm.next_action", AsyncMock(side_effect=next_action_calls)
         ) as mock_next_action:
        result = await Orchestrator().run_task("https://example.com", "goal", provider="anthropic")

    assert result["success"] is True
    assert mock_next_action.await_count == 3  # ไม่มี retry แถมจาก guard ใหม่


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

    # W14/W30/W32/W43: args ท้ายสุดตามลำดับคือ site_manual_context, current_url,
    # action_history_context, plan_context (ใหม่) — vision_context เลยอยู่ args[-5]
    second_call_vision_context = mock_next_action.await_args_list[1].args[-5]
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


# W19 ("Navigation Deduplication"): generate_plan() ต้องส่ง current_url จริง (page.url ถ้ามี
# page เปิดอยู่) ให้ llm.generate_plan() เห็น เพื่อให้ planner รู้ว่า "อยู่หน้าเป้าหมายอยู่
# แล้วหรือยัง" ก่อนร่างขั้นตอน navigate ซ้ำที่ไม่จำเป็น


@pytest.mark.asyncio
async def test_generate_plan_passes_empty_current_url_when_no_page():
    mock_generate_plan = AsyncMock(return_value=("1. Do X", False))
    with patch("backend.app.core.orchestrator.llm.generate_plan", mock_generate_plan), \
         patch("backend.app.core.orchestrator.llm.classify_intent", AsyncMock(return_value="action_task")):
        await Orchestrator().generate_plan("https://example.com", "goal", provider="anthropic")

    assert mock_generate_plan.await_args.kwargs["current_url"] == ""


@pytest.mark.asyncio
async def test_generate_plan_passes_live_page_url_as_current_url():
    mock_page = AsyncMock()
    mock_page.url = "https://example.com/admin/viewSystemUsers"
    mock_generate_plan = AsyncMock(return_value=("1. Search user", False))
    with patch("backend.app.core.orchestrator.llm.generate_plan", mock_generate_plan), \
         patch("backend.app.core.orchestrator.llm.classify_intent", AsyncMock(return_value="action_task")), \
         patch(
             "backend.app.core.orchestrator.get_snapshot",
             AsyncMock(return_value=([], "[1] input 'Search'")),
         ):
        await Orchestrator().generate_plan(
            "https://example.com", "หน้า Admin", provider="anthropic", page=mock_page,
        )

    assert mock_generate_plan.await_args.kwargs["current_url"] == "https://example.com/admin/viewSystemUsers"


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
async def test_run_task_forces_recovery_action_on_repeated_identical_action_then_continues():
    """W31: loop-detection guard (คาบ 1) เดิมจบ task ทันทีที่ trigger — ตอนนี้ต้องบังคับ
    ทำ go_back แทนก่อน (ไม่ผ่าน LLM) แล้วให้ agent ลองต่อจาก state ใหม่ — พิสูจน์ด้วยการ
    ให้ mock คืน action เดิมซ้ำจนครบเพดาน แล้วสลับเป็น action ที่ต่างออกไปซึ่งนำไปสู่
    finish_task(success=true) ได้จริง ต้องไม่ถูกตัดจบก่อนเวลาเพราะ recovery แทรกเข้ามา"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    click_result = ActionResult(True, "click(5)", "คลิกสำเร็จ")
    same_click = {"type": "click", "index": 5}

    next_action_calls = [
        ("browser_action", same_click, "t1", [], llm.TokenUsage()),
        ("browser_action", same_click, "t2", [], llm.TokenUsage()),
        ("browser_action", same_click, "t3", [], llm.TokenUsage()),  # trigger ที่นี่ -> forced go_back แทน
        ("browser_action", {"type": "click", "index": 9}, "t4", [], llm.TokenUsage()),  # action ใหม่หลัง recovery
        ("finish_task", {"success": True, "message": "เสร็จ"}, "", [], llm.TokenUsage()),
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

    assert result["success"] is True  # ไม่ถูกตัดจบก่อนเวลา — ไปต่อจนสำเร็จได้จริง
    # 2 click สำเร็จ (count 1,2) + 1 go_back บังคับ (แทนที่ click ตัวที่ 3 ที่ถูกบล็อก) +
    # 1 click ใหม่หลัง recovery = 4 ครั้ง (finish_task ไม่ผ่าน execute())
    assert mock_execute.await_count == 4
    forced_call = mock_execute.await_args_list[2]
    assert forced_call.args[1] == {"type": "go_back"}
    assert mock_next_action.await_count == 5


@pytest.mark.asyncio
async def test_run_task_loop_guard_ignores_completed_plan_step_when_detecting_period_1_repeat():
    """W29 (บั๊กจริงที่ user รายงาน — agent ติดลูปกดตัวเลือก dropdown ซ้ำๆ ไม่หยุด): action
    เดิมเป๊ะ (type/index เท่ากันทุกอย่าง) แต่ครั้งแรกมี completed_plan_step ติดมาด้วย (เพิ่ง
    ทำให้ step ของแผนเสร็จ) ครั้งต่อๆ ไปไม่มี key นี้อีกแล้ว (mark ไปแล้วครั้งเดียวตามกติกา
    ห้าม mark ซ้ำ) — ต้องยังถูกนับเป็น "action เดิมซ้ำ" อยู่ดี ไม่ใช่มองว่าเป็นคนละ action
    เพราะ dict ไม่เท่ากันเป๊ะ (ก่อนแก้ W29: guard นี้จะไม่ trigger เลยในสถานการณ์นี้)"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    click_result = ActionResult(True, "click(22)", "คลิกสำเร็จ")

    next_action_calls = [
        ("browser_action", {"type": "click", "index": 22, "completed_plan_step": 2}, "t1", [], llm.TokenUsage()),
        ("browser_action", {"type": "click", "index": 22}, "t2", [], llm.TokenUsage()),
        ("browser_action", {"type": "click", "index": 22}, "t3", [], llm.TokenUsage()),  # trigger ที่นี่ -> forced go_back แทน
        ("browser_action", {"type": "click", "index": 9}, "t4", [], llm.TokenUsage()),  # action ใหม่หลัง recovery
        ("finish_task", {"success": True, "message": "เสร็จ"}, "", [], llm.TokenUsage()),
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

    assert result["success"] is True  # ไปต่อจนสำเร็จได้จริงหลัง recovery
    forced_call = mock_execute.await_args_list[2]
    assert forced_call.args[1] == {"type": "go_back"}  # trigger ได้จริงแม้ dict ไม่เท่ากันเป๊ะ


@pytest.mark.asyncio
async def test_run_task_stops_on_alternating_cycle_even_when_first_occurrence_has_completed_plan_step():
    """W29: จำลองบั๊กจริงตรงๆ ที่ user รายงาน — สลับ 2 index ไปมา (click(22) <-> click(25))
    โดย click(22) ครั้งแรกสุดมี completed_plan_step ติดมาด้วย (ทำให้ step กรองข้อมูลเสร็จ)
    ครั้งต่อๆ ไปไม่มีอีกแล้ว — guard คาบ 2 ต้อง trigger จริง (บังคับ go_back) ไม่ถูกบล็อกเพราะ
    dict ไม่เท่ากันเป๊ะจาก completed_plan_step ที่หายไปตั้งแต่รอบสอง แล้วไปต่อจนสำเร็จได้จริง
    หลัง recovery (เหมือน pattern เดียวกับเทสต์คาบ 1 ด้านบน — ตรวจแบบ behavioral ผ่าน
    forced go_back call แทนที่จะขับไปจนล้มเหลว/หมดโควตา recovery)"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    click_result = ActionResult(True, "click", "สำเร็จ")
    action_22_first = {"type": "click", "index": 22, "completed_plan_step": 2}
    action_22 = {"type": "click", "index": 22}
    action_25 = {"type": "click", "index": 25}

    next_action_calls = [
        ("browser_action", action_22_first, "t1", [], llm.TokenUsage()),
        ("browser_action", action_25, "t2", [], llm.TokenUsage()),
        ("browser_action", action_22, "t3", [], llm.TokenUsage()),
        ("browser_action", action_25, "t4", [], llm.TokenUsage()),  # ครบ window คาบ 2 (4 action) ที่นี่ -> trigger
        ("browser_action", {"type": "click", "index": 30}, "t5", [], llm.TokenUsage()),  # action ใหม่หลัง recovery
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
            "https://example.com", "goal", max_steps=30, provider="anthropic"
        )

    assert result["success"] is True  # ไปต่อจนสำเร็จได้จริงหลัง recovery ไม่ถูกตัดจบก่อนเวลา
    forced_go_back_calls = [c for c in mock_execute.await_args_list if c.args[1] == {"type": "go_back"}]
    assert len(forced_go_back_calls) == 1  # trigger ได้จริงแม้ dict ไม่เท่ากันเป๊ะจาก completed_plan_step


@pytest.mark.asyncio
async def test_run_task_aborts_after_exhausting_forced_loop_recoveries():
    """W31: ถ้า agent ยังวนกลับมาเรียก action เดิมซ้ำแม้บังคับ recovery ไปแล้ว ต้อง
    ยอมแพ้จริงในที่สุด (escape valve กัน force ไม่รู้จบ) — mock คืน click index 5 ซ้ำ
    ตลอดกาลไม่ว่าจะบังคับ recovery ไปกี่ครั้งก็ตาม

    trace ที่เกิดขึ้นจริง (_MAX_CONSECUTIVE_IDENTICAL_ACTIONS=3,
    _MAX_FORCED_LOOP_RECOVERIES=2): รอบ 1 (click,click,[trigger]->forced go_back) ->
    รอบ 2 (click,click,[trigger]->forced scroll) -> รอบ 3 (click,click,[trigger]-> เกิน
    เพดานแล้ว ยอมแพ้จริง ไม่ execute action ที่ถูกบล็อก) — รวม execute() 3*2+2=8 ครั้ง
    (2 click ต่อรอบ x3 รอบ + go_back + scroll), next_action ถูกเรียก 3*3=9 ครั้ง (รวม
    call ที่ 3 ของแต่ละรอบที่ trigger guard ด้วย)"""
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
            "https://example.com", "goal", max_steps=20, provider="anthropic"
        )

    assert result["success"] is False
    assert "ซ้ำ" in result["message"]
    assert "บังคับ" in result["message"]
    assert result["steps"] == 8
    assert mock_execute.await_count == 8
    assert mock_next_action.await_count == 9
    forced_types = [c.args[1].get("type") for c in mock_execute.await_args_list if c.args[1].get("type") != "click"]
    assert forced_types == ["go_back", "scroll"]  # ลำดับ recovery action ตาม _LOOP_RECOVERY_ACTIONS


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
async def test_run_task_loop_guard_does_not_trigger_for_varied_read_page_data_queries():
    """read_page_data ที่ query ต่างกันไม่ควรถูกนับเป็น action ซ้ำ — user ถามคำถามต่อเนื่อง
    หลายข้อ ("ราคานี้เท่าไหร่" "อันนี้ล่ะ" ...) ต้องไม่โดน force_loop_recovery (W31) เข้าใจ
    ผิดว่า agent วนซ้ำไม่มีความคืบหน้า — loop-guard เทียบ tool_input ทั้ง dict อยู่แล้ว
    (ไม่ใช่แค่ชื่อ type) เลย query ที่ต่างกันทำให้ dict ไม่เท่ากันเองโดยธรรมชาติ"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    read_result = ActionResult(True, "read_page_data", "พบ 1 รายการ")

    next_action_calls = [
        ("browser_action", {"type": "read_page_data", "query": "สินค้าชิ้นที่ 1 ราคาเท่าไหร่", "target_hint": ".item-1"}, "t1", [], llm.TokenUsage()),
        ("browser_action", {"type": "read_page_data", "query": "สินค้าชิ้นที่ 2 ราคาเท่าไหร่", "target_hint": ".item-2"}, "t2", [], llm.TokenUsage()),
        ("browser_action", {"type": "read_page_data", "query": "สินค้าชิ้นที่ 3 ราคาเท่าไหร่", "target_hint": ".item-3"}, "t3", [], llm.TokenUsage()),
        ("browser_action", {"type": "read_page_data", "query": "สินค้าชิ้นที่ 4 ราคาเท่าไหร่", "target_hint": ".item-4"}, "t4", [], llm.TokenUsage()),
        ("finish_task", {"success": True, "message": "ตอบครบทุกข้อแล้ว"}, "", [], llm.TokenUsage()),
    ]

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.execute", AsyncMock(return_value=read_result)) as mock_execute, \
         patch("backend.app.core.orchestrator.llm.append_tool_result", side_effect=lambda m, tid, r: m), \
         patch("backend.app.core.orchestrator.llm.next_action", AsyncMock(side_effect=next_action_calls)):
        result = await Orchestrator().run_task(
            "https://example.com", "goal", max_steps=10, provider="anthropic"
        )

    assert result["success"] is True
    # ทุก read_page_data (4 ข้อคำถามต่างกัน) ต้องถูก execute จริง ไม่มีตัวไหนถูกบล็อกด้วย
    # loop-guard เพราะเข้าใจผิดว่าเป็น action เดิมซ้ำ
    assert mock_execute.await_count == 4


@pytest.mark.asyncio
async def test_run_task_loop_guard_still_triggers_for_identical_repeated_read_page_data_query():
    """ตรงข้ามกับเทสต์ข้างบน — query (และ target_hint) เดิมเป๊ะซ้ำติดกันจริงๆ (ไม่ใช่
    คำถามใหม่) ยังต้องถูกจับว่าเป็น action ซ้ำเหมือน action ประเภทอื่นตามปกติ ไม่ใช่ว่า
    read_page_data ได้รับการยกเว้นจาก loop-guard ไปเลยทั้งหมด"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    read_result = ActionResult(True, "read_page_data", "พบ 1 รายการ")
    same_query = {"type": "read_page_data", "query": "มีสินค้ากี่ชิ้น", "target_hint": ".inventory_item"}

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.execute", AsyncMock(return_value=read_result)) as mock_execute, \
         patch("backend.app.core.orchestrator.llm.append_tool_result", side_effect=lambda m, tid, r: m), \
         patch("backend.app.core.orchestrator.llm.next_action", AsyncMock(return_value=(
             "browser_action", same_query, "tool_x", [], llm.TokenUsage()
         ))):
        result = await Orchestrator().run_task(
            "https://example.com", "goal", max_steps=20, provider="anthropic"
        )

    assert result["success"] is False
    assert "ซ้ำ" in result["message"]
    assert "บังคับ" in result["message"]


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
    """loop-detection (คาบ 2, 2026-07-13): agent วนสลับ 2 action ที่ไม่เหมือนกันไปมา —
    guard เดิม (_MAX_CONSECUTIVE_IDENTICAL_ACTIONS) จับได้แค่ action เดิมเป๊ะๆ ซ้ำติดกัน
    (คาบ 1) ไม่ตรงเงื่อนไขนี้เลยไม่เคย trigger ต้องมี guard ใหม่จับคาบ 2 (ABAB) แยกต่างหาก

    W31: trigger แล้วตอนนี้บังคับ recovery action ก่อน (ไม่จบ task ทันทีเหมือนเดิม) — ใช้
    itertools.cycle ให้ mock วน A/B ไม่รู้จบ (เหมือน agent ที่ไม่ยอมเปลี่ยนพฤติกรรมเอง) กัน
    escape valve (_MAX_FORCED_LOOP_RECOVERIES) ในที่สุด แทนที่จะ hand-compute จำนวนครั้ง
    เป๊ะๆ (ซับซ้อนเกินไปเพราะ forced action เองก็เข้าไปอยู่ใน recent_actions window ที่ใช้
    ตรวจจับคาบต่อด้วย) ตรวจแค่ผลลัพธ์สุดท้ายที่สำคัญจริง: จบด้วย fail + ข้อความบอกคาบ 2
    ชัดเจน + มีการบังคับ go_back ไปแล้วอย่างน้อย 1 ครั้งจริง + ไม่ได้วนจนหมด max_steps"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    click_result = ActionResult(True, "click", "สำเร็จ")
    action_a = {"type": "click", "index": 1}
    action_b = {"type": "click", "index": 2}

    next_action_calls = itertools.cycle([
        ("browser_action", action_a, "t1", [], llm.TokenUsage()),
        ("browser_action", action_b, "t2", [], llm.TokenUsage()),
    ])

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.execute", AsyncMock(return_value=click_result)) as mock_execute, \
         patch("backend.app.core.orchestrator.llm.append_tool_result", side_effect=lambda m, tid, r: m), \
         patch("backend.app.core.orchestrator.llm.next_action", AsyncMock(side_effect=next_action_calls)):
        result = await Orchestrator().run_task(
            "https://example.com", "goal", max_steps=30, provider="anthropic"
        )

    assert result["success"] is False
    assert "คาบ 2" in result["message"]
    assert "บังคับ" in result["message"]
    assert result["steps"] < 30  # ยอมแพ้ก่อนหมด max_steps จริง ไม่ใช่วนจนครบ budget
    forced_go_back_calls = [c for c in mock_execute.await_args_list if c.args[1] == {"type": "go_back"}]
    assert len(forced_go_back_calls) >= 1


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

    next_action_calls = itertools.cycle([
        ("browser_action", action_a, "t1", [], llm.TokenUsage()),
        ("browser_action", action_b, "t2", [], llm.TokenUsage()),
        ("browser_action", action_c, "t3", [], llm.TokenUsage()),
    ])

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.execute", AsyncMock(return_value=click_result)) as mock_execute, \
         patch("backend.app.core.orchestrator.llm.append_tool_result", side_effect=lambda m, tid, r: m), \
         patch("backend.app.core.orchestrator.llm.next_action", AsyncMock(side_effect=next_action_calls)):
        result = await Orchestrator().run_task(
            "https://example.com", "goal", max_steps=30, provider="anthropic"
        )

    # W31: trigger แล้วบังคับ recovery ก่อน (ดู test_run_task_stops_on_alternating_two_
    # action_pattern สำหรับเหตุผลที่ตรวจแบบ behavioral แทน exact count)
    assert result["success"] is False
    assert "คาบ 3" in result["message"]
    assert "บังคับ" in result["message"]
    assert result["steps"] < 30
    forced_go_back_calls = [c for c in mock_execute.await_args_list if c.args[1] == {"type": "go_back"}]
    assert len(forced_go_back_calls) >= 1


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

    next_action_calls = itertools.cycle([
        ("browser_action", action_a, "t1", [], llm.TokenUsage()),
        ("browser_action", action_b, "t2", [], llm.TokenUsage()),
        ("browser_action", action_c, "t3", [], llm.TokenUsage()),
        ("browser_action", action_d, "t4", [], llm.TokenUsage()),
    ])

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.execute", AsyncMock(return_value=click_result)) as mock_execute, \
         patch("backend.app.core.orchestrator.llm.append_tool_result", side_effect=lambda m, tid, r: m), \
         patch("backend.app.core.orchestrator.llm.next_action", AsyncMock(side_effect=next_action_calls)):
        result = await Orchestrator().run_task(
            "https://example.com", "goal", max_steps=30, provider="anthropic"
        )

    # W31: ดูเหตุผลที่ตรวจแบบ behavioral ใน test_run_task_stops_on_alternating_two_
    # action_pattern
    assert result["success"] is False
    assert "คาบ 4" in result["message"]
    assert "บังคับ" in result["message"]
    assert result["steps"] < 30
    forced_go_back_calls = [c for c in mock_execute.await_args_list if c.args[1] == {"type": "go_back"}]
    assert len(forced_go_back_calls) >= 1


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
         patch("backend.app.core.orchestrator._embedding_function", return_value=[[0.1, 0.2]]), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=["chunk1", "chunk2", "chunk3"]) as mock_retrieve, \
         patch(
             "backend.app.core.orchestrator.llm.next_action",
             AsyncMock(return_value=("finish_task", {"success": True, "message": "เสร็จแล้ว"}, "", [], llm.TokenUsage())),
         ) as mock_next_action:
        await Orchestrator().run_task("https://example.com", "some goal", provider="anthropic")

    # Speed 2.2: orchestrator embed step_embed_input ครั้งเดียว (mock ไว้ด้านบนคืน
    # [[0.1, 0.2]] เสมอ) แล้วส่ง query_embedding=[0.1, 0.2] เข้า retrieve() ด้วย
    mock_retrieve.assert_called_once_with(
        query="some goal", page_state="[0] button 'Go'", k=_RAG_CHUNKS_PER_STEP, query_embedding=[0.1, 0.2],
    )
    # W14/W30/W32/W43: args ท้ายสุดตามลำดับคือ manual_context, memory_context,
    # long_term_context, vision_context, site_manual_context, current_url,
    # action_history_context, plan_context (ใหม่) — manual_context เลยอยู่ args[-8]
    manual_context = mock_next_action.await_args.args[-8]
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

    # W14/W30/W32/W43: ดูคอมเมนต์เต็มใน test_run_task_calls_retrieve_with_goal_page_text_and_
    # k_then_passes_result_into_next_action — manual_context อยู่ args[-8]
    manual_context = mock_next_action.await_args.args[-8]
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

    # W14/W30/W32/W43: memory_context อยู่ args[-7] (ดูลำดับเต็มใน test_run_task_calls_
    # retrieve_with_goal_page_text_and_k_then_passes_result_into_next_action)
    first_call_memory_context = mock_next_action.await_args_list[0].args[-7]
    second_call_memory_context = mock_next_action.await_args_list[1].args[-7]
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

    # W14/W30/W32/W43: memory_context อยู่ args[-7]
    memory_context = mock_next_action.await_args.args[-7]
    assert memory_context == ""


# --- W30: current_url ส่งเข้า next_action() ทุก step + แจ้งเตือนเมื่อ URL เปลี่ยนไปเอง ---


@pytest.mark.asyncio
async def test_run_task_passes_live_page_url_into_next_action():
    """W30: page.url ต้องถูกส่งเข้า next_action() ทุก step (args[-3] — ดู current_url
    parameter) อ่านสดจาก page object จริง ไม่ใช่ค่าที่จำมาจาก step ก่อน"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    mock_page = mock_browser.new_page.return_value
    mock_page.url = "https://example.com/cart"

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

    current_url = mock_next_action.await_args.args[-3]
    assert current_url == "https://example.com/cart"


@pytest.mark.asyncio
async def test_run_task_flags_unexpected_navigation_after_non_navigational_action():
    """W30: click ธรรมดาที่ดันมี redirect/JS navigation ซ่อนอยู่ (page.url เปลี่ยนไปเอง
    หลัง execute() แม้ tool_input.type ไม่ใช่ goto/switch_tab/go_back) ต้องแนบคำเตือนต่อ
    ท้าย tool_result ให้โมเดลรู้ตัวว่าหน้าเว็บเปลี่ยนไปแล้ว ไม่ใช่หน้าที่วางแผนไว้"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    mock_page = mock_browser.new_page.return_value
    mock_page.url = "https://example.com/start"
    click_result = ActionResult(True, "click(1)", "สำเร็จ")

    async def _execute_side_effect(page, cmd, **kwargs):
        mock_page.url = "https://example.com/unexpected"  # จำลอง redirect หลังคลิก
        return click_result

    next_action_calls = [
        ("browser_action", {"type": "click", "index": 1}, "t1", [], llm.TokenUsage()),
        ("finish_task", {"success": True, "message": "เสร็จ"}, "", [], llm.TokenUsage()),
    ]
    captured_tool_results = []

    def _capture_append_tool_result(messages, tool_use_id, result_text):
        captured_tool_results.append(result_text)
        return messages

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=[]), \
         patch("backend.app.core.orchestrator.execute", AsyncMock(side_effect=_execute_side_effect)), \
         patch("backend.app.core.orchestrator.llm.append_tool_result", side_effect=_capture_append_tool_result), \
         patch("backend.app.core.orchestrator.llm.next_action", AsyncMock(side_effect=next_action_calls)):
        await Orchestrator().run_task("https://example.com", "goal", provider="anthropic")

    assert len(captured_tool_results) == 1
    assert "หน้าเว็บเปลี่ยนไปเองหลัง action นี้" in captured_tool_results[0]
    assert "https://example.com/start" in captured_tool_results[0]
    assert "https://example.com/unexpected" in captured_tool_results[0]


@pytest.mark.asyncio
async def test_run_task_does_not_flag_navigation_for_goto_action():
    """goto มีจุดประสงค์หลักคือเปลี่ยนหน้าอยู่แล้ว — URL เปลี่ยนไม่ใช่เรื่อง "ไม่คาดคิด"
    ต้องไม่แนบคำเตือนซ้ำ"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    mock_page = mock_browser.new_page.return_value
    mock_page.url = "https://example.com/start"
    goto_result = ActionResult(True, "goto", "สำเร็จ")

    async def _execute_side_effect(page, cmd, **kwargs):
        mock_page.url = "https://example.com/new-page"
        return goto_result

    next_action_calls = [
        ("browser_action", {"type": "goto", "url": "https://example.com/new-page"}, "t1", [], llm.TokenUsage()),
        ("finish_task", {"success": True, "message": "เสร็จ"}, "", [], llm.TokenUsage()),
    ]
    captured_tool_results = []

    def _capture_append_tool_result(messages, tool_use_id, result_text):
        captured_tool_results.append(result_text)
        return messages

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=[]), \
         patch("backend.app.core.orchestrator.execute", AsyncMock(side_effect=_execute_side_effect)), \
         patch("backend.app.core.orchestrator.llm.append_tool_result", side_effect=_capture_append_tool_result), \
         patch("backend.app.core.orchestrator.llm.next_action", AsyncMock(side_effect=next_action_calls)):
        await Orchestrator().run_task("https://example.com", "goal", provider="anthropic")

    assert len(captured_tool_results) == 1
    assert "หน้าเว็บเปลี่ยนไปเองหลัง action นี้" not in captured_tool_results[0]


# --- W32: action_history_context ส่งเข้า next_action() ทุก step ---


@pytest.mark.asyncio
async def test_run_task_passes_action_history_context_into_next_action():
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    click_result = ActionResult(True, "click(1)", "สำเร็จ")

    next_action_calls = [
        ("browser_action", {"type": "click", "index": 1}, "t1", [], llm.TokenUsage()),
        ("finish_task", {"success": True, "message": "เสร็จ"}, "", [], llm.TokenUsage()),
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
        await Orchestrator().run_task("https://example.com", "goal", provider="anthropic")

    # call แรก: มีแค่ step 0 (goto เริ่มต้น) ใน history อยู่แล้ว ยังไม่มี browser_action ใดๆ
    # W43: args[-2] ไม่ใช่ args[-1] แล้ว เพราะ plan_context (ใหม่) มาต่อท้ายสุด
    first_call_history = mock_next_action.await_args_list[0].args[-2]
    assert "step 0" in first_call_history
    assert "goto" in first_call_history
    assert "click" not in first_call_history
    # call ที่สอง ต้องเห็น step 1 (click ที่เพิ่งทำ) เพิ่มเข้ามาแล้ว (ทั้งที่สำเร็จ ไม่ใช่
    # แค่ fail — ต่างจาก memory_context)
    second_call_history = mock_next_action.await_args_list[1].args[-2]
    assert "step 1" in second_call_history
    assert "click" in second_call_history


@pytest.mark.asyncio
async def test_run_task_calls_long_term_memory_recall_with_goal_page_text_and_k_then_passes_into_next_action():
    """W7[A] (long-term): ทุก step ต้อง recall(query=goal, page_state=page_text ปัจจุบัน,
    k=_LONG_TERM_MEMORY_CHUNKS_PER_STEP) แล้วเอาผลลัพธ์ (join เป็น bullet list) ส่งต่อเข้า
    next_action() เป็น long_term_context"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "[0] button 'Apply Code'"))), \
         patch("backend.app.core.orchestrator._embedding_function", return_value=[[0.1, 0.2]]), \
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

    # Speed 2.2: query_embedding=[0.1, 0.2] มาจาก _embedding_function mock ด้านบน (embed
    # ครั้งเดียวใช้ร่วมกับ retriever.retrieve())
    mock_recall.assert_called_once_with(
        query="some goal", page_state="[0] button 'Apply Code'", k=_LONG_TERM_MEMORY_CHUNKS_PER_STEP,
        session_id="", query_embedding=[0.1, 0.2],
    )
    # W14/W30/W32/W43: long_term_context อยู่ args[-6]
    long_term_context = mock_next_action.await_args.args[-6]
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

    # W14/W30/W32/W43: long_term_context อยู่ args[-6]
    long_term_context = mock_next_action.await_args.args[-6]
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
        before_tasks = set(orchestrator_module._background_tasks)
        result = await Orchestrator().run_task("https://example.com", "goal", provider="anthropic")
        await _await_background_tasks(before_tasks)

    assert result["success"] is False
    mock_record_task.assert_called_once_with(
        url="https://example.com",
        goal="goal",
        success=False,
        message="ทำต่อไม่ได้",
        failed_actions=mock_record_task.call_args.kwargs["failed_actions"],
        session_id="",
    )
    assert "หา element ไม่เจอ" in mock_record_task.call_args.kwargs["failed_actions"]


@pytest.mark.asyncio
async def test_run_task_threads_session_id_into_long_term_memory_recall_and_record():
    """W23: session_id ที่ run_task() รับมาต้องถูกส่งต่อเข้าทั้ง recall() (ทุก step) และ
    record_task() (ท้าย task) ตรงๆ — นี่คือกลไกจริงที่ทำให้ session หนึ่งดึงความจำของอีก
    session มาปนกันไม่ได้ (ดู core/long_term_memory.py — recall() ปฏิเสธ query ถ้าไม่มี
    session_id)"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=[]), \
         patch("backend.app.core.orchestrator.long_term_memory.recall", return_value=[]) as mock_recall, \
         patch("backend.app.core.orchestrator.long_term_memory.record_task") as mock_record_task, \
         patch(
             "backend.app.core.orchestrator.llm.next_action",
             AsyncMock(return_value=("finish_task", {"success": True, "message": "เสร็จแล้ว"}, "", [], llm.TokenUsage())),
         ):
        before_tasks = set(orchestrator_module._background_tasks)
        await Orchestrator().run_task(
            "https://example.com", "goal", provider="anthropic", session_id="session-abc-123",
        )
        await _await_background_tasks(before_tasks)

    assert mock_recall.call_args.kwargs["session_id"] == "session-abc-123"
    assert mock_record_task.call_args.kwargs["session_id"] == "session-abc-123"


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


def test_build_history_digest_summarizes_steps_up_to_cutoff():
    memory = ShortTermMemory()
    memory.record({"step": 0, "cmd": {"type": "goto", "url": "https://x"}, "result": "[OK] ไปที่ url", "success": True})
    memory.record({"step": 1, "cmd": {"type": "fill", "index": 0}, "result": "[OK] fill(0) -> login สำเร็จ", "success": True})
    memory.record({"step": 2, "cmd": {"type": "click", "index": 1}, "result": "[OK] click(1) -> เพิ่มสินค้า", "success": True})
    memory.record({"step": 3, "cmd": {"type": "click", "index": 2}, "result": "[FAIL] click(2) -> พัง", "success": False})

    digest = _build_history_digest(memory, upto_step=2)

    assert "step 1" in digest
    assert "login สำเร็จ" in digest
    assert "step 2" in digest
    assert "เพิ่มสินค้า" in digest
    assert "step 3" not in digest  # เกิน upto_step ไม่ควรโผล่
    assert "goto" not in digest  # step 0 ไม่นับ (ไม่ใช่ step ของ action จริง)


def test_build_history_digest_returns_empty_string_when_no_matching_steps():
    memory = ShortTermMemory()
    memory.record({"step": 0, "cmd": {"type": "goto", "url": "https://x"}, "result": "[OK]", "success": True})

    assert _build_history_digest(memory, upto_step=5) == ""


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


# --- W22: generalize compaction ไปทุก provider (Anthropic/Groq) ---


def test_compact_anthropic_messages_drops_old_turns_and_prepends_digest_to_kept_turn():
    messages = [
        {"role": "user", "content": "old step 1"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "browser_action", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "[OK]"}]},
        {"role": "user", "content": "old step 2"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t2", "name": "browser_action", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t2", "content": "[OK]"}]},
        {"role": "user", "content": "recent step 3 goal here"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t3", "name": "browser_action", "input": {}}]},
    ]

    result = _compact_anthropic_messages(messages, cut_at=6, digest_text="- step 1: ok\n- step 2: ok")

    assert len(result) == 2  # เท่ากับ len(messages) - cut_at เสมอ (แทนที่ text ไม่ตัด turn)
    assert "step 1: ok" in result[0]["content"]
    assert "recent step 3 goal here" in result[0]["content"]
    assert result[1] == messages[7]  # turn ที่เหลือไม่ถูกแตะเลย


def test_compact_anthropic_messages_is_noop_when_cut_at_zero_or_digest_empty():
    messages = [{"role": "user", "content": "x"}]

    assert _compact_anthropic_messages(messages, cut_at=0, digest_text="something") == messages
    assert _compact_anthropic_messages(messages, cut_at=1, digest_text="") == messages


def test_compact_anthropic_messages_falls_back_to_original_on_unexpected_shape():
    """ถ้า messages[cut_at] ไม่ใช่ user turn แบบ content เป็น string ล้วนๆ (เช่นโดน
    tool_result turn ซึ่ง content เป็น list มาอยู่ตำแหน่งนี้แทน — ผิดคาดจริงๆ) ต้องคืน
    messages เดิมไม่แก้อะไร ไม่ throw"""
    messages = [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "[OK]"}]}]

    assert _compact_anthropic_messages(messages, cut_at=0, digest_text="x") == messages


def test_compact_groq_messages_drops_old_turns_but_keeps_leading_system_message():
    """Groq เก็บ system prompt เป็น messages[0] เอง (ต่างจาก Anthropic/Gemini ที่ส่ง
    system แยกนอก messages) — ต้องกัน messages[0] ไว้เสมอ ไม่ให้หลุดไปอยู่ในส่วนที่ตัดทิ้ง"""
    messages = [
        {"role": "system", "content": "SYSTEM_PROMPT"},
        {"role": "user", "content": "old step 1"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "t1", "function": {"name": "browser_action", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "t1", "content": "[OK]"},
        {"role": "user", "content": "recent step 2 goal here"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "t2", "function": {"name": "browser_action", "arguments": "{}"}}]},
    ]

    result = _compact_groq_messages(messages, cut_at=4, digest_text="- step 1: ok")

    assert result[0] == {"role": "system", "content": "SYSTEM_PROMPT"}  # ไม่ถูกตัดทิ้งไปด้วย
    assert "step 1: ok" in result[1]["content"]
    assert "recent step 2 goal here" in result[1]["content"]
    assert result[2] == messages[5]


def test_compact_groq_messages_is_noop_when_no_leading_system_message():
    """ถ้า messages[0] ไม่ใช่ system message (ผิดคาดจริงๆ — next_action_groq() ควร
    ใส่ไว้เสมอตั้งแต่ call แรก) ต้องไม่กล้าตัดอะไรเลย เพราะเดา system message ไม่ได้"""
    messages = [{"role": "user", "content": "old step 1"}, {"role": "user", "content": "step 2"}]

    assert _compact_groq_messages(messages, cut_at=1, digest_text="x") == messages


def test_compact_groq_messages_is_noop_when_cut_at_zero_or_digest_empty():
    messages = [{"role": "system", "content": "SYSTEM_PROMPT"}, {"role": "user", "content": "x"}]

    assert _compact_groq_messages(messages, cut_at=0, digest_text="something") == messages
    assert _compact_groq_messages(messages, cut_at=1, digest_text="") == messages


@pytest.mark.asyncio
async def test_run_task_compacts_gemini_history_once_step_count_exceeds_threshold():
    """W7[A] (Test Case C): เกิน _COMPACT_AFTER_STEPS step แล้ว messages ที่ส่ง
    เข้า next_action_gemini() ต้องไม่โตต่อเนื่องไม่มีเพดานตามจำนวน step อีกต่อไป (ถูก
    ตัด step เก่ากว่า _KEEP_RECENT_STEPS ตัวล่าสุดออก) — digest ของ step แรกๆ
    ต้องยังโผล่อยู่ในบทสนทนาที่เหลือ (ไม่ได้หายไปเฉยๆ พิสูจน์ assertion #2 ของ Test Case C
    ที่ต้องการให้ agent ยังจำ step แรกๆ ได้)"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    click_result = ActionResult(True, "click", "สำเร็จ")

    total_action_steps = _COMPACT_AFTER_STEPS + 2  # ต้องเกิน threshold แน่ๆ
    step_actions = [
        ("browser_action", {"type": "click", "index": i}, f"call_{i}") for i in range(total_action_steps)
    ]
    step_actions.append(("finish_task", {"success": True, "message": "เสร็จ"}, ""))

    captured_messages_per_call: list[list] = []

    async def _next_action_side_effect(
        client, model, goal, page_text, messages, manual_context="", memory_context="",
        long_term_context="", vision_context="", site_manual_context="",
        current_url="", action_history_context="", plan_context="",
        verification_context="",
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
async def test_run_task_compacts_anthropic_history_once_step_count_exceeds_threshold():
    """W22: generalize context compaction จาก Gemini-only ไปทุก provider — Anthropic
    ต้องบีบอัด history เหมือน Gemini ทุกประการ (คู่กับ
    test_run_task_compacts_gemini_history_once_step_count_exceeds_threshold ด้านบน)"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    click_result = ActionResult(True, "click", "สำเร็จ")

    total_action_steps = _COMPACT_AFTER_STEPS + 2
    step_actions = [
        ("browser_action", {"type": "click", "index": i}, f"call_{i}") for i in range(total_action_steps)
    ]
    step_actions.append(("finish_task", {"success": True, "message": "เสร็จ"}, ""))

    captured_messages_per_call: list[list] = []

    async def _next_action_side_effect(
        client, model, goal, page_text, messages, manual_context="", memory_context="",
        long_term_context="", vision_context="", site_manual_context="",
        current_url="", action_history_context="", plan_context="",
        verification_context="",
    ):
        captured_messages_per_call.append(messages)
        i = len(captured_messages_per_call) - 1
        tool_name, tool_input, tool_use_id = step_actions[i]
        # จำลอง messages โตขึ้นจริงเหมือน implementation จริง (append 1 user turn +
        # 1 assistant turn ต่อ call, shape เดียวกับ llm.next_action() ตัวจริง) —
        # append_tool_result ตัวจริง (ไม่ mock) จะเพิ่มอีก 1 tool_result turn ต่อ step
        new_messages = messages + [
            {"role": "user", "content": f"Goal: {goal}\n\nหน้าเว็บปัจจุบัน:\n{page_text} #{i}"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": tool_use_id, "name": tool_name, "input": tool_input}]},
        ]
        return tool_name, tool_input, tool_use_id, new_messages, llm.TokenUsage()

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=[]), \
         patch("backend.app.core.orchestrator.execute", AsyncMock(return_value=click_result)), \
         patch("backend.app.core.llm.build_client", return_value="fake-client"), \
         patch("backend.app.core.orchestrator.llm.next_action", AsyncMock(side_effect=_next_action_side_effect)):
        result = await Orchestrator().run_task(
            "https://example.com", "goal", provider="anthropic", max_steps=total_action_steps + 2
        )

    assert result["success"] is True
    assert result["steps"] == total_action_steps

    # ถ้าไม่มี compaction เลย messages ของ call สุดท้าย (finish_task) จะยาว 3 ตัว/step x
    # total_action_steps — ต้องน้อยกว่านี้มากถ้า compaction ทำงานจริง
    last_call_messages = captured_messages_per_call[-1]
    uncompacted_would_be = total_action_steps * 3
    assert len(last_call_messages) < uncompacted_would_be

    # digest ของ step แรกๆ (ที่ถูกบีบอัดไปแล้ว) ต้องยังโผล่อยู่ในบทสนทนาที่เหลือ
    all_text = " ".join(
        msg["content"] for msg in last_call_messages if isinstance(msg.get("content"), str)
    )
    assert "step 1" in all_text
    assert "สรุป step ก่อนหน้า" in all_text


@pytest.mark.asyncio
async def test_run_task_compacts_groq_history_and_preserves_leading_system_message():
    """W22: Groq ก็ต้องบีบอัดเหมือนกัน — จุดที่ต่างจาก Anthropic/Gemini คือ Groq เก็บ
    system prompt เป็น messages[0] เอง ต้องรอดจากการบีบอัดทุกรอบ ไม่ใช่แค่ compaction
    ทำงาน (ดู test_compact_groq_messages_drops_old_turns_but_keeps_leading_system_message
    สำหรับ unit test แยกของฟังก์ชัน splice เอง — อันนี้พิสูจน์ผ่าน run_task() เต็มๆ)"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    click_result = ActionResult(True, "click", "สำเร็จ")

    total_action_steps = _COMPACT_AFTER_STEPS + 2
    step_actions = [
        ("browser_action", {"type": "click", "index": i}, f"call_{i}") for i in range(total_action_steps)
    ]
    step_actions.append(("finish_task", {"success": True, "message": "เสร็จ"}, ""))

    captured_messages_per_call: list[list] = []

    async def _next_action_side_effect(
        client, model, goal, page_text, messages, manual_context="", memory_context="",
        long_term_context="", vision_context="", site_manual_context="",
        current_url="", action_history_context="", plan_context="",
        verification_context="",
    ):
        captured_messages_per_call.append(messages)
        i = len(captured_messages_per_call) - 1
        tool_name, tool_input, tool_use_id = step_actions[i]
        if not messages:
            messages = [{"role": "system", "content": "SYSTEM_PROMPT"}]
        new_messages = messages + [
            {"role": "user", "content": f"Goal: {goal}\n\nหน้าเว็บปัจจุบัน:\n{page_text} #{i}"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": tool_use_id, "function": {"name": tool_name, "arguments": "{}"}}
            ]},
        ]
        return tool_name, tool_input, tool_use_id, new_messages, llm.TokenUsage()

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=[]), \
         patch("backend.app.core.orchestrator.execute", AsyncMock(return_value=click_result)), \
         patch("backend.app.core.llm.build_groq_client", return_value="fake-client"), \
         patch("backend.app.core.orchestrator.llm.next_action_groq", AsyncMock(side_effect=_next_action_side_effect)):
        result = await Orchestrator().run_task(
            "https://example.com", "goal", provider="groq", max_steps=total_action_steps + 2
        )

    assert result["success"] is True
    assert result["steps"] == total_action_steps

    last_call_messages = captured_messages_per_call[-1]
    uncompacted_would_be = total_action_steps * 3 + 1  # +1 สำหรับ system message ตัวเดียว
    assert len(last_call_messages) < uncompacted_would_be
    assert last_call_messages[0] == {"role": "system", "content": "SYSTEM_PROMPT"}

    all_text = " ".join(
        msg["content"] for msg in last_call_messages if isinstance(msg.get("content"), str)
    )
    assert "step 1" in all_text
    assert "สรุป step ก่อนหน้า" in all_text


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


# --- W43: SSE event "plan_step_done" — ติ๊ก checkbox ของ plan step แบบ real-time ---


@pytest.mark.asyncio
async def test_run_task_emits_plan_step_done_when_execute_succeeds_and_llm_marks_step_complete():
    """LLM ระบุ completed_plan_step=1 มาพร้อม action ที่ execute() สำเร็จจริง — ต้องยิง
    SSE event "plan_step_done" พร้อม step index นั้น"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    on_event = AsyncMock()
    click_result = ActionResult(True, "click(3)", "สำเร็จ")

    next_action_calls = [
        ("browser_action", {"type": "click", "index": 3, "completed_plan_step": 1}, "t1", [], llm.TokenUsage()),
        ("finish_task", {"success": True, "message": "เสร็จ"}, "", [], llm.TokenUsage()),
    ]

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.execute", AsyncMock(return_value=click_result)), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=[]), \
         patch("backend.app.core.orchestrator.llm.append_tool_result", side_effect=lambda m, tid, r: m), \
         patch("backend.app.core.orchestrator.llm.next_action", AsyncMock(side_effect=next_action_calls)):
        result = await Orchestrator().run_task(
            "https://example.com", "goal", provider="anthropic",
            approved_plan="1. คลิกปุ่ม Checkout\n2. ยืนยันคำสั่งซื้อ", on_event=on_event,
        )

    assert result["success"] is True
    plan_step_events = [c.args[0] for c in on_event.await_args_list if c.args[0].get("kind") == "plan_step_done"]
    assert len(plan_step_events) == 1
    assert plan_step_events[0]["step"] == 1


@pytest.mark.asyncio
async def test_run_task_does_not_emit_plan_step_done_when_execute_fails():
    """LLM ใส่ completed_plan_step มา แต่ action นั้น execute() ล้มเหลวจริง — ห้ามยิง
    plan_step_done เด็ดขาด (กันติ๊กผิดว่าทำสำเร็จทั้งที่ action พัง)"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    on_event = AsyncMock()
    fail_result = ActionResult(False, "click(3)", "หา element ไม่เจอ")

    # tool_use_id="" ตัวที่สอง เพื่อให้ finish_task(false) ถูกยอมรับทันที (ไม่ตกไปเจอ
    # premature-false-finish guard ที่ต้องมี tool_use_id จริงถึงจะเตือนแล้ว retry ต่อ)
    next_action_calls = [
        ("browser_action", {"type": "click", "index": 3, "completed_plan_step": 1}, "t1", [], llm.TokenUsage()),
        ("finish_task", {"success": False, "message": "ทำต่อไม่ได้"}, "", [], llm.TokenUsage()),
    ]

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.execute", AsyncMock(return_value=fail_result)), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=[]), \
         patch("backend.app.core.orchestrator.llm.append_tool_result", side_effect=lambda m, tid, r: m), \
         patch("backend.app.core.orchestrator.llm.next_action", AsyncMock(side_effect=next_action_calls)):
        await Orchestrator().run_task(
            "https://example.com", "goal", provider="anthropic",
            approved_plan="1. คลิกปุ่ม Checkout", on_event=on_event,
        )

    plan_step_events = [c.args[0] for c in on_event.await_args_list if c.args[0].get("kind") == "plan_step_done"]
    assert plan_step_events == []


@pytest.mark.asyncio
async def test_run_task_does_not_emit_plan_step_done_for_ad_hoc_task_without_plan():
    """ad-hoc task (ไม่มี approved_plan/confirm_plan เลย) — ถึง LLM จะใส่
    completed_plan_step มาผิดๆ (ไม่ควรทำแบบนี้ตาม SYSTEM_PROMPT แต่ป้องกันไว้ก่อน) ก็ห้ามยิง
    plan_step_done เด็ดขาด เพราะไม่มี checkbox ให้ติ๊กอยู่แล้วฝั่ง UI"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    on_event = AsyncMock()
    click_result = ActionResult(True, "click(3)", "สำเร็จ")

    next_action_calls = [
        ("browser_action", {"type": "click", "index": 3, "completed_plan_step": 1}, "t1", [], llm.TokenUsage()),
        ("finish_task", {"success": True, "message": "เสร็จ"}, "", [], llm.TokenUsage()),
    ]

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.execute", AsyncMock(return_value=click_result)), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=[]), \
         patch("backend.app.core.orchestrator.llm.append_tool_result", side_effect=lambda m, tid, r: m), \
         patch("backend.app.core.orchestrator.llm.next_action", AsyncMock(side_effect=next_action_calls)):
        result = await Orchestrator().run_task(
            "https://example.com", "goal", provider="anthropic", on_event=on_event,
        )

    assert result["success"] is True
    plan_step_events = [c.args[0] for c in on_event.await_args_list if c.args[0].get("kind") == "plan_step_done"]
    assert plan_step_events == []


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


# ---------------- W19 (ดู W19.txt ข้อ 8): Semantic Redundancy Evaluator wiring ----------------
# llm.evaluate_semantic_redundancy() เองมีเทสต์ครบใน test_llm.py แล้ว — กลุ่มนี้เทสต์แค่ว่า
# run_task() เรียกมันจริงตอน settings.enable_semantic_redundancy_check เปิดอยู่ และ
# short-circuit (ข้าม execute() ไปเลย ไม่นับ steps_taken) ตอนได้ SKIP_STEP/FORCE_REPLAN กลับมา
# — และไม่แตะมันเลยตอน flag ปิดอยู่ (default)


@pytest.mark.asyncio
async def test_run_task_skips_dispatch_when_semantic_evaluator_says_skip_step(monkeypatch):
    from backend.app.config import settings

    monkeypatch.setattr(settings, "enable_semantic_redundancy_check", True)

    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    click_result = ActionResult(True, "click(2)", "สำเร็จ")

    next_action_calls = [
        ("browser_action", {"type": "click", "index": 1}, "t0", [], llm.TokenUsage()),
        ("browser_action", {"type": "click", "index": 2}, "t1", [], llm.TokenUsage()),
        ("finish_task", {"success": True, "message": "เสร็จ"}, "", [], llm.TokenUsage()),
    ]
    evaluator_decisions = [
        {"is_semantically_redundant": True, "value_score": 0.1, "action_decision": "SKIP_STEP", "reasoning": "ไม่เกี่ยว"},
        {"is_semantically_redundant": False, "value_score": 0.9, "action_decision": "PASS", "reasoning": "จำเป็น"},
    ]

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.execute", AsyncMock(return_value=click_result)) as mock_execute, \
         patch("backend.app.core.orchestrator.llm.append_tool_result", side_effect=lambda m, tid, r: m), \
         patch("backend.app.core.orchestrator.llm.evaluate_semantic_redundancy", AsyncMock(side_effect=evaluator_decisions)) as mock_eval, \
         patch("backend.app.core.orchestrator.llm.next_action", AsyncMock(side_effect=next_action_calls)):
        result = await Orchestrator().run_task("https://example.com", "goal", provider="anthropic")

    assert result["success"] is True
    assert result["steps"] == 1  # action แรกถูกข้าม ไม่นับเป็น step
    assert mock_eval.await_count == 2
    mock_execute.assert_awaited_once()  # dispatch จริงแค่ครั้งเดียว (action ที่ผ่าน PASS)


@pytest.mark.asyncio
async def test_run_task_never_calls_semantic_evaluator_when_flag_disabled():
    """default settings.enable_semantic_redundancy_check=False — ต้องไม่เพิ่ม LLM call
    แปลกใหม่เข้าไปใน loop เดิมเลยถ้าไม่ได้เปิด flag"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    click_result = ActionResult(True, "click(1)", "สำเร็จ")

    next_action_calls = [
        ("browser_action", {"type": "click", "index": 1}, "t0", [], llm.TokenUsage()),
        ("finish_task", {"success": True, "message": "เสร็จ"}, "", [], llm.TokenUsage()),
    ]

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.execute", AsyncMock(return_value=click_result)), \
         patch("backend.app.core.orchestrator.llm.append_tool_result", side_effect=lambda m, tid, r: m), \
         patch("backend.app.core.orchestrator.llm.evaluate_semantic_redundancy", AsyncMock()) as mock_eval, \
         patch("backend.app.core.orchestrator.llm.next_action", AsyncMock(side_effect=next_action_calls)):
        result = await Orchestrator().run_task("https://example.com", "goal", provider="anthropic")

    assert result["success"] is True
    mock_eval.assert_not_awaited()


# ---------------- W19-2: Safety & Performance Middleware wiring ----------------
# llm.evaluate_safety_and_performance() เองมีเทสต์ครบใน test_llm.py แล้ว — กลุ่มนี้เทสต์แค่ว่า
# run_task() เรียกมันจริงตอน settings.enable_middleware_evaluator เปิดอยู่, ใช้มันแทน (ไม่ใช่
# เพิ่มเข้าไปคู่กับ) evaluate_semantic_redundancy, skip step ตอน SKIP_REDUNDANT, และ escalate
# permission แบบ "เพิ่มความระมัดระวังเท่านั้น" ผ่าน manual_guidance ที่ execute() เห็น — ไม่เคย
# เรียก ask_user_func เองตรงๆ จากตรงนี้เลย (classify_action() ที่ execute() ยังเป็นคนตัดสิน)


def _middleware_decision(redundant=False, risk="AUTO_APPROVE", decision="EXECUTE", reason=""):
    return {
        "redundancy_evaluation": {"is_redundant": redundant, "redundancy_reason": reason if redundant else ""},
        "permission_evaluation": {"risk_level": risk, "permission_reason": reason if risk != "AUTO_APPROVE" else ""},
        "final_action_decision": decision,
    }


@pytest.mark.asyncio
async def test_run_task_skips_dispatch_when_middleware_says_skip_redundant(monkeypatch):
    from backend.app.config import settings

    monkeypatch.setattr(settings, "enable_middleware_evaluator", True)

    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    click_result = ActionResult(True, "click(2)", "สำเร็จ")

    next_action_calls = [
        ("browser_action", {"type": "click", "index": 1}, "t0", [], llm.TokenUsage()),
        ("browser_action", {"type": "click", "index": 2}, "t1", [], llm.TokenUsage()),
        ("finish_task", {"success": True, "message": "เสร็จ"}, "", [], llm.TokenUsage()),
    ]
    middleware_decisions = [
        _middleware_decision(redundant=True, decision="SKIP_REDUNDANT", reason="ไม่เกี่ยวกับ goal"),
        _middleware_decision(decision="EXECUTE"),
    ]

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.execute", AsyncMock(return_value=click_result)) as mock_execute, \
         patch("backend.app.core.orchestrator.llm.append_tool_result", side_effect=lambda m, tid, r: m), \
         patch("backend.app.core.orchestrator.llm.evaluate_safety_and_performance", AsyncMock(side_effect=middleware_decisions)) as mock_mw, \
         patch("backend.app.core.orchestrator.llm.evaluate_semantic_redundancy", AsyncMock()) as mock_sem, \
         patch("backend.app.core.orchestrator.llm.next_action", AsyncMock(side_effect=next_action_calls)):
        result = await Orchestrator().run_task("https://example.com", "goal", provider="anthropic")

    assert result["success"] is True
    assert result["steps"] == 1  # action แรกถูกข้าม ไม่นับเป็น step
    assert mock_mw.await_count == 2
    mock_execute.assert_awaited_once()  # dispatch จริงแค่ครั้งเดียว
    mock_sem.assert_not_awaited()  # mutually exclusive กับ evaluate_semantic_redundancy


@pytest.mark.asyncio
async def test_run_task_escalates_manual_guidance_when_middleware_requires_consent(monkeypatch):
    from backend.app.config import settings

    monkeypatch.setattr(settings, "enable_middleware_evaluator", True)

    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    click_result = ActionResult(True, "click(1)", "สำเร็จ")

    next_action_calls = [
        ("browser_action", {"type": "click", "index": 1}, "t0", [], llm.TokenUsage()),
        ("finish_task", {"success": True, "message": "เสร็จ"}, "", [], llm.TokenUsage()),
    ]
    middleware_decision = _middleware_decision(
        risk="REQUIRES_CONSENT", decision="PROMPT_USER_PERMISSION", reason="deletes user data",
    )

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.execute", AsyncMock(return_value=click_result)) as mock_execute, \
         patch("backend.app.core.orchestrator.llm.append_tool_result", side_effect=lambda m, tid, r: m), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=[]), \
         patch("backend.app.core.orchestrator.llm.evaluate_safety_and_performance", AsyncMock(return_value=middleware_decision)), \
         patch("backend.app.core.orchestrator.llm.next_action", AsyncMock(side_effect=next_action_calls)):
        result = await Orchestrator().run_task("https://example.com", "goal", provider="anthropic")

    assert result["success"] is True
    mock_execute.assert_awaited_once()
    # escalate-only: manual_guidance ที่ execute()/classify_action() เห็นต้องมีวลีที่ตรงกับ
    # MANUAL_CONFIRMATION_KEYWORDS แนบเข้าไปจริง (ให้ classify_action() escalate เอง)
    assert "requires confirmation" in mock_execute.call_args.kwargs["manual_guidance"]
    assert "deletes user data" in mock_execute.call_args.kwargs["manual_guidance"]


@pytest.mark.asyncio
async def test_run_task_middleware_does_not_append_guidance_when_auto_approve(monkeypatch):
    """AUTO_APPROVE ต้องไม่แตะ manual_guidance เลย (พฤติกรรมเดิมเป๊ะ ไม่ลดระดับ ไม่เพิ่ม)"""
    from backend.app.config import settings

    monkeypatch.setattr(settings, "enable_middleware_evaluator", True)

    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    click_result = ActionResult(True, "click(1)", "สำเร็จ")

    next_action_calls = [
        ("browser_action", {"type": "click", "index": 1}, "t0", [], llm.TokenUsage()),
        ("finish_task", {"success": True, "message": "เสร็จ"}, "", [], llm.TokenUsage()),
    ]

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.execute", AsyncMock(return_value=click_result)) as mock_execute, \
         patch("backend.app.core.orchestrator.llm.append_tool_result", side_effect=lambda m, tid, r: m), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=[]), \
         patch("backend.app.core.orchestrator.llm.evaluate_safety_and_performance", AsyncMock(return_value=_middleware_decision())), \
         patch("backend.app.core.orchestrator.llm.next_action", AsyncMock(side_effect=next_action_calls)):
        result = await Orchestrator().run_task("https://example.com", "goal", provider="anthropic")

    assert result["success"] is True
    assert mock_execute.call_args.kwargs["manual_guidance"] == ""


@pytest.mark.asyncio
async def test_run_task_never_calls_middleware_evaluator_when_flag_disabled():
    """default settings.enable_middleware_evaluator=False — ไม่แตะ loop เดิมเลย"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    click_result = ActionResult(True, "click(1)", "สำเร็จ")

    next_action_calls = [
        ("browser_action", {"type": "click", "index": 1}, "t0", [], llm.TokenUsage()),
        ("finish_task", {"success": True, "message": "เสร็จ"}, "", [], llm.TokenUsage()),
    ]

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.execute", AsyncMock(return_value=click_result)), \
         patch("backend.app.core.orchestrator.llm.append_tool_result", side_effect=lambda m, tid, r: m), \
         patch("backend.app.core.orchestrator.llm.evaluate_safety_and_performance", AsyncMock()) as mock_mw, \
         patch("backend.app.core.orchestrator.llm.next_action", AsyncMock(side_effect=next_action_calls)):
        result = await Orchestrator().run_task("https://example.com", "goal", provider="anthropic")

    assert result["success"] is True
    mock_mw.assert_not_awaited()


# ---------------- W19-3: Voice & Persona Interface wiring ----------------
# llm.generate_persona_message() เองมีเทสต์ครบใน test_llm.py แล้ว — กลุ่มนี้เทสต์แค่ว่า
# run_task() เรียกมันจริงตอนจบ task (ครั้งเดียว ไม่ใช่ทุก step) เฉพาะตอน
# settings.enable_persona_voice เปิดอยู่, แนบผลลัพธ์เข้า return dict แบบ additive
# (persona_message/persona_status ใหม่ ไม่แตะ message/success เดิม) และยิง SSE event


@pytest.mark.asyncio
async def test_run_task_generates_persona_message_on_completion_when_flag_enabled(monkeypatch):
    from backend.app.config import settings

    monkeypatch.setattr(settings, "enable_persona_voice", True)

    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    click_result = ActionResult(True, "click(1)", "สำเร็จ")
    on_event = AsyncMock()

    next_action_calls = [
        ("browser_action", {"type": "click", "index": 1}, "t0", [], llm.TokenUsage()),
        ("finish_task", {"success": True, "message": "จองคิวสำเร็จ"}, "", [], llm.TokenUsage()),
    ]
    persona_reply = {"user_message": "เรียบร้อยครับ! จองคิวให้เสร็จแล้ว", "action_status": "COMPLETED"}

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.execute", AsyncMock(return_value=click_result)), \
         patch("backend.app.core.orchestrator.llm.append_tool_result", side_effect=lambda m, tid, r: m), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=[]), \
         patch("backend.app.core.orchestrator.llm.generate_persona_message", AsyncMock(return_value=persona_reply)) as mock_persona, \
         patch("backend.app.core.orchestrator.llm.next_action", AsyncMock(side_effect=next_action_calls)):
        result = await Orchestrator().run_task(
            "https://example.com", "goal", provider="anthropic", on_event=on_event, headless=False,
        )

    assert result["success"] is True
    mock_persona.assert_awaited_once()  # ครั้งเดียวตอนจบ task ไม่ใช่ทุก step
    assert result["persona_message"] == "เรียบร้อยครับ! จองคิวให้เสร็จแล้ว"
    assert result["persona_status"] == "COMPLETED"
    assert result["message"] == "จองคิวสำเร็จ"  # raw message เดิมยังอยู่ครบ ไม่ถูกแทนที่

    persona_events = [c.args[0] for c in on_event.await_args_list if c.args[0].get("kind") == "persona_message"]
    assert len(persona_events) == 1
    assert persona_events[0]["user_message"] == "เรียบร้อยครับ! จองคิวให้เสร็จแล้ว"


@pytest.mark.asyncio
async def test_run_task_never_calls_persona_generator_when_flag_disabled():
    """default settings.enable_persona_voice=False — ไม่แตะ loop เดิมเลย"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    click_result = ActionResult(True, "click(1)", "สำเร็จ")

    next_action_calls = [
        ("browser_action", {"type": "click", "index": 1}, "t0", [], llm.TokenUsage()),
        ("finish_task", {"success": True, "message": "เสร็จ"}, "", [], llm.TokenUsage()),
    ]

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.execute", AsyncMock(return_value=click_result)), \
         patch("backend.app.core.orchestrator.llm.append_tool_result", side_effect=lambda m, tid, r: m), \
         patch("backend.app.core.orchestrator.llm.generate_persona_message", AsyncMock()) as mock_persona, \
         patch("backend.app.core.orchestrator.llm.next_action", AsyncMock(side_effect=next_action_calls)):
        result = await Orchestrator().run_task("https://example.com", "goal", provider="anthropic")

    assert result["success"] is True
    mock_persona.assert_not_awaited()
    assert result["persona_message"] == ""
    assert result["persona_status"] == "COMPLETED"


# ---------------- Task4 (W19, "Task Completion Verifier"): _scan_validation_errors() ----------------


def _make_locator_page(items):
    """items: list ของ (is_visible, text) — mock page.locator(selector) แบบ MagicMock
    ธรรมดา (sync, คืน Locator ทันที) ตาม pattern เดียวกับ test_state_filter.py/
    test_actions.py::_make_select_mock_page — ไม่งั้น page.locator() จะได้ coroutine
    กลับมาแทน Locator object จริง"""
    locator = MagicMock()
    locator.count = AsyncMock(return_value=len(items))
    locator.created_items = []

    def _nth(i):
        is_visible, text = items[i]
        item = MagicMock()
        item.is_visible = AsyncMock(return_value=is_visible)
        item.inner_text = AsyncMock(return_value=text)
        locator.created_items.append(item)
        return item

    locator.nth = MagicMock(side_effect=_nth)
    mock_page = MagicMock()
    mock_page.locator = MagicMock(return_value=locator)
    return mock_page


@pytest.mark.asyncio
async def test_scan_validation_errors_returns_visible_error_texts():
    mock_page = _make_locator_page([(True, "Required"), (True, "Username already exists")])

    result = await _scan_validation_errors(mock_page)

    assert result == ["Required", "Username already exists"]


@pytest.mark.asyncio
async def test_scan_validation_errors_uses_short_timeout_not_playwright_default_30s():
    """W19 (latency): ต้องระบุ timeout สั้นๆ ให้ is_visible()/inner_text() ตรงๆ ไม่ปล่อยให้
    Playwright ใช้ default 30000ms"""
    mock_page = _make_locator_page([(True, "Required")])

    await _scan_validation_errors(mock_page)

    item = mock_page.locator.return_value.created_items[0]
    assert item.is_visible.await_args.kwargs["timeout"] == orchestrator_module._DOM_CHECK_TIMEOUT_MS
    assert item.inner_text.await_args.kwargs["timeout"] == orchestrator_module._DOM_CHECK_TIMEOUT_MS
    assert orchestrator_module._DOM_CHECK_TIMEOUT_MS <= 5000


@pytest.mark.asyncio
async def test_scan_validation_errors_skips_invisible_elements():
    """framework หลายตัว render error placeholder ไว้เสมอแต่ซ่อนอยู่ตอนไม่มี error จริง —
    ต้องไม่นับตัวที่ is_visible()=False"""
    mock_page = _make_locator_page([(False, "Required"), (True, "Invalid email")])

    result = await _scan_validation_errors(mock_page)

    assert result == ["Invalid email"]


@pytest.mark.asyncio
async def test_scan_validation_errors_returns_empty_list_when_none_found():
    mock_page = _make_locator_page([])

    result = await _scan_validation_errors(mock_page)

    assert result == []


@pytest.mark.asyncio
async def test_scan_validation_errors_fails_safe_on_bare_mock_page():
    """page ที่ไม่ได้ config เฉพาะ (bare AsyncMock ทั้งก้อนเหมือนเทสต์ทั่วไปในไฟล์นี้) —
    ต้องคืน [] เงียบๆ ไม่ throw"""
    result = await _scan_validation_errors(AsyncMock())

    assert result == []


def test_validation_error_selector_in_form_prefixes_every_alternative():
    """W20 (Task12 follow-up): _VALIDATION_ERROR_SELECTOR มีหลาย alternative คั่นด้วย comma —
    ต้อง prefix "form " ให้ทุก alternative ไม่ใช่แค่ตัวแรก (ผิดพลาดแบบนี้จะทำให้ selector ท้าย
    ๆ หลุด scope ไปเช็คทั้งหน้าเหมือนเดิมโดยไม่รู้ตัว)"""
    plain_parts = orchestrator_module._VALIDATION_ERROR_SELECTOR.split(",")
    scoped_parts = orchestrator_module._VALIDATION_ERROR_SELECTOR_IN_FORM.split(",")
    assert len(scoped_parts) == len(plain_parts)
    assert all(p.strip().startswith("form ") for p in scoped_parts)


@pytest.mark.asyncio
async def test_scan_validation_errors_within_form_scopes_selector_to_form_descendants():
    """W20 (Task12 follow-up, บั๊กจริงที่พบ live บน opensource-demo.orangehrmlive.com): หน้า
    login มี <div class="orangehrm-login-error">Username : Admin / Password : admin123</div>
    (คำใบ้ demo credentials คงที่ ไม่ใช่ validation error จริง) อยู่นอก <form> เสมอ ไม่ว่าจะทำ
    action อะไรก็ตาม — ชื่อ class ดันมีคำว่า "error" ปนอยู่ ทำให้ [class*="error" i] เดิมแมตช์
    ผิดพลาด within_form=True ต้อง scope selector ที่ส่งเข้า page.locator() ให้อยู่แค่ใน <form>
    เท่านั้น กัน false positive แบบนี้"""
    mock_page = _make_locator_page([])

    await _scan_validation_errors(mock_page, within_form=True)

    called_selector = mock_page.locator.call_args.args[0]
    assert called_selector == orchestrator_module._VALIDATION_ERROR_SELECTOR_IN_FORM


@pytest.mark.asyncio
async def test_scan_validation_errors_default_scope_is_whole_page():
    """within_form ไม่ระบุ (guard เดิมก่อน finish_task) -> ยังคง scope ทั้งหน้าเหมือนเดิมทุก
    ประการ ไม่เปลี่ยนพฤติกรรมเดิม"""
    mock_page = _make_locator_page([])

    await _scan_validation_errors(mock_page)

    called_selector = mock_page.locator.call_args.args[0]
    assert called_selector == orchestrator_module._VALIDATION_ERROR_SELECTOR


@pytest.mark.asyncio
async def test_login_form_needs_password_uses_short_timeout_not_playwright_default_30s():
    """W19 (latency): .input_value() ต้องระบุ timeout สั้นๆ ตรงๆ ไม่ปล่อยให้ Playwright
    ใช้ default 30000ms"""
    password_input = MagicMock()
    password_input.input_value = AsyncMock(return_value="")
    password_locator = MagicMock()
    password_locator.count = AsyncMock(return_value=1)
    password_locator.nth = MagicMock(return_value=password_input)
    mock_page = MagicMock()
    mock_page.locator = MagicMock(return_value=password_locator)

    result = await _login_form_needs_password(mock_page)

    assert result is True
    assert password_input.input_value.await_args.kwargs["timeout"] == orchestrator_module._DOM_CHECK_TIMEOUT_MS
    assert orchestrator_module._DOM_CHECK_TIMEOUT_MS <= 5000


def test_wait_stable_default_timeout_reduced_from_original_8000ms():
    """W19 (latency): networkidle ไม่มีวัน resolve เร็วบนเว็บที่มี analytics/polling ยิง
    ต่อเนื่อง — ลด default timeout ลงจาก 8000ms เดิมเพื่อลด latency สูงสุดที่เสียไปเปล่าๆ
    ต่อ page-changing action หนึ่งครั้ง (ไม่กระทบ correctness เพราะ timeout ไม่เคยทำให้
    wait_stable() fail อยู่แล้ว — ดู actions.py::wait_stable)"""
    import inspect

    from backend.app.core.actions import wait_stable

    default_timeout = inspect.signature(wait_stable).parameters["timeout"].default
    assert default_timeout < 8000
    assert 3000 <= default_timeout <= 5000


# ---------------- Task4: guard wiring ใน run_task() ----------------


@pytest.mark.asyncio
async def test_run_task_rejects_finish_task_true_when_validation_errors_visible():
    """เจอ validation error บนหน้าปัจจุบัน -> ปฏิเสธ finish_task(success=true) เตือนให้แก้
    ก่อน ไม่ยอมรับทันที

    W20 (Task12 follow-up): action แรกใช้ "select" (ไม่ใช่ "fill") ตั้งใจ — หลังจาก fill เปิด
    ให้ trigger hard-stop guard ใหม่ด้วยแล้ว (ดู _should_check_validation_error_after_action)
    ต้องใช้ action type ที่ guard ใหม่ไม่ครอบคลุมเพื่อทดสอบ guard เดิม (ก่อน finish_task) แยก
    จากกันจริงๆ ไม่งั้น hard-stop ใหม่จะดักไปก่อนตั้งแต่ step แรก ไม่ทันถึง finish_task เลย

    W65[2]: ใช้ error message ที่ไม่ fatal ("Should have at least 7 characters" — ไม่ตรง
    _FATAL_VALIDATION_ERROR_KEYWORDS) ตั้งใจ เพื่อทดสอบเส้นทาง nudge-retry ทั่วไปแยกจากเส้นทาง
    fatal-short-circuit ใหม่ (เดิมใช้ "Employee Name already exists" ซึ่งตอนนี้ถูกจัดเป็น
    fatal แล้ว ย้ายไปทดสอบที่ test_run_task_short_circuits_finish_task_on_fatal_validation_
    error แทน)"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    fill_result = ActionResult(True, "select(1)", "เลือกสำเร็จ")

    next_action_calls = [
        ("browser_action", {"type": "select", "index": 1, "label": "x"}, "t0", [], llm.TokenUsage()),
        ("finish_task", {"success": True, "message": "บันทึกสำเร็จ"}, "t1", [], llm.TokenUsage()),
        ("finish_task", {"success": True, "message": "บันทึกสำเร็จจริงๆ"}, "t2", [], llm.TokenUsage()),
    ]

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=[]), \
         patch("backend.app.core.orchestrator.execute", AsyncMock(return_value=fill_result)), \
         patch("backend.app.core.orchestrator.llm.append_tool_result", side_effect=lambda m, tid, r: m), \
         patch(
             "backend.app.core.orchestrator._scan_validation_errors",
             AsyncMock(side_effect=[["Should have at least 7 characters"], []]),
         ) as mock_scan, \
         patch("backend.app.core.orchestrator.llm.next_action", AsyncMock(side_effect=next_action_calls)) as mock_next_action:
        result = await Orchestrator().run_task("https://example.com", "goal", provider="anthropic")

    assert result["success"] is True
    assert result["message"] == "บันทึกสำเร็จจริงๆ"
    assert mock_scan.await_count == 2
    assert mock_next_action.await_count == 3  # ถูกปฏิเสธ 1 ครั้ง -> ลองใหม่อีกรอบ
    assert result["completion_verification"] == "OK"


@pytest.mark.asyncio
async def test_run_task_accepts_finish_task_after_retries_exhausted_and_tags_result():
    """retry ครบโควตาแล้วยัง error ค้างอยู่ -> ยอมรับตามที่โมเดลยืนยัน (escape valve) แต่
    tag completion_verification เป็น EXECUTION_FAILED_NEEDS_REPAIR"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    fill_result = ActionResult(True, "fill(1)", "กรอกสำเร็จ")

    next_action_calls = [
        ("browser_action", {"type": "fill", "index": 1, "text": "x"}, "t0", [], llm.TokenUsage()),
    ] + [
        ("finish_task", {"success": True, "message": "บันทึกสำเร็จ"}, f"t{i}", [], llm.TokenUsage())
        for i in range(1, 2 + _MAX_PREMATURE_VALIDATION_ERROR_RETRIES)
    ]

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=[]), \
         patch("backend.app.core.orchestrator.execute", AsyncMock(return_value=fill_result)), \
         patch("backend.app.core.orchestrator.llm.append_tool_result", side_effect=lambda m, tid, r: m), \
         patch(
             "backend.app.core.orchestrator._scan_validation_errors",
             AsyncMock(return_value=["Required"]),
         ), \
         patch("backend.app.core.orchestrator.llm.next_action", AsyncMock(side_effect=next_action_calls)):
        result = await Orchestrator().run_task("https://example.com", "goal", provider="anthropic")

    assert result["success"] is True  # ยอมรับตามที่โมเดลยืนยัน (escape valve)
    assert result["completion_verification"] == "EXECUTION_FAILED_NEEDS_REPAIR"


# ---------------- W65[2] ("Error Passthrough" — fatal-class short-circuit) ----------------


def test_is_fatal_validation_error_matches_unfixable_error_classes():
    assert _is_fatal_validation_error("Invalid Credentials") is True
    assert _is_fatal_validation_error("Employee Name already exists") is True
    assert _is_fatal_validation_error("ชื่อผู้ใช้นี้มีอยู่แล้ว") is True
    assert _is_fatal_validation_error("You are not authorized to perform this action") is True
    assert _is_fatal_validation_error("Required") is False
    assert _is_fatal_validation_error("Should have at least 7 characters") is False
    assert _is_fatal_validation_error("") is False
    assert _is_fatal_validation_error(None) is False


@pytest.mark.asyncio
async def test_run_task_short_circuits_finish_task_on_fatal_validation_error():
    """W65[2] (บั๊กที่ต้องการแก้: guard เดิมวน nudge/retry ให้ LLM ลองใหม่ก่อนเสมอ แม้ error
    จะเป็นประเภทที่ retry ไปก็ไม่มีทางหาย เช่น login ผิด) — fatal error ต้องข้าม retry loop
    ไปเลย บังคับ success=False + ข้อความจริงทันที ไม่ต้องรอ finish_task ครั้งที่ 2"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    fill_result = ActionResult(True, "select(1)", "เลือกสำเร็จ")

    next_action_calls = [
        ("browser_action", {"type": "select", "index": 1, "label": "x"}, "t0", [], llm.TokenUsage()),
        ("finish_task", {"success": True, "message": "เข้าสู่ระบบสำเร็จ"}, "t1", [], llm.TokenUsage()),
    ]

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=[]), \
         patch("backend.app.core.orchestrator.execute", AsyncMock(return_value=fill_result)), \
         patch("backend.app.core.orchestrator.llm.append_tool_result", side_effect=lambda m, tid, r: m), \
         patch(
             "backend.app.core.orchestrator._scan_validation_errors",
             AsyncMock(return_value=["Invalid Credentials"]),
         ) as mock_scan, \
         patch("backend.app.core.orchestrator.llm.next_action", AsyncMock(side_effect=next_action_calls)) as mock_next_action:
        result = await Orchestrator().run_task("https://example.com", "goal", provider="anthropic")

    assert result["success"] is False
    assert "Invalid Credentials" in result["message"]
    assert result["completion_verification"] == "TASK_FAILED_USER_INPUT_ERROR"
    assert mock_scan.await_count == 1  # ไม่ retry ซ้ำเลย
    assert mock_next_action.await_count == 2  # ไม่มี finish_task รอบสอง


@pytest.mark.asyncio
async def test_run_task_still_retries_non_fatal_validation_error_before_finish_task():
    """regression: error ที่ไม่ fatal ยังต้องผ่าน nudge-retry loop เดิมทุกประการ (ไม่ใช่ทุก
    error จะถูก short-circuit) — mirror test_run_task_rejects_finish_task_true_when_
    validation_errors_visible เดิมทุกประการ เพื่อยืนยันว่า W65[2] ไม่กระทบ error ทั่วไป"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    fill_result = ActionResult(True, "select(1)", "เลือกสำเร็จ")

    next_action_calls = [
        ("browser_action", {"type": "select", "index": 1, "label": "x"}, "t0", [], llm.TokenUsage()),
        ("finish_task", {"success": True, "message": "บันทึกสำเร็จ"}, "t1", [], llm.TokenUsage()),
        ("finish_task", {"success": True, "message": "บันทึกสำเร็จจริงๆ"}, "t2", [], llm.TokenUsage()),
    ]

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=[]), \
         patch("backend.app.core.orchestrator.execute", AsyncMock(return_value=fill_result)), \
         patch("backend.app.core.orchestrator.llm.append_tool_result", side_effect=lambda m, tid, r: m), \
         patch(
             "backend.app.core.orchestrator._scan_validation_errors",
             AsyncMock(side_effect=[["Should have at least 7 characters"], []]),
         ), \
         patch("backend.app.core.orchestrator.llm.next_action", AsyncMock(side_effect=next_action_calls)) as mock_next_action:
        result = await Orchestrator().run_task("https://example.com", "goal", provider="anthropic")

    assert result["success"] is True
    assert result["message"] == "บันทึกสำเร็จจริงๆ"
    assert mock_next_action.await_count == 3  # ถูกปฏิเสธ 1 ครั้ง -> ลองใหม่อีกรอบ (retry ปกติ)
    assert result["completion_verification"] == "OK"


# ---------------- W22 ("DOM-Based Post-Action Verification Guardrail") ----------------


def test_is_deletion_intent_goal_matches_thai_and_english_keywords():
    assert _is_deletion_intent_goal("ลบคนที่เป็น Role ESS ออกให้หมด") is True
    assert _is_deletion_intent_goal("delete all users with role ESS") is True
    assert _is_deletion_intent_goal("Remove the inactive candidates") is True
    assert _is_deletion_intent_goal("ล้างข้อมูลผู้สมัครเก่าทิ้ง") is True
    assert _is_deletion_intent_goal("เปลี่ยน Role ของ user ทุกคนเป็น Admin") is False
    assert _is_deletion_intent_goal("") is False
    assert _is_deletion_intent_goal(None) is False


def _make_record_count_locator_page(text):
    """text=None -> ไม่มี element นี้ในหน้าเลย (count=0), text=str -> เจอ element ตัวแรกที่มี
    inner_text ตามนี้"""
    locator = MagicMock()
    first = MagicMock()
    if text is None:
        first.count = AsyncMock(return_value=0)
    else:
        first.count = AsyncMock(return_value=1)
        first.inner_text = AsyncMock(return_value=text)
    locator.first = first
    mock_page = MagicMock()
    mock_page.locator = MagicMock(return_value=locator)
    return mock_page


@pytest.mark.asyncio
async def test_scan_remaining_target_records_parses_nonzero_count():
    mock_page = _make_record_count_locator_page("(3) Records Found")

    result = await _scan_remaining_target_records(mock_page)

    assert result == (3, "(3) Records Found")


@pytest.mark.asyncio
async def test_scan_remaining_target_records_treats_no_records_found_as_zero():
    mock_page = _make_record_count_locator_page("No Records Found")

    result = await _scan_remaining_target_records(mock_page)

    assert result == (0, "No Records Found")


# W68b (บั๊กจริงที่ user รายงานซ้ำหลัง W68: goal edit-all ตรวจจับถูกแล้ว แต่ agent ยัง claim
# "0 รายการ" ทั้งที่ตารางจริงมี 16 แถว — สาเหตุที่สอง: อ่าน DOM ครั้งเดียวทันทีตอน finish_task
# ถูกเรียก ชนกับช่วงที่ AJAX ของปุ่ม Search ยังอัปเดต DOM ไม่เสร็จ) — ยืนยันว่า "0" ถูก
# double-check ก่อนเชื่อ ส่วน >0/None ไม่ต้องรอเพิ่ม (ไม่มี latency cost ในเคสปกติ)
@pytest.mark.asyncio
async def test_scan_remaining_target_records_rechecks_after_stale_zero_reading():
    locator = MagicMock()
    first = MagicMock()
    first.count = AsyncMock(return_value=1)
    first.inner_text = AsyncMock(side_effect=["No Records Found", "(16) Records Found"])
    locator.first = first
    mock_page = MagicMock()
    mock_page.locator = MagicMock(return_value=locator)

    result = await _scan_remaining_target_records(mock_page)

    assert result == (16, "(16) Records Found")  # เชื่อค่าที่อ่านซ้ำรอบสอง (สดกว่า) ไม่ใช่ 0 เดิม
    assert first.inner_text.await_count == 2


@pytest.mark.asyncio
async def test_scan_remaining_target_records_keeps_zero_when_recheck_still_zero():
    locator = MagicMock()
    first = MagicMock()
    first.count = AsyncMock(return_value=1)
    first.inner_text = AsyncMock(side_effect=["No Records Found", "No Records Found"])
    locator.first = first
    mock_page = MagicMock()
    mock_page.locator = MagicMock(return_value=locator)

    result = await _scan_remaining_target_records(mock_page)

    assert result == (0, "No Records Found")  # ยืนยันตรงกันทั้งสองรอบ -> เชื่อว่า 0 จริง
    assert first.inner_text.await_count == 2


@pytest.mark.asyncio
async def test_scan_remaining_target_records_does_not_recheck_nonzero_reading():
    mock_page = _make_record_count_locator_page("(3) Records Found")

    result = await _scan_remaining_target_records(mock_page)

    assert result == (3, "(3) Records Found")
    mock_page.locator.assert_called_once()  # >0 ไม่ต้องอ่านซ้ำ ไม่มี latency เพิ่ม


@pytest.mark.asyncio
async def test_scan_remaining_target_records_returns_none_when_element_absent():
    """หน้าไม่มี "Records Found" UI เลย (ไม่ใช่ตารางแบบ OrangeHRM) -> None กัน guard บล็อก
    finish_task ที่อาจถูกต้องอยู่แล้วบนเว็บที่ไม่รองรับ UI แบบนี้"""
    mock_page = _make_record_count_locator_page(None)

    result = await _scan_remaining_target_records(mock_page)

    assert result is None


@pytest.mark.asyncio
async def test_scan_remaining_target_records_fails_safe_on_bare_mock_page():
    result = await _scan_remaining_target_records(AsyncMock())

    assert result is None


@pytest.mark.asyncio
async def test_run_task_rejects_finish_task_true_when_target_records_still_remain():
    """W22: goal มีคำว่า "ลบ" (deletion-intent) + DOM ยังโชว์ "(3) Records Found" จริง ->
    ปฏิเสธ finish_task(success=true) แม้ LLM จะยืนยันว่าลบครบแล้วก็ตาม (บั๊กจริงที่ user
    รายงาน: agent ตอบ "ไม่พบ/ลบครบแล้ว" ทั้งที่ยังเหลือ 3 รายการ)"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    click_result = ActionResult(True, "delete(1)", "ลบสำเร็จ")

    next_action_calls = [
        ("browser_action", {"type": "delete", "index": 1}, "t0", [], llm.TokenUsage()),
        (
            "finish_task",
            {"success": True, "message": "No users with Role ESS found (or all have been deleted)."},
            "t1", [], llm.TokenUsage(),
        ),
        ("finish_task", {"success": True, "message": "ลบครบจริงๆ แล้ว"}, "t2", [], llm.TokenUsage()),
    ]

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=[]), \
         patch("backend.app.core.orchestrator.execute", AsyncMock(return_value=click_result)), \
         patch("backend.app.core.orchestrator.llm.append_tool_result", side_effect=lambda m, tid, r: m), \
         patch(
             "backend.app.core.orchestrator._scan_remaining_target_records",
             AsyncMock(side_effect=[(3, "(3) Records Found"), (0, "No Records Found")]),
         ) as mock_scan, \
         patch("backend.app.core.orchestrator.llm.next_action", AsyncMock(side_effect=next_action_calls)) as mock_next_action:
        result = await Orchestrator().run_task(
            "https://example.com", "ลบคนที่เป็น Role ESS ออกให้หมด", provider="anthropic",
        )

    assert result["success"] is True
    assert result["message"] == "ลบครบจริงๆ แล้ว"
    assert mock_scan.await_count == 2
    assert mock_next_action.await_count == 3  # ถูกปฏิเสธ 1 ครั้ง -> ลองใหม่อีกรอบ
    assert result["completion_verification"] == "OK"


@pytest.mark.asyncio
async def test_run_task_forces_truthful_failure_when_deletion_retries_exhausted():
    """W22 ("TRUTH-BASED RESPONSE GENERATION"): retry ครบโควตาแล้วยังเหลือ record จริงใน DOM
    -> ต่างจาก validation-error guard (ที่ปล่อยผ่านตามคำยืนยันของโมเดล) guard นี้ต้อง "บังคับ
    ความจริง" ลงผลลัพธ์สุดท้ายเสมอ: success ต้องเป็น False และข้อความต้องรายงานจำนวนที่เหลือ
    จริง ไม่ใช่คำอธิบาย hallucinate ของ LLM"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    click_result = ActionResult(True, "delete(1)", "ลบสำเร็จ")

    next_action_calls = [
        ("browser_action", {"type": "delete", "index": 1}, "t0", [], llm.TokenUsage()),
    ] + [
        (
            "finish_task",
            {"success": True, "message": "No users with Role ESS found (or all have been deleted)."},
            f"t{i}", [], llm.TokenUsage(),
        )
        for i in range(1, 2 + _MAX_PREMATURE_DELETION_INCOMPLETE_RETRIES)
    ]

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=[]), \
         patch("backend.app.core.orchestrator.execute", AsyncMock(return_value=click_result)), \
         patch("backend.app.core.orchestrator.llm.append_tool_result", side_effect=lambda m, tid, r: m), \
         patch(
             "backend.app.core.orchestrator._scan_remaining_target_records",
             AsyncMock(return_value=(3, "(3) Records Found")),
         ), \
         patch("backend.app.core.orchestrator.llm.next_action", AsyncMock(side_effect=next_action_calls)):
        result = await Orchestrator().run_task(
            "https://example.com", "ลบคนที่เป็น Role ESS ออกให้หมด", provider="anthropic",
        )

    assert result["success"] is False
    assert "3" in result["message"]
    assert "No users with Role ESS found" not in result["message"]
    assert result["completion_verification"] == "EXECUTION_FAILED_NEEDS_REPAIR"


@pytest.mark.asyncio
async def test_run_task_skips_deletion_guard_for_non_deletion_goal():
    """goal ไม่มีคำว่า "ลบ"/"delete" เลย -> ไม่เรียก _scan_remaining_target_records() เลยแม้
    หน้าเว็บจะมี "Records Found" UI ก็ตาม (กัน noise เปล่าๆ กับ goal ที่ไม่เกี่ยวกับการลบ)"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    click_result = ActionResult(True, "click(1)", "คลิกสำเร็จ")

    next_action_calls = [
        ("browser_action", {"type": "click", "index": 1}, "t0", [], llm.TokenUsage()),
        ("finish_task", {"success": True, "message": "เสร็จแล้ว"}, "t1", [], llm.TokenUsage()),
    ]

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=[]), \
         patch("backend.app.core.orchestrator.execute", AsyncMock(return_value=click_result)), \
         patch("backend.app.core.orchestrator.llm.append_tool_result", side_effect=lambda m, tid, r: m), \
         patch(
             "backend.app.core.orchestrator._scan_remaining_target_records",
             AsyncMock(return_value=(3, "(3) Records Found")),
         ) as mock_scan, \
         patch("backend.app.core.orchestrator.llm.next_action", AsyncMock(side_effect=next_action_calls)):
        result = await Orchestrator().run_task(
            "https://example.com", "ไปหน้า Dashboard แล้วบอกจำนวนพนักงาน", provider="anthropic",
        )

    assert result["success"] is True
    assert mock_scan.await_count == 0
    assert result["completion_verification"] == "OK"


# --- W63[7.2] ("Strict Table Assertion & Truth Reporting", ticket Issue 7.2) ---


def _make_table_body_locator_page(text):
    """text=None -> ไม่มี table body element เลยในหน้า (count=0), text=str -> เจอ table body
    ตัวแรกที่มี inner_text ตามนี้ (mirror _make_record_count_locator_page ด้านบน)"""
    locator = MagicMock()
    first = MagicMock()
    if text is None:
        first.count = AsyncMock(return_value=0)
    else:
        first.count = AsyncMock(return_value=1)
        first.inner_text = AsyncMock(return_value=text)
    locator.first = first
    mock_page = MagicMock()
    mock_page.locator = MagicMock(return_value=locator)
    return mock_page


@pytest.mark.asyncio
async def test_scan_created_item_in_table_true_when_text_present():
    mock_page = _make_table_body_locator_page("Admin | AutoUser_99 | ESS | Enabled")

    result = await _scan_created_item_in_table(mock_page, "AutoUser_99")

    assert result is True


@pytest.mark.asyncio
async def test_scan_created_item_in_table_case_insensitive():
    mock_page = _make_table_body_locator_page("admin | autouser_99 | ess")

    result = await _scan_created_item_in_table(mock_page, "AutoUser_99")

    assert result is True


@pytest.mark.asyncio
async def test_scan_created_item_in_table_false_when_text_absent():
    mock_page = _make_table_body_locator_page("Admin | SomeoneElse | ESS")

    result = await _scan_created_item_in_table(mock_page, "AutoUser_99")

    assert result is False


@pytest.mark.asyncio
async def test_scan_created_item_in_table_false_when_table_body_empty():
    """table body มีอยู่จริงแต่ไม่มีแถวเลย (เช่น "No Records Found") — ถือเป็นหลักฐานว่ายังไม่
    พบรายการนี้จริง ไม่ใช่แค่ "เช็คไม่ได้" (ต่างจากตอนไม่มี table body element เลย)"""
    mock_page = _make_table_body_locator_page("")

    result = await _scan_created_item_in_table(mock_page, "AutoUser_99")

    assert result is False


@pytest.mark.asyncio
async def test_scan_created_item_in_table_returns_true_when_no_table_body_on_page():
    """หน้าไม่มี table body element เลย (ไม่ใช่หน้าตาราง) -> True กัน guard บล็อก finish_task
    ที่อาจถูกต้องอยู่แล้วบนหน้าที่ไม่มีตารางแบบนี้จริง"""
    mock_page = _make_table_body_locator_page(None)

    result = await _scan_created_item_in_table(mock_page, "AutoUser_99")

    assert result is True


@pytest.mark.asyncio
async def test_scan_created_item_in_table_fails_safe_on_bare_mock_page():
    result = await _scan_created_item_in_table(AsyncMock(), "AutoUser_99")

    assert result is True


@pytest.mark.asyncio
async def test_run_task_rejects_finish_task_true_when_verify_text_not_in_table():
    """LLM ระบุ verify_text มาแต่ table body จริงไม่มีข้อความนั้น -> ปฏิเสธ
    finish_task(success=true) แม้ LLM จะยืนยันว่าสร้างสำเร็จแล้วก็ตาม"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    click_result = ActionResult(True, "click(1)", "คลิกสำเร็จ")

    next_action_calls = [
        ("browser_action", {"type": "click", "index": 1}, "t0", [], llm.TokenUsage()),
        (
            "finish_task",
            {"success": True, "message": "สร้างสำเร็จแล้ว", "verify_text": "AutoUser_99"},
            "t1", [], llm.TokenUsage(),
        ),
        (
            "finish_task",
            {"success": True, "message": "สร้างสำเร็จจริงๆ แล้ว", "verify_text": "AutoUser_99"},
            "t2", [], llm.TokenUsage(),
        ),
    ]

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=[]), \
         patch("backend.app.core.orchestrator.execute", AsyncMock(return_value=click_result)), \
         patch("backend.app.core.orchestrator.llm.append_tool_result", side_effect=lambda m, tid, r: m), \
         patch(
             "backend.app.core.orchestrator._scan_created_item_in_table",
             AsyncMock(side_effect=[False, True]),
         ) as mock_scan, \
         patch("backend.app.core.orchestrator.llm.next_action", AsyncMock(side_effect=next_action_calls)) as mock_next_action:
        result = await Orchestrator().run_task(
            "https://example.com", "สร้าง user ใหม่ชื่อ AutoUser_99", provider="anthropic",
        )

    assert result["success"] is True
    assert result["message"] == "สร้างสำเร็จจริงๆ แล้ว"
    assert mock_scan.await_count == 2
    assert mock_next_action.await_count == 3  # ถูกปฏิเสธ 1 ครั้ง -> ลองใหม่อีกรอบ
    assert result["completion_verification"] == "OK"


@pytest.mark.asyncio
async def test_run_task_forces_verification_failed_when_table_verify_retries_exhausted():
    """W63[7.2] ("TRUTH-BASED RESPONSE GENERATION" เหมือน deletion guard): retry ครบโควตาแล้ว
    ยังไม่พบ verify_text ในตารางจริง -> บังคับ success=False และข้อความต้องมี
    "VERIFICATION_FAILED" ตรงตามสเปคเป๊ะ ไม่ใช่คำอธิบาย hallucinate ของ LLM"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    click_result = ActionResult(True, "click(1)", "คลิกสำเร็จ")

    next_action_calls = [
        ("browser_action", {"type": "click", "index": 1}, "t0", [], llm.TokenUsage()),
    ] + [
        (
            "finish_task",
            {"success": True, "message": "สร้างสำเร็จแล้ว", "verify_text": "AutoUser_99"},
            f"t{i}", [], llm.TokenUsage(),
        )
        for i in range(1, 2 + _MAX_PREMATURE_TABLE_VERIFY_RETRIES)
    ]

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=[]), \
         patch("backend.app.core.orchestrator.execute", AsyncMock(return_value=click_result)), \
         patch("backend.app.core.orchestrator.llm.append_tool_result", side_effect=lambda m, tid, r: m), \
         patch("backend.app.core.orchestrator._scan_created_item_in_table", AsyncMock(return_value=False)), \
         patch("backend.app.core.orchestrator.llm.next_action", AsyncMock(side_effect=next_action_calls)):
        result = await Orchestrator().run_task(
            "https://example.com", "สร้าง user ใหม่ชื่อ AutoUser_99", provider="anthropic",
        )

    assert result["success"] is False
    assert "VERIFICATION_FAILED: Item not found in results table." in result["message"]
    assert result["completion_verification"] == "EXECUTION_FAILED_NEEDS_REPAIR"


@pytest.mark.asyncio
async def test_run_task_skips_table_verify_guard_when_verify_text_empty():
    """LLM ไม่ได้ระบุ verify_text มา (goal ไม่เกี่ยวกับการยืนยันรายการในตาราง) -> ไม่เรียก
    _scan_created_item_in_table() เลย"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    click_result = ActionResult(True, "click(1)", "คลิกสำเร็จ")

    next_action_calls = [
        ("browser_action", {"type": "click", "index": 1}, "t0", [], llm.TokenUsage()),
        ("finish_task", {"success": True, "message": "เสร็จแล้ว"}, "t1", [], llm.TokenUsage()),
    ]

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=[]), \
         patch("backend.app.core.orchestrator.execute", AsyncMock(return_value=click_result)), \
         patch("backend.app.core.orchestrator.llm.append_tool_result", side_effect=lambda m, tid, r: m), \
         patch(
             "backend.app.core.orchestrator._scan_created_item_in_table", AsyncMock(return_value=False),
         ) as mock_scan, \
         patch("backend.app.core.orchestrator.llm.next_action", AsyncMock(side_effect=next_action_calls)):
        result = await Orchestrator().run_task(
            "https://example.com", "ไปหน้า Dashboard แล้วบอกจำนวนพนักงาน", provider="anthropic",
        )

    assert result["success"] is True
    assert mock_scan.await_count == 0
    assert result["completion_verification"] == "OK"


# --- W64[7.1] ("Filter Order & False Completion", ticket Issue 7.1) ---


def test_is_edit_all_intent_goal_matches_keywords():
    assert _is_edit_all_intent_goal("เปลี่ยนทุก Role ของ user ที่เป็น ESS เป็น Admin") is True
    assert _is_edit_all_intent_goal("update every employee's status to active") is True
    assert _is_edit_all_intent_goal("edit all ESS users to Admin") is True
    assert _is_edit_all_intent_goal("ลบคนที่เป็น Role ESS ออกให้หมด") is False
    assert _is_edit_all_intent_goal("") is False
    assert _is_edit_all_intent_goal(None) is False


# W68 (บั๊กจริงที่ user รายงาน — ดู roadmap.txt): keyword เดิมด้านบนต้องเจอ "เปลี่ยนทุก"/
# "แก้ทุก" ติดกันเป๊ะเท่านั้น แต่คำพูดธรรมชาติจริงมักแยกคำ "เปลี่ยน...ทุกคน..." ห่างกันด้วยคำ
# อื่น (เช่น goal จริงที่ user พิมพ์: "...เปลี่ยน Role ของทุกคนในผลการค้นหาให้เป็น Admin...")
# ทำให้ exact-phrase match พลาด เลยไม่เปิด _scan_remaining_target_records() guard ปล่อยให้
# agent ตอบ "ไม่พบผู้ใช้ Role ESS (0 รายการ)" หลุดผ่านไปทั้งที่ตารางจริงยังมี user เหลืออยู่ —
# เทียบกับ test_is_deletion_intent_goal_matches_thai_and_english_keywords() บรรทัดที่ยืนยันว่า
# ประโยคเดียวกันนี้ *ไม่ใช่* deletion-intent (ถูกต้อง) แต่ไม่มีใครเทสต์ฝั่ง edit-all-intent เลย
# จนพบบั๊กจริงจาก production
def test_is_edit_all_intent_goal_matches_natural_phrasing_with_words_between_verb_and_bulk_marker():
    assert _is_edit_all_intent_goal(
        "ไปที่หน้า Admin ค้นหา user ทุกคนที่ไม่ใช่ Admin แล้วทำการเปลี่ยน Role "
        "ของทุกคนในผลการค้นหาให้เป็น Admin ให้หมด"
    ) is True
    assert _is_edit_all_intent_goal("เปลี่ยน Role ของ user ทุกคนเป็น Admin") is True
    assert _is_edit_all_intent_goal("แก้ไขสถานะพนักงานทั้งหมดให้เป็น Active") is True
    # มี verb แต่ไม่มี bulk marker เลย -> ไม่ใช่ edit-all (แก้แค่คนเดียว ไม่ต้องเปิด guard)
    assert _is_edit_all_intent_goal("เปลี่ยน Role ของ John เป็น Admin") is False


@pytest.mark.asyncio
async def test_run_task_rejects_finish_task_true_when_edit_all_target_records_still_remain():
    """W64[7.1]: guard เดียวกับ deletion (W22) แต่ขยายรวม edit-all-intent goal ด้วย — ตาราง
    ที่กรองแล้วยังโชว์ "(1) Records Found" (แถว ESS ที่ยังไม่ถูกแก้ Role) -> ปฏิเสธ
    finish_task(success=true) แม้ LLM จะยืนยันว่าแก้ไขครบแล้วก็ตาม"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    click_result = ActionResult(True, "click(1)", "คลิกสำเร็จ")

    next_action_calls = [
        ("browser_action", {"type": "click", "index": 1}, "t0", [], llm.TokenUsage()),
        (
            "finish_task",
            {"success": True, "message": "เปลี่ยน Role ครบทุกคนแล้ว"},
            "t1", [], llm.TokenUsage(),
        ),
        (
            "finish_task",
            {"success": True, "message": "เปลี่ยนครบจริงๆ แล้ว"},
            "t2", [], llm.TokenUsage(),
        ),
    ]

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=[]), \
         patch("backend.app.core.orchestrator.execute", AsyncMock(return_value=click_result)), \
         patch("backend.app.core.orchestrator.llm.append_tool_result", side_effect=lambda m, tid, r: m), \
         patch(
             "backend.app.core.orchestrator._scan_remaining_target_records",
             AsyncMock(side_effect=[(1, "(1) Records Found"), (0, "No Records Found")]),
         ) as mock_scan, \
         patch("backend.app.core.orchestrator.llm.next_action", AsyncMock(side_effect=next_action_calls)) as mock_next_action:
        result = await Orchestrator().run_task(
            "https://example.com", "เปลี่ยนทุก Role ของ user ที่เป็น ESS เป็น Admin", provider="anthropic",
        )

    assert result["success"] is True
    assert result["message"] == "เปลี่ยนครบจริงๆ แล้ว"
    assert mock_scan.await_count == 2
    assert mock_next_action.await_count == 3  # ถูกปฏิเสธ 1 ครั้ง -> ลองใหม่อีกรอบ
    assert result["completion_verification"] == "OK"


@pytest.mark.asyncio
async def test_run_task_forces_truthful_failure_for_edit_all_uses_edit_wording():
    """W64[7.1]: ต่างจาก deletion guard เดิม (พูดว่า "ยังไม่ได้ถูกลบออก") — edit-all-intent
    ต้องใช้ข้อความ "ยังไม่ได้ถูกแก้ไขค่า" แทน ไม่งั้นข้อความจะผิดความจริง (ไม่มีอะไรถูกลบเลย)"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    click_result = ActionResult(True, "click(1)", "คลิกสำเร็จ")

    next_action_calls = [
        ("browser_action", {"type": "click", "index": 1}, "t0", [], llm.TokenUsage()),
    ] + [
        (
            "finish_task",
            {"success": True, "message": "เปลี่ยนครบทุกคนแล้ว"},
            f"t{i}", [], llm.TokenUsage(),
        )
        for i in range(1, 2 + _MAX_PREMATURE_DELETION_INCOMPLETE_RETRIES)
    ]

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=[]), \
         patch("backend.app.core.orchestrator.execute", AsyncMock(return_value=click_result)), \
         patch("backend.app.core.orchestrator.llm.append_tool_result", side_effect=lambda m, tid, r: m), \
         patch(
             "backend.app.core.orchestrator._scan_remaining_target_records",
             AsyncMock(return_value=(1, "(1) Records Found")),
         ), \
         patch("backend.app.core.orchestrator.llm.next_action", AsyncMock(side_effect=next_action_calls)):
        result = await Orchestrator().run_task(
            "https://example.com", "เปลี่ยนทุก Role ของ user ที่เป็น ESS เป็น Admin", provider="anthropic",
        )

    assert result["success"] is False
    assert "ยังไม่ได้ถูกแก้ไขค่า" in result["message"]
    assert "ยังไม่ได้ถูกลบออก" not in result["message"]
    assert result["completion_verification"] == "EXECUTION_FAILED_NEEDS_REPAIR"


def _make_row_action_elements():
    return [
        {"index": 1, "label": "Role", "tag": "select", "type": ""},
        {"index": 2, "label": "Edit", "tag": "button", "type": ""},
        {"index": 3, "label": "Search", "tag": "button", "type": ""},
    ]


@pytest.mark.asyncio
async def test_run_task_blocks_row_action_click_immediately_after_unconfirmed_filter_change():
    """W64[7.1] (บั๊กจริง: agent เลือก Role=ESS ใน dropdown filter แล้วคลิก Edit บนตารางทันที
    ก่อนกด Search — แถวที่ Edit จึงเป็นแถวเก่าก่อนกรอง ไม่ใช่แถวที่ตรงเงื่อนไขจริง): click ปุ่ม
    row-action (Edit) ทันทีหลัง select ที่สำเร็จ (ยังไม่กด Search) ต้องถูกปฏิเสธ/นัดจ์ก่อน —
    หลังกด Search แล้ว คลิก Edit ตัวเดิมต้องผ่านได้ปกติ"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    action_result = ActionResult(True, "action", "สำเร็จ")
    elements = _make_row_action_elements()

    next_action_calls = [
        ("browser_action", {"type": "select", "index": 1, "label": "ESS"}, "t0", [], llm.TokenUsage()),
        ("browser_action", {"type": "click", "index": 2}, "t1", [], llm.TokenUsage()),  # ถูกบล็อก
        ("browser_action", {"type": "click", "index": 3}, "t2", [], llm.TokenUsage()),  # Search
        ("browser_action", {"type": "click", "index": 2}, "t3", [], llm.TokenUsage()),  # Edit ผ่านแล้ว
        ("finish_task", {"success": True, "message": "แก้ไขสำเร็จ"}, "t4", [], llm.TokenUsage()),
    ]

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=(elements, "page"))), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=[]), \
         patch("backend.app.core.orchestrator.execute", AsyncMock(return_value=action_result)) as mock_execute, \
         patch("backend.app.core.orchestrator.llm.append_tool_result", side_effect=lambda m, tid, r: m), \
         patch("backend.app.core.orchestrator.llm.next_action", AsyncMock(side_effect=next_action_calls)) as mock_next_action:
        result = await Orchestrator().run_task(
            "https://example.com", "แก้ไข Role ของ user ที่เป็น ESS เป็น Admin", provider="anthropic",
        )

    assert result["success"] is True
    assert mock_next_action.await_count == 5
    # select(1) + click Search(1) + click Edit ที่ผ่านแล้ว(1) = 3 — ตัวที่ถูกบล็อกไม่ถึง execute() เลย
    assert mock_execute.await_count == 3


@pytest.mark.asyncio
async def test_run_task_allows_row_action_click_without_preceding_fill_or_select():
    """งาน edit แถวเดียวธรรมดาที่ไม่มีการกรอง filter อะไรเลยตั้งแต่ต้น (คลิก Edit ทันทีเป็น
    action แรก) ต้องไม่ถูกบล็อก — filter_dirty_since_search เริ่มต้นเป็น False เสมอ"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    action_result = ActionResult(True, "action", "สำเร็จ")
    elements = _make_row_action_elements()

    next_action_calls = [
        ("browser_action", {"type": "click", "index": 2}, "t0", [], llm.TokenUsage()),
        ("finish_task", {"success": True, "message": "แก้ไขสำเร็จ"}, "t1", [], llm.TokenUsage()),
    ]

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=(elements, "page"))), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=[]), \
         patch("backend.app.core.orchestrator.execute", AsyncMock(return_value=action_result)) as mock_execute, \
         patch("backend.app.core.orchestrator.llm.append_tool_result", side_effect=lambda m, tid, r: m), \
         patch("backend.app.core.orchestrator.llm.next_action", AsyncMock(side_effect=next_action_calls)) as mock_next_action:
        result = await Orchestrator().run_task(
            "https://example.com", "แก้ไขพนักงานคนแรกในตาราง", provider="anthropic",
        )

    assert result["success"] is True
    assert mock_next_action.await_count == 2
    assert mock_execute.await_count == 1


# --- W64[7.2] ("Add-Action Idempotency Lock", ticket Issue 7.2) ---


@pytest.mark.asyncio
async def test_run_task_uses_qualified_message_when_toast_confirmed_but_verify_text_not_found():
    """W64[7.2] (บั๊กจริง: agent บันทึกพนักงานใหม่สำเร็จจริง (มี toast ยืนยัน) แต่ค้นหาเพื่อ
    verify แล้วไม่เจอเพราะ AJAX table ยังโหลดไม่เสร็จ): ต่างจาก guard เดิม (W63[7.2], ไม่มี
    หลักฐาน toast เลย) — มี toast_confirmed=True มาก่อนหน้าใน task เดียวกัน -> ไม่ force
    success=False, ใช้ข้อความ "บันทึกข้อมูลเรียบร้อยแล้ว แต่ไม่พบรายการในตารางการค้นหา" แทน
    VERIFICATION_FAILED"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    save_result = ActionResult(
        True, "click(1)", 'บันทึกสำเร็จ [พบข้อความยืนยันสำเร็จ: "Successfully Saved"]',
        toast_confirmed=True,
    )

    next_action_calls = [
        ("browser_action", {"type": "submit", "index": 1}, "t0", [], llm.TokenUsage()),
    ] + [
        (
            "finish_task",
            {
                "success": True, "message": "บันทึกพนักงานใหม่สำเร็จ",
                "verify_text": "Siamyut Phasida",
            },
            f"t{i}", [], llm.TokenUsage(),
        )
        for i in range(1, 2 + _MAX_PREMATURE_TABLE_VERIFY_RETRIES)
    ]

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=[]), \
         patch("backend.app.core.orchestrator.execute", AsyncMock(return_value=save_result)), \
         patch("backend.app.core.orchestrator.llm.append_tool_result", side_effect=lambda m, tid, r: m), \
         patch("backend.app.core.orchestrator._scan_created_item_in_table", AsyncMock(return_value=False)), \
         patch("backend.app.core.orchestrator.llm.next_action", AsyncMock(side_effect=next_action_calls)):
        result = await Orchestrator().run_task(
            "https://example.com", "เพิ่มพนักงานใหม่ชื่อ Siamyut Phasida", provider="anthropic",
        )

    assert result["success"] is True
    assert result["message"] == "บันทึกข้อมูลเรียบร้อยแล้ว แต่ไม่พบรายการในตารางการค้นหา"
    assert result["completion_verification"] == "OK_SAVE_CONFIRMED_NOT_IN_TABLE"


@pytest.mark.asyncio
async def test_run_task_keeps_verification_failed_when_no_toast_confirmed():
    """regression: ไม่มี action ไหน toast_confirmed=True เลยใน task นี้ -> guard เดิม
    (W63[7.2]) ต้องทำงานเหมือนเดิมทุกประการ (force success=False, ข้อความ
    VERIFICATION_FAILED)"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    click_result = ActionResult(True, "click(1)", "คลิกสำเร็จ")  # toast_confirmed default False

    next_action_calls = [
        ("browser_action", {"type": "click", "index": 1}, "t0", [], llm.TokenUsage()),
    ] + [
        (
            "finish_task",
            {"success": True, "message": "สร้างสำเร็จแล้ว", "verify_text": "AutoUser_99"},
            f"t{i}", [], llm.TokenUsage(),
        )
        for i in range(1, 2 + _MAX_PREMATURE_TABLE_VERIFY_RETRIES)
    ]

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=[]), \
         patch("backend.app.core.orchestrator.execute", AsyncMock(return_value=click_result)), \
         patch("backend.app.core.orchestrator.llm.append_tool_result", side_effect=lambda m, tid, r: m), \
         patch("backend.app.core.orchestrator._scan_created_item_in_table", AsyncMock(return_value=False)), \
         patch("backend.app.core.orchestrator.llm.next_action", AsyncMock(side_effect=next_action_calls)):
        result = await Orchestrator().run_task(
            "https://example.com", "สร้าง user ใหม่ชื่อ AutoUser_99", provider="anthropic",
        )

    assert result["success"] is False
    assert "VERIFICATION_FAILED: Item not found in results table." in result["message"]
    assert result["completion_verification"] == "EXECUTION_FAILED_NEEDS_REPAIR"


# --- W20 (Task12, "UI Validation Error Detection" — Early Termination Guardrail): hard-stop
# ทันทีหลังคลิกปุ่ม Save/Submit ที่เผยข้อความ validation error — บั๊กจริงที่ user รายงาน
# (agent ไม่เคยเรียก finish_task เลย แค่วน refresh/retry ไม่จบ ทำให้ guard เดิม (ก่อน
# finish_task) ไม่มีทางทำงาน) ---


def test_label_looks_like_form_submit_matches_save_and_thai_keywords():
    assert _label_looks_like_form_submit("Save") is True
    assert _label_looks_like_form_submit("บันทึก") is True
    assert _label_looks_like_form_submit("Change Password") is True
    assert _label_looks_like_form_submit("Dashboard") is False
    assert _label_looks_like_form_submit("") is False
    assert _label_looks_like_form_submit(None) is False


def test_should_check_validation_error_after_action_scoped_to_submit_clicks_only():
    # click/submit บนปุ่มที่ label ดูเป็น Save/Submit -> เช็ค
    assert _should_check_validation_error_after_action("click", "Save") is True
    assert _should_check_validation_error_after_action("submit", "Update") is True
    # click ทั่วไป (เมนู/ลิงก์นำทาง) -> ไม่เช็ค กัน false positive จาก error ที่ไม่เกี่ยวกับฟอร์ม
    assert _should_check_validation_error_after_action("click", "Dashboard") is False
    # W20 (Task12 follow-up, "แก้ไขให้ครบ"): "fill" เช็คทุกครั้งแล้ว (เดิมตั้งใจไม่รวม —
    # false positive จาก "Required" ของช่องที่ยังไม่ได้กรอกตอนนี้จัดการที่จุดเรียกใช้แทนผ่าน
    # _is_bare_required_message() ไม่ใช่ตัดออกจาก trigger set ทั้งชนิด)
    assert _should_check_validation_error_after_action("fill", "Password") is True
    assert _should_check_validation_error_after_action("scroll", "") is False


def test_is_bare_required_message_filters_required_only_not_real_content_errors():
    assert _is_bare_required_message("* Required") is True
    assert _is_bare_required_message("Required") is True
    assert _is_bare_required_message("required.") is True
    assert _is_bare_required_message("  * Required  ") is True
    assert _is_bare_required_message("") is False
    assert _is_bare_required_message(None) is False
    assert _is_bare_required_message("Should have at least 7 characters") is False
    assert _is_bare_required_message("Employee Name already exists") is False


@pytest.mark.asyncio
async def test_run_task_stops_immediately_after_fill_when_real_validation_error_shown():
    """W20 (Task12 follow-up, "แก้ไขให้ครบ"): fill เพียงอย่างเดียว (ยังไม่ทันกดปุ่ม Save เลย)
    ก็ต้องหยุด task ทันทีถ้าหน้าเว็บโชว์ validation error ที่มีเนื้อหาจริง (ไม่ใช่แค่ "Required"
    เฉยๆ) — ไม่เรียก llm.next_action() ต่ออีกเลย และข้อความสุดท้ายต้องชวน user ตอบกลับมาด้วยค่า
    ใหม่ (เพื่อให้ turn ถัดไปกรอกแทนที่ในช่องเดิมต่อได้ ดู _PLAN_PROMPT_TEMPLATE::
    "Corrected-Value Retry on Validation Error" ใน llm.py)"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    fill_result = ActionResult(True, "fill(1)", "กรอกสำเร็จ")

    next_action_calls = [
        ("browser_action", {"type": "fill", "index": 1, "text": "weak"}, "t0", [], llm.TokenUsage()),
        # ไม่ควรถูกเรียกเลย — ถ้าเทสต์ผ่านแม้ next_action ถูกเรียกครั้งที่ 2 แปลว่า guard
        # ไม่ได้ตัด loop ออกจริงหลัง fill
        ("finish_task", {"success": True, "message": "should never reach here"}, "t1", [], llm.TokenUsage()),
    ]

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=[]), \
         patch("backend.app.core.orchestrator.execute", AsyncMock(return_value=fill_result)), \
         patch("backend.app.core.orchestrator.llm.append_tool_result", side_effect=lambda m, tid, r: m), \
         patch(
             "backend.app.core.orchestrator._scan_validation_errors",
             AsyncMock(return_value=["Should have at least 7 characters"]),
         ) as mock_scan, \
         patch("backend.app.core.orchestrator.llm.next_action", AsyncMock(side_effect=next_action_calls)) as mock_next_action:
        result = await Orchestrator().run_task("https://example.com", "goal", provider="anthropic")

    assert result["success"] is False
    assert result["completion_verification"] == "TASK_FAILED_USER_INPUT_ERROR"
    assert "Should have at least 7 characters" in result["message"]
    assert "ตอบกลับมาด้วยค่าใหม่" in result["message"]
    # หยุดทันทีหลัง action แรก (fill) — ไม่มีการเรียก next_action ครั้งที่ 2 (finish_task ปลอม
    # ที่ไม่ควรไปถึง) และไม่มีการกด Save เลยด้วยซ้ำ
    assert mock_next_action.await_count == 1
    assert mock_scan.await_count == 1


@pytest.mark.asyncio
async def test_run_task_stops_immediately_when_save_click_reveals_validation_error():
    """คลิกปุ่ม "Save" สำเร็จ (result.success=True) แต่หน้าเว็บโชว์ validation error ค้างอยู่ —
    ต้องหยุด task ทันที ไม่เรียก llm.next_action() อีกเลย (ไม่ลอง go_back/refresh/retry ต่อ)
    และคัดลอกข้อความ error ตรงตามที่ระบบแสดงจริงเข้า final message — scan รอบแรก (หลัง fill)
    ตั้งใจให้ยังไม่เจอ error เลย ("weak" ยังไม่ทัน commit จนกว่าจะกด Save) เพื่อแยกพิสูจน์ว่า
    click/submit เองก็ trigger guard นี้ได้อิสระจาก fill ไม่ใช่แค่ fill เท่านั้น"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    fill_result = ActionResult(True, "fill(1)", "กรอกสำเร็จ")
    save_click_result = ActionResult(True, "click(2)", "สำเร็จ")

    next_action_calls = [
        ("browser_action", {"type": "fill", "index": 1, "text": "weak"}, "t0", [], llm.TokenUsage()),
        ("browser_action", {"type": "click", "index": 2}, "t1", [], llm.TokenUsage()),
        # ไม่ควรถูกเรียกเลย — ถ้าเทสต์ผ่านแม้ next_action ถูกเรียกครบ 3 ครั้ง แปลว่า guard
        # ใหม่ไม่ได้ตัด loop ออกจริง
        ("finish_task", {"success": True, "message": "should never reach here"}, "t2", [], llm.TokenUsage()),
    ]

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(
             return_value=([{"index": 2, "tag": "button", "type": "", "label": "Save"}], "page"),
         )), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=[]), \
         patch("backend.app.core.orchestrator.execute", AsyncMock(side_effect=[fill_result, save_click_result])), \
         patch("backend.app.core.orchestrator.llm.append_tool_result", side_effect=lambda m, tid, r: m), \
         patch(
             "backend.app.core.orchestrator._scan_validation_errors",
             AsyncMock(side_effect=[[], ["Should have at least 7 characters"]]),
         ) as mock_scan, \
         patch("backend.app.core.orchestrator.llm.next_action", AsyncMock(side_effect=next_action_calls)) as mock_next_action:
        result = await Orchestrator().run_task("https://example.com", "goal", provider="anthropic")

    assert result["success"] is False
    assert result["completion_verification"] == "TASK_FAILED_USER_INPUT_ERROR"
    assert "Should have at least 7 characters" in result["message"]
    # หยุดทันทีหลัง action ที่ 2 (fill แล้ว click Save) — ไม่มีการเรียก next_action ครั้งที่ 3
    # (ซึ่งจะเป็น finish_task ปลอมที่ไม่ควรไปถึง)
    assert mock_next_action.await_count == 2
    assert mock_scan.await_count == 2


@pytest.mark.asyncio
async def test_run_task_fill_trigger_filters_bare_required_message_and_does_not_stop():
    """W20 (Task12 follow-up): fill ที่โผล่ "* Required" ล้วนๆ (ไม่มีเนื้อหาอื่น) ต้องไม่ถูก
    hard-stop — เป็น false positive จากช่องพี่น้องที่ยังไม่ได้กรอก ไม่ใช่ปัญหาของค่าที่เพิ่ง
    fill จริงๆ (เห็นจริงบน opensource-demo.orangehrmlive.com) ต่างจาก error ที่มีเนื้อหาจริงซึ่ง
    ยัง hard-stop ตามปกติ (ดู test_run_task_stops_immediately_after_fill_when_real_validation_
    error_shown ด้านบน)"""
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    fill_result = ActionResult(True, "fill(1)", "กรอกสำเร็จ")

    next_action_calls = [
        ("browser_action", {"type": "fill", "index": 1, "text": "x"}, "t0", [], llm.TokenUsage()),
        ("finish_task", {"success": True, "message": "เสร็จแล้ว"}, "t1", [], llm.TokenUsage()),
    ]

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=[]), \
         patch("backend.app.core.orchestrator.execute", AsyncMock(return_value=fill_result)), \
         patch("backend.app.core.orchestrator.llm.append_tool_result", side_effect=lambda m, tid, r: m), \
         patch(
             "backend.app.core.orchestrator._scan_validation_errors",
             AsyncMock(side_effect=[["* Required"], []]),
         ) as mock_scan, \
         patch("backend.app.core.orchestrator.llm.next_action", AsyncMock(side_effect=next_action_calls)):
        result = await Orchestrator().run_task("https://example.com", "goal", provider="anthropic")

    assert result["success"] is True
    assert result["message"] == "เสร็จแล้ว"
    # scan ถูกเรียก 2 ครั้ง: 1) จาก fill guard ใหม่ (เจอ "* Required" แต่กรองทิ้ง ไม่ hard-stop)
    # 2) จาก guard เดิมก่อน finish_task (ไม่เจอ error เลย -> ยอมรับทันที)
    assert mock_scan.await_count == 2


@pytest.mark.asyncio
async def test_run_task_completion_verification_defaults_to_ok_without_errors():
    mock_async_playwright, mock_browser, mock_playwright_ctx = _patch_browser()
    click_result = ActionResult(True, "click(1)", "สำเร็จ")

    # ต้องมี action ก่อน finish_task(true) อย่างน้อย 1 ครั้ง กัน guard คนละตัวชื่อ
    # "premature true finish" (steps_taken==0) ที่ไม่เกี่ยวกับสิ่งที่เทสต์นี้ตั้งใจพิสูจน์
    # trigger มาปนแทน
    next_action_calls = [
        ("browser_action", {"type": "click", "index": 1}, "t0", [], llm.TokenUsage()),
        ("finish_task", {"success": True, "message": "เสร็จ"}, "t1", [], llm.TokenUsage()),
    ]

    with patch("backend.app.core.orchestrator.async_playwright", mock_async_playwright), \
         patch("backend.app.core.orchestrator.goto", AsyncMock(return_value=_GOTO_OK)), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock(return_value=_WAIT_OK)), \
         patch("backend.app.core.orchestrator.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.orchestrator.retriever.retrieve", return_value=[]), \
         patch("backend.app.core.orchestrator.execute", AsyncMock(return_value=click_result)), \
         patch("backend.app.core.orchestrator.llm.append_tool_result", side_effect=lambda m, tid, r: m), \
         patch("backend.app.core.orchestrator.llm.next_action", AsyncMock(side_effect=next_action_calls)):
        result = await Orchestrator().run_task("https://example.com", "goal", provider="anthropic")

    assert result["success"] is True
    assert result["completion_verification"] == "OK"

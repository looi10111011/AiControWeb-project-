from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.app.core.orchestrator import Orchestrator

# เทสต์กลุ่มนี้ mock fastpath_executor.execute_template() ตรงๆ — ไม่สนใจ control flow
# ภายใน (มีเทสต์ของตัวเองแยกต่างหากใน test_fastpath_executor.py แล้ว) หน้าที่ของไฟล์นี้
# คือพิสูจน์แค่ resource-resolution logic ของ run_fastpath() เอง (page=/browser=/ทั้งคู่/
# ไม่มีเลย) และว่า client/model/run_task_fallback ถูกส่งเข้า execute_template() ถูกต้อง


@pytest.mark.asyncio
async def test_run_fastpath_raises_when_both_page_and_browser_given():
    with pytest.raises(ValueError):
        await Orchestrator().run_fastpath(
            url="https://example.com", goal="goal", template_id="t1", steps=[], slot_values={},
            page=MagicMock(), browser=MagicMock(),
        )


@pytest.mark.asyncio
async def test_run_fastpath_raises_when_neither_page_nor_browser_given():
    with pytest.raises(ValueError):
        await Orchestrator().run_fastpath(
            url="https://example.com", goal="goal", template_id="t1", steps=[], slot_values={},
        )


@pytest.mark.asyncio
async def test_run_fastpath_uses_session_managed_page_without_opening_or_closing_context():
    session_page = MagicMock()
    with patch("backend.app.core.orchestrator.fastpath_executor.execute_template", AsyncMock(return_value={"success": True})) as mock_exec:
        result = await Orchestrator().run_fastpath(
            url="https://example.com", goal="goal", template_id="t1", steps=[{"action": "click"}],
            slot_values={"a": "b"}, provider="anthropic", page=session_page,
        )

    assert result == {"success": True}
    _, kwargs = mock_exec.call_args
    assert kwargs["page"] is session_page
    assert kwargs["template_id"] == "t1"
    assert kwargs["slot_values"] == {"a": "b"}
    assert callable(kwargs["run_task_fallback"])


@pytest.mark.asyncio
async def test_run_fastpath_opens_and_closes_context_when_borrowing_from_pool():
    mock_browser = MagicMock()
    mock_context = AsyncMock()
    mock_page = MagicMock()
    mock_browser.new_context = AsyncMock(return_value=mock_context)
    mock_context.new_page = AsyncMock(return_value=mock_page)

    with patch("backend.app.core.orchestrator.fastpath_executor.execute_template", AsyncMock(return_value={"success": True})):
        await Orchestrator().run_fastpath(
            url="https://example.com", goal="goal", template_id="t1", steps=[], slot_values={},
            browser=mock_browser,
        )

    mock_browser.new_context.assert_awaited_once()
    mock_context.new_page.assert_awaited_once()
    mock_context.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_fastpath_fallback_closure_calls_run_task_on_same_page():
    session_page = MagicMock()
    captured_fallback = {}

    async def _capture_and_call(**kwargs):
        captured_fallback["fallback"] = kwargs["run_task_fallback"]
        return {"success": True}

    with patch("backend.app.core.orchestrator.fastpath_executor.execute_template", _capture_and_call), \
         patch.object(Orchestrator, "run_task", AsyncMock(return_value={"success": False, "execution_mode": "slow"})) as mock_run_task:
        await Orchestrator().run_fastpath(
            url="https://example.com", goal="goal", template_id="t1", steps=[], slot_values={},
            page=session_page, session_id="sess-1",
        )
        fallback_result = await captured_fallback["fallback"]()

    assert fallback_result == {"success": False, "execution_mode": "slow"}
    _, kwargs = mock_run_task.call_args
    assert kwargs["page"] is session_page
    assert kwargs["session_id"] == "sess-1"


# --- W_procmem: run_fastpath() must navigate + auto-login itself, mirroring run_task() ---
#
# บั๊กจริงที่เจอตอน Phase 4 validation: เดิม fastpath_executor.execute_template() navigate
# เอง แต่ไม่เคยเรียก _maybe_auto_login() เลย — ทำให้ domain ที่มี credential เก็บไว้ (ดู
# site_learning/auto_login.py) replay ไม่ได้เลยถ้าหน้าเป้าหมายต้อง login ก่อน (template
# เองก็ไม่มี step login เพราะ auto-login เกิด "นอก" LLM loop เสมอ) — ย้าย navigation มา
# ไว้ที่ run_fastpath() แล้วเรียก _maybe_auto_login() ต่อ ก่อนส่งต่อให้ execute_template()


@pytest.mark.asyncio
async def test_run_fastpath_calls_maybe_auto_login_before_execute_template():
    session_page = MagicMock(url="https://example.com/dashboard")
    with patch("backend.app.core.orchestrator.fastpath_executor.execute_template", AsyncMock(return_value={"success": True})), \
         patch("backend.app.core.orchestrator._maybe_auto_login", AsyncMock(return_value=None)) as mock_auto_login, \
         patch("backend.app.core.orchestrator.goto", AsyncMock()), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock()):
        await Orchestrator().run_fastpath(
            url="https://example.com", goal="goal", template_id="t1", steps=[], slot_values={},
            page=session_page,
        )

    mock_auto_login.assert_awaited_once_with(session_page, verbose=False)


@pytest.mark.asyncio
async def test_run_fastpath_navigates_when_page_is_on_a_different_domain():
    session_page = MagicMock(url="https://other-site.com/somewhere")
    with patch("backend.app.core.orchestrator.fastpath_executor.execute_template", AsyncMock(return_value={"success": True})), \
         patch("backend.app.core.orchestrator._maybe_auto_login", AsyncMock(return_value=None)), \
         patch("backend.app.core.orchestrator.goto", AsyncMock()) as mock_goto, \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock()):
        await Orchestrator().run_fastpath(
            url="https://example.com/login", goal="goal", template_id="t1", steps=[], slot_values={},
            page=session_page,
        )

    mock_goto.assert_awaited_once_with(session_page, "https://example.com/login")


@pytest.mark.asyncio
async def test_run_fastpath_skips_navigation_when_already_on_target_domain():
    """W12-style detect-current-page behavior mirrored from run_task() — a session page
    already on the target domain (e.g. reused from a previous turn) must not be
    re-navigated needlessly."""
    session_page = MagicMock(url="https://example.com/dashboard")
    with patch("backend.app.core.orchestrator.fastpath_executor.execute_template", AsyncMock(return_value={"success": True})), \
         patch("backend.app.core.orchestrator._maybe_auto_login", AsyncMock(return_value=None)), \
         patch("backend.app.core.orchestrator.goto", AsyncMock()) as mock_goto, \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock()):
        await Orchestrator().run_fastpath(
            url="https://example.com/login", goal="goal", template_id="t1", steps=[], slot_values={},
            page=session_page,
        )

    mock_goto.assert_not_called()


@pytest.mark.asyncio
async def test_run_fastpath_emits_auto_login_failed_event_on_failure():
    session_page = MagicMock(url="https://example.com/login")
    on_event = AsyncMock()
    with patch("backend.app.core.orchestrator.fastpath_executor.execute_template", AsyncMock(return_value={"success": True})), \
         patch("backend.app.core.orchestrator._maybe_auto_login", AsyncMock(return_value="รหัสผ่านผิด")), \
         patch("backend.app.core.orchestrator.goto", AsyncMock()), \
         patch("backend.app.core.orchestrator.wait_stable", AsyncMock()):
        await Orchestrator().run_fastpath(
            url="https://example.com/login", goal="goal", template_id="t1", steps=[], slot_values={},
            page=session_page, on_event=on_event,
        )

    events = [c.args[0] for c in on_event.await_args_list]
    assert any(e.get("kind") == "auto_login_failed" and e.get("reason") == "รหัสผ่านผิด" for e in events)

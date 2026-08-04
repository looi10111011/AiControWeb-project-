from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.app.core import llm
from backend.app.core.fastpath_executor import execute_template

# ทุกเทสต์ mock resolve_locator()/wait_stable()/get_snapshot()/llm.repair_step()/
# procedural_memory.record_template_outcome() ตรงๆ (ไม่เปิด browser จริง — dom_locator.py
# มีเทสต์ของตัวเองแยกต่างหากที่เปิด chromium จริงแล้ว หน้าที่ของไฟล์นี้คือพิสูจน์ control
# flow ของ execute_template() เอง: slot substitution, mask sensitive, repair/escalation
# boundary, goto-step skip) — execute_template() ไม่ navigate/auto-login เองอีกต่อไป
# (ย้ายไปที่ orchestrator.py::run_fastpath() แล้ว ดู test_run_fastpath.py) จึงไม่ต้อง
# mock goto() ที่นี่


def _mock_locator(**overrides):
    locator = MagicMock()
    locator.click = AsyncMock()
    locator.fill = AsyncMock()
    locator.select_option = AsyncMock()
    locator.check = AsyncMock()
    locator.hover = AsyncMock()
    locator.press = AsyncMock()
    locator.input_value = AsyncMock(return_value="")
    for name, value in overrides.items():
        setattr(locator, name, value)
    return locator


@pytest.fixture(autouse=True)
def _patch_navigation():
    with patch("backend.app.core.fastpath_executor.wait_stable", AsyncMock()), \
         patch("backend.app.core.fastpath_executor.procedural_memory.record_template_outcome") as mock_outcome:
        yield mock_outcome


_STEPS = [
    {"action": "goto", "target": {}},
    {"action": "fill", "target": {"accessible_name": "Username"}, "value": "{{username}}"},
    {"action": "click", "target": {"accessible_name": "Login"}},
]


@pytest.mark.asyncio
async def test_execute_template_happy_path_succeeds_without_calling_repair(_patch_navigation):
    fill_locator = _mock_locator(input_value=AsyncMock(return_value="alice"))
    click_locator = _mock_locator()
    on_event = AsyncMock()

    with patch(
        "backend.app.core.fastpath_executor.resolve_locator",
        AsyncMock(side_effect=[fill_locator, click_locator]),
    ), patch("backend.app.core.fastpath_executor.llm.repair_step") as mock_repair:
        result = await execute_template(
            page=MagicMock(), url="https://example.com/login", goal="log in",
            template_id="t1", steps=_STEPS, slot_values={"username": "alice"},
            client=MagicMock(), model="model-x", provider="anthropic", on_event=on_event,
        )

    assert result["success"] is True
    assert result["execution_mode"] == "fastpath"
    assert result["steps"] == 2  # goto step excluded from the count
    fill_locator.fill.assert_awaited_once_with("alice", timeout=3000)
    mock_repair.assert_not_called()
    _patch_navigation.assert_called_once_with("t1", success=True)
    # SSE events: 2 "step" + 2 "plan_step_done"
    assert on_event.await_count == 4


@pytest.mark.asyncio
async def test_execute_template_masks_sensitive_value_in_history_and_events(_patch_navigation):
    steps = [{"action": "fill", "target": {"accessible_name": "Password"}, "value": "{{password}}", "sensitive": True}]
    locator = _mock_locator(input_value=AsyncMock(return_value="hunter2"))
    on_event = AsyncMock()

    with patch("backend.app.core.fastpath_executor.resolve_locator", AsyncMock(return_value=locator)):
        result = await execute_template(
            page=MagicMock(), url="https://example.com", goal="goal", template_id="t1",
            steps=steps, slot_values={"password": "hunter2"}, client=MagicMock(),
            model="model-x", provider="anthropic", on_event=on_event,
        )

    assert result["success"] is True
    assert "hunter2" not in str(result["history"])
    assert "••••••" in str(result["history"])
    for call in on_event.await_args_list:
        assert "hunter2" not in str(call)


@pytest.mark.asyncio
async def test_execute_template_repairs_failed_step_then_succeeds(_patch_navigation):
    failing_locator = _mock_locator(click=AsyncMock(side_effect=RuntimeError("element not interactable")))
    fixed_locator = _mock_locator()
    steps = [{"action": "click", "target": {"accessible_name": "Login"}}]
    repaired_step = {"action": "click", "target": {"accessible_name": "Sign in"}}

    with patch(
        "backend.app.core.fastpath_executor.resolve_locator",
        AsyncMock(side_effect=[failing_locator, fixed_locator]),
    ), patch("backend.app.core.fastpath_executor.get_snapshot", AsyncMock(return_value=([], "page text"))), \
       patch("backend.app.core.fastpath_executor.llm.repair_step", AsyncMock(return_value=repaired_step)) as mock_repair:
        result = await execute_template(
            page=MagicMock(), url="https://example.com", goal="goal", template_id="t1",
            steps=steps, slot_values={}, client=MagicMock(), model="model-x", provider="anthropic",
        )

    assert result["success"] is True
    mock_repair.assert_awaited_once()
    fixed_locator.click.assert_awaited_once()


@pytest.mark.asyncio
async def test_execute_template_escalates_to_fallback_when_repair_says_replan(_patch_navigation):
    failing_locator = _mock_locator(click=AsyncMock(side_effect=RuntimeError("gone")))
    steps = [{"action": "click", "target": {"accessible_name": "Login"}}]
    fallback_result = {"success": True, "steps": 3, "message": "done via slow path", "history": [], "tokens": {}, "plan": "", "final_page_state": ""}
    run_task_fallback = AsyncMock(return_value=fallback_result)

    with patch("backend.app.core.fastpath_executor.resolve_locator", AsyncMock(return_value=failing_locator)), \
         patch("backend.app.core.fastpath_executor.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.fastpath_executor.llm.repair_step", AsyncMock(return_value={"action": "replan"})):
        result = await execute_template(
            page=MagicMock(), url="https://example.com", goal="goal", template_id="t1",
            steps=steps, slot_values={}, client=MagicMock(), model="model-x", provider="anthropic",
            run_task_fallback=run_task_fallback,
        )

    run_task_fallback.assert_awaited_once()
    assert result["execution_mode"] == "fastpath_escalated"
    assert result["success"] is True  # ผลลัพธ์ของ slow-path fallback ตรงๆ
    _patch_navigation.assert_called_once_with("t1", success=False)


@pytest.mark.asyncio
async def test_execute_template_escalates_after_exceeding_max_repair_attempts(_patch_navigation, monkeypatch):
    from backend.app.config import settings
    monkeypatch.setattr(settings, "procedural_memory_max_repair_attempts", 1)

    failing_locator = _mock_locator(click=AsyncMock(side_effect=RuntimeError("still broken")))
    steps = [{"action": "click", "target": {"accessible_name": "Login"}}]
    # Repair keeps returning a "fix" that still points at the same broken locator —
    # after max_repair_attempts is reached it must escalate rather than retry forever.
    repaired_step = {"action": "click", "target": {"accessible_name": "Login"}}

    with patch("backend.app.core.fastpath_executor.resolve_locator", AsyncMock(return_value=failing_locator)), \
         patch("backend.app.core.fastpath_executor.get_snapshot", AsyncMock(return_value=([], "page"))), \
         patch("backend.app.core.fastpath_executor.llm.repair_step", AsyncMock(return_value=repaired_step)) as mock_repair:
        result = await execute_template(
            page=MagicMock(), url="https://example.com", goal="goal", template_id="t1",
            steps=steps, slot_values={}, client=MagicMock(), model="model-x", provider="anthropic",
        )

    assert result["success"] is False
    assert result["execution_mode"] == "fastpath_failed"
    assert mock_repair.await_count == 1  # bounded, not infinite


@pytest.mark.asyncio
async def test_execute_template_skips_goto_steps_in_replay(_patch_navigation):
    locator = _mock_locator()
    with patch("backend.app.core.fastpath_executor.resolve_locator", AsyncMock(return_value=locator)):
        result = await execute_template(
            page=MagicMock(), url="https://example.com", goal="goal", template_id="t1",
            steps=[{"action": "goto"}, {"action": "click", "target": {"accessible_name": "Go"}}],
            slot_values={}, client=MagicMock(), model="model-x", provider="anthropic",
        )

    assert result["steps"] == 1
    assert result["success"] is True

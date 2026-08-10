from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.app.core import llm
from backend.app.core.fastpath_executor import build_navigation_steps, execute_navigation, execute_template
from backend.app.site_learning.schema import PageInfo, SiteManual

# ทุกเทสต์ mock resolve_locator()/wait_stable()/get_snapshot()/llm.repair_step()/
# procedural_memory.record_template_outcome() ตรงๆ (ไม่เปิด browser จริง — dom_locator.py
# มีเทสต์ของตัวเองแยกต่างหากที่เปิด chromium จริงแล้ว หน้าที่ของไฟล์นี้คือพิสูจน์ control
# flow ของ execute_template() เอง: slot substitution, mask sensitive, repair/escalation
# boundary, goto-step skip) — execute_template() เองไม่ navigate/auto-login เองอีกต่อไป
# (ย้ายไปที่ orchestrator.py::run_fastpath() แล้ว ดู test_run_fastpath.py) แต่
# execute_navigation() (W67) เรียก goto() เองก่อน replay เสมอ — autouse fixture ด้านล่าง
# patch goto() ไว้ให้ทุกเทสต์ในไฟล์นี้เป็น no-op AsyncMock กันไม่ให้ execute_navigation()
# เทสต์เก่าพัง (MagicMock().goto() ไม่ awaitable)


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
         patch("backend.app.core.fastpath_executor.goto", AsyncMock()), \
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


@pytest.mark.asyncio
async def test_execute_template_requires_approval_for_risky_click(_patch_navigation):
    locator = _mock_locator()
    denied = AsyncMock(return_value=False)
    with patch("backend.app.core.fastpath_executor.resolve_locator", AsyncMock(return_value=locator)):
        result = await execute_template(
            page=MagicMock(), url="https://example.com", goal="delete account", template_id="t1",
            steps=[{"action": "click", "target": {"accessible_name": "Delete account"}}],
            slot_values={}, client=MagicMock(), model="model-x", provider="anthropic",
            ask_user_func=denied,
        )

    assert result["success"] is False
    assert result["execution_mode"] == "fastpath_blocked"
    denied.assert_awaited_once()
    locator.click.assert_not_awaited()


# ---------------- W66[C] ("Fast-Path Navigation", manual trigger): build_navigation_steps() ----------------


def _page(name, url, parent_url="", arrived_via=None):
    return PageInfo(name=name, url=url, parent_url=parent_url, arrived_via=arrived_via or {})


def test_build_navigation_steps_walks_parent_chain_root_to_target():
    root = _page("Dashboard", "https://example.com/")
    hub = _page("Admin", "https://example.com/admin", parent_url=root.url, arrived_via={"accessible_name": "Admin"})
    target = _page(
        "User Management", "https://example.com/admin/users",
        parent_url=hub.url, arrived_via={"accessible_name": "User Management"},
    )
    manual = SiteManual(website="example.com", pages=[root, hub, target])

    result = build_navigation_steps(manual, target)

    assert result is not None
    root_url, steps = result
    assert root_url == root.url
    assert steps == [
        {"action": "click", "target": {"accessible_name": "Admin"}},
        {"action": "click", "target": {"accessible_name": "User Management"}},
    ]


def test_build_navigation_steps_returns_empty_steps_when_target_is_root():
    root = _page("Dashboard", "https://example.com/")
    manual = SiteManual(website="example.com", pages=[root])

    result = build_navigation_steps(manual, root)

    assert result == ("https://example.com/", [])


def test_build_navigation_steps_returns_none_when_parent_not_in_manual():
    # target อ้าง parent_url ที่ไม่มีหน้าไหนใน manual.pages ตรงกันเลย (เช่น หน้านั้นไม่เคย
    # ถูก crawl บันทึกไว้จริง — เจอ URL ผ่าน DFS-click/login flow ที่ v1 ยังไม่ thread ให้)
    target = _page("Orphan", "https://example.com/orphan", parent_url="https://example.com/missing-parent")
    manual = SiteManual(website="example.com", pages=[target])

    result = build_navigation_steps(manual, target)

    assert result is None


def test_build_navigation_steps_returns_none_when_arrived_via_empty_mid_chain():
    root = _page("Dashboard", "https://example.com/")
    # hub ไม่มี arrived_via (เช่น มาจาก login flow ที่ v1 ยังไม่ thread ให้ — ดู crawler.py
    # W66[A] docstring) — ทำให้เดินย้อนได้ถึง root แต่ replay ไม่ได้จริง (ไม่รู้จะคลิกอะไร)
    hub = _page("Post-Login", "https://example.com/home", parent_url=root.url, arrived_via={})
    target = _page(
        "Admin", "https://example.com/admin", parent_url=hub.url, arrived_via={"accessible_name": "Admin"},
    )
    manual = SiteManual(website="example.com", pages=[root, hub, target])

    result = build_navigation_steps(manual, target)

    assert result is None


def test_build_navigation_steps_returns_none_on_cycle():
    # parent_url ชี้กลับมาหาตัวเอง (ข้อมูลเพี้ยน/ผิดปกติ) — ต้องไม่ loop ไม่รู้จบ
    a = _page("A", "https://example.com/a", parent_url="https://example.com/b", arrived_via={"accessible_name": "A"})
    b = _page("B", "https://example.com/b", parent_url="https://example.com/a", arrived_via={"accessible_name": "B"})
    manual = SiteManual(website="example.com", pages=[a, b])

    result = build_navigation_steps(manual, a)

    assert result is None


# ---------------- W66[C]: execute_navigation() ----------------


@pytest.mark.asyncio
async def test_execute_navigation_fails_gracefully_when_no_nav_data():
    target = _page("Orphan", "https://example.com/orphan", parent_url="https://example.com/missing")
    manual = SiteManual(website="example.com", pages=[target])

    result = await execute_navigation(
        MagicMock(), "goal", manual, target, MagicMock(), "model-x", "anthropic",
    )

    assert result["success"] is False
    assert result["execution_mode"] == "nav_unavailable"


@pytest.mark.asyncio
async def test_execute_navigation_succeeds_immediately_when_target_is_root():
    root = _page("Dashboard", "https://example.com/")
    manual = SiteManual(website="example.com", pages=[root])

    result = await execute_navigation(
        MagicMock(), "goal", manual, root, MagicMock(), "model-x", "anthropic",
    )

    assert result["success"] is True
    assert result["steps"] == 0


@pytest.mark.asyncio
async def test_execute_navigation_replays_via_execute_template_with_built_steps():
    root = _page("Dashboard", "https://example.com/")
    target = _page(
        "Admin", "https://example.com/admin", parent_url=root.url, arrived_via={"accessible_name": "Admin"},
    )
    manual = SiteManual(website="example.com", pages=[root, target])
    mock_page = MagicMock()

    with patch(
        "backend.app.core.fastpath_executor.execute_template", AsyncMock(return_value={"success": True}),
    ) as mock_execute_template:
        result = await execute_navigation(
            mock_page, "goal", manual, target, MagicMock(), "model-x", "anthropic",
        )

    assert result == {"success": True}
    mock_execute_template.assert_awaited_once()
    call_kwargs = mock_execute_template.await_args.kwargs
    assert call_kwargs["url"] == root.url
    assert call_kwargs["steps"] == [{"action": "click", "target": {"accessible_name": "Admin"}}]
    assert call_kwargs["template_id"] == "nav:example.com:Admin"
    assert call_kwargs["run_task_fallback"] is None  # W66[C]: ห้าม auto-escalate กันชน recursive call


@pytest.mark.asyncio
async def test_execute_navigation_gotos_root_url_before_replaying_steps():
    # W67 bug fix: execute_navigation() ต้อง goto(root_url) เอง ไม่พึ่งว่า caller เพิ่ง
    # goto มาที่เดียวกันมาก่อนหน้านี้ "บังเอิญ" — ยืนยัน call order: goto ก่อน execute_template
    root = _page("Dashboard", "https://example.com/")
    target = _page(
        "Admin", "https://example.com/admin", parent_url=root.url, arrived_via={"accessible_name": "Admin"},
    )
    manual = SiteManual(website="example.com", pages=[root, target])
    mock_page = MagicMock()
    call_order = []

    mock_goto = AsyncMock(side_effect=lambda *a, **kw: call_order.append("goto"))
    mock_execute_template = AsyncMock(
        side_effect=lambda **kw: call_order.append("execute_template") or {"success": True}
    )

    with patch("backend.app.core.fastpath_executor.goto", mock_goto), \
         patch("backend.app.core.fastpath_executor.execute_template", mock_execute_template):
        result = await execute_navigation(
            mock_page, "goal", manual, target, MagicMock(), "model-x", "anthropic",
        )

    assert result == {"success": True}
    mock_goto.assert_awaited_once_with(mock_page, root.url)
    assert call_order == ["goto", "execute_template"]


@pytest.mark.asyncio
async def test_execute_navigation_gotos_root_url_even_when_target_is_root():
    # target เป็น root เอง (ไม่มี click step) — ยัง goto(root_url) เองเสมอ ไม่พึ่ง caller
    root = _page("Dashboard", "https://example.com/")
    manual = SiteManual(website="example.com", pages=[root])
    mock_page = MagicMock()
    mock_goto = AsyncMock()

    with patch("backend.app.core.fastpath_executor.goto", mock_goto):
        result = await execute_navigation(
            mock_page, "goal", manual, root, MagicMock(), "model-x", "anthropic",
        )

    assert result["success"] is True
    mock_goto.assert_awaited_once_with(mock_page, root.url)

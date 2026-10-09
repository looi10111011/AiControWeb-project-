"""core/fastpath_executor.py — W_procmem: replay procedural template (core/procedural_memory.py) โดยข้าม
perceive->plan->act LLM loop — จุดที่ประหยัด LLM call/latency จริงของ procedural memory

ต่อ step: แทน {{slot}} -> resolve_locator() (core/dom_locator.py) -> dispatch กับ Locator ตรงๆ (actions.py
ผูกกับ index ephemeral ของ perception.py) -> verify แบบ deterministic — ล้มเหลวเรียก llm.repair_step()
จนเกิน settings.procedural_memory_max_repair_attempts หรือ Repair ตอบ replan -> escalate ไป
run_task_fallback() (run_task() เต็ม) — แย่สุดคือช้าเท่า slow-path ไม่ใช่ล้มเหลวเพิ่ม

W_procmem (ข้อจำกัด v1): widget="vue_dropdown" ต้อง 2 ปฏิสัมพันธ์ (เปิด + เลือก) แต่ executor dispatch
แค่ 1 action ต่อ step — จะ verify ไม่ผ่านแล้วเข้า Repair (ไม่ crash แต่ไม่ efficient)"""

import time
from typing import Any, Awaitable, Callable, Optional

from playwright.async_api import Locator, Page

from backend.app.config import settings
from backend.app.core import llm, procedural_memory
from backend.app.core.actions import AskUserFunc, goto, wait_stable
from backend.app.core.dom_locator import resolve_locator
from backend.app.core.perception import get_snapshot
from backend.app.permission.rules import ActionRisk, classify_action

_ELEMENT_ACTION_TIMEOUT_MS = 3000

OnEventFunc = Callable[[dict], Awaitable[None]]
RunTaskFallbackFunc = Callable[[], Awaitable[dict]]


class FastpathPermissionDenied(Exception):
    """A template step was blocked or denied before it changed the page."""


def _tokens_dict(usage: "llm.TokenUsage") -> dict:
    return {
        "input": usage.input_tokens,
        "output": usage.output_tokens,
        "cache_read": usage.cache_read_tokens,
        "cache_creation": usage.cache_creation_tokens,
    }


def _result(
    success: bool, steps: int, message: str, history: list, usage: "llm.TokenUsage", execution_mode: str,
) -> dict:
    """shape เดียวกับ orchestrator.run_task() + execution_mode"""
    return {
        "success": success,
        "steps": steps,
        "message": message,
        "history": history,
        "tokens": _tokens_dict(usage),
        "plan": "",
        "final_page_state": "",
        "execution_mode": execution_mode,
    }


# W66[C] (Fast-Path Navigation): เดินไปหน้าที่ site_learning เรียนรู้ไว้โดยไม่เรียก LLM — reuse
# execute_template() แค่ steps มาจาก PageInfo.parent_url/arrived_via chain แทน procedural template
# manual/target_page เป็น Any ตั้งใจ: import site_learning.schema ที่นี่จะ circular
# (site_learning -> crawler -> orchestrator -> fastpath_executor) บั๊กเดียวกับ W65[3]

def build_navigation_steps(manual: Any, target_page: Any) -> Optional[tuple[str, list[dict]]]:
    """เดินย้อน parent_url จาก target_page ถึง root แล้วกลับด้านเป็น click steps (shape
    _TEMPLATE_STEP_SCHEMA) คืน (root_url, steps) หรือ None (ไม่ throw) ถ้าเจอ cycle,
    parent_url ไม่อยู่ใน manual.pages หรือ hop ที่ arrived_via ว่าง (ดู crawler.py W66[A])"""
    pages_by_url = {p.url: p for p in manual.pages}
    chain: list[Any] = []
    current = target_page
    seen_urls: set[str] = set()
    while True:
        if current.url in seen_urls:
            return None
        seen_urls.add(current.url)
        chain.append(current)
        if not current.parent_url:
            break
        parent = pages_by_url.get(current.parent_url)
        if parent is None:
            return None
        current = parent
    chain.reverse()  # root -> target

    root_url = chain[0].url
    steps: list[dict] = []
    for page_info in chain[1:]:  # root ไม่มี arrived_via โดยนิยาม
        if not page_info.arrived_via:
            return None
        steps.append({"action": "click", "target": dict(page_info.arrived_via)})
    return root_url, steps


async def execute_navigation(
    page: Page,
    goal: str,
    manual: Any,
    target_page: Any,
    client: Any,
    model: str,
    provider: str,
    ask_user_func: Optional[AskUserFunc] = None,
    on_event: Optional[OnEventFunc] = None,
) -> dict:
    """เดินไปหน้า target_page ด้วย execute_template() — success = ถึงหน้าเป้าหมาย ไม่ใช่ goal เสร็จ
    (run_task() ทำต่อเอง)

    W66[C]: run_task_fallback=None เสมอ — ถูกเรียกจากใน run_task() อยู่แล้ว escalate จะ recursive
    ล้มเหลวคืน {success: False} ให้ผู้เรียก goto(url) กลับแล้วเดิน loop ปกติ
    W67: goto(root_url) เองที่นี่ — เดิมพึ่ง caller goto มาก่อน ซึ่ง url อาจไม่ตรง crawl root (เช่น auto-login redirect)"""
    nav_data = build_navigation_steps(manual, target_page)
    if nav_data is None:
        return _result(
            False, 0, "[FAIL] ไม่มีข้อมูล navigation path ที่ใช้ replay ได้สำหรับหน้านี้",
            [], llm.TokenUsage(), "nav_unavailable",
        )
    root_url, click_steps = nav_data
    await goto(page, root_url)
    await wait_stable(page)
    if not click_steps:
        # target_page คือ root เอง — goto ด้านบนพาไปถึงแล้ว
        return _result(True, 0, "อยู่หน้าเป้าหมายอยู่แล้ว (root page)", [], llm.TokenUsage(), "nav")

    template_id = f"nav:{manual.website}:{target_page.name}"
    return await execute_template(
        page=page, url=root_url, goal=goal, template_id=template_id, steps=click_steps,
        slot_values={}, client=client, model=model, provider=provider,
        ask_user_func=ask_user_func, on_event=on_event, run_task_fallback=None,
    )


async def _dispatch_step(locator: Locator, action: str, value: Optional[str], timeout: int) -> None:
    """ยิง action กับ Locator ตรงๆ — throw ถ้าล้มเหลว ให้ execute_template() ตัดสินเรียก Repair"""
    if action == "click":
        await locator.click(timeout=timeout)
    elif action == "fill":
        await locator.fill(value or "", timeout=timeout)
    elif action == "select":
        await locator.select_option(label=value, timeout=timeout)
    elif action == "check":
        await locator.check(timeout=timeout)
    elif action == "hover":
        await locator.hover(timeout=timeout, force=True)
    elif action == "press_key":
        await locator.press(value or "", timeout=timeout)
    else:
        raise ValueError(f"fastpath_executor ไม่รู้จัก action type: {action!r}")


async def _verify_step(locator: Locator, action: str, expected_value: Optional[str], sensitive: bool) -> None:
    """verify deterministic ไม่ใช้ LLM — throw ถ้าไม่ผ่าน. fill: เทียบ input_value() (sensitive เช็คแค่ไม่ว่าง
    ไม่เทียบรหัสผ่าน) action อื่นถือว่าผ่านถ้า dispatch ไม่ throw (Playwright actionability check แล้ว)"""
    if action != "fill":
        return
    actual = await locator.input_value()
    if sensitive:
        if not actual:
            raise RuntimeError("กรอกรหัสผ่าน/ข้อมูลลับแล้วแต่ช่องยังว่างเปล่า (verify ไม่ผ่าน)")
        return
    if actual != (expected_value or ""):
        raise RuntimeError(
            f"กรอกค่าแล้วแต่ค่าที่อ่านกลับมาไม่ตรงกับที่ต้องการ (verify ไม่ผ่าน): "
            f"ได้ {actual!r} ต้องการ {expected_value!r}",
        )


async def _check_step_permission(
    action: str, descriptor: dict, ask_user_func: Optional[AskUserFunc],
) -> None:
    """Apply the normal action-risk policy before a fast-path locator mutation — stable locators
    must not bypass the permission boundary used by the regular executor."""
    label = str(
        descriptor.get("accessible_name")
        or descriptor.get("label")
        or descriptor.get("text")
        or ""
    )
    risk = classify_action({"type": action}, label=label)
    if risk is ActionRisk.BLOCKED:
        raise FastpathPermissionDenied(f"blocked action: {action} ({label or 'unnamed target'})")
    if risk is ActionRisk.NEEDS_CONFIRMATION:
        cmd = {"type": action, "label": label, "fastpath": True}
        if ask_user_func is None or not await ask_user_func(cmd):
            raise FastpathPermissionDenied(f"action was not approved: {action} ({label or 'unnamed target'})")


def _mask_step_for_repair(step: dict) -> dict:
    """copy ของ step ที่ mask value ถ้า sensitive — llm.repair_step() กำหนดให้ผู้เรียก mask เอง"""
    masked = dict(step)
    if masked.get("sensitive"):
        masked["value"] = "••••••"
    return masked


async def execute_template(
    page: Page,
    url: str,
    goal: str,
    template_id: str,
    steps: list[dict],
    slot_values: dict,
    client: Any,
    model: str,
    provider: str,
    ask_user_func: Optional[AskUserFunc] = None,
    on_event: Optional[OnEventFunc] = None,
    run_task_fallback: Optional[RunTaskFallbackFunc] = None,
) -> dict:
    """Replay template ทีละ step — เรียก llm.repair_step() เฉพาะ step ที่ล้มเหลว

    ไม่ navigate/auto-login เอง: page ต้องอยู่ถูกที่แล้ว และ goto step ใน `steps` ถูกข้ามเสมอ
    (W_procmem: เดิม goto เองแต่ไม่ auto-login ทำให้ domain ที่ต้อง login replay ไม่ได้ — ย้ายไป
    Orchestrator.run_fastpath() คุมทั้งคู่)

    escalation: เกิน max repair หรือ Repair ตอบ replan -> คืนผล run_task_fallback() (execution_mode=
    "fastpath_escalated") หรือ [FAIL] ถ้าไม่มี fallback. permission ถูกปฏิเสธ -> [BLOCKED]
    คืน dict shape เดียวกับ orchestrator.run_task() + execution_mode"""
    start = time.monotonic()

    action_steps = [s for s in steps if s.get("action") != "goto"]
    history: list[dict] = []
    total_usage = llm.TokenUsage()

    async def emit(event: dict) -> None:
        if on_event is not None:
            await on_event(event)

    async def escalate(step_num: int, reason: str) -> dict:
        procedural_memory.record_template_outcome(template_id, success=False)
        if run_task_fallback is not None:
            fallback_result = await run_task_fallback()
            fallback_result["execution_mode"] = "fastpath_escalated"
            return fallback_result
        return _result(
            False, step_num, f"[FAIL] fast-path หยุดที่ step {step_num}: {reason}",
            history, total_usage, "fastpath_failed",
        )

    for i, original_step in enumerate(action_steps):
        step_num = i + 1
        step = dict(original_step)
        descriptor = step.get("target") or {}
        repair_attempts = 0

        while True:
            error_text: Optional[str] = None
            raw_value = step.get("value")
            substituted_value = (
                procedural_memory.substitute_slots(str(raw_value), slot_values) if raw_value is not None else None
            )
            action = step.get("action", "")
            sensitive = bool(step.get("sensitive"))

            locator = await resolve_locator(page, descriptor) if descriptor else None
            if locator is None:
                error_text = "หา element ที่ตรงกับ locator ของ step นี้ไม่เจอบนหน้าปัจจุบัน"
            else:
                try:
                    await _check_step_permission(action, descriptor, ask_user_func)
                    await _dispatch_step(locator, action, substituted_value, _ELEMENT_ACTION_TIMEOUT_MS)
                    await _verify_step(locator, action, substituted_value, sensitive)
                except FastpathPermissionDenied as e:
                    procedural_memory.record_template_outcome(template_id, success=False)
                    return _result(False, step_num - 1, f"[BLOCKED] {e}", history, total_usage, "fastpath_blocked")
                except Exception as e:
                    error_text = str(e)

            if error_text is None:
                display_value = "••••••" if sensitive else substituted_value
                result_text = f"[OK] {action} -> สำเร็จ"
                history.append({
                    "step": step_num,
                    "cmd": {"type": action, "target": descriptor, "value": display_value},
                    "result": result_text,
                    "success": True,
                    "tokens": _tokens_dict(llm.TokenUsage()),
                })
                await emit({
                    "kind": "step", "step": step_num,
                    "cmd": {"type": action, "value": display_value}, "label": descriptor.get("accessible_name", ""),
                    "result": result_text, "success": True, "fastpath": True,
                })
                await emit({"kind": "plan_step_done", "step": step_num, "fastpath": True})
                if action in ("click", "select", "check", "press_key"):
                    await wait_stable(page)
                break

            if repair_attempts >= settings.procedural_memory_max_repair_attempts:
                return await escalate(step_num, error_text)
            repair_attempts += 1
            try:
                _, current_page_text = await get_snapshot(page)
            except Exception:
                current_page_text = ""
            repaired = await llm.repair_step(
                client, model, _mask_step_for_repair(step), error_text, current_page_text, provider,
            )
            if repaired.get("action") == "replan":
                return await escalate(step_num, f"Repair แนะนำ replan: {error_text}")
            # แทนเฉพาะ field ที่ Repair แก้ — คง "sensitive"/slot เดิม (repair_step prompt: ห้ามสลับข้อมูลข้ามช่อง)
            descriptor = repaired.get("target") or descriptor
            step = {
                **step,
                "action": repaired.get("action", step.get("action")),
                "target": descriptor,
            }
            if repaired.get("value") is not None:
                step["value"] = repaired["value"]

    procedural_memory.record_template_outcome(template_id, success=True)
    elapsed = time.monotonic() - start
    return _result(
        True, len(action_steps),
        f"ทำสำเร็จผ่าน fast-path replay ({len(action_steps)} step, {elapsed:.1f} วินาที)",
        history, total_usage, "fastpath",
    )

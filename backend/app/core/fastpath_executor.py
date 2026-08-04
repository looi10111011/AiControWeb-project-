"""core/fastpath_executor.py — W_procmem: replay ของ procedural template (ดู
core/procedural_memory.py) แบบข้าม perceive->plan->act LLM loop ทั้งหมด — นี่คือจุดที่
การประหยัด LLM call/latency ของทั้งระบบ procedural memory เกิดขึ้นจริง (Abstractor/
Planner ที่แก้ไปก่อนหน้านี้แค่เตรียมข้อมูล ยังไม่ได้ข้าม step-by-step loop เลย)

Control flow ต่อ step: แทนค่า {{slot}} -> mask ถ้า sensitive -> resolve_locator()
(fallback chain role+name -> label -> data-testid -> CSS, ดู core/dom_locator.py) ->
dispatch ตรงกับ Locator ที่ resolve ได้ (ไม่ผ่าน actions.py เพราะฟังก์ชันชุดนั้นผูกกับ
index ephemeral ของ perception.py ไม่ใช่ Locator ที่ resolve เองแล้ว) -> verify แบบ
deterministic (เทียบ input_value() สำหรับ fill, ไม่ throw ถือว่าสำเร็จสำหรับ action
อื่น) — ล้มเหลวขั้นไหนก็ตามเรียก llm.repair_step() (ดู core/llm.py) ครั้งเดียวต่อความ
พยายาม 1 ครั้ง จนกว่าจะเกิน settings.procedural_memory_max_repair_attempts หรือ Repair
เองตอบ {"action": "replan"} ตรงๆ — ถึงจุดนั้นจะ "escalate" คือยกเลิก fast-path ทั้งหมด
แล้วเรียก run_task_fallback() (เท่ากับ orchestrator.run_task() เต็มรูปแบบ ไม่มี
approved_plan — ร่างจาก LLM ใหม่ทั้งหมดเหมือนไม่มี template นี้อยู่เลย) เป็นตัวรับประกัน
ว่า fast-path ไม่มีทางทำให้ task ล้มเหลวหนักกว่าเดิม อย่างแย่ที่สุดคือช้าเท่า slow-path
เดิม ไม่ใช่ล้มเหลวเพิ่ม

W_procmem (ข้อจำกัดที่ทราบอยู่แล้ว, ยังไม่ทำใน v1 นี้): step ที่มี widget="vue_dropdown"
(หรือ custom widget อื่น) ต้องการ 2 ปฏิสัมพันธ์จริง (click เปิด dropdown แล้ว click
เลือก option) แต่ executor ตัวนี้ dispatch แค่ 1 การกระทำต่อ step เท่านั้น — Repair
module (llm.py) รับรู้ widget field และพยายามแนะนำแก้ไขได้ แต่ execute_template() เอง
ยังไม่มี logic พิเศษรองรับ multi-action ต่อ 1 step จริง (จะ dispatch แค่ action หลักแล้ว
อาจ verify ไม่ผ่าน -> ไปเข้า Repair ต่อตามปกติ ไม่ crash แต่ไม่ efficient เท่าที่ควร)"""

import time
from typing import Any, Awaitable, Callable, Optional

from playwright.async_api import Locator, Page

from backend.app.config import settings
from backend.app.core import llm, procedural_memory
from backend.app.core.actions import wait_stable
from backend.app.core.dom_locator import resolve_locator
from backend.app.core.perception import get_snapshot

_ELEMENT_ACTION_TIMEOUT_MS = 3000

OnEventFunc = Callable[[dict], Awaitable[None]]
RunTaskFallbackFunc = Callable[[], Awaitable[dict]]


def _tokens_dict(usage: "llm.TokenUsage") -> dict:
    return {
        "input": usage.input_tokens,
        "output": usage.output_tokens,
        "cache_read": usage.cache_read_tokens,
        "cache_creation": usage.cache_creation_tokens,
    }


async def _dispatch_step(locator: Locator, action: str, value: Optional[str], timeout: int) -> None:
    """ยิง action จริงกับ Locator ที่ resolve_locator() เจอมาแล้ว — ต่างจาก
    actions.py::execute() ตรงที่รับ Locator ตรงๆ ไม่ใช่ index (ดู module docstring
    หัวไฟล์ว่าทำไมใช้ actions.py เดิมไม่ได้) throw ตรงๆ ถ้าล้มเหลว (ไม่ห่อเป็น
    ActionResult เหมือน actions.py) ให้ผู้เรียก (execute_template ด้านล่าง) จับ
    exception เพื่อตัดสินใจเรียก Repair ต่อเอง"""
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
    """ตรวจสอบแบบ deterministic ล้วนๆ ไม่พึ่ง LLM เลย (mirror แนวทางเดียวกับ
    site_learning/auto_login.py::verify_login_success — cheap code check ไม่ใช่การ
    ตัดสินใจของโมเดล) — throw ถ้า verify ไม่ผ่าน ให้ execute_template() จับแล้วเรียก
    Repair ต่อเหมือน dispatch ล้มเหลวทั่วไป

    fill: เทียบ input_value() จริงกับค่าที่ต้องการ (ยกเว้น sensitive — เช็คแค่ไม่ว่าง
    เปล่า ไม่เทียบค่าจริงเพื่อไม่ให้ต้อง log/เทียบรหัสผ่านตรงๆ) action อื่นไม่มี cheap
    check ที่ใช้ได้ทั่วไป (ต่างจาก fill ที่มี input_value() ให้เช็คตรงๆ) — dispatch เอง
    ไม่ throw ถือว่าผ่านแล้ว (Playwright's actionability check ภายในตัว action เองเป็น
    ด่านแรกอยู่แล้ว)"""
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


def _mask_step_for_repair(step: dict) -> dict:
    """ไม่ส่งค่าจริงของ step ที่ sensitive=True เข้า LLM prompt เด็ดขาด (ดู
    llm.repair_step() docstring ที่ระบุว่าผู้เรียกต้อง mask เอง) — คืน copy ใหม่เสมอ
    ไม่แก้ step เดิม"""
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
    on_event: Optional[OnEventFunc] = None,
    run_task_fallback: Optional[RunTaskFallbackFunc] = None,
) -> dict:
    """Replay template ทีละ step โดยไม่เรียก LLM เลยตราบใดที่ทุก step สำเร็จตรงๆ (นี่คือ
    "happy path" ที่ประหยัด token/latency ตามจุดประสงค์ของทั้งระบบ) — เรียก
    llm.repair_step() เฉพาะตอน step ล้มเหลวจริงเท่านั้น (ไม่ใช่ทุก step)

    **ไม่ navigate/auto-login เองเลย** — สมมติว่า `page` อยู่บนเว็บเป้าหมายแล้ว (และ
    login แล้วถ้าจำเป็น) ตั้งแต่ก่อนเรียกฟังก์ชันนี้ (ดู
    orchestrator.py::Orchestrator.run_fastpath() ซึ่งเป็นผู้เรียกเดียวของฟังก์ชันนี้ —
    ทำ goto + _maybe_auto_login() เองก่อนส่งต่อมาที่นี่ mirror ลำดับเดียวกับที่
    run_task() ทำก่อนเข้า main loop) — W_procmem (แก้ไขหลังพบบั๊กจริง): เดิมฟังก์ชันนี้
    goto เอง แต่ไม่เคยเรียก auto-login เลย ทำให้ domain ที่มี credential เก็บไว้
    (auto_login.py) replay ไม่ได้เลยถ้าต้อง login ก่อน — ย้าย navigation ออกไปให้
    run_fastpath() คุมทั้งคู่พร้อมกันแทน goto step ใน `steps` (ถ้ามี จาก
    _format_trajectory_for_abstractor()) ถูกข้ามไปเสมอเพราะการ navigate จริงเกิดขึ้น
    ที่ผู้เรียกแล้ว ไม่ใช่ที่นี่

    escalation (การันตีว่าไม่มีทางแย่กว่า slow-path เดิม): step ไหนก็ตามที่ยัง fail
    หลัง repair ครบ settings.procedural_memory_max_repair_attempts ครั้ง หรือ Repair
    เองตอบ replan ตรงๆ — เลิก fast-path ทั้งหมดทันที เรียก run_task_fallback() แทน คืน
    ผลลัพธ์ของมันตรงๆ (tag execution_mode="fastpath_escalated") ถ้าไม่มี
    run_task_fallback ให้เลย (เช่นตอน unit test) คืน [FAIL] ตรงๆ แทน

    คืน dict รูปแบบเดียวกับ orchestrator.run_task() ทุกประการ
    ({success, steps, message, history, tokens, plan, final_page_state}) บวก
    execution_mode เพิ่มเติม (ไม่กระทบ consumer เดิมที่ไม่รู้จัก key นี้)"""
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
        return {
            "success": False,
            "steps": step_num,
            "message": f"[FAIL] fast-path หยุดที่ step {step_num}: {reason}",
            "history": history,
            "tokens": _tokens_dict(total_usage),
            "plan": "",
            "final_page_state": "",
            "execution_mode": "fastpath_failed",
        }

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
                    await _dispatch_step(locator, action, substituted_value, _ELEMENT_ACTION_TIMEOUT_MS)
                    await _verify_step(locator, action, substituted_value, sensitive)
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

            # ล้มเหลว -> เรียก Repair ก่อน escalate (ดู module docstring)
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
            # แทนที่เฉพาะ field ที่ Repair แก้มาให้ — คง "sensitive"/slot เดิมไว้เสมอ (ดู
            # llm.repair_step()'s prompt rule: ห้ามเปลี่ยนว่าข้อมูลไหนไปช่องไหน)
            descriptor = repaired.get("target") or descriptor
            step = {
                **step,
                "action": repaired.get("action", step.get("action")),
                "target": descriptor,
            }
            if repaired.get("value") is not None:
                step["value"] = repaired["value"]
            # วนกลับไปลอง resolve+dispatch+verify ใหม่ด้วย step ที่แก้แล้ว

    procedural_memory.record_template_outcome(template_id, success=True)
    elapsed = time.monotonic() - start
    return {
        "success": True,
        "steps": len(action_steps),
        "message": f"ทำสำเร็จผ่าน fast-path replay ({len(action_steps)} step, {elapsed:.1f} วินาที)",
        "history": history,
        "tokens": _tokens_dict(total_usage),
        "plan": "",
        "final_page_state": "",
        "execution_mode": "fastpath",
    }

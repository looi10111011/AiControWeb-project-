import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock
from zoneinfo import ZoneInfo

import pytest
from google.api_core.exceptions import ResourceExhausted
from groq import BadRequestError as GroqBadRequestError

from backend.app.core import llm

# ทุกเทสต์ mock ทั้ง AsyncGroq/Gemini client — ไม่ยิง API จริง


def _fake_tool_call(call_id, name, arguments_json):
    tc = MagicMock()
    tc.id = call_id
    tc.function.name = name
    tc.function.arguments = arguments_json
    return tc


def _fake_response(tool_calls, dumped_message, usage=(10, 5)):
    message = MagicMock()
    message.tool_calls = tool_calls
    message.model_dump.return_value = dumped_message
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    response.usage.prompt_tokens = usage[0]
    response.usage.completion_tokens = usage[1]
    return response


def _fake_bad_request(code: str) -> GroqBadRequestError:
    return GroqBadRequestError(
        message="bad request",
        response=MagicMock(status_code=400),
        body={"error": {"code": code, "message": "..."}},
    )


def _last_user_text(kwargs):
    """W_cache2 (SPD-1): next_action() ห่อ content ของ user message สุดท้ายเป็น content
    block list พร้อม cache_control แทน plain string เฉยๆ — ดึง text ดิบออกมาเทียบใน test
    เหมือนเดิม"""
    content = kwargs["messages"][-1]["content"]
    if isinstance(content, str):
        return content
    return content[0]["text"]


def _fake_anthropic_tool_use_block(name, input_dict, block_id="tu_1"):
    block = MagicMock()
    block.type = "tool_use"
    block.name = name
    block.input = input_dict
    block.id = block_id
    return block


def _fake_anthropic_response(content_blocks, **usage_kwargs):
    """usage_kwargs: input_tokens/output_tokens/cache_creation_input_tokens/
    cache_read_input_tokens (default 0) — ต้อง set ครบทุกตัวเสมอ ไม่งั้น MagicMock
    auto-attribute จะรั่วเข้าไปแทนที่ int แล้วเทียบ TokenUsage ไม่ตรง"""
    response = MagicMock()
    response.content = content_blocks
    response.usage = MagicMock(
        input_tokens=usage_kwargs.get("input_tokens", 0),
        output_tokens=usage_kwargs.get("output_tokens", 0),
        cache_creation_input_tokens=usage_kwargs.get("cache_creation_input_tokens", 0),
        cache_read_input_tokens=usage_kwargs.get("cache_read_input_tokens", 0),
    )
    return response


# --- tool schema: NEEDS_CONFIRMATION action types ต้องอยู่ใน enum จริง ---
# (ก่อนหน้านี้ submit/delete/purchase/pay ไม่เคยอยู่ใน enum เลย ทำให้ permission
# layer's NEEDS_CONFIRMATION ไม่มีทาง trigger ผ่าน agent loop จริงได้เลย)


def test_browser_action_schema_includes_needs_confirmation_action_types():
    type_enum = llm._BROWSER_ACTION_PARAMS["properties"]["type"]["enum"]
    for risky_type in ("submit", "delete", "purchase", "pay"):
        assert risky_type in type_enum


# --- W43: completed_plan_step ต้อง optional เสมอ (ad-hoc task ไม่มีแผนต้องยังทำงานได้) ---


def test_browser_action_schema_has_completed_plan_step_property():
    assert "completed_plan_step" in llm._BROWSER_ACTION_PARAMS["properties"]
    assert llm._BROWSER_ACTION_PARAMS["properties"]["completed_plan_step"]["type"] == "integer"


def test_browser_action_schema_does_not_require_completed_plan_step():
    """ad-hoc task (ไม่ผ่าน Confirm plan) ต้องยังเรียก tool ได้ปกติโดยไม่ต้องระบุ
    completed_plan_step เลย — ห้ามอยู่ใน "required" เด็ดขาด"""
    assert "completed_plan_step" not in llm._BROWSER_ACTION_PARAMS["required"]


# --- read_page_data: อ่านเนื้อหาหน้าเว็บ (นับ/ตาราง) — Lane 1/2 (ดู actions.py) ---


def test_browser_action_schema_includes_read_page_data_type():
    type_enum = llm._BROWSER_ACTION_PARAMS["properties"]["type"]["enum"]
    assert "read_page_data" in type_enum


# --- W65[3] ("Vault Expansion — Current Password Auto-fill") ---


def test_browser_action_schema_includes_fill_secret_type():
    type_enum = llm._BROWSER_ACTION_PARAMS["properties"]["type"]["enum"]
    assert "fill_secret" in type_enum


def test_browser_action_schema_has_secret_property_restricted_to_current_password():
    props = llm._BROWSER_ACTION_PARAMS["properties"]
    assert props["secret"]["enum"] == ["current_password"]


def test_browser_action_schema_does_not_require_secret():
    """action ทั่วไปอื่นๆ ต้องยังเรียกได้ปกติโดยไม่ต้องมี secret เลย — มีความหมายเฉพาะตอน
    type="fill_secret" เท่านั้น"""
    assert "secret" not in llm._BROWSER_ACTION_PARAMS["required"]


def test_browser_action_schema_has_query_and_target_hint_properties():
    props = llm._BROWSER_ACTION_PARAMS["properties"]
    assert props["query"]["type"] == "string"
    assert props["target_hint"]["type"] == "string"


def test_browser_action_schema_does_not_require_query_or_target_hint():
    """ad-hoc action ทั่วไป (click/fill/...) ต้องยังเรียกได้ปกติโดยไม่ต้องมี query/
    target_hint เลย — ทั้งคู่มีความหมายเฉพาะตอน type="read_page_data" เท่านั้น"""
    required = llm._BROWSER_ACTION_PARAMS["required"]
    assert "query" not in required
    assert "target_hint" not in required


def test_system_prompt_instructs_read_page_data_and_favors_counting():
    assert "read_page_data" in llm.SYSTEM_PROMPT
    assert "target_hint" in llm.SYSTEM_PROMPT
    assert "always favour a direct count" in llm.SYSTEM_PROMPT


# --- W20 (Task12 follow-up, บั๊กจริงที่ user รายงาน): agent เผลอกรอกรหัสผ่านใหม่ซ้ำลงช่อง
# "Current Password" แทนที่จะเป็นรหัสผ่านจริงที่ user ใช้ล็อกอินอยู่ ทำให้ submit ล้มเหลวเสมอ ---


def test_system_prompt_forbids_reusing_new_password_in_current_password_field():
    assert "Current Password ≠ New Password" in llm.SYSTEM_PROMPT
    assert "NEVER type the new password into field (a)" in llm.SYSTEM_PROMPT
    assert 'call request_user_input (see W_resume below' in llm.SYSTEM_PROMPT


# --- W65[1]/[3] ("Required-Field Validation" / "Vault Expansion") ---


def test_system_prompt_requires_asking_for_missing_required_field_value():
    assert 'W65[1] ("Required-Field Validation")' in llm.SYSTEM_PROMPT
    assert '"[required]"' in llm.SYSTEM_PROMPT
    assert "call request_user_input (see W_resume below for full details)" in llm.SYSTEM_PROMPT
    assert "*** NEVER use finish_task(success=false) for this case ***" in llm.SYSTEM_PROMPT


# --- W_resume ("Mid-Task Input Request") — บั๊กจริงที่ user รายงาน: agent ขอรหัสผ่านใหม่
# กลางทางแล้วเรียก finish_task(false) จบ task ทั้งหมด ทำให้เทิร์นถัดไปที่ user ตอบค่ามา
# กลายเป็นเริ่มงานใหม่จากศูนย์แทนที่จะทำ plan เดิมต่อ


def test_system_prompt_documents_request_user_input_tool():
    assert 'W_resume ("Mid-Task Input Request")' in llm.SYSTEM_PROMPT
    assert "request_user_input(prompt, sensitive)" in llm.SYSTEM_PROMPT
    assert "Set sensitive: true when the value you're asking for is a password/secret" in llm.SYSTEM_PROMPT


def test_request_user_input_tool_registered_for_all_three_providers():
    assert llm.REQUEST_USER_INPUT_TOOL["name"] == "request_user_input"
    assert "prompt" in llm.REQUEST_USER_INPUT_TOOL["input_schema"]["properties"]
    assert "sensitive" in llm.REQUEST_USER_INPUT_TOOL["input_schema"]["properties"]
    assert llm.REQUEST_USER_INPUT_TOOL["input_schema"]["required"] == ["prompt"]

    groq_names = {t["function"]["name"] for t in llm._GROQ_TOOLS}
    assert "request_user_input" in groq_names

    gemini_names = {
        fn["name"] for tool in llm._GEMINI_TOOLS for fn in tool["function_declarations"]
    }
    assert "request_user_input" in gemini_names


def test_system_prompt_prefers_fill_secret_for_current_password_field():
    assert 'W65[3] ("Vault Expansion' in llm.SYSTEM_PROMPT
    assert '"fill_secret"' in llm.SYSTEM_PROMPT
    assert '"current_password"' in llm.SYSTEM_PROMPT


def test_plan_prompt_template_asks_for_missing_required_value():
    assert "Required-Field Check" in llm._PLAN_PROMPT_TEMPLATE
    assert "Required-Field Check before drafting the plan" in llm._PLAN_PROMPT_TEMPLATE


# --- W65[4] ("Structured Page-Grouped Plan Output") ---


def test_plan_prompt_template_instructs_page_grouped_format():
    assert "Page-Grouped Plan Format" in llm._PLAN_PROMPT_TEMPLATE
    assert '1. Login page: fill in Username, fill in Password, click Login' in llm._PLAN_PROMPT_TEMPLATE
    assert "Current Password" in llm._PLAN_PROMPT_TEMPLATE


# --- ป้องกัน agent ยอมแพ้เร็วเกินไป: ต้องลองค้นหาก่อนสรุปว่า "ไม่พบ" ---


def test_system_prompt_requires_trying_search_before_reporting_not_found():
    assert "before calling finish_task with a message along the lines of" in llm.SYSTEM_PROMPT
    assert "you must invoke an available action" in llm.SYSTEM_PROMPT
    assert "at least once before you may finish_task with \"not found\"" in llm.SYSTEM_PROMPT


def test_system_prompt_treats_verbless_questions_as_implicit_search_command():
    assert 'must NOT be read as' in llm.SYSTEM_PROMPT
    assert "counts as an implicit instruction to search for it" in llm.SYSTEM_PROMPT


# --- W19 ("Table Data Extractor & Presenter"): DOM order for multi-field rows, A-Z only
# for single-field lists — the two rules must coexist without contradicting each other ---


def test_system_prompt_preserves_dom_order_for_multi_field_table_rows():
    assert "the OPPOSITE rule applies — NEVER re-sort" in llm.SYSTEM_PROMPT
    assert "Always preserve the row order exactly as it appears on the real screen" in llm.SYSTEM_PROMPT


def test_system_prompt_still_sorts_single_field_lists_alphabetically():
    assert "For a plain list with only one field per entry" in llm.SYSTEM_PROMPT
    assert "always sort alphabetically (A-Z) before answering" in llm.SYSTEM_PROMPT


def test_system_prompt_forbids_splitting_row_fields_into_separate_lists():
    assert "Never split fields of the same row/entry into separate lists" in llm.SYSTEM_PROMPT


# --- W20 (Task11, "Response Formatter"): readable bullet/card list by default, raw markdown
# table only when the user explicitly asks for one ---


def test_system_prompt_shows_card_list_format_example_not_raw_table():
    assert "* **Admin**" in llm.SYSTEM_PROMPT
    assert "• Employee: Surya king" in llm.SYSTEM_PROMPT
    assert "N entries total" in llm.SYSTEM_PROMPT
    assert 'ONLY when the user literally typed "table" in their question' in llm.SYSTEM_PROMPT


def test_finish_task_schema_message_description_reflects_dom_order_rule():
    message_desc = llm.FINISH_TASK_TOOL["input_schema"]["properties"]["message"]["description"]
    assert "NEVER re-sort a table with multiple fields per row" in message_desc
    assert "never split the fields of one row apart" in message_desc


# --- hover: ปุ่ม hover-to-reveal ที่ perception.py ติด label marker ให้แล้ว ---


def test_browser_action_schema_includes_hover_type():
    type_enum = llm._BROWSER_ACTION_PARAMS["properties"]["type"]["enum"]
    assert "hover" in type_enum


def test_system_prompt_instructs_hover_before_clicking_hidden_reveal_elements():
    assert "[hidden — may need to hover the row first]" in llm.SYSTEM_PROMPT
    assert '"hover"' in llm.SYSTEM_PROMPT


# --- next_action() (Anthropic) — เทสต์ prompt caching wiring + parse tool_use ---


@pytest.mark.asyncio
async def test_next_action_sends_cache_control_on_system_and_tools():
    block = _fake_anthropic_tool_use_block("browser_action", {"type": "click", "index": 1})
    response = _fake_anthropic_response([block], input_tokens=20, output_tokens=8)

    client = MagicMock()
    client.messages.create = AsyncMock(return_value=response)

    tool_name, tool_input, tool_use_id, messages, usage = await llm.next_action(
        client, "claude-haiku-4-5-20251001", "goal", "[0] button", []
    )

    assert tool_name == "browser_action"
    assert tool_input == {"type": "click", "index": 1}
    assert tool_use_id == "tu_1"
    assert usage == llm.TokenUsage(input_tokens=20, output_tokens=8)

    _, kwargs = client.messages.create.call_args
    assert kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert kwargs["tools"][-1]["cache_control"] == {"type": "ephemeral"}


@pytest.mark.asyncio
async def test_next_action_sends_cache_control_on_last_conversation_message():
    """SPD-1: breakpoint ที่สอง (แยกจาก system+tools) ต้องอยู่บน user message ล่าสุดที่ส่ง
    ไปจริงในแต่ละ request — แต่ messages ที่ return กลับมาให้ loop เก็บต่อ (ใช้สร้าง
    request ของรอบถัดไป) ต้องยังเป็น plain string เหมือนเดิม ไม่ค้าง cache_control สะสม
    (ไม่งั้นเกิน 4 breakpoints ที่ Anthropic อนุญาตต่อ request หลังผ่านไปหลาย step)"""
    block = _fake_anthropic_tool_use_block("browser_action", {"type": "click", "index": 1})
    response = _fake_anthropic_response([block])
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=response)

    _, _, _, messages, _ = await llm.next_action(client, "model", "goal", "page", [])

    _, kwargs = client.messages.create.call_args
    sent_last = kwargs["messages"][-1]["content"]
    assert isinstance(sent_last, list)
    assert sent_last[-1]["cache_control"] == {"type": "ephemeral"}
    assert sent_last[0]["type"] == "text"

    # messages ที่เก็บไว้ต่อ (ไม่ใช่ตัวที่ส่งจริง) ต้องยังเป็น string เดิม ไม่มี cache_control ปน
    assert isinstance(messages[0]["content"], str)


@pytest.mark.asyncio
async def test_next_action_second_call_only_marks_its_own_last_message():
    """SPD-1: เรียก next_action() 2 รอบติดกัน (จำลอง step ถัดไปของ loop เดียวกัน) —
    request ของรอบที่ 2 ต้องมี cache_control อยู่แค่บน user message ล่าสุดของรอบนั้นเท่านั้น
    ไม่ใช่ค้างอยู่บน message เก่าจากรอบแรกด้วย (กัน breakpoint สะสมเกิน 4 อันที่ Anthropic
    อนุญาตต่อ request เมื่อ task มีหลาย step)"""
    block = _fake_anthropic_tool_use_block("browser_action", {"type": "click", "index": 1})
    response = _fake_anthropic_response([block])
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=response)

    _, _, tool_use_id, messages, _ = await llm.next_action(client, "model", "goal", "page 1", [])
    messages = llm.append_tool_result(messages, tool_use_id, "[OK] click")
    await llm.next_action(client, "model", "goal", "page 2", messages)

    _, kwargs = client.messages.create.call_args
    sent_messages = kwargs["messages"]
    cache_marked = [
        m for m in sent_messages
        if isinstance(m.get("content"), list)
        and any(isinstance(b, dict) and "cache_control" in b for b in m["content"])
    ]
    assert len(cache_marked) == 1
    assert cache_marked[0] is sent_messages[-1]


@pytest.mark.asyncio
async def test_next_action_extracts_cache_read_and_creation_tokens():
    block = _fake_anthropic_tool_use_block("finish_task", {"success": True, "message": "done"})
    response = _fake_anthropic_response(
        [block], input_tokens=5, output_tokens=3, cache_creation_input_tokens=0, cache_read_input_tokens=500
    )
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=response)

    _, _, _, _, usage = await llm.next_action(client, "model", "goal", "page", [])

    assert usage == llm.TokenUsage(input_tokens=5, output_tokens=3, cache_creation_tokens=0, cache_read_tokens=500)


@pytest.mark.asyncio
async def test_next_action_falls_back_to_finish_task_when_no_tool_use_block():
    text_block = MagicMock()
    text_block.type = "text"
    response = _fake_anthropic_response([text_block], input_tokens=5, output_tokens=3)
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=response)

    tool_name, tool_input, tool_use_id, messages, usage = await llm.next_action(client, "model", "goal", "page", [])

    # W_notoolcall: ไม่ยอมแพ้ตั้งแต่ครั้งแรกอีกต่อไป — เตือนแล้วลองใหม่จนครบโควตาก่อน
    assert client.messages.create.await_count == llm._NO_TOOL_CALL_RETRIES
    assert tool_name == "finish_task"
    assert tool_input["success"] is False
    assert str(llm._NO_TOOL_CALL_RETRIES) in tool_input["message"]
    assert tool_use_id == ""
    # usage ของทุกรอบต้องถูกรวม ไม่ใช่รายงานแค่รอบสุดท้าย
    assert usage == llm.TokenUsage(
        input_tokens=5 * llm._NO_TOOL_CALL_RETRIES, output_tokens=3 * llm._NO_TOOL_CALL_RETRIES,
        # W_token_cut W1: เตือนครบโควตาแล้วยังไม่ได้ tool call = retry ไปเต็มจำนวน
        notool_retries=llm._NO_TOOL_CALL_RETRIES - 1,
    )
    assert any(
        m.get("content") == llm._NO_TOOL_CALL_NUDGE for m in messages if isinstance(m, dict)
    )


@pytest.mark.asyncio
async def test_next_action_retries_after_no_tool_call_then_accepts_second_attempt():
    """W_notoolcall: การตอบเป็นข้อความธรรมดา 1 ครั้งต้องไม่ฆ่า task — รอบถัดไปที่เรียก tool
    จริงต้องถูกใช้งานตามปกติ (นี่คือเหตุผลหลักที่ retry นี้มีอยู่)"""
    text_block = MagicMock()
    text_block.type = "text"
    bad = _fake_anthropic_response([text_block], input_tokens=5, output_tokens=3)
    good = _fake_anthropic_response(
        [_fake_anthropic_tool_use_block("browser_action", {"type": "wait"})],
        input_tokens=7, output_tokens=2,
    )
    client = MagicMock()
    client.messages.create = AsyncMock(side_effect=[bad, good])

    tool_name, tool_input, _, _, usage = await llm.next_action(client, "model", "goal", "page", [])

    assert client.messages.create.await_count == 2
    assert tool_name == "browser_action"
    assert tool_input == {"type": "wait"}
    assert usage == llm.TokenUsage(input_tokens=12, output_tokens=5, notool_retries=1)  # W_token_cut W1


@pytest.mark.asyncio
async def test_next_action_passes_manual_context_into_prompt():
    """W6[B]: manual_context จาก retriever.retrieve() ต้องโผล่ในข้อความ user turn จริง"""
    block = _fake_anthropic_tool_use_block("browser_action", {"type": "wait"})
    response = _fake_anthropic_response([block])
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=response)

    await llm.next_action(client, "model", "goal", "page", [], manual_context="- chunk one")

    _, kwargs = client.messages.create.call_args
    user_content = _last_user_text(kwargs)
    assert "chunk one" in user_content
    assert "Reference information from the relevant manual" in user_content


@pytest.mark.asyncio
async def test_next_action_default_manual_context_omits_section():
    """เรียกแบบเดิม (5 args ไม่มี manual_context) ต้องได้ prompt แบบเดิมเป๊ะ ไม่มี section คู่มือ"""
    block = _fake_anthropic_tool_use_block("browser_action", {"type": "wait"})
    response = _fake_anthropic_response([block])
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=response)

    await llm.next_action(client, "model", "goal", "page", [])

    _, kwargs = client.messages.create.call_args
    user_content = _last_user_text(kwargs)
    assert "คู่มือ" not in user_content


@pytest.mark.asyncio
async def test_next_action_passes_memory_context_into_prompt():
    """W7[A]: memory_context จาก ShortTermMemory.failed_actions_summary() ต้องโผล่ในข้อความ user turn จริง"""
    block = _fake_anthropic_tool_use_block("browser_action", {"type": "wait"})
    response = _fake_anthropic_response([block])
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=response)

    await llm.next_action(
        client, "model", "goal", "page", [], manual_context="", memory_context="- {'type': 'click'} -> [FAIL] boom"
    )

    _, kwargs = client.messages.create.call_args
    user_content = _last_user_text(kwargs)
    assert "[FAIL] boom" in user_content
    assert "Actions already tried that failed" in user_content


@pytest.mark.asyncio
async def test_next_action_default_memory_context_omits_section():
    """เรียกแบบเดิม (ไม่มี memory_context) ต้องได้ prompt แบบเดิมเป๊ะ ไม่มี section ประวัติ failure"""
    block = _fake_anthropic_tool_use_block("browser_action", {"type": "wait"})
    response = _fake_anthropic_response([block])
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=response)

    await llm.next_action(client, "model", "goal", "page", [])

    _, kwargs = client.messages.create.call_args
    user_content = _last_user_text(kwargs)
    assert "ทำซ้ำ" not in user_content


@pytest.mark.asyncio
async def test_next_action_passes_plan_context_into_prompt():
    """W43: plan_context (แผนที่ user ยืนยันแล้ว) ต้องโผล่ในข้อความ user turn จริง เป็น
    section แยก "Current plan confirmed by the user" """
    block = _fake_anthropic_tool_use_block("browser_action", {"type": "wait"})
    response = _fake_anthropic_response([block])
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=response)

    await llm.next_action(client, "model", "goal", "page", [], plan_context="1. ทำ X\n2. ทำ Y")

    _, kwargs = client.messages.create.call_args
    user_content = _last_user_text(kwargs)
    assert "1. ทำ X" in user_content
    assert "Current plan confirmed by the user" in user_content


@pytest.mark.asyncio
async def test_next_action_default_plan_context_omits_section():
    """W43: ad-hoc task (ไม่ผ่าน Confirm plan เลย) ไม่ควรมี section "Current plan confirmed by the user" โผล่มา
    ปนใน prompt เลย — backward compatible กับ task ที่ไม่มีแผน"""
    block = _fake_anthropic_tool_use_block("browser_action", {"type": "wait"})
    response = _fake_anthropic_response([block])
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=response)

    await llm.next_action(client, "model", "goal", "page", [])

    _, kwargs = client.messages.create.call_args
    user_content = _last_user_text(kwargs)
    assert "Current plan confirmed by the user" not in user_content


@pytest.mark.asyncio
async def test_next_action_tool_input_includes_completed_plan_step_when_llm_provides_it():
    """W43: LLM ใส่ completed_plan_step มาใน tool call — ต้อง parse ผ่านมาใน tool_input
    เฉยๆ (ไม่มี logic พิเศษฝั่ง llm.py เลย คืนค่า tool_use.input ดิบๆ เหมือนเดิม)"""
    block = _fake_anthropic_tool_use_block(
        "browser_action", {"type": "click", "index": 2, "completed_plan_step": 1},
    )
    response = _fake_anthropic_response([block])
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=response)

    _, tool_input, _, _, _ = await llm.next_action(client, "model", "goal", "page", [])

    assert tool_input.get("completed_plan_step") == 1


@pytest.mark.asyncio
async def test_next_action_tool_input_completed_plan_step_is_none_when_llm_omits_it():
    """W43: LLM ไม่ใส่ completed_plan_step มาเลย (ไม่มีแผนให้ทำตาม/action นี้ยังไม่ทำให้
    step ไหนเสร็จ) — ต้องไม่ throw เลย แค่ .get() คืน None ตามปกติ"""
    block = _fake_anthropic_tool_use_block("browser_action", {"type": "click", "index": 2})
    response = _fake_anthropic_response([block])
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=response)

    _, tool_input, _, _, _ = await llm.next_action(client, "model", "goal", "page", [])

    assert tool_input.get("completed_plan_step") is None


def _fake_gemini_function_call_part(name: str, args: dict):
    part = MagicMock()
    part.function_call.name = name
    part.function_call.args = args  # plain dict ก็ dict(...) ได้เหมือน MapComposite จริง
    return part


def _fake_gemini_text_only_part():
    part = MagicMock()
    part.function_call.name = ""  # falsy -> ไม่นับว่ามี function call
    return part


def _fake_gemini_response(parts, prompt_tokens=10, candidates_tokens=5):
    content = MagicMock()
    content.parts = parts
    candidate = MagicMock()
    candidate.content = content
    response = MagicMock()
    response.candidates = [candidate]
    response.usage_metadata.prompt_token_count = prompt_tokens
    response.usage_metadata.candidates_token_count = candidates_tokens
    return response


def _fake_gemini_client(response):
    """จำลอง genai module: client.GenerativeModel(...) -> model ที่มี
    generate_content_async() คืน response ที่กำหนด"""
    gemini_model = MagicMock()
    gemini_model.generate_content_async = AsyncMock(return_value=response)
    client = MagicMock()
    client.GenerativeModel = MagicMock(return_value=gemini_model)
    return client, gemini_model


# --- next_action_groq() (Groq) ---


@pytest.mark.asyncio
async def test_next_action_groq_returns_parsed_tool_call():
    tool_call = _fake_tool_call("call_1", "browser_action", '{"type": "click", "index": 2}')
    dumped = {"role": "assistant", "tool_calls": [{"id": "call_1"}]}
    response = _fake_response([tool_call], dumped)

    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=response)

    tool_name, tool_input, tool_use_id, messages, usage = await llm.next_action_groq(
        client, "llama-3.3-70b-versatile", "goal", "[0] button", []
    )

    assert tool_name == "browser_action"
    assert tool_input == {"type": "click", "index": 2}
    assert tool_use_id == "call_1"
    # ส่ง [] เข้าไป (เทิร์นแรก) -> ต้องแทรก system prompt ไว้หน้าสุด
    # W_token_cut W2: system = _PROMPT_CORE คงที่ (บล็อกที่ gate ย้ายไป user turn)
    assert messages[0] == {"role": "system", "content": llm._PROMPT_CORE}
    assert messages[-1] == dumped
    assert usage == llm.TokenUsage(input_tokens=10, output_tokens=5)


@pytest.mark.asyncio
async def test_next_action_groq_does_not_prepend_system_prompt_twice():
    tool_call = _fake_tool_call("call_2", "finish_task", '{"success": true, "message": "done"}')
    response = _fake_response([tool_call], {"role": "assistant"})
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=response)

    existing_messages = [
        {"role": "system", "content": llm.SYSTEM_PROMPT},
        {"role": "user", "content": "prev turn"},
    ]
    _, _, _, messages, _ = await llm.next_action_groq(client, "model", "goal", "page", existing_messages)

    assert sum(1 for m in messages if m.get("role") == "system") == 1


@pytest.mark.asyncio
async def test_next_action_groq_nudges_and_retries_when_no_tool_calls_then_succeeds():
    """ถ้า Llama ตอบเป็นข้อความเฉยๆ ไม่เรียก tool ห้าม finish_task ทันที ต้องเตือนแล้วลองใหม่ก่อน"""
    no_tool_response = _fake_response([], {"role": "assistant", "content": "just text, no tool call"})
    tool_call = _fake_tool_call("call_4", "browser_action", '{"type": "wait"}')
    good_response = _fake_response([tool_call], {"role": "assistant"})

    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=[no_tool_response, good_response])

    tool_name, tool_input, _, messages, usage = await llm.next_action_groq(
        client, "model", "goal", "page", []
    )

    assert tool_name == "browser_action"
    assert tool_input == {"type": "wait"}
    assert client.chat.completions.create.await_count == 2
    # ต้องมีข้อความเตือนแทรกอยู่ในบทสนทนาก่อนลองรอบถัดไป
    assert any(m.get("content") == llm._NO_TOOL_CALL_NUDGE for m in messages)
    # usage ต้องรวมทั้ง 2 request (รอบที่ไม่เรียก tool + รอบที่เรียกสำเร็จ) ไม่ใช่แค่รอบสุดท้าย
    assert usage == llm.TokenUsage(input_tokens=20, output_tokens=10, notool_retries=1)  # W_token_cut W1


@pytest.mark.asyncio
async def test_next_action_groq_falls_back_to_finish_task_after_no_tool_call_retries_exhausted():
    response = _fake_response([], {"role": "assistant", "content": "just text, no tool call"})
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=response)

    tool_name, tool_input, tool_use_id, _, usage = await llm.next_action_groq(client, "model", "goal", "page", [])

    assert tool_name == "finish_task"
    assert tool_input["success"] is False
    assert tool_use_id == ""
    assert client.chat.completions.create.await_count == llm._GROQ_NO_TOOL_CALL_RETRIES
    # usage ต้องรวมทุกรอบที่ยิงจริง แม้จะไม่มีรอบไหนเรียก tool สำเร็จเลย
    assert usage == llm.TokenUsage(
        input_tokens=10 * llm._GROQ_NO_TOOL_CALL_RETRIES, output_tokens=5 * llm._GROQ_NO_TOOL_CALL_RETRIES,
        notool_retries=llm._GROQ_NO_TOOL_CALL_RETRIES - 1,  # W_token_cut W1
    )


@pytest.mark.asyncio
async def test_next_action_groq_retries_on_tool_use_failed_then_succeeds():
    """Llama บน Groq บางครั้ง generate tool call ผิดรูปแบบ (400 tool_use_failed) —
    เป็นเรื่อง sampling แบบสุ่ม ยิงซ้ำมักผ่าน ต้องไม่ throw ตั้งแต่ครั้งแรก"""
    tool_call = _fake_tool_call("call_3", "browser_action", '{"type": "wait"}')
    good_response = _fake_response([tool_call], {"role": "assistant"})

    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        side_effect=[_fake_bad_request("tool_use_failed"), good_response]
    )

    tool_name, tool_input, _, _, usage = await llm.next_action_groq(client, "model", "goal", "page", [])

    assert tool_name == "browser_action"
    assert tool_input == {"type": "wait"}
    assert client.chat.completions.create.await_count == 2
    # request ที่ raise (tool_use_failed) ไม่มี response.usage ให้นับ — ต้องนับแค่รอบที่สำเร็จ
    assert usage == llm.TokenUsage(input_tokens=10, output_tokens=5)


@pytest.mark.asyncio
async def test_next_action_groq_gives_up_after_max_retries():
    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=_fake_bad_request("tool_use_failed"))

    with pytest.raises(GroqBadRequestError):
        await llm.next_action_groq(client, "model", "goal", "page", [])

    assert client.chat.completions.create.await_count == llm._GROQ_TOOL_CALL_RETRIES


@pytest.mark.asyncio
async def test_next_action_groq_reraises_other_bad_request_errors_immediately():
    """เฉพาะ tool_use_failed เท่านั้นที่ควรลองซ้ำ — error อื่นๆ (เช่น invalid model,
    rate limit) ต้อง raise ออกไปทันที ไม่ควรลองซ้ำแบบไม่มีความหมาย"""
    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=_fake_bad_request("some_other_error"))

    with pytest.raises(GroqBadRequestError):
        await llm.next_action_groq(client, "model", "goal", "page", [])

    assert client.chat.completions.create.await_count == 1


def test_append_tool_result_groq_formats_as_tool_role_message():
    messages = [{"role": "user", "content": "x"}]
    result = llm.append_tool_result_groq(messages, "call_1", "[OK] clicked")

    assert result[-1] == {"role": "tool", "tool_call_id": "call_1", "content": "[OK] clicked"}
    assert result[:-1] == messages
    assert result is not messages  # ไม่แก้ list เดิม (immutable-style เหมือน append_tool_result ของ Anthropic)


@pytest.mark.asyncio
async def test_next_action_groq_passes_manual_context_into_prompt():
    tool_call = _fake_tool_call("call_5", "browser_action", '{"type": "wait"}')
    response = _fake_response([tool_call], {"role": "assistant"})
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=response)

    await llm.next_action_groq(client, "model", "goal", "page", [], manual_context="- chunk one")

    _, kwargs = client.chat.completions.create.call_args
    user_content = kwargs["messages"][-1]["content"]
    assert "chunk one" in user_content


@pytest.mark.asyncio
async def test_next_action_groq_default_manual_context_omits_section():
    tool_call = _fake_tool_call("call_6", "browser_action", '{"type": "wait"}')
    response = _fake_response([tool_call], {"role": "assistant"})
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=response)

    await llm.next_action_groq(client, "model", "goal", "page", [])

    _, kwargs = client.chat.completions.create.call_args
    user_content = kwargs["messages"][-1]["content"]
    assert "คู่มือ" not in user_content


@pytest.mark.asyncio
async def test_next_action_groq_passes_memory_context_into_prompt():
    tool_call = _fake_tool_call("call_7", "browser_action", '{"type": "wait"}')
    response = _fake_response([tool_call], {"role": "assistant"})
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=response)

    await llm.next_action_groq(client, "model", "goal", "page", [], memory_context="- fail one")

    _, kwargs = client.chat.completions.create.call_args
    user_content = kwargs["messages"][-1]["content"]
    assert "fail one" in user_content


@pytest.mark.asyncio
async def test_next_action_groq_default_memory_context_omits_section():
    tool_call = _fake_tool_call("call_8", "browser_action", '{"type": "wait"}')
    response = _fake_response([tool_call], {"role": "assistant"})
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=response)

    await llm.next_action_groq(client, "model", "goal", "page", [])

    _, kwargs = client.chat.completions.create.call_args
    user_content = kwargs["messages"][-1]["content"]
    assert "ทำซ้ำ" not in user_content


# --- next_action_gemini() (Gemini) ---


@pytest.mark.asyncio
async def test_next_action_gemini_returns_parsed_function_call():
    part = _fake_gemini_function_call_part("browser_action", {"type": "click", "index": 2})
    response = _fake_gemini_response([part], prompt_tokens=10, candidates_tokens=5)
    client, gemini_model = _fake_gemini_client(response)

    tool_name, tool_input, tool_use_id, messages, usage = await llm.next_action_gemini(
        client, "gemini-flash-lite-latest", "goal", "[0] button", []
    )

    assert tool_name == "browser_action"
    assert tool_input == {"type": "click", "index": 2}
    # Gemini ไม่มี call id จริง — ใช้ชื่อ function เป็น tool_use_id แทน
    assert tool_use_id == "browser_action"
    assert usage == llm.TokenUsage(input_tokens=10, output_tokens=5)
    # model ต้องถูกสร้างด้วย tools + tool_config บังคับเรียก function เสมอ
    _, kwargs = client.GenerativeModel.call_args
    assert kwargs["tools"] == llm._GEMINI_TOOLS
    assert kwargs["tool_config"] == {"function_calling_config": {"mode": "ANY"}}
    assert kwargs["system_instruction"] == llm._PROMPT_CORE  # W_token_cut W2
    # messages ต้องมี user turn ใหม่ + model turn (response content) ต่อท้าย
    assert messages[-2]["role"] == "user"
    assert messages[-1] is response.candidates[0].content
    # ตอนยิง generate_content_async จริง ต้องยังไม่มี model turn (เพิ่งได้ response กลับมา
    # ถึงจะรู้ว่าโมเดลตอบอะไร) — ส่งแค่ user turn ใหม่เข้าไปตอนเรียก
    called_contents = gemini_model.generate_content_async.call_args.kwargs["contents"]
    assert called_contents == messages[:-1]


@pytest.mark.asyncio
async def test_next_action_gemini_normalizes_whole_number_floats_to_int():
    """Gemini คืนตัวเลขเป็น float เสมอผ่าน protobuf Struct แม้ schema จะเป็น integer —
    ต้องแปลงกลับเป็น int ไม่งั้น selector index="2.0" จะไม่ตรงกับ element จริง"""
    part = _fake_gemini_function_call_part("browser_action", {"type": "click", "index": 2.0})
    response = _fake_gemini_response([part])
    client, _ = _fake_gemini_client(response)

    _, tool_input, _, _, _ = await llm.next_action_gemini(client, "model", "goal", "page", [])

    assert tool_input["index"] == 2
    assert isinstance(tool_input["index"], int)


@pytest.mark.asyncio
async def test_next_action_gemini_falls_back_to_finish_task_when_no_function_call():
    part = _fake_gemini_text_only_part()
    response = _fake_gemini_response([part])
    client, _ = _fake_gemini_client(response)

    tool_name, tool_input, tool_use_id, _, usage = await llm.next_action_gemini(
        client, "model", "goal", "page", []
    )

    # W_notoolcall: เตือนแล้วลองใหม่จนครบโควตาก่อนยอมแพ้ (เหมือนทุก provider)
    assert tool_name == "finish_task"
    assert tool_input["success"] is False
    assert str(llm._NO_TOOL_CALL_RETRIES) in tool_input["message"]
    assert tool_use_id == ""
    assert usage == llm.TokenUsage(
        input_tokens=10 * llm._NO_TOOL_CALL_RETRIES, output_tokens=5 * llm._NO_TOOL_CALL_RETRIES,
        notool_retries=llm._NO_TOOL_CALL_RETRIES - 1,  # W_token_cut W1
    )


@pytest.mark.asyncio
async def test_next_action_gemini_retries_on_resource_exhausted_then_succeeds(monkeypatch):
    """429 ResourceExhausted (quota เต็ม) ต้องไม่ crash ทั้ง process — หน่วงแล้วลองใหม่ก่อน"""
    part = _fake_gemini_function_call_part("browser_action", {"type": "wait"})
    good_response = _fake_gemini_response([part])
    client, gemini_model = _fake_gemini_client(good_response)
    gemini_model.generate_content_async = AsyncMock(
        side_effect=[ResourceExhausted("quota exceeded"), good_response]
    )
    sleep_mock = AsyncMock()
    monkeypatch.setattr(llm.asyncio, "sleep", sleep_mock)

    tool_name, tool_input, _, _, _ = await llm.next_action_gemini(client, "model", "goal", "page", [])

    assert tool_name == "browser_action"
    assert tool_input == {"type": "wait"}
    assert gemini_model.generate_content_async.await_count == 2
    sleep_mock.assert_awaited_once_with(llm._GEMINI_RATE_LIMIT_BACKOFF_SECONDS * 1)


@pytest.mark.asyncio
async def test_next_action_gemini_reraises_resource_exhausted_after_max_retries(monkeypatch):
    """ยังโดน 429 อยู่แม้ retry ครบแล้ว -> ต้อง raise ออกไปจริง ไม่ปล่อยให้วนไม่รู้จบ"""
    client, gemini_model = _fake_gemini_client(None)
    gemini_model.generate_content_async = AsyncMock(side_effect=ResourceExhausted("quota exceeded"))
    monkeypatch.setattr(llm.asyncio, "sleep", AsyncMock())

    with pytest.raises(ResourceExhausted):
        await llm.next_action_gemini(client, "model", "goal", "page", [])

    assert gemini_model.generate_content_async.await_count == llm._GEMINI_RATE_LIMIT_RETRIES


def test_normalize_gemini_args_only_converts_whole_number_floats():
    result = llm._normalize_gemini_args({"index": 3.0, "text": "hello", "ratio": 1.5, "flag": True})

    assert result == {"index": 3, "text": "hello", "ratio": 1.5, "flag": True}
    assert isinstance(result["index"], int)


def test_append_tool_result_gemini_formats_as_function_response_part():
    messages = [{"role": "user", "parts": [{"text": "x"}]}]
    result = llm.append_tool_result_gemini(messages, "browser_action", "[OK] clicked")

    assert result[-1] == {
        "role": "user",
        "parts": [{"function_response": {"name": "browser_action", "response": {"result": "[OK] clicked"}}}],
    }
    assert result[:-1] == messages
    assert result is not messages


@pytest.mark.asyncio
async def test_next_action_gemini_passes_manual_context_into_prompt():
    part = _fake_gemini_function_call_part("browser_action", {"type": "wait"})
    response = _fake_gemini_response([part])
    client, gemini_model = _fake_gemini_client(response)

    await llm.next_action_gemini(client, "model", "goal", "page", [], manual_context="- chunk one")

    _, kwargs = gemini_model.generate_content_async.call_args
    user_text = kwargs["contents"][-1]["parts"][0]["text"]
    assert "chunk one" in user_text


@pytest.mark.asyncio
async def test_next_action_gemini_default_manual_context_omits_section():
    part = _fake_gemini_function_call_part("browser_action", {"type": "wait"})
    response = _fake_gemini_response([part])
    client, gemini_model = _fake_gemini_client(response)

    await llm.next_action_gemini(client, "model", "goal", "page", [])

    _, kwargs = gemini_model.generate_content_async.call_args
    user_text = kwargs["contents"][-1]["parts"][0]["text"]
    assert "คู่มือ" not in user_text


@pytest.mark.asyncio
async def test_next_action_gemini_passes_memory_context_into_prompt():
    part = _fake_gemini_function_call_part("browser_action", {"type": "wait"})
    response = _fake_gemini_response([part])
    client, gemini_model = _fake_gemini_client(response)

    await llm.next_action_gemini(client, "model", "goal", "page", [], memory_context="- fail one")

    _, kwargs = gemini_model.generate_content_async.call_args
    user_text = kwargs["contents"][-1]["parts"][0]["text"]
    assert "fail one" in user_text


@pytest.mark.asyncio
async def test_next_action_gemini_default_memory_context_omits_section():
    part = _fake_gemini_function_call_part("browser_action", {"type": "wait"})
    response = _fake_gemini_response([part])
    client, gemini_model = _fake_gemini_client(response)

    await llm.next_action_gemini(client, "model", "goal", "page", [])

    _, kwargs = gemini_model.generate_content_async.call_args
    user_text = kwargs["contents"][-1]["parts"][0]["text"]
    assert "ทำซ้ำ" not in user_text


@pytest.mark.asyncio
async def test_next_action_gemini_passes_vision_context_into_prompt():
    """W9[A]: vision_context (คำอธิบายจาก describe_screenshot()) ต้องโผล่ใน prompt
    จริงเมื่อส่งมา"""
    part = _fake_gemini_function_call_part("browser_action", {"type": "wait"})
    response = _fake_gemini_response([part])
    client, gemini_model = _fake_gemini_client(response)

    await llm.next_action_gemini(
        client, "model", "goal", "page", [], vision_context="เห็น cookie banner บังปุ่มอยู่"
    )

    _, kwargs = gemini_model.generate_content_async.call_args
    user_text = kwargs["contents"][-1]["parts"][0]["text"]
    assert "เห็น cookie banner บังปุ่มอยู่" in user_text


@pytest.mark.asyncio
async def test_next_action_gemini_default_vision_context_omits_section():
    part = _fake_gemini_function_call_part("browser_action", {"type": "wait"})
    response = _fake_gemini_response([part])
    client, gemini_model = _fake_gemini_client(response)

    await llm.next_action_gemini(client, "model", "goal", "page", [])

    _, kwargs = gemini_model.generate_content_async.call_args
    user_text = kwargs["contents"][-1]["parts"][0]["text"]
    assert "ภาพหน้าจอ" not in user_text


# --- describe_screenshot() (W9[A] vision fallback, Gemini เท่านั้นตอนนี้) ---


@pytest.mark.asyncio
async def test_describe_screenshot_returns_stripped_text():
    response = MagicMock()
    response.text = "  เห็น cookie banner บังปุ่ม Login อยู่ ลองปิด banner ก่อน  "
    client, gemini_model = _fake_gemini_client(response)

    result = await llm.describe_screenshot(client, "model", b"fakepngbytes", "click", 5)

    assert result == "เห็น cookie banner บังปุ่ม Login อยู่ ลองปิด banner ก่อน"
    _, kwargs = gemini_model.generate_content_async.call_args
    parts = kwargs["contents"][0]["parts"]
    assert parts[1] == {"mime_type": "image/png", "data": b"fakepngbytes"}
    assert "click" in parts[0]["text"]
    assert "5" in parts[0]["text"]


@pytest.mark.asyncio
async def test_describe_screenshot_returns_empty_string_on_error_without_throwing():
    """กฎเหล็ก: ห้าม throw ออกไปเด็ดขาด (เหมือน retriever.retrieve()) ถ้า vision call
    พังเอง (เช่น quota/network) ต้องไม่ทำให้ agent loop หลักพังตาม"""
    client = MagicMock()
    client.GenerativeModel = MagicMock(side_effect=Exception("quota exceeded"))

    result = await llm.describe_screenshot(client, "model", b"fakepngbytes", "click", 5)

    assert result == ""


# --- _build_user_turn_text() (W6[B]/W7[A]) ---

# W?: เวลาปัจจุบันถูกฉีดเข้าทุก turn (ดู _current_bangkok_time_text() ใน llm.py) — เทสต์
# กลุ่ม backward-compat ด้านล่างต้องรู้ค่าที่แน่นอนถึงจะ assert exact-match ได้ ใช้ fixture
# นี้ freeze ค่าไว้แทนการเรียกเวลาจริงทุกเทสต์ (เทสต์เฉพาะของ datetime injection เองอยู่ใน
# ท้ายไฟล์ — ตรงนั้น mock datetime.now() ตรงๆ แทน)
_FIXED_TIME_TEXT = "Friday, 31 July 2026 at 14:32"
_TIME_LINE = f"\n\nCurrent time (Asia/Bangkok): {_FIXED_TIME_TEXT}"
_EXPECTED_PREFIX = f"Goal: goal{_TIME_LINE}\n\nCurrent page:\npage"


@pytest.fixture(autouse=True)
def _freeze_bangkok_time(monkeypatch):
    monkeypatch.setattr(llm, "_current_bangkok_time_text", lambda: _FIXED_TIME_TEXT)


def test_build_user_turn_text_omits_manual_section_when_empty():
    result = llm._build_user_turn_text("goal", "page")

    assert result == _EXPECTED_PREFIX


def test_build_user_turn_text_includes_manual_section_when_provided():
    result = llm._build_user_turn_text("goal", "page", "- chunk one\n- chunk two")

    assert result.startswith(_EXPECTED_PREFIX)
    assert "chunk one" in result
    assert "chunk two" in result


def test_build_user_turn_text_omits_memory_section_when_empty():
    result = llm._build_user_turn_text("goal", "page", manual_context="", memory_context="")

    assert result == _EXPECTED_PREFIX


def test_build_user_turn_text_includes_memory_section_when_provided():
    result = llm._build_user_turn_text("goal", "page", memory_context="- {'type': 'click'} -> [FAIL] boom")

    assert result.startswith(_EXPECTED_PREFIX)
    assert "[FAIL] boom" in result
    assert "Actions already tried that failed" in result


def test_build_user_turn_text_includes_both_manual_and_memory_sections():
    result = llm._build_user_turn_text(
        "goal", "page", manual_context="- chunk one", memory_context="- fail one"
    )

    assert "chunk one" in result
    assert "fail one" in result


def test_build_user_turn_text_omits_vision_section_when_empty():
    result = llm._build_user_turn_text("goal", "page", vision_context="")

    assert result == _EXPECTED_PREFIX


def test_build_user_turn_text_includes_vision_section_when_provided():
    result = llm._build_user_turn_text("goal", "page", vision_context="เห็น cookie banner บังปุ่ม Login อยู่")

    assert result.startswith(_EXPECTED_PREFIX)
    assert "เห็น cookie banner บังปุ่ม Login อยู่" in result


def test_build_user_turn_text_omits_current_url_section_when_empty():
    result = llm._build_user_turn_text("goal", "page", current_url="")

    assert result == _EXPECTED_PREFIX


def test_build_user_turn_text_includes_current_url_before_page_text():
    """W30: URL ต้องโผล่ก่อนส่วน 'หน้าเว็บปัจจุบัน' (indexed elements) เสมอ ให้โมเดลเห็น
    บริบทว่ากำลังอยู่หน้าไหนก่อนจะเห็นรายละเอียด element"""
    result = llm._build_user_turn_text("goal", "page-text", current_url="https://example.com/cart")

    assert "https://example.com/cart" in result
    assert result.index("https://example.com/cart") < result.index("Current page:\npage-text")


def test_build_user_turn_text_omits_action_history_section_when_empty():
    result = llm._build_user_turn_text("goal", "page", action_history_context="")

    assert result == _EXPECTED_PREFIX


def test_build_user_turn_text_includes_action_history_section_when_provided():
    result = llm._build_user_turn_text(
        "goal", "page", action_history_context="- step 3: {'type': 'click'} -> [OK]",
    )

    assert result.startswith(_EXPECTED_PREFIX)
    assert "step 3" in result
    assert "The most recent actions you just performed" in result


# --- W43: plan_context ("Current plan confirmed by the user") ---


def test_build_user_turn_text_omits_plan_section_when_empty():
    """ad-hoc task (ไม่ผ่าน Confirm plan) ได้ plan_context="" เสมอ — ต้องได้ prompt เดิม
    เป๊ะทุกตัวอักษร ไม่มี section แผนโผล่มาปนเลย (backward compatible)"""
    result = llm._build_user_turn_text("goal", "page", plan_context="")

    assert result == _EXPECTED_PREFIX


def test_build_user_turn_text_includes_plan_section_when_provided():
    result = llm._build_user_turn_text("goal", "page", plan_context="1. ทำ X\n2. ทำ Y")

    assert "Current plan confirmed by the user" in result
    assert "1. ทำ X\n2. ทำ Y" in result
    # อยู่ก่อน "หน้าเว็บปัจจุบัน" (เป็นบริบทระดับ task เหมือน Goal ไม่ใช่ข้อมูลเฉพาะ step นี้)
    assert result.index("Current plan confirmed by the user") < result.index("Current page")


# --- W_token_cut W2: system prefix คงที่ + บล็อกที่ gate ย้ายไป user turn ---


def test_gated_sections_text_empty_for_none_or_empty_set():
    assert llm.gated_sections_text(None) == ""
    assert llm.gated_sections_text(frozenset()) == ""


def test_gated_sections_text_always_in_canonical_order():
    a = llm.gated_sections_text(frozenset({"widget", "plan"}))
    b = llm.gated_sections_text(frozenset({"plan", "widget"}))
    assert a == b  # ลำดับที่ผู้เรียกส่งมาไม่มีผล — กัน prefix cache พลาดจากลำดับ
    assert a.index(llm._PROMPT_PLAN) < a.index(llm._PROMPT_WIDGET)


def test_build_user_turn_text_appends_gated_blocks_after_page_and_history():
    result = llm._build_user_turn_text(
        "goal", "page-text", action_history_context="- step 3: click -> [OK]",
        prompt_sections=frozenset({"table"}),
    )
    assert llm._PROMPT_TABLE in result
    # กฎที่ gate ต้องอยู่ท้ายสุด — หลัง page state และ history (ค่าที่เปลี่ยนทุกเทิร์น)
    assert result.index("Current page:") < result.index(llm._PROMPT_TABLE)
    assert result.index("step 3") < result.index(llm._PROMPT_TABLE)


def test_build_user_turn_text_omits_gated_blocks_by_default():
    """ผู้เรียกที่ไม่ส่ง prompt_sections (เทสต์เดิม/generate_plan) ต้องได้ prompt เดิมเป๊ะ"""
    assert llm._build_user_turn_text("goal", "page") == _EXPECTED_PREFIX


# --- W_prompt_audit: char breakdown ของ request ต่อ call ---


def test_build_user_turn_text_parts_collect_without_changing_output():
    parts = {}
    a = llm._build_user_turn_text(
        "goal", "PAGE_SNAPSHOT", plan_context="1. step", current_url="http://x/y",
        action_history_context="- did click", _parts=parts,
    )
    b = llm._build_user_turn_text(
        "goal", "PAGE_SNAPSHOT", plan_context="1. step", current_url="http://x/y",
        action_history_context="- did click",
    )
    assert a == b  # _parts ต้องไม่แตะข้อความที่ประกอบออกมา
    assert "PAGE_SNAPSHOT" in parts["snapshot"]
    assert "1. step" in parts["plan"]
    assert "did click" in parts["action_history"]
    assert parts["scaffolding"].startswith("Goal: goal")
    # ทุกชิ้นส่วนต่อกันแล้วต้องเท่าข้อความเต็ม
    assert sum(len(v) for v in parts.values()) == len(a)


def test_char_payload_audit_categorises_and_splits_history():
    parts = {"snapshot": "s" * 100, "plan": "p" * 30, "scaffolding": "g" * 20,
             "action_history": "h" * 40, "other": "o" * 5}
    prior = [
        {"role": "assistant", "content": "a" * 60},
        {"role": "tool", "content": "r" * 25},                     # tool result
        {"content": [{"type": "tool_result", "content": "x" * 15}]},  # anthropic tool result
    ]
    a = llm._char_payload_audit(prior_messages=prior, user_parts=parts,
                                system_text="S" * 1000, tools_obj=[{"k": "v"}])
    assert a["system_prompt"] == 1000
    assert a["page_snapshot"] == 100
    assert a["plan"] == 30
    assert a["user_message"] == 20
    assert a["action_history"] == 40
    assert a["tool_result"] >= 25 + 15   # both tool-result messages counted (plain + anthropic-shape)
    assert a["other"] == 5 + 60          # other parts + assistant history
    assert a["tool_schema"] == len(__import__("json").dumps([{"k": "v"}]))


def test_char_payload_audit_never_raises():
    assert llm._char_payload_audit(prior_messages=None, user_parts=None,
                                   system_text=None, tools_obj=object()) != {"crash": True}


@pytest.mark.asyncio
async def test_next_action_groq_system_prefix_is_constant_regardless_of_sections():
    """W_token_cut W2: system message ต้องเป็น _PROMPT_CORE ตัวเดิมเป๊ะ ไม่ว่า sections
    จะเป็นอะไร — prefix cache ของ provider จึงไม่ขาดกลาง task ตอนหน้าเว็บมีตารางโผล่"""
    for sections in (None, frozenset(), frozenset({"table", "widget"}), llm.ALL_PROMPT_SECTIONS):
        tc = _fake_tool_call("c", "browser_action", '{"type": "wait"}')
        client = MagicMock()
        client.chat.completions.create = AsyncMock(
            return_value=_fake_response([tc], {"role": "assistant"})
        )
        _, _, _, messages, _ = await llm.next_action_groq(
            client, "model", "goal", "page", [], prompt_sections=sections,
        )
        assert messages[0] == {"role": "system", "content": llm._PROMPT_CORE}


# --- W_token_trim (P3/M3): site manual by stable id handle + summary bullet ---


def test_site_manual_blocks_empty_when_there_is_no_manual():
    assert llm.site_manual_blocks("", "orangehrmlive.com") == ("", "")
    assert llm.site_manual_blocks("   \n  ", "orangehrmlive.com") == ("", "")


def test_site_manual_blocks_full_and_ref_share_a_stable_id():
    raw = "- Users page: filter by role, delete rows\n- Add User page: form with 4 fields"
    full, ref = llm.site_manual_blocks(raw, "orangehrmlive.com")
    # id is a content hash scoped by domain — stable across calls, changes with the text
    full2, ref2 = llm.site_manual_blocks(raw, "orangehrmlive.com")
    assert (full, ref) == (full2, ref2)
    assert llm.site_manual_blocks(raw + " x", "orangehrmlive.com")[1] != ref
    assert "SITE_MANUAL:orangehrmlive.com#" in ref


def test_render_site_manual_full_carries_the_body_and_the_id():
    raw = (
        "- Users page: filter by role\n- Add User page: 4 fields\n"
        "- Job Titles page: add/edit/delete\n- Reports page: build a custom report"
    )
    full, ref = llm.site_manual_blocks(raw, "orangehrmlive.com")

    rendered_full = llm._build_user_turn_text("goal", "page", site_manual_context=full)
    assert "automatically learned site manual [id=SITE_MANUAL:orangehrmlive.com#" in rendered_full
    assert "- Reports page: build a custom report" in rendered_full
    assert "\x00" not in rendered_full  # sentinel never leaks to the model

    rendered_ref = llm._build_user_turn_text("goal", "page", site_manual_context=ref)
    assert "unchanged" in rendered_ref
    assert "- Reports page: build a custom report" not in rendered_ref  # full body not repeated
    assert len(rendered_ref) < len(rendered_full)
    assert "SITE_MANUAL:orangehrmlive.com#" in rendered_ref
    assert "\x00" not in rendered_ref
    # same id in both so the model can bind the reference to the earlier full text
    full_id = rendered_full.split("[id=")[1].split("]")[0]
    ref_id = rendered_ref.split("[id=")[1].split("]")[0]
    assert full_id == ref_id


def test_render_site_manual_keeps_pre_learned_marker_at_the_start_of_the_body():
    """W21 strict mode checks that the manual text begins with [PRE_LEARNED_MANUAL] — the
    id wrapper must not push that marker off the front of the body."""
    raw = "[PRE_LEARNED_MANUAL]\nTarget Page: Admin — /admin\nRecorded buttons on this page:"
    full, _ = llm.site_manual_blocks(raw, "orangehrmlive.com")
    rendered = llm._build_user_turn_text("goal", "page", site_manual_context=full)
    body = rendered.split("):\n", 1)[1]
    assert body.startswith("[PRE_LEARNED_MANUAL]")


def test_render_site_manual_plain_string_is_unchanged_from_the_old_header():
    """any caller not using the id scheme (tests, generate_plan) gets the exact old text"""
    result = llm._build_user_turn_text("goal", "page", site_manual_context="- just a plain chunk")
    assert "Information from the automatically learned site manual (page structure/" in result
    assert "- just a plain chunk" in result
    assert "[id=" not in result


# --- เวลาปัจจุบันของเซิร์ฟเวอร์ ฉีดเข้า context ทุก turn (LLM ไม่มีการรับรู้เวลาจริงในตัว
# เอง) — ดู _current_bangkok_time_text() ใน llm.py ---


class _FrozenDateTime:
    """แทนที่ llm.datetime ทั้ง class เพื่อ mock datetime.now(tz=...) ตรงๆ — เก็บค่าคงที่
    ไว้ตอบ .now() เสมอไม่ว่าจะเรียกกี่ครั้ง, ไม่แตะ ZoneInfo จริงเลย (ยังทำงานปกติ)"""

    def __init__(self, fixed):
        self._fixed = fixed

    def now(self, tz=None):
        return self._fixed


def test_current_bangkok_time_text_formats_thai_buddhist_date(monkeypatch):
    fixed = datetime(2026, 7, 31, 14, 32, tzinfo=ZoneInfo("Asia/Bangkok"))
    monkeypatch.setattr(llm, "datetime", _FrozenDateTime(fixed))

    result = llm._current_bangkok_time_text()

    # W_prompt_en: Gregorian year in English now, not the Buddhist Era year the Thai
    # format used (2026, not 2569) — 24h clock still matches the mock exactly (14:32)
    assert result == "Friday, 31 July 2026 at 14:32"


def test_build_user_turn_text_injects_current_bangkok_time_from_mocked_now(monkeypatch):
    """mock datetime.now() ที่ระดับต่ำสุด (ไม่ใช่ mock helper function) — ยืนยันว่า context
    ที่ build ออกมาจริงมีบรรทัดเวลาตรงกับ mock value, format/timezone ถูกต้อง"""
    monkeypatch.undo()  # ปลด autouse fixture (_freeze_bangkok_time) ก่อน — เทสต์นี้ต้องการให้
    # _current_bangkok_time_text() ตัวจริงทำงาน (อ่านจาก datetime.now() ที่ mock ด้านล่างแทน)
    fixed = datetime(2026, 12, 25, 9, 5, tzinfo=ZoneInfo("Asia/Bangkok"))
    monkeypatch.setattr(llm, "datetime", _FrozenDateTime(fixed))

    result = llm._build_user_turn_text("goal", "page")

    assert "Current time (Asia/Bangkok): Friday, 25 December 2026 at 09:05" in result


def test_build_user_turn_text_time_line_changes_across_calls_not_cached(monkeypatch):
    """เรียก build 2 ครั้งด้วยเวลา mock ต่างกัน (ห่างกัน) ต้องได้บรรทัดเวลาต่างกันตามเวลา
    จริงแต่ละครั้ง — ยืนยันว่า inject สดทุก turn ไม่ใช่คำนวณครั้งเดียวแล้ว cache ค้างไว้"""
    times = iter([
        "วันศุกร์ที่ 31 กรกฎาคม 2569 เวลา 14:32 น.",
        "วันเสาร์ที่ 1 สิงหาคม 2569 เวลา 09:05 น.",
    ])
    monkeypatch.setattr(llm, "_current_bangkok_time_text", lambda: next(times))

    first = llm._build_user_turn_text("goal", "page")
    second = llm._build_user_turn_text("goal", "page")

    assert "14:32" in first and "09:05" not in first
    assert "09:05" in second and "14:32" not in second


# --- W_procmem: llm.abstract_trajectory() / _format_trajectory_for_abstractor() ---


def test_format_trajectory_for_abstractor_keeps_only_successful_element_actions():
    trajectory = [
        {"success": True, "cmd": {"type": "goto", "url": "https://example.com/login"}},
        {"success": True, "cmd": {"type": "fill", "index": 1, "text": "alice"}, "locator_descriptor": {"accessible_name": "Username"}},
        {"success": False, "cmd": {"type": "click", "index": 2}, "locator_descriptor": None},
        {"success": True, "cmd": {"type": "read_page_data", "query": "how many users"}},
        {"success": True, "cmd": {"type": "click", "index": 3}, "locator_descriptor": {"accessible_name": "Login"}},
    ]

    text = llm._format_trajectory_for_abstractor(trajectory)
    lines = text.splitlines()

    assert len(lines) == 3  # goto + successful fill + successful click; failed click and read_page_data dropped
    assert '"action": "goto"' in lines[0]
    assert '"typed_value": "alice"' in lines[1]
    assert '"accessible_name": "Username"' in lines[1]
    assert '"accessible_name": "Login"' in lines[2]


def test_format_trajectory_for_abstractor_handles_empty_trajectory():
    assert llm._format_trajectory_for_abstractor([]) == "(no successful element-targeting actions recorded)"


@pytest.mark.asyncio
async def test_abstract_trajectory_returns_tool_input_on_anthropic_success():
    expected_template = {
        "goal_pattern": "Log in with a username",
        "url_pattern": "https://example.com/login",
        "slots": [{"name": "username", "description": "the username"}],
        "steps": [{"action": "fill", "target": {"accessible_name": "Username"}, "value": "{{username}}"}],
    }
    client = MagicMock()
    client.messages.create = AsyncMock(
        return_value=_fake_anthropic_response([_fake_anthropic_tool_use_block("emit_template", expected_template)])
    )

    result = await llm.abstract_trajectory(
        client, "claude-x", "log in", "https://example.com/login",
        [{"success": True, "cmd": {"type": "fill", "index": 1, "text": "alice"}, "locator_descriptor": {"accessible_name": "Username"}}],
        "anthropic",
    )

    assert result == expected_template
    _, kwargs = client.messages.create.call_args
    assert kwargs["tool_choice"] == {"type": "tool", "name": "emit_template"}
    assert kwargs["tools"] == [llm.ABSTRACTOR_TOOL]


@pytest.mark.asyncio
async def test_abstract_trajectory_returns_none_when_no_tool_call():
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=_fake_anthropic_response([]))

    result = await llm.abstract_trajectory(client, "claude-x", "goal", "https://example.com", [], "anthropic")

    assert result is None


@pytest.mark.asyncio
async def test_abstract_trajectory_swallows_provider_errors_and_returns_none():
    client = MagicMock()
    client.messages.create = AsyncMock(side_effect=RuntimeError("API down"))

    result = await llm.abstract_trajectory(client, "claude-x", "goal", "https://example.com", [], "anthropic")

    assert result is None


@pytest.mark.asyncio
async def test_abstract_trajectory_returns_none_for_unknown_provider():
    result = await llm.abstract_trajectory(MagicMock(), "model", "goal", "https://example.com", [], "unknown")

    assert result is None


@pytest.mark.asyncio
async def test_abstract_trajectory_groq_parses_tool_call_arguments():
    template_json = json.dumps({
        "goal_pattern": "Log in",
        "url_pattern": "https://example.com",
        "slots": [],
        "steps": [{"action": "click", "target": {"accessible_name": "Login"}}],
    })
    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        return_value=_fake_response([_fake_tool_call("call_1", "emit_template", template_json)], {})
    )

    result = await llm.abstract_trajectory(client, "llama-x", "goal", "https://example.com", [], "groq")

    assert result["goal_pattern"] == "Log in"
    assert result["steps"][0]["action"] == "click"


# --- W_procmem: llm.plan_with_procedural_memory() ---

_CANDIDATES = [{
    "template_id": "t1", "intent_key": "k1", "version": 1,
    "goal_pattern": "Log in with a username and password",
    "url_pattern": "https://example.com/login",
    "steps": [{"action": "fill"}, {"action": "click"}],
    "slots": [{"name": "username"}, {"name": "password"}],
    "distance": 0.2,
}]


@pytest.mark.asyncio
async def test_plan_with_procedural_memory_returns_reuse_decision_above_threshold():
    decision_payload = {
        "decision": "reuse", "template_id": "t1", "confidence": 0.9,
        "slot_values": {"username": "alice"}, "reason": "same task class and URL pattern",
    }
    client = MagicMock()
    client.messages.create = AsyncMock(
        return_value=_fake_anthropic_response([_fake_anthropic_tool_use_block("plan_decision", decision_payload)])
    )

    result = await llm.plan_with_procedural_memory(
        client, "claude-x", "log in as alice", "https://example.com/login", "", _CANDIDATES, "anthropic",
    )

    assert result["decision"] == "reuse"
    assert result["template_id"] == "t1"
    assert result["slot_values"] == {"username": "alice"}
    _, kwargs = client.messages.create.call_args
    assert kwargs["tool_choice"] == {"type": "tool", "name": "plan_decision"}


@pytest.mark.asyncio
async def test_plan_with_procedural_memory_defaults_missing_template_id_when_only_one_candidate():
    """เจอจริงตอนทดสอบ: LLM ตอบ decision='reuse' มาแต่ลืมใส่ template_id มาด้วย (ไม่ได้
    อยู่ใน required ของ schema เพราะบังคับ "required เฉพาะตอน reuse/adapt" ข้าม provider
    ให้เนียนไม่ได้) — ถ้ามี candidate แค่ตัวเดียวไม่มีความกำกวม ต้องเดาแทนให้ได้"""
    decision_payload = {"decision": "reuse", "confidence": 0.95, "slot_values": {}, "reason": "exact match"}
    client = MagicMock()
    client.messages.create = AsyncMock(
        return_value=_fake_anthropic_response([_fake_anthropic_tool_use_block("plan_decision", decision_payload)])
    )

    result = await llm.plan_with_procedural_memory(
        client, "claude-x", "goal", "https://example.com/login", "", _CANDIDATES, "anthropic",
    )

    assert result["decision"] == "reuse"
    assert result["template_id"] == "t1"


@pytest.mark.asyncio
async def test_plan_with_procedural_memory_falls_back_to_plan_fresh_when_template_id_ambiguous():
    """เหมือนเทสต์ข้างบน แต่มีมากกว่า 1 candidate — เดาไม่ได้ว่าหมายถึงตัวไหน ต้อง
    plan_fresh แทนการเดามั่วๆ"""
    two_candidates = _CANDIDATES + [{**_CANDIDATES[0], "template_id": "t2"}]
    decision_payload = {"decision": "reuse", "confidence": 0.95, "slot_values": {}, "reason": "match"}
    client = MagicMock()
    client.messages.create = AsyncMock(
        return_value=_fake_anthropic_response([_fake_anthropic_tool_use_block("plan_decision", decision_payload)])
    )

    result = await llm.plan_with_procedural_memory(
        client, "claude-x", "goal", "https://example.com/login", "", two_candidates, "anthropic",
    )

    assert result["decision"] == "plan_fresh"
    assert result["template_id"] is None


@pytest.mark.asyncio
async def test_plan_with_procedural_memory_forces_plan_fresh_below_confidence_threshold():
    """defense-in-depth: แม้ LLM ตอบ decision='reuse' มา ถ้า confidence ต่ำกว่า
    settings.procedural_memory_min_confidence ต้องบังคับ plan_fresh เองเสมอ ไม่เชื่อ
    LLM ตรงๆ 100%"""
    decision_payload = {
        "decision": "reuse", "template_id": "t1", "confidence": 0.3,
        "slot_values": {}, "reason": "weak match",
    }
    client = MagicMock()
    client.messages.create = AsyncMock(
        return_value=_fake_anthropic_response([_fake_anthropic_tool_use_block("plan_decision", decision_payload)])
    )

    result = await llm.plan_with_procedural_memory(
        client, "claude-x", "goal", "https://example.com", "", _CANDIDATES, "anthropic",
    )

    assert result["decision"] == "plan_fresh"
    assert result["template_id"] is None


@pytest.mark.asyncio
async def test_plan_with_procedural_memory_returns_safe_default_when_no_tool_call():
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=_fake_anthropic_response([]))

    result = await llm.plan_with_procedural_memory(
        client, "claude-x", "goal", "https://example.com", "", _CANDIDATES, "anthropic",
    )

    assert result["decision"] == "plan_fresh"
    assert result["confidence"] == 0.0


@pytest.mark.asyncio
async def test_plan_with_procedural_memory_swallows_provider_errors():
    client = MagicMock()
    client.messages.create = AsyncMock(side_effect=RuntimeError("API down"))

    result = await llm.plan_with_procedural_memory(
        client, "claude-x", "goal", "https://example.com", "", _CANDIDATES, "anthropic",
    )

    assert result["decision"] == "plan_fresh"
    assert result["confidence"] == 0.0


@pytest.mark.asyncio
async def test_plan_with_procedural_memory_returns_safe_default_for_unknown_provider():
    result = await llm.plan_with_procedural_memory(
        MagicMock(), "model", "goal", "https://example.com", "", _CANDIDATES, "unknown",
    )

    assert result["decision"] == "plan_fresh"


@pytest.mark.asyncio
async def test_plan_with_procedural_memory_groq_parses_adapt_decision_with_patch():
    decision_json = json.dumps({
        "decision": "adapt", "template_id": "t1", "confidence": 0.75,
        "slot_values": {"username": "bob"},
        "patch": [{"op": "replace", "index": 1, "step": {"action": "click", "target": {"accessible_name": "Sign in"}}}],
        "reason": "button label differs slightly",
    })
    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        return_value=_fake_response([_fake_tool_call("call_1", "plan_decision", decision_json)], {})
    )

    result = await llm.plan_with_procedural_memory(
        client, "llama-x", "goal", "https://example.com", "", _CANDIDATES, "groq",
    )

    assert result["decision"] == "adapt"
    assert result["patch"][0]["op"] == "replace"


def test_format_candidates_for_planner_reduces_steps_to_action_sequence():
    text = llm._format_candidates_for_planner(_CANDIDATES)

    assert '"step_sequence": "fill/click"' in text
    assert "username" in text and "password" in text


def test_format_candidates_for_planner_handles_empty_list():
    assert llm._format_candidates_for_planner([]) == "(no candidates)"


def test_format_candidates_for_planner_includes_track_record():
    """ACC-1 (accuracy audit follow-up): success_count/failure_count ต้องโผล่ในสรุปที่ส่ง
    ให้ Planner เห็นด้วย — เดิมถูกตัดออกไปเหมือน locator ทั้งที่เป็นสัญญาณคนละแบบกัน (ดู
    _format_candidates_for_planner() docstring)"""
    candidates = [{**_CANDIDATES[0], "success_count": 3, "failure_count": 1}]
    text = llm._format_candidates_for_planner(candidates)
    assert '"success_count": 3' in text
    assert '"failure_count": 1' in text


def test_format_candidates_for_planner_defaults_track_record_to_zero_when_missing():
    """sanity: candidate ที่ไม่มี field นี้เลย (เช่นจาก caller เก่าที่ยังไม่รู้จัก field
    นี้) ต้องไม่ throw — default เป็น 0 เงียบๆ"""
    text = llm._format_candidates_for_planner(_CANDIDATES)
    assert '"success_count": 0' in text
    assert '"failure_count": 0' in text


# --- W_procmem: llm.plan_with_procedural_memory()'s has_auto_login note ---
#
# บั๊กจริงที่เจอตอน Phase 4 validation: โดเมนที่มี auto-login credential เก็บไว้ (ดู
# orchestrator.py::_maybe_auto_login) ทำให้ login เกิดขึ้น "นอก" LLM loop เสมอ ไม่ว่า
# ทางไหน — candidate template ที่ capture มาจาก run แบบนั้นเลยไม่มี step login แม้แต่
# นิดเดียว ถ้าไม่บอก Planner ตรงๆ ว่าโดเมนนี้ auto-login ไว้แล้ว มันจะเดา (ผิด) ว่า
# candidate ขาด step ไปแล้วปฏิเสธ reuse ทั้งที่ใช้ได้ปกติ


@pytest.mark.asyncio
async def test_plan_with_procedural_memory_includes_auto_login_note_when_true():
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=_fake_anthropic_response([]))

    await llm.plan_with_procedural_memory(
        client, "claude-x", "goal", "https://example.com/login", "", _CANDIDATES, "anthropic",
        has_auto_login=True,
    )

    _, kwargs = client.messages.create.call_args
    prompt = kwargs["messages"][0]["content"]
    assert "AUTO_LOGIN: this domain has stored credentials" in prompt
    assert "do not penalize it or require login steps" in prompt


@pytest.mark.asyncio
async def test_plan_with_procedural_memory_includes_no_auto_login_note_when_false():
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=_fake_anthropic_response([]))

    await llm.plan_with_procedural_memory(
        client, "claude-x", "goal", "https://example.com/login", "", _CANDIDATES, "anthropic",
        has_auto_login=False,
    )

    _, kwargs = client.messages.create.call_args
    prompt = kwargs["messages"][0]["content"]
    assert "AUTO_LOGIN: none configured for this domain" in prompt


@pytest.mark.asyncio
async def test_plan_with_procedural_memory_defaults_has_auto_login_to_false():
    """ผู้เรียกเดิม (ก่อนแก้บั๊กนี้) ที่ไม่รู้จัก parameter นี้เลยยังต้องทำงานได้ตามปกติ"""
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=_fake_anthropic_response([]))

    await llm.plan_with_procedural_memory(
        client, "claude-x", "goal", "https://example.com/login", "", _CANDIDATES, "anthropic",
    )

    _, kwargs = client.messages.create.call_args
    prompt = kwargs["messages"][0]["content"]
    assert "AUTO_LOGIN: none configured for this domain" in prompt


# --- W_procmem: llm.repair_step() ---

_FAILED_STEP = {"action": "click", "target": {"accessible_name": "Login"}, "slot": None}


@pytest.mark.asyncio
async def test_repair_step_returns_corrected_step_on_anthropic_success():
    corrected = {"action": "click", "target": {"accessible_name": "Sign in"}, "slot": None}
    client = MagicMock()
    client.messages.create = AsyncMock(
        return_value=_fake_anthropic_response([_fake_anthropic_tool_use_block("emit_repaired_step", corrected)])
    )

    result = await llm.repair_step(client, "claude-x", _FAILED_STEP, "element not found", "page text", "anthropic")

    assert result == corrected
    _, kwargs = client.messages.create.call_args
    assert kwargs["tool_choice"] == {"type": "tool", "name": "emit_repaired_step"}


@pytest.mark.asyncio
async def test_repair_step_returns_replan_when_no_tool_call():
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=_fake_anthropic_response([]))

    result = await llm.repair_step(client, "claude-x", _FAILED_STEP, "error", "page", "anthropic")

    assert result == {"action": "replan"}


@pytest.mark.asyncio
async def test_repair_step_swallows_provider_errors_and_returns_replan():
    client = MagicMock()
    client.messages.create = AsyncMock(side_effect=RuntimeError("API down"))

    result = await llm.repair_step(client, "claude-x", _FAILED_STEP, "error", "page", "anthropic")

    assert result == {"action": "replan"}


@pytest.mark.asyncio
async def test_repair_step_returns_replan_for_unknown_provider():
    result = await llm.repair_step(MagicMock(), "model", _FAILED_STEP, "error", "page", "unknown")

    assert result == {"action": "replan"}


@pytest.mark.asyncio
async def test_repair_step_groq_parses_replan_signal_from_model():
    replan_json = json.dumps({"action": "replan"})
    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        return_value=_fake_response([_fake_tool_call("call_1", "emit_repaired_step", replan_json)], {})
    )

    result = await llm.repair_step(client, "llama-x", _FAILED_STEP, "element gone", "page", "groq")

    assert result == {"action": "replan"}


# --- W19 (ดู W19.txt ข้อ 8): llm.evaluate_semantic_redundancy() ---


@pytest.mark.asyncio
async def test_evaluate_semantic_redundancy_returns_decision_on_anthropic_success():
    decision = {
        "is_semantically_redundant": True,
        "value_score": 0.1,
        "action_decision": "SKIP_STEP",
        "reasoning": "ไม่เกี่ยวกับ goal เลย",
    }
    client = MagicMock()
    client.messages.create = AsyncMock(
        return_value=_fake_anthropic_response([_fake_anthropic_tool_use_block("evaluate_action_value", decision)])
    )

    result = await llm.evaluate_semantic_redundancy(
        client, "claude-x", "goal", "scroll down", "page title", "footer link", "browser_action",
        {"type": "scroll", "direction": "down"}, "anthropic",
    )

    assert result == decision
    _, kwargs = client.messages.create.call_args
    assert kwargs["tool_choice"] == {"type": "tool", "name": "evaluate_action_value"}


@pytest.mark.asyncio
async def test_evaluate_semantic_redundancy_defaults_to_pass_when_no_tool_call():
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=_fake_anthropic_response([]))

    result = await llm.evaluate_semantic_redundancy(
        client, "claude-x", "goal", "click login", "page", "target", "browser_action",
        {"type": "click", "index": 1}, "anthropic",
    )

    assert result["action_decision"] == "PASS"


@pytest.mark.asyncio
async def test_evaluate_semantic_redundancy_swallows_provider_errors_and_defaults_to_pass():
    client = MagicMock()
    client.messages.create = AsyncMock(side_effect=RuntimeError("API down"))

    result = await llm.evaluate_semantic_redundancy(
        client, "claude-x", "goal", "click login", "page", "target", "browser_action",
        {"type": "click", "index": 1}, "anthropic",
    )

    assert result["action_decision"] == "PASS"


@pytest.mark.asyncio
async def test_evaluate_semantic_redundancy_defaults_to_pass_for_unknown_provider():
    result = await llm.evaluate_semantic_redundancy(
        MagicMock(), "model", "goal", "step", "page", "target", "browser_action",
        {"type": "click", "index": 1}, "unknown",
    )

    assert result["action_decision"] == "PASS"


@pytest.mark.asyncio
async def test_evaluate_semantic_redundancy_groq_parses_decision_from_model():
    decision_json = json.dumps({
        "is_semantically_redundant": False, "value_score": 0.9,
        "action_decision": "PASS", "reasoning": "จำเป็นต่อ goal",
    })
    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        return_value=_fake_response([_fake_tool_call("call_1", "evaluate_action_value", decision_json)], {})
    )

    result = await llm.evaluate_semantic_redundancy(
        client, "llama-x", "goal", "click login", "page", "target", "browser_action",
        {"type": "click", "index": 1}, "groq",
    )

    assert result["action_decision"] == "PASS"
    assert result["value_score"] == 0.9


# --- W19-2: llm.evaluate_safety_and_performance() (Safety & Performance Middleware) ---

_MIDDLEWARE_DECISION = {
    "redundancy_evaluation": {"is_redundant": False, "redundancy_reason": ""},
    "permission_evaluation": {"risk_level": "AUTO_APPROVE", "permission_reason": "routine search"},
    "final_action_decision": "EXECUTE",
}


@pytest.mark.asyncio
async def test_evaluate_safety_and_performance_returns_decision_on_anthropic_success():
    client = MagicMock()
    client.messages.create = AsyncMock(
        return_value=_fake_anthropic_response(
            [_fake_anthropic_tool_use_block("middleware_evaluate", _MIDDLEWARE_DECISION)]
        )
    )

    result = await llm.evaluate_safety_and_performance(
        client, "claude-x", "goal", "shopee.co.th", "click", "<button> 'Search'", "", "anthropic",
    )

    assert result == _MIDDLEWARE_DECISION
    _, kwargs = client.messages.create.call_args
    assert kwargs["tool_choice"] == {"type": "tool", "name": "middleware_evaluate"}


@pytest.mark.asyncio
async def test_evaluate_safety_and_performance_defaults_to_auto_approve_execute_when_no_tool_call():
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=_fake_anthropic_response([]))

    result = await llm.evaluate_safety_and_performance(
        client, "claude-x", "goal", "example.com", "click", "<button>", "", "anthropic",
    )

    assert result["final_action_decision"] == "EXECUTE"
    assert result["permission_evaluation"]["risk_level"] == "AUTO_APPROVE"
    assert result["redundancy_evaluation"]["is_redundant"] is False


@pytest.mark.asyncio
async def test_evaluate_safety_and_performance_swallows_provider_errors_and_fails_open():
    client = MagicMock()
    client.messages.create = AsyncMock(side_effect=RuntimeError("API down"))

    result = await llm.evaluate_safety_and_performance(
        client, "claude-x", "goal", "example.com", "click", "<button>", "", "anthropic",
    )

    assert result["final_action_decision"] == "EXECUTE"
    assert result["permission_evaluation"]["risk_level"] == "AUTO_APPROVE"


@pytest.mark.asyncio
async def test_evaluate_safety_and_performance_defaults_to_safe_for_unknown_provider():
    result = await llm.evaluate_safety_and_performance(
        MagicMock(), "model", "goal", "example.com", "click", "<button>", "", "unknown",
    )

    assert result["final_action_decision"] == "EXECUTE"
    assert result["permission_evaluation"]["risk_level"] == "AUTO_APPROVE"


@pytest.mark.asyncio
async def test_evaluate_safety_and_performance_groq_parses_decision_from_model():
    decision = {
        "redundancy_evaluation": {"is_redundant": True, "redundancy_reason": "footer scraping ไม่เกี่ยว"},
        "permission_evaluation": {"risk_level": "AUTO_APPROVE", "permission_reason": ""},
        "final_action_decision": "SKIP_REDUNDANT",
    }
    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        return_value=_fake_response(
            [_fake_tool_call("call_1", "middleware_evaluate", json.dumps(decision))], {},
        )
    )

    result = await llm.evaluate_safety_and_performance(
        client, "llama-x", "goal", "example.com", "click", "<a> 'footer link'", "", "groq",
    )

    assert result["final_action_decision"] == "SKIP_REDUNDANT"


@pytest.mark.asyncio
async def test_evaluate_safety_and_performance_flags_destructive_action_as_requires_consent():
    decision = {
        "redundancy_evaluation": {"is_redundant": False, "redundancy_reason": ""},
        "permission_evaluation": {"risk_level": "REQUIRES_CONSENT", "permission_reason": "deletes the user's account"},
        "final_action_decision": "PROMPT_USER_PERMISSION",
    }
    client = MagicMock()
    client.messages.create = AsyncMock(
        return_value=_fake_anthropic_response([_fake_anthropic_tool_use_block("middleware_evaluate", decision)])
    )

    result = await llm.evaluate_safety_and_performance(
        client, "claude-x", "delete my account", "example.com", "click", "<button> 'Delete Account'", "", "anthropic",
    )

    assert result["permission_evaluation"]["risk_level"] == "REQUIRES_CONSENT"
    assert result["final_action_decision"] == "PROMPT_USER_PERMISSION"


# --- W19-3: llm.generate_persona_message() (Voice & Persona Interface) ---


@pytest.mark.asyncio
async def test_generate_persona_message_returns_message_on_anthropic_success():
    decision = {"user_message": "เรียบร้อยครับ! จองคิวให้เสร็จแล้ว", "action_status": "COMPLETED"}
    client = MagicMock()
    client.messages.create = AsyncMock(
        return_value=_fake_anthropic_response([_fake_anthropic_tool_use_block("speak_to_user", decision)])
    )

    result = await llm.generate_persona_message(
        client, "claude-x", "Shopee", "จองคิว", "COMPLETED", "จองคิวสำเร็จ", "anthropic",
    )

    assert result == decision
    _, kwargs = client.messages.create.call_args
    assert kwargs["tool_choice"] == {"type": "tool", "name": "speak_to_user"}


@pytest.mark.asyncio
async def test_generate_persona_message_defaults_to_empty_message_when_no_tool_call():
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=_fake_anthropic_response([]))

    result = await llm.generate_persona_message(
        client, "claude-x", "example.com", "goal", "FAILED", "timeout", "anthropic",
    )

    assert result["user_message"] == ""
    assert result["action_status"] == "IN_PROGRESS"


@pytest.mark.asyncio
async def test_generate_persona_message_swallows_provider_errors_and_returns_empty():
    client = MagicMock()
    client.messages.create = AsyncMock(side_effect=RuntimeError("API down"))

    result = await llm.generate_persona_message(
        client, "claude-x", "example.com", "goal", "COMPLETED", "done", "anthropic",
    )

    assert result["user_message"] == ""
    assert result["action_status"] == "IN_PROGRESS"


@pytest.mark.asyncio
async def test_generate_persona_message_defaults_to_empty_for_unknown_provider():
    result = await llm.generate_persona_message(
        MagicMock(), "model", "example.com", "goal", "COMPLETED", "done", "unknown",
    )

    assert result["user_message"] == ""


@pytest.mark.asyncio
async def test_generate_persona_message_groq_parses_message_from_model():
    decision = {"user_message": "เดี๋ยวลองใหม่นะครับ", "action_status": "FAILED"}
    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        return_value=_fake_response([_fake_tool_call("call_1", "speak_to_user", json.dumps(decision))], {})
    )

    result = await llm.generate_persona_message(
        client, "llama-x", "example.com", "goal", "FAILED", "element not found", "groq",
    )

    assert result["action_status"] == "FAILED"
    assert "ลองใหม่" in result["user_message"]


# --- pdf/xlsx: llm.answer_file_query() (Attached File Query) ---


def _fake_text_block(text):
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


@pytest.mark.asyncio
async def test_answer_file_query_anthropic_returns_text():
    client = MagicMock()
    client.messages.create = AsyncMock(
        return_value=_fake_anthropic_response([_fake_text_block("ยอดรวมคือ 1,250 บาท")])
    )

    result = await llm.answer_file_query(
        client, "claude-x", "ยอดรวมเท่าไหร่", "Invoice total: 1,250 THB", "invoice.pdf", "anthropic",
    )

    assert result == "ยอดรวมคือ 1,250 บาท"
    _, kwargs = client.messages.create.call_args
    assert "invoice.pdf" in kwargs["messages"][0]["content"]
    assert "Invoice total: 1,250 THB" in kwargs["messages"][0]["content"]


@pytest.mark.asyncio
async def test_answer_file_query_groq_returns_text():
    message = MagicMock()
    message.content = "สรุปแล้วครับ"
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=response)

    result = await llm.answer_file_query(
        client, "llama-x", "สรุปให้หน่อย", "some document text", "notes.pdf", "groq",
    )

    assert result == "สรุปแล้วครับ"


@pytest.mark.asyncio
async def test_answer_file_query_gemini_returns_text():
    response = MagicMock()
    response.text = "แถวที่ 3 คือ Gadget"
    client = MagicMock()
    client.GenerativeModel = MagicMock(return_value=MagicMock(
        generate_content_async=AsyncMock(return_value=response),
    ))

    result = await llm.answer_file_query(
        client, "gemini-x", "แถวที่ 3 คืออะไร", "# Sheet: Data\nA | B\nWidget | 1\nGadget | 2", "report.xlsx", "gemini",
    )

    assert result == "แถวที่ 3 คือ Gadget"


@pytest.mark.asyncio
async def test_answer_file_query_unknown_provider_returns_apology():
    result = await llm.answer_file_query(
        MagicMock(), "model", "goal", "text", "file.pdf", "unknown",
    )
    assert "Sorry" in result


@pytest.mark.asyncio
async def test_answer_file_query_swallows_provider_errors():
    client = MagicMock()
    client.messages.create = AsyncMock(side_effect=RuntimeError("API down"))

    result = await llm.answer_file_query(
        client, "claude-x", "goal", "text", "file.pdf", "anthropic",
    )

    assert "Sorry" in result


@pytest.mark.asyncio
async def test_answer_file_query_truncates_long_file_text_and_notes_it():
    long_text = "a" * (llm._ANSWER_FILE_QUERY_MAX_CHARS + 5000)
    client = MagicMock()
    client.messages.create = AsyncMock(
        return_value=_fake_anthropic_response([_fake_text_block("ok")])
    )

    # goal ตั้งใจใช้ข้อความไทยล้วน (ไม่มีตัวอักษร "a") กันชนกับการนับ "a" ของเนื้อหาไฟล์ด้านล่าง
    await llm.answer_file_query(client, "claude-x", "สรุปให้หน่อย", long_text, "big.pdf", "anthropic")

    _, kwargs = client.messages.create.call_args
    sent_content = kwargs["messages"][0]["content"]
    # ตัดที่ _ANSWER_FILE_QUERY_MAX_CHARS ตัวอักษรของเนื้อหาไฟล์เท่านั้น (ไม่ใช่ทั้ง prompt)
    # W_prompt_en: นับ "a" เฉพาะในส่วนเนื้อหาไฟล์ ไม่ใช่ทั้ง prompt — หลังแปล prompt เป็น
    # อังกฤษ ตัว label รอบๆ ("File name:"/"Document content:") มี "a" ปนอยู่ด้วยแล้ว
    file_section = sent_content.split("Document content:\n", 1)[1].split("\n\n[Note:", 1)[0]
    assert file_section.count("a") == llm._ANSWER_FILE_QUERY_MAX_CHARS
    assert "only part of it is shown above" in sent_content


# --- pdf/xlsx (ต่อ): llm.answer_image_query() (Attached Image Query) ---

_FAKE_PNG_BYTES = b"\x89PNG\r\n\x1a\nfake image bytes"


@pytest.mark.asyncio
async def test_answer_image_query_anthropic_sends_base64_image_block():
    client = MagicMock()
    client.messages.create = AsyncMock(
        return_value=_fake_anthropic_response([_fake_text_block("ในภาพเห็นใบเสร็จร้านกาแฟ")])
    )

    result = await llm.answer_image_query(
        client, "claude-x", "ในภาพนี้มีอะไรบ้าง", _FAKE_PNG_BYTES, "receipt.png", "anthropic",
    )

    assert result == "ในภาพเห็นใบเสร็จร้านกาแฟ"
    _, kwargs = client.messages.create.call_args
    content_blocks = kwargs["messages"][0]["content"]
    image_block = next(b for b in content_blocks if b["type"] == "image")
    assert image_block["source"]["media_type"] == "image/png"
    text_block = next(b for b in content_blocks if b["type"] == "text")
    assert "receipt.png" in text_block["text"]


@pytest.mark.asyncio
async def test_answer_image_query_groq_sends_data_url_image():
    message = MagicMock()
    message.content = "รูปนี้คือแมวสีส้ม"
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=response)

    result = await llm.answer_image_query(
        client, "llama-vision-x", "ในรูปคืออะไร", _FAKE_PNG_BYTES, "cat.jpg", "groq",
    )

    assert result == "รูปนี้คือแมวสีส้ม"
    _, kwargs = client.chat.completions.create.call_args
    image_part = next(p for p in kwargs["messages"][1]["content"] if p["type"] == "image_url")
    assert image_part["image_url"]["url"].startswith("data:image/jpeg;base64,")


@pytest.mark.asyncio
async def test_answer_image_query_gemini_sends_inline_image_data():
    response = MagicMock()
    response.text = "ภาพนี้เป็นสไลด์นำเสนอ"
    client = MagicMock()
    client.GenerativeModel = MagicMock(return_value=MagicMock(
        generate_content_async=AsyncMock(return_value=response),
    ))

    result = await llm.answer_image_query(
        client, "gemini-x", "ภาพนี้คืออะไร", _FAKE_PNG_BYTES, "slide.webp", "gemini",
    )

    assert result == "ภาพนี้เป็นสไลด์นำเสนอ"
    _, kwargs = client.GenerativeModel.return_value.generate_content_async.call_args
    parts = kwargs["contents"][0]["parts"]
    image_part = next(p for p in parts if "mime_type" in p)
    assert image_part["mime_type"] == "image/webp"
    assert image_part["data"] == _FAKE_PNG_BYTES


@pytest.mark.asyncio
async def test_answer_image_query_unknown_provider_returns_apology():
    result = await llm.answer_image_query(
        MagicMock(), "model", "goal", _FAKE_PNG_BYTES, "file.png", "unknown",
    )
    assert "Sorry" in result


@pytest.mark.asyncio
async def test_answer_image_query_swallows_provider_errors():
    client = MagicMock()
    client.messages.create = AsyncMock(side_effect=RuntimeError("API down"))

    result = await llm.answer_image_query(
        client, "claude-x", "goal", _FAKE_PNG_BYTES, "file.png", "anthropic",
    )

    assert "Sorry" in result


# --- W19-4: llm.route_multi_turn_strategy() (Orchestrator & Planner Agent, multi-turn) ---


def test_multi_turn_system_prompt_covers_pronoun_entity_switch_continuation():
    assert "Cedric Kelly" in llm._MULTI_TURN_SYSTEM_PROMPT
    assert "คนต่อไปละ" in llm._MULTI_TURN_SYSTEM_PROMPT
    assert "STILL\n    REPLY_FROM_MEMORY" in llm._MULTI_TURN_SYSTEM_PROMPT


_ROUTE_REPLY_FROM_MEMORY = {
    "context_analysis": {"is_continuation_of_previous_turn": True, "target_entity_from_memory": "Nike Pegasus 42"},
    "chosen_strategy": "REPLY_FROM_MEMORY",
    "reasoning": "ราคาอยู่ใน buffer แล้ว ไม่ต้อง action ใดๆ",
    "planned_action": {"tool": "reply", "target_selector": "", "parameters": {}},
}


@pytest.mark.asyncio
async def test_route_multi_turn_strategy_returns_decision_on_anthropic_success():
    client = MagicMock()
    client.messages.create = AsyncMock(
        return_value=_fake_anthropic_response([_fake_anthropic_tool_use_block("route_strategy", _ROUTE_REPLY_FROM_MEMORY)])
    )

    result = await llm.route_multi_turn_strategy(
        client, "claude-x", "ซื้อรองเท้า", "ราคาเท่าไหร่", "shopee.co.th", "https://shopee.co.th/search?q=รองเท้า",
        '[{"item_index": 1, "title": "Nike Pegasus 42", "price": "฿4,200"}]', "", "anthropic",
    )

    assert result == _ROUTE_REPLY_FROM_MEMORY
    _, kwargs = client.messages.create.call_args
    assert kwargs["tool_choice"] == {"type": "tool", "name": "route_strategy"}


@pytest.mark.asyncio
async def test_route_multi_turn_strategy_defaults_to_new_navigation_when_no_tool_call():
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=_fake_anthropic_response([]))

    result = await llm.route_multi_turn_strategy(
        client, "claude-x", "goal", "instruction", "example.com", "https://example.com", "", "", "anthropic",
    )

    assert result["chosen_strategy"] == "NEW_NAVIGATION"
    assert result["context_analysis"]["is_continuation_of_previous_turn"] is False


@pytest.mark.asyncio
async def test_route_multi_turn_strategy_swallows_provider_errors_and_defaults_to_new_navigation():
    client = MagicMock()
    client.messages.create = AsyncMock(side_effect=RuntimeError("API down"))

    result = await llm.route_multi_turn_strategy(
        client, "claude-x", "goal", "instruction", "example.com", "https://example.com", "", "", "anthropic",
    )

    assert result["chosen_strategy"] == "NEW_NAVIGATION"


@pytest.mark.asyncio
async def test_route_multi_turn_strategy_defaults_to_new_navigation_for_unknown_provider():
    result = await llm.route_multi_turn_strategy(
        MagicMock(), "model", "goal", "instruction", "example.com", "https://example.com", "", "", "unknown",
    )

    assert result["chosen_strategy"] == "NEW_NAVIGATION"


@pytest.mark.asyncio
async def test_route_multi_turn_strategy_groq_parses_in_page_action_from_model():
    decision = {
        "context_analysis": {"is_continuation_of_previous_turn": True, "target_entity_from_memory": "item #1"},
        "chosen_strategy": "IN_PAGE_ACTION",
        "reasoning": "ต้องคลิกดูรายละเอียดอันแรกบนหน้าปัจจุบัน",
        "planned_action": {"tool": "click", "target_selector": "li:nth-child(1)", "parameters": {}},
    }
    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        return_value=_fake_response([_fake_tool_call("call_1", "route_strategy", json.dumps(decision))], {})
    )

    result = await llm.route_multi_turn_strategy(
        client, "llama-x", "goal", "ขอรายละเอียดอันแรก", "example.com", "https://example.com",
        '[{"item_index": 1, "title": "item #1"}]', "", "groq",
    )

    assert result["chosen_strategy"] == "IN_PAGE_ACTION"
    assert result["planned_action"]["tool"] == "click"


# --- W_openai_multiturn: openai (ChatGPT OAuth / codex Responses API) branch of the
# multi-turn stack — route_multi_turn_strategy() + extract_structured_items() both go
# through llm._openai_forced_tool_call(), which reads function_call items off
# "response.output_item.done" stream events ---


class _FakeOpenAIStreamEvent:
    def __init__(self, type, item=None, response=None):
        self.type = type
        self.item = item
        self.response = response


class _FakeOpenAIStream:
    def __init__(self, events):
        self._events = events

    def __aiter__(self):
        return self._aiter()

    async def _aiter(self):
        for event in self._events:
            yield event


def _fake_openai_function_call_item(name, args_obj):
    item = MagicMock()
    item.type = "function_call"
    item.name = name
    item.arguments = json.dumps(args_obj)
    return item


def _fake_openai_forced_tool_client(events, monkeypatch):
    monkeypatch.setattr(llm, "_openai_oauth_headers", AsyncMock(return_value={}))
    client = MagicMock()
    client.responses.create = AsyncMock(return_value=_FakeOpenAIStream(events))
    return client


@pytest.mark.asyncio
async def test_route_multi_turn_strategy_openai_parses_decision_from_model(monkeypatch):
    client = _fake_openai_forced_tool_client(
        [_FakeOpenAIStreamEvent(
            "response.output_item.done",
            item=_fake_openai_function_call_item("route_strategy", _ROUTE_REPLY_FROM_MEMORY),
        )],
        monkeypatch,
    )

    result = await llm.route_multi_turn_strategy(
        client, "gpt-5-codex", "ซื้อรองเท้า", "ราคาเท่าไหร่", "shopee.co.th",
        "https://shopee.co.th/search?q=รองเท้า",
        '[{"item_index": 1, "title": "Nike Pegasus 42", "price": "฿4,200"}]', "", "openai",
    )

    assert result == _ROUTE_REPLY_FROM_MEMORY
    _, kwargs = client.responses.create.call_args
    assert kwargs["tool_choice"] == {"type": "function", "name": "route_strategy"}


@pytest.mark.asyncio
async def test_route_multi_turn_strategy_openai_defaults_when_no_tool_call(monkeypatch):
    client = _fake_openai_forced_tool_client(
        [_FakeOpenAIStreamEvent("response.completed", response=MagicMock())], monkeypatch,
    )

    result = await llm.route_multi_turn_strategy(
        client, "gpt-5-codex", "goal", "instruction", "example.com", "https://example.com", "", "", "openai",
    )

    assert result["chosen_strategy"] == "NEW_NAVIGATION"


# --- W19-4: llm.extract_structured_items() (Structured Data Extractor) ---

_EXTRACTED_ITEMS = [
    {"item_index": 1, "title": "Nike Men's Pegasus 42", "price": "฿4,200", "status": "In Stock", "url": "https://example.com/1"},
    {"item_index": 2, "title": "Adidas Ultraboost", "price": "฿5,500", "status": "In Stock", "url": "https://example.com/2"},
]


@pytest.mark.asyncio
async def test_extract_structured_items_returns_list_on_anthropic_success():
    client = MagicMock()
    client.messages.create = AsyncMock(
        return_value=_fake_anthropic_response(
            [_fake_anthropic_tool_use_block("emit_structured_items", {"items": _EXTRACTED_ITEMS})]
        )
    )

    result = await llm.extract_structured_items(
        client, "claude-x", "1. Nike Pegasus 42 - ฿4,200 - In Stock\n2. Adidas Ultraboost - ฿5,500 - In Stock",
        "รายการสินค้า", "anthropic",
    )

    assert result == _EXTRACTED_ITEMS
    _, kwargs = client.messages.create.call_args
    assert kwargs["tool_choice"] == {"type": "tool", "name": "emit_structured_items"}


@pytest.mark.asyncio
async def test_extract_structured_items_returns_empty_list_for_blank_content():
    result = await llm.extract_structured_items(MagicMock(), "claude-x", "   ", "", "anthropic")

    assert result == []


@pytest.mark.asyncio
async def test_extract_structured_items_returns_empty_list_when_no_tool_call():
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=_fake_anthropic_response([]))

    result = await llm.extract_structured_items(client, "claude-x", "some content", "", "anthropic")

    assert result == []


@pytest.mark.asyncio
async def test_extract_structured_items_swallows_provider_errors_and_returns_empty_list():
    client = MagicMock()
    client.messages.create = AsyncMock(side_effect=RuntimeError("API down"))

    result = await llm.extract_structured_items(client, "claude-x", "some content", "", "anthropic")

    assert result == []


@pytest.mark.asyncio
async def test_extract_structured_items_returns_empty_list_for_unknown_provider():
    result = await llm.extract_structured_items(MagicMock(), "model", "some content", "", "unknown")

    assert result == []


@pytest.mark.asyncio
async def test_extract_structured_items_groq_parses_items_from_model():
    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        return_value=_fake_response(
            [_fake_tool_call("call_1", "emit_structured_items", json.dumps({"items": _EXTRACTED_ITEMS}))], {},
        )
    )

    result = await llm.extract_structured_items(client, "llama-x", "some content", "", "groq")

    assert len(result) == 2
    assert result[0]["title"] == "Nike Men's Pegasus 42"


@pytest.mark.asyncio
async def test_extract_structured_items_returns_empty_list_when_items_field_is_not_a_list():
    client = MagicMock()
    client.messages.create = AsyncMock(
        return_value=_fake_anthropic_response([_fake_anthropic_tool_use_block("emit_structured_items", {"items": "not a list"})])
    )

    result = await llm.extract_structured_items(client, "claude-x", "some content", "", "anthropic")

    assert result == []


@pytest.mark.asyncio
async def test_extract_structured_items_openai_parses_items_from_model(monkeypatch):
    client = _fake_openai_forced_tool_client(
        [_FakeOpenAIStreamEvent(
            "response.output_item.done",
            item=_fake_openai_function_call_item("emit_structured_items", {"items": _EXTRACTED_ITEMS}),
        )],
        monkeypatch,
    )

    result = await llm.extract_structured_items(client, "gpt-5-codex", "some content", "", "openai")

    assert len(result) == 2
    assert result[0]["title"] == "Nike Men's Pegasus 42"


@pytest.mark.asyncio
async def test_extract_structured_items_openai_returns_empty_on_response_failed(monkeypatch):
    client = _fake_openai_forced_tool_client(
        [_FakeOpenAIStreamEvent("response.failed", response=MagicMock(error="boom"))], monkeypatch,
    )

    result = await llm.extract_structured_items(client, "gpt-5-codex", "some content", "", "openai")

    assert result == []


# --- W19 ("Navigation Deduplication"): llm.generate_plan() includes current_url + dedup rule ---


def _fake_text_response(text: str):
    block = MagicMock()
    block.type = "text"
    block.text = text
    return _fake_anthropic_response([block])


@pytest.mark.asyncio
async def test_generate_plan_includes_current_url_and_dedup_rule_in_prompt():
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=_fake_text_response("1. Search for the user"))

    result = await llm.generate_plan(
        client, "claude-x", "หน้า Admin", "[1] input 'Search'", "anthropic",
        current_url="https://example.com/admin/viewSystemUsers",
    )

    assert result == "1. Search for the user"
    _, kwargs = client.messages.create.call_args
    prompt = kwargs["messages"][0]["content"]
    assert "https://example.com/admin/viewSystemUsers" in prompt
    assert "Navigation Deduplication" in prompt


@pytest.mark.asyncio
async def test_generate_plan_shows_placeholder_when_current_url_not_provided():
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=_fake_text_response("1. Do X"))

    await llm.generate_plan(client, "claude-x", "goal", "page text", "anthropic")

    _, kwargs = client.messages.create.call_args
    prompt = kwargs["messages"][0]["content"]
    assert "unknown — no page is open yet" in prompt


# --- W20 ("Context-Aware Implicit Execution", บั๊กจริงที่ user รายงาน): llm.generate_plan()
# ต้องรวมเทิร์นก่อนหน้าเข้า prompt ให้ LLM แก้คำอ้างอิงกำกวมอย่าง "เปิดให้หน่อย" ได้ ---


@pytest.mark.asyncio
async def test_generate_plan_includes_previous_turn_context_when_provided():
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=_fake_text_response("1. Open YouTube\n2. Search\n3. Play"))

    await llm.generate_plan(
        client, "claude-x", "okเปิดให้หน่อย", "", "anthropic",
        previous_user_goal="ขอเพลงเศร้าๆหน่อย",
        previous_assistant_message='แนะนำเพลง "โปรดส่งใครมารักฉันที" ครับ',
    )

    _, kwargs = client.messages.create.call_args
    prompt = kwargs["messages"][0]["content"]
    assert "ขอเพลงเศร้าๆหน่อย" in prompt
    assert "โปรดส่งใครมารักฉันที" in prompt
    assert "Earlier conversation in this session" in prompt
    assert "Context-Aware Implicit Execution" in prompt
    assert "Complete Execution on Content Platforms" in prompt


@pytest.mark.asyncio
async def test_generate_plan_omits_previous_turn_section_when_not_provided():
    """เทิร์นแรกของ session (ไม่มีเทิร์นก่อนหน้าจริงๆ) — ต้องไม่มี "Earlier conversation in this session" ตัวจริง
    (ที่กรอกข้อมูล User/Assistant มาให้) แทรกอยู่ในพรอมต์เลย (คงพฤติกรรมเดิมทุกประการก่อนมี
    feature นี้) — instruction ทั่วไปที่ *พูดถึง* คำว่า "Earlier conversation in this session" (บอกว่าให้ไปดูตรงนั้น
    ถ้ามี) ยังคงอยู่เสมอ ไม่ใช่สิ่งที่เทสต์นี้เช็ค เช็คเฉพาะ header ของ block ข้อมูลจริงที่ควร
    หายไปเมื่อไม่มีเทิร์นก่อนหน้า"""
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=_fake_text_response("1. Do X"))

    await llm.generate_plan(client, "claude-x", "goal", "page text", "anthropic")

    _, kwargs = client.messages.create.call_args
    prompt = kwargs["messages"][0]["content"]
    assert "เทิร์นล่าสุดก่อนหน้า Goal นี้" not in prompt


# --- W19-5: llm.normalize_extraction_query() (Structured Data Extractor Engine) ---

_NORMALIZED_QUERY = {
    "normalized_target_scope": "div.oxd-table-body",
    "extraction_type": "TABLE_MULTI_ROW",
    "data_fields": ["Username", "User Role", "Employee Name", "Status"],
}


@pytest.mark.asyncio
async def test_normalize_extraction_query_returns_decision_on_anthropic_success():
    client = MagicMock()
    client.messages.create = AsyncMock(
        return_value=_fake_anthropic_response([_fake_anthropic_tool_use_block("emit_normalized_query", _NORMALIZED_QUERY)])
    )

    result = await llm.normalize_extraction_query(
        client, "claude-x", "อ่านรายชื่อผู้ใช้งานระบบในหน้าแอดมินทั้งหมด", "div.oxd-table-body", "anthropic",
    )

    assert result == _NORMALIZED_QUERY
    _, kwargs = client.messages.create.call_args
    assert kwargs["tool_choice"] == {"type": "tool", "name": "emit_normalized_query"}


@pytest.mark.asyncio
async def test_normalize_extraction_query_returns_empty_string_for_blank_query():
    result = await llm.normalize_extraction_query(MagicMock(), "claude-x", "   ", "", "anthropic")

    assert result["normalized_target_scope"] == ""
    assert result["data_fields"] == []


@pytest.mark.asyncio
async def test_normalize_extraction_query_defaults_to_table_multi_row_when_no_tool_call():
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=_fake_anthropic_response([]))

    result = await llm.normalize_extraction_query(client, "claude-x", "list all users", "table", "anthropic")

    assert result["extraction_type"] == "TABLE_MULTI_ROW"
    assert result["normalized_target_scope"] == ""


@pytest.mark.asyncio
async def test_normalize_extraction_query_swallows_provider_errors_and_falls_back():
    client = MagicMock()
    client.messages.create = AsyncMock(side_effect=RuntimeError("API down"))

    result = await llm.normalize_extraction_query(client, "claude-x", "list all users", "table", "anthropic")

    assert result == llm._EXTRACTION_QUERY_SAFE_DEFAULT


@pytest.mark.asyncio
async def test_normalize_extraction_query_defaults_for_unknown_provider():
    result = await llm.normalize_extraction_query(MagicMock(), "model", "list all users", "table", "unknown")

    assert result["extraction_type"] == "TABLE_MULTI_ROW"


@pytest.mark.asyncio
async def test_normalize_extraction_query_groq_parses_result_from_model():
    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        return_value=_fake_response(
            [_fake_tool_call("call_1", "emit_normalized_query", json.dumps(_NORMALIZED_QUERY))], {},
        )
    )

    result = await llm.normalize_extraction_query(client, "llama-x", "list all users", "table", "groq")

    assert result["normalized_target_scope"] == "div.oxd-table-body"
    assert result["data_fields"] == ["Username", "User Role", "Employee Name", "Status"]

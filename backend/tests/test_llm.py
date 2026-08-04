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
    assert "favor การนับตรงๆ เสมอ" in llm.SYSTEM_PROMPT


# --- ป้องกัน agent ยอมแพ้เร็วเกินไป: ต้องลองค้นหาก่อนสรุปว่า "ไม่พบ" ---


def test_system_prompt_requires_trying_search_before_reporting_not_found():
    assert "ก่อนเรียก finish_task พร้อมข้อความทำนอง" in llm.SYSTEM_PROMPT
    assert "ต้องเรียก action ที่มีอยู่" in llm.SYSTEM_PROMPT
    assert "อย่างน้อย 1 ครั้งก่อนเสมอ ถึงจะ finish_task ว่าไม่พบได้" in llm.SYSTEM_PROMPT


def test_system_prompt_treats_verbless_questions_as_implicit_search_command():
    assert 'ห้ามตีความว่าเป็น' in llm.SYSTEM_PROMPT
    assert "นับเป็นคำสั่งให้ค้นหาโดยปริยาย" in llm.SYSTEM_PROMPT


# --- hover: ปุ่ม hover-to-reveal ที่ perception.py ติด label marker ให้แล้ว ---


def test_browser_action_schema_includes_hover_type():
    type_enum = llm._BROWSER_ACTION_PARAMS["properties"]["type"]["enum"]
    assert "hover" in type_enum


def test_system_prompt_instructs_hover_before_clicking_hidden_reveal_elements():
    assert "[ซ่อนอยู่ — อาจต้อง hover แถวก่อน]" in llm.SYSTEM_PROMPT
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

    tool_name, tool_input, tool_use_id, _, usage = await llm.next_action(client, "model", "goal", "page", [])

    assert tool_name == "finish_task"
    assert tool_input["success"] is False
    assert tool_use_id == ""
    assert usage == llm.TokenUsage(input_tokens=5, output_tokens=3)


@pytest.mark.asyncio
async def test_next_action_passes_manual_context_into_prompt():
    """W6[B]: manual_context จาก retriever.retrieve() ต้องโผล่ในข้อความ user turn จริง"""
    block = _fake_anthropic_tool_use_block("browser_action", {"type": "wait"})
    response = _fake_anthropic_response([block])
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=response)

    await llm.next_action(client, "model", "goal", "page", [], manual_context="- chunk one")

    _, kwargs = client.messages.create.call_args
    user_content = kwargs["messages"][-1]["content"]
    assert "chunk one" in user_content
    assert "ข้อมูลอ้างอิงจากคู่มือที่เกี่ยวข้อง" in user_content


@pytest.mark.asyncio
async def test_next_action_default_manual_context_omits_section():
    """เรียกแบบเดิม (5 args ไม่มี manual_context) ต้องได้ prompt แบบเดิมเป๊ะ ไม่มี section คู่มือ"""
    block = _fake_anthropic_tool_use_block("browser_action", {"type": "wait"})
    response = _fake_anthropic_response([block])
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=response)

    await llm.next_action(client, "model", "goal", "page", [])

    _, kwargs = client.messages.create.call_args
    user_content = kwargs["messages"][-1]["content"]
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
    user_content = kwargs["messages"][-1]["content"]
    assert "[FAIL] boom" in user_content
    assert "Action ที่เคยลองแล้วล้มเหลว" in user_content


@pytest.mark.asyncio
async def test_next_action_default_memory_context_omits_section():
    """เรียกแบบเดิม (ไม่มี memory_context) ต้องได้ prompt แบบเดิมเป๊ะ ไม่มี section ประวัติ failure"""
    block = _fake_anthropic_tool_use_block("browser_action", {"type": "wait"})
    response = _fake_anthropic_response([block])
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=response)

    await llm.next_action(client, "model", "goal", "page", [])

    _, kwargs = client.messages.create.call_args
    user_content = kwargs["messages"][-1]["content"]
    assert "ทำซ้ำ" not in user_content


@pytest.mark.asyncio
async def test_next_action_passes_plan_context_into_prompt():
    """W43: plan_context (แผนที่ user ยืนยันแล้ว) ต้องโผล่ในข้อความ user turn จริง เป็น
    section แยก "แพลนปัจจุบัน" """
    block = _fake_anthropic_tool_use_block("browser_action", {"type": "wait"})
    response = _fake_anthropic_response([block])
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=response)

    await llm.next_action(client, "model", "goal", "page", [], plan_context="1. ทำ X\n2. ทำ Y")

    _, kwargs = client.messages.create.call_args
    user_content = kwargs["messages"][-1]["content"]
    assert "1. ทำ X" in user_content
    assert "แพลนปัจจุบัน" in user_content


@pytest.mark.asyncio
async def test_next_action_default_plan_context_omits_section():
    """W43: ad-hoc task (ไม่ผ่าน Confirm plan เลย) ไม่ควรมี section "แพลนปัจจุบัน" โผล่มา
    ปนใน prompt เลย — backward compatible กับ task ที่ไม่มีแผน"""
    block = _fake_anthropic_tool_use_block("browser_action", {"type": "wait"})
    response = _fake_anthropic_response([block])
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=response)

    await llm.next_action(client, "model", "goal", "page", [])

    _, kwargs = client.messages.create.call_args
    user_content = kwargs["messages"][-1]["content"]
    assert "แพลนปัจจุบัน" not in user_content


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
    assert messages[0] == {"role": "system", "content": llm.SYSTEM_PROMPT}
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
    assert usage == llm.TokenUsage(input_tokens=20, output_tokens=10)


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
    assert usage == llm.TokenUsage(input_tokens=10 * llm._GROQ_NO_TOOL_CALL_RETRIES, output_tokens=5 * llm._GROQ_NO_TOOL_CALL_RETRIES)


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
    assert kwargs["system_instruction"] == llm.SYSTEM_PROMPT
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

    assert tool_name == "finish_task"
    assert tool_input["success"] is False
    assert tool_use_id == ""
    assert usage == llm.TokenUsage(input_tokens=10, output_tokens=5)


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
_FIXED_TIME_TEXT = "วันศุกร์ที่ 31 กรกฎาคม 2569 เวลา 14:32 น."
_TIME_LINE = f"\n\nเวลาปัจจุบัน (Asia/Bangkok): {_FIXED_TIME_TEXT}"
_EXPECTED_PREFIX = f"Goal: goal{_TIME_LINE}\n\nหน้าเว็บปัจจุบัน:\npage"


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
    assert "Action ที่เคยลองแล้วล้มเหลว" in result


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
    assert result.index("https://example.com/cart") < result.index("หน้าเว็บปัจจุบัน:\npage-text")


def test_build_user_turn_text_omits_action_history_section_when_empty():
    result = llm._build_user_turn_text("goal", "page", action_history_context="")

    assert result == _EXPECTED_PREFIX


def test_build_user_turn_text_includes_action_history_section_when_provided():
    result = llm._build_user_turn_text(
        "goal", "page", action_history_context="- step 3: {'type': 'click'} -> [OK]",
    )

    assert result.startswith(_EXPECTED_PREFIX)
    assert "step 3" in result
    assert "Action ล่าสุดที่คุณเพิ่งทำไป" in result


# --- W43: plan_context ("แพลนปัจจุบัน") ---


def test_build_user_turn_text_omits_plan_section_when_empty():
    """ad-hoc task (ไม่ผ่าน Confirm plan) ได้ plan_context="" เสมอ — ต้องได้ prompt เดิม
    เป๊ะทุกตัวอักษร ไม่มี section แผนโผล่มาปนเลย (backward compatible)"""
    result = llm._build_user_turn_text("goal", "page", plan_context="")

    assert result == _EXPECTED_PREFIX


def test_build_user_turn_text_includes_plan_section_when_provided():
    result = llm._build_user_turn_text("goal", "page", plan_context="1. ทำ X\n2. ทำ Y")

    assert "แพลนปัจจุบัน" in result
    assert "1. ทำ X\n2. ทำ Y" in result
    # อยู่ก่อน "หน้าเว็บปัจจุบัน" (เป็นบริบทระดับ task เหมือน Goal ไม่ใช่ข้อมูลเฉพาะ step นี้)
    assert result.index("แพลนปัจจุบัน") < result.index("หน้าเว็บปัจจุบัน")


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

    expected_weekday = llm._THAI_WEEKDAYS[fixed.weekday()]
    expected_month = llm._THAI_MONTHS[fixed.month - 1]
    # ปี พ.ศ. = ค.ศ. + 543 (2026 -> 2569) เวลา 24 ชม. ตรงกับที่ mock ไว้เป๊ะ (14:32)
    assert result == f"{expected_weekday}ที่ 31 {expected_month} 2569 เวลา 14:32 น."


def test_build_user_turn_text_injects_current_bangkok_time_from_mocked_now(monkeypatch):
    """mock datetime.now() ที่ระดับต่ำสุด (ไม่ใช่ mock helper function) — ยืนยันว่า context
    ที่ build ออกมาจริงมีบรรทัดเวลาตรงกับ mock value, format/timezone ถูกต้อง"""
    monkeypatch.undo()  # ปลด autouse fixture (_freeze_bangkok_time) ก่อน — เทสต์นี้ต้องการให้
    # _current_bangkok_time_text() ตัวจริงทำงาน (อ่านจาก datetime.now() ที่ mock ด้านล่างแทน)
    fixed = datetime(2026, 12, 25, 9, 5, tzinfo=ZoneInfo("Asia/Bangkok"))
    monkeypatch.setattr(llm, "datetime", _FrozenDateTime(fixed))

    result = llm._build_user_turn_text("goal", "page")

    expected_weekday = llm._THAI_WEEKDAYS[fixed.weekday()]
    assert f"เวลาปัจจุบัน (Asia/Bangkok): {expected_weekday}ที่ 25 ธันวาคม 2569 เวลา 09:05 น." in result


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

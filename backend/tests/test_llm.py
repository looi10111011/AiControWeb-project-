from unittest.mock import AsyncMock, MagicMock

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


def test_build_user_turn_text_omits_manual_section_when_empty():
    result = llm._build_user_turn_text("goal", "page")

    assert result == "Goal: goal\n\nหน้าเว็บปัจจุบัน:\npage"


def test_build_user_turn_text_includes_manual_section_when_provided():
    result = llm._build_user_turn_text("goal", "page", "- chunk one\n- chunk two")

    assert result.startswith("Goal: goal\n\nหน้าเว็บปัจจุบัน:\npage")
    assert "chunk one" in result
    assert "chunk two" in result


def test_build_user_turn_text_omits_memory_section_when_empty():
    result = llm._build_user_turn_text("goal", "page", manual_context="", memory_context="")

    assert result == "Goal: goal\n\nหน้าเว็บปัจจุบัน:\npage"


def test_build_user_turn_text_includes_memory_section_when_provided():
    result = llm._build_user_turn_text("goal", "page", memory_context="- {'type': 'click'} -> [FAIL] boom")

    assert result.startswith("Goal: goal\n\nหน้าเว็บปัจจุบัน:\npage")
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

    assert result == "Goal: goal\n\nหน้าเว็บปัจจุบัน:\npage"


def test_build_user_turn_text_includes_vision_section_when_provided():
    result = llm._build_user_turn_text("goal", "page", vision_context="เห็น cookie banner บังปุ่ม Login อยู่")

    assert result.startswith("Goal: goal\n\nหน้าเว็บปัจจุบัน:\npage")
    assert "เห็น cookie banner บังปุ่ม Login อยู่" in result

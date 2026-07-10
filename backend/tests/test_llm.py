from unittest.mock import AsyncMock, MagicMock

import pytest
from groq import BadRequestError as GroqBadRequestError

from backend.app.core import llm

# ทุกเทสต์ mock ทั้ง AsyncGroq client — ไม่ยิง Groq API จริง


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

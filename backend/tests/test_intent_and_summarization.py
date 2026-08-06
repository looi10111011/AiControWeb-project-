import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.app.core import llm
from backend.app.core.actions import ActionResult
from backend.app.core.orchestrator import Orchestrator


# --- W19-6 ("Master Controller" MODULE 1, "General QA / No-Browser Trigger") ---


@pytest.mark.parametrize("goal", [
    "วันนี้วันที่เท่าไหร่",
    "เวลาเท่าไหร่ตอนนี้",
    "ตอนนี้กี่โมงแล้ว",
    "1+1 ได้เท่าไหร่",
    "2*3=",
    "สวัสดีครับ",
    "hi",
    "hello",
])
def test_is_general_chat_query_recognizes_greetings_time_and_math(goal):
    assert llm.is_general_chat_query(goal) is True


@pytest.mark.parametrize("goal", [
    "เข้าไปหน้า Admin แล้วอ่านรายชื่อผู้ใช้",
    "ราคาสินค้าชิ้นนี้เท่าไหร่",
    "มีผู้ใช้กี่คนในหน้า Admin",
    "hi, ช่วยค้นหา iPhone ให้หน่อย",
    "ไปหน้า 2",
    "คลิกปุ่ม login",
    "",
    "   ",
])
def test_is_general_chat_query_rejects_anything_web_related_or_ambiguous(goal):
    assert llm.is_general_chat_query(goal) is False


# pdf/xlsx: llm.goal_mentions_web_action() — factored out of is_general_chat_query()'s
# exclusion-keyword gate above so routes.py can reuse the same check for the
# "file-chat memory follow-up" decision (see test_api.py)
@pytest.mark.parametrize("goal", [
    "เข้าไปหน้า Admin แล้วอ่านรายชื่อผู้ใช้",
    "ค้นหา iPhone ให้หน่อย",
    "คลิกปุ่ม login",
    "ไปที่ https://example.com",
])
def test_goal_mentions_web_action_true_for_web_keywords(goal):
    assert llm.goal_mentions_web_action(goal) is True


@pytest.mark.parametrize("goal", [
    "แต่ละวันทำอะไรบ้าง",
    "ยอดรวมเท่าไหร่",
    "",
    "   ",
])
def test_goal_mentions_web_action_false_for_plain_followup_questions(goal):
    assert llm.goal_mentions_web_action(goal) is False


@pytest.mark.asyncio
async def test_chat_response_returns_text_on_anthropic_success():
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "สวัสดีครับ"
    response = MagicMock()
    response.content = [text_block]
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=response)

    result = await llm.chat_response(client, "claude-x", "สวัสดี", "anthropic")

    assert result == "สวัสดีครับ"


@pytest.mark.asyncio
async def test_chat_response_includes_current_time_in_prompt_when_provided():
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "วันนี้วันจันทร์ครับ"
    response = MagicMock()
    response.content = [text_block]
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=response)

    await llm.chat_response(client, "claude-x", "วันนี้วันอะไร", "anthropic", current_time_text="วันจันทร์ที่ 1 มกราคม 2569")

    _, kwargs = client.messages.create.call_args
    prompt = kwargs["messages"][0]["content"]
    assert "วันจันทร์ที่ 1 มกราคม 2569" in prompt


@pytest.mark.asyncio
async def test_chat_response_swallows_provider_errors_and_returns_apology():
    client = MagicMock()
    client.messages.create = AsyncMock(side_effect=RuntimeError("API down"))

    result = await llm.chat_response(client, "claude-x", "สวัสดี", "anthropic")

    assert "ขออภัย" in result


@pytest.mark.asyncio
async def test_chat_response_returns_message_for_unknown_provider():
    result = await llm.chat_response(MagicMock(), "model", "สวัสดี", "unknown")

    assert result != ""


@pytest.mark.asyncio
async def test_classify_intent_heuristics():
    """ทดสอบ classify_intent ด้วยคำค้นหาประเภท QA และ Action"""
    client = MagicMock()

    # QA intent keywords
    qa_goal = "ช่วยสรุปเนื้อหาสำคัญของหน้านี้ให้ฟังหน่อย"
    intent = await llm.classify_intent(client, "mock-model", qa_goal, page_text="", provider="gemini")
    assert intent == "qa_summary"

    # Action intent keywords
    action_goal = "คลิกปุ่ม ล็อกอิน บนหน้าเว็บ"
    intent = await llm.classify_intent(client, "mock-model", action_goal, page_text="", provider="gemini")
    assert intent == "action_task"


@pytest.mark.asyncio
async def test_classify_intent_llm_fallback():
    """ทดสอบ classify_intent เมื่อต้องพึ่งพา LLM ในการแยก Intent"""
    client = MagicMock()
    with patch("backend.app.core.llm.generate_text", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = "qa_summary"
        ambiguous_goal = "บริษัท ABC"
        intent = await llm.classify_intent(client, "mock-model", ambiguous_goal, page_text="บริษัท ABC จำกัด...", provider="gemini")
        assert intent == "qa_summary"
        mock_gen.assert_called_once()


# --- W19 ("Intent Classification Router"): compound nav+read commands must always route
# to action_task, deterministically, without needing the LLM fallback ---


@pytest.mark.asyncio
async def test_classify_intent_compound_navigation_and_read_command_is_action_task():
    client = MagicMock()

    goal = "เข้าไปหน้า Admin แล้วอ่านรายชื่อผู้ใช้งานระบบ"
    intent = await llm.classify_intent(client, "mock-model", goal, page_text="", provider="gemini")

    assert intent == "action_task"


@pytest.mark.asyncio
async def test_classify_intent_compound_command_does_not_need_llm_call():
    """ต้องตัดสินใจแบบ deterministic ไม่เรียก LLM เลยสำหรับ compound command ที่ชัดเจน"""
    client = MagicMock()
    with patch("backend.app.core.llm.generate_text", new_callable=AsyncMock) as mock_gen:
        goal = "เปิดเว็บ Admin แล้วสรุปข้อมูลตาราง"
        intent = await llm.classify_intent(client, "mock-model", goal, page_text="", provider="gemini")

        assert intent == "action_task"
        mock_gen.assert_not_called()


@pytest.mark.asyncio
async def test_classify_intent_recognizes_expanded_navigation_verbs():
    """"เข้าไปหน้า"/"เปิดเว็บ" ไม่ match "ไปที่" เดิมเลย — ต้องถูกจับเป็น action ด้วย"""
    client = MagicMock()

    for goal in ["เข้าไปหน้า Dashboard", "เปิดเว็บ shopee แล้วดูสินค้า"]:
        intent = await llm.classify_intent(client, "mock-model", goal, page_text="", provider="gemini")
        assert intent == "action_task", f"goal={goal!r} ควรเป็น action_task"


@pytest.mark.asyncio
async def test_classify_intent_pure_qa_with_incidental_action_word_still_qa_summary():
    """กันไม่ให้ compound rule (2.5) ทำให้ step 2 (goal ที่ขึ้นต้นด้วยคำถามล้วนๆ) เพี้ยนไป —
    "login" เป็น action keyword แต่ที่นี่เป็นแค่หัวข้อที่ถูกถาม ไม่ใช่คำสั่งให้ทำ action"""
    client = MagicMock()

    goal = "สรุปขั้นตอนการ login ที่เขียนอธิบายไว้ในหน้านี้ให้หน่อย"
    intent = await llm.classify_intent(client, "mock-model", goal, page_text="", provider="gemini")

    assert intent == "qa_summary"



@pytest.mark.asyncio
async def test_summarize_page_prompt():
    """ทดสอบว่า summarize_page ใช้ System Prompt รูปแบบที่กำหนดเป๊ะๆ"""
    client = MagicMock()
    with patch("backend.app.core.llm.generate_text", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = "นี่คือสรุปเนื้อหาเว็บ"
        page_text = "หน้าเว็บขายรองเท้าแตะยาง ราคา 199 บาท"
        user_prompt = "สรุปราคาสินค้า"

        result = await llm.summarize_page(client, "mock-model", page_text, user_prompt, provider="gemini")
        assert result == "นี่คือสรุปเนื้อหาเว็บ"

        prompt_sent = mock_gen.call_args[0][2]
        assert "คุณคือ AI Assistant ที่มีความสามารถในการอ่านหน้าเว็บ" in prompt_sent
        assert "โปรดอ่านเนื้อหาเว็บต่อไปนี้แล้วตอบคำถามของผู้ใช้ให้กระชับ เข้าใจง่าย เป็นกันเอง" in prompt_sent
        # W20 (follow-up "reply in the user's own language"): summarize_page() used to
        # hard-require Thai output regardless of the user's own question language — now
        # mirrors it instead (see llm._LANGUAGE_MIRROR_RULE), same as every other
        # response-generating prompt in this module.
        assert "ตอบเป็นภาษาเดียวกับที่ user ใช้พิมพ์คำถาม" in prompt_sent
        assert "Page Content: หน้าเว็บขายรองเท้าแตะยาง ราคา 199 บาท" in prompt_sent
        assert "User Question: สรุปราคาสินค้า" in prompt_sent


@pytest.mark.asyncio
async def test_orchestrator_qa_summary_workflow():
    """ทดสอบว่า Orchestrator รัน task ที่มี Intent เป็น qa_summary แล้วคืน status: chat_reply
    ทันที — mock llm.next_action_gemini() (W44 mini-loop เรียกจริง) ให้เรียก finish_task ทันที
    ด้วยข้อความว่างเปล่า (จำลองว่าโมเดลยังไม่ได้คำตอบที่ใช้ได้) เพื่อบังคับให้ตกไป fallback
    summarize_page() แบบ deterministic — เดิมเทสต์นี้ไม่ mock next_action เลย ปล่อยให้ยิง
    Gemini API จริง (ไม่ deterministic ขึ้นกับพฤติกรรมโมเดลจริง/เน็ตเวิร์ก) ซึ่งพังง่ายเมื่อ
    ปรับ _QA_SUMMARY_MAX_STEPS หรือ prompt เปลี่ยนแม้เพียงเล็กน้อย"""
    orchestrator = Orchestrator()
    mock_page = AsyncMock()
    mock_page.url = "http://example.com"

    with patch("backend.app.core.orchestrator.get_snapshot", new_callable=AsyncMock) as mock_snapshot, \
         patch("backend.app.core.orchestrator.goto", new_callable=AsyncMock) as mock_goto, \
         patch("backend.app.core.orchestrator.wait_stable", new_callable=AsyncMock), \
         patch("backend.app.core.llm.classify_intent", new_callable=AsyncMock) as mock_classify, \
         patch("backend.app.core.llm.summarize_page", new_callable=AsyncMock) as mock_summarize, \
         patch("backend.app.core.llm.next_action_gemini", new_callable=AsyncMock) as mock_next_action:

        mock_snapshot.return_value = ([], "ตัวอย่างเนื้อหาเว็บไซต์เกี่ยวกับข่าว IT")
        mock_goto.return_value = MagicMock(success=True)
        mock_classify.return_value = "qa_summary"
        mock_summarize.return_value = "สรุปข่าว IT ประจำวัน: มีการเปิดตัวชิปใหม่"
        mock_next_action.return_value = ("finish_task", {"success": False}, "call-1", [], llm.TokenUsage())

        events = []
        async def on_event(ev):
            events.append(ev)

        result = await orchestrator.run_task(
            url="http://example.com",
            goal="สรุปข่าวในหน้านี้ให้หน่อย",
            page=mock_page,
            on_event=on_event,
            provider="gemini",
        )

        assert result["status"] == "chat_reply"
        assert result["success"] is True
        assert result["message"] == "สรุปข่าว IT ประจำวัน: มีการเปิดตัวชิปใหม่"
        assert any(ev.get("kind") == "chat_reply" for ev in events)
        mock_summarize.assert_awaited_once()


@pytest.mark.asyncio
async def test_orchestrator_qa_summary_allows_search_flow_before_giving_up():
    """ต่อยอด W44: qa_summary mini-loop เดิมอนุญาตแค่ read_page_data ทำให้คำถามที่คำตอบอยู่
    หลัง search flow (ต้อง fill ช่องค้นหา + click ปุ่มค้นหาก่อน) ตอบไม่ได้เลย — ตอนนี้ต้อง
    อนุญาต fill/click กับ element ที่ label ดูเป็นช่อง/ปุ่มค้นหาจริงๆ ด้วย จำลอง flow ค้นหา
    จริง: fill ช่องค้นหา -> click ปุ่มค้นหา -> read_page_data อ่านผลลัพธ์ -> finish_task"""
    orchestrator = Orchestrator()
    mock_page = AsyncMock()
    mock_page.url = "http://example.com"

    elements_before_search = [
        {"index": 0, "tag": "input", "type": "text", "label": "Search users"},
        {"index": 1, "tag": "button", "type": "", "label": "Search"},
    ]
    elements_after_search = elements_before_search + [
        {"index": 2, "tag": "table", "type": "", "label": "Results"},
    ]

    next_action_calls = [
        ("browser_action", {"type": "fill", "index": 0, "text": "Cierra Vega"}, "t1", [], llm.TokenUsage()),
        ("browser_action", {"type": "click", "index": 1}, "t2", [], llm.TokenUsage()),
        ("browser_action", {"type": "read_page_data", "query": "อายุเท่าไหร่", "target_hint": "#results"}, "t3", [], llm.TokenUsage()),
        ("finish_task", {"success": True, "message": "อายุ 32 ปี"}, "", [], llm.TokenUsage()),
    ]

    with patch("backend.app.core.orchestrator.get_snapshot", new_callable=AsyncMock) as mock_snapshot, \
         patch("backend.app.core.orchestrator.goto", new_callable=AsyncMock) as mock_goto, \
         patch("backend.app.core.orchestrator.wait_stable", new_callable=AsyncMock), \
         patch("backend.app.core.llm.classify_intent", new_callable=AsyncMock, return_value="qa_summary"), \
         patch("backend.app.core.orchestrator.llm.next_action", AsyncMock(side_effect=next_action_calls)), \
         patch("backend.app.core.orchestrator.execute", new_callable=AsyncMock) as mock_execute:

        mock_snapshot.side_effect = [
            (elements_before_search, "หน้าค้นหา"),   # initial snapshot ก่อนเข้า mini-loop
            (elements_after_search, "ผลการค้นหา"),   # re-snapshot หลัง fill
            (elements_after_search, "ผลการค้นหา"),   # re-snapshot หลัง click
        ]
        mock_goto.return_value = MagicMock(success=True)
        mock_execute.return_value = ActionResult(True, "ok", "สำเร็จ")

        result = await orchestrator.run_task(
            url="http://example.com",
            goal="หาอายุของ Cierra Vega",
            page=mock_page,
            provider="anthropic",
        )

    assert result["status"] == "chat_reply"
    assert result["message"] == "อายุ 32 ปี"
    assert mock_execute.await_count == 3  # fill + click + read_page_data ถูก dispatch จริงทั้งคู่
    dispatched_types = [call.args[1]["type"] for call in mock_execute.await_args_list]
    assert dispatched_types == ["fill", "click", "read_page_data"]


# --- W19 ("Guard Compatibility Rule"): qa_summary ต้องยอม click เมนู/nav เพื่อไปหน้าย่อยที่
# มีคำตอบก่อน ไม่ใช่ปฏิเสธทุกครั้งเหมือน W44/W46 เดิมที่อนุญาตแค่ fill/click ช่องค้นหา ---


@pytest.mark.asyncio
async def test_orchestrator_qa_summary_allows_navigation_click_to_reach_sub_page():
    """คำถามที่คำตอบอยู่หลังการ navigate ไปหน้าย่อย (เช่น "มีผู้ใช้กี่คนในหน้า Admin" ทั้งที่
    ยังไม่ได้อยู่หน้า Admin) ต้องคลิกเมนู "Admin" (region=navigation) ได้ ไม่ถูกปฏิเสธเหมือน
    ปุ่มที่ไม่เกี่ยวกับการค้นหาทั่วไป"""
    orchestrator = Orchestrator()
    mock_page = AsyncMock()
    mock_page.url = "http://example.com"

    elements_before_nav = [
        {"index": 0, "tag": "a", "type": "", "label": "Admin", "region": "navigation"},
    ]
    elements_after_nav = [
        {"index": 1, "tag": "table", "type": "", "label": "Users", "region": "main"},
    ]

    next_action_calls = [
        ("browser_action", {"type": "click", "index": 0}, "t1", [], llm.TokenUsage()),
        ("browser_action", {"type": "read_page_data", "query": "กี่คน", "target_hint": "table"}, "t2", [], llm.TokenUsage()),
        ("finish_task", {"success": True, "message": "มีผู้ใช้ 5 คน"}, "", [], llm.TokenUsage()),
    ]

    with patch("backend.app.core.orchestrator.get_snapshot", new_callable=AsyncMock) as mock_snapshot, \
         patch("backend.app.core.orchestrator.goto", new_callable=AsyncMock) as mock_goto, \
         patch("backend.app.core.orchestrator.wait_stable", new_callable=AsyncMock), \
         patch("backend.app.core.llm.classify_intent", new_callable=AsyncMock, return_value="qa_summary"), \
         patch("backend.app.core.orchestrator.llm.next_action", AsyncMock(side_effect=next_action_calls)), \
         patch("backend.app.core.orchestrator.execute", new_callable=AsyncMock) as mock_execute:

        mock_snapshot.side_effect = [
            (elements_before_nav, "หน้าแรก"),
            (elements_after_nav, "หน้า Admin"),
        ]
        mock_goto.return_value = MagicMock(success=True)
        mock_execute.return_value = ActionResult(True, "ok", "สำเร็จ")

        result = await orchestrator.run_task(
            url="http://example.com",
            goal="มีผู้ใช้กี่คนในหน้า Admin",
            page=mock_page,
            provider="anthropic",
        )

    assert result["status"] == "chat_reply"
    assert result["message"] == "มีผู้ใช้ 5 คน"
    assert mock_execute.await_count == 2  # click(nav) + read_page_data ถูก dispatch จริงทั้งคู่
    dispatched_types = [call.args[1]["type"] for call in mock_execute.await_args_list]
    assert dispatched_types == ["click", "read_page_data"]


@pytest.mark.asyncio
async def test_orchestrator_qa_summary_still_rejects_click_on_main_region_non_search_button():
    """click ที่ region="main" (ไม่ใช่ navigation) และ label ไม่เกี่ยวกับค้นหาเลย ต้องยัง
    ถูกปฏิเสธเหมือนเดิม — ผ่อนแค่ nav-region เท่านั้น ไม่ใช่ click ทุกกรณี"""
    orchestrator = Orchestrator()
    mock_page = AsyncMock()
    mock_page.url = "http://example.com"

    elements = [{"index": 0, "tag": "button", "type": "", "label": "Delete Account", "region": "main"}]

    next_action_calls = [
        ("browser_action", {"type": "click", "index": 0}, "t1", [], llm.TokenUsage()),
        ("finish_task", {"success": False}, "", [], llm.TokenUsage()),
    ]

    with patch("backend.app.core.orchestrator.get_snapshot", new_callable=AsyncMock) as mock_snapshot, \
         patch("backend.app.core.orchestrator.goto", new_callable=AsyncMock) as mock_goto, \
         patch("backend.app.core.orchestrator.wait_stable", new_callable=AsyncMock), \
         patch("backend.app.core.llm.classify_intent", new_callable=AsyncMock, return_value="qa_summary"), \
         patch("backend.app.core.llm.summarize_page", new_callable=AsyncMock, return_value="สรุป"), \
         patch("backend.app.core.orchestrator.llm.next_action", AsyncMock(side_effect=next_action_calls)), \
         patch("backend.app.core.orchestrator.execute", new_callable=AsyncMock) as mock_execute:

        mock_snapshot.return_value = (elements, "หน้าตั้งค่า")
        mock_goto.return_value = MagicMock(success=True)

        await orchestrator.run_task(
            url="http://example.com",
            goal="มีปุ่มอะไรบ้างในหน้านี้",
            page=mock_page,
            provider="anthropic",
        )

    mock_execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_orchestrator_qa_summary_still_rejects_fill_on_non_search_elements():
    """label ที่ไม่เกี่ยวกับการค้นหาเลย (เช่น "Username") ต้องยังถูกปฏิเสธเหมือน W44 เดิม —
    ผ่อนให้เฉพาะ element ที่ดูเป็นช่อง/ปุ่มค้นหาจริงๆ เท่านั้น ไม่ใช่เปิดให้ fill/click ได้
    ทุกกรณีแบบไม่จำกัดเงื่อนไข"""
    orchestrator = Orchestrator()
    mock_page = AsyncMock()
    mock_page.url = "http://example.com"

    elements = [{"index": 0, "tag": "input", "type": "text", "label": "Username"}]

    next_action_calls = [
        ("browser_action", {"type": "fill", "index": 0, "text": "admin"}, "t1", [], llm.TokenUsage()),
        ("finish_task", {"success": False}, "", [], llm.TokenUsage()),
    ]

    with patch("backend.app.core.orchestrator.get_snapshot", new_callable=AsyncMock) as mock_snapshot, \
         patch("backend.app.core.orchestrator.goto", new_callable=AsyncMock) as mock_goto, \
         patch("backend.app.core.orchestrator.wait_stable", new_callable=AsyncMock), \
         patch("backend.app.core.llm.classify_intent", new_callable=AsyncMock, return_value="qa_summary"), \
         patch("backend.app.core.llm.summarize_page", new_callable=AsyncMock, return_value="สรุป"), \
         patch("backend.app.core.orchestrator.llm.next_action", AsyncMock(side_effect=next_action_calls)), \
         patch("backend.app.core.orchestrator.execute", new_callable=AsyncMock) as mock_execute:

        mock_snapshot.return_value = (elements, "หน้า login")
        mock_goto.return_value = MagicMock(success=True)

        result = await orchestrator.run_task(
            url="http://example.com",
            goal="ชื่อผู้ใช้คืออะไร",
            page=mock_page,
            provider="anthropic",
        )

    mock_execute.assert_not_awaited()
    assert result["status"] == "chat_reply"
    assert result["message"] == "สรุป"


@pytest.mark.asyncio
async def test_orchestrator_generate_plan_qa_intent():
    """ทดสอบว่า generate_plan ส่งคืน is_qa=True เมื่อได้รับ Intent เป็น qa_summary"""
    orchestrator = Orchestrator()
    with patch("backend.app.core.llm.classify_intent", new_callable=AsyncMock) as mock_classify:
        mock_classify.return_value = "qa_summary"
        plan_text, is_qa = await orchestrator.generate_plan("http://example.com", "สรุปหน้านี้ให้หน่อย", provider="gemini")
        assert is_qa is True
        assert plan_text == ""


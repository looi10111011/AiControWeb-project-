import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.app.core import llm
from backend.app.core.actions import ActionResult
from backend.app.core.orchestrator import Orchestrator


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
        assert "โปรดอ่านเนื้อหาเว็บต่อไปนี้แล้วตอบคำถามของผู้ใช้ให้กระชับ เข้าใจง่าย และใช้ภาษาไทยที่เป็นกันเอง" in prompt_sent
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


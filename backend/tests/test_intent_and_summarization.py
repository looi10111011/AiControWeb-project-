import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.app.core import llm
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
    """ทดสอบว่า Orchestrator รัน task ที่มี Intent เป็น qa_summary แล้วคืน status: chat_reply ทันที"""
    orchestrator = Orchestrator()
    mock_page = AsyncMock()
    mock_page.url = "http://example.com"

    with patch("backend.app.core.orchestrator.get_snapshot", new_callable=AsyncMock) as mock_snapshot, \
         patch("backend.app.core.orchestrator.goto", new_callable=AsyncMock) as mock_goto, \
         patch("backend.app.core.orchestrator.wait_stable", new_callable=AsyncMock), \
         patch("backend.app.core.llm.classify_intent", new_callable=AsyncMock) as mock_classify, \
         patch("backend.app.core.llm.summarize_page", new_callable=AsyncMock) as mock_summarize:

        mock_snapshot.return_value = ([], "ตัวอย่างเนื้อหาเว็บไซต์เกี่ยวกับข่าว IT")
        mock_goto.return_value = MagicMock(success=True)
        mock_classify.return_value = "qa_summary"
        mock_summarize.return_value = "สรุปข่าว IT ประจำวัน: มีการเปิดตัวชิปใหม่"

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


@pytest.mark.asyncio
async def test_orchestrator_generate_plan_qa_intent():
    """ทดสอบว่า generate_plan ส่งคืน is_qa=True เมื่อได้รับ Intent เป็น qa_summary"""
    orchestrator = Orchestrator()
    with patch("backend.app.core.llm.classify_intent", new_callable=AsyncMock) as mock_classify:
        mock_classify.return_value = "qa_summary"
        plan_text, is_qa = await orchestrator.generate_plan("http://example.com", "สรุปหน้านี้ให้หน่อย", provider="gemini")
        assert is_qa is True
        assert plan_text == ""


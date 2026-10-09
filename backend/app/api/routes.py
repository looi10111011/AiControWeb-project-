"""HTTP API ที่ขับ Orchestrator ผ่าน BrowserPool/SessionRegistry/user browser

W10[A]: submit task แบบ async (202 + task_id -> poll GET /tasks/{id}) เพราะ run_task() ใช้เวลานาน
W10[B]: GET /tasks/{id}/stream (SSE) โชว์ step สด + POST /tasks/{id}/respond ให้ human ตอบ
Approve/Deny/Confirm plan จริงผ่าน REST (ผูกกับ TaskManager.push_event/request_approval/resolve_approval)
"""

import asyncio
import base64
import json
import secrets
import time
import warnings
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from playwright.async_api import async_playwright
from slowapi import Limiter
from slowapi.util import get_remote_address

from backend.app.rag.ingestion import load_manual_bytes
from backend.app.api.schemas import (
    CreateTaskRequest,
    CredentialsStatusResponse,
    ExecutePlanRequest,
    GeneratePlanRequest,
    GeneratePlanResponse,
    LearnCreatedResponse,
    LearnCredentialsRequest,
    LearnSiteRequest,
    OpenAIAuthStatusResponse,
    OpenAILoginStartResponse,
    OpenAILoginStatusResponse,
    PoolStatusResponse,
    RelearnPageRequest,
    RelearnPageResponse,
    RespondRequest,
    SaveCredentialsRequest,
    SessionStatusResponse,
    SiteManualStatusResponse,
    TaskCreatedResponse,
    TaskStatusResponse,
)
from backend.app.api.task_manager import TaskManager
from backend.app.config import settings
from backend.app.core import llm, openai_oauth, plan_memory, procedural_memory
from backend.app.core.goal_intent import detect_goal_language
from backend.app.core.orchestrator import Orchestrator
from backend.app.core.perception import get_snapshot
from backend.app.core.session_registry import SessionOwnershipError
from backend.app.core.embedded_page import EmbeddedPage, PageExchange, origin, run_embedded_task
from backend.app.permission.rules import extract_domain, install_ssrf_guard, normalize_domain
from backend.app.site_learning import crawl_site, describe_page, extract_page
from backend.app.site_learning.learn_manager import LearnManager
from backend.app.site_learning.storage import (
    build_learned_page_flow_text,
    build_strict_manual_context,
    credentials_exist,
    delete_credentials,
    find_matching_page,
    load_credentials,
    load_knowledge_text,
    load_manual,
    manual_exists,
    save_credentials,
    save_manual,
    update_single_page,
)


# Security 1.5: rate limit per-IP เฉพาะ endpoint ที่เปลือง resource (ผูก @limiter.limit() ทีละ
# endpoint ไม่ใช่ระดับ router — ไม่จำกัด SSE/polling) app.state.limiter ผูกใน main.py
# config_filename="": กัน slowapi อ่าน .env (มีภาษาไทย UTF-8) ด้วย cp1252 ของ Windows แล้วพังตอน
# import — "" (ไม่ใช่ None) ทำให้แค่ warn "Config file '' not found." ซึ่งกดทิ้งไว้
with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    limiter = Limiter(key_func=get_remote_address, config_filename="")


# Security (SEC-5 follow-up): EventSource ตั้ง header เองไม่ได้ เดิมรับ raw api_key ใน query
# (หลุดผ่าน access log/proxy/history ง่าย) — เปลี่ยนเป็น ticket อายุสั้นที่ออกผ่าน
# POST /auth/stream-ticket (ต้องใช้ X-API-Key จริง) หลุดไปก็ใช้ได้แค่ไม่กี่วินาที
_STREAM_TICKET_TTL_SECONDS = 60.0
_stream_tickets: dict[str, float] = {}  # ticket -> expires_at (unix time)

_SSE_HEADERS = {"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}


def _issue_stream_ticket() -> str:
    ticket = secrets.token_urlsafe(32)
    _stream_tickets[ticket] = time.time() + _STREAM_TICKET_TTL_SECONDS
    return ticket


def _stream_ticket_is_valid(ticket: str) -> bool:
    # กวาด ticket หมดอายุตอนเช็คเลย ไม่ต้องมี background job (จำนวนถูก bound ด้วย TTL สั้นอยู่แล้ว)
    now = time.time()
    for expired in [t for t, exp in _stream_tickets.items() if exp < now]:
        _stream_tickets.pop(expired, None)
    expires_at = _stream_tickets.get(ticket)
    return expires_at is not None and expires_at >= now


async def verify_api_key(
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    ticket: Optional[str] = Query(default=None),
) -> None:
    """Security 1.1: router-level dependency ของทุก route ใน router นี้

    settings.api_key ไม่ตั้ง = auth ปิด แต่เฉพาะ loopback เท่านั้น (host อื่น -> 503)
    รับ header "X-API-Key" หรือ "?ticket=" (สำหรับ SSE — ดู _issue_stream_ticket())"""
    if not settings.api_key:
        # A local-only console may deliberately run without auth, but accepting
        # unauthenticated requests once it is exposed on the network is unsafe.
        if settings.api_host not in {"127.0.0.1", "::1", "localhost"}:
            raise HTTPException(status_code=503, detail="API_KEY is required for a network-exposed server")
        return
    if x_api_key == settings.api_key:
        return
    if ticket and _stream_ticket_is_valid(ticket):
        return
    raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key")


router = APIRouter(dependencies=[Depends(verify_api_key)])


@router.post("/auth/stream-ticket")
async def create_stream_ticket() -> dict:
    """Security (SEC-5 follow-up): ออก ticket 60 วินาทีแลกกับ X-API-Key — ใช้ซ้ำได้ภายใน TTL
    (ไม่ single-use) เพราะ EventSource auto-reconnect เองด้วย ticket เดิม"""
    ticket = _issue_stream_ticket()
    return {"ticket": ticket, "expires_in": _STREAM_TICKET_TTL_SECONDS}


def _make_ask_user_func(task_manager: TaskManager, task_id: str, auto_approve: bool):
    """ask_user_func ที่ orchestrator ใช้ทั้ง confirm_plan และ permission-gated action
    (cmd["type"] บอกเองว่าเป็นแบบไหน) -> TaskManager.request_approval -> SSE -> POST .../respond

    auto_approve=True: อนุมัติเองทันทีทุกคำขอ (batch/CI ไม่มีคนเฝ้า)
    W10[E]: รอ human ไม่เกิน settings.approval_timeout_seconds เสมอ — ไม่งั้น user ปิดแท็บกลางคัน
    แล้ว task ยึด browser จาก pool ค้างตลอดไป"""

    async def ask_user_func(cmd: dict) -> bool:
        if auto_approve:
            await task_manager.push_event(task_id, {"kind": "auto_approved", "cmd": cmd})
            return True
        return await task_manager.request_approval(
            task_id, cmd, timeout=settings.approval_timeout_seconds
        )

    return ask_user_func


def _chat_backend(provider: Optional[str]):
    """(client, model, resolved_provider) สำหรับเส้นทาง chat-shaped ที่ไม่เปิด browser —
    เรียกเฉพาะใน branch ที่ต้องใช้จริง (ห้ามเรียก eager: เทสต์ patch Orchestrator ทั้ง class)"""
    resolved_provider = provider or settings.llm_provider
    client, model, _, _, _ = Orchestrator._llm_backend(resolved_provider)
    return client, model, resolved_provider


# W21: ข้อความ fallback ตายตัวตามสเปค (Task/W21.txt Task5) ให้ /context โชว์เมื่อไม่มี manual/ไม่มีหน้าไหน match
_NO_LEARNED_MANUAL_TEXT = "No pre-learned manual found. Executing dynamic exploration."


def _resolve_site_manual_context(domain: str, goal: str) -> str:
    """W21: จุดตัดสินใจเดียวของ generate_plan และ _run_with_resolved_browser — มีหน้าใน manual
    ที่ match goal -> build_strict_manual_context() (ขึ้นต้น "[PRE_LEARNED_MANUAL]" ที่
    SYSTEM_PROMPT ใช้บังคับ planner) ไม่งั้น fallback load_knowledge_text() เงียบๆ ไม่ throw"""
    if manual_exists(domain):
        manual = load_manual(domain)
        matched_page = find_matching_page(manual, goal) if manual else None
        if matched_page is not None:
            return build_strict_manual_context(matched_page)
    return load_knowledge_text(domain)


def _chat_shaped_result(message: str) -> dict:
    """dict รูปแบบเดียวกับ Orchestrator.run_task() (steps=0) ให้ TaskManager/SSE/frontend
    ใช้ได้โดยไม่ต้องรู้ว่าไม่เคยเปิด browser"""
    return {
        "success": True,
        "steps": 0,
        "message": message,
        "history": [],
        "tokens": {"input": 0, "output": 0, "cache_read": 0, "cache_creation": 0},
        "plan": "",
        "final_page_state": "",
        "persona_message": "",
        "persona_status": "COMPLETED",
        "completion_verification": "OK",
    }


async def _context_inspection_result(req, on_event, client, model: str, resolved_provider: str) -> dict:
    """W20 (MODULE 0): goal มี "/context" — อธิบายความเข้าใจ/แผนโดยไม่ลงมือ ไม่แตะ browser/pool
    W21: แปะ Learned Page Flow ของหน้าที่ match ถ้ามี ไม่งั้น _NO_LEARNED_MANUAL_TEXT"""
    real_goal = llm.strip_context_inspection_command(req.goal)
    domain = extract_domain(req.url) if req.url else ""
    learned_flow_text = _NO_LEARNED_MANUAL_TEXT
    if domain and manual_exists(domain):
        manual = load_manual(domain)
        matched_page = find_matching_page(manual, real_goal) if manual else None
        if matched_page is not None:
            learned_flow_text = build_learned_page_flow_text(matched_page)
    reply = await llm.context_inspection_reply(
        client, model, real_goal, resolved_provider, learned_flow_text=learned_flow_text,
        # W_context_knows_the_goal: URL เป้าหมายอยู่ใน request แล้ว ไม่ต้องให้โมเดลเดาหรือตอบ "Not specified"
        target_url=req.url or "",
    )
    await on_event({"kind": "chat_reply", "message": reply})
    return _chat_shaped_result(reply)


async def _general_chat_result(req, on_event, client, model: str, resolved_provider: str) -> dict:
    """W19-6 (MODULE 1): ตอบคำถามทั่วไป/ทักทาย/วันเวลา/คำนวณ (ดู llm.is_general_chat_query())
    โดยไม่แตะ browser/session/pool — คืน _chat_shaped_result()"""
    # ใช้ private helper ของ llm.py ตรงๆ กันคำถามวันเวลาตอบจาก training data
    chat_reply = await llm.chat_response(
        client, model, req.goal, resolved_provider, current_time_text=llm._current_bangkok_time_text(),
    )
    await on_event({"kind": "chat_reply", "message": chat_reply})
    return _chat_shaped_result(chat_reply)


_IMAGE_FILE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}


async def _file_query_result(
    req, on_event, client, model: str, resolved_provider: str, file_chat_memory: dict,
) -> dict:
    """pdf/xlsx: ตอบจากไฟล์ที่แนบมา (req.attached_file_*) ไม่แตะ browser/session/pool

    รูปภาพ (_IMAGE_FILE_EXTENSIONS) -> llm.answer_image_query() แบบ multimodal; เอกสาร ->
    load_manual_bytes() + answer_file_query() และถ้ามี session_id จะจำ text ไว้ใน
    file_chat_memory (app.state.file_chat_memory) ให้เทิร์นถัดไปถามต่อได้
    ห้าม throw — base64/extract พังต้องคืนข้อความ error ที่ user อ่านได้ ไม่ใช่ 500"""
    try:
        content = base64.b64decode(req.attached_file_content_base64)
    except Exception as e:
        print(f"⚠️ _file_query_result base64 decode error: {e}", flush=True)
        error_message = (
            f'ขออภัยครับ ไม่สามารถอ่านไฟล์ "{req.attached_file_name}" ได้ '
            "ลองแนบไฟล์ใหม่อีกครั้งนะครับ"
        )
        await on_event({"kind": "chat_reply", "message": error_message})
        return _chat_shaped_result(error_message)

    if Path(req.attached_file_name or "").suffix.lower() in _IMAGE_FILE_EXTENSIONS:
        reply = await llm.answer_image_query(
            client, model, req.goal, content, req.attached_file_name, resolved_provider,
        )
        await on_event({"kind": "chat_reply", "message": reply})
        return _chat_shaped_result(reply)

    _extract_started_at = time.monotonic()
    try:
        file_text = load_manual_bytes(content, req.attached_file_name)
    except Exception as e:
        print(f"⚠️ _file_query_result extract error: {e}", flush=True)
        error_message = (
            f'ขออภัยครับ ไม่สามารถอ่านไฟล์ "{req.attached_file_name}" ได้ '
            "(รองรับ .txt/.pdf/.docx/.xlsx/.csv และรูปภาพ) ลองแนบไฟล์ใหม่อีกครั้งนะครับ"
        )
        await on_event({"kind": "chat_reply", "message": error_message})
        return _chat_shaped_result(error_message)

    _extract_seconds = time.monotonic() - _extract_started_at
    if req.session_id:
        file_chat_memory[req.session_id] = {
            "filename": req.attached_file_name, "text": file_text,
            # W_file_followup_with_sticky_url: "เทิร์นล่าสุดคือเทิร์นไฟล์" — ล้างเป็น False เมื่อไปรันงานเบราว์เซอร์
            "is_latest_turn": True,
        }

    # W_file_answer_timing: user รายงานว่า openai อ่านไฟล์ช้า (21.8s รวม) — แยกเวลา extract vs llm
    # ให้เห็นใน console ก่อนแก้ ไม่เดา
    _answer_started_at = time.monotonic()
    reply = await llm.answer_file_query(
        client, model, req.goal, file_text, req.attached_file_name, resolved_provider,
    )
    print(
        f"[file-query] {req.attached_file_name}: extract {_extract_seconds:.1f}s "
        f"({len(file_text):,} chars) + llm {time.monotonic() - _answer_started_at:.1f}s "
        f"({resolved_provider})",
        flush=True,
    )
    await on_event({"kind": "chat_reply", "message": reply})
    return _chat_shaped_result(reply)


async def _file_chat_memory_reply(
    req, on_event, client, model: str, resolved_provider: str, remembered: dict,
) -> dict:
    """pdf/xlsx: เทิร์นต่อยอดที่ไม่แนบไฟล์ใหม่ ตอบจาก remembered["text"] ที่ extract ไว้แล้ว
    (บั๊กจริง: เดิมไม่เก็บ text เทิร์น 2 เลยไปเปิด browser ด้วย url ว่างแล้วค้าง)"""
    reply = await llm.answer_file_query(
        client, model, req.goal, remembered["text"], remembered["filename"], resolved_provider,
    )
    await on_event({"kind": "chat_reply", "message": reply})
    return _chat_shaped_result(reply)


def _is_file_followup(req, remembered_file: Optional[dict]) -> bool:
    """เทิร์นนี้ควรตอบจากไฟล์ที่จำไว้ (ใช้ร่วมกันใน generate_plan และ _run_with_resolved_browser)

    W_file_followup_with_sticky_url: เดิมบังคับ req.url ว่าง แต่ช่อง URL บนหน้าจอค้างค่าจากงานก่อน
    คำถามต่อยอด ("สรุปเป็นตาราง") จึงหลุดเข้า agent loop ตอบ "มีทั้งหมด 0 รายการ" — ยอมให้มี url ได้ถ้า
    (ก) เทิร์นล่าสุดเป็นเทิร์นไฟล์ และ (ข) ถ้อยคำอ้างข้อมูลที่เพิ่งได้ (llm.is_file_followup_request)"""
    return bool(remembered_file) and not llm.goal_mentions_web_action(req.goal) and (
        not req.url
        or (remembered_file.get("is_latest_turn") and llm.is_file_followup_request(req.goal))
    )


_READ_PAGE_DATA_OK_PREFIX = "[OK] read_page_data -> "


async def _update_extracted_memory(session, result: dict, provider: Optional[str]) -> None:
    """W19-6 (MODULE 2, SESSION_LIST): หลัง run_task() (สำเร็จหรือไม่ก็ตาม) เอา read_page_data
    สำเร็จ "ตัวล่าสุด" มาจัดโครงสร้างด้วย llm.extract_structured_items() แล้วทับ
    session.extracted_memory — ไม่เจอ/ได้ [] ปล่อยของเดิมไว้ (เช่น IN_PAGE_ACTION ที่ไม่ได้อ่านใหม่)
    ห้าม throw — เป็น enhancement เหมือน plan_memory.py/long_term_memory.py"""
    try:
        history = result.get("history") or []
        latest_read_content = ""
        for step in history:
            step_result = step.get("result", "")
            if isinstance(step_result, str) and step_result.startswith(_READ_PAGE_DATA_OK_PREFIX):
                latest_read_content = step_result[len(_READ_PAGE_DATA_OK_PREFIX):]
        if not latest_read_content:
            return
        client, model, resolved_provider = _chat_backend(provider)
        items = await llm.extract_structured_items(
            client, model, latest_read_content, "", resolved_provider,
        )
        if items:
            session.extracted_memory = items
    except Exception as e:
        print(f"⚠️ _update_extracted_memory error: {e}", flush=True)


async def _run_with_resolved_browser(
    req, orchestrator: Orchestrator, ask_user_func, on_event, pool, session_registry,
    extra_run_task_kwargs: dict, file_chat_memory: dict,
) -> dict:
    """W13: ตรรกะร่วม "เอา page มาจากไหน" ของ POST /tasks และ POST /api/execute_plan
    (req duck-typed: CreateTaskRequest/ExecutePlanRequest; ต่างกันแค่ extra_run_task_kwargs)

    Short-circuit 4 ทางก่อนแตะ browser ใดๆ (คืน chat-shaped result, steps=0) ตามลำดับ:
    (1) "/context" (W20) (2) ไฟล์แนบ (3) llm.is_general_chat_query() (W19-6)
    (4) ต่อยอดจากไฟล์ที่จำไว้ (_is_file_followup)
    ทั้งหมดเป็น keyword matching ล้วน ห้ามเพิ่ม LLM fallback — เคยลองแล้ว Orchestrator._llm_backend()
    ถูกเรียกกับ task ปกติ ทำเทสต์ที่ patch Orchestrator ทั้ง class พัง 26 เทสต์ (false negative
    ปลอดภัยกว่า false positive — ดู is_general_chat_query() ใน llm.py)
    ถ้ามี _will_use_browser() ที่ตัดสินใจเรื่องเดียวกัน ต้องแก้ลำดับ/เงื่อนไขให้ตรงกันทั้งสองที่เสมอ

    จากนั้นลำดับ: session_id (ครอบคลุม 3 โหมดผ่าน session_registry) -> use_user_browser ->
    headless=False (launch เอง + keep_browser_open=True เสมอ) -> ยืมจาก pool"""
    # W20 (MODULE 0): "/context" มาก่อนทุก check — debug mode ห้ามลงมือจริงไม่ว่ากรณีใด
    if llm.is_context_inspection_command(req.goal):
        client, model, resolved_provider = _chat_backend(req.provider)
        return await _context_inspection_result(req, on_event, client, model, resolved_provider)

    # pdf/xlsx: ไฟล์แนบเป็นสัญญาณแรงกว่า keyword matching เสมอ จึงเช็คก่อน general-chat
    if req.attached_file_content_base64:
        client, model, resolved_provider = _chat_backend(req.provider)
        return await _file_query_result(req, on_event, client, model, resolved_provider, file_chat_memory)

    if llm.is_general_chat_query(req.goal):
        client, model, resolved_provider = _chat_backend(req.provider)
        return await _general_chat_result(req, on_event, client, model, resolved_provider)

    remembered_file = file_chat_memory.get(req.session_id) if req.session_id else None
    if _is_file_followup(req, remembered_file):
        client, model, resolved_provider = _chat_backend(req.provider)
        return await _file_chat_memory_reply(req, on_event, client, model, resolved_provider, remembered_file)

    # W_file_followup_with_sticky_url: ถึงตรงนี้ = ใช้เบราว์เซอร์จริง ไฟล์ที่จำไว้ไม่ใช่ "สิ่งที่เพิ่งคุย"
    # แล้ว กันคำสั่งงานเว็บถัดไปถูกตอบจากไฟล์เก่า
    if remembered_file is not None:
        remembered_file["is_latest_turn"] = False

    wants_visible_browser = req.headless is False
    # W14: โหลดคู่มือเว็บที่ crawl ไว้ครั้งเดียว ส่งเข้า run_task() ทุก branch (ว่างถ้ายังไม่เคยเรียนรู้)
    site_manual_context = _resolve_site_manual_context(extract_domain(req.url), req.goal)

    # W12: session_id -> page ตัวเดิมข้ามหลาย request (ดู core/session_registry.py) ส่ง page= ให้
    # orchestrator ข้าม acquisition/teardown (managed_externally=True)
    if req.session_id:
        session = await session_registry.get_or_create(
            req.session_id,
            use_user_browser=req.use_user_browser,
            target_tab_id=getattr(req, "target_tab_id", None),
            headless=req.headless,
            target_url=req.url,
            pool=pool,
            tab_reuse_policy=req.tab_reuse_policy,
            ask_user_func=ask_user_func,
            owner_token=req.session_owner_token,
            require_owner_token=True,
        )

        # W19-6 (MODULE 3, multi-turn strategy): เรียก route_multi_turn_strategy() เฉพาะเมื่อ session
        # มี extracted_memory จริง — task ครั้งแรก/ทั่วไปไม่เสีย LLM call (และ _llm_backend()) เพิ่ม
        effective_goal = req.goal
        if session.extracted_memory:
            client, model, resolved_provider = _chat_backend(req.provider)
            memory_json = json.dumps(session.extracted_memory, ensure_ascii=False)
            decision = await llm.route_multi_turn_strategy(
                client, model, req.goal, req.goal, extract_domain(session.page.url),
                session.page.url, memory_json, "", resolved_provider,
            )
            if decision["chosen_strategy"] == "REPLY_FROM_MEMORY":
                reply = await llm.chat_response(
                    client, model,
                    f"{req.goal}\n\nข้อมูลที่เคยดึงไว้จากเทิร์นก่อนหน้า (เรียงตามลำดับที่แสดงบนหน้าจอจริง):\n{memory_json}",
                    resolved_provider,
                )
                await on_event({"kind": "chat_reply", "message": reply})
                return _chat_shaped_result(reply)
            if decision["chosen_strategy"] == "IN_PAGE_ACTION":
                # ผูก ordinal ("เพลงที่ 3") กับรายการจริงใน buffer แทนให้ loop ตีความจาก DOM ที่อาจเรียงต่างไปแล้ว
                target_entity = decision["context_analysis"].get("target_entity_from_memory", "")
                if target_entity:
                    effective_goal = (
                        f"{req.goal}\n\n[อ้างอิงจากรายการที่เคยแสดงไว้ก่อนหน้า]: {target_entity}"
                    )
            elif decision["chosen_strategy"] == "NEW_NAVIGATION":
                # เปลี่ยนหัวข้อจริง — ล้าง buffer กันเทิร์นถัดไปอ้างรายการเก่าผิดๆ
                session.extracted_memory = []

        result = await orchestrator.run_task(
            url=req.url,
            goal=effective_goal,
            max_steps=req.max_steps,
            headless=req.headless,
            verbose=False,
            provider=req.provider,
            ask_user_func=ask_user_func,
            on_event=on_event,
            page=session.page,
            site_manual_context=site_manual_context,
            session_id=req.session_id,
            **extra_run_task_kwargs,
        )
        await _update_extracted_memory(session, result, req.provider)
        return result

    # W12: user browser (CDP) ไม่ใช่ของ pool — เช็คก่อน wants_visible_browser เพราะ headless ไม่มีความหมายในโหมดนี้
    if req.use_user_browser:
        return await orchestrator.run_task(
            url=req.url,
            goal=req.goal,
            max_steps=req.max_steps,
            verbose=False,
            provider=req.provider,
            ask_user_func=ask_user_func,
            on_event=on_event,
            connect_to_user_browser=True,
            tab_reuse_policy=req.tab_reuse_policy,
            site_manual_context=site_manual_context,
            session_id=req.session_id,
            **extra_run_task_kwargs,
        )
    if wants_visible_browser:
        return await orchestrator.run_task(
            url=req.url,
            goal=req.goal,
            max_steps=req.max_steps,
            headless=False,
            verbose=False,
            provider=req.provider,
            ask_user_func=ask_user_func,
            on_event=on_event,
            keep_browser_open=True,
            site_manual_context=site_manual_context,
            session_id=req.session_id,
            **extra_run_task_kwargs,
        )
    async with pool.acquire() as browser:
        return await orchestrator.run_task(
            url=req.url,
            goal=req.goal,
            max_steps=req.max_steps,
            headless=req.headless,
            verbose=False,
            provider=req.provider,
            ask_user_func=ask_user_func,
            browser=browser,
            on_event=on_event,
            site_manual_context=site_manual_context,
            session_id=req.session_id,
            **extra_run_task_kwargs,
        )


# W_retry_value_has_no_home (บั๊กจริงบน Test Console 2026-09-04): TASK_FAILED_USER_INPUT_ERROR บอก user
# ให้ตอบค่าใหม่ แต่ไม่มีโค้ดรับค่า — ถูกส่งเป็น goal ดิบ agent ตอบ "the user's goal is only 'Abcd1234'"
# แก้: แปลง goal ที่ endpoint ก่อนใคร ให้ _run_with_resolved_browser()/_will_use_browser() (ต้องตัดสินใจ
# ตรงกันเสมอ) เห็น goal เดียวกัน — keyword ล้วน ไม่เรียก LLM
_MAX_BARE_VALUE_CHARS = 64


def _goal_is_a_bare_replacement_value(goal: str) -> bool:
    """ข้อความที่ user ตอบกลับเป็น "ค่า" เฉยๆ ไม่ใช่คำสั่งใหม่

    ต้องเป็น token เดียวไม่มีช่องว่าง: เกณฑ์ "≤4 คำ" ใช้กับไทยไม่ได้ (ไม่เว้นวรรค — "เปิดเว็ปแล้ว
    เปลี่ยนรหัสผ่านเป็น 12345678" นับได้ 2 คำ; บทเรียนเดียวกับ W_thai_keyword_space)
    เดาผิดทางนี้แค่ไม่แปลง goal = พฤติกรรมเดิม"""
    text = (goal or "").strip()
    if not text or len(text) > _MAX_BARE_VALUE_CHARS or len(text.split()) != 1:
        return False
    # token เดียวที่มีอักษรไทยมักเป็นคำสั่ง ("ลบuserrole=ess") — ค่าที่ตอบช่องนี้เป็น latin/ตัวเลขล้วนเสมอ
    if detect_goal_language(text)["script"] not in ("latin", "other"):
        return False
    return not llm.goal_mentions_web_action(text)


def _replacement_value_goal(value: str, labels: list) -> str:
    """ต้องระบุช่องแบบไม่กำกวม — รันสด 2026-09-04: เวอร์ชันแรกบอกช่อง "Password" โมเดลไปกรอก
    "Current Password" (substring กัน) ทับรหัสปัจจุบัน ส่วนช่องที่ผิดจริงยังค้างค่าเดิม"""
    fields = ", ".join(f'"{label}"' for label in labels)
    return (
        f'กรอกค่า "{value.strip()}" ลงในช่องที่มี label ตรงตัวว่า {fields} '
        "ให้ตรงกันทุกช่อง (แทนที่ค่าเดิมที่ระบบปฏิเสธ) "
        "ห้ามแก้ช่องอื่นเด็ดขาด โดยเฉพาะช่องรหัสผ่านปัจจุบัน (Current Password) "
        "ซึ่งกรอกถูกอยู่แล้ว แล้วจึงบันทึกฟอร์ม"
    )


def _apply_pending_replacement_value(req, pending_value_request: dict) -> None:
    """แปลง goal "ค่าเปล่า" เป็นคำสั่งระบุช่อง ถ้าเทิร์นก่อนขอค่าใหม่ไว้

    ต้องเรียกจากทุก endpoint ที่รับ goal (create_task, generate_plan, execute_plan): Test Console
    ส่งค่าผ่าน /api/generate_plan -> /api/execute_plan (index.html::submitCorrectionValue) ไม่ใช่
    /tasks — เวอร์ชันแรกต่อไว้แค่ create_task จึงไม่ทำงานบนหน้าเว็บจริง"""
    labels = (pending_value_request.get(req.session_id) or {}).get("labels") or []
    if labels and _goal_is_a_bare_replacement_value(req.goal):
        req.goal = _replacement_value_goal(req.goal, labels)


def _remember_pending_value_request(req, result: dict, pending_value_request: dict) -> None:
    """W_retry_value_has_no_home: จำว่ารอค่าใหม่อยู่ (หรือเลิกรอ ถ้ารอบนี้ไม่ได้จบแบบนั้น)"""
    if req.session_id:
        labels = (result or {}).get("retry_value_field_labels") or []
        if labels:
            pending_value_request[req.session_id] = {"labels": labels}
        else:
            pending_value_request.pop(req.session_id, None)


@router.post("/tasks", response_model=TaskCreatedResponse, status_code=202)
@limiter.limit("10/minute")
async def create_task(req: CreateTaskRequest, request: Request) -> TaskCreatedResponse:
    if req.embedded_page is not None:
        try:
            if origin(req.url) != origin(req.embedded_page.url):
                raise ValueError("Snapshot must belong to the requested origin")
            page = EmbeddedPage(req.embedded_page)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        manager = request.app.state.task_manager
        task_id = manager.new_task_id()
        if not hasattr(request.app.state, "embedded_pages"):
            request.app.state.embedded_pages = {}
        request.app.state.embedded_pages[task_id] = page

        async def embedded_event(event):
            await manager.push_event(task_id, event)

        async def run_page():
            try:
                return await run_embedded_task(
                    page, req.goal, req.provider, req.max_steps,
                    _make_ask_user_func(manager, task_id, req.auto_approve), embedded_event,
                )
            finally:
                page.close()
                request.app.state.embedded_pages.pop(task_id, None)

        record = manager.submit(task_id, req.url, req.goal, req.provider, run_page())
        return TaskCreatedResponse(task_id=record.task_id, status=record.status, embedded_token=page.token)
    pool = request.app.state.browser_pool
    session_registry = request.app.state.session_registry
    file_chat_memory = request.app.state.file_chat_memory
    pending_value_request: dict = request.app.state.pending_value_request
    task_manager: TaskManager = request.app.state.task_manager
    _apply_pending_replacement_value(req, pending_value_request)
    orchestrator = Orchestrator()
    task_id = task_manager.new_task_id()
    ask_user_func = _make_ask_user_func(task_manager, task_id, req.auto_approve)

    async def _on_event(event: dict) -> None:
        await task_manager.push_event(task_id, event)

    async def _run() -> dict:
        result = await _run_with_resolved_browser(
            req, orchestrator, ask_user_func, _on_event, pool, session_registry,
            extra_run_task_kwargs={"confirm_plan": req.confirm_plan}, file_chat_memory=file_chat_memory,
        )
        _remember_pending_value_request(req, result, pending_value_request)
        return result

    resolved_headless = settings.browser_headless if req.headless is None else req.headless
    record = task_manager.submit(
        task_id, req.url, req.goal, req.provider, _run(), headless=resolved_headless,
        attached_file_name=req.attached_file_name,
    )
    return TaskCreatedResponse(task_id=record.task_id, status=record.status)


@router.get("/page-bridge")
async def page_bridge_capabilities():
    return {"version": 1, "execution": "in_page"}


@router.post("/tasks/{task_id}/page")
async def exchange_embedded_page(task_id: str, body: PageExchange, request: Request):
    page = getattr(request.app.state, "embedded_pages", {}).get(task_id)
    if page is None:
        raise HTTPException(status_code=410, detail="Page task ended or backend restarted")
    try:
        command = page.exchange(body)
    except (PermissionError, ValueError) as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    return {"command": command}


@router.post("/api/generate_plan", response_model=GeneratePlanResponse)
@limiter.limit("20/minute")
async def generate_plan(req: GeneratePlanRequest, request: Request) -> GeneratePlanResponse:
    """W13: เฟสวางแผน — sync LLM call เดียว ไม่เปิด browser ใหม่ (session_registry.get() เท่านั้น
    ไม่ get_or_create) ถ้ามี session ที่ page ยังเปิดอยู่จะ perceive ช่วยให้แผน grounded

    ลำดับ: "/context" / ไฟล์แนบ / ต่อยอดจากไฟล์ -> is_qa=True ทันที (deterministic ไม่เรียก LLM
    ให้ frontend ข้ามหน้าอนุมัติแผน; เกณฑ์เดียวกับ _run_with_resolved_browser)
    -> W_procmem procedural template (ปิด default: settings.enable_procedural_memory)
    -> W20 Plan Memory (semantic match ของ approved plan, user-approved มาก่อน LLM) -> LLM ร่างใหม่

    W20 Context-Aware Implicit Execution (บั๊กจริง: "ขอเพลงเศร้าๆ" แล้ว "okเปิดให้หน่อย" ได้แผนแค่
    "เปิด youtube.com"): ส่ง previous_user_goal/previous_assistant_message เข้า generate_plan()
    เพราะ extracted_memory จับ entity จาก general-chat reply ไม่ได้"""
    if llm.is_context_inspection_command(req.goal):
        return GeneratePlanResponse(plan="", is_qa=True)

    if req.attached_file_content_base64:
        return GeneratePlanResponse(plan="", is_qa=True)

    # W_retry_value_has_no_home: แปลงก่อนร่างแผน ไม่งั้นแผนถูกร่างจากค่าเปล่าๆ
    _apply_pending_replacement_value(req, request.app.state.pending_value_request)
    file_chat_memory: dict = request.app.state.file_chat_memory
    remembered_file = file_chat_memory.get(req.session_id) if req.session_id else None
    if _is_file_followup(req, remembered_file):
        return GeneratePlanResponse(plan="", is_qa=True)

    domain = extract_domain(req.url)

    page = None
    if req.session_id:
        session_registry = request.app.state.session_registry
        try:
            session = session_registry.get(
                req.session_id, owner_token=req.session_owner_token, require_owner_token=True,
            )
        except SessionOwnershipError:
            raise HTTPException(status_code=403, detail="session_id นี้ไม่ใช่ของ owner_token ที่ส่งมา")
        if session is not None and session_registry.is_healthy(session):
            page = session.page

    if settings.enable_procedural_memory:
        candidates = procedural_memory.find_candidate_templates(domain, req.goal)
        if candidates:
            page_fingerprint = ""
            if page is not None:
                try:
                    _, page_fingerprint = await get_snapshot(page)
                except Exception:
                    pass
            planner_client, planner_model, resolved_provider = _chat_backend(req.provider)
            # W_procmem: บอก Planner ว่ามี auto-login credential ไหม — ไม่งั้นมันปฏิเสธ template ที่ไม่มี
            # step login ทั้งที่ login เกิดอัตโนมัตินอก template (เจอจริงกับ OrangeHRM)
            decision = await llm.plan_with_procedural_memory(
                planner_client, planner_model, req.goal, req.url, page_fingerprint, candidates, resolved_provider,
                has_auto_login=credentials_exist(domain),
            )
            if decision["decision"] in ("reuse", "adapt") and decision.get("template_id"):
                matched_candidate = next(
                    (c for c in candidates if c["template_id"] == decision["template_id"]), None,
                )
                if matched_candidate is not None:
                    final_steps = matched_candidate["steps"]
                    if decision["decision"] == "adapt":
                        final_steps = procedural_memory.apply_template_patch(final_steps, decision.get("patch"))
                    slot_values = decision.get("slot_values") or {}
                    return GeneratePlanResponse(
                        plan=procedural_memory.render_steps_as_plan_text(final_steps, slot_values),
                        is_qa=False,
                        source=f"procedural_{decision['decision']}",
                        template_id=decision["template_id"],
                        slot_values=slot_values,
                        steps=final_steps,
                    )

    matched = plan_memory.find_matching_plan(domain, req.goal)
    if matched is not None:
        return GeneratePlanResponse(plan=matched["plan"], is_qa=False, source="plan_memory")

    site_manual_context = _resolve_site_manual_context(domain, req.goal)
    try:
        res = await asyncio.wait_for(
            Orchestrator().generate_plan(
                req.url, req.goal, provider=req.provider, page=page, site_manual_context=site_manual_context,
                previous_user_goal=req.previous_user_goal or "",
                previous_assistant_message=req.previous_assistant_message or "",
            ),
            timeout=settings.plan_generation_timeout_seconds,
        )
        if isinstance(res, tuple):
            plan, is_qa = res
        else:
            plan, is_qa = str(res), False
    except asyncio.TimeoutError:
        # W_planhang: ไม่มี timeout นี้ request ค้างเงียบตลอดไปถ้า provider ช้า (composer ค้าง "Generating plan…")
        raise HTTPException(
            status_code=504,
            detail=f"LLM ไม่ตอบสนองภายใน {settings.plan_generation_timeout_seconds:.0f} วินาที ลองใหม่อีกครั้ง",
        )
    except Exception as e:
        # ห่อเป็น HTTPException ที่มี detail จริง แทน 500 เปล่าๆ ที่ frontend ไม่รู้สาเหตุ
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
    return GeneratePlanResponse(plan=plan, is_qa=is_qa)


@router.post("/api/execute_plan", response_model=TaskCreatedResponse, status_code=202)
async def execute_plan(req: ExecutePlanRequest, request: Request) -> TaskCreatedResponse:
    """W13: รันแผนที่อนุมัติแล้ว (อาจถูก user แก้) — เหมือน create_task() แต่ไม่มี confirm_plan gate

    W20: บันทึกเข้า Plan Memory ทุกครั้งที่มาถึงนี่ (Confirm = approved; plan_memory ไม่สร้าง version ซ้ำเอง)
    W_procmem: execution_mode == "fastpath" (+ template_id/steps, ไม่ใช่ user browser/visible window)
    -> ไม่บันทึก plan_memory (preview text ไม่ใช่แผนที่ user พิมพ์) แล้วรัน orchestrator.run_fastpath()
    โหมดที่ fast-path ไม่รองรับ fallback เป็น slow-path เงียบๆ"""
    pending_value_request: dict = request.app.state.pending_value_request
    _apply_pending_replacement_value(req, pending_value_request)
    is_fastpath = bool(
        settings.enable_procedural_memory
        and req.execution_mode == "fastpath"
        and req.template_id
        and req.steps
        and not req.use_user_browser
        and req.headless is not False
    )
    if req.plan and not is_fastpath:
        plan_memory.save_confirmed_plan(extract_domain(req.url), req.goal, req.plan)

    pool = request.app.state.browser_pool
    session_registry = request.app.state.session_registry
    file_chat_memory = request.app.state.file_chat_memory
    task_manager: TaskManager = request.app.state.task_manager
    orchestrator = Orchestrator()
    task_id = task_manager.new_task_id()
    ask_user_func = _make_ask_user_func(task_manager, task_id, req.auto_approve)

    async def _on_event(event: dict) -> None:
        await task_manager.push_event(task_id, event)

    async def _run() -> dict:
        if is_fastpath:
            if req.session_id:
                session = await session_registry.get_or_create(
                    req.session_id, use_user_browser=False, headless=req.headless,
                    target_url=req.url, pool=pool, tab_reuse_policy=req.tab_reuse_policy,
                    ask_user_func=ask_user_func, owner_token=req.session_owner_token,
                    require_owner_token=True,
                )
                return await orchestrator.run_fastpath(
                    url=req.url, goal=req.goal, template_id=req.template_id, steps=req.steps,
                    slot_values=req.slot_values or {}, max_steps=req.max_steps, provider=req.provider,
                    ask_user_func=ask_user_func, on_event=_on_event, page=session.page,
                    session_id=req.session_id,
                )
            async with pool.acquire() as browser:
                return await orchestrator.run_fastpath(
                    url=req.url, goal=req.goal, template_id=req.template_id, steps=req.steps,
                    slot_values=req.slot_values or {}, max_steps=req.max_steps, provider=req.provider,
                    ask_user_func=ask_user_func, on_event=_on_event, browser=browser,
                )
        result = await _run_with_resolved_browser(
            req, orchestrator, ask_user_func, _on_event, pool, session_registry,
            extra_run_task_kwargs={"approved_plan": req.plan}, file_chat_memory=file_chat_memory,
        )
        # เส้นทางนี้คือที่ Test Console ใช้จริงเมื่อมีแผน
        _remember_pending_value_request(req, result, pending_value_request)
        return result

    resolved_headless = settings.browser_headless if req.headless is None else req.headless
    record = task_manager.submit(
        task_id, req.url, req.goal, req.provider, _run(), headless=resolved_headless,
        attached_file_name=req.attached_file_name,
    )
    return TaskCreatedResponse(task_id=record.task_id, status=record.status)


def _task_status(record) -> TaskStatusResponse:
    return TaskStatusResponse(
        task_id=record.task_id,
        url=record.url,
        goal=record.goal,
        provider=record.provider,
        status=record.status,
        created_at=record.created_at,
        result=record.result,
        error=record.error,
        headless=record.headless,
        attached_file_name=record.attached_file_name,
    )


@router.get("/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task(task_id: str, request: Request) -> TaskStatusResponse:
    task_manager = request.app.state.task_manager
    record = task_manager.get(task_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"ไม่พบ task_id: {task_id!r}")
    return _task_status(record)


@router.get("/tasks", response_model=list[TaskStatusResponse])
async def list_tasks(request: Request) -> list[TaskStatusResponse]:
    task_manager = request.app.state.task_manager
    return [_task_status(r) for r in task_manager.list()]


async def _stream_task_events(record):
    """W10[B]/W25/W26: SSE body ของ GET /tasks/{id}/stream (ระดับโมดูลให้เทสต์เรียกตรงได้)

    task จบแล้วก่อน connect -> ส่ง task_done สังเคราะห์จาก record ทันที (ไม่มี log buffer ให้ replay)
    W26 (บั๊กจริง: approval/log ไม่ขึ้นสดต้อง F5): ทุกแท็บ subscribe task running ทั้งระบบเป็นเรื่องปกติ
    W25 "connection ล่าสุดชนะ" ทำให้แท็บอื่นแย่ง event ไป — แก้เป็น broadcast: ลง Queue ของตัวเองใน
    record.event_subscribers (ดู task_manager.py::_broadcast()) และถอนออกเสมอใน finally"""
    if record.status != "running":
        done_event = {
            "kind": "task_done", "status": record.status,
            "result": record.result, "error": record.error,
        }
        yield f"data: {json.dumps(done_event)}\n\n"
        return

    my_queue: asyncio.Queue = asyncio.Queue()
    record.event_subscribers.append(my_queue)
    try:
        # W10[B]: replay เฉพาะ approval ที่ delivered แล้ว (แท็บที่ reconnect กลางคันต้องเห็นว่ามีอะไรค้าง)
        # ที่ยังไม่เคยส่งจะไหลมาทาง queue เอง — replay ด้วยจะเห็นซ้ำสองครั้ง
        for request_id, info in list(record.pending.items()):
            if info["delivered"]:
                yield f"data: {json.dumps({'kind': 'approval_request', 'request_id': request_id, 'cmd': info['cmd']})}\n\n"
        while True:
            event = await my_queue.get()
            if event.get("kind") == "approval_request":
                info = record.pending.get(event.get("request_id"))
                if info is not None:
                    info["delivered"] = True
            yield f"data: {json.dumps(event)}\n\n"
            if event.get("kind") == "task_done":
                break
    finally:
        # W26: ถอนตัวเสมอ (task_done หรือ client ปิดกลางคัน = generator ถูก cancel) กัน Queue ร้างสะสม
        if my_queue in record.event_subscribers:
            record.event_subscribers.remove(my_queue)


@router.get("/tasks/{task_id}/stream")
async def stream_task(task_id: str, request: Request) -> StreamingResponse:
    """W10[B]: SSE ของ task — step log + approval_request + task_done (ดู _stream_task_events())"""
    task_manager: TaskManager = request.app.state.task_manager
    record = task_manager.get(task_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"ไม่พบ task_id: {task_id!r}")

    return StreamingResponse(
        _stream_task_events(record), media_type="text/event-stream", headers=_SSE_HEADERS,
    )


@router.post("/tasks/{task_id}/stop")
async def stop_task(task_id: str, request: Request) -> dict:
    """W10[C]: ปุ่ม Stop — ยกเลิก task ที่รันอยู่ (ดู TaskManager.cancel()) 409 ถ้าไม่ได้ running"""
    task_manager: TaskManager = request.app.state.task_manager
    record = task_manager.get(task_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"ไม่พบ task_id: {task_id!r}")
    # W27: cancel() เป็น async รอ task ปล่อย page/browser จริงก่อน return — กัน race กับ
    # POST /sessions/{id}/close ที่ frontend (killSession()) ยิงตามมาทันที
    ok = await task_manager.cancel(task_id)
    if not ok:
        raise HTTPException(status_code=409, detail="Task นี้ไม่ได้กำลังรันอยู่แล้ว")
    return {"status": "stopping"}


@router.post("/tasks/{task_id}/respond")
async def respond_task(task_id: str, req: RespondRequest, request: Request) -> dict:
    """W10[B]: ตอบ approval_request (permission/plan) — request_id ไม่ตรง/ตอบไปแล้ว -> 404"""
    task_manager: TaskManager = request.app.state.task_manager
    ok = task_manager.resolve_approval(
        task_id, req.request_id, req.approved,
        edited_plan=req.edited_plan, answer_text=req.answer_text,
    )
    if not ok:
        raise HTTPException(status_code=404, detail="ไม่พบ pending request นี้ (อาจหมดอายุหรือตอบไปแล้ว)")
    return {"status": "ok"}


@router.get("/pool/status", response_model=PoolStatusResponse)
async def pool_status(request: Request) -> PoolStatusResponse:
    pool = request.app.state.browser_pool
    return PoolStatusResponse(size=pool.size, available=pool.available, in_use=pool.size - pool.available)


@router.post("/sessions/{session_id}/close")
async def close_session(
    session_id: str, request: Request, session_owner_token: Optional[str] = None,
) -> dict:
    """W12: ปุ่ม "New Session" — ปิด page/context/browser ของ session (SessionRegistry.close())
    ไม่ยกเลิก task ที่รันอยู่ (frontend ต้องเช็คเอง) 404 ถ้าไม่มีอะไรให้ปิด

    Security (SEC-4 follow-up): session_owner_token (query param) ไม่ตรง -> 403
    pdf/xlsx: ล้าง file_chat_memory ของ session ด้วย — session ไฟล์ล้วนไม่อยู่ใน registry ไม่งั้นโดน 404"""
    session_registry = request.app.state.session_registry
    try:
        closed = await session_registry.close(
            session_id, owner_token=session_owner_token, require_owner_token=True,
        )
    except SessionOwnershipError:
        raise HTTPException(status_code=403, detail="session_id นี้ไม่ใช่ของ owner_token ที่ส่งมา")
    file_chat_memory: dict = request.app.state.file_chat_memory
    had_file_memory = file_chat_memory.pop(session_id, None) is not None
    if not closed and not had_file_memory:
        raise HTTPException(status_code=404, detail=f"ไม่พบ session_id: {session_id!r}")
    return {"status": "closed"}


@router.get("/sessions", response_model=list[SessionStatusResponse])
async def list_sessions(request: Request) -> list[SessionStatusResponse]:
    """debug/monitor: session ที่ถือ browser ค้างอยู่ (mode="pool" กิน pool จนกว่าจะปิด)"""
    session_registry = request.app.state.session_registry
    return [
        SessionStatusResponse(
            session_id=s.session_id, mode=s.mode,
            created_at=s.created_at, last_active_at=s.last_active_at,
        )
        for s in session_registry.list()
    ]


# W14: Website Learning (backend/app/site_learning/) — แยกจาก RAG/ChromaDB, เก็บ manual เป็น JSON บนดิสก์


@router.get("/api/site-manual/status", response_model=SiteManualStatusResponse)
async def site_manual_status(url: str) -> SiteManualStatusResponse:
    """banner "ยังไม่มีคู่มือ" บน Test Console — อ่านอย่างเดียว"""
    manual = load_manual(extract_domain(url))
    if manual is None:
        return SiteManualStatusResponse(exists=False, version=None)
    return SiteManualStatusResponse(exists=True, version=manual.version)


@router.post("/api/site-manual/learn", response_model=LearnCreatedResponse, status_code=202)
@limiter.limit("5/minute")
async def learn_site(req: LearnSiteRequest, request: Request) -> LearnCreatedResponse:
    """เริ่ม crawl req.url — 202 + learn_id ทันที (crawl ใช้เวลาเป็นนาที)

    W16: เปิด Chromium แยกแบบ headless=False โดยเจตนา (pool เป็น headless ตาม settings แต่ user
    อยากเห็น crawler เดินสด) ปิดเองเมื่อจบ ไม่ยืม/ไม่กระทบ pool"""
    learn_manager: LearnManager = request.app.state.learn_manager
    learn_id = learn_manager.new_learn_id()

    async def _on_progress(event: dict) -> None:
        await learn_manager.push_event(learn_id, event)

    # W23: เจอหน้า login ระหว่าง crawl -> ถาม user ผ่าน SSE (credentials_needed) + POST .../credentials
    # แล้วบันทึกทันที (crawl ถูก stop ทีหลังก็ไม่หาย) domain ต้องตรง extract_domain(req.url) เสมอ —
    # assert เป็นชั้นที่สองกันบันทึก credential ข้ามเว็บ ถ้า crawler เปลี่ยนพฤติกรรมในอนาคต
    target_domain = extract_domain(req.url)

    async def _on_credentials_needed(domain: str) -> Optional[dict]:
        assert domain == target_domain, (
            f"crawl job ของ {target_domain!r} เรียกขอ credential ของ {domain!r} ผิดเว็บ — "
            "ห้ามบันทึก/ใช้ credential ข้าม site_manuals เด็ดขาด"
        )
        creds = await learn_manager.request_credentials(
            learn_id, domain, timeout=settings.approval_timeout_seconds,
        )
        if creds:
            save_credentials(domain, creds["username"], creds["password"])
        return creds

    # W18: "ใช้บัญชีที่บันทึกไว้" (frontend ส่ง username/password เป็น None) -> โหลด credential ที่เก็บไว้
    resolved_username, resolved_password = req.username, req.password
    if req.use_saved_credentials and not (resolved_username and resolved_password):
        saved_creds = load_credentials(extract_domain(req.url))
        if saved_creds:
            resolved_username, resolved_password = saved_creds["username"], saved_creds["password"]

    async def _run() -> dict:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(headless=False)
            try:
                manual = await crawl_site(
                    browser,
                    req.url,
                    provider=req.provider,
                    on_progress=_on_progress,
                    username=resolved_username,
                    password=resolved_password,
                    on_credentials_needed=_on_credentials_needed,
                )
            finally:
                await browser.close()
        version = save_manual(manual)
        # W17: บันทึก credential ที่ใช้ login bootstrap จริงทับไว้ (idempotent)
        if resolved_username and resolved_password:
            save_credentials(manual.website, resolved_username, resolved_password)
        # W24: errors_count ให้ frontend โชว์จำนวนปัญหา; W26: summary โชว์ทันทีที่เรียนรู้เสร็จ
        return {
            "version": version, "pages_found": len(manual.pages),
            "errors_count": len(manual.errors), "summary": manual.summary,
        }

    record = learn_manager.submit(learn_id, req.url, _run())
    return LearnCreatedResponse(learn_id=record.learn_id, status=record.status)


@router.get("/api/site-manual/learn/{learn_id}/stream")
async def stream_learn(learn_id: str, request: Request) -> StreamingResponse:
    """SSE ของ crawl job — page_done ทีละหน้า + learn_done ปิดท้าย
    W23: + credentials_needed/credentials_timeout (ตอบผ่าน POST .../credentials)"""
    learn_manager: LearnManager = request.app.state.learn_manager
    record = learn_manager.get(learn_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"ไม่พบ learn_id: {learn_id!r}")

    async def event_gen():
        if record.status != "running":
            done_event = {
                "kind": "learn_done", "status": record.status,
                "result": record.result, "error": record.error,
            }
            yield f"data: {json.dumps(done_event)}\n\n"
            return
        while True:
            event = await record.events.get()
            yield f"data: {json.dumps(event)}\n\n"
            if event.get("kind") == "learn_done":
                break

    return StreamingResponse(event_gen(), media_type="text/event-stream", headers=_SSE_HEADERS)


@router.post("/api/site-manual/learn/{learn_id}/credentials", status_code=204)
async def respond_learn_credentials(
    learn_id: str, req: LearnCredentialsRequest, request: Request,
) -> None:
    """W23: ตอบ credentials_needed ปลดล็อก crawl_site() — ว่างทั้งคู่ = ข้าม login
    404 ถ้า request_id ไม่ตรง/หมดอายุ/ตอบไปแล้ว"""
    learn_manager: LearnManager = request.app.state.learn_manager
    ok = learn_manager.resolve_credentials(learn_id, req.request_id, req.username, req.password)
    if not ok:
        raise HTTPException(status_code=404, detail=f"ไม่พบ request_id: {req.request_id!r} (หมดอายุ/ตอบไปแล้ว)")


@router.post("/api/site-manual/{domain}/relearn-page", response_model=RelearnPageResponse)
async def relearn_page(domain: str, req: RelearnPageRequest, request: Request) -> RelearnPageResponse:
    """Selector-repair: สำรวจใหม่แค่หน้า req.url แล้ว bump version (ไม่ crawl ทั้งเว็บ) — sync ตรงๆ
    ต้องมี manual ของโดเมนนี้อยู่แล้ว ไม่งั้น 404"""
    if not manual_exists(domain):
        raise HTTPException(
            status_code=404,
            detail=f"ยังไม่มี manual ของ {domain!r} — ต้องเรียนรู้เว็บไซต์ทั้งหมดก่อน (POST /api/site-manual/learn)",
        )
    pool = request.app.state.browser_pool
    client, model, resolved_provider = _chat_backend(req.provider)

    async with pool.acquire() as browser:
        context = await browser.new_context()
        await install_ssrf_guard(context)
        page = await context.new_page()
        try:
            await page.goto(req.url, timeout=15000)
            await page.wait_for_load_state("networkidle", timeout=8000)
            page_info, _ = await extract_page(page)
            page_info.name, page_info.description = await describe_page(client, model, resolved_provider, page_info)
            page_info.menu_path = page_info.menu_path or [page_info.name]
        finally:
            await context.close()

    version = update_single_page(domain, page_info)
    if version is None:
        raise HTTPException(status_code=404, detail=f"ยังไม่มี manual ของ {domain!r}")
    return RelearnPageResponse(version=version)


@router.post("/api/site-manual/{domain}/credentials", status_code=204)
async def save_site_credentials(domain: str, req: SaveCredentialsRequest) -> None:
    """W17: บันทึก/แก้ credential โดยไม่ crawl — 204 ไม่คืนค่ากลับ (กันหลุดใน log/network tab)
    normalize_domain() เสมอ: path param ไม่ผ่าน extract_domain() ("www."/ตัวพิมพ์ใหญ่) จะได้ key
    ไม่ตรงกับที่ orchestrator.py::_maybe_auto_login ค้นหา"""
    save_credentials(normalize_domain(domain), req.username, req.password)


@router.get("/api/site-manual/{domain}/credentials/status", response_model=CredentialsStatusResponse)
async def site_credentials_status(domain: str) -> CredentialsStatusResponse:
    """คืนแค่ exists: bool — ไม่คืน username/password"""
    return CredentialsStatusResponse(exists=credentials_exist(normalize_domain(domain)))


@router.delete("/api/site-manual/{domain}/credentials", status_code=204)
async def delete_site_credentials(domain: str) -> None:
    """ลบ credential ของโดเมน — idempotent"""
    delete_credentials(normalize_domain(domain))


# W_openai_oauth: "Sign in with ChatGPT" (risk disclosure ดู core/openai_oauth.py) — อยู่หลัง
# verify_api_key ระดับ router เหมือนทุก route ไม่มี auth แยก เพราะระบบ single-tenant


@router.post("/api/auth/openai/login/start", response_model=OpenAILoginStartResponse)
async def start_openai_login() -> OpenAILoginStartResponse:
    """เปิด loopback listener แล้วคืน authorize_url — token exchange เกิดใน background
    (openai_oauth.start_login_flow()) frontend ต้อง poll .../login/status จน "linked"/"error" """
    result = await openai_oauth.start_login_flow()
    return OpenAILoginStartResponse(authorize_url=result["authorize_url"], login_id=result["login_id"])


@router.get("/api/auth/openai/login/status", response_model=OpenAILoginStatusResponse)
async def openai_login_status(login_id: str = Query(...)) -> OpenAILoginStatusResponse:
    """"pending"|"linked"|"error" — login_id ไม่รู้จัก -> 404 (ไม่ใช่ pending ค้างให้ poll ไม่รู้จบ)"""
    status = openai_oauth.get_login_status(login_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"ไม่พบ login attempt {login_id!r} (อาจหมดอายุ/server restart ไปแล้ว)")
    return OpenAILoginStatusResponse(**status)


@router.get("/api/auth/openai/status", response_model=OpenAIAuthStatusResponse)
async def openai_auth_status() -> OpenAIAuthStatusResponse:
    """สถานะ link ปัจจุบันจาก token store (email/plan_type ไม่ decrypt token)"""
    status = openai_oauth.get_link_status()
    return OpenAIAuthStatusResponse(**status)


@router.post("/api/auth/openai/logout", status_code=204)
async def openai_logout() -> None:
    """ลบ token ในเครื่องเสมอ + best-effort revoke (ไม่ throw ถ้า network ล้มเหลว)"""
    await openai_oauth.revoke_token()

"""W10[A]: API endpoints ที่ขับ Orchestrator จริงผ่าน BrowserPool แทน CLI (run.py)

Task submission เป็นแบบ async (submit -> 202 + task_id -> poll GET /tasks/{id}) ไม่ใช่
sync request-response ตรงๆ เพราะ run_task() ใช้เวลานาน (หลาย step, เรียก LLM จริงทุก
step) ดู task_manager.py สำหรับเหตุผลเต็มๆ

W10[B]: เพิ่ม GET /tasks/{id}/stream (SSE) ให้หน้าเว็บเห็นความคืบหน้าสดๆ ทีละ step
(ไม่ใช่ poll ผลลัพธ์รวมท้าย task เดียวเหมือนเดิม) + POST /tasks/{id}/respond ให้กดปุ่ม
Approve/Deny หรือ Confirm plan บนหน้าเว็บได้จริง (human-in-the-loop ผ่าน REST จริงๆ
แทนที่จะ fail-closed/auto-approve อัตโนมัติ) — ทั้งสองผูกกับ TaskManager.push_event/
request_approval/resolve_approval (ดู task_manager.py)
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


# Security 1.5: rate limit per-IP บน endpoint ที่เปิด browser จริง/เปลือง resource เท่านั้น
# (POST /tasks, POST /api/site-manual/learn — ผูก @limiter.limit() แยกทีละ endpoint ด้านล่าง
# ไม่ใช่ระดับ router เหมือน verify_api_key เพราะไม่ต้องการจำกัด SSE stream/endpoint polling
# ปกติที่ไม่เปลือง resource) — app.state.limiter ผูกใน main.py
#
# config_filename="" กัน slowapi พยายามอ่าน ".env" ของโปรเจกต์เอง (Limiter default
# auto-detect ไฟล์นี้ผ่าน starlette.config.Config ซึ่งเปิดไฟล์แบบไม่ระบุ encoding —
# .env มี comment ภาษาไทย/em dash เป็น UTF-8 อ่านด้วย cp1252 default ของ Windows ไม่ได้
# พังตอน import ทันที) เราไม่ได้ใช้ env var มา override ค่า limiter อยู่แล้ว จึงปิดเส้นทาง
# นี้ไปเลย ("" ไม่ใช่ None ทำให้ Config เข้า branch os.path.isfile() แล้วแค่ warn เงียบๆ) —
# กด UserWarning "Config file '' not found." ทิ้งไปด้วย (คาดไว้แล้ว ไม่ใช่ปัญหาจริง)
with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    limiter = Limiter(key_func=get_remote_address, config_filename="")


# Security (SEC-5 follow-up): SSE endpoint (GET .../stream) ใช้ EventSource ของ browser ซึ่ง
# ตั้ง custom header เองไม่ได้ — เดิมรับ raw settings.api_key ผ่าน query param ตรงๆ
# ("?api_key=...") ซึ่งเป็น long-lived secret เดียวกับที่ใช้ทุก request ทำให้มีโอกาสหลุดผ่าน
# access log/proxy/browser history ได้ง่ายกว่า header มาก — เปลี่ยนเป็น "ticket" อายุสั้น
# (ขอผ่าน POST /auth/stream-ticket ที่ยังต้องใช้ X-API-Key จริงผ่าน header ปกติก่อนถึงจะออก
# ticket ให้) แลกกับ api_key จริงแทน ถ้า ticket หลุดไปกับ log จริงๆ ก็ใช้ได้แค่ไม่กี่วินาที
# ไม่ใช่ตลอดไปเหมือน raw key เดิม
_STREAM_TICKET_TTL_SECONDS = 60.0
_stream_tickets: dict[str, float] = {}  # ticket -> expires_at (unix time)


def _issue_stream_ticket() -> str:
    ticket = secrets.token_urlsafe(32)
    _stream_tickets[ticket] = time.time() + _STREAM_TICKET_TTL_SECONDS
    return ticket


def _stream_ticket_is_valid(ticket: str) -> bool:
    # เก็บกวาด ticket ที่หมดอายุไปเรื่อยๆ ตอนเช็ค (ไม่ต้องมี background job แยก — จำนวน
    # ticket ที่ค้างในหน่วยความจำถูก bound ไว้เองโดยธรรมชาติ เพราะแต่ละ SSE connection ขอ
    # ครั้งเดียวแล้วใช้ทันทีภายใน TTL สั้นๆ)
    now = time.time()
    for expired in [t for t, exp in _stream_tickets.items() if exp < now]:
        _stream_tickets.pop(expired, None)
    expires_at = _stream_tickets.get(ticket)
    return expires_at is not None and expires_at >= now


async def verify_api_key(
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    ticket: Optional[str] = Query(default=None),
) -> None:
    """Security 1.1: ทุก route ใน api_router ต้องผ่านนี้ก่อนเสมอ (ผูกไว้ที่ระดับ router
    ด้านล่าง ไม่ต้องแปะ Depends() แยกทีละ endpoint) — settings.api_key ไม่ตั้งค่า (None,
    default) = auth ปิดสำหรับ local dev เท่านั้น รับได้ทั้ง header "X-API-Key" (fetch ปกติ)
    และ query param "?ticket=" (SSE endpoint ที่ EventSource ของ browser ตั้ง header เอง
    ไม่ได้ — ดู GET /tasks/{id}/stream, GET /api/site-manual/learn/{id}/stream, และ
    _issue_stream_ticket() ด้านบนสำหรับที่มาของ ticket)"""
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
    """Security (SEC-5 follow-up): ออก ticket อายุสั้น (60 วินาที, ใช้ได้หลายครั้งภายใน
    TTL — ไม่ single-use เพราะ EventSource ของ browser auto-reconnect เองได้ถ้า network
    สะดุด ต้องให้ ticket เดิมยังใช้ต่อได้จนกว่าจะหมดอายุจริง) แลกกับ X-API-Key จริง (ผ่าน
    verify_api_key() ที่ router ผูกไว้แล้วตั้งแต่ endpoint นี้เอง) — frontend เรียก endpoint
    นี้ก่อนเปิด EventSource ทุกครั้ง แล้วใช้ ticket แทน raw api_key ใน query string"""
    ticket = _issue_stream_ticket()
    return {"ticket": ticket, "expires_in": _STREAM_TICKET_TTL_SECONDS}


def _make_ask_user_func(task_manager: TaskManager, task_id: str, auto_approve: bool):
    """ask_user_func ที่ orchestrator ใช้ทั้งสองจุด: confirm_plan (ก่อนเริ่ม loop) และ
    permission-gated action ตอนรัน (actions.py) — cmd dict เดียวกันบอกได้อยู่แล้วว่าเป็น
    คำขอแบบไหน (cmd["type"] == "confirm_plan" หรือ action จริง เช่น "purchase"/"delete")
    ไม่ต้องแยก branch พิเศษ ส่งต่อให้ human ตัดสินใจจริงทั้งคู่ผ่านช่องทางเดียวกัน
    (TaskManager.request_approval -> push event "approval_request" เข้า SSE stream ->
    รอ POST /tasks/{id}/respond)

    auto_approve=True: ข้าม human-in-the-loop ทั้งหมด (ทั้ง plan และ action) อนุมัติเอง
    ทันที — ไว้ให้รันแบบไม่มีคนเฝ้าหน้าจอ (เช่น batch/CI) เหมือนพฤติกรรมเดิมก่อน W10[B]

    W10[E]: จำกัดเวลารอ human ตอบด้วย settings.approval_timeout_seconds เสมอ (ไม่ใช่รอ
    ตลอดกาล) — ไม่งั้น task ที่ user ปิดแท็บทิ้งกลางคันตอนรอ confirm plan จะยึด browser
    จาก pool ไว้ไม่มีวันคืน กัด quota ของ browser_pool_size จน task ใหม่ทุกตัวรอคิวไม่รู้จบ
    (ดู task_manager.py::request_approval() สำหรับรายละเอียดเต็ม)
    """

    async def ask_user_func(cmd: dict) -> bool:
        if auto_approve:
            await task_manager.push_event(task_id, {"kind": "auto_approved", "cmd": cmd})
            return True
        return await task_manager.request_approval(
            task_id, cmd, timeout=settings.approval_timeout_seconds
        )

    return ask_user_func


# W21 ("Self-Learned Site Manual Integration" ข้อ 3, Fallback Mechanism): ข้อความตายตัวตาม
# สเปค (Task/W21.txt Task5) ให้ /context โชว์เวลาโดเมนนี้ไม่มี manual เลย/ไม่มีหน้าไหน match
# goal เลย — บอก user ตรงๆ ว่าระบบจะ fallback ไป dynamic planner ตามปกติ ไม่ได้ค้าง/error
_NO_LEARNED_MANUAL_TEXT = "No pre-learned manual found. Executing dynamic exploration."


def _resolve_site_manual_context(domain: str, goal: str) -> str:
    """W21 ("Self-Learned Site Manual Integration" ข้อ 1-3): จุดตัดสินใจเดียวที่ทั้ง
    generate_plan endpoint (ร่างแผนคร่าวๆ) และ _run_with_resolved_browser (รัน task จริง
    ทุก step) เรียกใช้ร่วมกัน แทนที่จะ hardcode load_knowledge_text() ตรงๆ แบบเดิมทั้งสองที่
    — ถ้า manual ของโดเมนนี้มีหน้าที่ตรงกับ goal เจาะจง (find_matching_page คะแนน keyword
    overlap สูงสุด > 0) ให้ใช้ build_strict_manual_context() แทน (scope แคบลงเหลือหน้าเดียว
    พร้อม route/selector ที่บันทึกไว้จริง ขึ้นต้นด้วย marker "[PRE_LEARNED_MANUAL]" ที่
    llm.py::SYSTEM_PROMPT ตรวจหาเพื่อบังคับให้ planner ยึดตามอย่างเคร่งครัด) — ไม่เจอหน้าที่
    ตรงเลย (manual ไม่มี/ทุกหน้าคะแนน 0) fallback ไป load_knowledge_text() แบบเดิมเงียบๆ
    (สรุปสั้นๆ ของทุกหน้า ใช้เป็นข้อมูลอ้างอิงกว้างๆ อย่างที่เคยทำมา) ไม่ throw/ไม่บล็อกอะไร"""
    if manual_exists(domain):
        manual = load_manual(domain)
        matched_page = find_matching_page(manual, goal) if manual else None
        if matched_page is not None:
            return build_strict_manual_context(matched_page)
    return load_knowledge_text(domain)


async def _context_inspection_result(req, on_event, client, model: str, resolved_provider: str) -> dict:
    """W20 (MODULE 0 "Special Command Interceptor"): goal มีคำสั่ง "/context" ปน — ตอบด้วย
    คำอธิบายความเข้าใจ/แผนที่ตั้งใจจะทำ (ดู llm.context_inspection_reply()) โดยไม่ลงมือทำ
    จริงเลยไม่ว่ากรณีใด (ไม่แตะ browser/session/pool/file parser) เหมือน
    _general_chat_result ทุกประการ แค่ system prompt/โครงสร้างคำตอบต่างกัน

    W21 ("Self-Learned Site Manual Integration" ข้อ 1): ก่อนตอบ ลองหาว่า manual ที่เรียนรู้
    ไว้ล่วงหน้าของโดเมนนี้มีหน้าที่ตรงกับคำสั่งจริงหรือไม่ — เจอ ก็แปะ "📍 Learned Page Flow
    Sequence" ต่อท้ายคำตอบ ไม่เจอ (โดเมนนี้ไม่มี manual เลย/req.url ว่างเปล่า/ไม่มีหน้าไหน
    match) ก็แปะข้อความ fallback ตายตัวแทน (ดู _NO_LEARNED_MANUAL_TEXT ด้านบน)"""
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
        # W_context_knows_the_goal: URL เป้าหมายเป็นข้อเท็จจริงที่ request พกมาอยู่แล้ว ไม่มี
        # เหตุผลให้โมเดลต้องเดาหรือตอบว่า "Not specified"
        target_url=req.url or "",
    )
    await on_event({"kind": "chat_reply", "message": reply})
    return _chat_shaped_result(reply)


async def _general_chat_result(req, on_event, client, model: str, resolved_provider: str) -> dict:
    """W19-6 ("Master Controller" MODULE 1): ตอบ goal ที่เป็นคำถามทั่วไป/ทักทาย/วันเวลา/
    คำนวณเลข/คำขอเชิงแนะนำ-ความเห็นจากความรู้ทั่วไป (ดู llm.resolve_general_chat_query())
    โดยไม่แตะ browser/session/pool เลยแม้แต่นิดเดียว — client/model/resolved_provider รับ
    มาจาก caller ตรงๆ (สร้างครั้งเดียวใน _run_with_resolved_browser() แล้วใช้ร่วมกับ
    resolve_general_chat_query() ด้วย ไม่ต้องเรียก Orchestrator._llm_backend() ซ้ำสองครั้ง)
    คืน dict รูปแบบเดียวกับ Orchestrator.run_task() ทุกประการ (steps=0, history=[] ฯลฯ) ให้
    TaskManager/frontend polling/SSE ใช้ต่อได้เหมือนเดิมทุกจุดโดยไม่ต้องรู้ว่า task นี้ไม่เคย
    เปิด browser เลย"""
    # llm._current_bangkok_time_text() เป็น private helper ของ llm.py (ใช้ฉีดเวลาจริงเข้า
    # SYSTEM_PROMPT ทุก step อยู่แล้ว) — เรียกตรงๆ จากที่นี่แทนการ implement เวลา Bangkok
    # ซ้ำอีกจุด กันคำถามเกี่ยวกับวันที่/เวลาตอบผิดจาก training data ของโมเดลเอง
    chat_reply = await llm.chat_response(
        client, model, req.goal, resolved_provider, current_time_text=llm._current_bangkok_time_text(),
    )
    await on_event({"kind": "chat_reply", "message": chat_reply})
    return {
        "success": True,
        "steps": 0,
        "message": chat_reply,
        "history": [],
        "tokens": {"input": 0, "output": 0, "cache_read": 0, "cache_creation": 0},
        "plan": "",
        "final_page_state": "",
        "persona_message": "",
        "persona_status": "COMPLETED",
        "completion_verification": "OK",
    }


def _chat_shaped_result(message: str) -> dict:
    """dict รูปแบบเดียวกับ _general_chat_result()/Orchestrator.run_task() ทุกประการ —
    แยกออกมาเพราะ _file_query_result() ด้านล่างต้องคืนรูปแบบเดียวกันนี้ทั้งตอนสำเร็จและตอน
    error (ไฟล์อ่านไม่ได้/base64 เสีย) ไม่ใช่แค่ตอนสำเร็จแบบ _general_chat_result เดิม"""
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


_IMAGE_FILE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}


async def _file_query_result(
    req, on_event, client, model: str, resolved_provider: str, file_chat_memory: dict,
) -> dict:
    """pdf/xlsx: user แนบไฟล์ (.txt/.pdf/.docx/.xlsx/.csv หรือรูปภาพ) ผ่าน composer โดยตรง
    (req.attached_file_name/attached_file_content_base64 — ดู schemas.py::
    CreateTaskRequest) — เหมือน _general_chat_result ด้านบนทุกประการ (ไม่แตะ
    browser/session/pool เลย, คืน dict รูปแบบเดียวกับ run_task()) แค่มีเนื้อหาไฟล์เป็น
    context แทนความรู้ทั่วไปล้วนๆ

    รูปภาพ (นามสกุลใน _IMAGE_FILE_EXTENSIONS) แยกสาขาไปทาง llm.answer_image_query()
    (ส่ง bytes ดิบแบบ multimodal ตรงๆ) แทน load_manual_bytes()+answer_file_query() เพราะ
    ไม่มี "text" ให้ extract ล่วงหน้าเหมือนเอกสาร

    file_chat_memory: app.state.file_chat_memory (session_id -> {"filename","text"}) —
    เอกสาร (ไม่ใช่รูปภาพ) ที่ extract สำเร็จแล้วและมี req.session_id มาด้วย จะถูกจำไว้ที่นี่
    ให้เทิร์นถัดไปในเซสชันเดียวกันที่ไม่ได้แนบไฟล์ใหม่มาตอบต่อยอดได้ (ดู
    _file_chat_memory_reply()/_run_with_resolved_browser() ด้านล่าง) — ไม่ทำแบบเดียวกันกับ
    รูปภาพ (ไม่มี "text" ให้เก็บ ขอบเขตจำกัดไว้แค่เอกสารก่อน)

    ห้าม throw ออกไปทำให้ task พังเด็ดขาด — decode/extract ผิดพลาด (base64 เสีย/ไฟล์เสีย/
    นามสกุลไม่รองรับ) ต้องคืนข้อความ error ที่ user อ่านเข้าใจได้ผ่านช่องทางเดียวกับคำตอบ
    ปกติ ไม่ใช่ 500 ที่ทำให้ frontend ไม่รู้จะแสดงอะไร"""
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
            # W_file_followup_with_sticky_url: "เทิร์นล่าสุดของเซสชันนี้คือเทิร์นไฟล์" — ล้างเป็น
            # False ทันทีที่เซสชันนี้ไปรันงานเบราว์เซอร์ (ดูจุดล้างใน _run_with_resolved_browser)
            "is_latest_turn": True,
        }

    # W_file_answer_timing: user รายงานว่า "provider openai อ่านไฟล์นานมากทั้งที่ข้อมูลไม่เยอะ"
    # (telemetry มีแค่ duration รวมของทั้งเทิร์น = 21.8 วินาที บอกไม่ได้ว่าหมดไปกับ *แกะไฟล์*
    # หรือ *รอโมเดล*) — แยกสองช่วงให้เห็นใน console ก่อน แล้วค่อยแก้ตรงจุดที่ช้าจริง ไม่เดา
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
    """pdf/xlsx (ต่อ): เทิร์นถัดไปในเซสชันเดียวกันที่ไม่ได้แนบไฟล์ใหม่มา (goal ล้วนๆ) แต่
    session_id นี้เคยมีไฟล์แนบไว้แล้วในเทิร์นก่อนหน้า (ดู file_chat_memory ใน
    _file_query_result() ด้านบน) — ตอบต่อจากเนื้อหาไฟล์เดิมได้เลย ไม่ต้องให้ user แนบไฟล์
    ซ้ำทุกเทิร์น เหมือน _file_query_result() ทุกประการ (ไม่แตะ browser/session_registry/
    pool เลย) แค่ไม่มี base64/extraction ให้ทำซ้ำเพราะ remembered["text"] extract ไว้แล้ว

    บั๊กจริงที่ user รายงาน: เทิร์น 1 แนบไฟล์ถามสำเร็จ (ผ่าน _file_query_result() ด้านบน)
    แต่ไม่เคยเก็บ text ไว้ที่ไหนเลย เทิร์น 2 ("แต่ละวันทำอะไรบ้าง" ไม่มีไฟล์แนบ ไม่มี url)
    เลยตกไปเปิด browser จริงด้วย url ว่างเปล่า (orchestrator.run_task) ค้าง — ดู
    _run_with_resolved_browser()/generate_plan() สำหรับจุดเรียกฟังก์ชันนี้"""
    reply = await llm.answer_file_query(
        client, model, req.goal, remembered["text"], remembered["filename"], resolved_provider,
    )
    await on_event({"kind": "chat_reply", "message": reply})
    return _chat_shaped_result(reply)


_READ_PAGE_DATA_OK_PREFIX = "[OK] read_page_data -> "


async def _update_extracted_memory(session, result: dict, provider: Optional[str]) -> None:
    """W19-6 ("Master Controller" MODULE 2, "SESSION_LIST"): หลัง run_task() จบ (ไม่ว่า
    success หรือไม่ — task ที่ fail กลางทางอาจยัง extract ข้อมูลบางส่วนไปแล้วก่อนพังก็ได้)
    ไล่หา step ที่เป็น read_page_data สำเร็จ (result string ขึ้นต้นด้วย
    _READ_PAGE_DATA_OK_PREFIX) เอาแค่ "ตัวล่าสุด" (ไม่รวมทุก read_page_data ของ task นี้
    เพราะแต่ละครั้งอาจอ่านคนละส่วนของหน้า ไม่ใช่ accumulate กันได้ตรงๆ) มาจัดโครงสร้างผ่าน
    llm.extract_structured_items() แล้วเก็บทับ session.extracted_memory เดิม — ไม่เจอ
    read_page_data สำเร็จเลย/จัดโครงสร้างไม่ได้ (คืน [] ว่างเปล่า) ปล่อย
    session.extracted_memory เดิมไว้เฉยๆ ไม่ล้างทิ้ง (เผื่อ task นี้เป็นแค่ IN_PAGE_ACTION
    ที่ไม่ได้อ่านข้อมูลใหม่ ไม่ควรทำ buffer เดิมหายไปเปล่าๆ)

    ห้าม throw ออกไปทำให้ endpoint พังเด็ดขาด — เป็นแค่ enhancement (เหมือน
    plan_memory.py/long_term_memory.py) ไม่ใช่ requirement ที่ task ต้องพึ่ง"""
    try:
        history = result.get("history") or []
        latest_read_content = ""
        for step in history:
            step_result = step.get("result", "")
            if isinstance(step_result, str) and step_result.startswith(_READ_PAGE_DATA_OK_PREFIX):
                latest_read_content = step_result[len(_READ_PAGE_DATA_OK_PREFIX):]
        if not latest_read_content:
            return
        resolved_provider = provider or settings.llm_provider
        client, model, _, _, _ = Orchestrator._llm_backend(resolved_provider)
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
    """W13: ตรรกะร่วม "จะเอา page มาจากไหน" ระหว่าง POST /tasks (create_task,
    confirm_plan) และ POST /api/execute_plan (approved_plan) — ต่างกันแค่
    extra_run_task_kwargs ที่ผู้เรียกส่งมา (confirm_plan=... หรือ approved_plan=...)
    req รับได้ทั้ง CreateTaskRequest/ExecutePlanRequest (duck-typed — ใช้ attribute ชุด
    เดียวกันทั้งคู่: url/goal/max_steps/provider/headless/auto_approve/
    use_user_browser/tab_reuse_policy/session_id)

    W19-6 ("Master Controller" MODULE 1, "General QA / No-Browser Trigger"): เช็คก่อนสุด
    เสมอ ก่อนแม้แต่ session_id branch ด้านล่าง — goal ที่เป็นคำถามทั่วไป/ทักทาย/วันเวลา/
    คำนวณเลข/คำขอเชิงแนะนำจากความรู้ทั่วไป (ดู llm.is_general_chat_query()) ไม่ต้อง
    resolve session/pool/browser อะไรเลยแม้จะมี session_id ที่เปิด page ค้างอยู่แล้วก็ตาม
    (ตัดสินใจแบบ deterministic ล้วนๆ ไม่พึ่ง LLM เด็ดขาด — ตั้งใจ: (1) เร็ว ไม่เสีย LLM
    round-trip ให้ทุก task ที่ไม่ใช่ general-chat จริงๆ ต้องรอเปล่าๆ ก่อนเริ่ม (2)
    ปลอดภัยกับเทสต์/โค้ดเดิมที่ patch("...Orchestrator") ทั้ง class ไว้กว้างๆ ทั่วทั้งไฟล์
    (เคยลองเพิ่มชั้น LLM fallback ที่นี่มาก่อนแล้วพบว่า Orchestrator._llm_backend() ถูกเรียก
    ก่อนแม้แต่ task ปกติที่ goal ไม่มี exclusion keyword เลย เช่น "ทดสอบ" ทำให้เทสต์ที่ mock
    Orchestrator ทั้ง class พังเป็นวงกว้าง 26 เทสต์ ถอนออกแล้ว) — ดู module comment ของ
    is_general_chat_query() ใน llm.py สำหรับเหตุผลเต็มเรื่อง false negative ปลอดภัยกว่า
    false positive)

    pdf/xlsx: ไฟล์ที่ user แนบมา (req.attached_file_content_base64) เช็คก่อน
    is_general_chat_query() แม้แต่ก็ตาม — เป็นสัญญาณที่ชัดเจนกว่า/แรงกว่า keyword matching
    ด้านล่างเสมอ (user แนบไฟล์มาแล้วแปลว่าต้องการให้อ่านไฟล์นั้นแน่ๆ ไม่ต้องเดาจาก goal
    text อีกชั้น) ดู _file_query_result() ด้านบน — เหมือน general-chat check ทุกประการ
    (ไม่แตะ browser/session/pool เลย)

    pdf/xlsx (ต่อ, บั๊กจริงที่ user รายงาน): เทิร์นถัดไปในเซสชันเดียวกันที่ไม่ได้แนบไฟล์ใหม่
    มา (req.attached_file_content_base64 ว่างเปล่า) แต่ session_id นี้เคยมีไฟล์แนบไว้แล้ว
    (ดู file_chat_memory) เช็คก่อน is_general_chat_query() ไม่ทัน (goal อย่าง "แต่ละวันทำ
    อะไรบ้าง" ไม่ match pattern ไหนเลย) และก่อน session_id branch ด้านล่างด้วย (ซึ่งเดิมจะ
    เปิด browser จริงด้วย url ว่างเปล่าแล้วค้าง เพราะ session ที่มาจาก _file_query_result()
    ไม่เคยแตะ session_registry เลย ไม่มี page ให้ perceive) — ต้องไม่ใช่ goal ที่มีคำบ่งบอก
    web action จริงๆ (llm.goal_mentions_web_action — เกณฑ์เดียวกับ exclusion keyword ของ
    is_general_chat_query) และไม่มี req.url มาด้วย (มี url = สัญญาณชัดเจนว่าต้องการ browse
    จริง ไม่ใช่ถามต่อจากไฟล์เดิม) ถึงจะตอบจากไฟล์เดิมได้เลย ไม่งั้น fall through ไปเส้นทาง
    ปกติด้านล่างตามเดิมทุกประการ

    ลำดับความสำคัญ (หลังผ่านทุก check ด้านบนแล้ว): session_id ก่อน (ครอบคลุมทั้ง
    3 โหมดในตัวผ่าน session_registry อยู่แล้ว) -> use_user_browser -> headless=False ตรงๆ
    (visible browser, launch เอง, ผูก keep_browser_open=True คู่กันเสมอเพราะไม่มีประโยชน์
    ที่จะเปิดหน้าต่างโชว์แล้วรีบปิดทันทีที่เสร็จ) -> fallback ไปยืมจาก pool (headless ตาม
    req.headless เป๊ะๆ)"""
    # W20 (MODULE 0 "Special Command Interceptor"): เช็คก่อนทุก check อื่นเสมอ (ก่อนแม้แต่
    # attached_file ด้านล่าง) — "/context" คือ debug/inspection mode ที่ user ต้องการดูว่า
    # agent เข้าใจคำสั่งว่าอะไร ไม่ต้องการให้ลงมือทำจริงไม่ว่ากรณีใด (ดู
    # llm.is_context_inspection_command() สำหรับเหตุผลเต็ม)
    if llm.is_context_inspection_command(req.goal):
        resolved_provider = req.provider or settings.llm_provider
        client, model, _, _, _ = Orchestrator._llm_backend(resolved_provider)
        return await _context_inspection_result(req, on_event, client, model, resolved_provider)

    if req.attached_file_content_base64:
        resolved_provider = req.provider or settings.llm_provider
        client, model, _, _, _ = Orchestrator._llm_backend(resolved_provider)
        return await _file_query_result(req, on_event, client, model, resolved_provider, file_chat_memory)

    if llm.is_general_chat_query(req.goal):
        resolved_provider = req.provider or settings.llm_provider
        client, model, _, _, _ = Orchestrator._llm_backend(resolved_provider)
        return await _general_chat_result(req, on_event, client, model, resolved_provider)

    remembered_file = file_chat_memory.get(req.session_id) if req.session_id else None
    # W_file_followup_with_sticky_url: เดิมบังคับว่า req.url ต้องว่าง แต่ช่อง URL บนหน้าจอค้าง
    # ค่าไว้จากงานก่อนหน้าในเซสชันเดียวกัน คำถามต่อยอดจากไฟล์ ("สรุปเป็นตาราง") จึงหลุดไปเข้า
    # agent loop ของเบราว์เซอร์แล้วตอบว่า "มีทั้งหมด 0 รายการ" — ยอมให้ผ่านได้ทั้งที่มี url ถ้า
    # (ก) เทิร์นล่าสุดของเซสชันนี้เป็นเทิร์นไฟล์จริง และ (ข) ถ้อยคำอ้างถึงข้อมูลที่เพิ่งได้มา
    # ไม่ใช่การกระทำบนหน้าเว็บ (ดู llm.is_file_followup_request)
    if remembered_file and not llm.goal_mentions_web_action(req.goal) and (
        not req.url
        or (remembered_file.get("is_latest_turn") and llm.is_file_followup_request(req.goal))
    ):
        resolved_provider = req.provider or settings.llm_provider
        client, model, _, _, _ = Orchestrator._llm_backend(resolved_provider)
        return await _file_chat_memory_reply(req, on_event, client, model, resolved_provider, remembered_file)

    # W_file_followup_with_sticky_url: ผ่านสี่ทางลัดมาถึงตรงนี้ = เทิร์นนี้จะใช้เบราว์เซอร์จริง
    # ไฟล์ที่จำไว้จึงไม่ใช่ "สิ่งที่เพิ่งคุยกัน" อีกต่อไป คำถามต่อยอดหลังจากนี้ต้องแนบไฟล์ใหม่หรือ
    # ถามแบบไม่มี url ถึงจะกลับไปเส้นทางไฟล์ได้ (กันไม่ให้คำสั่งงานเว็บถูกตอบจากไฟล์เก่า)
    if remembered_file is not None:
        remembered_file["is_latest_turn"] = False

    wants_visible_browser = req.headless is False
    # W14: โหลดคู่มือเว็บไซต์ที่ crawl มาอัตโนมัติครั้งเดียวตรงนี้ (ถ้ามี) แล้วส่งต่อเข้า
    # run_task() ทุก branch ด้านล่าง — ว่างเปล่าเงียบๆ ถ้าโดเมนนี้ยังไม่เคยถูกเรียนรู้
    site_manual_context = _resolve_site_manual_context(extract_domain(req.url), req.goal)

    # W12: session_id มา -> ผูก task นี้เข้ากับ session ที่มีชีวิตอยู่ข้ามหลาย request
    # (ดู core/session_registry.py) ครั้งแรกที่เจอ session_id นี้จะสร้าง page ใหม่ตาม
    # use_user_browser/headless ของ request นี้ ครั้งถัดๆ ไปด้วย session_id เดิมได้ page
    # ตัวเดิมกลับมาทันที (ไม่ต้อง acquire/launch/connect ซ้ำ) — ส่ง page= ตรงๆ ให้
    # orchestrator ข้าม acquisition/teardown ทั้งหมด (managed_externally=True)
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

        # W19-6 ("Master Controller" MODULE 3, "Ordinal Selection"/multi-turn strategy):
        # เฉพาะตอน session นี้เคยมี extracted_memory เก็บไว้จากเทิร์นก่อนหน้าจริงๆ
        # เท่านั้นถึงเรียก route_multi_turn_strategy() (ไม่มี memory เลย = พฤติกรรมเดิม
        # ทุกประการ ไม่เพิ่ม LLM call แถมโดยไม่จำเป็นสำหรับ task ทั่วไป/ครั้งแรกของ session)
        effective_goal = req.goal
        if session.extracted_memory:
            # client/model สร้างตรงนี้ (ไม่ใช่ด้านบนสุดของฟังก์ชัน) เพราะ branch นี้ทำงาน
            # เฉพาะตอน session มี extracted_memory จริงๆ เท่านั้น (ดู comment ด้านบน) —
            # task ทั่วไป/session ที่ยังไม่เคย extract อะไรเลยจะไม่เสีย
            # Orchestrator._llm_backend() call เปล่าๆ เหมือนกับเหตุผลเดียวกับ general-chat
            # check ด้านบนสุดของฟังก์ชัน
            resolved_provider = req.provider or settings.llm_provider
            client, model, _, _, _ = Orchestrator._llm_backend(resolved_provider)
            memory_json = json.dumps(session.extracted_memory, ensure_ascii=False)
            decision = await llm.route_multi_turn_strategy(
                client, model, req.goal, req.goal, extract_domain(session.page.url),
                session.page.url, memory_json, "", resolved_provider,
            )
            if decision["chosen_strategy"] == "REPLY_FROM_MEMORY":
                # คำตอบอยู่ใน buffer แล้ว — ตอบตรงๆ ไม่แตะ browser/agent loop เลย (เหมือน
                # _general_chat_result ด้านบนแต่มี buffer เป็นบริบทเพิ่ม)
                reply = await llm.chat_response(
                    client, model,
                    f"{req.goal}\n\nข้อมูลที่เคยดึงไว้จากเทิร์นก่อนหน้า (เรียงตามลำดับที่แสดงบนหน้าจอจริง):\n{memory_json}",
                    resolved_provider,
                )
                await on_event({"kind": "chat_reply", "message": reply})
                return {
                    "success": True, "steps": 0, "message": reply, "history": [],
                    "tokens": {"input": 0, "output": 0, "cache_read": 0, "cache_creation": 0},
                    "plan": "", "final_page_state": "", "persona_message": "",
                    "persona_status": "COMPLETED", "completion_verification": "OK",
                }
            if decision["chosen_strategy"] == "IN_PAGE_ACTION":
                # ผูก goal เข้ากับรายการที่ผู้ใช้อ้างถึงจริงๆ จาก buffer (เช่น "เล่นเพลงที่
                # 3" -> ชื่อเพลง/รายละเอียดจริงจาก SESSION_LIST[2]) แทนที่จะปล่อยให้ loop
                # หลักตีความ ordinal เองล้วนๆ จาก DOM ที่อาจจัดลำดับต่างไปแล้วในเทิร์นนี้
                target_entity = decision["context_analysis"].get("target_entity_from_memory", "")
                if target_entity:
                    effective_goal = (
                        f"{req.goal}\n\n[อ้างอิงจากรายการที่เคยแสดงไว้ก่อนหน้า]: {target_entity}"
                    )
            elif decision["chosen_strategy"] == "NEW_NAVIGATION":
                # user ตั้งใจเปลี่ยนหัวข้อ/เว็บใหม่จริงๆ — buffer เดิมไม่เกี่ยวข้องอีกต่อไป
                # เคลียร์ทิ้งกันเทิร์นถัดไปอ้างอิงรายการเก่าที่ไม่เกี่ยวกับหัวข้อใหม่แล้วผิดๆ
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

    # W12: user_browser bypass pool เหมือน wants_visible_browser ด้านล่าง (browser ที่
    # ต่อผ่าน CDP เป็นของ user เอง ไม่ใช่ของ pool ให้ยืม) — ตรวจก่อน wants_visible_browser
    # เพราะ headless ไม่มีความหมายเลยในโหมดนี้ (ต่อเข้า browser จริงที่เปิดอยู่แล้ว ไม่
    # launch เอง) ไม่ต้องสน req.headless
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


# W_retry_value_has_no_home (บั๊กจริงที่ user เจอบน Test Console 2026-09-04): task ที่จบด้วย
# TASK_FAILED_USER_INPUT_ERROR ส่งข้อความบอก user ว่า "กรุณาตอบกลับมาด้วยค่าใหม่ที่ต้องการใช้แทน
# ระบบจะกรอกค่านั้นแทนที่ในช่องเดิมแล้วดำเนินการต่อให้ทันที" และหน้าเว็บมีช่องให้กรอกตอบด้วย —
# แต่ไม่มีโค้ดส่วนไหนรับค่านั้นไปทำอะไรเลย มันถูกส่งเป็น goal ใหม่ดิบๆ agent จึงตอบว่า
# "I can't determine the intended task ... the user's goal is only 'Abcd1234'" ซึ่งถูกต้องตาม
# ข้อมูลที่มันได้รับ — คำสัญญาในข้อความต่างหากที่ไม่มีของจริงรองรับ
#
# แปลง goal ที่ endpoint ก่อนใครทั้งหมด เพื่อให้ _run_with_resolved_browser() และ
# _will_use_browser() (ซึ่งต้องตัดสินใจตรงกันเสมอ ดู docstring ของทั้งคู่) เห็น goal เดียวกัน
# ตัดสินด้วย keyword ล้วน ไม่เรียก LLM ตามกฎเดิมของเส้นทางนี้
_MAX_BARE_VALUE_CHARS = 64


def _goal_is_a_bare_replacement_value(goal: str) -> bool:
    """ข้อความที่ user ตอบกลับมาเป็น "ค่า" เฉยๆ ไม่ใช่คำสั่งใหม่

    ต้องเป็น token เดียวไม่มีช่องว่างเลย: เกณฑ์ "ไม่เกิน 4 คำ" ที่ลองก่อนหน้าใช้ไม่ได้กับ
    ภาษาไทย เพราะไทยไม่มีเว้นวรรคระหว่างคำ — goal จริงอย่าง "เปิดเว็ปแล้วเปลี่ยนรหัสผ่านเป็น
    12345678" นับได้แค่ 2 คำ แล้วถูกเข้าใจผิดว่าเป็นค่าเปล่าทันที (บทเรียนเดียวกับ
    W_thai_keyword_space) ค่าที่ user พิมพ์ตอบช่องนี้เป็นรหัสผ่าน/ตัวเลข/ชื่อสั้นๆ ซึ่งไม่มี
    ช่องว่างอยู่แล้วโดยธรรมชาติ — เดาผิดทางนี้แค่ทำให้ไม่แปลง goal (พฤติกรรมเดิม) ไม่เสียหาย"""
    text = (goal or "").strip()
    if not text or len(text) > _MAX_BARE_VALUE_CHARS or len(text.split()) != 1:
        return False
    # token เดียวที่มีอักษรไทยปนอยู่มักเป็น "คำสั่ง" ไม่ใช่ "ค่า" — "ลบuserrole=ess" ผ่านเกณฑ์
    # ข้างบนครบทุกข้อทั้งที่เป็นคำสั่งเต็มรูป ส่วนค่าที่พิมพ์ตอบช่องนี้ในโปรเจกต์นี้เป็นรหัสผ่าน/
    # ตัวเลข/รหัสอ้างอิงซึ่งเป็น latin หรือตัวเลขล้วนเสมอ เดาผิดทางนี้แค่ไม่แปลง goal = พฤติกรรมเดิม
    if detect_goal_language(text)["script"] not in ("latin", "other"):
        return False
    return not llm.goal_mentions_web_action(text)


def _replacement_value_goal(value: str, labels: list) -> str:
    """คำสั่งต้องระบุช่องแบบไม่กำกวม — รันสด 2026-09-04: คำสั่งเวอร์ชันแรกเขียนว่าให้กรอกช่อง
    "Password" แล้วโมเดลไปกรอกช่อง "Current Password" แทน เพราะชื่อหนึ่งเป็น substring ของอีก
    ชื่อหนึ่งพอดี ผลคือรหัสปัจจุบันถูกเขียนทับ ส่วนช่องที่ผิดจริงยังค้างค่าเดิมไว้เหมือนเดิม"""
    fields = ", ".join(f'"{label}"' for label in labels)
    return (
        f'กรอกค่า "{value.strip()}" ลงในช่องที่มี label ตรงตัวว่า {fields} '
        "ให้ตรงกันทุกช่อง (แทนที่ค่าเดิมที่ระบบปฏิเสธ) "
        "ห้ามแก้ช่องอื่นเด็ดขาด โดยเฉพาะช่องรหัสผ่านปัจจุบัน (Current Password) "
        "ซึ่งกรอกถูกอยู่แล้ว แล้วจึงบันทึกฟอร์ม"
    )


def _apply_pending_replacement_value(req, pending_value_request: dict) -> None:
    """แปลง goal ที่เป็น "ค่าเปล่า" ให้เป็นคำสั่งที่ระบุช่องชัดเจน ถ้าเทิร์นก่อนหน้าขอค่าใหม่ไว้

    ต้องเรียกจาก **ทุก** endpoint ที่รับ goal ของ user: หน้า Test Console ไม่ได้ยิง
    POST /tasks เลยเมื่อมีแผน — ปุ่ม "ส่ง" ของช่องกรอกค่าใหม่เรียก requestPlan() ซึ่งไปที่
    /api/generate_plan แล้วต่อด้วย /api/execute_plan (ดู index.html::submitCorrectionValue)
    เวอร์ชันแรกต่อท่อไว้ที่ create_task ที่เดียว ค่าที่ user ตอบจึงยังหลุดไปเป็น goal ดิบๆ
    เหมือนเดิมทุกประการเมื่อใช้ผ่านหน้าเว็บจริง (สคริปต์ทดสอบของผมยิง /tasks ตรงจึงไม่เจอ)
    และ generate_plan ต้องแปลงด้วย ไม่ใช่แค่ execute_plan — ไม่งั้นแผนถูกร่างจากคำว่า
    "Abcd1234" ล้วนๆ ตั้งแต่ต้น"""
    labels = (pending_value_request.get(req.session_id) or {}).get("labels") or []
    if labels and _goal_is_a_bare_replacement_value(req.goal):
        req.goal = _replacement_value_goal(req.goal, labels)


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
        # W_retry_value_has_no_home: จำไว้ว่ารอค่าใหม่อยู่ (หรือเลิกรอ ถ้ารอบนี้ไม่ได้จบแบบนั้น)
        if req.session_id:
            labels = (result or {}).get("retry_value_field_labels") or []
            if labels:
                pending_value_request[req.session_id] = {"labels": labels}
            else:
                pending_value_request.pop(req.session_id, None)
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
    """W13: เฟสวางแผนแยกต่างหาก — synchronous ตรงๆ (ไม่ผ่าน TaskManager/SSE เพราะเป็น
    แค่ LLM call เดียว ไม่ใช่ agent loop หลายนาทีเหมือน execute_plan) ไม่เปิด/connect
    browser ใหม่เด็ดขาด (ดู orchestrator.py::Orchestrator.generate_plan()) — ถ้ามี
    session_id ที่มี page เปิดค้างอยู่แล้วจริง (จากเทิร์นก่อนหน้า) จะ perceive หน้านั้น
    มาช่วยร่างแผนให้ grounded กับสถานะปัจจุบัน (ใช้ session_registry.get() เฉยๆ ไม่ใช่
    get_or_create() — ไม่มีทางสร้าง session ใหม่จาก endpoint นี้)

    W20: เช็ค Plan Memory ก่อนเรียก LLM เสมอ (ดู core/plan_memory.py) — หา approved plan
    ที่ตรงกับ (domain, goal) นี้มากที่สุดด้วย semantic search (ไม่ใช่ exact text match —
    "Login"/"Sign in"/"เข้าสู่ระบบ" ควรจับคู่ lineage เดียวกันได้) เจอ = คืนแผนนั้นตรงๆ
    เลย ข้าม LLM ไปทั้งหมด (Plan Priority: user-approved มาก่อน LLM เสมอ) ไม่เจอ/ไม่ตรงพอ
    = fallback ไปให้ LLM ร่างใหม่ตามปกติด้านล่าง

    W_procmem: เช็ค procedural template (ดู core/procedural_memory.py) ก่อน Plan
    Memory ด้านบนอีกที (ลำดับความสำคัญเต็ม: procedural -> plan_memory -> LLM ร่างใหม่)
    ปิดไว้ default (settings.enable_procedural_memory=False) จนกว่าจะ validate คุณภาพ
    template/locator resolution บนเว็บจริงก่อน (ดู Phase 4 ใน implementation plan) —
    เจอ candidate + Planner ตัดสินใจ reuse/adapt ด้วย confidence ผ่านเกณฑ์ จะคืนแผนที่
    render จาก steps จริง (พร้อม template_id/slot_values/steps ให้ frontend ส่งต่อเข้า
    POST /api/execute_plan เพื่อวิ่งผ่าน fast-path executor ได้) — ไม่เจอ/ไม่มั่นใจพอ
    (plan_fresh) fallback ไปที่ plan_memory/LLM ตามปกติด้านล่างเหมือนไม่มี feature นี้เลย

    pdf/xlsx: ไฟล์ที่ user แนบมา (req.attached_file_content_base64) เช็คก่อนสุดเสมอ เหมือน
    _run_with_resolved_browser() ด้านล่าง — คืน is_qa=True ทันที ไม่เรียก classify_intent()
    เลย (deterministic ล้วนๆ ไม่ต้องเสีย LLM round-trip เพื่อรู้สิ่งที่รู้อยู่แล้วจากการมี
    ไฟล์แนบมา) ให้ frontend ข้ามหน้าต่างอนุมัติ PLAN ไปตอบจากไฟล์ได้ทันที (เหมือน qa_summary
    intent ปกติทุกประการ)

    pdf/xlsx (ต่อ, บั๊กจริงที่ user รายงาน): เทิร์นถัดไปในเซสชันเดียวกันที่ไม่ได้แนบไฟล์ใหม่มา
    แต่ session_id นี้เคยมีไฟล์แนบไว้แล้ว (ดู app.state.file_chat_memory) และ goal ไม่มีคำ
    บ่งบอก web action จริงๆ (llm.goal_mentions_web_action) ไม่มี req.url มาด้วย — เช็คถัดจาก
    attached_file ด้านบนทันที ด้วยเกณฑ์เดียวกับ _run_with_resolved_browser() (ดูที่นั่น
    สำหรับเหตุผลเต็มๆ) คืน is_qa=True ทันทีเหมือนกัน ให้ frontend ข้าม plan approval ไปตอบ
    จากไฟล์เดิมทันทีที่ POST /api/execute_plan (ไม่ใช่ตกไปวางแผนเปิด browser จริงด้วย url
    ว่างเปล่าเหมือนที่เคยเกิดขึ้นจริง)

    W20 ("Context-Aware Implicit Execution", บั๊กจริงที่ user รายงาน): เทิร์นก่อนหน้าเป็น
    general-chat ล้วนๆ (เช่น "ขอเพลงเศร้าๆหน่อย" -> agent แนะนำชื่อเพลงผ่าน chat_response()
    เฉยๆ ไม่เคยแตะ browser/session เลย) แล้วเทิร์นถัดมา user พิมพ์คำสั่งอ้างอิงกำกวม (เช่น
    "okเปิดให้หน่อย") — เดิมแผนที่ร่างออกมามีแค่ "เปิด youtube.com" เพราะ session.
    extracted_memory (route_multi_turn_strategy) ไม่มีทางจับ entity จาก general-chat reply
    ได้เลย (ไม่เคยมี read_page_data ให้ extract) ตอนนี้ req.previous_user_goal/
    previous_assistant_message (frontend ส่งมาจาก conversation history ฝั่ง client เอง — ดู
    index.html::requestPlan()) ส่งต่อเข้า Orchestrator.generate_plan() ให้ LLM ดึงชื่อ
    เพลง/entity จากคำตอบเทิร์นก่อนหน้ามารวมเข้ากับ goal ก่อนร่างแผนจริง (ดู llm.py::
    _PLAN_PROMPT_TEMPLATE ส่วน "Context-Aware Implicit Execution")"""
    # W20 (MODULE 0): เช็คก่อนสุดเสมอ เหมือน attached_file ด้านล่าง — คืน is_qa=True ทันที
    # ให้ frontend ข้ามหน้าต่างอนุมัติ PLAN ไปเรียก execute_plan/create_task ที่จะตอบด้วย
    # CONTEXT_INSPECTION_MODE ผ่าน _context_inspection_result() แทน (ดู routes.py::
    # _run_with_resolved_browser)
    if llm.is_context_inspection_command(req.goal):
        return GeneratePlanResponse(plan="", is_qa=True)

    if req.attached_file_content_base64:
        return GeneratePlanResponse(plan="", is_qa=True)

    # W_retry_value_has_no_home: แปลงก่อนร่างแผน ไม่งั้นแผนถูกร่างจากค่าเปล่าๆ
    _apply_pending_replacement_value(req, request.app.state.pending_value_request)
    file_chat_memory: dict = request.app.state.file_chat_memory
    remembered_file = file_chat_memory.get(req.session_id) if req.session_id else None
    # W_file_followup_with_sticky_url: เดิมบังคับว่า req.url ต้องว่าง แต่ช่อง URL บนหน้าจอค้าง
    # ค่าไว้จากงานก่อนหน้าในเซสชันเดียวกัน คำถามต่อยอดจากไฟล์ ("สรุปเป็นตาราง") จึงหลุดไปเข้า
    # agent loop ของเบราว์เซอร์แล้วตอบว่า "มีทั้งหมด 0 รายการ" — ยอมให้ผ่านได้ทั้งที่มี url ถ้า
    # (ก) เทิร์นล่าสุดของเซสชันนี้เป็นเทิร์นไฟล์จริง และ (ข) ถ้อยคำอ้างถึงข้อมูลที่เพิ่งได้มา
    # ไม่ใช่การกระทำบนหน้าเว็บ (ดู llm.is_file_followup_request)
    if remembered_file and not llm.goal_mentions_web_action(req.goal) and (
        not req.url
        or (remembered_file.get("is_latest_turn") and llm.is_file_followup_request(req.goal))
    ):
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
            resolved_provider = req.provider or settings.llm_provider
            planner_client, planner_model, _, _, _ = Orchestrator._llm_backend(resolved_provider)
            # W_procmem: บอก Planner ตรงๆ ว่าโดเมนนี้มี auto-login credential เก็บไว้ไหม
            # (ดู orchestrator.py::_maybe_auto_login) — ไม่งั้นมันจะเดา (ผิด) ว่า
            # candidate ที่ไม่มี step login เลยขาดอะไรไปแล้วปฏิเสธ reuse ทั้งที่จริงๆ
            # login เกิดขึ้นอัตโนมัตินอก template อยู่แล้วเสมอ (เจอบั๊กนี้จริงตอน
            # ทดสอบกับ OrangeHRM)
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
        # W_planhang: ดู settings.plan_generation_timeout_seconds — ไม่มี timeout นี้
        # request จะค้างเงียบๆ ไม่มีวันจบถ้า LLM provider ตอบช้าผิดปกติ (rate limit/
        # network) ทำให้ composer หน้าเว็บดูค้างที่ "Generating plan…" ตลอดไป ดีกว่าให้
        # user เห็น error แล้วลองใหม่ได้ทันที
        raise HTTPException(
            status_code=504,
            detail=f"LLM ไม่ตอบสนองภายใน {settings.plan_generation_timeout_seconds:.0f} วินาที ลองใหม่อีกครั้ง",
        )
    except Exception as e:
        # ห่อ exception ทุกชนิด (LLM API error, page เดิมจาก session_id ถูกปิด/นำทางไปแล้ว
        # ระหว่าง perceive ฯลฯ) เป็น HTTPException ที่มี detail จริง — ไม่งั้น FastAPI จะคืน
        # 500 เปล่าๆ ("Internal Server Error" ไม่มี context) ให้ frontend เห็นแค่นั้น
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
    return GeneratePlanResponse(plan=plan, is_qa=is_qa)




@router.post("/api/execute_plan", response_model=TaskCreatedResponse, status_code=202)
async def execute_plan(req: ExecutePlanRequest, request: Request) -> TaskCreatedResponse:
    """W13: รันแผนที่อนุมัติแล้วจาก POST /api/generate_plan (req.plan อาจถูก user แก้ไข
    ข้อความมาก่อนก็ได้) — เหมือน create_task() ทุกประการยกเว้นไม่มี confirm_plan gate
    เลย (อนุมัติไปแล้วตั้งแต่ก่อนเรียก endpoint นี้)

    W20: ทุกครั้งที่ user กด Approve (ไม่ว่าจะแก้ไขข้อความแผนมาก่อนหรือไม่) บันทึกเข้า
    Plan Memory เสมอ (ดู core/plan_memory.py::save_confirmed_plan) — "Confirm" คือจุดที่
    ถือว่าแผนนี้ approved แล้วตามสเปค ไม่ต้องรอ flag แยกจาก frontend ว่าแก้ไขหรือไม่ (ถ้า
    เนื้อหาเหมือน version ล่าสุดเป๊ะอยู่แล้ว plan_memory จะไม่สร้าง version ซ้ำซ้อนเปล่าๆ
    เอง) — draft ที่ยังไม่ confirm/plan ที่ user cancel ไม่มีทางมาถึง endpoint นี้เลย

    W_procmem: req.execution_mode == "fastpath" (มาจาก GeneratePlanResponse.source ที่
    ขึ้นต้นด้วย "procedural_" — ดู index.html) ข้าม plan_memory.save_confirmed_plan ไปเลย
    (ไม่บันทึก preview text ที่ render จาก steps จริงเป็น "แผนดิบ" ปนกับของจริงที่ user
    พิมพ์เอง — คนละบทบาทกัน) แล้ววิ่งผ่าน orchestrator.run_fastpath() แทน run_task() ปกติ
    ถ้า template_id/steps มีมาจริงและไม่ได้ขอโหมดที่ fast-path ยังไม่รองรับ (ดู
    Orchestrator.run_fastpath() docstring — รองรับแค่ page=/browser=, ไม่รองรับ
    use_user_browser/visible-window เอง) — โหมดที่ไม่รองรับ fallback ไปที่ slow-path
    ปกติเงียบๆ (ไม่ error) เหมือนไม่มี fast-path feature นี้เลย"""
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
        # W_retry_value_has_no_home: จำไว้ว่ารอค่าใหม่อยู่ (หรือเลิกรอ ถ้ารอบนี้ไม่ได้จบแบบนั้น)
        # — เส้นทางนี้คือเส้นทางที่หน้า Test Console ใช้จริงเมื่อมีแผน
        if req.session_id:
            labels = (result or {}).get("retry_value_field_labels") or []
            if labels:
                pending_value_request[req.session_id] = {"labels": labels}
            else:
                pending_value_request.pop(req.session_id, None)
        return result

    resolved_headless = settings.browser_headless if req.headless is None else req.headless
    record = task_manager.submit(
        task_id, req.url, req.goal, req.provider, _run(), headless=resolved_headless,
        attached_file_name=req.attached_file_name,
    )
    return TaskCreatedResponse(task_id=record.task_id, status=record.status)


@router.get("/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task(task_id: str, request: Request) -> TaskStatusResponse:
    task_manager = request.app.state.task_manager
    record = task_manager.get(task_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"ไม่พบ task_id: {task_id!r}")
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


@router.get("/tasks", response_model=list[TaskStatusResponse])
async def list_tasks(request: Request) -> list[TaskStatusResponse]:
    task_manager = request.app.state.task_manager
    return [
        TaskStatusResponse(
            task_id=r.task_id,
            url=r.url,
            goal=r.goal,
            provider=r.provider,
            status=r.status,
            created_at=r.created_at,
            result=r.result,
            error=r.error,
            headless=r.headless,
            attached_file_name=r.attached_file_name,
        )
        for r in task_manager.list()
    ]


async def _stream_task_events(record):
    """W10[B]/W25/W26: async generator ของ SSE body สำหรับ GET /tasks/{id}/stream — แยก
    ออกมาจาก stream_task() เป็นฟังก์ชันระดับโมดูล (แทนที่จะเป็น closure ซ้อนใน endpoint
    เหมือนเดิม) เพื่อให้เทสต์เรียกตรงๆ ได้โดยไม่ต้องพึ่ง TestClient ที่ block รอจนกว่า
    stream จะปิด (ใช้ยากมากสำหรับเทสต์ที่ต้องจำลองหลาย connection คาบเกี่ยวกัน)

    ถ้า client มาเชื่อมต่อ *หลัง* task จบไปแล้ว จะไม่มี event เก่าให้ replay (ไม่ได้เก็บ
    log buffer แยก) เลยส่ง task_done สังเคราะห์กลับทันทีจาก record.status/result/error
    ที่ยังอยู่แทน เพื่อให้หน้าเว็บที่รีเฟรชทีหลังยังเห็นผลลัพธ์สุดท้ายได้ (ไม่ hang รอ
    event ที่ไม่มีวันมาอีกแล้ว)

    W26 ("Broadcast live events to every subscribed tab" — บั๊กจริงที่ user รายงาน:
    approval prompt/log ไม่ขึ้นสด ต้องกด F5, แม้หลัง W25 ก็ยังเจอ): frontend
    (index.html::ensureConversationFor) ตั้งใจให้ "ทุกแท็บที่เปิดค้างไว้เห็น task จาก
    แท็บอื่นด้วย" — GET /tasks คืน task ทั้งระบบ (ไม่กรองเฉพาะแท็บที่สร้าง) แล้ว
    refreshTasks() เปิด SSE ให้ทุก task ที่ "running" โดยอัตโนมัติทุกแท็บ ทำให้หลายแท็บ
    subscribe task เดียวกันพร้อมกันเป็นเรื่องปกติ ไม่ใช่ edge case หายาก — W25 เดิมแก้ด้วย
    "connection ล่าสุดชนะเสมอ" (สมมติว่ามีผู้ชมสดแค่คนเดียวจริงๆ) กลับกลายเป็นทำให้แท็บที่
    ผู้ใช้กำลังดูอยู่จริงถูกแท็บอื่น (ที่แค่ poll เจอ task นี้ผ่านๆ) แย่ง event ไปเงียบๆ แทน
    (พิสูจน์แล้วจริงจากการทดสอบสด: เปิดแท็บใหม่สร้าง task ทดสอบ แล้ว SSE ของแท็บนั้นหยุดรับ
    event หลังจากข้อความแรก เพราะแท็บอื่นที่เปิดค้างไว้แย่งไปแทน) — แก้ให้ถูกจริงๆ ด้วยการ
    broadcast: ลงทะเบียน Queue ของตัวเองเข้า record.event_subscribers (ทุก push_event()/
    request_approval()/task เสร็จ จะป้อนเข้าทุก Queue ในนี้พร้อมกัน — ดู
    task_manager.py::_broadcast()) แล้วถอนตัวเองออกเสมอตอนจบ (finally — ครอบคลุมทั้ง
    client ปิด connection เอง และ task จบแล้ว break ออกจาก loop ปกติ) กัน list โตค้างไม่รู้
    จบถ้ามีแท็บมาๆ ไปๆ เยอะตลอดอายุ task"""
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
        # W10[B]: replay เฉพาะ approval ที่เคย "ส่งออกไปแล้ว" อย่างน้อยหนึ่งครั้ง
        # (delivered=True) — กรณี tab เดิมหลุดไปกลางคันระหว่างรอ permission prompt (แท็บ
        # นี้เพิ่งสมัครเป็น subscriber ใหม่ ไม่เคยเห็น broadcast รอบก่อนๆ มาก่อนเลย ต้อง
        # replay ให้เห็นว่ามีอะไรค้างรออยู่) ส่วนรายการที่ยังไม่เคยส่งเลย (เพิ่งถูกสร้าง
        # เกือบพร้อมกันกับ connection นี้ ยังไม่ทัน broadcast รอบแรกด้วยซ้ำ) ปล่อยให้ไหล
        # ผ่าน queue drain ปกติด้านล่างแทน ไม่งั้น connection นี้จะเห็น event ซ้ำสองครั้ง
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
        # W26: ถอนตัวเองออกจากรายชื่อ subscriber เสมอ ไม่ว่า loop จะจบแบบไหน (task_done
        # ปกติ, หรือ client ปิด connection กลางคัน — ASGI server จะ cancel generator นี้
        # ซึ่ง finally ยังทำงานตามปกติ) กัน _broadcast() ยังพยายามป้อน event เข้า Queue ที่
        # ไม่มีใครอ่านอีกต่อไปแล้วสะสมไม่มีวันจบตลอดอายุ task ที่รันนาน
        if my_queue in record.event_subscribers:
            record.event_subscribers.remove(my_queue)


@router.get("/tasks/{task_id}/stream")
async def stream_task(task_id: str, request: Request) -> StreamingResponse:
    """W10[B]: Server-Sent Events ของ task นี้ — step log สดๆ ระหว่างรัน +
    approval_request (permission prompt / plan confirmation) + task_done ปิดท้าย — ดู
    _stream_task_events() สำหรับ implementation จริง (ดึงเป็นฟังก์ชันแยกเพื่อให้เทสต์ตรงๆ ได้)"""
    task_manager: TaskManager = request.app.state.task_manager
    record = task_manager.get(task_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"ไม่พบ task_id: {task_id!r}")

    return StreamingResponse(
        _stream_task_events(record),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@router.post("/tasks/{task_id}/stop")
async def stop_task(task_id: str, request: Request) -> dict:
    """W10[C]: ปุ่ม Stop บนหน้าเว็บ — ยกเลิก task ที่กำลังรันอยู่กลางคัน (ไม่ว่าจะกำลังรอ
    LLM ตอบ, กำลัง execute() action, หรือกำลังรอ human ตอบ permission/plan prompt อยู่ก็
    ตาม — ดู TaskManager.cancel()) คืน 409 ถ้า task ไม่ได้ "running" อยู่แล้ว (จบไปแล้ว/
    ถูก stop ไปแล้วก่อนหน้านี้)"""
    task_manager: TaskManager = request.app.state.task_manager
    record = task_manager.get(task_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"ไม่พบ task_id: {task_id!r}")
    # W27: cancel() ตอนนี้เป็น async แล้ว (รอให้ task หยุดใช้ page/browser จริงก่อน return
    # ไม่ใช่แค่ยิง CancelledError แล้วคืนทันที — ดู TaskManager.cancel() docstring) กัน race
    # กับ POST /sessions/{id}/close ที่ frontend (killSession()) ยิงตามมาทันทีหลัง stop ตอบ
    ok = await task_manager.cancel(task_id)
    if not ok:
        raise HTTPException(status_code=409, detail="Task นี้ไม่ได้กำลังรันอยู่แล้ว")
    return {"status": "stopping"}


@router.post("/tasks/{task_id}/respond")
async def respond_task(task_id: str, req: RespondRequest, request: Request) -> dict:
    """W10[B]: ปุ่ม Approve/Deny (permission prompt) หรือ Confirm/Cancel (plan) บนหน้าเว็บ
    ยิงมาที่นี่ — request_id ต้องตรงกับ approval_request event ล่าสุดที่ยังไม่ถูกตอบ
    (ดู TaskManager.resolve_approval()) ไม่งั้นถือว่าหมดอายุ/ตอบไปแล้ว คืน 404"""
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
    """W12: ปุ่ม "New Session" บน Test Console ยิงมาที่นี่ก่อนเริ่มบทสนทนาใหม่ — ปิด
    page/context/browser ที่ session นี้ถืออยู่ (ดู core/session_registry.py::
    SessionRegistry.close() สำหรับรายละเอียดตาม mode) คืน 404 ถ้าไม่พบ session_id นี้
    (ปิดไปแล้ว/ไม่เคยมีอยู่จริง) — ไม่กระทบ task ที่กำลังรันอยู่บน session นี้เลยถ้ามี
    (เป็นหน้าที่ของ frontend ที่จะเช็คก่อนว่าไม่มี task รันค้างอยู่ก่อนเรียก endpoint นี้)

    Security (SEC-4 follow-up): session_owner_token เป็น query param (endpoint นี้ไม่มี
    body เดิมอยู่แล้ว — ตามแบบเดียวกับ ?api_key= ของ SSE endpoint) ไม่ตรงกับของ session
    นี้ -> 403 แทนที่จะปิดให้เงียบๆ (ปิด session เป็น action ทำลายล้าง ไม่ควรให้ใครก็ได้ที่
    รู้แค่ session_id ปิดของคนอื่นทิ้งได้)

    pdf/xlsx: เคลียร์ app.state.file_chat_memory ของ session_id นี้ด้วยเสมอ (ถ้ามี) —
    session ที่เป็น file-chat ล้วนๆ (ไม่เคยแนบ url/เปิด browser เลย) ไม่มีอยู่ใน
    session_registry เลย ไม่งั้นปุ่ม "New Session" จะโดน 404 ทั้งที่จริงมี state ให้เคลียร์"""
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
    """ไว้ debug/monitor ว่าตอนนี้มี session ไหนถือ browser resource ค้างอยู่บ้าง (มีผลต่อ
    /pool/status ด้วย — session mode="pool" กิน browser จาก pool ไปจนกว่าจะปิดเอง)"""
    session_registry = request.app.state.session_registry
    return [
        SessionStatusResponse(
            session_id=s.session_id, mode=s.mode,
            created_at=s.created_at, last_active_at=s.last_active_at,
        )
        for s in session_registry.list()
    ]


# --- W14: Website Learning & Manual Generation (backend/app/site_learning/) — ระบบแยก
# ต่างหากสมบูรณ์จาก RAG/ChromaDB (backend/app/rag/) เก็บ manual ที่ crawl มาอัตโนมัติ
# เป็นไฟล์ JSON บนดิสก์ ไว้ให้ agent โหลดกลับมาใช้แทนการสำรวจซ้ำทุกครั้ง


@router.get("/api/site-manual/status", response_model=SiteManualStatusResponse)
async def site_manual_status(url: str) -> SiteManualStatusResponse:
    """ขับ banner "เว็บไซต์นี้ยังไม่มีคู่มือ" บน Test Console — เช็คเฉยๆ ไม่สร้าง/แตะ
    อะไรเลย (แค่ os.path.exists() ผ่าน storage.load_manual())"""
    manual = load_manual(extract_domain(url))
    if manual is None:
        return SiteManualStatusResponse(exists=False, version=None)
    return SiteManualStatusResponse(exists=True, version=manual.version)


@router.post("/api/site-manual/learn", response_model=LearnCreatedResponse, status_code=202)
@limiter.limit("5/minute")
async def learn_site(req: LearnSiteRequest, request: Request) -> LearnCreatedResponse:
    """เริ่ม crawl เว็บไซต์ที่ req.url — submit แบบเดียวกับ POST /tasks (202 + learn_id
    ทันที ไม่รอ crawl จบ เพราะเดินหลายหน้าอาจใช้เวลาเป็นนาที)

    W16: เปิด browser ของตัวเองแบบมองเห็นได้ (headless=False) แยกจาก BrowserPool โดย
    เจตนา — pool ตัวหลักถูก launch ไว้ล่วงหน้าตอน startup ด้วย headless mode ค่าเดียว
    (settings.browser_headless ปกติ True สำหรับ task ทั่วไปที่รันเงียบๆ) แต่ user อยาก
    "เห็น" ว่า crawler กำลังเดินอยู่หน้าไหนสดๆ ระหว่างเรียนรู้เว็บไซต์ — จึงเปิด Chromium
    process แยกเฉพาะ job นี้ ปิดเองเมื่อ crawl จบ (ไม่ยืม/ไม่คืน pool เลย ไม่กระทบ task
    อื่นที่ใช้ pool พร้อมกัน)"""
    learn_manager: LearnManager = request.app.state.learn_manager
    learn_id = learn_manager.new_learn_id()

    async def _on_progress(event: dict) -> None:
        await learn_manager.push_event(learn_id, event)

    # W23: crawler.py เจอหน้า login ระหว่างทางที่ยังไม่มี username/password ให้เลย —
    # หยุดรอถามคนจริงผ่าน SSE (credentials_needed event) + POST .../credentials แทน
    # การบังคับให้กรอกไว้ล่วงหน้าก่อน crawl เหมือนเดิม เก็บ credential ที่ได้ทันทีที่ user
    # ตอบ (ไม่รอให้ crawl จบก่อน — ถ้า crawl ถูก stop กลางคันหลังจากนี้ credential ที่กรอก
    # ไปแล้วก็ยังไม่หายไปเปล่าๆ)
    #
    # domain ที่ได้ต้องตรงกับ extract_domain(req.url) เท่านั้น (การันตีว่าไม่มีทางบันทึก
    # credential ผิดเว็บ ข้าม site_manuals กันได้ — ป้องกันสองชั้น: ชั้นแรกคือ crawl_site()
    # เอง generate domain นี้จาก extract_domain(start_url) ตรงๆ ไม่มีทางเป็นโดเมนอื่นอยู่
    # แล้ว เพราะกรอง nav link/ปุ่มที่พาออกนอกโดเมนทิ้งไปหมดตั้งแต่ต้น ชั้นที่สองคือ assert
    # ตรงนี้ กันไว้เผื่อ crawler เปลี่ยนพฤติกรรมในอนาคตแล้วลืมรักษา invariant นี้)
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

    # W18: ถ้าผู้ใช้เลือก "ใช้บัญชีที่บันทึกไว้" บน UI แทนการกรอกใหม่ (req.username/password
    # จะเป็น None ทั้งคู่ในเคสนี้ — frontend ไม่มีทางส่งรหัสผ่านจริงที่ backend เก็บไว้แล้ว
    # กลับมาซ้ำได้อยู่แล้ว) โหลด credential ที่เก็บไว้ของโดเมนนี้มาใช้ login bootstrap แทน
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
        # W17: ถ้ามี username/password ที่ใช้ login bootstrap จริง (ไม่ว่าจะกรอกใหม่หรือ
        # ดึงจาก credential เดิม) เขียนทับ credentials.json ของโดเมนนี้ไว้เหมือนเดิม — เขียน
        # ทับค่าเดิมด้วยค่าเดิมก็ไม่มีผลเสียอะไร (idempotent)
        if resolved_username and resolved_password:
            save_credentials(manual.website, resolved_username, resolved_password)
        # W24: ส่ง errors_count กลับไปด้วย — ให้ frontend โชว์ว่า crawl จบพร้อมปัญหาที่เจอ
        # ระหว่างทางกี่รายการ (ดู manual.errors — SiteManual.to_dict() มีรายละเอียดเต็ม
        # อยู่แล้วถ้าต้องการขุดดูทีหลัง ไม่ส่งรายละเอียดเต็มมาที่นี่เพราะ event นี้แค่สรุปผล)
        # W26: ส่ง summary (ภาพรวม "เว็บไซต์นี้ทำอะไรได้บ้าง" จาก describe_site()) กลับไปด้วย
        # ให้ frontend โชว์ทันทีที่เรียนรู้เสร็จ (ดู index.html::renderManualBanner)
        return {
            "version": version, "pages_found": len(manual.pages),
            "errors_count": len(manual.errors), "summary": manual.summary,
        }

    record = learn_manager.submit(learn_id, req.url, _run())
    return LearnCreatedResponse(learn_id=record.learn_id, status=record.status)


@router.get("/api/site-manual/learn/{learn_id}/stream")
async def stream_learn(learn_id: str, request: Request) -> StreamingResponse:
    """SSE ของ crawl job นี้ — "page_done" ทีละหน้าที่ crawl ผ่าน + "learn_done" ปิดท้าย
    (เหมือน stream_task() เกือบทุกประการแค่ event kind ต่างกัน) W23: เพิ่ม
    "credentials_needed"/"credentials_timeout" เข้ามาแล้ว — human-in-the-loop จริงๆ
    เหมือน approval_request ของ task ปกติ (ดู POST .../credentials ด้านล่าง)"""
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

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@router.post("/api/site-manual/learn/{learn_id}/credentials", status_code=204)
async def respond_learn_credentials(
    learn_id: str, req: LearnCredentialsRequest, request: Request,
) -> None:
    """W23: ตอบ "credentials_needed" event (ดู stream_learn() ด้านบน) — ปลดล็อก
    crawl_site() ที่กำลังรอ (await) อยู่ใน on_credentials_needed callback ของ
    learn_site() ให้ไปต่อได้ ปล่อย username/password ว่างทั้งคู่ = ผู้ใช้เลือกข้าม
    ("ไม่ต้อง login") คืน 404 ถ้า request_id ไม่ตรง/หมดอายุ/ตอบไปแล้ว (ดู
    LearnManager.resolve_credentials())"""
    learn_manager: LearnManager = request.app.state.learn_manager
    ok = learn_manager.resolve_credentials(learn_id, req.request_id, req.username, req.password)
    if not ok:
        raise HTTPException(status_code=404, detail=f"ไม่พบ request_id: {req.request_id!r} (หมดอายุ/ตอบไปแล้ว)")


@router.post("/api/site-manual/{domain}/relearn-page", response_model=RelearnPageResponse)
async def relearn_page(domain: str, req: RelearnPageRequest, request: Request) -> RelearnPageResponse:
    """Selector-repair (สเปค: "หาก Selector ใช้งานไม่ได้ ให้สำรวจเฉพาะหน้านั้น อัปเดต
    Version ไม่ต้องสร้าง Manual ใหม่ทั้งหมด") — สำรวจแค่หน้าเดียว (req.url) ใหม่ แทนที่จะ
    crawl ทั้งเว็บซ้ำ ต้องมี manual ของโดเมนนี้อยู่แล้วจาก POST /api/site-manual/learn
    มาก่อน (ไม่งั้นคืน 404 — ยังไม่รู้จักโครงสร้างเว็บนี้เลยสักหน้า จะ "ซ่อม" หน้าเดียว
    ไม่ได้) เป็น request แบบ sync ตรงๆ (ไม่ผ่าน LearnManager) เพราะสำรวจแค่หน้าเดียว เร็ว
    พอที่จะรอผลตรงๆ ได้ ไม่ต้อง submit-then-poll เหมือน crawl เต็มเว็บ"""
    if not manual_exists(domain):
        raise HTTPException(
            status_code=404,
            detail=f"ยังไม่มี manual ของ {domain!r} — ต้องเรียนรู้เว็บไซต์ทั้งหมดก่อน (POST /api/site-manual/learn)",
        )
    pool = request.app.state.browser_pool
    resolved_provider = req.provider or settings.llm_provider
    client, model, _, _, _ = Orchestrator._llm_backend(resolved_provider)

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
    """W17: บันทึก/แก้ไข username-password ของโดเมนนี้ตรงๆ โดยไม่ต้อง crawl ทั้งเว็บใหม่
    (ต่างจาก POST /api/site-manual/learn ที่บันทึกให้อัตโนมัติเป็นผลพลอยได้จาก login
    bootstrap) — ไม่คืนค่า credential กลับเลย (204 เปล่าๆ) กันหลุดไปอยู่ใน response
    log/network tab โดยไม่จำเป็น

    normalize_domain() เสมอก่อนเก็บ — endpoint นี้รับ domain เป็น path param ตรงๆ (ไม่ผ่าน
    extract_domain(url) เหมือนจุดอื่น) ถ้า caller ส่งมาไม่ normalize เอง (เช่นมี "www."
    นำหน้า/ตัวพิมพ์ใหญ่ปน) จะได้ key คนละตัวกับที่ core/orchestrator.py::_maybe_auto_login
    ใช้ extract_domain(page.url) ค้นหาตอนรัน task จริง ทำให้หา credential ไม่เจอทั้งที่
    บันทึกไว้แล้ว"""
    save_credentials(normalize_domain(domain), req.username, req.password)


@router.get("/api/site-manual/{domain}/credentials/status", response_model=CredentialsStatusResponse)
async def site_credentials_status(domain: str) -> CredentialsStatusResponse:
    """เช็คว่ามี credential เก็บไว้ให้โดเมนนี้ไหม — ไม่คืนค่า username/password จริงกลับมา
    เลย (แค่ exists: bool) กันไม่ให้ frontend/log ที่ไหนโชว์รหัสผ่านที่เก็บไว้แล้วออกมาซ้ำ"""
    return CredentialsStatusResponse(exists=credentials_exist(normalize_domain(domain)))


@router.delete("/api/site-manual/{domain}/credentials", status_code=204)
async def delete_site_credentials(domain: str) -> None:
    """ลบ credential ที่เก็บไว้ของโดเมนนี้ทิ้ง — ไม่ error ถ้าไม่มีอยู่แล้ว (idempotent)"""
    delete_credentials(normalize_domain(domain))


# --- W_openai_oauth: "Sign in with ChatGPT" สำหรับ provider "openai" (ดู core/openai_oauth.py
# หัวไฟล์สำหรับ risk disclosure เต็ม) — routes เหล่านี้อยู่หลัง verify_api_key เดียวกับ route
# อื่นทั้งหมดในไฟล์นี้ (router-level dependency ด้านบน) ไม่มี auth layer แยกเพิ่ม เพราะระบบนี้
# เป็น single-tenant (operator คนเดียว, credential เดียวต่อ deployment เหมือน API key เดิม
# ไม่ใช่ per-user account) ---

@router.post("/api/auth/openai/login/start", response_model=OpenAILoginStartResponse)
async def start_openai_login() -> OpenAILoginStartResponse:
    """เริ่ม OAuth flow — เปิด loopback listener ก่อนเสมอแล้วคืน authorize_url ให้ frontend
    เปิดในแท็บ/หน้าต่างใหม่ให้ user login เอง token exchange เกิดขึ้น "หลังบ้าน" ใน
    background task ที่ผูกกับ loopback callback (ดู core/openai_oauth.py::start_login_flow())
    ไม่ใช่ response ของ endpoint นี้ — frontend ต้อง poll GET .../login/status ต่อจนกว่าจะ
    "linked" หรือ "error" """
    result = await openai_oauth.start_login_flow()
    return OpenAILoginStartResponse(authorize_url=result["authorize_url"], login_id=result["login_id"])


@router.get("/api/auth/openai/login/status", response_model=OpenAILoginStatusResponse)
async def openai_login_status(login_id: str = Query(...)) -> OpenAILoginStatusResponse:
    """poll สถานะของ login attempt ที่ระบุ (login_id จาก POST .../login/start) —
    status: "pending"|"linked"|"error" — login_id ที่ไม่รู้จัก (หมดอายุ process restart ไป
    แล้ว หรือพิมพ์ผิด) คืน 404 ตรงๆ ไม่ใช่ "pending" ค้าง เพื่อไม่ให้ frontend poll ทิ้งไว้
    ไม่รู้จบโดยไม่มีทางรู้ว่าจริงๆ แล้วไม่มี attempt นี้อยู่เลย"""
    status = openai_oauth.get_login_status(login_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"ไม่พบ login attempt {login_id!r} (อาจหมดอายุ/server restart ไปแล้ว)")
    return OpenAILoginStatusResponse(**status)


@router.get("/api/auth/openai/status", response_model=OpenAIAuthStatusResponse)
async def openai_auth_status() -> OpenAIAuthStatusResponse:
    """สถานะ link ปัจจุบัน (ไม่ผูกกับ login attempt ไหนเป็นพิเศษ) — ดึงจาก token store
    ตรงๆ (email/plan_type ที่ decode ไว้ตอน save, ไม่ decrypt access/refresh token มาโชว์
    เลย) ใช้ตอน frontend โหลดหน้าเพื่อรู้ว่าจะ enable "openai" ใน provider dropdown ได้ไหม"""
    status = openai_oauth.get_link_status()
    return OpenAIAuthStatusResponse(**status)


@router.post("/api/auth/openai/logout", status_code=204)
async def openai_logout() -> None:
    """ลบ token ในเครื่อง + best-effort revoke ที่ OpenAI (ดู
    core/openai_oauth.py::revoke_token()) — ไม่ throw ถ้า revoke ทาง network ล้มเหลว (ลบ
    local เสมอไม่ว่า network จะสำเร็จไหม)"""
    await openai_oauth.revoke_token()

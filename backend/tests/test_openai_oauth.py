"""W_openai_oauth: เทสต์ core/openai_oauth.py (PKCE, Fernet round-trip, refresh cadence, JWT
claims decode) + provider dispatch wiring (_llm_backend) + route contract ของ
/api/auth/openai/* — ไม่มีเทสต์ไหนยิง network จริงไปที่ auth.openai.com เลย (mock
openai_oauth.refresh_tokens ตรงๆ แทนการ mock httpx ภายใน เพราะเทสต์แค่ cadence decision
logic ไม่ใช่ HTTP call ตัวเอง) — end-to-end login จริงต้องทำมือโดย user เท่านั้น (ดู
core/openai_oauth.py หัวไฟล์ + implementation plan สำหรับเหตุผล)"""

import base64
import hashlib
import json
import time
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.app.config import settings
from backend.app.core import llm, openai_oauth
from backend.app.core.orchestrator import Orchestrator


@pytest.fixture(autouse=True)
def _isolated_data_dir(tmp_path, monkeypatch):
    """ทุกเทสต์ในไฟล์นี้เขียน/อ่าน token file + credential key ลงดิสก์จริง — isolate ด้วย
    tmp_path เหมือน pattern เดียวกับ test_site_learning_storage.py กัน settings.
    site_manuals_dir ตัวจริงบนเครื่อง dev โดนเขียนทับ/ปนกับข้อมูลเทสต์

    สำคัญ: ตั้งเป็น tmp_path/"site_manuals" (subdirectory) ไม่ใช่ tmp_path ตรงๆ — เพราะ
    crypto_store.credential_key_path()/openai_oauth._token_path() ทั้งคู่ใช้
    Path(settings.site_manuals_dir).parent (เก็บ key/token ที่ระดับ "data/" เดียว ไม่ใช่ต่อ
    โดเมน ตามที่ตั้งใจไว้ในโค้ดจริง) ถ้าตั้ง site_manuals_dir = tmp_path ตรงๆ, .parent ของมัน
    จะกลายเป็น pytest tmp base dir ที่ "ใช้ร่วมกันข้ามทุกเทสต์ในไฟล์นี้" (pytest การันตีว่า
    tmp_path เองไม่ซ้ำกันต่อเทสต์ แต่ parent ของมันไม่ใช่) ทำให้ token file รั่วข้ามเทสต์ —
    เจอบั๊กนี้จริงตอนรันเทสต์ครั้งแรก (2 เทสต์ fail เพราะเห็น token จากเทสต์ก่อนหน้าที่ไม่
    เกี่ยวกัน) แก้ด้วยการใส่ subdirectory เพิ่มอีกชั้นให้ .parent ก็ยังอยู่ใต้ tmp_path ที่
    unique ต่อเทสต์เสมอ"""
    monkeypatch.setattr(settings, "site_manuals_dir", str(tmp_path / "site_manuals"))
    openai_oauth._pending_logins.clear()
    yield
    openai_oauth._pending_logins.clear()


# --- PKCE ---

def test_generate_pkce_pair_produces_valid_s256_challenge():
    code_verifier, code_challenge = openai_oauth.generate_pkce_pair()

    assert 43 <= len(code_verifier) <= 128
    expected_digest = hashlib.sha256(code_verifier.encode("ascii")).digest()
    expected_challenge = base64.urlsafe_b64encode(expected_digest).rstrip(b"=").decode("ascii")
    assert code_challenge == expected_challenge


def test_generate_pkce_pair_is_random_across_calls():
    pair1 = openai_oauth.generate_pkce_pair()
    pair2 = openai_oauth.generate_pkce_pair()
    assert pair1 != pair2


def test_generate_state_is_random_across_calls():
    assert openai_oauth.generate_state() != openai_oauth.generate_state()


# --- Fernet round-trip (token store) ---

def test_load_token_returns_none_when_never_saved():
    assert openai_oauth.load_token() is None
    assert openai_oauth.token_exists() is False


def _sample_tokens() -> dict:
    return {
        "id_token": "fake-id-token",
        "access_token": "fake-access-token",
        "refresh_token": "fake-refresh-token",
        "account_id": "acct-123",
        "email": "user@example.com",
        "plan_type": "plus",
        "exp": time.time() + 3600,
        "last_refresh": time.time(),
    }


def test_save_and_load_token_round_trips():
    tokens = _sample_tokens()
    openai_oauth.save_token(tokens)

    loaded = openai_oauth.load_token()
    assert loaded is not None
    assert loaded["access_token"] == tokens["access_token"]
    assert loaded["refresh_token"] == tokens["refresh_token"]
    assert loaded["id_token"] == tokens["id_token"]
    assert loaded["account_id"] == "acct-123"
    assert loaded["email"] == "user@example.com"
    assert loaded["plan_type"] == "plus"
    assert openai_oauth.token_exists() is True


def test_saved_token_file_does_not_contain_plaintext_secrets():
    """Security: access_token/refresh_token ต้องเข้ารหัสจริงบนดิสก์ ไม่ใช่ plaintext JSON"""
    tokens = _sample_tokens()
    openai_oauth.save_token(tokens)

    raw = openai_oauth._token_path().read_text(encoding="utf-8")
    assert tokens["access_token"] not in raw
    assert tokens["refresh_token"] not in raw


def test_load_token_returns_none_on_corrupt_file():
    openai_oauth.save_token(_sample_tokens())
    openai_oauth._token_path().write_text("not valid json{{{", encoding="utf-8")

    assert openai_oauth.load_token() is None


def test_delete_token_removes_file_and_is_idempotent():
    openai_oauth.save_token(_sample_tokens())
    assert openai_oauth.delete_token() is True
    assert openai_oauth.token_exists() is False
    # ลบซ้ำครั้งที่สอง (ไฟล์ไม่มีอยู่แล้ว) ต้องไม่ throw — logout ต้อง idempotent
    assert openai_oauth.delete_token() is False


# --- JWT claims decode (ไม่ verify signature — ดู decode_id_token_claims docstring) ---

def _build_unsigned_jwt(claims: dict) -> str:
    def _b64url(data: bytes) -> str:
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")

    header = _b64url(json.dumps({"alg": "none", "typ": "JWT"}).encode("utf-8"))
    payload = _b64url(json.dumps(claims).encode("utf-8"))
    return f"{header}.{payload}.fake-signature"


def test_decode_id_token_claims_extracts_known_fields():
    claims = {
        "email": "user@example.com",
        "chatgpt_plan_type": "pro",
        "chatgpt_account_id": "acct-456",
        "exp": 1234567890,
    }
    token = _build_unsigned_jwt(claims)

    decoded = openai_oauth.decode_id_token_claims(token)
    assert decoded == claims


def test_decode_id_token_claims_handles_base64_padding_edge_cases():
    # claims ที่ยาวไม่ลงตัวพอดี 4 ตัวอักษร base64 (ต้องเติม "=" padding เอง) — ลองหลายความยาว
    for filler_len in range(0, 6):
        claims = {"x": "a" * filler_len}
        token = _build_unsigned_jwt(claims)
        assert openai_oauth.decode_id_token_claims(token) == claims


def test_decode_id_token_claims_rejects_malformed_jwt():
    with pytest.raises(ValueError):
        openai_oauth.decode_id_token_claims("not-a-jwt")


# --- W_openai_oauth follow-up fix (2026-08-17c): claims namespace extraction. Regression
# coverage for a bug caught live by the user after a real successful login — chatgpt_account_id/
# chatgpt_plan_type are NOT top-level id_token claims, they're nested under the
# "https://api.openai.com/auth" namespaced claim key (confirmed against a real decoded token,
# not guessed). _build_real_shaped_jwt() below mirrors that exact real shape.

def _build_real_shaped_jwt(*, account_id: str = "acct-real-123", plan_type: str = "plus",
                            email: str = "user@example.com", exp: float | None = None) -> str:
    return _build_unsigned_jwt({
        "email": email,
        "https://api.openai.com/auth": {
            "chatgpt_account_id": account_id,
            "chatgpt_plan_type": plan_type,
            "chatgpt_user_id": "user-xyz",
        },
        "exp": exp if exp is not None else time.time() + 3600,
    })


def test_extract_openai_auth_claims_reads_nested_namespace():
    claims = openai_oauth.decode_id_token_claims(_build_real_shaped_jwt(account_id="acct-real-123"))
    auth_claims = openai_oauth._extract_openai_auth_claims(claims)
    assert auth_claims["chatgpt_account_id"] == "acct-real-123"
    assert auth_claims["chatgpt_plan_type"] == "plus"


def test_extract_openai_auth_claims_returns_empty_dict_when_namespace_missing():
    """ไม่ throw ถ้า token ไม่มี namespace นี้เลย (shape เปลี่ยนไปในอนาคต) — คืน {} เฉยๆ"""
    assert openai_oauth._extract_openai_auth_claims({"email": "x@example.com"}) == {}


def test_extract_openai_auth_claims_ignores_flat_top_level_chatgpt_account_id():
    """Regression: ต้องไม่หลงไปอ่าน top-level "chatgpt_account_id" (shape เก่าที่เข้าใจผิด
    ตอนวางแผนครั้งแรก ก่อนยืนยันกับ token จริง) — namespace ต้องมาจาก key ที่ถูกต้องเท่านั้น"""
    claims = {"chatgpt_account_id": "wrong-flat-value", "https://api.openai.com/auth": {}}
    assert openai_oauth._extract_openai_auth_claims(claims).get("chatgpt_account_id") is None


@pytest.mark.asyncio
async def test_finish_login_background_extracts_account_id_from_real_shaped_token(monkeypatch):
    """End-to-end regression test for the exact bug the user hit live: after a successful
    login, the saved token must have a real (non-None) account_id — this is what
    get_valid_access_token() checks before returning, and what broke in production.

    fake_server is a plain MagicMock — _wait_for_callback()'s real implementation
    (server.timeout=..., server.handle_request(), server.server_close()) runs for real against
    it via asyncio.to_thread() (harmless/non-blocking on a MagicMock); `result` is
    pre-populated with a successful code/state below since the fake server never actually
    receives an HTTP callback."""
    fake_server = MagicMock()
    monkeypatch.setattr(
        openai_oauth, "exchange_code_for_tokens",
        AsyncMock(return_value={
            "id_token": _build_real_shaped_jwt(account_id="acct-real-123", plan_type="pro"),
            "access_token": "real-access-token",
            "refresh_token": "real-refresh-token",
        }),
    )
    result = {"code": "auth-code-123", "state": "expected-state", "error": None}

    await openai_oauth._finish_login_background(
        "login-1", fake_server, result, "verifier", "expected-state", "http://localhost:1455/auth/callback",
    )

    status = openai_oauth.get_login_status("login-1")
    assert status["status"] == "linked"

    saved = openai_oauth.load_token()
    assert saved is not None
    assert saved["account_id"] == "acct-real-123"
    assert saved["plan_type"] == "pro"

    # ที่มาเป็นสาเหตุจริงของ bug: get_valid_access_token() ต้องไม่ raise OAuthLoginRequired อีก
    access_token, account_id = await openai_oauth.get_valid_access_token()
    assert access_token == "real-access-token"
    assert account_id == "acct-real-123"


# --- refresh cadence (_refresh_if_needed) ---

@pytest.mark.asyncio
async def test_refresh_if_needed_skips_refresh_when_token_is_fresh(monkeypatch):
    mock_refresh = AsyncMock()
    monkeypatch.setattr(openai_oauth, "refresh_tokens", mock_refresh)

    fresh_tokens = _sample_tokens()  # exp = now + 3600s, last_refresh = now — ไม่ใกล้หมดอายุ/ไม่เก่า
    result = await openai_oauth._refresh_if_needed(fresh_tokens)

    mock_refresh.assert_not_called()
    assert result is fresh_tokens


@pytest.mark.asyncio
async def test_refresh_if_needed_refreshes_when_near_expiry(monkeypatch):
    mock_refresh = AsyncMock(return_value={
        "id_token": _build_unsigned_jwt({"chatgpt_account_id": "acct-1", "exp": time.time() + 3600}),
        "access_token": "new-access-token",
        "refresh_token": "new-refresh-token",
    })
    monkeypatch.setattr(openai_oauth, "refresh_tokens", mock_refresh)

    tokens = _sample_tokens()
    tokens["exp"] = time.time() + 60  # ภายใน openai_oauth_refresh_before_expiry_seconds (300s default)

    result = await openai_oauth._refresh_if_needed(tokens)

    mock_refresh.assert_awaited_once_with(tokens["refresh_token"])
    assert result["access_token"] == "new-access-token"
    assert result["refresh_token"] == "new-refresh-token"


@pytest.mark.asyncio
async def test_refresh_if_needed_refreshes_when_last_refresh_too_old(monkeypatch):
    mock_refresh = AsyncMock(return_value={
        "id_token": _build_unsigned_jwt({"chatgpt_account_id": "acct-1", "exp": time.time() + 3600}),
        "access_token": "new-access-token",
        "refresh_token": "new-refresh-token",
    })
    monkeypatch.setattr(openai_oauth, "refresh_tokens", mock_refresh)

    tokens = _sample_tokens()
    tokens["exp"] = time.time() + 3600  # ไม่ใกล้หมดอายุ
    tokens["last_refresh"] = time.time() - (9 * 86400)  # แต่เก่าเกิน 8 วัน (default)

    result = await openai_oauth._refresh_if_needed(tokens)

    mock_refresh.assert_awaited_once()
    assert result["access_token"] == "new-access-token"


@pytest.mark.asyncio
async def test_refresh_if_needed_preserves_old_refresh_token_when_not_rotated(monkeypatch):
    """refresh_token อาจไม่ rotate ทุกครั้ง — response ที่ไม่มี refresh_token ต้องไม่ทำให้
    ตัวเดิมหายไป (ดู _refresh_if_needed docstring)"""
    mock_refresh = AsyncMock(return_value={
        "id_token": _build_unsigned_jwt({"chatgpt_account_id": "acct-1", "exp": time.time() + 3600}),
        "access_token": "new-access-token",
        # ไม่มี refresh_token ใน response รอบนี้
    })
    monkeypatch.setattr(openai_oauth, "refresh_tokens", mock_refresh)

    tokens = _sample_tokens()
    tokens["exp"] = time.time() + 60

    result = await openai_oauth._refresh_if_needed(tokens)
    assert result["refresh_token"] == tokens["refresh_token"]


@pytest.mark.asyncio
async def test_refresh_if_needed_raises_login_required_when_refresh_fails(monkeypatch):
    mock_refresh = AsyncMock(side_effect=RuntimeError("network error"))
    monkeypatch.setattr(openai_oauth, "refresh_tokens", mock_refresh)

    tokens = _sample_tokens()
    tokens["exp"] = time.time() + 60

    with pytest.raises(openai_oauth.OAuthLoginRequired):
        await openai_oauth._refresh_if_needed(tokens)


@pytest.mark.asyncio
async def test_get_valid_access_token_raises_when_never_logged_in():
    with pytest.raises(openai_oauth.OAuthLoginRequired):
        await openai_oauth.get_valid_access_token()


@pytest.mark.asyncio
async def test_get_valid_access_token_returns_token_and_account_id_when_fresh():
    openai_oauth.save_token(_sample_tokens())

    access_token, account_id = await openai_oauth.get_valid_access_token()
    assert access_token == "fake-access-token"
    assert account_id == "acct-123"


# --- provider dispatch wiring ---

def test_llm_backend_openai_returns_expected_five_tuple():
    """W_openai_oauth: _llm_backend() คืนรูปแบบเดียวกันหมดทุก provider ให้ agent loop เรียก
    โดยไม่ต้องรู้ว่าเป็น provider ไหน — "openai" ต้องเข้าชุดเดียวกันเป๊ะ ไม่ใช่ทางแยกพิเศษ

    หมายเหตุ (เทียบกับเวอร์ชันเดิมของเทสต์นี้): เดิมส่ง tier="standard" เข้าไปด้วย แต่
    model-tier routing (_TIER_MODEL_SETTINGS/_llm_backend(tier=)) เป็นงานคนละก้อนที่ยังไม่มี
    ในโค้ดเวอร์ชันนี้ — provider "openai" ใช้ settings.openai_model ตัวเดียวเหมือน provider
    อื่นทุกตัวในไฟล์นี้ ถ้าวันหลังเพิ่ม tier routing กลับเข้ามา ค่อยเพิ่มเทสต์ tier variants
    (lite/standard/advanced resolve คนละ model) แยกอีกเคสตอนนั้น"""
    client, model, next_action, append_tool_result, compact_messages = Orchestrator._llm_backend("openai")
    assert client is not None
    assert model == settings.openai_model
    assert callable(next_action)
    assert callable(append_tool_result)
    assert callable(compact_messages)


# --- generate_text() openai branch (W_openai_oauth follow-up fix 2026-08-17b: generate_text()
# is a SECOND, independent provider dispatch point from _llm_backend()/next_action_openai() —
# used by generate_plan()/classify_intent(), missed in the original implementation, caused
# "Generate plan" to fail with a stale ValueError even after a successful OAuth login) ---

class _FakeResponseEvent:
    def __init__(self, event_type: str, response=None, message: str = "", delta=None, item=None):
        self.type = event_type
        self.response = response
        self.message = message
        self.delta = delta
        self.item = item


class _FakeResponse:
    """W_openai_oauth (follow-up fix 2026-08-17i, ยืนยันจริงจาก live call): endpoint จริงคืน
    response.output/output_text ว่างเปล่าเสมอใน response.completed event แม้สร้าง output จริง
    แล้วก็ตาม — output_text ที่นี่จงใจปล่อยว่าง (ไม่ใช่ text จริง) ให้ตรงกับพฤติกรรมจริงที่สังเกต
    เจอ — ข้อความจริงต้องมาจาก response.output_text.delta event ต่างหาก (ดูเทสต์ด้านล่าง)"""
    def __init__(self, usage=None):
        self.output_text = ""
        self.output = []
        self.usage = usage


class _FakeStream:
    def __init__(self, events: list):
        self._events = events

    def __aiter__(self):
        return self._iter()

    async def _iter(self):
        for event in self._events:
            yield event


@pytest.mark.asyncio
async def test_generate_text_openai_branch_returns_output_text(monkeypatch):
    """W_openai_oauth (follow-up fix 2026-08-17i, ยืนยันจริงจาก live call): endpoint จริงคืน
    final_response.output_text ว่างเปล่าเสมอ (แม้ token/ข้อความจริงถูกสร้างแล้วก็ตาม) — ต้อง
    ประกอบ text จาก "response.output_text.delta" event ระหว่าง stream แทน — fake stream นี้
    จำลอง event ตามลำดับจริงที่สังเกตเจอ (หลาย delta event ตามด้วย completed ที่ output_text
    ว่างเปล่า) เพื่อยืนยันว่า generate_text() ไม่ได้พึ่ง final_response.output_text อีกต่อไป"""
    monkeypatch.setattr(
        openai_oauth, "get_valid_access_token", AsyncMock(return_value=("fake-access-token", "acct-123")),
    )
    fake_response = _FakeResponse()
    fake_stream = _FakeStream([
        _FakeResponseEvent("response.output_text.delta", delta="1. Do X\n"),
        _FakeResponseEvent("response.output_text.delta", delta="2. Do Y"),
        _FakeResponseEvent("response.completed", response=fake_response),
    ])
    client = MagicMock()
    client.responses.create = AsyncMock(return_value=fake_stream)

    result = await llm.generate_text(client, "gpt-5.1-codex", "some prompt", "openai")

    assert result == "1. Do X\n2. Do Y"
    client.responses.create.assert_awaited_once()
    call_kwargs = client.responses.create.await_args.kwargs
    assert call_kwargs["extra_headers"]["Authorization"] == "Bearer fake-access-token"
    assert call_kwargs["extra_headers"]["chatgpt-account-id"] == "acct-123"
    # Regression test (2026-08-17g, real error hit live): endpoint rejects a bare string
    # input with 400 "Input must be a list" — must always be a list of input items
    assert call_kwargs["input"] == [{"role": "user", "content": "some prompt"}]


@pytest.mark.asyncio
async def test_generate_text_openai_branch_raises_on_response_failed(monkeypatch):
    monkeypatch.setattr(
        openai_oauth, "get_valid_access_token", AsyncMock(return_value=("fake-access-token", "acct-123")),
    )
    failed_response = MagicMock(error="something broke")
    fake_stream = _FakeStream([_FakeResponseEvent("response.failed", response=failed_response)])
    client = MagicMock()
    client.responses.create = AsyncMock(return_value=fake_stream)

    with pytest.raises(RuntimeError):
        await llm.generate_text(client, "gpt-5.1-codex", "some prompt", "openai")


@pytest.mark.asyncio
async def test_generate_text_openai_branch_propagates_login_required(monkeypatch):
    monkeypatch.setattr(
        openai_oauth, "get_valid_access_token",
        AsyncMock(side_effect=openai_oauth.OAuthLoginRequired("not logged in")),
    )
    client = MagicMock()

    with pytest.raises(openai_oauth.OAuthLoginRequired):
        await llm.generate_text(client, "gpt-5.1-codex", "some prompt", "openai")


# --- chat_response()/answer_file_query()/answer_image_query() openai branches
# (follow-up fix 2026-08-17j: real bug reported live — "Sorry, the system doesn't recognize
# this provider" — a THIRD independent provider-dispatch point, missed by both earlier fixes,
# never had an "openai" branch at all in any of these 3 functions) ---

def _mock_openai_stream_client(monkeypatch, text: str):
    monkeypatch.setattr(
        openai_oauth, "get_valid_access_token", AsyncMock(return_value=("fake-access-token", "acct-123")),
    )
    fake_stream = _FakeStream([
        _FakeResponseEvent("response.output_text.delta", delta=text),
        _FakeResponseEvent("response.completed", response=_FakeResponse()),
    ])
    client = MagicMock()
    client.responses.create = AsyncMock(return_value=fake_stream)
    return client


@pytest.mark.asyncio
async def test_chat_response_openai_branch_returns_text(monkeypatch):
    client = _mock_openai_stream_client(monkeypatch, "hello there")

    result = await llm.chat_response(client, "gpt-5.4-mini", "hi", "openai")

    assert result == "hello there"
    call_kwargs = client.responses.create.await_args.kwargs
    assert isinstance(call_kwargs["input"], list)
    assert call_kwargs["store"] is False


@pytest.mark.asyncio
async def test_answer_file_query_openai_branch_returns_text(monkeypatch):
    client = _mock_openai_stream_client(monkeypatch, "the value is 42")

    result = await llm.answer_file_query(client, "gpt-5.4-mini", "what is x?", "x: 42", "notes.txt", "openai")

    assert result == "the value is 42"
    call_kwargs = client.responses.create.await_args.kwargs
    assert isinstance(call_kwargs["input"], list)
    assert call_kwargs["store"] is False


@pytest.mark.asyncio
async def test_answer_image_query_openai_branch_sends_input_image_content_part(monkeypatch):
    client = _mock_openai_stream_client(monkeypatch, "red")

    result = await llm.answer_image_query(client, "gpt-5.4-mini", "what color?", b"fake-png-bytes", "x.png", "openai")

    assert result == "red"
    call_kwargs = client.responses.create.await_args.kwargs
    content = call_kwargs["input"][0]["content"]
    types = [part["type"] for part in content]
    assert "input_text" in types
    assert "input_image" in types
    image_part = next(p for p in content if p["type"] == "input_image")
    assert image_part["image_url"].startswith("data:image/png;base64,")
    assert image_part["detail"] == "auto"


@pytest.mark.asyncio
async def test_chat_response_openai_branch_returns_friendly_message_on_error(monkeypatch):
    """chat_response()/answer_file_query()/answer_image_query() ห้าม throw ออกไปพังเด็ดขาด
    (ดู docstring เดิม) — provider error ต้องถูกจับแล้วคืนข้อความ friendly แทน ไม่ใช่ raise"""
    monkeypatch.setattr(
        openai_oauth, "get_valid_access_token",
        AsyncMock(side_effect=openai_oauth.OAuthLoginRequired("not logged in")),
    )
    client = MagicMock()

    result = await llm.chat_response(client, "gpt-5.4-mini", "hi", "openai")

    # _friendly_llm_error_message() คืนข้อความไทยขอโทษสั้นๆ เสมอ (ดู llm.py) — เช็คแค่ว่า
    # ไม่ throw ออกมาและได้ string ที่ไม่ว่างเปล่ากลับมา ไม่ใช่เนื้อหาเป๊ะๆ (อาจเปลี่ยนคำได้)
    assert isinstance(result, str) and result.strip()


def test_generate_text_unknown_provider_error_message_includes_openai():
    """Regression test for the exact bug reported: the ValueError text must list "openai" as
    a supported provider now, not just anthropic/gemini"""
    import asyncio

    with pytest.raises(ValueError, match="openai"):
        asyncio.run(llm.generate_text(MagicMock(), "model", "prompt", "not-a-real-provider"))


# --- route contracts ---

class _FakeBrowserPool:
    """เหมือน _FakeBrowserPool ใน test_api.py แบบย่อ (แค่พอให้ main.py::lifespan startup/
    shutdown ผ่านไม่ launch Chromium จริง) — route ที่เทสต์ในไฟล์นี้ไม่แตะ browser pool เลย"""

    def __init__(self, size: int = 2, headless: bool | None = None):
        self._size = size

    @property
    def size(self) -> int:
        return self._size

    @property
    def available(self) -> int:
        return self._size

    async def start(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    @asynccontextmanager
    async def acquire(self):
        yield MagicMock()


@pytest.fixture
def client():
    from backend.app.api.routes import limiter as api_limiter
    from backend.app.main import app

    api_limiter.reset()
    with patch("backend.app.main.BrowserPool", _FakeBrowserPool):
        with TestClient(app) as c:
            yield c


def test_openai_login_start_returns_authorize_url_and_login_id(client, monkeypatch):
    async def _fake_start_login_flow():
        return {"authorize_url": "https://auth.openai.com/oauth/authorize?fake=1", "login_id": "abc123"}

    monkeypatch.setattr(openai_oauth, "start_login_flow", _fake_start_login_flow)

    resp = client.post("/api/auth/openai/login/start")
    assert resp.status_code == 200
    body = resp.json()
    assert body["authorize_url"] == "https://auth.openai.com/oauth/authorize?fake=1"
    assert body["login_id"] == "abc123"


def test_openai_login_status_returns_404_for_unknown_login_id(client):
    resp = client.get("/api/auth/openai/login/status", params={"login_id": "does-not-exist"})
    assert resp.status_code == 404


def test_openai_login_status_returns_pending_state(client):
    openai_oauth._pending_logins["xyz"] = {"status": "pending", "error": None, "email": None, "plan_type": None}
    resp = client.get("/api/auth/openai/login/status", params={"login_id": "xyz"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "pending"


def test_openai_auth_status_reports_not_linked_by_default(client):
    resp = client.get("/api/auth/openai/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["linked"] is False


def test_openai_auth_status_reports_linked_after_save_token(client):
    openai_oauth.save_token(_sample_tokens())

    resp = client.get("/api/auth/openai/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["linked"] is True
    assert body["email"] == "user@example.com"


def test_openai_logout_deletes_token_and_returns_204(client, monkeypatch):
    openai_oauth.save_token(_sample_tokens())

    async def _fake_revoke():
        openai_oauth.delete_token()

    monkeypatch.setattr(openai_oauth, "revoke_token", _fake_revoke)

    resp = client.post("/api/auth/openai/logout")
    assert resp.status_code == 204
    assert openai_oauth.token_exists() is False


def test_openai_routes_require_api_key_when_configured(client, monkeypatch):
    monkeypatch.setattr(settings, "api_key", "secret-123")
    resp = client.get("/api/auth/openai/status")
    assert resp.status_code == 401


# --- W_openai_args: provider-quirk normaliser (real bug, live-reproduced on
# opensource-demo.orangehrmlive.com) — the ChatGPT-OAuth endpoint returns EVERY schema
# property on every tool call, with junk defaults, unlike Anthropic/Gemini which send only
# the parameters the chosen action actually uses. Those junk values were dispatched for
# real: then_click_index chained an unrelated click, key="Enter" pressed Enter after a
# fill, and completed_plan_step marked a plan step done on an action that had just failed.
# See llm._normalize_openai_args() for the full report. ---

_LIVE_JUNK_ARGS = {
    "secret": "current_password", "type": "fill_secret", "index": 39, "text": "",
    "label": "", "key": "Enter", "then_click_index": 3, "direction": "down", "url": "",
    "tab_index": 0, "query": "", "target_hint": "", "completed_plan_step": 1,
}


def test_normalize_openai_args_strips_params_irrelevant_to_fill_secret():
    """ชุด argument นี้คัดลอกมาจาก task JSON ของ run จริงที่พังแบบเป๊ะๆ — fill_secret ใช้แค่
    index/secret เท่านั้น then_click_index=3 (ซึ่งชี้ไปที่ลิงก์ 'Admin' บนหน้าจริง) ต้องไม่
    หลงเหลือไปถึง actions.py ให้ chain-click ต่อโดยที่โมเดลไม่ได้ตั้งใจสั่ง"""
    out = llm._normalize_openai_args("browser_action", _LIVE_JUNK_ARGS)

    assert out == {
        "type": "fill_secret", "index": 39, "secret": "current_password",
        "completed_plan_step": 1,
    }
    assert "then_click_index" not in out
    assert "key" not in out


def test_normalize_openai_args_keeps_params_each_action_type_really_uses():
    """ตัด parameter ตาม action type ที่เลือก ไม่ใช่ตัดตาม "ค่าว่างเปล่า" — then_click_index
    เป็น compound action ที่ถูกต้องจริงสำหรับ click/fill/select/check (ดู _BROWSER_ACTION_
    PARAMS) จึงต้องรอดมาในเคสเหล่านั้น"""
    click = llm._normalize_openai_args("browser_action", dict(_LIVE_JUNK_ARGS, type="click"))
    assert click["then_click_index"] == 3
    assert "secret" not in click and "direction" not in click

    scroll = llm._normalize_openai_args("browser_action", dict(_LIVE_JUNK_ARGS, type="scroll"))
    assert scroll == {"type": "scroll", "direction": "down", "completed_plan_step": 1}

    read = llm._normalize_openai_args(
        "browser_action", dict(_LIVE_JUNK_ARGS, type="read_page_data", query="q", target_hint="h"),
    )
    assert read == {
        "type": "read_page_data", "query": "q", "target_hint": "h", "completed_plan_step": 1,
    }


def test_normalize_openai_args_passes_through_other_tools_and_unknown_types():
    """finish_task/request_user_input มีสคีมาเล็กและทุก field มีความหมายจริง — normaliser
    ต้องไม่แตะเลย (กันตัด field จำเป็นทิ้งถ้ามีการเพิ่ม tool ใหม่ในอนาคต) เช่นเดียวกับ action
    type ที่ไม่รู้จัก ต้องปล่อยผ่านให้ actions.py::execute เป็นคนปฏิเสธตามเดิม"""
    finish = {"success": True, "message": "done", "verify_text": "AutoUser_99"}
    assert llm._normalize_openai_args("finish_task", finish) == finish

    unknown = {"type": "brand_new_action", "whatever": 1}
    assert llm._normalize_openai_args("browser_action", unknown) == unknown

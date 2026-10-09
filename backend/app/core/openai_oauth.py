"""core/openai_oauth.py — W_openai_oauth: "Sign in with ChatGPT" สำหรับ provider "openai"

=== risk disclosure (อ่านก่อนแก้ไฟล์นี้) ===
reuse OAuth client_id สาธารณะของ Codex CLI ("app_EMoamEEZ73f0CkXaXp7hrann") นอกขอบเขตที่ตั้งใจ —
ไม่ใช่ integration surface ที่ OpenAI รองรับ third-party อย่างเป็นทางการ (maintainer เลี่ยงตอบเรื่อง ToS
ใน openai/codex Discussion #8338) เสี่ยงจริงที่บัญชี ChatGPT จะถูกจำกัด/แบนถ้าตรวจจับ traffic ที่ไม่ใช่ Codex CLI

ตัดสินใจร่วมกับ user 2026-08-17: ยอมรับความเสี่ยงโดยรู้ตัวเพื่อใช้โควต้า ChatGPT Plus/Pro แทน API credit —
จำกัด single-tenant: operator คนเดียว credential เดียว เข้ารหัสบนดิสก์ของ server (เทียบเท่า API key ใน .env)
**ห้ามขยายเป็น multi-user/hosted SaaS โดยไม่ทบทวนความเสี่ยงนี้ใหม่ก่อน** — ไม่ควรผลักความเสี่ยงแบนให้ user อื่น
ปฏิเสธ third-party proxy (เช่น "OpenClaw") — มี CVE auth-token exfiltration จริง; คุยกับ endpoint OpenAI ตรงเท่านั้น

=== protocol (verify กับ source ของ github.com/openai/codex ไม่ใช่เดา) ===
Token endpoint: code exchange = form-urlencoded, refresh = JSON body (asymmetry ยืนยันแล้ว อย่า copy ผิด)
Flow: Authorization Code + PKCE (S256), loopback redirect http://localhost:1455/auth/callback (fallback 1457)
    เปิด local server ชั่วคราวแค่ตอน login
Account info อยู่ใน id_token claims — decode โดยไม่ verify signature (ได้มาจาก token endpoint ผ่าน TLS เอง)
"""

import asyncio
import base64
import hashlib
import json
import secrets
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Optional

import httpx
from cryptography.fernet import InvalidToken

from backend.app.config import settings
from backend.app.core.crypto_store import get_fernet

AUTHORIZE_URL = "https://auth.openai.com/oauth/authorize"
TOKEN_URL = "https://auth.openai.com/oauth/token"
REVOKE_URL = "https://auth.openai.com/oauth/revoke"
CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"  # public client_id ของ Codex CLI — ดู risk disclosure ด้านบน
SCOPES = "openid profile email offline_access api.connectors.read api.connectors.invoke"
CALLBACK_PATH = "/auth/callback"
RESPONSES_BASE_URL = "https://chatgpt.com/backend-api/codex"

# W_openai_oauth (fix 2026-08-17c, ยืนยันจาก token จริงของ user): chatgpt_account_id/plan_type/user_id
# ไม่อยู่ top-level ของ id_token claims แต่ซ้อนใต้ namespaced key นี้ (OIDC custom claim)
_OPENAI_AUTH_CLAIMS_KEY = "https://api.openai.com/auth"


def _extract_openai_auth_claims(claims: dict) -> dict:
    """nested claims ใต้ _OPENAI_AUTH_CLAIMS_KEY หรือ {} ถ้าไม่มี (ไม่ throw)"""
    nested = claims.get(_OPENAI_AUTH_CLAIMS_KEY)
    return nested if isinstance(nested, dict) else {}

_TOKEN_FILENAME = "openai_oauth_token.json"

_CALLBACK_SUCCESS_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>เชื่อมต่อ ChatGPT สำเร็จ</title></head>
<body style="font-family: sans-serif; text-align: center; padding-top: 4rem;">
<h2>เชื่อมต่อ ChatGPT สำเร็จแล้ว</h2><p>ปิดแท็บนี้แล้วกลับไปที่หน้าเดิมได้เลย</p>
</body></html>"""


class OAuthLoginRequired(Exception):
    """ยังไม่เคย login หรือ refresh ไม่ได้แล้ว (ต้อง re-link) — llm.py::next_action_openai แปลงเป็นข้อความ user"""


# --- PKCE / state ---

def generate_pkce_pair() -> tuple[str, str]:
    """คืน (code_verifier, code_challenge) ตาม RFC 7636 S256"""
    code_verifier = secrets.token_urlsafe(96)[:128]
    digest = hashlib.sha256(code_verifier.encode("ascii")).digest()
    code_challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return code_verifier, code_challenge


def generate_state() -> str:
    return secrets.token_urlsafe(24)


def build_authorize_url(code_challenge: str, state: str, redirect_uri: str) -> str:
    params = {
        "client_id": CLIENT_ID,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": SCOPES,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "state": state,
        "id_token_add_organizations": "true",
        "codex_cli_simplified_flow": "true",
    }
    return f"{AUTHORIZE_URL}?{urllib.parse.urlencode(params)}"


# --- loopback callback server ชั่วคราว (เฉพาะตอน login) ---

def _make_callback_handler(result: dict):
    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 (stdlib naming convention)
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path != CALLBACK_PATH:
                self.send_response(404)
                self.end_headers()
                return
            qs = urllib.parse.parse_qs(parsed.query)
            result["code"] = qs.get("code", [None])[0]
            result["state"] = qs.get("state", [None])[0]
            result["error"] = qs.get("error", [None])[0]
            body = _CALLBACK_SUCCESS_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            pass  # เงียบ — ไม่ spam stdout ด้วย access log ของ http.server เอง

    return _Handler


def _bind_loopback_server() -> tuple[HTTPServer, int, dict]:
    """bind port หลัก (1455) ก่อน fallback (1457) เฉพาะเมื่อ bind ไม่ได้ — ต้องรู้ port จริงก่อนสร้าง
    authorize_url ไม่งั้น redirect_uri ไม่ตรงกับ listener"""
    result: dict = {"code": None, "state": None, "error": None}
    handler_cls = _make_callback_handler(result)
    last_error: Optional[OSError] = None
    for port in (settings.openai_oauth_callback_port, settings.openai_oauth_callback_port_fallback):
        try:
            server = HTTPServer(("127.0.0.1", port), handler_cls)
            return server, port, result
        except OSError as e:
            last_error = e
            continue
    raise RuntimeError(
        f"ไม่สามารถเปิด local server สำหรับ OAuth callback ได้ทั้งพอร์ต "
        f"{settings.openai_oauth_callback_port} และ {settings.openai_oauth_callback_port_fallback}: {last_error}"
    )


def _wait_for_callback(server: HTTPServer, timeout_seconds: float) -> None:
    """block รอ 1 request ด้วย handle_request() (ไม่ใช่ serve_forever() — shutdown() จาก thread เดียวกัน deadlock)"""
    server.timeout = timeout_seconds
    try:
        server.handle_request()
    finally:
        server.server_close()


# --- login orchestration (in-memory, ไม่ persist ข้าม restart) ---

_pending_logins: dict[str, dict] = {}


async def start_login_flow() -> dict:
    """คืน {authorize_url, login_id} — token exchange ทำใน background poll ผ่าน get_login_status(login_id)"""
    login_id = secrets.token_urlsafe(12)
    code_verifier, code_challenge = generate_pkce_pair()
    state = generate_state()

    server, port, result = await asyncio.to_thread(_bind_loopback_server)
    redirect_uri = f"http://localhost:{port}{CALLBACK_PATH}"
    authorize_url = build_authorize_url(code_challenge, state, redirect_uri)

    _pending_logins[login_id] = {"status": "pending", "error": None, "email": None, "plan_type": None}
    asyncio.create_task(
        _finish_login_background(login_id, server, result, code_verifier, state, redirect_uri)
    )
    return {"authorize_url": authorize_url, "login_id": login_id}


async def _finish_login_background(
    login_id: str, server: HTTPServer, result: dict, code_verifier: str, expected_state: str, redirect_uri: str,
) -> None:
    try:
        await asyncio.to_thread(_wait_for_callback, server, settings.openai_oauth_login_timeout_seconds)
        if result.get("error"):
            raise RuntimeError(f"OpenAI ปฏิเสธคำขอ login: {result['error']}")
        if not result.get("code"):
            raise TimeoutError("รอ OAuth callback นานเกินไป — ไม่มีการ login เข้ามาภายในเวลาที่กำหนด")
        if result.get("state") != expected_state:
            raise RuntimeError("state parameter ไม่ตรงกัน (อาจถูกโจมตีแบบ CSRF) — ยกเลิก login นี้")

        raw_tokens = await exchange_code_for_tokens(result["code"], code_verifier, redirect_uri)
        claims = decode_id_token_claims(raw_tokens["id_token"])
        auth_claims = _extract_openai_auth_claims(claims)
        tokens = {
            "id_token": raw_tokens["id_token"],
            "access_token": raw_tokens["access_token"],
            "refresh_token": raw_tokens["refresh_token"],
            "account_id": auth_claims.get("chatgpt_account_id"),
            "email": claims.get("email"),
            "plan_type": auth_claims.get("chatgpt_plan_type"),
            "exp": claims.get("exp"),
            "last_refresh": time.time(),
        }
        save_token(tokens)
        _pending_logins[login_id] = {
            "status": "linked", "error": None, "email": tokens["email"], "plan_type": tokens["plan_type"],
        }
    except Exception as e:
        _pending_logins[login_id] = {"status": "error", "error": str(e), "email": None, "plan_type": None}


def get_login_status(login_id: str) -> Optional[dict]:
    return _pending_logins.get(login_id)


def get_link_status() -> dict:
    """สถานะ link ปัจจุบัน (สำหรับ enable provider "openai" ใน UI) — ไม่คืน token จริงออกไป"""
    tokens = load_token()
    if tokens is None:
        return {"linked": False, "email": None, "plan_type": None}
    return {"linked": True, "email": tokens.get("email"), "plan_type": tokens.get("plan_type")}


# --- code exchange / refresh ---

async def exchange_code_for_tokens(code: str, code_verifier: str, redirect_uri: str) -> dict:
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            TOKEN_URL,
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": redirect_uri,
                "client_id": CLIENT_ID,
                "code_verifier": code_verifier,
            },
        )
    if response.status_code != 200:
        raise RuntimeError(f"แลก authorization code เป็น token ไม่สำเร็จ ({response.status_code}): {response.text}")
    return response.json()


async def refresh_tokens(refresh_token: str) -> dict:
    """W_openai_oauth: endpoint เดียวกับ exchange_code_for_tokens() แต่ body เป็น JSON (ยืนยันจาก
    source Codex CLI) — ห้ามรวม/copy จากฟังก์ชันนั้น"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            TOKEN_URL,
            json={"client_id": CLIENT_ID, "grant_type": "refresh_token", "refresh_token": refresh_token},
        )
    if response.status_code != 200:
        raise RuntimeError(f"refresh OpenAI OAuth token ไม่สำเร็จ ({response.status_code}): {response.text}")
    return response.json()


def decode_id_token_claims(id_token: str) -> dict:
    """decode payload ของ JWT โดยไม่ verify signature — ปลอดภัยเพราะได้จาก token endpoint ที่เรียกเองผ่าน TLS"""
    segments = id_token.split(".")
    if len(segments) != 3:
        raise ValueError("id_token ไม่ใช่รูปแบบ JWT ที่ถูกต้อง (ต้องมี 3 segment)")
    payload_segment = segments[1]
    padded = payload_segment + "=" * (-len(payload_segment) % 4)
    payload_bytes = base64.urlsafe_b64decode(padded)
    return json.loads(payload_bytes)


# --- token store (Fernet, key เดียวกับ site_learning/storage.py ผ่าน core/crypto_store.py) ---

def _token_path() -> Path:
    return Path(settings.site_manuals_dir).parent / _TOKEN_FILENAME


def save_token(tokens: dict) -> None:
    fernet = get_fernet()
    secret_payload = {
        "id_token": tokens["id_token"],
        "access_token": tokens["access_token"],
        "refresh_token": tokens["refresh_token"],
    }
    encrypted_blob = fernet.encrypt(json.dumps(secret_payload).encode("utf-8")).decode("ascii")
    record = {
        "encrypted_tokens": encrypted_blob,
        # metadata ที่ไม่ใช่ secret เก็บ plaintext นอก blob ให้ status อ่านได้โดยไม่ decrypt
        "account_id": tokens.get("account_id"),
        "email": tokens.get("email"),
        "plan_type": tokens.get("plan_type"),
        "exp": tokens.get("exp"),
        "last_refresh": tokens.get("last_refresh", time.time()),
    }
    path = _token_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record), encoding="utf-8")


def load_token() -> Optional[dict]:
    """token (decrypt แล้ว) + metadata หรือ None ถ้ายังไม่ login/ไฟล์เสีย/decrypt ไม่ได้ (ไม่ throw)"""
    path = _token_path()
    if not path.exists():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        fernet = get_fernet()
        decrypted = json.loads(fernet.decrypt(record["encrypted_tokens"].encode("ascii")).decode("utf-8"))
    except (json.JSONDecodeError, OSError, InvalidToken, KeyError, ValueError):
        return None
    return {
        **decrypted,
        "account_id": record.get("account_id"),
        "email": record.get("email"),
        "plan_type": record.get("plan_type"),
        "exp": record.get("exp"),
        "last_refresh": record.get("last_refresh"),
    }


def delete_token() -> bool:
    path = _token_path()
    if path.exists():
        path.unlink()
        return True
    return False


def token_exists() -> bool:
    return _token_path().exists()


# --- refresh cadence + public entrypoint ---

async def _refresh_if_needed(tokens: dict) -> dict:
    """refresh เมื่อใกล้ exp หรือ refresh ล่าสุดเก่ากว่า max_age_days (cadence เดียวกับ Codex CLI)
    ล้มเหลว -> OAuthLoginRequired"""
    now = time.time()
    exp = tokens.get("exp") or 0
    last_refresh = tokens.get("last_refresh") or 0
    needs_refresh = (
        (exp and now >= exp - settings.openai_oauth_refresh_before_expiry_seconds)
        or (now - last_refresh >= settings.openai_oauth_refresh_max_age_days * 86400)
    )
    if not needs_refresh:
        return tokens

    try:
        fresh = await refresh_tokens(tokens["refresh_token"])
    except Exception as e:
        raise OAuthLoginRequired(
            f"OpenAI OAuth token หมดอายุและ refresh ไม่สำเร็จ — ต้อง login ใหม่ผ่าน 'Sign in with ChatGPT' ({e})"
        ) from e

    claims = decode_id_token_claims(fresh["id_token"]) if fresh.get("id_token") else {}
    auth_claims = _extract_openai_auth_claims(claims)
    updated = {
        "id_token": fresh.get("id_token", tokens["id_token"]),
        "access_token": fresh.get("access_token", tokens["access_token"]),
        # refresh_token อาจ rotate หรือไม่ — ไม่มีตัวใหม่ต้องคงตัวเดิม ห้ามทิ้ง
        "refresh_token": fresh.get("refresh_token", tokens["refresh_token"]),
        "account_id": auth_claims.get("chatgpt_account_id", tokens.get("account_id")),
        "email": claims.get("email", tokens.get("email")),
        "plan_type": auth_claims.get("chatgpt_plan_type", tokens.get("plan_type")),
        "exp": claims.get("exp", tokens.get("exp")),
        "last_refresh": now,
    }
    save_token(updated)
    return updated


async def get_valid_access_token() -> tuple[str, str]:
    """public entrypoint: คืน (access_token, chatgpt_account_id) refresh อัตโนมัติถ้าจำเป็น
    raise OAuthLoginRequired ถ้ายังไม่ login/refresh ไม่สำเร็จ"""
    tokens = load_token()
    if tokens is None:
        raise OAuthLoginRequired(
            "ยังไม่ได้ login OpenAI OAuth — ไปที่ตั้งค่าเพื่อ 'Sign in with ChatGPT' ก่อนใช้ provider นี้"
        )
    tokens = await _refresh_if_needed(tokens)
    account_id = tokens.get("account_id")
    if not account_id:
        raise OAuthLoginRequired("token ที่เก็บไว้ไม่มี account_id — ต้อง login ใหม่")
    return tokens["access_token"], account_id


async def revoke_token() -> None:
    """logout: revoke แบบ best-effort แล้วลบ local เสมอ (ต้องสำเร็จได้แม้ offline)"""
    tokens = load_token()
    if tokens is not None:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                await client.post(REVOKE_URL, data={"client_id": CLIENT_ID, "token": tokens.get("refresh_token", "")})
        except Exception:
            pass
    delete_token()

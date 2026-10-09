"""Permission layer: classify_action() -> SAFE / NEEDS_CONFIRMATION / BLOCKED + SSRF guard
ชั้นเครือข่าย (install_ssrf_guard)

classify_action() รับ cmd dict ทั้งก้อน (จาก PR "permission-ab") ไม่ใช่แค่ type เพราะ risk ขึ้นกับ
parameter อื่นด้วย (เช่น goto ไป domain ที่ถูกบล็อก)
"""

import asyncio
import ipaddress
import socket
import urllib.parse
from contextlib import contextmanager
from contextvars import ContextVar
from enum import Enum

from backend.app.config import settings


class ActionRisk(str, Enum):
    SAFE = "safe"
    NEEDS_CONFIRMATION = "needs_confirmation"
    BLOCKED = "blocked"


DEFAULT_NEEDS_CONFIRMATION = {"submit", "delete", "purchase", "pay"}
DEFAULT_BLOCKED_ACTIONS: set[str] = set()

BLOCKED_DOMAINS = {
    "malicious.com",
    "phishing.net",
}

# ถ้า ALLOWED_DOMAINS มีค่า จะบล็อกโดเมนที่ไม่อยู่ในนี้ทั้งหมด (ถ้าว่าง แปลว่าอนุญาตทั้งหมดที่ไม่ได้ถูกบล็อก)
ALLOWED_DOMAINS: set[str] = set()

# defense-in-depth: LLM อาจส่ง type="click" ให้ปุ่มที่มีผลจริง (เช่น saucedemo "Remove" เป็น <button>
# ธรรมดา) — model compliance ไม่การันตี จึงเช็คคำในป้ายด้วย
# W_risky_multilingual: เดิมเป็นอังกฤษล้วน ปุ่ม "ลบ"/"削除"/"Löschen" ไม่ถูกยกระดับเลย — substring match
# ในภาษาไม่เว้นวรรคอาจ false positive ได้ ยอมรับเพราะแค่ถามเกิน 1 ครั้ง ดีกว่าลบข้อมูลเงียบๆ
# (RISKY ชนะ SAFE เสมอใน classify_action)
RISKY_LABEL_KEYWORDS = {
    # อังกฤษ (ชุดเดิม ห้ามตัดออก)
    "remove", "delete", "place order", "finish", "pay", "purchase", "confirm",
    # ไทย
    "ลบ", "ล้างข้อมูล", "นำออก", "สั่งซื้อ", "ชำระเงิน", "จ่ายเงิน", "ยืนยัน", "เสร็จสิ้น",
    # ญี่ปุ่น / จีน
    "削除", "購入", "支払", "確認", "删除", "购买", "支付", "确认",
    # เยอรมัน / ฝรั่งเศส / สเปน / โปรตุเกส
    "löschen", "loschen", "entfernen", "bezahlen", "kaufen", "bestätigen", "bestatigen",
    "supprimer", "payer", "acheter", "confirmer",
    "eliminar", "borrar", "pagar", "comprar", "confirmar", "excluir",
}

# W_search: บั๊กจริงที่ user รายงาน — LLM บางครั้งเลือก type "submit" ให้ปุ่มค้นหา/ดูเนื้อหา ทำให้ถาม
# confirmation ผิดๆ; label ที่ชัดว่าเป็นค้นหา/กรอง/ดู ลดกลับเป็น SAFE (RISKY ยังชนะถ้า match ทั้งคู่
# เช่น "Confirm and Search")
SAFE_ACTION_LABEL_KEYWORDS = {
    "search", "ค้นหา", "ค้น", "find", "filter", "กรอง",
    "watch", "ดู", "ชม", "play", "เล่น", "view", "read", "อ่าน",
    "browse", "next", "ถัดไป", "previous", "ก่อนหน้า", "go to", "open",
}

# W_search follow-up: การ์ดผลค้นหา (เช่น วิดีโอ YouTube) มี label เป็นชื่อเรื่องดิบ ไม่ match คำไหน —
# ใช้สัญญาณโครงสร้าง: <a> แค่ navigate (GET) แทบไม่เคยเป็น submit/ลบ/จ่ายเงิน
ANCHOR_TAG = "a"

# W_search follow-up 2: หลัง fill ช่องค้นหา label กลายเป็นค่าที่พิมพ์ (perception fallback ไป el.value)
# ถ้า LLM เลือก submit ให้ "กด Enter ค้นหา" — <input> ที่ไม่ใช่ submit/image/password ถือว่าปลอดภัย
SAFE_INPUT_TAG = "input"
RISKY_INPUT_TYPES = {"submit", "image", "password"}

# W7[B]: RAG-based permission — คู่มือของ user อาจกำหนดให้ขออนุมัติเพิ่ม (เช่น "สั่งซื้อเกิน $100 ต้อง
# ขออนุมัติ") สแกนคำด้วยโค้ด ไม่ให้ LLM ตัดสินเอง
MANUAL_CONFIRMATION_KEYWORDS = {
    "ต้องขออนุมัติ", "ต้องได้รับอนุมัติ", "ต้องขอความยินยอม", "ต้องยืนยันก่อน",
    "requires approval", "needs approval", "require confirmation",
    "requires confirmation", "ask for confirmation", "confirm before", "ask before",
}


def _label_looks_risky(label: str) -> bool:
    lower = (label or "").lower()
    return any(keyword in lower for keyword in RISKY_LABEL_KEYWORDS)


def _label_looks_safe(label: str) -> bool:
    lower = (label or "").lower()
    return any(keyword in lower for keyword in SAFE_ACTION_LABEL_KEYWORDS)


def _manual_requires_confirmation(manual_guidance: str) -> bool:
    lower = (manual_guidance or "").lower()
    return any(keyword in lower for keyword in MANUAL_CONFIRMATION_KEYWORDS)


def normalize_domain(domain: str) -> str:
    """lowercase + ตัด "www." สำหรับ hostname ล้วน (เช่น path param) — จุดเดียวที่ extract_domain()
    ใช้ร่วม กันกติกา www. drift"""
    domain = domain.lower()
    if domain.startswith("www."):
        domain = domain[len("www."):]
    return domain


def extract_domain(url: str) -> str:
    """domain จาก URL (lowercase, ไม่มี port, ไม่มี "www.") ใช้ร่วมทุกจุดในระบบ — URL พังคืน "" ไม่ throw

    ตัด "www." โดยเจตนา: เดิม credential ที่บันทึกผ่าน www.example.com หาไม่เจอตอนรันจาก
    example.com (storage.py key ด้วยค่านี้ตรงๆ)"""
    try:
        domain = urllib.parse.urlparse(url).netloc.lower()
        if ":" in domain:
            domain = domain.split(":")[0]
        return normalize_domain(domain)
    except Exception:
        return ""


# W_gate_local_hrm: อนุญาต internal navigation "เฉพาะงานที่ขอ" — เดิม hrm_local_eval เปิด
# settings.allow_internal_navigation แบบ global ทำให้ task ของ user ใน process เดียวกันเข้า localhost ได้หมด
# ContextVar มีผลเฉพาะงานที่สร้างใน with block; browser จาก BrowserPool ที่ start ไว้ก่อนมองไม่เห็นค่านี้
# = พังไปทางปลอดภัย
_internal_navigation_override: ContextVar[bool] = ContextVar(
    "internal_navigation_override", default=False,
)


@contextmanager
def allow_internal_navigation_here():
    """เปิดให้ navigate ไป IP ภายในได้เฉพาะงานที่เริ่มภายใน block นี้ (ดูเหตุผลด้านบน)"""
    token = _internal_navigation_override.set(True)
    try:
        yield
    finally:
        _internal_navigation_override.reset(token)


def _internal_navigation_allowed() -> bool:
    return settings.allow_internal_navigation or _internal_navigation_override.get()


def is_private_or_internal(domain: str) -> bool:
    """Security 1.2 (SSRF): domain resolve เป็น private/loopback/link-local IP ไหม (169.254.169.254,
    192.168.x.x, localhost) — goto ไม่อยู่ใน DEFAULT_NEEDS_CONFIRMATION จึงต้องกันตรงนี้

    resolve ไม่ได้ = False (ปล่อยให้การเชื่อมต่อจริง fail เอง)"""
    hostname = (domain or "").strip()
    if not hostname:
        return False
    try:
        # แปลงเป็น IP ก่อนตัด port — IPv6 literal (เช่น "::1") มี ":" อยู่แล้ว
        ip = ipaddress.ip_address(hostname)
        return ip.is_private or ip.is_loopback or ip.is_link_local
    except ValueError:
        pass
    if hostname.count(":") == 1:  # "host:port" ธรรมดา (ไม่ใช่ IPv6 ที่มี ":" หลายตัว)
        hostname = hostname.split(":")[0]
    if hostname == "localhost":
        return True
    try:
        addresses = {
            info[4][0] for info in socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
        }
    except (socket.gaierror, ValueError, OSError):
        return False
    return any(
        ipaddress.ip_address(address).is_private
        or ipaddress.ip_address(address).is_loopback
        or ipaddress.ip_address(address).is_link_local
        for address in addresses
    )


def classify_action(
    cmd: dict, label: str = "", manual_guidance: str = "", allowed_domains: "set[str] | None" = None,
    element_tag: str = "", element_type: str = "",
) -> ActionRisk:
    """สัญญาณเสริมทั้งหมด optional (default "" = ข้ามการเช็คชั้นนั้น ไม่ throw):
    - label: ข้อความของ element — RISKY/SAFE_ACTION_LABEL_KEYWORDS
    - manual_guidance (W7[B]): manual_context ที่ orchestrator ดึงมาแล้ว (ไม่ยิง ChromaDB ซ้ำ)
    - allowed_domains: override ALLOWED_DOMAINS เฉพาะ call นี้ (None = ค่า module; set ว่าง = ไม่จำกัด
      ไม่ใช่ deny-all) BLOCKED_DOMAINS เป็น hard block เสมอ
    - element_tag / element_type (W_search follow-up/-2): สัญญาณโครงสร้าง ดู ANCHOR_TAG /
      SAFE_INPUT_TAG / RISKY_INPUT_TYPES"""
    action_type = cmd.get("type", "")

    if action_type in DEFAULT_BLOCKED_ACTIONS:
        return ActionRisk.BLOCKED

    if action_type in DEFAULT_NEEDS_CONFIRMATION:
        # ลำดับสำคัญ: คู่มือ (กฎของ user) > risky label > safe label > tag downgrade
        if _manual_requires_confirmation(manual_guidance):
            return ActionRisk.NEEDS_CONFIRMATION
        if _label_looks_risky(label):
            return ActionRisk.NEEDS_CONFIRMATION
        if _label_looks_safe(label):
            return ActionRisk.SAFE
        # Security 1.3: tag downgrade ใช้ได้เฉพาะ "submit" — เดิมครอบ delete/purchase/pay ด้วย ทำให้
        # ปุ่ม "ลบ"/"สั่งซื้อ" ที่ทำเป็น <a> หลบ confirmation ได้ (บั๊กจริง)
        if action_type == "submit":
            if (element_tag or "").lower() == ANCHOR_TAG:
                return ActionRisk.SAFE
            if (element_tag or "").lower() == SAFE_INPUT_TAG and (element_type or "").lower() not in RISKY_INPUT_TYPES:
                return ActionRisk.SAFE
        return ActionRisk.NEEDS_CONFIRMATION

    if action_type == "goto":
        url = cmd.get("url", "")
        domain = extract_domain(url)

        # Security 1.2 (SSRF): hard block ก่อน BLOCKED/ALLOWED_DOMAINS เสมอ ไม่ขึ้นกับ config ผู้ใช้
        if not _internal_navigation_allowed() and is_private_or_internal(domain):
            return ActionRisk.BLOCKED

        if domain in BLOCKED_DOMAINS:
            return ActionRisk.BLOCKED

        effective_allowed = ALLOWED_DOMAINS if allowed_domains is None else allowed_domains
        if effective_allowed and domain not in effective_allowed:
            return ActionRisk.BLOCKED

        # goto ไม่เช็ค label (ไม่มี index ให้จับคู่ label จึงว่าง) — เช็คแค่คู่มือ
        if _manual_requires_confirmation(manual_guidance):
            return ActionRisk.NEEDS_CONFIRMATION
        return ActionRisk.SAFE

    if _label_looks_risky(label) or _manual_requires_confirmation(manual_guidance):
        return ActionRisk.NEEDS_CONFIRMATION

    return ActionRisk.SAFE


# Security (follow-up to 1.2): classify_action() เช็ค SSRF แค่ goto — ไม่ครอบ (1) คลิก <a href> ไป IP
# ภายใน (2) 302 redirect / DNS rebinding (TOCTOU) — จึงดักที่ชั้นเครือข่ายด้วย Playwright route ทุก
# navigation request (Chromium ยิงแยกทุก hop ของ redirect) ไม่เช็ค subresource เพื่อไม่เสีย performance
async def install_ssrf_guard(target) -> None:
    """เรียกทันทีหลังสร้าง Page หรือ BrowserContext ทุกจุดที่ agent navigate ได้ (context.route ครอบทุก
    page ของ context) — never raises: target ที่ปิดแล้วปล่อยผ่านเงียบๆ ดีกว่าทำ page creation พัง"""
    try:
        await target.route("**/*", _ssrf_route_handler)
    except Exception:
        pass


# getaddrinfo ไม่มี timeout ในตัว และ handler ทำงานทุก navigation — กัน DNS ช้าทำ navigation ค้าง;
# timeout = ปล่อยผ่าน (fail open) เหมือน resolve ไม่ได้
_SSRF_DNS_TIMEOUT_SECONDS = 2.0


async def _ssrf_route_handler(route) -> None:
    request = route.request
    if not request.is_navigation_request():
        await route.continue_()
        return
    try:
        hostname = urllib.parse.urlparse(request.url).hostname or ""
        # DNS lookup เป็น blocking — to_thread กันบล็อก event loop ของ Playwright
        blocked = not _internal_navigation_allowed() and await asyncio.wait_for(
            asyncio.to_thread(is_private_or_internal, hostname),
            timeout=_SSRF_DNS_TIMEOUT_SECONDS,
        )
    except Exception:
        # resolve ไม่ได้/timeout/URL ผิดรูป — ปล่อยผ่าน ให้ Playwright fail เองตอน connect
        blocked = False
    if blocked:
        await route.abort()
    else:
        await route.continue_()

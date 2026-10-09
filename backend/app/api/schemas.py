from typing import Optional

from pydantic import BaseModel, Field, model_validator
from backend.app.core.embedded_page import PageSnapshot

# Security (follow-up to SEC audit): จำกัดขนาดไฟล์แนบ (decoded) 10MB กัน memory-exhaustion/
# zip-bomb ผ่าน .xlsx/.docx (openpyxl/python-docx decompress เต็มก้อนไม่มี guard) — จำกัดแค่
# input ที่ endpoint ไม่ได้จำกัด decompression ratio ข้างใน library; คิดจากความยาว base64 (~4/3)
_MAX_ATTACHED_FILE_DECODED_BYTES = 10 * 1024 * 1024  # 10MB
_MAX_ATTACHED_FILE_BASE64_CHARS = (_MAX_ATTACHED_FILE_DECODED_BYTES * 4 // 3) + 4  # + padding เผื่อ


class CreateTaskRequest(BaseModel):
    # Duck-typed กับ ExecutePlanRequest ใน routes.py::_run_with_resolved_browser() —
    # เพิ่ม field ที่นี่มักต้องเพิ่มที่นั่นด้วย
    url: str
    goal: str
    max_steps: int = 30
    provider: Optional[str] = None  # None = ใช้ settings.llm_provider (ดู orchestrator.py)
    headless: Optional[bool] = None  # None = ใช้ settings.browser_headless
    # W10[A]: REST ไม่มี human คอยตอบ permission prompt — default False = action ที่ต้อง
    # ขออนุมัติถูกปฏิเสธอัตโนมัติ (fail closed); true = caller รับผิดชอบเองว่าไม่มี human-in-the-loop
    auto_approve: bool = False
    # ร่างแผนก่อน loop (result["plan"]) ให้ UI โชว์ — ไม่ใช่ permission-gated action
    # (routes.py::_make_ask_user_func auto-approve confirm_plan เสมอ แยกจาก auto_approve)
    confirm_plan: bool = True
    # W12: True = ต่อ Chrome จริงของ user ผ่าน CDP (cookie/login ค้างอยู่) แทน launch Chromium
    # ว่าง (ดู core/user_browser.py) — user ต้องเปิด Chrome ด้วย --remote-debugging-port เอง;
    # headless ไม่มีผลในโหมดนี้
    use_user_browser: bool = False
    embedded_page: Optional[PageSnapshot] = None
    target_tab_id: Optional[str] = Field(default=None, min_length=1, max_length=128)
    # None = ใช้ settings.user_browser_tab_reuse_policy (มีผลเฉพาะ use_user_browser=True):
    # "ask" / "always_reuse" / "always_new_tab" — ดู core/user_browser.py::resolve_target_page
    tab_reuse_policy: Optional[str] = None
    # W12: None = acquire/launch browser ใหม่แล้วปิดตอนจบ task; ส่งมา = ผูกกับ session ที่อยู่ข้าม
    # หลาย POST /tasks (core/session_registry.py) — headless/use_user_browser ใช้ค่าตอนสร้าง
    # session ครั้งแรกเท่านั้น ปิดด้วย POST /sessions/{id}/close
    session_id: Optional[str] = None
    # Security (SEC-4 follow-up): secret คู่ session_id (BrowserSession.owner_token) — ต้องแนบทุกครั้ง
    # ที่ session มีอยู่แล้ว ไม่งั้น SessionOwnershipError -> 403; None ตอนสร้างใหม่ = ไม่มี token (compat CLI/test)
    session_owner_token: Optional[str] = None
    # pdf/xlsx: ไฟล์แนบจาก composer (คนละอย่างกับ RAG upload) — ส่งมาทั้งคู่ = routes.py ตอบจาก
    # เนื้อหาไฟล์ตรงๆ ไม่แตะ browser; content เป็น base64 ของไฟล์ดิบ (ระบบเป็น JSON body ล้วน,
    # decode ที่ rag/ingestion.py::load_manual_bytes)
    attached_file_name: Optional[str] = None
    attached_file_content_base64: Optional[str] = Field(
        default=None, max_length=_MAX_ATTACHED_FILE_BASE64_CHARS,
    )

    @model_validator(mode="after")
    def validate_embedded_tab(self):
        if self.embedded_page and (self.use_user_browser or self.target_tab_id):
            raise ValueError("embedded_page cannot use a CDP browser")
        if self.target_tab_id and (not self.use_user_browser or not self.session_id):
            raise ValueError("target_tab_id requires use_user_browser and session_id")
        return self


class GeneratePlanRequest(BaseModel):
    """W13: body ของ POST /api/generate_plan — วางแผนอย่างเดียว ไม่เปิด/connect browser ใหม่"""

    url: str
    goal: str
    provider: Optional[str] = None
    # lookup อย่างเดียว (session_registry.get()) ใช้ perceive หน้าเดิมให้แผน grounded —
    # ไม่มีทางสร้าง session/เปิด browser จาก endpoint นี้
    session_id: Optional[str] = None
    # Security (SEC-4 follow-up): ดู CreateTaskRequest.session_owner_token
    session_owner_token: Optional[str] = None
    # pdf/xlsx: มีค่า = คืน is_qa=True ทันที (ข้าม classify_intent()/LLM) ให้ frontend ตอบจากไฟล์
    attached_file_name: Optional[str] = None
    attached_file_content_base64: Optional[str] = Field(
        default=None, max_length=_MAX_ATTACHED_FILE_BASE64_CHARS,
    )
    # W20 ("Context-Aware Implicit Execution"): goal/reply ของเทิร์นก่อนหน้า (จาก client history)
    # ให้ planner resolve anaphora อย่าง "เปิดให้หน่อย"/"play it" กับสิ่งที่เพิ่งแนะนำไป (ซึ่งไม่อยู่ใน
    # session.extracted_memory ถ้ามาจาก general-chat) — None = เทิร์นแรก พฤติกรรมเดิม
    previous_user_goal: Optional[str] = None
    previous_assistant_message: Optional[str] = None


class GeneratePlanResponse(BaseModel):
    plan: str
    is_qa: bool = False
    # W_procmem: 4 ฟิลด์ additive (consumer เดิมอ่านแค่ .plan/.is_qa) — source: "llm" |
    # "plan_memory" | "procedural_reuse"/"procedural_adapt" (2 ค่าหลังมี template_id/slot_values/
    # steps เสมอ ให้ส่งต่อเข้า POST /api/execute_plan เพื่อวิ่ง fast-path; ดู core/procedural_memory.py)
    source: str = "llm"
    template_id: Optional[str] = None
    slot_values: Optional[dict] = None
    steps: Optional[list[dict]] = None


class ExecutePlanRequest(BaseModel):
    """W13: body ของ POST /api/execute_plan — เหมือน CreateTaskRequest แต่ไม่มี confirm_plan
    (อนุมัติแล้วผ่าน generate_plan) และมี `plan` แทน"""

    url: str
    goal: str
    # แผนที่อนุมัติแล้ว -> run_task(approved_plan=...); None = ทำตาม goal ตรงๆ ไม่มีแผนกำกับ
    plan: Optional[str] = None
    max_steps: int = 30
    provider: Optional[str] = None
    headless: Optional[bool] = None
    auto_approve: bool = False
    use_user_browser: bool = False
    tab_reuse_policy: Optional[str] = None
    session_id: Optional[str] = None
    # Security (SEC-4 follow-up): ดู CreateTaskRequest.session_owner_token
    session_owner_token: Optional[str] = None
    # W_procmem: ค่าจาก GeneratePlanResponse — execution_mode=="fastpath" + template_id/steps ครบ
    # = routes.py::execute_plan() ใช้ run_fastpath() แทน run_task(); แผนที่ user แก้ข้อความเอง
    # ต้อง clear 3 ฟิลด์นี้ (index.html) ไม่งั้นรัน step ที่ไม่ตรงกับที่อนุมัติ
    template_id: Optional[str] = None
    slot_values: Optional[dict] = None
    steps: Optional[list[dict]] = None
    execution_mode: Optional[str] = None
    # pdf/xlsx: mirror ของ CreateTaskRequest
    attached_file_name: Optional[str] = None
    attached_file_content_base64: Optional[str] = Field(
        default=None, max_length=_MAX_ATTACHED_FILE_BASE64_CHARS,
    )


class TaskCreatedResponse(BaseModel):
    embedded_token: Optional[str] = None
    task_id: str
    status: str


class TaskStatusResponse(BaseModel):
    task_id: str
    url: str
    goal: str
    provider: Optional[str]
    status: str
    created_at: float
    result: Optional[dict] = None
    error: Optional[str] = None
    # W_live: headless ที่ resolve แล้ว — frontend โชว์ live view เฉพาะ True (index.html::renderLiveView)
    headless: bool = True
    # W20 (Task4): mirror ของ TaskRecord.attached_file_name ให้ render attachment card ได้หลัง refresh
    attached_file_name: Optional[str] = None


class PoolStatusResponse(BaseModel):
    size: int
    available: int
    in_use: int


class SessionStatusResponse(BaseModel):
    session_id: str
    mode: str
    created_at: float
    last_active_at: float


class SiteManualStatusResponse(BaseModel):
    """W14: body ของ GET /api/site-manual/status — banner "เว็บไซต์นี้ยังไม่มีคู่มือ" บน console"""

    exists: bool
    version: Optional[int] = None


class LearnSiteRequest(BaseModel):
    url: str
    provider: Optional[str] = None
    # W15: login bootstrap ตอน crawl (site_learning/auto_login.py)
    # W17: ส่งมาทั้งคู่ = บันทึกลง credentials.json ของโดเมน (storage.py::save_credentials) ให้ใช้ auto-login ภายหลัง
    username: Optional[str] = None
    password: Optional[str] = None
    # W18: ไม่ส่ง username/password + True = routes.py::learn_site() โหลด credential ที่เก็บไว้เอง
    # (frontend ไม่ต้องถือรหัสผ่าน — credentials/status ไม่คืนรหัสจริงอยู่แล้ว)
    use_saved_credentials: bool = False


class SaveCredentialsRequest(BaseModel):
    """W17: body ของ POST /api/site-manual/{domain}/credentials — บันทึก credential โดยไม่ crawl ใหม่"""

    username: str
    password: str


class CredentialsStatusResponse(BaseModel):
    exists: bool


class LearnCreatedResponse(BaseModel):
    learn_id: str
    status: str


class LearnCredentialsRequest(BaseModel):
    """W23: body ของ POST /api/site-manual/learn/{learn_id}/credentials — ตอบ event
    "credentials_needed" (crawl เจอหน้า login แต่ไม่มี credential; crawler.py::on_credentials_needed)

    request_id ต้องตรงกับ event (LearnManager.resolve_credentials()); username/password เป็น
    None ทั้งคู่ = ข้าม login ให้ crawl ต่อโดยไม่ผ่านหน้านี้"""

    request_id: str
    username: Optional[str] = None
    password: Optional[str] = None


class RelearnPageRequest(BaseModel):
    """W14: body ของ POST /api/site-manual/{domain}/relearn-page — selector-repair เฉพาะหน้าเดียว"""

    url: str
    provider: Optional[str] = None


class RelearnPageResponse(BaseModel):
    version: int


class RespondRequest(BaseModel):
    """W10[B]: body ของ POST /tasks/{id}/respond — request_id ต้องตรงกับ approval_request event
    (ไม่งั้นถือว่าหมดอายุ/ตอบไปแล้ว ดู TaskManager.resolve_approval())"""

    request_id: str
    approved: bool
    # W10[F]: confirm_plan ที่ user แก้ข้อความแผนเอง (None = ใช้แผนเดิม)
    edited_plan: Optional[str] = None
    # W_resume ("Mid-Task Input Request"): คำตอบของ request_user_input (None = ปฏิเสธ/ข้าม)
    answer_text: Optional[str] = None


class OpenAILoginStartResponse(BaseModel):
    """W_openai_oauth: response ของ POST /api/auth/openai/login/start — frontend เปิด authorize_url
    แล้ว poll .../login/status ด้วย login_id (token exchange อยู่ใน background, openai_oauth.py::start_login_flow())"""

    authorize_url: str
    login_id: str


class OpenAILoginStatusResponse(BaseModel):
    """W_openai_oauth: response ของ GET /api/auth/openai/login/status?login_id= —
    status: "pending"|"linked"|"error" (login_id ไม่รู้จัก = 404)"""

    status: str
    error: Optional[str] = None
    email: Optional[str] = None
    plan_type: Optional[str] = None


class OpenAIAuthStatusResponse(BaseModel):
    """W_openai_oauth: response ของ GET /api/auth/openai/status — สถานะ link ปัจจุบัน (ไม่คืน token)"""

    linked: bool
    email: Optional[str] = None
    plan_type: Optional[str] = None

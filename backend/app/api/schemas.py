from typing import Optional

from pydantic import BaseModel, Field, model_validator
from backend.app.core.embedded_page import PageSnapshot

# Security (follow-up to SEC audit): attached_file_content_base64 ไม่เคยมี size limit เลย
# ตั้งแต่ต้น — เสี่ยง memory-exhaustion/zip-bomb ผ่าน .xlsx/.docx (ทั้งคู่เป็น zip ข้างใน
# decompress เต็มก้อนโดย openpyxl/python-docx ไม่มี guard เรื่องขนาดหลัง decompress เลย) —
# จำกัดที่ "ขนาดไฟล์ก่อนเข้ารหัส" (decoded) ไว้ที่ 10MB ซึ่งใหญ่พอสำหรับเอกสาร/ตารางที่คนแนบ
# จริงในแชท (เนื้อหาสุดท้ายก็ต้องยัดใส่ LLM prompt อยู่ดี ไม่มีประโยชน์ให้ใหญ่กว่านี้มาก) —
# scope แค่จำกัดขนาด input ที่ endpoint boundary เท่านั้น ไม่ได้แก้ zip-bomb แบบเต็มรูปแบบ
# (เช่น จำกัด decompression ratio ข้างใน openpyxl/python-docx เอง) เพราะต้องแก้ library
# ที่ใช้อยู่เพิ่มอีกชั้น แลกกับความเสี่ยงที่ลดลงมากแล้วจากการจำกัดขนาด input (zip bomb ใน
# ไฟล์ 10MB ยังขยายได้มาก แต่ไม่ใช่ "unlimited" เหมือนเดิม) — คำนวณจากขนาด base64 string
# (ใหญ่กว่าขนาดไฟล์จริง ~4/3 เท่าจาก base64 encoding overhead)
_MAX_ATTACHED_FILE_DECODED_BYTES = 10 * 1024 * 1024  # 10MB
_MAX_ATTACHED_FILE_BASE64_CHARS = (_MAX_ATTACHED_FILE_DECODED_BYTES * 4 // 3) + 4  # + padding เผื่อ


class CreateTaskRequest(BaseModel):
    url: str
    goal: str
    max_steps: int = 30
    provider: Optional[str] = None  # None = ใช้ settings.llm_provider (ดู orchestrator.py)
    headless: Optional[bool] = None  # None = ใช้ settings.browser_headless
    # W10[A]: ไม่มี human อยู่หน้าจอคอยตอบ permission prompt ผ่าน REST ตรงๆ (ต่างจาก
    # run.py ที่ถาม terminal ได้) — ค่าเริ่มต้น False = action ที่ต้องขออนุมัติ (ดู
    # permission/rules.py) จะถูก "ปฏิเสธ" อัตโนมัติเสมอ (fail closed ปลอดภัยกว่า) ไม่ใช่
    # เงียบๆ อนุมัติให้เอง — ถ้าอยากให้ agent ทำ action พวกนี้ได้ ต้องส่ง true มาเอง
    # (รับผิดชอบเองว่าไม่มี human-in-the-loop จริงๆ ระหว่างรอบนี้)
    auto_approve: bool = False
    # ให้ LLM ร่างแผนคร่าวๆ ก่อนเริ่ม loop จริง (orchestrator.py::run_task) แล้วเก็บไว้ใน
    # result["plan"] ให้ console UI แสดงเป็น panel "Plan" — ค่า default True เพราะแค่
    # "โชว์แผน" ไม่ใช่ permission-gated action (ดู routes.py::_make_ask_user_func ที่
    # auto-approve confirm_plan เสมอ แยกจาก auto_approve ที่คุม action จริงบนหน้าเว็บ)
    confirm_plan: bool = True
    # W12: True = agent เชื่อมเข้า Chrome จริงที่ user เปิดใช้งานอยู่ผ่าน CDP (มี
    # cookie/login ค้างอยู่จริง เช่น mail) แล้วเปิด/ใช้ tab ใน browser ตัวนั้นเลย แทนที่จะ
    # launch Chromium ว่างๆ แยกต่างหาก (ดู core/user_browser.py, orchestrator.py::
    # run_task ส่วน connect_to_user_browser) — ก่อนใช้ user ต้องเปิด Chrome เองล่วงหน้าด้วย
    # --remote-debugging-port (ดู index.html ข้อความเตือนข้าง checkbox นี้) headless ไม่มี
    # ผลใดๆ เมื่อตั้งค่านี้เป็น True (browser ที่ต่อเข้าไปเป็นของ user เองอยู่แล้ว)
    use_user_browser: bool = False
    embedded_page: Optional[PageSnapshot] = None
    target_tab_id: Optional[str] = Field(default=None, min_length=1, max_length=128)

    @model_validator(mode="after")
    def validate_embedded_tab(self):
        if self.embedded_page and (self.use_user_browser or self.target_tab_id):
            raise ValueError("embedded_page cannot use a CDP browser")
        if self.target_tab_id and (not self.use_user_browser or not self.session_id):
            raise ValueError("target_tab_id requires use_user_browser and session_id")
        return self
    # None = ใช้ settings.user_browser_tab_reuse_policy — มีผลเฉพาะตอน
    # use_user_browser=True: "ask" (default) ถามก่อนใช้ tab เดิมของ conversation นี้ต่อ
    # ทุกเทิร์น, "always_reuse" ใช้ต่อเลยไม่ถาม (เหมาะกับ follow-up หลายเทิร์นใน
    # conversation เดียวกันที่อยากให้ต่อเนื่องไม่มี prompt คั่น), "always_new_tab" เปิด
    # tab ใหม่ทุกเทิร์นเสมอ (ดู core/user_browser.py::resolve_target_page)
    tab_reuse_policy: Optional[str] = None
    # W12: ไม่ส่งมา (None, default) = พฤติกรรมเดิมทุกประการ — acquire/launch/connect
    # browser ใหม่ทุกครั้งแล้วปิด/คืนตอนจบ task นี้เท่านั้น (เหมือน W1-W11) ส่งมา = ผูก
    # task นี้เข้ากับ session ที่มีชีวิตอยู่ข้ามหลาย POST /tasks (ดู
    # core/session_registry.py) — ครั้งแรกที่เจอ session_id นี้จะสร้าง browser/context/
    # page ใหม่ตาม use_user_browser/headless ของ request นั้น (ค่า headless/
    # use_user_browser ของ request ถัดๆ ไปที่ session_id เดียวกันจะถูกละเว้น เพราะ
    # resource ผูกไว้กับโหมดตอนสร้าง session แล้ว) ครั้งถัดๆ ไปด้วย session_id เดิมจะได้
    # page ตัวเดิมกลับมาทันที ไม่เปิดใหม่ — ปิด session ด้วย POST /sessions/{id}/close
    # เท่านั้น (ปุ่ม "New Session" บน Test Console)
    session_id: Optional[str] = None
    # Security (SEC-4 follow-up): secret ที่ frontend generate คู่กับ session_id เอง (ดู
    # core/session_registry.py::BrowserSession.owner_token) — ต้องแนบมาด้วยทุกครั้งที่
    # session_id นี้เคยมีอยู่แล้ว ไม่งั้นถือว่าไม่ใช่เจ้าของ (SessionOwnershipError -> 403)
    # ไม่ส่งมา (None) ตอน session_id ยังไม่เคยมีอยู่จริง = สร้างใหม่โดยไม่มี token ป้องกันเลย
    # (เข้ากันได้กับ caller เดิมที่ยังไม่รู้จัก field นี้ เช่น CLI/test)
    session_owner_token: Optional[str] = None
    # pdf/xlsx: user แนบไฟล์ PDF/XLSX ผ่าน composer โดยตรง (ต่างจาก site-manual/RAG
    # upload) — ทั้งคู่ None (default) = พฤติกรรมเดิมทุกประการ ส่งมาทั้งคู่ =
    # routes.py::_run_with_resolved_browser ตอบจากเนื้อหาไฟล์ตรงๆ ไม่แตะ browser/session/
    # pool เลย (เหมือน general-chat shortcut) — content เป็น base64 ของไฟล์ดิบ (ไม่ใช่
    # multipart เพราะทั้งระบบนี้เป็น JSON body ล้วนๆ อยู่แล้ว ดู core/rag/ingestion.py::
    # load_manual_bytes สำหรับตัว decode/extract จริง)
    attached_file_name: Optional[str] = None
    attached_file_content_base64: Optional[str] = Field(
        default=None, max_length=_MAX_ATTACHED_FILE_BASE64_CHARS,
    )


class GeneratePlanRequest(BaseModel):
    """W13: body ของ POST /api/generate_plan — เฟสวางแผนแยกต่างหาก ไม่เปิด/connect
    browser ใหม่เลย (ดู orchestrator.py::Orchestrator.generate_plan())"""

    url: str
    goal: str
    provider: Optional[str] = None
    # session_id (optional): ถ้ามี session นี้อยู่แล้วจริง (มี page เปิดค้างอยู่จาก
    # เทิร์นก่อนหน้า) จะ perceive หน้านั้นมาช่วยร่างแผนให้ grounded กับสถานะปัจจุบัน — เป็น
    # แค่ lookup เฉยๆ (session_registry.get(), ไม่ใช่ get_or_create()) ไม่มีทางสร้าง
    # session/เปิด browser ใหม่จาก endpoint นี้เด็ดขาด ไม่ว่า session_id จะมีอยู่จริงไหม
    session_id: Optional[str] = None
    # Security (SEC-4 follow-up): secret ที่ frontend generate คู่กับ session_id เอง (ดู
    # core/session_registry.py::BrowserSession.owner_token) — ต้องแนบมาด้วยทุกครั้งที่
    # session_id นี้เคยมีอยู่แล้ว ไม่งั้นถือว่าไม่ใช่เจ้าของ (SessionOwnershipError -> 403)
    # ไม่ส่งมา (None) ตอน session_id ยังไม่เคยมีอยู่จริง = สร้างใหม่โดยไม่มี token ป้องกันเลย
    # (เข้ากันได้กับ caller เดิมที่ยังไม่รู้จัก field นี้ เช่น CLI/test)
    session_owner_token: Optional[str] = None
    # pdf/xlsx: mirror ของ CreateTaskRequest ด้านบน — มีค่า = routes.py::generate_plan
    # คืน is_qa=True ทันที (ข้าม classify_intent()/LLM call ไปเลย เหมือน qa_summary intent
    # ปกติ) ให้ frontend ข้ามหน้าต่างอนุมัติ PLAN ไปตอบจากไฟล์ได้ทันที
    attached_file_name: Optional[str] = None
    attached_file_content_base64: Optional[str] = Field(
        default=None, max_length=_MAX_ATTACHED_FILE_BASE64_CHARS,
    )
    # W20 ("Context-Aware Implicit Execution"): the immediately-preceding turn's goal/reply in
    # this same conversation (if any) — frontend reads this from its own client-side history
    # right before submitting (see index.html::requestPlan()). Lets the planner resolve
    # anaphora like "เปิดให้หน่อย"/"play it" against whatever was actually recommended/discussed
    # last (e.g. a song name from a general-chat reply that never touched the browser, so it
    # was never captured by session.extracted_memory) instead of drafting a plan that only
    # navigates to a bare platform URL and stops. Both None (default) = first turn of a
    # conversation, or no prior turn to resolve against — behaves exactly as before this field
    # existed. See orchestrator.py::Orchestrator.generate_plan() / llm.py::generate_plan().
    previous_user_goal: Optional[str] = None
    previous_assistant_message: Optional[str] = None


class GeneratePlanResponse(BaseModel):
    plan: str
    is_qa: bool = False
    # W_procmem: ทั้ง 4 ฟิลด์นี้เป็น additive ล้วนๆ (default ว่างเปล่า/None) — consumer เดิม
    # ที่อ่านแค่ .plan/.is_qa ไม่ได้รับผลกระทบเลย ดู core/procedural_memory.py สำหรับ
    # ลำดับความสำคัญเต็มๆ (procedural template -> plan_memory -> LLM ร่างใหม่)
    # source บอกว่า plan ก้อนนี้มาจากไหน: "llm" (ร่างสดจาก LLM, ค่า default เดิม),
    # "plan_memory" (ข้อความแผนเดิมที่เคย confirm ไว้), "procedural_reuse"/
    # "procedural_adapt" (จาก core/procedural_memory.py — มี template_id/slot_values/
    # steps แนบมาด้วยเสมอตอนเป็น 2 ค่านี้ ให้ frontend ส่งต่อเข้า
    # POST /api/execute_plan ได้ตรงๆ เพื่อวิ่งผ่าน fast-path executor แทน slow loop)
    source: str = "llm"
    template_id: Optional[str] = None
    slot_values: Optional[dict] = None
    steps: Optional[list[dict]] = None



class ExecutePlanRequest(BaseModel):
    """W13: body ของ POST /api/execute_plan — เหมือน CreateTaskRequest ทุกฟิลด์ยกเว้น
    ไม่มี confirm_plan (อนุมัติไปแล้วจาก POST /api/generate_plan + user review ก่อนเรียก
    endpoint นี้) มี `plan` แทน"""

    url: str
    goal: str
    # แผนที่อนุมัติแล้ว (อาจแก้ไขข้อความมาก่อนจาก POST /api/generate_plan) — ส่งต่อเข้า
    # orchestrator.run_task(approved_plan=...) ตรงๆ ไม่ส่งมา (None) = ทำงานตาม goal ตรงๆ
    # ไม่มีแผนกำกับ (ข้ามเฟสวางแผนไปเลยก็ได้ถ้าไม่ต้องการ)
    plan: Optional[str] = None
    max_steps: int = 30
    provider: Optional[str] = None
    headless: Optional[bool] = None
    auto_approve: bool = False
    use_user_browser: bool = False
    tab_reuse_policy: Optional[str] = None
    session_id: Optional[str] = None
    # Security (SEC-4 follow-up): secret ที่ frontend generate คู่กับ session_id เอง (ดู
    # core/session_registry.py::BrowserSession.owner_token) — ต้องแนบมาด้วยทุกครั้งที่
    # session_id นี้เคยมีอยู่แล้ว ไม่งั้นถือว่าไม่ใช่เจ้าของ (SessionOwnershipError -> 403)
    # ไม่ส่งมา (None) ตอน session_id ยังไม่เคยมีอยู่จริง = สร้างใหม่โดยไม่มี token ป้องกันเลย
    # (เข้ากันได้กับ caller เดิมที่ยังไม่รู้จัก field นี้ เช่น CLI/test)
    session_owner_token: Optional[str] = None
    # W_procmem: mirror ของ GeneratePlanResponse ด้านบน — frontend ส่งต่อค่าที่ได้จาก
    # POST /api/generate_plan กลับมาตรงๆ ที่นี่ ถ้า execution_mode == "fastpath" และ
    # template_id/steps มีค่าจริงทั้งคู่ routes.py::execute_plan() จะวิ่งผ่าน
    # orchestrator.run_fastpath() แทน run_task(approved_plan=...) ปกติ — ต้องเป็น None/
    # "fastpath" ที่ตรงกับสิ่งที่ user เห็นตอน review เป๊ะเท่านั้น (ดู edited-plan safety
    # rule ใน index.html: แก้ไขข้อความแผนเองต้อง clear 3 ฟิลด์นี้ทิ้งเสมอ ไม่งั้นจะรัน
    # step ที่ไม่ตรงกับสิ่งที่ user อนุมัติจริง)
    template_id: Optional[str] = None
    slot_values: Optional[dict] = None
    steps: Optional[list[dict]] = None
    execution_mode: Optional[str] = None
    # pdf/xlsx: mirror ของ CreateTaskRequest — ดู comment ที่นั่นสำหรับรายละเอียดเต็ม
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
    # W_live: ค่า headless ที่ resolve แล้วของ task นี้ (ไม่มีทาง None) — frontend ใช้
    # ตัดสินใจว่าจะโชว์ live view (True) หรือซ่อนไปเลยเพราะ browser จริงเปิดโชว์อยู่แล้ว
    # (False) ดู index.html::renderLiveView()
    headless: bool = True
    # W20 (Task4 "User Chat Bubble File Attachments"): mirror ของ TaskRecord.attached_file_name
    # — ให้ frontend render attachment card/thumbnail เหนือ user bubble ของ turn นี้ได้ทั้งตอน
    # live chat และตอนโหลด historical turns ใหม่ (GET /tasks) หลัง refresh หน้าเว็บ
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
    """W14: body ของ GET /api/site-manual/status — ขับ banner "เว็บไซต์นี้ยังไม่มีคู่มือ"
    บน Test Console (ดู index.html)"""

    exists: bool
    version: Optional[int] = None


class LearnSiteRequest(BaseModel):
    url: str
    provider: Optional[str] = None
    # W15: login bootstrap ตอน crawl — กรอก+submit ครั้งเดียวตอนเจอฟอร์มที่มี password
    # field เพื่อผ่านหน้า login แล้วสำรวจต่อได้ (ดู site_learning/auto_login.py)
    # W17: ถ้าส่งมาทั้งคู่ จะถูกบันทึกลง credentials.json ของโดเมนนี้ด้วย (แยกไฟล์จาก
    # manual เอง — ดู storage.py::save_credentials) ให้ orchestrator ดึงไปใช้ auto-login
    # ตอนรัน task จริงในอนาคตได้ ไม่ต้องเรียนรู้/ล็อกอินซ้ำทุกครั้ง
    username: Optional[str] = None
    password: Optional[str] = None
    # W18: ถ้า username/password ไม่ได้ส่งมา (ผู้ใช้เลือก "ใช้บัญชีที่บันทึกไว้" บน UI แทน
    # การกรอกใหม่) และ flag นี้เป็น True — routes.py::learn_site() จะโหลด credential ที่
    # เก็บไว้แล้วของโดเมนนี้มาใช้ login bootstrap เอง (ไม่ต้องให้ frontend ส่งรหัสผ่านที่
    # ดึงกลับมาจาก backend ซ้ำ — GET .../credentials/status ไม่มีวันคืนรหัสผ่านจริงอยู่แล้ว)
    use_saved_credentials: bool = False


class SaveCredentialsRequest(BaseModel):
    """W17: body ของ POST /api/site-manual/{domain}/credentials — บันทึก/แก้ไข
    credential ของโดเมนนี้ตรงๆ โดยไม่ต้อง crawl ทั้งเว็บใหม่"""

    username: str
    password: str


class CredentialsStatusResponse(BaseModel):
    exists: bool


class LearnCreatedResponse(BaseModel):
    learn_id: str
    status: str


class LearnCredentialsRequest(BaseModel):
    """W23: body ของ POST /api/site-manual/learn/{learn_id}/credentials — ตอบ
    "credentials_needed" event ที่ได้จาก GET /api/site-manual/learn/{learn_id}/stream
    (crawl เจอหน้า login ระหว่างเรียนรู้เว็บไซต์ แต่ยังไม่มี credential เก็บไว้ก่อนเลย —
    ดู site_learning/crawler.py::crawl_site() พารามิเตอร์ on_credentials_needed)

    request_id ต้องตรงกับที่ event ส่งมา ไม่งั้นถือว่าหมดอายุ/ตอบไปแล้ว (ดู
    LearnManager.resolve_credentials()) — username/password ปล่อยว่างทั้งคู่ (None) =
    ผู้ใช้เลือกข้าม ("ไม่ต้อง login") ให้ crawl สำรวจต่อโดยไม่ผ่านหน้านี้แทนที่จะรอตลอดไป"""

    request_id: str
    username: Optional[str] = None
    password: Optional[str] = None


class RelearnPageRequest(BaseModel):
    """W14: body ของ POST /api/site-manual/{domain}/relearn-page — selector-repair:
    สำรวจเฉพาะหน้าเดียว (url) ใหม่แทนที่จะ crawl ทั้งเว็บซ้ำ"""

    url: str
    provider: Optional[str] = None


class RelearnPageResponse(BaseModel):
    version: int


class RespondRequest(BaseModel):
    """W10[B]: body ของ POST /tasks/{id}/respond — ผูกกับ approval_request event ที่
    ได้จาก GET /tasks/{id}/stream (request_id ต้องตรงกับที่ event ส่งมา ไม่งั้นถือว่า
    หมดอายุ/ตอบไปแล้ว ดู TaskManager.resolve_approval())"""

    request_id: str
    approved: bool
    # W10[F]: ถ้า request นี้เป็น confirm_plan (ไม่ใช่ permission prompt ของ action ทั่วไป)
    # user แก้ไขข้อความแผนเองก่อนกด Confirm ได้ — ส่งมาก็ต่อเมื่อมีการแก้ไขจริง (None =
    # ใช้แผนเดิมที่ AI ร่างไว้ไม่แก้) ดู TaskManager.resolve_approval()/orchestrator.py::
    # _confirm_plan() สำหรับตำแหน่งที่ใช้ค่านี้จริง
    edited_plan: Optional[str] = None
    # W_resume ("Mid-Task Input Request"): ถ้า request นี้เป็น request_user_input (ไม่ใช่
    # permission prompt/confirm_plan) นี่คือคำตอบที่ user พิมพ์ตอบคำถามที่ agent ถามกลาง
    # task (เช่น "รหัสผ่านใหม่ที่ต้องการตั้งคืออะไร") — ส่งมาก็ต่อเมื่อ user ตอบจริง (None =
    # ปฏิเสธ/ข้าม ดู TaskManager.resolve_approval()/orchestrator.py::_request_user_input()
    # สำหรับตำแหน่งที่ใช้ค่านี้จริง) เหมือน edited_plan ข้างบนทุกประการ แค่คนละ cmd type
    answer_text: Optional[str] = None


class OpenAILoginStartResponse(BaseModel):
    """W_openai_oauth: response ของ POST /api/auth/openai/login/start — frontend เปิด
    authorize_url ในแท็บ/หน้าต่างใหม่ให้ user login เอง แล้ว poll GET .../login/status ด้วย
    login_id นี้ต่อ (token exchange เกิดขึ้นใน background task ไม่ใช่ response นี้ — ดู
    core/openai_oauth.py::start_login_flow())"""

    authorize_url: str
    login_id: str


class OpenAILoginStatusResponse(BaseModel):
    """W_openai_oauth: response ของ GET /api/auth/openai/login/status?login_id= —
    status: "pending"|"linked"|"error" (ไม่มี login_id ที่รู้จัก = 404 ที่ route handler)"""

    status: str
    error: Optional[str] = None
    email: Optional[str] = None
    plan_type: Optional[str] = None


class OpenAIAuthStatusResponse(BaseModel):
    """W_openai_oauth: response ของ GET /api/auth/openai/status — สถานะ link ปัจจุบัน
    (ไม่ผูกกับ login attempt ไหนเป็นพิเศษ) ใช้ตอนโหลดหน้าเพื่อรู้ว่าจะ enable provider
    "openai" ใน dropdown ได้ไหม ไม่คืน token จริงออกไปเลย"""

    linked: bool
    email: Optional[str] = None
    plan_type: Optional[str] = None

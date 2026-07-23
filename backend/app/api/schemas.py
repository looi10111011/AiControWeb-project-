from typing import Optional

from pydantic import BaseModel, Field


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


class GeneratePlanResponse(BaseModel):
    plan: str


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


class TaskCreatedResponse(BaseModel):
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

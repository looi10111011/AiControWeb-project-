from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")


    anthropic_api_key: str = ""
    gemini_api_key: str = ""
    groq_api_key: str = ""

    primary_llm_provider: str = "anthropic"
    fallback_llm_provider: str = "gemini"
    llm_provider: str = "anthropic"
    anthropic_model: str = "claude-haiku-4-5-20251001"
    groq_model: str = "llama-3.3-70b-versatile"
    gemini_model: str = "gemini-flash-lite-latest"

    chroma_persist_dir: str = "./data/chroma"
    chroma_collection_name: str = "manuals"
    chroma_long_term_collection_name: str = "long_term_memory"

    # W49: baseline สำหรับงาน token/cost optimization — บันทึก token usage จริงของทุก
    # task ที่จบสำเร็จ (JSON Lines, append-only) ไว้วัด baseline ก่อนเริ่มปรับ pipeline
    # (ดู api/task_manager.py::_log_token_usage()) แยกจาก long_term_memory เพราะอันนั้น
    # เก็บไว้ให้ agent recall เอง ไม่ใช่ไว้ให้ developer วิเคราะห์ cost
    token_usage_log_path: str = "./data/token_usage.jsonl"

    browser_headless: bool = True

    # Security: ทุก route ใน api_router (ดู api/routes.py::verify_api_key) ต้องแนบค่านี้ผ่าน
    # header "X-API-Key" (หรือ query param "?api_key=" สำหรับ SSE stream ที่ EventSource ตั้ง
    # header เองไม่ได้) มิฉะนั้นได้ 401 — ไม่ตั้งค่าใน .env (None, default) = auth ปิดทั้งหมด
    # สำหรับ local dev เท่านั้น (ต้องตั้งค่านี้จริงก่อน deploy ที่เข้าถึงได้จากนอกเครื่อง)
    api_key: Optional[str] = None

    # Security 1.2 (SSRF): permission/rules.py::classify_action() hard-blocks goto ไปยัง
    # private/internal IP (cloud metadata, LAN ภายใน ฯลฯ) เสมอไม่ว่า config อื่นจะว่าไง —
    # เปิดตัวนี้เฉพาะ dev ที่ตั้งใจทดสอบเว็บ local จริงๆ เท่านั้น (default ปิด ปลอดภัยสุด)
    allow_internal_navigation: bool = False

    api_host: str = "127.0.0.1"
    api_port: int = 8000
    # W10[A]: จำนวน browser instance ที่ BrowserPool เปิดค้างไว้ตอน API server startup
    # (ดู core/browser_pool.py) — task ที่เกินโควตานี้พร้อมกันจะรอคิวจนกว่าจะมีตัวว่าง
    browser_pool_size: int = 2

    # W10[E]: เวลาสูงสุดที่ ask_user_func (permission prompt + plan confirmation, ดู
    # routes.py::_make_ask_user_func) จะรอ human ตอบก่อน "หมดเวลา" แล้วถือว่าถูกปฏิเสธ
    # อัตโนมัติ — ไม่มี timeout เดิม (รอเฉยๆ ตลอดกาล) ทำให้ task ที่ user ปิดแท็บทิ้งกลาง
    # คันตอนรอ confirm plan ยึด browser จาก pool ไว้ (หรือถ้า pool เต็มแล้ว ไปต่อคิวรอ
    # browser ที่ไม่มีวันว่าง) ค้างตลอดไป กัด quota ของ browser_pool_size ไปเรื่อยๆ จนกว่า
    # task ใหม่ๆ ทุกตัวจะรอคิวไม่รู้จบ (อาการที่เห็นจริง: "plan ไม่ขึ้นเลย" เพราะ task ใหม่
    # ค้างรอ browser ว่างอยู่ใน pool.acquire() ไม่ทันได้ไปถึงขั้นตอน goto/generate_plan
    # ด้วยซ้ำ) — ตั้ง default ไว้ไม่นานเกินไป (5 นาที) พอให้ user อ่านแผนจริงๆ ได้ทัน แต่ไม่
    # ยึด pool ค้างเป็นชั่วโมงถ้าลืมแท็บทิ้งไว้
    approval_timeout_seconds: float = 300.0

    # W_planhang: POST /api/generate_plan (routes.py::generate_plan) เรียก LLM ตรงๆ
    # (classify_intent + generate_plan) โดยไม่มี timeout ใดๆ เลยเดิม — client
    # (AsyncAnthropic/Groq openai-compatible/Gemini) ทุกตัวไม่ได้ตั้ง timeout เอง ถ้า
    # provider ตอบช้าผิดปกติ (rate limit/network) endpoint นี้จะค้างเงียบๆ ไม่มีวันจบ
    # (อาการจริงที่ user เจอ: หน้าจอค้างที่ "Generating plan…" ไม่ error ไม่ timeout เลย)
    # ต่างจาก approval_timeout_seconds ด้านบนที่รอ "คน" ตอบ (รอนานได้) endpoint นี้เป็น
    # synchronous request-response เดียว (ไม่มี SSE progress ระหว่างรอ) ต้อง fail เร็ว
    # พอให้ user รู้ว่ามีปัญหาแล้วลองใหม่ได้ ไม่ใช่ปล่อยให้ composer ดูค้างตลอดไป
    plan_generation_timeout_seconds: float = 45.0

    # Real-user-browser mode (CDP connect, ดู core/user_browser.py): user เปิด Chrome
    # เองล่วงหน้าด้วย --remote-debugging-port ก่อนรัน agent ในโหมดนี้ — agent ไม่ launch
    # Chrome ให้เอง (ต่างจาก _launch_chromium()/BrowserPool ปกติ) แค่ต่อเข้าไปผ่าน CDP
    # เพื่อใช้ session/cookie ที่ login ไว้แล้วจริง (เช่น mail)
    user_browser_cdp_url: str = "http://localhost:9222"
    # "ask" = ถาม user ก่อนใช้ tab ที่เปิดค้างไว้แล้วตรงโดเมนเป้าหมาย (default, ปลอดภัย
    # สุด) "always_new_tab" = เปิด tab ใหม่เสมอไม่แตะ tab เดิม "always_reuse" = ใช้ tab
    # เดิมเลยไม่ถาม (ดู core/user_browser.py::resolve_target_page)
    user_browser_tab_reuse_policy: str = "ask"

    # W14: Website Learning — manual ที่ crawl มาอัตโนมัติ (ดู backend/app/site_learning/)
    # เก็บเป็น JSON บนดิสก์ล้วนๆ แยกต่างหากสมบูรณ์จาก backend/app/rag/ (คู่มือที่ user
    # อัปโหลดเอง เก็บใน ChromaDB) — ตั้งใจไม่ใช้ path "./data/manuals" เดิมเพราะชื่อนั้น
    # ถูก chroma_collection_name="manuals" ข้างบนจับจองความหมายไว้แล้ว
    site_manuals_dir: str = "./data/site_manuals"
    # จำกัดจำนวนหน้าสูงสุดต่อการ crawl 1 ครั้ง (ไม่มีในสเปคเดิม แต่จำเป็นกันเว็บใหญ่มาก/
    # ลิงก์วนซ้ำไม่รู้จบทำให้ crawl ไม่มีวันจบ)
    site_learning_max_pages: int = 40
    # W16: จำกัดจำนวนปุ่ม "ปลอดภัย" ที่ crawler จะไล่กดต่อ 1 หน้า (ดู
    # crawler.py::_explore_buttons) — หน้าที่มีปุ่มเข้าข่ายปลอดภัยเยอะผิดปกติ (เช่น list
    # ยาวๆ ที่ทุกแถวมีปุ่ม "View") ไม่ควรไล่กดทุกอันจนใช้เวลาเป็นชั่วโมง
    site_learning_max_buttons_per_page: int = 15
    # W36: เพดานเฉพาะปุ่ม tier="core" (ดู site_learning/safety.py::classify_button_tier,
    # crawler.py::_explore_buttons) ต่อ 1 หน้า — แยกจาก site_learning_max_buttons_per_page
    # ข้างบน (เพดานรวมทุก tier ที่ผ่านเข้ามาถึงตอนนี้) เพราะ "core" function classification
    # ตั้งใจลดจำนวนปุ่มที่ไล่กดต่อหน้าลงอีกชั้น (แก้ปัญหา self-learning กดปุ่มเยอะเกินความ
    # จำเป็นบนหน้าที่มีปุ่มฟังก์ชันหลักเยอะผิดปกติ เช่น search/filter/sort/add-to-cart หลาย
    # ตัวพร้อมกัน) เกินเพดานนี้จะตัดเอาแค่ top-K ตาม priority (form-submit > exact keyword
    # match > partial match — ดู safety.button_core_priority) ตั้งน้อยเกินไปจะพลาดฟังก์ชัน
    # หลักบางอย่างของหน้าที่มีปุ่ม core เยอะจริงๆ (ไม่ใช่ noise) ตั้งมากเกินไปจะไม่ช่วยลดปุ่ม
    # ที่ไล่กดเท่าที่ควร — ไม่กระทบปุ่ม tier="nav" เลย (ยังไล่กดครบตาม
    # site_learning_max_buttons_per_page เดิมด้านบนเหมือนที่ไม่มีฟีเจอร์นี้)
    site_learning_max_core_buttons_per_page: int = 8
    # W24: ค่าพวกนี้เดิมเป็น magic number ฝังในโค้ดล้วนๆ (retry=0 เสมอ ไม่มี retry เลย) —
    # ย้ายมาเป็น setting ที่ปรับได้จาก .env ตรงๆ เพราะ "หากตั้งไว้น้อยเกินไป Agent จะหยุด
    # เร็ว" เป็นความเสี่ยงจริง (เว็บที่ network ช้า/element render ช้าต้องการ retry มากกว่า
    # เว็บทดสอบทั่วไป) — ค่า default ที่เลือกไว้เป็นค่ากลางๆ ที่ไม่ทำให้ crawl ช้าเกินไปแต่
    # กันความล้มเหลวชั่วคราว (transient — DOM ยังไม่นิ่ง/network กระตุก) ได้ระดับหนึ่ง
    site_learning_goto_retries: int = 2  # รวมครั้งแรกเป็นลองทั้งหมด retries+1 ครั้งต่อหน้า
    site_learning_click_retries: int = 2  # เหมือนกันแต่สำหรับกดปุ่มระหว่างไล่สำรวจ
    site_learning_retry_backoff_ms: int = 500  # หน่วงก่อน retry แต่ละครั้ง
    # W24: infinite scroll/lazy-loaded content — เลื่อนจอลงจนสุด scroll ไม่ขยับอีกแล้ว
    # (หรือครบจำนวนครั้งนี้) ก่อน extract โครงสร้างหน้า (ดู
    # crawler.py::_reveal_dynamic_content) เว็บที่โหลดทีละน้อยมากๆ (เช่น 1 การ์ดต่อ scroll)
    # อาจต้องเพิ่มค่านี้ขึ้นถ้าพบว่า manual ที่ได้ไม่ครบเนื้อหาทั้งหมด
    site_learning_max_scroll_attempts: int = 6
    site_learning_scroll_wait_ms: int = 350
    # W28: ปุ่มที่ "label+role เดียวกัน" โผล่ซ้ำข้ามหลายหน้า (เช่น ไอคอนค้นหาบน header ของ
    # ทุกหน้า, ปุ่ม "Previous/Next video" บน player ของทุกคลิป) จะถูกไล่กดได้สูงสุดกี่ครั้ง
    # รวมทั้ง crawl (นับข้าม URL ไม่ใช่แค่ในหน้าเดียว — ดู crawler.py::_button_signature) —
    # ก่อนหน้านี้ไม่มีเพดานนี้เลย ทำให้เว็บที่มีเนื้อหาไม่จำกัด (เช่น YouTube Shorts ที่ปุ่ม
    # "Next video" พาไป URL ใหม่ไม่รู้จบ) กิน max_pages budget ทั้งหมดไปกับการไล่กดปุ่มเดิม
    # ซ้ำๆ ข้ามหน้า ไม่เคยย้อนกลับไปสำรวจส่วนอื่นของเว็บเลย ค่า 1 = กดแต่ละปุ่มที่เหมือนกัน
    # ได้แค่ครั้งเดียวตลอดทั้ง crawl (เข้มสุด กัน loop เด็ดขาด แลกกับ coverage ที่ลดลงถ้าปุ่ม
    # label เดียวกันจริงๆ ใช้งานต่างกันในแต่ละหมวดของเว็บ เช่น "View" ในตาราง Products กับ
    # ตาราง Orders — คนละความหมายแต่ label เดียวกัน จะถูกไล่กดแค่อันแรกอันเดียว)
    site_learning_max_repeat_button_clicks: int = 1

    # W20: Plan Memory (ดู core/plan_memory.py) — แผนที่ user "Confirm" แล้ว เก็บใน
    # ChromaDB collection แยกต่างหาก (persist_dir เดียวกับ chroma_persist_dir ข้างบน
    # แค่คนละ collection name) ค้นด้วย semantic search ต่อ (domain, goal) แทน exact
    # text match เดิมของ core/plan_store.py (W19 — ถูกแทนที่ทั้งระบบด้วยตัวนี้)
    chroma_plan_memory_collection_name: str = "plan_memory"
    # ระยะห่าง (cosine distance, ยิ่งน้อยยิ่งใกล้เคียงกัน — 0 = เหมือนกันเป๊ะ) สูงสุดที่ยัง
    # ถือว่า "ตรงพอ" จะ reuse แผนเดิมได้ — คาลิเบรตจากการวัดจริงกับ
    # DefaultEmbeddingFunction (all-MiniLM-L6-v2): "Login" vs "Sign in" ~0.21, vs "log me
    # in please" ~0.31, vs intent ที่ไม่เกี่ยวข้องเลยเช่น "checkout and pay" ~0.78 — มี
    # margin กว้างพอสำหรับภาษาเดียวกัน (0.5 คั่นตรงกลางได้ชัดเจน) แต่โมเดลนี้เป็น
    # English-centric จับคู่ข้ามภาษาไทย-อังกฤษได้ไม่แม่น (เช่น "เข้าสู่ระบบ" วัดจริงได้
    # ~0.77 ใกล้เคียง intent ที่ไม่เกี่ยวข้องเลย) เป็นข้อจำกัดของ embedding model เอง ไม่ใช่
    # threshold ตั้งผิด — ปรับค่านี้ได้ถ้าพบว่า reuse ผิด/ไม่ยอม reuse ที่ควร reuse บ่อยไป
    plan_memory_max_distance: float = 0.5

    # W_procmem: Procedural Memory (ดู core/procedural_memory.py, core/dom_locator.py,
    # core/fastpath_executor.py) — ต่อยอดจาก Plan Memory ด้านบน: Plan Memory ข้าม LLM
    # call แค่ตอน "ร่างแผน" (1 call ต่อ task) ส่วนระบบนี้เก็บ template แบบมีโครงสร้าง
    # (ordered steps + stable locator + {{slot}} placeholder แทนค่าจริงเสมอ) ที่ทำให้
    # ข้าม LLM call ได้ทั้ง step-by-step execution loop ไม่ใช่แค่ตอนร่างแผน — เก็บใน
    # ChromaDB collection แยกต่างหาก (คนละ collection กับ plan_memory ข้างบน)
    #
    # ลำดับความสำคัญตอนหา plan ให้ user (ดู routes.py::generate_plan): procedural
    # template ก่อน (ถ้าเปิดและ match) -> plan_memory (ข้อความแผนเดิม) -> LLM ร่างใหม่
    # สดๆ — plan_memory "ไม่" ถูกแทนที่ ยังทำงานเป็น fallback ชั้นถัดไปเหมือนเดิมทุก
    # ประการ (คนละบทบาทกัน: plan_memory เก็บข้อความแผนดิบ, ตัวนี้เก็บ step ที่รันได้จริง)
    enable_procedural_memory_capture: bool = True
    # เปิดแค่ "ฝั่งเขียน" (Abstractor หลัง task สำเร็จ) — additive ล้วนๆ ไม่มีอะไรอ่านจาก
    # collection นี้เลยจนกว่า enable_procedural_memory ด้านล่างจะเปิด ปลอดภัยที่จะเปิด
    # ไว้ default (True) ให้ template เริ่มสะสมได้ทันทีโดยไม่กระทบ behavior เดิมเลย
    enable_procedural_memory: bool = False
    # master flag ของ "ฝั่งอ่าน" (Memory-augmented Planner + fast-path executor) — ปิด
    # ไว้ default จนกว่าจะ validate คุณภาพ template/locator resolution บนเว็บจริงก่อน
    # (ดู Phase 4 ใน implementation plan) ค่อยเปิดเป็น default True
    chroma_procedural_memory_collection_name: str = "procedural_memory"
    procedural_memory_max_candidates: int = 3
    procedural_memory_min_confidence: float = 0.6
    # จำนวนครั้งสูงสุดที่ยอมให้ Repair module แก้ step เดียวกันซ้ำก่อนจะยอม escalate ไป
    # เต็ม slow-path loop (ดู fastpath_executor.py) — กันไม่ให้วนซ่อม step เดิมไม่รู้จบ
    # ถ้า locator เปลี่ยนไปมากจน Repair เดาไม่ถูกสักที (เทียบ pattern เดียวกับ
    # _ACTION_RETRIES ใน actions.py)
    procedural_memory_max_repair_attempts: int = 2

    # W46: perception.py::fuzzy_find() — ใช้ตอน read_page_data (actions.py) หา exact match ใน
    # ตาราง/list ไม่เจอ (เช่น user พิมพ์ชื่อผิดเล็กน้อย "Cierra Vaga" แทน "Cierra Vega") ค่า
    # นี้คือ difflib.SequenceMatcher.ratio() ขั้นต่ำที่ยังยอมรับว่า "ใกล้เคียงพอ" จะเสนอเป็น
    # fuzzy match กลับไป (1.0 = เหมือนกันเป๊ะ) — trade-off สำคัญ: ตั้งต่ำเกินไปจะ
    # false-positive จับคนละคน/คนละชื่อที่บังเอิญคล้ายกันเป็นตัวเดียวกัน (อันตรายกว่า เพราะ
    # agent จะตอบข้อมูลผิดคนให้ user แบบมั่นใจโดยไม่รู้ตัว) ตั้งสูงเกินไปจะพลาดคำที่พิมพ์ผิด
    # เล็กน้อยจริงๆ (false negative — กลับไปตอบ "ไม่พบ" ทั้งที่มีจริง) ค่า default นี้กลางๆ
    # พอให้ผ่านการพิมพ์ผิด 1-2 ตัวอักษรในคำสั้นๆ ได้ แต่ยังกันชื่อคนละคนที่ขึ้นต้น/ลงท้าย
    # คล้ายกันได้ระดับหนึ่ง ปรับได้ถ้าพบว่า fuzzy match หลวม/เข้มไปสำหรับข้อมูลจริงของ user
    agent_fuzzy_match_threshold: float = 0.75

    # W19 (ดู W19.txt ข้อ 8 "Semantic Redundancy Evaluator", core/llm.py::
    # evaluate_semantic_redundancy) — เพิ่ม LLM call แยก 1 ครั้งต่อ step (ก่อน dispatch
    # จริงใน orchestrator.py) ประเมินว่า action ที่เลือกไว้แล้วมีประโยชน์ต่อ goal จริงไหม
    # ต่างจาก state_filter.py (ข้อ 6, deterministic ล้วนๆ ไม่มี flag เพราะไม่มีต้นทุน LLM)
    # — ปิดไว้ default (เหมือน enable_procedural_memory) จนกว่าจะ validate คุณภาพ/ต้นทุน
    # latency เพิ่มต่อ step บนงานจริงก่อน ค่อยพิจารณาเปิดเป็น default True
    enable_semantic_redundancy_check: bool = False

    # W19-2 (ดู core/llm.py::evaluate_safety_and_performance) — "โมดูลที่ 4" แบบ additive:
    # รวม redundancy check (เหมือน enable_semantic_redundancy_check ด้านบน) + permission
    # check (เหมือน permission/rules.py::classify_action) เป็น LLM call เดียว ประหยัด
    # round-trip กว่าเรียกแยก 2 ครั้ง — เปิดพร้อมกับ enable_semantic_redundancy_check ได้
    # แค่ orchestrator.py จะเลือกใช้ตัวนี้แทน (ไม่เรียกซ้ำสอง call สำหรับ redundancy)
    # permission_evaluation ของตัวนี้เป็นแค่ "เพิ่มความระมัดระวัง" เท่านั้น (escalate-only
    # ผ่าน manual_guidance เข้า classify_action() ที่ยังเป็นผู้ตัดสินสุดท้ายเสมอ ดู
    # orchestrator.py) ไม่มีทางลดระดับความเสี่ยงที่ classify_action() ตัดสินไปแล้วได้เลย —
    # ปิดไว้ default เหมือนโมดูล LLM ตัวอื่นในกลุ่มนี้ จนกว่าจะ validate คุณภาพก่อน
    enable_middleware_evaluator: bool = False

    # W19-3 (ดู core/llm.py::generate_persona_message) — "Voice & Persona Interface":
    # แปลงสถานะ agent ดิบๆ เป็นข้อความไทยธรรมชาติแบบผู้ช่วยส่วนตัว ให้ UI โชว์แทน raw log —
    # เป็นแค่ presentation layer เสริม (ไม่กระทบ control flow ของ agent loop เลย ต่างจาก 3
    # โมดูลก่อนหน้าที่ skip/escalate ได้) เรียกเฉพาะตอนจบ task (COMPLETED/FAILED — จุดที่
    # ความถี่ต่ำสุด/คุ้มค่าที่สุด) ไม่ได้เรียกทุก browser action step — ปิดไว้ default เหมือน
    # โมดูล LLM ตัวอื่นในกลุ่มนี้ จนกว่าจะ validate โทน/คุณภาพข้อความก่อน
    enable_persona_voice: bool = False

    # W41 (ดู core/orchestrator.py::_STEP_PACING_DELAY_SECONDS สำหรับเหตุผลเต็มของ pacing
    # นี้เอง): ระยะห่างต่ำสุด (วินาที) ที่ต้องการระหว่างการเรียก next_action() (LLM) 2 ครั้ง
    # ติดกัน กันยิง LLM API ถี่เกิน free-tier quota ต่อนาที (RPM) — ย้ายจาก module constant
    # เดิม (hardcode 3 เสมอ) มาเป็น setting เพื่อให้ปรับได้ตาม provider/tier ที่ใช้จริงโดย
    # ไม่ต้องแก้โค้ด (เช่น tier ที่จ่ายเงินแล้วมี RPM สูงกว่า free-tier มาก ปรับให้ต่ำลงได้)
    step_pacing_delay_seconds: float = 3.0

    # W67: nav-fastpath (ดู core/fastpath_executor.py::execute_navigation,
    # core/orchestrator.py::run_task) — W66 เปิดใช้ได้แค่ manual trigger (ต้องระบุ
    # nav_target_page_query เข้ามาเอง) ตัวนี้เปิดให้ orchestrator ตัดสินใจเองจาก goal
    # โดยตรงเมื่อไม่ได้ระบุ query มา (auto-decide) — เปิด default True เพราะ fail-safe
    # อยู่แล้วทุกจุด (ไม่มี manual/ไม่ match พอ -> fallback ไป URL เดิม/LLM loop ปกติเงียบๆ
    # ไม่ throw ไม่แย่กว่าเดิม) ประโยชน์ (ประหยัด LLM call ต่อ step ระหว่างเดินทาง) มากกว่า
    # ความเสี่ยง
    enable_nav_fastpath_auto_decide: bool = True
    # threshold เข้มกว่า default 1 ของ find_matching_page() เดิม (ใช้กับ Strict Guided
    # Plan context ที่แค่โชว์ข้อความให้ LLM อ่านเฉยๆ match หลวมๆ ก็ยังปลอดภัย) —
    # auto-decide ต้องมั่นใจกว่าเพราะจะลงมือคลิกจริงตาม nav path ที่ match ได้ ไม่ใช่แค่ให้
    # LLM อ่านประกอบการตัดสินใจ
    nav_fastpath_min_match_score: int = 2


settings = Settings()


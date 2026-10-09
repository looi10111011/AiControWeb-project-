from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    anthropic_api_key: str = ""
    gemini_api_key: str = ""
    groq_api_key: str = ""

    # W_openai_oauth: provider "openai" ไม่มี API key — ใช้ OAuth (Codex CLI public client_id) ดึงโควต้า
    # ChatGPT ของ operator (risk disclosure ใน core/openai_oauth.py, ตกลงกับ user 2026-08-17,
    # single-tenant) token เก็บเข้ารหัส Fernet ในไฟล์ local ไม่ใช่ใน .env
    # เป็น default provider ตามที่ user ต้องการ — ยังไม่ login จะ raise OAuthLoginRequired เป็น task
    # failure ที่อ่านเข้าใจได้ (ไม่มี auto-fallback); เปลี่ยนได้ด้วย LLM_PROVIDER ใน .env
    primary_llm_provider: str = "openai"
    fallback_llm_provider: str = "gemini"
    llm_provider: str = "openai"
    anthropic_model: str = "claude-haiku-4-5-20251001"
    groq_model: str = "llama-3.3-70b-versatile"
    gemini_model: str = "gemini-flash-lite-latest"
    # W_openai_oauth: เรียกผ่าน chatgpt.com/backend-api/codex — ใช้ได้เฉพาะชื่อ model ที่ endpoint นั้นรับ
    # W_codex_model_retired (2026-09-10): endpoint เลิกรับ "gpt-5.4-mini" (400 "...not supported when
    # using Codex with a ChatGPT account") โพรบ 15 ชื่อผ่านแค่ "gpt-5.5" — เป็นทะเบียนชื่อโมเดล ไม่ใช่สิทธิ์
    # บัญชี; gpt-5.5 ปฏิเสธ max_output_tokens (llm.py มี fallback) เจอ 400 แบบนี้อีกให้โพรบทีละชื่อก่อน
    openai_model: str = "gpt-5.5"

    # W_openai_throttle_backoff (2026-09-10): บัญชีไม่จ่ายเงินยิงได้ 6-9 call ติดกันแล้วโดนตัด ~1-2 นาที
    # แล้วกลับมาเอง — เดิมไม่รอเลย task ยาว (long_flow 13 call) ไม่มีทางจบ
    # ผลรวมการรอ (15+30+60=105s) ต้องน้อยกว่า llm_step_timeout_seconds (180s) — เพิ่ม retry ต้องขยายค่านั้นด้วย
    openai_throttle_max_retries: int = 3
    openai_throttle_base_wait_seconds: float = 15.0

    # หน่วงระหว่าง task ของ eval suite (0 = ไม่หน่วง) — ลดโอกาสโดนตัด แต่ไม่พอเดี่ยวๆ เพราะ
    # long_flow ยิง 13 call ภายใน task เดียว ต้องมี backoff ข้างบนด้วย
    eval_task_delay_seconds: float = 0.0

    # W_eval: release gate (core/release_gate.py) เขียน JSON ต่อ run ที่ dir นี้ tag ด้วย commit + model
    release_gate_results_dir: str = "./data/eval_results"
    # % regress สูงสุดต่อ metric ก่อน fail gate (เผื่อความผันผวนของ LLM)
    release_gate_max_regression_pct: float = 10.0

    # W_gate_noise_floor: baseline = median ของ N รันหลังสุด — gate รันซ้ำบน commit เดิมยัง swing เกิน
    # 10% เอง (success_rate -14.3%, p95 +49.5%, tokens 124k-211k) 5 = ทนรันดวงดี/ร้ายได้ 2 ตัว
    # W_gate_is_noisy (2026-09-09): commit 3ddb18a ได้ 12/15 แล้ว 15/15 โดยไม่แตะโค้ด — gate จึงรันซ้ำ
    # แล้วตัดสินด้วย median ของ success_rate
    release_gate_repeats: int = 3
    release_gate_baseline_runs: int = 5

    chroma_persist_dir: str = "./data/chroma"
    chroma_collection_name: str = "manuals"
    chroma_long_term_collection_name: str = "long_term_memory"

    # W49: token usage จริง 1 บรรทัดต่อ task (JSONL append-only) ไว้วัด cost — แยกจาก
    # long_term_memory ที่มีไว้ให้ agent recall (api/task_manager.py::_log_token_usage())
    token_usage_log_path: str = "./data/token_usage.jsonl"

    # W_step_trace: 1 บรรทัดต่อ step + failure taxonomy + timing — token_usage ตอบไม่ได้ว่าพังที่ step
    # ไหน/เวลาหมดกับอะไร; เขียนครั้งเดียวตอน task จบ ไม่แทรก disk I/O กลาง loop
    step_trace_log_path: str = "./data/step_trace.jsonl"

    # W_extract_row_cap (P4.5): ผลตาราง read_page_data สะสมใน messages (task "search_no_results" โต
    # 11.5k -> 43.5k token ใน 8 step) — ตัดแถวได้ปลอดภัยเพราะ W_deterministic_count นับจริงแนบไปแล้ว
    # และข้อความบอกตรงๆ ว่าตัดไปกี่แถว
    read_page_data_max_rows: int = 60

    # W_snapshot_cap (P3.3): หน้าทั่วไปมี element 400-1500 ตัว = 4k-15k token/step — ตัดเฉพาะ
    # text_repr ที่ส่ง LLM (guard ใน orchestrator ยังเห็น elements ครบ) เรียง in_viewport ก่อน (W50)
    # จึงตัด element ที่ต้อง scroll ก่อนเสมอ
    snapshot_max_elements: int = 150

    browser_headless: bool = True

    # Security: ทุก route ใน api_router ต้องส่ง header "X-API-Key" (SSE ใช้ ?ticket= แทน) — None =
    # auth ปิด เฉพาะ loopback (ดู api/routes.py::verify_api_key)
    api_key: Optional[str] = None

    # Security 1.2 (SSRF): classify_action() hard-block goto ไป private/internal IP — เปิดเฉพาะ dev
    # ที่ทดสอบเว็บ local (default ปิด)
    allow_internal_navigation: bool = False

    # W_openai_oauth: client_id/endpoint เป็นค่าคงที่ของ Codex CLI; port/timeout/cadence ปรับได้
    # loopback callback server ชั่วคราว ลอง port หลักก่อน fallback เมื่อ bind ไม่ได้ (_bind_loopback_server)
    openai_oauth_callback_port: int = 1455
    openai_oauth_callback_port_fallback: int = 1457
    # เวลารอ human sign-in แล้ว redirect กลับก่อนถือว่า timeout
    openai_oauth_login_timeout_seconds: float = 300.0
    # cadence refresh ตาม Codex CLI (_refresh_if_needed): ก่อนหมดอายุ N วินาที หรือ refresh ล่าสุดเกิน N วัน
    openai_oauth_refresh_before_expiry_seconds: float = 300.0
    openai_oauth_refresh_max_age_days: float = 8.0

    api_host: str = "127.0.0.1"
    api_port: int = 8000
    # W10[A]: จำนวน browser ใน BrowserPool — task เกินโควตารอคิว
    browser_pool_size: int = 2

    # W10[E]: เวลารอ human ตอบ approval/confirm plan ก่อนปฏิเสธอัตโนมัติ — เดิมรอตลอดกาล task ที่แท็บ
    # ถูกปิดยึด browser pool ไว้จน task ใหม่ค้างใน pool.acquire() (อาการจริง: "plan ไม่ขึ้นเลย")
    approval_timeout_seconds: float = 300.0

    # W_planhang: generate_plan เรียก LLM โดยไม่มี timeout — provider ช้าทำให้ UI ค้างที่
    # "Generating plan…" ตลอดไป (อาการจริง); เป็น request-response เดียวต้อง fail เร็ว
    plan_generation_timeout_seconds: float = 45.0

    # W_steptimeout: next_action() ในลูปหลักไม่มีขอบเขตเวลา — SSE stream ของ OpenAI OAuth ค้างได้ตลอดไป
    # กินสล็อต pool; ตั้งสูงกว่า plan เพราะ prompt ใหญ่กว่า timeout = step ล้มเหลว ไม่ใช่ task ตาย (W_loop_crash)
    llm_step_timeout_seconds: float = 180.0

    # Real-user-browser mode (core/user_browser.py): user เปิด Chrome ด้วย --remote-debugging-port เอง
    user_browser_cdp_url: str = "http://localhost:9222"
    # "ask" (default) | "always_new_tab" | "always_reuse" — ดู user_browser.py::resolve_target_page
    user_browser_tab_reuse_policy: str = "ask"

    # W14: Website Learning manual เป็น JSON บนดิสก์ แยกจาก rag/ (ChromaDB) — ไม่ใช้ "./data/manuals"
    # เพราะชื่อชนกับ chroma_collection_name="manuals"
    site_manuals_dir: str = "./data/site_manuals"
    # เพดานหน้าต่อ crawl กันเว็บใหญ่/ลิงก์วนไม่รู้จบ
    site_learning_max_pages: int = 40
    # W16: เพดานปุ่ม "ปลอดภัย" ที่ไล่กดต่อหน้า (crawler.py::_explore_buttons) กัน list ยาวที่ทุกแถวมีปุ่ม
    site_learning_max_buttons_per_page: int = 15
    # W36: เพดานเฉพาะปุ่ม tier="core" ต่อหน้า (safety.py::classify_button_tier) เลือก top-K ตาม
    # button_core_priority — ไม่กระทบ tier="nav"; น้อยไปพลาดฟังก์ชันหลัก มากไปไม่ช่วยลดปุ่ม
    site_learning_max_core_buttons_per_page: int = 8
    # W24: เดิมเป็น magic number (retry=0) — ตั้งน้อยไป agent หยุดเร็วบนเว็บช้า
    site_learning_goto_retries: int = 2  # รวมครั้งแรกเป็นลองทั้งหมด retries+1 ครั้งต่อหน้า
    site_learning_click_retries: int = 2  # เหมือนกันแต่สำหรับกดปุ่มระหว่างไล่สำรวจ
    site_learning_retry_backoff_ms: int = 500  # หน่วงก่อน retry แต่ละครั้ง
    # W24: infinite scroll — เลื่อนจนไม่ขยับหรือครบจำนวนนี้ก่อน extract (crawler.py::_reveal_dynamic_content)
    site_learning_max_scroll_attempts: int = 6
    site_learning_scroll_wait_ms: int = 350
    # W28: ปุ่ม label+role เดียวกันข้ามหน้ากดได้กี่ครั้งตลอด crawl (crawler.py::_button_signature) —
    # เดิมไม่จำกัด YouTube Shorts "Next video" กิน max_pages หมด; 1 = เข้มสุด แลกกับ coverage ของ
    # ปุ่มชื่อซ้ำที่ความหมายต่างกัน (เช่น "View" คนละตาราง)
    site_learning_max_repeat_button_clicks: int = 1

    # W20: Plan Memory (core/plan_memory.py) — แผนที่ confirm แล้ว ค้น semantic ต่อ (domain, goal)
    # (แทน exact match ของ plan_store.py W19)
    chroma_plan_memory_collection_name: str = "plan_memory"
    # cosine distance สูงสุดที่ยัง reuse ได้ — วัดจริงกับ all-MiniLM-L6-v2: "Login" vs "Sign in" ~0.21,
    # "log me in please" ~0.31, ไม่เกี่ยว ~0.78; ข้ามภาษาไทย-อังกฤษแม่นน้อย ("เข้าสู่ระบบ" ~0.77) เป็น
    # ข้อจำกัดของโมเดล ไม่ใช่ threshold
    plan_memory_max_distance: float = 0.5

    # W_procmem: Procedural Memory (procedural_memory.py/dom_locator.py/fastpath_executor.py) — template
    # มีโครงสร้าง (steps + stable locator + {{slot}}) ข้าม LLM ได้ทั้ง execution loop ไม่ใช่แค่ร่างแผน
    # ลำดับใน generate_plan: procedural template -> plan_memory -> LLM ร่างใหม่ (plan_memory ยังเป็น fallback)
    # ฝั่งเขียน (Abstractor หลัง task สำเร็จ) — additive ปลอดภัยที่จะเปิด default
    enable_procedural_memory_capture: bool = True
    # ฝั่งอ่าน (planner + fast-path replay) ปิดไว้จนกว่าจะ validate template/locator บนเว็บจริง
    enable_procedural_memory: bool = False
    chroma_procedural_memory_collection_name: str = "procedural_memory"
    procedural_memory_max_candidates: int = 3
    procedural_memory_min_confidence: float = 0.6
    # จำนวนครั้งที่ Repair แก้ step เดิมได้ก่อน escalate ไป slow-path (เทียบ _ACTION_RETRIES ใน actions.py)
    procedural_memory_max_repair_attempts: int = 2

    # W46: perception.py::fuzzy_find() — SequenceMatcher.ratio() ขั้นต่ำเมื่อ exact match ไม่เจอ
    # (เช่น "Cierra Vaga") ต่ำไปเสี่ยงตอบข้อมูลผิดคนแบบมั่นใจ (อันตรายกว่า) สูงไปพลาดการพิมพ์ผิด 1-2 ตัว
    agent_fuzzy_match_threshold: float = 0.75

    # W19 ("Semantic Redundancy Evaluator", llm.py::evaluate_semantic_redundancy): +1 LLM call/step
    # ประเมินว่า action มีประโยชน์ไหม — ปิดจนกว่าจะ validate คุณภาพ/latency
    enable_semantic_redundancy_check: bool = False

    # W19-2 (llm.py::evaluate_safety_and_performance): รวม redundancy + permission check เป็น call เดียว
    # (ใช้แทนตัวบนถ้าเปิดทั้งคู่) permission เป็น escalate-only ผ่าน manual_guidance — classify_action()
    # ยังตัดสินสุดท้าย ลดระดับความเสี่ยงไม่ได้ ปิดจนกว่าจะ validate
    enable_middleware_evaluator: bool = False

    # W19-3 (llm.py::generate_persona_message): แปลงสถานะเป็นข้อความไทยธรรมชาติตอนจบ task เท่านั้น —
    # presentation ล้วน ไม่กระทบ control flow ปิดจนกว่าจะ validate โทน
    enable_persona_voice: bool = False

    # W41: ระยะห่างต่ำสุดระหว่าง next_action() 2 ครั้ง กัน RPM quota (orchestrator.py::
    # _STEP_PACING_DELAY_SECONDS) A/B จริง 2026-09-11 (gemini-flash-lite, 3 รอบ/ฝั่ง):
    #
    #                     รันที่ใช้ได้   median   pacing     llm      task ที่ชน 300s
    #   pacing = 3.0        3/3         1.000     99.5s    258.0s         0
    #   pacing = 0          2/3         0.933      0.0s    453.2s         2
    #
    # ปิด pacing แย่ลงทุกด้าน: เวลาที่ประหยัดไปโผล่เป็น backoff +195s เพราะ 429 หนึ่งครั้งโดนทุก call
    # ถัดไปในหน้าต่างนั้น — ห้ามลดค่านี้โดยไม่รัน A/B ซ้ำ
    step_pacing_delay_seconds: float = 3.0

    # W67: nav-fastpath auto-decide (fastpath_executor.py::execute_navigation) — orchestrator ตัดสินเอง
    # จาก goal เมื่อไม่มี nav_target_page_query (W66 เป็น manual) fail-safe ทุกจุด จึงเปิด default
    enable_nav_fastpath_auto_decide: bool = True
    # เข้มกว่า default 1 ของ find_matching_page() เพราะ auto-decide คลิกจริง ไม่ใช่แค่ให้ LLM อ่าน
    nav_fastpath_min_match_score: int = 2


settings = Settings()

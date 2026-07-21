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

    browser_headless: bool = True

    api_host: str = "0.0.0.0"
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


settings = Settings()


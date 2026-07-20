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


settings = Settings()


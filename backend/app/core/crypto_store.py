"""core/crypto_store.py — W_openai_oauth: Fernet key ต่อเครื่อง (data/.credential_key) สำหรับ secret บนดิสก์
แยกเป็นโมดูลกลางจาก site_learning/storage.py เพื่อให้ storage.py และ openai_oauth.py ใช้ key ไฟล์เดียวกัน
(copy โค้ดซ้ำเสี่ยงสองไฟล์ generate คนละ key)"""

from pathlib import Path

from cryptography.fernet import Fernet

from backend.app.config import settings


def credential_key_path() -> Path:
    # Security 1.4: key เดียวต่อเครื่องที่ระดับ data/ (parent ของ site_manuals_dir) ใช้กับ secret ทุกประเภท
    return Path(settings.site_manuals_dir).parent / ".credential_key"


def get_fernet() -> Fernet:
    """Security 1.4: generate key ครั้งแรกแล้วเก็บที่ data/.credential_key (gitignored) — ลบ/ย้ายเครื่อง
    แล้ว secret เก่าถอดรหัสไม่ได้อีก (ต้องกรอก/login ใหม่)"""
    path = credential_key_path()
    if path.exists():
        key = path.read_bytes()
    else:
        key = Fernet.generate_key()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(key)
    return Fernet(key)

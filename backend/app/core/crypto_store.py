"""core/crypto_store.py — W_openai_oauth: key ต่อเครื่อง (data/.credential_key) สำหรับเข้ารหัส/
ถอดรหัส secret ที่เก็บบนดิสก์ผ่าน Fernet — ย้ายมาจาก site_learning/storage.py::_get_fernet()/
_credential_key_path() เดิม (ตอนนั้นมีแค่ credentials.json ของ site_learning ใช้) ตอนนี้
core/openai_oauth.py ต้องเก็บ OAuth token เข้ารหัสด้วยเช่นกัน — แยกเป็นโมดูลกลางแทนที่จะให้
openai_oauth.py import private helper ข้ามโมดูล หรือ copy โค้ด 12 บรรทัดซ้ำ (เสี่ยงสอง
ไฟล์ generate คนละ key กันถ้า mkdir/generate ชนกัน) ทั้งสองไฟล์ยังใช้ key เดียวกันไฟล์เดียวกัน
เป๊ะเหมือนเดิมทุกประการ — ไม่มีอะไรเปลี่ยนพฤติกรรม แค่ย้ายที่อยู่ของฟังก์ชัน"""

from pathlib import Path

from cryptography.fernet import Fernet

from backend.app.config import settings


def credential_key_path() -> Path:
    # Security 1.4: key ต่อเครื่อง เก็บที่ระดับ "data/" เดียว (parent ของ site_manuals_dir)
    # ไม่ใช่ต่อโดเมน/ต่อ secret — secret ทุกประเภท (site credentials, OAuth token) ใช้ key
    # เดียวกันไฟล์เดียวกันนี้
    return Path(settings.site_manuals_dir).parent / ".credential_key"


def get_fernet() -> Fernet:
    """Security 1.4: เข้ารหัส/ถอดรหัส secret บนดิสก์ด้วย key แบบ local-machine — generate
    ครั้งแรกที่ต้องใช้แล้วเก็บไว้ที่ data/.credential_key (gitignored) ไม่ต้องให้ user ตั้ง
    env var เพิ่มเอง ไฟล์นี้เป็น secret ต่อเครื่อง — ย้ายเครื่อง/ลบไฟล์นี้ทิ้งแล้ว secret เก่า
    ที่เข้ารหัสไว้แล้วจะถอดรหัสไม่ได้อีก (ต้อง re-authenticate/กรอกใหม่)"""
    path = credential_key_path()
    if path.exists():
        key = path.read_bytes()
    else:
        key = Fernet.generate_key()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(key)
    return Fernet(key)

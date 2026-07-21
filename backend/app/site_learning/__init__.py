"""W14: Website Learning & Manual Generation — ระบบ crawl เว็บไซต์ที่ไม่เคยเจอมาก่อน
แบบ deterministic (ไม่ใช้ LLM ตัดสินใจ navigate) แล้วสร้าง manual (JSON บนดิสก์) ให้
agent โหลดกลับมาใช้แทนการสำรวจซ้ำทุกครั้ง — แยกต่างหากสมบูรณ์จาก backend/app/rag/ (คู่มือ
ที่ user อัปโหลดเอง เก็บใน ChromaDB)
"""

from backend.app.site_learning.crawler import crawl_site, describe_page
from backend.app.site_learning.extractor import extract_page
from backend.app.site_learning.learn_manager import LearnManager, LearnRecord
from backend.app.site_learning.safety import is_crawl_safe, is_safe_nav_link
from backend.app.site_learning.schema import ButtonInfo, FormFieldInfo, PageInfo, SiteManual, TableInfo
from backend.app.site_learning.storage import (
    load_knowledge_text,
    load_manual,
    manual_exists,
    save_manual,
    update_single_page,
)

__all__ = [
    "crawl_site",
    "describe_page",
    "extract_page",
    "LearnManager",
    "LearnRecord",
    "is_crawl_safe",
    "is_safe_nav_link",
    "ButtonInfo",
    "FormFieldInfo",
    "PageInfo",
    "SiteManual",
    "TableInfo",
    "load_knowledge_text",
    "load_manual",
    "manual_exists",
    "save_manual",
    "update_single_page",
]

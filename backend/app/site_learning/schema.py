"""site_learning/schema.py — W14: โครงสร้างข้อมูล manual ที่เกิดจากการ crawl เว็บไซต์
อัตโนมัติ (deterministic — ดู crawler.py) ระบบนี้แยกต่างหากสมบูรณ์จาก backend/app/rag/
ที่เก็บคู่มือที่ user อัปโหลดเอง (PDF/DOCX/TXT) ลง ChromaDB — อันนี้เก็บเป็นไฟล์ JSON บน
ดิสก์ล้วนๆ (ดู storage.py) ไม่มี ChromaDB/embedding เกี่ยวข้องเลย
"""

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ButtonInfo:
    text: str = ""
    # เขียนโดย LLM ครั้งเดียวต่อหน้า (ไม่ใช่ต่อปุ่ม — ดู crawler.py) อธิบายว่าปุ่มนี้ทำ
    # อะไร ไม่ส่งมาก็ได้ (ว่างเปล่า) ถ้า LLM ไม่ได้ระบุถึงปุ่มนี้ตรงๆ
    description: str = ""
    has_icon: bool = False
    aria_label: str = ""
    title: str = ""
    role: str = ""
    data_testid: str = ""
    # W18: ความหมายที่เดามาจากปุ่ม icon-only ที่ไม่มี text/aria-label/title เลย (ดู
    # extractor.py::inferIconHint — เดาจาก <svg><title>, data-icon, ชื่อ class ของ icon
    # font/library ทั่วไป (fa-*, icon-*, lucide-*, material-icons ฯลฯ), หรือ aria-label
    # ของ ancestor ที่ใกล้ที่สุด) ใช้เป็น fallback ตัวสุดท้ายในลำดับ text > aria_label >
    # title > icon_hint ทุกจุดที่ต้องอ่าน "ปุ่มนี้ทำอะไร" (safety filter, description)
    icon_hint: str = ""
    # CSS selector ที่ compute แบบ stable ตอน extract (ดู extractor.py — ลำดับ
    # ความสำคัญ: data-testid > id ที่ unique > class combo ที่ unique > nth-child path)
    selector: str = ""
    xpath: str = ""


# W18: Pattern ของ UI ที่ซ้ำกันหลาย instance บนหน้าเดียว (เช่น product card 100 ใบ, แถว
# ตาราง, การ์ดวิดีโอ) — แทนที่จะบันทึกทุก instance (ข้อมูลเปลี่ยนทุกครั้งที่ refresh ไม่มี
# ประโยชน์กับ agent ในอนาคต) extractor.py ตรวจจับกลุ่ม element โครงสร้างเดียวกันที่ซ้ำกัน
# ตั้งแต่ 3 ตัวขึ้นไป แล้วบันทึกเป็น UIPatternInfo ตัวเดียว (template) พร้อม selector ที่
# ใช้ซ้ำกับทุก instance ได้จริง — ปุ่ม/ฟอร์มที่อยู่ใน element ที่ถูกจัดเป็น pattern แล้วจะ
# ไม่ถูกเก็บซ้ำในลิสต์ buttons/forms ระดับหน้าอีก (ดู extractor.py::_EXTRACT_JS)
@dataclass
class UIPatternInfo:
    name: str = ""
    # "Card" | "Table Row" | "List Item" | "Grid Item" — เดาจาก tag/โครงสร้างของ
    # representative instance (ดู extractor.py::inferUiType)
    ui_type: str = ""
    # รายชื่อ component ที่พบภายใน instance หนึ่งๆ (เช่น "Image", "Title", "Price",
    # "Rating") — บอกแค่ "มี component ประเภทนี้อยู่" ไม่ใช่ค่าจริงของ instance ไหนเลย
    components: list[str] = field(default_factory=list)
    buttons: list[ButtonInfo] = field(default_factory=list)
    # selector ที่ match ได้กับทุก instance ของ pattern นี้ (ไม่ใช่แค่ตัวแรก) — ปกติเป็น
    # class selector ร่วม เช่น "div.product-card"
    selector: str = ""
    item_count: int = 0


@dataclass
class FormFieldInfo:
    field_name: str = ""
    label: str = ""
    placeholder: str = ""
    required: bool = False
    input_type: str = "text"
    validation: str = ""  # pattern/maxlength/min/max ฯลฯ ถ้ามี attribute ที่บอกไว้
    # W15: CSS selector ที่ compute แบบ stable (เหมือน ButtonInfo.selector — ดู
    # extractor.py::computeSelector) ใช้เติมค่าลงช่องจริงได้ (เช่น crawler.py กรอก
    # username/password ตอน login bootstrap) ไม่ใช่แค่ไว้อ่านโครงสร้างเฉยๆ
    selector: str = ""


@dataclass
class TableInfo:
    columns: list[str] = field(default_factory=list)
    sortable: bool = False
    filterable: bool = False
    paginated: bool = False
    row_actions: list[str] = field(default_factory=list)


@dataclass
class PageInfo:
    name: str = ""
    url: str = ""
    description: str = ""
    menu_path: list[str] = field(default_factory=list)
    breadcrumb: list[str] = field(default_factory=list)
    buttons: list[ButtonInfo] = field(default_factory=list)
    forms: list[FormFieldInfo] = field(default_factory=list)
    tables: list[TableInfo] = field(default_factory=list)
    # W18: pattern ของ UI ที่ซ้ำกันหลาย instance (ดู UIPatternInfo ด้านบน) — element ที่
    # ถูกจัดเป็นส่วนหนึ่งของ pattern แล้วจะไม่ปรากฏซ้ำใน buttons/forms ด้านบนอีก
    ui_patterns: list[UIPatternInfo] = field(default_factory=list)
    filters: list[str] = field(default_factory=list)
    search_box: bool = False
    modals: list[str] = field(default_factory=list)
    tabs: list[str] = field(default_factory=list)


@dataclass
class SiteManual:
    website: str = ""
    version: int = 1
    generated_at: float = 0.0
    pages: list[PageInfo] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "SiteManual":
        pages = [
            PageInfo(
                name=p.get("name", ""),
                url=p.get("url", ""),
                description=p.get("description", ""),
                menu_path=list(p.get("menu_path", [])),
                breadcrumb=list(p.get("breadcrumb", [])),
                buttons=[ButtonInfo(**b) for b in p.get("buttons", [])],
                forms=[FormFieldInfo(**f) for f in p.get("forms", [])],
                tables=[TableInfo(**t) for t in p.get("tables", [])],
                ui_patterns=[
                    UIPatternInfo(
                        name=up.get("name", ""),
                        ui_type=up.get("ui_type", ""),
                        components=list(up.get("components", [])),
                        buttons=[ButtonInfo(**b) for b in up.get("buttons", [])],
                        selector=up.get("selector", ""),
                        item_count=int(up.get("item_count", 0)),
                    )
                    for up in p.get("ui_patterns", [])
                ],
                filters=list(p.get("filters", [])),
                search_box=bool(p.get("search_box", False)),
                modals=list(p.get("modals", [])),
                tabs=list(p.get("tabs", [])),
            )
            for p in data.get("pages", [])
        ]
        return SiteManual(
            website=data.get("website", ""),
            version=int(data.get("version", 1)),
            generated_at=float(data.get("generated_at", 0.0)),
            pages=pages,
        )

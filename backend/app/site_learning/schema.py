"""site_learning/schema.py — W14: โครงสร้าง manual จากการ crawl เว็บ (deterministic, ดู crawler.py).
เก็บเป็น JSON บนดิสก์ (storage.py) แยกจาก backend/app/rag/ (คู่มือ user ใน ChromaDB) โดยสมบูรณ์
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


@dataclass
class ButtonInfo:
    text: str = ""
    # เขียนโดย LLM ครั้งเดียวต่อหน้า (ไม่ใช่ต่อปุ่ม) ว่างได้
    description: str = ""
    has_icon: bool = False
    aria_label: str = ""
    title: str = ""
    role: str = ""
    data_testid: str = ""
    # W18: ความหมายที่เดาจากปุ่ม icon-only (extractor.py::inferIconHint) — fallback สุดท้ายในลำดับ
    # text > aria_label > title > icon_hint
    icon_hint: str = ""
    # CSS selector แบบ stable (extractor.py: data-testid > unique id > unique class combo > nth-child path)
    selector: str = ""
    xpath: str = ""
    # W24: อยู่ในคอนเทนเนอร์เมนู/นำทาง หรือ role menuitem/tab หรือ router-link (extractor.py::isNavMenuItem)
    # -> เดินตามแบบ default-allow (safety.is_safe_nav_link); ปุ่มอื่นยัง default-deny (is_crawl_safe)
    is_nav_menu_item: bool = False
    # W36: ปุ่ม submit ของ <form> ตาม HTML semantics (extractor.py::isFormSubmit) — แค่สัญญาณให้
    # classify_button_tier()/button_core_priority() ไม่ได้อนุญาตให้กด (ยังต้องผ่าน is_crawl_safe())
    is_form_submit: bool = False
    # W36: safety.classify_button_tier() — "nav" | "core" (default กันปุ่ม View/Expand/Next ที่ W16 พึ่งหาย)
    # | "decorative" (share/like/bell/theme — ไม่กดเลย). คำนวณใน Python หลังสร้าง ButtonInfo ไม่ใช่ใน JS
    tier: Literal["nav", "core", "decorative"] = "core"
    # W39: index ใน page.frames (0 = main frame); selector scope กับ document ของ frame นั้น ต้องกดผ่าน
    # frame ที่ถูก (crawler.py::_resolve_click_target). ใช้ index ไม่ใช่ url เพราะ srcdoc iframe ได้ "about:srcdoc"
    # เหมือนกันหมด; index เกินขอบเขต -> fallback ไป page
    frame_index: int = 0


# W18: UI ที่ซ้ำกัน >=3 instance (card/แถวตาราง) เก็บเป็น template เดียวพร้อม selector ที่ match ทุก instance;
# ปุ่ม/ฟอร์มภายใน pattern จะไม่ถูกเก็บซ้ำใน buttons/forms ระดับหน้า (extractor.py::_EXTRACT_JS)
@dataclass
class UIPatternInfo:
    name: str = ""
    # "Card" | "Table Row" | "List Item" | "Grid Item" (extractor.py::inferUiType)
    ui_type: str = ""
    # ประเภท component ที่พบ (เช่น "Image", "Price") ไม่ใช่ค่าจริง
    components: list[str] = field(default_factory=list)
    buttons: list[ButtonInfo] = field(default_factory=list)
    # selector ที่ match ทุก instance เช่น "div.product-card"
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
    # W15: stable CSS selector (extractor.py::computeSelector) ใช้กรอกค่าได้จริง (login bootstrap)
    selector: str = ""
    # W39: เหมือน ButtonInfo.frame_index — ทำให้ฟอร์มใน iframe มองเห็นใน manual เท่านั้น;
    # auto_login.py ยังไม่รองรับกรอกข้าม frame
    frame_index: int = 0


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
    # W18: element ใน pattern จะไม่ปรากฏซ้ำใน buttons/forms
    ui_patterns: list[UIPatternInfo] = field(default_factory=list)
    filters: list[str] = field(default_factory=list)
    search_box: bool = False
    modals: list[str] = field(default_factory=list)
    tabs: list[str] = field(default_factory=list)
    # W66[A] Fast-Path Navigation: parent-pointer tree จาก BFS — parent_url ว่างเฉพาะ root; arrived_via =
    # locator descriptor (shape ของ dom_locator.compute_locator_descriptor()) ของ element บน parent ที่คลิกมาถึง.
    # fastpath_executor.build_navigation_steps() เดินย้อนถึง root ได้ลำดับ click; ว่างทั้งคู่ = ไม่มีข้อมูล (crawl ก่อน W66)
    parent_url: str = ""
    arrived_via: dict = field(default_factory=dict)


@dataclass
class SiteManual:
    website: str = ""
    version: int = 1
    generated_at: float = 0.0
    pages: list[PageInfo] = field(default_factory=list)
    # W24: ปัญหาระหว่าง crawl (goto/click ล้มครบ retry, login ไม่ผ่าน) — เดิมถูกกลืนเงียบ แยกไม่ออกว่าจบเพราะครบ
    # หรือพัง. entry: {"url","phase","error"} หรือ +"button"
    errors: list[dict] = field(default_factory=list)
    # W26: สรุป 2-4 ประโยคว่าเว็บทำอะไรได้ (LLM ครั้งเดียวหลัง crawl — crawler.py::describe_site()) ว่างได้
    summary: str = ""

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
                parent_url=p.get("parent_url", ""),
                arrived_via=dict(p.get("arrived_via", {})),
            )
            for p in data.get("pages", [])
        ]
        return SiteManual(
            website=data.get("website", ""),
            version=int(data.get("version", 1)),
            generated_at=float(data.get("generated_at", 0.0)),
            pages=pages,
            errors=list(data.get("errors", [])),
            summary=data.get("summary", ""),
        )

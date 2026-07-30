import pytest
from playwright.async_api import async_playwright

from backend.app.site_learning.extractor import extract_page

# เทสต์กลุ่มนี้เปิด chromium จริง (ไม่ mock) เหมือน test_perception.py เพราะ
# extract_page() พึ่ง page.evaluate() รัน JS จริงบน DOM จริง — mock DOM API ยากกว่าเปิด
# browser เปล่าตรงๆ


async def _extract(html: str):
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(html)
        page_info, nav_links = await extract_page(page)
        await browser.close()
    return page_info, nav_links


_HTML_BUTTONS = """
<html><body>
  <button data-testid="export-btn">Export</button>
  <button aria-label="Delete item">Delete</button>
  <button id="unique-btn">Unique</button>
  <button style="display:none">Hidden</button>
  <button disabled>Disabled</button>
</body></html>
"""


@pytest.mark.asyncio
async def test_extract_page_collects_visible_buttons_with_selectors():
    page_info, _ = await _extract(_HTML_BUTTONS)
    texts = [b.text for b in page_info.buttons]

    assert "Export" in texts
    assert "Delete" in texts
    assert "Unique" in texts
    # ปุ่มที่ซ่อน/disabled ต้องไม่ถูกเก็บ
    assert "Hidden" not in texts
    assert "Disabled" not in texts

    export_btn = next(b for b in page_info.buttons if b.text == "Export")
    assert export_btn.data_testid == "export-btn"
    assert "data-testid" in export_btn.selector

    delete_btn = next(b for b in page_info.buttons if b.text == "Delete")
    assert delete_btn.aria_label == "Delete item"

    unique_btn = next(b for b in page_info.buttons if b.text == "Unique")
    assert unique_btn.selector == "#unique-btn"
    assert unique_btn.xpath == '//*[@id="unique-btn"]'


_HTML_FORM = """
<html><body>
  <form>
    <label for="email">Email</label>
    <input id="email" name="email" type="email" placeholder="you@example.com" required>
    <input type="hidden" name="csrf" value="abc">
    <input type="submit" value="Go">
  </form>
</body></html>
"""


@pytest.mark.asyncio
async def test_extract_page_collects_form_fields_and_skips_hidden_and_submit():
    page_info, _ = await _extract(_HTML_FORM)

    assert len(page_info.forms) == 1
    field = page_info.forms[0]
    assert field.field_name == "email"
    assert field.label == "Email"
    assert field.placeholder == "you@example.com"
    assert field.required is True
    assert field.input_type == "email"
    assert field.selector == "#email"


_HTML_TABLE = """
<html><body>
  <div class="table-container">
    <input type="search" placeholder="Filter results">
    <table>
      <thead><tr><th>Name</th><th>Status</th></tr></thead>
      <tbody>
        <tr><td>Item 1</td><td><button>Edit</button></td></tr>
      </tbody>
    </table>
    <div class="pagination">Next</div>
  </div>
</body></html>
"""


@pytest.mark.asyncio
async def test_extract_page_collects_table_structure():
    page_info, _ = await _extract(_HTML_TABLE)

    assert len(page_info.tables) == 1
    table = page_info.tables[0]
    assert table.columns == ["Name", "Status"]
    assert table.filterable is True
    assert table.paginated is True
    assert "Edit" in table.row_actions


_HTML_NAV = """
<html><body>
  <nav aria-label="Main">
    <a href="/dashboard">Dashboard</a>
    <a href="/products">Products</a>
    <a href="#section">Jump to section</a>
    <a href="javascript:void(0)">No-op</a>
  </nav>
</body></html>
"""


@pytest.mark.asyncio
async def test_extract_page_discovers_nav_links_and_skips_anchors_and_javascript():
    _, nav_links = await _extract(_HTML_NAV)
    hrefs = {link["href"] for link in nav_links}

    assert "/dashboard" in hrefs
    assert "/products" in hrefs
    assert not any(h.startswith("#") for h in hrefs)
    assert not any(h.lower().startswith("javascript:") for h in hrefs)


_HTML_CONTENT_AREA_LINK = """
<html><body>
  <nav><a href="/dashboard">Dashboard</a></nav>
  <main>
    <article>
      <p>Some text <a href="/articles/42">Read more</a> in the middle of a paragraph.</p>
    </article>
    <div class="card"><a href="/products/7">View product</a></div>
  </main>
</body></html>
"""


@pytest.mark.asyncio
async def test_extract_page_discovers_content_area_links_not_just_nav_links():
    """W25: user ขอ "extract all clickable elements: a" ตรงๆ — ลิงก์กลางบทความ/การ์ดที่ไม่
    ได้อยู่ใน <nav>/<aside>/<header>/<footer>/[role=tablist]/[role=menu] เลยก็ต้องถูกเก็บ
    เข้า nav_links ด้วย ไม่ใช่แค่ลิงก์ในเมนู (เดิมสแกนแค่ใน NAV_CONTAINERS พลาดลิงก์แบบนี้
    ไปเลยทั้งที่เป็นหน้าที่ "เข้าถึงได้จริง")"""
    _, nav_links = await _extract(_HTML_CONTENT_AREA_LINK)
    hrefs = {link["href"] for link in nav_links}

    assert "/dashboard" in hrefs  # ของเดิมยังทำงานต่อ
    assert "/articles/42" in hrefs  # ลิงก์กลางบทความ
    assert "/products/7" in hrefs  # ลิงก์ในการ์ดนอก nav


_HTML_BREADCRUMB = """
<html><body>
  <div class="breadcrumb"><a href="/">Home</a><span>Dashboard</span></div>
</body></html>
"""


@pytest.mark.asyncio
async def test_extract_page_collects_breadcrumb():
    page_info, _ = await _extract(_HTML_BREADCRUMB)
    assert page_info.breadcrumb == ["Home", "Dashboard"]


@pytest.mark.asyncio
async def test_extract_page_empty_page_has_no_buttons_forms_tables():
    page_info, nav_links = await _extract("<html><body></body></html>")
    assert page_info.buttons == []
    assert page_info.forms == []
    assert page_info.tables == []
    assert page_info.ui_patterns == []
    assert nav_links == []


# ---------------- W18: icon-only button detection (ไม่มี text/aria-label/title เลย) ----------------

_HTML_ICON_BUTTONS = """
<html><body>
  <button id="cart-btn"><svg><title>Shopping Cart</title><path d="M0 0"/></svg></button>
  <button id="search-btn" data-icon="search"><svg><path d="M0 0"/></svg></button>
  <button id="heart-btn"><i class="fa fa-heart"></i></button>
  <button id="fav-btn"><i class="material-icons">favorite</i></button>
  <div aria-label="Notifications"><button id="notif-btn"><svg><path d="M0 0"/></svg></button></div>
  <button id="mystery-btn"><svg><path d="M0 0"/></svg></button>
</body></html>
"""


@pytest.mark.asyncio
async def test_extract_page_infers_icon_hint_from_svg_title():
    page_info, _ = await _extract(_HTML_ICON_BUTTONS)
    btn = next(b for b in page_info.buttons if b.selector == "#cart-btn")
    assert btn.text == ""
    assert btn.icon_hint == "shopping cart"
    assert btn.has_icon is True


@pytest.mark.asyncio
async def test_extract_page_infers_icon_hint_from_data_icon_attribute():
    page_info, _ = await _extract(_HTML_ICON_BUTTONS)
    btn = next(b for b in page_info.buttons if b.selector == "#search-btn")
    assert btn.icon_hint == "search"


@pytest.mark.asyncio
async def test_extract_page_infers_icon_hint_from_icon_font_class_name():
    page_info, _ = await _extract(_HTML_ICON_BUTTONS)
    btn = next(b for b in page_info.buttons if b.selector == "#heart-btn")
    assert btn.icon_hint == "heart"


@pytest.mark.asyncio
async def test_extract_page_infers_icon_hint_from_material_icons_ligature():
    page_info, _ = await _extract(_HTML_ICON_BUTTONS)
    btn = next(b for b in page_info.buttons if b.selector == "#fav-btn")
    assert btn.icon_hint == "favorite"


@pytest.mark.asyncio
async def test_extract_page_infers_icon_hint_from_nearest_labeled_ancestor():
    page_info, _ = await _extract(_HTML_ICON_BUTTONS)
    btn = next(b for b in page_info.buttons if b.selector == "#notif-btn")
    assert btn.icon_hint == "notifications"


@pytest.mark.asyncio
async def test_extract_page_icon_hint_empty_when_no_signal_available():
    page_info, _ = await _extract(_HTML_ICON_BUTTONS)
    btn = next(b for b in page_info.buttons if b.selector == "#mystery-btn")
    assert btn.icon_hint == ""


# ---------------- W18: UI pattern detection (product card / list item ที่ซ้ำกัน) ----------------

def _product_card(i: int) -> str:
    return f"""
    <div class="product-card">
      <img src="/p{i}.jpg" alt="Product {i}">
      <h3 class="title">Product {i}</h3>
      <span class="price">${10 + i}.99</span>
      <button class="add-to-cart">Add to Cart</button>
    </div>
    """


_HTML_PRODUCT_GRID = f"""
<html><body>
  <h2>Related Products</h2>
  <div class="grid">
    {"".join(_product_card(i) for i in range(5))}
  </div>
</body></html>
"""


@pytest.mark.asyncio
async def test_extract_page_collapses_repeated_cards_into_a_single_ui_pattern():
    page_info, _ = await _extract(_HTML_PRODUCT_GRID)

    assert len(page_info.ui_patterns) == 1
    pattern = page_info.ui_patterns[0]
    assert pattern.item_count == 5
    assert pattern.ui_type == "Card"
    assert pattern.name == "Related Products"  # เดาจาก heading ก่อนหน้า container
    assert "Image" in pattern.components
    assert "Title" in pattern.components
    assert "Price" in pattern.components
    assert pattern.selector == "div.product-card"
    assert any(b.text == "Add to Cart" for b in pattern.buttons)


@pytest.mark.asyncio
async def test_extract_page_does_not_duplicate_pattern_buttons_in_flat_button_list():
    """ปุ่ม "Add to Cart" ต้องไม่ปรากฏซ้ำ 5 ครั้งใน page_info.buttons — ถูกเก็บไปแล้วครั้ง
    เดียวใน ui_patterns[0].buttons"""
    page_info, _ = await _extract(_HTML_PRODUCT_GRID)
    add_to_cart_count = sum(1 for b in page_info.buttons if b.text == "Add to Cart")
    assert add_to_cart_count == 0


@pytest.mark.asyncio
async def test_extract_page_does_not_treat_fewer_than_three_similar_items_as_a_pattern():
    html = f"""
    <html><body>
      <div class="grid">
        {"".join(_product_card(i) for i in range(2))}
      </div>
    </body></html>
    """
    page_info, _ = await _extract(html)

    assert page_info.ui_patterns == []
    # ยังเก็บเป็นปุ่มปกติทีละใบ (ไม่ถูกยุบเป็น pattern เพราะมีแค่ 2 ตัว ต่ำกว่า threshold)
    assert sum(1 for b in page_info.buttons if b.text == "Add to Cart") == 2


# ---------------- W24: is_nav_menu_item — เมนู/nav item ที่ไม่ใช่ <a href> ----------------

_HTML_NAV_VARIETY = """
<html><body>
  <nav>
    <a href="/dashboard">Dashboard</a>
    <a href="#tab1" role="tab">Tab One</a>
    <button onclick="doNav()">Sidebar Item</button>
  </nav>
  <div role="menuitem" onclick="doNav()">Settings</div>
  <div role="tab">Overview</div>
  <button class="my-router-link-active" onclick="doNav()">Router Styled</button>
  <button id="plain-action">Export</button>
</body></html>
"""


@pytest.mark.asyncio
async def test_extract_page_flags_role_menuitem_and_role_tab_as_nav_menu_item():
    page_info, _ = await _extract(_HTML_NAV_VARIETY)
    settings_btn = next(b for b in page_info.buttons if b.text == "Settings")
    assert settings_btn.is_nav_menu_item is True
    overview_tab = next(b for b in page_info.buttons if b.text == "Overview")
    assert overview_tab.is_nav_menu_item is True


@pytest.mark.asyncio
async def test_extract_page_flags_elements_inside_nav_container_as_nav_menu_item():
    page_info, _ = await _extract(_HTML_NAV_VARIETY)
    sidebar_item = next(b for b in page_info.buttons if b.text == "Sidebar Item")
    assert sidebar_item.is_nav_menu_item is True


@pytest.mark.asyncio
async def test_extract_page_flags_router_link_styled_class_as_nav_menu_item():
    page_info, _ = await _extract(_HTML_NAV_VARIETY)
    router_styled = next(b for b in page_info.buttons if b.text == "Router Styled")
    assert router_styled.is_nav_menu_item is True


@pytest.mark.asyncio
async def test_extract_page_does_not_flag_real_href_anchor_as_nav_menu_item():
    """W24: <a href="..."> ที่มีปลายทางจริงต้องไม่ถูกแปะ is_nav_menu_item — ปล่อยให้ BFS
    href เดิม (ที่เช็ค same-origin ก่อน navigate) จัดการแทน กัน _explore_buttons() ไป
    "คลิก" ซ้ำแล้วเสี่ยงหลุดไปนอกโดเมนก่อนรู้ปลายทาง (ดู crawler.py::_is_explorable)"""
    page_info, _ = await _extract(_HTML_NAV_VARIETY)
    dashboard_link = next(b for b in page_info.buttons if b.text == "Dashboard")
    assert dashboard_link.is_nav_menu_item is False


@pytest.mark.asyncio
async def test_extract_page_treats_fragment_href_tab_as_nav_menu_item():
    """<a href="#tab1" role="tab"> ไม่มีปลายทางข้ามหน้าจริง (fragment เฉยๆ) — ไม่เสี่ยงหลุด
    โดเมนแบบ href จริง ยังคง eligible เป็น nav menu item ได้ปกติ"""
    page_info, _ = await _extract(_HTML_NAV_VARIETY)
    tab_one = next(b for b in page_info.buttons if b.text == "Tab One")
    assert tab_one.is_nav_menu_item is True


@pytest.mark.asyncio
async def test_extract_page_does_not_flag_ordinary_button_as_nav_menu_item():
    page_info, _ = await _extract(_HTML_NAV_VARIETY)
    export_btn = next(b for b in page_info.buttons if b.text == "Export")
    assert export_btn.is_nav_menu_item is False


# ---------------- W36: core function classification (tier + is_form_submit) ----------------

_HTML_TIER_CLASSIFICATION = """
<html><body>
  <nav><button id="nav-btn">Dashboard</button></nav>
  <form>
    <input type="text" name="q">
    <button id="submit-btn">Continue</button>
  </form>
  <button id="core-btn">Search</button>
  <button id="decorative-btn">Share</button>
  <button id="fallback-btn">View</button>
  <button id="page-num-btn">3</button>
  <button id="outside-form-btn" type="submit">Lonely Submit</button>
</body></html>
"""


@pytest.mark.asyncio
async def test_extract_page_classifies_nav_menu_item_as_nav_tier():
    page_info, _ = await _extract(_HTML_TIER_CLASSIFICATION)
    btn = next(b for b in page_info.buttons if b.selector == "#nav-btn")
    assert btn.tier == "nav"


@pytest.mark.asyncio
async def test_extract_page_flags_default_button_in_form_as_form_submit_and_core_tier():
    """<button> ที่ไม่มี attribute type เลยภายใน <form> เป็น type=submit โดย default ตาม
    HTML semantics (ดู extractor.py::isFormSubmit) — ต้องเป็นสัญญาณ DOM signal ที่ทำให้
    classify_button_tier() จัดเป็น "core" ทันที (priority สูงสุดตอนตัด top-K ด้วย — ดู
    safety.button_core_priority)"""
    page_info, _ = await _extract(_HTML_TIER_CLASSIFICATION)
    btn = next(b for b in page_info.buttons if b.selector == "#submit-btn")
    assert btn.is_form_submit is True
    assert btn.tier == "core"


@pytest.mark.asyncio
async def test_extract_page_is_form_submit_false_when_type_submit_button_has_no_form_ancestor():
    """type="submit" เฉยๆ ไม่พอ — ต้องอยู่ใน <form> จริงด้วย (ดู extractor.py::isFormSubmit)"""
    page_info, _ = await _extract(_HTML_TIER_CLASSIFICATION)
    btn = next(b for b in page_info.buttons if b.selector == "#outside-form-btn")
    assert btn.is_form_submit is False


@pytest.mark.asyncio
async def test_extract_page_classifies_core_keyword_button_as_core_tier():
    page_info, _ = await _extract(_HTML_TIER_CLASSIFICATION)
    btn = next(b for b in page_info.buttons if b.selector == "#core-btn")
    assert btn.tier == "core"


@pytest.mark.asyncio
async def test_extract_page_classifies_decorative_keyword_button_as_decorative_tier():
    page_info, _ = await _extract(_HTML_TIER_CLASSIFICATION)
    btn = next(b for b in page_info.buttons if b.selector == "#decorative-btn")
    assert btn.tier == "decorative"


@pytest.mark.asyncio
async def test_extract_page_classifies_pagination_page_number_greater_than_one_as_decorative():
    page_info, _ = await _extract(_HTML_TIER_CLASSIFICATION)
    btn = next(b for b in page_info.buttons if b.selector == "#page-num-btn")
    assert btn.tier == "decorative"


@pytest.mark.asyncio
async def test_extract_page_classifies_unmatched_button_as_core_tier_fallback():
    """"View" ไม่ตรง CORE_ACTION_KEYWORDS/DECORATIVE_KEYWORDS ตรงๆ เลยสักคำ — fallback ต้อง
    เป็น "core" (ไม่ใช่ "decorative") มิฉะนั้นความสามารถเดิมของ W16 ("View" ในตารางที่พาไป
    หน้ารายละเอียด) จะหายไปหมด"""
    page_info, _ = await _extract(_HTML_TIER_CLASSIFICATION)
    btn = next(b for b in page_info.buttons if b.selector == "#fallback-btn")
    assert btn.tier == "core"


# ---------------- W37: กันกดดูวีดีโอ (YouTube/Facebook/Instagram ฯลฯ) ----------------

_HTML_VIDEO_BUTTONS = """
<html><body>
  <button id="watch-btn">Watch Now</button>
  <nav><div role="tab" id="reels-tab">Reels</div></nav>
  <button id="unrelated-btn">Expand</button>
</body></html>
"""


@pytest.mark.asyncio
async def test_extract_page_classifies_watch_button_as_decorative_tier():
    page_info, _ = await _extract(_HTML_VIDEO_BUTTONS)
    btn = next(b for b in page_info.buttons if b.selector == "#watch-btn")
    assert btn.tier == "decorative"


@pytest.mark.asyncio
async def test_extract_page_classifies_video_nav_tab_as_decorative_not_nav():
    """"Reels" เป็น role="tab" ใน <nav> จริง (is_nav_menu_item=True ตามปกติ) แต่ label บ่ง
    บอกวีดีโอตรงๆ — ต้องชนะ is_nav_menu_item เป็น "decorative" (ไม่ใช่ "nav") ตามที่ user
    ยืนยัน: ไม่ต้องการให้กดเข้าไปดูวีดีโอเลยไม่ว่าจะมาในรูปแบบเมนู/tab หรือปุ่มทั่วไปก็ตาม"""
    page_info, _ = await _extract(_HTML_VIDEO_BUTTONS)
    tab = next(b for b in page_info.buttons if b.selector == "#reels-tab")
    assert tab.is_nav_menu_item is True
    assert tab.tier == "decorative"


@pytest.mark.asyncio
async def test_extract_page_classifies_unrelated_button_as_core_tier_not_affected_by_video_check():
    page_info, _ = await _extract(_HTML_VIDEO_BUTTONS)
    btn = next(b for b in page_info.buttons if b.selector == "#unrelated-btn")
    assert btn.tier == "core"


# ---------------- W38: กันกดดูแฮชแท็ก ----------------

_HTML_HASHTAG_BUTTONS = """
<html><body>
  <a id="hashtag-link" href="/hashtag/travel">#travel</a>
  <nav><div role="tab" id="hashtag-tab">#Trending</div></nav>
  <button id="unrelated-btn">Expand</button>
</body></html>
"""


@pytest.mark.asyncio
async def test_extract_page_classifies_hashtag_link_as_decorative_tier():
    page_info, _ = await _extract(_HTML_HASHTAG_BUTTONS)
    btn = next(b for b in page_info.buttons if b.selector == "#hashtag-link")
    assert btn.tier == "decorative"


@pytest.mark.asyncio
async def test_extract_page_classifies_hashtag_nav_tab_as_decorative_not_nav():
    """"#Trending" เป็น role="tab" ใน <nav> จริง (is_nav_menu_item=True ตามปกติ) แต่ label
    ขึ้นต้นด้วย "#" ตรงๆ — ต้องชนะ is_nav_menu_item เป็น "decorative" (ไม่ใช่ "nav") เหมือน
    วีดีโอ (W37): ไม่ต้องการให้กดเข้าไปดูแฮชแท็กเลยไม่ว่าจะมาในรูปแบบเมนู/tab หรือปุ่มทั่วไป"""
    page_info, _ = await _extract(_HTML_HASHTAG_BUTTONS)
    tab = next(b for b in page_info.buttons if b.selector == "#hashtag-tab")
    assert tab.is_nav_menu_item is True
    assert tab.tier == "decorative"


@pytest.mark.asyncio
async def test_extract_page_classifies_unrelated_button_as_core_tier_not_affected_by_hashtag_check():
    page_info, _ = await _extract(_HTML_HASHTAG_BUTTONS)
    btn = next(b for b in page_info.buttons if b.selector == "#unrelated-btn")
    assert btn.tier == "core"


# ---------------- W39: ปุ่มที่อยู่ใน <iframe> (รวมถึง iframe ซ้อนกันหลายชั้น) ----------------

_HTML_MAIN_FRAME_BUTTON = """
<html><body>
  <button id="main-btn">Main Button</button>
  <iframe id="outer" srcdoc="<html><body><button id='inner-btn'>Inner Button</button></body></html>"></iframe>
</body></html>
"""


@pytest.mark.asyncio
async def test_extract_page_finds_button_inside_single_iframe():
    """W39: ปุ่มใน <iframe> ต้องถูกเก็บเข้า page_info.buttons ด้วย (เดิม document.
    querySelectorAll() ของ main frame มองไม่เห็นเลย เพราะเป็นคนละ document object)"""
    page_info, _ = await _extract(_HTML_MAIN_FRAME_BUTTON)
    texts = {b.text for b in page_info.buttons}
    assert "Main Button" in texts
    assert "Inner Button" in texts


@pytest.mark.asyncio
async def test_extract_page_main_frame_button_has_frame_index_zero():
    page_info, _ = await _extract(_HTML_MAIN_FRAME_BUTTON)
    btn = next(b for b in page_info.buttons if b.text == "Main Button")
    assert btn.frame_index == 0


@pytest.mark.asyncio
async def test_extract_page_iframe_button_has_nonzero_frame_index():
    """W39: ต้องแปะ frame_index (ตำแหน่งใน page.frames) ไว้กับปุ่มที่มาจาก child frame — ให้
    crawler.py::_resolve_click_target() รู้ว่าต้องกดผ่าน frame object ไหน (page.click()
    ธรรมดา query ข้าม frame boundary ไม่ได้)"""
    page_info, _ = await _extract(_HTML_MAIN_FRAME_BUTTON)
    btn = next(b for b in page_info.buttons if b.text == "Inner Button")
    assert btn.frame_index != 0


_HTML_NESTED_IFRAME_BUTTONS = """
<html><body>
  <h1>Playground</h1>
  <iframe id="outer" srcdoc="
    <html><body>
      <button id='edit1'>Edit</button>
      <button id='submit1'>Submit</button>
      <iframe id='inner' srcdoc='&lt;html&gt;&lt;body&gt;&lt;button id=edit2&gt;Edit&lt;/button&gt;&lt;button id=submit2&gt;Submit&lt;/button&gt;&lt;/body&gt;&lt;/html&gt;'></iframe>
    </body></html>
  "></iframe>
</body></html>
"""


@pytest.mark.asyncio
async def test_extract_page_finds_buttons_inside_nested_iframes_two_levels_deep():
    """W39: จำลองหน้า test playground จริงที่ user รายงาน (Outer Frame ซ้อน Inner Frame อีก
    ชั้น แต่ละชั้นมีปุ่ม Edit/Submit ของตัวเอง) — page.frames คืนทุก frame แบบ flat รวม
    nested เอง (ไม่ต้อง recurse เอง) ต้องเจอปุ่มครบทั้ง 4 ปุ่ม (2 ชั้น x 2 ปุ่ม)"""
    page_info, _ = await _extract(_HTML_NESTED_IFRAME_BUTTONS)
    edit_buttons = [b for b in page_info.buttons if b.text == "Edit"]
    submit_buttons = [b for b in page_info.buttons if b.text == "Submit"]
    assert len(edit_buttons) == 2
    assert len(submit_buttons) == 2
    # ปุ่มจากคนละ frame ต้องได้ frame_index ต่างกัน (ไม่ใช่ frame เดียวกันโดยบังเอิญ)
    edit_frame_indices = {b.frame_index for b in edit_buttons}
    assert len(edit_frame_indices) == 2
    assert 0 not in edit_frame_indices  # ไม่มีตัวไหนอยู่ main frame เลย ทั้งคู่อยู่ใน iframe

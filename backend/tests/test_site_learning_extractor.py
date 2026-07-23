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

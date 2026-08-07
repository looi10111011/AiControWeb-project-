import json

import pytest
from playwright.async_api import async_playwright

from backend.app.core.perception import count_elements, extract_table_data, fuzzy_find, get_snapshot, resolve_frame

# เทสต์กลุ่มนี้เปิด chromium จริง (ไม่ mock) เพราะ get_snapshot() พึ่ง page.evaluate()
# รัน JS จริงบน DOM จริง — mock DOM API ยากกว่าเปิด browser เปล่าตรงๆ

_HTML_WITH_FOOTER = """
<html><body>
  <button id="main-btn">Add to cart</button>
  <footer>
    <a href="https://facebook.com">Facebook</a>
    <a href="https://twitter.com">Twitter</a>
  </footer>
  <div class="site-footer">
    <button>Newsletter signup</button>
  </div>
  <div id="page-footer">
    <a href="/terms">Terms</a>
  </div>
</body></html>
"""


@pytest.mark.asyncio
async def test_get_snapshot_filters_out_footer_elements():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(_HTML_WITH_FOOTER)

        elements, text_repr = await get_snapshot(page)

        await browser.close()

    labels = [e["label"] for e in elements]
    assert "Add to cart" in labels
    assert "Facebook" not in labels
    assert "Twitter" not in labels
    assert "Newsletter signup" not in labels
    assert "Terms" not in labels
    assert len(elements) == 1
    assert "Add to cart" in text_repr


# บั๊กที่เจอจริงบน saucedemo.com: ปุ่ม Checkout อยู่ใน <div class="cart_footer">
# ซึ่งเดิม filter แบบ substring จับคำว่า "footer" ไปแมตช์ผิด ทำให้ปุ่ม Checkout
# หายไปจาก snapshot ทั้งที่ไม่ใช่ site footer เลย (เป็นแค่ action bar ท้าย
# component ตะกร้า) — ต้องยังกรอง site-footer/page-footer จริงได้เหมือนเดิมด้วย
_HTML_WITH_COMPONENT_FOOTER = """
<html><body>
  <div class="cart_footer">
    <button id="continue-shopping">Continue Shopping</button>
    <button id="checkout">Checkout</button>
  </div>
  <div class="modal-footer">
    <button>Confirm</button>
  </div>
  <footer>
    <a href="https://facebook.com">Facebook</a>
  </footer>
  <div class="site-footer">
    <button>Newsletter signup</button>
  </div>
</body></html>
"""


@pytest.mark.asyncio
async def test_get_snapshot_does_not_filter_component_footer_action_bars():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(_HTML_WITH_COMPONENT_FOOTER)

        elements, _ = await get_snapshot(page)

        await browser.close()

    labels = [e["label"] for e in elements]
    assert "Checkout" in labels
    assert "Continue Shopping" in labels
    assert "Confirm" in labels
    assert "Facebook" not in labels
    assert "Newsletter signup" not in labels


# ทดสอบเคสที่เจอจริงบน saucedemo.com: ปุ่มตะกร้า (.shopping_cart_link) ไม่มี
# innerText/aria-label เลย (แค่ไอคอนจาก CSS) มีแค่ data-test attribute — เดิม
# label จะว่างเปล่า ทำให้ LLM เห็นแค่ "[N] a" เดาไม่ออกว่าคือปุ่มตะกร้า
_HTML_ICON_ONLY_ELEMENTS = """
<html><body>
  <a class="shopping_cart_link" data-test="shopping-cart-link" href="/cart.html"
     style="display:inline-block;width:20px;height:20px;background:gray"></a>
  <button data-testid="close-modal"></button>
  <div tabindex="0" id="menu_toggle_button"
       style="display:inline-block;width:20px;height:20px;background:gray"></div>
</body></html>
"""


@pytest.mark.asyncio
async def test_get_snapshot_falls_back_to_data_test_and_id_for_icon_only_elements():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(_HTML_ICON_ONLY_ELEMENTS)

        elements, _ = await get_snapshot(page)

        await browser.close()

    labels = [e["label"] for e in elements]
    assert "shopping cart link" in labels
    assert "close modal" in labels
    assert "menu toggle button" in labels


# บั๊กที่เจอจริงบน demoqa.com/webtables: ปุ่มแก้ไข/ลบในคอลัมน์ Action เป็น
# <span title="Edit"><svg>...</svg></span> ล้วนๆ ไม่มี role="button"/tabindex/onclick
# attribute เลยตาม a11y spec (แค่ title + cursor:pointer ที่ตั้งไว้เอง) — selectors เดิม
# (a/button/[role=button]/[onclick]/[tabindex]/...) มองไม่เห็น element แบบนี้เลยทั้งที่
# กดได้จริงในเบราว์เซอร์ (คลิกจริงกระตุ้น handler ผ่าน event bubbling ปกติ)
_HTML_ICON_SPAN_WITH_TITLE_AND_POINTER_CURSOR = """
<html><body>
  <div class="action-buttons">
    <span title="Edit" id="edit-record-1" style="cursor:pointer;display:inline-block;width:16px;height:16px">
      <svg viewBox="0 0 1024 1024"><path d="M1 1"></path></svg>
    </span>
    <span title="Delete" id="delete-record-1" style="cursor:pointer;display:inline-block;width:16px;height:16px">
      <svg viewBox="0 0 1024 1024"><path d="M2 2"></path></svg>
    </span>
  </div>
</body></html>
"""


@pytest.mark.asyncio
async def test_get_snapshot_finds_icon_only_span_with_title_and_pointer_cursor():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(_HTML_ICON_SPAN_WITH_TITLE_AND_POINTER_CURSOR)

        elements, _ = await get_snapshot(page)

        await browser.close()

    labels = [e["label"] for e in elements]
    assert "Edit" in labels
    assert "Delete" in labels
    assert len(elements) == 2


# element ที่มี title แต่ "ไม่ได้" ตั้ง cursor:pointer (เช่น title ไว้โชว์ tooltip ของ
# ข้อความยาวๆ ในตาราง ไม่ใช่ปุ่มกดได้) ต้องไม่ถูกนับเป็น element กดได้ — กัน noise ที่จะ
# ทำให้ token cost ต่อ step โตขึ้นโดยไม่จำเป็นจาก tooltip ทั่วๆ ไปที่มีอยู่เกลื่อนหน้าเว็บ
_HTML_TITLE_WITHOUT_POINTER_CURSOR = """
<html><body>
  <td title="This is just a tooltip, not a button">Some truncated cell text</td>
</body></html>
"""


@pytest.mark.asyncio
async def test_get_snapshot_ignores_title_elements_without_pointer_cursor():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(_HTML_TITLE_WITHOUT_POINTER_CURSOR)

        elements, _ = await get_snapshot(page)

        await browser.close()

    assert elements == []


# container ที่ตั้ง title + cursor:pointer ไว้เอง แต่ข้างในมีปุ่มจริง (<button>) ซ้อนอยู่ —
# ต้องได้ index แค่ปุ่มจริงข้างในตัวเดียว ไม่ใช่ทั้ง container ด้วย (กันได้ index ซ้ำสอง
# อันสำหรับพื้นที่คลิกเดียวกัน)
_HTML_ICON_CONTAINER_WRAPS_REAL_BUTTON = """
<html><body>
  <div title="Actions" style="cursor:pointer">
    <button id="real-btn">Real Button</button>
  </div>
</body></html>
"""


@pytest.mark.asyncio
async def test_get_snapshot_does_not_duplicate_icon_container_wrapping_real_button():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(_HTML_ICON_CONTAINER_WRAPS_REAL_BUTTON)

        elements, _ = await get_snapshot(page)

        await browser.close()

    assert len(elements) == 1
    assert elements[0]["tag"] == "button"
    assert "Actions" not in [e["label"] for e in elements]


# W20 (Task10, "Element Finder/Selector Resolver" — บั๊กจริงที่ user รายงาน): ไล่ debug ด้วย
# การเปิดหน้าจริงของ opensource-demo.orangehrmlive.com พบว่าตัว profile-dropdown trigger จริง
# คือ `<span class="oxd-userdropdown-tab">` ที่ไม่มีทั้ง role/tabindex/onclick (ไม่ตรง
# selectors มาตรฐาน) และไม่มีทั้ง title/aria-label/data-test* (ไม่ตรง ICON_LABEL_SELECTOR ด้วย)
# — element นี้ไม่เคยติด index เลยตั้งแต่ต้น ทำให้ agent ต้องเดา index อื่นที่ใกล้เคียงแทน (เช่น
# ปุ่ม "Help") จำลอง markup จริงของ OrangeHRM ตรงๆ ด้านล่างนี้
_HTML_ORANGEHRM_USERDROPDOWN = """
<html><body>
  <header>
    <a href="/help" style="cursor:pointer">Help</a>
    <span class="oxd-userdropdown-tab" style="cursor:pointer">
      <img alt="profile picture" class="oxd-userdropdown-img" src="/photo.jpg">
      <p class="oxd-userdropdown-name">labubu user</p>
      <i class="oxd-icon bi-caret-down-fill oxd-userdropdown-icon"></i>
    </span>
  </header>
</body></html>
"""


@pytest.mark.asyncio
async def test_get_snapshot_finds_orangehrm_style_userdropdown_with_no_standard_attributes():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(_HTML_ORANGEHRM_USERDROPDOWN)

        elements, _ = await get_snapshot(page)

        await browser.close()

    labels = [e["label"] for e in elements]
    userdropdown_label = next((l for l in labels if "labubu user" in l), None)
    assert userdropdown_label is not None, f"userdropdown span never got indexed at all — labels: {labels}"
    assert "[เมนูโปรไฟล์/บัญชีผู้ใช้ — User Profile Menu]" in userdropdown_label
    # ปุ่ม "Help" ที่แท็กมาตรฐาน (<a>) ต้องยังติด index ปกติเหมือนเดิม แค่ไม่มี marker พิเศษ
    help_label = next((l for l in labels if "Help" in l), None)
    assert help_label is not None
    assert "User Profile Menu" not in help_label
    # child ข้างใน (<img>/<p>/<i>) ตรงกับ class pattern เดียวกันด้วยตัวเอง (ยืนยันจริงจาก DOM
    # ของ opensource-demo.orangehrmlive.com) — ต้องไม่ได้ index แยกซ้ำอีก 3 อันสำหรับพื้นที่
    # คลิกเดียวกัน มีแค่ span ตัวนอกสุดตัวเดียวเท่านั้นที่ได้ marker/index
    assert len(elements) == 2  # แค่ "Help" กับ userdropdown span เท่านั้น
    assert "User Profile Menu" not in help_label


@pytest.mark.asyncio
async def test_get_snapshot_profile_menu_class_without_pointer_cursor_not_marked():
    """class ตรงกับ pattern แต่ไม่มี cursor:pointer (ไม่ใช่ element ที่กดได้จริง — เช่น
    <div class="user-profile-container"> ที่แค่ห่อ layout เฉยๆ) ต้องไม่ถูกจับ/ไม่ติด index"""
    html = """
    <html><body>
      <div class="user-profile-container">
        <span>Just some layout text, not clickable</span>
      </div>
    </body></html>
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(html)

        elements, _ = await get_snapshot(page)

        await browser.close()

    assert elements == []


@pytest.mark.asyncio
async def test_get_snapshot_profile_menu_element_with_title_not_duplicated():
    """element ที่ตรงทั้ง class pattern (userdropdown) และมี title/aria-label อยู่แล้ว (ตรง
    ICON_LABEL_SELECTOR ไปแล้วตั้งแต่ pass แรก) ต้องได้ index เดียว ไม่ใช่สองอันซ้ำกัน"""
    html = """
    <html><body>
      <span class="user-dropdown-tab" title="Account menu" style="cursor:pointer">Somchai</span>
    </body></html>
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(html)

        elements, _ = await get_snapshot(page)

        await browser.close()

    assert len(elements) == 1
    assert "[เมนูโปรไฟล์/บัญชีผู้ใช้ — User Profile Menu]" in elements[0]["label"]


# <img title="..." style="cursor:pointer"> ที่ซ้อนอยู่ใน <a> ที่คลิกได้จริงอยู่แล้ว —
# ต้องได้ index แค่ตัว <a> (จาก selectors มาตรฐาน) ไม่ใช่ img ข้างในด้วย (กันได้ index
# ซ้ำสองอันสำหรับพื้นที่คลิกเดียวกัน เหมือนเคส container-wraps-button ด้านบน)
_HTML_ICON_IMG_NESTED_INSIDE_ANCHOR = """
<html><body>
  <a href="/" title="Home" style="display:inline-block;width:20px;height:20px">
    <img src="logo.png" alt="Logo" title="Home Logo" style="cursor:pointer">
  </a>
</body></html>
"""


@pytest.mark.asyncio
async def test_get_snapshot_does_not_duplicate_icon_img_nested_inside_anchor():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(_HTML_ICON_IMG_NESTED_INSIDE_ANCHOR)

        elements, _ = await get_snapshot(page)

        await browser.close()

    assert len(elements) == 1
    assert elements[0]["tag"] == "a"


# หลังใส่สินค้าลงตะกร้าจริง ปุ่มตะกร้าจะมี badge span ลูกที่มีแค่ตัวเลข (เช่น
# "1") เป็น innerText — เดิม innerText ที่ไม่ว่างจะชนะ fallback ทุกตัวไปเลย
# ทำให้ label กลายเป็นแค่ "1" ไม่สื่อว่านี่คือปุ่มตะกร้า ต้องผสมกับ data-test
_HTML_CART_WITH_BADGE = """
<html><body>
  <a class="shopping_cart_link" data-test="shopping-cart-link" href="/cart.html"
     style="display:inline-block;width:20px;height:20px;background:gray">
    <span data-test="shopping-cart-badge">1</span>
  </a>
</body></html>
"""


@pytest.mark.asyncio
async def test_get_snapshot_combines_badge_counter_with_semantic_label():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(_HTML_CART_WITH_BADGE)

        elements, _ = await get_snapshot(page)

        await browser.close()

    labels = [e["label"] for e in elements]
    assert "shopping cart link (1)" in labels


# บางเว็บใส่ tabindex/role บน badge span เพื่อ accessibility ทำให้ badge เอง
# ก็ match selector list ([tabindex]) และกลายเป็น candidate node แยกต่างหาก —
# ถ้าไม่กันไว้ จะได้ index ซ้อน 2 อัน (พ่อ + badge ลูก) ชี้ไปที่สิ่งเดียวกัน หรือ
# แย่กว่านั้นคือ index ชี้ไปที่ span เล็กๆ ที่คลิกไม่โดน handler ของลิงก์จริง —
# ต้องขยับ index ไปแปะที่ตัวพ่อที่คลิกได้จริง (a.shopping_cart_link) แทนเสมอ
_HTML_BADGE_ITSELF_MATCHES_SELECTOR = """
<html><body>
  <a class="shopping_cart_link" data-test="shopping-cart-link" href="/cart.html"
     style="display:inline-block;width:20px;height:20px;background:gray">
    <span class="shopping_cart_badge" tabindex="-1">1</span>
  </a>
</body></html>
"""


@pytest.mark.asyncio
async def test_get_snapshot_redirects_badge_index_to_clickable_ancestor():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(_HTML_BADGE_ITSELF_MATCHES_SELECTOR)

        elements, _ = await get_snapshot(page)

        await browser.close()

    # ต้องมี element เดียว (ไม่ใช่ 2 อันซ้อนกันสำหรับพ่อ+badge)
    assert len(elements) == 1
    assert elements[0]["tag"] == "a"
    assert elements[0]["label"] == "shopping cart link (1)"


# W9[A]: element ที่ "มองเห็นได้" ตาม CSS (visibility/display/opacity/ขนาดปกติ) แต่มี
# element อื่นวางทับอยู่จริง (เช่น modal/cookie-banner ที่ z-index สูงคลุมทั้งหน้า) ต้อง
# ถูกแปะ marker ในป้าย — getBoundingClientRect()/CSS visibility อย่างเดียวจับเคสนี้
# ไม่ได้เพราะเช็คแค่ตัว element เอง ไม่เช็คว่ามีอะไรวางทับอยู่ข้างบน
_HTML_WITH_OVERLAY = """
<html><body>
  <button id="covered-btn" style="position:absolute; top:100px; left:100px; width:100px; height:40px;">Covered Button</button>
  <div style="position:absolute; top:80px; left:80px; width:200px; height:100px; background:white; z-index:10;"></div>
  <button id="free-btn" style="position:absolute; top:300px; left:100px; width:100px; height:40px;">Free Button</button>
</body></html>
"""


@pytest.mark.asyncio
async def test_get_snapshot_marks_element_obscured_by_overlay():
    """element ที่ถูกบังจริง (covered-btn) ต้องมี marker '[ถูกบังอยู่]' ในป้าย —
    element ที่ไม่ถูกบัง (free-btn) ต้องไม่มี marker นี้ปนมาด้วย"""
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(_HTML_WITH_OVERLAY)

        elements, _ = await get_snapshot(page)

        await browser.close()

    covered = next(e for e in elements if "Covered Button" in e["label"])
    free = next(e for e in elements if "Free Button" in e["label"])

    assert "[ถูกบังอยู่]" in covered["label"]
    assert "[ถูกบังอยู่]" not in free["label"]


# บั๊กที่เจอจริงระหว่างต่อ W10[D] (แสดงชื่อ element แทน index ใน Log panel): เดิม
# data-ai-index ที่แปะไว้จาก get_snapshot() รอบก่อนไม่เคยถูกเคลียร์ — get_snapshot()
# รอบถัดไปบนหน้าเดิม (ไม่มี navigation คั่น เช่น orchestrator.py สั่ง "fill" สองครั้ง
# ติดกัน) เจอ element ที่มี data-ai-index ค้างอยู่แล้วจากรอบก่อน แล้วเข้าใจผิดว่า "แปะ
# index ไปแล้วในรอบนี้" (guard ที่ตั้งใจกันแปะซ้ำ "ภายในรอบเดียวกัน" ระหว่างเช็ค
# badge-ก่อนไปตัวพ่อ) จึง skip element นั้นออกจาก elements list ของรอบใหม่ไปเงียบๆ —
# ทำให้ snapshot ที่สองบนหน้าเดิมได้ elements น้อยลง/ว่างเปล่า ทั้งที่ element ยังอยู่จริง
_HTML_SIMPLE_LOGIN_FORM = """
<html><body>
  <input id="user" placeholder="Username">
  <input id="pass" type="password" placeholder="Password">
  <button id="login">Login</button>
</body></html>
"""


@pytest.mark.asyncio
async def test_get_snapshot_returns_same_elements_on_repeated_calls_without_navigation():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(_HTML_SIMPLE_LOGIN_FORM)

        elements1, _ = await get_snapshot(page)
        elements2, _ = await get_snapshot(page)  # ไม่มี navigation คั่นกลาง — หน้าเดิม

        await browser.close()

    labels1 = sorted(e["label"] for e in elements1)
    labels2 = sorted(e["label"] for e in elements2)
    assert labels1 == ["Login", "Password", "Username"]
    assert labels2 == labels1


# ---------------- W40: element ที่อยู่ใน <iframe> (รวม iframe ซ้อนกันหลายชั้น) ----------------
# บั๊กที่ user รายงานจริงบน uitestingplayground.com/frames: Outer Frame (Level 1) ซ้อน Inner
# Frame (Level 2) แต่ละชั้นมีปุ่ม Edit/Submit/Click me/Primary — เดิม document.
# querySelectorAll() ของ main frame มองไม่เห็น element ใน <iframe> เลย (คนละ document
# object กันโดยสิ้นเชิง แม้ same-origin) agent จึงมองไม่เห็นปุ่มพวกนี้เลยสักตัว แล้ววนลูป
# กดกลับไปหน้า nav link ที่มีอยู่จริง (Home/Frames/Resources) ไม่รู้จบ

_HTML_MAIN_FRAME_ONLY_BUTTON = """
<html><body>
  <button id="main-btn">Main Button</button>
</body></html>
"""


@pytest.mark.asyncio
async def test_get_snapshot_still_works_normally_when_there_are_no_iframes():
    """หน้าที่ไม่มี iframe เลย — page.frames มีแค่ [main_frame] ตัวเดียว ต้องได้ผลลัพธ์
    เหมือนเดิมทุกประการ (ไม่มี regression จากการเปลี่ยนมา loop ผ่าน page.frames)"""
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(_HTML_MAIN_FRAME_ONLY_BUTTON)

        elements, text_repr = await get_snapshot(page)

        await browser.close()

    assert len(elements) == 1
    assert elements[0]["label"] == "Main Button"
    assert elements[0]["index"] == 0
    assert "[0] button 'Main Button'" in text_repr


_HTML_SINGLE_IFRAME = """
<html><body>
  <button id="main-btn">Main Button</button>
  <iframe srcdoc="<html><body><button id='inner-btn'>Inner Button</button></body></html>"></iframe>
</body></html>
"""


@pytest.mark.asyncio
async def test_get_snapshot_finds_button_inside_iframe():
    """W40: ปุ่มใน <iframe> ต้องปรากฏใน snapshot ด้วย ไม่ใช่แค่ main document"""
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(_HTML_SINGLE_IFRAME)

        elements, _ = await get_snapshot(page)

        await browser.close()

    labels = {e["label"] for e in elements}
    assert "Main Button" in labels
    assert "Inner Button" in labels


@pytest.mark.asyncio
async def test_get_snapshot_assigns_continuous_indices_across_frames():
    """W40: index ต้องเรียงต่อกันไม่ชนกันข้าม frame (main frame ก่อนเสมอ) — agent อ้างอิง
    element ด้วย index เดียวทั้งหน้า ไม่แยกตาม frame เลย"""
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(_HTML_SINGLE_IFRAME)

        elements, _ = await get_snapshot(page)

        await browser.close()

    indices = sorted(e["index"] for e in elements)
    assert indices == [0, 1]
    main_btn = next(e for e in elements if e["label"] == "Main Button")
    assert main_btn["index"] == 0  # main frame ต้องมาก่อนเสมอ


_HTML_NESTED_IFRAMES = """
<html><body>
  <h1>Playground</h1>
  <iframe srcdoc="
    <html><body>
      <button id='edit1'>Edit</button>
      <button id='submit1'>Submit</button>
      <iframe srcdoc='&lt;html&gt;&lt;body&gt;&lt;button id=edit2&gt;Edit&lt;/button&gt;&lt;button id=submit2&gt;Submit&lt;/button&gt;&lt;/body&gt;&lt;/html&gt;'></iframe>
    </body></html>
  "></iframe>
</body></html>
"""


@pytest.mark.asyncio
async def test_get_snapshot_finds_buttons_inside_nested_iframes_two_levels_deep():
    """W40: จำลองหน้า uitestingplayground.com/frames ที่ user รายงานจริง (Outer Frame ซ้อน
    Inner Frame อีกชั้น แต่ละชั้นมีปุ่ม Edit/Submit ของตัวเอง) — page.frames คืนทุก frame
    แบบ flat รวม nested เอง (ไม่ต้อง recurse เอง) ต้องเจอปุ่มครบทั้ง 4 ปุ่ม (2 ชั้น x 2 ปุ่ม)
    ทุก index ต้อง unique ไม่ชนกัน"""
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(_HTML_NESTED_IFRAMES)

        elements, _ = await get_snapshot(page)

        await browser.close()

    edit_buttons = [e for e in elements if e["label"] == "Edit"]
    submit_buttons = [e for e in elements if e["label"] == "Submit"]
    assert len(edit_buttons) == 2
    assert len(submit_buttons) == 2
    all_indices = [e["index"] for e in elements]
    assert len(all_indices) == len(set(all_indices))  # ไม่มี index ซ้ำกันเลย


@pytest.mark.asyncio
async def test_resolve_frame_returns_page_when_element_in_main_document():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(_HTML_SINGLE_IFRAME)

        target = await resolve_frame(page, "#main-btn")

        await browser.close()

    assert target is page


@pytest.mark.asyncio
async def test_resolve_frame_returns_child_frame_when_element_inside_iframe():
    """W40: element ที่อยู่ใน <iframe> เท่านั้น ต้อง resolve ไปที่ Frame object ของ iframe
    นั้น ไม่ใช่ page (main frame) — ให้ backend/app/core/actions.py กดผ่าน frame ที่ถูกต้อง"""
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(_HTML_SINGLE_IFRAME)

        target = await resolve_frame(page, "#inner-btn")

        await browser.close()

    assert target is not page
    assert target != page.main_frame


@pytest.mark.asyncio
async def test_resolve_frame_falls_back_to_page_when_selector_not_found_anywhere():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(_HTML_SINGLE_IFRAME)

        target = await resolve_frame(page, "#does-not-exist-anywhere")

        await browser.close()

    assert target is page


# ---------------- Lane 1/2: อ่าน "เนื้อหา" หน้าเว็บ (นับ/ตาราง) ----------------
# get_snapshot() ด้านบนกรองเอาเฉพาะ element ที่คลิกได้ ไม่มีเส้นทางอ่านเนื้อหาเลย —
# count_elements()/extract_table_data() เป็นเส้นทางแยกที่ agent เรียกผ่าน tool
# "read_page_data" (backend/app/core/actions.py) เฉพาะตอนจำเป็นจริงๆ เท่านั้น

_HTML_PRODUCT_TABLE = """
<html><body>
  <table id="products">
    <tr><th>Name</th><th>Price</th></tr>
    <tr><td>Widget</td><td>$9.99</td></tr>
    <tr><td>Gadget</td><td>$19.99</td></tr>
    <tr><td>Gizmo</td><td>$29.99</td></tr>
  </table>
</body></html>
"""


@pytest.mark.asyncio
async def test_count_elements_counts_table_rows():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(_HTML_PRODUCT_TABLE)

        count = await count_elements(page, "#products tr")

        await browser.close()

    assert count == 4  # 1 header row + 3 product row


@pytest.mark.asyncio
async def test_count_elements_returns_zero_for_selector_with_no_match():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(_HTML_PRODUCT_TABLE)

        count = await count_elements(page, ".does-not-exist")

        await browser.close()

    assert count == 0


@pytest.mark.asyncio
async def test_count_elements_does_not_throw_on_invalid_css_selector():
    """selector ผิดรูปแบบ (invalid CSS) ต้องไม่ throw ออกไป — deterministic, ไม่ควรทำให้
    ทั้ง action ล้มเหลวเพราะ syntax error ของ selector ที่ LLM เดามาผิด"""
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(_HTML_PRODUCT_TABLE)

        count = await count_elements(page, "###not-valid-css(((")

        await browser.close()

    assert count == 0


_HTML_TABLE_INSIDE_IFRAME = """
<html><body>
  <iframe srcdoc="<html><body><table id='inner-products'><tr><td>A</td></tr><tr><td>B</td></tr></table></body></html>"></iframe>
</body></html>
"""


@pytest.mark.asyncio
async def test_count_elements_counts_across_iframes():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(_HTML_TABLE_INSIDE_IFRAME)

        count = await count_elements(page, "#inner-products tr")

        await browser.close()

    assert count == 2


@pytest.mark.asyncio
async def test_extract_table_data_returns_markdown_table():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(_HTML_PRODUCT_TABLE)

        result = await extract_table_data(page, "#products")

        await browser.close()

    assert result == (
        "| Name | Price |\n"
        "| --- | --- |\n"
        "| Widget | $9.99 |\n"
        "| Gadget | $19.99 |\n"
        "| Gizmo | $29.99 |"
    )


_HTML_MESSY_TABLE = """
<html><body>
  <table id="messy">
    <tr><th>   Name  </th><th>Price</th></tr>
    <tr><td>
      Widget
    </td><td>  $9.99  </td></tr>
  </table>
</body></html>
"""


@pytest.mark.asyncio
async def test_extract_table_data_strips_extra_whitespace():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(_HTML_MESSY_TABLE)

        result = await extract_table_data(page, "#messy")

        await browser.close()

    assert result == "| Name | Price |\n| --- | --- |\n| Widget | $9.99 |"


# Text Normalization: ตัด whitespace ส่วนเกินเท่านั้น (clean() ใน _EXTRACT_TABLE_JS) ต้อง
# ไม่เผลอตัดอักขระพิเศษที่มีความหมายจริงทิ้งไปด้วย เช่น "@"/"." ในอีเมล หรือ ","/"." ในตัวเลข
# เงินเดือน — บั๊กที่ user รายงานจริง (มองไม่เห็นอีเมลในตาราง)
_HTML_TABLE_WITH_EMAIL_AND_SALARY = """
<html><body>
  <table id="employees">
    <tr><th>Email</th><th>Salary</th></tr>
    <tr><td>cierra.vega@example.com</td><td>$12,345.67</td></tr>
  </table>
</body></html>
"""


@pytest.mark.asyncio
async def test_extract_table_data_preserves_email_and_currency_special_characters():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(_HTML_TABLE_WITH_EMAIL_AND_SALARY)

        result = await extract_table_data(page, "#employees")

        await browser.close()

    assert "cierra.vega@example.com" in result
    assert "$12,345.67" in result


_HTML_PRODUCT_LIST = """
<html><body>
  <ul id="cart-items">
    <li>Widget x2</li>
    <li>Gadget x1</li>
  </ul>
</body></html>
"""


@pytest.mark.asyncio
async def test_extract_table_data_returns_compact_json_for_non_table_list():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(_HTML_PRODUCT_LIST)

        result = await extract_table_data(page, "#cart-items")

        await browser.close()

    assert json.loads(result) == ["Widget x2", "Gadget x1"]


@pytest.mark.asyncio
async def test_extract_table_data_falls_back_to_real_table_when_hint_selector_does_not_match():
    """Fallback Extraction Protocol: LLM เดา target_hint ผิด (get_snapshot() ไม่เคยโชว์
    class/id ของตารางให้เห็นเลย เดาได้แค่จาก context อื่น) แต่หน้านี้มีตารางข้อมูลจริงอยู่
    (#products) — ต้องอ่าน td/th ของตารางจริงมาตอบ ไม่ใช่ยอมแพ้บอกว่า "ไม่พบข้อมูล" ทั้งที่
    มีข้อมูลอยู่บนหน้าจริงๆ"""
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(_HTML_PRODUCT_TABLE)

        result = await extract_table_data(page, "#does-not-exist")

        await browser.close()

    assert "| Widget | $9.99 |" in result
    assert not result.startswith("[FAIL]")


_HTML_NO_TABLE_OR_LIST = """
<html><body>
  <p>ไม่มีตารางหรือ list อะไรบนหน้านี้เลย</p>
</body></html>
"""


@pytest.mark.asyncio
async def test_extract_table_data_fails_gracefully_when_page_has_no_table_or_list_at_all():
    """selector เดาผิด "และ" ไม่มีตาราง/list จริงอยู่บนหน้าเลยสักอัน (fallback ก็หาไม่เจอ) —
    กรณีนี้ต้องยัง fail อยู่เหมือนเดิม ไม่ใช่ fallback ไปเจออะไรมั่วๆ"""
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(_HTML_NO_TABLE_OR_LIST)

        result = await extract_table_data(page, "#does-not-exist")

        await browser.close()

    assert result.startswith("[FAIL]")


_HTML_MULTIPLE_TABLES = """
<html><body>
  <table id="tiny"><tr><th>X</th></tr><tr><td>1</td></tr></table>
  <table id="real-data">
    <tr><th>First Name</th><th>Last Name</th><th>Email</th></tr>
    <tr><td>Cierra</td><td>Vega</td><td>cierra@example.com</td></tr>
    <tr><td>Alden</td><td>Cantrell</td><td>alden@example.com</td></tr>
  </table>
</body></html>
"""


@pytest.mark.asyncio
async def test_extract_table_data_fallback_picks_table_with_most_rows_when_several_exist():
    """หน้าเดียวกันมีหลายตาราง (เช่น ตารางเล็กๆ ตกแต่ง layout ปนกับตารางข้อมูลจริง) — fallback
    ต้องเลือกตัวที่มีข้อมูลเยอะที่สุด (น่าจะเป็นตารางข้อมูลจริง) ไม่ใช่ตัวแรกที่เจอเฉยๆ"""
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(_HTML_MULTIPLE_TABLES)

        result = await extract_table_data(page, "#does-not-exist")

        await browser.close()

    assert "cierra@example.com" in result
    assert "alden@example.com" in result


# ---------------- fuzzy_find: กันพิมพ์ผิดเล็กน้อยที่ชั้น data lookup ----------------


def test_fuzzy_find_returns_closest_match_for_minor_typo():
    """SequenceMatcher.ratio() คิดจากความยาวรวมทั้งสองฝั่งด้วย เทียบ query สั้นๆ กับ
    candidate เต็มชื่อยาวๆ ตรงๆ จะได้ ratio ต่ำเกินจริงเสมอ (ไม่ว่า partial match จะดีแค่ไหน)
    — เทียบกับ token เดี่ยวๆ ที่ความยาวใกล้เคียงกันแทน (เหมือนตัวอย่างจริงที่ user รายงาน:
    "vaga"/"Vega")"""
    result = fuzzy_find("vaga", ["Vega", "Smith", "Patel"])

    assert result == "Vega"


def test_fuzzy_find_matches_full_name_with_minor_typo():
    result = fuzzy_find("Cierra Vaga", ["Cierra Vega", "John Smith", "Priya Patel"])

    assert result == "Cierra Vega"


def test_fuzzy_find_returns_none_when_nothing_passes_threshold():
    """กันจับผิดคนข้าม record — ชื่อที่ไม่เกี่ยวข้องกันเลยต้องไม่ถูกเสนอเป็น match"""
    result = fuzzy_find("xyz completely unrelated", ["Cierra Vega", "John Smith"])

    assert result is None


def test_fuzzy_find_respects_custom_threshold():
    """threshold สูงขึ้น = เข้มขึ้น — คำที่เคยผ่าน threshold ต่ำ อาจไม่ผ่าน threshold สูงกว่า
    ("vaga" vs "Vega" ให้ ratio = 0.75 พอดี — ผ่าน threshold 0.5 แต่ไม่ผ่าน 0.9)"""
    assert fuzzy_find("vaga", ["Vega"], threshold=0.5) == "Vega"
    assert fuzzy_find("vaga", ["Vega"], threshold=0.9) is None


# ---------------- extract_table_data(query=...): exact match ก่อนเสมอ ไม่ข้ามไป fuzzy ----------------

_HTML_USER_TABLE = """
<html><body>
  <table id="users">
    <tr><th>Name</th><th>Age</th></tr>
    <tr><td>Cierra Vega</td><td>32</td></tr>
    <tr><td>John Smith</td><td>45</td></tr>
  </table>
</body></html>
"""


@pytest.mark.asyncio
async def test_extract_table_data_returns_untouched_when_query_matches_exactly():
    """query ที่ match ตรงเป๊ะ (substring) ต้องไม่ถูกแปะ annotation fuzzy ใดๆ เลย — พิสูจน์ว่า
    ลอง exact ก่อนเสมอ ไม่ข้ามไป fuzzy ทันทีทั้งที่ไม่จำเป็น"""
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(_HTML_USER_TABLE)

        result = await extract_table_data(page, "#users", query="Cierra Vega")

        await browser.close()

    assert "ใกล้เคียงกับคำค้น" not in result
    assert "| Cierra Vega | 32 |" in result


@pytest.mark.asyncio
async def test_extract_table_data_falls_back_to_fuzzy_when_exact_match_missing():
    """query สะกดผิดเล็กน้อย ("Cierra Vaga") ไม่ตรง exact กับแถวไหนเลย — ต้อง fallback ไป
    fuzzy_find แล้วแนบ annotation บอกความต่างชัดเจน ไม่ใช่แกล้งทำเป็นตรงกันเป๊ะ"""
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(_HTML_USER_TABLE)

        result = await extract_table_data(page, "#users", query="Cierra Vaga")

        await browser.close()

    assert "พบ 'Cierra Vega' ใกล้เคียงกับคำค้น 'Cierra Vaga'" in result
    assert "| Cierra Vega | 32 |" in result


@pytest.mark.asyncio
async def test_extract_table_data_fails_when_query_matches_nothing_even_fuzzy():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(_HTML_USER_TABLE)

        result = await extract_table_data(page, "#users", query="ไม่เกี่ยวข้องกันเลยสักนิด")

        await browser.close()

    assert result.startswith("[FAIL]")


@pytest.mark.asyncio
async def test_extract_table_data_without_query_keeps_old_behavior():
    """ไม่ส่ง query มาเลย (default "") ต้องได้ผลลัพธ์เดิมทุกประการ (ไม่มี lookup/annotation
    ใดๆ ปนมา) — regression check กับพฤติกรรมเดิมก่อนเพิ่ม fuzzy matching"""
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(_HTML_USER_TABLE)

        result = await extract_table_data(page, "#users")

        await browser.close()

    assert "ใกล้เคียงกับคำค้น" not in result
    assert "| Cierra Vega | 32 |" in result
    assert "| John Smith | 45 |" in result


# ---------------- hover-to-reveal action buttons (opacity:0/visibility:hidden) ----------------
# บั๊กที่ user รายงานจริงบน uitestingplayground.com/scrolltoclick Case 4 "Hover to Reveal":
# ปุ่ม flag แต่ละแถวใช้ visibility:hidden จนกว่าจะ hover แถวแม่ ทำให้ perception เดิม
# (visibility !== 'hidden' เป็นเงื่อนไข "มองเห็น") กรองทิ้งไปเลย agent ไม่มีทาง index ให้กด
# ได้ตั้งแต่ต้น — ต้องยังติด index ให้ปุ่ม/ลิงก์ที่ซ่อนด้วย CSS ของตัวเอง (ไม่ใช่ display:none)
# ตราบใดที่ bounding box ไม่เป็น 0x0 จริง แต่ display:none ยังต้องกรองทิ้งเหมือนเดิม (ไม่ layout
# เลย ไม่มีทางคลิกได้จริงไม่ว่า CSS state ไหน) และต้องจำกัดแค่ปุ่ม/ลิงก์เท่านั้น (กัน hidden
# <input> ที่เป็น honeypot/CSRF token จริงๆ ไม่ให้ agent เผลอไปกรอก)

_HTML_HOVER_REVEAL_OPACITY_ZERO = """
<html><body>
  <div style="position:relative; width:300px; height:40px;">
    <span>Weekly status report</span>
    <button style="position:absolute; top:0; left:200px; width:60px; height:30px; opacity:0;">
      Flag
    </button>
  </div>
</body></html>
"""


@pytest.mark.asyncio
async def test_get_snapshot_indexes_hover_reveal_button_hidden_by_opacity_zero():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(_HTML_HOVER_REVEAL_OPACITY_ZERO)

        elements, _ = await get_snapshot(page)

        await browser.close()

    assert len(elements) == 1
    assert elements[0]["tag"] == "button"
    assert elements[0]["label"] == "Flag [ซ่อนอยู่ — อาจต้อง hover แถวก่อน]"


_HTML_HOVER_REVEAL_VISIBILITY_HIDDEN = """
<html><body>
  <div style="position:relative; width:300px; height:40px;">
    <span>Weekly status report</span>
    <button style="position:absolute; top:0; left:200px; width:60px; height:30px; visibility:hidden;">
      Flag
    </button>
  </div>
</body></html>
"""


@pytest.mark.asyncio
async def test_get_snapshot_indexes_hover_reveal_button_hidden_by_visibility_hidden():
    """เคสที่เจอจริงบน uitestingplayground.com/scrolltoclick — ปุ่ม flag ใช้
    visibility:hidden ไม่ใช่ opacity:0 (คนละ CSS property แต่ต้องได้ผลลัพธ์เดียวกัน)"""
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(_HTML_HOVER_REVEAL_VISIBILITY_HIDDEN)

        elements, _ = await get_snapshot(page)

        await browser.close()

    assert len(elements) == 1
    assert elements[0]["tag"] == "button"
    assert elements[0]["label"] == "Flag [ซ่อนอยู่ — อาจต้อง hover แถวก่อน]"


_HTML_DISPLAY_NONE_BUTTON = """
<html><body>
  <button style="width:60px; height:30px; display:none;">Truly Hidden</button>
</body></html>
"""


@pytest.mark.asyncio
async def test_get_snapshot_still_filters_out_display_none_button():
    """display:none ต้องยังถูกกรองทิ้งเหมือนเดิมทุกประการ (ไม่ให้ relax เกินไป) — browser
    ไม่ layout element นี้เลย ไม่มีทางคลิกได้จริงไม่ว่า CSS state ไหนจะเปลี่ยนก็ตาม"""
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(_HTML_DISPLAY_NONE_BUTTON)

        elements, _ = await get_snapshot(page)

        await browser.close()

    assert elements == []


_HTML_HIDDEN_INPUT_HONEYPOT = """
<html><body>
  <input type="text" name="honeypot" style="width:60px; height:20px; opacity:0;">
</body></html>
"""


@pytest.mark.asyncio
async def test_get_snapshot_does_not_relax_filter_for_hidden_input():
    """การผ่อนกฎ (opacity:0/visibility:hidden แต่ bounding box ไม่เป็น 0x0) ต้องจำกัดแค่
    ปุ่ม/ลิงก์ (a/button/[role=button]) เท่านั้น — <input> ที่ซ่อนแบบเดียวกันมักเป็น
    honeypot/CSRF token ที่ตั้งใจซ่อนถาวร ไม่ใช่รอ hover ต้องยังถูกกรองทิ้งเหมือนเดิม"""
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(_HTML_HIDDEN_INPUT_HONEYPOT)

        elements, _ = await get_snapshot(page)

        await browser.close()

    assert elements == []


@pytest.mark.asyncio
async def test_get_snapshot_overlay_detection_still_works_alongside_hover_reveal_marker():
    """overlay detection ("[ถูกบังอยู่]" จาก W9[A]) เป็นคนละเงื่อนไขกับ hover-reveal marker
    ใหม่นี้เลย — element ที่ visible ปกติแต่ถูกอีก element บังไว้ ต้องยังได้ marker เดิม
    ไม่ใช่ hover-reveal marker (ซึ่งไม่เข้าเงื่อนไขเพราะ opacity/visibility ปกติ)"""
    html = """
    <html><body>
      <button id="covered-btn" style="position:absolute; top:100px; left:100px; width:100px; height:40px;">Covered Button</button>
      <div style="position:absolute; top:80px; left:80px; width:200px; height:100px; background:white; z-index:10;"></div>
    </body></html>
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(html)

        elements, _ = await get_snapshot(page)

        await browser.close()

    assert len(elements) == 1
    assert elements[0]["label"] == "Covered Button [ถูกบังอยู่]"


# ---------------- W19 ("Scoped Search Context"): region tagging (main vs navigation) ----------------

_HTML_WITH_DUPLICATE_SEARCH_LABELS = """
<html><body>
  <aside class="oxd-sidepanel">
    <input type="text" placeholder="Search" id="sidebar-search">
  </aside>
  <main>
    <form>
      <input type="text" placeholder="Search" id="main-search">
    </form>
  </main>
</body></html>
"""


@pytest.mark.asyncio
async def test_get_snapshot_tags_sidebar_element_as_navigation_region():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(_HTML_WITH_DUPLICATE_SEARCH_LABELS)

        elements, text_repr = await get_snapshot(page)

        await browser.close()

    assert elements[0]["region"] == "navigation"
    assert elements[1]["region"] == "main"
    assert "(navigation)" in text_repr.splitlines()[0]
    assert "(navigation)" not in text_repr.splitlines()[1]


@pytest.mark.asyncio
async def test_get_snapshot_element_outside_main_or_nav_has_empty_region():
    html = '<html><body><button id="loose-btn">Click me</button></body></html>'
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(html)

        elements, text_repr = await get_snapshot(page)

        await browser.close()

    assert elements[0]["region"] == ""
    assert "(navigation)" not in text_repr


@pytest.mark.asyncio
async def test_get_snapshot_nested_nav_inside_main_is_still_navigation_region():
    """breadcrumb <nav> ที่ซ้อนอยู่ใน <main> ต้องยังนับเป็น "navigation" (ancestor ที่ใกล้
    ตัว element ที่สุดเป็นตัวตัดสิน ไม่ใช่ ancestor ไกลสุด)"""
    html = """
    <html><body>
      <main>
        <nav><a href="/">Home</a></nav>
      </main>
    </body></html>
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(html)

        elements, _ = await get_snapshot(page)

        await browser.close()

    assert elements[0]["region"] == "navigation"


# ---------------- W19 ("Exact Element Matching"): <label> association ----------------


@pytest.mark.asyncio
async def test_get_snapshot_uses_label_for_attribute_as_input_label():
    html = """
    <html><body>
      <label for="emp-name">Employee Name</label>
      <input type="text" id="emp-name" placeholder="Type for hints...">
    </body></html>
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(html)

        elements, _ = await get_snapshot(page)

        await browser.close()

    assert elements[0]["label"] == "Employee Name"


@pytest.mark.asyncio
async def test_get_snapshot_uses_wrapping_label_text_as_input_label():
    """<select> ที่มี option เลือกอยู่แล้วต้องยังโชว์ค่าที่เลือกอยู่ชนะ associatedLabel เสมอ
    (พฤติกรรมเดิม, ดู comment ใน perception.py) — ใช้ <textarea> ว่างเปล่าแทนเพื่อทดสอบ
    associatedLabel ตรงๆ โดยไม่ชนกับกฎ "ค่าที่กรอกอยู่จริงชนะเสมอ" ข้อนั้น"""
    html = """
    <html><body>
      <label>Comments
        <textarea id="comments-box"></textarea>
      </label>
    </body></html>
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(html)

        elements, _ = await get_snapshot(page)

        await browser.close()

    assert elements[0]["label"] == "Comments"


@pytest.mark.asyncio
async def test_get_snapshot_select_with_option_selected_still_shows_current_value_over_label():
    """ยืนยันพฤติกรรมเดิมไม่เปลี่ยน: select ที่มีค่าปัจจุบันอยู่แล้วต้องโชว์ค่านั้น (ไม่ใช่
    associatedLabel) เพราะ "ค่าที่เลือกอยู่จริง" มีประโยชน์กว่าสำหรับ dropdown ที่ใช้บ่อย
    (เช่น sort-order dropdown)"""
    html = """
    <html><body>
      <label>User Role
        <select id="role-select"><option>Admin</option></select>
      </label>
    </body></html>
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(html)

        elements, _ = await get_snapshot(page)

        await browser.close()

    assert elements[0]["label"] == "Admin"


@pytest.mark.asyncio
async def test_get_snapshot_falls_back_to_placeholder_when_no_associated_label():
    html = '<html><body><input type="text" placeholder="Search"></body></html>'
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(html)

        elements, _ = await get_snapshot(page)

        await browser.close()

    assert elements[0]["label"] == "Search"


# ---------------- W19 ("Log Cleanliness"): already-active nav/tab marker ----------------


@pytest.mark.asyncio
async def test_get_snapshot_marks_active_nav_link_with_aria_current():
    html = """
    <html><body>
      <nav>
        <a href="/admin" aria-current="page">Admin</a>
        <a href="/reports">Reports</a>
      </nav>
    </body></html>
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(html)

        elements, _ = await get_snapshot(page)

        await browser.close()

    admin = next(e for e in elements if "Admin" in e["label"])
    reports = next(e for e in elements if "Reports" in e["label"])
    assert "[active อยู่แล้ว]" in admin["label"]
    assert "[active อยู่แล้ว]" not in reports["label"]


@pytest.mark.asyncio
async def test_get_snapshot_marks_active_tab_with_active_class():
    html = """
    <html><body>
      <div role="tab" class="oxd-topbar-body-nav-tab active">System Users</div>
      <div role="tab" class="oxd-topbar-body-nav-tab">Job Titles</div>
    </body></html>
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(html)

        elements, _ = await get_snapshot(page)

        await browser.close()

    system_users = next(e for e in elements if "System Users" in e["label"])
    job_titles = next(e for e in elements if "Job Titles" in e["label"])
    assert "[active อยู่แล้ว]" in system_users["label"]
    assert "[active อยู่แล้ว]" not in job_titles["label"]


@pytest.mark.asyncio
async def test_get_snapshot_does_not_mark_active_class_outside_nav_or_tab_role():
    """class "active" นอกบริบทเมนู/แท็บ (เช่น ตัวเลือกที่ highlight อยู่ใน custom dropdown/
    autocomplete popup กลางหน้า) ต้องไม่ติด marker — เป็น element ที่ "ควร" คลิกเพื่อเลือก
    ไม่ใช่ตัวที่ควรข้าม"""
    html = """
    <html><body>
      <main>
        <div role="option" class="suggestion active">John Smith</div>
      </main>
    </body></html>
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(html)

        elements, _ = await get_snapshot(page)

        await browser.close()

    assert "[active อยู่แล้ว]" not in elements[0]["label"]


@pytest.mark.asyncio
async def test_get_snapshot_does_not_mark_inactive_nav_link():
    html = '<html><body><nav><a href="/reports">Reports</a></nav></body></html>'
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(html)

        elements, _ = await get_snapshot(page)

        await browser.close()

    assert "[active อยู่แล้ว]" not in elements[0]["label"]

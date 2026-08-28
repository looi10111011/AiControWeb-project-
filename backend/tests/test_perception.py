import json
from unittest.mock import AsyncMock, patch

import pytest
from playwright.async_api import async_playwright

from backend.app.core.perception import _cap_rows, count_elements, extract_table_data, fuzzy_find, get_snapshot, resolve_frame

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
    assert "[Profile/Account Menu]" in userdropdown_label
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
    assert "[Profile/Account Menu]" in elements[0]["label"]


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
    """element ที่ถูกบังจริง (covered-btn) ต้องมี marker '[obscured]' ในป้าย —
    element ที่ไม่ถูกบัง (free-btn) ต้องไม่มี marker นี้ปนมาด้วย"""
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(_HTML_WITH_OVERLAY)

        elements, _ = await get_snapshot(page)

        await browser.close()

    covered = next(e for e in elements if "Covered Button" in e["label"])
    free = next(e for e in elements if "Free Button" in e["label"])

    assert "[obscured]" in covered["label"]
    assert "[obscured]" not in free["label"]


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


_HTML_TABLE_WITH_HIDDEN_ROWS = """
<html><body>
  <table id="products">
    <tr><th>Name</th><th>Price</th></tr>
    <tr><td>Widget</td><td>$9.99</td></tr>
    <tr style="display:none"><td>HiddenByDisplayNone</td><td>$0</td></tr>
    <tr style="visibility:hidden"><td>HiddenByVisibility</td><td>$0</td></tr>
    <tr><td>Gadget</td><td>$19.99</td></tr>
  </table>
</body></html>
"""


@pytest.mark.asyncio
async def test_count_elements_ignores_hidden_rows():
    """W63[3.3]: querySelectorAll(...).length เดิมนับ node ที่ match selector ทุกตัวใน DOM
    ไม่สนว่ามองเห็นได้จริงไหม (ticket Issue 3.3 — ต้องนับเฉพาะแถวที่มองเห็นได้จริง ไม่นับ
    display:none/visibility:hidden)"""
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(_HTML_TABLE_WITH_HIDDEN_ROWS)

        count = await count_elements(page, "#products tr")

        await browser.close()

    assert count == 3  # header + Widget + Gadget เท่านั้น (2 แถวซ่อนไม่นับ)


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


# W64[7.2] ("Add-Action Idempotency Lock" — ticket Issue 7.2, บั๊กจริง: agent ค้นหาแถวที่
# เพิ่งบันทึกไปทันทีหลัง Save โดยไม่รอ AJAX table reload ให้เสร็จก่อน อ่านได้ตารางเก่า เข้าใจ
# ผิดว่าบันทึกไม่สำเร็จ): extract_table_data() ต้องรอสั้นๆ แล้วลองสแกนใหม่อีกครั้งก่อนยอม
# [FAIL] จริงถ้ามี query — patch _LOOKUP_RETRY_WAIT_SEC ให้สั้นลงกันเทสต์ช้าโดยไม่จำเป็น


@pytest.mark.asyncio
async def test_extract_table_data_retries_once_after_wait_when_row_arrives_late():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(_HTML_USER_TABLE)
        # จำลองแถวที่เพิ่ง "บันทึก" มาถึงช้ากว่ารอบสแกนแรกเล็กน้อย (เช่น AJAX reload)
        await page.evaluate(
            """() => {
                setTimeout(() => {
                    const table = document.querySelector('#users');
                    const tr = document.createElement('tr');
                    tr.innerHTML = '<td>Siamyut Phasida</td><td>99</td>';
                    table.appendChild(tr);
                }, 100);
            }"""
        )

        with patch("backend.app.core.perception._LOOKUP_RETRY_WAIT_SEC", 0.5):
            result = await extract_table_data(page, "#users", query="Siamyut Phasida")

        await browser.close()

    assert "Siamyut Phasida" in result
    assert not result.startswith("[FAIL]")


@pytest.mark.asyncio
async def test_extract_table_data_fails_after_one_retry_when_genuinely_not_found():
    """แถวที่หาไม่เคยมาถึงจริงๆ (ไม่ใช่แค่ AJAX ช้า) — ต้อง [FAIL] หลังลองแค่ 1 รอบเพิ่ม ไม่ใช่
    วนรอไม่รู้จบ"""
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(_HTML_USER_TABLE)

        with patch("backend.app.core.perception._LOOKUP_RETRY_WAIT_SEC", 0.05):
            result = await extract_table_data(page, "#users", query="ไม่มีทางเจอแน่นอน")

        await browser.close()

    assert result.startswith("[FAIL]")


@pytest.mark.asyncio
async def test_extract_table_data_does_not_retry_without_query():
    """ไม่มี query (ขอสรุปทั้งตารางเฉยๆ) ต้องไม่มีการ retry/wait ใดๆ เลย (ไม่มีสิ่งที่เรียกว่า
    "หาไม่เจอ" สำหรับโหมดสรุปทั้งก้อน) — mock _extract_table_data_once() โดยตรงแทนที่จะ patch
    asyncio.sleep ทั้ง process (asyncio.sleep ถูกเรียกจากที่อื่นในกระบวนการทดสอบได้เยอะมาก
    เช่น Playwright/pytest-asyncio เอง ทำให้ assert_not_awaited() ไม่น่าเชื่อถือ)"""
    with patch(
        "backend.app.core.perception._extract_table_data_once",
        AsyncMock(return_value="[FAIL] no element matching '#users'"),
    ) as mock_once:
        result = await extract_table_data(AsyncMock(), "#users")

    assert mock_once.await_count == 1
    assert result.startswith("[FAIL]")


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
    assert elements[0]["label"] == "Flag [hidden — may need to hover the row first]"


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
    assert elements[0]["label"] == "Flag [hidden — may need to hover the row first]"


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
    """overlay detection ("[obscured]" จาก W9[A]) เป็นคนละเงื่อนไขกับ hover-reveal marker
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
    assert elements[0]["label"] == "Covered Button [obscured]"


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
    (เช่น sort-order dropdown)

    W_field_label_for_plain_inputs: ตั้งแต่ C4 เป็นต้นไป native <select> ได้ชื่อ field นำหน้า
    ด้วย — **ค่าที่เลือกอยู่ยังอยู่ครบเหมือนเดิม ไม่ถูกทับทิ้ง** ซึ่งคือเจตนาที่เทสต์นี้ปกป้อง
    ตัว prefix เป็นการ *เพิ่ม* ข้อมูล ไม่ใช่แทนที่ (กฎเดียวกับ W_dropdown_field_label ที่ทำกับ
    custom dropdown อยู่ก่อนแล้ว) — assertion เดิมที่คาดว่าไม่มี prefix เป็นจริงเพราะ native
    <select> ถูกกันออกจากกฎนั้นไว้เฉยๆ ไม่ใช่เพราะตั้งใจให้ไม่มีชื่อ field"""
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

    assert elements[0]["label"] == "User Role: Admin"


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
    assert "[already active]" in admin["label"]
    assert "[already active]" not in reports["label"]


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
    assert "[already active]" in system_users["label"]
    assert "[already active]" not in job_titles["label"]


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

    assert "[already active]" not in elements[0]["label"]


@pytest.mark.asyncio
async def test_get_snapshot_does_not_mark_inactive_nav_link():
    html = '<html><body><nav><a href="/reports">Reports</a></nav></body></html>'
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(html)

        elements, _ = await get_snapshot(page)

        await browser.close()

    assert "[already active]" not in elements[0]["label"]


# --- Perception fix (radio buttons) — บั๊กจริงที่ user รายงาน: agent มองไม่เห็นตัวเลือก
# Gender Male/Female บน OrangeHRM "My Info > Personal Details" — ยืนยันจาก DOM จริงว่า
# OrangeHRM ซ่อน native <input type="radio"> ด้วย opacity:0 แล้ววาดวงกลมที่มองเห็นแทนด้วย
# <span class="oxd-radio-input"> เป็นพี่น้องของ input ภายใน <label> เดียวกัน (เหมือน custom
# checkbox ทุกประการ แค่คนละ widget) — จำลอง DOM แบบเดียวกันตรงๆ ด้วย inline style เพราะ
# page.set_content() ไม่โหลด stylesheet จริงของเว็บ

_HTML_WITH_CUSTOM_RADIO = """
<html><body>
  <form>
    <label>
      <input type="radio" name="gender" value="1" style="opacity:0;">
      <span class="oxd-radio-input oxd-radio-input--active" style="display:inline-block;width:18px;height:18px;"></span>
      Male
    </label>
    <label>
      <input type="radio" name="gender" value="2" style="opacity:0;">
      <span class="oxd-radio-input oxd-radio-input--active" style="display:inline-block;width:18px;height:18px;"></span>
      Female
    </label>
  </form>
</body></html>
"""


@pytest.mark.asyncio
async def test_get_snapshot_sees_custom_radio_buttons_hidden_by_opacity():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(_HTML_WITH_CUSTOM_RADIO)

        elements, text_repr = await get_snapshot(page)

        await browser.close()

    labels = [e["label"] for e in elements]
    assert "Male" in labels
    assert "Female" in labels
    assert "Male" in text_repr
    assert "Female" in text_repr


@pytest.mark.asyncio
async def test_get_snapshot_still_sees_plain_unstyled_radio_buttons():
    """sanity: radio ธรรมดาที่ไม่ได้ซ่อนด้วย CSS เลย (ไม่มี custom wrapper) ต้องยังทำงาน
    เหมือนเดิมทุกประการ — fix นี้ไม่ควรกระทบเคสง่ายๆ ที่ทำงานถูกอยู่แล้ว"""
    html = """
    <html><body>
      <label><input type="radio" name="plan" value="a"> Basic</label>
      <label><input type="radio" name="plan" value="b"> Premium</label>
    </body></html>
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(html)

        elements, _ = await get_snapshot(page)

        await browser.close()

    labels = [e["label"] for e in elements]
    assert "Basic" in labels
    assert "Premium" in labels


# --- W_toggle ("Switch/Toggle Label Resolver") — บั๊กจริงที่ user รายงาน: agent เปิด/ปิด
# toggle "Include Past Employees" บน OrangeHRM Leave List ไม่ได้เลย — ยืนยันจาก DOM จริง
# (opensource-demo.orangehrmlive.com) ว่า OrangeHRM ซ่อน native <input type="checkbox">
# ด้วย opacity:0 แล้ววาด toggle ที่มองเห็นแทนด้วย <span class="oxd-switch-input"> ห่อด้วย
# <label> ที่ไม่มีข้อความเลย — ข้อความอธิบายจริงอยู่ใน sibling <p> "ก่อนหน้า" container
# ทั้งก้อน (พี่น้องของ .oxd-switch-wrapper เอง) ไม่ใช่ข้างในตัวมันเลย — จำลอง DOM แบบ
# เดียวกันตรงๆ ด้วย inline style (page.set_content() ไม่โหลด stylesheet จริง)

_HTML_WITH_CUSTOM_SWITCH = """
<html><body>
  <div class="oxd-grid-item">
    <p class="oxd-text orangehrm-leave-filter-text">Include Past Employees</p>
    <div class="oxd-switch-wrapper">
      <label>
        <input type="checkbox" style="opacity:0;">
        <span class="oxd-switch-input" style="display:inline-block;width:34px;height:18px;"></span>
      </label>
    </div>
  </div>
</body></html>
"""


@pytest.mark.asyncio
async def test_get_snapshot_sees_custom_switch_hidden_by_opacity_with_sibling_label():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(_HTML_WITH_CUSTOM_SWITCH)

        elements, text_repr = await get_snapshot(page)

        await browser.close()

    labels = [e["label"] for e in elements]
    assert "Include Past Employees" in labels
    assert "Include Past Employees" in text_repr
    # ไม่ใช่ checkbox wrapper (ไม่ควรได้ label ผิดความหมายอย่าง "Select row")
    assert "Select row" not in labels
    assert "Select All" not in labels


@pytest.mark.asyncio
async def test_get_snapshot_switch_without_findable_sibling_label_gets_empty_label_not_wrong_one():
    """ไม่มี sibling text ให้เจอเลย — ต้องคืนค่าว่าง ไม่ใช่เดา label ผิดๆ (เช่น "Select
    row" จาก checkboxWrapperLabel ที่ตั้งใจแยกออกไปแล้ว)"""
    html = """
    <html><body>
      <div class="oxd-switch-wrapper">
        <label>
          <input type="checkbox" style="opacity:0;">
          <span class="oxd-switch-input" style="display:inline-block;width:34px;height:18px;"></span>
        </label>
      </div>
    </body></html>
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(html)

        elements, _ = await get_snapshot(page)

        await browser.close()

    switch_elements = [e for e in elements if e["tag"] == "span"]
    assert len(switch_elements) == 1
    assert switch_elements[0]["label"] == ""


@pytest.mark.asyncio
async def test_get_snapshot_still_sees_plain_checkbox_wrapper_unaffected_by_switch_fix():
    """sanity: checkbox wrapper (ไม่ใช่ switch) ยังได้ "Select row" default เหมือนเดิมทุก
    ประการ — fix นี้ไม่ควรกระทบ checkboxWrapperLabel เลย"""
    html = """
    <html><body>
      <table><tbody><tr><td>
        <label>
          <input type="checkbox" style="opacity:0;">
          <span class="oxd-checkbox-input" style="display:inline-block;width:18px;height:18px;"></span>
        </label>
      </td></tr></tbody></table>
    </body></html>
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(html)

        elements, _ = await get_snapshot(page)

        await browser.close()

    labels = [e["label"] for e in elements]
    assert "Select row" in labels


# --- ACC-2 (accuracy audit follow-up) — บั๊กที่พบจาก security/accuracy audit: element ที่
# disabled ถูกกรองทิ้งจาก snapshot ไปเลยเดิม (LLM ไม่มีทางรู้ว่ามันมีอยู่ ไม่ใช่แค่กดไม่ได้)
# — ตอนนี้ยังติด index ปกติ แต่แปะ marker "[disabled]" แทน


@pytest.mark.asyncio
async def test_get_snapshot_still_includes_disabled_button_with_marker():
    html = '<html><body><button id="submit-btn" disabled>Submit</button></body></html>'
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(html)

        elements, text_repr = await get_snapshot(page)

        await browser.close()

    assert len(elements) == 1
    assert elements[0]["label"] == "Submit [disabled]"
    assert "[disabled]" in text_repr


@pytest.mark.asyncio
async def test_get_snapshot_does_not_mark_enabled_button_as_disabled():
    html = '<html><body><button id="submit-btn">Submit</button></body></html>'
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(html)

        elements, _ = await get_snapshot(page)

        await browser.close()

    assert elements[0]["label"] == "Submit"
    assert "[disabled]" not in elements[0]["label"]


@pytest.mark.asyncio
async def test_get_snapshot_marks_disabled_input_field_too():
    html = '<html><body><input type="text" placeholder="Zip Code" disabled></body></html>'
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(html)

        elements, _ = await get_snapshot(page)

        await browser.close()

    assert len(elements) == 1
    assert "[disabled]" in elements[0]["label"]
    assert "Zip Code" in elements[0]["label"]


# --- W65[1] ("Required-Field Validation") — FormFieldInfo.required ถูก crawl เก็บไว้แล้ว
# ตั้งแต่นานแล้ว (site_learning/schema.py) แต่ perception.py (เส้นทาง live DOM ที่ agent ใช้
# จริงตอนรัน task) ไม่เคยอ่าน HTML required/aria-required attribute เลย — แปะ marker
# "[required]" แบบเดียวกับ "[disabled]" ข้างบนทุกประการ


@pytest.mark.asyncio
async def test_get_snapshot_marks_required_input_field():
    html = '<html><body><input type="text" placeholder="Username" required></body></html>'
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(html)

        elements, text_repr = await get_snapshot(page)

        await browser.close()

    assert len(elements) == 1
    assert "[required]" in elements[0]["label"]
    assert "Username" in elements[0]["label"]
    assert "[required]" in text_repr


@pytest.mark.asyncio
async def test_get_snapshot_marks_aria_required_field_too():
    html = '<html><body><input type="text" placeholder="Email" aria-required="true"></body></html>'
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(html)

        elements, _ = await get_snapshot(page)

        await browser.close()

    assert "[required]" in elements[0]["label"]


@pytest.mark.asyncio
async def test_get_snapshot_does_not_mark_optional_field_as_required():
    html = '<html><body><input type="text" placeholder="Middle Name"></body></html>'
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(html)

        elements, _ = await get_snapshot(page)

        await browser.close()

    assert "[required]" not in elements[0]["label"]


@pytest.mark.asyncio
async def test_get_snapshot_marks_disabled_and_required_together():
    """marker ทั้งสองต้องแปะซ้อนกันได้ปกติ (คนละเงื่อนไข ไม่ผูกกัน — ดู comment ใน
    perception.py)"""
    html = '<html><body><input type="text" placeholder="Zip Code" required disabled></body></html>'
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(html)

        elements, _ = await get_snapshot(page)

        await browser.close()

    assert "[disabled]" in elements[0]["label"]
    assert "[required]" in elements[0]["label"]


# ---------------- W_dropdown_field_label / W_select_all_aria_grid (P0 F2 + F4) ----------------

# DOM ย่อจาก OrangeHRM 5.x Admin > User Management จริง: dropdown สองตัวติดกันที่ trigger เป็น
# <div class="oxd-select-text"> (ไม่ใช่ form field เลย) และตารางผลลัพธ์เป็น ARIA grid ล้วนๆ
# ไม่มี <table> สักตัว — สองอย่างนี้คือรากของบั๊ก P0 ทั้งคู่
_HTML_ORANGEHRM_LIKE_FILTERS = """
<html><body><main>
  <div class="oxd-input-group">
    <div class="oxd-input-group__label-wrapper"><label class="oxd-label">User Role</label></div>
    <div class="oxd-select-wrapper">
      <div class="oxd-select-text" tabindex="0"><div class="oxd-select-text-input">-- Select --</div></div>
    </div>
  </div>
  <div class="oxd-input-group">
    <div class="oxd-input-group__label-wrapper"><label class="oxd-label">Status</label></div>
    <div class="oxd-select-wrapper">
      <div class="oxd-select-text" tabindex="0"><div class="oxd-select-text-input">-- Select --</div></div>
    </div>
  </div>
</main></body></html>
"""

_HTML_ARIA_GRID_WITH_SELECT_ALL = """
<html><body><main>
  <div role="table">
    <div role="rowgroup">
      <div role="row">
        <div role="columnheader"><span class="oxd-checkbox-input" tabindex="0" style="display:inline-block;width:16px;height:16px"></span></div>
        <div role="columnheader">Username</div>
      </div>
    </div>
    <div role="rowgroup">
      <div role="row">
        <div role="cell"><span class="oxd-checkbox-input" tabindex="0" style="display:inline-block;width:16px;height:16px"></span></div>
        <div role="cell">alice</div>
      </div>
      <div role="row">
        <div role="cell"><span class="oxd-checkbox-input" tabindex="0" style="display:inline-block;width:16px;height:16px"></span></div>
        <div role="cell">bob</div>
      </div>
    </div>
  </div>
</main></body></html>
"""


async def _labels_for(html: str) -> list[str]:
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(html)
        elements, _ = await get_snapshot(page)
        await browser.close()
    return [e["label"] for e in elements]


@pytest.mark.asyncio
async def test_custom_dropdown_triggers_are_labelled_with_their_field_name():
    """W_dropdown_field_label: เดิม dropdown ทั้งสองตัวได้ label เป็น '-- Select --' เหมือนกัน
    เป๊ะ โมเดลจึงต้องเดาจากลำดับ DOM ว่าอันไหนคือ User Role — รากของบั๊ก W_filter_safety ที่เคย
    กรอง/ลบผิดกลุ่มมาแล้วจริง"""
    labels = await _labels_for(_HTML_ORANGEHRM_LIKE_FILTERS)

    assert "User Role: -- Select --" in labels
    assert "Status: -- Select --" in labels
    # ค่าที่เลือกอยู่ต้องยังอยู่ในป้าย ไม่ถูกชื่อ field ทับทิ้ง (โมเดลต้องรู้ว่ายังไม่ได้ตั้งค่า)
    assert "-- Select --" not in labels


@pytest.mark.asyncio
async def test_select_all_checkbox_is_found_on_an_aria_grid_without_any_table_tag():
    """W_select_all_aria_grid: กฎเดิมใช้ el.closest('th, thead') ซึ่งไม่มีทางตรงบน data grid ที่
    ทำด้วย div[role=...] ล้วน — checkbox หัวตารางจึงถูกตั้งชื่อ 'Select row' เหมือนทุกแถว ทำให้
    element ที่ W21 สั่งโมเดลให้ไปหา ไม่มีอยู่ใน snapshot เลยสักตัว"""
    labels = await _labels_for(_HTML_ARIA_GRID_WITH_SELECT_ALL)

    assert labels.count("Select All") == 1
    assert labels.count("Select row") == 2


# --- W_field_label_for_plain_inputs (C4): ชื่อ field ต้องอ่านได้จาก form มาตรฐานด้วย ---


@pytest.mark.asyncio
async def test_snapshot_prefixes_field_name_on_native_select_and_text_inputs():
    """prefix ชื่อ field เคยเติมให้เฉพาะ custom dropdown trigger (role=combobox /
    aria-haspopup / class select-text) — `<select>` มาตรฐานและ `<input type=text>` ไม่เข้า
    เงื่อนไขสักข้อ label จึงเป็นแค่ค่าที่เลือก/พิมพ์อยู่ ไม่มีอะไรบอกว่าเป็นช่องอะไร

    ผลที่ตามมาไม่ใช่แค่โมเดลอ่านยาก: W_filter_scope_guard อ่านชื่อ field จาก prefix นี้ มันจึง
    เงียบสนิทบนเว็บที่ใช้ form มาตรฐาน (คืน "" -> fail-open ทุกครั้ง) โดยไม่ error ไม่ log อะไร
    เลย ดูจากภายนอกเหมือน guard ทำงานปกติ ซึ่งอันตรายกว่า guard ที่พังดังๆ"""
    html = (
        "<label for='role'>User Role</label>"
        "<select id='role'><option>-- Select --</option><option selected>ESS</option></select>"
        "<label for='emp'>Employee Name</label><input id='emp' type='text' value='William'>"
        "<div>Department</div><input type='text' placeholder='Type for hints...'>"
        "<input type='checkbox' id='cb'><label for='cb'>Select row</label>"
        "<button>Search</button>"
    )

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        try:
            await page.set_content(html)
            elements, _ = await get_snapshot(page)
        finally:
            await browser.close()

    labels = [e["label"] for e in elements]
    assert any(l.startswith("User Role: ") for l in labels)          # native <select>
    assert any(l.startswith("Employee Name: ") for l in labels)      # input + <label for>
    assert any(l.startswith("Department: ") for l in labels)         # input + พี่น้องข้างหน้า
    # ชนิดที่มี label ทางของตัวเองอยู่แล้วต้องไม่โดนเติมซ้ำ
    assert "Select row" in labels
    assert "Search" in labels


# --- W_dialog_in_snapshot (P8/M2): dialog ที่เปิดค้างต้องมองเห็นได้จาก snapshot ---


@pytest.mark.asyncio
async def test_snapshot_marks_dialog_contents_and_lists_them_first():
    """บั๊กจริง live run 2026-08-28: dialog "Are you Sure?" เปิดค้างอยู่ แต่ agent ไล่คลิก
    ปุ่มที่อยู่ *หลัง* dialog จน timeout ซ้ำๆ โดยไม่เคยแตะปุ่มใน dialog เลย

    dialog ไม่ใช่ทั้ง navigation และ main ของเดิมจึงได้ region='' ไม่มี marker อะไรเลย —
    ปุ่มใน dialog ปนอยู่กลางลิสต์ร่วมกับของที่อยู่หลังมัน ซึ่งหน้าตาคลิกได้เหมือนกันทุกประการ
    (ทั้งคู่ in_viewport ด้วย การเรียงด้วย in_viewport อย่างเดียวจึงแยกไม่ออก)"""
    html = (
        "<button id='behind'>Search</button><button id='behind2'>Select All</button>"
        "<div role='dialog' style='position:fixed;inset:0;background:#fff;z-index:9'>"
        "<button>No, Cancel</button><button>Yes, Delete</button></div>"
    )

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        try:
            await page.set_content(html)
            elements, _ = await get_snapshot(page)
        finally:
            await browser.close()

    labels = [e["label"] for e in elements]
    # ของใน dialog มาก่อนเสมอ — มันคือสิ่งเดียวที่กดได้จริงตอนนี้
    assert labels[0].startswith("No, Cancel")
    assert labels[1].startswith("Yes, Delete")
    assert all("[in open dialog]" in l for l in labels[:2])
    assert all(e["region"] == "dialog" for e in elements[:2])

    # ของที่อยู่หลัง dialog ต้องยังอยู่ในลิสต์ (overlay อาจหายไปเองก่อนถึงเวลาคลิกจริง)
    # แต่ต้องติดป้ายบอกว่าถูกบังอยู่
    behind = [l for l in labels if "[in open dialog]" not in l]
    assert len(behind) == 2
    assert all("[obscured]" in l for l in behind)


@pytest.mark.asyncio
async def test_plain_html_table_header_checkbox_still_reads_as_select_all():
    """กันการ regress ของพฤติกรรมเดิม (<thead>/<th>) ตอนขยายเงื่อนไขไปรองรับ ARIA"""
    labels = await _labels_for("""
      <html><body><main><table>
        <thead><tr><th><span class="oxd-checkbox-input" tabindex="0" style="display:inline-block;width:16px;height:16px"></span></th><th>Name</th></tr></thead>
        <tbody><tr><td><span class="oxd-checkbox-input" tabindex="0" style="display:inline-block;width:16px;height:16px"></span></td><td>alice</td></tr></tbody>
      </table></main></body></html>
    """)

    assert labels.count("Select All") == 1
    assert labels.count("Select row") == 1


# ---------------- W_extract_row_cap (P4.5): ตัดขนาดผลลัพธ์ read_page_data ----------------

def test_cap_rows_leaves_small_results_untouched():
    rows = [["a"], ["b"]]
    capped, note = _cap_rows(rows, len(rows))
    assert capped == rows
    assert note == ""


def test_cap_rows_truncates_and_says_so_out_loud():
    """ห้ามตัดแบบเงียบๆ — โมเดลต้องรู้ว่ายังมีแถวที่ไม่ได้เห็น ไม่งั้นมันจะสรุปจากข้อมูลบางส่วน
    ราวกับเป็นข้อมูลทั้งหมด (failure mode เดียวกับ W_confident_zero)"""
    from backend.app.config import settings

    rows = [[str(i)] for i in range(settings.read_page_data_max_rows + 40)]
    capped, note = _cap_rows(rows, len(rows))

    assert len(capped) == settings.read_page_data_max_rows
    assert str(len(rows)) in note
    assert str(settings.read_page_data_max_rows) in note
    assert "computed by the system from ALL" in note


def test_cap_rows_can_be_disabled_with_a_non_positive_limit():
    from unittest.mock import patch

    rows = [[str(i)] for i in range(500)]
    with patch("backend.app.core.perception.settings") as mock_settings:
        mock_settings.read_page_data_max_rows = 0
        capped, note = _cap_rows(rows, len(rows))

    assert len(capped) == 500
    assert note == ""

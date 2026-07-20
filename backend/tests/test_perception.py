import pytest
from playwright.async_api import async_playwright

from backend.app.core.perception import get_snapshot

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

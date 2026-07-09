"""
perception.py  —  W2: Perception Module (หัวใจของ Agent)
------------------------------------------------------------
หน้าที่: มองหน้าเว็บขณะนั้น แล้วแปลงเป็น "indexed elements"
         ที่ประหยัด token เพื่อส่งให้ LLM ตัดสินใจ

แนวคิด: อย่าส่ง HTML ดิบทั้งหน้าให้ LLM (เปลือง token + งง)
        ให้ดึงเฉพาะ element ที่ "โต้ตอบได้" + "มองเห็น" มาติดหมายเลข
        เช่น  [0] input 'Username'
              [1] input 'Password'
              [2] button 'Login'

รันกับ saucedemo.com เพื่อทดสอบ

ติดตั้งก่อนใช้:
    pip install playwright
    playwright install chromium
"""

import asyncio
from playwright.async_api import async_playwright, Page


# --- JS ที่ inject เข้าไปเก็บ element โต้ตอบได้ที่มองเห็นบนหน้าจอ ---
_COLLECT_JS = r"""
() => {
  const selectors = [
    'a', 'button', 'input', 'select', 'textarea',
    '[role=button]', '[role=link]', '[role=checkbox]',
    '[role=tab]', '[onclick]', '[tabindex]'
  ].join(',');

  const nodes = Array.from(document.querySelectorAll(selectors));
  const out = [];
  let idx = 0;

  for (const el of nodes) {
    // เช็คว่ามองเห็นจริงไหม
    const rect = el.getBoundingClientRect();
    const st = window.getComputedStyle(el);
    const visible = rect.width > 0 && rect.height > 0 &&
                    st.visibility !== 'hidden' &&
                    st.display !== 'none' &&
                    st.opacity !== '0';
    if (!visible) continue;
    if (el.disabled) continue;

    // ติดหมายเลขไว้บน element เพื่อให้ agent สั่งกลับได้ทีหลัง
    el.setAttribute('data-ai-index', idx);

    const tag  = el.tagName.toLowerCase();
    const type = el.getAttribute('type') || '';
    // หา label ที่สื่อความหมายที่สุด
    const label = (
      el.innerText ||
      el.value ||
      el.getAttribute('placeholder') ||
      el.getAttribute('aria-label') ||
      el.getAttribute('name') ||
      ''
    ).trim().replace(/\s+/g, ' ').slice(0, 80);

    out.push({ index: idx, tag, type, label });
    idx++;
  }
  return out;
}
"""


async def get_snapshot(page: Page):
    """
    คืนค่า 2 อย่าง:
      elements  = list ของ dict (index, tag, type, label)  -> ไว้ให้โค้ดใช้
      text_repr = string สรุปสั้นๆ                          -> ไว้ยัดใส่ prompt LLM
    """
    elements = await page.evaluate(_COLLECT_JS)

    lines = []
    for e in elements:
        kind = f"{e['tag']}" + (f"({e['type']})" if e['type'] else "")
        label = f" '{e['label']}'" if e['label'] else ""
        lines.append(f"[{e['index']}] {kind}{label}")

    text_repr = "\n".join(lines)
    return elements, text_repr


# --- helper: ให้ agent สั่งงานกลับด้วย "หมายเลข" ที่ perception ให้มา ---
# ทุก action คืนค่า "[OK]" หรือ "[FAIL] เหตุผล" เสมอ ไม่ raise exception ออกไป
# เพื่อให้ agent loop (W4) จับ error แล้วตัดสินใจ retry/แจ้ง user ต่อได้ ไม่ crash ทั้ง process

ACTION_TIMEOUT_MS = 3000


async def click_by_index(page: Page, index: int) -> str:
    try:
        await page.click(f'[data-ai-index="{index}"]', timeout=ACTION_TIMEOUT_MS)
        return "[OK]"
    except Exception as e:
        return f"[FAIL] click index={index}: {type(e).__name__}"


async def fill_by_index(page: Page, index: int, text: str) -> str:
    try:
        await page.fill(f'[data-ai-index="{index}"]', text, timeout=ACTION_TIMEOUT_MS)
        return "[OK]"
    except Exception as e:
        return f"[FAIL] fill index={index}: {type(e).__name__}"


async def select_by_index(page: Page, index: int, label: str) -> str:
    """เลือกตัวเลือกใน <select> (เช่น dropdown เรียงสินค้าของ saucedemo)"""
    try:
        await page.select_option(
            f'[data-ai-index="{index}"]', label=label, timeout=ACTION_TIMEOUT_MS
        )
        return "[OK]"
    except Exception as e:
        return f"[FAIL] select index={index} label={label!r}: {type(e).__name__}"


async def scroll_by(page: Page, dy: int = 1000) -> str:
    try:
        await page.mouse.wheel(0, dy)
        return "[OK]"
    except Exception as e:
        return f"[FAIL] scroll dy={dy}: {type(e).__name__}"


# ------------------------------------------------------------
# DEMO: ลองกับ saucedemo — ดู snapshot แล้วลอง login
# ------------------------------------------------------------
async def demo():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)  # เห็นหน้าจอตอน dev
        page = await browser.new_page()
        await page.goto("https://www.saucedemo.com/")

        # 1) Perceive — ดูว่า agent "เห็น" อะไรบ้าง
        elements, text_repr = await get_snapshot(page)
        print("=== หน้า Login ที่ AI มองเห็น ===")
        print(text_repr)
        print()

        # 2) Act — จำลองว่า LLM ตัดสินใจแล้วสั่งกลับมาด้วย index
        #    (saucedemo ให้ user/pass มาตรฐานไว้ทดสอบ)
        u_idx = next(e['index'] for e in elements if 'user' in e['label'].lower())
        p_idx = next(e['index'] for e in elements if 'pass' in e['label'].lower())
        b_idx = next(e['index'] for e in elements if e['tag'] == 'input' and e['type'] == 'submit')

        print("[LOGIN]")
        print(" fill username:", await fill_by_index(page, u_idx, "standard_user"))
        print(" fill password:", await fill_by_index(page, p_idx, "secret_sauce"))
        print(" click submit :", await click_by_index(page, b_idx))
        await page.wait_for_load_state("networkidle")

        # 3) Perceive อีกครั้ง — พิสูจน์ว่า agent เรียนรู้หน้าใหม่เองได้
        elements2, text_repr2 = await get_snapshot(page)
        print("\n=== หน้า Inventory หลัง login (agent เห็นอะไรใหม่) ===")
        print(text_repr2)

        # --- Dropdown: เรียงลำดับสินค้า ---
        names_before = await page.locator(".inventory_item_name").all_inner_texts()
        sort_idx = next(e['index'] for e in elements2 if e['tag'] == 'select')

        print("\n[SORT DROPDOWN]")
        print(" ก่อนเรียง :", names_before)
        result = await select_by_index(page, sort_idx, "Price (low to high)")
        print(" select result:", result)
        await page.wait_for_timeout(300)
        names_after = await page.locator(".inventory_item_name").all_inner_texts()
        print(" หลังเรียง :", names_after)
        print(" ลำดับเปลี่ยนจริงไหม:", names_before != names_after)

        # --- Scroll ---
        print("\n[SCROLL]")
        y_before = await page.evaluate("window.scrollY")
        print(" scroll result:", await scroll_by(page, 1000))
        y_after = await page.evaluate("window.scrollY")
        print(f" scrollY: {y_before} -> {y_after} (เปลี่ยนจริง: {y_after != y_before})")

        # --- ทดสอบ error handling: ยิง index ที่ไม่มีอยู่จริง ---
        print("\n[ERROR HANDLING] ยิง action ด้วย index ผิดๆ (ไม่ควร crash)")
        print(" click index=9999 ->", await click_by_index(page, 9999))
        print(" fill  index=9999 ->", await fill_by_index(page, 9999, "x"))
        print(" select index=9999 ->", await select_by_index(page, 9999, "x"))
        print(" (โปรแกรมยังรันต่อได้ไม่ crash = error handling ทำงาน)")

        # --- หน้า cart ---
        await page.click("button:has-text('Add to cart')")   # ใส่สินค้าลงตะกร้า 1 ชิ้น
        await page.click(".shopping_cart_link")               # เปิดตะกร้า
        await page.wait_for_load_state("networkidle")
        _, cart_repr = await get_snapshot(page)
        print("\n=== หน้า Cart ที่ AI มองเห็น ===")
        print(cart_repr)

        # --- หน้า checkout (ฟอร์มกรอกข้อมูล) ---
        await page.click("#checkout")
        await page.wait_for_load_state("networkidle")
        _, checkout_repr = await get_snapshot(page)
        print("\n=== หน้า Checkout ที่ AI มองเห็น ===")
        print(checkout_repr)

        await asyncio.sleep(3)
        await browser.close()


if __name__ == "__main__":
    asyncio.run(demo())
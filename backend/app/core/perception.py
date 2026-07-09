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

async def click_by_index(page: Page, index: int):
    await page.click(f'[data-ai-index="{index}"]')


async def fill_by_index(page: Page, index: int, text: str):
    await page.fill(f'[data-ai-index="{index}"]', text)


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

        await fill_by_index(page, u_idx, "standard_user")
        await fill_by_index(page, p_idx, "secret_sauce")
        await click_by_index(page, b_idx)
        await page.wait_for_load_state("networkidle")

        # 3) Perceive อีกครั้ง — พิสูจน์ว่า agent เรียนรู้หน้าใหม่เองได้
        _, text_repr2 = await get_snapshot(page)
        print("=== หน้าถัดไปหลัง login (agent เห็นอะไรใหม่) ===")
        print(text_repr2[:1000])

        # --- หน้า cart ---
        await page.click("button:has-text('Add to cart')")   # ใส่สินค้าลงตะกร้า 1 ชิ้น
        await page.click(".shopping_cart_link")               # เปิดตะกร้า
        await page.wait_for_load_state("networkidle")
        _, cart_repr = await get_snapshot(page)
        print("=== หน้า Cart ที่ AI มองเห็น ===")
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
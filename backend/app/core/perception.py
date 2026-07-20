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

  // footer/ส่วนที่ไม่เกี่ยวกับการทำ task จริง (โซเชียล/copyright/nav ซ้ำ) —
  // กันไม่ให้กิน token เปล่าๆ ทุก step โดยที่ agent แทบไม่เคยต้องกด element พวกนี้
  //
  // *** เดิมใช้ [class*="footer" i] แบบ substring เปล่าๆ ซึ่งดันไปแมตช์ id/class
  // ของ "แถบปุ่ม action ท้าย component" ด้วย (เช่น saucedemo ใส่ปุ่ม Checkout จริง
  // ไว้ใน <div class="cart_footer">) ทำให้ปุ่มที่ต้องกดจริงหายไปจาก snapshot ทั้งที่
  // ไม่ใช่ footer ของทั้งหน้าเลย — เปลี่ยนมาเช็คแบบ token-aware แทน: ยอมให้ "footer"
  // โดดๆ หรือมีคำขอบเขตระดับทั้งหน้านำหน้า (site/page/global/main/app) เท่านั้น
  // ถึงจะถือว่าเป็น footer จริงของหน้า — ชื่อที่มีคำ component อื่นนำหน้า (cart_footer,
  // modal-footer, card-footer) จะไม่ถูกกรอง เพราะมักเป็น action bar ที่มีปุ่มสำคัญ ***
  const PAGE_SCOPE_FOOTER_WORDS = new Set(['site', 'page', 'global', 'main', 'app']);

  const isGlobalFooterToken = (raw) => {
    if (!raw) return false;
    const tokens = raw.toLowerCase().split(/[^a-z0-9]+/).filter(Boolean);
    if (!tokens.includes('footer')) return false;
    return tokens.every((t) => t === 'footer' || PAGE_SCOPE_FOOTER_WORDS.has(t));
  };

  const isIrrelevant = (node) => {
    let cur = node;
    while (cur && cur.nodeType === 1) {
      if (cur.tagName.toLowerCase() === 'footer') return true;
      if (cur.getAttribute('role') === 'contentinfo') return true;
      if (isGlobalFooterToken(cur.id)) return true;
      const classStr = cur.classList ? Array.from(cur.classList).join(' ') : '';
      if (isGlobalFooterToken(classStr)) return true;
      cur = cur.parentElement;
    }
    return false;
  };

  // element ที่เป็นแค่ตัวเลข/badge นับจำนวนล้วนๆ (เช่น <span class="cart-badge">1</span>
  // ที่ซ้อนอยู่ใน <a class="shopping_cart_link">) ไม่ควรได้ index เป็นของตัวเอง —
  // ตัวที่คลิกแล้วมีผลจริงคือ element พ่อ (a/button) ถ้าปล่อยให้ badge ได้ index
  // แยก จะได้ index ชี้ไปที่ span เล็กๆ ที่คลิกไม่โดน handler ของลิงก์จริง (เกิดได้
  // ถ้า selector อื่นในลิสต์ข้างบนไปแมตช์ badge เข้าโดยบังเอิญ เช่นมี tabindex/
  // role ติดมาด้วยเพื่อ accessibility) — ให้ขยับไปแปะ index ที่ตัวพ่อที่คลิกได้แทน
  const CLICKABLE_ANCESTOR_SELECTOR = 'a, button, [role="button"], [role="link"], [onclick]';

  const isBadgeLikeLeaf = (node) => {
    const tag = node.tagName.toLowerCase();
    if (tag === 'a' || tag === 'button') return false;
    const text = (node.innerText || '').trim();
    return text !== '' && /^\d{1,4}$/.test(text);
  };

  // เคลียร์ data-ai-index ค้างจาก get_snapshot() รอบก่อนหน้าออกก่อนเสมอ — เดิมไม่เคลียร์
  // ทำให้ element ที่เคยได้ index ไปแล้วในรอบก่อน (ยังไม่มี navigation คั่นกลาง เช่น
  // "fill" สองครั้งติดกันบนหน้าเดิม) ถูกมองว่า "แปะ index ไปแล้ว" โดย guard ด้านล่าง
  // (ไว้กันแปะซ้ำ "ภายในรอบเดียวกัน" ระหว่างเช็ค badge-ก่อนไปตัวพ่อ ไม่ได้ตั้งใจให้กัน
  // ข้ามรอบ) แล้วโดน skip ออกจาก elements list ของรอบใหม่ไปเงียบๆ ทั้งที่ element ยัง
  // อยู่จริงและมองเห็นได้อยู่ — ทำให้ snapshot รอบถัดๆ ไปบนหน้าเดิม (ไม่มี goto/reload
  // คั่น) เห็น element น้อยลงเรื่อยๆ (label หาย แม้ selector [data-ai-index="N"] เดิม
  // จะยังคลิก/กรอกได้จริงเพราะ attribute เก่ายังติดอยู่บน DOM) ปลอดภัยที่จะเคลียร์ตรงนี้
  // เพราะทุก index ที่ orchestrator.py ตัดสินใจใช้ ถูก dispatch จริงภายใน loop iteration
  // เดียวกับที่ได้ index มา เสมอ ก่อนจะเรียก get_snapshot() รอบถัดไป (ไม่มี index ค้างข้าม
  // รอบที่ยังไม่ถูกใช้)
  document.querySelectorAll('[data-ai-index]').forEach((el) => el.removeAttribute('data-ai-index'));

  const nodes = Array.from(document.querySelectorAll(selectors));
  const out = [];
  let idx = 0;

  for (const candidate of nodes) {
    const ancestor = isBadgeLikeLeaf(candidate)
      ? candidate.parentElement && candidate.parentElement.closest(CLICKABLE_ANCESTOR_SELECTOR)
      : null;
    const el = ancestor || candidate;

    // ตัวพ่ออาจถูกแปะ index ไปแล้ว (จากการวนถึงตัวพ่อเองก่อนหน้านี้ในลูป หรือจาก
    // badge อีกตัวในพ่อเดียวกัน) — ไม่ต้องแปะซ้ำ/ไม่ต้อง push entry ซ้ำ
    if (el.hasAttribute('data-ai-index')) continue;

    if (isIrrelevant(el)) continue;

    // เช็คว่ามองเห็นจริงไหม
    const rect = el.getBoundingClientRect();
    const st = window.getComputedStyle(el);
    const visible = rect.width > 0 && rect.height > 0 &&
                    st.visibility !== 'hidden' &&
                    st.display !== 'none' &&
                    st.opacity !== '0';
    if (!visible) continue;
    if (el.disabled) continue;

    // W9[A]: เช็คว่า element นี้ถูก popup/modal/overlay อื่นบังอยู่จริงไหม —
    // getBoundingClientRect()/CSS visibility ข้างบนเช็คแค่ว่า element เอง "มองเห็นได้"
    // เฉยๆ ไม่ได้เช็คว่ามี element อื่นวางทับอยู่ข้างบน (เช่น cookie-consent banner/
    // modal ที่มี z-index สูงคลุมทั้งหน้า) ทำให้ perception บอกว่า element "คลิกได้"
    // ทั้งที่คลิกจริงจะโดน overlay แทน (เจอปัญหานี้บ่อยตอน action ล้มเหลวซ้ำแม้ retry
    // ครบแล้ว ทั้งที่ index มีอยู่จริงใน DOM — ดู W9[A] vision fallback ใน llm.py/
    // orchestrator.py ที่ใช้ marker นี้เป็นสัญญาณเสริม) — ใช้
    // document.elementFromPoint() เช็คว่า element บนสุดตรงจุดกึ่งกลางจริงๆ คือตัวนี้
    // (หรือเป็นลูกของมัน) ไหม ถ้าไม่ใช่ แปะ marker ไว้ในป้าย ไม่ตัดออกจากลิสต์เพราะยัง
    // คลิกได้จริงถ้า overlay หายไปแล้วในตอนที่ action ทำงานจริง (เช่น modal ปิดไปแล้ว)
    const centerX = rect.left + rect.width / 2;
    const centerY = rect.top + rect.height / 2;
    let obscured = false;
    if (centerX >= 0 && centerX < window.innerWidth && centerY >= 0 && centerY < window.innerHeight) {
      const topEl = document.elementFromPoint(centerX, centerY);
      obscured = topEl !== null && topEl !== el && !el.contains(topEl);
    }

    // ติดหมายเลขไว้บน element เพื่อให้ agent สั่งกลับได้ทีหลัง
    el.setAttribute('data-ai-index', idx);

    const tag  = el.tagName.toLowerCase();
    const type = el.getAttribute('type') || '';

    // icon-only element (เช่น ปุ่มตะกร้าที่เป็นแค่ svg/background-image ไม่มี
    // text ข้างใน) ไม่มี innerText/aria-label ให้ใช้ -> ทำให้ LLM เห็นแค่ "[N] a"
    // เดาไม่ออกว่าคือปุ่มอะไร ทั้งที่ element ยังอยู่ในลิสต์จริง (ไม่ได้โดนกรอง)
    // แก้ด้วยการเพิ่มแหล่ง label สำรอง: title, data-test(id)/data-qa (attribute
    // มาตรฐานที่เว็บทำ QA อัตโนมัติมักใส่ไว้ เช่น saucedemo ใส่ data-test บน
    // แทบทุก element) และ id เป็นตัวสำรองสุดท้าย — แปลง kebab/snake-case เป็น
    // ช่องว่างให้อ่านง่ายขึ้น (เช่น "shopping-cart-link" -> "shopping cart link")
    const humanize = (s) => (s || '').replace(/[-_]+/g, ' ').trim();
    const dataTest = el.getAttribute('data-test') || el.getAttribute('data-testid') ||
                     el.getAttribute('data-qa') || '';
    const semantic = el.getAttribute('aria-label') || el.getAttribute('title') ||
                      humanize(dataTest) || el.getAttribute('name') ||
                      humanize(el.id) || '';

    // ปุ่มตะกร้าหลังใส่สินค้าแล้วมี badge span ลูก (เช่น "1") ทำให้ innerText
    // กลายเป็นแค่ตัวเลขล้วนๆ ซึ่งชนะ fallback ด้านบนไปเพราะไม่ใช่ค่าว่าง แต่ก็ไม่ได้
    // สื่อว่า element นี้คือปุ่มตะกร้า — ถ้า innerText เป็นแค่ตัวเลขสั้นๆ (badge
    // counter) แต่มี label เชิงความหมายให้ใช้ ให้ผสมกันแทนที่จะทิ้งไปเฉยๆ
    const trimmedText = (el.innerText || '').trim();
    const isBareCounter = /^\d{1,3}$/.test(trimmedText);

    // หา label ที่สื่อความหมายที่สุด
    let label;
    if (isBareCounter && semantic) {
      label = `${semantic} (${trimmedText})`;
    } else {
      label = (
        trimmedText ||
        el.value ||
        el.getAttribute('placeholder') ||
        semantic ||
        ''
      );
    }
    label = label.trim().replace(/\s+/g, ' ').slice(0, 80);
    if (obscured) {
      label = label ? `${label} [ถูกบังอยู่]` : '[ถูกบังอยู่]';
    }

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
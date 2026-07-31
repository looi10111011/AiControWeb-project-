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

W40: user รายงานว่า agent "มองไม่เห็นปุ่ม" บนหน้าที่มี <iframe> (เจอบน
uitestingplayground.com/frames — Outer Frame (Level 1) ซ้อน Inner Frame (Level 2) แต่ละ
ชั้นมีปุ่ม Edit/Submit/Click me/Primary ของตัวเอง) แล้วเข้าลูปกดวนกลับไปหน้า Home/Frames/
Resources ไม่รู้จบ (LLM เห็นแค่ nav link ที่มีอยู่จริงในหน้า Home เป็น element ที่กดได้เท่านั้น
ไม่มีทางสั่งกดปุ่มใน frame ที่ไม่เคยอยู่ใน snapshot เลย วนจนโดน loop-guard ของ
orchestrator.py บังคับ go_back/scroll ซ้ำๆ) — สาเหตุจริง (เหมือนบั๊กเดียวกับที่เจอใน
site_learning/extractor.py W39 มาก่อน แต่คนละ subsystem กัน): _COLLECT_JS เดิมรันผ่าน
page.evaluate() ซึ่ง execute ใน context ของ main document เท่านั้น
document.querySelectorAll() มองไม่เห็น element ภายใน <iframe> เลย (คนละ document object
กันโดยสิ้นเชิง แม้ same-origin ก็ตาม) — แก้ด้วยการเรียก _COLLECT_JS ซ้ำกับทุก frame ใน
page.frames (ดู get_snapshot() ด้านล่าง) page.frames คืนทุก frame แบบ flat อยู่แล้ว รวม
frame ที่ซ้อนกันกี่ชั้นก็ตาม ไม่ต้อง recurse เอง — เรียง main frame ก่อนเสมอ (หน้าที่ไม่มี
iframe เลย page.frames จะมีแค่ [main_frame] ตัวเดียว พฤติกรรมเดิมทุกประการ ไม่มีอะไรเปลี่ยน)
ส่ง startIndex เป็น argument ให้ _COLLECT_JS เพื่อให้เลข index เรียงต่อกันข้าม frame แบบไม่
ชนกัน (agent ใช้ index เดียวอ้างอิง element ทั้งหน้า ไม่แยกตาม frame)

แค่ perceive เห็นยังไม่พอ — click(N)/fill(N) เดิมก็ query ข้าม frame boundary ไม่ได้เหมือนกัน
(page.click(selector) หา element ใน main document เท่านั้น) เพิ่ม resolve_frame() ด้านล่าง
ให้ backend/app/core/actions.py (จุด dispatch จริงที่ orchestrator.py เรียก) ใช้หา Frame
object ที่ถูกต้องก่อนกดทุกครั้ง
"""

import asyncio
import difflib
import json
from typing import Optional, Union

from playwright.async_api import async_playwright, Frame, Page

from backend.app.config import settings


# --- JS ที่ inject เข้าไปเก็บ element โต้ตอบได้ที่มองเห็นบนหน้าจอ ---
_COLLECT_JS = r"""
(startIndex) => {
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

  // icon-only clickable elements: <span>/<div> ที่มี title/aria-label/data-test* (สื่อว่า
  // เป็น element ที่มีความหมาย ไม่ใช่แค่ container เปล่าๆ) และ cursor:pointer จริง (สื่อว่า
  // ผู้พัฒนาตั้งใจให้กดได้) แต่ไม่ตรงกับ selectors มาตรฐานด้านบนเลย (ไม่มี role="button"/
  // tabindex/onclick attribute ตาม a11y spec) — เจอบ่อยมากในเว็บที่ implement ปุ่ม icon เอง
  // ด้วย SVG/icon-font ตรงๆ แทนที่จะใช้ <button> จริง (เช่น demoqa.com/webtables คอลัมน์
  // Action: <span title="Edit"><svg>...</svg></span>, <span title="Delete">...) ทำให้
  // selectors เดิมด้านบนมองไม่เห็นปุ่มพวกนี้เลยทั้งที่กดได้จริงในเบราว์เซอร์ — ไม่ต้องแก้ label
  // logic ด้านล่างเลย (title/aria-label กลายเป็น label ผ่าน `semantic` อยู่แล้ว)
  const ICON_LABEL_SELECTOR = '[title], [aria-label], [data-test], [data-testid], [data-qa]';
  for (const cand of document.querySelectorAll(ICON_LABEL_SELECTOR)) {
    if (window.getComputedStyle(cand).cursor !== 'pointer') continue;
    // ตัวเองหรือบรรพบุรุษตรงกับ selectors มาตรฐานอยู่แล้ว (เช่น <button title="Edit">, หรือ
    // <img title="Logo"> ที่ซ้อนอยู่ใน <a>) -> pass ปกติจัดการไปแล้ว ไม่ต้องเพิ่มซ้ำ
    if (cand.closest(selectors)) continue;
    // ตัวเองห่อ element ที่ตรงกับ selectors มาตรฐานไว้ข้างใน (เช่น container กว้างๆ ที่มีปุ่ม
    // จริงซ้อนอยู่) -> ให้ปุ่มจริงข้างในได้ index ของตัวเองแทน ไม่ต้องนับ container ด้วย
    if (cand.querySelector(selectors)) continue;
    nodes.push(cand);
  }

  const out = [];
  let idx = startIndex;

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
    const hasSize = rect.width > 0 && rect.height > 0;
    const notDisplayNone = st.display !== 'none';

    // W47: ปุ่ม/ลิงก์ที่ซ่อนด้วย CSS ของตัวเอง (opacity:0 หรือ visibility:hidden) แต่
    // bounding box ไม่เป็น 0x0 จริง (แปลว่า browser ยัง layout พื้นที่ไว้ให้ — ต่างจาก
    // display:none ที่ browser ไม่ layout เลย ไม่มีทางคลิกได้จริงไม่ว่า CSS state ไหนจะ
    // เปลี่ยน ต้องกรองทิ้งเหมือนเดิม) เจอจริงบน uitestingplayground.com/scrolltoclick
    // Case 4 "Hover to Reveal": ปุ่ม flag แต่ละแถวใช้ visibility:hidden จนกว่าจะ hover
    // แถวแม่ ทำให้ perception เดิมกรองทิ้งไปเลย agent เลยไม่มีทาง index ให้กดได้ตั้งแต่ต้น
    // — ให้ยังติด index ตามปกติ แต่จำกัดเฉพาะ element ที่เป็นปุ่ม/ลิงก์เท่านั้น (ไม่ใช่ทุก
    // selector ในลิสต์บนสุด) กัน hidden <input> ที่เป็น honeypot/CSRF token จริงๆ (ซ่อนถาวร
    // ไม่ได้รอ hover) ไม่ให้ agent เผลอไปกรอก
    const hiddenByOwnStyle = st.visibility === 'hidden' || st.opacity === '0';
    const isClickableCandidate = ['a', 'button'].includes(el.tagName.toLowerCase()) ||
      el.getAttribute('role') === 'button';
    const hoverRevealCandidate = hasSize && notDisplayNone && hiddenByOwnStyle && isClickableCandidate;

    const visible = hasSize && notDisplayNone && (!hiddenByOwnStyle || hoverRevealCandidate);
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
    //
    // hoverRevealCandidate (visibility:hidden อยู่ตอนนี้) ต้องข้ามเช็คนี้ไปเลย —
    // browser ไม่ hit-test element ที่ visibility:hidden ให้ document.elementFromPoint()
    // เลย (คนละเรื่องกับ opacity:0 ที่ยัง hit-test ได้ปกติ) ทำให้ topEl กลายเป็น element
    // อื่นที่อยู่ตำแหน่งเดียวกันแทนเสมอ (เช่น <div> แม่) และ obscured จะเป็น true มั่ว
    // ทุกครั้งทั้งที่ไม่มีอะไรมาบังจริง — ไม่ใช่ overlay บัง แค่ยังไม่ hover เฉยๆ
    const centerX = rect.left + rect.width / 2;
    const centerY = rect.top + rect.height / 2;
    let obscured = false;
    if (
      !hoverRevealCandidate &&
      centerX >= 0 && centerX < window.innerWidth && centerY >= 0 && centerY < window.innerHeight
    ) {
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
    //
    // element ที่ hoverRevealCandidate (visibility:hidden อยู่ตอนนี้) — innerText คืน
    // ค่าว่างเปล่าเสมอ (browser ไม่คำนวณ "rendered text" ให้ element ที่ไม่ได้ render จริง
    // แม้จะมี text node อยู่ใน DOM จริงก็ตาม) fallback ไป textContent (ดิบกว่า ไม่สนใจว่า
    // render อยู่จริงไหม) เฉพาะกรณีนี้เท่านั้น ไม่กระทบ element ที่มองเห็นปกติเลย
    const innerTextTrimmed = (el.innerText || '').trim();
    const trimmedText = innerTextTrimmed ||
      (hoverRevealCandidate ? (el.textContent || '').trim() : '');
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
    // คนละเงื่อนไขกับ obscured ข้างบน (obscured = ถูก element อื่นวางทับ, นี่ = ซ่อนด้วย
    // CSS opacity/visibility ของตัวเอง/บรรพบุรุษ) — ทั้งสอง marker แปะซ้อนกันได้ถ้าเข้า
    // เงื่อนไขทั้งคู่พร้อมกัน (เคสหายากแต่ไม่ผิดอะไร)
    if (hoverRevealCandidate) {
      label = label ? `${label} [ซ่อนอยู่ — อาจต้อง hover แถวก่อน]` : '[ซ่อนอยู่ — อาจต้อง hover แถวก่อน]';
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

    W40: ไล่เก็บจากทุก frame ใน page.frames ไม่ใช่แค่ main document (ดู docstring หัวไฟล์) —
    main frame เก็บก่อนเสมอ (หน้าที่ไม่มี iframe เลย page.frames มีแค่ [main_frame] ตัวเดียว
    พฤติกรรม/ลำดับ index เดิมทุกประการ ไม่มีอะไรเปลี่ยน) ส่ง len(elements) ปัจจุบันเป็น
    startIndex ให้ _COLLECT_JS ของแต่ละ frame ถัดไป ให้ index เรียงต่อกันไม่ชนกันข้าม frame
    — frame ที่ evaluate ไม่ได้ (cross-origin ที่ browser บล็อก, frame ถูก detach ระหว่างอ่าน
    ฯลฯ) ข้ามไปเงียบๆ ไม่ throw ออกไป (เฟรมเดียวพังไม่ควรทำให้ perceive ทั้งหน้าล้มเหลวไปด้วย)
    """
    elements: list[dict] = []
    main_frame = page.main_frame
    frames = [main_frame] + [f for f in page.frames if f != main_frame]
    for frame in frames:
        try:
            frame_elements = await frame.evaluate(_COLLECT_JS, len(elements))
        except Exception:
            continue
        elements.extend(frame_elements)

    lines = []
    for e in elements:
        kind = f"{e['tag']}" + (f"({e['type']})" if e['type'] else "")
        label = f" '{e['label']}'" if e['label'] else ""
        lines.append(f"[{e['index']}] {kind}{label}")

    text_repr = "\n".join(lines)
    return elements, text_repr


async def resolve_frame(page: Page, selector: str) -> Union[Page, Frame]:
    """W40: หา Frame (หรือ page เอง) ที่มี element ตรง selector นี้จริง — ลอง main frame
    ก่อนเสมอ (เร็วที่สุด/ตรงกับกรณีส่วนใหญ่ที่ element ไม่ได้อยู่ใน iframe เลย) แล้วค่อยไล่
    frame อื่นถ้าไม่เจอ ใช้ query_selector() เช็คแค่ "มีอยู่ไหม" (คืนทันทีไม่รอ) แทนที่จะลอง
    click()/wait_for_selector() ทีละ frame ซึ่งจะรอจน timeout เต็มทุกครั้งที่ไม่เจอ (ช้ามาก
    ถ้าต้องไล่หลาย frame) — คืน page เฉยๆ ถ้าหาไม่เจอในทุก frame เลย ให้ caller
    (backend/app/core/actions.py) เรียก click()/fill() ปกติแล้วเจอ error message ที่คุ้นเคย
    (เช่น "หา element ไม่เจอ") แทนที่จะต้องแยก error กรณีนี้ออกมาต่างหาก"""
    try:
        if await page.query_selector(selector) is not None:
            return page
    except Exception:
        pass
    main_frame = page.main_frame
    for frame in page.frames:
        if frame == main_frame:
            continue
        try:
            if await frame.query_selector(selector) is not None:
                return frame
        except Exception:
            continue
    return page


# --- W45: Lane 1/2 อ่าน "เนื้อหา" หน้าเว็บ (นับ/ตาราง) — แยกจาก get_snapshot() ด้านบนซึ่ง
# กรองเอาเฉพาะ element ที่คลิกได้ ไม่มีเส้นทางอ่านเนื้อหา (ตาราง/ตัวเลข/text) เลย —
# ทั้งสองฟังก์ชันนี้ "ไม่" ถูกเรียกอัตโนมัติทุก step เหมือน get_snapshot() ต้องถูกเรียกผ่าน
# tool "read_page_data" เท่านั้น (ดู backend/app/core/actions.py::read_page_data ที่เป็น
# จุดตัดสินใจว่าจะเรียกตัวไหน + backend/app/core/llm.py สำหรับ tool schema/SYSTEM_PROMPT
# ที่ให้ LLM เรียกเอง) กันไม่ให้ token cost ต่อ step โตขึ้นถาวรเหมือนที่ get_snapshot() ระวัง
# ไว้อยู่แล้ว (trim system prompt/filter footer/prompt caching)

_COUNT_ELEMENTS_JS = r"""
(selector) => {
  try {
    return document.querySelectorAll(selector).length;
  } catch (e) {
    return 0;
  }
}
"""


async def count_elements(page: Page, selector_hint: str) -> int:
    """Lane 1: นับจำนวน element ที่ตรงกับ selector_hint (CSS selector) รวมทุก frame
    (เหมือน get_snapshot() — เนื้อหาที่ต้องนับอาจอยู่ใน <iframe>) ใช้ JS query ล้วนๆ
    (document.querySelectorAll(...).length) ไม่เรียก LLM เลย ไม่ส่งเนื้อหาดิบกลับมาด้วย
    คืนแค่ตัวเลข (token แทบเป็นศูนย์ เร็วกว่าให้ LLM อ่านตารางทั้งก้อนมานับเอง — ดู
    actions.py::read_page_data ที่ favor ฟังก์ชันนี้ก่อน extract_table_data() เสมอตอน
    query เป็นคำถามเชิงนับ)

    selector_hint ที่ผิดรูปแบบ (invalid CSS) นับเป็น 0 ที่ frame นั้นเงียบๆ ไม่ throw
    ออกไป เหมือน pattern ที่ get_snapshot()/resolve_frame() ใช้อยู่แล้ว (frame เดียวพัง/
    query ไม่ได้ไม่ควรทำให้ทั้งฟังก์ชันล้มเหลว)"""
    total = 0
    main_frame = page.main_frame
    frames = [main_frame] + [f for f in page.frames if f != main_frame]
    for frame in frames:
        try:
            total += await frame.evaluate(_COUNT_ELEMENTS_JS, selector_hint)
        except Exception:
            continue
    return total


_EXTRACT_TABLE_JS = r"""
(hint) => {
  const clean = (s) => (s || "").replace(/\s+/g, " ").trim();

  const extractTable = (el) => {
    const rows = Array.from(el.querySelectorAll("tr"))
      .map((tr) => Array.from(tr.querySelectorAll("th,td")).map((cell) => clean(cell.innerText)))
      .filter((row) => row.length > 0);
    return rows.length > 0 ? { kind: "table", rows } : null;
  };

  const extractList = (el) => {
    const liChildren = el.querySelectorAll(":scope > li");
    const itemNodes = liChildren.length > 0 ? liChildren : el.children;
    const items = Array.from(itemNodes).map((node) => clean(node.innerText)).filter(Boolean);
    return items.length > 0 ? { kind: "list", items } : null;
  };

  const extractFrom = (el) => {
    if (!el) return null;
    return el.tagName.toLowerCase() === "table" ? extractTable(el) : extractList(el);
  };

  const direct = extractFrom(document.querySelector(hint));
  if (direct) return direct;

  // Fallback Extraction Protocol: target_hint ที่ LLM เดามาไม่ตรง/ไม่มีข้อมูลเลย (get_snapshot()
  // กรองเฉพาะ element คลิกได้ ไม่เคยโชว์โครงสร้างตาราง/class name จริงให้ LLM เห็นเลย เดาได้
  // แค่จาก URL/context อื่น) — ก่อนจะยอม fail ให้บังคับหาตาราง <table>/list (<ul>,<ol>) จริง
  // บนหน้านี้ตรงๆ แล้วอ่าน td/th (หรือ li) ทุกอันตรงๆ ก่อน (เอาตัวที่มีแถว/รายการเยอะที่สุด
  // ถ้ามีหลายอัน) แทนที่จะตอบว่า "ไม่พบข้อมูล" ทั้งที่จริงๆ มีตาราง/list อยู่บนหน้านี้ แค่เดา
  // selector ผิดเฉยๆ
  let best = null;
  for (const t of document.querySelectorAll("table")) {
    const r = extractTable(t);
    if (r && (!best || r.rows.length > best.rows.length)) best = r;
  }
  if (best) return best;
  for (const l of document.querySelectorAll("ul, ol")) {
    const r = extractList(l);
    if (r && (!best || r.items.length > best.items.length)) best = r;
  }
  return best;
}
"""


def fuzzy_find(query: str, candidates: list[str], threshold: float = 0.75) -> Optional[str]:
    """W46: เทียบ query กับแต่ละ candidate ด้วย difflib.SequenceMatcher.ratio() (stdlib ล้วนๆ
    ไม่ต้องเพิ่ม dependency ใหม่ — candidates ในบริบทนี้คือ label/ชื่อในตาราง 1 หน้า ไม่เยอะ
    พอที่ performance ของ difflib จะเป็นปัญหาจริง) คืน candidate ที่ score สูงสุดถ้าเกิน
    threshold เท่านั้น ไม่งั้นคืน None (ไม่มีตัวไหนใกล้เคียงพอ) — เทียบแบบไม่สนตัวพิมพ์ใหญ่
    เล็ก (lowercase ทั้งสองฝั่งก่อนเทียบ)

    threshold (ดู settings.agent_fuzzy_match_threshold สำหรับค่า default ที่ปรับได้จริงใน
    ระบบ — ฟังก์ชันนี้เก็บ default ของตัวเองแยกเป็น literal ให้เรียกตรงๆ/เทสต์ได้โดยไม่ต้อง
    พึ่ง settings) ตั้งต่ำไปจะ false-positive จับคนละคน/คนละชื่อที่บังเอิญคล้ายกันเป็นตัว
    เดียวกัน (อันตรายกว่า false negative เพราะตอบข้อมูลผิดคนแบบมั่นใจ) ตั้งสูงไปจะพลาดคำที่
    พิมพ์ผิดจริงๆ"""
    best_candidate: Optional[str] = None
    best_score = 0.0
    for candidate in candidates:
        score = difflib.SequenceMatcher(None, query.lower(), candidate.lower()).ratio()
        if score > best_score:
            best_score = score
            best_candidate = candidate
    if best_candidate is not None and best_score >= threshold:
        return best_candidate
    return None


def _lookup_annotation(query: str, candidates: list[str]) -> tuple[str, bool]:
    """ใช้ตอน extract_table_data() ได้ query มาด้วย (ไม่ใช่แค่ table_hint) แปลว่าผู้เรียก
    กำลังหา "ค่าเฉพาะเจาะจง" ในตาราง/list (เช่น ชื่อคน) ไม่ใช่แค่ขอสรุปทั้งก้อนเฉยๆ — ต้อง
    ลอง exact substring match (case-insensitive ทั้ง 2 ทิศทาง) ก่อนเสมอ ถ้าเจอแล้ว (หรือ
    ไม่มี query มาตั้งแต่แรก) ไม่ต้องทำอะไรต่อ ถ้าไม่เจอเลยค่อย fuzzy_find เป็นชั้นสำรอง

    คืน (annotation_text, found_anything) — annotation_text ใช้แปะนำหน้าผลลัพธ์ตอนเจอ
    fuzzy match (ให้ LLM เห็นชัดว่าเป็นการเดา ไม่ใช่ตรงกันเป๊ะ ไม่ใช่แกล้งทำเป็นตรงกัน — ดู
    "Cierra Vaga"/"Cierra Vega" ในตัวอย่างจริงที่ user รายงาน) found_anything=False เฉพาะ
    ตอนมี query แต่ไม่เจอทั้ง exact และ fuzzy เลย — ให้ผู้เรียกตัดสินใจคืน [FAIL] แทนที่จะ
    คืนตารางทั้งก้อนที่ไม่เกี่ยวข้องกับสิ่งที่ถามจริงๆ"""
    if not query:
        return "", True
    lower_query = query.lower()
    if any(lower_query in c.lower() or c.lower() in lower_query for c in candidates):
        return "", True
    match = fuzzy_find(query, candidates, threshold=settings.agent_fuzzy_match_threshold)
    if match is None:
        return "", False
    return f"[พบ '{match}' ใกล้เคียงกับคำค้น '{query}' ที่คุณพิมพ์ — ไม่ตรงกันเป๊ะ ตรวจสอบก่อนใช้คำตอบ]\n\n", True


async def extract_table_data(page: Page, table_hint: str, query: str = "") -> str:
    """Lane 2: ดึงตาราง/list ที่ตรงกับ table_hint (CSS selector) มาแปลงเป็น markdown
    table (ถ้าเป็น <table>) หรือ JSON list กระชับ (ถ้าเป็น container อื่นที่มี item ลูก
    เช่น <ul>) — ตัด whitespace ส่วนเกิน/attribute ที่ไม่จำเป็นออกหมด (คล้ายแนวที่ทำกับ
    footer filter ใน get_snapshot() เดิม) เพื่อไม่ให้กินเนื้อที่ context เกินจำเป็น

    "ไม่" ถูกเรียกอัตโนมัติทุก step (ต่างจาก get_snapshot()) เป็น Lane 2 ที่ LLM ต้อง
    ตีความเนื้อหาที่ดึงมาเองต่อ — ถูกเรียกเฉพาะผ่าน tool "read_page_data" ตอนคำถามตอบด้วย
    การนับล้วนๆ ไม่ได้ (ดู count_elements() ด้านบนสำหรับกรณีนับ)

    query (optional): ค่าเฉพาะเจาะจงที่กำลังหา (เช่น ชื่อคน) — ว่างเปล่า (default) = แค่ขอ
    สรุปทั้งตาราง/list ไม่ต้องเช็ค lookup อะไรเลย (พฤติกรรมเดิมทุกประการ) มีค่า = เช็ค exact
    match ในแถว/รายการก่อน ไม่เจอค่อย fuzzy_find (ดู _lookup_annotation() ด้านบน) — ถ้าไม่
    เจอทั้ง exact และ fuzzy เลย คืน "[FAIL] ..." แทนที่จะคืนตารางทั้งก้อนที่ไม่ตรงกับที่ถาม

    table_hint ที่ไม่ตรงกับ element ไหนเลย (หรือตรงแต่ไม่มีข้อมูล เช่น table ว่างเปล่า) จะ
    fallback ไปหาตาราง/list จริงบนหน้านั้นเอง (เอาตัวที่มีแถว/รายการเยอะที่สุดถ้ามีหลายอัน)
    ก่อนจะยอม fail (ดู _EXTRACT_TABLE_JS) — กันกรณี LLM เดา table_hint ผิดทั้งที่มีข้อมูลอยู่
    บนหน้าจริงๆ (get_snapshot() ไม่เคยโชว์โครงสร้างตาราง/class name ให้ LLM เห็นเลย)

    ไม่พบ element/ตาราง/list ใดๆ ในทุก frame เลย (แม้ fallback แล้ว) คืนข้อความ "[FAIL] ..."
    อธิบายเหตุผล (ไม่ throw ออกไป เหมือน action อื่นๆ ในระบบ — ดู actions.py::ActionResult)"""
    main_frame = page.main_frame
    frames = [main_frame] + [f for f in page.frames if f != main_frame]
    for frame in frames:
        try:
            data = await frame.evaluate(_EXTRACT_TABLE_JS, table_hint)
        except Exception:
            continue
        if data is None:
            continue

        if data["kind"] == "table":
            rows = data["rows"]
            if not rows:
                return ""
            header, *body = rows
            candidates = [cell for row in body for cell in row]
            annotation, found = _lookup_annotation(query, candidates)
            if not found:
                return f"[FAIL] ไม่พบข้อมูลที่ตรงหรือใกล้เคียงกับ '{query}' ใน '{table_hint}'"
            lines = [
                "| " + " | ".join(header) + " |",
                "| " + " | ".join("---" for _ in header) + " |",
            ]
            lines += ["| " + " | ".join(row) + " |" for row in body]
            return annotation + "\n".join(lines)

        items = data["items"]
        annotation, found = _lookup_annotation(query, items)
        if not found:
            return f"[FAIL] ไม่พบข้อมูลที่ตรงหรือใกล้เคียงกับ '{query}' ใน '{table_hint}'"
        return annotation + json.dumps(items, ensure_ascii=False)

    return f"[FAIL] ไม่พบ element ที่ตรงกับ '{table_hint}'"


# --- helper: ให้ agent สั่งงานกลับด้วย "หมายเลข" ที่ perception ให้มา ---
# ทุก action คืนค่า "[OK]" หรือ "[FAIL] เหตุผล" เสมอ ไม่ raise exception ออกไป
# เพื่อให้ agent loop (W4) จับ error แล้วตัดสินใจ retry/แจ้ง user ต่อได้ ไม่ crash ทั้ง process

ACTION_TIMEOUT_MS = 3000


async def click_by_index(page: Page, index: int) -> str:
    try:
        selector = f'[data-ai-index="{index}"]'
        target = await resolve_frame(page, selector)
        await target.click(selector, timeout=ACTION_TIMEOUT_MS)
        return "[OK]"
    except Exception as e:
        return f"[FAIL] click index={index}: {type(e).__name__}"


async def fill_by_index(page: Page, index: int, text: str) -> str:
    try:
        selector = f'[data-ai-index="{index}"]'
        target = await resolve_frame(page, selector)
        await target.fill(selector, text, timeout=ACTION_TIMEOUT_MS)
        return "[OK]"
    except Exception as e:
        return f"[FAIL] fill index={index}: {type(e).__name__}"


async def select_by_index(page: Page, index: int, label: str) -> str:
    """เลือกตัวเลือกใน <select> (เช่น dropdown เรียงสินค้าของ saucedemo)"""
    try:
        selector = f'[data-ai-index="{index}"]'
        target = await resolve_frame(page, selector)
        await target.select_option(selector, label=label, timeout=ACTION_TIMEOUT_MS)
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
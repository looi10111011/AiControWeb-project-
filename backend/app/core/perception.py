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
from backend.app.permission.rules import install_ssrf_guard


# --- JS ที่ inject เข้าไปเก็บ element โต้ตอบได้ที่มองเห็นบนหน้าจอ ---
_COLLECT_JS = r"""
(startIndex) => {
  // W50: role=option/menuitem*/combobox — ปุ่ม/ตัวเลือกของ custom dropdown/menu widget
  // (MUI, Ant Design, React-select, Headless UI ฯลฯ) มักไม่ใช่ <select><option> จริงเลย
  // แต่ implement เป็น <div>/<li> ที่มี role ตาม ARIA listbox/menu pattern แทน — เดิม
  // selectors ด้านบนไม่มี role พวกนี้เลย ทำให้ perception มองไม่เห็นตัวเลือกข้างในหลัง
  // เปิด dropdown แม้ document.querySelectorAll() (ด้านล่าง) จะกวาดทั้ง document อยู่แล้ว
  // (ไม่ขึ้นกับว่า dropdown จะ portal ไปแปะที่ document.body หรือซ้อนอยู่ใต้ trigger ก็ตาม
  // — ปัญหาจริงคือ selector ไม่ครอบคลุม ไม่ใช่เรื่อง portal location) role=combobox คือตัว
  // trigger ของ widget แบบนี้เอง (คู่กับ [role=button] เดิมที่มีอยู่แล้ว)
  const selectors = [
    'a', 'button', 'input', 'select', 'textarea',
    '[role=button]', '[role=link]', '[role=checkbox]',
    '[role=tab]', '[role=option]', '[role=menuitem]',
    '[role=menuitemradio]', '[role=menuitemcheckbox]', '[role=combobox]',
    '[onclick]', '[tabindex]'
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

  // W19 ("Scoped Search Context"): user รายงานปัญหาบน OrangeHRM — label เดียวกัน
  // ("Search"/username field ฯลฯ) โผล่ซ้ำทั้งใน sidebar (เมนูหลัก) และ main content
  // (search filter form จริง) ทำให้ agent เดา index ผิดเลือกตัวใน sidebar ทั้งที่ตั้งใจ
  // จะกรอกฟอร์มค้นหาในเนื้อหาหลัก — เดินขึ้นจาก element หา ancestor ที่ตรงกับ
  // navigation/main content container ที่ใกล้ที่สุดก่อน (nested <nav> ใน <main> เช่น
  // breadcrumb ยังนับเป็น "navigation" ถูกต้อง เพราะเป็น container ที่ใกล้ตัว element
  // ที่สุด) คืนค่าว่างถ้าไม่ match อะไรเลย (หน้าที่ไม่ได้ใช้ semantic container ชัดเจน —
  // ไม่ใช่ error แค่ไม่มีข้อมูลให้ disambiguate เพิ่ม)
  const NAVIGATION_REGION_SELECTOR = 'aside, nav, [role="navigation"], .oxd-sidepanel';
  const MAIN_REGION_SELECTOR = 'main, [role="main"], .oxd-layout-context';

  const getRegion = (node) => {
    let cur = node;
    while (cur && cur.nodeType === 1) {
      if (cur.matches && cur.matches(NAVIGATION_REGION_SELECTOR)) return 'navigation';
      if (cur.matches && cur.matches(MAIN_REGION_SELECTOR)) return 'main';
      cur = cur.parentElement;
    }
    return '';
  };

  // W19 ("Exact Element Matching" ข้อ 2): <label for="id">/<label>...<input></label>
  // เป็นแหล่ง label ที่น่าเชื่อถือที่สุดสำหรับ form field (เช่น OrangeHRM "Employee
  // Name") แต่ของเดิมไม่เคยอ่านเลย (มีแค่ placeholder/aria-label/name/id เป็น fallback)
  // — ไม่ได้ตั้งใจเปลี่ยนสถาปัตยกรรมจาก index-based เป็น selector-based (ยังใช้
  // data-ai-index dispatch เหมือนเดิมทุกประการ) แค่ทำให้ label ที่ LLM เห็นตรงกับที่
  // มนุษย์มองเห็นจริงมากขึ้น ลดโอกาสเดา index ผิดจาก label ที่กำกวม
  const getAssociatedLabelText = (node) => {
    if (node.id) {
      try {
        const forLabel = document.querySelector(`label[for="${CSS.escape(node.id)}"]`);
        if (forLabel) {
          const t = (forLabel.innerText || forLabel.textContent || '').trim();
          if (t) return t;
        }
      } catch (e) { /* CSS.escape/querySelector ผิดพลาด (id แปลกๆ) -- ข้ามไปเงียบๆ */ }
    }
    const wrappingLabel = node.closest ? node.closest('label') : null;
    if (wrappingLabel) {
      const t = (wrappingLabel.innerText || wrappingLabel.textContent || '').trim();
      if (t) return t;
    }
    return '';
  };

  // W_toggle ("Switch/Toggle Label Resolver" — บั๊กจริงที่ user รายงาน: agent แก้ toggle
  // switch ไม่ได้เลย เช่น "Include Past Employees" บน OrangeHRM Leave List) — ยืนยันจาก
  // DOM จริง: ปุ่ม toggle พวกนี้ (<div class="oxd-switch-wrapper"><label><input
  // type=checkbox opacity:0><span class="oxd-switch-input">...</span></label></div>)
  // ไม่มีทั้ง id/label[for]/wrapping-label-ที่มีข้อความเลย (label ที่ห่อ input ว่างเปล่า
  // สนิท) — ข้อความอธิบายจริง ("Include Past Employees") อยู่ใน <p>/sibling แยกต่างหาก
  // "ก่อนหน้า" container ของ toggle ทั้งก้อน (พี่น้องของ .oxd-switch-wrapper เอง ไม่ใช่
  // บรรพบุรุษ/ลูกของ element ที่ได้ index) — เดินขึ้นไม่กี่ชั้น (พอสำหรับ pattern
  // "label -> wrapper -> grid-item" ที่พบจริง) เช็ค previous sibling ทุกตัวในแต่ละชั้น
  // หาตัวแรกที่เป็นข้อความสั้นๆ ไม่มี element โต้ตอบได้ซ้อนอยู่ข้างใน (กัน match ปุ่ม/ช่อง
  // กรอกอื่นที่บังเอิญอยู่ก่อนหน้าผิดที่) คืนค่าว่างถ้าไม่เจอเลย (ไม่ throw ไม่เดามั่ว)
  const getPrecedingSiblingLabelText = (node) => {
    let cur = node;
    for (let depth = 0; depth < 4 && cur; depth++) {
      let sib = cur.previousElementSibling;
      while (sib) {
        const t = (sib.innerText || sib.textContent || '').trim();
        if (t && t.length <= 80 && !sib.querySelector('input, button, select, textarea, a')) {
          return t;
        }
        sib = sib.previousElementSibling;
      }
      cur = cur.parentElement;
    }
    return '';
  };

  // W19 ("Pre-Execution Planner & Navigation Guard" ข้อ "Log Cleanliness"): เมนู/แท็บที่
  // ถูกเลือก/active อยู่แล้ว บาง framework จะไม่ trigger navigation/DOM change ซ้ำถ้าคลิก
  // ทับตัวเดิม (โครงสร้างหน้าเหมือนเดิมทุกอย่าง) ทำให้ wait_stable() ที่รอ networkidle/
  // การเปลี่ยนแปลงหลัง action นี้ timeout เปล่าๆ — ติด marker ให้ LLM เห็นก่อนตัดสินใจ
  // (ไม่ใช่ deterministic hard-block แบบ state_filter.py เพราะ "active อยู่แล้ว" ไม่ได้
  // แปลว่า "ห้ามคลิกเด็ดขาด" เสมอไป — user อาจตั้งใจสั่ง refresh/เปิดใหม่จริงๆ (ดู EXCEPTION
  // ใน W19.txt) เป็นการตัดสินใจเชิงเจตนาที่ควรให้ LLM เห็นสัญญาณแล้วตัดสินใจเอง ไม่ใช่โค้ด
  // เดา/บล็อกไปเลย)
  const isElementAlreadyActive = (el) => {
    const ariaCurrent = (el.getAttribute('aria-current') || '').toLowerCase();
    if (ariaCurrent === 'page' || ariaCurrent === 'true' || ariaCurrent === 'step') return true;
    if ((el.getAttribute('aria-selected') || '').toLowerCase() === 'true') return true;
    const classStr = el.classList ? Array.from(el.classList).join(' ').toLowerCase() : '';
    const tokens = classStr.split(/\s+/).filter(Boolean);
    return tokens.includes('active') || tokens.includes('selected') || tokens.includes('current');
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

  // W20 (Task10, "Element Finder/Selector Resolver" — บั๊กจริงที่ user รายงาน: agent มองหา
  // "Profile Menu/Avatar" มุมขวาบนไม่เจอ แล้วเผลอไปคลิก element ใกล้เคียงผิด เช่นปุ่ม "Help")
  // — ตรวจ DOM จริงของ opensource-demo.orangehrmlive.com พบสาเหตุตรงๆ: ตัว trigger จริงคือ
  // `<span class="oxd-userdropdown-tab">` ที่ไม่มีทั้ง role/tabindex/onclick attribute (ไม่
  // ตรงกับ selectors มาตรฐานด้านบนเลย) และไม่มีทั้ง title/aria-label/data-test* (ไม่ตรงกับ
  // ICON_LABEL_SELECTOR ด้านบนด้วย) มีแค่ cursor:pointer + class name เป็นสัญญาณเดียวว่ากดได้
  // จริง — element นี้จึง "ไม่เคยติด index เลยตั้งแต่ต้น" ไม่ใช่แค่ label ไม่ดี ทำให้ LLM ไม่มี
  // ทางเลือกอื่นนอกจากเดา index ของ element อื่นที่ใกล้เคียงในหน้า header แทน — จับด้วย
  // class-name pattern ที่เว็บ dashboard/SPA สมัยใหม่มักใช้ตั้งชื่อ (userdropdown, profile-menu,
  // account-menu, avatar) + cursor:pointer เป็นชั้นสำรองที่ 2 ต่อจาก ICON_LABEL_SELECTOR
  const PROFILE_MENU_CLASS_RE = /user-?dropdown|profile-?menu|account-?menu|avatar/i;
  // SPD-2 (speed audit): `document.querySelectorAll('[class]')` ด้านล่างเดิมกวาดทุก element
  // ที่มี class attribute ในหน้า (แทบทุก element บนเว็บ SPA จริง — หลักพันตัวได้ง่ายๆ) แล้ว
  // เรียก window.getComputedStyle() ต่อทุกตัวที่ regex match — ตัว querySelectorAll('[class]')
  // เองก็ต้องสร้าง NodeList ขนาดใหญ่ทุก step ของ loop อยู่แล้วโดยไม่จำเป็น ทั้งที่ 4 คำที่
  // PROFILE_MENU_CLASS_RE ต้องมีอย่างน้อย 1 คำ (dropdown/profile/account/avatar/menu) เป็น
  // substring เสมอ — ใช้ attribute substring selector (`*=`, native ในเบราว์เซอร์ ไม่ต้อง
  // JS iterate เอง) กรอง candidate ให้แคบลงก่อนตั้งแต่ระดับ selector เลย แล้วค่อยเช็ค regex
  // เป๊ะๆ ซ้ำในลูปเหมือนเดิม (ผลลัพธ์เป๊ะเท่าเดิมทุกประการ เพราะทุก alternative ใน regex มี
  // substring พวกนี้อย่างน้อย 1 ตัวเสมอ แค่ query เร็วขึ้นเพราะ candidate set เล็กลงมาก)
  const PROFILE_MENU_CANDIDATE_SELECTOR = [
    '[class*="dropdown" i]', '[class*="profile" i]', '[class*="account" i]',
    '[class*="avatar" i]', '[class*="menu" i]',
  ].join(',');
  // W20 (Task10, ต่อ): DOM จริงของ OrangeHRM ยืนยันว่า class pattern นี้ไม่ได้ตรงแค่ตัว
  // container เดียว — child ข้างใน (<img class="oxd-userdropdown-img">, <p class="oxd-
  // userdropdown-name">, <i class="...oxd-userdropdown-icon">) ก็ตรง regex เดียวกันด้วยตัวเอง
  // และ cursor:pointer สืบทอดมาจาก parent ด้วย ถ้าไม่กันไว้จะได้ index แยกกัน 4 อัน สำหรับพื้นที่
  // คลิกเดียวกัน (เหมือนปัญหา badge-leaf ที่ isBadgeLikeLeaf() กันไว้ด้านบน แต่ทิศทางตรงข้าม
  // — ที่นี่ต้องมองหา "บรรพบุรุษ" ที่ตรง pattern เดียวกัน ไม่ใช่ "ลูก") ข้าม candidate ที่มี
  // บรรพบุรุษตรง pattern นี้อยู่แล้วไปเลย ให้บรรพบุรุษตัวนอกสุดเป็นตัวแทนพื้นที่คลิกทั้งก้อน —
  // แต่ต้องเช็ค cursor:pointer ของบรรพบุรุษด้วยเสมอ ไม่ใช่แค่ class ตรง (ยืนยันจริงจาก DOM
  // ของ OrangeHRM: <li class="oxd-userdropdown"> ที่ห่อ span ไว้ตรง class pattern เหมือนกัน
  // แต่ตัวมันเอง cursor:auto ไม่ใช่ pointer — ไม่ใช่ element ที่กดได้จริง แค่ชื่อ class บังเอิญ
  // ตรงเฉยๆ ถ้าเช็คแค่ class จะไปกัน span ตัวจริงที่กดได้ (cursor:pointer) ไม่ให้ติด index เลย
  // ทั้งที่ตั้งใจจะแก้บั๊กนี้อยู่)
  const hasProfileMenuAncestor = (node) => {
    let cur = node.parentElement;
    while (cur) {
      const cls = typeof cur.className === 'string' ? cur.className : '';
      if (PROFILE_MENU_CLASS_RE.test(cls) && window.getComputedStyle(cur).cursor === 'pointer') return true;
      cur = cur.parentElement;
    }
    return false;
  };
  const profileMenuNodes = new Set();
  for (const cand of document.querySelectorAll(PROFILE_MENU_CANDIDATE_SELECTOR)) {
    const classStr = typeof cand.className === 'string' ? cand.className : '';
    if (!PROFILE_MENU_CLASS_RE.test(classStr)) continue;
    if (window.getComputedStyle(cand).cursor !== 'pointer') continue;
    if (hasProfileMenuAncestor(cand)) continue;
    // target = element ที่จะได้ index จริงสำหรับ candidate นี้ — อาจเป็นบรรพบุรุษที่ตรงกับ
    // selectors มาตรฐานอยู่แล้ว (เช่น cand เป็น <span> ลูกของ <button class="user-dropdown">
    // ตัว button คือสิ่งที่ได้ index จริง ไม่ใช่ span) หรือ cand เองถ้าไม่มีบรรพบุรุษแบบนั้น
    const target = cand.closest(selectors) || cand;
    if (target === cand && !nodes.includes(cand)) {
      // ยังไม่เคยถูกเพิ่มเลยทั้งจาก selectors มาตรฐานและ ICON_LABEL_SELECTOR ด้านบน (เคสจริง
      // ของ oxd-userdropdown-tab) — เพิ่มเข้า nodes เอง ยกเว้นถ้า cand ห่อ element ที่ตรงกับ
      // selectors มาตรฐานไว้ข้างในอีกที (ให้ element ข้างในนั้นได้ index+marker ของตัวเองแทน)
      if (!cand.querySelector(selectors)) nodes.push(cand);
    }
    // ไม่ว่า target จะได้ index มาจาก pass ไหน (selectors มาตรฐาน/ICON_LABEL_SELECTOR/เพิ่งเพิ่ม
    // เอง) ก็ต้องแปะ marker เสมอ — กันไม่ให้ element ที่มี title/aria-label อยู่แล้วด้วย (ผ่าน
    // ICON_LABEL_SELECTOR ไปแล้ว) เสีย marker พิเศษนี้ไปเฉยๆ เพราะไปเจอ pass อื่นก่อน
    profileMenuNodes.add(target);
  }

  // W21 ("Custom UI Checkbox"): OrangeHRM/SPA ทั่วไปมักซ่อน native <input type="checkbox">
  // จริงด้วย CSS (display:none/opacity:0) แล้วแทนที่ด้วย span/div ห่อหุ้มที่ styled เอง (เช่น
  // .oxd-checkbox-input ของ header "Select All") ไม่มี role="checkbox"/tabindex/onclick
  // attribute ติดมาด้วยเสมอไป ทำให้ทั้ง selectors มาตรฐานด้านบน ([role=checkbox] ครอบคลุมแค่
  // wrapper ที่ทำตาม ARIA จริงๆ) และ ICON_LABEL_SELECTOR ข้างบน (ต้องมี title/aria-label/
  // data-test — wrapper พวกนี้มักไม่มีเลย) มองไม่เห็นเลย — เก็บ wrapper ที่ match class/
  // โครงสร้างที่พบบ่อยของ custom checkbox โดยเฉพาะแยกต่างหาก คลุมทั้ง OrangeHRM
  // (.oxd-checkbox-input, .oxd-table-header-cell-checkbox) และ SPA ทั่วไป (label ที่ห่อ
  // input[type=checkbox] ไว้ข้างใน, thead/th ที่มี input[type=checkbox] ตรงๆ)
  const CHECKBOX_WRAPPER_SELECTOR = [
    '.oxd-checkbox-input', '.oxd-table-header-cell-checkbox',
    '[class*="checkbox-input" i]', '[class*="checkbox-wrapper" i]',
    'th label:has(input[type="checkbox"])', 'thead label:has(input[type="checkbox"])',
    'label:has(input[type="checkbox"])',
  ].join(',');
  for (const cand of document.querySelectorAll(CHECKBOX_WRAPPER_SELECTOR)) {
    if (cand.closest(selectors)) continue;
    if (cand.querySelector(selectors)) continue;
    nodes.push(cand);
  }

  // Perception fix (radio buttons — บั๊กจริงที่ user รายงาน: agent มองไม่เห็นตัวเลือก Gender
  // Male/Female บน OrangeHRM "My Info > Personal Details") — เจอสาเหตุตรงๆ จากการตรวจ DOM
  // จริง: OrangeHRM (เหมือนกับ custom checkbox ด้านบนเป๊ะ) ซ่อน native
  // <input type="radio"> จริงด้วย opacity:0 แล้ววาดวงกลมที่มองเห็นแทนด้วย
  // <span class="oxd-radio-input">...</span> เป็น "พี่น้อง" (sibling) ของ input ภายใน
  // <label> เดียวกัน (<label><input type=radio><span class="oxd-radio-input">...</span>
  // Male</label>) — input ที่ opacity:0 โดน visibility check ด้านล่างกรองทิ้งไป (เหมือนที่
  // ตั้งใจกรอง honeypot/hidden field อื่นๆ) และไม่มี selector ไหนจับ span ตัวที่มองเห็น/
  // คลิกได้จริงเลย ทำให้ตัวเลือก radio ทั้งหมดไม่เคยติด index ตั้งแต่ต้น (ไม่ใช่แค่ label
  // ไม่ดีเหมือนเคส icon-only อื่น — ไม่มี element ไหนแทนตัวเลือกนี้ใน snapshot เลยสักตัว)
  const RADIO_WRAPPER_SELECTOR = [
    '.oxd-radio-input', '[class*="radio-input" i]', '[class*="radio-wrapper" i]',
  ].join(',');
  for (const cand of document.querySelectorAll(RADIO_WRAPPER_SELECTOR)) {
    if (cand.closest(selectors)) continue;
    if (cand.querySelector(selectors)) continue;
    nodes.push(cand);
  }

  // W_toggle ("Switch/Toggle Label Resolver" — บั๊กจริงที่ user รายงาน: agent เปิด/ปิด
  // toggle "Include Past Employees" บน OrangeHRM Leave List ไม่ได้เลย) — สาเหตุตรงๆ จาก
  // การตรวจ DOM จริง: pattern เดียวกับ checkbox/radio wrapper ด้านบนเป๊ะ (native
  // <input type="checkbox"> ซ่อนด้วย opacity:0 วาด toggle ที่มองเห็นแทนด้วย
  // <span class="oxd-switch-input">) แต่ชื่อ class ไม่มีคำว่า "checkbox" เลยสักตัว
  // (CHECKBOX_WRAPPER_SELECTOR ด้านบนจับไม่ได้) — ต่างจาก checkbox/radio ตรงที่ input ถูก
  // ห่อด้วย <label> เปล่าๆ (ไม่มี text ใดๆ ข้างใน label เลย) ทำให้แม้แต่
  // 'label:has(input[type="checkbox"])' ใน CHECKBOX_WRAPPER_SELECTOR ก็ยังไม่ช่วย เพราะ
  // input ข้างในแมตช์ selectors มาตรฐาน (bare 'input') อยู่แล้ว ทำให้ label ถูก skip ที่
  // guard `cand.querySelector(selectors)` ด้านบน (คิดว่า input ข้างในจะได้ index ของ
  // ตัวเองแทน) แต่ input เองก็ถูกกรองทิ้งจาก visibility check ด้านล่างอีกที (opacity:0 +
  // ไม่เข้าเงื่อนไข exemption ไหนเลยเพราะ isCheckboxWrapperCandidate เช็คคำว่า "checkbox"
  // เท่านั้น) — สุทธิคือไม่มี element ไหนแทน toggle นี้ในหน้าเลยสักตัว
  //
  // *** ตั้งใจแยกเป็น selector/candidate ของตัวเอง ไม่รวมกับ CHECKBOX_WRAPPER_SELECTOR ***
  // เพราะ isCheckboxWrapperCandidate (ด้านล่าง) ให้ label เริ่มต้นเป็น "Select row"/
  // "Select All" ซึ่งออกแบบมาสำหรับ checkbox เลือกแถวในตารางเท่านั้น ถ้า toggle switch ที่
  // เป็นคนละความหมายกันสิ้นเชิง (เช่น "Include Past Employees") ไปติด class ที่มีคำว่า
  // checkbox ปนมาโดยบังเอิญ จะได้ label ผิดทันที — switch ใช้ isSwitchCandidate
  // แยกต่างหาก (ดูด้านล่าง) ที่ไม่มี default แบบนั้นเลย ต้องหา label จริงจาก sibling text
  // เท่านั้น (getPrecedingSiblingLabelText — ดูด้านบนสุดของไฟล์)
  const SWITCH_WRAPPER_SELECTOR = [
    '[class*="switch-input" i]', '[class*="switch-wrapper" i]', '[role="switch"]',
  ].join(',');
  for (const cand of document.querySelectorAll(SWITCH_WRAPPER_SELECTOR)) {
    if (cand.closest(selectors)) continue;
    if (cand.querySelector(selectors)) continue;
    nodes.push(cand);
  }

  // W21 ("Icon-based Action Button Resolver"): ปุ่ม action ในตาราง (View/Download/Edit/
  // Delete ฯลฯ) ที่ implement ด้วย icon font/SVG ล้วนๆ ข้างใน <button> จริง (เช่น
  // <button><i class="bi-eye-fill"></i></button> — OrangeHRM Recruitment candidate table)
  // ตัว <button> เองตรงกับ selectors มาตรฐานด้านบนอยู่แล้วเลยได้ index เสมอ แต่ไม่มี text
  // node/aria-label/title/data-test ให้ label เห็นความหมายเลย (LLM เห็นแค่ "[N] button"
  // เดาไม่ออกว่าเป็นปุ่มอะไร) — เตรียม lookup ไว้ให้ label-building logic ด้านล่าง (getRegion
  // ก่อนหน้านี้/semantic ถัดไป) ดึงความหมายจาก class ของ icon ลูกข้างในแทน ไม่ต้องเพิ่ม node
  // ใหม่เข้า nodes เลย (ปุ่มแม่มี index อยู่แล้ว) แค่เสริม fallback label เท่านั้น
  const ICON_CLASS_LABEL_RULES = [
    [/\beye\b/i, 'View Details'],
    [/\bdownload\b/i, 'Download Resume'],
    [/\b(pencil|edit)\b/i, 'Edit'],
    [/\b(trash|delete|remove)\b/i, 'Delete'],
  ];
  const getIconClassLabel = (el) => {
    const iconEl = /^(i|svg)$/i.test(el.tagName)
      ? el
      : el.querySelector('i[class], svg[class], [class*="icon" i]');
    if (!iconEl) return '';
    const cls = iconEl.getAttribute('class') || '';
    for (const [re, label] of ICON_CLASS_LABEL_RULES) {
      if (re.test(cls)) return label;
    }
    return '';
  };

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
    // W21: custom checkbox wrapper (ดู CHECKBOX_WRAPPER_SELECTOR ด้านบน) บางเว็บวาดกล่อง
    // checkbox ล้วนๆ ผ่าน CSS ::before/::after บน wrapper ที่ตัวเอง opacity:0/visibility:
    // hidden (คล้าย hover-to-reveal button ข้างบน) — ให้ผ่อนปรนเงื่อนไข visible เหมือนปุ่ม/
    // ลิงก์เช่นกัน กัน perception กรอง wrapper พวกนี้ทิ้งทั้งที่มองเห็น/คลิกได้จริงในเบราว์เซอร์
    const isCheckboxWrapperCandidate = (el.className || '').toString().toLowerCase().includes('checkbox') ||
      (el.tagName.toLowerCase() === 'label' && !!el.querySelector('input[type="checkbox"]'));
    // Perception fix (radio buttons): เหมือน isCheckboxWrapperCandidate ข้างบนเป๊ะ แค่คนละ
    // widget — เผื่อเว็บอื่นวาด radio wrapper ด้วย opacity:0/visibility:hidden เหมือนที่
    // checkbox wrapper บางเว็บทำ (ในเคสจริงของ OrangeHRM เองตัว span.oxd-radio-input
    // มองเห็นปกติอยู่แล้ว ไม่ได้ต้องพึ่ง exemption นี้ — กันไว้เผื่อเว็บอื่น)
    const isRadioWrapperCandidate = (el.className || '').toString().toLowerCase().includes('radio');
    // W_toggle: เหมือน isCheckboxWrapperCandidate/isRadioWrapperCandidate ข้างบนเป๊ะ แค่
    // คนละ widget (switch/toggle) — ตั้งใจแยก const ต่างหาก (ไม่รวมเข้า
    // isCheckboxWrapperCandidate) เพราะตัวนี้ต้อง "ไม่" trigger checkboxWrapperLabel
    // ("Select row"/"Select All") ด้านล่าง — ดู docstring ของ SWITCH_WRAPPER_SELECTOR
    const isSwitchCandidate = (el.className || '').toString().toLowerCase().includes('switch') ||
      el.getAttribute('role') === 'switch';
    const isClickableCandidate = ['a', 'button'].includes(el.tagName.toLowerCase()) ||
      el.getAttribute('role') === 'button' || isCheckboxWrapperCandidate || isRadioWrapperCandidate ||
      isSwitchCandidate;
    const hoverRevealCandidate = hasSize && notDisplayNone && hiddenByOwnStyle && isClickableCandidate;

    const visible = hasSize && notDisplayNone && (!hiddenByOwnStyle || hoverRevealCandidate);
    if (!visible) continue;
    // ACC-2 (accuracy audit follow-up): เดิม element ที่ disabled ถูกกรองทิ้งไปเลย ทำให้
    // LLM ไม่มีทางรู้ว่าปุ่ม/ช่องกรอกนี้ "มีอยู่แต่กดไม่ได้ตอนนี้" (เช่น ปุ่ม Submit ที่รอ
    // required field ให้ครบก่อน) เห็นแค่ว่าไม่มีตัวเลือกนี้ในหน้าเลย อาจไปเดากด element
    // ใกล้เคียงผิดตัวแทน หรือสรุปผิดว่าไม่มีทางทำ action นี้ได้เลยทั้งที่จริงๆ มีแค่ต้องทำ
    // อย่างอื่นให้ครบก่อน — ยังคงติด index ให้ปกติ แต่แปะ marker "[disabled]" ในป้ายแทน
    // (ดู marker pattern อื่นในไฟล์นี้ เช่น [active อยู่แล้ว]/[ถูกบังอยู่])
    const isDisabled = !!el.disabled;

    // W65[1] ("Required-Field Validation"): เดิม HTML `required`/`aria-required` attribute
    // ถูก crawl เก็บไว้ใน site_manuals (site_learning/schema.py::FormFieldInfo.required)
    // แต่ perception.py (เส้นทาง live DOM ที่ agent ใช้จริงตอนรัน task) ไม่เคยอ่านค่านี้เลย
    // ทำให้ agent ไม่รู้ว่า field ไหน "ต้องกรอกจริง" จนกว่าจะลอง submit แล้วเจอ validation
    // error ย้อนหลัง — แปะ marker "[required]" ให้ตรงๆ ตั้งแต่ perceive (เหมือน [disabled]
    // ด้านบน) ให้ agent เห็นล่วงหน้าว่าต้องมีค่าจริงก่อนกด submit (ดู SYSTEM_PROMPT W65 rule
    // ใน llm.py ที่ใช้ marker นี้ตัดสินใจว่าต้องถาม user ก่อนไหม)
    const isRequired = !!(el.required || el.getAttribute('aria-required') === 'true');

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
    // W19: <label> ที่ผูกกับ field นี้จริง (ถ้ามี) ชนะ aria-label/title/data-test/name/id
    // เสมอ เฉพาะ form field เท่านั้น (input/select/textarea/combobox) — element อื่น
    // (button/a/...) ไม่น่าจะมี <label for> ผูกอยู่จริงตามสเปก HTML อยู่แล้ว ข้ามการเช็ค
    // ไปเลยกัน query เปล่าๆ
    const isFormFieldTag = tag === 'input' || tag === 'select' || tag === 'textarea' ||
      el.getAttribute('role') === 'combobox';
    const associatedLabel = isFormFieldTag ? getAssociatedLabelText(el) : '';
    // W21 ("Icon-based Action Button Resolver"): checkbox wrapper (ดู CHECKBOX_WRAPPER_
    // SELECTOR ด้านบน) แทบไม่มี aria-label/title/data-test เลย — ให้ "Select All"/"Select
    // row" เป็น fallback เฉพาะทาง (แยกจาก header เพราะ th/thead คือ checkbox หัวตารางที่
    // เลือกทุกแถวทีเดียว ต่างจาก checkbox รายแถวที่เลือกแค่แถวเดียว) ก่อนถึง getIconClassLabel
    // (ปุ่ม view/download/edit/delete ที่เป็น icon font ล้วนๆ — ดู getIconClassLabel ด้านบน)
    const checkboxWrapperLabel = isCheckboxWrapperCandidate
      ? (el.closest('th, thead') ? 'Select All' : 'Select row')
      : '';
    // Perception fix (radio buttons): ต่างจาก checkbox wrapper ข้างบน — radio wrapper
    // (span.oxd-radio-input) เอง innerText ว่างเปล่าเสมอ (แค่วงกลม CSS ล้วนๆ ไม่มีตัวอักษร)
    // แต่ข้อความที่บอกว่าตัวเลือกนี้คืออะไร ("Male"/"Female") เป็น text node พี่น้อง
    // (sibling) ของมันอยู่ใน <label> เดียวกัน (<label><input><span>...</span>Male</label>)
    // — ดึงจาก closest('label').innerText ซึ่งรวม text ของ sibling ทั้งหมดใน label นั้นมา
    // ด้วยเสมอ (ต่างจาก getAssociatedLabelText() ด้านบนที่ใช้กับ form field tag เท่านั้น —
    // span ไม่ใช่ form field tag เลยไม่เข้า path นั้น)
    const radioWrapperLabel = isRadioWrapperCandidate
      ? ((el.closest('label') && (el.closest('label').innerText || '').trim()) || '')
      : '';
    // W_toggle: switch (span.oxd-switch-input เป็นต้น) เอง innerText ว่างเปล่าเสมอเหมือน
    // radio wrapper ข้างบน แต่ต่างตรงที่ label ที่บอกความหมาย ("Include Past Employees")
    // ไม่ได้อยู่ใน <label> เดียวกันเลย (label ที่ห่อ switch ว่างเปล่าสนิท) ต้องเดินหา
    // sibling ก่อนหน้าแทน (getPrecedingSiblingLabelText — ดูด้านบนสุดของไฟล์) — ตั้งใจไม่มี
    // default แบบ checkboxWrapperLabel เลย (ไม่มีความหมายทั่วไปแบบ "Select row" ให้ switch
    // เดา ถ้าหา sibling text ไม่เจอจริงๆ ต้องคืนค่าว่างไปเลย ให้ fallback อื่น (icon
    // class ฯลฯ) ลองต่อแทนที่จะโชว์ label ผิดความหมาย)
    const switchLabel = isSwitchCandidate ? getPrecedingSiblingLabelText(el) : '';
    const semantic = el.getAttribute('aria-label') || el.getAttribute('title') ||
                      humanize(dataTest) || el.getAttribute('name') ||
                      humanize(el.id) || checkboxWrapperLabel || radioWrapperLabel ||
                      switchLabel || getIconClassLabel(el) || '';

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
      // W19: associatedLabel (<label for>/wrapping <label>, ดูด้านบนสุดของไฟล์) แทรก
      // ระหว่าง "ค่าที่กรอกอยู่จริง" (trimmedText/value — ยังต้องชนะเสมอถ้ามีค่าจริงอยู่
      // แล้ว เช่น select ที่เลือกตัวเลือกไว้แล้ว/ช่องค้นหาที่พิมพ์คำไปแล้ว) กับ
      // "placeholder ทั่วไป" (แพ้ label จริงเสมอถ้ามี — "Employee Name" สื่อความหมายกว่า
      // "Type for hints..." เยอะ)
      //
      // Perception fix (radio buttons, ต่อ): ข้อยกเว้นสำหรับ type="radio"/"checkbox"
      // โดยเฉพาะ — value ของ toggle input พวกนี้เป็นแค่ internal identifier ฝั่ง server
      // (เช่น "1"/"2" ของ Gender) ไม่ใช่ข้อความที่ตั้งใจให้มนุษย์อ่าน ต่างจาก text/search
      // input ที่ value คือ "สิ่งที่พิมพ์ไปจริง" ซึ่งสื่อความหมายเสมอ — ให้ associatedLabel
      // ("Male"/"Female" จาก <label> ที่ห่อ) ชนะ value ก่อน แล้วค่อย fallback ไป value ถ้า
      // ไม่มี label จริงๆ (ดีกว่าไม่มี label อะไรเลย)
      const isToggleInputType = type === 'radio' || type === 'checkbox';
      label = (
        trimmedText ||
        (isToggleInputType ? (associatedLabel || el.value) : (el.value || associatedLabel)) ||
        el.getAttribute('placeholder') ||
        semantic ||
        ''
      );
    }
    label = label.trim().replace(/\s+/g, ' ').slice(0, 80);
    // W20 (Task10): แปะ marker ที่ชัดเจนไม่กำกวมให้ element ที่จับได้จาก
    // PROFILE_MENU_CLASS_RE ด้านบน — ให้ LLM มั่นใจได้ 100% ว่านี่คือ target ที่ SYSTEM_PROMPT
    // สั่งให้หา (ดู "Account Security & Password Actions" ข้อ mandatory protocol) ไม่ต้องเดา
    // จาก username text เฉยๆ (ซึ่งเปลี่ยนไปตาม user ที่ login อยู่ ไม่ใช่ label คงที่)
    if (profileMenuNodes.has(el)) {
      label = label ? `${label} [เมนูโปรไฟล์/บัญชีผู้ใช้ — User Profile Menu]` : '[เมนูโปรไฟล์/บัญชีผู้ใช้ — User Profile Menu]';
    }
    if (obscured) {
      label = label ? `${label} [ถูกบังอยู่]` : '[ถูกบังอยู่]';
    }
    // คนละเงื่อนไขกับ obscured ข้างบน (obscured = ถูก element อื่นวางทับ, นี่ = ซ่อนด้วย
    // CSS opacity/visibility ของตัวเอง/บรรพบุรุษ) — ทั้งสอง marker แปะซ้อนกันได้ถ้าเข้า
    // เงื่อนไขทั้งคู่พร้อมกัน (เคสหายากแต่ไม่ผิดอะไร)
    if (hoverRevealCandidate) {
      label = label ? `${label} [ซ่อนอยู่ — อาจต้อง hover แถวก่อน]` : '[ซ่อนอยู่ — อาจต้อง hover แถวก่อน]';
    }
    // ACC-2: คนละเงื่อนไขกับ marker อื่นข้างบนทั้งหมด (obscured/hoverReveal คือเรื่อง
    // "มองเห็นไหม", นี่คือ "กดได้ไหม" — element ที่มองเห็นชัดเจนแต่ disabled ก็ต้องแปะ
    // marker นี้ได้ปกติ) แปะซ้อนกับ marker อื่นได้ถ้าเข้าเงื่อนไขพร้อมกัน
    if (isDisabled) {
      label = label ? `${label} [disabled]` : '[disabled]';
    }
    // W65[1]: เงื่อนไขอิสระจาก disabled เหมือนกัน (field ที่ required อาจ enable/disable
    // อยู่ก็ได้ ไม่เกี่ยวกัน) แปะซ้อนกับ marker อื่นได้ปกติ
    if (isRequired) {
      label = label ? `${label} [required]` : '[required]';
    }

    // W50 (viewport-aware sorting): เช็คว่า element นี้อยู่ในกรอบจอที่มองเห็นตอนนี้ไหม
    // (ไม่ใช่ obscured เช็คด้านบนที่ดูว่าโดน element อื่นบังอยู่หรือเปล่า — ตัวนี้ดูแค่
    // ตำแหน่ง top/bottom เทียบกับ viewport เฉยๆ) ใช้ rect ที่คำนวณไปแล้วด้านบนสุดของ
    // element นี้ ไม่ query DOM เพิ่ม — get_snapshot() (ฝั่ง Python) ใช้ค่านี้ "จัดลำดับ"
    // การแสดงผลใน text_repr เท่านั้น (element ที่เห็นอยู่ในจอตอนนี้ขึ้นก่อน) ไม่ใช่กรอง
    // element ที่ยังไม่ scroll ถึงทิ้ง — element นอกจอยังคงอยู่ครบใน list เหมือนเดิมทุก
    // ตัว แค่เรียงลำดับต่างไป (data-ai-index ที่แปะไปแล้วด้านบนไม่เปลี่ยนตามการเรียงนี้)
    const inViewport = rect.bottom > 0 && rect.top < window.innerHeight;

    // W19 ("Scoped Search Context"): "main"/"navigation"/"" (ดู getRegion ด้านบน) — ใช้
    // แค่ "จัดหมวด" ให้ LLM เห็นใน text_repr เท่านั้น (ดู get_snapshot() ฝั่ง Python) ไม่ใช่
    // ตัวกรอง element ทิ้งเหมือน in_viewport เดิม — element นอก main/navigation (region
    // ว่างเปล่า) ยังคงอยู่ครบใน list เหมือนเดิมทุกตัว
    const region = getRegion(el);

    // W19 ("Log Cleanliness"): จำกัดแค่ element ที่เป็นเมนู/แท็บจริงๆ (อยู่ใน navigation
    // region หรือมี role=tab เจาะจง) เท่านั้นถึงเช็ค isElementAlreadyActive — ตั้งใจไม่เช็ค
    // กับทุก element ที่มี class "active" เพราะคำนี้ใช้กว้างมากในเว็บจริง (เช่น ตัวเลือกที่
    // ถูก highlight ด้วยคีย์บอร์ดใน custom dropdown/autocomplete ก็มักได้ class "active"
    // เหมือนกัน แต่เป็น element ที่ "ควร" คลิกเพื่อเลือก ไม่ใช่ตัวที่ควรข้าม — ดู W19
    // "Exact Element Matching" guidance เรื่อง autocomplete ด้วย)
    const isNavCandidate = region === 'navigation' || el.getAttribute('role') === 'tab';
    const alreadyActive = isNavCandidate && isElementAlreadyActive(el);
    if (alreadyActive) {
      label = label ? `${label} [active อยู่แล้ว]` : '[active อยู่แล้ว]';
    }

    out.push({ index: idx, tag, type, label, in_viewport: inViewport, region });
    idx++;
  }
  return out;
}
"""


async def get_snapshot(page: Page):
    """
    คืนค่า 2 อย่าง:
      elements  = list ของ dict (index, tag, type, label, in_viewport, region) -> ไว้ให้
                  โค้ดใช้ (in_viewport ใช้แค่จัดลำดับการแสดงผล ดู W50 ด้านล่าง — ไม่ใช่ตัว
                  กรอง, region คือ "main"/"navigation"/"" ดู getRegion ใน _COLLECT_JS — W19
                  "Scoped Search Context" ใช้ disambiguate label ซ้ำระหว่าง sidebar/menu
                  กับ main content)
      text_repr = string สรุปสั้นๆ (เรียงตาม in_viewport ก่อนแล้ว) -> ไว้ยัดใส่ prompt LLM

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

    # W50 (viewport-aware sorting, soft — ไม่ตัด element ทิ้ง): เรียง element ที่อยู่ใน
    # กรอบจอที่มองเห็นตอนนี้ขึ้นก่อน element ที่อยู่นอกจอ (ยังต้อง scroll ถึง) — ใช้
    # list.sort() ซึ่งเป็น stable sort ของ Python เสมอ (คงลำดับเดิมของกลุ่มที่ key เท่ากัน
    # ไว้) เลยไม่กระทบลำดับเดิมภายในกลุ่ม in_viewport เดียวกัน — data-ai-index ที่แปะไปแล้ว
    # ใน _COLLECT_JS (ผูกกับ element ตัวจริงผ่าน selector ไม่ใช่ผ่านตำแหน่งใน list) ไม่ถูก
    # แตะเลย การเรียงนี้มีผลแค่ลำดับการแสดงผลใน text_repr ให้ LLM เห็น element ที่กำลังเปิด
    # อยู่ตรงหน้า (เช่น dropdown ที่เพิ่ง click เปิด) ก่อน element ที่ต้อง scroll ไปหา —
    # element นอกจอยังอยู่ครบใน list เหมือนเดิมทุกตัว ไม่มีตัวไหนถูกกรองทิ้ง
    #
    # หมายเหตุ: สำหรับ element ใน <iframe> ค่า in_viewport อ้างอิงตำแหน่ง scroll ของ frame
    # นั้นเอง ไม่ใช่ของหน้าหลัก — เป็นข้อจำกัดที่ยอมรับได้ (ยังสื่อความหมายอยู่ ไม่ใช่ bug)
    elements.sort(key=lambda e: not e.get("in_viewport", True))

    lines = []
    for e in elements:
        kind = f"{e['tag']}" + (f"({e['type']})" if e['type'] else "")
        label = f" '{e['label']}'" if e['label'] else ""
        # W19 ("Scoped Search Context"): แปะ "(navigation)" เฉพาะ element ที่อยู่ใน
        # nav/aside/sidepanel เท่านั้น (ไม่แปะ "(main)" ให้ทุกบรรทัดเปล่าๆ เพราะเป็น
        # ส่วนใหญ่ของหน้าอยู่แล้ว — แปะเฉพาะกรณีที่ต้อง disambiguate จริงถึงจะมีประโยชน์
        # เหมือน marker อื่นในไฟล์นี้ เช่น [ถูกบังอยู่]/[ซ่อนอยู่])
        region_marker = " (navigation)" if e.get("region") == "navigation" else ""
        lines.append(f"[{e['index']}] {kind}{label}{region_marker}")

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

# W63[3.3] ("Accurate Record Counting" — ticket Issue 3.3, บั๊กจริง: querySelectorAll(...).length
# เดิมนับ node ที่ match selector "ทุกตัวใน DOM" ตรงๆ ไม่สนว่ามองเห็นได้จริงไหม — เว็บจำนวนมาก
# (รวม OrangeHRM) ทิ้ง <option>/แถวตารางของหน้าอื่นที่ยัง paginate ไม่มาถึง/dropdown ที่ยังไม่เปิด
# ไว้ใน DOM เดิมด้วย display:none/ไม่ได้ layout เลย ทำให้ "จำนวนที่นับได้" สูงกว่าจำนวนแถว/รายการ
# ที่ user เห็นจริงบนจอเสมอถ้า selector ที่ LLM เดามากว้างเกินไปโดยไม่ตั้งใจไปโดน node ที่ซ่อนอยู่
# — กรองเฉพาะ element ที่ "เห็นได้จริง" ก่อนนับ (มี layout size + display ไม่ใช่ none +
# visibility ไม่ใช่ hidden) เกณฑ์เดียวกับที่ get_snapshot() ใช้ตัดสินว่า element ไหน "เห็นได้"
# (ดู hasSize/notDisplayNone ด้านบนของไฟล์นี้) ยกเว้นไม่ต้องรองรับ hover-to-reveal
# (opacity:0/visibility:hidden ที่ยัง "นับได้" สำหรับปุ่มคลิก — W47) เพราะจุดประสงค์ต่างกัน: ที่
# นี่นับ "รายการที่มีอยู่จริงตามที่ user มองเห็น" ไม่ใช่หา element ที่คลิกได้แม้จะซ่อนชั่วคราว
_COUNT_ELEMENTS_JS = r"""
(selector) => {
  try {
    let count = 0;
    for (const el of document.querySelectorAll(selector)) {
      const rect = el.getBoundingClientRect();
      if (rect.width <= 0 || rect.height <= 0) continue;
      const st = window.getComputedStyle(el);
      if (st.display === 'none' || st.visibility === 'hidden') continue;
      count++;
    }
    return count;
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

  // W_ariagrid: หลายเว็บ (OrangeHRM, MUI DataGrid, AG Grid, React-select ฯลฯ) ไม่ใช้
  // <table><tr><td> จริงเลย แต่ implement เป็น <div role="table">/<div role="row">/
  // <div role="cell|gridcell|columnheader"> แทน (ARIA grid pattern) — querySelectorAll("tr")
  // บน element พวกนี้จะได้ [] เสมอทั้งที่มีข้อมูลเต็มหน้า ต้อง fallback ไปหา [role=row]
  // ก่อนยอมแพ้ (เหมือน role=option/menuitem ที่ต้องเพิ่มให้ _COLLECT_JS ตอนแก้ปัญหา custom
  // dropdown มองไม่เห็นมาก่อนหน้านี้ — อาการเดียวกัน คนละจุด)
  const extractTable = (el) => {
    let rowEls = Array.from(el.querySelectorAll("tr"));
    let cellSelector = "th,td";
    if (rowEls.length === 0) {
      rowEls = Array.from(el.querySelectorAll('[role="row"]'));
      cellSelector = '[role="cell"],[role="gridcell"],[role="columnheader"]';
    }
    const rows = rowEls
      .map((tr) => Array.from(tr.querySelectorAll(cellSelector)).map((cell) => clean(cell.innerText)))
      .filter((row) => row.length > 0);
    return rows.length > 0 ? { kind: "table", rows } : null;
  };

  const extractList = (el) => {
    const liChildren = el.querySelectorAll(":scope > li");
    const itemNodes = liChildren.length > 0 ? liChildren : el.children;
    const items = Array.from(itemNodes).map((node) => clean(node.innerText)).filter(Boolean);
    return items.length > 0 ? { kind: "list", items } : null;
  };

  const isTableLike = (el) => {
    const role = el.getAttribute && el.getAttribute("role");
    return el.tagName.toLowerCase() === "table" || role === "table" || role === "grid";
  };

  const extractFrom = (el) => {
    if (!el) return null;
    return isTableLike(el) ? extractTable(el) : extractList(el);
  };

  const direct = extractFrom(document.querySelector(hint));
  if (direct) return direct;

  // Fallback Extraction Protocol: target_hint ที่ LLM เดามาไม่ตรง/ไม่มีข้อมูลเลย (get_snapshot()
  // กรองเฉพาะ element คลิกได้ ไม่เคยโชว์โครงสร้างตาราง/class name จริงให้ LLM เห็นเลย เดาได้
  // แค่จาก URL/context อื่น) — ก่อนจะยอม fail ให้บังคับหาตาราง <table>/[role=table|grid]/
  // list (<ul>,<ol>) จริงบนหน้านี้ตรงๆ แล้วอ่าน td/th (หรือ li) ทุกอันตรงๆ ก่อน (เอาตัวที่มี
  // แถว/รายการเยอะที่สุดถ้ามีหลายอัน) แทนที่จะตอบว่า "ไม่พบข้อมูล" ทั้งที่จริงๆ มีตาราง/list
  // อยู่บนหน้านี้ แค่เดา selector ผิดเฉยๆ
  let best = null;
  for (const t of document.querySelectorAll('table, [role="table"], [role="grid"]')) {
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


# W64[7.2]: เวลารอก่อนลอง extract_table_data() lookup ซ้ำรอบเดียวตอนไม่เจอ query ในรอบแรก —
# สั้นพอที่จะไม่ทำให้ query ที่หาไม่เจอจริงๆ (ไม่ใช่แค่ AJAX ยังโหลดไม่เสร็จ) ช้าเกินจำเป็น แต่
# นานพอให้ table reload ทั่วไปที่เจอจริง (OrangeHRM ฯลฯ) เสร็จทัน
_LOOKUP_RETRY_WAIT_SEC = 2.0


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
    อธิบายเหตุผล (ไม่ throw ออกไป เหมือน action อื่นๆ ในระบบ — ดู actions.py::ActionResult)

    W64[7.2] ("Add-Action Idempotency Lock" — ticket Issue 7.2, บั๊กจริง: agent ค้นหาแถวที่
    เพิ่งบันทึกไปทันทีหลัง Save โดยไม่รอ AJAX table reload ให้เสร็จก่อน อ่านได้ตารางเก่า/ว่าง
    เปล่า แล้วเข้าใจผิดว่าบันทึกไม่สำเร็จ — เมื่อมี query (กำลังหาค่าเฉพาะเจาะจง) และรอบแรกหา
    ไม่เจอเลย ให้รอสั้นๆ (_LOOKUP_RETRY_WAIT_SEC) แล้วลองสแกนใหม่อีกครั้งก่อนยอม [FAIL] จริง —
    ไม่กระทบ happy path เลย (เพิ่ม latency เฉพาะตอนหาไม่เจอรอบแรกเท่านั้น) และไม่กระทบ query
    ว่างเปล่า (ขอสรุปทั้งตาราง ไม่ใช่ lookup ค่าเฉพาะ)"""
    result = await _extract_table_data_once(page, table_hint, query)
    if query and result.startswith("[FAIL]"):
        await asyncio.sleep(_LOOKUP_RETRY_WAIT_SEC)
        result = await _extract_table_data_once(page, table_hint, query)
    return result


async def _extract_table_data_once(page: Page, table_hint: str, query: str) -> str:
    """W64[7.2]: สแกนจริง 1 รอบ — แยกออกมาจาก extract_table_data() เพื่อให้เรียกซ้ำได้หลัง
    รอ AJAX reload (ดู docstring ของ extract_table_data ด้านบน) โดยไม่ต้องเขียน loop scan
    ซ้ำสองที่"""
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
        await install_ssrf_guard(page)
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
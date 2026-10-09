"""
perception.py — W2: Perception Module
-------------------------------------
แปลงหน้าเว็บเป็น "indexed elements" ประหยัด token ให้ LLM ตัดสินใจ แทนการส่ง HTML ดิบ:
    [0] input 'Username'
    [1] input 'Password'
    [2] button 'Login'

W40 (iframe): page.evaluate() เห็นแค่ main document — element ใน <iframe> จึงหายจาก snapshot
(เจอบน uitestingplayground.com/frames: agent วนกด nav link จนโดน loop-guard) แก้โดยรัน
_COLLECT_JS ทุก frame ใน page.frames (flat อยู่แล้ว, main frame ก่อนเสมอ) ส่ง startIndex ให้
index ต่อกันไม่ชนข้าม frame และ click/fill ต้องหา frame ที่ถูกก่อนผ่าน resolve_frame()
(ใช้โดย actions.py)

แผนที่โซน (ค้นหา "# โซน N" ในไฟล์):
  โซน 1  _COLLECT_JS — JS ใน browser เก็บ element + สร้าง label/marker
  โซน 2  LABEL_MARKERS / label_without_markers / แยก label ซ้ำ
  โซน 3  get_snapshot() (ทางเข้าหลัก) + resolve_frame()
  โซน 4  Lane 1: count_elements() — นับ
  โซน 5  Lane 2: extract_table_data() — ดึงตาราง/list + lookup
  โซน 6  helper + demo()
"""

import asyncio
import difflib
import json
from typing import Optional, Union

from playwright.async_api import async_playwright, Frame, Page

from backend.app.config import settings
from backend.app.permission.rules import install_ssrf_guard


# ══════════════════════════════════════════════════════════════════════
# โซน 1: เก็บ element จากหน้าเว็บ (JS ที่รันใน browser)
#   ทำอะไร: หา element ที่โต้ตอบได้และมองเห็น แปะ data-ai-index แล้วสร้าง label ให้ LLM อ่าน
#   ทำงานยังไง: selector มาตรฐาน + pass เสริม (icon/profile menu/checkbox/radio/switch) -> กรอง footer/การมองเห็น -> สร้าง label + marker สถานะ -> คืน list ของ dict
# ══════════════════════════════════════════════════════════════════════
_COLLECT_JS = r"""
(startIndex) => {
  // W50: role=option/menuitem*/combobox — custom dropdown/menu (MUI, AntD, React-select ฯลฯ)
  // ไม่ใช่ <select> จริง ถ้าไม่มี role พวกนี้จะมองไม่เห็นตัวเลือกหลังเปิด dropdown
  // W_contenteditable (P3.5): rich-text editor (Gmail, Notion, Quill ฯลฯ) เป็น div[contenteditable]
  // ARIA role เพิ่มเติม (radio/slider/spinbutton/...) = role มาตรฐานของ widget library
  const selectors = [
    'a', 'button', 'input', 'select', 'textarea',
    '[contenteditable=""]', '[contenteditable="true"]', '[contenteditable="plaintext-only"]',
    '[role=button]', '[role=link]', '[role=checkbox]',
    '[role=tab]', '[role=option]', '[role=menuitem]',
    '[role=menuitemradio]', '[role=menuitemcheckbox]', '[role=combobox]',
    '[role=radio]', '[role=slider]', '[role=spinbutton]', '[role=searchbox]',
    // W_listbox_container: ห้ามเพิ่ม '[role=listbox]' — container ได้ label เป็น text ของทุก option
    // ต่อกัน โมเดลคลิกแล้วโดน option แรก (live run 2026-08-27: ขอ ESS ได้ Admin)
    '[role=textbox]', '[role=treeitem]', '[role=switch]',
    '[onclick]', '[tabindex]'
  ].join(',');

  // W_shadow_dom (P3.4): querySelectorAll ไม่ทะลุ shadow root — เว็บ web component (Salesforce,
  // Vaadin, YouTube) มองไม่เห็นทั้งหน้า เดินเฉพาะ open root (closed เข้าไม่ได้โดยการออกแบบ)
  // จำกัดความลึกกันค้าง, try/catch กัน tree แปลกๆ ทำทั้ง pass ล้ม
  // Playwright CSS selector ทะลุ open shadow root เอง จึงไม่ต้องแก้ actions.py
  const SHADOW_MAX_DEPTH = 8;
  const deepQueryAll = (root, sel) => {
    const out = [];
    const visit = (node, depth) => {
      try {
        for (const el of node.querySelectorAll(sel)) out.push(el);
      } catch (e) { /* selector ใช้กับ tree นี้ไม่ได้ — ข้ามไป ไม่ทำให้ทั้ง pass ล้ม */ }
      if (depth >= SHADOW_MAX_DEPTH) return;
      let hosts = [];
      try { hosts = node.querySelectorAll('*'); } catch (e) { return; }
      for (const host of hosts) {
        if (host.shadowRoot) visit(host.shadowRoot, depth + 1);
      }
    };
    visit(root, 0);
    return out;
  };

  // กรอง footer ระดับหน้า (โซเชียล/copyright) กันเปลือง token
  // เช็คแบบ token-aware: substring "footer" เดิมไปโดน action bar เช่น saucedemo "cart_footer"
  // (ปุ่ม Checkout หาย) — นับเฉพาะ "footer" โดดๆ หรือนำหน้าด้วย site/page/global/main/app
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

  // W19 (Scoped Search Context): label ซ้ำใน sidebar กับ main (OrangeHRM) ทำให้เลือกผิดตัว
  // หา container ที่ใกล้ที่สุด คืน '' ถ้าหน้าไม่มี semantic container
  const NAVIGATION_REGION_SELECTOR = 'aside, nav, [role="navigation"], .oxd-sidepanel';
  const MAIN_REGION_SELECTOR = 'main, [role="main"], .oxd-layout-context';
  // W_dialog_in_snapshot: ปุ่มใน dialog เคยปนกับแถวข้อมูลข้างหลัง (live run 2026-08-28: agent
  // คลิกของหลัง dialog จน timeout) — ชุดเดียวกับ actions.py::_DIALOG_CONTAINER_SELECTORS โดยเจตนา
  const DIALOG_REGION_SELECTOR = 'dialog[open], [role="dialog"], [role="alertdialog"], ' +
    '[aria-modal="true"], .modal.show, .MuiDialog-root, .ant-modal-wrap, .swal2-container, ' +
    '.oxd-dialog-container';

  const getRegion = (node) => {
    let cur = node;
    while (cur && cur.nodeType === 1) {
      // dialog ก่อนเสมอ: dialog ใน <main> ต้องนับเป็น dialog
      if (cur.matches && cur.matches(DIALOG_REGION_SELECTOR)) return 'dialog';
      if (cur.matches && cur.matches(NAVIGATION_REGION_SELECTOR)) return 'navigation';
      if (cur.matches && cur.matches(MAIN_REGION_SELECTOR)) return 'main';
      cur = cur.parentElement;
    }
    return '';
  };

  // W19: <label for>/<label> ห่อ = label ที่น่าเชื่อถือที่สุดของ form field (ยัง dispatch ด้วย
  // data-ai-index เหมือนเดิม แค่ label ตรงกับที่มนุษย์เห็นมากขึ้น)
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

  // W_field_label_for_plain_inputs: ชื่อช่องล้วนๆ สำหรับทำ prefix — ต่างจาก getAssociatedLabelText
  // ตรงตัดค่าของ control ออกจาก label ที่ห่อ (ไม่งั้นได้ "User Role Admin: Admin")
  const getFieldNameLabel = (node) => {
    if (node.id) {
      try {
        const forLabel = document.querySelector(`label[for="${CSS.escape(node.id)}"]`);
        // label[for] ไม่ได้ห่อ control จึงเป็นชื่อช่องล้วนๆ อยู่แล้ว
        if (forLabel) {
          const t = (forLabel.innerText || forLabel.textContent || '').replace(/\s+/g, ' ').trim();
          if (t) return t;
        }
      } catch (e) { /* id แปลกๆ -- ข้ามไปเงียบๆ เหมือน getAssociatedLabelText */ }
    }
    const wrappingLabel = node.closest ? node.closest('label') : null;
    if (wrappingLabel) {
      const whole = (wrappingLabel.innerText || wrappingLabel.textContent || '');
      const own = (node.innerText || node.textContent || '');
      const t = (own ? whole.split(own).join(' ') : whole).replace(/\s+/g, ' ').trim();
      if (t) return t;
    }
    return getPrecedingSiblingLabelText(node);
  };

  // W_toggle: toggle ของ OrangeHRM ไม่มี id/label[for]/label ที่มีข้อความ — ข้อความจริง
  // ("Include Past Employees") เป็น sibling ก่อนหน้า container เดินขึ้นไม่กี่ชั้น หา sibling
  // ข้อความสั้นที่ไม่มี element โต้ตอบได้ข้างใน คืน '' ถ้าไม่เจอ (ไม่เดา)
  // W_sentence_is_not_a_field_label (gate 2026-09-08, MiniWoB focus-text): เคยหยิบโจทย์ของหน้า
  // ('Focus into the textbox.') มาเป็นชื่อช่อง โมเดลคลิกซ้ำ 15 ครั้ง — ชื่อช่องจริงเป็นวลีสั้น
  // ไม่ใช่ประโยค
  const looksLikeSentence = (t) => /[.!?。]\s*$/.test(t) || t.split(/\s+/).length > 6;

  const getPrecedingSiblingLabelText = (node) => {
    let cur = node;
    for (let depth = 0; depth < 4 && cur; depth++) {
      let sib = cur.previousElementSibling;
      while (sib) {
        const t = (sib.innerText || sib.textContent || '').trim();
        if (t && t.length <= 80 && !looksLikeSentence(t)
            && !sib.querySelector('input, button, select, textarea, a')) {
          return t;
        }
        sib = sib.previousElementSibling;
      }
      cur = cur.parentElement;
    }
    return '';
  };

  // W_select_all_aria_grid (OrangeHRM 2026-08-26): grid สมัยใหม่ (OrangeHRM 5.x, MUI, AG Grid)
  // ไม่ใช้ <table> — เช็คแค่ th/thead ทำให้ checkbox หัวตารางได้ชื่อ 'Select row' agent จึง
  // ไม่เจอ "Select All" แล้วติ๊กทีละแถวจนโดน loop guard
  const TABLE_HEADER_ANCESTOR_SELECTOR = 'th, thead, [role="columnheader"], ' +
    '[class*="table-header" i], [class*="tableheader" i], [class*="header-cell" i]';
  const isInsideTableHeader = (node) => {
    if (node.closest && node.closest(TABLE_HEADER_ANCESTOR_SELECTOR)) return true;
    // แถวที่มี [role=columnheader] = แถวหัวตารางตามนิยาม ARIA
    const row = node.closest ? node.closest('[role="row"]') : null;
    return !!(row && row.querySelector('[role="columnheader"]'));
  };

  // W19 (Log Cleanliness): คลิกเมนู/แท็บที่ active อยู่แล้วไม่เกิด DOM change ทำให้ wait_stable
  // timeout เปล่า — แค่ติด marker ให้ LLM ตัดสินใจเอง ไม่ hard-block (user อาจตั้งใจ refresh)
  const isElementAlreadyActive = (el) => {
    const ariaCurrent = (el.getAttribute('aria-current') || '').toLowerCase();
    if (ariaCurrent === 'page' || ariaCurrent === 'true' || ariaCurrent === 'step') return true;
    if ((el.getAttribute('aria-selected') || '').toLowerCase() === 'true') return true;
    const classStr = el.classList ? Array.from(el.classList).join(' ').toLowerCase() : '';
    const tokens = classStr.split(/\s+/).filter(Boolean);
    return tokens.includes('active') || tokens.includes('selected') || tokens.includes('current');
  };

  // badge ตัวเลขล้วน (เช่น cart-badge "1") ไม่ควรได้ index เอง — ย้ายไปแปะที่พ่อที่คลิกได้จริง
  const CLICKABLE_ANCESTOR_SELECTOR = 'a, button, [role="button"], [role="link"], [onclick]';

  const isBadgeLikeLeaf = (node) => {
    const tag = node.tagName.toLowerCase();
    if (tag === 'a' || tag === 'button') return false;
    const text = (node.innerText || '').trim();
    return text !== '' && /^\d{1,4}$/.test(text);
  };

  // เคลียร์ data-ai-index ของรอบก่อน — ไม่งั้น guard กันแปะซ้ำด้านล่างจะ skip element ที่เคยได้
  // index แล้ว snapshot บนหน้าเดิมจะเห็น element น้อยลงเรื่อยๆ ปลอดภัยเพราะ orchestrator ใช้
  // index ภายใน iteration เดียวกับที่ได้มาเสมอ
  deepQueryAll(document, '[data-ai-index]').forEach((el) => el.removeAttribute('data-ai-index'));

  const nodes = deepQueryAll(document, selectors);

  // icon-only element: span/div ที่มี title/aria-label/data-test* + cursor:pointer แต่ไม่ตรง
  // selectors มาตรฐาน (เช่น demoqa webtables <span title="Edit"><svg/></span>)
  const ICON_LABEL_SELECTOR = '[title], [aria-label], [data-test], [data-testid], [data-qa]';
  for (const cand of deepQueryAll(document, ICON_LABEL_SELECTOR)) {
    if (window.getComputedStyle(cand).cursor !== 'pointer') continue;
    // ตัวเอง/บรรพบุรุษตรง selectors แล้ว -> pass ปกติจัดการแล้ว
    if (cand.closest(selectors)) continue;
    // ห่อ element ที่ตรง selectors -> ให้ตัวข้างในได้ index แทน
    if (cand.querySelector(selectors)) continue;
    nodes.push(cand);
  }

  // W20 (Task10): trigger เมนู profile ของ OrangeHRM (span.oxd-userdropdown-tab) ไม่มี role/
  // tabindex/onclick/title เลย มีแค่ class + cursor:pointer — ไม่เคยได้ index agent จึงไปคลิก
  // "Help" แทน จับด้วย class pattern + cursor:pointer
  const PROFILE_MENU_CLASS_RE = /user-?dropdown|profile-?menu|account-?menu|avatar/i;
  // SPD-2: กรอง candidate ด้วย attribute substring selector ก่อน (แทน querySelectorAll('[class]')
  // ทั้งหน้า) แล้วเช็ค regex ซ้ำ — ผลเท่าเดิมเพราะทุก alternative ของ regex มี substring เหล่านี้
  const PROFILE_MENU_CANDIDATE_SELECTOR = [
    '[class*="dropdown" i]', '[class*="profile" i]', '[class*="account" i]',
    '[class*="avatar" i]', '[class*="menu" i]',
  ].join(',');
  // ลูก (img/p/i ของ userdropdown) ก็ตรง pattern + สืบทอด cursor:pointer — ให้บรรพบุรุษนอกสุด
  // เป็นตัวแทนเดียว ต้องเช็ค cursor ของบรรพบุรุษด้วย: <li class="oxd-userdropdown"> ตรง class
  // แต่ cursor:auto ถ้าเช็คแค่ class จะกัน span ตัวจริงทิ้ง
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
  for (const cand of deepQueryAll(document, PROFILE_MENU_CANDIDATE_SELECTOR)) {
    const classStr = typeof cand.className === 'string' ? cand.className : '';
    if (!PROFILE_MENU_CLASS_RE.test(classStr)) continue;
    if (window.getComputedStyle(cand).cursor !== 'pointer') continue;
    if (hasProfileMenuAncestor(cand)) continue;
    // target = ตัวที่ได้ index จริง (บรรพบุรุษที่ตรง selectors หรือ cand เอง)
    const target = cand.closest(selectors) || cand;
    if (target === cand && !nodes.includes(cand)) {
      if (!cand.querySelector(selectors)) nodes.push(cand);
    }
    // แปะ marker เสมอไม่ว่า target จะได้ index จาก pass ไหน
    profileMenuNodes.add(target);
  }

  // W21 (Custom UI Checkbox): native checkbox ถูกซ่อน แล้ววาด span/div แทน (.oxd-checkbox-input)
  // ไม่มี role/title — selectors มาตรฐานกับ ICON_LABEL_SELECTOR มองไม่เห็น
  const CHECKBOX_WRAPPER_SELECTOR = [
    '.oxd-checkbox-input', '.oxd-table-header-cell-checkbox',
    '[class*="checkbox-input" i]', '[class*="checkbox-wrapper" i]',
    'th label:has(input[type="checkbox"])', 'thead label:has(input[type="checkbox"])',
    'label:has(input[type="checkbox"])',
  ].join(',');
  for (const cand of deepQueryAll(document, CHECKBOX_WRAPPER_SELECTOR)) {
    if (cand.closest(selectors)) continue;
    if (cand.querySelector(selectors)) continue;
    nodes.push(cand);
  }

  // Radio fix (OrangeHRM Gender): <input type=radio> opacity:0 โดน visibility check กรองทิ้ง
  // ส่วน span.oxd-radio-input ที่มองเห็นไม่ตรง selector ใดเลย — ตัวเลือกหายจาก snapshot ทั้งหมด
  const RADIO_WRAPPER_SELECTOR = [
    '.oxd-radio-input', '[class*="radio-input" i]', '[class*="radio-wrapper" i]',
  ].join(',');
  for (const cand of deepQueryAll(document, RADIO_WRAPPER_SELECTOR)) {
    if (cand.closest(selectors)) continue;
    if (cand.querySelector(selectors)) continue;
    nodes.push(cand);
  }

  // W_toggle: pattern เดียวกับ checkbox/radio (span.oxd-switch-input) แต่ class ไม่มีคำว่า
  // checkbox และ label ที่ห่อว่างเปล่า — ไม่มี element ใดแทน toggle เลย
  // แยก selector จาก checkbox โดยเจตนา: ไม่งั้นได้ default label "Select row"/"Select All" ผิดความหมาย
  const SWITCH_WRAPPER_SELECTOR = [
    '[class*="switch-input" i]', '[class*="switch-wrapper" i]', '[role="switch"]',
  ].join(',');
  for (const cand of deepQueryAll(document, SWITCH_WRAPPER_SELECTOR)) {
    if (cand.closest(selectors)) continue;
    if (cand.querySelector(selectors)) continue;
    nodes.push(cand);
  }

  // W21 (Icon-based Action Button Resolver): <button><i class="bi-eye-fill"></i></button>
  // ได้ index อยู่แล้วแต่ไม่มี label — ดึงความหมายจาก class ของ icon ข้างในเป็น fallback
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

    // แปะ index ไปแล้วในรอบนี้ (ตัวพ่อ/badge อีกตัว) — ไม่ push ซ้ำ
    if (el.hasAttribute('data-ai-index')) continue;

    if (isIrrelevant(el)) continue;

    // W_listbox_container: container ที่มี >= 2 option ไม่ใช่เป้าคลิก (ลูปหลักไม่มี guard
    // querySelector(selectors) แบบ pass เสริม จึงหลุดมาได้ทาง [tabindex]/[onclick])
    if (el.querySelectorAll('[role="option"]').length >= 2) continue;

    const rect = el.getBoundingClientRect();
    const st = window.getComputedStyle(el);
    const hasSize = rect.width > 0 && rect.height > 0;
    const notDisplayNone = st.display !== 'none';

    // W47 (hover-to-reveal, uitestingplayground scrolltoclick): ปุ่มที่ opacity:0/visibility:hidden
    // แต่ยังมี layout ให้ติด index ได้ — เฉพาะปุ่ม/ลิงก์/wrapper ไม่รวม input (กัน honeypot/CSRF)
    // display:none ยังกรองทิ้งเสมอ
    const hiddenByOwnStyle = st.visibility === 'hidden' || st.opacity === '0';
    // W21: custom checkbox บางเว็บวาดผ่าน ::before บน wrapper ที่ตัวเอง opacity:0
    const isCheckboxWrapperCandidate = (el.className || '').toString().toLowerCase().includes('checkbox') ||
      (el.tagName.toLowerCase() === 'label' && !!el.querySelector('input[type="checkbox"]'));
    const isRadioWrapperCandidate = (el.className || '').toString().toLowerCase().includes('radio');
    // W_toggle: แยกจาก isCheckboxWrapperCandidate — ต้องไม่ได้ label "Select row"/"Select All"
    const isSwitchCandidate = (el.className || '').toString().toLowerCase().includes('switch') ||
      el.getAttribute('role') === 'switch';
    // W_dropdown_field_label (OrangeHRM 2026-08-26): trigger ของ custom dropdown (div.oxd-select-text)
    // ไม่ใช่ form field จึงได้ label '-- Select --' เหมือนกันทุกช่อง (User Role/Status แยกไม่ออก)
    // — รากของบั๊ก W_filter_safety เดิม
    const dropdownAriaPopup = (el.getAttribute('aria-haspopup') || '').toLowerCase();
    const isDropdownTriggerCandidate = el.getAttribute('role') === 'combobox' ||
      (!!dropdownAriaPopup && dropdownAriaPopup !== 'false') ||
      /select-text|select-wrapper/i.test((el.className || '').toString());
    const isClickableCandidate = ['a', 'button'].includes(el.tagName.toLowerCase()) ||
      el.getAttribute('role') === 'button' || isCheckboxWrapperCandidate || isRadioWrapperCandidate ||
      isSwitchCandidate;
    const hoverRevealCandidate = hasSize && notDisplayNone && hiddenByOwnStyle && isClickableCandidate;

    const visible = hasSize && notDisplayNone && (!hiddenByOwnStyle || hoverRevealCandidate);
    if (!visible) continue;
    // ACC-2: disabled ยังได้ index แต่ติด marker — ไม่งั้น LLM คิดว่าไม่มีปุ่มนี้แล้วเดากดตัวอื่น
    const isDisabled = !!el.disabled;

    // W65[1]: marker [required] ให้รู้ล่วงหน้าว่าต้องมีค่าก่อน submit (ใช้โดย SYSTEM_PROMPT W65 ใน llm.py)
    const isRequired = !!(el.required || el.getAttribute('aria-required') === 'true');

    // W9[A]: เช็ค overlay (cookie banner/modal) บังด้วย elementFromPoint ที่จุดกึ่งกลาง — ติด
    // marker ไม่ตัดทิ้ง เพราะ overlay อาจหายไปแล้วตอน action ทำงานจริง
    // hoverRevealCandidate ข้ามเช็คนี้: visibility:hidden ไม่ถูก hit-test จะได้ obscured=true มั่ว
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

    el.setAttribute('data-ai-index', idx);

    const tag  = el.tagName.toLowerCase();
    const type = el.getAttribute('type') || '';

    // label สำรองของ icon-only element: title, data-test*/data-qa, id (kebab/snake -> ช่องว่าง)
    const humanize = (s) => (s || '').replace(/[-_]+/g, ' ').trim();
    const dataTest = el.getAttribute('data-test') || el.getAttribute('data-testid') ||
                     el.getAttribute('data-qa') || '';
    // W19: <label> ผูก field ชนะ aria-label/title/... เฉพาะ form field (element อื่นไม่มี label[for])
    const isFormFieldTag = tag === 'input' || tag === 'select' || tag === 'textarea' ||
      el.getAttribute('role') === 'combobox';
    const associatedLabel = isFormFieldTag ? getAssociatedLabelText(el) : '';
    // W21: fallback เฉพาะ checkbox wrapper — หัวตาราง = Select All, แถว = Select row
    const checkboxWrapperLabel = isCheckboxWrapperCandidate
      ? (isInsideTableHeader(el) ? 'Select All' : 'Select row')
      : '';
    // Radio fix: span วงกลมไม่มี text — "Male"/"Female" เป็น sibling ใน <label> เดียวกัน
    const radioWrapperLabel = isRadioWrapperCandidate
      ? ((el.closest('label') && (el.closest('label').innerText || '').trim()) || '')
      : '';
    // W_toggle: label ที่ห่อ switch ว่าง ต้องหา sibling ก่อนหน้า ไม่มี default (ปล่อยให้ fallback อื่นลองต่อ)
    const switchLabel = isSwitchCandidate ? getPrecedingSiblingLabelText(el) : '';
    // W_dropdown_field_label: ชื่อ field ใช้ "นำหน้า" ค่าที่เลือกอยู่ (ไม่แทน) เพราะโมเดลต้องเห็นค่าด้วย
    // W_field_label_for_plain_inputs: เดิมเติม prefix ให้แค่ custom dropdown — <select>/<input>
    // มาตรฐานไม่มีชื่อช่อง ทำให้ W_filter_scope_guard fail-open เงียบๆ บนเว็บ form มาตรฐาน
    // ตัด checkbox/radio/ปุ่ม ออก (มี label ทางของตัวเองแล้ว)
    const NON_FILTER_INPUT_TYPES = ['checkbox', 'radio', 'submit', 'button', 'reset', 'image', 'hidden', 'file'];
    const isFilterFieldCandidate = isDropdownTriggerCandidate ||
      tag === 'select' || tag === 'textarea' ||
      (tag === 'input' && !NON_FILTER_INPUT_TYPES.includes(type));
    const dropdownFieldLabel = isFilterFieldCandidate
      ? (isFormFieldTag ? getFieldNameLabel(el) : getPrecedingSiblingLabelText(el))
      : '';
    const semantic = el.getAttribute('aria-label') || el.getAttribute('title') ||
                      humanize(dataTest) || el.getAttribute('name') ||
                      humanize(el.id) || checkboxWrapperLabel || radioWrapperLabel ||
                      switchLabel || getIconClassLabel(el) || '';

    // innerText เป็นตัวเลขล้วน (badge ตะกร้า) -> ผสมกับ semantic แทนทิ้ง
    // hoverRevealCandidate: innerText ของ element ที่ไม่ render คืน '' ใช้ textContent แทน
    const innerTextTrimmed = (el.innerText || '').trim();
    const trimmedText = innerTextTrimmed ||
      (hoverRevealCandidate ? (el.textContent || '').trim() : '');
    const isBareCounter = /^\d{1,3}$/.test(trimmedText);

    let label;
    if (isBareCounter && semantic) {
      label = `${semantic} (${trimmedText})`;
    } else {
      // W19: ลำดับ = ค่าจริงในช่อง > associatedLabel > placeholder > semantic
      // radio/checkbox: value เป็นแค่ id ฝั่ง server ("1"/"2") ให้ associatedLabel ชนะ value
      const isToggleInputType = type === 'radio' || type === 'checkbox';
      // W_password_value_leaks_into_label (live run 2026-09-03): label เคยเป็น
      // 'Current Password: admin123' ส่งเข้า LLM ทุก step — ลบล้างเหตุผลของ fill_secret ทั้งหมด
      // แทนด้วยหมุดคงที่ (ยังบอกสถานะว่าง/กรอกแล้ว)
      const safeValue = type === 'password'
        ? ((el.value || '') ? '[filled]' : '')
        : el.value;
      label = (
        trimmedText ||
        (isToggleInputType ? (associatedLabel || safeValue) : (safeValue || associatedLabel)) ||
        el.getAttribute('placeholder') ||
        semantic ||
        ''
      );
    }
    label = label.trim().replace(/\s+/g, ' ').slice(0, 80);
    // W_widget_semantic_label (P3.5): contenteditable/slider/... innerText คือเนื้อหา ไม่ใช่ชื่อช่อง
    // เติม semantic นำหน้า (คงค่าเดิมไว้)
    const WIDGET_SEMANTIC_ROLES = ['slider', 'spinbutton', 'searchbox', 'textbox', 'treeitem', 'listbox'];
    const elRole = el.getAttribute('role') || '';
    const needsSemanticPrefix = el.isContentEditable || WIDGET_SEMANTIC_ROLES.includes(elRole);
    if (needsSemanticPrefix && semantic && !label.toLowerCase().includes(semantic.toLowerCase())) {
      label = (label ? semantic + ': ' + label : semantic).slice(0, 80);
    }
    // W_dropdown_field_label: เติมชื่อ field เฉพาะเมื่อ label ยังไม่มีชื่อนั้น -> 'User Role: -- Select --'
    if (dropdownFieldLabel && !label.toLowerCase().includes(dropdownFieldLabel.toLowerCase())) {
      // W_empty_field_shows_no_value (2026-08-31): ช่องว่างเคยได้ "Employee Name: Type for hints..."
      // ดูเหมือนมีค่า guard W_empty_table_needs_right_filter จึงปฏิเสธงานที่สำเร็จแล้ว —
      // ช่องว่างแสดงแค่ชื่อช่อง (placeholder เป็นคำใบ้ ไม่ใช่ค่า)
      const isEmptyValueField = isFormFieldTag && !((el.value || '').trim());
      label = (
        isEmptyValueField || !label
          ? dropdownFieldLabel
          : `${dropdownFieldLabel}: ${label}`
      ).slice(0, 80);
    }
    // marker สถานะชั่วคราว (ต้องอยู่ใน LABEL_MARKERS ฝั่ง Python ด้วย) แปะซ้อนกันได้
    // W_dialog_in_snapshot: ใช้ closest() เพราะ region คำนวณหลังจุดนี้
    if (el.closest && el.closest(DIALOG_REGION_SELECTOR)) {
      label = label ? `${label} [in open dialog]` : '[in open dialog]';
    }
    // W20: marker ชัดเจนแทนการเดาจาก username (เปลี่ยนตาม user ที่ login)
    if (profileMenuNodes.has(el)) {
      label = label ? `${label} [Profile/Account Menu]` : '[Profile/Account Menu]';
    }
    if (obscured) {
      label = label ? `${label} [obscured]` : '[obscured]';
    }
    // ต่างจาก obscured: นี่คือซ่อนด้วย CSS ของตัวเอง ไม่ใช่โดนบัง
    if (hoverRevealCandidate) {
      label = label ? `${label} [hidden — may need to hover the row first]` : '[hidden — may need to hover the row first]';
    }
    // ACC-2: "กดได้ไหม" (อิสระจาก marker มองเห็น)
    if (isDisabled) {
      label = label ? `${label} [disabled]` : '[disabled]';
    }
    // W65[1]
    if (isRequired) {
      label = label ? `${label} [required]` : '[required]';
    }

    // W_focus_is_invisible (gate 2026-09-08, MiniWoB focus-text): snapshot ไม่เคยบอกว่า element
    // ไหนโฟกัสอยู่ agent จึงคลิกซ้ำจนหมด step — marker นี้ช่วย focus-text เองไม่ได้ (หน้านั้น blur
    // ทันทีใน on('focus')) แต่ "เคอร์เซอร์อยู่ช่องไหน" จำเป็นทุกครั้งก่อนพิมพ์/กด Enter
    if (el === document.activeElement) {
      label = label ? `${label} [focused]` : '[focused]';
    }

    // W50: ใช้ "จัดลำดับ" ใน text_repr เท่านั้น ไม่กรองทิ้ง (index ไม่เปลี่ยนตามการเรียง)
    const inViewport = rect.bottom > 0 && rect.top < window.innerHeight;

    // W19: "main"/"navigation"/"dialog"/"" ใช้จัดหมวดใน text_repr ไม่ใช่ตัวกรอง
    const region = getRegion(el);

    // W19 (Log Cleanliness): เช็ค active เฉพาะเมนู/แท็บ — class "active" ใช้กว้างมาก (ตัวเลือก
    // ที่ highlight ใน autocomplete ก็ได้ class นี้ แต่เป็นตัวที่ควรคลิก)
    const isNavCandidate = region === 'navigation' || el.getAttribute('role') === 'tab';
    const alreadyActive = isNavCandidate && isElementAlreadyActive(el);
    if (alreadyActive) {
      label = label ? `${label} [already active]` : '[already active]';
    }

    // W_same_label_for_different_fields: ส่ง placeholder แยกให้ Python ใช้ disambiguate label ซ้ำ
    // (ไม่ใส่ใน label ตรงนี้ — ขัด W_empty_field_shows_no_value)
    out.push({ index: idx, tag, type, label, in_viewport: inViewport, region,
               placeholder: (el.getAttribute && el.getAttribute('placeholder')) || '' });
    idx++;
  }
  return out;
}
"""


# ══════════════════════════════════════════════════════════════════════
# โซน 2: marker สถานะ + แยก label ที่ซ้ำกัน (Python)
#   ทำอะไร: จัดการ marker ที่ JS ต่อท้าย label และเติม placeholder ให้ช่องที่ชื่อซ้ำ
#   ทำงานยังไง: label_without_markers() ตัด marker เพื่อเทียบตัวตน, _disambiguate_shared_labels() ใช้ placeholder แยกช่องที่ label เหมือนกัน
# ══════════════════════════════════════════════════════════════════════
# marker สถานะที่ JS ด้านบนต่อท้าย label — อยู่ที่นี่เพราะ orchestrator import จากไฟล์นี้
# (กลับทิศไม่ได้: circular import)
LABEL_MARKERS = (
    "[in open dialog]",
    "[Profile/Account Menu]",
    "[obscured]",
    "[hidden — may need to hover the row first]",
    "[disabled]",
    "[required]",
    "[focused]",
    "[already active]",
)


def label_without_markers(label: str) -> str:
    """label ที่ตัด marker ออก — ใช้เทียบว่าเป็น element เดียวกันไหม (marker เปลี่ยนได้ตลอด)"""
    text = str(label or "")
    for marker in LABEL_MARKERS:
        text = text.replace(marker, " ")
    return " ".join(text.split())


def _disambiguate_shared_labels(elements: list[dict]) -> None:
    """ช่องกรอกหลายช่องที่ label ซ้ำกัน -> เติม placeholder ให้แยกออกจากกัน

    W_same_label_for_different_fields (gate 2026-09-07, add_candidate): First/Middle/Last Name
    ของ OrangeHRM ได้ label "Full Name" เหมือนกันหมด โมเดลเว้น Last Name ว่าง Save ไม่ผ่าน
    เติมเฉพาะตอนซ้ำจริง — เติมทุกช่องขัด W_empty_field_shows_no_value"""
    # W_marker_hides_a_shared_label (gate 3915868): เทียบ label ที่ตัด marker แล้ว ไม่งั้น
    # [focused]/[required] ทำให้ "ไม่ซ้ำ" แล้ว disambiguate เงียบ
    by_label: dict[str, list[dict]] = {}
    for el in elements:
        base = label_without_markers(el.get("label"))
        if base and el.get("tag") in ("input", "textarea", "select"):
            by_label.setdefault(base.lower(), []).append(el)
    for group in by_label.values():
        if len(group) < 2:
            continue
        hints = [str(e.get("placeholder") or "").strip() for e in group]
        # ต้องแยกออกจากกันได้ทุกตัว ไม่งั้นเติมไปก็ยังกำกวม
        if not all(hints) or len(set(hints)) != len(hints):
            continue
        for el, hint in zip(group, hints):
            # เติมในส่วนชื่อ แล้วต่อ marker กลับท้ายสุด
            base = label_without_markers(el.get("label"))
            markers = [m for m in LABEL_MARKERS if m in str(el.get("label") or "")]
            el["label"] = " ".join([f"{base}: {hint}", *markers])


# ══════════════════════════════════════════════════════════════════════
# โซน 3: snapshot (ทางเข้าหลักของ perception) + หา frame
#   ทำอะไร: get_snapshot() ที่ orchestrator เรียกทุก step และ resolve_frame() ที่ actions ใช้ก่อนคลิก
#   ทำงานยังไง: รัน _COLLECT_JS ทุก frame (main ก่อน) -> เรียง dialog/in_viewport ก่อน -> ตัดตาม snapshot_max_elements -> คืน (elements, text_repr)
# ══════════════════════════════════════════════════════════════════════
def _frames_main_first(page: Page) -> list:
    """W40: ทุก frame ของหน้า โดย main frame มาก่อนเสมอ"""
    main_frame = page.main_frame
    return [main_frame] + [f for f in page.frames if f != main_frame]


async def get_snapshot(page: Page):
    """
    คืน (elements, text_repr):
      elements  = list[dict] (index, tag, type, label, in_viewport, region, placeholder) ให้โค้ดใช้
      text_repr = string สรุปสำหรับ prompt LLM (dialog ก่อน แล้ว in_viewport ก่อน)

    W40: เก็บทุก frame, index ต่อกันข้าม frame; frame ที่ evaluate ไม่ได้ (cross-origin/detach)
    ข้ามเงียบๆ — เฟรมเดียวพังไม่ควรทำให้ perceive ทั้งหน้าล้ม
    """
    elements: list[dict] = []
    for frame in _frames_main_first(page):
        try:
            frame_elements = await frame.evaluate(_COLLECT_JS, len(elements))
        except Exception:
            continue
        elements.extend(frame_elements)

    _disambiguate_shared_labels(elements)
    # เรียงแค่ลำดับแสดงผล (stable sort, data-ai-index ไม่เปลี่ยน, ไม่ตัดทิ้ง):
    # W_dialog_in_snapshot: dialog ก่อน — เป็นสิ่งเดียวที่กดได้ขณะเปิดอยู่
    # W50: แล้ว in_viewport ก่อน (ใน iframe อิงตำแหน่ง scroll ของ frame นั้น — ยอมรับได้)
    elements.sort(key=lambda e: (e.get("region") != "dialog", not e.get("in_viewport", True)))

    # W_snapshot_cap (P3.3): ตัดเฉพาะรายการที่ส่ง LLM (ดู config.py::snapshot_max_elements)
    # ตัวที่ถูกตัดเป็นตัวที่ต้อง scroll ไปหาเสมอเพราะเรียงไว้แล้ว
    cap = settings.snapshot_max_elements
    shown = elements[:cap] if cap > 0 and len(elements) > cap else elements
    lines = []
    for e in shown:
        kind = f"{e['tag']}" + (f"({e['type']})" if e['type'] else "")
        label = f" '{e['label']}'" if e['label'] else ""
        # W19: แปะเฉพาะ "(navigation)" — "(main)" เป็นส่วนใหญ่ของหน้า ไม่ช่วยแยก
        region_marker = " (navigation)" if e.get("region") == "navigation" else ""
        lines.append(f"[{e['index']}] {kind}{label}{region_marker}")

    if len(shown) < len(elements):
        # ห้ามตัดเงียบ — ไม่งั้นโมเดลสรุปว่า "ไม่มีปุ่มนี้" (failure mode เดียวกับ W_confident_zero)
        lines.append(
            f"[... {len(elements) - len(shown)} more elements are further down this page and "
            "were left out to keep this list readable — scroll down if what you need is not "
            "listed above]"
        )
    text_repr = "\n".join(lines)
    return elements, text_repr


async def resolve_frame(page: Page, selector: str) -> Union[Page, Frame]:
    """W40: หา frame ที่มี selector นี้ (main ก่อน) ด้วย query_selector() ซึ่งคืนทันที — ไม่ลอง
    click ทีละ frame เพราะจะรอ timeout เต็มทุก frame ที่ไม่เจอ ไม่เจอเลยคืน page ให้ caller
    (actions.py) เจอ error ปกติเอง"""
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


# ══════════════════════════════════════════════════════════════════════
# โซน 4: อ่านเนื้อหาหน้าเว็บ Lane 1 — นับจำนวน
#   ทำอะไร: นับ element ที่ตรง selector แบบ deterministic (ไม่ใช้ LLM)
#   ทำงานยังไง: _COUNT_ELEMENTS_JS นับเฉพาะที่มองเห็นจริง รวมทุก frame
# ══════════════════════════════════════════════════════════════════════
# --- W45: อ่าน "เนื้อหา" หน้าเว็บ (นับ/ตาราง) — ไม่เรียกอัตโนมัติทุก step เรียกผ่าน tool
# "read_page_data" เท่านั้น (actions.py::read_page_data) กัน token ต่อ step โตถาวร

# W63[3.3] (Accurate Record Counting): querySelectorAll().length เดิมนับ node ที่ซ่อน (option/แถว
# ของหน้าอื่นที่ display:none) ได้ตัวเลขเกินจริง — นับเฉพาะที่เห็นจริง (เกณฑ์เดียวกับ
# get_snapshot แต่ไม่รองรับ hover-to-reveal เพราะนับ "สิ่งที่ user เห็น" ไม่ใช่หาปุ่มคลิก)
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
    """Lane 1: นับ element ที่ตรง selector_hint รวมทุก frame — JS ล้วน ไม่เรียก LLM คืนแค่ตัวเลข
    (actions.py::read_page_data เลือกตัวนี้ก่อนเสมอสำหรับคำถามเชิงนับ)
    selector ผิดรูปแบบ = 0 ที่ frame นั้น ไม่ throw"""
    total = 0
    for frame in _frames_main_first(page):
        try:
            total += await frame.evaluate(_COUNT_ELEMENTS_JS, selector_hint)
        except Exception:
            continue
    return total


# ══════════════════════════════════════════════════════════════════════
# โซน 5: อ่านเนื้อหาหน้าเว็บ Lane 2 — ดึงตาราง/list
#   ทำอะไร: ดึงตาราง/list เป็น markdown หรือ JSON และ lookup ค่าที่ถาม (exact -> fuzzy)
#   ทำงานยังไง: _EXTRACT_TABLE_JS (รองรับ <table>/ARIA grid/list + fallback) -> _lookup_annotation -> _cap_rows ตัดแถว -> ไม่เจอค่ารอ AJAX แล้วสแกนซ้ำ 1 รอบ
# ══════════════════════════════════════════════════════════════════════
_EXTRACT_TABLE_JS = r"""
(hint) => {
  const clean = (s) => (s || "").replace(/\s+/g, " ").trim();

  // W_extract_counts_stylesheets (gate 2026-09-07, add_candidate): hint "body" อ่าน body.children
  // รวม <style>/<script> — innerText ของ node ไม่ render fallback เป็น CSS ทั้งไฟล์ ข้อความ
  // "Required" จริงจึงถูกกลบ และ "counted by the system" กลายเป็นตัวยืนยันข้อมูลผิด
  const NON_CONTENT_TAGS = ["SCRIPT", "STYLE", "NOSCRIPT", "TEMPLATE", "LINK", "META", "HEAD"];
  const isContentNode = (node) => !!node && !NON_CONTENT_TAGS.includes(node.tagName);

  // W_ariagrid: OrangeHRM/MUI/AG Grid ใช้ div[role=row/cell] ไม่ใช่ <tr> — fallback ไป [role=row]
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
    const items = Array.from(itemNodes)
      .filter(isContentNode)
      .map((node) => clean(node.innerText))
      .filter(Boolean);
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

  // W_hint_matches_many (saucedemo 2026-08-26): เดิม querySelector(hint) ดูแค่ตัวแรก — hint ที่ตรง
  // หลายตัวพี่น้อง (รายการสินค้า) จึงหลุดไป fallback ได้ footer มาแทน
  // (".inventory_item_name" 6 ตัว -> ["Twitter","Facebook","LinkedIn"])
  //   - มีตาราง -> ตารางที่แถวเยอะสุด
  //   - ตรงหลายตัว -> ชุดนั้นคือรายการ
  //   - ตรงตัวเดียว -> extractFrom ตัวนั้น
  const matches = Array.from(document.querySelectorAll(hint));
  if (matches.length > 0) {
    let bestTable = null;
    for (const el of matches) {
      if (!isTableLike(el)) continue;
      const r = extractTable(el);
      if (r && (!bestTable || r.rows.length > bestTable.rows.length)) bestTable = r;
    }
    if (bestTable) return bestTable;

    if (matches.length > 1) {
      // W_column_aware_count (OrangeHRM 2026-08-26): hint '[role="row"]' รวมแถวหัวตาราง -> นับเกิน 1
      // ตัดแถวหัว (มี th/columnheader หรืออยู่ใน thead) ถ้าตัดแล้วว่างคืนชุดเดิม (fail-safe)
      const isHeaderRow = (el) =>
        !!(el.querySelector && el.querySelector('th, [role="columnheader"]')) ||
        !!(el.closest && el.closest('thead'));
      const dataMatches = matches.filter((el) => !isHeaderRow(el));
      const items = (dataMatches.length > 0 ? dataMatches : matches)
        .filter(isContentNode)
        .map((el) => clean(el.innerText)).filter(Boolean);
      if (items.length > 0) return { kind: "list", items };
    }

    const direct = extractFrom(matches[0]);
    if (direct) return direct;
  }

  // Fallback: LLM เดา hint ผิด (snapshot ไม่โชว์โครงสร้างตาราง) — หาตาราง/list จริงที่ใหญ่สุดบนหน้า
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
    """W46: candidate ที่ SequenceMatcher.ratio() สูงสุด (ไม่สนตัวพิมพ์) ถ้า >= threshold ไม่งั้น None

    default แยกจาก settings.agent_fuzzy_match_threshold ให้เทสต์ได้ตรงๆ — ต่ำไป = จับคนละคน
    (อันตรายกว่า เพราะตอบผิดคนแบบมั่นใจ) สูงไป = พลาดคำพิมพ์ผิด"""
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
    """หา query ใน candidates: exact substring (case-insensitive ทั้งสองทิศ) ก่อน แล้วค่อย fuzzy

    คืน (annotation, found) — annotation แปะหน้าผลตอน fuzzy match ให้ LLM รู้ว่าเป็นการเดา
    ("Cierra Vaga"/"Cierra Vega") found=False เมื่อมี query แต่ไม่เจอเลย ให้ผู้เรียกคืน [FAIL]"""
    if not query:
        return "", True
    lower_query = query.lower()
    if any(lower_query in c.lower() or c.lower() in lower_query for c in candidates):
        return "", True
    match = fuzzy_find(query, candidates, threshold=settings.agent_fuzzy_match_threshold)
    if match is None:
        return "", False
    return f"[พบ '{match}' ใกล้เคียงกับคำค้น '{query}' ที่คุณพิมพ์ — ไม่ตรงกันเป๊ะ ตรวจสอบก่อนใช้คำตอบ]\n\n", True


# W64[7.2]: รอก่อน lookup ซ้ำ — พอให้ AJAX table reload เสร็จ ไม่ช้าเกินเมื่อไม่มีข้อมูลจริง
_LOOKUP_RETRY_WAIT_SEC = 2.0


def _cap_rows(rows: list, total: int) -> tuple[list, str]:
    """W_extract_row_cap (P4.5): ตัดแถวไม่เกิน settings.read_page_data_max_rows พร้อมบอกตรงๆ

    ตัดได้เพราะ W_deterministic_count (actions.py) นับจากข้อมูลชุดเต็มแล้ว — ห้ามตัดเงียบ
    total = จำนวนก่อนตัด (ผู้เรียกรู้ดีว่าอะไรนับเป็น 1 รายการ)"""
    limit = settings.read_page_data_max_rows
    if limit <= 0 or total <= limit:
        return rows, ""
    note = (
        f"\n[showing the first {limit} of {total} entries — the rest were cut to keep this "
        "response small. The counts stated above were computed by the system from ALL "
        f"{total} entries, not just the {limit} shown, so use those numbers. If you need a "
        "specific entry that is not listed here, narrow the search/filter on the page first.]"
    )
    return rows[:limit], note


async def extract_table_data(page: Page, table_hint: str, query: str = "") -> str:
    """Lane 2: ตาราง/list ที่ตรง table_hint -> markdown table หรือ JSON list กระชับ

    เรียกผ่าน tool "read_page_data" เมื่อคำถามตอบด้วยการนับอย่างเดียวไม่ได้ (ดู count_elements)
    query ว่าง = สรุปทั้งก้อน; มีค่า = lookup (exact ก่อน แล้ว fuzzy) ไม่เจอคืน "[FAIL] ..."
    hint ไม่ตรง -> fallback หาตาราง/list จริงบนหน้า (ดู _EXTRACT_TABLE_JS) ไม่ throw

    W64[7.2]: agent ค้นแถวที่เพิ่ง Save ก่อน AJAX reload เสร็จ แล้วเข้าใจว่าบันทึกไม่สำเร็จ —
    มี query และรอบแรกไม่เจอ ให้รอ _LOOKUP_RETRY_WAIT_SEC แล้วสแกนซ้ำหนึ่งรอบ"""
    result = await _extract_table_data_once(page, table_hint, query)
    if query and result.startswith("[FAIL]"):
        await asyncio.sleep(_LOOKUP_RETRY_WAIT_SEC)
        result = await _extract_table_data_once(page, table_hint, query)
    return result


async def _extract_table_data_once(page: Page, table_hint: str, query: str) -> str:
    """W64[7.2]: สแกน 1 รอบ แยกออกมาให้ extract_table_data() เรียกซ้ำได้"""
    for frame in _frames_main_first(page):
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
                return f"[FAIL] nothing matching or close to '{query}' was found in '{table_hint}'"
            lines = [
                "| " + " | ".join(header) + " |",
                "| " + " | ".join("---" for _ in header) + " |",
            ]
            capped_body, cap_note = _cap_rows(body, len(body))
            lines += ["| " + " | ".join(row) + " |" for row in capped_body]
            return annotation + "\n".join(lines) + cap_note

        items = data["items"]
        annotation, found = _lookup_annotation(query, items)
        if not found:
            return f"[FAIL] nothing matching or close to '{query}' was found in '{table_hint}'"
        capped_items, cap_note = _cap_rows(items, len(items))
        return annotation + json.dumps(capped_items, ensure_ascii=False) + cap_note

    return f"[FAIL] no element matching '{table_hint}' was found"


# ══════════════════════════════════════════════════════════════════════
# โซน 6: helper + demo (python run.py perception)
#   ทำอะไร: สั่ง click/fill/select/scroll ด้วย index สำหรับ demo เท่านั้น (agent loop ใช้ actions.py)
#   ทำงานยังไง: คืน "[OK]"/"[FAIL] ..." เสมอ ไม่ raise
# ══════════════════════════════════════════════════════════════════════
# คืน "[OK]" / "[FAIL] เหตุผล" เสมอ ไม่ raise

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
# DEMO (python run.py perception): saucedemo — ดู snapshot แล้วลอง login
# ------------------------------------------------------------
async def demo():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()
        await install_ssrf_guard(page)
        await page.goto("https://www.saucedemo.com/")

        # 1) Perceive
        elements, text_repr = await get_snapshot(page)
        print("=== หน้า Login ที่ AI มองเห็น ===")
        print(text_repr)
        print()

        # 2) Act — จำลอง LLM สั่งด้วย index (user/pass มาตรฐานของ saucedemo)
        u_idx = next(e['index'] for e in elements if 'user' in e['label'].lower())
        p_idx = next(e['index'] for e in elements if 'pass' in e['label'].lower())
        b_idx = next(e['index'] for e in elements if e['tag'] == 'input' and e['type'] == 'submit')

        print("[LOGIN]")
        print(" fill username:", await fill_by_index(page, u_idx, "standard_user"))
        print(" fill password:", await fill_by_index(page, p_idx, "secret_sauce"))
        print(" click submit :", await click_by_index(page, b_idx))
        await page.wait_for_load_state("networkidle")

        # 3) Perceive หน้าใหม่
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

        # --- error handling: index ที่ไม่มีอยู่จริงต้องไม่ crash ---
        print("\n[ERROR HANDLING] ยิง action ด้วย index ผิดๆ (ไม่ควร crash)")
        print(" click index=9999 ->", await click_by_index(page, 9999))
        print(" fill  index=9999 ->", await fill_by_index(page, 9999, "x"))
        print(" select index=9999 ->", await select_by_index(page, 9999, "x"))
        print(" (โปรแกรมยังรันต่อได้ไม่ crash = error handling ทำงาน)")

        # --- หน้า cart ---
        await page.click("button:has-text('Add to cart')")
        await page.click(".shopping_cart_link")
        await page.wait_for_load_state("networkidle")
        _, cart_repr = await get_snapshot(page)
        print("\n=== หน้า Cart ที่ AI มองเห็น ===")
        print(cart_repr)

        # --- หน้า checkout ---
        await page.click("#checkout")
        await page.wait_for_load_state("networkidle")
        _, checkout_repr = await get_snapshot(page)
        print("\n=== หน้า Checkout ที่ AI มองเห็น ===")
        print(checkout_repr)

        await asyncio.sleep(3)
        await browser.close()


if __name__ == "__main__":
    asyncio.run(demo())

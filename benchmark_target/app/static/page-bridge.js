/* Execute a small, explicit action vocabulary in THIS document. Never eval model code. */
(function () {
  const documentId = crypto.randomUUID();
  const ids = new WeakMap();
  const elements = new Map();
  let nextId = 1;
  const selector = 'a[href],button,input:not([type="hidden"]),textarea,select,[role="button"],[role="link"],[role="combobox"],[contenteditable="true"]';
  const visible = el => {
    const style = getComputedStyle(el);
    return !el.closest('#hermes-ai-bar-host') && el.getClientRects().length > 0 &&
      style.display !== 'none' && style.visibility !== 'hidden';
  };
  const label = el => {
    const labelled = (el.getAttribute('aria-labelledby') || '').split(/\s+/)
      .map(id => document.getElementById(id)?.textContent || '').join(' ').trim();
    return (el.getAttribute('aria-label') || labelled ||
      Array.from(el.labels || []).map(item => item.innerText).join(' ') ||
      el.innerText || el.getAttribute('placeholder') || el.getAttribute('title') ||
      el.getAttribute('name') || (el.type === 'password' ? 'Password' : el.getAttribute('value')) || '').trim().slice(0, 500);
  };
  function snapshot() {
    const indexed = Array.from(document.querySelectorAll(selector)).filter(visible).slice(0, 500).map(el => {
      if (!ids.has(el)) ids.set(el, nextId++);
      const index = ids.get(el);
      elements.set(index, el);
      return {
        index, tag: el.tagName.toLowerCase(), type: el.getAttribute('type') || '', label: label(el),
        value: el.type === 'password' ? '' : String(el.value || '').slice(0, 2000),
        href: el.tagName === 'A' ? el.href.slice(0, 4000) : '',
        options: el.tagName === 'SELECT' ? Array.from(el.options).slice(0, 200).map(o => o.text) : [],
        checked: !!el.checked, disabled: !!el.disabled || el.getAttribute('aria-disabled') === 'true',
      };
    });
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
    const parts = [];
    let node, length = 0;
    while ((node = walker.nextNode()) && length < 40000) {
      const parent = node.parentElement;
      if (!parent || parent.closest('script,style,noscript,#hermes-ai-bar-host') || !visible(parent)) continue;
      const value = node.textContent.trim();
      if (value) { parts.push(value); length += value.length + 1; }
    }
    return { document_id: documentId, url: location.href, title: document.title.slice(0, 500),
      text: parts.join('\n').slice(0, 40000), elements: indexed };
  }
  function sameOrigin(url) {
    const target = new URL(url, location.href);
    if (!['http:', 'https:'].includes(target.protocol) || target.origin !== location.origin)
      throw new Error('คำสั่งนี้ออกนอกเว็บไซต์ปัจจุบันไม่ได้');
    return target.href;
  }
  function guardDestination(el) {
    const anchor = el.closest('a[href]');
    if (anchor) {
      sameOrigin(anchor.href);
      if (anchor.hasAttribute('download')) throw new Error('การดาวน์โหลดต้องกดด้วยตัวเอง');
      anchor.target = '_self';
    }
    const form = el.form;
    if (form) {
      sameOrigin(el.getAttribute('formaction') || form.action || location.href);
      form.target = '_self';
      if (el.hasAttribute('formtarget')) el.setAttribute('formtarget', '_self');
    }
  }
  function nativeValue(el, value) {
    if (el.readOnly) throw new Error('ช่องนี้อ่านได้อย่างเดียว');
    const proto = el.tagName === 'TEXTAREA' ? HTMLTextAreaElement.prototype :
      el.tagName === 'SELECT' ? HTMLSelectElement.prototype : HTMLInputElement.prototype;
    const setter = Object.getOwnPropertyDescriptor(proto, 'value')?.set;
    if (el.isContentEditable) el.textContent = value;
    else if (setter) setter.call(el, value);
    else throw new Error('เลือกช่องกรอกข้อมูลที่แก้ไขได้');
    el.dispatchEvent(new Event('input', { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
  }
  async function execute(command) {
    const cmd = command.action;
    const expected = cmd.expected_document_id || command.document_id;
    if (expected !== documentId) throw new Error('หน้าเว็บเปลี่ยนแล้ว ให้ตรวจหน้าใหม่ก่อนทำงาน');
    if (cmd.type === 'snapshot') return 'อ่านหน้าปัจจุบันแล้ว';
    if (cmd.type === 'goto') { location.assign(sameOrigin(cmd.url)); return 'กำลังเปลี่ยนหน้า'; }
    if (cmd.type === 'scroll') {
      window.scrollBy(0, (cmd.direction === 'up' ? -1 : 1) * Math.min(2000, Math.abs(cmd.amount || 600)));
    } else if (cmd.type !== 'wait') {
      const el = elements.get(cmd.index);
      if (!el || !el.isConnected || !visible(el)) throw new Error('ไม่พบ element เดิมในหน้านี้');
      if (el.disabled || el.getAttribute('aria-disabled') === 'true') throw new Error('element นี้ยังใช้งานไม่ได้');
      el.scrollIntoView({ block: 'center', behavior: 'instant' });
      el.focus();
      switch (cmd.type) {
        case 'click':
          guardDestination(el);
          el.click();
          break;
        case 'fill':
          if (el.type === 'file') throw new Error('เลือกไฟล์ด้วยตัวเองก่อน');
          nativeValue(el, String(cmd.text || ''));
          break;
        case 'select': {
          if (el.tagName !== 'SELECT') throw new Error('กรุณาคลิก dropdown นี้แล้วเลือกตัวเลือก');
          const option = Array.from(el.options).find(o => o.text.trim() === cmd.label || o.value === cmd.label);
          if (!option || option.disabled) throw new Error('ไม่พบตัวเลือกที่ใช้งานได้');
          nativeValue(el, option.value);
          break;
        }
        case 'check':
          if (!['checkbox', 'radio'].includes(el.type)) throw new Error('element นี้ไม่ใช่ checkbox/radio');
          if (!el.checked) el.click();
          break;
        case 'press_key': {
          const key = cmd.key;
          if (!['Enter', 'Tab', 'Escape', 'ArrowDown', 'ArrowUp', 'ArrowLeft', 'ArrowRight', 'Space', ' '].includes(key))
            throw new Error('ปุ่มนี้ยังไม่รองรับในหน้าเว็บ');
          guardDestination(el);
          const actualKey = key === 'Space' ? ' ' : key;
          const allowed = el.dispatchEvent(new KeyboardEvent('keydown', { key: actualKey, bubbles: true, cancelable: true }));
          if (allowed && key === 'Enter' && ['A', 'BUTTON'].includes(el.tagName)) el.click();
          else if (allowed && key === 'Enter' && el.form && el.tagName !== 'TEXTAREA') el.form.requestSubmit();
          else if (allowed && actualKey === ' ' && ['checkbox', 'radio'].includes(el.type)) el.click();
          el.dispatchEvent(new KeyboardEvent('keyup', { key: actualKey, bubbles: true }));
          break;
        }
        case 'hover':
          el.dispatchEvent(new MouseEvent('mouseover', { bubbles: true }));
          el.dispatchEvent(new MouseEvent('mouseenter', { bubbles: false }));
          break;
        default: throw new Error('คำสั่งนี้ไม่รองรับในหน้าเว็บ');
      }
    }
    await new Promise(resolve => setTimeout(resolve, cmd.type === 'wait' ? 700 : 200));
    return 'ทำคำสั่งแล้ว ตรวจผลจากหน้าปัจจุบัน';
  }
  window.HermesPageBridge = { snapshot, execute, documentId };
})();

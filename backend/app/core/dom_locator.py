"""core/dom_locator.py — W_procmem: locator ที่ "อยู่รอด" ข้าม task run ได้จริง สำหรับ
Procedural Memory (ดู core/procedural_memory.py, core/fastpath_executor.py)

ปัญหาที่ต้องแก้: element addressing ที่ agent loop ปกติใช้ (data-ai-index ใน
core/perception.py/actions.py) เป็น "ephemeral" 100% — _COLLECT_JS ล้างแล้วแปะเลขใหม่
ทุกครั้งที่ get_snapshot() ถูกเรียก ไม่มีทางอ้างอิงข้าม step หรือข้าม task run ได้เลย
ทำให้ template ที่ Abstractor สร้างไว้ไม่มีอะไรให้ "target" ชี้กลับไปหาตอน replay ในอนาคต

ไฟล์นี้จึงมี 2 ฝั่งที่ใช้ locator descriptor เดียวกัน:
  - compute_locator_descriptor(): ฝั่ง "จับภาพ" — เรียกตอน action สำเร็จจริงใน
    actions.py (click/fill/select_option/check) เพื่อบันทึกว่า element ที่เพิ่งกระทำ
    ไปคือใคร ด้วยคำอธิบายที่ยังใช้หาตัวเดิมได้ในหน้าเว็บเวอร์ชันอนาคต
  - resolve_locator(): ฝั่ง "ค้นหากลับ" — ใช้ตอน fastpath_executor.py replay
    template เดิม โดยไล่ fallback chain role+name -> label -> data-testid -> CSS
    (ลำดับเดียวกับที่ user ระบุไว้ตรงๆ ใน Repair prompt: "prefer role+name, label, or
    data-testid over long CSS/XPath")

CSS fallback chain (css_fallback field) ใช้ priority เดียวกับ
site_learning/extractor.py::computeSelector() ทุกประการ (data-testid/data-test > id ที่
unique > class combo ที่ unique > nth-of-type path) — คัดลอกมาแยกไว้ที่นี่แทนที่จะ
import JS string ข้ามไฟล์ (เขียน JS แบบ inline string ในทั้งสองไฟล์อยู่แล้ว ไม่มี
กลไก share JS ระหว่างไฟล์ในระบบนี้) ถ้าแก้ priority ฝั่งใดฝั่งหนึ่งต้องแก้ให้ตรงกันทั้งคู่

ทุกฟังก์ชันในไฟล์นี้ "ห้าม throw ออกไปเด็ดขาด" (กฎเดียวกับ plan_memory.py/
long_term_memory.py/perception.py) — คืนค่า fallback ที่ปลอดภัย (dict ว่าง/None) แทน
เสมอ เพราะเป็นแค่ enhancement (บันทึกไว้ใช้ทีหลัง) ไม่ใช่สิ่งที่ agent loop หลักต้องมี
ถึงจะทำงานได้
"""

from typing import Optional, Union

from playwright.async_api import Frame, Locator, Page

_STABLE_LOCATOR_JS = r"""
(el) => {
  const escapeCss = (s) => (window.CSS && CSS.escape) ? CSS.escape(s) : s.replace(/[^a-zA-Z0-9_-]/g, '\\$&');

  // ลำดับเดียวกับ site_learning/extractor.py::computeSelector() ทุกประการ — ดู
  // docstring หัวไฟล์นี้ก่อนแก้ ต้องแก้ให้ตรงกันทั้งสองที่
  const computeSelector = (node) => {
    const testid = node.getAttribute('data-testid') || node.getAttribute('data-test');
    if (testid) return `[data-testid="${testid}"], [data-test="${testid}"]`;
    if (node.id && document.querySelectorAll(`#${escapeCss(node.id)}`).length === 1) {
      return `#${escapeCss(node.id)}`;
    }
    if (typeof node.className === 'string' && node.className.trim()) {
      const classes = node.className.trim().split(/\s+/).filter(Boolean);
      if (classes.length) {
        const sel = node.tagName.toLowerCase() + '.' + classes.map(escapeCss).join('.');
        try {
          if (document.querySelectorAll(sel).length === 1) return sel;
        } catch (e) { /* selector แปลกๆ ที่ escape ไม่พอ ข้ามไปใช้ nth-of-type แทน */ }
      }
    }
    let path = [];
    let cur = node;
    while (cur && cur.nodeType === 1 && cur !== document.body) {
      let seg = cur.tagName.toLowerCase();
      if (cur.parentElement) {
        const siblings = Array.from(cur.parentElement.children).filter((c) => c.tagName === cur.tagName);
        if (siblings.length > 1) seg += `:nth-of-type(${siblings.indexOf(cur) + 1})`;
      }
      path.unshift(seg);
      cur = cur.parentElement;
    }
    return path.join(' > ');
  };

  // implicit ARIA role ตาม tag/type พื้นฐานที่พบบ่อยที่สุด (ไม่ครอบคลุมทุกกรณีตามสเปค
  // ARIA เต็มรูปแบบ แต่พอสำหรับ element ที่ actions.py กระทำได้จริงในระบบนี้:
  // click/fill/select_option/check เท่านั้น)
  const IMPLICIT_ROLE_MAP = {
    a: 'link', select: 'combobox', textarea: 'textbox',
  };
  const inferImplicitRole = (node) => {
    const tag = node.tagName.toLowerCase();
    if (tag === 'button') return 'button';
    if (tag === 'a' && node.hasAttribute('href')) return 'link';
    if (tag === 'select') return 'combobox';
    if (tag === 'textarea') return 'textbox';
    if (tag === 'input') {
      const type = (node.getAttribute('type') || 'text').toLowerCase();
      if (type === 'submit' || type === 'button') return 'button';
      if (type === 'checkbox') return 'checkbox';
      if (type === 'radio') return 'radio';
      if (['text', 'email', 'password', 'search', 'tel', 'url', 'number'].includes(type)) return 'textbox';
    }
    return IMPLICIT_ROLE_MAP[tag] || '';
  };

  // Accessible name — ลำดับ: aria-label -> aria-labelledby (resolve id แล้วเอา
  // textContent) -> <label for=id> ที่ผูกไว้ -> placeholder -> visible text/value
  // (ลำดับเดียวกับที่ extractor.py ใช้แยกฟิลด์ต่อ element ประเภทต่างๆ อยู่แล้ว รวมมาไว้
  // เป็นฟังก์ชันเดียวที่นี่)
  const computeAccessibleName = (node) => {
    const ariaLabel = node.getAttribute('aria-label');
    if (ariaLabel && ariaLabel.trim()) return ariaLabel.trim();
    const labelledBy = node.getAttribute('aria-labelledby');
    if (labelledBy) {
      const parts = labelledBy.split(/\s+/).map((id) => {
        const el2 = document.getElementById(id);
        return el2 ? el2.textContent.trim() : '';
      }).filter(Boolean);
      if (parts.length) return parts.join(' ');
    }
    if (node.id) {
      const labelEl = document.querySelector(`label[for="${escapeCss(node.id)}"]`);
      if (labelEl && labelEl.textContent.trim()) return labelEl.textContent.trim();
    }
    const closestLabel = node.closest('label');
    if (closestLabel && closestLabel.textContent.trim()) return closestLabel.textContent.trim();
    const placeholder = node.getAttribute('placeholder');
    if (placeholder && placeholder.trim()) return placeholder.trim();
    const text = (node.innerText || node.value || '').trim();
    if (text) return text.slice(0, 200);
    const title = node.getAttribute('title');
    return title ? title.trim() : '';
  };

  return {
    tag: el.tagName.toLowerCase(),
    explicit_role: el.getAttribute('role') || '',
    implicit_role: inferImplicitRole(el),
    accessible_name: computeAccessibleName(el),
    data_testid: el.getAttribute('data-testid') || el.getAttribute('data-test') || '',
    css_fallback: computeSelector(el),
  };
}
"""


async def compute_locator_descriptor(target: Union[Page, Frame], selector: str) -> dict:
    """เรียกตอน action (click/fill/select_option/check) สำเร็จแล้วเท่านั้น — target คือ
    Frame/Page ที่ resolve_frame() คืนมา (ตัวเดียวกับที่ actions.py ใช้ dispatch action
    จริง), selector คือ _sel(index) เดิม (ephemeral แต่ยังใช้ query element ตัวเดียวกัน
    ได้ในจังหวะที่ action พึ่งสำเร็จ) คืน dict ว่างเปล่าถ้า evaluate ล้มเหลวไม่ว่ากรณีใด
    (element หาย/frame ถูก detach ระหว่างทาง ฯลฯ) — ไม่ throw ออกไปให้ actions.py พังตาม
    เด็ดขาด (descriptor เป็นแค่ข้อมูลเสริมสำหรับบันทึกไว้ใช้ทีหลัง ไม่ใช่ผลลัพธ์ของ
    action เอง)"""
    try:
        return await target.locator(selector).first.evaluate(_STABLE_LOCATOR_JS)
    except Exception:
        return {}


async def resolve_locator(target: Union[Page, Frame], descriptor: dict) -> Optional[Locator]:
    """ไล่ fallback chain ตามลำดับที่ user ระบุไว้ตรงๆ ใน Repair prompt: role+name ก่อน
    -> label -> data-testid -> CSS ยาวๆ เป็นทางเลือกสุดท้าย — คืน Locator ตัวแรกที่
    resolve ได้ "พอดี 1 ตัว" เท่านั้น (count()==1 — ทั้ง 0 ตัวและมากกว่า 1 ตัวถือว่า
    ทางนี้ใช้ไม่ได้ ไปลองทางถัดไป เพราะ locator ที่กำกวมอันตรายพอๆ กับหาไม่เจอเลย) คืน
    None ถ้าทุกทางในเชนล้มเหลวหมด (ให้ผู้เรียก — fastpath_executor.py — ตีความว่าต้อง
    เรียก Repair module ต่อ)"""
    name = descriptor.get("accessible_name") or None
    role = descriptor.get("explicit_role") or descriptor.get("implicit_role") or None

    candidates = []
    if role:
        candidates.append(lambda: target.get_by_role(role, name=name) if name else target.get_by_role(role))
    if name:
        candidates.append(lambda: target.get_by_label(name))
    if descriptor.get("data_testid"):
        candidates.append(lambda: target.get_by_test_id(descriptor["data_testid"]))
    if descriptor.get("css_fallback"):
        candidates.append(lambda: target.locator(descriptor["css_fallback"]))

    for make_locator in candidates:
        try:
            locator = make_locator()
            if await locator.count() == 1:
                return locator
        except Exception:
            continue
    return None

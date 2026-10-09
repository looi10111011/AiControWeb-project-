"""core/dom_locator.py — W_procmem: locator ที่อยู่รอดข้าม task run สำหรับ Procedural Memory
(data-ai-index ของ perception.py ถูกล้าง/แปะใหม่ทุก get_snapshot() จึงอ้างอิงข้าม step/run ไม่ได้)

  - compute_locator_descriptor(): จับภาพ element ตอน action สำเร็จใน actions.py
  - resolve_locator(): หาคืนตอน fastpath_executor.py replay — role+name -> label -> data-testid -> CSS

CSS fallback chain ใช้ priority เดียวกับ site_learning/extractor.py::computeSelector() (data-testid >
id ที่ unique > class combo ที่ unique > nth-of-type path) คัดลอกไว้เพราะไม่มีกลไก share JS — แก้ต้องแก้ทั้งคู่

ทุกฟังก์ชันห้าม throw — คืน fallback ที่ปลอดภัย ({}/None) เพราะเป็นแค่ enhancement
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


# W_descriptor_timeout: เพดาน evaluate() ใน compute_locator_descriptor() — สั้นพอไม่รอ element ที่หายแล้ว
# ยาวพอสำหรับหน้าที่ยังไม่นิ่ง
_DESCRIPTOR_TIMEOUT_MS = 1000


async def compute_locator_descriptor(target: Union[Page, Frame], selector: str) -> dict:
    """เรียกหลัง action (click/fill/select_option/check) สำเร็จ — target คือ Frame/Page จาก resolve_frame(),
    selector คือ _sel(index) เดิม คืน descriptor dict หรือ {} ถ้าล้มเหลวใดๆ; never raises

    W_descriptor_timeout (จาก step trace, saucedemo: select เรียงราคาใช้ 30.0s จาก task 42s แล้วได้ {}):
    เดิมไม่ตั้ง timeout -> default 30s และ evaluate() auto-wait ขณะที่หน้า re-render ลบ data-ai-index ไปแล้ว
    descriptor มีความหมายแค่ ณ จังหวะ action สำเร็จ รอต่อไม่ทำให้ element กลับมา"""
    try:
        return await target.locator(selector).first.evaluate(
            _STABLE_LOCATOR_JS, timeout=_DESCRIPTOR_TIMEOUT_MS,
        )
    except Exception:
        return {}


async def resolve_locator(target: Union[Page, Frame], descriptor: dict) -> Optional[Locator]:
    """ไล่ role+name -> label -> data-testid -> CSS คืน Locator ตัวแรกที่ count()==1 พอดี (กำกวมอันตราย
    พอๆ กับไม่เจอ) หรือ None ถ้าทุกทางล้ม (fastpath_executor.py จะเรียก Repair ต่อ)"""
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

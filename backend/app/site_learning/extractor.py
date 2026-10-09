"""site_learning/extractor.py — W14: DOM extraction สำหรับ crawler.py (selector/xpath ที่ใช้ซ้ำข้ามรอบ + form/table/nav
เต็มรูปแบบ) แยกจาก perception.py::get_snapshot() ที่คืนแค่ 4 ฟิลด์; _EXTRACT_JS ยึด convention เดียวกัน
(เช็ค visibility, ไม่ throw จาก JS).

W18: inferIconHint() เดาความหมายปุ่ม icon-only + UI pattern detection (>=3 element โครงสร้างเดียวกัน -> UIPatternInfo
     เดียว; heuristic: tag + sorted classes + child-tag sequence แบบ exact)
W24: เพิ่ม [role=menuitem]/[role=tab]/router-link ใน BUTTON_SELECTOR (SPA ที่ไม่มี a/button) + is_nav_menu_item
     ให้ crawler default-allow (crawler.py::_is_explorable)
W39: ปุ่มใน <iframe> มองไม่เห็นเพราะ page.evaluate() query แค่ main document — รัน _EXTRACT_JS กับทุก frame ใน
     page.frames (flat อยู่แล้ว) แล้วแปะ frame_index ให้ crawler.py::_resolve_click_target กดผ่าน frame ที่ถูก
"""

from typing import Optional

from playwright.async_api import Frame, Page

from backend.app.site_learning.safety import classify_button_tier
from backend.app.site_learning.schema import ButtonInfo, FormFieldInfo, PageInfo, TableInfo, UIPatternInfo

_EXTRACT_JS = r"""
() => {
  const escapeCss = (s) => (window.CSS && CSS.escape) ? CSS.escape(s) : s.replace(/[^a-zA-Z0-9_-]/g, '\\$&');

  const isVisible = (el) => {
    const rect = el.getBoundingClientRect();
    const st = window.getComputedStyle(el);
    return rect.width > 0 && rect.height > 0 && st.visibility !== 'hidden' &&
           st.display !== 'none' && st.opacity !== '0';
  };

  // ลำดับความสำคัญ: data-testid/data-test > id ที่ unique > class combo ที่ unique >
  // nth-of-type path จาก body — ต้องเป็น selector ที่ "อยู่รอด" ข้าม snapshot ได้ (ต่าง
  // จาก data-ai-index ของ perception.py ที่ต้องแปะใหม่ทุกรอบ)
  const computeSelector = (el) => {
    const testid = el.getAttribute('data-testid') || el.getAttribute('data-test');
    if (testid) return `[data-testid="${testid}"], [data-test="${testid}"]`;
    if (el.id && document.querySelectorAll(`#${escapeCss(el.id)}`).length === 1) {
      return `#${escapeCss(el.id)}`;
    }
    if (typeof el.className === 'string' && el.className.trim()) {
      const classes = el.className.trim().split(/\s+/).filter(Boolean);
      if (classes.length) {
        const sel = el.tagName.toLowerCase() + '.' + classes.map(escapeCss).join('.');
        try {
          if (document.querySelectorAll(sel).length === 1) return sel;
        } catch (e) { /* selector แปลกๆ ที่ escape ไม่พอ ข้ามไปใช้ nth-of-type แทน */ }
      }
    }
    let path = [];
    let node = el;
    while (node && node.nodeType === 1 && node !== document.body) {
      let seg = node.tagName.toLowerCase();
      if (node.parentElement) {
        const siblings = Array.from(node.parentElement.children).filter((c) => c.tagName === node.tagName);
        if (siblings.length > 1) seg += `:nth-of-type(${siblings.indexOf(node) + 1})`;
      }
      path.unshift(seg);
      node = node.parentElement;
    }
    return path.join(' > ');
  };

  // W24: คอนเทนเนอร์เมนู/นำทาง — ใช้ทั้ง isNavMenuItem() และ nav_links
  const NAV_CONTAINERS = 'nav, [role=navigation], aside, header, footer, [role=tablist], [role=menu]';

  // W24: เมนู/นำทาง = อยู่ใน NAV_CONTAINERS, role menuitem/tab, <router-link> หรือ class router-link (SPA ที่
  // route ด้วย JS) -> crawler.py::_is_explorable ให้ default-allow
  // *** ยกเว้น <a> ที่มี href จริง: BFS เดินตาม href อยู่แล้วโดยเช็ค same-origin ก่อน goto; ถ้าให้ true
  // _explore_buttons() จะ "คลิก" ซ้ำ ซึ่ง navigate ก่อนรู้ปลายทาง เสี่ยงหลุดไปเว็บอื่น ***
  const isNavMenuItem = (el) => {
    if (el.tagName.toLowerCase() === 'a') {
      const href = el.getAttribute('href') || '';
      const isRealHref = href && !href.startsWith('#') && !href.toLowerCase().startsWith('javascript:');
      if (isRealHref) return false;
    }
    if (el.closest(NAV_CONTAINERS)) return true;
    if (el.matches('[role=menuitem], [role=tab]')) return true;
    if (el.tagName.toLowerCase() === 'router-link') return true;
    if (typeof el.className === 'string' && /router-link/i.test(el.className)) return true;
    return false;
  };

  // W36: ปุ่ม submit ของ <form> ตาม HTML (input/button[type=submit] หรือ <button> ไม่มี type) — แค่สัญญาณให้
  // classify_button_tier()/button_core_priority() ไม่เกี่ยวกับ is_crawl_safe() (submit ยังโดน BLOCKED)
  const isFormSubmit = (el) => {
    const form = el.closest('form');
    if (!form) return false;
    const tag = el.tagName.toLowerCase();
    const type = (el.getAttribute('type') || '').toLowerCase();
    if (tag === 'input') return type === 'submit';
    if (tag === 'button') return type === 'submit' || !el.hasAttribute('type');
    return false;
  };

  const computeXPath = (el) => {
    if (el.id) return `//*[@id="${el.id}"]`;
    let path = '';
    let node = el;
    while (node && node.nodeType === 1 && node !== document.documentElement) {
      let idx = 1;
      let sib = node.previousElementSibling;
      while (sib) { if (sib.tagName === node.tagName) idx++; sib = sib.previousElementSibling; }
      path = `/${node.tagName.toLowerCase()}[${idx}]` + path;
      node = node.parentElement;
    }
    return '/html' + path;
  };

  // W18: ความหมายปุ่ม icon-only — <svg><title> > data-icon > class ของ icon library (fa-/icon-/lucide-/feather-/
  // bi-/glyphicon-) > ligature ของ material-icons > aria-label ของ ancestor; '' ถ้าเดาไม่ได้
  const inferIconHint = (el) => {
    const svgTitle = el.querySelector('svg > title');
    if (svgTitle && svgTitle.textContent && svgTitle.textContent.trim()) {
      return svgTitle.textContent.trim().toLowerCase();
    }
    const dataIconEl = el.hasAttribute('data-icon') ? el : el.querySelector('[data-icon]');
    if (dataIconEl) {
      const v = dataIconEl.getAttribute('data-icon');
      if (v && v.trim()) return v.trim().toLowerCase();
    }
    const nodes = [el, ...Array.from(el.querySelectorAll('*'))].slice(0, 15);
    for (const node of nodes) {
      const cls = typeof node.className === 'string' ? node.className : '';
      if (!cls) continue;
      const m = cls.match(/(?:^|\s)(?:fa|fas|far|fab|icon|lucide|feather|bi|glyphicon)[-_]([a-z0-9-]+)/i);
      if (m && m[1]) return m[1].replace(/[-_]/g, ' ').toLowerCase();
      if (/material-icons/i.test(cls) && node.textContent && node.textContent.trim() && node.textContent.trim().length < 30) {
        return node.textContent.trim().toLowerCase();
      }
    }
    const labeledAncestor = el.closest('[aria-label]');
    if (labeledAncestor && labeledAncestor !== el) {
      const v = labeledAncestor.getAttribute('aria-label');
      if (v && v.trim()) return v.trim().toLowerCase();
    }
    return '';
  };

  // ใช้ร่วมทั้งปุ่มระดับหน้าและปุ่มภายใน UI pattern
  const describeButton = (el) => ({
    text: (el.innerText || el.value || '').trim().slice(0, 100),
    has_icon: !!el.querySelector('svg, img, [class*="icon" i]'),
    aria_label: el.getAttribute('aria-label') || '',
    title: el.getAttribute('title') || '',
    role: el.getAttribute('role') || '',
    data_testid: el.getAttribute('data-testid') || el.getAttribute('data-test') || '',
    icon_hint: inferIconHint(el),
    is_nav_menu_item: isNavMenuItem(el),
    is_form_submit: isFormSubmit(el),
    selector: computeSelector(el),
    xpath: computeXPath(el),
  });

  // W24: [role=menuitem]/[role=tab]/router-link — เมนู/แท็บของ SPA ที่ไม่ใช่ a/button
  const BUTTON_SELECTOR =
    'a, button, [role=button], [role=link], [role=menuitem], [role=tab], router-link, ' +
    'input[type=submit], input[type=button], [onclick]';

  // ---- W18: UI pattern detection — ทำก่อน buttons/forms เพื่อข้าม element ที่อยู่ใน pattern แล้ว ----
  const MIN_PATTERN_REPEAT = 3;
  const CONSUMED_ATTR = 'data-ui-pattern-consumed';
  const consumedMarked = [];  // เก็บ element ที่แปะ attribute ไว้ชั่วคราว ไว้ล้างทิ้งท้ายสคริปต์

  const structuralSignature = (el) => {
    const classes = (typeof el.className === 'string' ? el.className : '')
      .trim().split(/\s+/).filter(Boolean).sort().join('.');
    const childTags = Array.from(el.children).map((c) => c.tagName.toLowerCase()).join(',');
    return `${el.tagName.toLowerCase()}|${classes}|${childTags}`;
  };

  const humanize = (s) => s.replace(/[-_]+/g, ' ').trim().replace(/\b\w/g, (c) => c.toUpperCase());

  const inferUiType = (representative, parent) => {
    const tag = representative.tagName.toLowerCase();
    if (tag === 'tr') return 'Table Row';
    if (tag === 'li') return 'List Item';
    try {
      if (window.getComputedStyle(parent).display.includes('grid')) return 'Grid Item';
    } catch (e) { /* ignore */ }
    const hasImage = !!representative.querySelector('img, [style*="background-image"]');
    const hasAction = !!representative.querySelector(BUTTON_SELECTOR);
    if (hasImage && hasAction) return 'Card';
    return 'List Item';
  };

  const PRICE_PATTERN = /(?:[$£€¥₹]\s?\d[\d,.]*|\d[\d,.]*\s?(?:USD|THB|บาท|EUR|GBP))/i;

  const inferComponents = (representative) => {
    const components = [];
    if (representative.querySelector('img, [style*="background-image"]')) components.push('Image');
    if (representative.querySelector('h1,h2,h3,h4,h5,h6,[class*="title" i],[class*="name" i]')) components.push('Title');
    if (PRICE_PATTERN.test(representative.innerText || '')) components.push('Price');
    if (representative.querySelector('[class*="rating" i],[class*="star" i],[aria-label*="rating" i]')) components.push('Rating');
    if (representative.querySelector('[class*="badge" i],[class*="tag" i],[class*="label" i]')) components.push('Badge');
    if (representative.querySelector('p')) components.push('Description');
    if (representative.querySelectorAll(BUTTON_SELECTOR).length > 0) components.push('Action Button');
    return components;
  };

  const inferPatternName = (representative, parent, uiType) => {
    // heading ก่อน container (เช่น <h2>Related Products</h2>) บอกชื่อ section ได้ตรงกว่า class name
    let sib = parent.previousElementSibling;
    for (let i = 0; sib && i < 3; i++, sib = sib.previousElementSibling) {
      if (/^h[1-6]$/i.test(sib.tagName) && sib.innerText && sib.innerText.trim()) {
        return sib.innerText.trim().slice(0, 60);
      }
    }
    if (typeof representative.className === 'string' && representative.className.trim()) {
      const cls = representative.className.trim().split(/\s+/)[0];
      if (cls) return humanize(cls);
    }
    return `${uiType} Pattern`;
  };

  const uiPatterns = [];
  const candidateParents = [];
  document.querySelectorAll('body *').forEach((el) => {
    if (el.children.length >= MIN_PATTERN_REPEAT) candidateParents.push(el);
  });

  for (const parent of candidateParents) {
    if (uiPatterns.length >= 40) break;  // กันหน้าที่มี pattern ผิดปกติเยอะทำ payload บวม
    if (parent.closest(`[${CONSUMED_ATTR}]`)) continue;  // อยู่ใน pattern ที่เจอไปแล้ว ไม่ตรวจซ้ำ

    const sigMap = new Map();
    for (const child of parent.children) {
      const tag = child.tagName.toLowerCase();
      if (tag === 'script' || tag === 'style' || !isVisible(child)) continue;
      const sig = structuralSignature(child);
      if (!sigMap.has(sig)) sigMap.set(sig, []);
      sigMap.get(sig).push(child);
    }

    for (const [, elements] of sigMap) {
      if (elements.length < MIN_PATTERN_REPEAT) continue;

      const representative = elements[0];
      const uiType = inferUiType(representative, parent);
      uiPatterns.push({
        name: inferPatternName(representative, parent, uiType),
        ui_type: uiType,
        components: inferComponents(representative),
        buttons: Array.from(representative.querySelectorAll(BUTTON_SELECTOR))
          .filter((b) => isVisible(b) && !b.disabled)
          .slice(0, 20)
          .map(describeButton),
        selector: (() => {
          if (typeof representative.className !== 'string' || !representative.className.trim()) {
            return computeSelector(representative);
          }
          const classes = representative.className.trim().split(/\s+/).filter(Boolean);
          const classSelector = representative.tagName.toLowerCase() + '.' + classes.map(escapeCss).join('.');
          try {
            if (document.querySelectorAll(classSelector).length === elements.length) return classSelector;
          } catch (e) { /* ignore */ }
          return computeSelector(representative);
        })(),
        item_count: elements.length,
      });

      for (const e of elements) {
        e.setAttribute(CONSUMED_ATTR, '1');
        consumedMarked.push(e);
      }
    }
  }

  const isInConsumedPattern = (el) => !!el.closest(`[${CONSUMED_ATTR}]`);

  // ---- buttons ----
  const buttons = [];
  for (const el of document.querySelectorAll(BUTTON_SELECTOR)) {
    if (!isVisible(el) || el.disabled) continue;
    if (isInConsumedPattern(el)) continue;  // เก็บไปแล้วในฐานะปุ่มของ UI pattern ด้านบน
    buttons.push(describeButton(el));
    if (buttons.length >= 300) break;  // กันหน้าที่มี element เยอะผิดปกติทำ payload บวม
  }

  // ---- forms ----
  const forms = [];
  const SKIP_INPUT_TYPES = new Set(['submit', 'button', 'hidden', 'checkbox', 'radio', 'file', 'image', 'reset']);
  for (const el of document.querySelectorAll('input, select, textarea')) {
    if (!isVisible(el)) continue;
    if (isInConsumedPattern(el)) continue;  // ช่องกรอกต่อ instance (เช่น quantity ต่อสินค้า) ไม่เก็บซ้ำ
    const inputType = (el.getAttribute('type') || 'text').toLowerCase();
    if (el.tagName.toLowerCase() === 'input' && SKIP_INPUT_TYPES.has(inputType)) continue;
    let label = '';
    if (el.id) {
      const labelEl = document.querySelector(`label[for="${escapeCss(el.id)}"]`);
      if (labelEl) label = labelEl.innerText.trim();
    }
    if (!label) {
      const closestLabel = el.closest('label');
      if (closestLabel) label = closestLabel.innerText.trim();
    }
    if (!label) label = el.getAttribute('aria-label') || '';
    forms.push({
      field_name: el.getAttribute('name') || '',
      label,
      placeholder: el.getAttribute('placeholder') || '',
      required: !!(el.required || el.getAttribute('aria-required') === 'true'),
      input_type: el.tagName.toLowerCase() === 'select' ? 'select' : (el.tagName.toLowerCase() === 'textarea' ? 'textarea' : inputType),
      validation: el.getAttribute('pattern') || (el.maxLength > 0 ? `maxlength=${el.maxLength}` : ''),
      selector: computeSelector(el),
    });
    if (forms.length >= 200) break;
  }

  // ---- tables ----
  const tables = [];
  for (const table of document.querySelectorAll('table')) {
    if (!isVisible(table)) continue;
    const headerCells = table.querySelectorAll('thead th, thead td');
    const fallbackCells = table.querySelectorAll('tr:first-child th, tr:first-child td');
    const columns = Array.from(headerCells.length ? headerCells : fallbackCells)
      .map((c) => c.innerText.trim()).filter(Boolean);
    const sortable = !!table.querySelector('th[aria-sort], th.sortable, th[class*="sort" i]');
    const container = table.closest('div') || table.parentElement;
    const filterable = !!(container && container.querySelector(
      'input[type="search"], [placeholder*="filter" i], [aria-label*="filter" i]'
    ));
    const paginated = !!(container && container.querySelector(
      '[class*="pagination" i], [aria-label*="pagination" i], nav[aria-label*="page" i]'
    ));
    const rowActionsSet = new Set();
    table.querySelectorAll('tbody button, tbody a[role=button], tbody [role=button]').forEach((b) => {
      const t = (b.innerText || b.getAttribute('aria-label') || '').trim();
      if (t) rowActionsSet.add(t);
    });
    tables.push({ columns, sortable, filterable, paginated, row_actions: Array.from(rowActionsSet).slice(0, 30) });
  }

  // ---- nav links (ไว้ต่อคิว BFS ใน crawler.py — ไม่ใช่ส่วนหนึ่งของ PageInfo) ----
  // W25: สแกน a[href] ทั้งเอกสาร (เดิมแค่ใน NAV_CONTAINERS พลาดลิงก์ในเนื้อหาอย่าง "อ่านต่อ") — ยังกรองด้วย
  // is_safe_nav_link() ฝั่ง crawler.py เหมือนเดิม
  const navLinks = [];
  const seenHref = new Set();
  document.querySelectorAll('a[href]').forEach((a) => {
    const href = a.getAttribute('href');
    if (!href || href.startsWith('#') || href.toLowerCase().startsWith('javascript:')) return;
    if (seenHref.has(href)) return;
    const text = (a.innerText || a.getAttribute('aria-label') || '').trim();
    if (!text) return;
    seenHref.add(href);
    // W66[A]: selector ให้ crawler.py หา element กลับมาคำนวณ dom_locator.compute_locator_descriptor() —
    // ใช้ computeSelector() ตัวเดียวกับ buttons/forms (priority ต้องตรงกับ dom_locator.py)
    navLinks.push({ text, href, menu_path: [text], selector: computeSelector(a) });
  });

  // ---- breadcrumb ----
  let breadcrumb = [];
  const bcEl = document.querySelector('[aria-label="breadcrumb" i], .breadcrumb, nav[aria-label*="breadcrumb" i]');
  if (bcEl) {
    breadcrumb = Array.from(bcEl.querySelectorAll('a, span, li')).map((e) => e.innerText.trim()).filter(Boolean);
  }

  // ---- filters / search box / modals / tabs ----
  const filters = Array.from(document.querySelectorAll('[class*="filter" i], [aria-label*="filter" i]'))
    .map((e) => (e.innerText || e.getAttribute('aria-label') || '').trim())
    .filter(Boolean).slice(0, 20);
  const searchBox = !!document.querySelector('input[type="search"], input[placeholder*="search" i], [role="search"]');
  const modals = Array.from(document.querySelectorAll('[role="dialog"], .modal'))
    .map((e) => (e.getAttribute('aria-label') || e.getAttribute('title') || '').trim())
    .filter(Boolean);
  const tabs = Array.from(document.querySelectorAll('[role="tab"]'))
    .map((e) => (e.innerText || '').trim()).filter(Boolean);

  // ล้าง attribute ชั่วคราวของ UI pattern ไม่ทิ้งร่องรอยใน DOM จริง
  for (const e of consumedMarked) e.removeAttribute(CONSUMED_ATTR);

  return {
    buttons, forms, tables, ui_patterns: uiPatterns,
    nav_links: navLinks, breadcrumb, filters, search_box: searchBox, modals, tabs,
  };
}
"""


def _build_button(b: dict) -> ButtonInfo:
    """W36: ButtonInfo จาก dict ของ JS + tier (classify_button_tier เป็น Python จึงทำฝั่งนี้)"""
    button_info = ButtonInfo(**b)
    button_info.tier = classify_button_tier(button_info)
    return button_info


async def _extract_frame_data(frame: Frame) -> Optional[dict]:
    """W39: _EXTRACT_JS บน frame เดียว; คืน None ถ้าล้ม (cross-origin, detached) — ไม่ throw ให้ทั้งหน้าล้ม"""
    try:
        return await frame.evaluate(_EXTRACT_JS)
    except Exception:
        return None


async def extract_page(page: Page) -> tuple[PageInfo, list[dict]]:
    """คืน (PageInfo ที่ยังไม่มี name/description — crawler.py เติม, nav_links [{"text","href","menu_path","selector"}]
    สำหรับคิว BFS). W39: รวมปุ่ม/ฟอร์ม/ตาราง/pattern/nav link จากทุก child frame (ปุ่ม/ฟอร์มแปะ frame_index)"""
    data = await page.evaluate(_EXTRACT_JS)
    buttons_raw = list(data.get("buttons", []))
    forms_raw = list(data.get("forms", []))
    tables_raw = list(data.get("tables", []))
    ui_patterns_raw = list(data.get("ui_patterns", []))
    nav_links = list(data.get("nav_links", []))

    main_frame = page.main_frame
    for frame_index, frame in enumerate(page.frames):
        if frame == main_frame:
            continue
        frame_data = await _extract_frame_data(frame)
        if not frame_data:
            continue
        for b in frame_data.get("buttons", []):
            b["frame_index"] = frame_index
            buttons_raw.append(b)
        for f in frame_data.get("forms", []):
            f["frame_index"] = frame_index
            forms_raw.append(f)
        tables_raw.extend(frame_data.get("tables", []))
        for up in frame_data.get("ui_patterns", []):
            for b in up.get("buttons", []):
                b["frame_index"] = frame_index
            ui_patterns_raw.append(up)
        nav_links.extend(frame_data.get("nav_links", []))

    page_info = PageInfo(
        url=page.url,
        breadcrumb=data.get("breadcrumb", []),
        buttons=[_build_button(b) for b in buttons_raw],
        forms=[FormFieldInfo(**f) for f in forms_raw],
        tables=[TableInfo(**t) for t in tables_raw],
        ui_patterns=[
            UIPatternInfo(
                name=up.get("name", ""),
                ui_type=up.get("ui_type", ""),
                components=up.get("components", []),
                buttons=[_build_button(b) for b in up.get("buttons", [])],
                selector=up.get("selector", ""),
                item_count=int(up.get("item_count", 0)),
            )
            for up in ui_patterns_raw
        ],
        filters=data.get("filters", []),
        search_box=bool(data.get("search_box", False)),
        modals=data.get("modals", []),
        tabs=data.get("tabs", []),
    )
    return page_info, nav_links

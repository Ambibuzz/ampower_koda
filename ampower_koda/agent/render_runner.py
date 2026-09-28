"""The browser side of check_page, run as its own process (the worker never imports Playwright).

argv: route, steps (JSON), screenshot path, Page JSON path ("" for none), session file, report file.
Writes the report as JSON to the report file (the caller keeps only a bounded tail of stdout) and prints
``KODA_RENDER_REPORT_FILE``; without a report file it prints one ``KODA_RENDER_REPORT {json}`` line. Each step is reported by what it did while the page
settled, not only by the end state: the server calls it made, what changed on the page and in dialogs,
text shown only briefly, animations and layout shifts, and whether a table's columns are in order.
A report built this way let a model find 88-96% of seeded UI bugs, against 67% from the final snapshot
alone (compare-outputs/browser-lab/FINDINGS.md).
"""

MAX_STEPS = 20
DIFF_LINES = 25
REMOVED_SHOWN = 3  # removed lines listed per changed block before the rest are counted
RESPONSE_CHARS = 300  # of each server response body
WATCH_LOAD_MS = 2000
STILL_MS = 1000       # a step is over once nothing has moved for this long (a stall is shorter than this)
MOTION_WAIT_S = 4.5   # the longest a step waits for motion to stop
FROZEN_MS = 200       # a gap between frames this long is a freeze the user notices
# Calls Desk itself makes in the background; the page's own calls (frappe.client.* included) are reported.
CHECK_KEYS = {"expect", "layout"}  # a step with only these acts on nothing: it looks at the page as it is
DESK_BACKGROUND = ("frappe.desk.", "frappe.realtime.", "frappe.auth.", "frappe.core.", "frappe.sessions.",
                   "frappe.boot.", "frappe.www.", "frappe.email.", "frappe.social.", "frappe.utils.")

LAYOUT_JS = r"""() => {
  const pages = Array.from(document.querySelectorAll('.page-container')).filter(e => e.offsetParent !== null);
  const root = (pages[0] && (pages[0].querySelector('.layout-main') || pages[0])) || document.body;
  const out = [];
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_ELEMENT);
  let el;
  while ((el = walker.nextNode()) && out.length < 300) {
    const own = Array.from(el.childNodes).filter(n => n.nodeType === 3)
      .map(n => n.textContent.trim()).join(' ').replace(/\s+/g, ' ').trim();
    if (!own || own.length > 60) continue;
    const r = el.getBoundingClientRect();
    if (r.width < 1 || r.height < 1) continue;
    const st = getComputedStyle(el);
    if (st.visibility === 'hidden' || st.display === 'none' || Number(st.opacity) === 0) continue;
    out.push([own, Math.round(r.left + r.width / 2), Math.round(r.top + r.height / 2), Math.round(r.height),
              !!el.closest('svg')]);
  }
  return out;
}"""

# Content the user cannot see at this width: clipped by a container (unreachable) or scrollable to.
REACH_JS = r"""() => {
  const pages = [...document.querySelectorAll('.page-container')].filter(e => e.offsetParent !== null);
  const root = (pages[0] && (pages[0].querySelector('.layout-main') || pages[0])) || document.body;
  const W = innerWidth, clipped = new Set(), scroll = new Set(), page = new Set();
  for (const el of root.querySelectorAll('*')) {
    const own = [...el.childNodes].filter(n => n.nodeType === 3).map(n => n.textContent.trim()).join(' ').trim();
    if (!own || own.length > 40) continue;
    const r = el.getBoundingClientRect();
    if (r.width < 1 || r.height < 1) continue;
    let kind = '';
    for (let p = el.parentElement; p && p !== document.body; p = p.parentElement) {
      const pr = p.getBoundingClientRect();
      if (r.right <= pr.right + 1 && r.left >= pr.left - 1) continue;
      const ox = getComputedStyle(p).overflowX;
      if (/hidden|clip/.test(ox)) { kind = 'clipped'; break; }
      if (/auto|scroll/.test(ox) && p.scrollWidth > p.clientWidth) { kind = 'scroll'; break; }
    }
    if (kind === 'clipped') clipped.add(own);
    else if (kind === 'scroll') scroll.add(own);
    else if (r.right > W + 1) page.add(own);
  }
  return {clipped: [...clipped], scroll: [...scroll], page: [...page],
          pageScroll: document.scrollingElement.scrollWidth > W + 1};
}"""

TABLES_JS = r"""() => {
  const scopes = [...document.querySelectorAll('.page-container, .modal.show')].filter(e => e.offsetParent !== null);
  const out = [];
  for (const scope of scopes) for (const table of scope.querySelectorAll('table')) {
    if (!table.offsetParent) continue;
    const head = [...table.querySelectorAll('thead th')].map(th => th.innerText.trim());
    const rows = [...table.querySelectorAll('tbody tr')].filter(tr => tr.offsetParent)
      .map(tr => [...tr.children].map(td => td.innerText.trim().replace(/\s+/g, ' ')));
    out.push({head, rows});
  }
  return out;
}"""

# What an object expect asks of the page: where each text is shown, the boxes of a named region, and how
# many elements match. Texts are matched case-insensitively at the deepest element holding them.
FIND_JS = r"""(ask) => {
  const norm = s => (s || '').replace(/\s+/g, ' ').trim().toLowerCase();
  const own = el => [...el.childNodes].filter(n => n.nodeType === 3).map(n => n.textContent).join(' ');
  const tag = el => el.tagName.toLowerCase() + (el.classList[0] ? '.' + el.classList[0] : '');
  const shown = el => {
    if (!el.checkVisibility({opacityProperty: true, visibilityProperty: true})) return null;
    const r = el.getBoundingClientRect();
    return r.width >= 1 && r.height >= 1 && r.right > 0 && r.bottom > -scrollY ? r : null;
  };
  const scopes = [...document.querySelectorAll('.page-container, .modal.show, #alert-container')]
    .filter(e => e.checkVisibility());
  const all = scopes.flatMap(s => [s, ...s.querySelectorAll('*')])
    .filter(e => !e.closest('script, style, template, noscript'));
  const matches = (text, limit) => {
    const want = norm(text), out = [];
    for (const el of all) {
      if (!norm(el.textContent).includes(want) || [...el.children].some(c => norm(c.textContent).includes(want)))
        continue;
      const r = shown(el);
      if (r) out.push([(el.innerText || el.textContent).replace(/\s+/g, ' ').trim().slice(0, 60),
                       Math.round(r.left + r.width / 2), Math.round(r.top + r.height / 2)]);
      if (out.length >= limit) break;
    }
    return out;
  };
  // The region a heading or label names: its first ancestor that holds more than the label, by size (a
  // lane or panel under its title) or by text (a row of fields after it), or that is one of a repeated
  // set (an empty lane beside other lanes, not the whole board).
  const section = label => {
    const lr = label.getBoundingClientRect(), length = norm(label.textContent).length;
    for (let el = label.parentElement; el && el !== document.body; el = el.parentElement) {
      const repeated = el.className && [...el.parentElement.children]
        .some(s => s !== el && s.tagName === el.tagName && s.className === el.className);
      if (el.getBoundingClientRect().height >= 3 * lr.height || norm(el.textContent).length > length + 12
          || repeated) return el;
    }
    return label;
  };
  const boxes = (els, how) => els.map(el => [el, shown(el)]).filter(([, r]) => r).slice(0, 5)
    .map(([el, r]) => ({how, label: tag(el), box: [r.left, r.top, r.right, r.bottom].map(Math.round)}));
  const regions = name => {
    const want = norm(name);
    let out = [];
    try { out = boxes(all.filter(e => e.matches(name)), 'selector'); } catch (e) {}
    if (!out.length) out = boxes(all.filter(e => norm(e.getAttribute('aria-label')) === want
      || norm(e.getAttribute('role')) === want
      || (e.getAttribute('aria-labelledby') || '').split(/\s+/)
        .some(id => id && document.getElementById(id) && norm(document.getElementById(id).textContent) === want)),
      'aria label or role');
    if (!out.length) out = boxes(all.filter(e => norm(own(e)) === want || (norm(e.textContent) === want
      && ![...e.children].some(c => norm(c.textContent) === want))).map(section)
      .filter((el, i, list) => list.indexOf(el) === i), 'heading');
    return out;
  };
  const out = {};
  if (ask.texts) out.texts = Object.fromEntries(ask.texts.map(t => [t, matches(t, 8)]));
  if (ask.region) out.regions = regions(ask.region);
  if (ask.count) {
    let n = 0, how = 'selector';
    try { n = all.filter(e => e.matches(ask.count) && shown(e)).length; } catch (e) {}
    if (!n) { n = matches(ask.count, 200).length; how = n ? 'text' : 'selector or text'; }
    out.count = {how, n};
  }
  return out;
}"""


"""The browser side of check_page, run as its own process (the worker never imports Playwright).

argv: route, steps (JSON), screenshot path, Page JSON path ("" for none), session file, report file.
Writes the report as JSON to the report file (the caller keeps only a bounded tail of stdout) and prints
``KODA_RENDER_REPORT_FILE``; without a report file it prints one ``KODA_RENDER_REPORT {json}`` line. Each step is reported by what it did while the page
settled, not only by the end state: the server calls it made, what changed on the page and in dialogs,
text shown only briefly, animations and layout shifts, and whether a table's columns are in order.
A report built this way let a model find 88-96% of seeded UI bugs, against 67% from the final snapshot
alone (compare-outputs/browser-lab/FINDINGS.md).
"""

import difflib
import json
import os
import re
import sys
import time
import traceback
import urllib.parse

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

# Installed before any page script: the visible text every frame, running animations and layout shifts.
MOTION_JS = r"""
(() => {
  if (window.__motion) return;
  const shifts = [];
  const label = el => {
    if (!el) return '';
    const text = (el.getAttribute && el.getAttribute('aria-label')) || el.innerText || '';
    return el.tagName.toLowerCase() + (el.classList && el.classList[0] ? '.' + el.classList[0] : '')
      + (text.trim() ? ' "' + text.trim().replace(/\s+/g, ' ').slice(0, 30) + '"' : '');
  };
  try {
    new PerformanceObserver(list => {
      for (const e of list.getEntries()) {
        if (e.hadRecentInput) continue;
        const sources = (e.sources || []).map(s => [s, s.node && (s.node.nodeType === 1 ? s.node : s.node.parentElement)])
          .filter(([, n]) => n && n.closest && n.closest('.page-container, .modal, #alert-container'));
        if (!sources.length) continue;  // Desk's own sidebar and navbar
        shifts.push({t: e.startTime, value: e.value, sources: sources.slice(0, 3).map(([s, n]) => (
          {label: label(n), dy: Math.round(s.currentRect.y - s.previousRect.y),
           dx: Math.round(s.currentRect.x - s.previousRect.x)}))});
      }
    }).observe({type: 'layout-shift', buffered: true});
  } catch (e) {}
  const scope = '.page-container, .modal.show, #alert-container';
  const visibleLines = () => {
    const out = [];
    for (const root of document.querySelectorAll(scope)) {
      const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
      let n;
      while ((n = walker.nextNode())) {
        const text = n.textContent.trim();
        const el = n.parentElement;
        if (!text || !el || !el.checkVisibility({opacityProperty: true, visibilityProperty: true})) continue;
        const r = el.getBoundingClientRect();
        if (r.right <= 0 || r.left >= innerWidth || r.bottom <= 0 || r.top >= innerHeight) continue;
        out.push(text.replace(/\s+/g, ' ').slice(0, 80));
      }
    }
    return out;
  };
  let trace = null;
  const inScope = el => el && el.closest && el.closest(scope);
  window.__motion = {
    start() {
      const t0 = performance.now();
      trace = {t0, samples: [], anims: {}, from: shifts.length};
      let last = null;
      const tick = () => {
        if (!trace || trace.t0 !== t0) return;
        const t = Math.round(performance.now() - t0);
        const lines = visibleLines();
        const key = lines.join('\n');
        if (key !== last) { trace.samples.push({t, lines}); last = key; }
        for (const a of document.getAnimations()) {
          const el = a.effect && a.effect.target;
          if (!el || !el.isConnected || !inScope(el)) continue;
          const name = a.animationName || a.transitionProperty || 'animation';
          const k = label(el) + '|' + name;
          const r = el.getBoundingClientRect();
          const timing = a.effect.getTiming();
          const entry = trace.anims[k] || (trace.anims[k] = {target: label(el), name,
            duration: Math.round(Number(timing.duration) || 0), infinite: timing.iterations === Infinity,
            first: [Math.round(r.x), Math.round(r.y)], opacity: [getComputedStyle(el).opacity]});
          entry.last = [Math.round(r.x), Math.round(r.y)];
          entry.opacity[1] = getComputedStyle(el).opacity;
        }
        requestAnimationFrame(tick);
      };
      tick();
    },
    stop() {
      if (!trace) return null;
      const t0 = trace.t0;
      const still = document.getAnimations().filter(a => a.playState === 'running' && inScope(a.effect && a.effect.target))
        .map(a => ({target: label(a.effect.target), name: a.animationName || a.transitionProperty || 'animation',
                    infinite: a.effect.getTiming().iterations === Infinity}));
      const out = {samples: trace.samples, anims: Object.values(trace.anims), still,
                   shifts: shifts.slice(trace.from).map(s => Object.assign({}, s, {t: Math.round(s.t - t0)}))};
      trace = null;
      return out;
    },
  };
})();
"""

# Frame timing and element geometry each frame: freezes (with the script that was busy, where the browser
# reports it) and how every element that grew, shrank or moved got there.
SMOOTH_JS = r"""
(() => {
  if (window.__smooth) return;
  const loafs = [];
  const add = list => { for (const e of list.getEntries()) loafs.push({t: e.startTime, d: Math.round(e.duration),
    scripts: (e.scripts || []).slice(0, 3).map(s => ({fn: s.sourceFunctionName || '',
      src: (s.sourceURL || '').split('/').pop().split('?')[0], d: Math.round(s.duration)}))}); };
  try { new PerformanceObserver(add).observe({type: 'long-animation-frame', buffered: true}); }
  catch (e) { try { new PerformanceObserver(add).observe({type: 'longtask', buffered: true}); } catch (e2) {} }
  const scope = '.page-container, .modal, #alert-container';
  const label = el => {
    const text = (el.getAttribute && el.getAttribute('aria-label')) || el.innerText || '';
    return el.tagName.toLowerCase() + (el.classList && el.classList[0] ? '.' + el.classList[0] : '')
      + (text.trim() ? ' "' + text.trim().replace(/\s+/g, ' ').slice(0, 30) + '"' : '');
  };
  let run = null;
  const track = el => {
    if (run && el && el.closest && el.closest(scope) && !run.tracked.has(el) && run.tracked.size < 12)
      run.tracked.set(el, {label: label(el), points: []});
  };
  new MutationObserver(list => { for (const m of list) if (m.target.nodeType === 1) track(m.target); })
    .observe(document, {subtree: true, attributes: true, attributeFilter: ['style', 'class']});
  window.__smooth = {
    start() {
      const t0 = performance.now();
      run = {t0, tracked: new Map(), gaps: [], last: t0, changed: t0};
      const tick = () => {
        if (!run || run.t0 !== t0) return;
        const now = performance.now();
        if (now - run.last > FROZEN_MS) run.gaps.push([Math.round(run.last - t0), Math.round(now - run.last)]);
        run.last = now;
        for (const a of document.getAnimations()) track(a.effect && a.effect.target);
        for (const [el, entry] of run.tracked) {
          if (!el.isConnected) continue;
          const r = el.getBoundingClientRect();
          const prev = entry.points[entry.points.length - 1];
          // A hidden element measures 0x0 at 0,0: keep its place, only its size went to zero.
          const hidden = !r.width && !r.height && prev;
          const p = [Math.round(now - t0), hidden ? prev[1] : Math.round(r.x), hidden ? prev[2] : Math.round(r.y),
                     Math.round(r.width), Math.round(r.height)];
          if (!prev || prev.slice(1).join() !== p.slice(1).join()) { entry.points.push(p); run.changed = now; }
        }
        requestAnimationFrame(tick);
      };
      requestAnimationFrame(tick);
    },
    idle() { return run ? Math.round(performance.now() - run.changed) : 99999; },
    stop() {
      if (!run) return null;
      const t0 = run.t0;
      const out = {gaps: run.gaps, loafs: loafs.filter(l => l.t >= t0 - 50).map(l => Object.assign({}, l, {t: Math.round(l.t - t0)})),
                   tracks: [...run.tracked.values()].filter(e => e.points.length > 1)
                     .map(e => ({label: e.label, points: e.points.slice(0, 400)}))};
      run = null;
      return out;
    },
  };
})();
""".replace("FROZEN_MS", str(FROZEN_MS))

LEAF_CELL = re.compile(r'^\s*- (?:cell|columnheader|gridcell|rowheader)(?: "[^"]*")?$')
UNSELECTED_OPTION = re.compile(r'^\s*- option "[^"]*"$')
NUMBER = re.compile(r"^[^\d\-]{0,4}-?\d[\d,]*(?:\.\d+)?$")
DATE = re.compile(r"^(\d{2})-(\d{2})-(\d{4})$")


def compact(snapshot):
    """Drop what a row's own name already says (its plain cells) and a drop-down's unselected options."""
    return "\n".join(line for line in snapshot.splitlines()
                     if not LEAF_CELL.match(line) and not UNSELECTED_OPTION.match(line))


def diff_lines(before, after):
    old, new = compact(before).splitlines(), compact(after).splitlines()
    out = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, old, new, autojunk=False).get_opcodes():
        if tag != "equal":
            removed = ["  - " + line.strip() for line in old[i1:i2]]
            if len(removed) > REMOVED_SHOWN + 1:
                # What left was printed when it arrived; closing a panel need not re-print it.
                removed = removed[:REMOVED_SHOWN] + [f"  - … {len(removed) - REMOVED_SHOWN} more removed lines"]
            out += removed + ["  + " + line.strip() for line in new[j1:j2]]
    if len(out) > DIFF_LINES:
        out = out[:DIFF_LINES] + [f"  … {len(out) - DIFF_LINES} more changed lines"]
    return out


def quote(text, limit=50):
    return repr(text if len(text) <= limit else text[:limit] + "…")


def column_value(text):
    date = DATE.match(text)
    if date:
        return int(date.group(3) + date.group(2) + date.group(1))
    return float(re.sub(r"[^\d.\-]", "", text)) if NUMBER.match(text) else None


def table_readout(before, after):
    """For each table a step changed: its row count and the order of every numeric or date column."""
    lines = []
    for index, table in enumerate(after):
        old = before[index] if index < len(before) else None
        if old == table:
            continue
        rows = [r for r in table["rows"] if len(r) > 1]
        was = f" (was {len(old['rows'])})" if old and len(old["rows"]) != len(table["rows"]) else ""
        orders = []
        for column, name in enumerate(table["head"]):
            values = [column_value(r[column]) for r in rows if column < len(r)]
            if len(values) < 2 or any(v is None for v in values):
                continue
            order = ("ascending" if values == sorted(values) else
                     "descending" if values == sorted(values, reverse=True) else "unsorted")
            orders.append(f"{name or 'column ' + str(column + 1)} {order}")
        lines.append(f"  table {index + 1}: {len(table['rows'])} rows{was}" + (f"; {', '.join(orders)}" if orders else ""))
    return lines


def motion_lines(trace, seen, load=False):
    """What happened while the page settled, beyond its end state. ``seen`` holds endless animations
    already reported in this check, which are not repeated. During load, loading text is expected to
    come and go, and content moves while it first renders: only later shifts are reported."""
    if not trace:
        return []
    out, samples = [], trace.get("samples") or []
    if load:
        rendered = next((s["t"] for s in samples if len(s["lines"]) > 5), 0)
        trace = dict(trace, shifts=[s for s in trace.get("shifts") or [] if s["t"] > rendered + 300])
    if len(samples) > 2 and not load:
        first, last = set(samples[0]["lines"]), set(samples[-1]["lines"])
        brief, gone = {}, {}
        for i in range(1, len(samples) - 1):
            now, start, end = set(samples[i]["lines"]), samples[i]["t"], samples[i + 1]["t"]
            for line in now - first - last:
                brief.setdefault(line, [start, end])[1] = end
            for line in (first & last) - now:
                gone.setdefault(line, [start, end])[1] = end
        if brief:
            out.append("  shown only briefly (not before the step, not after it): "
                       + "; ".join(f"{quote(k)} {a}-{b}ms" for k, (a, b) in list(brief.items())[:6]))
        if gone:
            out.append("  briefly disappeared, then came back: "
                       + "; ".join(f"{quote(k)} {a}-{b}ms" for k, (a, b) in list(gone.items())[:6]))
    if len(samples) > 1:
        events = []
        for previous, current in zip(samples, samples[1:]):
            removed = [line for line in previous["lines"] if line not in current["lines"]]
            added = [line for line in current["lines"] if line not in previous["lines"]]
            parts = [f"-{quote(line, 30)}" for line in removed[:3]] + [f"+{quote(line, 30)}" for line in added[:3]]
            more = len(removed) + len(added) - len(parts)
            events.append(f"{current['t']}ms " + " ".join(parts) + (f" (+{more} more)" if more > 0 else ""))
        if len(events) > 8:
            events = events[:4] + [f"… {len(events) - 8} more changes"] + events[-4:]
        out.append("  text over time: " + " | ".join(events))
    repeat = lambda anim: anim.get("infinite") and anim["target"] + anim["name"] in seen
    for anim in [a for a in trace.get("anims") or [] if not repeat(a)][:6]:
        (x0, y0), (x1, y1) = anim["first"], anim.get("last") or anim["first"]
        move = ""
        if abs(x1 - x0) > 20 or abs(y1 - y0) > 20:
            move = f", moved from x={x0},y={y0} to x={x1},y={y1}"
            if abs(x1 - x0) > abs(y1 - y0):
                move += " (entering from the left)" if x1 > x0 else " (moving to the left)"
        opacity = anim.get("opacity") or []
        fade = f", opacity {opacity[0]}→{opacity[-1]}" if len(opacity) > 1 and opacity[0] != opacity[-1] else ""
        out.append(f"  animated: {anim['target']} {anim['name']} {anim['duration']}ms"
                   + (" repeating forever" if anim.get("infinite") else "") + move + fade)
    for anim in [a for a in trace.get("still") or [] if not repeat(a)][:4]:
        out.append(f"  STILL ANIMATING after the page settled: {anim['target']} {anim['name']}"
                   + (" (repeats forever)" if anim.get("infinite") else ""))
    for anim in (trace.get("still") or []) + (trace.get("anims") or []):
        if anim.get("infinite"):
            seen.add(anim["target"] + anim["name"])
    for shift in (trace.get("shifts") or [])[:4]:
        moved = "; ".join(f"{s['label']} moved {'down' if s['dy'] > 0 else 'up'} {abs(s['dy'])}px"
                          if abs(s["dy"]) >= abs(s["dx"]) else f"{s['label']} moved sideways {s['dx']}px"
                          for s in shift.get("sources") or [])
        out.append(f"  LAYOUT SHIFT {shift['value']:.3f} at {shift['t']}ms: {moved}")
    return out


def turning_points(values):
    """The values where the direction of change reverses, with the ends."""
    points, direction = [values[0]], 0
    for previous, value in zip(values, values[1:]):
        step = (value > previous) - (value < previous)
        if step and direction and step != direction:
            points.append(previous)
        direction = step or direction
    return points + [values[-1]]


def smooth_lines(data):
    """Freezes (no frame drawn, and which script was busy) and, per element that moved or resized:
    how far, over how long, where it stalled and whether it reversed."""
    if not data:
        return []
    out = []
    for start, gap in (data.get("gaps") or [])[:4]:
        busy = [l for l in data.get("loafs") or [] if l["t"] < start + gap + 50 and l["t"] + l["d"] > start - 50]
        who = "; ".join(f"{s['fn'] or 'anonymous'}{' in ' + s['src'] if s['src'] else ''} {s['d']}ms"
                        for l in busy for s in l["scripts"][:2])
        out.append(f"  PAGE FROZE: no frame drawn for {gap}ms from {start}ms after the action"
                   + (f" (busy in {who})" if who else ""))
    for track in (data.get("tracks") or [])[:6]:
        points = track["points"]
        # Height first: expanding and collapsing are the common motions, and a hidden element's width
        # dropping to 0 would otherwise hide what its height did.
        shown = [p for p in points if p[3] or p[4]] or points
        name, index, values = next(((n, i, [p[i] for p in (points if n == "height" else shown)])
                                    for n, i in (("height", 4), ("width", 3), ("y", 2), ("x", 1))
                                    if max(p[i] for p in points) - min(p[i] for p in points) >= 10), (None, 0, []))
        if not name:
            continue
        first, last = values[0], values[-1]
        parts = [f"{track['label']} {name} {first}→{last}px, moving from {points[0][0]}ms to {points[-1][0]}ms "
                 f"({points[-1][0] - points[0][0]}ms)"]
        for a, b in zip(points, points[1:]):
            gap, value = b[0] - a[0], a[index]
            if gap >= 250 and min(first, last) < value < max(first, last):
                share = round(100 * (value - first) / (last - first)) if last != first else 0
                parts.append(f"STALLED {gap}ms at {value}px ({share}% of the way) from {a[0]}ms")
        turns = turning_points(values)
        if len(turns) > 2:
            parts.append("REVERSED direction: " + "→".join(str(v) for v in turns) + "px")
        out.append("  motion: " + ", ".join(parts))
    return out


def reach_line(found, width):
    def sample(labels):
        return ", ".join(repr(label) for label in labels[:8]) + (" …" if len(labels) > 8 else "")
    notes = []
    if found["clipped"]:
        notes.append(f"CUT OFF at {width}px: {len(found['clipped'])} labels are cut by a container that hides its "
                     f"overflow, so the user can neither see nor scroll to them: {sample(found['clipped'])}")
    if found["scroll"]:
        notes.append(f"{len(found['scroll'])} labels are off to the side inside a scrollable area (reachable by "
                     f"scrolling it): {sample(found['scroll'])}")
    if found["page"]:
        notes.append(f"{len(found['page'])} labels are right of the {width}px viewport"
                     + (" (the page scrolls sideways)" if found["pageScroll"] else "") + f": {sample(found['page'])}")
    return "reach: " + ("; ".join(notes) if notes else f"everything visible fits the {width}px width")


def expects(step):
    """The step's expects: one text or object, or a list of them."""
    expect = step.get("expect")
    return [e for e in expect if e] if isinstance(expect, list) else [expect] if expect else []


def plain(text):
    return " ".join(str(text).split()).lower()


def pick(candidates, text):
    """The element an expect means by ``text``: one labelled exactly that before one that contains it."""
    return next((c for c in candidates if plain(c[0]) == plain(text)), candidates[0] if candidates else None)


def at(candidate):
    return f"x={candidate[1]},y={candidate[2]}"


def order_result(texts, axis, found):
    """Whether the texts' centres increase along the axis, in the given order. ``found`` maps each text
    to its visible matches as (label, x, y)."""
    index = 1 if axis == "x" else 2
    chosen = [(text, pick(found.get(text) or [], text)) for text in texts]
    missing = [text for text, candidate in chosen if candidate is None]
    if missing:
        return False, "not visible: " + ", ".join(quote(text) for text in missing)
    positions = ", ".join(f"{quote(text)} {axis}={candidate[index]}" for text, candidate in chosen)
    for (a, first), (b, second) in zip(chosen, chosen[1:]):
        if second[index] <= first[index]:
            return False, f"{positions}; {quote(b)} is not {'right of' if axis == 'x' else 'below'} {quote(a)}"
    return True, positions


def inside(candidate, box, slack=1):
    left, top, right, bottom = box
    return left - slack <= candidate[1] <= right + slack and top - slack <= candidate[2] <= bottom + slack


def region_name(region):
    left, top, right, bottom = region["box"]
    return f"{region['label']} (by {region['how']}) x={left}-{right} y={top}-{bottom}"


def within_result(text, name, candidates, regions):
    """Whether ``text`` is shown inside a region found by ``name``: some visible match has its centre in
    one of the region's boxes."""
    if not regions:
        return False, f"no visible region matches {quote(name)} (by selector, aria label, role or heading)"
    if not candidates:
        return False, f"{quote(text)} is not visible"
    for candidate in candidates:
        for region in regions:
            if inside(candidate, region["box"]):
                return True, f"{quote(text)} at {at(candidate)} inside {region_name(region)}"
    shown = "; ".join(at(c) for c in candidates[:3])
    return False, f"{quote(text)} is at {shown}, outside " + "; ".join(region_name(r) for r in regions[:3])


def absent_result(text, candidates):
    if not candidates:
        return True, ""
    return False, f"{quote(text)} is visible at " + "; ".join(f"{at(c)} ({quote(c[0], 30)})" for c in candidates[:3])


def count_result(counted, low, high):
    n, how = counted["n"], counted["how"]
    passed = n >= low and (high is None or n <= high)
    wanted = f"at least {low}" + (f" and at most {high}" if high is not None else "")
    return passed, f"{n} visible by {how}" + ("" if passed else f", wanted {wanted}")


def expect_result(expect, visible_text, find):
    """(passed, detail) for one expect. A text is looked for in the visible text as before; an object
    asks the page through ``find(ask)``, which returns FIND_JS's answer or None if the page gave none."""
    if isinstance(expect, dict) and "text" in expect and "within" not in expect:
        expect = expect["text"]
    if not isinstance(expect, dict):
        seen = str(expect).lower() in visible_text().lower()
        return seen, "" if seen else "not visible"
    if "order" in expect:
        texts = [str(t) for t in expect["order"]] if isinstance(expect["order"], list) else []
        axis = expect.get("axis", "x")
        if len(texts) < 2 or axis not in ("x", "y"):
            return False, "order needs a list of at least two texts and an axis of \"x\" or \"y\""
        found = find({"texts": texts})
        return order_result(texts, axis, found["texts"]) if found else (False, "the page could not be read")
    if "within" in expect:
        text, name = str(expect["text"]), str(expect["within"])
        found = find({"texts": [text], "region": name})
        return (within_result(text, name, found["texts"][text], found["regions"]) if found
                else (False, "the page could not be read"))
    if "absent" in expect:
        text = str(expect["absent"])
        found = find({"texts": [text]})
        return absent_result(text, found["texts"][text]) if found else (False, "the page could not be read")
    if "count" in expect:
        found = find({"count": str(expect["count"])})
        high = expect.get("max")
        return (count_result(found["count"], int(expect.get("min", 1)), None if high is None else int(high))
                if found else (False, "the page could not be read"))
    return False, "unknown expect; use a text or an object with order, text and within, absent, or count"


def expect_line(expect, passed, detail):
    shown = json.dumps(expect, ensure_ascii=False)
    if passed:
        return f"  expect {shown}: PASS" + (f" ({detail})" if detail else "")
    return f"  expect {shown}: FAIL ({detail})"


def body_summary(body, status):
    try:
        data = json.loads(body)
    except ValueError:
        return body[:200]
    if status >= 400 or "exc" in data or "exception" in data:
        text = str(data.get("exception") or data.get("exc") or "")
        messages = data.get("_server_messages")
        if messages:
            try:
                text += " " + " ".join(json.loads(m).get("message", "") for m in json.loads(messages))
            except (TypeError, ValueError, AttributeError):
                text += " " + str(messages)
        text = text.replace("\\n", "\n")
        found = re.findall(r"([A-Za-z_.]*(?:Error|Exception)\b[^\"\n]{0,200})", text)
        return (found[-1] if found else text)[:400]
    return json.dumps(data.get("message"), default=str, separators=(",", ":"))[:RESPONSE_CHARS]


def sign_in(context, page, base, login, session_file):
    """Reuse the last check's session: Frappe rate-limits one-time login keys per address."""
    sid = None
    try:
        sid = json.load(open(session_file)).get("sid")
    except (OSError, ValueError):
        pass
    if sid:
        context.add_cookies([{"name": "sid", "value": sid, "url": base}])
        probe = context.request.get(base + "/api/method/frappe.auth.get_logged_user")
        if not (probe.ok and probe.json().get("message") == "Administrator"):
            sid = None
            context.clear_cookies()
    if not sid:
        signed = page.goto(login, wait_until="domcontentloaded")
        if signed is not None and signed.status == 429:
            print("KODA_RENDER_UNAVAILABLE the site rate-limited the sign-in; check again in a minute.", flush=True)
            sys.exit(4)
        sid = next((c["value"] for c in context.cookies() if c["name"] == "sid"), None)
        if sid and sid != "Guest":
            with open(session_file, "w") as handle:
                json.dump({"sid": sid}, handle)
            os.chmod(session_file, 0o600)


class Page:
    """One browser tab on the site, with the page's server calls and console errors."""

    def __init__(self, context, page):
        self.context, self.page = context, page
        self.calls, self.console, self.inflight, self.seen_animations = [], [], 0, set()
        page.set_default_timeout(4000)  # a control that does not respond is reported, not waited on
        page.set_default_navigation_timeout(30000)  # the first load after registering a Page is slow
        page.on("console", self.on_console)
        page.on("pageerror", lambda error: self.remember("uncaught: " + str(error)))
        page.on("request", self.on_request)
        page.on("requestfinished", self.on_done)
        page.on("requestfailed", self.on_done)

    def remember(self, text):
        text = " ".join(str(text).split())[:300]
        if text and text not in self.console and len(self.console) < 30:
            self.console.append(text)

    def on_console(self, message):
        if message.type != "error":
            return
        source = str((message.location or {}).get("url") or "")
        if "socket.io" in source or "socket.io" in message.text:
            return  # realtime is not part of the page under test
        self.remember(message.text + (f" [{source.split('?')[0]}]" if source else ""))

    def on_request(self, request):
        if "/api/" not in request.url:
            return
        self.inflight += 1
        if "/api/method/" not in request.url:
            return
        args = {}
        try:
            args = {k: v[0] for k, v in urllib.parse.parse_qs(request.post_data or "").items()}
        except (TypeError, ValueError):
            pass
        self.calls.append({"request": request, "method": request.url.split("/api/method/", 1)[1].split("?")[0],
                           "args": args, "status": None, "body": ""})

    def on_done(self, request):
        if "/api/" not in request.url:
            return
        self.inflight = max(0, self.inflight - 1)
        for call in self.calls:
            if call["request"] is request and call["status"] is None:
                try:
                    response = request.response()
                    if response:
                        call["status"] = response.status
                        call["body"] = body_summary(response.text(), response.status)
                except Exception:
                    call["status"] = call["status"] or 0

    def settle(self, first=250):
        self.page.wait_for_timeout(first)
        deadline = time.time() + 6
        while self.inflight > 0 and time.time() < deadline:
            self.page.wait_for_timeout(100)
        self.page.wait_for_timeout(350)

    def snapshot(self):
        """The visible Desk page plus open dialogs and alerts, which live outside the page container."""
        parts = []
        for selector in (".page-container:visible .layout-main", ".page-container:visible", "body"):
            try:
                element = self.page.locator(selector).first
                if element.count():
                    parts.append(element.aria_snapshot())
                    break
            except Exception:
                continue
        for index, selector in enumerate((".modal.show", "#alert-container .desk-alert")):
            locator = self.page.locator(selector)
            for n in range(min(locator.count(), 4)):
                try:
                    if locator.nth(n).is_visible():
                        parts.append(("dialog:\n" if index == 0 else "alert:\n") + locator.nth(n).aria_snapshot())
                except Exception:
                    pass
        return "\n".join(parts)

    def visible_text(self):
        texts = []
        for selector in (".page-container:visible", ".modal.show", "#alert-container"):
            locator = self.page.locator(selector)
            for n in range(min(locator.count(), 4)):
                try:
                    if locator.nth(n).is_visible():
                        texts.append(locator.nth(n).inner_text())
                except Exception:
                    pass
        return " ".join(" ".join(texts).split())

    def evaluate(self, script, default=None, arg=None):
        try:
            return self.page.evaluate(script) if arg is None else self.page.evaluate(script, arg)
        except Exception:
            return default

    def target(self, text, kind):
        """A visible element by label, placeholder, role name or text, inside an open dialog first."""
        dialog = self.page.locator(".modal.show")
        scopes = [dialog.last] if dialog.count() and dialog.last.is_visible() else []
        scopes.append(self.page)
        for scope in scopes:
            finders = [lambda s=scope: s.get_by_label(text, exact=True), lambda s=scope: s.get_by_label(text),
                       lambda s=scope: s.get_by_placeholder(text)]
            if kind == "click":
                finders = [*(lambda s=scope, r=role, e=exact: s.get_by_role(r, name=text, exact=e)
                             for exact in (True, False) for role in ("button", "option", "link", "tab")),
                           *finders, lambda s=scope: s.get_by_text(text, exact=True),
                           lambda s=scope: s.get_by_text(text)]
            for finder in finders:
                try:
                    found = first_visible(finder())
                except Exception:
                    found = None
                if found is not None:
                    return found
        try:
            return first_visible(self.page.locator(text))
        except Exception:
            return None

    def run_step(self, step):
        """Run one step and let the page settle; return (outcome, what it acted on)."""
        page, acted = self.page, ""
        try:
            if "viewport" in step:
                width, height = (int(v) for v in step["viewport"])
                page.set_viewport_size({"width": width, "height": height})
                acted = f"viewport {width}x{height}"
                page.wait_for_timeout(400)
                # Desk opens its sidebar over the page at phone width; a phone user closes it first.
                if (page.locator(".body-sidebar-container.expanded").count()
                        and page.locator("div.overlay:visible").count()):
                    page.mouse.click(width - 6, height // 2)
                    acted += " (closed the Desk sidebar overlay)"
            elif "wait_ms" in step:
                page.wait_for_timeout(min(int(step["wait_ms"]), 5000))
            elif "press" in step:
                page.keyboard.press(str(step["press"]))
                acted = f"key {step['press']}"
            elif any(k in step for k in ("fill", "click", "select")):
                kind = next(k for k in ("fill", "click", "select") if k in step)
                element = self.target(str(step[kind]), "click" if kind == "click" else "fill")
                if element is None:
                    raise LookupError("no visible element matches " + json.dumps(step[kind]))
                acted = element.evaluate(
                    "e => e.tagName.toLowerCase() + ' \"' + (e.getAttribute('aria-label') || e.innerText || "
                    "e.value || e.placeholder || '').trim().replace(/\\s+/g, ' ').slice(0, 60) + '\"'")
                if kind == "click" and element.is_disabled():
                    return "FAILED: the element is disabled; nothing was clicked", acted
                if kind == "fill":
                    element.fill(str(step.get("value", "")))
                elif kind == "select":
                    try:
                        element.select_option(label=str(step.get("value", "")))
                    except Exception:
                        element.select_option(str(step.get("value", "")))
                else:
                    element.click()
            else:
                raise ValueError("unknown step " + json.dumps(step))
            self.settle()
            return "ok", acted
        except Exception as exc:
            message = " ".join(str(exc).split())
            blocker = re.search(r"<\w[^>]{0,160}>(?:[^<]{0,60}</\w+>)? (?:from <[^>]{0,80}> subtree )?"
                                r"intercepts pointer events", message)
            if blocker:
                message = "the click was blocked: " + blocker.group(0) + ". " + message
            return "FAILED: " + message[:320], acted

    def step_report(self, index, step):
        if step and set(step) <= CHECK_KEYS:
            return self.checks(step, "ok", [f"step {index} {json.dumps(step, ensure_ascii=False)}: ok"])
        before, before_tables = self.snapshot(), self.evaluate(TABLES_JS, [])
        calls, errors = len(self.calls), len(self.console)
        self.evaluate("window.__motion && window.__motion.start()")
        self.evaluate("window.__smooth && window.__smooth.start()")
        outcome, acted = self.run_step(step)
        # A slow or stalled animation is reported to its end, not cut off where the page usually settles.
        waited = time.time()
        while time.time() - waited < MOTION_WAIT_S and (self.evaluate("window.__smooth ? window.__smooth.idle() : 99999")
                                                       or 0) < STILL_MS:
            self.page.wait_for_timeout(100)
        smoothness = self.evaluate("window.__smooth ? window.__smooth.stop() : null")
        trace = self.evaluate("window.__motion ? window.__motion.stop() : null")
        if trace and smoothness:
            # Content pushed along by an element that is animating is not a layout shift of its own.
            spans = [(t["points"][0][0] - 50, t["points"][-1][0] + 50) for t in smoothness.get("tracks") or []]
            trace["shifts"] = [s for s in trace.get("shifts") or [] if not any(a <= s["t"] <= b for a, b in spans)]
        lines = [f"step {index} {json.dumps(step, ensure_ascii=False)}: {outcome}" + (f" — {acted}" if acted else "")]
        for call in self.calls[calls:]:
            if (call["status"] or 0) >= 400 or not call["method"].startswith(DESK_BACKGROUND):
                args = json.dumps(call["args"], ensure_ascii=False)[:300]
                lines.append(f"  server: {call['method'].rsplit('.', 1)[-1]} {args} -> {call['status']} {call['body']}")
        lines += ["  error: " + error for error in self.console[errors:]]
        changes = diff_lines(before, self.snapshot())
        lines += changes or ["  no visible change in the page, dialogs or alerts"]
        if changes:
            lines += table_readout(before_tables, self.evaluate(TABLES_JS, []))
        if "viewport" in step:
            # A resize moves things without changing the accessibility tree.
            found = self.evaluate(REACH_JS)
            if found:
                lines.append("  " + reach_line(found, page_width(self.page)))
        else:
            lines += motion_lines(trace, self.seen_animations) + smooth_lines(smoothness)
        return self.checks(step, outcome, lines)

    def checks(self, step, outcome, lines):
        """The step's expects and, if it asks, the label layout at this point (formatted by the caller)."""
        report = {"step": step, "outcome": outcome, "lines": lines}
        for expect in expects(step):
            try:
                passed, detail = expect_result(expect, self.visible_text,
                                               lambda ask: self.evaluate(FIND_JS, None, ask))
            except (KeyError, TypeError, ValueError) as exc:
                passed, detail = False, f"malformed expect: {type(exc).__name__}: {exc}"
            lines.append(expect_line(expect, passed, detail))
        if step.get("layout"):
            report["layout"] = self.evaluate(LAYOUT_JS, [])
            report["viewport"] = self.page.viewport_size
        return report


def first_visible(locator):
    for index in range(min(locator.count(), 12)):
        candidate = locator.nth(index)
        try:
            if candidate.is_visible():
                return candidate
        except Exception:
            pass
    return None


def page_width(page):
    return (page.viewport_size or {}).get("width") or 0


def main():
    route, steps, shot, page_json, session_file = (sys.argv[1], json.loads(sys.argv[2]), sys.argv[3], sys.argv[4],
                                                  sys.argv[5])
    report_file = sys.argv[6] if len(sys.argv) > 6 else ""
    report = {"route": route, "steps": []}
    # Before anything touches the site: a missing runtime must not leave a Page registered.
    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:
        print("KODA_RENDER_UNAVAILABLE the Python package playwright is not importable by " + sys.executable
              + " (" + type(exc).__name__ + ": " + " ".join(str(exc).split())[:160] + "). Install it with "
              "pip install 'ampower_koda[browser]' and then playwright install chromium, or point "
              "KODA_PLAYWRIGHT_PYTHONPATH at an existing install.", flush=True)
        sys.exit(4)
    try:
        os.chdir(os.environ["KODA_SITES_PATH"])
        import frappe
        frappe.init(site=os.environ["KODA_SITE"], sites_path=os.environ["KODA_SITES_PATH"])
        frappe.connect()
        if page_json:
            # What bench migrate does for this one Page, so a page just written can be opened.
            from frappe.modules.import_file import import_file_by_path
            try:
                import_file_by_path(page_json, force=True)
                frappe.db.commit()
                frappe.clear_cache()
                report["registered"] = page_json
            except Exception as exc:
                report["registration_error"] = " ".join(str(exc).split())[:300]
        from frappe.www.login import _generate_temporary_login_link
        login = _generate_temporary_login_link("Administrator", 2)
        base = (os.environ.get("KODA_RENDER_BASE_URL") or frappe.utils.get_url()).rstrip("/")
        frappe.destroy()
    except Exception:
        traceback.print_exc(limit=4)
        print("KODA_RUNNER_ERROR could not sign in to the site", flush=True)
        sys.exit(3)

    with sync_playwright() as playwright:
        try:
            browser = playwright.chromium.launch()
        except Exception as exc:
            print("KODA_RENDER_UNAVAILABLE Playwright is installed but could not launch Chromium (" + " ".join(
                str(exc).split())[:160] + "). Install the browser with playwright install chromium (and "
                "playwright install-deps chromium on Linux), or set PLAYWRIGHT_BROWSERS_PATH to where it is.",
                flush=True)
            sys.exit(4)
        try:
            context = browser.new_context(viewport={"width": 1440, "height": 1000})
            context.add_init_script(MOTION_JS)
            context.add_init_script(SMOOTH_JS)
            tab = Page(context, context.new_page())
            try:
                sign_in(context, tab.page, base, login, session_file)
            except SystemExit:
                raise
            except Exception as exc:
                print("KODA_RENDER_UNAVAILABLE the site is not reachable at " + base + " (is the web server "
                      "running?): " + " ".join(str(exc).split())[:200], flush=True)
                sys.exit(4)
            stage = "loading the page"
            try:
                response = tab.page.goto(base + route, wait_until="load")
                report["status"] = response.status if response else None
                # Watch the first seconds: late inserts, layout shifts and animations that never stop.
                tab.evaluate("window.__motion && window.__motion.start()")
                tab.settle(WATCH_LOAD_MS)
                report["load"] = motion_lines(tab.evaluate("window.__motion ? window.__motion.stop() : null"),
                                              tab.seen_animations, load=True)
                for index, step in enumerate(steps[:int(os.environ.get("KODA_RENDER_MAX_STEPS", MAX_STEPS))], 1):
                    stage = f"step {index}"
                    report["steps"].append(tab.step_report(index, step))
                stage = "reading the final page"
                report["url"] = tab.page.url
                text = tab.page.locator("body").inner_text()
                report["not_found"] = bool(re.search(r"\bPage\s+\S[^\n]{0,80}\s+not found", text, re.I))
                report["login_page"] = "/login" in tab.page.url
                report["snapshot"] = compact(tab.snapshot())
                report["layout"] = tab.evaluate(LAYOUT_JS, [])
                report["viewport"] = tab.page.viewport_size
                found = tab.evaluate(REACH_JS)
                report["reach"] = reach_line(found, page_width(tab.page)) if found else ""
                stage = "taking the screenshot"
                tab.page.screenshot(path=shot, full_page=True)
                report["screenshot"] = shot
            except Exception as exc:
                # Whatever was seen before the failure still goes back.
                report["runner_error"] = f"the check stopped while {stage}: " + " ".join(
                    f"{type(exc).__name__}: {exc}".split())[:300]
            report["console"] = tab.console
            report["api_failures"] = [f"{c['status']} {c['method']}: {c['body']}" for c in tab.calls
                                      if (c["status"] or 0) >= 400][:15]
        finally:
            browser.close()
    if report_file:
        with open(report_file, "w", encoding="utf-8") as handle:
            json.dump(report, handle, default=str)
        print("KODA_RENDER_REPORT_FILE", flush=True)
    else:
        print("KODA_RENDER_REPORT " + json.dumps(report, default=str), flush=True)


if __name__ == "__main__":
    main()

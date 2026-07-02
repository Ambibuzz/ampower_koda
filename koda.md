# KODA.md

Frappe-specific behavioral skills for **AmPower Koda**. Koda works on any Frappe app task — bug fixes, new features, improvements, DocTypes, Script Reports, Pages, hooks, APIs, client scripts, and bench workflows.

**Tradeoff:** These skills bias toward caution over speed. For trivial one-line fixes, use judgment.

---

## Task types Koda handles

| Type | Examples | Typical files |
|------|----------|---------------|
| **Bug Fix** | Wrong validation, broken API, JS error, bad query | `.py`, `.js`, `hooks.py` — often one file |
| **Feature Request** | New DocType, Report, Page, workflow, integration | `.json`, `.py`, `.js`, `hooks.py` |
| **Improvement** | UX tweak, performance, refactor within scope | `.js`, `.py`, `.css`, `.html` |

Match exploration and edits to the **request type** — don't default to DocType JSON + migrate for every task.

---

## 1. Think Before Coding

**Don't assume. Surface Frappe tradeoffs for THIS task.**

Before implementing:
- Identify the task type (bug fix / feature / improvement) and which artifact is involved (DocType, Report, Page, hook, API, client script).
- For **bug fixes**: locate the failing path first (error message → controller → client → hook) before changing anything.
- For **new artifacts**: find a similar Report, Page, or DocType in the app and copy its pattern.
- If multiple valid approaches exist (client script vs server validation vs Property Setter), present them — don't pick silently.
- NEVER guess `fieldname`, report columns, API method paths, or hook keys — read the actual files.

---

## 2. Simplicity First

**Minimum Frappe code for the requested task. Nothing speculative.**

- Bug fix: smallest change that fixes the root cause — no refactors.
- New feature: only the files the feature needs (a Report may be `.json` + `.py` + `.js`; a bug may be one line in `.py`).
- No extra DocTypes, reports, or APIs beyond what was asked.
- No custom SQL when `frappe.get_all` / `frappe.db.get_value` is enough.
- Copy how the same app already solves a similar problem.

---

## 3. Surgical Changes

**Touch only files the task requires.**

- Bug fix in a controller → don't rewrite the client script unless the bug is there.
- New Report → don't modify unrelated DocTypes.
- Match existing style: 4-space Python, tab JSON, existing `frappe.call` patterns.
- Remove only imports your change made unused.

---

## 4. Goal-Driven Execution

**Define success criteria for the specific task.**

| Task | Verify |
|------|--------|
| Bug fix | Original failure path resolved; `validate_code` passes; no regression |
| New DocType | `.json` valid, migrate succeeds, form loads |
| Script Report | Report JSON + `execute`/`get_columns` in `.py`; report runs in desk |
| Custom Page | Page loads, `frappe.pages` or route works, data fetches |
| New API | `@frappe.whitelist()` + `frappe.call` path match |
| Hook / event | Event fires at correct lifecycle; no duplicate hook keys |
| UI improvement | Client behavior matches request; build if assets changed |

In review: check **user request scope** and blocking bugs — not an exhaustive JSON metadata audit.

---

## Absolute Rules (Frappe)

1. NEVER guess field names, report config, or API paths — read `.json`, `read_doctype_schema`, or `read_file` first.
2. NEVER fabricate paths, method names, or module names not seen in the codebase.
3. NEVER create a new standard artifact without reading an existing one of the **same type** in the app.
4. NEVER guess file contents — `read_file` before every edit.
5. After every `.py`/`.js` edit: `validate_code`, then `read_file` the edited region.
6. NEVER insert code inside JS template literals or wrong scope.
7. `frappe.call` method path must match `@frappe.whitelist()` location when client↔server is involved.

---

## Frappe Exploration Skill

Explore based on **what the user asked for**:

**Any task**
1. `find_files()` — map `doctype/`, `report/`, `page/`, `public/`, `patches/`, `hooks.py`
2. `read_file("hooks.py")` — events, schedulers, assets, overrides

**Bug fix**
3. `search_code` for error text, function names, fieldnames from the description
4. Trace UI → `frappe.call` → Python → DB query end-to-end
5. Read only files on the failure path plus one similar working example

**New DocType / field**
3. `read_doctype_schema` + read `.json`, `.py`, `.js` together

**New Report**
3. Read an existing Script Report in the app: `.json`, `.py`, `.js`
4. Note `ref_doctype`, `report_type`, `execute` / column definitions

**New Page / feature**
3. Read similar page or feature: `.json`, `.py`, `.js`, `.html`
4. Trace how data is loaded and rendered

**API / server logic**
3. `search_code("@frappe.whitelist")` and `frappe.call` in client files

---

## Frappe Editing Skill

**Read → Anchor → Edit → Validate → Verify**

| Artifact | Files | Notes |
|----------|-------|-------|
| Bug fix | Usually `.py` or `.js` | Fix root cause only; read surrounding logic |
| DocType | `.json`, `.py`, `.js` | Schema first if fields change; then migrate |
| Script Report | `.json`, `.py`, `.js` | Copy peer report structure; `execute` returns columns/data |
| Page | `.json`, `.py`, `.js`, `.html` | `frappe.pages` or desk page pattern |
| API only | `.py` (+ `.js` if wired) | `@frappe.whitelist()` decorator and path |
| Hook / event | `hooks.py`, controller `.py` | Append to dicts; no duplicate keys |
| Patch | `patches.txt`, patch `.py` | `[post_model_sync]` for schema/data |
| Assets | `public/js`, `public/css` | `bench build` after changes |

Prefer `replace_lines`. DocType JSON → migrate. JS/CSS/HTML → build.

---

## Frappe Review Skill

Review only what the user asked for.

- **Bug fix**: fix addresses the reported issue; no unrelated edits.
- **Report**: report files present and consistent; no DocType audit unless requested.
- **Feature**: scope matches request; syntax valid; wiring correct if client↔server.
- Fail only for blocking bugs — max 5 issues. No field-by-field JSON metadata audits.

---

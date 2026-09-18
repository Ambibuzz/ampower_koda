# Copyright (c) 2026, Ambibuzz Technologies LLP and contributors
# System prompts for the AI coding agent — Cursor-inspired, anti-hallucination, production-grade

import frappe

from ampower_koda.agent.errors import log_agent_error

# Mapping internal fieldname -> prompt_key option label
PROMPT_LABEL_MAP = {
    "system_prompt": "System Prompt",
    "understand_prompt": "Understand Prompt",
    "plan_prompt": "Plan Prompt",
    "implement_prompt": "Implement Prompt",
    "review_prompt": "Review Prompt",
    "follow_up_prompt": "Follow-up Prompt",
}

class SafeDict(dict):
    """dict subclass that returns '{key}' for missing keys instead of raising KeyError.
    Used with str.format_map() to safely render prompt templates with partial context."""
    def __missing__(self, key):
        return "{" + key + "}"


def render_prompt_safe(template: str, context: dict, default_template: str) -> str:
    """
    Render a prompt template using context variables.
    Uses SafeDict so missing placeholders are left as-is rather than raising KeyError.
    If the template cannot be rendered at all, falls back to default_template.
    Context keys not present in the template are appended as titled sections.
    """
    used = template
    try:
        result = template.format_map(SafeDict(context))
    except Exception as e:
        log_agent_error(
            "Prompt Render Error",
            f"{e}\nTemplate preview: {template[:200]}\n{frappe.get_traceback()}",
        )
        used = default_template
        try:
            result = default_template.format_map(SafeDict(context))
        except Exception:
            result = default_template

    for key, value in context.items():
        if f"{{{key}}}" not in used and isinstance(value, str) and len(value) < 500:
            result += f"\n\n## {key.upper().replace('_', ' ')}\n{value}"

    return result



def get_config_prompt(fieldname: str, default_template: str, request_name: str = None) -> str:
    """
    Return the prompt template for a given fieldname.
    If request_name is provided and the request has a matching prompt override
    in its Agent Prompt Configuration child table, that is returned instead.
    If use_default_prompts is enabled on the request, always returns the default.
    """
    prompt_label = PROMPT_LABEL_MAP.get(fieldname, fieldname)

    # Only request-level override
    if request_name:
        try:
            # Fetch parent doc first
            doc = frappe.get_doc("Agent Request", request_name)

            # If "Use Default Prompts" is enabled → skip overrides completely
            if doc.use_default_prompts:
                return default_template

            overrides = frappe.get_all(
                "Agent Prompt Configuration",
                filters={
                    "parent": request_name,
                    "prompt_key": ["in", [prompt_label, fieldname]]  
                },
                fields=["content"],
                order_by="idx asc"
            )

            if len(overrides) > 1:
                log_agent_error(
                    "Duplicate Prompt Configuration",
                    f"Multiple prompts found for {request_name}, prompt_key={prompt_label}. Using first (idx asc).",
                )

            if overrides:
                return overrides[0]["content"]

        except Exception as e:
            log_agent_error(
                "Prompt Fetch Failed",
                f"request_name={request_name}, fieldname={fieldname}\n{e}\n{frappe.get_traceback()}",
            )

    return default_template


def get_system_prompt(app_name: str, request_name: str = None) -> str:
    default = """You are an expert Frappe Framework developer. You work on ALL types of Frappe app tasks:
- **Bug fixes** — broken validation, APIs, client scripts, hooks, queries
- **Feature requests** — new DocTypes, Standard Reports, Pages, workflows, integrations
- **Improvements** — UX, performance, refactors within scope

You write clean, production-ready Frappe code. You NEVER guess — verify by reading actual artifacts first.

## ABSOLUTE RULES — VIOLATION CAUSES IMMEDIATE FAILURE
1. NEVER guess field names, report config, or API paths etc — read_file BEFORE editing.
2. NEVER fabricate fieldname, method paths, hook keys, or module names not seen in the codebase.
3. NEVER create a new standard artifact (DocType, Report, Page, Workspace) without reading an existing one of the SAME type in the app.
4. NEVER guess file contents — ALWAYS read_file BEFORE any edit.
5. On EDIT_FAILED, re-read the file (line numbers may have shifted).
6. NEVER assume a file exists — use find_files, list_directory, or read_file.
7. NEVER repeat a failed tool call with the same arguments.
8. Read at least 20 lines above and below before any edit.
9. NEVER insert code inside a JS template literal, Python string, or comment.
10. After EVERY .py/.js edit: validate_code, then read_file on the edited region.
11. When client↔server is involved: frappe.call method path must match @frappe.whitelist() location.

## CORE ENGINEERING PRINCIPLES

### 1. Think Before Coding
- Identify task type (bug fix / feature / improvement) and artifact (DocType, Report, Page, hook, API, client script).
- Bug fix: trace the failure path before changing code. Feature: find a similar artifact in the app first.
- If ambiguous, state interpretation — never silently pick one. NEVER fabricate names or paths.

### 2. Simplicity First
- Bug fix: smallest change at the root cause. Feature: only files the feature needs.
- No extra DocTypes, reports, or APIs beyond the request. Follow existing app patterns.

### 3. Surgical Changes
- Touch only files the task requires. Don't modify unrelated DocTypes when fixing a report or API bug.
- Match existing style. Remove only imports YOUR change made unused.

### 4. Goal-Driven Execution
- Define success for THIS task: bug fixed, report runs, page loads, field appears, API returns data.
- Verify with validate_code, bench migrate/build as needed, request-scoped review — not metadata audits.

## Smart Frappe Exploration (match task type)

**All tasks:** find_files() → map doctype/, report/, page/, public/, patches/ → read hooks.py

**Bug fix:** search_code for error text / function / fieldname → trace UI → frappe.call → Python → DB

**DocType / field change:** read_doctype_schema + .json + .py + .js together

**New Report:** read existing Script Report in app (.json + .py + .js); note ref_doctype, execute()

**New Page / feature:** read similar page (.json, .py, .js, .html); trace data loading

**API / hooks:** search_code for @frappe.whitelist, doc_events, frappe.call

## Target app: {app_name}
- App root: {app_name}/ (all tool paths relative to this root)
- Standard layout:
  - {app_name}/<module>/doctype/<name>/ — DocType: .json, .py, .js
  - {app_name}/<module>/report/<name>/ — Script Report: .json, .py, .js
  - {app_name}/<module>/page/<name>/ — Page: .json, .py, .js, .html
  - {app_name}/<module>/print_format/<name>/ — Print Format
  - {app_name}/hooks.py — doc_events, scheduler_events, fixtures, includes
  - {app_name}/patches/ — data/schema patches
  - {app_name}/public/ — JS/CSS assets
  - {app_name}/<module>/*.py — whitelisted APIs, utilities

## Frappe conventions
- Controllers: Document subclass; @frappe.whitelist() for APIs
- Data: frappe.get_doc, frappe.get_all, frappe.db.get_value — avoid raw SQL unless app already uses it
- DocType names in code: spaces ("Sales Order"), not sales_order
- Forms: frappe.ui.form.on("DocType", {{ refresh(frm) {{ ... }} }})
- Reports: execute(filters) returns columns/data; report JSON sets ref_doctype, report_type
- Pages: frappe.pages['page-name'] or desk Page pattern
- hooks.py: append carefully; no duplicate keys
- bench migrate after schema JSON; bench build after JS/CSS/public changes

## New Frappe artifacts — copy peers
Before creating DocType, Report, or Page JSON: read an existing artifact of the SAME type in the app and copy its structure. Modify only request-specific fields. At review, only blocking issues are flagged.

## Editing workflow
1. Read target file(s) — order depends on task (see below)
2. Anchor verification: unique 3-line block must match before replace_lines
3. replace_lines (preferred) or insert_lines; validate_code + read_file after
4. Re-read before second edit

**Read order by task:**
- Bug fix: failing file first, then trace related files
- DocType: .json → .py → .js
- Report: peer report, then new .json → .py → .js
- Page: peer page, then .json → .py → .js → .html
- Hook only: hooks.py + affected controller

## Edit tools:
- **replace_lines(path, start_line, end_line, new_content)** — PREFERRED
- **insert_lines(path, after_line, new_content)** — insert after line (0 = start)
- **validate_code(path)** — MANDATORY after .py/.js edits
- **write_file(path, content)** — NEW files only
"""
    template = get_config_prompt("system_prompt", default, request_name)
    context = {"app_name": app_name}

    return render_prompt_safe(template, context, default)


def get_understand_system_prompt(app_name: str, request_name: str = None) -> str:
    """Small, read-only system prompt for the retrieval-backed understand pass.

    The implementation system prompt contains editing, validation, artifact and
    Frappe workflow rules. None of those rules can be acted on in the read-only
    phase, yet they used to be re-sent on every discovery round. A request-level
    System Prompt override is still honoured; only the built-in default is
    phase-specific.
    """
    default = """You are a senior Frappe engineer performing a read-only code investigation.

Find the smallest amount of verified repository evidence needed to plan the user's
request accurately. Stay inside the requested feature or failure path. Do not survey
the whole app, propose unrelated improvements, or infer details you have not read.

## Investigation discipline
- Start from the automatically retrieved working set; do not remap the repository.
- Prefer search and outline over full-file reads.
- Read targeted line ranges. Read a whole file only when it is small and all of it matters.
- Never repeat a tool call with the same arguments. Never re-read an unchanged span.
- Batch independent lookups in one response when possible.
- Stop calling tools as soon as the change surface, current behavior, and verification path are clear.
- Cite evidence as `path:line` and distinguish verified facts from assumptions.

## Target app: {app_name}
Tool paths are relative to this app's root.
"""
    template = get_config_prompt("system_prompt", default, request_name)
    return render_prompt_safe(template, {"app_name": app_name}, default)



def get_understand_prompt(user_message: str, request_type: str, request_name: str = None) -> str:
    default = """## USER REQUEST
**Type:** {request_type}
**Description:**
{user_message}

## TASK

Gather only the evidence needed to create an implementation plan for this request.
Trace the relevant failure path or feature path, identify the files that must change,
and verify the local pattern the implementation should follow.

### Tool budget and stopping rule

- Use the automatically supplied working set before calling another discovery tool.
- Prefer `search`, `outline`, `definition`, `symbols`, and `refs` over `read`.
- Use `read` for the smallest useful span; avoid whole-file reads above 300 lines.
- Do not call the same tool with the same arguments twice or read overlapping spans
  unless the first result explicitly says it was incomplete.
- Batch independent calls. Aim for at most 8 discovery calls total.
- Stop when you can name the exact change surface and a concrete verification path.
- Inspect `hooks.py`, schemas, peer artifacts, client/server wiring, or migrations only
  when the user request makes them relevant.

### OUTPUT (maximum 1,000 words; no code blocks)

#### 1. Request interpretation and current behavior
State the scoped goal and the verified flow or root cause.

#### 2. Evidence
List at most 8 relevant files. For each, give exact `path:line-range`, the symbol or
section that matters, and one sentence explaining why. Use very short excerpts only
when a name cannot be communicated otherwise.

#### 3. Change surface
Name files to modify/create, the behavior each change owns, existing patterns to
follow, and any assumptions that could not be verified.

#### 4. Verification
List focused tests/checks and any migration, build, or cache step actually required.

Do not include a generic app overview, exhaustive inventory, unrelated conventions,
or repeated evidence. The planner needs precise conclusions, not a transcript."""
    template = get_config_prompt("understand_prompt", default, request_name)

    context = {
        "user_message": user_message,
        "request_type": request_type
    }

    return render_prompt_safe(template, context, default)


def get_plan_prompt(understanding_summary: str, user_message: str = "", request_name: str = None) -> str:
    user_section = f"## USER REQUEST\n{user_message}\n\n" if user_message else ""
    default = """{user_section}## CODEBASE ANALYSIS
{understanding_summary}

## YOUR TASK: Create an Implementation Plan

Create the complete review plan that a human approves before implementation.
Your response is constrained to the plan schema by the provider. Populate every
field; do not write code, pseudocode, or prose outside the structured response.

### Planning principles
1. **Tasks, not code** — each task needs a complete description and measurable
   acceptance criteria.
2. **State assumptions, don't block** — if scope, behaviour, field names or UX
   are unclear, make the most reasonable evidence-based choice and record it in
   `assumptions`.
3. **Ground in the analysis** — use exact paths and line ranges from the
   codebase analysis.
4. **Minimal scope** — only what this request needs. No drive-by refactors.
5. **Exploration is done** — no read/inspect/investigate tasks.

### Task fields
- Use sequential ids: `TODO 1`, `TODO 2`, and so on.
- `title` is a short action phrase; `goal` is one sentence of outcome.
- `description` is the complete implementation brief: what changes, where, why,
  and which existing pattern to follow. Do not include source code.
- `action` is `MODIFY` or `CREATE`. Every file in one task must use that action;
  split a task when existing and new files are both involved.
- `files` contains app-relative owned paths.
- `context_refs` contains only the code spans needed to implement the task, at
  most 6. Use `start: 0` and `end: 0` for a whole-file reference. Always provide
  `symbol` and `why`; either may be an empty string, but not both.
- `acceptance_criteria` contains observable outcomes the reviewer can check by
  reading the resulting source or by a mechanical check available in the sandbox
  (syntax validation, import, migration). The reviewer has no browser, server or
  test runner. Never require a live UI session, manual clicks, screenshots or
  production data; phrase UI or runtime behavior as the code path that produces it
  (for example "stale responses are ignored because the callback compares the
  captured request token to the current one").
- `depends_on` contains earlier task ids. Tasks sharing a file must declare an
  ordering dependency.

Typically use 2–8 tasks and never more than 12. For a new file, reference an
existing peer file that supplies the pattern. Resolve ambiguity in `assumptions`;
do not ask questions or add exploration tasks.
"""
    template = get_config_prompt("plan_prompt", default, request_name)

    context = {
        "user_section": user_section,
        "understanding_summary": understanding_summary,
        "user_message": user_message
    }

    return render_prompt_safe(template, context, default)


def get_implement_prompt(plan: str, understanding_summary: str, user_message: str, file_contents: str, request_name: str = None) -> str:
    default = """## USER REQUEST
{user_message}

## ACTIVE STRUCTURED TASK
{plan}

## CODEBASE FINDINGS
{understanding_summary}

## APPROVED WRITE PATHS
{file_contents}

Implement this task's goal and acceptance criteria. Read the referenced current
source before editing; context line numbers are hints and may have shifted.
Retrieve missing definitions or related schemas narrowly when needed.
Use edit_file with a unique exact anchor for existing code. For line-based edits,
read fresh line numbers first. Use write_file for new files. After EDIT_FAILED,
read the current file and correct the anchor; do not repeat a failed edit blindly.

Preserve existing behavior outside the requested change. Follow the app's Frappe
patterns and verify field names, whitelisted methods, hooks and client/server
contracts from source. Do not guess. Only write approved task files; report a
plan blocker if additional files or a different approach are required.

Check modified code with validate_code. Report what checks actually ran and any
runtime behavior still unverified. If the task is already satisfied, explain the
source evidence without forcing an edit. Finish with the JSON completion report
specified in the execution contract, including blockers and unverified behavior.
Never write progress notes into the app.
"""
    template = get_config_prompt("implement_prompt", default, request_name)

    context = {
        "plan": plan,
        "understanding_summary": understanding_summary,
        "user_message": user_message,
        "file_contents": file_contents
    }

    return render_prompt_safe(template, context, default)


def get_follow_up_implement_prompt(
    follow_up_message: str,
    original_plan: str,
    implementation_memory: str,
    prior_files: str,
    file_contents: str,
    request_name: str = None,
) -> str:
    """Patch-only prompt for follow-up runs — fix the bug, don't re-implement."""
    default = """## FOLLOW-UP BUG FIX (PATCH MODE — NOT A RE-IMPLEMENTATION)

The user tested the previous implementation and reported a specific issue below.
Your job is to **fix only that issue** with minimal, surgical edits.

### USER FOLLOW-UP ISSUE
{follow_up_message}

### ORIGINAL APPROVED PLAN (for context only — do NOT re-execute from scratch)
{original_plan}

### WHAT WAS ALREADY BUILT (memory from the last run)
{implementation_memory}

### FILES CHANGED IN THE PREVIOUS RUN
{prior_files}

### CURRENT TARGET FILE MANIFEST
{file_contents}

File bodies are not pre-loaded. Read the smallest relevant ranges once, then use
those tool results already present in context.

## PATCH RULES (CRITICAL)
1. **Fix the follow-up issue only** — do not rebuild features that already work.
2. **Do NOT re-implement the full plan** — read the listed target spans and patch what is broken.
3. **Minimal diff** — change only the lines needed; no drive-by refactors.
4. **Read once → edit → validate** — batch independent reads and do not repeat unchanged spans.
5. Run validate_code on every changed .py/.js file.
6. Do NOT create README, notes, .txt markers, or scratch files.
7. If the bug is in a file not listed, locate it narrowly and read it before editing — do not guess.

### FINAL STEP
End with the JSON completion report specified in the execution contract.
Describe only what you fixed for this follow-up, including unresolved behavior.
"""
    template = get_config_prompt("follow_up_prompt", default, request_name)
    context = {
        "follow_up_message": follow_up_message,
        "original_plan": original_plan,
        "implementation_memory": implementation_memory or "(no prior implementation summary recorded)",
        "prior_files": prior_files,
        "file_contents": file_contents,
    }
    return render_prompt_safe(template, context, default)


def get_review_prompt(edits_made: list[dict], user_message: str, request_name: str = None) -> str:
    """Prompt for the test stage focused on clean code + Frappe standards."""
    paths = [e.get("path", "") for e in edits_made if e.get("path")]
    paths_list = "\n".join(f"- {p}" for p in paths) if paths else "(no specific paths recorded)"

    default = """## USER REQUEST
{user_message_short}

## FILES TO REVIEW
{paths_list}

Review the supplied task contract against the actual changes and current source.
Inspect related definitions, schemas, callers and dependencies when needed to
verify behavior. Use focused reads and searches. Check business logic, permissions,
client/server contracts and Frappe conventions. Prioritize concrete defects and
missing acceptance criteria; avoid stylistic preferences unrelated to the task.

Mechanical checks are provided separately. They do not prove behavioral correctness.
Do not claim tests were executed unless a test result is supplied. You cannot run a
browser, server or test suite here: judge runtime and UI criteria from the source
path that produces the behavior, and state in the evidence that the runtime was not
exercised, rather than withholding a verdict. The execution contract below specifies
the required verdict and per-criterion evidence format.
"""
    template = get_config_prompt("review_prompt", default, request_name)
    context = {
        "user_message_short": user_message or "",
        "paths_list": paths_list,
    }
    return render_prompt_safe(template, context, default)

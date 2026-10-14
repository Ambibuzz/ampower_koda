# Ampower Koda: AI Coding Agent for Frappe

Koda builds changes to your Frappe apps from a plain-language request. It investigates the app, proposes a plan for you to approve, implements it, checks the result against your live site, and opens a pull request on GitHub. You approve every step that changes anything.


## Key Features

- **Tested on your site**: Koda checks the real database schema, runs the endpoints it wrote, runs tests and opens pages in a headless browser before it reports done.
- **Human in the loop**: you approve the plan, the bench commands and the push. Plans can be edited or revised with feedback before anything is written.
- **Independent review**: a separate review checks every acceptance criterion against the current source and the test results, and sends defects back for repair.
- **One continuous session**: investigation, planning, implementation and review share one cached conversation, which keeps context and cost down.
- **Koda IDE**: browse the changed files, edit them, view diffs, deploy and push without leaving Frappe.
- **Git native**: every request works on its own branch and ends in a pull request.
- **Live monitoring**: a status dashboard and live log show each step, with token, cache and cost figures.
- **Multiple providers**: OpenAI, Google Gemini, Anthropic Claude and OpenRouter.


## Requirements

- Frappe v15 or v16
- Python 3.10+
- Node.js 18.18+ and npm
- An API key for OpenAI, Google AI Studio (Gemini), Anthropic (Claude) or OpenRouter
- A GitHub personal access token with repo scope
- On Linux, sudo once, for the browser's system libraries


## Installation

```
cd frappe-bench
bench get-app ampower_koda                          # Python packages, Playwright included
cd apps/ampower_koda && npm ci && cd ../..          # ESLint, for JavaScript checks
bench --site your-site install-app ampower_koda     # also downloads Chromium (about 150 MB)
bench restart
```

On Linux, install the browser's system libraries once:

```
cd frappe-bench
sudo ./env/bin/playwright install-deps chromium
```

When a request starts, Koda names any check this bench cannot run yet. Those checks are reported as unavailable, and everything else works.

### Updating

```
cd frappe-bench/apps/ampower_koda
git pull
npm ci
cd ../..
bench setup requirements --python
bench migrate                                       # also downloads Chromium if it is missing
bench restart
```

### Using a separate Playwright install

Set `KODA_PLAYWRIGHT_PYTHONPATH` (its site-packages) and `PLAYWRIGHT_BROWSERS_PATH`, or `KODA_VERIFICATION_LAB` to a directory holding `python/` and `browsers/`. Koda then uses that install and skips the Chromium download.


## Setup

1. Open **Agent Settings** from the search bar.
2. Check **Enable AI Agent**.
3. Enter the API key for your provider.
4. Optionally set a default provider and model. With OpenAI, the model list is loaded live from your account, newest first.
5. Save.


## Creating a Request

1. Open the **Agent Request** list and click **New**.
2. Fill in:
   - **Title**: a short summary.
   - **Type**: Bug Fix, Reports & analytics, DocTypes & data model, Forms & desk UI, Server & business logic, Documents & output, Integrations, Platform & maintenance or ERPNext-flavored.
   - **Description**: what you need, in as much detail as you have.
   - **Provider and Model**.
   - **Target App Name**: the app's directory name (e.g. `ampower_task_manager`).
   - **GitHub Repo URL**, **GitHub Token** and **Base Branch**.
3. Save, then click **Start Agent**.

Your target app, repository, provider, model, base branch, branch prefix and Git identity are remembered and pre-filled on your next request. The GitHub token is not; enter it on each request or store it in Agent Settings as the encrypted default.


## How It Works

    1. Investigate  -- Koda reads the app and, where useful, looks at real records
    2. Plan         -- tasks, files and acceptance criteria         (Awaiting Approval)
    3. Implement    -- Koda writes the change on its own branch
    4. Verify       -- tests, endpoint calls and browser checks; failures are repaired
    5. Review       -- an independent review of every criterion; defects go back to repair
    6. Bench        -- you approve the bench commands               (Awaiting Bench Approval)
    7. Push         -- you approve the push                         (Awaiting Push Approval)
    8. Done         -- the pull request is open on GitHub

If repair cannot fix everything, the work is still delivered, with the open findings marked **UNRESOLVED** in the change summary.


## Approvals

**Plan.** Read the plan in the Agent Plan section. Edit it directly, or click **Revise Plan** and describe what to change. Then **Approve Plan** or **Reject Plan**. Each task names the files it changes and the criteria it must meet.

**Bench commands.** Koda lists the commands the change needs (migrate, build, clear-cache, restart) as an editable checklist. Untick or edit them, then approve.

**Push.** Test the change on your site first. Then **Approve Push** to commit, push and open the pull request, or **Reject Push**.

### Other actions

- **Submit Follow-up Fix**: describe what is still wrong after delivery; Koda continues on the same branch.
- **Resume Execution**: continue a failed or cancelled run from its last checkpoint.
- **Re-run Agent**: start the request over.
- **Execute Existing Plan**: implement the saved plan without investigating again.
- **Checkout Base Branch**: discard this request's uncommitted changes and return to the base branch. Changes that belong to another request or to you are never discarded.
- **Cancel**: stop a running request.

Each target app has one checkout, so a new request cannot start while another request's uncommitted work is checked out. Finish, push or check out the base branch on that request first.


## Verification

Koda proves a change by running it, not by reading it:

- **call_method** runs a function of the app, or a read-only database query, on the live site as Administrator.
- **run_tests** runs the tests in the target app's `.koda/tests` (Python `unittest` against the live site, and Node `*.test.cjs`) plus any integration commands in `.koda/verification.json`.
- **check_page** opens a page or form in a headless browser, drives its controls and reports errors, failed calls and what is shown.

While these run, database writes are rolled back; file writes, background jobs and email are discarded; and commits and child processes are refused. Calls to external services are not contained, so Koda does not run functions that post to them. Tests must call the real code and may patch only external services; a test that patches the module it tests is rejected. A test that passes is kept as a regression test.


## Cost and Usage

Each request records its tokens, cache reads and estimated cost. OpenRouter reports the billed cost directly. For other providers the cost is estimated from token usage at list prices; a model without a known price shows 0. When a request passes its spend limit, Koda logs it and keeps less evidence in context.


## Configuration

Per-app settings live in the target app's `.koda/config.toml`:

```toml
[context]
input_tokens = 150000   # working context budget for every phase

[rerank]
enabled = true          # rank code with a dedicated reranker; uses the OpenRouter key in Agent Settings
model = "cohere/rerank-v3.5"
```

Without an OpenRouter key, retrieval uses local ranking only.

### Custom prompts

To override Koda's instructions for one request, open the **Prompts** tab, uncheck **Use Default Prompts** and add rows for System, Understand, Plan, Implement, Review or Follow-up prompts. Any prompt you do not override uses the default.


## Koda IDE

Once a request has a branch or a diff, choose **Open Koda IDE** from the Actions menu to:

- browse the changed files, with added, modified and deleted badges;
- edit any file in a code editor with light and dark themes;
- switch between **Code** and **Diff** views;
- **Save** edits to the request's branch;
- **Deploy** the selected bench commands;
- **Push** the branch and open the pull request.


## Compliance

Tokens are stored encrypted, secret files are never shown to the model, and every change to your site or repository needs explicit approval.

<img width="854" height="480" alt="1775651869174" src="https://github.com/user-attachments/assets/f39d1226-ab20-49f8-8d2c-d279f463faf2" />


## Data Security

Koda sends the code it reads to the AI provider you choose; an agent cannot work on code it cannot read. Apply your usual controls for sensitive repositories.


## In-App Help

Search for **Koda Docs** in the Frappe search bar.


## License

MIT

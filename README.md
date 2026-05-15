AI Coding Agent for Frappe apps. Describe a bug fix, feature request, or improvement -- the agent reads the target app codebase, creates a plan, implements the changes, and opens a pull request on GitHub.


## Key Features

- **Live Monitoring**: Track agent progress in real-time with an auto-scrolling log console and visual status dashboard.
- **Human-in-the-loop**: Full control over every phase. Review and edit implementation plans and bench commands before they run.
- **Smart Exploration**: Powered by LangGraph, the agent performs deep codebase analysis to minimize hallucinations.
- **Git Native**: Automatically manages branches and creates clean Pull Requests on GitHub.
- **Multi-Model Support**: Choose between OpenAI, Google Gemini, or Anthropic Claude.


## Installation

    cd frappe-bench
    bench get-app ampower_koda
    bench --site your-site install-app ampower_koda
    bench migrate
    bench restart


## Requirements

- Frappe v14 or later
- Python 3.10+
- One of: OpenAI API key, Google AI Studio (Gemini) API key, or Anthropic (Claude) API key
- GitHub personal access token with repo scope
- Node.js v18+ (managed via nvm recommended)

Python dependencies (installed automatically):

- langchain, langchain-openai, langchain-google-genai, langchain-anthropic
- langgraph


## Setup

1. Open AI Agent Settings from the search bar.
2. Check Enable AI Agent.
3. Enter your API key for the provider you want to use (OpenAI, Gemini, or Claude).
4. Optionally set a default provider and model.
5. Save.


## How It Works

Each request goes through a multi-step pipeline with human approval gates. You can monitor the progress in real-time via the **Live Log Panel** and the visual **Status Dashboard** at the top of each request.

    1. Explore    -- agent reads the target app codebase
    2. Plan       -- agent creates a detailed implementation plan
    3. Review     -- you review and optionally edit the plan, then approve
    4. Implement  -- agent writes the code changes
    5. Verify     -- agent reviews its own implementation
    6. Bench      -- you see the exact bench commands and approve them
    7. Push       -- you review branch details and approve the push
    8. Done       -- pull request is created on GitHub


## Creating a Request

1. Go to AI Agent Request list (search for it or find it under the AI Agents module).
2. Click New.
3. Fill in:
   - Title -- short summary of what you need.
   - Type -- Bug Fix, Feature Request, or Improvement.
   - Description -- detailed explanation of the change.
   - Provider and Model -- pick your AI provider and model.
   - Target App Name -- the Frappe app directory name (e.g. ampower_task_manager).
   - GitHub Repo URL -- full URL of the repository.
   - GitHub Token -- personal access token.
   - Base Branch -- the branch to base changes on (e.g. develop or main).
4. Save the document.
5. Click Start Agent.

### Persistence
The app remembers your most-used configuration (Target App Name, GitHub Repo URL, AI Provider, Model, Base Branch, Branch Prefix, Git identity) and pre-fills it on every new request. The GitHub token is never persisted across requests for security reasons you must enter it on each new request (or store it in AI Agent Settings as the encrypted default).


## Approval Steps

Plan Approval:
After the agent explores the codebase and creates a plan, the status changes to Awaiting Approval. Read the plan in the Agent Plan section. You can edit it directly -- it is a Markdown editor. Click Approve Plan to proceed, or Reject Plan to cancel.

Bench Command Approval:
After implementation and code review, the agent computes which bench commands are needed (migrate, build, clear-cache, restart). The status changes to Awaiting Bench Approval. Each command is shown as an **editable checklist** -- you can uncheck commands you want to skip, or edit the command strings directly before approving them.

Push Approval:
After bench commands run and changes are committed to a branch, the status changes to Awaiting Push Approval. You can test the changes on your instance first. Review the branch and repository details, then click Approve Push to commit, push, and create the pull request.


## Buttons Available at All Times

These buttons appear in the Actions dropdown whenever the agent is not actively running:

- Checkout Base Branch -- switch back to the base branch and discard uncommitted changes.
- Run Bench Commands -- manually run migrate, build, clear-cache, and restart.
- Re-run Agent -- start over from scratch (explore, plan, implement).
- Execute Existing Plan -- skip exploration and use the existing plan.


## Configuration Fields

Each request carries its own configuration:

- Target App Name -- directory name of the Frappe app to modify.
- GitHub Repo URL -- https://github.com/org/repo format.
- GitHub Token -- personal access token (stored encrypted).
- Base Branch -- branch to base changes on.
- Branch Prefix -- prefix for agent-created branches (default: ai-agent/).
- Git User Name / Email -- identity for commits.

These fields (except the GitHub Token) are remembered across requests using user settings.


## Custom Prompt Configuration

For advanced use cases, you can override the agent's default logic on a per-request basis:

1. Open an **AI Agent Request**.
2. Go to the **Prompts** tab.
3. Uncheck **Use Default Prompts**.
4. In the **Custom Prompts** table, add one or more overrides:
   - **System Prompt**: Global identity and constraints.
   - **Understand Prompt**: How the agent explores the codebase.
   - **Plan Prompt**: How the agent synthesizes findings into a task list.
   - **Implement Prompt**: Instructions for the file-editing phase.
   - **Review Prompt**: Criteria for self-review and verification.

If a prompt type is not explicitly added to the table, the agent will fall back to its internal system default for that phase.


## Supported AI Providers

OpenAI:
- gpt-4o-mini, gpt-4o, gpt-5-mini, o3-mini

Gemini (Google AI Studio):
- gemini-2.0-flash, gemini-2.5-pro

Claude (Anthropic):
- claude-sonnet-4, claude-3-5-sonnet, claude-3-5-haiku


## Troubleshooting

- **No changes produced**: If the agent finishes but no diff is visible, try a more capable model (e.g., GPT-4o or Claude 3.5 Sonnet) or provide more detail in the description.
- **Build failures**: Ensure Node.js v18+ is installed on your server. Using `nvm` is highly recommended.
- **Migration errors**: Check the "Bench Commands" log for specific SQL or JSON schema errors.
- **Token Issues**: Ensure your GitHub PAT has the `repo` scope and is not expired.


## In-App Help

For more detailed guidance and tips, search for **AI Agents Help** in your Frappe search bar.


## License

MIT

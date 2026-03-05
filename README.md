# AmPower AI Agents

AI Coding Agent for Frappe apps. Users describe a bug fix or feature request, and the agent reads the target app's codebase, creates a branch, implements the changes, and opens a pull request.

## Installation

```bash
cd frappe-bench
bench get-app ampower_ai_agents
bench --site [sitename] install-app ampower_ai_agents
```

## Configuration

1. Go to **AI Agents Settings**
2. Enable the AI agent
3. Set the **Target App Name** (the Frappe app the agent will work on)
4. Provide your **OpenAI API Key**, **GitHub Repo URL**, and **GitHub Token**
5. Configure git identity and branch settings

## Usage

Navigate to the **AI Coding Agent** page from the AI Agents workspace. Submit a request describing a bug fix, feature, or improvement. The agent will process it in the background and create a PR when done.

## License

MIT

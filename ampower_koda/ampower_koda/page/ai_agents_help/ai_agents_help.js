frappe.pages['ai-agents-help'].on_page_load = function(wrapper) {
    var page = frappe.ui.make_app_page({
        parent: wrapper,
        title: 'AI Agents - How to Use',
        single_column: true
    });

    var content = `
    <div style="max-width:800px;margin:0 auto;padding:20px 10px;font-size:14px;line-height:1.7;">

        <h3>Overview</h3>
        <p>
            AmPower AI Agents is a coding assistant for Frappe apps.
            You describe a bug fix, feature, or improvement -- the agent reads the target app,
            creates a plan, implements the changes, and opens a pull request on GitHub.
        </p>

        <hr>

        <h3>Getting Started</h3>
        <ol>
            <li>Open <b>Agent Settings</b> from the search bar.</li>
            <li>Check <b>Enable AI Agent</b>.</li>
            <li>Enter your API key for OpenAI, Gemini, or Claude.</li>
            <li>Save.</li>
        </ol>

        <hr>

        <h3>Creating a Request</h3>
        <ol>
            <li>Go to <a href="/app/agent-request/new">Agent Request</a> and click New.</li>
            <li>Fill in the Title and Description with what you need.</li>
            <li>Select the AI Provider and Model.</li>
            <li>Enter the Target App Name (e.g. ampower_task_manager).</li>
            <li>Enter the GitHub Repo URL and Token.</li>
            <li>Set the Base Branch (e.g. develop or main).</li>
            <li>Save and click <b>Start Agent</b>.</li>
        </ol>

        <hr>

        <h3>Pipeline Steps</h3>
        <table style="width:100%;border-collapse:collapse;font-size:13px;">
            <tr style="background:var(--gray-100);">
                <th style="padding:8px;text-align:left;border-bottom:1px solid var(--border-color);">Step</th>
                <th style="padding:8px;text-align:left;border-bottom:1px solid var(--border-color);">What Happens</th>
                <th style="padding:8px;text-align:left;border-bottom:1px solid var(--border-color);">Your Action</th>
            </tr>
            <tr>
                <td style="padding:8px;border-bottom:1px solid var(--border-color);">Explore</td>
                <td style="padding:8px;border-bottom:1px solid var(--border-color);">Agent reads the target app codebase.</td>
                <td style="padding:8px;border-bottom:1px solid var(--border-color);">Wait.</td>
            </tr>
            <tr>
                <td style="padding:8px;border-bottom:1px solid var(--border-color);">Plan</td>
                <td style="padding:8px;border-bottom:1px solid var(--border-color);">Agent creates a detailed implementation plan.</td>
                <td style="padding:8px;border-bottom:1px solid var(--border-color);">Wait.</td>
            </tr>
            <tr>
                <td style="padding:8px;border-bottom:1px solid var(--border-color);">Review Plan</td>
                <td style="padding:8px;border-bottom:1px solid var(--border-color);">Plan is ready for your review.</td>
                <td style="padding:8px;border-bottom:1px solid var(--border-color);">Read the plan. Edit if needed. Click Approve Plan or Reject Plan.</td>
            </tr>
            <tr>
                <td style="padding:8px;border-bottom:1px solid var(--border-color);">Implement</td>
                <td style="padding:8px;border-bottom:1px solid var(--border-color);">Agent writes the code changes.</td>
                <td style="padding:8px;border-bottom:1px solid var(--border-color);">Wait.</td>
            </tr>
            <tr>
                <td style="padding:8px;border-bottom:1px solid var(--border-color);">Verify</td>
                <td style="padding:8px;border-bottom:1px solid var(--border-color);">Agent reviews its own code.</td>
                <td style="padding:8px;border-bottom:1px solid var(--border-color);">Wait.</td>
            </tr>
            <tr>
                <td style="padding:8px;border-bottom:1px solid var(--border-color);">Bench Commands</td>
                <td style="padding:8px;border-bottom:1px solid var(--border-color);">System lists the bench commands needed.</td>
                <td style="padding:8px;border-bottom:1px solid var(--border-color);">Review commands. Uncheck or edit any. Click Approve Bench Commands.</td>
            </tr>
            <tr>
                <td style="padding:8px;border-bottom:1px solid var(--border-color);">Push</td>
                <td style="padding:8px;border-bottom:1px solid var(--border-color);">Changes are committed to a branch. You can test on the instance.</td>
                <td style="padding:8px;border-bottom:1px solid var(--border-color);">Test the changes. Click Approve Push to create the PR.</td>
            </tr>
            <tr>
                <td style="padding:8px;">Done</td>
                <td style="padding:8px;">Pull request is created on GitHub.</td>
                <td style="padding:8px;">Review and merge the PR on GitHub.</td>
            </tr>
        </table>

        <hr>

        <h3>Buttons Always Available</h3>
        <p>These buttons appear in the Actions dropdown when the agent is not running:</p>
        <ul>
            <li><b>Checkout Base Branch</b> -- switch back to the base branch and discard changes.</li>
            <li><b>Run Bench Commands</b> -- manually run migrate, build, clear-cache, restart.</li>
            <li><b>Re-run Agent</b> -- start the agent over from scratch.</li>
            <li><b>Execute Existing Plan</b> -- skip exploration, use the current plan.</li>
        </ul>

        <hr>

        <h3>Supported Providers and Models</h3>
        <table style="width:100%;border-collapse:collapse;font-size:13px;">
            <tr style="background:var(--gray-100);">
                <th style="padding:8px;text-align:left;border-bottom:1px solid var(--border-color);">Provider</th>
                <th style="padding:8px;text-align:left;border-bottom:1px solid var(--border-color);">Models</th>
            </tr>
            <tr>
                <td style="padding:8px;border-bottom:1px solid var(--border-color);">OpenAI</td>
                <td style="padding:8px;border-bottom:1px solid var(--border-color);">gpt-4o-mini, gpt-5-mini, gpt-5.1-codex-mini, gpt-5-codex, gpt-5.1-codex, gpt-5.2-codex</td>
            </tr>
            <tr>
                <td style="padding:8px;border-bottom:1px solid var(--border-color);">Gemini</td>
                <td style="padding:8px;border-bottom:1px solid var(--border-color);">gemini-2.0-flash, gemini-2.5-pro</td>
            </tr>
            <tr>
                <td style="padding:8px;">Claude</td>
                <td style="padding:8px;">claude-sonnet-4, claude-3-5-sonnet, claude-3-5-haiku</td>
            </tr>
        </table>

        <hr>

        <h3>Tips</h3>
        <ul>
            <li>Use a more capable model (gpt-5.2-codex, claude-sonnet-4, gemini-2.5-pro) for complex tasks.</li>
            <li>Write detailed descriptions -- the more context, the better the result.</li>
            <li>Edit the plan before approving if you see something wrong.</li>
            <li>Test changes on the instance before approving the push.</li>
            <li>If the agent fails, check the error log and stage log on the request.</li>
            <li>User settings (app name, repo URL, provider) are saved automatically for new requests.</li>
        </ul>

        <hr>

        <h3>Troubleshooting</h3>
        <ul>
            <li><b>bench build fails</b> -- Make sure Node v18+ is installed via nvm.</li>
            <li><b>bench migrate fails</b> -- Check the bench log for details. May be a JSON format issue.</li>
            <li><b>Agent produces no changes</b> -- Check stage log and error log. Try a more capable model.</li>
            <li><b>New report or DocType not visible</b> -- Run bench migrate using the Run Bench Commands button.</li>
        </ul>

        <hr>

        <p style="color:var(--text-muted);font-size:12px;">
            AmPower AI Agents by Ambibuzz Technologies LLP.
            For issues, contact buzz@ambibuzz.com.
        </p>
    </div>
    `;

    $(page.body).html(content);
};

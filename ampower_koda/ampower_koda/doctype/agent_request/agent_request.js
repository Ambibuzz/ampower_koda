// Copyright (c) 2026, Ambibuzz Technologies LLP and contributors
// Agent Request: client script

var PROVIDER_MODELS = {
    'OpenAI': [
        { value: 'gpt-4o-mini', label: 'GPT-4o Mini: fast, cost-effective' },
        { value: 'gpt-5-mini', label: 'GPT-5 Mini: next-gen, efficient' },
        { value: 'gpt-5.1-codex-mini', label: 'GPT-5.1 Codex Mini: compact coding' },
        { value: 'gpt-5-codex', label: 'GPT-5 Codex: coding model' },
        { value: 'gpt-5.1-codex', label: 'GPT-5.1 Codex: coding model' },
        { value: 'gpt-5.2-codex', label: 'GPT-5.2 Codex: latest coding model' }
    ],
    'Gemini': [
        { value: 'gemini-2.0-flash', label: 'Gemini 2.0 Flash: fast, multimodal' },
        { value: 'gemini-2.5-pro', label: 'Gemini 2.5 Pro: most capable' }
    ],
    'Claude': [
        { value: 'claude-sonnet-4-20250514', label: 'Claude Sonnet 4: balanced' },
        { value: 'claude-3-5-sonnet-20241022', label: 'Claude 3.5 Sonnet: proven' },
        { value: 'claude-3-5-haiku-20241022', label: 'Claude 3.5 Haiku: fast, light' }
    ],
    'OpenRouter': [
        { value: 'deepseek/deepseek-v4-flash-0731', label: 'DeepSeek V4 Flash: fast, cheap' },
        { value: 'deepseek/deepseek-chat', label: 'DeepSeek Chat: general' },
        { value: 'anthropic/claude-sonnet-4', label: 'Claude Sonnet 4: via OpenRouter' },
        { value: 'openai/gpt-4o-mini', label: 'GPT-4o Mini: via OpenRouter' },
        { value: 'qwen/qwen-2.5-coder-32b-instruct', label: 'Qwen 2.5 Coder 32B: coding' }
    ]
};

var DEFAULT_MODELS = {
    'OpenAI': 'gpt-4o-mini',
    'Gemini': 'gemini-2.0-flash',
    'Claude': 'claude-sonnet-4-20250514',
    'OpenRouter': 'deepseek/deepseek-v4-flash-0731'
};

var REQUEST_TYPE_HELP = {
    'Bug Fix': '<b>Highlights:</b> Fix a specific failure (validation error, broken API, JS error, wrong query).',
    'Reports & analytics': '<b>Highlights:</b> Create or modify Script Reports, filters, columns, or analytics.',
    'DocTypes & data model': '<b>Highlights:</b> Create/modify DocTypes, fields, child tables, permissions, naming rules.',
    'Forms & desk UI': '<b>Highlights:</b> Client scripts, buttons, list view tweaks, field visibility, desk UX changes.',
    'Server & business logic': '<b>Highlights:</b> Validation logic, controller changes, whitelisted APIs, queries.',
    'Documents & output': '<b>Highlights:</b> Print Formats, PDF templates, document output formatting.',
    'Integrations': '<b>Highlights:</b> External APIs, webhooks, sync, connectors, third-party services.',
    'Platform & maintenance': '<b>Highlights:</b> Patches, hooks, scheduler tasks, performance or cleanup work.',
    'ERPNext-flavored': '<b>Highlights:</b> ERPNext flows (Sales, Stock, Accounts) in your custom app.'
};

var STATUS_FLOW = [
    'Queued', 'Understanding', 'Planning', 'Awaiting Approval',
    'Implementing', 'Reviewing', 'Awaiting Bench Approval', 'Building',
    'Awaiting Push Approval', 'Pushing', 'Completed'
];

var STATUS_META = {
    'Queued': { color: 'grey', label: 'Queued' },
    'Understanding': { color: 'blue', label: 'Exploring Codebase' },
    'Planning': { color: 'blue', label: 'Creating Plan' },
    'Awaiting Approval': { color: 'orange', label: 'Review Plan' },
    'Implementing': { color: 'yellow', label: 'Implementing Changes' },
    'Reviewing': { color: 'yellow', label: 'Testing Changes' },
    'Awaiting Bench Approval': { color: 'orange', label: 'Approve Bench Commands' },
    'Building': { color: 'purple', label: 'Running Bench Commands' },
    'Awaiting Push Approval': { color: 'orange', label: 'Approve Push to GitHub' },
    'Pushing': { color: 'purple', label: 'Creating Pull Request' },
    'Completed': { color: 'green', label: 'Completed' },
    'Failed': { color: 'red', label: 'Failed' },
    'Cancelled': { color: 'grey', label: 'Cancelled' }
};

frappe.ui.form.on('Agent Request', {
    refresh: function (frm) {
        render_status_dashboard(frm);
        setup_action_buttons(frm);
        setup_realtime_listeners(frm);
        setup_status_polling(frm);
        setup_live_log_panel(frm);
        set_model_options_for_provider(frm, false);
        update_request_type_help(frm);
        toggle_config_readonly(frm);
        style_form(frm);
    },

    request_type: function (frm) {
        update_request_type_help(frm);
    },

    ai_provider: function (frm) {
        // The one place a reset is correct: the user changed provider, so a
        // model belonging to the old one is genuinely no longer valid.
        set_model_options_for_provider(frm, true);
        frm.set_value('ai_model', default_model_for(frm, frm.doc.ai_provider));
    },

    before_save: function (frm) {
        if (frm.is_new()) {
            save_user_defaults(frm);
        }
    },

    onload: function (frm) {
        // Settings' default model is not in the static catalogue below; fetch it
        // so it is offered here and becomes the default for a new request.
        frappe.call({ method: 'ampower_koda.agent.api.get_model_defaults', callback: function (r) {
            frm._settings_defaults = r.message || {};
            if (frm.is_new()) {
                load_user_defaults(frm);
            }
            set_model_options_for_provider(frm, false);
        } });
        update_request_type_help(frm);
    }
});

function default_model_for(frm, provider) {
    provider = provider || 'OpenAI';
    var settings = frm._settings_defaults || {};
    if (settings.model && settings.provider === provider) return settings.model;
    return DEFAULT_MODELS[provider] || DEFAULT_MODELS['OpenAI'];
}

// ---------------------------------------------------------------------------
// Status Dashboard: visual progress indicator
// ---------------------------------------------------------------------------

function render_status_dashboard(frm) {
    var wrapper = frm.fields_dict.status_dashboard_html;
    if (!wrapper) return;

    var status = frm.doc.status || 'Queued';
    var meta = STATUS_META[status] || STATUS_META['Queued'];
    var is_running = ['Understanding', 'Planning', 'Implementing', 'Reviewing', 'Building', 'Pushing'].indexOf(status) !== -1;
    var is_terminal = ['Completed', 'Failed', 'Cancelled'].indexOf(status) !== -1;

    var steps_html = '';
    var current_idx = STATUS_FLOW.indexOf(status);
    if (current_idx === -1 && status === 'Failed') current_idx = -2;
    if (current_idx === -1 && status === 'Cancelled') current_idx = -2;

    var display_steps = [
        { key: 'Queued', short: 'Queued' },
        { key: 'Understanding', short: 'Explore' },
        { key: 'Planning', short: 'Plan' },
        { key: 'Awaiting Approval', short: 'Review' },
        { key: 'Implementing', short: 'Build' },
        { key: 'Reviewing', short: 'Test' },
        { key: 'Awaiting Bench Approval', short: 'Bench' },
        { key: 'Awaiting Push Approval', short: 'Push' },
        { key: 'Completed', short: 'Done' }
    ];

    for (var i = 0; i < display_steps.length; i++) {
        var step = display_steps[i];
        var step_idx = STATUS_FLOW.indexOf(step.key);
        var cls = 'agent-step';
        if (step_idx < current_idx) cls += ' agent-step-done';
        else if (step_idx === current_idx) cls += ' agent-step-active';

        steps_html += '<div class="' + cls + '">'
            + '<div class="agent-step-dot"></div>'
            + '<div class="agent-step-label">' + step.short + '</div>'
            + '</div>';
    }

    var banner_class = 'agent-banner-' + meta.color;
    var spinner = is_running
        ? '<span class="agent-spinner"></span>'
        : '';

    var pr_html = '';
    if (status === 'Completed' && frm.doc.pr_url) {
        pr_html = '<div class="agent-pr-link">'
            + '<a href="' + frappe.utils.escape_html(frm.doc.pr_url) + '" target="_blank" class="btn btn-sm btn-success">'
            + '<svg class="icon icon-sm"><use href="#icon-link-url"></use></svg> '
            + 'Open Pull Request #' + (frm.doc.pr_number || '') + '</a>'
            + '</div>';
    }

    var bench_info_html = '';
    if (status === 'Awaiting Bench Approval') {
        var cmds = [];
        try { cmds = JSON.parse(frm.doc.pending_bench_commands || '[]'); } catch (e) { }
        if (cmds.length) {
            var cmds_html = cmds.map(function (c, i) {
                return '<div style="display:flex;align-items:center;gap:8px;margin:4px 0;">'
                    + '<input type="checkbox" checked class="bench-cmd-check" data-idx="' + i + '" style="margin:0;">'
                    + '<input type="text" class="bench-cmd-input input-xs" data-idx="' + i + '" value="' + frappe.utils.escape_html(c) + '"'
                    + ' style="flex:1;font-family:monospace;font-size:12px;padding:4px 8px;border:1px solid var(--border-color);border-radius:3px;">'
                    + '</div>';
            }).join('');
            bench_info_html = '<div class="agent-push-info">'
                + '<strong>Bench commands to execute (uncheck to skip, edit to modify):</strong>'
                + '<div style="margin:8px 0;" id="bench-cmd-list">' + cmds_html + '</div>'
                + '<p style="margin:6px 0 0;font-size:12px;color:var(--text-muted);">Click <b>Approve Bench Commands</b> to run the selected commands.</p>'
                + '</div>';
        }
    }

    var plan_info_html = '';
    if (status === 'Awaiting Approval' && (frm.doc.agent_plan || '').trim()) {
        plan_info_html = '<div class="agent-push-info">'
            + '<strong>Review the plan todos</strong>'
            + '<p style="margin:6px 0 0;font-size:12px;color:var(--text-muted);">'
            + 'The plan describes <b>what</b> to build — not source code. '
            + 'Edit todos if needed, then click <b>Approve Plan</b> to start implementation.</p>'
            + '</div>';
    }

    var push_info_html = '';
    if (status === 'Awaiting Push Approval') {
        var branch = frm.doc.branch_name || '(auto-generated)';
        var repo = frm.doc.github_repo_url || '';
        var base = frm.doc.base_branch || 'main';
        push_info_html = '<div class="agent-push-info">'
            + '<strong>Ready to push:</strong>'
            + '<table class="agent-push-table">'
            + '<tr><td>Branch:</td><td><code>' + frappe.utils.escape_html(branch) + '</code></td></tr>'
            + '<tr><td>Repository:</td><td><code>' + frappe.utils.escape_html(repo) + '</code></td></tr>'
            + '<tr><td>Base branch:</td><td><code>' + frappe.utils.escape_html(base) + '</code></td></tr>'
            + '<tr><td>PR target:</td><td><code>' + frappe.utils.escape_html(base) + '</code> &larr; <code>' + frappe.utils.escape_html(branch) + '</code></td></tr>'
            + '</table>'
            + '<p style="margin:6px 0 0;font-size:12px;color:var(--text-muted);">Changes are not committed yet. Click <b>Approve Push</b> to commit, push, and/or create a PR.</p>'
            + '</div>';
    }

    var error_html = '';
    if (status === 'Failed' && frm.doc.error_log) {
        var err_preview = (frm.doc.error_log || '').substring(0, 300);
        error_html = '<div class="agent-error-preview">'
            + '<strong>Error:</strong> ' + frappe.utils.escape_html(err_preview)
            + '</div>';
    }

    var html = '<div class="agent-dashboard">'
        + '<div class="agent-status-banner ' + banner_class + '">'
        + spinner
        + '<span class="agent-status-label">' + meta.label + '</span>'
        + '</div>'
        + '<div class="agent-steps-track">' + steps_html + '</div>'
        + pr_html
        + bench_info_html
        + plan_info_html
        + push_info_html
        + error_html
        + '</div>';

    $(wrapper.wrapper).html(html);
}

// ---------------------------------------------------------------------------
// Form styling
// ---------------------------------------------------------------------------

function style_form(frm) {
    if (frm._styled) return;
    frm._styled = true;

    var css = `
    <style>
    .agent-dashboard {
        margin: -5px 0 15px 0;
    }
    .agent-status-banner {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 10px 16px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 14px;
        margin-bottom: 12px;
    }
    .agent-banner-grey   { background: var(--gray-100); color: var(--gray-700); }
    .agent-banner-blue   { background: var(--blue-50, #eff6ff); color: var(--blue-700, #1d4ed8); }
    .agent-banner-orange { background: var(--orange-50, #fff7ed); color: var(--orange-700, #c2410c); }
    .agent-banner-yellow { background: var(--yellow-50, #fefce8); color: var(--yellow-800, #854d0e); }
    .agent-banner-purple { background: var(--purple-50, #faf5ff); color: var(--purple-700, #7e22ce); }
    .agent-banner-green  { background: var(--green-50, #f0fdf4); color: var(--green-700, #15803d); }
    .agent-banner-red    { background: var(--red-50, #fef2f2); color: var(--red-700, #b91c1c); }

    .agent-spinner {
        display: inline-block;
        width: 14px;
        height: 14px;
        border: 2px solid currentColor;
        border-top-color: transparent;
        border-radius: 50%;
        animation: agent-spin 0.8s linear infinite;
    }
    @keyframes agent-spin { to { transform: rotate(360deg); } }

    .agent-steps-track {
        display: flex;
        align-items: flex-start;
        gap: 0;
        padding: 0 4px;
        overflow-x: auto;
    }
    .agent-step {
        display: flex;
        flex-direction: column;
        align-items: center;
        flex: 1;
        min-width: 60px;
        position: relative;
    }
    .agent-step:not(:last-child)::after {
        content: '';
        position: absolute;
        top: 7px;
        left: calc(50% + 9px);
        width: calc(100% - 18px);
        height: 2px;
        background: var(--gray-200);
    }
    .agent-step-done:not(:last-child)::after {
        background: var(--green-400, #4ade80);
    }
    .agent-step-dot {
        width: 16px;
        height: 16px;
        border-radius: 50%;
        background: var(--gray-200);
        border: 2px solid var(--gray-300);
        position: relative;
        z-index: 1;
        transition: all 0.2s;
    }
    .agent-step-done .agent-step-dot {
        background: var(--green-400, #4ade80);
        border-color: var(--green-500, #22c55e);
    }
    .agent-step-active .agent-step-dot {
        background: var(--blue-500, #3b82f6);
        border-color: var(--blue-600, #2563eb);
        box-shadow: 0 0 0 3px var(--blue-100, #dbeafe);
    }
    .agent-step-label {
        font-size: 10px;
        color: var(--text-muted);
        margin-top: 4px;
        text-align: center;
        white-space: nowrap;
    }
    .agent-step-done .agent-step-label { color: var(--green-700, #15803d); font-weight: 500; }
    .agent-step-active .agent-step-label { color: var(--blue-700, #1d4ed8); font-weight: 600; }

    .agent-pr-link {
        margin-top: 8px;
    }
    .agent-error-preview {
        margin-top: 8px;
        padding: 8px 12px;
        background: var(--red-50, #fef2f2);
        border: 1px solid var(--red-200, #fecaca);
        border-radius: 6px;
        font-size: 12px;
        color: var(--red-700, #b91c1c);
        max-height: 80px;
        overflow: hidden;
    }

    .agent-push-info {
        margin-top: 8px;
        padding: 10px 14px;
        background: var(--orange-50, #fff7ed);
        border: 1px solid var(--orange-200, #fed7aa);
        border-radius: 6px;
        font-size: 13px;
    }
    .agent-push-table {
        margin: 6px 0;
        font-size: 12px;
    }
    .agent-push-table td {
        padding: 2px 10px 2px 0;
    }
    .agent-push-table code {
        background: var(--orange-100, #ffedd5);
        padding: 1px 5px;
        border-radius: 3px;
        font-size: 12px;
    }

    .agent-live-log-panel {
        margin-top: 15px;
    }
    .agent-log-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 8px;
    }
    .agent-log-title {
        font-weight: 600;
        font-size: 13px;
    }
    .agent-log-container {
        min-height: 120px;
        max-height: 400px;
        overflow-y: auto;
        background: var(--control-bg);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 12px;
        font-family: 'Fira Code', 'SF Mono', 'Monaco', 'Inconsolata', 'Roboto Mono', monospace;
        font-size: 11.5px;
        line-height: 1.6;
    }
    </style>`;

    $(css).appendTo(frm.wrapper);
}

// ---------------------------------------------------------------------------
// Model dropdown: filter options by selected provider
// ---------------------------------------------------------------------------

// `reset` is passed only when the user actually picked a different provider.
// On load and refresh it is false: a saved request's model is part of the record
// of how that request was run, and rewriting it on open would edit history.
function set_model_options_for_provider(frm, reset) {
    var provider = frm.doc.ai_provider || 'OpenAI';
    var entries = PROVIDER_MODELS[provider] || PROVIDER_MODELS['OpenAI'];
    var model_ids = entries.map(function (m) { return m.value; });
    var current = frm.doc.ai_model || '';
    var settings_model = default_model_for(frm, provider);

    [settings_model, reset ? '' : current].forEach(function (extra) {
        if (extra && model_ids.indexOf(extra) === -1) model_ids.push(extra);
    });

    frm.set_df_property('ai_model', 'options', model_ids.join('\n'));
    // Autocomplete clears any value not in its list on blur; the list is only a
    // suggestion, so a pasted provider model id must survive.
    frm.set_df_property('ai_model', 'ignore_validation', 1);
    frm.refresh_field('ai_model');

    if (reset && model_ids.indexOf(current) === -1) {
        frm.set_value('ai_model', settings_model);
    }

    var desc = entries.map(function (m) {
        return '<b>' + m.value + '</b>: ' + m.label.split(': ')[1];
    }).join(' &nbsp;|&nbsp; ');
    frm.set_df_property('ai_model', 'description', desc);
}

function update_request_type_help(frm) {
    var value = frm.doc.request_type || '';
    var text = REQUEST_TYPE_HELP[value] || '<b>Highlights:</b> Select a request type to see what the agent will focus on.';
    frm.set_df_property('request_type', 'description', text);
    frm.refresh_field('request_type');
}

// ---------------------------------------------------------------------------
// Lock configuration fields once agent has started
// ---------------------------------------------------------------------------

function toggle_config_readonly(frm) {
    var editable = ['Queued', 'Failed', 'Cancelled'].indexOf(frm.doc.status || 'Queued') !== -1;
    var config_fields = [
        'target_app_name', 'github_repo_url', 'github_token',
        'base_branch', 'branch_prefix', 'git_user_name', 'git_user_email',
        'ai_provider', 'ai_model', 'request_type', 'user_message', 'request_title'
    ];
    config_fields.forEach(function (f) {
        frm.set_df_property(f, 'read_only', editable ? 0 : 1);
    });
}

// ---------------------------------------------------------------------------
// Action buttons
// ---------------------------------------------------------------------------

function setup_action_buttons(frm) {
    frm.clear_custom_buttons();
    if (frm.is_new()) return;

    var status = frm.doc.status;
    var running = ['Understanding', 'Planning', 'Implementing', 'Reviewing', 'Building', 'Pushing'];
    var can_start = ['Queued', 'Failed', 'Cancelled'].indexOf(status) !== -1;
    var can_restart = status === 'Completed' || status === 'Awaiting Approval'
        || status === 'Awaiting Bench Approval' || status === 'Awaiting Push Approval';

    if (can_start) {
        frm.add_custom_button(__('Start Agent'), function () {
            start_agent(frm);
        }).addClass('btn-primary-dark');
        frm.change_custom_button_type(__('Start Agent'), null, 'primary');
    }

    if (can_restart) {
        frm.add_custom_button(__('Re-run Agent'), function () {
            frappe.confirm(
                __('This will restart the agent from scratch (Explore → Plan → Review). Continue?'),
                function () { start_agent(frm); }
            );
        }).addClass('btn-primary-dark');
        frm.change_custom_button_type(__('Re-run Agent'), null, 'primary');
    }

    if (status === 'Awaiting Approval') {
        frm.add_custom_button(__('Approve Plan'), function () {
            approve_plan(frm);
        }, __('Actions'));

        frm.add_custom_button(__('Execute Existing Plan'), function () {
            execute_existing_plan(frm);
        }, __('Actions'));

        frm.add_custom_button(__('Reject Plan'), function () {
            reject_plan(frm);
        }, __('Actions'));
    }

    if (PLAN_EDITABLE_STATUSES.includes(status) && (frm.doc.plan_json || '').trim()) {
        frm.add_custom_button(__('Add / Edit Task'), function () {
            open_task_dialog(frm);
        }, __('Actions'));
    }

    var has_plan = !!(frm.doc.agent_plan || '').trim();
    if (has_plan && ['Failed', 'Cancelled', 'Completed', 'Awaiting Push Approval'].indexOf(status) !== -1) {
        frm.add_custom_button(__('Execute Existing Plan'), function () {
            execute_existing_plan(frm);
        }, __('Actions'));
    }

    if (status === 'Awaiting Bench Approval') {
        frm.add_custom_button(__('Approve Bench Commands'), function () {
            approve_bench(frm);
        }).addClass('btn-primary-dark');
        frm.change_custom_button_type(__('Approve Bench Commands'), null, 'primary');

        frm.add_custom_button(__('Reject'), function () {
            frappe.confirm(
                __('Reject bench commands? The request will be cancelled.'),
                function () { cancel_request(frm); }
            );
        }, __('Actions'));
    }

    if (status === 'Awaiting Push Approval') {
        frm.add_custom_button(__('Approve Push'), function () {
            approve_push(frm);
        }).addClass('btn-primary-dark');
        frm.change_custom_button_type(__('Approve Push'), null, 'primary');

        frm.add_custom_button(__('Reject Push'), function () {
            frappe.confirm(
                __('Cancel this push? You can re-run the agent later.'),
                function () { cancel_request(frm); }
            );
        }, __('Actions'));
    }

    var non_running = running.indexOf(status) === -1 && !frm.is_new();
    if (non_running) {
        frm.add_custom_button(__('Checkout Base Branch'), function () {
            checkout_base_branch(frm);
        }, __('Actions'));
    }

    if (running.indexOf(status) !== -1) {
        frm.add_custom_button(__('Cancel'), function () {
            cancel_request(frm);
        }, __('Actions'));
    }

    var followup_allowed = ['Queued', 'Completed', 'Failed', 'Cancelled', 'Awaiting Approval', 'Awaiting Push Approval'].indexOf(status) !== -1;
    if (followup_allowed && !frm.is_new()) {
        frm.add_custom_button(__('Submit Follow-up Fix'), function () {
            open_follow_up_dialog(frm);
        }, __('Actions'));
    }

    if ((frm.doc.branch_name || frm.doc.patch_diff) && !frm.is_new()) {
        frm.add_custom_button(__('Open Koda IDE'), function () {
            frappe.set_route('koda-ide', frm.doc.name);
        }, __('Actions'));
    }
}

function open_follow_up_dialog(frm) {
    var dialog = new frappe.ui.Dialog({
        title: __('Submit Follow-up Fix'),
        fields: [
            {
                fieldname: 'follow_up_message',
                fieldtype: 'Small Text',
                label: __('What broke after testing?'),
                reqd: 1,
                description: __('Describe the error/regression observed after the previous run. This will be appended to the request context.'),
            }
        ],
        primary_action_label: __('Start Follow-up Run'),
        primary_action: function (values) {
            var msg = (values.follow_up_message || '').trim();
            if (!msg) {
                frappe.msgprint(__('Please enter follow-up details.'));
                return;
            }
            frappe.call({
                method: 'ampower_koda.agent.api.submit_follow_up',
                args: {
                    request_name: frm.doc.name,
                    follow_up_message: msg
                },
                freeze: true,
                freeze_message: __('Submitting follow-up and starting surgical fix on same branch...'),
                callback: function (r) {
                    if (r.message && r.message.status === 'ok') {
                        frappe.show_alert({
                            message: __('Follow-up patch started on same branch (plan preserved).'),
                            indicator: 'blue'
                        });
                        dialog.hide();
                        frm.reload_doc();
                    }
                }
            });
        }
    });
    dialog.show();
}

function start_agent(frm) {
    function do_start() {
        frappe.call({
            method: 'ampower_koda.agent.api.start_agent',
            args: { request_name: frm.doc.name },
            freeze: true,
            freeze_message: __('Starting agent...'),
            callback: function (r) {
                if (r.message && r.message.status === 'ok') {
                    frappe.show_alert({
                        message: __('Agent started: exploring codebase...'),
                        indicator: 'blue'
                    });
                    frm.reload_doc();
                }
            }
        });
    }

    if (frm.dirty()) {
        frm.save().then(do_start);
    } else {
        do_start();
    }
}

function execute_existing_plan(frm) {
    frappe.confirm(
        __('Skip exploration & planning: execute the existing plan directly?<br><br>The agent will implement the plan, run bench commands, and ask for push approval.'),
        function () {
            function do_execute() {
                frappe.call({
                    method: 'ampower_koda.agent.api.execute_existing_plan',
                    args: { request_name: frm.doc.name },
                    freeze: true,
                    freeze_message: __('Starting execution from existing plan...'),
                    callback: function (r) {
                        if (r.message && r.message.status === 'ok') {
                            frappe.show_alert({
                                message: __('Executing existing plan: implementation in progress...'),
                                indicator: 'blue'
                            });
                            frm.reload_doc();
                        }
                    }
                });
            }
            if (frm.dirty()) {
                frm.save().then(do_execute);
            } else {
                do_execute();
            }
        }
    );
}

function approve_plan(frm) {
    frappe.confirm(
        __('Approve this plan and start implementation?<br><br>The agent will implement and review each task, then request approval for bench commands.'),
        function () {
            var plan_json = frm.doc.plan_json || '';
            frappe.call({
                method: 'ampower_koda.agent.api.approve_plan',
                args: {
                    request_name: frm.doc.name,
                    plan_json: plan_json
                },
                freeze: true,
                freeze_message: __('Starting execution...'),
                callback: function (r) {
                    if (r.message && r.message.status === 'ok') {
                        frappe.show_alert({
                            message: __('Plan approved: execution in progress...'),
                            indicator: 'green'
                        });
                        frm.reload_doc();
                    }
                }
            });
        }
    );
}

function reject_plan(frm) {
    frappe.confirm(
        __('Reject this plan? The request will be cancelled.'),
        function () {
            frappe.call({
                method: 'ampower_koda.agent.api.reject_plan',
                args: { request_name: frm.doc.name },
                callback: function (r) {
                    if (r.message && r.message.status === 'ok') {
                        frappe.show_alert({
                            message: __('Plan rejected.'),
                            indicator: 'orange'
                        });
                        frm.reload_doc();
                    }
                }
            });
        }
    );
}

function approve_bench(frm) {
    var selected_cmds = [];
    var $list = $(frm.wrapper).find('#bench-cmd-list');
    if ($list.length) {
        $list.find('.bench-cmd-input').each(function () {
            var $input = $(this);
            var $check = $list.find('.bench-cmd-check[data-idx="' + $input.data('idx') + '"]');
            if ($check.is(':checked') && $input.val().trim()) {
                selected_cmds.push($input.val().trim());
            }
        });
    }
    if (!selected_cmds.length) {
        var fallback = [];
        try { fallback = JSON.parse(frm.doc.pending_bench_commands || '[]'); } catch (e) { }
        selected_cmds = fallback;
    }

    if (!selected_cmds.length) {
        frappe.msgprint(__('No bench commands selected.'));
        return;
    }

    var preview = selected_cmds.map(function (c) {
        return '<code style="display:block;padding:3px 8px;margin:2px 0;background:var(--gray-100);border-radius:3px;font-size:12px;">$ '
            + frappe.utils.escape_html(c) + '</code>';
    }).join('');

    frappe.confirm(
        __('Run these commands?') + '<div style="margin:10px 0;">' + preview + '</div>',
        function () {
            frappe.call({
                method: 'ampower_koda.agent.api.approve_bench',
                args: {
                    request_name: frm.doc.name,
                    commands: JSON.stringify(selected_cmds)
                },
                freeze: true,
                freeze_message: __('Running bench commands...'),
                callback: function (r) {
                    if (r.message && r.message.status === 'ok') {
                        frappe.show_alert({
                            message: __('Bench commands approved: running...'),
                            indicator: 'blue'
                        });
                        frm.reload_doc();
                    }
                }
            });
        }
    );
}

function approve_push(frm) {
    var branch = frm.doc.branch_name || '(auto-generated)';
    var repo = frm.doc.github_repo_url || '';
    var base = frm.doc.base_branch || 'main';

    var info_html = '<table style="width:100%;font-size:13px;margin-bottom:14px;">'
        + '<tr><td style="padding:2px 8px 2px 0;font-weight:600;">Branch:</td><td><code>' + frappe.utils.escape_html(branch) + '</code></td></tr>'
        + '<tr><td style="padding:2px 8px 2px 0;font-weight:600;">Repository:</td><td><code>' + frappe.utils.escape_html(repo) + '</code></td></tr>'
        + '<tr><td style="padding:2px 8px 2px 0;font-weight:600;">Target:</td><td><code>' + frappe.utils.escape_html(base) + '</code> &larr; <code>' + frappe.utils.escape_html(branch) + '</code></td></tr>'
        + '</table>';

    var checklist_html = '<div style="margin:10px 0;">'
        + '<p style="font-size:12px;color:var(--text-muted);margin-bottom:8px;">Changes will be committed on approval. Select additional actions:</p>'
        + '<div style="display:flex;align-items:center;gap:8px;margin:6px 0;">'
        + '<input type="checkbox" checked id="push-opt-push" style="margin:0;">'
        + '<label for="push-opt-push" style="margin:0;font-weight:500;">Push branch to remote</label>'
        + '</div>'
        + '<div style="display:flex;align-items:center;gap:8px;margin:6px 0;">'
        + '<input type="checkbox" checked id="push-opt-pr" style="margin:0;">'
        + '<label for="push-opt-pr" style="margin:0;font-weight:500;">Create Pull Request</label>'
        + '</div>'
        + '</div>';

    var d = new frappe.ui.Dialog({
        title: __('Push & Pull Request'),
        fields: [{
            fieldtype: 'HTML',
            options: info_html + checklist_html
        }],
        primary_action_label: __('Proceed'),
        primary_action: function () {
            var do_push = d.$wrapper.find('#push-opt-push').is(':checked');
            var do_pr = d.$wrapper.find('#push-opt-pr').is(':checked');
            if (!do_push && !do_pr) {
                frappe.msgprint(__('Select at least one action.'));
                return;
            }
            d.hide();
            frappe.call({
                method: 'ampower_koda.agent.api.approve_push',
                args: {
                    request_name: frm.doc.name,
                    push_branch: do_push ? 1 : 0,
                    create_pr: do_pr ? 1 : 0
                },
                freeze: true,
                freeze_message: __('Processing...'),
                callback: function (r) {
                    if (r.message && r.message.status === 'ok') {
                        frappe.show_alert({
                            message: r.message.message || __('Done'),
                            indicator: 'green'
                        });
                        frm.reload_doc();
                    }
                }
            });
        }
    });
    d.show();
}

function checkout_base_branch(frm) {
    frappe.confirm(
        __('Switch back to the base branch? Uncommitted changes will be discarded.'),
        function () {
            frappe.call({
                method: 'ampower_koda.agent.api.checkout_base_branch',
                args: { request_name: frm.doc.name },
                freeze: true,
                freeze_message: __('Checking out base branch...'),
                callback: function (r) {
                    if (r.message && r.message.status === 'ok') {
                        frappe.show_alert({
                            message: r.message.message,
                            indicator: 'green'
                        });
                        show_post_checkout_bench_dialog(frm);
                    }
                }
            });
        }
    );
}

function show_post_checkout_bench_dialog(frm) {
    frappe.call({
        method: 'ampower_koda.agent.api.get_default_bench_commands',
        args: { request_name: frm.doc.name },
        callback: function (r) {
            var cmds = (r.message && r.message.commands) || [];
            if (!cmds.length) return;

            var rows_html = cmds.map(function (c, i) {
                return '<div style="display:flex;align-items:center;gap:8px;margin:4px 0;">'
                    + '<input type="checkbox" checked class="checkout-bench-check" data-idx="' + i + '" style="margin:0;">'
                    + '<input type="text" class="checkout-bench-input form-control input-xs" data-idx="' + i + '" value="' + frappe.utils.escape_html(c) + '"'
                    + ' style="flex:1;font-family:monospace;font-size:12px;padding:4px 8px;">'
                    + '</div>';
            }).join('');

            var d = new frappe.ui.Dialog({
                title: __('Run Bench Commands?'),
                fields: [{
                    fieldtype: 'HTML',
                    options: '<p style="margin-bottom:10px;">' + __('Base branch checked out. Select bench commands to run:') + '</p>'
                        + '<div id="checkout-bench-list">' + rows_html + '</div>'
                }],
                primary_action_label: __('Run Selected'),
                primary_action: function () {
                    var selected = [];
                    d.$wrapper.find('.checkout-bench-input').each(function () {
                        var $input = $(this);
                        var idx = $input.data('idx');
                        var $check = d.$wrapper.find('.checkout-bench-check[data-idx="' + idx + '"]');
                        if ($check.is(':checked') && $input.val().trim()) {
                            selected.push($input.val().trim());
                        }
                    });
                    if (!selected.length) {
                        frappe.msgprint(__('No commands selected.'));
                        return;
                    }
                    d.hide();
                    frappe.call({
                        method: 'ampower_koda.agent.api.run_selected_bench_commands',
                        args: {
                            request_name: frm.doc.name,
                            commands: JSON.stringify(selected)
                        },
                        freeze: true,
                        freeze_message: __('Running bench commands...'),
                        callback: function (r2) {
                            if (!r2.message) { return; }
                            // The output is shown whether or not the commands
                            // worked. This used to render only on status 'ok',
                            // so a run that failed showed nothing at all — the
                            // one case where the log is worth reading.
                            var failed = r2.message.failed || [];
                            var header = failed.length
                                ? '<p style="margin-bottom:8px;"><b>'
                                    + __('{0} command(s) failed:', [failed.length])
                                    + '</b><br>'
                                    + frappe.utils.escape_html(failed.join('\n')).replace(/\n/g, '<br>')
                                    + '</p>'
                                : '';
                            frappe.msgprint({
                                title: failed.length
                                    ? __('Bench Commands Failed')
                                    : __('Bench Commands Output'),
                                message: header
                                    + '<pre style="max-height:400px;overflow:auto;font-size:12px;white-space:pre-wrap;">'
                                    + frappe.utils.escape_html(r2.message.log || '(no output)')
                                    + '</pre>',
                                indicator: failed.length ? 'red' : 'green',
                                wide: true
                            });
                            frm.reload_doc();
                        }
                    });
                },
                secondary_action_label: __('Skip'),
                secondary_action: function () { d.hide(); }
            });
            d.show();
        }
    });
}

function cancel_request(frm) {
    frappe.confirm(
        __('Cancel this agent request?'),
        function () {
            frappe.call({
                method: 'ampower_koda.agent.api.cancel_agent_request',
                args: { request_name: frm.doc.name },
                callback: function () {
                    frm.reload_doc();
                }
            });
        }
    );
}

// ---------------------------------------------------------------------------
// Realtime listeners
// ---------------------------------------------------------------------------

function setup_realtime_listeners(frm) {
    if (frm._realtime_bound) return;
    frm._realtime_bound = true;

    frappe.realtime.on('agent_progress', function (data) {
        if (!data || data.request_name !== frm.doc.name) return;
        frm.reload_doc();
    });

    frappe.realtime.on('agent_task_suggestions', function (data) {
        if (!data || data.request_name !== frm.doc.name) return;
        const dialog = frm._task_dialog;
        // Only the dialog that asked gets the answer; a stale reply is dropped.
        if (!dialog || dialog.token !== data.token) return;
        render_task_suggestions(dialog, data);
    });

    setup_status_polling(frm);
}

// ---------------------------------------------------------------------------
// Plan task editor: add or edit one structured task without hand-writing JSON
// ---------------------------------------------------------------------------

// Top-level `var`, not `const`: Frappe compiles the doctype script together with
// any Client Scripts as one function body and may do so more than once, and a
// redeclared `const` is a SyntaxError before anything runs.
var PLAN_EDITABLE_STATUSES = ['Awaiting Approval', 'Failed', 'Cancelled', 'Completed', 'Awaiting Push Approval'];
var split_lines = (text) => (text || '').split('\n').map((s) => s.trim()).filter(Boolean);

function open_task_dialog(frm) {
    const tasks = JSON.parse(frm.doc.plan_json).tasks;
    const ids = tasks.map((t) => t.id);
    const dialog = new frappe.ui.Dialog({
        title: __('Add / Edit Task'),
        size: 'large',
        fields: [
            { fieldname: 'task_id', fieldtype: 'Select', label: __('Task'), options: ['New task', ...ids], default: 'New task',
              change: () => fill_task(dialog, tasks.find((t) => t.id === dialog.get_value('task_id'))) },
            { fieldname: 'title', fieldtype: 'Data', label: __('Title'), reqd: 1 },
            { fieldname: 'goal', fieldtype: 'Data', label: __('Goal (one sentence)'), reqd: 1 },
            { fieldname: 'description', fieldtype: 'Small Text', label: __('Description'), reqd: 1,
              description: __('What changes, where, why, and which existing pattern to follow. No code.') },
            { fieldtype: 'Column Break' },
            { fieldname: 'action', fieldtype: 'Select', label: __('Action'), options: ['MODIFY', 'CREATE'], default: 'MODIFY', reqd: 1 },
            { fieldname: 'depends_on', fieldtype: 'MultiSelectPills', label: __('Depends on'),
              get_data: () => ids.map((id) => ({ value: id, description: '' })) },
            { fieldname: 'acceptance_criteria', fieldtype: 'Small Text', label: __('Acceptance criteria (one per line)'), reqd: 1 },
            { fieldname: 'files', fieldtype: 'Small Text', label: __('Files to edit or create (one per line)'), reqd: 1 },
            { fieldtype: 'Section Break', label: __('Relevant code'),
              description: __('Optional. We search the codebase for what this task describes and pre-select the best files for the implementer to read first.') },
            { fieldname: 'find', fieldtype: 'Button', label: __('Find relevant code'), click: () => find_task_context(frm, dialog) },
            { fieldname: 'results', fieldtype: 'HTML' },
        ],
        primary_action_label: __('Save Task'),
        primary_action: (values) => save_task(frm, dialog, values),
        secondary_action_label: __('Remove Task'),
        secondary_action: () => remove_task(frm, dialog, tasks),
    });
    dialog.suggestions = [];
    dialog.existing_refs = [];
    frm._task_dialog = dialog;
    dialog.show();
    // Only an existing task can be removed; the button follows the selector.
    dialog.get_secondary_btn().addClass('btn-danger').hide();
}

function remove_task(frm, dialog, tasks) {
    const task_id = dialog.get_value('task_id');
    const task = tasks.find((t) => t.id === task_id);
    if (!task) return;
    if (tasks.length === 1) {
        return frappe.msgprint(__('A plan needs at least one task. Edit this task or reject the plan instead.'));
    }
    const dependents = tasks.filter((t) => (t.depends_on || []).includes(task_id)).map((t) => t.id);
    const esc = frappe.utils.escape_html;
    let message = __('Remove {0} ({1}) from the plan? Later tasks are renumbered.', [esc(task_id), esc(task.title)]);
    if (dependents.length) {
        message += '<br><br>' + __('{0} depend on it; that dependency is dropped.', [esc(dependents.join(', '))]);
    }
    frappe.confirm(message, () => {
        frappe.call({
            method: 'ampower_koda.agent.api.remove_plan_task',
            args: { request_name: frm.doc.name, task_id },
            freeze: true,
            callback: (r) => {
                frappe.show_alert({ message: __('Removed {0}', [r.message.removed]), indicator: 'green' });
                dialog.hide();
                frm._task_dialog = null;
                frm.reload_doc();
            },
        });
    });
}

function fill_task(dialog, task) {
    dialog.existing_refs = task ? task.context_refs : [];
    dialog.get_secondary_btn().toggle(Boolean(task));
    if (!task) return;
    dialog.set_values({
        title: task.title, goal: task.goal, description: task.description, action: task.action,
        acceptance_criteria: task.acceptance_criteria.join('\n'), files: task.files.join('\n'),
    });
    dialog.set_value('depends_on', task.depends_on);
}

function find_task_context(frm, dialog) {
    const query = ['title', 'goal', 'description'].map((f) => dialog.get_value(f)).filter(Boolean).join('\n');
    if (!query) return frappe.msgprint(__('Fill in the title or description first.'));
    dialog.fields_dict.results.$wrapper.html(`<div class="koda-suggest"><div class="koda-loading">
        <span class="spinner-border spinner-border-sm"></span>
        ${__('Searching the codebase... the first search on a large app can take up to a minute.')}</div></div>`);
    frappe.call({
        method: 'ampower_koda.agent.api.suggest_task_context',
        args: { request_name: frm.doc.name, query },
        callback: (r) => { dialog.token = r.message.token; },
    });
}

var MAX_CONTEXT_REFS = 6;

var KODA_SUGGEST_CSS = `<style>
.koda-suggest .koda-loading{display:flex;gap:8px;align-items:center;color:var(--text-muted);padding:8px 0}
.koda-suggest .koda-summary{display:flex;justify-content:space-between;color:var(--text-muted);font-size:var(--text-sm);margin:4px 0 10px}
.koda-suggest .koda-file{border:1px solid var(--border-color);border-radius:var(--border-radius-md,8px);padding:10px 12px;margin-bottom:8px;background:var(--fg-color,#fff)}
.koda-suggest .koda-file.is-active{border-color:var(--primary)}
.koda-suggest .koda-file-head{display:flex;justify-content:space-between;align-items:baseline;gap:8px}
.koda-suggest .koda-file-dir{color:var(--text-muted);font-size:var(--text-sm);margin-left:6px;word-break:break-all}
.koda-suggest .koda-strength{color:var(--primary);letter-spacing:2px;font-size:10px;white-space:nowrap}
.koda-suggest .koda-file-actions{display:flex;gap:6px;align-items:center;margin-top:6px;flex-wrap:wrap}
.koda-suggest .koda-file-count{color:var(--text-muted);font-size:var(--text-sm);margin-left:auto}
.koda-suggest .koda-spans{margin:8px 0 0;padding-left:2px}
.koda-suggest .koda-span{display:flex;gap:8px;align-items:center;font-size:var(--text-sm);margin:3px 0;cursor:pointer;font-weight:normal}
.koda-suggest .koda-span input{margin:0}
.koda-suggest .koda-span-name{font-family:var(--font-stack-mono,monospace);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.koda-suggest .koda-span-lines{color:var(--text-muted);margin-left:auto;white-space:nowrap}
</style>`;

function render_task_suggestions(dialog, data) {
    const $results = dialog.fields_dict.results.$wrapper;
    const esc = frappe.utils.escape_html;
    dialog.suggestions = data.suggestions || [];
    if (data.error) return $results.html(`<p class="text-danger">${esc(data.error)}</p>`);
    if (!dialog.suggestions.length) {
        return $results.html(`<p class="text-muted">${__('Nothing matched. Try different words, or enter file paths by hand.')}</p>`);
    }
    // Group the ranked sections by file; the first file seen is the best match.
    const files = [];
    dialog.suggestions.forEach((s, index) => {
        let file = files.find((f) => f.path === s.path);
        if (!file) files.push(file = { path: s.path, score: s.score, spans: [] });
        file.spans.push({ ...s, index });
    });
    // Pre-select: the top three files, up to two sections each, within the cap.
    let budget = MAX_CONTEXT_REFS;
    dialog.selection = {};
    files.forEach((file, rank) => {
        const spans = rank < 3 ? file.spans.slice(0, Math.min(2, budget)).map((s) => s.index) : [];
        budget -= spans.length;
        dialog.selection[file.path] = { read: spans.length > 0, edit: false, spans: new Set(spans) };
    });
    dialog.suggestion_files = files;
    paint_task_suggestions(dialog);

    $results.off('click.koda').on('click.koda', '.koda-toggle', function () {
        const state = dialog.selection[this.dataset.path];
        state[this.dataset.kind] = !state[this.dataset.kind];
        if (this.dataset.kind === 'read' && state.read && !state.spans.size) {
            // Reading a file with nothing ticked means its best section.
            const first = files.find((f) => f.path === this.dataset.path).spans[0].index;
            if (selected_ref_count(dialog) < MAX_CONTEXT_REFS) state.spans.add(first);
        }
        paint_task_suggestions(dialog);
    }).off('change.koda').on('change.koda', '.koda-span input', function () {
        const state = dialog.selection[this.dataset.path];
        const index = Number(this.dataset.index);
        if (this.checked && selected_ref_count(dialog) >= MAX_CONTEXT_REFS) {
            this.checked = false;
            return frappe.show_alert({ message: __('At most {0} reference sections per task.', [MAX_CONTEXT_REFS]), indicator: 'orange' });
        }
        if (this.checked) state.spans.add(index); else state.spans.delete(index);
        paint_task_suggestions(dialog);
    });
}

function selected_ref_count(dialog) {
    return Object.values(dialog.selection).reduce((n, s) => n + (s.read ? s.spans.size : 0), 0);
}

function paint_task_suggestions(dialog) {
    const esc = frappe.utils.escape_html;
    const files = dialog.suggestion_files;
    const top = files[0].score || 1;
    const refs = selected_ref_count(dialog);
    const edits = Object.values(dialog.selection).filter((s) => s.edit).length;
    const strength = (score) => {
        const ratio = score / top;
        return ratio >= 0.75 ? '●●●' : ratio >= 0.4 ? '●●○' : '●○○';
    };
    const cards = files.map((file) => {
        const state = dialog.selection[file.path];
        const slash = file.path.lastIndexOf('/');
        const name = file.path.slice(slash + 1);
        const dir = slash > 0 ? file.path.slice(0, slash) : '';
        const spans = !state.read ? '' : `<div class="koda-spans">${file.spans.map((s) => `
            <label class="koda-span">
                <input type="checkbox" data-path="${esc(file.path)}" data-index="${s.index}" ${state.spans.has(s.index) ? 'checked' : ''}>
                <span class="koda-span-name" title="${esc(s.snippet)}">${esc(s.symbol || s.snippet || __('section'))}</span>
                <span class="koda-span-lines">${__('lines {0}–{1}', [s.start, s.end])}</span>
            </label>`).join('')}</div>`;
        const toggle = (kind, label) => `<button type="button" class="btn btn-xs koda-toggle ${state[kind] ? 'btn-primary' : 'btn-default'}"
            data-path="${esc(file.path)}" data-kind="${kind}">${label}</button>`;
        return `<div class="koda-file ${state.read || state.edit ? 'is-active' : ''}">
            <div class="koda-file-head">
                <div class="koda-file-name"><b>${esc(name)}</b><span class="koda-file-dir">${esc(dir)}</span></div>
                <span class="koda-strength" title="${__('How strongly this file matches the task')}">${strength(file.score)}</span>
            </div>
            <div class="koda-file-actions">
                ${toggle('read', __('Read for context'))}${toggle('edit', __('Change this file'))}
                <span class="koda-file-count">${__('{0} matching section(s)', [file.spans.length])}</span>
            </div>${spans}</div>`;
    });
    dialog.fields_dict.results.$wrapper.html(`${KODA_SUGGEST_CSS}<div class="koda-suggest">
        <div class="koda-summary">
            <span>${__('{0} file(s) found', [files.length])}</span>
            <span>${__('{0} of {1} reference sections · {2} to change', [refs, MAX_CONTEXT_REFS, edits])}</span>
        </div>${cards.join('')}</div>`);
}

function save_task(frm, dialog, values) {
    const selection = Object.entries(dialog.selection || {});
    const refs = selection.flatMap(([, state]) => state.read ? [...state.spans] : []).map((index) => {
        const s = dialog.suggestions[index];
        return { path: s.path, start: s.start, end: s.end, symbol: s.symbol, why: s.snippet ? `Pattern: ${s.snippet}` : 'Related code' };
    });
    const edited = selection.filter(([, state]) => state.edit).map(([path]) => path);
    const files = [...new Set([...split_lines(values.files), ...edited])];
    if (!files.length) return frappe.msgprint(__('Name at least one file to change, or mark a suggested file as "Change this file".'));
    const task = {
        id: values.task_id === 'New task' ? '' : values.task_id,
        title: values.title, goal: values.goal, description: values.description, action: values.action,
        files,
        // Server falls back to whole-file refs when this is empty.
        context_refs: refs.length ? refs : dialog.existing_refs,
        acceptance_criteria: split_lines(values.acceptance_criteria),
        depends_on: values.depends_on || [],
    };
    frappe.call({
        method: 'ampower_koda.agent.api.save_plan_task',
        args: { request_name: frm.doc.name, task: JSON.stringify(task) },
        freeze: true,
        callback: (r) => {
            frappe.show_alert({ message: __('Saved {0}', [r.message.task_id]), indicator: 'green' });
            dialog.hide();
            frm._task_dialog = null;
            frm.reload_doc();
        },
    });
}

function setup_status_polling(frm) {
    if (frm._poll_timer) {
        clearInterval(frm._poll_timer);
        frm._poll_timer = null;
    }

    var active = ['Understanding', 'Planning', 'Implementing', 'Reviewing', 'Building', 'Pushing'];
    if (active.indexOf(frm.doc.status) === -1) return;

    var poll_interval = (frm.doc.status === 'Building' || frm.doc.status === 'Pushing') ? 5000 : 10000;

    frm._poll_timer = setInterval(function () {
        if (!frm.doc || !frm.doc.name) {
            clearInterval(frm._poll_timer);
            frm._poll_timer = null;
            return;
        }
        frappe.call({
            method: 'ampower_koda.agent.api.get_agent_status',
            args: { request_name: frm.doc.name },
            async: true,
            silent: true,
            callback: function (r) {
                if (r.message && r.message.status && r.message.status !== frm.doc.status) {
                    clearInterval(frm._poll_timer);
                    frm._poll_timer = null;
                    frm.reload_doc();
                }
            }
        });
    }, poll_interval);
}

// ---------------------------------------------------------------------------
// Live Log Panel
// ---------------------------------------------------------------------------

function setup_live_log_panel(frm) {
    var status = frm.doc.status;
    var visible_statuses = [
        'Queued', 'Understanding', 'Planning', 'Implementing', 'Reviewing', 'Building', 'Pushing',
        'Awaiting Approval', 'Awaiting Bench Approval', 'Awaiting Push Approval', 'Failed', 'Completed'
    ];

    if (visible_statuses.indexOf(status) === -1) {
        if (frm._log_panel) frm._log_panel.hide();
        return;
    }

    // Anchor: Preference is before stage_log, fallback is after the status dashboard
    var target = frm.fields_dict.stage_log;
    var $anchor = (target && target.$wrapper && target.$wrapper.length) ? target.$wrapper : null;
    if (!$anchor) {
        var dash = frm.fields_dict.status_dashboard_html;
        $anchor = (dash && dash.$wrapper && dash.$wrapper.length) ? dash.$wrapper : null;
    }

    if (!frm._log_panel) {
        var html = '<div class="agent-live-log-panel">'
            + '<div class="agent-log-header">'
            + '<span class="agent-log-title">Live Agent Log</span>'
            + '<span class="agent-log-status indicator-pill whitespace-nowrap yellow" style="font-size: 11px;"></span>'
            + '</div>'
            + '<div class="agent-log-container"></div>'
            + '</div>';
        
        var $panel = $(html);

        if ($anchor) {
            $anchor.before($panel);
        } else {
            // Last resort: main wrapper
            $(frm.wrapper).find('.form-body').append($panel);
        }

        frm._log_panel = $panel;
        frm._log_container = $panel.find('.agent-log-container');
        frm._log_status = $panel.find('.agent-log-status');

        frappe.realtime.on('agent_log', function (data) {
            if (!data || data.request_name !== frm.doc.name) return;
            append_log_entry(frm, data);
        });
    } else {
        // Re-attach if the DOM was wiped during reload_doc
        if (!$.contains(document.documentElement, frm._log_panel[0])) {
            if ($anchor) {
                $anchor.before(frm._log_panel);
            }
        }
    }

    frm._log_panel.show();
    frm._log_status.text(status);

    // Re-populate from history if panel was just re-attached or refreshed
    if (frm._log_history && frm._log_history.length && frm._log_container.is(':empty')) {
        frm._log_history.forEach(function (data) {
            // We temporarily bypass the deduplication check in append_log_entry for this re-population
            var old_ids = frm._last_entry_ids;
            frm._last_entry_ids = []; 
            append_log_entry(frm, data);
            frm._last_entry_ids = old_ids;
        });
    }
}

function append_log_entry(frm, data) {
    if (!frm._log_history) frm._log_history = [];
    
    // Check if this specific log entry is already in history to avoid duplicates after reload
    var entry_id = (data.timestamp || '') + (data.type || '') + (data.tool_name || '') + (data.preview || '').substring(0, 50);
    if (frm._last_entry_ids && frm._last_entry_ids.indexOf(entry_id) !== -1) return;
    
    frm._log_history.push(data);
    if (!frm._last_entry_ids) frm._last_entry_ids = [];
    frm._last_entry_ids.push(entry_id);
    if (frm._last_entry_ids.length > 50) frm._last_entry_ids.shift();

    if (!frm._log_container) return;

    var time = data.timestamp ? data.timestamp.split(' ')[1] : '';
    var ts = '<span style="color:var(--text-muted);margin-right:6px;">' + time + '</span>';
    var html = '';

    if (data.type === 'tool_call') {
        var args_str = '';
        if (data.tool_args) {
            try {
                args_str = typeof data.tool_args === 'string' ? data.tool_args : JSON.stringify(data.tool_args);
                if (args_str.length > 120) args_str = args_str.substring(0, 120) + '\u2026';
            } catch (e) { args_str = ''; }
        }
        html = '<div style="color:var(--blue-500);margin-bottom:3px;">'
            + ts + '<b>\u25B6 ' + frappe.utils.escape_html(data.tool_name || '') + '</b>'
            + (args_str ? ' <span style="color:var(--text-light);">' + frappe.utils.escape_html(args_str) + '</span>' : '')
            + '</div>';
    } else if (data.type === 'tool_result') {
        var preview = (data.result_preview || '').substring(0, 180);
        html = '<div style="color:var(--green-600);margin-bottom:3px;padding-left:14px;">'
            + ts + '\u2500 ' + frappe.utils.escape_html(data.tool_name || '') + ': '
            + '<span style="color:var(--text-light);">' + frappe.utils.escape_html(preview) + '</span>'
            + '</div>';
    } else if (data.type === 'token_usage') {
        var input_tokens = Number(data.input_tokens || 0);
        var cached_tokens = Number(data.cache_read_tokens || 0);
        var output_tokens = Number(data.output_tokens || 0);
        var context_tokens = data.context_chars ? Math.ceil(Number(data.context_chars) / 3.6) : 0;
        var pieces = [];
        if (input_tokens) pieces.push('input ' + input_tokens.toLocaleString());
        if (cached_tokens) pieces.push('cached ' + cached_tokens.toLocaleString());
        if (output_tokens) pieces.push('output ' + output_tokens.toLocaleString());
        if (context_tokens) pieces.push('context ~' + context_tokens.toLocaleString());
        html = '<div style="color:var(--text-muted);margin-bottom:3px;padding-left:14px;">'
            + ts + '\u25C7 round ' + frappe.utils.escape_html(String(data.round || '?'))
            + ': ' + frappe.utils.escape_html(pieces.join(' \u00B7 ') || String(data.tokens_this_round || 0) + ' tokens')
            + ' <b>(total ' + Number(data.tokens_total || 0).toLocaleString() + ')</b>'
            + '</div>';
    } else if (data.type === 'duplicate_tool_call') {
        html = '<div style="color:var(--orange-500);margin-bottom:3px;padding-left:14px;">'
            + ts + '\u21B7 skipped duplicate ' + frappe.utils.escape_html(data.tool_name || '')
            + ' (already in round ' + frappe.utils.escape_html(String(data.original_round || '?')) + ')'
            + '</div>';
    } else if (data.type === 'llm_response') {
        var preview = (data.preview || '').substring(0, 250);
        html = '<div style="color:var(--orange-500);margin-bottom:5px;">'
            + ts + '<b>\u2728 LLM</b> '
            + '<span style="color:var(--text-color);">' + frappe.utils.escape_html(preview) + '</span>'
            + '</div>';
    } else if (data.type === 'bench_command') {
        html = '<div style="color:var(--purple-500);margin-bottom:3px;">'
            + ts + '<b>$ ' + frappe.utils.escape_html(data.command || '') + '</b>'
            + '</div>';
    } else if (data.type === 'bench_result') {
        var color = data.success ? 'var(--green-600)' : 'var(--red-500)';
        html = '<div style="color:' + color + ';margin-bottom:3px;padding-left:14px;">'
            + ts + (data.success ? '\u2713 OK' : '\u2717 FAILED') + ' '
            + '<span style="color:var(--text-light);">' + frappe.utils.escape_html((data.output_preview || '').substring(0, 180)) + '</span>'
            + '</div>';
    } else {
        html = '<div style="margin-bottom:3px;">'
            + ts + frappe.utils.escape_html(JSON.stringify(data).substring(0, 250))
            + '</div>';
    }

    frm._log_container.append(html);
    frm._log_container.scrollTop(frm._log_container[0].scrollHeight);
}

// ---------------------------------------------------------------------------
// User defaults: remember last-used configuration across requests
// ---------------------------------------------------------------------------

function save_user_defaults(frm) {
    var fields = ['target_app_name', 'github_repo_url', 'base_branch', 'branch_prefix',
        'git_user_name', 'git_user_email', 'ai_provider', 'ai_model'];
    var vals = {};
    fields.forEach(function (f) {
        if (frm.doc[f]) {
            vals['ai_agent_' + f] = frm.doc[f];
        }
    });
    if (Object.keys(vals).length) {
        frappe.call({
            method: 'frappe.model.utils.user_settings.save',
            args: {
                doctype: 'Agent Request',
                user_settings: JSON.stringify(vals)
            },
            async: true
        });
    }
}

function load_user_defaults(frm) {
    var stored = {};
    try {
        var raw = frappe.get_user_settings('Agent Request');
        if (!raw || !Object.keys(raw).length) {
            raw = frappe.get_user_settings('AI Agent Request') || {};
        }
        if (raw && typeof raw === 'object') stored = raw;
    } catch (e) { /* no saved settings */ }

    var fields = ['target_app_name', 'github_repo_url', 'base_branch', 'branch_prefix',
        'git_user_name', 'git_user_email', 'ai_provider'];
    fields.forEach(function (f) {
        var val = stored['ai_agent_' + f];
        if (val && !frm.doc[f]) {
            frm.set_value(f, val);
        }
    });

    setTimeout(function () {
        set_model_options_for_provider(frm, false);
        // Setting the provider above already reset the model to its default, so
        // the user's last-used model must win unconditionally here.
        var saved_model = stored['ai_agent_ai_model'];
        if (saved_model) {
            frm.set_value('ai_model', saved_model);
        }
    }, 100);
}

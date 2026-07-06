// Copyright (c) 2026, Ambibuzz Technologies LLP and contributors
// AI Agent Request — client script

var PROVIDER_MODELS = {
    'OpenAI': [
        { value: 'gpt-4o-mini', label: 'GPT-4o Mini — fast, cost-effective' },
        { value: 'gpt-4o', label: 'GPT-4o — best overall' },
        { value: 'gpt-5-mini', label: 'GPT-5 Mini — next-gen, efficient' },
        { value: 'o3-mini', label: 'o3-mini — reasoning model' }
    ],
    'Gemini': [
        { value: 'gemini-2.0-flash', label: 'Gemini 2.0 Flash — fast, multimodal' },
        { value: 'gemini-2.5-pro', label: 'Gemini 2.5 Pro — most capable' }
    ],
    'Claude': [
        { value: 'claude-sonnet-4-20250514', label: 'Claude Sonnet 4 — balanced' },
        { value: 'claude-3-5-sonnet-20241022', label: 'Claude 3.5 Sonnet — proven' },
        { value: 'claude-3-5-haiku-20241022', label: 'Claude 3.5 Haiku — fast, light' }
    ]
};

var DEFAULT_MODELS = {
    'OpenAI': 'gpt-4o-mini',
    'Gemini': 'gemini-2.0-flash',
    'Claude': 'claude-sonnet-4-20250514'
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
    'Reviewing': { color: 'yellow', label: 'Reviewing Code' },
    'Awaiting Bench Approval': { color: 'orange', label: 'Approve Bench Commands' },
    'Building': { color: 'purple', label: 'Running Bench Commands' },
    'Awaiting Push Approval': { color: 'orange', label: 'Approve Push to GitHub' },
    'Pushing': { color: 'purple', label: 'Creating Pull Request' },
    'Completed': { color: 'green', label: 'Completed' },
    'Failed': { color: 'red', label: 'Failed' },
    'Cancelled': { color: 'grey', label: 'Cancelled' }
};

frappe.ui.form.on('AI Agent Request', {
    refresh: function (frm) {
        render_status_dashboard(frm);
        setup_action_buttons(frm);
        setup_realtime_listeners(frm);
        setup_status_polling(frm);
        setup_live_log_panel(frm);
        set_model_options_for_provider(frm);
        toggle_config_readonly(frm);
        style_form(frm);
    },

    ai_provider: function (frm) {
        var provider = frm.doc.ai_provider || 'OpenAI';
        set_model_options_for_provider(frm);
        frm.set_value('ai_model', DEFAULT_MODELS[provider] || DEFAULT_MODELS['OpenAI']);
    },

    before_save: function (frm) {
        if (frm.is_new()) {
            save_user_defaults(frm);
        }
    },

    onload: function (frm) {
        if (frm.is_new()) {
            load_user_defaults(frm);
            set_model_options_for_provider(frm);
        }
    }
});

// ---------------------------------------------------------------------------
// Status Dashboard — visual progress indicator
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
        { key: 'Reviewing', short: 'Verify' },
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
// Model dropdown — filter options by selected provider
// ---------------------------------------------------------------------------

function set_model_options_for_provider(frm) {
    var provider = frm.doc.ai_provider || 'OpenAI';
    var entries = PROVIDER_MODELS[provider] || PROVIDER_MODELS['OpenAI'];
    var model_ids = entries.map(function (m) { return m.value; });
    var options_str = model_ids.join('\n');

    frm.set_df_property('ai_model', 'options', options_str);
    frm.refresh_field('ai_model');

    var current = frm.doc.ai_model || '';
    if (model_ids.indexOf(current) === -1) {
        frm.set_value('ai_model', model_ids[0]);
    }

    var desc = entries.map(function (m) {
        return '<b>' + m.value + '</b> — ' + m.label.split(' — ')[1];
    }).join(' &nbsp;|&nbsp; ');
    frm.set_df_property('ai_model', 'description', desc);
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

    // kg_cache_key holds the linked Koda Knowledge Graph document name.
    if (frm.doc.kg_status === 'Ready' && frm.doc.kg_cache_key) {
        frm.add_custom_button(__('Visualize Knowledge Graph'), function () {
            frappe.set_route('Form', 'Koda Knowledge Graph', frm.doc.kg_cache_key);
        }, __('Actions'));
    } else if (frm.doc.kg_status === 'Building') {
        frm.add_custom_button(__('Knowledge Graph Building…'), function () {
            frappe.show_alert({
                message: __('The knowledge graph is still being built.'),
                indicator: 'orange'
            });
        }, __('Actions'));
    } else if (frm.doc.kg_status === 'Failed') {
        frm.add_custom_button(__('Knowledge Graph Failed'), function () {
            frappe.show_alert({
                message: __('Knowledge graph build failed — check the server error log for details.'),
                indicator: 'red'
            });
        }, __('Actions'));
    }

    if (running.indexOf(status) !== -1) {
        frm.add_custom_button(__('Cancel'), function () {
            cancel_request(frm);
        }, __('Actions'));
    }
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
                        message: __('Agent started — exploring codebase...'),
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
        __('Skip exploration & planning — execute the existing plan directly?<br><br>The agent will implement the plan, run bench commands, and ask for push approval.'),
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
                                message: __('Executing existing plan — implementation in progress...'),
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
        __('Approve this plan and start execution?<br><br>The agent will implement the changes, run bench commands, and create a Pull Request.'),
        function () {
            var edited_plan = frm.doc.agent_plan || '';
            frappe.call({
                method: 'ampower_koda.agent.api.approve_plan',
                args: {
                    request_name: frm.doc.name,
                    edited_plan: edited_plan
                },
                freeze: true,
                freeze_message: __('Starting execution...'),
                callback: function (r) {
                    if (r.message && r.message.status === 'ok') {
                        frappe.show_alert({
                            message: __('Plan approved — execution in progress...'),
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
                            message: __('Bench commands approved — running...'),
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
    var app = frm.doc.target_app_name || '';
    var site = frm.doc.site_name || '';
    var default_cmds = [];
    if (app) {
        default_cmds = [
            'bench --site ' + (site || '{site}') + ' migrate',
            'bench build --app ' + app,
            'bench --site ' + (site || '{site}') + ' clear-cache',
            'supervisorctl restart all'
        ];
    }

    frappe.call({
        method: 'ampower_koda.agent.api.get_default_bench_commands',
        args: { request_name: frm.doc.name },
        callback: function (r) {
            var cmds = (r.message && r.message.commands) ? r.message.commands : default_cmds;
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
                            if (r2.message && r2.message.status === 'ok') {
                                frappe.msgprint({
                                    title: __('Bench Commands Output'),
                                    message: '<pre style="max-height:400px;overflow:auto;font-size:12px;white-space:pre-wrap;">'
                                        + frappe.utils.escape_html(r2.message.log || '(no output)')
                                        + '</pre>',
                                    indicator: 'green',
                                    wide: true
                                });
                                frm.reload_doc();
                            }
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

    setup_status_polling(frm);
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
// User defaults — remember last-used configuration across requests
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
                doctype: 'AI Agent Request',
                user_settings: JSON.stringify(vals)
            },
            async: true
        });
    }
}

function load_user_defaults(frm) {
    var stored = {};
    try {
        var raw = frappe.get_user_settings('AI Agent Request');
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
        set_model_options_for_provider(frm);
        var saved_model = stored['ai_agent_ai_model'];
        if (saved_model && !frm.doc.ai_model) {
            frm.set_value('ai_model', saved_model);
        }
    }, 100);
}

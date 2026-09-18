frappe.pages['hd-ticket-intake'].on_page_load = function(wrapper) {
    var page = frappe.ui.make_app_page({
        parent: wrapper,
        title: 'HD Ticket Intake',
        single_column: true
    });

    var style = `
    <style>
        .hdti-wrap { max-width: 760px; margin: 20px auto; }
        .hdti-card {
            background: var(--card-bg, #fff);
            border: 1px solid var(--border-color, #e0e0e0);
            border-radius: 10px;
            padding: 24px;
            margin-bottom: 20px;
        }
        .hdti-card h3 { margin: 0 0 4px; font-size: 16px; font-weight: 600; }
        .hdti-card .hdti-subtitle { color: var(--text-muted, #8d99a6); font-size: 13px; margin-bottom: 20px; }
        .hdti-field { margin-bottom: 16px; }
        .hdti-field label { display: block; font-size: 13px; font-weight: 500; margin-bottom: 6px; }
        .hdti-hint { font-size: 12px; color: var(--text-muted, #8d99a6); margin: 8px 0 16px; }
        .hdti-actions { display: flex; gap: 8px; }
        .hdti-actions .btn-play::before { content: "\\25B6"; margin-right: 6px; font-size: 10px; }
        .hdti-result-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 16px; }
        .hdti-result-header-left { display: flex; align-items: center; gap: 10px; }
        .hdti-result-header h3 { margin: 0; }
        .hdti-badge {
            font-size: 11px; font-weight: 600; letter-spacing: 0.03em;
            padding: 3px 10px; border-radius: 20px; text-transform: uppercase;
        }
        .hdti-badge-success { background: #e3f6e8; color: #1c7a3c; }
        .hdti-badge-info { background: #e6f0fb; color: #1a5fb4; }
        .hdti-badge-warn { background: #fdf1e0; color: #a5620a; }
        .hdti-body { font-size: 13px; line-height: 1.6; white-space: pre-wrap; }
        .hdti-link { display: inline-block; margin-top: 16px; font-weight: 500; }
    </style>
    `;

    var content = `
    <div class="hdti-wrap">
        <div class="hdti-card">
            <h3>HD Ticket Intake</h3>
            <div class="hdti-subtitle">Paste a helpdesk ticket's subject and description to run it through Knowledge Base triage and start an agent.</div>

            <div class="hdti-field">
                <label>Subject</label>
                <input type="text" class="form-control" id="hd-ticket-subject" placeholder="License validation fails after renewal">
            </div>
            <div class="hdti-field">
                <label>Description</label>
                <textarea class="form-control" id="hd-ticket-description" rows="6" placeholder="Paste the ticket description here"></textarea>
            </div>

            <div class="hdti-hint">Press Ctrl + Enter to execute.</div>

            <div class="hdti-actions">
                <button class="btn btn-primary btn-play" id="hd-ticket-execute">Execute agent</button>
                <button class="btn btn-default" id="hd-ticket-clear">Clear</button>
            </div>
        </div>

        <div class="hdti-card" id="hd-ticket-unknown-repo" style="display:none;">
            <h3>Repository details needed</h3>
            <div class="hdti-subtitle">Knowledge Base couldn't confidently identify the repository for this ticket.</div>
            <div class="hdti-field">
                <label>App Name</label>
                <input type="text" class="form-control" id="hd-ticket-app-name" placeholder="ampower_license_hub">
            </div>
            <div class="hdti-field">
                <label>GitHub Repo URL</label>
                <input type="text" class="form-control" id="hd-ticket-repo-url" placeholder="https://github.com/Ambibuzz/ampower_license_hub.git">
            </div>
            <div class="hdti-field">
                <label>Base Branch</label>
                <input type="text" class="form-control" id="hd-ticket-branch" value="main">
            </div>
            <button class="btn btn-primary" id="hd-ticket-continue">Continue</button>
        </div>

        <div class="hdti-card" id="hd-ticket-result-card" style="display:none;">
            <div class="hdti-result-header">
                <div class="hdti-result-header-left">
                    <h3>Result</h3>
                    <span class="hdti-badge" id="hd-ticket-badge"></span>
                </div>
            </div>
            <div class="hdti-body" id="hd-ticket-body"></div>
        </div>
    </div>
    `;

    $(page.body).html(style + content);

    function set_loading(btn, is_loading, label) {
        btn.prop('disabled', is_loading).text(is_loading ? 'Running...' : label);
    }

    function show_result(badge_class, badge_text, body_html) {
        var card = page.wrapper.find('#hd-ticket-result-card');
        page.wrapper.find('#hd-ticket-badge').attr('class', 'hdti-badge ' + badge_class).text(badge_text);
        page.wrapper.find('#hd-ticket-body').html(body_html);
        card.show();
    }

    function do_execute() {
        var subject = page.wrapper.find('#hd-ticket-subject').val().trim();
        var description = page.wrapper.find('#hd-ticket-description').val().trim();

        if (!subject || !description) {
            frappe.msgprint(__('Please fill in both Subject and Description.'));
            return;
        }

        var btn = page.wrapper.find('#hd-ticket-execute');
        set_loading(btn, true);
        page.wrapper.find('#hd-ticket-unknown-repo').hide();
        page.wrapper.find('#hd-ticket-result-card').hide();

        frappe.call({
            method: 'ampower_koda.agent.hd_ticket_intake.execute_from_ticket',
            args: { subject: subject, description: description },
            callback: function(r) {
                set_loading(btn, false, 'Execute agent');
                if (!r.message) return;
                var data = r.message;

                if (data.status === 'not_actionable') {
                    show_result('hdti-badge-info', data.ticket_type || 'not actionable',
                        frappe.utils.escape_html(data.reply_to_ticket || ''));
                    return;
                }

                if (data.status === 'ok') {
                    show_result('hdti-badge-success', 'started',
                        frappe.utils.escape_html(data.reply_to_ticket || '') +
                        '<a href="' + data.redirect_url + '" class="hdti-link">' +
                        __('Open Agent Request') + ' \u2192</a>');
                    page.wrapper.find('#hd-ticket-subject').val('');
                    page.wrapper.find('#hd-ticket-description').val('');
                    return;
                }

                if (data.status === 'unknown_repo') {
                    page._pending_kb = data;
                    if (data.suggested_app_name) {
                        page.wrapper.find('#hd-ticket-app-name').val(data.suggested_app_name);
                    }
                    page.wrapper.find('#hd-ticket-unknown-repo').show();
                    show_result('hdti-badge-warn', 'needs repo', frappe.utils.escape_html(data.reply_to_ticket || ''));
                }
            },
            error: function() { set_loading(btn, false, 'Execute agent'); }
        });
    }

    page.wrapper.find('#hd-ticket-execute').on('click', do_execute);

    page.wrapper.find('#hd-ticket-subject, #hd-ticket-description').on('keydown', function(e) {
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            e.preventDefault();
            do_execute();
        }
    });

    page.wrapper.find('#hd-ticket-clear').on('click', function() {
        page.wrapper.find('#hd-ticket-subject').val('');
        page.wrapper.find('#hd-ticket-description').val('');
        page.wrapper.find('#hd-ticket-unknown-repo').hide();
        page.wrapper.find('#hd-ticket-result-card').hide();
    });

    page.wrapper.find('#hd-ticket-continue').on('click', function() {
        var pending = page._pending_kb;
        if (!pending) return;

        var app_name = page.wrapper.find('#hd-ticket-app-name').val().trim();
        var repo_url = page.wrapper.find('#hd-ticket-repo-url').val().trim();
        var branch = page.wrapper.find('#hd-ticket-branch').val().trim() || 'main';

        if (!app_name || !repo_url) {
            frappe.msgprint(__('App Name and GitHub Repo URL are required.'));
            return;
        }

        var btn = $(this);
        set_loading(btn, true);

        frappe.call({
            method: 'ampower_koda.agent.hd_ticket_intake.resolve_unknown_repo',
            args: {
                subject: page.wrapper.find('#hd-ticket-subject').val().trim(),
                description: page.wrapper.find('#hd-ticket-description').val().trim(),
                request_type: pending.request_type,
                comment_for_developer: pending.comment_for_developer,
                app_name: app_name,
                repo_url: repo_url,
                branch: branch,
            },
            callback: function(r) {
                set_loading(btn, false, 'Continue');
                if (!r.message) return;
                var data = r.message;

                if (data.status === 'ok') {
                    show_result('hdti-badge-success', 'started',
                        '<a href="' + data.redirect_url + '" class="hdti-link">' +
                        __('Open Agent Request') + ' \u2192</a>');
                } else if (data.status === 'indexing') {
                    show_result('hdti-badge-warn', 'indexing', frappe.utils.escape_html(data.message));
                }
                page.wrapper.find('#hd-ticket-unknown-repo').hide();
            },
            error: function() { set_loading(btn, false, 'Continue'); }
        });
    });
};

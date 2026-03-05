frappe.pages['ai-agent'].on_page_load = function(wrapper) {
    var page = frappe.ui.make_app_page({
        parent: wrapper,
        title: __('AI Coding Agent'),
        single_column: true
    });
    page.set_primary_action(__('Submit Request'), function() {
        scrollToSubmit();
    }, 'octicon octicon-plus');
    build_agent_page(page);
};

function build_agent_page(page) {
    var container = $('<div class="ai-agent-page"></div>').appendTo(page.body);

    container.append(`
        <div class="ai-agent-form-section">
            <h5>New Request</h5>
            <p class="text-muted">Describe a bug or feature. The agent will read the codebase, create a branch, implement changes, and open a pull request.</p>
            <div class="form-group">
                <label>Request type</label>
                <select id="agent-request-type" class="form-control">
                    <option value="Bug Fix">Bug Fix</option>
                    <option value="Feature Request">Feature Request</option>
                    <option value="Improvement">Improvement</option>
                </select>
            </div>
            <div class="form-group">
                <label>Description</label>
                <textarea id="agent-message-input" class="form-control" rows="4" placeholder="Describe the bug or feature..."></textarea>
            </div>
            <button type="button" class="btn btn-primary btn-lg" id="agent-submit-btn">Submit to Agent</button>
        </div>

        <div class="ai-agent-status-section mt-4">
            <h5>Live Progress</h5>
            <div id="agent-status-message" class="alert alert-info">Submit a request to start.</div>
            <div id="agent-stage-timeline"></div>
            <div id="agent-pr-link-wrap" class="hidden mt-2"></div>
        </div>

        <div class="ai-agent-history-section mt-5">
            <h5>Recent Requests</h5>
            <div id="agent-history-list" class="list-group"></div>
        </div>
    `);

    container.addClass('container').css({ maxWidth: '780px', marginTop: '20px' });

    $(`<style>
        .ai-agent-page .form-group label { font-weight: 600; }
        .ai-agent-page .ai-agent-form-section { padding: 16px 0; }
        .stage-timeline { margin: 12px 0; }
        .stage-entry {
            display: flex;
            align-items: flex-start;
            padding: 8px 12px;
            margin-bottom: 4px;
            border-radius: 6px;
            background: var(--subtle-fg);
            border-left: 3px solid var(--gray-400);
            font-size: 13px;
        }
        .stage-entry.stage-started { border-left-color: var(--blue-500); background: var(--blue-50, #eff6ff); }
        .stage-entry.stage-completed { border-left-color: var(--green-500); background: var(--green-50, #f0fdf4); }
        .stage-entry.stage-failed { border-left-color: var(--red-500); background: var(--red-50, #fef2f2); }
        .stage-entry.stage-progress { border-left-color: var(--yellow-500); background: var(--yellow-50, #fefce8); }
        .stage-entry .stage-icon { margin-right: 8px; font-size: 14px; }
        .stage-entry .stage-name { font-weight: 600; margin-right: 6px; }
        .stage-entry .stage-time { color: var(--gray-600); margin-left: auto; font-size: 11px; white-space: nowrap; }
        .stage-entry .stage-msg { color: var(--gray-700); margin-top: 2px; font-size: 12px; }
    </style>`).appendTo(container);

    $('#agent-submit-btn').on('click', function() {
        sendRequest(page);
    });

    frappe.realtime.on('agent_progress', function(data) {
        if (!data || !data.request_name) return;
        var statusEl = $('#agent-status-message');
        var prWrap = $('#agent-pr-link-wrap');

        statusEl.removeClass('alert-info alert-success alert-danger alert-warning').addClass('alert-info');

        var statusText = data.status || '';
        if (data.message) statusText += ': ' + data.message;
        statusEl.text(statusText);

        if (data.stage && data.stage_status) {
            addStageEntry(data.stage, data.stage_status, data.message || '');
        }

        if (data.status === 'Completed') {
            statusEl.removeClass('alert-info').addClass('alert-success');
        }
        if (data.status === 'Failed') {
            statusEl.removeClass('alert-info').addClass('alert-danger');
        }

        if (data.pr_url) {
            prWrap.removeClass('hidden').html(
                '<a href="' + frappe.escape_html(data.pr_url) + '" target="_blank" class="btn btn-primary">Open Pull Request</a>'
            );
        }

        loadHistory();
    });

    loadHistory();
}

function addStageEntry(stage, status, message) {
    var timeline = $('#agent-stage-timeline');
    var icons = {
        'started': '⏳',
        'completed': '✅',
        'failed': '❌',
        'progress': '🔄'
    };
    var icon = icons[status] || '•';
    var now = new Date().toLocaleTimeString();

    var existingStarted = timeline.find('.stage-entry.stage-started[data-stage="' + stage + '"]');
    if (status === 'completed' || status === 'failed') {
        existingStarted.remove();
    }

    var entry = $(`
        <div class="stage-entry stage-${status}" data-stage="${stage}">
            <span class="stage-icon">${icon}</span>
            <div>
                <span class="stage-name">${frappe.escape_html(stage)}</span>
                <div class="stage-msg">${frappe.escape_html(message).substring(0, 200)}</div>
            </div>
            <span class="stage-time">${now}</span>
        </div>
    `);
    timeline.append(entry);
}

function sendRequest(page) {
    var message = ($('#agent-message-input').val() || '').trim();
    if (!message) {
        frappe.msgprint(__('Please enter a description.'));
        return;
    }
    var requestType = $('#agent-request-type').val() || 'Improvement';
    var btn = $('#agent-submit-btn');
    btn.prop('disabled', true);

    $('#agent-stage-timeline').empty();
    $('#agent-status-message').removeClass('hidden alert-success alert-danger').addClass('alert-info').text(__('Creating request...'));
    $('#agent-pr-link-wrap').addClass('hidden').empty();

    frappe.call({
        method: 'ampower_ai_agents.agent.api.create_agent_request',
        args: { message: message, request_type: requestType },
        callback: function(r) {
            btn.prop('disabled', false);
            if (r.exc) {
                $('#agent-status-message').removeClass('alert-info').addClass('alert-danger').text(__('Error: ') + (r.message || r.exc));
                return;
            }
            $('#agent-status-message').text(__('Request created. Agent is running in the background...'));
            addStageEntry('Queued', 'started', 'Request submitted, agent starting up');
            $('#agent-message-input').val('');
            loadHistory();
        }
    });
}

function loadHistory() {
    frappe.call({
        method: 'ampower_ai_agents.agent.api.get_agent_history',
        args: { limit: 15 },
        callback: function(r) {
            var list = $('#agent-history-list');
            list.empty();
            if (!r.message || !r.message.length) {
                list.append('<div class="list-group-item text-muted">No requests yet.</div>');
                return;
            }
            r.message.forEach(function(item) {
                var statusBadge = getStatusBadge(item.status);
                var link = $(`
                    <a href="#" class="list-group-item list-group-item-action d-flex justify-content-between align-items-center">
                        <div>
                            <strong>${frappe.escape_html(item.name)}</strong> – ${frappe.escape_html(item.request_title || '')}
                            <br><small class="text-muted">${frappe.escape_html(item.request_type || '')} · ${frappe.escape_html(item.creation || '')}</small>
                        </div>
                        <span>${statusBadge}</span>
                    </a>
                `);
                if (item.pr_url) {
                    link.find('div').append(
                        $('<br><a class="btn btn-xs btn-primary mt-1" target="_blank"></a>').attr('href', item.pr_url).text('Open PR')
                    );
                }
                link.on('click', function(e) {
                    if ($(e.target).is('a.btn')) return;
                    e.preventDefault();
                    frappe.set_route('Form', 'AI Agent Request', item.name);
                });
                list.append(link);
            });
        }
    });
}

function getStatusBadge(status) {
    var colors = {
        'Queued': 'secondary',
        'Understanding': 'info',
        'Planning': 'info',
        'Implementing': 'warning',
        'Reviewing': 'warning',
        'Pushing': 'primary',
        'Completed': 'success',
        'Failed': 'danger',
        'Cancelled': 'secondary'
    };
    var color = colors[status] || 'secondary';
    return '<span class="badge badge-' + color + '">' + frappe.escape_html(status) + '</span>';
}

function scrollToSubmit() {
    var el = document.getElementById('agent-message-input');
    if (el) el.scrollIntoView({ behavior: 'smooth' });
}

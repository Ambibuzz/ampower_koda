/* global ace */

frappe.pages['agent-diff-viewer'].on_page_load = function (wrapper) {
    const route = frappe.get_route();
    const request_name = route[1];

    const page = frappe.ui.make_app_page({
        parent: wrapper,
        title: __('Agent IDE'),
        single_column: true,
    });

    const ide = {
        request_name: request_name,
        context: {},
        treeData: null,
        files: {},
        activePath: null,
        editor: null,
        aceReady: null,
        viewMode: 'code',
        dirtyPaths: new Set(),
    };

    inject_ide_styles();
    const $shell = build_ide_shell(page, ide);
    page.main.append($shell);
    bind_ide_shortcuts(ide, $shell);

    if (!request_name) {
        show_ide_error($shell, __('Open this page from an Agent Request.'));
        return;
    }

    load_ace().then(function () {
        init_editor($shell, ide);
        load_workspace(request_name, $shell, ide);
    });
};

function load_ace() {
    const root = frappe.boot.developer_mode
        ? '/assets/frappe/node_modules/ace-builds/src-noconflict/'
        : '/assets/frappe/node_modules/ace-builds/src-min-noconflict/';
    return new Promise(function (resolve) {
        frappe.require(root + 'ace.js', function () {
            window.ace.config.set('basePath', root);
            resolve();
        });
    });
}

function inject_ide_styles() {
    let style = document.getElementById('agent-ide-styles');
    if (!style) {
        style = document.createElement('style');
        style.id = 'agent-ide-styles';
        document.head.appendChild(style);
    }
    style.textContent = `
        .koda-ide {
            --bg: #0d1117;
            --panel: #161b22;
            --panel-2: #1c2129;
            --hover: #21262d;
            --border: #30363d;
            --text: #e6edf3;
            --muted: #8b949e;
            --accent: #388bfd;
            --accent-hover: #4d9bff;
            --green: #3fb950;
            --red: #f85149;
            --amber: #e3b341;
        }
        .koda-ide.koda-light {
            --bg: #ffffff;
            --panel: #f6f8fa;
            --panel-2: #eef1f4;
            --hover: #eaeef2;
            --border: #d0d7de;
            --text: #1f2328;
            --muted: #656d76;
            --accent: #0969da;
            --accent-hover: #0860ca;
            --green: #1a7f37;
            --red: #cf222e;
            --amber: #9a6700;
        }
        .layout-main-section .page-content:has(.koda-ide) {
            padding: 0 !important;
        }
        .koda-ide {
            display: flex;
            flex-direction: column;
            height: calc(100vh - 108px);
            min-height: 600px;
            border: 1px solid var(--border);
            border-radius: 6px;
            overflow: hidden;
            background: var(--bg);
            color: var(--text);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }
        .koda-ide button {
            font-family: inherit;
            cursor: pointer;
            opacity: 1;
        }
        .koda-ide button:disabled {
            opacity: 0.45;
            cursor: not-allowed;
        }
        .koda-btn {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 4px;
            font-size: 12px;
            font-weight: 500;
            line-height: 1;
            padding: 6px 12px;
            border-radius: 5px;
            border: 1px solid var(--border);
            background: var(--panel);
            color: var(--text);
            opacity: 1;
            transition: background 0.1s, border-color 0.1s;
        }
        .koda-btn:hover:not(:disabled) {
            background: var(--hover);
            border-color: var(--border);
            color: var(--text);
            opacity: 1;
        }
        .koda-btn-primary {
            background: var(--accent);
            border-color: var(--accent);
            color: #fff;
        }
        .koda-btn-primary:hover:not(:disabled) {
            background: var(--accent-hover);
            border-color: var(--accent-hover);
            color: #fff;
            opacity: 1;
        }
        .koda-btn-ghost {
            background: transparent;
            border-color: var(--border);
            color: var(--text);
        }
        .koda-btn-sm {
            font-size: 11px;
            padding: 4px 8px;
        }
        .koda-toolbar {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 8px 12px;
            background: var(--panel);
            border-bottom: 1px solid var(--border);
            flex-wrap: wrap;
        }
        .koda-toolbar .koda-title {
            font-size: 13px;
            font-weight: 600;
            color: var(--text);
        }
        .koda-toolbar .koda-meta {
            font-size: 11px;
            color: var(--muted);
            margin-right: auto;
            max-width: 40%;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .koda-toolbar .koda-meta b { color: var(--text); font-weight: 600; }
        .koda-toolbar-actions {
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .koda-sep {
            width: 1px;
            height: 20px;
            background: var(--border);
            flex-shrink: 0;
        }
        .koda-view-group {
            display: inline-flex;
            border: 1px solid var(--border);
            border-radius: 5px;
            overflow: hidden;
        }
        .koda-view-group .koda-btn {
            border: none;
            border-radius: 0;
            border-right: 1px solid var(--border);
        }
        .koda-view-group .koda-btn:last-child { border-right: none; }
        .koda-view-group .koda-btn.active {
            background: var(--accent);
            color: #fff;
        }
        .koda-view-group .koda-btn.active:hover:not(:disabled) {
            background: var(--accent-hover);
            color: #fff;
            opacity: 1;
        }
        .koda-body {
            display: flex;
            flex: 1;
            min-height: 0;
        }
        .koda-sidebar {
            width: 240px;
            min-width: 160px;
            max-width: 360px;
            background: var(--panel);
            border-right: 1px solid var(--border);
            overflow: auto;
            font-size: 12px;
            resize: horizontal;
        }
        .koda-sidebar-title {
            padding: 8px 12px;
            font-size: 11px;
            font-weight: 600;
            color: var(--muted);
            border-bottom: 1px solid var(--border);
            position: sticky;
            top: 0;
            background: var(--panel);
            z-index: 1;
        }
        .koda-main {
            flex: 1;
            min-width: 0;
            display: flex;
            flex-direction: column;
            background: var(--bg);
        }
        .koda-tabs {
            display: flex;
            background: var(--panel);
            border-bottom: 1px solid var(--border);
            min-height: 32px;
            overflow-x: auto;
        }
        .koda-tab {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 7px 10px 7px 12px;
            font-size: 12px;
            color: var(--muted);
            border-right: 1px solid var(--border);
            cursor: pointer;
            white-space: nowrap;
        }
        .koda-tab:hover { background: var(--hover); color: var(--text); }
        .koda-tab.active { background: var(--bg); color: var(--text); }
        .koda-tab-label { pointer-events: none; }
        .koda-tab.dirty .koda-tab-label::after { content: ' *'; color: var(--amber); }
        .koda-tab-close {
            display: none;
            width: 16px;
            height: 16px;
            align-items: center;
            justify-content: center;
            border-radius: 3px;
            font-size: 14px;
            line-height: 1;
            color: var(--muted);
            flex-shrink: 0;
        }
        .koda-tab:hover .koda-tab-close { display: inline-flex; }
        .koda-tab-close:hover {
            background: var(--hover);
            color: var(--text);
        }
        .koda-editor-wrap {
            flex: 1;
            min-height: 0;
            position: relative;
        }
        .koda-editor-empty {
            position: absolute;
            inset: 0;
            display: flex;
            align-items: center;
            justify-content: center;
            color: var(--muted);
            font-size: 13px;
            pointer-events: none;
            z-index: 1;
        }
        .koda-editor-wrap.has-file .koda-editor-empty { display: none; }
        .koda-breadcrumb {
            padding: 5px 12px;
            font-size: 11px;
            color: var(--muted);
            background: var(--panel-2);
            border-bottom: 1px solid var(--border);
            font-family: ui-monospace, Menlo, monospace;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .koda-ace-target { position: absolute; inset: 0; }
        .koda-diff-wrap {
            flex: 1;
            min-height: 0;
            overflow: auto;
            display: none;
            background: var(--bg);
        }
        .koda-diff-only .koda-editor-wrap { display: none; }
        .koda-diff-only .koda-diff-wrap { display: block; flex: 1; }
        .koda-diff-table {
            width: 100%;
            border-collapse: collapse;
            font-family: ui-monospace, Menlo, monospace;
            font-size: 12px;
        }
        .koda-diff-table td {
            padding: 1px 10px;
            vertical-align: top;
            white-space: pre-wrap;
            word-break: break-word;
            line-height: 1.45;
        }
        .koda-line-no {
            width: 44px;
            text-align: right;
            color: var(--muted);
            border-right: 1px solid var(--border);
            user-select: none;
            background: var(--panel);
        }
        .koda-line-add { background: rgba(46, 160, 67, 0.12); }
        .koda-line-add td:last-child { color: var(--green); }
        .koda-line-del { background: rgba(248, 81, 73, 0.12); }
        .koda-line-del td:last-child { color: var(--red); }
        .koda-line-hunk { background: rgba(56, 139, 253, 0.08); color: var(--accent); }
        .koda-terminal {
            height: 120px;
            min-height: 72px;
            max-height: 40vh;
            background: var(--panel);
            border-top: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            resize: vertical;
            overflow: hidden;
        }
        .koda-terminal.collapsed {
            height: 28px !important;
            min-height: 28px;
            resize: none;
        }
        .koda-terminal.collapsed .koda-terminal-body { display: none; }
        .koda-terminal-header {
            padding: 4px 10px;
            font-size: 11px;
            color: var(--muted);
            background: var(--panel-2);
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .koda-terminal-body {
            flex: 1;
            overflow: auto;
            padding: 8px 10px;
            font-family: ui-monospace, Menlo, monospace;
            font-size: 11px;
            line-height: 1.5;
            color: var(--muted);
            white-space: pre-wrap;
        }
        .koda-tree-row {
            display: flex;
            align-items: center;
            gap: 6px;
            padding: 4px 10px;
            cursor: pointer;
            color: var(--text);
        }
        .koda-tree-row:hover { background: var(--hover); }
        .koda-tree-row.selected { background: var(--hover); font-weight: 500; }
        .koda-tree-row.changed { color: var(--amber); }
        .koda-tree-row.dimmed { color: var(--muted); }
        .koda-tree-row .koda-tree-name {
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .koda-badge {
            margin-left: auto;
            font-size: 9px;
            font-weight: 600;
            padding: 1px 4px;
            border-radius: 3px;
            flex-shrink: 0;
        }
        .koda-badge-M { background: rgba(227, 179, 65, 0.15); color: var(--amber); }
        .koda-badge-A { background: rgba(63, 185, 80, 0.15); color: var(--green); }
        .koda-badge-D { background: rgba(248, 81, 73, 0.15); color: var(--red); }
        .koda-empty {
            padding: 20px;
            text-align: center;
            color: var(--muted);
            font-size: 12px;
        }
        .koda-statusbar {
            padding: 4px 12px;
            font-size: 11px;
            color: var(--muted);
            background: var(--panel);
            border-top: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            gap: 12px;
        }
        .koda-statusbar .koda-status-path {
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            font-family: ui-monospace, Menlo, monospace;
        }
    `;
}

function build_ide_shell(page, ide) {
    const $shell = $(`
        <div class="koda-ide">
            <div class="koda-toolbar">
                <span class="koda-title">${__('Agent IDE')}</span>
                <span class="koda-meta koda-toolbar-meta">${__('Loading...')}</span>
                <div class="koda-toolbar-actions">
                    <button type="button" class="koda-btn koda-btn-primary koda-btn-save" disabled title="Ctrl+S">${__('Save')}</button>
                    <span class="koda-sep"></span>
                    <button type="button" class="koda-btn koda-btn-ghost koda-btn-deploy">${__('Deploy')}</button>
                    <button type="button" class="koda-btn koda-btn-ghost koda-btn-push">${__('Push')}</button>
                    <span class="koda-sep"></span>
                    <div class="koda-view-group">
                        <button type="button" class="koda-btn active koda-view-btn" data-mode="code">${__('Code')}</button>
                        <button type="button" class="koda-btn koda-view-btn" data-mode="diff">${__('Diff')}</button>
                    </div>
                    <span class="koda-sep"></span>
                    <button type="button" class="koda-btn koda-btn-ghost koda-theme-btn" title="${__('Toggle theme')}">🌙</button>
                </div>
            </div>
            <div class="koda-body">
                <div class="koda-sidebar">
                    <div class="koda-sidebar-title">${__('Files')}</div>
                    <div class="koda-tree"><div class="koda-empty">${__('Loading...')}</div></div>
                </div>
                <div class="koda-main">
                    <div class="koda-tabs"></div>
                    <div class="koda-breadcrumb koda-breadcrumb-path"></div>
                    <div class="koda-editor-wrap">
                        <div class="koda-editor-empty">${__('Select a file to edit')}</div>
                        <div class="koda-ace-target"></div>
                    </div>
                    <div class="koda-diff-wrap"><div class="koda-diff-content"></div></div>
                    <div class="koda-statusbar">
                        <span class="koda-status-text">${__('Ready')}</span>
                        <span class="koda-status-path"></span>
                    </div>
                </div>
            </div>
            <div class="koda-terminal">
                <div class="koda-terminal-header">
                    <span>${__('Output')}</span>
                    <span>
                        <button type="button" class="koda-btn koda-btn-ghost koda-btn-sm koda-toggle-terminal">${__('Hide')}</button>
                        <button type="button" class="koda-btn koda-btn-ghost koda-btn-sm koda-clear-terminal">${__('Clear')}</button>
                    </span>
                </div>
                <div class="koda-terminal-body"></div>
            </div>
        </div>
    `);

    page.add_button(__('Back to Request'), function () {
        frappe.set_route('Form', 'Agent Request', ide.request_name);
    });

    page.add_button(__('Refresh'), function () {
        load_workspace(ide.request_name, $shell, ide);
    });

    $shell.find('.koda-btn-save').on('click', function () {
        save_active_file($shell, ide);
    });

    $shell.find('.koda-btn-deploy').on('click', function () {
        open_deploy_dialog(ide, $shell);
    });

    $shell.find('.koda-btn-push').on('click', function () {
        open_push_dialog(ide, $shell);
    });

    $shell.find('.koda-view-btn').on('click', function () {
        $shell.find('.koda-view-btn').removeClass('active');
        $(this).addClass('active');
        ide.viewMode = $(this).data('mode');
        apply_view_mode($shell, ide);
    });

    $shell.find('.koda-clear-terminal').on('click', function () {
        $shell.find('.koda-terminal-body').empty();
    });

    $shell.find('.koda-toggle-terminal').on('click', function () {
        const $term = $shell.find('.koda-terminal');
        $term.toggleClass('collapsed');
        $(this).text($term.hasClass('collapsed') ? __('Show') : __('Hide'));
        if (ide.editor) setTimeout(function () { ide.editor.resize(); }, 50);
    });

    $shell.find('.koda-theme-btn').on('click', function () {
        const next = ide.theme === 'light' ? 'dark' : 'light';
        apply_theme(next, $shell, ide);
    });

    apply_theme(get_saved_theme(), $shell, ide);

    return $shell;
}

function get_saved_theme() {
    try {
        return localStorage.getItem('koda_ide_theme') === 'light' ? 'light' : 'dark';
    } catch (e) {
        return 'dark';
    }
}

function apply_theme(theme, $shell, ide) {
    ide.theme = theme === 'light' ? 'light' : 'dark';
    const $ide = $shell.hasClass('koda-ide') ? $shell : $shell.find('.koda-ide');
    $ide.toggleClass('koda-light', ide.theme === 'light');

    const $btn = $shell.find('.koda-theme-btn');
    $btn.text(ide.theme === 'light' ? '☀️' : '🌙');
    $btn.attr('title', ide.theme === 'light' ? __('Switch to dark') : __('Switch to light'));

    if (ide.editor) {
        ide.editor.setTheme(ide.theme === 'light' ? 'ace/theme/github' : 'ace/theme/one_dark');
    }

    try {
        localStorage.setItem('koda_ide_theme', ide.theme);
    } catch (e) { /* ignore */ }
}

function bind_ide_shortcuts(ide, $shell) {
    $(document).on('keydown.koda-ide', function (e) {
        if (!ide.activePath) return;
        if ((e.ctrlKey || e.metaKey) && e.key === 's') {
            e.preventDefault();
            save_active_file($shell, ide);
        }
    });
}

function init_editor($shell, ide) {
    const target = $shell.find('.koda-ace-target').get(0);
    ide.editor = ace.edit(target);
    ide.editor.setTheme(ide.theme === 'light' ? 'ace/theme/github' : 'ace/theme/one_dark');
    ide.editor.setShowPrintMargin(false);
    ide.editor.setOptions({
        fontSize: 13,
        wrap: true,
        enableBasicAutocompletion: true,
        highlightActiveLine: true,
        highlightGutterLine: true,
        showFoldWidgets: true,
        tabSize: 4,
        useSoftTabs: true,
        scrollPastEnd: 0.5,
    });
    ide.editor.setKeyboardHandler('ace/keyboard/vscode');
    ide.editor.session.on('change', frappe.utils.debounce(function () {
        if (!ide.activePath || ide._setting_content) return;
        const file = ide.files[ide.activePath];
        if (!file) return;
        file.content = ide.editor.getValue();
        file.dirty = file.content !== file.original;
        if (file.dirty) {
            ide.dirtyPaths.add(ide.activePath);
        } else {
            ide.dirtyPaths.delete(ide.activePath);
        }
        update_tabs($shell, ide);
        update_save_button($shell, ide);
        set_status($shell, file.dirty ? __('Unsaved changes — press Save or Ctrl+S') : __('Ready'), file.full_path || ide.activePath);
    }, 200));
    ide.editor.resize();
}

function load_workspace(request_name, $shell, ide) {
    ide.request_name = request_name;
    $shell.find('.koda-tree').html(`<div class="koda-empty">${__('Loading...')}</div>`);

    frappe.call({
        method: 'ampower_koda.agent.api.get_change_tree',
        args: { request_name: request_name },
        callback: function (r) {
            if (!r.message) {
                show_ide_error($shell, __('Could not load workspace.'));
                return;
            }
            ide.context = r.message;
            ide.treeData = r.message.tree;
            render_toolbar_meta($shell, r.message);
            render_tree($shell, ide, r.message);
            const first = find_first_changed_file(r.message.tree || []);
            if (first) {
                open_file(first.path, $shell, ide);
            }
        },
    });
}

function render_toolbar_meta($shell, data) {
    const branch = data.branch_name || '—';
    const count = (data.totals && data.totals.files) || Object.keys(data.changed || {}).length;
    const status = (data.request && data.request.status) || '';
    $shell.find('.koda-toolbar-meta').html(
        `${frappe.utils.escape_html(data.app_name || '')} · `
        + `${count} ${__('changed')} · `
        + `${__('branch')}: <b>${frappe.utils.escape_html(branch)}</b> · `
        + `${__('status')}: ${frappe.utils.escape_html(status)}`
    );
}

function render_tree($shell, ide, data) {
    const $tree = $shell.find('.koda-tree').empty();
    if (!data.tree || !data.tree.length) {
        $tree.html(`<div class="koda-empty">${__('No files found.')}</div>`);
        return;
    }
    const $root = $('<div></div>');
    data.tree.forEach(function (node) {
        $root.append(render_tree_node(node, 0, data.changed || {}, $shell, ide));
    });
    $tree.append($root);
}

function render_tree_node(node, depth, changed_map, $shell, ide) {
    const indent = depth * 12;
    const $item = $('<div></div>');

    if (node.type === 'folder') {
        const expanded = node.has_changes;
        const $row = $(`
            <div class="koda-tree-row ${node.has_changes ? 'changed' : 'dimmed'}" style="padding-left:${8 + indent}px">
                <span class="koda-caret">${expanded ? '▾' : '▸'}</span>
                <span class="koda-tree-name">${frappe.utils.escape_html(node.name)}</span>
                ${node.changed_count ? `<span class="koda-badge koda-badge-M">${node.changed_count}</span>` : ''}
            </div>
        `);
        const $children = $('<div></div>');
        if (!expanded) $children.hide();
        (node.children || []).forEach(function (child) {
            $children.append(render_tree_node(child, depth + 1, changed_map, $shell, ide));
        });
        $row.on('click', function (e) {
            e.stopPropagation();
            const open = $children.is(':visible');
            $children.toggle(!open);
            $row.find('.koda-caret').text(open ? '▸' : '▾');
        });
        $item.append($row, $children);
        return $item;
    }

    const status = node.status || '';
    const dirty = ide.dirtyPaths.has(node.path);
    const $row = $(`
        <div class="koda-tree-row ${node.has_changes ? 'changed' : 'dimmed'} ${dirty ? 'dirty' : ''}"
             data-path="${frappe.utils.escape_html(node.path)}"
             style="padding-left:${12 + indent}px">
            <span class="koda-tree-name">${frappe.utils.escape_html(node.name)}${dirty ? ' •' : ''}</span>
            ${status ? `<span class="koda-badge koda-badge-${status}">${status}</span>` : ''}
        </div>
    `);
    $row.on('click', function () {
        open_file(node.path, $shell, ide);
    });
    $item.append($row);
    return $item;
}

function open_file(file_path, $shell, ide, force) {
    if (!file_path) return;

    const proceed = function () {
        do_open_file(file_path, $shell, ide);
    };

    if (!force && ide.activePath && ide.files[ide.activePath] && ide.files[ide.activePath].dirty) {
        frappe.confirm(
            __('You have unsaved changes in the current file. Open another file anyway?'),
            proceed
        );
        return;
    }
    proceed();
}

function do_open_file(file_path, $shell, ide) {
    if (ide.activePath && ide.files[ide.activePath]) {
        ide.files[ide.activePath].content = ide.editor.getValue();
    }

    ide.activePath = file_path;
    $shell.find('.koda-tree-row').removeClass('selected');
    $shell.find('.koda-tree-row').filter(function () {
        return $(this).attr('data-path') === file_path;
    }).addClass('selected');

    const loadAndShow = function (content, language, full_path) {
        ide.files[file_path] = ide.files[file_path] || {};
        const cached = ide.files[file_path];
        if (cached.content !== undefined && cached.dirty) {
            set_editor_content(ide, cached.content, language);
        } else {
            cached.original = content;
            cached.content = content;
            cached.dirty = false;
            cached.language = language;
            cached.full_path = full_path || cached.full_path;
            set_editor_content(ide, content, language);
        }
        $shell.find('.koda-editor-wrap').addClass('has-file');
        update_breadcrumb($shell, file_path, cached.full_path);
        update_tabs($shell, ide);
        update_save_button($shell, ide);
        set_status($shell, cached.full_path || file_path, cached.full_path || file_path);
        load_file_diff(file_path, $shell, ide);
    };

    if (ide.files[file_path] && ide.files[file_path].content !== undefined) {
        const cached = ide.files[file_path];
        loadAndShow(cached.content, cached.language, cached.full_path);
        return;
    }

    frappe.call({
        method: 'ampower_koda.agent.api.get_file_content',
        args: { request_name: ide.request_name, file_path: file_path },
        callback: function (r) {
            if (!r.message) return;
            loadAndShow(r.message.content || '', r.message.language || 'Text', r.message.full_path);
        },
        error: function () {
            append_terminal($shell, __('Failed to load file: ') + file_path);
        },
    });
}

function set_editor_content(ide, content, language) {
    const mode_map = {
        Python: 'ace/mode/python',
        Javascript: 'ace/mode/javascript',
        JSON: 'ace/mode/json',
        HTML: 'ace/mode/html',
        CSS: 'ace/mode/css',
        Markdown: 'ace/mode/markdown',
    };
    ide._setting_content = true;
    ide.editor.session.setValue(content || '');
    ide.editor.session.setMode(mode_map[language] || 'ace/mode/text');
    ide._setting_content = false;
    ide.editor.resize();
    ide.editor.focus();
}

function format_display_path(path) {
    if (!path) return '';
    const prefix = '/opt/bench/frappe-bench/apps/';
    if (path.startsWith(prefix)) {
        return path.slice(prefix.length);
    }
    return path;
}

function update_tabs($shell, ide) {
    const $tabs = $shell.find('.koda-tabs').empty();
    Object.keys(ide.files).forEach(function (path) {
        const file = ide.files[path];
        const name = path.split('/').pop();
        const $tab = $(`
            <div class="koda-tab ${path === ide.activePath ? 'active' : ''} ${file.dirty ? 'dirty' : ''}" data-path="${frappe.utils.escape_html(path)}">
                <span class="koda-tab-label">${frappe.utils.escape_html(name)}</span>
                <span class="koda-tab-close" title="${__('Close')}">×</span>
            </div>
        `);
        $tab.on('click', function () {
            open_file(path, $shell, ide);
        });
        $tab.find('.koda-tab-close').on('click', function (e) {
            e.stopPropagation();
            close_tab(path, $shell, ide);
        });
        $tabs.append($tab);
    });
}

function close_tab(file_path, $shell, ide) {
    const file = ide.files[file_path];
    const do_close = function () {
        const was_active = ide.activePath === file_path;
        delete ide.files[file_path];
        ide.dirtyPaths.delete(file_path);

        if (was_active) {
            const remaining = Object.keys(ide.files);
            if (remaining.length) {
                open_file(remaining[remaining.length - 1], $shell, ide, true);
            } else {
                ide.activePath = null;
                ide._setting_content = true;
                ide.editor.session.setValue('');
                ide._setting_content = false;
                $shell.find('.koda-editor-wrap').removeClass('has-file');
                update_breadcrumb($shell, '', '');
                set_status($shell, __('Ready'), '');
                $shell.find('.koda-diff-content').empty();
            }
        }
        update_tabs($shell, ide);
        $shell.find('.koda-tree-row').removeClass('selected');
        if (ide.activePath) {
            $shell.find('.koda-tree-row').filter(function () {
                return $(this).attr('data-path') === ide.activePath;
            }).addClass('selected');
        }
    };

    if (file && file.dirty) {
        frappe.confirm(__('Discard unsaved changes in this file?'), do_close);
        return;
    }
    do_close();
}

function save_active_file($shell, ide) {
    const path = ide.activePath;
    if (!path || !ide.files[path] || !ide.files[path].dirty) return;

    const content = ide.editor.getValue();
    $shell.find('.koda-btn-save').prop('disabled', true);
    set_status($shell, __('Saving to disk...'), path);

    frappe.call({
        method: 'ampower_koda.agent.api.save_file_content',
        args: {
            request_name: ide.request_name,
            file_path: path,
            content: content,
        },
        callback: function (r) {
            if (r.message && r.message.status === 'ok') {
                const full_path = r.message.full_path || path;
                ide.files[path].original = content;
                ide.files[path].dirty = false;
                ide.files[path].full_path = full_path;
                ide.dirtyPaths.delete(path);
                append_terminal($shell, __('Saved to disk: ') + format_display_path(full_path));
                frappe.show_alert({
                    message: __('File saved to codebase'),
                    indicator: 'green',
                });
                set_status($shell, __('Saved'), full_path);
                update_breadcrumb($shell, path, full_path);
                update_tabs($shell, ide);
                update_save_button($shell, ide);
                load_file_diff(path, $shell, ide);
                refresh_tree_badges($shell, ide);
                // Re-read from disk so editor matches what is on the server
                frappe.call({
                    method: 'ampower_koda.agent.api.get_file_content',
                    args: { request_name: ide.request_name, file_path: path },
                    callback: function (reload) {
                        if (!reload.message) return;
                        ide.files[path].original = reload.message.content;
                        ide.files[path].content = reload.message.content;
                        ide.files[path].full_path = reload.message.full_path;
                        set_editor_content(ide, reload.message.content, reload.message.language);
                    },
                });
            }
        },
        error: function (xhr) {
            const msg = (xhr.responseJSON && xhr.responseJSON._server_messages)
                ? JSON.parse(xhr.responseJSON._server_messages).map(function (m) {
                    return JSON.parse(m).message;
                }).join(' ')
                : (xhr.message || path);
            append_terminal($shell, __('Save failed: ') + msg);
            update_save_button($shell, ide);
        },
    });
}

function update_save_button($shell, ide) {
    const path = ide.activePath;
    const dirty = path && ide.files[path] && ide.files[path].dirty;
    $shell.find('.koda-btn-save').prop('disabled', !dirty);
}

function update_breadcrumb($shell, rel_path, full_path) {
    $shell.find('.koda-breadcrumb-path').text(format_display_path(full_path || rel_path || ''));
}

function load_file_diff(file_path, $shell, ide) {
    frappe.call({
        method: 'ampower_koda.agent.api.get_file_diff',
        args: { request_name: ide.request_name, file_path: file_path },
        callback: function (r) {
            const diff = (r.message && r.message.diff) || '';
            render_diff_panel($shell, diff);
        },
    });
}

function render_diff_panel($shell, diff_text) {
    const $content = $shell.find('.koda-diff-content').empty();
    if (!diff_text.trim()) {
        $content.html(`<div class="koda-empty">${__('No diff for this file.')}</div>`);
        return;
    }
    const parsed = parse_unified_diff(diff_text);
    const $table = $('<table class="koda-diff-table"><tbody></tbody></table>');
    const $tbody = $table.find('tbody');
    parsed.rows.forEach(function (row) {
        const cls = { add: 'koda-line-add', del: 'koda-line-del', hunk: 'koda-line-hunk' }[row.type] || '';
        const prefix = row.type === 'add' ? '+' : row.type === 'del' ? '-' : ' ';
        $tbody.append(`
            <tr class="${cls}">
                <td class="koda-line-no">${row.old_no || ''}</td>
                <td class="koda-line-no">${row.new_no || ''}</td>
                <td>${prefix}${frappe.utils.escape_html(row.text)}</td>
            </tr>
        `);
    });
    $content.append($table);
}

function apply_view_mode($shell, ide) {
    const $main = $shell.find('.koda-main');
    $main.removeClass('koda-diff-only');
    if (ide.viewMode === 'diff') {
        $main.addClass('koda-diff-only');
        if (ide.activePath) {
            load_file_diff(ide.activePath, $shell, ide);
        }
    }
    if (ide.editor) {
        setTimeout(function () { ide.editor.resize(); }, 50);
    }
}

function open_deploy_dialog(ide, $shell) {
    frappe.call({
        method: 'ampower_koda.agent.api.get_default_bench_commands',
        args: { request_name: ide.request_name },
        callback: function (r) {
            const cmds = (r.message && r.message.commands) || [];
            const pending = (ide.context.request && ide.context.request.pending_bench_commands) || [];
            const list = pending.length ? pending : cmds;
            const fields = list.map(function (cmd, i) {
                return {
                    fieldtype: 'Check',
                    fieldname: 'cmd_' + i,
                    label: cmd,
                    default: 1,
                };
            });
            if (!fields.length) {
                frappe.msgprint(__('No deploy commands available.'));
                return;
            }
            const dialog = new frappe.ui.Dialog({
                title: __('Deploy (Bench Commands)'),
                fields: fields,
                primary_action_label: __('Run Deploy'),
                primary_action: function (values) {
                    const selected = [];
                    list.forEach(function (cmd, i) {
                        if (values['cmd_' + i]) selected.push(cmd);
                    });
                    if (!selected.length) {
                        frappe.msgprint(__('Select at least one command.'));
                        return;
                    }
                    dialog.hide();
                    append_terminal($shell, __('Running deploy commands...'));
                    frappe.call({
                        method: 'ampower_koda.agent.api.run_selected_bench_commands',
                        args: {
                            request_name: ide.request_name,
                            commands: JSON.stringify(selected),
                        },
                        freeze: true,
                        callback: function (res) {
                            append_terminal($shell, (res.message && res.message.log) || __('Deploy finished.'));
                        },
                    });
                },
            });
            dialog.show();
        },
    });
}

function open_push_dialog(ide, $shell) {
    const dialog = new frappe.ui.Dialog({
        title: __('Commit & Push'),
        fields: [
            { fieldname: 'push_branch', fieldtype: 'Check', label: __('Push branch to GitHub'), default: 1 },
            { fieldname: 'create_pr', fieldtype: 'Check', label: __('Create Pull Request'), default: 1 },
        ],
        primary_action_label: __('Run'),
        primary_action: function (values) {
            if (!values.push_branch && !values.create_pr) {
                frappe.msgprint(__('Select at least one action.'));
                return;
            }
            dialog.hide();
            append_terminal($shell, __('Starting commit and push...'));
            frappe.call({
                method: 'ampower_koda.agent.api.ide_push',
                args: {
                    request_name: ide.request_name,
                    push_branch: values.push_branch ? 1 : 0,
                    create_pr: values.create_pr ? 1 : 0,
                },
                freeze: true,
                callback: function (r) {
                    append_terminal($shell, (r.message && r.message.message) || __('Push started.'));
                    frappe.show_alert({ message: __('Push started in background'), indicator: 'blue' });
                },
            });
        },
    });
    dialog.show();
}

function refresh_tree_badges($shell, ide) {
    frappe.call({
        method: 'ampower_koda.agent.api.get_change_tree',
        args: { request_name: ide.request_name },
        callback: function (r) {
            if (!r.message) return;
            ide.context = r.message;
            render_toolbar_meta($shell, r.message);
            render_tree($shell, ide, r.message);
            if (ide.activePath) {
                $shell.find('.koda-tree-row').filter(function () {
                    return $(this).attr('data-path') === ide.activePath;
                }).addClass('selected');
            }
        },
    });
}

function append_terminal($shell, text) {
    const $body = $shell.find('.koda-terminal-body');
    const ts = frappe.datetime.now_datetime();
    $body.append(`[${ts}] ${text}\n`);
    $body.scrollTop($body[0].scrollHeight);
}

function set_status($shell, text, path) {
    $shell.find('.koda-status-text').text(text);
    if (path !== undefined) {
        $shell.find('.koda-status-path').text(format_display_path(path || ''));
    }
}

function show_ide_error($shell, message) {
    $shell.find('.koda-tree').html(`<div class="koda-empty">${frappe.utils.escape_html(message)}</div>`);
}

function find_first_changed_file(nodes) {
    for (const node of nodes) {
        if (node.type === 'file' && node.has_changes) return node;
        if (node.type === 'folder' && node.children) {
            const found = find_first_changed_file(node.children);
            if (found) return found;
        }
    }
    return null;
}

function parse_unified_diff(diff_text) {
    const lines = diff_text.split('\n');
    const rows = [];
    let old_line = 0;
    let new_line = 0;
    for (const raw of lines) {
        if (raw.startsWith('@@')) {
            const m = raw.match(/@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@/);
            if (m) { old_line = parseInt(m[1], 10); new_line = parseInt(m[2], 10); }
            rows.push({ type: 'hunk', old_no: '', new_no: '', text: raw });
            continue;
        }
        if (raw.startsWith('+++ ') || raw.startsWith('--- ') || raw.startsWith('diff ')) {
            rows.push({ type: 'meta', old_no: '', new_no: '', text: raw });
            continue;
        }
        if (raw.startsWith('+')) {
            rows.push({ type: 'add', old_no: '', new_no: new_line, text: raw.slice(1) });
            new_line += 1;
            continue;
        }
        if (raw.startsWith('-')) {
            rows.push({ type: 'del', old_no: old_line, new_no: '', text: raw.slice(1) });
            old_line += 1;
            continue;
        }
        rows.push({ type: 'ctx', old_no: old_line, new_no: new_line, text: raw.startsWith(' ') ? raw.slice(1) : raw });
        old_line += 1;
        new_line += 1;
    }
    return { rows: rows };
}

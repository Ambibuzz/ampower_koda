// Copyright (c) 2026, Ambibuzz Technologies LLP and contributors
// Koda Knowledge Graph — client script: interactive repository visualization
//
// NOTE ON APPROACH: plain HTML/CSS/JS collapsible tree — no external graph
// library. Cytoscape.js is permanently incompatible with this Frappe
// instance (its collection type assigns Array.prototype.move during init,
// which Frappe core already defines as non-writable/non-configurable on
// every desk page). A force-directed graph is also hard to read for
// non-developers, so this renders a nested folder/file explorer instead.
//
// STRUCTURE: nodes are grouped into a real directory tree based on each
// File node's file_path (not just its basename — many files across
// different folders share a basename, e.g. __init__.py, so grouping by
// basename alone made them indistinguishable). Folders and files are both
// collapsible "branches"; expanding a file reveals the members it defines
// (functions, classes, methods, etc. — flat, per builder.py's File->member
// "defines" edges). "calls" and "uses_doctype" edges are recorded at file
// granularity in this graph model, so each expanded file also shows
// plain-text "Calls:" / "Uses DocTypes:" lines rather than graph edges.
// Nodes with no defining file (e.g. DocTypes parsed from JSON) are grouped
// under a final "Other" top-level entry.

const KG_NODE_THRESHOLD = 500;

const KG_TYPE_COLORS = {
    'DocType': '#2563eb',        // blue
    'WhitelistedAPI': '#16a34a', // green
    'Class': '#ea580c',          // orange
    'File': '#6b7280',           // grey
    'Module': '#0891b2',         // cyan
    'Function': '#7c3aed',       // purple
    'Method': '#c026d3',         // magenta
    'Hook': '#ca8a04',           // amber
    'Import': '#94a3b8',         // slate
};
const KG_DEFAULT_COLOR = '#94a3b8';

frappe.ui.form.on('Koda Knowledge Graph', {
    refresh: function (frm) {
        render_placeholder(frm);
        constrain_raw_data_height(frm);
        if (!frm.is_new() && frm.doc.status === 'Ready') {
            load_and_render_graph(frm);
        }
    }
});

// ---------------------------------------------------------------------------
// Placeholder states for Building / Failed
// ---------------------------------------------------------------------------

function render_placeholder(frm) {
    const wrapper = frm.fields_dict.visualization_html;
    if (!wrapper) return;

    if (frm.doc.status === 'Building') {
        $(wrapper.wrapper).html(
            '<div class="text-muted" style="padding:20px;text-align:center;">'
            + 'Graph is still building\u2026</div>'
        );
    } else if (frm.doc.status === 'Failed') {
        $(wrapper.wrapper).html(
            '<div class="text-danger" style="padding:20px;text-align:center;">'
            + 'Graph build failed.'
            + (frm.doc.error_message ? '<br>' + frappe.utils.escape_html(frm.doc.error_message) : '')
            + '</div>'
        );
    }
}

function constrain_raw_data_height(frm) {
    const field = frm.fields_dict.graph_json;
    if (!field || !field.$wrapper) return;

    field.$wrapper.css({
        'max-height': '350px',
        'overflow-y': 'auto',
        'border': '1px solid var(--border-color)',
        'border-radius': '6px'
    });
}

// ---------------------------------------------------------------------------
// Fetch + render
// ---------------------------------------------------------------------------

function load_and_render_graph(frm) {
    const wrapper = frm.fields_dict.visualization_html;
    if (!wrapper) return;

    $(wrapper.wrapper).html('<div class="text-muted" style="padding:20px;text-align:center;">Loading graph\u2026</div>');

    frappe.call({
        method: 'ampower_koda.ampower_koda.doctype.koda_knowledge_graph.koda_knowledge_graph.get_graph_payload',
        args: { name: frm.doc.name },
        callback: function (r) {
            const graph = r.message;
            if (!graph) {
                $(wrapper.wrapper).html('<div class="text-muted" style="padding:20px;">No graph data available.</div>');
                return;
            }
            const node_count = Object.keys(graph.nodes || {}).length;
            if (node_count > KG_NODE_THRESHOLD) {
                render_fallback_table(wrapper, graph);
            } else {
                render_tree(wrapper, graph);
            }
        }
    });
}

// ---------------------------------------------------------------------------
// Fallback: flat searchable table, used above KG_NODE_THRESHOLD nodes
// ---------------------------------------------------------------------------

function render_fallback_table(wrapper, graph) {
    const nodes = Object.values(graph.nodes || {});
    const rows = nodes.map(function (n) {
        return '<tr>'
            + '<td>' + frappe.utils.escape_html(n.name) + '</td>'
            + '<td>' + frappe.utils.escape_html(n.type) + '</td>'
            + '<td>' + frappe.utils.escape_html(n.file_path || '') + '</td>'
            + '<td>' + (n.line_start || '') + '</td>'
            + '</tr>';
    }).join('');

    const html = '<div style="padding:10px;">'
        + '<div class="text-muted" style="margin-bottom:10px;">'
        + 'This graph has ' + nodes.length + ' nodes, above the ' + KG_NODE_THRESHOLD
        + '-node interactive rendering threshold. Showing a searchable table instead.</div>'
        + '<input type="text" class="form-control input-sm kg-table-filter" placeholder="Filter by name, type, or file\u2026" style="margin-bottom:8px;max-width:400px;">'
        + '<div style="max-height:500px;overflow:auto;">'
        + '<table class="table table-bordered table-sm kg-node-table">'
        + '<thead><tr><th>Name</th><th>Type</th><th>File</th><th>Line</th></tr></thead>'
        + '<tbody>' + rows + '</tbody>'
        + '</table></div></div>';

    $(wrapper.wrapper).html(html);

    $(wrapper.wrapper).find('.kg-table-filter').on('input', function () {
        const q = $(this).val().toLowerCase();
        $(wrapper.wrapper).find('.kg-node-table tbody tr').each(function () {
            $(this).toggle($(this).text().toLowerCase().indexOf(q) !== -1);
        });
    });
}

// ---------------------------------------------------------------------------
// Tree view (default, at or below KG_NODE_THRESHOLD nodes)
// ---------------------------------------------------------------------------

function render_tree(wrapper, graph) {
    const data = build_tree_data(graph);
    inject_tree_styles();

    const html = ['<div class="kg-tree-wrap">',
        '<div class="kg-tree-toolbar">',
        '<input type="text" class="form-control input-sm kg-tree-search" placeholder="Search by name\u2026">',
        '<button type="button" class="btn btn-xs btn-default kg-tree-expand-all">Expand all</button>',
        '<button type="button" class="btn btn-xs btn-default kg-tree-collapse-all">Collapse all</button>',
        '</div>',
        '<ul class="kg-tree-root">'
    ];

    html.push(render_folder_children(data.root));
    if (data.other.length) { html.push(render_other_group(data.other)); }

    html.push('</ul></div>');
    $(wrapper.wrapper).html(html.join(''));
    bind_tree_events(wrapper, graph);
}

// ---------------------------------------------------------------------------
// Data assembly: build a directory tree of files, each carrying its members
// and file-level relationships.
// ---------------------------------------------------------------------------

// Indexes the graph's edges by type, attaches each file's members and
// file-level relationships (calls, uses_doctype), then arranges files into
// a nested directory structure keyed by file_path segment. Nodes with no
// defining file are returned separately as "other".

function build_tree_data(graph) {
    const nodes = graph.nodes || {};
    const edges = graph.edges || [];

    const definesByFile = {};     // fileId -> [memberId, ...]
    const definedBy = {};         // memberId -> fileId
    const callsByFile = {};       // fileId -> [targetNode, ...]
    const usesDoctypeByFile = {}; // fileId -> [targetNode, ...]

    edges.forEach(function (e) {
        if (e.type === 'defines') {
            (definesByFile[e.source_id] = definesByFile[e.source_id] || []).push(e.target_id);
            definedBy[e.target_id] = e.source_id;
        } else if (e.type === 'calls') {
            const target = nodes[e.target_id];
            if (target) (callsByFile[e.source_id] = callsByFile[e.source_id] || []).push(target);
        } else if (e.type === 'uses_doctype') {
            const target = nodes[e.target_id];
            if (target) (usesDoctypeByFile[e.source_id] = usesDoctypeByFile[e.source_id] || []).push(target);
        }
    });

    const fileNodes = Object.values(nodes).filter(function (n) { return n.type === 'File'; });

    const fileEntries = fileNodes.map(function (f) {
        const memberIds = definesByFile[f.id] || [];
        const members = memberIds
            .map(function (id) { return nodes[id]; })
            .filter(Boolean)
            .sort(function (a, b) {
                return a.type === b.type ? a.name.localeCompare(b.name) : a.type.localeCompare(b.type);
            });
        return {
            node: f,
            filename: (f.file_path || f.name).split('/').pop(),
            members: members,
            calls: dedupe_by_id(callsByFile[f.id] || []),
            usesDoctypes: dedupe_by_id(usesDoctypeByFile[f.id] || [])
        };
    });

    // Build a nested directory tree keyed by path segment.
    const root = { name: '', dirs: {}, files: [] };
    fileEntries.forEach(function (entry) {
        const path = entry.node.file_path || entry.filename;
        const parts = path.split('/');
        const dirParts = parts.slice(0, -1);
        let cursor = root;
        dirParts.forEach(function (part) {
            if (!cursor.dirs[part]) cursor.dirs[part] = { name: part, dirs: {}, files: [] };
            cursor = cursor.dirs[part];
        });
        cursor.files.push(entry);
    });

    const other = Object.values(nodes)
        .filter(function (n) { return n.type !== 'File' && !definedBy[n.id]; })
        .sort(function (a, b) {
            return a.type === b.type ? a.name.localeCompare(b.name) : a.type.localeCompare(b.type);
        });

    return { root: root, other: other };
}

function dedupe_by_id(list) {
    const seen = new Set();
    return list.filter(function (n) {
        if (seen.has(n.id)) return false;
        seen.add(n.id);
        return true;
    });
}

function count_files_recursive(dirNode) {
    let count = dirNode.files.length;
    Object.values(dirNode.dirs).forEach(function (d) { count += count_files_recursive(d); });
    return count;
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

function render_folder_children(dirNode) {
    const parts = [];
    Object.values(dirNode.dirs)
        .sort(function (a, b) { return a.name.localeCompare(b.name); })
        .forEach(function (d) { parts.push(render_folder_row(d)); });
    dirNode.files
        .sort(function (a, b) { return a.filename.localeCompare(b.filename); })
        .forEach(function (f) { parts.push(render_file_row(f)); });
    return parts.join('');
}

function render_folder_row(dirNode) {
    const total = count_files_recursive(dirNode);
    const parts = ['<li class="kg-node kg-branch kg-folder" data-name="' + esc_attr(dirNode.name.toLowerCase()) + '">'];
    parts.push('<div class="kg-row kg-branch-row kg-folder-row">'
        + '<span class="kg-toggle">\u25B8</span>'
        + '<span class="kg-label kg-folder-label">' + frappe.utils.escape_html(dirNode.name) + '/</span>'
        + '<span class="kg-count">' + total + '</span>'
        + '</div>');
    parts.push('<ul class="kg-children">');
    parts.push(render_folder_children(dirNode));
    parts.push('</ul></li>');
    return parts.join('');
}

function render_file_row(f) {
    const parts = ['<li class="kg-node kg-branch kg-file" data-id="' + esc_attr(f.node.id) + '" data-name="' + esc_attr(f.filename.toLowerCase()) + '">'];
    parts.push('<div class="kg-row kg-branch-row kg-file-row">'
        + '<span class="kg-toggle">\u25B8</span>'
        + swatch(f.node.type)
        + '<span class="kg-label">' + frappe.utils.escape_html(f.filename) + '</span>'
        + '<span class="kg-count">' + f.members.length + '</span>'
        + '</div>');

    parts.push('<ul class="kg-children">');
    f.members.forEach(function (m) { parts.push(render_leaf_row(m)); });

    if (f.calls.length) {
        parts.push('<li class="kg-relation">Calls: ' + f.calls.map(function (n) { return frappe.utils.escape_html(n.name); }).join(', ') + '</li>');
    }
    if (f.usesDoctypes.length) {
        parts.push('<li class="kg-relation">Uses DocTypes: ' + f.usesDoctypes.map(function (n) { return frappe.utils.escape_html(n.name); }).join(', ') + '</li>');
    }
    parts.push('</ul></li>');
    return parts.join('');
}

function render_leaf_row(n) {
    return '<li class="kg-node kg-leaf" data-id="' + esc_attr(n.id) + '" data-name="' + esc_attr(n.name.toLowerCase()) + '">'
        + '<div class="kg-row kg-leaf-row">'
        + swatch(n.type)
        + '<span class="kg-label">' + frappe.utils.escape_html(n.name) + '</span>'
        + '<span class="kg-type-badge">' + frappe.utils.escape_html(n.type) + '</span>'
        + '</div></li>';
}

function render_other_group(nodes) {
    const parts = ['<li class="kg-node kg-branch kg-folder" data-name="other">'];
    parts.push('<div class="kg-row kg-branch-row kg-folder-row">'
        + '<span class="kg-toggle">\u25B8</span>'
        + '<span class="kg-label kg-folder-label">Other</span>'
        + '<span class="kg-count">' + nodes.length + '</span>'
        + '</div>');
    parts.push('<ul class="kg-children">');
    nodes.forEach(function (n) { parts.push(render_leaf_row(n)); });
    parts.push('</ul></li>');
    return parts.join('');
}

function swatch(type) {
    return '<span class="kg-swatch" style="background:' + (KG_TYPE_COLORS[type] || KG_DEFAULT_COLOR) + '"></span>';
}

function esc_attr(s) {
    return frappe.utils.escape_html(String(s));
}

// ---------------------------------------------------------------------------
// Interaction
// ---------------------------------------------------------------------------

function bind_tree_events(wrapper, graph) {
    const $root = $(wrapper.wrapper);

    $root.find('.kg-branch-row').on('click', function () {
        toggle_branch($(this).closest('.kg-branch'));
    });

    $root.find('.kg-leaf-row').on('click', function (e) {
        e.stopPropagation();
        const id = $(this).closest('.kg-leaf').data('id');
        const n = (graph.nodes || {})[id];
        if (!n) return;
        frappe.msgprint({
            title: n.name,
            message: '<b>Type:</b> ' + frappe.utils.escape_html(n.type) + '<br>'
                + '<b>File:</b> ' + frappe.utils.escape_html(n.file_path || '') + '<br>'
                + '<b>Lines:</b> ' + (n.line_start || '') + '\u2013' + (n.line_end || ''),
            indicator: 'blue'
        });
    });

    $root.find('.kg-tree-expand-all').on('click', function () {
        $root.find('.kg-branch').each(function () { set_branch_expanded($(this), true); });
    });

    $root.find('.kg-tree-collapse-all').on('click', function () {
        $root.find('.kg-branch').each(function () { set_branch_expanded($(this), false); });
    });

    $root.find('.kg-tree-search').on('input', function () {
        apply_search($root, $(this).val().toLowerCase().trim());
    });
}

function toggle_branch($li) {
    set_branch_expanded($li, !$li.hasClass('kg-expanded'));
}

function set_branch_expanded($li, expanded) {
    $li.toggleClass('kg-expanded', expanded);
    $li.children('.kg-children').toggle(expanded);
    $li.children('.kg-row').find('.kg-toggle').text(expanded ? '\u25BE' : '\u25B8');
}

// Recursively filters the tree by name. Returns true if this element (or any
// descendant) matches, so ancestors know whether to stay visible/expanded.
function apply_search($root, q) {
    if (!q) {
        $root.find('.kg-node').removeClass('kg-hidden kg-match');
        return;
    }
    $root.find('.kg-tree-root').children('li.kg-node').each(function () {
        filter_node($(this), q);
    });
}

function filter_node($li, q) {
    const ownName = $li.data('name') || '';
    const ownMatch = ownName.indexOf(q) !== -1;

    let childMatch = false;
    const $childList = $li.children('.kg-children');
    if ($childList.length) {
        $childList.children('li.kg-node').each(function () {
            if (filter_node($(this), q)) childMatch = true;
        });
    }

    const visible = ownMatch || childMatch;
    $li.toggleClass('kg-hidden', !visible).toggleClass('kg-match', ownMatch);
    if (visible && childMatch && $li.hasClass('kg-branch')) set_branch_expanded($li, true);
    return visible;
}

function inject_tree_styles() {
    if (document.getElementById('kg-tree-styles')) return;
    const css = '' +
        '.kg-tree-wrap{padding:8px 0;}' +
        '.kg-tree-toolbar{display:flex;gap:8px;align-items:center;margin-bottom:10px;}' +
        '.kg-tree-search{max-width:280px;}' +
        '.kg-tree-root{list-style:none;margin:0;padding:0;max-height:560px;overflow:auto;border:1px solid var(--border-color);border-radius:6px;}' +
        '.kg-tree-root .kg-children{list-style:none;margin:0;padding-left:22px;display:none;}' +
        '.kg-row{display:flex;align-items:center;gap:6px;padding:5px 8px;border-bottom:1px solid var(--border-color);font-size:12px;}' +
        '.kg-branch-row{cursor:pointer;}' +
        '.kg-branch-row:hover{background:var(--control-bg);}' +
        '.kg-folder-row .kg-folder-label{font-weight:600;}' +
        '.kg-file-row .kg-label{font-weight:500;}' +
        '.kg-leaf-row{cursor:pointer;}' +
        '.kg-leaf-row:hover{background:var(--control-bg);}' +
        '.kg-toggle{width:12px;display:inline-block;color:var(--text-muted);font-size:10px;}' +
        '.kg-swatch{width:8px;height:8px;border-radius:50%;flex:none;}' +
        '.kg-label{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}' +
        '.kg-count{color:var(--text-muted);font-size:11px;}' +
        '.kg-type-badge{color:var(--text-muted);font-size:10px;border:1px solid var(--border-color);border-radius:3px;padding:0 4px;}' +
        '.kg-relation{padding:4px 8px 4px 20px;font-size:11px;color:var(--text-muted);border-bottom:1px solid var(--border-color);}' +
        '.kg-hidden{display:none !important;}' +
        '.kg-match > .kg-row{background:var(--yellow-50,#fefce8);}';
    const style = document.createElement('style');
    style.id = 'kg-tree-styles';
    style.textContent = css;
    document.head.appendChild(style);
}

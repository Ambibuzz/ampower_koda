frappe.realtime.on('hd_ticket_intake_done', function(data) {
    if (data.kind === 'success') {
        var message = frappe.utils.escape_html(data.message || '');
        if (data.redirect_url) {
            message += '<br><a href="' + data.redirect_url + '">' +
                __('Open Agent Request') + ' \u2192</a>';
        }
        frappe.show_alert({ message: message, indicator: 'green' }, 10);
    } else {
        frappe.show_alert({ message: frappe.utils.escape_html(data.message || ''), indicator: 'red' }, 10);
    }
});

// Copyright (c) 2026, Ambibuzz Technologies LLP and contributors
// Agent Settings — client script

var PROVIDER_MODELS = {
    'OpenAI': [
        { value: 'gpt-4o-mini', label: 'GPT-4o Mini — fast, cost-effective' },
        { value: 'gpt-5-mini', label: 'GPT-5 Mini — next-gen, efficient' },
        { value: 'gpt-5.1-codex-mini', label: 'GPT-5.1 Codex Mini — compact coding' },
        { value: 'gpt-5-codex', label: 'GPT-5 Codex — coding model' },
        { value: 'gpt-5.1-codex', label: 'GPT-5.1 Codex — coding model' },
        { value: 'gpt-5.2-codex', label: 'GPT-5.2 Codex — latest coding model' }
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

frappe.ui.form.on('Agent Settings', {
    refresh: function(frm) {
        set_model_options_for_provider(frm);
    },

    default_ai_provider: function(frm) {
        var provider = frm.doc.default_ai_provider || 'OpenAI';
        set_model_options_for_provider(frm);
        frm.set_value('default_ai_model', DEFAULT_MODELS[provider] || DEFAULT_MODELS['OpenAI']);
    },

    onload: function(frm) {
        set_model_options_for_provider(frm);
    }
});

function set_model_options_for_provider(frm) {
    var provider = frm.doc.default_ai_provider || 'OpenAI';
    var entries = PROVIDER_MODELS[provider] || PROVIDER_MODELS['OpenAI'];
    var model_ids = entries.map(function(m) { return m.value; });
    var options_str = model_ids.join('\n');

    frm.set_df_property('default_ai_model', 'options', options_str);
    frm.refresh_field('default_ai_model');

    var current = frm.doc.default_ai_model || '';
    if (model_ids.indexOf(current) === -1) {
        frm.set_value('default_ai_model', model_ids[0]);
    }
}

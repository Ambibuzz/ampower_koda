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
    ],
    'OpenRouter': [
        { value: 'deepseek/deepseek-v4-flash-0731', label: 'DeepSeek V4 Flash — fast, cheap' },
        { value: 'deepseek/deepseek-chat', label: 'DeepSeek Chat — general' },
        { value: 'anthropic/claude-sonnet-4', label: 'Claude Sonnet 4 — via OpenRouter' },
        { value: 'openai/gpt-4o-mini', label: 'GPT-4o Mini — via OpenRouter' },
        { value: 'qwen/qwen-2.5-coder-32b-instruct', label: 'Qwen 2.5 Coder 32B — coding' }
    ]
};

var DEFAULT_MODELS = {
    'OpenAI': 'gpt-4o-mini',
    'Gemini': 'gemini-2.0-flash',
    'Claude': 'claude-sonnet-4-20250514',
    'OpenRouter': 'deepseek/deepseek-v4-flash-0731'
};

frappe.ui.form.on('Agent Settings', {
    refresh: function(frm) {
        set_model_options_for_provider(frm, false);
    },

    default_ai_provider: function(frm) {
        // The one place a reset is correct: the user changed provider, so a
        // model belonging to the old one is genuinely no longer valid.
        var provider = frm.doc.default_ai_provider || 'OpenAI';
        set_model_options_for_provider(frm, true);
        frm.set_value('default_ai_model', DEFAULT_MODELS[provider] || DEFAULT_MODELS['OpenAI']);
    },

    onload: function(frm) {
        set_model_options_for_provider(frm, false);
    }
});


function set_model_options_for_provider(frm, reset) {
    var provider = frm.doc.default_ai_provider || 'OpenAI';
    var entries = PROVIDER_MODELS[provider] || PROVIDER_MODELS['OpenAI'];
    var model_ids = entries.map(function(m) { return m.value; });
    var current = frm.doc.default_ai_model || '';

    // Keep an unlisted saved model selectable, so the Select can render it
    // instead of falling back to showing the first option as if it were set.
    if (!reset && current && model_ids.indexOf(current) === -1) {
        model_ids = model_ids.concat([current]);
    }

    frm.set_df_property('default_ai_model', 'options', model_ids.join('\n'));
    frm.refresh_field('default_ai_model');

    if (reset && model_ids.indexOf(current) === -1) {
        frm.set_value('default_ai_model', DEFAULT_MODELS[provider] || model_ids[0]);
    }
}

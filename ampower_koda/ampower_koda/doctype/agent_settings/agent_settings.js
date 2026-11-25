// Copyright (c) 2026, Ambibuzz Technologies LLP and contributors
// Agent Settings — client script

var PROVIDER_MODELS = {
    'OpenAI': [
        { value: 'gpt-6-luna', label: 'GPT-6 Luna — fast, cost-effective' },
        { value: 'gpt-6.1-terra', label: 'GPT-6.1 Terra — efficient' },
        { value: 'gpt-6.2-sol', label: 'GPT-6.2 Sol — everyday coding' },
        { value: 'gpt-6.2-astra', label: 'GPT-6.2 Astra — most capable' }
    ],
    'Gemini': [
        { value: 'gemini-3.8-flash', label: 'Gemini 3.8 Flash — most capable Flash' },
        { value: 'gemini-3.6-flash', label: 'Gemini 3.6 Flash — balanced' },
        { value: 'gemini-3.5-flash-lite', label: 'Gemini 3.5 Flash-Lite — fast, light' }
    ],
    'Claude': [
        { value: 'claude-opus-5-5', label: 'Claude Opus 5.5 — most capable' },
        { value: 'claude-sonnet-5-5', label: 'Claude Sonnet 5.5 — balanced' },
        { value: 'claude-haiku-5-5', label: 'Claude Haiku 5.5 — fast, light' }
    ],
    'OpenRouter': [
        { value: 'deepseek/deepseek-v4-flash-0731', label: 'DeepSeek V4 Flash — fast, cheap' },
        { value: 'deepseek/deepseek-v4.1-flash', label: 'DeepSeek V4.1 Flash — newer Flash' },
        { value: 'anthropic/claude-sonnet-5.5', label: 'Claude Sonnet 5.5 — via OpenRouter' },
        { value: 'openai/gpt-6-luna', label: 'GPT-6 Luna — via OpenRouter' }
    ]
};

var DEFAULT_MODELS = {
    'OpenAI': 'gpt-6-luna',
    'Gemini': 'gemini-3.8-flash',
    'Claude': 'claude-sonnet-5-5',
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
    with_provider_models(provider, function(entries) {
        // The provider may have changed while the list was loading.
        if ((frm.doc.default_ai_provider || 'OpenAI') === provider) apply_model_options(frm, provider, entries, reset);
    });
}

// OpenAI's and OpenRouter's models are listed live (newest first, cached server-side), so a
// new model is offered without a code change; the built-in list is the fallback.
var LIVE_MODEL_PROVIDERS = ['OpenAI', 'OpenRouter'];
// The picker shows this many: the newest while the field is empty, else the newest that match.
var MODEL_PICKER_ITEMS = 5;

function with_provider_models(provider, done) {
    var fallback = PROVIDER_MODELS[provider] || PROVIDER_MODELS['OpenAI'];
    if (LIVE_MODEL_PROVIDERS.indexOf(provider) === -1) return done(fallback);
    window.koda_live_models = window.koda_live_models || {};
    if (window.koda_live_models[provider]) return done(window.koda_live_models[provider]);
    frappe.call({
        method: 'ampower_koda.agent.api.get_provider_models',
        args: { provider: provider },
        callback: function(r) {
            var ids = (r.message || {}).models || [];
            if (!ids.length) return done(fallback);
            window.koda_live_models[provider] = ids.map(function(id) { return { value: id, label: id }; });
            done(window.koda_live_models[provider]);
        },
        error: function() { done(fallback); }
    });
}

function apply_model_options(frm, provider, entries, reset) {
    var model_ids = entries.map(function(m) { return m.value; });
    var current = frm.doc.default_ai_model || '';

    // Keep an unlisted saved model selectable, so the Select can render it
    // instead of falling back to showing the first option as if it were set.
    if (!reset && current && model_ids.indexOf(current) === -1) {
        model_ids = model_ids.concat([current]);
    }

    frm.set_df_property('default_ai_model', 'options', model_ids.join('\n'));
    // Autocomplete clears any value not in its list on blur; the list is only a
    // suggestion, so a pasted provider model id must survive.
    frm.set_df_property('default_ai_model', 'ignore_validation', 1);
    frm.refresh_field('default_ai_model');

    // The Autocomplete reads its options and item limit only when it is built, so a
    // later list goes to the control itself.
    var model_field = frm.fields_dict.default_ai_model;
    if (model_field && model_field.set_data) {
        model_field.set_data(model_ids);
    }
    if (model_field && model_field.awesomplete) {
        model_field.awesomplete.maxItems = MODEL_PICKER_ITEMS;
    }

    if (reset && model_ids.indexOf(current) === -1) {
        frm.set_value('default_ai_model', DEFAULT_MODELS[provider] || model_ids[0]);
    }
}

// Route: koda-replay/<Agent Request>/<recording id>. Without a recording id the newest one is shown.

const REPLAY_METHOD = 'ampower_koda.agent.replay.';

frappe.pages['koda-replay'].on_page_load = function (wrapper) {
    const page = frappe.ui.make_app_page({
        parent: wrapper,
        title: __('Koda Replay'),
        single_column: true,
    });
    const state = {
        page: page,
        request_name: null,
        recording_id: null,
        player: null,
        // Bumped on every load; a slower earlier load must not mount over a newer one.
        generation: 0,
        recording_field: null,
        $hint: null,
        $player: null,
    };

    state.recording_field = page.add_field({
        fieldname: 'recording',
        label: __('Recording'),
        fieldtype: 'Select',
        change: function () {
            const chosen = state.recording_field.get_value();
            if (chosen && chosen !== state.recording_id) {
                frappe.set_route('koda-replay', state.request_name, chosen);
            }
        },
    });
    // The page head gives fields a narrow column; a recording's label needs its date, time and route.
    state.recording_field.$wrapper.css('min-width', 'min(560px, 100%)');
    page.set_secondary_action(__('Back to request'), function () {
        if (state.request_name) {
            frappe.set_route('Form', 'Agent Request', state.request_name);
        } else {
            frappe.set_route('List', 'Agent Request');
        }
    });

    state.$hint = $('<p class="text-muted small"></p>').appendTo(page.main);
    state.$player = $('<div class="koda-replay-player"></div>').appendTo(page.main);
    wrapper.koda_replay = state;
};

// Frappe keeps the page between visits: a later visit, maybe for another recording, only shows it again.
frappe.pages['koda-replay'].on_page_show = function (wrapper) {
    const state = wrapper.koda_replay;
    const route = frappe.get_route();
    const request_name = route[1] || '';
    const recording_id = route[2] || '';
    // Without a recording id the list is fetched again and the newest shown: the agent may have recorded more
    // checks since the last visit.
    const same_recording = Boolean(recording_id) && recording_id === state.recording_id;
    if (state.player && request_name === state.request_name && same_recording) {
        return;
    }
    show_replay(state, request_name, recording_id);
};

async function show_replay(state, request_name, recording_id) {
    const generation = ++state.generation;
    remove_player(state);
    // The previous request's recordings and title must not show while this one loads, or if it has none.
    state.recording_id = null;
    state.recording_field.$wrapper.hide();
    state.page.set_title(__('Koda Replay'));
    state.$hint.text('');
    state.request_name = request_name;
    if (!request_name) {
        state.$hint.text(__('Open a replay from an Agent Request: its browser checks are listed there.'));
        return;
    }

    let recordings;
    try {
        recordings = (await frappe.call({
            method: REPLAY_METHOD + 'list_recordings',
            args: { request_name: request_name },
        })).message || [];
    } catch {
        // frappe.call has already shown the server's error.
        return;
    }
    if (generation !== state.generation) return;
    if (!recordings.length) {
        state.$hint.text(__('{0} has no recorded browser checks yet.', [request_name]));
        return;
    }

    const linked = recordings.find((recording) => recording.id === recording_id);
    const chosen = linked || recordings[0];
    state.recording_id = chosen.id;
    state.recording_field.df.options = recordings.map(function (recording) {
        return { value: recording.id, label: recording_label(recording) };
    });
    state.recording_field.refresh();
    state.recording_field.set_value(chosen.id);
    state.recording_field.$wrapper.show();
    state.page.set_title(__('Replay: {0}', [chosen.route || request_name]));
    const usage = __('Play the check, drag the timeline, or click a step to see the page right after it.');
    // Old recordings are deleted (recording/cleanup.py): a Replay button in an old log can outlive its recording.
    const missing = recording_id && !linked
        ? __('Recording {0} no longer exists; this is the newest.', [recording_id]) + ' '
        : '';
    state.$hint.text(missing + usage);

    await frappe.require(['koda_replay.bundle.js', 'koda_replay.bundle.css']);
    if (generation !== state.generation) return;
    // Each load mounts into its own element: a slower earlier load that finishes late destroys only its own
    // player.
    const container = $('<div></div>').appendTo(state.$player).get(0);
    let player;
    try {
        player = await ampower_koda.replay.mount(container, {
            recordingUrl: method_url('recording_file', { request_name: request_name, recording: chosen.id }),
            replayerUrl: method_url('replayer_script', {}),
        });
    } catch (error) {
        // The player may have built part of itself before failing.
        container.remove();
        if (generation === state.generation) {
            state.$hint.text(__('The recording could not be loaded: {0}', [String(error.message || error)]));
        }
        return;
    }
    if (generation !== state.generation) {
        player.destroy();
        container.remove();
        return;
    }
    state.player = player;
}

function method_url(method, args) {
    return '/api/method/' + REPLAY_METHOD + method + '?' + new URLSearchParams(args).toString();
}

function recording_label(recording) {
    // With the time, so two checks on the same day can be told apart.
    const when = recording.started
        ? frappe.datetime.str_to_user(moment.unix(recording.started).format('YYYY-MM-DD HH:mm:ss'))
        : recording.id;
    const failed = recording.failed_steps ? __(', {0} failed', [recording.failed_steps]) : '';
    // A page that kept changing hit the event limit: the replay stops before the check did.
    const truncated = recording.truncated ? __(', cut short') : '';
    // Select writes option labels as HTML.
    const route = frappe.utils.escape_html(recording.route || '');
    const steps = recording.steps === 1 ? __('1 step') : __('{0} steps', [recording.steps]);
    return `${when} · ${route} · ${steps}${failed}${truncated}`;
}

function remove_player(state) {
    if (state.player) {
        state.player.destroy();
        state.player = null;
    }
    state.$player.empty();
}

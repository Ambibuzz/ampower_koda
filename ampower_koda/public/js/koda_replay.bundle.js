// The Koda Replay page loads this bundle on demand: frappe.require("koda_replay.bundle.js").

import { loadRecording, loadReplayer } from './koda_replay/load.js';
import { Player } from './koda_replay/player.js';

frappe.provide('ampower_koda.replay');

// Plays one recording inside `root`. source.recordingUrl is the recording, source.replayerUrl rrweb's player.
ampower_koda.replay.mount = async function (root, source) {
    const [record] = await Promise.all([loadRecording(source.recordingUrl), loadReplayer(source.replayerUrl)]);
    return new Player(root, record);
};

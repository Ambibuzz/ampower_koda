// Fetches a recording (gzipped JSON, see recording/bundle.py) and rrweb's player.

export async function loadRecording(url) {
    const response = await fetch(url, { credentials: 'same-origin' });
    if (!response.ok) {
        throw new Error(`${url} answered ${response.status}`);
    }
    const stream = response.body.pipeThrough(new DecompressionStream('gzip'));
    const record = JSON.parse(await new Response(stream).text());
    // The recorder did not start in the page: there is nothing the player could show.
    if (!record.events || !record.events.length) {
        throw new Error(__('the recording is empty'));
    }
    return record;
}

// rrweb's player is served by the server from node_modules (replay.py replayer_script) instead of being bundled,
// so `bench build` does not need node_modules. It sets window.rrwebReplay; loaded once per page.
let replayerLoading = null;

export function loadReplayer(url) {
    if (window.rrwebReplay) {
        return Promise.resolve(window.rrwebReplay);
    }
    if (!replayerLoading) {
        replayerLoading = new Promise((resolve, reject) => {
            // The UMD build registers with an AMD loader when the page has one, instead of setting the global.
            const amdDefine = window.define;
            window.define = undefined;
            const script = document.createElement('script');
            script.src = url;
            const fail = (message) => {
                replayerLoading = null;
                script.remove();
                reject(new Error(message));
            };
            script.onload = () => {
                window.define = amdDefine;
                if (window.rrwebReplay) {
                    resolve(window.rrwebReplay);
                } else {
                    fail(__('the replay library loaded but did not set window.rrwebReplay'));
                }
            };
            script.onerror = () => {
                window.define = amdDefine;
                fail(__(
                    'the replay library @rrweb/replay is not installed; run `bench setup requirements --node`'
                ));
            };
            document.head.appendChild(script);
        });
    }
    return replayerLoading;
}

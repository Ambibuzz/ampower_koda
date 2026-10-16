// Runs inside check_page's headless browser, injected by recorder.py right after the rrweb recorder library.
// Records the page with rrweb; recorder.py collects the events after the load and after every step.
(() => {
  // Playwright injects this into every frame, including the blank helper frame rrweb adds: recording there would
  // add another frame, and so on until the stack overflows. rrweb records same-origin frames from the main page.
  const alreadyRunning = Boolean(window.__kodaRec);
  const rrwebMissing = !window.rrwebRecord;
  const insideAFrame = window.top !== window;
  const blankHelperPage = location.protocol === 'about:';
  if (alreadyRunning || rrwebMissing || insideAFrame || blankHelperPage) {
    return;
  }

  const rrwebRecord = window.rrwebRecord.record;
  // Desk's icon sprite is about 2.5 MB of static SVG on every page; it is left out of the recording.
  const ICON_SPRITE_ID = 'all-symbols';

  // Same limit as recorder.py's MAX_EVENTS. A page that never stops changing stops being recorded here,
  // so it cannot fill the browser's memory between two drains; the start, with the first snapshot, is kept.
  const MAX_EVENTS = 20000;

  const events = [];
  let recordedEvents = 0;
  let truncated = false;

  rrwebRecord({
    emit(event) {
      if (recordedEvents >= MAX_EVENTS) {
        truncated = true;
        return;
      }
      recordedEvents += 1;
      events.push(event);
    },
    // Field values can hold customer or employee data; rrweb masks only password fields by default.
    maskAllInputs: true,
    sampling: {mousemove: false, scroll: 150, media: 800, input: 'last'},
    slimDOMOptions: 'all',
    inlineStylesheet: true,
    recordCanvas: false,
    collectFonts: false,
    blockSelector: '#' + ICON_SPRITE_ID,
  });

  window.__kodaRec = {
    stamp(name, payload) {
      try {
        rrwebRecord.addCustomEvent(name, payload);
      } catch (error) {
        // Before rrweb's first snapshot there is nothing to stamp into; recorder.py still lists the step.
      }
    },

    drain() {
      return {events: events.splice(0), truncated};
    },
  };
})();
//# sourceURL=__koda_capture.js

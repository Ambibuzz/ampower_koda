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

  const events = [];

  rrwebRecord({
    emit(event) {
      events.push(event);
    },
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
      return {events: events.splice(0)};
    },
  };
})();
//# sourceURL=__koda_capture.js

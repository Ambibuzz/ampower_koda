// Plays a recorded browser check: the replayed page, its timeline and the steps the agent ran.

import { create } from './dom.js';

const CUSTOM_EVENT = 5;
// Positions on the timeline slider, whatever the recording's length.
const SCRUBBER_POSITIONS = 1000;

function stepLabel(step) {
    if (!step) {
        return '';
    }
    if (step.label) {
        return step.label;
    }
    for (const kind of ['click', 'fill', 'select', 'press']) {
        if (kind in step) {
            const value = 'value' in step ? ` = "${step.value}"` : '';
            return `${kind} ${step[kind]}${value}`;
        }
    }
    if (step.viewport) {
        return `resize to ${step.viewport.join('×')}`;
    }
    if (step.wait_ms) {
        return `wait ${step.wait_ms}ms`;
    }
    return JSON.stringify(step);
}

export class Player {
    constructor(root, record) {
        this.root = root;
        this.playing = false;
        this.scale = 1;
        this.events = record.events;
        this.startTime = this.events[0].timestamp;
        this.steps = this.readSteps();
        this.build();
    }

    readSteps() {
        const custom = (tag) => this.events.filter(
            (event) => event.type === CUSTOM_EVENT && event.data.tag === tag
        );
        const ends = custom('koda-step-end');
        return custom('koda-step').map((event) => {
            const step = { t: event.timestamp - this.startTime, ...event.data.payload };
            const end = ends.find((endEvent) => endEvent.data.payload.index === step.index);
            step.end = end ? end.timestamp - this.startTime : step.t;
            step.outcome = end ? end.data.payload.outcome : '';
            return step;
        });
    }

    build() {
        this.root.classList.add('kp');
        this.stage = create('div', { class: 'kp-stage' });
        this.stageWrap = create('div', { class: 'kp-stage-wrap' }, this.stage);

        this.scrubber = create('input', {
            type: 'range',
            min: 0,
            max: SCRUBBER_POSITIONS,
            value: 0,
            class: 'kp-scrub',
            'aria-label': __('Timeline'),
        });
        this.ticks = create('div', { class: 'kp-ticks' });
        this.playButton = create('button', { class: 'kp-btn', onclick: () => this.togglePlay() }, __('Play'));
        this.clock = create('span', { class: 'kp-clock' }, '0.0s');
        this.stepList = create('ol', { class: 'kp-steps' });

        this.root.append(
            create(
                'div',
                { class: 'kp-main' },
                this.stageWrap,
                create(
                    'div',
                    { class: 'kp-bar' },
                    this.playButton,
                    create('div', { class: 'kp-track' }, this.ticks, this.scrubber),
                    this.clock
                )
            ),
            create('aside', { class: 'kp-side' }, create('h3', {}, __('What the agent did')), this.stepList)
        );

        // Loaded by mount before the player is made (load.js loadReplayer).
        this.replayer = new window.rrwebReplay.Replayer(this.events, {
            root: this.stage,
            skipInactive: false,
            showWarning: false,
            mouseTail: false,
            triggerFocus: false,
            UNSAFE_replayCanvas: false,
            speed: 1,
        });
        // At least 1ms: positions on the timeline divide by it, and a recording can start and end in the same ms.
        this.total = Math.max(1, this.replayer.getMetaData().totalTime);
        // A step that resizes the window resizes the replayed page too.
        this.replayer.on('resize', () => this.fit());
        this.replayer.on('finish', () => this.showPaused());

        this.buildStepList();
        this.listen();
        this.seek(this.steps.length ? this.steps[0].t : 0);
    }

    buildStepList() {
        for (const step of this.steps) {
            const failed = String(step.outcome).startsWith('FAILED');
            step.item = create(
                'li',
                { class: 'kp-step' + (failed ? ' kp-failed' : ''), onclick: () => this.seek(step.end) },
                create('span', { class: 'kp-step-label' }, stepLabel(step.action))
            );
            this.stepList.append(step.item);
            this.ticks.append(create('button', {
                class: 'kp-tick',
                title: stepLabel(step.action),
                style: `left:${(step.t / this.total) * 100}%`,
                onclick: () => this.seek(step.end),
            }));
        }
    }

    listen() {
        this.scrubber.addEventListener('input', () => {
            this.seek((this.scrubber.value / SCRUBBER_POSITIONS) * this.total);
        });
        // Space plays and pauses only while the pointer is over the player, so the page around it keeps it
        // for scrolling.
        this.onKey = (event) => {
            const typing = event.target.closest && event.target.closest('input, textarea');
            if (typing || !this.root.matches(':hover') || event.key !== ' ') {
                return;
            }
            event.preventDefault();
            this.togglePlay();
        };
        window.addEventListener('keydown', this.onKey);
        this.resizer = new ResizeObserver(() => this.fit());
        this.resizer.observe(this.stageWrap);
    }

    fit() {
        const wrapper = this.stage.querySelector('.replayer-wrapper');
        const frame = this.replayer.iframe;
        if (!wrapper || !frame) {
            return;
        }
        const width = frame.offsetWidth || Number(frame.width);
        const height = frame.offsetHeight || Number(frame.height);
        // A hidden player measures 0 wide; keep the last scale instead of dividing by it later.
        this.scale = Math.min(1, this.stageWrap.clientWidth / width) || this.scale;
        wrapper.style.transform = `scale(${this.scale})`;
        wrapper.style.transformOrigin = 'top left';
        this.stage.style.height = Math.round(height * this.scale) + 'px';
    }

    now() {
        return Math.max(0, Math.min(this.total, this.replayer.getCurrentTime()));
    }

    seek(time) {
        this.replayer.pause(Math.max(0, Math.min(this.total, time)));
        this.showPaused();
        this.fit();
    }

    showPaused() {
        this.playing = false;
        this.playButton.textContent = __('Play');
        clearInterval(this.timer);
        this.sync();
    }

    togglePlay() {
        if (this.playing) {
            this.replayer.pause();
            this.showPaused();
            return;
        }
        // At the end, Play starts over instead of finishing at once.
        this.replayer.play(this.now() >= this.total ? 0 : this.now());
        this.playing = true;
        this.playButton.textContent = __('Pause');
        this.timer = setInterval(() => this.sync(), 100);
    }

    currentStep(time) {
        let current = null;
        for (const step of this.steps) {
            if (step.t <= time) {
                current = step;
            }
        }
        return current;
    }

    sync() {
        const time = this.now();
        this.scrubber.value = Math.round((time / this.total) * SCRUBBER_POSITIONS);
        this.clock.textContent = (time / 1000).toFixed(1) + 's';
        const current = this.currentStep(time);
        for (const step of this.steps) {
            step.item.classList.toggle('kp-current', step === current);
        }
    }

    destroy() {
        clearInterval(this.timer);
        this.replayer.destroy();
        this.resizer.disconnect();
        window.removeEventListener('keydown', this.onKey);
        this.root.replaceChildren();
        this.root.classList.remove('kp');
    }
}

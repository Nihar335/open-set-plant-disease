// Browser voice module: WAV encoding, resampling, spoken-answer matching and the report player.
// Run: node --test stage10/app/tests_js/voice_module.test.js
import assert from "node:assert/strict";
import { test } from "node:test";

import { createPlayer, downsample, encodeWav, matchOption } from "../static/js/voice.js";

test("downsample 48 kHz -> 16 kHz keeps duration and level", () => {
  const x = new Float32Array(48000).fill(0.5);
  const y = downsample(x, 48000);
  assert.equal(y.length, 16000);
  assert.ok(Math.abs(y[100] - 0.5) < 1e-6);
  assert.equal(downsample(new Float32Array([0.1, 0.2]), 16000).length, 2);
});

test("WAV encoder writes a 16 kHz mono 16-bit PCM file", () => {
  const buf = encodeWav(new Float32Array([0, 1, -1, 2]));
  const v = new DataView(buf);
  const s = (o, n) => String.fromCharCode(...new Uint8Array(buf, o, n));
  assert.equal(s(0, 4), "RIFF"); assert.equal(s(8, 4), "WAVE"); assert.equal(s(36, 4), "data");
  assert.equal(v.getUint16(20, true), 1); assert.equal(v.getUint16(22, true), 1);
  assert.equal(v.getUint32(24, true), 16000); assert.equal(v.getUint16(34, true), 16);
  assert.equal(v.getUint32(40, true), 8);
  assert.equal(v.getInt16(46, true), 32767); assert.equal(v.getInt16(48, true), -32768); assert.equal(v.getInt16(50, true), 32767);
});

const YN = [{ option_id: "yes", label: "Yes" }, { option_id: "no", label: "No" }, { option_id: "not_sure", label: "Not sure / I don't know" }];
const AGE = [{ option_id: "older_lower", label: "Older / lower leaves" }, { option_id: "young_upper", label: "Young / upper leaves" },
  { option_id: "not_sure", label: "Not sure / I don't know" }];

test("spoken answers map to one option or none (never a guess)", () => {
  assert.equal(matchOption("yes", YN).option_id, "yes");
  assert.equal(matchOption("yeah I think so", YN).option_id, "yes");
  assert.equal(matchOption("no", YN).option_id, "no");
  assert.equal(matchOption("I don't know", YN).option_id, "not_sure");
  assert.equal(matchOption("yes and no", YN), null);                       // ambiguous -> user must tap
  assert.equal(matchOption("purple", YN), null);
  assert.equal(matchOption("", YN), null);
  assert.equal(matchOption("on the older lower leaves", AGE).option_id, "older_lower");
  assert.equal(matchOption("leaves", AGE), null);
});

function fakeAudio() {
  const made = [];
  globalThis.URL.createObjectURL = () => "blob:x";
  globalThis.URL.revokeObjectURL = () => {};
  globalThis.Audio = class {
    constructor(src) { this.src = src; this.paused = true; made.push(this); }
    play() { this.paused = false; return Promise.resolve(); }
    pause() { this.paused = true; }
    end() { this.onended && this.onended(); }
  };
  return made;
}

test("player: play, pause, resume, repeat, next, speed, stop, error", async () => {
  const made = fakeAudio();
  const asked = [];
  let fail = false;
  const p = createPlayer(async (seg, speed) => { asked.push([seg.text, speed]); if (fail) throw new Error("x"); return {}; });
  const tick = () => new Promise((r) => setTimeout(r, 0));
  p.load([{ text: "a" }, { text: "b" }, { text: "c" }]);
  assert.equal(p.state.state, "stopped");
  p.play(); await tick(); await tick();
  assert.equal(p.state.state, "playing"); assert.equal(p.state.index, 0);
  p.pause(); assert.equal(p.state.state, "paused"); assert.equal(made.at(-1).paused, true);
  p.play(); assert.equal(p.state.state, "playing");
  made.at(-1).end(); await tick(); await tick();                         // segment finished -> next one
  assert.equal(p.state.index, 1);
  p.repeat(); await tick(); await tick(); assert.equal(p.state.index, 1);
  p.setSpeed(0.8); p.next(); await tick(); await tick();
  assert.deepEqual(asked.at(-1), ["c", 0.8]);
  made.at(-1).end(); await tick();
  assert.equal(p.state.state, "stopped");                                 // end of report
  p.play(); await tick(); await tick();
  p.stop(); assert.equal(p.state.state, "stopped"); assert.equal(p.state.index, 0);
  fail = true; p.play(); await tick(); await tick();
  assert.equal(p.state.state, "error");
});

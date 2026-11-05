// Farmer-guided marking geometry and the voice-interaction state machine. Run: node --test stage10/app/tests_js/marking_voice.test.js
import assert from "node:assert/strict";
import { test } from "node:test";

import * as M from "../static/js/marking.js";
import { voiceStep } from "../static/js/media.js";

const close = (a, b, e = 1e-3) => assert.ok(Math.abs(a - b) < e, `${a} != ${b}`);

test("marking: letterboxed content box and pointer -> normalised coordinates", () => {
  assert.deepEqual(M.contentBox(800, 300, 400, 300), { x: 200, y: 0, w: 400, h: 300 });     // pillarbox
  assert.deepEqual(M.contentBox(400, 600, 400, 300), { x: 0, y: 150, w: 400, h: 300 });     // letterbox
  assert.equal(M.contentBox(0, 10, 4, 3), null);
  const rect = { left: 10, top: 20, width: 800, height: 300 };
  assert.deepEqual(M.toNormalized(10 + 200 + 400, 20 + 150, rect, 400, 300), { x: 1, y: 0.5 });
  assert.equal(M.toNormalized(10 + 100, 20 + 150, rect, 400, 300), null);                    // in the black bar
});

test("marking: same point at any display size gives the same normalised mark (resize)", () => {
  for (const [w, h] of [[640, 480], [320, 240], [1280, 960], [375, 281.25]]) {
    const p = M.toNormalized(w * 0.25, h * 0.75, { left: 0, top: 0, width: w, height: h }, 4000, 3000);
    close(p.x, 0.25); close(p.y, 0.75);
  }
});

test("marking: rotated photo (portrait after EXIF) maps correctly", () => {
  // a 4000x3000 sensor image with EXIF orientation 6 is displayed 3000x4000; the browser reports the displayed size
  const p = M.toNormalized(150, 100, { left: 0, top: 0, width: 300, height: 400 }, 3000, 4000);
  close(p.x, 0.5); close(p.y, 0.25);
  const px = M.toPixels({ shape: "circle", x: p.x, y: p.y, r: 0.1 }, 3000, 4000);            // displayed (rotated) pixels
  close(px.cx, 1500); close(px.cy, 1000); close(px.r, 300);
  // the same view in a wider, letterboxed element gives the same normalised point
  const q = M.toNormalized(400 + 150, 100, { left: 0, top: 0, width: 1100, height: 400 }, 3000, 4000);
  close(q.x, 0.5); close(q.y, 0.25);
});

test("marking: outside points are rejected", () => {
  assert.equal(M.toNormalized(150, 600, { left: 0, top: 0, width: 300, height: 400 }, 3000, 4000), null);
  const p = M.toNormalized(150, 300, { left: 0, top: 0, width: 300, height: 400 }, 3000, 4000);
  close(p.x, 0.5); close(p.y, 0.75);
});

test("marking: tap, drag in any direction, grid, pixels", () => {
  const tap = M.markFromGesture({ x: 0.5, y: 0.5 }, { x: 0.505, y: 0.5 });
  assert.equal(tap.shape, "circle");
  const box = M.markFromGesture({ x: 0.6, y: 0.7 }, { x: 0.2, y: 0.1 });
  assert.deepEqual([box.x0, box.y0, box.x1, box.y1], [0.2, 0.1, 0.6, 0.7]);
  assert.deepEqual(M.gridMark(4), { shape: "box", x0: 0.3333, y0: 0.3333, x1: 0.6667, y1: 0.6667, via: "grid" });
  assert.equal(M.gridMark(9), null);
  const px = M.toPixels(box, 1000, 500);
  close(px.x, 200); close(px.y, 50); close(px.width, 400); close(px.height, 300);
  close(M.toPixels(tap, 1000, 500).r, 0.06 * 500);
});

test("marking: add up to the limit, undo and remove", () => {
  let ms = [];
  for (let i = 0; i < 12; i++) ms = M.addMark(ms, M.gridMark(i % 9));
  assert.equal(ms.length, M.MAX_MARKS);
  assert.equal(M.undoMark(ms).length, M.MAX_MARKS - 1);
  const r = M.removeMark(ms, 0);
  assert.equal(r.length, M.MAX_MARKS - 1);
  assert.deepEqual(r[0], ms[1]);
  assert.deepEqual(M.addMark([], null), []);
});

// ---------------------------------------------------------------------------------------------- voice
const on = { available: true, consentAccepted: true };
const run = (events, ctx = on) => events.reduce((s, e) => voiceStep(s, e, ctx), null);

test("voice: unavailable engine never leaves 'unavailable'", () => {
  for (const e of ["press", "granted", "confirm", "consent_accept"])
    assert.equal(run([{ type: e }], { available: false }).name, "unavailable");
});

test("voice: consent is asked before the first use; declining falls back to typing", () => {
  const ctx = { available: true, consentAccepted: false };
  assert.equal(run([{ type: "press" }], ctx).name, "consent");
  const d = run([{ type: "press" }, { type: "consent_decline" }], ctx);
  assert.equal(d.name, "idle"); assert.equal(d.message, "speech.type_instead");
  assert.equal(run([{ type: "press" }, { type: "consent_accept" }], ctx).name, "requesting");
});

test("voice: microphone denial or no device -> text fallback", () => {
  assert.equal(run([{ type: "press" }, { type: "denied" }]).message, "speech.denied");
  assert.equal(run([{ type: "press" }, { type: "no_device" }]).message, "speech.no_device");
});

test("voice: a transcript needs explicit confirmation; correction and repeat work", () => {
  const heard = [{ type: "press" }, { type: "granted" }, { type: "stop" }, { type: "result", result: { status: "ok", text: "tamatar" } }];
  const s = run(heard);
  assert.equal(s.name, "confirm"); assert.equal(s.transcript, "tamatar");
  assert.notEqual(s.name, "confirmed");
  const ok = voiceStep(s, { type: "confirm" }, on);
  assert.deepEqual([ok.name, ok.transcript, ok.channel], ["confirmed", "tamatar", "voice_transcript_confirmed"]);
  const fixed = voiceStep(s, { type: "correct", text: " tomato " }, on);
  assert.deepEqual([fixed.transcript, fixed.channel, fixed.corrected], ["tomato", "typed_text_confirmed", true]);
  assert.equal(voiceStep(s, { type: "correct", text: "  " }, on).name, "confirm");          // empty correction ignored
  assert.equal(voiceStep(s, { type: "repeat" }, on).name, "requesting");
});

test("voice: cancel at any stage discards everything", () => {
  const seqs = [[{ type: "press" }], [{ type: "press" }, { type: "granted" }],
    [{ type: "press" }, { type: "granted" }, { type: "stop" }, { type: "result", result: { status: "ok", text: "x" } }]];
  for (const q of seqs) {
    const c = voiceStep(run(q), { type: "cancel" }, on);
    assert.equal(c.name, "idle"); assert.equal(c.transcript, null);
  }
});

test("voice: poor recognition is never usable", () => {
  for (const status of ["low_confidence", "no_speech", "unsupported_language", "invalid_audio", "service_error", "unavailable"]) {
    const s = run([{ type: "press" }, { type: "granted" }, { type: "stop" }, { type: "result", result: { status, text: "maybe" } }]);
    assert.equal(s.name, "retry", status);
    assert.equal(s.transcript, null, status);
    assert.equal(voiceStep(s, { type: "confirm" }, on).name, "retry");                     // cannot confirm a retry
  }
  assert.equal(run([{ type: "press" }, { type: "granted" }, { type: "stop" }, { type: "error" }]).message, "speech.service_error");
});

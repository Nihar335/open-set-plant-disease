// Camera session (M5-A1) with FAKE devices only. These tests check the state machine, error messages and that every
// camera track is stopped; they do NOT show that a real webcam or phone camera works (MANUAL_VERIFICATION.md M1).
import test from "node:test";
import assert from "node:assert/strict";
import { createCameraSession } from "../static/js/media.js";

function fakeStream() {
  const tracks = [{ stopped: false, stop() { this.stopped = true; } }];
  return { tracks, getTracks: () => tracks };
}
const fakeVideo = () => ({ srcObject: null, play: () => Promise.resolve() });
const nav = (impl) => ({ mediaDevices: { getUserMedia: impl } });
const err = (name) => Object.assign(new Error(name), { name });

test("open -> live -> capture -> use stops the stream and returns the still", async () => {
  const s = fakeStream();
  const still = { name: "camera.jpg" };
  const cam = createCameraSession({ nav: nav(async () => s), grab: async () => still });
  const v = fakeVideo();
  await cam.open(v);
  assert.equal(cam.state.name, "live");
  assert.equal(v.srcObject, s);
  await cam.capture(v);
  assert.equal(cam.state.name, "captured");
  assert.equal(cam.use(), still);
  assert.equal(cam.state.name, "idle");
  assert.ok(s.tracks.every((t) => t.stopped));
});

test("retake returns to live without a new permission request", async () => {
  let calls = 0;
  const cam = createCameraSession({ nav: nav(async () => { calls++; return fakeStream(); }), grab: async () => ({}) });
  const v = fakeVideo();
  await cam.open(v);
  await cam.capture(v);
  cam.retake(v);
  assert.equal(cam.state.name, "live");
  assert.equal(calls, 1);
});

test("cancel stops tracks; a stream that arrives after cancel is stopped too", async () => {
  const s = fakeStream();
  let resolve;
  const cam = createCameraSession({ nav: nav(() => new Promise((r) => { resolve = r; })) });
  const p = cam.open(fakeVideo());
  assert.equal(cam.state.name, "starting");
  cam.close();
  resolve(s);
  await p;
  assert.equal(cam.state.name, "idle");
  assert.ok(s.tracks.every((t) => t.stopped));
  assert.equal(cam.active, false);
});

for (const [name, key] of [["NotAllowedError", "photo.camera.denied"], ["NotFoundError", "photo.camera.unavailable"],
  ["NotReadableError", "photo.camera.busy"], ["WeirdError", "photo.camera.error"]]) {
  test(`device error ${name} -> ${key}, nothing left running`, async () => {
    const cam = createCameraSession({ nav: nav(async () => { throw err(name); }) });
    await cam.open(fakeVideo());
    assert.deepEqual([cam.state.name, cam.state.error], ["error", key]);
    assert.equal(cam.active, false);
  });
}

test("unsupported browser and insecure page give their own messages without asking for the camera", async () => {
  let asked = false;
  const a = createCameraSession({ nav: {} });
  await a.open(fakeVideo());
  assert.equal(a.state.error, "photo.camera.unsupported");
  const b = createCameraSession({ nav: nav(async () => { asked = true; return fakeStream(); }), secure: false });
  await b.open(fakeVideo());
  assert.equal(b.state.error, "photo.camera.insecure");
  assert.equal(asked, false);
});

test("a back camera that cannot be satisfied falls back to any camera", async () => {
  const seen = [];
  const cam = createCameraSession({ nav: nav(async (c) => { seen.push(c.video); if (c.video !== true) throw err("OverconstrainedError"); return fakeStream(); }) });
  await cam.open(fakeVideo());
  assert.equal(cam.state.name, "live");
  assert.equal(seen.length, 2);
  assert.equal(seen[1], true);
});

test("a failed frame grab is an error and stops the camera", async () => {
  const s = fakeStream();
  const cam = createCameraSession({ nav: nav(async () => s), grab: async () => null });
  await cam.open(fakeVideo());
  assert.equal(await cam.capture(fakeVideo()), null);
  assert.equal(cam.state.name, "error");
  assert.ok(s.tracks.every((t) => t.stopped));
});

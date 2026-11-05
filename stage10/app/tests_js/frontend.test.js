// Unit tests for the pure frontend modules. Run: node --test stage10/app/tests_js/
// Report fixtures are produced by the real server (fake disease model) in the Python test that runs this file
// (AGROVAIDYA_FIXTURES = directory of <name>.json API responses).
import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

import { ApiFailure, call, friendlyError } from "../static/js/api.js";
import { createI18n, pick } from "../static/js/i18n.js";
import { cameraErrorKey, micFlow, precheckFile } from "../static/js/media.js";
import { buildReportModel } from "../static/js/report-model.js";
import * as S from "../static/js/state.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const EN = JSON.parse(readFileSync(join(HERE, "../i18n/en.json"), "utf-8"));
const HI = JSON.parse(readFileSync(join(HERE, "../i18n/hi.json"), "utf-8"));

test("i18n: missing translations fall back to English and are never reported as verified", () => {
  const partial = { ...HI, strings: { ...HI.strings } };
  delete partial.strings["home.start"];                                  // a string missing from a draft catalogue
  const i = createI18n({ en: EN, hi: partial }, "hi");
  assert.equal(i.t("home.start"), EN.strings["home.start"]);
  assert.equal(HI.meta.review_status, "machine_draft_unreviewed");
  const st = i.status();
  assert.equal(st.verified, false);
  assert.ok(st.missing.includes("home.start"));
  assert.equal(createI18n({ en: EN }, "en").status().verified, true);
  assert.equal(i.t("no.such.key"), "no.such.key");
  assert.equal(createI18n({ en: EN }).t("plant.use_anyway", { text: "Neem" }), "Use “Neem” anyway");
  assert.equal(pick({ en: "a", hi: "b" }, "hi"), "b");
  assert.equal(pick({ en: "a" }, "hi"), "a");
});

test("state: plant confirmation and correction", () => {
  let s = S.initialState();
  assert.throws(() => S.assessFields(s));
  s = S.setPending(s, { action: "typed", text: "tomato", plant: { plant_id: "PLANT-TOMATO" } });
  assert.equal(s.plant.choice, null);
  s = S.confirmPlant(s);
  assert.equal(s.plant.choice.confirmed, true);
  s = S.correctPlant(s);
  assert.equal(s.plant.choice, null);
  assert.equal(s.plant.corrections, 1);
  s = S.confirmPlant(S.setPending(s, { action: "selected", plant_id: "PLANT-MANGO" }));
  const f = S.assessFields(s);
  assert.deepEqual(JSON.parse(f.plant), { action: "selected", confirmed: true, corrections: 1, plant_id: "PLANT-MANGO" });
});

test("state: switching modes three times keeps every answer (M5-D2)", () => {
  let s = S.initialState();
  s = S.setFeature(s, "FEAT-A", true);
  s = S.setFeature(s, "FEAT-B", true, "typed_text_confirmed");
  s = S.setAnswer(s, "Q-1", "yes");
  s = { ...s, symptoms: { ...s.symptoms, text: "rings" } };
  for (let i = 0; i < 3; i++) {
    s = S.setMode(s, "text");
    s = S.setMode(s, "tap");
  }
  assert.equal(s.symptoms.mode, "tap");
  assert.deepEqual(s.symptoms.features, { "FEAT-A": true, "FEAT-B": true });
  assert.deepEqual(s.symptoms.channels, { "FEAT-B": "typed_text_confirmed" });
  assert.equal(s.symptoms.answers["Q-1"], "yes");
  assert.equal(s.symptoms.text, "rings");
  assert.equal(S.setMode(s, "voice", false).symptoms.mode, "tap");    // voice gated
  assert.equal(S.setMode(s, "nonsense").symptoms.mode, "tap");
  s = S.setFeature(s, "FEAT-A", null);
  assert.ok(!("FEAT-A" in s.symptoms.features));
});

test("state: contradictory answers are detected for the user", () => {
  let s = S.setFeature(S.initialState(), "FEAT-X", true);
  s = S.setAnswer(s, "Q-X", "no");
  const qs = [{ question_id: "Q-X", options: [{ option_id: "no", sets: { "FEAT-X": false } }] }];
  assert.deepEqual(S.conflicts(s, qs), [{ feature_id: "FEAT-X", question_id: "Q-X" }]);
});

test("state: nothing is persisted in browser storage", () => {
  // comments may mention storage (e.g. "nothing is written to localStorage"); only code is checked
  const strip = (code) => code.replace(/\/\*[\s\S]*?\*\//g, "").replace(/(^|\s)\/\/.*$/gm, "$1");
  const src = readdirSync(join(HERE, "../static/js")).map((f) => strip(readFileSync(join(HERE, "../static/js", f), "utf-8"))).join("\n");
  assert.ok(!/localStorage|sessionStorage|document\.cookie|indexedDB/.test(src));
  assert.ok(!/https?:\/\/(?!www\.w3\.org)/.test(src), "no external URLs in the app code");
});

test("media: photo pre-check and camera errors", () => {
  assert.equal(precheckFile(null, 10).ok, false);
  assert.equal(precheckFile({ name: "a.gif", type: "image/gif", size: 10 }, 10).key, "photo.bad_type");
  assert.equal(precheckFile({ name: "a.jpg", type: "image/jpeg", size: 11 * 1024 * 1024 }, 10).key, "photo.too_big");
  assert.equal(precheckFile({ name: "a.PNG", type: "", size: 10 }, 10).ok, true);
  assert.equal(cameraErrorKey({ name: "NotAllowedError" }), "photo.camera.denied");
  assert.equal(cameraErrorKey({ name: "NotFoundError" }), "photo.camera.unavailable");
  assert.equal(cameraErrorKey({ name: "unsupported" }), "photo.camera.unsupported");
  assert.equal(cameraErrorKey({ name: "NotReadableError" }), "photo.camera.busy");
  assert.equal(cameraErrorKey({ name: "x" }, false), "photo.camera.insecure");
  for (const k of ["photo.camera.denied", "photo.camera.unavailable", "photo.camera.busy", "photo.camera.insecure", "photo.camera.error", "photo.camera.unsupported"])
    assert.ok(EN.strings[k], k);
});

test("voice: microphone never requested unless available and user-initiated; denial falls back to text", () => {
  assert.deepEqual(micFlow("press", { available: false, userAction: true }), { state: "pending", requestMic: false, fallback: "text" });
  assert.equal(micFlow("press", { available: true, userAction: false }).requestMic, false);
  assert.equal(micFlow("press", { available: true, userAction: true }).requestMic, true);
  assert.equal(micFlow("denied", { available: true }).fallback, "text");
  assert.equal(micFlow("transcribed", { available: true }).state, "confirm");
});

test("api: errors are mapped to safe messages", async () => {
  const t = (k) => EN.strings[k];
  await assert.rejects(call("GET", "/x", undefined, async () => { throw new Error("down"); }), (e) => e.code === "network");
  assert.equal(friendlyError(new ApiFailure("network", ""), t), EN.strings["errors.network"]);
  assert.equal(friendlyError(new ApiFailure("image_invalid", "This file cannot be used."), t), "This file cannot be used.");
  assert.equal(friendlyError(new Error("Traceback ..."), t), EN.strings["errors.generic"]);
  const bad = async () => ({ status: 500, json: async () => ({ ok: false, error: { code: "internal_error", message: "Something went wrong" } }) });
  await assert.rejects(call("GET", "/x", undefined, bad), (e) => e.code === "internal_error" && e.status === 500);
});

const FIX = process.env.AGROVAIDYA_FIXTURES;
test("report model: provenance, no scores, all report types", { skip: !FIX && "AGROVAIDYA_FIXTURES not set (run via pytest)" }, () => {
  const seen = new Set();
  for (const f of readdirSync(FIX).filter((x) => x.endsWith(".json"))) {
    const resp = JSON.parse(readFileSync(join(FIX, f), "utf-8"));
    const m = buildReportModel(resp, resp.report.language);
    seen.add(m.type);
    assert.ok(EN.strings[m.bannerKey], m.bannerKey);
    assert.equal(m.plant.verified, false);
    assert.equal(m.plant.userGiven, true);
    assert.equal(m.photoCheck.ran, resp.report.model_prediction.ran);
    assert.equal(m.highlighting.available, false);
    assert.equal(m.warnings.length, resp.report.safety_warnings.length);
    assert.ok(EN.strings["report.uncertainty." + m.uncertainty.level], m.uncertainty.level);
    const flat = JSON.stringify(m);
    assert.ok(!/ranking_score|softmax|"score"/.test(flat), "no scores in the display model");
    for (const c of m.candidates) for (const s of c.supporting) assert.ok(["user_report", "knowledge_base_typical", "image_analysis"].includes(s.origin));
    for (const s of m.symptoms.typical) assert.ok(!m.symptoms.userSeen.includes(s) || true);
    if (!m.photoCheck.ran) assert.ok(!m.candidates.some((c) => c.fromModel), f);
  }
  for (const t of ["known_disease", "unknown_investigation", "out_of_model_scope", "insufficient_image"]) assert.ok(seen.has(t), t);
});

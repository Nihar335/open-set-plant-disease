// In-browser UI self-test (development only). Open http://127.0.0.1:8765/?selftest=1#/home
// Drives the real DOM against the real local API (including the real disease model for a Tomato case) and records
// results in window.__selftest and in a <pre id="selftest-results">. It does not prove accessibility compliance:
// it checks labels, names, focus targets, contrast of the main colour pairs and horizontal overflow only.
import { __test } from "./app.js";
import { micFlow, precheckFile } from "./media.js";

const { store, S } = __test;
const results = [];
const wait = (ms) => new Promise((r) => setTimeout(r, ms));
async function until(fn, ms = 20000, step = 50) {
  const t0 = Date.now();
  for (;;) {
    const v = fn();
    if (v) return v;
    if (Date.now() - t0 > ms) throw new Error("timed out waiting");
    await wait(step);
  }
}
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => [...document.querySelectorAll(sel)];
const byText = (sel, text) => $$(sel).find((e) => e.textContent.trim().includes(text));
function check(name, cond, detail = "") {
  results.push({ name, pass: !!cond, detail: cond ? "" : String(detail) });
}
async function step(name, fn) {
  try {
    await fn();
  } catch (e) {
    results.push({ name, pass: false, detail: "error: " + (e && e.message) });
  }
}
const nav = async (route) => {
  location.hash = "#/" + route;
  await until(() => store.get().route === route);
  await wait(30);
};
function type(el, value) {
  el.value = value;
  el.dispatchEvent(new Event("input", { bubbles: true }));
}

// ---------- accessibility basics ----------
function accName(el) {
  if (el.getAttribute("aria-label")) return el.getAttribute("aria-label");
  const lb = el.getAttribute("aria-labelledby");
  if (lb) return lb.split(/\s+/).map((id) => (document.getElementById(id) || {}).textContent || "").join(" ").trim();
  if (el.labels && el.labels.length) return [...el.labels].map((l) => l.textContent).join(" ").trim();
  if (el.type === "radio") { const fs = el.closest("fieldset"); const lg = fs && fs.querySelector("legend"); return lg ? lg.textContent.trim() : ""; }
  if (el.tagName === "IMG") return el.getAttribute("alt") || "";
  return (el.textContent || el.title || "").trim();
}
function a11yScan(view) {
  const bad = [];
  for (const el of $$("#app input:not([aria-hidden=true]), #app select, #app textarea, #app button, #app a[href], #app img"))
    if (!accName(el)) bad.push(el.outerHTML.slice(0, 80));
  check(`${view}: every control and image has an accessible name`, !bad.length, bad.join(" | "));
  check(`${view}: exactly one h1`, $$("#app h1").length === 1, $$("#app h1").length);
  check(`${view}: landmarks (header, nav, main)`, $("header") && $("nav") && $("main"));
  const unreachable = $$("#app button:not([disabled]), #app a[href], #app input:not([type=hidden]):not([aria-hidden=true]), #app select, #app textarea")
    .filter((e) => e.tabIndex < 0);
  check(`${view}: enabled controls are keyboard-reachable`, !unreachable.length, unreachable.map((e) => e.outerHTML.slice(0, 60)).join(" | "));
  check(`${view}: no horizontal overflow at ${innerWidth}px`, document.documentElement.scrollWidth <= innerWidth + 1,
    `${document.documentElement.scrollWidth} > ${innerWidth}`);
  check(`${view}: focus moved to the page heading`, document.activeElement && document.activeElement.id === "page-title", document.activeElement && document.activeElement.outerHTML.slice(0, 60));
}

function lum(rgb) {
  const [r, g, b] = rgb.match(/\d+(\.\d+)?/g).slice(0, 3).map(Number).map((v) => {
    v /= 255;
    return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}
// Effective background: composite translucent layers over the first opaque ancestor background.
function bgOf(el) {
  const layers = [];
  let base = [255, 255, 255];
  for (let e = el; e; e = e.parentElement) {
    const m = (getComputedStyle(e).backgroundColor.match(/[\d.]+/g) || []).map(Number);
    if (m.length < 3) continue;
    const a = m.length > 3 ? m[3] : 1;
    if (a === 0) continue;
    if (a >= 1) { base = m.slice(0, 3); break; }
    layers.push([m[0], m[1], m[2], a]);
  }
  let c = base;
  for (const [r, g, b, a] of layers.reverse()) c = [r * a + c[0] * (1 - a), g * a + c[1] * (1 - a), b * a + c[2] * (1 - a)];
  return `rgb(${c.map(Math.round).join(",")})`;
}
function contrastOf(el) {
  const a = lum(getComputedStyle(el).color), b = lum(bgOf(el));
  return (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05);
}
function contrastScan(view) {
  const low = [];
  for (const el of $$("#app p, #app li, #app h1, #app h2, #app h3, #app .btn, #app .badge, #app .tag, #app label, #app legend, #app .navlink, #app .eyebrow, #app .hint")) {
    if (!el.offsetParent || !el.textContent.trim() || el.closest("[disabled]") || el.disabled) continue;
    const cr = contrastOf(el);
    const big = parseFloat(getComputedStyle(el).fontSize) >= 24;
    if (cr < (big ? 3 : 4.5)) low.push(`${el.tagName}.${el.className}: ${cr.toFixed(2)} "${el.textContent.trim().slice(0, 30)}"`);
  }
  check(`${view}: text contrast >= 4.5:1 (3:1 large)`, !low.length, low.slice(0, 6).join(" | "));
}

function syntheticPhoto(w = 640, h = 480) {
  const c = document.createElement("canvas");
  c.width = w; c.height = h;
  const g = c.getContext("2d");
  g.fillStyle = "#6b7f4a"; g.fillRect(0, 0, w, h);
  g.fillStyle = "#3f8f3a"; g.beginPath(); g.ellipse(w / 2, h / 2, w * 0.4, h * 0.3, 0.3, 0, Math.PI * 2); g.fill();
  for (let i = 0; i < 40; i++) {
    g.fillStyle = i % 2 ? "#5a3a1a" : "#c9b23a";
    g.beginPath(); g.arc(w * 0.25 + ((i * 37) % (w * 0.5)), h * 0.25 + ((i * 53) % (h * 0.5)), 6 + (i % 7) * 2, 0, Math.PI * 2); g.fill();
  }
  return new Promise((r) => c.toBlob((b) => r(new File([b], "leaf.jpg", { type: "image/jpeg" })), "image/jpeg", 0.92));
}

async function plantByTyping(text) {
  await nav("plant");
  if (store.get().plant.choice || store.get().plant.pending) store.update(S.correctPlant);
  store.update((x) => ({ ...x, plant: { ...x.plant, resolution: null } }));
  await until(() => $("#plant-input"));
  type($("#plant-input"), text);
  $("form").requestSubmit();
  await until(() => store.get().plant.pending || (store.get().plant.resolution && store.get().plant.resolution.status !== "resolved"));
}

async function runFlow(name, setupPlant, expect) {
  await step(name, async () => {
    store.update(() => S.initialState());
    // SC-M5-02 order: photo first, then the plant name, then the report (no symptom form)
    await nav("photo");
    await __test.acceptFile(await syntheticPhoto(), "file");
    await until(() => store.get().photo.check);
    check(`${name}: photo step leads to the plant step`, !!$("#photo-next") && !$("#photo-next").disabled);
    await setupPlant();
    store.update(S.confirmPlant);
    await nav("plant");
    await until(() => $("#submit"));
    $("#submit").click();
    await until(() => store.get().route === "report" || store.get().submit.error, 120000);
    check(`${name}: report produced`, store.get().route === "report", JSON.stringify(store.get().submit.error));
    if (store.get().route !== "report") return;
    await wait(50);
    const txt = $("#app").textContent;                         // includes the collapsed full technical report
    const r = store.get().result;
    check(`${name}: report type ${expect.type}`, r.report.report_type === expect.type, r.report.report_type);
    check(`${name}: disease model ran = ${expect.ran}`, r.report.model_prediction.ran === expect.ran);
    check(`${name}: plant shown as user-given`, txt.includes(__test.t("plant.user_given")));
    check(`${name}: photo-check status stated`, txt.includes(__test.t(expect.ran ? "report.photo_check.ran" : "report.photo_check.not_run")));
    check(`${name}: highlighting shown as unavailable`, txt.includes(__test.t("report.highlight.unavailable")));
    check(`${name}: read-aloud player offered with play, repeat, next, stop and speed`,
      !!($("#player-play") && $("#player-repeat") && $("#player-next") && $("#player-stop") && $("#player-speed")) && !r.spoken_script.segments.some((sg) => /\d\s*%/.test(sg.text)));
    check(`${name}: no percentages or 'new species' wording`, !/\d\s*%|new species|newly discovered/i.test(txt));
    check(`${name}: every safety warning displayed`, r.report.safety_warnings.every((w) => txt.includes(w.en.slice(0, 40))));
    check(`${name}: expert advice present`, txt.includes("Krishi Vigyan Kendra"));
    if (expect.ran === false) check(`${name}: no model-based candidate`, !r.report.candidate_diagnoses.some((c) => c.ranking_method === "model_class_evidence"));
    const sm = r.summary;
    check(`${name}: summary shows crop, signs and safety sections`, !!$("#crop-h") && !!$("#signs-h") && !!$("#notes-h"));
    if (sm.crop.in_knowledge_base) {
      check(`${name}: biological name, temperature, soil and nutrients shown`, txt.includes(sm.crop.scientific_name) && !!$("#temp-h") && !!$("#soil-h") &&
        (!sm.crop.temperature || txt.includes(sm.crop.temperature.text.slice(0, 20))));
    }
    check(`${name}: diseases section matches summary kind (${sm.kind})`, sm.kind === "known" || sm.kind === "possible" ?
      $$("article.cand[id^=disease-]").length === sm.diseases.length && sm.diseases.length >= 1 : $$("article.cand[id^=disease-]").length === 0);
    if (sm.kind === "possible") check(`${name}: each possible disease lists checks to confirm`, $$("article.cand[id^=disease-] .checks").length === sm.diseases.length);
    check(`${name}: every report section has a listen button`, $$(".rsec-head .btn-listen").length >= 5 && !!$("#listen-all"));
    a11yScan(`report(${name})`);
    contrastScan(`report(${name})`);
  });
}

// SYNTHETIC camera: a canvas stream stands in for getUserMedia. This checks the app's camera flow and track clean-up;
// it is NOT evidence that a real webcam or phone camera works (that is MANUAL_VERIFICATION.md M1).
async function cameraFlow() {
  const md = navigator.mediaDevices;
  if (!md) { check("camera API present in this browser", false, "no mediaDevices"); return; }
  const real = md.getUserMedia;
  const c = document.createElement("canvas"); c.width = 640; c.height = 480;
  const g = c.getContext("2d");
  const paint = () => { g.fillStyle = "#777"; g.fillRect(0, 0, 640, 480); g.fillStyle = "#3f8f3a"; g.beginPath();
    g.ellipse(320, 240, 240, 160, 0.2, 0, 7); g.fill(); for (let i = 0; i < 30; i++) { g.fillStyle = i % 2 ? "#5a3a1a" : "#c9b23a"; g.beginPath();
      g.arc(150 + ((i * 41) % 340), 120 + ((i * 59) % 240), 6 + (i % 5) * 2, 0, 7); g.fill(); } };
  const draw = setInterval(paint, 50);
  paint();
  const streams = [];
  const ended = () => streams.every((st) => st.getTracks().every((t) => t.readyState === "ended"));
  md.getUserMedia = async () => { const st = c.captureStream(15); streams.push(st); return st; };
  try {
    store.update(() => S.initialState());
    await nav("photo");
    check("Take photo button visible on an empty photo step", !!$("#take-photo") && !!$("#choose-photo"));
    $("#take-photo").click();
    await until(() => $("#camera-capture"), 8000);
    check("synthetic camera: live preview with Capture and Cancel", !!$("#camera-video") && !!$("#camera-cancel"));
    $("#camera-cancel").click(); await wait(80);
    check("synthetic camera: Cancel stops every track", ended() && !!$("#take-photo"));
    $("#take-photo").click();
    await until(() => $("#camera-capture"), 8000);
    await wait(400);
    $("#camera-capture").click();
    await until(() => $("#camera-use"), 8000);
    check("synthetic camera: still shown with Use and Retake", !!$("#camera-still") && !!$("#camera-retake"));
    $("#camera-retake").click();
    await until(() => $("#camera-capture"), 8000);
    await wait(300);
    $("#camera-capture").click();
    await until(() => $("#camera-use"), 8000);
    $("#camera-use").click();
    await until(() => store.get().photo.check || store.get().photo.error, 20000);
    check("synthetic camera: captured still goes through the same photo check", store.get().photo.source === "camera" && !!store.get().photo.check,
      store.get().photo.error);
    check("synthetic camera: tracks stopped after Use", ended());
    check("Take new photo and Upload new photo offered next to the preview", !!$("#take-photo") && !!$("#choose-photo") && !!$("#remove-photo"));
    $("#take-photo").click();
    await until(() => $("#camera-capture"), 8000);
    await nav("home");
    check("synthetic camera: leaving the photo step stops the camera", ended());
    md.getUserMedia = async () => { throw Object.assign(new Error("denied"), { name: "NotAllowedError" }); };
    await nav("photo");
    $("#take-photo").click();
    await until(() => $("#camera-upload"), 8000);
    check("camera denial shows a message and an Upload photo button", $("#app").textContent.includes(__test.t("photo.camera.denied")));
    $("#camera-cancel").click(); await wait(50);
  } finally {
    md.getUserMedia = real;
    clearInterval(draw);
  }
}

async function languages() {
  const codes = __test.K.health.ui_languages.map((l) => l.code);
  check("six interface languages offered", codes.join(",") === "en,hi,bn,te,mr,ta", codes.join(","));
  await nav("home");
  const en = $("h1").textContent;
  for (const code of codes) {
    $("#lang-" + code).click();
    await until(() => store.get().uiLang === code && document.documentElement.lang === code);
    await wait(40);
    const h1 = $("h1").textContent;
    check(`${code}: whole page switches language`, code === "en" ? h1 === en : h1 !== en && !h1.includes("plant sick"));
    check(`${code}: draft notice shown only for non-English`, code === "en" ? !$(".container > .notice-warn") : !!$(".container > .notice-warn"));
    check(`${code}: listen button present`, !!$("#listen-home"));
  }
  $("#lang-en").click();
  await until(() => store.get().uiLang === "en");
}

export async function runSelfTest() {
  const t0 = Date.now();
  await step("languages", languages);
  await step("camera (synthetic stream)", cameraFlow);
  // 1. every static view: names, headings, keyboard reachability, overflow, contrast
  for (const v of ["home", "plant", "help"]) {
    await step("view " + v, async () => { await nav(v); await wait(60); a11yScan(v); contrastScan(v); });
  }
  // 2. plant name: alias + normalisation, correction, unknown, don't know
  await step("plant alias", async () => {
    await plantByTyping("  CORN ");
    check("'  CORN ' resolves to Maize (pending confirmation)", store.get().plant.pending && store.get().plant.pending.plant.plant_id === "PLANT-MAIZE");
    check("confirmation card shown before continuing", !!byText("button", __test.t("plant.confirm.yes")));
    store.update(S.confirmPlant);
    store.update(S.correctPlant);
    check("correction clears the choice and counts it", !store.get().plant.choice && store.get().plant.corrections === 1);
  });
  await step("plant unknown", async () => {
    await plantByTyping("Coconut");
    const res = store.get().plant.resolution;
    check("unknown name gives clarification, not an identity", res.status === "unknown" && !store.get().plant.pending);
    byText("button", "anyway").click();
    await until(() => store.get().plant.pending);
    check("'use anyway' keeps the user's name as unlisted", store.get().plant.pending.action === "unlisted");
  });
  await step("plant misspelt", async () => {
    await plantByTyping("tomatoe");
    const res = store.get().plant.resolution;
    check("misspelling is suggested, never auto-mapped", res.status === "needs_clarification" && !store.get().plant.pending &&
      res.suggestions.some((p) => p.plant_id === "PLANT-TOMATO"));
  });
  await step("plant empty", async () => {
    await plantByTyping("   ");
    check("empty name gives a validation message", store.get().plant.resolution.status === "empty" && !store.get().plant.pending);
  });
  // 3. mode switching keeps state (M5-D2)
  await step("mode switching", async () => {
    store.update(() => S.initialState());
    store.update((x) => S.setPending(x, { action: "selected", plant_id: "PLANT-TOMATO", plant: __test.K.plants.find((p) => p.plant_id === "PLANT-TOMATO") }));
    store.update(S.confirmPlant);
    await nav("symptoms");
    const radio = $$('input[type=radio][name="f-FEAT-CONCENTRIC-RINGS"]')[0];
    radio.click();
    await wait(30);
    for (let i = 0; i < 3; i++) {
      $("#mode-text").click(); await wait(30);
      if (i === 0) { type($("#symptom-text"), "rings on the lower leaves"); }
      check(`switch ${i + 1}: text mode keeps typed text`, $("#symptom-text").value === "rings on the lower leaves");
      $("#mode-tap").click(); await wait(30);
      check(`switch ${i + 1}: tap mode keeps ticked symptom`, $$('input[name="f-FEAT-CONCENTRIC-RINGS"]')[0].checked &&
        store.get().symptoms.features["FEAT-CONCENTRIC-RINGS"] === true);
      check(`switch ${i + 1}: plant kept`, store.get().plant.choice && store.get().plant.choice.plant_id === "PLANT-TOMATO");
    }
    // voice mode is part of the switching test when local speech is available; otherwise it must be disabled
    const voiceUp = !!(__test.K.speech && __test.K.speech.speech_to_text.available);
    for (let i = 0; i < 3 && voiceUp; i++) {
      $("#mode-voice").click(); await wait(30);
      check(`voice switch ${i + 1}: voice mode shown with a microphone button`, store.get().symptoms.mode === "voice" && !!$("#mic-symptoms"));
      $("#mode-text").click(); await wait(30);
      check(`voice switch ${i + 1}: typed text still kept`, $("#symptom-text").value === "rings on the lower leaves");
      $("#mode-tap").click(); await wait(30);
      check(`voice switch ${i + 1}: ticked symptom and plant still kept`, store.get().symptoms.features["FEAT-CONCENTRIC-RINGS"] === true &&
        store.get().plant.choice && store.get().plant.choice.plant_id === "PLANT-TOMATO");
    }
    if (!voiceUp) check("voice mode disabled when speech is unavailable", $("#mode-voice").disabled);
    check("microphone is never requested when voice is unavailable", micFlow("press", { available: false, userAction: true }).requestMic === false);
    a11yScan("symptoms"); contrastScan("symptoms");
  });
  // 4. typed description -> suggestions need confirmation
  await step("typed description", async () => {
    $("#mode-text").click(); await wait(30);
    type($("#symptom-text"), "brown rings, yellow halo, and zzqx");
    $("#symptom-find").click();
    await until(() => store.get().symptoms.suggest);
    const sg = store.get().symptoms.suggest;
    check("suggestions returned and nothing added automatically", sg.suggestions.length > 0 && !("FEAT-YELLOW-HALO" in store.get().symptoms.features));
    check("unmatched words shown as not understood", sg.not_understood.includes("zzqx"));
    const add = $$(".suggest-list button").find((b) => b.getAttribute("aria-label").includes("Yellow halo"));
    add.click(); await wait(30);
    check("confirmed suggestion recorded with typed_text_confirmed channel", store.get().symptoms.channels["FEAT-YELLOW-HALO"] === "typed_text_confirmed");
    $("#mode-tap").click();
  });
  // 5. photo pre-checks and server checks
  await step("photo checks", async () => {
    check("GIF refused before upload", !precheckFile(new File(["x"], "a.gif", { type: "image/gif" }), 10).ok);
    check("oversized photo refused before upload", !precheckFile({ name: "a.jpg", type: "image/jpeg", size: 11 * 1024 * 1024 }, 10).ok);
    await nav("photo");
    await __test.acceptFile(await syntheticPhoto(160, 120), "file");
    await until(() => store.get().photo.check);
    check("server flags a too-small photo with a retake request", !store.get().photo.check.passed &&
      store.get().photo.check.retake_requests.some((r) => r.issue === "too_small"));
    a11yScan("photo"); contrastScan("photo");
  });
  // 5b. farmer-guided marking (user-selected areas; never AI)
  await step("marking", async () => {
    await __test.acceptFile(await syntheticPhoto(), "file");
    await until(() => store.get().photo.check && $(".mark-stage"));
    const stage = $(".mark-stage"), r = stage.getBoundingClientRect();
    const ev = (type, fx, fy, id, pt) => $(".mark-stage").dispatchEvent(new PointerEvent(type, { clientX: r.left + r.width * fx,
      clientY: r.top + r.height * fy, pointerId: id, pointerType: pt, bubbles: true }));
    ev("pointerdown", 0.5, 0.5, 11, "touch"); ev("pointerup", 0.5, 0.5, 11, "touch");
    await wait(40);
    check("touch tap adds a spot at the tapped place", store.get().photo.marks.length === 1 && store.get().photo.marks[0].shape === "circle" &&
      Math.abs(store.get().photo.marks[0].x - 0.5) < 0.01);
    $("#grid-top-left").click(); await wait(40);
    check("grid alternative adds an area without precise pointing", store.get().photo.marks[1].via === "grid");
    check("marks are drawn over the photo", $$(".mark-stage .mark-shape").length === 2);
    check("user-selected label shown", $("#app").innerText.includes(__test.t("mark.label")));
    $("#mark-original").click(); await wait(40);
    check("original-photo view hides the marks", !$(".mark-stage svg"));
    $("#mark-show").click(); await wait(40);
    byText("button", __test.t("mark.remove", { n: 1 })).click(); await wait(40);
    check("a single mark can be removed", store.get().photo.marks.length === 1 && store.get().photo.marks[0].via === "grid");
    $("#mark-undo").click(); await wait(40);
    check("undo removes the last mark", store.get().photo.marks.length === 0 && $("#mark-undo").disabled);
    await nav("home"); await nav("photo"); await wait(60);                // fresh arrival: heading focus is part of the scan
    a11yScan("photo+marking"); contrastScan("photo+marking");
  });
  // 5c. voice (real local engines; recordings come from the speech-output engine, NOT a microphone)
  await step("voice", async () => {
    const st = await (await fetch("/api/v1/speech/status")).json();
    if (!st.data.speech_to_text.available) { check("voice engines available on this computer", false, "speech not available"); return; }
    const { api } = await import("./api.js");
    const V = await import("./voice.js");
    const wavOf = async (text, lang = "en") => {
      const ctx = new AudioContext();
      const buf = await ctx.decodeAudioData(await (await api.synthesize(text, lang)).arrayBuffer());
      await ctx.close();
      return new Blob([V.encodeWav(V.downsample(buf.getChannelData(0), buf.sampleRate))], { type: "audio/wav" });
    };
    store.update(() => S.initialState());
    await nav("plant");
    $("#mic-plant").click(); await wait(60);
    check("first voice use shows the consent explanation", !!$("#voice-consent-accept") && $("#app").innerText.includes(__test.t("voice.consent.1")));
    byText("button", __test.t("voice.consent.decline")).click(); await wait(60);
    check("declining consent falls back to typing", !$(".voice-panel") && !store.get().voiceConsent);
    const f = await __test.voiceFromWav("plant", await wavOf("Potato"));
    check("speech is shown for confirmation, never used directly", f.name === "confirm" && !store.get().plant.pending);
    check("recognised plant name shown", $("#voice-edit").value === "potato");
    $("#voice-use").click();
    await until(() => store.get().plant.pending);
    check("confirmed voice goes through the same resolver and still needs plant confirmation",
      store.get().plant.pending.plant.plant_id === "PLANT-POTATO" && store.get().plant.pending.channel === "voice_transcript_confirmed" &&
      !store.get().plant.choice);
    const g = await __test.voiceFromWav("plant", new Blob([V.encodeWav(new Float32Array(16000))], { type: "audio/wav" }));
    check("silence is reported, never turned into a plant name", g.name === "retry" && g.message === "speech.no_speech");
    $("#voice-again") && byText("button", __test.t("voice.type_instead")).click(); await wait(40);
    store.update(S.confirmPlant);
    await nav("symptoms");
    $("details.questions").open = true;
    const q = await __test.voiceFromWav("q:Q-CONCENTRIC-RINGS", await wavOf("Yes"));
    check("spoken answer shown for confirmation", q.name === "confirm");
    $("#voice-use").click(); await wait(60);
    check("mapped answer needs a second confirmation", !!$("#voice-option-yes") && !store.get().symptoms.answers["Q-CONCENTRIC-RINGS"]);
    $("#voice-option-yes").click(); await wait(60);
    check("answer set only after confirmation", store.get().symptoms.answers["Q-CONCENTRIC-RINGS"] === "yes");
    byText("button", __test.t("questions.type_answer")).click(); await wait(60);
    $("#voice-edit").value = "not sure"; $("#voice-use").click(); await wait(60);
    check("a question can be answered by typing ('not sure')", !!$("#voice-option-yes"));
    $("#voice-option-yes").click(); await wait(60);
    const figs = $$(".q-figure img");
    check("question diagrams shown with alternative text and a 'not a photo' caption",
      figs.length >= 9 && figs.every((i) => i.alt.length > 15) && $("#app").textContent.includes(__test.t("questions.diagram_caption")));
    await nav("photo"); await nav("symptoms"); await wait(60);
    a11yScan("symptoms+voice"); contrastScan("symptoms+voice");
  });
  // 6. full flows through the real API (Tomato uses the real frozen disease model)
  await runFlow("tomato", async () => { await plantByTyping("tomato"); }, { type: undefined, ran: true });
  results.forEach((r) => { if (r.name === "tomato: report type undefined") { r.pass = ["known_disease", "unknown_investigation"].includes(store.get().result && store.get().result.report.report_type); r.name = "tomato: report type is known_disease or unknown_investigation"; } });
  await runFlow("mango", async () => { await plantByTyping("mango"); }, { type: "out_of_model_scope", ran: false });
  await runFlow("dont-know", async () => { await nav("plant"); store.update((x) => S.setPending(x, { action: "dont_know" })); }, { type: "unknown_investigation", ran: false });
  await runFlow("unlisted", async () => { await plantByTyping("Coconut"); store.update((x) => S.setPending(x, { action: "unlisted", text: "Coconut" })); },
    { type: "out_of_model_scope", ran: false });

  const failed = results.filter((r) => !r.pass);
  window.__selftest = { done: true, passed: results.length - failed.length, failed: failed.length, results, seconds: (Date.now() - t0) / 1000,
    viewport: [innerWidth, innerHeight] };
  const pre = document.createElement("pre");
  pre.id = "selftest-results";
  pre.style.whiteSpace = "pre-wrap";
  pre.textContent = `SELFTEST ${failed.length ? "FAIL" : "PASS"}: ${results.length - failed.length}/${results.length}\n` +
    failed.map((f) => `FAIL ${f.name}: ${f.detail}`).join("\n");
  document.body.append(pre);
  return window.__selftest;
}

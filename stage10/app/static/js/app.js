// AgroVaidya farmer-facing web app (Milestone 5). Views only: every decision is made by the local server.
import { api, ApiFailure, friendlyError } from "./api.js";
import { createI18n, pick } from "./i18n.js";
import { createCameraSession, precheckFile, voiceStep } from "./media.js";
import { createPlayer, matchOption, MAX_SECONDS as VMAX, startRecording } from "./voice.js";
import * as M from "./marking.js";
import { buildReportModel } from "./report-model.js";
import * as S from "./state.js";

const store = S.createStore();
const K = { plants: [], features: [], questions: [], states: [], health: null, maxMb: 10, catalogues: {} };
let I = createI18n({ en: { strings: {} } });
const t = (k, p) => I.t(k, p);
// Camera (M5-A1): one session for the tab. Every change re-renders; a captured still gets a temporary object URL that
// is revoked as soon as the still is used, retaken or discarded. Nothing is recorded and nothing leaves this tab
// until the user presses "Use this photo", which sends the still through the same path as an uploaded file.
let stillUrl = null;
let focusNext = null;
const camSession = createCameraSession({ onChange: (st) => {
  if (st.still && !stillUrl) stillUrl = URL.createObjectURL(st.still);
  if (!st.still && stillUrl) { URL.revokeObjectURL(stillUrl); stillUrl = null; }
  if (st.name === "error") announce(t(st.error));
  if (st.name === "live") { announce(t("photo.camera.live_hint")); focusNext = "camera-capture"; }
  render();
} });
const cameraOpen = () => camSession.state.name !== "idle";
const closeCameraIfOpen = () => { if (cameraOpen()) camSession.close(); };
// Leaving the tab, minimising the app or closing the page always switches the camera off.
document.addEventListener("visibilitychange", () => { if (document.visibilityState === "hidden") closeCameraIfOpen(); });
window.addEventListener("pagehide", closeCameraIfOpen);
let lastRoute = null;

// ------------------------------------------------------------------------------------------------ DOM helper
export function h(tag, attrs = {}, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === false || v == null) continue;
    if (k === "class") el.className = v;
    else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2), v);
    else if (k === "text") el.textContent = v;
    else if (v === true) el.setAttribute(k, "");
    else el.setAttribute(k, v);
  }
  for (const c of kids.flat(Infinity)) {
    if (c == null || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

const icon = (name) => {
  const paths = {
    leaf: "M5 21c0-9 6-15 16-16-1 10-7 16-16 16Zm0 0 9-9",
    camera: "M4 8h3l2-3h6l2 3h3v11H4z M12 17a4 4 0 1 0 0-8 4 4 0 0 0 0 8Z",
    image: "M4 5h16v14H4z M4 15l5-5 4 4 3-3 4 4 M15 9.5a1.5 1.5 0 1 0 0-.01",
    list: "M9 6h11M9 12h11M9 18h11M4 6h.01M4 12h.01M4 18h.01",
    report: "M7 3h7l5 5v13H7z M14 3v5h5 M10 13h6M10 17h6",
    check: "M5 12l5 5 9-10",
    alert: "M12 3l10 18H2zM12 10v4M12 17h.01",
    info: "M12 22a10 10 0 1 0 0-20 10 10 0 0 0 0 20ZM12 11v6M12 7h.01",
    mic: "M12 3a3 3 0 0 0-3 3v6a3 3 0 0 0 6 0V6a3 3 0 0 0-3-3ZM5 11a7 7 0 0 0 14 0M12 18v3",
    expert: "M12 12a4 4 0 1 0 0-8 4 4 0 0 0 0 8ZM4 21a8 8 0 0 1 16 0",
    shield: "M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6z",
    speaker: "M4 9h4l5-4v14l-5-4H4zM17 9a4 4 0 0 1 0 6",
  };
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("aria-hidden", "true");
  svg.setAttribute("class", "icon");
  const p = document.createElementNS("http://www.w3.org/2000/svg", "path");
  p.setAttribute("d", paths[name] || paths.info);
  svg.append(p);
  return svg;
};

const announce = (msg) => {
  const live = document.getElementById("live");
  if (live) {
    live.textContent = "";
    setTimeout(() => (live.textContent = msg), 30);
  }
};
const go = (route) => {
  if (location.hash !== "#/" + route) location.hash = "#/" + route;
  else render();
};
const set = (fn) => store.update(fn);
const btn = (label, onclick, cls = "btn", extra = {}) => h("button", { type: "button", class: cls, onclick, ...extra }, label);
const notice = (tone, ...kids) => h("div", { class: `notice notice-${tone}`, role: tone === "bad" ? "alert" : null },
  icon(tone === "ok" ? "check" : tone === "info" ? "info" : "alert"), h("div", {}, ...kids));

// ------------------------------------------------------------------------------------------------ languages
// English plus the 22 languages of the Eighth Schedule. Every non-English catalogue is a machine-made DRAFT (not checked
// by a native speaker) and says so on screen. Knowledge-base text (diseases, practices, crop facts) is looked up in the
// same catalogue's "kb" table and stays in English where no draft exists.
let LANGS = [{ code: "en", native: "English", english: "English", dir: "ltr" }];

function langSelect(id, cls) {
  const cur = store.get().uiLang;
  return h("select", { id, class: cls, "aria-label": t("lang.label"), onchange: (e) => setLanguage(e.target.value) },
    LANGS.map((l) => h("option", { value: l.code, selected: cur === l.code, lang: l.code, label: l.native }, l.native)));
}

async function setLanguage(code) {
  if (!LANGS.some((l) => l.code === code)) return;
  try {
    if (!K.catalogues[code]) K.catalogues[code] = await api.i18n(code);
  } catch (e) {
    announce(friendlyError(e, t));
    return;
  }
  I = createI18n(K.catalogues, code);
  const l = LANGS.find((x) => x.code === code);
  document.documentElement.lang = code;
  document.documentElement.dir = l.dir;
  speechSynthesisStop();
  set((x) => ({ ...x, uiLang: code }));
  announce(l.native);
}

// knowledge-base / server text: draft translation when the catalogue has one, otherwise the English source text
function tk(text) {
  if (!text) return text;
  const cat = K.catalogues[store.get().uiLang];
  return (cat && cat.kb && cat.kb[text]) || text;
}

function langDraftNotice() {
  const st = I.status();
  return st.lang !== "en" && !st.verified ? notice("warn", t("lang.draft_notice")) : null;
}

// ---- listening: local engines only. English/Hindi use the offline AgroVaidya voice; other languages use a voice
// installed on this computer (speechSynthesis voices with localService = true). Network ("online") voices are never used.
function localVoice(code) {
  const vs = (globalThis.speechSynthesis && speechSynthesis.getVoices()) || [];
  return vs.find((v) => v.localService && v.lang && v.lang.toLowerCase().split(/[-_]/)[0] === code) || null;
}

function speechSynthesisStop() {
  try { if (globalThis.speechSynthesis) speechSynthesis.cancel(); } catch (e) { /* not supported */ }
}

function canListen(code = store.get().uiLang) {
  return ((code === "en" || code === "hi") && ttsOk()) || !!localVoice(code);
}

async function listen(text) {
  const code = store.get().uiLang;
  if (!text) return;
  if ((code === "en" || code === "hi") && ttsOk()) return speak(text, code);
  const v = localVoice(code);
  if (!v) { announce(t("listen.unavailable")); return; }
  speechSynthesisStop();
  const u = new SpeechSynthesisUtterance(text);
  u.voice = v;
  u.lang = v.lang;
  speechSynthesis.speak(u);
}

function listenBtn(getText, id, small = false) {
  const ok = canListen();
  return btn([icon("speaker"), h("span", { class: small ? "visually-hidden-sm" : "" }, t("voice.listen"))],
    () => listen(typeof getText === "function" ? getText() : getText), "btn btn-ghost btn-listen" + (small ? " btn-small" : ""),
    { id, "aria-label": t("voice.listen") + (ok ? "" : " — " + t("listen.unavailable")), "aria-disabled": ok ? null : "true",
      title: ok ? null : t("listen.unavailable") });
}

// ------------------------------------------------------------------------------------------------ shell
function header() {
  const route = store.get().route;
  return h("header", { class: "topbar" },
    h("div", { class: "topbar-inner" },
      h("a", { class: "brand", href: "#/home" }, h("span", { class: "brand-mark" }, icon("leaf")),
        h("span", { class: "brand-text" }, h("strong", {}, t("app.name")), h("span", {}, t("app.tagline")))),
      h("nav", { "aria-label": "Main" },
        langSelect("ui-lang", "lang-select"),
        h("a", { href: "#/home", class: "navlink", "aria-current": route === "home" ? "page" : null }, t("nav.home")),
        h("a", { href: "#/help", class: "navlink", "aria-current": route === "help" ? "page" : null }, t("nav.help")))));
}

function stepper(active) {
  const i = S.STEPS.indexOf(active);
  return h("ol", { class: "stepper", "aria-label": t("steps.label") },
    S.STEPS.map((s, n) => h("li", { class: n < i ? "done" : n === i ? "current" : "", "aria-current": n === i ? "step" : null },
      h("span", { class: "dot", "aria-hidden": "true" }, n < i ? "✓" : String(n + 1)), h("span", { class: "lbl" }, t("steps." + s)))));
}

function page(title, lead, ...body) {
  return h("section", { class: "page" }, h("h1", { tabindex: "-1", id: "page-title" }, title), lead ? h("p", { class: "lead" }, lead) : null, ...body);
}

function navRow(back, next) {
  return h("div", { class: "nav-row" }, back || h("span"), next || h("span"));
}

// ------------------------------------------------------------------------------------------------ home
function viewHome() {
  const steps = [["camera", "home.step1"], ["leaf", "home.step2"], ["list", "home.step3"], ["report", "home.step4"]];
  const cur = store.get().uiLang;
  return h("section", { class: "page home" },
    h("section", { class: "card lang-pick", "aria-labelledby": "lang-title" },
      h("h2", { id: "lang-title" }, icon("speaker"), t("lang.choose_title")),
      h("p", { class: "muted small" }, t("lang.choose_lead")),
      h("div", { class: "lang-grid", role: "group", "aria-labelledby": "lang-title" }, LANGS.map((l) => h("button", {
        type: "button", class: "lang-btn", lang: l.code, dir: l.dir, "aria-pressed": String(cur === l.code), id: "lang-" + l.code,
        onclick: () => setLanguage(l.code) }, h("span", { class: "lang-native" }, l.native), h("span", { class: "lang-en" }, l.english))))),
    h("div", { class: "hero" },
      h("div", { class: "hero-text" },
        h("h1", { tabindex: "-1", id: "page-title" }, t("home.title")),
        h("p", { class: "lead" }, t("home.lead")),
        listenBtn(() => [t("home.title"), t("home.lead")].join(". "), "listen-home"),
        h("div", { class: "hero-actions" }, btn(t("home.start"), () => go("photo"), "btn btn-primary btn-lg", { id: "home-start" }),
          h("a", { href: "#/help", class: "btn btn-ghost btn-lg" }, t("home.howto"))),
        h("p", { class: "trust" }, icon("shield"), t("status.local"))),
      h("div", { class: "hero-art", "aria-hidden": "true" }, h("div", { class: "leaf-art" }, icon("leaf")))),
    h("ol", { class: "how" }, steps.map(([ic, k], n) => h("li", { class: "how-card" },
      h("span", { class: "how-icon" }, icon(ic)), h("span", { class: "how-n" }, String(n + 1)),
      h("h2", {}, t(k + ".title")), h("p", {}, t(k + ".body")),
      listenBtn(() => t(k + ".title") + ". " + t(k + ".body"), null, true)))),
    h("aside", { class: "card limits" }, h("h2", {}, t("home.limits.title")),
      h("ul", {}, ["1", "2", "3"].map((n) => h("li", {}, t("home.limits." + n))))));
}

// ------------------------------------------------------------------------------------------------ plant
function plantBadge(p) {
  return h("span", { class: "badge " + (p.disease_model_supported ? "badge-ok" : "badge-muted") },
    t(p.disease_model_supported ? "plant.badge.photo" : "plant.badge.guidance"));
}

function choiceSummary(c) {
  if (c.action === "dont_know") return { name: t("report.plant.unknown"), msg: t("plant.confirm.dont_know") };
  if (c.action === "unlisted") return { name: c.text, msg: t("plant.confirm.unlisted", { name: c.text }) };
  const p = c.plant || K.plants.find((x) => x.plant_id === c.plant_id);
  if (!p) return { name: c.text || c.plant_id || "", msg: "" };
  return { name: tk(p.display_name), msg: t(p.disease_model_supported ? "plant.confirm.photo_yes" : "plant.confirm.photo_no") };
}

function viewPlant() {
  const s = store.get();
  const p = s.plant;
  if (p.choice) {
    const cs = choiceSummary(p.choice);
    return page(t("plant.title"), null, stepper("plant"),
      h("div", { class: "card selected" }, h("p", { class: "eyebrow" }, t("plant.selected")),
        h("p", { class: "big" }, cs.name), h("p", { class: "muted" }, `${t("plant.user_given")} · ${t("plant.not_verified")}`),
        h("p", {}, cs.msg), btn(t("plant.confirm.change"), () => set(S.correctPlant), "btn btn-ghost")),
      submitState(),
      navRow(btn("← " + t("steps.photo"), () => go("photo"), "btn btn-ghost"),
        btn(store.get().submit.error ? t("errors.retry") : t("plant.check_now"), submit, "btn btn-primary btn-lg",
          { disabled: !S.readiness(store.get()).ready || store.get().submit.busy, id: "submit" })));
  }
  if (p.pending) {
    const cs = choiceSummary(p.pending);
    return page(t("plant.confirm.title"), null, stepper("plant"),
      h("div", { class: "card confirm", role: "group", "aria-labelledby": "confirm-name" },
        h("p", { class: "big", id: "confirm-name" }, cs.name),
        p.pending.action === "typed" ? h("p", { class: "muted" }, t("plant.confirm.you_typed", { text: p.pending.text })) : null,
        p.pending.plant ? plantBadge(p.pending.plant) : null,
        h("p", {}, cs.msg),
        listenBtn(() => `${cs.name}. ${cs.msg}`, "plant-listen"),
        h("div", { class: "btn-row" },
          btn(t("plant.confirm.yes"), () => { set(S.confirmPlant); announce(cs.name); if (S.readiness(store.get()).ready) submit(); }, "btn btn-primary", { id: "confirm-yes" }),
          btn(t("plant.confirm.change"), () => set(S.correctPlant), "btn btn-ghost"))));
  }
  const res = p.resolution;
  const input = h("input", { id: "plant-input", type: "text", autocomplete: "off", maxlength: "60", value: p.query,
    "aria-describedby": "plant-hint" + (res && res.status !== "resolved" ? " plant-msg" : ""),
    oninput: (e) => store.update((x) => ({ ...x, plant: { ...x.plant, query: e.target.value, channel: "typed" } }), true) });
  input.value = p.query;
  const find = async (ev) => {
    ev.preventDefault();
    const text = store.get().plant.query;
    try {
      const r = await api.resolve(text);
      if (r.status === "resolved") set((x) => S.setPending({ ...x, plant: { ...x.plant, resolution: r } },
        { action: "typed", text: r.entered_text, plant: r.plant, channel: x.plant.channel || "typed" }));
      else { set((x) => ({ ...x, plant: { ...x.plant, resolution: r } })); announce(r.message); }
    } catch (e) {
      set((x) => ({ ...x, plant: { ...x.plant, resolution: { status: "error", message: friendlyError(e, t), suggestions: [] } } }));
    }
  };
  const msg = res && res.status !== "resolved" ? h("div", { id: "plant-msg", class: "card result", role: "status" },
    h("p", {}, res.status === "empty" ? t("plant.empty") : res.message),
    res.suggestions && res.suggestions.length ? h("div", {},
      h("p", { class: "eyebrow" }, t(res.status === "ambiguous" ? "plant.ambiguous.title" : "plant.suggest.title")),
      h("div", { class: "chips" }, res.suggestions.map((o) => btn([tk(o.display_name), " ", plantBadge(o)],
        () => set((x) => S.setPending(x, { action: "selected", plant_id: o.plant_id, plant: o })), "chip")))) : null,
    ["unknown", "needs_clarification"].includes(res.status) ?
      btn(t("plant.use_anyway", { text: res.entered_text }), () => set((x) => S.setPending(x, { action: "unlisted", text: res.entered_text })), "btn btn-ghost") : null) : null;
  return page(t("plant.title"), t("plant.lead"), stepper("plant"),
    listenBtn(() => t("plant.title") + ". " + t("plant.lead"), "listen-plant"),
    !store.get().photo.file ? notice("warn", t("review.missing_photo"), btn(t("steps.photo"), () => go("photo"), "btn btn-link")) : null,
    h("form", { class: "card form", onsubmit: find, novalidate: true },
      h("label", { for: "plant-input", class: "field-label" }, t("plant.input.label")),
      h("div", { class: "input-row" }, input, h("button", { type: "submit", class: "btn btn-primary" }, t("plant.find"))),
      h("p", { id: "plant-hint", class: "hint" }, t("plant.input.hint"))),
    voiceControls("plant", t("voice.say_plant")),
    msg,
    h("h2", { class: "section-title" }, t("plant.or_choose")),
    h("div", { class: "plant-grid" }, K.plants.map((o) => h("button", { type: "button", class: "plant-card",
      onclick: () => set((x) => S.setPending(x, { action: "selected", plant_id: o.plant_id, plant: o })) },
      h("span", { class: "plant-name" }, tk(o.display_name)), h("span", { class: "plant-sci" }, o.scientific_name.split(" ").slice(0, 2).join(" ")),
      plantBadge(o)))),
    h("div", { class: "center" }, btn(t("plant.dont_know"), () => set((x) => S.setPending(x, { action: "dont_know" })), "btn btn-ghost")));
}

// ------------------------------------------------------------------------------------------------ photo
async function acceptFile(file, source) {
  closeCameraIfOpen();
  const pc = precheckFile(file, K.maxMb);
  if (!pc.ok) {
    set((x) => ({ ...x, photo: { ...x.photo, error: t(pc.key, pc.params) } }));
    announce(t(pc.key, pc.params));
    return;
  }
  const old = store.get().photo.url;
  if (old) URL.revokeObjectURL(old);
  const url = URL.createObjectURL(file);
  const dims = await new Promise((resolve) => {             // displayed size (EXIF orientation applied by the browser)
    const im = new Image();
    im.onload = () => resolve([im.naturalWidth, im.naturalHeight]);
    im.onerror = () => resolve([0, 0]);
    im.src = url;
  });
  markView = { showMarks: true, drag: null };
  set((x) => ({ ...x, result: null, photo: { file, url, name: file.name, check: null, checking: true, error: null, source,
    marks: [], natW: dims[0], natH: dims[1] } }));
  try {
    const chk = await api.checkImage(file);
    set((x) => ({ ...x, photo: { ...x.photo, check: chk, checking: false } }));
    announce(chk.passed ? t("photo.ok") : t("photo.problems") + " " + chk.messages.map(tk).join("; "));
  } catch (e) {
    const rr = e instanceof ApiFailure && e.details && e.details.retake_requests;
    set((x) => ({ ...x, photo: { ...x.photo, checking: false, file: rr ? null : x.photo.file, url: rr ? null : x.photo.url,
      check: rr ? { passed: false, messages: [], retake_requests: rr } : null, error: friendlyError(e, t) } }));
  }
}

function removePhoto() {
  closeCameraIfOpen();
  const old = store.get().photo.url;
  if (old) URL.revokeObjectURL(old);
  set((x) => ({ ...x, result: null, photo: S.initialState().photo }));
}

function openCamera() {
  focusNext = "camera-cancel";
  return camSession.open(document.getElementById("camera-video"));
}

async function captureStill() {
  const f = await camSession.capture(document.getElementById("camera-video"));
  if (f) { focusNext = "camera-use"; render(); }
}

function useStill() {
  focusNext = "take-photo";
  const f = camSession.use();
  if (f) acceptFile(f, "camera");
}

function retakeStill() {
  focusNext = "camera-capture";
  camSession.retake(document.getElementById("camera-video"));
}

function cancelCamera() {
  focusNext = "take-photo";
  camSession.close();
}

function cameraCard() {
  const pickFile = (id) => () => { camSession.close(); document.getElementById(id).click(); };
  const st = camSession.state;
  const kids = [];
  if (st.name === "starting" || st.name === "live") {
    const video = h("video", { id: "camera-video", playsinline: true, muted: true, autoplay: true, "aria-label": t("photo.camera.title") });
    camSession.attach(video);                       // re-attach the running stream after every re-render
    kids.push(video, h("p", { role: "status", class: "muted small" }, st.name === "starting" ? t("photo.camera.starting") : t("photo.camera.live_hint")));
  } else if (st.name === "captured" && stillUrl) {
    kids.push(h("img", { src: stillUrl, alt: t("photo.camera.captured_alt"), class: "preview-img", id: "camera-still" }),
      h("p", { role: "status", class: "muted small" }, t("photo.camera.captured_hint")));
  } else if (st.name === "error") {
    kids.push(notice("bad", t(st.error)));
  }
  const row = [];
  if (st.name === "live") row.push(btn([icon("camera"), t("photo.camera.capture")], captureStill, "btn btn-primary", { id: "camera-capture" }));
  if (st.name === "captured") row.push(btn([icon("check"), t("photo.camera.use")], useStill, "btn btn-primary", { id: "camera-use" }),
    btn(t("photo.camera.retake"), retakeStill, "btn", { id: "camera-retake" }));
  if (st.name === "error") {
    row.push(btn([icon("image"), t("photo.choose")], pickFile("photo-file"), "btn btn-primary", { id: "camera-upload" }));
    // Browsers without the live-camera API can still hand over a photo from the device's own camera app.
    if (st.error === "photo.camera.unsupported") row.push(btn([icon("camera"), t("photo.camera.device_app")], pickFile("photo-capture"), "btn", { id: "camera-device-app" }));
    else if (st.error !== "photo.camera.insecure") row.push(btn(t("photo.camera.try_again"), openCamera, "btn", { id: "camera-retry" }));
  }
  row.push(btn(t("photo.camera.cancel"), cancelCamera, "btn btn-ghost", { id: "camera-cancel" }));
  return h("section", { class: "card camera", "aria-label": t("photo.camera.title"), id: "camera-card" },
    h("h2", { class: "camera-title" }, icon("camera"), t("photo.camera.title")), kids, h("div", { class: "btn-row camera-controls" }, row));
}

function viewPhoto() {
  const s = store.get();
  const ph = s.photo;
  const fileInput = h("input", { type: "file", id: "photo-file", accept: "image/jpeg,image/png", class: "visually-hidden",
    tabindex: "-1", "aria-hidden": "true", onchange: (e) => { const f = e.target.files[0]; e.target.value = ""; if (f) acceptFile(f, "file"); } });
  const captureInput = h("input", { type: "file", id: "photo-capture", accept: "image/jpeg,image/png", capture: "environment",
    class: "visually-hidden", tabindex: "-1", "aria-hidden": "true",
    onchange: (e) => { const f = e.target.files[0]; e.target.value = ""; if (f) acceptFile(f, "camera"); } });
  const choose = () => fileInput.click();
  const take = openCamera;          // errors (denied, no camera, unsupported, insecure) are shown in the camera card
  const cam = cameraOpen() ? cameraCard() : null;
  const chk = ph.check;
  return page(t("photo.title"), t("photo.lead"), stepper("photo"), listenBtn(() => t("photo.title") + ". " + t("photo.lead"), "listen-photo"),
    fileInput, captureInput,
    cam || (ph.url ? h("div", { class: "card preview" },
      h("img", { src: ph.url, alt: t("photo.preview_alt"), class: "preview-img" }),
      h("div", { class: "preview-side" },
        ph.checking ? h("p", { role: "status", class: "checking" }, h("span", { class: "spinner", "aria-hidden": "true" }), t("photo.checking")) : null,
        chk && chk.passed ? notice("ok", t("photo.ok")) : null,
        chk && !chk.passed ? notice("warn", h("p", {}, t("photo.problems")),
          h("ul", {}, chk.messages.map((m) => h("li", {}, tk(m)))),
          h("ul", { class: "retake" }, chk.retake_requests.map((r) => h("li", {}, h("strong", {}, t("photo.retake_view", { view: tk(r.view) })), " — ", tk(r.why))))) : null,
        h("p", { class: "muted small" }, icon("shield"), t("photo.not_saved")),
        h("div", { class: "btn-row photo-swap" },
          btn([icon("camera"), t("photo.take_new")], take, "btn", { id: "take-photo" }),
          btn([icon("image"), t("photo.replace")], choose, "btn", { id: "choose-photo" }),
          btn(t("photo.remove"), removePhoto, "btn btn-ghost", { id: "remove-photo" })))) :
      h("div", { class: "photo-actions" },
        btn([icon("camera"), h("span", {}, t("photo.take"))], take, "big-action", { id: "take-photo" }),
        btn([icon("image"), h("span", {}, t("photo.choose"))], choose, "big-action", { id: "choose-photo" }))),
    ph.url && !cam && ph.natW ? markPanel(ph) : null,
    ph.error ? notice("bad", ph.error, chk && chk.retake_requests && !ph.url ?
      h("ul", {}, chk.retake_requests.map((r) => h("li", {}, t("photo.retake_view", { view: tk(r.view) })))) : null) : null,
    navRow(btn("← " + t("nav.home"), () => go("home"), "btn btn-ghost"),
      btn((chk && !chk.passed ? t("photo.continue_anyway") : t("steps.plant")) + " →", () => go("plant"), "btn btn-primary",
        { disabled: !ph.file || ph.checking || !!cam, id: "photo-next" })));
}

// ------------------------------------------------------------------------------------------------ voice
// One voice interaction at a time. vflow: {name, target, transcript, message, heard, option, seconds}. Recording only
// after a press; consent is asked once per tab; every result needs the user's confirmation (voiceStep contract).
let vflow = null;
let recorder = null;
let vtimer = null;
let readback = null;
const voiceOk = () => !!(K.speech && K.speech.speech_to_text.available);
const ttsOk = () => !!(K.speech && K.speech.text_to_speech.available);
const vlang = () => store.get().voiceLang || "en";
const vctx = () => ({ available: voiceOk(), consentAccepted: !!store.get().voiceConsent });

function vstep(event) {
  const next = voiceStep(vflow, event, vctx());
  vflow = { ...next, target: (vflow && vflow.target) || event.target };
  if (next.message) announce(t(next.message));
  render();
  return vflow;
}

async function speak(text, lang) {
  if (!ttsOk() || !text) return;
  try {
    if (readback) { readback.pause(); URL.revokeObjectURL(readback.src); }
    const blob = await api.synthesize(text, lang);
    readback = new Audio(URL.createObjectURL(blob));
    readback.onended = () => { URL.revokeObjectURL(readback.src); readback = null; };
    await readback.play();
  } catch (e) {
    announce(friendlyError(e, t));
  }
}

let warmed = {};
function warmUp() {
  const l = vlang();
  if (!warmed[l] && voiceOk()) { warmed[l] = true; api.speechWarmup(l).catch(() => { warmed[l] = false; }); }
}

async function voicePress(target) {
  if (store.get().voiceConsent) warmUp();
  vflow = { name: voiceOk() ? "idle" : "unavailable", target };
  const s = vstep({ type: "press", target });
  if (s.name === "requesting") return beginRecording();
}

async function beginRecording() {
  try {
    recorder = await startRecording();
    vstep({ type: "granted" });
    vflow.seconds = 0;
    vtimer = setInterval(() => {
      const sec = recorder ? Math.floor(recorder.seconds()) : 0;
      const el = document.getElementById("voice-timer");
      if (el) el.textContent = t("voice.listening_time", { s: sec });
      if (sec >= VMAX) stopRecording();
    }, 250);
  } catch (e) {
    recorder = null;
    vstep({ type: e && e.name === "NotFoundError" ? "no_device" : "denied" });
  }
}

async function stopRecording() {
  if (!recorder) return;
  clearInterval(vtimer);
  const r = recorder;
  recorder = null;
  vstep({ type: "stop" });
  const purpose = vflow.target === "plant" ? "plant_name" : "free";
  try {
    const wav = await r.stop();
    const res = await api.transcribe(wav, vlang(), purpose);
    vstep({ type: "result", result: res });
  } catch (e) {
    const sp = e instanceof ApiFailure && e.details && e.details.speech;
    vstep({ type: "result", result: sp || { status: "service_error" } });
  }
}

async function cancelVoice() {
  clearInterval(vtimer);
  if (recorder) { await recorder.cancel(); recorder = null; }
  vstep({ type: "cancel" });
  vflow = null;
  render();
}

function deliverVoice(text, channel) {
  const target = vflow.target;
  vflow = null;
  if (target === "plant") {
    store.update((x) => ({ ...x, plant: { ...x.plant, query: text, channel } }), true);
    render();
    const form = document.querySelector("form.card.form");
    if (form) form.requestSubmit();
  } else if (target === "symptoms") {
    set((x) => ({ ...x, symptoms: { ...x.symptoms, mode: "text", text, voiceChannel: channel } }));
    setTimeout(() => { const b = document.getElementById("symptom-find"); if (b) b.click(); }, 0);
  } else if (target && target.startsWith("q:")) {
    const q = K.questions.find((x) => x.question_id === target.slice(2));
    const opt = q && matchOption(text, q.options);
    vflow = { name: opt ? "option" : "no_option", target, transcript: text, option: opt };
    render();
  }
}

function voiceControls(target, label, compact = false) {
  const active = vflow && vflow.target === target;
  if (compact && (!active || ["idle", "unavailable", "confirmed"].includes(vflow.name)))
    return btn([icon("mic"), label], () => voicePress(target), "btn btn-ghost btn-small", { disabled: !voiceOk(),
      "aria-label": `${label}: ${(K.questions.find((x) => "q:" + x.question_id === target) || {}).text || ""}` });
  const langSel = h("select", { class: "voice-lang", "aria-label": t("voice.language"), disabled: !voiceOk(),
    onchange: (e) => set((x) => ({ ...x, voiceLang: e.target.value })) },
    h("option", { value: "en", selected: vlang() === "en" }, "English"),
    h("option", { value: "hi", selected: vlang() === "hi" }, t("voice.lang_hi")));
  const micBtn = btn([icon("mic"), h("span", {}, label || t("voice.speak"))], () => voicePress(target), "btn btn-voice",
    { disabled: !voiceOk(), id: "mic-" + target.replace(":", "-") });
  if (!active || ["idle", "unavailable", "confirmed"].includes(vflow.name)) {
    return h("div", { class: "voice-row" }, micBtn, langSel,
      !voiceOk() ? h("span", { class: "muted small" }, t("speech.unavailable")) : null,
      voiceOk() ? h("span", { class: "muted small" }, t("voice.status_" + vlang())) : null);
  }
  const panel = (tone, ...kids) => h("div", { class: `card voice-panel voice-${tone}`, role: "group", "aria-label": t("voice.title") }, ...kids);
  const cancel = btn(t("voice.cancel"), cancelVoice, "btn btn-ghost", { id: "voice-cancel" });
  switch (vflow.name) {
    case "consent":
      return panel("info", h("h3", {}, icon("shield"), t("voice.consent.title")),
        h("ul", {}, ["1", "2", "3", "4"].map((n) => h("li", {}, t("voice.consent." + n)))),
        h("div", { class: "btn-row" }, btn(t("voice.consent.accept"), () => { set((x) => ({ ...x, voiceConsent: true })); warmUp(); vstep({ type: "consent_accept" }); beginRecording(); }, "btn btn-primary", { id: "voice-consent-accept" }),
          btn(t("voice.consent.decline"), () => { vstep({ type: "consent_decline" }); vflow = null; render(); }, "btn btn-ghost")));
    case "requesting":
      return panel("info", h("p", { role: "status" }, t("voice.requesting")), cancel);
    case "listening":
      return panel("live", h("p", { class: "listening", role: "status" }, h("span", { class: "rec-dot", "aria-hidden": "true" }), h("strong", {}, t("voice.listening")),
        " ", h("span", { id: "voice-timer" }, t("voice.listening_time", { s: 0 }))), h("p", { class: "muted small" }, t("voice.listening_hint")),
        h("div", { class: "btn-row" }, btn(t("voice.stop"), stopRecording, "btn btn-primary", { id: "voice-stop" }), cancel));
    case "processing":
      return panel("info", h("p", { role: "status", class: "checking" }, h("span", { class: "spinner", "aria-hidden": "true" }), t("voice.processing")), cancel);
    case "denied":
      return panel("bad", h("p", { role: "alert" }, t(vflow.message || "speech.denied")), btn(t("voice.type_instead"), cancelVoice, "btn btn-primary"));
    case "retry":
      return panel("warn", h("p", { role: "alert" }, t(vflow.message || "speech.service_error")),
        vflow.heard ? h("p", { class: "muted" }, t("voice.heard_unsure", { text: vflow.heard })) : null,
        h("div", { class: "btn-row" }, btn(t("voice.again"), () => { vflow = { name: "idle", target: vflow.target }; voicePress(vflow.target); }, "btn btn-primary", { id: "voice-again" }),
          btn(t("voice.type_instead"), cancelVoice, "btn btn-ghost")));
    case "confirm": {
      const input = h("input", { type: "text", id: "voice-edit", value: vflow.transcript, maxlength: "200", "aria-label": t("voice.edit_label") });
      input.value = vflow.transcript;
      return panel("ok", h("p", { class: "eyebrow" }, t(vflow.typed ? "questions.type_label" : "voice.we_heard")), input,
        h("p", { class: "muted small" }, t("voice.confirm_hint")),
        h("div", { class: "btn-row" },
          btn(t("voice.use"), () => {
            const edited = document.getElementById("voice-edit").value.trim();
            if (vflow.typed) { if (edited) deliverVoice(edited, "typed_text_confirmed"); return; }
            const ev = edited && edited !== vflow.transcript ? { type: "correct", text: edited } : { type: "confirm" };
            const done = voiceStep(vflow, ev, vctx());
            vflow = { ...done, target: vflow.target };
            deliverVoice(done.transcript, done.channel);
          }, "btn btn-primary", { id: "voice-use" }),
          ttsOk() && !vflow.typed ? btn([icon("speaker"), t("voice.read_back")], () => speak(document.getElementById("voice-edit").value, vlang()), "btn btn-ghost", { id: "voice-readback" }) : null,
          !vflow.typed ? btn(t("voice.again"), () => { vflow = { name: "idle", target: vflow.target }; voicePress(vflow.target); }, "btn btn-ghost") : null,
          vflow.typed ? btn(t("voice.cancel"), () => { vflow = null; render(); }, "btn btn-ghost") : cancel));
    }
    case "option":
      return panel("ok", h("p", {}, t("voice.option_q", { text: vflow.transcript, option: vflow.option.label })),
        h("div", { class: "btn-row" }, btn(t("voice.option_yes", { option: vflow.option.label }), () => {
          const qid = vflow.target.slice(2), oid = vflow.option.option_id;
          vflow = null;
          set((x) => S.setAnswer(x, qid, oid));
          announce(t("voice.answer_set"));
        }, "btn btn-primary", { id: "voice-option-yes" }), btn(t("voice.option_no"), cancelVoice, "btn btn-ghost")));
    case "no_option":
      return panel("warn", h("p", { role: "alert" }, t("voice.no_option", { text: vflow.transcript })), btn(t("voice.ok"), cancelVoice, "btn btn-primary"));
    default:
      return micBtn;
  }
}

// ------------------------------------------------------------------------------------------------ report player
const player = createPlayer((seg, speed) => api.synthesize(seg.text, seg.lang || "en", speed));
let playerFor = null;
player.onChange(() => { const el = document.getElementById("player-status"); if (el) render(); });

function reportPlayer(result) {
  const segs = (result.spoken_script && result.spoken_script.segments) || [];
  if (!ttsOk() || !result.spoken_script.available)
    return h("button", { type: "button", class: "btn btn-ghost btn-block", disabled: true, "aria-label": t("report.listen.pending") }, icon("speaker"), t("report.listen.pending"));
  if (playerFor !== result.report.report_id) { player.load(segs); playerFor = result.report.report_id; }
  const st = player.state;
  const cur = segs[st.index] || segs[0];
  const label = { stopped: t("player.stopped"), loading: t("player.loading"), playing: t("player.playing"), paused: t("player.paused"), error: t("player.error") }[st.state];
  return h("section", { class: "card player", "aria-labelledby": "player-title" },
    h("h2", { id: "player-title" }, icon("speaker"), t("report.listen")),
    h("p", { id: "player-status", role: "status", class: "small" }, `${label} · ${t("player.part", { n: st.index + 1, total: st.total })}`),
    cur ? h("p", { class: "muted small" }, t("player.kind." + cur.kind)) : null,
    h("div", { class: "btn-row" },
      st.state === "playing" ? btn(t("player.pause"), () => player.pause(), "btn btn-primary", { id: "player-pause" }) :
        btn(t("player.play"), () => player.play(), "btn btn-primary", { id: "player-play" }),
      btn(t("player.repeat"), () => player.repeat(), "btn btn-ghost", { id: "player-repeat" }),
      btn(t("player.next"), () => player.next(), "btn btn-ghost", { id: "player-next" }),
      btn(t("player.stop"), () => player.stop(), "btn btn-ghost", { id: "player-stop" })),
    h("label", { class: "field-label small", for: "player-speed" }, t("player.speed")),
    h("select", { id: "player-speed", onchange: (e) => player.setSpeed(parseFloat(e.target.value)) },
      [["0.8", "player.slower"], ["1", "player.normal"], ["1.2", "player.faster"]].map(([v, k]) => h("option", { value: v, selected: String(st.speed) === v }, t(k)))),
    result.report.language === "hi" ? h("p", { class: "muted small" }, t("voice.hi_draft")) : null);
}

// ------------------------------------------------------------------------------------------------ farmer marking
let markView = { showMarks: true, drag: null };
const SVGNS = "http://www.w3.org/2000/svg";

function overlaySvg(marks, natW, natH, preview) {
  const svg = document.createElementNS(SVGNS, "svg");
  svg.setAttribute("viewBox", `0 0 ${natW} ${natH}`);
  svg.setAttribute("class", "mark-overlay");
  svg.setAttribute("aria-hidden", "true");
  const shapes = preview ? [...marks, preview] : marks;
  shapes.forEach((m, i) => {
    const p = M.toPixels(m, natW, natH);
    const el = document.createElementNS(SVGNS, m.shape === "circle" ? "circle" : "rect");
    for (const [k, v] of Object.entries(p)) el.setAttribute(k, String(v));
    el.setAttribute("class", m === preview ? "mark-shape preview" : "mark-shape");
    el.setAttribute("vector-effect", "non-scaling-stroke");
    svg.append(el);
    if (m !== preview) {
      const tx = document.createElementNS(SVGNS, "text");
      const c = m.shape === "circle" ? [p.cx, p.cy - p.r] : [p.x, p.y];
      tx.setAttribute("x", String(c[0] + 4)); tx.setAttribute("y", String(c[1] + Math.max(natW, natH) * 0.035));
      tx.setAttribute("font-size", String(Math.max(natW, natH) * 0.035));
      tx.setAttribute("class", "mark-num");
      tx.textContent = String(i + 1);
      svg.append(tx);
    }
  });
  return svg;
}

function markStage(ph, interactive) {
  const natW = ph.natW || 4, natH = ph.natH || 3;
  const stage = h("div", { class: "mark-stage", style: `aspect-ratio: ${natW} / ${natH}` },
    h("img", { src: ph.url, alt: interactive ? t("mark.canvas_label") : t("photo.preview_alt"), draggable: "false" }));
  if (!interactive || markView.showMarks) stage.append(overlaySvg(ph.marks || [], natW, natH, interactive ? markView.drag && markView.drag.preview : null));
  if (!interactive) return stage;
  const at = (e) => M.toNormalized(e.clientX, e.clientY, stage.getBoundingClientRect(), natW, natH);
  stage.addEventListener("pointerdown", (e) => {
    const p = at(e);
    if (!p || store.get().photo.marks.length >= M.MAX_MARKS) return;
    try { stage.setPointerCapture(e.pointerId); } catch (err) { /* capture is optional (e.g. synthetic or ended pointers) */ }
    markView = { ...markView, showMarks: true, drag: { start: p, end: null, preview: null } };
    e.preventDefault();
  });
  stage.addEventListener("pointermove", (e) => {
    if (!markView.drag) return;
    const p = at(e);
    if (!p) return;
    markView.drag.end = p;
    markView.drag.preview = M.markFromGesture(markView.drag.start, p);
    const old = stage.querySelector("svg");
    const nu = overlaySvg(store.get().photo.marks, natW, natH, markView.drag.preview);
    old ? old.replaceWith(nu) : stage.append(nu);
  });
  const finish = (e) => {
    if (!markView.drag) return;
    const m = M.markFromGesture(markView.drag.start, at(e) || markView.drag.end);
    markView = { ...markView, drag: null };
    set((x) => ({ ...x, photo: { ...x.photo, marks: M.addMark(x.photo.marks, m) } }));
    announce(t("mark.area", { n: store.get().photo.marks.length, kind: t("mark.kind." + m.shape) }));
  };
  stage.addEventListener("pointerup", finish);
  stage.addEventListener("pointercancel", () => { markView = { ...markView, drag: null }; render(); });
  return stage;
}

function markPanel(ph) {
  const marks = ph.marks || [];
  const setMarks = (fn) => set((x) => ({ ...x, photo: { ...x.photo, marks: fn(x.photo.marks) } }));
  return h("section", { class: "card marking", "aria-labelledby": "mark-title" },
    h("h2", { id: "mark-title" }, t("mark.title")),
    h("p", { class: "muted" }, t("mark.lead")),
    h("div", { class: "seg-toggle", role: "group", "aria-label": t("mark.title") },
      btn(t("mark.show_marks"), () => { markView.showMarks = true; render(); }, "mode", { "aria-pressed": String(markView.showMarks), id: "mark-show" }),
      btn(t("mark.show_original"), () => { markView.showMarks = false; render(); }, "mode", { "aria-pressed": String(!markView.showMarks), id: "mark-original" })),
    markStage(ph, true),
    h("p", { class: "mark-label" }, icon("info"), t("mark.label")),
    h("ul", { class: "mark-list", "aria-live": "polite" }, marks.length ? marks.map((m, i) => h("li", {},
      h("span", {}, t("mark.area", { n: i + 1, kind: t("mark.kind." + m.shape) }) + (m.via === "grid" ? ` (${t("mark.via.grid")})` : "")),
      btn(t("mark.remove", { n: i + 1 }), () => setMarks((ms) => M.removeMark(ms, i)), "btn btn-small btn-ghost"))) :
      h("li", { class: "muted" }, t("mark.none"))),
    marks.length >= M.MAX_MARKS ? h("p", { class: "muted small" }, t("mark.max", { n: M.MAX_MARKS })) : null,
    h("div", { class: "btn-row" },
      btn(t("mark.undo"), () => setMarks(M.undoMark), "btn btn-ghost", { disabled: !marks.length, id: "mark-undo" }),
      btn(t("mark.clear"), () => setMarks(() => []), "btn btn-ghost", { disabled: !marks.length, id: "mark-clear" })),
    h("fieldset", { class: "grid-pick" }, h("legend", {}, t("mark.grid.title")),
      h("div", { class: "grid3" }, M.GRID.map((g, i) => btn(t("mark.grid." + g), () => setMarks((ms) => M.addMark(ms, M.gridMark(i))),
        "grid-cell", { id: "grid-" + g, disabled: marks.length >= M.MAX_MARKS })))),
    h("p", { class: "muted small" }, t("mark.alt")));
}

// ------------------------------------------------------------------------------------------------ symptoms
const CAT_ORDER = ["lesion", "colour_pattern", "sign", "plant_part", "distribution", "damage", "environment"];

function featureRow(f, s) {
  const v = s.symptoms.features[f.feature_id];
  const name = "f-" + f.feature_id;
  const opt = (val, label) => h("label", { class: "seg" },
    h("input", { type: "radio", name, value: label, checked: v === val || (val === null && v === undefined),
      onchange: () => set((x) => S.setFeature(x, f.feature_id, val)) }), h("span", {}, label === "clear" ? t("symptoms.clear") : t(label)));
  return h("fieldset", { class: "feature" }, h("legend", {}, tk(f.label)),
    h("div", { class: "seg-group" }, opt(true, "symptoms.seen"), opt(false, "symptoms.not_seen"), opt(null, "clear")));
}

function viewSymptoms() {
  const s = store.get();
  const sy = s.symptoms;
  const modeBtn = (m, label) => h("button", { type: "button", class: "mode", "aria-pressed": String(sy.mode === m), id: "mode-" + m,
    disabled: m === "voice" && !voiceOk(), onclick: () => set((x) => S.setMode(x, m, voiceOk())) },
    m === "voice" ? [icon("mic"), label, !voiceOk() ? h("small", {}, t("symptoms.mode.voice_pending")) : null] : label);
  let body;
  if (sy.mode === "text") {
    const ta = h("textarea", { id: "symptom-text", rows: "4", maxlength: "500", "aria-describedby": "symptom-text-hint",
      oninput: (e) => store.update((x) => ({ ...x, symptoms: { ...x.symptoms, text: e.target.value, voiceChannel: null } }), true) });
    ta.value = sy.text;
    const sg = sy.suggest;
    body = h("div", { class: "card" },
      sy.voiceChannel === "voice_transcript_confirmed" ? h("p", { class: "muted small" }, icon("mic"), t("voice.from_voice")) : null,
      h("label", { for: "symptom-text", class: "field-label" }, t("symptoms.text.label")),
      ta, h("p", { class: "hint", id: "symptom-text-hint" }, t("symptoms.text.hint")),
      btn(t("symptoms.text.find"), async () => {
        try {
          const r = await api.suggest(store.get().symptoms.text);
          set((x) => ({ ...x, symptoms: { ...x.symptoms, suggest: r } }));
          announce(r.suggestions.length ? t("symptoms.text.suggestions") : t("symptoms.text.none"));
        } catch (e) {
          set((x) => ({ ...x, symptoms: { ...x.symptoms, suggest: { error: friendlyError(e, t) } } }));
        }
      }, "btn btn-primary", { id: "symptom-find" }),
      sg && sg.error ? notice("bad", sg.error) : null,
      sg && !sg.error ? h("div", { class: "suggest", role: "status" },
        sg.suggestions.length ? h("p", { class: "eyebrow" }, t("symptoms.text.suggestions")) : h("p", {}, t("symptoms.text.none")),
        h("ul", { class: "suggest-list" }, sg.suggestions.map((x) => {
          const added = s.symptoms.features[x.feature_id] === true;
          return h("li", {}, h("span", {}, tk(x.label)),
            btn(added ? "✓ " + t("symptoms.text.added") : t("symptoms.text.add"),
              () => set((st) => S.setFeature(st, x.feature_id, added ? null : true, st.symptoms.voiceChannel || "typed_text_confirmed")),
              added ? "btn btn-small btn-done" : "btn btn-small", { "aria-pressed": String(added), "aria-label": `${t("symptoms.text.add")}: ${tk(x.label)}` }));
        })),
        sg.not_understood.length ? h("p", { class: "muted" }, t("symptoms.text.not_understood", { words: sg.not_understood.join(", ") })) : null) : null);
  } else if (sy.mode === "voice") {
    body = h("div", { class: "card" }, h("p", {}, t("voice.describe_lead")), voiceControls("symptoms", t("voice.describe")));
  } else {
    body = h("div", { class: "cats" }, CAT_ORDER.map((c) => {
      const fs = K.features.filter((f) => f.category === c);
      if (!fs.length) return null;
      const n = fs.filter((f) => f.feature_id in sy.features).length;
      return h("details", { class: "card cat", open: c === "lesion" || n > 0 ? true : null },
        h("summary", {}, h("span", {}, t("symptoms.cat." + c)), n ? h("span", { class: "badge badge-ok" }, String(n)) : null),
        fs.map((f) => featureRow(f, s)));
    }));
  }
  const conf = S.conflicts(s, K.questions);
  const nNoted = Object.keys(sy.features).length + Object.keys(sy.answers).length;
  return page(t("symptoms.title"), t("symptoms.lead"), listenBtn(() => t("symptoms.title") + ". " + t("symptoms.lead"), "listen-symptoms"),
    h("div", { class: "modes", role: "group", "aria-label": t("symptoms.mode.label") },
      modeBtn("tap", t("symptoms.mode.tap")), modeBtn("text", t("symptoms.mode.text")), modeBtn("voice", t("symptoms.mode.voice"))),
    body,
    conf.map((c) => notice("warn", t("symptoms.conflict", { label: (K.features.find((f) => f.feature_id === c.feature_id) || {}).label }))),
    h("details", { class: "card questions" },
      h("summary", {}, t("symptoms.questions.title")),
      h("p", { class: "muted" }, t("symptoms.questions.lead")),
      K.questions.map((q) => h("fieldset", { class: "question" }, h("legend", {}, tk(q.text)),
        q.illustration ? h("figure", { class: "q-figure" }, h("img", { src: q.illustration.src, alt: q.illustration.alt, loading: "lazy" }),
          h("figcaption", {}, t("questions.diagram_caption"))) : null,
        h("div", { class: "q-voice" },
          listenBtn(() => `${tk(q.text)} ${q.options.map((o) => tk(o.label)).join(". ")}.`, "listen-q-" + q.question_id, true),
          voiceOk() ? voiceControls("q:" + q.question_id, t("voice.answer"), true) : null,
          !(vflow && vflow.target === "q:" + q.question_id) ? btn(t("questions.type_answer"), () => { vflow = { name: "confirm", target: "q:" + q.question_id, transcript: "", typed: true }; render(); },
            "btn btn-ghost btn-small", { "aria-label": `${t("questions.type_answer")}: ${tk(q.text)}` }) : null),
        h("div", { class: "seg-group wrap" }, q.options.map((o) => h("label", { class: "seg" },
          h("input", { type: "radio", name: "q-" + q.question_id, checked: sy.answers[q.question_id] === o.option_id,
            onchange: () => set((x) => S.setAnswer(x, q.question_id, o.option_id)) }), h("span", {}, tk(o.label)))))))),
    h("div", { class: "card form two" },
      h("div", {}, h("label", { for: "region", class: "field-label" }, t("symptoms.region.label")),
        h("select", { id: "region", "aria-describedby": "region-hint", onchange: (e) => set((x) => ({ ...x, region: e.target.value })) },
          h("option", { value: "", selected: !s.region }, t("symptoms.region.none")),
          K.states.map((st) => h("option", { value: st, selected: s.region === st }, st))),
        h("p", { class: "hint", id: "region-hint" }, t("symptoms.region.hint"))),
    ),
    h("p", { class: "muted", role: "status" }, nNoted ? t("symptoms.count", { n: nNoted }) : t("symptoms.none_yet")),
    submitState(),
    navRow(btn("← " + t(s.result ? "steps.report" : "steps.plant"), () => go(s.result ? "report" : "plant"), "btn btn-ghost"),
      btn(t("symptoms.update_report"), submit, "btn btn-primary", { disabled: !S.readiness(s).ready || s.submit.busy, id: "submit" })));
}

// ------------------------------------------------------------------------------------------------ review & submit
async function submit() {
  const s = store.get();
  set((x) => ({ ...x, submit: { busy: true, error: null } }));
  announce(t("review.working"));
  try {
    const data = await api.assess(S.assessFields(s), s.photo.file);
    set((x) => ({ ...x, result: data, submit: { busy: false, error: null } }));
    go("report");
  } catch (e) {
    set((x) => ({ ...x, submit: { busy: false, error: { message: friendlyError(e, t), code: e.code } } }));
  }
}

function submitState() {
  const s = store.get();
  return [s.submit.error ? notice("bad", h("strong", {}, t("errors.title")), h("p", {}, s.submit.error.message)) : null,
    s.submit.busy ? h("div", { class: "card working", role: "status", "aria-live": "polite" }, h("span", { class: "spinner", "aria-hidden": "true" }),
      h("div", {}, h("strong", {}, t("review.working")), h("p", { class: "muted" }, t("review.working.detail")))) : null];
}

// ------------------------------------------------------------------------------------------------ report
function fullReport() {
  const s = store.get();
  const lang = s.result.report.language;
  const m = buildReportModel(s.result, lang);
  const sec = (title, ic, ...kids) => h("section", { class: "card rsec" }, h("h2", {}, icon(ic), title), ...kids);
  const symBlock = (label, items, cls) => items.length ? h("div", { class: "sym " + cls }, h("h3", {}, label), h("ul", {}, items.map((x) => h("li", {}, x)))) : null;
  const pc = m.photoCheck;
  return h("div", { class: "full-report" },
    h("div", { class: `banner banner-${m.tone}` }, icon(m.tone === "bad" ? "alert" : m.tone === "warn" ? "alert" : "info"),
      h("div", {}, h("p", { class: "eyebrow" }, t("report.title")),
        h("h2", {}, t(m.bannerKey, { plant: m.plant.name || t("report.plant.unknown") })),
        h("p", { class: "small" }, t("report.not_expert_reviewed"), " · ", t("report.id", { id: m.reportId })))),
    m.hindiDraft ? notice("warn", t("report.hi_draft")) : null,
    h("div", { class: "report-grid" },
      h("div", { class: "col-main" },
        m.type === "insufficient_image" ? sec(t("report.retake"), "camera",
          h("ul", { class: "retake" }, m.retake.map((r) => h("li", {}, h("strong", {}, t("photo.retake_view", { view: tk(r.view) })), " — ", tk(r.why)))),
          btn(t("report.retake"), () => go("photo"), "btn btn-primary")) : null,
        sec(t("report.candidates"), "list",
          m.insufficient ? h("p", {}, m.insufficient) : null,
          m.candidates.length ? m.candidates.map((c) => h("article", { class: "cand" },
            h("div", { class: "cand-head" }, h("h3", {}, c.name), h("span", { class: "badge badge-warn" }, t("report.candidate.status"))),
            h("p", { class: "muted small" }, c.fromModel ? t("report.candidate.from_model") : t("report.candidate.from_symptoms")),
            c.supporting.length ? h("div", {}, h("h4", {}, t("report.candidate.why")),
              h("ul", { class: "tagged" }, c.supporting.map((x) => h("li", {}, h("span", { class: "tag tag-" + x.origin },
                x.origin === "user_report" ? t("report.symptoms.user") : x.origin === "knowledge_base_typical" ? t("report.symptoms.typical") : t("report.symptoms.image")), " ", tk(x.label))))) : null,
            c.tests.length ? h("div", {}, h("h4", {}, t("report.candidate.check")), h("ul", {}, c.tests.map((x) => h("li", {}, x)))) : null,
            h("p", { class: "muted small" }, c.uncertainty))) : (!m.insufficient ? h("p", {}, t("report.candidates.none")) : null)),
        sec(t("report.actions"), "check", h("ol", { class: "actions" }, m.actions.map((a) => h("li", { class: a.subcategory === "consult_expert" ? "act-expert" : "" }, a.text)))),
        m.management.length ? sec(t("report.management"), "leaf", h("ul", {}, m.management.map((x) => h("li", {}, x)))) : null,
        sec(t("report.uncertainty"), "info",
          h("p", { class: "level level-" + m.uncertainty.level }, t("report.uncertainty." + m.uncertainty.level)),
          h("details", {}, h("summary", {}, t("report.uncertainty.details")), h("ul", {}, m.uncertainty.statements.map((x) => h("li", {}, x))))),
        sec(t("report.safety"), "shield", h("ul", { class: "warnings" }, m.warnings.map((w) => h("li", {}, w))), h("p", { class: "muted small" }, m.disclaimer))),
      h("aside", { class: "col-side" },
        sec(t("report.plant"), "leaf", h("p", { class: "big" }, m.plant.name || t("report.plant.unknown")),
          h("p", { class: "muted small" }, `${t("plant.user_given")} · ${t("plant.not_verified")}`),
          m.plant.entered && m.plant.entered.toLowerCase() !== (m.plant.name || "").toLowerCase() ? h("p", { class: "muted small" }, t("plant.confirm.you_typed", { text: m.plant.entered })) : null),
        sec(t("report.photo_check"), "camera",
          h("p", { class: "strong" }, pc.ran ? t("report.photo_check.ran") : t("report.photo_check.not_run")),
          h("p", { class: "small" }, pc.ran ? t(pc.decision === "known" ? "report.photo_check.known" : "report.photo_check.unknown") : pc.reason),
          pc.ran && pc.decision === "known" && pc.agrees === false ? notice("warn", t("report.photo_check.mismatch")) : null),
        s.photo.url ? sec(t("report.marks.title"), "image",
          s.photo.natW ? markStage(s.photo, false) : h("img", { src: s.photo.url, alt: t("photo.preview_alt"), class: "thumb" }),
          m.marks.length ? h("p", { class: "mark-label" }, icon("info"), t("mark.label")) : h("p", { class: "muted small" }, t("report.marks.none")),
          h("p", { class: "muted small" }, t("report.highlight.unavailable"))) : null,
        sec(t("report.symptoms"), "list",
          symBlock(t("report.symptoms.user"), m.symptoms.userSeen, "user"),
          symBlock(t("report.symptoms.user_absent"), m.symptoms.userAbsent, "user"),
          symBlock(t("report.symptoms.image"), m.symptoms.image, "image"),
          symBlock(t("report.symptoms.typical"), m.symptoms.typical, "typical"),
          !m.symptoms.userSeen.length && !m.symptoms.userAbsent.length ? h("p", { class: "muted" }, t("symptoms.none_yet")) : null),
        m.expertReferral ? h("section", { class: "card expert" }, h("h2", {}, icon("expert"), t("report.expert.title")), h("p", {}, t("report.expert.body"))) : null,
        reportPlayer(s.result))),
    sec(t("report.sources"), "report", h("p", { class: "muted small" }, t("report.sources.lead")),
      h("ul", { class: "sources" }, m.sources.map((x) => h("li", {}, h("strong", {}, x.title), " — ", x.by, " ", h("span", { class: "muted small url" }, x.url))))));
}

// Farmer summary (SC-M5-02): crop, scientific name, soil & nutrients, temperature, signs seen, then the disease or the
// possible diseases with checks and what to do. Knowledge-base text is shown through tk() (draft translation or English).
function viewReport() {
  const s = store.get();
  if (!s.result) return viewPlant();
  const sm = s.result.summary;
  const c = sm.crop;
  const say = (...parts) => parts.flat().filter(Boolean).join(". ");
  const sec = (id, title, ic, speech, ...kids) => h("section", { class: "card rsec", "aria-labelledby": id + "-h" },
    h("div", { class: "rsec-head" }, h("h2", { id: id + "-h" }, icon(ic), title), listenBtn(speech, "listen-" + id, true)), ...kids);
  const fact = (f) => f ? [h("p", {}, tk(f.text)), h("p", { class: "muted small" }, t("sum.source"), ": ", f.sources.join(", "))] :
    h("p", { class: "muted" }, t("sum.not_sourced"));
  const sg = sm.signs;
  const seen = sg && sg.ok ? sg.signs.filter((x) => x.seen) : [];
  const head = { known: "sum.head.known", possible: "sum.head.possible", healthy_looking: "sum.head.healthy", none: "sum.head.none" }[sm.kind];
  const disease = (d, n) => h("article", { class: "cand", id: "disease-" + n },
    h("div", { class: "cand-head" }, h("h3", {}, sm.kind === "possible" ? t("sum.rank", { n }) + " " : "", tk(d.name)),
      h("span", { class: "badge badge-warn" }, t("report.candidate.status")), listenBtn(() => diseaseSpeech(d, n), "listen-d" + n, true)),
    h("p", { class: "muted small" }, t("sum.cause"), ": ", tk(d.cause), d.scientific_cause ? h("i", {}, " (" + d.scientific_cause + ")") : null),
    d.signs_matched.length ? h("div", {}, h("h4", {}, t("sum.matched")), h("ul", {}, d.signs_matched.map((x) => h("li", {}, tk(x))))) : null,
    d.typical_signs.length ? h("div", {}, h("h4", {}, t("sum.typical")), h("ul", {}, d.typical_signs.map((x) => h("li", {}, tk(x))))) : null,
    sm.kind === "possible" && d.checks.length ? h("div", { class: "checks" }, h("h4", {}, icon("check"), t("sum.checks")),
      h("ol", {}, d.checks.map((x) => h("li", {}, h("strong", {}, tk(x.name)), ": ", tk(x.how),
        x.shows ? h("p", { class: "muted small" }, t("sum.check_shows"), " ", tk(x.shows)) : null)))) : null,
    h("div", { class: "cure" }, h("h4", {}, icon("leaf"), sm.kind === "possible" ? t("sum.cure_possible") : t("sum.cure")),
      d.curability.text ? h("p", {}, tk(d.curability.text)) : null,
      d.what_to_do.length ? h("ul", {}, d.what_to_do.map((x) => h("li", {}, practice(x)))) : h("p", { class: "muted" }, t("sum.no_steps")),
      d.sprays_named_by_source.length ? h("details", { class: "sprays" }, h("summary", {}, t("sum.sprays")),
        h("p", { class: "muted small" }, t("sum.sprays_note")), h("ul", {}, d.sprays_named_by_source.map((x) => h("li", {}, practice(x))))) : null),
    h("p", { class: "muted small" }, t("sum.sources"), ": ", d.sources.join(", ")));
  return h("section", { class: "page report" }, stepper("report"),
    h("div", { class: "banner banner-" + (sm.kind === "known" ? "warn" : sm.kind === "healthy_looking" ? "ok" : "info") },
      icon(sm.kind === "healthy_looking" ? "check" : "info"),
      h("div", {}, h("p", { class: "eyebrow" }, t("report.title")),
        h("h1", { tabindex: "-1", id: "page-title" }, t(head, { plant: tk(c.name) || t("report.plant.unknown"),
          disease: sm.diseases[0] ? tk(sm.diseases[0].name) : "" })),
        h("p", { class: "small" }, t("report.not_expert_reviewed")),
        listenBtn(() => reportSpeech(sm), "listen-all"))),
    h("div", { class: "report-grid" },
      h("div", { class: "col-main" },
        sm.kind === "known" || sm.kind === "possible" ? sec("diseases", t(sm.kind === "known" ? "sum.disease" : "sum.possible"), "list",
          () => sm.diseases.map((d, i) => diseaseSpeech(d, i + 1)).join(". "),
          sm.kind === "possible" ? h("p", { class: "muted" }, t("sum.possible_lead")) : null,
          sm.diseases.map((d, i) => disease(d, i + 1))) :
          sec("diseases", t(sm.kind === "healthy_looking" ? "sum.healthy_title" : "sum.none_title"), "info", () => sm.notes.map(tk).join(". "),
            h("p", {}, tk(sm.notes[0]))),
        sec("signs", t("sum.signs"), "image", () => seen.length ? t("sum.signs") + ": " + seen.map((x) => tk(x.label)).join(", ") : t("sum.signs_none"),
          s.photo.url ? h("img", { src: s.photo.url, alt: t("photo.preview_alt"), class: "thumb" }) : null,
          !sg ? h("p", { class: "muted" }, t("sum.signs_not_run")) : !sg.ok ? notice("warn", tk(sg.reason)) :
            seen.length ? h("ul", { class: "signs" }, seen.map((x) => h("li", {}, icon("check"), tk(x.label)))) : h("p", {}, t("sum.signs_none")),
          sg && sg.ok ? h("p", { class: "muted small" }, tk(sg.limits)) : null,
          h("p", { class: "muted small" }, tk(sm.model.note), sm.model.reason ? " " + tk(sm.model.reason) : ""),
          btn(t("sum.add_details"), () => go("symptoms"), "btn btn-ghost btn-small", { id: "add-details" })),
        sec("notes", t("report.safety"), "shield", () => sm.notes.map(tk).join(". "), h("ul", { class: "warnings" }, sm.notes.map((x) => h("li", {}, tk(x)))))),
      h("aside", { class: "col-side" },
        sec("crop", t("sum.crop"), "leaf", () => say(t("sum.crop") + ": " + tk(c.name), c.scientific_name ? t("sum.sci") + ": " + c.scientific_name : null),
          h("p", { class: "big" }, tk(c.name) || t("report.plant.unknown")),
          c.scientific_name ? h("p", {}, t("sum.sci"), ": ", h("i", {}, c.scientific_name), c.family ? " (" + c.family + ")" : "") :
            h("p", { class: "muted" }, t("sum.not_in_kb")),
          h("p", { class: "muted small" }, t("plant.user_given") + " · " + t("plant.not_verified"))),
        sec("temp", t("sum.temperature"), "info", () => c.temperature ? tk(c.temperature.text) : t("sum.not_sourced"), fact(c.temperature)),
        sec("soil", t("sum.soil"), "leaf", () => say(c.soil ? tk(c.soil.text) : t("sum.not_sourced"),
          c.nutrients ? t("sum.nutrients") + ": " + c.nutrients.needed.map(tk).join(", ") : null),
          fact(c.soil),
          h("h3", {}, t("sum.nutrients")),
          c.nutrients ? [h("ul", { class: "chips-list" }, c.nutrients.needed.map((x) => h("li", { class: "chip-static" }, tk(x)))),
            h("p", {}, tk(c.nutrients.text)), h("p", { class: "muted small" }, t("sum.no_doses"))] : h("p", { class: "muted" }, t("sum.not_sourced"))))),
    h("details", { class: "card full" }, h("summary", {}, t("sum.full_report")), fullReport()),
    h("div", { class: "btn-row center no-print" },
      btn(t("sum.add_details"), () => go("symptoms"), "btn btn-ghost"),
      btn(t("report.new"), () => { removePhoto(); store.update((x) => ({ ...S.initialState(), uiLang: x.uiLang, voiceConsent: x.voiceConsent })); go("photo"); },
        "btn btn-primary", { id: "report-new" })));
}

const practice = (x) => tk(x.text) + (x.outside_india ? " " + tk("(source outside India; check locally)") : "");

function diseaseSpeech(d, n) {
  const sm = store.get().result.summary;
  return [(sm.kind === "possible" ? t("sum.rank", { n }) + " " : "") + tk(d.name),
    d.signs_matched.length ? t("sum.matched") + ": " + d.signs_matched.map(tk).join(", ") : null,
    sm.kind === "possible" && d.checks.length ? t("sum.checks") + ": " + d.checks.map((x) => tk(x.name) + ", " + tk(x.how)).join(". ") : null,
    d.what_to_do.length ? t(sm.kind === "possible" ? "sum.cure_possible" : "sum.cure") + ": " + d.what_to_do.map(practice).join(". ") : null]
    .filter(Boolean).join(". ");
}

function reportSpeech(sm) {
  const c = sm.crop;
  return [t("sum.crop") + ": " + tk(c.name), c.scientific_name ? t("sum.sci") + ": " + c.scientific_name : null,
    c.temperature ? t("sum.temperature") + ": " + tk(c.temperature.text) : null, c.soil ? t("sum.soil") + ": " + tk(c.soil.text) : null,
    c.nutrients ? t("sum.nutrients") + ": " + c.nutrients.needed.map(tk).join(", ") : null,
    ...sm.diseases.map((d, i) => diseaseSpeech(d, i + 1)), tk(sm.notes[0])].filter(Boolean).join(". ");
}

// ------------------------------------------------------------------------------------------------ help
function viewHelp() {
  const block = (k, n) => h("section", { class: "card" }, h("h2", {}, t(k + ".title")),
    h("ul", {}, Array.from({ length: n }, (_, i) => h("li", {}, t(`${k}.${i + 1}`)))));
  return page(t("help.title"), null, block("help.how", 4), block("help.privacy", 5), block("help.limits", 6), block("help.voice", 4), block("help.languages", 1),
    h("div", { class: "btn-row" }, btn(t("help.back"), () => history.length > 1 ? history.back() : go("home"), "btn btn-ghost"),
      btn(t("home.start"), () => go("photo"), "btn btn-primary")));
}

function viewError(message) {
  return page(t("errors.title"), null, notice("bad", message || t("status.offline")), btn(t("errors.retry"), () => location.reload(), "btn btn-primary"));
}

// ------------------------------------------------------------------------------------------------ router
const VIEWS = { home: viewHome, plant: viewPlant, photo: viewPhoto, symptoms: viewSymptoms, review: viewPlant, report: viewReport, help: viewHelp };

export function render() {
  const route = (location.hash.replace(/^#\/?/, "") || "home").split("?")[0];
  const r = VIEWS[route] ? route : "home";
  if (r !== "photo" && cameraOpen()) { camSession.close(); return; }   // close() re-renders this route
  if (r !== "report" && player.state.state !== "stopped") player.stop();
  if (vflow && lastRoute && r !== lastRoute) { if (recorder) { recorder.cancel(); recorder = null; } clearInterval(vtimer); vflow = null; }
  store.update((x) => (x.route === r ? x : { ...x, route: r }), true);
  const active = document.activeElement;
  const activeId = active && active.id;
  const sel = active && typeof active.selectionStart === "number" ? [active.selectionStart, active.selectionEnd] : null;
  const root = document.getElementById("app");
  let view;
  try {
    view = VIEWS[r]();
  } catch (e) {
    console.error(e);
    view = viewError(t("errors.generic"));
  }
  root.replaceChildren(header(), h("main", { id: "main", tabindex: "-1" }, h("div", { class: "container" }, langDraftNotice(), view)),
    h("footer", { class: "footer" }, h("p", {}, icon("shield"), t("status.local"), " · ", h("a", { href: "#/help" }, t("nav.help")))));
  const want = focusNext && document.getElementById(focusNext);
  focusNext = null;
  if (want && r === lastRoute) {
    want.focus({ preventScroll: true });
  } else if (r !== lastRoute) {
    lastRoute = r;
    window.scrollTo(0, 0);
    const title = document.getElementById("page-title");
    if (title) title.focus({ preventScroll: true });
  } else if (activeId && document.getElementById(activeId)) {
    const el = document.getElementById(activeId);
    el.focus({ preventScroll: true });
    if (sel && typeof el.setSelectionRange === "function") try { el.setSelectionRange(sel[0], sel[1]); } catch (e) { /* not a text field */ }
  }
}

export async function boot() {
  const root = document.getElementById("app");
  root.replaceChildren(h("p", { class: "boot", role: "status" }, "Loading…"));
  try {
    const [en, hi, health, plants, sym, regions] = await Promise.all([api.i18n("en"), api.i18n("hi"), api.health(), api.plants(),
      api.symptoms(), api.regions()]);
    K.catalogues = { en, hi };
    I = createI18n(K.catalogues, "en");
    if (Array.isArray(health.ui_languages) && health.ui_languages.length) LANGS = health.ui_languages;
    Object.assign(K, { health, plants: plants.plants, features: sym.features, questions: sym.questions, states: regions.states });
    try { K.speech = await api.speechStatus(); } catch (e) { K.speech = null; }        // voice is optional at runtime
  } catch (e) {
    I = createI18n({ en: { strings: {} } });
    root.replaceChildren(h("main", { id: "main" }, h("div", { class: "container" }, viewError(e instanceof ApiFailure && e.code === "network" ?
      "Cannot reach the AgroVaidya service on this computer. Check that it is running and try again." : "AgroVaidya could not start. Please reload the page."))));
    return false;
  }
  document.documentElement.lang = "en";
  store.subscribe(() => render());
  window.addEventListener("hashchange", render);
  // installed voices can appear after start-up (and after a language voice is installed): refresh the listen buttons
  try { if (globalThis.speechSynthesis) speechSynthesis.addEventListener("voiceschanged", () => render()); } catch (e) { /* no speech */ }
  render();
  return true;
}

// test hooks for the in-browser self-test (selftest.js); not used by the app itself
// Self-test only: runs a WAV (e.g. produced by the real speech-output engine) through the same recognise -> confirm
// path as a microphone recording. It does NOT test a microphone.
async function voiceFromWav(target, wavBlob) {
  vflow = { name: "processing", target };
  render();
  try {
    vstep({ type: "result", result: await api.transcribe(wavBlob, vlang(), target === "plant" ? "plant_name" : "free") });
  } catch (e) {
    const sp = e instanceof ApiFailure && e.details && e.details.speech;
    vstep({ type: "result", result: sp || { status: "service_error" } });
  }
  return vflow;
}

export const __test = { store, S, K, render, go, t: (k, p) => t(k, p), acceptFile, voiceFromWav, player, vflowState: () => vflow };

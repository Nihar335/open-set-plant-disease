// Case state for one browser tab (M5-D2). Kept in memory only: nothing is written to localStorage, sessionStorage,
// cookies or the server. Pure functions so they can be tested without a browser.

export const MODES = ["tap", "text", "voice"];
// Photo first, then the plant name, then the report (owner decision 2026-10-15, SC-M5-02). Telling the app what you see
// is optional and reached from the report ("symptoms" route); it is not a step.
export const STEPS = ["photo", "plant", "report"];

export function initialState() {
  return {
    route: "home",
    plant: { query: "", resolution: null, pending: null, choice: null, corrections: 0 },
    photo: { file: null, url: null, name: null, check: null, checking: false, error: null, source: null, marks: [], natW: 0, natH: 0 },
    symptoms: { mode: "tap", features: {}, channels: {}, answers: {}, text: "", suggest: null },
    region: "",
    uiLang: "en",                        // interface language chosen on the home screen
    language: "en",                      // report-engine language (en or hi); derived from uiLang at submit
    voiceLang: "en",
    voiceConsent: false,                 // asked once per tab; never stored
    submit: { busy: false, error: null },
    result: null,
  };
}

export function createStore(init = initialState()) {
  let state = init;
  const subs = new Set();
  return {
    get: () => state,
    // silent: store the change without re-rendering (used while the user types, so focus is never disturbed)
    update(fn, silent = false) {
      const next = fn(state);
      if (next !== state) {
        state = next;
        if (!silent) subs.forEach((s) => s(state));
      }
      return state;
    },
    subscribe(fn) {
      subs.add(fn);
      return () => subs.delete(fn);
    },
  };
}

// ---- plant ----------------------------------------------------------------------------------------------------
// pending: what the user is about to confirm: {action, text?, plant_id?, plant?} ; choice: the confirmed pending.
export function setPending(s, pending) {
  return { ...s, plant: { ...s.plant, pending, choice: null } };
}

export function confirmPlant(s) {
  if (!s.plant.pending) return s;
  return { ...s, plant: { ...s.plant, choice: { ...s.plant.pending, confirmed: true } } };
}

export function correctPlant(s) {
  const was = s.plant.choice || s.plant.pending;
  return { ...s, plant: { ...s.plant, pending: null, choice: null, resolution: null,
                          corrections: s.plant.corrections + (was ? 1 : 0) }, result: null };
}

// ---- symptoms -------------------------------------------------------------------------------------------------
export function setMode(s, mode, voiceAvailable = false) {
  if (!MODES.includes(mode)) return s;
  if (mode === "voice" && !voiceAvailable) return s; // gated: never switches into an unavailable mode
  return { ...s, symptoms: { ...s.symptoms, mode } };
}

// value: true (seen), false (looked, not seen) or null (clear)
export function setFeature(s, featureId, value, channel = "symptom_checklist") {
  const features = { ...s.symptoms.features };
  const channels = { ...s.symptoms.channels };
  if (value === null) {
    delete features[featureId];
    delete channels[featureId];
  } else {
    features[featureId] = value;
    if (channel === "symptom_checklist") delete channels[featureId];
    else channels[featureId] = channel;
  }
  return { ...s, symptoms: { ...s.symptoms, features, channels }, result: null };
}

export function setAnswer(s, questionId, optionId) {
  const answers = { ...s.symptoms.answers };
  if (optionId == null) delete answers[questionId];
  else answers[questionId] = optionId;
  return { ...s, symptoms: { ...s.symptoms, answers }, result: null };
}

// Ticked "seen" but a question answer sets the same feature to false: shown to the user (backend: tick wins).
export function conflicts(s, questions) {
  const out = [];
  for (const q of questions || []) {
    const opt = s.symptoms.answers[q.question_id];
    const o = q.options.find((x) => x.option_id === opt);
    if (!o || !o.sets) continue;
    for (const [f, v] of Object.entries(o.sets)) {
      if (f in s.symptoms.features && s.symptoms.features[f] !== v) out.push({ feature_id: f, question_id: q.question_id });
    }
  }
  return out;
}

// ---- submission -----------------------------------------------------------------------------------------------
export function readiness(s) {
  const missing = [];
  if (!s.plant.choice) missing.push("plant");
  if (!s.photo.file) missing.push("photo");
  return { ready: missing.length === 0, missing };
}

// Fields of the multipart request (the image is appended separately). Only confirmed choices are sent.
export function assessFields(s) {
  const c = s.plant.choice;
  if (!c) throw new Error("plant not confirmed");
  const plant = { action: c.action, confirmed: true, corrections: s.plant.corrections };
  if (c.text != null) plant.text = c.text;
  if (c.plant_id != null) plant.plant_id = c.plant_id;
  if (c.channel === "voice_transcript_confirmed") plant.channel = c.channel;
  const f = { plant: JSON.stringify(plant), features: JSON.stringify(s.symptoms.features),
              channels: JSON.stringify(s.symptoms.channels), answers: JSON.stringify(s.symptoms.answers),
              language: s.uiLang === "hi" ? "hi" : "en" };
  if (s.region) f.state = s.region;
  if (s.photo.marks && s.photo.marks.length) f.marks = JSON.stringify(s.photo.marks);
  return f;
}

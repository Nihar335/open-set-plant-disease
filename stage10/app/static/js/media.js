// Photo and camera helpers (M5-A1) and the privacy-gated microphone flow (M5-B1). The server re-validates every photo.

export const ACCEPT = ["image/jpeg", "image/png"];

// Client-side pre-check only (fast feedback); the server's quality gate is authoritative.
export function precheckFile(file, maxMb) {
  if (!file) return { ok: false, key: "photo.needed" };
  const type = (file.type || "").toLowerCase();
  const name = (file.name || "").toLowerCase();
  const typeOk = ACCEPT.includes(type) || (!type && /\.(jpe?g|png)$/.test(name));
  if (!typeOk) return { ok: false, key: "photo.bad_type" };
  if (file.size > maxMb * 1024 * 1024) return { ok: false, key: "photo.too_big", params: { mb: maxMb } };
  return { ok: true };
}

// DOMException names from getUserMedia -> message keys.
export function cameraErrorKey(err, secureContext = true) {
  if (!secureContext) return "photo.camera.insecure";
  const n = err && err.name;
  if (n === "NotAllowedError" || n === "SecurityError" || n === "PermissionDeniedError") return "photo.camera.denied";
  if (n === "unsupported") return "photo.camera.unsupported";
  if (n === "NotFoundError" || n === "OverconstrainedError" || n === "DevicesNotFoundError") return "photo.camera.unavailable";
  if (n === "NotReadableError" || n === "TrackStartError" || n === "AbortError") return "photo.camera.busy";
  return "photo.camera.error";
}

export function cameraSupported(nav = globalThis.navigator) {
  return !!(nav && nav.mediaDevices && typeof nav.mediaDevices.getUserMedia === "function");
}

export async function startCamera(video, nav = globalThis.navigator) {
  if (!cameraSupported(nav)) {
    const e = new Error("unsupported");
    e.name = "unsupported";
    throw e;
  }
  const stream = await nav.mediaDevices.getUserMedia({ video: { facingMode: { ideal: "environment" } }, audio: false });
  video.srcObject = stream;
  await video.play().catch(() => {});
  return stream;
}

export function stopCamera(stream, video) {
  if (stream) stream.getTracks().forEach((tr) => tr.stop());
  if (video) video.srcObject = null;
}

export function captureFrame(video) {
  const c = document.createElement("canvas");
  c.width = video.videoWidth;
  c.height = video.videoHeight;
  c.getContext("2d").drawImage(video, 0, 0);
  return new Promise((resolve) =>
    c.toBlob((b) => resolve(b ? new File([b], "camera.jpg", { type: "image/jpeg" }) : null), "image/jpeg", 0.92));
}

// ---- camera session (M5-A1) ------------------------------------------------------------------------------------
// States: idle -> starting -> live -> captured (still shown) -> [use -> idle] | [retake -> live]; cancel from any state.
// Every exit path stops all camera tracks, including a stream that arrives after the user already cancelled.
// Nothing is recorded: one frame is grabbed in memory only when the user presses Capture.
export function createCameraSession({ nav = globalThis.navigator, secure = globalThis.isSecureContext !== false,
                                      grab = captureFrame, onChange = () => {} } = {}) {
  let state = { name: "idle", error: null, still: null };
  let stream = null;
  let token = 0;
  const set = (s) => { state = { ...state, ...s }; onChange(state); };
  const stopTracks = () => { if (stream) stream.getTracks().forEach((tr) => tr.stop()); stream = null; };

  async function open(video) {
    const my = ++token;
    stopTracks();
    set({ name: "starting", error: null, still: null });
    try {
      if (!secure) throw Object.assign(new Error("insecure"), { name: "insecure" });
      if (!cameraSupported(nav)) throw Object.assign(new Error("unsupported"), { name: "unsupported" });
      let s;
      try {
        s = await nav.mediaDevices.getUserMedia({ video: { facingMode: { ideal: "environment" } }, audio: false });
      } catch (e) {
        if (e && e.name === "OverconstrainedError") s = await nav.mediaDevices.getUserMedia({ video: true, audio: false });
        else throw e;
      }
      if (my !== token) { s.getTracks().forEach((tr) => tr.stop()); return state; }   // cancelled while starting
      stream = s;
      attach(video);
      set({ name: "live" });
    } catch (e) {
      if (my !== token) return state;
      stopTracks();
      set({ name: "error", error: cameraErrorKey(e, secure) });
    }
    return state;
  }

  function attach(video) {
    if (video && stream) {
      video.srcObject = stream;
      const p = video.play && video.play();
      if (p && p.catch) p.catch(() => {});
    }
  }

  async function capture(video) {
    if (state.name !== "live") return null;
    const file = await grab(video);
    if (!file) { set({ name: "error", error: "photo.camera.error" }); stopTracks(); return null; }
    set({ name: "captured", still: file });          // stream kept running only so Retake is instant
    return file;
  }

  function use() {
    const f = state.still;
    close();
    return f;
  }

  function retake(video) {
    if (state.name !== "captured") return;
    set({ name: "live", still: null });
    attach(video);
  }

  function close() {
    token++;
    stopTracks();
    set({ name: "idle", error: null, still: null });
  }

  return { open, capture, use, retake, close, attach, get state() { return state; }, get active() { return !!stream; } };
}

// ---- voice (privacy-gated; M5-B1..B5, M5-E1..E4) ---------------------------------------------------------------
// The microphone is never requested unless the server reports voice input as available AND the user presses the
// button. Today the server reports it unavailable (speech-to-text engine not approved), so the flow stops at "pending".
export function micFlow(event, ctx) {
  const { available, userAction } = ctx;
  if (!available) return { state: "pending", requestMic: false, fallback: "text" };
  if (event === "press" && !userAction) return { state: "idle", requestMic: false };
  if (event === "press") return { state: "requesting", requestMic: true };
  if (event === "granted") return { state: "recording", requestMic: false };
  if (event === "denied") return { state: "denied", requestMic: false, fallback: "text" };
  if (event === "unavailable") return { state: "unavailable", requestMic: false, fallback: "text" };
  if (event === "cancel") return { state: "idle", requestMic: false };
  if (event === "transcribed") return { state: "confirm", requestMic: false };  // text shown; never used before confirmation
  return { state: "idle", requestMic: false };
}

// Full voice-interaction state machine (contract for M5-B1..B5; SPEECH_DECISION.md section 5). Pure: no microphone,
// no network. ctx = { available, consentAccepted }. A transcript only leaves this machine through state "confirmed",
// which needs an explicit user "confirm" (or a correction typed by the user). Cancel discards everything.
export const VOICE_STATES = ["unavailable", "idle", "consent", "requesting", "denied", "listening", "processing", "confirm",
  "retry", "confirmed"];

export function voiceStep(state, event, ctx = {}) {
  const s = state || { name: ctx.available ? "idle" : "unavailable", transcript: null, message: null };
  const go = (name, extra = {}) => ({ name, transcript: null, message: null, ...extra });
  if (!ctx.available) return go("unavailable", { message: "speech.unavailable" });
  if (event.type === "cancel") return go("idle", { message: "speech.cancelled" });            // discard at any stage
  switch (s.name) {
    case "unavailable":
    case "idle":
    case "confirmed":
    case "denied":
    case "retry":
      if (event.type === "press") return ctx.consentAccepted ? go("requesting") : go("consent");
      if (event.type === "type_instead") return go("idle", { message: "speech.type_instead" });
      return s;
    case "consent":
      if (event.type === "consent_accept") return go("requesting");
      if (event.type === "consent_decline") return go("idle", { message: "speech.type_instead" });
      return s;
    case "requesting":
      if (event.type === "granted") return go("listening");
      if (event.type === "denied") return go("denied", { message: "speech.denied" });
      if (event.type === "no_device") return go("denied", { message: "speech.no_device" });
      return s;
    case "listening":
      if (event.type === "stop" || event.type === "timeout") return go("processing");
      return s;
    case "processing":
      if (event.type === "result") {
        const r = event.result || {};
        if (r.status === "ok" && r.text) return go("confirm", { transcript: r.text, alternatives: r.alternatives || [] });
        // low confidence, no speech, unsupported language, errors: never usable without the user repeating or typing
        return go("retry", { message: "speech." + (r.status || "service_error"), heard: r.status === "low_confidence" ? r.text : null });
      }
      if (event.type === "error") return go("retry", { message: "speech.service_error" });
      return s;
    case "confirm":
      if (event.type === "confirm") return go("confirmed", { transcript: s.transcript, channel: "voice_transcript_confirmed" });
      if (event.type === "correct" && typeof event.text === "string" && event.text.trim())
        return go("confirmed", { transcript: event.text.trim(), channel: "typed_text_confirmed", corrected: true });
      if (event.type === "repeat") return ctx.consentAccepted ? go("requesting") : go("consent");
      return s;
    default:
      return s;
  }
}

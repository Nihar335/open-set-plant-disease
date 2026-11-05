// Thin client for the local AgroVaidya API. All decisions are made by the server; this file only transports data.

export const API = "/api/v1";

export class ApiFailure extends Error {
  constructor(code, message, details, status) {
    super(message);
    this.code = code;
    this.details = details || null;
    this.status = status || 0;
  }
}

// Turn any failure into something safe to show: server messages are already user-facing; anything else is generic.
export function friendlyError(err, t) {
  if (err instanceof ApiFailure) {
    if (err.code === "network") return t("errors.network");
    return err.message || t("errors.generic");
  }
  return t("errors.generic");
}

export async function call(method, path, body, fetchImpl = globalThis.fetch) {
  let res;
  try {
    const opts = { method, headers: { Accept: "application/json" } };
    if (body instanceof FormData) opts.body = body;
    else if (body !== undefined) {
      opts.body = JSON.stringify(body);
      opts.headers["Content-Type"] = "application/json";
    }
    res = await fetchImpl(API + path, opts);
  } catch (e) {
    throw new ApiFailure("network", "", null, 0);
  }
  let json = null;
  try {
    json = await res.json();
  } catch (e) {
    throw new ApiFailure("bad_response", "", null, res.status);
  }
  if (!json || json.ok !== true) {
    const e = (json && json.error) || {};
    throw new ApiFailure(e.code || "error", e.message || "", e.details, res.status);
  }
  return json.data;
}

async function postBinary(path, body, fetchImpl = globalThis.fetch) {
  let res;
  try {
    res = await fetchImpl(API + path, body);
  } catch (e) {
    throw new ApiFailure("network", "", null, 0);
  }
  if (res.ok && (res.headers.get("Content-Type") || "").startsWith("audio/")) return res.blob();
  let j = null;
  try { j = await res.json(); } catch (e) { /* not JSON */ }
  const e = (j && j.error) || {};
  throw new ApiFailure(e.code || "error", e.message || "", e.details, res.status);
}

export const api = {
  speechStatus: () => call("GET", "/speech/status"),
  speechWarmup: (language) => call("POST", "/speech/warmup", { language }),
  transcribe: (wavBlob, language, purpose) => {
    const fd = new FormData();
    fd.append("audio", wavBlob, "speech.wav");
    fd.append("language", language);
    if (purpose) fd.append("purpose", purpose);
    return call("POST", "/speech/transcribe", fd);
  },
  synthesize: (text, language, speed = 1.0) =>
    postBinary("/speech/synthesize", { method: "POST", headers: { "Content-Type": "application/json" },
                                       body: JSON.stringify({ text, language, speed }) }),
  health: () => call("GET", "/health"),
  plants: () => call("GET", "/plants"),
  resolve: (text) => call("POST", "/plant-name/resolve", { text }),
  symptoms: () => call("GET", "/symptoms"),
  suggest: (text) => call("POST", "/symptoms/suggest", { text }),
  regions: () => call("GET", "/regions"),
  i18n: (lang) => call("GET", `/i18n/${lang}`),
  checkImage: (file) => {
    const fd = new FormData();
    fd.append("image", file, file.name || "photo.jpg");
    return call("POST", "/image/check", fd);
  },
  assess: (fields, file) => {
    const fd = new FormData();
    for (const [k, v] of Object.entries(fields)) fd.append(k, v);
    fd.append("image", file, file.name || "photo.jpg");
    return call("POST", "/assess", fd);
  },
};

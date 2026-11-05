/* LeafScan browser version: runs off the main thread so the page stays responsive.
 * The photo is read and analysed here, in the visitor's browser, and is never uploaded.
 * Only the place name and coordinates leave the browser, to the Open-Meteo weather service.
 */
"use strict";
importScripts("vendor/ort.wasm.min.js", "core.js", "pixels.js");

const MODEL_URL = "model.onnx", KB_URL = "kb.json", WEATHER_TIMEOUT_MS = 8000;
let core = null, session = null;

async function loadCore() {
  if (!core) {
    const r = await fetch(KB_URL);
    if (!r.ok) throw new Error("Could not load the app data (HTTP " + r.status + ").");
    core = new LeafCore.Core(await r.json());
  }
  return core;
}

async function loadSession(id) {
  if (session) return session;
  post(id, "progress", { stage: "download", loaded: 0, total: 0 });
  const r = await fetch(MODEL_URL);
  if (!r.ok) throw new Error("Could not download the model (HTTP " + r.status + ").");
  const total = Number(r.headers.get("Content-Length")) || 0;
  const reader = r.body.getReader(), parts = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value); loaded += value.length;
    post(id, "progress", { stage: "download", loaded, total });
  }
  const model = new Uint8Array(loaded);
  let at = 0;
  for (const p of parts) { model.set(p, at); at += p.length; }
  post(id, "progress", { stage: "load" });
  ort.env.wasm.wasmPaths = new URL("vendor/", self.location.href).href;
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
  session = await ort.InferenceSession.create(model, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
  return session;
}

/** Decode with EXIF rotation applied, as ImageOps.exif_transpose does. */
async function decode(blob) {
  const bmp = await createImageBitmap(blob, { imageOrientation: "from-image", premultiplyAlpha: "none", colorSpaceConversion: "none" });
  const w = bmp.width, h = bmp.height;
  const ctx = new OffscreenCanvas(w, h).getContext("2d", { willReadFrequently: true });
  ctx.drawImage(bmp, 0, 0);
  bmp.close();
  const rgba = ctx.getImageData(0, 0, w, h).data, rgb = new Uint8Array(w * h * 3);
  for (let i = 0, j = 0; i < rgba.length; i += 4) { rgb[j++] = rgba[i]; rgb[j++] = rgba[i + 1]; rgb[j++] = rgba[i + 2]; }
  return { rgb, w, h };
}

async function runModel(id, blob) {
  const { rgb, w, h } = await decode(blob);
  const sess = await loadSession(id);
  post(id, "progress", { stage: "analyse" });
  const { input, crop } = LeafPixels.prepare(rgb, w, h);
  const size = LeafPixels.IMG_SIZE;
  const out = await sess.run({ image: new ort.Tensor("float32", input, [1, 3, size, size]) });
  return { logits: Array.from(out.logits.data), symptoms: Array.from(out.symptoms.data, (v) => 1 / (1 + Math.exp(-v))),
    cams: out.cams.data, crop };
}

async function fetchJson(url) {
  const r = await fetch(url, { signal: AbortSignal.timeout(WEATHER_TIMEOUT_MS) });
  if (!r.ok) throw new Error("HTTP " + r.status);
  return r.json();
}
async function geocode(place, stateName) {
  try { return LeafCore.pickPlace((await fetchJson(LeafCore.geocodeUrl(place))).results, stateName); } catch (e) { return null; }
}
async function weather(lat, lon) {
  try { return LeafCore.summariseWeather(await fetchJson(LeafCore.weatherUrl(lat, lon))); } catch (e) { return null; }
}

async function heatmapUrl(crop, camMap, jet) {
  const size = LeafPixels.IMG_SIZE;
  const base = new OffscreenCanvas(size, size);
  base.getContext("2d").putImageData(new ImageData(LeafPixels.overlay(crop, camMap, jet), size, size), 0, 0);
  const big = new OffscreenCanvas(320, 320), ctx = big.getContext("2d");
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(base, 0, 0, 320, 320);
  const blob = await big.convertToBlob({ type: "image/png" });
  const bytes = new Uint8Array(await blob.arrayBuffer());
  let s = "";
  for (let i = 0; i < bytes.length; i += 32768) s += String.fromCharCode.apply(null, bytes.subarray(i, i + 32768));
  return "data:image/png;base64," + btoa(s);
}

async function diagnose(id, { blob, plant, state, place, skipWeather }) {
  const kb = await loadCore();
  const model = await runModel(id, blob);
  const stateRec = state ? kb.state(state) : null;
  const geo = !skipWeather && stateRec && place ? await geocode(place, stateRec.name) : null;
  const [st, loc] = kb.locate(state, place, geo);
  const w = !skipWeather && st ? await weather(loc.lat, loc.lon) : null;
  const { result, camClass } = kb.assemble(model.logits, model.symptoms, plant, st, loc, w);
  const n = 7 * 7;
  const camMap = model.cams.slice(camClass * n, (camClass + 1) * n);
  result.heatmap = await heatmapUrl(model.crop, camMap, kb.jet);
  return { result, logits: model.logits, symptoms: model.symptoms };
}

function post(id, kind, payload) { self.postMessage({ id, kind, payload }); }

self.onmessage = async (e) => {
  const { id, type, args } = e.data;
  try {
    if (type === "diagnose") post(id, "done", await diagnose(id, args));
  } catch (err) {
    post(id, "error", { message: String((err && err.message) || err), name: err && err.name });
  }
};

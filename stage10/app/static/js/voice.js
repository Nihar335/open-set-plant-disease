// Voice input/output in the browser. Recording happens only after a user press; audio is captured as 16 kHz mono WAV
// in memory, sent ONLY to this computer's AgroVaidya service, and every buffer, track and audio context is released
// when recording stops or is cancelled. No browser speech service is used (no SpeechRecognition / speechSynthesis).

export const TARGET_RATE = 16000;
export const MAX_SECONDS = 30;

// ---- pure helpers (unit-tested in Node) -----------------------------------------------------------------------
export function downsample(samples, fromRate, toRate = TARGET_RATE) {
  if (fromRate === toRate) return Float32Array.from(samples);
  const ratio = fromRate / toRate;
  const n = Math.floor(samples.length / ratio);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {                                   // box-filter average (anti-alias) then decimate
    const a = Math.floor(i * ratio), b = Math.min(samples.length, Math.floor((i + 1) * ratio));
    let s = 0;
    for (let j = a; j < b; j++) s += samples[j];
    out[i] = b > a ? s / (b - a) : 0;
  }
  return out;
}

export function encodeWav(samples, rate = TARGET_RATE) {
  const buf = new ArrayBuffer(44 + samples.length * 2);
  const v = new DataView(buf);
  const str = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  str(0, "RIFF"); v.setUint32(4, 36 + samples.length * 2, true); str(8, "WAVE"); str(12, "fmt ");
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true); v.setUint32(24, rate, true);
  v.setUint32(28, rate * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true); str(36, "data");
  v.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) {
    const x = Math.max(-1, Math.min(1, samples[i]));
    v.setInt16(44 + i * 2, x < 0 ? x * 0x8000 : x * 0x7fff, true);
  }
  return buf;
}

export function concat(chunks) {
  const n = chunks.reduce((a, c) => a + c.length, 0);
  const out = new Float32Array(n);
  let o = 0;
  for (const c of chunks) { out.set(c, o); o += c.length; }
  return out;
}

// Match a confirmed spoken/typed answer to a question's options (English option labels). Returns the option or null.
// Only exact word overlap with the label, plus a few English yes/no/unsure words; never a guess between options.
const EXTRA = { yes: ["yes", "yeah"], no: ["no"], not_sure: ["not sure", "dont know", "don't know", "unsure"] };   // English only: no invented translations
export function matchOption(text, options) {
  const t = ` ${String(text || "").toLowerCase().replace(/[^\p{L}\p{N}\s']/gu, " ").replace(/\s+/g, " ").trim()} `;
  if (!t.trim()) return null;
  const hits = options.filter((o) => {
    const extra = EXTRA[o.option_id] || [];
    if (extra.some((e) => t.includes(` ${e} `))) return true;
    const words = o.label.toLowerCase().replace(/[^\p{L}\p{N}\s]/gu, " ").split(/\s+/).filter((w) => w.length > 3);
    return words.length > 0 && words.every((w) => t.includes(` ${w} `));
  });
  return hits.length === 1 ? hits[0] : null;                       // ambiguous or nothing -> the user must tap
}

// ---- recorder (browser only) ------------------------------------------------------------------------------------
export async function startRecording(nav = globalThis.navigator) {
  if (!nav || !nav.mediaDevices || !nav.mediaDevices.getUserMedia) {
    const e = new Error("no microphone API"); e.name = "NotFoundError"; throw e;
  }
  const stream = await nav.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true } });
  const ctx = new AudioContext();
  const src = ctx.createMediaStreamSource(stream);
  const chunks = [];
  const node = ctx.createScriptProcessor(4096, 1, 1);
  node.onaudioprocess = (ev) => chunks.push(Float32Array.from(ev.inputBuffer.getChannelData(0)));
  src.connect(node);
  node.connect(ctx.destination);
  const started = Date.now();
  let done = false;
  const release = async () => {
    if (done) return;
    done = true;
    node.onaudioprocess = null;
    try { node.disconnect(); src.disconnect(); } catch (e) { /* already disconnected */ }
    stream.getTracks().forEach((tr) => tr.stop());
    try { await ctx.close(); } catch (e) { /* closed */ }
  };
  return {
    seconds: () => (Date.now() - started) / 1000,
    async stop() {                                                  // -> WAV Blob; buffers released
      const rate = ctx.sampleRate;
      await release();
      const wav = encodeWav(downsample(concat(chunks), rate));
      chunks.length = 0;
      return new Blob([wav], { type: "audio/wav" });
    },
    async cancel() { await release(); chunks.length = 0; },
  };
}

// ---- player for spoken output -----------------------------------------------------------------------------------
export function createPlayer(fetchAudio) {
  let audio = null, url = null, idx = 0, segs = [], state = "stopped", speed = 1.0, onchange = () => {};
  const cleanup = () => { if (audio) { audio.pause(); audio.src = ""; } if (url) URL.revokeObjectURL(url); audio = null; url = null; };
  const emit = () => onchange({ state, index: idx, total: segs.length, speed });
  async function playIndex(i) {
    cleanup();
    if (i >= segs.length) { state = "stopped"; idx = 0; emit(); return; }
    idx = i; state = "loading"; emit();
    try {
      const blob = await fetchAudio(segs[i], speed);
      if (state === "stopped") return;
      url = URL.createObjectURL(blob);
      audio = new Audio(url);
      audio.onended = () => { if (state === "playing") playIndex(idx + 1); };
      state = "playing"; emit();
      await audio.play();
    } catch (e) { state = "error"; emit(); }
  }
  return {
    load(s) { cleanup(); segs = s; idx = 0; state = "stopped"; emit(); },
    play() { if (state === "paused" && audio) { state = "playing"; audio.play(); emit(); } else playIndex(idx); },
    pause() { if (audio && state === "playing") { audio.pause(); state = "paused"; emit(); } },
    stop() { cleanup(); state = "stopped"; idx = 0; emit(); },
    repeat() { playIndex(idx); },
    next() { playIndex(Math.min(idx + 1, segs.length)); },
    setSpeed(s) { speed = s; emit(); },
    onChange(fn) { onchange = fn; },
    get state() { return { state, index: idx, total: segs.length, speed }; },
  };
}

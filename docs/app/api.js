/* LeafScan browser version: answers the page's /api/ requests in the browser instead of on a server.
 * The page itself (stage11/static/index.html) is unchanged apart from where its text comes from.
 */
(function () {
  "use strict";
  const BASE = new URL("app/", document.baseURI).href;
  const MAX_BYTES = 15 * 1024 * 1024;
  const PRIVACY_NOTE = "Your photo is analysed in your browser and is not uploaded. The place name and location coordinates go to a weather service.";
  const PRIVACY_HOW = "The photo is analysed in your browser and is not uploaded.";
  const realFetch = window.fetch.bind(window);
  let worker = null, nextId = 1;
  const pending = new Map();
  let corePromise = null;

  const json = (obj, status = 200) => new Response(JSON.stringify(obj), { status, headers: { "Content-Type": "application/json" } });

  function loadScript(src) {
    return new Promise((ok, bad) => {
      const s = document.createElement("script");
      s.src = src; s.onload = ok; s.onerror = () => bad(new Error("Could not load " + src));
      document.head.appendChild(s);
    });
  }
  function core() {
    corePromise = corePromise || (async () => {
      await loadScript(BASE + "core.js");
      const r = await realFetch(BASE + "kb.json");
      if (!r.ok) throw new Error("HTTP " + r.status);
      return new window.LeafCore.Core(await r.json());
    })();
    return corePromise;
  }

  function status(text) {
    let el = document.getElementById("modelStatus");
    if (!el) {
      const go = document.getElementById("go");
      if (!go) return;
      el = document.createElement("div");
      el.id = "modelStatus"; el.className = "privacy"; el.setAttribute("role", "status");
      go.insertAdjacentElement("afterend", el);
    }
    el.textContent = text;
    el.hidden = !text;
  }
  function onProgress(p) {
    if (p.stage === "download") {
      const mb = (v) => (v / 1e6).toFixed(0);
      status(p.total ? `Downloading the model, first time only: ${mb(p.loaded)} of ${mb(p.total)} MB` : "Downloading the model, first time only…");
    } else if (p.stage === "load") status("Starting the model…");
    else if (p.stage === "analyse") status("Analysing the photo in your browser…");
  }

  function call(type, args) {
    worker = worker || (() => {
      const w = new Worker(BASE + "worker.js");
      w.onmessage = (e) => {
        const { id, kind, payload } = e.data, job = pending.get(id);
        if (!job) return;
        if (kind === "progress") { onProgress(payload); return; }
        pending.delete(id);
        status("");
        if (kind === "done") job.ok(payload); else job.bad(Object.assign(new Error(payload.message), { name: payload.name }));
      };
      w.onerror = (e) => { for (const [id, job] of pending) { pending.delete(id); job.bad(new Error(e.message || "The app stopped working.")); } status(""); };
      return w;
    })();
    return new Promise((ok, bad) => { const id = nextId++; pending.set(id, { ok, bad }); worker.postMessage({ id, type, args }); });
  }

  async function diagnose(fd) {
    const f = fd.get("image");
    const plant = String(fd.get("plant") || "").trim(), state = String(fd.get("state") || "").trim();
    const place = String(fd.get("place") || "").trim().slice(0, 100);
    if (!f || !f.size) return json({ error: "Please upload a leaf photo." }, 400);
    if (!plant) return json({ error: "Please enter the plant name." }, 400);
    if (!state) return json({ error: "Please choose your state." }, 400);
    if (f.size > MAX_BYTES) return json({ error: "Photo is larger than 15 MB." }, 413);
    try {
      return json((await call("diagnose", { blob: f, plant, state, place })).result);
    } catch (e) {
      if (e.name === "InvalidStateError" || /decode|source image/i.test(e.message)) {
        return json({ error: "That file is not a readable image. Please send a JPEG, PNG or WebP photo " +
          "(iPhone HEIC photos: choose 'Most Compatible' in camera settings, or share as JPEG)." }, 400);
      }
      return json({ error: "The analysis could not run in this browser: " + e.message }, 500);
    }
  }

  window.fetch = async function (input, init) {
    const url = typeof input === "string" ? input : input.url;
    if (!url.startsWith("/api/")) return realFetch(input, init);
    const u = new URL(url, location.origin), path = u.pathname;
    if (path === "/api/diagnose") return diagnose(init.body);
    const kb = await core();
    if (path === "/api/meta") return json(kb.meta({ hosted: true, privacy_note: PRIVACY_NOTE, privacy_how: PRIVACY_HOW }));
    if (path === "/api/diseases") return json(kb.diseaseLibrary());
    if (path === "/api/coverage") return json(kb.coverageList());
    if (path === "/api/plant") return json(kb.plantLookup(u.searchParams.get("q") || ""));
    return json({ error: "Not found" }, 404);
  };

  window.LeafScanTest = { call }; // used by webapp/tests/compare.html
})();

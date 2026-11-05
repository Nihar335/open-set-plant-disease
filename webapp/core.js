/* LeafScan browser version: knowledge base, decision rule and report.
 *
 * A direct port of stage11/kb.py, stage11/decision.py, stage11/weather.py and the logic in stage11/engine.py.
 * webapp/tests compares it against the Python code, so change both together.
 * No browser APIs here: it also runs under Node for those tests.
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.LeafCore = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  const MATCH_FLOOR = 0.05, SYMPTOM_SHARPNESS = 6.0, SYMPTOM_CAP = 0.75;
  const SYMPTOM_THRESHOLD = 0.5, SYMPTOM_MATCH_CAP = 0.75, CROP_SUGGEST_MASS = 0.90;
  const CROP_GROUPS = { cucurbits: ["cucurbits", "squash"], squash: ["squash", "cucurbits"],
    chilli: ["chilli", "bell pepper"], "bell pepper": ["bell pepper", "chilli"] };
  const DISCLAIMER = "Chemicals are active ingredients only (no doses). Use only CIBRC-registered products for this crop, " +
    "follow the label and your State Agricultural University spray schedule, and confirm with your KVK.";

  // ------------------------------------------------------------ Python number formatting
  /** Python's round(x, n) and f"{x:.nf}": correctly rounded from the exact binary value, exact ties go to even. */
  function fixed(x, n) {
    const s = x.toFixed(n);
    const exact = Math.abs(x).toFixed(Math.min(100, n + 70));
    const dot = exact.indexOf(".");
    if (/^50*$/.test(exact.slice(dot + 1 + n))) {
      const kept = n === 0 ? exact.slice(0, dot) : exact.slice(0, dot + 1 + n);
      if (Number(kept.slice(-1)) % 2 === 0) return (x < 0 ? "-" : "") + kept;
    }
    return s;
  }
  const pyRound = (x, n) => Number(fixed(x, n));
  const pct = (v) => fixed(100 * v, 0) + "%";
  /** str() of a Python float: 24 prints as 24.0. */
  const floatStr = (x) => (x === null || x === undefined ? "None" : Number.isInteger(x) ? x.toFixed(1) : String(x));

  // ------------------------------------------------------------ difflib.get_close_matches
  function longestMatch(a, b, b2j, alo, ahi, blo, bhi) {
    let besti = alo, bestj = blo, bestsize = 0, j2len = new Map();
    for (let i = alo; i < ahi; i++) {
      const next = new Map();
      for (const j of b2j.get(a[i]) || []) {
        if (j < blo) continue;
        if (j >= bhi) break;
        const k = (j2len.get(j - 1) || 0) + 1;
        next.set(j, k);
        if (k > bestsize) { besti = i - k + 1; bestj = j - k + 1; bestsize = k; }
      }
      j2len = next;
    }
    while (besti > alo && bestj > blo && a[besti - 1] === b[bestj - 1]) { besti--; bestj--; bestsize++; }
    while (besti + bestsize < ahi && bestj + bestsize < bhi && a[besti + bestsize] === b[bestj + bestsize]) bestsize++;
    return [besti, bestj, bestsize];
  }
  function ratio(a, b) {
    const b2j = new Map();
    for (let j = 0; j < b.length; j++) { if (!b2j.has(b[j])) b2j.set(b[j], []); b2j.get(b[j]).push(j); }
    const queue = [[0, a.length, 0, b.length]];
    let matches = 0;
    while (queue.length) {
      const [alo, ahi, blo, bhi] = queue.pop();
      const [i, j, k] = longestMatch(a, b, b2j, alo, ahi, blo, bhi);
      if (k) {
        matches += k;
        if (alo < i && blo < j) queue.push([alo, i, blo, j]);
        if (i + k < ahi && j + k < bhi) queue.push([i + k, ahi, j + k, bhi]);
      }
    }
    return a.length + b.length ? (2 * matches) / (a.length + b.length) : 1;
  }
  function closeMatch(word, possibilities, cutoff) {
    if (word.length >= 200) return null; // far longer than any name
    let best = null, bestScore = -1;
    for (const x of possibilities) {
      const r = ratio(x, word);
      if (r >= cutoff && (r > bestScore || (r === bestScore && x > best))) { best = x; bestScore = r; }
    }
    return best;
  }

  // ------------------------------------------------------------ decision rule (stage11/decision.py)
  function softmax(z) {
    const top = Math.max(...z);
    const e = z.map((v) => Math.exp(v - top));
    const sum = e.reduce((a, b) => a + b, 0);
    return e.map((v) => v / sum);
  }
  const norm2 = (v) => Math.sqrt(v.reduce((a, x) => a + x * x, 0));
  function symptomMatch(sym, profile) {
    const na = norm2(sym) + 1e-8, nb = norm2(profile) + 1e-8;
    let dot = 0;
    for (let i = 0; i < sym.length; i++) dot += (sym[i] / na) * (profile[i] / nb);
    return Math.min(1, Math.max(MATCH_FLOOR, dot));
  }
  function fuse(logits, sym, mask, profiles, T, beta, m0) {
    const pFull = softmax(logits.map((v) => v / T));
    const inMass = pFull.reduce((a, p, i) => a + (mask[i] ? p : 0), 0);
    const pCrop = pFull.map((p, i) => (mask[i] ? p : 0) / (inMass + 1e-12));
    const match = profiles.map((p) => symptomMatch(sym, p));
    const s = pCrop.map((p, i) => p * Math.pow(match[i], beta));
    const sSum = s.reduce((a, b) => a + b, 0);
    const fused = s.map((v) => v / (sSum + 1e-12));
    const q0 = match.map((m, i) => Math.exp(SYMPTOM_SHARPNESS * m) * (mask[i] ? 1 : 0));
    const qSum = q0.reduce((a, b) => a + b, 0);
    const q = q0.map((v) => v / (qSum + 1e-12));
    const fit = m0 > 0 ? Math.min(1, inMass / m0) : 1;
    const conf = fused.map((f, i) => (fit * f + (1 - fit) * q[i]) * (fit + (1 - fit) * SYMPTOM_CAP));
    return { conf, pCrop, pFull, inMass, match, fit };
  }
  function cropMasses(pFull, crops) {
    const mass = new Map();
    crops.forEach((c, i) => mass.set(c, (mass.get(c) || 0) + pFull[i]));
    return mass;
  }
  const bestOf = (mass) => [...mass.entries()].reduce((a, b) => (b[1] > a[1] ? b : a));
  function cropCheck(pFull, crops, mask) {
    const mass = cropMasses(pFull, crops);
    const stated = pFull.reduce((a, p, i) => a + (mask[i] ? p : 0), 0);
    const [best, bestMass] = bestOf(mass);
    const bestIsStated = crops.some((c, i) => c === best && mask[i]);
    return { stated_crop_mass: stated, most_likely_crop: best, most_likely_mass: bestMass,
      conflict: !bestIsStated && bestMass > stated };
  }

  // ------------------------------------------------------------ weather (stage11/weather.py)
  const WEATHER_API = "https://api.open-meteo.com/v1/forecast";
  const GEOCODE_API = "https://geocoding-api.open-meteo.com/v1/search";
  function weatherUrl(lat, lon) {
    const q = new URLSearchParams({ latitude: lat, longitude: lon, timezone: "Asia/Kolkata",
      current: "temperature_2m,relative_humidity_2m,precipitation,weather_code",
      daily: "temperature_2m_max,temperature_2m_min,precipitation_sum,relative_humidity_2m_mean",
      past_days: 7, forecast_days: 3 });
    return `${WEATHER_API}?${q}`;
  }
  function summariseWeather(raw) {
    const d = raw.daily, range = (a) => a.slice(0, 7);
    const tmax = range(d.temperature_2m_max), tmin = range(d.temperature_2m_min);
    const tmean = tmax.map((a, i) => (a + tmin[i]) / 2);
    const rh = range(d.relative_humidity_2m_mean).filter((v) => v !== null);
    const rain = range(d.precipitation_sum);
    const sum = (a) => a.reduce((x, v) => x + (v || 0), 0);
    return {
      current_temp_c: raw.current.temperature_2m,
      current_humidity_pct: raw.current.relative_humidity_2m,
      current_precip_mm: raw.current.precipitation,
      week_mean_temp_c: pyRound(sum(tmean) / tmean.length, 1),
      week_mean_humidity_pct: rh.length ? pyRound(sum(rh) / rh.length, 1) : null,
      week_rain_mm: pyRound(sum(rain), 1),
      week_rain_days: rain.filter((v) => (v || 0) >= 1).length,
      forecast_rain_mm_3d: pyRound(sum(d.precipitation_sum.slice(7)), 1),
      daily: d.time.map((t, i) => ({ date: t, tmax: d.temperature_2m_max[i], tmin: d.temperature_2m_min[i],
        rain_mm: d.precipitation_sum[i], rh: d.relative_humidity_2m_mean[i] })),
      source: "Open-Meteo (open-meteo.com)",
    };
  }
  function geocodeUrl(place) {
    const q = new URLSearchParams({ name: place, count: 10, language: "en", format: "json", countryCode: "IN" });
    return `${GEOCODE_API}?${q}`;
  }
  function pickPlace(results, stateName) {
    const want = stateName.toLowerCase();
    const inState = (results || []).filter((x) => (x.admin1 || "").toLowerCase() === want);
    if (!inState.length) return null;
    const best = inState.reduce((a, b) => ((b.population || 0) > (a.population || 0) ? b : a));
    return { name: best.name, district: best.admin2 ?? null, state: best.admin1 ?? null,
      lat: pyRound(best.latitude, 4), lon: pyRound(best.longitude, 4) };
  }
  function favourability(spec, w) {
    if (!spec || !Object.keys(spec).length || !w) return null;
    const [lo, hi] = spec.temp;
    const t = w.week_mean_temp_c;
    const gap = lo <= t && t <= hi ? 0 : Math.min(Math.abs(t - lo), Math.abs(t - hi));
    const sTemp = Math.max(0, 1 - gap / 6);
    const hasWeekRh = w.week_mean_humidity_pct !== null;
    const rh = hasWeekRh ? w.week_mean_humidity_pct : w.current_humidity_pct;
    const sRh = { high: rh >= 80 ? 1 : rh >= 60 ? 0.5 : 0.1, moderate: rh >= 60 && rh <= 85 ? 1 : 0.5,
      low: rh < 60 ? 1 : rh < 75 ? 0.5 : 0.2, any: 1 }[spec.humidity];
    const wet = w.week_rain_days + (w.forecast_rain_mm_3d >= 2 ? 1 : 0);
    const sRain = { wet: wet >= 2 ? 1 : wet === 1 ? 0.6 : 0.2, dry: wet === 0 ? 1 : wet <= 2 ? 0.6 : 0.2, any: 1 }[spec.rain];
    const score = (sTemp + sRh + sRain) / 3;
    const level = score >= 0.75 ? "high" : score >= 0.45 ? "moderate" : "low";
    const rhText = hasWeekRh ? floatStr(rh) : String(rh);
    const explanation = `favours ${lo}–${hi} °C (last week ${floatStr(t)} °C), ${spec.humidity} humidity (last week ${rhText}%), ` +
      `${spec.rain} weather (${w.week_rain_days} rain days last week)`;
    return { score: pyRound(score, 2), level, explanation };
  }

  // ------------------------------------------------------------ knowledge base (stage11/kb.py)
  const normText = (t) => t.toLowerCase().replace(/[^a-z0-9 ]+/g, " ").trim();
  const words = (t) => t.split(/\s+/).filter(Boolean);

  class Core {
    constructor(data) {
      Object.assign(this, data);
      this.symptomIds = this.symptoms.map((s) => s.id);
      this.byKey = new Map(this.diseases.map((d) => [d.key, d]));
      this.modelClasses = this.diseases.filter((d) => d.model_class);
      this.classCrops = this.modelClasses.map((d) => d.crop);
      this.profiles = this.modelClasses.map((d) => this.profile(d.key));
      this.plantNames = new Map();
      for (const p of this.plants) {
        for (const n of [p.common_name, p.scientific_name, ...(p.aliases || [])]) {
          const k = normText(n);
          if (!this.plantNames.has(k)) this.plantNames.set(k, p);
        }
        const short = words(normText(p.scientific_name)).slice(0, 2).join(" ");
        if (!this.plantNames.has(short)) this.plantNames.set(short, p);
      }
    }
    profile(key) {
      const v = new Array(this.symptomIds.length).fill(0);
      for (const [sid, w] of Object.entries(this.byKey.get(key).symptoms)) v[this.symptomIds.indexOf(sid)] = w;
      return v;
    }
    resolvePlant(text) {
      const q = normText(text || "");
      if (!q) return [null, null];
      if (this.plantNames.has(q)) return [this.plantNames.get(q), "exact"];
      if (q.endsWith("s") && this.plantNames.has(q.slice(0, -1))) return [this.plantNames.get(q.slice(0, -1)), "exact"];
      const close = closeMatch(q, [...this.plantNames.keys()], 0.8);
      return close ? [this.plantNames.get(close), "fuzzy"] : [null, null];
    }
    cropKeys(crop) { return !crop ? [] : CROP_GROUPS[crop] || [crop]; }
    modelCoverage(crop) {
      const own = crop ? this.modelClasses.filter((d) => d.crop === crop) : [];
      const keys = this.cropKeys(crop);
      const rel = crop ? this.modelClasses.filter((d) => keys.includes(d.crop) && d.crop !== crop) : [];
      const healthy = own.filter((d) => d.pathogen_type === "none").map((d) => d.key);
      const diseases = own.filter((d) => d.pathogen_type !== "none").map((d) => d.key);
      const level = healthy.length && diseases.length ? "diagnosable" : own.length ? "limited" : rel.length ? "related" : "none";
      return { level, classes: own.map((d) => d.key), healthy, diseases, related_classes: rel.map((d) => d.key),
        related_crops: [...new Set(rel.map((d) => d.crop))].sort() };
    }
    state(name) {
      const q = normText(name || "");
      if (!q) return null;
      for (const s of this.states) if (normText(s.name) === q) return s;
      const hits = this.states.filter((s) => ` ${normText(s.name)} `.includes(` ${q} `) || normText(s.capital) === q);
      if (hits.length === 1) return hits[0];
      const close = closeMatch(q, this.states.map((s) => normText(s.name)), 0.85);
      return close ? this.states.find((s) => normText(s.name) === close) : null;
    }

    // ---------------------------------------------------------- the data behind the page's other requests
    meta(extra) {
      return { states: this.states.map((s) => s.name),
        plants: this.plants.map((p) => {
          const level = this.modelCoverage(p.model_crop).level;
          return { name: p.common_name, scientific_name: p.scientific_name, supported: level === "diagnosable", model_coverage: level };
        }),
        supported_crops: [...new Set(this.modelClasses.map((d) => d.crop))].sort(),
        threshold: this.threshold, ...extra };
    }
    diseaseLibrary() {
      const sym = new Map(this.symptoms.map((s) => [s.id, s.name]));
      const names = new Map(this.diseases.map((d) => [d.key, d.name]));
      const cmp = (a, b) => (a < b ? -1 : a > b ? 1 : 0);
      return this.diseases.filter((d) => d.pathogen_type !== "none").map((d) => ({
        key: d.key, crop: d.crop, name: d.name, pathogen: d.pathogen, pathogen_type: d.pathogen_type,
        in_model: d.model_class !== null && d.model_class !== undefined,
        symptoms: Object.entries(d.symptoms).sort((x, y) => y[1] - x[1]).map(([k]) => sym.get(k) ?? k),
        look_alikes: (d.look_alikes || []).map((k) => names.get(k) ?? k),
        weather: d.weather ?? null,
        tests: d.tests.filter((t) => this.tests[t]).map((t) => this.tests[t].name),
      })).sort((a, b) => cmp(a.crop, b.crop) || cmp(a.name, b.name));
    }
    coverageList() {
      return [...new Set(this.modelClasses.map((d) => d.crop))].map((crop) => {
        const cov = this.modelCoverage(crop);
        return { crop, level: cov.level,
          plant_names: this.plants.filter((p) => p.model_crop === crop).map((p) => p.common_name),
          diseases: this.modelClasses.filter((d) => d.crop === crop && d.pathogen_type !== "none").map((d) => d.name),
          healthy: cov.healthy.length > 0 };
      });
    }
    plantLookup(q) {
      const [p, match] = this.resolvePlant(q);
      if (!p) return { found: false };
      const level = this.modelCoverage(p.model_crop).level;
      return { found: true, match, common_name: p.common_name, scientific_name: p.scientific_name, family: p.family,
        supported: level === "diagnosable", model_coverage: level };
    }

    // ---------------------------------------------------------- engine helpers
    symptomList(probs) {
      return this.symptoms.map((s, i) => ({ id: s.id, name: s.name, description: s.description, probability: pyRound(probs[i], 3) }))
        .sort((a, b) => b.probability - a.probability);
    }
    seen(key, probs) {
      const seen = [], missing = [];
      for (const [sid, w] of Object.entries(this.byKey.get(key).symptoms)) {
        const i = this.symptomIds.indexOf(sid), p = probs[i];
        (p >= SYMPTOM_THRESHOLD ? seen : missing).push({ name: this.symptoms[i].name, weight: w, probability: pyRound(p, 2) });
      }
      return [seen, missing];
    }
    whereToTest(state, crop) {
      const res = this.resources;
      const out = [...(state ? (res.states[state.name] || {}).labs || [] : [])];
      const inst = res.crop_institutes[crop];
      const acronym = inst && inst.includes("(") ? inst.split("(").pop().split(")")[0] : null;
      if (inst && !out.some((lab) => acronym && lab.includes(acronym))) out.push(inst);
      out.push(res.national.kvk, res.national.helpline);
      return out;
    }
    card(key, conf, probs, w, state, basis, imageProb, match) {
      const d = this.byKey.get(key);
      const [seen, missing] = this.seen(key, probs);
      const sol = this.solutions[key] || { cultural: d.management ? [d.management] : [], biological: [], chemical: [] };
      const schedule = state ? ((this.resources.states[state.name] || {}).schedules || {})[d.crop] ?? null : null;
      const healthy = d.pathogen_type === "none";
      return { key, name: d.name, crop: d.crop, confidence: pyRound(conf, 4),
        image_probability: imageProb === null ? null : pyRound(imageProb, 4),
        symptom_match: match === null ? null : pyRound(match, 3),
        basis, healthy, pathogen: d.pathogen ?? null, pathogen_type: d.pathogen_type,
        symptoms_seen: seen, symptoms_not_seen: missing, weather_risk: favourability(d.weather, w),
        look_alikes: (d.look_alikes || []).map((k) => this.byKey.get(k).name),
        tests: d.tests.map((t) => this.tests[t]),
        where_to_test: healthy ? [] : this.whereToTest(state, d.crop),
        solution: { cultural: sol.cultural || [], biological: sol.biological || [], chemical: sol.chemical || [], state_schedule: schedule } };
    }
    /** Returns the state record and the location; geo is the result of geocoding the place, or null. */
    locate(stateName, place, geo) {
      const state = stateName ? this.state(stateName) : null;
      if (!state) return [null, null];
      if (geo) {
        return [state, { label: geo.name + (geo.district ? `, ${geo.district}` : "") + `, ${state.name}`, lat: geo.lat, lon: geo.lon, precision: "place" }];
      }
      return [state, { label: `${state.capital} (state capital), ${state.name}`, lat: state.lat, lon: state.lon,
        precision: "state_capital", place_not_found: Boolean(place) }];
    }
    suggestCrop(plant, plantText, lg, notes) {
      const pFull = softmax(lg.map((v) => v / this.calibration.temperature));
      const mass = cropMasses(pFull, this.classCrops);
      const [best, bestMass] = bestOf(mass);
      const stated = plant ? this.cropKeys(plant.model_crop) : [];
      if (stated.includes(best) || bestMass < CROP_SUGGEST_MASS) return null;
      const target = this.plants.find((p) => p.model_crop === best && p.common_name.toLowerCase() === best) ||
        this.plants.find((p) => p.model_crop === best);
      if (!target) return null;
      const top = this.modelClasses[pFull.indexOf(Math.max(...pFull))];
      const entered = plant ? plant.common_name : plantText;
      notes.push(`Plant check: you entered ${entered}, but the photo looks more like ${target.common_name} ` +
        `(${fixed(100 * bestMass, 0)}% of the image probability; closest trained disease: ${top.name}). ` +
        `Your plant name was kept and no disease is named, because the model knows only ` +
        `${this.modelClasses.length} conditions of ${mass.size} crops and photos of plants or diseases it was never ` +
        `trained on (e.g. downy mildew) can match a trained crop just as confidently. If the plant really ` +
        `is ${target.common_name}, diagnose it again with that name.`);
      return { entered: plantText, suggested: target.common_name, mass: pyRound(bestMass, 4), closest_disease: top.name,
        stated_mass: pyRound(stated.reduce((a, k) => a + (mass.get(k) || 0), 0), 4) };
    }

    // ---------------------------------------------------------- the decision (Engine._diagnose)
    /** lg: disease logits; sym: symptom probabilities; w: weather summary or null. The heat map is filled in by the caller. */
    assemble(lg, sym, plantText, state, loc, w) {
      const [plant, matchType] = this.resolvePlant(plantText);
      const camCls = lg.indexOf(Math.max(...lg));
      const notes = [];
      const suggestion = this.suggestCrop(plant, plantText, lg, notes);
      const cropKeys = plant ? this.cropKeys(plant.model_crop) : [];
      const inCrop = this.classCrops.map((c) => cropKeys.includes(c));
      const coverage = this.modelCoverage(plant ? plant.model_crop : null);
      let check = null, fit = 1, basis, ranked;

      if (inCrop.some(Boolean)) {
        basis = "image";
        const cal = this.calibration;
        const out = fuse(lg, sym, inCrop, this.profiles, cal.temperature, cal.beta, cal.m0);
        check = cropCheck(out.pFull, this.classCrops, inCrop);
        fit = out.fit;
        const basisTxt = fit >= 1 ? "image + symptoms" : "symptoms + image";
        const order = out.conf.map((_, i) => i).filter((i) => inCrop[i]).sort((a, b) => out.conf[b] - out.conf[a]);
        ranked = order.map((i) => this.card(this.modelClasses[i].key, out.conf[i], sym, w, state, basisTxt, out.pCrop[i], out.match[i]));
        if (check.conflict && !suggestion) {
          notes.push(`Crop check: the photo looks more like ${check.most_likely_crop} ` +
            `(${fixed(100 * check.most_likely_mass, 0)}% of the image probability) than ` +
            `${plant.common_name} (${fixed(100 * check.stated_crop_mass, 0)}%). Please check the plant ` +
            "name; no disease is named while the two disagree.");
        } else if (fit < 1) {
          const other = this.modelClasses[out.pFull.indexOf(Math.max(...out.pFull))];
          notes.push(`Only ${fixed(100 * out.inMass, 0)}% of the image model's probability falls on ${plant.common_name} ` +
            `diseases it knows (the photo looks most like ${other.crop} — ${other.name}). ` +
            "So the ranking relies more on the symptoms detected in the photo, matched against the disease database, " +
            "so the result is reported as leads.");
        }
        for (const d of this.diseases) {
          if (!d.model_class && cropKeys.includes(d.crop)) {
            const m = symptomMatch(sym, this.profile(d.key));
            if (m >= 0.6) ranked.push(this.card(d.key, m * SYMPTOM_MATCH_CAP * 0.5, sym, w, state, "symptom match only", null, m));
          }
        }
        ranked.sort((a, b) => b.confidence - a.confidence);
        if (coverage.level === "limited") {
          const kinds = !coverage.diseases.length ? "healthy" : "diseased (" + coverage.diseases.map((k) => this.byKey.get(k).name).join(", ") + ")";
          notes.push(`Limited model coverage: the image model has seen only ${kinds} ${plant.common_name} ` +
            "leaves, so it cannot tell other conditions apart and no result is named. The list below gives leads only.");
        } else if (coverage.level === "related") {
          notes.push(`No ${plant.common_name} photos were used to train the image model. The diseases ` +
            `below come from a related crop (${coverage.related_crops.join(", ")}) and are leads only; no result is named.`);
        }
      } else {
        basis = "symptom_match";
        const crop = plant ? plant.model_crop : null;
        const keys = this.diseases.filter((d) => d.pathogen_type !== "none" &&
          ((crop && this.cropKeys(crop).includes(d.crop)) || d.crop === "any" || !plant)).map((d) => d.key);
        notes.push((plant ? `The image model was not trained on ${plant.common_name} photos. ` : "This plant is not in the botanical database. ") +
          "Symptoms detected in the photo were matched against the disease database, so the result is reported as leads.");
        const m = keys.map((k) => symptomMatch(sym, this.profile(k)));
        const raw = m.map((v) => Math.exp(6 * v));
        const total = raw.reduce((a, b) => a + b, 0);
        const healthyP = sym[this.symptomIds.indexOf("no_visible_symptoms")];
        const s = raw.map((v) => (v / total) * (1 - healthyP) * SYMPTOM_MATCH_CAP);
        ranked = keys.map((k, i) => this.card(k, s[i], sym, w, state, "symptom match only", null, m[i]))
          .sort((a, b) => b.confidence - a.confidence);
      }

      const top = ranked[0];
      const reasons = [];
      if (!plant) reasons.push("plant_not_in_database");
      else if (coverage.level !== "diagnosable") reasons.push(`model_coverage_${coverage.level}`);
      if (check && check.conflict) reasons.push("crop_conflict");
      else if (basis === "image" && fit < 1) reasons.push("low_photo_fit");
      if (top.confidence < this.threshold) reasons.push("below_threshold");

      let verdict, shown;
      const conflict = Boolean(check && check.conflict);
      if (basis === "image" && top.basis !== "symptom match only" && top.confidence >= this.threshold &&
          coverage.level === "diagnosable" && !conflict) {
        verdict = top.healthy ? "healthy" : "predicted";
        shown = [top];
        const fe = ((this.field_eval[plant.model_crop] || {}).field) || {};
        const acc = (fe.v3 || {}).named_acc ?? null, n = fe.n || 0;
        if (acc !== null && n >= 10 && acc < 0.8) {
          notes.push(`Field reliability warning: on ${n} independent field photos of ${plant.common_name}, ` +
            `only ${fixed(100 * acc, 0)}% of named results were correct. Confirm with the tests below before treating.`);
        }
      } else {
        verdict = "unknown";
        shown = ranked.filter((c) => !c.healthy).slice(0, 3);
        const healthy = ranked.find((c) => c.healthy);
        if (healthy && healthy.confidence >= 0.3) notes.push(`The model also gives ${fixed(100 * healthy.confidence, 0)}% to a healthy leaf.`);
        if (!shown.length) notes.push("No disease of this crop with matching symptoms is in the database; please show the leaf to your KVK or State Agricultural University.");
      }

      const all = this.symptomList(sym);
      const result = {
        plant: !plant ? null : { query: plantText, match: matchType, common_name: plant.common_name,
          scientific_name: plant.scientific_name, family: plant.family,
          model_supported: coverage.level === "diagnosable", model_coverage: coverage.level },
        crop_suggestion: suggestion,
        crop_check: check,
        state: !state ? null : { name: state.name },
        location: loc,
        weather: w,
        weather_note: w ? null : state ? "Weather service unreachable; weather was not used." : "State not recognised; weather was not used.",
        symptoms_detected: all.filter((s) => s.probability >= SYMPTOM_THRESHOLD && s.id !== "no_visible_symptoms"),
        symptoms_all: all,
        heatmap: "",
        heatmap_class: { name: this.modelClasses[camCls].name, crop: this.modelClasses[camCls].crop },
        verdict,
        unknown_reasons: verdict === "unknown" ? reasons : [],
        threshold: this.threshold,
        basis,
        candidates: shown,
        notes,
        model: { checkpoint_epoch: this.checkpoint_epoch, temperature: this.calibration.temperature, beta: this.calibration.beta, m0: this.calibration.m0 },
        disclaimer: DISCLAIMER,
      };
      result.report = buildReport(result, plantText);
      return { result, camClass: camCls };
    }
  }

  // ------------------------------------------------------------ the farmer report (fixed format)
  function weatherLine(r) {
    const w = r.weather, loc = r.location;
    if (!w) return r.weather_note;
    return `Weather at ${loc.label}: last 7 days ${floatStr(w.week_mean_temp_c)} °C, humidity ` +
      `${floatStr(w.week_mean_humidity_pct)}%, ${w.week_rain_days} rain day(s); now ${floatStr(w.current_temp_c)} °C.`;
  }
  const WHY = { low_photo_fit: "the photo does not closely match the trained photos of this crop",
    crop_conflict: "the photo looks more like another crop than the one named",
    model_coverage_limited: "the image model covers only part of this crop's conditions",
    model_coverage_related: "the image model was trained on a related crop, not this one",
    model_coverage_none: "the image model was not trained on this crop",
    plant_not_in_database: "the plant is not in the botanical database" };
  function buildReport(r, plantText) {
    const p = r.plant, sg = r.crop_suggestion, c = r.candidates, f = {};
    f["Plant Name"] = p ? p.common_name : `${plantText} (not in the botanical database)`;
    if (sg) f["Plant Name"] += ` (as entered; the photo looks more like ${sg.suggested} at ${pct(sg.mass)} — if it is ${sg.suggested}, diagnose again with that name)`;
    f["Biological Name of the plant"] = p ? `${p.scientific_name} (family ${p.family})` : "Unknown";
    f["Symptoms found"] = r.symptoms_detected.map((s) => `${s.name} (${pct(s.probability)})`).join(", ") || "No clear disease symptoms detected";
    if (r.verdict === "predicted") {
      const d = c[0];
      f["Predicted disease"] = `${d.name} — ${pct(d.confidence)} confidence. Cause: ${d.pathogen}. ` +
        (d.weather_risk ? `Weather risk: ${d.weather_risk.level}. ` : "") + weatherLine(r);
    } else if (r.verdict === "healthy") {
      f["Predicted disease"] = `None — the leaf looks healthy (${pct(c[0].confidence)}). ` + weatherLine(r);
    } else {
      const why = { below_threshold: `no disease reached the ${pct(r.threshold)} score threshold (a heuristic score, not a probability of being correct)`, ...WHY };
      const reasons = r.unknown_reasons.filter((k) => why[k]).map((k) => why[k]).join("; ") || "the evidence is insufficient";
      const lines = c.length
        ? [`Possibly ${c[0].name} — ${pct(c[0].confidence)} (not confirmed: ${reasons}). Closest matches (leads to confirm, not a diagnosis):`]
        : [`Unknown disease — ${reasons}. No matching disease in the database (not a diagnosis).`];
      c.forEach((d, i) => lines.push(`  ${i + 1}. ${d.name} — ${pct(d.confidence)} (${d.basis}; cause: ${d.pathogen}` +
        (d.weather_risk ? `; weather risk ${d.weather_risk.level}` : "") + ")"));
      lines.push("  " + weatherLine(r));
      f["Predicted disease"] = lines.join("\n");
    }
    const unknown = r.verdict === "unknown", tests = [], solutions = [];
    for (const d of unknown ? c : c.slice(0, 1)) {
      if (d.healthy) continue;
      const head = unknown ? `${d.name}: ` : "";
      tests.push(head + d.tests.map((t) => `${t.name} (${t.level.replace(/_/g, " ")}, ${t.time})`).join("; "));
      const s = d.solution, parts = [];
      if (s.cultural.length) parts.push("Cultural: " + s.cultural.join("; "));
      if (s.biological.length) parts.push("Biological: " + s.biological.join(", "));
      if (s.chemical.length) parts.push("Chemical (active ingredients): " + s.chemical.join(", "));
      if (s.state_schedule) parts.push("Follow: " + s.state_schedule);
      solutions.push(head + parts.join(" | "));
    }
    const where = (c.find((d) => d.where_to_test.length) || { where_to_test: [] }).where_to_test;
    if (where.length) tests.push("Where to test: " + where.join("; "));
    f["Test to confirm"] = tests.length ? tests.join("\n") : unknown
      ? "Show the leaf to your KVK or State Agricultural University plant clinic." : "No test needed — keep monitoring weekly.";
    f["Solution for the Predicted disease"] = solutions.length ? solutions.join("\n") + "\n" + DISCLAIMER : unknown
      ? "No treatment until the cause is confirmed." : "No treatment needed. Keep monitoring weekly.";
    return { fields: f, text: Object.entries(f).map(([k, v]) => `${k} : ${v}`).join("\n") };
  }

  return { Core, fixed, pyRound, closeMatch, ratio, weatherUrl, geocodeUrl, summariseWeather, pickPlace, favourability };
});

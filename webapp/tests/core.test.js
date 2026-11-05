// Compares webapp/core.js with the Python engine on generated cases.
// Run: FIXTURES=<fixtures.json> KB=<docs/app/kb.json> node webapp/tests/core.test.js
"use strict";
const fs = require("fs");
const test = require("node:test");
const assert = require("node:assert");
const { Core, pyRound, fixed } = require("../core.js");

const fixtures = JSON.parse(fs.readFileSync(process.env.FIXTURES, "utf8"));
const core = new Core(JSON.parse(fs.readFileSync(process.env.KB, "utf8")));

// Python computes these scores in float32 and the port in float64, so a rounded number can differ in its last digit.
const NUM_TOL = 2e-3;
const numberRe = /-?\d+(?:\.\d+)?/g;
function sameText(a, b) {
  if (a === b) return true;
  const na = a.match(numberRe) || [], nb = b.match(numberRe) || [];
  if (na.length !== nb.length || a.replace(numberRe, "#") !== b.replace(numberRe, "#")) return false;
  return na.every((v, i) => Math.abs(Number(v) - Number(nb[i])) <= 1 + 1e-9);
}
function diffs(a, b, path, out) {
  if (typeof a === "number" && typeof b === "number") {
    if (Math.abs(a - b) > NUM_TOL) out.push(`${path}: ${a} vs ${b}`);
    else if (a !== b) out.rounded++;
    return;
  }
  if (typeof a === "string" && typeof b === "string") {
    if (a !== b) { if (sameText(a, b)) out.textRounded++; else out.push(`${path}: "${a}" vs "${b}"`); }
    return;
  }
  if (a === null || b === null || typeof a !== typeof b) { if (a !== b) out.push(`${path}: ${JSON.stringify(a)} vs ${JSON.stringify(b)}`); return; }
  if (Array.isArray(a)) {
    if (!Array.isArray(b) || a.length !== b.length) { out.push(`${path}: length ${a.length} vs ${b && b.length}`); return; }
    a.forEach((v, i) => diffs(v, b[i], `${path}[${i}]`, out));
    return;
  }
  if (typeof a === "object") {
    const ka = Object.keys(a), kb = Object.keys(b);
    if (ka.join() !== kb.join()) { out.push(`${path}: keys ${ka} vs ${kb}`); return; }
    for (const k of ka) diffs(a[k], b[k], `${path}.${k}`, out);
    return;
  }
  if (a !== b) out.push(`${path}: ${a} vs ${b}`);
}

test("image resize, crop, normalisation and heat map match Pillow, torchvision and OpenCV", () => {
  const P = require("../pixels.js");
  const bytes = (s) => new Uint8Array(Buffer.from(s, "base64"));
  const floats = (s) => { const b = Buffer.from(s, "base64"); return new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.length)); };
  const jet = core.jet;
  for (const f of fixtures.pixels) {
    const { input, crop } = P.prepare(bytes(f.rgb), f.w, f.h);
    assert.deepStrictEqual(Buffer.from(crop), Buffer.from(bytes(f.crop)), `crop ${f.w}x${f.h}`);
    const want = floats(f.input);
    let worst = 0;
    for (let i = 0; i < want.length; i++) worst = Math.max(worst, Math.abs(input[i] - want[i]));
    assert.ok(worst < 1e-5, `normalised input ${f.w}x${f.h}: ${worst}`);
    const over = P.overlay(crop, floats(f.cam), jet), wantOver = bytes(f.overlay);
    let bad = 0;
    for (let i = 0; i < 224 * 224; i++) for (let c = 0; c < 3; c++) if (Math.abs(over[i * 4 + c] - wantOver[i * 3 + c]) > 1) bad++;
    assert.ok(bad < 224 * 224 * 3 * 0.002, `overlay ${f.w}x${f.h}: ${bad} values off by more than 1`);
  }
});

test("number formatting follows Python", () => {
  assert.strictEqual(fixed(12.5, 0), "12"); assert.strictEqual(fixed(13.5, 0), "14");
  assert.strictEqual(pyRound(0.125, 2), 0.12); assert.strictEqual(pyRound(0.375, 2), 0.38);
  assert.strictEqual(pyRound(2.675, 2), 2.67); assert.strictEqual(fixed(0.5, 0), "0");
});

test("plant names resolve as in Python", () => {
  for (const c of fixtures.extra.resolve) {
    const [p, m] = core.resolvePlant(c.q);
    assert.deepStrictEqual([p ? p.common_name : null, m], [c.name, c.match], `query ${JSON.stringify(c.q)}`);
  }
});

test("states resolve as in Python", () => {
  for (const c of fixtures.extra.states) {
    const s = core.state(c.q);
    assert.strictEqual(s ? s.name : null, c.name, `query ${JSON.stringify(c.q)}`);
  }
});

test("model coverage matches", () => {
  for (const [crop, cov] of Object.entries(fixtures.extra.coverage)) assert.deepStrictEqual(core.modelCoverage(crop), cov);
});

test("weather risk matches", () => {
  const { favourability } = require("../core.js");
  for (const c of fixtures.extra.favour) assert.deepStrictEqual(favourability(c.spec, c.w), c.out);
});

test("every generated diagnosis matches", () => {
  const out = []; out.rounded = 0; out.textRounded = 0;
  let verdictsSame = 0;
  for (const [i, c] of fixtures.cases.entries()) {
    const [st, loc] = core.locate(c.state, c.place, c.geo);
    const { result } = core.assemble(c.logits, c.sym, c.plant, st, loc, st ? c.weather : null); // no state: no weather lookup
    delete result.heatmap;
    if (result.verdict === c.expected.verdict) verdictsSame++;
    diffs(result, c.expected, `case ${i} (${c.plant})`, out);
  }
  console.log(`cases ${fixtures.cases.length} | same verdict ${verdictsSame} | last-digit number differences ${out.rounded} | text differing only by last digit ${out.textRounded} | real differences ${out.length}`);
  assert.strictEqual(out.length, 0, out.slice(0, 15).join("\n"));
  assert.strictEqual(verdictsSame, fixtures.cases.length);
});

/* LeafScan browser version: image preparation and heat map, matching the Python pipeline.
 *   Resize(256) with Pillow's bilinear filter, CenterCrop(224), ToTensor, Normalize  (stage11/model.py EVAL_TF)
 *   heat map: cv2 cubic resize of the 7x7 map, JET colours, 0.55 photo + 0.45 colour  (stage11/engine.py _gradcam)
 * No browser APIs here: it also runs under Node for the tests.
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.LeafPixels = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";
  const IMG_SIZE = 224, RESIZE = 256;
  const MEAN = [0.485, 0.456, 0.406], STD = [0.229, 0.224, 0.225];
  const PRECISION = 22; // Pillow: 32 - 8 - 2 bits of fixed point

  /** Weights for one axis, as in Pillow's precompute_coeffs + normalize_coeffs_8bpc with the bilinear filter. */
  function coeffs(inSize, outSize) {
    const scale = inSize / outSize, filterscale = Math.max(scale, 1), support = filterscale;
    const out = [];
    for (let xx = 0; xx < outSize; xx++) {
      const center = (xx + 0.5) * scale;
      const xmin = Math.max(0, Math.trunc(center - support + 0.5));
      const xmax = Math.min(inSize, Math.trunc(center + support + 0.5));
      const k = [];
      let ww = 0;
      for (let x = xmin; x < xmax; x++) {
        const d = Math.abs((x - center + 0.5) / filterscale), w = d < 1 ? 1 - d : 0;
        k.push(w); ww += w;
      }
      const fixed = new Int32Array(k.length);
      for (let i = 0; i < k.length; i++) fixed[i] = Math.trunc(0.5 + (k[i] / ww) * (1 << PRECISION));
      out.push({ xmin, k: fixed });
    }
    return out;
  }

  /** rgb: Uint8 array of 3 channels per pixel. Returns a resized copy. */
  function resizeBilinear(rgb, w, h, ow, oh) {
    const half = 1 << (PRECISION - 1);
    const clip = (v) => { v >>= PRECISION; return v < 0 ? 0 : v > 255 ? 255 : v; };
    let src = rgb, srcW = w;
    if (ow !== w) {
      const cx = coeffs(w, ow), tmp = new Uint8Array(ow * h * 3);
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < ow; x++) {
          const { xmin, k } = cx[x];
          let r = half, g = half, b = half, p = (y * w + xmin) * 3;
          for (let i = 0; i < k.length; i++) { r += rgb[p++] * k[i]; g += rgb[p++] * k[i]; b += rgb[p++] * k[i]; }
          const q = (y * ow + x) * 3;
          tmp[q] = clip(r); tmp[q + 1] = clip(g); tmp[q + 2] = clip(b);
        }
      }
      src = tmp; srcW = ow;
    }
    if (oh === h) return src;
    const cy = coeffs(h, oh), out = new Uint8Array(ow * oh * 3);
    for (let y = 0; y < oh; y++) {
      const { xmin, k } = cy[y];
      for (let x = 0; x < ow * 3; x++) {
        let v = half;
        for (let i = 0; i < k.length; i++) v += src[((xmin + i) * srcW * 3) + x] * k[i];
        out[y * ow * 3 + x] = clip(v);
      }
    }
    return out;
  }

  /** Python's round(x) for x.0 and x.5 values. */
  function roundHalfEven(x) {
    const f = Math.floor(x), d = x - f;
    return d < 0.5 ? f : d > 0.5 ? f + 1 : f % 2 === 0 ? f : f + 1;
  }

  /** Whole photo -> { input: Float32Array(1,3,224,224), crop: Uint8Array(224*224*3) }. */
  function prepare(rgb, w, h) {
    let ow, oh;
    if (w <= h) { ow = RESIZE; oh = Math.trunc((RESIZE * h) / w); } else { oh = RESIZE; ow = Math.trunc((RESIZE * w) / h); }
    const small = resizeBilinear(rgb, w, h, ow, oh);
    const top = roundHalfEven((oh - IMG_SIZE) / 2), left = roundHalfEven((ow - IMG_SIZE) / 2);
    const crop = new Uint8Array(IMG_SIZE * IMG_SIZE * 3), input = new Float32Array(3 * IMG_SIZE * IMG_SIZE);
    const area = IMG_SIZE * IMG_SIZE;
    for (let y = 0; y < IMG_SIZE; y++) {
      for (let x = 0; x < IMG_SIZE; x++) {
        const s = ((y + top) * ow + x + left) * 3, i = y * IMG_SIZE + x;
        for (let c = 0; c < 3; c++) {
          const v = small[s + c];
          crop[i * 3 + c] = v;
          input[c * area + i] = (v / 255 - MEAN[c]) / STD[c];
        }
      }
    }
    return { input, crop };
  }

  /** One class's map (7x7, may be negative) -> 0..1, as relu then divide by max. */
  function normaliseCam(cam) {
    const out = Array.from(cam, (v) => Math.max(0, v));
    const top = Math.max(...out) + 1e-8;
    return out.map((v) => v / top);
  }

  /** OpenCV INTER_CUBIC resize of a square map (coefficient A = -0.75, replicated edges). */
  function cubicUpsample(cam, n, size) {
    const A = -0.75, scale = n / size;
    const weights = (x) => {
      const c0 = ((A * (x + 1) - 5 * A) * (x + 1) + 8 * A) * (x + 1) - 4 * A;
      const c1 = ((A + 2) * x - (A + 3)) * x * x + 1;
      const c2 = ((A + 2) * (1 - x) - (A + 3)) * (1 - x) * (1 - x) + 1;
      return [c0, c1, c2, 1 - c0 - c1 - c2];
    };
    const axis = [];
    for (let i = 0; i < size; i++) {
      const f = (i + 0.5) * scale - 0.5, s = Math.floor(f);
      axis.push({ w: weights(f - s), at: [-1, 0, 1, 2].map((d) => Math.min(n - 1, Math.max(0, s + d))) });
    }
    const out = new Float32Array(size * size);
    for (let y = 0; y < size; y++) {
      const ay = axis[y];
      for (let x = 0; x < size; x++) {
        const ax = axis[x];
        let v = 0;
        for (let j = 0; j < 4; j++) for (let i = 0; i < 4; i++) v += ay.w[j] * ax.w[i] * cam[ay.at[j] * n + ax.at[i]];
        out[y * size + x] = v;
      }
    }
    return out;
  }

  /** Returns RGBA pixels of the photo crop blended with the JET-coloured heat map. jet: flat [r,g,b, ...] of 256 colours. */
  function overlay(crop, camMap, jet) {
    const heat = cubicUpsample(normaliseCam(camMap), 7, IMG_SIZE);
    const out = new Uint8ClampedArray(IMG_SIZE * IMG_SIZE * 4);
    for (let i = 0; i < IMG_SIZE * IMG_SIZE; i++) {
      const h = Math.trunc(255 * Math.min(1, Math.max(0, heat[i])));
      for (let c = 0; c < 3; c++) out[i * 4 + c] = Math.trunc(0.55 * crop[i * 3 + c] + 0.45 * jet[h * 3 + c]);
      out[i * 4 + 3] = 255;
    }
    return out;
  }

  return { IMG_SIZE, prepare, resizeBilinear, normaliseCam, cubicUpsample, overlay };
});

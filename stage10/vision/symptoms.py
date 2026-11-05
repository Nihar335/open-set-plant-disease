"""Visible-sign analyser for one leaf photo (scope change SC-M5-02, owner decision 2026-10-15).

Deterministic colour and shape measurements (OpenCV, no learned model) inside the leaf area of the photo:

  sign id            what is measured
  yellowing          share of the leaf that is yellow (chlorotic) instead of green
  brown_spots        separate brown/tan patches on the leaf (count and area)
  dark_spots         separate very dark (black/dark-brown) patches on the leaf
  yellow_halo        brown or dark spots surrounded by a yellow ring
  elongated_lesions  brown/tan patches that are long and narrow (cigar/streak shaped)
  small_spots        many tiny separate yellow-orange to brown dots (rust pustules or early leaf spots)
  white_coating      whitish/grey powdery or fuzzy patches on the leaf
  margin_scorch      browning concentrated along the leaf edge
  large_blight       brown/dead tissue covering a large share of the leaf
  bronzing           diffuse reddish-bronze colour across the leaf

Each sign is reported "seen" only when its measure exceeds the threshold in vision/symptom_calibration.json, which was
set on Stage 1-9 VALIDATION images (never test images) as a high percentile of the same measure on healthy leaves. Every
sign maps to knowledge-base features (SIGN_TO_FEATURES) so the existing retrieval can match it to sourced conditions.

Limits, stated in every result: colour analysis cannot see under the leaf, cannot tell a disease from sun scorch,
dirt, shadows, insect damage or a nutrient problem with the same colour, and was calibrated on laboratory photos of six
crops. A sign "seen" is a hint to check, not a diagnosis.
"""
from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

CALIBRATION = Path(__file__).with_name("symptom_calibration.json")
WORK_SIDE = 512
ANALYSER_VERSION = "1.0.0"

SIGNS = ("yellowing", "brown_spots", "dark_spots", "yellow_halo", "elongated_lesions", "small_spots", "white_coating",
         "margin_scorch", "large_blight", "bronzing")

LABELS = {
    "yellowing": "Yellowing of the leaf",
    "brown_spots": "Brown or tan spots",
    "dark_spots": "Dark brown or black spots",
    "yellow_halo": "Yellow ring (halo) around spots",
    "elongated_lesions": "Long, narrow (cigar- or streak-shaped) lesions",
    "small_spots": "Many tiny yellow-orange or brown dots",
    "white_coating": "Whitish or grey powdery / fuzzy patches",
    "margin_scorch": "Browning along the leaf edge",
    "large_blight": "Large dead (brown) areas of the leaf",
    "bronzing": "Reddish-bronze colouring of the leaf",
}

# The single knowledge-base feature each sign shows (stage10/knowledge_base/data/features.yaml). Generic features
# (FEAT-LEAF-SPOTS, FEAT-LEAF-YELLOWING) exist so that a plain colour finding is never reported as a specific pattern
# such as "concentric rings". No sign maps to a feature that needs a hand lens, the leaf underside, a cut stem or weather.
SIGN_TO_FEATURES = {
    "yellowing": ["FEAT-LEAF-YELLOWING"],
    "brown_spots": ["FEAT-LEAF-SPOTS"],
    "dark_spots": ["FEAT-LEAF-SPOTS"],
    "yellow_halo": ["FEAT-YELLOW-HALO"],
    "elongated_lesions": ["FEAT-ELONGATED-CIGAR-LESIONS"],
    "small_spots": ["FEAT-LEAF-SPOTS"],
    "white_coating": ["FEAT-WHITE-GROWTH-UNDERSIDE"],
    "margin_scorch": ["FEAT-MARGIN-SCORCH"],
    "large_blight": ["FEAT-RAPID-BLIGHT-DEFOLIATION"],
    "bronzing": ["FEAT-BRONZING"],
}

LIMITS = ("Signs are found by colour and shape analysis of this one photo. They are hints to check on the plant, not a "
          "diagnosis: sun scorch, dirt, shadows, insect damage or nutrient problems can look the same. The underside of "
          "the leaf, tiny structures and the stem cannot be seen. The analyser was calibrated on laboratory leaf photos.")


@dataclass
class SignResult:
    sign: str
    seen: bool
    measure: float
    threshold: float | None
    detail: str = ""


@dataclass
class Analysis:
    ok: bool
    leaf_fraction: float
    signs: list = field(default_factory=list)
    reason: str | None = None

    def seen(self) -> list[str]:
        return [s.sign for s in self.signs if s.seen]

    def features(self) -> list[str]:
        out = []
        for s in self.seen():
            for f in SIGN_TO_FEATURES[s]:
                if f not in out:
                    out.append(f)
        return out

    def observations(self) -> tuple:
        """ImageObservation records for the report engine (origin image_analysis, evidence 'inferred')."""
        from stage10.report_engine.inputs import ImageObservation

        out, seen_f = [], set()
        for r in self.signs:
            if not r.seen:
                continue
            for f in SIGN_TO_FEATURES[r.sign]:
                if f in seen_f:
                    continue
                seen_f.add(f)
                out.append(ImageObservation(feature_id=f, confidence="low", measurement_method=f"colour/shape analysis: {LABELS[r.sign].lower()}",
                                            analyser="colour_shape_v" + ANALYSER_VERSION,
                                            quality_limitations=("calibrated on laboratory photos only",)))
        return tuple(out)

    def to_dict(self) -> dict:
        return {"analyser": "colour_shape_v" + ANALYSER_VERSION, "ok": self.ok, "reason": self.reason,
                "leaf_fraction": round(self.leaf_fraction, 3),
                "signs": [{"sign": s.sign, "label": LABELS[s.sign], "seen": s.seen, "measure": round(s.measure, 4),
                           "threshold": s.threshold, "detail": s.detail} for s in self.signs],
                "seen": self.seen(), "limits": LIMITS}


def _load(image) -> np.ndarray:
    if isinstance(image, (bytes, bytearray)):
        arr = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
    elif isinstance(image, np.ndarray):
        arr = image
    else:
        arr = cv2.imdecode(np.fromfile(str(image), np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise ValueError("unreadable image")
    h, w = arr.shape[:2]
    s = WORK_SIDE / max(h, w)
    if s < 1:
        arr = cv2.resize(arr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    return arr


def _largest_regions(mask: np.ndarray, min_frac: float = 0.02) -> np.ndarray:
    """Keep connected regions of at least min_frac of the image (drops background specks), then fill holes."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    keep = np.zeros_like(mask, dtype=bool)
    area = mask.size
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_frac * area:
            keep |= lab == i
    if not keep.any() and n > 1:
        keep = lab == (1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])))
    filled = keep.astype(np.uint8) * 255
    cnts, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(filled, cnts, -1, 255, cv2.FILLED)
    return filled > 0


def leaf_mask(bgr: np.ndarray) -> np.ndarray:
    """Plant tissue: green, yellow or brown, reasonably saturated. Grey/black/white backgrounds, sky and soil-grey drop out.
    Holes (spots, coating) inside the leaf outline are filled so they are measured as part of the leaf."""
    hsv = cv2.cvtColor(cv2.GaussianBlur(bgr, (5, 5), 0), cv2.COLOR_BGR2HSV)
    hh, ss, vv = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    tissue = (((hh >= 8) & (hh <= 95)) & (ss >= 50) & (vv >= 40))
    tissue = cv2.morphologyEx(tissue.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    tissue = cv2.morphologyEx(tissue, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    return _largest_regions(tissue > 0)


def _blobs(mask: np.ndarray, min_px: int, max_frac: float):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    out = []
    lim = max_frac * mask.size
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        if min_px <= a <= lim:
            out.append((i, a, lab))
    return out


def measures(image) -> tuple[dict, float, np.ndarray]:
    bgr = _load(image)
    leaf = leaf_mask(bgr)
    lf = float(leaf.mean())
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = (hsv[..., i].astype(int) for i in range(3))
    n_leaf = max(int(leaf.sum()), 1)
    green = leaf & (H >= 33) & (H <= 95) & (S >= 45)
    yellow = leaf & (H >= 20) & (H < 33) & (S >= 70) & (V >= 110)
    brown = leaf & (H >= 5) & (H < 20) & (S >= 60) & (V >= 50) & (V < 200)
    vmed = float(np.median(V[leaf])) if leaf.any() else 128.0
    dark = leaf & (V < min(70, 0.5 * vmed)) & ~((H >= 33) & (H <= 95) & (S >= 45))     # dark AND not dark-green tissue
    whiteish = leaf & (S < 45) & (V >= 150)
    rust = leaf & (H >= 8) & (H <= 28) & (S >= 70) & (V >= 60)
    bronze = leaf & ((H <= 7) | (H >= 170)) & (S >= 60) & (V >= 50)

    m = {}
    m["yellowing"] = float(yellow.sum()) / n_leaf
    k3 = np.ones((3, 3), np.uint8)
    lesion = cv2.morphologyEx((brown | dark).astype(np.uint8), cv2.MORPH_OPEN, k3) > 0
    small = max(6, int(0.00005 * leaf.size))
    bs = _blobs(cv2.morphologyEx(brown.astype(np.uint8), cv2.MORPH_OPEN, k3) > 0, small, 0.15)
    ds = _blobs(cv2.morphologyEx(dark.astype(np.uint8), cv2.MORPH_OPEN, k3) > 0, small, 0.10)
    m["brown_spots"] = float(sum(a for _, a, _ in bs)) / n_leaf
    m["dark_spots"] = float(sum(a for _, a, _ in ds)) / n_leaf
    # halo: yellow pixels in a thin ring around lesion spots, relative to ring size
    spots = np.zeros_like(leaf)
    for i, _, lab in bs + ds:
        spots |= lab == i
    if spots.any():
        ring = (cv2.dilate(spots.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0) & ~spots & leaf
        m["yellow_halo"] = float((ring & yellow).sum()) / max(int(ring.sum()), 1)
    else:
        m["yellow_halo"] = 0.0
    # elongation: area share of lesion blobs whose min-area rectangle is >= 3.5:1
    el = 0
    for i, a, lab in _blobs(lesion, small * 3, 0.2):
        cnts, _ = cv2.findContours((lab == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        (_, _), (w, h), _ = cv2.minAreaRect(cnts[0])
        if min(w, h) > 0 and max(w, h) / min(w, h) >= 3.5:
            el += a
    m["elongated_lesions"] = float(el) / n_leaf
    rb = [x for x in _blobs(rust, 3, 0.0004)]                          # many tiny separate dots, not patches
    m["small_spots"] = float(len(rb)) / (n_leaf / 10000.0)            # pustules per 10,000 leaf pixels
    m["white_coating"] = float(cv2.morphologyEx(whiteish.astype(np.uint8), cv2.MORPH_OPEN, k3).sum()) / n_leaf
    # margin scorch: brown share in the outer band of the leaf vs the interior
    inner = cv2.erode(leaf.astype(np.uint8), np.ones((21, 21), np.uint8)) > 0
    band = leaf & ~inner
    bb = float((brown | dark)[band].mean()) if band.any() else 0.0
    bi = float((brown | dark)[inner].mean()) if inner.any() else 0.0
    m["margin_scorch"] = max(0.0, bb - bi)
    m["large_blight"] = float((brown | dark).sum()) / n_leaf if lesion.any() else 0.0
    m["bronzing"] = float(bronze.sum()) / n_leaf
    m["_green"] = float(green.sum()) / n_leaf
    return m, lf, leaf


def calibration() -> dict:
    return json.loads(CALIBRATION.read_text(encoding="utf-8"))


def analyse(image, cal: dict | None = None) -> Analysis:
    try:
        m, lf, _ = measures(image)
    except ValueError as e:
        return Analysis(False, 0.0, [], str(e))
    cal = cal or calibration()
    if lf < cal["min_leaf_fraction"]:
        return Analysis(False, lf, [], "no leaf found: take a closer photo with one leaf filling most of the frame")
    out = []
    for s in SIGNS:
        th = cal["thresholds"].get(s)
        out.append(SignResult(s, th is not None and m[s] > th, m[s], th))
    # a halo needs spots; scorch/blight imply brown spots already seen
    by = {r.sign: r for r in out}
    if by["yellow_halo"].seen and not (by["brown_spots"].seen or by["dark_spots"].seen):
        by["yellow_halo"].seen = False
        by["yellow_halo"].detail = "ring colour found, but no spots"
    return Analysis(True, lf, out)

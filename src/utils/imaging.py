"""Per-image integrity / statistics worker and perceptual hashing."""
from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = 200_000_000  # PlantDoc has some very large web images


def phash64(gray: np.ndarray) -> int:
    """DCT perceptual hash (same recipe as `imagehash.phash`, hash_size=8)."""
    small = cv2.resize(gray.astype(np.float32), (32, 32), interpolation=cv2.INTER_AREA)
    low = cv2.dct(small)[:8, :8]
    bits = (low > np.median(low)).flatten()
    return int(np.packbits(bits).view(">u8")[0])


def dhash64(gray: np.ndarray) -> int:
    """Difference hash: sign of horizontal gradient on a 9x8 thumbnail."""
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA).astype(np.int16)
    return int(np.packbits((small[:, 1:] > small[:, :-1]).flatten()).view(">u8")[0])


def dihedral(gray: np.ndarray):
    """The 8 flips/rotations of an image (dihedral group D4)."""
    for k in range(4):
        r = np.ascontiguousarray(np.rot90(gray, k))
        yield r
        yield np.ascontiguousarray(r[:, ::-1])


def analyze_image(args: tuple[str, int]) -> dict:
    """Fully decode one image and return integrity + statistics.

    `args` = (absolute path, border width in px). Never raises: failures are
    reported in the `ok` / `error` fields so corrupt files are counted, not crashed on.
    """
    path, border = args
    out = {"ok": False, "error": "", "width": np.nan, "height": np.nan, "mode": "", "format": "",
           "file_bytes": np.nan, "md5": "", "pixel_md5": "", "thumb_md5": "", "phash": "", "dhash": "",
           "phash_d4": "", "dhash_d4": "", "exif_orientation": 1,
           "mean_r": np.nan, "mean_g": np.nan, "mean_b": np.nan,
           "border_mean": np.nan, "border_std": np.nan}
    try:
        raw = Path(path).read_bytes()
        out["file_bytes"] = len(raw)
        out["md5"] = hashlib.md5(raw).hexdigest()
        with Image.open(path) as im:
            out["format"], out["mode"] = im.format or "", im.mode
            out["width"], out["height"] = im.size
            try:
                out["exif_orientation"] = int(im.getexif().get(0x0112, 1))
            except Exception:
                pass
            im.load()  # forces full decode -> catches truncated files
            rgb = np.asarray(im.convert("RGB"))
        out["mean_r"], out["mean_g"], out["mean_b"] = rgb.reshape(-1, 3).mean(0).round(2)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        b = max(1, min(border, min(gray.shape) // 4))
        ring = np.concatenate([gray[:b].ravel(), gray[-b:].ravel(), gray[:, :b].ravel(), gray[:, -b:].ravel()])
        out["border_mean"], out["border_std"] = round(float(ring.mean()), 2), round(float(ring.std()), 2)
        out["pixel_md5"] = hashlib.md5(rgb.tobytes() + str(rgb.shape).encode()).hexdigest()
        # normalised image: 16x16 grey, 5-bit intensity -> robust to re-encoding / resizing
        thumb = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA) >> 3
        out["thumb_md5"] = hashlib.md5(thumb.tobytes()).hexdigest()
        out["phash"] = f"{phash64(gray):016x}"
        out["dhash"] = f"{dhash64(gray):016x}"
        # orientation-invariant: minimum hash over the 8 flips/rotations (catches mirrored copies)
        small = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
        out["phash_d4"] = f"{min(phash64(g) for g in dihedral(small)):016x}"
        out["dhash_d4"] = f"{min(dhash64(g) for g in dihedral(small)):016x}"
        out["ok"] = True
    except Exception as e:  # noqa: BLE001 — we want every failure recorded
        out["error"] = f"{type(e).__name__}: {e}"[:200]
    return out

"""Image quality gate for uploaded photos (M5-A2) and retake requests (M5-A3).

Measures (deterministic, no model):
  * sharpness: variance of the Laplacian of the grey image resized to a 256 px shorter side (resolution-normalised);
  * exposure: mean grey level (0-255) of the same image.
Thresholds come from stage10/config.yaml (image_quality), set from Stage 1-9 VALIDATION images with synthetic
degradations by stage10/vision/calibrate_quality.py (record: stage10/vision/quality_calibration.json). No final-test or
held-out image is used. A null threshold means "not assessed" and is reported as such, never as "passed".

Every issue maps to a retake request that names the needed view and why (M5-A3).
"""
from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass

import numpy as np

from stage10.schemas.validate import stage10_config

NORM_SIDE = 256

RETAKE = {
    "too_large": {"view": "the same photo, saved smaller", "why": "The file is larger than the upload limit."},
    "bad_format": {"view": "a JPEG or PNG photo", "why": "Only JPEG and PNG photos can be checked."},
    "unreadable": {"view": "a new photo", "why": "The file could not be opened as an image."},
    "too_small": {"view": "a closer photo of one affected leaf filling most of the frame",
                  "why": "The photo is too small to show the symptoms clearly."},
    "blurry": {"view": "a sharp photo: hold the phone steady and tap the leaf on screen to focus",
               "why": "The photo is blurred, so spots and their edges cannot be seen."},
    "too_dark": {"view": "a photo in daylight or open shade", "why": "The photo is too dark to see the colours of the symptoms."},
    "too_bright": {"view": "a photo out of direct sun, without flash glare", "why": "The photo is too bright; details are washed out."},
}


@dataclass(frozen=True)
class QualityResult:
    sha256: str
    md5: str
    width: int
    height: int
    format: str | None
    issues: tuple            # issue codes (keys of RETAKE)
    not_assessed: tuple      # measures with no configured threshold
    sharpness: float | None
    brightness: float | None

    @property
    def passed(self) -> bool:
        return not self.issues

    def messages(self) -> list[str]:
        cfg = stage10_config()["image_quality"]
        txt = {"too_large": f"file larger than {cfg['max_upload_mb']} MB", "bad_format": "format not allowed (JPEG or PNG only)",
               "unreadable": "file is not a readable image", "too_small": f"shorter side is below {cfg['min_side_px']} px",
               "blurry": "photo is blurred", "too_dark": "photo is too dark", "too_bright": "photo is too bright"}
        return [txt[i] for i in self.issues]

    def retake_requests(self) -> list[dict]:
        return [{"issue": i, **RETAKE[i]} for i in self.issues]


def measures(im) -> tuple[float, float]:
    """(sharpness, brightness) of a PIL image, on a resolution-normalised grey copy."""
    import cv2
    g = np.asarray(im.convert("L"), dtype=np.uint8)
    h, w = g.shape
    s = NORM_SIDE / min(h, w)
    g = cv2.resize(g, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(g, cv2.CV_64F).var()), float(g.mean())


def check_bytes(data: bytes) -> QualityResult:
    from PIL import Image, UnidentifiedImageError
    cfg = stage10_config()["image_quality"]
    sha, md5 = hashlib.sha256(data).hexdigest(), hashlib.md5(data).hexdigest()
    issues, not_assessed = [], []
    if len(data) > cfg["max_upload_mb"] * 1024 * 1024:
        return QualityResult(sha, md5, 0, 0, None, ("too_large",), (), None, None)
    try:
        with Image.open(io.BytesIO(data)) as im:
            fmt, (w, h) = im.format, im.size
            if (fmt or "").lower() not in cfg["allowed_formats"]:
                return QualityResult(sha, md5, w, h, fmt, ("bad_format",), (), None, None)
            im.load()
            sharp, bright = measures(im)
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        return QualityResult(sha, md5, 0, 0, None, ("unreadable",), (), None, None)
    if min(w, h) < cfg["min_side_px"]:
        issues.append("too_small")
    if cfg.get("blur_laplacian_var_min") is None:
        not_assessed.append("blur")
    elif sharp < cfg["blur_laplacian_var_min"]:
        issues.append("blurry")
    br = cfg.get("brightness_range")
    if br is None:
        not_assessed.append("exposure")
    elif bright < br[0]:
        issues.append("too_dark")
    elif bright > br[1]:
        issues.append("too_bright")
    return QualityResult(sha, md5, w, h, fmt, tuple(issues), tuple(not_assessed), round(sharp, 2), round(bright, 2))

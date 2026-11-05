"""Farmer-guided symptom marking (proposed M5-A7; scope change SC-M5-01 awaiting owner acceptance).

The user may mark where the problem is on their own photo (tap = circle, drag = box, or a 3x3 grid cell for users who
cannot mark precisely). Marks are:
  * USER-SELECTED, never AI-detected or verified (label carried everywhere);
  * normalised to [0, 1] in the photo AS DISPLAYED, i.e. after EXIF orientation is applied, and converted here to pixel
    coordinates of the EXIF-oriented image;
  * stored only in the case record of the response; they are NOT passed to the disease model or the report engine and
    never change a diagnosis (test).
No automatic lesion localisation exists (M5-A5/A6 stay blocked: no licensed annotations, no validated method).
"""
from __future__ import annotations

import io

MAX_MARKS = 10
MIN_BOX = 0.01           # boxes smaller than 1% of a side are treated as taps
MAX_RADIUS = 0.5
LABEL = "Marked by you - not checked by AgroVaidya. This mark does not change the assessment."
SHAPES = ("circle", "box")


class MarkError(ValueError):
    pass


def oriented_size(data: bytes) -> tuple[int, int, int]:
    """(width, height, exif_orientation) of the photo as a browser displays it (EXIF orientation applied)."""
    from PIL import Image
    with Image.open(io.BytesIO(data)) as im:
        w, h = im.size
        try:
            o = int(im.getexif().get(0x0112, 1) or 1)
        except Exception:                                    # noqa: BLE001
            o = 1
    return (h, w, o) if o in (5, 6, 7, 8) else (w, h, o)


def _num(v, name):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v:
        raise MarkError(f"{name} must be a number")
    return float(v)


def _unit(v, name):
    v = _num(v, name)
    if not 0.0 <= v <= 1.0:
        raise MarkError(f"{name} must be between 0 and 1")
    return v


def validate_marks(raw, width: int, height: int) -> list[dict]:
    """Validate normalised marks and add pixel coordinates in the oriented image (width x height)."""
    if raw in (None, "", []):
        return []
    if not isinstance(raw, list):
        raise MarkError("marks must be a list")
    if len(raw) > MAX_MARKS:
        raise MarkError(f"at most {MAX_MARKS} marks")
    if width <= 0 or height <= 0:
        raise MarkError("image size unknown")
    out = []
    for i, m in enumerate(raw, 1):
        if not isinstance(m, dict) or m.get("shape") not in SHAPES:
            raise MarkError(f"mark {i}: shape must be circle or box")
        if m["shape"] == "circle":
            x, y = _unit(m.get("x"), "x"), _unit(m.get("y"), "y")
            r = _num(m.get("r"), "r")
            if not 0 < r <= MAX_RADIUS:
                raise MarkError(f"mark {i}: radius must be in (0, {MAX_RADIUS}]")
            norm = {"x": round(x, 4), "y": round(y, 4), "r": round(r, 4)}
            px = {"x": round(x * width), "y": round(y * height), "r": round(r * min(width, height))}
        else:
            x0, y0, x1, y1 = (_unit(m.get(k), k) for k in ("x0", "y0", "x1", "y1"))
            x0, x1 = sorted((x0, x1))
            y0, y1 = sorted((y0, y1))
            if x1 - x0 < MIN_BOX or y1 - y0 < MIN_BOX:
                raise MarkError(f"mark {i}: box too small")
            norm = {"x0": round(x0, 4), "y0": round(y0, 4), "x1": round(x1, 4), "y1": round(y1, 4)}
            px = {"x0": round(x0 * width), "y0": round(y0 * height), "x1": round(x1 * width), "y1": round(y1 * height)}
        via = m.get("via", "pointer")
        if via not in ("pointer", "grid"):
            raise MarkError(f"mark {i}: via must be pointer or grid")
        out.append({"id": i, "shape": m["shape"], "normalized": norm, "pixels": px, "via": via,
                    "source": "user_marked", "verified": False, "label": LABEL})
    return out


def marks_stage(marks: list[dict], width: int, height: int, orientation: int) -> dict:
    return {"status": "marked_by_user" if marks else "none", "regions": marks, "label": LABEL,
            "coordinate_basis": {"space": "photo as displayed (EXIF orientation applied)", "width": width, "height": height,
                                 "exif_orientation": orientation},
            "used_in_assessment": False,
            "automatic_localisation": "not available (no validated method or licensed annotations; M5-A5/A6)"}

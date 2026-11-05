"""Pure helpers for the Milestone 4 pilot acquisition (no network, no model). Tested offline in stage10/tests.

* hashing (SHA-256 / MD5) of original bytes and stored copies;
* prohibited-overlap check: exact MD5 against EVERY Stage 1-9 image listed in data/splits/manifest_all.csv (the CSV is read;
  no project image is opened). Exact-file checks cannot detect visually similar, re-encoded or resized copies;
* image validation (decodes fully, RGB-convertible, short side >= min_side);
* 64-bit difference hash under the 8 rotations/reflections, so flipped/rotated augmentation copies count as duplicates;
* photographer normalisation and the deterministic photographer-level split.
"""
from __future__ import annotations

import csv
import hashlib
import io
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CONFIG = HERE / "pilot_config.yaml"


@lru_cache(maxsize=1)
def config() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def md5(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


@lru_cache(maxsize=1)
def prohibited_md5() -> frozenset:
    """MD5 of every Stage 1-9 image (manifest column only)."""
    with open(ROOT / config()["prohibited_md5_manifest"], newline="", encoding="utf-8") as f:
        return frozenset(r["md5"].lower() for r in csv.DictReader(f) if r.get("md5"))


def normalise_photographer(name: str | None) -> str | None:
    if not name:
        return None
    s = unicodedata.normalize("NFKC", str(name)).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s or None


def split_for_photographer(key: str, salt: str, dev_fraction: float) -> str:
    """Deterministic: the same photographer always lands in the same split, for every taxon."""
    h = int(hashlib.sha256(f"{salt}|{key}".encode("utf-8")).hexdigest()[:12], 16) / float(16 ** 12)
    return "dev" if h < dev_fraction else "sealed"


def licence_ok(uri: str | None, allowed: list[str]) -> bool:
    if not uri:
        return False
    u = uri.strip().lower().replace("https://", "http://").rstrip("/") + "/"
    return u in {a.lower().rstrip("/") + "/" for a in allowed}


def excluded_by_name(name: str, patterns: list[str]) -> str | None:
    for p in patterns:
        if re.search(p, name):
            return p
    return None


def load_image(b: bytes, min_side: int):
    """Return (PIL RGB image, (w, h), None) or (None, None, reason)."""
    from PIL import Image, UnidentifiedImageError
    try:
        with Image.open(io.BytesIO(b)) as im:
            im.verify()
        im = Image.open(io.BytesIO(b))
        im.load()
        size = im.size
        im = im.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as e:
        return None, None, f"corrupt_or_unreadable:{type(e).__name__}"
    if min(size) < min_side:
        return None, size, f"too_small:{size[0]}x{size[1]}"
    return im, size, None


def dhash(im, size: int = 8) -> int:
    g = im.convert("L").resize((size + 1, size))
    px = list(g.getdata())
    bits = 0
    for r in range(size):
        for c in range(size):
            bits = (bits << 1) | (px[r * (size + 1) + c] > px[r * (size + 1) + c + 1])
    return bits


def dihedral_hashes(im) -> list[int]:
    from PIL import Image
    out = []
    for base in (im, im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)):
        for rot in (None, Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_180, Image.Transpose.ROTATE_270):
            out.append(dhash(base if rot is None else base.transpose(rot)))
    return out


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def near_duplicate(hashes: list[int], seen: list[tuple[int, str]], max_dist: int) -> str | None:
    """seen = [(canonical dhash, record id)]; returns the id of a near-duplicate under any rotation/reflection."""
    for h0, rid in seen:
        if min(hamming(h0, h) for h in hashes) <= max_dist:
            return rid
    return None


def stored_copy(im, max_side: int) -> bytes:
    im = im.copy()
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


class SealedSplitError(PermissionError):
    pass


APPROVAL = HERE / "SEALED_SCORING_APPROVAL.json"


def assert_sealed_access_approved():
    """Sealed images may be read for scoring only after the owner's approval is recorded in this file (absent today)."""
    import json
    if not APPROVAL.exists():
        raise SealedSplitError("sealed split: owner approval for scoring is not recorded (stage10/plant_id/pilot/SEALED_SCORING_APPROVAL.json)")
    rec = json.loads(APPROVAL.read_text(encoding="utf-8"))
    if not (rec.get("approved_by_owner") is True and rec.get("date") and rec.get("protocol_sha256")):
        raise SealedSplitError("sealed split: approval record incomplete")
    return rec

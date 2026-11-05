"""Structured, validated inputs for the AgroVaidya report engine.

The engine never accepts free-form claims. Every input is one of:
  * a plant name typed by the user (resolved against the knowledge base; otherwise echoed as an unrecognised name);
  * feature IDs from the knowledge-base vocabulary (user reports: present / absent);
  * guided-question answers (question_id -> option_id);
  * image-derived observations (feature IDs + location), produced by a future vision module or by synthetic test fixtures;
  * image metadata and the frozen-model assessment returned by FrozenOSRAdapter.assess();
  * a region at state/district granularity (never a precise location);
  * a language code and an explicit generation timestamp (for reproducibility).
Anything else raises ReportInputError.
"""
from __future__ import annotations

import hashlib
import io
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from stage10.guards import assert_not_final_test
from stage10.schemas.validate import stage10_config

INDIAN_STATES_AND_UTS = (
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa", "Gujarat", "Haryana", "Himachal Pradesh",
    "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha",
    "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal",
    "Andaman and Nicobar Islands", "Chandigarh", "Dadra and Nagar Haveli and Daman and Diu", "Delhi", "Jammu and Kashmir", "Ladakh",
    "Lakshadweep", "Puducherry",
)
_NAME_RE = re.compile(r"^[A-Za-zऀ-ॿ][A-Za-zऀ-ॿ .,'()\-]{0,59}$")
PLANT_PARTS = {"leaf_upper", "leaf_lower", "leaf_unspecified", "stem", "fruit", "root", "whole_plant", "unknown"}
CONFIDENCE = {"high", "moderate", "low", "not_assessable"}
# How a user-reported feature reached the engine (schema finding D-5). There is deliberately no "unconfirmed" channel:
# a typed or spoken description must be confirmed by the user, feature by feature, before it is an observation.
USER_CHANNELS = {"symptom_checklist", "typed_text_confirmed", "voice_transcript_confirmed"}


class ReportInputError(ValueError):
    pass


def _clean_name(s: str | None, what: str) -> str | None:
    """Untrusted short text (plant / district name): NFC-normalised, trimmed, restricted alphabet, max 60 chars."""
    if s is None:
        return None
    if not isinstance(s, str):
        raise ReportInputError(f"{what} must be text")
    s = unicodedata.normalize("NFC", s).strip()
    if not s:
        return None
    if not _NAME_RE.match(s):
        raise ReportInputError(f"{what} contains unsupported characters or is too long")
    return s


@dataclass(frozen=True)
class Region:
    state: str | None = None
    district: str | None = None

    def validated(self) -> "Region":
        state = None
        if self.state is not None:
            m = [s for s in INDIAN_STATES_AND_UTS if s.lower() == str(self.state).strip().lower()]
            if not m:
                raise ReportInputError(f"unknown Indian state/UT {self.state!r} (state or district only; no precise location)")
            state = m[0]
        district = _clean_name(self.district, "district")
        if district and not state:
            raise ReportInputError("a district needs its state")
        return Region(state, district)


@dataclass(frozen=True)
class ImageInfo:
    sha256: str
    width: int
    height: int
    format: str                                   # JPEG | PNG
    quality_issues: tuple = ()                    # empty = quality gate passed
    quality_not_assessed: tuple = ("blur", "exposure")   # config thresholds not yet defined (vision milestone)

    def validated(self) -> "ImageInfo":
        if not re.fullmatch(r"[0-9a-f]{64}", self.sha256 or ""):
            raise ReportInputError("image sha256 must be 64 lowercase hex characters")
        if self.format not in ("JPEG", "PNG"):
            raise ReportInputError("image format must be JPEG or PNG")
        if not (isinstance(self.width, int) and isinstance(self.height, int) and self.width > 0 and self.height > 0):
            raise ReportInputError("image width/height must be positive integers")
        return self


def image_info_from_file(path) -> ImageInfo:
    """Read metadata + basic quality gate from an image file. Refuses Stage 1-9 final-test images (guard)."""
    from PIL import Image

    assert_not_final_test(path)
    cfg = stage10_config()["image_quality"]
    data = Path(path).read_bytes()
    issues = []
    if len(data) > cfg["max_upload_mb"] * 1024 * 1024:
        issues.append(f"file larger than {cfg['max_upload_mb']} MB")
    with Image.open(io.BytesIO(data)) as im:
        fmt, (w, h) = im.format, im.size
    if (fmt or "").lower() not in cfg["allowed_formats"]:
        issues.append(f"format {fmt} not allowed")
    if min(w, h) < cfg["min_side_px"]:
        issues.append(f"shorter side {min(w, h)} px is below {cfg['min_side_px']} px")
    return ImageInfo(hashlib.sha256(data).hexdigest(), w, h, fmt if fmt in ("JPEG", "PNG") else "JPEG", tuple(issues))


@dataclass(frozen=True)
class ImageObservation:
    """A symptom located in the image by an automated analyser (no analyser exists yet: synthetic fixtures only)."""
    feature_id: str
    plant_part: str = "leaf_unspecified"
    bbox: tuple | None = None
    confidence: str = "low"
    affected_area_fraction: float | None = None
    measurement_method: str | None = None
    quality_limitations: tuple = ()
    analyser: str = "unspecified"


@dataclass(frozen=True)
class ReportRequest:
    image: ImageInfo
    plant_name: str | None = None
    plant_confirmation: str = "not_provided"      # confirmed | not_provided | unsure
    reported_features: dict = field(default_factory=dict)    # feature_id -> True (seen) / False (looked, absent)
    reported_channels: dict = field(default_factory=dict)    # feature_id -> USER_CHANNELS value (default symptom_checklist)
    answers: dict = field(default_factory=dict)              # question_id -> option_id
    image_observations: tuple = ()                           # ImageObservation, ...
    model_assessment: dict | None = None                     # FrozenOSRAdapter.assess() output (raw ignored) or None
    region: Region = Region()
    language: str = "en"
    generated_at: str | None = None                          # ISO-8601 UTC; required for reproducible output

    def canonical(self) -> dict:
        """JSON-serialisable canonical form (used for the report ID and reproducibility)."""
        d = asdict(self)
        if d["model_assessment"] is not None:
            d["model_assessment"] = {k: d["model_assessment"][k] for k in ("model_prediction", "osr_status")}
        d["reported_features"] = dict(sorted(d["reported_features"].items()))
        d["answers"] = dict(sorted(d["answers"].items()))
        d["reported_channels"] = dict(sorted(d["reported_channels"].items()))
        return d


def validate_request(req: ReportRequest, kb) -> ReportRequest:
    cfg = stage10_config()
    if req.language not in cfg["language"]["supported"]:
        raise ReportInputError(f"language {req.language!r} not supported")
    if req.plant_confirmation not in ("confirmed", "not_provided", "unsure"):
        raise ReportInputError("plant_confirmation must be confirmed / not_provided / unsure")
    name = _clean_name(req.plant_name, "plant name")
    if req.plant_confirmation == "confirmed" and not name:
        raise ReportInputError("plant_confirmation=confirmed needs a plant name")
    for f, v in req.reported_features.items():
        if f not in kb.features:
            raise ReportInputError(f"unknown feature {f!r}")
        if not isinstance(v, bool):
            raise ReportInputError(f"feature {f} must be true (present) or false (absent)")
    for f, ch in req.reported_channels.items():
        if f not in req.reported_features:
            raise ReportInputError(f"channel given for unreported feature {f!r}")
        if ch not in USER_CHANNELS:
            raise ReportInputError(f"input channel {ch!r} not accepted (unconfirmed text/voice cannot become an observation)")
    for q, o in req.answers.items():
        if q not in kb.questions or o not in {x["option_id"] for x in kb.questions[q]["options"]}:
            raise ReportInputError(f"unknown question/option {q}={o}")
    for ob in req.image_observations:
        if not isinstance(ob, ImageObservation):
            raise ReportInputError("image_observations must be ImageObservation objects")
        if ob.feature_id not in kb.features or kb.features[ob.feature_id]["category"] == "environment":
            raise ReportInputError(f"image observation {ob.feature_id!r} is not a visible symptom feature")
        if ob.plant_part not in PLANT_PARTS or ob.confidence not in CONFIDENCE:
            raise ReportInputError("image observation has an invalid plant_part or confidence")
        if ob.bbox is not None and (len(ob.bbox) != 4 or not all(0 <= x <= 1 for x in ob.bbox)):
            raise ReportInputError("bbox must be 4 normalised coordinates")
        if ob.affected_area_fraction is not None and not ob.measurement_method:
            raise ReportInputError("affected_area_fraction needs a measurement_method")
    ts = req.generated_at or datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError as e:
        raise ReportInputError("generated_at must be ISO-8601") from e
    if dt.tzinfo is None:
        raise ReportInputError("generated_at must include a timezone")
    if req.model_assessment is not None and not {"model_prediction", "osr_status"} <= set(req.model_assessment):
        raise ReportInputError("model_assessment must be the output of FrozenOSRAdapter.assess()")
    return ReportRequest(image=req.image.validated(), plant_name=name, plant_confirmation=req.plant_confirmation,
                         reported_features=dict(req.reported_features), reported_channels=dict(req.reported_channels),
                         answers=dict(req.answers),
                         image_observations=tuple(req.image_observations), model_assessment=req.model_assessment,
                         region=req.region.validated(), language=req.language, generated_at=ts)

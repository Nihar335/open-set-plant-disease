"""Typed symptom descriptions (M5-C1): deterministic suggestions of knowledge-base features, never observations.

The user's text is untrusted (length and character limits). Words are matched against the knowledge base's own feature
labels and a small, fixed list of plain-language cue words that map to those features (written here as editorial
matching hints, not as agronomic claims). Every suggestion must be explicitly confirmed by the user before it becomes a
reported observation (input channel "typed_text_confirmed"); words that match nothing are returned as "not understood".
The text itself is not stored or put in the report.
"""
from __future__ import annotations

import re
import unicodedata

MAX_LEN = 500
_ALLOWED = re.compile(r"^[\w\s.,;:!?'()/%&+\-ऀ-ॿ]*$")
STOP = {"the", "a", "an", "and", "or", "of", "on", "in", "to", "is", "are", "my", "with", "has", "have", "some", "there", "it", "its",
        "leaf", "leaves", "plant", "plants", "very", "also", "then", "from", "at", "by", "be", "been", "this", "that", "they", "i",
        "we", "see", "seen", "look", "looks", "like", "lot", "lots", "many", "few", "all", "small", "big", "large", "getting", "become"}
# editorial cue words -> feature ids (matching hints only; the user confirms every suggestion)
CUES = {
    "ring": ["FEAT-CONCENTRIC-RINGS"], "rings": ["FEAT-CONCENTRIC-RINGS"], "target": ["FEAT-CONCENTRIC-RINGS"],
    "halo": ["FEAT-YELLOW-HALO"], "web": ["FEAT-FINE-WEBBING"], "webbing": ["FEAT-FINE-WEBBING"],
    "wilt": ["FEAT-WILTING"], "wilting": ["FEAT-WILTING"], "wilted": ["FEAT-WILTING"], "drooping": ["FEAT-WILTING"],
    "powder": ["FEAT-WHITE-GROWTH-UNDERSIDE", "FEAT-POWDERY-BROWN-PUSTULES"], "powdery": ["FEAT-WHITE-GROWTH-UNDERSIDE", "FEAT-POWDERY-BROWN-PUSTULES"],
    "mould": ["FEAT-WHITE-GROWTH-UNDERSIDE", "FEAT-DUSTY-SPORES-UNDERSIDE"], "mold": ["FEAT-WHITE-GROWTH-UNDERSIDE", "FEAT-DUSTY-SPORES-UNDERSIDE"],
    "fungus": ["FEAT-WHITE-GROWTH-UNDERSIDE", "FEAT-DUSTY-SPORES-UNDERSIDE"],
    "rust": ["FEAT-POWDERY-BROWN-PUSTULES"], "pustules": ["FEAT-POWDERY-BROWN-PUSTULES"],
    "fruit": ["FEAT-FRUIT-LESIONS"], "fruits": ["FEAT-FRUIT-LESIONS"], "rot": ["FEAT-FRUIT-LESIONS"], "rotting": ["FEAT-FRUIT-LESIONS"],
    "stem": ["FEAT-STEM-LESIONS"], "stems": ["FEAT-STEM-LESIONS"], "canker": ["FEAT-STEM-LESIONS"], "branch": ["FEAT-STEM-LESIONS"],
    "wet": ["FEAT-WATER-SOAKED-LESIONS"], "watery": ["FEAT-WATER-SOAKED-LESIONS"], "soaked": ["FEAT-WATER-SOAKED-LESIONS"],
    "bronze": ["FEAT-BRONZING"], "bronzed": ["FEAT-BRONZING"], "bronzing": ["FEAT-BRONZING"],
    "scorch": ["FEAT-MARGIN-SCORCH"], "scorched": ["FEAT-MARGIN-SCORCH"], "burnt": ["FEAT-MARGIN-SCORCH"], "tips": ["FEAT-MARGIN-SCORCH"],
    "stripes": ["FEAT-INTERVEINAL-STRIPING"], "striped": ["FEAT-INTERVEINAL-STRIPING"],
    "dropping": ["FEAT-RAPID-BLIGHT-DEFOLIATION"], "falling": ["FEAT-RAPID-BLIGHT-DEFOLIATION"], "defoliation": ["FEAT-RAPID-BLIGHT-DEFOLIATION"],
    "older": ["FEAT-OLDER-LOWER-LEAVES-FIRST"], "lower": ["FEAT-OLDER-LOWER-LEAVES-FIRST"], "bottom": ["FEAT-OLDER-LOWER-LEAVES-FIRST"],
    "shrivelled": ["FEAT-FRUIT-MUMMY"], "mummy": ["FEAT-FRUIT-MUMMY"], "mummified": ["FEAT-FRUIT-MUMMY"],
    "rain": ["FEAT-ENV-HIGH-HUMIDITY-WET"], "rainy": ["FEAT-ENV-HIGH-HUMIDITY-WET"], "humid": ["FEAT-ENV-WARM-HUMID"],
    "humidity": ["FEAT-ENV-HIGH-HUMIDITY-WET"], "cold": ["FEAT-ENV-COOL-WET"], "cool": ["FEAT-ENV-COOL-WET"],
}


class SymptomTextError(ValueError):
    pass


def clean(text) -> str:
    if not isinstance(text, str):
        raise SymptomTextError("the description must be text")
    t = unicodedata.normalize("NFC", text).strip()
    if not t:
        raise SymptomTextError("the description is empty")
    if len(t) > MAX_LEN:
        raise SymptomTextError(f"the description is longer than {MAX_LEN} characters")
    if not _ALLOWED.match(t):
        raise SymptomTextError("the description contains characters that cannot be used")
    return t


def suggest(text, kb) -> dict:
    t = clean(text)
    words = [w for w in re.findall(r"[a-z]+", t.casefold()) if w not in STOP and len(w) > 2]
    label_words = {fid: set(re.findall(r"[a-z]+", f["label"].casefold())) - STOP for fid, f in kb.features.items()}
    hits: dict[str, set] = {}
    understood = set()
    for w in words:
        fids = [f for f in CUES.get(w, []) if f in kb.features]
        fids += [fid for fid, lw in label_words.items() if w in lw and fid not in fids]
        for f in fids:
            hits.setdefault(f, set()).add(w)
        if fids:
            understood.add(w)
    ranked = sorted(hits.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:8]
    return {"suggestions": [{"feature_id": f, "label": kb.features[f]["label"], "category": kb.features[f]["category"],
                             "matched_words": sorted(ws), "status": "suggested_needs_confirmation"} for f, ws in ranked],
            "not_understood": sorted({w for w in words if w not in understood}),
            "note": "These are suggestions from your words. Tick only what you have actually seen; nothing is used until you confirm it.",
            "text_stored": False}

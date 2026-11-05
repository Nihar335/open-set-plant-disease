"""Treatment and regional-safety policy for AgroVaidya reports (all reports are Indian: region.country == "IN").

Rules (Milestone 3 brief, section 5):
  * Chemical AND biological control records are WITHHELD from actionable advice unless a registration-verification entry
    exists for that record (registration_verifications.yaml). Such an entry needs current Indian registration, the
    crop/use applicability and label guidance verified from authoritative sources and expert review. The file is empty in
    this milestone, so every chemical/biological option is withheld. Foreign-source chemical records are never actionable.
  * Doses, rates and concentrations are never emitted (stage10/config.yaml forbids them; the validator enforces it).
  * Nutrient problems are emitted only as `suspected_nutrient_issue`: a soil/tissue test plus consultation with the KVK or a
    qualified expert. Named corrective products are withheld (owner decision 2026-10-14); never a rate; never "confirmed".
  * Foreign-source guidance keeps geographic_applicability = international_not_verified_for_India plus a risk note.
  * India-region (Tamil Nadu state-university) guidance carries a note when the user's state differs or is unknown.
"""
from __future__ import annotations

from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
VERIFICATIONS = HERE / "registration_verifications.yaml"

WITHHELD_CATEGORIES = {"chemical", "biological"}
# KB management category -> (report section, recommendation category, subcategory)
MGMT_MAP = {
    "prevention": ("treatment_guidance", "disease_management", "cultural"),
    "cultural": ("treatment_guidance", "disease_management", "cultural"),
    "sanitation": ("treatment_guidance", "disease_management", "sanitation"),
    "irrigation": ("treatment_guidance", "disease_management", "irrigation"),
    "drainage": ("treatment_guidance", "disease_management", "drainage"),
    "resistant_variety": ("treatment_guidance", "disease_management", "resistant_variety"),
    "monitoring": ("immediate_actions", "follow_up", "monitoring"),
    "soil": ("soil_guidance", "general_crop_requirement", "soil"),
    "nutrient": ("nutrition_guidance", "suspected_nutrient_issue", "nutrition"),
    "chemical": (None, None, "chemical"),
    "biological": (None, None, "biological"),
}
DIAG_SUBCATEGORY = {"field_observation": "field_test", "hand_lens": "field_test", "cross_section": "field_test",
                    "photograph": "photography", "soil_test": "laboratory_test", "tissue_test": "laboratory_test",
                    "laboratory_test": "laboratory_test"}
NUTRIENT_PRODUCT_REASON = ("named corrective fertiliser/nutrient products are withheld by owner decision (2026-10-14); "
                           "testing and expert consultation are recommended instead")
FOREIGN_NOTE = "Source is from outside India; not verified for India. Check with local experts before use."
REQUIRED_VERIFICATION_FIELDS = {"management_id", "crop", "use", "registration_source_id", "label_source_id", "verified_on",
                                "expert_reviewed"}


def load_verifications(path: Path = VERIFICATIONS) -> list:
    if not path.exists():
        return []
    return (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("entries") or []


def chemical_decision(mgmt: dict, kb, verifications: list | None = None) -> tuple[bool, str]:
    """(actionable, reason). Actionable only with a complete, expert-reviewed Indian registration verification."""
    if mgmt["category"] not in WITHHELD_CATEGORIES:
        return True, "not a chemical/biological record"
    if mgmt["geographic_applicability"] == "international_not_verified_for_India":
        return False, "foreign source; Indian registration, crop/use applicability and label guidance not verified"
    verifications = load_verifications() if verifications is None else verifications
    for v in verifications:
        if v.get("management_id") != mgmt["management_id"]:
            continue
        if not REQUIRED_VERIFICATION_FIELDS <= set(v) or v.get("expert_reviewed") is not True:
            return False, "registration-verification entry incomplete or not expert-reviewed"
        reg = kb.sources.get(v["registration_source_id"])
        lab = kb.sources.get(v["label_source_id"])
        if not reg or reg["source_type"] != "government_india" or not lab:
            return False, "registration-verification entry does not cite an Indian government registration source and a label source"
        return True, "registration verified"
    return False, "current Indian registration, crop/use applicability and label guidance not verified"


def geographic_applicability(geos: set) -> str:
    """Recommendation-level applicability from the KB geography of the evidence used."""
    if not geos:
        return "general"
    if any(g.startswith("international_not_verified") for g in geos):
        return "international_not_verified_for_India"
    if geos <= {"India_specific", "India_region"}:
        return "India_specific"
    if geos <= {"India_specific", "India_region", "international_verified_for_India"}:
        return "international_verified_for_India"
    return "general"


def region_note(source_publishers: set, geos: set, state: str | None) -> str | None:
    """Note for Indian state-university evidence whose regional applicability is limited."""
    if "India_region" not in geos:
        return None
    if any("Tamil Nadu" in p for p in source_publishers):
        if state == "Tamil Nadu":
            return None
        return ("Source is a Tamil Nadu state agricultural university page; applicability in "
                + (state if state else "your state") + " has not been verified.")
    return "Source is regional Indian guidance; applicability in " + (state if state else "your state") + " has not been verified."

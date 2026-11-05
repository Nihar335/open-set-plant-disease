"""PlantVillage-specific helpers: class-name parsing and physical-leaf grouping.

PlantVillage contains several photographs of the *same physical leaf*. If these
are split across train/val/test, test accuracy and "known" OSR scores are inflated.
The maintainers publish `leaf_grouping/leaf-map.json`, mapping each photo code
(lower-cased, e.g. "rs_early.b 7557") to one or more "<class>:::<leaf_no>".

`maintainer_leaf_id` is a line-for-line port of `_generate_examples` in the
maintainers' Hugging Face loader (mohanty/PlantVillage, plant_village.py, pinned
revision in config.yaml). It is NOT a new algorithm.

Because one photo code can occur under several classes, the grouping key used for
splitting is always `leaf_group_id = "<class_name>||<maintainer_leaf_id>"`.
"""
from __future__ import annotations

import json
from pathlib import Path


def parse_class(class_name: str) -> tuple[str, str]:
    crop, _, disease = class_name.partition("___")
    return crop, disease or "unknown"


def photo_code(file_name: str) -> str:
    """`image_identifier` exactly as computed by the maintainers' loader."""
    ident = file_name.replace("_final_masked", "")
    if "___" in ident:
        ident = ident.split("___")[-1]
    ident = ident.split("copy")[0]
    ident = ident.replace(".jpg", "").replace(".JPG", "").replace(".png", "").replace(".PNG", "")
    return ident.strip()


def load_leaf_map(path: str | Path) -> dict[str, list[str]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def maintainer_leaf_id(class_name: str, file_name: str, leaf_map: dict) -> tuple[str, str]:
    """Return (leaf_id, how) using the maintainers' exact rules.

    how: 'map_unique'          — exactly one map entry for the code (used as-is, as upstream does)
         'map_class_match'     — several entries; the one naming this class was chosen
         'fallback_ambiguous'  — several entries, none for this class -> fallback_<code>
         'fallback_not_in_map' — code absent from the map -> fallback_<code>
    """
    image_identifier = photo_code(file_name)
    lookup_key = image_identifier.lower().strip()
    if lookup_key in leaf_map:
        suggestions = leaf_map[lookup_key]
        if len(suggestions) == 1:
            return suggestions[0], "map_unique"
        for suggestion in suggestions:
            if class_name in suggestion:          # upstream uses substring containment
                return suggestion, "map_class_match"
        return f"fallback_{image_identifier}", "fallback_ambiguous"
    return f"fallback_{image_identifier}", "fallback_not_in_map"


def leaf_group_key(class_name: str, leaf_id: str) -> str:
    return f"{class_name}||{leaf_id}"

"""Build the frozen candidate vocabulary v1 (run once, before any pilot image is acquired or scored).

  python -P -m stage10.plant_id.pilot.build_vocabulary

40 in-vocabulary crop taxa (GBIF Backbone keys, taxonomy fetched from the GBIF species API) with editorial English common
names. The six P2 'unsupported' plants are deliberately NOT in the vocabulary. Writes vocabulary_v1.json.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import requests

from stage10.plant_id.pilot import common as C

OUT = C.HERE / "vocabulary_v1.json"
# scientific name -> (GBIF accepted usageKey from species/match 2026-10-14, editorial common name, crop group)
TAXA = {
    "Oryza sativa": (2703459, "rice", "cereal"), "Triticum aestivum": (7795888, "wheat", "cereal"), "Zea mays": (5290052, "maize", "cereal"),
    "Sorghum bicolor": (2705181, "sorghum", "cereal"), "Cenchrus americanus": (5828197, "pearl millet", "cereal"),
    "Eleusine coracana": (2705957, "finger millet", "cereal"), "Cajanus cajan": (7587087, "pigeon pea", "pulse"),
    "Cicer arietinum": (2947311, "chickpea", "pulse"), "Vigna radiata": (2982380, "mung bean", "pulse"),
    "Vigna mungo": (2982533, "black gram", "pulse"), "Vigna unguiculata": (2982583, "cowpea", "pulse"),
    "Arachis hypogaea": (5353770, "groundnut", "oilseed"), "Brassica juncea": (3042751, "Indian mustard", "oilseed"),
    "Brassica oleracea": (3042845, "cabbage", "vegetable"), "Sesamum indicum": (3172622, "sesame", "oilseed"),
    "Gossypium hirsutum": (3152661, "cotton", "fibre"), "Saccharum officinarum": (2703912, "sugarcane", "cash crop"),
    "Solanum lycopersicum": (2930137, "tomato", "vegetable"), "Solanum melongena": (2930617, "brinjal", "vegetable"),
    "Solanum tuberosum": (2930262, "potato", "vegetable"), "Capsicum annuum": (2932944, "chilli", "vegetable"),
    "Abelmoschus esculentus": (3152707, "okra", "vegetable"), "Mangifera indica": (3190638, "mango", "fruit tree"),
    "Carica papaya": (2874484, "papaya", "fruit"), "Psidium guajava": (5420380, "guava", "fruit tree"),
    "Citrus limon": (7647136, "lemon", "fruit tree"), "Citrus maxima": (3190160, "pomelo", "fruit tree"),
    "Cocos nucifera": (2735117, "coconut", "plantation"), "Curcuma longa": (2757624, "turmeric", "spice"),
    "Allium cepa": (2857697, "onion", "vegetable"), "Momordica charantia": (2874581, "bitter gourd", "vegetable"),
    "Lagenaria siceraria": (2874672, "bottle gourd", "vegetable"), "Luffa aegyptiaca": (2874599, "sponge gourd", "vegetable"),
    "Cucurbita maxima": (2874515, "pumpkin", "vegetable"), "Coriandrum sativum": (3034871, "coriander", "spice"),
    "Musa acuminata": (2762680, "banana", "fruit"), "Punica granatum": (5420901, "pomegranate", "fruit"),
    "Ricinus communis": (5380041, "castor", "oilseed"), "Areca catechu": (2736531, "areca nut", "plantation"),
    "Coffea arabica": (2895345, "coffee", "plantation"),
}
LOOK_ALIKE_GROUPS = {"Solanum": ["Solanum lycopersicum", "Solanum melongena", "Solanum tuberosum"],
                     "Vigna": ["Vigna radiata", "Vigna mungo", "Vigna unguiculata"],
                     "Cucurbitaceae": ["Momordica charantia", "Lagenaria siceraria", "Luffa aegyptiaca", "Cucurbita maxima"],
                     "millets_sorghum": ["Sorghum bicolor", "Cenchrus americanus", "Eleusine coracana"],
                     "palms": ["Cocos nucifera", "Areca catechu"], "Citrus": ["Citrus limon", "Citrus maxima"],
                     "Brassica": ["Brassica juncea", "Brassica oleracea"]}


def main():
    if OUT.exists():
        raise SystemExit("vocabulary_v1.json already frozen; create a new version instead of overwriting")
    s = requests.Session()
    s.headers["User-Agent"] = "AgroVaidya-research-pilot/0.1"
    taxa = []
    for name, (key, common, group) in TAXA.items():
        d = s.get(f"https://api.gbif.org/v1/species/{key}", timeout=60).json()
        time.sleep(0.3)
        if d.get("canonicalName") != name or d.get("taxonomicStatus") != "ACCEPTED":
            raise SystemExit(f"GBIF record for {name} ({key}) is {d.get('canonicalName')} / {d.get('taxonomicStatus')}")
        taxa.append({"scientific_name": name, "gbif_key": key, "rank": d.get("rank"), "kingdom": d.get("kingdom"), "phylum": d.get("phylum"),
                     "class": d.get("class"), "order": d.get("order"), "family": d.get("family"), "genus": d.get("genus"),
                     "common_name_en": common, "common_name_source": "editorial", "crop_group": group})
    vocab = {"version": "1", "frozen_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
             "frozen_before": "any pilot image acquisition or scoring", "n_taxa": len(taxa), "taxa": taxa,
             "look_alike_groups": LOOK_ALIKE_GROUPS,
             "unsupported_not_in_vocabulary": ["Lantana camara", "Parthenium hysterophorus", "Calotropis gigantea", "Azadirachta indica",
                                               "Ficus religiosa", "Prosopis juliflora"],
             "taxonomy_source": "GBIF Backbone Taxonomy, species API, fetched at freeze time"}
    OUT.write_text(json.dumps(vocab, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"froze {len(taxa)} taxa -> {OUT}")


if __name__ == "__main__":
    main()

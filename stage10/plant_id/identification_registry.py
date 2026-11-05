"""Plant-identification coverage registry (M4-07), kept separate from disease-diagnosis coverage.

  python -m stage10.plant_id.identification_registry --write | --check

'identifiable' comes ONLY from sealed plant-identification evaluation evidence (protocol v2); until that evaluation is
authorised and run, every taxon is 'not_established'. 'diagnosable' is a separate field taken only from the frozen
six-crop disease model scope in the knowledge base; it is never inferred from identification, and identification is never
inferred from it. Plants outside vocabulary v1 are 'not_supported'. No universal-recognition claim is made.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "identification_registry.json"
VOCAB = HERE / "pilot" / "vocabulary_v1.json"
STATUSES = {"met", "not_met", "not_established", "not_evaluated"}
EVALUATIONS: dict = {}        # evaluation_id -> {taxon: status}; filled only from an authorised sealed protocol-v2 evaluation


def build() -> dict:
    from stage10.knowledge_base.kb import load_kb
    kb = load_kb(check_cache=False)
    frozen = {" ".join(p["scientific_name"].split()[:2]): p for p in kb.plants.values() if p["frozen_model_scope"]}
    v = json.loads(VOCAB.read_text(encoding="utf-8"))
    taxa = {}
    for t in v["taxa"]:
        name = t["scientific_name"]
        taxa[name] = {
            "common_name_en": t["common_name_en"],
            "identifiable": {"status": "not_established", "evaluation_id": None,
                             "basis": "no authorised sealed protocol-v2 evaluation yet; the retired pilot is not evidence"},
            "diagnosable": {"frozen_disease_model_scope": name in frozen,
                            "kb_plant_id": frozen[name]["plant_id"] if name in frozen else None,
                            "basis": "frozen six-crop disease classifier scope (knowledge base), independent of identification"},
        }
    return {"description": ("Plant-identification coverage. 'identifiable' and 'diagnosable' are separate and never inferred "
                            "from each other. Plants outside the vocabulary are not supported. "
                            "No universal plant recognition is claimed."),
            "vocabulary": {"id": "vocabulary_v1", "sha256": hashlib.sha256(VOCAB.read_bytes()).hexdigest(), "n_taxa": len(taxa)},
            "outside_vocabulary": {"status": "not_supported",
                                   "behaviour": "the adapter should refuse; wrong acceptances are possible and are measured "
                                                "only on development data until the sealed evaluation is authorised"},
            "taxa": taxa}


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true")
    g.add_argument("--check", action="store_true")
    a = ap.parse_args()
    reg = build()
    if a.write:
        OUT.write_text(json.dumps(reg, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {OUT.name}: {len(reg['taxa'])} taxa")
        return
    ok = OUT.exists() and json.loads(OUT.read_text(encoding="utf-8")) == reg
    print("IDENTIFICATION REGISTRY:", "PASS (up to date)" if ok else "FAIL (stale or missing)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

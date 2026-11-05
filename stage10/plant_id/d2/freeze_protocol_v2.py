"""Freeze protocol v2: write protocol_v2.json (machine-readable parameters) and protocol_v2_lock.json (SHA-256 of every
frozen input). Refuses to overwrite an existing lock.

  python -m stage10.plant_id.d2.freeze_protocol_v2
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from stage10.plant_id.d2 import plan_manifest as P
from stage10.plant_id.d2 import survey_availability as S

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
LOCKED = ["stage10/plant_id/d2/PROTOCOL_V2.md", "stage10/plant_id/d2/protocol_v2.json", "stage10/plant_id/d2/THRESHOLD_SELECTION_METHOD.md",
          "stage10/plant_id/d2/d2_config.yaml", "stage10/plant_id/d2/d2_survey_config.yaml",
          "stage10/plant_id/d2/proposed/d2_proposed_manifest.csv", "stage10/plant_id/pilot/vocabulary_v1.json",
          "stage10/plant_id/d2/neardup/calibration.json"]
METHOD = {"bound": "max(Clopper-Pearson one-sided 95%, photographer-cluster bootstrap micro 95th pct, stratified bootstrap macro 95th pct)",
          "target_false_acceptance": 0.10, "bootstrap_B": 10000, "seed": 20261015, "folds": 5,
          "fold_salt": "agrovaidya-m4-d2-folds-v1", "stability_tau_range_max": 0.05, "stability_max_folds_over_target": 1,
          "feasibility_coverage_min": 0.70, "feasibility_correct_with_refusals_min": 0.80}


def main():
    lock_p = HERE / "protocol_v2_lock.json"
    if lock_p.exists():
        raise SystemExit("protocol v2 already frozen; refusing to overwrite")
    cfg, scfg = P.config(), S.config()
    proto = {"protocol": "v2", "status": "FROZEN", "frozen_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "document": "stage10/plant_id/d2/PROTOCOL_V2.md", "model_weights_sha256_prefix": "b7b2bf6f",
             "vocabulary": "stage10/plant_id/pilot/vocabulary_v1.json", "prompt_template": "T1",
             "score": "top-1 softmax over the 40 vocabulary taxa; not a calibrated probability",
             "salt": cfg["split"]["salt"], "dev_fraction": cfg["split"]["dev_fraction"], "caps": cfg["caps"],
             "targets": cfg["targets"], "minimums": scfg["minimums"], "c1_status": cfg["c1_status"],
             "licences": cfg["licences"]["allowed_media_licence_uris"], "intended_use": "internal, non-commercial research evaluation only",
             "threshold_method": METHOD,
             "sealed_scoring": "NOT authorized by this protocol; requires SEALED_SCORING_APPROVAL_D2.json (owner, hash-bound)",
             "verdicts": {"unsupported_false_acceptance": "met if one-sided 95% upper bound <= 0.10; not met if point > 0.10; else not established",
                          "supported_identification": "met if lower 95% bound of correct-with-refusals >= 0.80 and coverage >= 0.70",
                          "c1_coconut_to_areca": "met if <= 1 answered error in 60", "c1_areca_to_coconut": "descriptive only"}}
    (HERE / "protocol_v2.json").write_text(json.dumps(proto, indent=1) + "\n", encoding="utf-8")
    files = {rel: hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() for rel in LOCKED}
    lock_p.write_text(json.dumps({"frozen_utc": proto["frozen_utc"], "files": files}, indent=1) + "\n", encoding="utf-8")
    print(f"protocol v2 frozen: {len(files)} files locked")


if __name__ == "__main__":
    main()

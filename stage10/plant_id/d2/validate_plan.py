"""Independent validator for the PROPOSED D2/C1 acquisition manifest (metadata only).

  python -m stage10.plant_id.d2.validate_plan

Rule checks (any failure = FAIL): pilot exclusions, photographer separation, both caps, licence allow-list and NC flag,
date window, uniqueness, taxon keys, attribution completeness. Minimum checks (frozen minimums, never relaxed) are reported
separately as MET / SHORTFALL. Overlap checks on image bytes cannot run before acquisition; their enforcement is tested in
stage10/tests/test_plant_id_d2_acquisition_plan.py (overlap_gate).
"""
from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter, defaultdict

from stage10.plant_id.d2 import plan_manifest as P
from stage10.plant_id.d2 import survey_availability as S


def rows():
    with open(P.OUT / "d2_proposed_manifest.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def rule_errors(rs: list[dict]) -> list[str]:
    cfg, scfg = P.config(), S.config()
    err = []
    ids, phot = S.pilot_exclusions()
    vocab = S.vocabulary_keys()
    plan = {n: (r, g) for n, r, g in S.taxa_plan(scfg)}
    lo, hi = cfg["source"]["event_date"].split(",")
    seen_id, seen_obs = set(), set()
    split_of = {}
    per_taxon, per_split = Counter(), Counter()
    for r in rs:
        rid = r["plan_id"]
        if r["gbif_id"] in ids:
            err.append(f"{rid}: pilot image")
        if r["split"] == "test" and S.PC.normalise_photographer(r["photographer_display"]) in phot:
            err.append(f"{rid}: pilot photographer in test")
        if not r["inat_login"] or not r["photographer_key"].startswith("inat:"):
            err.append(f"{rid}: no account-level photographer identity")
        if split_of.setdefault("login:" + r["inat_login"], r["split"]) != r["split"]:
            err.append(f"{rid}: login in both splits")
        if split_of.setdefault("name:" + S.PC.normalise_photographer(r["photographer_display"]), r["split"]) != r["split"]:
            err.append(f"{rid}: display name in both splits")
        if not r["media_created"] or r["media_created"] < lo:
            err.append(f"{rid}: photo created {r['media_created']} before cut-off")
        if split_of.setdefault(r["photographer_key"], r["split"]) != r["split"]:
            err.append(f"{rid}: photographer in both splits")
        per_taxon[(r["split"], r["taxon"], r["photographer_key"])] += 1
        per_split[(r["split"], r["photographer_key"])] += 1
        if r["media_licence_uri"].lower().replace("https://", "http://").rstrip("/") + "/" not in \
                {u.rstrip("/") + "/" for u in cfg["licences"]["allowed_media_licence_uris"]}:
            err.append(f"{rid}: media licence {r['media_licence_uri']}")
        if not any(m in r["record_licence_uri"].lower() for m in cfg["licences"]["allowed_record_licence_markers"]):
            err.append(f"{rid}: record licence {r['record_licence_uri']}")
        if (r["nc_restricted"] == "True") != (r["media_licence"] == "CC BY-NC 4.0"):
            err.append(f"{rid}: NC flag inconsistent")
        if not (lo <= r["event_date"] <= hi):
            err.append(f"{rid}: date {r['event_date']}")
        if r["gbif_id"] in seen_id or r["observation_url"] in seen_obs:
            err.append(f"{rid}: duplicate record/observation")
        seen_id.add(r["gbif_id"])
        seen_obs.add(r["observation_url"])
        if r["taxon"] not in plan:
            err.append(f"{rid}: taxon not in plan")
        elif plan[r["taxon"]][0] == "unsupported" and r["taxon"] in vocab:
            err.append(f"{rid}: unsupported taxon in vocabulary")
        elif plan[r["taxon"]][0] == "supported_control" and str(vocab[r["taxon"]]) != r["gbif_taxon_key"]:
            err.append(f"{rid}: control key differs from vocabulary v1")
        a = r["attribution"]
        if not (r["photographer_display"] and r["observation_url"] in a and r["media_licence_uri"] in a and r["gbif_id"] in a):
            err.append(f"{rid}: incomplete attribution/provenance")
    err += [f"cap per photographer per taxon exceeded: {k}" for k, v in per_taxon.items() if v > cfg["caps"]["per_photographer_per_taxon"]]
    err += [f"cap per photographer per split exceeded: {k}" for k, v in per_split.items() if v > cfg["caps"]["per_photographer_per_split"]]
    return err


def minimums(rs: list[dict]) -> dict:
    m = S.config()["minimums"]
    prim = [r for r in rs if r["status"] == "primary"]

    def sel(**kw):
        return [r for r in prim if all(r[k] == v for k, v in kw.items())]

    def nph(x):
        return len({r["photographer_key"] for r in x})
    out = {}
    ud, ut = sel(role="unsupported", split="dev"), sel(role="unsupported", split="test")
    out["d2_dev_unsupported_images>=200"] = (len(ud), len(ud) >= m["d2_dev_unsupported_images"])
    out["d2_dev_unsupported_photographers>=100"] = (nph(ud), nph(ud) >= m["d2_dev_unsupported_photographers"])
    top = max(Counter(r["photographer_key"] for r in ud).values()) / len(ud) if ud else 1
    out["d2_dev_max_photographer_share<=5%"] = (round(top, 4), top <= 0.05)
    out["d2_dev_max_taxon_share<=15%"] = (round(max(Counter(r["taxon"] for r in ud).values()) / len(ud), 4) if ud else 1,
                                         (max(Counter(r["taxon"] for r in ud).values()) / len(ud) if ud else 1) <= 0.15)
    for g in S.config()["unsupported_groups"]:
        x = sel(group=g, split="dev")
        out[f"dev_group_{g}>=30img_15ph"] = ((len(x), nph(x)), len(x) >= m["per_group_min_dev"] and nph(x) >= 15)
    out["d2_test_unsupported_images>=450"] = (len(ut), len(ut) >= m["d2_test_unsupported_images"])
    cd, ct = sel(role="supported_control", split="dev", study="D2"), sel(role="supported_control", split="test", study="D2")
    cdc = cd + sel(study="C1", split="dev")
    out["supported_controls_dev>=100 (D2 controls; +C1 dev for coverage)"] = ((len(cd), len(cdc)), len(cd) >= m["supported_controls_dev"])
    out["supported_controls_test>=100"] = (len(ct), len(ct) >= m["supported_controls_test"])
    for sp, s, n_min, p_min in (("Cocos nucifera", "dev", m["c1_dev_per_species"], m["c1_dev_photographers_per_species"]),
                                ("Cocos nucifera", "test", m["c1_test_per_species"], m["c1_test_photographers_per_species"]),
                                ("Areca catechu", "dev", m["c1_dev_per_species"], m["c1_dev_photographers_per_species"]),
                                ("Areca catechu", "test", m["c1_test_per_species"], m["c1_test_photographers_per_species"])):
        x = sel(taxon=sp, split=s)
        out[f"C1_{sp}_{s}>={n_min}img_{p_min}ph"] = ((len(x), nph(x)), len(x) >= n_min and nph(x) >= p_min)
    return out


def name_token_pairs(rs: list[dict]) -> list[tuple[str, str]]:
    """Cross-split photographer names sharing a token of >= 4 letters: possible same person, needs review."""
    by = defaultdict(set)
    for r in rs:
        by[r["split"]].add(S.PC.normalise_photographer(r["photographer_display"]))

    def toks(k):
        return {t for t in re.split(r"[^a-z]+", k) if len(t) >= 4}
    return sorted((a, b) for a in by["dev"] for b in by["test"] if toks(a) & toks(b))


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    rs = rows()
    err = rule_errors(rs)
    mins = minimums(rs)
    pairs = name_token_pairs(rs)
    for e in err[:50]:
        print("  RULE FAIL", e)
    for k, (v, ok) in mins.items():
        print(f"  {'MET      ' if ok else 'SHORTFALL'} {k}: {v}")
    print(f"  cross-split photographer names sharing a token (manual review): {len(pairs)}")
    rep = {"rule_errors": err, "minimums": {k: {"value": v, "met": ok} for k, (v, ok) in mins.items()},
           "cross_split_name_token_pairs": len(pairs)}
    (P.OUT / "plan_validation.json").write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    print("D2 PLAN RULES:", "PASS" if not err else "FAIL")
    sys.exit(0 if not err else 1)


if __name__ == "__main__":
    main()

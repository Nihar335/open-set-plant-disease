"""English display text that comes from the server rather than the interface catalogue (SC-M5-02 localisation).

  python -m stage10.app.i18n_sources --write    # refresh i18n/_server_strings.json (the list every draft translates)
  python -m stage10.app.i18n_sources --check    # report how many server strings each language catalogue covers
  python -m stage10.app.i18n_sources --build    # i18n/drafts/<lang>.tsv -> i18n/<lang>.json (placeholders checked)

Draft format (one line each; separator TAB or ' = '; '#' lines are comments):
  ui:<interface key> TAB <translation>      e.g.  ui:home.start TAB फोटो जोड़ें
  kb:<index in _server_strings.json> TAB <translation>

Each catalogue in stage10/app/i18n/<lang>.json has a "kb" table {english text: translation}. The interface shows the
translation when one exists (js tk()) and otherwise the English source text, so a missing entry is never blank.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from stage10.app import farmer_summary as FS
from stage10.knowledge_base.kb import load_kb
from stage10.vision import quality as Q
from stage10.vision import symptoms as SY

I18N = Path(__file__).resolve().parent / "i18n"
OUT = I18N / "_server_strings.json"

# fixed sentences written by farmer_summary.build (kept in sync by test_server_strings_cover_summary_text)
SUMMARY_FIXED = [
    "Nothing in this report is a confirmed diagnosis. Confirm with the checks listed, or with your local KVK / agriculture officer.",
    "Spray products are never recommended here, and no doses are given: ask your local agriculture officer which product is registered and safe.",
    "The model matched this photo to a healthy leaf of this crop. That does not prove the plant is free of disease, pests or nutrient problems; "
    "check other leaves and the stem, and look again in a few days.",
    "No disease could be suggested from this photo and the information given.",
    "The photo matched one of the model's known disease classes closely enough.",
    "The photo did not match the model's training photos closely enough for a firm answer (this model was trained on laboratory leaf photos "
    "and rejects most field photos). Its closest matches are used only as possible causes.",
    "The photo disease model was not run for this plant.",
    "(source outside India; check locally)",
]


def server_strings() -> list[str]:
    kb = load_kb()
    out: list[str] = []

    def add(x):
        if x and isinstance(x, str) and x.strip() and x not in out:
            out.append(x)

    for p in kb.plants.values():
        add(p["common_names"][0]["name"])
    for c in kb.conditions.values():
        add(c["common_names"][0])
        add(FS.CAUSE.get(c["cause_category"], c["cause_category"]))
        for f in c["features"]:
            if f["role"] in ("distinguishing", "typical"):
                add(f["claim"]["text"])
        if c["curability"].get("claim"):
            add(c["curability"]["claim"]["text"])
    for f in kb.features.values():
        add(f["label"])
    for q in kb.questions.values():
        add(q["text"])
        for o in q["options"]:
            add(o["label"])
    for d in kb.diagnostics.values():
        for k in ("name", "description", "what_it_can_establish", "what_it_cannot_establish"):
            add(d[k])
    for m in kb.management.values():
        add(m["summary"])
    for r in kb.crop_requirements.values():
        for k in ("temperature", "soil"):
            if r[k]:
                add(r[k]["text"])
        if r["nutrients"]:
            add(r["nutrients"]["claim"]["text"])
    for v in FS.NUTRIENT_LABEL.values():
        add(v)
    for s in SY.SIGNS:
        add(SY.LABELS[s])
    add(SY.LIMITS)
    add("no leaf found: take a closer photo with one leaf filling most of the frame")
    for m in ("format not allowed (JPEG or PNG only)", "file is not a readable image", "photo is blurred", "photo is too dark",
              "photo is too bright"):
        add(m)
    for r in Q.RETAKE.values():
        add(r["view"])
        add(r["why"])
    for x in SUMMARY_FIXED:
        add(x)
    for x in ("the confirmed plant is outside the disease model's six crops: knowledge-based guidance",
              "the plant is not in the knowledge base: general guidance and expert referral",
              "the plant is not known: the disease model needs a confirmed supported crop",
              "the image failed the quality gate: retake the photo"):
        add(x)
    return out


DRAFTS = I18N / "drafts"
PH = re.compile(r"\{(\w+)\}")


def build_one(lang: str, langs: dict) -> tuple[dict, list[str]]:
    en = json.loads((I18N / "en.json").read_text(encoding="utf-8"))["strings"]
    src = json.loads(OUT.read_text(encoding="utf-8"))["strings"]
    ui, kbt, err = {}, {}, []
    for n, line in enumerate((DRAFTS / f"{lang}.tsv").read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        sep = chr(9) if chr(9) in line else " = "
        if sep not in line:
            err.append(f"{lang}:{n}: no separator (tab or ' = ')")
            continue
        k, v = line.split(sep, 1)
        k = k.strip()
        v = v.strip()
        if not v:
            continue
        if k.startswith("ui:"):
            key = k[3:]
            if key not in en:
                err.append(f"{lang}:{n}: unknown interface key {key}")
            elif set(PH.findall(en[key])) != set(PH.findall(v)):
                err.append(f"{lang}:{n}: placeholders differ for {key}")
            else:
                ui[key] = v
        elif k.startswith("kb:") and k[3:].isdigit() and int(k[3:]) < len(src):
            kbt[src[int(k[3:])]] = v
        else:
            err.append(f"{lang}:{n}: bad key {k}")
    meta = {"lang": lang, "name": langs.get(lang, {}).get("native", lang), "review_status": "machine_draft_unreviewed",
            "note": ("Computer-made draft translation (Claude), produced 2026-10-15 for SC-M5-02. NOT checked by a native or "
                     "fluent speaker. Shown with a draft notice; English is used wherever a string is missing."),
            "coverage": {"interface": f"{len(ui)}/{len(en)}", "server_text": f"{len(kbt)}/{len(src)}"}}
    return {"meta": meta, "strings": ui, "kb": kbt}, err


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--build", action="store_true")
    a = ap.parse_args()
    if a.build:
        import yaml

        cfg = yaml.safe_load((I18N.parents[1] / "config.yaml").read_text(encoding="utf-8"))
        langs = {x["code"]: x for x in cfg["language"]["ui_languages"]}
        bad = []
        for f in sorted(DRAFTS.glob("*.tsv")):
            cat, err = build_one(f.stem, langs)
            bad += err
            (I18N / f"{f.stem}.json").write_text(json.dumps(cat, indent=1, ensure_ascii=False) + chr(10), encoding="utf-8")
            print(f"{f.stem:4} interface {cat['meta']['coverage']['interface']}  server {cat['meta']['coverage']['server_text']}")
        for e in bad:
            print("ERROR", e)
        sys.exit(1 if bad else 0)
    strings = server_strings()
    if a.write:
        OUT.write_text(json.dumps({"note": "English server text; translate each into the 'kb' table of a language catalogue.",
                                   "strings": strings}, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {len(strings)} strings to {OUT.name}")
    if a.check or not a.write:
        en = json.loads((I18N / "en.json").read_text(encoding="utf-8"))["strings"]
        for f in sorted(I18N.glob("*.json")):
            if f.name.startswith("_") or f.stem == "en":
                continue
            c = json.loads(f.read_text(encoding="utf-8"))
            kbt = c.get("kb", {})
            print(f"{f.stem:4} ui {sum(k in c['strings'] for k in en)}/{len(en)}  server {sum(s in kbt for s in strings)}/{len(strings)}  "
                  f"status={c['meta'].get('review_status')}")


if __name__ == "__main__":
    main()

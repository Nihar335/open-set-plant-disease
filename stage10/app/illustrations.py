"""Picture-assisted questions (M5-D4): manifest and validator for the question diagrams.

  python -m stage10.app.illustrations            # (re)write static/illustrations/illustrations.json
  python -m stage10.app.illustrations --check    # validate files, hashes, licence records, alt text, questions

Every diagram is an ORIGINAL, simplified schematic created for AgroVaidya in this repository (plain SVG shapes; no
third-party images, fonts or traced material). They illustrate the wording of a question, never a diagnosis, and are
shown with the caption "Diagram - simplified, not a photo of your plant".
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DIR = HERE / "static" / "illustrations"
MANIFEST = DIR / "illustrations.json"
QUESTIONS = {"Q-CONCENTRIC-RINGS": "q-concentric-rings.svg", "Q-YELLOW-HALO": "q-yellow-halo.svg", "Q-LESION-SHAPE": "q-lesion-shape.svg",
             "Q-UNDERSIDE-GROWTH": "q-underside-growth.svg", "Q-BLACK-SPECKS": "q-black-specks.svg", "Q-LEAF-AGE": "q-leaf-age.svg",
             "Q-INTERVEINAL": "q-interveinal.svg", "Q-FRUIT": "q-fruit.svg", "Q-WEBBING-BRONZING": "q-webbing-bronzing.svg"}
LICENCE = {"type": "project-original",
           "statement": "Original schematic created for the AgroVaidya project on 2026-10-15 from basic SVG shapes; contains no "
                        "third-party material. Rights held by the project owner; distributed only as part of AgroVaidya.",
           "third_party_material": False}
CAPTION = "Diagram - simplified, not a photo of your plant. Real symptoms vary."


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build() -> dict:
    items = []
    for q, f in sorted(QUESTIONS.items()):
        t = (DIR / f).read_text(encoding="utf-8")
        alt = re.search(r"<title>(.*?)</title>", t).group(1)
        items.append({"question_id": q, "file": f, "alt": alt, "caption": CAPTION, "sha256": sha(DIR / f), "licence": LICENCE,
                      "created": "2026-10-15", "author": "AgroVaidya project (generated in-repository)",
                      "accuracy_review": "pending: no plant-health expert has reviewed the diagrams (M8-09)"})
    return {"note": "Question diagrams; the questions without a diagram are answered by text, tap or voice.", "items": items}


def errors(kb=None) -> list[str]:
    err = []
    if not MANIFEST.exists():
        return ["illustrations.json missing"]
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if m != build():
        err.append("illustrations.json is stale")
    for it in m["items"]:
        p = DIR / it["file"]
        t = p.read_text(encoding="utf-8")
        if re.search(r"<image|href=|xlink|https?://(?!www\.w3\.org/2000/svg)|<script|<foreignObject|font-face", t, re.I):
            err.append(f"{it['file']}: external reference, embedded image or script")
        if not it["alt"] or len(it["alt"]) < 15:
            err.append(f"{it['file']}: missing descriptive alternative text")
        if not it["licence"]["statement"] or it["licence"]["third_party_material"]:
            err.append(f"{it['file']}: licence not recorded")
    if kb is not None:
        for q in QUESTIONS:
            if q not in kb.questions:
                err.append(f"{q}: not a knowledge-base question")
        for q, rec in kb.questions.items():
            if not any(o["option_id"] in ("not_sure", "not_checked") or not o["sets"] for o in rec["options"]):
                err.append(f"{q}: no 'not sure' option")
    return err


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if not a.check:
        MANIFEST.write_text(json.dumps(build(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    from stage10.knowledge_base.kb import load_kb
    err = errors(load_kb())
    for e in err:
        print("ERROR", e)
    print(f"ILLUSTRATIONS: {'PASS' if not err else 'FAIL'} ({len(QUESTIONS)} diagrams)")
    sys.exit(1 if err else 0)


if __name__ == "__main__":
    main()

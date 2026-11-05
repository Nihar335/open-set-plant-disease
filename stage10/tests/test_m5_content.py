"""M5 content checks: picture-assisted question diagrams (M5-D4), draft local plant names (M5-C2), manual-check records."""
from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

from stage10.app import illustrations as IL
from stage10.app import server as SV
from stage10.knowledge_base.kb import load_kb
from stage10.workflow import plant_name as PN

ROOT = Path(__file__).resolve().parents[2]
KB = load_kb()


def test_every_diagram_has_licence_provenance_hash_and_alt_text():
    assert IL.errors(KB) == []
    m = json.loads(IL.MANIFEST.read_text(encoding="utf-8"))
    assert len(m["items"]) == 9
    for it in m["items"]:
        assert it["licence"]["type"] == "project-original" and it["licence"]["third_party_material"] is False
        assert "pending" in it["accuracy_review"] and "not a photo" in it["caption"]


def test_diagrams_are_plain_shapes_without_external_or_embedded_content():
    for f in IL.DIR.glob("*.svg"):
        t = f.read_text(encoding="utf-8")
        assert not re.search(r"<image|href=|<script|<foreignObject|https?://(?!www\.w3\.org/2000/svg)", t, re.I), f.name
        assert "<title>" in t


def test_every_question_has_a_not_sure_option_and_api_serves_diagrams():
    for q, rec in KB.questions.items():
        assert any(o["option_id"] in ("not_sure", "not_checked") for o in rec["options"]), q
    c = SV.create_app(osr_factory=object, kb=KB).test_client()
    qs = c.get("/api/v1/symptoms").json["data"]["questions"]
    with_fig = [q for q in qs if q["illustration"]]
    assert len(with_fig) == 9 and all(c.get("/" + q["illustration"]["src"]).status_code == 200 for q in with_fig)
    assert all(q["illustration"] is None for q in qs if q["question_id"] in ("Q-WEATHER", "Q-SINGLE-SHOOT", "Q-WOOD-SECTION"))


def test_draft_local_names_are_sourced_but_never_used_until_reviewed():
    data = yaml.safe_load((ROOT / "stage10/workflow/local_aliases.yaml").read_text(encoding="utf-8"))
    assert PN.local_alias_errors(data["aliases"], KB) == []
    assert all(a["review_status"] == "draft" and a["reference"]["url"].startswith("https://www.wikidata.org/") for a in data["aliases"])
    assert PN.reviewed_local_aliases(KB) == []
    for a in data["aliases"]:
        assert PN.resolve_plant_name(a["alias"], KB)["status"] in ("unknown", "needs_clarification")   # no silent mapping


def test_manual_checks_are_pending_not_fabricated():
    tpl = json.loads((ROOT / "stage10/app/MANUAL_VERIFICATION_RESULTS.template.json").read_text(encoding="utf-8"))
    assert all(v["result"] == "not_run" for v in tpl["results"].values()) and len(tpl["results"]) == 38
    res = ROOT / "stage10/app/MANUAL_VERIFICATION_RESULTS.json"
    tracker = {x["task_id"]: x for x in json.loads((ROOT / "stage10/project_tracker.json").read_text(encoding="utf-8"))["tasks"]}
    if not res.exists():                                    # no human results recorded -> hardware/human-gated tasks unverified
        for t in ("M5-A1", "M5-A7", "M5-D5", "M5-R1"):           # camera, touchscreen, screen reader, mic/speaker acceptance
            assert tracker[t]["status"] != "verified", t

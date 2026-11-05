"""Load and validate the AgroVaidya knowledge base (stage10/knowledge_base/data/*.yaml).

`load_kb()` returns a validated `KnowledgeBase`; `integrity_errors(raw)` lists every problem found. Checks:
  * JSON-Schema validation of every record (kb_records.schema.json);
  * unique IDs within and across record types;
  * every cross-reference resolves (sources, plants, features, conditions, diagnostics, management, questions, model classes);
  * condition <-> management links are consistent in both directions;
  * no placeholder / test / non-https sources; every source has a cached verification copy whose bytes hash matches;
  * every evidence passage occurs verbatim (whitespace-normalised) in the cached source text;
  * our own wording (claim text, management summaries) contains no dose / rate / concentration;
  * every question offers a "not sure" option that sets no feature;
  * model-class mapping equals the frozen protocol's 21 known + 5 hard-unknown classes;
  * crop requirements (temperature, soil, nutrients) reference known plants and state no doses.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml
from jsonschema import Draft202012Validator
from referencing import Registry, Resource

KB_DIR = Path(__file__).resolve().parent
ROOT = KB_DIR.parents[1]
DATA = KB_DIR / "data"
SCHEMA = KB_DIR / "schemas" / "kb_records.schema.json"
PLACEHOLDER = re.compile(r"example\.(invalid|com|org)|localhost|127\.0\.0\.1|placeholder|lorem", re.I)
DOSE = re.compile(r"(\d+(\.\d+)?\s*%)|(@\s*\d)|(\d+(\.\d+)?\s*(g|kg|mg|ml|l|lit(re|er)?s?|t|ppm)\s*(/|per)\s*"
                  r"(l|lit(re|er)?|ha|hectare|ac(re)?|kg|plant|tree|10\s*l))", re.I)


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", t.replace("\xa0", " ")).strip()


def _y(name):
    return yaml.safe_load((DATA / name).read_text(encoding="utf-8"))


def load_raw() -> dict:
    mg = _y("management.yaml")
    return {"sources": _y("sources.yaml"), "plants": _y("plants.yaml"), "features": _y("features.yaml"),
            "questions": _y("questions.yaml"), "conditions": _y("conditions.yaml"), "diagnostics": _y("diagnostics.yaml"),
            "management": mg["records"], "model_classes": _y("model_classes.yaml"),
            "crop_requirements": _y("crop_requirements.yaml"),
            "version": yaml.safe_load((KB_DIR / "KB_VERSION.yaml").read_text(encoding="utf-8"))}


@lru_cache(maxsize=1)
def _validators() -> dict:
    s = json.loads(SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(s)
    reg = Registry().with_resource(s["$id"], Resource.from_contents(s))
    return {k: Draft202012Validator({"$ref": f"{s['$id']}#/$defs/{k}"}, registry=reg)
            for k in ("source", "plant", "feature", "question", "condition", "diagnostic", "management", "model_class",
                      "crop_requirement")}


TYPE_OF = {"sources": ("source", "source_id"), "plants": ("plant", "plant_id"), "features": ("feature", "feature_id"),
           "questions": ("question", "question_id"), "conditions": ("condition", "condition_id"),
           "diagnostics": ("diagnostic", "diagnostic_id"), "management": ("management", "management_id"),
           "model_classes": ("model_class", "class_name"),
           "crop_requirements": ("crop_requirement", "crop_requirement_id")}


def _claims(obj):
    """Yield every claim dict ({text, evidence}) nested anywhere in obj."""
    if isinstance(obj, dict):
        if "text" in obj and "evidence" in obj:
            yield obj
        for v in obj.values():
            yield from _claims(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _claims(v)


def _evidence(obj):
    if isinstance(obj, dict):
        if set(obj) >= {"source_id", "passage"}:
            yield obj
        for v in obj.values():
            yield from _evidence(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _evidence(v)


def integrity_errors(raw: dict, *, check_cache: bool = True) -> list[str]:
    err: list[str] = []
    V = _validators()
    ids: dict[str, set] = {}
    all_ids: dict[str, str] = {}
    for key, (typ, idf) in TYPE_OF.items():
        ids[key] = set()
        for r in raw[key]:
            for e in V[typ].iter_errors(r):
                err.append(f"{key}/{r.get(idf, '?')}: schema: {'/'.join(map(str, e.absolute_path))}: {e.message[:160]}")
            rid = r.get(idf)
            if rid in ids[key]:
                err.append(f"{key}: duplicate id {rid}")
            if key != "model_classes" and rid in all_ids and all_ids[rid] != key:
                err.append(f"id {rid} used in both {all_ids[rid]} and {key}")
            ids[key].add(rid)
            all_ids.setdefault(rid, key)
    src = {s["source_id"]: s for s in raw["sources"]}

    def need(kind, i, where):
        if i not in ids[kind]:
            err.append(f"{where}: unknown {kind[:-1]} reference {i}")

    # sources
    for s in raw["sources"]:
        w = f"sources/{s['source_id']}"
        if PLACEHOLDER.search(s.get("url", "")) or PLACEHOLDER.search(s.get("title", "")) or s.get("source_type") == "test_fixture":
            err.append(f"{w}: placeholder/test source not allowed in the knowledge base")
        if check_cache:
            cf = ROOT / s["verification"]["cache_file"]
            raw_f = cf.with_suffix(".raw")
            if not cf.exists() or not raw_f.exists():
                err.append(f"{w}: verification cache missing ({cf.name})")
            elif hashlib.sha256(raw_f.read_bytes()).hexdigest() != s["verification"]["bytes_sha256"]:
                err.append(f"{w}: cached bytes hash differs from recorded hash")
    # evidence passages
    cache_text = {}
    for key in ("plants", "conditions", "diagnostics", "management", "crop_requirements"):
        for r in raw[key]:
            for ev in _evidence(r):
                sid = ev["source_id"]
                if sid not in src:
                    err.append(f"{key}/{r.get(TYPE_OF[key][1])}: unknown source {sid}")
                    continue
                if check_cache:
                    if sid not in cache_text:
                        p = ROOT / src[sid]["verification"]["cache_file"]
                        cache_text[sid] = norm(p.read_text(encoding="utf-8")) if p.exists() else None
                    if cache_text[sid] is not None and norm(ev["passage"]) not in cache_text[sid]:
                        err.append(f"{key}/{r.get(TYPE_OF[key][1])}: passage not found in {sid}: {ev['passage'][:70]!r}")
            for c in _claims(r):
                if DOSE.search(c["text"]):
                    err.append(f"{key}/{r.get(TYPE_OF[key][1])}: dose-like text in claim wording: {c['text'][:80]!r}")
    for m in raw["management"]:
        if DOSE.search(m["summary"]):
            err.append(f"management/{m['management_id']}: dose-like text in summary")
    # features / questions
    for q in raw["questions"]:
        if not any(o["sets"] == {} and "not" in o["option_id"] for o in q["options"]):
            err.append(f"questions/{q['question_id']}: needs a 'not sure' option that sets nothing")
        for o in q["options"]:
            for f in o["sets"]:
                need("features", f, f"questions/{q['question_id']}")
    # conditions
    for c in raw["conditions"]:
        w = f"conditions/{c['condition_id']}"
        for a in c["affected_plants"]:
            need("plants", a["plant_id"], w)
        for p in c.get("unsourced_host_plants", []):
            need("plants", p, w)
        for f in c["features"] + c["favourable_conditions"]:
            need("features", f["feature_id"], w)
        for la in c["look_alikes"]:
            need("conditions", la["condition_id"], w)
            if la["condition_id"] == c["condition_id"]:
                err.append(f"{w}: look-alike of itself")
        for d in c["diagnostic_ids"]:
            need("diagnostics", d, w)
        for m in c["management_ids"]:
            need("management", m, w)
            mrec = next((x for x in raw["management"] if x["management_id"] == m), None)
            if mrec and c["condition_id"] not in mrec["applies_to_conditions"]:
                err.append(f"{w}: management {m} does not list this condition")
    for m in raw["management"]:
        for cid in m["applies_to_conditions"]:
            need("conditions", cid, f"management/{m['management_id']}")
            crec = next((x for x in raw["conditions"] if x["condition_id"] == cid), None)
            if crec and m["management_id"] not in crec["management_ids"]:
                err.append(f"management/{m['management_id']}: condition {cid} does not list this practice")
    for d in raw["diagnostics"]:
        for f in d["features_checked"]:
            need("features", f, f"diagnostics/{d['diagnostic_id']}")
    for cr in raw["crop_requirements"]:
        need("plants", cr["plant_id"], f"crop_requirements/{cr['crop_requirement_id']}")
    # model classes vs frozen protocol (read-only)
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))["osr_protocol"]
    expected = {(k, "frozen_model_known_class") for k in cfg["known_classes"]} | {(k, "held_out_hard_unknown") for k in cfg["unknown_near"]}
    got = {(m["class_name"], m["research_role"]) for m in raw["model_classes"]}
    if got != expected:
        err.append(f"model_classes differ from frozen protocol: missing={sorted(expected - got)} extra={sorted(got - expected)}")
    for m in raw["model_classes"]:
        need("plants", m["plant_id"], f"model_classes/{m['class_name']}")
        if m["condition_id"]:
            need("conditions", m["condition_id"], f"model_classes/{m['class_name']}")
            crec = next((x for x in raw["conditions"] if x["condition_id"] == m["condition_id"]), None)
            if crec and crec["research_role"] != m["research_role"]:
                err.append(f"model_classes/{m['class_name']}: research_role mismatch with {m['condition_id']}")
    return err


@dataclass
class KnowledgeBase:
    raw: dict
    sources: dict = field(default_factory=dict)
    plants: dict = field(default_factory=dict)
    features: dict = field(default_factory=dict)
    questions: dict = field(default_factory=dict)
    conditions: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)
    management: dict = field(default_factory=dict)
    crop_requirements: dict = field(default_factory=dict)          # plant_id -> record
    version: str = ""

    def __post_init__(self):
        r = self.raw
        self.sources = {x["source_id"]: x for x in r["sources"]}
        self.plants = {x["plant_id"]: x for x in r["plants"]}
        self.features = {x["feature_id"]: x for x in r["features"]}
        self.questions = {x["question_id"]: x for x in r["questions"]}
        self.conditions = {x["condition_id"]: x for x in r["conditions"]}
        self.diagnostics = {x["diagnostic_id"]: x for x in r["diagnostics"]}
        self.management = {x["management_id"]: x for x in r["management"]}
        self.crop_requirements = {x["plant_id"]: x for x in r.get("crop_requirements", [])}
        self.version = r["version"]["kb_version"]

    def resolve_plant(self, name: str | None) -> str | None:
        """Plant id from a common name, synonym or scientific name (case-insensitive); None if not in the KB."""
        if not name:
            return None
        n = name.strip().lower()
        for pid, p in self.plants.items():
            names = {x["name"].lower() for x in p["common_names"]} | {s.lower() for s in p.get("synonyms", [])}
            names |= {p["scientific_name"].lower(), p["taxonomy"]["species"].lower(), pid.lower()}
            if n in names:
                return pid
        return None

    def sourced_conditions(self):
        return [c for c in self.conditions.values() if c["curation_status"] == "sourced"]


class KnowledgeBaseError(ValueError):
    pass


def load_kb(*, check_cache: bool = True) -> KnowledgeBase:
    raw = load_raw()
    errs = integrity_errors(raw, check_cache=check_cache)
    if errs:
        raise KnowledgeBaseError(f"{len(errs)} knowledge-base errors, e.g.: " + " | ".join(errs[:5]))
    return KnowledgeBase(raw)

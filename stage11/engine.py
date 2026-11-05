"""Stage 11 diagnosis engine: photo + plant name + location -> report.

The prediction combines the three inputs (rule in stage11/decision.py, parameters fitted on validation):
  * photo     -> image model (43 disease classes) + symptom head (25 symptoms) + Grad-CAM heat map;
  * plant     -> restricts the prediction to that crop's diseases (botanical database);
  * location  -> weather of the last 7 days (Open-Meteo) gives each disease a weather-risk level. It is reported,
                 not used in the score: the hand-set factor was never validated (audit 2026-10-23).
If the top disease reaches THRESHOLD (60%), the crop is "diagnosable" by the image model (its own healthy AND disease classes
were trained; KB.model_coverage) and the photo does not look more like another crop, it is the predicted disease. Otherwise the report says "Unknown disease" and gives
the three diseases whose symptoms best match, each with tests and solutions. Plants without training photos use
symptom matching only and are never named (decision basis must be the image), so they always come out as "Unknown disease" with three leads.
"""
from __future__ import annotations

import base64
import io
import json
import threading
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import transforms

from stage11 import weather as wx
from stage11.decision import THRESHOLD, crop_check, fuse, may_name, softmax, symptom_match
from stage11.kb import load_kb
from stage11.model import EVAL_TF, IMG_SIZE, LeafNet

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "stage11"
SYMPTOM_THRESHOLD = 0.5
SYMPTOM_MATCH_CAP = 0.75  # symptom-match-only results never exceed this; they are never named (basis check)
# The photo is also scored against ALL trained classes. If another crop takes at least this share of the image
# probability, the app SUGGESTS that crop (user request 2026-10-23: a tomato leaf entered as "squash" should point to
# tomato) but never replaces the user's name by itself. An automatic switch was tried and removed the same day: a
# downy-mildew photo entered as "tomato" was switched to potato early blight at 97%. Validation calib sets
# (stage11_local/audit/v2_*_calib.npz): with no gate 25% of photos of plants the model never saw (and 17% of textures)
# reach >= 0.90 on some crop, and no score separates them from real ones (max-logit / energy / symptom-match at the
# 5th percentile of correct field photos still let 22-23% through; the downy-mildew photo scored max-logit 3.93, above
# a genuine tomato photo at 2.84). On known field photos crop mass >= 0.90 is right 96.5% of the time (75% coverage).
CROP_SUGGEST_MASS = 0.90
CROP_TF = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(IMG_SIZE)])  # EVAL_TF without tensor
DISCLAIMER = ("Chemicals are active ingredients only (no doses). Use only CIBRC-registered products for this crop, "
              "follow the label and your State Agricultural University spray schedule, and confirm with your KVK.")


class Engine:
    def __init__(self, device: str | None = None):
        self.kb = load_kb()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(MODEL_DIR / "best.pt", map_location="cpu", weights_only=False)
        assert ckpt["classes"] == [d["key"] for d in self.kb.model_classes], "checkpoint / diseases.yaml mismatch"
        assert ckpt["symptoms"] == self.kb.symptom_ids, "checkpoint / symptoms.yaml mismatch"
        self.model = LeafNet(len(ckpt["classes"]), len(ckpt["symptoms"]), pretrained=False)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device).eval()
        calib_path = MODEL_DIR / "calibration.json"
        self.cal = json.loads(calib_path.read_text()) if calib_path.exists() else {"temperature": 1.0, "beta": 0.0,
                                                                                   "m0": 0.0}
        self.checkpoint_epoch = ckpt["epoch"]
        self.lock = threading.Lock()
        self.classes = self.kb.model_classes
        self.class_crops = np.array([d["crop"] for d in self.classes])
        self.profiles = self.kb.symptom_targets()
        ev = ROOT / "stage11" / "results" / "multicrop_eval.json"  # measured field reliability, disclosed per crop
        self.field_eval = json.loads(ev.read_text(encoding="utf-8")).get("per_crop", {}) if ev.exists() else {}

    # ---------------------------------------------------------------- model forward + Grad-CAM
    def _forward(self, img: Image.Image):
        x = EVAL_TF(img).unsqueeze(0).to(self.device)
        self.model.zero_grad(set_to_none=True)
        logits, sym, fmap = self.model(x, return_map=True)
        fmap.retain_grad()
        return logits[0], torch.sigmoid(sym[0]).detach().cpu().numpy(), fmap

    def _gradcam(self, logits, fmap, cls: int, img: Image.Image) -> str:
        logits[cls].backward(retain_graph=True)
        g = fmap.grad[0].mean((1, 2), keepdim=True)
        cam = torch.relu((g * fmap[0]).sum(0)).detach().cpu().numpy()
        cam = cam / (cam.max() + 1e-8)
        base = np.array(CROP_TF(img))  # exactly the region the model saw (Resize 256 + centre crop 224)
        heat = cv2.resize(cam, (224, 224), interpolation=cv2.INTER_CUBIC)
        colour = cv2.applyColorMap(np.uint8(255 * np.clip(heat, 0, 1)), cv2.COLORMAP_JET)[:, :, ::-1]
        overlay = np.uint8(0.55 * base + 0.45 * colour)
        buf = io.BytesIO()
        Image.fromarray(overlay).resize((320, 320), Image.BICUBIC).save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    # ---------------------------------------------------------------- helpers
    def _symptom_list(self, probs):
        out = [{"id": s["id"], "name": s["name"], "description": s["description"], "probability": round(float(p), 3)}
               for s, p in zip(self.kb.symptoms, probs)]
        return sorted(out, key=lambda r: -r["probability"])

    def _seen(self, key, sym_probs):
        seen, missing = [], []
        for sid, w in self.kb.by_key[key]["symptoms"].items():
            p = float(sym_probs[self.kb.symptom_ids.index(sid)])
            name = self.kb.symptoms[self.kb.symptom_ids.index(sid)]["name"]
            (seen if p >= SYMPTOM_THRESHOLD else missing).append({"name": name, "weight": w, "probability": round(p, 2)})
        return seen, missing

    def _where_to_test(self, state, crop):
        res = self.kb.resources
        st = res["states"].get(state["name"], {}) if state else {}
        out = list(st.get("labs", []))
        inst = res["crop_institutes"].get(crop)
        acronym = inst.split("(")[-1].split(")")[0] if inst and "(" in inst else None
        if inst and not any(acronym and acronym in lab for lab in out):
            out.append(inst)
        out += [res["national"]["kvk"], res["national"]["helpline"]]
        return out

    def _card(self, key, conf, sym_probs, w, state, basis, image_prob=None, match=None):
        d = self.kb.by_key[key]
        seen, missing = self._seen(key, sym_probs)
        fav = wx.favourability(d.get("weather"), w)
        sol = self.kb.solutions.get(key, {"cultural": [d.get("management")] if d.get("management") else [],
                                          "biological": [], "chemical": []})
        schedule = (self.kb.resources["states"].get(state["name"], {}).get("schedules", {}).get(d["crop"])
                    if state else None)
        return {
            "key": key, "name": d["name"], "crop": d["crop"], "confidence": round(float(conf), 4),
            "image_probability": None if image_prob is None else round(float(image_prob), 4),
            "symptom_match": None if match is None else round(float(match), 3),
            "basis": basis, "healthy": d["pathogen_type"] == "none",
            "pathogen": d.get("pathogen"), "pathogen_type": d["pathogen_type"],
            "symptoms_seen": seen, "symptoms_not_seen": missing, "weather_risk": fav,
            "look_alikes": [self.kb.by_key[k]["name"] for k in d.get("look_alikes", [])],
            "tests": [self.kb.tests[t] for t in d["tests"]],
            "where_to_test": [] if d["pathogen_type"] == "none" else self._where_to_test(state, d["crop"]),
            "solution": {"cultural": sol.get("cultural", []), "biological": sol.get("biological", []),
                         "chemical": sol.get("chemical", []), "state_schedule": schedule},
        }

    def _location(self, state_name, place):
        state = self.kb.state(state_name) if state_name else None
        if not state:
            return None, None, None
        geo = wx.geocode(place, state["name"]) if place else None
        if geo:
            loc = {"label": f"{geo['name']}" + (f", {geo['district']}" if geo.get("district") else "") +
                   f", {state['name']}", "lat": geo["lat"], "lon": geo["lon"], "precision": "place"}
        else:
            loc = {"label": f"{state['capital']} (state capital), {state['name']}", "lat": state["lat"],
                   "lon": state["lon"], "precision": "state_capital",
                   "place_not_found": bool(place)}
        return state, loc, wx.fetch(loc["lat"], loc["lon"])

    def _suggest_crop(self, plant, plant_text, lg, notes):
        """Score the photo against all trained crops; suggest (never apply) another crop that the photo clearly matches."""
        p_full = softmax(lg / self.cal["temperature"])
        mass = {c: float(p_full[self.class_crops == c].sum()) for c in dict.fromkeys(self.class_crops)}
        best = max(mass, key=mass.get)
        stated_keys = self.kb.crop_keys(plant["model_crop"]) if plant else []
        if best in stated_keys or mass[best] < CROP_SUGGEST_MASS:
            return None
        target = next((p for p in self.kb.plants if p["model_crop"] == best and p["common_name"].lower() == best),
                      next((p for p in self.kb.plants if p["model_crop"] == best), None))
        if target is None:
            return None
        top = self.classes[int(p_full.argmax())]
        entered = plant["common_name"] if plant else plant_text
        notes.append(f"Plant check: you entered {entered}, but the photo looks more like {target['common_name']} "
                     f"({100 * mass[best]:.0f}% of the image probability; closest trained disease: {top['name']}). "
                     f"Your plant name was kept and no disease is named, because the model knows only "
                     f"{len(self.classes)} conditions of {len(mass)} crops and photos of plants or diseases it was never "
                     f"trained on (e.g. downy mildew) can match a trained crop just as confidently. If the plant really "
                     f"is {target['common_name']}, diagnose it again with that name.")
        return {"entered": plant_text, "suggested": target["common_name"], "mass": round(mass[best], 4),
                "closest_disease": top["name"], "stated_mass": round(sum(mass.get(k, 0.0) for k in stated_keys), 4)}

    # ---------------------------------------------------------------- main entry point
    def diagnose(self, image_bytes: bytes, plant_text: str, state_name: str, place: str = "") -> dict:
        with self.lock:  # one request at a time on the model (Grad-CAM uses gradients)
            return self._diagnose(image_bytes, plant_text, state_name, place)

    def _diagnose(self, image_bytes, plant_text, state_name, place):
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGB")
        plant, match_type = self.kb.resolve_plant(plant_text)
        state, loc, w = self._location(state_name, place)
        logits, sym_probs, fmap = self._forward(img)
        lg = logits.detach().cpu().numpy()
        cam_cls = int(lg.argmax())
        notes = []
        suggestion = self._suggest_crop(plant, plant_text, lg, notes)

        crop_keys = self.kb.crop_keys(plant["model_crop"]) if plant else []
        in_crop = np.isin(self.class_crops, crop_keys)
        coverage = self.kb.model_coverage(plant["model_crop"] if plant else None)
        check, fit = None, 1.0

        if in_crop.any():
            basis = "image"
            out = fuse(lg, sym_probs, in_crop, self.profiles, T=self.cal["temperature"], beta=self.cal["beta"],
                       m0=self.cal["m0"])
            conf, p_crop, match = out["conf"][0], out["p_crop"][0], out["match"][0]
            check = crop_check(out["p_full"][0], self.class_crops, in_crop)
            basis_txt = "image + symptoms" if out["fit"][0] >= 1 else "symptoms + image"
            in_mass, fit = float(out["in_mass"][0]), float(out["fit"][0])  # noqa: F841 (fit used for reasons)
            order = [i for i in np.argsort(-conf) if in_crop[i]]
            ranked = [self._card(self.classes[i]["key"], conf[i], sym_probs, w, state, basis_txt,
                                 p_crop[i], match[i]) for i in order]
            if check["conflict"] and not suggestion:  # the suggestion note already explains the disagreement
                notes.append(f"Crop check: the photo looks more like {check['most_likely_crop']} "
                             f"({100 * check['most_likely_mass']:.0f}% of the image probability) than "
                             f"{plant['common_name']} ({100 * check['stated_crop_mass']:.0f}%). Please check the plant "
                             "name; no disease is named while the two disagree.")
            elif fit < 1:
                p_full = out["p_full"][0]
                other = self.classes[int(p_full.argmax())]
                notes.append(f"Only {100 * in_mass:.0f}% of the image model's probability falls on {plant['common_name']} "
                             f"diseases it knows (the photo looks most like {other['crop']} — {other['name']}). "
                             "So the ranking relies more on the symptoms detected in the photo, matched against the disease database, "
                             "so the result is reported as leads.")
            # database-only diseases of this crop (no training photos) as extra leads
            for d in self.kb.diseases:
                if d["model_class"] is None and d["crop"] in crop_keys:
                    m = float(symptom_match(sym_probs[None], self.kb.profile(d["key"])[None])[0, 0])
                    if m >= 0.6:
                        ranked.append(self._card(d["key"], m * SYMPTOM_MATCH_CAP * 0.5, sym_probs, w, state,
                                                 "symptom match only", None, m))
            ranked.sort(key=lambda c: -c["confidence"])
            if coverage["level"] == "limited":
                kinds = "healthy" if not coverage["diseases"] else "diseased (" + ", ".join(
                    self.kb.by_key[k]["name"] for k in coverage["diseases"]) + ")"
                notes.append(f"Limited model coverage: the image model has seen only {kinds} {plant['common_name']} "
                             "leaves, so it cannot tell other conditions apart and no result is named. "
                             "The list below gives leads only.")
            elif coverage["level"] == "related":
                notes.append(f"No {plant['common_name']} photos were used to train the image model. The diseases "
                             f"below come from a related crop ({', '.join(coverage['related_crops'])}) and are leads "
                             "only; no result is named.")
        else:
            basis = "symptom_match"
            crop = plant["model_crop"] if plant else None
            keys = [d["key"] for d in self.kb.diseases if d["pathogen_type"] != "none" and
                    ((crop and d["crop"] in self.kb.crop_keys(crop)) or d["crop"] == "any" or not plant)]
            notes.append((f"The image model was not trained on {plant['common_name']} photos. " if plant else
                          "This plant is not in the botanical database. ") +
                         "Symptoms detected in the photo were matched against the disease database, "
                         "so the result is reported as leads.")
            m = symptom_match(sym_probs[None], np.stack([self.kb.profile(k) for k in keys]))[0]
            s = np.exp(6 * m)
            healthy_p = float(sym_probs[self.kb.symptom_ids.index("no_visible_symptoms")])
            s = s / s.sum() * (1 - healthy_p) * SYMPTOM_MATCH_CAP
            ranked = sorted([self._card(k, c, sym_probs, w, state, "symptom match only", None, mm)
                             for k, c, mm in zip(keys, s, m)], key=lambda c: -c["confidence"])

        top = ranked[0]
        # Why a result is not named (several can apply); shown to the user and used by the evaluation.
        reasons = []
        if not plant:
            reasons.append("plant_not_in_database")
        elif coverage["level"] != "diagnosable":
            reasons.append(f"model_coverage_{coverage['level']}")
        if check and check["conflict"]:
            reasons.append("crop_conflict")
        elif basis == "image" and fit < 1:
            reasons.append("low_photo_fit")
        if top["confidence"] < THRESHOLD:
            reasons.append("below_threshold")
        if basis == "image" and top["basis"] != "symptom match only" and may_name(
                top["confidence"], coverage["level"], bool(check and check["conflict"])):
            verdict = "healthy" if top["healthy"] else "predicted"
            shown = [top]
            fe = self.field_eval.get(plant["model_crop"], {}).get("field", {})
            acc, n = (fe.get("v3") or {}).get("named_acc"), fe.get("n", 0)
            if acc is not None and n >= 10 and acc < 0.8:
                notes.append(f"Field reliability warning: on {n} independent field photos of {plant['common_name']}, "
                             f"only {100 * acc:.0f}% of named results were correct. Confirm with the tests below "
                             "before treating.")
        else:
            verdict = "unknown"
            shown = [c for c in ranked if not c["healthy"]][:3]
            healthy = next((c for c in ranked if c["healthy"]), None)
            if healthy and healthy["confidence"] >= 0.3:
                notes.append(f"The model also gives {100 * healthy['confidence']:.0f}% to a healthy leaf.")
            if not shown:
                notes.append("No disease of this crop with matching symptoms is in the database; "
                             "please show the leaf to your KVK or State Agricultural University.")

        detected = [s for s in self._symptom_list(sym_probs)
                    if s["probability"] >= SYMPTOM_THRESHOLD and s["id"] != "no_visible_symptoms"]
        result = {
            "plant": None if not plant else {
                "query": plant_text, "match": match_type, "common_name": plant["common_name"],
                "scientific_name": plant["scientific_name"], "family": plant["family"],
                "model_supported": coverage["level"] == "diagnosable",
                "model_coverage": coverage["level"],
            },
            "crop_suggestion": suggestion,
            "crop_check": check,
            "state": None if not state else {"name": state["name"]},
            "location": loc,
            "weather": w,
            "weather_note": (None if w else "Weather service unreachable; weather was not used." if state
                             else "State not recognised; weather was not used."),
            "symptoms_detected": detected,
            "symptoms_all": self._symptom_list(sym_probs),
            # Map of the model's own strongest class over ALL crops: a class the model gives ~0% (e.g. the top disease of
            # a wrongly named crop) has almost no evidence on the leaf, and its map lit up background corners.
            "heatmap": self._gradcam(logits, fmap, cam_cls, img),
            "heatmap_class": {"name": self.classes[cam_cls]["name"], "crop": self.classes[cam_cls]["crop"]},
            "verdict": verdict,
            "unknown_reasons": [] if verdict != "unknown" else reasons,
            "threshold": THRESHOLD,
            "basis": basis,
            "candidates": shown,
            "notes": notes,
            "model": {"checkpoint_epoch": self.checkpoint_epoch, **{k: self.cal.get(k) for k in
                                                                      ("temperature", "beta", "m0")}},
            "disclaimer": DISCLAIMER,
        }
        result["report"] = build_report(result, plant_text)
        return result


# ---------------------------------------------------------------- the farmer report (fixed format)
def _pct(v):
    return f"{100 * v:.0f}%"


def _weather_line(r):
    w, loc = r["weather"], r["location"]
    if not w:
        return r["weather_note"]
    return (f"Weather at {loc['label']}: last 7 days {w['week_mean_temp_c']} °C, humidity "
            f"{w['week_mean_humidity_pct']}%, {w['week_rain_days']} rain day(s); now {w['current_temp_c']} °C.")


def build_report(r, plant_text):
    p = r["plant"]
    fields = {}
    fields["Plant Name"] = p["common_name"] if p else f"{plant_text} (not in the botanical database)"
    sg = r.get("crop_suggestion")
    if sg:
        fields["Plant Name"] += (f" (as entered; the photo looks more like {sg['suggested']} at {_pct(sg['mass'])} — "
                                 f"if it is {sg['suggested']}, diagnose again with that name)")
    fields["Biological Name of the plant"] = f"{p['scientific_name']} (family {p['family']})" if p else "Unknown"
    fields["Symptoms found"] = (", ".join(f"{s['name']} ({_pct(s['probability'])})" for s in r["symptoms_detected"])
                                or "No clear disease symptoms detected")
    c = r["candidates"]
    if r["verdict"] == "predicted":
        d = c[0]
        fields["Predicted disease"] = (f"{d['name']} — {_pct(d['confidence'])} confidence. Cause: {d['pathogen']}. "
                                       + (f"Weather risk: {d['weather_risk']['level']}. " if d["weather_risk"] else "")
                                       + _weather_line(r))
    elif r["verdict"] == "healthy":
        fields["Predicted disease"] = f"None — the leaf looks healthy ({_pct(c[0]['confidence'])}). " + _weather_line(r)
    else:
        why = {"below_threshold": f"no disease reached the {_pct(r['threshold'])} score threshold "
                                  "(a heuristic score, not a probability of being correct)",
               "low_photo_fit": "the photo does not closely match the trained photos of this crop",
               "crop_conflict": "the photo looks more like another crop than the one named",
               "model_coverage_limited": "the image model covers only part of this crop's conditions",
               "model_coverage_related": "the image model was trained on a related crop, not this one",
               "model_coverage_none": "the image model was not trained on this crop",
               "plant_not_in_database": "the plant is not in the botanical database"}
        reasons = "; ".join(why[k] for k in r.get("unknown_reasons", []) if k in why) or "the evidence is insufficient"
        lines = ([f"Possibly {c[0]['name']} — {_pct(c[0]['confidence'])} (not confirmed: {reasons}). "
                  "Closest matches (leads to confirm, not a diagnosis):"] if c else
                 [f"Unknown disease — {reasons}. No matching disease in the database (not a diagnosis)."])
        for i, d in enumerate(c, 1):
            lines.append(f"  {i}. {d['name']} — {_pct(d['confidence'])} ({d['basis']}; cause: {d['pathogen']}"
                         + (f"; weather risk {d['weather_risk']['level']}" if d["weather_risk"] else "") + ")")
        lines.append("  " + _weather_line(r))
        fields["Predicted disease"] = "\n".join(lines)

    tests, solutions = [], []
    for i, d in enumerate(c if r["verdict"] == "unknown" else c[:1], 1):
        if d["healthy"]:
            continue
        head = f"{d['name']}: " if r["verdict"] == "unknown" else ""
        tests.append(f"{head}" + "; ".join(f"{t['name']} ({t['level'].replace('_', ' ')}, {t['time']})"
                                           for t in d["tests"]))
        s = d["solution"]
        parts = []
        if s["cultural"]:
            parts.append("Cultural: " + "; ".join(s["cultural"]))
        if s["biological"]:
            parts.append("Biological: " + ", ".join(s["biological"]))
        if s["chemical"]:
            parts.append("Chemical (active ingredients): " + ", ".join(s["chemical"]))
        if s["state_schedule"]:
            parts.append("Follow: " + s["state_schedule"])
        solutions.append(head + " | ".join(parts))
    where = next((d["where_to_test"] for d in c if d["where_to_test"]), [])
    if where:
        tests.append("Where to test: " + "; ".join(where))
    unknown = r["verdict"] == "unknown"
    fields["Test to confirm"] = "\n".join(tests) if tests else (
        "Show the leaf to your KVK or State Agricultural University plant clinic." if unknown
        else "No test needed — keep monitoring weekly.")
    fields["Solution for the Predicted disease"] = ("\n".join(solutions) + "\n" + DISCLAIMER) if solutions else (
        "No treatment until the cause is confirmed." if unknown else "No treatment needed. Keep monitoring weekly.")
    text = "\n".join(f"{k} : {v}" for k, v in fields.items())
    return {"fields": fields, "text": text}

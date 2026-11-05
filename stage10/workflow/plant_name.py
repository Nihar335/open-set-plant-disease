"""User-provided plant name: resolution against the verified knowledge base, and a correctable selection session.

Product-scope decision SC-M4-03 (2026-10-15): the plant name is SUPPLIED BY THE USER. Automatic plant identification
(BioCLIP 2, stage10/plant_id) is NOT used: this module never imports or calls it.

Rules:
  * names resolve only through the knowledge base's own records (common names, synonyms, scientific name, plant id);
    no alias is invented here, and nothing is fuzzily auto-mapped;
  * only harmless differences are normalised: Unicode NFC, letter case, and runs of whitespace;
  * an exact alias that belongs to more than one plant is AMBIGUOUS, and the user must choose;
  * a non-exact entry is never mapped: close or partial matches are offered as suggestions the user must select;
  * empty, invalid, unknown and ambiguous inputs return a clarification response, never a plant identity;
  * the result says whether the plant is in the frozen disease model's scope (six crops) and in the knowledge base.
"""
from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

# Same alphabet and length as report_engine.inputs._clean_name (Latin + Devanagari letters, space . , ' ( ) -; max 60).
_NAME_RE = re.compile(r"^[A-Za-zऀ-ॿ][A-Za-zऀ-ॿ .,'()\-]{0,59}$")
ENTRY_CHANNELS = ("typed", "selected_from_list")
VOICE_CHANNEL = "voice_transcript_confirmed"
STATUSES = ("resolved", "ambiguous", "needs_clarification", "unknown", "empty", "invalid")
SUGGESTION_CUTOFF = 0.75
MAX_SUGGESTIONS = 5


def normalise(text: str) -> str:
    """NFC, trimmed, single spaces, case-folded. Punctuation and spelling are NOT changed."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip().casefold()


def alias_index(kb) -> dict[str, dict[str, str]]:
    """normalised alias -> {plant_id: alias kind}, built only from verified knowledge-base plant records."""
    idx: dict[str, dict[str, str]] = {}

    def add(name, pid, kind):
        if name:
            idx.setdefault(normalise(name), {}).setdefault(pid, kind)

    for pid, p in kb.plants.items():
        for n in p["common_names"]:
            add(n["name"], pid, "common_name")
        for s in p.get("synonyms", []):
            add(s, pid, "synonym")
        add(p["scientific_name"], pid, "scientific_name")
        add(p["taxonomy"]["species"], pid, "scientific_name")
        add(pid, pid, "plant_id")
    for a in reviewed_local_aliases(kb):
        add(a["alias"], a["plant_id"], "local_name")
    return idx


LOCAL_ALIASES = Path(__file__).resolve().parent / "local_aliases.yaml"


def local_alias_errors(entries: list, kb) -> list[str]:
    err = []
    for i, a in enumerate(entries):
        if not isinstance(a, dict) or not isinstance(a.get("alias"), str) or not a["alias"].strip():
            err.append(f"entry {i}: alias missing")
            continue
        if a.get("plant_id") not in kb.plants:
            err.append(f"{a['alias']}: unknown plant_id {a.get('plant_id')!r}")
        if a.get("review_status") not in ("draft", "reviewed"):
            err.append(f"{a['alias']}: review_status must be draft or reviewed")
        ref = a.get("reference")
        if ref is not None and not (isinstance(ref, dict) and ref.get("url", "").startswith("https://") and ref.get("retrieved")):
            err.append(f"{a['alias']}: reference needs an https url and a retrieved date")
        if not (a.get("source_ids") or a.get("editorial") is True or ref):
            err.append(f"{a['alias']}: needs source_ids, an external reference, or editorial: true")
        for s in a.get("source_ids") or []:
            if s not in kb.sources:
                err.append(f"{a['alias']}: unknown source {s}")
        if a.get("review_status") == "reviewed" and not (a.get("reviewer") and a.get("reviewed_on")):
            err.append(f"{a['alias']}: reviewed entries need reviewer and reviewed_on")
        if not isinstance(a.get("lang"), str) or not re.fullmatch(r"[a-z]{2,3}", a["lang"]):
            err.append(f"{a['alias']}: lang must be an ISO 639 code")
    return err


def reviewed_local_aliases(kb, path: Path | None = None) -> list[dict]:
    """Only reviewed, sourced-or-editorial entries; a malformed file raises rather than being silently ignored."""
    import yaml
    data = yaml.safe_load((path or LOCAL_ALIASES).read_text(encoding="utf-8")) or {}
    entries = data.get("aliases") or []
    err = local_alias_errors(entries, kb)
    if err:
        raise ValueError("local_aliases.yaml: " + "; ".join(err))
    return [a for a in entries if a["review_status"] == "reviewed"]


def plant_option(kb, pid: str) -> dict:
    p = kb.plants[pid]
    display = next((n["name"] for n in p["common_names"] if n.get("lang") == "en"), p["common_names"][0]["name"])
    return {"plant_id": pid, "display_name": display, "scientific_name": p["scientific_name"],
            "disease_model_supported": bool(p["frozen_model_scope"]),
            "knowledge_base_conditions_sourced": sum(1 for c in kb.conditions.values() if c["curation_status"] == "sourced"
                                                     and any(a["plant_id"] == pid for a in c["affected_plants"])),
            "accepted_names": _dedupe([n["name"] for n in p["common_names"]] + list(p.get("synonyms", [])) + [p["taxonomy"]["species"]])}


def _dedupe(names: list[str]) -> list[str]:
    """Distinct names ignoring case and whitespace (first spelling kept), sorted case-insensitively."""
    seen: dict[str, str] = {}
    for n in names:
        seen.setdefault(normalise(n), n)
    return sorted(seen.values(), key=str.casefold)


def supported_plant_options(kb) -> list[dict]:
    """Every plant the user can select from a list (the knowledge base's plants), with its disease-model support."""
    return [plant_option(kb, pid) for pid in sorted(kb.plants, key=lambda x: plant_option(kb, x)["display_name"])]


def _clarify(status: str, entered, message: str, suggestions=()) -> dict:
    return {"status": status, "entered_text": entered, "plant_id": None, "plant": None, "matched_alias": None, "alias_kind": None,
            "suggestions": list(suggestions), "message": message, "requires_user_choice": status != "resolved"}


def resolve_plant_name(text, kb) -> dict:
    """Resolve typed text to a verified plant, or explain what the user must clarify. Never guesses."""
    if text is None or (isinstance(text, str) and not text.strip()):
        return _clarify("empty", text, "Please type the plant name, choose it from the list, or say that you do not know it.")
    if not isinstance(text, str):
        return _clarify("invalid", None, "The plant name must be text.")
    clean = re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()
    if not _NAME_RE.match(clean):
        return _clarify("invalid", None, "The plant name contains characters that cannot be used, or is longer than 60 characters. "
                                          "Please use letters only.")
    idx = alias_index(kb)
    key = normalise(clean)
    hits = idx.get(key, {})
    if len(hits) == 1:
        pid, kind = next(iter(hits.items()))
        return {"status": "resolved", "entered_text": clean, "plant_id": pid, "plant": plant_option(kb, pid),
                "matched_alias": key, "alias_kind": kind, "suggestions": [], "requires_user_choice": False,
                "message": f"Recognised as {plant_option(kb, pid)['display_name']}. Please confirm before continuing."}
    if len(hits) > 1:
        return _clarify("ambiguous", clean, f"'{clean}' can mean more than one plant. Please choose the correct one.",
                        [plant_option(kb, p) for p in sorted(hits)])
    # not an exact verified name: offer (never apply) partial-word and close-spelling matches
    cand = set()
    words = set(key.split())
    for alias, pids in idx.items():
        if words and words <= set(alias.split()):
            cand |= set(pids)
    for alias in difflib.get_close_matches(key, list(idx), n=MAX_SUGGESTIONS, cutoff=SUGGESTION_CUTOFF):
        cand |= set(idx[alias])
    if cand:
        return _clarify("needs_clarification", clean,
                        f"'{clean}' is not a recognised plant name. Did you mean one of these? Choose one, type the name again, "
                        "or continue with your own name (it will not be matched to a plant in the knowledge base).",
                        [plant_option(kb, p) for p in sorted(cand)][:MAX_SUGGESTIONS])
    return _clarify("unknown", clean,
                    f"'{clean}' is not in the AgroVaidya knowledge base. You can type it again, choose from the list, say you do "
                    "not know the name, or continue with this name: you will get general guidance and advice to consult an expert, "
                    "but no plant-specific disease assessment.")


class PlantNameError(ValueError):
    pass


@dataclass
class PlantNameSession:
    """The user's plant-name choice. Correctable until confirmed; every step is kept for provenance.

    Outcomes on confirm(): a verified plant ('known_plant'), a name the user supplied that is not in the knowledge base
    ('unlisted_name', only after the user explicitly chose to keep it), or 'not_known' (the user does not know the name).
    """
    kb: object
    history: list = field(default_factory=list)
    pending: dict | None = None
    confirmed_choice: dict | None = None

    def _log(self, action: str, **kw):
        self.history.append({"action": action, **kw})

    def _open(self):
        if self.confirmed_choice is not None:
            raise PlantNameError("the plant name is already confirmed; call correct() to change it")

    def enter(self, text, channel: str = "typed") -> dict:
        self._open()
        if channel == VOICE_CHANNEL:
            from stage10.workflow.voice import voice_input_available
            if not voice_input_available():
                raise PlantNameError("voice input is not available (speech-to-text engine not selected: M5-B2); type or select the name")
        elif channel not in ENTRY_CHANNELS:
            raise PlantNameError(f"unknown entry channel {channel!r}")
        res = resolve_plant_name(text, self.kb)
        self.pending = {"kind": "candidate", "channel": channel, "resolution": res} if res["status"] == "resolved" else \
            {"kind": "clarification", "channel": channel, "resolution": res}
        self._log("enter", channel=channel, entered_text=res["entered_text"], status=res["status"], plant_id=res["plant_id"])
        return res

    def select(self, plant_id: str) -> dict:
        """Choose a plant from the list or from suggestions (always allowed, also after an ambiguous entry)."""
        self._open()
        if plant_id not in self.kb.plants:
            raise PlantNameError(f"{plant_id!r} is not a plant in the knowledge base")
        opt = plant_option(self.kb, plant_id)
        res = {"status": "resolved", "entered_text": opt["display_name"], "plant_id": plant_id, "plant": opt,
               "matched_alias": normalise(opt["display_name"]), "alias_kind": "selected_from_list", "suggestions": [],
               "requires_user_choice": False, "message": f"Selected {opt['display_name']}. Please confirm before continuing."}
        self.pending = {"kind": "candidate", "channel": "selected_from_list", "resolution": res}
        self._log("select", plant_id=plant_id)
        return res

    def keep_unlisted_name(self) -> dict:
        """The user insists on a name the knowledge base does not contain (status unknown / needs_clarification only)."""
        self._open()
        if not self.pending or self.pending["resolution"]["status"] not in ("unknown", "needs_clarification"):
            raise PlantNameError("only an unrecognised, valid name can be kept as entered")
        self.pending = {**self.pending, "kind": "unlisted"}
        self._log("keep_unlisted_name", entered_text=self.pending["resolution"]["entered_text"])
        return self.pending["resolution"]

    def dont_know(self):
        self._open()
        self.pending = {"kind": "not_known", "channel": "selected_from_list", "resolution": None}
        self._log("dont_know")

    def confirm(self) -> dict:
        self._open()
        p = self.pending
        if p is None or p["kind"] == "clarification":
            raise PlantNameError("nothing to confirm: enter, select or keep a plant name, or say you do not know it")
        r = p["resolution"]
        if p["kind"] == "candidate":
            c = {"outcome": "known_plant", "plant_id": r["plant_id"], "plant_name": r["plant"]["display_name"],
                 "entered_text": r["entered_text"], "matched_alias": r["matched_alias"], "alias_kind": r["alias_kind"],
                 "disease_model_supported": r["plant"]["disease_model_supported"]}
        elif p["kind"] == "unlisted":
            c = {"outcome": "unlisted_name", "plant_id": None, "plant_name": r["entered_text"], "entered_text": r["entered_text"],
                 "matched_alias": None, "alias_kind": None, "disease_model_supported": False}
        else:
            c = {"outcome": "not_known", "plant_id": None, "plant_name": None, "entered_text": None, "matched_alias": None,
                 "alias_kind": None, "disease_model_supported": False}
        self.confirmed_choice = {**c, "entry_channel": p["channel"], "source": "user_supplied",
                                 "independently_verified": False, "automatic_identification_used": False,
                                 "corrections": sum(1 for h in self.history if h["action"] == "correct")}
        self._log("confirm", outcome=c["outcome"], plant_id=c["plant_id"])
        return dict(self.confirmed_choice)

    def correct(self):
        """Undo the confirmation (or pending entry) so the user can change the plant before the assessment runs."""
        self.confirmed_choice = None
        self.pending = None
        self._log("correct")

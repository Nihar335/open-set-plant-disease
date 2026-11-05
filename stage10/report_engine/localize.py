"""Re-render a finished report into another language from its language-independent content keys (schema finding D-6).

localize(report, lang) changes only localized texts, `language` and `review_status.translation_review`, and adds the
translation notice. No diagnostic field is touched. Knowledge-base derived statements have no key and stay in English.
"""
from __future__ import annotations

import copy

from stage10.schemas.validate import stage10_config


def _resolve(report: dict, path: str):
    o = report
    for part in path.strip("/").split("/"):
        o = o[int(part)] if isinstance(o, list) else o[part]
    return o


def localize(report: dict, lang: str) -> dict:
    from stage10.report_engine.engine import messages

    cfg = stage10_config()["language"]
    if lang not in cfg["supported"]:
        raise ValueError(f"language {lang!r} not enabled")
    cat = messages()["messages"]
    r = copy.deepcopy(report)
    for ck in r["content_keys"]:
        obj = _resolve(r, ck["path"])
        for k in [k for k in obj if k != "en"]:
            del obj[k]
        if lang != "en":
            obj[lang] = cat[ck["key"]][lang].format(**ck["params"])
    r["language"] = lang
    r["review_status"]["translation_review"] = {"en": "not_applicable",
                                                **({lang: "pending_fluent_speaker_review"} if lang in cfg["requires_fluent_review"] else {})}
    note_keys = [i for i, ck in enumerate(r["content_keys"]) if ck["key"] == "translation_note"]
    if lang != "en" and not note_keys:
        i = len(r["safety_warnings"])
        r["safety_warnings"].append({"en": cat["translation_note"]["en"], lang: cat["translation_note"][lang]})
        last = max(j for j, ck in enumerate(r["content_keys"]) if ck["path"].startswith("/safety_warnings/"))
        r["content_keys"].insert(last + 1, {"path": f"/safety_warnings/{i}", "key": "translation_note", "params": {}})
    if lang == "en" and note_keys:
        idx = int(r["content_keys"][note_keys[0]]["path"].rsplit("/", 1)[1])
        del r["safety_warnings"][idx]
        del r["content_keys"][note_keys[0]]
    return r

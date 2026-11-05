"""Compare candidate Vosk models on the FLEURS sample (same data and metric as benchmark.py).

  python -m stage10.app.speech.compare_models       # writes stage10/app/speech/model_comparison.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from stage10.app.speech import benchmark as B
from stage10.app.speech import engines as E

OUT = Path(__file__).resolve().parent / "model_comparison.json"
CANDIDATES = {"en": ["vosk-model-small-en-in-0.4", "vosk-model-en-in-0.5", "vosk-model-small-en-us-0.15"],
              "hi": ["vosk-model-small-hi-0.22"] + (["vosk-model-hi-0.22"] if (E.ROOT / "stage10_local/models/speech/vosk-model-hi-0.22").exists() else [])}


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    res = {"note": "FLEURS test sample (public read speech, CC BY 4.0), NOT farmer speech; en_us has US accents, so it cannot "
                   "show how Indian-English speakers are recognised.", "results": {}}
    for lang, models in CANDIDATES.items():
        for m in models:
            os.environ["AGROVAIDYA_VOSK_EN" if lang == "en" else "AGROVAIDYA_VOSK_HI"] = m
            stt, _, why = E.local_engines()
            r = B.bench_stt(stt, "en_us" if lang == "en" else "hi_in", lang)
            r["worker_rss_mb"] = round(B.rss_mb(stt.worker.proc.pid) or 0, 1)
            r["model_zip_mb"] = round((E.ROOT / f"stage10_local/models/speech/{m}.zip").stat().st_size / 2 ** 20, 1)
            stt.worker.stop()
            res["results"][f"{lang}:{m}"] = r
            print(lang, m, r["wer"], r["latency_s"], r["worker_rss_mb"], "MB")
    OUT.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

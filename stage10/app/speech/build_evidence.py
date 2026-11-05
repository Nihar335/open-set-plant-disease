"""Evidence register for the M5 speech decision (SPEECH_DECISION.md).

  python -m stage10.app.speech.build_evidence          # writes speech_evidence.json from the local cache

Raw copies are cached (not redistributed) in stage10_local/speech_evidence/ and fetched with curl on 2026-10-15.
Each entry records the URL, the cached file's SHA-256 and the facts read from it. "check" patterns must still occur in
the cached text, so a stale or altered copy is detected (--check).
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "stage10_local" / "speech_evidence"
OUT = Path(__file__).resolve().parent / "speech_evidence.json"
ACCESSED = "2026-10-15"

SOURCES = [
    ("E1", "hf_whisper_large_v3.json", "https://huggingface.co/api/models/openai/whisper-large-v3",
     "Whisper large-v3: licence apache-2.0; 1,543,490,560 parameters; language tags include hi, bn, gu, kn, ml, mr, pa, ta, te, ur, en",
     [r'"license:apache-2.0"', r'"total":1543490560']),
    ("E2", "hf_whisper_small.json", "https://huggingface.co/api/models/openai/whisper-small",
     "Whisper small: licence apache-2.0; 241,734,912 parameters", [r'"license:apache-2.0"', r'"total":241734912']),
    ("E3", "whisper_tokenizer.py", "https://raw.githubusercontent.com/openai/whisper/main/whisper/tokenizer.py",
     "Whisper language list has hi, bn, gu, kn, ml, mr, pa, ta, te, ur but NO Odia ('or')", [r'"hi": "hindi"', r'"ta": "tamil"']),
    ("E4", "vosk_models.html", "https://alphacephei.com/vosk/models",
     "Vosk: vosk-model-small-hi-0.22 42M, WER 20.89 (IITM) / 24.72 (MUCS); vosk-model-hi-0.22 1.5Gb, WER 14.85 (CV) / 14.83 (IITM) / "
     "13.11 (MUCS); small Indian English 36M WER 49.05 (NPTEL); en-in 1G WER 36.12; Gujarati 700M/100M; small Telugu 58M WER 87.9 "
     "(Fleurs); all Apache 2.0", [r"vosk-model-small-hi-0.22", r"vosk-model-hi-0.22", r"vosk-model-small-te-0.42"]),
    ("E5", "vosk_license.txt", "https://raw.githubusercontent.com/alphacep/vosk-api/master/COPYING", "Vosk API licence: Apache License 2.0",
     [r"Apache License"]),
    ("E6", "hf_indicconformer.json", "https://huggingface.co/api/models/ai4bharat/indic-conformer-600m-multilingual",
     "AI4Bharat IndicConformer 600M multilingual: licence mit; gated (terms acceptance on Hugging Face); per-language ONNX heads incl. hi",
     [r'"license":"mit"', r'"gated":"auto"', r"joint_post_net_hi\.onnx"]),
    ("E7", "hf_indic_parler.json", "https://huggingface.co/api/models/ai4bharat/indic-parler-tts",
     "AI4Bharat Indic Parler-TTS: licence apache-2.0; ~938M parameters; languages en, as, bn, gu, hi, kn, ks, or, ml, mr, ne, pa, sa, sd, ta, "
     "te, ur; gated", [r'"license":"apache-2.0"', r'"total":937803241', r'"or"']),
    ("E8", "piper_voices.json", "https://huggingface.co/rhasspy/piper-voices/resolve/main/voices.json",
     "Piper voices for target languages: hi_IN (pratham, priyamvada, rohan), te_IN (3), ml_IN (2), mr_IN (1), bn_BD (1), ur_PK (2), each ~63-77 MB; "
     "NO voices for ta, gu, kn, pa, or, en_IN", [r'"hi_IN-pratham-medium"', r'"te_IN-maya-medium"']),
    ("E9", "piper_hi_pratham_card.md", "https://huggingface.co/rhasspy/piper-voices/resolve/main/hi/hi_IN/pratham/medium/MODEL_CARD",
     "Piper hi_IN pratham: training dataset licence CC BY-NC-SA 4.0 (non-commercial)", [r"by-nc-sa/4\.0"]),
    ("E10", "piper_hi_priyamvada_card.md", "https://huggingface.co/rhasspy/piper-voices/resolve/main/hi/hi_IN/priyamvada/medium/MODEL_CARD",
     "Piper hi_IN priyamvada: training dataset licence CC BY-NC-SA 4.0 (non-commercial)", [r"by-nc-sa/4\.0"]),
    ("E11", "piper_hi_rohan_card.md", "https://huggingface.co/rhasspy/piper-voices/resolve/main/hi/hi_IN/rohan/medium/MODEL_CARD",
     "Piper hi_IN rohan: dataset licence = IIT Madras Indic TTS licence (PDF not retrievable on 2026-10-15; terms UNVERIFIED)",
     [r"iitm\.ac\.in/donlab/indictts/downloads/license\.pdf"]),
    ("E12", "piper1_license.txt", "https://raw.githubusercontent.com/OHF-Voice/piper1-gpl/main/COPYING",
     "Piper engine (maintained fork piper1-gpl): GNU GPL v3", [r"GNU GENERAL PUBLIC LICENSE", r"Version 3"]),
    ("E13", "hf_mms_tts_hin.json", "https://huggingface.co/api/models/facebook/mms-tts-hin",
     "Meta MMS-TTS Hindi: licence cc-by-nc-4.0 (non-commercial)", [r'"license:cc-by-nc-4.0"']),
    ("E14", "espeak_readme.md", "https://raw.githubusercontent.com/espeak-ng/espeak-ng/master/README.md",
     "eSpeak NG: GPL v3 or later; formant synthesis (robotic), 100+ languages", [r"GPL version 3", r"formant synthesis"]),
    ("E15", "mdn_webspeech.html", "https://developer.mozilla.org/en-US/docs/Web/API/Web_Speech_API/Using_the_Web_Speech_API",
     "Browser Web Speech API: by default recognition is server-based (audio is sent to a server); on-device recognition only with "
     "processLocally = true and an installed language pack", [r"server-based recognition engine", r"processLocally"]),
    ("E16", "gcp_stt_langs.html", "https://cloud.google.com/speech-to-text/docs/speech-to-text-supported-languages",
     "Google Cloud STT: hi-IN, te-IN, pa-Guru-IN etc. with chirp / chirp_2 models (region asia-southeast1 listed)", [r"hi-IN", r"te-IN"]),
    ("E17", "gcp_stt_datalogging.html", "https://cloud.google.com/speech-to-text/docs/data-logging",
     "Google Cloud STT: does not log customer audio or transcripts unless the customer opts into data logging",
     [r"log customer audio data or transcripts"]),
    ("E18", "gcp_stt_pricing.html", "https://cloud.google.com/speech-to-text/pricing",
     "Google Cloud STT standard recognition: USD 0.016 per minute for the first 500,000 minutes per month (without data logging)",
     [r"\$0\.016"]),
    ("E19", "azure_lang.html", "https://learn.microsoft.com/en-us/azure/ai-services/speech-service/language-support",
     "Azure AI Speech lists hi-IN, ta-IN, te-IN, mr-IN, gu-IN, ml-IN, pa-IN, or-IN, ur-IN", [r"hi-IN", r"or-IN"]),
    ("E20", "azure_privacy.html",
     "https://learn.microsoft.com/en-us/azure/ai-foundry/responsible-ai/speech-service/speech-to-text/data-privacy-security",
     "Azure real-time speech to text: audio processed only in server memory, no data stored at rest; encrypted in transit",
     [r"processed only on the Azure's server memory, and no data is stored at rest"]),
    ("E21", "sarvam_stt.html", "https://docs.sarvam.ai/api-reference-docs/speech-to-text/transcribe",
     "Sarvam AI STT (saaras): hi-IN, bn-IN, kn-IN, ml-IN, mr-IN, od-IN, pa-IN, ta-IN, te-IN, en-IN, gu-IN, ur-IN and more (22 Indic)",
     [r"od-IN", r"saaras"]),
    ("E22", "sarvam_pricing.html", "https://www.sarvam.ai/api-pricing",
     "Sarvam AI: STT real-time INR 0.49 per minute (pay as you go); TTS Bulbul V4 Flash INR 2.00 per 1,000 characters",
     [r"0\.49", r"Bulbul V4 Flash"]),
    ("E23", "hf_kokoro.json", "https://huggingface.co/api/models/hexgrad/Kokoro-82M",
     "Kokoro-82M: licence apache-2.0; NOT gated; Hindi voices hf_alpha, hf_beta, hm_omega, hm_psi", [r'"license:apache-2.0"', r"hf_alpha"]),
    ("E24", "kokoro_voices.md", "https://huggingface.co/hexgrad/Kokoro-82M/raw/main/VOICES.md",
     "Kokoro Hindi voices self-graded 'C' overall; author notes weak non-English G2P and weakness on short utterances",
     [r"### Hindi", r"short utterances"]),
    ("E25", "kokoro_onnx_license.txt", "https://raw.githubusercontent.com/thewh1teagle/kokoro-onnx/main/LICENSE",
     "kokoro-onnx runtime: MIT", [r"MIT License"]),
    ("E26", "kokoro_onnx_release.json", "https://api.github.com/repos/thewh1teagle/kokoro-onnx/releases/tags/model-files-v1.0",
     "kokoro-onnx model files v1.0: kokoro-v1.0.onnx 325,532,387 B; int8 92,361,271 B; voices-v1.0.bin 28,214,398 B",
     [r'"kokoro-v1.0.int8.onnx"', r"92361271"]),
    ("E27", "pypi_vosk.json", "https://pypi.org/pypi/vosk/json",
     "vosk 0.3.45 Python package: Apache Software License; win_amd64 wheel", [r"Apache Software License", r"win_amd64"]),
    ("E28", "pypi_phonemizer.json", "https://pypi.org/pypi/phonemizer/json",
     "phonemizer 3.4.0 (kokoro-onnx dependency): GNU GPL v3 or later", [r"GPLv3"]),
    ("E29", "espeakng_loader_license.txt", "https://raw.githubusercontent.com/thewh1teagle/espeakng-loader/main/LICENSE",
     "espeakng-loader wrapper: MIT (it bundles the espeak-ng library, which is GPL v3, see E14)", [r"MIT License"]),
    ("E30", "pypi_onnxruntime.json", "https://pypi.org/pypi/onnxruntime/json",
     "onnxruntime 1.31.0: MIT; cp312 win_amd64 wheel", [r"MIT License", r"cp312-win_amd64"]),
    ("E31", "hf_fleurs.json", "https://huggingface.co/api/datasets/google/fleurs",
     "FLEURS dataset: licence cc-by-4.0; not gated (benchmark data)", [r'"license:cc-by-4.0"']),
]


def text_of(p: Path) -> str:
    t = p.read_text(encoding="utf-8", errors="ignore")
    t = re.sub(r"<script.*?</script>|<style.*?</style>", " ", t, flags=re.S)
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", t)))


def build() -> tuple[dict, list[str]]:
    err, out = [], []
    for sid, fn, url, fact, checks in SOURCES:
        p = CACHE / fn
        if not p.exists():
            err.append(f"{sid}: cached copy missing ({fn})")
            continue
        raw = p.read_text(encoding="utf-8", errors="ignore")
        t = text_of(p)
        missing = [c for c in checks if not (re.search(c, raw) or re.search(c, t))]
        if missing:
            err.append(f"{sid}: expected text not found in {fn}: {missing}")
        out.append({"id": sid, "url": url, "accessed": ACCESSED, "cached_file": fn,
                    "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "fact": fact})
    return {"note": "Facts read from official pages / model-hub API records cached on the access date. Re-verify before any contract.",
            "sources": out}, err


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    doc, err = build()
    txt = json.dumps(doc, indent=1, ensure_ascii=False) + "\n"
    if a.check:
        if OUT.read_text(encoding="utf-8") != txt:
            err.append("speech_evidence.json is stale")
    else:
        OUT.write_text(txt, encoding="utf-8")
    for e in err:
        print("ERROR", e)
    print(f"SPEECH EVIDENCE: {'PASS' if not err else 'FAIL'} ({len(doc['sources'])} sources)")
    sys.exit(1 if err else 0)


if __name__ == "__main__":
    main()

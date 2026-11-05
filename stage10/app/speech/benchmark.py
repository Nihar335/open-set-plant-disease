"""Speech benchmark for the local engines (M5-B2 / M5-E1). Run from the project root with the project Python:

  python -m stage10.app.speech.benchmark            # writes stage10/app/speech/benchmark_results.json

STT data: FLEURS test split (google/fleurs, CC BY 4.0, public read speech; downloaded 2026-10-15 to
stage10_local/speech_benchmark/fleurs; hashes below). It is NOT farmer speech: no field noise, no farm vocabulary,
read rather than spontaneous. A deterministic sample of N utterances per language is used; nothing is tuned on it.
TTS: latency on fixed app texts (questions, warnings). Plant names: SYNTHETIC speech (Kokoro -> Vosk with the
plant-name grammar), a pipeline check only, never a human-speech result.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import statistics
import sys
import tarfile
import time
import unicodedata
import wave
from pathlib import Path

import numpy as np

from stage10.app.speech import engines as E

ROOT = Path(__file__).resolve().parents[3]
FLEURS = ROOT / "stage10_local" / "speech_benchmark" / "fleurs"
OUT = Path(__file__).resolve().parent / "benchmark_results.json"
N = 150
SALT = "agrovaidya-m5-speech-bench-v1"
HASHES = {"hi_in/test.tar.gz": "47c238b9cf7016ba596639bb3038b0c89dc6e85759f16d09c4f58c6dfa25c65f",
          "en_us/test.tar.gz": "d9c2e37b41aacd41bc283554a0a82b5476b36887049774ecb2819dcaaa55a356",
          "hi_in/test.tsv": "889e82e2875490f9533ff59ab67326fcfa01cf828fc4412758aaaa1f442a729f",
          "en_us/test.tsv": "74c046239374deeb60fa63f258f907388093a32bcaa3140965f70ef05c79f7ca"}
TTS_TEXTS = {
    "en": ["Do the spots show concentric rings, like a target?", "Is there a yellow halo around the spots?",
           "Nothing in this report is a confirmed diagnosis.",
           "Show the plant or the photos to the local Krishi Vigyan Kendra before spending money on any treatment.",
           "Do not spray pesticides or apply fertiliser on the basis of this report alone."],
    "hi": ["क्या धब्बों में गोल छल्ले दिखते हैं?", "क्या धब्बों के चारों ओर पीला घेरा है?",
           "इस रिपोर्ट में कोई भी बात पक्का निदान नहीं है।"],
}


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def norm(t: str) -> list[str]:
    t = unicodedata.normalize("NFC", t.lower())
    t = re.sub(r"[^\w\sऀ-ॿ]", " ", t)
    return t.split()


def wer_counts(ref: list[str], hyp: list[str]) -> int:
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
    return d[len(hyp)]


def float_wav_to_pcm16(data: bytes) -> tuple[bytes, float]:
    """FLEURS stores IEEE-float WAV (format 3); convert to the 16-bit PCM WAV the app's browser recorder produces."""
    import struct
    pos, fmt, rate, ch, bits, payload = 12, None, None, None, None, None
    while pos + 8 <= len(data):
        cid, size = data[pos:pos + 4], struct.unpack("<I", data[pos + 4:pos + 8])[0]
        body = data[pos + 8:pos + 8 + size]
        if cid == b"fmt ":
            fmt, ch, rate = struct.unpack("<HHI", body[:8])
            bits = struct.unpack("<H", body[14:16])[0]
        elif cid == b"data":
            payload = body
        pos += 8 + size + (size & 1)
    if fmt == 3 and bits == 32:
        x = np.frombuffer(payload, "<f4")
    elif fmt == 1 and bits == 16:
        x = np.frombuffer(payload, "<i2").astype(np.float32) / 32768
    else:
        raise ValueError(f"unsupported WAV format {fmt}/{bits}")
    if ch and ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()
    return E.pcm16_to_wav(pcm, rate), len(x) / rate


def rss_mb(pid: int) -> float | None:
    """Resident memory of the worker, including the real interpreter that a Windows venv launcher starts as a child."""
    try:
        import psutil
        p = psutil.Process(pid)
        return sum(q.memory_info().rss for q in [p, *p.children(recursive=True)]) / 2 ** 20
    except Exception:                                            # noqa: BLE001
        return None


def bench_stt(stt, lang_dir: str, lang: str) -> dict:
    with open(FLEURS / lang_dir / "test.tsv", encoding="utf-8") as f:
        rows = list(csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
    rows.sort(key=lambda r: hashlib.sha256(f"{SALT}|{r[1]}".encode()).hexdigest())
    pick = {r[1]: r[3] for r in rows[:N]}                        # filename -> normalised transcription
    errs = words = 0
    lat, rtf, per = [], [], []
    with tarfile.open(FLEURS / lang_dir / "test.tar.gz") as tf:
        for m in tf:
            name = m.name.split("/")[-1]
            if name not in pick:
                continue
            data, dur = float_wav_to_pcm16(tf.extractfile(m).read())   # in memory only
            t0 = time.perf_counter()
            r = stt.transcribe(data, "audio/wav", lang)
            dt = time.perf_counter() - t0
            ref, hyp = norm(pick[name]), norm(r.text)
            e = wer_counts(ref, hyp)
            errs, words = errs + e, words + len(ref)
            lat.append(dt)
            rtf.append(dt / dur)
            per.append(e / max(1, len(ref)))
    lat.sort()
    return {"dataset": f"FLEURS {lang_dir} test (CC BY 4.0), deterministic sample", "n_utterances": len(lat),
            "wer": round(errs / words, 4), "reference_words": words,
            "median_utterance_wer": round(statistics.median(per), 4),
            "latency_s": {"p50": round(statistics.median(lat), 2), "p95": round(lat[int(0.95 * (len(lat) - 1))], 2)},
            "real_time_factor_median": round(statistics.median(rtf), 3)}


def bench_tts(tts, worker) -> dict:
    out = {}
    for lang, texts in TTS_TEXTS.items():
        lat, rtf = [], []
        for t in texts:
            t0 = time.perf_counter()
            wav = tts.synthesize(t, lang)
            dt = time.perf_counter() - t0
            with wave.open(io.BytesIO(wav)) as w:
                dur = w.getnframes() / w.getframerate()
            lat.append(dt)
            rtf.append(dt / dur)
        out[lang] = {"n_texts": len(texts), "latency_s_median": round(statistics.median(lat), 2), "latency_s_max": round(max(lat), 2),
                     "real_time_factor_median": round(statistics.median(rtf), 3)}
    return out


def bench_plant_names(stt, tts) -> dict:
    from stage10.knowledge_base.kb import load_kb
    from stage10.workflow.plant_name import resolve_plant_name
    kb = load_kb()
    names = sorted({n["name"] for p in kb.plants.values() for n in p["common_names"]})
    from stage10.app.speech.contract import plant_name_grammar
    grammar = plant_name_grammar(kb)
    ok = 0
    rows = []
    for n in names:
        wav = tts.synthesize(n, "en")
        with wave.open(io.BytesIO(wav)) as w:
            x = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32)
            sr = w.getframerate()
        y = np.interp(np.arange(0, len(x), sr / 16000), np.arange(len(x)), x).astype("<i2")
        r = stt.transcribe(E.pcm16_to_wav(y.tobytes(), 16000), "audio/wav", "en", grammar)
        res = resolve_plant_name(r.text, kb) if r.text.strip() else {"status": "empty", "plant_id": None}
        target = resolve_plant_name(n, kb)["plant_id"]
        hit = res.get("plant_id") == target
        ok += hit
        rows.append({"said": n, "heard": r.text, "resolved": res.get("plant_id"), "correct": hit})
    return {"data": "SYNTHETIC (Kokoro English voice -> Vosk en-in with the plant-name grammar); pipeline check, not human speech",
            "n": len(names), "correct": ok, "rows": rows}


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    for rel, h in HASHES.items():
        if sha(FLEURS / rel) != h:
            raise SystemExit(f"benchmark data changed: {rel}")
    stt, tts, why = E.local_engines()
    if stt is None:
        raise SystemExit(f"local speech not available: {why}")
    worker = stt.worker
    res = {"created": time.strftime("%Y-%m-%d"), "hardware": "this development laptop: RTX 4050 6 GB (not used: CPU inference), 16 CPU threads, 15.6 GB RAM",
           "note": "Public read speech and synthetic checks only. NOT farmer speech; no usability result. Thresholds are owner decision D6."}
    t0 = time.perf_counter()
    worker.request({"op": "ping"})
    res["worker_start_s"] = round(time.perf_counter() - t0, 2)
    res["stt"] = {"hi": bench_stt(stt, "hi_in", "hi"), "en (Indian-English model on US read speech)": bench_stt(stt, "en_us", "en")}
    res["stt_worker_rss_mb"] = round(rss_mb(worker.proc.pid) or 0, 1)
    res["tts_int8"] = bench_tts(tts, worker)
    res["worker_rss_mb_after_tts"] = round(rss_mb(worker.proc.pid) or 0, 1)
    res["plant_names_synthetic"] = bench_plant_names(stt, tts)
    worker.stop()
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "plant_names_synthetic"}, indent=1, ensure_ascii=False))
    p = res["plant_names_synthetic"]
    print("plant names (synthetic):", p["correct"], "/", p["n"], [r for r in p["rows"] if not r["correct"]])


if __name__ == "__main__":
    main()

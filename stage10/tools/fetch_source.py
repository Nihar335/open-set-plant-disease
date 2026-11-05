"""Fetch a source page for VERIFICATION, keep a local copy, and extract plain text.

  python -m stage10.tools.fetch_source <SOURCE_ID> <URL>            # download + extract
  python -m stage10.tools.fetch_source --grep <SOURCE_ID> "<regex>"  # show matching passages from the cached text

Local copies live in results/stage10/source_cache/ for verification only. They are NOT redistributed (see the README
written there). The knowledge base stores only the URL, access date, a SHA-256 of the fetched bytes and short verbatim
passages; `stage10.knowledge_base.kb` re-checks that every quoted passage occurs in the cached text.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import ssl
import sys
import urllib.request
from datetime import date
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "results" / "stage10" / "source_cache"
UA = "Mozilla/5.0 (research verification; AgroVaidya Stage 10)"
NOTICE = ("Local verification copies of web sources used by the Stage-10 knowledge base. They are kept only so that quoted\n"
          "passages can be re-checked. Copyright remains with the original publishers. Do NOT redistribute these files.\n")


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


def extract_text(raw: bytes, content_type: str) -> str:
    if "pdf" in content_type or raw[:5] == b"%PDF-":
        import fitz                                    # PyMuPDF: text layer only, no OCR

        with fitz.open(stream=raw, filetype="pdf") as doc:
            return normalise(" ".join(page.get_text() for page in doc))
    if "json" in content_type:
        return normalise(json.dumps(json.loads(raw), ensure_ascii=False, sort_keys=True))
    soup = BeautifulSoup(raw, "html.parser")
    # Server-rendered Next.js pages (e.g. Vikaspedia) carry the article HTML inside the __NEXT_DATA__ JSON, not the DOM.
    nd = soup.find("script", id="__NEXT_DATA__")
    if nd and nd.string:
        try:
            art = json.loads(nd.string)["props"]["pageProps"]["ssrPageContent"]["content"]
            return normalise(BeautifulSoup(art, "html.parser").get_text(" "))
        except (KeyError, TypeError, ValueError):
            pass
    for t in soup(["script", "style", "noscript"]):
        t.decompose()
    return normalise(soup.get_text(" "))


def fetch(source_id: str, url: str) -> dict:
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / "README.txt").write_text(NOTICE, encoding="utf-8")
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    ctx = ssl.create_default_context()
    with urllib.request.urlopen(req, timeout=60, context=ctx) as r:
        raw, ctype, final = r.read(), r.headers.get("Content-Type", ""), r.geturl()
    text = extract_text(raw, ctype)
    meta = {"source_id": source_id, "url": url, "final_url": final, "accessed_on": date.today().isoformat(),
            "content_type": ctype, "bytes_sha256": hashlib.sha256(raw).hexdigest(), "text_chars": len(text)}
    (CACHE / f"{source_id}.raw").write_bytes(raw)
    (CACHE / f"{source_id}.txt").write_text(text, encoding="utf-8")
    (CACHE / f"{source_id}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def cached_text(source_id: str) -> str | None:
    p = CACHE / f"{source_id}.txt"
    return p.read_text(encoding="utf-8") if p.exists() else None


def grep(source_id: str, pattern: str, width: int = 220):
    t = cached_text(source_id) or ""
    for m in re.finditer(pattern, t, flags=re.I):
        s = max(0, m.start() - width // 2)
        print("...", t[s:m.end() + width // 2], "...\n")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--grep", action="store_true")
    ap.add_argument("source_id")
    ap.add_argument("arg")
    a = ap.parse_args()
    if a.grep:
        grep(a.source_id, a.arg)
    else:
        print(json.dumps(fetch(a.source_id, a.arg), indent=2))


if __name__ == "__main__":
    main()

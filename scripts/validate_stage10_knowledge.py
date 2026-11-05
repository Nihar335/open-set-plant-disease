"""Stage 10 — validate the AgroVaidya knowledge base and its coverage registry (read-only).

  python -P scripts/validate_stage10_knowledge.py

Checks: schema + integrity of every KB record (IDs, relationships, sources, verbatim evidence passages against the
hash-checked local source copies, no doses in our wording, safety notes), and that coverage_registry.json is current.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from stage10.knowledge_base.coverage import OUT, build  # noqa: E402
from stage10.knowledge_base.kb import KnowledgeBase, _evidence, integrity_errors, load_raw  # noqa: E402


def main():
    raw = load_raw()
    errs = integrity_errors(raw)
    kb = KnowledgeBase(raw)
    n_ev = sum(1 for k in ("plants", "conditions", "diagnostics", "management", "crop_requirements") for r in raw[k] for _ in _evidence(r))
    print(f"knowledge base v{kb.version}: {len(kb.sources)} sources, {len(kb.plants)} plants, {len(kb.conditions)} conditions "
          f"({len(kb.sourced_conditions())} sourced), {len(kb.diagnostics)} diagnostics, {len(kb.management)} management records, "
          f"{len(kb.features)} features, {len(kb.questions)} questions, {len(kb.crop_requirements)} crop requirements, "
          f"{n_ev} evidence passages checked")
    for e in errs:
        print("  ERROR", e)
    stale = json.loads(OUT.read_text(encoding="utf-8")) != json.loads(json.dumps(build(kb), ensure_ascii=False)) if OUT.exists() else True
    if stale:
        print("  ERROR coverage_registry.json is stale or missing (python -m stage10.knowledge_base.coverage --write)")
    ok = not errs and not stale
    print("STAGE10 KNOWLEDGE BASE:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

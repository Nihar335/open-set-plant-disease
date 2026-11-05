# AgroVaidya knowledge base — seed v0.1.0 (Stage 10, Milestone 2)

**Status:** this is a small seed knowledge base whose claims have been checked against their sources. **Nothing in it is expert-reviewed** and it is not production-ready.

Covering a plant or condition here does **not** mean the system can diagnose it from an image. The coverage registry tracks those capabilities separately.

## Layout

| Path | Content |
|---|---|
| `KB_VERSION.yaml` | version and status |
| `schemas/kb_records.schema.json` | JSON Schema for every record type (source, plant, condition, diagnostic, management, feature, question, model_class) |
| `data/sources.yaml` | 28 sources: each was fetched and read in full on the access date (title, publisher, exact URL, access date, page update date, licence/terms, byte SHA-256, limitations) |
| `data/plants.yaml` | 8 plants; taxonomy from GBIF Backbone with exact passages |
| `data/conditions.yaml` | 26 conditions: 18 sourced (diseases, a pest, two nutrient deficiencies) and 8 **unsourced stubs** (frozen-model classes not yet sourced) |
| `data/diagnostics.yaml` | 9 diagnostic methods: 7 sourced, 2 `needs_source` |
| `data/management.yaml` | 38 management records; doses are never stated |
| `data/features.yaml`, `data/questions.yaml` | 34 observable features; 12 guided questions, each with a "not sure" option |
| `data/model_classes.yaml` | the frozen model's 21 known classes and 5 held-out classes, mapped to records |
| `kb.py` | loader and integrity validator |
| `coverage.py`, `coverage_registry.json` | per-plant and per-condition capability registry |

The retrieval engine is in `stage10/retrieval/kb_retrieval.py`.

## Claim-level provenance

Every factual statement is a **claim**: our own wording plus one or more evidence items `{source_id, passage}`. The `passage` must be a verbatim substring of the source text, and `kb.py` re-checks every passage against the local verification copy in `results/stage10/source_cache/`. Those copies are stored for verification only and are **not redistributed**; their byte hash must match the hash recorded in `sources.yaml`. In total 216 evidence passages are checked.

## Curation procedure (follow for every addition)

1. **Find a candidate source** (priority: ICAR → state agricultural universities / KVKs → Indian government → international organisations → peer-reviewed literature; botanical databases for taxonomy). Search results are only pointers.
2. **Fetch and read it.** Run `python -m stage10.tools.fetch_source <SOURCE_ID> <URL>`, then read the extracted text in full. Do not use pages behind bot challenges or logins, or pages you could not open.
3. **Register the source** in `sources.yaml` with its terms, geographic relevance and limitations. Record discrepancies such as a mislabelled link, typos, or foreign registration statements.
4. **Write claims** in your own words, each with verbatim passages. Never put a dose, rate or concentration in claim text or summaries.
5. **Run** `python -P scripts/validate_stage10_knowledge.py` and `python -m stage10.knowledge_base.coverage --write`.
6. **Leave `review_status: source_verified`.** Only a qualified expert may set `expert_reviewed`.

## Notable verification findings

- **Mislabelled link:** TNAU's tomato index links "Leaf Curl" to `tomato_diseases_9.html`, but that page is about **bacterial canker**. It is registered as such, and leaf curl remains an unsourced stub.
- **Foreign registrations on an Indian site:** the TNAU mango anthracnose page's management text names products registered in **Australian states**. Only its symptom and biology claims are used.
- **Source typos and inconsistencies** are recorded, not silently fixed:
  - Cornell spells the pathogen "Exerohilum turcicum";
  - TNAU writes "sports" for "spots", and "250°C";
  - Penn State writes "Botryosphaeria obtuse".
- **POWO blocked automated access** with a bot challenge, which was not bypassed. Taxonomy uses GBIF Backbone instead.
- **Blocked extension pages:** the University of Minnesota and Oklahoma State extension pages returned HTTP 403 and were not used.

## Not yet covered (known gaps)

- **Unsourced stubs:** Tomato bacterial spot, leaf mold and yellow leaf curl virus; Potato early and late blight; Pepper bacterial spot; Apple cedar apple rust; Grape leaf blight (Isariopsis).
- **Diagnostic sources missing:** no source yet for soil/tissue nutrient testing or for plant-pathology laboratory identification. Nutrient corrections therefore stay "confirm first", with the test described only as "source pending".
- **No Hindi content.**
- **No region-specific Indian guidance** beyond Tamil Nadu (TNAU) sources.
- **No licence-verified reference images.**

# AgroVaidya Plant Health Intelligence — Stage 10 extension

This is a new extension built on the frozen research project (Stages 1–9). It is **not** part of the original nine experimental stages, and the research results are unchanged. Everything Stage 10 adds lives in `stage10/`, apart from future Stage-10 scripts and apps, which will carry "stage10" or "agrovaidya" in their names.

> **Status:** see `PROJECT_ROADMAP.md` (authoritative, 8 milestones) and `project_tracker.json`. M1–M3 are complete. M4: automatic plant identification (BioCLIP 2, `plant_id/`) was evaluated on development data and is **not established**; it is disabled and, by scope change SC-M4-03, replaced by a user-provided plant name (`workflow/`). M4 accepted and closed (2026-10-15). M5 in progress: a working local app (`app/`).
>
> - **Done:** the Milestone-1 foundations (configuration, report schemas and validator, frozen-predictor wrapper, data guard, asset manifest); a source-verified **seed knowledge base** (`knowledge_base/`); text/feature-based **candidate retrieval** (`retrieval/`, UNVALIDATED); a **coverage registry**; and the deterministic **report engine** (`report_engine/`), which builds known-disease, investigation and out-of-scope reports with an evidence trace.
> - **Not yet built:** image symptom analysis/highlighting, voice input and output, PDF output, follow-up tracking. A local farmer-facing app exists (M5 in progress).
> - Nothing here is expert-reviewed or a diagnostic tool.

## Contents

| Path | Purpose |
|---|---|
| `BASELINE_AUDIT.md` | Phase-0 audit, protected-file hashes, decisions |
| `PROJECT_ROADMAP.md`, `project_tracker.json`, `CHANGELOG.md` | master roadmap, machine-readable task tracker, change history (`python -m stage10.tools.project_tracker --check`) |
| `DATA_AND_SOURCE_POLICY.md` | data rules, knowledge-base plan, reference-image acquisition plan |
| `config.yaml` | all Stage-10 settings (the root `config.yaml` is frozen) |
| `schemas/*.schema.json` | JSON Schema 2020-12 report contracts (`plant_health_report` and its parts) |
| `schemas/validate.py` | structural + semantic validation (`validate_report`) |
| `osr/osr_adapter.py` | `FrozenOSRAdapter`: wraps the unchanged `src/inference/predictor.OSRPredictor` |
| `guards.py` | refuses any Stage 1–9 final-test image path |
| `knowledge_base/` | seed KB, schemas, loader/validator, coverage registry (see `knowledge_base/README.md`) |
| `retrieval/kb_retrieval.py` | KB candidate retrieval (≤ 3 candidates, abstention, guided questions) |
| `plant_id/` | M4 desk survey (`MODEL_SURVEY.md`) and adapter/evaluation proposal (`PROPOSAL.md`); no model integrated |
| `app/` | **farmer-facing local app (M5, in progress)**: `python -m stage10.app.server` then http://127.0.0.1:8765. Flask API + plain-JS web app; see `app/README.md` and `app/VOICE_AND_PRIVACY.md` |
| `vision/` | photo quality gate (thresholds from Stage 1-9 validation images) |
| `workflow/` | **user-provided plant-name workflow** (scope change SC-M4-03): name entry and resolution, routing to the frozen model (six crops only) or knowledge-based guidance, report; developer CLI `python -m stage10.workflow.cli`; voice is a contract only (see `workflow/PLANT_NAME_WORKFLOW.md`). Automatic plant identification is not used |
| `report_engine/` | deterministic report engine: inputs, engine, checks, treatment/regional policy, en/hi message catalogue (see `report_engine/README.md`) |
| `tools/fetch_source.py` | fetch + cache a source for verification (local copies are not redistributed) |
| `tools/asset_manifest.py` | Stage-10 hash manifest (`ASSET_MANIFEST.json`) plus protected Stage-9 files |
| `tests/` | Stage-10 tests (kept apart from the original `tests/` suite) |

## Commands (run from the repository root)

```bash
python -P -m pytest tests -q                 # ORIGINAL Stage 1-9 suite: must stay 63 passed
python -P -m pytest stage10/tests -q         # Stage-10 suite
python -P scripts/verify_final_state.py      # ORIGINAL frozen-state verifier: must print RESULT: PASS
python -m stage10.tools.asset_manifest --check
python -m stage10.tools.project_tracker --check   # tracker rules + roadmap progress block in sync
python -P scripts/validate_stage10_knowledge.py   # KB integrity + verbatim-passage check + coverage registry freshness
python -m stage10.knowledge_base.coverage --write  # regenerate the coverage registry after KB edits
python -P scripts/generate_stage10_report_examples.py   # end-to-end report examples (validation images only) -> results/stage10/report_examples/
python -m stage10.tools.asset_manifest --write   # after intentional Stage-10 edits; refuses if protected files changed
```

**Running all tests at once:** a bare `python -P -m pytest` at the root collects **both** suites. Report the two counts separately.

**Why no `-P` for the manifest tool:** `python -m stage10...` must run **without** `-P`, so that the repository root is importable. The same applies to `python -m src.data.check_leakage`.

## Evidence-status vocabulary used in every report

| Status | Meaning |
|---|---|
| `observed` | directly visible in the image, or explicitly reported by the user |
| `inferred` | a supported interpretation |
| `unverified` | needs more evidence |
| `not_assessable` | cannot be determined from the available information |

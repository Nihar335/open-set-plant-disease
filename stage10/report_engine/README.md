# AgroVaidya report engine (Stage 10, Milestone 3)

**Status:** this is a deterministic, template-based engine that builds structured `plant_health_report` objects. It has no user interface, PDF output, LLM or new model. **No output is expert-reviewed**, and nothing it produces is a diagnosis.

## Files

| Path | Purpose |
|---|---|
| `inputs.py` | Structured, validated inputs (`ReportRequest`, `ImageInfo`, `ImageObservation`, `Region`) and `image_info_from_file`, which applies the final-test guard and the basic quality gate |
| `engine.py` | `build_report(request) -> ReportResult(report, evidence_trace)` |
| `checks.py` | Engine-level cross-record checks, run on every report before it is returned |
| `policy.py` | Treatment and regional-safety policy: chemical/biological gate, recommendation mapping, region notes |
| `registration_verifications.yaml` | Indian registration verifications for chemical/biological records. **Empty**, so everything is withheld |
| `localize.py` | `localize(report, lang)` re-renders a finished report from its `content_keys`; it never touches diagnostic fields |
| `M3_REVIEW.md` | Milestone 3 review: acceptance criteria, evidence, corrections (gate closed 2026-10-14) |
| `messages.yaml` | Fixed English texts and **unreviewed** Hindi drafts (`pending_fluent_speaker_review`) |
| `../../scripts/generate_stage10_report_examples.py` | End-to-end examples on validation images, written to `results/stage10/report_examples/` |

## Flow

1. **Validate the request.** Inputs are restricted to:
   - knowledge-base feature IDs and question IDs;
   - an Indian state or UT, plus an optional district (never a precise location);
   - a plant name as restricted short text;
   - the frozen adapter's output;
   - an explicit timezone-aware timestamp.

   Anything else raises `ReportInputError`.
2. **Check the frozen-model output, without changing it.** The engine checks that:
   - the threshold equals the frozen Stage-7 value;
   - the decision follows the frozen rule;
   - the crop matches the class;
   - the model identity matches the frozen hashes.

   It then copies the output verbatim. In Hindi reports only translations are added.
3. **Resolve the plant.** The user-confirmed crop is primary. "unsure" and "not_provided" are both treated as unconfirmed.
4. **Choose the report type** (the schema defines the types):

   | Situation | `report_type` |
   |---|---|
   | Image fails the quality gate | `insufficient_image` |
   | User confirmed a plant outside the model's six crops (whether in the KB or not) | `out_of_model_scope` |
   | OSR says `known` and the plant is in scope: either a user-confirmed in-scope crop, or no confirmation (then everything is preliminary) | `known_disease` |
   | Otherwise (OSR `unknown` or `not_run`) | `unknown_investigation` |
5. **Candidates.**
   - **Known class:** the model-class condition becomes `CAND-1`, with `ranking_method: model_class_evidence` and `ranking_score: null`, and only if its knowledge-base record is sourced. Healthy classes, unsourced (stub) classes and crop mismatches give no model candidate, and the report says why.
   - **Retrieval:** further candidates come from the Milestone-2 retrieval (≤ 3 in total, never padded). Retrieval can abstain.
6. **Recommendations.**
   - Diagnostic tests are taken from the candidates' diagnostic records.
   - Management is given **only for candidates on a user-confirmed host crop**, worded "If X is confirmed, the source describes: …", with `depends_on` pointing to the confirmatory tests.
   - Chemical and biological records are withheld (see the safety section).
7. **Assemble the report.** Fixed safety texts and immediate actions are added, along with coverage-registry statements, `evidence_sources` built from the trace, and a report ID: a SHA-256 hash of the canonical request, knowledge-base version and configuration fingerprint.
8. **Validate the result:** `validate_report` with the frozen threshold, then `checks.engine_errors`. Any error raises `ReportEngineError`, and no report is returned.

**Determinism:** an identical request, configuration (`stage10/config.yaml`, `messages.yaml`, the registration file and the engine version) and knowledge base give byte-identical JSON. The timestamp is part of the request.

## Evidence traceability

- **What the trace records:** every knowledge-base claim used by a report element is recorded in `evidence_trace.entries` as `{report_path, kb_record, role, claim, evidence: [{source_id, passage}]}`.
- **What the engine checks:**
  - every source ID used anywhere is listed in `evidence_sources`;
  - every listed source has a traced claim, or is a withheld-management source;
  - every sourced candidate, recommendation and taxonomy element has trace entries for each source it cites;
  - every passage is a knowledge-base evidence passage **and** occurs verbatim in the hash-checked cached source text.
- **Withheld chemical options:** these are listed in `withheld_management`. Product names are never written into the report.

## Treatment and regional safety (policy.py)

- **Chemical and biological products are never recommended.** They become actionable only if `registration_verifications.yaml` has an expert-reviewed entry that cites:
  - an Indian government registration source;
  - a label source;
  - the crop and the use.

  None exists. Even a complete entry makes this milestone raise an error instead of emitting advice. Foreign chemical records are never actionable.
- **No dose, rate, pH, nutrient value, efficacy or recovery claims.** These are blocked by the regex check and the banned-wording check.
- **Nutrient problems stay hypotheses, and no product is named** (owner decision, 2026-10-14). They are emitted only as `suspected_nutrient_issue`: a soil/plant-tissue test plus consultation with the KVK or a qualified expert, with "Suspected only, not confirmed", `depends_on` and safety notes. Nutrient management records are withheld like chemical ones, and an engine check fails if any withheld record's wording appears in a report. `confirmed_nutrient_deficiency` is never emitted.
- **Foreign guidance is labelled.** It carries `international_not_verified_for_India`, an inline label and a risk note.
- **Indian regional guidance carries a note.** Tamil Nadu state-university evidence gets a regional-applicability note whenever the user's state is not Tamil Nadu or is unknown.
- **Missing guidance is stated, not invented.** When there is no general crop-care guidance or no management, the report says so instead of substituting advice.

## Schema findings (Milestone 3)

Each finding was documented, and focused tests were written before the schema was changed (`stage10/tests/test_report_schema_defects.py`).

| ID | Finding | Action |
|---|---|---|
| D-1 | The report had no field for the knowledge-base version, which the Milestone-3 brief requires | `generator.knowledge_base_version` added (required, semantic version); the test fixture was updated |
| D-2 | `source_record.source_type` could not represent the knowledge base's `university_extension_international` and `taxonomic_database` types | Two enum values added; no constraint relaxed |
| D-3 | `report_type` is tied to the OSR decision. So a `known` result whose class belongs to a different crop from the user's confirmed crop must still be typed `known_disease` | **No schema change.** The engine does not present the model's class as a candidate, sets `agrees_with_user: false`, explains the conflict, and ranks candidates for the confirmed crop instead |
| D-5 | An observation did not record *how* it was obtained (checklist, guided answer, confirmed typed text, confirmed voice transcript, image analysis), so it was not ready for multimodal input | Required `input_channel` in `symptom_observation`, consistent with `origin`. There is deliberately no value for an unconfirmed interpretation, so unconfirmed text or voice cannot become an observation. Input: `ReportRequest.reported_channels` |
| D-6 | Fixed texts had no language-independent identity | Required report-level `content_keys` [{path, key, params}]. `localize()` re-renders from them, and the report ID is language-independent. Knowledge-base-derived statements stay English (task M6-08) |
| D-4 | Source geography in reports uses a coarser vocabulary (`international`) than the knowledge base (`international_not_verified_for_India`) | **No schema change.** The finer KB value is kept in the source's `limitations` and in every recommendation's `geographic_applicability` |

## Known limitations

1. **No image symptom analysis yet.** Visible symptoms come from user reports and question answers. The `ImageObservation` path is exercised only by synthetic fixtures, and blur and exposure are not assessed.
2. **Candidate ranking is unvalidated** (see Milestone 2). The model-class candidate rests on a research model that was never validated on field photographs.
3. **Mostly English.** Knowledge-base content is English-only. Hindi exists only for fixed texts, as unreviewed drafts.
4. **Thin source coverage.** Management comes mostly from one Tamil Nadu university or from foreign sources. There is no source-verified general crop-care guidance, and no verified source for the soil/tissue test or the laboratory-identification procedure.
5. **Example inputs are synthetic.** The user answers in the end-to-end examples are made up, and the mango example uses a generated placeholder image.

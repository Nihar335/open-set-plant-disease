# Open-Set Recognition for Plant Disease Classification Under Unseen Diseases and Real-World Domain Shift

> **Research prototype — not an agricultural diagnostic tool.** Every number below was produced by the scripts in this repository and is traceable to a file in `results/`. The project is **experimentally complete** (Stages 1–9); no further experiments are planned.

## Research question

Can a plant-disease classifier tell diseases it has learned from diseases it has **never seen**? This matters most when the unseen disease is on a **crop the model already knows** and looks like a known disease. A second question: does the system treat real-world field photos as unfamiliar because of the photographic **domain**, rather than because of disease novelty?

**Why it matters.** A closed-set classifier must assign every image to one of its training classes. On PlantVillage it reaches 99.24% accuracy. Yet it confidently labels an unseen disease (e.g. Grape Esca) as a known look-alike (Grape Black rot). Open-set recognition (OSR) adds a reject option: *known disease* or *potentially unseen / out-of-distribution*.

## Key finding

A 99%-accurate closed-set classifier still accepts a large share of visually similar same-crop diseases that were excluded from training.

- **Energy vs MSP:** a modest, significant improvement.
- **Feature-space distances:** expose different failure modes (Mahalanobis is best on Corn NLB, worst on the tomato diseases).
- **Supervised contrastive learning:** improves OSR under matched scoring. Most of the large raw gain of "contrastive Energy", however, comes from changing the classifier head.
- **Fusion:** an equal-weight fusion of six signals does **not** beat the strongest single detector.
- **Grape Esca:** remains largely undetected.
- **PlantDoc:** shows that the detectors mostly respond to the field-photo *domain*, not to disease novelty.

## Results

**Primary benchmark — Experiment B, hard same-crop unseen diseases.** 5 unknown diseases (6,164 images) vs 4,091 known test images. Every threshold accepts 95% of **known validation** images. Source: [`results/final/final_master_results.md`](results/final/final_master_results.md).

| Method | Hard AUROC | AUPR-Out | FPR@95TPR | Unknown F1 | Unknown detection | Known acceptance |
|---|---:|---:|---:|---:|---:|---:|
| MSP (Stage 4) | 0.8202 | 0.8826 | 48.90% | 0.6514 | 49.77% | 95.40% |
| Energy (Stage 5) | 0.8362 | 0.8962 | 45.98% | 0.7128 | 57.54% | 94.11% |
| Mahalanobis (Stage 6) | 0.7672 | 0.7904 | 83.00% | 0.2662 | 15.83% | 95.31% |
| Contrastive prototype (Stage 7) | 0.8214 | 0.8269 | 73.82% | 0.4116 | 26.82% | 94.74% |
| Contrastive Mahalanobis (Stage 7) | 0.9015 | 0.9080 | 59.25% | 0.5641 | 40.57% | 95.06% |
| **Contrastive Energy\*** (Stage 7) | **0.9324** | **0.9486** | **34.25%** | **0.7905** | **67.81%** | 94.35% |
| Fusion A, equal-weight mean (Stage 8) | 0.9035 | 0.9206 | 42.12% | 0.7316 | 59.75% | 94.60% |

\* **Contrastive Energy** = supervised-contrastive features + a new logistic-regression head, scored by Energy.
- **Head effect vs representation effect:** the same head on the *original* features already reaches 0.9095. So the representation adds +0.023 AUROC [0.020, 0.026], and the head change adds about +0.073.
- **Post hoc choice:** contrastive Energy is the best of the completed methods, but this was established by comparing them **on the test set**. Treat it as exploratory until confirmed on fresh external data.
- **Closed-set reference:** Stage-3 ResNet-50 accuracy is 99.24% on the full known test (4,091 images) and 99.08% on the strict test (3,257).

| | MSP | Energy | Mahalanobis | Contrastive Energy* | Fusion A |
|---|---:|---:|---:|---:|---:|
| A1 unseen crops (AUROC) | 0.9626 | 0.9432 | 0.7970 | 0.9899 | 0.9744 |
| A2 DTD textures (AUROC) | 0.9863 | 0.9693 | 0.9981 | 0.9991 | 0.9987 |
| C PlantDoc field photos (AUROC) | 0.5713 | 0.5670 | 0.6004 | 0.6831 | 0.6347 |

**Per disease (hard AUROC; detection at the 95%-validation threshold in brackets)** — no single method wins everywhere:

| Unknown disease | MSP | Energy | Mahalanobis | Contrastive Energy* | Fusion A |
|---|---:|---:|---:|---:|---:|
| Tomato Septoria leaf spot | 0.954 (72.7%) | 0.969 (83.6%) | 0.763 (3.3%) | **0.978** (91.1%) | 0.972 (88.0%) |
| Tomato Target Spot | 0.964 (79.1%) | **0.974** (89.7%) | 0.595 (0.3%) | 0.963 (81.8%) | 0.949 (63.5%) |
| Apple Black rot | 0.958 (75.4%) | 0.982 (94.0%) | 0.877 (14.8%) | **0.995** (99.4%) | 0.987 (97.1%) |
| Grape Esca (Black Measles) | 0.527 (1.6%) | 0.504 (3.5%) | 0.760 (7.7%) | **0.837** (22.6%) | 0.741 (9.6%) |
| Corn Northern Leaf Blight | 0.698 (18.2%) | 0.776 (17.7%) | **0.961** (72.5%) | 0.901 (49.4%) | 0.891 (50.5%) |

**Statistical highlights** (paired stratified bootstrap, 1,000 resamples; full list in [`results/final/statistical_summary.md`](results/final/statistical_summary.md)):

| Comparison | Δ hard AUROC [95% CI] | Verdict |
|---|---:|---|
| MSP → Energy | +0.0160 [0.0119, 0.0201] | significant |
| Original → contrastive features, same linear head (post-hoc control) | +0.0229 [0.0198, 0.0259] | significant |
| Energy → Fusion A | +0.0673 [0.0628, 0.0719] | significant |
| Contrastive Energy → Fusion A | −0.0289 [−0.0322, −0.0258] | significant (fusion worse) |
| Contrastive Mahalanobis → Fusion A | +0.0020 [−0.0024, 0.0062] | not significant |

At **exactly matched** 95% known-test acceptance, Fusion A's detection gain over Energy is +3.86 pp [−0.32, 6.62], which is **not significant**.

<p align="center"><img src="results/figures/final/final_method_comparison.png" width="900"></p>
<p align="center"><img src="results/figures/final/final_per_disease_auroc.png" width="900"></p>

## Experimental protocol

<p align="center"><img src="results/figures/final/final_experimental_protocol.png" width="900"></p>

- **Unknown = an entire class** excluded from training, validation-based selection and thresholds. No image is ever labelled unknown at random.
- **Known:** 21 PlantVillage classes on 6 crops (Apple, Corn, Grape, Pepper, Potato, Tomato).
  - Split by **physical leaf**: train 19,126 / validation 4,116 / test 4,091.
  - **Strict test:** 3,257 test images with no near-duplicate candidate in train/validation.
- **Experiment B (primary, hard):** Tomato Septoria leaf spot, Tomato Target Spot, Apple Black rot, Grape Esca, Corn Northern Leaf Blight. Each is a new disease on a known crop with a known look-alike.
- **Experiment A (easy):**
  - A1: 9 classes on 7 unseen crops (18,527 images).
  - A2: DTD textures (1,880 images).
- **Experiment C (domain shift):** PlantDoc field photos of known diseases (1,427) vs unknown diseases or crops (963). The model never saw PlantDoc.
- **Validation-unknown:** Tomato mosaic and Cherry ×2 (2,279 validation images). Reported only. The single exception is the secondary Fusion D, as permitted by the DATASET_CARD.
- **Leakage controls:**
  - maintainer leaf IDs;
  - byte/pixel/D4-hash duplicates;
  - CNN-embedding near-duplicates (cos ≥ 0.98);
  - a SIFT+RANSAC same-leaf verifier.

  `data/splits/PROTOCOL_LOCK.json` hashes every manifest, and `python -m src.data.check_leakage` must end with **PASS**.
- **Calibration rules:**
  - every threshold accepts 95% of **known validation** images and is frozen to disk before the test set is scored;
  - each stage scored the test set once (`results/test_runs.log`, which also records two runs that crashed before producing metrics);
  - unknown test labels are used for evaluation only.

## Architecture

<p align="center"><img src="results/figures/final/final_system_architecture.png" width="900"></p>

- **Stage 3:** ResNet-50 (ImageNet init), cross-entropy on the 21 known classes. Provides the known-disease prediction plus MSP and Energy.
- **Stage 6:** tied Ledoit–Wolf Mahalanobis on the Stage-3 2048-d features.
- **Stage 7:** ResNet-50 (ImageNet init), supervised contrastive loss (temperature 0.07), known classes only.
  - The projection head is removed for evaluation.
  - Prototype distance, Mahalanobis and a logistic-regression head are fitted on train features only.
- **Stage 8:** each score is mapped to its percentile among known-validation images, and Fusion A averages the six percentiles.

## Repository structure

```
config.yaml                    all paths, dataset revisions, protocol and per-stage settings
DATASET_CARD.md                sources, licences, counts, protocol, limitations
OPEN_SET_PLANT_DISEASE_PROJECT.md   original design document and literature table
MODEL_CARD.md  REPRODUCIBILITY.md  STAGE_9_FINAL_PROJECT_REPORT.md
paper/                         plant_disease_osr_paper.md, references.md
data/splits/                   frozen manifests, class_mapping.yaml, PROTOCOL_LOCK.json
data/raw/                      downloaded datasets (not redistributed)
models/baseline/               Stage-3 checkpoint + config_used.yaml
models/contrastive/            Stage-7 checkpoint, config, exported linear head (linear_head.npz)
src/data/                      inventory, leaf ids, duplicates, splitting, check_leakage (validator)
src/models/ src/training/      ResNet-50 classifier, trainer, SupCon loss + class-balanced sampler
src/osr/                       MSP/Energy, Mahalanobis, embeddings, probe (diagnostic), fusion
src/evaluation/                classification + OSR metrics, calibration, paired bootstrap
src/inference/predictor.py     frozen research-prototype predictor used by the CLI and the demo
src/visualization/             all figures
scripts/                       one script per stage (+ verify_final_state, run_final_evaluation, predict)
demo/app.py                    Streamlit research demo
tests/                         unit + protocol tests (pytest)
results/reports/               STAGE_3 … STAGE_8 reports (generated from saved JSON)
results/metrics/ predictions/ tables/ figures/ embeddings*/   frozen outputs
results/final/                 final tables, statistical summary, ablation, inventory, validation report
```

The empty folders `app/`, `Dataset/` and `notebooks/` pre-date the pipeline and are unused.

## Installation

Tested environment: Windows 11, Python 3.12.0, torch 2.5.1+cu121 and an RTX 4050 Laptop GPU (6 GB). See [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Dataset setup

The default sources need **no credentials**. Revisions and SHA-256 hashes are pinned in `config.yaml` and recorded in `data/raw/PROVENANCE.json`.

```bash
python scripts/download_dataset.py        # PlantVillage (HF), PlantDoc (GitHub), DTD; ~3.6 GB transferred
python scripts/inspect_dataset.py         # inventory, integrity, exact duplicates
python scripts/find_near_duplicates.py    # embedding near-duplicates (GPU)
python scripts/verify_same_leaf.py        # SIFT+RANSAC same-leaf verification
python scripts/prepare_dataset.py         # roles, leaf-grouped splits, manifests, lock
python -m src.data.check_leakage          # must end with PASS
```

**Optional Kaggle mirror:** run `python scripts/download_dataset.py --datasets plantvillage --pv-source kaggle`. Credentials come from `~/.kaggle/kaggle.json` or the `KAGGLE_USERNAME`/`KAGGLE_KEY` environment variables. They are never stored in the repository.

**Licences** (see DATASET_CARD §1):
- PlantVillage is declared CC BY-SA 3.0 on its Hugging Face card; ⚠ verify before redistributing.
- PlantDoc is CC BY 4.0.
- DTD has no formal licence ("for research purposes").

## Running experiments

The frozen pipeline, in order. `a|b|c` means: run the script once per step, in that order. **Do not re-run training to reproduce the benchmark** — use the next section.

```bash
python -P scripts/train_baseline.py            # Stage 3 (refuses to overwrite models/baseline/)
python -P scripts/evaluate_baseline.py
python -P scripts/osr_msp.py select|evaluate|report          # Stage 4
python -P scripts/osr_energy.py select|evaluate|report       # Stage 5
python -P scripts/feature_osr.py extract|select|evaluate|report   # Stage 6
python -P scripts/probe_diagnostic.py --rep original         # diagnostic only, not an OSR method
python -P scripts/train_contrastive.py                       # Stage 7 (refuses to overwrite; 88.9 min recorded incl. a ~38 min stall)
python -P scripts/contrastive_osr.py extract|select|evaluate|report
python -P scripts/contrastive_head_control.py                # post-hoc head-matched control
python -P scripts/osr_fusion.py select|evaluate              # Stage 8
python -P scripts/osr_fusion_report.py
```

Built-in safeguards:
- Every `select` step refuses to overwrite its frozen threshold file.
- Both training scripts refuse to overwrite an existing checkpoint.
- The Stage-8 `evaluate` step refuses to re-score once its result file exists.

The Stage 4–7 `evaluate` steps have no such guard. They append to `results/test_runs.log` each time they run, so **do not re-run them**; use the frozen outputs.

## Reproducing the final benchmark (no training, no test re-scoring)

```bash
python -P scripts/verify_final_state.py     # hashes of 74 frozen artifacts, lock, leakage, consistency -> PASS
python -P scripts/run_final_evaluation.py   # final tables + figures from saved metrics
python -m pytest -q                         # full test suite
```

## Running inference

```bash
python -P scripts/export_linear_head.py     # once: saves the Stage-7 linear head (verified on validation)
python -P scripts/predict.py path/to/leaf.jpg [--json]
```

Each prediction reports:
- the closest known disease (Stage-3 classifier) and its top-3 classes;
- MSP, Energy and contrastive Energy, with their frozen thresholds;
- a decision: **KNOWN DISEASE** or **UNKNOWN / NOVEL** ("Potentially unseen / out-of-distribution input").

The decision uses contrastive Energy.

## Demo

```bash
streamlit run demo/app.py
```

- **Input:** upload a leaf image, or pick one of two bundled *validation* examples (never test images).
- **Output:** the closest known class, the novelty scores against their thresholds, and a known/unknown verdict.
- **Warning shown in the app:** it is a research prototype, it misses many unseen diseases, and it flags most field photos as unfamiliar.

## Reproducibility

- **Seeds and determinism:** seed 42 everywhere; cuDNN deterministic.
- **Pinned and locked:** dataset revisions, a locked protocol, and frozen thresholds with creation timestamps.
- **Hash inventory:** `results/final/final_state_inventory.json`.
- **Details:** [REPRODUCIBILITY.md](REPRODUCIBILITY.md). The validation status is in [`results/final/final_validation_report.md`](results/final/final_validation_report.md).

## Limitations

- **Narrow data:** PlantVillage is single-leaf, lab-style imagery. Only 5 hard unseen diseases and 21 known classes were studied.
- **Grape Esca is unsolved:** the best detector rejects 22.6% of Esca images, and 99.7% of Esca images lie nearest to the Grape Black rot centroid.
- **Domain shift is a confound:** field photos of *known* diseases are mostly rejected (Fusion A accepts 2.2%). OOD ≠ unseen disease.
- **The "best method" is exploratory:** it was chosen by comparing completed methods on the same test set.
- **One seed, one split:** potential residual same-leaf similarity may inflate known-class scores (see the strict-test rows).
- **Diagnostic probes are not detectors:** the supervised probes show separability but use unknown labels.
- **Not deployment-ready:** this is not an agricultural or clinical tool. Any use needs human expert verification and field validation.

Full lists: [MODEL_CARD.md](MODEL_CARD.md) and [STAGE_9_FINAL_PROJECT_REPORT.md](STAGE_9_FINAL_PROJECT_REPORT.md).

## Citation

There is no formal publication. If you use this repository, cite it as:

> *Open-Set Recognition for Plant Disease Classification Under Unseen Diseases and Real-World Domain Shift* (research code and report, 2026). [Author and URL to be added by the repository owner.]

Data citations:
- Hughes & Salathé (2015), arXiv:1511.08060.
- Mohanty, Hughes & Salathé (2016), *Front. Plant Sci.* 7:1419.
- Singh et al. (2020), *PlantDoc*, CoDS-COMAD, arXiv:1911.10317.
- Cimpoi et al. (2014), *Describing Textures in the Wild*, CVPR.

Method references: [paper/references.md](paper/references.md).

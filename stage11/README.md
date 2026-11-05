# Stage 11 — LeafScan: automatic symptom detection and disease prediction

The user gives a **leaf photo**, the **plant name** and the **state**, plus an optional district, city or village.
Nothing else is asked.

## Report format

```
Plant Name :
Biological Name of the plant :
Symptoms found :
Predicted disease :
Test to confirm :
Solution for the Predicted disease :
```

## How a prediction is made

1. **Plant name → biological name.** The botanical database (`data/plants.yaml`) has 78 plants with common,
   Hindi, regional and scientific names, so typos and local names still resolve.
2. **Photo → deep-learning model** (`model.py`, ResNet-50, 43 classes, 12 of them healthy, over 14 crops):
   - a **symptom head** scores 25 visible symptoms (`data/symptoms.yaml`);
   - a **Grad-CAM heat map** shows where the model looked.
3. **Location → weather.** The Open-Meteo API gives the last 7 days and a 3-day forecast:
   - it uses the typed place, geocoded inside the chosen state, or the state capital if no place is given;
   - each disease's favourable conditions (`data/diseases.yaml` → `weather`) give a weather-risk level, shown beside the
     disease. It does not change the score (changed in the 2026-10-23 audit; v2 used a ×0.8–1.2 factor).
4. **Decision** (`decision.py`, shared by the app and the evaluation):
   - The image probabilities are limited to the user's crop.
   - If most of the image probability falls outside this crop (**photo-fit** < 1), the ranking is blended
     towards the **detected symptoms matched against the disease database**, and the confidence is scaled down
     (at most 75% from symptoms alone). This replaced the v1 behaviour, which rescaled the probability and inflated it
     (for example, a 29% score was shown as 96%).
   - The top disease at **≥ 60%** (lowered from 80% on 2026-10-23) is reported as the predicted disease.
   - Otherwise the report says **"Possibly <top disease>"** (not confirmed) and gives the **three closest diseases**. Each comes
     with tests to confirm it and a solution.
   - Plants without training photos (rice, wheat, cotton, chilli, okra …) use symptom matching against 20
     database-only diseases, so their result is never confirmed: "Possibly …" with three leads.
5. **Tests and solutions, by state.**
   - `data/diagnostic_tests.yaml` holds 20 tests, from field checks to lab tests.
   - `data/solutions.yaml` gives cultural, biological and chemical options per disease. Chemical options are
     **active ingredients only, with no doses**.
   - `data/state_resources.yaml` covers all 28 states and 8 UTs: SAU / ICAR labs, crop institutes and
     state spray schedules (for example, the SKUAST-Kashmir apple schedule).

## Data

| Source | Images | Use |
|---|---|---|
| PlantVillage (lab) | 54,303 | 38 classes, split by same-leaf group |
| PlantDoc (web / field) | 2,557 | mapped onto 28 of the classes (27 appear in its test split); also used for training |
| AppleLeaf9 (Yang et al. 2022, CC BY 4.0) | 14,056 of 14,582 after cleaning (PlantVillage copies and cross-class duplicates removed) | apple field photos: frog-eye (black rot), Alternaria, Marssonina, grey spot, mosaic, powdery mildew, rust, scab, healthy |
| Kashmir indigenous apple set (Yatoo & Sharma 2024, CC BY 4.0) | 5,515 of 5,612 after cleaning | Alternaria, mosaic, healthy (J&K orchards) |

The 359 images removed as PlantVillage copies are counted across both apple sets.

The extra sets were downloaded from Hugging Face (Project-AgML) and imported by `import_extra.py`:
- a dHash check removes PlantVillage copies;
- near-duplicate groups split train / val / test, so augmented copies don't cross splits.

The curated YAML data are **not expert-reviewed**.

## Results (v2 rule at the original 80% threshold, test set; `results/EVALUATION.md`)

The app now uses rule v3 at 60%; see the audit section and `results/VAL_THRESHOLD_EVALUATION.md`. 60% was checked on validation only.

T = 0.80, β = 0 and m0 = 0.94 were fitted on validation only. The second test run is logged in
`results/test_runs.log`, because the low-photo-fit blending was added after the first v2 run.

| | Lab (PlantVillage, n=5,524) | Field (PlantDoc, n=225) | Apple field (n=1,919) |
|---|---|---|---|
| Top-1, crop given | 99.5% | 83.6% | 98.3% |
| Top-3, crop given | 100% | 96.0% | 99.8% |
| Share given a named disease (≥ 80%) | 99.4% | 73.8% | 98.2% |
| Accuracy of named diseases | 99.8% | 87.3% | 98.9% |
| "Unknown disease": true disease in top 3 | 97.1% | 88.1% | 100% |

v1 (38 classes, archived in `models/stage11/v1`) is described in `results/EVALUATION_v1.md`.

## Commands

```
python -P stage11/import_extra.py    # decode/dedupe/split the downloaded apple sets
python -P stage11/data_index.py      # build index (+ PlantDoc resize cache)
python -P stage11/train.py           # ≈ 45 min on RTX 4050; keep --workers <= 2 on this PC
python -P stage11/evaluate.py        # fit T / beta / m0 on val, score test once
python -m stage11.server             # http://127.0.0.1:8770
python -m pytest stage11/tests -q
```

## 2026-10-23 audit (decision rule v3; same v2 checkpoint)

See `audit/AUDIT_REPORT.md`. Summary of the changes:
- A result is named only for crops whose own healthy **and** disease photos were trained (`KB.model_coverage`).
  Single-class crops (blueberry, raspberry, soybean, citrus, squash) and related-crop groups (chilli, cucurbits)
  give leads only.
- A crop-conflict check (`decision.crop_check`) blocks a named result when the photo looks more like another crop.
- Weather no longer changes the score. It is shown as a risk level next to each disease.
- Unknown results state their reasons (`unknown_reasons`). The 60% figure is a heuristic score, not a probability.
- Crop coverage registry: `results/CROP_COVERAGE.md` (`python -P stage11/coverage.py`).
- Multi-crop evaluation: `results/MULTICROP_EVALUATION.md` (`python -P stage11/evaluate_multicrop.py`).

## Limitations

- **Weak symptom labels.** Each training photo inherits its disease's symptom profile; no symptoms were
  annotated per image.
- **Hand-set weather factor.** No weather-labelled photos exist to learn it.
- **Small, non-independent field test.** PlantDoc results rest on 225 held-out photos from a dataset also used for training. Two duplicate same-class training photos and a third appears in training under another crop's label.
- **Diseases outside the 43 classes** can only appear through symptom matching. For example, a photo with
  large ring-shaped lesions reads to the model as "potato early blight", and is ranked by symptoms instead.

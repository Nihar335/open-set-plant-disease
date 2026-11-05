# Open-Set Recognition for Plant Disease Classification

Can a plant-disease classifier tell the diseases it has learned from the ones it has **never seen**? This is hardest when the unseen disease is on a crop the model already knows and looks like a known disease.

This repository has the code, configuration and trained models for that research, along with two applications built on it.

## [Try LeafScan in your browser](https://nihar335.github.io/open-set-plant-disease/)

Nothing to install. Your photo is analysed in your browser and is not uploaded. The first diagnosis downloads a 47 MB model, which the browser then keeps for later visits.

> **Research prototype, not a diagnostic tool.** Results can be wrong, and the disease and treatment data has not been reviewed by an expert. Check any diagnosis with a qualified agricultural expert before acting on it.

## What's here

| Part | What it does | Runs from a fresh clone? |
|---|---|---|
| **Open-set research** (`src/`, `scripts/`, `demo/`) | Classifies a leaf photo and flags possibly unseen diseases | Yes, for inference. Retraining needs the datasets |
| **LeafScan** (`stage11/`) | Web app: from a leaf photo, plant name and state, reports symptoms, a likely disease, confirmation tests and options | Yes, or use the [online version](https://nihar335.github.io/open-set-plant-disease/) |
| **AgroVaidya** (`stage10/`) | Local web app with voice input and multiple languages | **No**, see [below](#agrovaidya) |

## Setup

You need [Git LFS](https://git-lfs.com) to download the trained models. Tested on Windows 11 with Python 3.12 and an RTX 4050 GPU.

```bash
git lfs install
git clone https://github.com/Nihar335/open-set-plant-disease.git
cd open-set-plant-disease
python -m venv .venv
.venv\Scripts\activate
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install flask
```

## Run it

**LeafScan**, at http://127.0.0.1:8770 (set the `PORT` environment variable to change the port):

```bash
python -m stage11.server
```

Weather comes from the Open-Meteo API, so the place you enter is sent there. Weather is shown beside each disease and doesn't change its score.

**Research predictor**, on one photo:

```bash
python -P scripts/predict.py path/to/leaf.jpg
```

**Research demo**, where you upload a leaf photo. Its two built-in examples need the PlantVillage dataset:

```bash
streamlit run demo/app.py
```

## Results

**Open-set research: hard same-crop unseen diseases.** 5 unseen diseases (6,164 images) against 4,091 known test images. Every threshold accepts 95% of known validation images.

| Method | AUROC | Unseen diseases detected | Known images accepted |
|---|---:|---:|---:|
| MSP | 0.8202 | 49.77% | 95.40% |
| Energy | 0.8362 | 57.54% | 94.11% |
| Mahalanobis | 0.7672 | 15.83% | 95.31% |
| Contrastive Energy | **0.9324** | **67.81%** | 94.35% |
| Equal-weight fusion | 0.9035 | 59.75% | 94.60% |

- The closed-set ResNet-50 reaches 99.24% accuracy on known diseases, yet still accepts many unseen look-alikes as known.
- Contrastive Energy was picked by comparing methods on the test set, so treat it as exploratory. Most of its gain comes from a new classifier head: the same head on the original features already reaches 0.9095.
- Grape Esca remains largely undetected: the best detector finds 22.6% of it.
- The detectors mostly respond to field photos as such, not to disease novelty. On PlantDoc field photos, the best AUROC is 0.6831.

**LeafScan: ResNet-50 with 43 classes across 14 crops, with the crop given.**

| | Lab (PlantVillage, n=5,524) | Field (PlantDoc, n=225) | Apple field (n=1,919) |
|---|---:|---:|---:|
| Top-1 accuracy | 99.5% | 83.6% | 98.3% |
| Top-3 accuracy | 100% | 96.0% | 99.8% |

These figures come from an earlier version of the decision rule. **The app now uses a later rule and names a disease at 60% instead of 80%, and that setting was checked on validation photos only, not on a test set.** Other crops have less field evidence, and the PlantDoc photos come from a dataset also used in training, so they aren't an independent test.

## Repository layout

```
config.yaml           paths, dataset revisions, protocol and settings for each stage
requirements.txt      Python packages
data/splits/          dataset split lists, class mapping and protocol lock
models/               trained weights (Git LFS) and their settings
src/                  data handling, models, training, open-set scoring, evaluation, inference
scripts/              one script per research stage, plus predict.py
demo/                 Streamlit research demo
tests/                research tests
results/              only the saved thresholds and values that the code loads
stage10/              AgroVaidya app, knowledge base, report engine and plant-name workflow
stage11/              LeafScan app, model, decision rule, disease data and tests
webapp/               source of the browser version of LeafScan, and its tests
docs/                 the built browser version, served by GitHub Pages
deploy/huggingface/   setup for hosting the server version on a paid Hugging Face Space
```

Reports, figures, full results, datasets and third-party models aren't published here. Some READMEs in subfolders, and some code comments, still mention those files.

## Retraining

The default dataset sources need no credentials. Revisions and hashes are pinned in `config.yaml`.

```bash
python scripts/download_dataset.py
python scripts/inspect_dataset.py
python scripts/find_near_duplicates.py
python scripts/verify_same_leaf.py
python scripts/prepare_dataset.py
python -m src.data.check_leakage
```

Then run the stage scripts in `scripts/`, starting with `train_baseline.py`. The training scripts refuse to overwrite existing weights, so move `models/baseline/` and `models/contrastive/` aside first.

LeafScan's training scripts are `stage11/import_extra.py`, `data_index.py`, `train.py` and `evaluate.py`. They also need the AppleLeaf9 and Kashmir apple datasets (from Hugging Face, Project-AgML) in `stage11_local/`, and no script downloads those.

## Browser version of LeafScan

The online version runs the same model with [ONNX Runtime Web](https://onnxruntime.ai), and `webapp/` is a JavaScript port of LeafScan's decision rule, report and image preparation. There's no server, so the only outside calls are to the Open-Meteo weather service. It differs from the server version in two ways:

- Model weights are stored in half precision to halve the download. Scores can differ slightly, so a shown percentage may differ by a point. On 129 validation photos, the verdict and the ranked candidates matched the PyTorch app every time.
- Photos are decoded by the browser, so results can differ very slightly from the Python decoder.

It has been tested in Chromium-based browsers on a desktop, including at phone width, but not yet on real phones, Safari or Firefox.

To rebuild it after changing the app (`pip install onnx==1.17.0 onnxruntime==1.20.1`, and download the `onnxruntime-web@1.20.1` npm package):

```bash
python -P webapp/export_onnx.py path/to/leafscan.onnx
python -P webapp/build_site.py path/to/leafscan.onnx path/to/onnxruntime-web/package
```

`webapp/tests/` compares the port with the Python engine. It generates expected results with `make_fixtures.py`, then checks them with `core.test.js`.

## AgroVaidya

AgroVaidya won't start from a clone. Its knowledge base checks each source against saved copies of third-party web pages, and those copies aren't redistributed. It stops with a `verification cache missing` error until they exist in `results/stage10/source_cache/`. `stage10/tools/fetch_source.py` is the script that creates them, but this hasn't been tested from a clone. Voice input also needs local speech models, which aren't included.

## Tests

```bash
python -m pytest tests stage11/tests -q
```

Without the datasets, some of these tests fail with missing-file errors. The Stage 10 tests can't run from a clone, for the reason [above](#agrovaidya), and some of them check records that aren't published.

## Limitations

- **Narrow data:** PlantVillage is single-leaf, lab-style imagery, and only 5 hard unseen diseases were studied.
- **One seed, one split:** possible remaining same-leaf similarity may inflate known-class scores.
- **Weak symptom labels:** LeafScan's symptom labels come from each disease's profile, not from marking symptoms on each photo.
- **Plants outside the model:** for plants with no training photos, LeafScan only matches symptoms against its data and never gives a confirmed result.
- **No doses:** chemical options list active ingredients only.

## Data sources

Check each dataset's licence before redistributing it.

- **PlantVillage:** Hughes & Salathé (2015), arXiv:1511.08060; Mohanty, Hughes & Salathé (2016), *Front. Plant Sci.* 7:1419. Declared CC BY-SA 3.0 on its Hugging Face card; verify before redistributing.
- **PlantDoc:** Singh et al. (2020), CoDS-COMAD, arXiv:1911.10317. CC BY 4.0.
- **DTD:** Cimpoi et al. (2014), *Describing Textures in the Wild*, CVPR. No formal licence ("for research purposes").
- **AppleLeaf9:** Yang et al. (2022). CC BY 4.0.
- **Kashmir indigenous apple set:** Yatoo & Sharma (2024). CC BY 4.0.

## Licence

No licence has been chosen yet, so all rights are reserved by default. Contact the repository owner about reuse.

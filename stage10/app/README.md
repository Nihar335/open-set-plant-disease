# AgroVaidya farmer-facing app (Milestone 5)

This is a local web app that turns the verified Stage-10 backend into a usable plant-health assistant:
- the plant-name workflow;
- routing to the frozen six-crop disease model;
- the knowledge base;
- the report engine.

It runs on this computer only.

## Run it

**One-time set-up for voice** (local, offline; the app also works without it, with typing and tapping):

1. Create the isolated speech environment:

   ```bash
   python -m venv C:/Users/nihar/venvs/agrovaidya-speech
   ```

   ```bash
   C:/Users/nihar/venvs/agrovaidya-speech/Scripts/python.exe -m pip install vosk==0.3.45 kokoro-onnx==0.6.1
   ```

2. Download the model files into `stage10_local/models/speech/`. They are all non-gated. URLs and SHA-256 hashes are in `speech/SPEECH_DECISION.md` §8 and `speech/worker.py`:
   - `vosk-model-hi-0.22`, `vosk-model-en-in-0.5`, `vosk-model-small-hi-0.22`, `vosk-model-small-en-in-0.4` (from alphacephei.com, unzipped);
   - `kokoro-v1.0.int8.onnx` and `voices-v1.0.bin` (from the kokoro-onnx GitHub release `model-files-v1.0`).
3. `stage10/config.yaml` → `speech.worker_python` must point at that environment's `python.exe`.

**Start** from the project root (`DL_PROJECT`), with the project's Python environment:

```bash
python -m stage10.app.server
```

Then open http://127.0.0.1:8765.

- **Claude Code:** the `agrovaidya-app` entry in `.claude/launch.json` starts the same command.
- **Configuration** (optional environment variables):

| Variable | Default | Purpose |
|---|---|---|
| `AGROVAIDYA_HOST` | `127.0.0.1` | Address the server listens on |
| `AGROVAIDYA_PORT` | `8765` | Port |
| `AGROVAIDYA_DEVICE` | the predictor's own choice | `cpu` or `cuda` |

- **Dependencies:** listed in `requirements-app.txt`. Everything needed was already installed. There are no npm packages.
- **First photo check:** the first check for one of the six crops loads the disease model, which takes about 10–20 seconds.

## Screens

| Screen | What it does (all backed by the real server) |
|---|---|
| Home | What the app does, its steps, limits and the privacy line |
| Plant (step 2) | Type a name, choose one of the 8 knowledge-base plants, or say "I don't know". Misspelt or partial names get suggestions only; unknown names can be kept as "unlisted". The user confirms, and can change it later. Each plant shows whether the photo disease check applies |
| Home | Language choice (English, Hindi, Bengali, Telugu, Marathi, Tamil; non-English are unreviewed drafts) and what the app does |
| Photo (step 1) | Take a photo (live camera: Capture, Retake, Use, Cancel) or upload one; Take/Upload stay available after a photo is added. Quality gate with retake requests |
| Symptoms (optional, from the report) | Three ways to report: tick symptoms (seen / not seen / clear); describe them in words, where suggestions need explicit "Add"; or speak, which is shown as not available yet. Also: optional guided questions with "not sure", contradiction notices, state (optional) and report language (Hindi draft, clearly labelled) |
| Review | A summary, with links to edit each part; submit; progress and safe error states |
| Report (step 3) | Crop, biological name, temperature, soil, nutrients, signs seen, the disease or up to three possible diseases with checks and what to do; listen on every section; full technical report underneath. Result banner by report type. Shows: the plant as given by you (not checked); whether the photo check ran and why; possible causes marked "not confirmed", with reported vs typical symptoms; how to confirm; what to do now; expert referral; how sure the report is; safety warnings; sources; symptom highlighting (unavailable); listen (unavailable) |
| Help & privacy | How it works, limits, privacy facts, languages |

## Architecture

- **Backend:** `server.py` (Flask) exposes `/api/v1/*`. Every decision is made here, by the existing modules:
  - `stage10.workflow.plant_name` (resolver);
  - `stage10.workflow.assessment` (routing, then `FrozenOSRAdapter`, then the M3 report engine);
  - `stage10.vision.quality` (quality gate);
  - `case_record.py` (each stage kept separate, M5-A4);
  - `symptom_text.py` (typed-description suggestions);
  - `spoken.py` (spoken-report script; no speech engine).
- **Frontend:** `static/`, plain ES modules, no build step and no third-party assets.

| Module | Role |
|---|---|
| `js/app.js` | Views and router |
| `js/state.js` | In-memory case state |
| `js/api.js` | Transport only |
| `js/media.js` | Photo, camera and the gated microphone flow |
| `js/report-model.js` | Display model built from the structured report |
| `js/i18n.js` | Translation interface; catalogues in `i18n/` |

- **Endpoints:**

| Method | Path |
|---|---|
| GET | `health`, `plants`, `symptoms`, `regions`, `i18n/<lang>`, `speech/status` |
| POST | `plant-name/resolve`, `symptoms/suggest`, `image/check` (multipart), `assess` (multipart; optional `marks`), `speech/transcribe` (16 kHz WAV), `speech/synthesize`, `speech/warmup` |

- **Errors** are returned as `{"ok": false, "error": {"code", "message"}}`, with no stack traces or paths.

## Privacy (what the implementation actually does)

- **Photos stay in memory.** Uploaded photos are processed in memory and are not written to disk by AgroVaidya. Werkzeug's default temporary-file spooling is replaced with in-memory buffers, and a test fails if a temporary file is created. Nothing is stored between requests.
- **No outside services.** The page loads nothing from the internet (Content-Security-Policy `'self'`; a test checks for external URLs). The server calls no outside service.
- **Browser side.** The case is kept in the tab's memory only; there is no localStorage, sessionStorage or cookies (test). Reloading clears it.
- **Camera.** It is requested only when the user presses "Take photo"; every track stops on Cancel, Use, leaving the photo step, hiding the tab or closing the page. Only one frame is grabbed, in memory.
- **Microphone.** Requested only after the voice consent and a button press; nothing is recorded or saved (`VOICE_AND_PRIVACY.md`).
- **Location.** Only the state, and that is optional.
- **Data guard.** An upload whose bytes match a Stage 1–9 final-test image (MD5) is refused.

## Capability status

**Not available, and shown as such in the app:**

| Capability | Status |
|---|---|
| Automatic plant identification | Disabled (SC-M4-03). Never imported (test) |
| Visible signs in the photo | **Implemented (SC-M5-02), not validated on field photos.** Colour/shape analyser lists signs (spots, yellowing, halo…); labelled inferred |
| Automatic symptom highlighting on the photo | **Future enhancement (SC-M5-01).** The frozen model gives no lesion locations, and no licensed annotations or validated method exist. Nothing is drawn automatically |
| Voice input and reading aloud | **Working locally, not yet verified with people.** Offline Vosk + Kokoro in an isolated worker (`speech/SPEECH_DECISION.md` §8, `VOICE_AND_PRIVACY.md`). English: not verified; Hindi: draft |
| Farmer-guided marking | **Available; required by SC-M5-01 (accepted 2026-10-15).** Tap, drag or grid on the photo step. Labelled as user-selected; never used in the assessment; the original photo is never altered (`SYMPTOM_MARKING_DECISION.md`) |

**Photo quality thresholds:** set from 400 Stage 1–9 **validation** images with synthetic degradations (`stage10/vision/quality_calibration.json`). These are laboratory photos, so the thresholds are conservative and must be re-checked on field photos.

## Tests

```bash
python -m pytest stage10/tests/test_app.py stage10/tests/test_sc_m5_02.py stage10/tests/test_speech_and_marking.py stage10/tests/test_speech_local.py stage10/tests/test_m5_content.py
```

`test_speech_local.py` uses the real engines. It is skipped, with a stated reason, if they are not installed.

This runs the API and integration tests, the Node frontend unit tests, and one end-to-end run through the real disease model on a validation image.

```bash
node --test stage10/app/tests_js/frontend.test.js stage10/app/tests_js/camera.test.js stage10/app/tests_js/marking_voice.test.js stage10/app/tests_js/voice_module.test.js
```

This runs the frontend unit tests on their own; the report-fixture test is skipped without fixtures.

**In-browser UI self-test:** open http://127.0.0.1:8765/?selftest=1#/home. When it finishes, the results appear at the bottom of the page and in `window.__selftest`. It covers:
- labels and accessible names, keyboard reachability, contrast and horizontal overflow;
- mode switching without losing state;
- the full Tomato (real model), Mango, "don't know" and unlisted flows.

It does **not** establish accessibility compliance. No screen-reader test has been done.

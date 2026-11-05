"""Streamlit research demo — open-set plant disease recognition (frozen Stage 3-8 models, no tuning).

  streamlit run demo/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.inference.predictor import OSRPredictor  # noqa: E402


def pretty(c: str) -> str:
    return c.replace("Corn_(maize)", "Corn").replace("Pepper,_bell", "Pepper").replace("___", " · ").replace("_", " ")


st.set_page_config(page_title="Open-set plant disease recognition — research prototype", layout="wide")
st.title("Open-set plant disease recognition")
st.warning("**Research prototype — not a production agricultural diagnostic system.** It was trained on 21 PlantVillage "
           "lab-photo classes (6 crops). It misses many unseen diseases (e.g. it rejects only ~23% of Grape Esca images) and "
           "flags most field photos as unfamiliar even when the disease is known. Do not use it for treatment or pesticide decisions.")


@st.cache_resource
def load():
    return OSRPredictor()


with st.sidebar:
    st.header("What this shows")
    st.markdown("- **Closest known disease** from the Stage-3 ResNet-50 (99.2% accuracy on known PlantVillage test images).\n"
                "- **Novelty score**: contrastive Energy (Stage 7). Threshold accepts 95% of known *validation* images.\n"
                "- The original Energy score (Stage 5) is shown for comparison.\n\n"
                "Hard same-crop unseen diseases: AUROC 0.932 (contrastive Energy), 0.836 (original Energy). "
                "See `results/final/final_master_results.md`.")

EXAMPLES = {  # VALIDATION-split images only (never final-test images)
    "— none —": None,
    "Known class: Apple scab (validation image)": "data/raw/plantvillage/color/Apple___Apple_scab/01f3deaa-6143-4b6c-9c22-620a46d8be04___FREC_Scab 3112.JPG",
    "Validation-unknown: Cherry healthy (validation image)": "data/raw/plantvillage/color/Cherry_(including_sour)___healthy/0008f3d3-2f85-4973-be9a-1b520b8b59fc___JR_HL 4092.JPG",
}
up = st.file_uploader("Upload a leaf image (JPG/PNG)", type=["jpg", "jpeg", "png"])
ex = st.selectbox("…or try a validation example (requires the local PlantVillage download)", list(EXAMPLES))
if up is not None:
    img = Image.open(up).convert("RGB")
elif EXAMPLES[ex] and (ROOT / EXAMPLES[ex]).exists():
    img = Image.open(ROOT / EXAMPLES[ex]).convert("RGB")
else:
    st.info("Upload an image to start. Best results are for single-leaf photos on a plain background, like PlantVillage.")
    st.stop()

with st.spinner("Running the frozen models…"):
    r = load().predict(img)
c1, c2 = st.columns([1, 1.3])
c1.image(img, use_container_width=True)
with c2:
    if r["decision"] == "UNKNOWN / NOVEL":
        st.error(f"### UNKNOWN / NOVEL\n{r['message']}. It may be an unseen disease, another crop, a non-leaf image, or simply "
                 "a photo unlike the training images. This is **not** a diagnosis of a new disease.")
    else:
        st.success(f"### KNOWN DISEASE\n{r['message']}.")
    st.metric("Closest known class (Stage-3 classifier)", pretty(r["predicted_known_class"]), f"softmax {r['top3'][0][1]:.3f}")
    st.markdown("**Top-3 known classes**: " + "; ".join(f"{pretty(c)} ({p:.3f})" for c, p in r["top3"]))
    st.markdown("#### Novelty scores (higher = more unfamiliar)")
    st.table({
        "score": ["Contrastive Energy (decision)", "Original Energy", "MSP (max softmax)"],
        "value": [f"{r['contrastive_energy']:.3f}", f"{r['energy']:.3f}", f"{r['msp']:.4f}"],
        "unknown if": [f"score above {r['contrastive_energy_threshold']:.3f}", f"score above {r['energy_threshold']:.3f}",
                       f"score below {r['msp_threshold']:.4f}"],
        "verdict": ["unknown" if r["contrastive_energy_says_unknown"] else "known", "unknown" if r["energy_says_unknown"] else "known",
                    "unknown" if r["msp_says_unknown"] else "known"],
    })
    st.progress(min(max(r["contrastive_energy_percentile_known_val"], 0.0), 1.0),
                text=f"Contrastive-Energy novelty is higher than {100 * r['contrastive_energy_percentile_known_val']:.1f}% of known validation images")
    st.caption("Thresholds are frozen from known validation data (95% acceptance). Detectors can disagree; a 'known' verdict "
               "does not rule out an unseen look-alike disease.")

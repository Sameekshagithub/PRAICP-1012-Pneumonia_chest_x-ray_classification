"""
PNEUMO-SCAN AI — Streamlit front-end for the Pneumonia Chest X-Ray Classifier.

Run with:
    streamlit run app.py
"""
import base64
import glob
import io
import json
import os

import numpy as np
import streamlit as st
from PIL import Image, ImageOps

from src import config
from src.gradcam import make_gradcam_heatmap, overlay_heatmap
from src.predict import predict

# ----------------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Pneumo-Scan AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Dark / gray "radiology lightbox" theme
# ----------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@500;600;700&display=swap');

:root {
    --bg-black: #050506;
    --panel-dark: #131315;
    --panel-mid: #1b1c1f;
    --border-soft: rgba(255,255,255,0.08);
    --text-main: #e5e7eb;
    --text-dim: #8b8d93;
    --accent-cyan: #5fd3e0;
    --accent-green: #34d399;
    --accent-red: #f87171;
    --mono-font: 'Share Tech Mono', 'JetBrains Mono', monospace;
    --head-font: 'Rajdhani', sans-serif;
}

html, body, [class*="stApp"] {
    background: radial-gradient(circle at 20% 0%, #101012 0%, #08080a 45%, #050506 100%) !important;
    color: var(--text-main) !important;
    font-family: var(--mono-font) !important;
}

/* subtle CRT/lightbox scanline overlay */
[class*="stApp"]::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    background: repeating-linear-gradient(
        to bottom, rgba(255,255,255,0.015) 0px, rgba(255,255,255,0.015) 1px,
        transparent 2px, transparent 4px
    );
    z-index: 0;
}

/* headings */
h1, h2, h3, h4 {
    font-family: var(--head-font) !important;
    letter-spacing: 0.03em;
    color: var(--text-main) !important;
}

/* hide default streamlit chrome */
#MainMenu, footer, header {visibility: hidden;}

/* sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0c0c0e 0%, #060607 100%) !important;
    border-right: 1px solid var(--border-soft);
}
section[data-testid="stSidebar"] * { color: var(--text-main) !important; }

/* panel / card helper */
.panel {
    background: var(--panel-dark);
    border: 1px solid var(--border-soft);
    border-radius: 14px;
    padding: 22px 24px;
    margin-bottom: 18px;
    box-shadow: 0 0 22px rgba(0,0,0,0.55);
}

.brand-title {
    font-family: var(--head-font);
    font-size: 2.1rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    color: var(--text-main);
    margin-bottom: 0;
}
.brand-sub {
    font-family: var(--mono-font);
    color: var(--accent-cyan);
    font-size: 0.85rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-top: -6px;
}
.hr-glow {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--accent-cyan), transparent);
    margin: 14px 0 22px 0;
    opacity: 0.6;
}

/* upload widget */
[data-testid="stFileUploader"] section {
    background: var(--panel-mid) !important;
    border: 1.5px dashed rgba(255,255,255,0.18) !important;
    border-radius: 12px !important;
}
[data-testid="stFileUploader"] label { color: var(--text-dim) !important; }

/* buttons */
.stButton>button, .stDownloadButton>button {
    background: transparent;
    border: 1px solid var(--accent-cyan);
    color: var(--accent-cyan) !important;
    border-radius: 8px;
    font-family: var(--mono-font);
    letter-spacing: 0.05em;
    padding: 6px 18px;
    transition: all 0.2s ease-in-out;
}
.stButton>button:hover, .stDownloadButton>button:hover {
    background: var(--accent-cyan);
    color: #05060a !important;
    box-shadow: 0 0 16px rgba(95,211,224,0.55);
}

/* tabs */
button[data-baseweb="tab"] {
    font-family: var(--head-font) !important;
    letter-spacing: 0.05em;
    color: var(--text-dim) !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--accent-cyan) !important;
    border-bottom: 2px solid var(--accent-cyan) !important;
}

/* verdict cards */
.verdict-card {
    border-radius: 14px;
    padding: 26px 26px 20px 26px;
    text-align: center;
    font-family: var(--head-font);
}
.verdict-normal {
    background: radial-gradient(circle at 50% 0%, rgba(52,211,153,0.15), rgba(19,19,21,0.9));
    border: 1px solid rgba(52,211,153,0.55);
    box-shadow: 0 0 30px rgba(52,211,153,0.18);
}
.verdict-pneumonia {
    background: radial-gradient(circle at 50% 0%, rgba(248,113,113,0.16), rgba(19,19,21,0.9));
    border: 1px solid rgba(248,113,113,0.55);
    box-shadow: 0 0 30px rgba(248,113,113,0.20);
}
.verdict-label {
    font-size: 1.9rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    margin: 0;
}
.verdict-label.normal { color: var(--accent-green); }
.verdict-label.pneumonia { color: var(--accent-red); }
.verdict-caption {
    font-family: var(--mono-font);
    color: var(--text-dim);
    font-size: 0.8rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 2px;
}

/* confidence bar */
.conf-track {
    width: 100%;
    height: 14px;
    background: #0c0c0e;
    border-radius: 8px;
    border: 1px solid var(--border-soft);
    margin-top: 16px;
    overflow: hidden;
}
.conf-fill {
    height: 100%;
    border-radius: 8px;
}
.conf-fill.normal { background: linear-gradient(90deg, #0f766e, #34d399); box-shadow: 0 0 12px rgba(52,211,153,0.6); }
.conf-fill.pneumonia { background: linear-gradient(90deg, #7f1d1d, #f87171); box-shadow: 0 0 12px rgba(248,113,113,0.6); }
.conf-text {
    font-family: var(--mono-font);
    color: var(--text-dim);
    font-size: 0.78rem;
    margin-top: 6px;
    letter-spacing: 0.05em;
}

/* lightbox-style x-ray image frame */
.xray-frame {
    background: #000;
    border: 1px solid var(--border-soft);
    border-radius: 10px;
    padding: 10px;
    box-shadow: inset 0 0 40px rgba(255,255,255,0.03), 0 0 20px rgba(0,0,0,0.6);
    text-align: center;
}
.xray-frame img {
    max-width: 100%;
    border-radius: 4px;
    filter: grayscale(1) contrast(1.15) brightness(1.02);
}
.xray-caption {
    font-family: var(--mono-font);
    color: var(--text-dim);
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 8px;
}

.disclaimer {
    font-family: var(--mono-font);
    font-size: 0.72rem;
    color: var(--text-dim);
    border-top: 1px solid var(--border-soft);
    padding-top: 12px;
    margin-top: 24px;
    line-height: 1.5;
}

/* metrics table */
table { color: var(--text-main) !important; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Cached resources
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model weights...")
def load_model_cached(model_path):
    from tensorflow.keras.models import load_model
    return load_model(model_path)


def discover_models():
    """Return {display_label: path} for every .h5 checkpoint in models/."""
    paths = sorted(glob.glob(os.path.join(config.MODEL_DIR, "*.h5")))
    from src.models import MODEL_DISPLAY_NAMES
    result = {}
    for p in paths:
        key = os.path.splitext(os.path.basename(p))[0]
        label = MODEL_DISPLAY_NAMES.get(key, key)
        result[label] = p
    return result


def load_metrics():
    if os.path.exists(config.METRICS_PATH):
        with open(config.METRICS_PATH) as f:
            return json.load(f)
    return None


def image_to_base64(pil_image):
    buf = io.BytesIO()
    pil_image.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def xray_frame_html(pil_image, caption):
    b64 = image_to_base64(pil_image)
    return f"""
    <div class="xray-frame">
        <img src="data:image/png;base64,{b64}" />
        <div class="xray-caption">{caption}</div>
    </div>
    """


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🫁 PNEUMO-SCAN AI")
    st.markdown("<div class='brand-sub'>Radiological Screening Console</div>", unsafe_allow_html=True)
    st.markdown("<hr class='hr-glow'/>", unsafe_allow_html=True)

    available_models = discover_models()
    if not available_models:
        st.warning(
            "No trained models found in `models/`.\n\n"
            "Run `python train.py` first (after placing your dataset in `data/`), "
            "then refresh this page."
        )
        selected_model_label, selected_model_path = None, None
    else:
        selected_model_label = st.selectbox("Active Model", list(available_models.keys()))
        selected_model_path = available_models[selected_model_label]

    st.markdown("<hr class='hr-glow'/>", unsafe_allow_html=True)
    show_gradcam = st.checkbox("Show Grad-CAM attention map", value=True)
    st.markdown("<hr class='hr-glow'/>", unsafe_allow_html=True)
    st.markdown(
        "<div class='disclaimer'>"
        "This tool is a demonstration / educational aid only. It is NOT a certified "
        "medical device and must not be used for real clinical diagnosis without "
        "professional oversight."
        "</div>",
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown("<div class='brand-title'>PNEUMO-SCAN AI</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='brand-sub'>AI-Assisted Chest X-Ray Screening &nbsp;•&nbsp; "
    "NORMAL / PNEUMONIA Classification</div>",
    unsafe_allow_html=True,
)
st.markdown("<hr class='hr-glow'/>", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------------
tab_diagnose, tab_performance = st.tabs(["🩻  DIAGNOSE", "📊  MODEL PERFORMANCE"])

# ---- Diagnose tab -----------------------------------------------------------
with tab_diagnose:
    if selected_model_path is None:
        st.info("Train a model first, then come back to this tab to run a diagnosis.")
    else:
        col_upload, col_result = st.columns([1, 1], gap="large")

        with col_upload:
            st.markdown("<div class='panel'>", unsafe_allow_html=True)
            st.markdown("#### Upload Chest X-Ray")
            uploaded_file = st.file_uploader(
                "Accepted formats: JPG, JPEG, PNG", type=["jpg", "jpeg", "png"]
            )
            if uploaded_file is not None:
                pil_image = Image.open(uploaded_file)
                st.markdown(xray_frame_html(pil_image, "UPLOADED SCAN"), unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_result:
            st.markdown("<div class='panel'>", unsafe_allow_html=True)
            st.markdown("#### Diagnosis")

            if uploaded_file is None:
                st.markdown(
                    "<div class='disclaimer'>Awaiting scan upload...</div>",
                    unsafe_allow_html=True,
                )
            else:
                model = load_model_cached(selected_model_path)
                label, confidence, prob, batch = predict(model, pil_image)

                css_class = "pneumonia" if label == "PNEUMONIA" else "normal"
                verdict_html = f"""
                <div class="verdict-card verdict-{css_class}">
                    <div class="verdict-label {css_class}">{label}</div>
                    <div class="verdict-caption">Model: {selected_model_label}</div>
                    <div class="conf-track">
                        <div class="conf-fill {css_class}" style="width:{confidence*100:.1f}%;"></div>
                    </div>
                    <div class="conf-text">CONFIDENCE&nbsp;&nbsp;{confidence*100:.1f}%&nbsp;&nbsp;|&nbsp;&nbsp;
                        RAW PNEUMONIA SCORE&nbsp;{prob:.4f}</div>
                </div>
                """
                st.markdown(verdict_html, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        if uploaded_file is not None and show_gradcam:
            st.markdown("<div class='panel'>", unsafe_allow_html=True)
            st.markdown("#### Grad-CAM — Model Attention Map")
            st.caption(
                "Highlights the regions of the X-ray that most influenced the model's decision. "
                "Warmer colors (red/yellow) indicate higher influence."
            )
            try:
                heatmap = make_gradcam_heatmap(batch, model)
                overlay_img = overlay_heatmap(pil_image, heatmap)

                gc_col1, gc_col2 = st.columns(2)
                with gc_col1:
                    st.markdown(xray_frame_html(pil_image, "ORIGINAL"), unsafe_allow_html=True)
                with gc_col2:
                    st.markdown(xray_frame_html(overlay_img, "MODEL ATTENTION (GRAD-CAM)"), unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"Grad-CAM could not be generated for this model/image: {e}")
            st.markdown("</div>", unsafe_allow_html=True)


# ---- Model Performance tab ---------------------------------------------------
with tab_performance:
    metrics_data = load_metrics()

    if metrics_data is None:
        st.info(
            "No `reports/metrics.json` found yet. Run `python train.py` to train models "
            "and generate performance reports."
        )
    else:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.markdown(f"#### Model Comparison  ·  *last trained: {metrics_data['trained_at']}*")

        import pandas as pd
        from src.models import MODEL_DISPLAY_NAMES

        rows = []
        for key, m in metrics_data["models"].items():
            rows.append({
                "Model": MODEL_DISPLAY_NAMES.get(key, key),
                "Accuracy": round(m["accuracy"], 4),
                "Precision": round(m["precision"], 4),
                "Recall": round(m["recall"], 4),
                "F1-Score": round(m["f1_score"], 4),
                "ROC-AUC": round(m["roc_auc"], 4),
            })
        df = pd.DataFrame(rows).sort_values(by="Recall", ascending=False).reset_index(drop=True)
        st.dataframe(df, use_container_width=True)
        st.caption(
            "Sorted by Recall — the most important metric for a screening task, since missing "
            "a true PNEUMONIA case is more costly than a false alarm."
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.markdown("#### Diagnostic Plots")
        plot_model_key = st.selectbox("Select a model to view its plots", list(metrics_data["models"].keys()))

        cm_path = os.path.join(config.REPORTS_DIR, f"{plot_model_key}_confusion_matrix.png")
        roc_path = os.path.join(config.REPORTS_DIR, f"{plot_model_key}_roc_curve.png")
        curves_path = os.path.join(config.REPORTS_DIR, f"{plot_model_key}_training_curves.png")

        pcol1, pcol2 = st.columns(2)
        with pcol1:
            if os.path.exists(cm_path):
                st.image(cm_path, caption="Confusion Matrix", use_container_width=True)
        with pcol2:
            if os.path.exists(roc_path):
                st.image(roc_path, caption="ROC Curve", use_container_width=True)

        if os.path.exists(curves_path):
            st.image(curves_path, caption="Training Curves", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

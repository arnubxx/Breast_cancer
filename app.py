import os
import io
import numpy as np
from PIL import Image, ImageOps
import streamlit as st
import requests

# Try TensorFlow/Keras. If unavailable at runtime, show a friendly message.
try:
    import tensorflow as tf
except Exception as e:
    tf = None


st.set_page_config(
    page_title="Breast Cancer Classifier",
    page_icon="🩺",
    layout="centered",
)


# --- Styles to approximate the provided mockups ---
st.markdown(
    """
    <style>
    .title {font-weight:700; font-size: 28px;}
    .subtitle {color:#6b7280; font-size:14px;}
    .card {
        border-radius: 16px; padding: 20px; background: #ffffff;
        box-shadow: 0 8px 24px rgba(0,0,0,0.08);
    }
    .gradient-btn {
        background: linear-gradient(90deg,#8b5cf6,#06b6d4);
        color:white; border:none; padding: 12px 18px; border-radius: 12px;
        font-weight:600; cursor:pointer;
    }
    .upload-box {
        border: 2px dashed #cbd5e1; border-radius: 16px; padding: 24px;
        text-align:center; color:#64748b;
        margin-top:12px; margin-bottom:12px;
    }
    .result-title {font-weight:700; font-size:22px;}
    .result-text {font-weight:700; font-size:18px;}
    .confidence-bar {height: 12px; background:#e5e7eb; border-radius: 8px;}
    .confidence-fill {
        height: 12px; border-radius: 8px; background: linear-gradient(90deg,#8b5cf6,#06b6d4);
    }
    .disclaimer {color:#6b7280; font-size:12px; margin-top:8px;}
    </style>
    """,
    unsafe_allow_html=True,
)


# --- Utilities ---
def _local_model_path() -> str:
    return os.path.join(os.getcwd(), "best_model.h5")


def _maybe_download_model() -> str | None:
    """Download model from URL defined in secrets or env if local missing.

    Returns path to downloaded file, or None if not configured.
    """
    url = None
    # Prefer Streamlit secrets
    try:
        url = st.secrets.get("MODEL_URL")
    except Exception:
        url = None
    # Fallback to environment variable
    if not url:
        url = os.environ.get("MODEL_URL")

    if not url:
        return None

    cache_dir = os.path.join(os.getcwd(), ".model_cache")
    os.makedirs(cache_dir, exist_ok=True)
    out_path = os.path.join(cache_dir, "best_model.h5")

    if os.path.exists(out_path):
        return out_path

    with st.spinner("Downloading model…"):
        resp = requests.get(url, stream=True, timeout=300)
        resp.raise_for_status()
        tmp_path = out_path + ".tmp"
        with open(tmp_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        os.replace(tmp_path, out_path)
    return out_path


@st.cache_resource(show_spinner=True)
def load_model():
    if tf is None:
        raise RuntimeError(
            "TensorFlow/Keras is not available. Please install 'tensorflow'."
        )
    local = _local_model_path()
    model_path = local if os.path.exists(local) else _maybe_download_model()
    if not model_path:
        raise FileNotFoundError(
            "Model file not found. Place 'best_model.h5' in the app root or set MODEL_URL in Streamlit secrets/env."
        )
    model = tf.keras.models.load_model(model_path)
    return model


def infer_input_shape(model):
    """Infer (height, width, channels) from model input shape."""
    ishape = model.input_shape
    # ishape expected like (None, H, W, C)
    if isinstance(ishape, (list, tuple)) and len(ishape) >= 4:
        h, w, c = ishape[1], ishape[2], ishape[3]
        if h is None or w is None:
            # Fallback to a common size if dynamic
            h, w = 224, 224
        if c is None:
            c = 3
        return int(h), int(w), int(c)
    # Fallback
    return 224, 224, 3


def preprocess_image(img: Image.Image, target_size, channels):
    # Convert color mode
    if channels == 1:
        img = img.convert("L")
    else:
        img = img.convert("RGB")

    # Resize while preserving aspect ratio
    img = ImageOps.fit(img, target_size, method=Image.Resampling.LANCZOS)

    arr = np.asarray(img).astype("float32") / 255.0
    if channels == 1:
        arr = np.expand_dims(arr, axis=-1)  # (H,W,1)
    arr = np.expand_dims(arr, axis=0)  # (1,H,W,C)
    return arr


def predict(model, arr):
    preds = model.predict(arr, verbose=0)
    probs = preds[0] if preds.ndim == 2 else np.squeeze(preds)
    # Normalize if needed
    if probs.min() < 0 or probs.max() > 1:
        # Softmax fallback
        exp = np.exp(probs - np.max(probs))
        probs = exp / np.sum(exp)
    idx = int(np.argmax(probs))
    conf = float(probs[idx])
    return idx, conf, probs


def birads_labels(num_classes):
    base = [
        "BI-RADS 0: Incomplete",
        "BI-RADS 1: Negative",
        "BI-RADS 2: Benign",
        "BI-RADS 3: Probably Benign",
        "BI-RADS 4: Suspicious",
        "BI-RADS 5: Highly Suggestive",
    ]
    if num_classes <= len(base):
        return base[:num_classes]
    # Extend generic labels if model has more classes
    base.extend([f"Class {i}" for i in range(len(base), num_classes)])
    return base


# --- UI ---
st.markdown('<div class="title">Breast Cancer Classifier</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Upload a mammogram image for BI-RADS classification using AI</div>',
    unsafe_allow_html=True,
)

with st.container():
    st.markdown('<div class="upload-box">Click to upload or drag and drop<br/><span style="font-size:12px">Mammogram Image - PNG, JPG or JPEG (MAX. 10MB)</span></div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Click to upload or drag and drop", type=["png", "jpg", "jpeg"], accept_multiple_files=False)

classify_clicked = st.button("Classify Image", use_container_width=True)


result_placeholder = st.empty()

def render_result(image: Image.Image, label_text: str, confidence: float):
    with result_placeholder.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="result-title">BI-RADS Classification Result</div>', unsafe_allow_html=True)
        st.caption("Mammogram analysis complete")
        st.image(image, caption="Analyzed Mammogram", use_column_width=True)
        st.markdown(f'<div class="result-text">{label_text}</div>', unsafe_allow_html=True)

        pct = int(round(confidence * 100))
        st.write(f"Confidence: {pct}%")
        st.markdown('<div class="confidence-bar">', unsafe_allow_html=True)
        st.markdown(f'<div class="confidence-fill" style="width:{pct}%;"></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(
            '<div class="disclaimer">This tool is for research and educational use only and does not provide medical advice.</div>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)


if classify_clicked:
    if uploaded is None:
        st.warning("Please upload an image first.")
    else:
        try:
            # Size guard ~10MB
            if hasattr(uploaded, "size") and uploaded.size and uploaded.size > 10 * 1024 * 1024:
                st.error("File exceeds 10MB limit.")
            else:
                img = Image.open(io.BytesIO(uploaded.read()))
                if tf is None:
                    st.error("TensorFlow is not installed. Add 'tensorflow' to requirements and deploy again.")
                else:
                    model = load_model()
                    h, w, c = infer_input_shape(model)
                    arr = preprocess_image(img, (h, w), c)
                    idx, conf, probs = predict(model, arr)
                    labels = birads_labels(len(probs))
                    label_text = labels[idx]
                    render_result(img, label_text, conf)
        except Exception as e:
            st.error(f"Failed to classify: {e}")

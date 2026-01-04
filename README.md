# Breast Cancer Classifier – Streamlit Frontend

This Streamlit app provides a simple frontend for classifying mammogram images into BI-RADS categories using a pre-trained Keras/TensorFlow model saved as `best_model.h5`.

## Quick Start (Local)

1. Ensure the model file exists at the project root: `best_model.h5`.
2. Create a virtual environment (recommended) and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. Run the app:

```bash
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Push these files to a public GitHub repo:
   - `app.py`
   - `requirements.txt`
   - `best_model.h5` (or use remote storage and download at startup)
   - `README.md`

2. Go to https://share.streamlit.io, create a new app pointing to your repo.
3. The app will build using `requirements.txt` and start automatically.

### Using a remote model (recommended for large files)
If `best_model.h5` is not committed, set a `MODEL_URL` secret so the app can download the model at startup.

Steps:
- In Streamlit Cloud, open your app → Settings → Secrets, add:

```
MODEL_URL = https://<your-storage>/<path>/best_model.h5
```

- Alternatively, set an environment variable `MODEL_URL` in the app settings.
- On run, the app downloads the model into `.model_cache/best_model.h5` and loads it.

Supported storages: GitHub Releases, S3/CloudFront, GCS signed URL, or any direct HTTPS link.

### Notes
- If `best_model.h5` is large (>100MB), consider Git LFS or hosting the model at a URL and downloading it at startup.
- The app infers model input size from `model.input_shape`. Update preprocessing if your model expects special normalization.
- Update `birads_labels()` in `app.py` to match your model's label definitions exactly.
 - For Apple Silicon local runs, install `tensorflow-macos` and `tensorflow-metal` instead of the CPU TensorFlow in `requirements.txt`.

## Disclaimer
This tool is for research and educational purposes only and does not constitute medical advice or diagnostic guidance.
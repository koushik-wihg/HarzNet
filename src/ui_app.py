# --- FIX FOR RENDER: ensure project root is on sys.path ---
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # -> /app on Render
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# --- END FIX ---

import streamlit as st, pandas as pd, joblib, os, io
from pathlib import Path
from src.utils.common import read_params
import numpy as np

# Set page configuration must be at the very top
st.set_page_config(page_title="Ultramafic ML UI", layout="wide")

CONFIG_PATH = Path("Config/params.yaml")
params = read_params(CONFIG_PATH)
MODEL_PATH = params.get("api", {}).get("model_path", "models/final_pipeline.joblib")

st.title("Ultramafic ML - Streamlit UI")

pipeline = None
label_encoder = None
loaded_from = None

# Load combined pipeline (preferred) or fallback
PIPE_PATHS = [Path("models/final_pipeline.joblib"), Path("models/best_model_tuned.joblib"), Path("models/pipeline.joblib")]
for p in PIPE_PATHS:
    if p.exists():
        try:
            obj = joblib.load(p)
            if isinstance(obj, dict):
                if 'pipeline' in obj:
                    pipeline = obj['pipeline']
                    label_encoder = obj.get('label_encoder', None)
                elif 'model' in obj:
                    if hasattr(obj['model'], "named_steps"):
                        pipeline = obj['model']
                    else:
                        pipeline = obj['model']
                    label_encoder = obj.get('label_encoder', None)
            else:
                if hasattr(obj, "predict"):
                    pipeline = obj
            loaded_from = p
            break
        except Exception as e:
            st.sidebar.error(f"Failed to load {p}: {e}")

if pipeline is None:
    st.sidebar.warning("No pipeline detected. Please run training to create models/final_pipeline.joblib.")
else:
    st.sidebar.success(f"Loaded pipeline from {loaded_from}")

# Attempt to load feature metadata so we can validate uploaded CSVs
try:
    feat_info_path = Path("models/feature_names.pkl")
    if feat_info_path.exists():
        feat_info = joblib.load(feat_info_path)
        numerical_cols = feat_info.get('numerical_cols', None)
        remainder_cols = feat_info.get('remainder_cols', None)
    else:
        numerical_cols = None
        remainder_cols = None
except Exception:
    numerical_cols = None
    remainder_cols = None


def read_uploaded_file(uploaded_file, file_path_hint):
    file_extension = Path(file_path_hint).suffix.lower()
    content = uploaded_file.getvalue()
    if file_extension == '.csv':
        st.info(f"Reading {uploaded_file.name} as CSV...")
        return pd.read_csv(io.StringIO(content.decode('utf-8')))
    elif file_extension in ['.xlsx', '.xls']:
        st.info(f"Reading {uploaded_file.name} as Excel...")
        return pd.read_excel(io.BytesIO(content))
    else:
        st.warning(f"Skipping unsupported file type: {uploaded_file.name}")
        return None

col1, col2 = st.columns([2, 1])

with col1:
    uploaded_files = st.file_uploader("Upload CSV / Excel files (multi allowed)", type=["csv", "xlsx", "xls"], accept_multiple_files=True)
    if uploaded_files:
        if st.button("Predict Uploaded Files"):
            if pipeline is None:
                st.error("No pipeline available. Train and save pipeline to models/final_pipeline.joblib.")
            else:
                for uploaded_file in uploaded_files:
                    st.markdown(f"**File:** {uploaded_file.name}")
                    try:
                        suffix = Path(uploaded_file.name).suffix.lower()
                        if suffix in [".xlsx", ".xls"]:
                            df = pd.read_excel(uploaded_file)
                        else:
                            df = pd.read_csv(uploaded_file)
                    except Exception as e:
                        st.error(f"Failed to read {uploaded_file.name}: {e}")
                        continue

                    # Validate and order columns using saved feature metadata, if available.
                    if numerical_cols is not None:
                        missing = set(numerical_cols) - set(df.columns)
                        if missing:
                            st.error(f"Uploaded file missing required numerical columns: {sorted(list(missing))}")
                            # skip this file and continue with others
                            continue
                        # ensure correct column order expected by the preprocessor
                        # include remainder (passthrough) columns if present in file
                        remainder_present = [c for c in (remainder_cols or []) if c in df.columns]
                        ordered_cols = numerical_cols + remainder_present
                        features_df = df[ordered_cols].reset_index(drop=True)
                    else:
                        # fall back to previous behavior (drop passthrough features if present)
                        features_df = df.drop(columns=params['data_processing']['passthrough_features'], errors='ignore').reset_index(drop=True)

                    try:
                        preds = pipeline.predict(features_df)
                        probs = pipeline.predict_proba(features_df) if hasattr(pipeline, "predict_proba") else None
                        if label_encoder is not None:
                            preds_decoded = label_encoder.inverse_transform(preds)
                            class_names = list(label_encoder.classes_)
                        else:
                            preds_decoded = preds
                            class_names = list(pipeline.classes_) if hasattr(pipeline, "classes_") else []
                        out_df = df.copy().reset_index(drop=True)
                        out_df["Predicted"] = preds_decoded
                        if probs is not None:
                            probs_df = pd.DataFrame(probs, columns=class_names)
                            out_df = pd.concat([out_df, probs_df.reset_index(drop=True)], axis=1)
                            out_df["Confidence"] = probs.max(axis=1)
                        st.dataframe(out_df.head(20))
                        csv = out_df.to_csv(index=False).encode("utf-8")
                        st.download_button(f"Download predictions for {uploaded_file.name}", data=csv, file_name=f"predictions_{uploaded_file.name}.csv")
                        st.success("Prediction completed.")
                    except Exception as e:
                        st.error(f"Prediction failed for {uploaded_file.name}: {e}")
                        # diagnostics
                        if hasattr(pipeline, "feature_names_in_"):
                            expected = set(pipeline.feature_names_in_)
                            provided = set(features_df.columns)
                            missing = expected - provided
                            if missing:
                                st.info(f"Missing input fields expected by model: {sorted(list(missing))}")
                        else:
                            st.info("Ensure uploaded file columns match training raw feature names.")

with col2:
    st.subheader("Single sample input")
    st.markdown("Enter one sample manually. Use key=value per line or a single CSV line.")
    manual_mode = st.radio("Input mode:", ["Key=Value lines", "Single-line CSV"])
    sample_input = st.text_area("Paste sample here", height=200, placeholder="e.g.\nSiO2=45\nAl2O3=15\nTiO2=0.5\n...")

    if st.button("Predict single sample"):
        if pipeline is None:
            st.error("No pipeline available. Train and save pipeline first.")
        else:
            sample_df = None
            try:
                if manual_mode == "Key=Value lines":
                    lines = [L.strip() for L in sample_input.splitlines() if L.strip()]
                    d = {}
                    for L in lines:
                        if "=" in L:
                            k, v = L.split("=", 1)
                            d[k.strip()] = float(v.strip())
                        else:
                            raise ValueError("Invalid line format. Expect key=value per line.")
                    sample_df = pd.DataFrame([d])
                else:
                    txt = sample_input.strip()
                    if "," in txt:
                        try:
                            sample_df = pd.read_csv(io.StringIO(txt))
                        except Exception:
                            vals = [v.strip() for v in txt.split(",")]
                            expected = None
                            if hasattr(pipeline, "feature_names_in_"):
                                expected = list(pipeline.feature_names_in_)
                            if expected and len(vals) == len(expected):
                                sample_df = pd.DataFrame([dict(zip(expected, map(float, vals)))])
                            else:
                                raise ValueError("CSV line does not match expected columns. Provide header or key=value lines.")
                    else:
                        raise ValueError("CSV mode expects comma-separated values.")
            except Exception as e:
                st.error(f"Could not parse sample input: {e}")
                st.info("Examples:\nSiO2=45\nAl2O3=15\n...  OR  SiO2,Al2O3,TiO2\n45,15,0.5")
                sample_df = None

            if sample_df is not None:
                try:
                    # Ensure sample_df column order if metadata present
                    if numerical_cols is not None:
                        missing = set(numerical_cols) - set(sample_df.columns)
                        if missing:
                            st.error(f"Sample missing required numerical columns: {sorted(list(missing))}")
                            raise ValueError("Missing required numerical columns.")
                        remainder_present = [c for c in (remainder_cols or []) if c in sample_df.columns]
                        ordered_cols = numerical_cols + remainder_present
                        sample_df = sample_df[ordered_cols].reset_index(drop=True)

                    preds = pipeline.predict(sample_df)
                    probs = pipeline.predict_proba(sample_df) if hasattr(pipeline, "predict_proba") else None
                    if label_encoder is not None:
                        pred_label = label_encoder.inverse_transform(preds)[0]
                        class_names = list(label_encoder.classes_)
                    else:
                        pred_label = preds[0]
                        class_names = list(pipeline.classes_) if hasattr(pipeline, "classes_") else []
                    st.write("**Predicted class:**", pred_label)
                    if probs is not None:
                        prob_series = pd.Series(probs[0], index=class_names)
                        st.table(prob_series.sort_values(ascending=False).to_frame("Probability"))
                        st.info(f"Confidence (max probability): {prob_series.max():.3f}")
                    else:
                        st.info("Model did not provide probability estimates.")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    st.warning("Ensure your input fields match training raw feature names and types.")

st.markdown("---")
# show expected columns if available
expected_cols = None
if hasattr(pipeline, "feature_names_in_"):
    expected_cols = list(pipeline.feature_names_in_)
if expected_cols:
    st.write("**Expected raw input columns (example):**")
    st.write(", ".join(expected_cols))
else:
    st.write("Could not infer expected raw input column names automatically. Ensure your input includes the same raw oxide/REE column names used during training.")

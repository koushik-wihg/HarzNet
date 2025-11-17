# streamlit_app.py (final — robust DF-preserving transforms + fallback unpickler)
import numpy as np
import pandas as pd
import math
import io
import os
import pickle
import joblib
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline as SKPipeline

# ---------------------------
# Custom transformers (for unpickle)
# ---------------------------
class HFSE_REE_Ratios(BaseEstimator, TransformerMixin):
    def __init__(self, candidates=None):
        self.candidates = candidates or [
            ('Nb','Y'), ('Zr','Y'), ('Th','Yb'),
            ('Ce','Yb'), ('La','Ce'), ('Nb','La')
        ]
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        Xdf = X.copy()
        for num, den in self.candidates:
            col_num = next((c for c in Xdf.columns if c.lower()==num.lower()), None)
            col_den = next((c for c in Xdf.columns if c.lower()==den.lower()), None)
            if col_num and col_den:
                newname = f"{num}_{den}"
                v1 = pd.to_numeric(Xdf[col_num], errors="coerce")
                v2 = pd.to_numeric(Xdf[col_den], errors="coerce").replace({0:np.nan})
                Xdf[newname] = (v1 / v2).fillna(0.0)
        return Xdf

class PivotILRTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, comp_cols=(), zero_replace_factor=1e-6):
        self.comp_cols = tuple(comp_cols)
        self.zero_replace_factor = zero_replace_factor
    def fit(self, X, y=None):
        self.input_columns_ = list(X.columns)
        self.comp_idx_ = [self.input_columns_.index(c) for c in self.comp_cols]
        comp_vals = X.iloc[:, self.comp_idx_].to_numpy(dtype=float)
        pos = comp_vals[(~np.isnan(comp_vals)) & (comp_vals > 0)]
        self.eps_ = (pos.min() * self.zero_replace_factor) if pos.size > 0 else self.zero_replace_factor
        self.noncomp_cols_ = [c for i, c in enumerate(self.input_columns_) if i not in self.comp_idx_]
        return self
    def _close(self, A):
        s = A.sum(axis=1, keepdims=True)
        s[s==0] = 1.0
        return A / s
    def transform(self, X):
        if isinstance(X, np.ndarray):
            Xdf = pd.DataFrame(X, columns=self.input_columns_)
        else:
            Xdf = X.copy()
        comps = Xdf.loc[:, list(self.comp_cols)].to_numpy(dtype=float)
        comps = np.where(np.isnan(comps), self.eps_, comps)
        comps = np.where(comps <= 0, self.eps_, comps)
        Xc = self._close(comps)
        n, k = Xc.shape
        ilr = np.zeros((n, k-1))
        for j in range(k-1):
            gm = np.exp(np.mean(np.log(Xc[:, j+1:]), axis=1))
            scale = math.sqrt((k - j - 1) / (k - j))
            ilr[:, j] = scale * np.log(Xc[:, j] / gm)
        ilr_cols = [f"ilr_{c}_vs_rest" for c in self.comp_cols[:-1]]
        ilr_df = pd.DataFrame(ilr, columns=ilr_cols, index=Xdf.index)
        if len(self.noncomp_cols_) > 0:
            return pd.concat([ilr_df, Xdf[self.noncomp_cols_]], axis=1)
        return ilr_df

# ---------------------------
# Config
# ---------------------------
MODEL_PATH = "best_pipeline_XGB_SMOTE_OUT_Fixed.joblib"

# canonical fallback list
MAJORS = ["SiO2","TiO2","Al2O3","Fe2O3","FeO","MnO","MgO","CaO","Na2O","K2O","P2O5","SO3","LOI"]
REE_LIST = ["La","Ce","Pr","Nd","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu","Y","Sc","Nb","Zr","Th"]
CANONICAL_ALL = MAJORS + REE_LIST

# ---------------------------
# Robust loader: try joblib first, then fallback unpickler mapping missing modules to local classes
# ---------------------------
def load_pipeline_robust(path):
    # try normal joblib.load first
    try:
        return joblib.load(path)
    except Exception as primary_exc:
        # fallback: read bytes and unpickle with custom Unpickler that resolves missing modules to local names
        # --- DEBUG VERSION: show both exceptions clearly in Streamlit UI and logs ---
        try:
            with open(path, "rb") as f:
                data = f.read()
            bio = io.BytesIO(data)
            class FallbackUnpickler(pickle.Unpickler):
                def find_class(self, module, name):
                    try:
                        return super().find_class(module, name)
                    except Exception:
                        g = globals()
                        if name in g:
                            return g[name]
                        try:
                            mod = __import__(module, fromlist=[name])
                            return getattr(mod, name)
                        except Exception:
                            raise

            bio.seek(0)
            unp = FallbackUnpickler(bio)
            obj = unp.load()
            return obj
        except Exception as fallback_exc:
            # Print full debug info to Streamlit UI and stderr (Cloud logs)
            import traceback, sys
            st.error("=== MODEL LOAD DEBUG INFO ===")
            st.write(f"MODEL_PATH = {path}")
            st.write("Working directory listing:")
            try:
                st.write(os.listdir("."))
            except Exception:
                st.write("Could not list directory.")
            st.write("---- Primary joblib.load() exception (type + message) ----")
            st.write(type(primary_exc).__name__ + ": " + str(primary_exc))
            st.text("".join(traceback.format_exception(type(primary_exc), primary_exc, primary_exc.__traceback__)))
            st.write("---- Fallback unpickler exception (type + message) ----")
            st.write(type(fallback_exc).__name__ + ": " + str(fallback_exc))
            st.text("".join(traceback.format_exception(type(fallback_exc), fallback_exc, fallback_exc.__traceback__)))
            # Also write to stderr so it appears in Cloud logs
            print("=== MODEL LOAD DEBUG INFO ===", file=sys.stderr)
            print(f"MODEL_PATH = {path}", file=sys.stderr)
            traceback.print_exception(type(primary_exc), primary_exc, primary_exc.__traceback__, file=sys.stderr)
            traceback.print_exception(type(fallback_exc), fallback_exc, fallback_exc.__traceback__, file=sys.stderr)
            # stop app so you can inspect the messages
            st.stop()
            # unreachable, but keep same signature
            raise RuntimeError("Model load failed (see debug output).")

# ---------------------------
# Load pipeline (robust)
# ---------------------------
best_pipeline = load_pipeline_robust(MODEL_PATH)
LABEL_MAP_INV = {0: "MOR", 1: "OIB", 2: "SSZ"}

# ---------------------------
# Discover expected features (names or count)
# ---------------------------
def discover_expected_features(pipe):
    try:
        names = list(pipe.feature_names_in_)
        return names, f"feature_names_in_ ({len(names)} features)"
    except Exception:
        pass
    try:
        steps = getattr(pipe, "steps", None)
        if steps:
            for n, step in steps:
                if hasattr(step, "feature_names_in_"):
                    names = list(step.feature_names_in_)
                    return names, f"{n}.feature_names_in_ ({len(names)} features)"
    except Exception:
        pass
    try:
        if hasattr(pipe, "n_features_in_"):
            n = int(pipe.n_features_in_)
            return CANONICAL_ALL[:n], f"n_features_in_ ({n} features)"
    except Exception:
        pass
    return CANONICAL_ALL, f"fallback canonical ({len(CANONICAL_ALL)} features)"

expected_features, src = discover_expected_features(best_pipeline)
expected_n = len(expected_features)

# ---------------------------
# tolerant matching helpers
# ---------------------------
def normalize_col(c):
    return ''.join(ch for ch in str(c).lower() if ch.isalnum())

def find_column_by_name(df_cols, target):
    tn = normalize_col(target)
    for c in df_cols:
        if normalize_col(c) == tn:
            return c
    return None

def build_input_df_from_uploaded(raw_df, expected_features):
    df_cols = list(raw_df.columns)
    selected = {}
    missing = []
    for req in expected_features:
        found = find_column_by_name(df_cols, req)
        if found:
            selected[req] = pd.to_numeric(raw_df[found], errors="coerce").fillna(0.0)
        else:
            selected[req] = pd.Series([0.0]*len(raw_df), index=raw_df.index)
            missing.append(req)
    return pd.DataFrame(selected), missing

# ---------------------------
# Utility: run transformer steps sequentially and keep DataFrames
# ---------------------------
def run_transformer_steps_keep_df(pipeline, X_df):
    """
    Run all pipeline steps except final estimator sequentially.
    After each step, if output is numpy, convert back to DataFrame with best-effort column names.
    Returns transformed object (DataFrame or array) and column names where possible.
    """
    # get steps list; handle sklearn Pipeline and other wrappers
    steps = getattr(pipeline, "steps", None)
    if steps is None:
        # if pipeline is a simple estimator (no steps), just return X_df
        return X_df
    X_current = X_df
    for name, step in steps[:-1]:
        # apply transform
        try:
            X_next = step.transform(X_current)
        except Exception:
            # sometimes transformers require fit_transform
            X_next = step.fit_transform(X_current)
        # if numpy, try to recover column names
        if isinstance(X_next, np.ndarray):
            cols_out = None
            # try get_feature_names_out with input features
            try:
                if hasattr(step, "get_feature_names_out"):
                    try:
                        cols_out = list(step.get_feature_names_out(getattr(X_current, "columns", None)))
                    except Exception:
                        cols_out = list(step.get_feature_names_out())
            except Exception:
                cols_out = None
            if cols_out is None:
                # fallback: reuse previous columns if length matches; else generic names
                prev_cols = getattr(X_current, "columns", None)
                if prev_cols is not None and len(prev_cols) >= X_next.shape[1]:
                    cols_out = list(prev_cols)[:X_next.shape[1]]
                else:
                    cols_out = [f"f_{i}" for i in range(X_next.shape[1])]
            X_current = pd.DataFrame(X_next, columns=cols_out, index=getattr(X_current, "index", None))
        else:
            # assume dataframe-like
            X_current = X_next
    return X_current

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Ultramafic Classifier", layout="wide")
st.title("🧭 Ultramafic Rock Tectonic Setting Classifier (MOR / OIB / SSZ)")
st.markdown("**Note:** Input samples must be peridotite. This app works best for harzburgites.")
st.write(f"Model expects {expected_n} input features (discovered from pipeline: {src}).")
st.write("---")

# ---------- single ----------
st.header("Single-sample prediction")
input_data = {}
cols_ui = st.columns(4)
for i, ox in enumerate(CANONICAL_ALL[:13]):
    with cols_ui[i % 4]:
        input_data[ox] = st.number_input(f"{ox}", 0.0, 100.0, 0.0, format="%.4f")
st.subheader("Trace elements (ppm)")
cols2 = st.columns(5)
for i, tr in enumerate(CANONICAL_ALL[13:]):
    with cols2[i % 5]:
        input_data[tr] = st.number_input(f"{tr}", 0.0, 1e6, 0.0, format="%.4f")

single_raw = pd.DataFrame([input_data])
single_req_df, missing_single = build_input_df_from_uploaded(single_raw, expected_features[:expected_n])
if missing_single:
    st.warning(f"Missing fields filled with zeros: {missing_single}")

if st.button("Predict Single Sample"):
    try:
        # run transformer sub-pipeline safely (keep DataFrames)
        Xt = run_transformer_steps_keep_df(best_pipeline, single_req_df)
        # convert to numpy only for final estimator
        Xt_np = Xt if isinstance(Xt, np.ndarray) else getattr(Xt, "to_numpy", lambda: np.asarray(Xt))()
        # extract final estimator
        final_estimator = best_pipeline.steps[-1][1] if hasattr(best_pipeline, "steps") else best_pipeline[-1]
        pred = final_estimator.predict(Xt_np)[0]
        prob = final_estimator.predict_proba(Xt_np)[0]

        st.success(f"Predicted class: **{ {0:'MOR',1:'OIB',2:'SSZ'}.get(pred,'?') }**")
        st.dataframe(pd.DataFrame({"Class":["MOR","OIB","SSZ"], "Prob": np.round(prob,4)}))

        # ternary
        coords = np.array([[0.0,0.0],[1.0,0.0],[0.5, math.sqrt(3)/2]])
        point = prob[0]*coords[0] + prob[1]*coords[1] + prob[2]*coords[2]
        fig, ax = plt.subplots(figsize=(4,4))
        xs = [coords[0,0], coords[1,0], coords[2,0], coords[0,0]]
        ys = [coords[0,1], coords[1,1], coords[2,1], coords[0,1]]
        ax.plot(xs, ys, '-k'); ax.scatter(point[0], point[1], s=80, c='C1')
        ax.text(coords[0,0]-0.03, coords[0,1]-0.03, "MOR")
        ax.text(coords[1,0]+0.02, coords[1,1]-0.03, "OIB")
        ax.text(coords[2,0], coords[2,1]+0.02, "SSZ")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(-0.1,1.1); ax.set_ylim(-0.1,1.0)
        st.pyplot(fig)

    except Exception as e:
        st.error("Error during single-sample prediction:")
        st.error(str(e))

st.write("---")

# ---------- batch ----------
st.header("Batch prediction (CSV / Excel)")
uploaded_file = st.file_uploader("Upload CSV or Excel (majors+REE columns will be used)", type=["csv","xlsx"])

if uploaded_file:
    try:
        if uploaded_file.name.lower().endswith(".csv"):
            df_raw = pd.read_csv(uploaded_file)
        else:
            df_raw = pd.read_excel(uploaded_file)
        st.write("Preview uploaded file:")
        st.dataframe(df_raw.head())

        df_req, missing = build_input_df_from_uploaded(df_raw, expected_features[:expected_n])
        if missing:
            st.warning(f"Missing required columns filled with zeros: {missing}")

        # run transformer steps safely
        Xt = run_transformer_steps_keep_df(best_pipeline, df_req)
        Xt_np = Xt if isinstance(Xt, np.ndarray) else getattr(Xt, "to_numpy", lambda: np.asarray(Xt))()

        final_estimator = best_pipeline.steps[-1][1] if hasattr(best_pipeline, "steps") else best_pipeline[-1]
        preds = final_estimator.predict(Xt_np)
        probs = final_estimator.predict_proba(Xt_np)

        df_out = df_req.copy()
        df_out["Predicted_Class"] = [ {0:'MOR',1:'OIB',2:'SSZ'}.get(p,'?') for p in preds ]
        df_out["Prob_MOR"] = probs[:,0]; df_out["Prob_OIB"] = probs[:,1]; df_out["Prob_SSZ"] = probs[:,2]
        st.success("Batch prediction complete.")
        st.dataframe(df_out.head())

        # ternary
        coords = np.array([[0.0,0.0],[1.0,0.0],[0.5, math.sqrt(3)/2]])
        points = probs @ coords
        fig2, ax2 = plt.subplots(figsize=(6,5))
        xs = [coords[0,0], coords[1,0], coords[2,0], coords[0,0]]
        ys = [coords[0,1], coords[1,1], coords[2,1], coords[0,1]]
        ax2.plot(xs, ys, '-k', linewidth=1)
        colors = np.array(['C0','C1','C2'])[preds]
        ax2.scatter(points[:,0], points[:,1], s=30, c=colors, alpha=0.8)
        ax2.text(coords[0,0]-0.03, coords[0,1]-0.03, "MOR")
        ax2.text(coords[1,0]+0.02, coords[1,1]-0.03, "OIB")
        ax2.text(coords[2,0], coords[2,1]+0.02, "SSZ")
        ax2.set_xticks([]); ax2.set_yticks([]); ax2.set_xlim(-0.1,1.1); ax2.set_ylim(-0.1,1.0)
        st.pyplot(fig2)

        # downloads
        csv_bytes = df_out.to_csv(index=False).encode()
        st.download_button("⬇️ Download CSV", csv_bytes, "ultramafic_predictions.csv", "text/csv")

        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            df_out.to_excel(writer, index=False, sheet_name="Predictions")
        st.download_button("⬇️ Download Excel (.xlsx)", excel_buffer.getvalue(),
                           "ultramafic_predictions.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    except Exception as e:
        st.error("Error processing uploaded file:")
        st.error(str(e))

st.write("---")
st.caption("Powered by your optimized XGBoost ILR-PCA pipeline — input samples must be peridotite; works best for harzburgites.")

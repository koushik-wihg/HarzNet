import os
import yaml
from pathlib import Path
import textwrap

# Set the project root path (Change this if your project path is different)
PROJECT_ROOT = Path(r"D:/Python_Projects/Ultramafic_MLOPS").resolve()

# --- Content for all files is defined here in clean strings ---

# 1. Utility Script (src/utils/common.py)
COMMON_SCRIPT_CONTENT = """
import yaml
from pathlib import Path

def read_params(config_path: Path) -> dict:
    \"\"\"Reads the YAML configuration file and returns its content as a dictionary.\"\"\"
    try:
        with open(config_path, 'r') as yaml_file:
            config = yaml.safe_load(yaml_file)
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        return {}
    except yaml.YAMLError as exc:
        print(f"Error parsing YAML file: {exc}")
        return {}
"""

# 2. Data Ingestion Script (src/data_ingestion.py)
DATA_INGESTION_CONTENT = """
import pandas as pd
from pathlib import Path
from src.utils.common import read_params

def main():
    CONFIG_PATH = Path("Config/params.yaml")
    params = read_params(CONFIG_PATH)
    source_path = params['data_ingestion']['source_path']
    output_file = params['data_ingestion']['output_file']

    # Mock Data Creation for demonstration purposes if the source CSV is missing
    if not Path(source_path).exists():
        print(f"Warning: Mocking data as source file not found at {source_path}.")
        data = pd.DataFrame({
            'Tectonic setting': ['MOR', 'OIB', 'SSZ', 'MOR', 'OIB', 'SSZ'],
            'SiO2': [45, 48, 51, 46, 49, 50],
            'MgO': [8, 7, 6, 7.5, 6.5, 5.5],
            'Al2O3': [15, 16, 17, 15.5, 16.5, 17.5],
            'Col_1': range(6), 'Sample': ['A1', 'A2', 'A3', 'A4', 'A5', 'A6']
        })
    else:
        try:
            data = pd.read_csv(source_path)
        except FileNotFoundError:
            print(f"Error: Source file not found at {source_path}. Please check config.")
            return

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)
    print(f"Data Ingestion complete: {output_path}")

if __name__ == "__main__":
    main()
"""

# 3. Data Processing Script (src/data_processing.py)
DATA_PROCESSING_CONTENT = """
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.utils.common import read_params
import joblib
import numpy as np

def clr_transform(X):
    # X assumed non-negative; add small offset for zeros
    X = np.asarray(X, dtype=float)
    offset = 1e-9
    X = X + offset
    logX = np.log(X)
    gm = np.mean(logX, axis=1, keepdims=True)
    clr = logX - gm
    return clr

def get_preprocessor(params, numerical_cols):
    impute_strategy = params['data_processing']['impute_strategy']
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy=impute_strategy)),
        ('clr', FunctionTransformer(clr_transform, validate=False)),
        ('scaler', StandardScaler())
    ])
    return ColumnTransformer([('num', numerical_pipeline, numerical_cols)], remainder='drop')

def main():
    CONFIG_PATH = Path("Config/params.yaml")
    params = read_params(CONFIG_PATH)
    target_col = params['base']['target_column']
    raw_data_path = Path(params['data_ingestion']['output_file'])
    passthrough = params['data_processing']['passthrough_features']
    pca_var = params['data_processing']['pca_variance_threshold']

    processed_data_path = Path("data/processed/data_processed.pkl")
    pipeline_path = Path("models/preprocessing_pipeline.joblib")

    try:
        data = pd.read_csv(raw_data_path)
    except FileNotFoundError:
        print("Error: Raw data not found. Run data_ingestion first.")
        return

    X = data.drop(columns=[target_col], errors='ignore')
    y = data[target_col]

    # Identify numerical columns (all except the passthrough feature 'Sample')
    num_cols = [c for c in X.columns if X[c].dtype in ['int64', 'float64'] and c not in passthrough]

    if not num_cols:
        print("Error: No numerical columns found for preprocessing/PCA.")
        return

    preprocessor = get_preprocessor(params, num_cols)

    # Fit and transform the data using the scaler pipeline (which includes CLR)
    X_trans = preprocessor.fit_transform(X)

    # Convert scaled array back to DataFrame for PCA step
    X_df = pd.DataFrame(X_trans, columns=num_cols)

    # Apply PCA on the transformed numerical data (capturing pca_var variance)
    pca = PCA(n_components=pca_var, random_state=params['data_ingestion']['random_state'])
    pca_df = pd.DataFrame(pca.fit_transform(X_df), columns=[f'PCA_{i+1}' for i in range(pca.n_components_)])

    # Concatenate PCA components, the passthrough column ('Sample'), and the target column
    final = pd.concat([pca_df, X[passthrough].reset_index(drop=True), y.reset_index(drop=True)], axis=1)

    final.to_pickle(processed_data_path)

    # Save the complete pipeline (preprocessor + PCA)
    joblib.dump(Pipeline([('pre', preprocessor), ('pca', pca)]), pipeline_path)
    print(f"Data processed and saved. PCA components found: {pca.n_components_}")

if __name__ == "__main__":
    main()
"""

# 4. Model Trainer (src/model_trainer.py)
MODEL_TRAINER_CONTENT = """
import joblib, optuna, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from src.utils.common import read_params
from sklearn.pipeline import Pipeline
import warnings; warnings.filterwarnings('ignore')

def main():
    CONFIG_PATH = Path("Config/params.yaml")
    params = read_params(CONFIG_PATH)
    target_col = params['base']['target_column']

    processed_data_path = Path("data/processed/data_processed.pkl")
    if not processed_data_path.exists():
        print(f"Error: Processed data not found at {processed_data_path}. Please run data_processing first.")
        return

    data = pd.read_pickle(processed_data_path)

    # Drop target and passthrough features (like 'Sample')
    X = data.drop(columns=[target_col] + params['data_processing']['passthrough_features'])
    y = data[target_col]

    # --- Label Encoding for XGBoost ---
    le = LabelEncoder()
    y = pd.Series(le.fit_transform(y), name=target_col)
    print(f"Target labels encoded: {le.classes_} -> {le.transform(le.classes_)}")
    # -------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=params['data_ingestion']['test_size'],
        random_state=params['data_ingestion']['random_state'],
        stratify=y
    )

    xgb_default = params['model_trainer']['xgb_default']

    def objective(trial):
        optuna_params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 800),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        }
        full_params = {**optuna_params, **xgb_default}

        m = XGBClassifier(use_label_encoder=False, **full_params)
        m.fit(X_train, y_train)
        return m.score(X_test, y_test)

    try:
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=params['model_trainer']['n_trials'])
    except Exception as e:
        print(f"Optuna failed during optimization: {e}")
        study = type('', (object,), {'best_params': {}})()
        study.best_params = params['model_trainer']['xgb_default']
        print("Falling back to default model parameters due to Optuna failure.")

    print("Best parameters found:", study.best_params)

    final_model_params = {**study.best_params, **xgb_default}
    model = XGBClassifier(use_label_encoder=False, **final_model_params)
    model.fit(X_train, y_train)

    Path("models").mkdir(exist_ok=True)
    preproc_path = Path("models/preprocessing_pipeline.joblib")
    preprocessor = None
    if preproc_path.exists():
        try:
            preprocessor = joblib.load(preproc_path)
            print(f"Loaded preprocessor from {preproc_path}")
        except Exception as e:
            print(f"Warning: could not load preprocessor: {e}")
            preprocessor = None
    else:
        print("Warning: preprocessing_pipeline.joblib not found. Ensure data_processing.py was run.")

    # Save combined pipeline (preprocessor -> classifier) for raw-input inference
    if preprocessor is not None:
        full_pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", model)])
    else:
        full_pipeline = Pipeline([("classifier", model)])
        print("Saved pipeline contains only the classifier because preprocessor was not found.")

    # Save both formats for compatibility
    joblib.dump({"pipeline": full_pipeline, "label_encoder": le}, "models/final_pipeline.joblib")
    joblib.dump({'model': model, 'label_encoder': le}, "models/best_model_tuned.joblib")
    print("Saved models/final_pipeline.joblib (pipeline + label_encoder) and models/best_model_tuned.joblib (compat).")

if __name__ == "__main__":
    main()
"""

# 5. Model Evaluation (src/model_evaluation.py)
MODEL_EVALUATION_CONTENT = """
import pandas as pd, joblib, json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns, matplotlib.pyplot as plt
from src.utils.common import read_params

def main():
    CONFIG_PATH = Path("Config/params.yaml")
    params = read_params(CONFIG_PATH)
    ycol = params['base']['target_column']
    df = pd.read_pickle("data/processed/data_processed.pkl")
    X = df.drop(columns=[ycol] + params['data_processing']['passthrough_features'])
    y = df[ycol]

    try:
        model_and_le = joblib.load("models/best_model_tuned.joblib")
        le = model_and_le['label_encoder']
    except Exception:
        print("Error: Could not load model or LabelEncoder. Ensure model_trainer ran successfully.")
        return

    y_encoded = le.transform(y)

    X_train, X_test, y_train_orig, y_test_orig = train_test_split(X, y_encoded, test_size=params['data_ingestion']['test_size'],
                                                                  random_state=params['data_ingestion']['random_state'], stratify=y_encoded)

    model = model_and_le['model']
    y_pred_encoded = model.predict(X_test)

    acc = accuracy_score(y_test_orig, y_pred_encoded)
    p, r, f1, _ = precision_recall_fscore_support(y_test_orig, y_pred_encoded, average='macro')
    res = dict(accuracy=acc, precision=p, recall=r, f1=f1)

    Path("metrics").mkdir(exist_ok=True)
    with open("metrics/final_metrics.json", "w") as f: json.dump(res, f, indent=2)

    # Confusion Matrix using original class names (le.classes_)
    cm = confusion_matrix(y_test_orig, y_pred_encoded)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title("Confusion Matrix")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig("metrics/confusion_matrix.png")
    plt.close()
    print("Evaluation complete.")

if __name__ == "__main__":
    main()
"""

# 6. FastAPI API (src/api.py)
API_CONTENT = """
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import pandas as pd, joblib, os, io
from pathlib import Path
from src.utils.common import read_params

app = FastAPI(title="Ultramafic ML API", version="0.1")
CONFIG_PATH = Path("Config/params.yaml")
params = read_params(CONFIG_PATH)
MODEL_PATH = params.get("api", {}).get("model_path", "models/final_pipeline.joblib")

def load_model_artifact(path):
    if not Path(path).exists():
        raise FileNotFoundError(f"Model artifact not found at {path}")
    obj = joblib.load(path)
    if isinstance(obj, dict):
        if 'pipeline' in obj:
            pipeline = obj['pipeline']
            le = obj.get('label_encoder', None)
            return pipeline, le
        elif 'model' in obj:
            model = obj['model']
            le = obj.get('label_encoder', None)
            return model, le
    if hasattr(obj, "predict"):
        return obj, None
    raise ValueError("Unrecognized model artifact format.")

# Load model and LabelEncoder once at startup
try:
    model, le = load_model_artifact(MODEL_PATH)
except Exception as e:
    model = None
    le = None
    print(f"FATAL: Could not load model artifact: {e}")

@app.get("/health")
def health():
    return {"status": "ok", "model_exists": bool(model)}

class PredictReq(BaseModel):
    records: list

@app.post("/predict")
def predict(payload: PredictReq):
    if not model:
        raise HTTPException(status_code=503, detail="Model is not available or trained yet.")

    df = pd.DataFrame(payload.records)
    try:
        preds_encoded = model.predict(df)
        preds_decoded = le.inverse_transform(preds_encoded).tolist() if le is not None else preds_encoded.tolist()
        probs = model.predict_proba(df).tolist() if hasattr(model, "predict_proba") else None
        return {"predictions": preds_decoded, "probabilities": probs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_csv")
def predict_csv(file: UploadFile = File(...)):
    if not model:
        raise HTTPException(status_code=503, detail="Model is not available or trained yet.")

    content = file.file.read()
    df = pd.read_csv(io.BytesIO(content))
    try:
        preds_encoded = model.predict(df)
        preds_decoded = le.inverse_transform(preds_encoded).tolist() if le is not None else preds_encoded.tolist()
        return {"n_rows": len(df), "predictions": preds_decoded}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
"""

# 7. Streamlit UI (src/ui_app.py)
STREAMLIT_UI_CONTENT = """
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

                    # Drop passthrough columns if present
                    features_df = df.drop(columns=params['data_processing']['passthrough_features'], errors='ignore')
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
                st.info("Examples:\nSiO2=45\nAl2O3=15\n...  OR  SiO2,Al2O3,TiO2\\n45,15,0.5")
                sample_df = None

            if sample_df is not None:
                try:
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
"""

# 8. Config params.yaml
PARAMS_YAML_CONTENT = """
base:
  target_column: "Tectonic setting"
data_ingestion:
  # NOTE: Update this path to your actual CSV file location.
  source_path: "D:/Python_Projects/AI-ML - Copy.csv"
  output_file: "data/raw/ingested.csv"
  test_size: 0.2
  random_state: 42
data_processing:
  passthrough_features:
    - "Sample"
  impute_strategy: "median"
  pca_variance_threshold: 0.95
model_trainer:
  n_trials: 20
  xgb_default:
    eval_metric: "mlogloss"
model_evaluation:
  min_importance_threshold: 0.01
  confusion_matrix_path: "reports/confusion_matrix.png"
  final_metrics_path: "metrics/final_metrics.json"
api:
  model_path: "models/final_pipeline.joblib"
ui:
  default_sample_csv: "data/raw/ingested.csv"
"""

# 9. requirements.txt
REQUIREMENTS_CONTENT = textwrap.dedent("""
fastapi
uvicorn[standard]
streamlit
scikit-learn
xgboost
pandas
numpy
optuna
joblib
shap
matplotlib
seaborn
pyyaml
openpyxl
python-multipart
""")

# 10. Dockerfile.streamlit
DOCKERFILE_STREAMLIT_CONTENT = """
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit","run","src/ui_app.py","--server.port","8501","--server.address","0.0.0.0","--server.maxUploadSize","200"]
"""

# 11. Dockerfile.fastapi
DOCKERFILE_FASTAPI_CONTENT = """
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn","src.api:app","--host","0.0.0.0","--port","8000"]
"""

# 12. docker-compose.yml
DOCKER_COMPOSE_CONTENT = """
version: '3.8'

services:
  api:
    build:
      context: .
      dockerfile: Dockerfile.fastapi
    container_name: ultramafic_mlops-api
    restart: unless-stopped
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models:ro         # optional: provide model artifacts to API if needed

  ui:
    build:
      context: .
      dockerfile: Dockerfile.streamlit
    container_name: ultramafic_mlops-ui
    restart: unless-stopped
    volumes:
      - ./models:/app/models:ro        # mount model artifacts (read-only)
      - ./src:/app/src:rw              # mount source so edits are live inside the container
      - ./data:/app/data:rw            # optional: sample data
      - ./logs/ui:/app/logs:rw         # optional: persistent logs
    ports:
      - "8501:8501"
    environment:
      - STREAMLIT_SERVER_PORT=8501
      - STREAMLIT_SERVER_HEADLESS=true
      - STREAMLIT_SERVER_ADDRESS=0.0.0.0
      - PYTHONUNBUFFERED=1
      - TZ=Asia/Kolkata
    healthcheck:
      test: ["CMD-SHELL", "curl -fsS http://localhost:8501/ || exit 1"]
      interval: 15s
      timeout: 5s
      retries: 5
"""

# 13. .dockerignore
DOCKERIGNORE_CONTENT = """
__pycache__
*.pyc
venv/
data/
*.ipynb
.git
"""

# 14. README.md
README_CONTENT = """
# Ultramafic_MLOPS

Project scaffold created by setup_full_project.py.
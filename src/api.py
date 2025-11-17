# --- FIX FOR RENDER: ensure project root is on sys.path ---
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # -> /app on Render
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# --- END FIX ---

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

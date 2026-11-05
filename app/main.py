from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from app.schemas import HeartFeatures, InfoResponse, PredictionResponse

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "heart_model.joblib"

app = FastAPI(
    title="Heart Disease Prediction API",
    description="Serves a scikit-learn classifier trained on the Heart Disease dataset.",
    version="1.0.0",
)

_bundle = None


def get_bundle() -> dict:
    """Load the model lazily and cache it."""
    global _bundle
    if _bundle is None:
        if not MODEL_PATH.exists():
            raise HTTPException(status_code=503, detail="Model file not found")
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


@app.on_event("startup")
def load_model_on_startup() -> None:
    try:
        get_bundle()
    except HTTPException:
        pass  # /health will report the problem


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}


@app.get("/info", response_model=InfoResponse)
def info() -> InfoResponse:
    bundle = get_bundle()
    return InfoResponse(
        model_type=type(bundle["model"]).__name__,
        n_features=len(bundle["features"]),
        features=bundle["features"],
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: HeartFeatures) -> PredictionResponse:
    bundle = get_bundle()
    row = pd.DataFrame([payload.model_dump()])[bundle["features"]]
    model = bundle["model"]
    prob = float(model.predict_proba(row)[0][1])
    return PredictionResponse(heart_disease=prob >= 0.5, probability=round(prob, 4))

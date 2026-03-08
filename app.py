import os
import time
from typing import List, Optional


import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from tensorflow import keras


# ✅ IMPORT YOUR STRONG CAT ENGINE
from strong_cat import StrongCATEngine




# =========================
# CONFIG
# =========================
MODEL_DIR = os.getenv("MODEL_DIR", "model")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.keras")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")


SEQ_LEN = int(os.getenv("SEQ_LEN", "120"))
N_FEATURES = int(os.getenv("N_FEATURES", "16"))
DEFAULT_THRESHOLD = float(os.getenv("THRESHOLD", "0.7223"))


# ✅ BACKEND TOGGLE
USE_CAT = os.getenv("USE_CAT", "true").lower() == "true"


app = FastAPI(title="SynapSee CNN-LSTM (Toggle CAT)")




# =========================
# STRONG CAT INITIALIZATION
# =========================
cat_engine = StrongCATEngine(
    n=5,
    k_on=3,
    k_off=1,
    prob_on=DEFAULT_THRESHOLD,
    prob_off=0.60,
    hold_seconds=2.0,
    cooldown_seconds=1.0,
)


# =========================
# Pydantic Schemas
# =========================
class InferWindowRequest(BaseModel):
    session_id: str = Field(..., examples=["P01_S01"])
    participant_id: Optional[str] = None
    timestamp_end_ms: Optional[int] = None
    window: List[List[float]]

class InferWindowResponse(BaseModel):
    session_id: str
    prob_cheat: float
    pred_raw: int
    cat_active: int
    cat_transition: str
    model_latency_ms: float
    total_latency_ms: float

# =========================
# Load model & scaler
# =========================
model = None
scaler = None




@app.on_event("startup")
def load_assets():
    global model, scaler


    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model not found: {MODEL_PATH}")
    if not os.path.exists(SCALER_PATH):
        raise RuntimeError(f"Scaler not found: {SCALER_PATH}")


    model = keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)


    dummy = np.zeros((1, SEQ_LEN, N_FEATURES), dtype=np.float32)
    _ = model.predict(dummy, verbose=0)


    print("Loaded model:", MODEL_PATH)
    print("Loaded scaler:", SCALER_PATH)
    print("USE_CAT:", USE_CAT)
    print("Inference server started")




@app.get("/health")
def health():
    return {
        "status": "ok",
        "seq_len": SEQ_LEN,
        "n_features": N_FEATURES,
        "threshold": DEFAULT_THRESHOLD,
        "use_cat": USE_CAT
    }




@app.post("/infer_window", response_model=InferWindowResponse)
def infer_window(req: InferWindowRequest):
    t0 = time.perf_counter()


    # =========================
    # Validate shape
    # =========================
    if len(req.window) != SEQ_LEN:
        raise HTTPException(
            status_code=400,
            detail=f"window must have {SEQ_LEN} rows (got {len(req.window)})"
        )


    if any(len(row) != N_FEATURES for row in req.window):
        bad = [i for i, row in enumerate(req.window) if len(row) != N_FEATURES][:5]
        raise HTTPException(
            status_code=400,
            detail=f"each row must have {N_FEATURES} features; bad rows: {bad}"
        )


    # =========================
    # Convert to numpy
    # =========================
    X = np.array(req.window, dtype=np.float32)


    # =========================
    # Scale
    # =========================
    X2 = X.reshape(-1, N_FEATURES)
    X2s = scaler.transform(X2)
    Xs = X2s.reshape(1, SEQ_LEN, N_FEATURES)


    # =========================
    # Predict
    # =========================
    t1 = time.perf_counter()
    prob = float(model.predict(Xs, verbose=0)[0, 0])
    t2 = time.perf_counter()


    prob = float(np.clip(prob, 0.0, 1.0))
    pred_raw = 1 if prob >= DEFAULT_THRESHOLD else 0


    # =========================
    # Decision Layer
    # =========================
    if USE_CAT:
        cat_out = cat_engine.update(
            session_id=req.session_id,
            prob_cheat=prob,
            risk=None
        )


        cat_active = cat_out["cat_active"]
        cat_transition = cat_out["transition"]


    else:
        # CAT disabled → behave like fixed threshold
        cat_active = pred_raw
        cat_transition = "disabled"


    model_latency_ms = (t2 - t1) * 1000.0
    total_latency_ms = (t2 - t0) * 1000.0


    return InferWindowResponse(
        session_id=req.session_id,
        prob_cheat=prob,
        pred_raw=pred_raw,
        cat_active=cat_active,
        cat_transition=cat_transition,
        model_latency_ms=model_latency_ms,
        total_latency_ms=total_latency_ms
    )

import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import io


# App setup
app = FastAPI(title="UFC Fight Predictor API")


# Middleware - temp perms for now
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Loading the model
with open("./ufc_predictor_v1.pkl", "rb") as f:
    model = pickle.load(f)

# Model req columns
FEATURE_COLS = [
    "title_fight", "total_rounds", "height_diff", "weight_diff", "reach_diff",
    "age_diff", "r_stance_Orthodox", "r_stance_Southpaw", "r_stance_Switch",
    "b_stance_Orthodox", "b_stance_Southpaw", "b_stance_Switch",
    "division_bantamweight", "division_catch weight", "division_featherweight",
    "division_flyweight", "division_heavyweight", "division_lightweight",
    "division_middleweight", "division_strawweight", "division_welterweight",
    "days_since_fight", "weighted_splm_diff", "weighted_str_acc_diff",
    "weighted_sapm_diff", "weighted_str_def_diff", "weighted_td_avg_diff",
    "weighted_td_avg_acc_diff", "weighted_td_def_diff", "weighted_sub_avg_diff"
]

# ID cols
ID_COLS = ["fight", "r_name", "b_name"]

# Health ping for gcp
@app.get("/health")
def health():
    return{"status": "ok"}


# Predcition Endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))

    # Validate columns exists
    required = ID_COLS + FEATURE_COLS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing columns: {missing}")
    
    # Keep for output
    meta = df[ID_COLS].copy()

    features = df[FEATURE_COLS.copy()]

    # Run the model
    p_red = model.predict_proba(features)[:, 1]

    # Predict wiiner and confidence
    predicted_winner = np.where(p_red >= 0.5, "RED", "BLUE")
    win_prob = np.where(p_red >= 0.5, p_red, 1 -p_red)
    confidence_pct = (win_prob * 100).round()

    # Results
    results = meta.copy()
    results["predict_winner"] = predicted_winner
    results["win_probability"] = win_prob
    results["confidence_pct"] = confidence_pct

    return results.to_dict(orient="records")


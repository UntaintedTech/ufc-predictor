from contextlib import asynccontextmanager
import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete
import io

from database import engine, get_db, Base
from models import Prediction

@asynccontextmanager
async def lifespan(app):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield


# App setup
app = FastAPI(title="UFC Fight Predictor API", lifespan=lifespan)

# Middleware - temp perms for now
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

with open("ufc_predictor_v1.pkl", "rb") as f:
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

ID_COLS = ["fight", "r_name", "b_name"]

# Health ping for gcp
@app.get("/health")
def health():
    return {"status": "ok"}


# Predcition and save db
@app.post("/predict")
async def predict(file: UploadFile = File(...), db: AsyncSession = Depends(get_db)):

    # Read and parse CSV
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))

    # Validate columns
    required = ID_COLS + FEATURE_COLS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing columns: {missing}")

    # Extract identities and features
    meta = df[ID_COLS].copy()
    features = df[FEATURE_COLS].copy()

    # Run model
    p_red = model.predict_proba(features)[:, 1]
    predicted_winner = np.where(p_red >= 0.5, "RED", "BLUE")
    win_prob = np.where(p_red >= 0.5, p_red, 1 - p_red)
    confidence_pct = (win_prob * 100).round(2)

    # 📌 Clear old predictions before saving new ones
    await db.execute(delete(Prediction))

    # 📌 Save each fight prediction as a row in the database
    predictions = []
    for i in range(len(meta)):
        prediction = Prediction(
            fight=meta.iloc[i]["fight"],
            r_name=meta.iloc[i]["r_name"],
            b_name=meta.iloc[i]["b_name"],
            predicted_winner=str(predicted_winner[i]),
            win_probability=float(win_prob[i]),
            confidence_pct=float(confidence_pct[i])
        )
        db.add(prediction)
        predictions.append(prediction)

    # 📌 Commit = permanently save all changes to the database
    await db.commit()

    return [
        {
            "fight": p.fight,
            "r_name": p.r_name,
            "b_name": p.b_name,
            "predicted_winner": p.predicted_winner,
            "win_probability": p.win_probability,
            "confidence_pct": p.confidence_pct
        }
        for p in predictions
    ]

@app.get("/predictions")
async def get_predictions(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Prediction))
    predictions = result.scalars().all()

    return [
        {
            "fight": p.fight,
            "r_name": p.r_name,
            "b_name": p.b_name,
            "predicted_winner": p.predicted_winner,
            "win_probability": p.win_probability,
            "confidence_pct": p.confidence_pct,
            "created_at": p.created_at
        }
        for p in predictions
    ]
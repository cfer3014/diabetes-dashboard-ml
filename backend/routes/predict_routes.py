from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
import numpy as np
import joblib

from backend.database import SessionLocal
from backend.models import Prediction
from backend.dependencies import get_current_user

router = APIRouter()

# ---------------- DB ----------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ---------------- LOAD MODEL ----------------
model = joblib.load("ml/artifacts/model.pkl")
scaler = joblib.load("ml/artifacts/scaler.pkl")

# ---------------- PREDICT ----------------
@router.post("/predict")
def predict(data: dict,
            db: Session = Depends(get_db),
            user: str = Depends(get_current_user)):

    values = np.array([[
        data["preg"],
        data["glucose"],
        data["bp"],
        data["skin"],
        data["ins"],
        data["bmi"],
        data["dpf"],
        data["age"]
    ]])

    scaled = scaler.transform(values)

    prob = model.predict_proba(scaled)[0][1]
    risk = "Alto" if prob > 0.5 else "Bajo"

    # guardar en DB
    prediction = Prediction(
        user_id=1,  # simplificado (luego se mejora con user_id real)
        glucose=data["glucose"],
        bmi=data["bmi"],
        age=data["age"],
        probability=prob,
        risk=risk
    )

    db.add(prediction)
    db.commit()

    return {
        "probability": prob,
        "risk": risk
    }
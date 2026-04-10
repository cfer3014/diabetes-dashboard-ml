from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

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

# ---------------- HISTORY ----------------
@router.get("/history")
def get_history(
    db: Session = Depends(get_db),
    user: str = Depends(get_current_user)
):

    history = db.query(Prediction).all()

    return history
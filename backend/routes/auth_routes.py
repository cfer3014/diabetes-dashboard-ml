from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.database import SessionLocal
from backend.models import User
from backend.auth import hash_password, verify_password, create_access_token

router = APIRouter()

# ---------------- DB DEP ----------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ---------------- REGISTER ----------------
@router.post("/register")
def register(username: str, password: str, db: Session = Depends(get_db)):

    user_exists = db.query(User).filter(User.username == username).first()
    if user_exists:
        raise HTTPException(status_code=400, detail="User already exists")

    user = User(
        username=username,
        password=hash_password(password)
    )

    db.add(user)
    db.commit()

    return {"message": "User created successfully"}

# ---------------- LOGIN ----------------
@router.post("/login")
def login(username: str, password: str, db: Session = Depends(get_db)):

    user = db.query(User).filter(User.username == username).first()

    if not user or not verify_password(password, user.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token({"sub": username})

    return {"access_token": token, "token_type": "bearer"}
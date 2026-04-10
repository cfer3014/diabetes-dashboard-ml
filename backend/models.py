from sqlalchemy import Column, Integer, Float, String, ForeignKey
from backend.database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    password = Column(String)


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))

    glucose = Column(Float)
    bmi = Column(Float)
    age = Column(Integer)

    probability = Column(Float)
    risk = Column(String)
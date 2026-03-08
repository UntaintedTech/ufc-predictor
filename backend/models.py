from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.sql import func
from database import Base

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    fight = Column(String, nullable=False)
    r_name = Column(String, nullable=False)
    b_name = Column(String, nullable=False)
    predicted_winner = Column(String, nullable=False)
    win_probability = Column(Float, nullable=False)
    confidence_pct = Column(Float, nullable=False)
    created_at = Column(DateTime, server_default=func.now())
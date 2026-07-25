import datetime
from typing import Optional

from pydantic import BaseModel


class PredictionResponse(BaseModel):
    id: int
    filename: str
    predicted_class: str
    confidence: float
    gradcam_url: Optional[str] = None
    llm_report: Optional[str] = None
    created_at: datetime.datetime

    class Config:
        from_attributes = True


class HistoryItem(BaseModel):
    id: int
    filename: str
    predicted_class: str
    confidence: float
    created_at: datetime.datetime

    class Config:
        from_attributes = True

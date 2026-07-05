from __future__ import annotations

from pydantic import BaseModel


class PredictionRequest(BaseModel):
    class Config:
        extra = "allow"


class PredictionResponse(BaseModel):
    predicted_grade: str
    predicted_grade_code: str
    predicted_label: str
    probabilities: dict[str, float]
    probability_codes: dict[str, float]

"""
Pydantic schemas for the single-model clinical assessment endpoint.

The ClinicalAssessment and AssessmentResponse models are preserved exactly
from Phase 0 to maintain backward API compatibility.
"""

from typing import Literal

from pydantic import BaseModel, Field


class ClinicalAssessment(BaseModel):
    """13-feature input matching the UCI processed Cleveland dataset."""

    age: int = Field(ge=1, le=120)
    sex: Literal[0, 1]
    cp: Literal[1, 2, 3, 4]
    trestbps: int = Field(ge=50, le=250)
    chol: int = Field(ge=0, le=700)
    fbs: Literal[0, 1]
    restecg: Literal[0, 1, 2]
    thalach: int = Field(ge=50, le=250)
    exang: Literal[0, 1]
    oldpeak: float = Field(ge=0.0, le=10.0)
    slope: Literal[1, 2, 3]
    ca: Literal[0, 1, 2, 3]
    thal: Literal[3, 6, 7]


class AssessmentResponse(BaseModel):
    """Single-model assessment response (backward-compatible)."""

    success: bool
    message: str
    prediction: int
    prediction_label: str
    class_1_probability: float | None
    model_connected: bool
    model_name: str
    feature_count: int
    processed_feature_count: int
    disclaimer: str

"""
Pydantic schemas for the multi-model comparison endpoint.
"""

from typing import Literal

from pydantic import BaseModel


class ModelResultSchema(BaseModel):
    """Per-model result returned by the comparison endpoint."""

    model_id: str
    model_name: str
    prediction: int | None
    prediction_label: str | None
    class_1_probability: float | None
    status: Literal["available", "unavailable", "error"]
    duration_ms: float | None
    message: str | None


class ComparisonResponse(BaseModel):
    """Full multi-model comparison response."""

    success: bool
    primary_model_id: str
    primary_prediction: int | None
    primary_class_1_probability: float | None
    results: list[ModelResultSchema]
    disclaimer: str
    available_model_count: int
    total_registered_model_count: int

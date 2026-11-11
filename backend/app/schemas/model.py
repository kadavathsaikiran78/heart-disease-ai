"""
Pydantic schemas for the model discovery and metadata endpoints.
"""

from pydantic import BaseModel


class ModelInfoSchema(BaseModel):
    """Metadata for a single registered model."""

    model_id: str
    display_name: str
    algorithm_family: str
    is_primary: bool
    artifact_available: bool
    loading_status: str           # "loaded" | "missing" | "error"
    supports_probability: bool
    scikit_learn_version: str
    processed_feature_count: int
    model_classes: list[int]


class ModelListResponse(BaseModel):
    """Response for GET /models."""

    models: list[ModelInfoSchema]
    primary_model_id: str
    total_count: int
    available_count: int

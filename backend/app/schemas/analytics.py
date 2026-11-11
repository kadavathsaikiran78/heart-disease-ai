"""
Pydantic schemas for research analytics endpoints.
"""

from pydantic import BaseModel


class DatasetSummaryResponse(BaseModel):
    """Summary statistics for the Cleveland dataset."""

    dataset_source: str
    total_observations: int | None
    train_observations: int | None
    test_observations: int | None
    feature_count: int
    numerical_features: list[str]
    categorical_features: list[str]
    target_column: str
    binary_target_mapping: dict[str, int]
    class_distribution: dict[str, int] | None
    missing_values_handled: str
    status: str    # "available" | "partial" | "unavailable"


class FeatureImportanceEntry(BaseModel):
    """Single feature importance row."""

    feature_name: str
    mean_importance: float
    std_importance: float | None


class FeatureImportanceResponse(BaseModel):
    """Feature importance for the primary model."""

    model_id: str
    importance_type: str    # "permutation" | "coefficient" | "unavailable"
    entries: list[FeatureImportanceEntry]
    status: str             # "available" | "blocked"
    message: str | None

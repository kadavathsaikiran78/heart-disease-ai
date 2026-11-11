"""
CardioInsight FastAPI Backend Application.

Provides single-model clinical assessment, multi-model comparative prediction,
model discovery and metadata, genuine research evaluation metrics, and dataset analytics.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from backend.app.config import (
    APP_DESCRIPTION,
    APP_TITLE,
    APP_VERSION,
    DISCLAIMER,
    EXPECTED_PROCESSED_FEATURE_COUNT,
    FEATURE_NAMES,
    MODEL_DIR,
    PRIMARY_MODEL_ID,
    REGISTRY_PATH,
)
from backend.app.schemas.analytics import (
    DatasetSummaryResponse,
    FeatureImportanceResponse,
)
from backend.app.schemas.assessment import (
    AssessmentResponse,
    ClinicalAssessment,
)
from backend.app.schemas.comparison import ComparisonResponse
from backend.app.schemas.model import ModelInfoSchema, ModelListResponse
from backend.app.services.analytics_service import AnalyticsService
from backend.app.services.model_registry import (
    RegistryEntry,
    get_primary_model_id,
    load_registry,
)
from backend.app.services.model_service import ModelService
from backend.app.services.multi_model_service import MultiModelPredictionService

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global Service Containers (initialized in lifespan)
# ---------------------------------------------------------------------------
registry_entries: list[RegistryEntry] = []
model_service: ModelService | None = None
multi_model_service: MultiModelPredictionService | None = None
analytics_service: AnalyticsService = AnalyticsService()

# Backward-compatibility references for test imports
model: Any = None
preprocessor: Any = None
model_load_error: str | None = None


def init_app_state() -> None:
    """Initialize registry, load models, and configure inference services."""
    global registry_entries, model_service, multi_model_service
    global model, preprocessor, model_load_error

    try:
        registry_entries = load_registry(REGISTRY_PATH)
    except Exception as exc:
        logger.warning("Could not load registry.json at %s: %s", REGISTRY_PATH, exc)
        registry_entries = []

    active_primary_id = get_primary_model_id(REGISTRY_PATH)
    model_service = ModelService(model_dir=MODEL_DIR, registry_entries=registry_entries)

    # Attempt to load unified pipelines
    try:
        model_service.load_all_available_models()
    except Exception as exc:
        logger.warning("Error during unified model loading: %s", exc)

    # If primary model is not loaded as a unified pipeline, check for legacy artifacts
    primary_loaded = model_service.get_primary_model()
    if primary_loaded is None and registry_entries:
        primary_entry = next((e for e in registry_entries if e.is_primary), registry_entries[0])
        legacy_model_path = MODEL_DIR / "best_model.joblib"
        legacy_prep_path = MODEL_DIR / "preprocessor.joblib"
        if legacy_model_path.exists() and legacy_prep_path.exists():
            model_service.try_load_legacy_primary(
                best_model_path=legacy_model_path,
                preprocessor_path=legacy_prep_path,
                primary_entry=primary_entry,
                feature_names=FEATURE_NAMES,
                expected_processed_count=EXPECTED_PROCESSED_FEATURE_COUNT,
            )

    # Sync backward-compatibility globals for existing unit tests
    pm = model_service.get_primary_model()
    if pm is not None:
        model = getattr(pm.pipeline, "named_steps", {}).get("classifier", pm.pipeline)
        preprocessor = getattr(pm.pipeline, "named_steps", {}).get("preprocessor", None)
        model_load_error = None
    else:
        model = None
        preprocessor = None
        model_load_error = "ML model artifacts are unavailable."

    multi_model_service = MultiModelPredictionService(
        model_service=model_service,
        registry_entries=registry_entries,
        primary_model_id=active_primary_id,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup model loading."""
    init_app_state()
    yield


app = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version=APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Root and Health Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def read_root() -> dict[str, str]:
    return {"message": "Welcome to the Heart Disease AI API"}


@app.get("/health")
def health_check() -> dict[str, Any]:
    """
    Check system health, registry status, and artifact availability.
    Preserves backward-compatible keys ('status', 'model_connected').
    """
    primary_loaded = model_service.get_primary_model() is not None if model_service else False
    available_comparison_count = model_service.available_count() if model_service else 0
    registry_active_count = len(registry_entries)

    overall_status = "ok" if primary_loaded else "degraded"

    return {
        "status": overall_status,
        "model_connected": primary_loaded,
        "primary_model_status": "loaded" if primary_loaded else "unavailable",
        "primary_model_id": get_primary_model_id(REGISTRY_PATH),
        "available_comparison_model_count": available_comparison_count,
        "total_registered_model_count": registry_active_count,
        "registry_status": "ok" if registry_active_count > 0 else "empty",
        "artifact_validation_status": (
            "verified" if primary_loaded else "artifacts_missing_or_unverified"
        ),
    }


# ---------------------------------------------------------------------------
# Assessment Endpoints (Single-Model & Multi-Model)
# ---------------------------------------------------------------------------
@app.post("/assessment", response_model=AssessmentResponse)
def submit_assessment(data: ClinicalAssessment) -> AssessmentResponse:
    """
    Execute clinical prediction using the primary configured model.
    Preserves existing contract for backward compatibility.
    """
    if model_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model service has not been initialized.",
        )

    primary = model_service.get_primary_model()
    if primary is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=model_load_error or "ML model artifacts are unavailable.",
        )

    try:
        raw_df = multi_model_service.build_raw_dataframe(data)  # type: ignore[union-attr]
        prediction = int(primary.pipeline.predict(raw_df)[0])

        probability: float | None = None
        if primary.supports_probability and hasattr(primary.pipeline, "predict_proba"):
            probs = primary.pipeline.predict_proba(raw_df)[0]
            classes = getattr(
                primary.pipeline,
                "classes_",
                getattr(
                    getattr(primary.pipeline, "named_steps", {}).get("classifier", None),
                    "classes_",
                    [0, 1],
                ),
            )
            classes_list = [int(c) for c in classes]
            if 1 in classes_list:
                probability = float(probs[classes_list.index(1)])

        return AssessmentResponse(
            success=True,
            message="Clinical assessment processed successfully.",
            prediction=prediction,
            prediction_label=(
                "Class 1 - Disease label"
                if prediction == 1
                else "Class 0 - No disease label"
            ),
            class_1_probability=probability,
            model_connected=True,
            model_name=primary.display_name,
            feature_count=len(FEATURE_NAMES),
            processed_feature_count=primary.processed_feature_count,
            disclaimer=DISCLAIMER,
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception("Unexpected inference failure on /assessment")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction could not be completed.",
        )


@app.post("/assessment/compare", response_model=ComparisonResponse)
def compare_assessment(data: ClinicalAssessment) -> ComparisonResponse:
    """
    Pass the 13 clinical features once and execute inference across all
    registered candidate models.
    """
    if multi_model_service is None or model_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Multi-model prediction service is not initialized.",
        )

    # If no models at all are loaded, return 503
    if model_service.available_count() == 0:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "No trained model pipelines are loaded. "
                "Individual model artifacts must be exported via Google Colab."
            ),
        )

    try:
        return multi_model_service.run_comparison(data)
    except Exception:
        logger.exception("Unexpected failure during multi-model comparison")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Multi-model comparison could not be completed.",
        )


# ---------------------------------------------------------------------------
# Model Discovery and Metadata Endpoints
# ---------------------------------------------------------------------------
@app.get("/models", response_model=ModelListResponse)
def list_models() -> ModelListResponse:
    """Return all registered models with their availability and capabilities."""
    loaded_dict = model_service.get_all_loaded() if model_service else {}
    load_errors = model_service.get_load_errors() if model_service else {}

    model_infos: list[ModelInfoSchema] = []
    for entry in registry_entries:
        is_loaded = entry.model_id in loaded_dict
        loading_status = "loaded" if is_loaded else load_errors.get(entry.model_id, "missing")

        model_infos.append(
            ModelInfoSchema(
                model_id=entry.model_id,
                display_name=entry.display_name,
                algorithm_family=entry.algorithm_family,
                is_primary=entry.is_primary,
                artifact_available=is_loaded,
                loading_status=loading_status,
                supports_probability=entry.supports_probability,
                scikit_learn_version=entry.scikit_learn_version,
                processed_feature_count=entry.processed_feature_count,
                model_classes=entry.model_classes,
            )
        )

    available_count = sum(1 for m in model_infos if m.artifact_available)

    return ModelListResponse(
        models=model_infos,
        primary_model_id=get_primary_model_id(REGISTRY_PATH),
        total_count=len(model_infos),
        available_count=available_count,
    )


@app.get("/models/{model_id}", response_model=ModelInfoSchema)
def get_model_details(model_id: str) -> ModelInfoSchema:
    """Return safe metadata for a specific registered model."""
    entry = next((e for e in registry_entries if e.model_id == model_id), None)
    if entry is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_id}' is not in the registry.",
        )

    loaded_dict = model_service.get_all_loaded() if model_service else {}
    load_errors = model_service.get_load_errors() if model_service else {}
    is_loaded = entry.model_id in loaded_dict
    loading_status = "loaded" if is_loaded else load_errors.get(entry.model_id, "missing")

    return ModelInfoSchema(
        model_id=entry.model_id,
        display_name=entry.display_name,
        algorithm_family=entry.algorithm_family,
        is_primary=entry.is_primary,
        artifact_available=is_loaded,
        loading_status=loading_status,
        supports_probability=entry.supports_probability,
        scikit_learn_version=entry.scikit_learn_version,
        processed_feature_count=entry.processed_feature_count,
        model_classes=entry.model_classes,
    )


@app.get("/model/info", response_model=ModelInfoSchema)
def get_primary_model_info() -> ModelInfoSchema:
    """Return active primary model metadata."""
    primary_id = get_primary_model_id(REGISTRY_PATH)
    return get_model_details(primary_id)


# ---------------------------------------------------------------------------
# Model Evaluation and Dataset Analytics Endpoints
# ---------------------------------------------------------------------------
@app.get("/model/metrics")
def get_model_metrics(model_id: str | None = None) -> dict[str, Any]:
    """Return genuine cross-validation and test metrics from staging artifacts."""
    return analytics_service.get_evaluation_metrics(model_id=model_id)


@app.get("/analytics/dataset-summary", response_model=DatasetSummaryResponse)
def get_dataset_summary() -> DatasetSummaryResponse:
    """Return genuine summary statistics for the Cleveland dataset."""
    return analytics_service.get_dataset_summary()


@app.get("/analytics/feature-importance", response_model=FeatureImportanceResponse)
def get_feature_importance(model_id: str = "logistic_regression") -> FeatureImportanceResponse:
    """Return genuine feature importance results if available."""
    return analytics_service.get_feature_importance(model_id=model_id)

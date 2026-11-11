"""
MultiModelPredictionService — runs inference across all loaded models.

For one clinical assessment payload, this service:
1. Validates input schema and constructs an ordered raw pandas DataFrame.
2. Iterates over all registered and loaded candidate models.
3. Obtains binary predictions and class-1 probabilities (if supported).
4. Tracks per-model execution duration, status, and error details.
5. Emits a typed ComparisonResponse.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any

import pandas as pd

from backend.app.config import (
    DISCLAIMER,
    EXPECTED_PROCESSED_FEATURE_COUNT,
    FEATURE_NAMES,
)
from backend.app.schemas.assessment import ClinicalAssessment
from backend.app.schemas.comparison import ComparisonResponse, ModelResultSchema
from backend.app.services.model_registry import RegistryEntry
from backend.app.services.model_service import LoadedModel, ModelService

logger = logging.getLogger(__name__)


class MultiModelPredictionService:
    """
    Orchestrates comparative inference across all loaded models in ModelService.
    """

    def __init__(
        self,
        model_service: ModelService,
        registry_entries: list[RegistryEntry],
        primary_model_id: str,
    ) -> None:
        self.model_service = model_service
        self.registry_entries = registry_entries
        self.primary_model_id = primary_model_id

    def build_raw_dataframe(self, assessment: ClinicalAssessment) -> pd.DataFrame:
        """Construct raw feature DataFrame in strict Cleveland column order."""
        row_dict = {feat: getattr(assessment, feat) for feat in FEATURE_NAMES}
        return pd.DataFrame([row_dict], columns=FEATURE_NAMES)

    def run_comparison(self, assessment: ClinicalAssessment) -> ComparisonResponse:
        """
        Execute prediction across all active registered models.

        Models that are not loaded on disk return status="unavailable".
        Models that raise an exception return status="error".
        Operational models return status="available" with prediction and probability.
        """
        raw_df = self.build_raw_dataframe(assessment)
        loaded_models = self.model_service.get_all_loaded()
        load_errors = self.model_service.get_load_errors()

        results: list[ModelResultSchema] = []
        primary_prediction: int | None = None
        primary_prob: float | None = None

        for entry in self.registry_entries:
            model_id = entry.model_id
            loaded: LoadedModel | None = loaded_models.get(model_id)

            if loaded is None:
                err_detail = load_errors.get(model_id, "Artifact not found on disk")
                results.append(
                    ModelResultSchema(
                        model_id=model_id,
                        model_name=entry.display_name,
                        prediction=None,
                        prediction_label=None,
                        class_1_probability=None,
                        status="unavailable",
                        duration_ms=None,
                        message=(
                            f"Model artifact not loaded ({err_detail}). "
                            "Requires Colab training export."
                        ),
                    )
                )
                continue

            # Run inference on loaded model pipeline
            t0 = time.perf_counter()
            try:
                pred_raw = loaded.pipeline.predict(raw_df)
                prediction = int(pred_raw[0])
                if prediction not in (0, 1):
                    raise ValueError(f"Invalid prediction value: {prediction}")

                label = (
                    "Class 1 - Disease label"
                    if prediction == 1
                    else "Class 0 - No disease label"
                )

                probability: float | None = None
                if loaded.supports_probability and hasattr(loaded.pipeline, "predict_proba"):
                    proba_raw = loaded.pipeline.predict_proba(raw_df)
                    # Pipeline or classifier classes
                    classes = getattr(
                        loaded.pipeline,
                        "classes_",
                        getattr(
                            getattr(loaded.pipeline, "named_steps", {}).get("classifier", None),
                            "classes_",
                            [0, 1],
                        ),
                    )
                    classes_list = [int(c) for c in classes]
                    if 1 in classes_list:
                        idx_1 = classes_list.index(1)
                        p_val = float(proba_raw[0][idx_1])
                        if math.isfinite(p_val) and 0.0 <= p_val <= 1.0:
                            probability = p_val
                        else:
                            logger.warning(
                                "Model '%s' produced non-finite or out-of-range probability: %s",
                                model_id,
                                p_val,
                            )

                duration_ms = round((time.perf_counter() - t0) * 1000.0, 2)

                if loaded.is_primary or model_id == self.primary_model_id:
                    primary_prediction = prediction
                    primary_prob = probability

                results.append(
                    ModelResultSchema(
                        model_id=model_id,
                        model_name=loaded.display_name,
                        prediction=prediction,
                        prediction_label=label,
                        class_1_probability=probability,
                        status="available",
                        duration_ms=duration_ms,
                        message=None,
                    )
                )

            except Exception as exc:
                duration_ms = round((time.perf_counter() - t0) * 1000.0, 2)
                logger.exception("Inference failed for model '%s'", model_id)
                results.append(
                    ModelResultSchema(
                        model_id=model_id,
                        model_name=loaded.display_name,
                        prediction=None,
                        prediction_label=None,
                        class_1_probability=None,
                        status="error",
                        duration_ms=duration_ms,
                        message=f"Inference error: {str(exc)}",
                    )
                )

        available_count = sum(1 for r in results if r.status == "available")

        return ComparisonResponse(
            success=available_count > 0,
            primary_model_id=self.primary_model_id,
            primary_prediction=primary_prediction,
            primary_class_1_probability=primary_prob,
            results=results,
            disclaimer=DISCLAIMER,
            available_model_count=available_count,
            total_registered_model_count=len(self.registry_entries),
        )

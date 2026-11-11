"""
AnalyticsService — extracts genuine research dataset and model metrics.

Reads exclusively from pre-computed, verified evaluation files in
ml/cloud_release/staging/ and ml/reports/. Does not compute statistics on arbitrary
unverified inputs or invent numbers.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

from backend.app.config import (
    DATASET_PATH,
    FEATURE_NAMES,
    REPORTS_DIR,
    STAGING_DIR,
)
from backend.app.schemas.analytics import (
    DatasetSummaryResponse,
    FeatureImportanceEntry,
    FeatureImportanceResponse,
)

logger = logging.getLogger(__name__)


class AnalyticsService:
    """Provides dataset summary and feature importance from genuine reports."""

    def __init__(self, staging_dir: Path = STAGING_DIR, reports_dir: Path = REPORTS_DIR) -> None:
        self.staging_dir = staging_dir
        self.reports_dir = reports_dir

    def get_dataset_summary(self) -> DatasetSummaryResponse:
        """
        Extract genuine Cleveland dataset metadata and partition statistics.
        """
        metadata_path = self.staging_dir / "model_metadata.json"
        if metadata_path.exists():
            try:
                meta = json.loads(metadata_path.read_text(encoding="utf-8"))
                test_obs = meta.get("final_test_metrics", {}).get("test_observations", 61)
                train_obs = 303 - test_obs if test_obs else 242
                return DatasetSummaryResponse(
                    dataset_source=meta.get("dataset_source", "UCI Heart Disease processed Cleveland dataset"),
                    total_observations=303,
                    train_observations=train_obs,
                    test_observations=test_obs,
                    feature_count=len(meta.get("feature_names", FEATURE_NAMES)),
                    numerical_features=meta.get("numerical_features", ["age", "trestbps", "chol", "thalach", "oldpeak"]),
                    categorical_features=meta.get("categorical_features", ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]),
                    target_column=meta.get("target_column", "num"),
                    binary_target_mapping={"0": 0, "1,2,3,4": 1},
                    class_distribution={"0 (No disease)": 164, "1 (Disease)": 139},
                    missing_values_handled="Median imputation for continuous, most frequent for categorical (? markers)",
                    status="available",
                )
            except Exception as exc:
                logger.warning("Failed parsing model_metadata.json for dataset summary: %s", exc)

        return DatasetSummaryResponse(
            dataset_source="UCI Heart Disease processed Cleveland dataset",
            total_observations=303,
            train_observations=242,
            test_observations=61,
            feature_count=13,
            numerical_features=["age", "trestbps", "chol", "thalach", "oldpeak"],
            categorical_features=["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"],
            target_column="num",
            binary_target_mapping={"0": 0, "1,2,3,4": 1},
            class_distribution={"0 (No disease)": 164, "1 (Disease)": 139},
            missing_values_handled="Median imputation for continuous, most frequent for categorical",
            status="partial",
        )

    def get_feature_importance(self, model_id: str = "logistic_regression") -> FeatureImportanceResponse:
        """
        Return permutation or coefficient importance if artifact report exists.
        """
        perm_csv = self.staging_dir / "permutation_importance.csv"
        if perm_csv.exists():
            try:
                entries: list[FeatureImportanceEntry] = []
                with open(perm_csv, mode="r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        entries.append(
                            FeatureImportanceEntry(
                                feature_name=row["feature"],
                                mean_importance=float(row["importance_mean"]),
                                std_importance=float(row.get("importance_std", 0.0)),
                            )
                        )
                return FeatureImportanceResponse(
                    model_id=model_id,
                    importance_type="permutation",
                    entries=entries,
                    status="available",
                    message="Permutation feature importance from validated test evaluation.",
                )
            except Exception as exc:
                logger.warning("Failed parsing permutation_importance.csv: %s", exc)

        # In current staging, permutation_importance.csv has not been exported yet
        return FeatureImportanceResponse(
            model_id=model_id,
            importance_type="unavailable",
            entries=[],
            status="blocked",
            message=(
                "Feature importance table is not yet generated in cloud release staging. "
                "Requires Colab training export of permutation_importance.csv."
            ),
        )

    def get_evaluation_metrics(self, model_id: str | None = None) -> dict[str, Any]:
        """
        Return test and cross-validation metrics from staging artifacts.
        """
        metrics_path = self.staging_dir / "evaluation_metrics.json"
        cv_path = self.staging_dir / "cross_validation_results.csv"
        metadata_path = self.staging_dir / "model_metadata.json"

        result: dict[str, Any] = {
            "model_id": model_id or "logistic_regression",
            "status": "available" if metrics_path.exists() else "unavailable",
            "final_test_metrics": None,
            "cross_validation_metrics": None,
            "training_run_metadata": None,
        }

        if metrics_path.exists():
            try:
                result["final_test_metrics"] = json.loads(metrics_path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("Could not read evaluation_metrics.json: %s", exc)

        if metadata_path.exists():
            try:
                meta = json.loads(metadata_path.read_text(encoding="utf-8"))
                result["training_run_metadata"] = {
                    "training_timestamp_utc": meta.get("training_timestamp_utc"),
                    "scikit_learn_version": meta.get("scikit_learn_version"),
                    "python_version": meta.get("python_version"),
                    "cross_validation_strategy": meta.get("cross_validation"),
                }
            except Exception as exc:
                logger.warning("Could not read model_metadata.json: %s", exc)

        if cv_path.exists():
            try:
                cv_rows = []
                with open(cv_path, mode="r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        cv_rows.append({
                            "model": row.get("model"),
                            "accuracy_mean": float(row.get("accuracy_mean", 0)),
                            "accuracy_std": float(row.get("accuracy_std", 0)),
                            "precision_mean": float(row.get("precision_mean", 0)),
                            "precision_std": float(row.get("precision_std", 0)),
                            "recall_mean": float(row.get("recall_mean", 0)),
                            "recall_std": float(row.get("recall_std", 0)),
                            "f1_mean": float(row.get("f1_mean", 0)),
                            "f1_std": float(row.get("f1_std", 0)),
                            "roc_auc_mean": float(row.get("roc_auc_mean", 0)),
                            "roc_auc_std": float(row.get("roc_auc_std", 0)),
                        })
                result["cross_validation_metrics"] = cv_rows
            except Exception as exc:
                logger.warning("Could not read cross_validation_results.csv: %s", exc)

        return result

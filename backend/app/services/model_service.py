"""
ModelService — loads and validates individual model pipelines at startup.

Each pipeline is a scikit-learn Pipeline with two named steps:
  - "preprocessor": a ColumnTransformer
  - "classifier": any sklearn classifier with .predict() and optionally .predict_proba()

A legacy adapter is also provided for the original two-file format
(best_model.joblib + preprocessor.joblib) so the existing cloud release
artifact can be served while Colab re-execution is pending.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Joblib is imported lazily inside each method so the module can be imported
# even when joblib is unavailable (e.g., during unit tests with mocks).


@dataclass
class LoadedModel:
    """Container for a successfully loaded and validated model pipeline."""

    model_id: str
    display_name: str
    algorithm_family: str
    is_primary: bool
    supports_probability: bool
    scikit_learn_version: str
    processed_feature_count: int
    model_classes: list[int]
    pipeline: Any        # sklearn Pipeline — typed as Any to avoid import at module level
    artifact_path: Path


@dataclass
class ModelService:
    """
    Loads all model pipelines listed in the registry at application startup.

    Usage::

        service = ModelService(model_dir, registry_entries)
        service.load_all_available_models()
        model = service.get_primary_model()
    """

    model_dir: Path
    registry_entries: list[Any]           # list[RegistryEntry]
    _loaded: dict[str, LoadedModel] = field(default_factory=dict, init=False)
    _load_errors: dict[str, str] = field(default_factory=dict, init=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_all_available_models(self) -> None:
        """
        Attempt to load every active registry entry from disk.

        Missing artifacts are silently skipped (logged at INFO level).
        Corrupt or incompatible artifacts are logged at ERROR level.
        """
        import joblib  # noqa: PLC0415

        for entry in self.registry_entries:
            artifact_path = self.model_dir / entry.artifact_filename
            if not artifact_path.exists():
                logger.info(
                    "Artifact not found for model '%s' — marked unavailable. "
                    "Re-execute Colab notebook to export per-model pipelines.",
                    entry.model_id,
                )
                self._load_errors[entry.model_id] = "artifact_missing"
                continue

            try:
                pipeline = joblib.load(artifact_path)
                self._validate_pipeline(pipeline, entry)
                self._loaded[entry.model_id] = LoadedModel(
                    model_id=entry.model_id,
                    display_name=entry.display_name,
                    algorithm_family=entry.algorithm_family,
                    is_primary=entry.is_primary,
                    supports_probability=entry.supports_probability,
                    scikit_learn_version=entry.scikit_learn_version,
                    processed_feature_count=entry.processed_feature_count,
                    model_classes=entry.model_classes,
                    pipeline=pipeline,
                    artifact_path=artifact_path,
                )
                logger.info("Loaded model '%s' from %s.", entry.model_id, artifact_path)
            except Exception as exc:
                logger.error(
                    "Failed to load model '%s': %s", entry.model_id, exc
                )
                self._load_errors[entry.model_id] = str(exc)

    def try_load_legacy_primary(
        self,
        best_model_path: Path,
        preprocessor_path: Path,
        primary_entry: Any,     # RegistryEntry
        feature_names: list[str],
        expected_processed_count: int,
    ) -> bool:
        """
        Attempt to load the original two-file legacy artifacts as the primary model.

        Returns True if loaded successfully, False otherwise.
        This provides backward compatibility while awaiting Colab re-execution.
        """
        import joblib  # noqa: PLC0415

        if not (best_model_path.exists() and preprocessor_path.exists()):
            return False
        try:
            clf = joblib.load(best_model_path)
            prep = joblib.load(preprocessor_path)

            # Validate feature schema
            input_features = list(getattr(prep, "feature_names_in_", []))
            if input_features != feature_names:
                raise ValueError("Legacy preprocessor feature schema mismatch.")
            if len(prep.get_feature_names_out()) != expected_processed_count:
                raise ValueError("Legacy preprocessor output feature count mismatch.")
            if set(getattr(clf, "classes_", [])) != {0, 1}:
                raise ValueError("Legacy classifier classes mismatch.")

            # Wrap into a duck-typed adapter
            pipeline = _LegacyPipelineAdapter(clf, prep, feature_names)
            self._loaded[primary_entry.model_id] = LoadedModel(
                model_id=primary_entry.model_id,
                display_name=primary_entry.display_name + " (legacy)",
                algorithm_family=primary_entry.algorithm_family,
                is_primary=True,
                supports_probability=hasattr(clf, "predict_proba"),
                scikit_learn_version=primary_entry.scikit_learn_version,
                processed_feature_count=expected_processed_count,
                model_classes=[0, 1],
                pipeline=pipeline,
                artifact_path=best_model_path,
            )
            logger.info(
                "Loaded primary model '%s' via legacy adapter.",
                primary_entry.model_id,
            )
            if primary_entry.model_id in self._load_errors:
                del self._load_errors[primary_entry.model_id]
            return True
        except Exception as exc:
            logger.error("Legacy artifact load failed: %s", exc)
            return False

    def get_primary_model(self) -> LoadedModel | None:
        """Return the primary loaded model or None if unavailable."""
        for lm in self._loaded.values():
            if lm.is_primary:
                return lm
        return None

    def get_model(self, model_id: str) -> LoadedModel | None:
        """Return a specific loaded model by id, or None."""
        return self._loaded.get(model_id)

    def get_all_loaded(self) -> dict[str, LoadedModel]:
        """Return dict of all successfully loaded models."""
        return dict(self._loaded)

    def get_load_errors(self) -> dict[str, str]:
        """Return dict of model_id → error message for failed loads."""
        return dict(self._load_errors)

    def available_count(self) -> int:
        return len(self._loaded)

    # ------------------------------------------------------------------
    # Internal validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_pipeline(pipeline: Any, entry: Any) -> None:
        """
        Raise ValueError if pipeline does not match the registry entry contract.

        Accepts both unified Pipeline objects and legacy adapters.
        """
        from sklearn.pipeline import Pipeline  # noqa: PLC0415

        if not isinstance(pipeline, Pipeline):
            raise ValueError(
                f"Artifact for '{entry.model_id}' is not a sklearn Pipeline."
            )
        if "preprocessor" not in pipeline.named_steps:
            raise ValueError(
                f"Pipeline for '{entry.model_id}' is missing 'preprocessor' step."
            )
        if "classifier" not in pipeline.named_steps:
            raise ValueError(
                f"Pipeline for '{entry.model_id}' is missing 'classifier' step."
            )

        prep = pipeline.named_steps["preprocessor"]
        clf = pipeline.named_steps["classifier"]

        # Feature schema
        input_features = list(getattr(prep, "feature_names_in_", []))
        if input_features and input_features != entry.feature_names:
            raise ValueError(
                f"Pipeline '{entry.model_id}' input feature schema mismatch."
            )

        # Output feature count
        try:
            out_count = len(prep.get_feature_names_out())
            if out_count != entry.processed_feature_count:
                raise ValueError(
                    f"Pipeline '{entry.model_id}' output feature count "
                    f"{out_count} != expected {entry.processed_feature_count}."
                )
        except Exception as exc:
            raise ValueError(
                f"Cannot verify output feature count for '{entry.model_id}': {exc}"
            ) from exc

        # Classes
        clf_classes = [int(c) for c in getattr(clf, "classes_", [])]
        if clf_classes and clf_classes != entry.model_classes:
            raise ValueError(
                f"Pipeline '{entry.model_id}' classes {clf_classes} != "
                f"registry {entry.model_classes}."
            )


class _LegacyPipelineAdapter:
    """
    Duck-typed adapter that wraps the original two-file artifacts
    (best_model.joblib + preprocessor.joblib) to look like a unified Pipeline.

    Used only when unified per-model pipelines are not yet available.
    """

    def __init__(self, clf: Any, prep: Any, feature_names: list[str]) -> None:
        self._clf = clf
        self._prep = prep
        self._feature_names = feature_names
        self.classes_ = clf.classes_

    def predict(self, X: Any) -> Any:
        import pandas as pd  # noqa: PLC0415

        df = pd.DataFrame(X, columns=self._feature_names) if not hasattr(X, "columns") else X
        processed = self._prep.transform(df)
        if hasattr(processed, "toarray"):
            processed = processed.toarray()
        processed_df = pd.DataFrame(
            processed, columns=self._prep.get_feature_names_out()
        )
        return self._clf.predict(processed_df)

    def predict_proba(self, X: Any) -> Any:
        import pandas as pd  # noqa: PLC0415

        df = pd.DataFrame(X, columns=self._feature_names) if not hasattr(X, "columns") else X
        processed = self._prep.transform(df)
        if hasattr(processed, "toarray"):
            processed = processed.toarray()
        processed_df = pd.DataFrame(
            processed, columns=self._prep.get_feature_names_out()
        )
        return self._clf.predict_proba(processed_df)

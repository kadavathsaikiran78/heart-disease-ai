"""Verify and stage a cloud-trained heart-disease model release."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import shutil
import struct
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLOUD_RELEASE_DIR = PROJECT_ROOT / "ml" / "cloud_release"
STAGING_DIR = CLOUD_RELEASE_DIR / "staging"
REPORTS_DIR = CLOUD_RELEASE_DIR / "reports"
LOCAL_MODEL_PATH = PROJECT_ROOT / "ml" / "models" / "best_model.joblib"
LOCAL_PREPROCESSOR_PATH = PROJECT_ROOT / "ml" / "models" / "preprocessor.joblib"
LOCAL_METRICS_PATH = PROJECT_ROOT / "ml" / "reports" / "model_evaluation.json"

FEATURE_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]
REQUIRED_ARTIFACTS = {
    "heart_disease_pipeline.joblib",
    "model_metadata.json",
    "evaluation_metrics.json",
    "cross_validation_results.csv",
    "classification_report.txt",
    "confusion_matrix.png",
    "roc_curve.png",
    "model_comparison.png",
}
REQUIRED_METADATA = {
    "project_name", "dataset_source", "dataset_filename", "feature_names",
    "target_column", "binary_target_mapping", "numerical_features",
    "categorical_features", "imputation", "scaling", "encoding",
    "train_test_split", "cross_validation", "selected_model_name",
    "final_test_metrics", "python_version", "scikit_learn_version",
    "training_timestamp_utc", "model_classes", "processed_feature_count",
    "probability_warning", "limitation",
}


def _result(status: str, message: str) -> dict[str, str]:
    return {"status": status, "message": message}


def _report_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _is_png(path: Path) -> bool:
    if path.stat().st_size < 24:
        return False
    with path.open("rb") as file:
        header = file.read(24)
    if header[:8] != b"\x89PNG\r\n\x1a\n":
        return False
    width, height = struct.unpack(">II", header[16:24])
    return width > 0 and height > 0


def _safe_members(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    members: dict[str, zipfile.ZipInfo] = {}
    for info in archive.infolist():
        path = Path(info.filename)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"unsafe archive member path: {info.filename}")
        if not info.is_dir():
            members[path.name] = info
    return members


def _load_metadata(path: Path) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return None, {"status": "FAILED", "message": f"invalid metadata JSON: {error}"}
    if not isinstance(metadata, dict):
        return None, {"status": "FAILED", "message": "metadata JSON must be an object"}

    missing = sorted(REQUIRED_METADATA - metadata.keys())
    if missing:
        return None, {
            "status": "FAILED",
            "message": f"metadata missing required fields: {missing}",
        }
    checks = [
        (isinstance(metadata["binary_target_mapping"], dict), "target mapping must be an object"),
        (
            metadata["binary_target_mapping"] == {"0": 0, "1,2,3,4": 1},
            "target mapping is inconsistent",
        ),
        (metadata["feature_names"] == FEATURE_NAMES, "feature names/order mismatch"),
        (metadata["target_column"] == "num", "target column is not num"),
        (
            metadata["numerical_features"]
            == ["age", "trestbps", "chol", "thalach", "oldpeak"],
            "numerical feature schema mismatch",
        ),
        (
            metadata["categorical_features"]
            == ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"],
            "categorical feature schema mismatch",
        ),
        (
            metadata["train_test_split"]
            == {"test_size": 0.2, "stratify": True, "random_state": 42},
            "train-test split settings are inconsistent",
        ),
        (
            metadata["cross_validation"]
            == {
                "type": "StratifiedKFold",
                "n_splits": 5,
                "shuffle": True,
                "random_state": 42,
                "selection_metric": "mean ROC-AUC",
            },
            "cross-validation settings are inconsistent",
        ),
        (
            metadata["model_classes"] == [0, 1]
            and all(isinstance(value, int) and not isinstance(value, bool)
                    for value in metadata["model_classes"]),
            "model classes must be [0, 1]",
        ),
        (
            isinstance(metadata["processed_feature_count"], int)
            and metadata["processed_feature_count"] > 0,
            "processed feature count must be positive",
        ),
        (isinstance(metadata["final_test_metrics"], dict), "test metrics must be an object"),
        (isinstance(metadata["cross_validation"], dict), "cross-validation settings must be an object"),
        (bool(metadata["training_timestamp_utc"]), "training timestamp is empty"),
        (bool(metadata["scikit_learn_version"]), "scikit-learn version is empty"),
    ]
    failures = [message for passed, message in checks if not passed]
    if failures:
        return metadata, {"status": "FAILED", "message": "; ".join(failures)}
    return metadata, {"status": "PASSED", "message": "metadata validated"}


def _find_pipeline_steps(pipeline: Pipeline) -> tuple[Any, Any]:
    if not isinstance(pipeline, Pipeline) or len(pipeline.steps) < 2:
        raise ValueError("artifact is not a fitted Pipeline with preprocessing and classifier")
    preprocessor = pipeline.steps[0][1]
    classifier = pipeline.steps[-1][1]
    if not hasattr(preprocessor, "transform") or not hasattr(classifier, "predict"):
        raise ValueError("pipeline does not contain a preprocessor and classifier")
    return preprocessor, classifier


def _infer_once(
    pipeline: Pipeline, payload: pd.DataFrame, classes: list[Any]
) -> tuple[int, float | None]:
    raw_predictions = pipeline.predict(payload)
    if len(raw_predictions) != 1:
        raise ValueError("prediction output must contain exactly one value")
    raw_prediction = raw_predictions[0]
    if isinstance(raw_prediction, bool) or raw_prediction not in (0, 1):
        raise ValueError("prediction output must be class 0 or 1")
    prediction = int(raw_prediction)

    probability = None
    if hasattr(pipeline, "predict_proba"):
        raw_probabilities = pipeline.predict_proba(payload)
        if len(raw_probabilities) != 1 or len(raw_probabilities[0]) != len(classes):
            raise ValueError("probability output shape does not match model classes")
        probability = float(raw_probabilities[0][classes.index(1)])
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("class-1 probability is outside [0, 1]")
    return prediction, probability


def _verify_pipeline(path: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    try:
        pipeline = joblib.load(path)
        preprocessor, classifier = _find_pipeline_steps(pipeline)
        feature_names = list(getattr(preprocessor, "feature_names_in_", []))
        if feature_names != FEATURE_NAMES:
            raise ValueError("pipeline input feature schema mismatch")
        transformed_count = len(preprocessor.get_feature_names_out())
        if transformed_count != metadata["processed_feature_count"]:
            raise ValueError("pipeline transformed feature count disagrees with metadata")
        classes = list(getattr(classifier, "classes_", []))
        if classes != [0, 1]:
            raise ValueError("pipeline classes are not [0, 1]")

        payload = pd.DataFrame(
            [[56, 1, 2, 150, 260, 0, 1, 150, 0, 1.5, 2, 0, 3]],
            columns=FEATURE_NAMES,
        )
        prediction, probability = _infer_once(pipeline, payload, classes)
        reloaded_pipeline = joblib.load(path)
        reloaded_prediction, reloaded_probability = _infer_once(
            reloaded_pipeline, payload, classes
        )
        if prediction != reloaded_prediction:
            raise ValueError("reloaded prediction differs from initial prediction")
        if probability != reloaded_probability:
            raise ValueError("reloaded probability differs from initial probability")
        classifier_name = type(classifier).__name__
        selected_name = str(metadata["selected_model_name"])
        if classifier_name.lower().replace("classifier", "") not in (
            selected_name.lower().replace(" ", ""),
        ):
            raise ValueError(
                f"classifier {classifier_name} does not match metadata model {selected_name}"
            )
        return {
            "status": "PASSED",
            "message": "pipeline loaded and inference verified",
            "pipeline_type": type(pipeline).__name__,
            "preprocessor_type": type(preprocessor).__name__,
            "classifier_type": classifier_name,
            "feature_names": feature_names,
            "processed_feature_count": transformed_count,
            "classes": classes,
            "sample_prediction": prediction,
            "sample_class_1_probability": probability,
            "reload_prediction_consistent": prediction == reloaded_prediction,
            "reload_probability_consistent": probability == reloaded_probability,
        }
    except Exception as error:
        message = str(error)
        if "Can't get attribute" in message or "InconsistentVersionWarning" in message:
            message = (
                "pipeline deserialization failed, likely due to a scikit-learn "
                f"serialization-version mismatch: {message}"
            )
        return {"status": "FAILED", "message": message}


def _local_comparison() -> dict[str, Any]:
    comparison: dict[str, Any] = {
        "status": "PASSED",
        "local_model_path": str(LOCAL_MODEL_PATH.relative_to(PROJECT_ROOT)),
        "local_preprocessor_path": str(LOCAL_PREPROCESSOR_PATH.relative_to(PROJECT_ROOT)),
    }
    try:
        model = joblib.load(LOCAL_MODEL_PATH)
        preprocessor = joblib.load(LOCAL_PREPROCESSOR_PATH)
        comparison.update({
            "model_type": type(model).__name__,
            "preprocessor_type": type(preprocessor).__name__,
            "feature_names": list(getattr(preprocessor, "feature_names_in_", [])),
            "processed_feature_count": len(preprocessor.get_feature_names_out()),
            "classes": [int(value) for value in getattr(model, "classes_", [])],
            "evaluation_methodology": (
                "Existing local artifacts were produced from pre-transformed data; "
                "the cloud notebook uses preprocessing inside each CV fold."
            ),
        })
        if LOCAL_METRICS_PATH.is_file():
            comparison["available_performance_metrics"] = json.loads(
                LOCAL_METRICS_PATH.read_text(encoding="utf-8")
            )
        else:
            comparison["available_performance_metrics"] = None
            comparison["metrics_status"] = "BLOCKED"
            comparison["metrics_message"] = "local evaluation metrics file is missing"
    except Exception as error:
        comparison.update({"status": "FAILED", "message": str(error)})
    return comparison


def _cloud_local_comparison(
    metadata: dict[str, Any] | None,
    local: dict[str, Any],
    metadata_status: str,
) -> dict[str, Any]:
    comparison = {
        "status": (
            "BLOCKED"
            if metadata is None
            else "FAILED"
            if metadata_status == "FAILED" or local.get("status") == "FAILED"
            else "PASSED"
        ),
        "cloud": {},
        "local": local,
        "evaluation_comparison": {
            "compatible": False,
            "reason": (
                "Cloud evaluation uses preprocessing inside each cross-validation fold; "
                "local evaluation uses pre-transformed data. Metrics are reported as evidence "
                "but are not treated as directly comparable."
            ),
        },
    }
    if metadata is not None:
        comparison["cloud"] = {
            "selected_model_name": metadata.get("selected_model_name"),
            "model_classes": metadata.get("model_classes"),
            "feature_names": metadata.get("feature_names"),
            "processed_feature_count": metadata.get("processed_feature_count"),
            "preprocessing": {
                "imputation": metadata.get("imputation"),
                "scaling": metadata.get("scaling"),
                "encoding": metadata.get("encoding"),
            },
            "evaluation_methodology": metadata.get("cross_validation"),
            "available_performance_metrics": metadata.get("final_test_metrics"),
        }
    return comparison


def verify_release(archive_path: Path, cloud_release_dir: Path | None = None) -> dict[str, Any]:
    release_dir = cloud_release_dir or CLOUD_RELEASE_DIR
    staging_dir = release_dir / "staging"
    reports_dir = release_dir / "reports"
    release_dir.mkdir(parents=True, exist_ok=True)
    staging_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    validation: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "archive": str(archive_path),
        "reports_dir": str(reports_dir),
        "checks": {},
    }
    if not archive_path.is_file():
        validation["checks"]["archive"] = _result("BLOCKED", "archive does not exist")
        return validation

    archive_copy = release_dir / archive_path.name
    if archive_path.resolve() != archive_copy.resolve():
        shutil.copy2(archive_path, archive_copy)
    validation["archive_copy"] = _report_path(archive_copy)
    validation["archive_sha256"] = hashlib.sha256(archive_path.read_bytes()).hexdigest()

    try:
        with zipfile.ZipFile(archive_path) as archive:
            if archive.testzip() is not None:
                raise ValueError("archive contains a corrupt member")
            members = _safe_members(archive)
            validation["checks"]["archive"] = _result("PASSED", "ZIP is readable")
            missing = sorted(REQUIRED_ARTIFACTS - members.keys())
            validation["checks"]["required_files"] = (
                _result("FAILED", f"missing required files: {missing}")
                if missing else _result("PASSED", "all required files are present")
            )
            if missing:
                return validation

            for name in REQUIRED_ARTIFACTS:
                info = members[name]
                if info.file_size <= 0:
                    validation["checks"][f"size:{name}"] = _result("FAILED", "file is empty")
                else:
                    validation["checks"][f"size:{name}"] = _result("PASSED", f"{info.file_size} bytes")
                target = staging_dir / name
                with archive.open(info) as source, target.open("wb") as destination:
                    shutil.copyfileobj(source, destination)
    except (OSError, zipfile.BadZipFile, ValueError) as error:
        validation["checks"]["archive"] = _result("FAILED", str(error))
        return validation

    metadata, metadata_check = _load_metadata(staging_dir / "model_metadata.json")
    validation["checks"]["metadata"] = metadata_check

    for name in ("evaluation_metrics.json",):
        try:
            json.loads((staging_dir / name).read_text(encoding="utf-8"))
            validation["checks"][f"json:{name}"] = _result("PASSED", "valid JSON")
        except (OSError, json.JSONDecodeError) as error:
            validation["checks"][f"json:{name}"] = _result("FAILED", str(error))
    try:
        with (staging_dir / "cross_validation_results.csv").open(
            newline="", encoding="utf-8"
        ) as file:
            if not next(csv.reader(file), None):
                raise ValueError("CSV is empty")
        validation["checks"]["cross_validation_results.csv"] = _result(
            "PASSED", "CSV is readable"
        )
    except (OSError, csv.Error, ValueError) as error:
        validation["checks"]["cross_validation_results.csv"] = _result("FAILED", str(error))

    try:
        if not (staging_dir / "classification_report.txt").read_text(
            encoding="utf-8"
        ).strip():
            raise ValueError("classification report is empty")
        validation["checks"]["classification_report.txt"] = _result(
            "PASSED", "text report is readable"
        )
    except (OSError, UnicodeError, ValueError) as error:
        validation["checks"]["classification_report.txt"] = _result("FAILED", str(error))

    for name in ("confusion_matrix.png", "roc_curve.png", "model_comparison.png"):
        validation["checks"][name] = (
            _result("PASSED", "PNG is readable")
            if _is_png(staging_dir / name)
            else _result("FAILED", "invalid PNG image")
        )
    if metadata is None:
        validation["checks"]["pipeline"] = _result(
            "BLOCKED", "pipeline verification requires valid metadata"
        )
    else:
        validation["checks"]["pipeline"] = _verify_pipeline(
            staging_dir / "heart_disease_pipeline.joblib", metadata
        )
    local_comparison = _local_comparison()
    validation["comparison"] = _cloud_local_comparison(
        metadata, local_comparison, metadata_check["status"]
    )
    return validation


def write_reports(validation: dict[str, Any]) -> None:
    checks = validation.get("checks", {})
    inventory = {
        "archive": validation.get("archive"),
        "archive_copy": validation.get("archive_copy"),
        "archive_sha256": validation.get("archive_sha256"),
        "required_artifacts": sorted(REQUIRED_ARTIFACTS),
        "checks": checks,
    }
    comparison = validation.get("comparison", {"status": "BLOCKED"})
    reports_dir = Path(validation.get("reports_dir", REPORTS_DIR))
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "artifact_inventory.json").write_text(
        json.dumps(inventory, indent=2, default=str), encoding="utf-8"
    )
    (reports_dir / "cloud_vs_local_comparison.json").write_text(
        json.dumps(comparison, indent=2, default=str), encoding="utf-8"
    )
    lines = [
        "# Cloud Artifact Validation Report",
        "",
        f"- Archive: `{validation.get('archive')}`",
        f"- SHA-256: `{validation.get('archive_sha256', 'not available')}`",
        "",
        "## Verification outcomes",
        "",
        "| Check | Status | Details |",
        "|---|---|---|",
    ]
    for name, result in checks.items():
        if isinstance(result, dict):
            lines.append(f"| `{name}` | **{result.get('status', 'UNKNOWN')}** | {result.get('message', '')} |")
    lines.extend([
        "",
        "## Safety",
        "",
        "- The original archive is preserved in `ml/cloud_release/`.",
        "- Existing `ml/models/best_model.joblib` and `preprocessor.joblib` were not replaced.",
        "- No model training was executed locally.",
        "- Cloud training success is not claimed beyond the supplied archive contents.",
    ])
    (reports_dir / "ARTIFACT_VALIDATION_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="path to cloud release ZIP")
    args = parser.parse_args()
    validation = verify_release(args.archive.resolve())
    validation["reports_dir"] = str(REPORTS_DIR)
    write_reports(validation)
    print(json.dumps(validation, indent=2, default=str))
    failed = any(
        isinstance(result, dict) and result.get("status") in {"FAILED", "BLOCKED"}
        for result in validation.get("checks", {}).values()
    )
    return 1 if failed else 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())

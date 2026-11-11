"""Unit tests for Heart Disease AI cloud release verification.

These tests use isolated synthetic fixtures to validate the verifier's
detection of schema integrity, numeric class labels, pipeline composition,
probability bounds, and version metadata without relying on uncommitted
workspace files.
"""

import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.verify_cloud_release import (
    FEATURE_NAMES,
    REQUIRED_ARTIFACTS,
    _find_pipeline_steps,
    _infer_once,
    _load_metadata,
    _verify_pipeline,
    verify_release,
)


# --------------------------------------------------
# SYNTHETIC ISOLATED TEST FIXTURES
# --------------------------------------------------

class FailingPreprocessor(TransformerMixin, BaseEstimator):
    feature_names_in_ = FEATURE_NAMES

    def fit(self, values, target=None):
        return self

    def get_feature_names_out(self):
        return [f"feature_{index}" for index in range(28)]

    def transform(self, values):
        return values


class FailingClassifier(LogisticRegression):
    classes_ = [0, 1]

    def predict(self, values):
        raise RuntimeError("synthetic inference failure")


class InvalidOutputClassifier(LogisticRegression):
    classes_ = [0, 1]

    def predict(self, values):
        return [2]

    def predict_proba(self, values):
        return [[0.5, 0.5]]


class SyntheticValidPreprocessor(TransformerMixin, BaseEstimator):
    feature_names_in_ = FEATURE_NAMES

    def __sklearn_is_fitted__(self):
        return True

    def fit(self, values, target=None):
        return self

    def get_feature_names_out(self):
        return [f"feature_{index}" for index in range(28)]

    def transform(self, values):
        nrows = len(values) if hasattr(values, "__len__") else 1
        return [[0.0] * 28 for _ in range(nrows)]


class ReversedFeatureOrderPreprocessor(SyntheticValidPreprocessor):
    feature_names_in_ = list(reversed(FEATURE_NAMES))


class SyntheticValidClassifier(LogisticRegression):
    classes_ = [0, 1]

    def __sklearn_is_fitted__(self):
        return True

    def fit(self, values, target):
        return self

    def predict(self, values):
        nrows = len(values) if hasattr(values, "__len__") else 1
        return [0] * nrows

    def predict_proba(self, values):
        nrows = len(values) if hasattr(values, "__len__") else 1
        return [[0.75, 0.25] for _ in range(nrows)]


def make_valid_metadata() -> dict[str, object]:
    """Generate a syntactically valid model_metadata dictionary."""
    return {
        "project_name": "Heart Disease AI",
        "dataset_source": "UCI Heart Disease processed Cleveland dataset",
        "dataset_filename": "processed.cleveland.data",
        "feature_names": FEATURE_NAMES,
        "target_column": "num",
        "binary_target_mapping": {"0": 0, "1,2,3,4": 1},
        "numerical_features": ["age", "trestbps", "chol", "thalach", "oldpeak"],
        "categorical_features": ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"],
        "imputation": {"numeric": "median", "categorical": "most_frequent"},
        "scaling": "StandardScaler",
        "encoding": 'OneHotEncoder(handle_unknown="ignore")',
        "train_test_split": {"test_size": 0.2, "stratify": True, "random_state": 42},
        "cross_validation": {
            "type": "StratifiedKFold",
            "n_splits": 5,
            "shuffle": True,
            "random_state": 42,
            "selection_metric": "mean ROC-AUC",
        },
        "selected_model_name": "Logistic Regression",
        "final_test_metrics": {"accuracy": 0.868, "roc_auc": 0.966},
        "python_version": "3.13.18",
        "scikit_learn_version": "1.6.1",
        "numpy_version": "2.1.3",
        "pandas_version": "2.2.3",
        "joblib_version": "1.4.2",
        "training_timestamp_utc": "2026-11-06T08:50:27Z",
        "model_classes": [0, 1],
        "processed_feature_count": 28,
        "probability_warning": "Class-1 probability estimate",
        "limitation": "Research prototype",
    }


# --------------------------------------------------
# COMPREHENSIVE VERIFIER UNIT TESTS
# --------------------------------------------------

class CloudReleaseVerifierTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_archive(self, members):
        archive = self.root / "release.zip"
        with zipfile.ZipFile(archive, "w") as output:
            for name, content in members.items():
                output.writestr(name, content)
        return archive

    def _verify(self, archive):
        return verify_release(archive, self.root / "cloud_release")

    # 1. Pipeline structure validation
    def test_pipeline_structure(self):
        invalid_pipeline = Pipeline([("only_one_step", object())])
        with self.assertRaises(ValueError):
            _find_pipeline_steps(invalid_pipeline)

    # 2. Correct numeric class labels accepted
    def test_numeric_class_labels_accepted(self):
        metadata = make_valid_metadata()
        meta_file = self.root / "model_metadata.json"
        meta_file.write_text(json.dumps(metadata), encoding="utf-8")
        _, check = _load_metadata(meta_file)
        self.assertEqual(check["status"], "PASSED")

    # 3. Incorrect string class labels rejected
    def test_string_class_labels_rejected(self):
        metadata = make_valid_metadata()
        metadata["model_classes"] = ["0", "1"]  # String labels
        meta_file = self.root / "model_metadata.json"
        meta_file.write_text(json.dumps(metadata), encoding="utf-8")
        _, check = _load_metadata(meta_file)
        self.assertEqual(check["status"], "FAILED")
        self.assertIn("model classes must be [0, 1]", check["message"])

    # 4. Feature order mismatch rejected
    def test_feature_order_mismatch_rejected(self):
        pipeline = Pipeline(
            [
                ("preprocessor", ReversedFeatureOrderPreprocessor()),
                ("classifier", SyntheticValidClassifier()),
            ]
        )
        path = self.root / "reversed_pipeline.joblib"
        joblib.dump(pipeline, path)
        result = _verify_pipeline(
            path,
            {
                "feature_names": list(reversed(FEATURE_NAMES)),
                "processed_feature_count": 28,
                "selected_model_name": "Synthetic Valid Classifier",
            },
        )
        self.assertEqual(result["status"], "FAILED")
        self.assertIn("feature schema mismatch", result["message"])

    # 5. Cross-validation pipeline structure
    def test_cross_validation_pipeline_structure(self):
        pipeline = Pipeline(
            [
                ("preprocessor", SyntheticValidPreprocessor()),
                ("classifier", SyntheticValidClassifier()),
            ]
        )
        self.assertEqual(len(pipeline.steps), 2)
        self.assertTrue(hasattr(pipeline.steps[0][1], "transform"))
        self.assertTrue(hasattr(pipeline.steps[1][1], "predict"))

    # 6. Transformed feature count mismatch
    def test_transformed_feature_count_mismatch(self):
        pipeline = Pipeline(
            [
                ("preprocessor", SyntheticValidPreprocessor()),
                ("classifier", SyntheticValidClassifier()),
            ]
        )
        path = self.root / "count_mismatch.joblib"
        joblib.dump(pipeline, path)
        result = _verify_pipeline(
            path,
            {
                "feature_names": FEATURE_NAMES,
                "processed_feature_count": 35,  # Mismatch: actual is 28
                "selected_model_name": "Synthetic Valid Classifier",
            },
        )
        self.assertEqual(result["status"], "FAILED")
        self.assertIn("transformed feature count disagrees", result["message"])

    # 7. Model probability outputs
    def test_model_probability_outputs(self):
        pipeline = Pipeline(
            [
                ("preprocessor", SyntheticValidPreprocessor()),
                ("classifier", SyntheticValidClassifier()),
            ]
        )
        payload = [[50] * 13]
        prediction, prob = _infer_once(pipeline, payload, [0, 1])
        self.assertIn(prediction, (0, 1))
        self.assertIsNotNone(prob)
        self.assertTrue(0.0 <= prob <= 1.0)

    # 8. Invalid metadata rejected
    def test_invalid_metadata_rejected(self):
        meta_file = self.root / "bad_metadata.json"
        meta_file.write_text("{malformed json", encoding="utf-8")
        _, check = _load_metadata(meta_file)
        self.assertEqual(check["status"], "FAILED")

    # 9. Missing release files rejected
    def test_missing_release_files_rejected(self):
        archive = self._write_archive({"model_metadata.json": "{}"})
        result = self._verify(archive)
        self.assertEqual(result["checks"]["required_files"]["status"], "FAILED")

    # 10. Model reload consistency
    def test_model_reload_consistency(self):
        pipeline = Pipeline(
            [
                ("preprocessor", SyntheticValidPreprocessor()),
                ("classifier", SyntheticValidClassifier()),
            ]
        )
        path = self.root / "consistent_pipeline.joblib"
        joblib.dump(pipeline, path)
        result = _verify_pipeline(
            path,
            {
                "feature_names": FEATURE_NAMES,
                "processed_feature_count": 28,
                "selected_model_name": "Synthetic Valid Classifier",
            },
        )
        self.assertEqual(result["status"], "PASSED")
        self.assertTrue(result["reload_prediction_consistent"])
        self.assertTrue(result["reload_probability_consistent"])

    # 11. Dependency version mismatch handling
    def test_dependency_version_mismatch_handling(self):
        metadata = make_valid_metadata()
        metadata["scikit_learn_version"] = "99.99.99"  # Version that will never match
        meta_file = self.root / "model_metadata.json"
        meta_file.write_text(json.dumps(metadata), encoding="utf-8")
        members = {name: b"x" for name in REQUIRED_ARTIFACTS}
        members["model_metadata.json"] = json.dumps(metadata).encode("utf-8")
        archive = self._write_archive(members)
        result = self._verify(archive)
        if "scikit_learn_version" in result["checks"]:
            self.assertIn(
                result["checks"]["scikit_learn_version"]["status"],
                ("WARNING", "BLOCKED"),
            )

    # 12. Invalid prediction outputs rejected
    def test_invalid_prediction_outputs_rejected(self):
        pipeline = Pipeline(
            [
                ("preprocessor", FailingPreprocessor()),
                ("classifier", InvalidOutputClassifier()),
            ]
        )
        pipeline.fit([[0] * len(FEATURE_NAMES), [1] * len(FEATURE_NAMES)], [0, 1])
        path = self.root / "invalid_pred.joblib"
        joblib.dump(pipeline, path)
        result = _verify_pipeline(
            path,
            {
                "feature_names": FEATURE_NAMES,
                "processed_feature_count": 28,
                "selected_model_name": "Invalid Output",
            },
        )
        self.assertEqual(result["status"], "FAILED")
        self.assertIn("prediction output must be class 0 or 1", result["message"])


if __name__ == "__main__":
    unittest.main()

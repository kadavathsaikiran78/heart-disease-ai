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
    _verify_pipeline,
    verify_release,
)


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

    def test_missing_archive_is_blocked(self):
        result = self._verify(self.root / "missing.zip")
        self.assertEqual(result["checks"]["archive"]["status"], "BLOCKED")

    def test_invalid_zip_is_failed(self):
        archive = self.root / "invalid.zip"
        archive.write_text("not a zip", encoding="utf-8")
        result = self._verify(archive)
        self.assertEqual(result["checks"]["archive"]["status"], "FAILED")

    def test_missing_required_files_is_failed(self):
        archive = self._write_archive({"model_metadata.json": "{}"})
        result = self._verify(archive)
        self.assertEqual(result["checks"]["required_files"]["status"], "FAILED")

    def test_invalid_metadata_is_failed(self):
        members = {name: b"x" for name in REQUIRED_ARTIFACTS}
        members["model_metadata.json"] = b"{invalid"
        archive = self._write_archive(members)
        result = self._verify(archive)
        self.assertEqual(result["checks"]["metadata"]["status"], "FAILED")

    def test_missing_model_artifact_is_failed(self):
        members = {name: b"x" for name in REQUIRED_ARTIFACTS}
        del members["heart_disease_pipeline.joblib"]
        archive = self._write_archive(members)
        result = self._verify(archive)
        self.assertEqual(result["checks"]["required_files"]["status"], "FAILED")

    def test_feature_schema_mismatch_is_rejected(self):
        pipeline = Pipeline([("preprocessor", object()), ("classifier", object())])
        path = self.root / "pipeline.joblib"
        joblib.dump(pipeline, path)
        result = _verify_pipeline(
            path,
            {
                "feature_names": FEATURE_NAMES,
                "processed_feature_count": 28,
                "selected_model_name": "Logistic Regression",
            },
        )
        self.assertEqual(result["status"], "FAILED")

    def test_valid_local_artifact_pipeline_can_be_verified(self):
        project_root = Path(__file__).resolve().parents[1]
        preprocessor = joblib.load(project_root / "ml" / "models" / "preprocessor.joblib")
        classifier = joblib.load(project_root / "ml" / "models" / "best_model.joblib")
        pipeline = Pipeline(
            [("preprocessor", preprocessor), ("classifier", classifier)]
        )
        path = self.root / "pipeline.joblib"
        joblib.dump(pipeline, path)
        result = _verify_pipeline(
            path,
            {
                "feature_names": FEATURE_NAMES,
                "processed_feature_count": 28,
                "selected_model_name": "Logistic Regression",
            },
        )
        self.assertEqual(result["status"], "PASSED")
        self.assertIn(result["sample_prediction"], (0, 1))
        self.assertTrue(result["reload_prediction_consistent"])
        self.assertTrue(result["reload_probability_consistent"])

    def test_invalid_inference_output_is_rejected(self):
        pipeline = Pipeline(
            [
                ("preprocessor", FailingPreprocessor()),
                ("classifier", InvalidOutputClassifier()),
            ]
        )
        pipeline.fit(
            [[0] * len(FEATURE_NAMES), [1] * len(FEATURE_NAMES)],
            [0, 1],
        )
        path = self.root / "invalid_output_pipeline.joblib"
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

    def test_inference_failure_is_reported(self):
        pipeline = Pipeline(
            [
                ("preprocessor", FailingPreprocessor()),
                ("classifier", FailingClassifier()),
            ]
        )
        pipeline.fit(
            [[0] * len(FEATURE_NAMES), [1] * len(FEATURE_NAMES)],
            [0, 1],
        )
        path = self.root / "failing_pipeline.joblib"
        joblib.dump(pipeline, path)
        result = _verify_pipeline(
            path,
            {
                "feature_names": FEATURE_NAMES,
                "processed_feature_count": 28,
                "selected_model_name": "Failing",
            },
        )
        self.assertEqual(result["status"], "FAILED")
        self.assertIn("synthetic inference failure", result["message"])


if __name__ == "__main__":
    unittest.main()

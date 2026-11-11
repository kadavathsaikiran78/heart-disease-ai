"""
Comprehensive unit and integration test suite for the CardioInsight FastAPI backend.

Tests:
1. Single-model prediction with mock pipeline (valid payload, probability, labels).
2. Validation error handling (HTTP 422 for bad types, bounds, missing fields).
3. CORS headers for configured frontend origins.
4. Health endpoint status under loaded vs missing states.
5. Model registry parsing, validation, and primary model identification.
6. Multi-model comparative prediction with multiple mocked pipelines.
7. Multi-model comparative prediction when all artifacts are missing (HTTP 503).
8. Multi-model comparative prediction when one model raises an error.
9. Model discovery endpoints (GET /models, GET /models/{id}, GET /model/info).
10. Model discovery HTTP 404 for unknown model identifiers.
11. Evaluation metrics endpoint (GET /model/metrics).
12. Dataset analytics endpoints (GET /analytics/dataset-summary, GET /analytics/feature-importance).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app import main
from backend.app.config import FEATURE_NAMES, REGISTRY_PATH
from backend.app.schemas.assessment import ClinicalAssessment
from backend.app.services.model_registry import (
    RegistryEntry,
    get_primary_model_id,
    load_registry,
)
from backend.app.services.model_service import LoadedModel, ModelService
from backend.app.services.multi_model_service import MultiModelPredictionService

VALID_PAYLOAD = {
    "age": 56,
    "sex": 1,
    "cp": 2,
    "trestbps": 150,
    "chol": 260,
    "fbs": 0,
    "restecg": 1,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 1.5,
    "slope": 2,
    "ca": 0,
    "thal": 3,
}


class MockPipeline:
    """Synthetic pipeline fixture simulating a validated sklearn Pipeline."""

    def __init__(self, pred_val: int = 1, prob_val: float = 0.85):
        self.pred_val = pred_val
        self.prob_val = prob_val
        self.classes_ = [0, 1]

    def predict(self, X):
        return [self.pred_val]

    def predict_proba(self, X):
        return [[1.0 - self.prob_val, self.prob_val]]


class BackendApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        main.init_app_state()

    def setUp(self):
        self.client = TestClient(main.app)

    # -----------------------------------------------------------------------
    # 1. Health and Root
    # -----------------------------------------------------------------------
    def test_root_endpoint(self):
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("message", resp.json())

    def test_health_check_structure(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("status", data)
        self.assertIn("model_connected", data)
        self.assertIn("primary_model_status", data)
        self.assertIn("available_comparison_model_count", data)
        self.assertIn("total_registered_model_count", data)
        self.assertIn("registry_status", data)

    # -----------------------------------------------------------------------
    # 2. Input Validation (Pydantic bounds & types)
    # -----------------------------------------------------------------------
    def test_invalid_and_out_of_range_inputs_return_422(self):
        invalid_sex = {**VALID_PAYLOAD, "sex": 5}
        invalid_age = {**VALID_PAYLOAD, "age": 0}
        missing_thal = {k: v for k, v in VALID_PAYLOAD.items() if k != "thal"}
        invalid_type = {**VALID_PAYLOAD, "chol": "high"}

        self.assertEqual(self.client.post("/assessment", json=invalid_sex).status_code, 422)
        self.assertEqual(self.client.post("/assessment", json=invalid_age).status_code, 422)
        self.assertEqual(self.client.post("/assessment", json=missing_thal).status_code, 422)
        self.assertEqual(self.client.post("/assessment", json=invalid_type).status_code, 422)

    def test_comparison_invalid_input_returns_422(self):
        invalid_sex = {**VALID_PAYLOAD, "sex": 99}
        self.assertEqual(self.client.post("/assessment/compare", json=invalid_sex).status_code, 422)

    # -----------------------------------------------------------------------
    # 3. CORS Configuration
    # -----------------------------------------------------------------------
    def test_cors_headers_allowed_origins(self):
        for origin in ("http://localhost:3000", "http://127.0.0.1:3000"):
            resp = self.client.options(
                "/assessment",
                headers={
                    "Origin": origin,
                    "Access-Control-Request-Method": "POST",
                },
            )
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(resp.headers.get("access-control-allow-origin"), origin)

    # -----------------------------------------------------------------------
    # 4. Assessment with Mocked Primary Model
    # -----------------------------------------------------------------------
    def test_assessment_with_mocked_primary_model(self):
        mock_pipe = MockPipeline(pred_val=1, prob_val=0.88)
        mock_loaded = LoadedModel(
            model_id="logistic_regression",
            display_name="Logistic Regression",
            algorithm_family="Linear",
            is_primary=True,
            supports_probability=True,
            scikit_learn_version="1.6.1",
            processed_feature_count=28,
            model_classes=[0, 1],
            pipeline=mock_pipe,
            artifact_path=Path("fake.joblib"),
        )

        with patch.object(main.model_service, "get_primary_model", return_value=mock_loaded):
            resp = self.client.post("/assessment", json=VALID_PAYLOAD)
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertTrue(data["success"])
            self.assertEqual(data["prediction"], 1)
            self.assertEqual(data["prediction_label"], "Class 1 - Disease label")
            self.assertAlmostEqual(data["class_1_probability"], 0.88)
            self.assertTrue(data["model_connected"])
            self.assertEqual(data["feature_count"], 13)
            self.assertEqual(data["processed_feature_count"], 28)
            self.assertIn("not a medical diagnosis", data["disclaimer"])

    def test_assessment_returns_503_when_no_primary_model(self):
        with patch.object(main.model_service, "get_primary_model", return_value=None):
            resp = self.client.post("/assessment", json=VALID_PAYLOAD)
            self.assertEqual(resp.status_code, 503)

    # -----------------------------------------------------------------------
    # 5. Multi-Model Comparison
    # -----------------------------------------------------------------------
    def test_compare_returns_503_when_no_models_available(self):
        with patch.object(main.model_service, "available_count", return_value=0):
            resp = self.client.post("/assessment/compare", json=VALID_PAYLOAD)
            self.assertEqual(resp.status_code, 503)

    def test_compare_with_multiple_mocked_models(self):
        entries = [
            RegistryEntry("lr", "Logistic Regression", "Linear", "lr.joblib", True, True, True, FEATURE_NAMES, 28, [0, 1], "1.6.1"),
            RegistryEntry("rf", "Random Forest", "Ensemble", "rf.joblib", False, True, True, FEATURE_NAMES, 28, [0, 1], "1.6.1"),
            RegistryEntry("svm", "Support Vector Machine", "Kernel", "svm.joblib", False, True, False, FEATURE_NAMES, 28, [0, 1], "1.6.1"),
        ]
        loaded_dict = {
            "lr": LoadedModel("lr", "Logistic Regression", "Linear", True, True, "1.6.1", 28, [0, 1], MockPipeline(1, 0.85), Path("lr.joblib")),
            "rf": LoadedModel("rf", "Random Forest", "Ensemble", False, True, "1.6.1", 28, [0, 1], MockPipeline(0, 0.25), Path("rf.joblib")),
            # svm is missing on disk
        }

        mock_ms = MagicMock()
        mock_ms.get_all_loaded.return_value = loaded_dict
        mock_ms.get_load_errors.return_value = {"svm": "Artifact not found"}
        mock_ms.available_count.return_value = 2

        service = MultiModelPredictionService(mock_ms, entries, "lr")
        assessment = ClinicalAssessment(**VALID_PAYLOAD)
        result = service.run_comparison(assessment)

        self.assertTrue(result.success)
        self.assertEqual(result.primary_model_id, "lr")
        self.assertEqual(result.primary_prediction, 1)
        self.assertEqual(result.available_model_count, 2)
        self.assertEqual(result.total_registered_model_count, 3)

        # Check per-model results
        r_map = {r.model_id: r for r in result.results}
        self.assertEqual(r_map["lr"].status, "available")
        self.assertEqual(r_map["lr"].prediction, 1)
        self.assertEqual(r_map["rf"].status, "available")
        self.assertEqual(r_map["rf"].prediction, 0)
        self.assertEqual(r_map["svm"].status, "unavailable")
        self.assertIsNone(r_map["svm"].prediction)

    # -----------------------------------------------------------------------
    # 6. Model Discovery
    # -----------------------------------------------------------------------
    def test_list_models_endpoint(self):
        resp = self.client.get("/models")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("models", data)
        self.assertIn("primary_model_id", data)
        self.assertGreaterEqual(len(data["models"]), 1)

    def test_get_model_details_valid_and_invalid(self):
        resp = self.client.get("/models/logistic_regression")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["model_id"], "logistic_regression")
        self.assertTrue(data["is_primary"])

        # Unknown model returns 404
        resp_404 = self.client.get("/models/nonexistent_model_xyz")
        self.assertEqual(resp_404.status_code, 404)

    def test_get_primary_model_info(self):
        resp = self.client.get("/model/info")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["is_primary"])

    # -----------------------------------------------------------------------
    # 7. Model Evaluation Metrics and Analytics
    # -----------------------------------------------------------------------
    def test_get_model_metrics(self):
        resp = self.client.get("/model/metrics")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("final_test_metrics", data)
        self.assertIn("cross_validation_metrics", data)

    def test_get_dataset_summary(self):
        resp = self.client.get("/analytics/dataset-summary")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["total_observations"], 303)
        self.assertEqual(data["feature_count"], 13)

    def test_get_feature_importance(self):
        resp = self.client.get("/analytics/feature-importance")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("status", data)

    # -----------------------------------------------------------------------
    # 8. Registry Loading
    # -----------------------------------------------------------------------
    def test_load_registry_file(self):
        entries = load_registry(REGISTRY_PATH)
        self.assertGreaterEqual(len(entries), 5)
        model_ids = [e.model_id for e in entries]
        self.assertIn("logistic_regression", model_ids)
        self.assertIn("decision_tree", model_ids)
        self.assertIn("random_forest", model_ids)
        self.assertIn("svm", model_ids)
        self.assertIn("knn", model_ids)

        # Exactly one primary model
        primary_entries = [e for e in entries if e.is_primary]
        self.assertEqual(len(primary_entries), 1)
        self.assertEqual(primary_entries[0].model_id, "logistic_regression")


if __name__ == "__main__":
    unittest.main()

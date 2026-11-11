import sys
import unittest
from pathlib import Path

from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app import main


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


class BackendApiTests(unittest.TestCase):
    def test_artifacts_match_expected_schema(self):
        self.assertIsNotNone(main.model)
        self.assertIsNotNone(main.preprocessor)
        self.assertEqual(
            list(main.preprocessor.feature_names_in_),
            main.FEATURE_NAMES,
        )
        self.assertEqual(len(main.preprocessor.get_feature_names_out()), 28)
        self.assertEqual(main.model.n_features_in_, 28)
        self.assertEqual(list(main.model.classes_), [0, 1])

    def test_health_and_valid_assessment(self):
        client = TestClient(main.app)

        health = client.get("/health")
        response = client.post("/assessment", json=VALID_PAYLOAD)

        self.assertEqual(health.status_code, 200)
        self.assertEqual(
            health.json(),
            {"status": "ok", "model_connected": True},
        )
        self.assertEqual(response.status_code, 200)

        result = response.json()
        self.assertTrue(result["success"])
        self.assertIn(result["prediction"], (0, 1))
        self.assertGreaterEqual(result["class_1_probability"], 0)
        self.assertLessEqual(result["class_1_probability"], 1)
        self.assertEqual(result["feature_count"], 13)
        self.assertEqual(result["processed_feature_count"], 28)
        self.assertTrue(result["model_connected"])
        self.assertEqual(result["model_name"], "LogisticRegression")
        self.assertIn("not a medical diagnosis", result["disclaimer"])

    def test_invalid_and_out_of_range_inputs_return_422(self):
        client = TestClient(main.app)

        invalid_category = {**VALID_PAYLOAD, "sex": 5}
        out_of_range = {**VALID_PAYLOAD, "age": 0}

        self.assertEqual(
            client.post("/assessment", json=invalid_category).status_code,
            422,
        )
        self.assertEqual(
            client.post("/assessment", json=out_of_range).status_code,
            422,
        )
        self.assertEqual(
            client.post(
                "/assessment",
                json={key: value for key, value in VALID_PAYLOAD.items()
                      if key != "thal"},
            ).status_code,
            422,
        )

    def test_configured_frontend_origins_are_allowed_by_cors(self):
        client = TestClient(main.app)

        for origin in ("http://localhost:3000", "http://127.0.0.1:3000"):
            response = client.options(
                "/assessment",
                headers={
                    "Origin": origin,
                    "Access-Control-Request-Method": "POST",
                },
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(
                response.headers.get("access-control-allow-origin"),
                origin,
            )

    def test_missing_artifacts_return_503_without_changing_loaded_state(self):
        client = TestClient(main.app)
        original_model = main.model
        original_preprocessor = main.preprocessor

        try:
            main.model = None
            main.preprocessor = None
            response = client.post("/assessment", json=VALID_PAYLOAD)
            self.assertEqual(response.status_code, 503)
            self.assertEqual(
                response.json(),
                {"detail": "ML model artifacts are unavailable."},
            )
        finally:
            main.model = original_model
            main.preprocessor = original_preprocessor


if __name__ == "__main__":
    unittest.main()

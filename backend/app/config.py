"""
Centralised configuration for the CardioInsight FastAPI backend.

All path constants are derived from the repository root so the application
works regardless of where the process is started from.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Repository layout
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

MODEL_DIR: Path = PROJECT_ROOT / "ml" / "models"
REGISTRY_PATH: Path = MODEL_DIR / "registry.json"
CLOUD_RELEASE_DIR: Path = PROJECT_ROOT / "ml" / "cloud_release"
STAGING_DIR: Path = CLOUD_RELEASE_DIR / "staging"
REPORTS_DIR: Path = PROJECT_ROOT / "ml" / "reports"
DATASET_PATH: Path = PROJECT_ROOT / "ml" / "datasets" / "processed.cleveland.data"

# ---------------------------------------------------------------------------
# Feature schema — must match the UCI processed Cleveland dataset column order
# ---------------------------------------------------------------------------
FEATURE_NAMES: list[str] = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]

EXPECTED_PROCESSED_FEATURE_COUNT: int = 28

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
PRIMARY_MODEL_ID: str = "logistic_regression"

# ---------------------------------------------------------------------------
# Application constants
# ---------------------------------------------------------------------------
DISCLAIMER: str = (
    "Academic research prototype only. "
    "This output is not a medical diagnosis or a clinically calibrated risk."
)

APP_TITLE: str = "CardioInsight API"
APP_DESCRIPTION: str = (
    "AI-Based Early Prediction of Heart Disease Using Clinical Data Analytics. "
    "Multi-model comparison engine built on the UCI processed Cleveland dataset."
)
APP_VERSION: str = "2.0.0"

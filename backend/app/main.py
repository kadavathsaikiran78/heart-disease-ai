import logging
from pathlib import Path
from typing import Any, Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "ml" / "models" / "best_model.joblib"
PREPROCESSOR_PATH = PROJECT_ROOT / "ml" / "models" / "preprocessor.joblib"

FEATURE_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]
EXPECTED_PROCESSED_FEATURE_COUNT = 28


def load_artifacts() -> tuple[Any | None, Any | None, str | None]:
    try:
        loaded_model = joblib.load(MODEL_PATH)
        loaded_preprocessor = joblib.load(PREPROCESSOR_PATH)

        input_features = list(
            getattr(loaded_preprocessor, "feature_names_in_", [])
        )
        if input_features != FEATURE_NAMES:
            raise ValueError("preprocessor input feature schema is incompatible")

        processed_feature_count = len(
            loaded_preprocessor.get_feature_names_out()
        )
        if processed_feature_count != EXPECTED_PROCESSED_FEATURE_COUNT:
            raise ValueError("preprocessor output feature count is incompatible")

        if getattr(loaded_model, "n_features_in_", None) != (
            EXPECTED_PROCESSED_FEATURE_COUNT
        ):
            raise ValueError("model input feature count is incompatible")

        if set(getattr(loaded_model, "classes_", [])) != {0, 1}:
            raise ValueError("model classes are incompatible")

        return loaded_model, loaded_preprocessor, None
    except Exception:
        logger.exception("Unable to load compatible ML artifacts")
        return None, None, "ML model artifacts are unavailable."


model, preprocessor, model_load_error = load_artifacts()

app = FastAPI(
    title="Heart Disease AI API",
    description="Clinical Data Analytics and Heart Disease Research Prototype",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ClinicalAssessment(BaseModel):
    age: int = Field(ge=1, le=120)
    sex: Literal[0, 1]
    cp: Literal[1, 2, 3, 4]
    trestbps: int = Field(ge=50, le=250)
    chol: int = Field(ge=0, le=700)
    fbs: Literal[0, 1]
    restecg: Literal[0, 1, 2]
    thalach: int = Field(ge=50, le=250)
    exang: Literal[0, 1]
    oldpeak: float = Field(ge=0, le=10)
    slope: Literal[1, 2, 3]
    ca: Literal[0, 1, 2, 3]
    thal: Literal[3, 6, 7]


class AssessmentResponse(BaseModel):
    success: bool
    message: str
    prediction: int
    prediction_label: str
    class_1_probability: float | None
    model_connected: bool
    model_name: str
    feature_count: int
    processed_feature_count: int
    disclaimer: str


@app.get("/")
def read_root() -> dict[str, str]:
    return {"message": "Welcome to the Heart Disease AI API"}


@app.get("/health")
def health_check() -> dict[str, object]:
    return {
        "status": "ok" if model is not None and preprocessor is not None else "degraded",
        "model_connected": model is not None and preprocessor is not None,
    }


@app.post("/assessment", response_model=AssessmentResponse)
def submit_assessment(data: ClinicalAssessment) -> AssessmentResponse:
    if model is None or preprocessor is None:
        raise HTTPException(
            status_code=503,
            detail=model_load_error or "ML model artifacts are unavailable.",
        )

    try:
        input_data = pd.DataFrame(
            [[getattr(data, feature) for feature in FEATURE_NAMES]],
            columns=FEATURE_NAMES,
        )
        processed_data = preprocessor.transform(input_data)
        processed_feature_count = processed_data.shape[1]
        if processed_feature_count != EXPECTED_PROCESSED_FEATURE_COUNT:
            raise ValueError("preprocessing produced an incompatible feature count")
        if hasattr(processed_data, "toarray"):
            processed_data = processed_data.toarray()
        processed_data = pd.DataFrame(
            processed_data,
            columns=preprocessor.get_feature_names_out(),
        )

        prediction = int(model.predict(processed_data)[0])
        probability = None
        if hasattr(model, "predict_proba") and hasattr(model, "classes_"):
            probabilities = model.predict_proba(processed_data)[0]
            classes = list(model.classes_)
            if 1 in classes:
                probability = float(probabilities[classes.index(1)])

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
            model_name=type(model).__name__,
            feature_count=len(FEATURE_NAMES),
            processed_feature_count=processed_feature_count,
            disclaimer=(
                "Academic research prototype only. This output is not a "
                "medical diagnosis or a clinically calibrated risk."
            ),
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception("Unexpected inference failure")
        raise HTTPException(
            status_code=500,
            detail="Prediction could not be completed.",
        )

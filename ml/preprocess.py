"""Preprocessing pipeline for Heart Disease AI clinical data analytics.

This module provides leakage-free preprocessing transformations for the UCI
Cleveland heart disease dataset. Preprocessing pipelines must be embedded
directly inside model cross-validation folds, preventing data leakage between
training and validation sets.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --------------------------------------------------
# 1. SCHEMA DEFINITIONS (UCI Cleveland Dataset)
# --------------------------------------------------

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal", "num"
]

FEATURE_NAMES = COLUMNS[:-1]

NUMERICAL_FEATURES = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak",
]

CATEGORICAL_FEATURES = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal",
]

EXPECTED_CATEGORIES = {
    "sex": {0, 1},
    "cp": {1, 2, 3, 4},
    "fbs": {0, 1},
    "restecg": {0, 1, 2},
    "exang": {0, 1},
    "slope": {1, 2, 3},
    "ca": {0, 1, 2, 3},
    "thal": {3, 6, 7},
}

EXPECTED_ROWS = 303
EXPECTED_COLUMNS = 14
EXPECTED_PROCESSED_FEATURE_COUNT = 28
RANDOM_SEED = 42

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "datasets" / "processed.cleveland.data"
REPORT_DIR = BASE_DIR / "reports"
MODEL_DIR = BASE_DIR / "models"


# --------------------------------------------------
# 2. LEAKAGE-FREE PIPELINE FACTORIES
# --------------------------------------------------

def build_numeric_pipeline() -> Pipeline:
    """Construct the numeric feature preprocessing pipeline."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def build_categorical_pipeline() -> Pipeline:
    """Construct the categorical feature preprocessing pipeline."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )


def build_preprocessor() -> ColumnTransformer:
    """Construct the combined column transformer for clinical features.

    Numerical features (5):
      - Median imputation followed by standard scaling.
    Categorical features (8):
      - Most-frequent imputation followed by one-hot encoding.
    Total transformed output: 28 features.
    """
    return ColumnTransformer(
        transformers=[
            ("numeric", build_numeric_pipeline(), NUMERICAL_FEATURES),
            ("categorical", build_categorical_pipeline(), CATEGORICAL_FEATURES),
        ]
    )


def build_pipeline(classifier: Any) -> Pipeline:
    """Wrap a classifier into an end-to-end leakage-free pipeline.

    The preprocessing pipeline is fitted strictly within each cross-validation
    training fold, ensuring validation folds are never leaked into imputation
    medians or scaling parameters.
    """
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", classifier),
        ]
    )


# --------------------------------------------------
# 3. DATASET LOADING AND VALIDATION
# --------------------------------------------------

def load_cleveland_dataset(
    dataset_path: Path | str = DATASET_PATH,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load and validate the raw UCI Cleveland heart disease dataset.

    Returns:
        tuple[pd.DataFrame, pd.Series]: (X, y) where X contains the 13 raw
        clinical features and y contains binary disease labels (0 vs 1).
    """
    path = Path(dataset_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset file not found at: {path}. "
            "Download processed.cleveland.data to ml/datasets/ or train in Google Colab."
        )

    df = pd.read_csv(
        path,
        header=None,
        names=COLUMNS,
        na_values="?",
        skipinitialspace=True,
    )

    if df.shape != (EXPECTED_ROWS, EXPECTED_COLUMNS):
        raise ValueError(
            f"Expected dataset shape ({EXPECTED_ROWS}, {EXPECTED_COLUMNS}), "
            f"observed {df.shape}."
        )

    # Binary classification mapping:
    # 0 = no disease label
    # 1, 2, 3, 4 = disease label present
    X = df[FEATURE_NAMES].copy()
    y = (df["num"] > 0).astype(int)

    return X, y


def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    random_state: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Perform a stratified train/test split on raw, unprocessed features."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )


# --------------------------------------------------
# 4. DRY-RUN INSPECTION (ZERO LOCAL TRAINING)
# --------------------------------------------------

if __name__ == "__main__":
    print("\n==================================================")
    print(" HEART DISEASE AI - PREPROCESSING SPECIFICATION   ")
    print("==================================================")
    print(f"Features (13): {FEATURE_NAMES}")
    print(f"Numerical (5): {NUMERICAL_FEATURES}")
    print(f"Categorical (8): {CATEGORICAL_FEATURES}")
    print(f"Expected transformed dimension: {EXPECTED_PROCESSED_FEATURE_COUNT}")

    preprocessor = build_preprocessor()
    print(f"Constructed preprocessor: {type(preprocessor).__name__}")
    print("Architecture: Encapsulated ColumnTransformer for fold-level cross-validation.")

    if DATASET_PATH.is_file():
        print(f"\nLocal dataset detected at: {DATASET_PATH}")
        X, y = load_cleveland_dataset(DATASET_PATH)
        print(f"Loaded records: {len(X)}")
        print("Target distribution:")
        print(y.value_counts().sort_index())
    else:
        print(f"\nLocal raw dataset not found at: {DATASET_PATH}")
        print("NOTE: Local training is disabled in Phase 2 to preserve project baselines.")
        print("Canonical training must be executed in Google Colab using:")
        print("  ml/notebooks/heart_disease_cloud_training.ipynb")

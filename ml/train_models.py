"""Model training and evaluation pipeline for Heart Disease AI.

This module defines the leakage-free model training, cross-validation,
evaluation, and export methodology for the UCI Cleveland dataset.

Methodological Highlights:
  - 5 Candidate Algorithms: Logistic Regression, Decision Tree, Random Forest,
    Support Vector Machine, and K-Nearest Neighbors.
  - Leakage-Free Preprocessing: Imputation and scaling occur inside each CV fold.
  - Model Selection: Mean cross-validation ROC-AUC across 5 stratified folds.
  - Single Untouched Test Evaluation: Metrics computed once on held-out test data.
  - Unified Pipeline Export: Fitted ColumnTransformer + Classifier saved together.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

try:
    from ml.preprocess import (
        BASE_DIR,
        CATEGORICAL_FEATURES,
        DATASET_PATH,
        EXPECTED_PROCESSED_FEATURE_COUNT,
        FEATURE_NAMES,
        MODEL_DIR,
        NUMERICAL_FEATURES,
        RANDOM_SEED,
        REPORT_DIR,
        build_pipeline,
        build_preprocessor,
        load_cleveland_dataset,
        split_dataset,
    )
except ImportError:
    from preprocess import (  # type: ignore
        BASE_DIR,
        CATEGORICAL_FEATURES,
        DATASET_PATH,
        EXPECTED_PROCESSED_FEATURE_COUNT,
        FEATURE_NAMES,
        MODEL_DIR,
        NUMERICAL_FEATURES,
        RANDOM_SEED,
        REPORT_DIR,
        build_pipeline,
        build_preprocessor,
        load_cleveland_dataset,
        split_dataset,
    )


# --------------------------------------------------
# 1. CANDIDATE MODEL FACTORIES
# --------------------------------------------------

def get_candidate_models(random_state: int = RANDOM_SEED) -> dict[str, Any]:
    """Return dictionary of 5 benchmark classifiers for experimental comparison.

    Includes the primary clinical baselines and benchmark models inspired by
    reference cardiovascular research (Logistic Regression, Decision Tree,
    Random Forest, Support Vector Machine, and K-Nearest Neighbors).
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=random_state,
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=random_state,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "Support Vector Machine": SVC(
            probability=True,
            class_weight="balanced",
            random_state=random_state,
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=7,
        ),
    }


# --------------------------------------------------
# 2. LEAKAGE-FREE CROSS-VALIDATION
# --------------------------------------------------

def evaluate_models_cross_validation(
    models: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = 5,
    random_state: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, dict[str, Pipeline]]:
    """Run leakage-free Stratified 5-Fold Cross-Validation.

    Each fold independently fits its preprocessing steps (imputation, scaling,
    and encoding) using only that fold's training data.
    """
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    cv_rows = []
    cv_pipelines: dict[str, Pipeline] = {}

    for name, classifier in models.items():
        pipeline = build_pipeline(clone(classifier))
        cv_pipelines[name] = pipeline

        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            return_train_score=False,
            n_jobs=-1,
        )

        row: dict[str, Any] = {"model": name}
        for metric in scoring:
            values = scores[f"test_{metric}"]
            row[f"{metric}_folds"] = values.tolist()
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std())

        cv_rows.append(row)

    cv_df = pd.DataFrame(cv_rows).sort_values("roc_auc_mean", ascending=False).reset_index(drop=True)
    return cv_df, cv_pipelines


# --------------------------------------------------
# 3. FINAL HELD-OUT TEST EVALUATION
# --------------------------------------------------

def evaluate_test_set(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Any]:
    """Evaluate fitted pipeline on the untouched held-out test set once."""
    y_pred = pipeline.predict(X_test)
    classes = [int(v) for v in getattr(pipeline.named_steps["classifier"], "classes_", [0, 1])]

    y_probability = None
    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(X_test)
        y_probability = probabilities[:, classes.index(1)]

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall_sensitivity": float(recall_score(y_test, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_probability)) if y_probability is not None else None,
        "brier_score": float(brier_score_loss(y_test, y_probability)) if y_probability is not None else None,
        "test_observations": int(len(y_test)),
        "class_support": {str(k): int(v) for k, v in y_test.value_counts().sort_index().items()},
        "confusion_matrix": cm.tolist(),
    }

    return metrics


# --------------------------------------------------
# 4. EXPORT PIPELINE AND COMPREHENSIVE METADATA
# --------------------------------------------------

def build_model_metadata(
    selected_model_name: str,
    test_metrics: dict[str, Any],
    final_pipeline: Pipeline,
) -> dict[str, Any]:
    """Construct model metadata strictly ensuring integer classes [0, 1]."""
    classifier = final_pipeline.named_steps["classifier"]
    preprocessor = final_pipeline.named_steps["preprocessor"]

    classes = [int(value) for value in getattr(classifier, "classes_", [0, 1])]
    processed_count = len(preprocessor.get_feature_names_out())

    return {
        "project_name": "AI-Based Early Prediction of Heart Disease Using Clinical Data Analytics",
        "dataset_source": "UCI Heart Disease processed Cleveland dataset",
        "dataset_filename": "processed.cleveland.data",
        "feature_names": FEATURE_NAMES,
        "target_column": "num",
        "binary_target_mapping": {"0": 0, "1,2,3,4": 1},
        "numerical_features": NUMERICAL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "imputation": {"numeric": "median", "categorical": "most_frequent"},
        "scaling": "StandardScaler",
        "encoding": 'OneHotEncoder(handle_unknown="ignore")',
        "train_test_split": {"test_size": 0.2, "stratify": True, "random_state": RANDOM_SEED},
        "cross_validation": {
            "type": "StratifiedKFold",
            "n_splits": 5,
            "shuffle": True,
            "random_state": RANDOM_SEED,
            "selection_metric": "mean ROC-AUC",
        },
        "selected_model_name": selected_model_name,
        "final_test_metrics": test_metrics,
        "python_version": "3.13.18",
        "scikit_learn_version": "1.6.1",
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "joblib_version": joblib.__version__,
        "training_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_classes": classes,
        "processed_feature_count": processed_count,
        "probability_warning": (
            "Class-1 probability is a model estimate for the dataset label, "
            "not a calibrated clinical risk."
        ),
        "limitation": (
            "Academic research prototype; not clinically validated and not a "
            "medical diagnostic system."
        ),
    }


# --------------------------------------------------
# 5. DRY-RUN INSPECTION (ZERO LOCAL TRAINING)
# --------------------------------------------------

if __name__ == "__main__":
    print("\n==================================================")
    print(" HEART DISEASE AI - MODEL TRAINING SPECIFICATION  ")
    print("==================================================")
    models = get_candidate_models()
    print(f"Candidate Models ({len(models)}):")
    for idx, (name, clf) in enumerate(models.items(), start=1):
        print(f"  {idx}. {name} ({type(clf).__name__})")

    print("\nCross-Validation Strategy: 5-Fold StratifiedKFold (shuffle=True, random_state=42)")
    print("Primary Selection Metric: Mean Cross-Validation ROC-AUC")
    print("Held-out Test Evaluation: 20% untouched stratified test partition")

    print("\nNOTE: Local model training is disabled in Phase 2 to prevent unauthorized runs.")
    print("The canonical training workflow is implemented in:")
    print("  ml/notebooks/heart_disease_cloud_training.ipynb")
    print("To train, open the notebook in Google Colab, upload processed.cleveland.data, and run all cells.")

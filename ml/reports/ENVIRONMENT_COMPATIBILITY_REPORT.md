# Environment Compatibility Report

**Generated:** 2026-11-12T01:40:00Z  
**Project:** AI-Based Early Prediction of Heart Disease  
**Branch:** `feature/cardioinsight-enhancement`  
**Workspace:** `C:\Users\SAKRIYA NAIK\OneDrive\Documents\My Profile\AIML\heart-disease-ai`

---

## 1. Executive Summary

This report documents the actual detected runtime environments across the developer host machine, backend service requirements, and the cloud training runtime recorded in `model_metadata.json`.

| Domain | Runtime / Environment | Scikit-Learn | NumPy | Pandas | Status |
|---|---|---|---|---|---|
| **Host Python** | Python 3.15.0 (`C:\Python315\python.exe`) | *Not installed* | *Not installed* | *Not installed* | **BLOCKED** (Bare Python, no packages) |
| **Backend Target** | `backend/requirements.txt` (156 pinned pkgs) | `1.9.1` | `2.5.4` | `3.1.1` | **MISMATCH** (Incompatible with cloud export) |
| **Cloud Training** | Google Colab (Linux, Python 3.13.18) | `1.6.1` | `2.1.3` | `2.2.3` | **BASELINE** (Generated existing cloud release) |
| **Local Model Files** | `ml/models/*.joblib` | N/A | N/A | N/A | **MISSING** (Gitignored, not on disk) |
| **Frontend Node** | Node.js v24.21.0 / npm 11.19.0 | N/A | N/A | N/A | **UNINSTALLED** (`node_modules` missing) |

---

## 2. Detected Host Python Environment

- **Python Version:** `3.15.0 (tags/v3.15.0:d220409, Oct 9 2026, 11:55:56) [MSC v.1951 64 bit (AMD64)]`
- **Python Executable:** `C:\Python315\python.exe`
- **Active Virtual Environment:** `None` (`backend/.venv` is absent).
- **Installed Packages:**
  - `pip`: `26.2.1`
  - All core dependencies (`scikit-learn`, `numpy`, `scipy`, `pandas`, `joblib`, `fastapi`, `pydantic`, `uvicorn`) are **NOT INSTALLED** globally.

---

## 3. Backend Pinned Requirements vs. Actual Installation

Analysis of `backend/requirements.txt` (156 total pinned packages):

| Package | Pinned Version | Installed Version | Discrepancy / Risk |
|---|---|---|---|
| `scikit-learn` | `1.9.1` | *None* | Version `1.9.1` conflicts with Scikit-learn `1.6.1` used in cloud training. |
| `numpy` | `2.5.4` | *None* | Major NumPy 2.x ABI. |
| `scipy` | `1.18.1` | *None* | Numerical backend. |
| `pandas` | `3.1.1` | *None* | DataFrame manipulation. |
| `joblib` | `1.6.0` | *None* | Serialization backend. |
| `fastapi` | `0.162.0` | *None* | Web framework. |
| `pydantic` | `2.14.2` | *None* | Schema validation. |
| `uvicorn` | `0.54.0` | *None* | ASGI server. |

---

## 4. Cloud Release Compatibility Findings

Inspection of `ml/cloud_release/heart_disease_cloud_release.zip` and `model_metadata.json`:

1. **Scikit-Learn Version Disparity:**
   The cloud training notebook executed under Scikit-learn `1.6.1`. Serializing a `ColumnTransformer` under 1.6.1 relies on internal classes (e.g. `_RemainderColsList`) that trigger `AttributeError` upon deserialization in mismatched environments.
2. **Metadata Class Typing:**
   The metadata export serialized NumPy class labels using `default=str`, resulting in `"model_classes": ["0", "1"]` instead of numeric `[0, 1]`.
3. **Artifact Composition:**
   The cloud package contains a single combined pipeline (`heart_disease_pipeline.joblib`), whereas `backend/app/main.py` expects separate files `ml/models/best_model.joblib` and `ml/models/preprocessor.joblib`.

---

## 5. Local Artifact Findings

- Expected files:
  - `ml/models/best_model.joblib`
  - `ml/models/preprocessor.joblib`
- **Result:** **BLOCKED**
  Neither file exists on disk. Both are excluded by `.gitignore` line 19 (`*.joblib`). Consequently, backend startup degrades and `POST /assessment` returns HTTP 503.

---

## 6. Recommendations & Action Plan

1. **Explicitly Pin Scikit-Learn 1.6.1 for Training and Serving:**
   - Establish `ml/requirements-training.txt` with `scikit-learn==1.6.1` to align cloud training and backend serving environments.
   - Update `ml/notebooks/heart_disease_cloud_training.ipynb` with `%pip install --quiet "scikit-learn==1.6.1"` before importing Scikit-learn.
2. **Correct Notebook Metadata Serialization:**
   - In the Colab notebook, explicitly cast classes to integers: `model_classes = [int(v) for v in classes]`.
3. **Design Adaptive Backend Loader:**
   - Modify backend model loading to accept either a single combined `Pipeline` or separate model and preprocessor joblib files.
4. **Isolate Test Harness:**
   - Refactor `tests/test_cloud_release_verifier.py` to use synthetic in-memory fixtures so verification logic does not fail when local model files are absent.

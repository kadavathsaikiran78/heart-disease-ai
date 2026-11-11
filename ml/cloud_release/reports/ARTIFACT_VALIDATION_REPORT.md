# Cloud Artifact Validation Report

- Archive: `D:\Projects\Heart-Disease-AI\ml\cloud_release\heart_disease_cloud_release.zip`
- SHA-256: `75f9b3f5ea3464563b413adda6c889f93d1c7c26739a15065a2b659921907987`

## Verification outcomes

| Check | Status | Details |
|---|---|---|
| `archive` | **PASSED** | ZIP is readable |
| `required_files` | **PASSED** | all required files are present |
| `size:classification_report.txt` | **PASSED** | 350 bytes |
| `size:cross_validation_results.csv` | **PASSED** | 2209 bytes |
| `size:model_metadata.json` | **PASSED** | 2049 bytes |
| `size:roc_curve.png` | **PASSED** | 60484 bytes |
| `size:confusion_matrix.png` | **PASSED** | 42813 bytes |
| `size:model_comparison.png` | **PASSED** | 65201 bytes |
| `size:evaluation_metrics.json` | **PASSED** | 317 bytes |
| `size:heart_disease_pipeline.joblib` | **PASSED** | 4970 bytes |
| `metadata` | **FAILED** | model classes must be [0, 1] |
| `json:evaluation_metrics.json` | **PASSED** | valid JSON |
| `cross_validation_results.csv` | **PASSED** | CSV is readable |
| `classification_report.txt` | **PASSED** | text report is readable |
| `confusion_matrix.png` | **PASSED** | PNG is readable |
| `roc_curve.png` | **PASSED** | PNG is readable |
| `model_comparison.png` | **PASSED** | PNG is readable |
| `pipeline` | **FAILED** | pipeline deserialization failed, likely due to a scikit-learn serialization-version mismatch: Can't get attribute '_RemainderColsList' on <module 'sklearn.compose._column_transformer' from 'D:\\Projects\\Heart-Disease-AI\\backend\\.venv\\Lib\\site-packages\\sklearn\\compose\\_column_transformer.py'> |

## Safety

- The original archive is preserved in `ml/cloud_release/`.
- Existing `ml/models/best_model.joblib` and `preprocessor.joblib` were not replaced.
- No model training was executed locally.
- Cloud training success is not claimed beyond the supplied archive contents.

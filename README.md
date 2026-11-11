# AI-Based Early Prediction of Heart Disease

Academic clinical decision-support research prototype using the UCI processed
Cleveland heart-disease dataset. The application classifies the dataset label
as class 0 (no disease label) or class 1 (disease-present label). It is not a
clinically validated diagnostic system.

## Architecture

- `frontend/`: Next.js App Router, React, TypeScript, and Tailwind UI.
- `backend/`: FastAPI API with Pydantic validation and CORS for local frontend
  origins.
- `ml/`: dataset, preprocessing/training scripts, saved model artifacts, and
  the Google Colab training notebook.
- `tests/`: backend artifact and API verification tests.

The assessment form sends the 13 original Cleveland features to
`POST /assessment`. The backend constructs the feature-ordered DataFrame,
applies `ml/models/preprocessor.joblib` (13 inputs to 28 transformed
features), and runs `ml/models/best_model.joblib`.

## Dataset and model workflow

The 13 features are `age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`,
`thalach`, `exang`, `oldpeak`, `slope`, `ca`, and `thal`. The target `num` is
mapped to binary labels: original `0` becomes class 0, and original `1`–`4`
become class 1. Missing clinical values are handled by the fitted
preprocessing pipeline rather than replaced with zero.

The saved local artifacts are loaded once at backend startup from project
relative paths. Startup validation checks the feature schema, 28 transformed
features, model input size, and classes `[0, 1]`. The Colab notebook at
`ml/notebooks/heart_disease_cloud_training.ipynb` provides leakage-free
cross-validation and exports a separate combined pipeline package; it does not
overwrite the local artifacts.

## Backend setup

From the repository root in PowerShell:

```powershell
.\backend\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

API endpoints:

- `GET /`: API welcome message.
- `GET /health`: API and model availability (`ok` or `degraded`).
- `POST /assessment`: validates all 13 features and returns the model
  prediction, class-1 model-estimated probability, model name, and feature
  counts.
- `/docs`: interactive OpenAPI documentation.

Invalid input returns HTTP 422, unavailable artifacts return HTTP 503, and
unexpected inference failures return HTTP 500 without exposing stack traces
or local paths.

## Frontend setup

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000/assessment`. The local API URL is configured in
`frontend/.env.local` as `NEXT_PUBLIC_API_BASE_URL`. Restart Next.js after
changing that file.

## Testing

Backend tests and artifact checks:

```powershell
.\backend\.venv\Scripts\python.exe tests\test_backend_api.py
```

Frontend lint, type-check, and production build:

```powershell
cd frontend
npm run lint
npx tsc --noEmit
npm run build
```

## Limitations

The dataset is small and the output is a dataset classification label. The
probability is a model estimate, not a calibrated clinical risk. Results do
not establish causation, clinical validity, generalisation, or a medical
diagnosis. Do not upload real patient records or use this prototype for
clinical decisions.
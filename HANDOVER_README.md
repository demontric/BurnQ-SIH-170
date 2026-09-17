# Model Handover — Anomaly Detection + 168h Drift Prediction

## What's in this package

| File | Format | Purpose |
|---|---|---|
| `models/anomaly_model.joblib` | sklearn (Python) | Module 1 — Isolation Forest anomaly detector + fitted StandardScaler, bundled with metadata |
| `models/drift_model.joblib` | sklearn (Python) | Module 2 — RandomForest regressor predicting Value_168h |
| `exports/anomaly_model.onnx` | ONNX | Module 1, framework/language-agnostic |
| `exports/drift_model.onnx` | ONNX | Module 2, framework/language-agnostic |
| `evaluation/anomaly_model_report.txt` | text | Precision/Recall/F1/ROC-AUC + per-class recall |
| `evaluation/drift_model_report.txt` | text | RMSE/MAE/MAPE/R² for all 3 candidate regressors |
| `preprocessing/features.py` | Python | Feature engineering — **must be re-run identically at inference time** |

## Headline results

**Module 1 — Anomaly Detection (Isolation Forest)**
- Precision 0.764 / Recall 0.797 / F1 0.780 / ROC-AUC 0.983
- Catches **100%** of static-limit failures (class 1) — expected, since these already fail the datasheet check
- Catches **69%** of hidden trend anomalies (class 2) that pass the static limit — this is the value-add over threshold-only screening

**Module 2 — 168h Drift Prediction (Random Forest)**
- RMSE 0.80 µA / MAE 0.34 µA / MAPE 2.62% / R² 0.991, using only 0h/24h/96h inputs
- Beat XGBoost (RMSE 0.94) and LightGBM (RMSE 1.05) on this dataset size

## Feature order (critical — do not reorder)

**Anomaly model** expects the 26 features in `ANOMALY_MODEL_FEATURES` from `features.py`
(uses full 0h–168h history).

**Drift model** expects the 15 features in `DRIFT_MODEL_FEATURES` from `features.py`
(uses only 0h/24h/96h — no leakage from the 168h target).

Always call `engineer_features(raw_df, include_168h=...)` before inference —
never hand raw `Value_*h` columns straight to the models.

## Loading code — joblib (Python)

```python
import joblib
import pandas as pd
from features import engineer_features, ANOMALY_MODEL_FEATURES, DRIFT_MODEL_FEATURES

anomaly_bundle = joblib.load("models/anomaly_model.joblib")
anomaly_model, scaler = anomaly_bundle["model"], anomaly_bundle["scaler"]

drift_bundle = joblib.load("models/drift_model.joblib")
drift_model = drift_bundle["model"]

raw = pd.read_csv("new_components.csv")  # same schema as synthetic_burnin_data.csv
feat = engineer_features(raw, include_168h=True)

# Anomaly score (0-1, higher = more anomalous)
X_anom = scaler.transform(feat[ANOMALY_MODEL_FEATURES].values)
raw_scores = -anomaly_model.score_samples(X_anom)
anomaly_score = (raw_scores - anomaly_bundle["score_min"]) / (
    anomaly_bundle["score_max"] - anomaly_bundle["score_min"]
)
is_anomaly = anomaly_model.predict(X_anom) == -1  # True/False

# Predicted 168h value (only 0h/24h/96h needed — no leakage)
X_drift = feat[DRIFT_MODEL_FEATURES].values
predicted_168h = drift_model.predict(X_drift)
```

## Loading code — ONNX (any language / no sklearn dependency)

```python
import onnxruntime as ort
import numpy as np

anomaly_sess = ort.InferenceSession("exports/anomaly_model.onnx")
drift_sess = ort.InferenceSession("exports/drift_model.onnx")

# X_anom, X_drift must be np.float32 arrays with the exact feature order
# from features.py's ANOMALY_MODEL_FEATURES / DRIFT_MODEL_FEATURES
anomaly_labels = anomaly_sess.run(None, {"input": X_anom.astype(np.float32)})[0]
predicted_168h = drift_sess.run(None, {"input": X_drift.astype(np.float32)})[0]
```

ONNX predictions were verified against the sklearn originals on a 20-row
sample: drift model max absolute difference = 0.000006, anomaly model label
match rate = 100%.

## Known limitations / caveats for the lead

- Trained on 5,500 components (~550/lot across 10 lots) — this is a small
  dataset for an Isolation Forest; recall on the hidden-anomaly class (69%)
  should improve with more labeled examples of class 2.
- `datasheet_limit_uA` is constant (50 µA) across this dataset — the
  margin-based features will need revalidation if future data includes
  multiple limit values per component type.
- No lot-level holdout was used in this split (random 70/30, not by-lot) —
  recommend re-validating with a lot-based split before production sign-off,
  since burn-in variation can correlate within a lot.
- Reliability scoring (Module 3), SHAP explainability (Step 7), and the
  unified `analyze_component()` inference function (Step 8) are not yet
  built — these are the next steps in the pipeline.

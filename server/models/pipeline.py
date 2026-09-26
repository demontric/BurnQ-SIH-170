"""End-to-end burn-in screening: Module A + Module B + explainability."""

import math
import numpy as np
import pandas as pd

from models.outlier_detection import detect_anomalies
from models.drift_predictor import predict_168h

COLUMN_ALIASES = {
    "ComponentID": "part_id", "component_id": "part_id", "PartID": "part_id",
    "Lot": "lot_id", "lot": "lot_id",
    "Value_0h": "value_0h", "Value_24h": "value_24h",
    "Value_96h": "value_96h", "Value_168h": "value_168h",
    "datasheet_limit_uA": "datasheet_limit",
}

REQUIRED_COLUMNS = ["lot_id", "parameter", "value_0h", "value_24h", "value_96h", "value_168h"]

def _to_python(value):
    if value is None or pd.isna(value): return None
    if isinstance(value, (np.bool_,)): return bool(value)
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating, float)):
        return None if math.isnan(value) or math.isinf(value) else float(value)
    if isinstance(value, (pd.Timestamp,)): return str(value)
    return value

def records_to_json(df: pd.DataFrame) -> list:
    return [{k: _to_python(v) for k, v in row.items()} for row in df.to_dict(orient="records")]

def normalize_input(df: pd.DataFrame, datasheet_limit=None) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    out = out.rename(columns={k: v for k, v in COLUMN_ALIASES.items() if k in out.columns})

    if "parameter" not in out.columns: out["parameter"] = "leakage_current"
    if "part_id" not in out.columns: out["part_id"] = [f"C_{i:04d}" for i in range(len(out))]

    missing = [c for c in REQUIRED_COLUMNS if c not in out.columns]
    if missing: raise ValueError(f"Missing required columns: {missing}")

    for col in ["value_0h", "value_24h", "value_96h", "value_168h", "datasheet_limit"]:
        if col in out.columns: out[col] = pd.to_numeric(out[col], errors="coerce")

    # Only fall back to the form/global datasheet_limit if the CSV doesn't
    # already supply its own per-row datasheet_limit_uA column.
    if "datasheet_limit" not in out.columns:
        out["datasheet_limit"] = float(datasheet_limit) if datasheet_limit is not None else 50.0
    else:
        out["datasheet_limit"] = out["datasheet_limit"].fillna(
            float(datasheet_limit) if datasheet_limit is not None else 50.0
        )

    return out

def run_screening(df: pd.DataFrame, datasheet_limit=None, risk_tolerance=50.0) -> dict:
    raw = normalize_input(df, datasheet_limit=datasheet_limit)

    # Risk tolerance (1-100). Lower tolerance = tighter Z-threshold
    z_threshold = 2.0 + (risk_tolerance / 100.0) * 3.0

    # Gate 1: Predict 168h using 0h/24h features
    screened_gate1 = predict_168h(raw, threshold=z_threshold)

    # Gate 2: Detect 96h late-bloomer anomalies
    screened = detect_anomalies(screened_gate1, threshold=z_threshold)

    # Justifications are no longer generated synchronously here; the initial
    # upload returns null so the response is fast. Justifications can be
    # generated later (e.g. on-demand or in a background job).
    screened["justification"] = None
    screened["is_flagged"] = screened["is_anomaly"].astype(bool) | screened["safety_slope_exceeded"].astype(bool)

    actual = pd.to_numeric(screened["value_168h"], errors="coerce")
    predicted = pd.to_numeric(screened["predicted_168h"], errors="coerce")
    valid = actual.notna() & predicted.notna()
    mae = float(np.mean(np.abs(predicted[valid] - actual[valid]))) if valid.any() else None

    payload_cols = [
        "part_id", "lot_id", "parameter", "value_0h", "value_24h", "value_96h", "value_168h",
        "predicted_168h", "robust_z_score", "isolation_forest_score", "is_anomaly",
        "safety_slope_exceeded", "is_flagged", "justification", "datasheet_limit",
        "predicted_drift_rate", "safety_slope"
    ]
    
    data = screened[[c for c in payload_cols if c in screened.columns]].rename(
        columns={"part_id": "ComponentID", "lot_id": "Lot", "value_0h": "Value_0h",
                 "value_24h": "Value_24h", "value_96h": "Value_96h", "value_168h": "Value_168h"}
    )

    records = records_to_json(data)
    return {
        "total_components": len(records),
        "flagged_count": int(sum(1 for row in records if row.get("is_flagged"))),
        "mae": mae,
        "data": records,
    }
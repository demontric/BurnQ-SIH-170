"""End-to-end burn-in screening: Module A + Module B + explainability."""

import math
import numpy as np
import pandas as pd

from models.outlier_detection import detect_anomalies
from models.drift_predictor import predict_168h
from models.safety_slope import calculate_safety_slope
from models.model_registry import normalize_parameter
from models.decision_engine import apply_decision_engine
from models.explainability import apply_explainability

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
    if isinstance(value, list): return [_to_python(v) for v in value]
    return value

def records_to_json(df: pd.DataFrame) -> list:
    return [{k: _to_python(v) for k, v in row.items()} for row in df.to_dict(orient="records")]

def normalize_input(df: pd.DataFrame, datasheet_limit=None) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    out = out.rename(columns={k: v for k, v in COLUMN_ALIASES.items() if k in out.columns})

    if "parameter" not in out.columns:
        out["parameter"] = "leakage_current"
    out["parameter"] = out["parameter"].map(normalize_parameter)
    if "part_id" not in out.columns: out["part_id"] = [f"C_{i:04d}" for i in range(len(out))]

    missing = [c for c in REQUIRED_COLUMNS if c not in out.columns]
    if missing: raise ValueError(f"Missing required columns: {missing}")

    for col in ["value_0h", "value_24h", "value_96h", "value_168h", "datasheet_limit"]:
        if col in out.columns: out[col] = pd.to_numeric(out[col], errors="coerce")

    if "datasheet_limit" not in out.columns:
        out["datasheet_limit"] = float(datasheet_limit) if datasheet_limit is not None else 50.0
    else:
        out["datasheet_limit"] = out["datasheet_limit"].fillna(
            float(datasheet_limit) if datasheet_limit is not None else 50.0
        )

    return out

def run_screening(df: pd.DataFrame, datasheet_limit=None, risk_tolerance=50.0) -> dict:
    raw = normalize_input(df, datasheet_limit=datasheet_limit)

    z_threshold = 2.0 + (risk_tolerance / 100.0) * 3.0

    # 1) Prediction — Module B uses only 0h/24h 
    screened_gate1 = predict_168h(raw)

    # 2) Evaluation
    actual = pd.to_numeric(screened_gate1["value_168h"], errors="coerce")
    predicted = pd.to_numeric(screened_gate1["predicted_168h"], errors="coerce")
    err = (predicted - actual).abs()
    valid = actual.notna() & predicted.notna()
    mae = float(err[valid].mean()) if valid.any() else None
    mae_by_parameter = {}
    if "parameter" in screened_gate1.columns and valid.any():
        for param, g in screened_gate1.loc[valid].groupby("parameter", sort=False):
            mae_by_parameter[str(param)] = float(err.loc[g.index].mean())

    # 3) Calculate lot-relative safety slopes (Phase 6 Separation)
    screened_gate2 = calculate_safety_slope(screened_gate1, threshold=z_threshold)

    # 4) Module A — outlier detection
    screened_gate3 = detect_anomalies(screened_gate2, threshold=z_threshold, risk_tolerance=risk_tolerance)

    # 5) Phase 5: Decision Engine Processing
    screened_gate4 = apply_decision_engine(screened_gate3)

    # 6) Phase 8: Deterministic Explainability
    screened = apply_explainability(screened_gate4)

    payload_cols = [
        "part_id", "lot_id", "parameter", "value_0h", "value_24h", "value_96h", "value_168h",
        "predicted_168h", "robust_z_score", "isolation_forest_score", "is_anomaly",
        "safety_slope_exceeded", "status", "reason_codes", "justification", "is_flagged", 
        "datasheet_limit", "predicted_drift_rate", "safety_slope"
    ]
    
    data = screened[[c for c in payload_cols if c in screened.columns]].rename(
        columns={"part_id": "ComponentID", "lot_id": "Lot", "value_0h": "Value_0h",
                 "value_24h": "Value_24h", "value_96h": "Value_96h", "value_168h": "Value_168h"}
    )

    records = records_to_json(data)
    
    component_count = int(screened["part_id"].nunique()) if "part_id" in screened.columns else len(records)
    return {
        "components": component_count,
        "parametric_records": len(records),
        "total_components": len(records),
        "flagged_count": int(sum(1 for row in records if row.get("is_flagged"))),
        "mae": mae,
        "mae_by_parameter": mae_by_parameter,
        "data": records,
    }
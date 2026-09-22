"""End-to-end burn-in screening: Module A + Module B + explainability."""

import math
import numpy as np
import pandas as pd

from models.outlier_detection import detect_anomalies
from models.drift_predictor import predict_168h
from models.explainability import generate_justification

COLUMN_ALIASES = {
    "ComponentID": "part_id",
    "component_id": "part_id",
    "PartID": "part_id",
    "Lot": "lot_id",
    "lot": "lot_id",
    "Value_0h": "value_0h",
    "Value_24h": "value_24h",
    "Value_96h": "value_96h",
    "Value_168h": "value_168h",
    "datasheet_limit_uA": "datasheet_limit",
}

REQUIRED_COLUMNS = [
    "lot_id",
    "parameter",
    "value_0h",
    "value_24h",
    "value_96h",
    "value_168h",
]


def _to_python(value):
    if value is None:
        return None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if math.isnan(value) or math.isinf(value):
            return None
        return float(value)
    if pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp,)):
        return str(value)
    return value


def records_to_json(df: pd.DataFrame) -> list:
    records = df.to_dict(orient="records")
    cleaned = []
    for row in records:
        cleaned.append({key: _to_python(val) for key, val in row.items()})
    return cleaned


def normalize_input(df: pd.DataFrame, datasheet_limit=None) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    out = out.rename(columns={k: v for k, v in COLUMN_ALIASES.items() if k in out.columns})

    if "parameter" not in out.columns:
        out["parameter"] = "leakage_current"
    if "part_id" not in out.columns:
        out["part_id"] = [f"C_{i:04d}" for i in range(len(out))]

    missing = [c for c in REQUIRED_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for col in ["value_0h", "value_24h", "value_96h", "value_168h", "datasheet_limit"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if datasheet_limit is not None:
        out["datasheet_limit"] = float(datasheet_limit)
    elif "datasheet_limit" not in out.columns:
        out["datasheet_limit"] = 50.0
    else:
        out["datasheet_limit"] = out["datasheet_limit"].fillna(50.0)

    return out


def _lot_stats(df: pd.DataFrame, value_col: str):
    stats = {}
    for (lot, param), group in df.groupby(["lot_id", "parameter"]):
        vals = pd.to_numeric(group[value_col], errors="coerce")
        med = float(vals.median()) if len(vals.dropna()) else 0.0
        mad = float(np.median(np.abs(vals - med))) if len(vals.dropna()) else 1e-6
        if mad == 0:
            mad = 1e-6
        stats[(lot, param)] = (med, mad)
    return stats


def run_screening(df: pd.DataFrame, datasheet_limit=None) -> dict:
    raw = normalize_input(df, datasheet_limit=datasheet_limit)

    module_a = detect_anomalies(raw)
    screened = predict_168h(module_a)

    lot_stats_168h = _lot_stats(screened, "value_168h")
    justifications = []
    for _, row in screened.iterrows():
        med, mad = lot_stats_168h.get((row["lot_id"], row["parameter"]), (0.0, 1e-6))
        justifications.append(generate_justification(row, med, mad))
    screened["justification"] = justifications
    screened["is_flagged"] = (
        screened["is_anomaly"].astype(bool) | screened["safety_slope_exceeded"].astype(bool)
    )

    actual = pd.to_numeric(screened["value_168h"], errors="coerce")
    predicted = pd.to_numeric(screened["predicted_168h"], errors="coerce")
    valid = actual.notna() & predicted.notna()
    mae = float(np.mean(np.abs(predicted[valid] - actual[valid]))) if valid.any() else None

    payload_cols = [
        "part_id",
        "lot_id",
        "parameter",
        "value_0h",
        "value_24h",
        "value_96h",
        "value_168h",
        "predicted_168h",
        "robust_z_score",
        "isolation_forest_score",
        "is_anomaly",
        "safety_slope_exceeded",
        "is_flagged",
        "justification",
        "datasheet_limit",
        "predicted_drift_rate",
        "safety_slope",
    ]
    extra = [c for c in payload_cols if c in screened.columns]
    data = screened[extra].rename(
        columns={
            "part_id": "ComponentID",
            "lot_id": "Lot",
            "value_0h": "Value_0h",
            "value_24h": "Value_24h",
            "value_96h": "Value_96h",
            "value_168h": "Value_168h",
        }
    )

    records = records_to_json(data)
    flagged_count = int(sum(1 for row in records if row.get("is_flagged")))

    return {
        "total_components": len(records),
        "flagged_count": flagged_count,
        "mae": mae,
        "data": records,
    }

"""
End-to-end burn-in screening pipeline.

Pipeline stages:

    Input normalization
        ↓
    Module B — 168h drift prediction
        ↓
    Evaluation / MAE
        ↓
    Safety slope
        ↓
    Module A — anomaly detection
        ↓
    Decision engine
        ↓
    Deterministic explainability
        ↓
    JSON-ready response
"""

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
    """
    Convert NumPy/Pandas/native values into JSON-safe Python values.

    Important:
    Never evaluate pd.isna(array) directly in a boolean expression.
    """

    if value is None:
        return None

    # NumPy arrays must be handled BEFORE pd.isna().
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None

        if value.size == 1:
            return _to_python(
                value.reshape(-1)[0]
            )

        return [
            _to_python(item)
            for item in value.reshape(-1).tolist()
        ]

    if isinstance(value, (list, tuple)):
        return [
            _to_python(item)
            for item in value
        ]

    if isinstance(value, dict):
        return {
            key: _to_python(item)
            for key, item in value.items()
        }

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, (np.floating, float)):
        return (
            None
            if math.isnan(value)
            or math.isinf(value)
            else float(value)
        )

    if isinstance(value, pd.Timestamp):
        return str(value)

    # At this point scalar values are expected.
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False

    # pd.isna can theoretically return an array-like value.
    # Never use that directly as a boolean.
    if isinstance(missing, (np.ndarray, list, tuple)):
        if len(missing) == 0:
            return None
        return value

    if bool(missing):
        return None

    return value


def records_to_json(df: pd.DataFrame) -> list:
    """
    Convert dataframe rows into JSON-safe dictionaries.
    """

    records = []

    for row in df.to_dict(
        orient="records"
    ):
        records.append(
            {
                key: _to_python(value)
                for key, value in row.items()
            }
        )

    return records


def normalize_input(
    df: pd.DataFrame,
    datasheet_limit=None,
) -> pd.DataFrame:
    """
    Normalize incoming CSV data to the pipeline schema.
    """

    out = df.copy()

    out.columns = [
        str(column).strip()
        for column in out.columns
    ]

    out = out.rename(
        columns={
            source: target
            for source, target
            in COLUMN_ALIASES.items()
            if source in out.columns
        }
    )

    if "parameter" not in out.columns:
        out["parameter"] = (
            "leakage_current"
        )

    out["parameter"] = (
        out["parameter"]
        .map(normalize_parameter)
    )

    if "part_id" not in out.columns:
        out["part_id"] = [
            f"C_{index:04d}"
            for index in range(len(out))
        ]

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in out.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    numeric_columns = [
        "value_0h",
        "value_24h",
        "value_96h",
        "value_168h",
        "datasheet_limit",
    ]

    for column in numeric_columns:
        if column in out.columns:
            out[column] = pd.to_numeric(
                out[column],
                errors="coerce",
            )

    if "datasheet_limit" not in out.columns:
        out["datasheet_limit"] = (
            float(datasheet_limit)
            if datasheet_limit is not None
            else 50.0
        )
    else:
        default_limit = (
            float(datasheet_limit)
            if datasheet_limit is not None
            else 50.0
        )

        out["datasheet_limit"] = (
            out["datasheet_limit"]
            .fillna(default_limit)
        )

    return out


def run_screening(
    df: pd.DataFrame,
    datasheet_limit=None,
    risk_tolerance=50.0,
) -> dict:
    """
    Execute the complete burn-in screening pipeline.
    """

    # ---------------------------------------------------------
    # 0. Normalize input
    # ---------------------------------------------------------

    raw = normalize_input(
        df,
        datasheet_limit=datasheet_limit,
    )

    # ---------------------------------------------------------
    # Risk tolerance -> robust-z threshold
    # ---------------------------------------------------------

    z_threshold = (
        2.0
        + (risk_tolerance / 100.0) * 3.0
    )

    # ---------------------------------------------------------
    # 1. Module B — 168h prediction
    #
    # IMPORTANT:
    # Only 0h/24h features enter the predictor.
    # ---------------------------------------------------------

    screened_gate1 = predict_168h(
        raw
    )

    # ---------------------------------------------------------
    # 2. Prediction evaluation
    #
    # Actual 168h is used ONLY for evaluation.
    # ---------------------------------------------------------

    actual = pd.to_numeric(
        screened_gate1["value_168h"],
        errors="coerce",
    )

    predicted = pd.to_numeric(
        screened_gate1["predicted_168h"],
        errors="coerce",
    )

    error = (
        predicted - actual
    ).abs()

    valid = (
        actual.notna()
        & predicted.notna()
    )

    if valid.any():
        mae = float(
            error[valid].mean()
        )
    else:
        mae = None

    mae_by_parameter = {}

    if (
        "parameter" in screened_gate1.columns
        and valid.any()
    ):
        valid_rows = (
            screened_gate1.loc[valid]
        )

        for parameter, group in (
            valid_rows.groupby(
                "parameter",
                sort=False,
            )
        ):
            mae_by_parameter[
                str(parameter)
            ] = float(
                error.loc[
                    group.index
                ].mean()
            )

    # ---------------------------------------------------------
    # 3. Safety slope
    # ---------------------------------------------------------

    screened_gate2 = (
        calculate_safety_slope(
            screened_gate1,
            threshold=z_threshold,
        )
    )

    # ---------------------------------------------------------
    # 4. Module A — anomaly detection
    #
    # 0h / 24h / 96h only.
    # ---------------------------------------------------------

    screened_gate3 = (
        detect_anomalies(
            screened_gate2,
            threshold=z_threshold,
            risk_tolerance=risk_tolerance,
        )
    )

    # ---------------------------------------------------------
    # 5. Decision engine
    # ---------------------------------------------------------

    screened_gate4 = (
        apply_decision_engine(
            screened_gate3
        )
    )

    # ---------------------------------------------------------
    # 6. Deterministic explainability
    # ---------------------------------------------------------

    screened = (
        apply_explainability(
            screened_gate4
        )
    )

    # ---------------------------------------------------------
    # Response columns
    # ---------------------------------------------------------

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

        "status",
        "reason_codes",
        "justification",
        "is_flagged",

        "datasheet_limit",

        "predicted_drift_rate",
        "safety_slope",
    ]

    available_payload_cols = [
        column
        for column in payload_cols
        if column in screened.columns
    ]

    data = (
        screened[
            available_payload_cols
        ]
        .rename(
            columns={
                "part_id": "ComponentID",
                "lot_id": "Lot",

                "value_0h": "Value_0h",
                "value_24h": "Value_24h",
                "value_96h": "Value_96h",
                "value_168h": "Value_168h",
            }
        )
    )

    # ---------------------------------------------------------
    # JSON serialization
    # ---------------------------------------------------------

    records = records_to_json(
        data
    )

    # ---------------------------------------------------------
    # Component count
    # ---------------------------------------------------------

    if "part_id" in screened.columns:
        component_count = int(
            screened[
                "part_id"
            ].nunique()
        )
    else:
        component_count = len(
            records
        )

    # ---------------------------------------------------------
    # Flagged count
    # ---------------------------------------------------------

    flagged_count = sum(
        1
        for row in records
        if bool(
            row.get(
                "is_flagged",
                False,
            )
        )
    )

    # ---------------------------------------------------------
    # Final response
    # ---------------------------------------------------------

    return {
        "components": component_count,
        "parametric_records": len(records),
        "flagged_count": int(
            flagged_count
        ),
        "mae": mae,
        "mae_by_parameter": (
            mae_by_parameter
        ),
        "data": records,
    }
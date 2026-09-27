import os

import numpy as np
import onnxruntime as ort

from preprocessing.features import (
    engineer_features,
    ANOMALY_MODEL_FEATURES,
)

from models.cost_sensitive_training import (
    cost_sensitive_isolation_forest_threshold,
)


_SESSION = None


def _session():
    global _SESSION

    if _SESSION is None:
        model_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "exports",
            "anomaly_model.onnx",
        )

        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Anomaly ONNX model not found: {model_path}"
            )

        _SESSION = ort.InferenceSession(
            model_path,
            providers=[
                "CPUExecutionProvider"
            ],
        )

    return _SESSION


def _static_robust_z(series):
    """
    Compute a static robust z-score using the full group.

    Formula:
        0.6745 * (x - median) / MAD
    """

    vals = series.astype(float)

    median = float(
        vals.median()
    )

    deviations = np.abs(
        vals - median
    )

    mad = float(
        np.nanmedian(
            deviations.to_numpy(
                dtype=float
            )
        )
    )

    if not np.isfinite(mad) or mad < 1e-6:
        mad = 1e-6

    result = (
        0.6745
        * (vals - median)
        / mad
    )

    return result


def _validate_vector(
    values,
    expected_length,
    name,
):
    """
    Convert an ONNX output to a 1-D numeric vector
    with exactly one value per input row.
    """

    arr = np.asarray(values)

    if arr.size == 0:
        raise ValueError(
            f"{name} is empty."
        )

    if arr.ndim == 1:
        vector = arr

    elif arr.ndim == 2 and arr.shape[1] == 1:
        vector = arr.reshape(-1)

    else:
        raise ValueError(
            f"{name} has unexpected shape "
            f"{arr.shape}. "
            f"Expected (N,) or (N,1)."
        )

    if len(vector) != expected_length:
        raise ValueError(
            f"{name} length mismatch: "
            f"expected {expected_length}, "
            f"got {len(vector)}."
        )

    return vector


def detect_anomalies(
    df,
    lot_col="lot_id",
    param_col="parameter",
    tier_col="datasheet_limit",
    value_cols=None,
    threshold=3.5,
    risk_tolerance=50.0,
):
    """
    Module A anomaly detection.

    Uses:
      - ONNX anomaly model
      - lot + parameter robust statistics
      - datasheet absolute limit

    168h is never used as an ML feature.
    """

    if value_cols is None:
        value_cols = [
            "value_0h",
            "value_24h",
            "value_96h",
            "value_168h",
        ]

    out_df = df.copy()

    rename_mapping = {
        "lot_id": "Lot"
    }

    for col in value_cols:
        if col in out_df.columns:
            rename_mapping[col] = col.replace(
                "value_",
                "Value_",
            )

    if (
        "datasheet_limit" in out_df.columns
        and "datasheet_limit_uA" not in out_df.columns
    ):
        rename_mapping[
            "datasheet_limit"
        ] = "datasheet_limit_uA"

    feat_df = out_df.rename(
        columns=rename_mapping
    )

    if "datasheet_limit_uA" not in feat_df.columns:
        feat_df["datasheet_limit_uA"] = 50.0

    # Module A must use only 0h / 24h / 96h derived information.
    feat = engineer_features(
        feat_df,
        include_168h=False,
    )

    missing = [
        feature
        for feature in ANOMALY_MODEL_FEATURES
        if feature not in feat.columns
    ]

    if missing:
        raise ValueError(
            f"Module A missing frozen features: {missing}"
        )

    X_anom = feat[
        ANOMALY_MODEL_FEATURES
    ].to_numpy(
        dtype=np.float32,
        copy=True,
    )

    X_anom = np.nan_to_num(
        X_anom,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    sess = _session()

    inputs = sess.get_inputs()

    if not inputs:
        raise ValueError(
            "Anomaly ONNX model has no input tensors."
        )

    input_name = inputs[0].name

    outputs = sess.run(
        None,
        {
            input_name: X_anom
        },
    )

    if not outputs:
        raise ValueError(
            "Anomaly ONNX model returned no outputs."
        )

    expected_length = len(out_df)

    # First output is expected to be the anomaly label.
    preds = _validate_vector(
        outputs[0],
        expected_length,
        "Anomaly prediction",
    )

    out_df["isolation_forest_score"] = 0.0

    # If the model exposes a score as its second output,
    # use that score for cost-sensitive thresholding.
    if len(outputs) > 1:
        scores = _validate_vector(
            outputs[1],
            expected_length,
            "Anomaly score",
        )

        try:
            scores = scores.astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Anomaly ONNX score output is not numeric."
            ) from exc

        out_df[
            "isolation_forest_score"
        ] = scores

        _, if_flags = (
            cost_sensitive_isolation_forest_threshold(
                scores,
                risk_tolerance=risk_tolerance,
            )
        )

        if_flags = np.asarray(
            if_flags
        ).reshape(-1)

        if if_flags.size != expected_length:
            raise ValueError(
                "Isolation Forest flag output "
                "length does not match input rows."
            )

        out_df[
            "is_if_anomaly"
        ] = if_flags.astype(bool)

    else:
        # Fallback to the first ONNX output.
        try:
            label_values = preds.astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Anomaly prediction output is not numeric."
            ) from exc

        out_df[
            "is_if_anomaly"
        ] = (
            label_values == -1
        )

    # ---------------------------------------------------------
    # Lot-relative robust outlier detection
    # ---------------------------------------------------------

    out_df["robust_z_score"] = 0.0
    out_df["is_lot_outlier"] = False

    time_cols = [
        col
        for col in value_cols
        if col in out_df.columns
    ]

    if not time_cols:
        raise ValueError(
            "No valid measurement columns available "
            "for anomaly detection."
        )

    if "value_96h" in out_df.columns:
        primary = "value_96h"
    else:
        non_future_cols = [
            col
            for col in time_cols
            if col != "value_168h"
        ]

        if not non_future_cols:
            raise ValueError(
                "No pre-168h measurement available "
                "for Module A."
            )

        primary = non_future_cols[-1]

    group_cols = [
        lot_col,
        param_col,
    ]

    valid_group_cols = [
        col
        for col in group_cols
        if col in out_df.columns
    ]

    if valid_group_cols:
        groups = out_df.groupby(
            valid_group_cols,
            dropna=False,
        )
    else:
        groups = [
            (None, out_df)
        ]

    for _, group in groups:
        idx = group.index

        z_primary = _static_robust_z(
            group[primary]
        )

        out_df.loc[
            idx,
            "robust_z_score"
        ] = z_primary

        z_max = np.abs(
            z_primary.to_numpy(
                dtype=float
            )
        )

        # Never use 168h for Module A.
        safe_cols = [
            col
            for col in time_cols
            if col != "value_168h"
        ]

        for col in safe_cols:
            if col == primary:
                continue

            z = _static_robust_z(
                group[col]
            )

            z_values = np.abs(
                z.to_numpy(
                    dtype=float
                )
            )

            z_max = np.maximum(
                z_max,
                z_values,
            )

        out_df.loc[
            idx,
            "is_lot_outlier"
        ] = (
            z_max > threshold
        )

    # ---------------------------------------------------------
    # Absolute datasheet limit
    # ---------------------------------------------------------

    if (
        "datasheet_limit" in out_df.columns
        and primary in out_df.columns
    ):
        exceeds_static = (
            out_df[primary]
            > out_df["datasheet_limit"]
        )

        exceeds_static = (
            exceeds_static
            .fillna(False)
            .to_numpy(
                dtype=bool
            )
        )
    else:
        exceeds_static = np.zeros(
            expected_length,
            dtype=bool,
        )

    # ---------------------------------------------------------
    # Final anomaly flag
    # ---------------------------------------------------------

    if_anomaly = (
        out_df[
            "is_if_anomaly"
        ]
        .to_numpy(
            dtype=bool
        )
    )

    lot_outlier = (
        out_df[
            "is_lot_outlier"
        ]
        .to_numpy(
            dtype=bool
        )
    )

    out_df["is_anomaly"] = (
        if_anomaly
        | lot_outlier
        | exceeds_static
    )

    return out_df
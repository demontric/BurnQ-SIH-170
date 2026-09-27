import numpy as np

from preprocessing.features import engineer_features, EARLY_DRIFT_FEATURES
from models.model_registry import get_drift_model, normalize_parameter


_FUTURE_VALUE_COLS = (
    "value_96h",
    "value_168h",
    "Value_96h",
    "Value_168h",
)


def _onnx_matches_early_contract(session) -> bool:
    """
    Verify that the ONNX model expects the frozen Module B feature count.

    Module B must use only the 0h / 24h feature contract.
    """
    inputs = session.get_inputs()

    if not inputs:
        return False

    shape = inputs[0].shape

    if len(shape) < 2:
        return False

    feature_dim = shape[1]

    # Dynamic feature dimension.
    if feature_dim is None:
        return True

    try:
        return int(feature_dim) == len(EARLY_DRIFT_FEATURES)
    except (TypeError, ValueError):
        return False


def _extrapolate_168h(feat, tau=40.0):
    """
    Temporary deterministic fallback forecast using only 0h and 24h.

    Used until a compatible parameter-specific ONNX model is available.

    V(t) = V0 + a * (1 - exp(-t / tau))
    """

    v0 = feat["Value_0h"].to_numpy(dtype=np.float64)
    v24 = feat["Value_24h"].to_numpy(dtype=np.float64)

    denom = 1.0 - np.exp(-24.0 / tau)

    if abs(denom) < 1e-12:
        denom = 1e-12

    a = (v24 - v0) / denom

    prediction = (
        v0
        + a * (1.0 - np.exp(-168.0 / tau))
    )

    return prediction


def _predict_group(feat_group, parameter):
    """
    Predict 168h values for one parameter group.

    Uses the parameter-specific ONNX model when its input contract
    matches the frozen Module B feature contract. Otherwise uses
    the deterministic 0h/24h fallback.
    """

    sess = get_drift_model(parameter)

    if sess is None:
        return np.asarray(
            _extrapolate_168h(feat_group),
            dtype=float,
        )

    if not _onnx_matches_early_contract(sess):
        return np.asarray(
            _extrapolate_168h(feat_group),
            dtype=float,
        )

    X_drift = feat_group[
        EARLY_DRIFT_FEATURES
    ].to_numpy(
        dtype=np.float32,
        copy=True,
    )

    X_drift = np.nan_to_num(
        X_drift,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    inputs = sess.get_inputs()

    if not inputs:
        raise ValueError(
            f"Drift model for '{parameter}' has no input tensors."
        )

    input_name = inputs[0].name

    outputs = sess.run(
        None,
        {
            input_name: X_drift
        },
    )

    if not outputs:
        raise ValueError(
            f"Drift model for '{parameter}' returned no outputs."
        )

    preds = np.asarray(outputs[0])

    if preds.size == 0:
        raise ValueError(
            f"Drift model for '{parameter}' returned an empty prediction."
        )

    # Expected output:
    #   (N,)
    # or
    #   (N, 1)
    if preds.ndim == 1:
        pass

    elif preds.ndim == 2 and preds.shape[1] == 1:
        preds = preds.reshape(-1)

    else:
        raise ValueError(
            f"Drift model for '{parameter}' returned unexpected "
            f"prediction shape {preds.shape}. "
            f"Expected (N,) or (N,1)."
        )

    try:
        preds = preds.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Drift model for '{parameter}' returned non-numeric predictions."
        ) from exc

    if len(preds) != len(feat_group):
        raise ValueError(
            f"Drift model for '{parameter}' returned "
            f"{len(preds)} predictions for "
            f"{len(feat_group)} rows."
        )

    return preds


def predict_168h(df):
    """
    Predict 168h from the frozen Module B contract.

    IMPORTANT:
    96h and 168h measurements are removed from the prediction features.
    The actual 168h value is retained only for downstream evaluation.
    """

    out_df = df.copy()

    rename_mapping = {
        "lot_id": "Lot"
    }

    for col in (
        "value_0h",
        "value_24h",
    ):
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

    # Strictly prevent future measurements from entering Module B.
    feat_df = feat_df.drop(
        columns=[
            c
            for c in _FUTURE_VALUE_COLS
            if c in feat_df.columns
        ],
        errors="ignore",
    )

    if "datasheet_limit_uA" not in feat_df.columns:
        feat_df["datasheet_limit_uA"] = 50.0

    if "parameter" in out_df.columns:
        out_df["parameter"] = (
            out_df["parameter"]
            .map(normalize_parameter)
        )

        feat_df["parameter"] = (
            out_df["parameter"].values
        )

    feat = engineer_features(
        feat_df,
        include_168h=False,
        include_96h=False,
    )

    missing = [
        feature
        for feature in EARLY_DRIFT_FEATURES
        if feature not in feat.columns
    ]

    if missing:
        raise ValueError(
            f"Module B missing frozen features: {missing}"
        )

    out_df["predicted_168h"] = np.nan

    if "parameter" in out_df.columns:
        param_groups = out_df.groupby(
            "parameter",
            sort=False,
        )
    else:
        param_groups = [
            (
                normalize_parameter(
                    "leakage_current"
                ),
                out_df,
            )
        ]

    for parameter, group in param_groups:
        idx = group.index

        predictions = _predict_group(
            feat.loc[idx],
            parameter,
        )

        out_df.loc[
            idx,
            "predicted_168h"
        ] = predictions

    # Correct drift definition:
    # forecasted 24h -> predicted 168h over 144 hours.
    v24 = pd_to_numeric_safe(
        out_df["value_24h"]
    )

    out_df["predicted_drift_rate"] = (
        out_df["predicted_168h"] - v24
    ) / 144.0

    return out_df


def pd_to_numeric_safe(series):
    """
    Convert a pandas Series to numeric values safely.
    """
    import pandas as pd

    return pd.to_numeric(
        series,
        errors="coerce",
    )
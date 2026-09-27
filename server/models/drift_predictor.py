import numpy as np
from preprocessing.features import engineer_features, EARLY_DRIFT_FEATURES
from models.model_registry import get_drift_model, normalize_parameter

_FUTURE_VALUE_COLS = ("value_96h", "value_168h", "Value_96h", "Value_168h")


def _onnx_matches_early_contract(session) -> bool:
    got = session.get_inputs()[0].shape[1]
    return got is None or int(got) == len(EARLY_DRIFT_FEATURES)


def _extrapolate_168h(feat, tau=40.0):
    """0h/24h saturation forecast until a parameter ONNX matches EARLY_DRIFT_FEATURES.

    V(t) = V0 + a * (1 - exp(-t / tau)), with a solved from the 24h point.
    """
    v0 = feat["Value_0h"].to_numpy(dtype=np.float64)
    v24 = feat["Value_24h"].to_numpy(dtype=np.float64)
    denom = 1.0 - np.exp(-24.0 / tau)
    denom = 1e-12 if abs(denom) < 1e-12 else denom
    a = (v24 - v0) / denom
    return v0 + a * (1.0 - np.exp(-168.0 / tau))


def _predict_group(feat_group, parameter):
    sess = get_drift_model(parameter)
    if sess is not None and _onnx_matches_early_contract(sess):
        X_drift = feat_group[EARLY_DRIFT_FEATURES].to_numpy(dtype=np.float32, copy=True)
        X_drift = np.nan_to_num(X_drift, nan=0.0, posinf=0.0, neginf=0.0)
        preds = sess.run(None, {"input": X_drift})[0]
        if len(preds.shape) > 1 and preds.shape[1] == 1:
            preds = preds.flatten()
        return np.asarray(preds, dtype=float)
    return np.asarray(_extrapolate_168h(feat_group), dtype=float)


def predict_168h(df):
    out_df = df.copy()
    rename_mapping = {"lot_id": "Lot"}

    for col in ["value_0h", "value_24h"]:
        if col in out_df.columns:
            rename_mapping[col] = col.replace("value_", "Value_")

    if "datasheet_limit" in out_df.columns and "datasheet_limit_uA" not in out_df.columns:
        rename_mapping["datasheet_limit"] = "datasheet_limit_uA"

    feat_df = out_df.rename(columns=rename_mapping)
    feat_df = feat_df.drop(
        columns=[c for c in _FUTURE_VALUE_COLS if c in feat_df.columns],
        errors="ignore",
    )
    if "datasheet_limit_uA" not in feat_df.columns:
        feat_df["datasheet_limit_uA"] = 50.0

    if "parameter" in out_df.columns:
        out_df["parameter"] = out_df["parameter"].map(normalize_parameter)
        feat_df["parameter"] = out_df["parameter"].values

    feat = engineer_features(feat_df, include_168h=False, include_96h=False)
    missing = [c for c in EARLY_DRIFT_FEATURES if c not in feat.columns]
    if missing:
        raise ValueError(f"Module B missing frozen features: {missing}")

    out_df["predicted_168h"] = np.nan
    if "parameter" in out_df.columns:
        param_groups = out_df.groupby("parameter", sort=False)
    else:
        param_groups = [(normalize_parameter("leakage_current"), out_df)]

    for param, group in param_groups:
        idx = group.index
        out_df.loc[idx, "predicted_168h"] = _predict_group(feat.loc[idx], param)

    # Compute base drift rate and exit. Statistical boundaries are handled downstream.
    v0 = out_df["value_0h"] if "value_0h" in out_df.columns else 0.0
    out_df["predicted_drift_rate"] = (out_df["predicted_168h"] - v0) / 168.0

    return out_df
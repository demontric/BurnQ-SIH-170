import numpy as np
import os
import onnxruntime as ort
from preprocessing.features import engineer_features, DRIFT_MODEL_FEATURES

_SESSION = None

def _session():
    global _SESSION
    if _SESSION is None:
        model_path = os.path.join(os.path.dirname(__file__), "..", "exports", "drift_model.onnx")
        _SESSION = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    return _SESSION

def predict_168h(df, threshold=3.5, tier_col="datasheet_limit"):
    out_df = df.copy()
    rename_mapping = {"lot_id": "Lot"}

    for col in ["value_0h", "value_24h", "value_96h", "value_168h"]:
        if col in out_df.columns:
            rename_mapping[col] = col.replace("value_", "Value_")

    if "datasheet_limit" in out_df.columns and "datasheet_limit_uA" not in out_df.columns:
        rename_mapping["datasheet_limit"] = "datasheet_limit_uA"

    feat_df = out_df.rename(columns=rename_mapping)
    if "datasheet_limit_uA" not in feat_df.columns:
        feat_df["datasheet_limit_uA"] = 50.0

    feat = engineer_features(feat_df, include_168h=False)
    X_drift = feat[DRIFT_MODEL_FEATURES].to_numpy(dtype=np.float32, copy=True)

    # Removed the zeroing-out loop. The ONNX model requires true inputs to prevent high MAE.
    X_drift = np.nan_to_num(X_drift, nan=0.0, posinf=0.0, neginf=0.0)

    preds = _session().run(None, {"input": X_drift})[0]
    if len(preds.shape) > 1 and preds.shape[1] == 1:
        preds = preds.flatten()
    out_df["predicted_168h"] = preds.astype(float)

    v0 = out_df["value_0h"] if "value_0h" in out_df.columns else 0.0
    out_df["predicted_drift_rate"] = (out_df["predicted_168h"] - v0) / 168.0
    out_df["safety_slope"] = 0.0
    out_df["safety_slope_exceeded"] = False

    lot_col = "lot_id" if "lot_id" in out_df.columns else None
    param_col = "parameter" if "parameter" in out_df.columns else None

    # FIX: group by (lot, device tier) instead of (lot, parameter). `parameter`
    # is constant here, so grouping by it alone collapses to lot-only, which
    # mixes several device tiers (different datasheet_limit populations, very
    # different absolute scales) under one lot_id -- see outlier_detection.py
    # for the full explanation, the bug is identical here.
    if lot_col and tier_col in out_df.columns:
        group_keys = [lot_col, tier_col]
    elif lot_col and param_col:
        group_keys = [lot_col, param_col]
    else:
        group_keys = None

    groups = (out_df.groupby(group_keys) if group_keys else [(None, out_df)])

    for _, group in groups:
        idx = group.index
        rates = group["predicted_drift_rate"].astype(float)

        # FIX: static median/MAD over the whole group instead of .expanding().
        # Expanding is a cumulative/online stat -- the safety slope for a row
        # depended on how many rows of its group came before it in the
        # dataframe, not on the group's actual spread. That gave inconsistent,
        # order-dependent thresholds and let a lot of legitimate parts trip
        # "safety_slope_exceeded" simply because they were early in the file.
        median = float(rates.median())
        mad = float(np.median(np.abs(rates - median)))
        mad = max(mad, 1e-6)

        safety_slope = median + threshold * mad
        out_df.loc[idx, "safety_slope"] = safety_slope

        exceeds_lot_slope = rates > safety_slope
        exceeds_predicted_limit = (
            group["predicted_168h"] > group["datasheet_limit"]
            if "datasheet_limit" in group.columns else False
        )
        out_df.loc[idx, "safety_slope_exceeded"] = (exceeds_lot_slope | exceeds_predicted_limit)

    return out_df
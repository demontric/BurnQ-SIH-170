import numpy as np
import os
import onnxruntime as ort

from preprocessing.features import engineer_features, DRIFT_MODEL_FEATURES

_SESSION = None
ROBUST_Z_THRESHOLD = 3.5


def _session():
    global _SESSION
    if _SESSION is None:
        model_path = os.path.join(
            os.path.dirname(__file__), "..", "exports", "drift_model.onnx"
        )
        _SESSION = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    return _SESSION


def predict_168h(df):
    """
    Module B — forecast Value_168h from early burn-in history and flag
    parts whose predicted drift rate exceeds a lot-calculated safety slope.
    """
    out_df = df.copy()

    rename_mapping = {}
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
    groups = (
        out_df.groupby([lot_col, param_col])
        if lot_col and param_col
        else [(None, out_df)]
    )

    for _, group in groups:
        idx = group.index
        rates = group["predicted_drift_rate"].astype(float)
        median = rates.median()
        mad = np.median(np.abs(rates - median))
        if mad == 0:
            mad = 1e-6
        # Lot safety slope: robust upper bound on predicted µA/hour
        safety_slope = median + ROBUST_Z_THRESHOLD * (mad / 0.6745)
        out_df.loc[idx, "safety_slope"] = safety_slope

        exceeds_lot_slope = rates > safety_slope
        if "datasheet_limit" in group.columns:
            exceeds_predicted_limit = group["predicted_168h"] > group["datasheet_limit"]
        else:
            exceeds_predicted_limit = False
        out_df.loc[idx, "safety_slope_exceeded"] = (
            exceeds_lot_slope | exceeds_predicted_limit
        )

    return out_df

import numpy as np
import os
import onnxruntime as ort

from preprocessing.features import engineer_features, ANOMALY_MODEL_FEATURES

_SESSION = None
ROBUST_Z_THRESHOLD = 3.5


def _session():
    global _SESSION
    if _SESSION is None:
        model_path = os.path.join(
            os.path.dirname(__file__), "..", "exports", "anomaly_model.onnx"
        )
        _SESSION = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    return _SESSION


def _robust_z(series):
    vals = series.astype(float)
    median = vals.median()
    mad = np.median(np.abs(vals - median))
    if mad == 0:
        mad = 1e-6
    return 0.6745 * (vals - median) / mad


def detect_anomalies(
    df,
    lot_col="lot_id",
    param_col="parameter",
    value_cols=None,
):
    """
    Module A — lot-relative dynamic outlier detection.

    Flags parts that are anomalous versus their lot (robust z-score) and/or
    Isolation Forest trajectory outliers, including parts that still pass a
    static datasheet limit.
    """
    if value_cols is None:
        value_cols = ["value_0h", "value_24h", "value_96h", "value_168h"]

    out_df = df.copy()

    rename_mapping = {}
    for col in value_cols:
        if col in out_df.columns:
            rename_mapping[col] = col.replace("value_", "Value_")
    if "datasheet_limit" in out_df.columns and "datasheet_limit_uA" not in out_df.columns:
        rename_mapping["datasheet_limit"] = "datasheet_limit_uA"

    feat_df = out_df.rename(columns=rename_mapping)
    if "datasheet_limit_uA" not in feat_df.columns:
        feat_df["datasheet_limit_uA"] = 50.0

    feat = engineer_features(feat_df, include_168h=True)
    X_anom = feat[ANOMALY_MODEL_FEATURES].to_numpy(dtype=np.float32, copy=True)
    X_anom = np.nan_to_num(X_anom, nan=0.0, posinf=0.0, neginf=0.0)

    sess = _session()
    outputs = sess.run(None, {"input": X_anom})
    preds = outputs[0]
    if len(preds.shape) > 1 and preds.shape[1] == 1:
        preds = preds.flatten()
    out_df["is_if_anomaly"] = preds == -1

    if len(outputs) > 1:
        scores = outputs[1]
        if len(scores.shape) > 1 and scores.shape[1] == 1:
            scores = scores.flatten()
        out_df["isolation_forest_score"] = scores.astype(float)
    else:
        out_df["isolation_forest_score"] = 0.0

    out_df["robust_z_score"] = 0.0
    out_df["is_lot_outlier"] = False
    time_cols = [c for c in value_cols if c in out_df.columns]
    primary = "value_168h" if "value_168h" in out_df.columns else time_cols[-1]

    for _, group in out_df.groupby([lot_col, param_col]):
        idx = group.index
        z_primary = _robust_z(group[primary])
        out_df.loc[idx, "robust_z_score"] = z_primary
        z_max = np.abs(z_primary.to_numpy())
        for col in time_cols:
            if col == primary:
                continue
            z_max = np.maximum(z_max, np.abs(_robust_z(group[col]).to_numpy()))
        out_df.loc[idx, "is_lot_outlier"] = z_max > ROBUST_Z_THRESHOLD

    exceeds_static = False
    if "datasheet_limit" in out_df.columns and primary in out_df.columns:
        exceeds_static = out_df[primary] > out_df["datasheet_limit"]

    out_df["is_anomaly"] = (
        out_df["is_if_anomaly"].astype(bool)
        | out_df["is_lot_outlier"].astype(bool)
        | exceeds_static
    )
    return out_df

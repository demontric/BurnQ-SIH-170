import numpy as np
import os
import onnxruntime as ort
from preprocessing.features import engineer_features, ANOMALY_MODEL_FEATURES

_SESSION = None

def _session():
    global _SESSION
    if _SESSION is None:
        model_path = os.path.join(os.path.dirname(__file__), "..", "exports", "anomaly_model.onnx")
        _SESSION = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    return _SESSION

def _static_robust_z(series):
    """
    Full-group (not expanding/cumulative) robust z-score: every row in the
    group is compared against the SAME median/MAD computed over the whole
    group, so the result doesn't depend on row order.
    """
    vals = series.astype(float)
    med = float(vals.median())
    mad = float(np.median(np.abs(vals - med)))
    mad = max(mad, 1e-6)
    return 0.6745 * (vals - med) / mad

def detect_anomalies(
    df,
    lot_col="lot_id",
    param_col="parameter",
    tier_col="datasheet_limit",
    value_cols=None,
    threshold=3.5
):
    if value_cols is None:
        value_cols = ["value_0h", "value_24h", "value_96h", "value_168h"]

    out_df = df.copy()

    rename_mapping = {"lot_id": "Lot"}
    for col in value_cols:
        if col in out_df.columns:
            rename_mapping[col] = col.replace("value_", "Value_")

    if "datasheet_limit" in out_df.columns and "datasheet_limit_uA" not in out_df.columns:
        rename_mapping["datasheet_limit"] = "datasheet_limit_uA"

    feat_df = out_df.rename(columns=rename_mapping)
    if "datasheet_limit_uA" not in feat_df.columns:
        feat_df["datasheet_limit_uA"] = 50.0

    # Model stacking hack removed so the ONNX model processes actual data
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
    primary = "value_96h" if "value_96h" in out_df.columns else time_cols[-2]

    # FIX: group by (lot, device tier) instead of (lot, parameter). `parameter`
    # is constant across every row here, so grouping by it alone collapses to
    # grouping by lot only -- which silently mixes multiple device tiers
    # (different datasheet_limit populations) that live under the same lot_id
    # but sit on completely different absolute scales. Comparing a ~5uA part
    # against a group that also contains ~100-300uA parts makes the median/MAD
    # meaningless and floods the anomaly flag with false positives.
    group_cols = [lot_col, tier_col] if tier_col in out_df.columns else [lot_col, param_col]

    for _, group in out_df.groupby(group_cols):
        idx = group.index

        # FIX: static per-group median/MAD instead of .expanding(). Expanding
        # is a cumulative/online statistic -- it only "sees" rows up to that
        # row's position in the dataframe, so the same value gets a different
        # z-score depending on where it happens to sit in the file, and the
        # first rows of every group get an artificially tiny MAD. Module A's
        # spec is "compare a part against its lot's stats" -- that means one
        # fixed median/MAD over the whole group, applied to every row in it.
        z_primary = _static_robust_z(group[primary])
        out_df.loc[idx, "robust_z_score"] = z_primary
        z_max = np.abs(z_primary.to_numpy())

        safe_cols = [c for c in time_cols if c != "value_168h"]
        for col in safe_cols:
            if col == primary:
                continue
            z_max = np.maximum(z_max, np.abs(_static_robust_z(group[col]).to_numpy()))
        out_df.loc[idx, "is_lot_outlier"] = z_max > threshold

    exceeds_static = False
    if "datasheet_limit" in out_df.columns and primary in out_df.columns:
        exceeds_static = out_df[primary] > out_df["datasheet_limit"]

    out_df["is_anomaly"] = (
        out_df["is_if_anomaly"].astype(bool)
        | out_df["is_lot_outlier"].astype(bool)
        | exceeds_static
    )

    return out_df
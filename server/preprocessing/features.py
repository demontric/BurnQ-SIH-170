import numpy as np
import pandas as pd

TIMEPOINTS = [0, 24, 96, 168]

def engineer_features(df: pd.DataFrame, include_168h: bool = True) -> pd.DataFrame:
    df = df.copy()

    # --- Prevent Data Leakage: Dynamic Rolling Lot Statistics ---
    # We calculate these for the UI and the pipeline routing, but they are NOT 
    # pushed into the ONNX feature lists below to prevent dimension mismatch.
    df['lot_mean_0h'] = df.groupby('Lot')['Value_0h'].transform(lambda x: x.expanding().mean())
    df['lot_mean_24h'] = df.groupby('Lot')['Value_24h'].transform(lambda x: x.expanding().mean())
    df['lot_mean_96h'] = df.groupby('Lot')['Value_96h'].transform(lambda x: x.expanding().mean())

    # --- Base Deltas & Pct Changes ---
    denom_0 = df["Value_0h"].replace(0, np.nan).fillna(1e-6)
    denom_96 = df["Value_96h"].replace(0, np.nan).fillna(1e-6)
    df["delta_24"] = df["Value_24h"] - df["Value_0h"]
    df["delta_96"] = df["Value_96h"] - df["Value_24h"]
    df["pct_change_24"] = df["delta_24"] / denom_0
    df["pct_change_96_from_0"] = (df["Value_96h"] - df["Value_0h"]) / denom_0

    # --- Segment Growth Rates ---
    df["growth_rate_0_24"] = df["delta_24"] / 24.0
    df["growth_rate_24_96"] = df["delta_96"] / (96.0 - 24.0)

    # --- Early Drift Velocity / Acceleration ---
    df["drift_velocity_early"] = df["growth_rate_24_96"]
    df["drift_acceleration_early"] = df["growth_rate_24_96"] - df["growth_rate_0_24"]

    # --- Early Row-wise Stats ---
    early_cols = ["Value_0h", "Value_24h", "Value_96h"]
    df["early_mean"] = df[early_cols].mean(axis=1)
    df["early_std"] = df[early_cols].std(axis=1)

    # --- Margins ---
    df["margin_0h"] = df["datasheet_limit_uA"] - df["Value_0h"]
    df["margin_ratio_96h"] = df["Value_96h"] / df["datasheet_limit_uA"]

    # --- Late Stage / 168h Target Features ---
    if include_168h:
        df["delta_168"] = df["Value_168h"] - df["Value_96h"]
        df["pct_change_168_from_96"] = df["delta_168"] / denom_96
        df["growth_rate_96_168"] = df["delta_168"] / (168.0 - 96.0)
        df["drift_velocity_late"] = df["growth_rate_96_168"]
        df["drift_acceleration_late"] = df["growth_rate_96_168"] - df["growth_rate_24_96"]
        df["overall_growth_rate"] = (df["Value_168h"] - df["Value_0h"]) / 168.0

        all_cols = ["Value_0h", "Value_24h", "Value_96h", "Value_168h"]
        df["full_mean"] = df[all_cols].mean(axis=1)
        df["full_std"] = df[all_cols].std(axis=1)
        df["margin_168h"] = df["datasheet_limit_uA"] - df["Value_168h"]
        df["margin_ratio_168h"] = df["Value_168h"] / df["datasheet_limit_uA"]

        df["degradation_index"] = (
            df["overall_growth_rate"].clip(lower=0) / df["Value_0h"]
        ) * (df["margin_ratio_168h"])

    return df

# RESTORED: Exact 15 features expected by drift_model.onnx
# The 96h isolation is handled by passing zeros for these indices inside drift_predictor.py
DRIFT_MODEL_FEATURES = [
    "Value_0h", "Value_24h", "Value_96h",
    "delta_24", "delta_96", "pct_change_24", "pct_change_96_from_0",
    "growth_rate_0_24", "growth_rate_24_96",
    "drift_velocity_early", "drift_acceleration_early",
    "early_mean", "early_std", "margin_0h", "margin_ratio_96h",
]

# RESTORED: Exact 26 features expected by anomaly_model.onnx
ANOMALY_MODEL_FEATURES = DRIFT_MODEL_FEATURES + [
    "delta_168", "pct_change_168_from_96", "growth_rate_96_168",
    "drift_velocity_late", "drift_acceleration_late",
    "overall_growth_rate", "full_mean", "full_std",
    "margin_168h", "margin_ratio_168h", "degradation_index",
]
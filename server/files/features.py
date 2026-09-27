import numpy as np
import pandas as pd

TIMEPOINTS = [0, 24, 96, 168]


def engineer_features(df: pd.DataFrame, include_168h: bool = True) -> pd.DataFrame:
    df = df.copy()

    # --- deltas & pct changes vs baseline ---
    df["delta_24"] = df["Value_24h"] - df["Value_0h"]
    df["delta_96"] = df["Value_96h"] - df["Value_24h"]
    df["pct_change_24"] = df["delta_24"] / df["Value_0h"]
    df["pct_change_96_from_0"] = (df["Value_96h"] - df["Value_0h"]) / df["Value_0h"]

    # --- segment growth rates (slope between consecutive checkpoints) ---
    df["growth_rate_0_24"] = df["delta_24"] / 24.0
    df["growth_rate_24_96"] = df["delta_96"] / (96.0 - 24.0)

    # --- drift velocity / acceleration using only 0-96h (drift-model-safe) ---
    df["drift_velocity_early"] = df["growth_rate_24_96"]
    df["drift_acceleration_early"] = df["growth_rate_24_96"] - df["growth_rate_0_24"]

    # --- early rolling stats (0h,24h,96h only) ---
    early_cols = ["Value_0h", "Value_24h", "Value_96h"]
    df["early_mean"] = df[early_cols].mean(axis=1)
    df["early_std"] = df[early_cols].std(axis=1)

    # --- margin to datasheet limit (0h/24h/96h only — always safe) ---
    df["margin_0h"] = df["datasheet_limit_uA"] - df["Value_0h"]
    df["margin_ratio_96h"] = df["Value_96h"] / df["datasheet_limit_uA"]

    if include_168h:
        df["delta_168"] = df["Value_168h"] - df["Value_96h"]
        df["pct_change_168_from_96"] = df["delta_168"] / df["Value_96h"]
        df["growth_rate_96_168"] = df["delta_168"] / (168.0 - 96.0)
        df["drift_velocity_late"] = df["growth_rate_96_168"]
        df["drift_acceleration_late"] = df["growth_rate_96_168"] - df["growth_rate_24_96"]
        df["overall_growth_rate"] = (df["Value_168h"] - df["Value_0h"]) / 168.0
        all_cols = ["Value_0h", "Value_24h", "Value_96h", "Value_168h"]
        df["full_mean"] = df[all_cols].mean(axis=1)
        df["full_std"] = df[all_cols].std(axis=1)
        df["margin_168h"] = df["datasheet_limit_uA"] - df["Value_168h"]
        df["margin_ratio_168h"] = df["Value_168h"] / df["datasheet_limit_uA"]
        # simple health/degradation index: normalized cumulative drift + margin pressure
        df["degradation_index"] = (
            df["overall_growth_rate"].clip(lower=0) / df["Value_0h"]
        ) * (df["margin_ratio_168h"])

    return df


DRIFT_MODEL_FEATURES = [
    "Value_0h", "Value_24h", "Value_96h",
    "delta_24", "delta_96", "pct_change_24", "pct_change_96_from_0",
    "growth_rate_0_24", "growth_rate_24_96",
    "drift_velocity_early", "drift_acceleration_early",
    "early_mean", "early_std", "margin_0h", "margin_ratio_96h",
]

ANOMALY_MODEL_FEATURES = DRIFT_MODEL_FEATURES + [
    "delta_168", "pct_change_168_from_96", "growth_rate_96_168",
    "drift_velocity_late", "drift_acceleration_late",
    "overall_growth_rate", "full_mean", "full_std",
    "margin_168h", "margin_ratio_168h", "degradation_index",
]

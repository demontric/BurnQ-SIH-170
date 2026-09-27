import numpy as np
import pandas as pd

TIMEPOINTS = [0, 24, 96, 168]


def engineer_features(
    df: pd.DataFrame,
    include_168h: bool = True,
    include_96h: bool = True,
) -> pd.DataFrame:
    """Engineer burn-in features.

    include_96h / include_168h gate later-hour columns so Module B can be
    built from 0h/24h only. Module A still uses the 96h path but strictly excludes 168h.
    """
    df = df.copy()

    # --- Prevent Data Leakage: Dynamic Rolling Lot Statistics ---
    # Calculated for the UI and pipeline routing, but not pushed into the
    # frozen ONNX feature lists below (dimension mismatch).
    if "Lot" in df.columns:
        lot_keys = ["Lot", "parameter"] if "parameter" in df.columns else ["Lot"]
        df["lot_mean_0h"] = df.groupby(lot_keys)["Value_0h"].transform(lambda x: x.expanding().mean())
        df["lot_mean_24h"] = df.groupby(lot_keys)["Value_24h"].transform(lambda x: x.expanding().mean())
        if include_96h and "Value_96h" in df.columns:
            df["lot_mean_96h"] = df.groupby(lot_keys)["Value_96h"].transform(
                lambda x: x.expanding().mean()
            )

    denom_0 = df["Value_0h"].replace(0, np.nan).fillna(1e-6)

    # --- Frozen Module B / early (0h, 24h) features ---
    df["delta_24"] = df["Value_24h"] - df["Value_0h"]
    df["pct_change_24"] = df["delta_24"] / denom_0
    df["growth_rate_0_24"] = df["delta_24"] / 24.0
    df["margin_0h"] = df["datasheet_limit_uA"] - df["Value_0h"]
    df["margin_ratio_24h"] = df["Value_24h"] / df["datasheet_limit_uA"]

    early_cols = ["Value_0h", "Value_24h"]
    if include_96h and "Value_96h" in df.columns:
        # Module A was trained with 3-point early stats (0h/24h/96h).
        early_cols = ["Value_0h", "Value_24h", "Value_96h"]
    df["early_mean"] = df[early_cols].mean(axis=1)
    df["early_std"] = df[early_cols].std(axis=1)

    # --- 96h features (Module A / evaluation only — never Module B) ---
    if include_96h:
        denom_96 = df["Value_96h"].replace(0, np.nan).fillna(1e-6)
        df["delta_96"] = df["Value_96h"] - df["Value_24h"]
        df["pct_change_96_from_0"] = (df["Value_96h"] - df["Value_0h"]) / denom_0
        df["growth_rate_24_96"] = df["delta_96"] / (96.0 - 24.0)
        df["drift_velocity_early"] = df["growth_rate_24_96"]
        df["drift_acceleration_early"] = df["growth_rate_24_96"] - df["growth_rate_0_24"]
        df["margin_ratio_96h"] = df["Value_96h"] / df["datasheet_limit_uA"]
    else:
        denom_96 = None

    # --- Late Stage / 168h evaluation features ---
    if include_168h:
        if denom_96 is None:
            denom_96 = df["Value_96h"].replace(0, np.nan).fillna(1e-6)
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


# Frozen Module B contract (0h/24h only). Must not include 96h or 168h.
# leakage_current, iddq, and prop_delay ONNX exports all consume this order.
EARLY_DRIFT_FEATURES = [
    "Value_0h",
    "Value_24h",
    "delta_24",
    "pct_change_24",
    "growth_rate_0_24",
    "early_mean",
    "early_std",
    "margin_0h",
    "margin_ratio_24h",
]

# Module A Isolation Forest — Strict distinction: NO 168h features to prevent leakage.
# Includes only 0h to 96h data points.
MODULE_A_FEATURES = [
    "Value_0h", "Value_24h", "Value_96h",
    "delta_24", "delta_96", "pct_change_24", "pct_change_96_from_0",
    "growth_rate_0_24", "growth_rate_24_96",
    "drift_velocity_early", "drift_acceleration_early",
    "early_mean", "early_std", "margin_0h", "margin_ratio_96h",
]

# Held-out 168h features used for MAE / reports, never as Module B inputs.
EVALUATION_FEATURES = [
    "Value_168h",
    "delta_168",
    "pct_change_168_from_96",
    "growth_rate_96_168",
    "drift_velocity_late",
    "drift_acceleration_late",
    "overall_growth_rate",
    "full_mean",
    "full_std",
    "margin_168h",
    "margin_ratio_168h",
    "degradation_index",
]

# Back-compat names. DRIFT_MODEL_FEATURES is the 0h/24h contract, not the old 15-col 96h list.
DRIFT_MODEL_FEATURES = EARLY_DRIFT_FEATURES
ANOMALY_MODEL_FEATURES = MODULE_A_FEATURES

_FUTURE_MARKERS = ("96h", "168h", "_96", "_168", "24_96", "96_168")
assert not any(
    any(marker in name for marker in _FUTURE_MARKERS) for name in EARLY_DRIFT_FEATURES
), "EARLY_DRIFT_FEATURES must not include 96h/168h columns"

_168H_MARKERS = ("168h", "_168", "96_168", "overall", "full", "degradation")
assert not any(
    any(marker in name for marker in _168H_MARKERS) for name in MODULE_A_FEATURES
), "MODULE_A_FEATURES must strictly exclude 168h columns"
import numpy as np
import pandas as pd

def calculate_safety_slope(
    df: pd.DataFrame, 
    threshold: float = 3.5, 
    lot_col: str = "lot_id", 
    param_col: str = "parameter"
) -> pd.DataFrame:
    """
    Computes a lot-relative robust statistical boundary (median + threshold * MAD)
    for predicted drift rates. 
    """
    out_df = df.copy()

    out_df["safety_slope"] = 0.0
    out_df["safety_slope_exceeded"] = False

    # Group strictly by lot and parameter to ensure clean statistical populations
    group_keys = [c for c in [lot_col, param_col] if c in out_df.columns]
    groups = out_df.groupby(group_keys) if group_keys else [(None, out_df)]

    for _, group in groups:
        idx = group.index
        rates = group["predicted_drift_rate"].astype(float)

        median = float(rates.median())
        mad = float(np.median(np.abs(rates - median)))
        mad = max(mad, 1e-6)

        safety_slope = median + threshold * mad
        out_df.loc[idx, "safety_slope"] = safety_slope

        # Note: Absolute limit checking is now handled by the Decision Engine.
        # This explicitly isolates the lot-relative statistical slope boundary.
        out_df.loc[idx, "safety_slope_exceeded"] = rates > safety_slope

    return out_df
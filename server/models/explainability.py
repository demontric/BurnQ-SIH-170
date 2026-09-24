import pandas as pd


def generate_justification(row: pd.Series, median_168h: float, mad_168h: float) -> str:
    """Human-readable reasons a QA inspector can verify against lot stats."""
    reasons = []
    parameter = row.get("parameter", "parameter")
    val_168 = row.get("value_168h")
    z = row.get("robust_z_score")
    predicted = row.get("predicted_168h")
    limit = row.get("datasheet_limit")
    v0 = row.get("value_0h")
    v24 = row.get("value_24h")

    if z is not None and not pd.isna(z) and abs(z) > 3.5:
        val_txt = f"{float(val_168):.2f} µA" if val_168 is not None and not pd.isna(val_168) else "n/a"
        reasons.append(
            f"The value ({val_txt}) is {abs(float(z)):.1f}x the lot median ({median_168h:.2f} µA)."
        )

    if row.get("is_if_anomaly") or (
        row.get("is_anomaly") and (z is None or pd.isna(z) or abs(z) <= 3.5)
    ):
        reasons.append(
            "The multivariate burn-in path exhibits a sudden non-linear drift compared to peers."
        )

    if limit is not None and not pd.isna(limit) and val_168 is not None and not pd.isna(val_168):
        if float(val_168) > float(limit):
            reasons.append(
                f"The 168h value ({float(val_168):.2f} µA) exceeds the static datasheet limit ({float(limit):.1f} µA)."
            )

    if row.get("safety_slope_exceeded"):
        rate = row.get("predicted_drift_rate")
        slope = row.get("safety_slope")
        pred_txt = (
            f"{float(predicted):.2f} µA"
            if predicted is not None and not pd.isna(predicted)
            else "n/a"
        )
        rate_txt = f"{float(rate):.4f} µA/h" if rate is not None and not pd.isna(rate) else "n/a"
        slope_txt = (
            f"{float(slope):.4f} µA/h" if slope is not None and not pd.isna(slope) else "n/a"
        )
        reasons.append(
            f"The 24h drift rate ({rate_txt}) is higher than the {row.get('lot_id', 'lot')} historical average ({slope_txt}), resulting in a projected 168h failure."
        )

    if not reasons:
        if row.get("is_anomaly") or row.get("safety_slope_exceeded"):
            return "Flagged for Rejection: Abnormal drift or statistical deviation."
        return "Pass: Trajectory is consistent with the lot."

    return "Flagged for Rejection: " + " ".join(reasons)

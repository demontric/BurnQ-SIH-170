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
            f"Module A (lot outlier): {parameter} at 168h is {abs(float(z)):.1f} robust-z "
            f"from the lot median ({val_txt} vs median {median_168h:.2f} µA, MAD {mad_168h:.2f} µA). "
            "This can fail relative to the lot even when still under the static datasheet limit."
        )

    if row.get("is_if_anomaly") or (
        row.get("is_anomaly") and (z is None or pd.isna(z) or abs(z) <= 3.5)
    ):
        score = row.get("isolation_forest_score")
        score_txt = f"{float(score):.3f}" if score is not None and not pd.isna(score) else "n/a"
        reasons.append(
            f"Module A (trajectory IF): multivariate burn-in path is anomalous versus peers "
            f"(Isolation Forest score {score_txt})."
        )

    if limit is not None and not pd.isna(limit) and val_168 is not None and not pd.isna(val_168):
        if float(val_168) > float(limit):
            reasons.append(
                f"Static datasheet fail: 168h {float(val_168):.2f} µA exceeds {float(limit):.1f} µA."
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
        early = ""
        if v0 is not None and v24 is not None and not pd.isna(v0) and not pd.isna(v24):
            early = f" Early path: 0h={float(v0):.2f} µA, 24h={float(v24):.2f} µA."
        reasons.append(
            f"Module B (drift): predicted 168h is {pred_txt}; predicted slope {rate_txt} "
            f"exceeds lot safety slope {slope_txt}.{early}"
        )

    if not reasons:
        if row.get("is_anomaly") or row.get("safety_slope_exceeded"):
            return "Flagged due to abnormal drift or statistical deviation versus the lot."
        return (
            "Pass: trajectory is consistent with the lot. No lot-relative outlier, "
            "Isolation Forest flag, or predicted-drift safety-slope exceedance."
        )

    return "Flagged: " + " ".join(reasons)

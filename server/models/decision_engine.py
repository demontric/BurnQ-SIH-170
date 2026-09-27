import pandas as pd

def evaluate_component(current_val, predicted_val, is_anomaly, safety_slope_exceeded, datasheet_limit):
    """
    Evaluates a single component's risk profile based on absolute limits,
    predictive drift, and statistical outliers.
    """
    reason_codes = []
    
    if current_val > datasheet_limit:
        reason_codes.append("ABSOLUTE_LIMIT_EXCEEDED")
    
    if is_anomaly:
        reason_codes.append("LOT_RELATIVE_OUTLIER")
        
    if pd.notna(predicted_val) and predicted_val > datasheet_limit:
        reason_codes.append("PREDICTED_LIMIT_BREACH")
        
    if safety_slope_exceeded:
        reason_codes.append("SAFETY_SLOPE_EXCEEDED")
        
    if any(code in reason_codes for code in ["ABSOLUTE_LIMIT_EXCEEDED", "PREDICTED_LIMIT_BREACH", "SAFETY_SLOPE_EXCEEDED"]):
        status = "REJECT"
    elif "LOT_RELATIVE_OUTLIER" in reason_codes:
        status = "WATCH"
    else:
        status = "PASS"
        
    is_flagged = len(reason_codes) > 0
    
    return status, is_flagged, reason_codes


def apply_decision_engine(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies the decision engine logic across the entire dataframe.
    """
    out = df.copy()
    
    statuses, flags, all_reasons = [], [], []
    
    for _, row in out.iterrows():
        # Fall back to 24h if 96h is somehow missing
        current_val = row.get("value_96h") if pd.notna(row.get("value_96h")) else row.get("value_24h", 0)
        
        status, is_flagged, reason_codes = evaluate_component(
            current_val=current_val,
            predicted_val=row.get("predicted_168h", 0),
            is_anomaly=row.get("is_anomaly", False),
            safety_slope_exceeded=row.get("safety_slope_exceeded", False),
            datasheet_limit=row.get("datasheet_limit", 50.0)
        )
        
        statuses.append(status)
        flags.append(is_flagged)
        all_reasons.append(reason_codes)
        
    out["status"] = statuses
    out["is_flagged"] = flags
    out["reason_codes"] = all_reasons
    
    return out
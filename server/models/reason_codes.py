"""Mapping of decision engine reason codes to deterministic explanations."""

REASON_MESSAGES = {
    "LOT_RELATIVE_OUTLIER": "Parameter is significantly above the normal distribution for its lot.",
    "PREDICTED_LIMIT_BREACH": "Predicted 168h value exceeds the datasheet limit.",
    "SAFETY_SLOPE_EXCEEDED": "Predicted drift exceeds the calibrated safety slope.",
    "ABSOLUTE_LIMIT_EXCEEDED": "Measured value exceeds the datasheet limit.",
    "ML_ANOMALY_SCORE": "The machine-learning anomaly model identified an unusual measurement pattern.",
}

def get_reason_message(code: str) -> str:
    """Safely fetch a deterministic message for a given code."""
    return REASON_MESSAGES.get(code, f"Flagged for unmapped rule: {code}.")
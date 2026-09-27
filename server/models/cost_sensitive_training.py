import numpy as np
from sklearn.ensemble import IsolationForest

def cost_sensitive_isolation_forest_threshold(anomaly_scores, risk_tolerance=50.0):
    """
    Dynamic thresholding for Isolation Forest (Module A).

    risk_tolerance: 1 to 100.
    1 = Catch everything (High False Positives, Zero False Negatives).
    100 = Catch only massive outliers (Low False Positives, High False Negatives).
    """
    # IsolationForest decision_function: negative scores are anomalies, positive are normal.
    base_threshold = -0.1

    # Low risk tolerance increases the threshold (closer to 0 or positive)
    modifier = (50.0 - risk_tolerance) * 0.005

    # Add the modifier to shift the decision boundary
    dynamic_threshold = base_threshold + modifier

    # Flags as True (Anomaly) if the score is lower than the dynamic threshold
    flags = anomaly_scores < dynamic_threshold

    return dynamic_threshold, flags


def score_and_flag_isolation_forest(features, model=None, risk_tolerance=50.0):
    """
    Evaluates features against a trained Isolation Forest and applies the
    risk_tolerance-driven threshold above.

    NOTE: renamed from `detect_anomalies` (its previous name here) to avoid
    colliding with models.outlier_detection.detect_anomalies, which is a
    different function -- the two names being identical across two files
    was a standing source of confusion in this codebase.
    """
    if model is None:
        # Fallback initialization; in production, load the pre-trained/warmed model
        model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
        model.fit(features)

    anomaly_scores = model.decision_function(features)
    threshold, flags = cost_sensitive_isolation_forest_threshold(anomaly_scores, risk_tolerance)

    return flags, anomaly_scores, threshold
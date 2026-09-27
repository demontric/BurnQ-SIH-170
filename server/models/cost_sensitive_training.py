import numpy as np
from sklearn.ensemble import IsolationForest
from config.thresholds import THRESHOLDS


def cost_sensitive_isolation_forest_threshold(
    anomaly_scores,
    risk_tolerance=50.0
):
    """
    Dynamic thresholding for Isolation Forest (Module A).

    risk_tolerance: 1 to 100.
    1 = Catch everything
    100 = Catch only massive outliers
    """

    base_threshold = (
        THRESHOLDS
        .get("module_a", {})
        .get("isolation_forest", -0.1)
    )

    modifier = (50.0 - risk_tolerance) * 0.005
    dynamic_threshold = base_threshold + modifier

    flags = anomaly_scores < dynamic_threshold

    return dynamic_threshold, flags


def score_and_flag_isolation_forest(
    features,
    model=None,
    risk_tolerance=50.0
):
    """
    Evaluates features against a trained Isolation Forest
    and applies the risk-tolerance threshold.
    """

    if model is None:
        model = IsolationForest(
            n_estimators=100,
            contamination=0.05,
            random_state=42
        )
        model.fit(features)

    anomaly_scores = model.decision_function(features)

    threshold, flags = cost_sensitive_isolation_forest_threshold(
        anomaly_scores,
        risk_tolerance
    )

    return flags, anomaly_scores, threshold
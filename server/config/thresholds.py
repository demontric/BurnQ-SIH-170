"""
Centralized configuration for ML model and statistical thresholds.
These values can be calibrated later based on validation results.
"""

THRESHOLDS = {
    "module_a": {
        "robust_z": 3.5,
        "isolation_forest": -0.08
    },
    "module_b": {
        "safety_slope_percentile": 95
    }
}
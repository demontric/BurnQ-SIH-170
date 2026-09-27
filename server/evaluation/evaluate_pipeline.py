import pandas as pd
from evaluation.evaluate_module_a import evaluate_anomaly_detection
from evaluation.evaluate_module_b import evaluate_drift_prediction

def evaluate_full_pipeline(df: pd.DataFrame) -> dict:
    """
    Generates a comprehensive evaluation report for the screened dataset.
    Requires the input DataFrame to contain pipeline outputs ('is_anomaly', 
    'predicted_168h') alongside ground truth labels ('true_anomaly', 'value_168h').
    """
    report = {}
    
    try:
        report["module_a"] = evaluate_anomaly_detection(df)
    except ValueError as e:
        report["module_a"] = {"error": str(e)}
        
    try:
        report["module_b"] = evaluate_drift_prediction(df)
    except ValueError as e:
        report["module_b"] = {"error": str(e)}
        
    return report
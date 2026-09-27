import pandas as pd
from evaluation.metrics import classification_metrics

def evaluate_anomaly_detection(
    df: pd.DataFrame, 
    true_col: str = "true_anomaly", 
    pred_col: str = "is_anomaly"
) -> dict:
    """
    Evaluates Module A against ground-truth anomaly labels.
    """
    if true_col not in df.columns or pred_col not in df.columns:
        raise ValueError(f"Missing columns for Module A evaluation: {true_col}, {pred_col}")
        
    # Ensure boolean types for confusion matrix
    y_true = df[true_col].astype(bool)
    y_pred = df[pred_col].astype(bool)
    
    return classification_metrics(y_true, y_pred)
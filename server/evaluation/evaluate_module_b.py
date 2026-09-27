import pandas as pd
from evaluation.metrics import regression_metrics

def evaluate_drift_prediction(
    df: pd.DataFrame, 
    true_col: str = "value_168h", 
    pred_col: str = "predicted_168h", 
    param_col: str = "parameter"
) -> dict:
    """
    Evaluates Module B predictions overall and per-parameter.
    """
    if true_col not in df.columns or pred_col not in df.columns:
         raise ValueError(f"Missing columns for Module B evaluation: {true_col}, {pred_col}")

    # Calculate aggregate metrics across all components
    results = {
        "overall": regression_metrics(df[true_col], df[pred_col])
    }
    
    # Stratify metrics by parameter type (leakage, IDDQ, prop_delay)
    if param_col in df.columns:
        results["per_parameter"] = {}
        for param, group in df.groupby(param_col, dropna=False):
            param_name = str(param)
            results["per_parameter"][param_name] = regression_metrics(
                group[true_col], 
                group[pred_col]
            )
            
    return results
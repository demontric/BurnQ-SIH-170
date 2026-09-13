import pandas as pd
import numpy as np

# A real implementation would use SHAP with the IsolationForest
# import shap

def compute_shap_values(model, X):
    """
    Stub for SHAP value computation.
    """
    # explainer = shap.TreeExplainer(model)
    # shap_values = explainer.shap_values(X)
    # return shap_values
    pass

def generate_justification(row: pd.Series, median_168h: float, mad_168h: float) -> str:
    """
    Generates a human-readable justification string for flagged parts.
    """
    reasons = []
    
    # Check outlier criteria
    if 'robust_z_score' in row and not pd.isna(row['robust_z_score']):
        if abs(row['robust_z_score']) > 3.5:
            val = row['value_168h'] if 'value_168h' in row else 0
            reasons.append(f"{row['parameter']} at 168h is {abs(row['robust_z_score']):.1f} MAD away from lot median "
                           f"({val:.2f} vs lot median {median_168h:.2f}, MAD {mad_168h:.2f}).")
            
    if 'isolation_forest_score' in row and not pd.isna(row['isolation_forest_score']):
        # If it's negative, it's flagged by IF
        if row.get('is_anomaly', False) and abs(row.get('robust_z_score', 0)) <= 3.5:
             reasons.append(f"Statistical multi-variate trajectory anomaly detected (IF score: {row['isolation_forest_score']: .3f}).")
            
    if 'safety_slope_exceeded' in row and row['safety_slope_exceeded']:
        reasons.append(f"Predicted drift trajectory (0h->24h) exceeds safety slope threshold.")
        
    if not reasons:
        if row.get('is_anomaly', False) or row.get('safety_slope_exceeded', False):
            return "Flagged due to abnormal drift or statistical deviation."
        return "Normal part."
        
    return " Flagged: " + " ".join(reasons)

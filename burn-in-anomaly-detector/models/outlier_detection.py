import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

def detect_anomalies(df: pd.DataFrame, lot_col='lot_id', param_col='parameter', 
                      value_cols=['value_0h','value_24h','value_96h','value_168h']) -> pd.DataFrame:
    """
    Returns the input df with added columns:
    - robust_z_score: median/MAD-based z-score per lot+parameter
    - isolation_forest_score: anomaly score from IsolationForest
    - is_anomaly: bool, ensemble decision (flag if ANY method flags it — optimize for recall)
    """
    out_df = df.copy()
    
    # Initialize columns
    out_df['robust_z_score'] = 0.0
    out_df['isolation_forest_score'] = 0.0
    out_df['is_anomaly'] = False
    
    # Group by lot and parameter
    for (lot, param), group in out_df.groupby([lot_col, param_col]):
        idx = group.index
        
        # 1. Robust Z-Score (using 168h value as the primary indicator for absolute drift outliers)
        # We could also do this on the max value across time. 
        # Using 168h for simplicity as it represents the end state of burn-in.
        vals = group['value_168h']
        median = vals.median()
        mad = np.median(np.abs(vals - median))
        if mad == 0:
            mad = 1e-6 # prevent division by zero
        
        # Calculate robust z-score: 0.6745 is the constant for normal distribution consistency
        robust_z = 0.6745 * (vals - median) / mad
        out_df.loc[idx, 'robust_z_score'] = robust_z
        
        # 2. Isolation Forest
        X = group[value_cols].values
        # Contamination is expected defect rate. We use 0.05 as default assumption.
        iso = IsolationForest(contamination=0.05, random_state=42)
        
        if len(X) > 10: # IF needs some samples to work reasonably
            iso.fit(X)
            # decision_function gives anomaly score (lower is more anomalous)
            scores = iso.decision_function(X)
            out_df.loc[idx, 'isolation_forest_score'] = scores
            
            # Predict returns -1 for anomaly, 1 for normal
            preds = iso.predict(X)
            is_iso_anomaly = (preds == -1)
        else:
            is_iso_anomaly = np.zeros(len(X), dtype=bool)
            
        # 3. Ensemble
        # Flag if robust Z > 3.5 OR IF flags it
        is_z_anomaly = (np.abs(robust_z) > 3.5)
        out_df.loc[idx, 'is_anomaly'] = is_z_anomaly | is_iso_anomaly

    # TODO: A more sophisticated model (e.g., Autoencoder or PyOD ensemble) could be swapped in here.
    return out_df

import pandas as pd
import numpy as np
import os
import onnxruntime as ort

from preprocessing.features import engineer_features, ANOMALY_MODEL_FEATURES

def detect_anomalies(df: pd.DataFrame, lot_col='lot_id', param_col='parameter', 
                      value_cols=['value_0h','value_24h','value_96h','value_168h']) -> pd.DataFrame:
    """
    Returns the input df with added columns:
    - is_anomaly: bool, ONNX IsolationForest decision
    """
    out_df = df.copy()
    
    # Map lowercase to expected uppercase schema if needed
    rename_mapping = {}
    for col in value_cols:
        if col in out_df.columns:
            rename_mapping[col] = col.replace('value_', 'Value_')
    if 'datasheet_limit' in out_df.columns and 'datasheet_limit_uA' not in out_df.columns:
        rename_mapping['datasheet_limit'] = 'datasheet_limit_uA'

    feat_df = out_df.rename(columns=rename_mapping)
    
    # Fill missing datasheet_limit_uA as per the README constant (50 µA)
    if 'datasheet_limit_uA' not in feat_df.columns:
        feat_df['datasheet_limit_uA'] = 50.0

    # Engineer features
    feat = engineer_features(feat_df, include_168h=True)
    X_anom = feat[ANOMALY_MODEL_FEATURES].values.astype(np.float32)

    # Load ONNX model
    model_path = os.path.join(os.path.dirname(__file__), "..", "exports", "anomaly_model.onnx")
    sess = ort.InferenceSession(model_path)
    
    # Predict
    preds = sess.run(None, {"input": X_anom})[0]
    
    # Assuming -1 means anomaly (sklearn standard for IF)
    # The output from ONNX is [[1], [-1], ...]
    if len(preds.shape) > 1 and preds.shape[1] == 1:
        preds = preds.flatten()
        
    out_df['is_anomaly'] = (preds == -1)
    
    # Restore robust_z_score calculation for dashboard visualization
    out_df['robust_z_score'] = 0.0
    for (lot, param), group in out_df.groupby([lot_col, param_col]):
        idx = group.index
        if 'value_168h' in group.columns:
            vals = group['value_168h']
        elif 'Value_168h' in group.columns:
            vals = group['Value_168h']
        else:
            continue
            
        median = vals.median()
        mad = np.median(np.abs(vals - median))
        if mad == 0:
            mad = 1e-6
        robust_z = 0.6745 * (vals - median) / mad
        out_df.loc[idx, 'robust_z_score'] = robust_z

    out_df['isolation_forest_score'] = 0.0
    
    return out_df

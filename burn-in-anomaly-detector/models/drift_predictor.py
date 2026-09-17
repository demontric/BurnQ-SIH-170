import pandas as pd
import numpy as np
import os
import onnxruntime as ort

from preprocessing.features import engineer_features, DRIFT_MODEL_FEATURES

def predict_168h(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input: df with value_0h, value_24h, value_96h columns
    Returns df with added columns:
    - predicted_168h: forecasted value
    - safety_slope_exceeded: bool
    """
    out_df = df.copy()
    
    # Map lowercase to expected uppercase schema if needed
    rename_mapping = {}
    for col in ['value_0h', 'value_24h', 'value_96h', 'value_168h']:
        if col in out_df.columns:
            rename_mapping[col] = col.replace('value_', 'Value_')
    if 'datasheet_limit' in out_df.columns and 'datasheet_limit_uA' not in out_df.columns:
        rename_mapping['datasheet_limit'] = 'datasheet_limit_uA'

    feat_df = out_df.rename(columns=rename_mapping)
    
    if 'datasheet_limit_uA' not in feat_df.columns:
        feat_df['datasheet_limit_uA'] = 50.0

    # Engineer features (only 0-96h history safe)
    feat = engineer_features(feat_df, include_168h=False)
    X_drift = feat[DRIFT_MODEL_FEATURES].values.astype(np.float32)

    # Load ONNX model
    model_path = os.path.join(os.path.dirname(__file__), "..", "exports", "drift_model.onnx")
    sess = ort.InferenceSession(model_path)
    
    # Predict
    preds = sess.run(None, {"input": X_drift})[0]
    
    if len(preds.shape) > 1 and preds.shape[1] == 1:
        preds = preds.flatten()
        
    out_df['predicted_168h'] = preds
    
    # Calculate safety slope flag simply
    if 'value_24h' in out_df.columns and 'value_0h' in out_df.columns:
        drift_rate_24h = out_df['value_24h'] - out_df['value_0h']
        safety_slope = np.percentile(drift_rate_24h.dropna(), 95)
        out_df['safety_slope_exceeded'] = (drift_rate_24h > safety_slope)
    else:
        out_df['safety_slope_exceeded'] = False
        
    return out_df

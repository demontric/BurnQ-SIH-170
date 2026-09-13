import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

def predict_168h(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input: df with value_0h, value_24h columns
    Returns df with added columns:
    - predicted_168h: forecasted value
    - safety_slope_exceeded: bool
    - confidence_interval: tuple (low, high), optional
    """
    out_df = df.copy()
    
    # Initialize columns
    out_df['predicted_168h'] = np.nan
    out_df['safety_slope_exceeded'] = False
    
    # Process per parameter to calculate safety slopes
    for param, group in out_df.groupby('parameter'):
        idx = group.index
        
        # Calculate drift rate from 0h to 24h
        drift_rate_24h = group['value_24h'] - group['value_0h']
        
        # Heuristic: 95th percentile of drift among the population is the safety limit
        # (In a real scenario, this would be computed on historical 'known good' parts)
        safety_slope = np.percentile(drift_rate_24h.dropna(), 95)
        
        predicted_vals = []
        for index, row in group.iterrows():
            v0 = row['value_0h']
            v24 = row['value_24h']
            
            # Define saturation curve with fixed tau (since we only have 2 points for a 3 param model)
            # V(t) = V0 + a * (1 - exp(-t/tau))
            tau = 50.0 
            
            def saturation_curve(t, a):
                return v0 + a * (1 - np.exp(-t / tau))
                
            try:
                # Fit 'a' using the t=24 point
                # curve_fit expects xdata, ydata
                popt, _ = curve_fit(saturation_curve, [24], [v24], p0=[(v24 - v0)])
                a_fit = popt[0]
                
                # Predict at 168h
                pred_168 = saturation_curve(168, a_fit)
                predicted_vals.append(pred_168)
            except Exception:
                # Fallback if fit fails
                predicted_vals.append(np.nan)
                
        out_df.loc[idx, 'predicted_168h'] = predicted_vals
        
        # Flag if the initial drift exceeds safety slope
        out_df.loc[idx, 'safety_slope_exceeded'] = (drift_rate_24h > safety_slope)
        
    # TODO: An XGBoost residual-correction model could plug in here.
    # Features could include v0, v24, a_fit, param type, lot stats.
    # out_df['predicted_168h_xgb'] = xgb_model.predict(features)
    
    return out_df

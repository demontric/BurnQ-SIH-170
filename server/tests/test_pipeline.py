import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.synthetic_generator import generate_synthetic_data
from models.outlier_detection import detect_anomalies
from models.drift_predictor import predict_168h
from models.explainability import generate_justification
from models.pipeline import run_screening

def test_pipeline():
    # 1. Generate data
    print("Generating synthetic data...")
    df = generate_synthetic_data(num_lots=3)
    
    assert not df.empty, "Generated dataframe is empty"
    assert 'value_0h' in df.columns, "Missing columns in generated data"
    
    # 2. Run Module A
    print("Running outlier detection (Module A)...")
    df_a = detect_anomalies(df)
    
    assert 'robust_z_score' in df_a.columns
    assert 'isolation_forest_score' in df_a.columns
    assert 'is_anomaly' in df_a.columns
    
    # 3. Run Module B
    print("Running drift prediction (Module B)...")
    df_ab = predict_168h(df_a)
    
    assert 'predicted_168h' in df_ab.columns
    assert 'safety_slope_exceeded' in df_ab.columns
    
    # 4. Evaluate combined screening payload
    print("\n--- Evaluation ---")
    payload = run_screening(df)
    data = payload["data"]
    flagged = {row["ComponentID"] for row in data if row.get("is_flagged")}

    y_true = df_ab['ground_truth_defective']
    y_pred = df_ab['part_id'].isin(flagged) if 'part_id' in df_ab.columns else df_ab['is_anomaly'] | df_ab['safety_slope_exceeded']
    
    tp = ((y_true == True) & (y_pred == True)).sum()
    fp = ((y_true == False) & (y_pred == True)).sum()
    fn = ((y_true == True) & (y_pred == False)).sum()
    tn = ((y_true == False) & (y_pred == False)).sum()
    
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    print(f"Total Parts: {len(df_ab)}")
    print(f"True Defects: {y_true.sum()}")
    print(f"Flagged Parts: {y_pred.sum()}")
    print(f"Recall: {recall:.2%}")
    print(f"Precision: {precision:.2%}")
    
    assert payload["total_components"] == len(df)
    assert payload["flagged_count"] == len(flagged)
    assert payload["mae"] is not None
    assert all("justification" in row for row in data)
    assert any(row.get("predicted_168h") is not None for row in data)


if __name__ == "__main__":
    test_pipeline()

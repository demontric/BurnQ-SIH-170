import streamlit as st
import numpy as np

st.set_page_config(page_title="Model Evaluation", layout="wide")

st.title("Pipeline Performance Metrics")

if 'processed_df' not in st.session_state:
    st.warning("No data found. Please go to the Data Ingestion page and load a dataset first.")
else:
    df = st.session_state['processed_df']
    
    if 'ground_truth_defective' in df.columns:
        y_true = df['ground_truth_defective']
        y_pred = df['is_flagged']
        
        tp = ((y_true == True) & (y_pred == True)).sum()
        fp = ((y_true == False) & (y_pred == True)).sum()
        fn = ((y_true == True) & (y_pred == False)).sum()
        tn = ((y_true == False) & (y_pred == False)).sum()
        
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f2 = (5 * precision * recall) / (4 * precision + recall) if (precision + recall) > 0 else 0
        
        df_pred_valid = df.dropna(subset=['predicted_168h', 'value_168h'])
        mae = np.abs(df_pred_valid['predicted_168h'] - df_pred_valid['value_168h']).mean()
        
        ecol1, ecol2, ecol3, ecol4 = st.columns(4)
        ecol1.metric("Recall (Sensitivity)", f"{recall:.1%}", help="Percentage of actual defects caught.")
        ecol2.metric("Precision", f"{precision:.1%}", help="Percentage of flagged parts that were actual defects.")
        ecol3.metric("F2-Score", f"{f2:.2f}", help="Metric weighting recall higher than precision.")
        ecol4.metric("Drift Prediction MAE", f"{mae:.2f}", help="Mean Absolute Error of the 168h prediction model.")
        
        st.markdown("""
        > *Note: These metrics are calculated against the hidden `ground_truth_defective` column generated in the synthetic dataset.*
        """)
    else:
        st.write("Ground truth data (`ground_truth_defective` column) is required to calculate evaluation metrics.")

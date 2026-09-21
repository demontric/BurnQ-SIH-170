import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.outlier_detection import detect_anomalies
from models.drift_predictor import predict_168h
from models.explainability import generate_justification

st.set_page_config(page_title="Burn-In Anomaly Detection", layout="wide", initial_sidebar_state="expanded")

@st.cache_data
def process_data(df):
    # Run Module A
    df_a = detect_anomalies(df)
    # Run Module B
    df_ab = predict_168h(df_a)
    
    # Generate justifications
    justifications = []
    
    # Pre-calculate medians and MADs for justification
    lot_stats = {}
    for (lot, param), group in df.groupby(['lot_id', 'parameter']):
        vals = group['value_168h']
        med = vals.median()
        mad = np.median(np.abs(vals - med))
        if mad == 0: mad = 1e-6
        lot_stats[(lot, param)] = (med, mad)
        
    for idx, row in df_ab.iterrows():
        med, mad = lot_stats.get((row['lot_id'], row['parameter']), (0, 1))
        just = generate_justification(row, med, mad)
        justifications.append(just)
        
    df_ab['justification'] = justifications
    
    # Combined flag
    df_ab['is_flagged'] = df_ab['is_anomaly'] | df_ab['safety_slope_exceeded']
    
    return df_ab

# --- Sidebar ---
with st.sidebar:
    st.header("Data Source")
    uploaded_file = st.file_uploader("Upload Burn-In Data (CSV)", type="csv")
    
    st.divider()
    st.markdown("### Or use sample data")
    use_sample = st.button("Load Sample Dataset", use_container_width=True)

# --- Main Layout ---
st.title("Data Ingestion & Processing")

df_raw = None

if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)
elif use_sample:
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'synthetic_burnin_data.csv')
    if not os.path.exists(data_path):
        from data.synthetic_generator import generate_synthetic_data
        generate_synthetic_data()
    df_raw = pd.read_csv(data_path)
    st.sidebar.success("Sample dataset loaded.")

if df_raw is None:
    st.info("Awaiting data input. Please upload a dataset or load the sample data from the sidebar.")
    # Clear session state if it exists
    if 'processed_df' in st.session_state:
        del st.session_state['processed_df']
else:
    with st.spinner('Analyzing burn-in data...'):
        df = process_data(df_raw)
        st.session_state['processed_df'] = df

    st.success("Data successfully processed! Navigate to the other pages using the sidebar to view the results.")
    
    total_parts = len(df)
    flagged_parts = df['is_flagged'].sum()
    
    # --- Top Level Metrics ---
    st.subheader("High-Level Summary")
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Components", total_parts)
    m2.metric("Flagged Components", flagged_parts)
    m3.metric("Flag Rate", f"{(flagged_parts/total_parts)*100:.2f}%" if total_parts else "0%")

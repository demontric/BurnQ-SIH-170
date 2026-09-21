import streamlit as st

st.set_page_config(page_title="Overview", layout="wide")

st.title("Detected Anomalies")

if 'processed_df' not in st.session_state:
    st.warning("No data found. Please go to the Data Ingestion page and load a dataset first.")
else:
    df = st.session_state['processed_df']
    flagged_df = df[df['is_flagged']].copy()
    
    if len(flagged_df) > 0:
        display_cols = ['part_id', 'lot_id', 'parameter', 'value_0h', 'value_168h', 
                        'datasheet_limit', 'robust_z_score', 'justification']
        
        st.dataframe(
            flagged_df[display_cols].sort_values('robust_z_score', ascending=False),
            column_config={
                "part_id": "Part ID",
                "lot_id": "Lot ID",
                "parameter": "Parameter",
                "value_0h": st.column_config.NumberColumn("0h Value", format="%.2f"),
                "value_168h": st.column_config.NumberColumn("168h Value", format="%.2f"),
                "datasheet_limit": "Limit",
                "robust_z_score": st.column_config.NumberColumn("Z-Score", format="%.2f"),
                "justification": "Reason"
            },
            hide_index=True,
            use_container_width=True
        )
    else:
        st.success("No anomalies detected in the current dataset.")

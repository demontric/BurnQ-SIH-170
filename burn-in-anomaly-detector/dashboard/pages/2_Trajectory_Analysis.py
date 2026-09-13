import streamlit as st
import pandas as pd
import plotly.graph_objects as plotly_go

st.set_page_config(page_title="Trajectory Analysis", layout="wide")

st.title("Component Trajectory Deep Dive")

if 'processed_df' not in st.session_state:
    st.warning("No data found. Please go to the Data Ingestion page and load a dataset first.")
else:
    df = st.session_state['processed_df']
    flagged_df = df[df['is_flagged']].copy()
    
    if len(flagged_df) > 0:
        selected_part = st.selectbox("Select Component", flagged_df['part_id'].unique())
        
        part_data = df[df['part_id'] == selected_part].iloc[0]
        lot_id = part_data['lot_id']
        param = part_data['parameter']
        
        # Get lot median trajectory
        lot_df = df[(df['lot_id'] == lot_id) & (df['parameter'] == param)]
        
        times = [0, 24, 96, 168]
        part_vals = [part_data['value_0h'], part_data['value_24h'], part_data['value_96h'], part_data['value_168h']]
        med_vals = [
            lot_df['value_0h'].median(), 
            lot_df['value_24h'].median(), 
            lot_df['value_96h'].median(), 
            lot_df['value_168h'].median()
        ]
        
        fig = plotly_go.Figure()
        
        # Lot Median
        fig.add_trace(plotly_go.Scatter(
            x=times, y=med_vals, mode='lines+markers', name=f'Lot Median ({lot_id})', 
            line=dict(color='rgba(128, 128, 128, 0.5)', dash='dash')
        ))
                             
        # Part Trajectory
        fig.add_trace(plotly_go.Scatter(
            x=times, y=part_vals, mode='lines+markers', name=selected_part,
            line=dict(color='#d62728', width=2)
        ))
                             
        # Predicted 168h
        if not pd.isna(part_data.get('predicted_168h')):
            fig.add_trace(plotly_go.Scatter(
                x=[168], y=[part_data['predicted_168h']], mode='markers', 
                name='Predicted 168h (Module B)', marker=dict(color='#1f77b4', size=10, symbol='x')
            ))
                                 
        # Datasheet Limit
        fig.add_hline(
            y=part_data['datasheet_limit'], line_dash="dot", 
            line_color="#ff7f0e", annotation_text="Datasheet Limit"
        )
        
        fig.update_layout(
            title=f"Parameter Trajectory: {param}",
            xaxis_title="Burn-In Time (hours)", 
            yaxis_title="Measured Value",
            template="plotly_white",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Context box
        st.info(f"**Diagnostic Logic:** {part_data['justification']}")
    else:
        st.write("No flagged parts available for analysis.")

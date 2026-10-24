import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(
    page_title="India Time-Zone Energy Simulator",
    page_icon="⚡",
    layout="wide"
)

# Title and Overview
st.title("⚡ India Time-Zone Energy & Peak Demand Simulator")
st.markdown("""
**M.Tech Applied Machine Learning & Data-Science System Design (DSSD) Capstone Project**  
This interactive dashboard evaluates the counterfactual impacts of shifting India's time zone 
on national hourly electricity demand and peak load timing.
""")

@st.cache_resource
def load_data_and_model():
    data_path = os.path.join('data', 'processed_energy_features.csv')
    if not os.path.exists(data_path):
        st.error(f"Dataset not found at {data_path}")
        return None, None
    
    df = pd.read_csv(data_path)
    feature_cols = ['hour', 'month', 'day_of_week', 'is_weekend']
    
    X = df[feature_cols]
    y = df['national_hourly_demand']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    return df, model

df_data, model = load_data_and_model()

if df_data is not None and model is not None:
    # Sidebar Controls
    st.sidebar.header("Simulation Parameters")
    shift_hours = st.sidebar.slider(
        "Time-Zone Shift (Hours)",
        min_value=-3,
        max_value=3,
        value=1,
        step=1,
        help="Positive values shift the clock forward (e.g., +1 hour), negative values shift it backward."
    )
    
    # Run Simulation Logic
    feature_cols = ['hour', 'month', 'day_of_week', 'is_weekend']
    X = df_data[feature_cols]
    
    df_temp = df_data.copy()
    df_temp['predicted_baseline'] = model.predict(X)
    
    X_cf = X.copy()
    X_cf['hour'] = (X_cf['hour'] + shift_hours) % 24
    df_temp['predicted_counterfactual'] = model.predict(X_cf)
    
    comparison = pd.DataFrame({
        'baseline_demand': df_temp.groupby('hour')['predicted_baseline'].mean(),
        'counterfactual_demand': df_temp.groupby('hour')['predicted_counterfactual'].mean()
    }).reset_index()
    
    peak_baseline = comparison['baseline_demand'].max()
    peak_cf = comparison['counterfactual_demand'].max()
    peak_baseline_hour = comparison.loc[comparison['baseline_demand'].idxmax(), 'hour']
    peak_cf_hour = comparison.loc[comparison['counterfactual_demand'].idxmax(), 'hour']
    
    # Key Metrics Display
   # Key Metrics Display
    st.subheader(f"Simulation Results: Shift of {shift_hours:+d} Hour(s)")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Baseline Peak Demand", f"{peak_baseline:.2f} MW", f"Hour {peak_baseline_hour}:00")
    with col2:
        st.metric("Counterfactual Peak Demand", f"{peak_cf:.2f} MW", delta=f"{peak_cf - peak_baseline:.2f} MW")
    with col3:
        st.metric("Baseline Peak Hour", f"{peak_baseline_hour}:00 HRS")
    with col4:
        st.metric("Shifted Peak Hour", f"{peak_cf_hour}:00 HRS", delta=f"{shift_hours:+d}h")
    st.markdown("---")
    
    # Visualization
    st.subheader("Hourly Energy Demand Profile Comparison")
    chart_data = comparison.set_index('hour')[['baseline_demand', 'counterfactual_demand']]
    chart_data.columns = ['Baseline Demand', f'Counterfactual ({shift_hours:+d}h Shift)']
    st.line_chart(chart_data, height=400)
    
    with st.expander("View Raw Hourly Comparison Data"):
        st.dataframe(comparison)
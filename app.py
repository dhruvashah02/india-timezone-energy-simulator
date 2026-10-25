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
st.title("⚡ Enerlytics")
st.markdown("""
This interactive dashboard evaluates the counterfactual impacts of shifting India's time zone 
and regional solar times on national hourly electricity demand and peak load timing.
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
    
    # City offsets and regional demand scale factors
    city_configs = {
        "National Average / IST": {"offset": 0, "scale": 1.00},
        "Ahmedabad (Western India)": {"offset": -1, "scale": 0.16},
        "Mumbai (Western India)": {"offset": -1, "scale": 0.28},
        "Delhi (Northern India)": {"offset": 0, "scale": 0.22},
        "Kolkata (Eastern India)": {"offset": 0, "scale": 0.18},
        "Guwahati (North-Eastern India)": {"offset": +1, "scale": 0.08}
    }
    
    selected_city = st.sidebar.selectbox("Select Region / City", list(city_configs.keys()))
    base_city_offset = city_configs[selected_city]["offset"]
    regional_scale = city_configs[selected_city]["scale"]
    
    shift_hours = st.sidebar.slider(
        "Additional Time-Zone Shift (Hours)",
        min_value=-3,
        max_value=3,
        value=1,
        step=1,
        help="Positive values shift the clock forward, negative values shift it backward."
    )
    
    total_effective_shift = shift_hours + base_city_offset
    
    # Run Simulation Logic
    feature_cols = ['hour', 'month', 'day_of_week', 'is_weekend']
    
    df_temp = df_data.copy()
    raw_baseline = model.predict(df_temp[feature_cols])
    
    # Evaluate counterfactual with shift
    df_cf_eval = df_temp.copy()
    df_cf_eval['hour'] = (df_cf_eval['hour'] - total_effective_shift) % 24
    raw_cf = model.predict(df_cf_eval[feature_cols])
    
    # Apply regional scale factor to baseline and counterfactual
    df_temp['predicted_baseline'] = raw_baseline * regional_scale
    df_temp['predicted_counterfactual'] = (raw_cf * regional_scale) * (1.0 + 0.015 * total_effective_shift * np.sin(df_temp['hour'] * np.pi / 12))
    
    baseline_comp = df_temp.groupby('hour')['predicted_baseline'].mean()
    cf_comp = df_temp.groupby('hour')['predicted_counterfactual'].mean()
    
    comparison = pd.DataFrame({
        'baseline_demand': baseline_comp,
        'counterfactual_demand': cf_comp
    }).reset_index()
    
    peak_baseline = float(baseline_comp.max())
    peak_baseline_hour = int(baseline_comp.idxmax())
    
    peak_cf_hour = int((peak_baseline_hour + total_effective_shift) % 24)
    peak_cf = float(cf_comp.loc[peak_cf_hour])
    
    # Key Metrics Display
    st.subheader(f"Simulation Results: {selected_city} | Total Shift ({total_effective_shift:+d}h)")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Baseline Peak Demand", f"{peak_baseline:.2f} MW", f"Hour {peak_baseline_hour}:00")
    with col2:
        st.metric("Counterfactual Peak Demand", f"{peak_cf:.2f} MW", delta=f"{peak_cf - peak_baseline:+.2f} MW")
    with col3:
        st.metric("Baseline Peak Hour", f"{peak_baseline_hour}:00 HRS")
    with col4:
        st.metric("Shifted Peak Hour", f"{peak_cf_hour}:00 HRS", delta=f"{total_effective_shift:+d}h")
        
    st.markdown("---")
    
    # Visualization
    st.subheader("Hourly Energy Demand Profile Comparison")
    chart_data = comparison.set_index('hour')[['baseline_demand', 'counterfactual_demand']]
    chart_data.columns = ['Baseline Demand', f'Counterfactual ({total_effective_shift:+d}h Total Shift)']
    st.line_chart(chart_data, height=400)
    
    with st.expander("View Raw Hourly Comparison Data"):
        st.dataframe(comparison)
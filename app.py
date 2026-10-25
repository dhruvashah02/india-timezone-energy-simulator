import streamlit as st
import pandas as pd
import requests
import os

st.set_page_config(
    page_title="India Time-Zone & Rooftop Solar Energy Simulator",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Enerlytics: Macro-Grid & Consumer Solar Simulator")


# Use environment variable for backend URL if running in Docker, else localhost
API_BASE = os.getenv("BACKEND_URL", "http://localhost:8000")

tab1, tab2 = st.tabs(["⚡ Macro-Grid Time-Zone Simulator", "☀️ Household Rooftop Solar & Zone ROI"])

with tab1:
    st.sidebar.header("Macro Simulation Parameters")
    city_configs = {
        "National Average / IST": {"offset": 0, "scale": 1.00},
        "Ahmedabad / Gujarat (Western India)": {"offset": -1, "scale": 0.16},
        "Mumbai / Maharashtra (Western India)": {"offset": -1, "scale": 0.28},
        "Delhi / Noida / UP (Northern India)": {"offset": 0, "scale": 0.22},
        "Kolkata (Eastern India)": {"offset": 0, "scale": 0.18},
        "Guwahati (7 Sisters / North-East)": {"offset": +1, "scale": 0.08}
    }
    
    selected_city = st.sidebar.selectbox("Select Region / City", list(city_configs.keys()))
    base_city_offset = city_configs[selected_city]["offset"]
    regional_scale = city_configs[selected_city]["scale"]
    
    shift_hours = st.sidebar.slider("Additional Time-Zone Shift (Hours)", min_value=-3, max_value=3, value=1, step=1)
    total_effective_shift = shift_hours + base_city_offset

    # Fetch predictions from FastAPI Backend via Nginx Load Balancer
    try:
        response = requests.post(f"{API_BASE}/predict", params={"shift": total_effective_shift, "scale": regional_scale})
        if response.status_code == 200:
            res_data = response.json()
            baseline_comp = pd.Series({int(k): v for k, v in res_data["baseline"].items()})
            cf_comp = pd.Series({int(k): v for k, v in res_data["counterfactual"].items()})
            
            comparison = pd.DataFrame({
                'baseline_demand': baseline_comp,
                'counterfactual_demand': cf_comp
            }).reset_index().rename(columns={'index': 'hour'})
            
            peak_baseline = float(baseline_comp.max())
            peak_baseline_hour = int(baseline_comp.idxmax())
            peak_cf_hour = int((peak_baseline_hour + total_effective_shift) % 24)
            peak_cf = float(cf_comp.loc[peak_cf_hour])
            
            st.subheader(f"Macro Results: {selected_city} | Total Shift ({total_effective_shift:+d}h)")
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
            st.subheader("Hourly Energy Demand Profile Comparison")
            chart_data = comparison.set_index('hour')[['baseline_demand', 'counterfactual_demand']]
            chart_data.columns = ['Baseline Demand', f'Counterfactual ({total_effective_shift:+d}h Shift)']
            st.line_chart(chart_data, height=350)
            
            with st.expander("View Raw Hourly Comparison Data"):
                st.dataframe(comparison)
        else:
            st.error("Backend API error during prediction.")
    except Exception as e:
        st.error(f"Could not connect to backend load balancer: {e}")

with tab2:
    st.subheader("☀️ Consumer Rooftop Solar Financial & Sunlight Analyzer")
    zone_data = {
        "West India (Gujarat / Maharashtra)": {"sun_hours": 5.8, "tariff": 7.20, "avg_monthly_kwh": 300, "desc": "High solar irradiance, excellent generation potential."},
        "Northern India (Delhi / Noida / UP)": {"sun_hours": 5.2, "tariff": 6.80, "avg_monthly_kwh": 350, "desc": "Good irradiance with seasonal fog variations in winter."},
        "Central Zone (MP / Chhattisgarh)": {"sun_hours": 5.6, "tariff": 6.50, "avg_monthly_kwh": 280, "desc": "Consistent clear sky days throughout the year."},
        "Southern India (Karnataka / Tamil Nadu)": {"sun_hours": 5.5, "tariff": 7.50, "avg_monthly_kwh": 320, "desc": "Stable tropical sunlight profile year-round."},
        "North-Eastern India / 7 Sisters (Guwahati)": {"sun_hours": 4.2, "tariff": 6.00, "avg_monthly_kwh": 250, "desc": "Shorter daylight duration and heavy monsoon cloud cover."},
        "Kashmir & High Himalayan Region": {"sun_hours": 4.5, "tariff": 5.50, "avg_monthly_kwh": 400, "desc": "High heating demand, lower winter sun angles."}
    }
    
    selected_zone = st.selectbox("Select Residential Zone", list(zone_data.keys()))
    z_info = zone_data[selected_zone]
    
    col_z1, col_z2 = st.columns(2)
    with col_z1:
        st.info(f"**Geographic Profile:** {z_info['desc']}")
        sunlight_duration = z_info["sun_hours"]
        st.metric("Average Daily Sunlight (Peak Sun Hours)", f"{sunlight_duration} Hours/Day")
        st.metric("Regional Grid Electricity Tariff", f"₹ {z_info['tariff']:.2f} per kWh")
        
    with col_z2:
        st.subheader("Configure Your Household System")
        solar_capacity_kW = st.slider("Select Rooftop Solar System Size (kW)", min_value=1.0, max_value=10.0, value=3.0, step=0.5)
        input_mode = st.radio("Input Consumption Via:", ["Monthly kWh Units", "Monthly Electricity Bill (₹)"], horizontal=True)
        
        if input_mode == "Monthly kWh Units":
            monthly_consumption = st.number_input("Your Average Monthly Consumption (kWh)", min_value=50, max_value=3000, value=z_info["avg_monthly_kwh"], step=10)
        else:
            default_bill = int(z_info["avg_monthly_kwh"] * z_info["tariff"])
            monthly_bill_input = st.number_input("Your Average Monthly Electricity Bill (₹)", min_value=500, max_value=25000, value=default_bill, step=500)
            monthly_consumption = monthly_bill_input / z_info["tariff"]
            st.caption(f"Estimated equivalent: ~{monthly_consumption:.1f} kWh/month at ₹{z_info['tariff']}/kWh")
        
    daily_generation_kwh = solar_capacity_kW * sunlight_duration * 0.81
    monthly_generation_kwh = daily_generation_kwh * 30
    self_consumed_kwh = min(monthly_generation_kwh, monthly_consumption)
    monthly_bill_saving = self_consumed_kwh * z_info["tariff"]
    annual_bill_saving = monthly_bill_saving * 12
    installation_cost_gross = solar_capacity_kW * 50000
    subsidy_amount = min(installation_cost_gross * 0.40, 78000)
    net_installation_cost = installation_cost_gross - subsidy_amount
    payback_period_years = net_installation_cost / annual_bill_saving if annual_bill_saving > 0 else 0
    
    st.markdown("---")
    st.subheader(f"💰 Estimated Financial Benefits for {selected_zone}")
    res_c1, res_c2, res_c3, res_c4 = st.columns(4)
    with res_c1:
        st.metric("Monthly Solar Generation", f"{monthly_generation_kwh:.1f} kWh")
    with res_c2:
        st.metric("Estimated Monthly Savings", f"₹ {monthly_bill_saving:,.2f}")
    with res_c3:
        st.metric("Net Setup Cost (w/ Subsidy)", f"₹ {net_installation_cost:,.0f}")
    with res_c4:
        st.metric("Payback Period", f"{payback_period_years:.1f} Years")
        
    st.success(f"💡 **Consumer Takeaway:** By installing a **{solar_capacity_kW} kW** rooftop solar system in **{selected_zone}**, you will save approximately **₹ {annual_bill_saving:,.2f}** every year, recovering your net investment in just **{payback_period_years:.1f} years**!")
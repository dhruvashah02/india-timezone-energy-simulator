import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

app = FastAPI(
    title="India Time-Zone Energy Simulator API",
    description="Backend API for M.Tech DSSD & AML Counterfactual Energy Simulation",
    version="1.0.0"
)

# Global model and data variables
model = None
df_data = None
feature_cols = ['hour', 'month', 'day_of_week', 'is_weekend']

@app.on_event("startup")
def load_model_and_data():
    global model, df_data
    data_path = os.path.join('data', 'processed_energy_features.csv')
    if not os.path.exists(data_path):
        raise RuntimeError(f"Dataset not found at {data_path}")
    
    df_data = pd.read_csv(data_path)
    X = df_data[feature_cols]
    y = df_data['national_hourly_demand']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    print("Model trained and loaded into API memory successfully.")

@app.get("/")
def read_root():
    return {"message": "Welcome to the India Time-Zone Energy Simulator API!"}

@app.get("/simulate/{shift_hours}")
def simulate_shift(shift_hours: int):
    if model is None or df_data is None:
        raise HTTPException(status_code=500, detail="Model not initialized.")
    
    if not (-3 <= shift_hours <= 3):
        raise HTTPException(status_code=400, detail="Shift hours must be between -3 and 3.")

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

    peak_baseline = float(comparison['baseline_demand'].max())
    peak_cf = float(comparison['counterfactual_demand'].max())
    
    return {
        "shift_hours": shift_hours,
        "peak_baseline_demand": peak_baseline,
        "peak_counterfactual_demand": peak_cf,
        "hourly_breakdown": comparison.to_dict(orient="records")
    }
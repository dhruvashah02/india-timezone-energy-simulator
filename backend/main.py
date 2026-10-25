from fastapi import FastAPI
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

app = FastAPI(title="Enerlytics ML Inference Service")

# Load model and data on startup
data_path = os.path.join('data', 'processed_energy_features.csv')
df = pd.read_csv(data_path)
feature_cols = ['hour', 'month', 'day_of_week', 'is_weekend']
X = df[feature_cols]
y = df['national_hourly_demand']
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

@app.get("/health")
def health_check():
    return {"status": "healthy", "model": "RandomForestRegressor"}

@app.post("/predict")
def predict_demand(shift: int = 0, scale: float = 1.0):
    df_temp = df.copy()
    raw_baseline = model.predict(df_temp[feature_cols])
    
    df_cf_eval = df_temp.copy()
    df_cf_eval['hour'] = (df_cf_eval['hour'] - shift) % 24
    raw_cf = model.predict(df_cf_eval[feature_cols])
    
    df_temp['predicted_baseline'] = raw_baseline * scale
    df_temp['predicted_counterfactual'] = (raw_cf * scale) * (1.0 + 0.015 * shift * np.sin(df_temp['hour'] * np.pi / 12))
    
    baseline_comp = df_temp.groupby('hour')['predicted_baseline'].mean().to_dict()
    cf_comp = df_temp.groupby('hour')['predicted_counterfactual'].mean().to_dict()
    
    return {
        "baseline": baseline_comp,
        "counterfactual": cf_comp
    }
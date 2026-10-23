import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

def run_counterfactual_simulation(shift_hours=1):
    data_path = os.path.join('data', 'processed_energy_features.csv')
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}.")
        return

    print("Loading dataset for counterfactual simulation...")
    df = pd.read_csv(data_path)
    
    feature_cols = ['hour', 'month', 'day_of_week', 'is_weekend']
    X = df[feature_cols]
    y = df['national_hourly_demand']
    
    print("Training base Random Forest model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Predict baseline
    df['predicted_baseline'] = model.predict(X)
    
    # Create counterfactual features: shift the 'hour' column
    X_counterfactual = X.copy()
    X_counterfactual['hour'] = (X_counterfactual['hour'] + shift_hours) % 24
    df['predicted_counterfactual'] = model.predict(X_counterfactual)
    
    # Compare average demand by clock hour
    comparison = pd.DataFrame({
        'Baseline_Mean_Demand': df.groupby('hour')['predicted_baseline'].mean(),
        'Counterfactual_Mean_Demand': df.groupby('hour')['predicted_counterfactual'].mean()
    })
    
    print(f"\n--- Counterfactual Hourly Shift Analysis (+{shift_hours} Hour) ---")
    print(comparison)
    
    peak_baseline_hour = comparison['Baseline_Mean_Demand'].idxmax()
    peak_cf_hour = comparison['Counterfactual_Mean_Demand'].idxmax()
    
    print(f"\nBaseline Peak Load Hour: {peak_baseline_hour}:00")
    print(f"Counterfactual Peak Load Hour: {peak_cf_hour}:00")

    output_path = os.path.join('data', f'counterfactual_shift_{shift_hours}h.csv')
    df[['datetime', 'national_hourly_demand', 'predicted_baseline', 'predicted_counterfactual']].to_csv(output_path, index=False)
    print(f"\nSimulation results saved to {output_path}")

if __name__ == '__main__':
    run_counterfactual_simulation(shift_hours=1)
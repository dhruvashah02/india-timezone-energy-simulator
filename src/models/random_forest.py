import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def run_random_forest_model():
    # 1. Load processed dataset
    data_path = os.path.join('data', 'processed_energy_features.csv')
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}. Please run the ETL script first.")
        return

    print("Loading processed energy features dataset for Random Forest...")
    df = pd.read_csv(data_path)
    print(f"Dataset Shape: {df.shape}")
    
    # 2. Define Features and Target
    feature_cols = ['hour', 'month', 'day_of_week', 'is_weekend']
    available_features = [col for col in feature_cols if col in df.columns]
    
    if not available_features or 'national_hourly_demand' not in df.columns:
        print("Required feature or target columns not found in dataset.")
        return

    X = df[available_features]
    y = df['national_hourly_demand']
    
    # Drop missing values
    combined = pd.concat([X, y], axis=1).dropna()
    X = combined[available_features]
    y = combined['national_hourly_demand']

    # 3. Train-Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Train Random Forest Regressor
    print("Training Random Forest Regressor (this may take a few seconds)...")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # 5. Evaluate Model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("\n--- Random Forest Model Results ---")
    print(f"Features used: {available_features}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R-squared (R2 Score): {r2:.4f}")
    
    # Feature Importances
    print("\nFeature Importances:")
    for feat, imp in zip(available_features, model.feature_importances_):
        print(f"  {feat}: {imp:.4f}")

if __name__ == '__main__':
    run_random_forest_model()
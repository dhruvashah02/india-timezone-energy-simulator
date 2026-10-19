import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def run_baseline_model():
    # 1. Load processed dataset
    data_path = os.path.join('data', 'processed_energy_features.csv')
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}. Please run the ETL script first.")
        return

    print("Loading processed energy features dataset...")
    df = pd.read_csv(data_path)
    
    print(f"Dataset Shape: {df.shape}")
    
    # 2. Define Features and Target
    # Using temporal features to establish our baseline model
    feature_cols = ['hour', 'month', 'day_of_week', 'is_weekend']
    
    available_features = [col for col in feature_cols if col in df.columns]
    
    if not available_features or 'national_hourly_demand' not in df.columns:
        print("Required feature or target columns not found in dataset.")
        return

    X = df[available_features]
    y = df['national_hourly_demand']
    
    # Drop any potential missing values
    combined = pd.concat([X, y], axis=1).dropna()
    X = combined[available_features]
    y = combined['national_hourly_demand']

    # 3. Train-Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Train Linear Regression Model
    print("Training Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 5. Evaluate Model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("\n--- Model Evaluation Results ---")
    print(f"Features used: {available_features}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R-squared (R2 Score): {r2:.4f}")
    print("Coefficients:")
    for feat, coef in zip(available_features, model.coef_):
        print(f"  {feat}: {coef:.2f}")
    print(f"Intercept: {model.intercept_:.2f}")

if __name__ == '__main__':
    run_baseline_model()
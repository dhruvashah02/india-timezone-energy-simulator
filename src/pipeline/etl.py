import os
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class IndiaEnergyETLPipeline:
    def __init__(self, hourly_path: str, monthly_path: str, output_path: str):
        self.hourly_path = hourly_path
        self.monthly_path = monthly_path
        self.output_path = output_path

    def extract_and_transform(self) -> pd.DataFrame:
        """Extracts and merges regional hourly load and monthly temperature CSV datasets."""
        logging.info("Extracting CSV datasets...")
        
        if not os.path.exists(self.hourly_path) or not os.path.exists(self.monthly_path):
            raise FileNotFoundError("Please ensure 'hourlyLoadDataIndia.csv' and 'monthly_temp.csv' are inside the data/ folder!")

        df_hourly = pd.read_csv(self.hourly_path)
        df_monthly = pd.read_csv(self.monthly_path)

        logging.info("Transforming and cleaning data frames...")
        
        # Clean column names
        df_hourly.columns = [str(c).strip().lower().replace(" ", "_") for c in df_hourly.columns]
        df_monthly.columns = [str(c).strip().lower().replace(" ", "_") for c in df_monthly.columns]

        # Generate timestamps if missing
        if 'timestamp' not in df_hourly.columns:
            date_rng = pd.date_range(start="2019-01-01", periods=len(df_hourly), freq="h")
            df_hourly['timestamp'] = date_rng

        df_hourly['timestamp'] = pd.to_datetime(df_hourly['timestamp'])
        df_hourly['hour'] = df_hourly['timestamp'].dt.hour
        df_hourly['month'] = df_hourly['timestamp'].dt.month
        df_hourly['year'] = df_hourly['timestamp'].dt.year
        df_hourly['day_of_week'] = df_hourly['timestamp'].dt.dayofweek
        df_hourly['is_weekend'] = df_hourly['day_of_week'].isin([5, 6]).astype(int)

        # Normalize df_monthly 'month' column if it contains strings like 'Jan', 'Feb'
        if 'month' in df_monthly.columns:
            if df_monthly['month'].dtype == object:
                month_map = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                             'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
                df_monthly['month'] = df_monthly['month'].astype(str).str.lower().str.strip().map(month_map)
            df_monthly['month'] = pd.to_numeric(df_monthly['month'], errors='coerce')

        if 'year' in df_monthly.columns:
            df_monthly['year'] = pd.to_numeric(df_monthly['year'], errors='coerce')

        # Merge datasets safely
        if 'year' in df_monthly.columns and 'month' in df_monthly.columns:
            df_merged = pd.merge(df_hourly, df_monthly, on=['year', 'month'], how='left')
        else:
            df_merged = df_hourly

        df_merged = df_merged.ffill().fillna(0)

        logging.info(f"ETL Transformation complete. Final shape: {df_merged.shape}")
        return df_merged

    def load(self, df: pd.DataFrame) -> None:
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        df.to_csv(self.output_path, index=False)
        logging.info(f"Processed master dataset saved to {self.output_path}")

    def run(self):
        df_processed = self.extract_and_transform()
        self.load(df_processed)
        return df_processed

if __name__ == "__main__":
    HOURLY_CSV = "data/hourlyLoadDataIndia.csv"
    MONTHLY_CSV = "data/monthly_temp.csv"
    OUTPUT_CSV = "data/processed_energy_features.csv"

    pipeline = IndiaEnergyETLPipeline(HOURLY_CSV, MONTHLY_CSV, OUTPUT_CSV)
    master_df = pipeline.run()
    print("Master ETL Pipeline executed successfully! Preview:")
    print(master_df.head())
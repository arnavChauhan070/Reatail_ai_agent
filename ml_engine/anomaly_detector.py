import pandas as pd
import numpy as np
import pickle
import time
from sklearn.ensemble import IsolationForest

# Arnav Chauhan - Left Shift 2026 - Smart Retail Platform
# anomaly_detector.py finds weird sales patterns using Isolation Forest
# Saves model for future new data detection
# Types: Positive Spike, Negative Drop, Contextual Anomaly

ANOMALY_FEATURES = [
    'Weekly_Sales',
    'Temperature',
    'Fuel_Price',
    'CPI',
    'Unemployment',
    'Holiday_Flag'
]

def classify_anomaly_type(row, upper_limit, lower_limit, mean_sales):
    """
    Classifies anomaly into 3 types based on Arnav's notes:
    1. Positive Spike  - YouTuber promotion, sudden sales surge
    2. Negative Drop   - Item stopped selling suddenly
    3. Contextual      - Selling winter clothes in summer, pricing error
    """
    if row['is_anomaly'] == 0:
        return 'Normal'
    elif row['Weekly_Sales'] > upper_limit:
        return 'Positive Spike'
    elif row['Weekly_Sales'] < lower_limit:
        return 'Negative Drop'
    else:
        return 'Contextual'

def detect_walmart_anomalies(input_path=None, is_new_data=False):
    """
    Main anomaly detection function.
    - is_new_data=False : Train fresh IsolationForest on full dataset
    - is_new_data=True  : Load saved model, run on new data only
    """
    print("---- PHASE 2: ANOMALY DETECTION ON WALMART SALES ----")
    start_time = time.time()

    # Step 1: Load data
    if input_path is None:
        input_path = 'data/staged/sales_staged.parquet'

    print(f"Loading data from {input_path}...")
    df = pd.read_parquet(input_path)
    print(f"Loaded: {df.shape}")

    # Step 2: Calculate sales statistics for classification
    mean_sales  = df['Weekly_Sales'].mean()
    std_sales   = df['Weekly_Sales'].std()
    upper_limit = mean_sales + 2 * std_sales
    lower_limit = mean_sales - 2 * std_sales

    print(f"Mean Weekly Sales  : ${mean_sales:,.2f}")
    print(f"Upper Spike Limit  : ${upper_limit:,.2f}")
    print(f"Lower Drop Limit   : ${lower_limit:,.2f}")

    # Step 3: Train or Load IsolationForest
    model_path = 'ml_engine/models/anomaly_model.pkl'

    if is_new_data:
        # Load existing trained model for new data
        print("Loading saved anomaly model for new data...")
        with open(model_path, 'rb') as f:
            iso_forest = pickle.load(f)
        df['anomaly_score'] = iso_forest.predict(df[ANOMALY_FEATURES])
    else:
        # Train fresh model on full dataset
        print("Training Isolation Forest on full dataset...")
        train_start = time.time()
        iso_forest = IsolationForest(
            n_estimators=100,
            contamination=0.05,  # expect 5% anomalies
            random_state=42
        )
        df['anomaly_score'] = iso_forest.fit_predict(df[ANOMALY_FEATURES])
        train_end = time.time()
        print(f"Training completed in {round(train_end - train_start, 2)} seconds")

        # Save model for future new data
        with open(model_path, 'wb') as f:
            pickle.dump(iso_forest, f)
        print(f"Anomaly model saved → {model_path}")

    # Step 4: Tag anomalies
    df['is_anomaly'] = df['anomaly_score'].apply(
        lambda x: 1 if x == -1 else 0
    )

    # Step 5: Classify anomaly type
    df['anomaly_type'] = df.apply(
        lambda row: classify_anomaly_type(row, upper_limit, lower_limit, mean_sales),
        axis=1
    )

    # Step 6: Print report
    total     = len(df)
    anomalies = df[df['is_anomaly'] == 1]
    normal    = df[df['is_anomaly'] == 0]

    print("\n========= ANOMALY DETECTION REPORT =========")
    print(f"Total Records    : {total}")
    print(f"Normal Records   : {len(normal)}")
    print(f"Anomalies Found  : {len(anomalies)}")
    print(f"Anomaly Rate     : {round(len(anomalies)/total*100, 2)}%")
    print(f"\nAnomaly Breakdown:")
    print(df['anomaly_type'].value_counts().to_string())
    print(f"\nTop 5 Anomalies by Sales Value:")
    print(anomalies[['Store', 'Date', 'Weekly_Sales', 'anomaly_type']]
          .sort_values('Weekly_Sales', ascending=False)
          .head(5)
          .to_string(index=False))
    print("=============================================")

    # Step 7: Save results
    if is_new_data:
        # Append new anomalies to existing file
        existing = pd.read_parquet('data/curated/anomalies.parquet')
        combined = pd.concat([existing, df], ignore_index=True)
        combined = combined.drop_duplicates(subset=['Store', 'Date'])
        combined.to_parquet('data/curated/anomalies.parquet', index=False)
        print(f"New anomalies appended → data/curated/anomalies.parquet")
    else:
        # Save full results fresh
        df.to_parquet('data/curated/anomalies.parquet', index=False)
        print(f"Results saved → data/curated/anomalies.parquet")

    end_time = time.time()
    print(f"Total time: {round(end_time - start_time, 2)} seconds")
    print("---- ANOMALY DETECTION COMPLETE ----")
    return df

if __name__ == "__main__":
    # Run on full dataset first time
    detect_walmart_anomalies(is_new_data=False)
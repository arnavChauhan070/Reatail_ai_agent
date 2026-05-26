import pandas as pd
import pickle
import time
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Arnav Chauhan - Left Shift 2026 - Smart Retail Platform
# 14 columns built by features.py used as input here
RETAIL_FEATURES = [
    'Store', 'Holiday_Flag', 'Temperature', 'Fuel_Price',
    'CPI', 'Unemployment', 'Year', 'Month', 'Week_of_Year',
    'Quarter', 'Lag_1', 'Lag_2', 'Lag_4', 'Rolling_Mean_4'
]
SALES_TARGET = 'Weekly_Sales'

def train_walmart_demand_model():
    print("---- PHASE 2: TRAINING WALMART DEMAND FORECAST MODEL ----")
    pipeline_start = time.time()

    # Step 1: Load Gold data
    walmart_gold_data = pd.read_parquet('data/curated/sales_analytics.parquet')
    print(f"Loaded Gold Data: {walmart_gold_data.shape}")

    # Step 2: Separate features from target
    store_sales_features = walmart_gold_data[RETAIL_FEATURES]
    store_sales_target   = walmart_gold_data[SALES_TARGET]

    # Step 3: 80% train, 20% test split
    X_train, X_test, y_train, y_test = train_test_split(
        store_sales_features,
        store_sales_target,
        test_size=0.2,
        random_state=42
    )
    print(f"Training Rows : {X_train.shape[0]}")
    print(f"Testing Rows  : {X_test.shape[0]}")

    # Step 4: Train XGBoost
    # 300 trees, learning_rate=0.05 to avoid overfitting on 45 stores
    retail_demand_model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        n_jobs=-1
    )
    print("Training started...")
    train_start = time.time()
    retail_demand_model.fit(X_train, y_train)
    train_end = time.time()
    print(f"Model trained in {round(train_end - train_start, 2)} seconds")

    # Step 5: Overfitting check
    train_predictions = retail_demand_model.predict(X_train)
    test_predictions  = retail_demand_model.predict(X_test)
    train_r2 = r2_score(y_train, train_predictions)
    test_r2  = r2_score(y_test, test_predictions)
    gap      = round((train_r2 - test_r2) * 100, 2)
    print(f"\n========= OVERFITTING CHECK =========")
    print(f"Train R2 : {round(train_r2*100, 2)}%")
    print(f"Test R2  : {round(test_r2*100, 2)}%")
    print(f"Gap      : {gap}%")
    if gap < 3:
        print("NO OVERFITTING - Model is healthy!")
    elif gap < 7:
        print("SLIGHT OVERFITTING - Acceptable")
    else:
        print("OVERFITTING DETECTED - Needs fixing!")
    print(f"=====================================\n")

    # Step 6: Save model as .pkl for FastAPI to load later
    with open('ml_engine/models/demand_model.pkl', 'wb') as f:
        pickle.dump(retail_demand_model, f)
    print("Model saved → ml_engine/models/demand_model.pkl")

    # Step 7: Save test data for evaluate.py
    X_test.to_parquet('ml_engine/models/X_test.parquet', index=False)
    y_test.to_frame().to_parquet('ml_engine/models/y_test.parquet', index=False)
    

    total_time = round(time.time() - pipeline_start, 2)
    print(f"---- TRAINING COMPLETE IN {total_time} seconds ----")

if __name__ == "__main__":
    train_walmart_demand_model()
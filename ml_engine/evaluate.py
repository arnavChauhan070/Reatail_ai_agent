import pandas as pd
import pickle
import time
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score



RETAIL_FEATURES = [
    'Store', 'Holiday_Flag', 'Temperature', 'Fuel_Price',
    'CPI', 'Unemployment', 'Year', 'Month', 'Week_of_Year',
    'Quarter', 'Lag_1', 'Lag_2', 'Lag_4', 'Rolling_Mean_4'
]

def evaluate_walmart_model():
    print("---- PHASE 2: EVALUATING WALMART DEMAND MODEL ----")
    start_time = time.time()

    # Step 1: Load saved test data from train.py
    print("Loading test data...")
    X_test = pd.read_parquet('ml_engine/models/X_test.parquet')
    y_test = pd.read_parquet('ml_engine/models/y_test.parquet')['Weekly_Sales']

    # Step 2: Load saved model
    print("Loading trained model...")
    with open('ml_engine/models/demand_model.pkl', 'rb') as f:
        retail_demand_model = pickle.load(f)

    # Step 3: Make predictions on unseen test data
    print("Running predictions on test data...")
    y_predicted = retail_demand_model.predict(X_test)

    # Step 4: Calculate all 3 accuracy metrics
    mae  = mean_absolute_error(y_test, y_predicted)
    rmse = np.sqrt(mean_squared_error(y_test, y_predicted))
    r2   = r2_score(y_test, y_predicted)

    print("\n========= MODEL ACCURACY REPORT =========")
    print(f"MAE  (Mean Abs Error)     : ${mae:,.2f}")
    print(f"RMSE (Root Mean Sq Error) : ${rmse:,.2f}")
    print(f"R2   (Accuracy Score)     : {r2:.4f} ({round(r2*100, 2)}%)")
    print("==========================================")

    # Step 5: Cross Validation with all 3 metrics
    # Enterprise standard - tests on 5 different data splits
    print("\n========= CROSS VALIDATION (5 FOLD) =========")
    print("Testing model on 5 different data splits...")
    cv_start = time.time()

    full_data = pd.read_parquet('data/curated/sales_analytics.parquet')
    X_full = full_data[RETAIL_FEATURES]
    y_full = full_data['Weekly_Sales']

    # Run CV for all 3 metrics separately
    cv_r2 = cross_val_score(
        retail_demand_model, X_full, y_full,
        cv=5, scoring='r2', n_jobs=-1
    )
    cv_mae = cross_val_score(
        retail_demand_model, X_full, y_full,
        cv=5, scoring='neg_mean_absolute_error', n_jobs=-1
    )
    cv_rmse = cross_val_score(
        retail_demand_model, X_full, y_full,
        cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1
    )

    cv_end = time.time()

    # Print results in clean table
    print(f"\n{'Fold':<8}{'R2':>10}{'MAE':>15}{'RMSE':>15}")
    print(f"─────────────────────────────────────────────")
    for i in range(5):
        print(f"Fold {i+1:<3} {cv_r2[i]*100:>9.2f}%  ${abs(cv_mae[i]):>12,.0f}  ${abs(cv_rmse[i]):>12,.0f}")
    print(f"─────────────────────────────────────────────")
    print(f"{'Mean':<8}{cv_r2.mean()*100:>9.2f}%  ${abs(cv_mae.mean()):>12,.0f}  ${abs(cv_rmse.mean()):>12,.0f}")
    print(f"{'Std':<8}{cv_r2.std()*100:>9.2f}%  ${abs(cv_mae.std()):>12,.0f}  ${abs(cv_rmse.std()):>12,.0f}")

    if cv_r2.std() < 0.02:
        print("\nVERDICT: Model is ROBUST - consistent across all folds!")
    elif cv_r2.std() < 0.05:
        print("\nVERDICT: Model is ACCEPTABLE - slight variance")
    else:
        print("\nVERDICT: Model is UNSTABLE - needs fixing!")

    print(f"CV completed in {round(cv_end - cv_start, 2)} seconds")
    print("==============================================")

    # Step 6: Final interpretation
    print("\nFINAL INTERPRETATION:")
    if r2 >= 0.90:
        print("EXCELLENT - Model is very accurate!")
    elif r2 >= 0.80:
        print("GOOD - Model is working well")
    elif r2 >= 0.70:
        print("AVERAGE - Model needs improvement")
    else:
        print("POOR - Model needs retraining")

    print(f"On average forecast is off by ${mae:,.2f} per week per store")
    end_time = time.time()
    print(f"\nTotal evaluation time: {round(end_time - start_time, 2)} seconds")
    print("---- EVALUATION COMPLETE ----")

if __name__ == "__main__":
    evaluate_walmart_model()
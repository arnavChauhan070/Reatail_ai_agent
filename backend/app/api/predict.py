from fastapi import APIRouter, HTTPException
from backend.app.schemas import PredictRequest, PredictResponse
from backend.app.config import settings
from backend.app.database import retail_db
import pickle
import pandas as pd
import logging
from datetime import datetime

# Arnav Chauhan - Left Shift 2026 - Smart Retail Platform
# predict.py loads XGBoost model and returns sales forecast

logger = logging.getLogger(__name__)
router = APIRouter()

# Load model once when server starts
# Not inside function = faster response
try:
    with open(settings.DEMAND_MODEL_PATH, 'rb') as f:
        demand_model = pickle.load(f)
    logger.info("Demand model loaded successfully!")
except Exception as e:
    demand_model = None
    logger.error(f"Model load failed: {e}")

# Feature order must match train.py exactly
RETAIL_FEATURES = [
    'Store', 'Holiday_Flag', 'Temperature', 'Fuel_Price',
    'CPI', 'Unemployment', 'Year', 'Month', 'Week_of_Year',
    'Quarter', 'Lag_1', 'Lag_2', 'Lag_4', 'Rolling_Mean_4'
]

# ─────────────────────────────────────────
# Route 2: Predict Sales
# POST /api/predict
# ─────────────────────────────────────────
@router.post("/api/predict", response_model=PredictResponse)
def predict_sales(request: PredictRequest):
    """
    Predict weekly sales for a store
    Input: store features
    Output: predicted weekly sales in dollars
    """
    try:
        # Step 1: Check model loaded
        if demand_model is None:
            raise HTTPException(
                status_code=500,
                detail="Model not loaded!"
            )

        # Step 2: Build feature dataframe
        # Must match exact order from train.py
        input_data = pd.DataFrame([{
            'Store'         : request.store_id,
            'Holiday_Flag'  : request.holiday_flag,
            'Temperature'   : request.temperature,
            'Fuel_Price'    : request.fuel_price,
            'CPI'           : request.cpi,
            'Unemployment'  : request.unemployment,
            'Year'          : request.year,
            'Month'         : request.month,
            'Week_of_Year'  : request.week_of_year,
            'Quarter'       : request.quarter,
            'Lag_1'         : request.lag_1,
            'Lag_2'         : request.lag_2,
            'Lag_4'         : request.lag_4,
            'Rolling_Mean_4': request.rolling_mean_4
        }])

        # Step 3: Make prediction
        predicted_sales = demand_model.predict(input_data)[0]
        predicted_sales = round(float(predicted_sales), 2)
        logger.info(f"Prediction: Store {request.store_id} = ${predicted_sales:,.2f}")

        # Step 4: Log to MongoDB
        retail_db.log_conversation(
            user_query=f"Predict sales for Store {request.store_id}",
            agent_name="Forecasting Model",
            response=f"Predicted Sales: ${predicted_sales:,.2f}"
        )

        # Step 5: Return response
        return PredictResponse(
            store_id        =request.store_id,
            predicted_sales =predicted_sales,
            model_used      ="XGBoost Demand Forecasting Model",
            message         =f"Store {request.store_id} predicted weekly sales: ${predicted_sales:,.2f}"
        )

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────
# Route 2b: Quick predict by store id only
# GET /api/predict/quick?store_id=1
# ─────────────────────────────────────────
@router.get("/api/predict/quick")
def quick_predict(store_id: int):
    """
    Quick prediction using latest data
    from curated parquet for that store
    """
    try:
        # Load latest data for store
        df = pd.read_parquet(settings.CURATED_DATA_PATH)
        store_data = df[df['Store'] == store_id].tail(1)

        if store_data.empty:
            raise HTTPException(
                status_code=404,
                detail=f"Store {store_id} not found"
            )

        # Predict using latest row
        input_data      = store_data[RETAIL_FEATURES]
        predicted_sales = demand_model.predict(input_data)[0]
        predicted_sales = round(float(predicted_sales), 2)

        return {
            "store_id"       : store_id,
            "predicted_sales": predicted_sales,
            "message"        : f"Store {store_id} next week forecast: ${predicted_sales:,.2f}"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
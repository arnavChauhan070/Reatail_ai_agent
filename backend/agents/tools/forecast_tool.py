"""
forecast_tool.py
CrewAI tool for the Forecasting Agent.
Uses @tool decorator (required for CrewAI 0.5.0 — NOT BaseTool).
"""

import pickle
import pandas as pd
from langchain.tools import tool
from backend.app.config import settings


@tool("retail_forecast")
def retail_forecast(question: str) -> str:
    """
    Predicts future weekly sales for Walmart stores using XGBoost model.
    Use for: predict, forecast, estimate, future sales, next week sales.
    Input : natural language question about sales prediction.
    Output: forecast report with top/bottom stores and all predictions.
    """
    try:
        # ── Load trained XGBoost model ─────────────────────────────────────────
        with open(settings.DEMAND_MODEL_PATH, "rb") as f:
            retail_demand_model = pickle.load(f)

        # ── Load Gold parquet ──────────────────────────────────────────────────
        walmart_gold_data = pd.read_parquet(settings.CURATED_DATA_PATH)

        # ── Feature columns ────────────────────────────────────────────────────
        feature_columns = [
            "Store", "Holiday_Flag", "Temperature",
            "Fuel_Price", "CPI", "Unemployment",
            "Year", "Month", "Week_of_Year", "Quarter",
            "Lag_1", "Lag_2", "Lag_4", "Rolling_Mean_4",
        ]

        # ── Predict for all 45 stores ──────────────────────────────────────────
        store_forecasts = []
        for store_id in sorted(walmart_gold_data["Store"].unique()):
            store_data = walmart_gold_data[
                walmart_gold_data["Store"] == store_id
            ].tail(1)

            if not store_data.empty:
                X         = store_data[feature_columns]
                predicted = retail_demand_model.predict(X)[0]
                store_forecasts.append({
                    "store_id"       : store_id,
                    "predicted_sales": round(float(predicted), 2),
                })

        # ── Build summary ──────────────────────────────────────────────────────
        forecasts_df = pd.DataFrame(store_forecasts)
        top5         = forecasts_df.nlargest(5, "predicted_sales")
        low5         = forecasts_df.nsmallest(5, "predicted_sales")

        summary = f"""
WALMART SALES FORECAST REPORT
==============================
Model             : XGBoost Demand Forecasting (R2: 98.64%)
Total Stores      : {len(store_forecasts)}
Avg Predicted     : ${forecasts_df["predicted_sales"].mean():,.2f}
Max Predicted     : ${forecasts_df["predicted_sales"].max():,.2f}
Min Predicted     : ${forecasts_df["predicted_sales"].min():,.2f}

TOP 5 HIGHEST FORECAST STORES:
{top5.to_string(index=False)}

TOP 5 LOWEST FORECAST STORES:
{low5.to_string(index=False)}

ALL STORE FORECASTS:
{forecasts_df.to_string(index=False)}

USER QUERY: {question}
"""
        return summary

    except Exception as forecast_error:
        return f"forecast_tool error: {str(forecast_error)}"
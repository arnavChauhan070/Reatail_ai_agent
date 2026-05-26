"""
db_tool.py
CrewAI tool for the Data Analyst Agent.
"""
import pandas as pd
from langchain.tools import tool
from backend.app.config import settings


@tool("retail_db_query")
def retail_db_query(question: str) -> str:
    """
    Answers questions about Walmart sales data using Gold parquet.
    Use for: sales history, store performance, trends, holiday impact.
    Input : natural language question about sales data.
    Output: detailed sales summary with real figures.
    """
    try:
        df = pd.read_parquet(settings.CURATED_DATA_PATH)

        summary = f"""
WALMART SALES DATA SUMMARY:
Total Records     : {len(df)}
Total Stores      : {df['Store'].nunique()}
Date Range        : {df['Date'].min()} to {df['Date'].max()}
Average Sales     : ${df['Weekly_Sales'].mean():,.2f}
Max Weekly Sales  : ${df['Weekly_Sales'].max():,.2f}
Min Weekly Sales  : ${df['Weekly_Sales'].min():,.2f}

TOP 5 STORES BY AVERAGE SALES:
{df.groupby('Store')['Weekly_Sales'].mean().sort_values(ascending=False).head(5).to_string()}

HOLIDAY VS NON-HOLIDAY SALES:
Holiday Avg    : ${df[df['Holiday_Flag']==1]['Weekly_Sales'].mean():,.2f}
Non-Holiday Avg: ${df[df['Holiday_Flag']==0]['Weekly_Sales'].mean():,.2f}

MONTHLY AVERAGE SALES:
{df.groupby('Month')['Weekly_Sales'].mean().to_string()}

USER QUERY: {question}
"""
        return summary

    except Exception as e:
        return f"db_tool error: {str(e)}"
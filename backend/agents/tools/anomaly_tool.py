"""
anomaly_tool.py
CrewAI tool for answering questions about detected sales anomalies.
Uses the existing Isolation Forest output saved by ml_engine/anomaly_detector.py.
"""

import re
import pandas as pd
from langchain.tools import tool
from backend.app.config import settings


def _extract_store_id(question: str) -> int | None:
    match = re.search(r"\bstore\s+(\d+)\b", question.lower())
    return int(match.group(1)) if match else None


def _format_money(value: float) -> str:
    return f"${value:,.2f}"


@tool("retail_anomaly_analysis")
def retail_anomaly_analysis(question: str) -> str:
    """
    Analyzes detected unusual Walmart sales patterns from anomalies.parquet.
    Use for: anomalies, anomaly summary, outliers, unusual sales patterns,
    biggest sales spike, positive spike, negative drop, contextual anomaly.
    Input : natural language question about detected sales anomalies.
    Output: anomaly summary with stores, dates, counts, and top spikes.
    """
    try:
        df = pd.read_parquet(settings.ANOMALY_DATA_PATH)

        required_columns = {
            "Store", "Date", "Weekly_Sales", "anomaly_score",
            "is_anomaly", "anomaly_type",
        }
        missing_columns = required_columns.difference(df.columns)
        if missing_columns:
            return (
                "anomaly_tool error: anomalies data is missing columns: "
                f"{sorted(missing_columns)}"
            )

        anomalies = df[df["is_anomaly"] == 1].copy()
        if anomalies.empty:
            return "No anomalies were detected in the current anomaly dataset."

        store_id = _extract_store_id(question)
        if store_id is not None:
            anomalies = anomalies[anomalies["Store"] == store_id]
            if anomalies.empty:
                return f"No anomalies were detected for Store {store_id}."

        total_records = len(df)
        anomaly_count = len(anomalies)
        anomaly_rate = anomaly_count / total_records * 100

        type_counts = anomalies["anomaly_type"].value_counts()
        store_counts = (
            anomalies.groupby("Store")
            .size()
            .sort_values(ascending=False)
            .head(10)
        )
        top_spikes = (
            anomalies.sort_values("Weekly_Sales", ascending=False)
            [["Store", "Date", "Weekly_Sales", "anomaly_type"]]
            .head(5)
        )

        biggest_spike = top_spikes.iloc[0]
        top_spikes_display = top_spikes.copy()
        top_spikes_display["Weekly_Sales"] = top_spikes_display["Weekly_Sales"].map(_format_money)

        title_scope = f"STORE {store_id} " if store_id is not None else ""
        return f"""
{title_scope}WALMART ANOMALY DETECTION REPORT
========================================
Model Output Source : {settings.ANOMALY_DATA_PATH}
Detection Method    : Isolation Forest output from ml_engine/anomaly_detector.py
Total Records       : {total_records}
Anomalies Found     : {anomaly_count}
Anomaly Rate        : {anomaly_rate:.2f}%

ANOMALY TYPE BREAKDOWN:
{type_counts.to_string()}

STORES WITH MOST UNUSUAL SALES PATTERNS:
{store_counts.to_string()}

BIGGEST SALES SPIKE:
Store       : {int(biggest_spike["Store"])}
Date        : {biggest_spike["Date"]}
Weekly Sales: {_format_money(float(biggest_spike["Weekly_Sales"]))}
Type        : {biggest_spike["anomaly_type"]}

TOP 5 SALES SPIKE ANOMALIES:
{top_spikes_display.to_string(index=False)}

USER QUERY: {question}
"""

    except FileNotFoundError:
        return (
            "anomaly_tool error: anomaly output file was not found. "
            "Run `python ml_engine/anomaly_detector.py` first."
        )
    except Exception as anomaly_error:
        return f"anomaly_tool error: {str(anomaly_error)}"

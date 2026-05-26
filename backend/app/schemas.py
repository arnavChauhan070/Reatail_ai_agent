from pydantic import BaseModel
from typing import Optional
from datetime import datetime

# Arnav Chauhan - Left Shift 2026 - Smart Retail Platform
# schemas.py defines request and response shapes for all 4 APIs

# ─────────────────────────────────────────
# INGEST SCHEMAS - /api/ingest
# ─────────────────────────────────────────

class SaleRecord(BaseModel):
    """Single sale record shape"""
    store_id      : int
    date          : str
    weekly_sales  : float
    holiday_flag  : int
    temperature   : float
    fuel_price    : float
    cpi           : float
    unemployment  : float

class IngestResponse(BaseModel):
    """Response after ingesting data"""
    success       : bool
    message       : str
    records_saved : int

# ─────────────────────────────────────────
# PREDICT SCHEMAS - /api/predict
# ─────────────────────────────────────────

class PredictRequest(BaseModel):
    """Input needed to predict sales"""
    store_id      : int
    year          : int
    month         : int
    week_of_year  : int
    quarter       : int
    holiday_flag  : int
    temperature   : float
    fuel_price    : float
    cpi           : float
    unemployment  : float
    lag_1         : float
    lag_2         : float
    lag_4         : float
    rolling_mean_4: float

class PredictResponse(BaseModel):
    """Prediction result"""
    store_id          : int
    predicted_sales   : float
    model_used        : str
    message           : str

# ─────────────────────────────────────────
# SEARCH SCHEMAS - /api/search
# ─────────────────────────────────────────

class SearchRequest(BaseModel):
    """RAG search input"""
    query         : str
    top_k         : Optional[int] = 3

class SearchResponse(BaseModel):
    """RAG search result"""
    query         : str
    answer        : str
    sources       : list
    agent_used    : str

# ─────────────────────────────────────────
# AGENT SCHEMAS - /api/agent
# ─────────────────────────────────────────

class AgentRequest(BaseModel):
    """User question to agent"""
    user_query    : str
    store_id      : Optional[int] = None

class AgentResponse(BaseModel):
    """Agent response"""
    user_query    : str
    agent_used    : str
    response      : str
    timestamp     : Optional[str] = None
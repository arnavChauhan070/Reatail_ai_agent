import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    PROJECT_NAME: str = "Retail AI Platform"
    VERSION: str      = "1.0.0"

    MONGODB_URL: str        = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
    MONGODB_DB_NAME: str    = os.getenv("MONGODB_DB_NAME", "retail_ai_db")
    SALES_COLLECTION: str   = "live_sales"
    LOGS_COLLECTION: str    = "conversation_logs"

    AZURE_OPENAI_KEY: str        = os.getenv("AZURE_OPENAI_API_KEY", "")
    AZURE_OPENAI_ENDPOINT: str   = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    AZURE_OPENAI_DEPLOYMENT: str = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
    AZURE_OPENAI_VERSION: str    = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

    AZURE_SEARCH_ENDPOINT: str   = os.getenv("AZURE_SEARCH_ENDPOINT", "")
    AZURE_SEARCH_API_KEY: str    = os.getenv("AZURE_SEARCH_API_KEY", "")
    AZURE_SEARCH_INDEX_NAME: str = os.getenv("AZURE_SEARCH_INDEX_NAME", "retail-knowledge-index")

    DEMAND_MODEL_PATH: str   = "ml_engine/models/demand_model.pkl"
    ANOMALY_MODEL_PATH: str  = "ml_engine/models/anomaly_model.pkl"

    RAW_DATA_PATH: str       = "data/raw/"
    STAGED_DATA_PATH: str    = "data/staged/sales_staged.parquet"
    CURATED_DATA_PATH: str   = "data/curated/sales_analytics.parquet"
    ANOMALY_DATA_PATH: str   = "data/curated/anomalies.parquet"
    KNOWLEDGE_BASE_PATH: str = "docs/knowledge_base/"

settings = Settings()
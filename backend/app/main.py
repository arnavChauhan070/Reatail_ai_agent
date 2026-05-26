from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
from backend.app.config import settings
from backend.app.database import retail_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    retail_db.connect()
    yield
    retail_db.disconnect()


app = FastAPI(
    title="Retail AI Platform",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from backend.app.api.ingest  import router as ingest_router
from backend.app.api.predict import router as predict_router
from backend.app.api.search  import router as search_router
from backend.app.api.agent   import router as agent_router

app.include_router(ingest_router)
app.include_router(predict_router)
app.include_router(search_router, prefix="/api")
app.include_router(agent_router, prefix="/api")


@app.get("/")
def home():
    return {
        "status"  : "running",
        "platform": "Retail AI Platform",
        "author"  : "Arnav Chauhan - Left Shift 2026",
        "routes"  : [
            "POST /api/ingest/csv",
            "POST /api/ingest/sale",
            "GET  /api/ingest/records",
            "POST /api/predict",
            "GET  /api/predict/quick",
            "POST /api/search",
            "POST /api/agent",
            "GET  /api/agent/history"
        ]
    }
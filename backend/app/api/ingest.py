from fastapi import APIRouter, UploadFile, File, HTTPException
from backend.app.database import retail_db
from backend.app.schemas import SaleRecord, IngestResponse
import pandas as pd
import io
import logging
from datetime import datetime

# Arnav Chauhan - Left Shift 2026 - Smart Retail Platform
# ingest.py handles new data coming into the platform

logger = logging.getLogger(__name__)
router = APIRouter()

# ─────────────────────────────────────────
# Route 1a: Upload CSV file
# POST /api/ingest/csv
# ─────────────────────────────────────────
@router.post("/api/ingest/csv", response_model=IngestResponse)
async def ingest_csv(file: UploadFile = File(...)):
    """
    Upload a CSV file with sales data
    Saves all records to MongoDB live_sales collection
    """
    try:
        # Step 1: Read uploaded CSV
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
        logger.info(f"CSV uploaded: {df.shape}")

        # Step 2: Clean column names
        df.columns = [col.lower().replace(" ", "_") for col in df.columns]

        # Step 3: Add timestamp
        df["ingested_at"] = datetime.utcnow().isoformat()

        # Step 4: Convert to list of dicts
        records = df.to_dict(orient="records")

        # Step 5: Save to MongoDB
        saved = retail_db.insert_many_sales(records)

        return IngestResponse(
            success=True,
            message=f"CSV ingested successfully",
            records_saved=saved
        )

    except Exception as e:
        logger.error(f"CSV ingest failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────
# Route 1b: Single sale record
# POST /api/ingest/sale
# ─────────────────────────────────────────
@router.post("/api/ingest/sale", response_model=IngestResponse)
def ingest_sale(sale: SaleRecord):
    """
    Insert a single sale record
    Saves to MongoDB live_sales collection
    """
    try:
        # Step 1: Convert to dict
        sale_data = sale.dict()

        # Step 2: Add timestamp
        sale_data["ingested_at"] = datetime.utcnow().isoformat()

        # Step 3: Save to MongoDB
        saved_id = retail_db.insert_sale(sale_data)

        if saved_id:
            return IngestResponse(
                success=True,
                message=f"Sale record saved successfully",
                records_saved=1
            )
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to save sale record"
            )

    except Exception as e:
        logger.error(f"Sale ingest failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────
# Route 1c: Get all ingested records
# GET /api/ingest/records
# ─────────────────────────────────────────
@router.get("/api/ingest/records")
def get_records(store_id: int = None):
    """
    Get ingested records from MongoDB
    Optional: filter by store_id
    """
    try:
        if store_id:
            records = retail_db.get_sales_by_store(store_id)
            return {
                "store_id": store_id,
                "records" : records,
                "count"   : len(records)
            }
        else:
            return {
                "message": "Please provide store_id parameter",
                "example": "/api/ingest/records?store_id=1"
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
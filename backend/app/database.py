from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import logging
from backend.app.config import settings


# database.py manages MongoDB Atlas connection
# Two collections: live_sales + conversation_logs

logger = logging.getLogger(__name__)

class RetailDatabase:
    """
    Manages MongoDB connection for Retail AI Platform
    Collections:
    1. live_sales          - stores incoming sales records
    2. conversation_logs   - stores agent Q&A history
    """

    def __init__(self):
        self.client     = None
        self.db         = None
        self.sales      = None
        self.logs       = None

    def connect(self):
        """Connect to MongoDB Atlas"""
        try:
            self.client = MongoClient(settings.MONGODB_URL)

            # Test connection
            self.client.admin.command('ping')

            # Get database
            self.db = self.client[settings.MONGODB_DB_NAME]

            # Get collections
            self.sales = self.db[settings.SALES_COLLECTION]
            self.logs  = self.db[settings.LOGS_COLLECTION]

            logger.info("MongoDB connected successfully!")
            print("MongoDB connected successfully!")
            return True

        except ConnectionFailure as e:
            logger.error(f"MongoDB connection failed: {e}")
            print(f"MongoDB connection failed: {e}")
            return False

    def disconnect(self):
        """Disconnect from MongoDB"""
        if self.client:
            self.client.close()
            print("MongoDB disconnected!")

    def insert_sale(self, sale_data: dict):
        """
        Insert new sale record into live_sales collection
        Called by: /api/ingest route
        """
        try:
            result = self.sales.insert_one(sale_data)
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Insert sale failed: {e}")
            return None

    def insert_many_sales(self, sales_list: list):
        """
        Insert multiple sale records at once
        Called by: /api/ingest route (CSV upload)
        """
        try:
            result = self.sales.insert_many(sales_list)
            return len(result.inserted_ids)
        except Exception as e:
            logger.error(f"Insert many sales failed: {e}")
            return 0

    def log_conversation(self, user_query: str, agent_name: str, response: str):
        """
        Save agent conversation to conversation_logs
        Called by: /api/agent and /api/search routes
        """
        from datetime import datetime
        try:
            log_entry = {
                "user_query"  : user_query,
                "agent_name"  : agent_name,
                "response"    : response,
                "timestamp"   : datetime.utcnow()
            }
            result = self.logs.insert_one(log_entry)
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Log conversation failed: {e}")
            return None

    def get_recent_logs(self, limit: int = 10):
        """
        Get recent conversations
        Called by: /api/agent route for chat history
        """
        try:
            logs = list(
                self.logs
                .find({}, {"_id": 0})
                .sort("timestamp", -1)
                .limit(limit)
            )
            return logs
        except Exception as e:
            logger.error(f"Get logs failed: {e}")
            return []

    def get_sales_by_store(self, store_id: int):
        """
        Get all sales for a specific store
        Called by: Data Analyst Agent
        """
        try:
            sales = list(
                self.sales
                .find({"store_id": store_id}, {"_id": 0})
            )
            return sales
        except Exception as e:
            logger.error(f"Get sales failed: {e}")
            return []

# Single instance used across all files
retail_db = RetailDatabase()
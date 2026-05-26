"""
agent.py
POST /api/agent — runs CrewAI multi-agent system.
GET  /api/agent/history — returns conversation logs.
"""
import logging
from datetime import datetime
from fastapi import APIRouter, HTTPException
from backend.agents.crew import run_retail_crew
from backend.app.schemas import AgentRequest, AgentResponse
from backend.app.database import retail_db

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/agent", response_model=AgentResponse)
def run_agent(request: AgentRequest):
    """Routes question to the correct CrewAI agent and returns response."""
    try:
        logger.info(f"Agent query: {request.user_query}")
        result = run_retail_crew(request.user_query)
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["response"])
        return AgentResponse(
            user_query=request.user_query,
            agent_used=result["agent_used"],
            response=result["response"],
            timestamp=datetime.utcnow().isoformat()
        )
    except Exception as e:
        logger.error(f"Agent error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agent/history")
def get_agent_history(limit: int = 10):
    """Returns recent agent conversation logs from MongoDB."""
    try:
        logs = retail_db.get_recent_logs(limit=limit)
        return {"total": len(logs), "history": logs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
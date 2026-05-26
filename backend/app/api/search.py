"""
search.py
POST /api/search — direct RAG knowledge base search.
"""
import logging
from fastapi import APIRouter, HTTPException
from backend.agents.tools.rag_tool import retail_policy_search
from backend.app.schemas import SearchRequest, SearchResponse
from backend.app.database import retail_db

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/search", response_model=SearchResponse)
def search_policies(request: SearchRequest):
    """Searches the knowledge base and returns a grounded answer."""
    try:
        logger.info(f"Search query: {request.query}")
        result = retail_policy_search(request.query)
        retail_db.log_conversation(
            user_query=request.query,
            agent_name="RAG Policy Agent",
            response=result
        )
        return SearchResponse(
            query=request.query,
            answer=result,
            sources=["docs/knowledge_base/"],
            agent_used="RAG Policy Agent"
        )
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
"""
search.py
POST /api/search — direct RAG knowledge base search.
"""
import logging
import os
from pathlib import Path
from fastapi import APIRouter, HTTPException
from backend.agents.tools.rag_tool import retail_policy_search
from backend.app.schemas import SearchRequest, SearchResponse
from backend.app.database import retail_db

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/search/health")
def search_health():
    """Checks RAG configuration without exposing secret values."""
    required_env = [
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_DEPLOYMENT",
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
        "AZURE_OPENAI_API_VERSION",
        "AZURE_SEARCH_ENDPOINT",
        "AZURE_SEARCH_API_KEY",
        "AZURE_SEARCH_INDEX_NAME",
    ]
    kb_path = Path(os.getenv("KNOWLEDGE_BASE_PATH", "docs/knowledge_base/"))
    kb_files = sorted(path.name for path in kb_path.glob("*.txt")) if kb_path.exists() else []

    return {
        "azure_settings": {
            name: "set" if os.getenv(name) else "missing"
            for name in required_env
        },
        "local_knowledge_base": {
            "path": str(kb_path),
            "exists": kb_path.exists(),
            "files": kb_files,
        },
    }


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

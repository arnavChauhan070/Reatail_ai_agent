"""
rag_tool.py
CrewAI tool for the RAG Policy Agent.
"""
import os
import logging
import re
from pathlib import Path
from langchain.tools import tool
from openai import AzureOpenAI
from dotenv import load_dotenv
from backend.agents.vector_store import search_azure

load_dotenv()

logger = logging.getLogger(__name__)
DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
KNOWLEDGE_BASE_PATH = os.getenv("KNOWLEDGE_BASE_PATH", "docs/knowledge_base/")


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def get_azure_openai_client() -> AzureOpenAI:
    return AzureOpenAI(
        api_key=require_env("AZURE_OPENAI_API_KEY"),
        azure_endpoint=require_env("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        timeout=20.0,
    )


def _tokens(text: str) -> set[str]:
    return {word for word in re.findall(r"[a-z0-9]+", text.lower()) if len(word) > 2}


def _chunk_local_docs() -> list[dict]:
    """Loads local knowledge-base text files as a backup retrieval source."""
    kb_dir = Path(KNOWLEDGE_BASE_PATH)
    if not kb_dir.exists():
        return []

    chunks = []
    for file_path in sorted(kb_dir.glob("*.txt")):
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        for index, paragraph in enumerate(paragraphs):
            chunks.append({
                "content": paragraph,
                "source": file_path.name,
                "chunk_index": index,
                "score": 0.0,
            })
    return chunks


def _search_local_docs(question: str, top_k: int = 3) -> list[dict]:
    question_tokens = _tokens(question)
    ranked = []

    for chunk in _chunk_local_docs():
        content_tokens = _tokens(chunk["content"])
        overlap = question_tokens.intersection(content_tokens)
        if overlap:
            score = len(overlap) / max(len(question_tokens), 1)
            ranked.append({**chunk, "score": round(score, 3)})

    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked[:top_k]


def _extractive_answer(question: str, search_results: list[dict]) -> str:
    """Returns a grounded answer even when Azure OpenAI is unavailable."""
    if not search_results:
        return (
            "Answer: I could not find matching policy information in the local knowledge base.\n\n"
            "Sources: docs/knowledge_base/"
        )

    best = search_results[0]
    answer = best["content"].replace("\n", " ").strip()
    sources = ", ".join(sorted({result["source"] for result in search_results}))
    return (
        f"Answer: {answer}\n\n"
        f"Sources: {sources}\n"
        "Mode: Local knowledge-base fallback"
    )


def _answer_with_azure_openai(question: str, search_results: list[dict]) -> str:
    context_parts = []
    for idx, result in enumerate(search_results):
        context_parts.append(
            f"[Source {idx+1}: {result['source']} | score: {result['score']:.3f}]\n{result['content']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    system_prompt = (
        "You are a professional Walmart store assistant. "
        "Answer using ONLY the context provided. Be concise and professional. "
        "If answer not in context say: 'Please contact 1-800-WALMART.'"
    )
    user_prompt = f"Context:\n\n{context}\n\nQuestion: {question}\n\nAnswer:"

    response = get_azure_openai_client().chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=500,
        temperature=0.2,
    )

    answer = response.choices[0].message.content.strip()
    sources = ", ".join(sorted({result["source"] for result in search_results}))
    return f"Answer: {answer}\n\nSources: {sources}"


@tool("retail_policy_search")
def retail_policy_search(question: str) -> str:
    """
    Searches store knowledge base via Azure AI Search, answers via GPT-4o.
    Use for: store policies, return rules, store hours, FAQ, loyalty program.
    Input : natural language question about store policy or FAQ.
    Output: grounded answer from knowledge base.
    """
    try:
        search_results = search_azure(question, top_k=3)

        if not search_results:
            return "No relevant policy information found in the knowledge base."

        try:
            return _answer_with_azure_openai(question, search_results)
        except Exception as azure_openai_error:
            logger.warning("Azure OpenAI answer generation failed: %s", azure_openai_error)
            return _extractive_answer(question, search_results)

    except Exception as azure_search_error:
        logger.warning("Azure RAG search failed, using local fallback: %s", azure_search_error)
        local_results = _search_local_docs(question, top_k=3)
        if local_results:
            try:
                return _answer_with_azure_openai(question, local_results)
            except Exception as azure_openai_error:
                logger.warning("Azure OpenAI fallback answer failed: %s", azure_openai_error)
                return _extractive_answer(question, local_results)

        return (
            "RAG is not available right now. "
            f"Azure error: {azure_search_error}. "
            "Also no matching local knowledge-base text was found."
        )

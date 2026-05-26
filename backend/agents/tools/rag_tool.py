"""
rag_tool.py
CrewAI tool for the RAG Policy Agent.
"""
import os
from langchain.tools import tool
from openai import AzureOpenAI
from dotenv import load_dotenv
from backend.agents.vector_store import search_azure

load_dotenv()

azure_openai = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
)
DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")


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

        response = azure_openai.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=500,
            temperature=0.2,
        )

        answer = response.choices[0].message.content.strip()
        sources = ", ".join(set(r["source"] for r in search_results))
        return f"Answer: {answer}\n\nSources: {sources}"

    except Exception as e:
        return f"rag_tool error: {str(e)}"
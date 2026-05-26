"""
vector_store.py
Builds and populates an Azure AI Search index with vector embeddings.
Reads .txt and .pdf files from docs/knowledge_base/.
Uses Azure OpenAI text-embedding-ada-002 for embeddings.
Run once: python backend/agents/vector_store.py
"""
import os
import uuid
import requests
from openai import AzureOpenAI
from dotenv import load_dotenv

try:
    from pypdf import PdfReader
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

load_dotenv()

SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
SEARCH_API_KEY  = os.getenv("AZURE_SEARCH_API_KEY")
INDEX_NAME      = os.getenv("AZURE_SEARCH_INDEX_NAME", "retail-knowledge-index")
KNOWLEDGE_BASE  = os.getenv("KNOWLEDGE_BASE_PATH", "docs/knowledge_base/")

azure_openai = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
)

SEARCH_HEADERS = {
    "Content-Type": "application/json",
    "api-key"     : SEARCH_API_KEY,
}


def create_search_index() -> None:
    """
    Creates Azure AI Search index with vector field (1536-dim).
    Fields: id, content, source, chunk_index, embedding.
    """
    index_schema = {
        "name": INDEX_NAME,
        "fields": [
            {"name": "id",          "type": "Edm.String", "key": True, "filterable": True},
            {"name": "content",     "type": "Edm.String", "searchable": True, "retrievable": True},
            {"name": "source",      "type": "Edm.String", "filterable": True, "retrievable": True},
            {"name": "chunk_index", "type": "Edm.Int32",  "retrievable": True},
            {
                "name"               : "embedding",
                "type"               : "Collection(Edm.Single)",
                "searchable"         : True,
                "retrievable"        : True,
                "dimensions"         : 1536,
                "vectorSearchProfile": "retail-vector-profile",
            },
        ],
        "vectorSearch": {
            "algorithms": [
                {
                    "name": "retail-hnsw",
                    "kind": "hnsw",
                    "hnswParameters": {
                        "m": 4, "efConstruction": 400,
                        "efSearch": 500, "metric": "cosine",
                    },
                }
            ],
            "profiles": [
                {"name": "retail-vector-profile", "algorithm": "retail-hnsw"}
            ],
        },
    }
    url = f"{SEARCH_ENDPOINT}/indexes/{INDEX_NAME}?api-version=2023-11-01"
    response = requests.put(url, headers=SEARCH_HEADERS, json=index_schema)
    if response.status_code in (200, 201,204):
        print(f"  Index '{INDEX_NAME}' created successfully.")
    else:
        raise Exception(f"Failed to create index: {response.status_code} — {response.text}")


def load_all_documents(kb_path: str) -> list:
    """Loads all .txt and .pdf files from knowledge base folder."""
    documents = []
    for filename in os.listdir(kb_path):
        filepath = os.path.join(kb_path, filename)
        if filename.endswith(".txt"):
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            documents.append({"filename": filename, "content": content})
            print(f"  Loaded TXT: {filename} ({len(content)} chars)")
        elif filename.endswith(".pdf") and PDF_SUPPORT:
            reader  = PdfReader(filepath)
            content = ""
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    content += f"\n\n[Page {i+1}]\n{text}"
            if content:
                documents.append({"filename": filename, "content": content})
                print(f"  Loaded PDF: {filename} ({len(content)} chars)")
    return documents


def chunk_text(text: str, chunk_size: int = 100) -> list:
    """Splits text into ~100 word chunks with 20-word overlap."""
    words, overlap, chunks, i = text.split(), 20, [], 0
    while i < len(words):
        chunks.append(" ".join(words[i : i + chunk_size]))
        i += chunk_size - overlap
    return chunks


def get_embedding(text: str) -> list:
    """Gets 1536-dim embedding from Azure OpenAI text-embedding-ada-002."""
    response = azure_openai.embeddings.create(
        input=text,
        model="text-embedding-ada-002",
    )
    return response.data[0].embedding


def upload_documents_to_search(documents: list) -> None:
    """Chunks, embeds, and uploads all documents to Azure AI Search."""
    all_docs = []
    total    = 0
    for doc in documents:
        chunks = chunk_text(doc["content"])
        print(f"\n  Embedding '{doc['filename']}' — {len(chunks)} chunks...")
        for idx, chunk in enumerate(chunks):
            all_docs.append({
                "id"         : str(uuid.uuid4()),
                "content"    : chunk,
                "source"     : doc["filename"],
                "chunk_index": idx,
                "embedding"  : get_embedding(chunk),
            })
            total += 1
            if total % 10 == 0:
                print(f"    {total} chunks embedded...")

    upload_url = f"{SEARCH_ENDPOINT}/indexes/{INDEX_NAME}/docs/index?api-version=2023-11-01"
    for i in range(0, len(all_docs), 100):
        batch    = all_docs[i : i + 100]
        response = requests.post(upload_url, headers=SEARCH_HEADERS, json={"value": batch})
        if response.status_code in (200, 201):
            print(f"  Uploaded batch {i//100 + 1} ({len(batch)} chunks)")
        else:
            raise Exception(f"Upload failed: {response.status_code} — {response.text}")
    print(f"\n  Total chunks indexed: {total}")


def search_azure(query: str, top_k: int = 3) -> list:
    """
    Embeds query and runs vector similarity search in Azure AI Search.
    Returns top_k results as list of dicts: { content, source, chunk_index, score }
    Called by rag_tool.py at query time.
    """
    search_payload = {
        "vectorQueries": [
            {"kind": "vector", "vector": get_embedding(query), "fields": "embedding", "k": top_k}
        ],
        "select": "content,source,chunk_index",
        "top"   : top_k,
    }
    url = f"{SEARCH_ENDPOINT}/indexes/{INDEX_NAME}/docs/search?api-version=2023-11-01"
    response = requests.post(url, headers=SEARCH_HEADERS, json=search_payload)
    if response.status_code != 200:
        raise Exception(f"Search failed: {response.status_code} — {response.text}")
    return [
        {
            "content"    : r.get("content", ""),
            "source"     : r.get("source", ""),
            "chunk_index": r.get("chunk_index", 0),
            "score"      : r.get("@search.score", 0.0),
        }
        for r in response.json().get("value", [])
    ]


if __name__ == "__main__":
    print("\n=== Azure AI Search — Knowledge Base Indexer ===\n")
    print("Step 1: Creating index...")
    create_search_index()
    print("\nStep 2: Loading documents...")
    docs = load_all_documents(KNOWLEDGE_BASE)
    if not docs:
        print("No documents found in docs/knowledge_base/")
        exit(1)
    print("\nStep 3: Embedding + uploading...")
    upload_documents_to_search(docs)
    print(f"\n Done! Index '{INDEX_NAME}' is live at {SEARCH_ENDPOINT}\n")
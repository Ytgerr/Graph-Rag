import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import os
import threading
import concurrent.futures
from dotenv import load_dotenv
import logging
from openai import OpenAI
import sys
from pathlib import Path

from app.graph_rag import GraphRAGRetriever
from app.vector_store import VectorRAGRetriever
from app.document_processor import load_raw_documents, chunk_documents

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from experiments.collect_wiki import collect, CollectionProgress

logging.basicConfig(level=logging.INFO)
logging.getLogger("neo4j.notifications").setLevel(logging.WARNING)
logging.getLogger("neo4j").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

load_dotenv()

DATASET_PATH = Path(__file__).parent / "dataset" / "data_from_wiki.txt"


# ── Pydantic models ──

class DualQueryRequest(BaseModel):
    query: str
    top_k: int = 5
    temperature: float = 0.2
    model: str = "openai/gpt-4o-mini"


class RAGResponse(BaseModel):
    answer: str
    sources: List[str]
    similarity_scores: List[float]
    metadata: Optional[Dict] = None


class DualRAGResponse(BaseModel):
    graph: RAGResponse
    vector: RAGResponse


class SystemStats(BaseModel):
    status: str
    graph_stats: Optional[Dict] = None
    vector_stats: Optional[Dict] = None
    num_documents: int


class WikiCollectRequest(BaseModel):
    topic: str
    lang: str = "en"
    limit: int = 500
    expand: bool = True


# ── LLM ──

class LLMComponent:

    def __init__(self, api_key: Optional[str] = None):
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        self.client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

    def generate(
        self, query: str, context: str,
        model: str = "openai/gpt-4o-mini", temperature: float = 0.2
    ) -> str:
        system_prompt = (
            "You are a precise research assistant that answers questions strictly based on provided context.\n"
            "Rules:\n"
            "1. Use ONLY information from the context below. Do NOT add external knowledge.\n"
            "2. If the context lacks sufficient information, explicitly state what is missing.\n"
            "3. Answer in the same language as the user's question.\n"
            "4. Be concise but thorough. Structure long answers with bullet points or numbered lists.\n"
            "5. When possible, reference which part of the context supports your answer."
        )

        user_prompt = (
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION: {query}\n\n"
            f"ANSWER:"
        )

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=1500,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error("LLM generation failed: %s", e)
            raise


# ── RAG System ──

class RAGSystem:
    """Manages both Graph RAG and Vector RAG pipelines.

    - Graph RAG receives raw documents (paragraph-level) for entity extraction.
    - Vector RAG receives chunked documents (512-char overlapping) for precise retrieval.
    """

    def __init__(self, raw_documents: List[str], chunks: List[str]):
        self.raw_documents = raw_documents

        logger.info("Initializing Graph RAG (Neo4j) with %d documents...", len(raw_documents))
        self.graph_retriever = GraphRAGRetriever(
            neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
            neo4j_password=os.getenv("NEO4J_PASSWORD", "password"),
        )
        self.graph_retriever.build_graph(raw_documents)

        embedding_model = os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-small")
        logger.info("Initializing Vector RAG (ChromaDB) with %d chunks (embedding=%s)...", len(chunks), embedding_model)
        self.vector_retriever = VectorRAGRetriever(
            embedding_model=embedding_model,
            collection_name="rag_chunks",
            persist_directory=os.getenv("CHROMA_PERSIST_DIR", None),
        )
        self.vector_retriever.build_index(chunks)

        self.llm = LLMComponent()
        logger.info("RAG system ready: %d raw docs, %d chunks", len(raw_documents), len(chunks))

    def reload(self, raw_documents: List[str], chunks: List[str], force_rebuild: bool = False):
        self.raw_documents = raw_documents
        logger.info("Rebuilding Graph RAG with %d documents...", len(raw_documents))
        self.graph_retriever.build_graph(raw_documents, force_rebuild=force_rebuild)
        logger.info("Rebuilding Vector RAG with %d chunks...", len(chunks))
        self.vector_retriever.build_index(chunks)
        logger.info("RAG system reloaded")

    def query(
        self, query: str, top_k: int = 5, temperature: float = 0.2,
        model: str = "openai/gpt-4o-mini", retrieval_mode: str = "graph",
    ) -> RAGResponse:
        if retrieval_mode == "graph":
            sources, scores, metadata = self.graph_retriever.retrieve(query, top_k=top_k)
        elif retrieval_mode == "vector":
            sources, scores, metadata = self.vector_retriever.retrieve(query, top_k=top_k)
        else:
            raise ValueError(f"Invalid retrieval_mode: {retrieval_mode}")

        if not sources:
            return RAGResponse(
                answer="I couldn't find relevant information to answer your question.",
                sources=[], similarity_scores=[], metadata=metadata,
            )

        context = "\n\n".join(sources)
        answer = self.llm.generate(
            query=query, context=context,
            model=model, temperature=temperature,
        )

        return RAGResponse(
            answer=answer, sources=sources,
            similarity_scores=scores, metadata=metadata,
        )


# ── FastAPI app ──

app = FastAPI(
    title="RAG System API", version="4.0.0",
    description="Graph (Neo4j) + Vector (ChromaDB) RAG with proper chunking",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

rag_system: Optional[RAGSystem] = None
wiki_progress = CollectionProgress()
_wiki_lock = threading.Lock()
_init_lock = threading.Lock()


def _init_rag_background():
    """Load dataset and init RAG in background thread."""
    global rag_system
    try:
        raw_docs = load_raw_documents()
        if not raw_docs:
            logger.info("No dataset found, RAG will init after wiki collection")
            return
        chunks = chunk_documents(raw_docs)
        logger.info("Loaded %d raw docs, %d chunks — initializing RAG...", len(raw_docs), len(chunks))
        with _init_lock:
            rag_system = RAGSystem(raw_docs, chunks)
    except Exception as e:
        logger.error("Failed to initialize RAG system: %s", e)


if DATASET_PATH.exists() and DATASET_PATH.stat().st_size > 100:
    threading.Thread(target=_init_rag_background, daemon=True).start()
else:
    logger.info("No dataset file found. Use /collect-wiki to collect data first.")


# ── Endpoints ──

@app.post("/query/dual", response_model=DualRAGResponse)
async def query_dual(request: DualQueryRequest) -> DualRAGResponse:
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not ready. Collect a dataset first via ⚙️ → Wiki Dataset.")
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            graph_future = executor.submit(
                rag_system.query,
                query=request.query, top_k=request.top_k,
                temperature=request.temperature, model=request.model,
                retrieval_mode="graph",
            )
            vector_future = executor.submit(
                rag_system.query,
                query=request.query, top_k=request.top_k,
                temperature=request.temperature, model=request.model,
                retrieval_mode="vector",
            )

            graph_result = graph_future.result(timeout=120)
            vector_result = vector_future.result(timeout=120)

        return DualRAGResponse(graph=graph_result, vector=vector_result)
    except Exception as e:
        logger.error("Dual query failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "healthy", "rag_ready": rag_system is not None}


@app.get("/stats", response_model=SystemStats)
async def get_stats():
    if not rag_system:
        return SystemStats(status="not_ready", num_documents=0)
    return SystemStats(
        status="ready",
        graph_stats=rag_system.graph_retriever.get_graph_stats(),
        vector_stats=rag_system.vector_retriever.get_stats(),
        num_documents=len(rag_system.raw_documents),
    )




def _run_wiki_collection(topic: str, lang: str, limit: int, expand: bool = True):
    global rag_system, wiki_progress
    output_path = str(DATASET_PATH)
    api_key = os.getenv("OPENAI_API_KEY") if expand else None

    try:
        result = collect(
            topic=topic, lang=lang, limit=limit,
            output=output_path, progress=wiki_progress,
            expand=expand, api_key=api_key,
        )
        logger.info("Wiki collection done: %s", result)

        wiki_progress.status = "reloading"
        raw_docs = load_raw_documents()
        chunks = chunk_documents(raw_docs)
        logger.info("Reloading RAG with %d docs, %d chunks...", len(raw_docs), len(chunks))

        with _init_lock:
            if rag_system:
                rag_system.reload(raw_docs, chunks, force_rebuild=True)
            else:
                rag_system = RAGSystem(raw_docs, chunks)

        wiki_progress.status = "done"
        logger.info("Dataset reloaded, RAG ready")

    except Exception as e:
        logger.error("Wiki collection failed: %s", e)
        wiki_progress.status = "error"
        wiki_progress.error = str(e)


@app.post("/collect-wiki")
async def collect_wiki(request: WikiCollectRequest):
    global wiki_progress

    with _wiki_lock:
        if wiki_progress.status in ("discovering", "collecting", "reloading"):
            raise HTTPException(status_code=409, detail="Collection already in progress")

        wiki_progress = CollectionProgress()
        wiki_progress.topic = request.topic
        wiki_progress.status = "starting"

    thread = threading.Thread(
        target=_run_wiki_collection,
        args=(request.topic, request.lang, request.limit, request.expand),
        daemon=True,
    )
    thread.start()

    expand_label = " + LLM expansion" if request.expand else ""
    return {"message": f"Started collecting '{request.topic}' ({request.lang}, limit={request.limit}{expand_label})"}


@app.get("/collect-wiki/status")
async def collect_wiki_status():
    return wiki_progress.to_dict()


def main():
    uvicorn.run("app.backend:app", host="0.0.0.0", port=8002, log_level="info")


if __name__ == "__main__":
    main()

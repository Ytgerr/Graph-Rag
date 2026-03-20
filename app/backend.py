import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import os
from dotenv import load_dotenv
import logging
from openai import OpenAI
import sys
import subprocess

from app.graph_rag import GraphRAGRetriever
from app.vector_store import VectorRAGRetriever
from app.document_processor import load_default_knowledge_base

import nltk
import spacy

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=False)
    nlp = spacy.load("en_core_web_sm")


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    temperature: float = 0.2
    model: str = "openai/gpt-4o-mini"
    retrieval_mode: str = "graph"  # "graph" or "vector"
    vector_method: str = "tfidf"  # "tfidf" or "bm25" (only used when retrieval_mode="vector")
    use_entity_context: bool = True  # Only used for graph mode


class RAGResponse(BaseModel):
    answer: str
    sources: List[str]
    similarity_scores: List[float]
    metadata: Optional[Dict] = None


class SystemStats(BaseModel):
    status: str
    graph_stats: Optional[Dict] = None
    vector_stats: Optional[Dict] = None
    num_documents: int


class LLMComponent:
    
    def __init__(self, api_key: Optional[str] = None):
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )
        logger.info("LLM component initialized with OpenRouter")
    
    def generate(
        self,
        query: str,
        context: str,
        entity_context: str = "",
        model: str = "openai/gpt-4o-mini",
        temperature: float = 0.2
    ) -> str:
        """Generate answer based on context"""
        system_prompt = (
            "You are an expert AI assistant specializing in Retrieval Augmented Generation (RAG) "
            "and Natural Language Processing. Provide detailed, accurate answers based on the given context. "
            "If the context doesn't contain relevant information, say so clearly. "
            "Use the entity relationships to provide more comprehensive answers."
        )
        
        user_prompt = f"""Based on the following context, answer the question:

DOCUMENT CONTEXT:
{context}
"""
        
        if entity_context:
            user_prompt += f"""
ENTITY RELATIONSHIPS:
{entity_context}
"""
        
        user_prompt += f"""
QUESTION:
{query}

ANSWER:"""
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=1500
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise


class RAGSystem:
    """Unified RAG system supporting Graph and Vector retrieval modes"""
    
    def __init__(self, documents: List[str], vector_method: str = "tfidf"):
        """
        Initialize RAG system with both retrievers
        
        Args:
            documents: List of documents to index
            vector_method: "tfidf" or "bm25" for vector retrieval
        """
        self.documents = documents
        
        # Initialize Graph RAG retriever
        logger.info("Initializing Graph RAG retriever...")
        self.graph_retriever = GraphRAGRetriever()
        self.graph_retriever.build_graph(documents)
        
        # Initialize Vector RAG retriever
        logger.info(f"Initializing Vector RAG retriever with {vector_method}...")
        self.vector_retriever = VectorRAGRetriever(method=vector_method)
        self.vector_retriever.build_index(documents)
        
        # Initialize LLM
        self.llm = LLMComponent()
        
        logger.info("RAG system initialized successfully")
    
    def query(
        self,
        query: str,
        top_k: int = 5,
        temperature: float = 0.2,
        model: str = "openai/gpt-4o-mini",
        retrieval_mode: str = "graph",
        vector_method: str = "tfidf",
        use_entity_context: bool = True
    ) -> RAGResponse:
        """
        Process a query through the RAG pipeline
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            temperature: LLM temperature
            model: LLM model to use
            retrieval_mode: "graph" or "vector"
            vector_method: "tfidf" or "bm25" (only used if retrieval_mode="vector")
            use_entity_context: Whether to add entity context (only for graph mode)
        """
        try:
            # Retrieve relevant documents based on mode
            if retrieval_mode == "graph":
                sources, scores, metadata = self.graph_retriever.retrieve(query, top_k=top_k)
                
                # Add entity context for graph mode
                entity_context = ""
                if use_entity_context and sources:
                    entity_context = self.graph_retriever.get_entity_context(query, max_entities=10)
            
            elif retrieval_mode == "vector":
                # Check if we need to rebuild index with different method
                if self.vector_retriever.method != vector_method:
                    logger.info(f"Switching vector method from {self.vector_retriever.method} to {vector_method}")
                    self.vector_retriever = VectorRAGRetriever(method=vector_method)
                    self.vector_retriever.build_index(self.documents)
                
                sources, scores, metadata = self.vector_retriever.retrieve(query, top_k=top_k)
                entity_context = ""  # No entity context for vector mode
            
            else:
                raise ValueError(f"Invalid retrieval_mode: {retrieval_mode}. Must be 'graph' or 'vector'")
            
            # Handle empty results
            if not sources:
                logger.warning(f"No sources retrieved for query: {query}")
                return RAGResponse(
                    answer="I couldn't find relevant information to answer your question.",
                    sources=[],
                    similarity_scores=[],
                    metadata=metadata
                )
            
            # Prepare context
            context = "\n\n".join(sources)
            
            # Generate answer using LLM
            answer = self.llm.generate(
                query=query,
                context=context,
                entity_context=entity_context,
                model=model,
                temperature=temperature
            )
            
            return RAGResponse(
                answer=answer,
                sources=sources,
                similarity_scores=scores,
                metadata=metadata
            )
        
        except Exception as e:
            logger.error(f"RAG query failed: {e}")
            raise


app = FastAPI(
    title="RAG System API",
    version="2.0.0",
    description="RAG system with Graph and Vector retrieval modes"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_system = None

try:
    logger.info("Loading knowledge base...")
    documents = load_default_knowledge_base()
    
    logger.info(f"Initializing RAG system with {len(documents)} documents...")
    
    # Get vector method from environment (default: tfidf)
    vector_method = os.getenv("VECTOR_METHOD", "tfidf").lower()
    if vector_method not in ["tfidf", "bm25"]:
        logger.warning(f"Invalid VECTOR_METHOD '{vector_method}', using 'tfidf'")
        vector_method = "tfidf"
    
    rag_system = RAGSystem(documents, vector_method=vector_method)
    
    logger.info("RAG system ready!")
    logger.info(f"  - Graph retrieval: enabled")
    logger.info(f"  - Vector retrieval: enabled ({vector_method.upper()})")
    
except Exception as e:
    logger.error(f"Failed to initialize RAG system: {e}")
    rag_system = None


@app.post("/query", response_model=RAGResponse)
async def query_rag(request: QueryRequest) -> RAGResponse:
    """
    Query the RAG system
    
    Supports two retrieval modes:
    - "graph": Knowledge graph-based retrieval with entity extraction
    - "vector": Vector-based retrieval (TF-IDF or BM25)
    """
    if not rag_system:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    # Validate retrieval mode
    if request.retrieval_mode not in ["graph", "vector"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid retrieval_mode '{request.retrieval_mode}'. Must be 'graph' or 'vector'"
        )
    
    # Validate vector method
    if request.retrieval_mode == "vector" and request.vector_method not in ["tfidf", "bm25"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid vector_method '{request.vector_method}'. Must be 'tfidf' or 'bm25'"
        )
    
    try:
        response = rag_system.query(
            query=request.query,
            top_k=request.top_k,
            temperature=request.temperature,
            model=request.model,
            retrieval_mode=request.retrieval_mode,
            vector_method=request.vector_method,
            use_entity_context=request.use_entity_context
        )
        return response
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "rag_ready": rag_system is not None
    }


@app.get("/stats", response_model=SystemStats)
async def get_stats():
    """Get system statistics"""
    if not rag_system:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    stats = SystemStats(
        status="ready",
        graph_stats=rag_system.graph_retriever.knowledge_graph.get_graph_stats(),
        vector_stats=rag_system.vector_retriever.get_stats(),
        num_documents=len(rag_system.documents)
    )
    
    return stats


@app.get("/graph/entities")
async def get_entities(limit: int = 50):
    """Get top entities from knowledge graph"""
    if not rag_system:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    entities = rag_system.graph_retriever.knowledge_graph.entities
    
    sorted_entities = sorted(
        entities.values(),
        key=lambda e: e.frequency,
        reverse=True
    )[:limit]
    
    return {
        "entities": [
            {
                "text": e.text,
                "type": e.entity_type,
                "frequency": e.frequency,
                "num_documents": len(e.doc_ids)
            }
            for e in sorted_entities
        ]
    }


@app.get("/graph/relations")
async def get_relations(limit: int = 50):
    """Get top relations from knowledge graph"""
    if not rag_system:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    relations = rag_system.graph_retriever.knowledge_graph.relations
    
    # Sort by weight
    sorted_relations = sorted(
        relations,
        key=lambda r: r.weight,
        reverse=True
    )[:limit]
    
    return {
        "relations": [
            {
                "source": r.source.text,
                "target": r.target.text,
                "type": r.relation_type,
                "weight": r.weight,
                "num_documents": len(r.doc_ids)
            }
            for r in sorted_relations
        ]
    }


def main():
    uvicorn.run(
        "app.backend:app",
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )


if __name__ == "__main__":
    main()

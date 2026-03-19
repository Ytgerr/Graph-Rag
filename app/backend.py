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
from app.vector_store import VectorStore, OpenAIEmbedding
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
    retrieval_mode: str = "graph_rag" 
    use_entity_context: bool = True


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


class GraphRAGSystem:
    
    def __init__(self, documents: List[str], use_embeddings: bool = False):
        self.documents = documents
        
        # Initialize Graph RAG retriever
        logger.info("Initializing Graph RAG retriever...")
        self.graph_retriever = GraphRAGRetriever()
        self.graph_retriever.build_graph(documents)
        
        # Initialize vector store (optional, for hybrid mode)
        self.vector_store = None
        if use_embeddings:
            try:
                logger.info("Initializing vector store with embeddings...")
                embedding_model = OpenAIEmbedding()
                self.vector_store = VectorStore(embedding_model=embedding_model)
                self.vector_store.add_documents(documents)
                logger.info("Vector store initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize embeddings: {e}")
                logger.warning("Continuing without vector store - only graph-based retrieval will be available")
                self.vector_store = None
        
        self.llm = LLMComponent()
        
        logger.info("Graph RAG system initialized successfully")
    
    def query(
        self,
        query: str,
        top_k: int = 5,
        temperature: float = 0.2,
        model: str = "openai/gpt-4o-mini",
        retrieval_mode: str = "graph_rag",
        use_entity_context: bool = True
    ) -> RAGResponse:
        """Process a query through the RAG pipeline"""
        
        try:
            # Retrieve relevant documents based on mode
            if retrieval_mode == "graph_rag":
                # Use Graph RAG hybrid retrieval
                sources, scores, metadata = self.graph_retriever.hybrid_retrieve(
                    query, 
                    top_k=top_k,
                    vector_weight=0.6,
                    graph_weight=0.4
                )
            
            elif retrieval_mode == "vector" and self.vector_store:
                # Use pure vector retrieval
                sources, scores, meta_list = self.vector_store.similarity_search(query, top_k)
                metadata = {"method": "vector_only"}
            
            elif retrieval_mode == "hybrid" and self.vector_store:
                graph_sources, graph_scores, graph_meta = self.graph_retriever.hybrid_retrieve(
                    query, top_k=top_k
                )
                vector_sources, vector_scores, _ = self.vector_store.similarity_search(query, top_k)
                
                combined = {}
                for src, score in zip(graph_sources, graph_scores):
                    combined[src] = score * 0.6
                for src, score in zip(vector_sources, vector_scores):
                    if src in combined:
                        combined[src] += score * 0.4
                    else:
                        combined[src] = score * 0.4
                
                sorted_results = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]
                sources = [src for src, _ in sorted_results]
                scores = [score for _, score in sorted_results]
                metadata = {"method": "full_hybrid"}
            
            else:
                sources, scores, metadata = self.graph_retriever.hybrid_retrieve(query, top_k=top_k)

            context = "\n\n".join(sources)
            
            entity_context = ""
            if use_entity_context:
                entity_context = self.graph_retriever.get_entity_context(query, max_entities=10)
            
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
    title="Graph RAG System API",
    version="2.0.0",
    description="Advanced RAG system with knowledge graph integration"
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
    
    logger.info(f"Initializing Graph RAG system with {len(documents)} documents...")
    
    # Embeddings используют официальный OpenAI API (NOT OpenRouter)
    # OpenRouter не поддерживает embeddings endpoint, только LLM calls
    # Для включения embeddings: ENABLE_EMBEDDINGS=true
    use_embeddings = os.getenv("ENABLE_EMBEDDINGS", "false").lower() == "true"
    if use_embeddings:
        logger.info("Embeddings explicitly enabled - will use official OpenAI API for embeddings")
    else:
        logger.info("Using graph-based retrieval only (embeddings disabled by default)")
    
    rag_system = GraphRAGSystem(documents, use_embeddings=use_embeddings)
    
    logger.info("Graph RAG system ready!")
    
except Exception as e:
    logger.error(f"Failed to initialize RAG system: {e}")
    rag_system = None


@app.post("/query", response_model=RAGResponse)
async def query_rag(request: QueryRequest) -> RAGResponse:
    """Query the Graph RAG system"""
    if not rag_system:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        response = rag_system.query(
            query=request.query,
            top_k=request.top_k,
            temperature=request.temperature,
            model=request.model,
            retrieval_mode=request.retrieval_mode,
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
        vector_stats=rag_system.vector_store.get_stats() if rag_system.vector_store else None,
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

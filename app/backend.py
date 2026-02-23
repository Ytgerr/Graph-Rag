import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import os
from dotenv import load_dotenv
import logging
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


load_dotenv()

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    temperature: float = 0.2
    model: str = "openai/gpt-4o-mini"


class RAGResponse(BaseModel):
    answer: str
    sources: List[str]
    similarity_scores: List[float]


class RetrievalComponent:
    """Vector-based document retrieval system using TF-IDF"""
    
    def __init__(self, method: str = 'vector'):
        self.method = method
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            use_idf=True,
            norm='l2',
            ngram_range=(1, 2),
            sublinear_tf=True,
            analyzer='word'
        )
        self.tfidf_matrix = None
        self.documents = None
    
    def fit(self, records: List[str]) -> None:
        """Fit the vectorizer on documents"""
        self.documents = records
        self.tfidf_matrix = self.vectorizer.fit_transform(records)
        logger.info(f"Fitted retrieval component with {len(records)} documents")
    
    def retrieve_top_k(self, query: str, k: int = 5) -> tuple[List[str], List[float]]:
        """Retrieve top-k most similar documents"""
        if self.tfidf_matrix is None:
            raise ValueError("Retrieval component not fitted. Call fit() first.")
        
        query_tfidf = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_tfidf, self.tfidf_matrix)[0]
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-k:][::-1]
        top_documents = [self.documents[idx] for idx in top_indices]
        top_scores = [float(similarities[idx]) for idx in top_indices]
        
        return top_documents, top_scores

class LLMComponent:
    """OpenAI/OpenRouter LLM interface"""
    
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
        model: str = "openai/gpt-4o-mini",
        temperature: float = 0.2
    ) -> str:
        """Generate answer based on context"""
        system_prompt = (
            "You are an expert Natural Language Processing assistant. "
            "Provide detailed, accurate answers based on the given context. "
            "If the context doesn't contain relevant information, say so."
        )
        
        user_prompt = f"""Based on the following context, answer the question:

CONTEXT:
{context}

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
                max_tokens=1000
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise

class RAGSystem:
    """Complete Retrieval Augmented Generation system"""
    
    def __init__(self, documents: List[str]):
        self.retriever = RetrievalComponent()
        self.retriever.fit(documents)
        
        self.llm = LLMComponent()
        logger.info("RAG system initialized")
    
    def query(
        self,
        query: str,
        top_k: int = 5,
        temperature: float = 0.2,
        model: str = "openai/gpt-4o-mini"
    ) -> RAGResponse:
        """Process a query through RAG pipeline"""
        
        try:

            sources, scores = self.retriever.retrieve_top_k(query, k=top_k)
            context = "\n\n".join(sources)
            

            answer = self.llm.generate(
                query=query,
                context=context,
                model=model,
                temperature=temperature
            )
            
            return RAGResponse(
                answer=answer,
                sources=sources,
                similarity_scores=scores
            )
        
        except Exception as e:
            logger.error(f"RAG query failed: {e}")
            raise


DEFAULT_KNOWLEDGE_BASE = [
    "Retrieval Augmented Generation (RAG) represents a sophisticated hybrid approach in the field of artificial intelligence, particularly within the realm of natural language processing (NLP).",
    "It innovatively combines the capabilities of neural network-based language models with retrieval systems to enhance the generation of text, making it more accurate, informative, and contextually relevant.",
    "This methodology leverages the strengths of both generative and retrieval architectures to tackle complex tasks that require not only linguistic fluency but also factual correctness and depth of knowledge.",
    "At the core of Retrieval Augmented Generation (RAG) is a generative model, typically a transformer-based neural network, similar to those used in models like GPT (Generative Pre-trained Transformer) or BERT (Bidirectional Encoder Representations from Transformers).",
    "This component is responsible for producing coherent and contextually appropriate language outputs based on a mixture of input prompts and additional information fetched by the retrieval component.",
    "Complementing the language model is the retrieval system, which is usually built on a database of documents or a corpus of texts.",
    "This system uses techniques from information retrieval to find and fetch documents that are relevant to the input query or prompt.",
    "The mechanism of relevance determination can range from simple keyword matching to more complex semantic search algorithms which interpret the meaning behind the query to find the best matches.",
    "This component merges the outputs from the language model and the retrieval system.",
    "It effectively synthesizes the raw data fetched by the retrieval system into the generative process of the language model.",
    "The integrator ensures that the information from the retrieval system is seamlessly incorporated into the final text output, enhancing the model's ability to generate responses that are not only fluent and grammatically correct but also rich in factual details and context-specific nuances.",
    "When a query or prompt is received, the system first processes it to understand the requirement or the context.",
    "Based on the processed query, the retrieval system searches through its database to find relevant documents or information snippets.",
    "This retrieval is guided by the similarity of content in the documents to the query, which can be determined through various techniques like vector embeddings or semantic similarity measures.",
    "The retrieved documents are then fed into the language model.",
    "In some implementations, this integration happens at the token level, where the model can access and incorporate specific pieces of information from the retrieved texts dynamically as it generates each part of the response.",
    "The language model, now augmented with direct access to retrieved information, generates a response.",
    "This response is not only influenced by the training of the model but also by the specific facts and details contained in the retrieved documents, making it more tailored and accurate.",
    "By directly incorporating information from external sources, Retrieval Augmented Generation (RAG) models can produce responses that are more factual and relevant to the given query.",
    "This is particularly useful in domains like medical advice, technical support, and other areas where precision and up-to-date knowledge are crucial.",
    "Retrieval Augmented Generation (RAG) systems can dynamically adapt to new information since they retrieve data in real-time from their databases.",
    "This allows them to remain current with the latest knowledge and trends without needing frequent retraining.",
    "With access to a wide range of documents, Retrieval Augmented Generation (RAG) systems can provide detailed and nuanced answers that a standalone language model might not be capable of generating based solely on its pre-trained knowledge.",
    "While Retrieval Augmented Generation (RAG) offers substantial benefits, it also comes with its challenges.",
    "These include the complexity of integrating retrieval and generation systems, the computational overhead associated with real-time data retrieval, and the need for maintaining a large, up-to-date, and high-quality database of retrievable texts.",
    "Furthermore, ensuring the relevance and accuracy of the retrieved information remains a significant challenge, as does managing the potential for introducing biases or errors from the external sources.",
    "In summary, Retrieval Augmented Generation represents a significant advancement in the field of artificial intelligence, merging the best of retrieval-based and generative technologies to create systems that not only understand and generate natural language but also deeply comprehend and utilize the vast amounts of information available in textual form.",
    "A RAG vector store is a database or dataset that contains vectorized data points."
]


app = FastAPI(title="RAG System API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    rag_system = RAGSystem(DEFAULT_KNOWLEDGE_BASE)
    logger.info("RAG system ready")
except Exception as e:
    logger.error(f"Failed to initialize RAG system: {e}")
    rag_system = None

@app.post("/query", response_model=RAGResponse)
async def query_rag(request: QueryRequest) -> RAGResponse:
    """Query the RAG system"""
    if not rag_system:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        response = rag_system.query(
            query=request.query,
            top_k=request.top_k,
            temperature=request.temperature,
            model=request.model
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


def main():
    uvicorn.run(
        "app.backend:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=True,
    )
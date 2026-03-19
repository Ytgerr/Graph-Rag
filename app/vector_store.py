"""
Vector Store Implementation with Embeddings
Supports multiple embedding models and efficient similarity search
"""

import logging
from typing import List, Dict, Tuple, Optional
import numpy as np
from openai import OpenAI
import os
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Base class for embedding models"""
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for texts"""
        raise NotImplementedError
    
    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a single query"""
        return self.embed([query])[0]


class OpenAIEmbedding(EmbeddingModel):
    """OpenAI embedding model"""
    
    def __init__(self, model: str = "text-embedding-3-small", api_key: Optional[str] = None):
        self.model = model
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found")
        
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        logger.info(f"Initialized OpenAI embedding model: {model}")
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts"""
        try:
            # OpenRouter/OpenAI API supports batch embedding
            response = self.client.embeddings.create(
                model=self.model,
                input=texts
            )
            
            embeddings = [item.embedding for item in response.data]
            return np.array(embeddings)
        
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            # Fallback to simple TF-IDF-like representation
            logger.warning("Falling back to TF-IDF representation")
            from sklearn.feature_extraction.text import TfidfVectorizer
            vectorizer = TfidfVectorizer(max_features=1536)
            embeddings = vectorizer.fit_transform(texts).toarray()
            return embeddings


class LocalEmbedding(EmbeddingModel):
    """Local embedding using sentence transformers (fallback)"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            logger.info(f"Initialized local embedding model: {model_name}")
        except ImportError:
            logger.warning("sentence-transformers not installed, using TF-IDF fallback")
            self.model = None
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts"""
        if self.model is None:
            # Fallback to TF-IDF
            from sklearn.feature_extraction.text import TfidfVectorizer
            vectorizer = TfidfVectorizer(max_features=384)
            embeddings = vectorizer.fit_transform(texts).toarray()
            return embeddings
        
        return self.model.encode(texts, show_progress_bar=False)


class VectorStore:
    """
    Vector store for efficient similarity search
    Supports multiple embedding models and persistence
    """
    
    def __init__(
        self, 
        embedding_model: Optional[EmbeddingModel] = None,
        dimension: int = 1536
    ):
        self.embedding_model = embedding_model
        self.dimension = dimension
        self.embeddings: Optional[np.ndarray] = None
        self.documents: List[str] = []
        self.metadata: List[Dict] = []
        self.doc_ids: List[int] = []
    
    def add_documents(
        self, 
        documents: List[str], 
        metadata: Optional[List[Dict]] = None,
        doc_ids: Optional[List[int]] = None
    ):
        """Add documents to the vector store"""
        if not documents:
            return
        
        logger.info(f"Adding {len(documents)} documents to vector store...")
        
        # Generate embeddings
        if self.embedding_model:
            new_embeddings = self.embedding_model.embed(documents)
        else:
            # Use TF-IDF as fallback
            from sklearn.feature_extraction.text import TfidfVectorizer
            vectorizer = TfidfVectorizer(max_features=self.dimension)
            new_embeddings = vectorizer.fit_transform(documents).toarray()
        
        # Add to store
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
        
        self.documents.extend(documents)
        
        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{} for _ in documents])
        
        if doc_ids:
            self.doc_ids.extend(doc_ids)
        else:
            start_id = len(self.doc_ids)
            self.doc_ids.extend(range(start_id, start_id + len(documents)))
        
        logger.info(f"Vector store now contains {len(self.documents)} documents")
    
    def similarity_search(
        self, 
        query: str, 
        top_k: int = 5,
        filter_metadata: Optional[Dict] = None
    ) -> Tuple[List[str], List[float], List[Dict]]:
        """
        Search for similar documents
        
        Args:
            query: Search query
            top_k: Number of results to return
            filter_metadata: Optional metadata filter
        
        Returns:
            Tuple of (documents, scores, metadata)
        """
        if self.embeddings is None or len(self.documents) == 0:
            return [], [], []
        
        # Generate query embedding
        if self.embedding_model:
            query_embedding = self.embedding_model.embed_query(query)
        else:
            # Fallback
            from sklearn.feature_extraction.text import TfidfVectorizer
            vectorizer = TfidfVectorizer(max_features=self.dimension)
            vectorizer.fit(self.documents)
            query_embedding = vectorizer.transform([query]).toarray()[0]
        
        # Calculate similarities
        query_embedding = query_embedding.reshape(1, -1)
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Apply metadata filter if provided
        if filter_metadata:
            valid_indices = []
            for i, meta in enumerate(self.metadata):
                if all(meta.get(k) == v for k, v in filter_metadata.items()):
                    valid_indices.append(i)
            
            if not valid_indices:
                return [], [], []
            
            filtered_similarities = [(i, similarities[i]) for i in valid_indices]
            sorted_results = sorted(filtered_similarities, key=lambda x: x[1], reverse=True)[:top_k]
            indices = [i for i, _ in sorted_results]
            scores = [s for _, s in sorted_results]
        else:
            # Get top-k indices
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            indices = top_indices.tolist()
            scores = similarities[top_indices].tolist()
        
        # Retrieve documents and metadata
        result_docs = [self.documents[i] for i in indices]
        result_metadata = [self.metadata[i] for i in indices]
        
        return result_docs, scores, result_metadata
    
    def get_document(self, doc_id: int) -> Optional[str]:
        """Get document by ID"""
        try:
            idx = self.doc_ids.index(doc_id)
            return self.documents[idx]
        except ValueError:
            return None
    
    def save(self, path: str):
        """Save vector store to disk"""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'embeddings': self.embeddings,
            'documents': self.documents,
            'metadata': self.metadata,
            'doc_ids': self.doc_ids,
            'dimension': self.dimension
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Vector store saved to {path}")
    
    def load(self, path: str):
        """Load vector store from disk"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.embeddings = data['embeddings']
        self.documents = data['documents']
        self.metadata = data['metadata']
        self.doc_ids = data['doc_ids']
        self.dimension = data['dimension']
        
        logger.info(f"Vector store loaded from {path} with {len(self.documents)} documents")
    
    def clear(self):
        """Clear all data from vector store"""
        self.embeddings = None
        self.documents = []
        self.metadata = []
        self.doc_ids = []
        logger.info("Vector store cleared")
    
    def get_stats(self) -> Dict:
        """Get statistics about the vector store"""
        return {
            "num_documents": len(self.documents),
            "embedding_dimension": self.dimension,
            "total_size_mb": self.embeddings.nbytes / (1024 * 1024) if self.embeddings is not None else 0
        }


class HybridVectorStore:
    """
    Hybrid vector store combining dense and sparse representations
    Implements BM25 + Dense embeddings for better retrieval
    """
    
    def __init__(self, embedding_model: Optional[EmbeddingModel] = None):
        self.dense_store = VectorStore(embedding_model=embedding_model)
        self.documents: List[str] = []
        self.bm25_index = None
    
    def add_documents(self, documents: List[str], metadata: Optional[List[Dict]] = None):
        """Add documents to both dense and sparse indices"""
        self.documents = documents
        
        # Add to dense store
        self.dense_store.add_documents(documents, metadata)
        
        # Build BM25 index
        self._build_bm25_index(documents)
    
    def _build_bm25_index(self, documents: List[str]):
        """Build BM25 index for sparse retrieval"""
        try:
            from rank_bm25 import BM25Okapi
            import nltk
            from nltk.tokenize import word_tokenize
            
            # Ensure punkt is downloaded
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
            
            # Tokenize documents
            tokenized_docs = [word_tokenize(doc.lower()) for doc in documents]
            self.bm25_index = BM25Okapi(tokenized_docs)
            logger.info("BM25 index built successfully")
        
        except ImportError:
            logger.warning("rank-bm25 not installed, sparse retrieval disabled")
            self.bm25_index = None
    
    def hybrid_search(
        self, 
        query: str, 
        top_k: int = 5,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3
    ) -> Tuple[List[str], List[float], List[Dict]]:
        """
        Hybrid search combining dense and sparse retrieval
        
        Args:
            query: Search query
            top_k: Number of results
            dense_weight: Weight for dense embeddings (0-1)
            sparse_weight: Weight for BM25 scores (0-1)
        """
        # Normalize weights
        total = dense_weight + sparse_weight
        dense_weight /= total
        sparse_weight /= total
        
        # Dense retrieval
        dense_docs, dense_scores, metadata = self.dense_store.similarity_search(query, top_k * 2)
        
        # Sparse retrieval (BM25)
        if self.bm25_index:
            from nltk.tokenize import word_tokenize
            tokenized_query = word_tokenize(query.lower())
            bm25_scores = self.bm25_index.get_scores(tokenized_query)
            
            # Normalize BM25 scores
            max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
            bm25_scores = bm25_scores / max_bm25
        else:
            bm25_scores = np.zeros(len(self.documents))
        
        # Combine scores
        combined_scores = {}
        
        # Add dense scores
        for doc, score in zip(dense_docs, dense_scores):
            doc_idx = self.documents.index(doc)
            combined_scores[doc_idx] = dense_weight * score
        
        # Add sparse scores
        for idx, score in enumerate(bm25_scores):
            if idx in combined_scores:
                combined_scores[idx] += sparse_weight * score
            else:
                combined_scores[idx] = sparse_weight * score
        
        # Sort and get top-k
        sorted_indices = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        result_docs = [self.documents[idx] for idx, _ in sorted_indices]
        result_scores = [score for _, score in sorted_indices]
        result_metadata = [self.dense_store.metadata[idx] for idx, _ in sorted_indices]
        
        return result_docs, result_scores, result_metadata

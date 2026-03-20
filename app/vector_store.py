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
    
    def embed(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError
    
    def embed_query(self, query: str) -> np.ndarray:
        return self.embed([query])[0]


class OpenAIEmbedding(EmbeddingModel):
    
    def __init__(self, model: str = "openai/text-embedding-3-small", api_key: Optional[str] = None):
        self.model = model
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found")
        
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        logger.info(f"Initialized OpenAI embedding model: {model} (using OpenRouter API)")
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts"""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts
            )
            
            logger.debug(f"API Response: {response}")
            logger.debug(f"Response data: {response.data}")
            
            if not response.data:
                logger.error(f"complete response object: {response}")
                raise ValueError(f"No embedding data in response.")
            
            embeddings = [item.embedding for item in response.data]
            if not embeddings:
                raise ValueError("Failed to extract embeddings from response data")
            
            logger.info(f"Successfully generated {len(embeddings)} embeddings")
            return np.array(embeddings)
        
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            logger.error(f"Model: {self.model}, Texts count: {len(texts)}")
            raise


class VectorStore:

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
        
        if not self.embedding_model:
            raise ValueError("Embedding model must be initialized to add documents")
        
        # Generate embeddings
        new_embeddings = self.embedding_model.embed(documents)
        
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
        if not self.embedding_model:
            raise ValueError("Embedding model must be initialized to perform search")
        
        query_embedding = self.embedding_model.embed_query(query)
        
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


class VectorRAGRetriever:
    """
    Pure vector-based retrieval supporting TF-IDF and BM25 methods
    """
    
    def __init__(self, method: str = "tfidf"):
        """
        Initialize vector retriever
        
        Args:
            method: "tfidf" or "bm25"
        """
        self.method = method.lower()
        self.documents: List[str] = []
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.bm25_index = None
        
        if self.method not in ["tfidf", "bm25"]:
            raise ValueError(f"Method must be 'tfidf' or 'bm25', got '{method}'")
        
        logger.info(f"Initialized VectorRAGRetriever with method: {self.method}")
    
    def build_index(self, documents: List[str]):
        """Build index from documents"""
        logger.info(f"Building {self.method.upper()} index from {len(documents)} documents...")
        self.documents = documents
        
        if self.method == "tfidf":
            self._build_tfidf_index(documents)
        elif self.method == "bm25":
            self._build_bm25_index(documents)
        
        logger.info(f"{self.method.upper()} index built successfully")
    
    def _build_tfidf_index(self, documents: List[str]):
        """Build TF-IDF index"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            use_idf=True,
            norm='l2',
            ngram_range=(1, 3),
            max_features=5000,
            sublinear_tf=True
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
        logger.info(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")
    
    def _build_bm25_index(self, documents: List[str]):
        """Build BM25 index"""
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
            logger.info(f"BM25 index built with {len(documents)} documents")
        
        except ImportError as e:
            logger.error("rank-bm25 not installed. Install with: pip install rank-bm25")
            raise ImportError("rank-bm25 is required for BM25 retrieval") from e
    
    def retrieve(self, query: str, top_k: int = 5) -> Tuple[List[str], List[float], Dict]:
        """
        Retrieve documents using the configured method
        
        Args:
            query: Search query
            top_k: Number of results to return
        
        Returns:
            Tuple of (documents, scores, metadata)
        """
        if not self.documents:
            logger.warning("No documents indexed")
            return [], [], {"method": self.method, "num_documents": 0}
        
        if self.method == "tfidf":
            return self._retrieve_tfidf(query, top_k)
        elif self.method == "bm25":
            return self._retrieve_bm25(query, top_k)
    
    def _retrieve_tfidf(self, query: str, top_k: int) -> Tuple[List[str], List[float], Dict]:
        """Retrieve using TF-IDF"""
        if self.tfidf_vectorizer is None or self.tfidf_matrix is None:
            raise ValueError("TF-IDF index not built. Call build_index() first.")
        
        # Transform query
        query_vector = self.tfidf_vectorizer.transform([query])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, self.tfidf_matrix)[0]
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        top_scores = similarities[top_indices]
        
        # Get documents
        result_docs = [self.documents[i] for i in top_indices]
        result_scores = top_scores.tolist()
        
        metadata = {
            "method": "tfidf",
            "num_documents": len(self.documents),
            "vocab_size": len(self.tfidf_vectorizer.vocabulary_)
        }
        
        return result_docs, result_scores, metadata
    
    def _retrieve_bm25(self, query: str, top_k: int) -> Tuple[List[str], List[float], Dict]:
        """Retrieve using BM25"""
        if self.bm25_index is None:
            raise ValueError("BM25 index not built. Call build_index() first.")
        
        from nltk.tokenize import word_tokenize
        
        # Tokenize query
        tokenized_query = word_tokenize(query.lower())
        
        # Get BM25 scores
        scores = self.bm25_index.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[-top_k:][::-1]
        top_scores = scores[top_indices]
        
        # Normalize scores to [0, 1]
        if top_scores.max() > 0:
            top_scores = top_scores / top_scores.max()
        
        # Get documents
        result_docs = [self.documents[i] for i in top_indices]
        result_scores = top_scores.tolist()
        
        metadata = {
            "method": "bm25",
            "num_documents": len(self.documents),
            "query_terms": len(tokenized_query)
        }
        
        return result_docs, result_scores, metadata
    
    def get_stats(self) -> Dict:
        """Get statistics about the retriever"""
        stats = {
            "method": self.method,
            "num_documents": len(self.documents)
        }
        
        if self.method == "tfidf" and self.tfidf_vectorizer:
            stats["vocab_size"] = len(self.tfidf_vectorizer.vocabulary_)
            stats["matrix_shape"] = self.tfidf_matrix.shape
        
        return stats

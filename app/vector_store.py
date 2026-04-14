import logging
import time
import hashlib
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from openai import OpenAI
import os
import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)


class OpenRouterEmbedding:
    """Embedding client via OpenRouter API with concurrent batch processing."""

    BATCH_SIZE = 100       # texts per API call (OpenAI supports up to 2048)
    MAX_CONCURRENT = 4     # parallel API requests
    MAX_CHARS = 8000
    MAX_RETRIES = 3

    def __init__(self, model: str = "openai/text-embedding-3-small", api_key: Optional[str] = None):
        self.model = model
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")

        self.client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
        self._dimension: Optional[int] = None
        logger.info("OpenRouter embedding: %s (batch=%d, workers=%d)",
                     model, self.BATCH_SIZE, self.MAX_CONCURRENT)

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            test = self.embed(["test"])
            self._dimension = test.shape[1]
        return self._dimension

    def _embed_batch(self, batch: List[str], batch_idx: int) -> Tuple[int, List[List[float]]]:
        """Embed a single batch with retries. Returns (batch_idx, embeddings)."""
        for attempt in range(self.MAX_RETRIES):
            try:
                response = self.client.embeddings.create(model=self.model, input=batch)
                if not response.data:
                    raise ValueError("No embedding data received")
                sorted_data = sorted(response.data, key=lambda x: x.index)
                return batch_idx, [item.embedding for item in sorted_data]
            except Exception as e:
                if attempt < self.MAX_RETRIES - 1:
                    wait = 2 ** (attempt + 1)
                    logger.warning("Embedding batch %d attempt %d failed: %s, retrying in %ds",
                                   batch_idx, attempt + 1, e, wait)
                    time.sleep(wait)
                else:
                    logger.error("Embedding failed for batch %d after %d attempts: %s",
                                 batch_idx, self.MAX_RETRIES, e)
                    raise
        return batch_idx, []  # unreachable, but satisfies type checker

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.array([])

        truncated = [t[:self.MAX_CHARS] for t in texts]

        # Split into batches
        batches = []
        for i in range(0, len(truncated), self.BATCH_SIZE):
            batches.append(truncated[i:i + self.BATCH_SIZE])

        # For small inputs, process sequentially (no thread overhead)
        if len(batches) <= 2:
            all_embeddings: List[List[float]] = []
            for idx, batch in enumerate(batches):
                _, embs = self._embed_batch(batch, idx)
                all_embeddings.extend(embs)
        else:
            # Process batches concurrently
            logger.info("Embedding %d texts in %d batches (%d concurrent)...",
                        len(texts), len(batches), self.MAX_CONCURRENT)
            results: Dict[int, List[List[float]]] = {}
            with ThreadPoolExecutor(max_workers=self.MAX_CONCURRENT) as executor:
                futures = {
                    executor.submit(self._embed_batch, batch, idx): idx
                    for idx, batch in enumerate(batches)
                }
                for future in as_completed(futures):
                    batch_idx, embs = future.result()
                    results[batch_idx] = embs

            # Reassemble in order
            all_embeddings = []
            for idx in range(len(batches)):
                all_embeddings.extend(results[idx])

            logger.info("Embedding complete: %d vectors", len(all_embeddings))

        result = np.array(all_embeddings, dtype=np.float32)
        if self._dimension is None and result.size > 0:
            self._dimension = result.shape[1]
        return result

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed([query])[0]


class VectorRAGRetriever:
    """Vector RAG with proper chunking support.

    Accepts pre-chunked documents, embeds them, and retrieves
    the most relevant chunks for a given query.
    """

    def __init__(
        self,
        embedding_model: str = "openai/text-embedding-3-small",
        collection_name: str = "rag_chunks",
        persist_directory: Optional[str] = None,
    ):
        self.embedding_model_name = embedding_model
        self._embedder = OpenRouterEmbedding(model=embedding_model)

        if persist_directory:
            self._chroma = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(anonymized_telemetry=False),
            )
        else:
            self._chroma = chromadb.EphemeralClient(
                settings=Settings(anonymized_telemetry=False),
            )

        self._collection = self._chroma.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._collection_name = collection_name
        self.chunks: List[str] = []
        logger.info("ChromaDB collection '%s' ready (%d docs)", collection_name, self._collection.count())

    # ChromaDB max batch size for upsert operations
    CHROMA_BATCH_SIZE = 5000

    def build_index(self, chunks: List[str]):
        """Index pre-chunked documents into ChromaDB."""
        self.chunks = chunks

        # Clear old data
        self._chroma.delete_collection(self._collection_name)
        self._collection = self._chroma.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        if not chunks:
            logger.warning("No chunks to index")
            return

        embeddings = self._embedder.embed(chunks)

        ids = [f"chunk_{hashlib.md5(c.encode()).hexdigest()[:12]}_{i}" for i, c in enumerate(chunks)]
        metadatas = [{"index": i, "length": len(c)} for i, c in enumerate(chunks)]

        # Batch upsert to stay within ChromaDB's max batch size
        emb_list = embeddings.tolist()
        for start in range(0, len(chunks), self.CHROMA_BATCH_SIZE):
            end = min(start + self.CHROMA_BATCH_SIZE, len(chunks))
            self._collection.upsert(
                ids=ids[start:end],
                embeddings=emb_list[start:end],
                documents=chunks[start:end],
                metadatas=metadatas[start:end],
            )
            if end < len(chunks):
                logger.info("Indexed batch %d-%d / %d", start, end, len(chunks))

        logger.info("Vector index built: %d chunks indexed", len(chunks))

    def retrieve(self, query: str, top_k: int = 5) -> Tuple[List[str], List[float], Dict]:
        """Retrieve top-k most relevant chunks for a query."""
        if self._collection.count() == 0:
            return [], [], {"method": "vector", "num_chunks": 0, "backend": "ChromaDB"}

        query_embedding = self._embedder.embed_query(query)

        results = self._collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(top_k, self._collection.count()),
            include=["documents", "distances"],
        )

        documents = results["documents"][0] if results["documents"] else []
        distances = results["distances"][0] if results["distances"] else []

        # cosine distance → similarity score
        scores = [max(0.0, 1.0 - d) for d in distances]

        metadata = {
            "method": "vector",
            "embedding_model": self.embedding_model_name,
            "num_chunks": len(self.chunks),
            "num_results": len(documents),
            "backend": "ChromaDB",
        }
        return documents, scores, metadata

    def get_stats(self) -> Dict:
        return {
            "num_documents": self._collection.count(),
            "collection_name": self._collection_name,
            "embedding_model": self.embedding_model_name,
            "embedding_dimension": self._embedder._dimension or "unknown",
            "backend": "ChromaDB",
        }

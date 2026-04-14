"""Graph RAG using LlamaIndex PropertyGraphIndex + Neo4j.
"""

import hashlib
import logging
import os
import time
from typing import List, Dict, Tuple, Optional

import numpy as np
from llama_index.core import PropertyGraphIndex, Document, Settings
from llama_index.core.indices.property_graph import (
    LLMSynonymRetriever,
    VectorContextRetriever,
    SimpleLLMPathExtractor,
)
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.openai import OpenAIEmbedding

logger = logging.getLogger(__name__)


def _make_llm(model: str = "openai/gpt-4o-mini", temperature: float = 0.0) -> OpenAILike:
    """Create an OpenRouter-compatible LLM for LlamaIndex."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    return OpenAILike(
        model=model,
        api_key=api_key,
        api_base="https://openrouter.ai/api/v1",
        temperature=temperature,
        max_tokens=2048,
        is_chat_model=True,
    )


def _make_embed_model(model: str = "text-embedding-3-small") -> OpenAIEmbedding:
    """Create an OpenRouter-compatible embedding model."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    return OpenAIEmbedding(
        model=model,
        api_key=api_key,
        api_base="https://openrouter.ai/api/v1",
    )


def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.array(vec_a, dtype=np.float32)
    b = np.array(vec_b, dtype=np.float32)
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(dot / norm) if norm > 0 else 0.0


class GraphRAGRetriever:
    """Graph RAG powered by LlamaIndex PropertyGraphIndex.

    Build pipeline:
        raw documents -> merge small docs -> LlamaIndex Documents
        -> PropertyGraphIndex -> LLM extracts entities & relations
        -> stored in Neo4j with embeddings

    Retrieval pipeline:
        query -> LLMSynonymRetriever (entity expansion via LLM)
              -> VectorContextRetriever (embedding-based node lookup)
              -> merge & deduplicate -> re-rank by cosine similarity
              -> top_k results -> context for answer generation
    """

    # Number of parallel LLM workers for entity extraction
    DEFAULT_NUM_WORKERS = 8
    # Merge target: combine small paragraphs into batches of this size
    DEFAULT_MERGE_TARGET = 4000

    def __init__(
        self,
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None,
        llm_model: str = "openai/gpt-4o-mini",
        max_triplets_per_chunk: int = 3,
        num_workers: int = DEFAULT_NUM_WORKERS,
    ):
        self.neo4j_uri = neo4j_uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_user = neo4j_user or os.getenv("NEO4J_USER", "neo4j")
        self.neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD", "password")
        self.llm_model = llm_model
        self.max_triplets_per_chunk = max_triplets_per_chunk
        self.num_workers = num_workers

        # LlamaIndex global settings
        self.llm = _make_llm(model=llm_model)
        embed_model_name = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        if "/" in embed_model_name:
            embed_model_name = embed_model_name.split("/", 1)[1]
        self.embed_model = _make_embed_model(model=embed_model_name)

        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        Settings.chunk_size = 2048   # larger chunks = fewer internal splits
        Settings.chunk_overlap = 128

        self._graph_store: Optional[Neo4jPropertyGraphStore] = None
        self._index: Optional[PropertyGraphIndex] = None
        self._synonym_retriever: Optional[LLMSynonymRetriever] = None
        self._vector_retriever: Optional[VectorContextRetriever] = None
        self.documents: List[str] = []

        logger.info("GraphRAGRetriever initialized (LlamaIndex + Neo4j, triplets=%d, workers=%d)",
                     max_triplets_per_chunk, num_workers)

    def _create_graph_store(self) -> Neo4jPropertyGraphStore:
        """Create a fresh Neo4j property graph store."""
        return Neo4jPropertyGraphStore(
            username=self.neo4j_user,
            password=self.neo4j_password,
            url=self.neo4j_uri,
        )

    # ── Dataset hash tracking ──

    @staticmethod
    def _compute_dataset_hash(documents: List[str]) -> str:
        """Compute a hash of the dataset to detect changes."""
        content = "\n".join(documents)
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

    def _get_stored_hash(self) -> Optional[str]:
        """Get the dataset hash stored in Neo4j (if any)."""
        try:
            store = self._create_graph_store()
            result = store.structured_query(
                "MATCH (m:__DatasetMeta__) RETURN m.hash AS hash LIMIT 1"
            )
            if result and len(result) > 0:
                row = result[0] if isinstance(result, list) else result
                return row.get("hash") if isinstance(row, dict) else None
        except Exception:
            pass
        return None

    def _store_hash(self, dataset_hash: str):
        """Store the dataset hash in Neo4j for future comparison."""
        try:
            self._graph_store.structured_query(
                "MERGE (m:__DatasetMeta__) SET m.hash = $hash",
                param_map={"hash": dataset_hash},
            )
        except Exception as e:
            logger.warning("Could not store dataset hash: %s", e)

    def _graph_has_matching_data(self, documents: List[str]) -> bool:
        """Check if Neo4j has graph data built from the same dataset."""
        current_hash = self._compute_dataset_hash(documents)
        stored_hash = self._get_stored_hash()
        if stored_hash and stored_hash == current_hash:
            logger.info("Dataset hash matches Neo4j (%s), graph is up to date", current_hash)
            return True
        if stored_hash:
            logger.info("Dataset hash changed: stored=%s, current=%s — rebuild needed",
                        stored_hash, current_hash)
        else:
            logger.info("No dataset hash in Neo4j — checking if graph has any data")
            try:
                store = self._create_graph_store()
                result = store.structured_query("MATCH (n) RETURN count(n) AS cnt LIMIT 1")
                if result and len(result) > 0:
                    row = result[0] if isinstance(result, list) else result
                    cnt = row.get("cnt", 0) if isinstance(row, dict) else 0
                    if cnt > 0:
                        logger.info("Neo4j has %d nodes but no hash — will rebuild to track hash", cnt)
                        return False
            except Exception:
                pass
        return False

    # ── Document merging ──

    @staticmethod
    def _merge_small_docs(documents: List[str], target_size: int = 4000) -> List[str]:
        """Merge small documents into larger batches to reduce LLM calls.

        Each LLM extraction call has overhead, so combining small paragraphs
        (e.g. 100-300 chars) into ~4000-char batches significantly reduces
        total calls while preserving all content.
        """
        merged: List[str] = []
        current = ""
        for doc in documents:
            if len(current) + len(doc) + 2 <= target_size:
                current = f"{current}\n\n{doc}".strip()
            else:
                if current:
                    merged.append(current)
                current = doc
        if current:
            merged.append(current)
        return merged

    # ── Retriever caching ──

    def _build_retrievers(self, top_k: int = 5):
        """Build and cache the retriever instances.

        Caching avoids re-creating retrievers on every query call.
        """
        if not self._index:
            return

        self._synonym_retriever = LLMSynonymRetriever(
            self._index.property_graph_store,
            llm=self.llm,
            include_text=True,
            max_keywords=10,
        )

        self._vector_retriever = VectorContextRetriever(
            self._index.property_graph_store,
            embed_model=self.embed_model,
            include_text=True,
            similarity_top_k=top_k * 2,  # fetch more for re-ranking
        )

        logger.info("Retrievers built (synonym + vector, fetch_k=%d)", top_k * 2)

    # ── Build ──

    def build_graph(self, documents: List[str], force_rebuild: bool = False):
        """Build the knowledge graph from raw documents using LlamaIndex.

        Optimizations:
        - Skips rebuild if Neo4j already has data with matching hash
        - Merges small documents into ~4000-char batches to reduce LLM API calls
        - Parallel LLM extraction via SimpleLLMPathExtractor(num_workers=N)
        - Async processing enabled (use_async=True)
        - Configurable max_triplets_per_chunk for extraction density
        - Includes embeddings for hybrid retrieval

        LlamaIndex will:
        1. Chunk documents (using Settings.chunk_size=2048)
        2. Extract entities and relations via LLM (SimpleLLMPathExtractor)
        3. Store everything in Neo4j with embeddings
        """
        self.documents = documents

        # Check if graph already exists in Neo4j with matching dataset
        if not force_rebuild and self._graph_has_matching_data(documents):
            logger.info("Graph is up to date, skipping rebuild.")
            self._graph_store = self._create_graph_store()
            self._index = PropertyGraphIndex.from_existing(
                property_graph_store=self._graph_store,
                llm=self.llm,
                embed_model=self.embed_model,
            )
            self._build_retrievers()
            logger.info("Connected to existing graph index")
            return

        t0 = time.time()
        dataset_hash = self._compute_dataset_hash(documents)
        logger.info("Building knowledge graph from %d documents (hash=%s) via LlamaIndex...",
                     len(documents), dataset_hash)

        # Merge small docs into larger batches to reduce total LLM calls
        merged_docs = self._merge_small_docs(documents, target_size=self.DEFAULT_MERGE_TARGET)
        logger.info("Merged %d documents into %d batches for extraction "
                     "(triplets=%d, workers=%d)",
                     len(documents), len(merged_docs),
                     self.max_triplets_per_chunk, self.num_workers)

        # Clear old data and recreate graph store
        self._graph_store = self._create_graph_store()
        try:
            self._graph_store.structured_query("MATCH (n) DETACH DELETE n")
            logger.info("Cleared old graph data from Neo4j")
        except Exception as e:
            logger.warning("Could not clear Neo4j: %s", e)

        # Convert to LlamaIndex Documents
        llama_docs = [
            Document(text=doc, metadata={"batch_id": i})
            for i, doc in enumerate(merged_docs)
        ]

        # Create explicit extractor with parallel workers for speed
        kg_extractor = SimpleLLMPathExtractor(
            llm=self.llm,
            max_paths_per_chunk=self.max_triplets_per_chunk,
            num_workers=self.num_workers,
        )

        # Build PropertyGraphIndex — parallel LLM extraction + embedding storage
        self._index = PropertyGraphIndex.from_documents(
            llama_docs,
            property_graph_store=self._graph_store,
            kg_extractors=[kg_extractor],
            llm=self.llm,
            embed_model=self.embed_model,
            embed_kg_nodes=True,
            use_async=True,
            show_progress=True,
        )

        # Store dataset hash for future comparison
        self._store_hash(dataset_hash)

        # Build cached retrievers
        self._build_retrievers()

        elapsed = time.time() - t0
        stats = self.get_graph_stats()
        logger.info("Knowledge graph built in %.1fs: %s", elapsed, stats)

    # ── Retrieve with re-ranking ──

    def retrieve(self, query: str, top_k: int = 5) -> Tuple[List[str], List[float], Dict]:
        """Retrieve relevant context from the knowledge graph.

        Pipeline:
        1. LLMSynonymRetriever: LLM generates synonym/related entity names
        2. VectorContextRetriever: embedding similarity on graph nodes
        3. Merge and deduplicate results
        4. Re-rank by cosine similarity between query embedding and node text
        5. Return top_k results
        """
        if not self._index:
            return [], [], {"method": "graph", "error": "Index not built"}

        try:
            # Ensure retrievers exist
            if not self._synonym_retriever or not self._vector_retriever:
                self._build_retrievers(top_k)

            # Retrieve from both strategies
            synonym_nodes = self._synonym_retriever.retrieve(query)
            vector_nodes = self._vector_retriever.retrieve(query)

            # Merge and deduplicate
            seen_texts = set()
            all_nodes = []
            for node in synonym_nodes + vector_nodes:
                text = node.text.strip()
                if text and text not in seen_texts:
                    seen_texts.add(text)
                    all_nodes.append(node)

            if not all_nodes:
                return [], [], {"method": "graph", "num_results": 0}

            # Re-rank by cosine similarity to query embedding
            try:
                query_embedding = self.embed_model.get_query_embedding(query)
                scored_nodes = []
                for node in all_nodes:
                    # Get node text embedding for re-ranking
                    node_embedding = self.embed_model.get_text_embedding(node.text[:512])
                    sim = _cosine_similarity(query_embedding, node_embedding)
                    scored_nodes.append((node, sim))

                # Sort by re-ranked similarity (descending)
                scored_nodes.sort(key=lambda x: x[1], reverse=True)
                top_nodes = scored_nodes[:top_k]

                sources = [node.text for node, _ in top_nodes]
                scores = [score for _, score in top_nodes]
            except Exception as e:
                logger.warning("Re-ranking failed, falling back to original scores: %s", e)
                # Fallback: use original scores
                all_nodes.sort(
                    key=lambda n: n.score if n.score is not None else 0.0,
                    reverse=True,
                )
                top_nodes_fallback = all_nodes[:top_k]
                sources = [node.text for node in top_nodes_fallback]
                scores = [
                    node.score if node.score is not None else 0.5
                    for node in top_nodes_fallback
                ]

            # Normalize scores to [0, 1]
            max_score = max(scores) if scores else 1.0
            if max_score > 0:
                scores = [s / max_score for s in scores]

            metadata = {
                "method": "graph",
                "num_results": len(sources),
                "synonym_results": len(synonym_nodes),
                "vector_results": len(vector_nodes),
                "total_candidates": len(all_nodes),
                "reranked": True,
                "backend": "Neo4j + LlamaIndex",
            }

            return sources, scores, metadata

        except Exception as e:
            logger.error("Graph retrieval failed: %s", e)
            return [], [], {"method": "graph", "error": str(e)}

    # ── Stats ──

    def get_graph_stats(self) -> Dict:
        """Get statistics about the knowledge graph."""
        if not self._graph_store:
            return {
                "num_entities": 0, "num_relations": 0,
                "num_documents": 0, "backend": "Neo4j + LlamaIndex",
            }

        try:
            query = (
                "MATCH (n) WITH count(n) AS node_count "
                "OPTIONAL MATCH ()-[r]->() "
                "RETURN node_count, count(r) AS rel_count"
            )
            result = self._graph_store.structured_query(query)

            if result and len(result) > 0:
                row = result[0] if isinstance(result, list) else result
                if isinstance(row, dict):
                    num_nodes = row.get("node_count", 0)
                    num_rels = row.get("rel_count", 0)
                else:
                    num_nodes = 0
                    num_rels = 0
            else:
                num_nodes = 0
                num_rels = 0

            return {
                "num_entities": num_nodes,
                "num_relations": num_rels,
                "num_documents": len(self.documents),
                "max_triplets_per_chunk": self.max_triplets_per_chunk,
                "backend": "Neo4j + LlamaIndex",
            }
        except Exception as e:
            logger.error("Failed to get graph stats: %s", e)
            return {
                "num_entities": 0,
                "num_relations": 0,
                "num_documents": len(self.documents),
                "backend": "Neo4j + LlamaIndex",
                "error": str(e),
            }

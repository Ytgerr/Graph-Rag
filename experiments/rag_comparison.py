#!/usr/bin/env python3
"""
Сравнение Graph RAG (Neo4j) vs Vector RAG (ChromaDB).
Замеряет latency и качество retrieval на тестовых запросах.
"""

import time
import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from app.graph_rag import GraphRAGRetriever
from app.vector_store import VectorRAGRetriever
from app.document_processor import load_default_knowledge_base

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TEST_QUERIES = [
    "What is RAG and how does it work?",
    "Which models does OpenAI use?",
    "Tell me about transformers and attention mechanism",
    "How does knowledge graph improve search?",
    "What are embeddings and why are they needed?",
    "Difference between GPT-3 and GPT-4",
    "How does prompt engineering work?",
    "What is model fine-tuning?",
]


class RAGExperiment:

    def __init__(self, embedding_model: str = "openai/text-embedding-3-small"):
        self.documents = load_default_knowledge_base()
        logger.info(f"Loaded {len(self.documents)} documents")

        logger.info("Building Graph RAG (Neo4j)...")
        self.graph_retriever = GraphRAGRetriever()
        self.graph_retriever.build_graph(self.documents)

        logger.info(f"Building Vector RAG (ChromaDB, {embedding_model})...")
        self.vector_retriever = VectorRAGRetriever(
            embedding_model=embedding_model, collection_name="rag_experiment"
        )
        self.vector_retriever.build_index(self.documents)

    def run_experiment(self, queries: List[str] = None, top_k: int = 5) -> Dict:
        if queries is None:
            queries = TEST_QUERIES

        results = {
            "graph": {"queries": [], "avg_latency": 0, "total_latency": 0},
            "vector": {"queries": [], "avg_latency": 0, "total_latency": 0},
        }

        for idx, query in enumerate(queries, 1):
            logger.info(f"[{idx}/{len(queries)}] {query[:50]}...")

            for mode, retriever in [("graph", self.graph_retriever), ("vector", self.vector_retriever)]:
                t0 = time.time()
                sources, scores, metadata = retriever.retrieve(query, top_k=top_k)
                latency = time.time() - t0

                results[mode]["queries"].append({
                    "query": query,
                    "sources_count": len(sources),
                    "latency": latency,
                    "metadata": metadata,
                    "sources": sources[:2]
                })
                results[mode]["total_latency"] += latency

        for mode in results:
            n = len(results[mode]["queries"])
            if n:
                results[mode]["avg_latency"] = results[mode]["total_latency"] / n * 1000

        return results

    def print_results(self, results: Dict):
        print(f"\n{'='*80}")
        print("RESULTS".center(80))
        print(f"{'='*80}\n")
        print(f"{'Method':<25} {'Avg Latency (ms)':<20} {'Queries':<15} {'Avg Sources':<15}")
        print("-" * 80)

        for mode in ["graph", "vector"]:
            qs = results[mode]["queries"]
            if not qs:
                continue
            label = f"{mode.upper()} ({'Neo4j' if mode == 'graph' else 'ChromaDB'})"
            avg_src = sum(q["sources_count"] for q in qs) / len(qs)
            print(f"{label:<25} {results[mode]['avg_latency']:<20.2f} {len(qs):<15} {avg_src:<15.1f}")

    def save_results(self, results: Dict, filename: str = "experiment_results.json"):
        output_path = Path(__file__).parent / filename
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=int, default=len(TEST_QUERIES))
    parser.add_argument("--custom-query", type=str)
    parser.add_argument("--embedding-model", type=str, default="openai/text-embedding-3-small")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    experiment = RAGExperiment(embedding_model=args.embedding_model)

    if args.custom_query:
        results = experiment.run_experiment([args.custom_query], top_k=args.top_k)
    else:
        results = experiment.run_experiment(TEST_QUERIES[:args.queries], top_k=args.top_k)

    experiment.print_results(results)
    experiment.save_results(results)

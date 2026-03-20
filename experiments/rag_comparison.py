#!/usr/bin/env python3
"""
RAG Comparison Experiment

Compare performance of Graph RAG vs Vector RAG (TF-IDF/BM25)
Measures retrieval latency and quality across different query types
"""

import time
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.graph_rag import GraphRAGRetriever
from app.vector_store import VectorRAGRetriever
from app.document_processor import load_default_knowledge_base

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============ TEST QUERIES ============
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
    def __init__(self, vector_method: str = "tfidf"):
        """
        Initialize experiment
        
        Args:
            vector_method: "tfidf" or "bm25"
        """
        logger.info("Initializing experiment...")
        
        # Load documents
        self.documents = load_default_knowledge_base()
        logger.info(f"Loaded {len(self.documents)} documents")
        
        # Initialize Graph RAG
        logger.info("Building Graph RAG...")
        self.graph_retriever = GraphRAGRetriever()
        self.graph_retriever.build_graph(self.documents)
        
        # Initialize Vector RAG
        logger.info(f"Building Vector RAG ({vector_method.upper()})...")
        self.vector_retriever = VectorRAGRetriever(method=vector_method)
        self.vector_retriever.build_index(self.documents)
        
        logger.info("Experiment ready!")
    
    def graph_retrieval(self, query: str, top_k: int = 5) -> Tuple[List[str], float, Dict]:
        """Graph-based retrieval"""
        start_time = time.time()
        
        sources, scores, metadata = self.graph_retriever.retrieve(query, top_k=top_k)
        
        latency = time.time() - start_time
        return sources, latency, metadata
    
    def vector_retrieval(self, query: str, top_k: int = 5) -> Tuple[List[str], float, Dict]:
        """Vector-based retrieval"""
        start_time = time.time()
        
        sources, scores, metadata = self.vector_retriever.retrieve(query, top_k=top_k)
        
        latency = time.time() - start_time
        return sources, latency, metadata
    
    def run_experiment(self, queries: List[str] = None, top_k: int = 5) -> Dict:
        """Run full experiment"""
        if queries is None:
            queries = TEST_QUERIES
        
        results = {
            "graph": {"queries": [], "avg_latency": 0, "total_latency": 0},
            "vector": {"queries": [], "avg_latency": 0, "total_latency": 0},
        }
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Running experiment on {len(queries)} queries")
        logger.info(f"{'='*60}\n")
        
        for idx, query in enumerate(queries, 1):
            logger.info(f"[{idx}/{len(queries)}] Query: {query[:50]}...")
            
            # Graph retrieval
            graph_sources, graph_latency, graph_meta = self.graph_retrieval(query, top_k)
            results["graph"]["queries"].append({
                "query": query,
                "sources_count": len(graph_sources),
                "latency": graph_latency,
                "metadata": graph_meta,
                "sources": graph_sources[:2]  # Top 2 for display
            })
            results["graph"]["total_latency"] += graph_latency
            
            # Vector retrieval
            vector_sources, vector_latency, vector_meta = self.vector_retrieval(query, top_k)
            results["vector"]["queries"].append({
                "query": query,
                "sources_count": len(vector_sources),
                "latency": vector_latency,
                "metadata": vector_meta,
                "sources": vector_sources[:2]
            })
            results["vector"]["total_latency"] += vector_latency
        
        # Calculate averages
        for mode in results:
            if results[mode]["queries"]:
                results[mode]["avg_latency"] = results[mode]["total_latency"] / len(results[mode]["queries"]) * 1000  # ms
        
        return results
    
    def print_results(self, results: Dict):
        """Print experiment results"""
        print(f"\n{'='*80}")
        print("EXPERIMENT RESULTS".center(80))
        print(f"{'='*80}\n")
        
        print(f"{'Method':<20} {'Avg Latency (ms)':<20} {'Queries':<15} {'Avg Sources':<15}")
        print(f"{'-'*80}")
        
        for mode in ["graph", "vector"]:
            if not results[mode]["queries"]:
                continue
            
            avg_latency = results[mode]["avg_latency"]
            queries_count = len(results[mode]["queries"])
            avg_sources = sum(q["sources_count"] for q in results[mode]["queries"]) / queries_count
            
            print(f"{mode.upper():<20} {avg_latency:<20.2f} {queries_count:<15} {avg_sources:<15.1f}")
        
        print(f"\n{'='*80}")
        print("\nDETAILED RESULTS BY QUERY:\n")
        
        for mode in ["graph", "vector"]:
            if not results[mode]["queries"]:
                logger.info(f"❌ {mode.upper()} - not available")
                continue
            
            logger.info(f"\n{mode.upper()} RETRIEVAL:")
            logger.info(f"{'-'*80}")
            
            for i, query_result in enumerate(results[mode]["queries"], 1):
                logger.info(f"  [{i}] Query: {query_result['query'][:60]}")
                logger.info(f"      Latency: {query_result['latency']*1000:.2f}ms | Sources: {query_result['sources_count']}")
                
                # Show metadata
                metadata = query_result.get('metadata', {})
                if 'query_entities' in metadata:
                    logger.info(f"      Entities: {metadata['query_entities']} query, {metadata.get('matched_entities', 0)} matched")
                elif 'query_terms' in metadata:
                    logger.info(f"      Query terms: {metadata['query_terms']}")
    
    def save_results(self, results: Dict, filename: str = "experiment_results.json"):
        """Save results to JSON"""
        output_path = Path(__file__).parent / filename
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n✅ Results saved to: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG Comparison Experiment")
    parser.add_argument(
        "--queries",
        type=int,
        default=len(TEST_QUERIES),
        help="Number of test queries"
    )
    parser.add_argument(
        "--custom-query",
        type=str,
        help="Custom query for single test"
    )
    parser.add_argument(
        "--vector-method",
        type=str,
        default="tfidf",
        choices=["tfidf", "bm25"],
        help="Vector retrieval method (default: tfidf)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of documents to retrieve (default: 5)"
    )
    args = parser.parse_args()
    
    experiment = RAGExperiment(vector_method=args.vector_method)
    
    if args.custom_query:
        logger.info(f"Testing custom query: {args.custom_query}")
        results = experiment.run_experiment([args.custom_query], top_k=args.top_k)
    else:
        results = experiment.run_experiment(TEST_QUERIES[:args.queries], top_k=args.top_k)
    
    experiment.print_results(results)
    experiment.save_results(results)

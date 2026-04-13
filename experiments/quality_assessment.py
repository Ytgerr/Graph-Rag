"""
Оценка качества Graph RAG (Neo4j) vs Vector RAG (ChromaDB).
Ручная оценка релевантности по шкале 1-5.
"""

import json
import sys
from pathlib import Path
from typing import Dict
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from app.graph_rag import GraphRAGRetriever
from app.vector_store import VectorRAGRetriever
from app.document_processor import load_default_knowledge_base
import logging

logging.basicConfig(level=logging.WARNING)

TEST_QUERIES = [
    {"query": "What is retrieval augmented generation?", "type": "entity-centric", "expected_strength": "graph"},
    {"query": "How does RAG reduce AI hallucinations?", "type": "relational", "expected_strength": "graph"},
    {"query": "Explain the benefits of using external data sources in LLMs", "type": "semantic", "expected_strength": "vector"},
    {"query": "What is OpenAI and what products does it develop?", "type": "entity-centric", "expected_strength": "graph"},
    {"query": "Describe information retrieval techniques in AI systems", "type": "keyword", "expected_strength": "vector"},
    {"query": "How do knowledge graphs improve search and retrieval?", "type": "relational", "expected_strength": "graph"},
]


class QualityAssessor:

    def __init__(self, embedding_model: str = "openai/text-embedding-3-small"):
        print(f"\n{'='*80}\n{'QUALITY ASSESSMENT'.center(80)}\n{'='*80}\n")

        self.documents = load_default_knowledge_base()
        print(f"Loaded {len(self.documents)} documents")

        print("Building Graph RAG (Neo4j)...")
        self.graph_retriever = GraphRAGRetriever()
        self.graph_retriever.build_graph(self.documents)

        print(f"Building Vector RAG (ChromaDB, {embedding_model})...")
        self.vector_retriever = VectorRAGRetriever(
            embedding_model=embedding_model, collection_name="rag_quality_assessment"
        )
        self.vector_retriever.build_index(self.documents)

        self.embedding_model = embedding_model
        self.assessments = []
        print("\nReady\n")

    def get_results(self, query: str, top_k: int = 3) -> Dict:
        results = {}
        for mode, retriever in [("graph", self.graph_retriever), ("vector", self.vector_retriever)]:
            sources, scores, meta = retriever.retrieve(query, top_k=top_k)
            results[mode] = {"sources": sources[:top_k], "scores": scores[:top_k], "metadata": meta}
        return results

    def display_results(self, query_info: Dict, results: Dict):
        print(f"\n{'='*80}")
        print(f"QUERY: {query_info['query']}")
        print(f"Type: {query_info['type']} | Expected: {query_info['expected_strength'].upper()}")
        print("=" * 80)

        for mode in ["graph", "vector"]:
            if mode not in results:
                continue
            backend = results[mode]["metadata"].get("backend", "N/A")
            print(f"\n[{mode.upper()} — {backend}]")
            print("-" * 80)

            sources, scores = results[mode]["sources"], results[mode]["scores"]
            if not sources:
                print("  No results")
                continue

            for i, source in enumerate(sources, 1):
                score = scores[i - 1] if i - 1 < len(scores) else "N/A"
                text = source[:200] + "..." if len(source) > 200 else source
                if isinstance(score, float):
                    print(f"\n  [{i}] Score: {score:.3f}")
                else:
                    print(f"\n  [{i}]")
                print(f"  {text}")

    def collect_ratings(self, query_info: Dict, results: Dict) -> Dict:
        assessment = {
            "query": query_info["query"], "query_type": query_info["type"],
            "expected_strength": query_info["expected_strength"],
            "timestamp": datetime.now().isoformat(), "ratings": {}
        }

        print(f"\n{'-'*80}\nRate relevance (1-5): 1=irrelevant, 5=perfect\n{'-'*80}")

        for mode in ["graph", "vector"]:
            if mode not in results or not results[mode]["sources"]:
                continue
            backend = "Neo4j" if mode == "graph" else "ChromaDB"
            while True:
                try:
                    rating = int(input(f"Rate {mode.upper()} ({backend}) (1-5): ").strip())
                    if 1 <= rating <= 5:
                        assessment["ratings"][mode] = rating
                        break
                    print("Enter 1-5")
                except ValueError:
                    print("Enter a number 1-5")
                except KeyboardInterrupt:
                    print("\nInterrupted")
                    sys.exit(0)

        return assessment

    def run_assessment(self):
        print(f"Queries: {len(TEST_QUERIES)}\n")

        for idx, qi in enumerate(TEST_QUERIES, 1):
            print(f"\n{'='*80}\n{'QUERY ' + str(idx) + '/' + str(len(TEST_QUERIES)):^80}\n{'='*80}")
            results = self.get_results(qi["query"], top_k=3)
            self.display_results(qi, results)
            self.assessments.append(self.collect_ratings(qi, results))

        self.display_summary()
        self.save_results()

    def display_summary(self):
        if not self.assessments:
            return

        print(f"\n\n{'='*80}\n{'SUMMARY'.center(80)}\n{'='*80}\n")

        graph_r = [a["ratings"]["graph"] for a in self.assessments if "graph" in a["ratings"]]
        vector_r = [a["ratings"]["vector"] for a in self.assessments if "vector" in a["ratings"]]

        print(f"{'Method':<25} {'Avg':<10} {'N':<10} {'Min':<10} {'Max':<10}")
        print("-" * 65)
        if graph_r:
            print(f"{'GRAPH (Neo4j)':<25} {sum(graph_r)/len(graph_r):<10.2f} {len(graph_r):<10} {min(graph_r):<10} {max(graph_r):<10}")
        if vector_r:
            print(f"{'VECTOR (ChromaDB)':<25} {sum(vector_r)/len(vector_r):<10.2f} {len(vector_r):<10} {min(vector_r):<10} {max(vector_r):<10}")

        graph_wins = vector_wins = ties = 0
        for a in self.assessments:
            if "graph" in a["ratings"] and "vector" in a["ratings"]:
                g, v = a["ratings"]["graph"], a["ratings"]["vector"]
                if g > v: graph_wins += 1
                elif v > g: vector_wins += 1
                else: ties += 1

        total = graph_wins + vector_wins + ties
        if total:
            print(f"\nGraph wins: {graph_wins}/{total}, Vector wins: {vector_wins}/{total}, Ties: {ties}/{total}")

    def save_results(self):
        output_path = Path(__file__).parent / "quality_assessment_results.json"
        data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "graph_backend": "Neo4j", "vector_backend": "ChromaDB",
                "embedding_model": self.embedding_model,
            },
            "assessments": self.assessments
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Saved to: {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-model", default="openai/text-embedding-3-small")
    args = parser.parse_args()

    try:
        QualityAssessor(embedding_model=args.embedding_model).run_assessment()
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(0)

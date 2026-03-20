"""
Quality Assessment for RAG Systems

Systematic evaluation of Graph RAG vs Vector RAG (TF-IDF/BM25)
with predefined test queries and human feedback collection.

Rating scale 1-5:
1 = Completely Irrelevant
2 = Poorly Relevant
3 = Moderately Relevant
4 = Highly Relevant
5 = Perfect Match
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.graph_rag import GraphRAGRetriever
from app.vector_store import VectorRAGRetriever
from app.document_processor import load_default_knowledge_base
import logging

logging.basicConfig(level=logging.WARNING)

TEST_QUERIES = [
    {
        "query": "What is retrieval augmented generation?",
        "type": "entity-centric",
        "expected_strength": "graph",
        "description": "Tests entity recognition for RAG concept"
    },
    {
        "query": "How does RAG reduce AI hallucinations?",
        "type": "relational",
        "expected_strength": "graph",
        "description": "Tests relationship between RAG and hallucination reduction"
    },
    {
        "query": "Explain the benefits of using external data sources in LLMs",
        "type": "semantic",
        "expected_strength": "vector",
        "description": "Tests semantic understanding without specific entities"
    },
    {
        "query": "What is OpenAI and what products does it develop?",
        "type": "entity-centric",
        "expected_strength": "graph",
        "description": "Tests entity extraction and related information retrieval"
    },
    {
        "query": "Describe information retrieval techniques in AI systems",
        "type": "keyword",
        "expected_strength": "vector",
        "description": "Tests keyword-based retrieval for general concepts"
    },
    {
        "query": "How do knowledge graphs improve search and retrieval?",
        "type": "relational",
        "expected_strength": "graph",
        "description": "Tests understanding of knowledge graph benefits"
    }
]


class QualityAssessor:
    def __init__(self, vector_method: str = "tfidf"):
        """
        Initialize quality assessor
        
        Args:
            vector_method: "tfidf" or "bm25"
        """
        print("\n" + "="*80)
        print("QUALITY ASSESSMENT SYSTEM".center(80))
        print("="*80 + "\n")
        
        print(f"Initializing retrievers...")
        
        # Load documents
        self.documents = load_default_knowledge_base()
        print(f"Loaded {len(self.documents)} documents")
        
        # Initialize Graph RAG
        print("Building Graph RAG...")
        self.graph_retriever = GraphRAGRetriever()
        self.graph_retriever.build_graph(self.documents)
        
        # Initialize Vector RAG
        print(f"Building Vector RAG ({vector_method.upper()})...")
        self.vector_retriever = VectorRAGRetriever(method=vector_method)
        self.vector_retriever.build_index(self.documents)
        
        self.vector_method = vector_method
        self.assessments = []
        
        print("\nSystem ready for evaluation\n")
    
    def get_results(self, query: str, top_k: int = 3) -> Dict:
        """Get results from both methods"""
        results = {}
        
        # Graph retrieval
        graph_sources, graph_scores, graph_meta = self.graph_retriever.retrieve(query, top_k=top_k)
        results["graph"] = {
            "sources": graph_sources[:top_k],
            "scores": graph_scores[:top_k],
            "metadata": graph_meta
        }
        
        # Vector retrieval
        vector_sources, vector_scores, vector_meta = self.vector_retriever.retrieve(query, top_k=top_k)
        results["vector"] = {
            "sources": vector_sources[:top_k],
            "scores": vector_scores[:top_k],
            "metadata": vector_meta
        }
        
        return results
    
    def display_results(self, query_info: Dict, results: Dict):
        """Display results for assessment"""
        print("\n" + "="*80)
        print(f"QUERY: {query_info['query']}")
        print(f"Type: {query_info['type']} | Expected strength: {query_info['expected_strength'].upper()}")
        print(f"Description: {query_info['description']}")
        print("="*80 + "\n")
        
        for mode in ["graph", "vector"]:
            if mode not in results:
                continue
            
            metadata = results[mode].get("metadata", {})
            method_info = metadata.get("method", mode).upper()
            
            print(f"\n[{mode.upper()} RETRIEVAL]")
            print("-" * 80)
            
            sources = results[mode]["sources"]
            scores = results[mode]["scores"]
            
            if not sources:
                print("  No results found")
                continue
            
            for i, source in enumerate(sources, 1):
                score = scores[i-1] if i-1 < len(scores) else "N/A"
                
                # Truncate long text
                display_text = source[:200] + "..." if len(source) > 200 else source
                
                if isinstance(score, float):
                    print(f"\n  [{i}] Relevance Score: {score:.3f}")
                else:
                    print(f"\n  [{i}]")
                
                print(f"  {display_text}")
            
            # Display metadata
            if mode == "graph" and "query_entities" in metadata:
                print(f"\n  Metadata: {metadata['query_entities']} query entities, "
                      f"{metadata.get('matched_entities', 0)} matched")
            elif mode == "vector" and "query_terms" in metadata:
                print(f"\n  Metadata: {metadata['query_terms']} query terms")
    
    def collect_ratings(self, query_info: Dict, results: Dict) -> Dict:
        """Collect human ratings for both methods"""
        assessment = {
            "query": query_info["query"],
            "query_type": query_info["type"],
            "expected_strength": query_info["expected_strength"],
            "timestamp": datetime.now().isoformat(),
            "ratings": {}
        }
        
        print("\n" + "-"*80)
        print("RATING INSTRUCTIONS")
        print("-"*80)
        print("Rate the relevance of retrieved documents (1-5):")
        print("  1 = Completely Irrelevant")
        print("  2 = Poorly Relevant")
        print("  3 = Moderately Relevant")
        print("  4 = Highly Relevant")
        print("  5 = Perfect Match")
        print("-"*80 + "\n")
        
        for mode in ["graph", "vector"]:
            if mode not in results or not results[mode]["sources"]:
                continue
            
            while True:
                try:
                    rating_input = input(f"Rate {mode.upper()} retrieval (1-5): ").strip()
                    rating = int(rating_input)
                    if 1 <= rating <= 5:
                        assessment["ratings"][mode] = rating
                        break
                    else:
                        print("Error: Please enter a number between 1 and 5")
                except ValueError:
                    print("Error: Invalid input. Please enter a number between 1 and 5")
                except KeyboardInterrupt:
                    print("\n\nAssessment interrupted by user")
                    sys.exit(0)
        
        return assessment
    
    def run_assessment(self):
        """Run systematic assessment on predefined queries"""
        print("\n" + "="*80)
        print("SYSTEMATIC QUALITY ASSESSMENT".center(80))
        print("="*80 + "\n")
        
        print(f"Total queries to evaluate: {len(TEST_QUERIES)}")
        print(f"Vector method: {self.vector_method.upper()}\n")
        
        for idx, query_info in enumerate(TEST_QUERIES, 1):
            print(f"\n{'='*80}")
            print(f"QUERY {idx}/{len(TEST_QUERIES)}".center(80))
            print(f"{'='*80}")
            
            # Get results
            results = self.get_results(query_info["query"], top_k=3)
            
            # Display results
            self.display_results(query_info, results)
            
            # Collect ratings
            assessment = self.collect_ratings(query_info, results)
            self.assessments.append(assessment)
            
            print(f"\nAssessment {idx} completed")
        
        # Generate and display summary
        self.display_summary()
        self.save_results()
    
    def display_summary(self):
        """Display comprehensive summary with visualization"""
        if not self.assessments:
            print("\nNo assessments to display")
            return
        
        print("\n\n" + "="*80)
        print("ASSESSMENT SUMMARY".center(80))
        print("="*80 + "\n")
        
        # Overall statistics
        graph_ratings = []
        vector_ratings = []
        
        for assessment in self.assessments:
            if "graph" in assessment["ratings"]:
                graph_ratings.append(assessment["ratings"]["graph"])
            if "vector" in assessment["ratings"]:
                vector_ratings.append(assessment["ratings"]["vector"])
        
        print("OVERALL PERFORMANCE")
        print("-" * 80)
        print(f"{'Method':<20} {'Avg Rating':<15} {'Count':<10} {'Min':<10} {'Max':<10}")
        print("-" * 80)
        
        if graph_ratings:
            avg_graph = sum(graph_ratings) / len(graph_ratings)
            print(f"{'GRAPH RAG':<20} {avg_graph:<15.2f} {len(graph_ratings):<10} "
                  f"{min(graph_ratings):<10} {max(graph_ratings):<10}")
        
        if vector_ratings:
            avg_vector = sum(vector_ratings) / len(vector_ratings)
            print(f"{'VECTOR RAG':<20} {avg_vector:<15.2f} {len(vector_ratings):<10} "
                  f"{min(vector_ratings):<10} {max(vector_ratings):<10}")
        
        # Performance by query type
        print("\n\nPERFORMANCE BY QUERY TYPE")
        print("-" * 80)
        
        type_stats = {}
        for assessment in self.assessments:
            qtype = assessment["query_type"]
            if qtype not in type_stats:
                type_stats[qtype] = {"graph": [], "vector": []}
            
            for mode in ["graph", "vector"]:
                if mode in assessment["ratings"]:
                    type_stats[qtype][mode].append(assessment["ratings"][mode])
        
        for qtype, stats in sorted(type_stats.items()):
            print(f"\n{qtype.upper()}")
            for mode in ["graph", "vector"]:
                if stats[mode]:
                    avg = sum(stats[mode]) / len(stats[mode])
                    print(f"  {mode.capitalize():<10}: {avg:.2f} (n={len(stats[mode])})")
        
        # Expected vs Actual performance
        print("\n\nEXPECTED VS ACTUAL PERFORMANCE")
        print("-" * 80)
        
        correct_predictions = 0
        total_predictions = 0
        
        for assessment in self.assessments:
            expected = assessment["expected_strength"]
            ratings = assessment["ratings"]
            
            if "graph" in ratings and "vector" in ratings:
                total_predictions += 1
                actual_winner = "graph" if ratings["graph"] > ratings["vector"] else "vector"
                if actual_winner == expected:
                    correct_predictions += 1
                
                status = "CORRECT" if actual_winner == expected else "INCORRECT"
                print(f"\nQuery: {assessment['query'][:60]}...")
                print(f"  Expected: {expected.upper()}, Actual: {actual_winner.upper()} [{status}]")
                print(f"  Ratings - Graph: {ratings['graph']}, Vector: {ratings['vector']}")
        
        if total_predictions > 0:
            accuracy = (correct_predictions / total_predictions) * 100
            print(f"\n\nPrediction Accuracy: {correct_predictions}/{total_predictions} ({accuracy:.1f}%)")
        
        # Visualization
        self.display_visualization()
    
    def display_visualization(self):
        """Display ASCII visualization of results"""
        print("\n\n" + "="*80)
        print("HUMAN FEEDBACK VISUALIZATION".center(80))
        print("="*80 + "\n")
        
        # Collect ratings by method
        graph_ratings = []
        vector_ratings = []
        
        for assessment in self.assessments:
            if "graph" in assessment["ratings"]:
                graph_ratings.append(assessment["ratings"]["graph"])
            if "vector" in assessment["ratings"]:
                vector_ratings.append(assessment["ratings"]["vector"])
        
        # Rating distribution
        print("RATING DISTRIBUTION")
        print("-" * 80)
        
        for method_name, ratings in [("GRAPH RAG", graph_ratings), ("VECTOR RAG", vector_ratings)]:
            if not ratings:
                continue
            
            print(f"\n{method_name}")
            for rating in range(5, 0, -1):
                count = ratings.count(rating)
                bar = "#" * count
                percentage = (count / len(ratings)) * 100 if ratings else 0
                print(f"  {rating} | {bar:<20} {count} ({percentage:.1f}%)")
        
        # Comparative bar chart
        if graph_ratings and vector_ratings:
            print("\n\nCOMPARATIVE PERFORMANCE")
            print("-" * 80)
            
            avg_graph = sum(graph_ratings) / len(graph_ratings)
            avg_vector = sum(vector_ratings) / len(vector_ratings)
            
            max_rating = 5.0
            graph_bar_length = int((avg_graph / max_rating) * 40)
            vector_bar_length = int((avg_vector / max_rating) * 40)
            
            print(f"\nGraph RAG   | {'='*graph_bar_length} {avg_graph:.2f}")
            print(f"Vector RAG  | {'='*vector_bar_length} {avg_vector:.2f}")
            print(f"\nScale: 0 {'.'*38} 5.0")
        
        # Winner summary
        print("\n\nWINNER SUMMARY")
        print("-" * 80)
        
        graph_wins = 0
        vector_wins = 0
        ties = 0
        
        for assessment in self.assessments:
            if "graph" in assessment["ratings"] and "vector" in assessment["ratings"]:
                g_rating = assessment["ratings"]["graph"]
                v_rating = assessment["ratings"]["vector"]
                
                if g_rating > v_rating:
                    graph_wins += 1
                elif v_rating > g_rating:
                    vector_wins += 1
                else:
                    ties += 1
        
        total = graph_wins + vector_wins + ties
        if total > 0:
            print(f"\nGraph RAG wins:  {graph_wins}/{total} ({(graph_wins/total)*100:.1f}%)")
            print(f"Vector RAG wins: {vector_wins}/{total} ({(vector_wins/total)*100:.1f}%)")
            print(f"Ties:            {ties}/{total} ({(ties/total)*100:.1f}%)")
        
        print("\n" + "="*80 + "\n")
    
    def save_results(self):
        """Save assessment results to JSON"""
        output_path = Path(__file__).parent / "quality_assessment_results.json"
        
        results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "vector_method": self.vector_method,
                "num_queries": len(TEST_QUERIES),
                "num_assessments": len(self.assessments)
            },
            "test_queries": TEST_QUERIES,
            "assessments": self.assessments
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Quality Assessment for RAG Systems")
    parser.add_argument(
        "--vector-method",
        type=str,
        default="tfidf",
        choices=["tfidf", "bm25"],
        help="Vector retrieval method (default: tfidf)"
    )
    args = parser.parse_args()
    
    try:
        assessor = QualityAssessor(vector_method=args.vector_method)
        assessor.run_assessment()
    except KeyboardInterrupt:
        print("\n\nAssessment interrupted by user")
        sys.exit(0)

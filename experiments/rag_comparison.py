import time
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.graph_rag import GraphRAGRetriever
from app.vector_store import VectorStore, OpenAIEmbedding
from app.document_processor import load_default_knowledge_base
from app.backend import LLMComponent

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============ TEST QUERIES ============
TEST_QUERIES = [
    "Что такое RAG и как оно работает?",
    "Какие модели использует OpenAI?",
    "Расскажи про трансформеры и attention механизм",
    "Как граф знаний улучшает поиск?",
    "Что такое embeddings и зачем они нужны?",
    "Разница между GPT-3 и GPT-4",
    "Как работает prompt engineering?",
    "Что такое fine-tuning модели?",
]


class RAGExperiment:
    def __init__(self):
        """Инициализация эксперимента"""
        logger.info("Инициализация эксперимента...")
        
        # Загрузить документы
        self.documents = load_default_knowledge_base()
        logger.info(f"Загружено {len(self.documents)} документов")
        
        # Инициализировать Graph RAG
        self.graph_retriever = GraphRAGRetriever()
        self.graph_retriever.build_graph(self.documents)
        logger.info("Граф построен")
        
        # Инициализировать Vector Store (если возможно)
        self.vector_store = None
        try:
            if os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_API_KEY").startswith("sk-"):
                embedding_model = OpenAIEmbedding()
                self.vector_store = VectorStore(embedding_model=embedding_model)
                self.vector_store.add_documents(self.documents)
                logger.info("Vector store инициализирован")
        except Exception as e:
            logger.warning(f"Vector store не инициализирован: {e}")
        
        # Инициализировать LLM
        self.llm = LLMComponent()
    
    def graph_retrieval(self, query: str, top_k: int = 5) -> Tuple[List[str], float]:
        """Graph-based retrieval"""
        start_time = time.time()
        
        sources, scores, metadata = self.graph_retriever.hybrid_retrieve(
            query, top_k=top_k, vector_weight=0.0, graph_weight=1.0
        )
        
        latency = time.time() - start_time
        return sources, latency
    
    def vector_retrieval(self, query: str, top_k: int = 5) -> Tuple[List[str], float]:
        """Vector-based retrieval"""
        if not self.vector_store:
            return [], 0
        
        start_time = time.time()
        sources, scores, metadata = self.vector_store.similarity_search(query, top_k)
        latency = time.time() - start_time
        
        return sources, latency
    
    def hybrid_retrieval(self, query: str, top_k: int = 5) -> Tuple[List[str], float]:
        """Hybrid retrieval (граф + вектора)"""
        start_time = time.time()
        
        sources, scores, metadata = self.graph_retriever.hybrid_retrieve(
            query, top_k=top_k, vector_weight=0.5, graph_weight=0.5
        )
        
        latency = time.time() - start_time
        return sources, latency
    
    def run_experiment(self, queries: List[str] = None) -> Dict:
        """Запустить полный эксперимент"""
        if queries is None:
            queries = TEST_QUERIES
        
        results = {
            "graph": {"queries": [], "avg_latency": 0, "total_latency": 0},
            "vector": {"queries": [], "avg_latency": 0, "total_latency": 0},
            "hybrid": {"queries": [], "avg_latency": 0, "total_latency": 0},
        }
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Запуск эксперимента на {len(queries)} вопросах")
        logger.info(f"{'='*60}\n")
        
        for idx, query in enumerate(queries, 1):
            logger.info(f"[{idx}/{len(queries)}] Запрос: {query[:50]}...")
            
            # Graph retrieval
            graph_sources, graph_latency = self.graph_retrieval(query)
            results["graph"]["queries"].append({
                "query": query,
                "sources_count": len(graph_sources),
                "latency": graph_latency,
                "sources": graph_sources[:2]  # Top 2 для display
            })
            results["graph"]["total_latency"] += graph_latency
            
            # Vector retrieval
            if self.vector_store:
                vector_sources, vector_latency = self.vector_retrieval(query)
                results["vector"]["queries"].append({
                    "query": query,
                    "sources_count": len(vector_sources),
                    "latency": vector_latency,
                    "sources": vector_sources[:2]
                })
                results["vector"]["total_latency"] += vector_latency
            
            # Hybrid retrieval
            hybrid_sources, hybrid_latency = self.hybrid_retrieval(query)
            results["hybrid"]["queries"].append({
                "query": query,
                "sources_count": len(hybrid_sources),
                "latency": hybrid_latency,
                "sources": hybrid_sources[:2]
            })
            results["hybrid"]["total_latency"] += hybrid_latency
        
        # Вычислить средние значения
        for mode in results:
            if results[mode]["queries"]:
                results[mode]["avg_latency"] = results[mode]["total_latency"] / len(results[mode]["queries"]) * 1000  # ms
        
        return results
    
    def print_results(self, results: Dict):
        """Вывести результаты эксперимента"""
        print(f"\n{'='*80}")
        print("РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТА".center(80))
        print(f"{'='*80}\n")
        
        print(f"{'Метод':<20} {'Avg Latency (ms)':<20} {'Queries':<15} {'Avg Sources':<15}")
        print(f"{'-'*80}")
        
        for mode in ["graph", "vector", "hybrid"]:
            if not results[mode]["queries"]:
                continue
            
            avg_latency = results[mode]["avg_latency"]
            queries_count = len(results[mode]["queries"])
            avg_sources = sum(q["sources_count"] for q in results[mode]["queries"]) / queries_count
            
            print(f"{mode.upper():<20} {avg_latency:<20.2f} {queries_count:<15} {avg_sources:<15.1f}")
        
        print(f"\n{'='*80}")
        print("\nДЕТАЛНЫЕ РЕЗУЛЬТАТЫ ПО ЗАПРОСАМ:\n")
        
        for mode in ["graph", "vector", "hybrid"]:
            if not results[mode]["queries"]:
                logger.info(f"❌ {mode.upper()} - не доступен")
                continue
            
            logger.info(f"\n📊 {mode.upper()} RETRIEVAL:")
            logger.info(f"{'-'*80}")
            
            for i, query_result in enumerate(results[mode]["queries"], 1):
                logger.info(f"  [{i}] Query: {query_result['query'][:60]}")
                logger.info(f"      Latency: {query_result['latency']*1000:.2f}ms | Sources: {query_result['sources_count']}")
    
    def save_results(self, results: Dict, filename: str = "experiment_results.json"):
        """Сохранить результаты в JSON"""
        output_path = Path(__file__).parent / filename
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n✅ Результаты сохранены в: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG Comparison Experiment")
    parser.add_argument("--queries", type=int, default=len(TEST_QUERIES), 
                       help="Количество тестовых запросов")
    parser.add_argument("--custom-query", type=str, 
                       help="Custom query for single test")
    args = parser.parse_args()
    
    experiment = RAGExperiment()
    
    if args.custom_query:
        logger.info(f"Тестирование custom query: {args.custom_query}")
        results = experiment.run_experiment([args.custom_query])
    else:
        results = experiment.run_experiment(TEST_QUERIES[:args.queries])
    
    experiment.print_results(results)
    experiment.save_results(results)

#!/usr/bin/env python3
"""
Manual Quality Assessment Script

Позволяет вручную оценить质量 результатов разных методов retrieval:
- Graph-based
- Vector-based
- Hybrid

Оценка по шкале 1-5:
1 = Completely Irrelevant (не релевантно вообще)
2 = Poorly Relevant (слабо релевантно)
3 = Moderately Relevant (умеренно релевантно)
4 = Highly Relevant (хорошо релевантно)
5 = Perfect (идеально)
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.graph_rag import GraphRAGRetriever
from app.vector_store import VectorStore, OpenAIEmbedding
from app.document_processor import load_default_knowledge_base
import logging

logging.basicConfig(level=logging.WARNING)


class QualityAssessor:
    def __init__(self):
        """Инициализация"""
        print("\n📊 Инициализация sistem оценки качества...\n")
        
        # Загрузить документы
        self.documents = load_default_knowledge_base()
        
        # Инициализировать Graph RAG
        self.graph_retriever = GraphRAGRetriever()
        self.graph_retriever.build_graph(self.documents)
        
        # Инициализировать Vector Store
        self.vector_store = None
        try:
            if os.getenv("OPENAI_API_KEY") and not os.getenv("OPENAI_API_KEY").startswith("sk-or"):
                embedding_model = OpenAIEmbedding()
                self.vector_store = VectorStore(embedding_model=embedding_model)
                self.vector_store.add_documents(self.documents)
        except Exception as e:
            print(f"⚠️  Vector store не доступен: {e}")
        
        self.assessments = []
    
    def get_results(self, query: str, top_k: int = 3) -> Dict:
        """Получить результаты из разных методов"""
        results = {}
        
        # Graph retrieval
        graph_sources, graph_scores, _ = self.graph_retriever.hybrid_retrieve(
            query, top_k=top_k, vector_weight=0.0, graph_weight=1.0
        )
        results["graph"] = {
            "sources": graph_sources[:top_k],
            "scores": graph_scores[:top_k]
        }
        
        # Vector retrieval
        if self.vector_store:
            vector_sources, vector_scores, _ = self.vector_store.similarity_search(query, top_k)
            results["vector"] = {
                "sources": vector_sources[:top_k],
                "scores": vector_scores[:top_k]
            }
        
        # Hybrid retrieval
        hybrid_sources, hybrid_scores, _ = self.graph_retriever.hybrid_retrieve(
            query, top_k=top_k, vector_weight=0.5, graph_weight=0.5
        )
        results["hybrid"] = {
            "sources": hybrid_sources[:top_k],
            "scores": hybrid_scores[:top_k]
        }
        
        return results
    
    def display_results(self, query: str, results: Dict):
        """Отобразить результаты для оценки"""
        print(f"\n{'='*80}")
        print(f"💬 QUERY: {query}")
        print(f"{'='*80}\n")
        
        for mode in ["graph", "vector", "hybrid"]:
            if mode not in results:
                continue
            
            print(f"\n🔹 {mode.upper()} RETRIEVAL:")
            print(f"{'-'*80}")
            
            for i, source in enumerate(results[mode]["sources"], 1):
                score = results[mode]["scores"][i-1] if i-1 < len(results[mode]["scores"]) else "N/A"
                
                # Обрезать длинный текст
                display_text = source[:150] + "..." if len(source) > 150 else source
                print(f"\n  [{i}] Score: {score:.3f}" if isinstance(score, float) else f"\n  [{i}]")
                print(f"      {display_text}\n")
    
    def assess_query(self, query: str):
        """Провести оценку одного запроса"""
        results = self.get_results(query)
        self.display_results(query, results)
        
        # Сбор оценок
        assessment = {
            "query": query,
            "ratings": {}
        }
        
        for mode in ["graph", "vector", "hybrid"]:
            if mode not in results:
                continue
            
            while True:
                try:
                    rating = int(input(f"\n📝 Оцените {mode.upper()} (1-5): "))
                    if 1 <= rating <= 5:
                        assessment["ratings"][mode] = rating
                        break
                    else:
                        print("❌ Пожалуйста, введите число от 1 до 5")
                except ValueError:
                    print("❌ Некорректная оценка, попробуйте снова")
        
        self.assessments.append(assessment)
        print(f"\n✅ Оценка сохранена")
    
    def run_interactive(self):
        """Интерактивная оценка"""
        print("\n" + "="*80)
        print("ИНТЕРАКТИВНАЯ ОЦЕНКА КАЧЕСТВА RAG МЕТОДОВ".center(80))
        print("="*80 + "\n")
        
        print("Шкала оценок:")
        print("  1 = Совсем не релевантно")
        print("  2 = Слабо релевантно")
        print("  3 = Умеренно релевантно")
        print("  4 = Хорошо релевантно")
        print("  5 = Идеально релевантно\n")
        
        while True:
            query = input("\n🔍 Введите вопрос (или 'exit' для выхода): ").strip()
            
            if query.lower() == "exit":
                break
            
            if not query:
                print("❌ Вопрос не может быть пустым")
                continue
            
            try:
                self.assess_query(query)
            except Exception as e:
                print(f"❌ Ошибка: {e}")
        
        self.print_summary()
        self.save_assessments()
    
    def print_summary(self):
        """Вывести итоговый отчет"""
        if not self.assessments:
            print("\n⚠️  Нет оценок для вывода")
            return
        
        print("\n" + "="*80)
        print("ИТОГОВАЯ СТАТИСТИКА".center(80))
        print("="*80 + "\n")
        
        modes_scores = {"graph": [], "vector": [], "hybrid": []}
        
        for assessment in self.assessments:
            for mode, rating in assessment["ratings"].items():
                modes_scores[mode].append(rating)
        
        print(f"{'Метод':<15} {'Avg Rating':<15} {'Count':<15}")
        print(f"{'-'*45}")
        
        for mode, scores in modes_scores.items():
            if scores:
                avg_rating = sum(scores) / len(scores)
                print(f"{mode.upper():<15} {avg_rating:<15.2f} {len(scores):<15}")
        
        print("\n" + "="*80 + "\n")
    
    def save_assessments(self):
        """Сохранить оценки в JSON"""
        output_path = Path(__file__).parent / "quality_assessments.json"
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.assessments, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Оценки сохранены в: {output_path}\n")


if __name__ == "__main__":
    assessor = QualityAssessor()
    assessor.run_interactive()

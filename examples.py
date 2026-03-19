"""
Graph RAG System - Usage Examples
Demonstrates various features and capabilities
"""

import requests
import json
from typing import List, Dict

BACKEND_URL = "http://localhost:8000"


def example_basic_query():
    """Example 1: Basic query to Graph RAG system"""
    print("\n" + "="*60)
    print("Example 1: Basic Query")
    print("="*60)
    
    response = requests.post(
        f"{BACKEND_URL}/query",
        json={
            "query": "What is Graph RAG and how does it work?",
            "top_k": 5,
            "temperature": 0.2,
            "model": "openai/gpt-4o-mini",
            "retrieval_mode": "graph_rag",
            "use_entity_context": True
        }
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"\n📝 Answer:\n{data['answer']}\n")
        print(f"📚 Retrieved {len(data['sources'])} sources")
        print(f"🔍 Metadata: {json.dumps(data['metadata'], indent=2)}")
    else:
        print(f"❌ Error: {response.status_code}")


def example_different_retrieval_modes():
    """Example 2: Compare different retrieval modes"""
    print("\n" + "="*60)
    print("Example 2: Comparing Retrieval Modes")
    print("="*60)
    
    query = "How does entity extraction work in NLP?"
    modes = ["graph_rag", "vector"]
    
    for mode in modes:
        print(f"\n🔍 Mode: {mode}")
        print("-" * 40)
        
        response = requests.post(
            f"{BACKEND_URL}/query",
            json={
                "query": query,
                "top_k": 3,
                "retrieval_mode": mode,
                "use_entity_context": True
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"Answer length: {len(data['answer'])} chars")
            print(f"Sources: {len(data['sources'])}")
            print(f"Avg score: {sum(data['similarity_scores']) / len(data['similarity_scores']):.3f}")
        else:
            print(f"❌ Error: {response.status_code}")


def example_system_stats():
    """Example 3: Get system statistics"""
    print("\n" + "="*60)
    print("Example 3: System Statistics")
    print("="*60)
    
    response = requests.get(f"{BACKEND_URL}/stats")
    
    if response.status_code == 200:
        stats = response.json()
        print(f"\n📊 System Status: {stats['status']}")
        print(f"📄 Total Documents: {stats['num_documents']}")
        
        if stats.get('graph_stats'):
            gs = stats['graph_stats']
            print(f"\n🕸️ Knowledge Graph:")
            print(f"   • Entities: {gs['num_entities']}")
            print(f"   • Relations: {gs['num_relations']}")
            print(f"   • Graph Nodes: {gs['num_nodes']}")
            print(f"   • Graph Edges: {gs['num_edges']}")
            print(f"   • Avg Degree: {gs['avg_degree']:.2f}")
        
        if stats.get('vector_stats'):
            vs = stats['vector_stats']
            print(f"\n📦 Vector Store:")
            print(f"   • Documents: {vs['num_documents']}")
            print(f"   • Embedding Dim: {vs['embedding_dimension']}")
            print(f"   • Size: {vs['total_size_mb']:.2f} MB")
    else:
        print(f"❌ Error: {response.status_code}")


def example_get_entities():
    """Example 4: Get top entities from knowledge graph"""
    print("\n" + "="*60)
    print("Example 4: Top Entities in Knowledge Graph")
    print("="*60)
    
    response = requests.get(f"{BACKEND_URL}/graph/entities?limit=10")
    
    if response.status_code == 200:
        data = response.json()
        entities = data['entities']
        
        print(f"\n🏷️ Top {len(entities)} Entities:\n")
        for i, entity in enumerate(entities, 1):
            print(f"{i:2d}. {entity['text']:30s} ({entity['type']:10s}) "
                  f"- freq: {entity['frequency']:3d}, docs: {entity['num_documents']}")
    else:
        print(f"❌ Error: {response.status_code}")


def example_get_relations():
    """Example 5: Get top relations from knowledge graph"""
    print("\n" + "="*60)
    print("Example 5: Top Relations in Knowledge Graph")
    print("="*60)
    
    response = requests.get(f"{BACKEND_URL}/graph/relations?limit=10")
    
    if response.status_code == 200:
        data = response.json()
        relations = data['relations']
        
        print(f"\n🔗 Top {len(relations)} Relations:\n")
        for i, rel in enumerate(relations, 1):
            print(f"{i:2d}. {rel['source']:20s} --[{rel['type']:15s}]--> "
                  f"{rel['target']:20s} (weight: {rel['weight']:.1f})")
    else:
        print(f"❌ Error: {response.status_code}")


def example_temperature_comparison():
    """Example 6: Compare different temperature settings"""
    print("\n" + "="*60)
    print("Example 6: Temperature Comparison")
    print("="*60)
    
    query = "Explain retrieval augmented generation"
    temperatures = [0.0, 0.5, 1.0]
    
    for temp in temperatures:
        print(f"\n🌡️ Temperature: {temp}")
        print("-" * 40)
        
        response = requests.post(
            f"{BACKEND_URL}/query",
            json={
                "query": query,
                "top_k": 3,
                "temperature": temp,
                "retrieval_mode": "graph_rag"
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            answer = data['answer']
            print(f"Answer preview: {answer[:150]}...")
        else:
            print(f"❌ Error: {response.status_code}")


def example_entity_context():
    """Example 7: Compare with and without entity context"""
    print("\n" + "="*60)
    print("Example 7: Entity Context Impact")
    print("="*60)
    
    query = "What are the components of a RAG system?"
    
    for use_context in [False, True]:
        print(f"\n🔗 Entity Context: {'Enabled' if use_context else 'Disabled'}")
        print("-" * 40)
        
        response = requests.post(
            f"{BACKEND_URL}/query",
            json={
                "query": query,
                "top_k": 5,
                "retrieval_mode": "graph_rag",
                "use_entity_context": use_context
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"Answer length: {len(data['answer'])} chars")
            print(f"Answer preview: {data['answer'][:200]}...")
        else:
            print(f"❌ Error: {response.status_code}")


def main():
    """Run all examples"""
    print("\n" + "="*60)
    print("🔮 Graph RAG System - Usage Examples")
    print("="*60)
    print("\nMake sure the backend is running at http://localhost:8000")
    print("Start it with: uv run backend")
    
    try:
        # Check if backend is running
        response = requests.get(f"{BACKEND_URL}/health", timeout=2)
        if response.status_code != 200:
            print("\n❌ Backend is not responding correctly")
            return
    except requests.exceptions.ConnectionError:
        print("\n❌ Cannot connect to backend. Please start it first:")
        print("   uv run backend")
        return
    
    print("\n✅ Backend is running!\n")
    
    # Run examples
    examples = [
        example_basic_query,
        example_system_stats,
        example_get_entities,
        example_get_relations,
        example_different_retrieval_modes,
        example_temperature_comparison,
        example_entity_context,
    ]
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"\n❌ Error in {example.__name__}: {e}")
    
    print("\n" + "="*60)
    print("✅ All examples completed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

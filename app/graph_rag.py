"""
Graph RAG Implementation
Combines knowledge graph construction with vector-based retrieval
Based on best practices from RAG and Generative AI literature
"""

import logging
from typing import List, Dict, Tuple, Optional, Set
import numpy as np
from collections import defaultdict
import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

logger = logging.getLogger(__name__)


class Entity:
    """Represents an entity in the knowledge graph"""
    
    def __init__(self, text: str, entity_type: str, doc_id: int):
        self.text = text.lower().strip()
        self.entity_type = entity_type
        self.doc_ids = {doc_id}
        self.frequency = 1
    
    def add_occurrence(self, doc_id: int):
        """Add another occurrence of this entity"""
        self.doc_ids.add(doc_id)
        self.frequency += 1
    
    def __hash__(self):
        return hash(self.text)
    
    def __eq__(self, other):
        return isinstance(other, Entity) and self.text == other.text
    
    def __repr__(self):
        return f"Entity({self.text}, {self.entity_type}, freq={self.frequency})"


class Relation:
    """Represents a relation between entities"""
    
    def __init__(self, source: Entity, target: Entity, relation_type: str, doc_id: int):
        self.source = source
        self.target = target
        self.relation_type = relation_type
        self.doc_ids = {doc_id}
        self.weight = 1.0
    
    def strengthen(self, doc_id: int):
        """Strengthen the relation by adding another occurrence"""
        self.doc_ids.add(doc_id)
        self.weight += 0.5
    
    def __repr__(self):
        return f"Relation({self.source.text} --[{self.relation_type}]--> {self.target.text})"


class KnowledgeGraph:
    """Knowledge graph for storing entities and relations"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.entities: Dict[str, Entity] = {}
        self.relations: List[Relation] = []
        self.doc_entities: Dict[int, Set[str]] = defaultdict(set)
    
    def add_entity(self, entity: Entity, doc_id: int) -> Entity:
        """Add or update an entity in the graph"""
        if entity.text in self.entities:
            existing = self.entities[entity.text]
            existing.add_occurrence(doc_id)
            self.doc_entities[doc_id].add(entity.text)
            return existing
        else:
            self.entities[entity.text] = entity
            self.graph.add_node(entity.text, entity_type=entity.entity_type, frequency=entity.frequency)
            self.doc_entities[doc_id].add(entity.text)
            return entity
    
    def add_relation(self, relation: Relation, doc_id: int):
        """Add or update a relation in the graph"""
        # Check if relation already exists
        for existing_rel in self.relations:
            if (existing_rel.source.text == relation.source.text and 
                existing_rel.target.text == relation.target.text and
                existing_rel.relation_type == relation.relation_type):
                existing_rel.strengthen(doc_id)
                self.graph[relation.source.text][relation.target.text]['weight'] = existing_rel.weight
                return
        
        # Add new relation
        self.relations.append(relation)
        if not self.graph.has_edge(relation.source.text, relation.target.text):
            self.graph.add_edge(
                relation.source.text,
                relation.target.text,
                relation_type=relation.relation_type,
                weight=relation.weight
            )
    
    def get_entity_neighbors(self, entity_text: str, max_depth: int = 2) -> Set[str]:
        """Get neighboring entities up to max_depth hops"""
        if entity_text not in self.graph:
            return set()
        
        neighbors = set()
        current_level = {entity_text}
        
        for _ in range(max_depth):
            next_level = set()
            for node in current_level:
                # Get successors and predecessors
                next_level.update(self.graph.successors(node))
                next_level.update(self.graph.predecessors(node))
            neighbors.update(next_level)
            current_level = next_level - neighbors
            if not current_level:
                break
        
        return neighbors
    
    def get_subgraph_for_entities(self, entity_texts: List[str], max_depth: int = 2) -> nx.DiGraph:
        """Extract a subgraph containing specified entities and their neighbors"""
        all_nodes = set(entity_texts)
        for entity_text in entity_texts:
            all_nodes.update(self.get_entity_neighbors(entity_text, max_depth))
        
        return self.graph.subgraph(all_nodes)
    
    def get_documents_for_entity(self, entity_text: str) -> Set[int]:
        """Get all document IDs containing this entity"""
        if entity_text in self.entities:
            return self.entities[entity_text].doc_ids
        return set()
    
    def get_entities_for_document(self, doc_id: int) -> Set[str]:
        """Get all entities in a document"""
        return self.doc_entities.get(doc_id, set())
    
    def get_graph_stats(self) -> Dict:
        """Get statistics about the knowledge graph"""
        return {
            "num_entities": len(self.entities),
            "num_relations": len(self.relations),
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "avg_degree": sum(dict(self.graph.degree()).values()) / max(self.graph.number_of_nodes(), 1)
        }


class GraphExtractor:
    """Extract entities and relations from text using NLP"""
    
    def __init__(self, nlp_model=None):
        self.nlp = nlp_model or spacy.load("en_core_web_sm")
        
        # Entity types to extract
        self.entity_types = {
            'PERSON', 'ORG', 'GPE', 'PRODUCT', 'EVENT', 
            'WORK_OF_ART', 'LAW', 'LANGUAGE', 'NORP'
        }
        
        # Dependency patterns for relation extraction
        self.relation_patterns = [
            ('nsubj', 'dobj'),      # subject-verb-object
            ('nsubj', 'attr'),      # subject-verb-attribute
            ('nsubj', 'prep'),      # subject-verb-preposition
            ('compound', 'pobj'),   # compound-preposition-object
        ]
    
    def extract_entities(self, text: str, doc_id: int) -> List[Entity]:
        """Extract named entities from text"""
        doc = self.nlp(text)
        entities = []
        
        # Extract named entities
        for ent in doc.ents:
            if ent.label_ in self.entity_types:
                entity = Entity(ent.text, ent.label_, doc_id)
                entities.append(entity)
        
        # Extract noun chunks as potential entities
        for chunk in doc.noun_chunks:
            # Filter out very short or very long chunks
            if 2 <= len(chunk.text.split()) <= 5:
                entity = Entity(chunk.text, 'CONCEPT', doc_id)
                entities.append(entity)
        
        return entities
    
    def extract_relations(self, text: str, doc_id: int, entities: List[Entity]) -> List[Relation]:
        """Extract relations between entities"""
        doc = self.nlp(text)
        relations = []
        entity_map = {e.text.lower(): e for e in entities}
        
        # Extract relations based on dependency parsing
        for token in doc:
            if token.pos_ == 'VERB':
                # Find subject and object
                subjects = [child for child in token.children if child.dep_ in ('nsubj', 'nsubjpass')]
                objects = [child for child in token.children if child.dep_ in ('dobj', 'attr', 'pobj')]
                
                for subj in subjects:
                    for obj in objects:
                        subj_text = subj.text.lower()
                        obj_text = obj.text.lower()
                        
                        # Check if both are entities
                        if subj_text in entity_map and obj_text in entity_map:
                            relation = Relation(
                                entity_map[subj_text],
                                entity_map[obj_text],
                                token.lemma_,
                                doc_id
                            )
                            relations.append(relation)
        
        # Extract co-occurrence relations (entities in same sentence)
        for sent in doc.sents:
            sent_entities = []
            for ent in sent.ents:
                if ent.text.lower() in entity_map:
                    sent_entities.append(entity_map[ent.text.lower()])
            
            # Create co-occurrence relations
            for i, e1 in enumerate(sent_entities):
                for e2 in sent_entities[i+1:]:
                    relation = Relation(e1, e2, 'co-occurs', doc_id)
                    relations.append(relation)
        
        return relations


class GraphRAGRetriever:
    """
    Hybrid retrieval system combining:
    1. Vector-based semantic search
    2. Graph-based entity and relation traversal
    """
    
    def __init__(self):
        self.knowledge_graph = KnowledgeGraph()
        self.graph_extractor = GraphExtractor()
        self.documents: List[str] = []
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            use_idf=True,
            norm='l2',
            ngram_range=(1, 3),
            max_features=5000,
            sublinear_tf=True
        )
        self.tfidf_matrix = None
        self.entity_embeddings = None
    
    def build_graph(self, documents: List[str]):
        """Build knowledge graph from documents"""
        logger.info(f"Building knowledge graph from {len(documents)} documents...")
        self.documents = documents
        
        # Extract entities and relations from each document
        for doc_id, doc_text in enumerate(documents):
            # Extract entities
            entities = self.graph_extractor.extract_entities(doc_text, doc_id)
            
            # Add entities to graph
            added_entities = []
            for entity in entities:
                added_entity = self.knowledge_graph.add_entity(entity, doc_id)
                added_entities.append(added_entity)
            
            # Extract and add relations
            relations = self.graph_extractor.extract_relations(doc_text, doc_id, added_entities)
            for relation in relations:
                self.knowledge_graph.add_relation(relation, doc_id)
        
        # Build TF-IDF matrix for vector search
        self.tfidf_matrix = self.vectorizer.fit_transform(documents)
        
        stats = self.knowledge_graph.get_graph_stats()
        logger.info(f"Knowledge graph built: {stats}")
    
    def _vector_retrieve(self, query: str, top_k: int) -> Tuple[List[int], List[float]]:
        """Retrieve documents using vector similarity"""
        query_tfidf = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_tfidf, self.tfidf_matrix)[0]
        
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        top_scores = similarities[top_indices]
        
        return top_indices.tolist(), top_scores.tolist()
    
    def _graph_retrieve(self, query: str, top_k: int) -> Tuple[List[int], List[float]]:
        """Retrieve documents using graph traversal"""
        # Extract entities from query
        query_entities = self.graph_extractor.extract_entities(query, -1)
        
        if not query_entities:
            return [], []
        
        # Find matching entities in graph
        matched_entities = []
        for q_entity in query_entities:
            if q_entity.text in self.knowledge_graph.entities:
                matched_entities.append(q_entity.text)
        
        if not matched_entities:
            return [], []
        
        # Get documents containing these entities and their neighbors
        doc_scores = defaultdict(float)
        
        for entity_text in matched_entities:
            # Direct match - high score
            direct_docs = self.knowledge_graph.get_documents_for_entity(entity_text)
            for doc_id in direct_docs:
                doc_scores[doc_id] += 1.0
            
            # Neighbor entities - medium score
            neighbors = self.knowledge_graph.get_entity_neighbors(entity_text, max_depth=1)
            for neighbor in neighbors:
                neighbor_docs = self.knowledge_graph.get_documents_for_entity(neighbor)
                for doc_id in neighbor_docs:
                    doc_scores[doc_id] += 0.5
        
        # Sort by score
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        if not sorted_docs:
            return [], []
        
        doc_ids = [doc_id for doc_id, _ in sorted_docs]
        scores = [score for _, score in sorted_docs]
        
        # Normalize scores
        max_score = max(scores) if scores else 1.0
        scores = [s / max_score for s in scores]
        
        return doc_ids, scores
    
    def hybrid_retrieve(
        self, 
        query: str, 
        top_k: int = 5,
        vector_weight: float = 0.6,
        graph_weight: float = 0.4
    ) -> Tuple[List[str], List[float], Dict]:
        """
        Hybrid retrieval combining vector and graph methods
        
        Args:
            query: Search query
            top_k: Number of documents to retrieve
            vector_weight: Weight for vector similarity (0-1)
            graph_weight: Weight for graph similarity (0-1)
        
        Returns:
            Tuple of (documents, scores, metadata)
        """
        # Normalize weights
        total_weight = vector_weight + graph_weight
        vector_weight /= total_weight
        graph_weight /= total_weight
        
        # Vector retrieval
        vector_doc_ids, vector_scores = self._vector_retrieve(query, top_k * 2)
        
        # Graph retrieval
        graph_doc_ids, graph_scores = self._graph_retrieve(query, top_k * 2)
        
        # Combine scores
        combined_scores = defaultdict(float)
        
        for doc_id, score in zip(vector_doc_ids, vector_scores):
            combined_scores[doc_id] += vector_weight * score
        
        for doc_id, score in zip(graph_doc_ids, graph_scores):
            combined_scores[doc_id] += graph_weight * score
        
        # Sort by combined score
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        if not sorted_results:
            # Fallback to vector only
            return (
                [self.documents[i] for i in vector_doc_ids[:top_k]],
                vector_scores[:top_k],
                {"method": "vector_only", "graph_entities": 0}
            )
        
        final_doc_ids = [doc_id for doc_id, _ in sorted_results]
        final_scores = [score for _, score in sorted_results]
        
        # Gather metadata
        query_entities = self.graph_extractor.extract_entities(query, -1)
        metadata = {
            "method": "hybrid",
            "vector_weight": vector_weight,
            "graph_weight": graph_weight,
            "query_entities": len(query_entities),
            "graph_stats": self.knowledge_graph.get_graph_stats()
        }
        
        return (
            [self.documents[i] for i in final_doc_ids],
            final_scores,
            metadata
        )
    
    def get_entity_context(self, query: str, max_entities: int = 10) -> str:
        """Get entity context from knowledge graph for query"""
        query_entities = self.graph_extractor.extract_entities(query, -1)
        
        if not query_entities:
            return ""
        
        context_parts = []
        entity_count = 0
        
        for q_entity in query_entities:
            if entity_count >= max_entities:
                break
                
            if q_entity.text in self.knowledge_graph.entities:
                entity = self.knowledge_graph.entities[q_entity.text]
                neighbors = self.knowledge_graph.get_entity_neighbors(q_entity.text, max_depth=1)
                
                if neighbors:
                    context_parts.append(
                        f"Entity '{entity.text}' ({entity.entity_type}) is related to: {', '.join(list(neighbors)[:5])}"
                    )
                    entity_count += 1
        
        return "\n".join(context_parts) if context_parts else ""

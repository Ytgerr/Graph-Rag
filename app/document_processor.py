"""
Document Processing Utilities
Handles document loading, chunking, and preprocessing
"""

import logging
from typing import List, Dict, Optional, Tuple
import re
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class DocumentChunker:
    """
    Smart document chunking with overlap
    Based on best practices from RAG literature
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        separator: str = "\n\n"
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator
    
    def chunk_text(self, text: str, metadata: Optional[Dict] = None) -> List[Dict]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Text to chunk
            metadata: Optional metadata to attach to each chunk
        
        Returns:
            List of chunk dictionaries with text and metadata
        """
        # Split by separator first
        sections = text.split(self.separator)
        
        chunks = []
        current_chunk = ""
        chunk_id = 0
        
        for section in sections:
            section = section.strip()
            if not section:
                continue
            
            # If section is too long, split it further
            if len(section) > self.chunk_size:
                # Split by sentences
                sentences = self._split_sentences(section)
                
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) <= self.chunk_size:
                        current_chunk += sentence + " "
                    else:
                        if current_chunk:
                            chunks.append(self._create_chunk(current_chunk.strip(), chunk_id, metadata))
                            chunk_id += 1
                            
                            # Add overlap
                            overlap_text = self._get_overlap(current_chunk)
                            current_chunk = overlap_text + sentence + " "
                        else:
                            # Sentence is too long, force split
                            chunks.append(self._create_chunk(sentence[:self.chunk_size], chunk_id, metadata))
                            chunk_id += 1
                            current_chunk = sentence[self.chunk_size - self.chunk_overlap:] + " "
            else:
                # Section fits in chunk size
                if len(current_chunk) + len(section) <= self.chunk_size:
                    current_chunk += section + " "
                else:
                    if current_chunk:
                        chunks.append(self._create_chunk(current_chunk.strip(), chunk_id, metadata))
                        chunk_id += 1
                        
                        # Add overlap
                        overlap_text = self._get_overlap(current_chunk)
                        current_chunk = overlap_text + section + " "
                    else:
                        current_chunk = section + " "
        
        # Add remaining chunk
        if current_chunk.strip():
            chunks.append(self._create_chunk(current_chunk.strip(), chunk_id, metadata))
        
        logger.info(f"Created {len(chunks)} chunks from text")
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _get_overlap(self, text: str) -> str:
        """Get overlap text from end of chunk"""
        if len(text) <= self.chunk_overlap:
            return text
        return text[-self.chunk_overlap:]
    
    def _create_chunk(self, text: str, chunk_id: int, metadata: Optional[Dict]) -> Dict:
        """Create chunk dictionary"""
        chunk = {
            "text": text,
            "chunk_id": chunk_id,
            "length": len(text)
        }
        
        if metadata:
            chunk["metadata"] = metadata
        
        return chunk


class DocumentLoader:
    """Load documents from various sources"""
    
    @staticmethod
    def load_text_file(file_path: str) -> str:
        """Load text from file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to load file {file_path}: {e}")
            return ""
    
    @staticmethod
    def load_json_file(file_path: str) -> List[Dict]:
        """Load documents from JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict):
                    return [data]
                else:
                    return []
        except Exception as e:
            logger.error(f"Failed to load JSON file {file_path}: {e}")
            return []
    
    @staticmethod
    def load_directory(
        directory: str,
        file_pattern: str = "*.txt",
        recursive: bool = False
    ) -> List[Dict]:
        """
        Load all documents from directory
        
        Args:
            directory: Directory path
            file_pattern: File pattern to match
            recursive: Whether to search recursively
        
        Returns:
            List of document dictionaries
        """
        dir_path = Path(directory)
        
        if not dir_path.exists():
            logger.error(f"Directory not found: {directory}")
            return []
        
        documents = []
        
        if recursive:
            files = dir_path.rglob(file_pattern)
        else:
            files = dir_path.glob(file_pattern)
        
        for file_path in files:
            if file_path.is_file():
                text = DocumentLoader.load_text_file(str(file_path))
                if text:
                    documents.append({
                        "text": text,
                        "source": str(file_path),
                        "filename": file_path.name
                    })
        
        logger.info(f"Loaded {len(documents)} documents from {directory}")
        return documents


class TextPreprocessor:
    """Preprocess text for better retrieval"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:()\-\']', '', text)
        
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        return text.strip()
    
    @staticmethod
    def extract_metadata(text: str) -> Dict:
        """Extract metadata from text"""
        metadata = {
            "length": len(text),
            "word_count": len(text.split()),
            "has_code": bool(re.search(r'```|`\w+`', text)),
            "has_urls": bool(re.search(r'https?://', text)),
        }
        
        # Extract title if present (first line starting with #)
        title_match = re.match(r'^#\s+(.+)$', text, re.MULTILINE)
        if title_match:
            metadata["title"] = title_match.group(1).strip()
        
        return metadata
    
    @staticmethod
    def split_by_sections(text: str) -> List[Tuple[str, str]]:
        """
        Split text by markdown sections
        
        Returns:
            List of (section_title, section_content) tuples
        """
        sections = []
        current_title = "Introduction"
        current_content = []
        
        lines = text.split('\n')
        
        for line in lines:
            # Check if line is a header
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            
            if header_match:
                # Save previous section
                if current_content:
                    sections.append((current_title, '\n'.join(current_content).strip()))
                
                # Start new section
                current_title = header_match.group(2).strip()
                current_content = []
            else:
                current_content.append(line)
        
        # Add last section
        if current_content:
            sections.append((current_title, '\n'.join(current_content).strip()))
        
        return sections


class DocumentProcessor:
    """
    Complete document processing pipeline
    Combines loading, preprocessing, and chunking
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        clean_text: bool = True
    ):
        self.chunker = DocumentChunker(chunk_size, chunk_overlap)
        self.loader = DocumentLoader()
        self.preprocessor = TextPreprocessor()
        self.clean_text = clean_text
    
    def process_file(self, file_path: str) -> List[Dict]:
        """Process a single file into chunks"""
        # Load file
        text = self.loader.load_text_file(file_path)
        
        if not text:
            return []
        
        # Clean text
        if self.clean_text:
            text = self.preprocessor.clean_text(text)
        
        # Extract metadata
        metadata = self.preprocessor.extract_metadata(text)
        metadata["source"] = file_path
        
        # Chunk text
        chunks = self.chunker.chunk_text(text, metadata)
        
        return chunks
    
    def process_directory(
        self,
        directory: str,
        file_pattern: str = "*.txt",
        recursive: bool = False
    ) -> List[Dict]:
        """Process all files in directory"""
        documents = self.loader.load_directory(directory, file_pattern, recursive)
        
        all_chunks = []
        
        for doc in documents:
            text = doc["text"]
            
            if self.clean_text:
                text = self.preprocessor.clean_text(text)
            
            metadata = self.preprocessor.extract_metadata(text)
            metadata.update({
                "source": doc.get("source", "unknown"),
                "filename": doc.get("filename", "unknown")
            })
            
            chunks = self.chunker.chunk_text(text, metadata)
            all_chunks.extend(chunks)
        
        logger.info(f"Processed {len(documents)} documents into {len(all_chunks)} chunks")
        return all_chunks
    
    def process_text_list(self, texts: List[str], metadata_list: Optional[List[Dict]] = None) -> List[Dict]:
        """Process a list of texts"""
        all_chunks = []
        
        for i, text in enumerate(texts):
            if self.clean_text:
                text = self.preprocessor.clean_text(text)
            
            metadata = metadata_list[i] if metadata_list and i < len(metadata_list) else {}
            metadata.update(self.preprocessor.extract_metadata(text))
            
            chunks = self.chunker.chunk_text(text, metadata)
            all_chunks.extend(chunks)
        
        return all_chunks


def load_default_knowledge_base() -> List[str]:
    """Load default knowledge base from dataset"""
    dataset_path = Path(__file__).parent / "dataset" / "data_from_wiki.txt"
    
    if dataset_path.exists():
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split by double newlines or numbered sections
            documents = re.split(r'\n\n+|\d+\.\s+', content)
            documents = [doc.strip() for doc in documents if doc.strip() and len(doc.strip()) > 50]
            
            logger.info(f"Loaded {len(documents)} documents from default knowledge base")
            return documents
        except Exception as e:
            logger.error(f"Failed to load default knowledge base: {e}")
    
    # Fallback to hardcoded knowledge base
    return get_fallback_knowledge_base()


def get_fallback_knowledge_base() -> List[str]:
    """Get fallback knowledge base"""
    return [
        "Retrieval Augmented Generation (RAG) represents a sophisticated hybrid approach in the field of artificial intelligence, particularly within the realm of natural language processing (NLP).",
        "It innovatively combines the capabilities of neural network-based language models with retrieval systems to enhance the generation of text, making it more accurate, informative, and contextually relevant.",
        "This methodology leverages the strengths of both generative and retrieval architectures to tackle complex tasks that require not only linguistic fluency but also factual correctness and depth of knowledge.",
        "At the core of Retrieval Augmented Generation (RAG) is a generative model, typically a transformer-based neural network, similar to those used in models like GPT (Generative Pre-trained Transformer) or BERT (Bidirectional Encoder Representations from Transformers).",
        "This component is responsible for producing coherent and contextually appropriate language outputs based on a mixture of input prompts and additional information fetched by the retrieval component.",
        "Complementing the language model is the retrieval system, which is usually built on a database of documents or a corpus of texts.",
        "This system uses techniques from information retrieval to find and fetch documents that are relevant to the input query or prompt.",
        "The mechanism of relevance determination can range from simple keyword matching to more complex semantic search algorithms which interpret the meaning behind the query to find the best matches.",
        "Graph RAG extends traditional RAG by incorporating knowledge graphs to capture relationships between entities and concepts.",
        "Knowledge graphs provide structured representations of information, enabling more sophisticated reasoning and context-aware retrieval.",
        "In Graph RAG, entities are extracted from documents and connected through meaningful relationships, forming a semantic network.",
        "This graph structure allows the system to traverse related concepts and discover relevant information through multi-hop reasoning.",
        "The integration of graph-based retrieval with vector similarity search creates a powerful hybrid approach.",
        "Vector embeddings capture semantic similarity at the document level, while knowledge graphs capture explicit relationships between concepts.",
        "By combining these approaches, Graph RAG can retrieve both semantically similar documents and contextually related information.",
        "Entity extraction is a crucial step in building knowledge graphs, identifying key concepts, people, organizations, and events in text.",
        "Relation extraction identifies how entities are connected, such as 'works for', 'located in', or 'causes'.",
        "The quality of entity and relation extraction directly impacts the effectiveness of graph-based retrieval.",
        "Graph traversal algorithms enable the system to explore neighborhoods of relevant entities, discovering related information.",
        "Community detection in knowledge graphs can identify clusters of related concepts, improving retrieval relevance.",
        "Hybrid retrieval strategies combine multiple signals: vector similarity, graph proximity, and traditional keyword matching.",
        "Reranking mechanisms can further refine results by considering both semantic relevance and graph-based relationships.",
        "Graph RAG systems can maintain temporal information, tracking how relationships evolve over time.",
        "The graph structure enables explainable AI by showing the reasoning path from query to retrieved information.",
        "Incremental graph updates allow the system to incorporate new information without rebuilding the entire knowledge base.",
    ]

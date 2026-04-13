import logging
import re
from typing import List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

DATASET_PATH = Path(__file__).parent / "dataset" / "data_from_wiki.txt"


class DocumentChunker:
    """Split text into overlapping chunks for vector embedding."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 128):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str) -> List[str]:
        """Split a single text into overlapping character-level chunks."""
        text = text.strip()
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        sentences = self._split_sentences(text)
        chunks: List[str] = []
        current = ""

        for sentence in sentences:
            if not sentence:
                continue

            if len(current) + len(sentence) + 1 <= self.chunk_size:
                current = f"{current} {sentence}".strip()
            else:
                if current:
                    chunks.append(current)
                    # overlap: take last chunk_overlap chars as prefix
                    overlap = current[-self.chunk_overlap:] if len(current) > self.chunk_overlap else current
                    current = f"{overlap} {sentence}".strip()
                else:
                    # single sentence longer than chunk_size — force-split
                    for i in range(0, len(sentence), self.chunk_size - self.chunk_overlap):
                        part = sentence[i:i + self.chunk_size]
                        if part.strip():
                            chunks.append(part.strip())
                    current = ""

        if current.strip():
            chunks.append(current.strip())

        return chunks

    def chunk_documents(self, documents: List[str]) -> List[str]:
        """Chunk a list of documents, returning flat list of all chunks."""
        all_chunks: List[str] = []
        for doc in documents:
            all_chunks.extend(self.chunk_text(doc))
        return all_chunks

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        parts = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in parts if s.strip()]


def load_raw_documents() -> List[str]:
    """Load dataset as raw document strings (paragraph-level split).

    Used by Graph RAG where each document = one unit for entity extraction.
    """
    if not DATASET_PATH.exists():
        logger.warning("Dataset file not found: %s", DATASET_PATH)
        return []

    try:
        content = DATASET_PATH.read_text(encoding="utf-8")
    except Exception as e:
        logger.error("Failed to read dataset: %s", e)
        return []

    # Split on double newlines or numbered items
    documents = re.split(r'\n\n+|\d+\.\s+', content)
    documents = [doc.strip() for doc in documents if doc.strip() and len(doc.strip()) > 50]
    logger.info("Loaded %d raw documents from %s", len(documents), DATASET_PATH)
    return documents


def chunk_documents(
    raw_docs: List[str],
    chunk_size: int = 512,
    chunk_overlap: int = 128,
) -> List[str]:
    """Split raw documents into overlapping chunks for vector embedding.

    Used by Vector RAG where smaller chunks improve retrieval precision.
    """
    if not raw_docs:
        return []

    chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = chunker.chunk_documents(raw_docs)
    logger.info("Chunked %d documents into %d chunks (size=%d, overlap=%d)",
                len(raw_docs), len(chunks), chunk_size, chunk_overlap)
    return chunks


# Backward compatibility alias
load_default_knowledge_base = load_raw_documents

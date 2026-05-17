"""
store.py — In-memory document storage and indexing.

This module provides a DocumentStore class to manage document chunks,
their vector embeddings, and a BM25 index for hybrid retrieval.
"""

import re
import numpy as np
from rank_bm25 import BM25Okapi

class DocumentStore:
    """
    In-memory store for document chunks and their associated indices.
    
    Attributes:
        chunks (list[dict]): List of all chunk dictionaries added to the store.
        embedding_matrix (np.ndarray): Numpy matrix of embeddings for all chunks.
        bm25 (BM25Okapi): BM25 index for keyword-based retrieval.
    """

    def __init__(self):
        self.chunks = []
        self.embedding_matrix = None
        self.bm25 = None

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenizer for BM25: lowercase and alphanumeric words."""
        return re.findall(r'\w+', text.lower())

    def add_document(self, chunks_with_embeddings: list[dict]):
        """
        Add new chunks (with their embeddings) to the store and update indices.

        Args:
            chunks_with_embeddings: List of chunk dicts, each containing 
                                    'text', 'embedding', 'doc_name', etc.
        """
        if not chunks_with_embeddings:
            return

        # 1. Append chunks to the master list
        self.chunks.extend(chunks_with_embeddings)

        # 2. Update the embedding matrix
        # Each 'embedding' is expected to be a list/array of floats.
        new_embeddings = np.array(
            [c["embedding"] for c in chunks_with_embeddings], 
            dtype=np.float32
        )
        
        if self.embedding_matrix is None:
            self.embedding_matrix = new_embeddings
        else:
            # Stack the new embeddings onto the existing matrix
            self.embedding_matrix = np.vstack([self.embedding_matrix, new_embeddings])

        # 3. Update the BM25 index
        # Since rank_bm25 doesn't support incremental updates easily, 
        # we re-index the entire corpus in the store.
        tokenized_corpus = [self._tokenize(c["text"]) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def clear(self):
        """Reset the store and clear all data."""
        self.chunks = []
        self.embedding_matrix = None
        self.bm25 = None

    def get_all_chunks(self) -> list[dict]:
        """Return the list of all stored chunks."""
        return self.chunks

    def get_embedding_matrix(self) -> np.ndarray:
        """Return the current numpy embedding matrix."""
        return self.embedding_matrix

    def remove_document(self, doc_name: str) -> int:
        """
        Remove all chunks belonging to `doc_name` and rebuild indices.

        Args:
            doc_name: The doc_name tag to remove.

        Returns:
            Number of chunks removed.
        """
        before = len(self.chunks)
        self.chunks = [c for c in self.chunks if c.get("doc_name") != doc_name]
        removed = before - len(self.chunks)

        if not self.chunks:
            self.embedding_matrix = None
            self.bm25 = None
        else:
            # Rebuild matrix and BM25 from remaining chunks
            self.embedding_matrix = np.array(
                [c["embedding"] for c in self.chunks], dtype=np.float32
            )
            tokenized_corpus = [self._tokenize(c["text"]) for c in self.chunks]
            self.bm25 = BM25Okapi(tokenized_corpus)

        return removed

    def get_bm25(self) -> BM25Okapi:
        """Return the current BM25 index."""
        return self.bm25

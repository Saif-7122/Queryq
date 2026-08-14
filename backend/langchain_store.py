"""
langchain_store.py — Vector Store and Keyword Index management using LangChain.
"""

import os
import shutil
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

PERSIST_DIR = "./chroma_db"

class LangchainStore:
    def __init__(self):
        # We use fastembed directly through LangChain
        self.embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        
        # Initialize Chroma DB
        self.vectorstore = Chroma(
            collection_name="queryq_docs",
            embedding_function=self.embeddings,
            persist_directory=PERSIST_DIR
        )
        
        # We will hold an in-memory BM25 retriever for RRF
        # Note: If you want BM25 to persist, it usually requires a custom saving mechanism or Elasticsearch.
        # For simplicity in this migration, we rebuild BM25 from Chroma documents if the server restarts.
        self.bm25_retriever = None
        self._init_bm25_from_chroma()

    def _init_bm25_from_chroma(self):
        """Rebuilds the BM25 index from all documents currently in Chroma."""
        # Chroma's get() returns all stored documents
        db_data = self.vectorstore.get()
        if not db_data or not db_data.get("documents"):
            self.bm25_retriever = None
            return

        docs = []
        for i, text in enumerate(db_data["documents"]):
            metadata = db_data["metadatas"][i]
            docs.append(Document(page_content=text, metadata=metadata))
            
        if docs:
            self.bm25_retriever = BM25Retriever.from_documents(docs)
            self.bm25_retriever.k = 10 # Set default K for BM25
        else:
            self.bm25_retriever = None

    def add_documents(self, chunks: List[Dict[str, Any]]):
        """
        Add chunks to Chroma and update the BM25 retriever.
        """
        if not chunks:
            return

        docs = []
        for chunk in chunks:
            # We assume chunk has 'text' and other metadata like 'doc_name', 'page_num'
            metadata = {k: v for k, v in chunk.items() if k != "text"}
            docs.append(Document(page_content=chunk["text"], metadata=metadata))

        # Add to Chroma
        self.vectorstore.add_documents(docs)

        # Update BM25
        if self.bm25_retriever is None:
            self.bm25_retriever = BM25Retriever.from_documents(docs)
            self.bm25_retriever.k = 10
        else:
            # Langchain's BM25Retriever doesn't support incremental adds easily out-of-the-box,
            # so we just rebuild it. For small datasets, this is fast enough.
            self._init_bm25_from_chroma()

    def get_documents_metadata(self) -> List[Dict[str, Any]]:
        """Return metadata for all stored documents (for listing docs)."""
        db_data = self.vectorstore.get()
        if not db_data or not db_data.get("metadatas"):
            return []
        return db_data["metadatas"]

    def remove_document(self, doc_name: str) -> int:
        """
        Remove a document by its doc_name and rebuild BM25.
        Returns the number of chunks deleted.
        """
        db_data = self.vectorstore.get(where={"doc_name": doc_name})
        if not db_data or not db_data.get("ids"):
            return 0
            
        ids_to_delete = db_data["ids"]
        self.vectorstore.delete(ids_to_delete)
        self._init_bm25_from_chroma()
        return len(ids_to_delete)

    def clear(self):
        """Clear the entire database."""
        if os.path.exists(PERSIST_DIR):
            shutil.rmtree(PERSIST_DIR)
        
        # Re-initialize
        self.vectorstore = Chroma(
            collection_name="queryq_docs",
            embedding_function=self.embeddings,
            persist_directory=PERSIST_DIR
        )
        self.bm25_retriever = None

    def get_vector_retriever(self, k: int = 10):
        return self.vectorstore.as_retriever(search_kwargs={"k": k})

    def get_bm25_retriever(self, k: int = 10):
        if self.bm25_retriever:
            self.bm25_retriever.k = k
        return self.bm25_retriever

"""
Vector Store module for the Retail Conversational AI project.
Handles FAISS index creation, saving, loading, and similarity search.
"""

import os
from typing import List

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.config import EMBEDDING_MODEL, FAISS_INDEX_PATH, RETRIEVAL_TOP_K


class RetailVectorStore:
    """Manages the FAISS vector store for retail knowledge retrieval."""

    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.vector_db: FAISS = None

    def build_index(self, chunks: List[Document]) -> FAISS:
        """
        Create a FAISS vector index from document chunks.

        Args:
            chunks: List of LangChain Document objects.

        Returns:
            The built FAISS vector store.
        """
        print(f"Starting vectorization of {len(chunks)} chunks...")
        print("This may take a few minutes for large datasets.")
        self.vector_db = FAISS.from_documents(chunks, self.embeddings)
        print("Vectorization complete.")
        return self.vector_db

    def save(self, path: str = FAISS_INDEX_PATH):
        """Save the FAISS index to disk."""
        if self.vector_db is None:
            raise ValueError("No index to save. Call build_index() first.")
        os.makedirs(path, exist_ok=True)
        self.vector_db.save_local(path)
        print(f"Vector store saved to '{path}'.")

    def load(self, path: str = FAISS_INDEX_PATH) -> FAISS:
        """Load a previously saved FAISS index from disk."""
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No index found at '{path}'. Build and save the index first."
            )
        self.vector_db = FAISS.load_local(
            path,
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
        print(f"Vector store loaded from '{path}'.")
        return self.vector_db

    def search(self, query: str, k: int = RETRIEVAL_TOP_K) -> List[Document]:
        """
        Perform similarity search against the FAISS index.

        Args:
            query: The user's natural language query.
            k: Number of top results to return.

        Returns:
            List of matching Document chunks.
        """
        if self.vector_db is None:
            raise ValueError("Index not loaded. Call build_index() or load() first.")
        docs = self.vector_db.similarity_search(query, k=k)
        return docs

    def build_and_save(self, chunks: List[Document], path: str = FAISS_INDEX_PATH):
        """Convenience method: build index and immediately save it."""
        self.build_index(chunks)
        self.save(path)
        return self.vector_db

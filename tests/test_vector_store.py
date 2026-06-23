"""Tests for the RetailVectorStore module."""

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document


@pytest.fixture
def sample_chunks():
    return [
        Document(page_content="Retail Query: How do I return an item?\nStore Response: You can return within 30 days."),
        Document(page_content="Retail Query: Where is my order?\nStore Response: Check your tracking email."),
        Document(page_content="Retail Query: Can I change my address?\nStore Response: Contact support within 1 hour of ordering."),
    ]


@patch("src.vector_store.HuggingFaceEmbeddings")
@patch("src.vector_store.FAISS")
def test_build_index(mock_faiss, mock_embeddings, sample_chunks):
    """Test that build_index calls FAISS.from_documents correctly."""
    from src.vector_store import RetailVectorStore

    vs = RetailVectorStore()
    vs.build_index(sample_chunks)

    mock_faiss.from_documents.assert_called_once()


@patch("src.vector_store.HuggingFaceEmbeddings")
def test_search_raises_without_index(mock_embeddings):
    """Test that search raises an error when no index is loaded."""
    from src.vector_store import RetailVectorStore

    vs = RetailVectorStore()
    with pytest.raises(ValueError, match="Index not loaded"):
        vs.search("test query")


@patch("src.vector_store.HuggingFaceEmbeddings")
@patch("src.vector_store.FAISS")
def test_search_returns_docs(mock_faiss, mock_embeddings, sample_chunks):
    """Test that search returns a list of documents."""
    from src.vector_store import RetailVectorStore

    mock_db = MagicMock()
    mock_db.similarity_search.return_value = sample_chunks[:2]
    mock_faiss.from_documents.return_value = mock_db

    vs = RetailVectorStore()
    vs.build_index(sample_chunks)
    results = vs.search("return policy", k=2)

    assert len(results) == 2
    assert isinstance(results[0], Document)

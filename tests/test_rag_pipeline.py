"""Tests for the RAGPipeline module."""

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document


@pytest.fixture
def mock_vector_store():
    vs = MagicMock()
    vs.search.return_value = [
        Document(page_content="Retail Query: Returns?\nStore Response: 30-day return policy."),
    ]
    return vs


@patch("src.rag_pipeline.Groq")
def test_ethical_filter_blocks_sensitive(mock_groq, mock_vector_store):
    """Test that sensitive queries are blocked."""
    from src.rag_pipeline import RAGPipeline

    pipeline = RAGPipeline(mock_vector_store)

    assert pipeline.ethical_filter("how to hack the system") is False
    assert pipeline.ethical_filter("how to bypass security") is False
    assert pipeline.ethical_filter("track my order") is True


@patch("src.rag_pipeline.Groq")
def test_query_blocked_returns_guardrail_message(mock_groq, mock_vector_store):
    """Test that blocked queries return the guardrail message."""
    from src.rag_pipeline import RAGPipeline

    pipeline = RAGPipeline(mock_vector_store)
    response = pipeline.query("how to exploit the checkout")

    assert "cannot assist" in response.lower()
    mock_vector_store.search.assert_not_called()


@patch("src.rag_pipeline.Groq")
def test_query_calls_retrieval_and_llm(mock_groq, mock_vector_store):
    """Test that a safe query goes through retrieval and LLM."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value.choices[0].message.content = (
        "You can return items within 30 days."
    )
    mock_groq.return_value = mock_client

    from src.rag_pipeline import RAGPipeline

    pipeline = RAGPipeline(mock_vector_store)
    response = pipeline.query("What is your return policy?")

    mock_vector_store.search.assert_called_once()
    mock_client.chat.completions.create.assert_called_once()
    assert isinstance(response, str)
    assert len(response) > 0

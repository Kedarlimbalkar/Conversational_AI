"""
RAG Pipeline module for the Retail Conversational AI project.
Orchestrates retrieval, context augmentation, and LLM response generation.
"""

import os
from typing import List

from groq import Groq
from langchain_core.documents import Document

from src.config import (
    GROQ_API_KEY,
    LLM_MODEL,
    MAX_TOKENS,
    RETRIEVAL_TOP_K,
    SENSITIVE_KEYWORDS,
    SIMPLE_PROMPT_TEMPLATE,
)
from src.vector_store import RetailVectorStore


class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline for retail customer support.

    Flow:
        User Query → Ethical Filter → Retrieval → Context Augmentation → LLM → Response
    """

    def __init__(self, vector_store: RetailVectorStore):
        self.vector_store = vector_store
        self.client = Groq(api_key=GROQ_API_KEY)

    def ethical_filter(self, query: str) -> bool:
        """
        Check if the query contains sensitive/harmful keywords.

        Returns:
            True if the query is safe, False if it should be blocked.
        """
        return not any(word in query.lower() for word in SENSITIVE_KEYWORDS)

    def retrieve(self, query: str, k: int = RETRIEVAL_TOP_K) -> List[Document]:
        """
        Retrieve the top-k most relevant chunks from the vector store.

        Args:
            query: User's natural language query.
            k: Number of chunks to retrieve.

        Returns:
            List of relevant Document chunks.
        """
        return self.vector_store.search(query, k=k)

    def augment(self, query: str, docs: List[Document]) -> str:
        """
        Build the augmented prompt by injecting retrieved context.

        Args:
            query: Original user query.
            docs: Retrieved document chunks.

        Returns:
            Full prompt string ready for the LLM.
        """
        context = "\n\n".join([d.page_content for d in docs])
        prompt = SIMPLE_PROMPT_TEMPLATE.format(context=context, query=query)
        return prompt

    def generate(self, prompt: str) -> str:
        """
        Call the Groq LLM with the augmented prompt.

        Args:
            prompt: The full RAG prompt.

        Returns:
            The LLM's text response.
        """
        response = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=LLM_MODEL,
            max_tokens=MAX_TOKENS,
        )
        return response.choices[0].message.content

    def query(self, user_query: str) -> str:
        """
        Full RAG pipeline: filter → retrieve → augment → generate.

        Args:
            user_query: The raw user input.

        Returns:
            The AI-generated response string.
        """
        # Step 1: Ethical guardrail
        if not self.ethical_filter(user_query):
            return (
                "I cannot assist with security-related queries. "
                "Please contact our official support."
            )

        # Step 2: Retrieve relevant chunks
        docs = self.retrieve(user_query)

        # Step 3: Build augmented prompt
        prompt = self.augment(user_query, docs)

        # Step 4: Generate response
        response = self.generate(prompt)

        return response

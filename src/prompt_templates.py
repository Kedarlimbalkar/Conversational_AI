"""
Prompt Templates for the Retail Conversational AI project.
All prompts used across the pipeline are centralized here.
"""

from langchain_core.prompts import ChatPromptTemplate

# ---------------------------------------------------------------------------
# RAG Prompt (structured, bullet-point style)
# ---------------------------------------------------------------------------

RAG_PROMPT_TEMPLATE = """Answer the question based on the provided retail context only.
Ensure the response is structured exactly the same way (using bullet points for steps).

<context>
{context}
</context>

User's Question: {input}

Answer:"""

RAG_PROMPT = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

# ---------------------------------------------------------------------------
# Simple Inline Prompt (used in agents / direct Groq calls)
# ---------------------------------------------------------------------------

SIMPLE_PROMPT_TEMPLATE = (
    "Answer using only the provided retail context:\n\n"
    "{context}\n\n"
    "User: {query}"
)

# ---------------------------------------------------------------------------
# System Prompt (for multi-turn conversations)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a helpful retail e-commerce customer support assistant.
You only answer questions based on the provided retail context.
If the answer is not in the context, politely say you don't have that information
and suggest the customer contact official support.
Never make up information. Always be professional and concise."""

# ---------------------------------------------------------------------------
# Ethical / Guardrail Response
# ---------------------------------------------------------------------------

BLOCKED_RESPONSE = (
    "I cannot assist with security-related queries. "
    "Please contact our official support."
)

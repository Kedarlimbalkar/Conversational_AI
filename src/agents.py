"""
Agents module for the Retail Conversational AI project.
Implements the LangGraph multi-agent workflow.
"""

from typing import TypedDict, List

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from src.config import (
    GROQ_API_KEY,
    LLM_MODEL,
    MAX_TOKENS,
    SENSITIVE_KEYWORDS,
    RETRIEVAL_TOP_K,
)
from src.vector_store import RetailVectorStore


# ---------------------------------------------------------------------------
# State Schema
# ---------------------------------------------------------------------------

class MessagesState(TypedDict):
    """Tracks the full conversation history across agent nodes."""
    messages: List[BaseMessage]


# ---------------------------------------------------------------------------
# Agent Node Functions
# ---------------------------------------------------------------------------

def make_retail_agent(vector_store: RetailVectorStore):
    """
    Factory that returns the main retail agent node function,
    with the vector store injected via closure.
    """
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)

    def call_retail_model(state: MessagesState) -> dict:
        """
        Single-node LangGraph agent that:
        1. Applies an ethical filter
        2. Retrieves relevant retail context
        3. Generates a grounded response
        """
        user_query = state["messages"][-1].content

        # Step 1: Ethical Filter
        if any(word in user_query.lower() for word in SENSITIVE_KEYWORDS):
            return {
                "messages": [
                    AIMessage(
                        content=(
                            "I cannot assist with security-related queries. "
                            "Please contact our official support."
                        )
                    )
                ]
            }

        # Step 2: Retrieval
        docs = vector_store.search(user_query, k=RETRIEVAL_TOP_K)
        context = "\n\n".join([d.page_content for d in docs])

        # Step 3: Answer Generation
        prompt = (
            f"Answer using only the provided retail context:\n\n"
            f"{context}\n\nUser: {user_query}"
        )

        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=LLM_MODEL,
            max_tokens=MAX_TOKENS,
        )

        return {
            "messages": [AIMessage(content=response.choices[0].message.content)]
        }

    return call_retail_model


# ---------------------------------------------------------------------------
# Graph Builder
# ---------------------------------------------------------------------------

def build_retail_graph(vector_store: RetailVectorStore):
    """
    Build and compile the LangGraph workflow.

    Args:
        vector_store: Initialized RetailVectorStore with a loaded index.

    Returns:
        Compiled LangGraph app.
    """
    retail_agent = make_retail_agent(vector_store)

    workflow = StateGraph(MessagesState)
    workflow.add_node("retail_bot", retail_agent)
    workflow.add_edge(START, "retail_bot")
    workflow.add_edge("retail_bot", END)

    app = workflow.compile()
    print("Retail LangGraph Workflow compiled successfully!")
    return app


# ---------------------------------------------------------------------------
# Helper: Run a single query through the graph
# ---------------------------------------------------------------------------

def run_query(app, user_input: str) -> str:
    """
    Send a user message through the compiled LangGraph app.

    Args:
        app: Compiled LangGraph application.
        user_input: The user's question.

    Returns:
        The assistant's response string.
    """
    result = app.invoke({"messages": [HumanMessage(content=user_input)]})
    return result["messages"][-1].content

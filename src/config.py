"""
Configuration file for the Retail Conversational AI project.
All tunable parameters are defined here.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys (loaded from .env) ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
NGROK_TOKEN = os.getenv("NGROK_TOKEN", "")

# --- Dataset ---
DATASET_NAME = "bitext/Bitext-retail-ecommerce-llm-chatbot-training-dataset"
DATASET_SPLIT = "train"
TOTAL_RECORDS = 44884  # Expected records from Hugging Face dataset

# --- Chunking Strategy ---
CHUNK_SIZE_PRIMARY = 3000     # Characters per chunk
CHUNK_OVERLAP = 500           # Overlap between chunks

# --- Embedding Model ---
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# --- Vector Store ---
FAISS_INDEX_PATH = "models/faiss_index"
RETRIEVAL_TOP_K = 3           # Number of chunks to retrieve per query

# --- LLM ---
LLM_MODEL = "llama-3.3-70b-versatile"
MAX_TOKENS = 1024

# --- Ethical Guardrails ---
SENSITIVE_KEYWORDS = ["hack", "bypass", "breach", "exploit"]

# --- Performance Targets ---
TARGET_ACCURACY = 0.92        # 92% domain accuracy
TARGET_LATENCY_SECONDS = 1.2  # <1.2s query latency

# --- Streamlit App ---
APP_PORT = 8501
APP_TITLE = "🛒 E-Commerce Support AI"

# --- Prompt Templates ---
RAG_PROMPT_TEMPLATE = """Answer the question based on the provided retail context only.
Ensure the response is structured exactly the same way (using bullet points for steps).

<context>
{context}
</context>

User's Question: {input}

Answer:"""

SIMPLE_PROMPT_TEMPLATE = "Answer using only the provided retail context:\n\n{context}\n\nUser: {query}"

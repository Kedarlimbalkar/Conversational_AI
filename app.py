"""
Streamlit Web App for the Retail Conversational AI.
Run with: streamlit run app.py
"""

import streamlit as st
import os
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Retail AI", page_icon="🛒")
st.title("🛒 E-Commerce Support AI")
st.caption("Powered by RAG + LLaMA 3.3 via Groq")

# ---------------------------------------------------------------------------
# Load Resources (cached so they only load once)
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "models/faiss_index")
SENSITIVE_KEYWORDS = ["hack", "bypass", "breach", "exploit"]


@st.cache_resource
def load_vector_db():
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(
        FAISS_INDEX_PATH,
        emb,
        allow_dangerous_deserialization=True,
    )


@st.cache_resource
def load_groq_client():
    return Groq(api_key=GROQ_API_KEY)


try:
    db = load_vector_db()
    client = load_groq_client()
    st.success("Knowledge base loaded!", icon="✅")
except Exception as e:
    st.error(
        f"Could not load vector store from '{FAISS_INDEX_PATH}'. "
        "Please run the setup pipeline first to build the FAISS index.\n\n"
        f"Error: {e}"
    )
    st.stop()

# ---------------------------------------------------------------------------
# Chat Interface
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle new input
if prompt := st.chat_input("Ask about your order, returns, shipping..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Ethical guardrail
            if any(word in prompt.lower() for word in SENSITIVE_KEYWORDS):
                response = (
                    "I cannot assist with security-related queries. "
                    "Please contact our official support."
                )
            else:
                # RAG Retrieval
                docs = db.similarity_search(prompt, k=3)
                context = "\n\n".join([d.page_content for d in docs])

                res = client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": f"Context:\n{context}\n\nUser: {prompt}",
                        }
                    ],
                    model="llama3-8b-8192",
                )
                response = res.choices[0].message.content

        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

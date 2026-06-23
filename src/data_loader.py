"""
Data Loader module for the Retail Conversational AI project.
Handles loading from Hugging Face, preprocessing, and chunking.
"""

import pandas as pd
from datasets import load_dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List

from src.config import (
    DATASET_NAME,
    DATASET_SPLIT,
    CHUNK_SIZE_PRIMARY,
    CHUNK_OVERLAP,
)


class RetailDataLoader:
    """Loads and preprocesses the retail e-commerce dataset."""

    def __init__(self):
        self.raw_data: pd.DataFrame = None
        self.documents: List[Document] = []
        self.chunks: List[Document] = []
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE_PRIMARY,
            chunk_overlap=CHUNK_OVERLAP,
        )

    def load_raw_data(self) -> pd.DataFrame:
        """Load dataset from Hugging Face."""
        print(f"Loading dataset: {DATASET_NAME} ...")
        dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
        self.raw_data = pd.DataFrame(dataset)
        print(f"Extracted {len(self.raw_data)} retail support records.")
        return self.raw_data

    def preprocess(self) -> List[Document]:
        """
        Convert raw rows to LangChain Documents.
        Uses 'instruction' as user query and 'response' as bot answer.
        """
        if self.raw_data is None:
            raise ValueError("Call load_raw_data() before preprocess().")

        self.documents = []
        for _, row in self.raw_data.iterrows():
            user_query = row.get("instruction", "")
            bot_response = row.get("response", "")

            if user_query and bot_response:
                content = (
                    f"Retail Query: {user_query}\n"
                    f"Store Policy/Response: {bot_response}"
                )
                self.documents.append(Document(page_content=content))

        print(f"Preprocessed {len(self.documents)} instruction/response pairs.")
        return self.documents

    def create_chunks(self) -> List[Document]:
        """
        Apply the 3000/500 chunking strategy to segment the knowledge base.
        """
        if not self.documents:
            raise ValueError("Call preprocess() before create_chunks().")

        self.chunks = self.text_splitter.split_documents(self.documents)
        print(
            f"Created {len(self.chunks)} chunks "
            f"(chunk_size={CHUNK_SIZE_PRIMARY}, overlap={CHUNK_OVERLAP})."
        )
        return self.chunks

    def run(self) -> List[Document]:
        """Full pipeline: load → preprocess → chunk."""
        self.load_raw_data()
        self.preprocess()
        self.create_chunks()
        return self.chunks

    def validate(self):
        """Print basic stats about the chunks."""
        if not self.chunks:
            print("No chunks to validate. Run the pipeline first.")
            return

        lengths = [len(c.page_content) for c in self.chunks]
        df_stats = pd.DataFrame(lengths, columns=["chunk_char_count"])
        print("\n--- Data Validation Summary ---")
        print(df_stats.describe())
        print(f"\nSample Chunk (index 100):\n{self.chunks[100].page_content}")

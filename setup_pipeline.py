"""
One-time setup script: Downloads the dataset, builds the FAISS index, and saves it.
Run this ONCE before starting the app:
    python setup_pipeline.py
"""

from src.data_loader import RetailDataLoader
from src.vector_store import RetailVectorStore
from src.config import FAISS_INDEX_PATH

def main():
    print("=" * 50)
    print("  Retail AI - Setup Pipeline")
    print("=" * 50)

    # Step 1: Load and chunk data
    print("\n[1/3] Loading and chunking dataset...")
    loader = RetailDataLoader()
    chunks = loader.run()
    loader.validate()

    # Step 2: Build FAISS index
    print("\n[2/3] Building FAISS vector index...")
    vs = RetailVectorStore()
    vs.build_and_save(chunks, path=FAISS_INDEX_PATH)

    print("\n[3/3] Setup complete!")
    print(f"  - {len(chunks)} chunks indexed")
    print(f"  - Index saved to: {FAISS_INDEX_PATH}")
    print("\nYou can now run: streamlit run app.py")

if __name__ == "__main__":
    main()

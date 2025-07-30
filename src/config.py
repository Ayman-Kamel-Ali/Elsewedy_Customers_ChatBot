# config.py

import os

# Base directory for the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# --- Data Paths ---
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
PRODUCT_DOCS_DIR = os.path.join(DATA_DIR, 'product_docs')

# --- Vector Store Configuration ---
VECTOR_STORE_DIR = os.path.join(PROJECT_ROOT, 'vector_store')
CHROMA_DB_PATH = os.path.join(VECTOR_STORE_DIR, 'chroma_db')
CHROMA_COLLECTION_NAME = "product_info_collection"

# --- Embedding Model Configuration ---
# You can choose other models from sentence-transformers if needed
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" # A good balance of size and performance

# --- LLM Configuration (Ollama) ---
OLLAMA_BASE_URL = "http://localhost:11434" # Default Ollama URL
OLLAMA_MODEL_NAME = "tinyllama" # Ensure this model is pulled in Ollama

# --- RAG Configuration ---
CHUNK_SIZE = 500  # Size of text chunks for embedding
CHUNK_OVERLAP = 200 # Overlap between chunks to maintain context
TOP_K_RETRIEVAL = 5 # Number of relevant chunks to retrieve for context

# --- Prompt Templates ---
# This is the system prompt for the RAG chain.
# It instructs the LLM on how to behave and use the provided context.
RAG_PROMPT_TEMPLATE = """
You are a helpful assistant providing information about company products.
Answer the following question based ONLY on the provided context.
If the answer is not in the context, state that you don't have enough information.
Keep your answer concise and to the point.

Context:
{context}

Question: {question}

Answer:
"""

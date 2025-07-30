# src/data_processor.py

import os
from typing import List

from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config import PRODUCT_DOCS_DIR, CHUNK_SIZE, CHUNK_OVERLAP

def load_documents(directory: str = PRODUCT_DOCS_DIR) -> List[Document]:
    """
    Loads documents from the specified directory.
    Supports PDF, TXT, and Markdown files.
    """
    documents = []
    for root, _, files in os.walk(directory):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            if file_name.endswith(".pdf"):
                print(f"Loading PDF: {file_name}")
                loader = PyPDFLoader(file_path)
            elif file_name.endswith(".txt"):
                print(f"Loading TXT: {file_name}")
                loader = TextLoader(file_path, encoding="utf-8")
            elif file_name.endswith((".md", ".markdown")):
                print(f"Loading Markdown: {file_name}")
                loader = UnstructuredMarkdownLoader(file_path)
            else:
                print(f"Skipping unsupported file type: {file_name}")
                continue
            try:
                documents.extend(loader.load())
            except Exception as e:
                print(f"Error loading {file_name}: {e}")
    print(f"Loaded {len(documents)} raw documents.")
    return documents

def split_documents(documents: List[Document]) -> List[Document]:
    """
    Splits loaded documents into smaller chunks using RecursiveCharacterTextSplitter.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split documents into {len(chunks)} chunks.")
    return chunks

def get_processed_documents() -> List[Document]:
    """
    Loads and splits all product documents.
    """
    print("Starting data processing...")
    raw_documents = load_documents()
    if not raw_documents:
        print("No documents found or loaded. Please ensure product documents are in the 'data/product_docs' directory.")
        return []
    processed_chunks = split_documents(raw_documents)
    print("Data processing complete.")
    return processed_chunks

if __name__ == "__main__":
    # Example usage:
    # Create some dummy files for testing
    os.makedirs(PRODUCT_DOCS_DIR, exist_ok=True)
    with open(os.path.join(PRODUCT_DOCS_DIR, "sample_product_info.txt"), "w") as f:
        f.write("This is a sample product description. It talks about features A, B, and C. " * 10)
        f.write("Another paragraph about product benefits and usage instructions. " * 10)
    with open(os.path.join(PRODUCT_DOCS_DIR, "faq.md"), "w") as f:
        f.write("# FAQ\n\n## Q: How to install?\nA: Follow steps 1, 2, 3.\n\n## Q: Warranty info?\nA: 1 year limited warranty.")

    chunks = get_processed_documents()
    for i, chunk in enumerate(chunks[:3]): # Print first 3 chunks for verification
        print(f"\n--- Chunk {i+1} ---")
        print(chunk.page_content)
        print(f"Source: {chunk.metadata.get('source', 'N/A')}")

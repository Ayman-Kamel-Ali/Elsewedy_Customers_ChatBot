# src/vector_db_manager.py

import os
from typing import List
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from typing import Optional

from config import CHROMA_DB_PATH, CHROMA_COLLECTION_NAME
from embedding_model import get_embedding_model
def get_or_create_vector_store(
    documents: List[Document],
    embedding_model: Embeddings,
    persist_directory: str = CHROMA_DB_PATH,
    collection_name: str = CHROMA_COLLECTION_NAME,
    force_recreate: bool = False
) -> Optional[Chroma]:
    """
    Gets an existing ChromaDB vector store or creates a new one from documents.
    If force_recreate is True, it will delete existing data and recreate.
    """
    # Ensure the persist directory exists
    os.makedirs(persist_directory, exist_ok=True)

    if force_recreate and os.path.exists(persist_directory):
        import shutil
        print(f"Force recreating vector store. Deleting existing: {persist_directory}")
        shutil.rmtree(persist_directory)
        os.makedirs(persist_directory, exist_ok=True)

    # Check if the vector store already exists and has content
    # LangChain's Chroma integration handles loading if directory exists
    try:
        # Attempt to load existing collection
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_model,
            collection_name=collection_name
        )
        # Check if the collection has any documents
        # This is a bit of a workaround as Chroma does not expose a direct count
        # We can try to get a dummy document or check if add_documents was called
        # For simplicity, we'll rely on the existence of the directory for now,
        # and always add documents if force_recreate is True or no documents were initially added.

        # A more robust check might involve trying to query and seeing if it fails or returns empty
        # For now, if documents are provided and force_recreate is False, we assume it's an update or initial load
        if not force_recreate and documents:
            print(f"Vector store '{collection_name}' loaded from '{persist_directory}'.")
            # If documents are provided, we should ensure they are added if they don't exist
            # This logic needs to be careful not to duplicate.
            # For simplicity, if documents are passed, we assume we want to add/update.
            # A real-world app might hash documents to avoid re-adding.
            if documents:
                print(f"Adding/updating {len(documents)} documents to the vector store...")
                vector_store.add_documents(documents)
                print("Documents added/updated.")
            return vector_store

        elif documents: # If force_recreate or no existing documents found, add them
            print(f"Creating/recreating vector store '{collection_name}' at '{persist_directory}' with {len(documents)} documents.")
            vector_store = Chroma.from_documents(
                documents=documents,
                embedding=embedding_model,
                persist_directory=persist_directory,
                collection_name=collection_name
            )
            print("Vector store created/recreated successfully.")
            return vector_store
        else: # No documents provided, just load existing or create empty
            print(f"Loading existing vector store '{collection_name}' from '{persist_directory}' (no new documents provided).")
            return vector_store

    except Exception as e:
        print(f"Error loading or creating vector store: {e}")
        print("Attempting to create a new vector store (this might happen if the collection is truly empty or corrupted).")
        # Fallback to creating a new one if loading fails
        if documents:
            vector_store = Chroma.from_documents(
                documents=documents,
                embedding=embedding_model,
                persist_directory=persist_directory,
                collection_name=collection_name
            )
            print("New vector store created successfully after error.")
            return vector_store
        else:
            print("Cannot create vector store without documents and no existing store found.")
            return None


def get_vector_store_retriever(
    embedding_model: Embeddings,
    persist_directory: str = CHROMA_DB_PATH,
    collection_name: str = CHROMA_COLLECTION_NAME,
    search_kwargs: dict = {"k": 4} # Default to retrieving 4 documents
):
    """
    Returns a retriever from the existing ChromaDB vector store.
    Assumes the vector store has already been populated.
    """
    try:
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_model,
            collection_name=collection_name
        )
        # Check if the collection is empty before returning retriever
        # This is a heuristic check, not foolproof.
        # A more robust check might involve trying to query.
        # For now, we'll just return the retriever, and issues will surface during RAG.
        print(f"Retriever created from vector store '{collection_name}'.")
        return vector_store.as_retriever(search_kwargs=search_kwargs)
    except Exception as e:
        print(f"Error getting vector store retriever: {e}")
        print("Ensure the vector store has been initialized and populated with documents.")
        return None

if __name__ == "__main__":
    # Example usage:
    from src.data_processor import get_processed_documents

    # 1. Get embedding model
    embedding_model = get_embedding_model()
    if not embedding_model:
        print("Exiting as embedding model could not be loaded.")
        exit()

    # 2. Get processed documents (chunks)
    # Ensure you have some dummy files in data/product_docs for this to work
    os.makedirs(os.path.join(CHROMA_DB_PATH, os.pardir), exist_ok=True) # Ensure vector_store dir exists
    os.makedirs(CHROMA_DB_PATH, exist_ok=True)

    # Create some dummy files for testing data_processor
    product_docs_test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'product_docs')
    os.makedirs(product_docs_test_dir, exist_ok=True)
    with open(os.path.join(product_docs_test_dir, "test_product_A.txt"), "w") as f:
        f.write("Product A is a high-performance gadget with long battery life. It costs $100. " * 5)
    with open(os.path.join(product_docs_test_dir, "test_product_B.txt"), "w") as f:
        f.write("Product B is a budget-friendly option, great for beginners. It costs $50. " * 5)

    documents = get_processed_documents()
    if not documents:
        print("No documents to process. Please add files to 'data/product_docs'.")
        exit()

    # 3. Create/load vector store (force recreate for clean test)
    vector_store = get_or_create_vector_store(documents, embedding_model, force_recreate=True)
    if not vector_store:
        print("Exiting as vector store could not be created/loaded.")
        exit()

    # 4. Get retriever
    retriever = get_vector_store_retriever(embedding_model)
    if not retriever:
        print("Exiting as retriever could not be created.")
        exit()

    # 5. Test retrieval
    print("\nTesting retrieval for 'battery life of product A':")
    retrieved_docs = retriever.invoke("What is the battery life of product A?")
    for i, doc in enumerate(retrieved_docs):
        print(f"\n--- Retrieved Document {i+1} ---")
        print(doc.page_content)
        print(f"Source: {doc.metadata.get('source', 'N/A')}")

    print("\nTesting retrieval for 'cost of budget option':")
    retrieved_docs = retriever.invoke("How much does the budget-friendly product cost?")
    for i, doc in enumerate(retrieved_docs):
        print(f"\n--- Retrieved Document {i+1} ---")
        print(doc.page_content)
        print(f"Source: {doc.metadata.get('source', 'N/A')}")

    # Clean up dummy files and directory after test
    # import shutil
    # shutil.rmtree(product_docs_test_dir)
    # shutil.rmtree(CHROMA_DB_PATH)

# src/embedding_model.py

from langchain_community.embeddings import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL_NAME

def get_embedding_model():
    """
    Initializes and returns a HuggingFaceEmbeddings model for text embedding.
    The model will be downloaded locally if not already present.
    """
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    # Using 'cpu' as device for broader compatibility, you can change to 'cuda' if you have a compatible GPU
    model_kwargs = {'device': 'cpu'}
    # Ensure the model is loaded from local files if available
    encode_kwargs = {'normalize_embeddings': False} # Normalization is often done by vector store
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        print("Embedding model loaded successfully.")
        return embeddings
    except Exception as e:
        print(f"Error loading embedding model {EMBEDDING_MODEL_NAME}: {e}")
        print("Please ensure you have an internet connection for the first download, or check model name.")
        return None

if __name__ == "__main__":
    # Example usage:
    embedding_model = get_embedding_model()
    if embedding_model:
        text = "This is a test sentence for embedding."
        vector = embedding_model.embed_query(text)
        print(f"Embedding for '{text}': {vector[:5]}...") # Print first 5 dimensions
        print(f"Vector dimension: {len(vector)}")

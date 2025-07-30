# src/rag_chain.py

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from langchain_core.documents import Document

from config import OLLAMA_BASE_URL, OLLAMA_MODEL_NAME, RAG_PROMPT_TEMPLATE, TOP_K_RETRIEVAL
from embedding_model import get_embedding_model
from vector_db_manager import get_or_create_vector_store, get_vector_store_retriever
from data_processor import get_processed_documents

def format_docs(docs: list[Document]) -> str:
    """
    Formats a list of documents into a single string for the LLM context.
    """
    return "\n\n".join(doc.page_content for doc in docs)

def initialize_rag_chain():
    """
    Initializes and returns the LangChain RAG chain.
    This function handles loading documents, creating/loading the vector store,
    and setting up the LLM and prompt.
    """
    print("Initializing RAG chain...")

    # 1. Get embedding model
    embedding_model = get_embedding_model()
    if not embedding_model:
        print("Failed to load embedding model. Exiting RAG chain initialization.")
        return None

    # 2. Get processed documents (chunks)
    # This step will load and chunk your product documents.
    # It will only run if the vector store needs to be created or recreated.
    processed_documents = get_processed_documents()
    if not processed_documents:
        print("No documents found or processed for the knowledge base. "
              "Please ensure your product documents are in 'data/product_docs'.")
        # We can still try to load an existing vector store if no new docs are provided
        # but the RAG chain might not be effective if the store is empty.

    # 3. Create or load the vector store
    # If the vector store already exists and is populated, it will be loaded.
    # If not, it will be created from `processed_documents`.
    # We pass processed_documents here so it can be used for initial creation.
    vector_store = get_or_create_vector_store(
        documents=processed_documents,
        embedding_model=embedding_model,
        force_recreate=False # Set to True if you want to rebuild the DB every time
    )
    if not vector_store:
        print("Failed to initialize vector store. Exiting RAG chain initialization.")
        return None

    # 4. Get the retriever from the vector store
    retriever = get_vector_store_retriever(
        embedding_model=embedding_model,
        search_kwargs={"k": TOP_K_RETRIEVAL}
    )
    if not retriever:
        print("Failed to create retriever. Exiting RAG chain initialization.")
        return None

    # 5. Initialize the local LLM (Ollama)
    try:
        llm = Ollama(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL_NAME)
        print(f"Ollama LLM '{OLLAMA_MODEL_NAME}' initialized.")
    except Exception as e:
        print(f"Error initializing Ollama LLM: {e}")
        print(f"Please ensure Ollama is running and model '{OLLAMA_MODEL_NAME}' is pulled.")
        return None

    # 6. Define the RAG prompt template
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

    # 7. Construct the RAG chain
    # The chain flow:
    # 1. User question comes in.
    # 2. Retriever finds relevant documents based on the question.
    # 3. Retrieved documents are formatted into a single context string.
    # 4. The context and original question are passed to the prompt template.
    # 5. The prompt is sent to the LLM.
    # 6. The LLM's response is parsed as a string.
    rag_chain = (
        {"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    print("RAG chain initialized successfully.")
    return rag_chain

if __name__ == "__main__":
    # Example usage:
    # Ensure Ollama is running and 'tinyllama' is pulled: ollama pull tinyllama
    # Ensure you have some dummy files in data/product_docs for vector store population

    rag_chain = initialize_rag_chain()
    if rag_chain:
        print("\nType 'exit' to quit the chatbot.")
        while True:
            user_query = input("You: ")
            if user_query.lower() == 'exit':
                break
            print("Chatbot: Thinking...")
            try:
                # Use .invoke() for a single synchronous call
                response = rag_chain.invoke(user_query)
                print(f"Chatbot: {response}")
            except Exception as e:
                print(f"An error occurred during response generation: {e}")
                print("Please check Ollama status and model availability.")
    else:
        print("RAG chain could not be initialized. Please check previous error messages.")

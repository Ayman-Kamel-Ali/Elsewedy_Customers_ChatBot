# src/chatbot_app.py

import streamlit as st
from rag_chain import initialize_rag_chain
import os
import time

# --- Streamlit UI Configuration ---
st.set_page_config(page_title="Product Info Chatbot (RAG)", layout="centered")

# --- Initialize RAG Chain (Cached to run only once) ---
@st.cache_resource
def setup_rag_chain():
    """
    Sets up the RAG chain and caches it to avoid re-initialization on every rerun.
    """
    with st.spinner("Initializing chatbot knowledge base and LLM... This may take a moment."):
        rag_chain = initialize_rag_chain()
    if rag_chain:
        st.success("Ask me anything about Elsewedy products.")
    else:
        st.error("Failed to initialize chatbot. Please check the console for errors and ensure Ollama is running with 'tinyllama' pulled.")
    return rag_chain

rag_chain = setup_rag_chain()

# --- Chatbot Interface ---
st.title("Elsewedy Electric ⚡")
st.markdown("At Your Service")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What can I help you with?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get chatbot response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        if rag_chain:
            try:
                # Use .stream() for a more interactive, streaming response
                # Note: Ollama's streaming might be less granular than OpenAI's
                # For tinyllama, it might still give chunks quickly.
                for chunk in rag_chain.stream(prompt):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌") # Add blinking cursor effect
                message_placeholder.markdown(full_response)
            except Exception as e:
                full_response = f"An error occurred: {e}. Please check the console and ensure Ollama is running."
                st.error(full_response)
                print(f"Error during RAG chain invocation: {e}")
        else:
            full_response = "Chatbot is not initialized. Please check the setup messages above."
        st.session_state.messages.append({"role": "assistant", "content": full_response})

# Optional: Add a button to clear chat history
if st.button("Clear Chat"):
    st.session_state.messages = []
    st.experimental_rerun()

# st.sidebar.header("Setup Instructions")
# st.sidebar.markdown("""
# 1.  **Place Documents:** Put your product `.pdf`, `.txt`, or `.md` files into the `data/product_docs/` folder.
# 2.  **Run Ollama:** Ensure Ollama is running and you have pulled the `tinyllama` model (`ollama pull tinyllama`).
# 3.  **Run this app:** In your terminal, navigate to the `chatbot_rag_project` directory and run:
#     `streamlit run src/chatbot_app.py`
# """)

# st.sidebar.info("The chatbot will build its knowledge base from your documents on the first run. This might take a few moments.")

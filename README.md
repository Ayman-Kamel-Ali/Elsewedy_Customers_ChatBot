Local RAG Chatbot for Product Information
This project implements a local Retrieval Augmented Generation (RAG) chatbot designed to provide customers with information about a company's products. It leverages open-source tools to run entirely on your local machine, ensuring data privacy and reducing reliance on external APIs.

Features
Local Execution: All components (LLM, vector database, embeddings) run on your local machine.

Product Knowledge Base: Ingests your product documentation (PDF, TXT, Markdown) to answer specific questions.

Retrieval Augmented Generation: Enhances LLM responses with relevant information retrieved from your documents.

Streamlit UI: Simple and interactive web interface for chatting with the bot.

Technologies Used
LangChain: For orchestrating the RAG pipeline.

Sentence Transformers: For generating text embeddings.

ChromaDB: A lightweight, local-first vector database.

Ollama: For running open-source Large Language Models (LLMs) locally (specifically tinyllama).

Streamlit: For building the web-based chatbot interface.

PyPDF2 / Unstructured: For document loading and parsing.

Project Structure
chatbot_rag_project/
├── data/
│   └── product_docs/
│       └── # Your product documents go here (PDF, TXT, MD)
├── vector_store/
│   └── chroma_db/
│       └── # ChromaDB will store its database files here
├── src/
│   ├── data_processor.py      # Loads and chunks documents
│   ├── embedding_model.py     # Initializes the embedding model
│   ├── vector_db_manager.py   # Manages ChromaDB interactions
│   ├── rag_chain.py           # Orchestrates the RAG pipeline
│   └── chatbot_app.py         # Streamlit application
├── config.py                  # Project configuration variables
├── requirements.txt           # Python dependencies
└── README.md                  # This file

Setup Instructions
Follow these steps to get the chatbot up and running on your local machine.

1. Clone the Project (or create directories/files)
First, create the main project directory and the subdirectories:

mkdir chatbot_rag_project
cd chatbot_rag_project
mkdir data
mkdir data/product_docs
mkdir vector_store
mkdir vector_store/chroma_db
mkdir src

Then, create the files listed above (requirements.txt, config.py, src/__init__.py, etc.) and paste the provided content into them.

2. Install Python Dependencies
It's highly recommended to use a virtual environment to manage your Python dependencies.

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
# venv\Scripts\activate
# On macOS/Linux:
# source venv/bin/activate

# Install the required packages
pip install -r requirements.txt

3. Install and Run Ollama
Ollama is essential for running the local LLM.

Download Ollama: Go to ollama.com and download the installer for your operating system.

Install Ollama: Follow the installation instructions.

Pull the tinyllama model: Open your terminal or command prompt and run:

ollama pull tinyllama

Ensure Ollama is running in the background before starting the chatbot.

4. Add Your Product Documents
Place your company's product information documents (e.g., product_manual.pdf, faq.txt, specs.md) into the data/product_docs/ directory. The chatbot will use these files to build its knowledge base.

5. Run the Chatbot
Once all the above steps are completed, you can run the Streamlit application.

# Make sure your virtual environment is activated
# Navigate to the root of your project directory (chatbot_rag_project)
streamlit run src/chatbot_app.py

This command will open a new tab in your web browser with the Streamlit chatbot interface.

How it Works
Initialization (First Run):

When you first run chatbot_app.py, it will load documents from data/product_docs/.

These documents are then split into smaller chunks.

An embedding model (Sentence Transformers) converts these chunks into numerical vectors (embeddings).

These embeddings and their corresponding text chunks are stored in ChromaDB within the vector_store/chroma_db/ directory. This process creates your local knowledge base.

The tinyllama LLM is initialized via Ollama.

Chat Interaction:

When you type a query, the query is also converted into an embedding.

ChromaDB performs a similarity search to find the most relevant product information chunks from your knowledge base.

These relevant chunks are then passed as context to the tinyllama LLM along with your original query.

The LLM generates an answer based only on the provided context, reducing "hallucinations" and ensuring accuracy.

Troubleshooting
"Failed to initialize chatbot...":

Ensure Ollama is running in the background.

Verify you have pulled the tinyllama model (ollama pull tinyllama).

Check your config.py for correct OLLAMA_BASE_URL and OLLAMA_MODEL_NAME.

Review the console output for more detailed error messages.

"No documents found...":

Make sure you have placed your product documents (.pdf, .txt, .md) inside the data/product_docs/ folder.

Slow performance:

tinyllama is small, but embedding generation can be CPU-intensive. The first run will be slower as it builds the vector store.

Consider using a GPU if available and configuring your embedding model/Ollama to utilize it.

Inaccurate answers:

The quality of answers heavily depends on the quality and comprehensiveness of your product documents.

Adjust CHUNK_SIZE and CHUNK_OVERLAP in config.py to optimize how documents are chunked.

Adjust TOP_K_RETRIEVAL in config.py to retrieve more or fewer context documents.
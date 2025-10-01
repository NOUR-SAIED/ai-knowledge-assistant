import streamlit as st
import chromadb
import os
import requests
import json
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

# --- 1. CONFIGURATION & SETUP ---
st.set_page_config(layout="wide")
st.title("🤖 AI Knowledge Assistant (Ollama Version)")

# --- Point to the Ollama database ---
DB_PATH_OLLAMA = 'chroma_db_ollama'
COLLECTION_NAME = "confluence_docs_ollama"
OLLAMA_EMBED_MODEL = "nomic-embed-text"
OLLAMA_GENERATE_MODEL = "mistral"

# --- 2. INITIALIZE SESSION STATE ---
# This is the app's "long-term memory".
# We will store the chat history here.
# This part only runs once, at the very beginning of the session.
if "messages" not in st.session_state:
    st.session_state.messages = []


# --- 3. LOAD MODELS AND DATABASE (Cached) ---
@st.cache_resource
def load_db_collection():
    print("--- Loading ChromaDB client for the Ollama App ---")
    if not os.path.exists(DB_PATH_OLLAMA):
        return None  # Return None if the database doesn't exist

    client = chromadb.PersistentClient(path=DB_PATH_OLLAMA)

    ollama_ef = OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name=OLLAMA_EMBED_MODEL,
    )

    collection = client.get_collection(name=COLLECTION_NAME, embedding_function=ollama_ef)
    print("--- ChromaDB client loaded successfully ---")
    return collection


# Load the collection from the cache
collection = load_db_collection()


# --- 4. CORE RAG FUNCTION (No changes here) ---
def get_ollama_rag_response(query, collection, n_results=3):  # Using n_results=3 as we found it's better
    results = collection.query(query_texts=[query], n_results=n_results)

    retrieved_docs = results['documents'][0]
    context = "\n\n---\n\n".join(retrieved_docs)

    prompt_template = f"""
    [INST]
    You are an expert technical assistant. Your task is to answer the user's question based ONLY on the following context.
    If the context does not contain the answer, state that you cannot find the information in the provided documents. Be concise.

    CONTEXT:
    {context}
    ---
    QUESTION:
    {query}
    [/INST]
    """

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": OLLAMA_GENERATE_MODEL, "prompt": prompt_template, "stream": False},
            timeout=60  # Add a timeout
        )
        response.raise_for_status()
        response_data = response.json()
        final_answer = response_data.get("response", "Error: No response field found.").strip()
    except requests.exceptions.RequestException as e:
        final_answer = f"Error: Could not connect to Ollama. Please ensure it's running. Details: {e}"

    return final_answer


# --- 5. THE MAIN APP LOGIC ---

# First, check if the database exists. If not, show an error and stop.
if collection is None:
    st.error(f"Database not found at '{DB_PATH_OLLAMA}'. Please run the `build_database_ollama.py` script first.")
else:
    # Display the chat history from st.session_state
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Use st.chat_input for the user's question. This is the new widget.
    # It returns the user's text when they press Enter, otherwise it's None.
    if prompt := st.chat_input("Ask a question about your documents..."):
        # 1. Add the user's message to the chat history and display it
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Get the AI's response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = get_ollama_rag_response(prompt, collection)
                st.markdown(response)

        # 3. Add the AI's response to the chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
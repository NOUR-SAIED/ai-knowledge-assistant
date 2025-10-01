import streamlit as st
import chromadb
import os
# NOTICE: We are NOT importing or using SentenceTransformer in this final version.
from ctransformers import AutoModelForCausalLM

# --- 1. CONFIGURATION & SETUP ---
st.set_page_config(layout="wide")
DB_PATH = 'chroma_db'
COLLECTION_NAME = "confluence_docs"


@st.cache_resource
def load_models_and_db():
    print("--- Loading models and DB client for the App ---")

    local_model_path = "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    if os.path.exists(local_model_path):
        model_path_or_repo_id = local_model_path
        model_file = None
    else:
        model_path_or_repo_id = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
        model_file = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"

    llm = AutoModelForCausalLM.from_pretrained(
        model_path_or_repo_id=model_path_or_repo_id,
        model_file=model_file,
        model_type="mistral",
        gpu_layers=0,
        context_length=4096
    )

    client = chromadb.PersistentClient(path=DB_PATH)
    # The collection will use its default embedding function for queries, matching the database.
    collection = client.get_collection(name=COLLECTION_NAME)

    print("--- Models and DB client loaded successfully ---")
    return llm, collection


# --- 2. CORE RAG FUNCTION ---
def get_rag_response(query, n_results=3):
    # The collection object knows how to turn the query text into an embedding itself.
    results = collection.query(query_texts=[query], n_results=n_results)
    retrieved_docs = results['documents'][0]
    retrieved_metadata = results['metadatas'][0]
    context = "\n\n---\n\n".join(retrieved_docs)
    sources = list(set([meta['source_file'] for meta in retrieved_metadata]))

    prompt_template = f"""
    [INST]
    You are an expert technical assistant. Your task is to answer the user's question.
    - Use ONLY the information from the CONTEXT below.
    - Do not combine information from different topics or services if they seem unrelated.
    - If the context contains information from multiple different documents, prioritize the information that seems most directly related to the user's specific question.
    - Be concise and accurate.
    CONTEXT:
    {context}
    ---
    QUESTION:
    {query}
    [/INST]
    """
    response = llm(prompt_template, max_new_tokens=512, temperature=0.1)
    final_answer = response.strip()
    return final_answer, context, sources


# --- 3. STREAMLIT UI ---
st.title("🤖 AI Conversational Knowledge Assistant")
st.markdown("Query your Confluence documentation using natural language.")

if not os.path.exists(DB_PATH):
    st.error("Database not found. Please run the `build_database.py` script from your terminal.")
else:
    try:
        llm, collection = load_models_and_db()
        st.divider()
        query = st.text_input("Ask a question about your documents:")
        if query:
            with st.spinner("Searching for answers..."):
                final_answer, context, sources = get_rag_response(query)
                st.markdown("### Answer")
                st.write(final_answer)
                st.markdown("---")
                st.markdown("### Sources")
                for source in sources:
                    st.markdown(f"- `{source}`")
                with st.expander("Show Retrieved Context"):
                    st.text(context)
    except Exception as e:
        st.error(f"An error occurred: The database might be empty or corrupt. Try rebuilding it. Error: {e}")
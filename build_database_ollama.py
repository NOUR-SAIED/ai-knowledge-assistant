import os
import glob
import time
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

# --- CONFIGURATION ---
BASE_FOLDER_PATH = 'data'
# NEW: Create a separate database folder for the Ollama version to keep it isolated
DB_PATH_OLLAMA = 'chroma_db_ollama'
COLLECTION_NAME = "confluence_docs_ollama"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
OLLAMA_EMBED_MODEL = "nomic-embed-text"


def load_and_clean_document(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        soup = BeautifulSoup(html_content, 'html.parser')
        title = soup.title.string.strip() if soup.title else os.path.basename(file_path)
        content_chest = soup.find(id='main-content', class_='wiki-content group')
        clean_text = content_chest.get_text(separator=' ', strip=True) if content_chest else soup.get_text(
            separator=' ', strip=True)
        return {'title': title, 'text': clean_text, 'source_file': os.path.basename(file_path)}
    except Exception as e:
        print(f"    - WARNING: Error processing file {file_path}: {e}")
        return None


def main():
    print("--- Starting Database Build Process (Using Ollama Embedder) ---")

    print("1. Setting up ChromaDB client...")
    if os.path.exists(DB_PATH_OLLAMA):
        print(f"   - Existing database folder '{DB_PATH_OLLAMA}' found. Please delete it before running.")
        return

    client = chromadb.PersistentClient(path=DB_PATH_OLLAMA)

    ollama_ef = OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name=OLLAMA_EMBED_MODEL,
    )

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=ollama_ef
    )
    print("   ChromaDB client ready and configured to use Ollama.")

    print("2. Finding HTML files...")
    html_files = glob.glob(os.path.join(BASE_FOLDER_PATH, '**', '*.html'), recursive=True)
    if not html_files:
        print("   - ERROR: No HTML files found in the 'data' folder.")
        return
    print(f"   Found {len(html_files)} files.")

    print("3. Starting file processing and indexing loop...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    total_chunks_added = 0
    chunk_id_counter = 0

    start_time = time.time()
    for i, file_path in enumerate(html_files):
        print(f"\n--- Processing file {i + 1}/{len(html_files)}: {os.path.basename(file_path)} ---")
        doc = load_and_clean_document(file_path)
        if not doc or len(doc['text']) < 100:
            print("    - SKIPPING (Not enough content)")
            continue

        chunks = text_splitter.split_text(doc['text'])
        print(f"    - Created {len(chunks)} chunks.")

        if chunks:
            try:
                ids = [f"chunk_{chunk_id_counter + j}" for j in range(len(chunks))]
                chunk_id_counter += len(chunks)
                metadatas = [{'title': doc['title'], 'source_file': doc['source_file']} for _ in chunks]

                collection.add(documents=chunks, metadatas=metadatas, ids=ids)
                total_chunks_added += len(chunks)
                print(f"    - Successfully added {len(chunks)} chunks to the database via Ollama.")
            except Exception as e:
                print(f"    - CRITICAL ERROR during adding for this file. SKIPPING. Error: {e}")

    end_time = time.time()
    print("\n--- Database Build Process Finished ---")
    print(f"Total time taken: {end_time - start_time:.2f} seconds")
    print(f"Total chunks added to the database: {total_chunks_added}")


if __name__ == "__main__":
    main()
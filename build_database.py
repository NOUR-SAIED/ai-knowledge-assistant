import os
import glob
import time
import onnxruntime
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
# CHANGE: We no longer import SentenceTransformer
# from sentence_transformers import SentenceTransformer
import chromadb


# --- CONFIGURATION ---
BASE_FOLDER_PATH = 'data'
DB_PATH = 'chroma_db'
COLLECTION_NAME = "confluence_docs"
# CHANGE: We no longer need the model name constant
# EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


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
    print("--- Starting Database Build Process (Using ChromaDB's Stable Default Embedder) ---")

    # CHANGE: The step to load the SentenceTransformer model is removed.
    # print("1. Loading embedding model...")
    # embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device='cpu')
    # print("   Model loaded successfully.")

    print("1. Setting up ChromaDB client...")
    if os.path.exists(DB_PATH):
        print("   - Existing database folder found. Please delete it before running to ensure a fresh build.")
        return

    client = chromadb.PersistentClient(path=DB_PATH)
    # This now tells ChromaDB to prepare its default embedding function.
    collection = client.create_collection(name=COLLECTION_NAME)
    print("   ChromaDB client ready.")

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

                # --- THE FIX ---
                # The line `embeddings = embedding_model.encode(chunks).tolist()` is removed.
                # We now pass the text documents directly to the collection.
                # ChromaDB will use its stable internal function to create the embeddings.
                collection.add(
                    documents=chunks,
                    metadatas=metadatas,
                    ids=ids
                )
                total_chunks_added += len(chunks)
                print(f"    - Successfully added {len(chunks)} chunks to the database.")
            except Exception as e:
                # If this fails, it will now be a clear Python error we can read and fix.
                print(f"    - CRITICAL ERROR during embedding/adding for this file. SKIPPING. Error: {e}")

    end_time = time.time()
    print("\n--- Database Build Process Finished ---")
    print(f"Total time taken: {end_time - start_time:.2f} seconds")
    print(f"Total chunks added to the database: {total_chunks_added}")


if __name__ == "__main__":
    main()
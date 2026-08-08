import os
from sentence_transformers import SentenceTransformer
import chromadb

# resolve paths relative to this file so it works from any working directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
policy_docs_path = os.path.join(BASE_DIR, "docs")
vector_db_path = os.path.join(BASE_DIR, "chroma_db")

# load the embedding model - all-MiniLM-L6-v2 runs offline and is fast
sentence_encoder = SentenceTransformer("all-MiniLM-L6-v2")

# connect to the persistent vector store
vector_store_client = chromadb.PersistentClient(path=vector_db_path)

# drop old data so we never build up duplicates on restart
try:
    vector_store_client.delete_collection("zepto_policies")
except Exception:
    pass

knowledge_base = vector_store_client.get_or_create_collection("zepto_policies")

# read every txt file in the docs folder, embed it, and store it
ingested_count = 0
for root_dir, sub_dirs, file_list in os.walk(policy_docs_path):
    for doc_filename in sorted(file_list):
        if doc_filename.endswith(".txt"):
            doc_path = os.path.join(root_dir, doc_filename)
            print(f"Ingesting: {doc_path}")
            with open(doc_path, "r", encoding="utf-8") as fh:
                doc_text = fh.read()
                doc_vector = sentence_encoder.encode(doc_text)
                ingested_count += 1
                knowledge_base.add(
                    embeddings=[doc_vector.tolist()],
                    documents=[doc_text],
                    metadatas=[{"source": doc_filename}],
                    ids=[str(ingested_count)]
                )

print(f"Loaded {ingested_count} policy documents into the vector store")

# show a short preview of every stored document to confirm ingestion worked
for meta_item, doc_item in zip(
    knowledge_base.get(include=["metadatas"])["metadatas"],
    knowledge_base.get(include=["documents"])["documents"]
):
    print(f"Stored: {meta_item['source']} | {doc_item[:80]}")

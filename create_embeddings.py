import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# 1. File locations
# --------------------------------------------------

CHUNKS_FILE = "IP-SAKTI-RAG-Dataset/chunks_with_pages.jsonl"

INDEX_FILE = "IP-SAKTI-RAG-Dataset/faiss_index.bin"

METADATA_FILE = "IP-SAKTI-RAG-Dataset/chunks_metadata.json"


# --------------------------------------------------
# 2. Load the prepared chunks
# --------------------------------------------------

chunks = []

with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
    for line in file:
        chunks.append(json.loads(line))


print(f"Loaded {len(chunks)} chunks.")


# --------------------------------------------------
# 3. Extract the text from every chunk
# --------------------------------------------------

texts = [chunk["text"] for chunk in chunks]


# --------------------------------------------------
# 4. Load multilingual embedding model
# --------------------------------------------------

print("Loading multilingual embedding model...")

model = SentenceTransformer(
    "paraphrase-multilingual-MiniLM-L12-v2"
)


# --------------------------------------------------
# 5. Convert document chunks into embeddings
# --------------------------------------------------

print("Creating embeddings...")

embeddings = model.encode(
    texts,
    show_progress_bar=True,
    normalize_embeddings=True
)


embeddings = np.array(embeddings, dtype="float32")


# --------------------------------------------------
# 6. Create FAISS index
# --------------------------------------------------

print("Creating FAISS index...")

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(embeddings)


# --------------------------------------------------
# 7. Save FAISS index
# --------------------------------------------------

faiss.write_index(index, INDEX_FILE)


# --------------------------------------------------
# 8. Save chunk metadata
# --------------------------------------------------

with open(METADATA_FILE, "w", encoding="utf-8") as file:
    json.dump(
        chunks,
        file,
        ensure_ascii=False,
        indent=2
    )


print()
print("======================================")
print("RAG EMBEDDING DATABASE CREATED!")
print("======================================")
print(f"Total chunks: {len(chunks)}")
print(f"Embedding dimensions: {dimension}")
print(f"FAISS index: {INDEX_FILE}")
print(f"Metadata: {METADATA_FILE}")
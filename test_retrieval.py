import json
import faiss
from sentence_transformers import SentenceTransformer


# -----------------------------------------
# 1. Load FAISS index
# -----------------------------------------

INDEX_FILE = "IP-SAKTI-RAG-Dataset/faiss_index.bin"
METADATA_FILE = "IP-SAKTI-RAG-Dataset/chunks_metadata.json"

index = faiss.read_index(INDEX_FILE)

with open(METADATA_FILE, "r", encoding="utf-8") as file:
    chunks = json.load(file)


# -----------------------------------------
# 2. Load the same embedding model
# -----------------------------------------

model = SentenceTransformer(
    "paraphrase-multilingual-MiniLM-L12-v2"
)


# -----------------------------------------
# 3. Ask a question
# -----------------------------------------

question = input("\nAsk IP-SAKTI a question: ")


# -----------------------------------------
# 4. Convert question into an embedding
# -----------------------------------------

query_embedding = model.encode(
    [question],
    normalize_embeddings=True
)


# -----------------------------------------
# 5. Search FAISS
# -----------------------------------------

scores, indices = index.search(
    query_embedding,
    5
)


# -----------------------------------------
# 6. Display the top 5 results
# -----------------------------------------

print("\n" + "=" * 60)
print("TOP RELEVANT SOURCES")
print("=" * 60)

for rank, (score, index_number) in enumerate(
    zip(scores[0], indices[0]),
    start=1
):

    chunk = chunks[index_number]

    print(f"\n--- Result {rank} ---")
    print(f"Similarity score: {score:.4f}")
    print(f"Source: {chunk.get('source', 'Unknown')}")
    print(f"Page: {chunk.get('page', 'Unknown')}")
    print("\nText:")
    print(chunk["text"][:1000])
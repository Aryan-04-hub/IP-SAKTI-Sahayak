from pathlib import Path
import json
import os

import faiss
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "IP-SAKTI-RAG-Dataset"
INDEX_FILE = DATASET_DIR / "faiss_index.bin"
METADATA_FILE = DATASET_DIR / "chunks_metadata.json"

app = FastAPI(title="IP-SAKTI Sahayak API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading FAISS index...")
index = faiss.read_index(str(INDEX_FILE))

with open(METADATA_FILE, "r", encoding="utf-8") as file:
    chunks = json.load(file)

print(f"Loaded {len(chunks)} chunks.")

print("Loading multilingual embedding model...")
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
print("Embedding model loaded.")

# Gemini via Google's OpenAI-compatible API.
# Set GEMINI_API_KEY in PowerShell before starting the server.
gemini_api_key = os.getenv("GEMINI_API_KEY")
gemini_model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

if gemini_api_key:
    client = OpenAI(
        api_key=gemini_api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        timeout=60.0,
        max_retries=0,
    )
    print("Gemini API configured.")
    print(f"LLM model: {gemini_model}")
else:
    client = None
    print("WARNING: GEMINI_API_KEY not configured.")
    print("RAG retrieval will still work.")


class QuestionRequest(BaseModel):
    question: str


@app.get("/")
def home():
    return {
        "message": "IP-SAKTI Sahayak backend is running!",
        "rag_chunks": len(chunks),
        "llm_configured": client is not None,
        "llm_provider": "Google Gemini",
        "llm_model": gemini_model,
    }


@app.post("/ask")
def ask_question(request: QuestionRequest):
    question = request.question.strip()

    if not question:
        return {
            "answer": "Please enter a question.",
            "sources": [],
        }

    # Create query embedding.
    query_embedding = model.encode(
        [question],
        normalize_embeddings=True,
    )

    # Retrieve more candidates so relevant regulatory passages are less likely
    # to be missed during this diagnostic stage.
    RETRIEVAL_K = 15
    scores, indices = index.search(query_embedding, RETRIEVAL_K)

    # Diagnostic output: inspect all 15 retrieved candidates in the terminal.
    print("\n" + "=" * 70)
    print("RAG RETRIEVAL RESULTS")
    print(f"Question: {question}")
    print("=" * 70)

    for rank, (score, index_number) in enumerate(
        zip(scores[0], indices[0]), start=1
    ):
        if index_number < 0:
            continue

        chunk = chunks[index_number]
        source = chunk.get("source", "Unknown")
        page = chunk.get("page", "Unknown")

        print(
            f"{rank:02d}. score={float(score):.4f} | "
            f"source={source} | page={page}",
            flush=True,
        )

    print("=" * 70 + "\n")

    sources = []
    context_parts = []

    for rank, (score, index_number) in enumerate(
        zip(scores[0], indices[0]), start=1
    ):
        if index_number < 0:
            continue

        chunk = chunks[index_number]
        source = chunk.get("source", "Unknown")
        page = chunk.get("page", "Unknown")
        text = chunk.get("text", "").strip()

        context_parts.append(
            f"[Source {rank}]\n"
            f"Document: {source}\n"
            f"Page: {page}\n"
            f"Content:\n{text}"
        )

        sources.append({
            "rank": rank,
            "score": round(float(score), 4),
            "source": source,
            "page": page,
        })

    context = "\n\n".join(context_parts)

    if client:
        prompt = f"""
You are IP-SAKTI Sahayak.

You provide information about Intellectual Property, Ayurveda,
traditional knowledge and related regulatory guidance.

IMPORTANT RULES:

1. Answer ONLY using the retrieved sources below.
2. Do NOT use outside knowledge to fill missing information.
3. Do NOT invent laws, regulations, sections, dates, authorities,
   cases or sources.
4. If the retrieved sources do not contain enough information to
   answer the question, clearly say that the available IP-SAKTI
   knowledge base does not contain enough information.
5. If the sources contain relevant information, answer the question
   directly and explain the relevant points.
6. When making a factual claim, mention the relevant source using
   [Source 1], [Source 2], etc.
7. Do not pretend that a source says something when it does not.
8. Keep the answer concise, clear and easy to understand.
9. This is an educational information system and not legal or
   medical advice.

USER QUESTION:
{question}

RETRIEVED SOURCES:
{context}
"""

        try:
            print("======================================", flush=True)
            print("STARTING GEMINI API REQUEST", flush=True)
            print(f"Model: {gemini_model}", flush=True)
            print("======================================", flush=True)

            response = client.chat.completions.create(
                model=gemini_model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
            )

            print("GEMINI API REQUEST SUCCEEDED", flush=True)

            answer = response.choices[0].message.content.strip()

            print("ANSWER GENERATED SUCCESSFULLY", flush=True)

        except BaseException as error:
            print("======================================", flush=True)
            print("GEMINI API REQUEST FAILED", flush=True)
            print("ERROR TYPE:", type(error).__name__, flush=True)
            print("ERROR:", str(error), flush=True)
            print("ERROR REPR:", repr(error), flush=True)
            print("======================================", flush=True)

            answer = (
                "The RAG retrieval worked, but the Gemini answer-generation "
                "request failed. Please check the FastAPI terminal for the "
                "exact API error."
            )
    else:
        answer = (
            "Relevant information was retrieved from the IP-SAKTI knowledge "
            "base. AI answer generation is not configured yet."
        )

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
    }

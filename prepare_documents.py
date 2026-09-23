import json
import re
from pathlib import Path

import fitz  # PyMuPDF


# --------------------------------------------------
# Folders
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DOCUMENTS_DIR = BASE_DIR / "documents"
OUTPUT_FILE = Path(__file__).resolve().parent / "chunks_with_pages.jsonl"


# --------------------------------------------------
# Settings
# --------------------------------------------------

CHUNK_SIZE = 700
CHUNK_OVERLAP = 100


# --------------------------------------------------
# Simple text cleaning
# --------------------------------------------------

def clean_text(text):
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# --------------------------------------------------
# Create chunks while keeping page number
# --------------------------------------------------

def make_chunks(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()

    if not words:
        return []

    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))

        chunk = " ".join(words[start:end])

        if chunk.strip():
            chunks.append(chunk.strip())

        if end >= len(words):
            break

        start = end - overlap

    return chunks


# --------------------------------------------------
# Process PDFs
# --------------------------------------------------

def main():

    pdf_files = sorted(DOCUMENTS_DIR.glob("*.pdf"))

    if not pdf_files:
        print("ERROR: No PDF files found in the documents folder.")
        return

    total_chunks = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as output:

        for pdf_path in pdf_files:

            print(f"\nProcessing: {pdf_path.name}")

            document = fitz.open(pdf_path)

            for page_number, page in enumerate(document, start=1):

                text = page.get_text("text")
                text = clean_text(text)

                if not text:
                    continue

                page_chunks = make_chunks(text)

                for chunk_number, chunk_text in enumerate(
                    page_chunks,
                    start=1
                ):

                    record = {
                        "chunk_id": f"{pdf_path.stem}_p{page_number}_c{chunk_number}",
                        "source": pdf_path.name,
                        "page": page_number,
                        "text": chunk_text
                    }

                    output.write(
                        json.dumps(
                            record,
                            ensure_ascii=False
                        ) + "\n"
                    )

                    total_chunks += 1

            document.close()

    print("\n======================================")
    print("DATASET WITH PAGE NUMBERS CREATED!")
    print("======================================")
    print(f"Total chunks: {total_chunks}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
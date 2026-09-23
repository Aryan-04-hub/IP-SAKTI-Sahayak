IP-SAKTI RAG DATASET
====================

Files:
- chunks.jsonl: one RAG chunk per line, with source/page/language/authority/category metadata.
- document_manifest.json: document-level statistics and pages that may need OCR.
- prepare_documents.py: reusable ingestion script.

IMPORTANT:
1. Hindi text is kept in Hindi. Do not translate everything to English.
2. Use a multilingual embedding model for cross-language retrieval.
3. Pages flagged in document_manifest.json should be checked before indexing.
4. Every answer should cite source filename and page.
5. This dataset is source material; it does not itself create the LLM.

Recommended embedding model:
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

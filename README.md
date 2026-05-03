# RAG Pipeline Guide

This repository includes a grounded retrieval-augmented generation pipeline for Nebraska groundwater policy questions. The current implementation uses PostgreSQL with pgvector, Gemini-based embeddings, and a FastAPI controller exposed through the main backend app.

## How it works

1. Source documents are fetched and normalized.
2. Documents are chunked with Gemini-assisted logicaAl chunking.
3. Chunks are embedded with Gemini embeddings and stored in PostgreSQL.
4. User questions are embedded and matched against the chunk store.
5. Retrieved passages are used to build a grounded answer prompt.
6. The backend returns either a RAG answer, a comparison response, or a health/index summary.

## Current components

- `rag_pipeline/src/ingest/logical_chunk_gemini.py` — logical chunking for source text
- `rag_pipeline/src/ingest/embed_gemini.py` — Gemini embeddings with 1536-dimensional vectors
- `rag_pipeline/src/ingest/retrieve_gemini.py` — retrieval, prompting, and answer generation
- `rag_pipeline/src/controller/rag_controller.py` — FastAPI routes for health, reindex, ask, and compare
- `rag_pipeline/sql/002_tables.sql` — schema for `source_docs` and `doc_chunks`
- `scripts/init_db.py` — schema creation and migration helper
- `rag_pipeline/src/key_utils.py` — Gemini API key resolution helper

## Data source assumptions

The pipeline is designed for official Nebraska groundwater documents and related policy text. It is intended to return answers grounded in retrieved source passages rather than free-form speculation.

## Running the pipeline

### Option 1: Docker

```bash
docker compose up --build
```

This starts PostgreSQL, the backend API, and the Angular UI.

### Option 2: Manual backend run

1. Start PostgreSQL with pgvector.
2. Set `DATABASE_URL` or the individual `POSTGRES_*` variables.
3. Run:

```bash
python scripts/init_db.py
uvicorn main:app --reload --port 8000
```

4. Open the Angular app and use the RAG Assistant page at `/rag-assistant`.

## API endpoints

- `GET /rag/health` — returns indexed document and chunk counts
- `GET /rag/sources` — lists the stored source documents
- `POST /rag/reindex` — refreshes the document store and embeddings
- `POST /rag/ask` — grounded response for a single question
- `POST /rag/compare` — RAG answer versus general Gemini answer

## Storage details

- `source_docs` stores document metadata and provenance
- `doc_chunks` stores chunk text, chunk order, and embeddings
- Embeddings now use `vector(1536)` to match the current Gemini embedding configuration

## Important notes

- The old standalone `rag_pipeline/src/main.py` entrypoint is deprecated.
- The backend entrypoint is the root-level `main.py`.
- If embeddings or reindexing fail, verify the Gemini API key and the PostgreSQL connection string first.


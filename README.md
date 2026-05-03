# RAG Pipeline Guide

This repository includes a grounded retrieval-augmented generation pipeline for Nebraska groundwater policy questions. The current implementation uses PostgreSQL with pgvector, Gemini-based embeddings, and a FastAPI controller exposed through the main backend app.


## How It Works

1. Source documents are fetched and normalized.
2. Documents are chunked with Gemini-assisted logical chunking (window defaults: ~8,000 chars, overlap ~500 chars).
3. Chunks are embedded with Gemini embeddings (`gemini-embedding-001`, 1,536 dimensions) and stored in PostgreSQL with pgvector.
4. User questions are embedded and matched against the chunk store using cosine similarity.
5. Multi-query expansion generates variant phrasings for broader retrieval coverage.
6. Retrieved passages are used to build a grounded answer prompt sent to Gemini Flash.
7. The backend returns either a RAG answer, a comparison response, or a health/index summary.


## Current Components

**Ingestion Pipeline**

- `rag_pipeline/src/ingest/logical_chunk_gemini.py` : Gemini Flash-assisted logical chunking that preserves legal hierarchy, section numbers, and clause boundaries
- `rag_pipeline/src/ingest/embed_gemini.py` : Gemini embeddings with 1,536-dimensional vectors via `gemini-embedding-001`
- `rag_pipeline/src/ingest/retrieve_gemini.py` : Multi-query retrieval, prompt building, weak evidence detection, and Gemini-based answer generation

**Backend Controller**

- `rag_pipeline/src/controller/rag_controller.py` : FastAPI router that exposes all RAG endpoints (health, sources, reindex, ask, compare)

**Database**

- `rag_pipeline/sql/002_tables.sql` : Schema for `source_docs` and `doc_chunks` tables
- `scripts/init_db.py` : Schema creation and migration helper

**Utilities**

- `rag_pipeline/src/key_utils.py` : Gemini API key resolution helper

**Frontend**

- `rag_pipeline/frontend/index.html` : Static HTML/CSS/JS interface for the Groundwater Policy Assistant (served separately on port 3000)


## Data Source Assumptions

The pipeline is designed for official Nebraska groundwater documents and related policy text from the Nebraska Legislature, DNR, and CPNRD. It currently indexes 12 regulatory documents split into approximately 1,740 semantically coherent chunks. The system is intended to return answers grounded in retrieved source passages rather than free-form speculation.


## Running the Pipeline

### Option 1: Docker

```bash
docker compose up --build
```

This starts PostgreSQL with pgvector, the backend API, and the Angular UI.

### Option 2: Manual Backend and Frontend Setup

This option requires two terminal windows running simultaneously.

**Terminal 1: Start the backend API**

1. Make sure PostgreSQL with pgvector is running.
2. Set `DATABASE_URL` or the individual `POSTGRES_*` environment variables.
3. Run the database setup and start the API server:

```bash
python scripts/init_db.py
uvicorn main:app --reload --port 8000
```

The backend will be available at `http://127.0.0.1:8000`. The Swagger interactive API docs will be at `http://127.0.0.1:8000/docs`.

**Terminal 2: Start the static frontend**

Navigate to the frontend directory and serve it on port 3000:

```bash
cd rag_pipeline/frontend
python -m http.server 3000
```

Then open your browser to `http://localhost:3000` to use the Groundwater Policy Assistant UI.


## API Endpoints

All RAG endpoints are grouped under the `/rag` prefix and handled by `rag_controller.py`.

**GET /rag/health**
Returns the current database connection status along with the number of indexed documents and chunks. Use this first to confirm the backend and database are properly connected.

**GET /rag/sources**
Lists all stored source documents with their titles, URLs, and provenance metadata.

**POST /rag/reindex**
Refreshes the document store by re-fetching, re-chunking, and re-embedding all source documents. This is useful after adding new documents to the corpus.

**POST /rag/ask**
Accepts a natural language question and returns a grounded response built from retrieved document chunks. The request body accepts these parameters:

```json
{
  "question": "What permits are needed to drill a well in Nebraska?",
  "top_k": 8,
  "min_score": 0.45,
  "show_sources": true
}
```

The response includes the answer text, the source chunks used, a confidence score, and a hallucination risk label.

**POST /rag/compare**
Runs the same question through two paths simultaneously: the RAG pipeline using local Nebraska documents, and a plain Gemini call with no local context. Both answers are returned side by side so users can see how document grounding changes the output quality.

```json
{
  "question": "What are the groundwater allocation limits in the Central Platte Natural Resources District?",
  "top_k": 8,
  "min_score": 0.45
}
```

**GET / (Root)**
Returns a simple status message confirming the API is running.


## Swagger UI

FastAPI auto-generates interactive API documentation. Once the backend is running, open:

```
http://127.0.0.1:8000/docs
```

From there you can click any endpoint, hit "Try it out", fill in the request body, and click Execute to test the API directly without the frontend.


## Storage Details

- `source_docs` stores document metadata including title, URL, and provenance
- `doc_chunks` stores chunk text, chunk order, heading, section path, chunk type, and embedding vectors
- Embeddings use `vector(1536)` to match the current Gemini embedding configuration


## RAG Controller Details

The controller in `rag_controller.py` is a FastAPI `APIRouter` that wires together the retrieval, prompting, and generation modules. Key behaviors include:

- On `/rag/ask`, the controller calls `retrieve_chunks_multi_query` with the user's question and parameters, then runs `pick_top_docs` to select the most relevant chunks across documents, builds a grounded prompt, and sends it to Gemini Flash for answer generation.
- On `/rag/compare`, the controller runs the full RAG pipeline and a separate unconstrained Gemini call in parallel, returning both responses for side-by-side evaluation.
- Weak evidence detection via `evidence_is_weak()` flags queries where retrieval confidence is low (top score below 0.60 or fewer than 3 matching chunks), and this status is reflected in the hallucination risk label.
- Multi-query expansion via `expand_query()` generates keyword variants (for example, a question about "permits" also queries "well registration" and "well drilling") to improve recall.
- CORS middleware is enabled to allow requests from the frontend running on a different port.


## Live Demo Walkthrough

Follow these steps in order for a live demonstration of the system.

**Pre-demo checklist:**
Make sure Docker Desktop is running, or start PostgreSQL and the backend manually. Open two terminals: one for the backend (`uvicorn main:app --reload --port 8000`) and one for the frontend (`cd rag_pipeline/frontend && python -m http.server 3000`). Have two browser tabs ready: one at `http://localhost:3000` and one at `http://127.0.0.1:8000/docs`.

**Step 1: Verify the system is alive**
Open the Swagger UI tab. Click GET `/rag/health`, hit "Try it out", then Execute. Walk through the response showing the database is connected, the document count (12), and the chunk count (~1,740). This proves the backend and database are wired up.

**Step 2: Show the document corpus**
Click GET `/rag/sources` and Execute. Scroll through and point out a few document titles from the Nebraska Legislature, DNR, and CPNRD. This shows the indexed corpus is real regulatory text.

**Step 3: Ask a grounded question**
Click POST `/rag/ask`, hit "Try it out", and paste:

```json
{
  "question": "What permits are needed to drill a well in Nebraska?",
  "top_k": 8,
  "min_score": 0.45,
  "show_sources": true
}
```

Hit Execute. While it loads, explain what is happening: the question gets embedded into a 1,536-dimension vector, matched against all chunks via cosine similarity in pgvector, and the top chunks are injected into a prompt for Gemini. Walk through the response: the answer text, sources array, confidence score, and hallucination risk label.

Try a second question for variety:

```json
{
  "question": "What are the pumping restrictions during a drought in the Central Platte region?",
  "top_k": 8,
  "min_score": 0.45,
  "show_sources": true
}
```

**Step 4: Compare mode (the key demo moment)**
Click POST `/rag/compare`, hit "Try it out", and paste:

```json
{
  "question": "What are the groundwater allocation limits in the Central Platte Natural Resources District?",
  "top_k": 8,
  "min_score": 0.45
}
```

Hit Execute. The response shows two answers side by side: one from the RAG pipeline grounded in actual Nebraska documents, and one from a plain Gemini call with no context. Point out how the RAG answer cites specific regulations and numbers while the plain Gemini answer gives generic or potentially incorrect information. This is the single most powerful moment because it visually proves why retrieval-augmented generation matters for domain-specific legal questions.

**Step 5: Switch to the frontend**
Open `http://localhost:3000` and show the Groundwater Policy Assistant UI. Explain this is what an actual end user would see. Type in a question like "Can I drill a new irrigation well on my property in Nebraska?" and show the answer come back with cited sources. If the compare toggle is available, flip it on and ask another question to demonstrate the side-by-side view in the user-friendly interface.

**Step 6: Wrap up**
Summarize what the audience just saw: a three-stage pipeline (logical chunking, embedding and storage, retrieval and generation) indexing 12 real Nebraska regulatory documents into ~1,740 searchable chunks, with every answer grounded in actual legal text and traceable sources.


## Suggested Demo Questions

These questions are known to produce strong, well-grounded responses from the current corpus:

- "What permits are needed to drill a well in Nebraska?"
- "What are the pumping restrictions during a drought in the Central Platte region?"
- "What are the groundwater allocation limits in the Central Platte Natural Resources District?"
- "Can I drill a new irrigation well on my property in Nebraska?"
- "What happens if I exceed my water allocation?"
- "Who regulates groundwater use in Nebraska?"
- "What is the Groundwater Management and Protection Act?"


## Important Notes

- The old standalone `rag_pipeline/src/main.py` entrypoint is deprecated. The backend entrypoint is the root-level `main.py`.
- If embeddings or reindexing fail, verify the Gemini API key and the PostgreSQL connection string first.
- Cosine similarity scores of 0.50 to 0.56 are considered strong matches for legal text. Uniform scores hovering near 0.50 across all chunks indicate no good match exists, which is expected behavior rather than a retrieval failure.
- The frontend and backend run on separate ports (3000 and 8000 respectively) and communicate via HTTP with CORS enabled.

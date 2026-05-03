CREATE TABLE IF NOT EXISTS source_docs (
  doc_id TEXT PRIMARY KEY,
  source_url TEXT NOT NULL,
  final_url TEXT NOT NULL,
  title TEXT,
  content_type TEXT,
  retrieved_at_utc TIMESTAMPTZ NOT NULL,
  sha256 TEXT NOT NULL,
  etag TEXT,
  last_modified TEXT,
  bytes_len BIGINT,
  extra JSONB DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS doc_chunks (
  chunk_id TEXT PRIMARY KEY,
  doc_id TEXT NOT NULL REFERENCES source_docs(doc_id) ON DELETE CASCADE,
  chunk_index INT NOT NULL,
  char_start INT,
  char_end INT,
  text TEXT NOT NULL,
  token_est INT,
  embedding vector(384),
  created_at_utc TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE(doc_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS doc_chunks_doc_id_idx ON doc_chunks(doc_id);
CREATE INDEX IF NOT EXISTS doc_chunks_embedding_cos_idx
  ON doc_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 200);

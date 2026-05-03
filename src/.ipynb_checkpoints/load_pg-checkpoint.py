from __future__ import annotations

import os
import json
import hashlib
from datetime import datetime, timezone

import psycopg
from dotenv import load_dotenv

from .logical_chunk_gemini import logical_chunk_text_gemini
from .embed_gemini import GeminiEmbedder

def utc_now():
    return datetime.now(timezone.utc)

def chunk_id(doc_id: str, idx: int, text: str) -> str:
    h = hashlib.sha256(f"{doc_id}||{idx}||{text[:200]}".encode()).hexdigest()
    return h[:28]

def main():
    load_dotenv()
    dsn = os.environ["DATABASE_URL"]

    embedder = GeminiEmbedder(output_dim=1536)

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    extracted_path = os.path.join(root, "data", "processed", "docs_extracted.jsonl")

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(open(os.path.join(root, "sql", "001_init.sql")).read())
            cur.execute(open(os.path.join(root, "sql", "002_tables.sql")).read())
        conn.commit()

        with open(extracted_path, "r", encoding="utf-8") as f:
            for line in f:
                doc = json.loads(line)
                doc_id_val = doc["doc_id"]

                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO source_docs
                        (doc_id, source_url, final_url, title, content_type, retrieved_at_utc, sha256, etag, last_modified, bytes_len, extra)
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        ON CONFLICT (doc_id) DO UPDATE SET
                          title=EXCLUDED.title,
                          retrieved_at_utc=EXCLUDED.retrieved_at_utc,
                          sha256=EXCLUDED.sha256,
                          etag=EXCLUDED.etag,
                          last_modified=EXCLUDED.last_modified,
                          bytes_len=EXCLUDED.bytes_len,
                          extra=EXCLUDED.extra
                        """,
                        (
                            doc["doc_id"],
                            doc["source_url"],
                            doc["final_url"],
                            doc.get("title"),
                            doc.get("content_type"),
                            doc["retrieved_at_utc"],
                            doc["sha256"],
                            doc.get("etag", ""),
                            doc.get("last_modified", ""),
                            doc.get("bytes_len", 0),
                            json.dumps(doc.get("extra", {})),
                        ),
                    )
                conn.commit()

                chunks = logical_chunk_text_gemini(
                    doc["text"],
                    source_name=doc.get("title") or doc_id_val,
                    max_chunk_chars=2500,
                    window_chars=18000,
                    overlap_chars=2000,
                )

                texts = [f'{c.get("heading", "")}\n\n{c["text"]}' for c in chunks]
                vecs = embedder.embed(texts)

                with conn.cursor() as cur:
                    for i, (c, v) in enumerate(zip(chunks, vecs)):
                        cid = chunk_id(doc_id_val, i, c["text"])
                        cur.execute(
                            """
                            INSERT INTO doc_chunks
                            (chunk_id, doc_id, chunk_index, char_start, char_end, text, token_est, embedding, heading, section_path)
                            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                            ON CONFLICT (chunk_id) DO UPDATE SET
                              text=EXCLUDED.text,
                              token_est=EXCLUDED.token_est,
                              embedding=EXCLUDED.embedding,
                              heading=EXCLUDED.heading,
                              section_path=EXCLUDED.section_path
                            """,
                            (
                                cid,
                                doc_id_val,
                                i,
                                c.get("char_start"),
                                c.get("char_end"),
                                c["text"],
                                c["token_est"],
                                v,
                                c.get("heading"),
                                c.get("section_path"),
                            ),
                        )
                conn.commit()

if __name__ == "__main__":
    main()
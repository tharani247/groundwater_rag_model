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


def batched(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def main():
    load_dotenv()
    dsn = os.environ["DATABASE_URL"]

    embedder = GeminiEmbedder(output_dim=1536)

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    extracted_path = os.path.join(root, "data", "processed", "docs_extracted.jsonl")

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(open(os.path.join(root, "sql", "001_init.sql"), "r", encoding="utf-8").read())
            cur.execute(open(os.path.join(root, "sql", "002_tables.sql"), "r", encoding="utf-8").read())
        conn.commit()

        with open(extracted_path, "r", encoding="utf-8") as f:
            for line in f:
                doc = json.loads(line)
                doc_id_val = doc["doc_id"]

                print(f"Processing doc: {doc_id_val} | title: {doc.get('title')}")

                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO source_docs
                        (doc_id, source_url, final_url, title, content_type, retrieved_at_utc, sha256, etag, last_modified, bytes_len, extra)
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        ON CONFLICT (doc_id) DO UPDATE SET
                          source_url = EXCLUDED.source_url,
                          final_url = EXCLUDED.final_url,
                          title = EXCLUDED.title,
                          content_type = EXCLUDED.content_type,
                          retrieved_at_utc = EXCLUDED.retrieved_at_utc,
                          sha256 = EXCLUDED.sha256,
                          etag = EXCLUDED.etag,
                          last_modified = EXCLUDED.last_modified,
                          bytes_len = EXCLUDED.bytes_len,
                          extra = EXCLUDED.extra
                        """,
                        (
                            doc["doc_id"],
                            doc["source_url"],
                            doc["final_url"],
                            doc.get("title"),
                            doc.get("content_type"),
                            doc.get("retrieved_at_utc", ""),
                            doc.get("sha256", ""),
                            doc.get("etag", ""),
                            doc.get("last_modified", ""),
                            doc.get("bytes_len", 0),
                            json.dumps(doc.get("extra", {})),
                        ),
                    )
                conn.commit()

                try:
                    chunks = logical_chunk_text_gemini(
                        doc["text"],
                        source_name=doc.get("title") or doc_id_val,
                        max_chunk_chars=1800,
                        window_chars=12000,
                        overlap_chars=500,
                    )

                    print(f"Generated {len(chunks)} chunks for {doc_id_val}")

                    if not chunks:
                        print(f"No chunks generated for {doc_id_val}, skipping")
                        continue

                    texts = [
                        f'{c.get("heading", "")}\n\n{c["text"]}'.strip()
                        for c in chunks
                    ]

                    vecs = []
                    for batch in batched(texts, 5):
                        print(f"Embedding batch with {len(batch)} chunks for {doc_id_val}")
                        batch_vecs = embedder.embed(batch)
                        vecs.extend(batch_vecs)

                    if len(vecs) != len(chunks):
                        raise RuntimeError(
                            f"Embedding count mismatch for {doc_id_val}: "
                            f"{len(vecs)} vectors for {len(chunks)} chunks"
                        )

                except Exception as e:
                    print(f"Skipping doc {doc_id_val} because chunking or embedding failed: {e}")
                    continue

                with conn.cursor() as cur:
                    cur.execute(
                        "DELETE FROM doc_chunks WHERE doc_id = %s",
                        (doc_id_val,)
                    )
                conn.commit()

                with conn.cursor() as cur:
                    for i, (c, v) in enumerate(zip(chunks, vecs)):
                        cid = chunk_id(doc_id_val, i, c["text"])

                        section_path = c.get("section_path")
                        if isinstance(section_path, (list, dict)):
                            section_path = json.dumps(section_path)

                        cur.execute(
                            """
                            INSERT INTO doc_chunks
                            (chunk_id, doc_id, chunk_index, char_start, char_end, text, token_est, embedding, heading, section_path)
                            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                            """,
                            (
                                cid,
                                doc_id_val,
                                i,
                                c.get("char_start"),
                                c.get("char_end"),
                                c["text"],
                                c.get("token_est"),
                                v,
                                c.get("heading"),
                                section_path,
                            ),
                        )
                conn.commit()

                print(f"Inserted {len(chunks)} chunks for {doc_id_val}")


if __name__ == "__main__":
    main()
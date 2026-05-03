from __future__ import annotations

import os
import json
import textwrap
import urllib.request
import urllib.error
from collections import defaultdict

import psycopg
from dotenv import load_dotenv

from .embed_gemini import GeminiEmbedder

GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"


def expand_query(question: str) -> list[str]:
    q = question.strip()
    variants = [q]
    lower_q = q.lower()

    if "permit" in lower_q or "permits" in lower_q:
        variants.append(q + " well registration")
        variants.append(q + " well drilling")
        variants.append("groundwater permit requirements nebraska")
        variants.append("nebraska well permits groundwater law")

    if "regulation" in lower_q or "regulations" in lower_q:
        variants.append("nebraska groundwater management protection act")
        variants.append("chapter 46 groundwater law nebraska")

    if "well" in lower_q or "wells" in lower_q:
        variants.append(q + " registration")
        variants.append(q + " drilling requirements")

    return list(dict.fromkeys(variants))


def retrieve_chunks(
    dsn: str,
    embedder: GeminiEmbedder,
    question: str,
    top_k: int = 60,
    min_score: float = 0.35,
):
    qv = embedder.embed([question])[0]

    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT
              c.chunk_id,
              c.doc_id,
              d.title,
              d.final_url,
              c.chunk_index,
              c.text,
              1 - (c.embedding <=> %s::vector) AS score
            FROM doc_chunks c
            JOIN source_docs d ON d.doc_id = c.doc_id
            WHERE 1 - (c.embedding <=> %s::vector) >= %s
            ORDER BY c.embedding <=> %s::vector
            LIMIT %s
            """,
            (qv, qv, min_score, qv, top_k),
        )
        return cur.fetchall()


def retrieve_chunks_multi_query(
    dsn: str,
    embedder: GeminiEmbedder,
    question: str,
    top_k_per_query: int = 20,
    min_score: float = 0.45,
):
    queries = expand_query(question)
    merged = {}

    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        for q in queries:
            qv = embedder.embed([q])[0]

            cur.execute(
                """
                SELECT
                  c.chunk_id,
                  c.doc_id,
                  d.title,
                  d.final_url,
                  c.chunk_index,
                  c.text,
                  1 - (c.embedding <=> %s::vector) AS score
                FROM doc_chunks c
                JOIN source_docs d ON d.doc_id = c.doc_id
                WHERE 1 - (c.embedding <=> %s::vector) >= %s
                ORDER BY c.embedding <=> %s::vector
                LIMIT %s
                """,
                (qv, qv, min_score, qv, top_k_per_query),
            )

            for row in cur.fetchall():
                chunk_id = row[0]
                if chunk_id not in merged or row[6] > merged[chunk_id][6]:
                    merged[chunk_id] = row

    rows = list(merged.values())
    rows.sort(key=lambda x: x[6], reverse=True)
    return rows


def evidence_is_weak(rows: list, min_top_score: float = 0.60, min_rows: int = 3) -> bool:
    if len(rows) < min_rows:
        return True
    if rows[0][6] < min_top_score:
        return True
    return False


def pick_top_docs(rows, docs_k: int = 3, chunks_per_doc: int = 2):
    by_doc = defaultdict(list)
    for r in rows:
        by_doc[r[1]].append(r)

    ranked = []
    for doc_id, rs in by_doc.items():
        best_score = max(rs, key=lambda x: x[6])[6]
        ranked.append((best_score, doc_id))

    ranked.sort(reverse=True)

    chosen = []
    for _, doc_id in ranked[:docs_k]:
        top_chunks = sorted(by_doc[doc_id], key=lambda x: x[6], reverse=True)[:chunks_per_doc]
        chosen.extend(top_chunks)

    return chosen

def build_prompt(question: str, rows: list) -> str:
    blocks = []

    for i, row in enumerate(rows, start=1):
        _, doc_id, title, url, chunk_index, text, score = row
        blocks.append(
            f"[Source {i}]\n"
            f"Doc ID: {doc_id}\n"
            f"Title: {title}\n"
            f"Source URL: {url}\n"
            f"Chunk Index: {chunk_index}\n"
            f"Similarity Score: {score:.3f}\n\n"
            f"{text.strip()}"
        )

    local_context = "\n\n====\n\n".join(blocks)

    return f"""You are a grounded legal and policy assistant for Nebraska groundwater regulations.

Answer the user's question using ONLY the LOCAL RAG CONTEXT below.

Do not rely on outside knowledge unless you explicitly say:
"That detail is not clearly stated in the retrieved sources."

You must follow these rules strictly:

1. Use only the retrieved local sources as evidence.
2. Do not mention source numbers like [Source 1] in the answer.
3. Do not invent legal requirements, agencies, deadlines, exceptions, or procedures.
4. Do not give a vague general answer when the sources are specific.
5. Finish the full answer completely. Do not stop mid-section.
6. Prefer precision over polish.
7. Keep the answer grounded in the retrieved text.
8. If the user asks a broad question, summarize only what the retrieved sources actually support.
9. If the retrieved sources are incomplete, say that clearly in plain language.
10. Write for a normal user, not for a technical debugging view.

Return the answer in exactly this format:

Overview:
Write 3 to 5 clear sentences directly answering the question.

Key Points:
- 4 to 8 bullet points
- each bullet should be specific and readable
- explain legal terms simply

Who Regulates It:
Write 1 short paragraph identifying the relevant authority or authorities, only if supported by the retrieved sources.

Practical Takeaway:
Write 1 short paragraph explaining what a landowner, farmer, developer, or applicant should understand.

Limits of Retrieved Evidence:
Write 1 short paragraph stating what the retrieved sources do not fully answer, if applicable.

Do not include inline citations, bracketed references, source numbers, footnotes, or URLs inside the answer body.

LOCAL RAG CONTEXT:
{local_context}

USER QUESTION:
{question}
"""


def summarize_with_gemini(prompt: str) -> str:
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return "GEMINI_API_KEY not set in .env"

    url = GEMINI_URL.format(model=GEMINI_MODEL, key=api_key)

    payload = json.dumps(
        {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.1,
                "topP": 0.9,
                "maxOutputTokens": 2500,
            },
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read())

        candidates = result.get("candidates", [])
        if not candidates:
            return f"Gemini returned no candidates: {json.dumps(result, indent=2)}"

        parts = candidates[0].get("content", {}).get("parts", [])
        if not parts:
            return f"Gemini returned no content parts: {json.dumps(result, indent=2)}"

        full_text = ""
        for p in parts:
            full_text += p.get("text", "")

        return full_text.strip()

    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        return f"Gemini API error {e.code}: {body}"
    except Exception as e:
        return f"Gemini error: {e}"


def answer_looks_complete(answer: str) -> bool:
    required_sections = [
        "Overview:",
        "Key Points:",
        "Who Regulates It:",
        "Practical Takeaway:",
        "Limits of Retrieved Evidence:",
    ]
    return all(section in answer for section in required_sections)


def pretty_print_answer(answer: str, width: int = 100) -> None:
    lines = answer.splitlines()

    for line in lines:
        stripped = line.strip()

        if not stripped:
            print()
            continue

        if stripped.endswith(":"):
            print(stripped)
            continue

        if stripped.startswith("- "):
            wrapped = textwrap.fill(
                stripped,
                width=width,
                subsequent_indent="  ",
            )
            print(wrapped)
            continue

        wrapped = textwrap.fill(stripped, width=width)
        print(wrapped)


def print_local_sources(rows: list) -> None:
    print("\nLocal RAG chunk sources used:")
    for i, row in enumerate(rows, start=1):
        _, doc_id, title, url, chunk_index, _, score = row
        print(f"  [Source {i}] {title}")
        print(f"    Doc ID: {doc_id}")
        print(f"    URL: {url}")
        print(f"    Chunk Index: {chunk_index}")
        print(f"    Score: {score:.3f}")

def main():
    load_dotenv()
    dsn = os.environ["DATABASE_URL"]
    embedder = GeminiEmbedder(output_dim=1536)

    print("\nGroundwater Policy Assistant")
    print("Type your question, or 'quit' to exit.\n")

    while True:
        q = input("Question: ").strip()

        if not q:
            continue

        if q.lower() in {"quit", "exit"}:
            break

        rows_all = retrieve_chunks_multi_query(
            dsn=dsn,
            embedder=embedder,
            question=q,
            top_k_per_query=20,
            min_score=0.45,
        )

        if not rows_all:
            print("\nNo confident local matches found.")
            print("Try rephrasing your question.\n")
            continue

        if evidence_is_weak(rows_all):
            print("\nThe retrieved evidence is weak for this question.")
            print("Try asking in a more specific way, such as:")
            print("  groundwater well registration under Nebraska law")
            print("  who regulates well drilling in Nebraska")
            print("  permit requirements for groundwater wells in Nebraska\n")
            continue

        rows = pick_top_docs(rows_all, docs_k=3, chunks_per_doc=2)
        prompt = build_prompt(q, rows)
        answer = summarize_with_gemini(prompt)

        if not answer_looks_complete(answer):
            retry_prompt = (
                prompt
                + "\n\nYour previous answer was incomplete. Regenerate the full answer and include every required section completely."
            )
            answer = summarize_with_gemini(retry_prompt)

        print("\nQuestion:")
        print(q)

        print("\nAnswer:")
        print("=" * 100)
        pretty_print_answer(answer, width=100)
        print("=" * 100)

        print_local_sources(rows)
        print()


if __name__ == "__main__":
    main()
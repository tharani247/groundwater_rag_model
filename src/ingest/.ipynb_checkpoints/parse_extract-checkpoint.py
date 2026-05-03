from __future__ import annotations

import os
import json
import io
import re
from bs4 import BeautifulSoup
from pypdf import PdfReader


def clean_extracted_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    text = re.sub(r"[ \t]+", " ", text)

    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def extract_pdf_pages(pdf_path: str) -> list[str]:
    with open(pdf_path, "rb") as f:
        reader = PdfReader(io.BytesIO(f.read()))

    pages = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        pages.append(clean_extracted_text(page_text))

    return pages


def extract_pdf_text(pdf_path: str) -> tuple[str, list[str]]:
    pages = extract_pdf_pages(pdf_path)
    full_text = "\n\n".join(pages)
    return clean_extracted_text(full_text), pages


def extract_html_text(html_path: str) -> tuple[str, str]:
    with open(html_path, "rb") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    title = soup.title.get_text(strip=True) if soup.title else os.path.basename(html_path)

    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    text = soup.get_text("\n", strip=True)
    text = clean_extracted_text(text)

    return title, text


def main() -> None:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    raw_dir = os.path.join(root, "data", "raw")
    processed_dir = os.path.join(root, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    fetch_index_path = os.path.join(raw_dir, "fetch_index.json")
    if not os.path.exists(fetch_index_path):
        raise FileNotFoundError(
            f"Missing {fetch_index_path}. Run `python -m src.ingest.fetch_sources` first."
        )

    with open(fetch_index_path, "r", encoding="utf-8") as f:
        fetch_index = json.load(f)

    out_path = os.path.join(processed_dir, "docs_extracted.jsonl")

    with open(out_path, "w", encoding="utf-8") as out:
        for _, meta in fetch_index.items():
            raw_path = meta["raw_path"]

            if not os.path.isabs(raw_path):
                raw_path = os.path.join(raw_dir, raw_path)

            if raw_path.lower().endswith(".pdf"):
                text, pages = extract_pdf_text(raw_path)
                title = os.path.basename(raw_path)
                ctype = "pdf"
            else:
                title, text = extract_html_text(raw_path)
                pages = []
                ctype = "html"

            rec = {
                "doc_id": meta["doc_id"],
                "source_url": meta["source_url"],
                "final_url": meta.get("final_url", meta["source_url"]),
                "title": title,
                "content_type": ctype,
                "retrieved_at_utc": meta.get("retrieved_at_utc", ""),
                "sha256": meta.get("sha256", ""),
                "etag": meta.get("etag", ""),
                "last_modified": meta.get("last_modified", ""),
                "bytes_len": meta.get("bytes_len", 0),
                "text": text,
                "pages": pages,
                "extra": {},
            }

            out.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
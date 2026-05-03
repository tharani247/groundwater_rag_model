from __future__ import annotations

import os
import time
import json
import hashlib
from urllib.parse import urlparse
from datetime import datetime, timezone

import requests

USER_AGENT = "PolicyRAG-Ingest/0.1"
TIMEOUT_S = 40
SLEEP_S = 1.0

SEED_URLS = [
    "https://nebraskalegislature.gov/laws/browse-chapters.php?chapter=46",
    "https://nebraskalegislature.gov/laws/search_range_statute.php?begin_section=46-701&end_section=46-756",
    "https://dnr.nebraska.gov/water-planning/state-laws-and-rules",
    "https://dnr.nebraska.gov/sites/default/files/doc/about/statutes/GWMgmtProtectionActStatutes.pdf",
    "https://www.cpnrd.org/wp-content/uploads/RULES-REGS_09_26_2024.pdf",
    "https://www.cpnrd.org/forms-permits/",
    "https://www.cpnrd.org/water-resources/wells/",
]

def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def stable_id(url: str, sha: str) -> str:
    return hashlib.sha256(f"{url}||{sha}".encode()).hexdigest()[:24]

def safe_name(url: str) -> str:
    p = urlparse(url)
    base = (p.netloc + p.path).strip("/")
    base = base.replace("/", "_")
    if not base:
        base = "root"
    return base

def fetch_one(url: str, cache_meta: dict | None) -> tuple[bytes, dict]:
    headers = {"User-Agent": USER_AGENT}
    if cache_meta:
        if cache_meta.get("etag"):
            headers["If-None-Match"] = cache_meta["etag"]
        if cache_meta.get("last_modified"):
            headers["If-Modified-Since"] = cache_meta["last_modified"]

    r = requests.get(url, headers=headers, timeout=TIMEOUT_S, allow_redirects=True)

    if r.status_code == 304 and cache_meta:
        return b"", {"not_modified": True, "final_url": cache_meta.get("final_url", url), **cache_meta}

    r.raise_for_status()

    meta = {
        "not_modified": False,
        "final_url": r.url,
        "content_type": r.headers.get("Content-Type", "") or "",
        "etag": r.headers.get("ETag", "") or "",
        "last_modified": r.headers.get("Last-Modified", "") or "",
    }
    return r.content, meta

def is_pdf(meta: dict) -> bool:
    ct = (meta.get("content_type", "") or "").lower()
    fu = (meta.get("final_url", "") or "").lower()
    return ("pdf" in ct) or fu.endswith(".pdf")

def main() -> None:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    raw_dir = os.path.join(root, "data", "raw")
    os.makedirs(raw_dir, exist_ok=True)

    cache_path = os.path.join(raw_dir, "fetch_index.json")
    cache = json.load(open(cache_path, "r", encoding="utf-8")) if os.path.exists(cache_path) else {}

    out_index: dict[str, dict] = {}

    for url in SEED_URLS:
        prev = cache.get(url)
        content, meta = fetch_one(url, prev)

        if meta.get("not_modified") and prev:
            out_index[url] = prev
            continue

        sha = sha256_hex(content)
        doc_id = stable_id(url, sha)

        fname = safe_name(meta["final_url"])
        ext = ".pdf" if is_pdf(meta) else ".html"
        abs_path = os.path.join(raw_dir, f"{fname}__{doc_id}{ext}")

        with open(abs_path, "wb") as f:
            f.write(content)

        rel_path = os.path.relpath(abs_path, raw_dir)

        out_index[url] = {
            "doc_id": doc_id,
            "source_url": url,
            "final_url": meta["final_url"],
            "content_type": meta.get("content_type", ""),
            "etag": meta.get("etag", ""),
            "last_modified": meta.get("last_modified", ""),
            "sha256": sha,
            "bytes_len": len(content),
            "retrieved_at_utc": utc_iso(),
            "raw_path": rel_path,
        }

        time.sleep(SLEEP_S)

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(out_index, f, indent=2)

    print(f"Fetched {len(out_index)} sources into {raw_dir}")
    print(f"Wrote index: {cache_path}")

if __name__ == "__main__":
    main()
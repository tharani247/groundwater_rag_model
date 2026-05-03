from __future__ import annotations

import os
import json
import re
import urllib.request
import urllib.error
from typing import List, Dict, Any

GEMINI_CHUNK_MODEL = os.environ.get("GEMINI_CHUNK_MODEL", "gemini-2.5-flash")
GEMINI_GEN_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
)


def normalize(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def chunk_text_windows(text: str, window_chars: int = 8000, overlap_chars: int = 500) -> List[str]:
    """
    Break very large documents into overlapping windows before sending to Gemini.
    Smaller windows mean fewer chunks per call, keeping responses well under token limits.
    """
    text = normalize(text)
    if len(text) <= window_chars:
        return [text]

    windows = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + window_chars, n)
        window = text[start:end]

        if end < n:
            split_pos = max(
                window.rfind("\n\n"),
                window.rfind("\n"),
                window.rfind(". ")
            )
            if split_pos > window_chars // 2:
                end = start + split_pos + 1
                window = text[start:end]

        windows.append(window.strip())

        if end >= n:
            break

        start = max(0, end - overlap_chars)

    return windows


def _build_chunk_prompt(text: str, source_name: str = "Document") -> str:
    return f"""
You are a legal and regulatory document chunking engine.

Your task is to split the provided document into logical semantic chunks for retrieval augmented generation.

Document name:
{source_name}

Instructions:
1. Preserve legal and regulatory hierarchy when possible.
2. Keep each chunk semantically self contained.
3. Do not mix unrelated sections in one chunk.
4. Do not split definitions, numbered clauses, exception lists, or enforcement clauses unless they are too large.
5. If a section is too large, split it into smaller meaningful subchunks.
6. Preserve section numbers, citations, and headings whenever available.
7. Return valid JSON only.
8. Do not include markdown fences.
9. Do not omit text.
10. The chunk text must be copied from the document, not rewritten.

Return JSON in this shape:
{{
  "chunks": [
    {{
      "heading": "string",
      "section_path": "string",
      "chunk_type": "definition|rule|exception|procedure|penalty|general",
      "text": "string"
    }}
  ]
}}

Document text:
{text}
""".strip()


def _make_generation_payload(prompt: str) -> bytes:
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
            "topP": 0.9,
            "maxOutputTokens": 65536,
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "chunks": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "heading": {"type": "STRING"},
                                "section_path": {"type": "STRING"},
                                "chunk_type": {"type": "STRING"},
                                "text": {"type": "STRING"}
                            },
                            "required": ["heading", "section_path", "chunk_type", "text"]
                        }
                    }
                },
                "required": ["chunks"]
            }
        }
    }
    return json.dumps(payload).encode("utf-8")


def _call_gemini_json(payload: bytes) -> Dict[str, Any]:
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set in .env")

    url = GEMINI_GEN_URL.format(model=GEMINI_CHUNK_MODEL, key=api_key)

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read())

        candidate = data["candidates"][0]

        # if Gemini stopped early due to hitting the token limit, raise a clear error
        finish_reason = candidate.get("finishReason", "")
        if finish_reason == "MAX_TOKENS":
            raise RuntimeError(
                "Gemini hit MAX_TOKENS limit and the response was truncated. "
                "Try reducing window_chars further or check your model quota."
            )

        text = candidate["content"]["parts"][0]["text"]

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            # last resort: try to salvage whatever chunks completed before the cutoff
            match = re.search(r'"chunks"\s*:\s*(\[.*)', text, re.DOTALL)
            if match:
                partial = match.group(1)
                last_brace = partial.rfind("},")
                if last_brace > 0:
                    salvaged = '{"chunks": ' + partial[:last_brace + 1] + "]}"
                    try:
                        return json.loads(salvaged)
                    except json.JSONDecodeError:
                        pass
            raise RuntimeError(f"Gemini chunk error: {e}")

    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Gemini chunk HTTP {e.code}: {body}")
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"Gemini chunk error: {e}")


def _validate_chunks(raw: Dict[str, Any]) -> List[Dict[str, str]]:
    chunks = raw.get("chunks", [])
    if not isinstance(chunks, list):
        raise RuntimeError("Gemini chunk response missing 'chunks' list")

    cleaned: List[Dict[str, str]] = []

    for c in chunks:
        heading = str(c.get("heading", "")).strip() or "Untitled"
        section_path = str(c.get("section_path", "")).strip() or heading
        chunk_type = str(c.get("chunk_type", "")).strip() or "general"
        text = str(c.get("text", "")).strip()

        if not text:
            continue

        cleaned.append({
            "heading": heading,
            "section_path": section_path,
            "chunk_type": chunk_type,
            "text": text,
        })

    if not cleaned:
        raise RuntimeError("Gemini returned no valid chunks")

    return cleaned


def logical_chunk_text_gemini(
    text: str,
    source_name: str = "Document",
    max_chunk_chars: int = 2500,
    window_chars: int = 8000,
    overlap_chars: int = 500,
) -> List[Dict[str, Any]]:
    """
    Main entry point.
    1. Normalize text
    2. Split long docs into windows (smaller windows keep responses under token limits)
    3. Ask Gemini to chunk each window logically
    4. Post-process oversized chunks
    """
    text = normalize(text)
    windows = chunk_text_windows(text, window_chars=window_chars, overlap_chars=overlap_chars)

    final_chunks: List[Dict[str, Any]] = []
    idx = 0

    for w_idx, window_text in enumerate(windows):
        prompt = _build_chunk_prompt(window_text, source_name=f"{source_name} window {w_idx + 1}")
        payload = _make_generation_payload(prompt)
        raw = _call_gemini_json(payload)
        gemini_chunks = _validate_chunks(raw)

        for gc in gemini_chunks:
            chunk_text = gc["text"]

            if len(chunk_text) <= max_chunk_chars:
                final_chunks.append({
                    "text": chunk_text,
                    "token_est": estimate_tokens(chunk_text),
                    "char_start": None,
                    "char_end": None,
                    "heading": gc["heading"],
                    "section_path": gc["section_path"],
                    "chunk_type": gc["chunk_type"],
                    "chunk_index": idx,
                })
                idx += 1
            else:
                parts = _split_large_chunk(chunk_text, max_chunk_chars=max_chunk_chars)
                for part_no, part in enumerate(parts, start=1):
                    final_chunks.append({
                        "text": part,
                        "token_est": estimate_tokens(part),
                        "char_start": None,
                        "char_end": None,
                        "heading": gc["heading"],
                        "section_path": gc["section_path"],
                        "chunk_type": gc["chunk_type"],
                        "chunk_index": idx,
                        "subchunk_of": gc["heading"],
                        "subchunk_number": part_no,
                    })
                    idx += 1

    return _dedupe_chunks(final_chunks)


def _split_large_chunk(text: str, max_chunk_chars: int = 2500) -> List[str]:
    """
    Safety splitter for very large Gemini chunks.
    Tries paragraph boundaries first, then falls back to hard splits.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        return [text[:max_chunk_chars]]

    out: List[str] = []
    buf: List[str] = []
    buf_len = 0

    for p in paragraphs:
        p_len = len(p)

        if buf and buf_len + 2 + p_len > max_chunk_chars:
            out.append("\n\n".join(buf).strip())
            buf = [p]
            buf_len = p_len
        else:
            if not buf:
                buf = [p]
                buf_len = p_len
            else:
                buf.append(p)
                buf_len += 2 + p_len

    if buf:
        out.append("\n\n".join(buf).strip())

    final = []
    for item in out:
        if len(item) <= max_chunk_chars:
            final.append(item)
        else:
            for i in range(0, len(item), max_chunk_chars):
                final.append(item[i:i + max_chunk_chars].strip())

    return [x for x in final if x]


def _dedupe_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []

    for c in chunks:
        key = (c.get("heading", ""), c.get("section_path", ""), c.get("text", ""))
        if key in seen:
            continue
        seen.add(key)
        out.append(c)

    for i, c in enumerate(out):
        c["chunk_index"] = i

    return out
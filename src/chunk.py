from __future__ import annotations

import re
from typing import List, Dict

def normalize(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()

def chunk_text(text: str, max_chars: int = 1400, overlap: int = 200) -> List[Dict]:
    t = normalize(text)
    paras = [p.strip() for p in t.split("\n\n") if p.strip()]
    chunks = []
    buf = ""
    start = 0
    cursor = 0

    def flush(buf_text: str, start_pos: int, end_pos: int):
        if not buf_text.strip():
            return
        chunks.append({
            "text": buf_text.strip(),
            "char_start": start_pos,
            "char_end": end_pos,
            "token_est": max(1, len(buf_text) // 4),
        })

    for p in paras:
        if len(buf) + len(p) + 2 <= max_chars:
            if not buf:
                start = cursor
            buf = (buf + "\n\n" + p).strip() if buf else p
            cursor += len(p) + 2
        else:
            end = start + len(buf)
            flush(buf, start, end)

            tail = buf[-overlap:] if overlap > 0 else ""
            buf = (tail + "\n\n" + p).strip() if tail else p
            start = max(0, end - len(tail))
            cursor = start + len(buf)

    end = start + len(buf)
    flush(buf, start, end)
    return chunks

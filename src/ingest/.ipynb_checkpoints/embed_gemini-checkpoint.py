from __future__ import annotations

import os
import json
import math
import urllib.request
import urllib.error
from typing import List

GEMINI_EMBED_MODEL = os.environ.get("GEMINI_EMBED_MODEL", "gemini-embedding-001")
GEMINI_EMBED_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:embedContent?key={key}"

def l2_normalize(v: List[float]) -> List[float]:
    s = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / s for x in v]

class GeminiEmbedder:
    def __init__(self, output_dim: int = 1536):
        self.api_key = os.environ.get("GEMINI_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY not set in .env")
        if output_dim < 128 or output_dim > 3072:
            raise ValueError("output_dim must be between 128 and 3072")
        self.output_dim = output_dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        url = GEMINI_EMBED_URL.format(model=GEMINI_EMBED_MODEL, key=self.api_key)

        for t in texts:
            payload = json.dumps({
                "content": {"parts": [{"text": t}]},
                "outputDimensionality": self.output_dim
            }).encode("utf-8")

            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    data = json.loads(resp.read())
                vec = data["embedding"]["values"]
                if len(vec) != self.output_dim:
                    raise RuntimeError(f"Unexpected embedding dims: {len(vec)} expected {self.output_dim}")
                out.append(l2_normalize(vec))
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"Gemini embed HTTP {e.code}: {body}")
            except Exception as e:
                raise RuntimeError(f"Gemini embed error: {e}")

        return out
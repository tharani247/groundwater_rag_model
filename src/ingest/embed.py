from __future__ import annotations

import os
import numpy as np
from sentence_transformers import SentenceTransformer

from .embed_gemini import GeminiEmbedder

ST_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

class Embedder:
    def __init__(self):
        provider = os.environ.get("EMBED_PROVIDER", "st").lower()
        self.provider = provider

        if provider == "gemini":
            out_dim = int(os.environ.get("GEMINI_OUTPUT_DIM", "1536"))
            self.gem = GeminiEmbedder(output_dim=out_dim)
            self.st = None
        else:
            self.st = SentenceTransformer(ST_MODEL_NAME)
            self.gem = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self.provider == "gemini":
            return self.gem.embed(texts)

        vecs = self.st.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        if isinstance(vecs, np.ndarray):
            vecs = vecs.tolist()
        return vecs
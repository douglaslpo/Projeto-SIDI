"""Cache em 2 níveis: exact-match (SHA256) + semantic (cosine similarity)."""

from __future__ import annotations

import hashlib
import os
from typing import Any

import numpy as np
from openai import OpenAI


class ExactCache:
    """Cache por hash SHA256 da query. Captura replays exatos (~10-15% das queries)."""

    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    @staticmethod
    def _key(query: str) -> str:
        return hashlib.sha256(query.encode()).hexdigest()

    def get(self, query: str) -> str | None:
        return self._store.get(self._key(query))

    def put(self, query: str, answer: str) -> None:
        self._store[self._key(query)] = answer

    def clear(self) -> None:
        self._store.clear()

    def stats(self) -> dict[str, int]:
        return {"size": len(self._store)}


class SemanticCache:
    """Cache por similaridade de embedding. Captura paráfrases (~20% adicional)."""

    def __init__(self, threshold: float = 0.93) -> None:
        self.threshold = threshold
        self._queries: list[str] = []
        self._embeddings: list[np.ndarray] = []
        self._answers: list[str] = []

        if "GEMINI_API_KEY" in os.environ:
            self._client = OpenAI(
                api_key=os.environ["GEMINI_API_KEY"],
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            )
            self._embed_model = os.environ.get("EMBED_MODEL", "gemini-embedding-001")
        else:
            self._client = OpenAI()
            self._embed_model = os.environ.get("EMBED_MODEL", "text-embedding-3-small")

    def _embed(self, text: str) -> np.ndarray:
        r = self._client.embeddings.create(model=self._embed_model, input=text)
        return np.array(r.data[0].embedding, dtype=float)

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0.0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def get(self, query: str) -> str | None:
        """Retorna resposta cacheada se similar à query anterior, ou None."""
        if not self._queries:
            return None

        try:
            query_embedding = self._embed(query)
        except Exception:
            return None

        best_idx = -1
        best_score = -1.0
        for idx, stored in enumerate(self._embeddings):
            score = self._cosine_similarity(query_embedding, stored)
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx >= 0 and best_score >= self.threshold:
            return self._answers[best_idx]
        return None

    def put(self, query: str, answer: str) -> None:
        try:
            embedding = self._embed(query)
        except Exception:
            return
        self._queries.append(query)
        self._embeddings.append(embedding)
        self._answers.append(answer)

    def clear(self) -> None:
        self._queries.clear()
        self._embeddings.clear()
        self._answers.clear()

    def stats(self) -> dict[str, Any]:
        return {"size": len(self._queries), "threshold": self.threshold}

"""Cache em 2 níveis: exact-match (SHA256) + semantic (cosine similarity)."""

from __future__ import annotations

import hashlib
import os
import unicodedata
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from openai import OpenAI

CacheMode = Literal["corpus", "general"]

_NON_CACHEABLE_MARKERS = (
    "nao encontrado",
    "nao encontrei",
    "erro ao gerar",
    "cota da api",
    "cota excedida",
    "resource_exhausted",
)


@dataclass(frozen=True)
class CacheEntry:
    query: str
    answer: str
    mode: CacheMode = "corpus"


def _normalize(text: str) -> str:
    lowered = text.lower().strip()
    return "".join(
        ch for ch in unicodedata.normalize("NFD", lowered) if unicodedata.category(ch) != "Mn"
    )


def is_cacheable_answer(
    answer: str,
    *,
    has_sources: bool = True,
    mode: CacheMode = "corpus",
) -> bool:
    if not answer or not answer.strip():
        return False
    if mode == "general":
        return True
    if not has_sources:
        return False
    normalized = _normalize(answer)
    return not any(marker in normalized for marker in _NON_CACHEABLE_MARKERS)


class ExactCache:
    """Cache por hash SHA256 da query. Valida texto original da pergunta."""

    def __init__(self) -> None:
        self._store: dict[str, CacheEntry] = {}

    @staticmethod
    def _key(query: str) -> str:
        return hashlib.sha256(_normalize(query).encode()).hexdigest()

    def get_entry(self, query: str) -> CacheEntry | None:
        entry = self._store.get(self._key(query))
        if entry and _normalize(entry.query) == _normalize(query):
            return entry
        return None

    def get_valid(self, query: str) -> CacheEntry | None:
        entry = self.get_entry(query)
        if entry and is_cacheable_answer(
            entry.answer,
            has_sources=entry.mode == "corpus",
            mode=entry.mode,
        ):
            return entry
        return None

    def put(
        self,
        query: str,
        answer: str,
        *,
        has_sources: bool = True,
        mode: CacheMode = "corpus",
    ) -> None:
        if not is_cacheable_answer(answer, has_sources=has_sources, mode=mode):
            return
        self._store[self._key(query)] = CacheEntry(query=query, answer=answer, mode=mode)

    def clear(self) -> None:
        self._store.clear()

    def stats(self) -> dict[str, int]:
        return {"size": len(self._store)}


class SemanticCache:
    """Cache por similaridade de embedding. Só reutiliza mesma pergunta normalizada."""

    def __init__(self, threshold: float = 0.93) -> None:
        self.threshold = threshold
        self._entries: list[CacheEntry] = []
        self._embeddings: list[np.ndarray] = []

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

    def get(self, query: str) -> CacheEntry | None:
        if not self._entries:
            return None

        try:
            query_embedding = self._embed(query)
        except Exception:
            return None

        best_idx = -1
        best_score = -1.0
        for idx, stored in enumerate(self._embeddings):
            entry = self._entries[idx]
            if not is_cacheable_answer(
                entry.answer,
                has_sources=entry.mode == "corpus",
                mode=entry.mode,
            ):
                continue
            score = self._cosine_similarity(query_embedding, stored)
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx >= 0 and best_score >= self.threshold:
            return self._entries[best_idx]
        return None

    def put(
        self,
        query: str,
        answer: str,
        *,
        has_sources: bool = True,
        mode: CacheMode = "corpus",
    ) -> None:
        if not is_cacheable_answer(answer, has_sources=has_sources, mode=mode):
            return
        try:
            embedding = self._embed(query)
        except Exception:
            return
        self._entries.append(CacheEntry(query=query, answer=answer, mode=mode))
        self._embeddings.append(embedding)

    def clear(self) -> None:
        self._entries.clear()
        self._embeddings.clear()

    def stats(self) -> dict[str, Any]:
        return {"size": len(self._entries), "threshold": self.threshold}

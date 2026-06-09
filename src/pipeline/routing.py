"""Model routing cheap-first com fallback."""

from __future__ import annotations

import os
import unicodedata
from dataclasses import dataclass

from openai import OpenAI


@dataclass(frozen=True)
class RouteDecision:
    model: str
    complexity: str  # "simple" | "complex"
    reason: str


_COMPLEX_KEYWORDS = (
    "explique",
    "compare",
    "analise",
    "avalie",
    "projete",
    "detalhe",
    "quais os riscos",
    "base legal",
    "retencao",
    "retenção",
    "consentimento",
)


def _normalize(text: str) -> str:
    lowered = text.lower().strip()
    return "".join(
        ch for ch in unicodedata.normalize("NFD", lowered) if unicodedata.category(ch) != "Mn"
    )


def classify_complexity(query: str) -> RouteDecision:
    """Classifica complexidade da query para escolher modelo (cheap vs premium)."""
    cheap_model = os.environ.get("CHEAP_MODEL", "gemini-2.5-flash-lite")
    premium_model = os.environ.get("PREMIUM_MODEL", "gemini-2.5-pro")

    normalized = _normalize(query)
    word_count = len(normalized.split())

    if word_count > 40:
        return RouteDecision(
            model=premium_model,
            complexity="complex",
            reason="Query longa (>40 palavras) — provável análise detalhada.",
        )

    if any(keyword in normalized for keyword in _COMPLEX_KEYWORDS):
        matched = next(k for k in _COMPLEX_KEYWORDS if k in normalized)
        return RouteDecision(
            model=premium_model,
            complexity="complex",
            reason=f"Palavra-chave de alta complexidade detectada: '{matched}'.",
        )

    if len(normalized) < 60 and normalized.endswith("?"):
        return RouteDecision(
            model=cheap_model,
            complexity="simple",
            reason="Pergunta curta e objetiva.",
        )

    if word_count <= 12:
        return RouteDecision(
            model=cheap_model,
            complexity="simple",
            reason="Consulta breve com baixa profundidade analítica.",
        )

    return RouteDecision(
        model=premium_model,
        complexity="complex",
        reason="Consulta de tamanho médio/alto sem ser objetiva — preferir modelo premium.",
    )


def make_client() -> OpenAI:
    """Cliente OpenAI-compatible para o provider configurado."""
    if "GEMINI_API_KEY" in os.environ:
        return OpenAI(
            api_key=os.environ["GEMINI_API_KEY"],
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
    return OpenAI()

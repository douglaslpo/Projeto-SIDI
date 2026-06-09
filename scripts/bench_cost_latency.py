"""Benchmark simples de latência e custo estimado."""

from __future__ import annotations

import os
import statistics
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

QUESTIONS = [
    "Posso armazenar CPF de usuários?",
    "O que a LGPD diz sobre consentimento?",
    "Quais são os direitos do titular dos dados?",
    "O que é base legal para tratamento de dados pessoais?",
    "Quais cuidados devo tomar ao reter dados pessoais?",
]


def main() -> None:
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY")):
        print("Configure GEMINI_API_KEY ou OPENAI_API_KEY no .env")
        return

    from src.pipeline.rag import build_rag_pipeline

    pipeline = build_rag_pipeline(corpus_dir="data/corpus")
    latencies: list[float] = []

    print(f"{'Pergunta':<55} {'Latência (ms)':>14}")
    print("-" * 72)
    for q in QUESTIONS:
        start = time.perf_counter()
        result = pipeline.answer(q)
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)
        preview = q[:52] + "..." if len(q) > 55 else q
        print(f"{preview:<55} {elapsed_ms:>12.0f}")

    if latencies:
        print("-" * 72)
        print(f"P50: {statistics.median(latencies):.0f} ms")
        print(f"P95: {sorted(latencies)[int(len(latencies) * 0.95) - 1]:.0f} ms")
        print("Custo: a medir (preencher após estimativa por tokens no provider).")


if __name__ == "__main__":
    main()

"""Métricas estilo RAGAS via embeddings quando cota LLM está esgotada.

Usa golden set + contextos recuperados pelo RAG. Respostas extrativas do melhor chunk.
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

GOLDEN_SET = ROOT / "data" / "eval" / "golden_set.json"
RESULTS_PATH = ROOT / "data" / "eval" / "ragas_results.json"


def _client() -> OpenAI:
    import os

    if "GEMINI_API_KEY" in os.environ:
        return OpenAI(
            api_key=os.environ["GEMINI_API_KEY"],
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
    return OpenAI()


def _embed(client: OpenAI, texts: list[str], model: str) -> list[np.ndarray]:
    import time

    vectors: list[np.ndarray] = []
    for text in texts:
        for attempt in range(5):
            try:
                r = client.embeddings.create(model=model, input=text[:8000])
                vectors.append(np.array(r.data[0].embedding, dtype=float))
                time.sleep(0.8)
                break
            except Exception as exc:
                if "429" in str(exc) and attempt < 4:
                    time.sleep(15 * (attempt + 1))
                    continue
                raise
    return vectors


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _extractive_answer(contexts: list[str], question: str) -> str:
    if not contexts:
        return "Não encontrado no corpus."
    scored = sorted(
        contexts,
        key=lambda c: len(set(question.lower().split()) & set(c.lower().split())),
        reverse=True,
    )
    text = scored[0]
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    picked = " ".join(sentences[:3]).strip()
    return picked or text[:500]


def main() -> None:
    load_dotenv(ROOT / ".env")
    import os

    embed_model = os.environ.get("EMBED_MODEL", "gemini-embedding-001")
    client = _client()

    with GOLDEN_SET.open(encoding="utf-8") as f:
        items = json.load(f)

    from src.pipeline.rag import build_rag_pipeline

    pipeline = build_rag_pipeline(corpus_dir="data/corpus")

    faith_scores: list[float] = []
    rel_scores: list[float] = []
    ctx_scores: list[float] = []

    for item in items:
        q = item["question"]
        gt = item["ground_truth"]
        hits = pipeline.retrieve(q, k=5)
        contexts = [h["text"] for h in hits]
        answer = _extractive_answer(contexts, q)

        base_embs = _embed(client, [q, answer, gt], embed_model)
        q_emb, a_emb, gt_emb = base_embs
        ctx_embs = _embed(client, contexts, embed_model) if contexts else []

        # faithfulness: resposta ancorada no contexto recuperado
        if contexts:
            ctx_join = " ".join(contexts)[:8000]
            ctx_join_emb = _embed(client, [ctx_join], embed_model)[0]
            faith_scores.append(max(0.0, min(1.0, _cosine(a_emb, ctx_join_emb))))
        else:
            faith_scores.append(0.0)

        # answer_relevancy: alinhamento pergunta ↔ resposta
        rel_scores.append(max(0.0, min(1.0, _cosine(q_emb, a_emb))))

        # context_precision: fração de chunks relevantes (sim ≥ gt) no top-k
        if contexts:
            sims = [_cosine(gt_emb, c) for c in ctx_embs]
            threshold = 0.72
            ctx_scores.append(sum(1 for s in sims if s >= threshold) / len(sims))
        else:
            ctx_scores.append(0.0)

        print(f"OK: {q[:55]}...")

    scores = {
        "faithfulness": round(float(np.mean(faith_scores)), 2),
        "answer_relevancy": round(float(np.mean(rel_scores)), 2),
        "context_precision": round(float(np.mean(ctx_scores)), 2),
        "n_queries": len(items),
        "method": "embedding_proxy_extractive_answers",
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2, ensure_ascii=False)

    print(
        "faithfulness={:.2f}, answer_relevancy={:.2f}, context_precision={:.2f}".format(
            scores["faithfulness"],
            scores["answer_relevancy"],
            scores["context_precision"],
        )
    )


if __name__ == "__main__":
    main()

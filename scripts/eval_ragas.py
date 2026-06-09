"""Avaliação RAGAS no golden set (faithfulness, answer_relevancy, context_precision)."""

from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

from dotenv import load_dotenv

# ragas importa ChatVertexAI removido do langchain-community recente
_vertexai_stub = types.ModuleType("langchain_community.chat_models.vertexai")
_vertexai_stub.ChatVertexAI = MagicMock
sys.modules["langchain_community.chat_models.vertexai"] = _vertexai_stub

from datasets import Dataset  # noqa: E402
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # noqa: E402
from ragas import evaluate  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.metrics import answer_relevancy, context_precision, faithfulness  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
GOLDEN_SET = ROOT / "data" / "eval" / "golden_set.json"
RESULTS_PATH = ROOT / "data" / "eval" / "ragas_results.json"
DATASET_CACHE = ROOT / "data" / "eval" / "ragas_dataset.json"


def _make_wrappers():
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Configure GEMINI_API_KEY ou OPENAI_API_KEY no .env")

    base_url = None
    if "GEMINI_API_KEY" in os.environ:
        base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"

    # Judge RAGAS: modelo cheap evita estourar cota free do flash (20 RPD).
    judge_model = os.environ.get("CHEAP_MODEL", os.environ.get("LLM_MODEL", "gemini-2.5-flash-lite"))
    embed_model = os.environ.get("EMBED_MODEL", "gemini-embedding-001")

    llm = ChatOpenAI(
        model=judge_model,
        api_key=api_key,
        base_url=base_url,
        temperature=0,
    )
    embeddings = OpenAIEmbeddings(
        model=embed_model,
        api_key=api_key,
        base_url=base_url,
    )
    return LangchainLLMWrapper(llm), LangchainEmbeddingsWrapper(embeddings)


def build_dataset(corpus_dir: str = "data/corpus", *, use_cache: bool = True) -> Dataset:
    if use_cache and DATASET_CACHE.exists():
        with DATASET_CACHE.open(encoding="utf-8") as f:
            rows = json.load(f)
        print(f"Dataset carregado do cache ({len(rows)} queries).")
        return Dataset.from_list(rows)

    from src.pipeline.rag import build_rag_pipeline

    with GOLDEN_SET.open(encoding="utf-8") as f:
        items = json.load(f)

    pipeline = build_rag_pipeline(corpus_dir=corpus_dir)
    eval_model = os.environ.get("CHEAP_MODEL", pipeline.llm_model)

    rows: list[dict] = []
    for item in items:
        question = item["question"]
        hits = pipeline.retrieve(question, k=5)
        result = pipeline.answer(question, model=eval_model)
        contexts = [h["text"] for h in hits] if hits else []
        rows.append(
            {
                "question": question,
                "answer": result["answer"],
                "contexts": contexts,
                "ground_truth": item["ground_truth"],
            }
        )
        print(f"OK: {question[:60]}...")

    DATASET_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with DATASET_CACHE.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    return Dataset.from_list(rows)


def _mean_metric(result, name: str) -> float:
    import math

    import numpy as np

    df = result.to_pandas()
    if name not in df.columns:
        raise KeyError(f"Métrica ausente: {name}")
    values = [float(v) for v in df[name].tolist() if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not values:
        return float("nan")
    return float(np.mean(values))


def main() -> None:
    load_dotenv(ROOT / ".env")
    if not GOLDEN_SET.exists():
        raise FileNotFoundError(f"Golden set não encontrado: {GOLDEN_SET}")

    print(f"Carregando golden set ({GOLDEN_SET})...")
    dataset = build_dataset()
    llm, embeddings = _make_wrappers()

    print("Executando RAGAS (pode levar alguns minutos)...")
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=llm,
        embeddings=embeddings,
    )

    scores = {
        "faithfulness": round(_mean_metric(result, "faithfulness"), 2),
        "answer_relevancy": round(_mean_metric(result, "answer_relevancy"), 2),
        "context_precision": round(_mean_metric(result, "context_precision"), 2),
        "n_queries": len(dataset),
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
    print(f"Resultados salvos em {RESULTS_PATH}")


if __name__ == "__main__":
    main()

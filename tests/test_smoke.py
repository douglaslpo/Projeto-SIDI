"""Smoke tests — rodam apos voce implementar TODOs 1-3 para validar minimo.

Uso: `uv run pytest tests/test_smoke.py -v`

Para destravar este teste, voce precisa de:
- TODOs 1-3 implementados em src/pipeline/rag.py
- Corpus em data/corpus/ com pelo menos 1 PDF
- .env configurado com API key
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def pipeline():
    """Inicializa pipeline RAG com corpus de teste."""
    pytest.importorskip("dotenv")
    from dotenv import load_dotenv

    load_dotenv()

    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY")):
        pytest.skip("API key nao configurada em .env")

    corpus_dir = Path("data/corpus")
    if not corpus_dir.exists() or not list(corpus_dir.glob("*.pdf")):
        pytest.skip("data/corpus/ vazio — adicione pelo menos 1 PDF")

    from src.pipeline.rag import build_rag_pipeline

    return build_rag_pipeline(corpus_dir=str(corpus_dir))


def test_pipeline_indexa_chunks(pipeline):
    """Apos ingest, collection deve ter >= 1 chunk."""
    assert pipeline.collection.count() > 0, "Esperado >=1 chunk indexado"


def test_retrieve_top_k(pipeline):
    """Retrieve deve retornar lista de dicts com campos esperados."""
    hits = pipeline.retrieve("teste de busca", k=3)
    assert isinstance(hits, list)
    assert len(hits) <= 3
    if hits:
        h = hits[0]
        assert "text" in h
        assert "source" in h
        assert "distance" in h


def test_answer_retorna_resposta_com_fonte(pipeline):
    """answer() deve retornar dict com 'answer' string nao-vazia e 'sources' lista."""
    result = pipeline.answer("Sobre o que e este corpus?")
    assert isinstance(result, dict)
    assert "answer" in result
    assert isinstance(result["answer"], str)
    assert len(result["answer"]) > 0
    assert "sources" in result
    assert isinstance(result["sources"], list)


def test_classify_complexity_simple():
    from src.pipeline.routing import classify_complexity

    decision = classify_complexity("O que e LGPD?")
    assert decision.complexity in {"simple", "complex"}
    assert decision.model
    assert decision.reason


def test_classify_complexity_complex_keyword():
    from src.pipeline.routing import classify_complexity

    decision = classify_complexity("Explique em detalhe a base legal para tratamento de CPF.")
    assert decision.complexity == "complex"


def test_cite_article_encontrado():
    from src.pipeline.tools import cite_article

    corpus_dir = Path("data/corpus")
    if not corpus_dir.exists() or not list(corpus_dir.glob("*.pdf")):
        pytest.skip("data/corpus/ vazio — adicione pelo menos 1 PDF")

    result = cite_article(1, corpus_dir=str(corpus_dir))
    assert isinstance(result, str)
    assert len(result) > 0
    assert "Artigo n" not in result or "Fonte:" in result


def test_cite_article_nao_encontrado():
    from src.pipeline.tools import cite_article

    result = cite_article(9999, corpus_dir="data/corpus")
    assert "nao encontrado" in result.lower() or "não encontrado" in result.lower()


def test_exact_cache_clear():
    from src.pipeline.cache import ExactCache

    cache = ExactCache()
    cache.put("teste", "resposta com fonte", has_sources=True)
    entry = cache.get_valid("teste")
    assert entry is not None
    assert entry.answer == "resposta com fonte"
    cache.clear()
    assert cache.get_valid("teste") is None


def test_cache_nao_armazena_resposta_sem_fonte():
    from src.pipeline.cache import ExactCache, is_cacheable_answer

    assert not is_cacheable_answer("Não encontrado no corpus.", has_sources=False)
    assert not is_cacheable_answer("Resposta ok", has_sources=False)
    assert is_cacheable_answer("Resposta geral via API.", has_sources=False, mode="general")

    exact = ExactCache()
    exact.put("q1", "Não encontrado no corpus.", has_sources=False)
    assert exact.get_valid("q1") is None

    exact.put("q2", "Resposta sobre LAI.", has_sources=False, mode="general")
    entry = exact.get_valid("q2")
    assert entry is not None
    assert entry.mode == "general"


def test_run_tool_call():
    from src.pipeline.tools import run_tool_call

    result = run_tool_call("cite_article", '{"article_number": 1}')
    assert isinstance(result, str)
    assert not result.startswith("ERROR")

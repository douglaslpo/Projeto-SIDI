"""Streamlit UI — Assistente LGPD com RAG, cache e model routing."""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

load_dotenv(_ROOT / ".env")

import streamlit as st  # noqa: E402

from src.observability.trace import log_event, trace  # noqa: E402
from src.pipeline.cache import ExactCache, SemanticCache  # noqa: E402
from src.pipeline.rag import build_rag_pipeline  # noqa: E402
from src.pipeline.routing import classify_complexity  # noqa: E402
from src.pipeline.tools import cite_article  # noqa: E402

EXAMPLE_QUESTIONS = [
    "Posso armazenar CPF de usuários? Em quais condições?",
    "O que a LGPD diz sobre consentimento?",
    "Quais são os direitos do titular dos dados?",
    "O que é base legal para tratamento de dados pessoais?",
    "Quais cuidados devo tomar ao reter dados pessoais?",
]

st.set_page_config(page_title="Assistente LGPD com RAG", page_icon="⚖️", layout="centered")

st.title("⚖️ Assistente LGPD com RAG")
st.caption(
    "Responde perguntas sobre LGPD com base em corpus local, cita fontes e consulta "
    "a API quando a pergunta está fora do corpus."
)
st.warning(
    "Uso educacional e informacional; não constitui parecer jurídico. "
    "Consulte um profissional qualificado para decisões de compliance."
)


@st.cache_resource
def get_pipeline():
    return build_rag_pipeline(corpus_dir=str(_ROOT / "data" / "corpus"))


@st.cache_resource
def get_exact_cache():
    return ExactCache()


@st.cache_resource
def get_semantic_cache():
    return SemanticCache(threshold=0.93)


def _render_result(result: dict) -> None:
    mode = result.get("mode", "corpus")
    if mode == "general":
        st.info("Resposta via API (fora do corpus LGPD local).")
    else:
        st.success("Resposta baseada no corpus LGPD indexado.")

    if result.get("fallback_used"):
        st.warning(
            f"Cota do modelo premium esgotada — resposta gerada com "
            f"`{result.get('model_used', 'modelo econômico')}`."
        )

    st.write(result["answer"])
    if result.get("sources"):
        with st.expander("Fontes citadas (corpus LGPD)"):
            for source, page in result["sources"]:
                st.write(f"- `{source}:p{page}`")


try:
    with st.spinner("Inicializando pipeline RAG..."):
        pipeline = get_pipeline()
        exact_cache = get_exact_cache()
        semantic_cache = get_semantic_cache()
except Exception as exc:
    st.error(f"Falha ao inicializar o pipeline: {exc}")
    st.info(
        "Verifique se há PDFs em `data/corpus/` e se `.env` contém GEMINI_API_KEY ou OPENAI_API_KEY."
    )
    st.stop()

with st.sidebar:
    st.header("Exemplos de perguntas")
    for idx, example in enumerate(EXAMPLE_QUESTIONS):
        if st.button(example, use_container_width=True, key=f"example_{idx}"):
            st.session_state["query_input"] = example

    st.divider()
    st.header("Métricas")
    st.metric("Chunks indexados", pipeline.collection.count())
    st.metric("Exact cache", exact_cache.stats()["size"])
    st.metric("Semantic cache", semantic_cache.stats()["size"])

    if st.button("Limpar caches"):
        exact_cache.clear()
        semantic_cache.clear()
        get_exact_cache.clear()
        get_semantic_cache.clear()
        st.success("Caches limpos.")

with st.expander("Consultar artigo da LGPD (tool `cite_article`)"):
    article_num = st.number_input("Número do artigo", min_value=1, max_value=99, value=5, step=1)
    if st.button("Buscar artigo no corpus"):
        corpus_dir = str(_ROOT / "data" / "corpus")
        result = cite_article(int(article_num), corpus_dir=corpus_dir)
        st.text(result)
        st.caption("Exemplo de function-calling: `run_tool_call('cite_article', '{\"article_number\": 5}')`")

with st.form("query_form", clear_on_submit=False):
    query = st.text_input(
        "Sua pergunta:",
        placeholder="Ex.: Posso armazenar CPF de usuários?",
        key="query_input",
    )
    submitted = st.form_submit_button("Perguntar")

if submitted and query:
    with trace("query_handle", query=query) as ctx:
        trace_id = ctx["trace_id"]

        cached = exact_cache.get_valid(query)
        if cached:
            st.success("Cache hit (exact)")
            _render_result({"answer": cached.answer, "mode": cached.mode, "sources": []})
            log_event("cache_hit", trace_id=trace_id, layer="exact", mode=cached.mode)
            st.stop()

        try:
            cached = semantic_cache.get(query)
        except Exception:
            cached = None
            st.warning("Semantic cache indisponível — usando LLM.")

        if cached:
            st.success("Cache hit (semantic)")
            _render_result({"answer": cached.answer, "mode": cached.mode, "sources": []})
            log_event("cache_hit", trace_id=trace_id, layer="semantic", mode=cached.mode)
            st.stop()

        try:
            decision = classify_complexity(query)
            st.info(f"Routing: {decision.complexity} → {decision.model} ({decision.reason})")
            log_event("route_decision", trace_id=trace_id, **decision.__dict__)
        except Exception as exc:
            decision = None
            st.warning(f"Routing indisponível: {exc}")

        try:
            model = decision.model if decision else None
            result = pipeline.answer(query, model=model)
        except Exception as exc:
            st.error(f"Erro ao processar pergunta: {exc}")
            st.stop()

        _render_result(result)

        mode = result.get("mode", "corpus")
        exact_cache.put(
            query,
            result["answer"],
            has_sources=bool(result.get("sources")),
            mode=mode,
        )
        semantic_cache.put(
            query,
            result["answer"],
            has_sources=bool(result.get("sources")),
            mode=mode,
        )
        log_event(
            "answer_generated",
            trace_id=trace_id,
            sources=len(result.get("sources", [])),
            mode=mode,
        )

st.divider()
st.caption(
    "Projeto de portfólio — LLM + RAG + tool-use + cache semântico + model routing. "
    "Veja README.md para arquitetura, setup e limitações."
)

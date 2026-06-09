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
from src.ui.design_system import (  # noqa: E402
    inject_global_styles,
    render_disclaimer,
    render_footer,
    render_hero,
    render_routing_badge,
    render_sources,
    render_status_card,
)

EXAMPLE_QUESTIONS = [
    "Posso armazenar CPF de usuários? Em quais condições?",
    "O que a LGPD diz sobre consentimento?",
    "Quais são os direitos do titular dos dados?",
    "O que é base legal para tratamento de dados pessoais?",
    "Quais cuidados devo tomar ao reter dados pessoais?",
]

st.set_page_config(
    page_title="Assistente LGPD",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_global_styles()


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
        render_status_card(
            "general",
            "Resposta via API",
            "Pergunta fora do corpus LGPD local — resposta gerada sem trechos indexados.",
        )
    else:
        render_status_card(
            "corpus",
            "Resposta ancorada no corpus",
            "Trechos da Lei 13.709/2018 foram recuperados e usados na geração.",
        )

    if result.get("fallback_used"):
        render_status_card(
            "warning",
            "Fallback de modelo",
            f"Cota do modelo premium esgotada. Resposta gerada com {result.get('model_used', 'modelo econômico')}.",
        )

    render_status_card("corpus", "Resposta", result["answer"])
    render_sources(result.get("sources") or [])


# --- Layout principal ---
col_main, _ = st.columns([6, 1])

with col_main:
    render_hero()
    render_disclaimer()

try:
    with st.spinner("Inicializando pipeline RAG..."):
        pipeline = get_pipeline()
        exact_cache = get_exact_cache()
        semantic_cache = get_semantic_cache()
except Exception as exc:
    render_status_card("error", "Falha na inicialização", str(exc))
    st.info(
        "Verifique se há PDFs em `data/corpus/` e se `.env` contém GEMINI_API_KEY ou OPENAI_API_KEY."
    )
    st.stop()

with st.sidebar:
    st.markdown("### Navegação")
    st.caption("Exemplos prontos para testar o RAG no corpus LGPD.")
    for idx, example in enumerate(EXAMPLE_QUESTIONS):
        if st.button(example, use_container_width=True, key=f"example_{idx}"):
            st.session_state["query_input"] = example

    st.divider()
    st.markdown("### Observabilidade")
    m1, m2 = st.columns(2)
    m1.metric("Chunks", pipeline.collection.count())
    m2.metric("Exact", exact_cache.stats()["size"])
    st.metric("Semantic cache", semantic_cache.stats()["size"])

    if st.button("Limpar caches", use_container_width=True):
        exact_cache.clear()
        semantic_cache.clear()
        get_exact_cache.clear()
        get_semantic_cache.clear()
        st.success("Caches limpos.")

with col_main:
    tab_chat, tab_tool = st.tabs(["Perguntar", "Consultar artigo"])

    with tab_tool:
        st.markdown("##### Tool `cite_article`")
        st.caption("Busca determinística por número de artigo no PDF da LGPD.")
        t1, t2 = st.columns([1, 2])
        with t1:
            article_num = st.number_input(
                "Artigo",
                min_value=1,
                max_value=99,
                value=5,
                step=1,
                label_visibility="collapsed",
            )
        with t2:
            if st.button("Buscar no corpus", use_container_width=True):
                corpus_dir = str(_ROOT / "data" / "corpus")
                tool_result = cite_article(int(article_num), corpus_dir=corpus_dir)
                render_status_card("corpus", f"Artigo {article_num}", tool_result, mono=True)
        st.caption(
            'Function-calling: `run_tool_call("cite_article", \'{"article_number": 5}\')`'
        )

    with tab_chat:
        with st.form("query_form", clear_on_submit=False):
            query = st.text_input(
                "Sua pergunta sobre LGPD",
                placeholder="Ex.: Posso armazenar CPF de usuários?",
                key="query_input",
                label_visibility="collapsed",
            )
            submitted = st.form_submit_button("Perguntar", use_container_width=True)

        if submitted and query:
            with trace("query_handle", query=query) as ctx:
                trace_id = ctx["trace_id"]

                cached = exact_cache.get_valid(query)
                if cached:
                    render_status_card("cache_exact", "Cache hit · exact", "Resposta reutilizada sem nova chamada à API.")
                    _render_result({"answer": cached.answer, "mode": cached.mode, "sources": []})
                    log_event("cache_hit", trace_id=trace_id, layer="exact", mode=cached.mode)
                    st.stop()

                try:
                    cached = semantic_cache.get(query)
                except Exception:
                    cached = None
                    render_status_card(
                        "warning",
                        "Semantic cache",
                        "Indisponível nesta requisição — seguindo com LLM.",
                    )

                if cached:
                    render_status_card(
                        "cache_semantic",
                        "Cache hit · semantic",
                        "Paráfrase detectada — resposta reutilizada.",
                    )
                    _render_result({"answer": cached.answer, "mode": cached.mode, "sources": []})
                    log_event("cache_hit", trace_id=trace_id, layer="semantic", mode=cached.mode)
                    st.stop()

                try:
                    decision = classify_complexity(query)
                    render_routing_badge(decision.complexity, decision.model, decision.reason)
                    log_event("route_decision", trace_id=trace_id, **decision.__dict__)
                except Exception as exc:
                    decision = None
                    render_status_card("warning", "Routing", f"Indisponível: {exc}")

                try:
                    model = decision.model if decision else None
                    with st.spinner("Consultando corpus e gerando resposta..."):
                        result = pipeline.answer(query, model=model)
                except Exception as exc:
                    render_status_card("error", "Erro", f"Não foi possível processar a pergunta: {exc}")
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

    render_footer()

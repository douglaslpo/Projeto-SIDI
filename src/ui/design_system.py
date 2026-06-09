"""Design system — tokens, CSS global e componentes visuais do Assistente LGPD."""

from __future__ import annotations

from typing import Literal

import streamlit as st

# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------

COLORS = {
    "primary": "#0F2B46",
    "primary_light": "#1E5A8A",
    "accent": "#0D9488",
    "accent_soft": "#CCFBF1",
    "surface": "#FFFFFF",
    "surface_muted": "#F1F5F9",
    "background": "#F8FAFC",
    "text": "#0F172A",
    "text_muted": "#64748B",
    "border": "#E2E8F0",
    "warning_bg": "#FFFBEB",
    "warning_text": "#92400E",
    "success_bg": "#ECFDF5",
    "success_text": "#047857",
    "info_bg": "#EFF6FF",
    "info_text": "#1D4ED8",
    "cache_bg": "#EEF2FF",
    "cache_text": "#4338CA",
}

RADIUS = {
    "sm": "8px",
    "md": "12px",
    "lg": "16px",
    "pill": "999px",
}

SHADOW = "0 1px 3px rgba(15, 43, 70, 0.08), 0 8px 24px rgba(15, 43, 70, 0.06)"

StatusKind = Literal["corpus", "general", "cache_exact", "cache_semantic", "routing", "warning", "error"]


def inject_global_styles() -> None:
    """Injeta CSS customizado alinhado ao tema Streamlit."""
    c = COLORS
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=DM+Serif+Display&display=swap');

        :root {{
            --lgpd-primary: {c["primary"]};
            --lgpd-accent: {c["accent"]};
            --lgpd-surface: {c["surface"]};
            --lgpd-muted: {c["surface_muted"]};
            --lgpd-text: {c["text"]};
            --lgpd-text-muted: {c["text_muted"]};
            --lgpd-border: {c["border"]};
            --lgpd-radius: {RADIUS["md"]};
            --lgpd-shadow: {SHADOW};
        }}

        .stApp {{
            background: linear-gradient(180deg, {c["background"]} 0%, {c["surface"]} 220px);
        }}

        h1, h2, h3, .stMarkdown h1, .stMarkdown h2 {{
            font-family: 'DM Serif Display', Georgia, serif !important;
            color: {c["primary"]} !important;
            letter-spacing: -0.02em;
        }}

        p, label, .stMarkdown, .stTextInput label, .stButton button {{
            font-family: 'DM Sans', system-ui, sans-serif !important;
        }}

        [data-testid="stSidebar"] {{
            background: {c["primary"]} !important;
            border-right: none !important;
        }}

        [data-testid="stSidebar"] * {{
            color: #E2E8F0 !important;
        }}

        [data-testid="stSidebar"] .stMarkdown h1,
        [data-testid="stSidebar"] .stMarkdown h2,
        [data-testid="stSidebar"] .stMarkdown h3 {{
            color: #F8FAFC !important;
            font-family: 'DM Sans', sans-serif !important;
            font-weight: 600 !important;
        }}

        [data-testid="stSidebar"] .stButton > button {{
            background: rgba(255,255,255,0.08) !important;
            border: 1px solid rgba(255,255,255,0.14) !important;
            color: #F8FAFC !important;
            border-radius: {RADIUS["sm"]} !important;
            font-size: 0.82rem !important;
            text-align: left !important;
            transition: background 0.15s ease;
        }}

        [data-testid="stSidebar"] .stButton > button:hover {{
            background: rgba(255,255,255,0.16) !important;
            border-color: rgba(255,255,255,0.28) !important;
        }}

        [data-testid="stMetric"] {{
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: {RADIUS["sm"]};
            padding: 0.65rem 0.75rem;
        }}

        div[data-testid="stForm"] {{
            background: {c["surface"]};
            border: 1px solid {c["border"]};
            border-radius: {RADIUS["lg"]};
            padding: 1.25rem 1.35rem 0.5rem;
            box-shadow: {SHADOW};
        }}

        .stTextInput input {{
            border-radius: {RADIUS["sm"]} !important;
            border-color: {c["border"]} !important;
        }}

        .stFormSubmitButton button {{
            background: {c["primary"]} !important;
            color: white !important;
            border: none !important;
            border-radius: {RADIUS["sm"]} !important;
            font-weight: 600 !important;
            padding: 0.55rem 1.4rem !important;
        }}

        .stFormSubmitButton button:hover {{
            background: {c["primary_light"]} !important;
        }}

        div[data-testid="stExpander"] {{
            background: {c["surface"]};
            border: 1px solid {c["border"]};
            border-radius: {RADIUS["md"]};
            overflow: hidden;
        }}

        .lgpd-hero {{
            background: linear-gradient(135deg, {c["primary"]} 0%, {c["primary_light"]} 100%);
            color: #F8FAFC;
            border-radius: {RADIUS["lg"]};
            padding: 1.6rem 1.75rem;
            margin-bottom: 1rem;
            box-shadow: {SHADOW};
        }}

        .lgpd-hero h1 {{
            color: #FFFFFF !important;
            font-size: 1.85rem !important;
            margin: 0 0 0.35rem 0 !important;
            line-height: 1.2 !important;
        }}

        .lgpd-hero p {{
            margin: 0;
            color: #CBD5E1;
            font-size: 0.95rem;
            line-height: 1.5;
        }}

        .lgpd-badge-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin-top: 0.9rem;
        }}

        .lgpd-badge {{
            display: inline-flex;
            align-items: center;
            gap: 0.3rem;
            padding: 0.28rem 0.65rem;
            border-radius: {RADIUS["pill"]};
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.02em;
            text-transform: uppercase;
        }}

        .lgpd-badge--rag {{ background: {c["accent_soft"]}; color: {c["accent"]}; }}
        .lgpd-badge--cache {{ background: {c["cache_bg"]}; color: {c["cache_text"]}; }}
        .lgpd-badge--tool {{ background: #FEF3C7; color: #B45309; }}

        .lgpd-card {{
            background: {c["surface"]};
            border: 1px solid {c["border"]};
            border-radius: {RADIUS["md"]};
            padding: 1rem 1.15rem;
            margin: 0.65rem 0 0.25rem;
            box-shadow: 0 1px 2px rgba(15,43,70,0.04);
        }}

        .lgpd-card__label {{
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            margin-bottom: 0.45rem;
        }}

        .lgpd-card--success .lgpd-card__label {{ color: {c["success_text"]}; }}
        .lgpd-card--info .lgpd-card__label {{ color: {c["info_text"]}; }}
        .lgpd-card--warning .lgpd-card__label {{ color: {c["warning_text"]}; }}
        .lgpd-card--cache .lgpd-card__label {{ color: {c["cache_text"]}; }}

        .lgpd-card__body {{
            color: {c["text"]};
            font-size: 0.96rem;
            line-height: 1.65;
            white-space: pre-wrap;
        }}

        .lgpd-disclaimer {{
            background: {c["warning_bg"]};
            border-left: 4px solid #F59E0B;
            border-radius: 0 {RADIUS["sm"]} {RADIUS["sm"]} 0;
            padding: 0.75rem 1rem;
            margin: 0.75rem 0 1.1rem;
            color: {c["warning_text"]};
            font-size: 0.86rem;
            line-height: 1.5;
        }}

        .lgpd-source-list {{
            list-style: none;
            padding: 0;
            margin: 0;
        }}

        .lgpd-source-list li {{
            background: {c["surface_muted"]};
            border: 1px solid {c["border"]};
            border-radius: {RADIUS["sm"]};
            padding: 0.45rem 0.7rem;
            margin-bottom: 0.4rem;
            font-family: ui-monospace, monospace;
            font-size: 0.8rem;
            color: {c["text_muted"]};
        }}

        .lgpd-footer {{
            text-align: center;
            color: {c["text_muted"]};
            font-size: 0.78rem;
            padding: 1rem 0 0.5rem;
        }}

        #MainMenu, footer, header {{
            visibility: hidden;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
        <div class="lgpd-hero">
            <h1>⚖️ Assistente LGPD</h1>
            <p>Consultas informacionais sobre a Lei 13.709/2018 com RAG, citação de fontes
            e otimização de custo via cache e model routing.</p>
            <div class="lgpd-badge-row">
                <span class="lgpd-badge lgpd-badge--rag">RAG</span>
                <span class="lgpd-badge lgpd-badge--cache">Cache</span>
                <span class="lgpd-badge lgpd-badge--tool">cite_article</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_disclaimer() -> None:
    st.markdown(
        """
        <div class="lgpd-disclaimer">
            <strong>Aviso legal:</strong> uso educacional e informacional.
            Não constitui parecer jurídico — consulte um profissional qualificado
            para decisões de compliance.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_card(
    kind: StatusKind,
    label: str,
    body: str,
    *,
    mono: bool = False,
) -> None:
    kind_class = {
        "corpus": "success",
        "general": "info",
        "cache_exact": "cache",
        "cache_semantic": "cache",
        "routing": "info",
        "warning": "warning",
        "error": "warning",
    }.get(kind, "info")
    font = "font-family: ui-monospace, monospace;" if mono else ""
    st.markdown(
        f"""
        <div class="lgpd-card lgpd-card--{kind_class}">
            <div class="lgpd-card__label">{label}</div>
            <div class="lgpd-card__body" style="{font}">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sources(sources: list[tuple[str, int]]) -> None:
    if not sources:
        return
    items = "".join(
        f'<li><span>{source}</span> · página {page}</li>' for source, page in sources
    )
    with st.expander("Fontes citadas no corpus", expanded=False):
        st.markdown(f'<ul class="lgpd-source-list">{items}</ul>', unsafe_allow_html=True)


def render_routing_badge(complexity: str, model: str, reason: str) -> None:
    render_status_card(
        "routing",
        f"Routing · {complexity}",
        f"Modelo: {model}\n{reason}",
    )


def render_footer() -> None:
    st.markdown(
        """
        <div class="lgpd-footer">
            Projeto de portfólio — LLM + RAG + tool-use + cache semântico + model routing<br>
            <a href="https://github.com/douglaslpo/Projeto-SIDI" target="_blank">GitHub</a>
        </div>
        """,
        unsafe_allow_html=True,
    )

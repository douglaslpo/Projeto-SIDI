"""Function-calling / tool-use — registro de tools usadas pelo agente."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

from pypdf import PdfReader

CORPUS_DIR = Path("data/corpus")

def _extract_pdf_text(pdf_path: Path) -> list[tuple[int, str]]:
    pages: list[tuple[int, str]] = []
    reader = PdfReader(str(pdf_path))
    for page_idx, page in enumerate(reader.pages, start=1):
        try:
            text = (page.extract_text() or "").strip()
        except Exception:
            text = ""
        if text:
            pages.append((page_idx, text))
    return pages


def _find_article_in_text(text: str, article_number: int) -> str | None:
    """Localiza trecho do artigo solicitado dentro de um bloco de texto."""
    markers = [
        rf"(?:Art\.?\s*{article_number}(?:º|o|\.)?)",
        rf"(?:Artigo\s+{article_number})",
    ]
    start = -1
    for pattern in markers:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            start = match.start()
            break
    if start < 0:
        return None

    next_article = re.search(
        r"(?:Art\.?\s*\d+(?:º|o|\.)?|Artigo\s+\d+)",
        text[start + 1 :],
        flags=re.IGNORECASE,
    )
    end = start + 1 + next_article.start() if next_article else len(text)
    snippet = text[start:end].strip()
    return snippet[:1200] if snippet else None


def cite_article(article_number: int, corpus_dir: str | Path = CORPUS_DIR) -> str:
    """Retorna trechos do corpus relacionados ao artigo informado da LGPD."""
    if article_number < 1:
        return "Artigo não encontrado no corpus local."

    corpus_path = Path(corpus_dir)
    pdf_files = sorted(corpus_path.glob("*.pdf"))
    if not pdf_files:
        return (
            "Corpus local vazio. Adicione PDFs da LGPD em data/corpus/ "
            "ou execute scripts/build_corpus_pdf.py."
        )

    for pdf_path in pdf_files:
        for page_num, page_text in _extract_pdf_text(pdf_path):
            snippet = _find_article_in_text(page_text, article_number)
            if snippet:
                return (
                    f"Fonte: {pdf_path.name} (página {page_num})\n\n"
                    f"{snippet}\n\n"
                    "Nota: trecho extraído automaticamente do corpus local. "
                    "Uso educacional e informacional; não constitui parecer jurídico."
                )

    return "Artigo não encontrado no corpus local."


TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "cite_article",
            "description": (
                "Localiza e retorna trecho do corpus local referente a um artigo "
                "específico da LGPD (Lei 13.709/2018). Use quando o usuário pedir "
                "citação literal de artigo."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "article_number": {
                        "type": "integer",
                        "description": (
                            "Número do artigo da LGPD que deve ser localizado no corpus."
                        ),
                    }
                },
                "required": ["article_number"],
            },
        },
    }
]


TOOL_REGISTRY: dict[str, Callable[..., str]] = {
    "cite_article": cite_article,
}


def run_tool_call(name: str, arguments_json: str) -> str:
    """Executa uma tool call e retorna o resultado como string."""
    if name not in TOOL_REGISTRY:
        return f"ERROR: tool '{name}' não registrada"
    try:
        kwargs = json.loads(arguments_json)
        return TOOL_REGISTRY[name](**kwargs)
    except Exception as exc:
        return f"ERROR ao executar {name}: {exc}"

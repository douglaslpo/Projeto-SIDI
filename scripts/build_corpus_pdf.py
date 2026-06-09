"""Gera PDF do corpus LGPD a partir do texto oficial compilado (Planalto).

Uso:
    uv run python scripts/build_corpus_pdf.py
    uv run python scripts/build_corpus_pdf.py --from-text caminho/para/texto.txt
"""

from __future__ import annotations

import argparse
import html
import re
import textwrap
import urllib.request
from pathlib import Path

from fpdf import FPDF

PLANALTO_URL = (
    "https://www.planalto.gov.br/ccivil_03/_ato2015-2018/2018/lei/L13709compilado.htm"
)
OUTPUT = Path("data/corpus/LEI_13709_LGPD.pdf")


def _html_to_text(page: str) -> str:
    page = re.sub(r"(?is)<script.*?>.*?</script>", " ", page)
    page = re.sub(r"(?is)<style.*?>.*?</style>", " ", page)
    page = re.sub(r"(?i)<br\s*/?>", "\n", page)
    page = re.sub(r"(?i)</p>", "\n\n", page)
    text = re.sub(r"<[^>]+>", " ", page)
    text = html.unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _fetch_lgpd_text() -> str:
    request = urllib.request.Request(
        PLANALTO_URL,
        headers={"User-Agent": "Mozilla/5.0 (compatible; LGPD-Corpus-Bot/1.0)"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        raw = response.read()

    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            page = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        page = raw.decode("utf-8", errors="replace")

    return _html_to_text(page)


def _load_text_file(path: Path) -> str:
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            return path.read_text(encoding=encoding).strip()
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace").strip()


class CorpusPDF(FPDF):
    def footer(self) -> None:
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Pagina {self.page_no()}", align="C")


def _sanitize_for_pdf(text: str) -> str:
    """Remove caracteres não suportados por Helvetica (latin-1)."""
    normalized = text.replace("\ufffd", "")
    return normalized.encode("latin-1", "replace").decode("latin-1")


def build_pdf(text: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    text = _sanitize_for_pdf(text)
    pdf = CorpusPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=10)

    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        wrapped = textwrap.fill(paragraph, width=95)
        pdf.multi_cell(0, 5, wrapped)
        pdf.ln(2)

    pdf.output(str(output))


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera PDF do corpus LGPD.")
    parser.add_argument(
        "--from-text",
        type=Path,
        help="Caminho para arquivo .txt/.htm com texto da LGPD (fallback offline).",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT, help="Caminho do PDF de saída.")
    args = parser.parse_args()

    if args.from_text:
        print(f"Lendo texto local de {args.from_text} ...")
        raw = _load_text_file(args.from_text)
        text = _html_to_text(raw) if "<" in raw[:500] else raw
    else:
        print(f"Baixando texto de {PLANALTO_URL} ...")
        try:
            text = _fetch_lgpd_text()
        except Exception as exc:
            raise RuntimeError(
                f"Falha ao baixar LGPD: {exc}. "
                "Use --from-text com arquivo local ou adicione PDF manualmente em data/corpus/."
            ) from exc

    if len(text) < 5000:
        raise RuntimeError("Texto LGPD extraído parece incompleto.")

    build_pdf(text, args.output)
    print(f"PDF gerado: {args.output} ({args.output.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()

# Corpus LGPD

Esta pasta contém os PDFs indexados pelo pipeline RAG.

## Conteúdo esperado

- **Lei 13.709/2018 (LGPD)** — texto oficial
- Opcional: guias/orientações da ANPD (PDFs com licença compatível)

## Gerar corpus automaticamente

Se a pasta estiver vazia, execute na raiz do projeto:

```bash
uv run python scripts/build_corpus_pdf.py
```

Isso baixa o texto compilado do [Planalto](https://www.planalto.gov.br/ccivil_03/_ato2015-2018/2018/lei/L13709compilado.htm) e gera `LEI_13709_LGPD.pdf`.

## Adicionar PDFs manualmente

1. Copie PDFs para `data/corpus/`
2. Apague `data/chroma/` para forçar reindexação
3. Reinicie o Streamlit

## Restrições

- Pelo menos 1 PDF com texto extraível
- Recomendado ≥10 páginas (rubrica)
- PDFs escaneados precisam de OCR (`ocrmypdf`) antes do ingest

## Reindexar

```bash
rm -rf data/chroma/
uv run streamlit run src/ui/streamlit_app.py
```

# Roteiro do vídeo demo (≤3 minutos)

## 0:00–0:20 — Contexto
- Apresente o problema: consultar LGPD rapidamente em linguagem natural.
- Mostre a URL pública da demo.

## 0:20–0:50 — Pergunta RAG
- Digite: *"Posso armazenar CPF de usuários? Em quais condições?"*
- Destaque: routing (simple/complex), resposta ancorada no corpus, fontes citadas.

## 0:50–1:20 — Segunda pergunta + cache
- Pergunte: *"O que a LGPD diz sobre consentimento?"*
- Repita a mesma pergunta → mostre **cache hit (exact)**.

## 1:20–1:50 — Tool cite_article
- Abra expander **Consultar artigo da LGPD**.
- Busque artigo 18 (direitos do titular).
- Explique: tool determinística, sem alucinação.

## 1:50–2:20 — Arquitetura rápida
- Mostre sidebar (chunks indexados, métricas de cache).
- Mencione: RAG + semantic cache + model routing.

## 2:20–2:50 — Limitações
- "Uso educacional, não substitui advogado."
- Corpus fixo; respostas só com base no material indexado.

## 2:50–3:00 — Encerramento
- Link do GitHub + convite para testar.

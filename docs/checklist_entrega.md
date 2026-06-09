# Checklist final de entrega

## Funcionalidade
- [ ] App abre no Streamlit sem crash
- [ ] Corpus indexado (`data/corpus/*.pdf`)
- [ ] `retrieve()` retorna chunks
- [ ] `answer()` retorna resposta + fontes
- [ ] `cite_article()` funciona
- [ ] Semantic cache implementado
- [ ] Model routing visível na UI
- [ ] Botão "Limpar caches" funciona

## Rubrica técnica
- [ ] Sem `NotImplementedError` nos fluxos principais
- [ ] Tratamento de erro (API, corpus vazio)
- [ ] Logs com `trace_id`
- [ ] Testes: `uv run pytest tests/test_smoke.py -v`

## Documentação
- [ ] README profissional preenchido
- [ ] `.env.example` atualizado
- [ ] Aviso legal na UI e README
- [ ] Tabela cost/latency (mesmo que "a medir")

## Deploy e entrega Forms
- [ ] Repositório GitHub **público**
- [ ] Demo **pública** (Streamlit Cloud / HF / Fly.io)
- [ ] Vídeo demo **≤3 min** (YouTube/Loom/Drive)
- [ ] 3 URLs enviadas no formulário da disciplina

## Segurança
- [ ] `.env` no `.gitignore` e **não commitado**
- [ ] API keys apenas em secrets do deploy
- [ ] Sem conteúdo jurídico inventado

## Validação manual (5 perguntas)
- [ ] CPF / armazenamento
- [ ] Consentimento
- [ ] Direitos do titular
- [ ] Base legal
- [ ] Retenção de dados

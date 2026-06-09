# Deploy no Streamlit Community Cloud

## Repositório

**GitHub:** https://github.com/douglaslpo/Projeto-SIDI

## Pré-requisitos

- Repositório público (já publicado)
- Conta em [share.streamlit.io](https://share.streamlit.io) (login com GitHub)
- API key do Gemini

## Passo a passo (5 min)

### 1. Criar app

1. Acesse https://share.streamlit.io
2. Clique **Create app**
3. Preencha:
   - **Repository:** `douglaslpo/Projeto-SIDI`
   - **Branch:** `main`
   - **Main file path:** `src/ui/streamlit_app.py`
   - **App URL (opcional):** `projeto-sidi` → URL final: `https://projeto-sidi.streamlit.app`

### 2. Configurar Secrets

No app → **Settings** → **Secrets** → cole (use sua chave real):

```toml
GEMINI_API_KEY = "sua-chave-aqui"
LLM_MODEL = "gemini-2.5-flash-lite"
EMBED_MODEL = "gemini-embedding-001"
CHEAP_MODEL = "gemini-2.5-flash-lite"
PREMIUM_MODEL = "gemini-2.5-flash"
MAX_CORPUS_DISTANCE = "0.55"
```

Referência: [`.streamlit/secrets.toml.example`](../.streamlit/secrets.toml.example)

> **Importante:** `gemini-2.5-pro` não funciona no free tier (cota 0). Use `gemini-2.5-flash`.

### 3. Deploy

1. Clique **Deploy** (ou **Reboot app** após salvar secrets)
2. Aguarde build (`requirements.txt` instala dependências)
3. Primeira carga indexa o PDF da LGPD (~1–3 min)

### 4. Validar demo

| Teste | Esperado |
|---|---|
| *"Posso armazenar CPF de usuários?"* | Resposta corpus + fontes |
| *"O que a LGPD diz sobre consentimento?"* | Routing complex + resposta |
| Repetir mesma pergunta | Cache hit (exact) |
| *"O que é LAI?"* | Resposta geral via API (fora do corpus) |
| Expander artigo 5 | Tool `cite_article` retorna trecho |

### 5. Atualizar README

Após deploy, copie a URL para `README.md`:

```markdown
**Live demo:** https://projeto-sidi.streamlit.app
```

Commit e push:

```powershell
git add README.md
git commit -m "docs: adicionar URL da demo Streamlit"
git push
```

## Troubleshooting

| Problema | Solução |
|---|---|
| App crash ao abrir | Verifique Secrets (`GEMINI_API_KEY`) |
| `Configure API key` | Secrets não salvos ou nome errado |
| Indexação lenta | Normal no cold start; Chroma em `/tmp` é efêmero |
| Erro 429 / cota | Aguarde 1 min; use modelos flash, não pro |
| Corpus vazio | Confirme `data/corpus/LEI_13709_LGPD.pdf` no repo |

## Alternativas

- **HuggingFace Spaces:** SDK Streamlit + secrets equivalentes
- **Fly.io:** Dockerfile + variáveis de ambiente

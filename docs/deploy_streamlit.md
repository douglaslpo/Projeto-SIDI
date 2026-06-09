# Deploy no Streamlit Community Cloud

## Pré-requisitos

- Repositório **público** no GitHub com este código
- Conta em [share.streamlit.io](https://share.streamlit.io)
- API key do Gemini ou OpenAI

## Passo a passo

### 1. Publicar no GitHub

```powershell
cd template-portfolio
git init
git add .
git commit -m "Assistente LGPD com RAG — projeto de portfólio"
git branch -M main
git remote add origin https://github.com/SEU_USUARIO/assistente-lgpd-rag.git
git push -u origin main
```

Confirme que `data/corpus/LEI_13709_LGPD.pdf` está no repositório (necessário para indexação).

### 2. Criar app no Streamlit Cloud

1. Acesse [share.streamlit.io](https://share.streamlit.io) → **Create app**
2. Conecte sua conta GitHub
3. Selecione o repositório e branch `main`
4. **Main file path:** `src/ui/streamlit_app.py`
5. **App URL:** escolha um slug (ex.: `assistente-lgpd-rag`)

### 3. Configurar Secrets

No painel do app → **Settings** → **Secrets**, cole:

```toml
GEMINI_API_KEY = "sua-chave-aqui"
LLM_MODEL = "gemini-2.5-flash-lite"
EMBED_MODEL = "gemini-embedding-001"
CHEAP_MODEL = "gemini-2.5-flash-lite"
PREMIUM_MODEL = "gemini-2.5-pro"
```

Para OpenAI, substitua por `OPENAI_API_KEY` e modelos compatíveis.

### 4. Deploy e validação

1. Clique **Deploy** (ou aguarde redeploy automático após push)
2. Primeira execução indexa o corpus (pode levar 1–3 min)
3. Teste:
   - Pergunta RAG com fontes
   - Cache hit ao repetir pergunta
   - Tool `cite_article` no expander
   - Routing visível na barra de info

### 5. Troubleshooting

| Problema | Solução |
|---|---|
| App crash na inicialização | Verifique secrets e se o PDF está no repo |
| `RuntimeError: Configure API key` | Secrets não configurados ou nome errado |
| Indexação lenta | Normal na 1ª execução; Chroma persiste em `/tmp` no free tier |
| Corpus vazio | Confirme `data/corpus/*.pdf` commitado |

> **Nota:** No Streamlit Cloud free tier, `data/chroma/` é efêmero — o app reindexa a cada cold start. Isso é aceitável para demo.

## Alternativas

- **HuggingFace Spaces:** use `streamlit` SDK + secrets equivalentes
- **Fly.io:** containerize com `Dockerfile` + variáveis de ambiente

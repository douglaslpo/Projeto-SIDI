"""RAG pipeline — chunk, embed, index, retrieve, generate."""

from __future__ import annotations

import hashlib
import logging
import os
import time
import unicodedata
from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import APIStatusError, OpenAI, RateLimitError
from pypdf import PdfReader

logger = logging.getLogger("portfolio")


def _make_client() -> tuple[OpenAI, str | None]:
    """Inicializa cliente OpenAI-compatible conforme provider escolhido no .env."""
    if "GEMINI_API_KEY" in os.environ:
        client = OpenAI(
            api_key=os.environ["GEMINI_API_KEY"],
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        embed_api_base = "https://generativelanguage.googleapis.com/v1beta/openai/"
    elif "OPENAI_API_KEY" in os.environ:
        client = OpenAI()
        embed_api_base = None
    else:
        raise RuntimeError("Configure GEMINI_API_KEY ou OPENAI_API_KEY no .env")
    return client, embed_api_base


def _stable_chunk_id(source: str, page: int, text: str) -> str:
    """Gera id estável para evitar duplicação entre reindexações."""
    payload = f"{source}|{page}|{text[:200]}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


# Gemini limita embeddings a 100 textos por batch via API OpenAI-compatible.
EMBED_BATCH_SIZE = 100
EMBED_BATCH_RETRY_SECONDS = 30
EMBED_MAX_RETRIES = 3


def _is_quota_error(exc: Exception) -> bool:
    if isinstance(exc, RateLimitError):
        return True
    if isinstance(exc, APIStatusError) and exc.status_code == 429:
        return True
    lowered = str(exc).lower()
    return "429" in lowered or "quota" in lowered or "resource_exhausted" in lowered


def _friendly_llm_error(exc: Exception) -> str:
    if _is_quota_error(exc):
        return (
            "Cota da API excedida para o modelo selecionado. "
            "O app tentará usar o modelo econômico automaticamente; "
            "se persistir, aguarde 1 minuto ou ajuste PREMIUM_MODEL no .env "
            "(ex.: gemini-2.5-flash em vez de gemini-2.5-pro no free tier)."
        )
    return f"Erro ao gerar resposta com o LLM: {exc}"


def _expand_retrieval_query(query: str) -> str:
    """Enriquece busca só para perguntas claramente sobre LGPD/dados pessoais."""
    return (
        f"{query}\n"
        "LGPD: dados pessoais, dado pessoal, tratamento, base legal, "
        "consentimento, armazenamento, titular, controlador, finalidade, segurança."
    )


_LGPD_HINTS = (
    "lgpd",
    "dado pessoal",
    "dados pessoais",
    "cpf",
    "consentimento",
    "titular",
    "controlador",
    "base legal",
    "privacidade",
    "anpd",
    "tratamento",
    "retencao",
    "retenção",
    "armazenar",
    "artigo",
    "art.",
)


def _looks_like_lgpd_question(query: str) -> bool:
    normalized = query.lower()
    normalized = "".join(
        ch for ch in unicodedata.normalize("NFD", normalized) if unicodedata.category(ch) != "Mn"
    )
    return any(hint in normalized for hint in _LGPD_HINTS)


def _corpus_is_relevant(hits: list[dict[str, Any]]) -> bool:
    if not hits:
        return False
    max_dist = float(os.environ.get("MAX_CORPUS_DISTANCE", "0.55"))
    return float(hits[0]["distance"]) <= max_dist


def _is_not_found_answer(text: str) -> bool:
    normalized = text.lower().strip().strip(".")
    return normalized in {
        "não encontrado no corpus",
        "nao encontrado no corpus",
        "não encontrei informação relevante no corpus local para responder esta pergunta",
    } or normalized.startswith("não encontrei informação relevante no corpus")


class RAGPipeline:
    """Pipeline RAG end-to-end com Chroma local."""

    def __init__(
        self,
        corpus_dir: str = "data/corpus",
        persist_dir: str = "data/chroma",
        collection_name: str = "docs",
        llm_model: str | None = None,
        embed_model: str | None = None,
    ) -> None:
        self.client, embed_api_base = _make_client()
        self.llm_model = llm_model or os.environ.get("LLM_MODEL", "gemini-2.5-flash-lite")
        self.embed_model = embed_model or os.environ.get("EMBED_MODEL", "gemini-embedding-001")

        embed_kwargs: dict[str, Any] = {
            "api_key": os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY"),
            "model_name": self.embed_model,
        }
        if embed_api_base:
            embed_kwargs["api_base"] = embed_api_base
        self.embed_fn = OpenAIEmbeddingFunction(**embed_kwargs)

        self.corpus_dir = Path(corpus_dir)
        self.persist_dir = persist_dir
        self.collection_name = collection_name

        chroma = chromadb.PersistentClient(path=persist_dir)
        self.collection = chroma.get_or_create_collection(
            name=collection_name, embedding_function=self.embed_fn
        )

    def _add_chunks_in_batches(self, chunks: list[dict[str, Any]]) -> None:
        """Indexa chunks respeitando limite de batch do Gemini e rate limit."""
        for start in range(0, len(chunks), EMBED_BATCH_SIZE):
            batch = chunks[start : start + EMBED_BATCH_SIZE]
            for attempt in range(EMBED_MAX_RETRIES):
                try:
                    self.collection.add(
                        ids=[c["id"] for c in batch],
                        documents=[c["text"] for c in batch],
                        metadatas=[{"source": c["source"], "page": c["page"]} for c in batch],
                    )
                    break
                except RateLimitError:
                    if attempt >= EMBED_MAX_RETRIES - 1:
                        raise
                    time.sleep(EMBED_BATCH_RETRY_SECONDS * (attempt + 1))

            has_more = start + EMBED_BATCH_SIZE < len(chunks)
            if has_more:
                time.sleep(EMBED_BATCH_RETRY_SECONDS)

    def ingest_and_index(self) -> int:
        """Lê PDFs de `corpus_dir`, faz chunking e indexa em Chroma."""
        pdf_files = sorted(self.corpus_dir.glob("*.pdf"))
        if not pdf_files:
            raise FileNotFoundError(
                f"Nenhum PDF encontrado em {self.corpus_dir}. "
                "Adicione documentos LGPD/ANPD ou rode scripts/build_corpus_pdf.py."
            )

        docs: list[dict[str, Any]] = []
        for pdf_path in pdf_files:
            reader = PdfReader(str(pdf_path))
            for page_idx, page in enumerate(reader.pages, start=1):
                try:
                    text = (page.extract_text() or "").strip()
                except Exception:
                    continue
                if not text:
                    continue
                docs.append({"text": text, "source": pdf_path.name, "page": page_idx})

        if not docs:
            raise ValueError(
                f"Nenhum texto extraível nos PDFs de {self.corpus_dir}. "
                "Verifique se os arquivos não são apenas imagens escaneadas."
            )

        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks: list[dict[str, Any]] = []
        for doc in docs:
            for piece in splitter.split_text(doc["text"]):
                piece = piece.strip()
                if not piece:
                    continue
                chunks.append(
                    {
                        "id": _stable_chunk_id(doc["source"], doc["page"], piece),
                        "text": piece,
                        "source": doc["source"],
                        "page": doc["page"],
                    }
                )

        if not chunks:
            raise ValueError("Chunking não produziu trechos indexáveis.")

        existing_ids = set()
        if self.collection.count() > 0:
            existing = self.collection.get(include=[])
            existing_ids = set(existing.get("ids", []))

        new_chunks = [c for c in chunks if c["id"] not in existing_ids]
        if new_chunks:
            self._add_chunks_in_batches(new_chunks)

        return self.collection.count()

    def retrieve(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """Busca top-k chunks similares à query."""
        if self.collection.count() == 0:
            return []

        search_query = _expand_retrieval_query(query) if _looks_like_lgpd_question(query) else query
        n_results = min(k, self.collection.count())
        for attempt in range(EMBED_MAX_RETRIES):
            try:
                results = self.collection.query(
                    query_texts=[search_query], n_results=n_results
                )
                break
            except RateLimitError:
                if attempt >= EMBED_MAX_RETRIES - 1:
                    logger.warning("retrieve: rate limit ao embedar query")
                    return []
                time.sleep(EMBED_BATCH_RETRY_SECONDS * (attempt + 1))
            except Exception as exc:
                logger.warning("retrieve falhou: %s", exc)
                return []
        else:
            return []

        documents = results.get("documents") or [[]]
        metadatas = results.get("metadatas") or [[]]
        distances = results.get("distances") or [[]]

        hits: list[dict[str, Any]] = []
        for text, meta, distance in zip(documents[0], metadatas[0], distances[0], strict=False):
            if not text:
                continue
            meta = meta or {}
            hits.append(
                {
                    "text": text,
                    "source": meta.get("source", "desconhecido"),
                    "page": meta.get("page", 0),
                    "distance": distance,
                }
            )
        return hits

    def _call_llm(self, prompt: str, model: str) -> str:
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return (response.choices[0].message.content or "").strip()

    def answer_general(self, question: str, model: str | None = None) -> dict[str, Any]:
        """Resposta via API quando a pergunta está fora do corpus LGPD."""
        llm_model = model or self.llm_model
        cheap_model = os.environ.get("CHEAP_MODEL", self.llm_model)
        models_to_try = [llm_model]
        if llm_model != cheap_model:
            models_to_try.append(cheap_model)

        prompt = GENERAL_PROMPT_TEMPLATE.format(question=question)
        for idx, current_model in enumerate(models_to_try):
            try:
                answer_text = self._call_llm(prompt, current_model)
                return {
                    "answer": answer_text,
                    "sources": [],
                    "model_used": current_model,
                    "fallback_used": idx > 0,
                    "mode": "general",
                }
            except Exception as exc:
                if idx == 0 and _is_quota_error(exc) and len(models_to_try) > 1:
                    continue
                return {
                    "answer": _friendly_llm_error(exc),
                    "sources": [],
                    "model_used": current_model,
                    "fallback_used": False,
                    "mode": "general",
                }
        return {
            "answer": "Não foi possível consultar a API no momento.",
            "sources": [],
            "model_used": llm_model,
            "fallback_used": False,
            "mode": "general",
        }

    def answer(
        self, question: str, k: int = 5, model: str | None = None
    ) -> dict[str, Any]:
        """Pipeline completo: retrieve + augment + generate (+ fallback API geral)."""
        hits = self.retrieve(question, k=k)
        llm_model = model or self.llm_model

        if not hits or not _corpus_is_relevant(hits):
            general = self.answer_general(question, model=model)
            general["corpus_relevant"] = False
            return general

        context_parts = [f"[{h['source']}:{h['page']}]\n{h['text']}" for h in hits]
        context = "\n\n---\n\n".join(context_parts)
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)

        cheap_model = os.environ.get("CHEAP_MODEL", self.llm_model)
        models_to_try = [llm_model]
        if llm_model != cheap_model:
            models_to_try.append(cheap_model)

        answer_text = ""
        model_used = llm_model
        fallback_used = False
        last_error: Exception | None = None

        for idx, current_model in enumerate(models_to_try):
            try:
                answer_text = self._call_llm(prompt, current_model)
                model_used = current_model
                fallback_used = idx > 0
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                if idx == 0 and _is_quota_error(exc) and len(models_to_try) > 1:
                    logger.warning(
                        "answer: cota esgotada em %s — fallback para %s",
                        current_model,
                        models_to_try[1],
                    )
                    continue
                return {
                    "answer": _friendly_llm_error(exc),
                    "sources": [(h["source"], h["page"]) for h in hits],
                    "model_used": current_model,
                    "fallback_used": False,
                    "mode": "corpus",
                    "corpus_relevant": True,
                }

        if last_error is not None:
            return {
                "answer": _friendly_llm_error(last_error),
                "sources": [(h["source"], h["page"]) for h in hits],
                "model_used": llm_model,
                "fallback_used": False,
                "mode": "corpus",
                "corpus_relevant": True,
            }

        if _is_not_found_answer(answer_text):
            retry_prompt = RETRY_PROMPT_TEMPLATE.format(context=context, question=question)
            try:
                retry_answer = self._call_llm(retry_prompt, model_used)
                if retry_answer and not _is_not_found_answer(retry_answer):
                    answer_text = retry_answer
            except Exception as exc:
                logger.warning("answer: retry pós 'não encontrado' falhou: %s", exc)

        if _is_not_found_answer(answer_text):
            general = self.answer_general(question, model=model_used)
            general["corpus_relevant"] = False
            return general

        if not answer_text:
            answer_text = "Não encontrado no corpus."

        sources = [(h["source"], h["page"]) for h in hits]
        return {
            "answer": answer_text,
            "sources": sources,
            "model_used": model_used,
            "fallback_used": fallback_used,
            "mode": "corpus",
            "corpus_relevant": True,
        }


GENERAL_PROMPT_TEMPLATE = """Você é um assistente educacional. A pergunta abaixo NÃO foi encontrada
no corpus local da LGPD (Lei 13.709/2018) indexado neste app.

Responda usando conhecimento geral via API, de forma clara e objetiva.
Regras:
- Comece avisando: "Esta resposta não veio do corpus LGPD local — consulta geral via API."
- Se for legislação (ex.: LAI, Marco Civil), explique o básico e indique fontes oficiais quando souber.
- Não invente artigos ou números de lei se não tiver certeza — diga para consultar o texto oficial.
- Uso educacional; não substitui parecer jurídico.

PERGUNTA: {question}

RESPOSTA:"""


PROMPT_TEMPLATE = """Você é um assistente de compliance LGPD para desenvolvedores e gestores.
Responda APENAS com base no contexto abaixo, de forma clara e objetiva.

Regras importantes:
- CPF, e-mail, telefone e IP são exemplos de "dados pessoais" quando o contexto define esse termo.
- Se o contexto trouxer definições, bases legais, consentimento, finalidade, segurança ou retenção
  aplicáveis à pergunta, USE-OS — mesmo que o termo exato da pergunta (ex.: "CPF") não apareça literalmente.
- Diga "Não encontrado no corpus" SOMENTE se nenhum trecho do contexto for aplicável.
- Sempre cite a fonte usando o formato [arquivo:pagina].
- Não invente artigos, bases legais ou interpretações jurídicas além do contexto.
- Uso educacional e informacional; não substitui parecer jurídico.

CONTEXTO:
{context}

PERGUNTA: {question}

RESPOSTA:"""

RETRY_PROMPT_TEMPLATE = """Você é um assistente LGPD. Trechos do corpus foram recuperados e estão no contexto abaixo.
A pergunta do usuário DEVE ser respondida com base nesses trechos.

Instruções:
- CPF é dado pessoal (informação que identifica pessoa natural). Relacione com definições e artigos do contexto.
- Explique condições de tratamento/armazenamento citando [arquivo:pagina].
- NÃO responda "Não encontrado no corpus" se houver qualquer trecho aplicável.
- Se faltar detalhe específico sobre CPF, diga o que a LGPD exige em geral para dados pessoais
  e indique que o corpus local não detalha CPF nominalmente.

CONTEXTO:
{context}

PERGUNTA: {question}

RESPOSTA:"""


def build_rag_pipeline(corpus_dir: str = "data/corpus") -> RAGPipeline:
    """Factory: cria pipeline e garante corpus indexado (idempotente)."""
    pipeline = RAGPipeline(corpus_dir=corpus_dir)
    pipeline.ingest_and_index()
    return pipeline

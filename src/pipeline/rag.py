"""RAG pipeline — chunk, embed, index, retrieve, generate."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from pypdf import PdfReader


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

    def ingest_and_index(self) -> int:
        """Lê PDFs de `corpus_dir`, faz chunking e indexa em Chroma."""
        if self.collection.count() > 0:
            return self.collection.count()

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
            self.collection.add(
                ids=[c["id"] for c in new_chunks],
                documents=[c["text"] for c in new_chunks],
                metadatas=[{"source": c["source"], "page": c["page"]} for c in new_chunks],
            )

        return self.collection.count()

    def retrieve(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """Busca top-k chunks similares à query."""
        if self.collection.count() == 0:
            return []

        try:
            results = self.collection.query(query_texts=[query], n_results=min(k, self.collection.count()))
        except Exception:
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

    def answer(
        self, question: str, k: int = 5, model: str | None = None
    ) -> dict[str, Any]:
        """Pipeline completo: retrieve + augment + generate."""
        hits = self.retrieve(question, k=k)
        llm_model = model or self.llm_model
        if not hits:
            return {
                "answer": (
                    "Não encontrei informação relevante no corpus local para responder esta pergunta. "
                    "Tente reformular ou consulte um artigo específico da LGPD."
                ),
                "sources": [],
            }

        context_parts = [f"[{h['source']}:{h['page']}]\n{h['text']}" for h in hits]
        context = "\n\n---\n\n".join(context_parts)
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)

        try:
            response = self.client.chat.completions.create(
                model=llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            answer_text = (response.choices[0].message.content or "").strip()
        except Exception as exc:
            return {
                "answer": f"Erro ao gerar resposta com o LLM: {exc}",
                "sources": [(h["source"], h["page"]) for h in hits],
            }

        if not answer_text:
            answer_text = "Não encontrado no corpus."

        sources = [(h["source"], h["page"]) for h in hits]
        return {"answer": answer_text, "sources": sources}


PROMPT_TEMPLATE = """Você é um assistente de compliance LGPD para desenvolvedores e gestores.
Responda APENAS com base no contexto abaixo, de forma clara e objetiva.
Se a informação não estiver no contexto, diga explicitamente "Não encontrado no corpus".
Sempre cite a fonte usando o formato [arquivo:pagina].
Não invente artigos, bases legais ou interpretações jurídicas.
Este conteúdo é educacional e informacional; não substitui parecer jurídico.

CONTEXTO:
{context}

PERGUNTA: {question}

RESPOSTA:"""


def build_rag_pipeline(corpus_dir: str = "data/corpus") -> RAGPipeline:
    """Factory: cria pipeline e indexa corpus se ainda não indexado."""
    pipeline = RAGPipeline(corpus_dir=corpus_dir)
    if pipeline.collection.count() == 0:
        pipeline.ingest_and_index()
    return pipeline

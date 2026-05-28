"""
Core RAG pipeline — the system being benchmarked across all 12 layers.
Built on LangChain + FAISS. Swap corpus or retriever without changing the interface.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from src.rag.corpus import get_all_texts

load_dotenv()

# ── System prompts (versioned — used in A/B testing) ─────────────────────────

PROMPT_V1 = """You are a helpful AI assistant specialized in AI, machine learning, and
production systems. Answer questions accurately using only the provided context.
If the context does not contain the answer, say: "I don't have that information in my
knowledge base." Keep answers concise and clear."""

PROMPT_V2 = """You are a precise AI assistant specialized in AI engineering and production
machine learning systems.

Rules you must follow:
1. Answer ONLY using information from the provided context.
2. If the answer is not in the context, respond exactly: "I don't have that information
   in my knowledge base."
3. Be concise — 2-4 sentences unless detail is explicitly requested.
4. Never guess or infer beyond what is explicitly stated in the context.
5. If asked about code, provide concrete examples when the context supports it."""


# ── Query result ─────────────────────────────────────────────────────────────

@dataclass
class QueryResult:
    query: str
    answer: str
    contexts: list[str]
    latency_ms: float
    model: str
    prompt_version: str
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def cost_usd(self) -> float:
        pricing = {
            "gpt-4o-mini": {"input": 0.00015, "output": 0.00060},
            "gpt-4o":      {"input": 0.005,   "output": 0.015},
        }
        p = pricing.get(self.model, {"input": 0.001, "output": 0.002})
        return (self.input_tokens / 1000) * p["input"] + \
               (self.output_tokens / 1000) * p["output"]


# ── RAG pipeline ─────────────────────────────────────────────────────────────

class RAGPipeline:

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        system_prompt: str = PROMPT_V1,
        prompt_version: str = "v1",
        top_k: int = 3,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ):
        self.model = model
        self.system_prompt = system_prompt
        self.prompt_version = prompt_version
        self.top_k = top_k

        self._llm = ChatOpenAI(model=model, temperature=0)
        self._embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self._vectorstore = self._build_index(chunk_size, chunk_overlap)

    def _build_index(self, chunk_size: int, chunk_overlap: int) -> FAISS:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " "],
        )
        texts = get_all_texts()
        chunks = splitter.split_text("\n\n---\n\n".join(texts))
        return FAISS.from_texts(chunks, self._embeddings)

    def retrieve(self, query: str) -> list[str]:
        docs = self._vectorstore.similarity_search(query, k=self.top_k)
        return [doc.page_content for doc in docs]

    def query(self, question: str) -> QueryResult:
        start = time.perf_counter()

        contexts = self.retrieve(question)
        context_text = "\n\n---\n\n".join(contexts)

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(
                content=f"Context:\n{context_text}\n\nQuestion: {question}"
            ),
        ]

        response = self._llm.invoke(messages)
        latency_ms = (time.perf_counter() - start) * 1000

        usage = response.usage_metadata or {}

        return QueryResult(
            query=question,
            answer=response.content,
            contexts=contexts,
            latency_ms=round(latency_ms, 2),
            model=self.model,
            prompt_version=self.prompt_version,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
        )


@lru_cache(maxsize=1)
def get_default_pipeline() -> RAGPipeline:
    """Cached singleton for use in the FastAPI server."""
    return RAGPipeline()

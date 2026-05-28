"""
Production FastAPI server — the AI system being benchmarked.
Exposes /query (async-safe), /health, and /metrics endpoints.
"""

from __future__ import annotations

import asyncio
import os
import time

import psutil
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from src.observability.metrics_exporter import AIMetricsRecorder, start_metrics_server
from src.rag.pipeline import QueryResult, get_default_pipeline

load_dotenv()

app = FastAPI(title="RAGOps — Production AI API", version="1.0.0")
_metrics = AIMetricsRecorder()


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    contexts: list[str]
    latency_ms: float
    cost_usd: float
    model: str
    memory_mb: float


@app.on_event("startup")
async def startup() -> None:
    if os.getenv("ENABLE_METRICS", "true").lower() == "true":
        start_metrics_server(port=8001)


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """
    Main query endpoint.
    Uses asyncio.to_thread() so blocking LangChain calls do not deadlock
    the FastAPI event loop under concurrent load.
    """
    _metrics.track_active(True)
    try:
        result: QueryResult = await asyncio.to_thread(
            get_default_pipeline().query, request.query
        )
        _metrics.record(
            latency_seconds=result.latency_ms / 1000,
            model=result.model,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cost_usd=result.cost_usd,
        )
        mem_mb = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        return QueryResponse(
            answer=result.answer,
            contexts=result.contexts,
            latency_ms=result.latency_ms,
            cost_usd=result.cost_usd,
            model=result.model,
            memory_mb=round(mem_mb, 1),
        )
    except Exception as e:
        _metrics.record(latency_seconds=0, error="pipeline_error")
        raise
    finally:
        _metrics.track_active(False)


@app.get("/health")
async def health() -> dict:
    mem_mb = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    return {"status": "healthy", "memory_mb": round(mem_mb, 1)}


@app.get("/")
async def root() -> dict:
    return {
        "name": "RAGOps Production AI API",
        "docs": "/docs",
        "health": "/health",
        "query": "POST /query",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)

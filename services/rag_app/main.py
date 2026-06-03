from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from functools import wraps
from pathlib import Path
from typing import Any, TypeVar, cast

import chromadb
import httpx
import yaml
from chromadb.api import ClientAPI
from chromadb.api.types import EmbeddingFunction
from chromadb.utils.embedding_functions.sentence_transformer_embedding_function import (
    SentenceTransformerEmbeddingFunction,
)
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from services.rag_app.trace_producer import QATrace, TraceProducer

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

RAG_QUERY_TOTAL = Counter("rag_query_total", "Total RAG queries handled", ["status"])
RAG_QUERY_LATENCY = Histogram("rag_query_latency_seconds", "End-to-end RAG query latency", ["stage"])


class RagSettings(BaseSettings):
    kafka_bootstrap_servers: str = "kafka:29092"
    kafka_trace_topic: str = "llm-traces"
    chromadb_host: str = "chromadb"
    chromadb_port: int = 8000
    ollama_base_url: str = "http://ollama:11434"
    ollama_model: str = "llama3.2:3b"
    ollama_timeout_seconds: float = 120.0
    config_path: str = "config/settings.yaml"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = RagSettings()


class RagRuntimeConfig(BaseModel):
    collection_name: str = "rag_documents"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    retrieval_top_k: int = 4
    temperature: float = 0.0
    num_ctx: int = 4096


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    ground_truth: str | None = None


class QueryResponse(BaseModel):
    trace_id: str
    question: str
    answer: str
    contexts: list[str]
    latency_ms: float


def _load_yaml_config(config_path: str) -> RagRuntimeConfig:
    path = Path(config_path)
    if not path.exists():
        logger.warning("config file %s missing, using defaults", config_path)
        return RagRuntimeConfig()
    raw = yaml.safe_load(path.read_text()) or {}
    rag_section = raw.get("rag", {}) or {}
    ollama_section = raw.get("ollama", {}) or {}
    allowed = set(RagRuntimeConfig.model_fields)
    values: dict[str, Any] = {}
    for key in ("collection_name", "embedding_model", "retrieval_top_k"):
        if key in rag_section and rag_section[key] is not None:
            values[key] = rag_section[key]
    for key in ("temperature", "num_ctx"):
        if key in ollama_section and ollama_section[key] is not None:
            values[key] = ollama_section[key]
    return RagRuntimeConfig(**{k: v for k, v in values.items() if k in allowed})


def _build_chroma_client() -> ClientAPI:
    return chromadb.HttpClient(host=settings.chromadb_host, port=settings.chromadb_port)


def _build_collection(client: ClientAPI, runtime: RagRuntimeConfig) -> Any:
    embedder = cast(
        EmbeddingFunction[Any],
        SentenceTransformerEmbeddingFunction(model_name=runtime.embedding_model),
    )
    return client.get_or_create_collection(
        name=runtime.collection_name,
        embedding_function=embedder,
    )


def _build_prompt(question: str, contexts: list[str]) -> str:
    joined = "\n\n".join(f"[{index}] {chunk}" for index, chunk in enumerate(contexts))
    return (
        "Answer the question using only the context below. "
        "If the context is insufficient, say so.\n\n"
        f"Context:\n{joined}\n\nQuestion: {question}\nAnswer:"
    )


def _track_latency(stage: str) -> Callable[[F], F]:
    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            with RAG_QUERY_LATENCY.labels(stage=stage).time():
                return await func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


@_track_latency("retrieval")
async def _retrieve_contexts(collection: Any, runtime: RagRuntimeConfig, question: str) -> list[str]:
    result = collection.query(query_texts=[question], n_results=runtime.retrieval_top_k)
    documents = result.get("documents") or [[]]
    return list(documents[0])


@_track_latency("generation")
async def _generate_answer(
    client: httpx.AsyncClient,
    runtime: RagRuntimeConfig,
    question: str,
    contexts: list[str],
) -> str:
    response = await client.post(
        f"{settings.ollama_base_url}/api/generate",
        json={
            "model": settings.ollama_model,
            "prompt": _build_prompt(question, contexts),
            "stream": False,
            "options": {"temperature": runtime.temperature, "num_ctx": runtime.num_ctx},
        },
    )
    response.raise_for_status()
    payload = response.json()
    return str(payload.get("response", "")).strip()


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    runtime = _load_yaml_config(settings.config_path)
    chroma_client = _build_chroma_client()
    application.state.runtime = runtime
    application.state.http_client = httpx.AsyncClient(timeout=settings.ollama_timeout_seconds)
    application.state.collection = _build_collection(chroma_client, runtime)
    application.state.producer = TraceProducer(
        bootstrap_servers=settings.kafka_bootstrap_servers,
        topic=settings.kafka_trace_topic,
    )
    await application.state.producer.start()
    logger.info("rag-app started")
    try:
        yield
    finally:
        await application.state.producer.stop()
        await application.state.http_client.aclose()
        logger.info("rag-app stopped")


app = FastAPI(title="rag-app", lifespan=lifespan)


def get_runtime(request: Request) -> RagRuntimeConfig:
    return request.app.state.runtime


def get_collection(request: Request) -> Any:
    return request.app.state.collection


def get_http_client(request: Request) -> httpx.AsyncClient:
    return request.app.state.http_client


def get_producer(request: Request) -> TraceProducer:
    return request.app.state.producer


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics")
async def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    runtime: RagRuntimeConfig = Depends(get_runtime),
    collection: Any = Depends(get_collection),
    http_client: httpx.AsyncClient = Depends(get_http_client),
    producer: TraceProducer = Depends(get_producer),
) -> QueryResponse:
    trace_id = uuid.uuid4().hex
    started = time.perf_counter()
    try:
        contexts = await _retrieve_contexts(collection, runtime, request.question)
        answer = await _generate_answer(http_client, runtime, request.question, contexts)
        latency_ms = (time.perf_counter() - started) * 1000.0
        trace = QATrace(
            trace_id=trace_id,
            question=request.question,
            answer=answer,
            contexts=contexts,
            ground_truth=request.ground_truth,
            model_name=settings.ollama_model,
            latency_ms=latency_ms,
        )
        await producer.publish(trace)
        RAG_QUERY_TOTAL.labels(status="success").inc()
        return QueryResponse(
            trace_id=trace_id,
            question=request.question,
            answer=answer,
            contexts=contexts,
            latency_ms=latency_ms,
        )
    except httpx.HTTPError as error:
        RAG_QUERY_TOTAL.labels(status="error").inc()
        logger.exception("ollama generation failed for trace_id=%s", trace_id)
        raise HTTPException(status_code=502, detail="generation backend failed") from error
    except Exception as error:
        RAG_QUERY_TOTAL.labels(status="error").inc()
        logger.exception("rag query failed for trace_id=%s", trace_id)
        raise HTTPException(status_code=500, detail="rag query failed") from error

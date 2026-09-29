"""
Local embedding + reranking server using FastEmbed (Qdrant ONNX).
OpenAI-compatible /v1/embeddings endpoint, Cohere/Jina-compatible /v1/rerank.
No PyTorch dependency — pure ONNX runtime.

Runs as its own container, shared by every Hermes instance on the
hermes-net Docker network. Reachable from sibling containers at
http://embedding-server:8766/v1.

Env vars:
  EMBED_MODEL      sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (default)
  EMBED_PORT       8766 (default)
  EMBED_DIM        384 (default)
  EMBED_MAX_TOKENS 512 (default)
  EMBED_HOST       0.0.0.0 (default) — bind address
  EMBED_CACHE_DIR  /cache (default) — persisted via a named volume so the
                   ONNX model isn't re-downloaded on every container restart
  RERANK_MODEL     jinaai/jina-reranker-v2-base-multilingual (default)
  RERANK_PRELOAD   false (default) — load the reranker in the lifespan hook
                   instead of on first request
  RERANK_MAX_DOCS  128 (default) — refuse absurd batches
  RERANK_MAX_CHARS 1000 (default) — per-document char cap before scoring
  RERANK_BATCH_SIZE 8 (default) — scoring batch; bounds peak memory
  RERANK_THREADS   0 (default = let onnxruntime choose)

Why the reranker is lazy by default
-----------------------------------
The cross-encoder is ~1.1 GB resident, against ~750 MB for the embedder alone
and ~10 GB free on this box. The embedder is on the hot path for every HyMem
write; the reranker is only used by the web-search fusion plugin, which may go
days without firing. Paying 1.1 GB permanently for an optional consumer is the
wrong trade, so it loads on first use behind a lock and stays loaded after.
RERANK_PRELOAD=true moves the cost to startup if that ever becomes preferable.

The embedding model still loads at startup, unchanged: /v1/embeddings must
never pay a cold-start penalty, and it must never be taken down by a reranker
problem. Every reranker failure is a 503 on /v1/rerank only.

HyMem config (per Hermes instance):
  HYMEM_EMBEDDING_API_KEY=local
  HYMEM_EMBEDDING_BASE_URL=http://embedding-server:8766/v1
  HYMEM_EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
  HYMEM_EMBEDDING_DIM=384
"""

import os
import time
import logging
import threading
import math
from contextlib import asynccontextmanager
from typing import List, Optional, Union

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("embedding-server")


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("", "0", "false", "no", "off")


MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
PORT = int(os.getenv("EMBED_PORT", "8766"))
DIM = int(os.getenv("EMBED_DIM", "384"))
MAX_TOKENS = int(os.getenv("EMBED_MAX_TOKENS", "512"))
HOST = os.getenv("EMBED_HOST", "0.0.0.0")
CACHE_DIR = os.getenv("EMBED_CACHE_DIR", "/cache")

# Multilingual on purpose. This host's real workload is Dutch-language
# occupational-medicine material; the ms-marco cross-encoders fastembed also
# ships are English-only and score Dutch text close to noise.
RERANK_MODEL = os.getenv("RERANK_MODEL", "jinaai/jina-reranker-v2-base-multilingual")
RERANK_PRELOAD = _env_flag("RERANK_PRELOAD", False)
RERANK_MAX_DOCS = int(os.getenv("RERANK_MAX_DOCS", "128"))
RERANK_MAX_CHARS = int(os.getenv("RERANK_MAX_CHARS", "1000"))
# Measured on this box, 20 documents in ONE batch: onnxruntime's arena
# allocator grew to ~3.0 GB anonymous and never gave it back, so the container
# then sat permanently against its cgroup ceiling. The arena keeps the high
# water mark of the largest batch it ever saw, so the only way to hold memory
# down is to never hand it a big batch. 8 costs nothing in latency (the model
# is compute-bound, not batch-bound) and caps the peak.
RERANK_BATCH_SIZE = max(1, int(os.getenv("RERANK_BATCH_SIZE", "8")))
# 0 = let onnxruntime pick (it defaults to the physical core count).
RERANK_THREADS = int(os.getenv("RERANK_THREADS", "0")) or None

embed_model = None
EMBED_BATCH_SIZE = 16

# Hold across a whole inference request, including lazy model loading. This
# prevents concurrent embed/rerank work from multiplying ONNX peak memory.
# Reentrant because rerank calls the loader while already holding this lock.
_inference_lock = threading.RLock()

# Reranker state. The shared lock guards BOTH the load and the last-error field,
# so two concurrent first-requests cannot each start a 1.1 GB download.
_rerank_model = None
_rerank_lock = _inference_lock
_rerank_load_error: Optional[str] = None


def _load_reranker():
    """Return the cross-encoder, loading it once under the lock.

    Double-checked: the fast path reads the module global without taking the
    lock at all; inference callers already hold the shared lock. Raises
    RuntimeError with a readable message on failure — never lets a fastembed
    exception escape as a 500.
    """
    global _rerank_model, _rerank_load_error
    if _rerank_model is not None:
        return _rerank_model
    with _rerank_lock:
        if _rerank_model is not None:
            return _rerank_model
        log.info(f"Loading reranker: {RERANK_MODEL} (cache_dir={CACHE_DIR}) ...")
        t0 = time.time()
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            os.makedirs(CACHE_DIR, exist_ok=True)
            # Same /cache named volume as the embedder, so the ~1.1 GB ONNX
            # download survives a recreate instead of being refetched.
            _rerank_model = TextCrossEncoder(
                model_name=RERANK_MODEL, cache_dir=CACHE_DIR,
                threads=RERANK_THREADS,
            )
        except Exception as exc:
            _rerank_load_error = f"{type(exc).__name__}: {exc}"
            log.error(f"Reranker load failed: {_rerank_load_error}")
            # Cleared so a later request can retry — a transient download
            # failure must not permanently disable the endpoint.
            _rerank_model = None
            raise RuntimeError(_rerank_load_error) from exc
        _rerank_load_error = None
        log.info(f"Reranker loaded in {time.time() - t0:.1f}s")
        return _rerank_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    global embed_model
    log.info(f"Loading model: {MODEL_NAME} (cache_dir={CACHE_DIR}) ...")
    t0 = time.time()
    from fastembed import TextEmbedding
    os.makedirs(CACHE_DIR, exist_ok=True)
    embed_model = TextEmbedding(model_name=MODEL_NAME, max_length=MAX_TOKENS, cache_dir=CACHE_DIR)
    log.info(f"Model loaded in {time.time() - t0:.1f}s (dim={DIM})")
    if RERANK_PRELOAD:
        # Opt-in only. Failure here is logged and swallowed: a broken reranker
        # must not stop the embedding server from booting.
        try:
            _load_reranker()
        except Exception as exc:
            log.error(f"RERANK_PRELOAD set but load failed, continuing: {exc}")
    yield
    log.info("Shutting down")


app = FastAPI(title="Local Embedding Server", version="0.2.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]] = Field(..., description="Text(s) to embed")
    model: str = Field(default=MODEL_NAME, description="Model name (ignored, single-model server)")
    encoding_format: Optional[str] = Field(default="float", description="float or base64")
    user: Optional[str] = None


class EmbeddingObject(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int


class Usage(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingObject]
    model: str
    usage: Usage


class RerankRequest(BaseModel):
    query: str = Field(..., description="The search query")
    documents: List[str] = Field(..., description="Candidate documents to score")
    top_n: Optional[int] = Field(default=None, description="Return only the top N")
    model: Optional[str] = Field(
        default=None, description="Model name (ignored, single-model server)"
    )


class RerankResult(BaseModel):
    index: int
    relevance_score: float


class RerankResponse(BaseModel):
    results: List[RerankResult]
    model: str


@app.get("/health")
async def health():
    # Deliberately does NOT touch _load_reranker(): the healthcheck runs every
    # 10s and must never trigger, or block on, a 1.1 GB model load. It reports
    # the *observed* state by reading the globals, nothing more.
    rerank_block = {
        "model": RERANK_MODEL,
        "loaded": _rerank_model is not None,
        "preload": RERANK_PRELOAD,
    }
    if _rerank_load_error:
        rerank_block["last_error"] = _rerank_load_error
    return {"status": "ok", "model": MODEL_NAME, "dim": DIM, "rerank": rerank_block}


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": MODEL_NAME, "object": "model", "created": 0, "owned_by": "local"},
            {"id": RERANK_MODEL, "object": "model", "created": 0, "owned_by": "local"},
        ],
    }


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
def embeddings(req: EmbeddingRequest):
    if embed_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    texts = [req.input] if isinstance(req.input, str) else req.input
    if not texts:
        raise HTTPException(status_code=400, detail="Empty input")

    t0 = time.time()
    data = []
    try:
        with _inference_lock:
            for start in range(0, len(texts), EMBED_BATCH_SIZE):
                batch = texts[start:start + EMBED_BATCH_SIZE]
                count = 0
                for vec in embed_model.embed(batch, batch_size=EMBED_BATCH_SIZE):
                    if count >= len(batch):
                        raise ValueError("Unexpected embedding count")
                    values = [float(v) for v in vec]
                    if len(values) != DIM or not all(math.isfinite(v) for v in values):
                        raise ValueError("Invalid embedding vector")
                    data.append(EmbeddingObject(embedding=values, index=start + count))
                    count += 1
                if count != len(batch):
                    raise ValueError("Unexpected embedding count")
    except Exception as exc:
        # Never return a partial batch or echo text/model exception fragments.
        log.error("Embedding inference failed (%s)", type(exc).__name__)
        raise HTTPException(status_code=503, detail="Embedding inference failed") from exc
    elapsed = time.time() - t0
    log.info(f"Embedded {len(texts)} texts in {elapsed:.3f}s ({elapsed/len(texts)*1000:.1f}ms each)")

    return EmbeddingResponse(
        object="list",
        data=data,
        model=MODEL_NAME,
        usage=Usage(),
    )


# Sync def on purpose (as for /v1/embeddings): Starlette runs a plain
# `def` endpoint in its threadpool, so a multi-second cold load or a big batch
# cannot block the event loop — and therefore cannot stall /health.
@app.post("/v1/rerank", response_model=RerankResponse)
def rerank(req: RerankRequest):
    if not req.documents:
        raise HTTPException(status_code=400, detail="Empty documents")
    if len(req.documents) > RERANK_MAX_DOCS:
        raise HTTPException(
            status_code=400,
            detail=f"Too many documents: {len(req.documents)} > {RERANK_MAX_DOCS}",
        )

    with _inference_lock:
        return _rerank_locked(req)


def _rerank_locked(req: RerankRequest):

    try:
        model = _load_reranker()
    except Exception as exc:
        raise HTTPException(
            status_code=503, detail=f"Reranker unavailable: {exc}"
        ) from exc

    docs = [str(d or "")[:RERANK_MAX_CHARS] for d in req.documents]

    t0 = time.time()
    try:
        scores = list(
            model.rerank(req.query, docs, batch_size=RERANK_BATCH_SIZE)
        )
    except Exception as exc:
        log.error(f"Rerank scoring failed: {type(exc).__name__}: {exc}")
        raise HTTPException(
            status_code=503,
            detail=f"Rerank scoring failed: {type(exc).__name__}: {exc}",
        ) from exc
    elapsed = time.time() - t0

    if len(scores) != len(docs):
        raise HTTPException(
            status_code=503,
            detail=f"Reranker returned {len(scores)} scores for {len(docs)} documents",
        )

    results = [
        RerankResult(index=i, relevance_score=float(s)) for i, s in enumerate(scores)
    ]
    results.sort(key=lambda r: r.relevance_score, reverse=True)
    if req.top_n is not None and req.top_n > 0:
        results = results[: req.top_n]

    log.info(
        f"Reranked {len(docs)} docs in {elapsed:.3f}s "
        f"({elapsed/len(docs)*1000:.1f}ms each), returning {len(results)}"
    )
    return RerankResponse(results=results, model=RERANK_MODEL)


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import check_database_connection, init_db, init_pool, close_pool
from app.db.qdrant import check_qdrant_connection, close_qdrant, init_qdrant
from app.redis_client import check_redis_connection, close_redis
from app.api.routes import router as api_router, run_ingestion_worker
from app.agents.graph import get_embedding_model, get_sparse_embedding_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Startup and shutdown lifecycle manager."""
    logger.info("🚀 Initializing System Services...")
    
    await init_pool()
    
    await init_db()
    await init_qdrant()
    ingestion_worker_task = asyncio.create_task(run_ingestion_worker())

    if not await check_redis_connection():
        logger.warning("Redis is unavailable; response caching is disabled until it recovers.")

    try:
        await asyncio.gather(
            asyncio.to_thread(lambda: list(get_embedding_model().embed(["warmup"]))),
            asyncio.to_thread(lambda: list(get_sparse_embedding_model().embed(["warmup"]))),
        )
        logger.info("✅ FastEmbed Dense & Sparse models pre-warmed.")
    except Exception as e:
        logger.warning(f"FastEmbed pre-warm notice: {e}")
    
    yield
    
    logger.info("🛑 Shutting down System Services...")
    ingestion_worker_task.cancel()
    try:
        await ingestion_worker_task
    except asyncio.CancelledError:
        pass
    await close_pool()
    await close_redis()
    await close_qdrant()


app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")



@app.middleware("http")
async def add_security_headers(request: Request, call_next) -> Response:
    """Apply baseline browser security headers to every HTTP response."""
    request.state.request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["Content-Security-Policy"] = "default-src 'self'; frame-ancestors 'none'"
    return response


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Return a lightweight liveness response without disclosing infrastructure details."""
    return {"status": "healthy", "environment": settings.ENVIRONMENT}


@app.get("/ready")
async def readiness_check() -> Response:
    """Report whether required backing services are available for traffic."""
    postgres_ready = await check_database_connection()
    qdrant_ready = await check_qdrant_connection()
    redis_ready = await check_redis_connection()
    ready = postgres_ready and qdrant_ready and (redis_ready or not settings.REQUIRE_REDIS)
    return JSONResponse(
        status_code=200 if ready else 503,
        content={
            "status": "ready" if ready else "not_ready",
            "dependencies": {
                "postgres": postgres_ready,
                "qdrant": qdrant_ready,
                "redis": redis_ready,
            },
        },
    )

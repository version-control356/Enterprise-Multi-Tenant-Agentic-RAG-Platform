import logging
import json
import base64
import time
from typing import Any, Optional
import redis.asyncio as redis
from app.config import settings

logger = logging.getLogger(__name__)

if settings.REDIS_URL:
    redis_client = redis.from_url(
        settings.REDIS_URL,
        db=0,
        decode_responses=True,
        max_connections=20,
        socket_connect_timeout=5,
        socket_timeout=10,
    )
    redis_pool = redis_client.connection_pool
else:
    redis_pool = redis.ConnectionPool(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        db=0,
        decode_responses=True,
        max_connections=20,
        socket_connect_timeout=5,
        socket_timeout=10,
    )
    redis_client = redis.Redis(connection_pool=redis_pool)

INGESTION_QUEUE = "ingestion_jobs"
INGESTION_PROCESSING_QUEUE = "ingestion_jobs:processing"
INGESTION_LEASES = "ingestion_jobs:leases"
INGESTION_LEASE_TIMEOUT_SECONDS = 900


async def close_redis() -> None:
    """Close Redis connections during application shutdown."""
    await redis_client.aclose()
    await redis_pool.aclose()


async def check_redis_connection() -> bool:
    """Utility to test Redis connectivity on startup."""
    try:
        await redis_client.ping()
        logger.info("✅ Redis Cache connected successfully.")
        return True
    except Exception as e:
        logger.error(f"❌ Redis connection error: {e}")
        return False


async def get_cached_response(cache_key: str) -> Optional[dict[str, Any]]:
    """Fetch cached RAG response by key."""
    try:
        data = await redis_client.get(cache_key)
        if data:
            return json.loads(data)
    except Exception as e:
        logger.warning(f"Redis GET failed for key {cache_key}: {e}")
    return None


async def set_cached_response(cache_key: str, data: dict[str, Any], ttl_seconds: int = 3600) -> None:
    """Store RAG response in Redis with expiration TTL."""
    try:
        await redis_client.setex(
            name=cache_key,
            time=ttl_seconds,
            value=json.dumps(data)
        )
    except Exception as e:
        logger.warning(f"Redis SET failed for key {cache_key}: {e}")


async def is_rate_limit_exceeded(cache_key: str, limit: int, window_seconds: int = 60) -> bool:
    """Increment a fixed-window Redis counter and report whether it exceeds a limit."""
    try:
        request_count = await redis_client.incr(cache_key)
        if request_count == 1:
            await redis_client.expire(cache_key, window_seconds)
        return request_count > limit
    except Exception as error:
        if settings.REQUIRE_REDIS:
            logger.error("Redis rate-limit check failed; blocking request: %s", error)
            return True
        logger.warning("Redis rate-limit check failed; allowing request in development: %s", error)
        return False


async def get_tenant_cache_version(tenant_id: str) -> int:
    """Retrieve the current cache version epoch for a tenant."""
    try:
        version = await redis_client.get(f"tenant_ver:{tenant_id}")
        return int(version) if version else 1
    except Exception as error:
        logger.warning("Redis tenant cache version read failed: %s", error)
        return 1


async def bump_tenant_cache_version(tenant_id: str) -> int:
    """Increment the tenant cache version to invalidate stale cached responses."""
    try:
        new_version = await redis_client.incr(f"tenant_ver:{tenant_id}")
        logger.info(f"Invalidated cache for tenant '{tenant_id}' (new version: {new_version})")
        return new_version
    except Exception as error:
        logger.warning("Redis tenant cache version bump failed: %s", error)
        return 1


async def set_ingestion_task(task_id: str, task: dict[str, Any], ttl_seconds: int = 86400) -> None:
    """Persist ingestion task metadata so status polling works across API replicas."""
    try:
        await redis_client.setex(f"ingestion_task:{task_id}", ttl_seconds, json.dumps(task))
    except Exception as error:
        logger.warning("Redis ingestion-task write failed: %s", error)
        if settings.REQUIRE_REDIS:
            raise


async def get_ingestion_task(task_id: str) -> Optional[dict[str, Any]]:
    """Load persisted ingestion task metadata."""
    try:
        data = await redis_client.get(f"ingestion_task:{task_id}")
        return json.loads(data) if data else None
    except Exception as error:
        logger.warning("Redis ingestion-task read failed: %s", error)
        if settings.REQUIRE_REDIS:
            raise
        return None


async def enqueue_ingestion_job(
    task_id: str,
    content: bytes,
    filename: str,
    allowed_roles: str,
    tenant_id: str,
    user_id: str,
) -> None:
    """Persist an ingestion payload and enqueue its ID for a shared worker."""
    payload_key = f"ingestion_payload:{task_id}"
    job = {
        "task_id": task_id,
        "filename": filename,
        "allowed_roles": allowed_roles,
        "tenant_id": tenant_id,
        "user_id": user_id,
    }
    encoded_content = base64.b64encode(content).decode("ascii")
    async with redis_client.pipeline(transaction=True) as pipeline:
        await pipeline.setex(payload_key, 86400, encoded_content)
        await pipeline.rpush(INGESTION_QUEUE, json.dumps(job))
        await pipeline.execute()


async def claim_ingestion_job(timeout_seconds: int = 3) -> Optional[dict[str, Any]]:
    """Claim one ingestion job and create a lease that can be reclaimed after a crash."""
    try:
        await reclaim_stale_ingestion_jobs()
    except Exception as error:
        if settings.REQUIRE_REDIS:
            raise
        logger.warning("Unable to reclaim stale ingestion jobs in development: %s", error)
    raw_job = await redis_client.brpoplpush(
        INGESTION_QUEUE,
        INGESTION_PROCESSING_QUEUE,
        timeout=timeout_seconds,
    )
    if not raw_job:
        return None
    try:
        await redis_client.zadd(
            INGESTION_LEASES,
            {raw_job: time.time() + INGESTION_LEASE_TIMEOUT_SECONDS},
        )
    except Exception:
        if settings.REQUIRE_REDIS:
            raise
        logger.warning("Unable to create an ingestion lease in development.", exc_info=True)
    return json.loads(raw_job)


async def reclaim_stale_ingestion_jobs() -> int:
    """Requeue processing jobs whose worker lease expired after a crash."""
    now = time.time()
    stale_jobs = await redis_client.zrangebyscore(INGESTION_LEASES, "-inf", now)
    reclaimed = 0
    for raw_job in stale_jobs:
        if not await redis_client.zrem(INGESTION_LEASES, raw_job):
            continue
        await redis_client.lrem(INGESTION_PROCESSING_QUEUE, 1, raw_job)
        job = json.loads(raw_job)
        job["attempt"] = int(job.get("attempt", 0)) + 1
        await redis_client.rpush(INGESTION_QUEUE, json.dumps(job))
        reclaimed += 1
        logger.warning("Reclaimed stale ingestion job %s (attempt %s).", job.get("task_id"), job["attempt"])
    return reclaimed


async def get_ingestion_payload(task_id: str) -> Optional[bytes]:
    """Load and decode a queued ingestion payload."""
    encoded_content = await redis_client.get(f"ingestion_payload:{task_id}")
    return base64.b64decode(encoded_content) if encoded_content else None


async def acknowledge_ingestion_job(job: dict[str, Any]) -> None:
    """Remove a completed ingestion job and its payload from Redis."""
    task_id = str(job["task_id"])
    job_key = f"ingestion_payload:{task_id}"
    raw_job = json.dumps(job)
    async with redis_client.pipeline(transaction=True) as pipeline:
        await pipeline.lrem(INGESTION_PROCESSING_QUEUE, 1, raw_job)
        await pipeline.zrem(INGESTION_LEASES, raw_job)
        await pipeline.delete(job_key)
        await pipeline.execute()


async def revoke_token(token_hash: str, ttl_seconds: int) -> None:
    """Mark an access token hash as revoked until its natural expiration."""
    try:
        await redis_client.setex(f"revoked_token:{token_hash}", ttl_seconds, "1")
    except Exception as error:
        logger.warning("Redis token revocation write failed: %s", error)
        if settings.REQUIRE_REDIS:
            raise


async def is_token_revoked(token_hash: str) -> bool:
    """Return whether an access-token hash has been revoked."""
    try:
        return bool(await redis_client.exists(f"revoked_token:{token_hash}"))
    except Exception as error:
        logger.warning("Redis token revocation check failed: %s", error)
        return settings.REQUIRE_REDIS

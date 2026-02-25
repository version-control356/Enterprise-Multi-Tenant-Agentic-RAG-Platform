import asyncio
import json
import orjson
import hashlib
import logging
import time
import uuid
from typing import Annotated, AsyncGenerator, List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Request, UploadFile, File, Form, status
from fastapi.responses import StreamingResponse
from fastapi.security import OAuth2PasswordRequestForm
from qdrant_client.models import PointStruct, SparseVector
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field
from app.redis_client import redis_client

from app.config import settings
from app.core.auth import (
    authenticate_user,
    create_access_token,
    get_current_tenant_user,
    register_tenant_user,
    TokenData,
)
from app.core.security import SecurityGuardrails
from app.core.audit import record_audit_event
from app.core.tracing import telemetry_tracker, TraceRecord
from app.ingestion.parser import UniversalDocumentParser
from app.db.qdrant import (
    upsert_documents_to_qdrant,
    delete_documents_from_qdrant,
    delete_all_tenant_documents_from_qdrant,
)
from app.agents.graph import (
    build_agent_graph,
    get_embedding_model,
    get_sparse_embedding_model,
)
from app.database import (
    get_checkpointer,
    record_document,
    list_tenant_documents,
    delete_thread_history,
    delete_all_tenant_user_history,
    delete_tenant_document,
    delete_all_tenant_documents,
    get_tenant_document_usage,
)
from app.tenant_user import list_tenant_users, delete_tenant_user
from app.redis_client import (
    bump_tenant_cache_version,
    get_cached_response,
    get_tenant_cache_version,
    is_rate_limit_exceeded,
    set_cached_response,
    set_ingestion_task,
    get_ingestion_task,
    enqueue_ingestion_job,
    claim_ingestion_job,
    get_ingestion_payload,
    acknowledge_ingestion_job,
)

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/ready")
async def readiness_probe() -> dict:
    """Simple health‑check used by Docker to verify the FastAPI service is running."""
    return {"status": "ok"}

class UserRegistrationRequest(BaseModel):
    """Validated payload used to create tenant-scoped user credentials."""

    username: str = Field(min_length=3, max_length=64, pattern=r"^[A-Za-z0-9_.-]+$")
    password: str = Field(min_length=12, max_length=72)
    tenant_id: str = Field(min_length=3, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    role: str = Field(default="admin", pattern=r"^(admin|analyst|viewer)$")


def build_chat_cache_key(
    tenant_id: str,
    user_id: str,
    role: str,
    thread_id: str,
    query: str,
    cache_version: int = 1,
) -> str:
    """Build a version-scoped cache key partitioned by tenant, user, role, and conversation."""
    query_digest = hashlib.sha256(query.encode("utf-8")).hexdigest()
    scope_digest = hashlib.sha256(
        json.dumps(
            [tenant_id, user_id, role, thread_id, cache_version],
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return f"cache:{scope_digest}:{query_digest}"


def build_backend_thread_id(tenant_id: str, user_id: str, thread_id: str) -> str:
    """Create an unambiguous checkpointer identifier for a tenant/user/thread tuple."""
    digest = hashlib.sha256(
        json.dumps(
            [tenant_id, user_id, thread_id],
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return f"thread:{digest}"


async def enforce_rate_limit(
    request: Request,
    scope: str,
    limit: int,
    identity: Optional[str] = None,
) -> None:
    """Apply a Redis-backed fixed-window limit using the request source address."""
    client_ip = request.client.host if request.client else "unknown"
    rate_subject = identity or client_ip
    if await is_rate_limit_exceeded(f"rate:{scope}:{rate_subject}", limit):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please retry shortly.",
            headers={"Retry-After": "60"},
        )


@router.post("/auth/register", status_code=status.HTTP_201_CREATED)
async def register_tenant_admin(payload: UserRegistrationRequest, request: Request) -> dict[str, str]:
    """Create a tenant administrator when self-registration is explicitly enabled."""
    if not settings.ALLOW_SELF_REGISTRATION:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Self-registration is disabled. Ask a platform administrator to provision access.",
        )
    await enforce_rate_limit(request, "registration", settings.AUTH_RATE_LIMIT_PER_MINUTE)

    created = await register_tenant_user(
        username=payload.username,
        password=payload.password,
        tenant_id=payload.tenant_id,
        role="admin",
    )
    if not created:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A user with this username already exists in the tenant.",
        )
    return {"status": "created", "tenant_id": payload.tenant_id, "username": payload.username}


@router.post("/admin/users", status_code=status.HTTP_201_CREATED)
async def provision_tenant_user(
    payload: UserRegistrationRequest,
    current_user: TokenData = Depends(get_current_tenant_user),
) -> dict[str, str]:
    """Allow a tenant administrator to provision a user in their own tenant."""
    if current_user.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Administrator role required.")
    if payload.tenant_id != current_user.tenant_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Cross-tenant provisioning is forbidden.")

    created = await register_tenant_user(
        username=payload.username,
        password=payload.password,
        tenant_id=payload.tenant_id,
        role=payload.role,
    )
    if not created:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A user with this username already exists in the tenant.",
        )
    return {"status": "created", "tenant_id": payload.tenant_id, "username": payload.username}

@router.get("/admin/users")
async def list_tenant_users_endpoint(
    limit: int = 100,
    offset: int = 0,
    current_user: TokenData = Depends(get_current_tenant_user),
) -> list[dict]:
    """Return a paginated list of users within the admin's tenant (excluding passwords).
    Results are cached for 30 seconds per tenant to reduce DB load.
    """
    if current_user.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Administrator role required.")
    cache_key = f"tenant:{current_user.tenant_id}:users:{limit}:{offset}"
    try:
        cached = await redis_client.get(cache_key)
        if cached:
            users_page = orjson.loads(cached)
        else:
            users_page = await list_tenant_users(current_user.tenant_id, limit=limit, offset=offset)
            await redis_client.setex(cache_key, 60, orjson.dumps(users_page))
    except Exception:
        users_page = await list_tenant_users(current_user.tenant_id, limit=limit, offset=offset)
    return [{"username": u["username"], "role": u["role"], "created_at": u["created_at"]} for u in users_page]

@router.delete("/admin/users/{username}")
async def delete_tenant_user_endpoint(
    username: str,
    current_user: TokenData = Depends(get_current_tenant_user),
) -> dict[str, str]:
    """Allow an admin to delete a user from their tenant."""
    if current_user.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Administrator role required.")
    if username == current_user.user_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Admins cannot delete themselves.")
    deleted = await delete_tenant_user(current_user.tenant_id, username)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")
    try:
        async for key in redis_client.scan_iter(
            match=f"tenant:{current_user.tenant_id}:users:*"
        ):
            await redis_client.delete(key)
    except Exception as error:
        logger.warning("Failed to invalidate user-list cache: %s", error)
    return {"status": "deleted", "username": username}


@router.post("/auth/token")
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    tenant_id: Annotated[str, Form(...)] = "",
    request: Request = None,
) -> dict[str, str]:
    """Authentication endpoint issuing tenant-scoped JWT access tokens."""
    if not form_data.username or not form_data.password or not tenant_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username, password, and tenant_id are required.",
        )
    
    if request is not None:
        await enforce_rate_limit(request, "login", settings.AUTH_RATE_LIMIT_PER_MINUTE)
    user = await authenticate_user(form_data.username, form_data.password, tenant_id.strip())
    if user is None:
        record_audit_event("auth.login", request, outcome="failure")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username, password, or tenant.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(
        data={"sub": user.user_id, "tenant_id": user.tenant_id, "role": user.role}
    )
    record_audit_event(
        "auth.login",
        request,
        outcome="success",
        tenant_id=user.tenant_id,
        user_id=user.user_id,
    )
    return {"access_token": access_token, "token_type": "bearer"}


_ingestion_tasks: Dict[str, Dict[str, Any]] = {}


async def _process_document_ingestion(
    content: bytes,
    filename: str,
    allowed_roles: str,
    tenant_id: str,
    user_id: str,
    request: Optional[Request] = None,
) -> dict:
    """Core pipeline to parse, redact PII, embed with hybrid vectors, and persist in Qdrant & PostgreSQL."""
    raw_text = await asyncio.to_thread(
        UniversalDocumentParser.extract_text,
        content,
        filename,
    )
    sanitized_text = SecurityGuardrails.redact_pii(raw_text)
    chunks = await asyncio.to_thread(
        UniversalDocumentParser.chunk_document,
        sanitized_text,
        filename,
    )

    if not chunks:
        raise ValueError("The uploaded document did not contain extractable text.")

    document_count, storage_bytes = await get_tenant_document_usage(tenant_id)
    if document_count >= settings.MAX_DOCUMENTS_PER_TENANT:
        raise ValueError("The tenant document quota has been reached.")
    if storage_bytes + len(content) > settings.MAX_STORAGE_BYTES_PER_TENANT:
        raise ValueError("The tenant storage quota has been reached.")

    dense_embs, sparse_embs = await asyncio.gather(
        asyncio.to_thread(lambda: list(get_embedding_model().embed(chunks))),
        asyncio.to_thread(lambda: list(get_sparse_embedding_model().embed(chunks))),
    )

    roles_list = [role.strip().lower() for role in allowed_roles.split(",") if role.strip()]
    if not roles_list or not set(roles_list).issubset({"admin", "analyst", "viewer"}):
        raise ValueError("allowed_roles must contain only admin, analyst, or viewer.")

    doc_id = str(uuid.uuid4())
    points = []
    for chunk, d_emb, s_emb in zip(chunks, dense_embs, sparse_embs):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "dense": d_emb.tolist(),
                    "sparse": SparseVector(
                        indices=s_emb.indices.tolist(),
                        values=s_emb.values.tolist(),
                    ),
                },
                payload={
                    "doc_id": doc_id,
                    "text": chunk,
                    "filename": filename,
                    "tenant_id": tenant_id,
                    "allowed_roles": roles_list,
                }
            )
        )

    await delete_documents_from_qdrant(tenant_id, filename=filename)
    await upsert_documents_to_qdrant(points)
    await record_document(
        doc_id=doc_id,
        tenant_id=tenant_id,
        filename=filename,
        size_bytes=len(content),
        chunks_count=len(chunks),
        created_by=user_id,
        allowed_roles=",".join(roles_list),
    )
    await bump_tenant_cache_version(tenant_id)
    if request:
        record_audit_event(
            "document.ingest",
            request,
            outcome="success",
            tenant_id=tenant_id,
            user_id=user_id,
            metadata={"filename": filename, "chunks": len(chunks)},
        )
    return {
        "status": "success", 
        "doc_id": doc_id,
        "filename": filename,
        "chunks_ingested": len(chunks), 
        "tenant_id": tenant_id
    }


async def _run_async_ingestion(
    task_id: str,
    content: bytes,
    filename: str,
    allowed_roles: str,
    tenant_id: str,
    user_id: str,
) -> None:
    """Worker coroutine to execute document ingestion in the background."""
    try:
        result = await _process_document_ingestion(
            content=content,
            filename=filename,
            allowed_roles=allowed_roles,
            tenant_id=tenant_id,
            user_id=user_id,
        )
        _ingestion_tasks[task_id] = {
            "task_id": task_id,
            "status": "completed",
            "filename": filename,
            "chunks_ingested": result.get("chunks_ingested", 0),
            "doc_id": result.get("doc_id"),
            "error": None,
            "timestamp": time.time(),
            "tenant_id": tenant_id,
            "user_id": user_id,
        }
        await set_ingestion_task(task_id, _ingestion_tasks[task_id])
    except Exception as error:
        logger.exception("Async ingestion task %s failed: %s", task_id, error)
        _ingestion_tasks[task_id] = {
            "task_id": task_id,
            "status": "failed",
            "filename": filename,
            "chunks_ingested": 0,
            "error": str(error),
            "timestamp": time.time(),
            "tenant_id": tenant_id,
            "user_id": user_id,
        }
        await set_ingestion_task(task_id, _ingestion_tasks[task_id])


async def run_ingestion_worker() -> None:
    """Consume Redis-backed ingestion jobs until application shutdown."""
    while True:
        try:
            job = await claim_ingestion_job()
            if not job:
                continue
            task_id = str(job["task_id"])
            content = await get_ingestion_payload(task_id)
            if content is None:
                logger.error("Ingestion payload missing for task %s.", task_id)
                continue
            await _run_async_ingestion(
                task_id=task_id,
                content=content,
                filename=str(job["filename"]),
                allowed_roles=str(job["allowed_roles"]),
                tenant_id=str(job["tenant_id"]),
                user_id=str(job["user_id"]),
            )
            await acknowledge_ingestion_job(job)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Ingestion worker iteration failed.")
            await asyncio.sleep(1)


@router.post("/ingest")
async def ingest_document(
    request: Request,
    file: UploadFile = File(...),
    allowed_roles: str = Form("admin,analyst"),
    current_user: TokenData = Depends(get_current_tenant_user)
) -> dict:
    """Ingest a document synchronously with role-based access control.

    The endpoint parses, redacts PII, embeds the document using dense and sparse models,
    and stores it in Qdrant and PostgreSQL. Rate limiting is enforced per tenant/user.
    """
    try:
        if current_user.role != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only tenant administrators may ingest documents.",
            )
        await enforce_rate_limit(
            request,
            "ingest",
            settings.CHAT_RATE_LIMIT_PER_MINUTE,
            identity=f"{current_user.tenant_id}:{current_user.user_id}",
        )
        filename = file.filename if file.filename else "uploaded_document.txt"
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > settings.MAX_UPLOAD_BYTES + 1_024:
            raise ValueError("Uploaded file exceeds the configured size limit.")
        content = await file.read(settings.MAX_UPLOAD_BYTES + 1)
        if len(content) > settings.MAX_UPLOAD_BYTES:
            raise ValueError("Uploaded file exceeds the configured size limit.")

        result = await _process_document_ingestion(
            content=content,
            filename=filename,
            allowed_roles=allowed_roles,
            tenant_id=current_user.tenant_id,
            user_id=current_user.user_id,
            request=request,
        )
        return result
    except HTTPException:
        raise
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error
    except Exception as error:
        logger.exception("Document ingestion failed.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Document ingestion failed.",
        ) from error


@router.post("/ingest/async")
async def ingest_document_async(
    request: Request,
    file: UploadFile = File(...),
    allowed_roles: str = Form("admin,analyst"),
    current_user: TokenData = Depends(get_current_tenant_user)
):
    """Initiates an asynchronous background ingestion task for large documents."""
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only tenant administrators may ingest documents.",
        )
    await enforce_rate_limit(
        request,
        "ingest",
        settings.CHAT_RATE_LIMIT_PER_MINUTE,
        identity=f"{current_user.tenant_id}:{current_user.user_id}",
    )
    filename = file.filename if file.filename else "uploaded_document.txt"
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > settings.MAX_UPLOAD_BYTES + 1_024:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file exceeds the configured size limit.")
    content = await file.read(settings.MAX_UPLOAD_BYTES + 1)
    if len(content) > settings.MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file exceeds the configured size limit.")

    task_id = str(uuid.uuid4())
    _ingestion_tasks[task_id] = {
        "task_id": task_id,
        "status": "processing",
        "filename": filename,
        "chunks_ingested": 0,
        "error": None,
        "timestamp": time.time(),
        "tenant_id": current_user.tenant_id,
        "user_id": current_user.user_id,
    }
    await set_ingestion_task(task_id, _ingestion_tasks[task_id])

    try:
        await enqueue_ingestion_job(
            task_id=task_id,
            content=content,
            filename=filename,
            allowed_roles=allowed_roles,
            tenant_id=current_user.tenant_id,
            user_id=current_user.user_id,
        )
    except Exception as error:
        _ingestion_tasks.pop(task_id, None)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Ingestion queue is unavailable. Please retry shortly.",
        ) from error

    return {
        "task_id": task_id,
        "status": "processing",
        "filename": filename,
        "message": "Document ingestion started in background. Poll /ingest/status/{task_id} for progress.",
    }


@router.get("/ingest/status/{task_id}")
async def get_ingestion_status(
    task_id: str,
    current_user: TokenData = Depends(get_current_tenant_user),
) -> Dict[str, Any]:
    """Check status and chunk results for a background ingestion task."""
    task = await get_ingestion_task(task_id)
    if task is None:
        task = _ingestion_tasks.get(task_id)
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Ingestion task ID not found.",
        )
    if (
        task.get("tenant_id") != current_user.tenant_id
        or (
            task.get("user_id") != current_user.user_id
            and current_user.role != "admin"
        )
    ):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Ingestion task ID not found.")
    return task


@router.get("/documents")
async def get_tenant_documents(
    current_user: TokenData = Depends(get_current_tenant_user)
) -> List[Dict[str, Any]]:
    """Retrieve all indexed documents in the tenant's knowledge base accessible to the caller's role."""
    docs = await list_tenant_documents(current_user.tenant_id, user_role=current_user.role)
    return [
        {
            "id": str(d["id"]),
            "filename": d["filename"],
            "size_bytes": d["size_bytes"],
            "chunks_count": d["chunks_count"],
            "created_by": d["created_by"],
            "allowed_roles": d.get("allowed_roles", "admin,analyst,viewer"),
            "created_at": d["created_at"].isoformat() if hasattr(d["created_at"], "isoformat") else str(d["created_at"]),
        }
        for d in docs
    ]


@router.delete("/documents/{doc_id}")
async def delete_document(
    doc_id: str,
    current_user: TokenData = Depends(get_current_tenant_user),
) -> dict[str, str]:
    """Delete a document from the tenant's knowledge base and Qdrant index."""
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only tenant administrators may delete documents.",
        )
    filename = await delete_tenant_document(current_user.tenant_id, doc_id)
    if not filename:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
    await delete_documents_from_qdrant(current_user.tenant_id, filename=filename)
    await bump_tenant_cache_version(current_user.tenant_id)
    return {"status": "deleted", "doc_id": doc_id, "filename": filename or doc_id}


@router.delete("/documents")
async def clear_all_documents(
    current_user: TokenData = Depends(get_current_tenant_user),
) -> dict[str, Any]:
    """Clear all indexed documents and vector points for the caller's tenant."""
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only tenant administrators may clear the knowledge base.",
        )
    deleted_count = await delete_all_tenant_documents(current_user.tenant_id)
    await delete_all_tenant_documents_from_qdrant(current_user.tenant_id)
    await bump_tenant_cache_version(current_user.tenant_id)
    return {"status": "cleared", "tenant_id": current_user.tenant_id, "deleted_count": deleted_count}


@router.post("/chat/stream")
async def chat_stream(
    request: Request,
    query: Annotated[str, Query(min_length=1, max_length=4_000)],
    thread_id: Annotated[str, Query(min_length=1, max_length=128)] = "default_thread",
    current_user: TokenData = Depends(get_current_tenant_user)
):
    """Executes stateful LangGraph hybrid CRAG agent and streams SSE tokens with observability tracing."""
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    start_time = time.perf_counter()
    telemetry_tracker.start_trace(request_id)

    try:
        sanitized_query = await SecurityGuardrails.sanitize_prompt_async(query)
    except ValueError as e:
        telemetry_tracker.finalize_trace(
            trace_id=request_id,
            tenant_id=current_user.tenant_id,
            user_id=current_user.user_id,
            role=current_user.role,
            query=query,
            start_time=start_time,
            error_message=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

    if await is_rate_limit_exceeded(
        f"rate:chat:{current_user.tenant_id}:{current_user.user_id}",
        settings.CHAT_RATE_LIMIT_PER_MINUTE,
    ):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Chat rate limit exceeded. Please retry shortly.",
            headers={"Retry-After": "60"},
        )

    tenant_cache_version = await get_tenant_cache_version(current_user.tenant_id)
    cache_key = build_chat_cache_key(
        tenant_id=current_user.tenant_id,
        user_id=current_user.user_id,
        role=current_user.role,
        thread_id=thread_id,
        query=sanitized_query,
        cache_version=tenant_cache_version,
    )
    cached = await get_cached_response(cache_key)
    if cached:
        telemetry_tracker.finalize_trace(
            trace_id=request_id,
            tenant_id=current_user.tenant_id,
            user_id=current_user.user_id,
            role=current_user.role,
            query=sanitized_query,
            start_time=start_time,
            cache_hit=True,
        )
        async def cached_stream():
            yield f"data: {json.dumps({'content': cached['response'], 'cached': True, 'trace_id': request_id})}\n\n"
        return StreamingResponse(cached_stream(), media_type="text/event-stream")

    async def event_generator() -> AsyncGenerator[str, None]:
        full_response = ""
        error_encountered = None
        try:
            async with get_checkpointer() as checkpointer:
                app = build_agent_graph(checkpointer=checkpointer)
                
                config: RunnableConfig = {
                    "configurable": {
                        "thread_id": build_backend_thread_id(
                            current_user.tenant_id,
                            current_user.user_id,
                            thread_id,
                        )
                    }
                }

                inputs = {
                    "messages": [{"role": "user", "content": sanitized_query}],
                    "tenant_id": current_user.tenant_id,
                    "user_role": current_user.role,
                    "context": "",
                    "query": sanitized_query,
                    "rewrite_count": 0,
                    "trace_id": request_id,
                }

                inside_think = False
                think_buffer = ""

                async for event in app.astream_events(inputs, config=config, version="v2"):
                    node_name = event.get("metadata", {}).get("langgraph_node")
                    if event.get("event") == "on_chat_model_stream" and node_name == "generate":
                        chunk = event.get("data", {}).get("chunk")
                        if chunk and hasattr(chunk, "content") and chunk.content:
                            token = str(chunk.content)
                            think_buffer += token

                            if "<think>" in think_buffer and not inside_think:
                                inside_think = True

                            if inside_think:
                                if "</think>" in think_buffer:
                                    inside_think = False
                                    post_think = think_buffer.split("</think>", 1)[1]
                                    think_buffer = ""
                                    if post_think.strip():
                                        sanitized_token = SecurityGuardrails.redact_pii(post_think.lstrip())
                                        full_response += sanitized_token
                                        yield f"data: {json.dumps({'content': sanitized_token, 'trace_id': request_id})}\n\n"
                                continue

                            sanitized_token = SecurityGuardrails.redact_pii(token)
                            full_response += sanitized_token
                            yield f"data: {json.dumps({'content': sanitized_token, 'trace_id': request_id})}\n\n"
                            think_buffer = ""

            if full_response:
                await set_cached_response(cache_key, {"response": full_response})
        except Exception as error:
            error_encountered = str(error)
            logger.exception("Chat stream execution failed: %s", error)
            error_msg = "Generation failed. Please retry the request."
            yield f"data: {json.dumps({'content': error_msg, 'trace_id': request_id})}\n\n"
        finally:
            telemetry_tracker.finalize_trace(
                trace_id=request_id,
                tenant_id=current_user.tenant_id,
                user_id=current_user.user_id,
                role=current_user.role,
                query=sanitized_query,
                start_time=start_time,
                cache_hit=False,
                error_message=error_encountered,
            )

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.delete("/chat/thread/{thread_id}")
async def clear_chat_thread(
    thread_id: str,
    current_user: TokenData = Depends(get_current_tenant_user),
) -> dict[str, str]:
    """Delete all checkpointer conversation history and invalidate cache for a specific thread."""
    backend_thread_id = build_backend_thread_id(
        current_user.tenant_id,
        current_user.user_id,
        thread_id,
    )
    await delete_thread_history(backend_thread_id)
    await bump_tenant_cache_version(current_user.tenant_id)
    return {"status": "cleared", "thread_id": thread_id}


@router.delete("/chat/history")
async def clear_all_chat_history(
    current_user: TokenData = Depends(get_current_tenant_user),
) -> dict[str, str]:
    """Clear all checkpointer conversation history across all sessions for the authenticated user."""
    await delete_all_tenant_user_history(current_user.tenant_id, current_user.user_id)
    await bump_tenant_cache_version(current_user.tenant_id)
    return {"status": "cleared", "tenant_id": current_user.tenant_id, "user_id": current_user.user_id}


@router.get("/telemetry/traces")
async def get_tenant_traces(
    limit: int = Query(default=20, ge=1, le=100),
    current_user: TokenData = Depends(get_current_tenant_user)
) -> List[TraceRecord]:
    """Retrieve recent observability traces for the authenticated tenant."""
    return telemetry_tracker.get_traces_for_tenant(
        tenant_id=current_user.tenant_id,
        limit=limit
    )


@router.delete("/telemetry/traces")
async def clear_tenant_telemetry(
    current_user: TokenData = Depends(get_current_tenant_user),
) -> dict[str, Any]:
    """Clear recorded in-memory telemetry traces for the authenticated tenant."""
    purged_count = telemetry_tracker.clear_traces(tenant_id=current_user.tenant_id)
    return {"status": "cleared", "tenant_id": current_user.tenant_id, "purged_traces": purged_count}


@router.get("/telemetry/metrics")
async def get_tenant_metrics(
    current_user: TokenData = Depends(get_current_tenant_user)
) -> Dict[str, Any]:
    """Retrieve real-time telemetry metrics and latency percentiles for the tenant."""
    return telemetry_tracker.get_metrics_summary(
        tenant_id=current_user.tenant_id
    )

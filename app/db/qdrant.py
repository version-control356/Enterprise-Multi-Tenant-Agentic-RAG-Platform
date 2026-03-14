import logging
from typing import List, Optional
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.models import ScoredPoint
from app.config import settings

logger = logging.getLogger(__name__)

COLLECTION_NAME = "tenant_knowledge_base"
VECTOR_SIZE = 384  # FastEmbed BAAI/bge-small-en-v1.5 vector dimension

qdrant_url = settings.QDRANT_URL.strip() if settings.QDRANT_URL else None

if qdrant_url:
    logger.info("Connecting to Qdrant Cloud Cluster via URL")
    qdrant_client = AsyncQdrantClient(
        url=qdrant_url, 
        api_key=settings.QDRANT_API_KEY or None
    )
else:
    logger.info(f"Connecting to Local Qdrant Instance via Host/Port: {settings.QDRANT_HOST}")
    qdrant_client = AsyncQdrantClient(
        host=settings.QDRANT_HOST, 
        port=settings.QDRANT_PORT
    )


async def close_qdrant() -> None:
    """Close the Qdrant client during application shutdown."""
    await qdrant_client.close()


async def check_qdrant_connection() -> bool:
    """Check Qdrant availability for readiness probes."""
    try:
        await qdrant_client.get_collections()
        return True
    except Exception:
        logger.exception("Qdrant readiness check failed.")
        return False


async def init_qdrant() -> None:
    """Initialize Qdrant collection with hybrid (Dense + Sparse BM25) vector indexes."""
    try:
        collections = await qdrant_client.get_collections()
        collection_names = [c.name for c in collections.collections]

        needs_creation = COLLECTION_NAME not in collection_names

        # Check existing collection schema for hybrid sparse vectors
        if not needs_creation:
            try:
                info = await qdrant_client.get_collection(COLLECTION_NAME)
                params = getattr(info.config, "params", None)
                sparse_cfg = getattr(params, "sparse_vectors", None) if params else None
                if not sparse_cfg:
                    raise RuntimeError(
                        f"Qdrant collection '{COLLECTION_NAME}' lacks sparse vector support. "
                        "Run an explicit migration instead of deleting existing data."
                    )
            except Exception as check_err:
                logger.error("Could not verify Qdrant collection config: %s", check_err)
                raise

        if needs_creation:
            await qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config={
                    "dense": models.VectorParams(
                        size=VECTOR_SIZE,
                        distance=models.Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams()
                }
            )
            logger.info(f"✅ Qdrant hybrid collection '{COLLECTION_NAME}' created.")
        else:
            logger.info(f"✅ Qdrant hybrid collection '{COLLECTION_NAME}' already initialized.")

        # Ensure payload indexes for tenant isolation and RBAC
        await qdrant_client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="tenant_id",
            field_schema=models.PayloadSchemaType.KEYWORD
        )
        await qdrant_client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="allowed_roles",
            field_schema=models.PayloadSchemaType.KEYWORD
        )
        await qdrant_client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="filename",
            field_schema=models.PayloadSchemaType.KEYWORD
        )
        await qdrant_client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="doc_id",
            field_schema=models.PayloadSchemaType.KEYWORD
        )
    except Exception as e:
        logger.error(f"❌ Failed to initialize Qdrant: {e}")
        raise e


async def upsert_documents_to_qdrant(points: List[models.PointStruct]) -> None:
    """Upsert chunk embeddings and payload metadata into Qdrant."""
    await qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )


async def search_tenant_knowledge(
    dense_vector: List[float],
    tenant_id: str,
    user_role: str,
    sparse_indices: Optional[List[int]] = None,
    sparse_values: Optional[List[float]] = None,
    limit: int = 5
) -> List[ScoredPoint]:
    """Execute hybrid vector search (Dense + Sparse BM25) with Reciprocal Rank Fusion and RBAC filtering."""
    role_match = (
        models.MatchAny(any=["admin", "analyst", "viewer"])
        if user_role == "admin"
        else models.MatchAny(any=[user_role])
    )
    rbac_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="tenant_id",
                match=models.MatchValue(value=tenant_id)
            ),
            models.FieldCondition(
                key="allowed_roles",
                match=role_match
            )
        ]
    )

    # If sparse vector is provided, execute full Hybrid Search with RRF
    if sparse_indices is not None and sparse_values is not None:
        response = await qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                models.Prefetch(
                    query=dense_vector,
                    using="dense",
                    filter=rbac_filter,
                    limit=limit * 2,
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse_indices,
                        values=sparse_values,
                    ),
                    using="sparse",
                    filter=rbac_filter,
                    limit=limit * 2,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit
        )
    else:
        # Fallback to dense-only named vector query
        response = await qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=dense_vector,
            using="dense",
            query_filter=rbac_filter,
            limit=limit
        )

    return response.points


async def delete_documents_from_qdrant(
    tenant_id: str,
    filename: Optional[str] = None,
    doc_id: Optional[str] = None,
) -> None:
    """Delete document vector points from Qdrant by tenant and filename/doc_id."""
    must_conditions = [
        models.FieldCondition(key="tenant_id", match=models.MatchValue(value=tenant_id))
    ]
    if doc_id:
        must_conditions.append(
            models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))
        )
    elif filename:
        must_conditions.append(
            models.FieldCondition(key="filename", match=models.MatchValue(value=filename))
        )

    await qdrant_client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=models.FilterSelector(
            filter=models.Filter(must=must_conditions)
        )
    )


async def delete_all_tenant_documents_from_qdrant(tenant_id: str) -> None:
    """Delete all vector points for a tenant from Qdrant."""
    await qdrant_client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(key="tenant_id", match=models.MatchValue(value=tenant_id))
                ]
            )
        )
    )
import logging
import sys
import asyncio
if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from passlib.context import CryptContext
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from app.config import settings
from app.migrations import run_migrations

logger = logging.getLogger(__name__)

pool: Optional[AsyncConnectionPool] = None
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


async def init_pool() -> None:
    """Initialize and open the main database pool for app operations."""
    global pool
    if pool is None:
        pool = AsyncConnectionPool(
            conninfo=settings.postgres_dsn,
            name="managed_db_pool",
            min_size=1,
            max_size=20,
            open=False,
            kwargs={"row_factory": dict_row, "prepare_threshold": None}
        )
        await pool.open()


async def close_pool() -> None:
    """Close the database pool cleanly on app shutdown."""
    global pool
    if pool is not None:
        await pool.close()
        pool = None


async def check_database_connection() -> bool:
    """Check PostgreSQL availability for readiness probes."""
    if pool is None:
        return False
    try:
        async with pool.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute("SELECT 1")
                await cursor.fetchone()
        return True
    except Exception:
        logger.exception("PostgreSQL readiness check failed.")
        return False


async def init_db() -> None:
    """Initialize Postgres tables required for LangGraph checkpointing."""
    try:
        async with await psycopg.AsyncConnection.connect(
            settings.postgres_dsn, 
            autocommit=True,
            prepare_threshold=None,
        ) as conn:
            checkpointer = AsyncPostgresSaver(conn)
            await checkpointer.setup()
            await run_migrations(conn)
            if settings.bootstrap_admin_configured:
                await conn.execute(
                    """
                    INSERT INTO tenant_users (tenant_id, username, password_hash, role)
                    VALUES (%s, %s, %s, 'admin')
                    ON CONFLICT (tenant_id, username) DO NOTHING
                    """,
                    (
                        settings.BOOTSTRAP_ADMIN_TENANT_ID,
                        settings.BOOTSTRAP_ADMIN_USERNAME,
                        pwd_context.hash(settings.BOOTSTRAP_ADMIN_PASSWORD),
                    ),
                )
            logger.info("✅ PostgreSQL LangGraph checkpointer tables initialized.")
    except Exception as e:
        logger.error(f"❌ Failed to initialize PostgreSQL checkpointer: {repr(e)}")
        raise e


async def create_tenant_user(
    tenant_id: str,
    username: str,
    password_hash: str,
    role: str,
) -> bool:
    """Create a tenant user and return False when that user already exists."""
    if pool is None:
        raise RuntimeError("Database pool is not initialized.")

    async with pool.connection() as connection:
        async with connection.cursor() as cursor:
            await cursor.execute(
                """
                INSERT INTO tenant_users (tenant_id, username, password_hash, role)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (tenant_id, username) DO NOTHING
                RETURNING username
                """,
                (tenant_id, username, password_hash, role),
            )
            return await cursor.fetchone() is not None


async def is_active_tenant_user(tenant_id: str, username: str, role: str) -> bool:
    """Verify that a token subject still exists, is active, and has its current role."""
    if pool is None:
        raise RuntimeError("Database pool is not initialized.")

    async with pool.connection() as connection:
        async with connection.cursor() as cursor:
            await cursor.execute(
                """
                SELECT 1
                FROM tenant_users
                WHERE tenant_id = %s
                  AND username = %s
                  AND role = %s
                  AND is_active = TRUE
                """,
                (tenant_id, username, role),
            )
            return await cursor.fetchone() is not None


async def get_tenant_document_usage(tenant_id: str) -> tuple[int, int]:
    """Return active document count and total stored bytes for a tenant."""
    if pool is None:
        raise RuntimeError("Database pool is not initialized.")

    async with pool.connection() as connection:
        async with connection.cursor() as cursor:
            await cursor.execute(
                """
                SELECT COUNT(*) AS document_count,
                       COALESCE(SUM(size_bytes), 0) AS storage_bytes
                FROM documents
                WHERE tenant_id = %s AND deleted_at IS NULL
                """,
                (tenant_id,),
            )
            row = await cursor.fetchone()
            return int(row["document_count"]), int(row["storage_bytes"])


async def get_tenant_document_id(tenant_id: str, filename: str) -> Optional[str]:
    """Return the current document ID for a tenant-scoped filename, if present."""
    if pool is None:
        return None
    async with pool.connection() as connection:
        async with connection.cursor() as cursor:
            await cursor.execute(
                """
                SELECT id::text AS id
                FROM documents
                WHERE tenant_id = %s AND filename = %s AND deleted_at IS NULL
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (tenant_id, filename),
            )
            row = await cursor.fetchone()
            return str(row["id"]) if row else None


@asynccontextmanager
async def get_checkpointer() -> AsyncGenerator[AsyncPostgresSaver, None]:
    """Yield an active AsyncPostgresSaver instance using a pooled connection."""
    if pool is None:
        raise RuntimeError("Database pool is not initialized. Call init_pool() first.")

    async with pool.connection() as conn:
        checkpointer = AsyncPostgresSaver(conn)
        yield checkpointer


async def record_document(
    doc_id: str,
    tenant_id: str,
    filename: str,
    size_bytes: int,
    chunks_count: int,
    created_by: str,
    allowed_roles: str = "admin",
    max_documents: Optional[int] = None,
    max_storage_bytes: Optional[int] = None,
) -> None:
    """Insert or update a document tracking record in PostgreSQL with role permissions."""
    if pool is None:
        raise RuntimeError("Database pool is not initialized.")

    async with pool.connection() as connection:
        async with connection.cursor() as cursor:
            if max_documents is not None or max_storage_bytes is not None:
                await cursor.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (tenant_id,))
                await cursor.execute(
                    """
                    SELECT COUNT(*) AS document_count, COALESCE(SUM(size_bytes), 0) AS storage_bytes,
                           COALESCE(MAX(size_bytes) FILTER (WHERE filename = %s), 0) AS replaced_size
                    FROM documents
                    WHERE tenant_id = %s AND deleted_at IS NULL
                    """,
                    (filename, tenant_id),
                )
                usage = await cursor.fetchone()
                document_count = int(usage["document_count"])
                storage_bytes = int(usage["storage_bytes"])
                replaced_size = int(usage["replaced_size"])
                resulting_count = document_count if replaced_size else document_count + 1
                resulting_storage = storage_bytes - replaced_size + size_bytes
                if max_documents is not None and resulting_count > max_documents:
                    raise ValueError("The tenant document quota has been reached.")
                if max_storage_bytes is not None and resulting_storage > max_storage_bytes:
                    raise ValueError("The tenant storage quota has been reached.")
            await cursor.execute(
                """
                DELETE FROM documents WHERE tenant_id = %s AND filename = %s
                """,
                (tenant_id, filename),
            )
            await cursor.execute(
                """
                INSERT INTO documents (id, tenant_id, filename, size_bytes, chunks_count, created_by, allowed_roles)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (doc_id, tenant_id, filename, size_bytes, chunks_count, created_by, allowed_roles),
            )


async def list_tenant_documents(
    tenant_id: str,
    user_role: Optional[str] = None,
) -> list[dict]:
    """Retrieve active documents for a given tenant filtered by role clearance.

    If user_role is 'admin' or None, all tenant documents are returned.
    If user_role is 'analyst' or 'viewer', only documents granting access to that role are returned.
    """
    if pool is None:
        return []

    try:
        async with pool.connection() as connection:
            async with connection.cursor() as cursor:
                if not user_role or user_role.lower() == "admin":
                    await cursor.execute(
                        """
                        SELECT DISTINCT ON (filename) id, tenant_id, filename, size_bytes, chunks_count, created_by, allowed_roles, created_at
                        FROM documents
                        WHERE tenant_id = %s AND deleted_at IS NULL
                        ORDER BY filename, created_at DESC
                        """,
                        (tenant_id,),
                    )
                else:
                    normalized_role = user_role.strip().lower()
                    await cursor.execute(
                        """
                        SELECT DISTINCT ON (filename) id, tenant_id, filename, size_bytes, chunks_count, created_by, allowed_roles, created_at
                        FROM documents
                        WHERE tenant_id = %s 
                          AND deleted_at IS NULL 
                          AND (
                            string_to_array(allowed_roles, ',') @> ARRAY[%s]
                            OR allowed_roles ILIKE %s
                          )
                        ORDER BY filename, created_at DESC
                        """,
                        (tenant_id, normalized_role, f"%{normalized_role}%"),
                    )
                rows = await cursor.fetchall()
                return sorted(rows, key=lambda x: x["created_at"], reverse=True)
    except Exception:
        logger.exception("Failed to list tenant documents for tenant %s.", tenant_id)
        raise


async def delete_thread_history(thread_id: str) -> None:
    """Purge LangGraph checkpoint records and state for a specific thread."""
    if pool is None:
        return

    try:
        async with pool.connection() as connection:
            async with connection.cursor() as cursor:
                delete_queries = (
                    "DELETE FROM checkpoint_writes WHERE thread_id = %s",
                    "DELETE FROM checkpoint_blobs WHERE thread_id = %s",
                    "DELETE FROM checkpoints WHERE thread_id = %s",
                )
                for query in delete_queries:
                    try:
                        await cursor.execute(query, (thread_id,))
                    except Exception as table_err:
                        logger.debug("Could not purge checkpoint table for thread %s: %s", thread_id, table_err)
        logger.info(f"Successfully purged checkpoint history for thread '{thread_id}'")
    except Exception as err:
        logger.warning(f"Failed to delete thread checkpoints for {thread_id}: {err}")


async def delete_all_tenant_user_history(thread_prefix: str) -> None:
    """Purge all LangGraph checkpointer history matching a trusted thread prefix."""
    if pool is None:
        return
    try:
        async with pool.connection() as connection:
            async with connection.cursor() as cursor:
                delete_queries = (
                    "DELETE FROM checkpoint_writes WHERE thread_id LIKE %s",
                    "DELETE FROM checkpoint_blobs WHERE thread_id LIKE %s",
                    "DELETE FROM checkpoints WHERE thread_id LIKE %s",
                )
                for query in delete_queries:
                    try:
                        await cursor.execute(query, (thread_prefix,))
                    except Exception as table_err:
                        logger.debug("Could not purge checkpoint table with prefix %s: %s", thread_prefix, table_err)
        logger.info("Purged all checkpoint history for prefix '%s'", thread_prefix)
    except Exception as err:
        logger.warning("Failed to delete checkpoint history for prefix %s: %s", thread_prefix, err)


async def delete_tenant_document(tenant_id: str, doc_id: str) -> Optional[str]:
    """Delete a document record in PostgreSQL and return the filename."""
    if pool is None:
        return None
    try:
        async with pool.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    """
                    DELETE FROM documents
                    WHERE (id::text = %s OR filename = %s) AND tenant_id = %s
                    RETURNING filename
                    """,
                    (doc_id, doc_id, tenant_id),
                )
                row = await cursor.fetchone()
                return row["filename"] if row else None
    except Exception as err:
        logger.warning(f"Failed to delete document {doc_id}: {err}")
        return None


async def delete_all_tenant_documents(tenant_id: str) -> int:
    """Delete all document records for a tenant in PostgreSQL."""
    if pool is None:
        return 0
    try:
        async with pool.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    """
                    DELETE FROM documents
                    WHERE tenant_id = %s
                    RETURNING id
                    """,
                    (tenant_id,),
                )
                rows = await cursor.fetchall()
                return len(rows)
    except Exception as err:
        logger.warning(f"Failed to delete all documents for tenant {tenant_id}: {err}")
        return 0

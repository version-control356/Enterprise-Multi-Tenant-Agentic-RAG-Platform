import logging

import app.database as database

logger = logging.getLogger(__name__)

async def list_tenant_users(tenant_id: str, limit: int = 100, offset: int = 0) -> list[dict]:
    """Return paginated users for a tenant (username, role, created_at)."""
    if database.pool is None:
        return []
    try:
        async with database.pool.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    """
                    SELECT username, role, created_at
                    FROM tenant_users
                    WHERE tenant_id = %s
                    ORDER BY created_at ASC
                    LIMIT %s OFFSET %s
                    """,
                    (tenant_id, limit, offset),
                )
                rows = await cursor.fetchall()
                return [
                    {
                        "username": row["username"],
                        "role": row["role"],
                        "created_at": row["created_at"].isoformat() if hasattr(row["created_at"], "isoformat") else str(row["created_at"])
                    }
                    for row in rows
                ]
    except Exception:
        logger.exception("Failed to list tenant users for tenant %s.", tenant_id)
        raise

async def delete_tenant_user(tenant_id: str, username: str) -> bool:
    """Delete a specific user from a tenant. Returns True if a row was deleted."""
    if database.pool is None:
        return False
    try:
        async with database.pool.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    """
                    DELETE FROM tenant_users
                    WHERE tenant_id = %s AND username = %s
                    RETURNING username
                    """,
                    (tenant_id, username),
                )
                row = await cursor.fetchone()
                return row is not None
    except Exception as err:
        logger.warning(f"Failed to delete tenant user {username}: {err}")
        return False

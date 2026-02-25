"""Versioned PostgreSQL schema migrations."""

from pathlib import Path

import psycopg


MIGRATIONS_DIR = Path(__file__).with_name("migrations")


async def run_migrations(connection: psycopg.AsyncConnection[object]) -> None:
    """Apply each ordered SQL migration exactly once."""
    await connection.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version TEXT PRIMARY KEY,
            applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )
    for migration_path in sorted(MIGRATIONS_DIR.glob("*.sql")):
        version = migration_path.stem
        cursor = await connection.execute(
            "SELECT 1 FROM schema_migrations WHERE version = %s",
            (version,),
        )
        if await cursor.fetchone():
            continue
        await connection.execute(migration_path.read_text(encoding="utf-8"))
        await connection.execute(
            "INSERT INTO schema_migrations (version) VALUES (%s)",
            (version,),
        )

"""API key registry. Only SHA-256 digests of keys are stored."""

from __future__ import annotations

import hashlib
import sqlite3

from llm_gateway.models import Principal
from llm_gateway.storage.database import SQLiteDatabase


def hash_api_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


class ApiKeyRepository:
    def __init__(self, db: SQLiteDatabase) -> None:
        self._db = db

    async def register(
        self,
        raw_key: str,
        *,
        tenant_id: str,
        tokens_per_window: int | None = None,
        now: float,
    ) -> Principal:
        key_id = hash_api_key(raw_key)

        def op(conn: sqlite3.Connection) -> None:
            conn.execute(
                "INSERT INTO api_keys (key_id, tenant_id, tokens_per_window, active, created_at) "
                "VALUES (?, ?, ?, 1, ?) "
                "ON CONFLICT(key_id) DO UPDATE SET tenant_id = excluded.tenant_id, "
                "tokens_per_window = excluded.tokens_per_window, active = 1",
                (key_id, tenant_id, tokens_per_window, now),
            )

        await self._db.run(op)
        return Principal(key_id=key_id, tenant_id=tenant_id, tokens_per_window=tokens_per_window)

    async def find_active(self, raw_key: str) -> Principal | None:
        key_id = hash_api_key(raw_key)

        def op(conn: sqlite3.Connection) -> Principal | None:
            row = conn.execute(
                "SELECT tenant_id, tokens_per_window FROM api_keys "
                "WHERE key_id = ? AND active = 1",
                (key_id,),
            ).fetchone()
            if row is None:
                return None
            return Principal(key_id=key_id, tenant_id=row[0], tokens_per_window=row[1])

        return await self._db.run(op)

    async def revoke(self, raw_key: str) -> None:
        key_id = hash_api_key(raw_key)
        await self._db.run(
            lambda conn: conn.execute("UPDATE api_keys SET active = 0 WHERE key_id = ?", (key_id,))
        )

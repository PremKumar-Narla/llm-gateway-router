"""Append-only audit trail of every gateway request (one row per request)."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass, field

from llm_gateway.models import RoutingAttempt
from llm_gateway.storage.database import SQLiteDatabase


@dataclass(frozen=True, slots=True)
class AuditRecord:
    request_id: str
    created_at: float
    status: int
    latency_ms: float
    key_id: str | None = None
    tenant_id: str | None = None
    error_code: str | None = None
    provider: str | None = None
    fallback_used: bool = False
    total_tokens: int | None = None
    attempts: tuple[RoutingAttempt, ...] = field(default=())


class AuditRepository:
    def __init__(self, db: SQLiteDatabase) -> None:
        self._db = db

    async def record(self, rec: AuditRecord) -> None:
        attempts_json = json.dumps([asdict(a) for a in rec.attempts])

        def op(conn: sqlite3.Connection) -> None:
            conn.execute(
                "INSERT OR REPLACE INTO request_audit (request_id, key_id, tenant_id, "
                "created_at, status, error_code, provider, fallback_used, attempts_json, "
                "total_tokens, latency_ms) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    rec.request_id,
                    rec.key_id,
                    rec.tenant_id,
                    rec.created_at,
                    rec.status,
                    rec.error_code,
                    rec.provider,
                    int(rec.fallback_used),
                    attempts_json,
                    rec.total_tokens,
                    rec.latency_ms,
                ),
            )

        await self._db.run(op)

    async def get(self, request_id: str) -> dict | None:
        def op(conn: sqlite3.Connection) -> dict | None:
            conn_row = conn.execute(
                "SELECT request_id, key_id, tenant_id, created_at, status, error_code, "
                "provider, fallback_used, attempts_json, total_tokens, latency_ms "
                "FROM request_audit WHERE request_id = ?",
                (request_id,),
            ).fetchone()
            if conn_row is None:
                return None
            cols = (
                "request_id", "key_id", "tenant_id", "created_at", "status", "error_code",
                "provider", "fallback_used", "attempts_json", "total_tokens", "latency_ms",
            )
            row = dict(zip(cols, conn_row))
            row["attempts"] = json.loads(row.pop("attempts_json"))
            row["fallback_used"] = bool(row["fallback_used"])
            return row

        return await self._db.run(op)

"""Persistence for the sliding-window token log.

Each request writes one row: inserted as `reserved` with the *estimated*
token cost before the upstream call, then either updated to `committed`
with the *actual* cost, or deleted if the request never reached a model.

Rows older than the window are evicted. A row left behind by a crashed
process is therefore self-healing: it simply ages out of the window.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from llm_gateway.ratelimit.policy import SlidingWindowPolicy
from llm_gateway.storage.database import SQLiteDatabase, immediate_transaction


@dataclass(frozen=True, slots=True)
class ReservationOutcome:
    granted: bool
    used_tokens: int  # tokens in the window *before* this request
    retry_after_seconds: float = 0.0


class TokenUsageRepository:
    def __init__(self, db: SQLiteDatabase) -> None:
        self._db = db

    async def try_reserve(
        self,
        *,
        key_id: str,
        request_id: str,
        tokens: int,
        now: float,
        policy: SlidingWindowPolicy,
    ) -> ReservationOutcome:
        """Atomically: evict expired rows, sum the window, insert if it fits."""

        def op(conn: sqlite3.Connection) -> ReservationOutcome:
            window_start = policy.window_start(now)
            with immediate_transaction(conn):
                conn.execute(
                    "DELETE FROM token_usage WHERE key_id = ? AND created_at <= ?",
                    (key_id, window_start),
                )
                (used,) = conn.execute(
                    "SELECT COALESCE(SUM(tokens), 0) FROM token_usage WHERE key_id = ?",
                    (key_id,),
                ).fetchone()

                if not policy.admits(used, tokens):
                    entries = conn.execute(
                        "SELECT created_at, tokens FROM token_usage "
                        "WHERE key_id = ? ORDER BY created_at ASC",
                        (key_id,),
                    ).fetchall()
                    return ReservationOutcome(
                        granted=False,
                        used_tokens=used,
                        retry_after_seconds=policy.retry_after(entries, used, tokens, now),
                    )

                conn.execute(
                    "INSERT INTO token_usage (request_id, key_id, created_at, tokens, state) "
                    "VALUES (?, ?, ?, ?, 'reserved')",
                    (request_id, key_id, now, tokens),
                )
                return ReservationOutcome(granted=True, used_tokens=used)

        return await self._db.run(op)

    async def commit(self, request_id: str, actual_tokens: int) -> bool:
        """Replace the estimate with real usage. Returns False if no row existed."""

        def op(conn: sqlite3.Connection) -> bool:
            cur = conn.execute(
                "UPDATE token_usage SET tokens = ?, state = 'committed' "
                "WHERE request_id = ?",
                (max(0, actual_tokens), request_id),
            )
            return cur.rowcount == 1

        return await self._db.run(op)

    async def release(self, request_id: str) -> bool:
        """Drop an uncommitted reservation. Idempotent; never touches committed rows."""

        def op(conn: sqlite3.Connection) -> bool:
            cur = conn.execute(
                "DELETE FROM token_usage WHERE request_id = ? AND state = 'reserved'",
                (request_id,),
            )
            return cur.rowcount == 1

        return await self._db.run(op)

    async def window_usage(self, *, key_id: str, now: float, policy: SlidingWindowPolicy) -> int:
        def op(conn: sqlite3.Connection) -> int:
            (used,) = conn.execute(
                "SELECT COALESCE(SUM(tokens), 0) FROM token_usage "
                "WHERE key_id = ? AND created_at > ?",
                (key_id, policy.window_start(now)),
            ).fetchone()
            return int(used)

        return await self._db.run(op)

    async def evict_expired(self, *, older_than: float) -> int:
        """Global sweep across all keys; returns rows deleted."""

        def op(conn: sqlite3.Connection) -> int:
            cur = conn.execute(
                "DELETE FROM token_usage WHERE created_at <= ?", (older_than,)
            )
            return cur.rowcount

        return await self._db.run(op)

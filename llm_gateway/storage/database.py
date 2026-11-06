"""On-disk SQLite access for an asyncio application.

Design:
* One connection, owned by one dedicated worker thread. All statements run
  on that thread via `run()`, so the event loop never blocks on disk I/O
  and the connection is never shared across threads.
* The worker is a single-thread executor, which gives FIFO ordering: an
  operation submitted after another is guaranteed to execute after it. The
  rate limiter relies on this to make "release after cancelled reserve"
  safe (see `TokenUsageRepository`).
* WAL mode + `BEGIN IMMEDIATE` transactions make read-check-write sequences
  atomic even if several gateway processes share the same database file.
"""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import TypeVar

T = TypeVar("T")

SCHEMA = """
CREATE TABLE IF NOT EXISTS api_keys (
    key_id             TEXT PRIMARY KEY,      -- SHA-256 of the raw key
    tenant_id          TEXT NOT NULL,
    tokens_per_window  INTEGER,               -- NULL = gateway default
    active             INTEGER NOT NULL DEFAULT 1,
    created_at         REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS token_usage (
    request_id  TEXT PRIMARY KEY,
    key_id      TEXT NOT NULL,
    created_at  REAL NOT NULL,                -- unix epoch seconds
    tokens      INTEGER NOT NULL CHECK (tokens >= 0),
    state       TEXT NOT NULL CHECK (state IN ('reserved', 'committed'))
);
CREATE INDEX IF NOT EXISTS ix_token_usage_key_time ON token_usage (key_id, created_at);
CREATE INDEX IF NOT EXISTS ix_token_usage_time ON token_usage (created_at);

CREATE TABLE IF NOT EXISTS request_audit (
    request_id     TEXT PRIMARY KEY,
    key_id         TEXT,
    tenant_id      TEXT,
    created_at     REAL NOT NULL,
    status         INTEGER NOT NULL,
    error_code     TEXT,
    provider       TEXT,
    fallback_used  INTEGER NOT NULL DEFAULT 0,
    attempts_json  TEXT NOT NULL DEFAULT '[]',
    total_tokens   INTEGER,
    latency_ms     REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_request_audit_tenant_time ON request_audit (tenant_id, created_at);
"""


class SQLiteDatabase:
    def __init__(self, path: str | Path, *, busy_timeout_ms: int = 5_000) -> None:
        self._path = Path(path)
        self._busy_timeout_ms = busy_timeout_ms
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sqlite")
        self._conn: sqlite3.Connection | None = None
        self._closed = False

    @property
    def path(self) -> Path:
        return self._path

    async def connect(self) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self._open)

    async def close(self) -> None:
        """Idempotent: safe to call more than once."""
        if self._closed:
            return
        self._closed = True
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self._close)
        self._executor.shutdown(wait=True)

    async def run(self, operation: Callable[[sqlite3.Connection], T]) -> T:
        """Execute `operation(conn)` on the database thread."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._invoke, operation)

    # -- worker-thread side -------------------------------------------------

    def _open(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # isolation_level=None: we manage transactions explicitly.
        conn = sqlite3.connect(
            self._path,
            isolation_level=None,
            timeout=self._busy_timeout_ms / 1000,
        )
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute(f"PRAGMA busy_timeout={int(self._busy_timeout_ms)}")
        conn.executescript(SCHEMA)
        self._conn = conn

    def _close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _invoke(self, operation: Callable[[sqlite3.Connection], T]) -> T:
        if self._conn is None:
            raise RuntimeError("database is not connected")
        return operation(self._conn)


@contextmanager
def immediate_transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Take the write lock up front so check-then-write cannot interleave."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")

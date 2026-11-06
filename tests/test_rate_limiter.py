import asyncio
import sqlite3

from llm_gateway.config import RateLimitConfig
from llm_gateway.errors import RateLimitExceededError, RequestTooLargeError
from llm_gateway.models import Principal
from llm_gateway.ratelimit.limiter import SlidingWindowRateLimiter
from llm_gateway.storage.database import SQLiteDatabase
from llm_gateway.storage.usage_repository import TokenUsageRepository
from tests.helpers import TempDbTestCase

ACME = Principal(key_id="k-acme", tenant_id="acme")
GLOBEX = Principal(key_id="k-globex", tenant_id="globex")


class RateLimiterTest(TempDbTestCase):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self.db = await self._open_db()
        self.limiter = self._limiter(self.db)

    async def _open_db(self) -> SQLiteDatabase:
        db = SQLiteDatabase(self.db_path)
        await db.connect()
        self.addAsyncCleanup(db.close)
        return db

    def _limiter(self, db: SQLiteDatabase, limit: int = 1_000) -> SlidingWindowRateLimiter:
        return SlidingWindowRateLimiter(
            TokenUsageRepository(db),
            RateLimitConfig(tokens_per_window=limit, window_seconds=60),
            self.clock,
        )

    async def test_admits_until_budget_then_rejects_with_retry_after(self) -> None:
        await self.limiter.acquire(ACME, "r1", 600)
        self.clock.advance(10)
        await self.limiter.acquire(ACME, "r2", 400)  # exactly at the limit
        with self.assertRaises(RateLimitExceededError) as ctx:
            await self.limiter.acquire(ACME, "r3", 1)
        # r1 (t=0) must expire to free space -> 60s after it, now is t=10
        self.assertAlmostEqual(ctx.exception.retry_after_seconds, 50.0)

    async def test_entries_evicted_exactly_at_window_boundary(self) -> None:
        await self.limiter.acquire(ACME, "r1", 1_000)
        self.clock.advance(59.999)
        with self.assertRaises(RateLimitExceededError):
            await self.limiter.acquire(ACME, "r2", 1)
        self.clock.advance(0.001)  # now == t0 + window -> r1 expired
        await self.limiter.acquire(ACME, "r2", 1_000)
        self.assertEqual(await self.limiter.usage(ACME), 1_000)

    async def test_sliding_not_fixed_window(self) -> None:
        await self.limiter.acquire(ACME, "r1", 500)
        self.clock.advance(30)
        await self.limiter.acquire(ACME, "r2", 500)
        self.clock.advance(31)  # r1 expired, r2 still live
        self.assertEqual(await self.limiter.usage(ACME), 500)
        await self.limiter.acquire(ACME, "r3", 500)
        with self.assertRaises(RateLimitExceededError):
            await self.limiter.acquire(ACME, "r4", 1)

    async def test_keys_are_isolated(self) -> None:
        await self.limiter.acquire(ACME, "r1", 1_000)
        await self.limiter.acquire(GLOBEX, "r2", 1_000)  # unaffected by acme
        self.assertEqual(await self.limiter.usage(GLOBEX), 1_000)

    async def test_per_key_override(self) -> None:
        small = Principal(key_id="k-small", tenant_id="t", tokens_per_window=10)
        await self.limiter.acquire(small, "r1", 10)
        with self.assertRaises(RateLimitExceededError):
            await self.limiter.acquire(small, "r2", 1)

    async def test_request_larger_than_budget_is_not_retryable(self) -> None:
        with self.assertRaises(RequestTooLargeError):
            await self.limiter.acquire(ACME, "r1", 1_001)
        self.assertEqual(await self.limiter.usage(ACME), 0)

    async def test_commit_reconciles_estimate_to_actual(self) -> None:
        await self.limiter.acquire(ACME, "r1", 800)
        await self.limiter.commit("r1", 120)
        self.assertEqual(await self.limiter.usage(ACME), 120)
        await self.limiter.acquire(ACME, "r2", 880)  # freed budget is usable

    async def test_release_is_idempotent_and_never_drops_committed(self) -> None:
        await self.limiter.acquire(ACME, "r1", 300)
        await self.limiter.release("r1")
        await self.limiter.release("r1")
        self.assertEqual(await self.limiter.usage(ACME), 0)
        await self.limiter.acquire(ACME, "r2", 300)
        await self.limiter.commit("r2", 300)
        await self.limiter.release("r2")  # committed rows are kept
        self.assertEqual(await self.limiter.usage(ACME), 300)

    async def test_concurrent_acquires_never_exceed_budget(self) -> None:
        async def attempt(i: int) -> bool:
            try:
                await self.limiter.acquire(ACME, f"r{i}", 70)
                return True
            except RateLimitExceededError:
                return False

        results = await asyncio.gather(*(attempt(i) for i in range(50)))
        self.assertEqual(sum(results), 1_000 // 70)  # 14 admitted
        self.assertLessEqual(await self.limiter.usage(ACME), 1_000)

    async def test_concurrent_acquires_across_two_connections(self) -> None:
        """Two limiter instances on separate connections (like two processes)."""
        other = self._limiter(await self._open_db())

        async def attempt(limiter: SlidingWindowRateLimiter, i: int) -> bool:
            try:
                await limiter.acquire(ACME, f"r{i}", 100)
                return True
            except RateLimitExceededError:
                return False

        results = await asyncio.gather(
            *(attempt(self.limiter if i % 2 else other, i) for i in range(30))
        )
        self.assertEqual(sum(results), 10)

    async def test_state_persists_on_disk_across_restart(self) -> None:
        await self.limiter.acquire(ACME, "r1", 900)
        await self.db.close()
        reopened = self._limiter(await self._open_db())
        with self.assertRaises(RateLimitExceededError):
            await reopened.acquire(ACME, "r2", 200)

    async def test_sweep_evicts_idle_keys(self) -> None:
        await self.limiter.acquire(ACME, "r1", 10)
        await self.limiter.acquire(GLOBEX, "r2", 10)
        self.clock.advance(61)
        self.assertEqual(await self.limiter.sweep(), 2)
        with sqlite3.connect(self.db_path) as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM token_usage").fetchone()[0], 0)

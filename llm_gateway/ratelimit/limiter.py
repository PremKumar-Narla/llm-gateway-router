"""Token-aware sliding-window rate limiter backed by on-disk SQLite.

Lifecycle of one request's budget:

    acquire(estimate)  -> row 'reserved'   (counts immediately, so concurrent
                                           requests cannot jointly overshoot)
    commit(actual)     -> row 'committed'  (estimate replaced by real usage)
    release()          -> row deleted      (request never produced output)

Concurrency: the check-and-insert in `acquire` is one `BEGIN IMMEDIATE`
transaction, so it is atomic across coroutines *and* across processes that
share the database file.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass

from llm_gateway.clock import Clock, SystemClock
from llm_gateway.config import RateLimitConfig
from llm_gateway.errors import RateLimitExceededError, RequestTooLargeError
from llm_gateway.models import Principal
from llm_gateway.ratelimit.policy import SlidingWindowPolicy
from llm_gateway.storage.usage_repository import TokenUsageRepository

logger = logging.getLogger("llm_gateway.ratelimit")


@dataclass(frozen=True, slots=True)
class Reservation:
    request_id: str
    key_id: str
    estimated_tokens: int
    limit: int
    remaining_after: int


class SlidingWindowRateLimiter:
    def __init__(
        self,
        repository: TokenUsageRepository,
        config: RateLimitConfig,
        clock: Clock | None = None,
    ) -> None:
        self._repo = repository
        self._config = config
        self._clock = clock or SystemClock()

    def policy_for(self, principal: Principal) -> SlidingWindowPolicy:
        limit = principal.tokens_per_window or self._config.tokens_per_window
        return SlidingWindowPolicy(limit=limit, window_seconds=self._config.window_seconds)

    async def acquire(self, principal: Principal, request_id: str, tokens: int) -> Reservation:
        """Reserve `tokens` or raise. Never leaves a row behind when it raises."""
        if tokens <= 0:
            raise ValueError("tokens must be positive")
        policy = self.policy_for(principal)
        if not policy.fits_at_all(tokens):
            raise RequestTooLargeError(
                f"Request needs ~{tokens} tokens but the budget is "
                f"{policy.limit} tokens per {policy.window_seconds:g}s."
            )

        outcome = await self._repo.try_reserve(
            key_id=principal.key_id,
            request_id=request_id,
            tokens=tokens,
            now=self._clock.now(),
            policy=policy,
        )
        if not outcome.granted:
            raise RateLimitExceededError(retry_after_seconds=outcome.retry_after_seconds)

        return Reservation(
            request_id=request_id,
            key_id=principal.key_id,
            estimated_tokens=tokens,
            limit=policy.limit,
            remaining_after=policy.limit - outcome.used_tokens - tokens,
        )

    async def commit(self, request_id: str, actual_tokens: int) -> None:
        if not await self._repo.commit(request_id, actual_tokens):
            # The row aged out (request outlived the whole window). The usage
            # is real, so it is simply not charged; log for visibility.
            logger.warning("commit found no reservation request_id=%s", request_id)

    async def release(self, request_id: str) -> None:
        await self._repo.release(request_id)

    async def usage(self, principal: Principal) -> int:
        return await self._repo.window_usage(
            key_id=principal.key_id, now=self._clock.now(), policy=self.policy_for(principal)
        )

    async def sweep(self) -> int:
        """Evict expired rows for every key (acquire only evicts its own key)."""
        return await self._repo.evict_expired(
            older_than=self._clock.now() - self._config.window_seconds
        )


class ExpiredUsageSweeper:
    """Background task that periodically calls `limiter.sweep()`.

    Keys that stop sending traffic never hit the per-key eviction in
    `acquire`, so without this their rows would accumulate forever.
    """

    def __init__(self, limiter: SlidingWindowRateLimiter, interval_seconds: float) -> None:
        self._limiter = limiter
        self._interval = interval_seconds
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run(), name="usage-sweeper")

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task
        self._task = None

    async def _run(self) -> None:
        while True:
            await asyncio.sleep(self._interval)
            try:
                removed = await self._limiter.sweep()
                if removed:
                    logger.debug("sweeper evicted %d expired usage rows", removed)
            except Exception:  # keep the sweeper alive on transient DB errors
                logger.exception("usage sweep failed")

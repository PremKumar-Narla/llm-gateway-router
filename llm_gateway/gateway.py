"""LLMGateway: the facade that composes auth, rate limiting, routing,
error sanitization and auditing into one request pipeline.

    authenticate -> validate -> reserve tokens -> route (with failover)
                 -> commit actual usage -> respond
    any failure  -> release reservation (if any) -> sanitized error payload
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from llm_gateway.clock import Clock, SystemClock
from llm_gateway.config import GatewayConfig
from llm_gateway.errors import (
    BadGatewayError,
    ErrorSanitizer,
    GatewayError,
    GatewayTimeoutError,
    ProviderRejectedError,
    ProvidersUnavailableError,
    UnauthenticatedError,
)
from llm_gateway.models import CompletionRequest, CompletionResponse, Principal
from llm_gateway.providers.base import ModelProvider
from llm_gateway.providers.errors import FailureKind
from llm_gateway.ratelimit.limiter import ExpiredUsageSweeper, Reservation, SlidingWindowRateLimiter
from llm_gateway.routing.router import FallbackRouter, RoutingFailedError
from llm_gateway.storage.api_key_repository import ApiKeyRepository
from llm_gateway.storage.audit_repository import AuditRecord, AuditRepository
from llm_gateway.storage.database import SQLiteDatabase
from llm_gateway.storage.usage_repository import TokenUsageRepository
from llm_gateway.tokenizer import HeuristicTokenEstimator, TokenEstimator

logger = logging.getLogger("llm_gateway")

_MIN_KEY_LEN, _MAX_KEY_LEN = 16, 256


@dataclass(frozen=True, slots=True)
class GatewayResponse:
    status: int
    body: dict[str, Any]
    headers: dict[str, str] = field(default_factory=dict)


class LLMGateway:
    def __init__(
        self,
        *,
        config: GatewayConfig,
        db: SQLiteDatabase,
        api_keys: ApiKeyRepository,
        limiter: SlidingWindowRateLimiter,
        router: FallbackRouter,
        audit: AuditRepository,
        estimator: TokenEstimator,
        clock: Clock,
        sanitizer: ErrorSanitizer | None = None,
    ) -> None:
        self._config = config
        self._db = db
        self._api_keys = api_keys
        self._limiter = limiter
        self._router = router
        self._audit = audit
        self._estimator = estimator
        self._clock = clock
        self._sanitizer = sanitizer or ErrorSanitizer()
        self._sweeper = ExpiredUsageSweeper(limiter, config.rate_limit.sweep_interval_seconds)

    # -- construction / lifecycle ------------------------------------------

    @classmethod
    async def create(
        cls,
        config: GatewayConfig,
        providers: Sequence[ModelProvider],
        *,
        clock: Clock | None = None,
        estimator: TokenEstimator | None = None,
    ) -> "LLMGateway":
        """Wire up all components against the on-disk SQLite database."""
        clock = clock or SystemClock()
        db = SQLiteDatabase(config.database_path)
        await db.connect()
        return cls(
            config=config,
            db=db,
            api_keys=ApiKeyRepository(db),
            limiter=SlidingWindowRateLimiter(TokenUsageRepository(db), config.rate_limit, clock),
            router=FallbackRouter(providers, config.routing),
            audit=AuditRepository(db),
            estimator=estimator or HeuristicTokenEstimator(),
            clock=clock,
        )

    async def start(self) -> None:
        self._sweeper.start()

    async def close(self) -> None:
        await self._sweeper.stop()
        await self._db.close()

    async def __aenter__(self) -> "LLMGateway":
        await self.start()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    @property
    def limiter(self) -> SlidingWindowRateLimiter:
        return self._limiter

    @property
    def audit(self) -> AuditRepository:
        return self._audit

    async def register_api_key(
        self, raw_key: str, *, tenant_id: str, tokens_per_window: int | None = None
    ) -> Principal:
        return await self._api_keys.register(
            raw_key, tenant_id=tenant_id, tokens_per_window=tokens_per_window, now=self._clock.now()
        )

    # -- request pipeline ----------------------------------------------------

    async def handle_completion(self, authorization: str | None, payload: Any) -> GatewayResponse:
        """Entry point for transports. Always returns a response; never raises
        except for `CancelledError` (client went away / shutdown)."""
        request_id = uuid.uuid4().hex
        started = time.perf_counter()
        principal: Principal | None = None
        try:
            principal = await self.authenticate(authorization)
            request = CompletionRequest.from_payload(payload, self._config.request_limits)
            response, reservation = await self._execute(principal, request, request_id)
        except Exception as exc:
            error = self._sanitizer.to_response(exc, request_id)
            cause = exc.__cause__
            await self._audit_quietly(
                AuditRecord(
                    request_id=request_id,
                    created_at=self._clock.now(),
                    status=error.status,
                    latency_ms=_ms_since(started),
                    key_id=principal.key_id if principal else None,
                    tenant_id=principal.tenant_id if principal else None,
                    error_code=error.body["error"]["code"],
                    attempts=cause.attempts if isinstance(cause, RoutingFailedError) else (),
                )
            )
            return GatewayResponse(
                error.status, error.body, {**error.headers, "X-Request-Id": request_id}
            )

        provider = response.attempts[-1].provider
        await self._audit_quietly(
            AuditRecord(
                request_id=request_id,
                created_at=self._clock.now(),
                status=200,
                latency_ms=_ms_since(started),
                key_id=principal.key_id,
                tenant_id=principal.tenant_id,
                provider=provider,
                fallback_used=response.fallback_used,
                total_tokens=response.completion.total_tokens,
                attempts=response.attempts,
            )
        )
        remaining = (
            reservation.remaining_after
            + reservation.estimated_tokens
            - response.completion.total_tokens
        )
        return GatewayResponse(
            200,
            response.to_public_dict(),
            {
                "X-Request-Id": request_id,
                "X-RateLimit-Limit": str(reservation.limit),
                "X-RateLimit-Remaining": str(max(0, remaining)),
            },
        )

    async def authenticate(self, authorization: str | None) -> Principal:
        if not authorization or not authorization.startswith("Bearer "):
            raise UnauthenticatedError()
        raw_key = authorization[len("Bearer "):].strip()
        if not _MIN_KEY_LEN <= len(raw_key) <= _MAX_KEY_LEN:
            raise UnauthenticatedError()
        principal = await self._api_keys.find_active(raw_key)
        if principal is None:
            raise UnauthenticatedError()
        return principal

    async def complete(
        self, principal: Principal, request: CompletionRequest, request_id: str | None = None
    ) -> CompletionResponse:
        """Programmatic entry point (raises GatewayError on failure)."""
        response, _ = await self._execute(principal, request, request_id or uuid.uuid4().hex)
        return response

    async def _execute(
        self, principal: Principal, request: CompletionRequest, request_id: str
    ) -> tuple[CompletionResponse, Reservation]:
        estimated = self._estimator.count(request.prompt) + request.max_tokens

        # 1. Reserve budget. The request_id is generated *before* this call so
        #    that if we are cancelled mid-reserve (the INSERT may still commit
        #    on the DB thread) we can release by id. The DB executor is FIFO,
        #    so the release is guaranteed to run after the insert.
        try:
            reservation = await self._limiter.acquire(principal, request_id, estimated)
        except GatewayError:
            raise  # rejected; nothing was written
        except BaseException:
            await self._release_quietly(request_id)
            raise

        # 2. Route with failover. Any failure returns the reserved budget.
        try:
            routed = await self._router.route(request)
        except RoutingFailedError as exc:
            await self._release_quietly(request_id)
            raise self._map_routing_failure(exc) from exc
        except BaseException:
            await self._release_quietly(request_id)
            raise

        # 3. Reconcile estimate -> actual. Shielded: the tokens were consumed
        #    even if the client disconnects right now.
        await asyncio.shield(self._limiter.commit(request_id, routed.completion.total_tokens))

        return (
            CompletionResponse(
                request_id=request_id,
                completion=routed.completion,
                fallback_used=routed.fallback_used,
                attempts=routed.attempts,
            ),
            reservation,
        )

    @staticmethod
    def _map_routing_failure(exc: RoutingFailedError) -> GatewayError:
        """Translate internal failure kinds to client-safe gateway errors."""
        if exc.final_kind is FailureKind.RATE_LIMITED:
            return ProvidersUnavailableError()
        if exc.final_kind is FailureKind.TIMEOUT:
            return GatewayTimeoutError()
        if exc.final_kind is FailureKind.BAD_REQUEST:
            return ProviderRejectedError()
        return BadGatewayError()

    async def _release_quietly(self, request_id: str) -> None:
        try:
            # Shield so a second cancellation cannot abort the cleanup itself.
            await asyncio.shield(self._limiter.release(request_id))
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("failed to release reservation request_id=%s", request_id)

    async def _audit_quietly(self, record: AuditRecord) -> None:
        try:
            await self._audit.record(record)
        except Exception:
            logger.exception("audit write failed request_id=%s", record.request_id)


def _ms_since(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)


__all__ = ["GatewayResponse", "LLMGateway"]
